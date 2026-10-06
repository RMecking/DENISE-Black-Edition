"""Offline A2.5 evidence authentication and deterministic report consolidation.

Only saved H arrays are reduced. This module cannot load an execution driver,
build a binary, invoke MPI, or evaluate J/JT. No missing-evidence fallback exists.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path, PurePosixPath

import numpy as np

from .campaign import encoded
from .image_products import EXECUTION_SHA, load_products, safe_output, write_json
from .image_conditioning_report import process_products
from .model_io import sha256_file
from .normal_operator import array_hash, basis, metrics, transpose
from .normal_operator_analysis import GROUPS, aperture, compare, compact, cosine, deep_cosine, reconstruct

PACKAGE = Path(__file__).resolve().parents[2] / 'applications/marmousi2_a25'
CASE = PACKAGE.parent / 'marmousi2_a25a/case.json'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def portable(value):
    """Persisted records must have logical roots, never workstation aliases."""
    text = encoded(value).decode()
    if re.search(r'[A-Za-z]:[/\\]|/mnt/[a-z]/|file:/+|\\\\[^\\]', text):
        raise ValueError('machine-specific path in publication record')
    return value


def contained(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or ':' in str(relative):
        raise ValueError('relative evidence path required')
    target = (Path(root) / str(relative)).resolve()
    if not target.is_relative_to(Path(root).resolve()):
        raise ValueError('evidence path aliases outside logical root')
    return target


def authenticate(root, relative, identity, checked):
    path = contained(root, relative)
    # Errors deliberately use only logical relative paths; no machine aliases.
    if not path.is_file():
        raise FileNotFoundError('missing frozen artifact: ' + relative)
    if sha256_file(path) != identity['sha256'] or ('bytes' in identity and path.stat().st_size != identity['bytes']):
        raise ValueError('frozen artifact identity mismatch: ' + relative)
    checked[path] = {'bytes': path.stat().st_size, 'sha256': identity['sha256']}
    return path


def saved_array(path, expected, shape):
    value = np.load(path, allow_pickle=False)
    if value.dtype.str != '<f8' or value.shape != tuple(shape) or not value.flags.c_contiguous or not np.isfinite(value).all():
        raise ValueError('saved H endian/shape/layout/finite contract')
    if array_hash(value) != expected:
        raise ValueError('saved H array identity mismatch')
    value.flags.writeable = False
    return value


def historical_h15_inventory(protected):
    normalized = {path.replace('\\', '/'): value for path, value in protected.items()}
    # A runtime continuation also has an H15/processes namespace. Identify the
    # actual campaign root by its campaign receipt, not by a substring match.
    campaign_paths = [p for p in normalized if p.endswith('/H15/campaign.json')]
    if len(campaign_paths) != 1:
        raise ValueError('unique historical H15 campaign root required')
    prefix = campaign_paths[0][:-len('campaign.json')]
    return {'H15/' + path[len(prefix):]: identity for path, identity in normalized.items() if path.startswith(prefix)}


def validate(roots, manifest=None):
    manifest = portable(manifest or load(PACKAGE / 'evidence.json'))
    checked = {}
    for root, refs in manifest['references'].items():
        for relative, identity in refs.items():
            authenticate(roots[root], relative, identity, checked)
    a_receipt = load(roots['A25A'] / 'receipt.json')
    if a_receipt['content_id'] != manifest['a25a_content_id']:
        raise ValueError('A2.5A historical content identity')
    for item in a_receipt['outputs']:
        authenticate(roots['A25A'], item['path'], item, checked)
    seal = load(roots['A25B'] / 'serial_h5_closure/final_sealed_receipt.json')
    for relative, identity in seal['files'].items():
        authenticate(roots['A25B'], relative, identity, checked)
    # Historical H15 metadata must be tied to the sealed preparation, not just
    # today's numeric fields. Convert paths in memory only, without rewriting it.
    historical = load(roots['A25B'] / 'serial_h5_closure/preparation.json')['protected_files']
    inventory = historical_h15_inventory(historical)
    if inventory != load(PACKAGE / 'h15_inventory.json'):
        raise ValueError('historical H15 inventory lineage')
    for relative, identity in inventory.items():
        authenticate(roots['A25B'], relative, identity, checked)
    identities = {}
    for label, relative in [('H15', 'blocked_handover/H15_numerical_identity.json'),
                            ('H5', 'serial_h5_closure/comparison/H5_numerical_identity.json')]:
        record = load(roots['A25B'] / relative)
        content = record['content']
        if record['numerical_content_id'] != manifest['numerical_content_ids'][label] or digest(content) != record['numerical_content_id']:
            raise ValueError('frozen numerical content identity: ' + label)
        if content['core_sha'] != EXECUTION_SHA or set(content['operators']) != {label}:
            raise ValueError('historical Core/operator contract')
        operator = content['operators'][label]
        if sum(len(p['shots']) for p in operator.values()) != manifest['counts'][label]:
            raise ValueError('campaign completeness: ' + label)
        for p in operator.values():
            if [s['shot'] for s in p['shots']] != list(range(1, 101)):
                raise ValueError('ordered complete physical shot identities')
        for path, sha in content['core_source_sha256'].items():
            authenticate(roots['CORE'], path, {'sha256': sha}, checked)
        if sha256_file(roots['A25B'] / 'build/liba25b.so') != content['library_sha256']:
            raise ValueError('unchanged historical binary identity')
        identities[label] = content
    return manifest, identities, checked


def characterize(root, label, content, models, checked):
    """Reconstruct ascending native sums, exact transpose and frozen measurements."""
    summary = load(root / label / 'summary.json')
    operator = content['operators'][label]
    if set(summary) != set(operator):
        raise ValueError('summary/probe completeness')
    all_fields = {}; apertures = {}; compact_summary = {}
    shape = (2, *models['vs'].shape)
    for name, product in operator.items():
        print('A25: saved-field reconstruction ' + label + '/' + name, file=sys.stderr, flush=True)
        items = []
        for expected in product['shots']:
            folder = root / label / 'shots' / f"{expected['shot']:03d}" / name
            receipt_path = folder / 'receipt.json'
            receipt = load(receipt_path)
            # Jz attestation is authenticated historical metadata, not a new Jz run.
            for key, value in expected.items():
                if receipt.get(key) != value:
                    raise ValueError('per-shot numerical receipt identity: ' + name)
            if not receipt['finite'] or not receipt['gate']['pass']:
                raise ValueError('historically accepted numerical gate')
            authenticate(folder, 'receipt.json', {'sha256': sha256_file(receipt_path)}, checked)
            p = authenticate(folder, 'h_native.npy', {'sha256': receipt['h_native_file_sha256']}, checked)
            native = saved_array(p, expected['h_native_array_sha256'], shape)
            jz_path = folder / 'jz.npy'
            if jz_path.is_file():
                jz = np.load(jz_path, allow_pickle=False)
                if not np.isfinite(jz).all() or array_hash(jz) != expected['jz_array_sha256']:
                    raise ValueError('retained Jz array identity')
                authenticate(folder, 'jz.npy', {'sha256': sha256_file(jz_path)}, checked)
            items.append((expected['shot'], native))
        sums = reconstruct(items)
        fields = {}; entries = {}
        for group in GROUPS:
            base = root / label / 'sums' / name / group
            hashes = product['ascending_groups'][group]
            entry = summary[name]['groups'][group]
            native = sums[group]; log = transpose(native, models)
            for key, array, file_key in [('h_native', native, 'native_file_sha256'), ('h_log', log, 'log_file_sha256')]:
                p = authenticate(base, key + '.npy', {'sha256': entry[file_key]}, checked)
                saved = saved_array(p, hashes[key], shape)
                if array_hash(array) != hashes[key] or not np.array_equal(saved, array):
                    raise ValueError('deterministic ascending reconstruction/transpose: ' + name)
            spec = entry['metrics']['probe']
            _, _, _, declared = basis(spec, models, 20)
            if declared != spec:
                raise ValueError('exact original unit-basis probe mapping')
            measured = metrics(log, spec, models)
            if measured != entry['metrics'] or measured != load(base / 'metrics.json'):
                raise ValueError('frozen PSF measurement reproduction: ' + name)
            authenticate(base, 'metrics.json', {'sha256': sha256_file(base / 'metrics.json')}, checked)
            fields[group] = log
            entries[group] = compact({'groups': {'full': {'metrics': measured}},
                                      'sampled_diagonal': measured['sampled_diagonal'],
                                      'summed_Jz_energy': sum(s['j_energy'] for s in summary[name]['per_shot'] if s['shot'] in
                                                            ([i for i in range(1, 101) if group in ('full', 'odd' if i % 2 else 'even', f'block{(i-1)//20+1}')]))})
        all_fields[name] = fields
        apertures[name] = aperture(fields, summary[name], models)
        compact_summary[name] = {'groups': entries, 'reuse': summary[name]['reuse'],
                                 'sampled_diagonal': summary[name]['sampled_diagonal'],
                                 'summed_Jz_energy': summary[name]['summed_Jz_energy']}
    return summary, all_fields, apertures, compact_summary


def render(summary, destination):
    """Metrics precede rendering; HTML is explicitly DISPLAY_ONLY."""
    portable(summary)
    write_json(destination / 'summary.json', summary)
    rows = []
    for name, entry in summary['frequency_comparison'].items():
        for operator in ('H15', 'H5'):
            values = entry[operator]
            rows.append('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in
                                        (name, operator, values['target_fraction'], values['r90_m'], values['full_cross_talk_norm'])) + '</tr>')
    text = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Marmousi-II A2.5</title>'
            '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto}td,th{padding:8px;border:1px solid #aaa}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>'
            '<h1>Marmousi-II A2.5: frozen evidence interpretation</h1><p>DISPLAY_ONLY report; no numerical execution.</p>'
            '<p><a href="A25A/report/index.html">A2.5A image diagnostics and interface-stripe negative control</a></p>'
            '<ul>' + ''.join('<li>' + html.escape(c) + '</li>' for c in summary['claims_and_limits']) + '</ul>'
            '<h2>Different normal operators, mixed localization</h2><table><tr><th>Probe</th><th>Operator</th><th>Target fraction</th><th>R90 (m)</th><th>Cross-talk norm ratio</th></tr>'
            + ''.join(rows) + '</table><p>Exact full precision metrics and aperture diagnostics: <a href="summary.json">summary.json</a>.</p></html>')
    with (destination / 'index.html').open('xb') as stream:
        stream.write(text.encode('utf-8'))


def recheck(checked):
    for path, identity in checked.items():
        if path.stat().st_size != identity['bytes'] or sha256_file(path) != identity['sha256']:
            raise ValueError('frozen input changed during publication')
    return {'unchanged': True, 'files_checked': len(checked)}


def reproduce(evidence, a25a_evidence, core, runs, models, output):
    roots = {k: Path(v).resolve() for k, v in {'A25B': evidence, 'A25A': a25a_evidence, 'CORE': core,
                                             'RUNS': runs, 'MODELS': models}.items()}
    destination = safe_output(output, tuple(roots.values()) + (PACKAGE.parent.parent,))
    print('A25: authenticating frozen external receipts/products', file=sys.stderr, flush=True)
    manifest, identities, checked = validate(roots)
    products = load_products(CASE, roots_override={'runs': roots['RUNS'], 'models': roots['MODELS']})
    print('A25: raw A2 campaign/model/input identities valid', file=sys.stderr, flush=True)
    results = {}; fields = {}; apertures = {}; compact_summaries = {}
    for label in ('H15', 'H5'):
        results[label], fields[label], apertures[label], compact_summaries[label] = characterize(
            roots['A25B'], label, identities[label], products.models, checked)
    comparisons = {}
    for name in results['H5']:
        comparisons[name] = compare(results['H15'][name], results['H5'][name])
        a = fields['H15'][name]['full']; b = fields['H5'][name]['full']
        comparisons[name]['full_response_cosine'] = cosine(a, b)
        comparisons[name]['deep_response_cosine'] = deep_cosine(a, b, products.models,
                                                               results['H15'][name]['groups']['full']['metrics']['row_depth_m'])
    if comparisons != load(PACKAGE / 'frequency_comparison.json'):
        raise ValueError('frozen H15/H5 comparison changed')
    old_apertures = load(roots['A25B'] / 'serial_h5_closure/comparison/apertures.json')
    for label in ('H15', 'H5'):
        for name, expected in old_apertures[label[1:]].items():
            if apertures[label][name] != expected:
                raise ValueError('frozen aperture comparison changed')
    destination.mkdir(parents=True, exist_ok=False)
    a_receipt = process_products(products, destination / 'A25A',
                                 progress=lambda name: print('A25: image diagnostics ' + name, file=sys.stderr, flush=True))
    old_a = load(roots['A25A'] / 'receipt.json')
    array_count = 0
    for item in old_a['outputs']:
        if item['path'].endswith('.npy'):
            if sha256_file(destination / 'A25A' / item['path']) != item['sha256']:
                raise ValueError('A2.5A accepted derived-array reproduction mismatch')
            array_count += 1
    summary = {'schema': 1, 'class': 'DERIVED_DIAGNOSTIC', 'accepted_content_ids': manifest['numerical_content_ids'],
               'a25a_historical_content_id': manifest['a25a_content_id'], 'a25a_current_derivation_id': a_receipt['content_id'],
               'normal_operator': compact_summaries, 'apertures': apertures, 'frequency_comparison': comparisons,
               'claims_and_limits': load(PACKAGE / 'claims.json')['claims_and_limits'],
               'a25a_accepted_arrays_bitwise_reproduced': array_count,
               'Jz_verification_scope': manifest['jz_evidence'], 'solver_executions': 0, 'builds': 0, 'J_JT_HVP_executions': 0}
    render(summary, destination)
    audit = recheck(checked)
    products.verify_unchanged()
    outputs = [{'path': p.relative_to(destination).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
               for p in sorted(destination.rglob('*')) if p.is_file()]
    source_hashes = {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    receipt = {'schema': 1, 'accepted_content_ids': manifest['numerical_content_ids'],
               'a25a_historical_content_id': manifest['a25a_content_id'], 'input_audit': audit,
               'outputs': outputs, 'sources': source_hashes, 'numpy_version': np.__version__,
               'solver_executions': 0, 'builds': 0, 'J_JT_HVP_executions': 0,
               'a25a_accepted_arrays_bitwise_reproduced': array_count,
               'content_id': digest({'summary': summary, 'sources': source_hashes, 'outputs': outputs})}
    write_json(destination / 'receipt.json', portable(receipt))
    return {'content_id': receipt['content_id'], 'input_audit': audit, 'solver_executions': 0, 'builds': 0}
