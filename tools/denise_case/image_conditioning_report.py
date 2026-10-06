"""Reproducible A2.5A parallel branches, conventional depth sections and QC."""
from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import sys
from pathlib import Path

import numpy as np

from .campaign import CAVEATS, encoded
from .image_products import (BASE_SHA, PUBLICATION_SHA, EXECUTION_SHA, BENCHMARK_SHA, RAW_HASHES, MODEL_HASHES,
                             load_products, safe_output, write_array, write_json)
from .image_conditioning import (parameter_transform, derivatives, bandpass, depth_gain,
                                 display_scale, signed_compress, PRIMARY_CORNERS, SECONDARY_CORNERS,
                                 FFT_CONVENTION, TAPER_CONVENTION)
from .image_diagnostics import (infer_water, domains, depth_metrics, shot_diagnostics,
                                subset_sums, stats, correlation, lateral_shift_control)
from .image_reference import define_reference, evaluate_reference
from .model_io import sha256_file
from .png import heatmap, write_png

DEPTH_BINS = ('1000_1500', '1500_2000', '2000_2500', '2500_3000', '3000_3280')
FILTER_VARIANTS = (
    ('primary', PRIMARY_CORNERS, 128, 10), ('secondary', SECONDARY_CORNERS, 128, 10),
    ('padding256', PRIMARY_CORNERS, 256, 10), ('no_taper', PRIMARY_CORNERS, 128, 0))


def processing_contract():
    """Frozen settings selected before evaluating this campaign's images."""
    return {'schema_version': 1, 'application_base': BASE_SHA, 'publication_commit': PUBLICATION_SHA,
            'historical_execution_core': EXECUTION_SHA, 'benchmark_revision': BENCHMARK_SHA,
            'product_classes': ['AUTHORITATIVE_RAW', 'DERIVED_NUMERICAL', 'DISPLAY_ONLY'],
            'operation_order': 'raw -> parallel diagnostics/transform/derivatives/filter/gain -> display; reference never feeds processing',
            'display_percentiles': [98, 99, 99.5], 'percentile_method': 'linear', 'compression_ratios': [3, 10, 30],
            'primary_display': 'full-domain P99', 'gain': {'z0_m': 650, 'powers': [0, .5, 1], 'caps': [3, 10]},
            'bandpass': {'primary': PRIMARY_CORNERS, 'secondary': SECONDARY_CORNERS, 'units': 'cycles/metre',
                         'padding': [128, 256], 'taper_cells': [10, 0], 'fft': FFT_CONVENTION, 'taper': TAPER_CONVENTION},
            'transform': 'FP64 (2*rho*Vp)*gLambda; (2*rho*Vs)*(gMu-2*gLambda); log views multiply by original Vp/Vs',
            'surrogates': {'realizations': 1999, 'seed': 250501, 'generator': 'PCG64',
                           'applied_to': 'odd/even aggregate pair, raw depth regions',
                           'interpretation': 'investigative cyclic shifts only; not independent-shot/null-geology inference'},
            'classification': 'conservative evidence triage; no new numerical acceptance thresholds',
            'solver_executions': 0, 'builds': 0}


def _svg(png, title, grid, q, label):
    """Deterministic physical axes around a nearest-neighbor conventional raster."""
    ny, nx = grid['shape_y_x']; dh = grid['dh_m']
    data = base64.b64encode(png.read_bytes()).decode()
    escape = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="480" viewBox="0 0 1100 480">',
             '<rect width="1100" height="480" fill="white"/>',
             f'<text x="80" y="25" font-family="sans-serif" font-size="15">{escape(title)}</text>',
             f'<image x="80" y="45" width="960" height="348" preserveAspectRatio="none" style="image-rendering:pixelated" href="data:image/png;base64,{data}"/>',
             '<rect x="80" y="45" width="960" height="348" fill="none" stroke="black"/>']
    for f in np.linspace(0, 1, 6):
        x = 80 + 960 * f; physical = (dh / 2 + nx * dh * f) / 1000
        parts.append(f'<text x="{x:.3f}" y="413" text-anchor="middle" font-size="12">{physical:.3g}</text>')
    for f in np.linspace(0, 1, 5):
        y = 45 + 348 * f; physical = (dh / 2 + ny * dh * f) / 1000
        parts.append(f'<text x="70" y="{y+4:.3f}" text-anchor="end" font-size="12">{physical:.3g}</text>')
    parts += ['<text x="550" y="435" text-anchor="middle" font-size="13">Horizontal x (km)</text>',
              '<text x="15" y="215" transform="rotate(-90 15 215)" text-anchor="middle" font-size="13">Depth (km), positive down</text>',
              f'<text x="80" y="460" font-size="12">Blue: -{q:.6g}; neutral: 0; red: +{q:.6g}. {escape(label)}. Invalid edges white.</text>', '</svg>']
    return ''.join(parts).encode()


def process_products(products, output, *, investigate=True, progress=None):
    """Products must originate from load_products; testing may use tiny authenticated fixtures."""
    output = safe_output(output, products.protected)
    contract = processing_contract()
    source_dir = Path(__file__).parent
    source_hashes = {p.name: sha256_file(p) for p in sorted(source_dir.glob('*.py'))}
    # platform.platform() may spawn cmd.exe /c ver on Windows. Read native state
    # without any subprocess, keeping the solver-free execution contract strict.
    native_platform = {'system': sys.platform, 'pointer_bits': np.dtype(np.intp).itemsize * 8}
    if sys.platform == 'win32':
        native_platform['windows_version'] = list(sys.getwindowsversion())
    elif hasattr(os, 'uname'):
        state = os.uname()
        native_platform.update(release=state.release, version=state.version, machine=state.machine)
    environment = {'python': '.'.join(map(str, sys.version_info[:3])), 'numpy': np.__version__, 'platform': native_platform,
                   'precision': 'FP64, input models original FP32', 'renderer': 'stdlib PNG/zlib + fixed SVG sans-serif axes; nearest-neighbor',
                   'png_zlib': __import__('zlib').ZLIB_VERSION, 'fft': FFT_CONVENTION}
    content_id = hashlib.sha256(encoded({'inputs': products.inventory, 'contract': contract,
                                       'sources': source_hashes, 'environment': environment,
                                       'investigate': investigate})).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('derived', 'qc', 'display', 'report'):
        (output / name).mkdir()
    write_json(output / 'input_inventory.json', products.inventory)
    write_json(output / 'processing_contract.json', contract)
    entries = []; panels = []; scales = {}

    def numerical(name, array, operation, parents, parameters=None, units='inherited raw sensitivity units', valid=None):
        path = output / 'derived' / (name + '.npy')
        write_array(path, array)
        entry = {'path': path.relative_to(output).as_posix(), 'sha256': sha256_file(path),
                 'class': 'DERIVED_NUMERICAL', 'operation': operation, 'parents': parents,
                 'parameters': parameters or {}, 'units': units, 'shape': list(array.shape), 'dtype': array.dtype.str,
                 'provenance_content_id': content_id}
        if valid is not None:
            entry['validity'] = valid
        entries.append(entry)
        return name

    def display(name, array, title, *, percentile=99, mask=None, compression=None, valid=None, common_q=None):
        finite = np.isfinite(array) if valid is None else valid
        if not np.isfinite(array[finite]).all():
            raise ValueError('nonfinite valid display samples')
        clean = np.where(finite, array, 0.)  # DISPLAY ONLY invalid-edge representation
        scale_domain = finite if mask is None else finite & mask
        if not scale_domain.any():
            return
        scale = display_scale(clean, percentile, scale_domain)
        scale['saturated_count'] = int(np.count_nonzero(np.abs(clean[finite]) > scale['q']))
        scale['total_count'] = int(finite.sum())
        scale['saturated_fraction'] = scale['saturated_count'] / scale['total_count']
        if common_q is not None:
            scale['q'] = common_q; scale['limits'] = [-common_q, common_q]
            scale['saturated_fraction'] = float(np.mean(np.abs(clean[finite]) > common_q))
            scale['saturated_count'] = int(np.count_nonzero(np.abs(clean[finite]) > common_q))
            scale['shared_reference_limit'] = True
        q = scale['q']
        if compression is not None:
            # An all-zero scale domain with nonzero samples elsewhere is undefined, no amplitude floor.
            if q == 0 and np.any(clean != 0):
                return
            mapped = signed_compress(clean, q, compression); limit = 1. if q else 0.
        else:
            mapped = clean; limit = q
        if limit == 0:
            if np.any(clean != 0):
                scales[name] = {'class': 'DISPLAY_ONLY', 'undefined': 'zero scale domain with nonzero samples elsewhere; no floor'}
                return
            rgb = np.full((*array.shape, 3), 245, dtype=np.uint8)  # neutral zero, not palette minimum
        else:
            rgb = heatmap(mapped, vmin=-limit, vmax=limit, diverging=True)
        rgb[~finite] = 255
        png = output / 'display' / (name + '.png')
        write_png(png, rgb)
        svg = output / 'display' / (name + '.svg')
        label = 'DISPLAY ONLY; signed asinh compression' if compression is not None else 'DISPLAY ONLY; symmetric color saturation'
        with svg.open('xb') as stream:
            stream.write(_svg(png, title, products.grid, limit, label))
        scales[name] = {**scale, 'compression_ratio': compression, 'source_title': title,
                        'scale_domain': name.rsplit('_scale_', 1)[-1] if mask is not None else 'all valid samples',
                        'shared_limit_source': title if common_q is not None else None,
                        'finite_count': int(finite.sum()), 'invalid_count': int((~finite).sum()), 'class': 'DISPLAY_ONLY'}
        for path in (png, svg):
            entries.append({'path': path.relative_to(output).as_posix(), 'sha256': sha256_file(path), 'class': 'DISPLAY_ONLY',
                            'operation': 'raster color mapping/physical axes', 'parameters': scales[name], 'provenance_content_id': content_id})
        panels.append({'path': svg.relative_to(output / 'report').as_posix() if svg.is_relative_to(output / 'report') else '../display/' + svg.name,
                       'title': title})

    vs = products.models['vs']; dh = products.grid['dh_m']
    geometry = infer_water(vs, dh); ds = domains(vs, dh)
    write_json(output / 'qc/geometry.json', {k: v.tolist() for k, v in geometry.items()})
    # Structural definitions depend ONLY on authenticated reference models. Persist BEFORE image processing.
    reference_spec = None; reference_arrays = None
    if products.reference:
        reference_spec, reference_arrays = define_reference(products.reference, products.models, dh)
        write_json(output / 'qc/reference_definition.json', reference_spec)
        for name, a in reference_arrays.items():
            numerical('reference_' + name, a, 'evaluation-only reference definition', ['reference models'],
                      {'reference_spec_sha256': sha256_file(output / 'qc/reference_definition.json')}, 'model/reference units')
            if name.startswith(('true_', 'contrast_', 'edge_')):
                display('reference_' + name, a, 'EVALUATION ONLY: ' + name, valid=np.isfinite(a))
    # Image-independent filter controls use the same full grid and fixed variants.
    impulse = np.zeros(vs.shape, dtype=np.float64)
    impulse[vs.shape[0] // 2, vs.shape[1] // 2] = 1.
    edge = np.zeros_like(impulse); edge[:, vs.shape[1] // 2:] = 1.
    operator_controls = {}
    for control_name, control in (('impulse', impulse), ('edge_step', edge)):
        numerical('control_' + control_name, control, 'image-independent synthetic filter input', [],
                  {'definition': 'center unit impulse' if control_name == 'impulse' else 'right half-plane unit step'}, 'dimensionless')
        operator_controls[control_name] = {}
        for variant, corners, padding, taper in FILTER_VARIANTS:
            response = bandpass(control, dh, corners, padding, taper)
            name = 'control_' + control_name + '_' + variant
            numerical(name, response, 'synthetic filter response, not campaign image', ['control_' + control_name],
                      {'corners_cycles_m': corners, 'padding': padding, 'taper_cells': taper, 'fft': FFT_CONVENTION}, 'dimensionless')
            display(name, response, 'CONTROL: ' + control_name + ' / ' + variant)
            operator_controls[control_name][variant] = stats(response)
    write_json(output / 'qc/operator_controls.json', operator_controls)
    images = dict(products.raw)
    transformed = parameter_transform(products.raw['lambda'], products.raw['mu'], **products.models)
    for key, a in transformed.items():
        numerical(key, a, 'fixed-density cotangent transformation', ['raw_lambda', 'raw_mu', 'smooth2 models'],
                  {'order': contract['transform'], 'water_log_vs': 'frozen-zero extension; log(0) never evaluated'},
                  'transformed raw sensitivities; not reflectivity or velocity update')
    images.update(transformed)
    baselines = {}; coherence = {}; structures = {}; filter_qc = {}; classifications = {}
    z = (np.arange(vs.shape[0]) + 1) * dh
    water = geometry['water']
    for key, a in images.items():
        if progress:
            progress(key)
        baselines[key] = depth_metrics(a, vs, dh)
        write_json(output / 'qc' / (key + '_depth.json'), baselines[key])
        # Display-only logarithmic profile. Zeros are omitted, never floored.
        rms = np.array([r['rms'] for r in baselines[key]['rows']]); nonzero = rms > 0
        coords = []
        if nonzero.any():
            logs = np.log10(rms[nonzero]); lo, hi = float(logs.min()), float(logs.max())
            width = hi - lo if hi > lo else 1.
            coords = [f'{80 + 800*(value-lo)/width:.4f},{45 + 348*row/len(rms):.4f}'
                      for row,value in zip(np.flatnonzero(nonzero),logs)]
        else:
            lo = hi = None
        profile = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="470"><rect width="1000" height="470" fill="white"/>'
                   f'<text x="80" y="25">{key}: un-gained row RMS versus depth; DISPLAY ONLY log10 axis</text>'
                   f'<polyline points="{" ".join(coords)}" fill="none" stroke="black"/>'
                   f'<text x="80" y="425">Horizontal log10 RMS range: {lo} to {hi}; zero rows omitted, no floor.</text>'
                   f'<text x="80" y="450">Vertical depth: 20 to {len(rms)*dh:g} m, positive down. Exact values in ../qc/{key}_depth.json</text></svg>')
        profile_path = output / 'display' / (key + '_row_rms.svg')
        with profile_path.open('xb') as stream:
            stream.write(profile.encode())
        entries.append({'path': profile_path.relative_to(output).as_posix(), 'sha256': sha256_file(profile_path),
                        'class': 'DISPLAY_ONLY', 'operation': 'logarithmic display of un-gained raw row RMS; zeros omitted',
                        'parents': ['qc/' + key + '_depth.json'], 'provenance_content_id': content_id})
        panels.append({'path': '../display/' + profile_path.name, 'title': key + ': un-gained row RMS profile'})
        if reference_spec is not None:
            structures[key] = evaluate_reference(a, reference_spec, reference_arrays)
        for p in (98, 99, 99.5, 100):
            display(f'{key}_p{str(p).replace(".", "_")}', a,
                    f'{key}: ' + ('full-linear unmodified reference' if p == 100 else f'P{p} full-domain color scale'), percentile=p)
        q = display_scale(a)['q']
        for r in (3, 10, 30):
            display(f'{key}_asinh{r}', a, f'{key}: signed compression s=q/{r}', compression=r)
        for domain in ('solid', 'deep'):
            display(f'{key}_scale_{domain}', a, f'{key}: explicitly region-scaled {domain} P99', mask=ds['vertical'][domain])
        interface_excluded = ds['vertical']['solid'] & ~ds['vertical']['guard650']
        display(f'{key}_scale_interface_excluded', a, f'{key}: interface-excluded P99 scale, no mute', mask=interface_excluded)
        d = derivatives(a, dh, water)
        for op in ('dx', 'dz', 'laplacian'):
            numerical(key + '_' + op + '_valid', d['valid'][op], 'derivative validity bitmap', [key])
            numerical(key + '_' + op + '_interface', d['crosses_interface'][op], 'cross-interface stencil bitmap', [key, 'original Vs'])
            numerical(key + '_' + op, d[op], op, [key], {'DH_m': dh, 'invalid_edges': 'NaN, declared bitmap'},
                      key + ' sensitivity units/' + ('m^2' if op == 'laplacian' else 'metre'), valid='derived/' + key + '_' + op + '_valid.npy')
            display(key + '_' + op, d[op], f'{key}: {op}; invalid stencil edges white', valid=d['valid'][op])
            if reference_spec is not None:
                structures[key + '_' + op] = evaluate_reference(d[op], reference_spec, reference_arrays)
        stripe = np.zeros_like(a)
        for col, row in enumerate(geometry['first_solid']):
            stripe[row, col] = a[row, col]
        filter_qc[key] = {}
        primary = None
        for variant, corners, padding, taper in FILTER_VARIANTS:
            filtered = bandpass(a, dh, corners, padding, taper)
            leak = bandpass(stripe, dh, corners, padding, taper)
            numerical(key + '_bp_' + variant, filtered, 'spatial band-pass', [key],
                      {'corners_cycles_m': corners, 'padding': padding, 'taper_cells': taper, 'fft': FFT_CONVENTION})
            numerical(key + '_stripe_bp_' + variant, leak, 'isolated first-solid-row control, not an accepted image', [key, 'original Vs'],
                      {'corners_cycles_m': corners, 'padding': padding, 'taper_cells': taper})
            if variant == 'primary':
                primary = filtered
            filter_qc[key][variant] = {'output_stats': stats(filtered),
                'stripe_leakage': {bin_name: {'stripe': stats(leak[ds['vertical'][bin_name]]),
                                            'filtered_image': stats(filtered[ds['vertical'][bin_name]]),
                                            'correlation': correlation(filtered, leak, ds['vertical'][bin_name])} for bin_name in DEPTH_BINS},
                'primary_difference_norm': stats(filtered - primary)['norm']}
            display(key + '_bp_' + variant, filtered, f'{key}: spatial band-pass {variant}; not recovered bandwidth')
            display(key + '_stripe_bp_' + variant, leak, f'{key}: first-solid-row-only leakage CONTROL', common_q=display_scale(filtered)['q'])
            if reference_spec is not None:
                structures[key + '_bp_' + variant] = evaluate_reference(filtered, reference_spec, reference_arrays)
        for p in (0, .5, 1):
            for cap in (3, 10):
                name = key + f'_gain_p{str(p).replace(".", "_")}_cap{cap}'
                gained = depth_gain(a, z, 650, p, cap)
                numerical(name, gained, 'diagnostic depth gain, not illumination compensation', [key], {'z0_m': 650, 'p': p, 'cap': cap})
                display(name, gained, f'{key}: diagnostic depth gain p={p}, cap={cap}; SAME raw P99 limit', common_q=q)
                if reference_spec is not None:
                    structures[name] = evaluate_reference(gained, reference_spec, reference_arrays)
        shot_images = products.shots.get(key)
        if shot_images is None:
            shot_images = {i: parameter_transform(products.shots['lambda'][i], products.shots['mu'][i], **products.models)[key] for i in range(1, 101)}
        subsets = subset_sums(shot_images)
        for name, a_subset in subsets.items():
            numerical(key + '_subset_' + name, a_subset, 'ascending physical-shot unweighted FP64 subset sum', [key, name],
                      {'members': list(range(1, 101, 2)) if name == 'odd' else list(range(2, 101, 2)) if name == 'even' else list(range((int(name[-2:])-1)*20+1,int(name[-2:])*20+1))})
        coherence[key] = shot_diagnostics(shot_images, vs, dh)
        if key in products.raw and investigate:
            coherence[key]['shift_controls'] = {bin_name: lateral_shift_control(subsets['odd'], subsets['even'],
                ds['lateral']['interior'] & ds['vertical'][bin_name], 1999, 250501) for bin_name in DEPTH_BINS}
        write_json(output / 'qc' / (key + '_shots.json'), coherence[key])
        classifications[key] = {}
        for bin_name in DEPTH_BINS:
            region = coherence[key]['regions']['interior'][bin_name]
            raw_region = baselines[key]['regions']['interior'][bin_name]
            block_values = [v['centered'] for v in region['block_pairs'].values() if v['centered'] is not None]
            classifications[key][bin_name] = {
                'classification': 'NO DEMONSTRATED USEFUL DEEP INFORMATION' if raw_region['max_abs'] == 0 else 'INDETERMINATE / CONFOUNDED',
                'reason': 'Nonzero reproducible amplitudes may exist, but correlated aperture, direct/interface response and geological-control specificity do not justify a confirmatory reflector claim.',
                'odd_even': region['odd_even'], 'block_centered_min': min(block_values) if block_values else None,
                'block_centered_max': max(block_values) if block_values else None,
                'raw_metrics': raw_region, 'not_geological_absence': True,
                'same_raw_sample_display_evidence': {'raw_full_p99_q': q,
                    'regional_p99_over_full_p99': raw_region['abs_p99'] / q if q and raw_region['abs_p99'] is not None else None,
                    'monotone_display_only': True},
                'filter_stripe_control': filter_qc[key]['primary']['stripe_leakage'][bin_name],
                'shift_control': {k:v for k,v in coherence[key].get('shift_controls', {}).get(bin_name, {}).items() if k != 'control_centered'},
                'reference_window': structures.get(key, {}).get('windows', {}).get('interior_' + bin_name, {}),
                'evidence_scope': 'saved sensitivities; no Hessian/illumination identification'}
    write_json(output / 'qc/filter_controls.json', filter_qc)
    write_json(output / 'qc/structural_comparison.json', structures)
    write_json(output / 'display/scales.json', scales)
    summary = {'content_id': content_id, 'product_classes': contract['product_classes'],
               'scientific_caveats': list(CAVEATS) + [
                   'Transformed raw sensitivities are not reflectivity or velocity updates.',
                   'Diagnostic depth gain is not illumination compensation; filtered images do not recover missing bandwidth.',
                   'Physical log(Vs) is undefined in water; g_lnVs uses frozen-zero extension, no floor.',
                   'Odd/even subsets share aperture; coherence is not independent geological proof.',
                   'Spatial derivative/filter outputs need not preserve restricted raw-water zeros.'],
               'baseline': {k: v['ratios'] for k, v in baselines.items()},
               'classifications': classifications, 'structural_comparison': structures,
               'interpretation': 'Conservative depth/channel triage; negative/ambiguous findings retained. No parameter was tuned against geology.',
               'A25B': 'Separate accepted H15/H5 evidence is characterized by the A2.5 publication tools; image conditioning does not estimate a Hessian diagonal or execute J/JT.',
               'solver_executions': 0, 'binary_builds': 0}
    write_json(output / 'report/summary.json', summary)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Marmousi-II A2.5A</title><style>body{font:16px system-ui;max-width:1150px;margin:2em auto}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:1em 0}</style><h1>Marmousi-II A2.5A: RTM sensitivity diagnostics</h1>'
    page += '<p>Authoritative raw / deterministic derived numerical / display-only products remain separate. No solver execution, no illumination correction, no scientific acceptance.</p>'
    page += '<ul>' + ''.join('<li>' + html.escape(c) + '</li>' for c in summary['scientific_caveats']) + '</ul>'
    page += '<p>Full quantitative findings: <a href="summary.json">summary</a>; all raw-depth, shot and filter-control records: ../qc/. Each panel has its recorded scale; gain panels share their source raw P99 limit.</p>'
    page += ''.join('<details' + (' open' if '_p99.' in p['path'] else '') + '><summary>' + html.escape(p['title']) + '</summary><img src="' + html.escape(p['path']) + '" alt="' + html.escape(p['title']) + '"></details>' for p in panels)
    page += '<h2>Quantitative findings and interpretation limits</h2><pre>' + html.escape(encoded(summary).decode()) + '</pre></html>'
    with (output / 'report/index.html').open('xb') as stream:
        stream.write(page.encode())
    unchanged = products.verify_unchanged()
    # Bind all outputs, not just representative figures. Each derived entry inherits full content provenance.
    all_outputs = [{'path': p.relative_to(output).as_posix(), 'sha256': sha256_file(p), 'bytes': p.stat().st_size}
                   for p in sorted(output.rglob('*')) if p.is_file()]
    # Diagnostic JSON reductions are numerical products too, not just .npy
    # arrays. Explicitly classify every data/display artifact; input inventory
    # and processing contract are provenance records rather than a fourth class.
    named = {entry['path'] for entry in entries}
    for item in all_outputs:
        path = item['path']
        if path in ('input_inventory.json', 'processing_contract.json'):
            item['role'] = 'PROVENANCE_RECORD'
            continue
        product_class = 'DISPLAY_ONLY' if path.startswith('display/') or path.endswith('.html') else 'DERIVED_NUMERICAL'
        item['class'] = product_class
        item['provenance_content_id'] = content_id
        if path not in named:
            entries.append({**item, 'operation': 'display/provenance rendering' if product_class == 'DISPLAY_ONLY' else 'fixed diagnostic/evaluation JSON reduction: ' + Path(path).name,
                'parents': ['input_inventory.json', 'processing_contract.json'],
                'parameters': {'operation_order': contract['operation_order'], 'settings': 'processing_contract.json; explicit domains/definitions in QC records'}})
    receipt = {'content_id': content_id, 'identities': {'application_base': BASE_SHA, 'publication_commit': PUBLICATION_SHA,
               'execution_core': products.inventory['execution_core'], 'benchmark': products.inventory['benchmark_revision']},
               'source_sha256': source_hashes, 'environment': environment, 'outputs': all_outputs,
               'products': entries, 'immutable_inputs': unchanged, 'solver_executions': 0, 'builds': 0,
               'classification_is_not_acceptance': True}
    write_json(output / 'receipt.json', receipt)
    return receipt


def run_conditioning(config, output, *, roots_override=None):
    cfg = json.loads(Path(config).read_text(encoding='utf-8'))
    if (cfg['execution_core'] != EXECUTION_SHA or cfg['benchmark_revision'] != BENCHMARK_SHA
            or cfg['aggregate_hashes'] != RAW_HASHES
            or {k: v['sha256'] for k, v in cfg['models'].items()} != MODEL_HASHES
            or cfg['grid'] != {'shape_y_x': [174, 500], 'dh_m': 20, 'image_dtype': '<f8', 'image_order': 'row-major [depth_y,x]'}
            or cfg['shots'] != list(range(1, 101))):
        raise ValueError('A2.5A frozen canonical input contract mismatch')
    return process_products(load_products(config, roots_override=roots_override), output)
