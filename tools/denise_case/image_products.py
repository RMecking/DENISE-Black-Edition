"""Authenticated immutable historical images; no live solver/core gate."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .campaign import discover_campaign, encoded, unchanged, verify_aggregate
from .model_io import sha256_file

BASE_SHA = '54c69297a6c1d5fef94c396b49451ebaadcec2ab'
PUBLICATION_SHA = '6c348d490862277d6c2c5d017bf2e4029ce54169'
EXECUTION_SHA = 'f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d'
BENCHMARK_SHA = '931131e8dc53649320a5620befe751ad32d2f7bd'
RAW_HASHES = {'lambda': '8e979872d9ec18f0fe7df4f9088a5722d707a32e6bd1c7af54153821dea8448a',
              'mu': 'f311970f856ff1a3239f056a74d4984b05249efab01e4a2f7ec4b84b3fbbee46'}
MODEL_HASHES = {'vp': 'c8330c87b83a0b89a6e727ce09cfac2b9f460ef5f61ff245705f029e7632ec1a',
                'vs': 'ce67dfcc61c43656d220e81e78f0f9d626706ecb68d5ee051da71ba92bdab3f0',
                'rho': '71ce144cc8d4817804cef2015711252800524ea6bd0cf09f56f7f707fdb929da'}


def immutable(values):
    """Bytes-backed data cannot be made writable by toggling ndarray flags."""
    a = np.ascontiguousarray(values)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def authenticated_array(path, expected_hash, shape, *, dtype='<f8', order='row-major [depth_y,x]'):
    path = Path(path)
    shape = tuple(shape)
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('positive 2D shape required')
    allowed = ('<f8', 'row-major [depth_y,x]') if dtype == '<f8' else ('<f4', 'x-major, y-fastest')
    if (dtype, order) != allowed or dtype not in ('<f8', '<f4'):
        raise ValueError('unsupported endian/layout')
    if path.stat().st_size != np.prod(shape) * np.dtype(dtype).itemsize:
        raise ValueError('image/model byte size mismatch')
    if sha256_file(path) != expected_hash:
        raise ValueError('input hash mismatch')
    payload = path.read_bytes()
    import hashlib
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError('input changed while loading')
    a = np.frombuffer(payload, dtype=dtype)
    a = a.reshape(shape) if dtype == '<f8' else a.reshape(shape[::-1]).T
    if not np.isfinite(a).all():
        raise ValueError('nonfinite authoritative input')
    return a


@dataclass(frozen=True)
class ImageProducts:
    raw: dict
    shots: dict
    models: dict
    reference: dict
    grid: dict
    inventory: dict
    protected: tuple
    identities: dict

    def verify_unchanged(self):
        for path, identity in self.identities.items():
            p = Path(path)
            if p.stat().st_size != identity['bytes'] or sha256_file(p) != identity['sha256']:
                raise ValueError('authoritative evidence changed during postprocessing')
        return {'checked_identities': len(self.identities), 'unchanged': True}


def load_products(config_path, *, roots_override=None):
    """Explicit config authenticates historical execution; never require_core()."""
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    if cfg['shots'] != list(range(1, 101)):
        raise ValueError('exact ordered physical shot IDs 1..100 required')
    shape = tuple(cfg['grid']['shape_y_x']); dh = cfg['grid']['dh_m']
    if not np.isfinite(dh) or dh <= 0 or cfg['grid'].get('image_dtype') != '<f8' or cfg['grid'].get('image_order') != 'row-major [depth_y,x]':
        raise ValueError('invalid declared image grid/endian/layout')
    roots = {k: (config_path.parent / v).resolve() for k, v in cfg['roots'].items()}
    if roots_override is not None:
        if set(roots_override) != set(roots):
            raise ValueError('exact logical roots required')
        roots = {k: Path(v).resolve() for k, v in roots_override.items()}
    runs = roots['runs']; aggregate = runs / cfg['aggregate_receipt']
    summary = runs / cfg['campaign_receipt']
    records = discover_campaign(runs, cfg['shots'], shape=shape,
                                authorized_core_sha=cfg['execution_core'], campaign_receipt=summary)
    old, outputs = verify_aggregate(aggregate, records, shape)
    identities = {}; inv = {'product_classes': {'raw': 'AUTHORITATIVE_RAW', 'models': 'AUTHORITATIVE_RAW'},
                            'execution_core': cfg['execution_core'], 'benchmark_revision': cfg['benchmark_revision'],
                            'binary_sha256': records[0]['meta']['binary_sha256'], 'grid': cfg['grid'], 'inputs': []}

    def record(path, logical):
        identity = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
        identities[str(path.resolve())] = identity
        inv['inputs'].append({'path': logical, **identity})
        return identity

    record(config_path, 'configuration/case.json')
    record(summary, 'runs/' + cfg['campaign_receipt'])
    record(aggregate, 'runs/' + cfg['aggregate_receipt'])
    raw = {}; shots = {'lambda': {}, 'mu': {}}
    for key in ('lambda', 'mu'):
        path = aggregate.parent / outputs[key]['path']
        if outputs[key]['sha256'] != cfg['aggregate_hashes'][key]:
            raise ValueError('aggregate differs from authorized identity')
        raw[key] = authenticated_array(path, cfg['aggregate_hashes'][key], shape)
        record(path, 'aggregate/' + outputs[key]['path'])
    for r in records:
        for entry in (r['metadata'], r['provenance']):
            record(runs / entry['path'], 'runs/' + entry['path'])
        provenance = json.loads((runs / r['provenance']['path']).read_text())
        run = (runs / r['metadata']['path']).parent
        for name, expected in provenance.get('inputs', {}).items():
            p = (run / name).resolve()
            if not p.is_relative_to(run) or sha256_file(p) != expected['sha256']:
                raise ValueError('prepared input path/hash mismatch')
            record(p, f'runs/{run.name}/{name}')
        # Authenticate geometry against the actual bound input, not checkout defaults.
        parameter_file = run / 'denise.inp'
        if parameter_file.is_file():
            values = {}
            for line in parameter_file.read_text().splitlines():
                if '=' in line and not line.lstrip().startswith('#'):
                    k, v = line.split('=', 1); values[k.strip()] = v.strip()
            for key, required in (('NX', shape[1]), ('NY', shape[0]), ('DH', dh)):
                if key in values and float(values[key]) != required:
                    raise ValueError('declared grid differs from retained execution input')
        for key in shots:
            entry = r['raw'][key]; path = runs / entry['path']
            shots[key][r['shot']] = authenticated_array(path, entry['sha256'], shape)
            record(path, 'runs/' + entry['path'])
    models = {}; reference = {}
    for group, destination in (('models', models), ('reference', reference)):
        for key, entry in cfg.get(group, {}).items():
            p = roots['models'] / entry['path']
            destination[key] = authenticated_array(p, entry['sha256'], shape, dtype='<f4', order='x-major, y-fastest')
            record(p, f'{group}/' + entry['path'])
    if set(models) != {'vp', 'vs', 'rho'} or np.any(models['vs'] < 0) or np.any(models['vp'] <= 0) or np.any(models['rho'] <= 0):
        raise ValueError('physical background models required')
    water = models['vs'] == 0
    if np.any(raw['mu'][water] != 0) or np.signbit(raw['mu'][water]).any():
        raise ValueError('raw water mu must be exact positive zero')
    for a in shots['mu'].values():
        if np.any(a[water] != 0) or np.signbit(a[water]).any():
            raise ValueError('per-shot raw water mu must be exact positive zero')
    unchanged(runs, records)
    inv['inputs'].sort(key=lambda r: r['path'])
    inv['shots'] = [r['shot'] for r in records]
    protected = tuple(set([runs.parent, roots['models'].parents[1], config_path.parent]))
    result = ImageProducts(raw, shots, models, reference, cfg['grid'], inv, protected, identities)
    result.verify_unchanged()
    return result


def safe_output(output, protected):
    output = Path(output).resolve()
    for path in protected:
        path = Path(path).resolve()
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError('output aliases or overlaps accepted evidence')
    if output.exists():
        raise FileExistsError('fresh output required; refusing overwrite')
    return output


def write_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))


def write_array(path, values):
    with Path(path).open('xb') as stream:
        np.save(stream, np.asarray(values), allow_pickle=False)
