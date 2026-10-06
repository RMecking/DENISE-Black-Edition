"""Real production report orchestration, tiny fixtures, no solver or compiler."""
import json
import subprocess

import numpy as np
import pytest
import zlib

from tools.denise_case.image_products import ImageProducts, immutable
from tools.denise_case.image_conditioning_report import process_products
from tools.denise_case.model_io import sha256_file


def synthetic_products(tmp_path):
    shape = (24, 50)  # fixed ten-cell taper must not overlap
    vp = np.full(shape, 3, dtype='<f4'); vs = np.ones(shape, dtype='<f4'); vs[0] = 0
    rho = np.ones(shape, dtype='<f4')
    lam = np.arange(np.prod(shape), dtype='<f8').reshape(shape); mu = lam.copy(); mu[0] = 0
    shots = {k: {i: immutable(a / 100) for i in range(1, 101)} for k,a in (('lambda',lam),('mu',mu))}
    evidence = tmp_path / 'source'; evidence.mkdir(); p = evidence / 'sentinel'; p.write_bytes(b'raw scientific bytes')
    identities = {str(p): {'sha256': sha256_file(p), 'bytes': p.stat().st_size}}
    return ImageProducts({'lambda': immutable(lam), 'mu': immutable(mu)}, shots,
        {k: immutable(a) for k,a in (('vp',vp),('vs',vs),('rho',rho))}, {},
        {'shape_y_x': list(shape), 'dh_m': 20},
        {'execution_core': 'f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d', 'benchmark_revision': 'a'*40,
         'inputs': [{'path': 'source/sentinel', **identities[str(p)]}]}, (evidence,), identities)


def test_actual_report_deterministic_portable_classes_and_caveats(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no subprocess')))
    products = synthetic_products(tmp_path)
    first = process_products(products, tmp_path / 'first', investigate=False)
    second = process_products(products, tmp_path / 'second', investigate=False)
    assert first == second
    assert {p['class'] for p in first['products']} == {'DERIVED_NUMERICAL', 'DISPLAY_ONLY'}
    assert first['immutable_inputs']['unchanged']
    by_path = {p['path']: p for p in first['products']}
    assert by_path['qc/lambda_depth.json']['class'] == 'DERIVED_NUMERICAL'
    assert by_path['display/scales.json']['class'] == 'DISPLAY_ONLY'
    assert by_path['report/index.html']['class'] == 'DISPLAY_ONLY'
    for p in first['outputs']:
        if p['path'] not in ('input_inventory.json', 'processing_contract.json'):
            assert p['class'] in ('DERIVED_NUMERICAL', 'DISPLAY_ONLY')
    for filename in ('report/summary.json', 'report/index.html', 'receipt.json'):
        a = (tmp_path / 'first' / filename).read_bytes()
        assert a == (tmp_path / 'second' / filename).read_bytes()
        assert str(tmp_path).encode() not in a
    text = (tmp_path / 'first/report/index.html').read_text()
    for requirement in ('Vs=0 / mu=0', 'no Vs floor', 'not reflectivity', 'not illumination compensation',
                        '15 Hz', '5.50625', 'FLUID-3/4', 'Shot 68', 'first-solid-row'):
        assert requirement in text
    summary = json.loads((tmp_path / 'first/report/summary.json').read_text())
    assert summary['solver_executions'] == 0
    svg = (tmp_path / 'first/display/lambda_p99.svg').read_text()
    assert 'Horizontal x (km)' in svg and 'Depth (km)' in svg
    scales = json.loads((tmp_path / 'first/display/scales.json').read_text())
    s = scales['lambda_gain_p1_cap3']
    gained = np.load(tmp_path / 'first/derived/lambda_gain_p1_cap3.npy')
    assert s['saturated_count'] == np.count_nonzero(np.abs(gained) > s['q'])
    assert s['saturated_fraction'] == s['saturated_count'] / gained.size
    valid = np.load(tmp_path / 'first/derived/lambda_dz_valid.npy')
    assert scales['lambda_dz']['total_count'] == int(valid.sum())
    assert (tmp_path / 'first/qc/operator_controls.json').is_file()


def test_zero_display_is_neutral(tmp_path):
    data = synthetic_products(tmp_path)
    zero = immutable(np.zeros(data.grid['shape_y_x']))
    products = ImageProducts({'lambda': zero, 'mu': zero},
        {k: {i: zero for i in range(1, 101)} for k in ('lambda', 'mu')},
        data.models, {}, data.grid, data.inventory, data.protected, data.identities)
    process_products(products, tmp_path / 'zeros', investigate=False)
    png = (tmp_path / 'zeros/display/lambda_p99.png').read_bytes()
    position = 8; payload = b''
    while position < len(png):
        n = int.from_bytes(png[position:position+4], 'big'); kind = png[position+4:position+8]
        if kind == b'IDAT':
            payload += png[position+8:position+8+n]
        position += n + 12
    rows = np.frombuffer(zlib.decompress(payload), dtype=np.uint8).reshape(24, 151)
    assert np.all(rows[:, 1:] == 245)


@pytest.mark.parametrize('changed', ['execution_core', 'benchmark_revision', 'aggregate_hashes', 'models', 'grid', 'shots'])
def test_production_rejects_changed_canonical_identity(tmp_path, changed):
    from pathlib import Path
    from tools.denise_case.image_conditioning_report import run_conditioning
    cfg = json.loads((Path(__file__).parents[2] / 'applications/marmousi2_a25a/case.json').read_text())
    if changed == 'aggregate_hashes':
        cfg[changed]['lambda'] = '0' * 64
    elif changed == 'models':
        cfg[changed]['vp']['sha256'] = '0' * 64
    elif changed == 'grid':
        cfg[changed]['dh_m'] = 40
    elif changed == 'shots':
        cfg[changed] = list(range(1, 100))
    else:
        cfg[changed] = '0' * 40
    path = tmp_path / 'case.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='frozen canonical'):
        run_conditioning(path, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_cli_routes_without_loading_forward_manifest(tmp_path, monkeypatch, capsys):
    from tools.denise_case import cli, image_conditioning_report
    calls = []
    def route(case, output):
        calls.append((case, output)); return {'content_id': 'sentinel'}
    monkeypatch.setattr(image_conditioning_report, 'run_conditioning', route)
    monkeypatch.setattr(cli, 'load_case', lambda *a: (_ for _ in ()).throw(AssertionError('no forward manifest')))
    case = tmp_path / 'case.json'; output = tmp_path / 'fresh'
    assert cli.main(['image-condition', '--case', str(case), '--output', str(output)]) == 0
    assert calls == [(case, output)]
    assert json.loads(capsys.readouterr().out)['solver_executions'] == 0
