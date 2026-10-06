"""Independent small authenticated-product contracts; no solver execution."""
import json
import shutil

import numpy as np
import pytest

from tests.applications.test_campaign import fixture, SHAPE
from tools.denise_case.campaign import aggregate_campaign
from tools.denise_case.image_products import authenticated_array, load_products, safe_output
from tools.denise_case.model_io import sha256_file, write_grid


def products_fixture(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    runs = source / 'application/runs'; runs.parent.mkdir()
    summary, vs_path = fixture(runs, ids=tuple(range(1, 101)))
    historical = source / 'historical/input/models'; historical.mkdir(parents=True)
    models = {}
    for k, a in (('vp', np.full(SHAPE, 3.)), ('vs', np.array([[0,0],[0,0],[1,1]])), ('rho', np.ones(SHAPE))):
        p = historical / ('smooth2.' + k); write_grid(p, a)
        models[k] = {'path': p.name, 'sha256': sha256_file(p)}
    result = aggregate_campaign(runs, runs / 'aggregate', range(1, 101), shape=SHAPE, vs_model=vs_path)
    config = tmp_path / 'configuration'; config.mkdir()
    cfg = {'execution_core': 'f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d',
           'benchmark_revision': 'a' * 40,
           'grid': {'shape_y_x': list(SHAPE), 'dh_m': 20, 'image_dtype': '<f8', 'image_order': 'row-major [depth_y,x]'},
           'roots': {'runs': '../source/application/runs', 'models': '../source/historical/input/models'},
           'shots': list(range(1, 101)), 'campaign_receipt': '../../campaign.json',
           'aggregate_receipt': 'aggregate/receipt.json', 'aggregate_hashes': {k:v['sha256'] for k,v in result['outputs'].items()},
           'models': models}
    # fixture writes campaign.json in runs.parent, not source root.
    cfg['campaign_receipt'] = '../campaign.json'
    path = config / 'case.json'; path.write_text(json.dumps(cfg))
    return path, runs


def test_loader_hash_bound_readonly_and_output_isolation(tmp_path):
    config, runs = products_fixture(tmp_path)
    data = load_products(config)
    assert len(data.shots['lambda']) == 100
    assert not data.raw['lambda'].flags.writeable
    with pytest.raises(ValueError):
        data.raw['lambda'].setflags(write=True)
    assert not data.models['vp'].flags.writeable
    assert data.verify_unchanged()['unchanged']
    for path in (runs / 'new', runs.parent, tmp_path / 'source'):
        with pytest.raises(ValueError):
            safe_output(path, data.protected)
    assert safe_output(tmp_path / 'fresh', data.protected) == tmp_path / 'fresh'


@pytest.mark.parametrize('bad', ['hash', 'size', 'endian', 'layout', 'nan'])
def test_array_input_rejections(tmp_path, bad):
    p = tmp_path / 'image.bin'; a = np.arange(6, dtype='<f8').reshape(3, 2)
    if bad == 'nan':
        a[0,0] = np.nan
    p.write_bytes(a.tobytes()); h = sha256_file(p)
    kwargs = {}
    if bad == 'hash':
        h = 'a' * 64
    if bad == 'size':
        p.write_bytes(p.read_bytes()[:-1])
    if bad == 'endian':
        kwargs['dtype'] = '>f8'
    if bad == 'layout':
        kwargs['order'] = 'x-major, y-fastest'
    with pytest.raises(ValueError):
        authenticated_array(p, h, (3, 2), **kwargs)


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'model_hash', 'metadata_raw', 'layout'])
def test_product_loader_rejects_bound_corruption(tmp_path, bad):
    config, runs = products_fixture(tmp_path)
    if bad == 'missing':
        (runs / 'shot_002/run_metadata.json').unlink()
    elif bad == 'duplicate':
        shutil.copytree(runs / 'shot_001', runs / 'shot_duplicate')
    elif bad == 'model_hash':
        p = tmp_path / 'source/historical/input/models/smooth2.vp'; p.write_bytes(b'0' * 24)
    else:
        p = runs / 'shot_001/run_metadata.json'; meta = json.loads(p.read_text())
        meta['raw_images']['lambda']['sha256' if bad == 'metadata_raw' else 'order'] = 'wrong'
        p.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_products(config)


def test_existing_and_symlink_alias_outputs_rejected(tmp_path):
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    existing = tmp_path / 'existing'; existing.mkdir()
    with pytest.raises(FileExistsError):
        safe_output(existing, [evidence])
    link = tmp_path / 'link'
    try:
        link.symlink_to(evidence, target_is_directory=True)
    except OSError:
        return  # lexical/resolved ancestry checks remain covered without Windows symlink privilege
    with pytest.raises(ValueError):
        safe_output(link / 'output', [evidence])
