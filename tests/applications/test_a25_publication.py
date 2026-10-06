"""Offline integrity, provenance, missing-artifact and report safety gates."""
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools.denise_case.a25_publication import PACKAGE, authenticate, contained, digest, historical_h15_inventory, portable, recheck, render, saved_array, validate
from tools.denise_case.campaign import encoded
from tools.denise_case.image_products import safe_output


def test_index_content_ids_counts_and_historical_identity():
    d = json.loads((PACKAGE / 'evidence.json').read_text())
    assert d['counts'] == {'H15': 900, 'H5': 200}
    assert d['a25a_content_id'] == 'c32882f58f196ce7928c379db9e512f1c62ae62e77e2a600cfb8e53800fb0e07'
    assert d['references']['A25A']['receipt.json']['sha256'] == 'ddd7f86b19dc07c4c202670961020217609238334c8331f55e74e4d2c410e4b0'
    assert d['references']['A25B']['H5/summary.json']['sha256'] == 'a75de5bd5d72f966f07e97068d7c9db7098501e3ea5aa1a7d848bd93f71de356'
    assert digest({'b': 2, 'a': 1}) == digest({'a': 1, 'b': 2})


def test_missing_evidence_no_process_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('no numerical fallback'))
    roots = {k: tmp_path for k in ('A25B', 'A25A', 'CORE', 'RUNS', 'MODELS')}
    with pytest.raises(FileNotFoundError, match='missing frozen artifact'):
        validate(roots)
    assert not list(tmp_path.iterdir())


def test_historical_inventory_does_not_confuse_runtime_namespace():
    identities = {'archive/campaign/H15/campaign.json': {'sha256': 'a'},
                  'archive/campaign/H15/shots/001/receipt.json': {'sha256': 'b'},
                  'archive/campaign/continuation3/H15/processes/001/process.json': {'sha256': 'c'}}
    assert historical_h15_inventory(identities) == {
        'H15/campaign.json': {'sha256': 'a'}, 'H15/shots/001/receipt.json': {'sha256': 'b'}}
    with pytest.raises(ValueError, match='unique'): historical_h15_inventory({})
    identities['other/H15/campaign.json'] = {'sha256': 'd'}
    with pytest.raises(ValueError, match='unique'): historical_h15_inventory(identities)


def test_authenticate_mutation_before_after(tmp_path):
    path = tmp_path / 'raw'; path.write_bytes(b'frozen')
    checked = {}
    identity = {'sha256': hashlib.sha256(b'frozen').hexdigest(), 'bytes': 6}
    assert authenticate(tmp_path, 'raw', identity, checked) == path
    assert recheck(checked) == {'unchanged': True, 'files_checked': 1}
    path.write_bytes(b'mutate')
    with pytest.raises(ValueError, match='identity mismatch'): authenticate(tmp_path, 'raw', identity, {})
    with pytest.raises(ValueError, match='changed'): recheck(checked)


@pytest.mark.parametrize('path', ['../raw', '/raw', 'x:raw', 'x\\raw'])
def test_escape_refused(tmp_path, path):
    with pytest.raises(ValueError): contained(tmp_path, path)


@pytest.mark.parametrize('prefix', [chr(67)+':/', chr(70)+':\\', '/mnt/'+chr(102)+'/', 'file:'+'///'])
def test_machine_alias_refused(prefix):
    with pytest.raises(ValueError, match='machine-specific'): portable({'path': prefix+'evidence'})


def test_saved_array_exact_layout_and_hash(tmp_path):
    a = np.zeros((2, 3, 4), dtype='<f8'); path = tmp_path / 'a.npy'; np.save(path, a)
    sha = hashlib.sha256(a.tobytes()).hexdigest()
    loaded = saved_array(path, sha, a.shape)
    assert not loaded.flags.writeable
    with pytest.raises(ValueError, match='identity'): saved_array(path, '0'*64, a.shape)
    with pytest.raises(ValueError, match='shape'): saved_array(path, sha, (2, 4, 3))
    np.save(path, a.astype('>f8'))
    with pytest.raises(ValueError, match='endian'): saved_array(path, sha, a.shape)


def test_fresh_output_and_protected_overlap(tmp_path):
    evidence = tmp_path / 'evidence'; evidence.mkdir()
    for path in (evidence, evidence/'out', tmp_path):
        with pytest.raises(ValueError): safe_output(path, (evidence,))
    existing = tmp_path/'existing'; existing.mkdir()
    with pytest.raises(FileExistsError): safe_output(existing, (evidence,))


def test_report_deterministic_classes_and_full_precision(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('no execution'))
    comparisons = json.loads((PACKAGE/'frequency_comparison.json').read_text())
    summary = {'class': 'DERIVED_DIAGNOSTIC', 'frequency_comparison': comparisons,
               'claims_and_limits': json.loads((PACKAGE/'claims.json').read_text())['claims_and_limits']}
    for name in ('one', 'two'):
        output = tmp_path/name; output.mkdir(); render(summary, output)
    for name in ('index.html', 'summary.json'):
        assert (tmp_path/'one'/name).read_bytes() == (tmp_path/'two'/name).read_bytes()
        assert str(tmp_path).encode() not in (tmp_path/'one'/name).read_bytes()
    text = (tmp_path/'one/index.html').read_text()
    for caveat in ('DISPLAY_ONLY', 'INDETERMINATE / CONFOUNDED', 'ROOT CAUSE UNRESOLVED', 'first-solid-row', '5.50625', 'FLUID-3/4'):
        assert caveat in text
    assert comparisons['vp_c1560']['H15']['target_fraction'] == 0.309267506830992
    assert comparisons['vp_c2960']['H5']['r90_m'] == 488.2622246293481
    with pytest.raises(FileExistsError): render(summary, tmp_path/'one')


def test_all_publication_records_portable_and_no_runtime_imports():
    for path in PACKAGE.glob('*.json'): portable(json.loads(path.read_text()))
    tools = PACKAGE.parents[1]/'tools/denise_case'
    for filename in ('normal_operator.py', 'normal_operator_analysis.py', 'a25_publication.py'):
        text = (tools/filename).read_text()
        assert 'import run' not in text and 'ctypes' not in text and 'subprocess' not in text
