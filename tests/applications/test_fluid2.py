"""Application integration checks, not a duplicate of the core scientific oracle."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.denise_case import fluid2


def image_fixture():
    vs=np.array([[0,0,0],[0,0,0],[1,1,1],[1,1,1]],np.float32)
    lam=np.arange(1,13,dtype=np.float64).reshape(4,3)
    mu=np.array([[0,0,0],[0,0,0],[1,2,3],[4,5,6]],np.float64)
    return lam,mu,vs


def test_cpu_material_accepts_exact_zero_without_floor_or_mutation():
    lam=np.ones((3,4),np.float32); mu=np.zeros_like(lam); rho=np.ones_like(lam)
    before=[a.tobytes() for a in (lam,mu,rho)]
    assert fluid2.material_check(lam,mu,rho)['zero_mu_cells']==12
    assert before==[a.tobytes() for a in (lam,mu,rho)]
    mu[1,2]=-np.nextafter(np.float32(0),np.float32(1))
    with pytest.raises(ValueError,match='invalid'):
        fluid2.material_check(lam,mu,rho)


def test_raw_water_check_is_exact_positive_zero_not_tolerance_or_postmask():
    lam,mu,vs=image_fixture()
    before=mu.tobytes()
    qc=fluid2.image_checks(lam,mu,vs)
    assert qc['water_mu_exact_positive_zero'] and not qc['post_mask_applied']
    assert qc['water_mu']['max_abs']==0 and qc['solid_mu']['nonzero_count']==6
    assert qc['water_interior_lambda']['nonzero_count']==3
    assert mu.tobytes()==before
    mu[0,0]=-0.0
    with pytest.raises(ValueError,match=r'exact \+0'):
        fluid2.image_checks(lam,mu,vs)
    mu[0,0]=np.nextafter(0.,1.)
    with pytest.raises(ValueError,match=r'exact \+0'):
        fluid2.image_checks(lam,mu,vs)


def test_expected_sensitivity_and_nonfinite_fail_closed():
    lam,mu,vs=image_fixture()
    lam[1]=0
    with pytest.raises(ValueError,match='sensitivity absent'):
        fluid2.image_checks(lam,mu,vs)
    lam,mu,vs=image_fixture(); mu[3,0]=np.inf
    with pytest.raises(ValueError,match='nonfinite'):
        fluid2.image_checks(lam,mu,vs)
    with pytest.raises(ValueError,match='shape'):
        fluid2.image_checks(lam.T,mu,vs)


def test_interface_profiles_use_raw_depth_rows():
    lam,mu,vs=image_fixture()
    qc=fluid2.image_checks(lam,mu,vs)
    interface=qc['interface']
    assert interface['last_water_row']==1 and interface['first_solid_row']==2
    assert interface['profiles']['mu'][2]['l2']==pytest.approx(np.linalg.norm(mu[2]))
    assert interface['profiles']['lambda'][1]['depth_center_m']==40
    assert 'NOT FLUID-3' in interface['acceptance_scope']


@pytest.mark.parametrize('backend,executable',[('CUDA-M9e-4','../../bin/denise'),('CPU-M9','../../bin/denise_cuda')])
def test_backend_selection_cannot_dispatch_to_cuda(tmp_path,backend,executable):
    p=tmp_path/'applications/fluid2/case.json'; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'core_sha':fluid2.CORE_SHA,'backend':backend,'executable':executable}))
    with pytest.raises(ValueError,match='CPU'):
        fluid2.settings(p)


def test_resumable_plan_is_100_deterministic_five_shot_batches(tmp_path,monkeypatch):
    monkeypatch.setattr(fluid2,'settings',lambda path:(tmp_path,tmp_path,{'batch_size':5},None,None))
    plan=fluid2.plan(tmp_path/'case.json')
    assert plan['batches'][0]==[1,2,3,4,5]
    assert plan['batches'][-1]==[96,97,98,99,100]
    assert sum(plan['batches'],[])==list(range(1,101))
    assert plan['no_automatic_monolithic_run']


def test_failed_attempt_is_never_relaunched_by_batch(tmp_path,monkeypatch):
    monkeypatch.setattr(fluid2,'settings',lambda path:(tmp_path,tmp_path,{'batch_size':5},None,None))
    monkeypatch.setattr(fluid2,'require_core',lambda *args,**kwargs:None)
    p=tmp_path/'runs/shot_050/run_metadata.json'; p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'complete':False}))
    before=p.read_bytes()
    with pytest.raises(ValueError,match='not safely resumable'):
        fluid2.batch([50],tmp_path/'case.json')
    assert p.read_bytes()==before


def test_interrupted_launch_marker_blocks_materialization(tmp_path,monkeypatch):
    monkeypatch.setattr(fluid2,'settings',lambda path:(tmp_path,tmp_path,{},None,None))
    p=tmp_path/'runs/shot_001/attempt.json'; p.parent.mkdir(parents=True)
    p.write_bytes(b'{"interrupted":true}')
    with pytest.raises(FileExistsError,match='relaunch'):
        fluid2.prepare([1],'shot_001',tmp_path/'case.json')
    assert p.read_bytes()==b'{"interrupted":true}'
