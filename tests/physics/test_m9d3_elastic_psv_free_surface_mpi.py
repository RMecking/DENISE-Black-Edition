"""M9d3 owned MPI surface graph; inherited mixed-precision gates unchanged."""
from functools import lru_cache
import json
import numpy as np
import pytest
from tests.physics.test_m9d2_elastic_psv_mpi import mpi_harness,run,read,TOPS,errors
from tests.physics.test_m9b1_elastic_psv_born_production import _production_dot_metrics,_assert_production_dot_closes,_assert_production_close
from tests.utilities.elastic_psv_free_surface_reference import fixture,forward,born,adjoint
from tests.utilities.elastic_psv_born_reference import dot

@lru_cache(None)
def reference(direction,component):
    exp=fixture();background,tapes,_=forward(exp)
    rng=np.random.default_rng(423)
    dl=np.asarray(.01*exp.lam*rng.normal(size=exp.lam.shape),np.float32).astype(float)
    dm=np.asarray(.01*exp.mu*rng.normal(size=exp.mu.shape),np.float32).astype(float)
    if direction=='lambda':dm[:]=0
    if direction=='mu':dl[:]=0
    j=born(exp,tapes,dl,dm)
    data=np.asarray(np.random.default_rng(424).normal(size=j.shape),np.float32).astype(float)
    if component is not None:data[...,1-component]=0
    gl,gm=adjoint(exp,tapes,data)
    scale=max(np.linalg.norm(j)*np.linalg.norm(data),np.linalg.norm(np.stack((dl,dm)))*np.linalg.norm(np.stack((gl,gm))))
    oracle_metrics=dict(absolute_ceiling=5.e-13*scale,absolute_residual=abs(dot(j,data)-dot(dl,gl)-dot(dm,gm)))
    return exp,background,dl,dm,data,j,gl,gm,oracle_metrics

@pytest.mark.parametrize('top',TOPS)
@pytest.mark.parametrize('direction',('lambda','mu','joint'))
@pytest.mark.parametrize('component',(0,1,None),ids=('vx','vy','both'))
def test_surface_distributed_j_jt_dot_and_reference(mpi_harness,top,direction,component):
    exp,bg,dl,dm,data,j,gl,gm,om=reference(direction,component)
    directory,_=run(mpi_harness,exp,dl,dm,data,top,env={'M9D3_FREE_SURF':'1'},label=f'surface-{top}-{direction}-{component}')
    actual=read(directory,exp)
    _assert_production_close(actual['background'],bg,2.e-6)
    metrics=_production_dot_metrics(actual['j'],data,dl,dm,actual['gl'],actual['gm'],j,gl,gm,om)
    _assert_production_dot_closes(metrics)
    (directory/'surface-dot.json').write_text(json.dumps(metrics,indent=2,default=float))
    for name in ('gl','gm'):
        serial=np.fromfile(directory/f'serial_{name}.bin',np.float64).reshape(exp.ny,exp.nx)
        mask=np.zeros_like(serial,bool);mask[:3]=True;mask[:,exp.nx//2-2:exp.nx//2+2]=True
        assert np.linalg.norm((actual[name]-serial)[mask])/np.linalg.norm(serial[mask])<=1.e-12
    print('FP32/FP64_PRODUCTION_MPI',top,direction,component,actual['gl_metrics'],actual['gm_metrics'],metrics['operand_scale_residual'])

@pytest.mark.parametrize('top',TOPS)
def test_surface_distributed_full_segmented_operands_and_continuation(mpi_harness,top):
    exp,_,dl,dm,data,_,_,_,_=reference('joint',None)
    full,_=run(mpi_harness,exp,dl,dm,data,top,0,env={'M9D3_FREE_SURF':'1'},label='surface-full')
    seg,_=run(mpi_harness,exp,dl,dm,data,top,3,env={'M9D3_FREE_SURF':'1'},label='surface-seg')
    read(full,exp);read(seg,exp)
    for name in ('background','j','gl','gm','strain0','strain1','strain2','strain3'):
        assert (full/f'{name}.bin').read_bytes()==(seg/f'{name}.bin').read_bytes()
    diagnostic=json.loads((seg/'diagnostics.json').read_text())
    for r in diagnostic['ranks']:
        assert r['retained']==r['payload']+r['metadata']+r['schedule']+r['operand']

@pytest.mark.parametrize('top',TOPS)
def test_surface_source_at_row_one_rejects_before_outputs(mpi_harness,top):
    exp=fixture(source=(4,1));zeros=np.zeros_like(exp.lam)
    data=np.zeros((exp.nt,len(exp.receivers),2))
    directory,p=run(mpi_harness,exp,zeros,zeros,data,top,
       env={'M9D3_FREE_SURF':'1','M9D2_FAIL_PHASE':'create','M9D2_FAIL_RANK':'-1'},label='surface-source-reject')
    assert 'M9D2_EXPECTED_COLLECTIVE_FAILURE' in p.stdout
    assert not (directory/'background.bin').exists()

def test_surface_minimum_top_tile_three_rejects(mpi_harness):
    exp=fixture(ny=6);zeros=np.zeros_like(exp.lam)
    data=np.zeros((exp.nt,len(exp.receivers),2))
    directory,p=run(mpi_harness,exp,zeros,zeros,data,(1,2),
       env={'M9D3_FREE_SURF':'1','M9D2_FAIL_PHASE':'create','M9D2_FAIL_RANK':'-1'},label='surface-depth-reject')
    assert 'M9D2_EXPECTED_COLLECTIVE_FAILURE' in p.stdout
    assert not (directory/'background.bin').exists()

@pytest.mark.parametrize('nt,expected',[(55,False),(120,True)])
def test_surface_collective_backend_selection(mpi_harness,nt,expected):
    from tests.physics.test_m9d2_elastic_psv_mpi import experiment
    from tests.utilities.elastic_psv_born_reference import deterministic_model_direction,deterministic_data
    exp=experiment(True,nx=32,ny=24,nt=nt,fw=3)
    dl,dm=deterministic_model_direction(exp)
    directory,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(4,1),-32,
                    env={'M9D3_FREE_SURF':'1'},label='surface-collective-selection')
    read(directory,exp)
    ranks=json.loads((directory/'diagnostics.json').read_text())['ranks']
    benefits=[r['estimate']<r['full'] for r in ranks]
    if expected:assert all(benefits)
    else:assert any(benefits) and not all(benefits)
    assert all(bool(r['segmented'])==expected for r in ranks)

def test_surface_exact_complete_byte_equality_selects_full(mpi_harness):
    # Owned 2x4 cells, NT24, S14: payload2080 + metadata624 + schedule112
    # + operand256 = FULL3072. Precision/field counts are unchanged.
    exp=fixture(nx=6,ny=8,nt=24,cpml=False,source=(3,2))
    rng=np.random.default_rng(423)
    dl=exp.lam*.01*rng.normal(size=exp.lam.shape)
    dm=exp.mu*.01*rng.normal(size=exp.mu.shape)
    data=np.random.default_rng(424).normal(size=(exp.nt,len(exp.receivers),2))
    directory,_=run(mpi_harness,exp,dl,dm,data,(3,2),-14,
                    env={'M9D3_FREE_SURF':'1'},label='surface-equal-complete-bytes')
    read(directory,exp)
    ranks=json.loads((directory/'diagnostics.json').read_text())['ranks']
    assert all(r['full']==r['estimate']==3072 and not r['segmented'] for r in ranks)
