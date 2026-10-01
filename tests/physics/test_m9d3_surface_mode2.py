"""Actual native mixed-precision migration/MODE=2, preserving paired files."""
import ctypes
from dataclasses import replace
import hashlib
import json
import subprocess
import numpy as np
import pytest
from tests.physics import test_m9c_elastic_psv_migration_driver as canonical
from tests.physics.test_m9c_elastic_psv_migration_driver import migration_library
from tests.physics.test_m9d2_elastic_psv_mpi import TOPS,errors
from tests.utilities.elastic_psv_free_surface_reference import fixture,forward,adjoint

def case(kind):
    exp=fixture()
    if kind=='normal':
        exp=replace(exp,lam=np.full_like(exp.lam,6.44e9),mu=np.full_like(exp.mu,5.78e9),
                    sources=((4,4),),receivers=((4,1),(4,6)))
    elif kind=='oblique':exp=replace(exp,sources=((2,4),),receivers=((8,1),(4,2),(7,6),(1,1)))
    else:exp=replace(exp,sources=((4,2),(5,2),(8,5)))
    data=[];oracle=[]
    for source in exp.sources:
        single=replace(exp,sources=(source,))
        d,tapes,_=forward(single);d=np.asarray(d,np.float32)
        data.append(d);oracle.append(adjoint(single,tapes,d.astype(float)))
    return exp,data,tuple(sum((g[k] for g in oracle),np.zeros_like(exp.lam)) for k in (0,1))

def owner(exp,data):
    result=canonical.RequestOwner(exp,data);result.request.free_surface=1
    return result

@pytest.mark.parametrize('kind',('normal','oblique','multishot'))
def test_surface_serial_migration_raw_images(migration_library,kind):
    exp,data,reference=case(kind)
    result,diag=canonical._run(migration_library,owner(exp,data))
    for a,b in zip(result,reference):assert canonical._relative(a,b)<=6e-5
    assert diag['shots_completed']==len(exp.sources)

@pytest.mark.integration
@pytest.mark.parametrize('top',TOPS)
@pytest.mark.parametrize('kind',('normal','oblique','multishot'))
def test_surface_true_mode2_against_serial(tmp_path,migration_library,denise_binary,top,kind):
    exp,data,_=case(kind)
    reference,_=canonical._run(migration_library,owner(exp,data))
    paths=canonical._write_mode2_case(tmp_path,exp,data)
    prepared={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'prepared').iterdir()}
    text=(tmp_path/'denise.inp').read_text().replace('FREE_SURF =0','FREE_SURF =1')
    text=text.replace('NPROCX =1',f'NPROCX ={top[0]}').replace('NPROCY =1',f'NPROCY ={top[1]}')
    (tmp_path/'denise.inp').write_text(text)
    p=subprocess.run(['mpiexec','--oversubscribe','-n',str(top[0]*top[1]),str(denise_binary),'denise.inp','workflow.inp'],cwd=tmp_path,capture_output=True,text=True,timeout=120)
    (tmp_path/'mode2.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stdout+p.stderr
    metrics=[]
    for name,b in zip(('lambda','mu'),reference):
        a=np.fromfile(paths[name],np.float64).reshape(exp.ny,exp.nx)
        metric=errors(a,b);metrics.append(metric)
        if top==(1,1):assert metric['bitwise']
        else:assert metric['relative_l2']<=1e-12,metric
        assert paths[name].stat().st_size==exp.nx*exp.ny*8
    assert prepared=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'prepared').iterdir()}
    assert not list((tmp_path/'image').glob('*.tmp'))
    (tmp_path/'surface-mode2-equivalence.json').write_text(json.dumps(metrics,indent=2))
    print('FP32/FP64_PRODUCTION_MODE2',kind,top,metrics)

@pytest.mark.integration
@pytest.mark.parametrize('top',TOPS)
def test_surface_mode2_rejects_surface_source_and_removes_pair(tmp_path,denise_binary,top):
    exp=fixture(source=(4,1));data=[np.zeros((exp.nt,len(exp.receivers),2),np.float32)]
    paths=canonical._write_mode2_case(tmp_path,exp,data)
    for key in ('lambda','mu'):paths[key].write_bytes(b'stale image')
    text=(tmp_path/'denise.inp').read_text().replace('FREE_SURF =0','FREE_SURF =1')
    text=text.replace('NPROCX =1',f'NPROCX ={top[0]}').replace('NPROCY =1',f'NPROCY ={top[1]}')
    (tmp_path/'denise.inp').write_text(text)
    p=subprocess.run(['mpiexec','--oversubscribe','-n',str(top[0]*top[1]),str(denise_binary),'denise.inp','workflow.inp'],cwd=tmp_path,capture_output=True,text=True,timeout=90)
    (tmp_path/'failure.log').write_text(p.stdout+p.stderr)
    assert p.returncode!=0 and 'j=1' in p.stdout+p.stderr
    assert not paths['lambda'].exists() and not paths['mu'].exists()
