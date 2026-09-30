"""Collective M9d2 scientific/ownership/failure gates; immutable M9b oracle."""
from dataclasses import replace
from pathlib import Path
import functools
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from tests.physics import test_m9b1_elastic_psv_born_production as accepted
from tests.physics import test_m9c_elastic_psv_migration_driver as mode2
from tests.utilities.elastic_psv_born_reference import (
    make_experiment, _source_samples, deterministic_model_direction,
    deterministic_data, nonlinear_forward, born_forward, born_adjoint, dot_metrics,
)

TOPS=((1,1),(2,1),(1,2),(2,2))

def experiment(cpml, nx=24, ny=20, nt=80, fw=None):
    base=make_experiment('active_cpml' if cpml else 'interior')
    y,x=np.indices((ny,nx))
    rho=np.asarray(2000*(1+.003*np.sin(x*.23+y*.17)),np.float32).astype(float)
    mu=np.asarray(rho*1700**2*(1+.004*np.cos(x*.21-y*.13)),np.float32).astype(float)
    lam=np.asarray(rho*(3000**2-2*1700**2)*(1+.005*np.sin(x*.13+y*.19)),np.float32).astype(float)
    # Every interface, diagonal, and global wrap is present in deliberately
    # scrambled global receiver order. Coordinates here are canonical 1-based.
    receivers=((nx,ny),(2,2),(nx//2+1,ny//2+1),(2,ny-1),
               (nx-1,2),(nx//2,ny//2),(nx//2+1,ny//2),(1,1))
    return replace(base,nx=nx,ny=ny,nt=nt,fw=(fw if fw is not None else 3) if cpml else 0,
                   rho=rho,mu=mu,lam=lam,sources=((nx//2,ny//2),),
                   receivers=receivers,source_t0=.002)

def write_input(path,exp,dl,dm,data):
    si,sj=exp.sources[0]
    with path.open('wb') as f:
        np.array([exp.nx,exp.ny,exp.nt,exp.fw,si-1,sj-1,len(exp.receivers)],np.int32).tofile(f)
        np.array([exp.dh,exp.dt,np.sqrt(np.max((exp.lam+2*exp.mu)/exp.rho)),exp.pml_reflection,exp.pml_power,exp.pml_kmax,exp.pml_fpml],np.float32).tofile(f)
        for a in [exp.lam,exp.mu,exp.rho,dl,dm,_source_samples(exp)]:np.asarray(a,np.float32).tofile(f)
        np.array([i-1 for i,j in exp.receivers],np.int32).tofile(f)
        np.array([j-1 for i,j in exp.receivers],np.int32).tofile(f)
        np.asarray(data,np.float32).tofile(f)

def errors(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    difference=a-b
    # Ordered IEEE integer encodings support signs/negative zeros without
    # unsigned subtraction overflow. Near-zero ULP maxima are quantified too.
    def ordered(x):
        bits=x.view(np.uint64)
        return np.where(bits>>63,~bits,bits|(np.uint64(1)<<np.uint64(63)))
    ai,bi=ordered(a),ordered(b)
    ulp=np.maximum(ai,bi)-np.minimum(ai,bi)
    location=np.unravel_index(int(np.argmax(np.abs(difference))),a.shape)
    ulp_location=np.unravel_index(int(np.argmax(ulp)),a.shape)
    return dict(bitwise=a.tobytes()==b.tobytes(),relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(b),np.finfo(float).tiny)),max_absolute=float(np.max(np.abs(difference))),max_ulp=int(ulp.max()),max_location=[int(i) for i in location],ulp_location=[int(i) for i in ulp_location])

@pytest.fixture(scope='session')
def mpi_harness(tmp_path_factory,repository_root):
    directory=tmp_path_factory.mktemp('m9d2-harness')
    executable=directory/'operator'
    compiler=shutil.which('mpicc');assert compiler,'local MPI C compiler required'
    subprocess.run([compiler,'-std=c99','-O2','-Wall','-Wextra','-Werror','-pedantic',
                    '-I',str(repository_root/'include'),str(repository_root/'tests/utilities/m9d2_elastic_psv_mpi_harness.c'),
                    str(repository_root/'src/PSV/elastic_psv_born_mpi.c'),'-Wl,--wrap=calloc','-lm','-o',str(executable)],check=True)
    return executable,directory

def run(mpi_harness,exp,dl,dm,data,top,segments=0,env=None,label='case'):
    executable,root=mpi_harness
    directory=root/f'{label}-{len(list(root.iterdir()))}';directory.mkdir()
    write_input(directory/'input.bin',exp,dl,dm,data)
    environment=os.environ.copy();environment.update(env or {})
    p=subprocess.run(['mpiexec','--oversubscribe','-n',str(top[0]*top[1]),str(executable),str(directory/'input.bin'),str(directory),str(top[0]),str(top[1]),str(segments)],capture_output=True,text=True,timeout=120,env=environment)
    (directory/'run.log').write_text(p.stdout+p.stderr)
    assert p.returncode==0,p.stdout+p.stderr
    return directory,p

def read(directory,exp):
    shapes={'background':(exp.nt,len(exp.receivers),2),'j':(exp.nt,len(exp.receivers),2),'gl':(exp.ny,exp.nx),'gm':(exp.ny,exp.nx)}
    result={}
    for name,shape in shapes.items():
        dtype=np.float32 if name in ('background','j') else np.float64
        result[name]=np.fromfile(directory/f'{name}.bin',dtype).reshape(shape)
        reference=np.fromfile(directory/f'serial_{name}.bin',dtype).reshape(shape)
        metrics=errors(result[name],reference)
        if name in ('background','j'):assert metrics['bitwise'],metrics
        else:assert metrics['relative_l2']<=1e-12,metrics
        result[name+'_metrics']=metrics
    (directory/'equivalence.json').write_text(json.dumps({k:v for k,v in result.items() if k.endswith('_metrics')},indent=2))
    return result

@functools.lru_cache(None)
def oracle(cpml,direction,component):
    exp=experiment(cpml)
    dl,dm=deterministic_model_direction(exp,seed=61)
    dl=np.asarray(dl,np.float32).astype(float);dm=np.asarray(dm,np.float32).astype(float)
    if direction=='lambda':dm[:]=0
    if direction=='mu':dl[:]=0
    data=np.asarray(deterministic_data(exp,seed=97,component=component),np.float32).astype(float)
    trajectory=nonlinear_forward(exp,0,save_strain=True)
    ref_j=born_forward(exp,trajectory,dl,dm)
    ref_g=born_adjoint(exp,trajectory,data)
    dot=dot_metrics(exp,dl,dm,data,trajectory)
    return exp,dl,dm,data,ref_j,ref_g,dot

@pytest.mark.parametrize('top',TOPS)
@pytest.mark.parametrize('cpml',[False,True],ids=['interior','cpml'])
@pytest.mark.parametrize('direction',['lambda','mu','joint'])
@pytest.mark.parametrize('component',[0,1,None],ids=['vx','vy','both'])
def test_decomposition_oracle_and_dot_matrix(mpi_harness,top,cpml,direction,component):
    exp,dl,dm,data,ref_j,ref_g,dot=oracle(cpml,direction,component)
    directory,_=run(mpi_harness,exp,dl,dm,data,top,label=f'{top}-{cpml}-{direction}-{component}')
    actual=read(directory,exp)
    metrics=accepted._production_dot_metrics(actual['j'],data,dl,dm,actual['gl'],actual['gm'],ref_j,ref_g.image_lambda_raw,ref_g.image_mu_raw,dot)
    accepted._assert_production_dot_closes(metrics)
    (directory/'dot.json').write_text(json.dumps(metrics,indent=2))
    # ULP/L2 diagnostic maxima are measured globally above; interface strips
    # must individually satisfy the same strict decomposition bound.
    for name in ['gl','gm']:
        ref=np.fromfile(directory/f'serial_{name}.bin',np.float64).reshape(exp.ny,exp.nx)
        mask=np.zeros(ref.shape,bool);mask[:,exp.nx//2-2:exp.nx//2+2]=True;mask[exp.ny//2-2:exp.ny//2+2,:]=True
        assert np.linalg.norm((actual[name]-ref)[mask])/max(np.linalg.norm(ref[mask]),1e-300)<=1e-12

@pytest.mark.parametrize('top',TOPS[1:])
@pytest.mark.parametrize('cpml',[False,True])
def test_owned_checkpoint_full_segmented_and_dense_fd(mpi_harness,top,cpml):
    exp=experiment(cpml);dl,dm=deterministic_model_direction(exp);data=deterministic_data(exp)
    full,_=run(mpi_harness,exp,dl,dm,data,top,0,label='full')
    segmented,_=run(mpi_harness,exp,dl,dm,data,top,7,label='segmented')
    read(full,exp);read(segmented,exp)
    for name in ['background','j','gl','gm']+[f'strain{k}' for k in range(4)]:assert (full/f'{name}.bin').read_bytes()==(segmented/f'{name}.bin').read_bytes()
    analytic=np.fromfile(full/'j.bin',np.float32).astype(float)
    finite=(np.fromfile(full/'plus.bin',np.float32).astype(float)-np.fromfile(full/'minus.bin',np.float32).astype(float))/.1
    # Canonical M9b-1 epsilon=.05 and ceiling; no fitted scale. The initial
    # epsilon=.01 fixture was independently falsified on the unchanged BASE:
    # MPI nonlinear bytes were identical, but FP32 subtraction failed there too.
    metric=float(np.linalg.norm(finite-analytic)/np.linalg.norm(analytic))
    assert metric<4e-3,metric
    (full/'fd.json').write_text(json.dumps({'relative_l2':metric,'epsilon':.05,'ceiling':.004}))
    diagnostics=json.loads((segmented/'diagnostics.json').read_text())
    for rank in diagnostics['ranks']:
        assert rank['retained']==rank['payload']+rank['metadata']+rank['schedule']+rank['operand']

@pytest.mark.parametrize('top',TOPS)
def test_synthetic_owned_image_gather(mpi_harness,top):
    exp=experiment(False,nt=15);dl,dm=deterministic_model_direction(exp)
    directory,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),top,label='rank-coded')
    actual=np.fromfile(directory/'rank_codes.bin',np.float64).reshape(exp.ny,exp.nx)
    nx,ny=exp.nx//top[0],exp.ny//top[1]
    for py in range(top[1]):
        for px in range(top[0]):
            expected=(py*top[0]+px)*1000000+np.arange(nx*ny).reshape(ny,nx)
            np.testing.assert_array_equal(actual[py*ny:(py+1)*ny,px*nx:(px+1)*nx],expected)

@pytest.mark.parametrize('nt,expected',[ (55,False),(120,True)])
def test_collective_backend_selection_mixed_boundary_and_all_benefit(mpi_harness,nt,expected):
    exp=experiment(True,nx=32,ny=24,nt=nt,fw=3);dl,dm=deterministic_model_direction(exp)
    directory,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(4,1),-32,label='backend-selection')
    read(directory,exp)
    ranks=json.loads((directory/'diagnostics.json').read_text())['ranks']
    benefits=[r['estimate']<r['full'] for r in ranks]
    if not expected:assert any(benefits) and not all(benefits),ranks
    else:assert all(benefits),ranks
    assert all(bool(r['segmented'])==expected for r in ranks)

def test_collective_equal_complete_bytes_selects_full(mpi_harness):
    # 2x2 owned cells, NT=69, S=32: exactly 4416 FULL and replay bytes,
    # including 1488 checkpoint object + 256 schedule + 192 operand bytes.
    exp=experiment(False,nx=6,ny=6,nt=69);dl,dm=deterministic_model_direction(exp)
    directory,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(3,3),-32,label='equal-bytes')
    ranks=json.loads((directory/'diagnostics.json').read_text())['ranks']
    assert all(r['full']==r['estimate']==4416 and not r['segmented'] for r in ranks)

@pytest.mark.parametrize('segments',[1,15,99])
def test_collective_replay_schedule_extremes(mpi_harness,segments):
    exp=experiment(True,nt=15);dl,dm=deterministic_model_direction(exp)
    full,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(2,2),0,label='schedule-full')
    seg,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(2,2),segments,label='schedule-extreme')
    for name in ['background','j','gl','gm']+[f'strain{k}' for k in range(4)]:assert (full/f'{name}.bin').read_bytes()==(seg/f'{name}.bin').read_bytes()

def test_cpml_strip_crosses_internal_interface(mpi_harness):
    exp=experiment(True,nx=32,ny=24,nt=60,fw=9);dl,dm=deterministic_model_direction(exp)
    directory,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(4,1),5,label='crossing-cpml')
    read(directory,exp)

@pytest.mark.parametrize('source',[(3,3),(13,3),(3,11),(12,10),(13,11)])
def test_unique_source_owner_at_all_interfaces(mpi_harness,source):
    exp=replace(experiment(True,nt=40),sources=(source,));dl,dm=deterministic_model_direction(exp)
    directory,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(2,2),3,label='source-owner')
    read(directory,exp)

@pytest.mark.parametrize('location',[(9,11),(9,12),(10,11),(10,12),(0,0),(19,23)])
def test_harmonic_mu_transpose_interfaces_corner_wrap(mpi_harness,location):
    exp=experiment(True);dl=np.zeros_like(exp.lam);dm=np.zeros_like(exp.mu);dm[location]=exp.mu[location]*.02
    directory,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(2,2),7,label='mu-interface')
    read(directory,exp)

@pytest.mark.parametrize('fault',['topology','divisibility','source','receiver','model','free_surface','boundary','size','minimum'])
def test_rank_local_invalid_configuration_fails_collectively(mpi_harness,fault):
    exp=experiment(False,nt=15);dl,dm=deterministic_model_direction(exp)
    directory,p=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(2,2),env={'M9D2_INVALID':fault,'M9D2_FAIL_PHASE':'create','M9D2_FAIL_RANK':'-1'},label='invalid')
    assert 'M9D2_EXPECTED_COLLECTIVE_FAILURE' in p.stdout

@pytest.mark.parametrize('phase,index', [('create',1),('create',2),('create',7),('create',13),('prepare',1),('prepare',4),('prepare',10),('j',1),('j',4),('j',18),('jt',1),('jt',5),('jt',16)])
def test_rank_local_allocation_failure_has_no_deadlock(mpi_harness,phase,index):
    exp=experiment(True,nt=30);dl,dm=deterministic_model_direction(exp)
    directory,p=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(2,1),5,env={'M9D2_FAIL_PHASE':phase,'M9D2_FAIL_AT':str(index)},label='allocation-fault')
    assert 'M9D2_EXPECTED_COLLECTIVE_FAILURE' in p.stdout

@pytest.mark.parametrize('top',[(2,1),(2,2)])
def test_twenty_context_reuse_sweeps_are_deterministic(mpi_harness,top):
    exp=experiment(True,nt=30);dl,dm=deterministic_model_direction(exp)
    directory,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),top,5,env={'M9D2_REPEAT':'20'},label='repeatability')
    read(directory,exp)

def test_exact_local_minimum_two(mpi_harness):
    exp=experiment(False,nx=6,ny=6,nt=20);dl,dm=deterministic_model_direction(exp)
    directory,_=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(3,1),3,label='minimum-two')
    read(directory,exp)

def test_local_minimum_one_rejected(mpi_harness):
    exp=experiment(False,nx=6,ny=6,nt=20);dl,dm=deterministic_model_direction(exp)
    directory,p=run(mpi_harness,exp,dl,dm,deterministic_data(exp),(6,1),env={'M9D2_FAIL_PHASE':'create','M9D2_FAIL_RANK':'-1'},label='below-minimum')
    assert 'M9D2_EXPECTED_COLLECTIVE_FAILURE' in p.stdout

@pytest.mark.integration
@pytest.mark.parametrize('top',TOPS)
@pytest.mark.parametrize('cpml',[False,True])
def test_true_mode2_multi_shot_mpi_matches_canonical(tmp_path,denise_binary,top,cpml):
    exp=replace(experiment(cpml,nt=80),sources=((12,10),(13,11)))
    data=[deterministic_data(exp,seed=17),deterministic_data(exp,seed=31)]
    paths=mode2._write_mode2_case(tmp_path,exp,data)
    serial=subprocess.run(['mpiexec','--oversubscribe','-n','1',str(denise_binary),'denise.inp','workflow.inp'],cwd=tmp_path,capture_output=True,text=True,timeout=120)
    assert serial.returncode==0,serial.stdout+serial.stderr
    reference=[paths[key].read_bytes() for key in ['lambda','mu']]
    text=(tmp_path/'denise.inp').read_text().replace('NPROCX =1',f'NPROCX ={top[0]}').replace('NPROCY =1',f'NPROCY ={top[1]}')
    (tmp_path/'denise.inp').write_text(text)
    completed=subprocess.run(['mpiexec','--oversubscribe','-n',str(top[0]*top[1]),str(denise_binary),'denise.inp','workflow.inp'],cwd=tmp_path,capture_output=True,text=True,timeout=120)
    (tmp_path/'mode2.log').write_text(completed.stdout+completed.stderr)
    assert completed.returncode==0,completed.stdout+completed.stderr
    metrics=[]
    for key,ref in zip(['lambda','mu'],reference):
        metric=errors(np.frombuffer(paths[key].read_bytes(),np.float64).reshape(exp.ny,exp.nx),np.frombuffer(ref,np.float64).reshape(exp.ny,exp.nx));metrics.append(metric)
        if top==(1,1):assert metric['bitwise']
        else:assert metric['relative_l2']<=1e-12,metric
    (tmp_path/'mode2-equivalence.json').write_text(json.dumps(metrics,indent=2))

@pytest.mark.integration
@pytest.mark.parametrize('fault',['missing-vx','truncated-model','nonfinite','output','divisibility','shot-groups','free_surface','zero-processors'])
def test_true_mode2_collective_failures_remove_pair(tmp_path,denise_binary,fault):
    exp=experiment(True,nt=40);data=[deterministic_data(exp)]
    paths=mode2._write_mode2_case(tmp_path,exp,data)
    text=(tmp_path/'denise.inp').read_text().replace('NPROCX =1','NPROCX =2')
    if fault=='missing-vx':paths['vx1'].unlink()
    elif fault=='truncated-model':(tmp_path/'model/background.mu').write_bytes(b'bad')
    elif fault=='nonfinite':paths['vx1'].write_bytes(np.full(exp.nt*len(exp.receivers),np.nan,np.float32).tobytes())
    elif fault=='output':
        blocked=tmp_path/'image/m9c.image_mu_raw.bin.tmp';blocked.mkdir();(blocked/'blocker').write_text('keep directory nonempty')
    elif fault=='divisibility':text=text.replace('NX =24','NX =23')
    elif fault=='shot-groups':text=text.replace('NPROCX =2','NPROCX =1')
    elif fault=='free_surface':text=text.replace('FREE_SURF =0','FREE_SURF =1')
    elif fault=='zero-processors':text=text.replace('NPROCX =2','NPROCX =0')
    (tmp_path/'denise.inp').write_text(text)
    for key in ['lambda','mu']:paths[key].write_bytes(b'stale image')
    completed=subprocess.run(['mpiexec','--oversubscribe','-n','2',str(denise_binary),'denise.inp','workflow.inp'],cwd=tmp_path,capture_output=True,text=True,timeout=90)
    (tmp_path/'failure.log').write_text(completed.stdout+completed.stderr)
    assert completed.returncode!=0,completed.stdout+completed.stderr
    assert not paths['lambda'].exists() and not paths['mu'].exists()

@pytest.mark.integration
@pytest.mark.parametrize('ranks',[1,2,4])
def test_mode2_missing_source_file_clears_pair(tmp_path,denise_binary,ranks):
    exp=experiment(True,nt=30);paths=mode2._write_mode2_case(tmp_path,exp,[deterministic_data(exp)])
    (tmp_path/'source.dat').unlink()
    for name in ['lambda','mu']:paths[name].write_bytes(b'stale image')
    text=(tmp_path/'denise.inp').read_text()
    if ranks>=2:text=text.replace('NPROCX =1','NPROCX =2')
    if ranks==4:text=text.replace('NPROCY =1','NPROCY =2')
    (tmp_path/'denise.inp').write_text(text)
    p=subprocess.run(['mpiexec','--oversubscribe','-n',str(ranks),str(denise_binary),'denise.inp','workflow.inp'],cwd=tmp_path,capture_output=True,text=True,timeout=90)
    (tmp_path/'missing-source.log').write_text(p.stdout+p.stderr)
    assert p.returncode!=0,p.stdout+p.stderr
    assert not paths['lambda'].exists() and not paths['mu'].exists()
