"""Historical CPU fluid fixtures with restricted J/JT and CUDA boundaries."""
import ctypes as C
from dataclasses import replace
import os
import subprocess
import numpy as np
import pytest
from tests.physics import test_m9_fluid_cpu_forward as serial
from tests.physics.test_m9_fluid_cpu_forward import fluid_library
from tests.physics import test_m9c_elastic_psv_migration_driver as driver
from tests.physics.test_m9c_elastic_psv_migration_driver import migration_library
from tests.physics import test_m9d2_elastic_psv_mpi as cpu
from tests.physics import test_m9e1_cuda_elastic_psv_forward as e1
from tests.physics import test_m9e2_cuda_free_surface_born as e2
from tests.physics.test_m9e4_cuda_replay_mode2 import replay
from tests.physics.test_m9e3_cuda_jt_migration import jt
from tests.physics.test_m9e2_cuda_free_surface_born import backends
from tests.utilities import zero_shear_reference as frozen

@pytest.fixture(scope='session')
def fluid_mpi(tmp_path_factory,repository_root):
    out=tmp_path_factory.mktemp('fluid-mpi')/'operator'
    subprocess.run(['mpicc','-std=c99','-O2','-Wall','-Wextra','-Werror','-pedantic',
     '-I'+str(repository_root/'include'),str(repository_root/'tests/utilities/m9_fluid_cpu_mpi.c'),'-lm','-o',str(out)],check=True)
    return out
def experiment(pattern='horizontal',nt=120):
    exp=cpu.experiment(False,nt=nt)
    rho=np.ones((exp.ny,exp.nx));lam=np.full_like(rho,4);mu=np.full_like(rho,3.375)
    if pattern=='homogeneous':mu[:]=0
    elif pattern=='horizontal':mu[:exp.ny//2+1]=0;rho[exp.ny//2+1:]=1.5;lam[exp.ny//2+1:]=6.75
    elif pattern=='vertical':mu[:,:exp.nx//2+1]=0
    elif pattern=='one_rank':mu[1,1]=0
    else:raise AssertionError(pattern)
    return replace(exp,dh=1,dt=.1,lam=lam,mu=mu,rho=rho,
      sources=((exp.nx//2+1,exp.ny//2+1),),
      receivers=((exp.nx//2+4,exp.ny//2+1),(exp.nx//2+1,exp.ny//2+4),(exp.nx//2-1,exp.ny//2-1),(2,2)),
      pml_fpml=.1,pml_power=2,pml_kmax=1,pml_reflection=1e-3)
def write(path,exp):
    nx,ny=exp.nx,exp.ny
    with path.open('wb') as f:
        np.array([nx,ny,exp.nt,exp.fw,nx//2,ny//2,len(exp.receivers)],np.int32).tofile(f)
        np.array([1,.1,np.sqrt(np.max((exp.lam+2*exp.mu)/exp.rho)),1e-3,2,1,.1],np.float32).tofile(f)
        for a in (exp.lam,exp.mu,exp.rho,np.zeros_like(exp.mu),np.zeros_like(exp.mu)):np.asarray(a,np.float32).tofile(f)
        frozen.prepared_source(exp.nt,.1,fc=.5,t0=3).astype(np.float32).tofile(f)
        np.array([i-1 for i,j in exp.receivers],np.int32).tofile(f)
        np.array([j-1 for i,j in exp.receivers],np.int32).tofile(f)
def run(exe,folder,exp,top,segments,fs=0,negative=False):
    folder.mkdir();write(folder/'input.bin',exp)
    env={k:v for k,v in os.environ.items() if not k.startswith(('OMPI_','PMIX_','PMI_','OPAL_'))}
    env['FLUID_FS']=str(fs)
    if negative:env['FLUID_NEGATIVE']='1'
    p=subprocess.run(['mpiexec','--oversubscribe','-n',str(top[0]*top[1]),str(exe),str(folder/'input.bin'),str(folder),
      str(top[0]),str(top[1]),str(segments)],env=env,capture_output=True,text=True,timeout=120)
    (folder/'run.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
    return p.stdout
@pytest.mark.parametrize('top',[(2,1),(1,2),(2,2)])
@pytest.mark.parametrize('pattern',['horizontal','vertical','one_rank'])
@pytest.mark.parametrize('fs',[0,1])
def test_actual_mpi_maps_forward_and_global_fluid_boundary(fluid_mpi,fluid_library,tmp_path,top,pattern,fs):
    compare_mpi(fluid_mpi,fluid_library,tmp_path,top,experiment(pattern),fs,pattern)
def compare_mpi(fluid_mpi,fluid_library,tmp_path,top,exp,fs,pattern):
    c=serial.Context(fluid_library,mu=exp.mu,lam=exp.lam,rho=exp.rho,fs=fs,fw=exp.fw)
    try:
        expected=c.prepare();maps=c.maps();reference=frozen.elastic_forward(frozen.material(exp.rho,exp.lam,exp.mu),free_surface=fs,fw=exp.fw)
        for segments in (0,3):
            folder=tmp_path/str(segments)
            assert 'FLUID_MPI_FORWARD_MAPS_BOUNDARY_RECOVERY_PASS' in run(fluid_mpi,folder,exp,top,segments,fs)
            got=np.fromfile(folder/'data.bin',np.float32).reshape(c.data_shape)
            assert got.tobytes()==expected.tobytes()
            error=np.linalg.norm(got-reference['data'])/np.linalg.norm(reference['data']);assert error<=1e-5
            for k,name in enumerate(('rx','ry','corner')):
                actual=np.fromfile(folder/(name+'.bin'),np.float64).reshape(c.shape)
                np.testing.assert_array_equal(actual,maps[k])
                if k==2:assert not np.signbit(actual).any()
        serial.RECORDS.append({'mpi_forward':[list(top),pattern,fs],'serial_bitwise':True,'independent_relative_l2':float(error),'restricted_J_JT_and_invalid_dMu_transaction':True})
    finally:c.close()
@pytest.mark.parametrize('pattern',['horizontal','vertical'])
@pytest.mark.parametrize('fs',[0,1])
def test_actual_mpi_fluid_cpml_interfaces(fluid_mpi,fluid_library,tmp_path,pattern,fs):
    exp=replace(experiment(pattern),fw=3,cpml=True)
    compare_mpi(fluid_mpi,fluid_library,tmp_path,(2,2),exp,fs,pattern+'/CPML')
@pytest.mark.parametrize('top',[(2,1),(1,2),(2,2)])
def test_negative_mu_one_rank_collective_rejection(fluid_mpi,tmp_path,top):
    assert 'FLUID_MPI_NEGATIVE_COLLECTIVE_PASS' in run(fluid_mpi,tmp_path/'bad',experiment(),top,0,negative=True)
@pytest.mark.parametrize('ranks',[1,2])
@pytest.mark.parametrize('fs',[0,1])
@pytest.mark.integration
def test_mode2_fluid_cfl_valid_and_restricted_migration_output(tmp_path,denise_binary,mpiexec,ranks,fs):
    exp=experiment('homogeneous',nt=20);data=[np.ones((exp.nt,len(exp.receivers),2),np.float32)]
    paths=driver._write_mode2_case(tmp_path,exp,data)
    inp=tmp_path/'denise.inp';inp.write_text(inp.read_text().replace('NPROCX =1',f'NPROCX ={ranks}').replace('FREE_SURF =0',f'FREE_SURF ={fs}'))
    p=subprocess.run([mpiexec,'-n',str(ranks),str(denise_binary),'denise.inp','workflow.inp'],cwd=tmp_path,
      capture_output=True,text=True,timeout=90)
    assert p.returncode==0 and 'FLUID-2' not in p.stdout+p.stderr,p.stdout+p.stderr
    assert 'invalid cell' not in p.stdout+p.stderr and 'CFL check failed' not in p.stdout+p.stderr
    assert paths['lambda'].exists() and paths['mu'].exists()
    images=np.stack([np.fromfile(paths[name],np.float64).reshape(exp.mu.shape) for name in ('lambda','mu')])
    assert np.isfinite(images).all()
    np.testing.assert_array_equal(images[1],0.)
    assert not np.signbit(images[1]).any()
    # Same request files, return to valid solid model: no stale publication.
    solid=replace(exp,mu=np.ones_like(exp.mu))
    driver._write_mode2_case(tmp_path,solid,data)
    inp.write_text(inp.read_text().replace('NPROCX =1',f'NPROCX ={ranks}').replace('FREE_SURF =0',f'FREE_SURF ={fs}'))
    q=subprocess.run([mpiexec,'-n',str(ranks),str(denise_binary),'denise.inp','workflow.inp'],cwd=tmp_path,capture_output=True,text=True,timeout=90)
    assert q.returncode==0,q.stdout+q.stderr
    assert paths['lambda'].exists() and paths['mu'].exists()
def test_fluid_migration_request_returns_restricted_images(migration_library):
    exp=experiment('homogeneous',nt=20);owner=driver.RequestOwner(exp,[np.ones((exp.nt,len(exp.receivers),2),np.float32)])
    result=driver.MigrationResult()
    try:
        assert migration_library.denise_elastic_psv_migrate(C.byref(owner.request),C.byref(result))==0
        assert b'FLUID-2' not in migration_library.denise_elastic_psv_migration_last_error()
        assert result.image_lambda_raw and result.image_mu_raw
        images=np.stack([np.ctypeslib.as_array(a,shape=(result.cell_count,)).copy().reshape(exp.mu.shape)
            for a in (result.image_lambda_raw,result.image_mu_raw)])
        assert np.isfinite(images).all()
        np.testing.assert_array_equal(images[1],0.)
        assert not np.signbit(images[1]).any()
    finally:migration_library.denise_elastic_psv_migration_result_destroy(C.byref(result))
    assert not result.image_lambda_raw and not result.image_mu_raw
    owner.model[1][0,0]=-1.
    assert migration_library.denise_elastic_psv_migrate(C.byref(owner.request),C.byref(result))!=0
    assert not result.image_lambda_raw and not result.image_mu_raw
    owner.model[1][0,0]=0.
    recovered,_=driver._run(migration_library,owner)
    assert np.stack(recovered).tobytes()==images.tobytes()
    owner.model[1].fill(1)
    images,_=driver._run(migration_library,owner);assert all(np.isfinite(a).all() for a in images)
@pytest.mark.parametrize('name',['forward','full','migration','replay'])
def test_published_cuda_fluid_constructor_is_fail_closed(replay,name):
    _,libs=replay;cfg,a=e2.fixture(0,nt=13);a['m'].fill(0)
    for lib in libs.values():
        c=e1.P()
        if name=='replay':rc=lib.denise_cuda_m9_create_replay(C.byref(cfg),C.byref(e1.Options()),3,0,C.byref(c))
        else:rc=getattr(lib,'denise_cuda_m9_create' if name=='forward' else 'denise_cuda_m9_create_'+name)(C.byref(cfg),C.byref(e1.Options()),C.byref(c))
        assert rc!=0 and not c.value
        assert b'FLUID-4' in lib.denise_cuda_m9_last_error()
        # Fourth ledger entry is operation count, not owned memory/resources.
        assert e1.ledger(lib)[:3]==(0,0,0)
@pytest.mark.parametrize('fs',[0,1])
def test_published_cuda_fluid_trial_preserves_solid_prepared_state(jt,fs):
    _,libs=jt;cfg,a=e2.fixture(fs,nt=13)
    for lib in libs.values():
        c=e1.P();e1.check(lib,lib.denise_cuda_m9_create_full(C.byref(cfg),C.byref(e1.Options()),C.byref(c)))
        try:
            e1.check(lib,lib.denise_cuda_m9_prepare(c));before=e1.download(lib,c,cfg)
            mu=a['m'].copy();mu[0,0]=0
            assert lib.denise_cuda_m9_nonlinear(c,e1.fp(a['l']),e1.fp(mu))!=0
            assert b'FLUID-4' in lib.denise_cuda_m9_last_error()
            after=e1.download(lib,c,cfg)
            for x,y in zip(before,after):assert x.tobytes()==y.tobytes()
            e1.check(lib,lib.denise_cuda_m9_prepare(c))
        finally:e1.check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
