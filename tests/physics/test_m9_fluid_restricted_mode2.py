"""Bounded water-layer MODE=2 integration and published CUDA capability gate."""
import ctypes as C
from dataclasses import replace
import subprocess
import numpy as np
import pytest
from tests.physics import test_m9c_elastic_psv_migration_driver as driver
from tests.physics.test_m9c_elastic_psv_migration_driver import migration_library
from tests.physics.test_m9e2_cuda_free_surface_born import backends
from tests.physics.test_m9e3_cuda_jt_migration import jt
from tests.physics.test_m9e4_cuda_replay_mode2 import replay
from tests.physics import test_m9e1_cuda_elastic_psv_forward as e1
from tests.utilities import zero_shear_reference as z
from tests.utilities import m9_fluid_restricted_reference as ref


def water_case(fs,fw):
    exp=ref.fixture("horizontal",fs,fw)
    m=z.material(exp.rho,exp.lam,exp.mu)
    dl,dm=(a.astype(np.float32).astype(float) for a in z.directions(m)["interface"])
    source=ref.b._source_samples(exp).astype(np.float32).astype(float)
    # Frozen prepared direct migration data: independent restricted Born shot.
    dummy=np.zeros((exp.nt,len(exp.receivers),2))
    data=ref.products(exp,source,dl,dm,dummy,fs)[0].astype(np.float32)
    return exp,data


def independent_image(exp,data,fs):
    source=ref.b._source_samples(exp).astype(np.float32).astype(float)
    zero=np.zeros_like(exp.mu)
    return np.stack(ref.products(exp,source,zero,zero,data.astype(float),fs)[1:3])


@pytest.mark.parametrize("fs,fw",[(0,0),(1,3)])
def test_fluid_driver_raw_images_and_transaction_recovery(migration_library,fs,fw):
    exp,data=water_case(fs,fw);owner=driver.RequestOwner(exp,[data])
    owner.request.free_surface=fs
    images,diagnostics=driver._run(migration_library,owner)
    expected=independent_image(exp,data,fs)
    actual=np.stack(images)
    error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    assert error<=6e-5,error
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[1][exp.mu==0],0.)
    assert not np.signbit(actual[1][exp.mu==0]).any()
    assert np.linalg.norm(actual[1][exp.mu>0])>0
    assert np.linalg.norm(actual[0][(exp.mu==0)&(np.indices(exp.mu.shape)[0]>0)])>0
    if fs:np.testing.assert_array_equal(actual[0][0],0.)
    owner.model[1][0,0]=-1.
    result=driver.MigrationResult()
    assert migration_library.denise_elastic_psv_migrate(C.byref(owner.request),C.byref(result))!=0
    assert not result.image_lambda_raw and not result.image_mu_raw
    owner.model[1][0,0]=0.
    recovered,_=driver._run(migration_library,owner)
    assert np.stack(recovered).tobytes()==actual.tobytes()
    ref.record("migration_api",fs=fs,fw=fw,reference_relative=error,
               water_gmu_positive_zero=True,solid_gmu_norm=float(np.linalg.norm(actual[1][exp.mu>0])),
               fluid_glambda_norm=float(np.linalg.norm(actual[0][exp.mu==0])),diagnostics=diagnostics,recovery=True)


@pytest.mark.integration
@pytest.mark.parametrize("top,fs,fw",[((1,1),0,0),((2,1),0,3),((1,2),1,0),((2,2),1,3)])
def test_water_layer_mode2_publishes_raw_pair(tmp_path,migration_library,denise_binary,mpiexec,top,fs,fw):
    exp,data=water_case(fs,fw)
    owner=driver.RequestOwner(exp,[data]);owner.request.free_surface=fs
    expected=np.stack(driver._run(migration_library,owner)[0])
    paths=driver._write_mode2_case(tmp_path,exp,[data])
    inp=tmp_path/"denise.inp"
    def topology():
        text=inp.read_text().replace("NPROCX =1",f"NPROCX ={top[0]}").replace("NPROCY =1",f"NPROCY ={top[1]}")
        inp.write_text(text.replace("FREE_SURF =0",f"FREE_SURF ={fs}"))
    def run():
        p=subprocess.run([mpiexec,"--oversubscribe","-n",str(top[0]*top[1]),str(denise_binary),"denise.inp","workflow.inp"],
            cwd=tmp_path,capture_output=True,text=True,timeout=120)
        return p,p.stdout+p.stderr
    topology();p,log=run()
    (tmp_path/"valid.log").write_text(log)
    assert p.returncode==0,log
    assert "FLUID-2" not in log
    actual=np.stack([np.fromfile(paths[n],np.float64).reshape(exp.mu.shape) for n in ("lambda","mu")])
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[1][exp.mu==0],0.)
    assert not np.signbit(actual[1][exp.mu==0]).any()
    assert np.linalg.norm(actual[1][exp.mu>0])>0
    error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    assert error<=1e-12,error
    # Invalid material removes an already published pair; same files recover.
    bad=exp.mu.copy();bad[0,0]=-1.
    bad.T.astype(np.float32).tofile(tmp_path/"model/background.mu")
    q,failed_log=run()
    (tmp_path/"invalid.log").write_text(failed_log)
    assert q.returncode!=0,failed_log
    assert not paths["lambda"].exists() and not paths["mu"].exists()
    exp.mu.T.astype(np.float32).tofile(tmp_path/"model/background.mu")
    q,recovery_log=run()
    assert q.returncode==0,recovery_log
    for k,n in enumerate(("lambda","mu")):
        assert np.fromfile(paths[n],np.float64).reshape(exp.mu.shape).tobytes()==actual[k].tobytes()
    ref.record("water_mode2",top=list(top),fs=fs,fw=fw,serial_relative=error,
               finite=True,raw_water_gmu_positive_zero=True,solid_gmu_active=True,
               invalid_pair_removed=True,recovered_bitwise=True)


@pytest.mark.parametrize("fs",[0,1])
def test_cuda_fluid_migration_request_fails_before_images(replay,fs):
    _,libs=replay;exp=ref.fixture("horizontal",fs)
    data=np.ones((exp.nt,len(exp.receivers),2),np.float32)
    owner=driver.RequestOwner(exp,[data]);owner.request.free_surface=fs
    for mode,lib in libs.items():
        result=driver.MigrationResult()
        rc=lib.denise_cuda_m9_migrate_request(C.byref(owner.request),C.byref(result))
        assert rc!=0
        assert b"FLUID-4" in lib.denise_cuda_m9_migration_last_error()
        assert not result.image_lambda_raw and not result.image_mu_raw
        assert e1.ledger(lib)[:3]==(0,0,0)
        lib.denise_cuda_m9_migration_result_destroy(C.byref(result))
        ref.record("cuda_boundary",fs=fs,mode=mode,FLUID4=True,no_images=True,no_owned_resources=True)
