"""Actual CPU FLUID-1B products against immutable FLUID-1A authority."""
import ctypes as C
import itertools
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
from tests.physics import test_m9b1_elastic_psv_born_production as old
from tests.physics import test_m9d1_elastic_psv_checkpoint_replay as replay
from tests.physics.test_m9c_elastic_psv_migration_driver import migration_library
from tests.physics import test_m9c_elastic_psv_migration_driver as driver
from tests.utilities import zero_shear_reference as frozen

F=old.F32P;D=old.F64P;P=C.c_void_p
RECORDS=[]
def fp(a): return a.ctypes.data_as(F)
def dp(a): return a.ctypes.data_as(D)
def library(path):
    a=C.CDLL(str(path));a.denise_elastic_psv_born_last_error.restype=C.c_char_p
    signatures={'create':[C.POINTER(old.Config),C.POINTER(P)],'destroy':[C.POINTER(P)],
     'prepare':[P,F],'apply_j':[P,F,F,F],'apply_jt':[P,F,D,D],'nonlinear':[P,F,F,F],
     'copy_strain':[P,C.c_int,F],'set_replay_segments':[P,C.c_int],
     'checkpoint_roundtrip':[P,C.c_int],'storage_diagnostics':[P,C.POINTER(replay.StorageDiagnostics)]}
    for name,args in signatures.items():getattr(a,'denise_elastic_psv_born_'+name).argtypes=args
    a.fluid_copy_maps.argtypes=[P,F];a.fluid_products.argtypes=[P,F,F,F,D]
    return a
@pytest.fixture(scope='session')
def fluid_library(tmp_path_factory,repository_root):
    path=tmp_path_factory.mktemp('fluid-products')/'cpu.so'
    subprocess.run(['cc','-std=c99','-O2','-Wall','-Wextra','-Werror','-pedantic','-fPIC','-shared',
      '-I'+str(repository_root/'include'),str(repository_root/'tests/utilities/m9_fluid_cpu_products.c'),
      '-lm','-o',str(path)],check=True)
    yield library(path)
    if os.environ.get('DENISE_FLUID_EVIDENCE'):Path(os.environ['DENISE_FLUID_EVIDENCE']).write_text(json.dumps(RECORDS,indent=2))

class Context:
    def __init__(self,api,mu=None,lam=None,rho=None,nt=120,fs=0,fw=0,segments=0):
        self.api=api
        shape=mu.shape if mu is not None else (20,24)
        ny,nx=shape; self.mu=np.array(np.zeros(shape) if mu is None else mu,np.float32,copy=True)
        self.lam=np.array(np.full(shape,4) if lam is None else lam,np.float32,copy=True)
        self.rho=np.array(np.ones(shape) if rho is None else rho,np.float32,copy=True)
        self.source=frozen.prepared_source(nt,.1,fc=.5 if nt==120 else .1,t0=3 if nt==120 else 15).astype(np.float32)
        self.ri=np.array([nx//2+3,nx//2,nx//2-2,1],np.int32)
        self.rj=np.array([ny//2,ny//2+3,ny//2-2,1],np.int32)
        self.cfg=old.Config(nx=nx,ny=ny,nt=nt,fw=fw,dh=1,dt=.1,invmat1=3,fdorder=4,ndt=1,dtinv=1,
         free_surface=fs,mpi_size=1,receiver_components=2,lambda_=fp(self.lam),mu=fp(self.mu),rho=fp(self.rho),
         source_i=nx//2,source_j=20 if ny==64 else ny//2,source_samples=fp(self.source),receiver_count=4,
         receiver_i=old._ptr(self.ri,old.I32P),receiver_j=old._ptr(self.rj,old.I32P),
         cpml_enabled=bool(fw),pml_reflection=1e-3,pml_power=2,pml_kmax=1,pml_fpml=.1,
         pml_damping_speed=float(np.sqrt(np.max((self.lam+2*self.mu)/self.rho))))
        self.c=P();self.check(api.denise_elastic_psv_born_create(C.byref(self.cfg),C.byref(self.c)))
        self.shape=shape;self.data_shape=(nt,4,2)
        if segments:self.check(api.denise_elastic_psv_born_set_replay_segments(self.c,segments))
    def check(self,rc):
        assert rc==0,self.api.denise_elastic_psv_born_last_error().decode()
    def prepare(self):
        d=np.empty(self.data_shape,np.float32);self.check(self.api.denise_elastic_psv_born_prepare(self.c,fp(d)));return d
    def maps(self):
        m=np.empty((3,*self.shape),np.float32);self.check(self.api.fluid_copy_maps(self.c,fp(m)));return m
    def close(self):self.api.denise_elastic_psv_born_destroy(C.byref(self.c))
    def snapshot(self):
        d=replay.StorageDiagnostics();self.check(self.api.denise_elastic_psv_born_storage_diagnostics(self.c,C.byref(d)))
        return bytes(d)
    def products(self):
        d=np.empty(self.data_shape,np.float32);s=np.empty((13,*self.shape),np.float32);q=np.empty((4,*self.shape),np.float32);r=C.c_double()
        self.check(self.api.fluid_products(self.c,fp(d),fp(s),fp(q),C.byref(r)));return d,s,q,r.value

@pytest.mark.parametrize('mask',list(itertools.product([0,1],repeat=4)))
def test_production_corner_all_occupancy_patterns(fluid_library,mask):
    mu=np.full((8,10),3.0);mu[3:5,4:6]=np.array([2,3,5,7]).reshape(2,2)*(1-np.array(mask).reshape(2,2))
    fluid_library.fluid_clear_arithmetic();c=Context(fluid_library,mu=mu,nt=9)
    try:
        actual=c.maps();ref=frozen.material(c.rho,c.lam,c.mu)
        np.testing.assert_array_equal(actual,np.stack((ref.rx,ref.ry,ref.corner)).astype(np.float32))
        assert not np.signbit(actual[2]).any()
        c.prepare();assert fluid_library.fluid_singular_arithmetic()==0
    finally:c.close()
@pytest.mark.parametrize('pattern',['horizontal','vertical','isolated','strip','pocket','edges','solid'])
def test_wrapped_production_material_maps(fluid_library,pattern):
    from tests.physics.test_zero_shear_oracle import fixture
    m=fixture(pattern=pattern);c=Context(fluid_library,mu=m.mu,lam=m.lam,rho=m.rho,nt=9)
    try:
        np.testing.assert_array_equal(c.maps(),np.stack((m.rx,m.ry,m.corner)).astype(np.float32))
    finally:c.close()
@pytest.mark.parametrize('segments',[0,1,3,7])
def test_actual_homogeneous_acoustic_forward_and_fail_closed(fluid_library,segments):
    c=Context(fluid_library,segments=segments)
    try:
        got=c.prepare();data,state,q,_=c.products();assert got.tobytes()==data.tobytes()
        frozen.require_homogeneous_products(frozen.material(c.rho,c.lam,c.mu),
            got,c.maps()[2],state[4],frozen.acoustic_forward()[0])
        assert np.isfinite(state).all() and np.isfinite(q).all() and np.linalg.norm(q)>0
        np.testing.assert_array_equal(state[2],state[3])
        assert np.linalg.norm(q[1])+np.linalg.norm(q[2])>0
        reference=frozen.acoustic_forward()[0];error=np.linalg.norm(got-reference)/np.linalg.norm(reference)
        assert error<=1e-5
        c.check(fluid_library.denise_elastic_psv_born_checkpoint_roundtrip(c.c,43))
        before=c.snapshot();out=np.full(c.data_shape,37,np.float32);image=np.full((2,*c.shape),41.,np.float64);direction=np.zeros(c.shape,np.float32)
        assert fluid_library.denise_elastic_psv_born_apply_j(c.c,fp(direction),fp(direction),fp(out))!=0
        assert b'FLUID-2' in fluid_library.denise_elastic_psv_born_last_error()
        assert fluid_library.denise_elastic_psv_born_apply_jt(c.c,fp(got),dp(image[0]),dp(image[1]))!=0
        assert b'FLUID-2' in fluid_library.denise_elastic_psv_born_last_error()
        assert np.all(out==37) and np.all(image==41) and c.snapshot()==before
        assert c.prepare().tobytes()==got.tobytes()
        RECORDS.append({'homogeneous_segments':segments,'data_relative_l2':float(error),'strain_norm':float(np.linalg.norm(q)),'sxy_zero':True,'J_JT_no_mutation':True})
    finally:c.close()
@pytest.mark.parametrize('fs',[0,1])
@pytest.mark.parametrize('interface',[False,True])
def test_actual_fluid_cpml_and_surface_products(fluid_library,fs,interface):
    rho=np.ones((64,64));lam=np.full((64,64),4.);mu=np.zeros((64,64))
    if interface:rho[32:]=1.5;lam[32:]=6.75;mu[32:]=3.375
    c=Context(fluid_library,mu=mu,lam=lam,rho=rho,nt=800,fs=fs,fw=10)
    try:
        actual=c.prepare();data,s,q,late=c.products();assert actual.tobytes()==data.tobytes()
        ref=frozen.elastic_forward(frozen.material(rho,lam,mu),nt=800,fw=10,free_surface=fs)
        error=np.linalg.norm(actual-ref['data'])/np.linalg.norm(ref['data'])
        assert error<=1e-5 and late<=.05 and np.isfinite(s).all() and np.isfinite(q).all()
        assert np.max(abs(s[5:]))>0 and np.linalg.norm(q[:,mu==0])>0
        assert np.all(s[4][c.maps()[2]==0]==0)
        if fs:
            assert np.all(s[3,0]==0) and np.all(s[2,0]==0)
            assert np.all(s[np.array([6,8,11,12]),:10]==0)
        RECORDS.append({'cpml_surface':[fs,interface],'data_relative_l2':float(error),'late_norm_ratio':late,'active_memory':float(np.max(abs(s[5:])))})
    finally:c.close()
@pytest.mark.parametrize('change',['fluid_to_solid','solid_to_fluid','negative','invalid_fluid_lambda'])
def test_nonlinear_classification_before_output_and_recovery(fluid_library,change):
    mu=np.full((20,24),3.375);mu[:10]=0;c=Context(fluid_library,mu=mu,segments=3)
    try:
        bg=c.prepare();lam=c.lam.copy()*1.01;trial=c.mu.copy();out=np.full(c.data_shape,53,np.float32)
        c.check(fluid_library.denise_elastic_psv_born_nonlinear(c.c,fp(lam),fp(trial),fp(out)));want=out.copy()
        before=c.snapshot();badlam=lam.copy();badmu=trial.copy()
        if change=='fluid_to_solid':badmu[0,0]=1
        elif change=='solid_to_fluid':badmu[15,0]=0
        elif change=='negative':badmu[15,0]=-1
        else:badlam[0,0]=0
        out.fill(53);assert fluid_library.denise_elastic_psv_born_nonlinear(c.c,fp(badlam),fp(badmu),fp(out))!=0
        assert np.all(out==53) and c.snapshot()==before
        c.check(fluid_library.denise_elastic_psv_born_nonlinear(c.c,fp(lam),fp(trial),fp(out)))
        assert out.tobytes()==want.tobytes() and c.prepare().tobytes()==bg.tobytes()
    finally:c.close()
@pytest.mark.parametrize('rho,lam,mu',[(1,4,-1),(0,4,0),(1,0,0),(1,-1,0),(np.inf,4,0),(1,np.inf,0)])
def test_invalid_fluid_material_rejected(fluid_library,rho,lam,mu):
    with pytest.raises(AssertionError):
        Context(fluid_library,mu=np.full((20,24),mu),lam=np.full((20,24),lam),rho=np.full((20,24),rho),nt=9)
def test_copied_fp32_classification_no_epsilon_or_caller_mask(fluid_library):
    mu=np.ones((20,24));mu[0,0]=0;mu[1,1]=np.float32(1e-40)
    c=Context(fluid_library,mu=mu,nt=9)
    try:
        expected=c.maps().copy();c.mu[0,0]=1 # External mutation cannot alter context classification.
        c.prepare();np.testing.assert_array_equal(c.maps(),expected)
        out=np.full(c.data_shape,19,np.float32);dm=np.zeros(c.shape,np.float32)
        assert fluid_library.denise_elastic_psv_born_apply_j(c.c,fp(dm),fp(dm),fp(out))!=0 and np.all(out==19)
        assert c.maps()[2,1,1]>0
    finally:c.close()
def test_preserve_negative_lambda_solid_envelope(fluid_library):
    c=Context(fluid_library,mu=np.ones((20,24)),lam=-np.ones((20,24)),nt=9,fs=1)
    try:c.prepare()
    finally:c.close()
def test_fluid_first_surface_source_row_remains_rejected(fluid_library):
    c=Context(fluid_library,nt=9,fs=1)
    cfg=c.cfg;c.close();cfg.source_j=0;out=P()
    assert fluid_library.denise_elastic_psv_born_create(C.byref(cfg),C.byref(out))!=0
    assert not out.value
    assert b"source at j=1" in fluid_library.denise_elastic_psv_born_last_error()
def test_validator_only_reciprocal_mutant_is_killed(tmp_path,repository_root):
    source=(repository_root/'src/PSV/elastic_psv_born.c').read_text()
    guarded='(m00 == 0.0 || m10 == 0.0 || m01 == 0.0 || m11 == 0.0)\n            ? 0.0f : '
    assert source.count(guarded)==1
    mutant=tmp_path/'validator_only.c';mutant.write_text(source.replace(guarded,''))
    output=tmp_path/'mutant.so'
    subprocess.run(['cc','-std=c99','-O2','-fPIC','-shared','-I'+str(repository_root/'include'),
     '-DDENISE_FLUID_SERIAL_SOURCE="'+str(mutant)+'"',str(repository_root/'tests/utilities/m9_fluid_cpu_products.c'),'-lm','-o',str(output)],check=True)
    api=library(output);api.fluid_clear_arithmetic();c=Context(api,nt=9)
    try:
        # Validator accepts, but the ACTUAL constructor raises division-by-zero.
        assert api.fluid_singular_arithmetic()!=0
    finally:c.close()

def test_preliminary_actual_horizontal_interface_p_and_sv(fluid_library):
    rho=np.ones((64,64));lam=np.full((64,64),4.);mu=np.zeros((64,64))
    rho[32:]=1.5;lam[32:]=6.75;mu[32:]=3.375
    c=Context(fluid_library,mu=mu,lam=lam,rho=rho,nt=400,fw=10)
    try:
        data=c.prepare();_,s,q,_=c.products();maps=c.maps()
        assert np.all(maps[2,31]==0) and np.all(maps[2,32]>0)
        assert np.isfinite(s).all() and np.linalg.norm(data)>0
        assert np.all(s[4,:32]==0) and np.linalg.norm(s[4,34:50,12:52])>0
        # Off-axis point-shot shear/curl in solid, not a claimed plane coefficient.
        curl=frozen.derivative(s[1],1,False)-frozen.derivative(s[0],0,False)
        assert np.linalg.norm(curl[34:50,12:52])>0
        reference=frozen.elastic_forward(frozen.material(rho,lam,mu),nt=400,fw=10)
        error=np.linalg.norm(data-reference['data'])/np.linalg.norm(reference['data'])
        state_error=np.linalg.norm(s[:5]-reference['state'])/np.linalg.norm(reference['state'])
        assert error<=1e-5 and state_error<=1e-5
        RECORDS.append({'horizontal_interface_preliminary':True,'classification':'bounded B',
         'data_relative_l2':float(error),'state_relative_l2':float(state_error),
         'solid_shear_norm':float(np.linalg.norm(s[4,34:50,12:52])),'solid_curl_norm':float(np.linalg.norm(curl[34:50,12:52]))})
    finally:c.close()
