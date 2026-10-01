"""Frozen M9e-1 CPU/independent FP64/CUDA gates; internal forward ABI only.

No numerical or compile failure is skipped. Only missing CUDA prerequisites
may skip hardware tests. M9 mixed precision ceilings are frozen at 2e-6/8e-6.
"""
from __future__ import annotations
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/utilities'))
import elastic_psv_born_reference as ref

F=C.POINTER(C.c_float); I=C.POINTER(C.c_int); Z=C.c_size_t; P=C.c_void_p
class Config(C.Structure):
    _fields_=[(n,C.c_int) for n in ('nx','ny','nt','fw')]+[('dh',C.c_float),('dt',C.c_float)]+[
        (n,C.c_int) for n in ('l','invmat1','fdorder','ndt','dtinv','free_surface','boundary','mpi_size','receiver_components')]+[
        (n,F) for n in ('lambda','mu','rho')]+[('source_i',C.c_int),('source_j',C.c_int),('source_samples',F),
        ('receiver_count',C.c_int),('receiver_i',I),('receiver_j',I),('cpml_enabled',C.c_int)]+[
        (n,C.c_float) for n in ('pml_reflection','pml_power','pml_kmax','pml_fpml','pml_damping_speed')]
class Options(C.Structure):
    _fields_=[('device',C.c_int),('cap_bytes',Z),('reserve_bytes',Z)]
class Diagnostics(C.Structure):
    _fields_=[(n,Z) for n in ('mandatory_bytes','usable_budget','remaining_budget','owned_bytes','model_bytes',
        'wavefield_bytes','cpml_bytes','profile_bytes','source_bytes','receiver_bytes','trajectory_bytes','workspace_bytes','host_metadata_bytes')]+[
        (n,C.c_int) for n in ('valid_steps','prepared','visible_devices','runtime_version','major','minor')]+[('elapsed_ms',C.c_float)]

def fp(a): return a.ctypes.data_as(F) if a is not None else F()
def ip(a): return a.ctypes.data_as(I)
def same(a,b): return a.tobytes()==b.tobytes()
FIELDS=('vx','vy','sxx','syy','sxy')
MEM=('sxx_x','sxy_y','sxy_x','syy_y','vxx','vyx','vxy','vyy')
OPS=('VXX','VYX','VXY','VYY')
EVIDENCE=[]

@pytest.fixture(scope='session')
def backends(tmp_path_factory):
    candidates=[Path(os.environ.get('NVCC','/usr/local/cuda/bin/nvcc'))]
    candidates+=sorted(Path('/usr/local').glob('cuda-*/bin/nvcc'),reverse=True)
    nvcc=next((str(p) for p in candidates if p.is_file()),shutil.which('nvcc'))
    if not nvcc: pytest.skip('CUDA prerequisite unavailable: nvcc not found')
    out=tmp_path_factory.mktemp('m9e1-build')
    subprocess.run(['make','-C',str(ROOT/'src'),'cuda_m9_elastic_psv','cuda_m9_elastic_psv_nofma',f'NVCC={nvcc}',
                    'CUDA_ARCHS='+os.environ.get('CUDA_ARCHS','86'),f'M9_CUDA_BUILD_DIR={out}'],check=True,capture_output=True,text=True)
    cc=shutil.which('cc')
    assert cc,'host C compiler required'
    subprocess.run([cc,'-std=c99','-O3','-Wall','-Wextra','-Werror','-pedantic','-fPIC','-shared','-I'+str(ROOT/'include'),
        str(ROOT/'tests/utilities/m9e1_cuda_elastic_psv_forward_harness.c'),'-lm','-o',str(out/'cpu.so')],check=True,capture_output=True,text=True)
    cpu=C.CDLL(str(out/'cpu.so'))
    cpu.m9e1_cpu_view.argtypes=[C.POINTER(Config),F,F,F,F,F]
    cpu.m9e1_cpu_static.argtypes=[C.POINTER(Config),F,F]
    libs={}
    for mode in ('nofma','fma'):
        lib=C.CDLL(str(out/('libdenise_m9_cuda_nofma.so' if mode=='nofma' else 'libdenise_m9_cuda.so')))
        lib.denise_cuda_m9_last_error.restype=C.c_char_p
        lib.denise_cuda_m9_create.argtypes=[C.POINTER(Config),C.POINTER(Options),C.POINTER(P)]
        lib.denise_cuda_m9_destroy.argtypes=[C.POINTER(P)]
        lib.denise_cuda_m9_prepare.argtypes=[P]
        lib.denise_cuda_m9_nonlinear.argtypes=[P,F,F]
        lib.denise_cuda_m9_download.argtypes=[P,F,F,F,F]
        lib.denise_cuda_m9_diagnostics.argtypes=[P,C.POINTER(Diagnostics)]
        lib.denise_cuda_m9_test_step.argtypes=[P,F,F]
        lib.denise_cuda_m9_test_halo.argtypes=[P,C.c_int,F]
        lib.denise_cuda_m9_test_static.argtypes=[P,F,F,F,I]
        lib.denise_cuda_m9_test_trajectory_roundtrip.argtypes=[P,F]
        lib.denise_cuda_m9_fault.argtypes=[Z]
        lib.denise_cuda_m9_ledger.argtypes=[C.POINTER(Z)]*4
        cfg,_=fixture('nt1');ctx=P()
        rc=lib.denise_cuda_m9_create(C.byref(cfg),C.byref(Options()),C.byref(ctx))
        if rc:
            message=lib.denise_cuda_m9_last_error().decode()
            # Real missing hardware only: never compiler, injected, memory, or numeric errors.
            if 'no CUDA-capable device' in message or 'driver version is insufficient' in message or 'visible=0' in message:
                pytest.skip('CUDA prerequisite unavailable: '+message)
            pytest.fail(message)
        assert lib.denise_cuda_m9_destroy(C.byref(ctx))==0
        libs[mode]=lib
    yield cpu,libs
    dest=os.environ.get('DENISE_M9E1_EVIDENCE')
    if dest: Path(dest).write_text(json.dumps(EVIDENCE,indent=2))

def fixture(name,nx=17,ny=13,nt=24):
    fw=2 if name in ('cpml','heterogeneous','arbitrary') else 0
    if name=='nt1': nt=1
    y,x=np.mgrid[:ny,:nx]
    heterogeneous=name in ('heterogeneous','arbitrary')
    l=np.asarray(5e9*(1+.1*np.sin(.31*x+.22*y)) if heterogeneous else np.full((ny,nx),5e9),np.float32)
    m=np.asarray(3e9*(1+.12*np.cos(.19*x-.17*y)) if heterogeneous else np.full((ny,nx),3e9),np.float32)
    r=np.asarray(2000*(1+.05*np.sin(.27*x+.13*y)) if heterogeneous else np.full((ny,nx),2000),np.float32)
    t=np.arange(nt);source=np.asarray(1e5*(np.cos(.41*t)+.21*np.sin(.73*t))*np.exp(-t/8),np.float32)
    ri=np.array([nx-2,2,nx//2,2,1,nx-1],np.int32);rj=np.array([ny//2,ny-2,ny//2,ny-2,1,ny-1],np.int32)
    cfg=Config(nx=nx,ny=ny,nt=nt,fw=fw,dh=10,dt=.0005,invmat1=3,fdorder=4,ndt=1,dtinv=1,
        mpi_size=1,receiver_components=2,mu=fp(m),rho=fp(r),source_i=nx//2-1,source_j=ny//2+1,
        source_samples=fp(source),receiver_count=len(ri),receiver_i=ip(ri),receiver_j=ip(rj),cpml_enabled=int(fw>0),
        pml_reflection=.001,pml_power=2,pml_kmax=1.3,pml_fpml=15,
        pml_damping_speed=float(np.sqrt(np.max((l.astype(float)+2*m)/r))))
    setattr(cfg,'lambda',fp(l))
    return cfg,dict(l=l,m=m,r=r,source=source,ri=ri,rj=rj)

def outputs(cfg):
    return (np.empty((cfg.nt,cfg.receiver_count,2),np.float32),np.empty((5,cfg.ny,cfg.nx),np.float32),
        np.empty((8,cfg.ny,cfg.nx),np.float32),np.empty((cfg.nt,4,cfg.ny,cfg.nx),np.float32))
def cpu_run(cpu,cfg,state=None,profiles=None):
    d,f,p,q=outputs(cfg);allstate=np.empty((13,cfg.ny,cfg.nx),np.float32)
    assert cpu.m9e1_cpu_view(C.byref(cfg),fp(state),fp(profiles),fp(d),fp(allstate),fp(q))==0
    f[:]=allstate[:5];p[:]=allstate[5:];return d,f,p,q
def check(lib,rc): assert rc==0,lib.denise_cuda_m9_last_error().decode()
def create(lib,cfg):
    ctx=P();check(lib,lib.denise_cuda_m9_create(C.byref(cfg),C.byref(Options()),C.byref(ctx)));return ctx
def download(lib,ctx,cfg):
    out=outputs(cfg);check(lib,lib.denise_cuda_m9_download(ctx,*map(fp,out)));return out
def ledger(lib):
    values=[Z() for _ in range(4)];lib.denise_cuda_m9_ledger(*map(C.byref,values));return tuple(v.value for v in values)

def metrics(a,b):
    delta=a.astype(float)-b.astype(float);peak=float(np.max(np.abs(b)));absdelta=np.abs(delta)
    ordered=lambda x: np.where(x.view(np.int32)<0,-2147483648-x.view(np.int32).astype(np.int64),x.view(np.int32).astype(np.int64))
    ulps=np.abs(ordered(a.astype(np.float32))-ordered(b.astype(np.float32)))
    return dict(rel_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(b.astype(float)),1e-300)),
        max_abs=float(absdelta.max()),peak_normalized=float(absdelta.max()/max(peak,1e-300)),max_ulp=int(ulps.max()),
        max_abs_location=list(map(int,np.unravel_index(absdelta.argmax(),a.shape))),
        max_ulp_location=list(map(int,np.unravel_index(ulps.argmax(),a.shape))),bitwise=same(a,b))
def compare(label,a,b,exact=False):
    pairs=[('data',a[0],b[0])]+[(n,a[1][k],b[1][k]) for k,n in enumerate(FIELDS)]+[
        (n,a[2][k],b[2][k]) for k,n in enumerate(MEM)]+[(n,a[3][:,k],b[3][:,k]) for k,n in enumerate(OPS)]
    record={'case':label,'arrays':{}}
    for n,x,y in pairs:
        z=metrics(x,y);record['arrays'][n]=z
        assert z['rel_l2']<=2e-6 and z['peak_normalized']<=8e-6,(label,n,z)
        if exact: assert same(x,y),(label,n,z)
    EVIDENCE.append(record)

def oracle(cfg,a,state=None,profiles=None):
    """Extend the existing independent FP64 M9b primitives to final state.

    Frozen prepared FP32 source samples are the input, not reinterpreted wavelets.
    Host profile formula is independent M9b; damping speed is fixed across models.
    """
    nx,ny=cfg.nx,cfg.ny
    if profiles is None:
        args=dict(damping_speed=cfg.pml_damping_speed,reflection=cfg.pml_reflection,power=cfg.pml_power,kmax=cfg.pml_kmax,fpml=cfg.pml_fpml)
        prof={}
        for name,n,half in [('x',nx,False),('xh',nx,True),('y',ny,False),('yh',ny,True)]:
            v=ref._cpml_1d(n,cfg.dh,cfg.dt,cfg.fw,half=half,**args)
            prof[name]=ref.PMLProfile(*(x[None,:] if name.startswith('x') else x[:,None] for x in v))
    else:
        prof={};offset=0
        for name,n in [('x',nx),('xh',nx),('y',ny),('yh',ny)]:
            v=profiles[offset:offset+3*n].astype(float).reshape(3,n);offset+=3*n
            prof[name]=ref.PMLProfile(*(x[None,:] if name.startswith('x') else x[:,None] for x in v))
    s=np.zeros((13,ny,nx),float) if state is None else state.astype(float).copy()
    f=s[:5];p=s[5:];dt=cfg.dt/cfg.dh
    invx,invy=ref._density_faces(a['r'].astype(float));corner=ref.shear_corner_mu(a['m'].astype(float))
    data=np.empty((cfg.nt,cfg.receiver_count,2));qsave=np.empty((cfg.nt,4,ny,nx))
    def corr(v,k,name):
        q,nextp=ref._pml_forward(v,p[k],prof[name]);p[k]=nextp;return q
    for t in range(cfg.nt):
        xx=corr(dt*ref._stencil(f[2],ref._DX_FWD),0,'xh')
        xy=corr(dt*ref._stencil(f[4],ref._DY_BACK),1,'y')
        yx=corr(dt*ref._stencil(f[4],ref._DX_BACK),2,'x')
        yy=corr(dt*ref._stencil(f[3],ref._DY_FWD),3,'yh')
        f[0]+=invx*(xx+xy);f[1]+=invy*(yx+yy)
        data[t,:,0]=f[0,a['rj'],a['ri']];data[t,:,1]=f[1,a['rj'],a['ri']]
        q=qsave[t]
        q[0]=corr(dt*ref._stencil(f[0],ref._DX_BACK),4,'x')
        q[1]=corr(dt*ref._stencil(f[1],ref._DX_FWD),5,'xh')
        q[2]=corr(dt*ref._stencil(f[0],ref._DY_FWD),6,'yh')
        q[3]=corr(dt*ref._stencil(f[1],ref._DY_BACK),7,'y')
        div=q[0]+q[3]
        f[2]+=a['l']*div+2*a['m'].astype(float)*q[0]
        f[3]+=a['l']*div+2*a['m'].astype(float)*q[3]
        f[4]+=corner*(q[1]+q[2])
        f[2,cfg.source_j,cfg.source_i]+=a['source'][t];f[3,cfg.source_j,cfg.source_i]+=a['source'][t]
    return data,f.copy(),p.copy(),qsave

@pytest.mark.parametrize('mode',['nofma','fma'])
@pytest.mark.parametrize('name',['interior','heterogeneous','cpml','nt1'])
@pytest.mark.parametrize('nonlinear',[False,True])
def test_background_nonlinear(backends,mode,name,nonlinear):
    cpu,libs=backends;lib=libs[mode];cfg,a=fixture(name);ctx=create(lib,cfg)
    try:
        original=Config.from_buffer_copy(cfg)
        background=a.copy()  # Own the original pointer targets throughout re-prepare.
        if nonlinear:
            y,x=np.mgrid[:cfg.ny,:cfg.nx];a['l']=np.asarray(a['l']*(1+.013*np.sin(.43*x+.21*y)),np.float32)
            a['m']=np.asarray(a['m']*(1+.017*np.cos(.19*x-.32*y)),np.float32)
            setattr(cfg,'lambda',fp(a['l']));cfg.mu=fp(a['m'])
            check(lib,lib.denise_cuda_m9_nonlinear(ctx,fp(a['l']),fp(a['m'])))
        else: check(lib,lib.denise_cuda_m9_prepare(ctx))
        got=download(lib,ctx,cfg);expected=cpu_run(cpu,cfg)
        label=f'{mode}/{name}/'+('nonlinear' if nonlinear else 'background')
        compare(label+'/cpu',got,expected,exact=mode=='nofma')
        independent=oracle(cfg,a)
        compare(label+'/oracle',got,independent)
        compare(label+'/cpu-oracle',expected,independent)
        if nonlinear: check(lib,lib.denise_cuda_m9_nonlinear(ctx,fp(a['l']),fp(a['m'])))
        else: check(lib,lib.denise_cuda_m9_prepare(ctx))
        repeat=download(lib,ctx,cfg)
        assert all(same(x,y) for x,y in zip(got,repeat))
        # Re-prepare must restore original model after nonlinear execution.
        check(lib,lib.denise_cuda_m9_prepare(ctx));restored=download(lib,ctx,original)
        compare(label+'/restored',restored,cpu_run(cpu,original),exact=mode=='nofma')
        assert same(got[0][:,1],got[0][:,3])
        assert np.count_nonzero(got[0][0])==0
        if name=='nt1':
            assert np.count_nonzero(got[3])==0
            assert got[1][2,cfg.source_j,cfg.source_i]==a['source'][0]
            assert same(got[1][2],got[1][3])
    finally: check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))
    assert ledger(lib)[:3]==(0,0,0)

@pytest.mark.parametrize('mode',['nofma','fma'])
def test_uploads_memory_halos_arbitrary(backends,mode):
    cpu,libs=backends;lib=libs[mode];cfg,a=fixture('arbitrary',nt=1);ctx=create(lib,cfg)
    try:
        maps=np.empty((5,cfg.ny+4,cfg.nx+4),np.float32);profiles=np.empty(6*(cfg.nx+cfg.ny),np.float32)
        source=np.empty(cfg.nt,np.float32);geometry=np.empty((cfg.receiver_count,3),np.int32)
        check(lib,lib.denise_cuda_m9_test_static(ctx,fp(maps),fp(profiles),fp(source),ip(geometry)))
        cm=np.empty_like(maps);cp=np.empty_like(profiles)
        assert cpu.m9e1_cpu_static(C.byref(cfg),fp(cm),fp(cp))==0
        assert same(maps,cm) and same(profiles,cp) and same(source,a['source'])
        assert same(geometry,np.stack((a['ri'],a['rj'],np.arange(cfg.receiver_count,dtype=np.int32)),axis=1))
        d=Diagnostics();check(lib,lib.denise_cuda_m9_diagnostics(ctx,C.byref(d)))
        assert d.mandatory_bytes==sum(getattr(d,n) for n in ('model_bytes','wavefield_bytes','cpml_bytes','profile_bytes','source_bytes','receiver_bytes','trajectory_bytes','workspace_bytes'))
        assert d.owned_bytes==d.mandatory_bytes==ledger(lib)[0]
        assert d.trajectory_bytes==cfg.nt*4*cfg.nx*cfg.ny*4
        EVIDENCE.append({'case':f'{mode}/memory','diagnostics':{n:getattr(d,n) for n,_ in Diagnostics._fields_},
                         'owned_device_host_requested_bytes_events':ledger(lib)[:3]})
        for vel in (0,1):
            h=np.arange(maps.size,dtype=np.float32).reshape(maps.shape);expected=h.copy()
            for k in (range(2) if vel else range(2,5)):
                expected[k,2:-2,:2]=expected[k,2:-2,-4:-2];expected[k,2:-2,-2:]=expected[k,2:-2,2:4]
                expected[k,:2]=expected[k,-4:-2];expected[k,-2:]=expected[k,2:4]
            check(lib,lib.denise_cuda_m9_test_halo(ctx,vel,fp(h)));assert same(h,expected)
        rng=np.random.default_rng(24681);state=rng.standard_normal((13,cfg.ny,cfg.nx)).astype(np.float32)
        state[2:5]*=1e5;state[5:9]*=10;state[9:]*=1e-7
        custom=profiles.copy();offset=0
        for n in (cfg.nx,cfg.nx,cfg.ny,cfg.ny):
            custom[offset:offset+n]=np.linspace(1.1,1.7,n,dtype=np.float32)
            custom[offset+n:offset+2*n]=np.linspace(-.17,-.03,n,dtype=np.float32)
            custom[offset+2*n:offset+3*n]=np.linspace(.51,.93,n,dtype=np.float32);offset+=3*n
        check(lib,lib.denise_cuda_m9_test_step(ctx,fp(state),fp(custom)))
        got=download(lib,ctx,cfg);expected=cpu_run(cpu,cfg,state,custom)
        compare(f'{mode}/arbitrary/cpu',got,expected,exact=mode=='nofma')
        compare(f'{mode}/arbitrary/oracle',got,oracle(cfg,a,state,custom))
    finally: check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))

def test_dense_basis(backends):
    cpu,libs=backends;lib=libs['nofma'];cfg,a=fixture('arbitrary',nx=5,ny=6,nt=1)
    cfg.fw=0;cfg.cpml_enabled=0;a['source'][:]=0;ctx=create(lib,cfg)
    try:
        # Every state basis vector on the minimal periodic non-square domain.
        for k in range(13):
            for j in range(cfg.ny):
                for i in range(cfg.nx):
                    state=np.zeros((13,cfg.ny,cfg.nx),np.float32)
                    state[k,j,i]=1e5 if 2<=k<=4 else (1e-7 if k>=9 else 1)
                    check(lib,lib.denise_cuda_m9_test_step(ctx,fp(state),F()))
                    got=download(lib,ctx,cfg);expected=cpu_run(cpu,cfg,state)
                    assert all(same(x,y) for x,y in zip(got,expected)),(k,j,i)
                    independent=oracle(cfg,a,state)
                    # Dense columns have nonzero errors bounded in both ceilings.
                    compare(f'basis/{k}/{j}/{i}',got,independent)
    finally: check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))

def test_reject_before_mutation(backends):
    _,libs=backends;lib=libs['nofma'];cfg,a=fixture('nt1')
    for name,value,needle in [('free_surface',1,'FREE_SURF=0'),('mpi_size',2,'one MPI rank'),('l',1,'L=0'),('fdorder',6,'FDORDER=4')]:
        c=Config.from_buffer_copy(cfg);setattr(c,name,value);ctx=P()
        assert lib.denise_cuda_m9_create(C.byref(c),C.byref(Options()),C.byref(ctx))<0
        assert needle in lib.denise_cuda_m9_last_error().decode();assert not ctx.value
        assert ledger(lib)[:3]==(0,0,0)
    for cap,reserve in [(1,0),(0,2**63)]:
        ctx=P();assert lib.denise_cuda_m9_create(C.byref(cfg),C.byref(Options(0,cap,reserve)),C.byref(ctx))<0
        assert 'budget' in lib.denise_cuda_m9_last_error().decode() and not ctx.value
    c=Config.from_buffer_copy(cfg);c.nx=2**31-1;ctx=P()
    assert lib.denise_cuda_m9_create(C.byref(c),C.byref(Options()),C.byref(ctx))<0
    assert 'overflow' in lib.denise_cuda_m9_last_error().decode() and ledger(lib)[:3]==(0,0,0)
    ctx=P();assert lib.denise_cuda_m9_create(C.byref(cfg),C.byref(Options(999,0,0)),C.byref(ctx))<0
    assert 'unavailable' in lib.denise_cuda_m9_last_error().decode()

def test_fault_sweep(backends):
    _,libs=backends;lib=libs['nofma'];cfg,a=fixture('nt1')
    lib.denise_cuda_m9_fault(0);ctx=create(lib,cfg);create_calls=ledger(lib)[3]
    check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))
    for at in range(1,create_calls+1):
        lib.denise_cuda_m9_fault(at);ctx=P()
        assert lib.denise_cuda_m9_create(C.byref(cfg),C.byref(Options()),C.byref(ctx))<0,at
        message=lib.denise_cuda_m9_last_error().decode();assert 'injected' in message,(at,message)
        check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)));assert ledger(lib)[:3]==(0,0,0),(at,ledger(lib))
    counts={}
    for operation in ('prepare','nonlinear','download','step','halo','static','trajectory'):
        lib.denise_cuda_m9_fault(0);ctx=create(lib,cfg);check(lib,lib.denise_cuda_m9_prepare(ctx))
        initial=np.zeros((13,cfg.ny,cfg.nx),np.float32)
        h=np.zeros((5,cfg.ny+4,cfg.nx+4),np.float32)
        maps=h.copy();profiles=np.ones(6*(cfg.nx+cfg.ny),np.float32);source=np.empty(cfg.nt,np.float32);g=np.empty((cfg.receiver_count,3),np.int32)
        def invoke():
            if operation=='prepare': return lib.denise_cuda_m9_prepare(ctx)
            if operation=='nonlinear': return lib.denise_cuda_m9_nonlinear(ctx,fp(a['l']),fp(a['m']))
            if operation=='download': return lib.denise_cuda_m9_download(ctx,*map(fp,out))
            if operation=='step': return lib.denise_cuda_m9_test_step(ctx,fp(initial),fp(profiles))
            if operation=='halo': return lib.denise_cuda_m9_test_halo(ctx,1,fp(h))
            if operation=='trajectory': return lib.denise_cuda_m9_test_trajectory_roundtrip(ctx,fp(out[3]))
            return lib.denise_cuda_m9_test_static(ctx,fp(maps),fp(profiles),fp(source),ip(g))
        out=outputs(cfg);lib.denise_cuda_m9_fault(0);check(lib,invoke());n=ledger(lib)[3]
        check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))
        for at in range(1,n+1):
            lib.denise_cuda_m9_fault(0);ctx=create(lib,cfg);check(lib,lib.denise_cuda_m9_prepare(ctx))
            out=tuple(np.full_like(x,137) for x in outputs(cfg));lib.denise_cuda_m9_fault(at)
            assert invoke()<0,(operation,at)
            assert 'injected' in lib.denise_cuda_m9_last_error().decode(),(operation,at)
            d=Diagnostics();check(lib,lib.denise_cuda_m9_diagnostics(ctx,C.byref(d)))
            assert d.valid_steps==0 and not d.prepared
            if operation=='download': assert all(np.all(x==137) for x in out)
            check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)));assert ledger(lib)[:3]==(0,0,0),(operation,at,ledger(lib))
        counts[operation]=n
    lib.denise_cuda_m9_fault(0);EVIDENCE.append({'case':'faults','create':create_calls,'operations':counts})

def test_fma_comparison_performance(backends):
    cpu,libs=backends;cfg,a=fixture('heterogeneous',nx=97,ny=79,nt=120);results={};timings={}
    for mode,lib in libs.items():
        ctx=create(lib,cfg)
        try:
            check(lib,lib.denise_cuda_m9_prepare(ctx));samples=[]
            for _ in range(5):
                start=time.perf_counter();check(lib,lib.denise_cuda_m9_prepare(ctx));wall=time.perf_counter()-start
                d=Diagnostics();check(lib,lib.denise_cuda_m9_diagnostics(ctx,C.byref(d)));samples.append((wall,d.elapsed_ms))
            results[mode]=download(lib,ctx,cfg);timings[mode]=samples
        finally: check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))
    start=time.perf_counter();expected=cpu_run(cpu,cfg);cpu_time=time.perf_counter()-start
    for mode in libs: compare(f'performance/{mode}/cpu',results[mode],expected,exact=mode=='nofma')
    compare('performance/fma-nofma',results['fma'],results['nofma'])
    EVIDENCE.append({'case':'timing','cpu_wall_seconds':cpu_time,'gpu_wall_seconds_event_ms':timings})

def test_zero_source_endpoints_and_layout(backends):
    cpu,libs=backends;lib=libs['nofma'];cfg,a=fixture('interior',nt=2)
    a['source'][:]=0;ctx=create(lib,cfg)
    try:
        sentinels=np.arange(cfg.nt*4*cfg.ny*cfg.nx,dtype=np.float32).reshape(cfg.nt,4,cfg.ny,cfg.nx)/8-37
        original=sentinels.copy()
        check(lib,lib.denise_cuda_m9_test_trajectory_roundtrip(ctx,fp(sentinels)));assert same(sentinels,original)
        check(lib,lib.denise_cuda_m9_prepare(ctx));out=download(lib,ctx,cfg)
        assert all(x.tobytes()==bytes(x.nbytes) for x in out)
        # First and last increments cannot receive an extra DT or endpoint rule.
        a['source'][:]=[1.25,-3.5]
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))
    ctx=create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(ctx));out=download(lib,ctx,cfg)
        compare('source-first-last',out,cpu_run(cpu,cfg),exact=True)
        independent=oracle(cfg,a);compare('source-first-last/oracle',out,independent)
        assert np.count_nonzero(out[3][0])==0 and np.count_nonzero(out[3][1])>0
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))

def test_actual_hidden_device_cpu_still_runs(backends):
    cpu,libs=backends;cfg,a=fixture('nt1')
    program='''
import ctypes as C
from tests.physics.test_m9e1_cuda_elastic_psv_forward import Config,Options,fixture
lib=C.CDLL(PATH)
lib.denise_cuda_m9_last_error.restype=C.c_char_p
cfg,owned=fixture('nt1');ctx=C.c_void_p()
assert lib.denise_cuda_m9_create(C.byref(cfg),C.byref(Options()),C.byref(ctx))<0
message=lib.denise_cuda_m9_last_error().decode()
assert 'unavailable' in message or 'no CUDA-capable device' in message,message
assert not ctx.value
print(message)
'''.replace('PATH',repr(libs['nofma']._name))
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1'
    result=subprocess.run([sys.executable,'-c',program],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    compare('cpu-without-cuda',cpu_run(cpu,cfg),oracle(cfg,a))
    EVIDENCE.append({'case':'hidden-device','diagnostic':result.stdout.strip()})

def test_physical_propagation_and_absorption(backends):
    cpu,libs=backends;lib=libs['nofma'];runs={};cfgs={}
    for damping in (False,True):
        cfg,a=fixture('interior',nx=61,ny=55,nt=450)
        cfg.source_i=30;cfg.source_j=27;cfg.fw=10 if damping else 0;cfg.cpml_enabled=int(damping)
        a['ri']=np.array([36,24,30,30],np.int32);a['rj']=np.array([27,27,33,21],np.int32)
        cfg.receiver_i=ip(a['ri']);cfg.receiver_j=ip(a['rj']);cfg.receiver_count=4
        t=np.arange(cfg.nt)*cfg.dt;tau=np.pi*60*(t-.025)
        a['source'][:]=np.asarray(1e5*(1-2*tau*tau)*np.exp(-tau*tau),np.float32)
        ctx=create(lib,cfg)
        try:
            check(lib,lib.denise_cuda_m9_prepare(ctx));out=download(lib,ctx,cfg)
            compare(f'physical/cpml={damping}/cpu',out,cpu_run(cpu,cfg),exact=True)
            runs[damping]=out;cfgs[damping]=(cfg,a)
        finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(ctx)))
    cfg,a=cfgs[False];data=runs[False][0]
    vp=np.sqrt((a['l'][0,0]+2*a['m'][0,0])/a['r'][0,0]);expected=.025+60/vp
    lo=int((expected-.012)/cfg.dt);hi=int((expected+.012)/cfg.dt)
    traces=[data[:,0,0],data[:,1,0],data[:,2,1],data[:,3,1]]
    peaks=[int(np.argmax(np.abs(v[lo:hi]))+lo) for v in traces]
    # P-wave first packet within the travel-time/source-bandwidth window.
    assert all(abs(k*cfg.dt-expected)<=.012 for k in peaks)
    assert np.dot(traces[0][lo:hi],traces[1][lo:hi])<0
    assert np.dot(traces[2][lo:hi],traces[3][lo:hi])<0
    def energy(out):
        f=out[1].astype(float)
        return float(np.sum(a['r']*(f[0]**2+f[1]**2)+(f[2]**2+f[3]**2+2*f[4]**2)/(a['l']+2*a['m'])))
    eoff,eon=energy(runs[False]),energy(runs[True]);assert eon<.5*eoff,(eon,eoff)
    EVIDENCE.append({'case':'physical','expected_arrival_seconds':float(expected),'peak_samples':peaks,
        'cpml_off_energy':eoff,'cpml_on_energy':eon,'cpml_energy_ratio':eon/eoff,
        'opposite_component_polarity':True})
