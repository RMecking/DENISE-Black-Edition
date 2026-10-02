"""M9e-3 scoped CUDA FP64 transpose and raw migration gates.

Ceilings inherited before execution: J 1e-5, JT images 6e-5;
independent FP64/local transpose 5e-13. No CPU oracle is edited.
"""
import ctypes as C
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import numpy as np
import pytest
from tests.physics import test_m9e2_cuda_free_surface_born as e2
from tests.physics import test_m9e1_cuda_elastic_psv_forward as e1
from tests.physics.test_m9b1_elastic_psv_born_production import (
    _production_dot_metrics, _assert_production_dot_closes)
from tests.utilities import elastic_psv_free_surface_reference as s

backends=e2.backends
F,P,Z=e1.F,e1.P,e1.Z
D=C.POINTER(C.c_double)
fp,check=e1.fp,e1.check
RECORDS=[]
class JTDiagnostics(C.Structure):
    _fields_=[(n,Z) for n in ('field_bytes','cpml_bytes','image_bytes','data_bytes','workspace_bytes','alignment_bytes')]+[
        ('valid',C.c_int),('elapsed_ms',C.c_float)]

def dp(a):return a.ctypes.data_as(D)
@pytest.fixture(scope='session')
def jt(backends):
    cpu,libs=backends
    cpu.denise_elastic_psv_born_create.argtypes=[C.POINTER(e1.Config),C.POINTER(P)]
    cpu.denise_elastic_psv_born_prepare.argtypes=[P,F]
    cpu.denise_elastic_psv_born_apply_jt.argtypes=[P,F,D,D]
    cpu.denise_elastic_psv_born_destroy.argtypes=[C.POINTER(P)]
    for lib in libs.values():
        lib.denise_cuda_m9_create_migration.argtypes=[C.POINTER(e1.Config),C.POINTER(e1.Options),C.POINTER(P)]
        lib.denise_cuda_m9_apply_jt.argtypes=[P,F,Z]
        lib.denise_cuda_m9_image_download.argtypes=[P,D,D,Z]
        lib.denise_cuda_m9_migrate.argtypes=[P,F,Z,D,D,Z]
        lib.denise_cuda_m9_adjoint_diagnostics.argtypes=[P,C.POINTER(JTDiagnostics)]
        lib.denise_cuda_m9_test_reverse.argtypes=[P,C.c_int,C.c_int,C.c_int,D,D,D,D,F,F]
    yield cpu,libs
    if os.environ.get('DENISE_M9E3_EVIDENCE'):
        Path(os.environ['DENISE_M9E3_EVIDENCE']).write_text(json.dumps(RECORDS,indent=2))

def create(lib,cfg,cap=0):
    c=P();check(lib,lib.denise_cuda_m9_create_migration(C.byref(cfg),C.byref(e1.Options(0,cap,0)),C.byref(c)));return c
def images(cfg):return np.empty((2,cfg.ny,cfg.nx),np.float64)
def download(lib,c,cfg):
    out=images(cfg);check(lib,lib.denise_cuda_m9_image_download(c,dp(out[0]),dp(out[1]),cfg.nx*cfg.ny));return out
def apply(lib,c,cfg,r):
    check(lib,lib.denise_cuda_m9_apply_jt(c,fp(r),r.size));return download(lib,c,cfg)
def cpu_jt(cpu,cfg,r):
    c=P();out=images(cfg)
    assert cpu.denise_elastic_psv_born_create(C.byref(cfg),C.byref(c))==0
    try:
        assert cpu.denise_elastic_psv_born_prepare(c,F())==0
        assert cpu.denise_elastic_psv_born_apply_jt(c,fp(r),dp(out[0]),dp(out[1]))==0
    finally:cpu.denise_elastic_psv_born_destroy(C.byref(c))
    return out
def metrics(a,b):
    z=e1.metrics(a,b)
    # FP64 ULP is diagnostic; cancellation-near-zero values dominate this count.
    ai=np.where(a.view(np.int64)<0,np.iinfo(np.int64).min-a.view(np.int64),a.view(np.int64)).astype(object)
    bi=np.where(b.view(np.int64)<0,np.iinfo(np.int64).min-b.view(np.int64),b.view(np.int64)).astype(object)
    z['max_fp64_ulp']=int(np.max(np.abs(ai-bi)));return z
def tight(a,b):
    delta=np.linalg.norm(a-b);scale=max(np.linalg.norm(b),np.finfo(float).tiny)
    assert delta/scale<=5e-13,(delta/scale,np.max(np.abs(a-b)))

@pytest.mark.parametrize('mode',['nofma','fma'])
@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('cpml',[False,True])
def test_images_dots_repeatability_and_migration(jt,mode,surface,cpml):
    cpu,libs=jt;lib=libs[mode]
    exp=s.fixture(cpml=cpml);cfg,a=e2.config_from_exp(exp,surface)
    c=create(lib,cfg);rng=np.random.default_rng(930)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c));bg=e1.download(lib,c,cfg)
        # Independent FP64 full graph, frozen source/geometry/coefficients.
        if surface:
            _,tapes,_=s.forward(exp)
            oracle_j=lambda dl,dm:s.born(exp,tapes,dl,dm)
            oracle_jt=lambda r:np.stack(s.adjoint(exp,tapes,r))
        else:
            tapes=e1.ref.nonlinear_forward(exp,save_strain=True)
            oracle_j=lambda dl,dm:e1.ref.born_forward(exp,tapes,dl,dm)
            def oracle_jt(r):
                image=e1.ref.born_adjoint(exp,tapes,r)
                return np.stack((image.image_lambda_raw,image.image_mu_raw))
        for kind in ('lambda','mu','joint'):
            dl,dm=e2.direction(cfg,a,kind)
            j=e2.gpu_j(lib,c,cfg,dl,dm)[0]
            r=np.asarray(j if kind!='joint' else rng.normal(size=j.shape),np.float32)
            got=apply(lib,c,cfg,r);want=cpu_jt(cpu,cfg,r)
            rec={'case':f'{mode}/fs{surface}/cpml{int(cpml)}/{kind}',
                 'images':{n:metrics(got[k],want[k]) for k,n in enumerate(('lambda','mu'))}}
            for z in rec['images'].values():assert z['rel_l2']<=6e-5,z
            oj=oracle_j(dl.astype(float),dm.astype(float));og=oracle_jt(r.astype(float))
            lhs=float(np.sum(oj*r));rhs=float(np.sum(dl*og[0])+np.sum(dm*og[1]))
            scale=max(np.linalg.norm(oj)*np.linalg.norm(r),np.linalg.norm(np.stack((dl,dm)))*np.linalg.norm(og),1e-300)
            oracle={'absolute_residual':abs(lhs-rhs),'absolute_ceiling':5e-13*scale}
            z=_production_dot_metrics(j,r,dl,dm,*got,oj,*og,oracle)
            _assert_production_dot_closes(z);rec['dot']=z;RECORDS.append(rec)
            assert got.tobytes()==apply(lib,c,cfg,r).tobytes()
            assert j.tobytes()==e2.gpu_j(lib,c,cfg,dl,dm)[0].tobytes()
            assert all(x.tobytes()==y.tobytes() for x,y in zip(bg,e1.download(lib,c,cfg)))
        out=images(cfg)
        check(lib,lib.denise_cuda_m9_migrate(c,fp(r),r.size,dp(out[0]),dp(out[1]),cfg.nx*cfg.ny))
        assert got.tobytes()==out.tobytes()
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
    assert e1.ledger(lib)[:3]==(0,0,0)

@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('cpml',[False,True])
def test_receiver_and_material_basis(jt,surface,cpml):
    cpu,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(surface,nx=5,ny=5,nt=12,cpml=cpml)
    if cpml:cfg.fw=1  # bottom CPML must leave the four surface rows available
    c=create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c));shape=(cfg.nt,cfg.receiver_count,2)
        for rec in range(cfg.receiver_count):
            for component in (0,1):
                r=np.zeros(shape,np.float32);r[-1,rec,component]=1
                got=apply(lib,c,cfg,r);want=cpu_jt(cpu,cfg,r)
                for k in (0,1):assert metrics(got[k],want[k])['rel_l2']<=6e-5
        # Joint impulse duplicate addition and scrambled indices.
        r=np.zeros(shape,np.float32);r[-1,:,0]=[1,2,-3,4,5,-6]
        g=apply(lib,c,cfg,r)
        expected=np.zeros_like(g)
        for rec in range(cfg.receiver_count):
            one=np.zeros(shape,np.float32);one[-1,rec,0]=r[-1,rec,0];expected+=apply(lib,c,cfg,one)
        tight(g,expected)
        # Each centered material cell, including surface and wrap corners.
        for channel in (0,1):
            for j,i in np.ndindex(cfg.ny,cfg.nx):
                dl=np.zeros_like(a['l']);dm=np.zeros_like(a['m'])
                (dl if channel==0 else dm)[j,i]=1e8
                out=e2.gpu_j(lib,c,cfg,dl,dm)[0]
                lhs=float(np.sum(out.astype(float)*r));rhs=1e8*g[channel,j,i]
                scale=max(np.linalg.norm(out)*np.linalg.norm(r),1e8*np.linalg.norm(g),1e-300)
                assert abs(lhs-rhs)<=7e-5*scale,(surface,cpml,channel,j,i,lhs,rhs)
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

def blocks(lib,c,cfg,mode,kind=0,field=0,f=None,p=None,q=None,g=None,bg=None,r=None):
    f=np.zeros((5,cfg.ny+4,cfg.nx+4)) if f is None else f.copy()
    p=np.zeros((8,cfg.ny,cfg.nx)) if p is None else p.copy()
    q=np.zeros((4,cfg.ny,cfg.nx)) if q is None else q.copy()
    g=np.zeros((2,cfg.ny,cfg.nx)) if g is None else g.copy()
    bg=np.zeros((4,cfg.ny,cfg.nx),np.float32) if bg is None else bg
    r=np.zeros((cfg.nt,cfg.receiver_count,2),np.float32) if r is None else r
    check(lib,lib.denise_cuda_m9_test_reverse(c,mode,kind,field,dp(f),dp(p),dp(q),dp(g),fp(bg),fp(r)))
    return f,p,q,g

@pytest.mark.parametrize('surface',[0,1])
def test_halo_dense_basis_and_receiver_block(jt,surface):
    _,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(surface,nx=5,ny=5,nt=1,cpml=False);c=create(lib,cfg)
    try:
        # Explicit independently constructed COPY matrix including corners.
        ny,nx=cfg.ny,cfg.nx;shape=(ny+4,nx+4);n=np.prod(shape)
        def copy(x):
            y=x.copy()
            y[2:-2,:2]=y[2:-2,nx:nx+2];y[2:-2,nx+2:]=y[2:-2,2:4]
            y[:2]=y[ny:ny+2];y[ny+2:]=y[2:4];return y
        matrix=s.matrix_of(copy,shape)
        for field in (0,2,4):
            for index in range(n):
                f=np.zeros((5,*shape));f[field].flat[index]=1
                got=blocks(lib,c,cfg,0,field=field,f=f)[0][field]
                expected=matrix.T[:,index].reshape(shape)
                np.testing.assert_array_equal(got,expected)
        for receiver in range(cfg.receiver_count):
            for component in (0,1):
                r=np.zeros((1,cfg.receiver_count,2),np.float32);r[0,receiver,component]=1
                got=blocks(lib,c,cfg,7,r=r)[0]
                expected=np.zeros_like(got);expected[component,a['rj'][receiver]+2,a['ri'][receiver]+2]=1
                np.testing.assert_array_equal(got,expected)
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('cpml',[False,True])
def test_cpml_all_channels_dense_recurrence(jt,surface,cpml):
    _,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(surface,nx=7,ny=7,nt=1,cpml=cpml);c=create(lib,cfg)
    try:
        maps=np.empty(5*(cfg.nx+4)*(cfg.ny+4),np.float32)
        prof=np.empty(6*(cfg.nx+cfg.ny),np.float32);src=np.empty(cfg.nt,np.float32);geo=np.empty(3*cfg.receiver_count,np.int32)
        check(lib,lib.denise_cuda_m9_test_static(c,fp(maps),fp(prof),fp(src),e1.ip(geo)))
        n=cfg.nx;x=prof[:3*n].reshape(3,n);xh=prof[3*n:6*n].reshape(3,n)
        y=prof[6*n:9*n].reshape(3,n);yh=prof[9*n:].reshape(3,n)
        for kind,pr in enumerate((xh,y,x,yh,x,xh,yh,y)):
            for j,i in np.ndindex(n,n):
                index=i if kind in (0,2,4,5) else j
                k,aa,b=pr[:,index].astype(float)
                matrix=np.array([[1/k+aa,b],[aa,b]])
                # Arbitrary nonzero q,m_old and both output basis bars.
                old=np.array([.713,-.419]);v=matrix@old
                for bar in (np.array([1.,0.]),np.array([0.,1.]),np.array([.43,-.91])):
                    q=np.zeros((4,n,n));p=np.zeros((8,n,n));q[0,j,i]=bar[0];p[kind,j,i]=bar[1]
                    out=blocks(lib,c,cfg,1,kind=kind,q=q,p=p)
                    got=np.array([out[2][0,j,i],out[1][kind,j,i]])
                    tight(got,matrix.T@bar);assert abs(v@bar-old@got)<=5e-13*max(np.linalg.norm(v)*np.linalg.norm(bar),1)
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('surface',[0,1])
def test_derivative_velocity_stress_transposes(jt,surface):
    _,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(surface,nx=5,ny=5,nt=1,cpml=False);c=create(lib,cfg)
    rng=np.random.default_rng(931)
    try:
        coef=float(np.float32(cfg.dt/cfg.dh));bg=rng.normal(size=(4,5,5)).astype(np.float32)
        alpha=s.surface_coefficients(a['l'][0].astype(float),a['m'][0].astype(float))[0].astype(np.float32).astype(float)
        h=float(np.float32(1)/np.float32(coef))
        for kind in range(4):
            for field,name in enumerate(('vx','vy','sxx','syy','sxy')):
                if surface and kind>=2 and field==2:continue
                def action(z):
                    if kind<2:return coef*s.dx(z,('xb','xf')[kind])
                    ext=s.extend(z,name,alpha=alpha,qxx=np.zeros_like(z),qyx=np.zeros_like(z),h_over_dt=h) if surface else np.concatenate((z[-2:],z,z[:2]))
                    # This block is mirror+FD, before the separate projection
                    # transpose. extend() also projects syy; undo only that
                    # extra forward stage here. Projection has its own gate.
                    if surface and field==3:ext[2]=z[0]
                    return coef*s.dy_extended(ext,('yb','yf')[kind-2])
                matrix=s.matrix_of(action,(5,5))
                for index in range(25):
                    q=np.zeros((4,5,5));q[0].flat[index]=1
                    f,_,qq,_=blocks(lib,c,cfg,2,kind,field,q=q,bg=bg)
                    tight(f[field,2:-2,2:-2],matrix.T[:,index].reshape(5,5))
                    assert not np.any(f[field,:2]) and not np.any(f[field,-2:])
                    assert not np.any(f[field,:,:2]) and not np.any(f[field,:,-2:])
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

def test_surface_ghost_mirror_projection_and_material_basis(jt):
    _,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(1,nx=5,ny=5,nt=1,cpml=False);c=create(lib,cfg)
    cfg.dt=np.float32(.000318);cfg.dh=np.float32(11.7)
    check(lib,lib.denise_cuda_m9_destroy(C.byref(c)));c=create(lib,cfg)
    coeff=np.float32(cfg.dt)/np.float32(cfg.dh);h=float(np.float32(1)/coeff)
    alpha,A,al,am,bl,bm=s.surface_coefficients(a['l'][0].astype(float),a['m'][0].astype(float))
    alpha=alpha.astype(np.float32).astype(float);A=A.astype(np.float32).astype(float)
    bg=np.random.default_rng(932).normal(size=(4,5,5)).astype(np.float32)
    try:
        for field,name in ((0,'vx'),(1,'vy'),(3,'syy'),(4,'sxy')):
            for m in (1,2):
                for i in range(5):
                    f=np.zeros((5,9,9));f[field,2-m,i+2]=1
                    got=blocks(lib,c,cfg,3 if field<2 else 4,field=field,f=f,bg=bg)
                    e=np.zeros((9,5));e[2-m,i]=1
                    physical,x,y,abar=s.extend_t(e,name,alpha=alpha,qxx=bg[0].astype(float),h_over_dt=h)
                    tight(got[0][field,2:-2,2:-2],physical)
                    assert not np.any(got[0][field,:2])
                    if field<2:
                        tight(got[2][0],x);tight(got[2][1],y)
                        if field==1:
                            tight(got[3][0,0],al*abar);tight(got[3][1,0],am*abar)
        f=np.ones((5,9,9));got=blocks(lib,c,cfg,5,f=f)[0]
        f[3,2,2:-2]=0;np.testing.assert_array_equal(got,f)
        for i in range(5):
            f=np.zeros((5,9,9));f[2,2,i+2]=1
            q=blocks(lib,c,cfg,8,f=f)[2]
            expected=np.zeros_like(q);expected[0,0,i]=A[i];tight(q,expected)
        # Every stress basis tests bulk/direct delta-A and all four harmonic contributors.
        mu=a['m'].astype(float);mc=e1.ref._harmonic_mu(mu) if hasattr(e1.ref,'_harmonic_mu') else s.corner(mu)
        # Corner map is rounded by production; get that exact coefficient.
        maps=np.empty(5*81,np.float32);prof=np.empty(60,np.float32);src=np.empty(1,np.float32);geo=np.empty(18,np.int32)
        check(lib,lib.denise_cuda_m9_test_static(c,fp(maps),fp(prof),fp(src),e1.ip(geo)))
        mc=maps.reshape(5,9,9)[4,2:-2,2:-2].astype(float)
        for field in (2,3,4):
            for j,i in np.ndindex(5,5):
                f=np.zeros((5,9,9));f[field,j+2,i+2]=1
                if field==3 and j==0:continue  # projection independently gated above
                got=blocks(lib,c,cfg,6,f=f,bg=bg)[3]
                g=np.zeros_like(got);xx,yx,xy,yy=bg[:,j,i].astype(float)
                if field<4:
                    if j==0:g[:,0,i]=[bl[i]*xx,bm[i]*xx]
                    else:g[:,j,i]=[xx+yy,2*(xx if field==2 else yy)]
                else:
                    common=.25*mc[j,i]**2*(yx+xy)
                    for jj,ii in ((j,i),(j,(i+1)%5),((j+1)%5,i),((j+1)%5,(i+1)%5)):
                        g[1,jj,ii]+=common/mu[jj,ii]**2
                tight(got,g)
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('surface',[0,1])
def test_memory_budget_and_invalid_transaction(jt,surface):
    _,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(surface,nx=7,ny=7,nt=3);c=create(lib,cfg)
    try:
        d=e1.Diagnostics();jd=JTDiagnostics()
        check(lib,lib.denise_cuda_m9_diagnostics(c,C.byref(d)));check(lib,lib.denise_cuda_m9_adjoint_diagnostics(c,C.byref(jd)))
        n=cfg.nx*cfg.ny;p=(cfg.nx+4)*(cfg.ny+4);tr=cfg.nt*cfg.receiver_count
        old=124*p+24*(cfg.nx+cfg.ny)+4*cfg.nt+12*cfg.receiver_count+16*tr+16*cfg.nt*n+28*n
        expected=old+(8-old%8)%8+40*p+112*n+8*tr
        assert d.mandatory_bytes==d.owned_bytes==expected
        assert (jd.field_bytes,jd.cpml_bytes,jd.image_bytes,jd.workspace_bytes,jd.data_bytes)==(40*p,64*n,16*n,32*n,8*tr)
        other=P();assert lib.denise_cuda_m9_create_migration(C.byref(cfg),C.byref(e1.Options(0,expected-1,0)),C.byref(other))!=0
        assert not other
        r=np.ones((cfg.nt,cfg.receiver_count,2),np.float32)
        out=np.full((2,cfg.ny,cfg.nx),-123.456)
        sentinel=out.tobytes()
        assert lib.denise_cuda_m9_image_download(c,dp(out[0]),dp(out[1]),n)!=0;assert out.tobytes()==sentinel
        check(lib,lib.denise_cuda_m9_prepare(c));r.flat[-1]=np.nan
        assert lib.denise_cuda_m9_migrate(c,fp(r),r.size,dp(out[0]),dp(out[1]),n)!=0
        assert out.tobytes()==sentinel
        check(lib,lib.denise_cuda_m9_adjoint_diagnostics(c,C.byref(jd)));assert not jd.valid
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('surface',[0,1])
def test_new_jt_download_and_migration_faults(jt,surface):
    _,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(surface,nx=5,ny=5,nt=2,cpml=False)
    r=np.ones((cfg.nt,cfg.receiver_count,2),np.float32);n=cfg.nx*cfg.ny
    counts={}
    # Arm only after construction/preparation: NEW reverse, image and wrapper sites.
    for operation in ('jt-download','migration'):
        def attempt(at):
            lib.denise_cuda_m9_fault(0);c=create(lib,cfg);check(lib,lib.denise_cuda_m9_prepare(c))
            out=np.full((2,5,5),-197.125);sentinel=out.tobytes()
            lib.denise_cuda_m9_fault(at)
            if operation=='migration':
                rc=lib.denise_cuda_m9_migrate(c,fp(r),r.size,dp(out[0]),dp(out[1]),n)
            else:
                rc=lib.denise_cuda_m9_apply_jt(c,fp(r),r.size)
                if not rc:rc=lib.denise_cuda_m9_image_download(c,dp(out[0]),dp(out[1]),n)
            sites=e1.ledger(lib)[3]
            if at:
                assert rc!=0 and out.tobytes()==sentinel
                d=JTDiagnostics();check(lib,lib.denise_cuda_m9_adjoint_diagnostics(c,C.byref(d)));assert not d.valid
            else:check(lib,rc)
            check(lib,lib.denise_cuda_m9_destroy(C.byref(c)));assert e1.ledger(lib)[:3]==(0,0,0)
            return sites
        sites=attempt(0)
        for at in range(1,sites+1):attempt(at)
        counts[operation]=sites
    lib.denise_cuda_m9_fault(0);RECORDS.append({'faults':f'fs{surface}',**counts})

def test_performance_diagnostics(jt):
    _,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(1,nx=97,ny=79,nt=120);c=create(lib,cfg)
    rng=np.random.default_rng(933);r=rng.normal(size=(cfg.nt,cfg.receiver_count,2)).astype(np.float32)
    def timed(f):
        t=time.perf_counter();f();return (time.perf_counter()-t)*1000
    try:
        prep=timed(lambda:check(lib,lib.denise_cuda_m9_prepare(c)));dl,dm=e2.direction(cfg,a)
        j=timed(lambda:e2.gpu_j(lib,c,cfg,dl,dm))
        rev=timed(lambda:check(lib,lib.denise_cuda_m9_apply_jt(c,fp(r),r.size)))
        down=timed(lambda:download(lib,c,cfg));out=images(cfg)
        migration=timed(lambda:check(lib,lib.denise_cuda_m9_migrate(c,fp(r),r.size,dp(out[0]),dp(out[1]),cfg.nx*cfg.ny)))
        d=e1.Diagnostics();check(lib,lib.denise_cuda_m9_diagnostics(c,C.byref(d)))
        RECORDS.append({'performance':{'prepare_ms':prep,'j_including_download_ms':j,'jt_ms':rev,'image_download_ms':down,'migration_ms':migration,'mandatory_bytes':d.mandatory_bytes}})
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('cpml',[False,True])
def test_complete_fp64_step_dense_transpose(jt,surface,cpml):
    """Independent FP64 linear graph with frozen FP32 coefficient nodes.

    A basis-built forward matrix is transposed by NumPy, never by a second
    implementation of the production reverse. Arbitrary all-eight memory bars
    and the receiver timing participate, as do every raw material column.
    """
    _,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(surface,nx=5,ny=5,nt=1,cpml=cpml)
    if cpml:cfg.fw=1
    c=create(lib,cfg);n=25;rng=np.random.default_rng(934)
    try:
        maps=np.empty(5*81,np.float32);pr=np.empty(60,np.float32);source=np.empty(1,np.float32);geo=np.empty(18,np.int32)
        check(lib,lib.denise_cuda_m9_test_static(c,fp(maps),fp(pr),fp(source),e1.ip(geo)))
        lam,mu,rx,ry,mc=maps.reshape(5,9,9)[:,2:-2,2:-2].astype(float)
        xx,xh,yy,yh=pr.reshape(4,3,5).astype(float)
        profiles=(xh,yy,xx,yh,xx,xh,yh,yy)
        coef=float(np.float32(np.float32(cfg.dt)/np.float32(cfg.dh)))
        h=float(np.float32(1)/np.float32(coef))
        alpha,A,al,am,bl,bm=s.surface_coefficients(lam[0],mu[0])
        alpha=alpha.astype(np.float32).astype(float);A=A.astype(np.float32).astype(float)
        bg=rng.normal(size=(4,5,5)).astype(np.float32)
        def forward(vector):
            state=vector[:13*n].reshape(13,5,5).copy()
            dl,dm=vector[13*n:].reshape(2,5,5)
            if surface:state[3,0]=0
            def pml(raw,kind):
                k,aa,b=profiles[kind];p=np.array((k,aa,b))
                if kind in (1,3,6,7):p=p[:,:,None]
                new=p[2]*state[5+kind]+p[1]*raw
                state[5+kind]=new
                return raw/p[0]+new
            def deriv(z,kind,field,qxx=None,qyx=None):
                if kind<2:return coef*s.dx(z,('xb','xf')[kind])
                ext=s.extend(z,field,alpha=alpha,qxx=qxx,qyx=qyx,h_over_dt=h) if surface else np.concatenate((z[-2:],z,z[:2]))
                # syy projection is already a distinct input overwrite.
                return coef*s.dy_extended(ext,('yb','yf')[kind-2])
            corr=[pml(deriv(state[f],d,name),k) for k,f,d,name in (
                (0,2,1,'sxx'),(1,4,2,'sxy'),(2,4,0,'sxy'),(3,3,3,'syy'))]
            state[0]+=rx*(corr[0]+corr[1]);state[1]+=ry*(corr[2]+corr[3])
            sample=np.array([[state[0,j,i],state[1,j,i]] for i,j in zip(a['ri'],a['rj'])])
            qxx=pml(deriv(state[0],0,'vx'),4);qyx=pml(deriv(state[1],1,'vy'),5)
            qxy=pml(deriv(state[0],3,'vx',qxx,qyx),6)
            rawyy=deriv(state[1],2,'vy',qxx,qyx)
            if surface:
                ghost=np.zeros((9,5));da=al*dl[0]+am*dm[0]
                for m in (1,2):ghost[2-m]=(2*m-1)*h*da*bg[0,0]
                rawyy+=coef*s.dy_extended(ghost,'yb')
            qyy=pml(rawyy,7)
            incx=lam*(qxx+qyy)+2*mu*qxx
            incy=lam*(qxx+qyy)+2*mu*qyy
            incx+=dl*(bg[0].astype(float)+bg[3])+2*dm*bg[0]
            incy+=dl*(bg[0].astype(float)+bg[3])+2*dm*bg[3]
            if surface:incx[0]=A*qxx[0]+(bl*dl[0]+bm*dm[0])*bg[0,0]
            state[2]+=incx;state[3]+=incy
            if surface:state[3,0]=0
            terms=dm/(mu*mu)
            dcorner=.25*mc*mc*(terms+np.roll(terms,-1,1)+np.roll(terms,-1,0)+np.roll(np.roll(terms,-1,1),-1,0))
            state[4]+=mc*(qyx+qxy)+dcorner*(bg[1].astype(float)+bg[2])
            return np.concatenate((state.ravel(),sample.ravel()))
        matrix=s.matrix_of(forward,(15*n,))
        # Random bars followed by all state/receiver basis cotangents.
        bar=rng.normal(size=matrix.shape[0]);worst=0
        for index in range(-1,matrix.shape[0]):
            b=bar if index==-1 else np.eye(1,matrix.shape[0],index).ravel()
            state=b[:13*n].reshape(13,5,5)
            f=np.zeros((5,9,9));f[:,2:-2,2:-2]=state[:5]
            r=b[13*n:].reshape(1,cfg.receiver_count,2).astype(np.float32)
            # Random receiver input is explicitly FP32 ABI.
            b=b.copy();b[13*n:]=r.ravel()
            out=blocks(lib,c,cfg,10,f=f,p=state[5:].copy(),bg=bg,r=r)
            got=np.concatenate((out[0][:,2:-2,2:-2].ravel(),out[1].ravel(),out[3].ravel()))
            want=matrix.T@b
            err=np.linalg.norm(got-want)/max(np.linalg.norm(want),1e-300)
            assert err<=5e-13,(surface,cpml,index,err);worst=max(worst,err)
        RECORDS.append({'dense_step':f'fs{surface}/cpml{int(cpml)}','forward_columns':375,
                        'reverse_basis_rows':matrix.shape[0],'arbitrary':1,'worst_rel_l2':worst})
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('surface',[0,1])
def test_new_migration_constructor_failure_ownership(jt,surface):
    _,libs=jt;lib=libs['nofma'];cfg,a=e2.fixture(surface,nx=7,ny=7,nt=1)
    lib.denise_cuda_m9_fault(0);c=create(lib,cfg);sites=e1.ledger(lib)[3]
    check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
    for at in range(1,sites+1):
        lib.denise_cuda_m9_fault(at);c=P()
        assert lib.denise_cuda_m9_create_migration(C.byref(cfg),C.byref(e1.Options()),C.byref(c))!=0
        check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
        assert e1.ledger(lib)[:3]==(0,0,0)
    lib.denise_cuda_m9_fault(0);RECORDS.append({'new_constructor_faults':f'fs{surface}','sites':sites})
