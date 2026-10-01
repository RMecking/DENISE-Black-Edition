"""Additive M9e-2 CPU/GPU gates. Frozen M9b/M9d3 oracles are unchanged."""
from __future__ import annotations
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import pytest
from tests.physics import test_m9e1_cuda_elastic_psv_forward as old
from tests.utilities import elastic_psv_free_surface_reference as surf
from tests.physics.test_m9d3_surface_blocks import surface_blocks,born_library

ROOT=Path(__file__).resolve().parents[2]
F,P,Z=old.F,old.P,old.Z
fp,check,same,metrics=old.fp,old.check,old.same,old.metrics
RECORDS=[]
class BornDiagnostics(C.Structure):
    _fields_=[(n,Z) for n in ('physical_bytes','cpml_bytes','direction_bytes','corner_bytes','operand_bytes','data_bytes')]+[('valid',C.c_int),('elapsed_ms',C.c_float)]

@pytest.fixture(scope='session')
def backends(tmp_path_factory):
    build=Path(os.environ['DENISE_M9E2_BUILD']) if 'DENISE_M9E2_BUILD' in os.environ else tmp_path_factory.mktemp('m9e2-build')
    if 'DENISE_M9E2_BUILD' not in os.environ:
        subprocess.run(['make','-C',str(ROOT/'src'),'cuda_m9_elastic_psv','cuda_m9_elastic_psv_nofma',
            'NVCC='+os.environ.get('NVCC','/usr/local/cuda-12.8/bin/nvcc'),'CUDA_ARCHS=86','M9_CUDA_BUILD_DIR='+str(build)],check=True)
        subprocess.run(['cc','-std=c99','-O3','-Wall','-Wextra','-Werror','-pedantic','-fPIC','-shared',
            '-I'+str(ROOT/'include'),str(ROOT/'tests/utilities/m9e2_cuda_free_surface_born_harness.c'),'-lm','-o',str(build/'cpu_e2.so')],check=True)
    cpu=C.CDLL(str(build/'cpu_e2.so'))
    cpu.m9e1_cpu_view.argtypes=[C.POINTER(old.Config),F,F,F,F,F]
    cpu.m9e1_cpu_static.argtypes=[C.POINTER(old.Config),F,F]
    cpu.m9e2_cpu_j.argtypes=[C.POINTER(old.Config)]+[F]*6
    cpu.m9e2_cpu_ghost.argtypes=[C.POINTER(old.Config),C.c_int]+[F]*6
    cpu.m9e2_cpu_data.argtypes=[C.POINTER(old.Config),F]
    libs={}
    for mode in ('nofma','fma'):
        lib=C.CDLL(str(build/('libdenise_m9_cuda_nofma.so' if mode=='nofma' else 'libdenise_m9_cuda.so')))
        lib.denise_cuda_m9_last_error.restype=C.c_char_p
        for n in ('create','create_full'):getattr(lib,'denise_cuda_m9_'+n).argtypes=[C.POINTER(old.Config),C.POINTER(old.Options),C.POINTER(P)]
        lib.denise_cuda_m9_destroy.argtypes=[C.POINTER(P)]
        lib.denise_cuda_m9_prepare.argtypes=[P]
        lib.denise_cuda_m9_nonlinear.argtypes=[P,F,F]
        lib.denise_cuda_m9_download.argtypes=[P,F,F,F,F]
        lib.denise_cuda_m9_diagnostics.argtypes=[P,C.POINTER(old.Diagnostics)]
        lib.denise_cuda_m9_born_diagnostics.argtypes=[P,C.POINTER(BornDiagnostics)]
        lib.denise_cuda_m9_apply_j.argtypes=[P,F,F,Z]
        lib.denise_cuda_m9_born_download.argtypes=[P]+[F]*5
        lib.denise_cuda_m9_test_step.argtypes=[P,F,F]
        lib.denise_cuda_m9_test_evolve.argtypes=[P,F]
        lib.denise_cuda_m9_test_surface.argtypes=[P,C.c_int]+[F]*5
        lib.denise_cuda_m9_test_static.argtypes=[P,F,F,F,old.I]
        lib.denise_cuda_m9_fault.argtypes=[Z]
        lib.denise_cuda_m9_ledger.argtypes=[C.POINTER(Z)]*4
        libs[mode]=lib
    yield cpu,libs
    if os.environ.get('DENISE_M9E2_EVIDENCE'):Path(os.environ['DENISE_M9E2_EVIDENCE']).write_text(json.dumps(RECORDS,indent=2))

def fixture(surface=1,nt=32,nx=17,ny=13,cpml=True,edge='left'):
    cfg,a=old.fixture('heterogeneous',nx=nx,ny=ny,nt=nt)
    cfg.free_surface=surface;cfg.source_j=1 if surface else ny//2
    cfg.source_i=1 if edge=='left' else nx-2
    if not cpml:cfg.cpml_enabled=0;cfg.fw=0
    a['ri'][:]=[nx-1,1,0,1,nx//2,nx-2];a['rj'][:]=[0,1,0,1,2,ny-1]
    return cfg,a

def create(lib,cfg,options=None):
    c=P();check(lib,lib.denise_cuda_m9_create_full(C.byref(cfg),C.byref(options or old.Options()),C.byref(c)));return c

def direction(cfg,a,kind='joint'):
    y,x=np.mgrid[:cfg.ny,:cfg.nx]
    dl=np.asarray(.013*a['l']*(np.sin(.79*x+.31*y)-.3*np.cos(.27*x-.7*y)),np.float32)
    dm=np.asarray(.017*a['m']*(np.cos(.53*x-.41*y)-.4*np.sin(.63*x+.17*y)),np.float32)
    if kind=='lambda':dm.fill(0)
    if kind=='mu':dl.fill(0)
    return dl,dm

def born_outputs(cfg):
    return (np.empty((cfg.nt,cfg.receiver_count,2),np.float32),np.empty((5,cfg.ny,cfg.nx),np.float32),
        np.empty((8,cfg.ny,cfg.nx),np.float32),np.empty((4,cfg.ny,cfg.nx),np.float32),np.empty((cfg.ny,cfg.nx),np.float32))

def cpu_j(cpu,cfg,dl,dm):
    d,f,p,q,mc=born_outputs(cfg);s=np.empty((13,cfg.ny,cfg.nx),np.float32)
    assert cpu.m9e2_cpu_j(C.byref(cfg),fp(dl),fp(dm),fp(d),fp(s),fp(q),fp(mc))==0
    f[:]=s[:5];p[:]=s[5:];return d,f,p,q,mc

def gpu_j(lib,c,cfg,dl,dm):
    check(lib,lib.denise_cuda_m9_apply_j(c,fp(dl),fp(dm),cfg.nx*cfg.ny))
    out=born_outputs(cfg);check(lib,lib.denise_cuda_m9_born_download(c,*map(fp,out)));return out

def compare_j(label,got,expected,exact):
    r={'case':label,'arrays':{}}
    for k,n in enumerate(('data','fields','psi','final_q','corner')):
        z=metrics(got[k],expected[k]);r['arrays'][n]=z
        if exact:assert same(got[k],expected[k]),(label,n,z)
        if k==0:assert z['rel_l2']<=1e-5 and z['peak_normalized']<=4e-5,(label,z)
    RECORDS.append(r)

@pytest.mark.parametrize('mode',['nofma','fma'])
@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('cpml',[False,True])
@pytest.mark.parametrize('edge',['left','right'])
def test_forward_and_full_j(backends,mode,surface,cpml,edge):
    cpu,libs=backends;lib=libs[mode];cfg,a=fixture(surface,cpml=cpml,edge=edge);c=create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c));bg=old.download(lib,c,cfg)
        label=f'{mode}/fs{surface}/cpml{int(cpml)}/{edge}'
        old.compare(label,bg,old.cpu_run(cpu,cfg),exact=mode=='nofma');RECORDS.append(old.EVIDENCE[-1])
        for kind in ('lambda','mu','joint'):
            dl,dm=direction(cfg,a,kind);got=gpu_j(lib,c,cfg,dl,dm)
            compare_j(label+'/'+kind,got,cpu_j(cpu,cfg,dl,dm),mode=='nofma')
            repeated=gpu_j(lib,c,cfg,dl,dm)
            assert all(same(x,y) for x,y in zip(got,repeated))
            assert all(same(x,y) for x,y in zip(bg,old.download(lib,c,cfg)))
        assert same(bg[0][:,1],bg[0][:,3]) and np.count_nonzero(bg[0][0])==0
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
    assert old.ledger(lib)[:3]==(0,0,0)

@pytest.mark.parametrize('mode',['nofma','fma'])
@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('kind',['lambda','mu','joint'])
def test_gpu_production_fd(backends,mode,surface,kind):
    _,libs=backends;lib=libs[mode];cfg,a=fixture(surface,nt=48);c=create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c));dl,dm=direction(cfg,a,kind);j=gpu_j(lib,c,cfg,dl,dm)[0]
        vals=[]
        for sign in (1,-1):
            l=np.asarray(a['l']+sign*.05*dl,np.float32);m=np.asarray(a['m']+sign*.05*dm,np.float32)
            check(lib,lib.denise_cuda_m9_nonlinear(c,fp(l),fp(m)));vals.append(old.download(lib,c,cfg)[0])
        fd=(vals[0].astype(float)-vals[1])/0.1
        z=metrics(j,fd);RECORDS.append({'case':f'FD/{mode}/fs{surface}/{kind}','epsilon':.05,**z})
        assert z['rel_l2']<=.004,z
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

def config_from_exp(exp,surface=1):
    cfg,a=fixture(surface,nt=exp.nt,nx=exp.nx,ny=exp.ny,cpml=exp.cpml)
    a['l']=np.asarray(exp.lam,np.float32);a['m']=np.asarray(exp.mu,np.float32);a['r']=np.asarray(exp.rho,np.float32)
    a['source']=np.asarray(old.ref._source_samples(exp),np.float32)
    a['ri']=np.asarray([i-1 for i,j in exp.receivers],np.int32);a['rj']=np.asarray([j-1 for i,j in exp.receivers],np.int32)
    setattr(cfg,'lambda',fp(a['l']));cfg.mu=fp(a['m']);cfg.rho=fp(a['r']);cfg.source_samples=fp(a['source'])
    cfg.receiver_count=len(a['ri']);cfg.receiver_i=old.ip(a['ri']);cfg.receiver_j=old.ip(a['rj'])
    cfg.source_i=exp.sources[0][0]-1;cfg.source_j=exp.sources[0][1]-1
    cfg.dh=exp.dh;cfg.dt=exp.dt;cfg.fw=exp.fw
    cfg.pml_reflection=exp.pml_reflection;cfg.pml_power=exp.pml_power;cfg.pml_kmax=exp.pml_kmax;cfg.pml_fpml=exp.pml_fpml
    cfg.pml_damping_speed=float(np.sqrt(np.max((exp.lam+2*exp.mu)/exp.rho)))
    return cfg,a

@pytest.mark.parametrize('mode',['nofma','fma'])
@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('cpml',[False,True])
def test_independent_full_j(backends,mode,surface,cpml):
    cpu,libs=backends;lib=libs[mode];exp=surf.fixture(cpml=cpml);cfg,a=config_from_exp(exp,surface);c=create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c));bg=old.download(lib,c,cfg)
        if surface:
            data,tapes,states=surf.forward(exp)
            expected=(data,states[-1][:5],states[-1][5:],np.asarray([t.strain for t in tapes]))
        else:expected=old.oracle(cfg,a)
        old.compare(f'oracle/{mode}/fs{surface}/cpml{int(cpml)}',bg,expected);RECORDS.append(old.EVIDENCE[-1])
        for kind in ('lambda','mu','joint'):
            dl,dm=direction(cfg,a,kind);got=gpu_j(lib,c,cfg,dl,dm)
            if surface:reference=surf.born(exp,tapes,dl.astype(float),dm.astype(float))
            else:
                from dataclasses import replace
                # Fixed independent CPML operator, prepared FP32 source represented
                # by the existing M9b source wavelet's canonical samples.
                run=old.ref.nonlinear_forward(exp,save_strain=True)
                reference=old.ref.born_forward(exp,run,dl.astype(float),dm.astype(float))
            z=metrics(got[0],reference);RECORDS.append({'case':f'J-oracle/{mode}/fs{surface}/{kind}/cpml{int(cpml)}',**z})
            assert z['rel_l2']<=1e-5 and z['peak_normalized']<=4e-5,z
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('cpml',[False,True])
def test_arbitrary_and_dense_state_basis(backends,cpml):
    cpu,libs=backends;lib=libs['nofma'];exp=surf.fixture(nx=5,ny=5,nt=1,cpml=False,source=(2,2))
    if cpml:
        from dataclasses import replace
        exp=replace(exp,cpml=True,fw=1)
    cfg,a=config_from_exp(exp);a['source'].fill(0);c=create(lib,cfg);op=surf.SurfaceOperator(exp)
    shape=(13,5,5);actual=[];expected=[]
    try:
        vectors=[np.random.default_rng(291).normal(size=shape).astype(np.float32)*.001]
        for k in range(np.prod(shape)):
            e=np.zeros(shape,np.float32);e.flat[k]=1;vectors.append(e)
        for index,state in enumerate(vectors):
            check(lib,lib.denise_cuda_m9_test_step(c,fp(state),F()));got=old.download(lib,c,cfg)
            want=old.cpu_run(cpu,cfg,state)
            assert all(same(x,y) for x,y in zip(got,want)),index
            z,tape=op.step(state.astype(float));r=np.concatenate((z.ravel(),tape.sample.ravel(),np.asarray(tape.strain).ravel()))
            v=np.concatenate((got[1].ravel(),got[2].ravel(),got[0].ravel(),got[3].ravel())).astype(float)
            actual.append(v);expected.append(r)
        z=metrics(np.asarray(actual),np.asarray(expected));assert z['rel_l2']<=2e-6 and z['peak_normalized']<=8e-6,z
        RECORDS.append({'case':f'state-basis/cpml{int(cpml)}','columns':325,'arbitrary':1,**z})
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

def test_surface_blocks_basis_ghosts_and_h(backends):
    cpu,libs=backends;lib=libs['nofma'];cfg,a=fixture(1,nt=1,nx=5,ny=5,cpml=False)
    # These rounded inputs distinguish reciprocal(float(DT/DH)) from DH/DT.
    cfg.dt=np.float32(.000318);cfg.dh=np.float32(11.7)
    coefficient=np.float32(np.float32(cfg.dt)/np.float32(cfg.dh));h=np.float32(1)/coefficient
    alternate=np.float32(np.float64(cfg.dh)/np.float64(cfg.dt));assert h.tobytes()!=alternate.tobytes()
    c=create(lib,cfg);rng=np.random.default_rng(781);q=rng.normal(size=(4,5,5)).astype(np.float32)
    dl,dm=direction(cfg,a);bg=rng.normal(size=q.shape).astype(np.float32)
    coeff=surf.surface_coefficients(a['l'][0].astype(float),a['m'][0].astype(float))
    try:
        for mode in (0,1,2):
            field=rng.normal(size=(5,5,5)).astype(np.float32);padded=np.zeros((5,9,9),np.float32);padded[:,2:7,2:7]=field
            check(lib,lib.denise_cuda_m9_test_surface(c,mode,fp(padded),fp(q),fp(bg) if mode==1 else F(),fp(dl),fp(dm)))
            if mode==0:
                assert np.count_nonzero(padded[3,2,2:7])==0
                for m in (1,2):
                    assert same(padded[3,2-m,2:7],-field[3,m])
                    assert same(padded[4,2-m,2:7],-field[4,m-1])
            if mode==1:
                for k in (0,1):
                    expected=np.empty((2,5),np.float32)
                    assert cpu.m9e2_cpu_ghost(C.byref(cfg),k,fp(field[k]),fp(q),fp(bg),fp(dl),fp(dm),fp(expected))==0
                    assert same(padded[k,:2,2:7],expected)
                slope=np.zeros(5,np.float32)
                for row,weight in enumerate((35/16,-35/16,21/16,-5/16)):
                    slope+=np.float32(weight)*q[1,row]
                wrong=np.stack((field[0,2]+np.float32(4*alternate)*slope,field[0,1]+np.float32(2*alternate)*slope))
                assert not same(padded[0,:2,2:7],wrong), 'DH/DT reassociation must be numerically distinguishable'
                # Independent ghost matrix checks without a material perturbation.
                for kind,name in ((0,'vx'),(1,'vy')):
                    for index in range(3*25):
                        fields=np.zeros((5,9,9),np.float32);qs=np.zeros_like(q)
                        k,p=divmod(index,25)
                        if k==0:fields[kind,2:7,2:7].flat[p]=1
                        else:qs[k-1].flat[p]=1
                        check(lib,lib.denise_cuda_m9_test_surface(c,1,fp(fields),fp(qs),F(),F(),F()))
                        ext=surf.extend(fields[kind,2:7,2:7].astype(float),name,alpha=coeff[0].astype(np.float32).astype(float),
                            qxx=qs[0].astype(float),qyx=qs[1].astype(float),h_over_dt=float(h))
                        z=metrics(fields[kind,:2,2:7],ext[:2]);assert z['rel_l2']<=2e-6 and z['peak_normalized']<=8e-6,z
                RECORDS.append({'case':'canonical-h-discriminator','h':float(h),'DH_over_DT':float(alternate),'ghost_bytes_cpu_equal':True})
            if mode==2:
                expected=field[2,0]+coeff[1].astype(np.float32)*q[0,0]
                assert same(padded[2,2,2:7],expected) and np.count_nonzero(padded[3,2,2:7])==0
        RECORDS.append({'case':'surface-blocks','ghost_basis_columns':150,'projection_mirrors_A':'PASS'})
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('surface',[0,1])
def test_material_basis_all_contributors(backends,surface):
    cpu,libs=backends;lib=libs['nofma'];exp=surf.fixture(nx=5,ny=5,nt=16,cpml=False,source=(2,2));cfg,a=config_from_exp(exp,surface);c=create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c))
        tapes=surf.forward(exp)[1] if surface else None
        for kind in ('lambda','mu'):
            for index in range(25):
                dl=np.zeros((5,5),np.float32);dm=np.zeros_like(dl);(dl if kind=='lambda' else dm).flat[index]=1e7
                got=gpu_j(lib,c,cfg,dl,dm);expected=cpu_j(cpu,cfg,dl,dm)
                assert all(same(x,y) for x,y in zip(got,expected)),(kind,index)
                if kind=='mu':
                    independent=surf.corner_jvp(a['m'].astype(float),dm.astype(float))
                    z=metrics(got[4],independent);assert z['rel_l2']<=2e-6,z
                    assert np.count_nonzero(got[4])==4
                if surface:
                    reference=surf.born(exp,tapes,dl.astype(float),dm.astype(float));z=metrics(got[0],reference)
                    assert z['rel_l2']<=1e-5 and z['peak_normalized']<=4e-5,z
        RECORDS.append({'case':f'material-basis/fs{surface}','columns':50,'cpu_byte_identity':True,'oracle':True})
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('mode',['nofma','fma'])
def test_canonical_normal_p_packet(backends,mode):
    from dataclasses import replace
    cpu,libs=backends;lib=libs[mode];exp=surf.fixture(nx=8,ny=96,nt=450,cpml=False,source=(4,30))
    exp=replace(exp,lam=np.full_like(exp.lam,6.44e9),mu=np.full_like(exp.mu,5.78e9),rho=np.full_like(exp.rho,2000.),receivers=((4,13),))
    cfg,a=config_from_exp(exp);a['source'].fill(0);state=np.zeros((13,96,8),np.float32)
    sy=np.exp(-(((np.arange(96)+.5)*10-285)/35)**2)
    vy=np.exp(-(((np.arange(96)+1)*10-285-3000*exp.dt/2)/35)**2)/(2000*3000)
    state[3]=sy[:,None];state[2]=(exp.lam/(exp.lam+2*exp.mu))*sy[:,None];state[1]=vy[:,None]
    c=create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_test_evolve(c,fp(state)));got=old.download(lib,c,cfg);expected=old.cpu_run(cpu,cfg,state)
        old.compare('normal-P/'+mode,got,expected,exact=mode=='nofma');RECORDS.append(old.EVIDENCE[-1])
        times=(np.arange(exp.nt)+1)*exp.dt;trace=got[0][:,0,1];targets=((285-130)/3000,(285-5+130-5)/3000);picks=[]
        for target in targets:
            inds=np.flatnonzero(abs(times-target)<.015);i=inds[np.argmax(abs(trace[inds]))];picks.append(i)
            assert abs(times[i]-target)<2*exp.dt+.005*target and trace[i]>0
        assert np.max(abs(got[0][:,0,0]))<1e-20
        RECORDS.append({'case':'normal-P-physics/'+mode,'picks_s':times[picks].tolist(),'target_s':targets,'polarity':'positive','peaks':trace[picks].tolist()})
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('mode',['nofma','fma'])
def test_canonical_oblique_controls(backends,born_library,tmp_path,monkeypatch,mode):
    from tests.physics import test_m9d3_surface_reflection as canonical
    cpu,libs=backends;lib=libs[mode]
    class Run:
        def __call__(self,config,data_pointer):
            cfg=C.cast(config,C.POINTER(old.Config)).contents;c=create(lib,cfg)
            want=np.empty((cfg.nt,cfg.receiver_count,2),np.float32)
            assert cpu.m9e2_cpu_data(C.byref(cfg),fp(want))==0
            try:
                check(lib,lib.denise_cuda_m9_prepare(c));got=np.empty_like(want)
                check(lib,lib.denise_cuda_m9_download(c,fp(got),F(),F(),F()))
                z=metrics(got,want);assert z['rel_l2']<=2e-6 and z['peak_normalized']<=8e-6,z
                if mode=='nofma':assert same(got,want)
                RECORDS.append({'case':f'oblique/{mode}/fs{cfg.free_surface}',**z})
                C.memmove(data_pointer,got.ctypes.data,got.nbytes)
            finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
            return 0
    class Facade:m9d3_run=Run()
    canonical.test_m9_forward_oblique_p_p_and_p_sv(Facade(),born_library,tmp_path,monkeypatch)
    RECORDS.append({'case':'oblique-physics/'+mode,'metrics':json.loads((tmp_path/'m9_reflection.json').read_text())})

@pytest.mark.parametrize('surface',[0,1])
def test_memory_profiles_nt1_and_fail_closed(backends,surface):
    cpu,libs=backends;lib=libs['nofma'];cfg,a=fixture(surface,nt=1,nx=5,ny=5,cpml=False);c=create(lib,cfg)
    try:
        d=old.Diagnostics();bd=BornDiagnostics()
        check(lib,lib.denise_cuda_m9_diagnostics(c,C.byref(d)));check(lib,lib.denise_cuda_m9_born_diagnostics(c,C.byref(bd)))
        n,p,t,r=cfg.nx*cfg.ny,(cfg.nx+4)*(cfg.ny+4),cfg.nt,cfg.receiver_count
        formula=124*p+24*(cfg.nx+cfg.ny)+4*t+12*r+16*t*r+16*t*n+28*n
        assert d.mandatory_bytes==d.owned_bytes==formula==old.ledger(lib)[0]
        assert sum(getattr(bd,k) for k,_ in BornDiagnostics._fields_ if k.endswith('_bytes'))==d.workspace_bytes==52*p+28*n+8*t*r
        RECORDS.append({'case':f'memory/fs{surface}','diagnostics':{k:getattr(d,k) for k,_ in old.Diagnostics._fields_},
            'born':{k:getattr(bd,k) for k,_ in BornDiagnostics._fields_},'ledger':old.ledger(lib)[:3]})
        check(lib,lib.denise_cuda_m9_prepare(c));dl,dm=direction(cfg,a);out=gpu_j(lib,c,cfg,dl,dm)
        assert all(np.count_nonzero(x)==0 for x in out[:4])
        assert old.download(lib,c,cfg)[1][2,cfg.source_j,cfg.source_i]==a['source'][0]
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
    opt=old.Options(0,formula-1,0);c=P();assert lib.denise_cuda_m9_create_full(C.byref(cfg),C.byref(opt),C.byref(c))!=0
    assert not c and 'mandatory budget' in lib.denise_cuda_m9_last_error().decode() and old.ledger(lib)[:3]==(0,0,0)
    changes=[('ny',4),('free_surface',2),('nx',2147483647),('nt',0),('receiver_components',1),('mpi_size',2)]
    if surface:changes.append(('source_j',0))
    for key,value in changes:
        bad=old.Config.from_buffer_copy(cfg);setattr(bad,key,value);c=P()
        assert lib.denise_cuda_m9_create_full(C.byref(bad),C.byref(old.Options()),C.byref(c))!=0,key
        assert not c and old.ledger(lib)[:3]==(0,0,0)
    for v in (np.nan,np.inf):
        l=a['l'].copy();l[0,0]=v;bad=old.Config.from_buffer_copy(cfg);setattr(bad,'lambda',fp(l));c=P()
        assert lib.denise_cuda_m9_create_full(C.byref(bad),C.byref(old.Options()),C.byref(c))!=0
        assert not c and old.ledger(lib)[:3]==(0,0,0)
    cfg,a=fixture(surface,nt=3)
    for count,null,nonfinite in ((cfg.nx*cfg.ny-1,False,False),(2**64-1,False,False),(cfg.nx*cfg.ny,True,False),(cfg.nx*cfg.ny,False,True)):
        c=create(lib,cfg)
        try:
            check(lib,lib.denise_cuda_m9_prepare(c));dl,dm=direction(cfg,a)
            if nonfinite:dm[0,0]=np.nan
            assert lib.denise_cuda_m9_apply_j(c,F() if null else fp(dl),fp(dm),count)!=0
            check(lib,lib.denise_cuda_m9_diagnostics(c,C.byref(d)));check(lib,lib.denise_cuda_m9_born_diagnostics(c,C.byref(bd)))
            assert not d.prepared and not bd.valid
            sentinel=np.full((cfg.nt,cfg.receiver_count,2),-19,np.float32)
            assert lib.denise_cuda_m9_born_download(c,fp(sentinel),F(),F(),F(),F())!=0
            assert np.all(sentinel==-19)
        finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
    if surface:
        c=create(lib,cfg)
        try:
            maps=np.empty((5,cfg.ny+4,cfg.nx+4),np.float32);prof=np.empty(6*(cfg.nx+cfg.ny),np.float32)
            source=np.empty(cfg.nt,np.float32);geo=np.empty((cfg.receiver_count,3),np.int32)
            check(lib,lib.denise_cuda_m9_test_static(c,fp(maps),fp(prof),fp(source),old.ip(geo)))
            cm=np.empty_like(maps);cp=np.empty_like(prof);assert cpu.m9e1_cpu_static(C.byref(cfg),fp(cm),fp(cp))==0
            assert same(maps,cm) and same(prof,cp)
            off=6*cfg.nx
            for half in (0,1):
                pr=prof[off+half*3*cfg.ny:off+(half+1)*3*cfg.ny].reshape(3,cfg.ny)
                assert np.all(pr[0,:3]==1) and np.all(pr[1,:3]==0) and np.all(pr[2,:3]==1)
            expected=np.float32(np.exp(-np.pi*cfg.pml_fpml*cfg.dt))
            assert prof[2*cfg.nx+cfg.nx//2]==expected
            RECORDS.append({'case':'surface-profile-identity','horizontal_inactive_b':float(expected),'byte_equal_cpu':True})
        finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.parametrize('surface',[0,1])
def test_exhaustive_faults(backends,surface):
    _,libs=backends;lib=libs['nofma'];cfg,a=fixture(surface,nt=3,nx=7,ny=7);dl,dm=direction(cfg,a)
    counts={}
    def setup(phase):
        c=create(lib,cfg)
        if phase in ('j','download'):check(lib,lib.denise_cuda_m9_prepare(c))
        if phase=='download':gpu_j(lib,c,cfg,dl,dm)
        return c
    for phase in ('create','prepare','nonlinear','j','download','block'):
        def operation(c):
            if phase=='create':return lib.denise_cuda_m9_create_full(C.byref(cfg),C.byref(old.Options()),C.byref(c))
            if phase=='prepare':return lib.denise_cuda_m9_prepare(c)
            if phase=='nonlinear':return lib.denise_cuda_m9_nonlinear(c,fp(a['l']),fp(a['m']))
            if phase=='j':return lib.denise_cuda_m9_apply_j(c,fp(dl),fp(dm),cfg.nx*cfg.ny)
            if phase=='download':return lib.denise_cuda_m9_born_download(c,*map(fp,outs))
            return lib.denise_cuda_m9_test_surface(c,1,fp(padded),fp(q),fp(q),fp(dl),fp(dm))
        if phase=='block' and not surface:continue
        lib.denise_cuda_m9_fault(0);c=P() if phase=='create' else setup(phase)
        outs=born_outputs(cfg);padded=np.zeros((5,cfg.ny+4,cfg.nx+4),np.float32);q=np.zeros((4,cfg.ny,cfg.nx),np.float32)
        lib.denise_cuda_m9_fault(0);check(lib,operation(c));count=old.ledger(lib)[3];counts[phase]=count
        check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
        for at in range(1,count+1):
            lib.denise_cuda_m9_fault(0);c=P() if phase=='create' else setup(phase)
            outs=tuple(np.full_like(x,-91) for x in born_outputs(cfg));padded.fill(-91)
            lib.denise_cuda_m9_fault(at);rc=operation(c)
            assert rc!=0,(phase,at)
            assert 'injected' in lib.denise_cuda_m9_last_error().decode(),(phase,at)
            if c:
                d=old.Diagnostics();bd=BornDiagnostics();check(lib,lib.denise_cuda_m9_diagnostics(c,C.byref(d)))
                check(lib,lib.denise_cuda_m9_born_diagnostics(c,C.byref(bd)))
                assert not d.prepared and not bd.valid
            if phase=='download':assert all(np.all(x==-91) for x in outs)
            if phase=='block':assert np.all(padded==-91)
            check(lib,lib.denise_cuda_m9_destroy(C.byref(c)));assert old.ledger(lib)[:3]==(0,0,0),(phase,at)
        lib.denise_cuda_m9_fault(0)
    RECORDS.append({'case':f'faults/fs{surface}','operation_sites':counts,'sum':sum(counts.values()),'ownership_after_destroy':[0,0,0]})

def test_performance_diagnostic(backends):
    cpu,libs=backends;lib=libs['fma'];records=[]
    for surface in (0,1):
        cfg,a=fixture(surface,nt=120,nx=97,ny=79);dl,dm=direction(cfg,a);start=time.perf_counter();c=create(lib,cfg)
        try:
            t=time.perf_counter();check(lib,lib.denise_cuda_m9_prepare(c));prepare_ms=(time.perf_counter()-t)*1e3
            d=old.Diagnostics();check(lib,lib.denise_cuda_m9_diagnostics(c,C.byref(d)))
            t=time.perf_counter();bg=old.download(lib,c,cfg);download_ms=(time.perf_counter()-t)*1e3
            r={'surface':surface,'prepare_wall_ms':prepare_ms,'forward_resident_ms':d.elapsed_ms,'forward_download_all_ms':download_ms,
                'trajectory_bytes':d.trajectory_bytes,'mandatory_bytes':d.mandatory_bytes,'tangent_workspace_bytes':d.workspace_bytes,'j':{}}
            for kind in ('lambda','mu','joint'):
                dl,dm=direction(cfg,a,kind);t=time.perf_counter();check(lib,lib.denise_cuda_m9_apply_j(c,fp(dl),fp(dm),cfg.nx*cfg.ny));wall=(time.perf_counter()-t)*1e3
                bd=BornDiagnostics();check(lib,lib.denise_cuda_m9_born_diagnostics(c,C.byref(bd)))
                t=time.perf_counter();o=born_outputs(cfg);check(lib,lib.denise_cuda_m9_born_download(c,*map(fp,o)));download=(time.perf_counter()-t)*1e3
                r['j'][kind]={'wall_ms':wall,'resident_ms':bd.elapsed_ms,'download_all_ms':download}
            records.append(r)
        finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
        r['create_prepare_three_j_download_destroy_ms']=(time.perf_counter()-start)*1e3
    RECORDS.append({'case':'performance-diagnostic','runs':records,'surface_forward_overhead_ms':records[1]['forward_resident_ms']-records[0]['forward_resident_ms']})

@pytest.mark.parametrize('mode',['nofma','fma'])
@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('cpml',[False,True])
def test_nonlinear_full_outputs(backends,mode,surface,cpml):
    cpu,libs=backends;lib=libs[mode];cfg,a=fixture(surface,cpml=cpml);c=create(lib,cfg)
    try:
        dl,dm=direction(cfg,a);l=np.asarray(a['l']+dl,np.float32);m=np.asarray(a['m']+dm,np.float32)
        check(lib,lib.denise_cuda_m9_nonlinear(c,fp(l),fp(m)))
        new=old.Config.from_buffer_copy(cfg);setattr(new,'lambda',fp(l));new.mu=fp(m)
        got=old.download(lib,c,cfg);want=old.cpu_run(cpu,new)
        old.compare(f'nonlinear/{mode}/fs{surface}/cpml{int(cpml)}',got,want,exact=mode=='nofma');RECORDS.append(old.EVIDENCE[-1])
        check(lib,lib.denise_cuda_m9_prepare(c));restored=old.download(lib,c,cfg)
        old.compare(f'restored/{mode}/fs{surface}/cpml{int(cpml)}',restored,old.cpu_run(cpu,cfg),exact=mode=='nofma');RECORDS.append(old.EVIDENCE[-1])
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
