"""Scoped compact CUDA replay and canonical MODE=2 integration gates."""
from __future__ import annotations
import ctypes as C
import json
import os
import time
import subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from tests.physics import test_m9e1_cuda_elastic_psv_forward as e1
from tests.physics import test_m9e2_cuda_free_surface_born as e2
from tests.physics import test_m9e3_cuda_jt_migration as e3
from tests.physics.test_m9e2_cuda_free_surface_born import backends
from tests.physics.test_m9e3_cuda_jt_migration import jt
from tests.physics import test_m9c_elastic_psv_migration_driver as driver
from tests.utilities import elastic_psv_free_surface_reference as surf
from tests.physics.test_m9d1_elastic_psv_checkpoint_replay import StorageDiagnostics
F,P,Z=e1.F,e1.P,e1.Z
fp,check=e1.fp,e1.check
RECORDS=[]
class Replay(C.Structure):
    _fields_=[(n,Z) for n in ('checkpoint_values','checkpoint_payload_bytes','checkpoint_metadata_bytes',
        'checkpoint_pointer_bytes','segment_schedule_bytes','segment_operand_bytes','alignment_bytes','retained_bytes',
        'full_operand_bytes','owned_device_bytes','initial_forward_steps','replayed_forward_steps')]+[(n,C.c_int) for n in
        ('requested_segments','effective_segments','checkpoint_count','max_segment_length','selected_replay')]
@pytest.fixture(scope='session')
def replay(jt):
    cpu,libs=jt
    cpu.denise_elastic_psv_born_set_replay_segments.argtypes=[P,C.c_int]
    cpu.denise_elastic_psv_born_storage_diagnostics.argtypes=[P,C.POINTER(StorageDiagnostics)]
    for lib in libs.values():
        lib.denise_cuda_m9_create_replay.argtypes=[C.POINTER(e1.Config),C.POINTER(e1.Options),C.c_int,C.c_int,C.POINTER(P)]
        lib.denise_cuda_m9_replay_diagnostics.argtypes=[P,C.POINTER(Replay)]
        lib.denise_cuda_m9_estimate_replay.argtypes=[Z,Z,Z,Z,C.POINTER(Z),C.POINTER(Replay)]
        lib.denise_cuda_m9_segment_bounds.argtypes=[C.c_int]*3+[C.POINTER(C.c_int)]*2
        lib.denise_cuda_m9_test_replay_probe.argtypes=[P,C.c_int,C.c_int,F,F]
        lib.denise_cuda_m9_test_checkpoints.argtypes=[P,F,Z]
        lib.denise_cuda_m9_replay_fault.argtypes=[Z]
        lib.denise_cuda_m9_migrate_request.argtypes=[C.POINTER(driver.MigrationRequest),C.POINTER(driver.MigrationResult)]
        lib.denise_cuda_m9_migration_result_destroy.argtypes=[C.POINTER(driver.MigrationResult)]
        lib.denise_cuda_m9_migration_last_error.restype=C.c_char_p
    yield cpu,libs
    if os.environ.get('DENISE_M9E4_EVIDENCE'):
        Path(os.environ['DENISE_M9E4_EVIDENCE']).write_text(json.dumps(RECORDS,indent=2))
def create(lib,cfg,segments,automatic=0):
    c=P();check(lib,lib.denise_cuda_m9_create_replay(C.byref(cfg),C.byref(e1.Options()),segments,automatic,C.byref(c)));return c
def diag(lib,c):
    d=Replay();check(lib,lib.denise_cuda_m9_replay_diagnostics(c,C.byref(d)));return d
def probe(lib,c,cfg,segment,time,q=True):
    state=np.empty((13,cfg.ny,cfg.nx),np.float32)
    operands=np.empty((4,cfg.ny,cfg.nx),np.float32) if q else None
    check(lib,lib.denise_cuda_m9_test_replay_probe(c,segment,time,fp(state),fp(operands) if q else F()))
    return state,operands
def payload(lib,c):
    d=diag(lib,c);a=np.empty(d.checkpoint_payload_bytes//4,np.float32)
    check(lib,lib.denise_cuda_m9_test_checkpoints(c,fp(a),a.size));return a
def assert_same(a,b):assert a.tobytes()==b.tobytes()

@pytest.mark.parametrize('mode',['nofma','fma'])
@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('cpml',[False,True])
@pytest.mark.parametrize('segments',[1,2,3,7,32,41,59])
def test_full_replay_bitwise_matrix(replay,mode,surface,cpml,segments):
    cpu,libs=replay;lib=libs[mode];cfg,a=e2.fixture(surface,nt=41,nx=11,ny=9,cpml=cpml)
    full=e3.create(lib,cfg);c=create(lib,cfg,segments)
    try:
        check(lib,lib.denise_cuda_m9_prepare(full));check(lib,lib.denise_cuda_m9_prepare(c))
        fd=e1.download(lib,full,cfg)[0];rd=np.empty_like(fd)
        check(lib,lib.denise_cuda_m9_download(c,fp(rd),F(),F(),F()));assert_same(fd,rd)
        d=diag(lib,c);e=min(segments,cfg.nt);m=(cfg.nt+e-1)//e
        cpu_context=P();assert cpu.denise_elastic_psv_born_create(C.byref(cfg),C.byref(cpu_context))==0
        try:
            assert cpu.denise_elastic_psv_born_set_replay_segments(cpu_context,segments)==0
            assert cpu.denise_elastic_psv_born_prepare(cpu_context,F())==0
            cd=StorageDiagnostics();assert cpu.denise_elastic_psv_born_storage_diagnostics(cpu_context,C.byref(cd))==0
            assert cd.checkpoint_payload_bytes==4*d.checkpoint_values
        finally:cpu.denise_elastic_psv_born_destroy(C.byref(cpu_context))
        assert (d.effective_segments,d.checkpoint_count,d.max_segment_length)==(e,e-1,m)
        assert d.initial_forward_steps==cfg.nt and d.segment_operand_bytes==16*m*cfg.nx*cfg.ny
        assert d.retained_bytes==sum(getattr(d,n) for n in ('checkpoint_payload_bytes','checkpoint_metadata_bytes',
            'checkpoint_pointer_bytes','segment_schedule_bytes','segment_operand_bytes','alignment_bytes'))
        before=payload(lib,c)
        previous=0
        for s in range(e):
            lo,hi=C.c_int(),C.c_int();check(lib,lib.denise_cuda_m9_segment_bounds(cfg.nt,segments,s,C.byref(lo),C.byref(hi)))
            assert lo.value==previous and hi.value>lo.value
            assert (lo.value,hi.value)==(s*cfg.nt//e,(s+1)*cfg.nt//e);previous=hi.value
        assert previous==cfg.nt
        for s in sorted({0,e//2,e-1}):
            start=s*cfg.nt//e;end=(s+1)*cfg.nt//e
            restored=probe(lib,c,cfg,s,-1,False)[0]
            wanted=probe(lib,full,cfg,0,start-1,False)[0]
            assert_same(restored,wanted);assert_same(restored,probe(lib,c,cfg,s,-1,False)[0])
            for t in sorted({start,end-1}):
                for got,want in zip(probe(lib,c,cfg,s,t),probe(lib,full,cfg,0,t)):assert_same(got,want)
        r=np.random.default_rng(904).normal(size=fd.shape).astype(np.float32)
        started=time.perf_counter();expected=e3.apply(lib,full,cfg,r);full_s=time.perf_counter()-started
        started=time.perf_counter();got=e3.apply(lib,c,cfg,r);replay_s=time.perf_counter()-started
        assert_same(got,expected);assert_same(got,e3.apply(lib,c,cfg,r));assert_same(before,payload(lib,c))
        d=diag(lib,c);assert d.replayed_forward_steps==cfg.nt
        want=e3.cpu_jt(cpu,cfg,r)
        metrics=[e3.metrics(got[k],want[k]) for k in (0,1)]
        assert all(z['rel_l2']<=6e-5 for z in metrics)
        dl,dm=e2.direction(cfg,a)
        assert lib.denise_cuda_m9_apply_j(c,fp(dl),fp(dm),cfg.nx*cfg.ny)!=0
        assert 'unsupported' in lib.denise_cuda_m9_last_error().decode()
        RECORDS.append({'case':f'{mode}/FS{surface}/CPML{cpml}/S{segments}',
            'cpu_image_metrics':metrics,'diagnostics':{n:getattr(d,n) for n,_ in Replay._fields_},
            'full_seconds':full_s,'replay_seconds':replay_s})
    finally:
        check(lib,lib.denise_cuda_m9_destroy(C.byref(c)));check(lib,lib.denise_cuda_m9_destroy(C.byref(full)))
    assert e1.ledger(lib)[:3]==(0,0,0)

def test_near_tie_complete_metadata(replay):
    _,libs=replay;lib=libs['fma'];active=(Z*8)();found={}
    for n in range(1,1500):
        d=Replay();check(lib,lib.denise_cuda_m9_estimate_replay(n,1,41,32,active,C.byref(d)))
        margin=d.full_operand_bytes-d.retained_bytes
        if margin in (-8,0,8):found[margin]=(n,d)
    assert set(found)=={-8,0,8}
    for margin,(n,d) in found.items():
        assert d.selected_replay==(margin>0)
        assert d.checkpoint_metadata_bytes>31*32 and d.checkpoint_pointer_bytes==0
        RECORDS.append({'near_tie':margin,'cells':n,'diagnostics':{name:getattr(d,name) for name,_ in Replay._fields_}})

@pytest.mark.parametrize('surface',[0,1])
@pytest.mark.parametrize('cpml',[False,True])
def test_frozen_j_replay_jt_dot(replay,surface,cpml):
    _,libs=replay;lib=libs['nofma'];exp=surf.fixture(cpml=cpml);cfg,a=e2.config_from_exp(exp,surface)
    full=e3.create(lib,cfg);c=create(lib,cfg,7)
    try:
        check(lib,lib.denise_cuda_m9_prepare(full));check(lib,lib.denise_cuda_m9_prepare(c))
        dl,dm=e2.direction(cfg,a);j=e2.gpu_j(lib,full,cfg,dl,dm)[0]
        r=np.random.default_rng(917).normal(size=j.shape).astype(np.float32)
        g=e3.apply(lib,c,cfg,r)
        if surface:
            _,tapes,_=surf.forward(exp);oj=surf.born(exp,tapes,dl.astype(float),dm.astype(float));og=np.stack(surf.adjoint(exp,tapes,r.astype(float)))
        else:
            tapes=e1.ref.nonlinear_forward(exp,save_strain=True);oj=e1.ref.born_forward(exp,tapes,dl.astype(float),dm.astype(float))
            image=e1.ref.born_adjoint(exp,tapes,r.astype(float));og=np.stack((image.image_lambda_raw,image.image_mu_raw))
        lhs=float(np.sum(oj*r));rhs=float(np.sum(dl*og[0])+np.sum(dm*og[1]))
        scale=max(np.linalg.norm(oj)*np.linalg.norm(r),np.linalg.norm(np.stack((dl,dm)))*np.linalg.norm(og),1e-300)
        z=e3._production_dot_metrics(j,r,dl,dm,*g,oj,*og,{'absolute_residual':abs(lhs-rhs),'absolute_ceiling':5e-13*scale})
        e3._assert_production_dot_closes(z);RECORDS.append({'dot':f'FS{surface}/CPML{cpml}',**z})
    finally:
        check(lib,lib.denise_cuda_m9_destroy(C.byref(c)));check(lib,lib.denise_cuda_m9_destroy(C.byref(full)))

@pytest.mark.parametrize('surface',[0,1])
def test_new_replay_operation_faults(replay,surface):
    _,libs=replay;lib=libs['nofma'];cfg,a=e2.fixture(surface,nx=7,ny=7,nt=7);cfg.fw=1
    r=np.ones((cfg.nt,cfg.receiver_count,2),np.float32);sites={}
    for operation in ('create','migration','probe','checkpoints'):
        def attempt(at):
            lib.denise_cuda_m9_fault(0);c=P()
            if operation!='create':
                c=create(lib,cfg,3);check(lib,lib.denise_cuda_m9_prepare(c))
            lib.denise_cuda_m9_replay_fault(at)
            out=np.full((2,cfg.ny,cfg.nx),-193.5);saved=out.tobytes()
            if operation=='create':
                rc=lib.denise_cuda_m9_create_replay(C.byref(cfg),C.byref(e1.Options()),3,0,C.byref(c))
            elif operation=='migration':rc=lib.denise_cuda_m9_migrate(c,fp(r),r.size,e3.dp(out[0]),e3.dp(out[1]),cfg.nx*cfg.ny)
            elif operation=='probe':
                out=np.full((17,cfg.ny,cfg.nx),-193.5,np.float32);saved=out.tobytes()
                rc=lib.denise_cuda_m9_test_replay_probe(c,2,6,fp(out[:13]),fp(out[13:]))
            else:
                d=diag(lib,c);out=np.full(d.checkpoint_payload_bytes//4,-193.5,np.float32);saved=out.tobytes()
                rc=lib.denise_cuda_m9_test_checkpoints(c,fp(out),out.size)
            count=e1.ledger(lib)[3]
            if at:assert rc!=0 and out.tobytes()==saved,(operation,at,count)
            else:check(lib,rc)
            check(lib,lib.denise_cuda_m9_destroy(C.byref(c)));assert e1.ledger(lib)[:3]==(0,0,0)
            return count
        count=attempt(0)
        for at in range(1,count+1):attempt(at)
        sites[operation]=count
    lib.denise_cuda_m9_fault(0);RECORDS.append({'new_faults':surface,**sites})

def request_run(lib,owner):
    r=driver.MigrationResult()
    rc=lib.denise_cuda_m9_migrate_request(C.byref(owner.request),C.byref(r))
    if rc:return rc,r,None
    image=np.stack((np.ctypeslib.as_array(r.image_lambda_raw,shape=(r.cell_count,)).copy(),
        np.ctypeslib.as_array(r.image_mu_raw,shape=(r.cell_count,)).copy()))
    lib.denise_cuda_m9_migration_result_destroy(C.byref(r));return rc,r,image

def test_request_ordered_multishot_fault_transaction(replay):
    _,libs=replay;lib=libs['fma'];exp=replace(surf.fixture(nx=7,ny=7,nt=7,cpml=False),sources=((2,2),(5,3)))
    data=[np.random.default_rng(921+s).normal(size=(exp.nt,len(exp.receivers),2)).astype(np.float32) for s in range(2)]
    owner=driver.RequestOwner(exp,data);owner.request.free_surface=1
    lib.denise_cuda_m9_replay_fault(0);rc,result,fresh=request_run(lib,owner);assert rc==0
    count=e1.ledger(lib)[3];assert e1.ledger(lib)[:3]==(0,0,0)
    shot2_failures=0
    for at in range(1,count+1):
        lib.denise_cuda_m9_replay_fault(at);rc,result,_=request_run(lib,owner)
        assert rc!=0 and not result.image_lambda_raw and not result.image_mu_raw
        assert e1.ledger(lib)[:3]==(0,0,0)
        shot2_failures+=('shot 2' in lib.denise_cuda_m9_migration_last_error().decode())
    assert shot2_failures>0
    lib.denise_cuda_m9_fault(0);rc,_,recovered=request_run(lib,owner);assert rc==0;assert_same(fresh,recovered)
    manual=np.zeros_like(fresh)
    cfg,a=e2.config_from_exp(exp,1)
    for k,shot in enumerate(owner.shots):
        cfg.source_i=shot.source_i;cfg.source_j=shot.source_j
        c=e3.create(lib,cfg)
        try:
            check(lib,lib.denise_cuda_m9_prepare(c));manual+=e3.apply(lib,c,cfg,data[k]).reshape(manual.shape)
        finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
    assert_same(fresh,manual)
    RECORDS.append({'request_faults':count,'shot2_failures':shot2_failures})

@pytest.fixture(scope='session')
def cuda_mode2_binary(denise_binary):
    candidate=denise_binary.with_name('denise_cuda')
    if not candidate.is_file():
        pytest.skip(f'CUDA executable prerequisite unavailable: {candidate}')
    return candidate

@pytest.mark.integration
@pytest.mark.optional_prerequisite
@pytest.mark.parametrize('surface,cpml,multiple',[(0,False,False),(0,True,False),(1,True,False),(1,False,True)])
def test_mode2_same_files_dispatch_and_images(tmp_path,denise_binary,mpiexec,cuda_mode2_binary,surface,cpml,multiple):
    exp=surf.fixture(nx=11,ny=9,nt=43,cpml=cpml)
    if multiple:exp=replace(exp,sources=((3,2),(8,3)))
    data=[np.random.default_rng(927+s).normal(size=(exp.nt,len(exp.receivers),2)).astype(np.float32) for s in range(len(exp.sources))]
    paths=driver._write_mode2_case(tmp_path,exp,data)
    inp=tmp_path/'denise.inp';inp.write_text(inp.read_text().replace('FREE_SURF =0',f'FREE_SURF ={surface}'))
    prepared={p:p.read_bytes() for p in (tmp_path/'prepared').glob('*')}
    cpu=driver._run_denise(tmp_path,denise_binary,mpiexec);assert cpu.returncode==0,cpu.stdout
    assert 'M9 MODE=2 backend: CPU-M9' in cpu.stdout
    expected=np.stack([np.fromfile(paths[n],np.float64) for n in ('lambda','mu')])
    cuda=driver._run_denise(tmp_path,cuda_mode2_binary,mpiexec);assert cuda.returncode==0,cuda.stdout
    assert 'M9 MODE=2 backend: CUDA-M9e-4' in cuda.stdout and 'CPU-M9\n' not in cuda.stdout
    got=np.stack([np.fromfile(paths[n],np.float64) for n in ('lambda','mu')])
    metrics=[e3.metrics(got[k],expected[k]) for k in (0,1)];assert all(z['rel_l2']<=6e-5 for z in metrics)
    assert all(p.read_bytes()==content for p,content in prepared.items())
    RECORDS.append({'mode2':[surface,cpml,multiple],'cpu_images':metrics,'cuda_log':cuda.stdout})

@pytest.mark.integration
@pytest.mark.optional_prerequisite
@pytest.mark.parametrize('fault',['ranks','unavailable','shot2'])
def test_mode2_cuda_fail_closed(tmp_path,denise_binary,mpiexec,cuda_mode2_binary,fault):
    exp=replace(surf.fixture(nx=8,ny=8,nt=12,cpml=False),sources=((3,2),(6,3)))
    data=[np.ones((exp.nt,len(exp.receivers),2),np.float32)]*2
    paths=driver._write_mode2_case(tmp_path,exp,data);env=os.environ.copy();ranks=1
    if fault=='ranks':
        ranks=2;inp=tmp_path/'denise.inp';inp.write_text(inp.read_text().replace('NPROCX =1','NPROCX =2'))
    elif fault=='unavailable':env['CUDA_VISIBLE_DEVICES']=''
    else:Path(str(paths['vx1']).replace('shot_1','shot_2')).write_bytes(b'bad')
    completed=subprocess.run([mpiexec,'-n',str(ranks),str(cuda_mode2_binary),'denise.inp','workflow.inp'],
        cwd=tmp_path,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
    assert completed.returncode!=0,completed.stdout
    marker={'ranks':'CUDA-M9e-4 unsupported envelope','unavailable':'CUDA-M9e-4 migration failed','shot2':'shot_2'}[fault]
    assert marker in completed.stdout,completed.stdout
    assert not paths['lambda'].exists() and not paths['mu'].exists()
    if fault!='ranks':
        driver._write_mode2_case(tmp_path,exp,data)
        recovered=driver._run_denise(tmp_path,cuda_mode2_binary,mpiexec);assert recovered.returncode==0,recovered.stdout
        restored=np.stack([np.fromfile(paths[n],np.float64) for n in ('lambda','mu')])
        fresh_dir=tmp_path/'fresh';fresh_paths=driver._write_mode2_case(fresh_dir,exp,data)
        fresh=driver._run_denise(fresh_dir,cuda_mode2_binary,mpiexec);assert fresh.returncode==0,fresh.stdout
        expected=np.stack([np.fromfile(fresh_paths[n],np.float64) for n in ('lambda','mu')]);assert_same(restored,expected)
@pytest.mark.parametrize('nt',[1,17,43,67])
def test_schedule_and_checked_estimator(replay,nt):
    _,libs=replay;lib=libs['nofma'];active=(Z*8)()
    for s in (1,2,3,7,32,nt,nt+23):
        d=Replay();check(lib,lib.denise_cuda_m9_estimate_replay(7,9,nt,s,active,C.byref(d)))
        e=min(s,nt);assert d.full_operand_bytes==16*nt*63
        assert d.segment_operand_bytes==16*((nt+e-1)//e)*63
        assert d.selected_replay==(d.retained_bytes<d.full_operand_bytes)
        previous=0
        for k in range(e):
            lo,hi=C.c_int(),C.c_int();check(lib,lib.denise_cuda_m9_segment_bounds(nt,s,k,C.byref(lo),C.byref(hi)))
            assert lo.value==previous;previous=hi.value
        assert previous==nt
    for nx,ny,t,s in ((0,9,nt,1),(7,9,nt,0),(2**64-1,9,nt,1),(7,2**64-1,nt,1),(7,9,2**64-1,1),(2**60,1,43,3)):
        d=Replay();C.memset(C.byref(d),0xa5,C.sizeof(d));before=e1.ledger(lib)
        assert lib.denise_cuda_m9_estimate_replay(nx,ny,t,s,active,C.byref(d))!=0
        assert bytes(d)==bytes(C.sizeof(d)) and e1.ledger(lib)==before

@pytest.mark.parametrize('nx,ny,nt,selected',[(5,5,41,False),(51,43,97,True),(7,64,41,True),(6,74,41,False)])
def test_automatic_selection_and_device_accounting(replay,nx,ny,nt,selected):
    _,libs=replay;lib=libs['fma'];cfg,a=e2.fixture(0,nx=nx,ny=ny,nt=nt,cpml=False)
    full=e3.create(lib,cfg);fd=e1.Diagnostics();check(lib,lib.denise_cuda_m9_diagnostics(full,C.byref(fd)))
    old_full=diag(lib,full);assert old_full.owned_device_bytes==fd.owned_bytes and old_full.full_operand_bytes==16*nt*nx*ny
    full_host=e1.ledger(lib)[1]
    c=create(lib,cfg,min(32,nt),1)
    try:
        d=diag(lib,c);assert bool(d.selected_replay)==selected
        cd=e1.Diagnostics();check(lib,lib.denise_cuda_m9_diagnostics(c,C.byref(cd)))
        expected=fd.owned_bytes-d.full_operand_bytes+d.segment_operand_bytes+d.checkpoint_payload_bytes+d.alignment_bytes if selected else fd.owned_bytes
        assert cd.owned_bytes==expected==d.owned_device_bytes
        extra_host=e1.ledger(lib)[1]-2*full_host
        assert extra_host==(d.checkpoint_metadata_bytes+d.segment_schedule_bytes-16 if selected else 0)
        r=np.random.default_rng(912).normal(size=(nt,cfg.receiver_count,2)).astype(np.float32)
        check(lib,lib.denise_cuda_m9_prepare(full));check(lib,lib.denise_cuda_m9_prepare(c))
        assert_same(e3.apply(lib,c,cfg,r),e3.apply(lib,full,cfg,r))
        RECORDS.append({'automatic':selected,'sizes':[nx,ny,nt],'diagnostics':{n:getattr(diag(lib,c),n) for n,_ in Replay._fields_}})
    finally:
        check(lib,lib.denise_cuda_m9_destroy(C.byref(c)));check(lib,lib.denise_cuda_m9_destroy(C.byref(full)))

def test_replay_envelope_and_legacy_probe_rejections(replay):
    _,libs=replay;lib=libs['nofma'];cfg,a=e2.fixture(0,nx=7,ny=7,nt=17,cpml=False)
    c=P();assert lib.denise_cuda_m9_create_replay(C.byref(cfg),C.byref(e1.Options(0,1,0)),3,0,C.byref(c))!=0
    assert not c and e1.ledger(lib)[:3]==(0,0,0)
    for s in (0,-1):
        assert lib.denise_cuda_m9_create_replay(C.byref(cfg),C.byref(e1.Options()),s,0,C.byref(c))!=0
        assert not c and e1.ledger(lib)[:3]==(0,0,0)
    c=create(lib,cfg,3)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c));state=np.zeros((13,7,7),np.float32)
        assert lib.denise_cuda_m9_test_evolve(c,fp(state))!=0
        assert lib.denise_cuda_m9_test_trajectory_roundtrip(c,fp(state))!=0
        assert lib.denise_cuda_m9_nonlinear(c,fp(a['l']),fp(a['m']))!=0
        assert lib.denise_cuda_m9_download(c,F(),fp(state),F(),F())!=0
        assert diag(lib,c).initial_forward_steps==17
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))

@pytest.mark.integration
@pytest.mark.optional_prerequisite
def test_mode2_small_full_fallback_matches_cuda_full(tmp_path,denise_binary,mpiexec,cuda_mode2_binary,replay):
    _,libs=replay;lib=libs['fma'];exp=surf.fixture(nx=5,ny=5,nt=41,cpml=False)
    data=[np.random.default_rng(936).normal(size=(exp.nt,len(exp.receivers),2)).astype(np.float32)]
    cfg,a=e2.config_from_exp(exp,0);c=e3.create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c));expected=e3.apply(lib,c,cfg,data[0])
    finally:check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
    paths=driver._write_mode2_case(tmp_path,exp,data)
    completed=driver._run_denise(tmp_path,cuda_mode2_binary,mpiexec);assert completed.returncode==0,completed.stdout
    assert 'trajectory backend: FULL' in completed.stdout
    got=np.stack([np.fromfile(paths[n],np.float64).reshape(5,5) for n in ('lambda','mu')]);assert_same(got,expected)
