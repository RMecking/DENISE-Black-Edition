"""Actual C surface blocks versus frozen independent matrix/VJP equations."""
import ctypes
from dataclasses import replace
import shutil
import subprocess
import numpy as np
import pytest
from tests.physics.test_m9b1_elastic_psv_born_production import (
    Config,Operator,born_library,_ptr,_f32,F32P,F64P,
    _production_dot_metrics,_assert_production_dot_closes,
)
from tests.utilities.elastic_psv_free_surface_reference import (
    fixture,SurfaceOperator,extend,extend_t,surface_coefficients,matrix_of,
    SXX,SYY,SXY,VX,VY,forward,boundary_metrics,
)

@pytest.fixture(scope='session')
def surface_blocks(tmp_path_factory,repository_root):
    out=tmp_path_factory.mktemp('m9d3-blocks')/'blocks.so'
    subprocess.run([shutil.which('cc'),'-std=c99','-O2','-Wall','-Wextra','-Werror','-pedantic',
        '-shared','-fPIC','-I',str(repository_root/'include'),str(repository_root/'tests/utilities/m9d3_surface_blocks.c'),'-lm','-o',str(out)],check=True)
    api=ctypes.CDLL(str(out));cfg=ctypes.POINTER(Config)
    api.m9d3_step.argtypes=[cfg,F32P,ctypes.c_int,F32P,F32P,F32P]
    api.m9d3_reverse.argtypes=[cfg,F64P,F32P,F32P,F64P,F64P]
    api.m9d3_extend.argtypes=[cfg,ctypes.c_int,F32P,F32P,F32P,F32P]
    api.m9d3_extend_reverse.argtypes=[cfg,ctypes.c_int,F64P,F32P,F64P,F64P,F64P,F64P,F64P]
    api.m9d3_surface_material.argtypes=[cfg,F32P,F64P,F64P,F64P,F64P,F64P,F64P]
    return api

def rel(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))

@pytest.mark.parametrize('kind,name',[(SYY,'syy'),(SXY,'sxy'),(VX,'vx'),(VY,'vy')])
def test_actual_boundary_forward_reverse_and_basis(surface_blocks,born_library,kind,name):
    exp=fixture(cpml=False)
    owner=Operator(born_library,exp,free_surface=1)
    rng=np.random.default_rng(411)
    physical,xx,yx=map(_f32,rng.normal(size=(3,exp.ny,exp.nx)))
    bar=np.ascontiguousarray(rng.normal(size=(exp.ny+4,exp.nx)))
    h=float(np.float32(1/np.float32(exp.dt/exp.dh)))
    coeff=surface_coefficients(owner.lam[0].astype(float),owner.mu[0].astype(float))
    alpha=coeff[0].astype(np.float32).astype(float)
    out=np.empty_like(bar,dtype=np.float32)
    reverse=[np.zeros_like(physical,dtype=np.float64) for _ in range(5)]
    try:
        assert surface_blocks.m9d3_extend(ctypes.byref(owner.config),kind,_ptr(physical),_ptr(xx),_ptr(yx),_ptr(out))==0
        expect=extend(physical.astype(float),name,alpha=alpha,qxx=xx.astype(float),qyx=yx.astype(float),h_over_dt=h)
        assert rel(out.astype(float),expect)<=2e-6
        assert surface_blocks.m9d3_extend_reverse(ctypes.byref(owner.config),kind,_ptr(bar,F64P),_ptr(xx),*[_ptr(a,F64P) for a in reverse])==0
        p,qx,qy,abar=extend_t(bar,name,alpha=alpha,qxx=xx.astype(float),h_over_dt=h)
        for got,ref in zip(reverse[:3],(p,qx,qy)):assert rel(got,ref)<5e-12 or not np.any(got-ref)
        if kind==VY:
            for got,ref in zip(reverse[3:],(coeff[2]*abar,coeff[3]*abar)):
                assert rel(got[0],ref)<5e-12
                assert not np.any(got[1:])
        def action(v):return extend(v[0],name,alpha=alpha,qxx=v[1],qyx=v[2],h_over_dt=h)
        matrix=matrix_of(action,(3,exp.ny,exp.nx))
        expected=(matrix.T@bar.ravel()).reshape(3,exp.ny,exp.nx)
        assert rel(np.stack(reverse[:3]),expected)<5e-12
        # Pure overwrite and mirror basis cotangents: exact ADD/consume.
        if kind in (SYY,SXY):
            unit=np.zeros_like(bar);unit[1]=1
            assert surface_blocks.m9d3_extend_reverse(ctypes.byref(owner.config),kind,_ptr(unit,F64P),_ptr(xx),*[_ptr(a,F64P) for a in reverse])==0
            target=np.zeros_like(p);target[1 if kind==SYY else 0]=-1
            np.testing.assert_array_equal(reverse[0],target)
            unit[:]=0;unit[2]=1
            assert surface_blocks.m9d3_extend_reverse(ctypes.byref(owner.config),kind,_ptr(unit,F64P),_ptr(xx),*[_ptr(a,F64P) for a in reverse])==0
            if kind==SYY:assert not np.any(reverse[0])
    finally:owner.close()

def test_four_separate_surface_material_images(surface_blocks,born_library):
    exp=fixture();owner=Operator(born_library,exp,free_surface=1)
    rng=np.random.default_rng(439);q=_f32(rng.normal(size=(exp.ny,exp.nx)))
    sx=rng.normal(size=exp.nx);ghost=rng.normal(size=(2,exp.nx))
    images=[np.zeros(exp.nx) for _ in range(4)]
    try:
        assert surface_blocks.m9d3_surface_material(ctypes.byref(owner.config),_ptr(q),_ptr(sx,F64P),_ptr(ghost,F64P),*[_ptr(a,F64P) for a in images])==0
        _,_,al,am,bl,bm=surface_coefficients(owner.lam[0].astype(float),owner.mu[0].astype(float))
        h=float(np.float32(1/np.float32(exp.dt/exp.dh)))
        abar=h*q[0].astype(float)*(ghost[0]+3*ghost[1])
        for got,ref in zip(images,(al*abar,am*abar,bl*q[0]*sx,bm*q[0]*sx)):
            assert rel(got,ref)<5e-13
    finally:owner.close()

@pytest.mark.parametrize('cpml',(False,True))
def test_complete_step_state_sample_and_vjp(surface_blocks,born_library,cpml):
    exp=fixture();exp=replace(exp,cpml=cpml,fw=2 if cpml else 0)
    owner=Operator(born_library,exp,free_surface=1);owner.source[:]=0
    reference=SurfaceOperator(exp);rng=np.random.default_rng(417)
    state=_f32(rng.normal(size=reference.zero().shape)*1.e-3)
    expected,tape=reference.step(state.astype(float))
    actual=state.copy();strains=np.zeros((4,exp.ny,exp.nx),np.float32)
    samples=np.zeros(owner.data_shape,np.float32);ghosts=np.empty((4,2,exp.nx),np.float32)
    bar=rng.normal(size=state.shape);samplebar=_f32(rng.normal(size=tape.sample.shape))
    gl=np.empty_like(exp.lam);gm=np.empty_like(exp.mu)
    try:
        assert surface_blocks.m9d3_step(ctypes.byref(owner.config),_ptr(actual),0,_ptr(strains),_ptr(samples),_ptr(ghosts))==0
        assert rel(actual,expected)<=2e-6
        assert rel(samples[0],tape.sample)<=2e-6
        reversed_bar=bar.copy()
        assert surface_blocks.m9d3_reverse(ctypes.byref(owner.config),_ptr(reversed_bar,F64P),_ptr(strains),_ptr(samplebar),_ptr(gl,F64P),_ptr(gm,F64P))==0
        b,rl,rm=reference.reverse(bar,tape,samplebar.astype(float))
        assert rel(reversed_bar,b)<=6e-5
        assert rel(gl,rl)<=6e-5 and rel(gm,rm)<=6e-5
        assert not np.any(actual[SYY,0])
        np.testing.assert_array_equal(ghosts[2,0],-actual[SYY,1])
        np.testing.assert_array_equal(ghosts[2,1],-actual[SYY,2])
        np.testing.assert_array_equal(ghosts[3,0],-actual[SXY,0])
        np.testing.assert_array_equal(ghosts[3,1],-actual[SXY,1])
    finally:owner.close()

def test_background_all_fields_memories_ghosts_and_cpml(surface_blocks,born_library):
    exp=fixture();_,tapes,checkpoints=forward(exp)
    owner=Operator(born_library,exp,free_surface=1)
    state=np.zeros((13,exp.ny,exp.nx),np.float32)
    strains=np.zeros((4,exp.ny,exp.nx),np.float32)
    samples=np.zeros(owner.data_shape,np.float32);ghosts=np.empty((4,2,exp.nx),np.float32)
    maxima=[]
    try:
        for k in range(exp.nt):
            assert surface_blocks.m9d3_step(ctypes.byref(owner.config),_ptr(state),k,_ptr(strains),_ptr(samples),_ptr(ghosts))==0
            for got,expected in zip(state,checkpoints[k+1]):
                error=rel(got,expected) if np.any(expected) else np.linalg.norm(got)
                assert error<=2e-6
                maxima.append(error)
            assert not np.any(state[SYY,0])
            for memory in (6,8,11,12):assert not np.any(state[memory,:3])
        assert np.linalg.norm(state[[9,10],0])>0
        assert np.linalg.norm(state[[6,8,11,12],-3:])>0
        print('FP32/FP64_PRODUCTION_ALL_STATE_MAX',max(maxima))
    finally:owner.close()

def test_complete_step_actual_basis_matrix_and_vjp(surface_blocks,born_library):
    exp=fixture(nx=5,ny=5,nt=1,cpml=False,source=(2,2))
    owner=Operator(born_library,exp,free_surface=1);owner.source[:]=0
    op=SurfaceOperator(exp);shape=op.zero().shape
    strain=np.empty((4,exp.ny,exp.nx),np.float32)
    data=np.empty(owner.data_shape,np.float32);ghost=np.empty((4,2,exp.nx),np.float32)
    def independent(z):
        state,tape=op.step(z)
        return np.concatenate((state.ravel(),tape.sample.ravel()))
    def production(z):
        z=_f32(z).copy()
        assert surface_blocks.m9d3_step(ctypes.byref(owner.config),_ptr(z),0,_ptr(strain),_ptr(data),_ptr(ghost))==0
        return np.concatenate((z.ravel(),data[0].ravel())).astype(float)
    try:
        matrix=matrix_of(independent,shape)
        actual=matrix_of(production,shape)
        assert rel(actual,matrix)<=2e-6
        rng=np.random.default_rng(417);bar=rng.normal(size=shape)
        sampling=_f32(rng.normal(size=(len(exp.receivers),2)))
        combined=np.concatenate((bar.ravel(),sampling.ravel()))
        state=_f32(rng.normal(size=shape)*1e-3)
        output=production(state)
        reverse=bar.copy();gl=np.empty_like(exp.lam);gm=np.empty_like(exp.mu)
        assert surface_blocks.m9d3_reverse(ctypes.byref(owner.config),_ptr(reverse,F64P),_ptr(strain),_ptr(sampling),_ptr(gl,F64P),_ptr(gm,F64P))==0
        expected=matrix.T@combined
        assert rel(reverse.ravel(),expected)<=6e-5
        reference_output=matrix@state.ravel()
        zero=np.zeros_like(state)
        scale=max(np.linalg.norm(reference_output)*np.linalg.norm(combined),np.linalg.norm(state)*np.linalg.norm(expected))
        oracle_residual=abs(np.dot(reference_output,combined)-np.dot(state.ravel(),expected))
        metrics=_production_dot_metrics(output,combined,state,zero,reverse,zero,reference_output,
                    expected.reshape(state.shape),zero,dict(absolute_ceiling=5e-13*scale,absolute_residual=oracle_residual))
        _assert_production_dot_closes(metrics)
        print('FP32/FP64_PRODUCTION_COMPLETE_MATRIX',rel(actual,matrix),rel(reverse.ravel(),expected))
        print('FP32/FP64_PRODUCTION_COMPLETE_STEP_DOT',metrics)
    finally:owner.close()

def test_production_normal_p_packet_physical_reflection(surface_blocks,born_library):
    exp=fixture(nx=8,ny=96,nt=450,cpml=False,source=(4,30))
    vp=3000.;rho=2000.
    exp=replace(exp,lam=np.full_like(exp.lam,6.44e9),mu=np.full_like(exp.mu,5.78e9),
                rho=np.full_like(exp.rho,rho),receivers=((4,13),))
    owner=Operator(born_library,exp,free_surface=1);owner.source[:]=0
    state=np.zeros((13,exp.ny,exp.nx),np.float32)
    center=285.;width=35.
    stress_y=(np.arange(exp.ny)+.5)*exp.dh
    velocity_y=(np.arange(exp.ny)+1)*exp.dh
    sy=np.exp(-((stress_y-center)/width)**2)
    vy=np.exp(-((velocity_y-center-vp*exp.dt/2)/width)**2)/(rho*vp)
    state[SYY]=sy[:,None];state[SXX]=(exp.lam/(exp.lam+2*exp.mu))*sy[:,None]
    state[VY]=vy[:,None]
    strains=np.empty((4,exp.ny,exp.nx),np.float32);data=np.empty(owner.data_shape,np.float32)
    ghosts=np.empty((4,2,exp.nx),np.float32);vxpeak=0.
    try:
        for k in range(exp.nt):
            assert surface_blocks.m9d3_step(ctypes.byref(owner.config),_ptr(state),k,_ptr(strains),_ptr(data),_ptr(ghosts))==0
            vxpeak=max(vxpeak,float(np.max(np.abs(state[VX]))))
        times=(np.arange(exp.nt)+1)*exp.dt;samples=data[:,0,1]
        targets=((center-130.)/vp,(center-5.+130.-5.)/vp)
        picks=[]
        for target in targets:
            indices=np.flatnonzero(abs(times-target)<.015)
            pick=indices[np.argmax(abs(samples[indices]))];picks.append(pick)
            assert abs(times[pick]-target)<2*exp.dt+.005*target
            assert samples[pick]>0
        assert vxpeak<1e-20
        print('FP32/FP64_PRODUCTION_NORMAL_P',times[picks],targets,samples[picks],vxpeak)
    finally:owner.close()
