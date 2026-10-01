"""Frozen before M9d3 production: exact discrete traction and Born/VJP gates."""
from dataclasses import replace
import json
import numpy as np
import pytest

from tests.utilities.elastic_psv_free_surface_reference import (
    EXTRAPOLATE_SURFACE, SXX, SYY, SurfaceOperator, adjoint, born,
    boundary_metrics, dx, extend, extend_t, fixture, forward,
    halo_surface_maps, matrix_of, profiles, surface_coefficients,
    validate_geometry, pml, pml_t, checkpoint_pack, checkpoint_restore,
    distributed_derivative, distributed_derivative_t,
)
from tests.utilities.elastic_psv_born_reference import dot

# Frozen numerical envelopes, not fitted to production output.
DOT_MAX = 5.e-13
MATRIX_MAX = 5.e-12
TRACTION_MAX = 5.e-14
EPSILONS = (1., .5, .25, .125)
FD_FINAL_MAX = 2.e-5
FD_REDUCTION_MIN = 3.9


def rel(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1.e-300))


def assert_dot(x,bx,y,bty):
    residual=abs(dot(bx,y)-dot(x,bty))
    scale=max(np.linalg.norm(bx)*np.linalg.norm(y),np.linalg.norm(x)*np.linalg.norm(bty),1.e-300)
    assert residual/scale <= DOT_MAX
    return residual/scale


def test_surface_geometry_and_constitutive_elimination():
    # One-based normal/vx y=(j-.5)DH; vy/sxy y=j DH.
    dh=10.; j=1
    assert (j-.5)*dh == 5.
    assert ((j-1)*dh+j*dh)/2 == 5.
    lam=np.array([6.,8.,11.]); mu=np.array([3.,4.,5.])
    alpha,a,al,am,bl,bm=surface_coefficients(lam,mu)
    ex=np.array([.3,-.2,.7]); ey=-alpha*ex
    np.testing.assert_allclose(lam*ex+(lam+2*mu)*ey,0.,atol=1.e-15)
    np.testing.assert_allclose((lam+2*mu)*ex+lam*ey,a*ex,rtol=1.e-15)
    # Independently derive A as Schur complement of the 2x2 stiffness matrix.
    np.testing.assert_allclose(a,lam+2*mu-lam**2/(lam+2*mu),rtol=1.e-15)
    for dl,dm,da,db in ((1.,0.,al,bl),(0.,1.,am,bm)):
        eps=1.e-4
        p=surface_coefficients(lam+eps*dl,mu+eps*dm)
        m=surface_coefficients(lam-eps*dl,mu-eps*dm)
        np.testing.assert_allclose((p[0]-m[0])/(2*eps),da,rtol=2.e-8)
        np.testing.assert_allclose((p[1]-m[1])/(2*eps),db,rtol=2.e-8)
    # Cubic evaluation from half nodes onto y=0 is exact, not a fitted weight.
    y=np.arange(4)+.5
    for degree in range(4):
        assert abs(EXTRAPOLATE_SURFACE@y**degree-(1. if degree==0 else 0.))<1.e-14


@pytest.mark.parametrize("kind",("syy","sxy"))
def test_stress_extension_matrix_and_overwrite_cotangents(kind):
    rng=np.random.default_rng(410); shape=(4,3)
    x=rng.normal(size=shape); y=rng.normal(size=(8,3))
    b=matrix_of(lambda v:extend(v,kind),shape)
    got=extend(x,kind); rev=extend_t(y,kind)[0]
    np.testing.assert_allclose(got.ravel(),b@x.ravel(),rtol=0,atol=1.e-15)
    np.testing.assert_allclose(rev.ravel(),b.T@y.ravel(),rtol=0,atol=1.e-15)
    residual=assert_dot(x,got,y,rev)
    if kind=="syy":
        assert np.all(rev[0]==y[6])  # Only bottom periodic ghost contributes.
        unit=np.zeros_like(y);unit[2]=1
        assert not np.any(extend_t(unit,kind)[0])
    unit=np.zeros_like(y); unit[1]=1
    expected=np.zeros_like(x);expected[1 if kind=="syy" else 0]=-1
    np.testing.assert_array_equal(extend_t(unit,kind)[0],expected)
    print("LOCAL_STRESS",kind,residual)


@pytest.mark.parametrize("kind",("vx","vy"))
def test_velocity_surface_matrix_and_positive_mirror_transpose(kind):
    rng=np.random.default_rng(411);shape=(3,4,3)
    x=rng.normal(size=shape); y=rng.normal(size=(8,3));alpha=np.array([.3,.4,.5])
    def apply(v):
        return extend(v[0],kind,alpha=alpha,qxx=v[1],qyx=v[2],h_over_dt=.7)
    b=matrix_of(apply,shape)
    physical,qx,qy,_=extend_t(y,kind,alpha=alpha,qxx=x[1],h_over_dt=.7)
    rev=np.stack((physical,qx,qy))
    np.testing.assert_allclose(rev.ravel(),b.T@y.ravel(),rtol=0,atol=2.e-15)
    residual=assert_dot(x,apply(x),y,rev)
    unit=np.zeros_like(y);unit[1]=1
    reflected=extend_t(unit,kind,alpha=alpha,qxx=x[1],h_over_dt=.7)[0]
    assert np.all(reflected[1 if kind=="vx" else 0]==1)
    print("LOCAL_VELOCITY",kind,residual)


def test_velocity_extension_satisfies_both_surface_derivative_constraints():
    from tests.utilities.elastic_psv_free_surface_reference import dy_extended
    rng=np.random.default_rng(431);vx=rng.normal(size=(8,8));vy=rng.normal(size=(8,8))
    qxx=rng.normal(size=(8,8));qyx=rng.normal(size=(8,8));alpha=np.linspace(.2,.5,8)
    ex=extend(vx,"vx",qyx=qyx,h_over_dt=.7)
    ey=extend(vy,"vy",alpha=alpha,qxx=qxx,h_over_dt=.7)
    np.testing.assert_allclose(dy_extended(ey,"yb")[0],-.7*alpha*qxx[0],rtol=0,atol=5.e-15)
    central=(ex[0]-8*ex[1]+8*ex[3]-ex[4])/12
    slope=-.7*np.einsum("j,ji->i",EXTRAPOLATE_SURFACE,qyx[:4])
    np.testing.assert_allclose(central,slope,rtol=0,atol=5.e-15)


@pytest.mark.parametrize("kind,field",(("yf","syy"),("yb","sxy"),("yf","vx"),("yb","vy")))
def test_distributed_surface_derivative_jvp_vjp(kind,field):
    rng=np.random.default_rng(432);topology=(2,2)
    a=rng.normal(size=(8,8));qxx=rng.normal(size=a.shape);qyx=rng.normal(size=a.shape)
    bar=rng.normal(size=a.shape);alpha=np.linspace(.2,.5,8);args={}
    if field=="vx": args=dict(qyx=qyx,h_over_dt=.7)
    if field=="vy": args=dict(alpha=alpha,qxx=qxx,h_over_dt=.7)
    result=distributed_derivative(a,kind,topology,field=field,**args)
    ba,bx,by,_=distributed_derivative_t(bar,kind,topology,field=field,**args)
    x=np.stack((a,qxx,qyx));btx=np.stack((ba,bx,by))
    assert_dot(x,result,bar,btx)
    global_result=distributed_derivative(a,kind,(1,1),field=field,**args)
    np.testing.assert_array_equal(result,global_result)


def test_complete_timestep_state_and_parameter_vjp_vs_dense_matrices():
    exp=fixture(nx=4,ny=4,nt=3,cpml=False,source=(2,2))
    op=SurfaceOperator(exp);rng=np.random.default_rng(417)
    state=rng.normal(size=op.zero().shape)*1.e-3
    _,tape=op.step(state);bar=rng.normal(size=state.shape)
    samplebar=rng.normal(size=tape.sample.shape)
    rev,gl,gm=op.reverse(bar,tape,samplebar)
    def step_and_sample(z):
        out,t=op.step(z)
        return np.concatenate((out.ravel(),t.sample.ravel()))
    b=matrix_of(step_and_sample,state.shape)
    combined_bar=np.concatenate((bar.ravel(),samplebar.ravel()))
    assert rel(rev.ravel(),b.T@combined_bar)<MATRIX_MAX
    ds=rng.normal(size=state.shape)
    residual=assert_dot(ds,step_and_sample(ds),combined_bar,rev)
    dirs=np.stack((exp.lam*.01,exp.mu*.01))
    # Actual dense parameter Jacobian from analytic forward tangents.
    def parameter(v):
        return op.tangent(op.zero(),tape,v[0]*dirs[0],v[1]*dirs[1])[0]
    j=matrix_of(parameter,dirs.shape)
    expect=j.T@bar.ravel()
    actual=np.stack((gl*dirs[0],gm*dirs[1])).ravel()
    assert rel(actual,expect)<MATRIX_MAX
    direction=rng.normal(size=dirs.shape)
    eps=1.e-4
    p=SurfaceOperator(exp,lam=exp.lam+eps*direction[0]*dirs[0],mu=exp.mu+eps*direction[1]*dirs[1]).step(state)[0]
    m=SurfaceOperator(exp,lam=exp.lam-eps*direction[0]*dirs[0],mu=exp.mu-eps*direction[1]*dirs[1]).step(state)[0]
    assert rel((p-m)/(2*eps),parameter(direction))<1.e-7
    # Replacing the surface coefficient with the bulk coefficient is falsified.
    bulk=exp.lam[0]+2*exp.mu[0]
    assert np.linalg.norm((bulk-op.a)*tape.strain[0][0])>0
    print("TIMESTEP_STATE_DOT",residual,"STATE_MATRIX",rel(rev.ravel(),b.T@combined_bar),
          "MATERIAL_MATRIX",rel(actual,expect),"MATERIAL_FD",rel((p-m)/(2*eps),parameter(direction)))


def test_active_horizontal_cpml_and_surface_correction_matrix():
    from tests.utilities.elastic_psv_born_reference import PMLProfile
    p=PMLProfile(np.array([1.3,1.1]),np.array([-.2,-.1]),np.array([.8,.9]))
    rng=np.random.default_rng(430)
    x=rng.normal(size=(3,2));y=rng.normal(size=(2,2))
    lam=np.array([6.,8.]);mu=np.array([3.,4.]);_,a,*_=surface_coefficients(lam,mu)
    def block(z):
        q,mem=pml(z[1],z[2],p)
        return np.stack((z[0]+a*q,mem))
    b=matrix_of(block,x.shape)
    raw,old=pml_t(a*y[0],y[1],p)
    rev=np.stack((y[0],raw,old))
    np.testing.assert_allclose(rev.ravel(),b.T@y.ravel(),rtol=0,atol=2.e-15)
    print("LOCAL_SURFACE_CPML_SXX",assert_dot(x,block(x),y,rev))


@pytest.mark.parametrize("topology",((1,1),(2,1),(1,2),(2,2)))
@pytest.mark.parametrize("kind",("syy","sxy"))
def test_composed_halo_surface_matrix_and_ownership(topology,kind):
    px,py=topology; h,b=halo_surface_maps(8,8,px,py,kind)
    rng=np.random.default_rng(418);x=rng.normal(size=h.size);y=rng.normal(size=h.size)
    c=b.matrix()@h.matrix()
    got=b.forward(h.forward(x));rev=h.transpose(b.transpose(y))
    assert rel(got,c@x)<MATRIX_MAX
    assert rel(rev,c.T@y)<MATRIX_MAX
    residual=assert_dot(x,got,y,rev)
    lx,ly=8//px,8//py;cells=(lx+4)*(ly+4)
    surface=b.forward(x).reshape(py*px,ly+4,lx+4)
    original=x.reshape(py*px,ly+4,lx+4)
    for r in range(px*py):
        if r//px:
            np.testing.assert_array_equal(surface[r],original[r])
    for stage in b.stages:
        for dst,src,sign in stage:
            unit=np.zeros(h.size);unit[dst]=1
            cot=b.transpose(unit)
            assert cot[dst]==0.
            if src is not None: assert cot[src]==sign
    assert cells*px*py==h.size
    print("MPI_LOCAL",topology,kind,residual)


def distributed_dx(a,kind,topology):
    """Use actual per-rank copied x ghosts, not a global stencil shortcut."""
    ny,nx=a.shape;px,py=topology;lx,ly=nx//px,ny//py
    h,_=halo_surface_maps(nx,ny,px,py,"velocity")
    local=np.zeros((px*py,ly+4,lx+4))
    for r in range(px*py):
        ox,oy=(r%px)*lx,(r//px)*ly
        local[r,2:ly+2,2:lx+2]=a[oy:oy+ly,ox:ox+lx]
    local=h.forward(local.ravel()).reshape(local.shape)
    from tests.utilities.elastic_psv_free_surface_reference import TERMS
    out=np.zeros_like(a)
    for r in range(px*py):
        ox,oy=(r%px)*lx,(r//px)*ly
        out[oy:oy+ly,ox:ox+lx]=sum(w*local[r,2:ly+2,2+o:2+o+lx] for o,w in TERMS[kind])
    return out


@pytest.mark.parametrize("topology",((1,1),(2,1),(1,2),(2,2)))
def test_x_surface_derivative_is_decomposition_independent(topology):
    rng=np.random.default_rng(420);v=rng.normal(size=(8,8))
    for kind in ("xb","xf"):
        result=distributed_dx(v,kind,topology)
        np.testing.assert_array_equal(result,dx(v,kind))
        # All x locations include interior, two/one cell from split, and wrap.
        np.testing.assert_array_equal(result[0],dx(v,kind)[0])


@pytest.fixture(scope="module")
def multistep():
    exp=fixture();data,tapes,checkpoints=forward(exp)
    rng=np.random.default_rng(423)
    dl=.01*exp.lam*rng.normal(size=exp.lam.shape)
    dm=.01*exp.mu*rng.normal(size=exp.mu.shape)
    return exp,data,tapes,checkpoints,dl,dm


@pytest.mark.parametrize("direction",("lambda","mu","joint"))
@pytest.mark.parametrize("component",(0,1,None),ids=("vx","vy","both"))
def test_multitimestep_born_exact_transpose(multistep,direction,component):
    exp,_,tapes,_,dl,dm=multistep
    if direction=="lambda": dm=np.zeros_like(dm)
    if direction=="mu": dl=np.zeros_like(dl)
    jdm=born(exp,tapes,dl,dm)
    data=np.random.default_rng(424).normal(size=jdm.shape)
    if component is not None: data[...,1-component]=0
    gl,gm=adjoint(exp,tapes,data)
    lhs=dot(jdm,data);rhs=dot(dl,gl)+dot(dm,gm)
    scale=max(np.linalg.norm(jdm)*np.linalg.norm(data),
              np.linalg.norm(np.stack((dl,dm)))*np.linalg.norm(np.stack((gl,gm))))
    residual=abs(lhs-rhs)/scale
    assert residual<DOT_MAX
    assert np.linalg.norm(gl[0])+np.linalg.norm(gm[0])>0
    print("MULTISTEP_DOT",direction,component,residual)


@pytest.mark.parametrize("direction",("lambda","mu","joint"))
def test_frozen_nonlinear_fd_ladder(multistep,direction):
    exp,_,tapes,_,dl,dm=multistep
    if direction=="lambda": dm=np.zeros_like(dm)
    if direction=="mu": dl=np.zeros_like(dl)
    jdm=born(exp,tapes,dl,dm);errors=[]
    for eps in EPSILONS:
        p=forward(exp,lam=exp.lam+eps*dl,mu=exp.mu+eps*dm)[0]
        m=forward(exp,lam=exp.lam-eps*dl,mu=exp.mu-eps*dm)[0]
        errors.append(rel((p-m)/(2*eps),jdm))
    assert all(errors[k+1]<errors[k]/FD_REDUCTION_MIN for k in range(3))
    assert errors[-1]<FD_FINAL_MAX
    print("FD_LADDER",direction,errors)


def test_traction_parity_extension_and_replay_closure(multistep):
    exp,_,_,checkpoints,_,_=multistep
    worst={k:0. for k in ("normal_traction","shear_traction","ghost_parity","extension_closure")}
    for state in checkpoints:
        metrics=boundary_metrics((extend(state[SYY],"syy"),extend(state[4],"sxy")))
        for key,val in metrics.items():
            assert val<=TRACTION_MAX
            worst[key]=max(worst[key],val)
    cut=13
    op=SurfaceOperator(exp)
    packed=checkpoint_pack(op,checkpoints[cut])
    restored=checkpoint_restore(op,packed)
    np.testing.assert_array_equal(restored,checkpoints[cut])
    resumed=forward(exp,initial=restored,start=cut)
    complete=forward(exp)
    np.testing.assert_array_equal(resumed[0],complete[0][cut:])
    np.testing.assert_array_equal(resumed[2][-1],complete[2][-1])
    for reconstructed,original in zip(resumed[1],complete[1][cut:]):
        np.testing.assert_array_equal(reconstructed.strain,original.strain)
    # No halo/ghost/strain payload is passed to restart.
    assert checkpoints[cut].shape==(13,exp.ny,exp.nx)
    invalid=restored.copy();invalid[12,0,0]=1
    with pytest.raises(ValueError,match="inactive"):
        checkpoint_pack(op,invalid)
    print("BOUNDARY",json.dumps(worst),"REPLAY",0.,"CHECKPOINT_VALUES",packed.size)


def test_independent_normal_p_packet_reflection_timing_polarity_and_mode():
    exp=fixture(nx=8,ny=96,nt=450,cpml=False,source=(4,30))
    vp=3000.;rho=2000.;lam=np.full_like(exp.lam,6.44e9);mu=np.full_like(exp.mu,5.78e9)
    exp=replace(exp,lam=lam,mu=mu,rho=np.full_like(lam,rho),receivers=((4,13),))
    op=SurfaceOperator(exp);z=op.zero()
    center=285.;width=35.
    stress_y=(np.arange(exp.ny)+.5)*exp.dh
    velocity_y=(np.arange(exp.ny)+1)*exp.dh
    sy=np.exp(-((stress_y-center)/width)**2)
    # Upward P at t=-DT/2 for the staggered velocity initial condition.
    vy=np.exp(-((velocity_y-center-vp*exp.dt/2)/width)**2)/(rho*vp)
    z[SYY]=sy[:,None];z[SXX]=(lam/(lam+2*mu))*sy[:,None];z[1]=vy[:,None]
    samples=[];vxpeak=0.
    for _ in range(exp.nt):
        z,tape=op.step(z);samples.append(tape.sample[0,1])
        vxpeak=max(vxpeak,float(np.max(np.abs(z[0]))))
    times=(np.arange(exp.nt)+1)*exp.dt;samples=np.asarray(samples)
    direct=(center-130.)/vp
    reflected=(center-5.+130.-5.)/vp
    picks=[]
    for target in (direct,reflected):
        indices=np.flatnonzero(abs(times-target)<.015)
        pick=indices[np.argmax(abs(samples[indices]))];picks.append(pick)
        tolerance=2*exp.dt+.005*target
        assert abs(times[pick]-target)<tolerance
    assert samples[picks[0]]>0 and samples[picks[1]]>0
    assert vxpeak<1.e-20  # Normal incidence has no converted SV polarization.
    print("REFERENCE_NORMAL_P",float(times[picks[0]]),float(times[picks[1]]),
          "expected",direct,reflected,"vy",float(samples[picks[0]]),float(samples[picks[1]]),"vx",vxpeak)


@pytest.mark.parametrize("topology",((2,1),(1,2),(2,2)))
def test_global_multistep_and_born_are_decomposition_independent(multistep,topology):
    exp,data,tapes,checkpoints,dl,dm=multistep
    variant=forward(exp,topology=topology)
    np.testing.assert_array_equal(variant[0],data)
    np.testing.assert_array_equal(variant[2][-1],checkpoints[-1])
    j=born(exp,tapes,dl,dm)
    np.testing.assert_array_equal(born(exp,variant[1],dl,dm,topology=topology),j)
    # Local halo ADD transpose has already been independently proved by matrix.
    gl,gm=adjoint(exp,variant[1],np.ones_like(data),topology=topology)
    rl,rm=adjoint(exp,tapes,np.ones_like(data))
    # ADD grouping across ranks is allowed to differ by FP64 rounding.
    assert rel(gl,rl)<MATRIX_MAX and rel(gm,rm)<MATRIX_MAX
    print("GLOBAL_TOPOLOGY",topology,"background/Born",0.,"images",rel(gl,rl),rel(gm,rm))


def test_free_top_and_other_cpml_layers_have_independent_memories(multistep):
    exp,_,_,checkpoints,_,_=multistep;p=profiles(exp)
    for name in ("y","yh"):
        np.testing.assert_array_equal(p[name].a[:4],0.)
        np.testing.assert_array_equal(p[name].kappa[:4],1.)
        assert np.any(p[name].a[-2:]!=0)
    assert np.any(p["x"].a[:,:2]!=0) and np.any(p["x"].a[:,-2:]!=0)
    z=checkpoints[-1]
    assert np.linalg.norm(z[12,-2:])+np.linalg.norm(z[8,-2:])>0
    assert np.linalg.norm(z[9,:,:2])+np.linalg.norm(z[9,:,-2:])>0
    np.testing.assert_array_equal(z[[6,8,11,12],:4],0.)
    print("PML_ACTIVE_BOTTOM",float(np.linalg.norm(z[[6,8,11,12],-2:])),
          "PML_ACTIVE_X",float(np.linalg.norm(z[[5,7,9,10],:,:2])))


@pytest.mark.parametrize("source",((4,2),(4,5)))
def test_source_and_receiver_within_surface_stencil_reach(source):
    exp=fixture(source=source)
    data,_,_=forward(exp)
    assert np.linalg.norm(data)>0
    validate_geometry(exp)
    with pytest.raises(ValueError,match="row j >= 2"):
        validate_geometry(replace(exp,sources=((4,1),)))
    with pytest.raises(ValueError,match="local_ny"):
        validate_geometry(replace(exp,ny=6),(1,2))
    validate_geometry(exp,(2,2))
