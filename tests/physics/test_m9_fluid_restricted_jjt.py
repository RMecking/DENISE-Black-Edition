"""Actual restricted CPU J/JT, immutable independent references and FD gates."""
import ctypes as C
from dataclasses import replace
import numpy as np
import pytest
from tests.physics import test_m9_fluid_cpu_forward as f
from tests.physics.test_m9_fluid_cpu_forward import fluid_library
from tests.physics import test_m9b1_elastic_psv_born_production as accepted
from tests.utilities import zero_shear_reference as z
from tests.utilities import m9_fluid_restricted_reference as ref


def j(c, dl, dm):
    out=np.empty(c.data_shape,np.float32)
    c.check(c.api.denise_elastic_psv_born_apply_j(c.c,f.fp(np.asarray(dl,np.float32)),f.fp(np.asarray(dm,np.float32)),f.fp(out)))
    return out


def jt(c, data):
    out=np.empty((2,*c.shape),np.float64)
    c.check(c.api.denise_elastic_psv_born_apply_jt(c.c,f.fp(np.asarray(data,np.float32)),f.dp(out[0]),f.dp(out[1])))
    return out


def nonlinear(c, lam, mu):
    out=np.empty(c.data_shape,np.float32)
    c.check(c.api.denise_elastic_psv_born_nonlinear(c.c,f.fp(np.asarray(lam,np.float32)),f.fp(np.asarray(mu,np.float32)),f.fp(out)))
    return out


def context(api, exp, fs=0, segments=0):
    c=f.Context(api,mu=exp.mu,lam=exp.lam,rho=exp.rho,nt=exp.nt,fs=fs,fw=exp.fw,segments=segments)
    # Test acquisition is explicit and shared with independent references.
    c.source[:]=ref.b._source_samples(exp).astype(np.float32)
    # Context constructor has already copied source; same fc=.5/t0=3 for NT120.
    assert exp.nt==120
    return c


@pytest.mark.parametrize("pattern,fs,fw",[
    ("homogeneous",0,0),("horizontal",0,0),("offset",0,0),
    ("horizontal",1,0),("horizontal",0,3),("horizontal",1,3)])
@pytest.mark.parametrize("segments",[0,3])
def test_restricted_reference_dot_fd(fluid_library,pattern,fs,fw,segments):
    exp=ref.fixture(pattern,fs,fw);c=context(fluid_library,exp,fs,segments)
    try:
        bg=c.prepare();prepared=ref.trajectory(exp,c.source.astype(float),fs)
        accepted._assert_production_close(bg,prepared[2],1e-5)
        material=z.material(exp.rho,exp.lam,exp.mu)
        for name,(dl,dm) in z.directions(material).items():
            if np.linalg.norm(dl)+np.linalg.norm(dm)==0:continue
            dl,dm=(a.astype(np.float32).astype(float) for a in (dl,dm))
            c.api.fluid_clear_arithmetic()
            actual=j(c,dl,dm)
            assert np.isfinite(actual).all() and np.linalg.norm(actual)>0
            plus=nonlinear(c,exp.lam+.05*dl,exp.mu+.05*dm)
            minus=nonlinear(c,exp.lam-.05*dl,exp.mu-.05*dm)
            fd=np.linalg.norm((plus.astype(float)-minus.astype(float))/.1-actual)/np.linalg.norm(actual)
            assert fd<=.004,fd
            for component in (0,1,None):
                data=np.random.default_rng(973).normal(size=c.data_shape).astype(np.float32)
                if component is not None:data[:,:,1-component]=0
                raw=jt(c,data)
                expected=ref.products(exp,c.source.astype(float),dl,dm,data.astype(float),fs,prepared)
                metric=accepted._production_dot_metrics(actual,data,dl,dm,*raw,*expected)
                accepted._assert_production_dot_closes(metric)
                assert np.isfinite(raw).all()
                np.testing.assert_array_equal(raw[1][material.fluid],0.)
                assert not np.signbit(raw[1][material.fluid]).any()
                assert np.linalg.norm(raw[0][material.fluid & (np.indices(c.shape)[0]>0)])>0
                if fs:
                    np.testing.assert_array_equal(raw[0][0,material.fluid[0]],0.)
                    assert not np.signbit(raw[0][0,material.fluid[0]]).any()
                if pattern!="homogeneous":
                    near=(~material.fluid)&np.roll(material.fluid,1,0)
                    assert np.linalg.norm(raw[1][near])>0
                assert c.api.fluid_singular_arithmetic()==0
                ref.record("serial",pattern=pattern,fs=fs,fw=fw,segments=segments,direction=name,
                           component=component,fd_relative=float(fd),**metric)
        # FULL/replay retain actual fluid strains and exact continuation.
        q=np.empty((4,*c.shape),np.float32)
        c.check(c.api.denise_elastic_psv_born_copy_strain(c.c,60,f.fp(q)))
        assert np.linalg.norm(q[:,material.fluid])>0
        c.check(c.api.denise_elastic_psv_born_checkpoint_roundtrip(c.c,43))
    finally:c.close()


@pytest.mark.parametrize("fs,fw",[(0,0),(1,3)])
@pytest.mark.parametrize("segments",[0,3,7])
def test_invalid_fluid_dmu_before_output_and_recovery(fluid_library,fs,fw,segments):
    exp=ref.fixture("horizontal",fs,fw);c=context(fluid_library,exp,fs,segments)
    try:
        c.prepare();dl,dm=z.directions(z.material(exp.rho,exp.lam,exp.mu))["joint"]
        dl,dm=(a.astype(np.float32) for a in (dl,dm))
        expected=j(c,dl,dm);before=c.snapshot()
        bad=dm.copy();bad[1,1]=1.;out=np.full(c.data_shape,37,np.float32)
        assert c.api.denise_elastic_psv_born_apply_j(c.c,f.fp(dl),f.fp(bad),f.fp(out))!=0
        assert b"nonzero fluid dMu" in c.api.denise_elastic_psv_born_last_error()
        assert np.all(out==37) and c.snapshot()==before
        assert j(c,dl,dm).tobytes()==expected.tobytes()
        dm[c.mu==0]=-0.
        assert j(c,dl,dm).tobytes()==expected.tobytes()
        for p,value in (((1,1),1.),((exp.ny-2,1),0.)):
            mu=c.mu.copy();mu[p]=value;out.fill(37);before=c.snapshot()
            assert c.api.denise_elastic_psv_born_nonlinear(c.c,f.fp(c.lam),f.fp(mu),f.fp(out))!=0
            assert np.all(out==37) and c.snapshot()==before
            assert j(c,dl,dm).tobytes()==expected.tobytes()
        ref.record("serial_transaction",fs=fs,fw=fw,segments=segments,unchanged=True,recovered=True)
    finally:c.close()


@pytest.mark.parametrize("fs,fw",[(0,0),(1,2)])
def test_every_allowed_dense_column(fluid_library,fs,fw):
    exp=ref.fixture("horizontal",fs,fw,nx=8,ny=8,nt=120)
    c=context(fluid_library,exp,fs,3)
    try:
        c.prepare();m=z.material(c.rho,c.lam,c.mu)
        error=scale=0.;count=0
        for channel,y,x in z.dense_columns(m):
            dl=np.zeros(c.shape,np.float32);dm=dl.copy()
            (dl if channel=="lambda" else dm)[y,x]=1.
            analytic=j(c,dl,dm).astype(float)
            plus=nonlinear(c,c.lam+.05*dl,c.mu+.05*dm).astype(float)
            minus=nonlinear(c,c.lam-.05*dl,c.mu-.05*dm).astype(float)
            error+=np.sum(((plus-minus)/.1-analytic)**2);scale+=np.sum(analytic**2);count+=1
        metric=float(np.sqrt(error/scale))
        assert metric<=.004,metric
        ref.record("dense_columns",fs=fs,fw=fw,columns=count,epsilon=.05,relative_l2=metric)
    finally:c.close()


@pytest.mark.parametrize("fs,fw",[(0,0),(1,3)])
def test_full_replay_bitwise(fluid_library,fs,fw):
    exp=ref.fixture("offset",fs,fw);contexts=[context(fluid_library,exp,fs,s) for s in (0,1,3,7)]
    try:
        m=z.material(exp.rho,exp.lam,exp.mu);dl,dm=z.directions(m)["joint"]
        data=np.random.default_rng(973).normal(size=contexts[0].data_shape).astype(np.float32)
        products=[]
        for c in contexts:
            products.append((c.prepare(),j(c,dl,dm),jt(c,data)))
        for actual in products[1:]:
            for a,b in zip(actual,products[0]):assert a.tobytes()==b.tobytes()
        ref.record("full_replay",fs=fs,fw=fw,segments=[0,1,3,7],bitwise=True)
    finally:
        for c in contexts:c.close()


@pytest.mark.parametrize("segments",[0,3])
def test_preserved_homogeneous_pressure_only_authority(fluid_library,segments):
    c=f.Context(fluid_library,segments=segments)
    try:
        got=c.prepare();data,state,q,_=c.products()
        assert got.tobytes()==data.tobytes()
        m=z.material(c.rho,c.lam,c.mu)
        expected=z.acoustic_forward()[0]
        z.require_homogeneous_products(m,got,c.maps()[2],state[4],expected)
        np.testing.assert_array_equal(state[2],state[3])
        assert np.linalg.norm(q)>0 and np.linalg.norm(q[1])+np.linalg.norm(q[2])>0
        ref.record("homogeneous_acoustic",segments=segments,
                   relative_l2=float(np.linalg.norm(got-expected)/np.linalg.norm(expected)),zero_shear=True)
    finally:c.close()
