"""M9d3 mixed-precision production gates against the immutable FP64 oracle."""
import ctypes
from dataclasses import replace
import numpy as np
import pytest

from tests.physics.test_m9b1_elastic_psv_born_production import (
    Operator, born_library, _f32, _ptr, _assert_production_close,
    _production_dot_metrics, _assert_production_dot_closes,
)
from tests.utilities.elastic_psv_free_surface_reference import fixture, forward, born, adjoint
from tests.utilities.elastic_psv_born_reference import dot


@pytest.fixture(scope='module')
def surface_reference():
    exp=fixture()
    data,tapes,checkpoints=forward(exp)
    rng=np.random.default_rng(423)
    dl=.01*exp.lam*rng.normal(size=exp.lam.shape)
    dm=.01*exp.mu*rng.normal(size=exp.mu.shape)
    return exp,data,tapes,checkpoints,dl,dm


def directions(dl,dm,kind):
    return (_f32(np.zeros_like(dl) if kind=='mu' else dl),
            _f32(np.zeros_like(dm) if kind=='lambda' else dm))


def configure_replay(api):
    api.denise_elastic_psv_born_set_replay_segments.argtypes=[ctypes.c_void_p,ctypes.c_int]
    api.denise_elastic_psv_born_checkpoint_roundtrip.argtypes=[ctypes.c_void_p,ctypes.c_int]


@pytest.mark.parametrize('segments',(0,3))
def test_surface_background_and_four_operands(born_library,surface_reference,segments):
    exp,data,tapes,_,_,_=surface_reference
    op=Operator(born_library,exp,free_surface=1)
    configure_replay(born_library)
    try:
        if segments:
            assert born_library.denise_elastic_psv_born_set_replay_segments(op.context,segments)==0
        got=op.prepare()
        _assert_production_close(got,data,2.e-6)
        for k,tape in enumerate(tapes):
            strains=np.empty((4,exp.ny,exp.nx),dtype=np.float32)
            assert born_library.denise_elastic_psv_born_copy_strain(op.context,k,_ptr(strains))==0
            _assert_production_close(strains,np.stack(tape.strain),2.e-6)
        assert born_library.denise_elastic_psv_born_checkpoint_roundtrip(op.context,13)==0
    finally: op.close()


@pytest.mark.parametrize('direction',('lambda','mu','joint'))
@pytest.mark.parametrize('component',(0,1,None),ids=('vx','vy','both'))
def test_surface_j_jt_and_inherited_production_dot(born_library,surface_reference,direction,component):
    exp,_,tapes,_,dl,dm=surface_reference
    dl,dm=directions(dl,dm,direction)
    op=Operator(born_library,exp,free_surface=1)
    try:
        op.prepare()
        j=op.j(dl,dm).astype(np.float64)
        reference_j=born(exp,tapes,dl.astype(np.float64),dm.astype(np.float64))
        _assert_production_close(j,reference_j,1.e-5)
        data=_f32(np.random.default_rng(424).normal(size=j.shape))
        if component is not None: data[...,1-component]=0
        gl,gm=op.jt(data)
        rl,rm=adjoint(exp,tapes,data.astype(np.float64))
        _assert_production_close(gl,rl,6.e-5)
        _assert_production_close(gm,rm,6.e-5)
        scale=max(np.linalg.norm(reference_j)*np.linalg.norm(data),
                  np.linalg.norm(np.stack((dl,dm)))*np.linalg.norm(np.stack((rl,rm))))
        oracle_residual=abs(dot(reference_j,data)-dot(dl,rl)-dot(dm,rm))
        metrics=_production_dot_metrics(j,data,dl,dm,gl,gm,reference_j,rl,rm,
            dict(absolute_ceiling=5.e-13*scale,absolute_residual=oracle_residual))
        _assert_production_dot_closes(metrics)
        print('FP32/FP64_PRODUCTION_DOT',direction,component,metrics)
    finally: op.close()


@pytest.mark.parametrize('direction',('lambda','mu','joint'))
def test_surface_production_centered_fd_fixed_epsilon(born_library,surface_reference,direction):
    exp,_,_,_,dl,dm=surface_reference
    dl,dm=directions(dl,dm,direction)
    op=Operator(born_library,exp,free_surface=1)
    try:
        op.prepare();j=op.j(dl,dm).astype(np.float64)
        eps=.05
        p=op.nonlinear(exp.lam+eps*dl,exp.mu+eps*dm).astype(np.float64)
        m=op.nonlinear(exp.lam-eps*dl,exp.mu-eps*dm).astype(np.float64)
        error=float(np.linalg.norm((p-m)/(2*eps)-j)/np.linalg.norm(j))
        assert error<=.004
        print('FP32/FP64_PRODUCTION_FD',direction,eps,error)
    finally: op.close()


def test_surface_full_segmented_bit_equality(born_library,surface_reference):
    exp,_,_,_,dl,dm=surface_reference
    configure_replay(born_library)
    operators=[Operator(born_library,exp,free_surface=1) for _ in range(2)]
    try:
        assert born_library.denise_elastic_psv_born_set_replay_segments(operators[1].context,3)==0
        np.testing.assert_array_equal(operators[0].prepare(),operators[1].prepare())
        for direction in ('lambda','mu','joint'):
            l,m=directions(dl,dm,direction)
            np.testing.assert_array_equal(operators[0].j(l,m),operators[1].j(l,m))
        data=_f32(np.random.default_rng(424).normal(size=operators[0].data_shape))
        for component in (0,1,None):
            d=data.copy()
            if component is not None: d[...,1-component]=0
            a=operators[0].jt(d);b=operators[1].jt(d)
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    finally:
        for op in operators:op.close()


def test_surface_source_row_one_rejected_and_row_two_supported(born_library):
    with pytest.raises(RuntimeError,match='j=1'):
        Operator(born_library,fixture(source=(4,1)),free_surface=1)
    op=Operator(born_library,fixture(source=(4,2)),free_surface=1)
    try: assert np.any(op.prepare())
    finally:op.close()


def test_surface_cpml_profiles_do_not_overlap_closure(born_library):
    with pytest.raises(RuntimeError,match='overlaps'):
        Operator(born_library,replace(fixture(),ny=5),free_surface=1)
