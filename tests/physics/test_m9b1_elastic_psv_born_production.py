"""Production gates for the M9b-1 elastic P/SV Born operator."""

from __future__ import annotations

import ctypes
from dataclasses import replace
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.utilities.elastic_psv_born_reference import (
    _source_samples,
    born_adjoint,
    born_forward,
    deterministic_data,
    deterministic_model_direction,
    dot_metrics,
    image_diagnostics,
    lambda_reflector,
    make_experiment,
    migrate_shot_stack,
    mu_compact_direction,
    nonlinear_forward,
)


F32P = ctypes.POINTER(ctypes.c_float)
I32P = ctypes.POINTER(ctypes.c_int)
F64P = ctypes.POINTER(ctypes.c_double)

# These independently accepted full-operator contracts include the complete
# FD4/CPML recurrence and its FP32 dynamic state.  The production dot bound
# below combines them with the frozen M9b-0 FP64 transpose bound by the triangle
# inequality; it therefore needs no interface-count or observed-residual factor.
PRODUCTION_J_RELATIVE_L2_CEILING = 1.0e-5
PRODUCTION_JT_RELATIVE_L2_CEILING = 6.0e-5
FALSIFICATION_PAIR_RELATIVE_MIN = 2.5e-2


class Config(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_int), ("ny", ctypes.c_int), ("nt", ctypes.c_int),
        ("fw", ctypes.c_int), ("dh", ctypes.c_float), ("dt", ctypes.c_float),
        ("l", ctypes.c_int), ("invmat1", ctypes.c_int), ("fdorder", ctypes.c_int),
        ("ndt", ctypes.c_int), ("dtinv", ctypes.c_int),
        ("free_surface", ctypes.c_int), ("boundary", ctypes.c_int),
        ("mpi_size", ctypes.c_int), ("receiver_components", ctypes.c_int),
        ("lambda_", F32P), ("mu", F32P), ("rho", F32P),
        ("source_i", ctypes.c_int), ("source_j", ctypes.c_int),
        ("source_samples", F32P), ("receiver_count", ctypes.c_int),
        ("receiver_i", I32P), ("receiver_j", I32P),
        ("cpml_enabled", ctypes.c_int),
        ("pml_reflection", ctypes.c_float), ("pml_power", ctypes.c_float),
        ("pml_kmax", ctypes.c_float), ("pml_fpml", ctypes.c_float),
        ("pml_damping_speed", ctypes.c_float),
    ]


def _f32(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)


def _ptr(array: np.ndarray, pointer_type=F32P):
    return array.ctypes.data_as(pointer_type)


@pytest.fixture(scope="session")
def born_library(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("cc") or shutil.which("gcc")
    assert compiler, "a C99 compiler is required for the M9b-1 production gate"
    output = tmp_path_factory.mktemp("m9b1") / "libdenise_m9b1.so"
    subprocess.run([
        compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
        "-pedantic", "-fPIC", "-shared", "-I", str(repository_root / "include"),
        str(repository_root / "src/PSV/elastic_psv_born.c"), "-lm", "-o", str(output),
    ], check=True)
    api = ctypes.CDLL(str(output))
    api.denise_elastic_psv_born_last_error.restype = ctypes.c_char_p
    api.denise_elastic_psv_born_create.argtypes = [ctypes.POINTER(Config), ctypes.POINTER(ctypes.c_void_p)]
    api.denise_elastic_psv_born_create.restype = ctypes.c_int
    api.denise_elastic_psv_born_prepare.argtypes = [ctypes.c_void_p, F32P]
    api.denise_elastic_psv_born_prepare.restype = ctypes.c_int
    api.denise_elastic_psv_born_apply_j.argtypes = [ctypes.c_void_p, F32P, F32P, F32P]
    api.denise_elastic_psv_born_apply_j.restype = ctypes.c_int
    api.denise_elastic_psv_born_apply_jt.argtypes = [ctypes.c_void_p, F32P, F64P, F64P]
    api.denise_elastic_psv_born_apply_jt.restype = ctypes.c_int
    api.denise_elastic_psv_born_nonlinear.argtypes = [ctypes.c_void_p, F32P, F32P, F32P]
    api.denise_elastic_psv_born_nonlinear.restype = ctypes.c_int
    api.denise_elastic_psv_born_copy_strain.argtypes = [ctypes.c_void_p, ctypes.c_int, F32P]
    api.denise_elastic_psv_born_copy_strain.restype = ctypes.c_int
    api.denise_elastic_psv_born_is_prepared.argtypes = [ctypes.c_void_p]
    api.denise_elastic_psv_born_is_prepared.restype = ctypes.c_int
    api.denise_elastic_psv_born_cpml_memory_peak.argtypes = [ctypes.c_void_p]
    api.denise_elastic_psv_born_cpml_memory_peak.restype = ctypes.c_float
    api.denise_elastic_psv_born_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    return api


class Operator:
    def __init__(self, api, exp, shot=0, **overrides):
        self.api, self.exp = api, exp
        self.lam, self.mu, self.rho = map(_f32, (exp.lam, exp.mu, exp.rho))
        self.source = _f32(_source_samples(exp))
        self.ri = np.asarray([i - 1 for i, _ in exp.receivers], dtype=np.int32)
        self.rj = np.asarray([j - 1 for _, j in exp.receivers], dtype=np.int32)
        si, sj = exp.sources[shot]
        values = dict(
            nx=exp.nx, ny=exp.ny, nt=exp.nt, fw=exp.fw,
            dh=exp.dh, dt=exp.dt, l=0, invmat1=3, fdorder=4, ndt=1, dtinv=1,
            free_surface=0, boundary=0, mpi_size=1, receiver_components=2,
            source_i=si - 1, source_j=sj - 1, receiver_count=len(exp.receivers),
            cpml_enabled=int(exp.cpml), pml_reflection=exp.pml_reflection,
            pml_power=exp.pml_power, pml_kmax=exp.pml_kmax,
            pml_fpml=exp.pml_fpml,
            pml_damping_speed=float(np.sqrt(np.max((exp.lam + 2.0 * exp.mu) / exp.rho))),
        )
        values.update(overrides)
        self.config = Config(
            values["nx"], values["ny"], values["nt"], values["fw"],
            values["dh"], values["dt"], values["l"], values["invmat1"], values["fdorder"],
            values["ndt"], values["dtinv"], values["free_surface"],
            values["boundary"], values["mpi_size"], values["receiver_components"],
            _ptr(self.lam), _ptr(self.mu), _ptr(self.rho),
            values["source_i"], values["source_j"], _ptr(self.source),
            values["receiver_count"], _ptr(self.ri, I32P), _ptr(self.rj, I32P),
            values["cpml_enabled"], values["pml_reflection"], values["pml_power"],
            values["pml_kmax"], values["pml_fpml"], values["pml_damping_speed"],
        )
        self.context = ctypes.c_void_p()
        status = api.denise_elastic_psv_born_create(ctypes.byref(self.config), ctypes.byref(self.context))
        if status:
            raise RuntimeError(api.denise_elastic_psv_born_last_error().decode())

    @property
    def data_shape(self):
        return self.exp.nt, len(self.exp.receivers), 2

    def prepare(self):
        out = np.empty(self.data_shape, dtype=np.float32)
        assert self.api.denise_elastic_psv_born_prepare(self.context, _ptr(out)) == 0
        return out

    def j(self, dlam, dmu):
        out = np.empty(self.data_shape, dtype=np.float32)
        assert self.api.denise_elastic_psv_born_apply_j(
            self.context, _ptr(_f32(dlam)), _ptr(_f32(dmu)), _ptr(out)) == 0
        return out

    def jt(self, data):
        gl = np.empty((self.exp.ny, self.exp.nx), dtype=np.float64)
        gm = np.empty_like(gl)
        assert self.api.denise_elastic_psv_born_apply_jt(
            self.context, _ptr(_f32(data)), _ptr(gl, F64P), _ptr(gm, F64P)) == 0
        return gl, gm

    def nonlinear(self, lam, mu):
        out = np.empty(self.data_shape, dtype=np.float32)
        assert self.api.denise_elastic_psv_born_nonlinear(
            self.context, _ptr(_f32(lam)), _ptr(_f32(mu)), _ptr(out)) == 0
        return out

    def close(self):
        if self.context:
            self.api.denise_elastic_psv_born_destroy(ctypes.byref(self.context))


def _errors(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    difference = actual - expected
    peak = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    norm = max(float(np.linalg.norm(expected)), np.finfo(float).tiny)
    return {
        "max_absolute": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / norm),
        "peak_relative": float(np.max(np.abs(difference)) / peak),
    }


def _assert_production_close(actual, expected, relative=3.0e-5):
    metrics = _errors(actual, expected)
    assert metrics["relative_l2"] < relative, metrics
    assert metrics["peak_relative"] < 4.0 * relative, metrics


def _norm(*arrays):
    return math.sqrt(math.fsum(
        float(value) * float(value)
        for array in arrays
        for value in np.asarray(array).ravel()))


def _production_dot_metrics(jdm, data, dlam, dmu, glambda, gmu,
                            reference_jdm, reference_glambda, reference_gmu,
                            oracle_metrics):
    """Oracle-bounded metrics for <Jp m,d> = <m,Ap d>."""
    jdm = np.asarray(jdm, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    dlam = np.asarray(_f32(dlam), dtype=np.float64)
    dmu = np.asarray(_f32(dmu), dtype=np.float64)
    glambda = np.asarray(glambda, dtype=np.float64)
    gmu = np.asarray(gmu, dtype=np.float64)
    reference_jdm = np.asarray(reference_jdm, dtype=np.float64)
    reference_glambda = np.asarray(reference_glambda, dtype=np.float64)
    reference_gmu = np.asarray(reference_gmu, dtype=np.float64)
    lhs = math.fsum(float(x) * float(y) for x, y in zip(jdm.ravel(), data.ravel()))
    rhs = math.fsum(float(x) * float(y) for x, y in zip(dlam.ravel(), glambda.ravel()))
    rhs += math.fsum(float(x) * float(y) for x, y in zip(dmu.ravel(), gmu.ravel()))
    absolute_residual = abs(lhs - rhs)
    lhs_operand_scale = _norm(jdm) * _norm(data)
    rhs_operand_scale = _norm(dlam, dmu) * _norm(glambda, gmu)
    reference_j_norm = _norm(reference_jdm)
    reference_jt_norm = _norm(reference_glambda, reference_gmu)
    model_norm = _norm(dlam, dmu)
    data_norm = _norm(data)
    forward_relative_l2 = _norm(jdm - reference_jdm) / max(
        reference_j_norm, np.finfo(float).tiny)
    adjoint_relative_l2 = _norm(
        glambda - reference_glambda, gmu - reference_gmu) / max(
            reference_jt_norm, np.finfo(float).tiny)
    forward_error_bound = (
        PRODUCTION_J_RELATIVE_L2_CEILING * reference_j_norm * data_norm)
    adjoint_error_bound = (
        PRODUCTION_JT_RELATIVE_L2_CEILING * model_norm * reference_jt_norm)
    oracle_dot_bound = oracle_metrics["absolute_ceiling"]
    absolute_ceiling = forward_error_bound + oracle_dot_bound + adjoint_error_bound
    tiny = np.finfo(float).tiny
    operand_scale = max(lhs_operand_scale, rhs_operand_scale, tiny)
    pair_scale = max(abs(lhs), abs(rhs), tiny)
    pair_condition = (lhs_operand_scale + rhs_operand_scale) / pair_scale
    return {
        "lhs": lhs,
        "rhs": rhs,
        "absolute_residual": absolute_residual,
        "lhs_operand_scale": lhs_operand_scale,
        "rhs_operand_scale": rhs_operand_scale,
        "forward_relative_l2": forward_relative_l2,
        "adjoint_relative_l2": adjoint_relative_l2,
        "forward_error_bound": forward_error_bound,
        "oracle_dot_bound": oracle_dot_bound,
        "oracle_dot_residual": oracle_metrics["absolute_residual"],
        "adjoint_error_bound": adjoint_error_bound,
        "production_dot_absolute_ceiling": absolute_ceiling,
        "operand_scale": operand_scale,
        "operand_scale_residual": absolute_residual / operand_scale,
        "operand_scale_ceiling": absolute_ceiling / operand_scale,
        "pair_relative": absolute_residual / pair_scale,
        "pair_condition": pair_condition,
        "pair_relative_ceiling": absolute_ceiling / pair_scale,
    }


def _assert_production_dot_closes(metrics):
    assert metrics["forward_relative_l2"] <= PRODUCTION_J_RELATIVE_L2_CEILING, metrics
    assert metrics["adjoint_relative_l2"] <= PRODUCTION_JT_RELATIVE_L2_CEILING, metrics
    assert metrics["oracle_dot_residual"] <= metrics["oracle_dot_bound"], metrics
    assert metrics["absolute_residual"] <= metrics["production_dot_absolute_ceiling"], metrics


@pytest.mark.parametrize("case", ("interior", "active_cpml"))
def test_production_background_j_and_jt_match_frozen_oracle(born_library, case):
    exp = make_experiment(case)
    oracle = nonlinear_forward(exp, 0, save_strain=True)
    operator = Operator(born_library, exp)
    try:
        background = operator.prepare()
        _assert_production_close(background, oracle.data, 2.0e-6)
        if exp.cpml:
            assert born_library.denise_elastic_psv_born_cpml_memory_peak(operator.context) > 0.0
        dlam, dmu = deterministic_model_direction(exp)
        zero = np.zeros_like(dlam)
        for dl, dm in ((dlam, zero), (zero, dmu), (dlam, dmu)):
            _assert_production_close(operator.j(dl, dm), born_forward(exp, oracle, dl, dm), 1.0e-5)
        for component in (0, 1, None):
            data = deterministic_data(exp, component=component)
            gl, gm = operator.jt(data)
            reference = born_adjoint(exp, oracle, data)
            _assert_production_close(gl, reference.image_lambda_raw, 6.0e-5)
            _assert_production_close(gm, reference.image_mu_raw, 6.0e-5)
    finally:
        operator.close()


@pytest.mark.parametrize("case", ("interior", "active_cpml"))
def test_production_dot_product_matrix_closes(born_library, case):
    exp = make_experiment(case)
    reference_trajectory = nonlinear_forward(exp, 0, save_strain=True)
    operator = Operator(born_library, exp)
    operator.prepare()
    try:
        dlam, dmu = deterministic_model_direction(exp)
        zero = np.zeros_like(dlam)
        for dl, dm in ((dlam, zero), (zero, dmu), (dlam, dmu)):
            dl = np.asarray(_f32(dl), dtype=np.float64)
            dm = np.asarray(_f32(dm), dtype=np.float64)
            jdm = operator.j(dl, dm).astype(np.float64)
            reference_jdm = born_forward(
                exp, reference_trajectory, dl, dm)
            for component in (0, 1, None):
                data = deterministic_data(exp, component=component)
                gl, gm = operator.jt(data)
                reference_image = born_adjoint(
                    exp, reference_trajectory, data)
                oracle_metrics = dot_metrics(
                    exp, dl, dm, data, reference_trajectory)
                metrics = _production_dot_metrics(
                    jdm, data, dl, dm, gl, gm,
                    reference_jdm, reference_image.image_lambda_raw,
                    reference_image.image_mu_raw, oracle_metrics)
                _assert_production_dot_closes(metrics)
    finally:
        operator.close()


def test_production_dense_cell_finite_differences(born_library):
    exp = make_experiment("interior")
    operator = Operator(born_library, exp)
    operator.prepare()
    try:
        for channel, j, i in (("lambda", 18, 20), ("mu", 16, 18), ("mu", 19, 21)):
            direction_l = np.zeros_like(exp.lam)
            direction_m = np.zeros_like(exp.mu)
            base = exp.lam[j, i] if channel == "lambda" else exp.mu[j, i]
            (direction_l if channel == "lambda" else direction_m)[j, i] = base
            analytic = operator.j(direction_l, direction_m).astype(np.float64)
            # Five percent of one isolated cell is still a small global model
            # perturbation and keeps the FP32 gather difference above its ULPs.
            epsilon = 5.0e-2
            plus_l = exp.lam + epsilon * direction_l
            minus_l = exp.lam - epsilon * direction_l
            plus_m = exp.mu + epsilon * direction_m
            minus_m = exp.mu - epsilon * direction_m
            finite = (operator.nonlinear(plus_l, plus_m).astype(np.float64)
                      - operator.nonlinear(minus_l, minus_m).astype(np.float64)) / (2.0 * epsilon)
            active = np.abs(analytic) > max(np.max(np.abs(analytic)) * 1.0e-5, 1.0e-20)
            assert np.linalg.norm((finite - analytic)[active]) / np.linalg.norm(analytic[active]) < 4.0e-3
    finally:
        operator.close()


def test_short_active_cpml_micro_oracle_covers_tangent_and_transpose(born_library):
    """Four steps isolate the first nonzero Born recurrence near active CPML."""
    base = make_experiment("active_cpml")
    y, x = np.mgrid[1:base.ny + 1, 1:base.nx + 1]
    # Asymmetry prevents cancellations from hiding a component or stagger error.
    rho = base.rho * (1.0 + 0.003 * np.sin(0.17 * x + 0.11 * y))
    lam = base.lam * (1.0 + 0.004 * np.cos(0.13 * x - 0.19 * y))
    mu = base.mu * (1.0 + 0.005 * np.sin(0.07 * x * y + 0.23))
    exp = replace(
        base, nt=4, rho=rho, lam=lam, mu=mu,
        sources=((2, 3),), receivers=((2, 3), (3, 2), (5, 4)),
        source_t0=5.0e-4,
    )
    reference = nonlinear_forward(exp, 0, save_strain=True)
    dlam, dmu = deterministic_model_direction(exp, seed=113)
    prepared = deterministic_data(exp, seed=127)
    operator = Operator(born_library, exp)
    try:
        _assert_production_close(operator.prepare(), reference.data, 2.0e-6)
        strain = np.empty((4, exp.ny, exp.nx), dtype=np.float32)
        assert born_library.denise_elastic_psv_born_copy_strain(
            operator.context, 1, _ptr(strain)) == 0
        _assert_production_close(strain, reference.strain[1], 3.0e-6)
        _assert_production_close(
            operator.j(dlam, dmu), born_forward(exp, reference, dlam, dmu), 4.0e-6)
        gl, gm = operator.jt(prepared)
        expected = born_adjoint(exp, reference, prepared)
        _assert_production_close(gl, expected.image_lambda_raw, 4.0e-6)
        _assert_production_close(gm, expected.image_mu_raw, 4.0e-6)
        assert born_library.denise_elastic_psv_born_cpml_memory_peak(operator.context) > 0.0
    finally:
        operator.close()


def test_targeted_mutations_are_numerically_falsified(born_library):
    """Deliberate operator errors miss the production identity by a wide margin."""
    exp = make_experiment("active_cpml")
    operator = Operator(born_library, exp)
    operator.prepare()
    try:
        dlam, dmu = deterministic_model_direction(exp, seed=61)
        dlam = np.asarray(_f32(dlam), dtype=np.float64)
        dmu = np.asarray(_f32(dmu), dtype=np.float64)
        data = deterministic_data(exp, seed=97)
        jdm = operator.j(dlam, dmu).astype(np.float64)
        gl, gm = operator.jt(data)
        reference_trajectory = nonlinear_forward(exp, 0, save_strain=True)
        reference_jdm = born_forward(
            exp, reference_trajectory, dlam, dmu)
        reference_image = born_adjoint(
            exp, reference_trajectory, data)
        oracle_metrics = dot_metrics(
            exp, dlam, dmu, data, reference_trajectory)

        def mutation_metrics(mutated_jdm, mutated_gl, mutated_gm):
            return _production_dot_metrics(
                mutated_jdm, data, dlam, dmu, mutated_gl, mutated_gm,
                reference_jdm, reference_image.image_lambda_raw,
                reference_image.image_mu_raw, oracle_metrics)

        correct = mutation_metrics(jdm, gl, gm)
        _assert_production_dot_closes(correct)

        shifted_receiver = np.roll(jdm, 1, axis=1)
        shifted_minus = np.zeros_like(jdm); shifted_minus[:-1] = jdm[1:]
        shifted_plus = np.zeros_like(jdm); shifted_plus[1:] = jdm[:-1]
        component_swap = jdm[..., ::-1]
        mutations = {
            "receiver_component_swap": mutation_metrics(
                component_swap, gl, gm),
            "receiver_location_shift": mutation_metrics(
                shifted_receiver, gl, gm),
            "minus_one_timestep": mutation_metrics(
                shifted_minus, gl, gm),
            "plus_one_timestep": mutation_metrics(
                shifted_plus, gl, gm),
            "global_adjoint_sign": mutation_metrics(
                jdm, -gl, -gm),
            "lambda_mu_channel_swap": mutation_metrics(
                jdm, gm, gl),
            "wrong_harmonic_shear_transpose": mutation_metrics(
                jdm, gl, np.roll(gm, 1, axis=1)),
        }
        # Normal-mu omission mutation: a checkerboard dmu/mu^2 has a zero sum
        # in every four-cell corner,
        # so H_mu' dmu is exactly zero. Its Born data therefore contains only
        # the normal-mu term; replacing that image term by zero must fail.
        yy, xx = np.mgrid[:exp.ny, :exp.nx]
        normal_only_dmu = 1.0e-11 * exp.mu**2 * np.where((xx + yy) % 2, -1.0, 1.0)
        normal_only_dmu = np.asarray(_f32(normal_only_dmu), dtype=np.float64)
        normal_only_dlam = np.zeros_like(normal_only_dmu)
        normal_only_j = operator.j(np.zeros_like(dlam), normal_only_dmu).astype(np.float64)
        normal_reference_j = born_forward(
            exp, reference_trajectory, normal_only_dlam, normal_only_dmu)
        normal_oracle_metrics = dot_metrics(
            exp, normal_only_dlam, normal_only_dmu, data,
            reference_trajectory)
        mutations["omit_normal_mu"] = _production_dot_metrics(
            normal_only_j, data, normal_only_dlam, normal_only_dmu,
            np.zeros_like(gl), np.zeros_like(gm), normal_reference_j,
            reference_image.image_lambda_raw, reference_image.image_mu_raw,
            normal_oracle_metrics)

        # CPML-disabled reverse mutation: a FW=0 transpose is intentionally
        # paired with active-CPML forward data and therefore cannot close.
        no_cpml_exp = replace(exp, fw=0, cpml=False)
        no_cpml = Operator(born_library, no_cpml_exp)
        no_cpml.prepare()
        try:
            bad_gl, bad_gm = no_cpml.jt(data)
            mutations["cpml_transpose_disabled"] = mutation_metrics(
                jdm, bad_gl, bad_gm)
        finally:
            no_cpml.close()
        assert min(item["pair_relative"] for item in mutations.values()) > (
            FALSIFICATION_PAIR_RELATIVE_MIN), mutations
        assert all(item["absolute_residual"] >
                   item["production_dot_absolute_ceiling"]
                   for item in mutations.values()), mutations
    finally:
        operator.close()


def test_reflector_normal_two_shot_linearity_and_lifecycle(born_library):
    exp = make_experiment("active_cpml")
    dlam, dmu = lambda_reflector(exp), np.zeros_like(exp.mu)
    images = []
    oracle_trajectories = []
    oracle_data = []
    for shot in range(2):
        operator = Operator(born_library, exp, shot=shot)
        operator.prepare()
        data = operator.j(dlam, dmu)
        first = operator.jt(data)
        second = operator.jt(data)
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        images.append(first)
        operator.close()
        trajectory = nonlinear_forward(exp, shot, save_strain=True)
        oracle_trajectories.append(trajectory)
        oracle_data.append(born_forward(exp, trajectory, dlam, dmu))
    stacked = (images[0][0] + images[1][0], images[0][1] + images[1][1])
    oracle_stack = migrate_shot_stack(exp, oracle_trajectories, oracle_data)
    _assert_production_close(stacked[0], oracle_stack.image_lambda_raw, 6.0e-5)
    _assert_production_close(stacked[1], oracle_stack.image_mu_raw, 6.0e-5)
    diagnostics = image_diagnostics(exp, type("Image", (), {
        "image_lambda_raw": stacked[0], "image_mu_raw": stacked[1]})())
    assert abs(diagnostics["peak_j"] - 34.0) <= 1.0
    assert diagnostics["lambda_energy_fraction_near_reflector"] > 0.25
    assert diagnostics["lateral_symmetry_relative_residual"] < 0.1


def test_repeated_j_and_jt_match_fresh_contexts(born_library):
    exp = make_experiment("interior")
    dlam1, dmu1 = deterministic_model_direction(exp, seed=31)
    dlam2, dmu2 = deterministic_model_direction(exp, seed=73)
    data1 = deterministic_data(exp, seed=29)
    data2 = deterministic_data(exp, seed=101)
    shared = Operator(born_library, exp)
    shared.prepare()
    try:
        shared_outputs = (shared.j(dlam1, dmu1), shared.j(dlam2, dmu2),
                          shared.jt(data1), shared.jt(data2))
    finally:
        shared.close()
    fresh_outputs = []
    for dl, dm, data in ((dlam1, dmu1, data1), (dlam2, dmu2, data2)):
        fresh = Operator(born_library, exp)
        fresh.prepare()
        try:
            fresh_outputs.extend((fresh.j(dl, dm), fresh.jt(data)))
        finally:
            fresh.close()
    assert np.array_equal(shared_outputs[0], fresh_outputs[0])
    assert np.array_equal(shared_outputs[1], fresh_outputs[2])
    for got, expected in zip(shared_outputs[2], fresh_outputs[1]):
        assert np.array_equal(got, expected)
    for got, expected in zip(shared_outputs[3], fresh_outputs[3]):
        assert np.array_equal(got, expected)


@pytest.mark.parametrize("field,value,fragment", [
    ("l", 1, "L=0"), ("invmat1", 1, "INVMAT1=3"),
    ("fdorder", 2, "FDORDER=4"),
    ("ndt", 2, "NDT=DTINV=1"), ("dtinv", 2, "NDT=DTINV=1"),
    ("free_surface", 2, "FREE_SURF"), ("boundary", 1, "BOUNDARY"),
    ("mpi_size", 2, "one MPI rank"), ("receiver_components", 1, "vx/vy"),
])
def test_failure_closed_envelope(born_library, field, value, fragment):
    exp = make_experiment("interior")
    with pytest.raises(RuntimeError, match=fragment):
        Operator(born_library, exp, **{field: value})
