"""M6.3c-6a exact local SH material-map VJP verification."""

from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import random
import shutil
import subprocess

import pytest

from tests.utilities.m63c_material_map_reference import (
    C6A_DOUBLE_DOT_RELATIVE_MAX,
    C6A_FD_RELATIVE_MAX,
    C6A_REFERENCE_RELATIVE_MAX,
    QMapping,
    dot,
    forward,
    harmonic_jvp,
    harmonic_vjp,
    jvp,
    physical_mapping,
    q_to_tau,
    q_to_tau_derivative,
    relative,
    vjp,
)


class CMapping(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("sample_count", ctypes.c_int),
        ("inverse_tau_per_q", ctypes.c_double),
        ("inverse_tau_offset", ctypes.c_double),
    ]


D4 = ctypes.c_double * 4
D5 = ctypes.c_double * 5


@pytest.fixture(scope="module")
def c6a(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for M6.3c-6a"
    library = tmp_path_factory.mktemp("m63c6a") / "libm63c6a.so"
    command = [
        compiler, "-std=c99", "-O2", "-fPIC", "-shared", "-fcommon",
        "-I", str(repository_root / "include"),
        str(repository_root / "tests/utilities/m63c_material_map_harness.c"),
        str(repository_root / "src/q_parameterization.c"),
        str(repository_root / "src/SH/visco_sh_material_vjp.c"),
        "-o", str(library), "-lm",
    ]
    result = subprocess.run(command, cwd=repository_root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    api = ctypes.CDLL(str(library))
    api.m63c6a_init_mapping.argtypes = [
        ctypes.POINTER(CMapping), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float,
        ctypes.c_float,
    ]
    api.m63c6a_q_to_tau.argtypes = [ctypes.c_float, ctypes.POINTER(CMapping)]
    api.m63c6a_q_to_tau.restype = ctypes.c_float
    api.m63c6a_q_derivative.argtypes = [ctypes.c_float, ctypes.POINTER(CMapping)]
    api.m63c6a_q_derivative.restype = ctypes.c_double
    api.visco_sh_harmonic_pair_vjp.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    api.visco_sh_harmonic_pair_vjp.restype = ctypes.c_int
    api.visco_sh_av_tau_local_vjp.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)
    ]
    api.visco_sh_rhoi_value.argtypes = [ctypes.c_double]
    api.visco_sh_rhoi_value.restype = ctypes.c_double
    api.visco_sh_rhoi_vjp.argtypes = [ctypes.c_double, ctypes.c_double]
    api.visco_sh_rhoi_vjp.restype = ctypes.c_double
    api.visco_sh_velocity_rhoi_vjp.argtypes = [ctypes.c_double] * 5
    api.visco_sh_velocity_rhoi_vjp.restype = ctypes.c_double
    api.visco_sh_material_patch_forward.argtypes = [
        ctypes.c_int, ctypes.POINTER(CMapping), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    api.visco_sh_material_patch_forward.restype = ctypes.c_int
    api.visco_sh_material_patch_vjp.argtypes = [
        ctypes.c_int, ctypes.POINTER(CMapping), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    api.visco_sh_material_patch_vjp.restype = ctypes.c_int
    return api


def mappings(c6a):
    legacy = CMapping()
    frequencies = (ctypes.c_float * 1)(6.0)
    c6a.m63c6a_init_mapping(
        ctypes.byref(legacy), 0, 1, frequencies, 2.0, 18.0, 0.5
    )
    physical = CMapping()
    values = (ctypes.c_float * 4)(0.0, 3.0, 7.0, 13.0)
    c6a.m63c6a_init_mapping(
        ctypes.byref(physical), 1, 3, values, 2.0, 18.0, 0.5
    )
    return {
        "legacy": (legacy, QMapping(0)),
        "physical": (
            physical, physical_mapping((3.0, 7.0, 13.0), 2.0, 18.0, 0.5)
        ),
    }


def c_forward(c6a, invmat1, mapping, primary, rho_values, q_values):
    output = D5()
    status = c6a.visco_sh_material_patch_forward(
        invmat1, ctypes.byref(mapping), D4(*primary), D4(*rho_values),
        D4(*q_values), output
    )
    assert status == 0
    return list(output)


def c_vjp(c6a, invmat1, mapping, primary, rho_values, q_values, bars):
    outputs = (D4(), D4(), D4())
    status = c6a.visco_sh_material_patch_vjp(
        invmat1, ctypes.byref(mapping), D4(*primary), D4(*rho_values),
        D4(*q_values), D5(*bars), *outputs
    )
    assert status == 0
    return tuple(list(value) for value in outputs)


def case(invmat1):
    primary = [1800.0, 2450.0, 1325.0, 2150.0] if invmat1 == 1 else [
        7.2e9, 12.4e9, 4.9e9, 9.1e9
    ]
    return primary, [2100.0, 2350.0, 1980.0, 2260.0], [24.0, 41.0, 73.0, 118.0]


def test_harmonic_vjp_double_dot_reference_and_domain(c6a):
    maximum = 0.0
    for left, right, dleft, dright, bar in (
        (1.2, 19.0, 0.3, -0.8, 1.7),
        (9.0e8, 1.7e10, -2.1e7, 4.4e8, -0.25),
        (31.0, 2.0, 1.1, 0.2, 3.0),
    ):
        c_left, c_right = ctypes.c_double(), ctypes.c_double()
        assert c6a.visco_sh_harmonic_pair_vjp(
            left, right, bar, ctypes.byref(c_left), ctypes.byref(c_right)
        ) == 0
        expected = harmonic_vjp(left, right, bar)
        reference_error = max(relative(c_left.value, expected[0]),
                              relative(c_right.value, expected[1]))
        lhs = harmonic_jvp(left, right, dleft, dright) * bar
        rhs = dleft * c_left.value + dright * c_right.value
        maximum = max(maximum, relative(lhs, rhs), reference_error)
    bad_left, bad_right = ctypes.c_double(), ctypes.c_double()
    assert c6a.visco_sh_harmonic_pair_vjp(
        0.0, 1.0, 2.0, ctypes.byref(bad_left), ctypes.byref(bad_right)
    ) == -2
    print("M63C6A_HARMONIC " + json.dumps({"maximum": maximum}))
    assert maximum <= C6A_DOUBLE_DOT_RELATIVE_MAX


def test_av_tau_exact_scatter(c6a):
    output = D4(1.0, -2.0, 3.0, -4.0)
    c6a.visco_sh_av_tau_local_vjp(6.0, -1.5, output)
    actual = list(output)
    expected = [1.0, -2.0, 3.0, -4.0]
    for index in range(4):
        expected[index] += 1.5
    expected[0] -= 1.5
    assert actual == expected


def test_q_mapping_derivatives_against_reference_and_forward_fd(c6a):
    diagnostics = []
    for name, (cmapping, reference_mapping) in mappings(c6a).items():
        assert cmapping.sample_count == reference_mapping.sample_count
        assert relative(cmapping.inverse_tau_per_q, reference_mapping.a) <= C6A_REFERENCE_RELATIVE_MAX
        assert relative(cmapping.inverse_tau_offset, reference_mapping.b) <= C6A_REFERENCE_RELATIVE_MAX
        for q in (18.0, 37.0, 95.0, 180.0):
            actual = c6a.m63c6a_q_derivative(q, ctypes.byref(cmapping))
            expected = q_to_tau_derivative(q, reference_mapping)
            reference_error = relative(actual, expected)
            step = 0.02 * q
            values = {
                offset: float(c6a.m63c6a_q_to_tau(
                    q + offset * step, ctypes.byref(cmapping)
                ))
                for offset in (-2, -1, 1, 2)
            }
            fd = (
                values[-2] - 8.0 * values[-1]
                + 8.0 * values[1] - values[2]
            ) / (12.0 * step)
            fd_error = relative(actual, fd)
            diagnostics.append({"mode": name, "q": q,
                                "reference_error": reference_error,
                                "fd_error": fd_error})
            assert reference_error <= C6A_REFERENCE_RELATIVE_MAX
            assert fd_error <= C6A_FD_RELATIVE_MAX
    print("M63C6A_Q " + json.dumps(diagnostics, sort_keys=True))


def test_rhoi_piecewise_derivative(c6a):
    diagnostics = []
    for rho, expected_value, expected_derivative in (
        (2200.0, 1.0 / 2200.0, -1.0 / 2200.0**2),
        (5.0e-5, 0.0, 0.0),
    ):
        assert c6a.visco_sh_rhoi_value(rho) == expected_value
        actual = c6a.visco_sh_rhoi_vjp(rho, 1.0)
        assert actual == expected_derivative
        step = min(1.0e-3 * rho, 1.0e-6)
        plus = c6a.visco_sh_rhoi_value(rho + step)
        minus = c6a.visco_sh_rhoi_value(rho - step)
        fd = (plus - minus) / (2.0 * step)
        error = relative(actual, fd)
        diagnostics.append({"rho": rho, "value": expected_value,
                            "derivative": actual, "fd_error": error})
        assert error <= C6A_FD_RELATIVE_MAX
    print("M63C6A_RHOI " + json.dumps(diagnostics, sort_keys=True))


@pytest.mark.parametrize("qx,qy", ((3.0, 2.0), (3.0, -2.0), (0.0, 4.0), (-1.7, 2.9)))
def test_velocity_rhoi_coefficient_gradient(c6a, qx, qy):
    dt, dh, bar, rhoi_value = 0.0012, 8.0, -1.7, 0.00047
    actual = c6a.visco_sh_velocity_rhoi_vjp(dt, dh, qx, qy, bar)
    step = 1.0e-7
    objective = lambda value: bar * (0.25 + dt / dh * value * (qx + qy))
    fd = (objective(rhoi_value + step) - objective(rhoi_value - step)) / (2.0 * step)
    error = relative(actual, fd)
    print("M63C6A_VELOCITY_RHOI " + json.dumps(
        {"qx": qx, "qy": qy, "analytic": actual, "fd": fd,
         "fd_error": error}, sort_keys=True
    ))
    assert error <= C6A_FD_RELATIVE_MAX


def test_velocity_rhoi_uses_cpml_corrected_terms_not_raw_derivatives(c6a):
    raw_qx, raw_qy = 7.0, -1.0
    corrected_qx, corrected_qy = 1.25, 2.75
    dt, dh, bar = 0.0015, 10.0, -2.0
    actual = c6a.visco_sh_velocity_rhoi_vjp(
        dt, dh, corrected_qx, corrected_qy, bar
    )
    expected = dt / dh * (corrected_qx + corrected_qy) * bar
    raw_result = dt / dh * (raw_qx + raw_qy) * bar
    assert actual == expected
    assert corrected_qx != raw_qx and corrected_qy != raw_qy
    assert not math.isclose(actual, raw_result, rel_tol=0.0, abs_tol=1.0e-15)


@pytest.mark.parametrize("invmat1", (1, 3))
@pytest.mark.parametrize("mapping_name", ("legacy", "physical"))
def test_full_patch_c_reference_dot_and_finite_difference(c6a, invmat1, mapping_name):
    cmapping, reference_mapping = mappings(c6a)[mapping_name]
    primary, rho_values, q_values = case(invmat1)
    rng = random.Random(6100 + invmat1 + cmapping.mode)
    dprimary = [rng.uniform(-0.02, 0.02) * value for value in primary]
    drho = [rng.uniform(-0.02, 0.02) * value for value in rho_values]
    dq = [rng.uniform(-0.02, 0.02) * value for value in q_values]
    bars = [0.7, -1.1, 2.3, -0.4, 8.0e5]
    actual_forward = c_forward(c6a, invmat1, cmapping, primary, rho_values, q_values)
    expected_forward = forward(invmat1, reference_mapping, primary, rho_values, q_values)
    actual_bars = c_vjp(c6a, invmat1, cmapping, primary, rho_values, q_values, bars)
    expected_bars = vjp(invmat1, reference_mapping, primary, rho_values, q_values, bars)
    reference_error = max(
        [relative(a, b) for a, b in zip(actual_forward, expected_forward)]
        + [relative(a, b) for group_a, group_b in zip(actual_bars, expected_bars)
           for a, b in zip(group_a, group_b)]
    )
    tangent = jvp(invmat1, reference_mapping, primary, rho_values, q_values,
                  dprimary, drho, dq)
    lhs = dot(tangent, bars)
    rhs = dot(dprimary, actual_bars[0]) + dot(drho, actual_bars[1]) + dot(dq, actual_bars[2])
    dot_error = relative(lhs, rhs)
    epsilon = 1.0e-5
    plus = forward(
        invmat1, reference_mapping,
        [a + epsilon * b for a, b in zip(primary, dprimary)],
        [a + epsilon * b for a, b in zip(rho_values, drho)],
        [a + epsilon * b for a, b in zip(q_values, dq)],
    )
    minus = forward(
        invmat1, reference_mapping,
        [a - epsilon * b for a, b in zip(primary, dprimary)],
        [a - epsilon * b for a, b in zip(rho_values, drho)],
        [a - epsilon * b for a, b in zip(q_values, dq)],
    )
    fd = dot([(a - b) / (2.0 * epsilon) for a, b in zip(plus, minus)], bars)
    fd_error = relative(rhs, fd)
    record = {"invmat1": invmat1, "mapping": mapping_name,
              "dot_error": dot_error, "reference_error": reference_error,
              "fd_error": fd_error, "lhs": lhs, "rhs": rhs, "fd": fd}
    print("M63C6A_PATCH " + json.dumps(record, sort_keys=True))
    assert reference_error <= C6A_REFERENCE_RELATIVE_MAX
    assert dot_error <= C6A_DOUBLE_DOT_RELATIVE_MAX
    assert fd_error <= C6A_FD_RELATIVE_MAX


def test_invmat1_1_density_contains_opposing_mu_and_rhoi_paths(c6a):
    cmapping, reference_mapping = mappings(c6a)["physical"]
    primary, rho_values, q_values = case(1)
    bars = [1.0, 0.8, 0.0, 0.0, 1.0e10]
    actual = c_vjp(c6a, 1, cmapping, primary, rho_values, q_values, bars)[1][0]
    mu_only = vjp(1, reference_mapping, primary, rho_values, q_values,
                   [*bars[:4], 0.0])[1][0]
    rhoi_only = vjp(1, reference_mapping, primary, rho_values, q_values,
                    [0.0, 0.0, 0.0, 0.0, bars[4]])[1][0]
    assert mu_only * rhoi_only < 0.0
    assert math.isclose(actual, mu_only + rhoi_only, rel_tol=1.0e-14)
    print("M63C6A_DENSITY_DUAL " + json.dumps(
        {"combined": actual, "mu_path": mu_only, "rhoi_path": rhoi_only},
        sort_keys=True
    ))


def test_scope_contract_and_inactive_path(repository_root: Path):
    header = (repository_root / "include/fd.h").read_text()
    source = (repository_root / "src/SH/visco_sh_material_vjp.c").read_text()
    assert "visco_sh_material_patch_vjp" in header
    assert "invmat1 != 1 && invmat1 != 3" in source
    assert "DTINV" not in source and "MPI_" not in source
    for path in ("src/SH/sh_visc.c", "src/SH/FWI_SH.c", "src/SH/FWI_SH_visc.c"):
        assert "visco_sh_material_patch_vjp" not in (repository_root / path).read_text()
