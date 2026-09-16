from __future__ import annotations

import ctypes
import json
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.utilities.m63c_acceptance import (
    local_gsls_coefficients,
    local_gsls_tangent,
    local_gsls_transpose,
    relative_agreement,
)


# Frozen before the first compiled-production comparison.  Both the C helper
# and the independent reference evaluate in binary64; this allows ordinary
# operation-order roundoff without admitting a constitutive discrepancy.
C_DOUBLE_REFERENCE_RELATIVE_MAX = 5.0e-13
COEFFICIENT_FD_RELATIVE_MAX = 2.0e-8


def _array(values):
    return (ctypes.c_double * len(values))(*values)


def _inner(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _normalized_dot_residual(lhs, rhs, tangent, adjoint):
    scale = math.sqrt(_inner(tangent, tangent) * _inner(adjoint, adjoint))
    return abs(lhs - rhs) / max(scale, 1.0e-300)


def _reference_sum(frequencies):
    theta = tuple(1.0 / (2.0 * math.pi * value) for value in frequencies)
    omega_reference = 2.0 * math.pi * frequencies[0]
    return math.fsum(
        (omega_reference * value) ** 2
        / (1.0 + (omega_reference * value) ** 2)
        for value in theta
    )


def _eta_b_c(dt, frequencies):
    eta = tuple(dt * 2.0 * math.pi * value for value in frequencies)
    b_values = tuple(1.0 / (1.0 + 0.5 * value) for value in eta)
    c_values = tuple(1.0 - 0.5 * value for value in eta)
    return eta, b_values, c_values


@pytest.fixture(scope="module")
def compiled_local_gsls_vjp(tmp_path_factory, repository_root):
    compiler = shutil.which("mpicc")
    assert compiler is not None, "mpicc is required for the production C VJP test"
    output = tmp_path_factory.mktemp("m63c_local_gsls") / "libm63c_gsls_vjp.so"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fPIC",
        "-shared",
        "-I",
        str(repository_root / "include"),
        str(repository_root / "src/SH/visco_sh_gsls_vjp.c"),
        "-o",
        str(output),
        "-lm",
    ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    library = ctypes.CDLL(str(output))
    double_pointer = ctypes.POINTER(ctypes.c_double)
    library.visco_sh_gsls_local_derivatives.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
    ]
    library.visco_sh_gsls_local_derivatives.restype = ctypes.c_int
    library.visco_sh_gsls_local_vjp.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        double_pointer,
        ctypes.c_double,
        double_pointer,
        double_pointer,
        ctypes.c_double,
        ctypes.c_double,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
        double_pointer,
    ]
    library.visco_sh_gsls_local_vjp.restype = ctypes.c_int
    return library


def _production_derivatives(library, *, modulus, tau, dt, frequencies):
    mechanisms = len(frequencies)
    eta, b_values, _ = _eta_b_c(dt, frequencies)
    f_tau = ctypes.c_double()
    f_modulus = ctypes.c_double()
    c_tau = _array([0.0] * mechanisms)
    c_modulus = _array([0.0] * mechanisms)
    returncode = library.visco_sh_gsls_local_derivatives(
        mechanisms,
        dt,
        modulus,
        tau,
        _reference_sum(frequencies),
        _array(eta),
        _array(b_values),
        ctypes.byref(f_tau),
        ctypes.byref(f_modulus),
        c_tau,
        c_modulus,
    )
    assert returncode == 0
    return (
        f_tau.value,
        f_modulus.value,
        tuple(c_tau),
        tuple(c_modulus),
    )


def _production_vjp(
    library,
    *,
    coefficients,
    derivatives,
    dt,
    strain,
    output_adjoint,
    initial,
):
    mechanisms = len(coefficients.recurrence)
    bar_s_prev = ctypes.c_double(initial[0])
    bar_r_prev = _array(initial[1])
    bar_strain = ctypes.c_double(initial[2])
    g_tau = ctypes.c_double(initial[3])
    g_modulus = ctypes.c_double(initial[4])
    returncode = library.visco_sh_gsls_local_vjp(
        mechanisms,
        dt,
        strain,
        output_adjoint[0],
        _array(output_adjoint[1]),
        coefficients.stress,
        _array(coefficients.recurrence),
        _array(coefficients.coupling),
        derivatives[0],
        derivatives[1],
        _array(derivatives[2]),
        _array(derivatives[3]),
        ctypes.byref(bar_s_prev),
        bar_r_prev,
        ctypes.byref(bar_strain),
        ctypes.byref(g_tau),
        ctypes.byref(g_modulus),
    )
    assert returncode == 0
    result = (
        bar_s_prev.value,
        tuple(bar_r_prev),
        bar_strain.value,
        g_tau.value,
        g_modulus.value,
    )
    assert all(
        math.isfinite(value)
        for value in (
            result[0],
            *result[1],
            result[2],
            result[3],
            result[4],
        )
    )
    return result


@pytest.mark.parametrize("mechanisms", (1, 4))
def test_compiled_local_vjp_matches_reference_and_closes_dot_product(
    compiled_local_gsls_vjp, mechanisms
):
    randomizer = random.Random(631100 + mechanisms)
    frequencies = tuple(5.0 * 2.1**index for index in range(mechanisms))
    maxima = {
        "dot": 0.0,
        "bar_s_abs": 0.0,
        "bar_s_rel": 0.0,
        "bar_r_abs": 0.0,
        "bar_r_rel": 0.0,
        "bar_e_abs": 0.0,
        "bar_e_rel": 0.0,
        "g_tau_abs": 0.0,
        "g_tau_rel": 0.0,
        "g_M_abs": 0.0,
        "g_M_rel": 0.0,
    }
    for tau in (0.01, 0.035, 0.10):  # legacy Q=200, intermediate, Q=20
        for _ in range(4):
            dt = 0.0004
            modulus = randomizer.uniform(3.0e9, 8.0e9)
            strain = randomizer.uniform(-0.6, 0.6)
            coefficients = local_gsls_coefficients(
                unrelaxed_shear_modulus=modulus,
                tau=tau,
                dt=dt,
                relaxation_frequencies_hz=frequencies,
            )
            derivatives = _production_derivatives(
                compiled_local_gsls_vjp,
                modulus=modulus,
                tau=tau,
                dt=dt,
                frequencies=frequencies,
            )
            output_adjoint = (
                randomizer.uniform(-0.9, 0.9),
                tuple(randomizer.uniform(-0.9, 0.9) for _ in range(mechanisms)),
            )
            initial = (
                randomizer.uniform(-0.2, 0.2),
                tuple(randomizer.uniform(-0.2, 0.2) for _ in range(mechanisms)),
                randomizer.uniform(-0.2, 0.2),
                randomizer.uniform(-0.2, 0.2),
                randomizer.uniform(-0.2, 0.2),
            )
            production = _production_vjp(
                compiled_local_gsls_vjp,
                coefficients=coefficients,
                derivatives=derivatives,
                dt=dt,
                strain=strain,
                output_adjoint=output_adjoint,
                initial=initial,
            )
            reference_increment = local_gsls_transpose(
                stress_next_adjoint=output_adjoint[0],
                memory_next_adjoint=output_adjoint[1],
                strain=strain,
                coefficients=coefficients,
            )
            reference = (
                initial[0] + reference_increment[0],
                tuple(
                    left + right
                    for left, right in zip(initial[1], reference_increment[1])
                ),
                initial[2] + reference_increment[2],
                initial[3] + reference_increment[3],
                initial[4] + reference_increment[4],
            )
            maxima["bar_s_abs"] = max(
                maxima["bar_s_abs"], abs(production[0] - reference[0])
            )
            maxima["bar_s_rel"] = max(
                maxima["bar_s_rel"], relative_agreement(production[0], reference[0])
            )
            maxima["bar_r_abs"] = max(
                maxima["bar_r_abs"],
                *(abs(left - right) for left, right in zip(production[1], reference[1])),
            )
            maxima["bar_r_rel"] = max(
                maxima["bar_r_rel"],
                *(relative_agreement(left, right) for left, right in zip(production[1], reference[1])),
            )
            maxima["bar_e_abs"] = max(
                maxima["bar_e_abs"], abs(production[2] - reference[2])
            )
            maxima["bar_e_rel"] = max(
                maxima["bar_e_rel"], relative_agreement(production[2], reference[2])
            )
            maxima["g_tau_abs"] = max(
                maxima["g_tau_abs"], abs(production[3] - reference[3])
            )
            maxima["g_tau_rel"] = max(
                maxima["g_tau_rel"], relative_agreement(production[3], reference[3])
            )
            maxima["g_M_abs"] = max(
                maxima["g_M_abs"], abs(production[4] - reference[4])
            )
            maxima["g_M_rel"] = max(
                maxima["g_M_rel"], relative_agreement(production[4], reference[4])
            )

            tangent_input = (
                randomizer.uniform(-0.7, 0.7),
                tuple(randomizer.uniform(-0.7, 0.7) for _ in range(mechanisms)),
                randomizer.uniform(-0.7, 0.7),
                randomizer.uniform(-0.07, 0.07),
                randomizer.uniform(-2.0e8, 2.0e8),
            )
            tangent = local_gsls_tangent(
                stress_previous_tangent=tangent_input[0],
                memory_previous_tangent=tangent_input[1],
                strain_tangent=tangent_input[2],
                tau_tangent=tangent_input[3],
                modulus_tangent=tangent_input[4],
                strain=strain,
                coefficients=coefficients,
            )
            zero_initial = (0.0, (0.0,) * mechanisms, 0.0, 0.0, 0.0)
            transpose = _production_vjp(
                compiled_local_gsls_vjp,
                coefficients=coefficients,
                derivatives=derivatives,
                dt=dt,
                strain=strain,
                output_adjoint=output_adjoint,
                initial=zero_initial,
            )
            lhs = tangent[0] * output_adjoint[0] + _inner(
                tangent[1], output_adjoint[1]
            )
            rhs = (
                tangent_input[0] * transpose[0]
                + _inner(tangent_input[1], transpose[1])
                + tangent_input[2] * transpose[2]
                + tangent_input[3] * transpose[3]
                + tangent_input[4] * transpose[4]
            )
            tangent_flat = (tangent[0], *tangent[1])
            adjoint_flat = (output_adjoint[0], *output_adjoint[1])
            maxima["dot"] = max(
                maxima["dot"],
                _normalized_dot_residual(lhs, rhs, tangent_flat, adjoint_flat),
            )

    assert maxima["dot"] <= C_DOUBLE_REFERENCE_RELATIVE_MAX
    assert maxima["bar_s_rel"] <= C_DOUBLE_REFERENCE_RELATIVE_MAX
    assert maxima["bar_r_rel"] <= C_DOUBLE_REFERENCE_RELATIVE_MAX
    assert maxima["bar_e_rel"] <= C_DOUBLE_REFERENCE_RELATIVE_MAX
    assert maxima["g_tau_rel"] <= C_DOUBLE_REFERENCE_RELATIVE_MAX
    assert maxima["g_M_rel"] <= C_DOUBLE_REFERENCE_RELATIVE_MAX
    print("M63C1_VJP " + json.dumps({"L": mechanisms, **maxima}, sort_keys=True))


@pytest.mark.parametrize("mechanisms", (1, 4))
@pytest.mark.parametrize("tau", (0.01, 0.10))
def test_compiled_parameter_derivatives_match_five_point_fd(
    compiled_local_gsls_vjp, mechanisms, tau
):
    frequencies = tuple(6.0 * 1.9**index for index in range(mechanisms))
    dt = 0.00035
    modulus = 6.2e9
    production = _production_derivatives(
        compiled_local_gsls_vjp,
        modulus=modulus,
        tau=tau,
        dt=dt,
        frequencies=frequencies,
    )

    def coefficients(tau_value, modulus_value):
        return local_gsls_coefficients(
            unrelaxed_shear_modulus=modulus_value,
            tau=tau_value,
            dt=dt,
            relaxation_frequencies_hz=frequencies,
        )

    tau_step = 1.0e-4 * tau
    modulus_step = 1.0e-4 * modulus
    tau_rows = [
        coefficients(tau + factor * tau_step, modulus) for factor in (-2, -1, 1, 2)
    ]
    modulus_rows = [
        coefficients(tau, modulus + factor * modulus_step) for factor in (-2, -1, 1, 2)
    ]

    def five_point(rows, attribute, step):
        values = [getattr(row, attribute) for row in rows]
        return (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (
            12.0 * step
        )

    f_tau_fd = five_point(tau_rows, "stress", tau_step)
    f_modulus_fd = five_point(modulus_rows, "stress", modulus_step)
    c_tau_fd = tuple(
        (
            tau_rows[0].coupling[index]
            - 8.0 * tau_rows[1].coupling[index]
            + 8.0 * tau_rows[2].coupling[index]
            - tau_rows[3].coupling[index]
        )
        / (12.0 * tau_step)
        for index in range(mechanisms)
    )
    c_modulus_fd = tuple(
        (
            modulus_rows[0].coupling[index]
            - 8.0 * modulus_rows[1].coupling[index]
            + 8.0 * modulus_rows[2].coupling[index]
            - modulus_rows[3].coupling[index]
        )
        / (12.0 * modulus_step)
        for index in range(mechanisms)
    )
    errors = {
        "F_tau": relative_agreement(production[0], f_tau_fd),
        "C_tau": max(
            relative_agreement(left, right)
            for left, right in zip(production[2], c_tau_fd)
        ),
        "F_M": relative_agreement(production[1], f_modulus_fd),
        "C_M": max(
            relative_agreement(left, right)
            for left, right in zip(production[3], c_modulus_fd)
        ),
    }
    assert max(errors.values()) <= COEFFICIENT_FD_RELATIVE_MAX
    print(
        "M63C1_FD "
        + json.dumps({"L": mechanisms, "tau": tau, **errors}, sort_keys=True)
    )
