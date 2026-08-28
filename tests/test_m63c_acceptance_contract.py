from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import pytest

from tests.utilities.m63c_acceptance import (
    DIRECTIONAL_GRADIENT_RELATIVE_MAX,
    FORWARD_OPERATOR_ORDER,
    PRODUCTION_DOT_CASES,
    PRODUCTION_DOT_RELATIVE_MAX,
    REVERSE_OPERATOR_ORDER,
    TAU_VJP_DECOMPOSITIONS,
    ZERO_STEP_OBJECTIVE_RELATIVE_MAX,
    dtinv_quadrature,
    local_gsls_coefficients,
    local_gsls_forward,
    local_gsls_tangent,
    local_gsls_transpose,
    q_gradient_from_native_tau,
    q_to_tau,
    relative_agreement,
)
from tests.utilities.visco_sh_fwi_attenuation import (
    av_tau,
    av_tau_partitioned,
    av_tau_partitioned_vjp,
    av_tau_vjp,
    dense_linearization,
    global_visco_sh_reference,
    transpose_matvec,
)


BASELINE_SHA = "8c1bfc9c5a5c9f39396b9be5030464f683d3ab5d"
M63B_VALIDATION_SHA = "15a8b21077f03e902d2edc735941442b384935431b749540401a0d018e5e0552"
M63B_INSTRUMENTATION_SHA = "84a821686303b9b8166ec884b381348900e7158f074dc57259d12142a0d991cd"


class KnownM63cProductionDotDefect(AssertionError):
    """Raised only for frozen M6.3b baseline production dot-product evidence."""


class KnownM63cProductionGradientDisconnected(AssertionError):
    """Raised only for frozen M6.3b baseline production gradient evidence."""


def _inner(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _flatten(field):
    return [value for row in field for value in row]


def _dot_closure_error(left_values, right_values, left_transpose, right_transpose):
    lhs = _inner(left_values, right_values)
    rhs = _inner(left_transpose, right_transpose)
    scale = math.sqrt(
        _inner(left_values, left_values) * _inner(right_values, right_values)
    )
    return abs(lhs - rhs) / max(scale, 1.0e-300)


def _frozen_report(repository_root: Path):
    path = repository_root / "tests/m6.3b_visco_sh_fwi_attenuation_validation.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == M63B_VALIDATION_SHA
    instrumentation = (
        repository_root
        / "tests/utilities/m63b_production_adjoint_instrumentation.py"
    )
    assert hashlib.sha256(instrumentation.read_bytes()).hexdigest() == M63B_INSTRUMENTATION_SHA
    return json.loads(path.read_text(encoding="utf-8"))


def test_m63c_machine_readable_contract_predeclares_complete_acceptance(repository_root):
    contract = json.loads(
        (repository_root / "tests/m6.3c_acceptance_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["baseline_commit"] == BASELINE_SHA
    assert tuple(contract["operator_order"]["forward"]) == FORWARD_OPERATOR_ORDER
    assert tuple(contract["operator_order"]["reverse"]) == REVERSE_OPERATOR_ORDER
    assert tuple(
        (bool(row["free_surface"]), row["nprocx"], row["nprocy"])
        for row in contract["global_production_dot_product"]["cases"]
    ) == PRODUCTION_DOT_CASES
    assert contract["global_production_dot_product"]["relative_residual_max"] == PRODUCTION_DOT_RELATIVE_MAX
    assert contract["directional_gradient"]["dtinv_1_relative_max"] == DIRECTIONAL_GRADIENT_RELATIVE_MAX
    assert contract["physics_consistency"]["relative_objective_max"] == ZERO_STEP_OBJECTIVE_RELATIVE_MAX
    assert tuple(map(tuple, contract["tau_vjp"]["decompositions"])) == TAU_VJP_DECOMPOSITIONS
    assert contract["tau_vjp"]["requires_corner_case"] is True
    assert contract["temporal_quadrature"]["dtinv_greater_than_1_exactness_requires_separate_proof"] is True
    assert set(contract["operator_order"]["not_assumed_self_adjoint"]) == {
        "mpi_velocity_exchange",
        "free_surface_velocity_completion",
        "free_surface_stress_completion",
        "mpi_stress_exchange",
    }


@pytest.mark.parametrize("mechanisms", (1, 4))
def test_local_gsls_exact_transpose_and_tau_modulus_derivatives(mechanisms):
    frequencies = tuple(4.0 * 2.3**index for index in range(mechanisms))
    modulus = 5.4e9
    tau = 0.047
    dt = 0.0004
    coefficients = local_gsls_coefficients(
        unrelaxed_shear_modulus=modulus,
        tau=tau,
        dt=dt,
        relaxation_frequencies_hz=frequencies,
    )
    strain = -0.23
    tangent_input = (
        0.17,
        tuple(-0.03 * (index + 1) for index in range(mechanisms)),
        0.31,
        -0.08,
        2.1e8,
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
    output_adjoint = (
        0.63,
        tuple(0.09 * (-1) ** index for index in range(mechanisms)),
    )
    transpose = local_gsls_transpose(
        stress_next_adjoint=output_adjoint[0],
        memory_next_adjoint=output_adjoint[1],
        strain=strain,
        coefficients=coefficients,
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
    assert relative_agreement(lhs, rhs) < 3.0e-15

    for variable, value, stress_derivative, coupling_derivative in (
        (
            "tau",
            tau,
            coefficients.stress_tau_derivative,
            coefficients.coupling_tau_derivative,
        ),
        (
            "unrelaxed_shear_modulus",
            modulus,
            coefficients.stress_modulus_derivative,
            coefficients.coupling_modulus_derivative,
        ),
    ):
        step = 2.0e-6 if variable == "tau" else 2.0e4
        arguments = {
            "unrelaxed_shear_modulus": modulus,
            "tau": tau,
            "dt": dt,
            "relaxation_frequencies_hz": frequencies,
        }
        plus = local_gsls_coefficients(**(arguments | {variable: value + step}))
        minus = local_gsls_coefficients(**(arguments | {variable: value - step}))
        stress_fd = (plus.stress - minus.stress) / (2.0 * step)
        coupling_fd = tuple(
            (left - right) / (2.0 * step)
            for left, right in zip(plus.coupling, minus.coupling)
        )
        assert relative_agreement(stress_fd, stress_derivative) < 2.0e-9
        assert max(
            relative_agreement(left, right)
            for left, right in zip(coupling_fd, coupling_derivative)
        ) < 2.0e-9


def test_local_gsls_tangent_matches_full_state_finite_difference():
    mechanisms = 3
    frequencies = (4.0, 11.0, 31.0)
    base = [0.7, 0.03, -0.02, 0.05, -0.19, 0.046, 5.8e9]
    direction = [-0.11, 0.04, 0.02, -0.03, 0.27, -0.06, 1.7e8]

    def forward(values):
        coefficients = local_gsls_coefficients(
            unrelaxed_shear_modulus=values[-1],
            tau=values[-2],
            dt=0.0005,
            relaxation_frequencies_hz=frequencies,
        )
        stress, memory = local_gsls_forward(
            values[0], values[1 : 1 + mechanisms], values[-3], coefficients
        )
        return [stress, *memory]

    coefficients = local_gsls_coefficients(
        unrelaxed_shear_modulus=base[-1],
        tau=base[-2],
        dt=0.0005,
        relaxation_frequencies_hz=frequencies,
    )
    analytic_stress, analytic_memory = local_gsls_tangent(
        stress_previous_tangent=direction[0],
        memory_previous_tangent=direction[1 : 1 + mechanisms],
        strain_tangent=direction[-3],
        tau_tangent=direction[-2],
        modulus_tangent=direction[-1],
        strain=base[-3],
        coefficients=coefficients,
    )
    epsilon = 2.0e-6
    plus = forward([value + epsilon * delta for value, delta in zip(base, direction)])
    minus = forward([value - epsilon * delta for value, delta in zip(base, direction)])
    finite_difference = [
        (left - right) / (2.0 * epsilon) for left, right in zip(plus, minus)
    ]
    analytic = [analytic_stress, *analytic_memory]
    assert math.sqrt(
        _inner(
            [left - right for left, right in zip(analytic, finite_difference)],
            [left - right for left, right in zip(analytic, finite_difference)],
        )
    ) / math.sqrt(_inner(finite_difference, finite_difference)) < 3.0e-9


@pytest.mark.parametrize("mode", ("legacy", "physical"))
def test_q_chain_rule_order_and_directional_product(mode):
    q_field = [
        [38.0 + 2.0 * i + 1.5 * j for i in range(6)] for j in range(6)
    ]
    native_gradient = [
        [0.02 * (i + 1) - 0.015 * j for i in range(5)] for j in range(5)
    ]
    direction = [
        [0.3 * math.sin(0.4 * (i + 1) * (j + 2)) for i in range(6)]
        for j in range(6)
    ]
    parameters = {"mode": mode}
    if mode == "physical":
        parameters |= {"a": 0.37, "b": -0.11}
    q_gradient = q_gradient_from_native_tau(
        native_gradient, q_field, **parameters
    )
    predicted = _inner(_flatten(q_gradient), _flatten(direction))

    epsilon = 1.0e-3

    def objective(scale):
        tau_field = [
            [
                q_to_tau(
                    q_field[j][i] + scale * direction[j][i], **parameters
                )[0]
                for i in range(6)
            ]
            for j in range(6)
        ]
        return _inner(_flatten(av_tau(tau_field)), _flatten(native_gradient))

    finite_difference = (
        -objective(2.0 * epsilon)
        + 8.0 * objective(epsilon)
        - 8.0 * objective(-epsilon)
        + objective(-2.0 * epsilon)
    ) / (12.0 * epsilon)
    assert relative_agreement(predicted, finite_difference) < 2.0e-9


@pytest.mark.parametrize("nprocx,nprocy", TAU_VJP_DECOMPOSITIONS)
def test_staggered_tau_vjp_scatter_and_mpi_corner(nprocx, nprocy):
    randomizer = random.Random(6300 + 10 * nprocx + nprocy)
    field = [[randomizer.uniform(0.02, 0.08) for _ in range(8)] for _ in range(8)]
    direction = [[randomizer.uniform(-1.0, 1.0) for _ in range(8)] for _ in range(8)]
    sensitivity = [[randomizer.uniform(-1.0, 1.0) for _ in range(7)] for _ in range(7)]
    partitioned, seams = av_tau_partitioned(
        field, nprocx=nprocx, nprocy=nprocy
    )
    transpose = av_tau_partitioned_vjp(
        (8, 8), sensitivity, nprocx=nprocx, nprocy=nprocy
    )
    assert _flatten(partitioned) == pytest.approx(_flatten(av_tau(field)), abs=0.0)
    assert _flatten(transpose) == pytest.approx(
        _flatten(av_tau_vjp((8, 8), sensitivity)), abs=0.0
    )
    assert _dot_closure_error(
        _flatten(av_tau(direction)),
        _flatten(sensitivity),
        _flatten(direction),
        _flatten(transpose),
    ) < 2.0e-16
    if (nprocx, nprocy) == (2, 2):
        assert (3, 3) in seams
        assert transpose[4][4] == pytest.approx(
            0.25
            * (
                sensitivity[3][3]
                + sensitivity[3][4]
                + sensitivity[4][3]
                + sensitivity[4][4]
            ),
            rel=0.0,
            abs=2.0e-16,
        )


@pytest.mark.parametrize(
    "free_surface,nprocx,nprocy",
    (*PRODUCTION_DOT_CASES, (False, 2, 2)),
)
def test_independent_global_reference_closes_including_2x2(
    free_surface, nprocx, nprocy
):
    layout, operator = global_visco_sh_reference(
        free_surface=free_surface,
        nprocx=nprocx,
        nprocy=nprocy,
    )
    matrix = dense_linearization(operator, layout.input_count)
    randomizer = random.Random(6310 + 100 * int(free_surface) + 10 * nprocx + nprocy)
    tangent_input = [
        randomizer.uniform(-1.0, 1.0) for _ in range(layout.input_count)
    ]
    output_adjoint = [
        randomizer.uniform(-1.0, 1.0) for _ in range(layout.output_count)
    ]
    tangent = [
        _inner(row, tangent_input) for row in matrix
    ]
    transpose = transpose_matvec(matrix, output_adjoint)
    assert relative_agreement(
        _inner(tangent, output_adjoint), _inner(tangent_input, transpose)
    ) < 2.0e-14


def test_dt_dtinv_quadrature_convention_and_exactness_scope():
    dt = 0.0005
    constant = [2.0] * 24
    for dtinv in (1, 2, 3, 4):
        assert dtinv_quadrature(constant, dt=dt, dtinv=dtinv) == pytest.approx(
            len(constant) * dt * 2.0, rel=0.0, abs=2.0e-18
        )

    varying = [math.sin(0.23 * index) + 0.03 * index for index in range(24)]
    fine = dtinv_quadrature(varying, dt=dt, dtinv=1)
    assert all(
        relative_agreement(dtinv_quadrature(varying, dt=dt, dtinv=value), fine)
        > 1.0e-3
        for value in (2, 3, 4)
    )


def test_directional_and_zero_step_acceptance_thresholds_are_unfitted():
    finite_difference = 0.081
    predicted = finite_difference * (1.0 + 0.004)
    assert relative_agreement(predicted, finite_difference) <= DIRECTIONAL_GRADIENT_RELATIVE_MAX
    assert relative_agreement(
        finite_difference * (1.0 + 0.006), finite_difference
    ) > DIRECTIONAL_GRADIENT_RELATIVE_MAX

    base_objective = 0.054
    zero_step_trial = base_objective * (1.0 + 5.0e-13)
    assert relative_agreement(base_objective, zero_step_trial) <= ZERO_STEP_OBJECTIVE_RELATIVE_MAX
    assert relative_agreement(
        base_objective, base_objective * (1.0 + 2.0e-12)
    ) > ZERO_STEP_OBJECTIVE_RELATIVE_MAX


@pytest.mark.xfail(
    strict=True,
    raises=KnownM63cProductionDotDefect,
    reason="M63C-FROZEN-M63B-PRODUCTION-DOT: nominal adjoint misses GREEN target",
)
def test_frozen_m63b_production_dot_is_red_against_m63c_target(repository_root):
    report = _frozen_report(repository_root)
    rows = report["production_adjoint_dot_product"]["cases"]
    assert set(rows) == {"fs0_1x1", "fs0_1x2", "fs0_2x1", "fs1_1x1"}
    assert all(row["returncode"] == 0 for row in rows.values())
    assert all(row["timed_out"] is False for row in rows.values())
    failures = {
        label: row["relative_residual"]
        for label, row in rows.items()
        if row["relative_residual"] > PRODUCTION_DOT_RELATIVE_MAX
    }
    if failures:
        raise KnownM63cProductionDotDefect(
            f"relative residuals exceed {PRODUCTION_DOT_RELATIVE_MAX}: {failures}"
        )


@pytest.mark.xfail(
    strict=True,
    raises=KnownM63cProductionGradientDisconnected,
    reason="M63C-FROZEN-M63B-PRODUCTION-GRADIENT: Q/tau sensitivities are not wired",
)
def test_frozen_m63b_production_directional_gradients_are_disconnected(repository_root):
    report = _frozen_report(repository_root)
    q_fd = report["q_directional_derivatives"]["fs0_1x1"]["five_point"]
    tau_fd = report["tau_directional_derivative"]["five_point"]
    assert abs(q_fd) > 1.0e-8
    assert abs(tau_fd) > 1.0e-8
    if "production_q_directional_product" not in report or "production_tau_directional_product" not in report:
        raise KnownM63cProductionGradientDisconnected(
            f"nonzero FD derivatives q={q_fd:.17g}, tau={tau_fd:.17g} have no production products"
        )
