from __future__ import annotations

import json
import math
import random
from array import array

import pytest

from tests.cases.visco_sh_fwi_attenuation import (
    ViscoSHFWIAttenuationConfig,
    generate_case,
)
from tests.utilities.qstd_reference import target_q_to_tau
from tests.utilities.visco_sh_fwi_attenuation import (
    QTauMapping,
    av_tau,
    av_tau_partitioned,
    av_tau_partitioned_vjp,
    av_tau_vjp,
    dense_linearization,
    global_visco_sh_reference,
    gsls_coefficients,
    gsls_forward,
    gsls_tangent,
    gsls_transpose,
    physical_q_mapping,
    q_to_tau_and_derivative,
    relative_dot_error,
    transpose_matvec,
)


def _inner(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _flatten(field):
    return [value for row in field for value in row]


def _read_float_grid(path):
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, path.stat().st_size // 4)
    return list(values)


@pytest.mark.parametrize("mechanisms", (1, 4))
def test_local_gsls_tangent_transpose_and_coefficient_derivatives(mechanisms):
    frequencies = tuple(3.0 * 2.5**index for index in range(mechanisms))
    tau = 0.041
    arguments = {
        "shear_modulus": 7.2e9,
        "tau": tau,
        "dt": 0.0004,
        "relaxation_frequencies_hz": frequencies,
    }
    coefficients = gsls_coefficients(**arguments)
    stress_previous = -1.4
    memory_previous = tuple(0.2 * (index + 1) for index in range(mechanisms))
    strain_rate = 0.37
    tangent_input = (0.13, tuple(-0.04 * (index + 1) for index in range(mechanisms)), 0.29, -0.17)
    tangent_output = gsls_tangent(
        stress_previous_tangent=tangent_input[0],
        memory_previous_tangent=tangent_input[1],
        strain_rate_tangent=tangent_input[2],
        tau_tangent=tangent_input[3],
        stress_previous=stress_previous,
        memory_previous=memory_previous,
        strain_rate=strain_rate,
        coefficients=coefficients,
    )
    output_adjoint = (0.61, tuple(-0.11 * (index + 1) for index in range(mechanisms)))
    transpose = gsls_transpose(
        stress_next_adjoint=output_adjoint[0],
        memory_next_adjoint=output_adjoint[1],
        strain_rate=strain_rate,
        coefficients=coefficients,
    )
    lhs = tangent_output[0] * output_adjoint[0] + _inner(tangent_output[1], output_adjoint[1])
    rhs = (
        tangent_input[0] * transpose[0]
        + _inner(tangent_input[1], transpose[1])
        + tangent_input[2] * transpose[2]
        + tangent_input[3] * transpose[3]
    )
    dot_error = relative_dot_error(lhs, rhs)
    assert dot_error < 2.0e-15

    steps = (1.0e-5, 5.0e-6, 2.5e-6)
    derivative_rows = []
    for step in steps:
        plus = gsls_coefficients(**(arguments | {"tau": tau + step}))
        minus = gsls_coefficients(**(arguments | {"tau": tau - step}))
        stress_fd = (plus.stress - minus.stress) / (2.0 * step)
        memory_fd = tuple(
            (a - b) / (2.0 * step) for a, b in zip(plus.memory, minus.memory)
        )
        stress_error = abs(stress_fd - coefficients.stress_derivative) / abs(stress_fd)
        memory_error = max(
            abs(actual - expected) / abs(actual)
            for actual, expected in zip(memory_fd, coefficients.memory_derivative)
        )
        derivative_rows.append(
            {"step": step, "F_relative_error": stress_error, "C_max_relative_error": memory_error}
        )
    assert derivative_rows[-1]["F_relative_error"] < 2.0e-10
    assert derivative_rows[-1]["C_max_relative_error"] < 2.0e-10
    print(json.dumps({"mechanisms": mechanisms, "dot_error": dot_error, "coefficient_fd": derivative_rows}))


@pytest.mark.parametrize("mechanisms", (1, 4))
def test_local_gsls_full_state_jvp_matches_centered_finite_difference(mechanisms):
    frequencies = tuple(4.0 * 2.0**index for index in range(mechanisms))
    base = [0.8] + [0.03 * (index + 1) for index in range(mechanisms)] + [-0.21, 0.052]
    direction = [-0.17] + [0.02 * (index + 1) for index in range(mechanisms)] + [0.31, -0.09]

    def forward(values):
        stress = values[0]
        memory = values[1:1 + mechanisms]
        strain = values[-2]
        tau = values[-1]
        coefficients = gsls_coefficients(
            shear_modulus=5.1e9,
            tau=tau,
            dt=0.0005,
            relaxation_frequencies_hz=frequencies,
        )
        next_stress, next_memory = gsls_forward(stress, memory, strain, coefficients)
        return [next_stress, *next_memory]

    coefficients = gsls_coefficients(
        shear_modulus=5.1e9,
        tau=base[-1],
        dt=0.0005,
        relaxation_frequencies_hz=frequencies,
    )
    tangent = gsls_tangent(
        stress_previous_tangent=direction[0],
        memory_previous_tangent=direction[1:1 + mechanisms],
        strain_rate_tangent=direction[-2],
        tau_tangent=direction[-1],
        stress_previous=base[0],
        memory_previous=base[1:1 + mechanisms],
        strain_rate=base[-2],
        coefficients=coefficients,
    )
    analytic = [tangent[0], *tangent[1]]
    errors = []
    for epsilon in (1.0e-4, 5.0e-5, 2.5e-5):
        plus = forward([value + epsilon * delta for value, delta in zip(base, direction)])
        minus = forward([value - epsilon * delta for value, delta in zip(base, direction)])
        numerical = [(a - b) / (2.0 * epsilon) for a, b in zip(plus, minus)]
        errors.append(
            math.sqrt(_inner([a - b for a, b in zip(analytic, numerical)], [a - b for a, b in zip(analytic, numerical)]))
            / math.sqrt(_inner(numerical, numerical))
        )
    assert errors[-1] < 2.0e-9
    assert errors[-1] < errors[0]


@pytest.mark.parametrize("mode", ("legacy", "physical"))
@pytest.mark.parametrize("mechanisms", (1, 4))
def test_q_to_tau_chain_rule_matches_centered_finite_difference(mode, mechanisms):
    frequencies = tuple(3.5 * 2.8**index for index in range(mechanisms))
    mapping = (
        QTauMapping(mode="legacy")
        if mode == "legacy"
        else physical_q_mapping(
            relaxation_frequencies_hz=frequencies,
            fmin_hz=3.0,
            fmax_hz=90.0,
            df_hz=3.0,
        )
    )
    q_value = 45.0
    tau, derivative = q_to_tau_and_derivative(q_value, mapping)
    rows = []
    for step in (1.0e-3, 5.0e-4, 2.5e-4):
        plus = q_to_tau_and_derivative(q_value + step, mapping)[0]
        minus = q_to_tau_and_derivative(q_value - step, mapping)[0]
        finite_difference = (plus - minus) / (2.0 * step)
        relative_error = abs(finite_difference - derivative) / abs(finite_difference)
        rows.append({"step": step, "fd": finite_difference, "analytic": derivative, "relative_error": relative_error})
    assert rows[-1]["relative_error"] < 2.0e-9
    if mode == "physical":
        expected = target_q_to_tau(
            target_q=q_value,
            relaxation_frequencies_hz=frequencies,
            fmin_hz=3.0,
            fmax_hz=90.0,
            df_hz=3.0,
        )
        assert tau == pytest.approx(expected, rel=2.0e-15)
    print(json.dumps({"mode": mode, "mechanisms": mechanisms, "tau": tau, "rows": rows}))


@pytest.mark.parametrize(("nprocx", "nprocy"), ((1, 1), (2, 1), (1, 2)))
def test_av_tau_transpose_and_partition_seams(nprocx, nprocy):
    randomizer = random.Random(2026082700 + 10 * nprocx + nprocy)
    field = [[0.02 + 0.05 * randomizer.random() for _ in range(8)] for _ in range(6)]
    direction = [[2.0 * randomizer.random() - 1.0 for _ in range(8)] for _ in range(6)]
    sensitivity = [[2.0 * randomizer.random() - 1.0 for _ in range(7)] for _ in range(5)]
    reference = av_tau(field)
    partitioned, seams = av_tau_partitioned(field, nprocx=nprocx, nprocy=nprocy)
    assert _flatten(partitioned) == pytest.approx(_flatten(reference), rel=0.0, abs=0.0)
    jvp = av_tau(direction)
    transpose = av_tau_partitioned_vjp(
        (6, 8), sensitivity, nprocx=nprocx, nprocy=nprocy
    )
    reference_transpose = av_tau_vjp((6, 8), sensitivity)
    assert _flatten(transpose) == pytest.approx(_flatten(reference_transpose), rel=0.0, abs=0.0)
    lhs = _inner(_flatten(jvp), _flatten(sensitivity))
    rhs = _inner(_flatten(direction), _flatten(transpose))
    dot_error = relative_dot_error(lhs, rhs)
    assert dot_error < 5.0e-15
    if (nprocx, nprocy) != (1, 1):
        assert seams
    print(json.dumps({"decomposition": [nprocx, nprocy], "seam_edges": len(seams), "dot_error": dot_error}))


@pytest.mark.parametrize("free_surface", (False, True))
@pytest.mark.parametrize(("nprocx", "nprocy"), ((1, 1), (2, 1), (1, 2)))
def test_global_viscoelastic_sh_state_operator_dot_product(free_surface, nprocx, nprocy):
    layout, operator = global_visco_sh_reference(
        free_surface=free_surface,
        nprocx=nprocx,
        nprocy=nprocy,
    )
    matrix = dense_linearization(operator, layout.input_count)
    randomizer = random.Random(2026082710 + 100 * int(free_surface) + 10 * nprocx + nprocy)
    state_tangent = [2.0 * randomizer.random() - 1.0 for _ in range(layout.input_count)]
    output_adjoint = [2.0 * randomizer.random() - 1.0 for _ in range(layout.output_count)]
    tangent = [
        math.fsum(coefficient * value for coefficient, value in zip(row, state_tangent))
        for row in matrix
    ]
    transpose = transpose_matvec(matrix, output_adjoint)
    lhs = _inner(tangent, output_adjoint)
    rhs = _inner(state_tangent, transpose)
    dot_error = relative_dot_error(lhs, rhs)
    assert dot_error < 2.0e-14
    assert len(tangent) == layout.output_count
    assert len(transpose) == layout.input_count
    print(
        json.dumps(
            {
                "free_surface": int(free_surface),
                "decomposition": [nprocx, nprocy],
                "input_count": layout.input_count,
                "output_count": layout.output_count,
                "dot_error": dot_error,
                "components": [
                    "velocity_update",
                    "gsls_stress_memory_update",
                    "cpml_state_update",
                    "logical_mpi_exchange",
                    "source_injection",
                    "receiver_sampling_metric",
                    "free_surface_completion" if free_surface else "absorbing_top_state",
                ],
            },
            sort_keys=True,
        )
    )


def test_production_case_provenance_is_decomposition_invariant(tmp_path):
    config = ViscoSHFWIAttenuationConfig()
    hashes = []
    for label, nprocx, nprocy in (("one", 1, 1), ("x", 2, 1), ("y", 1, 2)):
        directory = tmp_path / label
        generate_case(
            directory,
            config=config,
            perturbation="q",
            epsilon=0.01,
            free_surface=True,
            nprocx=nprocx,
            nprocy=nprocy,
            dtinv=3,
        )
        metadata = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        hashes.append(metadata["q_model_sha256"])
        parameters = (directory / "denise.inp").read_text(encoding="ascii")
        assert f"NPROCX ={nprocx}" in parameters
        assert f"NPROCY ={nprocy}" in parameters
        assert "\n L =2\n" in parameters
        assert "FREE_SURF =1" in parameters
        assert "DTINV =3" in parameters
        assert "Q_PARAMETERIZATION_MODE =1" in parameters
    assert len(set(hashes)) == 1


def test_tau_direction_case_applies_fractional_internal_tau_perturbation(tmp_path):
    config = ViscoSHFWIAttenuationConfig()
    epsilon = 0.02
    directory = tmp_path / "tau"
    generate_case(
        directory,
        config=config,
        perturbation="tau",
        epsilon=epsilon,
    )
    q_values = _read_float_grid(directory / "model" / "current.qs")
    mapping = config.mapping()
    tau_values = [q_to_tau_and_derivative(value, mapping)[0] for value in q_values]
    baseline_tau = q_to_tau_and_derivative(config.baseline_qs, mapping)[0]
    expected = [baseline_tau * (1.0 + epsilon * value) for value in config.direction()]
    relative_error = math.sqrt(
        _inner([left - right for left, right in zip(tau_values, expected)], [left - right for left, right in zip(tau_values, expected)])
    ) / math.sqrt(_inner(expected, expected))
    assert relative_error < 8.0e-8
