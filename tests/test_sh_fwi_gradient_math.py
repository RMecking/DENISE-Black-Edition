from __future__ import annotations

import re
import math
import random
import json
from pathlib import Path

import pytest

from tests.utilities.fwi_gradient import (
    compliance_average,
    compliance_average_vjp,
    compliance_density_factor,
    compliance_vs_factor,
    harmonic_mean,
    harmonic_mean_jvp,
    harmonic_mean_vjp,
    sh_one_step_forward,
    sh_one_step_reverse,
)


def _inner(left, right):
    return math.fsum(a * b for left_row, right_row in zip(left, right) for a, b in zip(left_row, right_row))


def test_compliance_chain_rule_matches_mu_equals_rho_vs_squared():
    rho = 2000.0
    vs = 2000.0
    step = 0.01
    c_plus = 1.0 / (rho * (vs + step) ** 2)
    c_minus = 1.0 / (rho * (vs - step) ** 2)
    numerical_magnitude = abs((c_plus - c_minus) / (2.0 * step))
    assert numerical_magnitude == pytest.approx(compliance_vs_factor(rho, vs), rel=1.0e-9)
    assert compliance_density_factor(rho, vs) == pytest.approx(1.0 / (rho**2 * vs**2))


def test_harmonic_material_map_jvp_and_vjp():
    randomizer = random.Random(20260813)
    field = [[1.0 + 4.0 * randomizer.random() for _ in range(7)] for _ in range(6)]
    perturbation = [[2.0 * randomizer.random() - 1.0 for _ in range(7)] for _ in range(6)]
    qx = [[2.0 * randomizer.random() - 1.0 for _ in range(6)] for _ in range(6)]
    qy = [[2.0 * randomizer.random() - 1.0 for _ in range(7)] for _ in range(5)]

    analytic_x, analytic_y = harmonic_mean_jvp(field, perturbation)
    epsilon = 1.0e-5
    plus = [[a + epsilon * p for a, p in zip(row, prow)] for row, prow in zip(field, perturbation)]
    minus = [[a - epsilon * p for a, p in zip(row, prow)] for row, prow in zip(field, perturbation)]
    plus_x, plus_y = harmonic_mean(plus)
    minus_x, minus_y = harmonic_mean(minus)
    fd_x = [[(a - b) / (2.0 * epsilon) for a, b in zip(row_a, row_b)] for row_a, row_b in zip(plus_x, minus_x)]
    fd_y = [[(a - b) / (2.0 * epsilon) for a, b in zip(row_a, row_b)] for row_a, row_b in zip(plus_y, minus_y)]
    fd_error = math.sqrt(_inner([[a-b for a,b in zip(ar,fr)] for ar,fr in zip(analytic_x,fd_x)], [[a-b for a,b in zip(ar,fr)] for ar,fr in zip(analytic_x,fd_x)]) + _inner([[a-b for a,b in zip(ar,fr)] for ar,fr in zip(analytic_y,fd_y)], [[a-b for a,b in zip(ar,fr)] for ar,fr in zip(analytic_y,fd_y)]))
    fd_norm = math.sqrt(_inner(fd_x, fd_x) + _inner(fd_y, fd_y))
    relative_jvp_error = fd_error / fd_norm
    assert relative_jvp_error < 2.0e-10

    lhs = _inner(analytic_x, qx) + _inner(analytic_y, qy)
    transpose = harmonic_mean_vjp(field, qx, qy)
    rhs = _inner(perturbation, transpose)
    relative_adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert relative_adjoint_error < 5.0e-15
    print(
        json.dumps(
            {
                "operator": "harmonic_modulus",
                "relative_jvp_error": relative_jvp_error,
                "vjp_lhs": lhs,
                "vjp_rhs": rhs,
                "relative_adjoint_error": relative_adjoint_error,
            }
        )
    )


def test_compliance_average_identity_and_vjp():
    randomizer = random.Random(20260814)
    modulus = [[1.0 + 4.0 * randomizer.random() for _ in range(7)] for _ in range(6)]
    compliance = [[1.0 / value for value in row] for row in modulus]
    harmonic_x, harmonic_y = harmonic_mean(modulus)
    averaged_x, averaged_y = compliance_average(compliance)
    reciprocal_x = [value for row in harmonic_x for value in (1.0 / item for item in row)]
    reciprocal_y = [value for row in harmonic_y for value in (1.0 / item for item in row)]
    averaged_x_flat = [value for row in averaged_x for value in row]
    averaged_y_flat = [value for row in averaged_y for value in row]
    assert reciprocal_x == pytest.approx(averaged_x_flat)
    assert reciprocal_y == pytest.approx(averaged_y_flat)
    identity_error = max(
        max(abs(a - b) for a, b in zip(reciprocal_x, averaged_x_flat)),
        max(abs(a - b) for a, b in zip(reciprocal_y, averaged_y_flat)),
    )

    perturbation = [[2.0 * randomizer.random() - 1.0 for _ in range(7)] for _ in range(6)]
    qx = [[2.0 * randomizer.random() - 1.0 for _ in range(6)] for _ in range(6)]
    qy = [[2.0 * randomizer.random() - 1.0 for _ in range(7)] for _ in range(5)]
    jvp_x, jvp_y = compliance_average(perturbation)
    epsilon = 1.0e-5
    plus = [[a + epsilon * p for a, p in zip(row, prow)] for row, prow in zip(compliance, perturbation)]
    minus = [[a - epsilon * p for a, p in zip(row, prow)] for row, prow in zip(compliance, perturbation)]
    plus_x, plus_y = compliance_average(plus)
    minus_x, minus_y = compliance_average(minus)
    fd_x = [[(a - b) / (2.0 * epsilon) for a, b in zip(ar, br)] for ar, br in zip(plus_x, minus_x)]
    fd_y = [[(a - b) / (2.0 * epsilon) for a, b in zip(ar, br)] for ar, br in zip(plus_y, minus_y)]
    error_x = [[a - b for a, b in zip(ar, br)] for ar, br in zip(jvp_x, fd_x)]
    error_y = [[a - b for a, b in zip(ar, br)] for ar, br in zip(jvp_y, fd_y)]
    relative_jvp_error = math.sqrt(_inner(error_x, error_x) + _inner(error_y, error_y)) / math.sqrt(
        _inner(fd_x, fd_x) + _inner(fd_y, fd_y)
    )
    assert relative_jvp_error < 2.0e-10
    lhs = _inner(jvp_x, qx) + _inner(jvp_y, qy)
    rhs = _inner(perturbation, compliance_average_vjp((6, 7), qx, qy))
    relative_adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert relative_adjoint_error < 5.0e-15
    print(
        json.dumps(
            {
                "operator": "compliance_average",
                "reciprocal_identity_max_abs_error": identity_error,
                "relative_jvp_error": relative_jvp_error,
                "vjp_lhs": lhs,
                "vjp_rhs": rhs,
                "relative_adjoint_error": relative_adjoint_error,
            }
        )
    )


def test_discrete_sh_one_step_transpose_and_material_time_level():
    velocity_previous = [0.7, -0.2, 0.4]
    stress_previous = [-0.3, 0.6]
    material = [2.1, 1.7]
    inverse_density = [0.8, 0.9, 1.1]
    divergence = [[0.4, -0.2], [-0.3, 0.5], [0.1, 0.6]]
    gradient = [[-0.7, 0.2, 0.3], [0.4, -0.1, 0.8]]
    dt = 0.03
    adjoint_velocity_output = [-0.5, 0.9, 0.2]
    adjoint_stress_output = [0.6, -0.4]

    velocity, stress, gradient_velocity = sh_one_step_forward(
        velocity_previous, stress_previous, material, divergence, gradient,
        inverse_density, dt,
    )
    adjoint_velocity_input, adjoint_stress_input, material_gradient, stress_pre = (
        sh_one_step_reverse(
            adjoint_velocity_output, adjoint_stress_output, material,
            gradient_velocity, divergence, gradient, inverse_density, dt,
        )
    )
    lhs = _inner([velocity, stress], [adjoint_velocity_output, adjoint_stress_output])
    rhs = _inner(
        [velocity_previous, stress_previous],
        [adjoint_velocity_input, adjoint_stress_input],
    )
    relative_adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    assert relative_adjoint_error < 5.0e-15

    material_direction = [0.25, -0.35]
    epsilon = 1.0e-4
    objectives = []
    for sign in (1.0, -1.0):
        perturbed = [
            value + sign * epsilon * direction
            for value, direction in zip(material, material_direction)
        ]
        perturbed_velocity, perturbed_stress, _ = sh_one_step_forward(
            velocity_previous, stress_previous, perturbed, divergence, gradient,
            inverse_density, dt,
        )
        objectives.append(
            _inner(
                [perturbed_velocity, perturbed_stress],
                [adjoint_velocity_output, adjoint_stress_output],
            )
        )
    finite_difference = (objectives[0] - objectives[1]) / (2.0 * epsilon)
    correct_product = math.fsum(
        value * direction for value, direction in zip(material_gradient, material_direction)
    )
    wrong_post_gradient = [
        dt * adjoint * derivative
        for adjoint, derivative in zip(adjoint_stress_input, gradient_velocity)
    ]
    wrong_post_product = math.fsum(
        value * direction for value, direction in zip(wrong_post_gradient, material_direction)
    )
    correct_relative_error = abs(finite_difference - correct_product) / abs(finite_difference)
    wrong_post_relative_error = abs(finite_difference - wrong_post_product) / abs(finite_difference)
    assert stress_pre == adjoint_stress_output
    assert correct_relative_error < 2.0e-9
    assert wrong_post_relative_error > 1.0e-3
    print(
        json.dumps(
            {
                "operator": "discrete_sh_one_step",
                "transpose_relative_error": relative_adjoint_error,
                "material_fd": finite_difference,
                "material_pre_stress_product": correct_product,
                "material_pre_stress_relative_error": correct_relative_error,
                "material_post_stress_product": wrong_post_product,
                "material_post_stress_relative_error": wrong_post_relative_error,
            }
        )
    )

def test_exact_elastic_sh_path_bypasses_legacy_chain_rule(repository_root: Path):
    sh_source = (repository_root / "src" / "SH" / "sh.c").read_text(encoding="utf-8")
    assembly_source = (
        repository_root / "src" / "SH" / "assemble_gradSH_exact.c"
    ).read_text(encoding="utf-8")
    legacy_source = (
        repository_root / "src" / "SH" / "ass_gradSH.c"
    ).read_text(encoding="utf-8")
    sh_compact = re.sub(r"\s+", "", sh_source)
    assembly_compact = re.sub(r"\s+", "", assembly_source)
    legacy_compact = re.sub(r"\s+", "", legacy_source)

    assert "exact_elastic_sh_adjoint=((MODE==1)&&(mode==1)&&(INVMAT1==1));" in sh_compact
    assert "if((MODE!=1)||(INVMAT1!=1))return;" in assembly_compact
    assert "assemble_gradSH_exact(fwiSH,matSH,mpiPSV,req_send,req_rec);" in (
        re.sub(
            r"\s+",
            "",
            (repository_root / "src" / "SH" / "grad_obj_sh.c").read_text(
                encoding="utf-8"
            ),
        )
    )
    # The legacy conversion may remain for other parameterizations, but the
    # repaired physical-Vs path exits this cell before reaching it.
    physical_parameter_early_exits = re.findall(
        r"if\(INVMAT1==1\)\{.*?continue;\}", legacy_compact
    )
    assert len(physical_parameter_early_exits) == 2
