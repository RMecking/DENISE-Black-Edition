from __future__ import annotations

import json
import math
import random

from tests.utilities.fwi_gradient import (
    sh_density_one_step_forward,
    sh_density_one_step_reverse,
)


def _inner(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _objective(velocity, stress, adjoint_velocity, adjoint_stress):
    return _inner(velocity, adjoint_velocity) + _inner(stress, adjoint_stress)


def test_exact_discrete_sh_density_gradient_decomposition():
    randomizer = random.Random(20260815)
    rows, columns = 4, 5
    cells = rows * columns
    edges = rows * (columns - 1) + (rows - 1) * columns
    density = [
        [1.5 + randomizer.random() for _ in range(columns)]
        for _ in range(rows)
    ]
    vs = [
        [1.5 + randomizer.random() for _ in range(columns)]
        for _ in range(rows)
    ]
    velocity_previous = [2.0 * randomizer.random() - 1.0 for _ in range(cells)]
    stress_previous = [2.0 * randomizer.random() - 1.0 for _ in range(edges)]
    divergence = [
        [0.2 * (2.0 * randomizer.random() - 1.0) for _ in range(edges)]
        for _ in range(cells)
    ]
    gradient = [
        [0.2 * (2.0 * randomizer.random() - 1.0) for _ in range(cells)]
        for _ in range(edges)
    ]
    adjoint_velocity = [2.0 * randomizer.random() - 1.0 for _ in range(cells)]
    adjoint_stress = [2.0 * randomizer.random() - 1.0 for _ in range(edges)]
    direction = [2.0 * randomizer.random() - 1.0 for _ in range(cells)]
    dt = 0.03

    velocity, stress, state = sh_density_one_step_forward(
        velocity_previous, stress_previous, density, vs, divergence, gradient, dt
    )
    reverse = sh_density_one_step_reverse(
        adjoint_velocity,
        adjoint_stress,
        density,
        vs,
        divergence,
        gradient,
        dt,
        state,
    )
    base_inverse_density = list(state["inverse_density"])
    base_staggered_modulus = list(state["staggered_modulus"])

    def perturbed_grid(epsilon):
        values = [
            value + epsilon * component
            for value, component in zip(
                [item for row in density for item in row], direction
            )
        ]
        return [values[j * columns:(j + 1) * columns] for j in range(rows)]

    def value(epsilon, mode):
        varied_density = perturbed_grid(epsilon)
        inverse_override = base_inverse_density if mode == "M" else None
        modulus_override = base_staggered_modulus if mode == "R" else None
        varied_velocity, varied_stress, _ = sh_density_one_step_forward(
            velocity_previous,
            stress_previous,
            varied_density,
            vs,
            divergence,
            gradient,
            dt,
            inverse_density_override=inverse_override,
            staggered_modulus_override=modulus_override,
        )
        return _objective(
            varied_velocity, varied_stress, adjoint_velocity, adjoint_stress
        )

    epsilon = 1.0e-5
    finite_differences = {
        mode: (value(epsilon, mode) - value(-epsilon, mode)) / (2.0 * epsilon)
        for mode in ("R", "M", "T")
    }
    products = {
        "R": _inner(reverse["g_rho_R"], direction),
        "M": _inner(reverse["g_rho_M"], direction),
        "T": _inner(reverse["g_rho_total"], direction),
    }
    relative_errors = {
        mode: abs(finite_differences[mode] - products[mode])
        / max(abs(finite_differences[mode]), abs(products[mode]))
        for mode in ("R", "M", "T")
    }
    fd_decomposition_error = abs(
        finite_differences["T"]
        - finite_differences["R"]
        - finite_differences["M"]
    ) / max(abs(finite_differences["T"]), 1.0e-30)
    gradient_decomposition_error = max(
        abs(total - kinetic - material)
        for total, kinetic, material in zip(
            reverse["g_rho_total"], reverse["g_rho_R"], reverse["g_rho_M"]
        )
    )

    assert relative_errors["R"] < 2.0e-7
    assert relative_errors["M"] < 2.0e-7
    assert relative_errors["T"] < 2.0e-7
    assert fd_decomposition_error < 2.0e-7
    assert gradient_decomposition_error < 5.0e-18
    print(
        json.dumps(
            {
                "finite_differences": finite_differences,
                "gradient_products": products,
                "relative_errors": relative_errors,
                "fd_decomposition_relative_error": fd_decomposition_error,
                "gradient_decomposition_max_abs_error": gradient_decomposition_error,
            }
        )
    )
