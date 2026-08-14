from __future__ import annotations

import math
import re
import struct
from array import array
from pathlib import Path
from typing import Sequence


Grid = Sequence[Sequence[float]]


def _matrix_vector(matrix: Grid, vector: Sequence[float]) -> list[float]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("matrix and vector sizes differ")
    return [math.fsum(a * b for a, b in zip(row, vector)) for row in matrix]


def _transpose_vector(matrix: Grid, vector: Sequence[float]) -> list[float]:
    if len(matrix) != len(vector) or not matrix:
        raise ValueError("matrix and vector sizes differ")
    columns = len(matrix[0])
    if any(len(row) != columns for row in matrix):
        raise ValueError("matrix must be rectangular")
    return [
        math.fsum(matrix[row][column] * vector[row] for row in range(len(matrix)))
        for column in range(columns)
    ]


def sh_one_step_forward(
    velocity_previous: Sequence[float],
    stress_previous: Sequence[float],
    material: Sequence[float],
    divergence: Grid,
    gradient: Grid,
    inverse_density: Sequence[float],
    dt: float,
) -> tuple[list[float], list[float], list[float]]:
    """Reference interior SH step V_n followed by S_n, without PML."""
    if dt <= 0.0 or len(velocity_previous) != len(inverse_density):
        raise ValueError("invalid timestep or velocity/inverse-density size")
    divergence_stress = _matrix_vector(divergence, stress_previous)
    velocity = [
        value + dt * inverse_density[index] * divergence_stress[index]
        for index, value in enumerate(velocity_previous)
    ]
    gradient_velocity = _matrix_vector(gradient, velocity)
    if len(stress_previous) != len(material) or len(material) != len(gradient_velocity):
        raise ValueError("stress, material and gradient sizes differ")
    stress = [
        value + dt * material[index] * gradient_velocity[index]
        for index, value in enumerate(stress_previous)
    ]
    return velocity, stress, gradient_velocity


def sh_one_step_reverse(
    adjoint_velocity_output: Sequence[float],
    adjoint_stress_output: Sequence[float],
    material: Sequence[float],
    gradient_velocity: Sequence[float],
    divergence: Grid,
    gradient: Grid,
    inverse_density: Sequence[float],
    dt: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Exact reverse accumulation for S_n then V_n.

    Returns adjoint velocity/stress inputs, the material gradient, and the
    stress multiplier after V_n^T. The material gradient deliberately uses
    ``adjoint_stress_output`` before that latter stress update.
    """
    stress_pre_update = list(adjoint_stress_output)
    material_gradient = [
        dt * adjoint * derivative
        for adjoint, derivative in zip(stress_pre_update, gradient_velocity)
    ]
    weighted_stress = [
        material[index] * stress_pre_update[index]
        for index in range(len(material))
    ]
    stress_to_velocity = _transpose_vector(gradient, weighted_stress)
    adjoint_velocity = [
        value + dt * stress_to_velocity[index]
        for index, value in enumerate(adjoint_velocity_output)
    ]
    weighted_velocity = [
        inverse_density[index] * adjoint_velocity[index]
        for index in range(len(adjoint_velocity))
    ]
    velocity_to_stress = _transpose_vector(divergence, weighted_velocity)
    adjoint_stress_input = [
        value + dt * velocity_to_stress[index]
        for index, value in enumerate(stress_pre_update)
    ]
    return adjoint_velocity, adjoint_stress_input, material_gradient, stress_pre_update


def sh_density_one_step_forward(
    velocity_previous: Sequence[float],
    stress_previous: Sequence[float],
    density: Grid,
    vs: Grid,
    divergence: Grid,
    gradient: Grid,
    dt: float,
    *,
    inverse_density_override: Sequence[float] | None = None,
    staggered_modulus_override: Sequence[float] | None = None,
) -> tuple[list[float], list[float], dict[str, list[float] | list[list[float]]]]:
    """Exact interior SH step with cell density/Vs and harmonic shear modulus."""
    shape = _validate_rectangular(density, name="density")
    if _validate_rectangular(vs, name="vs") != shape:
        raise ValueError("Density and Vs shapes differ")
    density_flat = [value for row in density for value in row]
    vs_flat = [value for row in vs for value in row]
    if any(value <= 0.0 for value in density_flat + vs_flat):
        raise ValueError("Density and Vs must be positive")
    inverse_density = (
        list(inverse_density_override)
        if inverse_density_override is not None
        else [1.0 / value for value in density_flat]
    )
    cell_modulus = [
        [density[j][i] * vs[j][i] ** 2 for i in range(shape[1])]
        for j in range(shape[0])
    ]
    modulus_x, modulus_y = harmonic_mean(cell_modulus)
    staggered_modulus = (
        list(staggered_modulus_override)
        if staggered_modulus_override is not None
        else [value for row in modulus_x for value in row]
        + [value for row in modulus_y for value in row]
    )
    velocity, stress, gradient_velocity = sh_one_step_forward(
        velocity_previous,
        stress_previous,
        staggered_modulus,
        divergence,
        gradient,
        inverse_density,
        dt,
    )
    return velocity, stress, {
        "density": density_flat,
        "vs": vs_flat,
        "inverse_density": inverse_density,
        "cell_modulus": cell_modulus,
        "staggered_modulus": staggered_modulus,
        "gradient_velocity": gradient_velocity,
        "divergence_stress": _matrix_vector(divergence, stress_previous),
    }


def sh_density_one_step_reverse(
    adjoint_velocity_output: Sequence[float],
    adjoint_stress_output: Sequence[float],
    density: Grid,
    vs: Grid,
    divergence: Grid,
    gradient: Grid,
    dt: float,
    forward_state: dict[str, list[float] | list[list[float]]],
) -> dict[str, list[float]]:
    """Return exact R, material, and total cell-density derivatives separately."""
    rows, columns = _validate_rectangular(density, name="density")
    density_flat = [value for row in density for value in row]
    vs_flat = [value for row in vs for value in row]
    inverse_density = list(forward_state["inverse_density"])
    staggered_modulus = list(forward_state["staggered_modulus"])
    gradient_velocity = list(forward_state["gradient_velocity"])
    divergence_stress = list(forward_state["divergence_stress"])

    stress_pre = list(adjoint_stress_output)
    staggered_modulus_gradient = [
        dt * adjoint * derivative
        for adjoint, derivative in zip(stress_pre, gradient_velocity)
    ]
    velocity_from_stress = _transpose_vector(
        gradient,
        [
            modulus * adjoint
            for modulus, adjoint in zip(staggered_modulus, stress_pre)
        ],
    )
    velocity_b = [
        value + dt * contribution
        for value, contribution in zip(adjoint_velocity_output, velocity_from_stress)
    ]
    density_r = [
        -dt * multiplier * divergence_value / density_value**2
        for multiplier, divergence_value, density_value in zip(
            velocity_b, divergence_stress, density_flat
        )
    ]

    x_count = rows * (columns - 1)
    x_sensitivity_flat = staggered_modulus_gradient[:x_count]
    y_sensitivity_flat = staggered_modulus_gradient[x_count:]
    x_sensitivity = [
        x_sensitivity_flat[j * (columns - 1):(j + 1) * (columns - 1)]
        for j in range(rows)
    ]
    y_sensitivity = [
        y_sensitivity_flat[j * columns:(j + 1) * columns]
        for j in range(rows - 1)
    ]
    cell_modulus = forward_state["cell_modulus"]
    cell_modulus_gradient_grid = harmonic_mean_vjp(
        cell_modulus, x_sensitivity, y_sensitivity
    )
    cell_modulus_gradient = [
        value for row in cell_modulus_gradient_grid for value in row
    ]
    density_m = [
        vs_value**2 * derivative
        for vs_value, derivative in zip(vs_flat, cell_modulus_gradient)
    ]
    density_total = [left + right for left, right in zip(density_r, density_m)]

    adjoint_stress_input = [
        value + dt * contribution
        for value, contribution in zip(
            stress_pre,
            _transpose_vector(
                divergence,
                [
                    inverse * multiplier
                    for inverse, multiplier in zip(inverse_density, velocity_b)
                ],
            ),
        )
    ]
    return {
        "adjoint_velocity_input": velocity_b,
        "adjoint_stress_input": adjoint_stress_input,
        "velocity_b": velocity_b,
        "g_rho_R": density_r,
        "g_mu_x": x_sensitivity_flat,
        "g_mu_y": y_sensitivity_flat,
        "g_mu_cell": cell_modulus_gradient,
        "g_rho_M": density_m,
        "g_rho_total": density_total,
    }


def _validate_rectangular(field: Grid, *, name: str) -> tuple[int, int]:
    rows = len(field)
    columns = len(field[0]) if rows else 0
    if rows < 2 or columns < 2 or any(len(row) != columns for row in field):
        raise ValueError(f"{name} must be a rectangular grid of at least 2x2")
    return rows, columns


def harmonic_mean(field: Grid) -> tuple[list[list[float]], list[list[float]]]:
    """Map cell values to x/y edges with H(a,b)=2ab/(a+b)."""
    rows, columns = _validate_rectangular(field, name="field")
    if any(value <= 0.0 for row in field for value in row):
        raise ValueError("Harmonic averaging requires positive values")
    x_edges = [
        [2.0 * row[i] * row[i + 1] / (row[i] + row[i + 1]) for i in range(columns - 1)]
        for row in field
    ]
    y_edges = [
        [2.0 * field[j][i] * field[j + 1][i] / (field[j][i] + field[j + 1][i]) for i in range(columns)]
        for j in range(rows - 1)
    ]
    return x_edges, y_edges


def harmonic_mean_jvp(field: Grid, perturbation: Grid) -> tuple[list[list[float]], list[list[float]]]:
    """Apply the exact Jacobian of :func:`harmonic_mean`."""
    rows, columns = _validate_rectangular(field, name="field")
    if _validate_rectangular(perturbation, name="perturbation") != (rows, columns):
        raise ValueError("Field and perturbation shapes differ")
    x_edges = []
    for j in range(rows):
        values = []
        for i in range(columns - 1):
            a, b = field[j][i], field[j][i + 1]
            denominator = (a + b) ** 2
            values.append(
                2.0 * b * b / denominator * perturbation[j][i]
                + 2.0 * a * a / denominator * perturbation[j][i + 1]
            )
        x_edges.append(values)
    y_edges = []
    for j in range(rows - 1):
        values = []
        for i in range(columns):
            a, b = field[j][i], field[j + 1][i]
            denominator = (a + b) ** 2
            values.append(
                2.0 * b * b / denominator * perturbation[j][i]
                + 2.0 * a * a / denominator * perturbation[j + 1][i]
            )
        y_edges.append(values)
    return x_edges, y_edges


def harmonic_mean_vjp(
    field: Grid, x_sensitivity: Grid, y_sensitivity: Grid
) -> list[list[float]]:
    """Apply the exact transpose of the cell-to-staggered harmonic map."""
    rows, columns = _validate_rectangular(field, name="field")
    if len(x_sensitivity) != rows or any(len(row) != columns - 1 for row in x_sensitivity):
        raise ValueError("x sensitivity has the wrong shape")
    if len(y_sensitivity) != rows - 1 or any(len(row) != columns for row in y_sensitivity):
        raise ValueError("y sensitivity has the wrong shape")
    result = [[0.0] * columns for _ in range(rows)]
    for j in range(rows):
        for i in range(columns - 1):
            a, b = field[j][i], field[j][i + 1]
            denominator = (a + b) ** 2
            value = x_sensitivity[j][i]
            result[j][i] += 2.0 * b * b / denominator * value
            result[j][i + 1] += 2.0 * a * a / denominator * value
    for j in range(rows - 1):
        for i in range(columns):
            a, b = field[j][i], field[j + 1][i]
            denominator = (a + b) ** 2
            value = y_sensitivity[j][i]
            result[j][i] += 2.0 * b * b / denominator * value
            result[j + 1][i] += 2.0 * a * a / denominator * value
    return result


def compliance_average(field: Grid) -> tuple[list[list[float]], list[list[float]]]:
    """Map cell compliance to staggered compliance by arithmetic averaging."""
    rows, columns = _validate_rectangular(field, name="field")
    x_edges = [
        [0.5 * (row[i] + row[i + 1]) for i in range(columns - 1)] for row in field
    ]
    y_edges = [
        [0.5 * (field[j][i] + field[j + 1][i]) for i in range(columns)]
        for j in range(rows - 1)
    ]
    return x_edges, y_edges


def compliance_average_vjp(
    shape: tuple[int, int], x_sensitivity: Grid, y_sensitivity: Grid
) -> list[list[float]]:
    """Apply the transpose of the cell-to-staggered compliance average."""
    rows, columns = shape
    if rows < 2 or columns < 2:
        raise ValueError("shape must be at least 2x2")
    if len(x_sensitivity) != rows or any(len(row) != columns - 1 for row in x_sensitivity):
        raise ValueError("x sensitivity has the wrong shape")
    if len(y_sensitivity) != rows - 1 or any(len(row) != columns for row in y_sensitivity):
        raise ValueError("y sensitivity has the wrong shape")
    result = [[0.0] * columns for _ in range(rows)]
    for j in range(rows):
        for i in range(columns - 1):
            result[j][i] += 0.5 * x_sensitivity[j][i]
            result[j][i + 1] += 0.5 * x_sensitivity[j][i]
    for j in range(rows - 1):
        for i in range(columns):
            result[j][i] += 0.5 * y_sensitivity[j][i]
            result[j + 1][i] += 0.5 * y_sensitivity[j][i]
    return result


def compliance_vs_factor(density: float, vs: float) -> float:
    """Return |d(1 / (rho Vs^2)) / dVs| under DENISE's sign convention."""
    if density <= 0.0 or vs <= 0.0:
        raise ValueError("Density and Vs must be positive")
    return 2.0 / (density * vs**3)


def compliance_density_factor(density: float, vs: float) -> float:
    """Return |d(1 / (rho Vs^2)) / drho|."""
    if density <= 0.0 or vs <= 0.0:
        raise ValueError("Density and Vs must be positive")
    return 1.0 / (density**2 * vs**2)


def gaussian_direction(
    *, nx: int, ny: int, dh_m: float, center_x_m: float, center_y_m: float, sigma_m: float
) -> list[float]:
    """Create an x-major/y-minor smooth direction with max(abs(p)) == 1."""
    if min(nx, ny) <= 0 or dh_m <= 0.0 or sigma_m <= 0.0:
        raise ValueError("Grid dimensions, spacing and sigma must be positive")
    values = []
    for ix in range(1, nx + 1):
        x = ix * dh_m
        for iy in range(1, ny + 1):
            y = iy * dh_m
            radius2 = (x - center_x_m) ** 2 + (y - center_y_m) ** 2
            values.append(math.exp(-0.5 * radius2 / sigma_m**2))
    peak = max(values)
    return [value / peak for value in values]


def flat_top_direction(
    *, nx: int, ny: int, dh_m: float, center_x_m: float, center_y_m: float,
    half_width_x_m: float, half_width_y_m: float, taper_m: float
) -> list[float]:
    """Create a unit plateau with a separable raised-cosine exterior taper."""
    if min(nx, ny) <= 0 or min(dh_m, half_width_x_m, half_width_y_m, taper_m) <= 0.0:
        raise ValueError("Grid, plateau and taper dimensions must be positive")

    def axis_weight(distance: float, half_width: float) -> float:
        outside = abs(distance) - half_width
        if outside <= 0.0:
            return 1.0
        if outside >= taper_m:
            return 0.0
        return 0.5 * (1.0 + math.cos(math.pi * outside / taper_m))

    return [
        axis_weight(ix * dh_m - center_x_m, half_width_x_m)
        * axis_weight(iy * dh_m - center_y_m, half_width_y_m)
        for ix in range(1, nx + 1)
        for iy in range(1, ny + 1)
    ]


def directional_derivative(gradient: Sequence[float], direction: Sequence[float]) -> float:
    """DENISE model-space convention: unweighted discrete cell sum, no DH^2."""
    if len(gradient) != len(direction):
        raise ValueError("Gradient and direction sizes differ")
    return math.fsum(left * right for left, right in zip(gradient, direction))


def central_difference(j_plus: float, j_minus: float, epsilon: float) -> float:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return (j_plus - j_minus) / (2.0 * epsilon)


def read_float_grid(path: Path, count: int) -> list[float]:
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, count)
        if stream.read(1):
            raise ValueError(f"Unexpected trailing data in {path}")
    if len(values) != count:
        raise ValueError(f"Expected {count} floats in {path}, found {len(values)}")
    return list(values)


def read_su_float_samples(path: Path, trace_count: int, samples_per_trace: int) -> list[float]:
    """Read native-endian IEEE float samples from DENISE's header-only SU output."""
    samples = []
    with path.open("rb") as stream:
        for _ in range(trace_count):
            header = stream.read(240)
            if len(header) != 240:
                raise ValueError(f"Truncated SU header in {path}")
            payload = stream.read(4 * samples_per_trace)
            if len(payload) != 4 * samples_per_trace:
                raise ValueError(f"Truncated SU trace in {path}")
            samples.extend(struct.unpack(f"={samples_per_trace}f", payload))
        if stream.read(1):
            raise ValueError(f"Unexpected trailing SU data in {path}")
    return samples


def l2_objective_from_reversed_residual_su(
    path: Path, trace_count: int, samples_per_trace: int
) -> float:
    residual = read_su_float_samples(path, trace_count, samples_per_trace)
    return 0.5 * math.fsum(value * value for value in residual)


def parse_initial_objective(stdout: str) -> float:
    matches = re.findall(r"L2t\[1\]\s*=\s*([+\-0-9.eE]+)", stdout)
    if not matches:
        raise ValueError("DENISE stdout does not contain L2t[1]")
    return float(matches[-1])
