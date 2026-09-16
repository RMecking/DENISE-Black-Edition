from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


Vector = list[float]
Matrix = list[list[float]]
Grid = list[list[float]]


@dataclass(frozen=True)
class GSLSCoefficients:
    relaxed_modulus: float
    relaxed_modulus_derivative: float
    stress: float
    stress_derivative: float
    memory: tuple[float, ...]
    memory_derivative: tuple[float, ...]
    recurrence: tuple[float, ...]
    half_dt: float


@dataclass(frozen=True)
class QTauMapping:
    mode: str
    inverse_tau_per_q: float = 0.0
    inverse_tau_offset: float = 0.0


@dataclass(frozen=True)
class GlobalOperatorLayout:
    nx: int
    ny: int
    mechanisms: int

    @property
    def velocity_count(self) -> int:
        return self.nx * self.ny

    @property
    def x_edge_count(self) -> int:
        return (self.nx - 1) * self.ny

    @property
    def y_edge_count(self) -> int:
        return self.nx * self.ny

    @property
    def input_count(self) -> int:
        edge_count = self.x_edge_count + self.y_edge_count
        return self.velocity_count + 2 * edge_count + self.mechanisms * edge_count + 1

    @property
    def output_count(self) -> int:
        edge_count = self.x_edge_count + self.y_edge_count
        receiver_count = 2
        return self.velocity_count + 2 * edge_count + self.mechanisms * edge_count + receiver_count


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def gsls_coefficients(
    *,
    shear_modulus: float,
    tau: float,
    dt: float,
    relaxation_frequencies_hz: Sequence[float],
) -> GSLSCoefficients:
    """Independent coefficients for DENISE's trapezoidal SH GSLS recurrence."""
    _validate_positive("shear_modulus", shear_modulus)
    _validate_positive("tau", tau)
    _validate_positive("dt", dt)
    if not relaxation_frequencies_hz:
        raise ValueError("at least one relaxation mechanism is required")
    for frequency in relaxation_frequencies_hz:
        _validate_positive("relaxation frequency", frequency)

    theta = tuple(1.0 / (2.0 * math.pi * value) for value in relaxation_frequencies_hz)
    omega_reference = 2.0 * math.pi * relaxation_frequencies_hz[0]
    reference_sum = math.fsum(
        (omega_reference * value) ** 2 / (1.0 + (omega_reference * value) ** 2)
        for value in theta
    )
    denominator = 1.0 + reference_sum * tau
    relaxed = shear_modulus / denominator
    relaxed_derivative = -shear_modulus * reference_sum / denominator**2
    mechanism_count = len(theta)
    stress = dt * relaxed * (1.0 + mechanism_count * tau)
    stress_derivative = dt * (
        relaxed_derivative * (1.0 + mechanism_count * tau)
        + mechanism_count * relaxed
    )

    memory = []
    memory_derivative = []
    recurrence = []
    for value in theta:
        eta = dt / value
        b_value = 1.0 / (1.0 + 0.5 * eta)
        c_value = 1.0 - 0.5 * eta
        recurrence.append(b_value * c_value)
        memory.append(-b_value * relaxed * eta * tau)
        memory_derivative.append(
            -b_value * eta * (relaxed_derivative * tau + relaxed)
        )
    return GSLSCoefficients(
        relaxed_modulus=relaxed,
        relaxed_modulus_derivative=relaxed_derivative,
        stress=stress,
        stress_derivative=stress_derivative,
        memory=tuple(memory),
        memory_derivative=tuple(memory_derivative),
        recurrence=tuple(recurrence),
        half_dt=0.5 * dt,
    )


def gsls_forward(
    stress_previous: float,
    memory_previous: Sequence[float],
    strain_rate: float,
    coefficients: GSLSCoefficients,
) -> tuple[float, tuple[float, ...]]:
    if len(memory_previous) != len(coefficients.memory):
        raise ValueError("memory state and coefficient sizes differ")
    memory_next = tuple(
        recurrence * previous + coupling * strain_rate
        for recurrence, previous, coupling in zip(
            coefficients.recurrence, memory_previous, coefficients.memory
        )
    )
    stress_next = (
        stress_previous
        + coefficients.stress * strain_rate
        + coefficients.half_dt
        * math.fsum(left + right for left, right in zip(memory_previous, memory_next))
    )
    return stress_next, memory_next


def gsls_tangent(
    *,
    stress_previous_tangent: float,
    memory_previous_tangent: Sequence[float],
    strain_rate_tangent: float,
    tau_tangent: float,
    stress_previous: float,
    memory_previous: Sequence[float],
    strain_rate: float,
    coefficients: GSLSCoefficients,
) -> tuple[float, tuple[float, ...]]:
    del stress_previous, memory_previous
    if len(memory_previous_tangent) != len(coefficients.memory):
        raise ValueError("memory tangent and coefficient sizes differ")
    memory_next_tangent = tuple(
        recurrence * previous_tangent
        + coupling * strain_rate_tangent
        + coupling_derivative * strain_rate * tau_tangent
        for recurrence, previous_tangent, coupling, coupling_derivative in zip(
            coefficients.recurrence,
            memory_previous_tangent,
            coefficients.memory,
            coefficients.memory_derivative,
        )
    )
    stress_next_tangent = (
        stress_previous_tangent
        + coefficients.stress * strain_rate_tangent
        + coefficients.stress_derivative * strain_rate * tau_tangent
        + coefficients.half_dt
        * math.fsum(
            left + right
            for left, right in zip(memory_previous_tangent, memory_next_tangent)
        )
    )
    return stress_next_tangent, memory_next_tangent


def gsls_transpose(
    *,
    stress_next_adjoint: float,
    memory_next_adjoint: Sequence[float],
    strain_rate: float,
    coefficients: GSLSCoefficients,
) -> tuple[float, tuple[float, ...], float, float]:
    """Exact transpose of :func:`gsls_tangent` for the local state map."""
    if len(memory_next_adjoint) != len(coefficients.memory):
        raise ValueError("memory adjoint and coefficient sizes differ")
    combined = tuple(
        value + coefficients.half_dt * stress_next_adjoint
        for value in memory_next_adjoint
    )
    stress_previous_adjoint = stress_next_adjoint
    memory_previous_adjoint = tuple(
        recurrence * value + coefficients.half_dt * stress_next_adjoint
        for recurrence, value in zip(coefficients.recurrence, combined)
    )
    strain_rate_adjoint = (
        coefficients.stress * stress_next_adjoint
        + math.fsum(
            coupling * value for coupling, value in zip(coefficients.memory, combined)
        )
    )
    tau_adjoint = strain_rate * (
        coefficients.stress_derivative * stress_next_adjoint
        + math.fsum(
            derivative * value
            for derivative, value in zip(coefficients.memory_derivative, combined)
        )
    )
    return (
        stress_previous_adjoint,
        memory_previous_adjoint,
        strain_rate_adjoint,
        tau_adjoint,
    )


def physical_q_mapping(
    *,
    relaxation_frequencies_hz: Sequence[float],
    fmin_hz: float,
    fmax_hz: float,
    df_hz: float,
) -> QTauMapping:
    if not relaxation_frequencies_hz or any(value <= 0.0 for value in relaxation_frequencies_hz):
        raise ValueError("positive relaxation frequencies are required")
    if fmin_hz <= 0.0 or fmax_hz < fmin_hz or df_hz <= 0.0:
        raise ValueError("invalid approximation band")
    sample_count = int(math.floor((fmax_hz - fmin_hz) / df_hz + 1.0e-12)) + 1
    sum_a = 0.0
    sum_ab = 0.0
    sum_aa = 0.0
    for index in range(sample_count):
        omega = 2.0 * math.pi * (fmin_hz + index * df_hz)
        a_sum = 0.0
        b_sum = 0.0
        for frequency in relaxation_frequencies_hz:
            theta = 1.0 / (2.0 * math.pi * frequency)
            omega_theta = omega * theta
            denominator = 1.0 + omega_theta * omega_theta
            a_sum += omega_theta * omega_theta / denominator
            b_sum += omega_theta / denominator
        a_value = 1.0 / b_sum
        b_value = a_sum / b_sum
        sum_a += a_value
        sum_ab += a_value * b_value
        sum_aa += a_value * a_value
    return QTauMapping(
        mode="physical",
        inverse_tau_per_q=sum_a / sum_aa,
        inverse_tau_offset=-sum_ab / sum_aa,
    )


def q_to_tau_and_derivative(q_value: float, mapping: QTauMapping) -> tuple[float, float]:
    _validate_positive("Q", q_value)
    if mapping.mode == "legacy":
        return 2.0 / q_value, -2.0 / q_value**2
    if mapping.mode != "physical":
        raise ValueError(f"unsupported Q mapping: {mapping.mode}")
    inverse_tau = mapping.inverse_tau_per_q * q_value + mapping.inverse_tau_offset
    _validate_positive("inverse tau", inverse_tau)
    tau = 1.0 / inverse_tau
    return tau, -mapping.inverse_tau_per_q * tau * tau


def av_tau(field: Grid) -> Grid:
    """Four-cell staggered average on every complete interior cell quartet."""
    ny, nx = _grid_shape(field)
    return [
        [
            0.25 * (field[j][i] + field[j][i + 1] + field[j + 1][i] + field[j + 1][i + 1])
            for i in range(nx - 1)
        ]
        for j in range(ny - 1)
    ]


def av_tau_vjp(shape: tuple[int, int], sensitivity: Grid) -> Grid:
    ny, nx = shape
    if ny < 2 or nx < 2:
        raise ValueError("tau grid must be at least 2x2")
    if len(sensitivity) != ny - 1 or any(len(row) != nx - 1 for row in sensitivity):
        raise ValueError("staggered sensitivity has the wrong shape")
    result = [[0.0] * nx for _ in range(ny)]
    for j in range(ny - 1):
        for i in range(nx - 1):
            value = 0.25 * sensitivity[j][i]
            result[j][i] += value
            result[j][i + 1] += value
            result[j + 1][i] += value
            result[j + 1][i + 1] += value
    return result


def av_tau_partitioned(
    field: Grid, *, nprocx: int, nprocy: int
) -> tuple[Grid, tuple[tuple[int, int], ...]]:
    """Evaluate the global average by subdomain-owned edges and record seams."""
    ny, nx = _grid_shape(field)
    if nx % nprocx or ny % nprocy:
        raise ValueError("grid must be divisible by the process decomposition")
    width = nx // nprocx
    height = ny // nprocy
    result = [[math.nan] * (nx - 1) for _ in range(ny - 1)]
    seams = []
    for py in range(nprocy):
        for px in range(nprocx):
            i0, i1 = px * width, (px + 1) * width
            j0, j1 = py * height, (py + 1) * height
            for j in range(j0, min(j1, ny - 1)):
                for i in range(i0, min(i1, nx - 1)):
                    result[j][i] = 0.25 * (
                        field[j][i]
                        + field[j][i + 1]
                        + field[j + 1][i]
                        + field[j + 1][i + 1]
                    )
                    if (px + 1 < nprocx and i == i1 - 1) or (
                        py + 1 < nprocy and j == j1 - 1
                    ):
                        seams.append((j, i))
    if any(not math.isfinite(value) for row in result for value in row):
        raise AssertionError("partitioned averaging left an edge unassigned")
    return result, tuple(seams)


def av_tau_partitioned_vjp(
    shape: tuple[int, int], sensitivity: Grid, *, nprocx: int, nprocy: int
) -> Grid:
    ny, nx = shape
    if nx % nprocx or ny % nprocy:
        raise ValueError("grid must be divisible by the process decomposition")
    if len(sensitivity) != ny - 1 or any(len(row) != nx - 1 for row in sensitivity):
        raise ValueError("staggered sensitivity has the wrong shape")
    width = nx // nprocx
    height = ny // nprocy
    result = [[0.0] * nx for _ in range(ny)]
    for py in range(nprocy):
        for px in range(nprocx):
            i0, i1 = px * width, (px + 1) * width
            j0, j1 = py * height, (py + 1) * height
            for j in range(j0, min(j1, ny - 1)):
                for i in range(i0, min(i1, nx - 1)):
                    value = 0.25 * sensitivity[j][i]
                    result[j][i] += value
                    result[j][i + 1] += value
                    result[j + 1][i] += value
                    result[j + 1][i + 1] += value
    return result


def _grid_shape(field: Grid) -> tuple[int, int]:
    ny = len(field)
    nx = len(field[0]) if ny else 0
    if ny < 2 or nx < 2 or any(len(row) != nx for row in field):
        raise ValueError("field must be a rectangular grid of at least 2x2")
    return ny, nx


def _matvec(matrix: Matrix, values: Sequence[float]) -> Vector:
    if matrix and len(matrix[0]) != len(values):
        raise ValueError("matrix and vector sizes differ")
    return [math.fsum(coefficient * value for coefficient, value in zip(row, values)) for row in matrix]


def transpose_matvec(matrix: Matrix, values: Sequence[float]) -> Vector:
    if len(matrix) != len(values):
        raise ValueError("matrix and vector sizes differ")
    columns = len(matrix[0]) if matrix else 0
    return [
        math.fsum(matrix[row][column] * values[row] for row in range(len(matrix)))
        for column in range(columns)
    ]


def dense_linearization(function: Callable[[Sequence[float]], Sequence[float]], input_count: int) -> Matrix:
    zero = [0.0] * input_count
    baseline = list(function(zero))
    columns = []
    for index in range(input_count):
        basis = [0.0] * input_count
        basis[index] = 1.0
        value = list(function(basis))
        columns.append([left - right for left, right in zip(value, baseline)])
    return [list(row) for row in zip(*columns)]


def global_visco_sh_reference(
    *,
    free_surface: bool,
    nprocx: int = 1,
    nprocy: int = 1,
    nx: int = 4,
    ny: int = 4,
    mechanisms: int = 2,
) -> tuple[GlobalOperatorLayout, Callable[[Sequence[float]], Vector]]:
    """Return a small complete linear SH state step independent of DENISE C code.

    The composed map includes velocity/stress updates, GSLS memory, CPML state,
    source injection, receiver sampling, free-surface closure, and a logically
    partitioned exchange.  Its dense linearization is the frozen global oracle.
    """
    if nx % nprocx or ny % nprocy:
        raise ValueError("reference grid must be divisible by the decomposition")
    layout = GlobalOperatorLayout(nx=nx, ny=ny, mechanisms=mechanisms)
    x_count = layout.x_edge_count
    y_count = layout.y_edge_count
    edge_count = x_count + y_count
    dt = 0.03
    inv_rho = [0.7 + 0.02 * index for index in range(layout.velocity_count)]
    tau_x = [0.035 + 0.001 * (index % 5) for index in range(x_count)]
    tau_y = [0.040 + 0.001 * (index % 7) for index in range(y_count)]
    frequencies = tuple(4.0 * (index + 1) for index in range(mechanisms))
    coefficients = [
        gsls_coefficients(
            shear_modulus=2.5 + 0.05 * index,
            tau=tau,
            dt=dt,
            relaxation_frequencies_hz=frequencies,
        )
        for index, tau in enumerate(tau_x + tau_y)
    ]
    cpml_a = [0.07 + 0.002 * (index % 3) for index in range(edge_count)]
    cpml_b = [0.81 + 0.003 * (index % 5) for index in range(edge_count)]
    cpml_k = [1.1 + 0.01 * (index % 4) for index in range(edge_count)]
    source_weights = [0.0] * layout.velocity_count
    source_weights[1] = 0.8
    source_weights[-2] = -0.3
    receiver_indices = (nx + 1, 2 * nx - 2)
    receiver_weights = (1.2, 0.9)

    def exchange(values: Sequence[float], *, rows: int, columns: int) -> Vector:
        # Gather from process-owned interiors.  This is identity globally, but
        # exercises the same ownership partition for 1x1, 2x1 and 1x2.
        width = columns // nprocx
        height = rows // nprocy
        gathered = [0.0] * (rows * columns)
        for py in range(nprocy):
            for px in range(nprocx):
                for j in range(py * height, (py + 1) * height):
                    for i in range(px * width, (px + 1) * width):
                        gathered[j * columns + i] = values[j * columns + i]
        return gathered

    def gradient(velocity: Sequence[float]) -> Vector:
        x_values = []
        for j in range(ny):
            for i in range(nx - 1):
                x_values.append(velocity[j * nx + i + 1] - velocity[j * nx + i])
        y_values = []
        for j in range(ny):
            for i in range(nx):
                if j == 0:
                    y_values.append(0.0 if free_surface else velocity[i])
                else:
                    y_values.append(velocity[j * nx + i] - velocity[(j - 1) * nx + i])
        return x_values + y_values

    gradient_matrix = dense_linearization(gradient, layout.velocity_count)

    def step(values: Sequence[float]) -> Vector:
        if len(values) != layout.input_count:
            raise ValueError("global reference input has the wrong size")
        cursor = 0
        velocity = list(values[cursor:cursor + layout.velocity_count])
        cursor += layout.velocity_count
        stress = list(values[cursor:cursor + edge_count])
        cursor += edge_count
        psi = list(values[cursor:cursor + edge_count])
        cursor += edge_count
        memory_flat = list(values[cursor:cursor + mechanisms * edge_count])
        cursor += mechanisms * edge_count
        source_amplitude = values[cursor]

        divergence = transpose_matvec(gradient_matrix, stress)
        velocity_next = [
            value - dt * inverse * divergence_value + source_amplitude * source_weight
            for value, inverse, divergence_value, source_weight in zip(
                velocity, inv_rho, divergence, source_weights
            )
        ]
        velocity_next = exchange(velocity_next, rows=ny, columns=nx)
        strain = gradient(velocity_next)
        psi_next = [
            b_value * old + a_value * rate
            for old, rate, a_value, b_value in zip(psi, strain, cpml_a, cpml_b)
        ]
        effective_strain = [
            rate / k_value + state
            for rate, k_value, state in zip(strain, cpml_k, psi_next)
        ]
        stress_next = []
        memory_next_by_edge = []
        for edge in range(edge_count):
            old_memory = [memory_flat[mechanism * edge_count + edge] for mechanism in range(mechanisms)]
            new_stress, new_memory = gsls_forward(
                stress[edge], old_memory, effective_strain[edge], coefficients[edge]
            )
            stress_next.append(new_stress)
            memory_next_by_edge.append(new_memory)
        if free_surface:
            for i in range(nx):
                stress_next[x_count + i] = 0.0
        x_stress = exchange(stress_next[:x_count], rows=ny, columns=nx - 1)
        y_stress = exchange(stress_next[x_count:], rows=ny, columns=nx)
        stress_next = x_stress + y_stress
        memory_next = [
            memory_next_by_edge[edge][mechanism]
            for mechanism in range(mechanisms)
            for edge in range(edge_count)
        ]
        receivers = [
            receiver_weights[index] * velocity_next[receiver]
            for index, receiver in enumerate(receiver_indices)
        ]
        return velocity_next + stress_next + psi_next + memory_next + receivers

    return layout, step


def relative_dot_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)
