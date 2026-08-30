"""Independent local forward/finite-difference oracle for M6.3c-7b."""

from __future__ import annotations

from dataclasses import dataclass
import math


C7B_REFERENCE_RELATIVE_MAX = 2.0e-7
C7B_DIRECTIONAL_RELATIVE_MAX = 3.0e-7


@dataclass(frozen=True)
class Case:
    name: str
    dt: float
    dh: float
    qsum: float
    strain_x: float
    strain_y: float
    bar_v: float
    bar_sx: float
    bar_sy: float
    bar_rx: tuple[float, ...]
    bar_qy: tuple[float, ...]
    memory_x: tuple[float, ...]
    memory_y: tuple[float, ...]
    frequencies: tuple[float, ...]
    rhoi: float
    mu_x: float
    mu_y: float
    tau_x: float
    tau_y: float

    @property
    def mechanisms(self):
        return len(self.frequencies)


CASES = (
    Case(
        "l1_mixed", 0.0013, 7.5, -0.37, 0.021, -0.033, 0.62, -0.41,
        0.53, (0.17,), (-0.29,), (0.08,), (-0.11,), (6.0,), 0.00043,
        4.7, 3.9, 0.035, 0.052,
    ),
    Case(
        "l2_signed", 0.0021, 9.0, 0.51, -0.044, 0.027, -0.73, 0.38,
        -0.46, (0.13, -0.21), (-0.19, 0.31), (0.07, -0.05),
        (-0.09, 0.04), (4.0, 11.0), 0.00051, 5.2, 4.4, 0.061, 0.028,
    ),
    Case(
        "l3_mixed", 0.0008, 6.25, 0.29, 0.036, 0.019, 0.44, -0.57,
        -0.35, (0.22, -0.18, 0.09), (-0.14, 0.27, -0.12),
        (0.03, -0.06, 0.1), (-0.08, 0.05, -0.02), (3.0, 9.0, 24.0),
        0.00039, 6.1, 5.6, 0.022, 0.074,
    ),
)


def coefficients(modulus, tau, dt, frequencies):
    """Reconstruct the local production constitutive coefficients."""
    theta = tuple(1.0 / (2.0 * math.pi * value) for value in frequencies)
    eta = tuple(dt / value for value in theta)
    b = tuple(1.0 / (1.0 + 0.5 * value) for value in eta)
    c = tuple(1.0 - 0.5 * value for value in eta)
    omega_reference = 2.0 * math.pi * frequencies[0]
    reference_sum = math.fsum(
        (omega_reference * value) ** 2
        / (1.0 + (omega_reference * value) ** 2)
        for value in theta
    )
    relaxed = modulus / (1.0 + reference_sum * tau)
    stress = dt * relaxed * (1.0 + len(frequencies) * tau)
    recurrence = tuple(left * right for left, right in zip(b, c))
    coupling = tuple(
        -b_value * relaxed * eta_value * tau
        for b_value, eta_value in zip(b, eta)
    )
    return {
        "eta": eta,
        "b": b,
        "reference_sum": reference_sum,
        "stress": stress,
        "recurrence": recurrence,
        "coupling": coupling,
    }


def _constitutive_objective(
    modulus, tau, strain, memory_previous, bar_stress, bar_memory, dt, frequencies
):
    coeff = coefficients(modulus, tau, dt, frequencies)
    memory_next = tuple(
        a * previous + c * strain
        for a, c, previous in zip(
            coeff["recurrence"], coeff["coupling"], memory_previous
        )
    )
    stress_next = coeff["stress"] * strain + 0.5 * dt * math.fsum(
        previous + following
        for previous, following in zip(memory_previous, memory_next)
    )
    return bar_stress * stress_next + math.fsum(
        bar * value for bar, value in zip(bar_memory, memory_next)
    )


def objective(parameters, case: Case):
    rhoi, mu_x, mu_y, tau_x, tau_y = parameters
    velocity = case.dt / case.dh * rhoi * case.qsum
    return (
        case.bar_v * velocity
        + _constitutive_objective(
            mu_x, tau_x, case.strain_x, case.memory_x, case.bar_sx,
            case.bar_rx, case.dt, case.frequencies,
        )
        + _constitutive_objective(
            mu_y, tau_y, case.strain_y, case.memory_y, case.bar_sy,
            case.bar_qy, case.dt, case.frequencies,
        )
    )


def five_point_partial(case: Case, index: int):
    base = [case.rhoi, case.mu_x, case.mu_y, case.tau_x, case.tau_y]
    scale = max(abs(base[index]), 1.0)
    step = 2.0e-4 * scale
    values = []
    for multiple in (-2.0, -1.0, 1.0, 2.0):
        trial = list(base)
        trial[index] += multiple * step
        values.append(objective(trial, case))
    return (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (
        12.0 * step
    )


def finite_difference_gradient(case: Case):
    return tuple(five_point_partial(case, index) for index in range(5))


def five_point_directional(case: Case, direction):
    base = [case.rhoi, case.mu_x, case.mu_y, case.tau_x, case.tau_y]
    step = 1.0e-4
    values = []
    for multiple in (-2.0, -1.0, 1.0, 2.0):
        trial = [
            value + multiple * step * delta
            for value, delta in zip(base, direction)
        ]
        values.append(objective(trial, case))
    return (values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]) / (
        12.0 * step
    )
