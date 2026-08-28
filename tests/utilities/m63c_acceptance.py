from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


PRODUCTION_DOT_RELATIVE_MAX = 1.0e-5
DIRECTIONAL_GRADIENT_RELATIVE_MAX = 5.0e-3
ZERO_STEP_OBJECTIVE_RELATIVE_MAX = 1.0e-12

FORWARD_OPERATOR_ORDER = (
    "source_injection",
    "velocity_update",
    "mpi_velocity_exchange",
    "free_surface_velocity_completion",
    "strain_spatial_derivative_and_cpml",
    "gsls_stress_memory_update",
    "free_surface_stress_completion",
    "mpi_stress_exchange",
    "receiver_sampling",
)

REVERSE_OPERATOR_ORDER = (
    "receiver_sampling_transpose",
    "mpi_stress_exchange_transpose",
    "free_surface_stress_completion_transpose",
    "gsls_stress_memory_update_transpose",
    "strain_spatial_derivative_and_cpml_transpose",
    "free_surface_velocity_completion_transpose",
    "mpi_velocity_exchange_transpose",
    "velocity_update_transpose",
    "source_injection_transpose",
)

PRODUCTION_DOT_CASES = (
    (False, 1, 1),
    (False, 1, 2),
    (False, 2, 1),
    (True, 1, 1),
)

TAU_VJP_DECOMPOSITIONS = (
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
)


@dataclass(frozen=True)
class LocalGSLSCoefficients:
    """Coefficients and analytic material derivatives for one SH GSLS point."""

    recurrence: tuple[float, ...]
    coupling: tuple[float, ...]
    stress: float
    coupling_tau_derivative: tuple[float, ...]
    stress_tau_derivative: float
    coupling_modulus_derivative: tuple[float, ...]
    stress_modulus_derivative: float
    half_dt: float


def local_gsls_coefficients(
    *,
    unrelaxed_shear_modulus: float,
    tau: float,
    dt: float,
    relaxation_frequencies_hz: Sequence[float],
) -> LocalGSLSCoefficients:
    """Independent Black-Edition GSLS coefficient reference.

    ``unrelaxed_shear_modulus`` is the native material variable M.  The
    reference-frequency relaxation correction is included before forming F
    and C_l, so derivatives with respect to both tau and M are explicit.
    """
    if unrelaxed_shear_modulus <= 0.0 or tau <= 0.0 or dt <= 0.0:
        raise ValueError("M, tau, and dt must be positive")
    if not relaxation_frequencies_hz or any(
        frequency <= 0.0 for frequency in relaxation_frequencies_hz
    ):
        raise ValueError("positive relaxation frequencies are required")

    theta = tuple(
        1.0 / (2.0 * math.pi * frequency)
        for frequency in relaxation_frequencies_hz
    )
    omega_reference = 2.0 * math.pi * relaxation_frequencies_hz[0]
    reference_sum = math.fsum(
        (omega_reference * value) ** 2
        / (1.0 + (omega_reference * value) ** 2)
        for value in theta
    )
    denominator = 1.0 + reference_sum * tau
    relaxed = unrelaxed_shear_modulus / denominator
    relaxed_tau = -unrelaxed_shear_modulus * reference_sum / denominator**2
    relaxed_modulus = 1.0 / denominator
    mechanism_count = len(theta)

    stress = dt * relaxed * (1.0 + mechanism_count * tau)
    stress_tau = dt * (
        relaxed_tau * (1.0 + mechanism_count * tau)
        + mechanism_count * relaxed
    )
    stress_modulus = dt * relaxed_modulus * (1.0 + mechanism_count * tau)

    recurrence = []
    coupling = []
    coupling_tau = []
    coupling_modulus = []
    for value in theta:
        eta = dt / value
        b_value = 1.0 / (1.0 + 0.5 * eta)
        recurrence.append(b_value * (1.0 - 0.5 * eta))
        coupling.append(-b_value * relaxed * eta * tau)
        coupling_tau.append(-b_value * eta * (relaxed_tau * tau + relaxed))
        coupling_modulus.append(-b_value * eta * relaxed_modulus * tau)

    return LocalGSLSCoefficients(
        recurrence=tuple(recurrence),
        coupling=tuple(coupling),
        stress=stress,
        coupling_tau_derivative=tuple(coupling_tau),
        stress_tau_derivative=stress_tau,
        coupling_modulus_derivative=tuple(coupling_modulus),
        stress_modulus_derivative=stress_modulus,
        half_dt=0.5 * dt,
    )


def local_gsls_forward(
    stress_previous: float,
    memory_previous: Sequence[float],
    strain: float,
    coefficients: LocalGSLSCoefficients,
) -> tuple[float, tuple[float, ...]]:
    if len(memory_previous) != len(coefficients.recurrence):
        raise ValueError("memory/coefficient size mismatch")
    memory_next = tuple(
        recurrence * previous + coupling * strain
        for recurrence, previous, coupling in zip(
            coefficients.recurrence,
            memory_previous,
            coefficients.coupling,
        )
    )
    stress_next = (
        stress_previous
        + coefficients.stress * strain
        + coefficients.half_dt
        * math.fsum(
            previous + following
            for previous, following in zip(memory_previous, memory_next)
        )
    )
    return stress_next, memory_next


def local_gsls_tangent(
    *,
    stress_previous_tangent: float,
    memory_previous_tangent: Sequence[float],
    strain_tangent: float,
    tau_tangent: float,
    modulus_tangent: float,
    strain: float,
    coefficients: LocalGSLSCoefficients,
) -> tuple[float, tuple[float, ...]]:
    memory_next_tangent = tuple(
        recurrence * previous
        + coupling * strain_tangent
        + strain
        * (coupling_tau * tau_tangent + coupling_modulus * modulus_tangent)
        for recurrence, previous, coupling, coupling_tau, coupling_modulus in zip(
            coefficients.recurrence,
            memory_previous_tangent,
            coefficients.coupling,
            coefficients.coupling_tau_derivative,
            coefficients.coupling_modulus_derivative,
        )
    )
    stress_next_tangent = (
        stress_previous_tangent
        + coefficients.stress * strain_tangent
        + strain
        * (
            coefficients.stress_tau_derivative * tau_tangent
            + coefficients.stress_modulus_derivative * modulus_tangent
        )
        + coefficients.half_dt
        * math.fsum(
            previous + following
            for previous, following in zip(
                memory_previous_tangent, memory_next_tangent
            )
        )
    )
    return stress_next_tangent, memory_next_tangent


def local_gsls_transpose(
    *,
    stress_next_adjoint: float,
    memory_next_adjoint: Sequence[float],
    strain: float,
    coefficients: LocalGSLSCoefficients,
) -> tuple[float, tuple[float, ...], float, float, float]:
    """Exact reverse of :func:`local_gsls_tangent`.

    The returned entries are adjoints of previous stress, previous memory,
    strain, tau, and unrelaxed shear modulus M, respectively.
    """
    if len(memory_next_adjoint) != len(coefficients.recurrence):
        raise ValueError("memory/coefficient size mismatch")
    t_values = tuple(
        value + coefficients.half_dt * stress_next_adjoint
        for value in memory_next_adjoint
    )
    stress_previous_adjoint = stress_next_adjoint
    memory_previous_adjoint = tuple(
        recurrence * t_value + coefficients.half_dt * stress_next_adjoint
        for recurrence, t_value in zip(coefficients.recurrence, t_values)
    )
    strain_adjoint = (
        coefficients.stress * stress_next_adjoint
        + math.fsum(
            coupling * t_value
            for coupling, t_value in zip(coefficients.coupling, t_values)
        )
    )
    tau_adjoint = strain * (
        coefficients.stress_tau_derivative * stress_next_adjoint
        + math.fsum(
            derivative * t_value
            for derivative, t_value in zip(
                coefficients.coupling_tau_derivative, t_values
            )
        )
    )
    modulus_adjoint = strain * (
        coefficients.stress_modulus_derivative * stress_next_adjoint
        + math.fsum(
            derivative * t_value
            for derivative, t_value in zip(
                coefficients.coupling_modulus_derivative, t_values
            )
        )
    )
    return (
        stress_previous_adjoint,
        memory_previous_adjoint,
        strain_adjoint,
        tau_adjoint,
        modulus_adjoint,
    )


def q_to_tau(q_value: float, *, mode: str, a: float = 0.0, b: float = 0.0):
    if q_value <= 0.0:
        raise ValueError("Q must be positive")
    if mode == "legacy":
        return 2.0 / q_value, -2.0 / q_value**2
    if mode != "physical":
        raise ValueError("mode must be legacy or physical")
    denominator = a * q_value + b
    if a <= 0.0 or denominator <= 0.0:
        raise ValueError("physical-Q mapping requires positive a and a*Q+b")
    tau = 1.0 / denominator
    return tau, -a * tau**2


def q_gradient_from_native_tau(
    native_tau_gradient: Sequence[Sequence[float]],
    q_field: Sequence[Sequence[float]],
    *,
    mode: str,
    a: float = 0.0,
    b: float = 0.0,
) -> list[list[float]]:
    """Apply av_tau transpose, then the cell-centered Q chain rule."""
    from tests.utilities.visco_sh_fwi_attenuation import av_tau_vjp

    ny = len(q_field)
    nx = len(q_field[0]) if ny else 0
    cell_tau_gradient = av_tau_vjp((ny, nx), [list(row) for row in native_tau_gradient])
    return [
        [
            cell_tau_gradient[j][i]
            * q_to_tau(q_field[j][i], mode=mode, a=a, b=b)[1]
            for i in range(nx)
        ]
        for j in range(ny)
    ]


def dtinv_quadrature(
    values: Sequence[float], *, dt: float, dtinv: int
) -> float:
    """The predeclared DT*DTINV sampled correlation convention."""
    if dt <= 0.0 or dtinv < 1:
        raise ValueError("dt must be positive and DTINV must be at least one")
    return dt * dtinv * math.fsum(values[::dtinv])


def relative_agreement(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)
