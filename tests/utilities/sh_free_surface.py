from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


SUPPORTED_FD_ORDERS = (2, 4, 6, 8, 10, 12)
FLOAT32_EPSILON = 2.0**-24


# Independent transcription of the coefficient contract in holbergcoeff.c.
# Each tuple contains c_1..c_M; the dispersion-points entry is intentionally
# excluded because it is not part of the staggered derivative.
_COEFFICIENTS: dict[int, dict[int, tuple[float, ...]]] = {
    0: {
        2: (1.0,),
        4: (9.0 / 8.0, -1.0 / 24.0),
        6: (75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0),
        8: (1225.0 / 1024.0, -245.0 / 3072.0, 49.0 / 5120.0, -5.0 / 7168.0),
        10: (
            19845.0 / 16384.0, -735.0 / 8192.0, 567.0 / 40960.0,
            -405.0 / 229376.0, 35.0 / 294912.0,
        ),
        12: (
            160083.0 / 131072.0, -12705.0 / 131072.0,
            22869.0 / 1310720.0, -5445.0 / 1835008.0,
            847.0 / 2359296.0, -63.0 / 2883584.0,
        ),
    },
    1: {
        2: (1.0010,),
        4: (1.1382, -0.046414),
        6: (1.1965, -0.078804, 0.0081781),
        8: (1.2257, -0.099537, 0.018063, -0.0026274),
        10: (1.2415, -0.11231, 0.026191, -0.0064682, 0.001191),
        12: (1.2508, -0.12034, 0.032131, -0.010142, 0.0029857, -0.00066667),
    },
    2: {
        2: (1.0050,),
        4: (1.1534, -0.052806),
        6: (1.2111, -0.088313, 0.011768),
        8: (1.2367, -0.10815, 0.023113, -0.0046905),
        10: (1.2496, -0.11921, 0.031130, -0.0093272, 0.0025161),
        12: (1.2568, -0.12573, 0.036423, -0.013132, 0.0047484, -0.0015979),
    },
    3: {
        2: (1.0100,),
        4: (1.1640, -0.057991),
        6: (1.2192, -0.094070, 0.014608),
        8: (1.2422, -0.11269, 0.026140, -0.0064054),
        10: (1.2534, -0.12257, 0.033755, -0.011081, 0.0036784),
        12: (1.2596, -0.12825, 0.038550, -0.014763, 0.0058619, -0.0024538),
    },
    4: {
        2: (1.0300,),
        4: (1.1876, -0.072518),
        6: (1.2341, -0.10569, 0.022589),
        8: (1.2516, -0.12085, 0.032236, -0.011459),
        10: (1.2596, -0.12829, 0.038533, -0.014681, 0.0072580),
        12: (1.2640, -0.13239, 0.042217, -0.017803, 0.0081959, -0.0051848),
    },
}


@dataclass(frozen=True)
class GhostRows:
    fd_order: int
    half_order: int
    active_velocity_rows: tuple[int, ...]
    full_velocity_rows: tuple[int, ...]
    stress_rows: tuple[int, ...]


@dataclass(frozen=True)
class SurfaceState:
    vz: dict[int, float]
    syz: dict[int, float]


@dataclass(frozen=True)
class Dispersion:
    frequency_hz: float
    angle_rad: float
    wavenumber_rad_m: float
    group_velocity_m_s: float
    group_delay_s: float
    continuum_delay_s: float

    @property
    def delay_error_s(self) -> float:
        return self.group_delay_s - self.continuum_delay_s


@dataclass(frozen=True)
class SurfaceTimes:
    y0_s: float
    y_half_h_s: float
    y_h_s: float

    @property
    def half_minimum_separation_s(self) -> float:
        return 0.5 * min(abs(self.y0_s - self.y_half_h_s), abs(self.y0_s - self.y_h_s))


@dataclass(frozen=True)
class SurfaceBoundaryAcceptance:
    physical_traction: bool
    velocity_parity: bool
    stress_parity: bool
    image_closure: bool

    @property
    def all_pass(self) -> bool:
        return all((
            self.physical_traction,
            self.velocity_parity,
            self.stress_parity,
            self.image_closure,
        ))


@dataclass(frozen=True)
class ReflectionAcceptance:
    timing: bool
    surface_y0: bool
    reject_y_half_h: bool
    reject_y_h: bool
    signed_amplitude: bool
    phase: bool
    absorbing_control: bool

    @property
    def all_pass(self) -> bool:
        return all((
            self.timing,
            self.surface_y0,
            self.reject_y_half_h,
            self.reject_y_h,
            self.signed_amplitude,
            self.phase,
            self.absorbing_control,
        ))


@dataclass(frozen=True)
class ProductionAcceptance:
    healthy: bool
    reflection: ReflectionAcceptance
    boundary: SurfaceBoundaryAcceptance

    @property
    def all_pass(self) -> bool:
        return self.healthy and self.reflection.all_pass and self.boundary.all_pass


def half_order(fd_order: int) -> int:
    if fd_order not in SUPPORTED_FD_ORDERS:
        raise ValueError(f"Unsupported FDORDER {fd_order}")
    return fd_order // 2


def holberg_coefficients(fd_order: int, max_relative_error: int = 1) -> tuple[float, ...]:
    try:
        return _COEFFICIENTS[max_relative_error][fd_order]
    except KeyError as error:
        raise ValueError(
            f"Unsupported coefficient selection FDORDER={fd_order}, "
            f"MAX_RELATIVE_ERROR={max_relative_error}"
        ) from error


def required_ghost_rows(fd_order: int) -> GhostRows:
    m = half_order(fd_order)
    return GhostRows(
        fd_order=fd_order,
        half_order=m,
        active_velocity_rows=tuple(range(2 - m, 1)) if m > 1 else (),
        full_velocity_rows=tuple(range(1 - m, 1)),
        stress_rows=tuple(range(1 - m, 1)),
    )


def extend_surface_state(
    vz_positive: Mapping[int, float],
    syz_positive: Mapping[int, float],
    fd_order: int,
) -> SurfaceState:
    """Build the approved full even-vz/odd-syz row-zero surface state."""
    m = half_order(fd_order)
    missing_vz = [index for index in range(1, m + 1) if index not in vz_positive]
    missing_syz = [index for index in range(1, m) if index not in syz_positive]
    if missing_vz or missing_syz:
        raise ValueError(f"Missing interior rows: vz={missing_vz}, syz={missing_syz}")
    vz = dict(vz_positive)
    syz = dict(syz_positive)
    for k in range(1, m + 1):
        vz[1 - k] = vz_positive[k]
    syz[0] = 0.0
    for k in range(1, m):
        syz[-k] = -syz_positive[k]
    return SurfaceState(vz=vz, syz=syz)


def forward_staggered(values: Mapping[int, float], row: int, coefficients: Sequence[float]) -> float:
    return sum(
        coefficient * (values[row + m] - values[row - (m - 1)])
        for m, coefficient in enumerate(coefficients, start=1)
    )


def backward_staggered(values: Mapping[int, float], row: int, coefficients: Sequence[float]) -> float:
    return sum(
        coefficient * (values[row + m - 1] - values[row - m])
        for m, coefficient in enumerate(coefficients, start=1)
    )


def surface_contract_errors(
    state: SurfaceState,
    fd_order: int,
    *,
    max_relative_error: int = 1,
    tolerance: float = 0.0,
) -> list[str]:
    """Return implementation-independent violations of the approved parity contract."""
    m = half_order(fd_order)
    errors: list[str] = []
    for k in range(1, m + 1):
        if 1 - k not in state.vz:
            errors.append(f"missing vz row {1-k}")
        elif k not in state.vz or not math.isclose(
            state.vz[1 - k], state.vz[k], rel_tol=0.0, abs_tol=tolerance
        ):
            errors.append(f"vz parity k={k}")
    if 0 not in state.syz or state.syz[0] != 0.0:
        errors.append("syz[0] is not zero")
    for k in range(1, m):
        if -k not in state.syz:
            errors.append(f"missing syz row {-k}")
        elif k not in state.syz or not math.isclose(
            state.syz[-k], -state.syz[k], rel_tol=0.0, abs_tol=tolerance
        ):
            errors.append(f"syz parity k={k}")
    try:
        derivative = forward_staggered(
            state.vz, 0, holberg_coefficients(fd_order, max_relative_error)
        )
    except KeyError as error:
        errors.append(f"surface derivative missing row {error.args[0]}")
    else:
        bound = 64.0 * FLOAT32_EPSILON * sum(
            abs(value) for value in holberg_coefficients(fd_order, max_relative_error)
        )
        scale = max((abs(value) for value in state.vz.values()), default=1.0)
        if abs(derivative) > max(tolerance, bound * scale):
            errors.append(f"D+vz[0]={derivative}")
    return errors


def denise_grid_index(input_coordinate_m: float, dh_m: float) -> int:
    if input_coordinate_m < 0.0 or dh_m <= 0.0:
        raise ValueError("DENISE coordinates require non-negative input and DH > 0")
    return math.floor(input_coordinate_m / dh_m + 0.5)


def native_vz_position(input_xy_m: tuple[float, float], dh_m: float) -> tuple[float, float]:
    i = denise_grid_index(input_xy_m[0], dh_m)
    j = denise_grid_index(input_xy_m[1], dh_m)
    if i < 1 or j < 1:
        raise ValueError("Native SH vz positions require one-based positive indices")
    return ((i - 0.5) * dh_m, (j - 0.5) * dh_m)


def image_source(source_xy_m: tuple[float, float], surface_y_m: float = 0.0) -> tuple[float, float]:
    return (source_xy_m[0], 2.0 * surface_y_m - source_xy_m[1])


def distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def image_path_distance(
    source_xy_m: tuple[float, float],
    receiver_xy_m: tuple[float, float],
    *,
    surface_y_m: float = 0.0,
) -> float:
    return distance(image_source(source_xy_m, surface_y_m), receiver_xy_m)


def surface_candidate_times(
    source_xy_m: tuple[float, float],
    receiver_xy_m: tuple[float, float],
    *,
    dh_m: float,
    vs_m_s: float,
) -> SurfaceTimes:
    if vs_m_s <= 0.0:
        raise ValueError("Vs must be positive")
    return SurfaceTimes(*(
        image_path_distance(source_xy_m, receiver_xy_m, surface_y_m=y) / vs_m_s
        for y in (0.0, 0.5 * dh_m, dh_m)
    ))


def staggered_symbol(xi: float, coefficients: Sequence[float]) -> float:
    return 2.0 * sum(
        coefficient * math.sin((m - 0.5) * xi)
        for m, coefficient in enumerate(coefficients, start=1)
    )


def _staggered_symbol_derivative(xi: float, coefficients: Sequence[float]) -> float:
    return 2.0 * sum(
        coefficient * (m - 0.5) * math.cos((m - 0.5) * xi)
        for m, coefficient in enumerate(coefficients, start=1)
    )


def numerical_dispersion(
    *,
    distance_m: float,
    angle_rad: float,
    frequency_hz: float,
    vs_m_s: float,
    dt_s: float,
    dh_m: float,
    fd_order: int,
    max_relative_error: int = 1,
) -> Dispersion:
    if min(distance_m, frequency_hz, vs_m_s, dt_s, dh_m) <= 0.0:
        raise ValueError("Dispersion inputs must be positive")
    coefficients = holberg_coefficients(fd_order, max_relative_error)
    omega = 2.0 * math.pi * frequency_hz
    target = math.sin(0.5 * omega * dt_s) / (vs_m_s * dt_s / (2.0 * dh_m))
    cos_angle, sin_angle = math.cos(angle_rad), math.sin(angle_rad)

    def spatial_norm(kappa: float) -> float:
        kx = staggered_symbol(kappa * dh_m * cos_angle, coefficients)
        ky = staggered_symbol(kappa * dh_m * sin_angle, coefficients)
        return math.hypot(kx, ky)

    lower = 0.0
    upper = math.pi / dh_m
    if target >= spatial_norm(upper):
        raise ValueError("Frequency is outside the monotone numerical dispersion branch")
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if spatial_norm(midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint
    kappa = 0.5 * (lower + upper)
    xi_x, xi_y = kappa * dh_m * cos_angle, kappa * dh_m * sin_angle
    k_x = staggered_symbol(xi_x, coefficients)
    k_y = staggered_symbol(xi_y, coefficients)
    norm = math.hypot(k_x, k_y)
    dnorm_dk = (
        k_x * _staggered_symbol_derivative(xi_x, coefficients) * dh_m * cos_angle
        + k_y * _staggered_symbol_derivative(xi_y, coefficients) * dh_m * sin_angle
    ) / norm
    cosine_time = math.cos(0.5 * omega * dt_s)
    group_velocity = (vs_m_s / dh_m) * dnorm_dk / cosine_time
    if group_velocity <= 0.0:
        raise ValueError("Computed non-positive numerical group velocity")
    return Dispersion(
        frequency_hz=frequency_hz,
        angle_rad=angle_rad,
        wavenumber_rad_m=kappa,
        group_velocity_m_s=group_velocity,
        group_delay_s=distance_m / group_velocity,
        continuum_delay_s=distance_m / vs_m_s,
    )


def ricker_f95(source_frequency_hz: float, *, bins: int = 20000) -> float:
    """95%-energy frequency of DENISE's QUELLART=1 Ricker pulse."""
    if source_frequency_hz <= 0.0 or bins < 100:
        raise ValueError("Ricker spectrum requires positive frequency and at least 100 bins")
    peak_hz = math.sqrt(2.0) * source_frequency_hz / 1.5
    maximum_hz = 8.0 * peak_hz
    step = maximum_hz / bins
    energies = []
    for index in range(bins):
        frequency = (index + 0.5) * step
        ratio = frequency / peak_hz
        energies.append((frequency**4) * math.exp(-2.0 * ratio * ratio))
    target = 0.95 * sum(energies)
    cumulative = 0.0
    for index, energy in enumerate(energies):
        cumulative += energy
        if cumulative >= target:
            return (index + 0.5) * step
    return maximum_hz


def arrival_tolerance(
    *, dt_s: float, reference_distance_m: float, calibration_distance_m: float,
    vs_m_s: float, differential_dispersion_s: float,
) -> float:
    if dt_s <= 0.0 or vs_m_s <= 0.0:
        raise ValueError("Timing tolerance requires DT > 0 and Vs > 0")
    return (
        2.0 * dt_s
        + abs(reference_distance_m - calibration_distance_m) / vs_m_s
        + abs(differential_dispersion_s)
    )


def signed_amplitude_alpha(reflection: Sequence[float], calibration: Sequence[float]) -> float:
    if len(reflection) != len(calibration) or not reflection:
        raise ValueError("Amplitude windows must be non-empty and equally sized")
    denominator = sum(value * value for value in calibration)
    if denominator == 0.0:
        raise ValueError("Calibration window has zero energy")
    return sum(left * right for left, right in zip(reflection, calibration)) / denominator


def normalized_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Correlation windows must be non-empty and equally sized")
    norm_first = math.sqrt(sum(value * value for value in first))
    norm_second = math.sqrt(sum(value * value for value in second))
    if norm_first == 0.0 or norm_second == 0.0:
        raise ValueError("Cannot correlate a zero-energy window")
    return sum(a * b for a, b in zip(first, second)) / (norm_first * norm_second)


def relative_l2(reference: Sequence[float], candidate: Sequence[float]) -> float:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("L2 windows must be non-empty and equally sized")
    denominator = math.sqrt(sum(value * value for value in reference))
    if denominator == 0.0:
        raise ValueError("Reference window has zero energy")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(reference, candidate))) / denominator


def surface_roundoff_limits(coefficients: Sequence[float]) -> tuple[float, float]:
    """Return physical-traction and discrete-image-closure limits."""
    return (
        32.0 * FLOAT32_EPSILON,
        64.0 * FLOAT32_EPSILON * sum(abs(value) for value in coefficients),
    )


def center_shear_modulus(*, rho: float, pu: float, invmat1: int) -> float:
    """Interpret the SH center material according to DENISE INVMAT1."""
    if rho <= 0.0 or pu <= 0.0:
        raise ValueError("Center material values must be positive")
    if invmat1 == 1:
        return rho * pu * pu
    if invmat1 == 3:
        return pu
    raise ValueError(f"Unsupported SH diagnostic INVMAT1={invmat1}")


def impedance_scaled_velocity(
    *, rho: float, pu: float, vz: float, invmat1: int
) -> float:
    mu_center = center_shear_modulus(rho=rho, pu=pu, invmat1=invmat1)
    return math.sqrt(rho * mu_center) * abs(vz)


def normalized_surface_residuals(
    *,
    max_abs_syz0: float,
    max_abs_dplus_vz0: float,
    max_abs_interior_stress: float,
    max_impedance_vz: float,
    max_abs_dx_vz: float,
    max_abs_vz: float,
    f95_hz: float,
    vs_m_s: float,
) -> tuple[float, float]:
    """Normalize physical traction and discrete image closure separately."""
    if f95_hz <= 0.0 or vs_m_s <= 0.0:
        raise ValueError("Surface normalization requires positive f95 and Vs")
    stress_floor = float.fromhex("0x1.0p-126")
    stress_scale = max(max_abs_interior_stress, max_impedance_vz, stress_floor)
    frequency_gradient_scale = 2.0 * math.pi * f95_hz * max_abs_vz / vs_m_s
    gradient_scale = max(max_abs_dx_vz, frequency_gradient_scale, stress_floor)
    return max_abs_syz0 / stress_scale, max_abs_dplus_vz0 / gradient_scale


def evaluate_surface_boundary(
    *,
    normalized_physical_traction: float,
    physical_traction_limit: float,
    max_velocity_parity_residual: float,
    max_stress_parity_residual: float,
    normalized_image_closure: float,
    image_closure_limit: float,
) -> SurfaceBoundaryAcceptance:
    values = (
        normalized_physical_traction,
        physical_traction_limit,
        max_velocity_parity_residual,
        max_stress_parity_residual,
        normalized_image_closure,
        image_closure_limit,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Surface boundary metrics and limits must be finite and non-negative")
    return SurfaceBoundaryAcceptance(
        physical_traction=normalized_physical_traction <= physical_traction_limit,
        velocity_parity=max_velocity_parity_residual == 0.0,
        stress_parity=max_stress_parity_residual == 0.0,
        image_closure=normalized_image_closure <= image_closure_limit,
    )


def evaluate_reflection(
    *,
    timing_error_s: float,
    observed_propagation_s: float,
    surface_times: SurfaceTimes,
    timing_tolerance_s: float,
    signed_amplitude_alpha_value: float,
    normalized_correlation_value: float,
    absorbing_l2_ratio: float,
    signed_amplitude_error_max: float = 0.05,
    normalized_correlation_min: float = 0.99,
    absorbing_l2_ratio_max: float = 0.10,
    comparison_epsilon: float = 1.0e-12,
) -> ReflectionAcceptance:
    values = (
        timing_error_s,
        observed_propagation_s,
        timing_tolerance_s,
        signed_amplitude_alpha_value,
        normalized_correlation_value,
        absorbing_l2_ratio,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Reflection metrics must be finite")
    if min(
        timing_error_s,
        timing_tolerance_s,
        absorbing_l2_ratio,
        signed_amplitude_error_max,
        normalized_correlation_min,
        absorbing_l2_ratio_max,
        comparison_epsilon,
    ) < 0.0:
        raise ValueError("Reflection limits and unsigned metrics must be non-negative")
    residual_y0 = abs(observed_propagation_s - surface_times.y0_s)
    residual_y_half_h = abs(observed_propagation_s - surface_times.y_half_h_s)
    residual_y_h = abs(observed_propagation_s - surface_times.y_h_s)
    return ReflectionAcceptance(
        timing=timing_error_s <= timing_tolerance_s + comparison_epsilon,
        surface_y0=residual_y0 <= timing_tolerance_s + comparison_epsilon,
        reject_y_half_h=residual_y_half_h > timing_tolerance_s,
        reject_y_h=residual_y_h > timing_tolerance_s,
        signed_amplitude=(
            abs(signed_amplitude_alpha_value - 1.0) <= signed_amplitude_error_max
        ),
        phase=normalized_correlation_value >= normalized_correlation_min,
        absorbing_control=absorbing_l2_ratio <= absorbing_l2_ratio_max,
    )


def evaluate_production_acceptance(
    *,
    healthy: bool,
    reflection: ReflectionAcceptance,
    boundary: SurfaceBoundaryAcceptance,
) -> ProductionAcceptance:
    return ProductionAcceptance(
        healthy=bool(healthy), reflection=reflection, boundary=boundary
    )


def stability_modulation_limit(
    *, dt_s: float, f95_hz: float, coefficients: Sequence[float]
) -> float:
    """Return the predeclared relative leapfrog-energy allowance delta_E."""
    if dt_s <= 0.0 or f95_hz <= 0.0:
        raise ValueError("Stability allowance requires positive dt and f95")
    return (
        4.0 * (2.0 * math.pi * f95_hz * dt_s) ** 2
        + 64.0 * FLOAT32_EPSILON * sum(abs(value) for value in coefficients)
    )


def finite_nonzero(trace: Sequence[float]) -> bool:
    return bool(trace) and all(math.isfinite(value) for value in trace) and any(value != 0.0 for value in trace)


def peak_time(
    trace: Sequence[float], *, expected_s: float, half_width_s: float, dt_s: float
) -> float:
    if not finite_nonzero(trace) or dt_s <= 0.0 or half_width_s <= 0.0:
        raise ValueError("Peak picking requires a finite non-zero trace and positive window")
    first = max(0, math.ceil((expected_s - half_width_s) / dt_s) - 1)
    last = min(len(trace), math.floor((expected_s + half_width_s) / dt_s))
    if first >= last:
        raise ValueError("Peak window does not overlap the trace")
    index = max(range(first, last), key=lambda sample: abs(trace[sample]))
    if trace[index] == 0.0:
        raise ValueError("Peak window has zero energy")
    return (index + 1) * dt_s


def centered_window(trace: Sequence[float], *, center_s: float, half_width_s: float, dt_s: float) -> list[float]:
    center_index = round(center_s / dt_s) - 1
    half_width_samples = round(half_width_s / dt_s)
    first = center_index - half_width_samples
    last = center_index + half_width_samples + 1
    if first < 0 or last > len(trace) or first >= last:
        raise ValueError("Requested window is not fully contained")
    return list(trace[first:last])
