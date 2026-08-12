from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LinearFit:
    slope: float
    intercept: float
    residuals: tuple[float, ...]
    r_squared: float

    @property
    def maximum_absolute_residual(self) -> float:
        return max(abs(value) for value in self.residuals)


@dataclass(frozen=True)
class TransferSample:
    frequency_hz: float
    value: complex
    elastic_amplitude: float

    @property
    def log_amplitude(self) -> float:
        return math.log(abs(self.value))

    @property
    def phase_rad(self) -> float:
        return cmath.phase(self.value)


@dataclass(frozen=True)
class RheologyPrediction:
    complex_modulus_pa: complex
    effective_q: float
    wavenumber_per_m: complex
    log_amplitude_slope_per_m: float
    phase_slope_rad_per_m: float


HOLBERG_FD8_01_PERCENT = (1.2257, -0.099537, 0.018063, -0.0026274)


def complex_shear_modulus(
    *,
    frequency_hz: float,
    reference_shear_modulus_pa: float,
    qs_input: float,
    relaxation_frequencies_hz: Sequence[float],
    tau_override: float | None = None,
) -> complex:
    """Continuous generalized-Maxwell modulus represented by DENISE SH coefficients."""
    if frequency_hz <= 0.0 or reference_shear_modulus_pa <= 0.0 or qs_input <= 0.0:
        raise ValueError("Frequency, modulus and Qs must be positive")
    if not relaxation_frequencies_hz or any(value <= 0.0 for value in relaxation_frequencies_hz):
        raise ValueError("At least one positive relaxation frequency is required")

    tau = 2.0 / qs_input if tau_override is None else tau_override
    if tau <= 0.0:
        raise ValueError("GSLS tau must be positive")
    theta = [1.0 / (2.0 * math.pi * value) for value in relaxation_frequencies_hz]
    omega_reference = 2.0 * math.pi * relaxation_frequencies_hz[0]
    reference_sum = sum(
        (omega_reference * value) ** 2 / (1.0 + (omega_reference * value) ** 2)
        for value in theta
    )
    relaxed_modulus = reference_shear_modulus_pa / (1.0 + tau * reference_sum)
    omega = 2.0 * math.pi * frequency_hz
    mechanisms = sum(
        (1.0j * omega * value) / (1.0 + 1.0j * omega * value) for value in theta
    )
    return relaxed_modulus * (1.0 + tau * mechanisms)


def rheology_prediction(
    *,
    frequency_hz: float,
    vs_m_s: float,
    density_kg_m3: float,
    qs_input: float,
    relaxation_frequencies_hz: Sequence[float],
    tau_override: float | None = None,
) -> RheologyPrediction:
    reference_modulus = density_kg_m3 * vs_m_s * vs_m_s
    modulus = complex_shear_modulus(
        frequency_hz=frequency_hz,
        reference_shear_modulus_pa=reference_modulus,
        qs_input=qs_input,
        relaxation_frequencies_hz=relaxation_frequencies_hz,
        tau_override=tau_override,
    )
    effective_q = modulus.real / modulus.imag
    omega = 2.0 * math.pi * frequency_hz
    wavenumber = omega * cmath.sqrt(density_kg_m3 / modulus)
    elastic_wavenumber = omega / vs_m_s
    return RheologyPrediction(
        complex_modulus_pa=modulus,
        effective_q=effective_q,
        wavenumber_per_m=wavenumber,
        log_amplitude_slope_per_m=wavenumber.imag,
        phase_slope_rad_per_m=-(wavenumber.real - elastic_wavenumber),
    )


def discrete_rheology_prediction(
    *,
    frequency_hz: float,
    vs_m_s: float,
    density_kg_m3: float,
    qs_input: float,
    relaxation_frequencies_hz: Sequence[float],
    dt_s: float,
    dh_m: float,
    fd_coefficients: Sequence[float] = HOLBERG_FD8_01_PERCENT,
    tau_override: float | None = None,
) -> RheologyPrediction:
    """Predict the 1-D staggered-grid dispersion of DENISE's SH update.

    The memory response is the exact harmonic response of the trapezoidal
    recurrence used by ``update_s_visc_PML_SH.c``.  The complex wavenumber is
    then obtained from the actual staggered FD symbol.
    """
    if (
        frequency_hz <= 0.0
        or vs_m_s <= 0.0
        or density_kg_m3 <= 0.0
        or qs_input <= 0.0
        or dt_s <= 0.0
        or dh_m <= 0.0
        or not fd_coefficients
    ):
        raise ValueError("Frequency, material values, DT, DH and FD coefficients must be positive")
    reference_modulus = density_kg_m3 * vs_m_s * vs_m_s
    tau = 2.0 / qs_input if tau_override is None else tau_override
    if tau <= 0.0:
        raise ValueError("GSLS tau must be positive")
    theta = [1.0 / (2.0 * math.pi * value) for value in relaxation_frequencies_hz]
    if not theta:
        raise ValueError("At least one relaxation frequency is required")
    omega_reference = 2.0 * math.pi * relaxation_frequencies_hz[0]
    reference_sum = sum(
        (omega_reference * value) ** 2 / (1.0 + (omega_reference * value) ** 2)
        for value in theta
    )
    relaxed_modulus = reference_modulus / (1.0 + tau * reference_sum)
    omega = 2.0 * math.pi * frequency_hz
    z_value = cmath.exp(1.0j * omega * dt_s)
    memory_sum = 0.0j
    for value in theta:
        eta = dt_s / value
        b_value = 1.0 / (1.0 + 0.5 * eta)
        c_value = 1.0 - 0.5 * eta
        memory_sum += eta * tau * b_value / (z_value - b_value * c_value)
    modulus = relaxed_modulus * (
        1.0 + len(theta) * tau - 0.5 * (1.0 + z_value) * memory_sum
    )
    effective_q = modulus.real / modulus.imag
    temporal_symbol = 2.0 * math.sin(0.5 * omega * dt_s) / dt_s

    def solve_wavenumber(target: complex) -> complex:
        value = target
        for _ in range(30):
            symbol = (2.0 / dh_m) * sum(
                coefficient * cmath.sin((index - 0.5) * value * dh_m)
                for index, coefficient in enumerate(fd_coefficients, start=1)
            )
            derivative = 2.0 * sum(
                coefficient
                * (index - 0.5)
                * cmath.cos((index - 0.5) * value * dh_m)
                for index, coefficient in enumerate(fd_coefficients, start=1)
            )
            correction = (symbol - target) / derivative
            value -= correction
            if abs(correction) <= 1.0e-14:
                break
        return value

    wavenumber = solve_wavenumber(temporal_symbol * cmath.sqrt(density_kg_m3 / modulus))
    elastic_wavenumber = solve_wavenumber(temporal_symbol / vs_m_s)
    return RheologyPrediction(
        complex_modulus_pa=modulus,
        effective_q=effective_q,
        wavenumber_per_m=wavenumber,
        log_amplitude_slope_per_m=wavenumber.imag,
        phase_slope_rad_per_m=-(wavenumber.real - elastic_wavenumber.real),
    )


def effective_q_from_transfer_slopes(
    *,
    frequency_hz: float,
    vs_m_s: float,
    density_kg_m3: float,
    log_amplitude_slope_per_m: float,
    phase_slope_rad_per_m: float,
) -> float:
    """Recover complex-modulus Q from viscoelastic/elastic distance slopes."""
    if frequency_hz <= 0.0 or vs_m_s <= 0.0 or density_kg_m3 <= 0.0:
        raise ValueError("Frequency, velocity and density must be positive")
    omega = 2.0 * math.pi * frequency_hz
    wavenumber = complex(
        omega / vs_m_s - phase_slope_rad_per_m,
        log_amplitude_slope_per_m,
    )
    modulus = density_kg_m3 * (omega / wavenumber) ** 2
    if modulus.imag <= 0.0:
        raise ValueError("Observed slopes do not imply a passive positive-loss modulus")
    return modulus.real / modulus.imag


def window_weights(length: int, *, kind: str, tukey_alpha: float = 0.2) -> tuple[float, ...]:
    if length < 2:
        raise ValueError("A spectral window requires at least two samples")
    if kind == "hann":
        return tuple(
            0.5 - 0.5 * math.cos(2.0 * math.pi * index / (length - 1))
            for index in range(length)
        )
    if kind != "tukey":
        raise ValueError(f"Unsupported spectral window: {kind}")
    if not 0.0 <= tukey_alpha <= 1.0:
        raise ValueError("Tukey alpha must lie between zero and one")
    if tukey_alpha == 0.0:
        return (1.0,) * length
    if tukey_alpha == 1.0:
        return window_weights(length, kind="hann")
    edge = 0.5 * tukey_alpha * (length - 1)
    result = []
    for index in range(length):
        distance = min(index, length - 1 - index)
        if distance < edge:
            result.append(0.5 * (1.0 - math.cos(math.pi * distance / edge)))
        else:
            result.append(1.0)
    return tuple(result)


def approximate_main_lobe_width_hz(
    *, duration_s: float, kind: str, tukey_alpha: float = 0.2
) -> float:
    """Approximate zero-to-zero main-lobe width of the selected time gate."""
    if duration_s <= 0.0:
        raise ValueError("Window duration must be positive")
    if kind == "hann":
        return 4.0 / duration_s
    if kind == "tukey" and 0.0 <= tukey_alpha <= 1.0:
        return 2.0 / (duration_s * (1.0 - 0.5 * tukey_alpha))
    raise ValueError("Unsupported spectral window")


def windowed_spectrum(
    trace: Sequence[float],
    *,
    dt_s: float,
    frequencies_hz: Sequence[float],
    window_kind: str = "hann",
    tukey_alpha: float = 0.2,
) -> tuple[complex, ...]:
    if len(trace) < 2 or dt_s <= 0.0:
        raise ValueError("A windowed spectrum requires at least two samples and dt > 0")
    if not frequencies_hz or any(value < 0.0 for value in frequencies_hz):
        raise ValueError("Spectrum frequencies must be non-empty and non-negative")
    weights = window_weights(len(trace), kind=window_kind, tukey_alpha=tukey_alpha)
    windowed = [sample * weight for sample, weight in zip(trace, weights)]
    return tuple(
        sum(
            sample * cmath.exp(-2.0j * math.pi * frequency * dt_s * index)
            for index, sample in enumerate(windowed)
        )
        for frequency in frequencies_hz
    )


def transfer_spectrum(
    viscoelastic_trace: Sequence[float],
    elastic_trace: Sequence[float],
    *,
    dt_s: float,
    frequencies_hz: Sequence[float],
    minimum_elastic_fraction: float = 0.05,
    window_kind: str = "hann",
    tukey_alpha: float = 0.2,
) -> tuple[TransferSample, ...]:
    if len(viscoelastic_trace) != len(elastic_trace):
        raise ValueError("Transfer traces must have equal lengths")
    if not 0.0 < minimum_elastic_fraction < 1.0:
        raise ValueError("Elastic energy fraction must lie between zero and one")
    viscoelastic = windowed_spectrum(
        viscoelastic_trace,
        dt_s=dt_s,
        frequencies_hz=frequencies_hz,
        window_kind=window_kind,
        tukey_alpha=tukey_alpha,
    )
    elastic = windowed_spectrum(
        elastic_trace,
        dt_s=dt_s,
        frequencies_hz=frequencies_hz,
        window_kind=window_kind,
        tukey_alpha=tukey_alpha,
    )
    maximum = max(abs(value) for value in elastic)
    if maximum == 0.0:
        raise ValueError("Elastic reference spectrum is zero")
    threshold = minimum_elastic_fraction * maximum
    return tuple(
        TransferSample(frequency, visco / reference, abs(reference))
        for frequency, visco, reference in zip(frequencies_hz, viscoelastic, elastic)
        if abs(reference) >= threshold
    )


def _fft(values: Sequence[complex], *, inverse: bool = False) -> list[complex]:
    length = len(values)
    if length == 0 or length & (length - 1):
        raise ValueError("FFT length must be a positive power of two")
    result = [complex(value) for value in values]
    target = 0
    for source in range(1, length):
        bit = length >> 1
        while target & bit:
            target ^= bit
            bit >>= 1
        target ^= bit
        if source < target:
            result[source], result[target] = result[target], result[source]
    size = 2
    sign = 1.0 if inverse else -1.0
    while size <= length:
        root = cmath.exp(sign * 2.0j * math.pi / size)
        for start in range(0, length, size):
            factor = 1.0 + 0.0j
            half = size // 2
            for index in range(start, start + half):
                even = result[index]
                odd = factor * result[index + half]
                result[index] = even + odd
                result[index + half] = even - odd
                factor *= root
        size *= 2
    if inverse:
        return [value / length for value in result]
    return result


def synthetic_rheology_pair(
    *,
    dt_s: float,
    time_s: float,
    pulse_center_s: float,
    source_frequency_hz: float,
    distance_m: float,
    vs_m_s: float,
    density_kg_m3: float,
    qs_input: float,
    relaxation_frequencies_hz: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Return elastic and analytically filtered broadband Ricker traces."""
    sample_count = round(time_s / dt_s)
    elastic = []
    for index in range(sample_count):
        time = (index + 1) * dt_s
        argument = math.pi * source_frequency_hz * (time - pulse_center_s)
        squared = argument * argument
        elastic.append((1.0 - 2.0 * squared) * math.exp(-squared))
    fft_length = 1
    while fft_length < 2 * sample_count:
        fft_length *= 2
    spectrum = _fft(elastic + [0.0] * (fft_length - sample_count))
    for index in range(fft_length):
        signed_frequency = (
            index / (fft_length * dt_s)
            if index <= fft_length // 2
            else -(fft_length - index) / (fft_length * dt_s)
        )
        if signed_frequency == 0.0:
            transfer = 1.0 + 0.0j
        else:
            prediction = rheology_prediction(
                frequency_hz=abs(signed_frequency),
                vs_m_s=vs_m_s,
                density_kg_m3=density_kg_m3,
                qs_input=qs_input,
                relaxation_frequencies_hz=relaxation_frequencies_hz,
            )
            transfer = cmath.exp(
                complex(
                    prediction.log_amplitude_slope_per_m * distance_m,
                    prediction.phase_slope_rad_per_m * distance_m,
                )
            )
            if signed_frequency < 0.0:
                transfer = transfer.conjugate()
            elif index == fft_length // 2:
                transfer = complex(abs(transfer), 0.0)
        spectrum[index] *= transfer
    viscoelastic = _fft(spectrum, inverse=True)
    return elastic, [value.real for value in viscoelastic[:sample_count]]


def unwrap_phase(phases_rad: Sequence[float]) -> tuple[float, ...]:
    if not phases_rad:
        raise ValueError("Phase unwrapping requires at least one value")
    result = [float(phases_rad[0])]
    for value in phases_rad[1:]:
        adjusted = float(value)
        while adjusted - result[-1] > math.pi:
            adjusted -= 2.0 * math.pi
        while adjusted - result[-1] < -math.pi:
            adjusted += 2.0 * math.pi
        result.append(adjusted)
    return tuple(result)


def linear_fit(x_values: Sequence[float], y_values: Sequence[float]) -> LinearFit:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Linear fitting requires equally sized inputs with at least two values")
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    denominator = sum((value - mean_x) ** 2 for value in x_values)
    if denominator == 0.0:
        raise ValueError("Linear fitting requires distinct x values")
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    intercept = mean_y - slope * mean_x
    residuals = tuple(
        y_value - (intercept + slope * x_value)
        for x_value, y_value in zip(x_values, y_values)
    )
    total = sum((value - mean_y) ** 2 for value in y_values)
    residual = sum(value * value for value in residuals)
    r_squared = 1.0 if total == 0.0 and residual == 0.0 else 1.0 - residual / total
    return LinearFit(slope, intercept, residuals, r_squared)
