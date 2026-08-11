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


def complex_shear_modulus(
    *,
    frequency_hz: float,
    reference_shear_modulus_pa: float,
    qs_input: float,
    relaxation_frequencies_hz: Sequence[float],
) -> complex:
    """Continuous generalized-Maxwell modulus represented by DENISE SH coefficients."""
    if frequency_hz <= 0.0 or reference_shear_modulus_pa <= 0.0 or qs_input <= 0.0:
        raise ValueError("Frequency, modulus and Qs must be positive")
    if not relaxation_frequencies_hz or any(value <= 0.0 for value in relaxation_frequencies_hz):
        raise ValueError("At least one positive relaxation frequency is required")

    tau = 2.0 / qs_input
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
) -> RheologyPrediction:
    reference_modulus = density_kg_m3 * vs_m_s * vs_m_s
    modulus = complex_shear_modulus(
        frequency_hz=frequency_hz,
        reference_shear_modulus_pa=reference_modulus,
        qs_input=qs_input,
        relaxation_frequencies_hz=relaxation_frequencies_hz,
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


def hann_windowed_spectrum(
    trace: Sequence[float], *, dt_s: float, frequencies_hz: Sequence[float]
) -> tuple[complex, ...]:
    if len(trace) < 2 or dt_s <= 0.0:
        raise ValueError("A windowed spectrum requires at least two samples and dt > 0")
    if not frequencies_hz or any(value < 0.0 for value in frequencies_hz):
        raise ValueError("Spectrum frequencies must be non-empty and non-negative")
    denominator = len(trace) - 1
    windowed = [
        sample * (0.5 - 0.5 * math.cos(2.0 * math.pi * index / denominator))
        for index, sample in enumerate(trace)
    ]
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
) -> tuple[TransferSample, ...]:
    if len(viscoelastic_trace) != len(elastic_trace):
        raise ValueError("Transfer traces must have equal lengths")
    if not 0.0 < minimum_elastic_fraction < 1.0:
        raise ValueError("Elastic energy fraction must lie between zero and one")
    viscoelastic = hann_windowed_spectrum(
        viscoelastic_trace, dt_s=dt_s, frequencies_hz=frequencies_hz
    )
    elastic = hann_windowed_spectrum(elastic_trace, dt_s=dt_s, frequencies_hz=frequencies_hz)
    maximum = max(abs(value) for value in elastic)
    if maximum == 0.0:
        raise ValueError("Elastic reference spectrum is zero")
    threshold = minimum_elastic_fraction * maximum
    return tuple(
        TransferSample(frequency, visco / reference, abs(reference))
        for frequency, visco, reference in zip(frequencies_hz, viscoelastic, elastic)
        if abs(reference) >= threshold
    )


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
