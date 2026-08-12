from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class QualityFactorBandStatistics:
    minimum_q: float
    maximum_q: float
    mean_q: float
    rms_deviation: float
    relative_rms_deviation: float
    minimum_frequency_hz: float
    maximum_frequency_hz: float


def linear_frequency_samples(*, fmin_hz: float, fmax_hz: float, df_hz: float) -> tuple[float, ...]:
    """Reproduce MATLAB's inclusive ``fmin:df:fmax`` sampling deterministically."""
    if fmin_hz <= 0.0 or fmax_hz < fmin_hz or df_hz <= 0.0:
        raise ValueError("Frequency sampling requires 0 < fmin <= fmax and df > 0")
    count = int(math.floor((fmax_hz - fmin_hz) / df_hz + 1.0e-12)) + 1
    return tuple(fmin_hz + index * df_hz for index in range(count))


def target_q_to_tau(
    *,
    target_q: float,
    relaxation_frequencies_hz: Sequence[float],
    fmin_hz: float,
    fmax_hz: float,
    df_hz: float,
) -> float:
    """Least-squares fixed-FL target-Q to GSLS relaxation-strength mapping."""
    if target_q <= 0.0:
        raise ValueError("Target Q must be positive")
    if not relaxation_frequencies_hz or any(value <= 0.0 for value in relaxation_frequencies_hz):
        raise ValueError("At least one positive relaxation frequency is required")

    numerator = 0.0
    denominator = 0.0
    for frequency_hz in linear_frequency_samples(
        fmin_hz=fmin_hz, fmax_hz=fmax_hz, df_hz=df_hz
    ):
        omega = 2.0 * math.pi * frequency_hz
        a_sum = 0.0
        b_sum = 0.0
        for relaxation_frequency_hz in relaxation_frequencies_hz:
            theta = 1.0 / (2.0 * math.pi * relaxation_frequency_hz)
            omega_theta = omega * theta
            divisor = 1.0 + omega_theta * omega_theta
            a_sum += omega_theta * omega_theta / divisor
            b_sum += omega_theta / divisor
        a = 1.0 / b_sum
        b = a_sum / b_sum
        numerator += a * (target_q - b)
        denominator += a * a

    inverse_tau = numerator / denominator
    if not math.isfinite(inverse_tau) or inverse_tau <= 0.0:
        raise ValueError("Target Q and rheology band do not yield a positive tau")
    return 1.0 / inverse_tau


def numerical_target_q_to_tau(
    *,
    target_q: float,
    relaxation_frequencies_hz: Sequence[float],
    fmin_hz: float,
    fmax_hz: float,
    df_hz: float,
) -> float:
    """Independent golden-section minimization of the original qstd residual."""
    frequencies = linear_frequency_samples(fmin_hz=fmin_hz, fmax_hz=fmax_hz, df_hz=df_hz)

    def objective(log_tau: float) -> float:
        tau = math.exp(log_tau)
        return sum(
            (
                qstd_quality_factor(
                    frequency_hz=frequency,
                    relaxation_frequencies_hz=relaxation_frequencies_hz,
                    tau=tau,
                )
                - target_q
            )
            ** 2
            for frequency in frequencies
        )

    left = math.log(1.0e-6)
    right = math.log(10.0)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)
    for _ in range(160):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = objective(x2)
    return math.exp(0.5 * (left + right))


def qstd_quality_factor(
    *,
    frequency_hz: float,
    relaxation_frequencies_hz: Sequence[float],
    tau: float,
) -> float:
    """Reproduce the generalized-SLS Q expression in the recovered qstd.m."""
    if frequency_hz <= 0.0 or tau <= 0.0:
        raise ValueError("Frequency and tau must be positive")
    if not relaxation_frequencies_hz or any(
        frequency <= 0.0 for frequency in relaxation_frequencies_hz
    ):
        raise ValueError("At least one positive relaxation frequency is required")
    omega = 2.0 * math.pi * frequency_hz
    numerator_sum = 0.0
    denominator_sum = 0.0
    for relaxation_frequency in relaxation_frequencies_hz:
        theta = 1.0 / (2.0 * math.pi * relaxation_frequency)
        denominator = 1.0 + omega * omega * theta * theta
        numerator_sum += omega * omega * theta * theta * tau / denominator
        denominator_sum += omega * theta * tau / denominator
    return (1.0 + numerator_sum) / denominator_sum


def quality_factor_band_statistics(
    *,
    frequencies_hz: Sequence[float],
    quality_factors: Sequence[float],
    target_q: float,
) -> QualityFactorBandStatistics:
    if len(frequencies_hz) != len(quality_factors) or not frequencies_hz:
        raise ValueError("Band statistics require equally sized non-empty inputs")
    if target_q <= 0.0:
        raise ValueError("Target Q must be positive")
    minimum_index = min(range(len(quality_factors)), key=quality_factors.__getitem__)
    maximum_index = max(range(len(quality_factors)), key=quality_factors.__getitem__)
    mean_q = sum(quality_factors) / len(quality_factors)
    rms_deviation = math.sqrt(
        sum((quality_factor - target_q) ** 2 for quality_factor in quality_factors)
        / len(quality_factors)
    )
    return QualityFactorBandStatistics(
        minimum_q=quality_factors[minimum_index],
        maximum_q=quality_factors[maximum_index],
        mean_q=mean_q,
        rms_deviation=rms_deviation,
        relative_rms_deviation=rms_deviation / target_q,
        minimum_frequency_hz=frequencies_hz[minimum_index],
        maximum_frequency_hz=frequencies_hz[maximum_index],
    )
