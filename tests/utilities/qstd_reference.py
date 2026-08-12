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
