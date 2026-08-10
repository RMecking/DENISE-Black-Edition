from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class VelocityFit:
    velocity_m_s: float
    intercept_s: float
    residuals_s: list[float]

    @property
    def maximum_absolute_residual_s(self) -> float:
        return max(abs(value) for value in self.residuals_s)


def read_ascii_seismograms(path: Path, receiver_count: int, samples_per_trace: int) -> list[list[float]]:
    """Read DENISE SEIS_FORMAT=2 output, which stores traces consecutively."""
    values = [float(token) for token in path.read_text(encoding="ascii").split()]
    expected = receiver_count * samples_per_trace
    if len(values) != expected:
        raise ValueError(f"Expected {expected} samples in {path}, found {len(values)}")
    return [
        values[index * samples_per_trace : (index + 1) * samples_per_trace]
        for index in range(receiver_count)
    ]


def all_finite(traces: Iterable[Iterable[float]]) -> bool:
    return all(math.isfinite(sample) for trace in traces for sample in trace)


def _smoothed_absolute(trace: Sequence[float], half_width: int) -> list[float]:
    absolute = [abs(value) for value in trace]
    prefix = [0.0]
    for value in absolute:
        prefix.append(prefix[-1] + value)
    smoothed: list[float] = []
    for index in range(len(absolute)):
        left = max(0, index - half_width)
        right = min(len(absolute), index + half_width + 1)
        smoothed.append((prefix[right] - prefix[left]) / (right - left))
    return smoothed


def first_break_index(
    trace: Sequence[float],
    *,
    smoothing_samples: int,
    threshold_fraction: float = 0.05,
) -> int:
    """Pick the first sustained arrival at a fraction of the smoothed peak."""
    if not trace:
        raise ValueError("Cannot pick an empty trace")
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("threshold_fraction must lie between zero and one")
    envelope = _smoothed_absolute(trace, max(0, smoothing_samples // 2))
    peak = max(envelope)
    if peak <= 0.0 or not math.isfinite(peak):
        raise ValueError("Trace has no finite non-zero signal")
    threshold = threshold_fraction * peak
    return next(index for index, value in enumerate(envelope) if value >= threshold)


def ricker_wavelet(samples: int, dt: float, frequency_hz: float) -> list[float]:
    """Reproduce DENISE QUELLART=1, including its 1.5/f source delay."""
    period = 1.0 / frequency_hz
    result = []
    for index in range(samples):
        time_s = (index + 1) * dt
        tau = math.pi * (time_s - 1.5 * period) / (1.5 * period)
        result.append((1.0 - 4.0 * tau * tau) * math.exp(-2.0 * tau * tau))
    return result


def source_pick_delay(
    *,
    samples: int,
    dt: float,
    frequency_hz: float,
    smoothing_samples: int,
    threshold_fraction: float = 0.05,
) -> float:
    wavelet = ricker_wavelet(samples, dt, frequency_hz)
    index = first_break_index(
        wavelet,
        smoothing_samples=smoothing_samples,
        threshold_fraction=threshold_fraction,
    )
    return (index + 1) * dt


def fit_propagation_velocity(offsets_m: Sequence[float], pick_times_s: Sequence[float]) -> VelocityFit:
    """Fit pick_time = intercept + offset / velocity by ordinary least squares."""
    if len(offsets_m) != len(pick_times_s) or len(offsets_m) < 2:
        raise ValueError("Velocity fitting requires equally sized inputs with at least two picks")
    mean_offset = sum(offsets_m) / len(offsets_m)
    mean_time = sum(pick_times_s) / len(pick_times_s)
    denominator = sum((offset - mean_offset) ** 2 for offset in offsets_m)
    if denominator == 0.0:
        raise ValueError("Velocity fitting requires at least two distinct offsets")
    slope = sum(
        (offset - mean_offset) * (pick - mean_time)
        for offset, pick in zip(offsets_m, pick_times_s)
    ) / denominator
    if slope <= 0.0:
        raise ValueError("Fitted travel-time slope must be positive")
    intercept = mean_time - slope * mean_offset
    residuals = [
        pick - (intercept + slope * offset)
        for offset, pick in zip(offsets_m, pick_times_s)
    ]
    return VelocityFit(velocity_m_s=1.0 / slope, intercept_s=intercept, residuals_s=residuals)


def relative_l2(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float:
    if len(first) != len(second) or any(len(a) != len(b) for a, b in zip(first, second)):
        raise ValueError("Seismogram arrays have different shapes")
    pairs = [(a, b) for trace_a, trace_b in zip(first, second) for a, b in zip(trace_a, trace_b)]
    numerator = math.sqrt(sum((a - b) ** 2 for a, b in pairs))
    denominator = math.sqrt(sum(a * a for a, _ in pairs))
    if denominator == 0.0:
        raise ValueError("Reference seismograms have zero L2 norm")
    return numerator / denominator


def normalized_correlation(first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]) -> float:
    if len(first) != len(second) or any(len(a) != len(b) for a, b in zip(first, second)):
        raise ValueError("Seismogram arrays have different shapes")
    pairs = [(a, b) for trace_a, trace_b in zip(first, second) for a, b in zip(trace_a, trace_b)]
    dot = sum(a * b for a, b in pairs)
    norm_a = math.sqrt(sum(a * a for a, _ in pairs))
    norm_b = math.sqrt(sum(b * b for _, b in pairs))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot correlate zero-norm seismograms")
    return dot / (norm_a * norm_b)
