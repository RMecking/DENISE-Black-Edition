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


@dataclass(frozen=True)
class CPMLReflectionMetrics:
    direct_l2: float
    late_residual_l2: float
    reflection_ratio: float
    reflection_db: float


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


def project_components(
    first: Sequence[float], second: Sequence[float], direction: tuple[float, float]
) -> tuple[list[float], list[float]]:
    """Project two particle-velocity components parallel/perpendicular to direction."""
    if len(first) != len(second):
        raise ValueError("Component traces have different lengths")
    norm = math.hypot(*direction)
    if norm == 0.0:
        raise ValueError("Projection direction must be non-zero")
    nx, ny = direction[0] / norm, direction[1] / norm
    parallel = [nx * vx + ny * vy for vx, vy in zip(first, second)]
    perpendicular = [-ny * vx + nx * vy for vx, vy in zip(first, second)]
    return parallel, perpendicular


def time_window(
    trace: Sequence[float], *, center_s: float, half_width_s: float, dt_s: float
) -> list[float]:
    if dt_s <= 0.0 or half_width_s <= 0.0:
        raise ValueError("Window timestep and half-width must be positive")
    return time_interval(
        trace,
        start_s=center_s - half_width_s,
        stop_s=center_s + half_width_s,
        dt_s=dt_s,
    )


def time_interval(
    trace: Sequence[float], *, start_s: float, stop_s: float, dt_s: float
) -> list[float]:
    """Return a fully contained interval; never clip analytical windows."""
    if dt_s <= 0.0 or start_s < 0.0 or stop_s <= start_s:
        raise ValueError("Interval requires dt > 0 and 0 <= start < stop")
    first_sample_s = dt_s
    last_sample_s = len(trace) * dt_s
    if start_s < first_sample_s - 1.0e-12 or stop_s > last_sample_s + 1.0e-12:
        raise ValueError(
            f"Requested interval [{start_s}, {stop_s}] s is not fully contained in "
            f"seismogram [{first_sample_s}, {last_sample_s}] s"
        )
    start = math.ceil(start_s / dt_s - 1.0e-12) - 1
    stop = math.floor(stop_s / dt_s + 1.0e-12)
    if start >= stop:
        raise ValueError("Time interval does not overlap the trace")
    return list(trace[start:stop])


def absolute_peak_index_in_window(
    trace: Sequence[float], *, center_s: float, half_width_s: float, dt_s: float
) -> int:
    """Return the global index of the largest absolute sample in a time window."""
    if dt_s <= 0.0 or half_width_s <= 0.0:
        raise ValueError("Peak-pick timestep and half-width must be positive")
    start = max(0, math.ceil((center_s - half_width_s) / dt_s - 1.0e-12) - 1)
    stop = min(len(trace), math.floor((center_s + half_width_s) / dt_s + 1.0e-12))
    if start >= stop:
        raise ValueError("Peak-pick window does not overlap the trace")
    local_index = max(range(stop - start), key=lambda index: abs(trace[start + index]))
    if trace[start + local_index] == 0.0:
        raise ValueError("Peak-pick window contains no signal")
    return start + local_index


def absolute_peak_index_in_interval(
    trace: Sequence[float], *, start_s: float, stop_s: float, dt_s: float
) -> int:
    """Return the global absolute-peak index in a predeclared time interval."""
    window = time_interval(trace, start_s=start_s, stop_s=stop_s, dt_s=dt_s)
    start = max(0, math.ceil(start_s / dt_s - 1.0e-12) - 1)
    local_index = max(range(len(window)), key=lambda index: abs(window[index]))
    if window[local_index] == 0.0:
        raise ValueError("Peak-pick interval contains no signal")
    return start + local_index


def cpml_reflection_metrics(
    compact: Sequence[float],
    reference: Sequence[float],
    *,
    dt_s: float,
    direct_window_s: tuple[float, float],
    reflection_window_s: tuple[float, float],
) -> CPMLReflectionMetrics:
    """Measure late compact-domain residual relative to reference direct energy."""
    if len(compact) != len(reference):
        raise ValueError("Compact and reference traces have different lengths")
    direct = time_interval(
        reference, start_s=direct_window_s[0], stop_s=direct_window_s[1], dt_s=dt_s
    )
    compact_late = time_interval(
        compact, start_s=reflection_window_s[0], stop_s=reflection_window_s[1], dt_s=dt_s
    )
    reference_late = time_interval(
        reference, start_s=reflection_window_s[0], stop_s=reflection_window_s[1], dt_s=dt_s
    )
    direct_l2 = math.sqrt(signal_energy(direct))
    late_residual_l2 = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(compact_late, reference_late))
    )
    if direct_l2 == 0.0:
        raise ValueError("CPML metric requires non-zero reference direct energy")
    ratio = late_residual_l2 / direct_l2
    reflection_db = -math.inf if ratio == 0.0 else 20.0 * math.log10(ratio)
    return CPMLReflectionMetrics(direct_l2, late_residual_l2, ratio, reflection_db)


def signal_energy(trace: Sequence[float]) -> float:
    return sum(value * value for value in trace)


def relative_amplitude_error(first: Sequence[float], second: Sequence[float]) -> float:
    first_norm = math.sqrt(signal_energy(first))
    second_norm = math.sqrt(signal_energy(second))
    scale = max(first_norm, second_norm)
    if scale == 0.0:
        raise ValueError("Amplitude comparison requires a non-zero signal")
    return abs(first_norm - second_norm) / scale


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
