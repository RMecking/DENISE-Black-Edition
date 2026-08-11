from __future__ import annotations

import cmath
import math
from typing import Sequence

from tests.utilities.seismogram import signal_energy


def root_mean_square(trace: Sequence[float]) -> float:
    if not trace:
        raise ValueError("RMS requires at least one sample")
    return math.sqrt(signal_energy(trace) / len(trace))


def peak_absolute(trace: Sequence[float]) -> float:
    if not trace:
        raise ValueError("Peak amplitude requires at least one sample")
    return max(abs(value) for value in trace)


def spectral_band_rms(
    trace: Sequence[float], *, dt_s: float, lower_hz: float, upper_hz: float
) -> float:
    """Return RMS of unnormalized DFT amplitudes in the inclusive frequency band."""
    if not trace or dt_s <= 0.0 or not 0.0 <= lower_hz <= upper_hz:
        raise ValueError("Spectral RMS requires samples, dt > 0 and a valid frequency band")
    frequency_step = 1.0 / (len(trace) * dt_s)
    first_bin = max(0, math.ceil(lower_hz / frequency_step))
    last_bin = min(len(trace) // 2, math.floor(upper_hz / frequency_step))
    if first_bin > last_bin:
        raise ValueError("The requested band contains no DFT bins")
    amplitudes = []
    for bin_index in range(first_bin, last_bin + 1):
        phase_step = -2.0j * math.pi * bin_index / len(trace)
        value = sum(sample * cmath.exp(phase_step * index) for index, sample in enumerate(trace))
        amplitudes.append(abs(value))
    return math.sqrt(sum(value * value for value in amplitudes) / len(amplitudes))
