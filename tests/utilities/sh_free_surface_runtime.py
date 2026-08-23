"""Independent runtime support for the M6.1 elastic SH surface oracle."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass


DENISE_PI = 3.141592653589793


def binary32(value: float) -> float:
    """Round to IEEE-754 binary32, round-to-nearest, ties-to-even."""
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


@dataclass(frozen=True)
class RickerReference:
    samples: tuple[float, ...]
    peak_timestep: int
    n_off: int


@dataclass(frozen=True)
class PostSourceQuarters:
    n_off: int
    nt: int
    quarter_size: int
    inclusive_bounds: tuple[tuple[int, int], ...]


def denise_ricker_reference(
    *,
    nt: int,
    dt_s: float,
    frequency_hz: float,
    amplitude: float = 1.0,
    timeshift_s: float = 0.0,
    quellart: int = 1,
    n_order: int = 0,
) -> RickerReference:
    """Return the explicitly quantized stored DENISE QUELLART=1 signal."""
    if quellart != 1:
        raise ValueError("M6.1 source-off is defined only for QUELLART=1")
    if n_order != 0:
        raise ValueError("M6.1 source-off is defined only for N_ORDER=0")
    if nt < 1 or dt_s <= 0.0 or frequency_hz <= 0.0:
        raise ValueError("Ricker reference requires positive NT, DT, and frequency")

    dt = binary32(dt_s)
    fc = binary32(frequency_hz)
    tshift = binary32(timeshift_s)
    scale = binary32(amplitude)
    ts = binary32(1.0 / fc)
    denominator = 1.5 * ts
    samples: list[float] = []
    for nt_index in range(1, nt + 1):
        time_s = binary32(nt_index * dt)
        tau = binary32(
            DENISE_PI * (time_s - 1.5 * ts - tshift) / denominator
        )
        tau_squared = float(tau) * float(tau)
        amp = binary32(
            (1.0 - 4.0 * tau_squared) * math.exp(-2.0 * tau_squared)
        )
        samples.append(binary32(float(amp) * float(scale)))

    peak_offset = max(range(nt), key=lambda index: abs(samples[index]))
    nonzero_after_peak = [
        index + 1
        for index in range(peak_offset, nt)
        if samples[index] != 0.0
    ]
    if not nonzero_after_peak:
        raise ValueError("Quantized Ricker sequence has no nonzero post-peak sample")
    return RickerReference(
        samples=tuple(samples),
        peak_timestep=peak_offset + 1,
        n_off=max(nonzero_after_peak),
    )


def post_source_quarters(*, nt: int, n_off: int) -> PostSourceQuarters:
    """Partition every post-source sample into four inclusive quarters."""
    n_post = nt - n_off
    if n_post < 4:
        raise ValueError("Post-source interval must contain at least four samples")
    if n_post % 4:
        raise ValueError("Post-source sample count must be divisible by four")
    quarter_size = n_post // 4
    bounds = tuple(
        (
            n_off + quarter * quarter_size + 1,
            n_off + (quarter + 1) * quarter_size,
        )
        for quarter in range(4)
    )
    return PostSourceQuarters(
        n_off=n_off,
        nt=nt,
        quarter_size=quarter_size,
        inclusive_bounds=bounds,
    )
