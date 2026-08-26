from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from tests.utilities.seismogram import time_interval


@dataclass(frozen=True)
class WaveformMetrics:
    relative_l2: float
    normalized_correlation: float
    signed_amplitude_ratio: float
    arrival_lag_samples: int
    arrival_lag_s: float


@dataclass(frozen=True)
class FrozenWaveformAcceptance:
    relative_l2_max: float = 0.02
    normalized_correlation_min: float = 0.999
    signed_amplitude_error_max: float = 0.03
    arrival_lag_max_s: float = 0.001


FROZEN_WAVEFORM_ACCEPTANCE = FrozenWaveformAcceptance()
REFERENCE_TRANSLATION_RELATIVE_L2_MAX = 5.0e-5
REFERENCE_TRANSLATION_CORRELATION_MIN = 0.999999
REFERENCE_TRANSLATION_LAG_MAX_S = 0.0005
SUPERPOSITION_RELATIVE_L2_MAX = 2.0e-6
FINITE_Q_SENSITIVITY_RELATIVE_L2_MIN = 1.0e-3
HIGH_Q_ENDPOINT_RELATIVE_L2_MAX = 0.05
HIGH_Q_ENDPOINT_CORRELATION_MIN = 0.999
FREE_SURFACE_ZERO_CONTROL_L2_RATIO_MAX = 0.10
MPI_RELATIVE_L2_MAX = 1.0e-6
MPI_CORRELATION_MIN = 0.999999
STABILITY_Q4_TO_Q1_MAX = 0.01
HARD_BOUNDARY_KEYS = (
    "traction_residual",
    "dplus_vz_residual",
    "vz_parity_residual",
    "total_syz_parity_residual",
    "q_surface_residual",
)
DIAGNOSTIC_BOUNDARY_KEYS = ("q_parity_residual",)
FROZEN_BOUNDARY_LIMITS = {
    "traction_residual_max": 5.0e-6,
    "dplus_vz_residual_max": 5.0e-5,
    "vz_parity_residual_max": 2.0e-6,
    "total_syz_parity_residual_max": 2.0e-6,
    "q_surface_residual_max": 2.0e-6,
}

TOLERANCE_RATIONALE = {
    "frozen_before_candidate_execution": True,
    "waveform": (
        "The candidate and reference use identical DH, DT, FDORDER, source, "
        "receiver component, and rheology. The 0.001 s lag is two temporal "
        "samples. The 2% L2, 0.999 correlation, and 3% signed-amplitude bounds "
        "are deliberately wider than the separately measured translation and "
        "linear-superposition floors, while remaining small relative to a "
        "missing high-order image closure."
    ),
    "translation": (
        "Integer-cell translation preserves the discrete phase until outer-boundary "
        "returns. The measured full-window reference/reference floor was 1.2432e-5 "
        "L2 with correlation 0.99999999992 and zero lag; the frozen 5e-5/0.999999/"
        "one-sample limits give about fourfold L2 headroom without using surface-"
        "candidate data."
    ),
    "high_q": (
        "The Q=1000 endpoint is required to be within 5% L2 and 0.999 "
        "correlation of the elastic control. The pre-candidate complete direct-plus-"
        "image calibration measured 3.9546% L2; the longer two-path interval is not "
        "assigned the earlier short direct-window 2.5% bound."
    ),
}


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def normalized_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Waveforms have different lengths")
    denominator = _norm(first) * _norm(second)
    if denominator == 0.0:
        raise ValueError("Cannot correlate zero-norm waveform")
    return sum(left * right for left, right in zip(first, second)) / denominator


def relative_l2(reference: Sequence[float], candidate: Sequence[float]) -> float:
    if len(reference) != len(candidate):
        raise ValueError("Waveforms have different lengths")
    denominator = _norm(reference)
    if denominator == 0.0:
        raise ValueError("Reference waveform has zero norm")
    return _norm([left - right for left, right in zip(reference, candidate)]) / denominator


def signed_amplitude_ratio(reference: Sequence[float], candidate: Sequence[float]) -> float:
    denominator = sum(value * value for value in reference)
    if denominator == 0.0:
        raise ValueError("Reference waveform has zero norm")
    return sum(left * right for left, right in zip(reference, candidate)) / denominator


def best_lag_samples(
    reference: Sequence[float], candidate: Sequence[float], *, maximum: int = 8
) -> int:
    if len(reference) != len(candidate):
        raise ValueError("Waveforms have different lengths")
    scored = []
    for lag in range(-maximum, maximum + 1):
        if lag < 0:
            left, right = reference[-lag:], candidate[:lag]
        elif lag > 0:
            left, right = reference[:-lag], candidate[lag:]
        else:
            left, right = reference, candidate
        scored.append((normalized_correlation(left, right), lag))
    return max(scored)[1]


def waveform_metrics(
    reference: Sequence[float], candidate: Sequence[float], *, dt_s: float
) -> WaveformMetrics:
    lag = best_lag_samples(reference, candidate)
    return WaveformMetrics(
        relative_l2=relative_l2(reference, candidate),
        normalized_correlation=normalized_correlation(reference, candidate),
        signed_amplitude_ratio=signed_amplitude_ratio(reference, candidate),
        arrival_lag_samples=lag,
        arrival_lag_s=lag * dt_s,
    )


def window(trace: Sequence[float], interval_s: Sequence[float], dt_s: float) -> list[float]:
    return time_interval(
        trace, start_s=float(interval_s[0]), stop_s=float(interval_s[1]), dt_s=dt_s
    )


def accepted(metrics: WaveformMetrics) -> bool:
    limits = FROZEN_WAVEFORM_ACCEPTANCE
    return (
        metrics.relative_l2 <= limits.relative_l2_max
        and metrics.normalized_correlation >= limits.normalized_correlation_min
        and abs(metrics.signed_amplitude_ratio - 1.0)
        <= limits.signed_amplitude_error_max
        and abs(metrics.arrival_lag_s) <= limits.arrival_lag_max_s
    )


def acceptance_metadata() -> dict[str, object]:
    return {
        "waveform": asdict(FROZEN_WAVEFORM_ACCEPTANCE),
        "reference_translation_relative_l2_max": REFERENCE_TRANSLATION_RELATIVE_L2_MAX,
        "reference_translation_correlation_min": REFERENCE_TRANSLATION_CORRELATION_MIN,
        "reference_translation_lag_max_s": REFERENCE_TRANSLATION_LAG_MAX_S,
        "superposition_relative_l2_max": SUPERPOSITION_RELATIVE_L2_MAX,
        "finite_q_sensitivity_relative_l2_min": FINITE_Q_SENSITIVITY_RELATIVE_L2_MIN,
        "high_q_endpoint_relative_l2_max": HIGH_Q_ENDPOINT_RELATIVE_L2_MAX,
        "high_q_endpoint_correlation_min": HIGH_Q_ENDPOINT_CORRELATION_MIN,
        "free_surface_zero_image_window_l2_ratio_max": FREE_SURFACE_ZERO_CONTROL_L2_RATIO_MAX,
        "mpi_relative_l2_max": MPI_RELATIVE_L2_MAX,
        "mpi_correlation_min": MPI_CORRELATION_MIN,
        "stability": {
            "metric": "fixed post-source quarter max_abs_vz Q4/Q1",
            "q4_to_q1_max": STABILITY_Q4_TO_Q1_MAX,
            "source_off": {
                "quellart": 1,
                "n_order": 0,
                "n_off": 1257,
            },
            "calibration_role": "finite-Q FD12 FREE_SURF=0 absorbing reference",
            "reference_calibration_q4_to_q1": 0.0003110815438951515,
        },
        "boundary": {
            "hard_keys": list(HARD_BOUNDARY_KEYS),
            "hard_limits": FROZEN_BOUNDARY_LIMITS,
            "diagnostic_only": {
                key: {"acceptance_effect": "none"}
                for key in DIAGNOSTIC_BOUNDARY_KEYS
            },
        },
        "rationale": TOLERANCE_RATIONALE,
    }


def constitutive_surface_state_accepts(
    *, total_syz0: float, q_surface: Sequence[float], tolerance: float
) -> bool:
    """Require both total traction and every constitutive memory state to close."""
    return abs(total_syz0) <= tolerance and all(
        abs(value) <= tolerance for value in q_surface
    )
