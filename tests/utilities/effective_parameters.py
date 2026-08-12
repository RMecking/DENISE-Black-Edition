from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?"


@dataclass(frozen=True)
class EffectiveDeniseParameters:
    mode: int
    physics: int
    relaxation_mechanisms: int
    relaxation_frequencies_hz: tuple[float, ...]
    q_parameterization_mode: int = 0
    q_approx_fmin_hz: float | None = None
    q_approx_fmax_hz: float | None = None
    q_approx_df_hz: float | None = None


def parse_effective_parameters(output: str) -> EffectiveDeniseParameters:
    """Parse parameters echoed by DENISE itself, not the requested input file."""

    def required_integer(pattern: str, label: str) -> int:
        match = re.search(pattern, output, flags=re.MULTILINE)
        if match is None:
            raise ValueError(f"DENISE output does not report effective {label}")
        return int(match.group(1))

    mode = required_integer(r"^\s*MODE\s*=\s*(\d+)\s*:", "MODE")
    physics = required_integer(r"^\s*PHYSICS\s*=\s*(\d+)\s*:", "PHYSICS")
    mechanisms = required_integer(
        r"Number of relaxation mechanisms \(L\):\s*(\d+)", "L"
    )
    frequency_block = re.search(
        r"The L relaxation frequencies are at:\s*(.*?)\s*Hz",
        output,
        flags=re.DOTALL,
    )
    if frequency_block is None:
        raise ValueError("DENISE output does not report effective FL values")
    frequencies = tuple(float(value) for value in re.findall(_FLOAT, frequency_block.group(1)))
    if len(frequencies) != mechanisms:
        raise ValueError(
            f"DENISE reports L={mechanisms} but echoes {len(frequencies)} FL values"
        )
    q_mode_match = re.search(r"Q parameterization mode:\s*(\d+)", output)
    q_mode = int(q_mode_match.group(1)) if q_mode_match is not None else 0

    def optional_float(label: str) -> float | None:
        match = re.search(rf"{label}:\s*({_FLOAT})", output)
        return float(match.group(1)) if match is not None else None

    return EffectiveDeniseParameters(
        mode,
        physics,
        mechanisms,
        frequencies,
        q_mode,
        optional_float("Q approximation fmin"),
        optional_float("Q approximation fmax"),
        optional_float("Q approximation df"),
    )


def read_effective_parameters(path: Path) -> EffectiveDeniseParameters:
    return parse_effective_parameters(path.read_text(encoding="utf-8", errors="replace"))


def require_effective_parameters(
    actual: EffectiveDeniseParameters,
    *,
    mode: int,
    physics: int,
    relaxation_frequencies_hz: Sequence[float],
    absolute_frequency_tolerance_hz: float = 5.0e-7,
    q_parameterization_mode: int = 0,
    q_approx_fmin_hz: float | None = None,
    q_approx_fmax_hz: float | None = None,
    q_approx_df_hz: float | None = None,
) -> None:
    expected_frequencies = tuple(float(value) for value in relaxation_frequencies_hz)
    assert actual.mode == mode, f"effective MODE={actual.mode}, expected {mode}"
    assert actual.physics == physics, f"effective PHYSICS={actual.physics}, expected {physics}"
    assert actual.relaxation_mechanisms == len(expected_frequencies), (
        f"effective L={actual.relaxation_mechanisms}, expected {len(expected_frequencies)}"
    )
    assert len(actual.relaxation_frequencies_hz) == len(expected_frequencies)
    for index, (observed, expected) in enumerate(
        zip(actual.relaxation_frequencies_hz, expected_frequencies), start=1
    ):
        assert math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=absolute_frequency_tolerance_hz
        ), f"effective FL[{index}]={observed}, expected {expected}"
    assert actual.q_parameterization_mode == q_parameterization_mode
    for label, observed, expected in (
        ("Q_APPROX_FMIN", actual.q_approx_fmin_hz, q_approx_fmin_hz),
        ("Q_APPROX_FMAX", actual.q_approx_fmax_hz, q_approx_fmax_hz),
        ("Q_APPROX_DF", actual.q_approx_df_hz, q_approx_df_hz),
    ):
        if expected is None:
            assert observed is None, f"effective {label}={observed}, expected not active"
        else:
            assert observed is not None and math.isclose(observed, expected, abs_tol=5.0e-7), (
                f"effective {label}={observed}, expected {expected}"
            )
