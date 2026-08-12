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
    return EffectiveDeniseParameters(mode, physics, mechanisms, frequencies)


def read_effective_parameters(path: Path) -> EffectiveDeniseParameters:
    return parse_effective_parameters(path.read_text(encoding="utf-8", errors="replace"))


def require_effective_parameters(
    actual: EffectiveDeniseParameters,
    *,
    mode: int,
    physics: int,
    relaxation_frequencies_hz: Sequence[float],
    absolute_frequency_tolerance_hz: float = 5.0e-7,
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
