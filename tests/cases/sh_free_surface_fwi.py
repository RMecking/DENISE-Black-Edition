from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

from tests.cases.sh_fwi_gradient import SHFWIGradientConfig


def surface_fwi_config(**changes: object) -> SHFWIGradientConfig:
    """Return the frozen shallow, surface-coupled M6.1e configuration."""
    config = replace(
        SHFWIGradientConfig(),
        time_s=0.60,
        source_y_m=200.0,
        receiver_y_m=200.0,
        anomaly_x_m=520.0,
        anomaly_y_m=210.0,
        anomaly_sigma_m=60.0,
    )
    return replace(config, **changes)


def set_surface_case(
    directory: Path,
    *,
    free_surface: bool,
    role: str,
    nprocx: int | None = None,
    nprocy: int | None = None,
) -> None:
    """Strictly set FREE_SURF and optional decomposition on a generated case."""
    parameter_path = directory / "denise.inp"
    lines = parameter_path.read_text(encoding="ascii").splitlines(keepends=True)

    surface_indices = [
        index
        for index, line in enumerate(lines)
        if line.rstrip("\r\n") in ("FREE_SURF =0", "FREE_SURF =1")
    ]
    if len(surface_indices) != 1:
        raise AssertionError(
            f"Expected exactly one FREE_SURF record, found {len(surface_indices)}"
        )
    index = surface_indices[0]
    ending = "\r\n" if lines[index].endswith("\r\n") else "\n"
    lines[index] = f"FREE_SURF ={int(free_surface)}{ending}"

    for key, value in (("NPROCX", nprocx), ("NPROCY", nprocy)):
        if value is None:
            continue
        matches = [
            line_index
            for line_index, line in enumerate(lines)
            if line.rstrip("\r\n").startswith(f"{key} =")
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"Expected exactly one {key} record, found {len(matches)}"
            )
        line_index = matches[0]
        line_ending = "\r\n" if lines[line_index].endswith("\r\n") else "\n"
        lines[line_index] = f"{key} ={value}{line_ending}"

    parameter_path.write_text("".join(lines), encoding="ascii", newline="")
    assert sum(
        line.rstrip("\r\n") == f"FREE_SURF ={int(free_surface)}"
        for line in lines
    ) == 1

    metadata_path = directory / "case.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "role": role,
            "milestone": "M6.1e",
            "free_surface": int(free_surface),
        }
    )
    if nprocx is not None:
        metadata["nprocx"] = nprocx
    if nprocy is not None:
        metadata["nprocy"] = nprocy
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def surface_reflection_timing(config: SHFWIGradientConfig) -> dict[str, float]:
    """Image-source timing at native vz grid locations for the far receiver."""
    def native_vz_coordinate(value: float) -> float:
        if value <= 0.0:
            raise ValueError("M6.1e uses DENISE's positive-coordinate mapping")
        index = math.floor(value / config.dh_m + 0.5)
        return (index - 0.5) * config.dh_m

    source_x = native_vz_coordinate(config.source_x_m)
    source_y = native_vz_coordinate(config.source_y_m)
    receiver_x = max(
        config.receiver_x_m,
        key=lambda value: abs(value - config.source_x_m),
    )
    receiver_x = native_vz_coordinate(receiver_x)
    receiver_y = native_vz_coordinate(config.receiver_y_m)
    distance = math.hypot(receiver_x - source_x, receiver_y + source_y)
    travel = distance / config.vs_m_s
    source_peak = 1.5 / config.source_frequency_hz
    arrival = travel + source_peak
    return {
        "source_native_x_m": source_x,
        "source_native_y_m": source_y,
        "receiver_native_x_m": receiver_x,
        "receiver_native_y_m": receiver_y,
        "image_path_distance_m": distance,
        "continuum_travel_time_s": travel,
        "source_peak_delay_s": source_peak,
        "predicted_reflection_peak_s": arrival,
        "recording_time_s": config.time_s,
        "post_arrival_margin_s": config.time_s - arrival,
    }


def symbolic_surface_operators(fd_order: int) -> tuple[dict, dict, int]:
    """Return exact active-half-space D+ and D- coefficient maps.

    Every matrix entry maps the symbolic coefficient index ``m`` to its
    integer multiplicity.  The returned core depth excludes only an unrelated
    artificial bottom truncation.
    """
    if fd_order not in (2, 4, 6, 8, 10, 12):
        raise ValueError("Unsupported symbolic SH finite-difference order")
    half_order = fd_order // 2
    depth = 6 * half_order + 6
    core_depth = depth - half_order

    def add(matrix: dict, row: int, column: int, coefficient: int, value: int):
        if row < 1 or column < 1 or row > depth or column > depth:
            return
        entry = matrix.setdefault((row, column), {})
        entry[coefficient] = entry.get(coefficient, 0) + value
        if entry[coefficient] == 0:
            del entry[coefficient]

    def velocity_image(index: int) -> tuple[int, int]:
        return (index, 1) if index >= 1 else (1 - index, 1)

    def stress_image(index: int) -> tuple[int | None, int]:
        if index >= 1:
            return index, 1
        if index == 0:
            return None, 0
        return -index, -1

    d_plus: dict[tuple[int, int], dict[int, int]] = {}
    for stress_row in range(1, depth + 1):
        for coefficient in range(1, half_order + 1):
            for raw_velocity, sign in (
                (stress_row + coefficient, 1),
                (stress_row - (coefficient - 1), -1),
            ):
                velocity_column, parity = velocity_image(raw_velocity)
                add(
                    d_plus,
                    stress_row,
                    velocity_column,
                    coefficient,
                    sign * parity,
                )

    d_minus: dict[tuple[int, int], dict[int, int]] = {}
    for velocity_row in range(1, depth + 1):
        for coefficient in range(1, half_order + 1):
            for raw_stress, sign in (
                (velocity_row + (coefficient - 1), 1),
                (velocity_row - coefficient, -1),
            ):
                stress_column, parity = stress_image(raw_stress)
                if stress_column is not None:
                    add(
                        d_minus,
                        velocity_row,
                        stress_column,
                        coefficient,
                        sign * parity,
                    )
    return d_plus, d_minus, core_depth
