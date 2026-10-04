from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class GeometryError(ValueError):
    """Raised when source or receiver geometry violates its file contract."""


@dataclass(frozen=True)
class Source:
    ordinal: int
    x_m: float
    y_m: float
    delay_s: float
    frequency_hz: float
    amplitude: float
    angle_deg: float
    source_type: int


@dataclass(frozen=True)
class Receiver:
    ordinal: int
    x_m: float
    y_m: float


def _records(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]


def read_sources(path: str | Path) -> list[Source]:
    source = Path(path)
    lines = _records(source)
    if not lines:
        raise GeometryError(f"empty source file: {source}")
    try:
        declared = int(lines[0].split()[0])
    except ValueError as error:
        raise GeometryError(f"invalid source count in {source}") from error
    result: list[Source] = []
    for ordinal, line in enumerate(lines[1:], start=1):
        fields = line.split()
        if len(fields) != 8:
            raise GeometryError(f"source {ordinal} in {source} has {len(fields)} fields, expected 8")
        values = [float(value) for value in fields]
        result.append(
            Source(ordinal, values[0], values[2], values[3], values[4], values[5], values[6], int(values[7]))
        )
    if declared != len(result):
        raise GeometryError(f"{source} declares {declared} sources but contains {len(result)}")
    return result


def read_receivers(path: str | Path) -> list[Receiver]:
    source = Path(path)
    result: list[Receiver] = []
    for ordinal, line in enumerate(_records(source), start=1):
        fields = line.split()
        if len(fields) < 2:
            raise GeometryError(f"receiver {ordinal} in {source} has fewer than two fields")
        result.append(Receiver(ordinal, float(fields[0]), float(fields[1])))
    if not result:
        raise GeometryError(f"empty receiver file: {source}")
    return result


def geometry_issues(
    sources: list[Source], receivers: list[Receiver], *, nx: int, ny: int, dh_m: float,
    absorbing_width: int, free_surface: bool, require_grid_alignment: bool = True,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    xmax, ymax = nx * dh_m, ny * dh_m
    side_min, side_max = absorbing_width * dh_m, (nx - absorbing_width) * dh_m
    bottom_max = (ny - absorbing_width) * dh_m
    all_points = [("source", p.ordinal, p.x_m, p.y_m) for p in sources]
    all_points += [("receiver", p.ordinal, p.x_m, p.y_m) for p in receivers]
    seen: dict[tuple[str, float, float], int] = {}
    shallow_counts = {"source": 0, "receiver": 0}
    for kind, ordinal, x_m, y_m in all_points:
        if not (0.0 < x_m <= xmax and 0.0 < y_m <= ymax):
            errors.append(f"{kind} {ordinal} at ({x_m},{y_m}) is outside (0,{xmax}]x(0,{ymax}]")
        if x_m <= side_min or x_m > side_max:
            errors.append(f"{kind} {ordinal} at x={x_m} m lies in a side absorbing boundary")
        if y_m > bottom_max:
            errors.append(f"{kind} {ordinal} at y={y_m} m lies in the bottom absorbing boundary")
        if require_grid_alignment and (
            not np.isclose(x_m / dh_m, round(x_m / dh_m))
            or not np.isclose(y_m / dh_m, round(y_m / dh_m))
        ):
            errors.append(f"{kind} {ordinal} at ({x_m},{y_m}) is not aligned to the {dh_m} m grid")
        key = (kind, x_m, y_m)
        if key in seen:
            errors.append(f"duplicate {kind} coordinates at ordinals {seen[key]} and {ordinal}: ({x_m},{y_m})")
        seen[key] = ordinal
        if free_surface and y_m <= absorbing_width * dh_m:
            shallow_counts[kind] += 1
    for kind, count in shallow_counts.items():
        if count:
            warnings.append(
                f"{count} {kind}(s) lie within {absorbing_width * dh_m} m of the free surface; the top boundary is FREE_SURF, not CPML"
            )
    return errors, warnings
