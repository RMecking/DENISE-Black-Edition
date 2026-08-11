from __future__ import annotations

import math
from typing import Literal, Sequence


Field = Literal["material", "sxx", "syy", "vx", "vy", "sxy"]


_FIELD_OFFSETS: dict[Field, tuple[float, float]] = {
    "material": (-0.5, -0.5),
    "sxx": (-0.5, -0.5),
    "syy": (-0.5, -0.5),
    "vx": (0.0, -0.5),
    "vy": (-0.5, 0.0),
    "sxy": (0.0, 0.0),
}


def denise_grid_index(input_coordinate_m: float, dh_m: float) -> int:
    """Reproduce DENISE's positive-coordinate nearest-gridpoint conversion."""
    if dh_m <= 0.0 or input_coordinate_m < 0.0:
        raise ValueError("DENISE coordinates require DH > 0 and a non-negative input")
    return math.floor(input_coordinate_m / dh_m + 0.5)


def field_position(i: int, j: int, dh_m: float, field: Field) -> tuple[float, float]:
    """Return the physical position of a DENISE field sample at indices (i, j)."""
    if dh_m <= 0.0 or i < 1 or j < 1:
        raise ValueError("DENISE field indices are one-based and DH must be positive")
    offset_x, offset_y = _FIELD_OFFSETS[field]
    return ((i + offset_x) * dh_m, (j + offset_y) * dh_m)


def input_field_position(
    input_coordinates_m: tuple[float, float], dh_m: float, field: Field
) -> tuple[float, float]:
    """Round DENISE input coordinates, then locate the selected staggered field."""
    i = denise_grid_index(input_coordinates_m[0], dh_m)
    j = denise_grid_index(input_coordinates_m[1], dh_m)
    return field_position(i, j, dh_m, field)


def input_coordinate_for_field_position(
    physical_coordinate_m: float, dh_m: float, *, axis: Literal["x", "y"], field: Field
) -> float:
    """Return the DENISE input coordinate representing an exact field location."""
    if dh_m <= 0.0 or physical_coordinate_m < 0.0:
        raise ValueError("Field coordinates require DH > 0 and a non-negative position")
    offset = _FIELD_OFFSETS[field][0 if axis == "x" else 1]
    index = physical_coordinate_m / dh_m - offset
    rounded_index = round(index)
    if not math.isclose(index, rounded_index, abs_tol=1.0e-10):
        raise ValueError(
            f"{physical_coordinate_m} m is not representable by {field} on the {axis} axis"
        )
    if rounded_index < 1:
        raise ValueError("DENISE field indices are one-based")
    return float(rounded_index) * dh_m


def sxy_collocation_stencil(
    central_input_m: tuple[float, float], dh_m: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Receivers needed to collocate vx and vy at the central sxy location.

    The returned order is central, one receiver at +DH in y, and one at +DH in x.
    """
    x_m, y_m = central_input_m
    return ((x_m, y_m), (x_m, y_m + dh_m), (x_m + dh_m, y_m))


def collocate_velocity_at_sxy(
    vx_central: Sequence[float],
    vx_y_plus: Sequence[float],
    vy_central: Sequence[float],
    vy_x_plus: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Linearly average native velocity samples onto one sxy gridpoint."""
    lengths = {len(vx_central), len(vx_y_plus), len(vy_central), len(vy_x_plus)}
    if len(lengths) != 1:
        raise ValueError("Collocation traces must have equal lengths")
    vx = [0.5 * (lower + upper) for lower, upper in zip(vx_central, vx_y_plus)]
    vy = [0.5 * (left + right) for left, right in zip(vy_central, vy_x_plus)]
    return vx, vy
