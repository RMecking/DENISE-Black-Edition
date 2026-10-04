from __future__ import annotations


def pixel_coordinate(x_m: float, y_m: float, dh_m: float) -> tuple[float, float]:
    """Nominal DENISE sample centers (one-based grid -> zero-based display)."""
    if dh_m <= 0:
        raise ValueError("DH must be positive")
    return x_m / dh_m - 1.0, y_m / dh_m - 1.0


def coordinate_contract(nx: int, ny: int, dh_m: float) -> dict:
    if nx < 1 or ny < 1 or dh_m <= 0:
        raise ValueError("positive dimensions and DH required")
    return {
        "nx": nx, "ny": ny, "dh_m": dh_m,
        "shape_y_x": [ny, nx], "first_sample_xy_m": [dh_m, dh_m],
        "last_sample_xy_m": [nx * dh_m, ny * dh_m],
        "imshow_origin": "upper",
        "imshow_extent_m": [dh_m / 2, (nx + .5) * dh_m, (ny + .5) * dh_m, dh_m / 2],
        "numerical_mapping": "iround(coordinate/DH), one-based; display coordinate/DH-1",
        "orientation": "[depth_y,x], row 0 shallow; no numerical flip",
        "staggering_note": "nominal scalar model samples; velocity fields remain staggered",
    }
