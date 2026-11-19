"""Independent double-precision reference for SH-OPT-SCALE-01.

Derived solely from ``docs/sh_opt_scale_01_contract.md`` (sections 2 and 3).
No production source is consulted or imported.  Fields are plain nested lists
indexed ``[j][i]`` with 0-based ``j`` (row, y) and ``i`` (column, x); the
contract's local 1-based cell ``(i, j)`` corresponds to ``[j - 1][i - 1]``.
"""

from __future__ import annotations

import math
import struct
from typing import Mapping, Sequence

CLASSES = ("primary", "rho", "q")

Field = Sequence[Sequence[float]]


def f32(value: float) -> float:
    """Round a Python float (double) to the nearest IEEE-754 binary32 value."""
    return struct.unpack("=f", struct.pack("=f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("=I", struct.pack("=f", value))[0]


def _ordered_f32(value: float) -> int:
    """Monotone integer image of a finite binary32 value (+0 and -0 coincide)."""
    bits = f32_bits(value)
    if bits & 0x80000000:
        return -(bits & 0x7FFFFFFF)
    return bits


def ulp_distance_f32(left: float, right: float) -> int:
    """Number of binary32 representable steps between two finite values."""
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("ulp distance requires finite values")
    return abs(_ordered_f32(f32(left)) - _ordered_f32(f32(right)))


# --------------------------------------------------------------------------
# Section 3: update mask
# --------------------------------------------------------------------------

def update_mask_status(nx: int, ny: int, pos_x: int, pos_y: int,
                       nprocx: int, nprocy: int, fw: int,
                       mask_is_null: bool = False) -> int:
    """Return code of the mask builder for the given arguments (0 or -1)."""
    if mask_is_null:
        return -1
    if nx < 1 or ny < 1 or fw < 0 or nprocx < 1 or nprocy < 1:
        return -1
    if not 0 <= pos_x <= nprocx - 1:
        return -1
    if not 0 <= pos_y <= nprocy - 1:
        return -1
    return 0


def update_mask(nx: int, ny: int, pos_x: int, pos_y: int, nprocx: int,
                nprocy: int, fw: int, free_surf: int,
                boundary: int) -> list[list[float]]:
    """Local CPML update mask, ``[j-1][i-1]`` for local cell ``(i, j)``."""
    if update_mask_status(nx, ny, pos_x, pos_y, nprocx, nprocy, fw) != 0:
        raise ValueError("invalid update-mask arguments")
    mask = []
    for j in range(1, ny + 1):
        row = []
        for i in range(1, nx + 1):
            in_cpml = (
                ((not boundary) and pos_x == 0 and i <= fw)
                or ((not boundary) and pos_x == nprocx - 1 and i >= nx - fw + 1)
                or (pos_y == 0 and (not free_surf) and j <= fw)
                or (pos_y == nprocy - 1 and j >= ny - fw + 1)
            )
            row.append(0.0 if in_cpml else 1.0)
        mask.append(row)
    return mask


# --------------------------------------------------------------------------
# Section 2: class-scaled search direction (modes 1 and 2)
# --------------------------------------------------------------------------

def class_scale(mode: int, gradient: Field, base: Field, mask: Field) -> float:
    """s_k = r_k / N_k (or 0 when N_k == 0), all statistics over M == 1."""
    masked_g = []
    masked_m = []
    for g_row, m_row, mask_row in zip(gradient, base, mask):
        for g_value, m_value, mask_value in zip(g_row, m_row, mask_row):
            if mask_value == 1.0:
                masked_g.append(float(g_value))
                masked_m.append(float(m_value))
    n = len(masked_g)
    if n == 0:
        raise ValueError("global masked cell count is zero")
    if mode == 1:
        # float*float is exact in double; fsum is correctly rounded.
        norm = math.sqrt(math.fsum(value * value for value in masked_g) / n)
        reference = math.sqrt(math.fsum(value * value for value in masked_m) / n)
    elif mode == 2:
        norm = max(abs(value) for value in masked_g)
        reference = max(abs(value) for value in masked_m)
    else:
        raise ValueError("closed form exists for modes 1 and 2 only")
    return reference / norm if norm > 0.0 else 0.0


def scaled_step(mode: int, weights: Mapping[str, float],
                gradients: Mapping[str, Field], bases: Mapping[str, Field],
                mask: Field) -> dict[str, list[list[float]]]:
    """p_k(cell) = M * w_k * s_k * g_k in double, rounded to binary32."""
    step: dict[str, list[list[float]]] = {}
    for name in CLASSES:
        weight = float(weights[name])
        scale = class_scale(mode, gradients[name], bases[name], mask)
        rows = []
        for g_row, mask_row in zip(gradients[name], mask):
            row = []
            for g_value, mask_value in zip(g_row, mask_row):
                if mask_value != 1.0 or weight == 0.0:
                    row.append(0.0)
                else:
                    row.append(f32(weight * scale * float(g_value)))
            rows.append(row)
        step[name] = rows
    return step


def class_dot(gradient: Field, step: Field) -> float:
    """Double-precision sum over cells of g * p for one class."""
    return math.fsum(
        float(g_value) * float(p_value)
        for g_row, p_row in zip(gradient, step)
        for g_value, p_value in zip(g_row, p_row)
    )


def descent_dot(gradients: Mapping[str, Field], step: Mapping[str, Field]) -> float:
    """sum over classes and cells of g . p (section 2, descent property)."""
    return math.fsum(class_dot(gradients[name], step[name]) for name in CLASSES)
