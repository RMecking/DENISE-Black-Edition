"""Independent reference for SH-MASK-01 (docs/sh_mask_01_contract.md,
sections 3 and 4), written from the contract text, not from the C code.

Everything is formulated on the global grid: the mask files are parsed as
global float32 fields (x outer, y inner), the taper is evaluated for every
global cell, and a subdomain result is the cut-out of the global answer.
Global cell indices are 1-based as in the contract (cell (I, J) lies at
x = I * DH, y = J * DH).  Pure Python, double precision; values that the
program stores as float32 are rounded with ``f32``."""
from __future__ import annotations

import math
import struct
from pathlib import Path

OK, BAD_ARGUMENTS, UNREADABLE, WRONG_SIZE, BAD_VALUE, FILES_DIFFER = 0, -1, -2, -3, -4, -5


def f32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


def read_global_mask(tfile: str, nxg: int, nyg: int) -> tuple[int, list[list[float]] | None]:
    """Contract section 3: status and the global field G[I-1][J-1]."""
    fields = []
    for suffix in ("_u.bin", "_rho.bin"):
        path = Path(tfile + suffix)
        if not path.is_file():
            return UNREADABLE, None
        raw = path.read_bytes()
        if len(raw) != 4 * nxg * nyg:
            return WRONG_SIZE, None
        fields.append(struct.unpack(f"={nxg * nyg}f", raw))
    for values in fields:
        if any(not (v == 0.0 or v == 1.0) for v in values):
            return BAD_VALUE, None
    if fields[0] != fields[1]:
        return FILES_DIFFER, None
    u = fields[0]
    return OK, [[u[x * nyg + y] for y in range(nyg)] for x in range(nxg)]


def apply_global_mask(cpml: list[list[float]], grid: list[list[float]]) -> list[list[float]]:
    """Logical AND of the CPML mask (indexed [J-1][I-1]) with the file mask
    (indexed [I-1][J-1]): file value 0 fixes the cell, otherwise the CPML
    value is kept."""
    return [[0.0 if grid[x][y] == 0.0 else cpml[y][x] for x in range(len(cpml[0]))]
            for y in range(len(cpml))]


def source_cell(position: float, dh: float) -> int:
    """iround(x_s / DH) of the forward injection: the float32 quotient,
    rounded half up."""
    return math.floor(f32(f32(position) / f32(dh)) + 0.5)


def taper_value(cell_x: int, cell_y: int, sources: list[tuple[float, float]], dh: float,
                radius: float, filtsize: int) -> float:
    """Contract section 4: T = min over sources of T_s, rounded to float32."""
    dh_d, radius_d = float(f32(dh)), float(f32(radius))
    taper = 1.0
    for sx, sy in sources:
        di = cell_x - source_cell(sx, dh)
        dj = cell_y - source_cell(sy, dh)
        if abs(di) <= filtsize and abs(dj) <= filtsize:
            value = 0.0
        else:
            r = dh_d * math.hypot(di, dj)
            value = 1.0 if r >= radius_d else math.sin(math.pi * r / (2.0 * radius_d)) ** 2
        taper = min(taper, value)
    return f32(taper)


def global_taper(nxg: int, nyg: int, sources, dh, radius, filtsize) -> list[list[float]]:
    """T on the global grid, indexed [J-1][I-1]."""
    return [[taper_value(x, y, sources, dh, radius, filtsize) for x in range(1, nxg + 1)]
            for y in range(1, nyg + 1)]


def taper_arguments_valid(sources, dh, radius, filtsize) -> bool:
    """The rejections of the source taper (contract section 5, unit level)."""
    values = [dh, radius] + [v for s in sources for v in s]
    return (bool(sources) and all(math.isfinite(v) for v in values) and f32(dh) > 0.0
            and f32(radius) >= f32(dh) and filtsize >= 0)


def metric_product(taper: float, metric: float) -> float:
    """D = T * P, product in double, stored as float32 (contract section 4)."""
    return f32(float(taper) * float(metric))
