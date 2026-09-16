"""Independent field-map reference for M6.3c-4 MPI/free-surface VJPs."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
from pathlib import Path
import struct


SUPPORTED_FDORDERS = (2, 4, 6, 8, 10, 12)


def f32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", float(value)))[0]


@dataclass(frozen=True)
class Layout:
    nx: int
    ny: int
    fdorder: int

    @property
    def half(self) -> int:
        return self.fdorder // 2

    @property
    def vertical_depth(self) -> int:
        return self.half + 1

    @property
    def row_min(self) -> int:
        return 1 - self.vertical_depth

    @property
    def row_max(self) -> int:
        return self.ny + self.vertical_depth

    @property
    def col_min(self) -> int:
        return 1 - self.half

    @property
    def col_max(self) -> int:
        return self.nx + self.half

    @property
    def rows(self) -> int:
        return self.row_max - self.row_min + 1

    @property
    def cols(self) -> int:
        return self.col_max - self.col_min + 1

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    def index(self, j: int, i: int) -> int:
        assert self.row_min <= j <= self.row_max
        assert self.col_min <= i <= self.col_max
        return (j - self.row_min) * self.cols + (i - self.col_min)


def topology(nproc_x: int, nproc_y: int):
    ranks = nproc_x * nproc_y
    result = []
    for rank in range(ranks):
        x = rank % nproc_x
        y = rank // nproc_x
        result.append(
            {
                "x": x,
                "y": y,
                "left": rank - 1 if x else rank + nproc_x - 1,
                "right": rank + 1 if x != nproc_x - 1 else rank - nproc_x + 1,
                "upper": rank - nproc_x if y else ranks + rank - nproc_x,
                "lower": rank + nproc_x if y != nproc_y - 1 else rank + nproc_x - ranks,
            }
        )
    return result


def deterministic_field(layout: Layout, rank: int, field: int, dual: bool):
    values = []
    shift = 0.19 if dual else -0.07
    for j in range(layout.row_min, layout.row_max + 1):
        for i in range(layout.col_min, layout.col_max + 1):
            raw = (
                0.013 * (rank + 1)
                + 0.007 * (field + 1)
                + 0.0011 * j
                - 0.00073 * i
                + shift
            )
            values.append(f32(math.sin(raw * 7.0) + 0.17 * math.cos(raw * 3.0)))
    return values


def _copy_rank_fields(fields):
    return [[list(field) for field in rank_fields] for rank_fields in fields]


def _add(values, index, contribution, round_to_float):
    value = values[index] + contribution
    values[index] = f32(value) if round_to_float else value


def exchange_forward(fields, layout, nproc_x, nproc_y, boundary, kind):
    """Apply the current production halo-copy field map."""
    result = _copy_rank_fields(fields)
    topo = topology(nproc_x, nproc_y)
    vertical_field = 0 if kind == "velocity" else 1
    horizontal_field = 0
    for rank, neighbors in enumerate(topo):
        if neighbors["y"] < nproc_y - 1:
            lower = neighbors["lower"]
            for i in range(1, layout.nx + 1):
                for layer in range(1, layout.vertical_depth + 1):
                    result[rank][vertical_field][layout.index(layout.ny + layer, i)] = (
                        fields[lower][vertical_field][layout.index(layer, i)]
                    )
        if neighbors["y"] > 0:
            upper = neighbors["upper"]
            for i in range(1, layout.nx + 1):
                for layer in range(1, layout.vertical_depth + 1):
                    result[rank][vertical_field][layout.index(1 - layer, i)] = (
                        fields[upper][vertical_field][layout.index(layout.ny - layer + 1, i)]
                    )
        left_active = bool(boundary or neighbors["x"] > 0)
        right_active = bool(boundary or neighbors["x"] < nproc_x - 1)
        if right_active:
            right = neighbors["right"]
            for j in range(1, layout.ny + 1):
                for layer in range(1, layout.half + 1):
                    result[rank][horizontal_field][layout.index(j, layout.nx + layer)] = (
                        fields[right][horizontal_field][layout.index(j, layer)]
                    )
        if left_active:
            left = neighbors["left"]
            for j in range(1, layout.ny + 1):
                for layer in range(1, layout.half + 1):
                    result[rank][horizontal_field][layout.index(j, 1 - layer)] = (
                        fields[left][horizontal_field][layout.index(j, layout.nx - layer + 1)]
                    )
    return result


def exchange_transpose(
    bars, layout, nproc_x, nproc_y, boundary, kind, *, round_to_float=False
):
    """Exact transpose of :func:`exchange_forward`."""
    result = _copy_rank_fields(bars)
    topo = topology(nproc_x, nproc_y)
    vertical_field = 0 if kind == "velocity" else 1
    horizontal_field = 0
    for rank, neighbors in enumerate(topo):
        if neighbors["y"] < nproc_y - 1:
            lower = neighbors["lower"]
            for i in range(1, layout.nx + 1):
                for layer in range(1, layout.vertical_depth + 1):
                    destination = layout.index(layout.ny + layer, i)
                    source = layout.index(layer, i)
                    _add(
                        result[lower][vertical_field], source,
                        bars[rank][vertical_field][destination], round_to_float,
                    )
                    result[rank][vertical_field][destination] = 0.0
        if neighbors["y"] > 0:
            upper = neighbors["upper"]
            for i in range(1, layout.nx + 1):
                for layer in range(1, layout.vertical_depth + 1):
                    destination = layout.index(1 - layer, i)
                    source = layout.index(layout.ny - layer + 1, i)
                    _add(
                        result[upper][vertical_field], source,
                        bars[rank][vertical_field][destination], round_to_float,
                    )
                    result[rank][vertical_field][destination] = 0.0
        left_active = bool(boundary or neighbors["x"] > 0)
        right_active = bool(boundary or neighbors["x"] < nproc_x - 1)
        if right_active:
            right = neighbors["right"]
            for j in range(1, layout.ny + 1):
                for layer in range(1, layout.half + 1):
                    destination = layout.index(j, layout.nx + layer)
                    source = layout.index(j, layer)
                    _add(
                        result[right][horizontal_field], source,
                        bars[rank][horizontal_field][destination], round_to_float,
                    )
                    result[rank][horizontal_field][destination] = 0.0
        if left_active:
            left = neighbors["left"]
            for j in range(1, layout.ny + 1):
                for layer in range(1, layout.half + 1):
                    destination = layout.index(j, 1 - layer)
                    source = layout.index(j, layout.nx - layer + 1)
                    _add(
                        result[left][horizontal_field], source,
                        bars[rank][horizontal_field][destination], round_to_float,
                    )
                    result[rank][horizontal_field][destination] = 0.0
    return result


def surface_velocity_forward(fields, layout, nproc_x, free_surface=True):
    result = _copy_rank_fields(fields)
    if not free_surface:
        return result
    for rank in range(len(fields)):
        if rank // nproc_x != 0:
            continue
        for i in range(1, layout.nx + 1):
            for layer in range(1, layout.half + 1):
                result[rank][0][layout.index(1-layer, i)] = fields[rank][0][layout.index(layer, i)]
    return result


def surface_velocity_transpose(fields, layout, nproc_x, *, round_to_float=False):
    result = _copy_rank_fields(fields)
    for rank in range(len(fields)):
        if rank // nproc_x != 0:
            continue
        for i in range(1, layout.nx + 1):
            for layer in range(1, layout.half + 1):
                ghost = layout.index(1-layer, i)
                physical = layout.index(layer, i)
                _add(result[rank][0], physical, fields[rank][0][ghost], round_to_float)
                result[rank][0][ghost] = 0.0
    return result


def surface_stress_forward(fields, layout, nproc_x, free_surface=True):
    result = _copy_rank_fields(fields)
    if not free_surface:
        return result
    for rank in range(len(fields)):
        if rank // nproc_x != 0:
            continue
        for i in range(1, layout.nx + 1):
            result[rank][1][layout.index(0, i)] = 0.0
            for layer in range(1, layout.half):
                result[rank][1][layout.index(-layer, i)] = -fields[rank][1][layout.index(layer, i)]
    return result


def surface_stress_transpose(fields, layout, nproc_x, *, round_to_float=False):
    result = _copy_rank_fields(fields)
    for rank in range(len(fields)):
        if rank // nproc_x != 0:
            continue
        for i in range(1, layout.nx + 1):
            result[rank][1][layout.index(0, i)] = 0.0
            for layer in range(1, layout.half):
                ghost = layout.index(-layer, i)
                physical = layout.index(layer, i)
                _add(result[rank][1], physical, -fields[rank][1][ghost], round_to_float)
                result[rank][1][ghost] = 0.0
    return result


def build_case(operation, layout, nproc_x, nproc_y, boundary, *, round_to_float):
    field_count = 1 if operation.startswith("v_") else 2
    ranks = nproc_x * nproc_y
    inputs = [
        [deterministic_field(layout, rank, field, False) for field in range(field_count)]
        for rank in range(ranks)
    ]
    bars = [
        [deterministic_field(layout, rank, field, True) for field in range(field_count)]
        for rank in range(ranks)
    ]
    kind = "velocity" if field_count == 1 else "stress"
    if operation.endswith("exchange"):
        forward = exchange_forward(inputs, layout, nproc_x, nproc_y, boundary, kind)
        transpose = exchange_transpose(
            bars, layout, nproc_x, nproc_y, boundary, kind,
            round_to_float=round_to_float,
        )
    elif operation == "v_surface":
        forward = surface_velocity_forward(inputs, layout, nproc_x)
        transpose = surface_velocity_transpose(
            bars, layout, nproc_x, round_to_float=round_to_float
        )
    elif operation == "s_surface":
        forward = surface_stress_forward(inputs, layout, nproc_x)
        transpose = surface_stress_transpose(
            bars, layout, nproc_x, round_to_float=round_to_float
        )
    elif operation == "v_composed":
        forward = surface_velocity_forward(
            exchange_forward(inputs, layout, nproc_x, nproc_y, boundary, kind),
            layout, nproc_x,
        )
        transpose = exchange_transpose(
            surface_velocity_transpose(
                bars, layout, nproc_x, round_to_float=round_to_float
            ),
            layout, nproc_x, nproc_y, boundary, kind,
            round_to_float=round_to_float,
        )
    elif operation == "s_composed":
        forward = exchange_forward(
            surface_stress_forward(inputs, layout, nproc_x),
            layout, nproc_x, nproc_y, boundary, kind,
        )
        transpose = surface_stress_transpose(
            exchange_transpose(
                bars, layout, nproc_x, nproc_y, boundary, kind,
                round_to_float=round_to_float,
            ),
            layout, nproc_x, round_to_float=round_to_float,
        )
    else:
        raise ValueError(operation)
    return inputs, bars, forward, transpose


def dot(left, right):
    return math.fsum(
        a * b
        for rank_left, rank_right in zip(left, right)
        for field_left, field_right in zip(rank_left, rank_right)
        for a, b in zip(field_left, field_right)
    )


def write_case_files(directory: Path, operation, layout, nproc_x, nproc_y, boundary):
    inputs, bars, forward, transpose = build_case(
        operation, layout, nproc_x, nproc_y, boundary, round_to_float=True
    )
    directory.mkdir(parents=True, exist_ok=True)
    for rank in range(nproc_x * nproc_y):
        payload = array("f")
        for block in (inputs, bars, forward, transpose):
            for field in block[rank]:
                payload.extend(field)
        with (directory / f"rank_{rank}.bin").open("wb") as stream:
            payload.tofile(stream)
    return inputs, bars
