from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


SUPPORTED_FDORDERS = (2, 4, 6, 8, 10, 12)


@dataclass(frozen=True)
class CpmlBranch:
    active: bool
    K: float = 1.0
    a: float = 0.0
    b: float = 0.0


def stress_derivatives(
    sxz: Sequence[float],
    syz: Sequence[float],
    *,
    side: int,
    center: int,
    fdorder: int,
    hc: Sequence[float],
) -> tuple[float, float]:
    half_order = fdorder // 2
    dx = math.fsum(
        hc[m] * (sxz[center + (m - 1)] - sxz[center - m])
        for m in range(1, half_order + 1)
    )
    dy = math.fsum(
        hc[m]
        * (syz[center + (m - 1) * side] - syz[center - m * side])
        for m in range(1, half_order + 1)
    )
    return dx, dy


def stress_derivatives_transpose(
    sxz: Sequence[float],
    syz: Sequence[float],
    *,
    side: int,
    center: int,
    fdorder: int,
    hc: Sequence[float],
    bar_dx: float,
    bar_dy: float,
) -> tuple[list[float], list[float]]:
    out_x = list(sxz)
    out_y = list(syz)
    for m in range(1, fdorder // 2 + 1):
        scale = hc[m]
        out_x[center + (m - 1)] += scale * bar_dx
        out_x[center - m] -= scale * bar_dx
        out_y[center + (m - 1) * side] += scale * bar_dy
        out_y[center - m * side] -= scale * bar_dy
    return out_x, out_y


def cpml_forward(raw: float, psi_previous: float, cpml: CpmlBranch):
    if not cpml.active:
        return raw, None
    psi_next = cpml.b * psi_previous + cpml.a * raw
    return raw / cpml.K + psi_next, psi_next


def cpml_transpose(
    bar_corrected: float, bar_psi_next: float, cpml: CpmlBranch
) -> tuple[float, float]:
    if not cpml.active:
        if bar_psi_next != 0.0:
            raise ValueError("inactive CPML has no state output")
        return bar_corrected, 0.0
    t_psi = bar_psi_next + bar_corrected
    return bar_corrected / cpml.K + cpml.a * t_psi, cpml.b * t_psi


def velocity_forward(
    *,
    vz_previous: float,
    sxz: Sequence[float],
    syz: Sequence[float],
    psi_previous: Sequence[float],
    cpml: Sequence[CpmlBranch],
    side: int,
    center: int,
    fdorder: int,
    hc: Sequence[float],
    dt: float,
    dh: float,
    rhoi: float,
):
    raw = stress_derivatives(
        sxz, syz, side=side, center=center, fdorder=fdorder, hc=hc
    )
    corrected = []
    psi_next = []
    for axis in range(2):
        value, psi = cpml_forward(raw[axis], psi_previous[axis], cpml[axis])
        corrected.append(value)
        psi_next.append(psi)
    alpha = dt * rhoi / dh
    return {
        "vz_next": vz_previous + alpha * math.fsum(corrected),
        "psi_next": tuple(psi_next),
        "raw": raw,
        "corrected": tuple(corrected),
    }


def velocity_transpose(
    *,
    initial_vz: float,
    initial_sxz: Sequence[float],
    initial_syz: Sequence[float],
    initial_psi: Sequence[float],
    bar_vz_next: float,
    bar_psi_next: Sequence[float],
    cpml: Sequence[CpmlBranch],
    side: int,
    center: int,
    fdorder: int,
    hc: Sequence[float],
    dt: float,
    dh: float,
    rhoi: float,
):
    alpha = dt * rhoi / dh
    raw = []
    psi = []
    for axis in range(2):
        value, state = cpml_transpose(
            alpha * bar_vz_next, bar_psi_next[axis], cpml[axis]
        )
        raw.append(value)
        psi.append(initial_psi[axis] + state)
    sxz, syz = stress_derivatives_transpose(
        initial_sxz,
        initial_syz,
        side=side,
        center=center,
        fdorder=fdorder,
        hc=hc,
        bar_dx=raw[0],
        bar_dy=raw[1],
    )
    return {
        "bar_vz_previous": initial_vz + bar_vz_next,
        "bar_sxz": tuple(sxz),
        "bar_syz": tuple(syz),
        "bar_psi_previous": tuple(psi),
    }


def receiver_sample(
    vz: Sequence[float], positions: Sequence[tuple[int, int]], *, stride: int
) -> tuple[float, ...]:
    return tuple(vz[y * stride + x] for x, y in positions)


def receiver_transpose(
    initial_vz: Sequence[float],
    positions: Sequence[tuple[int, int]],
    bar_data: Sequence[float],
    *,
    stride: int,
) -> tuple[float, ...]:
    result = list(initial_vz)
    for (x, y), value in zip(positions, bar_data):
        result[y * stride + x] += value
    return tuple(result)


def source_inject(
    vz: Sequence[float],
    positions: Sequence[tuple[int, int]],
    source_types: Sequence[int],
    signals: Sequence[float],
    *,
    stride: int,
) -> tuple[float, ...]:
    result = list(vz)
    for (x, y), source_type, signal in zip(positions, source_types, signals):
        if source_type == 1:
            result[y * stride + x] += signal
    return tuple(result)


def source_transpose(
    initial_vz: Sequence[float],
    initial_signal: Sequence[float],
    bar_vz_after: Sequence[float],
    positions: Sequence[tuple[int, int]],
    source_types: Sequence[int],
    *,
    stride: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    bar_vz = tuple(a + b for a, b in zip(initial_vz, bar_vz_after))
    bar_signal = list(initial_signal)
    for source, ((x, y), source_type) in enumerate(zip(positions, source_types)):
        if source_type == 1:
            bar_signal[source] += bar_vz_after[y * stride + x]
    return bar_vz, tuple(bar_signal)
