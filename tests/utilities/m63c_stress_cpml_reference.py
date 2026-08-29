from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from tests.utilities.m63c_acceptance import (
    LocalGSLSCoefficients,
    local_gsls_forward,
    local_gsls_transpose,
)


SUPPORTED_FDORDERS = (2, 4, 6, 8, 10, 12)


@dataclass(frozen=True)
class CpmlBranch:
    active: bool
    K: float = 1.0
    a: float = 0.0
    b: float = 0.0


def spatial_forward(
    patch: Sequence[float],
    *,
    side: int,
    center: int,
    fdorder: int,
    dh: float,
    hc: Sequence[float],
) -> tuple[float, float]:
    half_order = fdorder // 2
    ex = math.fsum(
        hc[m]
        * (patch[center + m] - patch[center - (m - 1)])
        / dh
        for m in range(1, half_order + 1)
    )
    ey = math.fsum(
        hc[m]
        * (patch[center + m * side] - patch[center - (m - 1) * side])
        / dh
        for m in range(1, half_order + 1)
    )
    return ex, ey


def spatial_transpose(
    *,
    patch: Sequence[float],
    side: int,
    center: int,
    fdorder: int,
    dh: float,
    hc: Sequence[float],
    bar_ex: float,
    bar_ey: float,
) -> list[float]:
    result = list(patch)
    for m in range(1, fdorder // 2 + 1):
        scale = hc[m] / dh
        result[center + m] += scale * bar_ex
        result[center - (m - 1)] -= scale * bar_ex
        result[center + m * side] += scale * bar_ey
        result[center - (m - 1) * side] -= scale * bar_ey
    return result


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


def stress_block_forward(
    *,
    patch: Sequence[float],
    side: int,
    center: int,
    fdorder: int,
    dh: float,
    hc: Sequence[float],
    stress_previous: Sequence[float],
    memory_previous: Sequence[Sequence[float]],
    psi_previous: Sequence[float],
    cpml: Sequence[CpmlBranch],
    coefficients: Sequence[LocalGSLSCoefficients],
):
    raw = spatial_forward(
        patch, side=side, center=center, fdorder=fdorder, dh=dh, hc=hc
    )
    corrected = []
    psi_next = []
    stress_next = []
    memory_next = []
    for axis in range(2):
        value, psi = cpml_forward(raw[axis], psi_previous[axis], cpml[axis])
        corrected.append(value)
        psi_next.append(psi)
        stress, memory = local_gsls_forward(
            stress_previous[axis], memory_previous[axis], value, coefficients[axis]
        )
        stress_next.append(stress)
        memory_next.append(memory)
    return {
        "raw": tuple(raw),
        "corrected": tuple(corrected),
        "psi_next": tuple(psi_next),
        "stress_next": tuple(stress_next),
        "memory_next": tuple(memory_next),
    }


def stress_block_transpose(
    *,
    initial_patch: Sequence[float],
    side: int,
    center: int,
    fdorder: int,
    dh: float,
    hc: Sequence[float],
    corrected_strain: Sequence[float],
    bar_stress_next: Sequence[float],
    bar_memory_next: Sequence[Sequence[float]],
    bar_psi_next: Sequence[float],
    cpml: Sequence[CpmlBranch],
    coefficients: Sequence[LocalGSLSCoefficients],
    initial_stress: Sequence[float] = (0.0, 0.0),
    initial_memory: Sequence[Sequence[float]] | None = None,
    initial_psi: Sequence[float] = (0.0, 0.0),
    initial_g_tau: Sequence[float] = (0.0, 0.0),
    initial_g_modulus: Sequence[float] = (0.0, 0.0),
):
    if initial_memory is None:
        initial_memory = tuple(
            (0.0,) * len(branch.recurrence) for branch in coefficients
        )
    bar_stress_previous = []
    bar_memory_previous = []
    bar_raw = []
    bar_psi_previous = []
    g_tau = []
    g_modulus = []
    for axis in range(2):
        local = local_gsls_transpose(
            stress_next_adjoint=bar_stress_next[axis],
            memory_next_adjoint=bar_memory_next[axis],
            strain=corrected_strain[axis],
            coefficients=coefficients[axis],
        )
        raw, psi = cpml_transpose(local[2], bar_psi_next[axis], cpml[axis])
        bar_stress_previous.append(initial_stress[axis] + local[0])
        bar_memory_previous.append(
            tuple(
                left + right
                for left, right in zip(initial_memory[axis], local[1])
            )
        )
        bar_raw.append(raw)
        bar_psi_previous.append(initial_psi[axis] + psi)
        g_tau.append(initial_g_tau[axis] + local[3])
        g_modulus.append(initial_g_modulus[axis] + local[4])
    bar_patch = spatial_transpose(
        patch=initial_patch,
        side=side,
        center=center,
        fdorder=fdorder,
        dh=dh,
        hc=hc,
        bar_ex=bar_raw[0],
        bar_ey=bar_raw[1],
    )
    return {
        "bar_patch": tuple(bar_patch),
        "bar_stress_previous": tuple(bar_stress_previous),
        "bar_memory_previous": tuple(bar_memory_previous),
        "bar_psi_previous": tuple(bar_psi_previous),
        "g_tau": tuple(g_tau),
        "g_modulus": tuple(g_modulus),
    }
