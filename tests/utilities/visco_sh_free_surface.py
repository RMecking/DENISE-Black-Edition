from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


SUPPORTED_FDORDERS = (2, 4, 6, 8, 10, 12)


@dataclass(frozen=True)
class ViscoSurfaceRows:
    fd_order: int
    half_order: int
    active_vz_ghosts: tuple[int, ...]
    full_vz_extension: tuple[int, ...]
    total_syz_extension: tuple[int, ...]
    minimum_q_ghosts: tuple[int, ...]
    full_q_extension: tuple[int, ...]


@dataclass(frozen=True)
class GSLSSplitState:
    total_stress: float
    memory: tuple[float, ...]


@dataclass(frozen=True)
class ViscoImageGeometry:
    candidate_source: tuple[float, float]
    candidate_receiver: tuple[float, float]
    reference_plane_y: float
    reference_real_source: tuple[float, float]
    reference_image_source: tuple[float, float]
    reference_receiver: tuple[float, float]

    @staticmethod
    def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
        return math.hypot(left[0] - right[0], left[1] - right[1])

    @property
    def candidate_direct_distance(self) -> float:
        return self._distance(self.candidate_source, self.candidate_receiver)

    @property
    def candidate_image_distance(self) -> float:
        mirrored = (self.candidate_source[0], -self.candidate_source[1])
        return self._distance(mirrored, self.candidate_receiver)

    @property
    def reference_direct_distance(self) -> float:
        return self._distance(self.reference_real_source, self.reference_receiver)

    @property
    def reference_image_distance(self) -> float:
        return self._distance(self.reference_image_source, self.reference_receiver)


def visco_surface_rows(fd_order: int) -> ViscoSurfaceRows:
    if fd_order not in SUPPORTED_FDORDERS:
        raise ValueError(f"Unsupported FDORDER {fd_order}")
    half_order = fd_order // 2
    active_vz = tuple(range(2 - half_order, 1)) if half_order > 1 else ()
    full = tuple(range(1 - half_order, 1))
    return ViscoSurfaceRows(
        fd_order=fd_order,
        half_order=half_order,
        active_vz_ghosts=active_vz,
        full_vz_extension=full,
        total_syz_extension=full,
        minimum_q_ghosts=(),
        full_q_extension=full,
    )


def gsls_split_step(
    *,
    total_stress: float,
    memory: Sequence[float],
    derivative: float,
    instantaneous_increment_coefficient: float,
    dt: float,
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
) -> GSLSSplitState:
    """Apply the split sequence implemented by update_s_visc_PML_SH.c."""
    if not (len(memory) == len(b) == len(c) == len(d)):
        raise ValueError("GSLS mechanism arrays must have identical lengths")
    half_dt = 0.5 * dt
    partial = (
        total_stress
        + instantaneous_increment_coefficient * derivative
        + half_dt * sum(memory)
    )
    updated = tuple(
        b_l * (old_l * c_l - d_l * derivative)
        for old_l, b_l, c_l, d_l in zip(memory, b, c, d)
    )
    return GSLSSplitState(
        total_stress=partial + half_dt * sum(updated),
        memory=updated,
    )


def staggered_forward_2d(
    values: Mapping[tuple[int, int], float],
    *,
    row: int,
    column: int,
    axis: str,
    coefficients: Sequence[float],
) -> float:
    """Evaluate the unscaled D+ stencil used by the SH stress update."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    result = 0.0
    for m, coefficient in enumerate(coefficients, start=1):
        if axis == "x":
            positive = (row, column + m)
            negative = (row, column - (m - 1))
        else:
            positive = (row + m, column)
            negative = (row - (m - 1), column)
        result += coefficient * (values[positive] - values[negative])
    return result


def translated_image_geometry(
    *,
    candidate_source: tuple[float, float],
    candidate_receiver: tuple[float, float],
    reference_plane_y: float,
    dh: float,
) -> ViscoImageGeometry:
    """Translate a y=0 half-space pair to an interior image-source plane."""
    if dh <= 0.0 or reference_plane_y <= 0.0:
        raise ValueError("DH and the reference-plane depth must be positive")
    if not math.isclose(reference_plane_y / dh, round(reference_plane_y / dh)):
        raise ValueError("The reference plane must lie on a native syz row")
    for name, point in (("source", candidate_source), ("receiver", candidate_receiver)):
        if point[1] <= 0.0:
            raise ValueError(f"Candidate {name} must lie below y=0")
        for coordinate in point:
            phase = coordinate / dh - 0.5
            if not math.isclose(phase, round(phase)):
                raise ValueError(f"Candidate {name} is not on a native vz point")
    if reference_plane_y <= max(candidate_source[1], candidate_receiver[1]):
        raise ValueError("Reference plane must leave room for the mirrored source")
    return ViscoImageGeometry(
        candidate_source=candidate_source,
        candidate_receiver=candidate_receiver,
        reference_plane_y=reference_plane_y,
        reference_real_source=(
            candidate_source[0],
            reference_plane_y + candidate_source[1],
        ),
        reference_image_source=(
            candidate_source[0],
            reference_plane_y - candidate_source[1],
        ),
        reference_receiver=(
            candidate_receiver[0],
            reference_plane_y + candidate_receiver[1],
        ),
    )
