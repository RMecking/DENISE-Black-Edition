from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from tests.cases.homogeneous_psv import HomogeneousPSVConfig
from tests.cases.homogeneous_sh import HomogeneousSHConfig


Physics = Literal["sh", "psv"]
Component = Literal["vz", "vx", "vy"]


@dataclass(frozen=True)
class CPMLPair:
    name: str
    physics: Physics
    compact: HomogeneousSHConfig | HomogeneousPSVConfig
    reference: HomogeneousSHConfig | HomogeneousPSVConfig
    component: Component
    direct_window_s: tuple[float, float]
    reflection_window_s: tuple[float, float]
    acceptance_db: float
    compact_to_reference_translation_m: tuple[float, float]


def _windows(
    *, frequency_hz: float, velocity_m_s: float, direct_path_m: float,
    first_boundary_path_m: float, last_boundary_path_m: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    source_peak = 1.5 / frequency_hz
    direct_center = source_peak + direct_path_m / velocity_m_s
    return (
        (direct_center - 0.75 / frequency_hz, direct_center + 0.75 / frequency_hz),
        (
            source_peak + first_boundary_path_m / velocity_m_s - 0.5 / frequency_hz,
            source_peak + last_boundary_path_m / velocity_m_s + 0.5 / frequency_hz,
        ),
    )


def normal_sh_pair(side: Literal["left", "right", "top", "bottom"]) -> CPMLPair:
    base = HomogeneousSHConfig(nx=120, ny=240, time_s=1.0)
    translation = (0.0, 0.0)
    if side == "right":
        compact = replace(base, source_x_m=500.0, source_y_m=1200.0,
                          receiver_x_m=(400.0,), receiver_y_m=1200.0)
        translation = (1200.0, 0.0)
        reference = replace(compact, nx=360, source_x_m=1700.0, receiver_x_m=(1600.0,))
    elif side == "left":
        compact = replace(base, source_x_m=700.0, source_y_m=1200.0,
                          receiver_x_m=(800.0,), receiver_y_m=1200.0)
        translation = (1200.0, 0.0)
        reference = replace(compact, nx=360, source_x_m=1900.0, receiver_x_m=(2000.0,))
    elif side == "bottom":
        compact = replace(base, nx=240, ny=120, source_x_m=1200.0, source_y_m=500.0,
                          receiver_x_m=(1200.0,), receiver_y_m=400.0)
        translation = (0.0, 1200.0)
        reference = replace(compact, ny=360, source_y_m=1700.0, receiver_y_m=1600.0)
    else:
        compact = replace(base, nx=240, ny=120, source_x_m=1200.0, source_y_m=700.0,
                          receiver_x_m=(1200.0,), receiver_y_m=800.0)
        translation = (0.0, 1200.0)
        reference = replace(compact, ny=360, source_y_m=1900.0, receiver_y_m=2000.0)
    direct, reflection = _windows(
        frequency_hz=base.source_frequency_hz, velocity_m_s=base.vs_m_s,
        direct_path_m=100.0, first_boundary_path_m=1200.0, last_boundary_path_m=1500.0,
    )
    return CPMLPair(side, "sh", compact, reference, "vz", direct, reflection, -30.0, translation)


def oblique_sh_pair() -> CPMLPair:
    compact = HomogeneousSHConfig(
        nx=120, ny=240, time_s=1.05, source_x_m=500.0, source_y_m=1500.0,
        receiver_x_m=(400.0,), receiver_y_m=900.0,
    )
    reference = replace(compact, nx=360, source_x_m=1700.0, receiver_x_m=(1600.0,))
    direct, reflection = _windows(
        frequency_hz=compact.source_frequency_hz, velocity_m_s=compact.vs_m_s,
        direct_path_m=math.hypot(100.0, 600.0),
        first_boundary_path_m=math.hypot(1200.0, 600.0),
        last_boundary_path_m=math.hypot(1500.0, 600.0),
    )
    return CPMLPair("oblique_right", "sh", compact, reference, "vz", direct, reflection,
                    -25.0, (1200.0, 0.0))


def corner_sh_pair() -> CPMLPair:
    compact = HomogeneousSHConfig(
        nx=120, ny=120, time_s=1.32, source_x_m=500.0, source_y_m=500.0,
        receiver_x_m=(400.0,), receiver_y_m=400.0,
    )
    reference = replace(
        compact, nx=360, ny=360, source_x_m=1700.0, source_y_m=1700.0,
        receiver_x_m=(1600.0,), receiver_y_m=1600.0,
    )
    direct, reflection = _windows(
        frequency_hz=compact.source_frequency_hz, velocity_m_s=compact.vs_m_s,
        direct_path_m=math.hypot(100.0, 100.0),
        # Earliest side-PML return (inner x/y edge) through latest outer-corner return.
        first_boundary_path_m=math.hypot(600.0, 100.0),
        last_boundary_path_m=math.hypot(900.0, 900.0),
    )
    return CPMLPair("corner", "sh", compact, reference, "vz", direct, reflection,
                    -25.0, (1200.0, 1200.0))


def normal_p_pair(axis: Literal["x", "y"]) -> CPMLPair:
    base = HomogeneousPSVConfig(nx=120, ny=240, time_s=0.75)
    if axis == "x":
        compact = replace(base, source_x_m=500.0, source_y_m=1200.0,
                          receivers_m=((400.0, 1200.0),))
        reference = replace(compact, nx=360, source_x_m=1700.0,
                            receivers_m=((1600.0, 1200.0),))
        component, translation = "vx", (1200.0, 0.0)
    else:
        compact = replace(base, nx=240, ny=120, source_x_m=1200.0, source_y_m=500.0,
                          receivers_m=((1200.0, 400.0),))
        reference = replace(compact, ny=360, source_y_m=1700.0,
                            receivers_m=((1200.0, 1600.0),))
        component, translation = "vy", (0.0, 1200.0)
    direct, reflection = _windows(
        frequency_hz=base.source_frequency_hz, velocity_m_s=base.vp_m_s,
        direct_path_m=100.0, first_boundary_path_m=1200.0, last_boundary_path_m=1500.0,
    )
    return CPMLPair(f"p_{axis}", "psv", compact, reference, component, direct, reflection,
                    -30.0, translation)


def normal_sv_pair() -> CPMLPair:
    compact = HomogeneousPSVConfig(
        nx=120, ny=240, time_s=1.08, source_x_m=500.0, source_y_m=1200.0,
        receivers_m=((400.0, 1200.0),), source_type=3,
    )
    reference = replace(compact, nx=360, source_x_m=1700.0,
                        receivers_m=((1600.0, 1200.0),))
    direct, reflection = _windows(
        frequency_hz=compact.source_frequency_hz, velocity_m_s=compact.vs_m_s,
        direct_path_m=100.0, first_boundary_path_m=1200.0, last_boundary_path_m=1500.0,
    )
    return CPMLPair("sv_x", "psv", compact, reference, "vy", direct, reflection,
                    -30.0, (1200.0, 0.0))
