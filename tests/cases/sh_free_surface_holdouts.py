from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case
from tests.utilities.sh_free_surface import ricker_f95


HoldoutName = Literal["heterogeneous", "corner_left", "corner_right"]


@dataclass(frozen=True)
class SHFreeSurfaceHoldout:
    name: HoldoutName
    config: HomogeneousSHConfig
    free_surface: bool = True


def heterogeneous_holdout(fd_order: int, *, free_surface: bool = True) -> SHFreeSurfaceHoldout:
    return SHFreeSurfaceHoldout(
        name="heterogeneous",
        config=HomogeneousSHConfig(
            nx=240,
            ny=192,
            dh_m=10.0,
            time_s=1.6005,
            dt_s=0.0005,
            vs_m_s=2000.0,
            density_kg_m3=2000.0,
            source_x_m=800.0,
            source_y_m=350.0,
            source_frequency_hz=8.0,
            receiver_x_m=(1600.0,),
            receiver_y_m=450.0,
            fd_order=fd_order,
            absorbing_width_gridpoints=20,
            damping_velocity_m_s=2550.0,
            pml_frequency_hz=8.0,
        ),
        free_surface=free_surface,
    )


def corner_holdout(side: Literal["left", "right"], *, free_surface: bool = True) -> SHFreeSurfaceHoldout:
    source_x = 450.0 if side == "left" else 1960.0
    receiver_x = 300.0 if side == "left" else 2110.0
    return SHFreeSurfaceHoldout(
        name=f"corner_{side}",
        config=HomogeneousSHConfig(
            nx=240,
            ny=192,
            dh_m=10.0,
            time_s=1.6005,
            dt_s=0.0005,
            vs_m_s=2000.0,
            density_kg_m3=2000.0,
            source_x_m=source_x,
            source_y_m=350.0,
            source_frequency_hz=8.0,
            receiver_x_m=(receiver_x,),
            receiver_y_m=100.0,
            fd_order=12,
            absorbing_width_gridpoints=20,
            damping_velocity_m_s=2000.0,
            pml_frequency_hz=8.0,
        ),
        free_surface=free_surface,
    )


def corner_wide_reference(side: Literal["left", "right"]) -> SHFreeSurfaceHoldout:
    narrow = corner_holdout(side)
    x_shift = 800.0 if side == "left" else 0.0
    return replace(
        narrow,
        config=replace(
            narrow.config,
            nx=320,
            source_x_m=narrow.config.source_x_m + x_shift,
            receiver_x_m=(narrow.config.receiver_x_m[0] + x_shift,),
        ),
    )


def _heterogeneous_values(config: HomogeneousSHConfig) -> tuple[array, array]:
    vs = array("f")
    rho = array("f")
    for i in range(1, config.nx + 1):
        for j in range(1, config.ny + 1):
            if j <= 24:
                phase = ((i - 1) // 24 + (j - 1) // 6) % 2
                vs.append(1450.0 if phase == 0 else 2550.0)
                rho.append(1700.0 if phase == 0 else 2400.0)
            elif j <= 48:
                phase = ((i - 1) // 30) % 2
                vs.append(1750.0 if phase == 0 else 2250.0)
                rho.append(1850.0 if phase == 0 else 2250.0)
            else:
                vs.append(2000.0)
                rho.append(2000.0)
    return vs, rho


def _write_float_model(path: Path, values: array) -> None:
    if values.itemsize != 4:
        raise AssertionError("DENISE hold-out model requires binary32 floats")
    with path.open("wb") as stream:
        values.tofile(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_holdout_case(
    directory: Path,
    *,
    holdout: SHFreeSurfaceHoldout,
    nprocx: int = 1,
    nprocy: int = 1,
) -> HomogeneousSHConfig:
    config = holdout.config
    generate_case(directory, config=config, nprocx=nprocx, nprocy=nprocy)

    parameters = directory / "denise.inp"
    text = parameters.read_text(encoding="ascii")
    if text.count("FREE_SURF =0") != 1:
        raise AssertionError("Expected exactly one FREE_SURF record")
    parameters.write_text(
        text.replace("FREE_SURF =0", f"FREE_SURF ={int(holdout.free_surface)}"),
        encoding="ascii",
    )

    model_definition: dict[str, object]
    if holdout.name == "heterogeneous":
        vs, rho = _heterogeneous_values(config)
        _write_float_model(directory / "model" / "homogeneous.vs", vs)
        _write_float_model(directory / "model" / "homogeneous.rho", rho)
        model_definition = {
            "kind": "deterministic_lateral_near_surface",
            "top_rows": 24,
            "transition_rows": [25, 48],
            "vs_range_m_s": [1450.0, 2550.0],
            "rho_range_kg_m3": [1700.0, 2400.0],
            "x_major_y_minor": True,
            "crosses_x_seam_at_grid_index": config.nx // 2,
        }
    else:
        model_definition = {
            "kind": "homogeneous_corner_illumination",
            "vs_m_s": config.vs_m_s,
            "rho_kg_m3": config.density_kg_m3,
        }

    metadata = {
        "milestone": "M6.1d",
        "holdout": holdout.name,
        "free_surface": holdout.free_surface,
        "configuration": config.as_metadata(),
        "nprocx": nprocx,
        "nprocy": nprocy,
        "model_definition": model_definition,
        "source_spectrum": {
            "source_frequency_hz": config.source_frequency_hz,
            "f95_hz": ricker_f95(config.source_frequency_hz),
        },
        "input_sha256": {
            name: _sha256(directory / name)
            for name in (
                "model/homogeneous.vs",
                "model/homogeneous.rho",
                "source.dat",
                "receiver.dat",
            )
        },
        "contracts": {
            "boundary": "M6.1 exact surface-state contract",
            "mpi_relative_l2_max": 1.0e-6,
            "mpi_correlation_min": 0.999999,
            "stability": "M6.1 compatible-energy bound",
            "heterogeneous_wavefield_reference": None,
            "corner_waveform_tolerance": None,
        },
    }
    (directory / "case.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config
