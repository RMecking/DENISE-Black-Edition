from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Literal, Sequence

from tests.cases.sh_fwi_gradient import (
    SHFWIGradientConfig,
    _records,
    _write_common,
    _write_float_grid,
    _write_parameters,
)


Parameterization = Literal["T", "R", "M"]


def _float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _denise_float_modulus(density: float, vs: float) -> float:
    """Match C float evaluation of ``rho * u * u`` in av_mu_SH()."""
    density32 = _float32(density)
    vs32 = _float32(vs)
    return _float32(_float32(density32 * vs32) * vs32)


def _density_workflow(directory: Path) -> None:
    header = (
        "PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ "
        "INV_VP_ITER INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP "
        "WD_DAMP1 EPRECOND LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE "
        "OFFSET_MUTE OFFSETC SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER\n"
    )
    values = (
        "0.01 0 0.0 0.0 4 0 0.0 0.0 0.0 99 1 1 99 0 0.0 0.0 0 2 0 0 "
        "0.0 0.0 0 0 0.0 1.0 1.0 0 0.0 0\n"
    )
    (directory / "workflow.inp").write_text(header + values, encoding="ascii")


def _write_model(
    directory: Path,
    config: SHFWIGradientConfig,
    *,
    parameterization: Parameterization,
    epsilon_fraction: float,
    direction: Sequence[float],
    density_background: Sequence[float] | None = None,
) -> int:
    if len(direction) != config.cell_count:
        raise ValueError("Direction size differs from model grid")
    density0 = (
        list(density_background)
        if density_background is not None
        else [config.density_kg_m3] * config.cell_count
    )
    if len(density0) != config.cell_count or any(value <= 0.0 for value in density0):
        raise ValueError("Invalid density background")
    vs0 = config.background_vs()
    density = list(density0)
    if parameterization in ("T", "R"):
        density = [
            value * (1.0 + epsilon_fraction * component)
            for value, component in zip(density0, direction)
        ]
    _write_float_grid(directory / "model" / "current.rho", density)
    if parameterization == "T":
        _write_float_grid(directory / "model" / "current.vs", vs0)
        return 1
    modulus0 = [
        _denise_float_modulus(rho, vs) for rho, vs in zip(density0, vs0)
    ]
    modulus = list(modulus0)
    if parameterization == "M":
        modulus = [
            value * (1.0 + epsilon_fraction * component)
            for value, component in zip(modulus0, direction)
        ]
    _write_float_grid(directory / "model" / "current.mu", modulus)
    return 3


def generate_density_case(
    directory: Path,
    *,
    config: SHFWIGradientConfig,
    parameterization: Parameterization,
    epsilon_fraction: float,
    direction: Sequence[float],
    grad_form: int,
    mode: int,
    observed_su: Path | None = None,
    density_background: Sequence[float] | None = None,
    nprocx: int = 1,
    nprocy: int = 1,
) -> None:
    _write_common(directory, config)
    invmat1 = _write_model(
        directory,
        config,
        parameterization=parameterization,
        epsilon_fraction=epsilon_fraction,
        direction=direction,
        density_background=density_background,
    )
    records = _records(
        config, mode=mode, grad_form=grad_form,
        nprocx=nprocx, nprocy=nprocy,
    )
    records[65] = f"INVMAT1 ={invmat1}"
    _write_parameters(directory, records)
    _density_workflow(directory)
    if observed_su is not None:
        (directory / "observed_y.su.shot1").write_bytes(observed_su.read_bytes())
    (directory / "case.json").write_text(
        json.dumps(
            config.as_metadata()
            | {
                "role": "m5f_density",
                "parameterization": parameterization,
                "epsilon_fraction": epsilon_fraction,
                "grad_form": grad_form,
                "mode": mode,
                "invmat1": invmat1,
                "direction_normalization": "delta_rho=rho_background*p",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def generate_density_observed_case(
    directory: Path, *, config: SHFWIGradientConfig, direction: Sequence[float],
    density_background: Sequence[float] | None = None,
    nprocx: int = 1, nprocy: int = 1,
) -> None:
    generate_density_case(
        directory,
        config=config,
        parameterization="T",
        epsilon_fraction=config.target_fraction,
        direction=direction,
        grad_form=2,
        mode=0,
        density_background=density_background,
        nprocx=nprocx,
        nprocy=nprocy,
    )
