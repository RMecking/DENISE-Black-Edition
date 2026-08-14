from __future__ import annotations

import json
import math
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from tests.cases.homogeneous_sh import HomogeneousSHConfig, _parameter_lines
from tests.utilities.fwi_gradient import gaussian_direction


@dataclass(frozen=True)
class SHFWIGradientConfig:
    nx: int = 96
    ny: int = 80
    dh_m: float = 10.0
    time_s: float = 0.40
    dt_s: float = 0.0004
    dtinv: int = 1
    vs_m_s: float = 2300.0
    density_kg_m3: float = 1800.0
    target_fraction: float = 0.02
    source_x_m: float = 260.0
    source_y_m: float = 400.0
    source_frequency_hz: float = 12.0
    receiver_x_m: tuple[float, ...] = (420.0, 500.0, 580.0, 660.0, 740.0)
    receiver_y_m: float = 400.0
    anomaly_x_m: float = 520.0
    anomaly_y_m: float = 400.0
    anomaly_sigma_m: float = 70.0
    background_contrast_fraction: float = 0.0
    fd_order: int = 8
    absorbing_width_gridpoints: int = 12

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny

    def direction(self) -> list[float]:
        return gaussian_direction(
            nx=self.nx,
            ny=self.ny,
            dh_m=self.dh_m,
            center_x_m=self.anomaly_x_m,
            center_y_m=self.anomaly_y_m,
            sigma_m=self.anomaly_sigma_m,
        )

    def background_vs(self) -> list[float]:
        """Return a smooth lateral background with the requested total contrast."""
        if not 0.0 <= self.background_contrast_fraction <= 0.4:
            raise ValueError("background contrast must be between zero and 0.4")
        half_contrast = 0.5 * self.background_contrast_fraction
        return [
            self.vs_m_s
            * (1.0 + half_contrast * math.sin(2.0 * math.pi * (ix - 0.5) / self.nx))
            for ix in range(1, self.nx + 1)
            for _iy in range(1, self.ny + 1)
        ]

    def as_metadata(self) -> dict[str, object]:
        result = asdict(self)
        result["receiver_x_m"] = list(self.receiver_x_m)
        result["direction_normalization"] = "delta_Vs=Vs_background*p; max_abs_p_equals_1"
        return result


def _write_float_grid(path: Path, values: Sequence[float]) -> None:
    data = array("f", values)
    with path.open("wb") as stream:
        data.tofile(stream)


def _base_config(config: SHFWIGradientConfig) -> HomogeneousSHConfig:
    return HomogeneousSHConfig(
        nx=config.nx,
        ny=config.ny,
        dh_m=config.dh_m,
        time_s=config.time_s,
        dt_s=config.dt_s,
        vs_m_s=config.vs_m_s,
        density_kg_m3=config.density_kg_m3,
        source_x_m=config.source_x_m,
        source_y_m=config.source_y_m,
        source_frequency_hz=config.source_frequency_hz,
        receiver_x_m=config.receiver_x_m,
        receiver_y_m=config.receiver_y_m,
        fd_order=config.fd_order,
        absorbing_width_gridpoints=config.absorbing_width_gridpoints,
        damping_velocity_m_s=config.vs_m_s * (1.0 + 0.5 * config.background_contrast_fraction),
        pml_frequency_hz=config.source_frequency_hz,
    )


def _write_common(directory: Path, config: SHFWIGradientConfig) -> None:
    for name in (
        "model", "su", "log", "snap", "wavelet", "jacobian", "taper",
        "picked_times", "trace_kill", "gravity", "inverted",
    ):
        (directory / name).mkdir(parents=True, exist_ok=True)
    source = (
        f"1\n{config.source_x_m} 0.0 {config.source_y_m} 0.0 "
        f"{config.source_frequency_hz} 1.0 0.0 1\n"
    )
    (directory / "source.dat").write_text(source, encoding="ascii")
    receivers = "".join(f"{x} {config.receiver_y_m}\n" for x in config.receiver_x_m)
    (directory / "receiver.dat").write_text(receivers, encoding="ascii")


def _write_parameters(directory: Path, records: Sequence[str]) -> None:
    text = "# Generated elastic SH FWI gradient verification case\n" + "".join(
        f"# positional parameter {index:03d}\n{line}\n"
        for index, line in enumerate(records, start=1)
    )
    (directory / "denise.inp").write_text(text, encoding="ascii")


def _records(
    config: SHFWIGradientConfig, *, mode: int, grad_form: int,
    nprocx: int = 1, nprocy: int = 1,
) -> list[str]:
    records = _parameter_lines(_base_config(config), nprocx, nprocy)
    overrides = {
        1: f"MODE ={mode}",
        23: "MFILE =model/current",
        52: "SEIS_FORMAT =1",
        54: "SEIS_FILE_VY =su/synthetic_y.su",
        60: "ITERMAX =1",
        61: "JACOBIAN =jacobian/gradient",
        62: "DATA_DIR =observed",
        67: f"GRAD_FORM ={grad_form}",
        80: "INV_MOD_OUT =0",
        81: "INV_MODELFILE =inverted/model",
        90: "GRAD_METHOD =1",
        95: f"DTINV ={config.dtinv}",
        97: "STEPMAX =1",
    }
    for position, value in overrides.items():
        records[position - 1] = value
    return records


def _write_workflow(directory: Path) -> None:
    header = (
        "PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ "
        "INV_VP_ITER INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP "
        "WD_DAMP1 EPRECOND LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE "
        "OFFSET_MUTE OFFSETC SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER\n"
    )
    # Vs active at iteration 1; density and Q inactive. Every optional data or
    # gradient transformation is disabled for the derivative check.
    values = "0.01 0 0.0 0.0 4 0 0.0 0.0 0.0 99 1 99 99 0 0.0 0.0 0 2 0 0 0.0 0.0 0 0 0.0 1.0 1.0 0 0.0 0\n"
    (directory / "workflow.inp").write_text(header + values, encoding="ascii")


def generate_forward_observed_case(
    directory: Path, *, config: SHFWIGradientConfig | None = None,
    nprocx: int = 1, nprocy: int = 1,
) -> SHFWIGradientConfig:
    config = config or SHFWIGradientConfig()
    _write_common(directory, config)
    direction = config.direction()
    background = config.background_vs()
    target = [base * (1.0 + config.target_fraction * value) for base, value in zip(background, direction)]
    _write_float_grid(directory / "model" / "current.vs", target)
    _write_float_grid(
        directory / "model" / "current.rho",
        [config.density_kg_m3] * config.cell_count,
    )
    _write_parameters(
        directory,
        _records(config, mode=0, grad_form=1, nprocx=nprocx, nprocy=nprocy),
    )
    _write_workflow(directory)
    (directory / "case.json").write_text(
        json.dumps(config.as_metadata() | {"role": "observed_target"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return config


def generate_fwi_case(
    directory: Path,
    *,
    observed_su: Path,
    epsilon_fraction: float,
    grad_form: int,
    config: SHFWIGradientConfig | None = None,
    direction: Sequence[float] | None = None,
    nprocx: int = 1,
    nprocy: int = 1,
) -> SHFWIGradientConfig:
    config = config or SHFWIGradientConfig()
    _write_common(directory, config)
    direction = list(direction) if direction is not None else config.direction()
    if len(direction) != config.cell_count:
        raise ValueError("direction size differs from model grid")
    background = config.background_vs()
    model = [base * (1.0 + epsilon_fraction * value) for base, value in zip(background, direction)]
    _write_float_grid(directory / "model" / "current.vs", model)
    _write_float_grid(
        directory / "model" / "current.rho",
        [config.density_kg_m3] * config.cell_count,
    )
    (directory / "observed_y.su.shot1").write_bytes(observed_su.read_bytes())
    _write_parameters(
        directory,
        _records(
            config, mode=1, grad_form=grad_form,
            nprocx=nprocx, nprocy=nprocy,
        ),
    )
    _write_workflow(directory)
    (directory / "case.json").write_text(
        json.dumps(
            config.as_metadata()
            | {
                "role": "fwi_derivative",
                "epsilon_fraction": epsilon_fraction,
                "grad_form": grad_form,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return config
