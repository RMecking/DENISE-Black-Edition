from __future__ import annotations

import json
import math
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, _parameter_lines
from tests.utilities.fwi_gradient import gaussian_direction


@dataclass(frozen=True)
class PSVFWIGradientConfig:
    nx: int = 80
    ny: int = 72
    dh_m: float = 10.0
    time_s: float = 0.32
    dt_s: float = 0.0004
    dtinv: int = 1
    vp_m_s: float = 3000.0
    vs_m_s: float = 1700.0
    density_kg_m3: float = 1900.0
    source_x_m: float = 200.0
    source_y_m: float = 360.0
    source_frequency_hz: float = 12.0
    source_azimuth_deg: float = 25.0
    receivers_m: tuple[tuple[float, float], ...] = (
        (360.0, 250.0),
        (420.0, 300.0),
        (480.0, 350.0),
        (540.0, 400.0),
        (600.0, 450.0),
    )
    target_x_m: float = 450.0
    target_y_m: float = 350.0
    target_sigma_m: float = 65.0
    direction_x_m: float = 500.0
    direction_y_m: float = 400.0
    direction_sigma_m: float = 80.0
    target_fraction: float = 0.02
    fd_order: int = 8
    absorbing_width_gridpoints: int = 10

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny

    @property
    def samples_per_trace(self) -> int:
        return round(self.time_s / self.dt_s)

    @property
    def receiver_count(self) -> int:
        return len(self.receivers_m)

    def direction(self) -> list[float]:
        return gaussian_direction(
            nx=self.nx,
            ny=self.ny,
            dh_m=self.dh_m,
            center_x_m=self.direction_x_m,
            center_y_m=self.direction_y_m,
            sigma_m=self.direction_sigma_m,
        )

    def target_shape(self) -> list[float]:
        return gaussian_direction(
            nx=self.nx,
            ny=self.ny,
            dh_m=self.dh_m,
            center_x_m=self.target_x_m,
            center_y_m=self.target_y_m,
            sigma_m=self.target_sigma_m,
        )

    def as_metadata(self) -> dict[str, object]:
        result = asdict(self)
        result["receivers_m"] = [list(value) for value in self.receivers_m]
        return result


def baseline_model(config: PSVFWIGradientConfig) -> dict[str, list[float]]:
    return {
        "vp": [config.vp_m_s] * config.cell_count,
        "vs": [config.vs_m_s] * config.cell_count,
        "rho": [config.density_kg_m3] * config.cell_count,
    }


def target_model(config: PSVFWIGradientConfig, active: Sequence[str]) -> dict[str, list[float]]:
    model = baseline_model(config)
    shape = config.target_shape()
    for component in active:
        model[component] = [value * (1.0 + config.target_fraction * weight) for value, weight in zip(model[component], shape)]
    return model


def perturbed_model(
    config: PSVFWIGradientConfig,
    directions: Mapping[str, Sequence[float]],
    epsilon: float,
) -> dict[str, list[float]]:
    model = baseline_model(config)
    for component, direction in directions.items():
        if len(direction) != config.cell_count:
            raise ValueError(f"{component} direction size differs from grid")
        model[component] = [value * (1.0 + epsilon * weight) for value, weight in zip(model[component], direction)]
    if any(value <= 0.0 for values in model.values() for value in values):
        raise ValueError("perturbed physical model is not positive")
    return model


def _centered_normalized(values: Sequence[float]) -> list[float]:
    mean = math.fsum(values) / len(values)
    centered = [value - mean for value in values]
    scale = max(abs(value) for value in centered)
    if scale == 0.0:
        raise ValueError("spatial pattern must vary")
    return [value / scale for value in centered]


def _float32_values(values: Sequence[float]) -> list[float]:
    return array("f", values).tolist()


def heterogeneous_model(config: PSVFWIGradientConfig) -> dict[str, list[float]]:
    """Smooth positive current model with variation across both MPI seams."""
    patterns: dict[str, list[float]] = {name: [] for name in ("vp", "vs", "rho")}
    for ix in range(1, config.nx + 1):
        x = (ix - 0.5) / config.nx
        for iy in range(1, config.ny + 1):
            y = (iy - 0.5) / config.ny
            patterns["vp"].append(
                math.sin(2.0 * math.pi * x) * math.cos(2.0 * math.pi * y)
                + 0.30 * math.cos(4.0 * math.pi * x + math.pi * y)
            )
            patterns["vs"].append(
                math.cos(2.0 * math.pi * x - 0.35) * math.sin(2.0 * math.pi * y)
                + 0.25 * math.sin(math.pi * x + 3.0 * math.pi * y)
            )
            patterns["rho"].append(
                math.sin(2.0 * math.pi * (x + y))
                + 0.35 * math.cos(3.0 * math.pi * x) * math.cos(2.0 * math.pi * y)
            )
    amplitudes = {"vp": 0.06, "vs": 0.07, "rho": 0.05}
    references = {
        "vp": config.vp_m_s,
        "vs": config.vs_m_s,
        "rho": config.density_kg_m3,
    }
    return {
        name: _float32_values([
            references[name] * (1.0 + amplitudes[name] * weight)
            for weight in _centered_normalized(patterns[name])
        ])
        for name in patterns
    }


def heterogeneous_direction(config: PSVFWIGradientConfig) -> dict[str, list[float]]:
    """Joint smooth direction distinct from the homogeneous Gaussian gate."""
    raw: dict[str, list[float]] = {name: [] for name in ("vp", "vs", "rho")}
    for ix in range(1, config.nx + 1):
        x = (ix - 0.5) / config.nx
        for iy in range(1, config.ny + 1):
            y = (iy - 0.5) / config.ny
            boundary_taper = math.sin(math.pi * x) ** 2 * math.sin(math.pi * y) ** 2
            envelope = (
                boundary_taper
                * math.exp(-((x - 0.54) ** 2 + (y - 0.48) ** 2) / 0.16)
            )
            raw["vp"].append(envelope * math.sin(math.pi * x) * math.cos(2.0 * math.pi * y))
            raw["vs"].append(envelope * math.cos(2.0 * math.pi * x + 0.2) * math.sin(math.pi * y))
            raw["rho"].append(envelope * math.sin(2.0 * math.pi * (x - y) + 0.4))
    return {name: _centered_normalized(values) for name, values in raw.items()}


def heterogeneous_target_model(
    config: PSVFWIGradientConfig,
    model: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    shapes = {
        "vp": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=360.0, center_y_m=430.0, sigma_m=95.0,
        ),
        "vs": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=520.0, center_y_m=300.0, sigma_m=85.0,
        ),
        "rho": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=430.0, center_y_m=390.0, sigma_m=105.0,
        ),
    }
    return {
        name: _float32_values([
            value * (1.0 + 0.015 * weight)
            for value, weight in zip(model[name], shapes[name])
        ])
        for name in ("vp", "vs", "rho")
    }


def heterogeneous_perturbed_model(
    model: Mapping[str, Sequence[float]],
    direction: Mapping[str, Sequence[float]],
    epsilon: float,
) -> dict[str, list[float]]:
    result = {
        name: _float32_values([
            value * (1.0 + epsilon * weight)
            for value, weight in zip(model[name], direction[name])
        ])
        for name in ("vp", "vs", "rho")
    }
    if any(value <= 0.0 for values in result.values() for value in values):
        raise ValueError("heterogeneous perturbed model is not positive")
    return result


def _base_config(config: PSVFWIGradientConfig) -> HomogeneousPSVConfig:
    return HomogeneousPSVConfig(
        nx=config.nx,
        ny=config.ny,
        dh_m=config.dh_m,
        time_s=config.time_s,
        dt_s=config.dt_s,
        vp_m_s=config.vp_m_s,
        vs_m_s=config.vs_m_s,
        density_kg_m3=config.density_kg_m3,
        source_x_m=config.source_x_m,
        source_y_m=config.source_y_m,
        source_frequency_hz=config.source_frequency_hz,
        source_type=4,
        source_azimuth_deg=config.source_azimuth_deg,
        receivers_m=config.receivers_m,
        fd_order=config.fd_order,
        absorbing_width_gridpoints=config.absorbing_width_gridpoints,
        damping_velocity_m_s=config.vp_m_s,
        pml_frequency_hz=config.source_frequency_hz,
    )


def _replace(records: list[str], key: str, value: str) -> None:
    indices = [index for index, record in enumerate(records) if record.split("=", 1)[0].strip() == key]
    if len(indices) != 1:
        raise AssertionError(f"expected one {key} record, found {len(indices)}")
    records[indices[0]] = f"{key} ={value}"


def _records(config: PSVFWIGradientConfig, *, mode: int, grad_form: int,
             data_components: int, nprocx: int, nprocy: int) -> list[str]:
    records = _parameter_lines(_base_config(config), nprocx, nprocy)
    overrides = {
        "MODE": str(mode),
        "MFILE": "model/current",
        "SEIS_FORMAT": "1",
        "SEIS_FILE_VX": "su/synthetic_x.su",
        "SEIS_FILE_VY": "su/synthetic_y.su",
        "ITERMAX": "1",
        "JACOBIAN": "jacobian/gradient",
        "DATA_DIR": "observed",
        "INVMAT1": "1",
        "GRAD_FORM": str(grad_form),
        "QUELLTYPB": str(data_components),
        "GRAD_METHOD": "2",
        "NLBFGS": "3",
        "DTINV": str(config.dtinv),
        "INV_MOD_OUT": "0",
        "MODEL_FILTER": "0",
        "GRAD_FILTER": "0",
    }
    for key, value in overrides.items():
        _replace(records, key, value)
    return records


def _write_float_grid(path: Path, values: Sequence[float]) -> None:
    with path.open("wb") as stream:
        array("f", values).tofile(stream)


def _write_workflow(directory: Path) -> None:
    header = (
        "PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ "
        "INV_VP_ITER INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP "
        "WD_DAMP1 EPRECOND LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE "
        "OFFSET_MUTE OFFSETC SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER\n"
    )
    values = "0.01 0 0.0 0.0 4 0 0.0 0.0 0.0 1 1 1 99 0 0.0 0.0 0 2 0 0 0.0 0.0 0 0 0.0 1.0 1.0 0 0.0 0\n"
    (directory / "workflow.inp").write_text(header + values, encoding="ascii")


def generate_case(
    directory: Path,
    *,
    model: Mapping[str, Sequence[float]],
    config: PSVFWIGradientConfig,
    mode: int,
    grad_form: int = 2,
    data_components: int = 1,
    observed_x: Path | None = None,
    observed_y: Path | None = None,
    metadata: Mapping[str, object] | None = None,
    nprocx: int = 1,
    nprocy: int = 1,
) -> None:
    for name in ("model", "su", "log", "snap", "wavelet", "jacobian", "taper", "picked_times", "trace_kill", "gravity", "inverted"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    source = (
        f"1\n{config.source_x_m} 0.0 {config.source_y_m} 0.0 "
        f"{config.source_frequency_hz} 1.0 {config.source_azimuth_deg} 4\n"
    )
    (directory / "source.dat").write_text(source, encoding="ascii")
    (directory / "receiver.dat").write_text(
        "".join(f"{x} {y}\n" for x, y in config.receivers_m), encoding="ascii"
    )
    for component in ("vp", "vs", "rho"):
        values = list(model[component])
        if len(values) != config.cell_count:
            raise ValueError(f"{component} model size differs from grid")
        _write_float_grid(directory / "model" / f"current.{component}", values)
    if mode == 1:
        if observed_x is None or observed_y is None:
            raise ValueError("FWI case requires both observed velocity files")
        (directory / "observed_x.su.shot1").write_bytes(observed_x.read_bytes())
        (directory / "observed_y.su.shot1").write_bytes(observed_y.read_bytes())
    records = _records(config, mode=mode, grad_form=grad_form,
                       data_components=data_components, nprocx=nprocx,
                       nprocy=nprocy)
    text = "# Generated elastic PSV FWI gradient audit case\n" + "".join(
        f"# positional parameter {index:03d}\n{record}\n"
        for index, record in enumerate(records, start=1)
    )
    (directory / "denise.inp").write_text(text, encoding="ascii")
    _write_workflow(directory)
    payload = config.as_metadata() | {"mode": mode, "grad_form": grad_form,
                                      "data_components": data_components,
                                      "nprocx": nprocx, "nprocy": nprocy}
    if metadata:
        payload.update(metadata)
    (directory / "case.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
