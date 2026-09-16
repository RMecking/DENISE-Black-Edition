from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from tests.cases.homogeneous_sh import HomogeneousSHConfig, _parameter_lines
from tests.utilities.fwi_gradient import gaussian_direction
from tests.utilities.visco_sh_fwi_attenuation import (
    QTauMapping,
    physical_q_mapping,
    q_to_tau_and_derivative,
)


Perturbation = Literal["baseline", "q", "tau", "observed"]


@dataclass(frozen=True)
class ViscoSHFWIAttenuationConfig:
    nx: int = 80
    ny: int = 64
    dh_m: float = 10.0
    time_s: float = 0.34
    dt_s: float = 0.0005
    dtinv: int = 1
    vs_m_s: float = 2200.0
    density_kg_m3: float = 1800.0
    baseline_qs: float = 60.0
    observed_qs: float = 35.0
    source_x_m: float = 180.0
    source_y_m: float = 320.0
    source_frequency_hz: float = 14.0
    receiver_x_m: tuple[float, ...] = (360.0, 440.0, 520.0, 600.0)
    receiver_y_m: float = 320.0
    anomaly_x_m: float = 430.0
    anomaly_y_m: float = 320.0
    anomaly_sigma_m: float = 90.0
    fd_order: int = 8
    absorbing_width_gridpoints: int = 10
    relaxation_frequencies_hz: tuple[float, ...] = (7.0, 28.0)
    q_parameterization_mode: int = 1
    q_approx_fmin_hz: float = 4.0
    q_approx_fmax_hz: float = 80.0
    q_approx_df_hz: float = 2.0

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny

    @property
    def samples_per_trace(self) -> int:
        return round(self.time_s / self.dt_s)

    @property
    def receiver_count(self) -> int:
        return len(self.receiver_x_m)

    def direction(self) -> list[float]:
        return gaussian_direction(
            nx=self.nx,
            ny=self.ny,
            dh_m=self.dh_m,
            center_x_m=self.anomaly_x_m,
            center_y_m=self.anomaly_y_m,
            sigma_m=self.anomaly_sigma_m,
        )

    def mapping(self) -> QTauMapping:
        if self.q_parameterization_mode == 0:
            return QTauMapping(mode="legacy")
        if self.q_parameterization_mode != 1:
            raise ValueError("Q parameterization mode must be 0 or 1")
        return physical_q_mapping(
            relaxation_frequencies_hz=self.relaxation_frequencies_hz,
            fmin_hz=self.q_approx_fmin_hz,
            fmax_hz=self.q_approx_fmax_hz,
            df_hz=self.q_approx_df_hz,
        )

    def as_metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["receiver_x_m"] = list(self.receiver_x_m)
        data["relaxation_frequencies_hz"] = list(self.relaxation_frequencies_hz)
        return data


def q_model(
    config: ViscoSHFWIAttenuationConfig,
    *,
    perturbation: Perturbation,
    epsilon: float = 0.0,
) -> list[float]:
    if perturbation == "observed":
        return [config.observed_qs] * config.cell_count
    if perturbation == "baseline":
        return [config.baseline_qs] * config.cell_count
    direction = config.direction()
    if perturbation == "q":
        return [config.baseline_qs * (1.0 + epsilon * value) for value in direction]
    if perturbation != "tau":
        raise ValueError(f"unsupported perturbation: {perturbation}")
    mapping = config.mapping()
    baseline_tau = q_to_tau_and_derivative(config.baseline_qs, mapping)[0]
    result = []
    for value in direction:
        tau = baseline_tau * (1.0 + epsilon * value)
        if mapping.mode == "legacy":
            q_value = 2.0 / tau
        else:
            q_value = (1.0 / tau - mapping.inverse_tau_offset) / mapping.inverse_tau_per_q
        if q_value <= 0.0:
            raise ValueError("tau perturbation produced non-positive Q")
        result.append(q_value)
    return result


def _base_config(config: ViscoSHFWIAttenuationConfig) -> HomogeneousSHConfig:
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
        damping_velocity_m_s=config.vs_m_s,
        pml_frequency_hz=config.source_frequency_hz,
    )


def _write_float_grid(path: Path, values: Sequence[float]) -> None:
    with path.open("wb") as stream:
        array("f", values).tofile(stream)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_workflow(directory: Path) -> None:
    header = (
        "PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ "
        "INV_VP_ITER INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP "
        "WD_DAMP1 EPRECOND LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE "
        "OFFSET_MUTE OFFSETC SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER\n"
    )
    values = (
        "0.01 0 0.0 0.0 4 0 0.0 0.0 0.0 99 99 99 1 0 0.0 0.0 0 2 0 0 "
        "0.0 0.0 0 0 0.0 1.0 1.0 0 0.0 0\n"
    )
    (directory / "workflow.inp").write_text(header + values, encoding="ascii")


def generate_case(
    directory: Path,
    *,
    config: ViscoSHFWIAttenuationConfig,
    perturbation: Perturbation,
    epsilon: float = 0.0,
    free_surface: bool = False,
    nprocx: int = 1,
    nprocy: int = 1,
    dtinv: int | None = None,
    mode: int = 0,
    observed_su: Path | None = None,
) -> ViscoSHFWIAttenuationConfig:
    if config.nx % nprocx or config.ny % nprocy:
        raise ValueError("grid dimensions must be divisible by the decomposition")
    if mode not in (0, 1):
        raise ValueError("only forward and FWI modes are supported")
    if mode == 1 and observed_su is None:
        raise ValueError("FWI mode requires immutable observed data")
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

    model = directory / "model" / "current"
    _write_float_grid(model.with_suffix(".vs"), [config.vs_m_s] * config.cell_count)
    _write_float_grid(model.with_suffix(".rho"), [config.density_kg_m3] * config.cell_count)
    q_values = q_model(config, perturbation=perturbation, epsilon=epsilon)
    _write_float_grid(model.with_suffix(".qs"), q_values)
    if observed_su is not None:
        (directory / "observed_y.su.shot1").write_bytes(observed_su.read_bytes())

    records = _parameter_lines(_base_config(config), nprocx, nprocy)
    overrides = {
        1: f"MODE ={mode}",
        23: "MFILE =model/current",
        25: f" L ={len(config.relaxation_frequencies_hz)}",
        26: "FL =" + " ".join(str(value) for value in config.relaxation_frequencies_hz),
        28: f"FREE_SURF ={int(free_surface)}",
        52: "SEIS_FORMAT =1",
        54: "SEIS_FILE_VY =su/synthetic_y.su",
        60: "ITERMAX =1",
        61: "JACOBIAN =jacobian/gradient",
        62: "DATA_DIR =observed",
        67: "GRAD_FORM =2",
        80: "INV_MOD_OUT =0",
        81: "INV_MODELFILE =inverted/model",
        90: "GRAD_METHOD =1",
        95: f"DTINV ={config.dtinv if dtinv is None else dtinv}",
        97: "STEPMAX =1",
    }
    for position, value in overrides.items():
        records[position - 1] = value
    optional = []
    if config.q_parameterization_mode == 1:
        optional = [
            "Q_PARAMETERIZATION_MODE =1",
            f"Q_APPROX_FMIN ={config.q_approx_fmin_hz}",
            f"Q_APPROX_FMAX ={config.q_approx_fmax_hz}",
            f"Q_APPROX_DF ={config.q_approx_df_hz}",
        ]
    parameters = "# Generated M6.3b viscoelastic SH FWI attenuation oracle case\n" + "".join(
        f"# positional parameter {index:03d}\n{line}\n"
        for index, line in enumerate([*records, *optional], start=1)
    )
    (directory / "denise.inp").write_text(parameters, encoding="ascii")
    _write_workflow(directory)
    mapping = config.mapping()
    tau_values = [q_to_tau_and_derivative(value, mapping)[0] for value in q_values]
    metadata = config.as_metadata() | {
        "milestone": "M6.3b",
        "mode": mode,
        "perturbation": perturbation,
        "epsilon": epsilon,
        "free_surface": int(free_surface),
        "nprocx": nprocx,
        "nprocy": nprocy,
        "dtinv": config.dtinv if dtinv is None else dtinv,
        "q_model_sha256": _sha256(model.with_suffix(".qs")),
        "q_min": min(q_values),
        "q_max": max(q_values),
        "tau_min": min(tau_values),
        "tau_max": max(tau_values),
    }
    (directory / "case.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config
