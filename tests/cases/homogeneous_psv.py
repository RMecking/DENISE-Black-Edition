from __future__ import annotations

import json
import math
from array import array
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class HomogeneousPSVConfig:
    nx: int = 200
    ny: int = 200
    dh_m: float = 10.0
    time_s: float = 0.55
    dt_s: float = 0.0004
    vp_m_s: float = 3000.0
    vs_m_s: float = 1800.0
    density_kg_m3: float = 2000.0
    source_x_m: float = 1000.0
    source_y_m: float = 1000.0
    source_frequency_hz: float = 10.0
    source_type: int = 1
    source_azimuth_deg: float = 0.0
    receivers_m: tuple[tuple[float, float], ...] = (
        (1200.0, 1000.0),
        (1300.0, 1000.0),
        (1400.0, 1000.0),
        (1500.0, 1000.0),
        (1600.0, 1000.0),
    )
    fd_order: int = 8
    max_relative_error: int = 1
    absorbing_width_gridpoints: int = 15
    damping_velocity_m_s: float = 3000.0
    pml_frequency_hz: float = 10.0
    free_surface: bool = False

    @property
    def samples_per_trace(self) -> int:
        return round(self.time_s / self.dt_s)

    @property
    def receiver_count(self) -> int:
        return len(self.receivers_m)

    @property
    def courant_number(self) -> float:
        return self.vp_m_s * self.dt_s / self.dh_m

    @property
    def conservative_s_wavelength_points(self) -> float:
        shortest_wavelength = self.vs_m_s / (2.5 * self.source_frequency_hz)
        return shortest_wavelength / self.dh_m

    def receiver_offsets_m(self) -> list[float]:
        return [
            math.hypot(x - self.source_x_m, y - self.source_y_m)
            for x, y in self.receivers_m
        ]

    def analytical_travel_times(self, velocity_m_s: float) -> list[float]:
        return [offset / velocity_m_s for offset in self.receiver_offsets_m()]

    def as_metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["receivers_m"] = [list(receiver) for receiver in self.receivers_m]
        data["courant_number"] = self.courant_number
        data["conservative_s_wavelength_points"] = self.conservative_s_wavelength_points
        return data


def with_geometry(
    config: HomogeneousPSVConfig,
    *,
    source_m: tuple[float, float] | None = None,
    receivers_m: tuple[tuple[float, float], ...] | None = None,
    source_type: int | None = None,
    source_azimuth_deg: float | None = None,
) -> HomogeneousPSVConfig:
    changes: dict[str, object] = {}
    if source_m is not None:
        changes.update(source_x_m=source_m[0], source_y_m=source_m[1])
    if receivers_m is not None:
        changes["receivers_m"] = receivers_m
    if source_type is not None:
        changes["source_type"] = source_type
    if source_azimuth_deg is not None:
        changes["source_azimuth_deg"] = source_azimuth_deg
    return replace(config, **changes)


def _write_constant_float_model(path: Path, value: float, count: int) -> None:
    values = array("f", [value]) * count
    with path.open("wb") as stream:
        values.tofile(stream)


def _parameter_lines(config: HomogeneousPSVConfig, nprocx: int, nprocy: int) -> list[str]:
    # read_par.c consumes non-comment records positionally. Keep this aligned
    # with cases 1..115; labels are descriptive rather than parsed as keys.
    lines = [
        "MODE =0",
        "PHYSICS =1",
        f"NPROCX ={nprocx}",
        f"NPROCY ={nprocy}",
        f"FD_ORDER ={config.fd_order}",
        f"MAX_RELATIVE_ERROR ={config.max_relative_error}",
        f"NX ={config.nx}",
        f"NY ={config.ny}",
        f"DH ={config.dh_m}",
        f"TIME ={config.time_s}",
        f"DT ={config.dt_s}",
        "QUELLART =1",
        "SIGNAL_FILE =wavelet/source",
        "TS =0.1",
        "SRCREC =1",
        "SOURCE_FILE =source.dat",
        "RUN_MULTIPLE_SHOTS =1",
        "FC_SPIKE_1 =1.0",
        "FC_SPIKE_2 =20.0",
        "ORDER_SPIKE =4",
        "WRITE_STF =0",
        "READMOD =1",
        "MFILE =model/homogeneous",
        "WRITEMOD =0",
        "L =0",
        "FL =10.0",
        "TAU =0.0",
        f"FREE_SURF ={int(config.free_surface)}",
        f"FW ={config.absorbing_width_gridpoints}",
        f"DAMPING ={config.damping_velocity_m_s}",
        f"FPML ={config.pml_frequency_hz}",
        "NPOWER =2.0",
        "K_MAX_PML =1.0",
        "BOUNDARY =0",
        "SNAP =0",
        "SNAP_SHOT =1",
        "TSNAP1 =0.05",
        "TSNAP2 =0.50",
        "TSNAPINC =0.05",
        "IDX =1",
        "IDY =1",
        "SNAP_FORMAT =3",
        "SNAP_FILE =snap/homogeneous",
        "SEISMO =1",
        "READREC =1",
        "REC_FILE =receiver",
        "REFREC =0.0,0.0",
        "N_STREAMER =0",
        "REC_INCR_X =0.0",
        "REC_INCR_Y =0.0",
        "NDT =1",
        "SEIS_FORMAT =2",
        "SEIS_FILE_VX =su/homogeneous_vx.asc",
        "SEIS_FILE_VY =su/homogeneous_vy.asc",
        "SEIS_FILE_CURL =su/homogeneous_curl.asc",
        "SEIS_FILE_DIV =su/homogeneous_div.asc",
        "SEIS_FILE_P =su/homogeneous_p.asc",
        "LOG_FILE =log/denise.log",
        "LOG =1",
        "ITERMAX =1",
        "JACOBIAN =jacobian/unused",
        "DATA_DIR =su/unused",
        "TAPER =0",
        "TAPERLENGTH =1",
        f"GRADT =1,1,{config.nx},{config.nx}",
        "INVMAT1 =1",
        "GRAD_FORM =1",
        "QUELLTYPB =1",
        "TESTSHOTS =1,1,1",
        "SWS_TAPER_GRAD_VERT =0",
        "SWS_TAPER_GRAD_HOR =0",
        "EXP_TAPER_GRAD_HOR =0.0",
        "SWS_TAPER_GRAD_SOURCES =0",
        "SWS_TAPER_CIRCULAR_PER_SHOT =0",
        "SRTSHAPE =1",
        "SRTRADIUS =50.0",
        "FILTSIZE =1",
        "SWS_TAPER_FILE =0",
        "TFILE =taper/unused",
        "INV_MOD_OUT =0",
        "INV_MODELFILE =model/unused",
        "VPUPPERLIM =5000.0",
        "VPLOWERLIM =0.0",
        "VSUPPERLIM =4000.0",
        "VSLOWERLIM =0.0",
        "RHOUPPERLIM =4000.0",
        "RHOLOWERLIM =500.0",
        "QSUPPERLIM =1000.0",
        "QSLOWERLIM =1.0",
        "GRAD_METHOD =1",
        "PCG_BETA =1",
        "NLBFGS =1",
        "MODEL_FILTER =0",
        "FILT_SIZE =1",
        "DTINV =1",
        "EPS_SCALE =0.01",
        "STEPMAX =1",
        "SCALEFAC =2.0",
        "TRKILL =0",
        "TRKILL_FILE =trace_kill/unused",
        "PICKS_FILE =picked_times/unused",
        "MISFIT_LOG_FILE =misfit.log",
        "MIN_ITER =0",
        "GRAD_FILTER =0",
        "FILT_SIZE_GRAD =1",
        "TIMELAPSE =0",
        "DATA_DIR_T0 =su/unused_t0",
        "RTMOD =0",
        "GRAVITY =0",
        f"NGRAVB ={config.nx}",
        "NZGRAV =0",
        "GRAV_TYPE =1",
        "BACK_DENS =1",
        "DFILE =gravity/unused",
        "RTM_SHOT =0",
    ]
    if len(lines) != 115:
        raise AssertionError(f"DENISE parameter template has {len(lines)} records, expected 115")
    return lines


def generate_case(
    directory: Path,
    *,
    config: HomogeneousPSVConfig | None = None,
    nprocx: int = 1,
    nprocy: int = 1,
) -> HomogeneousPSVConfig:
    config = config or HomogeneousPSVConfig()
    if config.nx % nprocx or config.ny % nprocy:
        raise ValueError("Grid dimensions must be divisible by the MPI decomposition")
    if config.source_type not in (1, 2, 3, 4):
        raise ValueError("P/SV verification supports source types 1 through 4")

    for name in (
        "model", "su", "log", "snap", "wavelet", "jacobian", "taper",
        "picked_times", "trace_kill", "gravity",
    ):
        (directory / name).mkdir(parents=True, exist_ok=True)

    cell_count = config.nx * config.ny
    model = directory / "model" / "homogeneous"
    _write_constant_float_model(model.with_suffix(".vp"), config.vp_m_s, cell_count)
    _write_constant_float_model(model.with_suffix(".vs"), config.vs_m_s, cell_count)
    _write_constant_float_model(model.with_suffix(".rho"), config.density_kg_m3, cell_count)

    source_line = (
        f"{config.source_x_m} 0.0 {config.source_y_m} 0.0 "
        f"{config.source_frequency_hz} 1.0 {config.source_azimuth_deg} {config.source_type}\n"
    )
    (directory / "source.dat").write_text("1\n" + source_line, encoding="ascii")
    receivers = "".join(f"{x} {y}\n" for x, y in config.receivers_m)
    (directory / "receiver.dat").write_text(receivers, encoding="ascii")

    records = _parameter_lines(config, nprocx, nprocy)
    parameters = "# Generated homogeneous elastic P/SV verification case\n" + "".join(
        f"# positional parameter {index:03d}\n{line}\n"
        for index, line in enumerate(records, start=1)
    )
    (directory / "denise.inp").write_text(parameters, encoding="ascii")
    (directory / "workflow.inp").write_text(
        "# Unused in MODE=0; argv[2] remains mandatory.\n", encoding="ascii"
    )
    case_metadata = config.as_metadata() | {"nprocx": nprocx, "nprocy": nprocy}
    (directory / "case.json").write_text(
        json.dumps(case_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config
