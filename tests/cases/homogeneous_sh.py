from __future__ import annotations

import json
import math
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HomogeneousSHConfig:
    nx: int = 200
    ny: int = 120
    dh_m: float = 10.0
    time_s: float = 0.55
    dt_s: float = 0.0005
    vs_m_s: float = 2000.0
    density_kg_m3: float = 2000.0
    source_x_m: float = 700.0
    source_y_m: float = 600.0
    source_frequency_hz: float = 10.0
    receiver_x_m: tuple[float, ...] = (900.0, 1000.0, 1100.0, 1200.0, 1300.0)
    receiver_y_m: float = 600.0
    fd_order: int = 8
    max_relative_error: int = 1
    absorbing_width_gridpoints: int = 15
    damping_velocity_m_s: float = 2000.0
    pml_frequency_hz: float = 10.0

    @property
    def samples_per_trace(self) -> int:
        return round(self.time_s / self.dt_s)

    @property
    def receiver_count(self) -> int:
        return len(self.receiver_x_m)

    def analytical_travel_times(self) -> list[float]:
        return [
            math.hypot(x - self.source_x_m, self.receiver_y_m - self.source_y_m) / self.vs_m_s
            for x in self.receiver_x_m
        ]

    def as_metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["receiver_x_m"] = list(self.receiver_x_m)
        return data


def _write_constant_float_model(path: Path, value: float, count: int) -> None:
    values = array("f", [value]) * count
    with path.open("wb") as stream:
        values.tofile(stream)


def _parameter_lines(config: HomogeneousSHConfig, nprocx: int, nprocy: int) -> list[str]:
    # read_par.c consumes non-comment records positionally. Keep this list aligned
    # with cases 1..115 in that function; labels are intentionally descriptive.
    lines = [
        "MODE =0",
        "PHYSICS =5",
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
        "FREE_SURF =0",
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
        "SEIS_FILE_VY =su/homogeneous_vz.asc",
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
    config: HomogeneousSHConfig | None = None,
    nprocx: int = 1,
    nprocy: int = 1,
) -> HomogeneousSHConfig:
    config = config or HomogeneousSHConfig()
    if config.nx % nprocx or config.ny % nprocy:
        raise ValueError("Grid dimensions must be divisible by the MPI decomposition")

    for name in ("model", "su", "log", "snap", "wavelet", "jacobian", "taper", "picked_times", "trace_kill", "gravity"):
        (directory / name).mkdir(parents=True, exist_ok=True)

    cell_count = config.nx * config.ny
    _write_constant_float_model(directory / "model" / "homogeneous.vs", config.vs_m_s, cell_count)
    _write_constant_float_model(directory / "model" / "homogeneous.rho", config.density_kg_m3, cell_count)

    source_line = (
        f"{config.source_x_m} 0.0 {config.source_y_m} 0.0 "
        f"{config.source_frequency_hz} 1.0 0.0 1\n"
    )
    (directory / "source.dat").write_text("1\n" + source_line, encoding="ascii")
    receivers = "".join(f"{x} {config.receiver_y_m}\n" for x in config.receiver_x_m)
    (directory / "receiver.dat").write_text(receivers, encoding="ascii")

    records = _parameter_lines(config, nprocx, nprocy)
    parameters = "# Generated homogeneous elastic SH verification case\n" + "".join(
        f"# positional parameter {index:03d}\n{line}\n" for index, line in enumerate(records, start=1)
    )
    (directory / "denise.inp").write_text(parameters, encoding="ascii")
    (directory / "workflow.inp").write_text("# Unused in MODE=0; argv[2] remains mandatory.\n", encoding="ascii")
    case_metadata = config.as_metadata() | {"nprocx": nprocx, "nprocy": nprocy}
    (directory / "case.json").write_text(
        json.dumps(case_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config
