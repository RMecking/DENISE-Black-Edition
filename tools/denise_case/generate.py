from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import __version__
from .config import CaseConfig, dump_json_yaml
from .model_io import sha256_file


MODEL_COMPONENTS = ("vp", "vs", "rho")


def _copy_exact(source: Path, target: Path) -> None:
    payload = source.read_bytes()
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace non-identical prepared input: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _write_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite non-identical generated file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _json_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def prepare_case(config: CaseConfig) -> dict[str, Any]:
    """Import benchmark inputs byte-for-byte into the ignored local input area."""

    benchmark = config.resolve("provenance.benchmark_checkout")
    if not benchmark.is_dir():
        raise FileNotFoundError(f"frozen benchmark checkout not found: {benchmark}")
    prepared: dict[str, Any] = {"benchmark_checkout": str(benchmark), "files": {}}
    for model_name in ("true", "start_1d", "smooth2"):
        prefix = benchmark / str(config.require(f"models.{model_name}.prefix"))
        for component in MODEL_COMPONENTS:
            source = prefix.with_suffix(f".{component}")
            target = config.root / "input" / "models" / f"{model_name}.{component}"
            _copy_exact(source, target)
            prepared["files"][str(target.relative_to(config.root))] = {
                "source": str(source.relative_to(benchmark)),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
    for name, dotted in (("sources.dat", "acquisition.sources"), ("receivers.dat", "acquisition.receivers")):
        source = benchmark / str(config.require(dotted))
        target = config.root / "input" / "geometry" / name
        _copy_exact(source, target)
        prepared["files"][str(target.relative_to(config.root))] = {
            "source": str(source.relative_to(benchmark)),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    dump_json_yaml(config.root / "input" / "inventory.json", prepared)
    return prepared


def _parameter_records(config: CaseConfig) -> list[str]:
    grid, time_cfg = config.data["grid"], config.data["time"]
    backend, physics = config.data["backend"], config.data["physics"]
    boundary, wavelet = config.data["boundary"], config.data["wavelet"]
    nx, ny = int(grid["nx"]), int(grid["ny"])
    return [
        "MODE =0",
        "PHYSICS =1",
        f"NPROCX ={int(backend['nprocx'])}",
        f"NPROCY ={int(backend['nprocy'])}",
        f"FD_ORDER ={int(physics['fd_order'])}",
        f"MAX_RELATIVE_ERROR ={int(physics['max_relative_error'])}",
        f"NX ={nx}",
        f"NY ={ny}",
        f"DH ={float(grid['dh_m'])}",
        f"TIME ={float(time_cfg['time_s'])}",
        f"DT ={float(time_cfg['dt_s'])}",
        f"QUELLART ={int(wavelet['type'])}",
        "SIGNAL_FILE =wavelet/source",
        f"TS ={float(wavelet.get('klauder_duration_s', 8.0))}",
        "SRCREC =1",
        "SOURCE_FILE =source/sources.dat",
        "RUN_MULTIPLE_SHOTS =1",
        f"FC_SPIKE_1 ={float(wavelet['highpass_hz'])}",
        f"FC_SPIKE_2 ={float(wavelet['lowpass_hz'])}",
        f"ORDER_SPIKE ={int(wavelet['filter_order'])}",
        "WRITE_STF =1",
        "READMOD =1",
        "MFILE =model/true",
        "WRITEMOD =0",
        "L =0",
        "FL =10.0",
        "TAU =0.0",
        f"FREE_SURF ={int(bool(boundary['free_surface']))}",
        f"FW ={int(boundary['absorbing_width_gridpoints'])}",
        f"DAMPING ={float(boundary['damping_velocity_m_s'])}",
        f"FPML ={float(boundary['pml_frequency_hz'])}",
        f"NPOWER ={float(boundary.get('npower', 4.0))}",
        f"K_MAX_PML ={float(boundary.get('k_max_pml', 1.0))}",
        "BOUNDARY =0",
        "SNAP =0",
        "SNAP_SHOT =1",
        "TSNAP1 =0.002",
        "TSNAP2 =3.0",
        "TSNAPINC =0.06",
        "IDX =1",
        "IDY =1",
        "SNAP_FORMAT =3",
        "SNAP_FILE =snap/forward",
        "SEISMO =1",
        "READREC =1",
        "REC_FILE =receiver/receivers",
        "REFREC =0.0,0.0",
        "N_STREAMER =0",
        "REC_INCR_X =0.0",
        "REC_INCR_Y =0.0",
        f"NDT ={int(time_cfg.get('sample_stride', 1))}",
        "SEIS_FORMAT =1",
        "SEIS_FILE_VX =data/observed_vx.su",
        "SEIS_FILE_VY =data/observed_vy.su",
        "SEIS_FILE_CURL =data/unused_curl.su",
        "SEIS_FILE_DIV =data/unused_div.su",
        "SEIS_FILE_P =data/unused_pressure.su",
        "LOG_FILE =log/denise.log",
        "LOG =1",
        "ITERMAX =1",
        "JACOBIAN =jacobian/unused",
        "DATA_DIR =data/unused",
        "TAPER =0",
        "TAPERLENGTH =1",
        f"GRADT =1,1,{nx},{nx}",
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
        "VPUPPERLIM =6000.0",
        "VPLOWERLIM =0.0",
        "VSUPPERLIM =4000.0",
        "VSLOWERLIM =0.0",
        "RHOUPPERLIM =3000.0",
        "RHOLOWERLIM =1000.0",
        "QSUPPERLIM =1000.0",
        "QSLOWERLIM =1.0",
        "GRAD_METHOD =2",
        "PCG_BETA =1",
        "NLBFGS =20",
        "MODEL_FILTER =0",
        "FILT_SIZE =5",
        "DTINV =3",
        "EPS_SCALE =0.01",
        "STEPMAX =6",
        "SCALEFAC =2.0",
        "TRKILL =0",
        "TRKILL_FILE =trace_kill/unused",
        "PICKS_FILE =picked_times/unused",
        "MISFIT_LOG_FILE =misfit.log",
        "MIN_ITER =0",
        "GRAD_FILTER =0",
        "FILT_SIZE_GRAD =10",
        "TIMELAPSE =0",
        "DATA_DIR_T0 =data/unused_t0",
        "RTMOD =0",
        "GRAVITY =0",
        f"NGRAVB ={nx}",
        "NZGRAV =0",
        "GRAV_TYPE =1",
        "BACK_DENS =2",
        "DFILE =gravity/unused",
        "RTM_SHOT =0",
    ]


def _generator_identity(base_tree_sha: str) -> dict[str, Any]:
    package = Path(__file__).parent
    return {
        "version": __version__,
        "base_tree_sha": base_tree_sha,
        "source_sha256": {
            f"tools/denise_case/{path.name}": sha256_file(path)
            for path in sorted(package.glob("*.py"))
        },
    }


def materialize_run(config: CaseConfig, run_name: str = "true_forward") -> Path:
    """Create an idempotent, provenance-rich current-DENISE forward directory."""

    records = _parameter_records(config)
    if len(records) != 115:
        raise AssertionError(f"current DENISE positional input has {len(records)} records, expected 115")
    run = config.root / "generated" / run_name
    for name in ("model", "source", "receiver", "data", "log", "snap", "wavelet", "jacobian", "taper", "trace_kill", "picked_times", "gravity"):
        (run / name).mkdir(parents=True, exist_ok=True)
    for component in MODEL_COMPONENTS:
        _copy_exact(config.root / "input" / "models" / f"true.{component}", run / "model" / f"true.{component}")
    _copy_exact(config.root / "input" / "geometry" / "sources.dat", run / "source" / "sources.dat")
    _copy_exact(config.root / "input" / "geometry" / "receivers.dat", run / "receiver" / "receivers.dat")
    parameter_text = "# Generated by tools.denise_case; positional records follow.\n" + "".join(
        f"# record {index:03d}\n{line}\n" for index, line in enumerate(records, start=1)
    )
    _write_exact(run / "denise.inp", parameter_text.encode("ascii"))
    _write_exact(run / "workflow.inp", b"# Unused in MODE=0; argv[2] remains mandatory.\n")
    models = {
        component: {"path": f"model/true.{component}", "sha256": sha256_file(run / "model" / f"true.{component}")}
        for component in MODEL_COMPONENTS
    }
    executable_relative = os.path.relpath(config.resolve("backend.executable"), run).replace("\\", "/")
    provenance = {
        "case_id": config.require("case.id"),
        "application_manifest": {"path": "case.yaml", "sha256": config.sha256},
        "generator": _generator_identity(str(config.require("provenance.denise_core_sha"))),
        "denise_core_sha": config.require("provenance.denise_core_sha"),
        "benchmark_source_sha": config.require("provenance.benchmark_sha"),
        "models": models,
        "geometry": {
            "sources": {"path": "source/sources.dat", "sha256": sha256_file(run / "source" / "sources.dat")},
            "receivers": {"path": "receiver/receivers.dat", "sha256": sha256_file(run / "receiver" / "receivers.dat")},
        },
        "resolved_parameter_sha256": hashlib.sha256(parameter_text.encode("ascii")).hexdigest(),
        "workflow_sha256": sha256_file(run / "workflow.inp"),
        "command": [str(config.require("backend.launcher")), "-np", str(config.require("backend.ranks")), executable_relative, "denise.inp", "workflow.inp"],
        "environment": {"backend": config.require("backend.environment")},
    }
    _write_exact(run / "resolved_case.json", config.canonical_bytes())
    _write_exact(run / "provenance.json", _json_bytes(provenance))
    return run
