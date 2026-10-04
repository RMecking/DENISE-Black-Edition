from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import CaseConfig
from .geometry import geometry_issues, read_receivers, read_sources
from .model_io import ModelIOError, read_grid, sha256_file


HOLBERG = {
    0: {4: (9.0 / 8.0, -1.0 / 24.0)},
    1: {
        2: (1.001,), 4: (1.1382, -0.046414), 6: (1.1965, -0.078804, 0.0081781),
        8: (1.2257, -0.099537, 0.018063, -0.0026274),
        10: (1.2415, -0.11231, 0.026191, -0.0064682, 0.001191),
        12: (1.2508, -0.12034, 0.032131, -0.010142, 0.0029857, -0.00066667),
    }
}
GRIDPOINTS = {0: {4: 8.0}, 1: {2: 49.7, 4: 8.32, 6: 4.77, 8: 3.69, 10: 3.19, 12: 2.91}}


@dataclass
class ValidationReport:
    case_id: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "ok": self.ok}


def _git_sha(repository: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repository.as_posix()}", "rev-parse", "HEAD"], cwd=repository, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def validate_case(config: CaseConfig, *, require_executable: bool = True) -> ValidationReport:
    report = ValidationReport(str(config.require("case.id")))
    nx, ny, dh = int(config.require("grid.nx")), int(config.require("grid.ny")), float(config.require("grid.dh_m"))
    report.facts["grid"] = {"nx": nx, "ny": ny, "dh_m": dh, "extent_m": [nx * dh, ny * dh]}
    repository = config.root.parents[1]
    live_sha = _git_sha(repository)
    expected_sha = str(config.require("provenance.denise_core_sha"))
    report.facts["denise_core"] = {"expected_sha": expected_sha, "live_sha": live_sha}
    if live_sha is not None and live_sha != expected_sha:
        report.errors.append(f"live DENISE SHA {live_sha} does not match manifest {expected_sha}")

    model_facts: dict[str, object] = {}
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for model_name in ("true", "start_1d", "smooth2"):
        model_facts[model_name] = {}
        for component in ("vp", "vs", "rho"):
            path = config.root / "input" / "models" / f"{model_name}.{component}"
            try:
                values = read_grid(path, nx, ny)
            except (FileNotFoundError, ModelIOError) as error:
                report.errors.append(str(error))
                continue
            arrays[(model_name, component)] = values
            stats = {
                "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
                "min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()),
                "finite": bool(np.isfinite(values).all()),
            }
            model_facts[model_name][component] = stats
            if not stats["finite"]:
                report.errors.append(f"{path} contains non-finite values")
            if component in ("vp", "rho") and stats["min"] <= 0:
                report.errors.append(f"{path} must be strictly positive")
            if component == "vs" and stats["min"] < 0:
                report.errors.append(f"{path} contains negative shear velocity")
    report.facts["models"] = model_facts
    if ("true", "vp") in arrays and ("true", "vs") in arrays:
        if np.any(arrays[("true", "vp")] < arrays[("true", "vs")]):
            report.errors.append("true Vp is smaller than Vs in at least one cell")

    try:
        sources = read_sources(config.root / "input" / "geometry" / "sources.dat")
        receivers = read_receivers(config.root / "input" / "geometry" / "receivers.dat")
        errors, warnings = geometry_issues(
            sources, receivers, nx=nx, ny=ny, dh_m=dh,
            absorbing_width=int(config.require("boundary.absorbing_width_gridpoints")),
            free_surface=bool(config.require("boundary.free_surface")),
            require_grid_alignment=bool(config.get("acquisition.require_grid_alignment", True)),
        )
        report.errors.extend(errors)
        report.warnings.extend(warnings)
        report.facts["geometry"] = {
            "source_count": len(sources), "receiver_count": len(receivers),
            "source_bounds_m": [min(s.x_m for s in sources), max(s.x_m for s in sources), min(s.y_m for s in sources), max(s.y_m for s in sources)],
            "receiver_bounds_m": [min(r.x_m for r in receivers), max(r.x_m for r in receivers), min(r.y_m for r in receivers), max(r.y_m for r in receivers)],
        }
    except (FileNotFoundError, ValueError) as error:
        report.errors.append(str(error))

    dt = float(config.require("time.dt_s"))
    duration = float(config.require("time.time_s"))
    samples = duration / dt
    if not math.isclose(samples, round(samples), rel_tol=0.0, abs_tol=1e-9):
        report.errors.append(f"TIME/DT is not integral: {samples}")
    fd_order = int(config.require("physics.fd_order"))
    error_class = int(config.require("physics.max_relative_error"))
    vp = arrays.get(("true", "vp"))
    vs = arrays.get(("true", "vs"))
    if vp is not None and vs is not None and error_class in HOLBERG and fd_order in HOLBERG[error_class]:
        vmax = float(max(vp.max(), vs.max()))
        gamma = sum(abs(value) for value in HOLBERG[error_class][fd_order])
        cfl_dt_max = dh / (math.sqrt(2.0) * gamma * vmax)
        nonzero_min = min(float(vp[vp > 0].min()), float(vs[vs > 0].min()))
        dispersion_fmax = nonzero_min / (GRIDPOINTS[error_class][fd_order] * dh)
        report.facts["time_and_accuracy"] = {
            "time_s": duration, "dt_s": dt, "steps": int(round(samples)),
            "cfl_dt_max_s": cfl_dt_max, "cfl_fraction": dt / cfl_dt_max,
            "minimum_nonzero_velocity_m_s": nonzero_min,
            "dispersion_fmax_hz": dispersion_fmax,
            "source_lowpass_hz": float(config.require("wavelet.lowpass_hz")),
        }
        if dt >= cfl_dt_max:
            report.errors.append(f"DT={dt} violates current CFL estimate {cfl_dt_max:.6g} s")
        if float(config.require("wavelet.lowpass_hz")) > dispersion_fmax:
            report.warnings.append(
                f"{float(config.require('wavelet.lowpass_hz')):g} Hz historical spike low-pass exceeds the current FD{fd_order}/error-class-{error_class} dispersion estimate {dispersion_fmax:.3f} Hz; acquisition was preserved, not silently altered"
            )

    nprocx, nprocy = int(config.require("backend.nprocx")), int(config.require("backend.nprocy"))
    if nx % nprocx or ny % nprocy:
        report.errors.append(f"grid {nx}x{ny} is not divisible by MPI layout {nprocx}x{nprocy}")
    if int(config.require("backend.ranks")) != nprocx * nprocy:
        report.errors.append("backend.ranks does not equal nprocx*nprocy")
    executable = config.resolve("backend.executable")
    report.facts["backend"] = {
        "executable": str(executable), "exists": executable.is_file(),
        "launcher": str(config.require("backend.launcher")), "ranks": int(config.require("backend.ranks")),
        "environment": str(config.require("backend.environment")),
    }
    if require_executable and not executable.is_file():
        report.errors.append(f"DENISE executable does not exist: {executable}")
    elif executable.is_file() and os.name == "nt" and executable.suffix.lower() != ".exe":
        report.warnings.append("selected DENISE executable is a POSIX binary; execute the CLI inside WSL/Linux")
    output = config.root / "generated"
    try:
        output.mkdir(parents=True, exist_ok=True)
        probe = output / ".write-probe"
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as error:
        report.errors.append(f"output directory is not writable: {output}: {error}")
    return report


def write_validation(report: ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
