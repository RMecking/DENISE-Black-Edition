#!/usr/bin/env python3
"""Measure single-node strong scaling of the blocking P/SV MPI path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from array import array
from dataclasses import replace
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case


LAYOUTS = {"1x1": (1, 1), "2x1": (2, 1), "1x2": (1, 2), "2x2": (2, 2)}
INTERLEAVED_ORDER = ("1x1_early", "2x2", "2x1", "1x2", "1x1_late")
PHASES = (
    "timestep_loop",
    "velocity_update",
    "velocity_exchange",
    "stress_update",
    "stress_exchange",
)


def _capture(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git(*arguments: str) -> str:
    git_entry = REPOSITORY_ROOT / ".git"
    command = ["git"]
    if git_entry.is_file():
        marker = "gitdir: "
        git_dir_text = git_entry.read_text(encoding="utf-8").strip()
        if not git_dir_text.startswith(marker):
            raise RuntimeError(f"unexpected worktree git file: {git_dir_text}")
        git_dir = git_dir_text[len(marker) :]
        if len(git_dir) >= 3 and git_dir[1:3] == ":/":
            git_dir = f"/mnt/{git_dir[0].lower()}/{git_dir[3:]}"
        command.extend(
            [
                f"--git-dir={git_dir}",
                f"--work-tree={REPOSITORY_ROOT}",
            ]
        )
    result = subprocess.run(
        [*command, *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def _write_constant(path: Path, value: float, count: int) -> None:
    with path.open("wb") as stream:
        (array("f", [value]) * count).tofile(stream)


def _make_case(
    root: Path,
    workload: str,
    measurement_label: str,
    layout: str,
    *,
    config: HomogeneousPSVConfig,
) -> dict[str, object]:
    nprocx, nprocy = LAYOUTS[layout]
    directory = root / workload / measurement_label
    if directory.exists():
        raise FileExistsError(f"Benchmark case already exists: {directory}")
    generate_case(directory, config=config, nprocx=nprocx, nprocy=nprocy)
    viscoelastic = workload == "visco_psv_l1"
    if viscoelastic:
        parameter_file = directory / "denise.inp"
        text = parameter_file.read_text(encoding="ascii")
        old = "\nL =0\n"
        assert text.count(old) == 1
        # read_par consumes the first character of a non-comment record.
        parameter_file.write_text(text.replace(old, "\n L =1\n"), encoding="ascii")
        count = config.nx * config.ny
        prefix = directory / "model" / "homogeneous"
        _write_constant(prefix.with_suffix(".qp"), 60.0, count)
        _write_constant(prefix.with_suffix(".qs"), 40.0, count)
    return {
        "directory": str(directory),
        "measurement_label": measurement_label,
        "layout": layout,
        "ranks": nprocx * nprocy,
        "decomposition": [nprocx, nprocy],
        "global_grid": [config.nx, config.ny],
        "local_grid": [config.nx // nprocx, config.ny // nprocy],
        "timesteps": config.samples_per_trace,
        "fd_order": config.fd_order,
        "L": 1 if viscoelastic else 0,
        "m8b_fd8_l1_fast_path": bool(viscoelastic and config.fd_order == 8),
    }


def _trace_hashes(directory: Path) -> dict[str, str]:
    return {
        component: hashlib.sha256(
            (directory / "su" / f"homogeneous_v{component}.asc.shot1").read_bytes()
        ).hexdigest()
        for component in ("x", "y")
    }


def _run(
    binary: Path,
    mpiexec: str,
    mpiexec_flags: list[str],
    case: dict[str, object],
    *,
    timing: bool,
    timeout_seconds: float,
) -> tuple[float, dict[str, object] | None]:
    directory = Path(str(case["directory"]))
    timing_path = directory / "psv_mpi_timing.json"
    command = [
        mpiexec,
        *mpiexec_flags,
        "-np",
        str(case["ranks"]),
        str(binary),
        "denise.inp",
        "workflow.inp",
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    if timing:
        environment["DENISE_PSV_MPI_TIMING_FILE"] = str(timing_path)
    else:
        environment.pop("DENISE_PSV_MPI_TIMING_FILE", None)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=directory,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    wall = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-4000:]}"
        )
    report = None
    if timing:
        report = json.loads(timing_path.read_text(encoding="utf-8"))
        if report["schema"] != "denise.psv_mpi_timing.v2":
            raise RuntimeError("benchmark requires the rank-consistent timing schema v2")
        if report["ranks"] != case["ranks"]:
            raise RuntimeError("timing report rank count does not match the case")
    return wall, report


def _interior_fraction(nx: int, ny: int, nprocx: int, nprocy: int, radius: int) -> float:
    local_nx, local_ny = nx // nprocx, ny // nprocy
    total = nx * ny
    interior = 0
    for px in range(nprocx):
        for py in range(nprocy):
            width = local_nx - radius * int(px > 0) - radius * int(px + 1 < nprocx)
            height = local_ny - radius * int(py > 0) - radius * int(py + 1 < nprocy)
            interior += max(width, 0) * max(height, 0)
    return interior / total


def _summarize_case(
    case: dict[str, object],
    *,
    warmup_external_seconds: list[float],
    individual_runs: list[dict[str, object]],
    normal_hashes: dict[str, str],
    instrumented_hashes: dict[str, str],
) -> dict[str, object]:
    component_samples = {phase: [] for phase in PHASES}
    critical_samples = {name: [] for name in ("timestep_loop", "communication", "compute")}
    critical_comm_v_samples = []
    critical_comm_s_samples = []
    external_samples = []
    critical_fraction_samples = []
    legacy_fraction_samples = []
    for run in individual_runs:
        external_samples.append(run["external_seconds"])
        components = run["component_wise_rank_maxima_seconds"]
        critical = run["rank_consistent_critical_seconds"]
        for phase in PHASES:
            component_samples[phase].append(components[phase])
        for name in critical_samples:
            critical_samples[name].append(critical[name])
        critical_fraction_samples.append(critical["communication"] / critical["timestep_loop"])
        legacy_fraction_samples.append(
            (components["velocity_exchange"] + components["stress_exchange"])
            / critical["timestep_loop"]
        )
        communication_rank = run["critical_rank_ids"]["communication"]
        communication_record = next(
            entry for entry in run["rank_timings"] if entry["rank"] == communication_rank
        )
        critical_comm_v_samples.append(communication_record["seconds"]["velocity_exchange"])
        critical_comm_s_samples.append(communication_record["seconds"]["stress_exchange"])

    component_medians = {
        phase: statistics.median(values) for phase, values in component_samples.items()
    }
    critical_medians = {
        name: statistics.median(values) for name, values in critical_samples.items()
    }
    total = critical_medians["timestep_loop"]
    result = {
        **case,
        "warmup_external_seconds": warmup_external_seconds,
        "individual_runs": individual_runs,
        "external_median_seconds": statistics.median(external_samples),
        "component_wise_rank_maxima_samples_seconds": component_samples,
        "component_wise_rank_maxima_median_seconds": component_medians,
        "rank_consistent_critical_samples_seconds": critical_samples,
        "rank_consistent_critical_median_seconds": critical_medians,
        "critical_rank_communication_fraction_samples": critical_fraction_samples,
        "critical_rank_communication_fraction": critical_medians["communication"] / total,
        "sum_of_component_maxima_fraction_samples": legacy_fraction_samples,
        "sum_of_component_maxima_fraction": (
            component_medians["velocity_exchange"] + component_medians["stress_exchange"]
        )
        / total,
        "critical_communication_rank_component_samples_seconds": {
            "velocity_exchange": critical_comm_v_samples,
            "stress_exchange": critical_comm_s_samples,
        },
        "critical_communication_rank_component_median_seconds": {
            "velocity_exchange": statistics.median(critical_comm_v_samples),
            "stress_exchange": statistics.median(critical_comm_s_samples),
        },
        "minimum_timestep_loop_seconds": min(critical_samples["timestep_loop"]),
        "maximum_timestep_loop_seconds": max(critical_samples["timestep_loop"]),
        "relative_timestep_loop_range": (
            max(critical_samples["timestep_loop"]) - min(critical_samples["timestep_loop"])
        )
        / total,
        "normal_trace_sha256": normal_hashes,
        "instrumented_trace_sha256": instrumented_hashes,
        "instrumentation_bit_identical": instrumented_hashes == normal_hashes,
    }
    if not result["instrumentation_bit_identical"]:
        raise RuntimeError("timing instrumentation changed P/SV traces")
    return result


def _measure_case(
    binary: Path,
    mpiexec: str,
    mpiexec_flags: list[str],
    case: dict[str, object],
    *,
    warmups: int,
    repetitions: int,
    timeout_seconds: float,
) -> dict[str, object]:
    _, _ = _run(
        binary,
        mpiexec,
        mpiexec_flags,
        case,
        timing=False,
        timeout_seconds=timeout_seconds,
    )
    normal_hashes = _trace_hashes(Path(str(case["directory"])))
    warmup_walls = []
    for _ in range(warmups):
        wall, _ = _run(
            binary,
            mpiexec,
            mpiexec_flags,
            case,
            timing=True,
            timeout_seconds=timeout_seconds,
        )
        warmup_walls.append(wall)
    instrumented_hashes = _trace_hashes(Path(str(case["directory"])))

    individual_runs = []
    for repetition in range(repetitions):
        wall, report = _run(
            binary,
            mpiexec,
            mpiexec_flags,
            case,
            timing=True,
            timeout_seconds=timeout_seconds,
        )
        assert report is not None
        individual_runs.append(
            {
                "repetition": repetition + 1,
                "external_seconds": wall,
                "component_wise_rank_maxima_seconds": report[
                    "component_wise_rank_maxima_seconds"
                ],
                "rank_consistent_critical_seconds": report[
                    "rank_consistent_critical_seconds"
                ],
                "critical_rank_ids": report["critical_rank_ids"],
                "rank_timings": report["rank_timings"],
            }
        )
    final_instrumented_hashes = _trace_hashes(Path(str(case["directory"])))
    if final_instrumented_hashes != instrumented_hashes:
        raise RuntimeError("P/SV traces changed between instrumented repetitions")
    return _summarize_case(
        case,
        warmup_external_seconds=warmup_walls,
        individual_runs=individual_runs,
        normal_hashes=normal_hashes,
        instrumented_hashes=final_instrumented_hashes,
    )


def _combine_references(early: dict[str, object], late: dict[str, object]) -> dict[str, object]:
    if early["normal_trace_sha256"] != late["normal_trace_sha256"]:
        raise RuntimeError("early and late 1x1 reference traces differ")
    case = {
        key: early[key]
        for key in (
            "layout",
            "ranks",
            "decomposition",
            "global_grid",
            "local_grid",
            "timesteps",
            "fd_order",
            "L",
            "m8b_fd8_l1_fast_path",
        )
    }
    case["measurement_label"] = "1x1_combined_early_and_late"
    case["directories"] = [early["directory"], late["directory"]]
    return _summarize_case(
        case,
        warmup_external_seconds=(
            early["warmup_external_seconds"] + late["warmup_external_seconds"]
        ),
        individual_runs=early["individual_runs"] + late["individual_runs"],
        normal_hashes=early["normal_trace_sha256"],
        instrumented_hashes=late["instrumented_trace_sha256"],
    )


def _orientation_assessment(layouts: dict[str, dict[str, object]]) -> dict[str, object]:
    first = layouts["2x1"]
    second = layouts["1x2"]
    first_median = first["rank_consistent_critical_median_seconds"]["timestep_loop"]
    second_median = second["rank_consistent_critical_median_seconds"]["timestep_loop"]
    relative_difference = abs(first_median - second_median) / statistics.mean(
        (first_median, second_median)
    )
    variability = max(
        first["relative_timestep_loop_range"], second["relative_timestep_loop_range"]
    )
    meaningful = relative_difference > variability
    return {
        "median_relative_difference": relative_difference,
        "2x1_relative_range": first["relative_timestep_loop_range"],
        "1x2_relative_range": second["relative_timestep_loop_range"],
        "criterion": (
            "Meaningful only when the relative median difference exceeds the larger "
            "full run-to-run relative range."
        ),
        "conclusion": (
            ("2x1 faster" if first_median < second_median else "1x2 faster")
            if meaningful
            else "INCONCLUSIVE"
        ),
    }


def _environment(binary: Path, mpiexec: str, mpiexec_flags: list[str]) -> dict[str, object]:
    makefile = (REPOSITORY_ROOT / "src" / "Makefile").read_text(encoding="utf-8")
    active_flags = {}
    for line in makefile.splitlines():
        if line.startswith(("CC=", "CFLAGS=", "LFLAGS=", "SFLAGS=", "IFLAGS=")):
            key, value = line.split("=", 1)
            active_flags[key] = value
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "os_release": _capture(["sh", "-c", "cat /etc/os-release"]),
        "cpu": _capture(["lscpu"]),
        "compiler": _capture([active_flags.get("CC", "mpicc"), "--version"]),
        "mpi": _capture([mpiexec, "--version"]),
        "make_variables": active_flags,
        "binary": str(binary),
        "mpiexec": mpiexec,
        "mpiexec_flags": mpiexec_flags,
        "process_binding": " ".join(mpiexec_flags) if mpiexec_flags else "launcher default",
        "omp_num_threads": 1,
    }


def _provenance(expected_base_sha: str | None) -> dict[str, object]:
    head = _git("rev-parse", "HEAD")
    if expected_base_sha is not None and head != expected_base_sha:
        raise RuntimeError(f"HEAD {head} does not match requested base {expected_base_sha}")
    status = _git("status", "--short").splitlines()
    return {
        "base_sha": head,
        "branch": _git("branch", "--show-current"),
        "candidate_dirty": bool(status),
        "candidate_status_short": status,
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "revision_context": "Uncommitted benchmark candidate based on base_sha; no candidate commit exists.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--mpiexec-flags", default="")
    parser.add_argument("--layouts", nargs="+", choices=tuple(LAYOUTS), default=list(LAYOUTS))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--fd-order", type=int, default=4)
    parser.add_argument("--time-seconds", type=float, default=0.55)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--base-sha")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.warmups < 1:
        parser.error("at least one warmup is required")
    if args.repetitions < 5:
        parser.error("at least five measured runs are required")
    if "1x1" not in args.layouts:
        parser.error("1x1 is required to compute speedup")

    binary = args.binary.resolve(strict=True)
    mpiexec_flags = shlex.split(args.mpiexec_flags)
    config = replace(
        HomogeneousPSVConfig(),
        fd_order=args.fd_order,
        time_s=args.time_seconds,
    )
    args.root.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(args.base_sha)
    workloads = {}
    for workload in ("elastic_psv", "visco_psv_l1"):
        measured_by_label = {}
        actual_order = []
        for measurement_label in INTERLEAVED_ORDER:
            layout = "1x1" if measurement_label.startswith("1x1_") else measurement_label
            if layout not in args.layouts:
                continue
            case = _make_case(
                args.root,
                workload,
                measurement_label,
                layout,
                config=config,
            )
            measured_by_label[measurement_label] = _measure_case(
                binary,
                args.mpiexec,
                mpiexec_flags,
                case,
                warmups=args.warmups,
                repetitions=args.repetitions,
                timeout_seconds=args.timeout_seconds,
            )
            actual_order.append(measurement_label)

        early = measured_by_label["1x1_early"]
        late = measured_by_label["1x1_late"]
        early_time = early["rank_consistent_critical_median_seconds"]["timestep_loop"]
        late_time = late["rank_consistent_critical_median_seconds"]["timestep_loop"]
        drift = abs(early_time - late_time) / statistics.mean((early_time, late_time))
        combined_reference = _combine_references(early, late)
        layouts = {"1x1": combined_reference}
        for layout in ("2x1", "1x2", "2x2"):
            if layout in measured_by_label:
                layouts[layout] = measured_by_label[layout]
        reference = combined_reference["rank_consistent_critical_median_seconds"][
            "timestep_loop"
        ]
        reference_hashes = combined_reference["normal_trace_sha256"]
        for result in layouts.values():
            total = result["rank_consistent_critical_median_seconds"]["timestep_loop"]
            result["speedup"] = reference / total
            result["parallel_efficiency"] = result["speedup"] / result["ranks"]
            nprocx, nprocy = result["decomposition"]
            result["interior_cell_fraction"] = _interior_fraction(
                config.nx,
                config.ny,
                nprocx,
                nprocy,
                config.fd_order // 2,
            )
            if result["normal_trace_sha256"] != reference_hashes:
                raise RuntimeError(f"{workload} traces changed with MPI decomposition")
            result["decomposition_trace_bit_identical"] = True
        workloads[workload] = {
            "run_order": actual_order,
            "reference_1x1": {
                "early": early,
                "late": late,
                "relative_drift": drift,
                "interpretation_threshold": 0.05,
                "material_drift": drift > 0.05,
                "speedup_reference": (
                    "median of all early and late measured 1x1 timestep-loop values"
                ),
            },
            "orientation_assessment": (
                _orientation_assessment(layouts)
                if "2x1" in layouts and "1x2" in layouts
                else {"conclusion": "NOT MEASURED"}
            ),
            "layouts": layouts,
        }

    l1_2x2 = workloads["visco_psv_l1"]["layouts"]["2x2"]
    l1_components = l1_2x2[
        "critical_communication_rank_component_median_seconds"
    ]
    stress_dominates = l1_components["stress_exchange"] > l1_components["velocity_exchange"]
    report = {
        "schema": "denise.psv_mpi_baseline.v2",
        "label": args.label,
        "base_sha": provenance["base_sha"],
        "branch": provenance["branch"],
        "candidate_dirty": provenance["candidate_dirty"],
        "provenance": provenance,
        "scope": (
            "Single-node WSL decomposition scaling; these measurements do not establish "
            "distributed-cluster scaling."
        ),
        "performance_interpretation": (
            "The fixed global workload may exhibit superlinear single-node speedup, plausibly "
            "due to cache/working-set effects; hardware-counter confirmation was not performed."
        ),
        "run_protocol": {
            "order_per_workload": list(INTERLEAVED_ORDER),
            "order_policy": "deterministic interleave with early and late 1x1 references",
            "warmups_per_measurement": args.warmups,
            "measured_runs_per_measurement": args.repetitions,
            "primary_statistic": "median",
            "speedup_reference": "combined median of early and late 1x1 measured runs",
        },
        "metric_definitions": {
            "component_wise_rank_maxima": (
                "For each phase independently, max_r(phase_r); sums can combine different ranks."
            ),
            "critical_total": "max_r(timestep_loop_r) for each run",
            "critical_communication": (
                "max_r(velocity_exchange_r + stress_exchange_r) for each run"
            ),
            "critical_compute": "max_r(velocity_update_r + stress_update_r) for each run",
            "critical_rank_communication_fraction": (
                "median(critical_communication) / median(critical_total); authoritative M8d-2A metric"
            ),
            "sum_of_component_maxima_fraction": (
                "(median(max_r velocity_exchange_r) + median(max_r stress_exchange_r)) / "
                "median(critical_total); legacy diagnostic only"
            ),
            "speedup": "combined early/late 1x1 critical-total median / layout critical-total median",
            "parallel_efficiency": "speedup / ranks",
        },
        "timer": (
            "MPI_Wtime around four coarse phases; disabled unless "
            "DENISE_PSV_MPI_TIMING_FILE is set"
        ),
        "timer_overhead": (
            "eight MPI_Wtime calls per timestep, one pre-loop barrier, no barriers inside the "
            "timestep loop; one small rank-record gather and JSON output occur after the loop"
        ),
        "environment": _environment(binary, args.mpiexec, mpiexec_flags),
        "benchmark_problem": {
            "global_grid": [config.nx, config.ny],
            "fd_order": config.fd_order,
            "stencil_radius": config.fd_order // 2,
            "timesteps": config.samples_per_trace,
            "workloads": {"elastic_psv": {"L": 0}, "visco_psv_l1": {"L": 1}},
        },
        "m8d2b_assessment": {
            "metric_basis": "rank-consistent critical communication path",
            "overlap_opportunity_supported": True,
            "target_velocity_exchange": True,
            "target_stress_exchange": True,
            "dominant_l1_2x2_exchange_on_critical_communication_rank": (
                "stress_exchange" if stress_dominates else "velocity_exchange"
            ),
            "packed_start_finish_infrastructure_recommended": True,
            "recommendation": (
                "Prioritize stress exchange while implementing symmetric start/finish "
                "infrastructure for both exchanges."
                if stress_dominates
                else "Implement symmetric start/finish infrastructure for both exchanges."
            ),
        },
        "workloads": workloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
