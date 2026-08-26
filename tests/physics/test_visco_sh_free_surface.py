from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.cases.visco_sh_free_surface import (
    PHYSICAL_L4_FREQUENCIES_HZ,
    generate_case,
    runtime_scenario,
)
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import read_ascii_seismograms
from tests.utilities.sh_free_surface import finite_nonzero
from tests.utilities.visco_sh_free_surface_runtime import (
    FINITE_Q_SENSITIVITY_RELATIVE_L2_MIN,
    FREE_SURFACE_ZERO_CONTROL_L2_RATIO_MAX,
    FROZEN_BOUNDARY_LIMITS,
    FROZEN_WAVEFORM_ACCEPTANCE,
    HARD_BOUNDARY_KEYS,
    DIAGNOSTIC_BOUNDARY_KEYS,
    HIGH_Q_ENDPOINT_CORRELATION_MIN,
    HIGH_Q_ENDPOINT_RELATIVE_L2_MAX,
    MPI_CORRELATION_MIN,
    MPI_RELATIVE_L2_MAX,
    REFERENCE_TRANSLATION_CORRELATION_MIN,
    REFERENCE_TRANSLATION_LAG_MAX_S,
    REFERENCE_TRANSLATION_RELATIVE_L2_MAX,
    SUPERPOSITION_RELATIVE_L2_MAX,
    acceptance_metadata,
    accepted,
    normalized_correlation,
    relative_l2,
    waveform_metrics,
    window,
)


pytestmark = [pytest.mark.integration, pytest.mark.extended]


@contextmanager
def _diagnostic_environment(path: Path | None):
    old = os.environ.get("M62B_SH_DIAGNOSTICS")
    if path is None:
        os.environ.pop("M62B_SH_DIAGNOSTICS", None)
    else:
        os.environ["M62B_SH_DIAGNOSTICS"] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("M62B_SH_DIAGNOSTICS", None)
        else:
            os.environ["M62B_SH_DIAGNOSTICS"] = old


def _run(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    scenario,
    role: str,
    nprocx: int = 1,
    nprocy: int = 1,
    elastic: bool = False,
    diagnostic: bool = False,
    retain_diagnostic_series: bool = False,
) -> dict[str, object]:
    config = generate_case(
        directory,
        scenario=scenario,
        role=role,
        nprocx=nprocx,
        nprocy=nprocy,
        elastic=elastic,
    )
    case_metadata = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    diagnostic_path = directory / "m62b_diagnostics.csv" if diagnostic else None
    with _diagnostic_environment(diagnostic_path):
        result = run_denise(
            repository_root=repository_root,
            case_directory=directory,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            ranks=nprocx * nprocy,
            configuration=case_metadata,
            timeout_seconds=240.0,
        )
    assert result.returncode == 0, result_summary(result)
    output = directory / "su" / "homogeneous_vz.asc.shot1"
    assert output.is_file() and output.stat().st_size > 0
    trace = read_ascii_seismograms(
        output, config.receiver_count, config.samples_per_trace
    )[0]
    assert finite_nonzero(trace)
    parsed_diagnostic = None
    if diagnostic:
        assert diagnostic_path is not None and diagnostic_path.is_file()
        with diagnostic_path.open(newline="", encoding="utf-8") as handle:
            rows = [
                {key: (int(value) if key == "timestep" else float(value)) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        assert len(rows) == config.samples_per_trace
        assert all(
            math.isfinite(value)
            for row in rows
            for key, value in row.items()
            if key != "timestep"
        )
        parsed_diagnostic = {
            "row_count": len(rows),
            "maxima": {
                key: max(row[key] for row in rows)
                for key in rows[0]
                if key != "timestep"
            },
            "tail_max_abs_vz": max(
                row["max_abs_vz"] for row in rows[int(0.75 * len(rows)) :]
            ),
            "global_max_abs_vz": max(row["max_abs_vz"] for row in rows),
        }
        if retain_diagnostic_series:
            parsed_diagnostic["max_abs_vz_series"] = [
                row["max_abs_vz"] for row in rows
            ]
    run_metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    return {
        "trace": trace,
        "case": case_metadata,
        "run": run_metadata,
        "seismogram_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "diagnostic": parsed_diagnostic,
    }


def _interval_trace(run: dict[str, object]) -> list[float]:
    case = run["case"]
    dt_s = case["numerics"]["dt_s"]
    return window(run["trace"], (dt_s, case["comparison_stop_s"]), dt_s)


def _window_trace(run: dict[str, object], key: str) -> list[float]:
    case = run["case"]
    return window(run["trace"], case[key], case["numerics"]["dt_s"])


def _metrics(reference: dict[str, object], candidate: dict[str, object], *, reflected=True):
    key = "expected_reflected_image_window_s" if reflected else "expected_direct_window_s"
    dt_s = reference["case"]["numerics"]["dt_s"]
    return waveform_metrics(
        _window_trace(reference, key), _window_trace(candidate, key), dt_s=dt_s
    )


def _compact_run(run: dict[str, object]) -> dict[str, object]:
    return {
        "case": run["case"],
        "run": run["run"],
        "seismogram_sha256": run["seismogram_sha256"],
        "diagnostic": run["diagnostic"],
    }


def test_m62b_prefixed_runtime_oracle_and_red_baseline(
    tmp_path, repository_root, denise_binary, mpiexec
):
    """Lock reference/calibration first, then characterize unchanged FREE_SURF=1."""
    instrumented_text = os.environ.get("M62B_INSTRUMENTED_DENISE")
    assert instrumented_text, "M62B_INSTRUMENTED_DENISE is required for the locked audit"
    instrumented_binary = Path(instrumented_text).resolve()
    assert instrumented_binary.is_file()
    patch_path = repository_root / "tests" / "m6.2b_visco_sh_free_surface_instrumentation.patch"
    patch_sha = hashlib.sha256(patch_path.read_bytes()).hexdigest()

    report: dict[str, object] = {
        "base_git_sha": "f0fad66c2521951d26ca40acf0460fed86e43eca",
        "scope": "test/oracle/instrumentation only; unchanged pre-fix solver",
        "acceptance": acceptance_metadata(),
        "instrumentation_patch_sha256": patch_sha,
        "matrix": {
            "mandatory_fdorders": [2, 4, 12],
            "extended_fdorders": [2, 4, 6, 8, 10, 12],
            "mandatory_geometries": ["normal", "oblique"],
            "mandatory_mechanisms": {"l": 1, "fl_hz": [10.0]},
            "extended_mechanisms": {"l": 4, "fl_hz": list(PHYSICAL_L4_FREQUENCIES_HZ)},
            "mpi": [[1, 1], [2, 1], [1, 2]],
            "fwi_adjoint_taylor": "excluded",
        },
        "reference_runs": {},
        "candidate_runs": {},
    }

    # Phase 1: full-space references only. No FREE_SURF=1 run occurs above this line.
    references = {}
    for plane in (1200.0, 1600.0):
        scenario = runtime_scenario(reference_plane_y_m=plane)
        for role in (("reference_combined", "reference_real", "reference_image") if plane == 1200.0 else ("reference_combined",)):
            key = f"plane_{int(plane)}_{role}"
            references[key] = _run(
                tmp_path / key,
                repository_root=repository_root,
                denise_binary=denise_binary,
                mpiexec=mpiexec,
                scenario=scenario,
                role=role,
            )
            assert references[key]["case"]["external_return_outside_comparison"] is True

    combined = references["plane_1200_reference_combined"]
    real = references["plane_1200_reference_real"]
    image = references["plane_1200_reference_image"]
    direct_real = _window_trace(real, "expected_direct_window_s")
    image_only = _window_trace(image, "expected_reflected_image_window_s")
    assert math.sqrt(sum(value * value for value in direct_real)) > 0.0
    assert math.sqrt(sum(value * value for value in image_only)) > 0.0
    assert combined["case"]["expected_direct_window_s"][1] < combined["case"]["expected_reflected_image_window_s"][0]
    summed = [left + right for left, right in zip(real["trace"], image["trace"])]
    superposition_l2 = relative_l2(combined["trace"], summed)
    assert superposition_l2 <= SUPERPOSITION_RELATIVE_L2_MAX

    translated = references["plane_1600_reference_combined"]
    translation_metrics = waveform_metrics(
        _interval_trace(combined),
        _interval_trace(translated),
        dt_s=combined["case"]["numerics"]["dt_s"],
    )
    assert translation_metrics.relative_l2 <= REFERENCE_TRANSLATION_RELATIVE_L2_MAX
    assert translation_metrics.normalized_correlation >= REFERENCE_TRANSLATION_CORRELATION_MIN
    assert abs(translation_metrics.arrival_lag_s) <= REFERENCE_TRANSLATION_LAG_MAX_S
    report["reference_health"] = {
        "separated_direct_and_image_windows": True,
        "both_source_contributions_nonzero": True,
        "outer_returns_outside_comparison": True,
        "linear_superposition_relative_l2": superposition_l2,
        "translation_self_consistency": asdict(translation_metrics),
    }

    # Phase 2: rheology calibration in FREE_SURF=0 full space.
    calibration = {50.0: combined}
    for qs in (200.0, 1000.0):
        scenario = runtime_scenario(qs=qs)
        calibration[qs] = _run(
            tmp_path / f"calibration_q{int(qs)}",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            scenario=scenario,
            role="reference_combined",
        )
    elastic_scenario = runtime_scenario(qs=1000.0)
    elastic = _run(
        tmp_path / "calibration_elastic",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        scenario=elastic_scenario,
        role="reference_combined",
        elastic=True,
    )
    high_q_rows = []
    for qs in (50.0, 200.0, 1000.0):
        metrics = waveform_metrics(
            _interval_trace(elastic),
            _interval_trace(calibration[qs]),
            dt_s=elastic["case"]["numerics"]["dt_s"],
        )
        high_q_rows.append({"qs": qs, **asdict(metrics)})
    errors = [row["relative_l2"] for row in high_q_rows]
    assert all(left > right for left, right in zip(errors, errors[1:]))
    assert high_q_rows[-1]["relative_l2"] <= HIGH_Q_ENDPOINT_RELATIVE_L2_MAX
    assert high_q_rows[-1]["normalized_correlation"] >= HIGH_Q_ENDPOINT_CORRELATION_MIN
    assert relative_l2(calibration[50.0]["trace"], elastic["trace"]) >= FINITE_Q_SENSITIVITY_RELATIVE_L2_MIN

    l4_scenario = runtime_scenario(
        qs=30.0, frequencies_hz=PHYSICAL_L4_FREQUENCIES_HZ
    )
    l4_reference = _run(
        tmp_path / "calibration_l4_q30",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        scenario=l4_scenario,
        role="reference_combined",
    )
    l4_sensitivity = relative_l2(l4_reference["trace"], elastic["trace"])
    assert l4_sensitivity >= FINITE_Q_SENSITIVITY_RELATIVE_L2_MIN
    report["rheology_calibration"] = {
        "high_q_rows": high_q_rows,
        "finite_q_l1_vs_elastic_relative_l2": relative_l2(calibration[50.0]["trace"], elastic["trace"]),
        "finite_q_l4_vs_elastic_relative_l2": l4_sensitivity,
        "l1": combined["case"]["rheology"],
        "l4": l4_reference["case"]["rheology"],
        "model_hashes": {
            "l1": combined["case"]["model_sha256"],
            "l4": l4_reference["case"]["model_sha256"],
            "elastic": elastic["case"]["model_sha256"],
        },
    }
    report["reference_runs"] = {key: _compact_run(value) for key, value in references.items()}
    report["reference_runs"].update({
        f"high_q_{int(qs)}": _compact_run(run) for qs, run in calibration.items()
    })
    report["reference_runs"]["elastic"] = _compact_run(elastic)
    report["reference_runs"]["l4_q30"] = _compact_run(l4_reference)

    # Freeze matching-FD references and the absorbing-top control before candidates.
    reference_by_fd = {12: combined}
    for fd_order in (2, 4, 6, 8, 10):
        reference_by_fd[fd_order] = _run(
            tmp_path / f"reference_normal_fd{fd_order}",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            scenario=runtime_scenario(fd_order=fd_order),
            role="reference_combined",
        )
    oblique_reference = _run(
        tmp_path / "reference_oblique_fd12",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        scenario=runtime_scenario(geometry="oblique"),
        role="reference_combined",
    )
    absorbing_control = _run(
        tmp_path / "control_absorbing_fd12",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        scenario=runtime_scenario(),
        role="absorbing",
    )
    absorbing_image = _window_trace(
        absorbing_control, "expected_reflected_image_window_s"
    )
    reference_image = _window_trace(combined, "expected_reflected_image_window_s")
    absorbing_ratio = math.sqrt(sum(value * value for value in absorbing_image)) / math.sqrt(
        sum(value * value for value in reference_image)
    )
    assert absorbing_ratio <= FREE_SURFACE_ZERO_CONTROL_L2_RATIO_MAX
    report["reference_runs"].update(
        {f"normal_fd{fd_order}": _compact_run(run) for fd_order, run in reference_by_fd.items()}
    )
    report["reference_runs"]["oblique_fd12"] = _compact_run(oblique_reference)
    report["reference_runs"]["absorbing_control_fd12"] = _compact_run(absorbing_control)

    # Phase 3: unchanged FREE_SURF=1 candidate, only after every lock above.
    candidate_results = {}
    for fd_order in (2, 4, 6, 8, 10, 12):
        scenario = runtime_scenario(fd_order=fd_order)
        candidate_results[f"normal_fd{fd_order}_1x1"] = _run(
            tmp_path / f"candidate_normal_fd{fd_order}",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            scenario=scenario,
            role="candidate",
        )
    candidate_results["oblique_fd12_1x1"] = _run(
        tmp_path / "candidate_oblique_fd12",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        scenario=runtime_scenario(geometry="oblique"),
        role="candidate",
    )
    candidate_results["absorbing_fd12_1x1"] = absorbing_control
    for nprocx, nprocy in ((2, 1), (1, 2)):
        candidate_results[f"normal_fd12_{nprocx}x{nprocy}"] = _run(
            tmp_path / f"candidate_normal_fd12_{nprocx}x{nprocy}",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            scenario=runtime_scenario(),
            role="candidate",
            nprocx=nprocx,
            nprocy=nprocy,
        )
    candidate_results["normal_fd12_l4"] = _run(
        tmp_path / "candidate_normal_fd12_l4",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        scenario=l4_scenario,
        role="candidate",
    )

    # High-Q FREE_SURF=1 convergence against the elastic M6.1 endpoint.
    candidate_high_q = {50.0: candidate_results["normal_fd12_1x1"]}
    for qs in (200.0, 1000.0):
        candidate_high_q[qs] = _run(
            tmp_path / f"candidate_high_q_{int(qs)}",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            scenario=runtime_scenario(qs=qs),
            role="candidate",
        )
    elastic_candidate = _run(
        tmp_path / "candidate_elastic_endpoint",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        scenario=runtime_scenario(qs=1000.0),
        role="candidate",
        elastic=True,
    )
    high_q_surface_rows = []
    for qs in (50.0, 200.0, 1000.0):
        row = waveform_metrics(
            _interval_trace(elastic_candidate),
            _interval_trace(candidate_high_q[qs]),
            dt_s=elastic_candidate["case"]["numerics"]["dt_s"],
        )
        high_q_surface_rows.append({"qs": qs, **asdict(row)})
    surface_errors = [row["relative_l2"] for row in high_q_surface_rows]
    assert all(left > right for left, right in zip(surface_errors, surface_errors[1:]))
    assert high_q_surface_rows[-1]["relative_l2"] <= HIGH_Q_ENDPOINT_RELATIVE_L2_MAX
    assert high_q_surface_rows[-1]["normalized_correlation"] >= HIGH_Q_ENDPOINT_CORRELATION_MIN
    assert abs(high_q_surface_rows[-1]["signed_amplitude_ratio"] - 1.0) <= 0.03
    assert abs(high_q_surface_rows[-1]["arrival_lag_s"]) <= 0.001

    # Frozen test-only diagnostic build: it must not perturb the waveform.
    instrumented = _run(
        tmp_path / "candidate_instrumented_fd12",
        repository_root=repository_root,
        denise_binary=instrumented_binary,
        mpiexec=mpiexec,
        scenario=runtime_scenario(),
        role="candidate",
        diagnostic=True,
    )
    equivalence = waveform_metrics(
        candidate_results["normal_fd12_1x1"]["trace"],
        instrumented["trace"],
        dt_s=instrumented["case"]["numerics"]["dt_s"],
    )
    assert equivalence.relative_l2 <= 1.0e-12
    assert equivalence.normalized_correlation >= 1.0 - 1.0e-12

    waveform_rows = {}
    for key, run in candidate_results.items():
        if key == "absorbing_fd12_1x1":
            continue
        if key == "normal_fd12_l4":
            reference = l4_reference
        elif "oblique" in key:
            reference = oblique_reference
        else:
            fd_order = int(key.split("_fd", 1)[1].split("_", 1)[0])
            reference = reference_by_fd[fd_order]
        metrics = _metrics(reference, run)
        waveform_rows[key] = {**asdict(metrics), "accepted": accepted(metrics)}

    mpi_rows = {}
    mpi_reference = candidate_results["normal_fd12_1x1"]
    for decomposition in ("2x1", "1x2"):
        run = candidate_results[f"normal_fd12_{decomposition}"]
        metrics = waveform_metrics(
            mpi_reference["trace"], run["trace"], dt_s=run["case"]["numerics"]["dt_s"]
        )
        mpi_rows[decomposition] = asdict(metrics)
        assert metrics.relative_l2 <= MPI_RELATIVE_L2_MAX
        assert metrics.normalized_correlation >= MPI_CORRELATION_MIN

    diagnostic = instrumented["diagnostic"]
    maxima = diagnostic["maxima"]
    traction_scale = max(maxima["max_abs_interior_stress"], 1.0e-30)
    velocity_scale = max(maxima["max_abs_vz"], 1.0e-30)
    q_scale = max(maxima["max_abs_active_q"], 1.0e-30)
    boundary = {
        "traction_residual": maxima["max_abs_syz0"] / traction_scale,
        "dplus_vz_residual": maxima["max_abs_dplus_vz0"] / (velocity_scale / 10.0),
        "vz_parity_residual": maxima["max_vz_parity_residual"] / velocity_scale,
        "total_syz_parity_residual": maxima["max_syz_parity_residual"] / traction_scale,
        "q_surface_residual": maxima["max_abs_q0"] / q_scale,
        "q_parity_residual": maxima["max_q_parity_residual"] / q_scale,
    }
    boundary_limits = FROZEN_BOUNDARY_LIMITS
    boundary_pass = {
        name: boundary[name] <= boundary_limits[f"{name}_max"]
        for name in HARD_BOUNDARY_KEYS
    }
    expected_surface_failures = [
        name for name, passed in boundary_pass.items() if not passed
    ]
    high_order_waveform_failures = [
        key for key in ("normal_fd4_1x1", "normal_fd12_1x1", "oblique_fd12_1x1")
        if not waveform_rows[key]["accepted"]
    ]
    assert expected_surface_failures or high_order_waveform_failures

    report.update(
        {
            "candidate_runs": {key: _compact_run(value) for key, value in candidate_results.items()},
            "candidate_waveform_results": waveform_rows,
            "free_surface_zero_control_image_window_l2_ratio": absorbing_ratio,
            "mpi_results": mpi_rows,
            "high_q_free_surface_results": high_q_surface_rows,
            "instrumented_uninstrumented_equivalence": asdict(equivalence),
            "boundary_residuals": boundary,
            "boundary_limits": boundary_limits,
            "boundary_contract_pass": boundary_pass,
            "boundary_contract": {
                "hard": {
                    name: {
                        "value": boundary[name],
                        "limit": boundary_limits[f"{name}_max"],
                        "pass": boundary_pass[name],
                    }
                    for name in HARD_BOUNDARY_KEYS
                },
                "diagnostic_only": {
                    name: {
                        "value": boundary[name],
                        "acceptance_effect": "none",
                    }
                    for name in DIAGNOSTIC_BOUNDARY_KEYS
                },
            },
            "stability": {
                "acceptance_effect": "none",
                "classification": "legacy tail/global diagnostic is not a hard gate",
                "tail_max_abs_vz": diagnostic["tail_max_abs_vz"],
                "global_max_abs_vz": diagnostic["global_max_abs_vz"],
            },
            "red_classification": {
                "expected_missing_surface_failures": expected_surface_failures,
                "high_order_waveform_failures": high_order_waveform_failures,
                "unexpected_reference_or_calibration_failure": False,
            },
            "final_verdict": "M6.2b VISCOELASTIC SH FREE-SURFACE ORACLE LOCKED — PRE-FIX RED BASELINE ESTABLISHED",
        }
    )
    report["candidate_runs"].update(
        {
            "high_q_200": _compact_run(candidate_high_q[200.0]),
            "high_q_1000": _compact_run(candidate_high_q[1000.0]),
            "elastic_m61_endpoint": _compact_run(elastic_candidate),
            "instrumented_fd12": _compact_run(instrumented),
        }
    )
    report["executed_run_count"] = 29
    assert all(
        run["run"]["returncode"] == 0
        for collection in (report["reference_runs"], report["candidate_runs"])
        for run in collection.values()
    )
    live_path = tmp_path / "m6.2b_visco_sh_free_surface_live_report.json"
    live_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if os.environ.get("M62B_REGENERATE_LOCKED_VALIDATION") == "1":
        validation_path = (
            repository_root / "tests" / "m6.2b_visco_sh_free_surface_validation.json"
        )
        validation_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
