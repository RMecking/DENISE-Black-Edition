from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.cases.sh_free_surface import generate_case, normal_scenario
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import read_ascii_seismograms
from tests.utilities.sh_free_surface import (
    SurfaceTimes,
    centered_window,
    evaluate_production_acceptance,
    evaluate_reflection,
    evaluate_surface_boundary,
    finite_nonzero,
    holberg_coefficients,
    normalized_correlation,
    normalized_surface_residuals,
    peak_time,
    signed_amplitude_alpha,
    surface_roundoff_limits,
)


pytestmark = [pytest.mark.integration, pytest.mark.extended]


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
) -> tuple[list[float], dict[str, object], dict[str, object] | None]:
    config = generate_case(
        directory,
        scenario=scenario,
        role=role,
        nprocx=nprocx,
        nprocy=nprocy,
    )
    case_metadata = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration=case_metadata,
        timeout_seconds=180.0,
    )
    assert result.returncode == 0, result_summary(result)
    output = directory / "su" / "homogeneous_vz.asc.shot1"
    assert output.is_file() and output.stat().st_size > 0
    traces = read_ascii_seismograms(output, config.receiver_count, config.samples_per_trace)
    assert len(traces) == 1
    assert finite_nonzero(traces[0])
    run_metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    run_metadata["seismogram_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    diagnostic_path = directory / "m61b_diagnostics.csv"
    diagnostic = None
    if diagnostic_path.is_file():
        with diagnostic_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        parsed = [
            {
                "timestep": int(row["timestep"]),
                "max_abs_syz0": float(row["max_abs_syz0"]),
                "max_abs_dplus_vz0": float(row["max_abs_dplus_vz0"]),
                "max_vz_parity_residual": float(
                    row["max_vz_parity_residual"]
                ),
                "max_syz_parity_residual": float(
                    row["max_syz_parity_residual"]
                ),
                "max_abs_interior_stress": float(
                    row["max_abs_interior_stress"]
                ),
                "max_impedance_vz": float(row["max_impedance_vz"]),
                "max_abs_dx_vz": float(row["max_abs_dx_vz"]),
                "max_abs_vz": float(row["max_abs_vz"]),
                "centered_energy": float(row["centered_energy"]),
            }
            for row in rows
        ]
        assert all(
            math.isfinite(value)
            for row in parsed
            for key, value in row.items()
            if key != "timestep"
        )
        diagnostic = {
            "path": str(diagnostic_path.resolve()),
            "row_count": len(parsed),
            "maximum_abs_syz0": max(row["max_abs_syz0"] for row in parsed),
            "maximum_abs_dplus_vz0": max(
                row["max_abs_dplus_vz0"] for row in parsed
            ),
            "maximum_vz_parity_residual": max(
                row["max_vz_parity_residual"] for row in parsed
            ),
            "maximum_syz_parity_residual": max(
                row["max_syz_parity_residual"] for row in parsed
            ),
            "maximum_abs_interior_stress": max(
                row["max_abs_interior_stress"] for row in parsed
            ),
            "maximum_impedance_vz": max(
                row["max_impedance_vz"] for row in parsed
            ),
            "maximum_abs_dx_vz": max(
                row["max_abs_dx_vz"] for row in parsed
            ),
            "maximum_abs_vz": max(row["max_abs_vz"] for row in parsed),
            "minimum_centered_energy": min(
                row["centered_energy"] for row in parsed
            ),
            "maximum_centered_energy": max(
                row["centered_energy"] for row in parsed
            ),
        }
        coefficients = holberg_coefficients(config.fd_order)
        normalized_syz, normalized_dplus = normalized_surface_residuals(
            max_abs_syz0=diagnostic["maximum_abs_syz0"],
            max_abs_dplus_vz0=diagnostic["maximum_abs_dplus_vz0"],
            max_abs_interior_stress=diagnostic["maximum_abs_interior_stress"],
            max_impedance_vz=diagnostic["maximum_impedance_vz"],
            max_abs_dx_vz=diagnostic["maximum_abs_dx_vz"],
            max_abs_vz=diagnostic["maximum_abs_vz"],
            f95_hz=case_metadata["source_spectrum"]["f95_hz"],
            vs_m_s=config.vs_m_s,
        )
        syz_limit, dplus_limit = surface_roundoff_limits(coefficients)
        diagnostic["normalized_physical_traction_residual"] = normalized_syz
        diagnostic["normalized_image_closure_residual"] = normalized_dplus
        diagnostic["physical_traction_limit"] = syz_limit
        diagnostic["image_closure_limit"] = dplus_limit
    return traces[0], run_metadata, diagnostic


def _window_at_pick(
    trace: list[float], *, expected_s: float, half_width_s: float, dt_s: float
) -> tuple[float, list[float]]:
    picked = peak_time(
        trace, expected_s=expected_s, half_width_s=half_width_s, dt_s=dt_s
    )
    analysis_half_width = 0.5 * half_width_s
    return picked, centered_window(
        trace, center_s=picked, half_width_s=analysis_half_width, dt_s=dt_s
    )


@pytest.mark.parametrize("fd_order", (2, 4, 12))
def test_prefixed_normal_sh_free_surface_behavior(
    tmp_path, repository_root, denise_binary, mpiexec, fd_order
):
    """Record unmodified-solver behavior without treating it as M6.1c acceptance."""
    scenario = normal_scenario(fd_order)
    metadata = scenario.metadata()
    assert metadata["surface_location_resolved"] is True
    assert (
        metadata["expected_reflection_peak_s"] - metadata["expected_direct_peak_s"]
        > 2.0 * scenario.reflection_window_half_width_s
    )

    traces: dict[str, list[float]] = {}
    runs: dict[str, object] = {}
    diagnostics: dict[str, object] = {}
    for role in ("free_surface", "absorbing", "calibration"):
        traces[role], runs[role], diagnostic = _run(
            tmp_path / role,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            scenario=scenario,
            role=role,
        )
        if diagnostic is not None:
            diagnostics[role] = diagnostic

    free_pick, free_window = _window_at_pick(
        traces["free_surface"],
        expected_s=metadata["expected_reflection_peak_s"],
        half_width_s=scenario.reflection_window_half_width_s,
        dt_s=scenario.free_surface.dt_s,
    )
    calibration_pick, calibration_window = _window_at_pick(
        traces["calibration"],
        expected_s=metadata["expected_calibration_peak_s"],
        half_width_s=scenario.reflection_window_half_width_s,
        dt_s=scenario.free_surface.dt_s,
    )
    _, absorbing_window = _window_at_pick(
        traces["absorbing"],
        expected_s=metadata["expected_reflection_peak_s"],
        half_width_s=scenario.reflection_window_half_width_s,
        dt_s=scenario.free_surface.dt_s,
    )
    assert len(free_window) == len(calibration_window) == len(absorbing_window)

    alpha = signed_amplitude_alpha(free_window, calibration_window)
    correlation = normalized_correlation(free_window, calibration_window)
    free_norm = math.sqrt(sum(value * value for value in free_window))
    absorbing_ratio = math.sqrt(sum(value * value for value in absorbing_window)) / free_norm
    timing_error = abs(free_pick - calibration_pick)
    candidate_times = metadata["surface_candidate_propagation_times_s"]
    # The equal-vector calibration removes the common source/picker delay and
    # numerical group delay.  Its y=0 travel time anchors the native-grid
    # surface candidates without trusting the analytic Ricker peak time.
    observed_propagation = (
        candidate_times["y0_s"] + free_pick - calibration_pick
    )
    shifted_residuals = {
        "y_half_h": abs(observed_propagation - candidate_times["y_half_h_s"]),
        "y_h": abs(observed_propagation - candidate_times["y_h_s"]),
    }
    timing_tolerance = metadata["timing_tolerance_s"]
    reflection_acceptance = evaluate_reflection(
        timing_error_s=timing_error,
        observed_propagation_s=observed_propagation,
        surface_times=SurfaceTimes(**candidate_times),
        timing_tolerance_s=timing_tolerance,
        signed_amplitude_alpha_value=alpha,
        normalized_correlation_value=correlation,
        absorbing_l2_ratio=absorbing_ratio,
        signed_amplitude_error_max=metadata["acceptance"][
            "signed_amplitude_error_max"
        ],
        normalized_correlation_min=metadata["acceptance"][
            "normalized_correlation_min"
        ],
        absorbing_l2_ratio_max=metadata["acceptance"][
            "absorbing_l2_ratio_max"
        ],
    )
    reflection_report = asdict(reflection_acceptance) | {
        "all_pass": reflection_acceptance.all_pass
    }
    healthy = all(run["returncode"] == 0 for run in runs.values()) and all(
        finite_nonzero(trace) for trace in traces.values()
    )
    boundary_acceptance = None
    boundary_report = None
    production_acceptance = None
    production_report = None
    if "free_surface" in diagnostics:
        surface = diagnostics["free_surface"]
        boundary_acceptance = evaluate_surface_boundary(
            normalized_physical_traction=surface[
                "normalized_physical_traction_residual"
            ],
            physical_traction_limit=surface["physical_traction_limit"],
            max_velocity_parity_residual=surface[
                "maximum_vz_parity_residual"
            ],
            max_stress_parity_residual=surface[
                "maximum_syz_parity_residual"
            ],
            normalized_image_closure=surface[
                "normalized_image_closure_residual"
            ],
            image_closure_limit=surface["image_closure_limit"],
        )
        boundary_report = asdict(boundary_acceptance) | {
            "all_pass": boundary_acceptance.all_pass
        }
        production_acceptance = evaluate_production_acceptance(
            healthy=healthy,
            reflection=reflection_acceptance,
            boundary=boundary_acceptance,
        )
        production_report = {
            "healthy": production_acceptance.healthy,
            "reflection": reflection_report,
            "boundary": boundary_report,
            "all_pass": production_acceptance.all_pass,
        }
    report = {
        "phase": "M6.1b locked production-acceptance diagnostic",
        "fd_order": fd_order,
        "geometry": metadata,
        "observed": {
            "free_surface_peak_s": free_pick,
            "calibration_peak_s": calibration_pick,
            "timing_error_s": timing_error,
            "observed_propagation_s": observed_propagation,
            "shifted_surface_residuals_s": shifted_residuals,
            "signed_amplitude_alpha": alpha,
            "normalized_correlation": correlation,
            "absorbing_to_free_l2_ratio": absorbing_ratio,
        },
        "reflection_acceptance": reflection_report,
        "boundary_acceptance": boundary_report,
        "production_acceptance": production_report,
        "runs": runs,
        "seismogram_sha256": {
            role: run["seismogram_sha256"] for role, run in runs.items()
        },
        "instrumentation": diagnostics,
    }
    (tmp_path / "m61b_prefixed_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "M61B_REPORT "
        + json.dumps(
            {
                "fd_order": fd_order,
                "observed": report["observed"],
                "reflection_acceptance": report["reflection_acceptance"],
                "boundary_acceptance": report["boundary_acceptance"],
                "production_acceptance": report["production_acceptance"],
                "seismogram_sha256": report["seismogram_sha256"],
                "instrumentation": report["instrumentation"],
            },
            sort_keys=True,
        )
    )

    assert healthy
    if os.environ.get("M61_SH_ENFORCE_ACCEPTANCE") == "1":
        assert production_acceptance is not None, (
            "M61_SH_ENFORCE_ACCEPTANCE requires the retained diagnostic "
            "instrumentation"
        )
        assert production_acceptance.all_pass, json.dumps(
            production_report, sort_keys=True
        )
