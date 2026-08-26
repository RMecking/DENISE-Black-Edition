"""M6.1b.2 runtime oracles that must exist before the production repair."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tests.cases.sh_free_surface import (
    SHFreeSurfaceScenario,
    generate_case,
    normal_scenario,
    oblique_scenario,
)
from tests.physics.test_sh_free_surface import _run, _window_at_pick
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import read_ascii_seismograms
from tests.utilities.sh_free_surface import (
    SurfaceTimes,
    evaluate_production_acceptance,
    evaluate_reflection,
    evaluate_surface_boundary,
    finite_nonzero,
    holberg_coefficients,
    normalized_correlation,
    relative_l2,
    signed_amplitude_alpha,
    stability_modulation_limit,
)
from tests.utilities.sh_free_surface_runtime import (
    denise_ricker_reference,
    post_source_quarters,
)


pytestmark = [pytest.mark.integration, pytest.mark.extended]

FDORDERS = (2, 4, 6, 8, 10, 12)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _boundary_acceptance(diagnostic):
    if diagnostic is None:
        return None
    return evaluate_surface_boundary(
        normalized_physical_traction=diagnostic[
            "normalized_physical_traction_residual"
        ],
        physical_traction_limit=diagnostic["physical_traction_limit"],
        max_velocity_parity_residual=diagnostic[
            "maximum_vz_parity_residual"
        ],
        max_stress_parity_residual=diagnostic[
            "maximum_syz_parity_residual"
        ],
        normalized_image_closure=diagnostic[
            "normalized_image_closure_residual"
        ],
        image_closure_limit=diagnostic["image_closure_limit"],
    )


def _reflection_runtime(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    scenario: SHFreeSurfaceScenario,
) -> dict[str, object]:
    metadata = scenario.metadata()
    assert metadata["surface_location_resolved"] is True
    assert (
        metadata["expected_reflection_peak_s"]
        - metadata["expected_direct_peak_s"]
        > 2.0 * scenario.reflection_window_half_width_s
    )

    traces: dict[str, list[float]] = {}
    runs: dict[str, object] = {}
    diagnostics: dict[str, object] = {}
    for role in ("free_surface", "absorbing", "calibration"):
        trace, run, diagnostic = _run(
            directory / role,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            scenario=scenario,
            role=role,
        )
        traces[role] = trace
        runs[role] = run
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

    free_norm = math.sqrt(math.fsum(value * value for value in free_window))
    timing_error = abs(free_pick - calibration_pick)
    surface_times = SurfaceTimes(
        **metadata["surface_candidate_propagation_times_s"]
    )
    observed_propagation = surface_times.y0_s + free_pick - calibration_pick
    reflection = evaluate_reflection(
        timing_error_s=timing_error,
        observed_propagation_s=observed_propagation,
        surface_times=surface_times,
        timing_tolerance_s=metadata["timing_tolerance_s"],
        signed_amplitude_alpha_value=signed_amplitude_alpha(
            free_window, calibration_window
        ),
        normalized_correlation_value=normalized_correlation(
            free_window, calibration_window
        ),
        absorbing_l2_ratio=(
            math.sqrt(math.fsum(value * value for value in absorbing_window))
            / free_norm
        ),
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
    boundary = _boundary_acceptance(diagnostics.get("free_surface"))
    healthy = all(run["returncode"] == 0 for run in runs.values()) and all(
        finite_nonzero(trace) for trace in traces.values()
    )
    production = (
        evaluate_production_acceptance(
            healthy=healthy, reflection=reflection, boundary=boundary
        )
        if boundary is not None
        else None
    )
    report = {
        "scenario": metadata["name"],
        "fd_order": scenario.free_surface.fd_order,
        "healthy": healthy,
        "reflection": asdict(reflection) | {"all_pass": reflection.all_pass},
        "boundary": (
            asdict(boundary) | {"all_pass": boundary.all_pass}
            if boundary is not None
            else None
        ),
        "production": (
            {"all_pass": production.all_pass} if production is not None else None
        ),
        "metrics": {
            "free_surface_peak_s": free_pick,
            "calibration_peak_s": calibration_pick,
            "timing_error_s": timing_error,
            "observed_propagation_s": observed_propagation,
            "signed_amplitude_alpha": signed_amplitude_alpha(
                free_window, calibration_window
            ),
            "normalized_correlation": normalized_correlation(
                free_window, calibration_window
            ),
            "absorbing_l2_ratio": (
                math.sqrt(
                    math.fsum(value * value for value in absorbing_window)
                )
                / free_norm
            ),
        },
        "diagnostics": diagnostics,
        "runs": runs,
    }
    (directory / "m61b2_reflection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("M61B2_REFLECTION " + json.dumps(report, sort_keys=True))
    assert healthy
    assert reflection.all_pass, json.dumps(report, sort_keys=True)
    if os.environ.get("M61_SH_ENFORCE_ACCEPTANCE") == "1":
        assert production is not None, (
            "M61_SH_ENFORCE_ACCEPTANCE requires the retained diagnostics"
        )
        assert production.all_pass, json.dumps(report, sort_keys=True)
    return report


def test_oblique_reflection_runtime(
    tmp_path, repository_root, denise_binary, mpiexec
):
    report = _reflection_runtime(
        tmp_path,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        scenario=oblique_scenario(fd_order=4),
    )
    if report["boundary"] is not None:
        assert report["reflection"]["all_pass"] is True


def test_oblique_mpi_runtime_reproducibility(
    tmp_path, repository_root, denise_binary, mpiexec
):
    scenario = oblique_scenario(fd_order=4)
    metadata = scenario.metadata()
    native = metadata["native_vz_coordinates_m"]
    x_seam_m = scenario.free_surface.nx * scenario.free_surface.dh_m / 2.0
    y_seam_m = scenario.free_surface.ny * scenario.free_surface.dh_m / 2.0
    assert native["source"][0] < x_seam_m < native["receiver"][0]
    assert max(native["source"][1], native["receiver"][1]) < y_seam_m

    variants = {"1x1": (1, 1), "2x1": (2, 1), "1x2": (1, 2)}
    traces: dict[str, list[float]] = {}
    inputs: dict[str, dict[str, str]] = {}
    runs: dict[str, object] = {}
    for label, (nprocx, nprocy) in variants.items():
        case = tmp_path / label
        trace, run, _ = _run(
            case,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            scenario=scenario,
            role="free_surface",
            nprocx=nprocx,
            nprocy=nprocy,
        )
        traces[label] = trace
        runs[label] = run
        inputs[label] = {
            name: _hash(case / name)
            for name in (
                "model/homogeneous.vs",
                "model/homogeneous.rho",
                "source.dat",
                "receiver.dat",
            )
        }
    assert inputs["1x1"] == inputs["2x1"] == inputs["1x2"]

    comparisons = {}
    for label in ("2x1", "1x2"):
        comparison = {
            "relative_l2": relative_l2(traces["1x1"], traces[label]),
            "normalized_correlation": normalized_correlation(
                traces["1x1"], traces[label]
            ),
        }
        comparisons[label] = comparison
        assert comparison["relative_l2"] <= metadata["acceptance"][
            "mpi_relative_l2_max"
        ]
        assert comparison["normalized_correlation"] >= metadata["acceptance"][
            "mpi_correlation_min"
        ]
    report = {
        "geometry": {
            "x_seam_m": x_seam_m,
            "y_seam_m": y_seam_m,
            "native_coordinates_m": native,
            "oblique_path_crosses_2x1_x_seam": True,
            "1x2_y_seam_below_reflection_region": True,
        },
        "input_sha256": inputs,
        "comparisons": comparisons,
        "runs": runs,
    }
    (tmp_path / "m61b2_mpi.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("M61B2_MPI " + json.dumps(report, sort_keys=True))


@pytest.mark.parametrize("fd_order", FDORDERS)
def test_extended_fdorder_runtime_definition(
    tmp_path, repository_root, denise_binary, mpiexec, fd_order
):
    _reflection_runtime(
        tmp_path,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        scenario=normal_scenario(fd_order),
    )


def _stability_scenario() -> SHFreeSurfaceScenario:
    normal = normal_scenario(fd_order=12)
    stability = replace(
        normal.free_surface,
        time_s=2.6005,
        receiver_x_m=tuple(float(x) for x in range(200, 2201, 50)),
        receiver_y_m=100.0,
    )
    return replace(normal, free_surface=stability)


def test_fdorder12_stability_runtime(
    tmp_path, repository_root, denise_binary, mpiexec
):
    scenario = _stability_scenario()
    config = generate_case(tmp_path, scenario=scenario, role="free_surface")
    assert config.fd_order == 12
    assert config.time_s >= 2.0 * normal_scenario(12).free_surface.time_s
    parameters = (tmp_path / "denise.inp").read_text(encoding="ascii")
    assert "MODE =0" in parameters
    assert "QUELLART =1" in parameters
    source_scope = {"quellart": 1, "n_order": 0}
    assert source_scope == {"quellart": 1, "n_order": 0}

    reference = denise_ricker_reference(
        nt=config.samples_per_trace,
        dt_s=config.dt_s,
        frequency_hz=config.source_frequency_hz,
        amplitude=1.0,
        timeshift_s=0.0,
        **source_scope,
    )
    assert reference.n_off == 1257
    quarters = post_source_quarters(
        nt=config.samples_per_trace, n_off=reference.n_off
    )
    case_metadata = json.loads((tmp_path / "case.json").read_text(encoding="utf-8"))
    case_metadata["source_off"] = {
        **source_scope,
        "n_off": reference.n_off,
        "nominal_t_off_s": reference.n_off * config.dt_s,
        "quarter_size": quarters.quarter_size,
        "quarter_bounds": quarters.inclusive_bounds,
    }
    (tmp_path / "case.json").write_text(
        json.dumps(case_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = run_denise(
        repository_root=repository_root,
        case_directory=tmp_path,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration=case_metadata,
        timeout_seconds=300.0,
    )
    assert result.returncode == 0, result_summary(result)
    output = tmp_path / "su" / "homogeneous_vz.asc.shot1"
    assert output.is_file() and output.stat().st_size > 0
    traces = read_ascii_seismograms(
        output, config.receiver_count, config.samples_per_trace
    )
    assert len(traces) == config.receiver_count
    assert all(finite_nonzero(trace) for trace in traces)

    diagnostic_path = tmp_path / "m61b_diagnostics.csv"
    assert diagnostic_path.is_file(), (
        "M6.1 stability requires the retained instrumented binary and "
        "M61B_SH_DIAGNOSTICS=m61b_diagnostics.csv"
    )
    with diagnostic_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["timestep"]) for row in rows] == list(
        range(1, config.samples_per_trace + 1)
    )
    energies = [float(row["centered_energy"]) for row in rows]
    assert all(math.isfinite(value) for value in energies)

    def maximum(first: int, last: int) -> float:
        return max(energies[first - 1 : last])

    active_max = maximum(1, reference.n_off)
    post_max = maximum(reference.n_off + 1, config.samples_per_trace)
    q1_max = maximum(*quarters.inclusive_bounds[0])
    q4_max = maximum(*quarters.inclusive_bounds[3])
    delta_e = stability_modulation_limit(
        dt_s=config.dt_s,
        f95_hz=scenario.metadata()["source_spectrum"]["f95_hz"],
        coefficients=holberg_coefficients(config.fd_order),
    )
    report = {
        "source_scope": source_scope,
        "n_off": reference.n_off,
        "nominal_t_off_s": reference.n_off * config.dt_s,
        "quarter_bounds": quarters.inclusive_bounds,
        "delta_e": delta_e,
        "energy": {
            "active_max": active_max,
            "post_max": post_max,
            "q1_max": q1_max,
            "q4_max": q4_max,
            "post_to_active": post_max / active_max,
            "q4_to_q1": q4_max / q1_max,
        },
        "returncode": result.returncode,
        "receiver_count": config.receiver_count,
    }
    (tmp_path / "m61b2_stability.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("M61B2_STABILITY " + json.dumps(report, sort_keys=True))
    assert post_max <= (1.0 + delta_e) * active_max
    assert q4_max <= (1.0 + delta_e) * q1_max
