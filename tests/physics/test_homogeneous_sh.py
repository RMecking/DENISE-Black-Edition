from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import (
    all_finite,
    fit_propagation_velocity,
    first_break_index,
    normalized_correlation,
    read_ascii_seismograms,
    relative_l2,
    source_pick_delay,
)


pytestmark = pytest.mark.integration


def _run_case(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    nprocx: int,
    nprocy: int,
) -> tuple[HomogeneousSHConfig, list[list[float]]]:
    config = generate_case(directory, nprocx=nprocx, nprocy=nprocy)
    metadata = config.as_metadata() | {"nprocx": nprocx, "nprocy": nprocy}
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration=metadata,
    )
    assert result.returncode == 0, result_summary(result)
    output = directory / "su" / "homogeneous_vz.asc.shot1"
    assert output.is_file() and output.stat().st_size > 0
    traces = read_ascii_seismograms(output, config.receiver_count, config.samples_per_trace)
    assert all_finite(traces)
    return config, traces


def test_homogeneous_elastic_sh_travel_times(tmp_path, repository_root, denise_binary, mpiexec):
    config, traces = _run_case(
        tmp_path / "one_rank",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        nprocx=1,
        nprocy=1,
    )
    smoothing_samples = round(0.25 / config.source_frequency_hz / config.dt_s)
    source_delay = source_pick_delay(
        samples=config.samples_per_trace,
        dt=config.dt_s,
        frequency_hz=config.source_frequency_hz,
        smoothing_samples=smoothing_samples,
    )
    tolerance = 2.0 * config.dt_s + 0.25 / config.source_frequency_hz
    results = []
    raw_pick_times = []
    offsets = config.receiver_offsets_m()
    for receiver_x, offset, trace, expected in zip(
        config.receiver_x_m, offsets, traces, config.analytical_travel_times()
    ):
        index = first_break_index(trace, smoothing_samples=smoothing_samples)
        raw_pick = (index + 1) * config.dt_s
        raw_pick_times.append(raw_pick)
        observed = raw_pick - source_delay
        error = observed - expected
        results.append(
            {
                "receiver_x_m": receiver_x,
                "offset_m": offset,
                "raw_pick_s": raw_pick,
                "expected_s": expected,
                "observed_s": observed,
                "error_s": error,
            }
        )
        assert abs(error) <= tolerance, (
            f"receiver x={receiver_x} m: observed propagation time {observed:.6f} s, "
            f"analytical {expected:.6f} s, error {error:.6f} s, tolerance {tolerance:.6f} s"
        )

    velocity_fit = fit_propagation_velocity(offsets, raw_pick_times)
    relative_velocity_error = abs(velocity_fit.velocity_m_s - config.vs_m_s) / config.vs_m_s
    for result, residual in zip(results, velocity_fit.residuals_s):
        result["fit_residual_s"] = residual
    fit_metrics = {
        "fitted_vs_m_s": velocity_fit.velocity_m_s,
        "model_vs_m_s": config.vs_m_s,
        "relative_vs_error": relative_velocity_error,
        "fitted_intercept_s": velocity_fit.intercept_s,
        "maximum_absolute_residual_s": velocity_fit.maximum_absolute_residual_s,
    }
    (tmp_path / "one_rank" / "travel_time_metrics.json").write_text(
        json.dumps(
            {"diagnostic_absolute_tolerance_s": tolerance, "velocity_fit": fit_metrics, "receivers": results},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    assert relative_velocity_error <= 0.01, (
        f"fitted Vs {velocity_fit.velocity_m_s:.3f} m/s differs from model Vs "
        f"{config.vs_m_s:.3f} m/s by {relative_velocity_error:.3%}"
    )
    assert velocity_fit.maximum_absolute_residual_s <= 2.0 * config.dt_s


def test_homogeneous_elastic_sh_mpi_reproducibility(
    tmp_path, repository_root, denise_binary, mpiexec
):
    _, one_rank = _run_case(
        tmp_path / "one_rank",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        nprocx=1,
        nprocy=1,
    )
    metrics = {}
    for label, nprocx, nprocy in (("2x1", 2, 1), ("1x2", 1, 2), ("2x2", 2, 2)):
        _, variant = _run_case(
            tmp_path / f"mpi_{label}",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            nprocx=nprocx,
            nprocy=nprocy,
        )
        rel_error = relative_l2(one_rank, variant)
        correlation = normalized_correlation(one_rank, variant)
        metrics[label] = {
            "mpi_ranks": nprocx * nprocy,
            "relative_l2": rel_error,
            "normalized_correlation": correlation,
        }
        assert rel_error <= 1.0e-5, f"{label} relative L2 error: {rel_error}"
        assert correlation >= 0.999999, f"{label} normalized correlation: {correlation}"
    (tmp_path / "mpi_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
