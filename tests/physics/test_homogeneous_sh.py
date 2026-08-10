from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import (
    all_finite,
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
    for receiver_x, trace, expected in zip(config.receiver_x_m, traces, config.analytical_travel_times()):
        index = first_break_index(trace, smoothing_samples=smoothing_samples)
        observed = (index + 1) * config.dt_s - source_delay
        error = observed - expected
        results.append(
            {"receiver_x_m": receiver_x, "expected_s": expected, "observed_s": observed, "error_s": error}
        )
        assert abs(error) <= tolerance, (
            f"receiver x={receiver_x} m: observed propagation time {observed:.6f} s, "
            f"analytical {expected:.6f} s, error {error:.6f} s, tolerance {tolerance:.6f} s"
        )
    (tmp_path / "one_rank" / "travel_time_metrics.json").write_text(
        json.dumps({"tolerance_s": tolerance, "receivers": results}, indent=2) + "\n", encoding="utf-8"
    )


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
    _, two_ranks = _run_case(
        tmp_path / "two_ranks",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        nprocx=2,
        nprocy=1,
    )
    rel_error = relative_l2(one_rank, two_ranks)
    correlation = normalized_correlation(one_rank, two_ranks)
    metrics = {"relative_l2": rel_error, "normalized_correlation": correlation}
    (tmp_path / "mpi_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    assert rel_error <= 1.0e-5
    assert correlation >= 0.999999
