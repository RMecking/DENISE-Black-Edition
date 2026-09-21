import re
import statistics
from pathlib import Path

import pytest

from tests.test_m8e1b2a_cuda_psv_forward import _build, _run


@pytest.fixture(scope="module")
def forward_binary(repository_root: Path) -> Path:
    return _build(repository_root)


def test_normal_forward_path_has_no_per_phase_blocking(
    repository_root: Path,
) -> None:
    source = (repository_root / "src/CUDA/psv_fd4_l1.cu").read_text(
        encoding="utf-8"
    )
    start = source.index("int denise_cuda_psv_forward_run(")
    end = source.index("int denise_cuda_psv_forward_download_traces", start)
    run = source[start:end]
    assert "timed_velocity" not in run
    assert "timed_stress" not in run
    assert "cudaDeviceSynchronize" not in run
    assert run.count("cudaEventSynchronize") == 1
    assert "DENISE_CUDA_PROFILE" in source

    harness = (
        repository_root / "tests/utilities/cuda_psv_forward_harness.c"
    ).read_text(encoding="utf-8")
    assert "CLOCK_MONOTONIC" in harness
    assert 'print_samples("cuda_host_e2e_ms"' in harness
    assert 'print_samples("cuda_internal_e2e_ms"' in harness
    assert 'print_samples("cuda_resident_propagation_ms"' in harness


def test_batched_profile_is_numerically_identical(forward_binary: Path) -> None:
    result = _run(
        [
            str(forward_binary),
            "--nx",
            "64",
            "--ny",
            "56",
            "--nt",
            "120",
            "--ntr",
            "3",
            "--profile-compare",
        ],
        cwd=forward_binary.parent,
    )
    assert result.returncode == 0, result.stdout
    assert re.search(
        r"TIMING_MODE profile=0 forward_syncs=1 per_timestep_syncs=0 .*"
        r"profile_event_records=0 profile_elapsed_queries=0",
        result.stdout,
    )
    profile = re.search(
        r"PROFILE_MODE diagnostic=1 numerical_traces_equal=1 "
        r"numerical_final_state_equal=1 forward_syncs=1 event_records=(\d+) "
        r"elapsed_queries=(\d+) velocity_ms=([0-9.]+) stress_ms=([0-9.]+) "
        r"source_ms=([0-9.]+) receiver_ms=([0-9.]+)",
        result.stdout,
    )
    assert profile
    assert int(profile.group(1)) == 4 * 120 + 1
    assert int(profile.group(2)) == 4 * 120
    assert all(float(profile.group(k)) > 0.0 for k in range(3, 7))
    assert "definitive_target_evidence=0" in result.stdout


def test_configurable_real_solver_benchmark(forward_binary: Path) -> None:
    result = _run(
        [
            str(forward_binary),
            "--benchmark",
            "--nx",
            "64",
            "--ny",
            "56",
            "--nt",
            "120",
            "--ntr",
            "3",
            "--warmup",
            "1",
            "--repetitions",
            "3",
        ],
        cwd=forward_binary.parent,
    )
    assert result.returncode == 0, result.stdout
    assert (
        "BENCHMARK_METHOD warmup=1 repetitions=3 fresh_cuda_context=1 "
        "primary=median profile=0 schedule=cpu_block_then_cuda_block "
        "clock=CLOCK_MONOTONIC "
        "headline=WARM_RUNTIME_SOLVER_END_TO_END_HOST_WALL "
        "cold_start_excluded=1 cleanup_included=1"
    ) in result.stdout
    assert "BENCHMARK_CASE nx=64 ny=56 nt=120 nrec=3" in result.stdout
    assert "managed_memory=0 paging_fallback=0" in result.stdout
    raw_values = {}
    for metric in (
        "cpu_host_e2e_ms",
        "cuda_host_e2e_ms",
        "cuda_internal_e2e_ms",
        "cuda_resident_propagation_ms",
        "cuda_internal_host_ratio",
    ):
        raw = re.search(rf"BENCHMARK_RAW metric={metric} values=([^\n]+)", result.stdout)
        assert raw and len(raw.group(1).split(",")) == 3
        raw_values[metric] = [float(value) for value in raw.group(1).split(",")]
    for metric in (
        "cpu_host_e2e_ms",
        "cuda_host_e2e_ms",
        "cuda_internal_e2e_ms",
        "cuda_resident_propagation_ms",
    ):
        assert f"BENCHMARK_SUMMARY metric={metric}" in result.stdout
    agreement = re.search(
        r"BENCHMARK_AGREEMENT .* warnings=(\d+) headline_valid=(\d+)",
        result.stdout,
    )
    assert agreement
    assert int(agreement.group(2)) == (int(agreement.group(1)) == 0)
    headline = re.search(
        r"BENCHMARK_HEADLINE valid=(\d+) cpu_host_median_ms=([0-9.]+) "
        r"cuda_host_median_ms=([0-9.]+) speedup=([0-9.]+) "
        r"runtime_fraction=([0-9.]+) runtime_reduction=([0-9.]+) "
        r"cuda_internal_median_ms=([0-9.]+) "
        r"cuda_resident_median_ms=([0-9.]+) resident_only_ratio=([0-9.]+) "
        r"cell_timesteps_per_second=([0-9.]+)",
        result.stdout,
    )
    assert headline
    cpu_median = statistics.median(raw_values["cpu_host_e2e_ms"])
    host_median = statistics.median(raw_values["cuda_host_e2e_ms"])
    resident_median = statistics.median(
        raw_values["cuda_resident_propagation_ms"]
    )
    assert int(headline.group(1)) == int(agreement.group(2))
    assert float(headline.group(2)) == pytest.approx(cpu_median, abs=1.0e-6)
    assert float(headline.group(3)) == pytest.approx(host_median, abs=1.0e-6)
    assert float(headline.group(4)) == pytest.approx(cpu_median / host_median, rel=1.0e-6)
    assert float(headline.group(5)) == pytest.approx(host_median / cpu_median, rel=1.0e-6)
    assert float(headline.group(6)) == pytest.approx(1.0 - host_median / cpu_median, rel=1.0e-6)
    assert float(headline.group(9)) == pytest.approx(cpu_median / resident_median, rel=1.0e-6)
    assert float(headline.group(4)) != pytest.approx(
        cpu_median / resident_median, rel=1.0e-3
    )
    for internal, host, ratio in zip(
        raw_values["cuda_internal_e2e_ms"],
        raw_values["cuda_host_e2e_ms"],
        raw_values["cuda_internal_host_ratio"],
    ):
        assert ratio == pytest.approx(internal / host, rel=1.0e-5)
    assert (
        "BENCHMARK_SYNC forward_syncs_per_run=1 per_timestep=0 "
        "profile_syncs=0 profile_event_records=0 profile_elapsed_queries=0"
    ) in result.stdout
