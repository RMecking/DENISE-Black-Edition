"""Complete segmented GPU reverse-state sweep against the frozen FP64 oracle."""

from __future__ import annotations

import math
from pathlib import Path
import struct

import pytest

from tests.test_m8e1b2a_cuda_psv_forward import _build, _run
from tests.utilities.visco_psv_one_step_adjoint_reference import (
    ALL_FIELDS,
    build_fixture,
    stress_gsls_transpose,
    velocity_transpose,
)


RTOL = 5.0e-13
ATOL = 5.0e-14
FIXTURES = (("standard", 24, 20), ("edge_non_square", 19, 27))


@pytest.fixture(scope="module", params=[False, True], ids=["normal", "no-fma"])
def sweep_binary(repository_root: Path, request: pytest.FixtureRequest) -> Path:
    return _build(repository_root, no_fma=request.param)


def _read_gate(path: Path, snapshots: int, nx: int, ny: int, nt: int, ntr: int):
    data = path.read_bytes()
    samples = nt * ntr
    offset = 0
    traces = []
    for _ in range(4):
        traces.append(struct.unpack_from(f"<{samples}f", data, offset))
        offset += samples * 4
    count = (nx + 6) * (ny + 6)
    result = []
    for _ in range(snapshots):
        state = {}
        for field in ALL_FIELDS:
            state[field] = struct.unpack_from(f"<{count}d", data, offset)
            offset += count * 8
        result.append(state)
    assert offset == len(data)
    return traces, result


def _inject_resident(fixture, fields, timestep: int, traces, nt: int) -> None:
    if timestep <= 1:
        return
    modeled_vx, observed_vx, modeled_vy, observed_vy = traces
    for receiver, (j, i, *_unused) in enumerate(fixture.receivers):
        sample = receiver * nt + timestep - 1
        point = fixture.index(j, i)
        fields["avx"][point] += modeled_vx[sample] - observed_vx[sample]
        fields["avy"][point] += modeled_vy[sample] - observed_vy[sample]


def _reference_boundaries(name: str, nx: int, ny: int, nt: int,
                          segments: int, traces):
    fixture = build_fixture(name, nx, ny, 4)
    fields = {field: values.copy() for field, values in fixture.fields.items()}
    boundaries = []
    for segment in range(segments - 1, -1, -1):
        begin = segment * nt // segments + 1
        end = (segment + 1) * nt // segments
        for timestep in range(end, begin - 1, -1):
            _inject_resident(fixture, fields, timestep, traces, nt)
            stress_gsls_transpose(fixture, fields)
            velocity_transpose(fixture, fields)
        boundaries.append({field: values.copy() for field, values in fields.items()})
    return boundaries


def _compare(case: str, actual, expected, nx: int):
    pitch = nx + 6
    failures = []
    worst = ("", -1.0, 0.0, 0)
    for field in ALL_FIELDS:
        for index, (observed, reference) in enumerate(
            zip(actual[field], expected[field], strict=True)
        ):
            absolute = abs(observed - reference)
            relative = absolute / abs(reference) if reference else (
                0.0 if absolute == 0.0 else math.inf
            )
            if absolute > worst[1]:
                worst = (field, absolute, relative, index)
            if absolute > ATOL + RTOL * abs(reference):
                failures.append((field, index, observed, reference, absolute, relative))
    row, column = divmod(worst[3], pitch)
    print(
        f"SEGMENTED_ADJOINT_ORACLE case={case} fields=16 pass={int(not failures)} "
        f"worst_field={worst[0]} max_abs={worst[1]:.17e} "
        f"max_rel={worst[2]:.17e} index={worst[3]} "
        f"j={row - 2} i={column - 2}"
    )
    assert not failures, (
        f"{case}: {len(failures)} values exceed rtol={RTOL} atol={ATOL}; "
        f"first={failures[0]}"
    )
    return worst


def _run_sweep(binary: Path, output: Path, nx: int, ny: int,
               *, nt: int = 11, segments: int = 4):
    result = _run(
        [str(binary), "--adjoint-sweep-output", str(output),
         "--adjoint-sweep-segments", str(segments),
         "--nx", str(nx), "--ny", str(ny), "--fw", "4",
         "--nt", str(nt), "--ntr", "3"],
        cwd=binary.parent,
    )
    if "no CUDA device is visible" in result.stdout:
        pytest.skip("no CUDA device is visible")
    assert result.returncode == 0, result.stdout
    assert "ADJOINT_SWEEP_RESIDENCY observed_h2d_calls=2" in result.stdout
    assert "per_step_residual_h2d=0" in result.stdout
    assert "per_step_full_h2d=0" in result.stdout
    assert "per_step_full_d2h=0" in result.stdout
    assert "per_step_alloc=0" in result.stdout
    assert "per_step_free=0" in result.stdout
    assert "per_step_sync=0" in result.stdout
    assert f"replay_syncs={segments}" in result.stdout
    assert f"reverse_segment_syncs={segments}" in result.stdout
    assert (
        "ADJOINT_SWEEP_FAILURES no_bank=1 incomplete_bank=1 "
        "incomplete_forward=1 missing_observed=1 budget_one_byte=1 "
        "out_of_order=1 repeated=1 skipped=1 "
        "adjoint_usable_after_budget_reject=1"
    ) in result.stdout
    assert (
        "PRODUCTION_RESIDUAL_FAILURES unavailable=1 wrong_count=1 "
        "duplicate=1 partial_upload=1 stale_reuse=1 h2d_calls=2 "
        "observed_h2d_calls=0"
    ) in result.stdout
    assert (
        f"ADJOINT_SWEEP_PASS nx={nx} ny={ny} nt={nt} ntr=3 "
        f"segments={segments} reverse_segments={segments} reverse_steps={nt}"
    ) in result.stdout
    assert "operands_exact=1 complete=1" in result.stdout
    assert (
        "ADJOINT_SWEEP_DESTROY_PASS observed_storage_prepared=1 "
        "contexts_null=1"
    ) in result.stdout
    return result.stdout


@pytest.mark.parametrize("name,nx,ny", FIXTURES)
def test_uneven_segment_boundaries_and_full_sweep(
    sweep_binary: Path, tmp_path: Path, name: str, nx: int, ny: int,
) -> None:
    nt, segments = 11, 4
    output = tmp_path / f"{name}-{sweep_binary.name}.bin"
    _run_sweep(sweep_binary, output, nx, ny, nt=nt, segments=segments)
    traces, actual = _read_gate(output, segments + 1, nx, ny, nt, 3)
    expected = _reference_boundaries(name, nx, ny, nt, segments, traces)
    for boundary, (observed, reference) in enumerate(
        zip(actual[:segments], expected, strict=True)
    ):
        segment = segments - 1 - boundary
        _compare(
            f"{name}-{sweep_binary.name}-after-segment-{segment}",
            observed, reference, nx,
        )
    _compare(f"{name}-{sweep_binary.name}-full", actual[-1], expected[-1], nx)


def test_one_timestep_segments_extreme(
    sweep_binary: Path, tmp_path: Path,
) -> None:
    name, nx, ny, nt, segments = "standard", 24, 20, 11, 11
    output = tmp_path / f"one-step-segments-{sweep_binary.name}.bin"
    _run_sweep(sweep_binary, output, nx, ny, nt=nt, segments=segments)
    traces, actual = _read_gate(output, segments + 1, nx, ny, nt, 3)
    expected = _reference_boundaries(name, nx, ny, nt, segments, traces)
    _compare(f"one-step-{sweep_binary.name}-full", actual[-1], expected[-1], nx)


def test_state_only_sweep_does_not_arm_native_gradient(repository_root: Path) -> None:
    source = (repository_root / "src/CUDA/psv_fd4_l1.cu").read_text()
    begin = source.index("int denise_cuda_psv_adjoint_sweep_prepare(")
    end = source.index("int denise_cuda_psv_adjoint_reverse_segment(", begin)
    prepare = source[begin:end]
    assert "denise_cuda_psv_native_gradient_prepare" not in prepare
    launcher_begin = source.index("int launch_adjoint_operator(")
    launcher_end = source.index("}  // namespace", launcher_begin)
    launcher = source[launcher_begin:launcher_end]
    assert launcher.count("if(f->native_gradient_storage)") == 3


def test_public_destructor_owns_observed_storage(repository_root: Path) -> None:
    source = (repository_root / "src/CUDA/psv_fd4_l1.cu").read_text()
    begin = source.index("int denise_cuda_psv_forward_destroy(")
    end = source.index("}  // extern \"C\"", begin)
    destructor = source[begin:end]
    assert "f->adjoint_observed_storage" in destructor
    assert "cudaFree(f->adjoint_observed_storage)" in destructor
    assert destructor.index("cudaFree(f->adjoint_observed_storage)") < (
        destructor.index("delete f;")
    )
