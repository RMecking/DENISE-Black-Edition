"""FP64 oracle gate for the persistent single-GPU P/SV adjoint state."""

from __future__ import annotations

import math
from pathlib import Path
import struct

import pytest

from tests.test_m8e1b2a_cuda_psv_forward import _build, _run
from tests.utilities.visco_psv_one_step_adjoint_reference import (
    ALL_FIELDS,
    build_fixture,
    inject_receiver_residuals,
    one_reverse_step,
    stress_gsls_transpose,
    velocity_transpose,
)


RTOL = 5.0e-13
ATOL = 5.0e-14
FIXTURES = (("standard", 24, 20), ("edge_non_square", 19, 27))


@pytest.fixture(scope="module", params=[False, True], ids=["normal", "no-fma"])
def adjoint_binary(repository_root: Path, request: pytest.FixtureRequest) -> Path:
    return _build(repository_root, no_fma=request.param)


def _read_snapshots(path: Path, count: int, nx: int, ny: int):
    values_per_field = (nx + 6) * (ny + 6)
    field_bytes = values_per_field * 8
    data = path.read_bytes()
    assert len(data) == count * len(ALL_FIELDS) * field_bytes
    snapshots = []
    offset = 0
    for _ in range(count):
        snapshot = {}
        for field in ALL_FIELDS:
            snapshot[field] = struct.unpack_from(
                f"<{values_per_field}d", data, offset
            )
            offset += field_bytes
        snapshots.append(snapshot)
    return snapshots


def _compare_fields(case: str, actual, expected, nx: int) -> None:
    pitch = nx + 6
    failures = []
    for field in ALL_FIELDS:
        worst_index = 0
        max_abs = -1.0
        max_rel = 0.0
        for index, (observed, reference) in enumerate(
            zip(actual[field], expected[field], strict=True)
        ):
            absolute = abs(observed - reference)
            relative = absolute / abs(reference) if reference else (
                0.0 if absolute == 0.0 else math.inf
            )
            if absolute > max_abs:
                max_abs = absolute
                worst_index = index
            max_rel = max(max_rel, relative)
            if absolute > ATOL + RTOL * abs(reference):
                failures.append((field, index, observed, reference, absolute, relative))
        row, column = divmod(worst_index, pitch)
        print(
            f"ADJOINT_ORACLE case={case} field={field} max_abs={max_abs:.17e} "
            f"max_rel={max_rel:.17e} index={worst_index} "
            f"j={row - 2} i={column - 2} pass={int(not any(f[0] == field for f in failures))}"
        )
    assert not failures, (
        f"{case}: {len(failures)} values exceed rtol={RTOL} atol={ATOL}; "
        f"first={failures[0]}"
    )


def _run_gate(binary: Path, output: Path, sequence: str, nx: int, ny: int) -> str:
    result = _run(
        [str(binary), "--adjoint-output", str(output),
         "--adjoint-sequence", sequence, "--nx", str(nx), "--ny", str(ny),
         "--fw", "4", "--nt", "2", "--ntr", "3"],
        cwd=binary.parent,
    )
    if "no CUDA device is visible" in result.stdout:
        pytest.skip("no CUDA device is visible")
    assert result.returncode == 0, result.stdout
    assert "ADJOINT_BUDGET_REJECT" in result.stdout
    assert "duplicate_rejected=1" in result.stdout
    assert "invalid_timestep_rejected=1" in result.stdout
    assert "allocation_per_step=0" in result.stdout
    assert "full_state_h2d_per_step=0" in result.stdout
    assert "full_state_d2h_per_step=0" in result.stdout
    assert "explicit_device_sync_per_step=0" in result.stdout
    assert "ADJOINT_DESTROY_PASS context_null=1" in result.stdout
    return result.stdout


@pytest.mark.parametrize("name,nx,ny", FIXTURES)
@pytest.mark.parametrize("timestep", [1, 2])
def test_one_step_matches_frozen_fp64_oracle(
    adjoint_binary: Path, tmp_path: Path, name: str, nx: int, ny: int,
    timestep: int,
) -> None:
    output = tmp_path / f"{name}-t{timestep}.bin"
    stdout = _run_gate(adjoint_binary, output, str(timestep), nx, ny)
    expected = one_reverse_step(build_fixture(name, nx, ny, 4), timestep)
    actual = _read_snapshots(output, 1, nx, ny)[0]
    _compare_fields(f"{name}-t{timestep}-{adjoint_binary.name}", actual, expected, nx)
    expected_residual_calls = 4 if timestep > 1 else 0
    assert f"residual_h2d_sync_calls={expected_residual_calls}" in stdout


@pytest.mark.parametrize("name,nx,ny", FIXTURES)
def test_repeated_t2_then_t1_preserves_persistent_state(
    adjoint_binary: Path, tmp_path: Path, name: str, nx: int, ny: int
) -> None:
    output = tmp_path / f"{name}-repeated.bin"
    stdout = _run_gate(adjoint_binary, output, "2,1", nx, ny)
    actual = _read_snapshots(output, 2, nx, ny)
    fixture = build_fixture(name, nx, ny, 4)
    fields = {field: values.copy() for field, values in fixture.fields.items()}
    expected = []
    for timestep in (2, 1):
        inject_receiver_residuals(fixture, fields, timestep)
        stress_gsls_transpose(fixture, fields)
        velocity_transpose(fixture, fields)
        expected.append({field: values.copy() for field, values in fields.items()})
    for step, (observed, reference) in enumerate(zip(actual, expected, strict=True), 1):
        _compare_fields(
            f"{name}-sequence-2,1-step{step}-{adjoint_binary.name}",
            observed, reference, nx,
        )
    assert "steps=2 kernels=10 residual_h2d_sync_calls=4" in stdout


def test_residual_upload_uses_synchronous_copy(repository_root: Path) -> None:
    source = (repository_root / "src/CUDA/psv_fd4_l1.cu").read_text()
    begin = source.index("int denise_cuda_psv_adjoint_step(")
    end = source.index("int denise_cuda_psv_adjoint_download(", begin)
    step = source[begin:end]
    assert "cudaMemcpyAsync" not in step
    assert "cudaMemcpy(" in step
    assert "cudaMemcpyHostToDevice" in step
