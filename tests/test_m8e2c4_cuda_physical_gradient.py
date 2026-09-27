"""M8e-2C4 device-resident physical gradient scientific gate."""

from __future__ import annotations

import math
from pathlib import Path
import struct

import pytest

from tests.test_m8e1b2a_cuda_psv_forward import _build, _run
from tests.utilities.cuda_psv_physical_gradient_reference import (
    MATERIAL_FIELDS,
    NATIVE_FIELDS,
    PHYSICAL_FIELDS,
    canonical_q_mapping,
    f32,
    physical_gradient_reference,
)


RTOL = 5.0e-13
ATOL = 5.0e-14
FIXTURES = (("standard", 24, 20), ("edge_non_square", 19, 27))


@pytest.fixture(scope="module", params=[False, True], ids=["normal", "no-fma"])
def physical_binary(repository_root: Path, request: pytest.FixtureRequest) -> Path:
    return _build(repository_root, no_fma=request.param)


def _read_gate(path: Path, nx: int, ny: int):
    data, cells, offset = path.read_bytes(), nx * ny, 0
    q_mode, eta = struct.unpack_from("<if", data, offset)
    offset += 8
    per_q, q_offset = struct.unpack_from("<dd", data, offset)
    offset += 16
    native = {}
    for field in NATIVE_FIELDS:
        native[field] = struct.unpack_from(f"<{cells}d", data, offset)
        offset += cells * 8
    physical = {}
    for field in PHYSICAL_FIELDS:
        physical[field] = struct.unpack_from(f"<{cells}d", data, offset)
        offset += cells * 8
    material = {}
    for field in MATERIAL_FIELDS:
        material[field] = struct.unpack_from(f"<{cells}f", data, offset)
        offset += cells * 4
    assert offset == len(data)
    return q_mode, eta, per_q, q_offset, native, physical, material


def _run_gate(binary: Path, output: Path, nx: int, ny: int, *, q_mode: int,
              nt: int = 11, segments: int = 4):
    result = _run(
        [str(binary), "--physical-gradient-output", str(output),
         "--physical-gradient-segments", str(segments),
         "--physical-gradient-q-mode", str(q_mode),
         "--nx", str(nx), "--ny", str(ny), "--fw", "4",
         "--nt", str(nt), "--ntr", "3"],
        cwd=binary.parent,
    )
    if "no CUDA device is visible" in result.stdout:
        pytest.skip("no CUDA device is visible")
    assert result.returncode == 0, result.stdout
    cells = nx * ny
    material_bytes, gradient_bytes = 7 * cells * 4, 5 * cells * 8
    alignment = (-material_bytes) % 8
    assert (
        f"PHYSICAL_GRADIENT_PLAN material={material_bytes} "
        f"gradients={gradient_bytes} alignment={alignment} "
        f"validation=4 total={material_bytes + alignment + gradient_bytes + 4}"
    ) in result.stdout
    assert "allocation=1 uploads=7" in result.stdout
    assert f"upload_bytes={material_bytes} zero=1" in result.stdout
    assert "PHYSICAL_GRADIENT_RESIDENCY per_step_h2d=0 per_step_d2h=0" in result.stdout
    assert "per_step_alloc=0 per_step_free=0 per_step_sync=0" in result.stdout
    assert "map_kernels=1 map_sync=1 validation_d2h=1 validation_bytes=4" in result.stdout
    assert "d2h_calls=5" in result.stdout
    assert (
        "PHYSICAL_GRADIENT_FAILURES premature_map=1 premature_download=1 "
        "double_map=1 budget_one_byte=1 invalid_q=1"
    ) in result.stdout
    assert (
        f"PHYSICAL_GRADIENT_PASS nx={nx} ny={ny} nt={nt} "
        f"segments={segments} q_mode={q_mode} fields=5 native_equal=1 "
        "native_unchanged=1 adjoint_equal=1 operands_exact=1"
    ) in result.stdout
    assert (
        "PHYSICAL_GRADIENT_LIFECYCLE prepared_only=1 complete=1 "
        "failed_prepare=1 contexts_null=1"
    ) in result.stdout
    return result.stdout, _read_gate(output, nx, ny)


def _compare(case: str, actual, expected):
    failures = []
    maxima = {}
    for field in PHYSICAL_FIELDS:
        worst = (0.0, 0.0, 0)
        for index, (observed, reference) in enumerate(
            zip(actual[field], expected[field], strict=True)
        ):
            absolute = abs(observed - reference)
            relative = absolute / abs(reference) if reference else (
                0.0 if absolute == 0.0 else math.inf
            )
            if absolute > worst[0]:
                worst = (absolute, relative, index)
            if absolute > ATOL + RTOL * abs(reference):
                failures.append((field, index, observed, reference,
                                 absolute, relative))
        maxima[field] = worst
        print(
            f"PHYSICAL_GRADIENT_ORACLE case={case} field={field} "
            f"max_abs={worst[0]:.17e} max_rel={worst[1]:.17e} "
            f"index={worst[2]} pass={int(not any(row[0] == field for row in failures))}"
        )
    assert not failures, (
        f"{case}: first physical-map mismatch={failures[0]}; "
        "isolate cell-centered/corner/face-density/Q-chain before repair"
    )
    return maxima


@pytest.mark.parametrize("q_mode", [0, 1], ids=["legacy-q", "physical-q"])
@pytest.mark.parametrize("name,nx,ny", FIXTURES)
def test_all_five_physical_fields_against_independent_reference(
    physical_binary: Path, tmp_path: Path, name: str, nx: int, ny: int,
    q_mode: int,
) -> None:
    output = tmp_path / f"{name}-{physical_binary.name}-q{q_mode}.bin"
    _stdout, payload = _run_gate(
        physical_binary, output, nx, ny, q_mode=q_mode
    )
    actual_mode, eta, per_q, offset, native, actual, material = payload
    assert actual_mode == q_mode
    expected_per_q, expected_offset = canonical_q_mapping(q_mode)
    assert per_q == pytest.approx(expected_per_q, rel=0.0, abs=2.0e-15)
    assert offset == pytest.approx(expected_offset, rel=0.0, abs=2.0e-15)
    expected = physical_gradient_reference(
        native, material, nx, ny, f32(0.0125), eta, q_mode, per_q, offset
    )
    _compare(f"{name}-{physical_binary.name}-q{q_mode}", actual, expected)
    for field in PHYSICAL_FIELDS:
        assert all(math.isfinite(value) for value in actual[field])
        assert any(value != 0.0 for value in actual[field])
    assert len({actual[field] for field in PHYSICAL_FIELDS}) == 5
    boundary_indices = (0, nx - 1, (ny - 1) * nx, nx * ny - 1)
    for field in PHYSICAL_FIELDS:
        assert len({actual[field][index] for index in boundary_indices}) > 1


def test_physical_gradient_is_bit_exact_across_segmentation(
    physical_binary: Path, tmp_path: Path,
) -> None:
    nx, ny, nt, q_mode = 24, 20, 11, 1
    fields = {}
    natives = {}
    for segments in (1, 4, nt):
        output = tmp_path / f"segments-{physical_binary.name}-s{segments}.bin"
        _stdout, payload = _run_gate(
            physical_binary, output, nx, ny, q_mode=q_mode,
            nt=nt, segments=segments
        )
        native, physical = payload[4], payload[5]
        natives[segments] = b"".join(
            struct.pack(f"<{nx * ny}d", *native[field])
            for field in NATIVE_FIELDS
        )
        fields[segments] = b"".join(
            struct.pack(f"<{nx * ny}d", *physical[field])
            for field in PHYSICAL_FIELDS
        )
    assert natives[1] == natives[4] == natives[nt]
    assert fields[1] == fields[4] == fields[nt]


def test_physical_contract_is_explicit_and_mode_one_stays_inactive(
    repository_root: Path,
) -> None:
    header = (repository_root / "include/denise_cuda_psv_forward.h").read_text()
    source = (repository_root / "src/CUDA/psv_fd4_l1.cu").read_text()
    dispatch = (repository_root / "src/PSV/psv_cuda_dispatch.c").read_text()
    assert "Compact field order Vp,Vs,rho,Qp,Qs" in header
    assert "init_q_tau_mapping()" in header
    assert "physical_gradient_map_kernel" in source
    assert "One thread owns one output cell" in source
    assert "denise_cuda_psv_physical_gradient_map" not in dispatch
