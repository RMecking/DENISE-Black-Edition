"""M8e-2C3 native CUDA gradient against an independent FP64 reference."""

from __future__ import annotations

import math
from pathlib import Path
import struct

import pytest

from tests.test_m8e1b2a_cuda_psv_forward import _build, _run
from tests.utilities.cuda_psv_native_gradient_reference import (
    NATIVE_FIELDS,
    OPERAND_FIELDS,
    native_gradient_boundaries,
)
from tests.utilities.visco_psv_one_step_adjoint_reference import build_fixture


RTOL = 5.0e-13
ATOL = 5.0e-14
FIXTURES = (("standard", 24, 20), ("edge_non_square", 19, 27))


@pytest.fixture(scope="module", params=[False, True], ids=["normal", "no-fma"])
def gradient_binary(repository_root: Path, request: pytest.FixtureRequest) -> Path:
    return _build(repository_root, no_fma=request.param)


def _read_gate(path: Path, nx: int, ny: int, nt: int, ntr: int, segments: int):
    data = path.read_bytes()
    samples, cells, offset = nt * ntr, nx * ny, 0
    traces = []
    for _ in range(4):
        traces.append(struct.unpack_from(f"<{samples}f", data, offset))
        offset += samples * 4
    operands = {}
    for field in OPERAND_FIELDS:
        operands[field] = struct.unpack_from(f"<{nt * cells}f", data, offset)
        offset += nt * cells * 4
    boundaries = []
    for _ in range(segments):
        gradient = {}
        for field in NATIVE_FIELDS:
            gradient[field] = struct.unpack_from(f"<{cells}d", data, offset)
            offset += cells * 8
        boundaries.append(gradient)
    assert offset == len(data)
    return tuple(traces), operands, boundaries


def _run_gate(binary: Path, output: Path, nx: int, ny: int,
              *, nt: int = 11, segments: int = 4):
    result = _run(
        [str(binary), "--native-gradient-output", str(output),
         "--native-gradient-segments", str(segments),
         "--nx", str(nx), "--ny", str(ny), "--fw", "4",
         "--nt", str(nt), "--ntr", "3"],
        cwd=binary.parent,
    )
    if "no CUDA device is visible" in result.stdout:
        pytest.skip("no CUDA device is visible")
    assert result.returncode == 0, result.stdout
    expected_bytes = 8 * nx * ny * 8
    assert f"NATIVE_GRADIENT_PLAN bytes={expected_bytes}" in result.stdout
    assert "allocation=1 zero=1" in result.stdout
    assert "NATIVE_GRADIENT_RESIDENCY per_step_h2d=0 per_step_d2h=0" in result.stdout
    assert "per_step_alloc=0 per_step_free=0 per_step_sync=0" in result.stdout
    assert f"correlation_kernels={2 * nt} steps={nt}" in result.stdout
    assert (
        "NATIVE_GRADIENT_FAILURES download_before=1 prepare_before_sweep=1 "
        "double_prepare=1 late_prepare=1 budget_one_byte=1 invalid_sweep=1"
    ) in result.stdout
    assert (
        f"NATIVE_GRADIENT_PASS nx={nx} ny={ny} nt={nt} ntr=3 "
        f"segments={segments} fields=8 operands_exact=1 "
        f"state_bit_identical=1 reverse_segments={segments}"
    ) in result.stdout
    assert (
        "NATIVE_GRADIENT_LIFECYCLE without_gradient=1 prepared_only=1 "
        "complete=1 failed_prepare=1 contexts_null=1"
    ) in result.stdout
    return result.stdout, _read_gate(output, nx, ny, nt, 3, segments)


def _compare(case: str, actual, expected):
    failures = []
    worst = ("", -1.0, 0.0, 0)
    for field in NATIVE_FIELDS:
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
                failures.append((field, index, observed, reference,
                                 absolute, relative))
    print(
        f"NATIVE_GRADIENT_ORACLE case={case} fields=8 "
        f"pass={int(not failures)} worst_channel={worst[0]} "
        f"max_abs={worst[1]:.17e} max_rel={worst[2]:.17e} "
        f"index={worst[3]}"
    )
    assert not failures, (
        f"{case}: {len(failures)} values exceed rtol={RTOL} atol={ATOL}; "
        f"first={failures[0]}"
    )
    return worst


@pytest.mark.parametrize("name,nx,ny", FIXTURES)
def test_uneven_full_native_gradient(
    gradient_binary: Path, tmp_path: Path, name: str, nx: int, ny: int,
) -> None:
    nt, segments = 11, 4
    output = tmp_path / f"{name}-{gradient_binary.name}-s4.bin"
    _stdout, (traces, operands, actual) = _run_gate(
        gradient_binary, output, nx, ny, nt=nt, segments=segments
    )
    fixture = build_fixture(name, nx, ny, 4)
    expected = native_gradient_boundaries(fixture, traces, operands, nt, segments)
    for boundary, (observed, reference) in enumerate(
        zip(actual, expected, strict=True)
    ):
        _compare(
            f"{name}-{gradient_binary.name}-after-segment-"
            f"{segments - 1 - boundary}", observed, reference
        )
    final = actual[-1]
    assert all(any(value != 0.0 for value in final[field])
               for field in NATIVE_FIELDS)
    assert len({final[field] for field in NATIVE_FIELDS}) == len(NATIVE_FIELDS)
    for left, right in (("gf", "gg"), ("gd", "ge"), ("grx", "gry")):
        assert final[left] != final[right]


def test_one_timestep_segments_freeze_every_temporal_boundary(
    gradient_binary: Path, tmp_path: Path,
) -> None:
    name, nx, ny, nt, segments = "standard", 24, 20, 11, 11
    output = tmp_path / f"temporal-{gradient_binary.name}.bin"
    _stdout, (traces, operands, actual) = _run_gate(
        gradient_binary, output, nx, ny, nt=nt, segments=segments
    )
    fixture = build_fixture(name, nx, ny, 4)
    expected = native_gradient_boundaries(fixture, traces, operands, nt, segments)
    for boundary, (observed, reference) in enumerate(
        zip(actual, expected, strict=True)
    ):
        timestep = nt - boundary
        _compare(f"temporal-{gradient_binary.name}-t{timestep}",
                 observed, reference)
    t1_increment = [
        actual[-1][field][index] - actual[-2][field][index]
        for field in NATIVE_FIELDS
        for index in range(nx * ny)
    ]
    assert any(value != 0.0 for value in t1_increment)


def test_segmentation_invariance_is_bit_exact(
    gradient_binary: Path, tmp_path: Path,
) -> None:
    nx, ny, nt = 24, 20, 11
    finals = {}
    for segments in (4, nt, 1):
        output = tmp_path / f"invariance-{gradient_binary.name}-s{segments}.bin"
        _stdout, (_traces, _operands, gradients) = _run_gate(
            gradient_binary, output, nx, ny, nt=nt, segments=segments
        )
        finals[segments] = b"".join(
            struct.pack(f"<{nx * ny}d", *gradients[-1][field])
            for field in NATIVE_FIELDS
        )
    assert finals[4] == finals[nt] == finals[1]


def test_formula_order_and_field_contract_are_explicit(repository_root: Path) -> None:
    header = (repository_root / "include/denise_cuda_psv_forward.h").read_text()
    source = (repository_root / "src/CUDA/psv_fd4_l1.cu").read_text()
    assert "GF,GG,GFC,GD,GE,GDC,GRX,GRY" in header
    assert "(j-1)*NX+(i-1)" in header
    for expression in (
        "g.gfc[q]+=a.asxy[p]*shear",
        "g.gf[q]+=-2.0*(a.asxx[p]*yy+a.asyy[p]*xx)",
        "g.gg[q]+=(a.asxx[p]+a.asyy[p])*div",
        "g.gdc[q]+=-b*lambda_r*shear",
        "g.gd[q]+=2.0*b*(lambda_p*yy+lambda_q*xx)",
        "g.ge[q]+=-b*(lambda_p+lambda_q)*div",
        "a.ar[p]+dt2*a.asxy[p]",
        "a.ap[p]+dt2*a.asxx[p]",
        "a.aq[p]+dt2*a.asyy[p]",
    ):
        assert expression in source
