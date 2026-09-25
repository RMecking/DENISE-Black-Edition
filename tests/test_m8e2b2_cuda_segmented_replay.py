"""M8c-aligned bounded CUDA operand replay, with bitwise same-GPU gates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_m8e1b2a_cuda_psv_forward import _build, _run


@pytest.fixture(scope="module", params=[False, True], ids=["normal", "no-fma"])
def segmented_binary(repository_root: Path, request: pytest.FixtureRequest) -> Path:
    return _build(repository_root, no_fma=request.param)


@pytest.mark.parametrize("nx,ny", [(96, 80), (97, 83)], ids=["standard", "edge"])
def test_m8c_segmented_replay_identity(
    segmented_binary: Path, nx: int, ny: int
) -> None:
    result = _run(
        [str(segmented_binary), "--segment-gate", "--nx", str(nx),
         "--ny", str(ny), "--nt", "500", "--ntr", "5"],
        cwd=segmented_binary.parent,
    )
    assert result.returncode == 0, result.stdout
    assert "SEGMENT_BUDGET_REJECT" in result.stdout
    assert "SEGMENT_FORWARD_IDENTITY fields=1 traces=1" in result.stdout
    plan = re.search(
        r"SEGMENT_PLAN segments=(\d+) interior_checkpoints=(\d+) seed=(\d+) "
        r"mandatory=(\d+) bank=(\d+) operands=(\d+) source_receiver=(\d+) "
        r"remaining=(\d+)", result.stdout,
    )
    assert plan, result.stdout
    assert tuple(map(int, plan.groups()[:3])) == (32, 31, 1)
    checkpoint = 4 * (8 * (nx + 6) * (ny + 6) + 4 * ny * 16 + 4 * nx * 16)
    assert int(plan.group(5)) == 32 * checkpoint
    assert int(plan.group(6)) == 4 * 6 * 16 * nx * ny
    replay = re.findall(
        r"SEGMENT_REPLAY index=(\d+) begin=(\d+) end=(\d+) length=(\d+) "
        r"repeat=(\d+) fx=1 fy=1 vxx=1 vyx=1 vxy=1 vyy=1 fields=1 traces=1",
        result.stdout,
    )
    assert len(replay) == 5, result.stdout
    assert {int(row[0]) for row in replay} == {0, 1, 16, 31}
    assert {int(row[3]) for row in replay} == {15, 16}
    assert "SEGMENT_GATE_PASS" in result.stdout
