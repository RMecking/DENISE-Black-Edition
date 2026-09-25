"""Exact same-GPU range and device-checkpoint replay invariants."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_m8e1b2a_cuda_psv_forward import _build, _run


@pytest.fixture(scope="module", params=[False, True], ids=["normal", "no-fma"])
def checkpoint_binary(repository_root: Path, request: pytest.FixtureRequest) -> Path:
    return _build(repository_root, no_fma=request.param)


@pytest.mark.parametrize("nx,ny", [(96, 80), (97, 83)], ids=["standard", "edge"])
def test_device_checkpoint_exact_replay(
    checkpoint_binary: Path, nx: int, ny: int
) -> None:
    result = _run(
        [
            str(checkpoint_binary), "--checkpoint-gate", "--nx", str(nx),
            "--ny", str(ny), "--nt", "500", "--ntr", "5",
        ],
        cwd=checkpoint_binary.parent,
    )
    assert result.returncode == 0, result.stdout
    assert "CHECKPOINT_CPU_DEFAULT_MODE1=1 CUDA_MODE1_REJECTED=1" in result.stdout
    assert re.search(
        rf"CHECKPOINT_SPLIT nx={nx} ny={ny} state=1 traces_vx=1 traces_vy=1 ",
        result.stdout,
    )
    expected_bytes = 4 * (8 * (nx + 6) * (ny + 6) + 4 * ny * 16 + 4 * nx * 16)
    for boundary in (17, 173, 421):
        assert re.search(
            rf"CHECKPOINT_REPLAY nx={nx} ny={ny} boundary={boundary} "
            rf"next={boundary + 1} fields=1 traces=1 d2d_calls=32 "
            rf"d2d_bytes={2 * expected_bytes} checkpoint_bytes={expected_bytes}\b",
            result.stdout,
        )
    assert (
        f"CHECKPOINT_GATE_PASS nx={nx} ny={ny} nt=500 nrec=5 "
        "boundaries=17,173,421 repeat=1"
    ) in result.stdout
