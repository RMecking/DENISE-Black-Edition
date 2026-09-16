"""Direct deterministic C8c exact optimizer-boundary contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def _read(repository_root: Path, relative_path: str) -> str:
    return (repository_root / relative_path).read_text(encoding="utf-8")


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_optimizer_boundary_api_and_static_contract(repository_root: Path) -> None:
    header = _read(repository_root, "include/fd.h")
    makefile = _read(repository_root, "src/Makefile")
    source = _compact(
        _read(repository_root, "src/SH/visco_sh_exact_optimizer_boundary.c")
    )

    assert "struct visco_sh_exact_optimizer_boundary" in header
    assert "visco_sh_exact_build_steepest_subtractive_step(" in header
    assert "visco_sh_exact_optimizer_boundary.c" in makefile
    for required in (
        "boundary->optimizer_step_primary[j][i]=boundary->grad_raw_primary[j][i]",
        "boundary->optimizer_step_rho[j][i]=boundary->grad_raw_rho[j][i]",
        "boundary->optimizer_step_q[j][i]=boundary->grad_raw_q[j][i]",
        "if(boundary==NULL)return-1",
        "boundary->nx<1",
        "boundary->ny<1",
    ):
        assert required in source
    for forbidden in (
        "ass_gradSH_visc(",
        "descent(",
        "PCG(",
        "LBFGS(",
        "waveconv_ts",
        "gradg_ts",
        "gradp_ts",
    ):
        assert forbidden not in source


@pytest.fixture(scope="module")
def optimizer_boundary_harness(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> Path:
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the C8c optimizer-boundary harness"
    executable = tmp_path_factory.mktemp("m63c_c8c_boundary") / "boundary"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fcommon",
        "-I",
        str(repository_root / "include"),
        str(repository_root / "tests/utilities/m63c_c8c_optimizer_boundary_harness.c"),
        str(repository_root / "src/SH/visco_sh_exact_optimizer_boundary.c"),
        "-o",
        str(executable),
        "-lm",
    ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    return executable


def test_real_optimizer_boundary_helper_contract(
    optimizer_boundary_harness: Path,
) -> None:
    result = subprocess.run(
        [str(optimizer_boundary_harness)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    assert len(records) == 1, result.stdout
    record = records[0]
    assert set(record) == {
        "successful_copy",
        "raw_immutable",
        "halos_untouched",
        "physical_q_identity",
        "dot_product",
        "invalid_cases",
    }
    assert record["successful_copy"] is True
    assert record["raw_immutable"] is True
    assert record["halos_untouched"] is True
    assert record["physical_q_identity"] is True
    assert record["invalid_cases"] == 9
    assert record["dot_product"] > 0.0
