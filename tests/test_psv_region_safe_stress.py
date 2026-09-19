"""Byte-exact region contract for the P/SV stress kernels."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_full_and_split_stress_updates_are_byte_identical(
    repository_root: Path, tmp_path: Path
) -> None:
    compiler = shutil.which("mpicc")
    if compiler is None:
        pytest.skip("mpicc is required for the P/SV region harness")
    executable = tmp_path / "psv_region_safe_stress"
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c99",
            "-O2",
            "-fcommon",
            "-DDENISE_REGION_TEST_HOOKS",
            "-I",
            str(repository_root / "include"),
            str(repository_root / "tests/utilities/psv_region_safe_stress_harness.c"),
            str(repository_root / "src/PSV/update_s_elastic_PML_PSV.c"),
            str(repository_root / "src/PSV/update_s_visc_PML_PSV.c"),
            str(repository_root / "src/util.c"),
            "-lm",
            "-o",
            str(executable),
        ],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    result = subprocess.run(
        [str(executable)], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALL_REGION_ORACLES_PASS" in result.stdout
    assert "FAST_PATH full=182 split=182" in result.stdout
    assert "EXACT_RECORDS fields=6 strain_hits=once" in result.stdout
