"""Independent real-MPI oracle for C8c exact SH material preparation."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def _read(root: Path, name: str) -> str:
    return (root / name).read_text(encoding="utf-8")


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_material_communicators_are_world_compatible(repository_root: Path) -> None:
    helper = _compact(_read(repository_root, "src/SH/visco_sh_exact_material_preparation.c"))
    matcopy = _compact(_read(repository_root, "src/SH/matcopy_SH.c"))
    assert "MPI_Allreduce(&invalid,&any_invalid,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD)" in helper
    assert "MPI_Bsend(" in matcopy and "MPI_Recv(" in matcopy
    assert "MPI_COMM_WORLD" in matcopy


@pytest.fixture(scope="module")
def material_preparation_binaries(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> tuple[Path, Path, Path]:
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the B4A.1 runtime oracle"
    directory = tmp_path_factory.mktemp("m63c_c8c_material_preparation")
    harness = repository_root / "tests/utilities/m63c_c8c_exact_material_preparation_harness.c"
    common = [
        "-std=c99", "-O1", "-g", "-fcommon", "-I", str(repository_root / "include"),
        str(harness), str(repository_root / "src/SH/visco_sh_exact_material_preparation.c"),
        str(repository_root / "src/SH/visco_sh_exact_trial_state.c"),
        str(repository_root / "src/q_parameterization.c"), str(repository_root / "src/SH/av_mu_SH.c"),
        str(repository_root / "src/SH/inv_rho_SH.c"), str(repository_root / "src/av_tau.c"),
        str(repository_root / "src/SH/prepare_update_s_visc_SH.c"), str(repository_root / "src/util.c"),
        "-lm",
    ]

    def build(name: str, extra: list[str], matcopy: bool) -> Path:
        executable = directory / name
        command = [compiler, *extra, *common]
        if matcopy:
            command.insert(-1, str(repository_root / "src/SH/matcopy_SH.c"))
        command += ["-o", str(executable)]
        result = subprocess.run(command, cwd=repository_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 0, result.stdout
        return executable

    failure = build("failure", ["-DINSTRUMENT_MATCOPY"], False)
    success = build("success", [], True)
    asan = build("asan", ["-fsanitize=address", "-fno-omit-frame-pointer"], True)
    return failure, success, asan


def _run(executable: Path, mode: str, ranks: int, root: Path, *, asan: bool = False) -> dict[str, object]:
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert launcher, "MPI launcher is required for the B4A.1 runtime oracle"
    environment = None
    if asan:
        environment = {**__import__("os").environ, "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1"}
    result = subprocess.run([launcher, "-n", str(ranks), str(executable), mode], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, timeout=90)
    assert result.returncode == 0, result.stdout
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert rows == [{"mode": mode, "ok": True}], result.stdout
    return rows[0]


def test_real_mpi_failure_success_and_state_oracles(
    material_preparation_binaries: tuple[Path, Path, Path], repository_root: Path
) -> None:
    failure, success, asan = material_preparation_binaries
    _run(failure, "failure-rank0", 2, repository_root)
    _run(failure, "failure-rank1", 2, repository_root)
    _run(success, "success", 2, repository_root)
    _run(asan, "success", 1, repository_root, asan=True)
