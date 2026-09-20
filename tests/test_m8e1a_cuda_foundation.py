"""Local CUDA foundation build, ABI, transfer, and runtime gates."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _find_nvcc() -> Path | None:
    configured = os.environ.get("NVCC")
    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return candidate
    resolved = shutil.which("nvcc")
    if resolved:
        return Path(resolved)
    candidates = [Path("/usr/local/cuda/bin/nvcc")]
    candidates.extend(
        sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True)
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def test_cpu_contract_stays_cuda_independent(repository_root: Path) -> None:
    header = (repository_root / "include/denise_cuda_backend.h").read_text(
        encoding="utf-8"
    )
    makefile = (repository_root / "src/Makefile").read_text(encoding="utf-8")
    assert "cuda_runtime" not in header
    assert "denise:\t\t$(DENISE_OBJ)" in makefile

    make = shutil.which("make")
    if not make:
        pytest.skip("make is unavailable")
    result = _run([make, "-C", "src", "-B", "-n", "denise"], cwd=repository_root)
    assert result.returncode == 0, result.stdout
    assert "nvcc" not in result.stdout.lower()
    assert "cudart" not in result.stdout.lower()


@pytest.fixture(scope="module")
def cuda_foundation_binary(repository_root: Path) -> Path:
    make = shutil.which("make")
    nvcc = _find_nvcc()
    if not make or not nvcc:
        pytest.skip("CUDA toolkit build commands are unavailable")

    result = _run(
        [
            make,
            "-C",
            "src",
            "cuda_foundation",
            f"NVCC={nvcc}",
            "CUDA_ARCHS=86",
        ],
        cwd=repository_root,
    )
    assert result.returncode == 0, result.stdout
    executable = repository_root / "bin/denise_cuda_foundation"
    assert executable.is_file()
    return executable


def test_cuda_foundation_oracle(cuda_foundation_binary: Path) -> None:
    result = _run([str(cuda_foundation_binary)], cwd=cuda_foundation_binary.parent)
    if result.returncode == 77:
        pytest.skip("no CUDA device is visible")
    assert result.returncode == 0, result.stdout
    assert "CUDA_DEVICE logical=0 visible=" in result.stdout
    assert " name=" in result.stdout
    assert " cc=" in result.stdout
    assert "CUDA_BUDGET_CAP cap=268435456 reserve=67108864 budget=" in result.stdout
    assert (
        "CUDA_BUDGET_GATES normal=1 equal=1 greater=1 "
        "cap_below=1 cap_equal=1 zero_alloc=1 "
        "over_alloc=1 in_budget=1"
    ) in result.stdout
    assert (
        "CUDA_TRANSFER_PATTERNS positive_integer=1 negative=1 zero=1 "
        "positive_fraction=1 negative_fraction=1"
    ) in result.stdout
    assert result.stdout.count("CUDA_BUDGET_FAIL_CLOSED") == 4
    assert "requested allocation 1048580 exceeds usable budget 1048576" in result.stdout
    assert (
        "CUDA_FOUNDATION_PASS cases=4 roundtrip=1 halo=1 "
        "sentinel=1 oracle=1 repeat=1"
    ) in result.stdout


def test_cuda_no_device_path_fails_closed(
    cuda_foundation_binary: Path,
) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = _run(
        [str(cuda_foundation_binary), "--expect-no-device"],
        cwd=cuda_foundation_binary.parent,
        env=environment,
    )
    assert result.returncode == 0, result.stdout
    assert "CUDA_NO_DEVICE_FAIL_CLOSED visible=0" in result.stdout
    assert "no CUDA device is visible" in result.stdout


def test_cuda_invalid_device_reports_context(
    cuda_foundation_binary: Path,
) -> None:
    result = _run(
        [str(cuda_foundation_binary), "--select-device", "9999"],
        cwd=cuda_foundation_binary.parent,
    )
    assert result.returncode != 0
    assert "CUDA backend contract failure during select_device" in result.stdout
    assert (
        "outside visible range" in result.stdout
        or "no CUDA device is visible" in result.stdout
    )


def test_cuda_and_cpu_link_dependencies(
    cuda_foundation_binary: Path, repository_root: Path
) -> None:
    ldd = shutil.which("ldd")
    if not ldd:
        pytest.skip("ldd is unavailable")
    cuda_linkage = _run([ldd, str(cuda_foundation_binary)], cwd=repository_root)
    assert cuda_linkage.returncode == 0, cuda_linkage.stdout
    assert "libcudart.so" in cuda_linkage.stdout

    cpu_binary = repository_root / "bin/denise"
    if not cpu_binary.is_file():
        pytest.skip("CPU DENISE executable has not been built")
    cpu_linkage = _run([ldd, str(cpu_binary)], cwd=repository_root)
    assert cpu_linkage.returncode == 0, cpu_linkage.stdout
    assert "libcudart" not in cpu_linkage.stdout
    assert "libcuda.so" not in cpu_linkage.stdout
