"""Local scientific gate for the device-resident FD4/L=1 CUDA P/SV core."""

from __future__ import annotations

import os
from pathlib import Path
import re
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
    if configured and Path(configured).is_file():
        return Path(configured)
    resolved = shutil.which("nvcc")
    if resolved:
        return Path(resolved)
    candidates = [Path("/usr/local/cuda/bin/nvcc")]
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True))
    return next((path for path in candidates if path.is_file()), None)


@pytest.fixture(scope="module")
def psv_binary(repository_root: Path) -> Path:
    make = shutil.which("make")
    nvcc = _find_nvcc()
    if not make or not nvcc:
        pytest.skip("CUDA build toolchain is unavailable")
    result = _run(
        [make, "-C", "src", "cuda_psv_fd4_l1", f"NVCC={nvcc}", "CUDA_ARCHS=86"],
        cwd=repository_root,
    )
    assert result.returncode == 0, result.stdout
    binary = repository_root / "bin/denise_cuda_psv_fd4_l1"
    assert binary.is_file()
    return binary


@pytest.fixture(scope="module")
def psv_output(psv_binary: Path) -> str:
    result = _run([str(psv_binary)], cwd=psv_binary.parent)
    if "no CUDA device is visible" in result.stdout:
        pytest.skip("no CUDA device is visible")
    assert result.returncode == 0, result.stdout
    return result.stdout


def test_b1_scientific_contract(psv_output: str) -> None:
    assert "DEVICE_STATE full_j=-2:83 full_i=-2:99" in psv_output
    assert "AGGREGATE_BUDGET fail_closed=1 retry=1 unsupported_config=1" in psv_output
    assert "INITIAL_STATE_BYTE_IDENTICAL fields=36 result=1" in psv_output
    assert "EXACT_ORACLES upload_download=1 untouched_halo=1 mapping=1" in psv_output
    assert "GPU_REPEAT_BITWISE_IDENTICAL steps=100 result=1" in psv_output
    assert "production_prepare_update_s_visc_PSV=1" in psv_output
    assert "CUDA_PSV_FD4_L1_PASS steps=1,10,100 fields=16" in psv_output


def test_calibrated_cpu_gpu_envelope(psv_output: str) -> None:
    rows = re.findall(
        r"PARITY steps=(\d+) field=(\S+) max_abs=([0-9.eE+-]+) "
        r"max_rel=([0-9.eE+-]+) rms=([0-9.eE+-]+) max_ulp=(\d+)",
        psv_output,
    )
    assert len(rows) == 48
    assert {int(row[0]) for row in rows} == {1, 10, 100}
    # Calibrated on the first production build before this gate was encoded:
    # observed maxima were 2.686e-3 absolute, 4.941e-3 relative, 3.103e-4 RMS.
    assert max(float(row[2]) for row in rows) <= 5.0e-3
    assert max(float(row[3]) for row in rows) <= 1.0e-2
    assert max(float(row[4]) for row in rows) <= 1.0e-3


def test_residency_and_optional_build(psv_output: str, repository_root: Path) -> None:
    assert re.search(
        r"TRANSFER_COUNTS initial_h2d=36 checkpoint_d2h=144 per_timestep=0",
        psv_output,
    )
    make = shutil.which("make")
    assert make
    dry_run = _run([make, "-C", "src", "-B", "-n", "denise"], cwd=repository_root)
    assert dry_run.returncode == 0, dry_run.stdout
    assert "nvcc" not in dry_run.stdout.lower()
    assert "cudart" not in dry_run.stdout.lower()
    makefile = (repository_root / "src/Makefile").read_text(encoding="utf-8")
    cuda_source = (repository_root / "src/CUDA/psv_fd4_l1.cu").read_text(encoding="utf-8")
    assert "--use_fast_math" not in makefile
    assert "--use_fast_math" not in cuda_source


def test_b1_hidden_gpu_fails_closed(psv_binary: Path) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = _run([str(psv_binary)], cwd=psv_binary.parent, env=environment)
    assert result.returncode != 0
    assert "no CUDA device is visible" in result.stdout


def test_b1_link_dependencies(psv_binary: Path, repository_root: Path) -> None:
    ldd = shutil.which("ldd")
    if not ldd:
        pytest.skip("ldd is unavailable")
    gpu = _run([ldd, str(psv_binary)], cwd=repository_root)
    assert gpu.returncode == 0, gpu.stdout
    assert "libcudart.so" in gpu.stdout
    cpu = repository_root / "bin/denise"
    if cpu.is_file():
        linkage = _run([ldd, str(cpu)], cwd=repository_root)
        assert linkage.returncode == 0, linkage.stdout
        assert "libcudart" not in linkage.stdout
        assert "libcuda.so" not in linkage.stdout
