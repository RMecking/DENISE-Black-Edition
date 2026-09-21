"""Solver-level gate for the single-GPU FD4/L=1 P/SV forward path."""

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


def _build(repository_root: Path, *, no_fma: bool = False) -> Path:
    make = shutil.which("make")
    nvcc = _find_nvcc()
    if not make or not nvcc:
        pytest.skip("CUDA build toolchain is unavailable")
    command = [make, "-C", "src", "cuda_psv_forward", f"NVCC={nvcc}", "CUDA_ARCHS=86"]
    binary = repository_root / "bin/denise_cuda_psv_forward"
    if no_fma:
        command.extend(
            [
                "CUDA_BUILD_DIR=.cuda-nofma",
                "CUDA_FORWARD_BIN=../bin/denise_cuda_psv_forward_nofma",
                "CUDA_CXXFLAGS=-O2 -std=c++14 --fmad=false",
            ]
        )
        binary = repository_root / "bin/denise_cuda_psv_forward_nofma"
    result = _run(command, cwd=repository_root)
    assert result.returncode == 0, result.stdout
    assert binary.is_file()
    return binary


@pytest.fixture(scope="module")
def forward_binary(repository_root: Path) -> Path:
    return _build(repository_root)


@pytest.fixture(scope="module")
def forward_output(forward_binary: Path) -> str:
    result = _run([str(forward_binary)], cwd=forward_binary.parent)
    if "no CUDA device is visible" in result.stdout:
        pytest.skip("no CUDA device is visible")
    assert result.returncode == 0, result.stdout
    return result.stdout


def test_solver_level_scientific_envelope(forward_output: str) -> None:
    fields = re.findall(
        r"FINAL_PARITY field=(\S+) max_abs=([0-9.eE+-]+) "
        r"max_rel=([0-9.eE+-]+) rms=([0-9.eE+-]+) at=(\d+)",
        forward_output,
    )
    assert len(fields) == 16
    assert {row[0] for row in fields} == {
        "vx", "vy", "sxx", "syy", "sxy", "r1", "p1", "q1",
        "psi_sxx_x", "psi_sxy_x", "psi_vxx", "psi_vyx",
        "psi_syy_y", "psi_sxy_y", "psi_vyy", "psi_vxy",
    }
    # Calibrated after the first production solver-level runs. Across the
    # 64x56x120, 96x80x500, and 112x88x800 cases, the largest observed final
    # absolute and aggregate RMS differences were 6.199e-6 and 2.619e-7.
    envelope = re.search(
        r"FINAL_STATE_ENVELOPE max_abs=([0-9.eE+-]+) "
        r"max_rel=([0-9.eE+-]+) rms=([0-9.eE+-]+)",
        forward_output,
    )
    assert envelope
    assert float(envelope.group(1)) <= 1.0e-5
    assert float(envelope.group(3)) <= 5.0e-7

    traces = re.findall(
        r"TRACE_PARITY component=(vx|vy) max_abs=([0-9.eE+-]+) "
        r"max_rel=([0-9.eE+-]+) rms=([0-9.eE+-]+) receiver=(\d+) "
        r"timestep=(\d+) min_correlation=([0-9.eE+-]+)",
        forward_output,
    )
    assert len(traces) == 2
    assert max(float(row[1]) for row in traces) <= 2.0e-12
    assert max(float(row[3]) for row in traces) <= 5.0e-13
    assert min(float(row[6]) for row in traces) >= 0.99999999


def test_residency_memory_and_failure_contract(forward_output: str) -> None:
    assert (
        "FAILURE_LIFECYCLE budget=1 source_bounds=1 receiver_bounds=1 "
        "unsupported=1 retry=1 destroy=1"
    ) in forward_output
    assert "GPU_REPEAT traces=1 final_state=1" in forward_output
    assert "CPU_SELECTOR_PARITY traces=1 final_state=1" in forward_output
    assert re.search(
        r"MEMORY_PLAN b1=\d+ source_signal=\d+ source_geometry=\d+ "
        r"receiver_geometry=\d+ traces=\d+ workspace=\d+ total=\d+ "
        r"usable=\d+ remaining=\d+",
        forward_output,
    )
    assert re.search(
        r"TRANSFER_CONTRACT h2d_calls=40 d2h_calls=18 h2d_bytes=\d+ "
        r"d2h_bytes=\d+ full_h2d_per_step=0 full_d2h_per_step=0 "
        r"source_h2d_per_step=0 receiver_d2h_per_step=0",
        forward_output,
    )
    assert (
        "SOLVER_FORWARD_PASS nx=96 ny=80 nt=500 ntr=5 source=explosive "
        "seismo=velocity exchange_state_effect=0"
    ) in forward_output


def test_no_fma_complete_forward_is_byte_identical(repository_root: Path) -> None:
    binary = _build(repository_root, no_fma=True)
    result = _run([str(binary)], cwd=binary.parent)
    if "no CUDA device is visible" in result.stdout:
        pytest.skip("no CUDA device is visible")
    assert result.returncode == 0, result.stdout
    assert "NO_FMA_DIAGNOSTIC traces_vx=1 traces_vy=1 final_state=1" in result.stdout
    assert result.stdout.count("max_abs=0.000000000e+00") >= 19


@pytest.mark.parametrize(
    "arguments,marker",
    [
        (["--nx", "64", "--ny", "56", "--nt", "120", "--ntr", "3"],
         "SOLVER_FORWARD_PASS nx=64 ny=56 nt=120 ntr=3"),
        (["--nx", "112", "--ny", "88", "--nt", "800", "--ntr", "7"],
         "SOLVER_FORWARD_PASS nx=112 ny=88 nt=800 ntr=7"),
        (["--nx", "96", "--ny", "80", "--nt", "360", "--ntr", "9"],
         "SOLVER_FORWARD_PASS nx=96 ny=80 nt=360 ntr=9"),
    ],
)
def test_reproducible_benchmark_dimensions(
    forward_binary: Path, arguments: list[str], marker: str
) -> None:
    result = _run([str(forward_binary), *arguments], cwd=forward_binary.parent)
    assert result.returncode == 0, result.stdout
    assert marker in result.stdout
    assert "GPU_REPEAT traces=1 final_state=1" in result.stdout


def test_backend_selector_fails_closed(forward_binary: Path) -> None:
    unknown = _run(
        [str(forward_binary), "--probe-backend", "bogus"],
        cwd=forward_binary.parent,
    )
    assert unknown.returncode != 0
    assert "unknown DENISE_PSV_BACKEND='bogus'" in unknown.stdout

    unsupported = _run(
        [str(forward_binary), "--probe-backend", "unsupported"],
        cwd=forward_binary.parent,
    )
    assert unsupported.returncode != 0
    assert "violates frozen envelope" in unsupported.stdout
    assert "FREE_SURF=1" in unsupported.stdout


def test_invalid_cuda_requests_fail_before_mutation(forward_binary: Path) -> None:
    result = _run(
        [str(forward_binary), "--mutation-oracle"],
        cwd=forward_binary.parent,
    )
    assert result.returncode == 0, result.stdout
    assert (
        "FAIL_BEFORE_MUTATION unknown=1 fdorder=1 L=1 free_surface=1 "
        "topology=1 boundary=1 source_type=1 source_count=1 seismo=1 snap=1 "
        "primary=1 gsls=1 cpml=1 receivers=1"
    ) in result.stdout
    assert "unknown DENISE_PSV_BACKEND='bogus'" in result.stdout
    assert result.stdout.count("violates frozen envelope") == 9


def test_preflight_precedes_psv_zeroing(repository_root: Path) -> None:
    source = (repository_root / "src/PSV/psv.c").read_text(encoding="utf-8")
    preflight = source.index("denise_cuda_psv_backend_preflight")
    viscous_zero = source.index("zero_denise_visc_PSV", preflight)
    elastic_zero = source.index("zero_denise_elast_PSV", preflight)
    dispatch = source.index("denise_cuda_psv_dispatch(", preflight)
    assert preflight < viscous_zero
    assert preflight < elastic_zero
    assert viscous_zero < dispatch
    assert elastic_zero < dispatch


def test_hidden_gpu_fails_closed(forward_binary: Path) -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = _run([str(forward_binary)], cwd=forward_binary.parent, env=environment)
    assert result.returncode != 0
    assert "no CUDA device is visible" in result.stdout

    mutation = _run(
        [str(forward_binary), "--mutation-oracle-no-device"],
        cwd=forward_binary.parent,
        env=environment,
    )
    assert mutation.returncode == 0, mutation.stdout
    assert (
        "NO_DEVICE_FAIL_BEFORE_MUTATION failure=1 primary=1 gsls=1 "
        "cpml=1 receivers=1"
    ) in mutation.stdout


def test_optional_solver_and_cpu_link_contract(
    forward_binary: Path, repository_root: Path
) -> None:
    make = shutil.which("make")
    nvcc = _find_nvcc()
    assert make and nvcc
    build = _run(
        [make, "-C", "src", "denise_cuda", f"NVCC={nvcc}", "CUDA_ARCHS=86"],
        cwd=repository_root,
    )
    assert build.returncode == 0, build.stdout
    cuda_solver = repository_root / "bin/denise_cuda"
    assert cuda_solver.is_file()

    dry_run = _run([make, "-C", "src", "-B", "-n", "denise"], cwd=repository_root)
    assert dry_run.returncode == 0, dry_run.stdout
    assert "nvcc" not in dry_run.stdout.lower()
    assert "cudart" not in dry_run.stdout.lower()

    ldd = shutil.which("ldd")
    if not ldd:
        return
    cuda_linkage = _run([ldd, str(cuda_solver)], cwd=repository_root)
    assert cuda_linkage.returncode == 0, cuda_linkage.stdout
    assert "libcudart.so" in cuda_linkage.stdout
    cpu_solver = repository_root / "bin/denise"
    if cpu_solver.is_file():
        cpu_linkage = _run([ldd, str(cpu_solver)], cwd=repository_root)
        assert cpu_linkage.returncode == 0, cpu_linkage.stdout
        assert "libcudart" not in cpu_linkage.stdout
        assert "libcuda.so" not in cpu_linkage.stdout


def test_single_rank_exchange_omission_is_state_equivalent(
    forward_output: str, repository_root: Path
) -> None:
    # The CPU oracle run includes exchange_v_PSV/exchange_s_PSV. For POS=(0,0),
    # NPROCX=NPROCY=1, and BOUNDARY=0 every pack/unpack mutation guard is false.
    # The no-FMA solver comparison independently proves byte equality against the
    # CUDA path, which omits these buffer-only MPI_Sendrecv_replace calls.
    for name in ("exchange_v_PSV.c", "exchange_s_PSV.c"):
        source = (repository_root / "src/PSV" / name).read_text(encoding="utf-8")
        assert "POS[2]!=0" in source
        assert "POS[2]!=NPROCY-1" in source
        assert "(BOUNDARY) || (POS[1]!=0)" in source
        assert "(BOUNDARY) || (POS[1]!=NPROCX-1)" in source
    assert "exchange_state_effect=0" in forward_output
