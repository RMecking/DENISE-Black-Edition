from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("denise")
    group.addoption(
        "--denise-bin",
        default=os.environ.get("DENISE_BIN"),
        help="Path to the DENISE executable (default: DENISE_BIN or bin/denise).",
    )
    group.addoption(
        "--mpiexec",
        default=os.environ.get("MPIEXEC", "mpirun"),
        help="MPI launcher executable (default: MPIEXEC or mpirun).",
    )


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return REPOSITORY_ROOT


@pytest.fixture(scope="session")
def denise_binary(pytestconfig: pytest.Config) -> Path:
    configured = pytestconfig.getoption("--denise-bin")
    candidate = Path(configured).expanduser() if configured else REPOSITORY_ROOT / "bin" / "denise"
    if not candidate.is_absolute():
        candidate = (REPOSITORY_ROOT / candidate).resolve()
    if not candidate.is_file():
        pytest.skip(f"DENISE executable not found: {candidate}. Build it with 'make -C src denise'.")
    return candidate


@pytest.fixture(scope="session")
def mpiexec(pytestconfig: pytest.Config) -> str:
    configured = str(pytestconfig.getoption("--mpiexec"))
    resolved = shutil.which(configured)
    if resolved is None:
        pytest.skip(f"MPI launcher not found: {configured}")
    return resolved
