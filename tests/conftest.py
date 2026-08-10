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
    group.addoption(
        "--require-denise",
        action="store_true",
        help="Fail instead of skipping when DENISE or MPI is unavailable; intended for CI/verification.",
    )


def unavailable_dependency(message: str, *, required: bool) -> None:
    if required:
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


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
        unavailable_dependency(
            f"DENISE executable not found: {candidate}. Build it with 'make -C src denise'.",
            required=pytestconfig.getoption("--require-denise"),
        )
    return candidate.resolve()


@pytest.fixture(scope="session")
def mpiexec(pytestconfig: pytest.Config) -> str:
    configured = str(pytestconfig.getoption("--mpiexec"))
    resolved = shutil.which(configured)
    if resolved is None:
        unavailable_dependency(
            f"MPI launcher not found: {configured}",
            required=pytestconfig.getoption("--require-denise"),
        )
    return resolved


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if (
        item.config.getoption("--require-denise")
        and item.get_closest_marker("integration") is not None
        and report.skipped
    ):
        report.outcome = "failed"
        report.longrepr = "DENISE integration tests may not skip in --require-denise mode"
