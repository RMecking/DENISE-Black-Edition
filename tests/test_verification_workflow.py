from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest as denise_contract


def _git_attributes(
    repository_root: Path, path: str, *attributes: str
) -> dict[str, str]:
    result = subprocess.run(
        ["git", "check-attr", *attributes, "--", path],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        reported_path, attribute, value = line.split(": ", 2)
        assert reported_path == path
        parsed[attribute] = value
    return parsed


def test_developer_verification_contract(repository_root: Path) -> None:
    guide = (repository_root / "docs/testing.md").read_text(encoding="utf-8")
    readme = (repository_root / "tests/README.md").read_text(encoding="utf-8")
    runner = (repository_root / "scripts/run_verification.sh").read_text(
        encoding="utf-8"
    )
    runner_bytes = (repository_root / "scripts/run_verification.sh").read_bytes()

    for level in ("QUICK", "MANDATORY", "EXTENDED", "TARGETED"):
        assert level in guide
    for status in (
        "VERIFIED",
        "PARTIALLY VERIFIED",
        "FORWARD VERIFIED ONLY",
        "UNVERIFIED",
        "QUARANTINED",
        "NOT COVERED",
    ):
        assert status in guide
    assert "--require-denise" in guide
    assert "docs/testing.md" in readme
    assert "docs/verification.md" in readme

    assert "set -euo pipefail" in runner
    assert "MPIEXEC_FLAGS" in runner
    assert "--oversubscribe" in runner
    assert "make -C libcseife" in runner
    assert "make -C src denise" in runner
    assert "-m 'not integration'" in runner
    assert "-m 'not extended'" in runner
    assert "-m extended" in runner
    assert "--require-denise" in runner
    assert runner_bytes.startswith(b"#!/usr/bin/env bash\n")
    assert not runner_bytes.startswith(b"#!/usr/bin/env bash\r\n")

    assert _git_attributes(
        repository_root, "scripts/run_verification.sh", "text", "eol"
    ) == {"text": "set", "eol": "lf"}
    for retained_path in (
        "tests/m5_provenance_contract.patch",
        "tests/m5_provenance_contract.json",
    ):
        assert _git_attributes(repository_root, retained_path, "text") == {
            "text": "unset"
        }


def test_optional_prerequisite_verification_contract(repository_root: Path) -> None:
    configuration = (repository_root / "pytest.ini").read_text(encoding="utf-8")
    fixtures = (repository_root / "tests/conftest.py").read_text(encoding="utf-8")
    workflow = (repository_root / ".github/workflows/verification.yml").read_text(
        encoding="utf-8"
    )
    assert "optional_prerequisite:" in configuration
    assert 'item.get_closest_marker("integration") is not None' in fixtures
    assert 'item.get_closest_marker("optional_prerequisite") is None' in fixtures
    assert 'item.config.getoption("--require-denise")' in fixtures
    assert "python3 -m pytest tests/physics -m 'not extended' -v --require-denise" in workflow


@pytest.mark.parametrize(
    "required,integration,optional,outcome,xfail,expected",
    (
        (True, True, False, "skipped", False, "failed"),
        (True, True, True, "skipped", False, "skipped"),
        (True, False, False, "skipped", False, "skipped"),
        (False, True, False, "skipped", False, "skipped"),
        (True, True, False, "skipped", True, "skipped"),
        (True, True, True, "failed", False, "failed"),
        (True, True, True, "passed", False, "passed"),
    ),
)
def test_require_denise_preserves_optional_skip_and_failure_semantics(
    required, integration, optional, outcome, xfail, expected,
) -> None:
    markers = {
        name for name, enabled in (
            ("integration", integration), ("optional_prerequisite", optional)
        ) if enabled
    }
    item = SimpleNamespace(
        config=SimpleNamespace(getoption=lambda name: required),
        get_closest_marker=lambda name: object() if name in markers else None,
    )
    report = SimpleNamespace(
        outcome=outcome, skipped=outcome == "skipped", longrepr="original result",
    )
    if xfail:
        report.wasxfail = "existing expected failure"
    hook = denise_contract.pytest_runtest_makereport(item, None)
    next(hook)
    with pytest.raises(StopIteration):
        hook.send(SimpleNamespace(get_result=lambda: report))
    assert report.outcome == expected
    if expected == outcome:
        assert report.longrepr == "original result"


@pytest.mark.parametrize("prerequisite", ("denise", "mpi"))
def test_require_denise_core_prerequisites_stay_hard_failures(
    tmp_path: Path, prerequisite: str,
) -> None:
    missing = str(tmp_path / "definitely-missing-core-prerequisite")
    options = {"--require-denise": True, "--denise-bin": missing, "--mpiexec": missing}
    config = SimpleNamespace(getoption=lambda name: options[name])
    fixture = (
        denise_contract.denise_binary if prerequisite == "denise"
        else denise_contract.mpiexec
    )
    message = "DENISE executable not found" if prerequisite == "denise" else "MPI launcher not found"
    with pytest.raises(pytest.fail.Exception, match=message):
        fixture.__wrapped__(config)
