from __future__ import annotations

import subprocess
from pathlib import Path


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
