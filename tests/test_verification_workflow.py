from __future__ import annotations

from pathlib import Path


def test_developer_verification_contract(repository_root: Path) -> None:
    guide = (repository_root / "docs/testing.md").read_text(encoding="utf-8")
    readme = (repository_root / "tests/README.md").read_text(encoding="utf-8")
    runner = (repository_root / "scripts/run_verification.sh").read_text(
        encoding="utf-8"
    )
    attributes = (repository_root / ".gitattributes").read_text(encoding="utf-8")

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
    assert "tests/m5*.patch -text" in attributes
    assert "tests/m5*.json  -text" in attributes
