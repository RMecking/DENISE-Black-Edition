from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


BASE_GIT_SHA = "47f3ab93ae7b27433980dafeb77646c9f5a6940a"
HISTORICAL_PATCH_SHA256 = (
    "2fe1813671502191e316df36de3a325deabb4f1a7dfe32931397376f212e67e2"
)
EXPECTED_MODES = {"post_xy", "post_x", "post_y", "pre_xy", "pre_x", "pre_y"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m5e_reconstructed_patch_and_provenance(
    repository_root: Path, tmp_path: Path
):
    historical_path = repository_root / "tests" / "m5e_temporal_alignment_diagnostics.json"
    patch_path = repository_root / "tests" / "m5e_sh_temporal_instrumentation.patch"
    reproduction_path = repository_root / "tests" / "m5e_provenance_reproduction.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))

    reconstructed_hash = _sha256(patch_path)
    assert historical["git_base_sha"] == BASE_GIT_SHA
    assert historical["source_patch_sha256"] == HISTORICAL_PATCH_SHA256
    assert reproduction["base_git_sha"] == BASE_GIT_SHA
    assert reproduction["historical_run_artifact_sha256"] == _sha256(historical_path)
    assert reproduction["historical_patch_sha256"] == HISTORICAL_PATCH_SHA256
    assert reproduction["reconstructed_patch_sha256"] == reconstructed_hash
    assert reproduction["historical_patch_byte_match"] is (
        reconstructed_hash == HISTORICAL_PATCH_SHA256
    )
    assert reproduction["byte_identical_to_historical_patch"] is (
        reconstructed_hash == HISTORICAL_PATCH_SHA256
    )
    if reconstructed_hash != HISTORICAL_PATCH_SHA256:
        assert reproduction["reconstruction_status"] == (
            "reconstructed after M5.0e; not original run patch"
        )

    patch_text = patch_path.read_text(encoding="utf-8")
    assert "diff --git a/src/SH/sh.c b/src/SH/sh.c" in patch_text
    assert "diff --git a/src/SH/ass_gradSH.c" not in patch_text
    for selector in (
        "M5_SH_GRAD_X_ONLY",
        "M5_SH_GRAD_Y_ONLY",
        "M5_SH_GRAD_PRE_STRESS",
    ):
        assert selector in patch_text

    historical_source = subprocess.run(
        ["git", "show", f"{BASE_GIT_SHA}:src/SH/sh.c"],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    historical_tree = tmp_path / "historical_base"
    (historical_tree / "src" / "SH").mkdir(parents=True)
    (historical_tree / "src" / "SH" / "sh.c").write_bytes(historical_source)
    apply_check = subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=historical_tree,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert apply_check.returncode == 0, apply_check.stdout
    assert reproduction["patch_applies_cleanly"] is True
    checks = reproduction["semantic_patch_checks"]
    assert checks["changed_files"] == ["src/SH/sh.c"]
    assert checks["component_selectors_present"] is True
    assert checks["pre_after_exchange_v"] is True
    assert checks["pre_before_update_s"] is True
    assert checks["post_correlations_retained_when_pre_undefined"] is True
    assert checks["removed_code_limited_to_two_replaced_correlation_statements"] is True

    assert set(reproduction["historical_binary_hashes"]) == EXPECTED_MODES
    assert set(reproduction["reproduced_binary_hashes"]) == EXPECTED_MODES
    assert set(reproduction["per_mode_hash_match"]) == EXPECTED_MODES
    assert (
        reproduction["reproduced_binary_hashes"]
        == reproduction["historical_binary_hashes"]
    )
    assert all(reproduction["per_mode_hash_match"].values())
    assert all(
        verdict == "EXACT HASH MATCH"
        for verdict in reproduction["per_mode_verdict"].values()
    )
    assert (
        reproduction["build_reproducibility"]["binary_build_reproducible"]
        is True
    )
    assert reproduction["semantic_reproduction_performed"] is True
    assert (
        reproduction["semantic_reproduction"]["hard_assertions_satisfied"]
        is True
    )
    assert reproduction["production_source_restored"] is True
