#!/usr/bin/env python3
"""Rebuild the reconstructed M5.0e instrumentation at the historical base.

This utility never applies the diagnostic patch to the caller's worktree.  It
uses a detached temporary worktree at the recorded base commit and removes that
worktree after recording hashes and semantic patch checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


BASE_GIT_SHA = "47f3ab93ae7b27433980dafeb77646c9f5a6940a"
BASE_CFLAGS = "-O3 -w -fno-stack-protector -D_FORTIFY_SOURCE=0 -fcommon"
MODES = {
    "post_xy": (),
    "post_x": ("M5_SH_GRAD_X_ONLY",),
    "post_y": ("M5_SH_GRAD_Y_ONLY",),
    "pre_xy": ("M5_SH_GRAD_PRE_STRESS",),
    "pre_x": ("M5_SH_GRAD_PRE_STRESS", "M5_SH_GRAD_X_ONLY"),
    "pre_y": ("M5_SH_GRAD_PRE_STRESS", "M5_SH_GRAD_Y_ONLY"),
}


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_mode(worktree: Path, mode: str) -> str:
    macros = MODES[mode]
    cflags = " ".join([BASE_CFLAGS, *(f"-D{name}" for name in macros)])
    environment = os.environ.copy()
    environment["CFLAGS"] = cflags
    _run(["make", "-e", "-C", "src", "-B", "sh.o"], cwd=worktree, env=environment)
    _run(["make", "-C", "src", "denise"], cwd=worktree)
    executable = worktree / "bin" / "denise"
    if not executable.is_file():
        raise RuntimeError(f"Expected executable was not produced: {executable}")
    return _sha256(executable)


def _semantic_patch_checks(worktree: Path) -> dict[str, object]:
    changed_files = _run(
        ["git", "diff", "--name-only", "--", "src", "include", "par"],
        cwd=worktree,
    ).splitlines()
    if changed_files != ["src/SH/sh.c"]:
        raise RuntimeError(f"Unexpected reconstructed patch scope: {changed_files}")

    source = (worktree / "src" / "SH" / "sh.c").read_text(encoding="utf-8")
    required_tokens = (
        "M5_SH_GRAD_X_ONLY",
        "M5_SH_GRAD_Y_ONLY",
        "M5_SH_GRAD_PRE_STRESS",
        "M5_SH_GRAD_CORRELATION",
    )
    missing = [token for token in required_tokens if token not in source]
    if missing:
        raise RuntimeError(f"Missing diagnostic selectors: {missing}")

    exchange_position = source.find("exchange_v_SH")
    pre_position = source.find("#if defined(M5_SH_GRAD_PRE_STRESS)")
    reverse_stress_position = source.find("update_s_elastic_PML_SH", pre_position)
    if min(exchange_position, pre_position, reverse_stress_position) < 0:
        raise RuntimeError(
            "Missing temporal landmarks after patch application: "
            f"exchange={exchange_position}, pre={pre_position}, "
            f"update_s={reverse_stress_position}, "
            f"pre_line={source[:pre_position].count(chr(10)) + 1 if pre_position >= 0 else -1}"
        )
    if not exchange_position < pre_position < reverse_stress_position:
        raise RuntimeError("PRE correlation is not between reverse exchange_v and update_s")
    post_guard_count = source.count("#if !defined(M5_SH_GRAD_PRE_STRESS)")
    if post_guard_count != 2:
        raise RuntimeError(f"Expected two guarded POST correlations, found {post_guard_count}")

    diff = _run(["git", "diff", "--unified=0", "--", "src/SH/sh.c"], cwd=worktree)
    removed_code = [
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]
    expected_removed = (
        "(*fwiSH).waveconv_u_shot[j][i] += "
        "((*fwiSH).forward_prop_syz[imat] * (*waveSH).psyz[j][i]) + "
        "((*fwiSH).forward_prop_sxz[imat] * (*waveSH).psxz[j][i]);"
    )
    if removed_code != [expected_removed, expected_removed]:
        raise RuntimeError(f"Unexpected production code removed by patch: {removed_code}")

    return {
        "changed_files": changed_files,
        "component_selectors_present": True,
        "pre_after_exchange_v": True,
        "pre_before_update_s": True,
        "post_correlations_retained_when_pre_undefined": True,
        "removed_code_limited_to_two_replaced_correlation_statements": True,
        "scope": "observability/correlation timing and component selection only",
    }


def _validate_semantic_results(path: Path) -> dict[str, object]:
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    rows = diagnostics["rows"]
    holdout_rows = diagnostics["holdout_rows"]
    component_identities = diagnostics["component_identity"]
    if len(rows) != 6:
        raise RuntimeError(f"Expected six mandatory rows, found {len(rows)}")
    if diagnostics.get("holdout_run") is not True or len(holdout_rows) != 2:
        raise RuntimeError("Mandatory hold-out evidence is absent")
    if not component_identities:
        raise RuntimeError("Component-identity evidence is absent")
    for row in rows:
        if not row["fd_diagnostics"]["five_point_relative_change"] < 5.0e-5:
            raise RuntimeError(f"Unstable finite difference: {row}")
        if not row["pre_absolute_residual"] < 5.0e-5:
            raise RuntimeError(f"PRE closure failed: {row}")
        if not row["pre_absolute_residual"] < row["post_absolute_residual"]:
            raise RuntimeError(f"PRE did not beat POST: {row}")
        if not row["post_absolute_residual"] > 1.0e-3:
            raise RuntimeError(f"POST negative control failed: {row}")
    for row in holdout_rows:
        if not row["absolute_residual"] < 5.0e-5:
            raise RuntimeError(f"Hold-out closure failed: {row}")
    for identity in component_identities:
        if not identity["relative_l2"] <= 2.0e-6:
            raise RuntimeError(f"Component reconstruction failed: {identity}")
        if not identity["normalized_correlation"] >= 0.999999999:
            raise RuntimeError(f"Component correlation failed: {identity}")
    return {
        "diagnostics_path": str(path),
        "diagnostics_sha256": _sha256(path),
        "mandatory_rows": len(rows),
        "holdout_rows": len(holdout_rows),
        "component_identity_rows": len(component_identities),
        "hard_assertions_satisfied": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("tests/m5e_provenance_reproduction.json")
    )
    parser.add_argument(
        "--binary-output-directory",
        type=Path,
        help="Optional directory in which to retain one rebuilt binary per mode.",
    )
    parser.add_argument(
        "--semantic-results",
        type=Path,
        help="Validated M5.0e.1 diagnostics used to finalize semantic provenance.",
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    output = args.output if args.output.is_absolute() else repository / args.output
    historical_path = repository / "tests" / "m5e_temporal_alignment_diagnostics.json"
    patch_path = repository / "tests" / "m5e_sh_temporal_instrumentation.patch"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    historical_run_artifact_sha = _sha256(historical_path)
    historical_patch_sha = historical["source_patch_sha256"]
    reconstructed_patch_sha = _sha256(patch_path)
    historical_binary_hashes = {
        f"{temporal}_{component}": values["sha256"]
        for temporal, components in historical["binaries"].items()
        for component, values in components.items()
    }
    if args.semantic_results is not None:
        if not output.is_file():
            raise RuntimeError(f"Build provenance does not exist: {output}")
        result = json.loads(output.read_text(encoding="utf-8"))
        result["semantic_reproduction"] = _validate_semantic_results(
            args.semantic_results.resolve()
        )
        result["semantic_reproduction_performed"] = True
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    main_production_diff_before = _run(
        ["git", "diff", "--", "src", "include", "par"], cwd=repository
    )
    if main_production_diff_before:
        raise RuntimeError("Main worktree has production changes before reproduction")

    compiler_version = _run(["mpicc", "--version"], cwd=repository).strip()
    temporary_root = Path(tempfile.mkdtemp(prefix="denise_m5e1_"))
    worktree = temporary_root / "worktree"
    worktree_added = False
    try:
        _run(
            ["git", "worktree", "add", "--detach", str(worktree), BASE_GIT_SHA],
            cwd=repository,
        )
        worktree_added = True
        _run(["git", "apply", "--check", str(patch_path)], cwd=worktree)
        _run(["git", "apply", str(patch_path)], cwd=worktree)
        semantic_checks = _semantic_patch_checks(worktree)
        _run(["make", "-C", "libcseife"], cwd=worktree)

        rebuilt_hashes = {}
        rebuilt_paths = {}
        binary_output_directory = (
            args.binary_output_directory.resolve()
            if args.binary_output_directory is not None
            else None
        )
        if binary_output_directory is not None:
            binary_output_directory.mkdir(parents=True, exist_ok=True)
        for mode in MODES:
            rebuilt_hashes[mode] = _build_mode(worktree, mode)
            if binary_output_directory is not None:
                destination = binary_output_directory / mode
                shutil.copy2(worktree / "bin" / "denise", destination)
                rebuilt_paths[mode] = str(destination)
        repeated_hashes = {
            "post_xy": _build_mode(worktree, "post_xy"),
            "pre_xy": _build_mode(worktree, "pre_xy"),
        }
        repeated_matches = {
            mode: rebuilt_hashes[mode] == repeated_hashes[mode]
            for mode in repeated_hashes
        }
        binary_build_reproducible = all(repeated_matches.values())
        per_mode_hash_match = {
            mode: rebuilt_hashes[mode] == historical_binary_hashes[mode]
            for mode in MODES
        }
        per_mode_verdict = {}
        for mode, exact_match in per_mode_hash_match.items():
            if exact_match:
                verdict = "EXACT HASH MATCH"
            elif binary_build_reproducible:
                verdict = "MISMATCH REQUIRES INVESTIGATION"
            else:
                verdict = "SEMANTIC REPRODUCTION ONLY"
            per_mode_verdict[mode] = verdict

        result = {
            "base_git_sha": BASE_GIT_SHA,
            "historical_run_artifact_sha256": historical_run_artifact_sha,
            "historical_patch_sha256": historical_patch_sha,
            "reconstructed_patch_sha256": reconstructed_patch_sha,
            "historical_patch_byte_match": reconstructed_patch_sha == historical_patch_sha,
            "byte_identical_to_historical_patch": reconstructed_patch_sha == historical_patch_sha,
            "reconstruction_status": (
                "reconstructed after M5.0e; not original run patch"
                if reconstructed_patch_sha != historical_patch_sha
                else "byte-identical reconstruction"
            ),
            "patch_applies_cleanly": True,
            "semantic_patch_checks": semantic_checks,
            "compiler": "mpicc",
            "compiler_version": compiler_version,
            "build_flags": {
                mode: " ".join(
                    [BASE_CFLAGS, *(f"-D{name}" for name in macros)]
                )
                for mode, macros in MODES.items()
            },
            "build_reproducibility": {
                "binary_build_reproducible": binary_build_reproducible,
                "repeated_modes": {
                    mode: {
                        "first_sha256": rebuilt_hashes[mode],
                        "second_sha256": repeated_hashes[mode],
                        "bit_identical": repeated_matches[mode],
                    }
                    for mode in repeated_hashes
                },
            },
            "historical_binary_hashes": historical_binary_hashes,
            "reproduced_binary_hashes": rebuilt_hashes,
            "reproduced_binary_paths": rebuilt_paths,
            "per_mode_hash_match": per_mode_hash_match,
            "per_mode_verdict": per_mode_verdict,
            "semantic_reproduction_performed": False,
            "production_source_restored": False,
        }
    finally:
        if worktree_added:
            _run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repository)
        shutil.rmtree(temporary_root)

    result["production_source_restored"] = not bool(
        _run(["git", "diff", "--", "src", "include", "par"], cwd=repository)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
