#!/usr/bin/env python3
"""Build and retain the temporary M5.0f diagnostic executables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


BASE_SHA = "47f3ab93ae7b27433980dafeb77646c9f5a6940a"
BASE_CFLAGS = "-O3 -w -fno-stack-protector -D_FORTIFY_SOURCE=0 -fcommon"
MODES = {
    "legacy_post_xy": (),
    "legacy_b_xy": ("M5_SH_RHO_B_VELOCITY",),
    "exact_metric_legacy_xy": ("M5_SH_EXACT_RECEIVER_METRIC",),
    "exact_b_pre_xy": (
        "M5_SH_EXACT_RECEIVER_METRIC",
        "M5_SH_RHO_B_VELOCITY",
        "M5_SH_GRAD_PRE_STRESS",
    ),
    "exact_b_integrated_pre_xy": (
        "M5_SH_EXACT_RECEIVER_METRIC",
        "M5_SH_RHO_B_INTEGRATED",
        "M5_SH_GRAD_PRE_STRESS",
    ),
    "exact_b_pre_x": (
        "M5_SH_EXACT_RECEIVER_METRIC",
        "M5_SH_RHO_B_VELOCITY",
        "M5_SH_GRAD_PRE_STRESS",
        "M5_SH_GRAD_X_ONLY",
    ),
    "exact_b_pre_y": (
        "M5_SH_EXACT_RECEIVER_METRIC",
        "M5_SH_RHO_B_VELOCITY",
        "M5_SH_GRAD_PRE_STRESS",
        "M5_SH_GRAD_Y_ONLY",
    ),
}


def _run(command: list[str], cwd: Path, env=None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed:\n{result.stdout}")
    return result.stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    worktree = args.worktree.resolve()
    output_directory = args.output_directory.resolve()
    metadata_path = args.metadata.resolve()
    patch_path = repository / "tests" / "m5f_sh_density_instrumentation.patch"

    if _run(["git", "rev-parse", "HEAD"], worktree).strip() != BASE_SHA:
        raise RuntimeError("Instrumented worktree is not at the M5.0f base")
    if _run(["git", "diff", "--name-only"], worktree).splitlines() != [
        "src/SH/sh.c",
        "src/SH/update_v_PML_SH.c",
    ]:
        raise RuntimeError("Unexpected temporary source modifications")

    output_directory.mkdir(parents=True, exist_ok=True)
    _run(["make", "-C", "libcseife"], worktree)
    binaries = {}
    for mode, macros in MODES.items():
        flags = " ".join([BASE_CFLAGS, *(f"-D{name}" for name in macros)])
        environment = os.environ.copy()
        environment["CFLAGS"] = flags
        _run(
            [
                "make", "-e", "-C", "src", "-B", "sh.o",
                "update_v_PML_SH.o",
            ],
            worktree,
            environment,
        )
        _run(["make", "-C", "src", "denise"], worktree)
        destination = output_directory / mode
        shutil.copy2(worktree / "bin" / "denise", destination)
        binaries[mode] = {
            "path": str(destination),
            "sha256": _sha256(destination),
            "compile_flags": flags,
            "macros": list(macros),
        }

    metadata = {
        "base_git_sha": BASE_SHA,
        "temporary_patch_path": str(patch_path),
        "temporary_patch_sha256": _sha256(patch_path),
        "compiler": "mpicc",
        "compiler_version": _run(["mpicc", "--version"], worktree).strip(),
        "binaries": binaries,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
