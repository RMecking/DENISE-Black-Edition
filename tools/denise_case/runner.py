from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import CaseConfig
from .generate import materialize_run
from .model_io import sha256_file
from .validation import validate_case, write_validation


def _output_inventory(run: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in sorted((run / "data").glob("*")):
        if path.is_file():
            result.append({"path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return result


def run_forward(
    config: CaseConfig, *, run_name: str = "true_forward", timeout_seconds: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    run = materialize_run(config, run_name)
    if (run / "run_metadata.json").exists():
        raise FileExistsError("refusing to rerun over existing execution metadata/data")
    validation = validate_case(config, require_executable=not dry_run)
    write_validation(validation, run / "validation.json")
    if not validation.ok:
        raise RuntimeError("forward validation failed: " + "; ".join(validation.errors))
    executable = config.resolve("backend.executable")
    launcher = str(config.require("backend.launcher"))
    launcher_args = shlex.split(os.environ.get("MPIEXEC_FLAGS", ""))
    command = [launcher, *launcher_args, "-np", str(config.require("backend.ranks")), str(executable), "denise.inp", "workflow.inp"]
    metadata: dict[str, Any] = {
        "command": command, "cwd": str(run), "dry_run": dry_run,
        "backend": config.require("backend.environment"), "platform": platform.platform(),
        "denise_core_sha": config.require("provenance.denise_core_sha"),
        "benchmark_source_sha": config.require("provenance.benchmark_sha"),
        "mpi_ranks": int(config.require("backend.ranks")), "launcher_args": launcher_args,
        "binary_sha256": sha256_file(executable) if executable.is_file() else None,
        "manifest_sha256": config.sha256,
    }
    if dry_run:
        metadata.update(returncode=None, runtime_seconds=0.0, output_inventory=[])
    else:
        metadata["started_utc"] = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        timed_out = False
        try:
            completed = subprocess.run(
                command, cwd=run, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout_seconds, check=False, env=os.environ.copy(),
            )
            returncode, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired as error:
            timed_out = True
            returncode = -1
            stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
            stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
            stderr += f"\nDENISE timed out after {timeout_seconds} seconds.\n"
        runtime = time.perf_counter() - started
        metadata["ended_utc"] = datetime.now(timezone.utc).isoformat()
        (run / "stdout.txt").write_text(stdout, encoding="utf-8")
        (run / "stderr.txt").write_text(stderr, encoding="utf-8")
        metadata.update(
            returncode=returncode, timed_out=timed_out, runtime_seconds=runtime,
            output_inventory=_output_inventory(run),
        )
    path = run / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
