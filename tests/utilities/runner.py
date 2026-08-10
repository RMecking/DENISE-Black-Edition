from __future__ import annotations

import json
import hashlib
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class RunResult:
    command: list[str]
    returncode: int
    runtime_seconds: float
    stdout_path: Path
    stderr_path: Path
    metadata_path: Path


def _capture(command: Sequence[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            list(command), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _active_make_variables(makefile: Path) -> dict[str, str]:
    wanted = {"CC", "CFLAGS", "LFLAGS", "SFLAGS", "IFLAGS"}
    values: dict[str, str] = {}
    for raw_line in makefile.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in wanted:
            values[key] = value.strip()
    return values


def executable_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_denise(
    *,
    repository_root: Path,
    case_directory: Path,
    denise_binary: Path,
    mpiexec: str,
    ranks: int,
    configuration: dict[str, Any],
    timeout_seconds: float = 120.0,
) -> RunResult:
    executable_path = denise_binary.resolve(strict=True)
    extra_args = shlex.split(os.environ.get("MPIEXEC_FLAGS", ""))
    command = [mpiexec, *extra_args, "-np", str(ranks), str(executable_path), "denise.inp", "workflow.inp"]
    started = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=case_directory,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = -1
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        stderr += f"\nDENISE test timed out after {timeout_seconds} seconds.\n"
    runtime = time.perf_counter() - started
    stdout_path = case_directory / "stdout.txt"
    stderr_path = case_directory / "stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    make_variables = _active_make_variables(repository_root / "src" / "Makefile")
    git_status = _capture(["git", "status", "--porcelain"], repository_root)
    metadata = {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_seconds": runtime,
        "mpi_ranks": ranks,
        "test_configuration": configuration,
        "executable": {
            "path": str(executable_path),
            "sha256": executable_sha256(executable_path),
        },
        "repository": {
            "git_commit": _capture(["git", "rev-parse", "HEAD"], repository_root),
            "dirty": None if git_status is None else bool(git_status),
            "status_porcelain": git_status or "",
        },
        "local_build_context": {
            "description": (
                "Observed repository Makefile/toolchain context only; not verified provenance "
                "for an externally supplied executable."
            ),
            "makefile": str((repository_root / "src" / "Makefile").resolve()),
            "compiler_command": make_variables.get("CC"),
            "compiler_version": _capture(
                [make_variables.get("CC", "mpicc"), "--version"], repository_root
            ),
            "make_variables": make_variables,
        },
        "mpi_version": _capture([mpiexec, "--version"], repository_root),
    }
    metadata_path = case_directory / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RunResult(command, returncode, runtime, stdout_path, stderr_path, metadata_path)


def result_summary(result: RunResult) -> str:
    stderr = result.stderr_path.read_text(encoding="utf-8", errors="replace")
    stdout = result.stdout_path.read_text(encoding="utf-8", errors="replace")
    return (
        f"command: {' '.join(result.command)}\nreturn code: {result.returncode}\n"
        f"runtime: {result.runtime_seconds:.3f} s\nstdout:\n{stdout[-4000:]}\nstderr:\n{stderr[-4000:]}"
    )
