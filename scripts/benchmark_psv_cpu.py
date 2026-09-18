#!/usr/bin/env python3
"""Generate and time the representative 200x200 P/SV CPU cases."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from array import array
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case


def _write_constant(path: Path, value: float, count: int) -> None:
    with path.open("wb") as stream:
        (array("f", [value]) * count).tofile(stream)


def _make_case(root: Path, name: str, *, viscoelastic: bool) -> dict[str, object]:
    directory = root / name
    if directory.exists():
        raise FileExistsError(f"Benchmark case already exists: {directory}")
    config = generate_case(directory, config=HomogeneousPSVConfig())
    if viscoelastic:
        parameter_file = directory / "denise.inp"
        text = parameter_file.read_text(encoding="ascii")
        old = "\nL =0\n"
        assert text.count(old) == 1
        # read_par consumes the first character before scanning a record; the
        # one-character L key therefore needs the historical sentinel space.
        parameter_file.write_text(text.replace(old, "\n L =1\n"), encoding="ascii")
        count = config.nx * config.ny
        prefix = directory / "model" / "homogeneous"
        _write_constant(prefix.with_suffix(".qp"), 60.0, count)
        _write_constant(prefix.with_suffix(".qs"), 40.0, count)
    return {
        "directory": str(directory),
        "grid": [config.nx, config.ny],
        "nt": config.samples_per_trace,
        "ranks": 1,
        "L": 1 if viscoelastic else 0,
    }


def _time_case(binary: Path, mpiexec: str, case: dict[str, object], repetitions: int) -> dict[str, object]:
    directory = Path(str(case["directory"]))
    command = [mpiexec, "-np", "1", str(binary), "denise.inp", "workflow.inp"]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"

    def run() -> float:
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300.0,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        return elapsed

    run()
    samples = [run() for _ in range(repetitions)]
    return {
        **case,
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


def _compare_case(
    baseline: Path,
    candidate: Path,
    mpiexec: str,
    case: dict[str, object],
    repetitions: int,
) -> dict[str, object]:
    directory = Path(str(case["directory"]))
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"

    def run(binary: Path) -> float:
        command = [mpiexec, "-np", "1", str(binary), "denise.inp", "workflow.inp"]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=directory,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=300.0,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        return elapsed

    run(baseline)
    run(candidate)
    samples = {"baseline": [], "candidate": []}
    for repetition in range(repetitions):
        order = (("baseline", baseline), ("candidate", candidate))
        if repetition % 2:
            order = tuple(reversed(order))
        for name, binary in order:
            samples[name].append(run(binary))
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        **case,
        "samples_seconds": samples,
        "median_seconds": medians,
        "speedup": medians["baseline"] / medians["candidate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--baseline-binary", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("at least three repetitions are required")

    args.root.mkdir(parents=True, exist_ok=True)
    cases = {
        "elastic_psv": _make_case(args.root, "elastic_psv", viscoelastic=False),
        "visco_psv_l1": _make_case(args.root, "visco_psv_l1", viscoelastic=True),
    }
    candidate = args.binary.resolve()
    baseline = args.baseline_binary.resolve() if args.baseline_binary else None
    report = {
        "label": args.label,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "omp_num_threads": 1,
        "repetitions": args.repetitions,
        "cases": {
            name: (
                _compare_case(baseline, candidate, args.mpiexec, case, args.repetitions)
                if baseline
                else _time_case(candidate, args.mpiexec, case, args.repetitions)
            )
            for name, case in cases.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
