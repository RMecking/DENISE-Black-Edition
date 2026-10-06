from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, dump_json_yaml, load_case
from .generate import materialize_run, prepare_case
from .qc import data_qc, geometry_qc, model_qc
from .report import build_report
from .runner import run_forward
from .validation import validate_case, write_validation
from .rtm import build_cuda, execute_rtm, forward_audit, freeze_baseline, verify_baseline


DEFAULT_MANIFEST = Path("applications/marmousi2/case.yaml")


def _template() -> dict[str, object]:
    return {
        "case": {"id": "new-case", "title": "New DENISE application case"},
        "provenance": {"denise_core_sha": "REPLACE", "benchmark_sha": "REPLACE", "benchmark_checkout": "../../DENISE-Benchmark"},
        "backend": {"executable": "../../bin/denise", "launcher": "mpiexec", "environment": "posix-mpi", "ranks": 1, "nprocx": 1, "nprocy": 1},
        "grid": {"nx": 100, "ny": 100, "dh_m": 10.0},
        "time": {"dt_s": 0.001, "time_s": 1.0, "sample_stride": 1},
        "physics": {"kind": "elastic-psv", "fd_order": 8, "max_relative_error": 1},
        "boundary": {"free_surface": True, "absorbing_width_gridpoints": 10, "damping_velocity_m_s": 1500.0, "pml_frequency_hz": 10.0},
        "models": {"true": {"prefix": "REPLACE"}, "start_1d": {"prefix": "REPLACE"}, "smooth2": {"prefix": "REPLACE"}},
        "acquisition": {"sources": "REPLACE", "receivers": "REPLACE", "require_grid_alignment": True},
        "wavelet": {"type": 6, "highpass_hz": -5.0, "lowpass_hz": 15.0, "filter_order": 5},
        "outputs": {"forward_data": "generated/true_forward/data", "rtm_future": "runs/rtm", "fwi_future": "runs/fwi"},
        "qc": {"selected_shots": [1], "selected_receivers": [1]},
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="denise-case", description="Reproducible DENISE application-case backend")
    result.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    commands = result.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create a new high-level case manifest")
    init.add_argument("directory", type=Path)
    commands.add_parser("prepare", help="import frozen external inputs byte-for-byte")
    validate = commands.add_parser("validate", help="validate all pre-run contracts")
    validate.add_argument("--allow-missing-executable", action="store_true")
    generate = commands.add_parser("generate", help="materialize deterministic current-DENISE inputs")
    generate.add_argument("--run-name", default="true_forward")
    forward = commands.add_parser("forward", help="validate and run the true elastic P/SV case")
    forward.add_argument("--run-name", default="true_forward")
    forward.add_argument("--timeout", type=float)
    forward.add_argument("--dry-run", action="store_true")
    commands.add_parser("qc-models", help="generate model and geometry QC")
    qc_data = commands.add_parser("qc-data", help="generate header-aware seismic-data QC")
    qc_data.add_argument("--run-name", default="true_forward")
    report = commands.add_parser("report", help="build a portable self-contained HTML report")
    report.add_argument("--run-name", default="true_forward")
    commands.add_parser("fd4-freeze", help="freeze protected FD8/core content identities for the separate FD4 lane")
    commands.add_parser("verify-baseline", help="recheck protected FD8 and core content identities")
    cuda = commands.add_parser("build-cuda", help="build the frozen optional MODE=2 executable with a recorded receipt")
    cuda.add_argument("--nvcc", help="compiler path/name; otherwise NVCC environment or make/PATH default")
    cuda.add_argument("--cuda-archs", help="space-separated architectures; otherwise CUDA_ARCHS environment or make default")
    commands.add_parser("audit-forward", help="validate/hash every FD4 SU trace and physical identity")
    rtm = commands.add_parser("rtm", help="run the isolated matched-FD4 CUDA preflight or full benchmark")
    rtm.add_argument("--stage", choices=("preflight", "full"), default="preflight")
    rtm.add_argument("--timeout", type=float)
    qc_rtm = commands.add_parser("qc-rtm", help="scientific-coordinate raw RTM and background QC")
    qc_rtm.add_argument("--run-name", default="full_rtm")
    commands.add_parser("report-rtm", help="consolidated matched-FD4 portable HTML report")
    conditioning = commands.add_parser("image-condition", help="solver-free A2.5A diagnostics")
    conditioning.add_argument("--case", type=Path, required=True)
    conditioning.add_argument("--output", type=Path, required=True)
    conditioning.add_argument("--runs", type=Path)
    conditioning.add_argument("--models", type=Path)
    publication = commands.add_parser("a25-publication", help="validate saved evidence and regenerate A2.5 reports; no solver")
    publication.add_argument("--evidence", type=Path, required=True)
    publication.add_argument("--a25a-evidence", type=Path, required=True)
    publication.add_argument("--core", type=Path, required=True)
    publication.add_argument("--runs", type=Path, required=True)
    publication.add_argument("--models", type=Path, required=True)
    publication.add_argument("--output", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "a25-publication":
            from .a25_publication import reproduce
            result = reproduce(args.evidence, args.a25a_evidence, args.core, args.runs, args.models, args.output)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "image-condition":
            from .image_conditioning_report import run_conditioning
            if bool(args.runs) != bool(args.models):
                raise ValueError('both logical roots --runs and --models required')
            roots = {'runs': args.runs, 'models': args.models} if args.runs else None
            receipt = run_conditioning(args.case, args.output) if roots is None else run_conditioning(args.case, args.output, roots_override=roots)
            print(json.dumps({'content_id': receipt['content_id'], 'solver_executions': 0, 'builds': 0}))
            return 0
        if args.command == "init":
            target = args.directory / "case.yaml"
            if target.exists():
                raise FileExistsError(f"refusing to overwrite {target}")
            dump_json_yaml(target, _template())
            print(target)
            return 0
        config = load_case(args.manifest)
        if args.command == "prepare":
            print(json.dumps(prepare_case(config), indent=2, sort_keys=True))
        elif args.command == "validate":
            validation = validate_case(config, require_executable=not args.allow_missing_executable)
            path = config.root / "qc" / "validation.json"
            write_validation(validation, path)
            print(json.dumps(validation.as_dict(), indent=2, sort_keys=True))
            return 0 if validation.ok else 1
        elif args.command == "generate":
            print(materialize_run(config, args.run_name))
        elif args.command == "forward":
            print(json.dumps(run_forward(config, run_name=args.run_name, timeout_seconds=args.timeout, dry_run=args.dry_run), indent=2, sort_keys=True))
        elif args.command == "qc-models":
            print(json.dumps({"models": model_qc(config), "geometry": geometry_qc(config)}, indent=2, sort_keys=True))
        elif args.command == "qc-data":
            print(json.dumps(data_qc(config, args.run_name), indent=2, sort_keys=True))
        elif args.command == "report":
            print(build_report(config, args.run_name))
        elif args.command == "fd4-freeze":
            result = freeze_baseline(config)
            print(json.dumps({"baseline_files":len(result['baseline_files']), "core_files":len(result['core_source_sha256'])}))
        elif args.command == "verify-baseline":
            result = verify_baseline(config)
            print(json.dumps({"unchanged":True, "baseline_files":len(result['baseline_files'])}))
        elif args.command == "build-cuda":
            print(json.dumps(build_cuda(config, nvcc=args.nvcc, cuda_archs=args.cuda_archs), indent=2))
        elif args.command == "audit-forward":
            result = forward_audit(config)
            print(json.dumps({"shots":result['shots'], "receivers":result['receivers'], "files":len(result['files'])}))
        elif args.command == "rtm":
            shots = [1,50,100] if args.stage == "preflight" else list(range(1,101))
            print(json.dumps(execute_rtm(config, shots, 'preflight' if args.stage == 'preflight' else 'full_rtm', timeout=args.timeout), indent=2))
        elif args.command == "qc-rtm":
            from .rtm_qc import rtm_qc
            print(json.dumps(rtm_qc(config, args.run_name), indent=2))
        elif args.command == "report-rtm":
            from .rtm_qc import rtm_report
            print(rtm_report(config))
        return 0
    except (ConfigError, FileNotFoundError, FileExistsError, ValueError, RuntimeError) as error:
        print(f"denise-case: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
