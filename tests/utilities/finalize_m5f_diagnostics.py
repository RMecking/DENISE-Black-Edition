#!/usr/bin/env python3
"""Attach computed build and secondary-run provenance to M5.0f diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--build-metadata", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository.resolve()
    primary_path = repository / "tests" / "m5f_density_gradient_diagnostics.json"
    heterogeneous_path = (
        repository / "tests" / "m5f_density_gradient_heterogeneous.json"
    )
    dtinv_path = repository / "tests" / "m5f_density_gradient_dtinv3.json"
    patch_path = repository / "tests" / "m5f_sh_density_instrumentation.patch"
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    build = json.loads(args.build_metadata.read_text(encoding="utf-8"))
    dtinv = json.loads(dtinv_path.read_text(encoding="utf-8"))
    dtinv_ceiling = max(
        5.0e-5,
        2.0 * dtinv["fd_stability"]["M"],
        dtinv["decomposition_relative"],
    )
    primary.update(
        {
            "temporary_patch_sha256": _sha256(patch_path),
            "build_provenance": build,
            "heterogeneous_run": True,
            "heterogeneous_artifact": {
                "path": str(heterogeneous_path),
                "sha256": _sha256(heterogeneous_path),
                "status": "PASS within independently measured FD uncertainty",
            },
            "dtinv_secondary_run": True,
            "dtinv_secondary_artifact": {
                "path": str(dtinv_path),
                "sha256": _sha256(dtinv_path),
                "status": "FAIL fixed uncertainty-based acceptance",
                "failed_component": "M",
                "absolute_k_error": abs(dtinv["k"]["M"] - 1.0),
                "acceptance_ceiling": dtinv_ceiling,
            },
            "verdict": "ADDITIONAL DEFECT IDENTIFIED",
            "verdict_reason": (
                "DTINV=1 homogeneous and heterogeneous R/M/T decomposition closes, "
                "but the single DTINV=3 material check exceeds its predefined "
                "finite-difference uncertainty ceiling."
            ),
        }
    )
    primary_path.write_text(json.dumps(primary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
