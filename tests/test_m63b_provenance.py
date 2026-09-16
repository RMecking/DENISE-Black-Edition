from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


AUDIT_SHA = "a4d2ca176f518a8414aa95aef256265dd89fa567"
M63B_ORACLE_SHA = "8c1bfc9c5a5c9f39396b9be5030464f683d3ab5d"
AUDIT_DOCUMENT_SHA256 = "03c757210f0b86db5be82cd0dbe3f6650ce9115c2933926ccda9c5f2f1bca28a"
VALIDATION_SHA256 = "15a8b21077f03e902d2edc735941442b384935431b749540401a0d018e5e0552"


def _git(repository_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_m63b_validation_artifact_locks_healthy_runs_and_raw_objectives(repository_root):
    path = repository_root / "tests/m6.3b_visco_sh_fwi_attenuation_validation.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == VALIDATION_SHA256
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["audit_commit"] == AUDIT_SHA
    executed_lineage = _git(
        repository_root,
        "merge-base",
        "--is-ancestor",
        AUDIT_SHA,
        report["executed_git_sha"],
    )
    assert executed_lineage.returncode == 0, executed_lineage.stderr
    assert report["run_count"] == 29
    assert report["total_denise_run_count"] == 33
    assert len(report["run_records"]) == 29
    assert len({record["role"] for record in report["run_records"]}) == 29
    assert all(record["returncode"] == 0 for record in report["run_records"])
    assert all(record["timed_out"] is False for record in report["run_records"])
    assert all(record["finite"] is True for record in report["run_records"])
    assert all(
        len(record["seismogram_payload_sha256"]) == 64
        for record in report["run_records"]
    )
    assert all("seismogram_sha256" not in record for record in report["run_records"])
    assert all(
        set(record["input_sha256"]) == {
            "vs", "rho", "qs", "source", "receiver", "parameters", "workflow"
        }
        for record in report["run_records"]
    )
    assert report["production_changed_files"] == []

    for metrics in report["q_directional_derivatives"].values():
        assert set(metrics["raw_objectives"]) == {"-0.02", "-0.01", "+0.01", "+0.02"}
        assert abs(metrics["five_point"]) > 1.0e-8
        assert metrics["relative_five_point_vs_centered"] < 5.0e-3
    tau = report["tau_directional_derivative"]
    assert set(tau["raw_objectives"]) == {"-0.02", "-0.01", "+0.01", "+0.02"}
    assert abs(tau["five_point"]) > 1.0e-8
    assert tau["relative_five_point_vs_centered"] < 5.0e-3
    assert all(row["seismogram_identical"] is True for row in report["dtinv"].values())
    assert all(row["relative_l2_vs_dtinv1"] == 0.0 for row in report["dtinv"].values())
    assert all(row["relative_l2"] <= 2.0e-6 for row in report["mpi"].values())
    assert all(row["normalized_correlation"] >= 0.999999999 for row in report["mpi"].values())

    baseline_hashes = report["baseline_q_model_sha256"]
    assert len(set(baseline_hashes.values())) == 1
    directional = report["directional_q_model_sha256"]
    assert directional["fs0_1x1"] == directional["fs0_2x1"]
    assert directional["fs0_1x1"] == directional["fs0_1x2"]

    dot = report["production_adjoint_dot_product"]
    artifact = repository_root / "tests/utilities/m63b_production_adjoint_instrumentation.py"
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == dot["instrumentation"]["artifact_sha256"]
    assert dot["instrumentation"]["base_git_sha"] == AUDIT_SHA
    assert dot["instrumentation"]["changed_files"] == [
        "src/SH/FWI_SH_visc.c",
        "src/SH/grad_obj_sh_visc.c",
        "src/SH/sh_visc.c",
    ]
    assert len(dot["instrumentation"]["sha256"]) == 64
    assert set(dot["cases"]) == {"fs0_1x1", "fs1_1x1", "fs0_2x1", "fs0_1x2"}
    threshold = dot["red_acceptance"]["minimum_relative_residual"]
    for row in dot["cases"].values():
        assert row["returncode"] == 0
        assert row["timed_out"] is False
        assert row["relative_residual"] >= threshold
        assert abs(row["left"]) > 1.0e-12
        assert abs(row["right"]) > 1.0e-12
        assert len(row["executable_sha256"]) == 64
        assert row["instrumentation_artifact_sha256"] == dot["instrumentation"]["artifact_sha256"]
        assert all(len(value) == 64 for value in row["input_sha256"].values())
        assert "observed" not in row["input_sha256"]
        assert len(row["observed_payload_sha256"]) == 64
    assert all(
        len(vector["sha256_float32"]) == 64
        and len(vector["observed_payload_sha256"]) == 64
        for vector in dot["independent_data_vectors"].values()
    )


def test_m63b_lineage_audit_immutability_and_strict_red_classification(repository_root):
    audit = repository_root / "docs/m6.3_visco_sh_fwi_attenuation_audit.md"
    assert hashlib.sha256(audit.read_bytes()).hexdigest() == AUDIT_DOCUMENT_SHA256
    head = _git(repository_root, "rev-parse", "HEAD")
    assert head.returncode == 0
    current_lineage = _git(
        repository_root,
        "merge-base",
        "--is-ancestor",
        M63B_ORACLE_SHA,
        head.stdout.strip(),
    )
    assert current_lineage.returncode == 0, current_lineage.stderr
    audit_lineage = _git(
        repository_root,
        "merge-base",
        "--is-ancestor",
        AUDIT_SHA,
        M63B_ORACLE_SHA,
    )
    assert audit_lineage.returncode == 0, audit_lineage.stderr
    production = _git(
        repository_root,
        "diff",
        "--name-only",
        f"{AUDIT_SHA}..{M63B_ORACLE_SHA}",
        "--",
        "src",
        "include",
        "par",
    )
    assert production.returncode == 0
    assert production.stdout.strip() == ""

    integration = (
        repository_root / "tests/physics/test_visco_sh_fwi_attenuation_oracle.py"
    ).read_text(encoding="utf-8")
    for exception in (
        "KnownM63ActivePhysicsSplit",
        "KnownM63ViscoAdjointDefect",
        "KnownM63TauGradientDisconnected",
        "KnownM63ProductionAdjointDotDefect",
    ):
        assert f"raises={exception}" in integration
        assert f"raise {exception}(" in integration
    assert integration.count("strict=True") == 4
