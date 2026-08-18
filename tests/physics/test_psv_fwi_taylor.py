from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest

from tests.cases.psv_fwi_gradient import generate_case
from tests.cases.psv_fwi_taylor import (
    PARAMETERS,
    PSVTaylorCase,
    case_hashes,
    gradient_contributions,
    model_at_epsilon,
    taylor_cases,
)
from tests.physics.test_psv_fwi_gradient_audit import _objective, _seismograms
from tests.physics.test_psv_fwi_production_gradient import _gradient, _run
from tests.utilities.runner import executable_sha256
from tests.utilities.taylor import analyze_taylor_remainders


pytestmark = [pytest.mark.integration, pytest.mark.extended]

BASE_SHA = "39587c07c3e01f91839147300d814197637e652b"
BRANCH = "codex/m5.5-psv-taylor-verification"
EPSILONS = (1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_payload_sha256(
    path: Path, *, trace_count: int, samples_per_trace: int
) -> str:
    raw = path.read_bytes()
    trace_size = 240 + 4 * samples_per_trace
    assert len(raw) == trace_count * trace_size
    digest = hashlib.sha256()
    for trace in range(trace_count):
        start = trace * trace_size + 240
        digest.update(raw[start : start + 4 * samples_per_trace])
    return digest.hexdigest()


def _record_run(report: dict[str, object], role: str, result) -> dict[str, object]:
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    record = {
        "role": role,
        "returncode": result.returncode,
        "runtime_seconds": result.runtime_seconds,
        "command": result.command,
    }
    report["run_records"].append(record)
    if report["toolchain"] is None:
        report["toolchain"] = {
            "local_build_context": metadata["local_build_context"],
            "mpi_version": metadata["mpi_version"],
        }
    return record


def _model_file_hashes(directory: Path) -> dict[str, str]:
    return {
        component: _sha256(directory / "model" / f"current.{component}")
        for component in PARAMETERS
    }


def _write_report(repository_root: Path, report: dict[str, object]) -> None:
    (repository_root / "tests/m5.5_psv_taylor_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_case(
    *,
    root: Path,
    case: PSVTaylorCase,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    report: dict[str, object],
) -> list[dict[str, object]]:
    config = case.config
    observed_directory = root / "observed"
    generate_case(observed_directory, model=case.target, config=config, mode=0)
    role = f"m5.5_{case.name}_observed"
    observed_result = _run(
        observed_directory,
        repository_root=repository_root,
        binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role=role,
    )
    _record_run(report, role, observed_result)
    observed = _seismograms(observed_directory, config)

    baseline_directories: list[Path] = []
    baseline_data: list[dict[str, list[float]]] = []
    payload_hashes: list[dict[str, str]] = []
    for repeat in (1, 2):
        directory = root / f"baseline_repeat{repeat}"
        generate_case(directory, model=case.background, config=config, mode=0)
        role = f"m5.5_{case.name}_baseline_repeat{repeat}"
        result = _run(
            directory,
            repository_root=repository_root,
            binary=denise_binary,
            mpiexec=mpiexec,
            config=config,
            role=role,
        )
        _record_run(report, role, result)
        baseline_directories.append(directory)
        baseline_data.append(_seismograms(directory, config))
        payload_hashes.append(
            {
                component: _sample_payload_sha256(
                    directory / "su" / f"synthetic_{component}.su.shot1",
                    trace_count=config.receiver_count,
                    samples_per_trace=config.samples_per_trace,
                )
                for component in ("x", "y")
            }
        )
    assert payload_hashes[0] == payload_hashes[1]

    perturbed_data: dict[float, dict[str, list[float]]] = {}
    perturbed_model_hashes: dict[str, dict[str, str]] = {}
    for epsilon in EPSILONS:
        directory = root / f"epsilon_{epsilon:.8f}"
        generate_case(
            directory,
            model=model_at_epsilon(case, epsilon),
            config=config,
            mode=0,
        )
        role = f"m5.5_{case.name}_epsilon_{epsilon:.8f}"
        result = _run(
            directory,
            repository_root=repository_root,
            binary=denise_binary,
            mpiexec=mpiexec,
            config=config,
            role=role,
        )
        _record_run(report, role, result)
        perturbed_data[epsilon] = _seismograms(directory, config)
        perturbed_model_hashes[f"{epsilon:.8f}"] = _model_file_hashes(directory)

    rows: list[dict[str, object]] = []
    observed_x = observed_directory / "su/synthetic_x.su.shot1"
    observed_y = observed_directory / "su/synthetic_y.su.shot1"
    baseline_objectives = {
        grad_form: [
            _objective(
                synthetic,
                observed,
                config=config,
                grad_form=grad_form,
                data_components=1,
            )
            for synthetic in baseline_data
        ]
        for grad_form in (1, 2)
    }

    for grad_form in (1, 2):
        objectives = [
            _objective(
                perturbed_data[epsilon],
                observed,
                config=config,
                grad_form=grad_form,
                data_components=1,
            )
            for epsilon in EPSILONS
        ]
        gradient_directory = root / f"gradient_gf{grad_form}"
        generate_case(
            gradient_directory,
            model=case.background,
            config=config,
            mode=1,
            grad_form=grad_form,
            data_components=1,
            observed_x=observed_x,
            observed_y=observed_y,
        )
        role = f"m5.5_{case.name}_gradient_gf{grad_form}"
        result = _run(
            gradient_directory,
            repository_root=repository_root,
            binary=denise_binary,
            mpiexec=mpiexec,
            config=config,
            role=role,
        )
        _record_run(report, role, result)
        gradients = {
            component: _gradient(gradient_directory, config, component)
            for component in PARAMETERS
        }
        contributions = gradient_contributions(gradients, case.delta_model)
        contribution_denominator = math.fsum(
            abs(contributions[component]) for component in PARAMETERS
        )
        repeatability_difference = abs(
            baseline_objectives[grad_form][0] - baseline_objectives[grad_form][1]
        )
        repeatability_tolerance = max(
            1.0e-18, 1.0e-12 * abs(baseline_objectives[grad_form][0])
        )
        assert repeatability_difference <= repeatability_tolerance
        assert contributions["total"] != 0.0, "DEGENERATE TAYLOR DIRECTION"
        if repeatability_difference != 0.0:
            assert abs(EPSILONS[0] * contributions["total"]) >= (
                100.0 * repeatability_difference
            ), "DEGENERATE TAYLOR DIRECTION"

        analysis = analyze_taylor_remainders(
            epsilons=EPSILONS,
            objectives=objectives,
            baseline_objective=baseline_objectives[grad_form][0],
            gradient_directional_product=contributions["total"],
            fit_points=4,
        )
        row = {
            "case": case.name,
            "grad_form": grad_form,
            "data_components": 1,
            "holdout": case.holdout,
            "objective_definition": report["objective_definitions"][f"GF{grad_form}"],
            "model_file_sha256": {
                "background": _model_file_hashes(baseline_directories[0]),
                "target": _model_file_hashes(observed_directory),
                "perturbed": perturbed_model_hashes,
            },
            "baseline_repeatability": {
                "vx_sample_payload_sha256": [
                    hashes["x"] for hashes in payload_hashes
                ],
                "vy_sample_payload_sha256": [
                    hashes["y"] for hashes in payload_hashes
                ],
                "j0_run1": baseline_objectives[grad_form][0],
                "j0_run2": baseline_objectives[grad_form][1],
                "absolute_difference": repeatability_difference,
                "acceptance_tolerance": repeatability_tolerance,
            },
            "gradient_contributions": {
                "gVp_dot_deltaVp": contributions["vp"],
                "gVs_dot_deltaVs": contributions["vs"],
                "gRho_dot_deltaRho": contributions["rho"],
                "g_total_dot_delta": contributions["total"],
                "fractional_absolute_contribution": {
                    component: (
                        abs(contributions[component]) / contribution_denominator
                        if contribution_denominator
                        else 0.0
                    )
                    for component in PARAMETERS
                },
            },
            "analysis": analysis,
            "smallest_point_floor_diagnostic": {
                "epsilon": EPSILONS[-1],
                "inside_acceptance_fit_window": False,
                "r0": analysis["rows"][-1]["r0"],
                "r1": analysis["rows"][-1]["r1"],
                "q0_from_previous": analysis["pairwise_q0"][-1],
                "q1_from_previous": analysis["pairwise_q1"][-1],
            },
            "accepted": analysis["accepted"],
        }
        rows.append(row)
        report["results"].append(row)
        _write_report(repository_root, report)

    report["case_execution"].append(
        {
            "case": case.name,
            "expected_denise_runs": 10,
            "actual_denise_runs": sum(
                record["role"].startswith(f"m5.5_{case.name}_")
                for record in report["run_records"]
            ),
            "forward_reused_for_both_grad_forms": True,
        }
    )
    return rows


def test_m55_formal_elastic_psv_taylor_verification(
    tmp_path: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
) -> None:
    actual_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    lineage = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, actual_sha],
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert lineage.returncode == 0, (
        f"executed HEAD {actual_sha} does not descend from reviewed M5.5 base "
        f"{BASE_SHA}: {lineage.stderr.strip()}"
    )
    cases = taylor_cases()
    assert len(cases) == 5
    validation_path = repository_root / "tests/m5.5_psv_taylor_validation.json"
    previous_report = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.exists()
        else {}
    )
    report: dict[str, object] = {
        "milestone": "M5.5 formal elastic PSV Taylor verification",
        "base_git_sha": BASE_SHA,
        "executed_git_sha": actual_sha,
        "branch": BRANCH,
        "denise_executable_sha256": executable_sha256(denise_binary.resolve()),
        "toolchain": None,
        "epsilon_ladder": list(EPSILONS),
        "fixed_acceptance_fit_window": list(EPSILONS[:4]),
        "smallest_point_role": "floating-point-floor diagnostic only",
        "physics_scope": {
            "PHYSICS": 1,
            "L": 0,
            "MODE": 1,
            "INVMAT1": 1,
            "parameters": list(PARAMETERS),
            "grad_forms": [1, 2],
            "data_components": 1,
            "receiver_components": ["vx", "vy"],
        },
        "objective_definitions": {
            "GF1": (
                "0.5*sum((DT*cumsum(raw synthetic-minus-observed residual))^2) "
                "for x and y; physical sample 1 forced to zero"
            ),
            "GF2": (
                "0.5*sum((raw synthetic-minus-observed residual)^2) for x and y; "
                "physical sample 1 forced to zero"
            ),
        },
        "gradient_storage_convention": (
            "gradient_p.old, gradient_p_u.old, and gradient_p_rho.old contain the "
            "positive objective gradient after established C_parameter normalization; "
            "no SH-style sign inversion and no fitted scale"
        ),
        "hash_encoding": "SHA256 of little-endian IEEE754 float64 definition arrays",
        "predeclared_case_hashes": {
            case.name: case_hashes(case) for case in cases
        },
        "run_records": [],
        "case_execution": [],
        "results": [],
        "m5.4_regression_results": previous_report.get(
            "m5.4_regression_results", "pending final regression rerun"
        ),
        "m5.4.1a_regression_results": previous_report.get(
            "m5.4.1a_regression_results", "pending final regression rerun"
        ),
        "changed_production_files": [],
        "final_verdict": "IN PROGRESS",
    }
    _write_report(repository_root, report)

    for case in cases[:4]:
        rows = _run_case(
            root=tmp_path / case.name,
            case=case,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            report=report,
        )
        failures = [row for row in rows if not row["accepted"]]
        if failures:
            report["heterogeneous_holdout_status"] = "HETEROGENEOUS HOLDOUT NOT RUN"
            report["final_verdict"] = "M5.5 TAYLOR CONVERGENCE BLOCKER"
            _write_report(repository_root, report)
            pytest.fail(json.dumps(failures, indent=2))

    holdout_rows = _run_case(
        root=tmp_path / cases[-1].name,
        case=cases[-1],
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        report=report,
    )
    failures = [row for row in holdout_rows if not row["accepted"]]
    if failures:
        report["heterogeneous_holdout_status"] = "RUN AND FAILED"
        report["final_verdict"] = "M5.5 TAYLOR CONVERGENCE BLOCKER"
        _write_report(repository_root, report)
        pytest.fail(json.dumps(failures, indent=2))

    assert len(report["results"]) == 10
    assert len(report["run_records"]) == 50
    assert all(record["returncode"] == 0 for record in report["run_records"])
    assert all(entry["actual_denise_runs"] == 10 for entry in report["case_execution"])
    report["heterogeneous_holdout_status"] = "RUN AND PASSED"
    report["final_verdict"] = "M5.5 ELASTIC PSV TAYLOR VERIFIED"
    _write_report(repository_root, report)
