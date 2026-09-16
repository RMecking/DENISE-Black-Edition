"""Freeze C8c simultaneous-source observed-data ownership before activation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tests" / "m6.3c_c8c_simultaneous_observed_data_contract.json"
A1_CONTRACT_PATH = ROOT / "tests" / "m6.3c_c8c_active_switch_contract.json"
BASE_SHA = "1a0d67b2fa49520a3e2fb5491643449d380ba3f7"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_freezes_supported_and_rejected_experiment_modes() -> None:
    contract = _contract()

    assert contract["schema_version"] == 1
    assert contract["task"] == "C8C-B3B.0A-SIMULTANEOUS-OBSERVED-DATA-CONTRACT-FREEZE"
    assert contract["baseline_sha"] == BASE_SHA
    assert contract["contract_kind"] == "forward scientific acceptance amendment"
    assert contract["supported_experiment_modes"] == {
        "independent_shots": {
            "condition": "RUN_MULTIPLE_SHOTS != 0 and nsrc >= 1",
            "experiment_count": "one per physical source",
            "observed_data": "one matching observed shot dataset per physical source",
            "existing_exact_experiment_semantics": "unchanged",
        },
        "one_source_simultaneous": {
            "condition": "RUN_MULTIPLE_SHOTS == 0 and nsrc == 1",
            "experiment_count": 1,
            "physical_source_count": 1,
            "observed_data": "observed dataset 1 is the matching single-source observation",
        },
    }
    assert contract["unsupported_experiment_mode"] == {
        "condition": "RUN_MULTIPLE_SHOTS == 0 and nsrc > 1",
        "initial_exact_c8c_path": "fail closed",
        "rejecting_entrypoints": [
            "visco_sh_exact_objective_gradient",
            "visco_sh_exact_objective",
        ],
        "rejection_must_precede": [
            "interpretation of observed data as a valid simultaneous experiment",
            "gradient output overwrite or accumulation",
            "presentation of objective output as scientifically valid",
        ],
        "forbidden_fallbacks": [
            "inseis(1) as a multi-source simultaneous observed-data surrogate",
            "naive sum of independent shot objectives",
        ],
    }


def test_amendment_preserves_historical_a1_identity_and_scope() -> None:
    contract = _contract()
    a1 = json.loads(A1_CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["historical_a1_contract"] == {
        "path": "tests/m6.3c_c8c_active_switch_contract.json",
        "task": "C8C-A1-ACTIVE-SWITCH-CONTRACT",
        "baseline_sha": "67f1040549ade5a7b93a723700448cbfedb92513",
        "preserved": True,
    }
    assert a1["task"] == "C8C-A1-ACTIVE-SWITCH-CONTRACT"
    assert a1["baseline_sha"] == "67f1040549ade5a7b93a723700448cbfedb92513"
    assert a1["final_simultaneous_source_gate"] == {
        "required": True,
        "run_multiple_shots": 0,
        "minimum_physical_sources": 2,
        "kind": "real E2E scientific acceptance",
    }
    assert contract["supersession"] == {
        "supersedes_only": (
            "A1 initial-activation requirement for a real RUN_MULTIPLE_SHOTS == 0 "
            "and nsrc > 1 simultaneous-source E2E case"
        ),
        "does_not_weaken": [
            "independent multi-shot testing",
            "one-source simultaneous testing",
            "objective/gradient identity",
            "physical-Q ownership",
            "sign semantics",
            "fail-closed requirements",
        ],
    }


def test_future_reenablement_requires_observed_data_authority_and_cross_term_oracle() -> None:
    contract = _contract()

    assert contract["future_reenablement"] == {
        "requires_separate_contract": True,
        "observed_data_contract_options": [
            "pre-combined observed simultaneous data",
            "validated construction of simultaneous observed data from per-source gathers",
        ],
        "must_not_choose_option_in_this_amendment": True,
        "required_e2e_oracle": (
            "distinguish 0.5 ||sum_s r_s||^2 from sum_s 0.5 ||r_s||^2 "
            "through nonzero cross terms"
        ),
    }


def test_rejection_targets_match_the_inactive_exact_multi_shot_symbols() -> None:
    objective = (ROOT / "src/SH/obj_sh_visc_exact.c").read_text(encoding="utf-8")
    gradient = (ROOT / "src/SH/grad_obj_sh_visc_exact.c").read_text(encoding="utf-8")
    header = (ROOT / "include/fd.h").read_text(encoding="utf-8")

    assert "int visco_sh_exact_objective(" in objective
    assert "int visco_sh_exact_objective_gradient(" in gradient
    assert "int visco_sh_exact_objective(" in header
    assert "int visco_sh_exact_objective_gradient(" in header
    assert "nshots = RUN_MULTIPLE_SHOTS ? request->nsrc : 1;" in objective
    assert "nshots = RUN_MULTIPLE_SHOTS ? request->nsrc : 1;" in gradient
