"""Freeze B4B's inactive exact viscoelastic SH trial-objective composition."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tests" / "m6.3c_c8c_trial_objective_contract.json"
BASE_SHA = "e88fcd2c321f3aacdfad98cb80f775f0fe0884d1"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_identity_composition_and_immutable_base_contract() -> None:
    contract = _contract()

    assert contract["schema_version"] == 1
    assert contract["task"] == "C8C-B4B.0-EXACT-TRIAL-OBJECTIVE-CONTRACT-FREEZE"
    assert contract["baseline_sha"] == BASE_SHA
    assert contract["contract_kind"] == (
        "forward exact viscoelastic SH trial-objective composition contract"
    )
    assert contract["canonical_trial_objective_graph"] == [
        "authoritative Base primary/rho/Q plus optimizer_step primary/rho/Q plus alpha",
        "visco_sh_exact_build_trial_parameter_state",
        "separate Trial authoritative primary/rho/Q",
        "visco_sh_exact_prepare_visco_material",
        "separate solver-ready Trial matSH",
        "visco_sh_exact_objective",
        "publish J_trial(alpha) only after success",
    ]
    assert contract["required_production_helpers"] == [
        "visco_sh_exact_build_trial_parameter_state",
        "visco_sh_exact_prepare_visco_material",
        "visco_sh_exact_objective",
    ]
    assert contract["base_state_immutability"] == {
        "unchanged": [
            "Base primary", "Base rho", "Base physical Q", "Base Tau",
            "Base material halos", "Base solver-derived material caches",
        ],
        "trial_material_uses_separate_storage": True,
        "base_restore_after_trial_evaluation_forbidden": True,
    }


def test_sign_physical_q_and_alpha_zero_identity_contract() -> None:
    contract = _contract()

    assert contract["optimizer_sign_contract"] == {
        "raw_gradient": "g_raw = dJ/dm",
        "trial_trajectory": "m_trial(alpha) = m_base - alpha * p",
        "steepest_optimizer_step": "p = g_raw",
        "mathematical_trajectory_direction": "-p",
        "b4b_receives_already_defined_optimizer_subtractive_step": True,
        "forbidden": [
            "extra negation", "descent() transformation", "extra -DT",
            "legacy mu-to-Vs gradient conversion", "legacy rho coupling",
        ],
    }
    assert contract["physical_q_authority"] == {
        "trial_update": "Q_trial = Q_base - alpha * p_Q",
        "bounds_and_validity_owner": "B2 frozen trial-state semantics",
        "required_derivation": "Q_trial -> Tau_trial -> B4A prepared attenuation caches",
        "forbidden": [
            "direct Tau update", "Tau optimizer state",
            "reuse Base Tau for changed Trial Q",
            "alias physical-Q step semantics to legacy Tau storage",
        ],
    }
    assert contract["alpha_zero_identity"] == {
        "authoritative_trial_equals_base_exactly": ["primary", "rho", "Q"],
        "material_comparison": "attempt B4A-valid bitwise material-state identity first",
        "material_equivalence": "Trial material equals canonically B4A-prepared Base material",
        "objective_relative_difference_maximum": 1e-12,
    }


def test_experiment_observed_data_and_transactional_contract() -> None:
    contract = _contract()

    assert contract["experiment_plan_identity"] == {
        "base_and_trial_use_identical_supported_experiment_semantics": True,
        "independent_mode": {
            "condition": "RUN_MULTIPLE_SHOTS != 0",
            "experiments": "one per physical source",
        },
        "one_source_simultaneous_mode": {
            "condition": "RUN_MULTIPLE_SHOTS == 0 && nsrc == 1", "experiments": 1,
        },
        "unsupported_simultaneous_mode": {
            "condition": "RUN_MULTIPLE_SHOTS == 0 && nsrc > 1",
            "action": "fail closed under Contract C",
        },
        "trial_shot_plan_reinterpretation_forbidden": True,
    }
    assert contract["observed_data_identity"] == {
        "same_observed_dataset_for_corresponding_experiment": True,
        "alpha_may_change_only": "model and Trial material state",
        "unchanged": [
            "observed shot index", "receiver geometry",
            "source experiment membership", "residual convention",
        ],
    }
    assert contract["transactional_failure"] == {
        "failing_stages": [
            "B4B-level validation", "B2 trial-state construction",
            "B4A material preparation", "B3B objective evaluation",
        ],
        "failure_action": "report failure",
        "valid_objective_output_unchanged_on_failure": True,
        "base_state_unchanged_on_failure": True,
        "partial_zero_or_default_objective_publication_forbidden": True,
        "stage_order": [
            "validate B4B pointers/configuration", "build B2 Trial authoritative state",
            "prepare B4A Trial material", "evaluate B3B objective",
            "publish J_trial after success",
        ],
    }


def test_scope_mpi_legacy_and_future_runtime_oracle_contract() -> None:
    contract = _contract()

    assert contract["temporary_state_ownership"] == {
        "caller_preallocated_trial_buffers_and_material_targets_may_be_reused": True,
        "scientifically_relevant_trial_state_fully_regenerated": True,
        "result_must_not_depend_on_stale_trial_contents": True,
        "failed_evaluation_must_not_make_stale_J_trial_current": True,
        "fresh_allocation_per_alpha_required": False,
    }
    assert contract["objective_only_scope"] == {
        "output": "success/failure plus exact Trial objective",
        "forbidden": [
            "calculate adjoint", "calculate gradient", "assemble raw gradients",
            "optimizer direction logic", "accept or reject line-search alpha",
            "change Base model", "update FWI_SH_visc",
        ],
    }
    assert contract["legacy_paths_forbidden"] == [
        "obj_sh()", "calc_mat_change_test_SH_visc()", "matcopy_elastic_SH()",
        "legacy Tau trial updates", "TESTSHOT_START/END/INCR semantics",
        "legacy optimizer sign machinery",
    ]
    assert contract["mpi_stage_consistency"] == {
        "all_participating_ranks_execute_same_composition_stage_order": True,
        "global_B4A_failure_prevents_all_ranks_entering_B3B": True,
        "incompatible_collective_rank_divergence_forbidden": True,
    }
    assert contract["line_search_policy"] == {
        "monotonic_positive_alpha_reduces_objective": False,
        "B4B_is_mathematically_neutral_objective_evaluation": True,
    }
    assert contract["expected_production_api"] == "visco_sh_exact_trial_objective(...)"
    assert contract["expected_production_scope"] == [
        "include/fd.h", "src/Makefile", "src/SH/obj_sh_visc_exact_trial.c",
    ]
    assert contract["scope_expansion_rule"] == (
        "NUMERICS must stop and report before requiring another production file"
    )
    assert contract["future_b4b1_runtime_oracle"] == [
        "real B2 helper used", "real B4A helper used", "real B3B objective used",
        "alpha-zero Base/Trial material identity", "alpha-zero objective identity",
        "nonzero alpha changes intended model/material state",
        "physical-Q perturbation reaches attenuation caches and objective",
        "Base remains unchanged", "failure sentinels remain unchanged",
        "supported independent multi-shot behavior", "Contract-C fail-closed behavior",
        "at least 2-rank MPI execution", "deterministic repeated evaluation",
    ]


def test_existing_building_blocks_are_the_declared_inactive_composition_boundary(
    repository_root: Path,
) -> None:
    header = (repository_root / "include/fd.h").read_text(encoding="utf-8")
    driver = (repository_root / "src/SH/FWI_SH_visc.c").read_text(encoding="utf-8")

    for helper in _contract()["required_production_helpers"]:
        assert f"{helper}(" in header
    assert "visco_sh_exact_trial_objective(" not in header
    assert "visco_sh_exact_trial_objective(" not in driver
