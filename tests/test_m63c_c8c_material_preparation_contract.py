"""Freeze C8c physical-Q to solver-ready viscoelastic SH preparation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tests" / "m6.3c_c8c_material_preparation_contract.json"
BASE_SHA = "15c7fbf6a8fb50397b2c4572533995693e653e52"


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_identity_and_authoritative_physical_q_state() -> None:
    contract = _contract()

    assert contract["schema_version"] == 1
    assert contract["task"] == "C8C-B4A.0-MATERIAL-PREPARATION-CONTRACT-FREEZE"
    assert contract["baseline_sha"] == BASE_SHA
    assert contract["contract_kind"] == "forward exact viscoelastic SH material-preparation contract"
    assert contract["authoritative_state"] == {
        "owned_primary": "authoritative SH inversion parameter",
        "owned_rho": "authoritative",
        "owned_physical_q": "authoritative",
        "tau": "derived from target physical Q only",
        "staggered_averaged_and_cache_fields": "derived solver state",
        "optimizer_ownership_of_tau_forbidden": True,
    }


def test_graph_global_validation_and_pre_mutation_requirements_are_frozen() -> None:
    contract = _contract()

    assert contract["canonical_preparation_graph"] == [
        "validate configuration and authoritative owned state",
        "globally reconcile validation across MPI ranks",
        "copy owned primary/rho/Q into separate target material only after global success",
        "derive target Tau exclusively through q_to_tau(target Q)",
        "matcopy_SH", "av_mu_SH", "inv_rho_SH", "av_tau",
        "prepare_update_s_visc_SH", "expose target as solver-ready",
    ]
    assert contract["global_fail_closed_mpi"] == {
        "local_validation_permitted": True,
        "global_reconciliation_before_target_mutation": True,
        "global_reconciliation_before_material_halo_collectives": True,
        "any_invalid_rank_causes_consistent_failure_on_all_ranks": True,
        "no_rank_enters_later_material_preparation_collectives_after_global_failure": True,
        "target_unchanged_on_every_rank_after_global_failure": True,
        "particular_mpi_reduction_primitive_prescribed": False,
    }
    assert contract["pre_mutation_validation"] == {
        "required_nonnull_inputs_and_target_storage": True,
        "supported_INVMAT1_only": True,
        "physical_Q_exact_mode_required": True,
        "minimum_relaxation_mechanisms": 1,
        "owned_primary": "finite and valid for the supported INVMAT1 representation",
        "owned_rho": "finite and strictly positive",
        "owned_physical_Q": "finite and valid for the repository q_to_tau mapping",
        "initialized_relaxation_configuration_required": True,
        "generic_primary_positivity_without_representation_justification_forbidden": True,
    }


def test_base_trial_separation_relaxation_and_q_tau_halo_ownership() -> None:
    contract = _contract()

    assert contract["relaxation_configuration"] == {
        "ownership": "external immutable physical configuration, never optimizer state",
        "base_and_trial_must_share": ["L", "FL", "DT", "peta-equivalent configuration"],
        "derived_relaxation_coefficients_may_refresh": True,
        "stale_incompatible_or_uninitialized_configuration_fails_closed": True,
        "Q_dependent_FL_DT_or_mechanism_count_changes_forbidden": True,
    }
    assert contract["base_trial_common_helper"] == {
        "one_production_helper_prepares_authoritative_base_and_B2_trial_states": True,
        "separate_base_only_or_trial_only_preparation_conventions_forbidden": True,
    }
    assert contract["separate_target_ownership"] == {
        "trial_target_must_not_overwrite_prepared_base_material": True,
        "rejected_or_unsuccessful_trial_requires_no_base_restoration": [
            "primary", "rho", "Q", "Tau", "halos", "derived viscoelastic caches"
        ],
    }
    assert contract["q_tau_halo_ownership"] == {
        "owned_Q_copied_into_target_pqs": True,
        "owned_Tau_regenerated_from_target_Q": True,
        "Tau_halos_established_by": "matcopy_SH",
        "solver_staggered_Tau_derives_after_valid_Tau_halos": True,
        "physical_Q_halos_not_required_by_current_solver_preparation_graph": True,
        "invented_physical_Q_halos_for_symmetry_forbidden": True,
    }


def test_identity_sensitivity_legacy_exclusions_and_scope_boundary() -> None:
    contract = _contract()

    assert contract["zero_step_identity"]["precondition"] == (
        "B2 alpha == 0 with identical Base and Trial authoritative states"
    )
    assert contract["zero_step_identity"]["authoritative_owned_exact_identity"] == [
        "primary", "rho", "Q"
    ]
    assert contract["zero_step_identity"]["strict_elementwise_equality_attempted_first"] is True
    assert contract["zero_step_identity"]["nonbitwise_exception_requires_explicit_justification_and_tight_tolerance"] is True
    assert contract["sensitivity_contracts"]["Q_only"] == {
        "required_chain": "Q -> Tau -> averaged Tau -> Tau-dependent viscoelastic update coefficients",
        "primary_and_rho_unchanged_when_not_perturbed": True,
        "stale_attenuation_cache_forbidden": True,
    }
    assert contract["high_level_forbidden_paths"] == [
        "calc_mat_change_test_SH_visc", "matcopy_elastic_SH",
        "direct authoritative Tau mutation", "obj_sh material setup",
    ]
    assert contract["permitted_low_level_helpers"] == [
        "q_to_tau", "matcopy_SH", "av_mu_SH", "inv_rho_SH", "av_tau",
        "prepare_update_s_visc_SH",
    ]
    assert contract["scope_boundary"]["responsibility"] == (
        "authoritative parameter state -> solver-ready exact viscoelastic SH material"
    )
    assert contract["expected_production_scope"] == [
        "include/fd.h", "src/SH/visco_sh_exact_material_preparation.c", "src/Makefile"
    ]
