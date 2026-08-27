from __future__ import annotations

import hashlib
import json
from pathlib import Path


BASE_SHA = "f0fad66c2521951d26ca40acf0460fed86e43eca"
INSTRUMENTATION_SHA = (
    "9fa327bccccf3e63ba2315c1642a3d81106b4adecb16601bcc71b193452bd061"
)
VALIDATION_SHA = (
    "658ba13c8247598462a12ff3c85f3ee35e08973f6015c854a8cae9f9b7c93f6f"
)
NORMAL_BINARY_SHA = (
    "a39b13a3fff6b79e4f95ec3cc68017e8492fc485f673f07588ea348a14d13510"
)
INSTRUMENTED_BINARY_SHA = (
    "96853f978a74a799cd69fc294c54cb322177c2ca70f1c50e17353644cc953db4"
)
HARD_BOUNDARY_KEYS = [
    "traction_residual",
    "dplus_vz_residual",
    "vz_parity_residual",
    "total_syz_parity_residual",
    "q_surface_residual",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m62b_locked_oracle_provenance(repository_root: Path):
    patch = repository_root / "tests" / "m6.2b_visco_sh_free_surface_instrumentation.patch"
    validation_path = (
        repository_root / "tests" / "m6.2b_visco_sh_free_surface_validation.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    assert _sha256(patch) == INSTRUMENTATION_SHA
    assert _sha256(validation_path) == VALIDATION_SHA
    assert validation["instrumentation_patch_sha256"] == INSTRUMENTATION_SHA
    assert validation["base_git_sha"] == BASE_SHA
    assert validation["executed_run_count"] == 29

    acceptance = validation["acceptance"]
    assert acceptance["waveform"] == {
        "arrival_lag_max_s": 0.001,
        "normalized_correlation_min": 0.999,
        "relative_l2_max": 0.02,
        "signed_amplitude_error_max": 0.03,
    }
    assert acceptance["reference_translation_relative_l2_max"] == 5.0e-5
    assert acceptance["reference_translation_correlation_min"] == 0.999999
    assert acceptance["reference_translation_lag_max_s"] == 0.0005
    assert acceptance["superposition_relative_l2_max"] == 2.0e-6
    assert acceptance["free_surface_zero_image_window_l2_ratio_max"] == 0.1
    assert acceptance["finite_q_sensitivity_relative_l2_min"] == 0.001
    assert acceptance["high_q_endpoint_relative_l2_max"] == 0.05
    assert acceptance["high_q_endpoint_correlation_min"] == 0.999
    assert acceptance["mpi_relative_l2_max"] == 1.0e-6
    assert acceptance["mpi_correlation_min"] == 0.999999
    assert acceptance["boundary"] == {
        "diagnostic_only": {
            "q_parity_residual": {"acceptance_effect": "none"}
        },
        "hard_keys": HARD_BOUNDARY_KEYS,
        "hard_limits": {
            "dplus_vz_residual_max": 5.0e-5,
            "q_surface_residual_max": 2.0e-6,
            "total_syz_parity_residual_max": 2.0e-6,
            "traction_residual_max": 5.0e-6,
            "vz_parity_residual_max": 2.0e-6,
        },
    }
    assert acceptance["stability"] == {
        "calibration_role": "finite-Q FD12 FREE_SURF=0 absorbing reference",
        "metric": "fixed post-source quarter max_abs_vz Q4/Q1",
        "q4_to_q1_max": 0.01,
        "reference_calibration_q4_to_q1": 0.0003110815438951515,
        "source_off": {"n_off": 1257, "n_order": 0, "quellart": 1},
    }

    assert validation["boundary_contract"]["hard_keys"] == HARD_BOUNDARY_KEYS
    assert validation["boundary_contract"]["diagnostic_only"] == {
        "q_parity_residual": {
            "acceptance_effect": "none",
            "measured": 0.04712212861207598,
        }
    }
    assert "q_parity_residual" not in validation["boundary_contract_pass"]
    assert "q_parity_residual_max" not in validation["boundary_limits"]
    assert validation["red_classification"][
        "expected_missing_surface_failures"
    ] == [
        "dplus_vz_residual",
        "vz_parity_residual",
        "total_syz_parity_residual",
    ]
    assert validation["red_classification"][
        "unexpected_reference_or_calibration_failure"
    ] is False

    build = validation["build_identity"]
    assert build["base_head_verified"] is True
    assert build["base_git_sha"] == BASE_SHA
    assert build["production_diff_empty"] is True
    assert build["twenty_nine_run_physics_rerun_required"] is False
    assert build["normal"] == {
        "exact_hash_match": True,
        "fresh_build_sha256": NORMAL_BINARY_SHA,
        "validation_executable_sha256": NORMAL_BINARY_SHA,
    }
    assert build["instrumented"] == {
        "exact_hash_match": True,
        "fresh_build_sha256": INSTRUMENTED_BINARY_SHA,
        "instrumentation_patch_sha256": INSTRUMENTATION_SHA,
        "same_base_source_as_normal": True,
        "validation_executable_sha256": INSTRUMENTED_BINARY_SHA,
    }

    stability = validation["stability"]
    assert stability["acceptance"] == {
        "calibration_role": "finite-Q FD12 FREE_SURF=0 absorbing reference",
        "frozen_from_candidate": False,
        "metric": "fixed post-source quarter max_abs_vz Q4/Q1",
        "q4_to_q1_max": 0.01,
    }
    assert stability["diagnostic_only_legacy_tail_global"]["acceptance_effect"] == "none"
    assert stability["source_off_definition"] == {
        "n_off": 1257,
        "n_order": 0,
        "post_source_quarters": [
            [1258, 2243],
            [2244, 3229],
            [3230, 4215],
            [4216, 5201],
        ],
        "quellart": 1,
        "time_s": 0.6285,
    }
    for result in stability["results"].values():
        assert result["returncode"] == 0
        assert result["executable_sha256"] == INSTRUMENTED_BINARY_SHA
        assert result["q4_to_q1"] <= 0.01

    assert validation["reference_health"] == {
        "both_source_contributions_nonzero": True,
        "linear_superposition_relative_l2": validation["reference_health"][
            "linear_superposition_relative_l2"
        ],
        "outer_returns_outside_comparison": True,
        "separated_direct_and_image_windows": True,
        "translation_self_consistency": validation["reference_health"][
            "translation_self_consistency"
        ],
    }
    assert validation["instrumented_uninstrumented_equivalence"]["relative_l2"] == 0.0
    assert validation["instrumented_uninstrumented_equivalence"][
        "normalized_correlation"
    ] == 1.0
    assert all(
        validation["candidate_runs"][key]["run"]["returncode"] == 0
        for key in (
            "high_q_200",
            "high_q_1000",
            "elastic_m61_endpoint",
            "instrumented_fd12",
        )
    )
    assert validation["final_verdict"] == (
        "M6.2b VISCOELASTIC SH FREE-SURFACE ORACLE LOCKED "
        "— PRE-FIX RED BASELINE ESTABLISHED"
    )


def test_m62b_normal_integration_does_not_overwrite_locked_validation(
    repository_root: Path,
):
    source = (
        repository_root / "tests" / "physics" / "test_visco_sh_free_surface.py"
    ).read_text(encoding="utf-8")
    assert 'tmp_path / "m6.2b_visco_sh_free_surface_live_report.json"' in source
    assert 'os.environ.get("M62B_REGENERATE_LOCKED_VALIDATION") == "1"' in source
    assert source.count('"m6.2b_visco_sh_free_surface_validation.json"') == 1
