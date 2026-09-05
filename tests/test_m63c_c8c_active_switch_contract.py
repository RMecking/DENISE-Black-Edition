"""Freeze C8c active exact-viscoelastic switch acceptance before implementation."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tests" / "m6.3c_c8c_active_switch_contract.json"
BASE_SHA = "67f1040549ade5a7b93a723700448cbfedb92513"
STRICT_XFAIL_SOURCE_FILES = (
    "tests/physics/test_visco_sh_fwi_attenuation_oracle.py",
    "tests/test_m63c_acceptance_contract.py",
)


def _wsl_git_dir_for_windows_worktree() -> str | None:
    """Translate this worktree's Windows gitdir when pytest runs under WSL."""
    if os.name == "nt":
        return None
    pointer = ROOT / ".git"
    if not pointer.is_file():
        return None
    prefix = "gitdir: "
    line = pointer.read_text(encoding="utf-8").strip()
    if not line.startswith(prefix):
        return None
    git_dir = line.removeprefix(prefix).replace("\\", "/")
    match = re.fullmatch(r"([A-Za-z]):/(.+)", git_dir)
    if match is None:
        return None
    return f"/mnt/{match.group(1).lower()}/{match.group(2)}"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    command = ["git", "-c", f"safe.directory={ROOT}"]
    git_dir = _wsl_git_dir_for_windows_worktree()
    if git_dir is not None:
        command.extend(("--git-dir", git_dir, "--work-tree", str(ROOT)))
    command.extend(args)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _baseline_source(relative_path: str) -> str:
    return _git("show", f"{BASE_SHA}:{relative_path}").stdout


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def _is_strict_xfail(decorator: ast.expr) -> bool:
    if not isinstance(decorator, ast.Call):
        return False
    if not isinstance(decorator.func, ast.Attribute) or decorator.func.attr != "xfail":
        return False
    mark = decorator.func.value
    if not isinstance(mark, ast.Attribute) or mark.attr != "mark":
        return False
    if not isinstance(mark.value, ast.Name) or mark.value.id != "pytest":
        return False
    return any(
        keyword.arg == "strict"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in decorator.keywords
    )


def _strict_xfail_ids() -> tuple[str, ...]:
    ids: list[str] = []
    for relative_path in STRICT_XFAIL_SOURCE_FILES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if any(_is_strict_xfail(decorator) for decorator in node.decorator_list):
                ids.append(f"{relative_path}::{node.name}")
    return tuple(ids)


def _contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_contract_schema_is_complete_and_base_is_resolvable() -> None:
    contract = _contract()
    assert set(contract) == {
        "schema_version",
        "task",
        "baseline_sha",
        "contract_kind",
        "exact_active_base_path",
        "raw_gradient_authority",
        "legacy_post_assembly_exclusion",
        "physical_q_ownership",
        "trial_material_chain",
        "zero_step_identity",
        "experiment_set_identity",
        "shot_semantics",
        "optimizer_sign_contract",
        "final_simultaneous_source_gate",
        "frozen_strict_xfails",
        "baseline_characterization",
    }
    assert contract["schema_version"] == 1
    assert contract["task"] == "C8C-A1-ACTIVE-SWITCH-CONTRACT"
    assert contract["baseline_sha"] == BASE_SHA
    _git("cat-file", "-e", f"{BASE_SHA}^{{commit}}")


def test_contract_freezes_target_scientific_and_architectural_semantics() -> None:
    contract = _contract()

    assert contract["exact_active_base_path"] == {
        "exact_entrypoint": "visco_sh_exact_objective_gradient",
        "supported_exact_c8c_configurations_use_exact_chain": True,
        "forbidden_legacy_raw_gradient_entrypoint": "grad_obj_sh",
        "unsupported_configurations_fail_closed": True,
        "silent_legacy_fallback_forbidden": True,
    }
    assert contract["raw_gradient_authority"] == {
        "primary": "dJ/dm_primary",
        "rho": "dJ/drho",
        "q": "dJ/dQ",
        "forbidden_additional_transforms": [
            "legacy_-DT",
            "mu_to_Vs_remapping",
            "legacy_rho_coupling",
            "Q_to_Tau_gradient_reinterpretation",
        ],
    }
    assert contract["legacy_post_assembly_exclusion"] == {
        "forbidden_call": "ass_gradSH_visc",
        "applies_to": "exact raw gradients",
    }
    assert contract["physical_q_ownership"] == {
        "optimizer_parameter": "Q",
        "derived_solver_state": "Tau",
        "q_and_tau_are_semantically_distinct": True,
        "exact_q_gradient_must_not_alias_legacy_tau_gradient_storage": True,
    }
    assert contract["trial_material_chain"]["ordered_steps"] == [
        "physical_Q_trial_update",
        "Q_bounds",
        "q_to_tau",
        "full_viscoelastic_material_copy_and_halo_preparation",
        "av_tau",
        "viscoelastic_stress_update_preparation",
    ]
    assert contract["zero_step_identity"] == {
        "trial_model_alpha_zero_equals_base_model": True,
        "trial_objective_alpha_zero_equals_base_exact_objective": True,
        "relative_objective_tolerance": 1.0e-12,
    }
    assert contract["experiment_set_identity"] == {
        "base_objective_gradient_and_trial_objective_use_same_source_shot_plan": True
    }
    assert contract["shot_semantics"] == {
        "RUN_MULTIPLE_SHOTS_nonzero": "one experiment per physical source",
        "RUN_MULTIPLE_SHOTS_zero": "one simultaneous experiment containing all physical sources",
    }
    assert contract["optimizer_sign_contract"] == {
        "raw_gradient": "g_raw = dJ/dm",
        "trial_update": "m_trial(alpha) = m_base - alpha * p",
        "optimizer_step": "p = g_raw",
        "model_trajectory_direction": "-p",
        "valid_subtractive_step": "g_raw dot p > 0",
        "directional_derivative_at_zero": "dJ/dalpha|alpha=0 = -g_raw dot p < 0",
    }
    assert contract["final_simultaneous_source_gate"] == {
        "required": True,
        "run_multiple_shots": 0,
        "minimum_physical_sources": 2,
        "kind": "real E2E scientific acceptance",
    }


def test_exactly_the_six_protected_strict_xfails_remain_locked() -> None:
    contract = _contract()
    expected = tuple(row["id"] for row in contract["frozen_strict_xfails"])
    assert len(expected) == 6
    assert len(set(expected)) == 6
    assert all(
        row["classification"] == "XFAIL exposing known defect"
        for row in contract["frozen_strict_xfails"]
    )
    assert _strict_xfail_ids() == expected


def test_base_snapshot_characterizes_the_inactive_switch_gap() -> None:
    contract = _contract()
    driver = _compact(_baseline_source("src/SH/FWI_SH_visc.c"))
    header = _compact(_baseline_source("include/fd.h"))
    trial = _compact(_baseline_source("src/SH/obj_sh.c"))
    exact = _compact(_baseline_source("src/SH/grad_obj_sh_visc_exact.c"))

    assert "L2sum=grad_obj_sh(" in driver
    assert "visco_sh_exact_objective_gradient(" not in driver
    assert "visco_sh_exact_objective_gradient_shot(" not in driver
    assert "structvisco_sh_exact_multi_shot_request" in header
    assert "visco_sh_exact_objective_gradient(" in header
    assert "pqs" in header and "ptaus" in header
    assert "waveconv_ts" in driver and "calc_mat_change_test_SH_visc(" in driver
    assert "matcopy_elastic_SH(" in trial and "matcopy_SH(" not in trial
    assert "nshots=RUN_MULTIPLE_SHOTS?request->nsrc:1" in exact
    assert "source_columns=RUN_MULTIPLE_SHOTS?1:request->nsrc" in exact

    assert contract["baseline_characterization"] == {
        "active_exact_integration": "UNSATISFIED",
        "physical_q_optimizer_ownership": "UNSATISFIED",
        "exact_multi_shot_aggregator": "PRESENT_BUT_INACTIVE",
        "inactive_aggregator_shot_semantics": "SUPPORTED",
        "evidence": [
            "FWI_SH_visc routes the base derivative through grad_obj_sh",
            "FWI_SH_visc does not call visco_sh_exact_objective_gradient",
            "owned pqs storage exists but the optimizer and trial path still use waveconv_ts and ptaus",
            "the trial path uses elastic material copy preparation",
        ],
    }
