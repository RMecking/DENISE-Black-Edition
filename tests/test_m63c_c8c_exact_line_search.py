"""Runtime oracle for the exact physical-Q SH B5A line search."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_b5a_is_active_from_fwi_and_retains_real_b4b_composition_without_legacy_routes(
    repository_root: Path,
) -> None:
    header = (repository_root / "include/fd.h").read_text(encoding="utf-8")
    helper = _compact(
        (repository_root / "src/SH/step_length_est_sh_visc_exact.c").read_text(
            encoding="utf-8"
        )
    )
    driver = _compact(
        (repository_root / "src/SH/FWI_SH_visc.c").read_text(encoding="utf-8")
    )
    assert "step_length_est_sh_visc_exact(" in header
    assert "visco_sh_exact_trial_objective(" in helper
    assert "calc_opt_step(" in helper
    for legacy in ("calc_mat_change_test_SH_visc(", "obj_sh(", "grad_obj_sh(", "ass_gradSH_visc("):
        assert legacy not in helper
    active = driver[driver.index("exact_status=visco_sh_exact_objective_gradient("):]
    b5a = active.index("step_length_est_sh_visc_exact(")
    accepted_b2 = active.index("visco_sh_exact_build_trial_parameter_state(", b5a)
    commit = active.index("exact_base_primary[exact_j][exact_i]=exact_trial_primary", accepted_b2)
    post_acceptance_b4a = active.index("visco_sh_exact_prepare_visco_material(", commit)
    assert b5a < accepted_b2 < commit < post_acceptance_b4a
    supported_block = active[:post_acceptance_b4a]
    for legacy in ("grad_obj_sh(", "ass_gradSH_visc(", "step_length_est_sh(",
                   "calc_mat_change_test_SH_visc(", "obj_sh(", "descent("):
        assert legacy not in supported_block


@pytest.fixture(scope="module")
def line_search_binary(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> tuple[str, Path]:
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher, "mpicc and an MPI launcher are required"
    build = tmp_path_factory.mktemp("m63c_c8c_b5a")
    full_state = build / "m63c_full_state_support.c"
    directional = build / "m63c_directional_support.c"
    b3a = build / "m63c_b3a_support.c"
    b3b = build / "m63c_b3b_support.c"
    b4b = build / "m63c_b4b_support.c"
    harness = build / "m63c_line_search_harness.c"
    signature = "int main(int argc, char **argv) {"
    full_state.write_text(
        (repository_root / "tests/utilities/m63c_full_state_step_harness.c").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    directional_text = (repository_root / "tests/utilities/m63c_objective_directional_fd_harness.c").read_text(encoding="utf-8")
    b3a_text = (repository_root / "tests/utilities/m63c_c8c_exact_objective_shot_harness.c").read_text(encoding="utf-8")
    b3b_text = (repository_root / "tests/utilities/m63c_c8c_exact_multi_shot_objective_harness.c").read_text(encoding="utf-8")
    assert signature in directional_text and signature in b3a_text and signature in b3b_text
    directional.write_text(
        directional_text.replace('#include "m63c_full_state_step_harness.c"', f'#include "{full_state}"').replace(signature, "int m63c_directional_support_main(int argc, char **argv) {"),
        encoding="utf-8",
    )
    b3a.write_text(
        b3a_text.replace('#define M63C_DIRECTIONAL_SUPPORT "m63c_objective_directional_fd_harness.c"', f'#define M63C_DIRECTIONAL_SUPPORT "{directional}"').replace(signature, "int m63c_b3a_support_main(int argc, char **argv) {"),
        encoding="utf-8",
    )
    b3b.write_text(
        b3b_text.replace(
            "static double objective_values[4];",
            "static double objective_values[4];\n"
            "static double b4b_diag_model_max, b4b_diag_observed_max, b4b_diag_residual_max;",
        ).replace('#define M63C_B3A_SUPPORT "m63c_c8c_exact_objective_shot_harness.c"', f'#define M63C_B3A_SUPPORT "{b3a}"').replace(signature, "int m63c_b3b_support_main(int argc, char **argv) {"),
        encoding="utf-8",
    )
    b4b_text = (repository_root / "tests/utilities/m63c_c8c_exact_trial_objective_harness.c").read_text(encoding="utf-8")
    b4b.write_text(
        b4b_text.replace('#define M63C_B3B_SUPPORT "m63c_b3b_support.c"', f'#define M63C_B3B_SUPPORT "{b3b}"').replace(signature, "int m63c_b4b_embedded_main(int argc, char **argv) {"),
        encoding="utf-8",
    )
    harness.write_text(
        (repository_root / "tests/utilities/m63c_c8c_exact_line_search_harness.c").read_text(encoding="utf-8").replace('#define M63C_B4B_SUPPORT "m63c_c8c_exact_trial_objective_harness.c"', f'#define M63C_B4B_SUPPORT "{b4b}"').replace("#define main m63c_b4b_embedded_main\n", "").replace("#undef main\n", ""),
        encoding="utf-8",
    )
    output = build / "line_search"
    sources = [
        str(harness),
        "src/SH/step_length_est_sh_visc_exact.c", "src/SH/obj_sh_visc_exact_trial.c",
        "src/SH/visco_sh_exact_trial_state.c", "src/SH/visco_sh_exact_material_preparation.c",
        "src/SH/obj_sh_visc_exact.c", "src/SH/obj_sh_visc_exact_shot.c",
        "src/SH/grad_obj_sh_visc_exact.c", "src/SH/grad_obj_sh_visc_exact_shot.c",
        "src/SH/visco_sh_reverse_time_material_gradient.c", "src/SH/visco_sh_material_gradient_assembly.c",
        "src/SH/sh_visc.c", "src/SH/alloc_SH.c", "src/SH/dealloc_SH.c", "src/SH/zero_denise_visc_SH.c",
        "src/seismo_ssg.c", "src/SH/update_v_PML_SH.c", "src/SH/update_s_visc_PML_SH.c",
        "src/SH/exchange_v_SH.c", "src/SH/exchange_s_SH.c", "src/SH/surface_elastic_SH.c",
        "src/SH/visco_sh_gsls_vjp.c", "src/SH/visco_sh_material_vjp.c", "src/SH/visco_sh_material_timestep_vjp.c",
        "src/SH/visco_sh_material_observable.c", "src/SH/update_s_visc_PML_SH_adjoint.c",
        "src/SH/update_v_PML_SH_adjoint.c", "src/SH/exchange_v_SH_adjoint.c", "src/SH/exchange_s_SH_adjoint.c",
        "src/SH/surface_elastic_SH_adjoint.c", "src/SH/visco_sh_full_state_adjoint_step.c",
        "src/SH/visco_sh_reverse_time_adjoint.c", "src/SH/matcopy_SH.c", "src/SH/matcopy_SH_adjoint.c",
        "src/SH/av_mu_SH.c", "src/SH/inv_rho_SH.c", "src/av_tau.c", "src/SH/prepare_update_s_visc_SH.c",
        "src/q_parameterization.c", "src/splitsrc.c", "src/wavelet.c", "src/calc_opt_step.c", "src/solvelin.c",
    ]
    command = [
        compiler, "-std=c99", "-O1", "-g", "-fcommon", "-I", str(repository_root / "include"),
        *(str(repository_root / source) for source in sources), "-lm",
        "-Wl,--wrap=visco_sh_exact_trial_objective", "-Wl,--wrap=visco_sh_exact_build_trial_parameter_state",
        "-Wl,--wrap=visco_sh_exact_prepare_visco_material", "-Wl,--wrap=visco_sh_exact_objective",
        "-Wl,--wrap=matcopy_SH", "-Wl,--wrap=sh_visc", "-Wl,--wrap=splitsrc",
        "-Wl,--wrap=visco_sh_exact_objective_shot", "-Wl,--wrap=visco_sh_exact_objective_gradient_shot",
        "-Wl,--wrap=visco_sh_reverse_time_adjoint_material", "-Wl,--wrap=visco_sh_distributed_material_gradient_vjp",
        "-Wl,--wrap=calc_mat_change_test_SH_visc", "-Wl,--wrap=obj_sh", "-Wl,--wrap=grad_obj_sh", "-Wl,--wrap=ass_gradSH_visc",
        "-o", str(output),
    ]
    completed = subprocess.run(command, cwd=repository_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert completed.returncode == 0, completed.stdout
    return launcher, output


def _run(launcher: str, executable: Path, ranks: int, mode: str) -> dict[str, object]:
    completed = subprocess.run(
        [launcher, "--oversubscribe", "-n", str(ranks), str(executable), mode],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180, env=os.environ.copy(),
    )
    assert completed.returncode == 0, completed.stdout
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1, completed.stdout
    return rows[0]


def test_real_b5a_b4b_search_preserves_base_and_scalar_policy(
    line_search_binary: tuple[str, Path],
) -> None:
    launcher, executable = line_search_binary
    record = _run(launcher, executable, 1, "success")
    for key in ("real_b5a_linked", "real_b4b_used", "scalar_policy_consistent", "base_unchanged", "trial_reuse", "physical_q_propagation", "subtractive_sign"):
        assert record[key] is True, key
    assert record["legacy_path_used"] is False
    assert record["base_trial_alias"] is False
    assert record["base_model_committed"] is False
    assert record["candidate_count"] >= 2
    assert record["b4b_calls"] >= 2
    assert len(record["candidate_alphas"]) == record["b4b_calls"]
    assert len(record["candidate_objectives"]) == record["b4b_calls"]
    assert len(set(record["candidate_alphas"])) >= 2
    assert record["alpha0"] != 0.0 and record["alpha0"] != record["alpha1"]


def test_b5a_contract_c_fails_closed_without_publication(
    line_search_binary: tuple[str, Path],
) -> None:
    launcher, executable = line_search_binary
    record = _run(launcher, executable, 1, "contract-c")
    assert record == {
        "mode": "contract-c", "failure": True, "fail_closed": True,
        "result_sentinel_unchanged": True, "base_unchanged": True, "b4b_calls": 1,
        "legacy_fallback": False, "mpi_ranks": 1, "mpi_identical_failure": True,
    }


def test_two_rank_b5a_has_identical_success_and_failure_decisions(
    line_search_binary: tuple[str, Path],
) -> None:
    launcher, executable = line_search_binary
    success = _run(launcher, executable, 2, "success")
    assert success["mpi_ranks"] == 2
    assert success["mpi_identical"] is True
    assert success["scalar_policy_consistent"] is True
    failure = _run(launcher, executable, 2, "mpi-failure")
    assert failure == {
        "mode": "mpi-failure", "failure": True, "fail_closed": True,
        "result_sentinel_unchanged": True, "base_unchanged": True, "b4b_calls": 1,
        "legacy_fallback": False, "mpi_ranks": 2, "mpi_identical_failure": True,
    }
