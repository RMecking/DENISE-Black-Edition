"""Real-MPI runtime oracle for the inactive B4B exact trial objective."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def _compact(text: str) -> str:
    text = re.sub(r"/\\*.*?\\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\\n]*", "", text)
    return re.sub(r"\\s+", "", text)


def test_b4b_is_public_but_remains_outside_the_active_fwi_driver(
    repository_root: Path,
) -> None:
    header = (repository_root / "include/fd.h").read_text(encoding="utf-8")
    helper = _compact(
        (repository_root / "src/SH/obj_sh_visc_exact_trial.c").read_text(
            encoding="utf-8"
        )
    )
    driver = _compact(
        (repository_root / "src/SH/FWI_SH_visc.c").read_text(encoding="utf-8")
    )

    assert "visco_sh_exact_trial_objective(" in header
    assert "visco_sh_exact_trial_objective(" in helper
    for downstream in (
        "visco_sh_exact_build_trial_parameter_state(",
        "visco_sh_exact_prepare_visco_material(",
        "visco_sh_exact_objective(",
    ):
        assert downstream in helper
    assert "visco_sh_exact_trial_objective(" not in driver


@pytest.fixture(scope="module")
def b4b_binaries(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> tuple[str, Path, Path]:
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher, "mpicc and an MPI launcher are required"
    build = tmp_path_factory.mktemp("m63c_c8c_b4b")

    full_state = build / "m63c_full_state_support.c"
    directional = build / "m63c_directional_support.c"
    b3a = build / "m63c_b3a_support.c"
    b3b = build / "m63c_b3b_support.c"
    signature = "int main(int argc, char **argv) {"
    b3a_text = (
        repository_root / "tests/utilities/m63c_c8c_exact_objective_shot_harness.c"
    ).read_text(encoding="utf-8")
    b3b_text = (
        repository_root / "tests/utilities/m63c_c8c_exact_multi_shot_objective_harness.c"
    ).read_text(encoding="utf-8")
    # Instrument only the temporary copied B3B support.  The B4B.1 oracle
    # needs to distinguish a material-only Q change from an acquisition that
    # actually samples a propagated, nonzero wavefield.
    b3b_text = b3b_text.replace(
        "static double objective_values[4];",
        "static double objective_values[4];\n"
        "static double b4b_diag_model_max, b4b_diag_observed_max, b4b_diag_residual_max;",
    ).replace(
        "    if (slot < 4) objective_values[slot] = result->objective;",
        "    for (int i = 1; i <= request->nrec_local; ++i)\n"
        "        for (int n = 1; n <= request->ns; ++n) {\n"
        "            double modelled = fabs(request->seismogram->sectionvz[i][n]);\n"
        "            double observed = fabs(request->observed_vz[i][n]);\n"
        "            b4b_diag_model_max = fmax(b4b_diag_model_max, modelled);\n"
        "            b4b_diag_observed_max = fmax(b4b_diag_observed_max, observed);\n"
        "            b4b_diag_residual_max = fmax(b4b_diag_residual_max,\n"
        "                fabs(request->seismogram->sectionvz[i][n] - request->observed_vz[i][n]));\n"
        "        }\n"
        "    if (slot < 4) objective_values[slot] = result->objective;",
    ).replace(
        "    memset(objective_values, 0, sizeof(objective_values));",
        "    memset(objective_values, 0, sizeof(objective_values));\n"
        "    b4b_diag_model_max = b4b_diag_observed_max = b4b_diag_residual_max = 0.0;",
    )
    assert signature in b3a_text and signature in b3b_text
    directional_text = (
        repository_root / "tests/utilities/m63c_objective_directional_fd_harness.c"
    ).read_text(encoding="utf-8")
    assert signature in directional_text
    full_state.write_text(
        (repository_root / "tests/utilities/m63c_full_state_step_harness.c").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    directional.write_text(
        directional_text.replace(
            '#include "m63c_full_state_step_harness.c"', f'#include "{full_state}"'
        ).replace(signature, "int m63c_directional_support_main(int argc, char **argv) {"),
        encoding="utf-8",
    )
    b3a.write_text(
        b3a_text.replace(signature, "int m63c_b3a_support_main(int argc, char **argv) {"),
        encoding="utf-8",
    )
    b3b.write_text(
        b3b_text.replace(signature, "int m63c_b3b_support_main(int argc, char **argv) {"),
        encoding="utf-8",
    )

    common = [
        "-std=c99", "-O1", "-g", "-fcommon", "-I", str(repository_root / "include"),
        f'-DM63C_DIRECTIONAL_SUPPORT="{directional}"',
        f'-DM63C_B3A_SUPPORT="{b3a}"', f'-DM63C_B3B_SUPPORT="{b3b}"',
        str(repository_root / "tests/utilities/m63c_c8c_exact_trial_objective_harness.c"),
        str(repository_root / "src/SH/obj_sh_visc_exact_trial.c"),
        str(repository_root / "src/SH/visco_sh_exact_trial_state.c"),
        str(repository_root / "src/SH/visco_sh_exact_material_preparation.c"),
        str(repository_root / "src/SH/obj_sh_visc_exact.c"),
        str(repository_root / "src/SH/obj_sh_visc_exact_shot.c"),
        str(repository_root / "src/SH/grad_obj_sh_visc_exact.c"),
        str(repository_root / "src/SH/grad_obj_sh_visc_exact_shot.c"),
        str(repository_root / "src/SH/visco_sh_reverse_time_material_gradient.c"),
        str(repository_root / "src/SH/visco_sh_material_gradient_assembly.c"),
        str(repository_root / "src/SH/sh_visc.c"), str(repository_root / "src/SH/alloc_SH.c"),
        str(repository_root / "src/SH/dealloc_SH.c"), str(repository_root / "src/SH/zero_denise_visc_SH.c"),
        str(repository_root / "src/seismo_ssg.c"), str(repository_root / "src/SH/update_v_PML_SH.c"),
        str(repository_root / "src/SH/update_s_visc_PML_SH.c"), str(repository_root / "src/SH/exchange_v_SH.c"),
        str(repository_root / "src/SH/exchange_s_SH.c"), str(repository_root / "src/SH/surface_elastic_SH.c"),
        str(repository_root / "src/SH/visco_sh_gsls_vjp.c"), str(repository_root / "src/SH/visco_sh_material_vjp.c"),
        str(repository_root / "src/SH/visco_sh_material_timestep_vjp.c"), str(repository_root / "src/SH/visco_sh_material_observable.c"),
        str(repository_root / "src/SH/update_s_visc_PML_SH_adjoint.c"), str(repository_root / "src/SH/update_v_PML_SH_adjoint.c"),
        str(repository_root / "src/SH/exchange_v_SH_adjoint.c"), str(repository_root / "src/SH/exchange_s_SH_adjoint.c"),
        str(repository_root / "src/SH/surface_elastic_SH_adjoint.c"), str(repository_root / "src/SH/visco_sh_full_state_adjoint_step.c"),
        str(repository_root / "src/SH/visco_sh_reverse_time_adjoint.c"), str(repository_root / "src/SH/matcopy_SH.c"),
        str(repository_root / "src/SH/matcopy_SH_adjoint.c"), str(repository_root / "src/SH/av_mu_SH.c"),
        str(repository_root / "src/SH/inv_rho_SH.c"), str(repository_root / "src/av_tau.c"),
        str(repository_root / "src/SH/prepare_update_s_visc_SH.c"), str(repository_root / "src/q_parameterization.c"),
        str(repository_root / "src/splitsrc.c"), str(repository_root / "src/wavelet.c"), "-lm",
        "-Wl,--wrap=sh_visc", "-Wl,--wrap=splitsrc",
        "-Wl,--wrap=visco_sh_exact_objective_shot",
        "-Wl,--wrap=visco_sh_exact_objective_gradient_shot",
        "-Wl,--wrap=visco_sh_reverse_time_adjoint_material",
        "-Wl,--wrap=visco_sh_distributed_material_gradient_vjp",
        "-Wl,--wrap=visco_sh_exact_build_trial_parameter_state",
        "-Wl,--wrap=visco_sh_exact_prepare_visco_material",
        "-Wl,--wrap=visco_sh_exact_objective",
        "-Wl,--wrap=matcopy_SH",
    ]

    def build_binary(name: str, asan: bool) -> Path:
        output = build / name
        command = [compiler, *( ["-fsanitize=address", "-fno-omit-frame-pointer"] if asan else [] ), *common, "-o", str(output)]
        if asan:
            command.append("-fsanitize=address")
        completed = subprocess.run(command, cwd=repository_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert completed.returncode == 0, completed.stdout
        return output

    return launcher, build_binary("trial_objective", False), build_binary("trial_objective_asan", True)


def _run(
    launcher: str, executable: Path, ranks: int, *, asan: bool, mode: str | None = None
) -> dict[str, object]:
    environment = os.environ.copy()
    if asan:
        environment["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
    completed = subprocess.run(
        [launcher, "--oversubscribe", "-n", str(ranks), str(executable), *([mode] if mode else [])],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=180, env=environment,
    )
    assert completed.returncode == 0, completed.stdout
    rows = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(rows) == 1, completed.stdout
    return rows[0]


def _assert_oracle(record: dict[str, object], ranks: int) -> None:
    for key in (
        "real_b4b_linked", "real_b2_used", "real_b4a_used", "real_b3b_used",
        "call_order", "b2_failure", "b4a_failure", "contract_c_failure",
        "sentinel_unchanged", "base_unchanged_failures", "base_unchanged_success",
        "zero_authoritative_identity", "zero_material_identity", "zero_objective_identity",
        "nonzero_propagation", "q_tau_propagation", "q_cache_propagation",
        "q_objective_sensitivity", "primary_sensitivity", "rho_sensitivity",
        "reuse_aba", "independent_multishot", "one_source_simultaneous",
        "two_source_contract_c", "b2_global_reconciliation", "no_collective_divergence",
    ):
        assert record[key] is True, key
    assert record["b4a_calls_after_b2_failure"] == 0
    assert record["b3b_calls_after_b2_failure"] == 0
    assert record["b3b_calls_after_b4a_failure"] == 0
    assert record["observed_dataset_order"] == [1, 2]
    assert record["mpi_ranks"] == ranks
    assert record["zero_relative_difference"] <= 1.0e-12
    assert record["j_base"] > 0.0 and record["j_trial_alpha0"] > 0.0


def test_real_trial_objective_composition_and_two_rank_transactions(
    b4b_binaries: tuple[str, Path, Path],
) -> None:
    launcher, executable, _ = b4b_binaries
    record = _run(launcher, executable, 2, asan=False, mode="two-rank")
    assert record == {
        "mode": "two-rank",
        "real_b4b_linked": True,
        "real_b2_used": True,
        "real_b4a_used": True,
        "real_b3b_used": True,
        "two_rank_success": True,
        "b2_global_reconciliation": True,
        "b4a_global_failure": True,
        "b3b_after_b2_failure": 0,
        "b3b_after_b4a_failure": 0,
        "objective_sentinel_unchanged": True,
        "base_unchanged": True,
        "collective_divergence_or_hang": False,
        "mpi_ranks": 2,
    }


def test_real_trial_objective_composition_is_asan_clean(
    b4b_binaries: tuple[str, Path, Path],
) -> None:
    launcher, _, executable = b4b_binaries
    _assert_oracle(_run(launcher, executable, 1, asan=True), 1)
