"""Runtime contracts for C8c B3b's inactive exact SH objective wrapper."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


PRODUCTION_GRADIENT_SOURCES = (
    "src/SH/grad_obj_sh_visc_exact.c",
    "src/SH/grad_obj_sh_visc_exact_shot.c",
    "src/SH/visco_sh_reverse_time_material_gradient.c",
    "src/SH/visco_sh_material_gradient_assembly.c",
)


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_exact_objective_multi_static_contract(repository_root: Path) -> None:
    objective = _compact(
        (repository_root / "src/SH/obj_sh_visc_exact.c").read_text(encoding="utf-8")
    )
    gradient = _compact(
        (repository_root / "src/SH/grad_obj_sh_visc_exact.c").read_text(
            encoding="utf-8"
        )
    )
    driver = _compact(
        (repository_root / "src/SH/FWI_SH_visc.c").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (repository_root / "tests/m6.3c_c8c_simultaneous_observed_data_contract.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["unsupported_experiment_mode"]["rejecting_entrypoints"] == [
        "visco_sh_exact_objective_gradient",
        "visco_sh_exact_objective",
    ]
    guard = "if((RUN_MULTIPLE_SHOTS==0)&&(request->nsrc>1))return-1;"
    assert guard in objective and guard in gradient
    assert "visco_sh_exact_objective_shot(" in objective
    assert "MPI_Allreduce(" not in objective
    assert "visco_sh_exact_objective(" not in driver
    assert "matrix(0,NY+1,0,NX+1)" in gradient
    assert "free_matrix(shot_primary,0,NY+1,0,NX+1)" in gradient
    for call in ("splitsrc(", "wavelet(", "inseis(", "MPI_Barrier("):
        assert call in objective and call in gradient
    for forbidden in (
        "visco_sh_exact_objective_gradient_shot",
        "visco_sh_reverse_time_adjoint_material",
        "grad_primary",
        "grad_rho",
        "grad_q",
        "waveconv_",
        "ass_gradSH_visc",
        "grad_obj_sh",
        "grad_obj_sh_visc",
        "obj_sh(",
        "calc_res_SH",
        "step_length_est_sh",
        "descent",
        "PCG",
        "LBFGS",
        "visco_sh_exact_build_trial_parameter_state",
        "q_to_tau",
        "matcopy_SH",
        "av_tau",
        "prepare_update_s_visc_SH",
    ):
        assert forbidden not in objective


def _build_harness(
    build_dir: Path, repository_root: Path, *, asan: bool
) -> tuple[str, Path]:
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher
    executable = build_dir / ("multi_objective_asan" if asan else "multi_objective")
    support_source = (
        repository_root / "tests/utilities/m63c_objective_directional_fd_harness.c"
    ).read_text(encoding="utf-8")
    signature = "int main(int argc, char **argv) {"
    assert support_source.count(signature) == 1
    support_copy = build_dir / "m63c_directional_support.c"
    support_copy.write_text(
        support_source.replace(signature, "int m63c_directional_support_main(int argc, char **argv) {"),
        encoding="utf-8",
    )
    b3a_source = (
        repository_root / "tests/utilities/m63c_c8c_exact_objective_shot_harness.c"
    ).read_text(encoding="utf-8")
    assert b3a_source.count(signature) == 1
    b3a_copy = build_dir / "m63c_b3a_support.c"
    b3a_copy.write_text(
        b3a_source.replace(signature, "int m63c_b3a_support_main(int argc, char **argv) {"),
        encoding="utf-8",
    )
    sources = (
        "tests/utilities/m63c_c8c_exact_multi_shot_objective_harness.c",
        "src/SH/obj_sh_visc_exact.c",
        *PRODUCTION_GRADIENT_SOURCES,
        "src/SH/obj_sh_visc_exact_shot.c",
        "src/SH/sh_visc.c",
        "src/SH/alloc_SH.c",
        "src/SH/dealloc_SH.c",
        "src/SH/zero_denise_visc_SH.c",
        "src/seismo_ssg.c",
        "src/SH/update_v_PML_SH.c",
        "src/SH/update_s_visc_PML_SH.c",
        "src/SH/exchange_v_SH.c",
        "src/SH/exchange_s_SH.c",
        "src/SH/surface_elastic_SH.c",
        "src/SH/visco_sh_gsls_vjp.c",
        "src/SH/visco_sh_material_vjp.c",
        "src/SH/visco_sh_material_timestep_vjp.c",
        "src/SH/visco_sh_material_observable.c",
        "src/SH/update_s_visc_PML_SH_adjoint.c",
        "src/SH/update_v_PML_SH_adjoint.c",
        "src/SH/exchange_v_SH_adjoint.c",
        "src/SH/exchange_s_SH_adjoint.c",
        "src/SH/surface_elastic_SH_adjoint.c",
        "src/SH/visco_sh_full_state_adjoint_step.c",
        "src/SH/visco_sh_reverse_time_adjoint.c",
        "src/SH/matcopy_SH.c",
        "src/SH/matcopy_SH_adjoint.c",
        "src/SH/av_mu_SH.c",
        "src/av_tau.c",
        "src/q_parameterization.c",
        "src/splitsrc.c",
        "src/wavelet.c",
    )
    command = [
        compiler,
        "-std=c99",
        "-O1" if asan else "-O2",
        "-fcommon",
        *( ["-fsanitize=address", "-fno-omit-frame-pointer"] if asan else [] ),
        f'-DM63C_DIRECTIONAL_SUPPORT="{support_copy}"',
        f'-DM63C_B3A_SUPPORT="{b3a_copy}"',
        "-I",
        str(repository_root / "include"),
        "-I",
        str(repository_root / "tests/utilities"),
        *(str(repository_root / source) for source in sources),
        "-Wl,--wrap=sh_visc",
        "-Wl,--wrap=splitsrc",
        "-Wl,--wrap=visco_sh_reverse_time_adjoint_material",
        "-Wl,--wrap=visco_sh_exact_objective_shot",
        "-Wl,--wrap=visco_sh_exact_objective_gradient_shot",
        "-Wl,--wrap=visco_sh_distributed_material_gradient_vjp",
        "-o",
        str(executable),
        "-lm",
        *( ["-fsanitize=address"] if asan else [] ),
    ]
    completed = subprocess.run(
        command, cwd=repository_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
    return launcher, executable


@pytest.fixture(scope="module")
def multi_objective_harness(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> tuple[str, Path]:
    return _build_harness(
        tmp_path_factory.mktemp("m63c_c8c_b3b"), repository_root, asan=False
    )


@pytest.fixture(scope="module")
def asan_multi_objective_harness(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> tuple[str, Path]:
    return _build_harness(
        tmp_path_factory.mktemp("m63c_c8c_b3b_asan"), repository_root, asan=True
    )


def _run(launcher: str, executable: Path, *, asan: bool) -> dict:
    environment = os.environ.copy()
    if asan:
        environment["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
    completed = subprocess.run(
        [launcher, "--oversubscribe", "-n", "1", str(executable)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout
    records = [
        json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")
    ]
    assert len(records) == 1, completed.stdout
    return records[0]


def _assert_runtime_contract(record: dict) -> None:
    assert record["rejected_objective_return"] == -1
    assert record["rejected_objective_sentinel_unchanged"] is True
    assert record["rejected_gradient_return"] == -1
    assert record["rejected_gradient_sentinels_unchanged"] is True
    assert record["rejection_inseis_calls"] == 0
    assert record["rejection_splitsrc_calls"] == 0
    assert record["rejection_objective_shot_calls"] == 0
    assert record["rejection_gradient_shot_calls"] == 0
    assert record["rejection_material_vjp_calls"] == 0
    assert record["separate_cardinalities"] == [1, 1]
    assert record["separate_observed_indices"] == [1, 2]
    assert record["separate_objective_shot_calls"] == 2
    assert record["separate_gradient_shot_calls"] == 2
    assert record["separate_shot_count"] == 2
    assert record["j1"] > 0.0 and record["j2"] > 0.0
    assert record["j_separate"] == pytest.approx(record["j1"] + record["j2"], rel=1e-12)
    assert record["separate_gradient_relative_difference"] <= 1.0e-12
    assert record["separate_material_vjp_calls"] >= 2
    assert record["one_source_cardinalities"] == [1]
    assert record["one_source_observed_indices"] == [1]
    assert record["one_source_objective_shot_calls"] == 1
    assert record["one_source_gradient_shot_calls"] == 1
    assert record["one_source_shot_count"] == 1
    assert record["one_source_gradient_relative_difference"] <= 1.0e-12
    assert record["one_source_material_vjp_calls"] >= 1
    assert record["objective_adjoint_calls"] == 0
    assert record["owned_gradients_finite"] is True
    assert record["gradient_repeatable"] is True
    assert record["real_production_gradient_chain"] is True
    assert record["preflight_transactional"] is True
    assert record["intermediate_failure_transactional"] is True
    assert record["resource_cleanup"] is True


def test_real_multi_shot_contract_c_and_halo_regression(
    multi_objective_harness: tuple[str, Path],
) -> None:
    _assert_runtime_contract(_run(*multi_objective_harness, asan=False))


def test_real_multi_shot_halo_regression_is_asan_clean(
    asan_multi_objective_harness: tuple[str, Path],
) -> None:
    _assert_runtime_contract(_run(*asan_multi_objective_harness, asan=True))
