"""Runtime contracts for the inactive C8c exact SH objective-only shot path."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_objective_only_static_contract(repository_root: Path) -> None:
    source = _compact(
        (repository_root / "src/SH/obj_sh_visc_exact_shot.c").read_text(
            encoding="utf-8"
        )
    )
    driver = _compact(
        (repository_root / "src/SH/FWI_SH_visc.c").read_text(encoding="utf-8")
    )
    assert "sh_visc(" in source and "MPI_Allreduce(" in source
    assert "sectionvz[i+1][n+1]-request->observed_vz[i+1][n+1]" in source
    assert "for(n=1;n<NT;++n)" in source
    for forbidden in (
        "visco_sh_material_observable_trajectory_init",
        "visco_sh_reverse_time_adjoint_material",
        "visco_sh_exact_objective_gradient_shot",
        "obj_sh(",
        "calc_res_SH(",
        "grad_obj_sh(",
        "grad_obj_sh_visc(",
        "ass_gradSH_visc(",
        "descent(",
        "PCG(",
        "LBFGS(",
        "waveconv_u",
        "waveconv_rho",
        "waveconv_ts",
        "forward_prop_rho_z",
        "forward_prop_sxz",
        "forward_prop_syz",
    ):
        assert forbidden not in source
    assert "visco_sh_exact_objective_shot(" not in driver


@pytest.fixture(scope="module")
def objective_shot_harness(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> tuple[str, Path]:
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher
    build_dir = tmp_path_factory.mktemp("m63c_c8c_b3a")
    executable = build_dir / "objective_shot"
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
    sources = (
        "tests/utilities/m63c_c8c_exact_objective_shot_harness.c",
        "src/SH/obj_sh_visc_exact_shot.c",
        "src/SH/grad_obj_sh_visc_exact_shot.c",
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
        "src/SH/visco_sh_reverse_time_material_gradient.c",
        "src/SH/visco_sh_material_gradient_assembly.c",
        "src/SH/matcopy_SH.c",
        "src/SH/matcopy_SH_adjoint.c",
        "src/SH/av_mu_SH.c",
        "src/av_tau.c",
        "src/q_parameterization.c",
    )
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fcommon",
        f'-DM63C_DIRECTIONAL_SUPPORT="{support_copy}"',
        "-I",
        str(repository_root / "include"),
        "-I",
        str(repository_root / "tests/utilities"),
        *(str(repository_root / source) for source in sources),
        "-Wl,--wrap=sh_visc",
        "-Wl,--wrap=visco_sh_reverse_time_adjoint_material",
        "-o",
        str(executable),
        "-lm",
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
    return launcher, executable


def test_real_objective_only_matches_exact_gradient_without_adjoint(
    objective_shot_harness: tuple[str, Path],
) -> None:
    launcher, executable = objective_shot_harness
    completed = subprocess.run(
        [launcher, "--oversubscribe", "-n", "1", str(executable)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout
    records = [
        json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")
    ]
    assert len(records) == 3, completed.stdout
    fs_records = {record["free_surface"]: record for record in records[:2]}
    assert set(fs_records) == {0, 1}
    for record in fs_records.values():
        assert record["objective_only"] > 0.0
        assert record["objective_gradient"] > 0.0
        assert record["relative_difference"] <= 1.0e-12
        assert record["manual_relative_difference"] <= 1.0e-12
        assert record["included_first_difference"] > 1.0e-3
        assert record["adjoint_calls_objective_only"] == 0
        assert record["adjoint_calls_gradient"] >= 1
        assert record["forward_calls_objective_only"] == 1
    preflight = records[2]
    assert preflight == {"preflight_transactional": True, "preflight_forward_calls": 0}
