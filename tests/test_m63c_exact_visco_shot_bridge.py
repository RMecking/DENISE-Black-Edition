"""M6.3c-8b2-b1 inactive exact viscoelastic SH shot-bridge contracts."""

from __future__ import annotations

import json
import re
from pathlib import Path
import shutil
import subprocess

import pytest


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_exact_shot_bridge_is_declared_and_built(repository_root: Path):
    header = _read(repository_root, "include/fd.h")
    makefile = _read(repository_root, "src/Makefile")
    assert "struct visco_sh_exact_shot_request" in header
    assert "visco_sh_exact_objective_gradient_shot(" in header
    assert "grad_obj_sh_visc_exact_shot.c" in makefile


def test_exact_shot_bridge_uses_only_locked_forward_and_reverse(
    repository_root: Path,
):
    source = _compact(
        _read(repository_root, "src/SH/grad_obj_sh_visc_exact_shot.c")
    )
    assert source.count("sh_visc_with_material_trajectory(") == 1
    assert source.count("visco_sh_reverse_time_adjoint_material(") == 1
    for forbidden in (
        "grad_obj_sh(",
        "grad_obj_sh_visc(",
        "assemble_gradSH_exact(",
        "ass_gradSH_visc(",
        "descent(",
        "PCG(",
        "LBFGS(",
        "step_length_est_sh(",
        "calc_mat_change_test_SH_visc(",
    ):
        assert forbidden not in source


def test_frozen_objective_and_chronological_cotangent_contract(
    repository_root: Path,
):
    source = _compact(
        _read(repository_root, "src/SH/grad_obj_sh_visc_exact_shot.c")
    )
    assert "for(n=1;n<NT;++n)" in source
    assert "sectionvz[i+1][n+1]-request->observed_vz[i+1][n+1]" in source
    assert "bar_receiver[(size_t)n*request->nrec_local+i]=residual" in source
    assert "local_objective+=0.5*residual*residual" in source
    assert "MPI_Allreduce(&local_objective,&global_objective" in source
    assert "DT*DTINV" not in source


def test_frozen_preconditions_are_fail_fast(repository_root: Path):
    source = _compact(
        _read(repository_root, "src/SH/grad_obj_sh_visc_exact_shot.c")
    )
    for condition in (
        "DTINV!=1",
        "LNORM!=2",
        "GRAD_FORM!=2",
        "N_ORDER!=0",
        "TIMEWIN!=0",
        "OFFSET_MUTE!=0",
        "TRKILL!=0",
        "SEISMO!=1",
    ):
        assert condition in source


def test_material_and_q_mapping_use_prepared_production_fields(
    repository_root: Path,
):
    source = _compact(
        _read(repository_root, "src/SH/grad_obj_sh_visc_exact_shot.c")
    )
    for assignment in (
        "material_context.mu_x=request->material->puip",
        "material_context.tau_x=request->material->ptausipjp",
        "material_context.mu_y=request->material->pujp",
        "material_context.tau_y=request->material->ptaus",
        "material_context.primary_post=request->material->pu",
        "material_context.rho_post=request->material->prho",
        "material_context.owned_q=request->material->pqs",
    ):
        assert assignment in source
    assert "init_q_tau_mapping(&mapping,Q_PARAMETERIZATION_MODE,L,FL" in source
    assert "q_to_tau(" not in source


def test_three_distinct_internal_cotangent_workspaces(repository_root: Path):
    source = _compact(
        _read(repository_root, "src/SH/grad_obj_sh_visc_exact_shot.c")
    )
    assert "structexact_shot_workspaceterminal,initial,scratch" in source
    assert "&terminal.state,&initial.state,&scratch.state" in source


def test_active_fwi_path_and_locked_capture_kernels_remain_unchanged(
    repository_root: Path,
):
    driver = _compact(_read(repository_root, "src/SH/FWI_SH_visc.c"))
    velocity = _read(repository_root, "src/SH/update_v_PML_SH.c")
    stress = _read(repository_root, "src/SH/update_s_visc_PML_SH.c")
    assert "L2sum=grad_obj_sh(" in driver
    assert "visco_sh_exact_objective_gradient_shot(" not in driver
    assert velocity.count("visco_sh_material_observable_is_active()") == 1
    assert stress.count("visco_sh_material_observable_is_active()") == 1


def test_reference_objective_contract_has_zero_first_sample():
    synthetic = ((2.0, 3.0, -1.0), (5.0, 4.5, 6.0))
    observed = ((9.0, 1.0, -2.0), (-3.0, 5.0, 2.0))
    cotangent = [0.0] * 6
    objective = 0.0
    for sample in range(1, 3):
        for receiver in range(2):
            residual = synthetic[receiver][sample] - observed[receiver][sample]
            cotangent[sample * 2 + receiver] = residual
            objective += 0.5 * residual * residual
    assert cotangent == [0.0, 0.0, 2.0, -0.5, 1.0, 4.0]
    assert objective == 10.625


@pytest.fixture(scope="module")
def exact_shot_harness(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> tuple[str, Path]:
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher
    build_dir = tmp_path_factory.mktemp("m63c8b2b1")
    executable = build_dir / "m63c8b2b1"
    locked_source = (
        repository_root / "tests/utilities/m63c_objective_directional_fd_harness.c"
    ).read_text(encoding="utf-8")
    signature = "int main(int argc, char **argv) {"
    assert locked_source.count(signature) == 1
    locked_copy = build_dir / "m63c_locked_objective_support.c"
    locked_copy.write_text(
        locked_source.replace(
            signature,
            "int m63c_locked_objective_harness_main(int argc, char **argv) {",
        ),
        encoding="utf-8",
    )
    sources = [
        "tests/utilities/m63c_exact_visco_shot_bridge_harness.c",
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
    ]
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fcommon",
        f'-DM63C_LOCKED_OBJECTIVE_HARNESS="{locked_copy}"',
        "-I",
        str(repository_root / "include"),
        "-I",
        str(repository_root / "tests/utilities"),
        *(str(repository_root / source) for source in sources),
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


@pytest.mark.parametrize(
    "case_name,ranks,invmat1,q_mode",
    (
        ("bridge_m1_physical", 1, 1, "physical"),
        ("bridge_m3_legacy", 1, 3, "legacy"),
        ("bridge_mpi_physical", 2, 1, "physical"),
    ),
)
def test_exact_production_shot_bridge_matches_directional_fd(
    exact_shot_harness: tuple[str, Path],
    case_name: str,
    ranks: int,
    invmat1: int,
    q_mode: str,
) -> None:
    launcher, executable = exact_shot_harness
    command = [
        launcher,
        "--oversubscribe",
        "-n",
        str(ranks),
        str(executable),
        case_name,
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout
    records = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    assert len(records) == 3, completed.stdout
    contract, rows = records[0], records[1:]
    assert contract["case"] == case_name
    assert contract["invmat1"] == invmat1
    assert contract["q_mode"] == q_mode
    assert contract["nproc_x"] == ranks and contract["nproc_y"] == 1
    assert contract["objective"] == pytest.approx(
        contract["objective_reference"], rel=2.0e-14, abs=1.0e-30
    )
    assert contract["max_cotangent_error"] == 0.0
    assert contract["first_cotangent"] == 0.0
    assert contract["repeat_objective"] == contract["objective"]
    assert contract["repeat_D_ad"] == contract["D_ad"]
    assert abs(contract["D_ad"]) > 1.0e-12
    if ranks == 2:
        assert contract["source_owner"] != contract["receiver_owner"]
        assert contract["remote_trace"] > 0.0
    assert tuple(row["epsilon"] for row in rows) == pytest.approx(
        (1.0e-2, 3.0e-3)
    )
    assert all(row["case"] == case_name for row in rows)
    assert all(row["D_ad"] == contract["D_ad"] for row in rows)
    assert max(row["relative_error"] for row in rows) <= 5.0e-3
    print("M63C8B2B1 " + json.dumps(contract, sort_keys=True))
    for row in rows:
        print("M63C8B2B1 " + json.dumps(row, sort_keys=True))
