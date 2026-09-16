"""M6.3c-7d-b2 distributed and boundary objective-gradient gates."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest


EPSILONS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)
ACCEPTANCE_EPSILONS = EPSILONS[:2]
RELATIVE_TOLERANCE = 5.0e-3
CASES = (
    ("b2_horizontal", 2, 1, 3, "physical", 7, 0, 0),
    ("b2_vertical", 1, 2, 1, "legacy", 2, 0, 0),
    ("b2_corner", 2, 2, 3, "physical", 4, 0, 0),
    ("b2_free_surface", 1, 1, 1, "physical", 1, 1, 0),
    ("b2_cpml", 1, 1, 3, "physical", 7, 0, 3),
    ("b2_combined", 2, 2, 1, "physical", 7, 1, 3),
)


@pytest.fixture(scope="module")
def distributed_fd_harness(
        tmp_path_factory: pytest.TempPathFactory,
        repository_root: Path) -> tuple[str, Path]:
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher
    executable = tmp_path_factory.mktemp("m63c7db2") / "m63c7db2"
    sources = [
        "tests/utilities/m63c_objective_directional_fd_harness.c",
        "src/SH/update_v_PML_SH.c", "src/SH/update_s_visc_PML_SH.c",
        "src/SH/exchange_v_SH.c", "src/SH/exchange_s_SH.c",
        "src/SH/surface_elastic_SH.c", "src/SH/visco_sh_gsls_vjp.c",
        "src/SH/visco_sh_material_vjp.c",
        "src/SH/visco_sh_material_timestep_vjp.c",
        "src/SH/visco_sh_material_observable.c",
        "src/SH/update_s_visc_PML_SH_adjoint.c",
        "src/SH/update_v_PML_SH_adjoint.c",
        "src/SH/exchange_v_SH_adjoint.c", "src/SH/exchange_s_SH_adjoint.c",
        "src/SH/surface_elastic_SH_adjoint.c",
        "src/SH/visco_sh_full_state_adjoint_step.c",
        "src/SH/visco_sh_reverse_time_adjoint.c",
        "src/SH/visco_sh_reverse_time_material_gradient.c",
        "src/SH/visco_sh_material_gradient_assembly.c",
        "src/SH/matcopy_SH.c", "src/SH/matcopy_SH_adjoint.c",
        "src/SH/av_mu_SH.c", "src/av_tau.c", "src/q_parameterization.c",
    ]
    command = [compiler, "-std=c99", "-O2", "-fcommon", "-I",
               str(repository_root / "include"),
               *(str(repository_root / source) for source in sources),
               "-o", str(executable), "-lm"]
    completed = subprocess.run(command, cwd=repository_root, text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    assert completed.returncode == 0, completed.stdout
    return launcher, executable


def _execute(launcher: str, executable: Path, case: str,
             ranks: int) -> list[dict[str, object]]:
    completed = subprocess.run(
        [launcher, "--oversubscribe", "-n", str(ranks),
         str(executable), case],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout
    records = [json.loads(line) for line in completed.stdout.splitlines()
               if line.startswith("{")]
    assert len(records) == 5, completed.stdout
    return records


@pytest.mark.parametrize(
    "case,nproc_x,nproc_y,invmat1,q_mode,directions,free_surface,fw", CASES)
def test_distributed_boundary_objective_gradient(
        distributed_fd_harness: tuple[str, Path], case: str,
        nproc_x: int, nproc_y: int, invmat1: int, q_mode: str,
        directions: int, free_surface: int, fw: int) -> None:
    launcher, executable = distributed_fd_harness
    first = _execute(launcher, executable, case, nproc_x * nproc_y)
    second = _execute(launcher, executable, case, nproc_x * nproc_y)
    assert second == first
    for record in first:
        print("M63C7DB2 " + json.dumps(record, sort_keys=True))
    contract, rows = first[0], first[1:]
    assert contract["case"] == case
    assert contract["nproc_x"] == nproc_x
    assert contract["nproc_y"] == nproc_y
    assert contract["invmat1"] == invmat1
    assert contract["q_mode"] == q_mode
    assert contract["directions"] == directions
    assert contract["free_surface"] == free_surface
    assert contract["fw"] == fw
    assert contract["contract"]["dtinv"] == 1
    assert contract["contract"]["objective"] == "0.5*sum(r^2)"
    assert contract["J_base"] > 0.0
    assert contract["max_trace"] > 0.0
    assert math.isfinite(contract["D_ad"])
    assert abs(contract["D_ad"]) > 1.0e-12
    assert contract["direction_norm"] > 0.0
    assert all(value > 0.0 for value in contract["rank_direction_norms"])
    if nproc_x * nproc_y > 1:
        assert contract["source_owner"] != contract["receiver_owner"]
        assert contract["remote_trace"] > 1.0e-12
    if free_surface:
        assert contract["source_owner"] // nproc_x == 0
        assert contract["receiver_owner"] // nproc_x == 0
        assert contract["source_global"][1] <= 3
        assert contract["receiver_global"][1] <= 3
        assert contract["surface_direction_norm"] > 0.0
    if fw:
        assert contract["cpml_coefficients_nontrivial"] == 1
        assert contract["velocity_cpml"] > 0.0
        assert contract["stress_cpml"] > 0.0
        assert contract["overlap_margin_x"] > 0
        assert contract["overlap_margin_y"] > 0
    if directions == 7:
        assert all(abs(contract[key]) > 1.0e-12
                   for key in ("D_primary", "D_rho", "D_Q"))
    assert tuple(row["epsilon"] for row in rows) == pytest.approx(EPSILONS)
    assert all(row["case"] == case for row in rows)
    assert all(math.isfinite(row["D_fd"]) and
               abs(row["D_fd"]) > 1.0e-12 for row in rows)
    acceptance = rows[:len(ACCEPTANCE_EPSILONS)]
    assert tuple(row["epsilon"] for row in acceptance) == pytest.approx(
        ACCEPTANCE_EPSILONS)
    assert max(row["relative_error"] for row in acceptance) <= (
        RELATIVE_TOLERANCE)


def test_distributed_gate_contract_is_global_and_independent(
        repository_root: Path) -> None:
    harness = (repository_root /
               "tests/utilities/m63c_objective_directional_fd_harness.c").read_text()
    assert "POS[1] * NX + i" in harness
    assert "POS[2] * NY + j" in harness
    assert "MPI_Allreduce(&j_base_local, &j_base" in harness
    assert "MPI_Allreduce(&d_primary_local, &d_primary" in harness
    assert "initialize_cpml_coefficients" in harness
    assert "velocity_cpml_local" in harness
    assert "stress_cpml_local" in harness
    assert "NX <= 2 * FW + h" in harness
    for forbidden in ("fitted", "sign_flip", "scale_fit", "time_shift",
                      "grad_obj_sh(", "grad_obj_sh_visc(", "FWI_SH_visc"):
        assert forbidden not in harness
