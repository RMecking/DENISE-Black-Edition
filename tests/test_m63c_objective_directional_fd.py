"""M6.3c-7d-a objective contract and first true material-direction FD gate."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


EPSILONS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)
ACCEPTANCE_EPSILONS = EPSILONS[:2]
RELATIVE_TOLERANCE = 5.0e-3
CASES = (
    ("m3_mu", 3, "physical", 1),
    ("m3_rho", 3, "physical", 2),
    ("m3_q_legacy", 3, "legacy", 4),
    ("m3_q_physical", 3, "physical", 4),
    ("m3_combined_legacy", 3, "legacy", 7),
    ("m3_combined_physical", 3, "physical", 7),
    ("m1_vs", 1, "physical", 1),
    ("m1_rho", 1, "physical", 2),
    ("m1_q_legacy", 1, "legacy", 4),
    ("m1_q_physical", 1, "physical", 4),
    ("m1_combined_legacy", 1, "legacy", 7),
    ("m1_combined_physical", 1, "physical", 7),
)


def test_production_sh_l2_objective_contract(repository_root: Path) -> None:
    """Freeze the executed SH LNORM=2/GF2 sampling and cotangent contract."""
    sampling = (repository_root / "src/SH/sh_visc.c").read_text()
    wrapper = (repository_root / "src/SH/calc_res_SH.c").read_text()
    residual = (repository_root / "src/calc_res.c").read_text()
    assert "seismo_ssg(nt, ntr, (*acq).recpos_loc, (*seisSH).sectionvz" in sampling
    assert "(*seisSH).sectionvz" in wrapper
    assert "(*seisSHfwi).sectionvzdiff" in wrapper
    assert "intseis = (section[i][j]-sectiondata[i][j])" in residual
    assert "sectiondiff[i][invtime]=intseis" in residual
    assert "L2+=0.5 * sectiondiff[i][invtime]*sectiondiff[i][invtime]" in residual
    # The first production sample is explicitly nulled by replacing observed
    # with synthetic before residual formation.  No DT enters GF2 LNORM=2.
    assert "if(j==1){sectiondata[i][j]=section[i][j];}" in residual
    l2 = residual[residual.index("/* calculate L2 residuals */") :]
    gf2 = l2[l2.index("if(GRAD_FORM==2){") :]
    gf2 = gf2[: gf2.index("sectiondiff[i][invtime]=intseis")]
    assert "DT*" not in gf2 and "DT *" not in gf2


@pytest.fixture(scope="module")
def directional_fd_harness(tmp_path_factory: pytest.TempPathFactory,
                           repository_root: Path) -> tuple[str, Path]:
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher
    executable = tmp_path_factory.mktemp("m63c7da") / "m63c7da"
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


@pytest.mark.parametrize("case_name,invmat1,q_mode,directions", CASES)
def test_real_forward_physical_directions_match_locked_gradient(
        directional_fd_harness: tuple[str, Path], case_name: str,
        invmat1: int, q_mode: str, directions: int) -> None:
    launcher, executable = directional_fd_harness
    command = [launcher, "--oversubscribe", "-n", "1", str(executable),
               case_name]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=120)
    assert completed.returncode == 0, completed.stdout
    records = [json.loads(line) for line in completed.stdout.splitlines()
               if line.startswith("{")]
    assert len(records) == 5, completed.stdout
    contract, rows = records[0], records[1:]
    assert contract["case"] == case_name
    assert contract["invmat1"] == invmat1
    assert contract["q_mode"] == q_mode
    assert contract["directions"] == directions
    assert contract["contract"] == {
        "lnorm": 2,
        "grad_form": 2,
        "quelltypb": 1,
        "residual": "synthetic-observed",
        "objective": "0.5*sum(r^2)",
        "receiver_cotangent": "r_chronological",
        "objective_dt_factor": 0,
        "receiver_dt_factor": 0,
        "material_quadrature": "discrete_sum_once",
        "dtinv": 1,
    }
    assert contract["J_base"] > 0.0
    assert contract["max_trace"] > 0.0
    assert abs(contract["D_ad"]) > 1.0e-12
    assert contract["direction_norm"] > 0.0
    assert contract["D_ad"] == pytest.approx(
        contract["D_primary"] + contract["D_rho"] + contract["D_Q"],
        rel=2.0e-15, abs=1.0e-30,
    )
    if directions == 1:
        assert contract["D_rho"] == 0.0 and contract["D_Q"] == 0.0
    elif directions == 2:
        assert contract["D_primary"] == 0.0 and contract["D_Q"] == 0.0
    elif directions == 4:
        assert contract["D_primary"] == 0.0 and contract["D_rho"] == 0.0
    else:
        assert all(abs(contract[key]) > 1.0e-12
                   for key in ("D_primary", "D_rho", "D_Q"))
    print("M63C7DB1 " + json.dumps(contract, sort_keys=True))
    assert tuple(row["epsilon"] for row in rows) == pytest.approx(EPSILONS)
    for row in rows:
        print("M63C7DB1 " + json.dumps(row, sort_keys=True))
        assert row["case"] == case_name
        assert row["D_ad"] == contract["D_ad"]
    # Use one fixed central-FD acceptance window for every material case.
    # The smaller predefined steps remain mandatory diagnostics because the
    # single-precision production forward may enter its subtraction-noise
    # floor there; they are never selected or omitted case by case.
    assert all(abs(row["D_fd"]) > 1.0e-12 for row in rows)
    acceptance_rows = rows[:len(ACCEPTANCE_EPSILONS)]
    assert tuple(row["epsilon"] for row in acceptance_rows) == pytest.approx(
        ACCEPTANCE_EPSILONS)
    assert max(row["relative_error"] for row in acceptance_rows) <= (
        RELATIVE_TOLERANCE)


def test_gate_is_independent_and_has_no_fitted_correction(
        repository_root: Path) -> None:
    harness = (repository_root /
               "tests/utilities/m63c_objective_directional_fd_harness.c").read_text()
    assert "objective(plus, observed, nsteps)" in harness
    assert "objective(minus, observed, nsteps)" in harness
    assert "visco_sh_reverse_time_adjoint_material(" in harness
    assert "grad_primary.v[j][i] * direction" in harness
    assert "grad_rho.v[j][i] * direction" in harness
    assert "grad_q.v[j][i] * direction" in harness
    assert "rho->v[j][i] * primary->v[j][i] * primary->v[j][i]" in harness
    assert "q_to_tau(q->v[j][i], mapping)" in harness
    for forbidden in ("fitted", "sign_flip", "scale_fit", "time_shift",
                      "grad_obj_sh(", "grad_obj_sh_visc(", "FWI_SH_visc"):
        assert forbidden not in harness
