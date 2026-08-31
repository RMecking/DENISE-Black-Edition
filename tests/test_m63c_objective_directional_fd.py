"""M6.3c-7d-a objective contract and first true material-direction FD gate."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


EPSILONS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)
RELATIVE_TOLERANCE = 5.0e-3


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


def test_real_forward_mu_direction_matches_locked_gradient(
        directional_fd_harness: tuple[str, Path]) -> None:
    launcher, executable = directional_fd_harness
    command = [launcher, "--oversubscribe", "-n", "1", str(executable)]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=120)
    assert completed.returncode == 0, completed.stdout
    records = [json.loads(line) for line in completed.stdout.splitlines()
               if line.startswith("{")]
    assert len(records) == 5, completed.stdout
    contract, rows = records[0], records[1:]
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
    print("M63C7DA " + json.dumps(contract, sort_keys=True))
    assert tuple(row["epsilon"] for row in rows) == pytest.approx(EPSILONS)
    for row in rows:
        print("M63C7DA " + json.dumps(row, sort_keys=True))
        assert row["D_ad"] == contract["D_ad"]
    # The two largest predefined steps should exhibit central-FD contraction;
    # smaller steps may enter the single-precision forward noise floor.
    fd_changes = [abs(rows[index + 1]["D_fd"] - rows[index]["D_fd"])
                  for index in range(len(rows) - 1)]
    assert fd_changes[1] < fd_changes[0]
    # This diagnostic is separate from, and stricter in scale interpretation
    # than, the unchanged relative-error acceptance gate below.
    assert max(abs(row["D_ad_over_D_fd"] - 1.0) for row in rows) < 5.0e-3
    assert max(row["relative_error"] for row in rows) <= RELATIVE_TOLERANCE


def test_gate_is_independent_and_has_no_fitted_correction(
        repository_root: Path) -> None:
    harness = (repository_root /
               "tests/utilities/m63c_objective_directional_fd_harness.c").read_text()
    assert "objective(plus, observed, nsteps)" in harness
    assert "objective(minus, observed, nsteps)" in harness
    assert "visco_sh_reverse_time_adjoint_material(" in harness
    assert "grad_mu.v[j][i] * direction" in harness
    for forbidden in ("fitted", "sign_flip", "scale_fit", "time_shift",
                      "grad_obj_sh(", "grad_obj_sh_visc(", "FWI_SH_visc"):
        assert forbidden not in harness
