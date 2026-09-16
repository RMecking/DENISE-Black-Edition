"""M6.3c-8b2-b2 inactive exact multi-shot orchestration contracts."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_multi_shot_api_is_declared_built_and_inactive(repository_root: Path):
    header = _read(repository_root, "include/fd.h")
    makefile = _read(repository_root, "src/Makefile")
    source = _compact(_read(repository_root, "src/SH/grad_obj_sh_visc_exact.c"))
    active = _compact(_read(repository_root, "src/SH/FWI_SH_visc.c"))
    assert "struct visco_sh_exact_multi_shot_request" in header
    assert "visco_sh_exact_objective_gradient(" in header
    assert "grad_obj_sh_visc_exact.c" in makefile
    assert "visco_sh_exact_objective_gradient(" not in active
    assert "visco_sh_exact_objective_gradient_shot(" not in active
    assert "L2sum=grad_obj_sh(" in active
    assert source.count("visco_sh_exact_objective_gradient_shot(") == 1


def test_multi_shot_raw_contract_excludes_legacy_postprocessing(
    repository_root: Path,
):
    source = _compact(_read(repository_root, "src/SH/grad_obj_sh_visc_exact.c"))
    for forbidden in (
        "ass_gradSH_visc(",
        "precond_SH(",
        "descent(",
        "PCG(",
        "LBFGS(",
        "step_length_est_sh(",
        "calc_mat_change_test_SH_visc(",
        "TESTSHOT_START",
        "TESTSHOT_END",
        "TESTSHOT_INCR",
        "MPI_Allreduce(request->grad",
    ):
        assert forbidden not in source
    assert "objective+=shot_result.objective" in source
    assert "request->grad_primary" in source
    assert "request->grad_rho" in source
    assert "request->grad_q" in source


def test_multi_shot_frozen_preconditions_and_selection(repository_root: Path):
    source = _compact(_read(repository_root, "src/SH/grad_obj_sh_visc_exact.c"))
    for condition in (
        "DTINV!=1",
        "LNORM!=2",
        "GRAD_FORM!=2",
        "N_ORDER!=0",
        "TIMEWIN!=0",
        "OFFSET_MUTE!=0",
        "TRKILL!=0",
        "SEISMO!=1",
        "EPRECOND!=0",
        "SWS_TAPER_CIRCULAR_PER_SHOT!=0",
        "TIME_FILT!=0",
        "INV_STF!=0",
    ):
        assert condition in source
    assert "nshots=RUN_MULTIPLE_SHOTS?request->nsrc:1" in source
    assert "source_columns=RUN_MULTIPLE_SHOTS?1:request->nsrc" in source


def test_multi_shot_uses_production_acquisition_and_observed_flow(
    repository_root: Path,
):
    source = _compact(_read(repository_root, "src/SH/grad_obj_sh_visc_exact.c"))
    for call in ("receiver(", "splitrec(", "splitsrc(", "wavelet(", "inseis("):
        assert call in source
    assert "sectionread[acquisition->recpos_loc[3][i]][n]" in source
    assert "calc_res_SH(" not in source


@pytest.fixture(scope="module")
def multi_shot_harness(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> tuple[str, Path]:
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher
    build_dir = tmp_path_factory.mktemp("m63c8b2b2")
    executable = build_dir / "m63c8b2b2"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fcommon",
        "-I",
        str(repository_root / "include"),
        str(repository_root / "tests/utilities/m63c_exact_visco_multi_shot_harness.c"),
        str(repository_root / "src/SH/grad_obj_sh_visc_exact.c"),
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
        ("multi_m1_physical", 1, 1, "physical"),
        ("multi_m3_legacy_readrec1", 1, 3, "legacy"),
        ("multi_mpi_physical", 2, 1, "physical"),
    ),
)
def test_exact_multi_shot_decomposition_and_directional_fd(
    multi_shot_harness: tuple[str, Path],
    case_name: str,
    ranks: int,
    invmat1: int,
    q_mode: str,
) -> None:
    launcher, executable = multi_shot_harness
    completed = subprocess.run(
        [launcher, "--oversubscribe", "-n", str(ranks), str(executable), case_name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
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
    assert contract["ranks"] == ranks
    assert contract["invmat1"] == invmat1
    assert contract["q_mode"] == q_mode
    assert contract["shots"] == 2
    assert contract["objective"] == contract["single_objective_sum"]
    assert contract["max_gradient_sum_error"] <= 2.0e-7
    assert contract["repeat_objective"] == contract["objective"]
    assert contract["repeat_D_ad"] == contract["D_ad"]
    assert contract["observed_mapping_failures"] == 0
    assert contract["source_flow_failures"] == 0
    assert contract["precondition_rejected"] is True
    assert contract["precondition_outputs_unchanged"] is True
    if ranks == 2:
        assert contract["source1_owner"] != contract["receiver_owner"]
        assert contract["cross_rank_activation"] is True
    assert abs(contract["D_ad"]) > 1.0e-10
    assert tuple(row["epsilon"] for row in rows) == pytest.approx((1.0e-2, 3.0e-3))
    assert all(row["D_ad"] == contract["D_ad"] for row in rows)
    assert max(row["relative_error"] for row in rows) <= 5.0e-3
    for record in records:
        print("M63C8B2B2 " + json.dumps(record, sort_keys=True))
