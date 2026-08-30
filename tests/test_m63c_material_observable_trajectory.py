"""M6.3c-7a passive forward material-observable trajectory verification."""

from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utilities.m63c_full_state_step_reference import make_states
from tests.utilities.m63c_material_observable_reference import (
    C7A_REFERENCE_RELATIVE_MAX,
    CASES,
    forward_trajectory,
)


def _relative_l2(actual, expected):
    numerator = math.fsum((a - b) ** 2 for a, b in zip(actual, expected))
    denominator = math.fsum(value * value for value in expected)
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def test_c7a_api_exposes_exactly_the_three_frozen_channels(repository_root: Path):
    header = (repository_root / "include/fd.h").read_text()
    body = header.split("struct visco_sh_material_observable_step {", 1)[1].split(
        "};", 1
    )[0]
    declarations = [line.strip() for line in body.splitlines() if "float **" in line]
    assert declarations == [
        "float **qsum;",
        "float **strain_x;",
        "float **strain_y;",
    ]
    assert "vz" not in body
    assert "sxz" not in body
    assert "syz" not in body
    assert "psi_" not in body


def test_c7a_capture_sites_preserve_the_frozen_timestep_order(repository_root: Path):
    velocity = (repository_root / "src/SH/update_v_PML_SH.c").read_text()
    stress = (repository_root / "src/SH/update_s_visc_PML_SH.c").read_text()
    assert velocity.count("visco_sh_material_observable_capture_qsum(\n") == 6
    assert stress.count("visco_sh_material_observable_capture_strain(j, i") == 6
    source_position = velocity.index("/* Forward Modelling (sw==0) */")
    start = 0
    for _ in range(6):
        capture = velocity.index(
            "visco_sh_material_observable_capture_qsum(\n", start
        )
        update = velocity.index("vz[j][i] +=", capture)
        assert capture < update < source_position
        start = capture + 1
    for block in stress.split("visco_sh_material_observable_capture_strain(")[1:]:
        assert block.index("computing sums of the old memory variables") < block.index(
            "updating components of the stress tensor"
        )


def test_c7a_matrix_covers_predeclared_temporal_spatial_and_boundary_scope():
    cases = [entry.case for entry in CASES]
    assert {case.fdorder for case in cases} == {2, 4, 6, 8, 10, 12}
    assert {case.mechanisms for case in cases} == {1, 3}
    assert {case.fw for case in cases} == {0, 2}
    assert {entry.nsteps for entry in CASES} >= {1, 2, 5, 6, 7, 8}
    assert {(case.nproc_x, case.nproc_y) for case in cases} >= {
        (1, 1),
        (2, 1),
        (1, 2),
        (2, 2),
    }
    assert any(case.boundary and case.nproc_x == 2 for case in cases)
    assert {case.free_surface for case in cases} == {0, 1}


def test_c7a_is_capture_only_and_does_not_switch_the_fwi_path(repository_root: Path):
    implementation = (
        repository_root / "src/SH/visco_sh_material_observable.c"
    ).read_text()
    assert "grad_obj_sh" not in implementation
    assert "matcopy_SH" not in implementation
    assert "q_to_tau" not in implementation
    assert "waveconv" not in implementation
    assert "DTINV" not in implementation
    for path in (
        "src/SH/sh_visc.c",
        "src/SH/FWI_SH.c",
        "src/SH/FWI_SH_visc.c",
        "src/SH/grad_obj_sh.c",
        "src/SH/grad_obj_sh_visc.c",
        "src/SH/matcopy_SH.c",
    ):
        assert "visco_sh_material_observable" not in (
            repository_root / path
        ).read_text()


@pytest.fixture(scope="module")
def c7a_harness(tmp_path_factory, repository_root):
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler, "mpicc is required for M6.3c-7a"
    assert launcher, "mpiexec/mpirun is required for M6.3c-7a"
    executable = tmp_path_factory.mktemp("m63c7a") / "m63c7a"
    sources = [
        "tests/utilities/m63c_material_observable_harness.c",
        "src/SH/update_v_PML_SH.c",
        "src/SH/update_s_visc_PML_SH.c",
        "src/SH/exchange_v_SH.c",
        "src/SH/exchange_s_SH.c",
        "src/SH/surface_elastic_SH.c",
        "src/SH/visco_sh_gsls_vjp.c",
        "src/SH/update_s_visc_PML_SH_adjoint.c",
        "src/SH/update_v_PML_SH_adjoint.c",
        "src/SH/exchange_v_SH_adjoint.c",
        "src/SH/exchange_s_SH_adjoint.c",
        "src/SH/surface_elastic_SH_adjoint.c",
        "src/SH/visco_sh_full_state_adjoint_step.c",
        "src/SH/visco_sh_material_observable.c",
    ]
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fcommon",
        "-DM63C_MATERIAL_OBSERVABLE_TEST_COUNTERS",
        "-I",
        str(repository_root / "include"),
    ]
    command.extend(str(repository_root / source) for source in sources)
    command.extend(["-o", str(executable), "-lm"])
    result = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    return launcher, executable


def _run(c7a_harness, tmp_path, observable_case, *, dtinv=1, overlap=False):
    launcher, executable = c7a_harness
    case = observable_case.case
    output = tmp_path / (observable_case.name + f"_d{dtinv}" + ("_overlap" if overlap else ""))
    output.mkdir()
    nx = 4 if overlap else case.nx
    command = [
        launcher,
        "--oversubscribe",
        "-n",
        str(case.ranks),
        str(executable),
        str(case.nproc_x),
        str(case.nproc_y),
        str(case.boundary),
        str(case.free_surface),
        str(case.fdorder),
        str(case.mechanisms),
        "3" if overlap else str(case.fw),
        str(nx),
        str(case.ny),
        str(observable_case.nsteps),
        str(dtinv),
        str(output),
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(records) == 1, result.stdout
    return records[0], output


def test_c7a_rejects_dtinv_greater_than_one(c7a_harness, tmp_path):
    record, _ = _run(c7a_harness, tmp_path, CASES[0], dtinv=2)
    assert record == {"precondition_status": -1}


def test_c7a_rejects_locked_same_axis_cpml_overlap(c7a_harness, tmp_path):
    record, _ = _run(c7a_harness, tmp_path, CASES[0], overlap=True)
    assert record == {"precondition_status": -2}


@pytest.mark.parametrize("observable_case", CASES, ids=lambda entry: entry.name)
def test_actual_forward_observable_trajectory_matches_independent_reference(
    c7a_harness, tmp_path, observable_case
):
    case = observable_case.case
    _, _, expected = forward_trajectory(make_states(case), observable_case)
    record, output = _run(c7a_harness, tmp_path, observable_case)
    expected_calls = observable_case.nsteps * case.nx * case.ny
    assert record == {
        "active_qsum_calls": expected_calls,
        "active_strain_calls": expected_calls,
        "inactive_qsum_calls": 0,
        "inactive_strain_calls": 0,
        "nsteps": observable_case.nsteps,
        "passive_exact": True,
    }
    maxima = {"qsum": 0.0, "strain_x": 0.0, "strain_y": 0.0}
    cells = case.nx * case.ny
    for rank in range(case.ranks):
        payload = array("f")
        with (output / f"rank_{rank}.bin").open("rb") as stream:
            payload.fromfile(stream, observable_case.nsteps * 3 * cells)
        offset = 0
        for step in range(observable_case.nsteps):
            for channel in ("qsum", "strain_x", "strain_y"):
                actual = payload[offset : offset + cells]
                offset += cells
                error = _relative_l2(actual, expected[rank][step][channel])
                maxima[channel] = max(maxima[channel], error)
    print(
        "M63C7A "
        + json.dumps(
            {
                "case": observable_case.name,
                "channel_relative_max": maxima,
                "passive_exact": record["passive_exact"],
            },
            sort_keys=True,
        )
    )
    assert all(value <= C7A_REFERENCE_RELATIVE_MAX for value in maxima.values())
