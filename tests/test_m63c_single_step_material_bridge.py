"""M6.3c-7c-b1 real single-step trajectory/cotangent bridge verification."""

from __future__ import annotations

from array import array
import ctypes
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.test_m63c_material_timestep_vjp import c7b, _production
from tests.utilities.m63c_single_step_material_bridge_reference import (
    BRIDGE_REFERENCE_RELATIVE_MAX,
    independent_expected,
    relative_l2,
    split_cases,
)


CASES = (
    ("interior_1x1", 1, 1, 0, 0, 2, 1, 0),
    ("mpi_2x1", 2, 1, 0, 0, 4, 2, 0),
    ("mpi_1x2", 1, 2, 0, 0, 6, 3, 0),
    ("mpi_2x2", 2, 2, 0, 0, 4, 2, 0),
    ("free_surface_1x1", 1, 1, 0, 1, 4, 2, 2),
)


@pytest.fixture(scope="module")
def bridge_harness(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher
    executable = tmp_path_factory.mktemp("m63c7cb1") / "m63c7cb1"
    sources = [
        "tests/utilities/m63c_single_step_material_bridge_harness.c",
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
    ]
    command = [compiler, "-std=c99", "-O2", "-fcommon", "-I", str(repository_root / "include")]
    command.extend(str(repository_root / source) for source in sources)
    command.extend(["-o", str(executable), "-lm"])
    result = subprocess.run(command, cwd=repository_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    return launcher, executable


def _run(bridge_harness, tmp_path, case, sentinel=0):
    launcher, executable = bridge_harness
    name, npx, npy, boundary, free_surface, fdorder, mechanisms, fw = case
    directory = tmp_path / f"{name}_{sentinel}"
    directory.mkdir()
    command = [
        launcher, "--oversubscribe", "-n", str(npx * npy), str(executable),
        str(npx), str(npy), str(boundary), str(free_surface), str(fdorder),
        str(mechanisms), str(fw), "10", "12", str(sentinel), str(directory),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, result.stdout
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(records) == 1, result.stdout
    outputs = []
    for rank in range(npx * npy):
        payload = array("d")
        with (directory / f"rank_{rank}.bin").open("rb") as stream:
            payload.fromfile(stream, 5 * 10 * 12)
        outputs.append(tuple(payload))
    return records[0], outputs


@pytest.mark.parametrize("case", CASES, ids=lambda value: value[0])
def test_material_capture_is_passive_finite_and_nonzero(bridge_harness, tmp_path, case):
    record, outputs = _run(bridge_harness, tmp_path, case)
    print("M63C7CB1_RUNTIME " + json.dumps({"case": case[0], **record}, sort_keys=True))
    assert record["passivity_max"] == 0.0
    assert record["reference_error"] == 0.0
    if case[0] != "interior_1x1":
        assert record["wrong_stress_hook_distance"] > 1.0e-12
        assert record["wrong_velocity_hook_distance"] > 1.0e-12
    assert record["wiring_mutant_distance"] > 1.0e-12
    assert record["input_change"] == 0.0
    assert record["nonzero"] > 0.0
    assert all(math.isfinite(value) for rank in outputs for value in rank)
    cells = 120
    for channel in range(5):
        norm = math.fsum(
            value * value
            for rank in outputs
            for value in rank[channel * cells : (channel + 1) * cells]
        )
        assert norm > 0.0, (case[0], channel)


def test_explicit_observable_step_controls_the_result_without_implicit_time_shift(
    bridge_harness, tmp_path
):
    case = CASES[0]
    base_record, base = _run(bridge_harness, tmp_path, case, sentinel=0)
    shifted_record, shifted = _run(bridge_harness, tmp_path, case, sentinel=1)
    assert base_record["passivity_max"] == shifted_record["passivity_max"] == 0.0
    assert base_record["reference_error"] == shifted_record["reference_error"] == 0.0
    assert base != shifted


def test_split_c7b_is_linear_and_matches_independent_five_point_reference(c7b):
    diagnostics = []
    for case in split_cases():
        zero = (0.0,) * case.mechanisms
        stress = _production(c7b, type(case)(
            case.name, case.dt, case.dh, case.qsum, case.strain_x, case.strain_y,
            0.0, case.bar_sx, case.bar_sy, case.bar_rx, case.bar_qy,
            case.memory_x, case.memory_y, case.frequencies, case.rhoi,
            case.mu_x, case.mu_y, case.tau_x, case.tau_y,
        ))
        density = _production(c7b, type(case)(
            case.name, case.dt, case.dh, case.qsum, case.strain_x, case.strain_y,
            case.bar_v, 0.0, 0.0, zero, zero, case.memory_x, case.memory_y,
            case.frequencies, case.rhoi, case.mu_x, case.mu_y,
            case.tau_x, case.tau_y,
        ))
        split = tuple(left + right for left, right in zip(stress, density))
        one_shot = _production(c7b, case)
        expected = independent_expected(case)
        assert split == pytest.approx(one_shot, rel=3.0e-15, abs=1.0e-18)
        error = relative_l2(one_shot, expected)
        diagnostics.append((case.name, error))
        assert error <= BRIDGE_REFERENCE_RELATIVE_MAX
    print("M63C7CB1_SPLIT " + json.dumps(diagnostics))


def test_hook_order_and_scope_are_locked_in_the_shared_reverse_step(repository_root: Path):
    source = (repository_root / "src/SH/visco_sh_full_state_adjoint_step.c").read_text()
    stress_exchange = source.index("status = exchange_s_SH_adjoint(")
    stress_surface = source.index("surface_elastic_SH_stress_adjoint(", stress_exchange)
    stress_capture = source.index("capture_material_stress_cotangents(", stress_surface)
    reverse_stress = source.index("status = reverse_stress_block(", stress_capture)
    velocity_surface = source.index("surface_elastic_SH_velocity_adjoint(", reverse_stress)
    velocity_exchange = source.index("status = exchange_v_SH_adjoint(", velocity_surface)
    source_transpose = source.index("status = source_transpose(", velocity_exchange)
    velocity_capture = source.index("capture_material_velocity_cotangent(", source_transpose)
    reverse_velocity = source.index("return reverse_velocity_block(", velocity_capture)
    assert stress_exchange < stress_surface < stress_capture < reverse_stress
    assert reverse_stress < velocity_surface < velocity_exchange < source_transpose
    assert source_transpose < velocity_capture < reverse_velocity
    assert source.count("reverse_stress_block(cfg") == 1
    assert source.count("reverse_velocity_block(cfg") == 1
    for forbidden in ("DTINV", "matcopy_SH_adjoint", "q_tau_derivative", "grad_obj_sh"):
        assert forbidden not in source


def test_runtime_reference_assembles_locked_c7b_without_capture_helpers(repository_root: Path):
    source = (
        repository_root
        / "tests/utilities/m63c_single_step_material_bridge_harness.c"
    ).read_text()
    reference = source[
        source.index("static int explicit_hook_reference(") : source.index("\nint main(")
    ]
    assert "independent_one_shot_c7b(" in reference
    assert "visco_sh_material_timestep_vjp(" in source[
        source.index("static int independent_one_shot_c7b(") :
        source.index("static int explicit_hook_reference(")
    ]
    assert "capture_material_stress_cotangents(" not in reference
    assert "capture_material_velocity_cotangent(" not in reference


def test_locked_production_files_outside_the_bridge_remain_untouched(repository_root: Path):
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", "src/SH/visco_sh_reverse_time_adjoint.c",
         "src/SH/visco_sh_material_timestep_vjp.c",
         "src/SH/visco_sh_material_observable.c", "src/SH/matcopy_SH_adjoint.c",
         "src/SH/FWI_SH.c", "src/SH/FWI_SH_visc.c", "src/SH/grad_obj_sh.c",
         "src/SH/grad_obj_sh_visc.c"],
        cwd=repository_root, text=True, stdout=subprocess.PIPE, check=True,
    )
    assert result.stdout == ""
