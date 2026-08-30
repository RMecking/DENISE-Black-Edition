"""M6.3c-5a complete one-timestep viscoelastic SH transpose verification."""

from __future__ import annotations

import math
from pathlib import Path
from array import array
import json
import shutil
import subprocess

import pytest

from tests.utilities.m63c_full_state_step_reference import (
    C5A_DOUBLE_REFERENCE_DOT_MAX,
    C5A_GLOBAL_DOT_RELATIVE_MAX,
    C5A_REFERENCE_RELATIVE_MAX,
    CASES,
    Case,
    forward,
    make_states,
    relative_dot,
    state_dot,
    transpose,
)


def _signal(case):
    return [0.021 + 0.003 * rank for rank in range(case.ranks)]


def _receiver_dual(case):
    return [-0.17 + 0.019 * rank for rank in range(case.ranks)]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_independent_double_full_state_reference_closes(case):
    state = make_states(case, dual=False)
    dual = make_states(case, dual=True)
    signal = _signal(case)
    bar_receiver = _receiver_dual(case)
    output, receiver = forward(state, signal, case, rounded=False)
    bar_state, bar_signal = transpose(dual, bar_receiver, case, rounded=False)
    lhs = state_dot(output, dual) + math.fsum(
        value * bar for value, bar in zip(receiver, bar_receiver)
    )
    rhs = state_dot(state, bar_state) + math.fsum(
        value * bar for value, bar in zip(signal, bar_signal)
    )
    residual = relative_dot(lhs, rhs)
    print("M63C5A_DOUBLE " + json.dumps({"case": case.name, "dot_residual": residual}, sort_keys=True))
    assert residual <= C5A_DOUBLE_REFERENCE_DOT_MAX


def test_reference_covers_the_predeclared_c5a_matrix():
    assert {case.fdorder for case in CASES} == {2, 4, 6, 8, 10, 12}
    assert {case.mechanisms for case in CASES} == {1, 3}
    assert {case.fw for case in CASES} == {0, 2}
    assert {(case.free_surface, case.nproc_x, case.nproc_y) for case in CASES} >= {
        (0, 1, 1),
        (0, 1, 2),
        (0, 2, 1),
        (1, 1, 1),
        (0, 2, 2),
    }
    assert any(case.boundary and case.nproc_x == 2 for case in CASES)


def test_same_axis_cpml_overlap_is_explicitly_outside_positive_cases():
    for case in CASES:
        assert case.fw == 0 or case.nproc_x != 1 or case.boundary or case.nx > 2 * case.fw
        assert case.fw == 0 or case.nproc_y != 1 or case.free_surface or case.ny > 2 * case.fw


def test_c5a_production_integration_does_not_switch_the_legacy_fwi_path(
    repository_root: Path,
):
    source = (repository_root / "src/SH/visco_sh_full_state_adjoint_step.c").read_text()
    assert "visco_sh_full_state_adjoint_step" in source
    assert "sh_visc(" not in source
    assert "update_s_visc_PML_SH_adjoint_point" in source
    assert "update_v_PML_SH_adjoint_point" in source
    assert "exchange_s_SH_adjoint" in source
    assert "exchange_v_SH_adjoint" in source
    assert "surface_elastic_SH_stress_adjoint" in source
    assert "surface_elastic_SH_velocity_adjoint" in source


def test_diagnostic_side_outputs_are_not_propagation_state():
    state_keys = set(make_states(CASES[0])[0])
    assert state_keys == {
        "vz",
        "sxz",
        "syz",
        "r",
        "q",
        "psi_sxz_x",
        "psi_syz_y",
        "psi_vzx",
        "psi_vzy",
    }
    assert state_keys.isdisjoint({"vzp1", "vzm1", "utty", "uz", "uzx", "pp"})


def _flatten_state(state, case):
    values = []
    for key in ("vz", "sxz", "syz"):
        values.extend(state[key])
    for key in ("r", "q"):
        for mechanism in state[key]:
            values.extend(mechanism)
    for key in ("psi_sxz_x", "psi_syz_y", "psi_vzx", "psi_vzy"):
        values.extend(state[key])
    return values


def _read_state(values, offset, case):
    cells = case.layout.cells
    state = {}
    for key in ("vz", "sxz", "syz"):
        state[key] = list(values[offset : offset + cells])
        offset += cells
    for key in ("r", "q"):
        state[key] = []
        for _ in range(case.mechanisms):
            state[key].append(list(values[offset : offset + cells]))
            offset += cells
    xcells = case.ny * 2 * case.fw
    ycells = case.nx * 2 * case.fw
    for key, count in (
        ("psi_sxz_x", xcells),
        ("psi_syz_y", ycells),
        ("psi_vzx", xcells),
        ("psi_vzy", ycells),
    ):
        state[key] = list(values[offset : offset + count])
        offset += count
    return state, offset


def _relative_l2(actual, expected):
    numerator = math.fsum((a - b) ** 2 for a, b in zip(actual, expected))
    denominator = math.fsum(b * b for b in expected)
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def _group_errors(actual, expected):
    result = {}
    for key in ("vz", "sxz", "syz", "psi_sxz_x", "psi_syz_y", "psi_vzx", "psi_vzy"):
        result[key] = _relative_l2(actual[key], expected[key])
    for key in ("r", "q"):
        result[key] = _relative_l2(
            [x for mechanism in actual[key] for x in mechanism],
            [x for mechanism in expected[key] for x in mechanism],
        )
    return result


@pytest.fixture(scope="module")
def c5a_harness(tmp_path_factory, repository_root):
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler, "mpicc is required for M6.3c-5a"
    assert launcher, "mpiexec/mpirun is required for M6.3c-5a"
    executable = tmp_path_factory.mktemp("m63c5a") / "m63c5a"
    sources = [
        "tests/utilities/m63c_full_state_step_harness.c",
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
    ]
    command = [compiler, "-std=c99", "-O2", "-fcommon", "-I", str(repository_root / "include")]
    command.extend(str(repository_root / source) for source in sources)
    command.extend(["-o", str(executable), "-lm"])
    result = subprocess.run(command, cwd=repository_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    return launcher, executable


def _run_compiled_case(c5a_harness, tmp_path, case):
    launcher, executable = c5a_harness
    output = tmp_path / case.name
    output.mkdir()
    command = [
        launcher, "--oversubscribe", "-n", str(case.ranks), str(executable),
        str(case.nproc_x), str(case.nproc_y), str(case.boundary),
        str(case.free_surface), str(case.fdorder), str(case.mechanisms),
        str(case.fw), str(case.nx), str(case.ny), str(output),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
    assert result.returncode == 0, result.stdout
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(records) == 1, result.stdout
    return records[0], output


def test_compiled_step_rejects_same_axis_cpml_overlap(c5a_harness, tmp_path):
    launcher, executable = c5a_harness
    output = tmp_path / "overlap"
    output.mkdir()
    command = [
        launcher,
        "--oversubscribe",
        "-n",
        "1",
        str(executable),
        "1",
        "1",
        "0",
        "0",
        "2",
        "1",
        "3",
        "4",
        "14",
        str(output),
        "precondition",
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    assert records == [{"precondition_status": -2}]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_actual_forward_and_full_state_transpose_match_reference(
    c5a_harness, tmp_path, case
):
    initial = make_states(case, dual=False)
    dual = make_states(case, dual=True)
    expected_forward, expected_receiver = forward(
        initial, _signal(case), case, rounded=True
    )
    expected_prev, expected_signal = transpose(
        dual, _receiver_dual(case), case, rounded=True
    )
    record, directory = _run_compiled_case(c5a_harness, tmp_path, case)
    assert record["diagnostic_change"] > 0.0
    maxima = []
    diagnostics = []
    for rank in range(case.ranks):
        payload = array("f")
        with (directory / f"rank_{rank}.bin").open("rb") as stream:
            payload.fromfile(stream, (len(_flatten_state(initial[rank], case)) * 2) + 3)
        actual_forward, offset = _read_state(payload, 0, case)
        actual_prev, offset = _read_state(payload, offset, case)
        receiver, source, diagnostic_change = payload[offset : offset + 3]
        forward_errors = _group_errors(actual_forward, expected_forward[rank])
        adjoint_errors = _group_errors(actual_prev, expected_prev[rank])
        diagnostics.append((rank, forward_errors, adjoint_errors, receiver, expected_receiver[rank], source, expected_signal[rank]))
        maxima.extend(forward_errors.values())
        maxima.extend(adjoint_errors.values())
        maxima.append(abs(receiver - expected_receiver[rank]) / max(abs(expected_receiver[rank]), 1.0e-30))
        maxima.append(abs(source - expected_signal[rank]) / max(abs(expected_signal[rank]), 1.0e-30))
        assert diagnostic_change > 0.0
    if max(maxima) > C5A_REFERENCE_RELATIVE_MAX:
        pytest.fail(json.dumps({"case": case.name, "maximum": max(maxima), "record": record, "diagnostics": diagnostics}, sort_keys=True))
    print(
        "M63C5A_PRODUCTION "
        + json.dumps(
            {
                "case": case.name,
                "dot_residual": record["dot_residual"],
                "component_relative_max": max(maxima),
                "rank_diagnostics": diagnostics,
            },
            sort_keys=True,
        )
    )
    assert record["dot_residual"] <= C5A_GLOBAL_DOT_RELATIVE_MAX, record
