"""M6.3c-5b multi-step fixed-material viscoelastic SH transpose tests."""

from __future__ import annotations

from array import array
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utilities.m63c_reverse_time_adjoint_reference import (
    C5B_DOUBLE_DOT_RELATIVE_MAX,
    C5B_GLOBAL_DOT_RELATIVE_MAX,
    C5B_N1_C5A_RELATIVE_MAX,
    C5B_N2_COMPOSITION_RELATIVE_MAX,
    C5B_REFERENCE_RELATIVE_MAX,
    CASES,
    MultiStepCase,
    forward_multi,
    multi_step_dot,
    receiver_dual_series,
    signal_series,
    terminal_dual,
    transpose_multi,
)
from tests.utilities.m63c_full_state_step_reference import make_states


def _flatten_state(state):
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
    for key, count in (
        ("psi_sxz_x", case.ny * 2 * case.fw),
        ("psi_syz_y", case.nx * 2 * case.fw),
        ("psi_vzx", case.ny * 2 * case.fw),
        ("psi_vzy", case.nx * 2 * case.fw),
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
            [value for mechanism in actual[key] for value in mechanism],
            [value for mechanism in expected[key] for value in mechanism],
        )
    return result


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_independent_multi_step_reference_closes(case):
    lhs, rhs, residual = multi_step_dot(case, rounded=False)
    print(
        "M63C5B_DOUBLE "
        + json.dumps(
            {"case": case.name, "lhs": lhs, "rhs": rhs, "dot_residual": residual},
            sort_keys=True,
        )
    )
    assert residual <= C5B_DOUBLE_DOT_RELATIVE_MAX


def test_reference_matrix_freezes_all_predeclared_dimensions():
    assert {case.nsteps for case in CASES} >= {1, 2, 5, 8, 12}
    assert {case.base.fdorder for case in CASES} == {2, 4, 6, 8, 10, 12}
    assert {case.base.mechanisms for case in CASES} == {1, 3}
    assert {case.base.fw for case in CASES} == {0, 2}
    assert {(case.base.nproc_x, case.base.nproc_y) for case in CASES} >= {
        (1, 1), (1, 2), (2, 1), (2, 2)
    }
    assert {case.base.free_surface for case in CASES} == {0, 1}
    assert any(case.base.boundary for case in CASES)


def test_reference_n1_and_n2_are_explicit_step_transpose_compositions():
    for case in CASES[:2]:
        bars = terminal_dual(case)
        receivers = receiver_dual_series(case)
        actual_state, actual_signal = transpose_multi(bars, receivers, case)
        current = bars
        expected_signal = [[0.0] * case.base.ranks for _ in range(case.nsteps)]
        from tests.utilities.m63c_full_state_step_reference import transpose
        for n in range(case.nsteps - 1, -1, -1):
            current, expected_signal[n] = transpose(current, receivers[n], case.base)
        assert actual_state == current
        assert actual_signal == expected_signal


def test_driver_contract_is_chronological_constant_state_memory_and_inactive(
    repository_root: Path,
):
    source = (repository_root / "src/SH/visco_sh_reverse_time_adjoint.c").read_text()
    header = (repository_root / "include/fd.h").read_text()
    assert "for (n = nsteps - 1; n >= 0; --n)" in source
    assert "bar_receiver_series + (size_t)n * base_config->nrec" in source
    assert "bar_signal_series + (size_t)n * base_config->nsrc" in source
    assert source.count("visco_sh_full_state_adjoint_step(") == 1
    assert "previous = bar_initial" in source
    assert "bar_terminal_work" in source and "scratch" in source
    assert "malloc(" not in source and "calloc(" not in source
    assert "static struct visco_sh_full_state" not in source
    assert "time-major" in header
    for active in ("src/SH/sh_visc.c", "src/SH/FWI_SH.c", "src/SH/FWI_SH_visc.c"):
        assert "visco_sh_reverse_time_adjoint" not in (repository_root / active).read_text()


def test_driver_owns_source_output_initialization_after_nonconsuming_preflight(
    repository_root: Path,
):
    source = (repository_root / "src/SH/visco_sh_reverse_time_adjoint.c").read_text()
    overlap = source.index("if (unsupported_cpml_overlap(base_config)) return -2;")
    zeroing = source.index("for (n = 0; n < nsteps; ++n)")
    reverse = source.index("for (n = nsteps - 1; n >= 0; --n)")
    assert overlap < zeroing < reverse
    assert "for (source = 0; source < base_config->nsrc; ++source)" in source
    assert (
        "bar_signal_series[(size_t)n * base_config->nsrc + source] = 0.0;"
        in source
    )


@pytest.fixture(scope="module")
def c5b_harness(tmp_path_factory, repository_root):
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler, "mpicc is required for M6.3c-5b"
    assert launcher, "mpiexec/mpirun is required for M6.3c-5b"
    executable = tmp_path_factory.mktemp("m63c5b") / "m63c5b"
    sources = [
        "tests/utilities/m63c_reverse_time_adjoint_harness.c",
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
        "src/SH/visco_sh_reverse_time_adjoint.c",
    ]
    command = [
        compiler, "-std=c99", "-O2", "-fcommon",
        "-I", str(repository_root / "include"),
        "-I", str(repository_root / "tests/utilities"),
    ]
    command.extend(str(repository_root / source) for source in sources)
    command.extend(["-o", str(executable), "-lm"])
    result = subprocess.run(
        command, cwd=repository_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    assert result.returncode == 0, result.stdout
    return launcher, executable


def _run(c5b_harness, tmp_path, case, mode="full", impulse=-1):
    launcher, executable = c5b_harness
    output = tmp_path / f"{case.name}_{mode}_{impulse}"
    output.mkdir()
    base = case.base
    command = [
        launcher, "--oversubscribe", "-n", str(base.ranks), str(executable),
        str(base.nproc_x), str(base.nproc_y), str(base.boundary),
        str(base.free_surface), str(base.fdorder), str(base.mechanisms),
        str(base.fw), str(base.nx), str(base.ny), str(case.nsteps),
        mode, str(impulse), str(output),
    ]
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(records) == 1, result.stdout
    return records[0], output


def _expected(case, mode="full", impulse=-1):
    initial = make_states(case.base, dual=False)
    signals = signal_series(case)
    final, receivers = forward_multi(initial, signals, case, rounded=True)
    bars = terminal_dual(case, mode)
    bar_receivers = receiver_dual_series(case, mode, impulse)
    bar_initial, bar_signals = transpose_multi(
        bars, bar_receivers, case, rounded=True
    )
    return final, receivers, bar_initial, bar_signals


def _compare_payload(directory, case, expected):
    final, receivers, bar_initial, bar_signals = expected
    maxima = []
    diagnostics = []
    state_values = len(_flatten_state(final[0]))
    for rank in range(case.base.ranks):
        payload = array("f")
        with (directory / f"rank_{rank}.bin").open("rb") as stream:
            payload.fromfile(stream, 2 * state_values + 2 * case.nsteps)
        actual_final, offset = _read_state(payload, 0, case.base)
        actual_initial, offset = _read_state(payload, offset, case.base)
        actual_receivers = list(payload[offset : offset + case.nsteps])
        offset += case.nsteps
        actual_signals = list(payload[offset : offset + case.nsteps])
        forward_errors = _group_errors(actual_final, final[rank])
        adjoint_errors = _group_errors(actual_initial, bar_initial[rank])
        receiver_error = _relative_l2(
            actual_receivers, [row[rank] for row in receivers]
        )
        source_error = _relative_l2(
            actual_signals, [row[rank] for row in bar_signals]
        )
        maxima.extend(forward_errors.values())
        maxima.extend(adjoint_errors.values())
        maxima.extend((receiver_error, source_error))
        diagnostics.append(
            {"rank": rank, "forward": forward_errors, "adjoint": adjoint_errors,
             "receiver": receiver_error, "source": source_error}
        )
    return max(maxima), diagnostics


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_actual_multi_step_forward_and_reverse_match_reference(
    c5b_harness, tmp_path, case
):
    record, directory = _run(c5b_harness, tmp_path, case)
    maximum, diagnostics = _compare_payload(directory, case, _expected(case))
    print(
        "M63C5B_PRODUCTION "
        + json.dumps(
            {"case": case.name, "dot_residual": record["dot_residual"],
             "component_relative_max": maximum,
             "structural_relative": record["structural_relative"],
             "diagnostics": diagnostics},
            sort_keys=True,
        )
    )
    assert maximum <= C5B_REFERENCE_RELATIVE_MAX
    assert record["dot_residual"] <= C5B_GLOBAL_DOT_RELATIVE_MAX
    if case.nsteps == 1:
        assert record["structural_relative"] <= C5B_N1_C5A_RELATIVE_MAX
    if case.nsteps == 2:
        assert record["structural_relative"] <= C5B_N2_COMPOSITION_RELATIVE_MAX


@pytest.mark.parametrize(
    "mode,impulse",
    (("impulse", 0), ("impulse", 2), ("impulse", 4),
     ("terminal", -1), ("receiver", -1)),
)
def test_temporal_indexing_terminal_and_receiver_decompositions(
    c5b_harness, tmp_path, mode, impulse
):
    case = CASES[2]
    record, directory = _run(c5b_harness, tmp_path, case, mode, impulse)
    maximum, diagnostics = _compare_payload(
        directory, case, _expected(case, mode, impulse)
    )
    print(
        "M63C5B_TEMPORAL "
        + json.dumps(
            {"mode": mode, "impulse": impulse, "maximum": maximum,
             "dot_residual": record["dot_residual"], "diagnostics": diagnostics},
            sort_keys=True,
        )
    )
    assert maximum <= C5B_REFERENCE_RELATIVE_MAX
    assert record["dot_residual"] <= C5B_GLOBAL_DOT_RELATIVE_MAX


def test_driver_rejects_overlap_before_consuming_terminal_state(
    c5b_harness, tmp_path
):
    from tests.utilities.m63c_full_state_step_reference import Case
    case = MultiStepCase(Case("overlap", 2, 1, 3, 0, 1, 1, nx=4), 5)
    record, _ = _run(c5b_harness, tmp_path, case, "precondition")
    assert record == {
        "precondition_status": -2,
        "source_modified": 0,
        "state_consumed": 0,
    }
