"""M6.3c-6b exact distributed SH material-map transpose verification."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import shutil
import struct
import subprocess

import pytest

from tests.utilities.m63c_distributed_material_map_reference import (
    C6B_DOUBLE_DOT_RELATIVE_MAX,
    C6B_FD_RELATIVE_MAX,
    C6B_MPI_DOT_RELATIVE_MAX,
    C6B_REFERENCE_RELATIVE_MAX,
    Layout,
    dot_fields,
    dot_outputs,
    f32,
    material_forward,
    material_jvp,
    material_vjp,
    matcopy_forward,
    matcopy_transpose,
    relative,
)
from tests.utilities.m63c_material_map_reference import QMapping, physical_mapping


TOPOLOGIES = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 2))
REFERENCE_CHANNELS = (
    "mu_x", "mu_y", "tau_x", "tau_y", "rhoi",
    "bar_primary", "bar_rho", "bar_q",
)


def _channel_relative_max(actual, expected):
    difference = max((abs(a - b) for a, b in zip(actual, expected)), default=0.0)
    scale = max(
        max((abs(value) for value in actual), default=0.0),
        max((abs(value) for value in expected), default=0.0),
        1.0e-300,
    )
    return difference / scale


@pytest.fixture(scope="module")
def c6b_harness(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler, "mpicc is required for M6.3c-6b"
    assert launcher, "mpiexec/mpirun is required for M6.3c-6b"
    executable = tmp_path_factory.mktemp("m63c6b") / "m63c6b"
    command = [
        compiler, "-std=c99", "-O2", "-fcommon",
        "-I", str(repository_root / "include"),
        str(repository_root / "tests/utilities/m63c_distributed_material_map_harness.c"),
        str(repository_root / "src/SH/matcopy_SH.c"),
        str(repository_root / "src/SH/matcopy_SH_adjoint.c"),
        str(repository_root / "src/SH/av_mu_SH.c"),
        str(repository_root / "src/SH/inv_rho_SH.c"),
        str(repository_root / "src/av_tau.c"),
        str(repository_root / "src/q_parameterization.c"),
        str(repository_root / "src/SH/visco_sh_material_vjp.c"),
        "-o", str(executable), "-lm",
    ]
    result = subprocess.run(command, cwd=repository_root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    return launcher, executable


def _write(directory, rank, values):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"rank_{rank}.bin").open("wb") as stream:
        stream.write(struct.pack(f"{len(values)}f", *[f32(v) for v in values]))


def _run(harness, directory, npx, npy, layout, mode="raw", invmat1=1, qmode=0):
    launcher, executable = harness
    command = [launcher, "--oversubscribe", "-n", str(npx*npy),
               str(executable), str(npx), str(npy), str(layout.nx),
               str(layout.ny), mode, str(invmat1), str(qmode), str(directory)]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=45)
    assert result.returncode == 0, result.stdout
    records = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith("{")]
    assert len(records) == 1, result.stdout
    return records[0]


def _raw_case(layout, npx, npy, selected="all", seed=6300):
    rng = random.Random(seed + 17*npx + npy)
    ranks = npx*npy
    inputs, bars = [], []
    for rank in range(ranks):
        rank_inputs, rank_bars = [], []
        for field in range(3):
            active = selected == "all" or selected == ("rho", "u", "tau")[field]
            rank_inputs.append([f32(rng.uniform(-2.0, 3.0)) if active else 0.0
                                for _ in range(layout.cells)])
            rank_bars.append([f32(rng.uniform(-1.5, 2.5)) if active else 0.0
                              for _ in range(layout.cells)])
        inputs.append(rank_inputs); bars.append(rank_bars)
    forward = [[None]*3 for _ in range(ranks)]
    transpose = [[None]*3 for _ in range(ranks)]
    for field in range(3):
        values = [inputs[rank][field] for rank in range(ranks)]
        dual = [bars[rank][field] for rank in range(ranks)]
        fwd = matcopy_forward(values, layout, npx, npy)
        adj = matcopy_transpose(dual, layout, npx, npy)
        for rank in range(ranks): forward[rank][field] = fwd[rank]; transpose[rank][field] = adj[rank]
    return inputs, bars, forward, transpose


def _write_raw(directory, case):
    inputs, bars, forward, transpose = case
    for rank in range(len(inputs)):
        _write(directory, rank, sum(inputs[rank] + bars[rank] + forward[rank] + transpose[rank], []))


def _material_fields(layout, npx, npy, invmat1, seed):
    rng = random.Random(seed)
    ranks = npx*npy
    primary, rho_values, q_values = [], [], []
    dprimary, drho, dq = [], [], []
    bars = []
    for rank in range(ranks):
        p, r, q = [0.0]*layout.cells, [0.0]*layout.cells, [0.0]*layout.cells
        dp, dr, qd = [0.0]*layout.cells, [0.0]*layout.cells, [0.0]*layout.cells
        for j in range(layout.ny + 2):
            for i in range(layout.nx + 2):
                k = layout.index(j, i)
                p[k] = (6.0e9 + 2.0e9*rng.random()) if invmat1 == 3 else (1450.0 + 900.0*rng.random())
                r[k] = 1900.0 + 650.0*rng.random()
                q[k] = 20.0 + 130.0*rng.random()
                if 1 <= i <= layout.nx and 1 <= j <= layout.ny:
                    dp[k] = p[k] * rng.uniform(-0.015, 0.015)
                    dr[k] = r[k] * rng.uniform(-0.015, 0.015)
                    qd[k] = q[k] * rng.uniform(-0.015, 0.015)
        primary.append(p); rho_values.append(r); q_values.append(q)
        dprimary.append(dp); drho.append(dr); dq.append(qd)
        bars.append([
            [rng.uniform(-1.0, 1.0) * 1.0e-9 for _ in range(layout.owned)],
            [rng.uniform(-1.0, 1.0) * 1.0e-9 for _ in range(layout.owned)],
            [rng.uniform(-1.0, 1.0) for _ in range(layout.owned)],
            [rng.uniform(-1.0, 1.0) for _ in range(layout.owned)],
            [rng.uniform(-1.0, 1.0) * 1.0e6 for _ in range(layout.owned)],
        ])
    return primary, rho_values, q_values, dprimary, drho, dq, bars


def _write_map(directory, fields, mapping, invmat1, layout, npx, npy):
    primary, rho_values, q_values, dp, dr, dq, bars = fields
    output = material_forward(invmat1, mapping, primary, rho_values, q_values, layout, npx, npy)
    jvp = material_jvp(invmat1, mapping, primary, rho_values, q_values,
                       dp, dr, dq, layout, npx, npy)
    bp, br, bq = material_vjp(invmat1, mapping, primary, rho_values, q_values,
                              bars, layout, npx, npy)
    for rank in range(npx*npy):
        payload = (primary[rank] + rho_values[rank] + q_values[rank]
                   + dp[rank] + dr[rank] + dq[rank]
                   + bp[rank] + br[rank] + bq[rank]
                   + sum(bars[rank], []) + sum(output[rank], []) + sum(jvp[rank], []))
        _write(directory, rank, payload)
    return output, jvp, (bp, br, bq)


def test_independent_raw_matcopy_double_dot_all_topologies():
    layout = Layout()
    diagnostics = []
    for npx, npy in TOPOLOGIES:
        inputs, bars, forward, transpose = _raw_case(layout, npx, npy)
        for field in range(3):
            lhs = dot_fields([forward[r][field] for r in range(npx*npy)],
                             [bars[r][field] for r in range(npx*npy)])
            rhs = dot_fields([inputs[r][field] for r in range(npx*npy)],
                             [transpose[r][field] for r in range(npx*npy)])
            residual = relative(lhs, rhs)
            assert residual <= C6B_DOUBLE_DOT_RELATIVE_MAX
            diagnostics.append({"topology": f"{npx}x{npy}", "field": field,
                                "dot_residual": residual})
    print("M63C6B_REFERENCE_RAW " + json.dumps(diagnostics, sort_keys=True))


def test_actual_raw_matcopy_transpose_fields_and_topologies(c6b_harness, tmp_path):
    layout = Layout(); records = []
    for npx, npy in TOPOLOGIES:
        for selected in ("rho", "u", "tau", "all"):
            directory = tmp_path / f"raw_{npx}x{npy}_{selected}"
            _write_raw(directory, _raw_case(layout, npx, npy, selected))
            record = _run(c6b_harness, directory, npx, npy, layout)
            assert record["dot_residual"] <= C6B_MPI_DOT_RELATIVE_MAX
            assert record["reference_error"] <= C6B_REFERENCE_RELATIVE_MAX
            records.append({**record, "topology": f"{npx}x{npy}", "field": selected})
    print("M63C6B_PRODUCTION_RAW " + json.dumps(records, sort_keys=True))


def test_explicit_diagonal_corner_provenance(c6b_harness, tmp_path):
    layout, npx, npy = Layout(), 2, 2
    ranks = npx*npy
    inputs = [[[0.0]*layout.cells for _ in range(3)] for _ in range(ranks)]
    bars = [[[0.0]*layout.cells for _ in range(3)] for _ in range(ranks)]
    source_rank, receiving_rank = 3, 0
    source_cell = layout.index(1, 1)
    corner = layout.index(layout.ny + 1, layout.nx + 1)
    inputs[source_rank][2][source_cell] = 7.25
    forward_field = matcopy_forward([inputs[r][2] for r in range(ranks)], layout, npx, npy)
    assert forward_field[receiving_rank][corner] == 7.25
    bars[receiving_rank][2][corner] = -2.5
    transpose_field = matcopy_transpose([bars[r][2] for r in range(ranks)], layout, npx, npy)
    nonzero = [(r, k, v) for r, values in enumerate(transpose_field)
               for k, v in enumerate(values) if v != 0.0]
    assert nonzero == [(source_rank, source_cell, -2.5)]
    forward = [[[0.0]*layout.cells for _ in range(3)] for _ in range(ranks)]
    transpose = [[[0.0]*layout.cells for _ in range(3)] for _ in range(ranks)]
    for r in range(ranks): forward[r][2] = forward_field[r]; transpose[r][2] = transpose_field[r]
    directory = tmp_path / "corner"
    _write_raw(directory, (inputs, bars, forward, transpose))
    record = _run(c6b_harness, directory, npx, npy, layout)
    assert record["dot_residual"] <= C6B_MPI_DOT_RELATIVE_MAX
    assert record["reference_error"] <= C6B_REFERENCE_RELATIVE_MAX
    print("M63C6B_CORNER " + json.dumps(record, sort_keys=True))


def test_channel_metric_rejects_low_amplitude_corruption():
    modulus_actual, modulus_expected = [1.0e10], [1.0e10]
    bar_q_actual, bar_q_expected = [2.0e-7], [1.0e-7]
    old_shared_scale_error = abs(bar_q_actual[0] - bar_q_expected[0]) / modulus_expected[0]
    channel_error = _channel_relative_max(bar_q_actual, bar_q_expected)
    assert old_shared_scale_error < C6B_REFERENCE_RELATIVE_MAX
    assert _channel_relative_max(modulus_actual, modulus_expected) == 0.0
    assert channel_error == 0.5
    assert channel_error > C6B_REFERENCE_RELATIVE_MAX


@pytest.mark.parametrize("invmat1", (1, 3))
@pytest.mark.parametrize("qmode", (0, 1))
def test_distributed_material_map_reference_fd_and_production(
        c6b_harness, tmp_path, invmat1, qmode):
    layout = Layout(); mapping = QMapping(0) if qmode == 0 else physical_mapping((3.0, 7.0, 13.0), 2.0, 18.0, 0.5)
    records = []
    for npx, npy in TOPOLOGIES:
        fields = _material_fields(layout, npx, npy, invmat1, 6600 + 100*invmat1 + 10*qmode + npx + npy)
        output, tangent, adjoint = _write_map(tmp_path / f"map_{invmat1}_{qmode}_{npx}x{npy}", fields, mapping, invmat1, layout, npx, npy)
        primary, rho_values, q_values, dp, dr, dq, bars = fields
        lhs = dot_outputs(tangent, bars)
        rhs = dot_fields(dp, adjoint[0]) + dot_fields(dr, adjoint[1]) + dot_fields(dq, adjoint[2])
        dot_error = relative(lhs, rhs)
        epsilon = 1.0e-4
        plus = material_forward(invmat1, mapping,
            [[a + epsilon*b for a, b in zip(x, dx)] for x, dx in zip(primary, dp)],
            [[a + epsilon*b for a, b in zip(x, dx)] for x, dx in zip(rho_values, dr)],
            [[a + epsilon*b for a, b in zip(x, dx)] for x, dx in zip(q_values, dq)], layout, npx, npy)
        minus = material_forward(invmat1, mapping,
            [[a - epsilon*b for a, b in zip(x, dx)] for x, dx in zip(primary, dp)],
            [[a - epsilon*b for a, b in zip(x, dx)] for x, dx in zip(rho_values, dr)],
            [[a - epsilon*b for a, b in zip(x, dx)] for x, dx in zip(q_values, dq)], layout, npx, npy)
        fd = dot_outputs([[[ (a-b)/(2*epsilon) for a,b in zip(pf,mf)] for pf,mf in zip(pr,mr)] for pr,mr in zip(plus,minus)], bars)
        fd_error = relative(lhs, fd)
        assert dot_error <= C6B_DOUBLE_DOT_RELATIVE_MAX
        assert fd_error <= C6B_FD_RELATIVE_MAX
        record = _run(c6b_harness, tmp_path / f"map_{invmat1}_{qmode}_{npx}x{npy}", npx, npy, layout, "map", invmat1, qmode)
        assert record["dot_residual"] <= C6B_MPI_DOT_RELATIVE_MAX
        assert record["reference_error"] <= C6B_REFERENCE_RELATIVE_MAX
        assert tuple(record["reference_errors"]) == REFERENCE_CHANNELS
        assert record["reference_error"] == max(record["reference_errors"].values())
        assert all(value <= C6B_REFERENCE_RELATIVE_MAX
                   for value in record["reference_errors"].values())
        records.append({**record, "topology": f"{npx}x{npy}", "fd_error": fd_error, "reference_dot": dot_error})
    print("M63C6B_MAP " + json.dumps(records, sort_keys=True))


def test_scope_and_reverse_order_contract(repository_root: Path):
    source = (repository_root / "src/SH/matcopy_SH_adjoint.c").read_text()
    makefile = (repository_root / "src/Makefile").read_text()
    assert source.index("/* H^T") < source.index("/* V^T")
    assert "INDEX[2]" in source and "INDEX[4]" in source
    assert "MPI_COMM_WORLD" in source and "+=" in source and "clear_cell" in source
    assert "matcopy_SH_adjoint.c" in makefile
    for path in ("src/SH/matcopy_SH.c", "src/SH/sh_visc.c", "src/SH/FWI_SH.c", "src/SH/FWI_SH_visc.c", "src/SH/grad_obj_sh.c", "src/SH/grad_obj_sh_visc.c"):
        assert "matcopy_SH_adjoint" not in (repository_root / path).read_text()
