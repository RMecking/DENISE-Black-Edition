"""M6.3c-7c-a temporal and distributed physical-gradient assembly tests."""

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
    Layout,
    dot_fields,
    dot_outputs,
    f32,
    material_jvp,
    relative,
)
from tests.utilities.m63c_material_gradient_assembly_reference import (
    C7CA_LINEARITY_RELATIVE_MAX,
    C7CA_MPI_REFERENCE_RELATIVE_MAX,
    C7CA_TEMPORAL_RELATIVE_MAX,
    c6_channel_order,
    distributed_gradient,
    sum_mapped_per_step,
    temporal_accumulate,
    wrong_q_chain_before_matcopy,
)
from tests.utilities.m63c_material_map_reference import QMapping, physical_mapping


TOPOLOGIES = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 2))
CHANNELS = ("rhoi", "mu_x", "mu_y", "tau_x", "tau_y")


def _mapping(qmode):
    return QMapping(0) if qmode == 0 else physical_mapping(
        (3.0, 7.0, 13.0), 2.0, 18.0, 0.5
    )


def _fields(layout, npx, npy, invmat1, seed):
    rng = random.Random(seed)
    primary, rho_values, q_values = [], [], []
    for rank in range(npx * npy):
        p, rho, q = [0.0] * layout.cells, [0.0] * layout.cells, [0.0] * layout.cells
        for j in range(layout.ny + 2):
            for i in range(layout.nx + 2):
                k = layout.index(j, i)
                p[k] = (5.5e9 + 3.5e9 * rng.random()) if invmat1 == 3 else (
                    1400.0 + 1100.0 * rng.random()
                )
                rho[k] = 1850.0 + 800.0 * rng.random()
                # Rank/position dependence deliberately discriminates seam/corner order.
                q[k] = 18.0 + 11.0 * rank + 4.0 * i + 7.0 * j + 45.0 * rng.random()
        primary.append(p); rho_values.append(rho); q_values.append(q)
    return primary, rho_values, q_values


def _series(layout, npx, npy, nsteps, selected, seed):
    rng = random.Random(seed)
    scales = (8.0e5, 1.0e-9, 1.0e-9, 0.8, 0.8)
    output = []
    for step in range(nsteps):
        ranks = []
        for rank in range(npx * npy):
            channels = []
            for channel, scale in zip(CHANNELS, scales):
                active = selected == "all" or selected == channel
                channels.append([
                    scale * rng.uniform(-1.0, 1.0) if active else 0.0
                    for _ in range(layout.owned)
                ])
            ranks.append(channels)
        output.append(ranks)
    return output


def _write(directory, rank, values):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"rank_{rank}.bin").open("wb") as stream:
        stream.write(struct.pack(f"{len(values)}f", *[f32(value) for value in values]))


def _flatten_step_rank(series, rank):
    return [value for step in series for channel in step[rank] for value in channel]


def _field_relative_l2(left, right):
    numerator = math.fsum(
        (a - b) ** 2 for lf, rf in zip(left, right) for a, b in zip(lf, rf)
    )
    denominator = math.fsum(
        value * value for field in left for value in field
    )
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def _prepare(directory, layout, npx, npy, invmat1, qmode, series, dt, dtinv):
    primary, rho_values, q_values = _fields(
        layout, npx, npy, invmat1,
        7700 + 100 * invmat1 + 10 * qmode + npx + 7 * npy,
    )
    mapping = _mapping(qmode)
    accumulated = temporal_accumulate(series, dtinv)
    gradients = distributed_gradient(
        invmat1, mapping, primary, rho_values, q_values, accumulated,
        layout, npx, npy,
    )
    wrong_q = wrong_q_chain_before_matcopy(
        mapping, q_values, accumulated, layout, npx, npy
    )
    for rank in range(npx * npy):
        native_order = [0, 1, 2, 3, 4]
        payload = (
            primary[rank] + rho_values[rank] + q_values[rank]
            + _flatten_step_rank(series, rank)
            + sum((accumulated[rank][channel] for channel in native_order), [])
            + gradients[0][rank] + gradients[1][rank] + gradients[2][rank]
            + wrong_q[rank]
        )
        _write(directory, rank, payload)
    return primary, rho_values, q_values, accumulated, gradients


@pytest.fixture(scope="module")
def c7ca_harness(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher, "mpicc and mpiexec/mpirun are required"
    executable = tmp_path_factory.mktemp("m63c7ca") / "m63c7ca"
    sources = (
        "tests/utilities/m63c_material_gradient_assembly_harness.c",
        "src/SH/visco_sh_material_gradient_assembly.c",
        "src/SH/visco_sh_material_vjp.c",
        "src/SH/matcopy_SH.c",
        "src/SH/matcopy_SH_adjoint.c",
        "src/q_parameterization.c",
    )
    command = [
        compiler, "-std=c99", "-O2", "-fcommon",
        "-I", str(repository_root / "include"),
        *(str(repository_root / source) for source in sources),
        "-o", str(executable), "-lm",
    ]
    result = subprocess.run(command, cwd=repository_root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    return launcher, executable


def _run(harness, directory, npx, npy, layout, invmat1, qmode,
         nsteps, dt, dtinv):
    launcher, executable = harness
    command = [
        launcher, "--oversubscribe", "-n", str(npx * npy), str(executable),
        str(npx), str(npy), str(layout.nx), str(layout.ny), str(invmat1),
        str(qmode), str(nsteps), repr(dt), str(dtinv), str(directory),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, result.stdout
    records = [json.loads(line) for line in result.stdout.splitlines()
               if line.startswith("{")]
    assert len(records) == 1, result.stdout
    return records[0]


@pytest.mark.parametrize("nsteps", (1, 2, 5))
def test_independent_temporal_contract_is_direct_discrete_sum(nsteps):
    dt = 0.0017
    dtinv = 1
    series = [[[[0.0] for _ in CHANNELS]] for _ in range(nsteps)]
    for step in range(nsteps):
        for channel in range(5):
            series[step][0][channel][0] = (step + 1) * (channel - 1.5)
    original = json.dumps(series)
    actual = temporal_accumulate(series, dtinv)
    assert json.dumps(series) == original
    for channel in range(5):
        expected = math.fsum(
            series[step][0][channel][0] for step in range(nsteps)
        )
        assert actual[0][channel][0] == pytest.approx(expected, rel=2.0e-15)
        if expected:
            assert actual[0][channel][0] != pytest.approx(
                dt * expected, rel=1.0e-12, abs=1.0e-15
            )
        else:
            assert actual[0][channel][0] == 0.0
    if nsteps >= 2:
        cancellation = [[[[0.0] for _ in CHANNELS]] for _ in range(nsteps)]
        cancellation[0][0][0][0] = 3.25
        cancellation[-1][0][0][0] = -3.25
        assert temporal_accumulate(cancellation, dtinv)[0][0][0] == 0.0


@pytest.mark.parametrize("dtinv", (0, 2, 4))
def test_independent_temporal_contract_rejects_unverified_dtinv(dtinv):
    series = [[[[1.0] for _ in CHANNELS]]]
    with pytest.raises(ValueError, match="only DTINV=1"):
        temporal_accumulate(series, dtinv)


@pytest.mark.parametrize("active_step", (0, 2, 4))
def test_actual_temporal_first_middle_last_impulse(
        c7ca_harness, tmp_path, active_step):
    layout, npx, npy, nsteps, dt, dtinv = Layout(), 1, 1, 5, 0.0019, 1
    series = [[[[0.0] * layout.owned for _ in CHANNELS]] for _ in range(nsteps)]
    for channel in range(5):
        for point in range(layout.owned):
            series[active_step][0][channel][point] = (channel + 1) * (point - 3.5)
    directory = tmp_path / f"impulse_{active_step}"
    _prepare(directory, layout, npx, npy, 3, 0, series, dt, dtinv)
    record = _run(c7ca_harness, directory, npx, npy, layout, 3, 0,
                  nsteps, dt, dtinv)
    assert record["temporal_error"] <= C7CA_MPI_REFERENCE_RELATIVE_MAX
    assert max(record["primary_error"], record["rho_error"],
               record["q_error"]) <= C7CA_MPI_REFERENCE_RELATIVE_MAX


@pytest.mark.parametrize("nsteps", (1, 2, 5))
def test_actual_temporal_step_counts(c7ca_harness, tmp_path, nsteps):
    layout, npx, npy, dt, dtinv = Layout(), 1, 1, 0.0014, 1
    series = _series(layout, npx, npy, nsteps, "all", 8050 + nsteps)
    directory = tmp_path / f"step_count_{nsteps}"
    _prepare(directory, layout, npx, npy, 1, 0, series, dt, dtinv)
    record = _run(c7ca_harness, directory, npx, npy, layout, 1, 0,
                  nsteps, dt, dtinv)
    assert record["temporal_error"] <= C7CA_MPI_REFERENCE_RELATIVE_MAX
    assert max(record["primary_error"], record["rho_error"],
               record["q_error"]) <= C7CA_MPI_REFERENCE_RELATIVE_MAX


def test_actual_temporal_cancellation(c7ca_harness, tmp_path):
    layout, npx, npy, nsteps, dt, dtinv = Layout(), 1, 1, 5, 0.0019, 1
    series = [[[[0.0] * layout.owned for _ in CHANNELS]] for _ in range(nsteps)]
    for channel in range(5):
        for point in range(layout.owned):
            value = (channel + 0.5) * (point - 4.25)
            series[0][0][channel][point] = value
            series[-1][0][channel][point] = -value
    directory = tmp_path / "cancellation"
    _prepare(directory, layout, npx, npy, 1, 1, series, dt, dtinv)
    record = _run(c7ca_harness, directory, npx, npy, layout, 1, 1,
                  nsteps, dt, dtinv)
    assert record["temporal_error"] == 0.0
    assert record["primary_error"] == 0.0
    assert record["rho_error"] == 0.0
    assert record["q_error"] == 0.0


def test_actual_temporal_sum_is_independent_of_driver_dt(c7ca_harness, tmp_path):
    layout, npx, npy, nsteps, dtinv = Layout(), 1, 1, 3, 1
    series = _series(layout, npx, npy, nsteps, "all", 8177)
    directory = tmp_path / "dt_independence"
    _prepare(directory, layout, npx, npy, 3, 1, series, 0.0013, dtinv)
    small = _run(c7ca_harness, directory, npx, npy, layout, 3, 1,
                 nsteps, 0.0013, dtinv)
    large = _run(c7ca_harness, directory, npx, npy, layout, 3, 1,
                 nsteps, 0.37, dtinv)
    assert small["temporal_checksum"] == large["temporal_checksum"]
    assert small["temporal_error"] <= C7CA_MPI_REFERENCE_RELATIVE_MAX
    assert large["temporal_error"] <= C7CA_MPI_REFERENCE_RELATIVE_MAX


def test_actual_temporal_sum_rejects_unverified_dtinv(c7ca_harness, tmp_path):
    layout, npx, npy, nsteps = Layout(), 1, 1, 2
    series = _series(layout, npx, npy, nsteps, "all", 8178)
    directory = tmp_path / "reject_dtinv2"
    _prepare(directory, layout, npx, npy, 3, 0, series, 0.0013, 1)
    record = _run(c7ca_harness, directory, npx, npy, layout, 3, 0,
                  nsteps, 0.0013, 2)
    assert record == {"temporal_status": -1, "driver_dt": 0.0013}


@pytest.mark.parametrize("invmat1", (1, 3))
@pytest.mark.parametrize("qmode", (0, 1))
def test_actual_c7ca_all_channels_all_mpi_topologies(
        c7ca_harness, tmp_path, invmat1, qmode):
    layout, dt, dtinv, nsteps = Layout(), 0.0013, 1, 5
    records = []
    for npx, npy in TOPOLOGIES:
        series = _series(layout, npx, npy, nsteps, "all",
                         7900 + 100 * invmat1 + 10 * qmode + npx + npy)
        directory = tmp_path / f"all_{invmat1}_{qmode}_{npx}x{npy}"
        _prepare(directory, layout, npx, npy, invmat1, qmode,
                 series, dt, dtinv)
        record = _run(c7ca_harness, directory, npx, npy, layout,
                      invmat1, qmode, nsteps, dt, dtinv)
        assert record["temporal_error"] <= C7CA_MPI_REFERENCE_RELATIVE_MAX
        assert max(record["primary_error"], record["rho_error"],
                   record["q_error"]) <= C7CA_MPI_REFERENCE_RELATIVE_MAX
        if npx * npy > 1:
            assert record["wrong_q_difference"] > 1.0e-5
        records.append({**record, "topology": f"{npx}x{npy}"})
    print("M63C7CA_MPI " + json.dumps(records, sort_keys=True))


@pytest.mark.parametrize("selected", CHANNELS)
def test_single_rank_channel_isolation(c7ca_harness, tmp_path, selected):
    layout, npx, npy, nsteps, dt, dtinv = Layout(), 1, 1, 2, 0.0021, 1
    series = _series(layout, npx, npy, nsteps, selected, 8100 + CHANNELS.index(selected))
    directory = tmp_path / f"isolated_{selected}"
    _prepare(directory, layout, npx, npy, 1, 1, series, dt, dtinv)
    record = _run(c7ca_harness, directory, npx, npy, layout, 1, 1,
                  nsteps, dt, dtinv)
    assert record["temporal_error"] <= C7CA_MPI_REFERENCE_RELATIVE_MAX
    assert max(record["primary_error"], record["rho_error"],
               record["q_error"]) <= C7CA_MPI_REFERENCE_RELATIVE_MAX


def test_c6_linearity_and_operator_dot_contract():
    layout, npx, npy, invmat1, mapping = Layout(), 2, 2, 1, _mapping(1)
    dt, dtinv, nsteps = 0.0011, 1, 5
    primary, rho_values, q_values = _fields(layout, npx, npy, invmat1, 8250)
    series = _series(layout, npx, npy, nsteps, "all", 8251)
    accumulated = temporal_accumulate(series, dtinv)
    mapped_once = distributed_gradient(
        invmat1, mapping, primary, rho_values, q_values, accumulated,
        layout, npx, npy,
    )
    mapped_each = sum_mapped_per_step(
        invmat1, mapping, primary, rho_values, q_values, series,
        dtinv, layout, npx, npy,
    )
    for left, right in zip(mapped_once, mapped_each):
        assert _field_relative_l2(left, right) <= C7CA_LINEARITY_RELATIVE_MAX

    rng = random.Random(8252)
    perturbations = []
    for field in (primary, rho_values, q_values):
        perturbations.append([
            [value * rng.uniform(-0.01, 0.01) if 1 <= i <= layout.nx and 1 <= j <= layout.ny else 0.0
             for j in range(layout.ny + 2) for i, value in enumerate(
                 rank[j * (layout.nx + 2):(j + 1) * (layout.nx + 2)]
             )]
            for rank in field
        ])
    tangent = material_jvp(
        invmat1, mapping, primary, rho_values, q_values,
        perturbations[0], perturbations[1], perturbations[2],
        layout, npx, npy,
    )
    lhs = dot_outputs(tangent, c6_channel_order(accumulated))
    rhs = math.fsum(
        dot_fields(delta, gradient)
        for delta, gradient in zip(perturbations, mapped_once)
    )
    assert relative(lhs, rhs) <= C7CA_LINEARITY_RELATIVE_MAX


def test_c7ca_scope_order_and_locked_files(repository_root: Path):
    source = (repository_root / "src/SH/visco_sh_material_gradient_assembly.c").read_text()
    assert "dtinv != 1" in source
    assert "weight" not in source
    assert source.index("matcopy_SH_adjoint(") < source.index("q_to_tau_derivative(")
    assert "visco_sh_harmonic_pair_vjp(" in source
    assert "visco_sh_av_tau_local_vjp(" in source
    assert "visco_sh_rhoi_vjp(" in source
    for forbidden in ("grad_obj_sh", "waveconv", "DTINV", "assemble_gradSH"):
        assert forbidden not in source
    for path in (
        "src/SH/visco_sh_reverse_time_adjoint.c",
        "src/SH/visco_sh_full_state_adjoint_step.c",
        "src/SH/FWI_SH.c", "src/SH/FWI_SH_visc.c",
        "src/SH/grad_obj_sh.c", "src/SH/grad_obj_sh_visc.c",
        "src/SH/visco_sh_material_timestep_vjp.c",
        "src/SH/matcopy_SH_adjoint.c",
    ):
        assert "visco_sh_distributed_material_gradient_vjp" not in (
            repository_root / path
        ).read_text()
