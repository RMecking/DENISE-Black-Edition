"""M6.3c-4 exact MPI-exchange and flat free-surface transpose tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utilities.m63c_mpi_free_surface_reference import (
    Layout,
    SUPPORTED_FDORDERS,
    build_case,
    dot,
    exchange_transpose,
    surface_stress_transpose,
    surface_velocity_transpose,
    write_case_files,
)


# Frozen before the first compiled C4 result was inspected.
C4_MPI_DOT_RELATIVE_MAX = 5.0e-6
C4_MPI_REFERENCE_RELATIVE_MAX = 5.0e-6
C4_DOUBLE_DOT_RELATIVE_MAX = 5.0e-12

TOPOLOGIES = (
    (1, 1, 0),
    (2, 1, 0),
    (2, 1, 1),
    (1, 2, 0),
    (2, 2, 0),
)


def _relative_dot(lhs, rhs):
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300)


@pytest.fixture(scope="module")
def c4_harness(tmp_path_factory, repository_root):
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler is not None, "mpicc is required for the M6.3c-4 MPI test"
    assert launcher is not None, "mpiexec/mpirun is required for the M6.3c-4 MPI test"
    output = tmp_path_factory.mktemp("m63c_mpi_surface") / "m63c_mpi_surface"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fcommon",
        "-I",
        str(repository_root / "include"),
        str(repository_root / "tests/utilities/m63c_mpi_free_surface_harness.c"),
        str(repository_root / "src/SH/exchange_v_SH.c"),
        str(repository_root / "src/SH/exchange_s_SH.c"),
        str(repository_root / "src/SH/exchange_v_SH_adjoint.c"),
        str(repository_root / "src/SH/exchange_s_SH_adjoint.c"),
        str(repository_root / "src/SH/surface_elastic_SH.c"),
        str(repository_root / "src/SH/surface_elastic_SH_adjoint.c"),
        "-o",
        str(output),
        "-lm",
    ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    return launcher, output


def _run_case(
    c4_harness, tmp_path, operation, fdorder, nproc_x, nproc_y, boundary
):
    launcher, executable = c4_harness
    layout = Layout(nx=8, ny=8, fdorder=fdorder)
    directory = tmp_path / (
        f"{operation}_fd{fdorder}_{nproc_x}x{nproc_y}_b{boundary}"
    )
    write_case_files(
        directory, operation, layout, nproc_x, nproc_y, boundary
    )
    command = [
        launcher,
        "--oversubscribe",
        "-n",
        str(nproc_x * nproc_y),
        str(executable),
        str(nproc_x),
        str(nproc_y),
        str(boundary),
        str(fdorder),
        str(layout.nx),
        str(layout.ny),
        operation,
        str(directory),
    ]
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(records) == 1, result.stdout
    record = records[0]
    assert record["dot_residual"] <= C4_MPI_DOT_RELATIVE_MAX, record
    assert record["reference_error"] <= C4_MPI_REFERENCE_RELATIVE_MAX, record
    return record


def test_independent_double_reference_closes_all_c4_maps():
    records = []
    operations = ("v_exchange", "s_exchange", "v_surface", "s_surface", "v_composed", "s_composed")
    for fdorder in SUPPORTED_FDORDERS:
        layout = Layout(nx=8, ny=8, fdorder=fdorder)
        for nproc_x, nproc_y, boundary in TOPOLOGIES:
            for operation in operations:
                if operation.endswith("surface") and (nproc_x, nproc_y, boundary) != (1, 1, 0):
                    continue
                if operation.endswith("composed") and (nproc_x, nproc_y, boundary) != (1, 2, 0):
                    continue
                inputs, bars, forward, transpose = build_case(
                    operation, layout, nproc_x, nproc_y, boundary,
                    round_to_float=False,
                )
                residual = _relative_dot(dot(forward, bars), dot(inputs, transpose))
                assert residual <= C4_DOUBLE_DOT_RELATIVE_MAX
                records.append(
                    {
                        "operation": operation,
                        "fdorder": fdorder,
                        "topology": f"{nproc_x}x{nproc_y}",
                        "boundary": boundary,
                        "dot_residual": residual,
                    }
                )
    print("M63C4_REFERENCE " + json.dumps(records, sort_keys=True))


def test_reference_preserves_untouched_and_consumes_overwritten_ghosts():
    layout = Layout(nx=8, ny=8, fdorder=12)
    _, bars, _, transpose = build_case(
        "v_exchange", layout, 1, 1, 0, round_to_float=False
    )
    assert transpose == bars

    _, bars, _, transpose = build_case(
        "v_exchange", layout, 2, 1, 0, round_to_float=False
    )
    left_global_ghost = layout.index(3, 0)
    internal_right_ghost = layout.index(3, layout.nx + 1)
    assert transpose[0][0][left_global_ghost] == bars[0][0][left_global_ghost]
    assert transpose[0][0][internal_right_ghost] == 0.0

    _, bars, _, transpose = build_case(
        "v_exchange", layout, 2, 1, 1, round_to_float=False
    )
    assert transpose[0][0][left_global_ghost] == 0.0

    _, bars, _, transpose = build_case(
        "s_surface", layout, 1, 1, 0, round_to_float=False
    )
    assert transpose[0][1][layout.index(-layout.half, 4)] == bars[0][1][layout.index(-layout.half, 4)]
    assert transpose[0][1][layout.index(0, 4)] == 0.0


def test_actual_production_mpi_exchange_transposes(c4_harness, tmp_path):
    records = []
    for fdorder in SUPPORTED_FDORDERS:
        for nproc_x, nproc_y, boundary in TOPOLOGIES:
            for operation in ("v_exchange", "s_exchange"):
                records.append(
                    _run_case(
                        c4_harness, tmp_path, operation, fdorder,
                        nproc_x, nproc_y, boundary,
                    )
                )
    print("M63C4_MPI " + json.dumps(records, sort_keys=True))


def test_actual_free_surface_transposes_all_orders(c4_harness, tmp_path):
    records = []
    for fdorder in SUPPORTED_FDORDERS:
        for operation in ("v_surface", "s_surface"):
            records.append(
                _run_case(c4_harness, tmp_path, operation, fdorder, 1, 1, 0)
            )
    print("M63C4_SURFACE " + json.dumps(records, sort_keys=True))


def test_actual_composed_boundary_blocks(c4_harness, tmp_path):
    records = []
    for fdorder in SUPPORTED_FDORDERS:
        for operation in ("v_composed", "s_composed"):
            records.append(
                _run_case(c4_harness, tmp_path, operation, fdorder, 1, 2, 0)
            )
    print("M63C4_COMPOSED " + json.dumps(records, sort_keys=True))
