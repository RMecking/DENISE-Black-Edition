"""M6.3c-7b local per-timestep native material-sensitivity verification."""

from __future__ import annotations

import ctypes
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.utilities.m63c_material_timestep_vjp_reference import (
    C7B_DIRECTIONAL_RELATIVE_MAX,
    C7B_REFERENCE_RELATIVE_MAX,
    CASES,
    coefficients,
    finite_difference_gradient,
    five_point_directional,
)


DP = ctypes.POINTER(ctypes.c_double)


def _relative(actual, expected):
    return abs(actual - expected) / max(abs(actual), abs(expected), 1.0e-300)


@pytest.fixture(scope="module")
def c7b(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for M6.3c-7b"
    library = tmp_path_factory.mktemp("m63c7b") / "libm63c7b.so"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fPIC",
        "-shared",
        "-fcommon",
        "-I",
        str(repository_root / "include"),
        str(repository_root / "tests/utilities/m63c_material_timestep_vjp_harness.c"),
        str(repository_root / "src/SH/visco_sh_material_timestep_vjp.c"),
        str(repository_root / "src/SH/visco_sh_gsls_vjp.c"),
        str(repository_root / "src/SH/visco_sh_material_vjp.c"),
        "-o",
        str(library),
        "-lm",
    ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    api = ctypes.CDLL(str(library))
    api.m63c_material_timestep_vjp_harness.argtypes = (
        [ctypes.c_int]
        + [ctypes.c_double] * 8
        + [DP, DP]
        + [ctypes.c_double] * 5
        + [DP, DP, DP, DP]
        + [ctypes.c_double] * 2
        + [DP, DP, DP, DP, DP]
    )
    api.m63c_material_timestep_vjp_harness.restype = ctypes.c_int
    return api


def _array(values):
    return (ctypes.c_double * len(values))(*values)


def _production(api, case):
    x = coefficients(case.mu_x, case.tau_x, case.dt, case.frequencies)
    y = coefficients(case.mu_y, case.tau_y, case.dt, case.frequencies)
    assert x["reference_sum"] == y["reference_sum"]
    output = (ctypes.c_double * 5)()
    status = api.m63c_material_timestep_vjp_harness(
        case.mechanisms,
        case.dt,
        case.dh,
        case.qsum,
        case.strain_x,
        case.strain_y,
        case.bar_v,
        case.bar_sx,
        case.bar_sy,
        _array(case.bar_rx),
        _array(case.bar_qy),
        case.mu_x,
        case.tau_x,
        case.mu_y,
        case.tau_y,
        x["reference_sum"],
        _array(x["eta"]),
        _array(x["b"]),
        _array(y["eta"]),
        _array(y["b"]),
        x["stress"],
        y["stress"],
        _array(x["recurrence"]),
        _array(x["coupling"]),
        _array(y["recurrence"]),
        _array(y["coupling"]),
        output,
    )
    assert status == 0
    assert all(math.isfinite(value) for value in output)
    return tuple(output)


def _zero_memory(case):
    return (0.0,) * case.mechanisms


def test_c7b_density_vjp_has_exact_sign_dt_dh_and_zero_contract(c7b):
    base = CASES[0]
    diagnostics = []
    for dt, dh, qsum, bar_v in (
        (0.001, 8.0, 0.7, 0.4),
        (0.003, 5.0, -0.9, 0.6),
        (0.002, 11.0, 0.5, -0.8),
        (0.004, 3.0, 0.0, 2.0),
        (0.004, 3.0, 2.0, 0.0),
    ):
        case = replace(
            base,
            dt=dt,
            dh=dh,
            qsum=qsum,
            bar_v=bar_v,
            bar_sx=0.0,
            bar_sy=0.0,
            bar_rx=_zero_memory(base),
            bar_qy=_zero_memory(base),
        )
        actual = _production(c7b, case)
        expected = dt / dh * qsum * bar_v
        assert actual[0] == pytest.approx(expected, rel=2.0e-15, abs=1.0e-18)
        assert actual[1:] == pytest.approx((0.0,) * 4, abs=1.0e-15)
        diagnostics.append({"actual": actual[0], "expected": expected})
    print("M63C7B_DENSITY " + json.dumps(diagnostics, sort_keys=True))


@pytest.mark.parametrize("case", CASES, ids=lambda value: value.name)
def test_c7b_all_native_channels_match_independent_five_point_fd(c7b, case):
    actual = _production(c7b, case)
    expected = finite_difference_gradient(case)
    errors = tuple(_relative(a, e) for a, e in zip(actual, expected))
    print(
        "M63C7B_CHANNEL_FD "
        + json.dumps(
            {"case": case.name, "actual": actual, "fd": expected, "errors": errors},
            sort_keys=True,
        )
    )
    assert max(errors) <= C7B_REFERENCE_RELATIVE_MAX


@pytest.mark.parametrize("case", CASES, ids=lambda value: value.name)
def test_c7b_combined_directional_derivative_matches_independent_fd(c7b, case):
    direction = (1.1e-4, 0.23, -0.17, 0.007, -0.005)
    gradient = _production(c7b, case)
    product = math.fsum(value * delta for value, delta in zip(gradient, direction))
    derivative = five_point_directional(case, direction)
    error = _relative(product, derivative)
    print(
        "M63C7B_DIRECTIONAL "
        + json.dumps(
            {"case": case.name, "product": product, "fd": derivative, "error": error},
            sort_keys=True,
        )
    )
    assert error <= C7B_DIRECTIONAL_RELATIVE_MAX


def test_c7b_x_y_and_density_channels_are_independent(c7b):
    base = CASES[2]
    zero = _zero_memory(base)
    density = _production(
        c7b,
        replace(base, bar_sx=0.0, bar_sy=0.0, bar_rx=zero, bar_qy=zero),
    )
    x_only = _production(
        c7b,
        replace(base, bar_v=0.0, bar_sy=0.0, bar_qy=zero),
    )
    y_only = _production(
        c7b,
        replace(base, bar_v=0.0, bar_sx=0.0, bar_rx=zero),
    )
    all_zero = _production(
        c7b,
        replace(
            base, bar_v=0.0, bar_sx=0.0, bar_sy=0.0,
            bar_rx=zero, bar_qy=zero,
        ),
    )
    assert density[1:] == pytest.approx((0.0,) * 4, abs=1.0e-15)
    assert x_only[0] == 0.0
    assert x_only[2] == 0.0
    assert x_only[4] == 0.0
    assert y_only[0] == 0.0
    assert y_only[1] == 0.0
    assert y_only[3] == 0.0
    assert all_zero == (0.0,) * 5


def test_c7b_composes_locked_helpers_without_temporal_or_parameter_mapping(
    repository_root: Path,
):
    source = (
        repository_root / "src/SH/visco_sh_material_timestep_vjp.c"
    ).read_text()
    header = (repository_root / "include/fd.h").read_text()
    assert source.count("visco_sh_gsls_local_derivatives(") == 1
    assert source.count("visco_sh_gsls_local_vjp(") == 1
    assert source.count("visco_sh_velocity_rhoi_vjp(") == 1
    for forbidden in (
        "DTINV", "matcopy_SH", "q_to_tau", "q_tau_derivative",
        "grad_obj_sh", "waveconv", "MPI_", "INVMAT1",
    ):
        assert forbidden not in source
    structure = header.split(
        "struct visco_sh_material_timestep_vjp_input {", 1
    )[1].split("};", 1)[0]
    assert "double rhoi" not in structure
    assert "double qsum, strain_x, strain_y;" in structure
