from __future__ import annotations

import ctypes
import json
import math
import random
import shutil
import subprocess

import pytest

from tests.utilities.m63c_acceptance import relative_agreement
from tests.utilities.m63c_velocity_sampling_reference import (
    CpmlBranch,
    SUPPORTED_FDORDERS,
    cpml_forward,
    cpml_transpose,
    receiver_sample,
    receiver_transpose,
    source_inject,
    source_transpose,
    stress_derivatives,
    stress_derivatives_transpose,
    velocity_forward,
    velocity_transpose,
)


# Frozen before the first comparison with the compiled C3 implementation.
C3_DOUBLE_DOT_RELATIVE_MAX = 5.0e-12
C3_DOUBLE_REFERENCE_RELATIVE_MAX = 5.0e-12


def _doubles(values):
    return (ctypes.c_double * len(values))(*values)


def _floats(values):
    return (ctypes.c_float * len(values))(*values)


def _ints(values):
    return (ctypes.c_int * len(values))(*values)


def _float32(values):
    return tuple(float(ctypes.c_float(value).value) for value in values)


def _inner(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _relative_dot(lhs, rhs):
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300)


def _max_relative(left, right):
    return max(relative_agreement(a, b) for a, b in zip(left, right))


@pytest.fixture(scope="module")
def compiled_velocity_vjp(tmp_path_factory, repository_root):
    compiler = shutil.which("mpicc")
    assert compiler is not None, "mpicc is required for the compiled C3 test"
    output = tmp_path_factory.mktemp("m63c_velocity") / "libm63c_velocity.so"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fcommon",
        "-fPIC",
        "-shared",
        "-I",
        str(repository_root / "include"),
        str(repository_root / "src/SH/update_v_PML_SH_adjoint.c"),
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
    library = ctypes.CDLL(str(output))
    integer = ctypes.c_int
    scalar = ctypes.c_double
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(integer)
    dp = ctypes.POINTER(scalar)
    library.visco_sh_velocity_cpml_select_x.argtypes = [
        integer, integer, integer, integer, integer, integer,
        fp, fp, fp, ip, ip, dp, dp, dp,
    ]
    library.visco_sh_velocity_cpml_select_x.restype = integer
    library.visco_sh_velocity_cpml_select_y.argtypes = [
        integer, integer, integer, integer, integer, integer,
        fp, fp, fp, ip, ip, dp, dp, dp,
    ]
    library.visco_sh_velocity_cpml_select_y.restype = integer
    library.visco_sh_velocity_cpml_local_vjp.argtypes = [
        integer, scalar, scalar, scalar, scalar, scalar, dp, dp,
    ]
    library.visco_sh_velocity_cpml_local_vjp.restype = integer
    library.visco_sh_velocity_spatial_local_vjp.argtypes = [
        integer, fp, scalar, scalar, dp, dp,
        integer, integer, integer, integer,
    ]
    library.visco_sh_velocity_spatial_local_vjp.restype = integer
    library.update_v_PML_SH_adjoint_point.argtypes = [
        integer, scalar, scalar, ctypes.c_float, fp,
        ip, dp, dp, dp, scalar, dp, dp, dp, dp, dp,
        integer, integer, integer, integer,
    ]
    library.update_v_PML_SH_adjoint_point.restype = integer
    library.visco_sh_receiver_velocity_sampling_vjp.argtypes = [
        integer, ip, ip, dp, dp, integer, integer,
    ]
    library.visco_sh_receiver_velocity_sampling_vjp.restype = integer
    library.visco_sh_velocity_source_injection_vjp.argtypes = [
        integer, integer, dp, dp, integer, ip, ip, ip, dp,
    ]
    library.visco_sh_velocity_source_injection_vjp.restype = integer
    return library


def _selector(library, axis, coordinate, extent, fw, free_or_boundary, pos, nproc, arrays):
    active = ctypes.c_int()
    index = ctypes.c_int()
    K = ctypes.c_double()
    a = ctypes.c_double()
    b = ctypes.c_double()
    function = (
        library.visco_sh_velocity_cpml_select_x
        if axis == "x"
        else library.visco_sh_velocity_cpml_select_y
    )
    status = function(
        coordinate, extent, fw, free_or_boundary, pos, nproc,
        *(_floats(values) for values in arrays),
        ctypes.byref(active), ctypes.byref(index),
        ctypes.byref(K), ctypes.byref(a), ctypes.byref(b),
    )
    assert status == 0
    return active.value, index.value, K.value, a.value, b.value


def test_velocity_cpml_selectors_match_production_indexing(compiled_velocity_vjp):
    library = compiled_velocity_vjp
    fw = 3
    K = tuple(1.0 + 0.1 * index for index in range(2 * fw + 1))
    a = tuple(-0.01 * index for index in range(2 * fw + 1))
    b = tuple(0.6 + 0.01 * index for index in range(2 * fw + 1))
    promoted = lambda values, index: float(ctypes.c_float(values[index]).value)
    expected = lambda index: (
        1, index, promoted(K, index), promoted(a, index), promoted(b, index)
    )

    assert _selector(library, "x", 2, 20, fw, 0, 0, 2, (K, a, b)) == expected(2)
    assert _selector(library, "x", 19, 20, fw, 0, 1, 2, (K, a, b)) == expected(5)
    assert _selector(library, "x", 10, 20, fw, 0, 0, 2, (K, a, b)) == (0, -1, 1.0, 0.0, 0.0)
    assert _selector(library, "x", 2, 20, fw, 1, 0, 2, (K, a, b)) == (0, -1, 1.0, 0.0, 0.0)
    assert _selector(library, "y", 2, 20, fw, 0, 0, 2, (K, a, b)) == expected(2)
    assert _selector(library, "y", 2, 20, fw, 1, 0, 2, (K, a, b)) == (0, -1, 1.0, 0.0, 0.0)
    assert _selector(library, "y", 19, 20, fw, 0, 1, 2, (K, a, b)) == expected(5)
    assert _selector(library, "y", 10, 20, fw, 0, 0, 2, (K, a, b)) == (0, -1, 1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "cpml",
    (CpmlBranch(True, 1.47, -0.18, 0.69), CpmlBranch(True, 1.81, -0.12, 0.63)),
)
def test_velocity_cpml_state_dot(compiled_velocity_vjp, cpml):
    raw = -0.41
    psi_previous = 0.28
    bar_q = 0.57
    bar_psi_next = -0.33
    corrected, psi_next = cpml_forward(raw, psi_previous, cpml)
    expected_raw, expected_psi = cpml_transpose(bar_q, bar_psi_next, cpml)
    actual_raw = ctypes.c_double(0.0)
    actual_psi = ctypes.c_double(0.0)
    status = compiled_velocity_vjp.visco_sh_velocity_cpml_local_vjp(
        1, cpml.K, cpml.a, cpml.b, bar_q, bar_psi_next,
        ctypes.byref(actual_raw), ctypes.byref(actual_psi),
    )
    assert status == 0
    assert relative_agreement(actual_raw.value, expected_raw) <= C3_DOUBLE_REFERENCE_RELATIVE_MAX
    assert relative_agreement(actual_psi.value, expected_psi) <= C3_DOUBLE_REFERENCE_RELATIVE_MAX
    residual = _relative_dot(
        corrected * bar_q + psi_next * bar_psi_next,
        raw * actual_raw.value + psi_previous * actual_psi.value,
    )
    assert residual <= C3_DOUBLE_DOT_RELATIVE_MAX
    print("M63C3_CPML " + json.dumps({"dot_residual": residual}, sort_keys=True))


def test_velocity_spatial_transpose_all_orders(compiled_velocity_vjp):
    randomizer = random.Random(63301)
    records = []
    for fdorder in SUPPORTED_FDORDERS:
        half = fdorder // 2
        side = 2 * half + 1
        center = half * side + half
        hc = _float32((0.0,) + tuple(((-1.0) ** (m + 1)) * 1.13 / m for m in range(1, half + 1)))
        sxz = tuple(randomizer.uniform(-0.8, 0.8) for _ in range(side * side))
        syz = tuple(randomizer.uniform(-0.8, 0.8) for _ in range(side * side))
        bar_dx = randomizer.uniform(-0.9, 0.9)
        bar_dy = randomizer.uniform(-0.9, 0.9)
        raw = stress_derivatives(sxz, syz, side=side, center=center, fdorder=fdorder, hc=hc)
        for label, selected_x, selected_y in (
            ("x", bar_dx, 0.0), ("y", 0.0, bar_dy), ("xy", bar_dx, bar_dy)
        ):
            expected_x, expected_y = stress_derivatives_transpose(
                (0.0,) * (side * side), (0.0,) * (side * side),
                side=side, center=center, fdorder=fdorder, hc=hc,
                bar_dx=selected_x, bar_dy=selected_y,
            )
            actual_x = _doubles((0.0,) * (side * side))
            actual_y = _doubles((0.0,) * (side * side))
            status = compiled_velocity_vjp.visco_sh_velocity_spatial_local_vjp(
                fdorder, _floats(hc), selected_x, selected_y,
                actual_x, actual_y, side, side, half, half,
            )
            assert status == 0
            error = max(_max_relative(actual_x, expected_x), _max_relative(actual_y, expected_y))
            residual = _relative_dot(
                raw[0] * selected_x + raw[1] * selected_y,
                _inner(sxz, actual_x) + _inner(syz, actual_y),
            )
            assert error <= C3_DOUBLE_REFERENCE_RELATIVE_MAX
            assert residual <= C3_DOUBLE_DOT_RELATIVE_MAX
            records.append({"fdorder": fdorder, "case": label, "dot_residual": residual, "reference_error": error})
    print("M63C3_SPATIAL " + json.dumps(records, sort_keys=True))


def _case(name, fdorder, cpml, seed):
    randomizer = random.Random(seed)
    half = fdorder // 2
    side = 2 * half + 1
    center = half * side + half
    hc = _float32((0.0,) + tuple(((-1.0) ** (m + 1)) * 1.21 / m for m in range(1, half + 1)))
    rhoi = float(ctypes.c_float(0.00043 + seed % 7 * 0.000017).value)
    return {
        "name": name, "fdorder": fdorder, "half": half, "side": side,
        "center": center, "hc": hc, "rhoi": rhoi, "dt": 0.0005,
        "dh": 8.0, "cpml": cpml, "randomizer": randomizer,
    }


def _call_velocity(library, case, bar_vz_next, bar_psi_next, initial):
    bar_vz = ctypes.c_double(initial["vz"])
    bar_psi = _doubles(initial["psi"])
    bar_sxz = _doubles(initial["sxz"])
    bar_syz = _doubles(initial["syz"])
    cpml = case["cpml"]
    status = library.update_v_PML_SH_adjoint_point(
        case["fdorder"], case["dt"], case["dh"], case["rhoi"], _floats(case["hc"]),
        _ints(tuple(int(value.active) for value in cpml)),
        _doubles(tuple(value.K for value in cpml)),
        _doubles(tuple(value.a for value in cpml)),
        _doubles(tuple(value.b for value in cpml)),
        bar_vz_next, _doubles(bar_psi_next), ctypes.byref(bar_vz), bar_psi,
        bar_sxz, bar_syz, case["side"], case["side"], case["half"], case["half"],
    )
    assert status == 0
    return {"vz": bar_vz.value, "psi": tuple(bar_psi), "sxz": tuple(bar_sxz), "syz": tuple(bar_syz)}


def test_full_velocity_block_reference_and_dot(compiled_velocity_vjp):
    inactive = CpmlBranch(False)
    x = CpmlBranch(True, 1.43, -0.16, 0.71)
    y = CpmlBranch(True, 1.69, -0.11, 0.64)
    cases = [
        _case("interior", 2, (inactive, inactive), 633100),
        _case("left", 4, (x, inactive), 633101),
        _case("right", 6, (x, inactive), 633102),
        _case("top_fs0", 8, (inactive, y), 633103),
        _case("bottom", 10, (inactive, y), 633104),
        _case("corner", 12, (x, y), 633105),
        _case("top_fs1_disabled", 8, (inactive, inactive), 633106),
    ]
    records = []
    for case in cases:
        r = case["randomizer"]
        count = case["side"] ** 2
        sxz = tuple(r.uniform(-0.8, 0.8) for _ in range(count))
        syz = tuple(r.uniform(-0.8, 0.8) for _ in range(count))
        vz = r.uniform(-0.7, 0.7)
        psi = tuple(r.uniform(-0.3, 0.3) if value.active else 0.0 for value in case["cpml"])
        bar_vz = r.uniform(-0.9, 0.9)
        bar_psi = tuple(r.uniform(-0.9, 0.9) if value.active else 0.0 for value in case["cpml"])
        zero = {"vz": 0.0, "psi": (0.0, 0.0), "sxz": (0.0,) * count, "syz": (0.0,) * count}
        actual = _call_velocity(compiled_velocity_vjp, case, bar_vz, bar_psi, zero)
        expected = velocity_transpose(
            initial_vz=0.0, initial_sxz=zero["sxz"], initial_syz=zero["syz"], initial_psi=zero["psi"],
            bar_vz_next=bar_vz, bar_psi_next=bar_psi, cpml=case["cpml"], side=case["side"],
            center=case["center"], fdorder=case["fdorder"], hc=case["hc"],
            dt=case["dt"], dh=case["dh"], rhoi=case["rhoi"],
        )
        error = max(
            relative_agreement(actual["vz"], expected["bar_vz_previous"]),
            _max_relative(actual["psi"], expected["bar_psi_previous"]),
            _max_relative(actual["sxz"], expected["bar_sxz"]),
            _max_relative(actual["syz"], expected["bar_syz"]),
        )
        forward = velocity_forward(
            vz_previous=vz, sxz=sxz, syz=syz, psi_previous=psi, cpml=case["cpml"],
            side=case["side"], center=case["center"], fdorder=case["fdorder"], hc=case["hc"],
            dt=case["dt"], dh=case["dh"], rhoi=case["rhoi"],
        )
        lhs = forward["vz_next"] * bar_vz
        rhs = vz * actual["vz"] + _inner(sxz, actual["sxz"]) + _inner(syz, actual["syz"])
        for axis in range(2):
            if case["cpml"][axis].active:
                lhs += forward["psi_next"][axis] * bar_psi[axis]
                rhs += psi[axis] * actual["psi"][axis]
        residual = _relative_dot(lhs, rhs)
        assert error <= C3_DOUBLE_REFERENCE_RELATIVE_MAX
        assert residual <= C3_DOUBLE_DOT_RELATIVE_MAX
        records.append({"case": case["name"], "dot_residual": residual, "reference_error": error})
    print("M63C3_VELOCITY " + json.dumps(records, sort_keys=True))


def test_receiver_sampling_euclidean_transpose(compiled_velocity_vjp):
    rows, stride = 4, 6
    positions = ((2, 1), (4, 2), (2, 1), (1, 3))
    vz = tuple(0.03 * (index + 1) for index in range(rows * stride))
    bar_data = (0.7, -0.2, 0.4, -0.5)
    initial = tuple(-0.01 * index for index in range(rows * stride))
    expected = receiver_transpose(initial, positions, bar_data, stride=stride)
    actual = _doubles(initial)
    status = compiled_velocity_vjp.visco_sh_receiver_velocity_sampling_vjp(
        len(positions), _ints(tuple(x for x, _ in positions)), _ints(tuple(y for _, y in positions)),
        _doubles(bar_data), actual, rows, stride,
    )
    assert status == 0
    assert tuple(actual) == pytest.approx(expected, rel=0.0, abs=0.0)
    lhs = _inner(receiver_sample(vz, positions, stride=stride), bar_data)
    transpose_only = tuple(a - b for a, b in zip(actual, initial))
    residual = _relative_dot(lhs, _inner(vz, transpose_only))
    assert residual <= C3_DOUBLE_DOT_RELATIVE_MAX
    # Primitive C^T is an unweighted Euclidean scatter: no rhoi factor.
    assert transpose_only[1 * stride + 2] == pytest.approx(1.1, rel=0.0, abs=1.0e-15)
    print("M63C3_RECEIVER " + json.dumps({"dot_residual": residual}, sort_keys=True))


def test_physical_source_injection_transpose(compiled_velocity_vjp):
    rows, stride = 4, 6
    positions = ((2, 1), (4, 2), (2, 1), (1, 3))
    types = (1, 2, 1, 1)
    signals = (0.3, 0.8, -0.2, 0.5)
    vz = tuple(0.02 * (index - 5) for index in range(rows * stride))
    bar_after = tuple(0.04 * (index + 1) for index in range(rows * stride))
    initial_vz = tuple(-0.01 * index for index in range(rows * stride))
    initial_signal = (0.1, 0.2, 0.3, 0.4)
    expected_vz, expected_signal = source_transpose(
        initial_vz, initial_signal, bar_after, positions, types, stride=stride
    )
    actual_vz = _doubles(initial_vz)
    actual_signal = _doubles(initial_signal)
    status = compiled_velocity_vjp.visco_sh_velocity_source_injection_vjp(
        rows, stride, _doubles(bar_after), actual_vz, len(positions),
        _ints(tuple(x for x, _ in positions)), _ints(tuple(y for _, y in positions)),
        _ints(types), actual_signal,
    )
    assert status == 0
    assert tuple(actual_vz) == pytest.approx(expected_vz, rel=0.0, abs=0.0)
    assert tuple(actual_signal) == pytest.approx(expected_signal, rel=0.0, abs=0.0)
    injected = source_inject(vz, positions, types, signals, stride=stride)
    source_dual = tuple(a - b for a, b in zip(actual_signal, initial_signal))
    vz_dual = tuple(a - b for a, b in zip(actual_vz, initial_vz))
    residual = _relative_dot(_inner(injected, bar_after), _inner(vz, vz_dual) + _inner(signals, source_dual))
    assert residual <= C3_DOUBLE_DOT_RELATIVE_MAX
    assert source_dual[1] == 0.0
    print("M63C3_SOURCE " + json.dumps({"dot_residual": residual}, sort_keys=True))


def test_composed_source_velocity_receiver_block(compiled_velocity_vjp):
    case = _case("composed_corner", 8, (CpmlBranch(True, 1.52, -0.14, 0.67), CpmlBranch(True, 1.74, -0.09, 0.62)), 633500)
    rows = stride = case["side"]
    center_xy = (case["half"], case["half"])
    positions_source = (center_xy, center_xy, (1, 1))
    source_types = (1, 1, 2)
    positions_receiver = (center_xy, (1, 1), center_xy)
    r = case["randomizer"]
    cells = rows * stride
    vz_before = tuple(r.uniform(-0.5, 0.5) for _ in range(cells))
    signals = (0.17, -0.08, 0.33)
    sxz = tuple(r.uniform(-0.6, 0.6) for _ in range(cells))
    syz = tuple(r.uniform(-0.6, 0.6) for _ in range(cells))
    psi = (0.11, -0.07)
    after_source = source_inject(vz_before, positions_source, source_types, signals, stride=stride)
    forward_velocity = velocity_forward(
        vz_previous=after_source[case["center"]], sxz=sxz, syz=syz, psi_previous=psi,
        cpml=case["cpml"], side=stride, center=case["center"], fdorder=case["fdorder"],
        hc=case["hc"], dt=case["dt"], dh=case["dh"], rhoi=case["rhoi"],
    )
    after_velocity = list(after_source)
    after_velocity[case["center"]] = forward_velocity["vz_next"]
    data = receiver_sample(after_velocity, positions_receiver, stride=stride)
    bar_data = (0.41, -0.23, 0.37)
    bar_psi_next = (-0.19, 0.29)

    bar_after_velocity = _doubles((0.0,) * cells)
    assert compiled_velocity_vjp.visco_sh_receiver_velocity_sampling_vjp(
        len(positions_receiver), _ints(tuple(x for x, _ in positions_receiver)),
        _ints(tuple(y for _, y in positions_receiver)), _doubles(bar_data),
        bar_after_velocity, rows, stride,
    ) == 0
    center_bar = bar_after_velocity[case["center"]]
    bar_after_source = list(bar_after_velocity)
    bar_after_source[case["center"]] = 0.0
    local = _call_velocity(
        compiled_velocity_vjp, case, center_bar, bar_psi_next,
        {"vz": 0.0, "psi": (0.0, 0.0), "sxz": (0.0,) * cells, "syz": (0.0,) * cells},
    )
    bar_after_source[case["center"]] += local["vz"]
    bar_vz_before = _doubles((0.0,) * cells)
    bar_signal = _doubles((0.0,) * len(signals))
    assert compiled_velocity_vjp.visco_sh_velocity_source_injection_vjp(
        rows, stride, _doubles(bar_after_source), bar_vz_before,
        len(positions_source), _ints(tuple(x for x, _ in positions_source)),
        _ints(tuple(y for _, y in positions_source)), _ints(source_types), bar_signal,
    ) == 0
    lhs = _inner(data, bar_data) + _inner(forward_velocity["psi_next"], bar_psi_next)
    rhs = (
        _inner(vz_before, bar_vz_before) + _inner(signals, bar_signal)
        + _inner(sxz, local["sxz"]) + _inner(syz, local["syz"])
        + _inner(psi, local["psi"])
    )
    residual = _relative_dot(lhs, rhs)
    assert residual <= C3_DOUBLE_DOT_RELATIVE_MAX
    reference_receiver = receiver_transpose((0.0,) * cells, positions_receiver, bar_data, stride=stride)
    reference_local = velocity_transpose(
        initial_vz=0.0, initial_sxz=(0.0,) * cells, initial_syz=(0.0,) * cells,
        initial_psi=(0.0, 0.0), bar_vz_next=reference_receiver[case["center"]],
        bar_psi_next=bar_psi_next, cpml=case["cpml"], side=stride, center=case["center"],
        fdorder=case["fdorder"], hc=case["hc"], dt=case["dt"], dh=case["dh"], rhoi=case["rhoi"],
    )
    reference_after_source = list(reference_receiver)
    reference_after_source[case["center"]] = reference_local["bar_vz_previous"]
    reference_vz, reference_signal = source_transpose(
        (0.0,) * cells, (0.0,) * len(signals), reference_after_source,
        positions_source, source_types, stride=stride,
    )
    error = max(
        _max_relative(bar_vz_before, reference_vz), _max_relative(bar_signal, reference_signal),
        _max_relative(local["sxz"], reference_local["bar_sxz"]),
        _max_relative(local["syz"], reference_local["bar_syz"]),
        _max_relative(local["psi"], reference_local["bar_psi_previous"]),
    )
    assert error <= C3_DOUBLE_REFERENCE_RELATIVE_MAX
    print("M63C3_COMPOSED " + json.dumps({"dot_residual": residual, "reference_error": error}, sort_keys=True))
