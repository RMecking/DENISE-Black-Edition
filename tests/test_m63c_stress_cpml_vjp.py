from __future__ import annotations

import ctypes
import json
import math
import random
import shutil
import subprocess

import pytest

from tests.utilities.m63c_acceptance import (
    local_gsls_coefficients,
    relative_agreement,
)
from tests.utilities.m63c_stress_cpml_reference import (
    CpmlBranch,
    SUPPORTED_FDORDERS,
    cpml_forward,
    cpml_transpose,
    spatial_forward,
    spatial_transpose,
    stress_block_forward,
    stress_block_transpose,
)


# Frozen before the first comparison with the compiled C2 implementation.
C2_DOUBLE_DOT_RELATIVE_MAX = 5.0e-12
C2_DOUBLE_REFERENCE_RELATIVE_MAX = 5.0e-12


def _doubles(values):
    return (ctypes.c_double * len(values))(*values)


def _floats(values):
    return (ctypes.c_float * len(values))(*values)


def _ints(values):
    return (ctypes.c_int * len(values))(*values)


def _inner(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _relative_dot(lhs, rhs):
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300)


@pytest.fixture(scope="module")
def compiled_stress_cpml_vjp(tmp_path_factory, repository_root):
    compiler = shutil.which("mpicc")
    assert compiler is not None, "mpicc is required for the compiled C2 test"
    output = tmp_path_factory.mktemp("m63c_stress_cpml") / "libm63c_stress_cpml.so"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fcommon",
        "-fPIC",
        "-shared",
        "-I",
        str(repository_root / "include"),
        str(repository_root / "src/SH/visco_sh_gsls_vjp.c"),
        str(repository_root / "src/SH/update_s_visc_PML_SH_adjoint.c"),
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
    ip = ctypes.POINTER(integer)
    dp = ctypes.POINTER(scalar)
    fp = ctypes.POINTER(ctypes.c_float)
    library.visco_sh_stress_cpml_select_x.argtypes = [
        integer, integer, integer, integer, integer, integer,
        fp, fp, fp, ip, ip, dp, dp, dp,
    ]
    library.visco_sh_stress_cpml_select_x.restype = integer
    library.visco_sh_stress_cpml_select_y.argtypes = [
        integer, integer, integer, integer, integer, integer,
        fp, fp, fp, fp, fp, fp, ip, ip, dp, dp, dp,
    ]
    library.visco_sh_stress_cpml_select_y.restype = integer
    library.visco_sh_stress_cpml_local_vjp.argtypes = [
        integer, scalar, scalar, scalar, scalar, scalar, dp, dp,
    ]
    library.visco_sh_stress_cpml_local_vjp.restype = integer
    library.visco_sh_stress_spatial_local_vjp.argtypes = [
        integer, scalar, fp, scalar, scalar, dp,
        integer, integer, integer, integer,
    ]
    library.visco_sh_stress_spatial_local_vjp.restype = integer
    library.update_s_visc_PML_SH_adjoint_point.argtypes = (
        [integer, integer, scalar, scalar, fp, ip]
        + [dp] * 24
        + [integer] * 4
        + [dp] * 2
    )
    library.update_s_visc_PML_SH_adjoint_point.restype = integer
    return library


def _select_x(library, *, i, nx2, fw, boundary, pos, nproc, arrays):
    active = ctypes.c_int()
    index = ctypes.c_int()
    K = ctypes.c_double()
    a = ctypes.c_double()
    b = ctypes.c_double()
    status = library.visco_sh_stress_cpml_select_x(
        i, nx2, fw, boundary, pos, nproc,
        *(_floats(values) for values in arrays),
        ctypes.byref(active), ctypes.byref(index),
        ctypes.byref(K), ctypes.byref(a), ctypes.byref(b),
    )
    assert status == 0
    return active.value, index.value, K.value, a.value, b.value


def _select_y(library, *, j, ny2, fw, free_surface, pos, nproc, arrays):
    active = ctypes.c_int()
    index = ctypes.c_int()
    K = ctypes.c_double()
    a = ctypes.c_double()
    b = ctypes.c_double()
    status = library.visco_sh_stress_cpml_select_y(
        j, ny2, fw, free_surface, pos, nproc,
        *(_floats(values) for values in arrays),
        ctypes.byref(active), ctypes.byref(index),
        ctypes.byref(K), ctypes.byref(a), ctypes.byref(b),
    )
    assert status == 0
    return active.value, index.value, K.value, a.value, b.value


def test_compiled_cpml_selection_matches_forward_indexing(compiled_stress_cpml_vjp):
    library = compiled_stress_cpml_vjp
    fw = 3
    primary = tuple(10.0 + index for index in range(2 * fw + 1))
    half = tuple(20.0 + index for index in range(2 * fw + 1))
    a_primary = tuple(-0.01 * index for index in range(2 * fw + 1))
    a_half = tuple(-0.02 * index for index in range(2 * fw + 1))
    b_primary = tuple(0.5 + 0.01 * index for index in range(2 * fw + 1))
    b_half = tuple(0.6 + 0.01 * index for index in range(2 * fw + 1))
    promoted = lambda value: float(ctypes.c_float(value).value)

    assert _select_x(
        library, i=2, nx2=20, fw=fw, boundary=0, pos=0, nproc=2,
        arrays=(half, a_half, b_half),
    ) == (1, 2, promoted(half[2]), promoted(a_half[2]), promoted(b_half[2]))
    assert _select_x(
        library, i=19, nx2=20, fw=fw, boundary=0, pos=1, nproc=2,
        arrays=(half, a_half, b_half),
    ) == (1, 5, promoted(half[5]), promoted(a_half[5]), promoted(b_half[5]))
    assert _select_x(
        library, i=2, nx2=20, fw=fw, boundary=1, pos=0, nproc=2,
        arrays=(half, a_half, b_half),
    ) == (0, -1, 1.0, 0.0, 0.0)
    y_arrays = (primary, a_primary, b_primary, half, a_half, b_half)
    assert _select_y(
        library, j=2, ny2=20, fw=fw, free_surface=0, pos=0, nproc=2,
        arrays=y_arrays,
    ) == (1, 2, promoted(primary[2]), promoted(a_primary[2]), promoted(b_primary[2]))
    assert _select_y(
        library, j=19, ny2=20, fw=fw, free_surface=0, pos=1, nproc=2,
        arrays=y_arrays,
    ) == (1, 5, promoted(half[5]), promoted(a_half[5]), promoted(b_half[5]))
    assert _select_y(
        library, j=2, ny2=20, fw=fw, free_surface=1, pos=0, nproc=2,
        arrays=y_arrays,
    ) == (0, -1, 1.0, 0.0, 0.0)


def test_compiled_cpml_temporal_state_transpose(compiled_stress_cpml_vjp):
    cpml = CpmlBranch(True, K=1.73, a=-0.19, b=0.68)
    raw_tangent = -0.37
    psi_tangent = 0.29
    bar_corrected = 0.61
    bar_psi_next = -0.44
    corrected, psi_next = cpml_forward(raw_tangent, psi_tangent, cpml)
    expected_raw, expected_psi = cpml_transpose(
        bar_corrected, bar_psi_next, cpml
    )
    actual_raw = ctypes.c_double(0.0)
    actual_psi = ctypes.c_double(0.0)
    status = compiled_stress_cpml_vjp.visco_sh_stress_cpml_local_vjp(
        1, cpml.K, cpml.a, cpml.b, bar_corrected, bar_psi_next,
        ctypes.byref(actual_raw), ctypes.byref(actual_psi),
    )
    assert status == 0
    assert relative_agreement(actual_raw.value, expected_raw) <= C2_DOUBLE_REFERENCE_RELATIVE_MAX
    assert relative_agreement(actual_psi.value, expected_psi) <= C2_DOUBLE_REFERENCE_RELATIVE_MAX
    lhs = corrected * bar_corrected + psi_next * bar_psi_next
    rhs = raw_tangent * actual_raw.value + psi_tangent * actual_psi.value
    residual = _relative_dot(lhs, rhs)
    assert residual <= C2_DOUBLE_DOT_RELATIVE_MAX
    print("M63C2_CPML " + json.dumps({"dot_residual": residual}, sort_keys=True))


def test_compiled_spatial_scatter_all_supported_orders(compiled_stress_cpml_vjp):
    randomizer = random.Random(63201)
    records = []
    for fdorder in SUPPORTED_FDORDERS:
        half_order = fdorder // 2
        side = 2 * half_order + 1
        center = half_order * side + half_order
        dh = 7.5
        hc = tuple(
            float(ctypes.c_float(value).value)
            for value in (
                (0.0,)
                + tuple(
                    ((-1.0) ** (m + 1)) * (1.1 / m)
                    for m in range(1, half_order + 1)
                )
            )
        )
        tangent = tuple(randomizer.uniform(-0.8, 0.8) for _ in range(side * side))
        bar_raw = (randomizer.uniform(-0.9, 0.9), randomizer.uniform(-0.9, 0.9))
        raw = spatial_forward(
            tangent, side=side, center=center, fdorder=fdorder, dh=dh, hc=hc
        )
        expected = spatial_transpose(
            patch=(0.0,) * (side * side), side=side, center=center,
            fdorder=fdorder, dh=dh, hc=hc,
            bar_ex=bar_raw[0], bar_ey=bar_raw[1],
        )
        actual = _doubles((0.0,) * (side * side))
        status = compiled_stress_cpml_vjp.visco_sh_stress_spatial_local_vjp(
            fdorder, dh, _floats(hc), bar_raw[0], bar_raw[1], actual,
            side, side, half_order, half_order,
        )
        assert status == 0
        error = max(relative_agreement(left, right) for left, right in zip(actual, expected))
        lhs = _inner(raw, bar_raw)
        rhs = _inner(tangent, actual)
        residual = _relative_dot(lhs, rhs)
        assert error <= C2_DOUBLE_REFERENCE_RELATIVE_MAX
        assert residual <= C2_DOUBLE_DOT_RELATIVE_MAX
        assert actual[center] == pytest.approx(expected[center], rel=0.0, abs=1.0e-15)
        records.append({"fdorder": fdorder, "dot_residual": residual, "reference_error": error})
    print("M63C2_SPATIAL " + json.dumps(records, sort_keys=True))


def _call_block(library, case, initial):
    coefficients = case["coefficients"]
    mechanisms = len(coefficients[0].recurrence)
    cpml = case["cpml"]
    arrays = {
        "bar_stress": _doubles(initial["bar_stress"]),
        "bar_memory_x": _doubles(initial["bar_memory"][0]),
        "bar_memory_y": _doubles(initial["bar_memory"][1]),
        "bar_psi": _doubles(initial["bar_psi"]),
        "bar_patch": _doubles(initial["bar_patch"]),
        "g_tau": _doubles(initial["g_tau"]),
        "g_modulus": _doubles(initial["g_modulus"]),
    }
    status = library.update_s_visc_PML_SH_adjoint_point(
        case["fdorder"], mechanisms, case["dh"], case["dt"], _floats(case["hc"]),
        _ints(tuple(int(value.active) for value in cpml)),
        _doubles(tuple(value.K for value in cpml)),
        _doubles(tuple(value.a for value in cpml)),
        _doubles(tuple(value.b for value in cpml)),
        _doubles(case["forward"]["corrected"]), _doubles(case["bar_stress_next"]),
        _doubles(case["bar_memory_next"][0]), _doubles(case["bar_memory_next"][1]),
        _doubles(tuple(value.stress for value in coefficients)),
        _doubles(coefficients[0].recurrence), _doubles(coefficients[1].recurrence),
        _doubles(coefficients[0].coupling), _doubles(coefficients[1].coupling),
        _doubles(tuple(value.stress_tau_derivative for value in coefficients)),
        _doubles(tuple(value.stress_modulus_derivative for value in coefficients)),
        _doubles(coefficients[0].coupling_tau_derivative),
        _doubles(coefficients[1].coupling_tau_derivative),
        _doubles(coefficients[0].coupling_modulus_derivative),
        _doubles(coefficients[1].coupling_modulus_derivative),
        _doubles(case["bar_psi_next"]), arrays["bar_stress"],
        arrays["bar_memory_x"], arrays["bar_memory_y"], arrays["bar_psi"],
        arrays["bar_patch"], case["side"], case["side"],
        case["half_order"], case["half_order"], arrays["g_tau"], arrays["g_modulus"],
    )
    assert status == 0
    return {
        "bar_stress_previous": tuple(arrays["bar_stress"]),
        "bar_memory_previous": (tuple(arrays["bar_memory_x"]), tuple(arrays["bar_memory_y"])),
        "bar_psi_previous": tuple(arrays["bar_psi"]),
        "bar_patch": tuple(arrays["bar_patch"]),
        "g_tau": tuple(arrays["g_tau"]),
        "g_modulus": tuple(arrays["g_modulus"]),
    }


def _make_case(name, fdorder, mechanisms, cpml, seed):
    randomizer = random.Random(seed)
    half_order = fdorder // 2
    side = 2 * half_order + 1
    center = half_order * side + half_order
    dt = 0.0004
    dh = 8.0
    hc = tuple(
        float(ctypes.c_float(value).value)
        for value in (
            (0.0,)
            + tuple(
                ((-1.0) ** (m + 1)) * (1.2 / m)
                for m in range(1, half_order + 1)
            )
        )
    )
    frequencies = tuple(5.0 * 1.8**index for index in range(mechanisms))
    coefficients = (
        local_gsls_coefficients(
            unrelaxed_shear_modulus=5.4e9, tau=0.035, dt=dt,
            relaxation_frequencies_hz=frequencies,
        ),
        local_gsls_coefficients(
            unrelaxed_shear_modulus=6.1e9, tau=0.052, dt=dt,
            relaxation_frequencies_hz=frequencies,
        ),
    )
    base_patch = tuple(randomizer.uniform(-0.8, 0.8) for _ in range(side * side))
    base_psi = tuple(randomizer.uniform(-0.4, 0.4) if value.active else 0.0 for value in cpml)
    forward = stress_block_forward(
        patch=base_patch, side=side, center=center, fdorder=fdorder, dh=dh, hc=hc,
        stress_previous=(0.2, -0.1),
        memory_previous=tuple(
            tuple(randomizer.uniform(-0.3, 0.3) for _ in range(mechanisms))
            for _ in range(2)
        ),
        psi_previous=base_psi, cpml=cpml, coefficients=coefficients,
    )
    return {
        "name": name, "fdorder": fdorder, "half_order": half_order,
        "side": side, "center": center, "dt": dt, "dh": dh, "hc": hc,
        "cpml": cpml, "coefficients": coefficients, "forward": forward,
        "bar_stress_next": tuple(randomizer.uniform(-0.9, 0.9) for _ in range(2)),
        "bar_memory_next": tuple(
            tuple(randomizer.uniform(-0.9, 0.9) for _ in range(mechanisms))
            for _ in range(2)
        ),
        "bar_psi_next": tuple(randomizer.uniform(-0.9, 0.9) if value.active else 0.0 for value in cpml),
        "randomizer": randomizer,
    }


def test_compiled_full_stress_side_block_matches_reference_and_dot_product(
    compiled_stress_cpml_vjp,
):
    inactive = CpmlBranch(False)
    x_cpml = CpmlBranch(True, K=1.41, a=-0.17, b=0.72)
    y_cpml = CpmlBranch(True, K=1.67, a=-0.13, b=0.66)
    cases = [
        _make_case(f"interior_fd{order}", order, 1 if order == 2 else 4,
                   (inactive, inactive), 632100 + order)
        for order in SUPPORTED_FDORDERS
    ]
    cases.extend(
        _make_case(name, 8, mechanisms, cpml, seed)
        for name, mechanisms, cpml, seed in (
            ("left_x", 1, (x_cpml, inactive), 632201),
            ("right_x", 4, (x_cpml, inactive), 632202),
            ("top_y_fs0", 1, (inactive, y_cpml), 632203),
            ("bottom_y", 4, (inactive, y_cpml), 632204),
            ("corner_xy", 4, (x_cpml, y_cpml), 632205),
            ("top_y_fs1_disabled", 1, (inactive, inactive), 632206),
        )
    )
    records = []
    for case in cases:
        randomizer = case["randomizer"]
        mechanisms = len(case["coefficients"][0].recurrence)
        initial = {
            "bar_stress": tuple(randomizer.uniform(-0.2, 0.2) for _ in range(2)),
            "bar_memory": tuple(
                tuple(randomizer.uniform(-0.2, 0.2) for _ in range(mechanisms))
                for _ in range(2)
            ),
            "bar_psi": tuple(randomizer.uniform(-0.2, 0.2) for _ in range(2)),
            "bar_patch": tuple(randomizer.uniform(-0.2, 0.2) for _ in range(case["side"] ** 2)),
            "g_tau": tuple(randomizer.uniform(-0.2, 0.2) for _ in range(2)),
            "g_modulus": tuple(randomizer.uniform(-0.2, 0.2) for _ in range(2)),
        }
        expected = stress_block_transpose(
            initial_patch=initial["bar_patch"], side=case["side"], center=case["center"],
            fdorder=case["fdorder"], dh=case["dh"], hc=case["hc"],
            corrected_strain=case["forward"]["corrected"],
            bar_stress_next=case["bar_stress_next"],
            bar_memory_next=case["bar_memory_next"], bar_psi_next=case["bar_psi_next"],
            cpml=case["cpml"], coefficients=case["coefficients"],
            initial_stress=initial["bar_stress"], initial_memory=initial["bar_memory"],
            initial_psi=initial["bar_psi"], initial_g_tau=initial["g_tau"],
            initial_g_modulus=initial["g_modulus"],
        )
        actual = _call_block(compiled_stress_cpml_vjp, case, initial)

        errors = {
            "bar_vz": max(relative_agreement(a, b) for a, b in zip(actual["bar_patch"], expected["bar_patch"])),
            "stress": max(relative_agreement(a, b) for a, b in zip(actual["bar_stress_previous"], expected["bar_stress_previous"])),
            "memory": max(relative_agreement(a, b) for axis_a, axis_b in zip(actual["bar_memory_previous"], expected["bar_memory_previous"]) for a, b in zip(axis_a, axis_b)),
            "cpml": max(relative_agreement(a, b) for a, b in zip(actual["bar_psi_previous"], expected["bar_psi_previous"])),
            "g_tau": max(relative_agreement(a, b) for a, b in zip(actual["g_tau"], expected["g_tau"])),
            "g_M": max(relative_agreement(a, b) for a, b in zip(actual["g_modulus"], expected["g_modulus"])),
        }
        assert max(errors.values()) <= C2_DOUBLE_REFERENCE_RELATIVE_MAX

        tangent_patch = tuple(randomizer.uniform(-0.7, 0.7) for _ in range(case["side"] ** 2))
        tangent_stress = tuple(randomizer.uniform(-0.7, 0.7) for _ in range(2))
        tangent_memory = tuple(
            tuple(randomizer.uniform(-0.7, 0.7) for _ in range(mechanisms))
            for _ in range(2)
        )
        tangent_psi = tuple(randomizer.uniform(-0.7, 0.7) if value.active else 0.0 for value in case["cpml"])
        tangent_forward = stress_block_forward(
            patch=tangent_patch, side=case["side"], center=case["center"],
            fdorder=case["fdorder"], dh=case["dh"], hc=case["hc"],
            stress_previous=tangent_stress, memory_previous=tangent_memory,
            psi_previous=tangent_psi, cpml=case["cpml"], coefficients=case["coefficients"],
        )
        zero = {
            "bar_stress": (0.0, 0.0),
            "bar_memory": ((0.0,) * mechanisms, (0.0,) * mechanisms),
            "bar_psi": (0.0, 0.0),
            "bar_patch": (0.0,) * (case["side"] ** 2),
            "g_tau": (0.0, 0.0), "g_modulus": (0.0, 0.0),
        }
        transpose = _call_block(compiled_stress_cpml_vjp, case, zero)
        lhs = _inner(tangent_forward["stress_next"], case["bar_stress_next"])
        for axis in range(2):
            lhs += _inner(tangent_forward["memory_next"][axis], case["bar_memory_next"][axis])
            if case["cpml"][axis].active:
                lhs += tangent_forward["psi_next"][axis] * case["bar_psi_next"][axis]
        rhs = _inner(tangent_patch, transpose["bar_patch"]) + _inner(tangent_stress, transpose["bar_stress_previous"])
        for axis in range(2):
            rhs += _inner(tangent_memory[axis], transpose["bar_memory_previous"][axis])
            if case["cpml"][axis].active:
                rhs += tangent_psi[axis] * transpose["bar_psi_previous"][axis]
        dot = _relative_dot(lhs, rhs)
        assert dot <= C2_DOUBLE_DOT_RELATIVE_MAX
        records.append({"case": case["name"], "fdorder": case["fdorder"], "L": mechanisms, "dot_residual": dot, **errors})
    print("M63C2_BLOCK " + json.dumps(records, sort_keys=True))
