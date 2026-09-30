"""Exact segmented checkpoint/replay gates for the M9b-1 Born operator."""

from __future__ import annotations

import ctypes
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tests.physics import test_m9b1_elastic_psv_born_production as m9b1
from tests.utilities.elastic_psv_born_reference import (
    born_adjoint,
    born_forward,
    deterministic_data,
    deterministic_model_direction,
    dot_metrics,
    make_experiment,
)


class StorageDiagnostics(ctypes.Structure):
    _fields_ = [
        ("segmented", ctypes.c_int),
        ("segment_count", ctypes.c_int),
        ("checkpoint_count", ctypes.c_int),
        ("max_segment_length", ctypes.c_int),
        ("checkpoint_payload_bytes", ctypes.c_size_t),
        ("checkpoint_bytes", ctypes.c_size_t),
        ("segment_operand_bytes", ctypes.c_size_t),
        ("retained_replay_bytes", ctypes.c_size_t),
        ("forward_working_bytes", ctypes.c_size_t),
        ("adjoint_working_bytes", ctypes.c_size_t),
        ("initial_forward_steps", ctypes.c_size_t),
        ("replayed_forward_steps_last", ctypes.c_size_t),
        ("checkpoint_metadata_bytes", ctypes.c_size_t),
        ("checkpoint_pointer_bytes", ctypes.c_size_t),
        ("segment_schedule_bytes", ctypes.c_size_t),
    ]


@pytest.fixture(scope="session")
def replay_library(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("cc") or shutil.which("gcc")
    assert compiler, "a C99 compiler is required for the M9d1 gate"
    output = tmp_path_factory.mktemp("m9d1") / "libdenise_m9d1.so"
    sanitizer = os.environ.get("DENISE_M9D1_SANITIZE") == "1"
    subprocess.run([
        compiler, "-std=c99", "-O1" if sanitizer else "-O2",
        "-Wall", "-Wextra", "-Werror",
        "-pedantic", "-fPIC", "-shared", "-I", str(repository_root / "include"),
        str(repository_root / "src/PSV/elastic_psv_born.c"),
        *(["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
          if sanitizer else []),
        "-lm", "-o", str(output),
    ], check=True)
    api = ctypes.CDLL(str(output))
    api.denise_elastic_psv_born_last_error.restype = ctypes.c_char_p
    api.denise_elastic_psv_born_create.argtypes = [
        ctypes.POINTER(m9b1.Config), ctypes.POINTER(ctypes.c_void_p)]
    api.denise_elastic_psv_born_create.restype = ctypes.c_int
    api.denise_elastic_psv_born_prepare.argtypes = [ctypes.c_void_p, m9b1.F32P]
    api.denise_elastic_psv_born_prepare.restype = ctypes.c_int
    api.denise_elastic_psv_born_apply_j.argtypes = [
        ctypes.c_void_p, m9b1.F32P, m9b1.F32P, m9b1.F32P]
    api.denise_elastic_psv_born_apply_j.restype = ctypes.c_int
    api.denise_elastic_psv_born_apply_jt.argtypes = [
        ctypes.c_void_p, m9b1.F32P, m9b1.F64P, m9b1.F64P]
    api.denise_elastic_psv_born_apply_jt.restype = ctypes.c_int
    api.denise_elastic_psv_born_copy_strain.argtypes = [
        ctypes.c_void_p, ctypes.c_int, m9b1.F32P]
    api.denise_elastic_psv_born_copy_strain.restype = ctypes.c_int
    api.denise_elastic_psv_born_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    api.denise_elastic_psv_born_set_replay_segments.argtypes = [
        ctypes.c_void_p, ctypes.c_int]
    api.denise_elastic_psv_born_set_replay_segments.restype = ctypes.c_int
    api.denise_elastic_psv_born_estimate_replay_storage.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
    api.denise_elastic_psv_born_estimate_replay_storage.restype = ctypes.c_int
    api.denise_elastic_psv_born_get_segment_bounds.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int)]
    api.denise_elastic_psv_born_get_segment_bounds.restype = ctypes.c_int
    api.denise_elastic_psv_born_storage_diagnostics.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(StorageDiagnostics)]
    api.denise_elastic_psv_born_storage_diagnostics.restype = ctypes.c_int
    api.denise_elastic_psv_born_checkpoint_roundtrip.argtypes = [
        ctypes.c_void_p, ctypes.c_int]
    api.denise_elastic_psv_born_checkpoint_roundtrip.restype = ctypes.c_int
    return api


def _segmented(api, exp, segments, shot=0):
    operator = m9b1.Operator(api, exp, shot=shot)
    status = api.denise_elastic_psv_born_set_replay_segments(
        operator.context, segments)
    if status:
        operator.close()
        raise RuntimeError(api.denise_elastic_psv_born_last_error().decode())
    return operator


def _bounds(api, operator, count):
    result = []
    for segment in range(count):
        start, end = ctypes.c_int(), ctypes.c_int()
        assert api.denise_elastic_psv_born_get_segment_bounds(
            operator.context, segment, ctypes.byref(start), ctypes.byref(end)) == 0
        result.append((start.value, end.value))
    return result


def _strain(api, operator, timestep):
    out = np.empty((4, operator.exp.ny, operator.exp.nx), dtype=np.float32)
    assert api.denise_elastic_psv_born_copy_strain(
        operator.context, timestep, m9b1._ptr(out)) == 0
    return out


@pytest.mark.parametrize("segments", (1, 2, 3, 7, 32, 37, 100))
def test_deterministic_segment_schedule_has_exact_cover(replay_library, segments):
    exp = replace(make_experiment("interior"), nt=37)
    operator = _segmented(replay_library, exp, segments)
    try:
        count = min(segments, exp.nt)
        bounds = _bounds(replay_library, operator, count)
        assert bounds == [
            (segment * exp.nt // count, (segment + 1) * exp.nt // count)
            for segment in range(count)]
        assert [time for start, end in bounds for time in range(start, end)] == list(
            range(exp.nt))
    finally:
        operator.close()


def test_checkpoint_restore_continues_bit_identically_with_active_cpml(
        replay_library):
    exp = make_experiment("active_cpml")
    operator = m9b1.Operator(replay_library, exp)
    try:
        assert replay_library.denise_elastic_psv_born_checkpoint_roundtrip(
            operator.context, 347) == 0
    finally:
        operator.close()


@pytest.mark.parametrize(("case", "segments"), (
    ("interior", 7), ("active_cpml", 32)))
def test_full_and_segmented_strain_j_and_jt_are_bit_identical(
        replay_library, case, segments):
    exp = make_experiment(case)
    full = m9b1.Operator(replay_library, exp)
    segmented = _segmented(replay_library, exp, segments)
    try:
        np.testing.assert_array_equal(segmented.prepare(), full.prepare())
        bounds = _bounds(replay_library, segmented, segments)
        probe = {0, exp.nt - 1}
        for start, end in bounds:
            probe.update((start, end - 1))
            if end - start > 1:
                probe.add(end - 2)
            if start > 0:
                probe.add(start - 1)
        for timestep in sorted(probe):
            np.testing.assert_array_equal(
                _strain(replay_library, segmented, timestep),
                _strain(replay_library, full, timestep))

        dlam, dmu = deterministic_model_direction(exp)
        zero = np.zeros_like(dlam)
        for dl, dm in ((dlam, zero), (zero, dmu), (dlam, dmu)):
            np.testing.assert_array_equal(segmented.j(dl, dm), full.j(dl, dm))
        for component in (0, 1, None):
            data = deterministic_data(exp, component=component)
            segmented_image = segmented.jt(data)
            full_image = full.jt(data)
            np.testing.assert_array_equal(segmented_image[0], full_image[0])
            np.testing.assert_array_equal(segmented_image[1], full_image[1])
    finally:
        segmented.close()
        full.close()


@pytest.mark.parametrize("segments", (1, 2, 3, 7, 32, 37))
def test_segment_count_matrix_replays_exact_strains(replay_library, segments):
    exp = replace(make_experiment("interior"), nt=37)
    full = m9b1.Operator(replay_library, exp)
    segmented = _segmented(replay_library, exp, segments)
    try:
        full.prepare()
        segmented.prepare()
        diagnostics = StorageDiagnostics()
        assert replay_library.denise_elastic_psv_born_storage_diagnostics(
            segmented.context, ctypes.byref(diagnostics)) == 0
        assert diagnostics.checkpoint_payload_bytes == 5 * exp.nx * exp.ny * 4
        for start, end in _bounds(replay_library, segmented, segments):
            for timestep in {start, end - 1}:
                np.testing.assert_array_equal(
                    _strain(replay_library, segmented, timestep),
                    _strain(replay_library, full, timestep))
    finally:
        segmented.close()
        full.close()


@pytest.mark.parametrize("case", ("interior", "active_cpml"))
def test_complete_18_case_dot_matrix_in_segmented_mode(replay_library, case):
    exp = make_experiment(case)
    reference = m9b1.nonlinear_forward(exp, 0, save_strain=True)
    operator = _segmented(replay_library, exp, min(32, exp.nt))
    operator.prepare()
    try:
        dlam, dmu = deterministic_model_direction(exp)
        zero = np.zeros_like(dlam)
        for dl, dm in ((dlam, zero), (zero, dmu), (dlam, dmu)):
            dl = np.asarray(m9b1._f32(dl), dtype=np.float64)
            dm = np.asarray(m9b1._f32(dm), dtype=np.float64)
            jdm = operator.j(dl, dm).astype(np.float64)
            reference_jdm = born_forward(exp, reference, dl, dm)
            for component in (0, 1, None):
                data = deterministic_data(exp, component=component)
                glambda, gmu = operator.jt(data)
                reference_image = born_adjoint(exp, reference, data)
                metrics = m9b1._production_dot_metrics(
                    jdm, data, dl, dm, glambda, gmu, reference_jdm,
                    reference_image.image_lambda_raw,
                    reference_image.image_mu_raw,
                    dot_metrics(exp, dl, dm, data, reference))
                m9b1._assert_production_dot_closes(metrics)
    finally:
        operator.close()


def test_canonical_active_cpml_storage_and_recomputation_diagnostics(
        replay_library):
    exp = make_experiment("active_cpml")
    operator = _segmented(replay_library, exp, 32)
    try:
        estimate = ctypes.c_size_t()
        assert replay_library.denise_elastic_psv_born_estimate_replay_storage(
            operator.context, 32, ctypes.byref(estimate)) == 0
        assert estimate.value == 4_467_472
        operator.prepare()
        operator.jt(deterministic_data(exp))
        diagnostics = StorageDiagnostics()
        assert replay_library.denise_elastic_psv_born_storage_diagnostics(
            operator.context, ctypes.byref(diagnostics)) == 0
        assert diagnostics.segmented == 1
        assert diagnostics.segment_count == 32
        assert diagnostics.checkpoint_count == 31
        assert diagnostics.max_segment_length == 22
        assert diagnostics.checkpoint_payload_bytes == 103_360
        assert diagnostics.checkpoint_bytes == 3_204_160
        assert diagnostics.segment_operand_bytes == 1_261_568
        assert diagnostics.checkpoint_metadata_bytes == 1_488
        assert diagnostics.checkpoint_pointer_bytes == 0
        assert diagnostics.segment_schedule_bytes == 256
        assert diagnostics.retained_replay_bytes == 4_467_472
        assert diagnostics.retained_replay_bytes < 40_140_800 // 4
        assert diagnostics.forward_working_bytes == 258_048
        assert diagnostics.adjoint_working_bytes == 430_080
        assert diagnostics.initial_forward_steps == 700
        assert diagnostics.replayed_forward_steps_last == 700
    finally:
        operator.close()


def test_segmented_lifecycle_and_failure_paths(replay_library):
    exp = make_experiment("interior")
    dlam_a, dmu_a = deterministic_model_direction(exp, seed=31)
    dlam_b, dmu_b = deterministic_model_direction(exp, seed=73)
    data_a = deterministic_data(exp, seed=29)
    data_b = deterministic_data(exp, seed=101)
    first = _segmented(replay_library, exp, 7)
    fresh = _segmented(replay_library, exp, 7)
    try:
        assert replay_library.denise_elastic_psv_born_set_replay_segments(
            first.context, 0) != 0
        first.prepare()
        first_j_a = first.j(dlam_a, dmu_a)
        first_jt_a = first.jt(data_a)
        repeated_jt_a = first.jt(data_a)
        np.testing.assert_array_equal(repeated_jt_a[0], first_jt_a[0])
        np.testing.assert_array_equal(repeated_jt_a[1], first_jt_a[1])
        first_j_b = first.j(dlam_b, dmu_b)
        first_jt_b = first.jt(data_b)
        np.testing.assert_array_equal(first.j(dlam_a, dmu_a), first_j_a)
        assert replay_library.denise_elastic_psv_born_set_replay_segments(
            first.context, 3) != 0

        fresh.prepare()
        np.testing.assert_array_equal(fresh.j(dlam_b, dmu_b), first_j_b)
        fresh_jt_b = fresh.jt(data_b)
        np.testing.assert_array_equal(fresh_jt_b[0], first_jt_b[0])
        np.testing.assert_array_equal(fresh_jt_b[1], first_jt_b[1])
    finally:
        fresh.close()
        first.close()


def test_create_rejects_oversized_storage_before_allocation(replay_library):
    exp = make_experiment("interior")
    with pytest.raises(RuntimeError, match="grid exceeds the supported flat index range"):
        m9b1.Operator(
            replay_library, exp,
            nx=46_341, ny=46_341, nt=2,
            fw=0, cpml_enabled=0,
        )
