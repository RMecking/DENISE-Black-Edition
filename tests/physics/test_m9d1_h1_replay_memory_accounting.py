"""Metadata-inclusive requested-allocation ledger and exact backend choice."""
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
from tests.physics import test_m9c_elastic_psv_migration_driver as m9c
from tests.physics import test_m9d1_elastic_psv_checkpoint_replay as m9d1
from tests.physics.test_m9c_elastic_psv_migration_driver import migration_library
from tests.physics.test_m9d1_elastic_psv_checkpoint_replay import replay_library
from tests.utilities.elastic_psv_born_reference import (
    deterministic_data, deterministic_model_direction, make_experiment,
)


class Checkpoint(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in (
        "layout_version", "timestep", "nx", "ny", "fw", "fdorder",
        "cpml_enabled")] + [
        ("value_count", ctypes.c_size_t), ("values", m9b1.F32P)]


def _small(nx=5, ny=5, nt=41):
    exp = make_experiment("interior")
    return replace(exp, nx=nx, ny=ny, nt=nt,
                   lam=np.full((ny, nx), exp.lam[0, 0]),
                   mu=np.full((ny, nx), exp.mu[0, 0]),
                   rho=np.full((ny, nx), exp.rho[0, 0]),
                   sources=((3, 3),), receivers=((2, 3), (3, 3), (4, 3)))


def _diagnostics(api, operator):
    result = m9d1.StorageDiagnostics()
    assert api.denise_elastic_psv_born_storage_diagnostics(
        operator.context, ctypes.byref(result)) == 0
    return result


@pytest.fixture(scope="session")
def accounting_unit_library(tmp_path_factory, repository_root):
    """Exercise private production arithmetic with synthetic size_t inputs."""
    directory = tmp_path_factory.mktemp("m9d1_h1_units")
    born = directory / "born_units.c"
    migration = directory / "migration_units.c"
    born.write_text(f'#include "{repository_root}/src/PSV/elastic_psv_born.c"\n' + r'''
size_t h1_checkpoint_size(void) { return sizeof(struct elastic_checkpoint); }
int h1_account(size_t cells, size_t payload, size_t segments, size_t length,
               size_t *out) {
    struct replay_storage_estimate e;
    int status = calculate_retained_replay_sizes(cells,payload,segments,length,&e);
    memset(out,0,8*sizeof(*out));
    if(status) return status;
    out[0]=e.payload_bytes; out[1]=e.checkpoint_bytes;
    out[2]=e.checkpoint_metadata_bytes; out[3]=e.checkpoint_pointer_bytes;
    out[4]=e.segment_schedule_bytes; out[5]=e.operand_bytes;
    out[6]=e.retained_bytes; out[7]=e.payload_values;
    return 0;
}
int h1_product(size_t count,size_t width,size_t *out) {
    return checked_product(count,width,out);
}
int h1_schedule(size_t count,size_t *out) {
    return calculate_schedule_bytes(count,out);
}
''', encoding="utf-8")
    migration.write_text(
        f'#include "{repository_root}/src/PSV/elastic_psv_migration.c"\n' + r'''
int h1_select(int status,size_t replay,size_t full,int *selected) {
    return select_replay_backend(status,replay,full,selected);
}
int h1_full(size_t cells,size_t nt,size_t *bytes) {
    return full_trajectory_size(cells,nt,bytes);
}
''', encoding="utf-8")
    output = directory / "libh1_units.so"
    sanitizer = os.environ.get("DENISE_M9D1_SANITIZE") == "1"
    subprocess.run([
        shutil.which("cc"), "-std=c99", "-O1", "-Wall", "-Wextra",
        "-Werror", "-pedantic", "-fPIC", "-shared",
        "-I", str(repository_root / "include"), str(born), str(migration),
        *(["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
          if sanitizer else []), "-lm", "-o", str(output)], check=True)
    api = ctypes.CDLL(str(output))
    size = ctypes.c_size_t
    sizep = ctypes.POINTER(size)
    api.h1_checkpoint_size.restype = size
    api.h1_account.argtypes = [size, size, size, size, sizep]
    api.h1_product.argtypes = [size, size, sizep]
    api.h1_schedule.argtypes = [size, sizep]
    api.h1_full.argtypes = [size, size, sizep]
    api.h1_select.argtypes = [ctypes.c_int, size, size,
                             ctypes.POINTER(ctypes.c_int)]
    return api


@pytest.mark.parametrize("case", ("loses", "wins", "narrow"))
def test_complete_ledger_estimate_and_production_choice(
        replay_library, migration_library, accounting_unit_library, case):
    exp = (make_experiment("active_cpml") if case == "wins"
           else _small(6, 73) if case == "narrow" else _small())
    segments = min(32, exp.nt)
    maximum = (exp.nt + segments - 1) // segments
    cells = exp.nx * exp.ny
    payload = (5 * cells + ((8 * exp.fw + 2) * (exp.nx + exp.ny)
                           if exp.cpml else 0)) * 4
    assert accounting_unit_library.h1_checkpoint_size() == ctypes.sizeof(Checkpoint)
    objects = (segments - 1) * ctypes.sizeof(Checkpoint)
    schedule = 2 * segments * ctypes.sizeof(ctypes.c_int)
    operand = 4 * maximum * cells * 4
    total = (segments - 1) * payload + objects + schedule + operand
    full = 4 * exp.nt * cells * 4
    operator = m9d1._segmented(replay_library, exp, segments)
    try:
        estimate = ctypes.c_size_t(123)
        assert replay_library.denise_elastic_psv_born_estimate_replay_storage(
            operator.context, segments, ctypes.byref(estimate)) == 0
        assert estimate.value == total
        # Before prepare, only the two schedule arrays are retained.
        before = _diagnostics(replay_library, operator)
        assert before.retained_replay_bytes == before.segment_schedule_bytes == schedule
        operator.prepare()
        d = _diagnostics(replay_library, operator)
        assert d.checkpoint_payload_bytes == payload
        assert d.checkpoint_bytes == (segments - 1) * payload
        assert d.checkpoint_metadata_bytes == objects
        assert d.checkpoint_pointer_bytes == 0  # Contiguous objects, no pointer table.
        assert d.segment_schedule_bytes == schedule
        assert d.segment_operand_bytes == operand
        assert d.retained_replay_bytes == total == estimate.value
    finally:
        operator.close()
    if case == "loses":
        assert (segments - 1) * payload + operand == 16_300 < full == 16_400
        assert total == 18_044 > full
    elif case == "wins":
        assert full == 40_140_800 and total == 4_467_472 < full // 4
    else:
        assert full == 287_328 and total == 287_320  # Exactly 8 bytes benefit.

    data = [deterministic_data(exp, seed=37 + shot)
            for shot in range(len(exp.sources))]
    actual, diagnostics = m9c._run(migration_library, m9c.RequestOwner(exp, data))
    assert diagnostics["segment_count"] == (segments if total < full else 0)
    assert diagnostics["peak_replay_storage_bytes"] == (total if total < full else 0)
    expected = [np.zeros((exp.ny, exp.nx)), np.zeros((exp.ny, exp.nx))]
    for shot, values in enumerate(data):
        reference = m9b1.Operator(replay_library, exp, shot=shot)
        try:
            reference.prepare()
            images = reference.jt(values)
            expected[0] += images[0]
            expected[1] += images[1]
        finally:
            reference.close()
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize("segments", (1, 41))
def test_s1_and_snt_forced_replay_include_all_metadata(replay_library, segments):
    exp = _small()
    operator = m9d1._segmented(replay_library, exp, segments)
    full = 16 * exp.nt * exp.nx * exp.ny
    try:
        estimate = ctypes.c_size_t()
        assert replay_library.denise_elastic_psv_born_estimate_replay_storage(
            operator.context, segments, ctypes.byref(estimate)) == 0
        operator.prepare()
        d = _diagnostics(replay_library, operator)
        assert d.retained_replay_bytes == estimate.value > full
        assert d.retained_replay_bytes == (16_408 if segments == 1 else 22_648)
        assert d.checkpoint_count == segments - 1
    finally:
        operator.close()


@pytest.mark.parametrize("status,replay,full,expected", [
    (0, 100, 100, 0), (0, 99, 100, 1), (0, 101, 100, 0),
    (-1, 99, 100, None), (0, 0, 100, None), (0, 99, 0, None),
])
def test_exact_strict_selection_and_invalid_estimate(
        accounting_unit_library, status, replay, full, expected):
    selected = ctypes.c_int(17)
    result = accounting_unit_library.h1_select(
        status, replay, full, ctypes.byref(selected))
    assert selected.value == (0 if expected is None else expected)
    assert (result != 0) is (expected is None)


def test_near_size_max_arithmetic_rejects_without_allocating(accounting_unit_library):
    api = accounting_unit_library
    maximum = ctypes.c_size_t(-1).value
    output = ctypes.c_size_t()
    # Actual checkpoint objects; no checkpoint pointer table is allocated.
    assert api.h1_product(maximum // api.h1_checkpoint_size() + 1,
                          api.h1_checkpoint_size(), ctypes.byref(output)) != 0
    assert api.h1_product(maximum // ctypes.sizeof(ctypes.c_void_p) + 1,
                          ctypes.sizeof(ctypes.c_void_p), ctypes.byref(output)) != 0
    assert api.h1_schedule(maximum // ctypes.sizeof(ctypes.c_int) + 1,
                           ctypes.byref(output)) != 0
    # Individual start/end arrays fit, but their sum does not.
    assert api.h1_schedule(maximum // (2 * ctypes.sizeof(ctypes.c_int)) + 1,
                           ctypes.byref(output)) != 0
    out = (ctypes.c_size_t * 8)(*([17] * 8))
    for cells, payload, segments, length in (
        (1, 1, maximum // api.h1_checkpoint_size() + 2, 1),
        (maximum // 16, 1, 2, 1),  # Every component fits; retained sum overflows.
        (1, maximum // 4 + 1, 1, 1),
        (1, 1, 0, 1),
    ):
        assert api.h1_account(cells, payload, segments, length, out) != 0
        assert list(out) == [0] * 8
    output.value = 17
    assert api.h1_full(maximum // 4 + 1, 1, ctypes.byref(output)) != 0
    assert output.value == 0
    assert api.h1_full(maximum // 16, 2, ctypes.byref(output)) != 0
    assert output.value == 0


def test_public_failed_estimate_has_no_valid_looking_value(replay_library):
    operator = m9b1.Operator(replay_library, _small())
    try:
        estimate = ctypes.c_size_t(17)
        assert replay_library.denise_elastic_psv_born_estimate_replay_storage(
            operator.context, 0, ctypes.byref(estimate)) != 0
        assert estimate.value == 0
        assert _diagnostics(replay_library, operator).segment_count == 0
    finally:
        operator.close()


@pytest.fixture(scope="session")
def pre_h1_root():
    value = os.environ.get("DENISE_M9D1_PRE_H1_ROOT")
    if not value:
        pytest.skip("set DENISE_M9D1_PRE_H1_ROOT for the frozen pre-H1 differential gate")
    return Path(value)


@pytest.fixture(scope="session")
def pre_h1_library(tmp_path_factory, pre_h1_root):
    return m9d1.replay_library.__wrapped__(tmp_path_factory, pre_h1_root)


@pytest.mark.parametrize("case", ("interior", "active_cpml"))
@pytest.mark.parametrize("segmented", (False, True))
def test_forced_backends_are_bit_identical_to_pre_h1(
        replay_library, pre_h1_library, case, segmented):
    exp = make_experiment(case)
    before = m9b1.Operator(pre_h1_library, exp)
    after = m9b1.Operator(replay_library, exp)
    try:
        if segmented:
            for api, operator in ((pre_h1_library, before), (replay_library, after)):
                assert api.denise_elastic_psv_born_set_replay_segments(
                    operator.context, min(32, exp.nt)) == 0
        np.testing.assert_array_equal(before.prepare(), after.prepare())
        for step in (0, exp.nt // 2, exp.nt - 1):
            np.testing.assert_array_equal(m9d1._strain(pre_h1_library, before, step),
                                          m9d1._strain(replay_library, after, step))
        dl, dm = deterministic_model_direction(exp)
        np.testing.assert_array_equal(before.j(dl, dm), after.j(dl, dm))
        image_before = before.jt(deterministic_data(exp))
        image_after = after.jt(deterministic_data(exp))
        for a, b in zip(image_before, image_after):
            np.testing.assert_array_equal(a, b)
    finally:
        before.close()
        after.close()


@pytest.mark.integration
@pytest.mark.optional_prerequisite
@pytest.mark.parametrize("case,backend", [
    ("loses", "FULL"), ("narrow", "SEGMENTED"),
    ("interior", "FULL"), ("active_cpml", "SEGMENTED"),
])
def test_mode2_explicit_backend_and_pre_h1_image_identity(
        tmp_path, replay_library, denise_binary, mpiexec, pre_h1_root, case, backend):
    exp = (_small() if case == "loses" else _small(6, 73) if case == "narrow"
           else make_experiment(case))
    data = [deterministic_data(exp, seed=37 + shot)
            for shot in range(len(exp.sources))]
    before_directory = tmp_path / "before"
    after_directory = tmp_path / "after"
    paths_before = m9c._write_mode2_case(before_directory, exp, data)
    paths_after = m9c._write_mode2_case(after_directory, exp, data)
    before = m9c._run_denise(before_directory, pre_h1_root / "bin/denise", mpiexec)
    after = m9c._run_denise(after_directory, denise_binary, mpiexec)
    assert before.returncode == 0, before.stdout
    assert after.returncode == 0, after.stdout
    assert f"trajectory backend: {backend}" in after.stdout
    if case == "loses":
        assert "replay segments/checkpoints/max length: 32 / 31 / 2" in before.stdout
        assert "replay segments/checkpoints/max length: 0 / 0 / 0" in after.stdout
    if case == "active_cpml":
        assert "replay object/pointer/schedule bytes: 1488 / 0 / 256" in after.stdout
        assert "segment operand/retained replay bytes: 1261568 / 4467472" in after.stdout
    expected = [np.zeros((exp.ny, exp.nx)), np.zeros((exp.ny, exp.nx))]
    for shot, values in enumerate(data):
        full = m9b1.Operator(replay_library, exp, shot=shot)
        try:
            full.prepare()
            images = full.jt(values)
            expected[0] += images[0]
            expected[1] += images[1]
        finally:
            full.close()
    for component, reference in zip(("lambda", "mu"), expected):
        assert paths_before[component].read_bytes() == paths_after[component].read_bytes()
        actual = np.fromfile(paths_after[component], dtype=np.float64).reshape(reference.shape)
        np.testing.assert_array_equal(actual, reference)
