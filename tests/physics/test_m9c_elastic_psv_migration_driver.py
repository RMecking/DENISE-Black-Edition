"""Scientific and lifecycle gates for the clean M9c migration driver."""

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
from tests.cases.homogeneous_psv import HomogeneousPSVConfig, _parameter_lines
from tests.utilities.elastic_psv_born_reference import (
    _source_samples,
    born_forward,
    lambda_reflector,
    make_experiment,
    migrate_shot_stack,
    nonlinear_forward,
)


F32P = ctypes.POINTER(ctypes.c_float)
F64P = ctypes.POINTER(ctypes.c_double)
I32P = ctypes.POINTER(ctypes.c_int)


class MigrationShot(ctypes.Structure):
    _fields_ = [
        ("physical_shot_index", ctypes.c_int),
        ("source_type", ctypes.c_int),
        ("source_i", ctypes.c_int),
        ("source_j", ctypes.c_int),
        ("source_samples", F32P),
        ("receiver_count", ctypes.c_int),
        ("receiver_i", I32P),
        ("receiver_j", I32P),
        ("migration_data", F32P),
    ]


class MigrationRequest(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_int), ("ny", ctypes.c_int), ("nt", ctypes.c_int),
        ("fw", ctypes.c_int), ("dh", ctypes.c_float), ("dt", ctypes.c_float),
        ("l", ctypes.c_int), ("invmat1", ctypes.c_int),
        ("fdorder", ctypes.c_int), ("ndt", ctypes.c_int),
        ("dtinv", ctypes.c_int), ("free_surface", ctypes.c_int),
        ("boundary", ctypes.c_int), ("mpi_size", ctypes.c_int),
        ("receiver_components", ctypes.c_int), ("inv_stf", ctypes.c_int),
        ("lambda_", F32P), ("mu", F32P), ("rho", F32P),
        ("cpml_enabled", ctypes.c_int),
        ("pml_reflection", ctypes.c_float), ("pml_power", ctypes.c_float),
        ("pml_kmax", ctypes.c_float), ("pml_fpml", ctypes.c_float),
        ("pml_damping_speed", ctypes.c_float),
        ("shot_count", ctypes.c_int),
        ("shots", ctypes.POINTER(MigrationShot)),
    ]


class MigrationResult(ctypes.Structure):
    _fields_ = [
        ("image_lambda_raw", F64P), ("image_mu_raw", F64P),
        ("cell_count", ctypes.c_size_t), ("shots_completed", ctypes.c_int),
        ("cpml_memory_peak", ctypes.c_float),
        ("trajectory_bytes", ctypes.c_size_t),
        ("global_image_bytes", ctypes.c_size_t),
        ("maximum_shot_data_bytes", ctypes.c_size_t),
        ("checkpoint_payload_bytes", ctypes.c_size_t),
        ("checkpoint_bytes", ctypes.c_size_t),
        ("segment_operand_bytes", ctypes.c_size_t),
        ("peak_replay_storage_bytes", ctypes.c_size_t),
        ("forward_working_bytes", ctypes.c_size_t),
        ("adjoint_working_bytes", ctypes.c_size_t),
        ("initial_forward_steps", ctypes.c_size_t),
        ("replayed_steps", ctypes.c_size_t),
        ("segment_count", ctypes.c_int),
        ("checkpoint_count", ctypes.c_int),
        ("max_segment_length", ctypes.c_int),
        ("checkpoint_metadata_bytes", ctypes.c_size_t),
        ("checkpoint_pointer_bytes", ctypes.c_size_t),
        ("segment_schedule_bytes", ctypes.c_size_t),
    ]


def _ptr(array, pointer_type=F32P):
    return array.ctypes.data_as(pointer_type)


@pytest.fixture(scope="session")
def migration_library(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("cc") or shutil.which("gcc")
    assert compiler, "a C99 compiler is required for the M9c driver gate"
    output = tmp_path_factory.mktemp("m9c") / "libdenise_m9c.so"
    subprocess.run([
        compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
        "-pedantic", "-fPIC", "-shared", "-I", str(repository_root / "include"),
        str(repository_root / "src/PSV/elastic_psv_born.c"),
        str(repository_root / "src/PSV/elastic_psv_migration.c"),
        "-lm", "-o", str(output),
    ], check=True)
    api = ctypes.CDLL(str(output))
    api.denise_elastic_psv_migration_last_error.restype = ctypes.c_char_p
    api.denise_elastic_psv_migrate.argtypes = [
        ctypes.POINTER(MigrationRequest), ctypes.POINTER(MigrationResult)]
    api.denise_elastic_psv_migrate.restype = ctypes.c_int
    api.denise_elastic_psv_migration_result_destroy.argtypes = [
        ctypes.POINTER(MigrationResult)]
    api.denise_elastic_psv_migration_pack_components.argtypes = [
        F32P, F32P, ctypes.c_int, ctypes.c_int, F32P]
    api.denise_elastic_psv_migration_pack_components.restype = ctypes.c_int
    api.denise_elastic_psv_born_last_error.restype = ctypes.c_char_p
    api.denise_elastic_psv_born_create.argtypes = [
        ctypes.POINTER(m9b1.Config), ctypes.POINTER(ctypes.c_void_p)]
    api.denise_elastic_psv_born_create.restype = ctypes.c_int
    api.denise_elastic_psv_born_prepare.argtypes = [ctypes.c_void_p, F32P]
    api.denise_elastic_psv_born_prepare.restype = ctypes.c_int
    api.denise_elastic_psv_born_apply_jt.argtypes = [
        ctypes.c_void_p, F32P, F64P, F64P]
    api.denise_elastic_psv_born_apply_jt.restype = ctypes.c_int
    api.denise_elastic_psv_born_destroy.argtypes = [
        ctypes.POINTER(ctypes.c_void_p)]
    return api


@pytest.fixture(scope="session")
def read_par_harness(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the read_par memory-safety gate"
    build_dir = tmp_path_factory.mktemp("m9c_read_par")
    source = build_dir / "read_par_harness.c"
    executable = build_dir / "read_par_harness"
    source.write_text(r'''
#include "fd.h"
#include "globvar.h"

void err(char err_text[])
{
    fprintf(stderr, "READ_PAR_ERROR: %s\n", err_text);
    exit(91);
}

float *vector(int nl, int nh)
{
    float *values;

    if ((nl != 1) || (nh < nl))
        err("invalid vector bounds in parser harness");
    values = (float *)calloc((size_t)nh + 1U, sizeof(*values));
    if (values == NULL)
        err("allocation failure in parser harness");
    return values;
}

int main(int argc, char **argv)
{
    FILE *input;
    int mechanism;

    if (argc != 2) return 2;
    input = fopen(argv[1], "r");
    if (input == NULL) return 3;
    read_par(input);
    printf("RESULT L=%d FL_NULL=%d", L, FL == NULL);
    for (mechanism = 1; mechanism <= L; ++mechanism)
        printf(" FL%d=%.9g", mechanism, FL[mechanism]);
    printf(" TAU=%.9g FREE_SURF=%d RTM_SHOT=%d Q_MODE=%d"
           " Q_FMIN=%.9g Q_FMAX=%.9g Q_DF=%.9g MODE=%d\n",
           TAU, FREE_SURF, RTM_SHOT, Q_PARAMETERIZATION_MODE,
           Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF, MODE);
    return 0;
}
''', encoding="ascii")
    completed = subprocess.run([
        compiler, "-std=c99", "-O1", "-g", "-Wall", "-Wextra",
        "-pedantic", "-fcommon", "-Wno-unused-result", "-Wno-unused-variable",
        "-Wno-format", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
        "-I", str(repository_root / "include"), str(source),
        str(repository_root / "src/read_par.c"), "-lm", "-o", str(executable),
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert completed.returncode == 0, completed.stdout
    return executable


def _write_read_par_case(path: Path, mechanism_count: int, frequency_record: str):
    records = _parameter_lines(HomogeneousPSVConfig(), 1, 1)
    records[24] = f" L ={mechanism_count}"
    records[25] = frequency_record
    records[26] = "TAU =0.3125"
    records[27] = "FREE_SURF =1"
    records[114] = "RTM_SHOT =73"
    records.extend([
        "Q_PARAMETERIZATION_MODE =1",
        "Q_APPROX_FMIN =2.75",
        "Q_APPROX_FMAX =91.25",
        "Q_APPROX_DF =0.625",
        "MIGRATION_SOURCE_PREFIX =parser/source",
        "MIGRATION_DATA_PREFIX =parser/data",
        "MIGRATION_IMAGE_PREFIX =parser/image",
    ])
    assert len(records) == 122
    path.write_text("".join(
        f"# positional parameter {index:03d}\n{line}\n"
        for index, line in enumerate(records, start=1)), encoding="ascii")


def _run_read_par_harness(executable: Path, parameter_file: Path):
    environment = os.environ.copy()
    environment["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
    environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    return subprocess.run(
        [str(executable), str(parameter_file)], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def _read_result(output: str) -> dict[str, str]:
    result_line = next(
        line for line in output.splitlines() if line.startswith("RESULT "))
    return dict(token.split("=", 1) for token in result_line.split()[1:])


def test_read_par_l0_consumes_placeholder_without_allocating_fl(
        tmp_path, read_par_harness):
    parameter_file = tmp_path / "l0.inp"
    _write_read_par_case(parameter_file, 0, "FL =16.0")
    completed = _run_read_par_harness(read_par_harness, parameter_file)
    assert completed.returncode == 0, completed.stdout
    result = _read_result(completed.stdout)
    assert result == {
        "L": "0", "FL_NULL": "1", "TAU": "0.3125", "FREE_SURF": "1",
        "RTM_SHOT": "73", "Q_MODE": "1", "Q_FMIN": "2.75",
        "Q_FMAX": "91.25", "Q_DF": "0.625", "MODE": "0",
    }


@pytest.mark.parametrize(("mechanism_count", "frequency_record", "expected"), [
    (1, "FL =8.0", ("8",)),
    (2, "FL =5.5 23.75", ("5.5", "23.75")),
])
def test_read_par_preserves_positive_relaxation_frequencies(
        tmp_path, read_par_harness, mechanism_count, frequency_record, expected):
    parameter_file = tmp_path / f"l{mechanism_count}.inp"
    _write_read_par_case(parameter_file, mechanism_count, frequency_record)
    completed = _run_read_par_harness(read_par_harness, parameter_file)
    assert completed.returncode == 0, completed.stdout
    result = _read_result(completed.stdout)
    assert result["L"] == str(mechanism_count)
    assert result["FL_NULL"] == "0"
    assert tuple(result[f"FL{index}"] for index in range(1, mechanism_count + 1)) == expected
    assert result["TAU"] == "0.3125"
    assert result["RTM_SHOT"] == "73"


def test_read_par_rejects_malformed_relaxation_frequency_record(
        tmp_path, read_par_harness):
    parameter_file = tmp_path / "malformed.inp"
    _write_read_par_case(parameter_file, 0, "FL =not-a-number")
    completed = _run_read_par_harness(read_par_harness, parameter_file)
    assert completed.returncode == 91, completed.stdout
    assert "Error while reading elastic FL placeholder (record 26)" in completed.stdout


class RequestOwner:
    def __init__(self, exp, migration_data, *, source_type=1):
        self.model = [np.ascontiguousarray(array, dtype=np.float32)
                      for array in (exp.lam, exp.mu, exp.rho)]
        self.receivers = (
            np.asarray([i - 1 for i, _ in exp.receivers], dtype=np.int32),
            np.asarray([j - 1 for _, j in exp.receivers], dtype=np.int32),
        )
        self.sources = []
        self.data = []
        self.shots = (MigrationShot * len(exp.sources))()
        for index, ((source_i, source_j), raw_data) in enumerate(
                zip(exp.sources, migration_data), start=1):
            samples = np.ascontiguousarray(_source_samples(exp), dtype=np.float32)
            data = np.ascontiguousarray(raw_data, dtype=np.float32)
            self.sources.append(samples)
            self.data.append(data)
            self.shots[index - 1] = MigrationShot(
                index, source_type, source_i - 1, source_j - 1, _ptr(samples),
                len(exp.receivers), _ptr(self.receivers[0], I32P),
                _ptr(self.receivers[1], I32P), _ptr(data))
        damping_speed = float(np.sqrt(np.max((exp.lam + 2.0 * exp.mu) / exp.rho)))
        self.request = MigrationRequest(
            exp.nx, exp.ny, exp.nt, exp.fw, exp.dh, exp.dt,
            0, 3, 4, 1, 1, 0, 0, 1, 2, 0,
            _ptr(self.model[0]), _ptr(self.model[1]), _ptr(self.model[2]),
            int(exp.cpml), exp.pml_reflection, exp.pml_power,
            exp.pml_kmax, exp.pml_fpml, damping_speed,
            len(exp.sources), self.shots)


def _migration_data_and_oracle(exp):
    dlam = lambda_reflector(exp)
    dmu = np.zeros_like(exp.mu)
    trajectories = [
        nonlinear_forward(exp, shot, save_strain=True)
        for shot in range(len(exp.sources))]
    data = [
        born_forward(exp, trajectory, dlam, dmu)
        for trajectory in trajectories]
    return data, migrate_shot_stack(exp, trajectories, data)


def _run(api, owner):
    result = MigrationResult()
    status = api.denise_elastic_psv_migrate(
        ctypes.byref(owner.request), ctypes.byref(result))
    if status:
        raise RuntimeError(api.denise_elastic_psv_migration_last_error().decode())
    shape = (owner.request.ny, owner.request.nx)
    images = (
        np.ctypeslib.as_array(result.image_lambda_raw,
                              shape=(result.cell_count,)).copy().reshape(shape),
        np.ctypeslib.as_array(result.image_mu_raw,
                              shape=(result.cell_count,)).copy().reshape(shape),
    )
    diagnostics = {
        "shots_completed": result.shots_completed,
        "cpml_memory_peak": result.cpml_memory_peak,
        "trajectory_bytes": result.trajectory_bytes,
        "global_image_bytes": result.global_image_bytes,
        "maximum_shot_data_bytes": result.maximum_shot_data_bytes,
        "checkpoint_payload_bytes": result.checkpoint_payload_bytes,
        "checkpoint_bytes": result.checkpoint_bytes,
        "segment_operand_bytes": result.segment_operand_bytes,
        "peak_replay_storage_bytes": result.peak_replay_storage_bytes,
        "forward_working_bytes": result.forward_working_bytes,
        "adjoint_working_bytes": result.adjoint_working_bytes,
        "initial_forward_steps": result.initial_forward_steps,
        "replayed_steps": result.replayed_steps,
        "segment_count": result.segment_count,
        "checkpoint_count": result.checkpoint_count,
        "max_segment_length": result.max_segment_length,
        "checkpoint_metadata_bytes": result.checkpoint_metadata_bytes,
        "checkpoint_pointer_bytes": result.checkpoint_pointer_bytes,
        "segment_schedule_bytes": result.segment_schedule_bytes,
    }
    api.denise_elastic_psv_migration_result_destroy(ctypes.byref(result))
    assert not result.image_lambda_raw and not result.image_mu_raw
    return images, diagnostics


def _relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


@pytest.mark.parametrize("case", ("interior", "active_cpml"))
def test_driver_matches_manual_ordered_m9b1_stack_and_oracle(
        migration_library, case):
    exp = make_experiment(case)
    if case == "interior":
        exp = replace(exp, sources=((17, 16), (25, 22)))
    data, oracle = _migration_data_and_oracle(exp)
    owner = RequestOwner(exp, data)
    first, diagnostics = _run(migration_library, owner)
    second, _ = _run(migration_library, owner)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])

    manual_lambda = np.zeros_like(first[0])
    manual_mu = np.zeros_like(first[1])
    for shot, shot_data in enumerate(data):
        operator = m9b1.Operator(migration_library, exp, shot=shot)
        try:
            operator.prepare()
            glambda, gmu = operator.jt(shot_data)
            manual_lambda += glambda
            manual_mu += gmu
        finally:
            operator.close()
    np.testing.assert_array_equal(first[0], manual_lambda)
    np.testing.assert_array_equal(first[1], manual_mu)
    assert _relative(first[0], oracle.image_lambda_raw) <= 6.0e-5
    assert _relative(first[1], oracle.image_mu_raw) <= 6.0e-5
    assert diagnostics["shots_completed"] == 2
    assert diagnostics["trajectory_bytes"] == 4 * exp.nt * exp.nx * exp.ny * 4
    assert diagnostics["global_image_bytes"] == 2 * exp.nx * exp.ny * 8
    assert diagnostics["maximum_shot_data_bytes"] == (
        exp.nt * len(exp.receivers) * 2 * 4)
    segments = min(32, exp.nt)
    maximum = exp.nt // segments + (exp.nt % segments != 0)
    payload = (5 * exp.nx * exp.ny
               + ((8 * exp.fw + 2) * (exp.nx + exp.ny)
                  if exp.cpml else 0)) * 4
    # Query the same authoritative estimate used by production; the focused
    # H1 tests independently audit the object/schedule allocation decomposition.
    reference = m9b1.Operator(migration_library, exp)
    estimate = ctypes.c_size_t()
    try:
        estimate_api = migration_library.denise_elastic_psv_born_estimate_replay_storage
        estimate_api.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                ctypes.POINTER(ctypes.c_size_t)]
        estimate_api.restype = ctypes.c_int
        assert estimate_api(reference.context, segments, ctypes.byref(estimate)) == 0
    finally:
        reference.close()
    selected = estimate.value < diagnostics["trajectory_bytes"]
    assert diagnostics["segment_count"] == (segments if selected else 0)
    assert diagnostics["checkpoint_count"] == (segments - 1 if selected else 0)
    assert diagnostics["max_segment_length"] == (maximum if selected else 0)
    assert diagnostics["peak_replay_storage_bytes"] == (
        diagnostics["checkpoint_bytes"] + diagnostics["segment_operand_bytes"]
        + diagnostics["checkpoint_metadata_bytes"]
        + diagnostics["checkpoint_pointer_bytes"]
        + diagnostics["segment_schedule_bytes"])
    assert diagnostics["checkpoint_payload_bytes"] == (payload if selected else 0)
    assert diagnostics["initial_forward_steps"] == exp.nt
    assert diagnostics["replayed_steps"] == (exp.nt if selected else 0)
    assert (diagnostics["cpml_memory_peak"] > 0.0) is exp.cpml


@pytest.mark.parametrize("time,receiver,component", [
    (0, 0, 0), (3, 1, 1), (2, 0, 2), (4, 1, 0),
], ids=("first-vx", "distinct-vy", "both", "last-time"))
def test_component_adapter_basis(migration_library, time, receiver, component):
    nt, nrec = 5, 2
    vx = np.zeros((nt, nrec), dtype=np.float32)
    vy = np.zeros_like(vx)
    if component in (0, 2):
        vx[time, receiver] = 3.25
    if component in (1, 2):
        vy[time, receiver] = -4.5
    packed = np.empty((nt, nrec, 2), dtype=np.float32)
    assert migration_library.denise_elastic_psv_migration_pack_components(
        _ptr(vx), _ptr(vy), nt, nrec, _ptr(packed)) == 0
    expected = np.stack((vx, vy), axis=-1)
    np.testing.assert_array_equal(packed, expected)


@pytest.mark.parametrize("field,value,fragment", [
    ("l", 1, "L=0"), ("invmat1", 1, "INVMAT1=3"),
    ("fdorder", 2, "FDORDER=4"), ("ndt", 2, "NDT=DTINV=1"),
    ("dtinv", 2, "NDT=DTINV=1"), ("free_surface", 1, "FREE_SURF"),
    ("boundary", 1, "BOUNDARY"), ("mpi_size", 2, "one MPI rank"),
    ("receiver_components", 1, "direct vx/vy"), ("inv_stf", 1, "INV_STF=0"),
])
def test_request_envelope_fails_closed(migration_library, field, value, fragment):
    exp = make_experiment("interior")
    data, _ = _migration_data_and_oracle(exp)
    owner = RequestOwner(exp, data)
    setattr(owner.request, field, value)
    result = MigrationResult()
    assert migration_library.denise_elastic_psv_migrate(
        ctypes.byref(owner.request), ctypes.byref(result)) != 0
    assert fragment in migration_library.denise_elastic_psv_migration_last_error().decode()
    assert not result.image_lambda_raw and not result.image_mu_raw


def test_source_type_and_shot_order_fail_closed(migration_library):
    exp = replace(make_experiment("interior"), sources=((17, 16), (25, 22)))
    data, _ = _migration_data_and_oracle(exp)
    owner = RequestOwner(exp, data, source_type=5)
    result = MigrationResult()
    assert migration_library.denise_elastic_psv_migrate(
        ctypes.byref(owner.request), ctypes.byref(result)) != 0
    assert "unsupported source type 5" in (
        migration_library.denise_elastic_psv_migration_last_error().decode())
    owner.shots[0].source_type = 1
    owner.shots[1].source_type = 1
    owner.shots[1].physical_shot_index = 1
    assert migration_library.denise_elastic_psv_migrate(
        ctypes.byref(owner.request), ctypes.byref(result)) != 0
    assert "strictly ascending" in (
        migration_library.denise_elastic_psv_migration_last_error().decode())


def test_shot_two_failure_publishes_no_result_and_leaves_no_stale_state(
        migration_library):
    exp = replace(make_experiment("interior"), sources=((17, 16), (25, 22)))
    data, _ = _migration_data_and_oracle(exp)
    owner = RequestOwner(exp, data)
    owner.data[1][3, 1, 1] = np.nan
    result = MigrationResult()
    assert migration_library.denise_elastic_psv_migrate(
        ctypes.byref(owner.request), ctypes.byref(result)) != 0
    assert "shot 2 migration data contains NaN or Inf" in (
        migration_library.denise_elastic_psv_migration_last_error().decode())
    assert not result.image_lambda_raw and not result.image_mu_raw
    owner.data[1][3, 1, 1] = data[1][3, 1, 1]
    recovered, _ = _run(migration_library, owner)
    fresh, _ = _run(migration_library, RequestOwner(exp, data))
    np.testing.assert_array_equal(recovered[0], fresh[0])
    np.testing.assert_array_equal(recovered[1], fresh[1])


def test_dataset_a_then_b_matches_fresh_b(migration_library):
    exp_a = make_experiment("interior")
    data_a, _ = _migration_data_and_oracle(exp_a)
    _run(migration_library, RequestOwner(exp_a, data_a))
    exp_b = replace(exp_a, sources=((17, 16), (25, 22)))
    data_b, _ = _migration_data_and_oracle(exp_b)
    after_a, _ = _run(migration_library, RequestOwner(exp_b, data_b))
    fresh_b, _ = _run(migration_library, RequestOwner(exp_b, data_b))
    np.testing.assert_array_equal(after_a[0], fresh_b[0])
    np.testing.assert_array_equal(after_a[1], fresh_b[1])


def test_grid_size_overflow_fails_before_allocation(migration_library):
    exp = make_experiment("interior")
    data, _ = _migration_data_and_oracle(exp)
    owner = RequestOwner(exp, data)
    owner.request.nx = 2_147_483_647
    owner.request.ny = 2_147_483_647
    result = MigrationResult()
    assert migration_library.denise_elastic_psv_migrate(
        ctypes.byref(owner.request), ctypes.byref(result)) != 0
    assert "NX*NY overflows" in (
        migration_library.denise_elastic_psv_migration_last_error().decode())
    assert not result.image_lambda_raw and not result.image_mu_raw


def test_active_driver_and_dispatch_are_decoupled_from_legacy_rtm(repository_root):
    driver = "\n".join(
        (repository_root / path).read_text()
        for path in (
            "src/PSV/elastic_psv_migration.c",
            "src/PSV/elastic_psv_migration_mode2.c",
        )
    )
    dispatch = (repository_root / "src/PSV/physics_PSV.c").read_text()
    forbidden = (
        "calc_res_PSV", "grad_obj_psv", "ass_gradPSV", "precond_PSV",
        "RTM_PSV_out", "P_image", "S_image",
    )
    assert not any(symbol in driver for symbol in forbidden)
    assert "denise_elastic_psv_migration_mode2" in dispatch
    assert "MODE==2" not in dispatch or "RTM_PSV();" not in dispatch


def _write_mode2_case(directory: Path, exp, data) -> dict[str, Path]:
    for name in ("model", "prepared", "image", "log"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    model_prefix = directory / "model" / "background"
    for suffix, values in (("lam", exp.lam), ("mu", exp.mu), ("rho", exp.rho)):
        # Canonical DENISE model files use x-major traversal (i outer, j inner).
        np.ascontiguousarray(values, dtype=np.float32).T.copy().tofile(
            model_prefix.with_suffix(f".{suffix}"))

    source_rows = [str(len(exp.sources))]
    source_prefix = directory / "prepared" / "source"
    data_prefix = directory / "prepared" / "migration"
    for shot, (i, j) in enumerate(exp.sources, start=1):
        source_rows.append(
            f"{i * exp.dh} 0.0 {j * exp.dh} 0.0 0.0 1.0 0.0 1")
        np.asarray(_source_samples(exp), dtype=np.float32).tofile(
            Path(f"{source_prefix}.shot_{shot}.bin"))
        np.ascontiguousarray(data[shot - 1][:, :, 0], dtype=np.float32).tofile(
            Path(f"{data_prefix}.vx.shot_{shot}.bin"))
        np.ascontiguousarray(data[shot - 1][:, :, 1], dtype=np.float32).tofile(
            Path(f"{data_prefix}.vy.shot_{shot}.bin"))
    (directory / "source.dat").write_text("\n".join(source_rows) + "\n", encoding="ascii")
    (directory / "receiver.dat").write_text("".join(
        f"{i * exp.dh} {j * exp.dh}\n" for i, j in exp.receivers), encoding="ascii")

    vmax = float(np.sqrt(np.max((exp.lam + 2.0 * exp.mu) / exp.rho)))
    config = HomogeneousPSVConfig(
        nx=exp.nx, ny=exp.ny, dh_m=exp.dh,
        time_s=exp.nt * exp.dt, dt_s=exp.dt,
        absorbing_width_gridpoints=exp.fw,
        damping_velocity_m_s=vmax, pml_frequency_hz=exp.pml_fpml,
        fd_order=4, free_surface=False,
    )
    records = _parameter_lines(config, 1, 1)
    changes = {
        0: "MODE =2", 4: "FD_ORDER =4", 6: f"NX ={exp.nx}",
        7: f"NY ={exp.ny}", 8: f"DH ={exp.dh}",
        9: f"TIME ={exp.nt * exp.dt:.17g}", 10: f"DT ={exp.dt:.17g}",
        11: "QUELLART =3", 12: "SIGNAL_FILE =prepared/source",
        14: "SRCREC =1", 15: "SOURCE_FILE =source.dat",
        16: "RUN_MULTIPLE_SHOTS =1", 21: "READMOD =1",
        22: "MFILE =model/background", 24: "L =0", 27: "FREE_SURF =0",
        28: f"FW ={exp.fw}", 29: f"DAMPING ={vmax:.17g}",
        30: f"FPML ={exp.pml_fpml}", 31: f"NPOWER ={exp.pml_power}",
        32: f"K_MAX_PML ={exp.pml_kmax}", 33: "BOUNDARY =0",
        34: "SNAP =0", 43: "SEISMO =1", 44: "READREC =1",
        45: "REC_FILE =receiver", 46: "REFREC =0.0,0.0",
        47: "N_STREAMER =0", 50: "NDT =1", 57: "LOG_FILE =log/denise.log",
        58: "LOG =1", 65: "INVMAT1 =3", 67: "QUELLTYPB =1",
        94: "DTINV =1",
    }
    for index, value in changes.items():
        records[index] = value
    records.extend([
        "Q_PARAMETERIZATION_MODE =0",
        "Q_APPROX_FMIN =1.0",
        "Q_APPROX_FMAX =2.0",
        "Q_APPROX_DF =1.0",
        "MIGRATION_SOURCE_PREFIX =prepared/source",
        "MIGRATION_DATA_PREFIX =prepared/migration",
        "MIGRATION_IMAGE_PREFIX =image/m9c",
    ])
    assert len(records) == 122
    parameters = "# M9c production integration fixture\n" + "".join(
        f"# positional parameter {index:03d}\n{line}\n"
        for index, line in enumerate(records, start=1))
    (directory / "denise.inp").write_text(parameters, encoding="ascii")
    (directory / "workflow.inp").write_text("# M9c does not use an FWI workflow.\n",
                                               encoding="ascii")
    return {
        "lambda": directory / "image" / "m9c.image_lambda_raw.bin",
        "mu": directory / "image" / "m9c.image_mu_raw.bin",
        "vx1": Path(f"{data_prefix}.vx.shot_1.bin"),
        "vy1": Path(f"{data_prefix}.vy.shot_1.bin"),
    }


def _run_denise(directory: Path, denise_binary: Path, mpiexec: str):
    return subprocess.run(
        [mpiexec, "-n", "1", str(denise_binary), "denise.inp", "workflow.inp"],
        cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False, timeout=180)


@pytest.mark.integration
@pytest.mark.parametrize("case", ("interior", "active_cpml"))
def test_mode2_executable_matches_direct_driver(
        tmp_path, migration_library, denise_binary, mpiexec, case):
    exp = make_experiment(case)
    if case == "interior":
        exp = replace(exp, sources=((17, 16), (25, 22)))
    data, _ = _migration_data_and_oracle(exp)
    expected, _ = _run(migration_library, RequestOwner(exp, data))
    paths = _write_mode2_case(tmp_path, exp, data)
    completed = _run_denise(tmp_path, denise_binary, mpiexec)
    assert completed.returncode == 0, completed.stdout
    shape = (exp.ny, exp.nx)
    actual = (
        np.fromfile(paths["lambda"], dtype=np.float64).reshape(shape),
        np.fromfile(paths["mu"], dtype=np.float64).reshape(shape),
    )
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert "chronological [time][receiver][vx,vy]" in completed.stdout
    assert "float64" in completed.stdout
    if case == "active_cpml":
        marker = "CPML memory peak:"
        assert marker in completed.stdout
        peak = float(completed.stdout.split(marker, 1)[1].splitlines()[0])
        assert peak > 0.0


@pytest.mark.integration
@pytest.mark.parametrize("fault,fragment", [
    ("missing-vx", "cannot open"),
    ("missing-vy", "cannot open"),
    ("wrong-receiver-count", "expected exactly"),
    ("wrong-sample-count", "expected exactly"),
    ("nonfinite", "NaN or Inf"),
])
def test_mode2_file_failures_remove_final_pair(
        tmp_path, denise_binary, mpiexec, fault, fragment):
    exp = replace(make_experiment("interior"), sources=((17, 16), (25, 22)))
    data, _ = _migration_data_and_oracle(exp)
    paths = _write_mode2_case(tmp_path, exp, data)
    paths["lambda"].write_bytes(b"stale lambda")
    paths["mu"].write_bytes(b"stale mu")
    if fault == "missing-vx":
        paths["vx1"].unlink()
    elif fault == "missing-vy":
        paths["vy1"].unlink()
    elif fault == "wrong-receiver-count":
        with (tmp_path / "receiver.dat").open("a", encoding="ascii") as stream:
            stream.write(f"{exp.dh} {exp.dh}\n")
    elif fault == "wrong-sample-count":
        raw = paths["vx1"].read_bytes()
        paths["vx1"].write_bytes(raw[:-4])
    else:
        values = np.fromfile(paths["vx1"], dtype=np.float32)
        values[len(values) // 2] = np.nan
        values.tofile(paths["vx1"])
    completed = _run_denise(tmp_path, denise_binary, mpiexec)
    assert completed.returncode != 0, completed.stdout
    assert fragment in completed.stdout
    assert not paths["lambda"].exists()
    assert not paths["mu"].exists()
