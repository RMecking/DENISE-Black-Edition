"""FLUID-3E: independent CPU three-shot raw JT, replay and real MPI MODE=2.

Expected images come only from the frozen restricted FP64 reference. Production
one-shot images are a separate exact orchestration check, never the oracle.
The 6e-5 reference and 1e-12 MPI ceilings are the published FLUID-2/M9 ceilings.
No residual/sign/time/space scaling, density image or CUDA fluid capability.
"""
import ctypes as C
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from tests.physics import test_m9b1_elastic_psv_born_production as born
from tests.physics import test_m9c_elastic_psv_migration_driver as driver
from tests.physics.test_m9c_elastic_psv_migration_driver import migration_library
from tests.physics import test_m9d1_elastic_psv_checkpoint_replay as replay
from tests.physics.test_m9d1_elastic_psv_checkpoint_replay import replay_library
from tests.physics import test_m9d2_elastic_psv_mpi as mpi
from tests.utilities import m9_fluid_restricted_reference as ref
from tests.utilities import zero_shear_reference as zero


TOPS = ((1, 1), (2, 1), (1, 2), (2, 2))
CASES = ((0, 0), (1, 3))


def record(kind, **values):
    path = os.environ.get("DENISE_FLUID3E_EVIDENCE")
    if path:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(kind=kind, **values)) + "\n")


def bit_equal(actual, expected):
    assert actual.dtype == expected.dtype == np.float64
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes(), mpi.errors(actual, expected)


def active_channels(image, exp):
    assert image.shape == (2, exp.ny, exp.nx)
    assert np.isfinite(image).all()
    fluid = exp.mu == 0
    assert (image[1][fluid].view(np.uint64) == 0).all()
    gl = float(np.linalg.norm(image[0][fluid]))
    gm = float(np.linalg.norm(image[1][~fluid]))
    assert gl > 0 and gm > 0
    return dict(fluid_glambda_norm=gl, solid_gmu_norm=gm,
                fluid_gmu_positive_zero=True)


def reference_errors(actual, expected):
    metrics = dict(lambda_relative=driver._relative(actual[0], expected[0]),
                   mu_relative=driver._relative(actual[1], expected[1]),
                   combined_relative=driver._relative(actual, expected))
    assert max(metrics.values()) <= 6e-5, metrics
    return metrics


@pytest.fixture(scope="module", params=CASES, ids=("FS0-FW0", "FS1-FW3"))
def experiment(request):
    fs, fw = request.param
    exp = ref.fixture("horizontal", fs, fw, nx=24, ny=20, nt=120)
    # Canonical 1-based physical geometry: two water shots near the interface,
    # one solid shot. Sources/receivers straddle both ownership axes in 2x2.
    mu = exp.mu.copy()
    y, x = np.indices(mu.shape)
    mu[(mu == 0) & ((x + y) % 2 == 0)] = -0.0
    exp = replace(exp, mu=mu, sources=((9, 10), (14, 14), (13, 11)))
    assert all(j > 1 for _, j in exp.sources)
    source = ref.b._source_samples(exp).astype(np.float32).astype(float)
    material = zero.material(exp.rho, exp.lam, exp.mu)
    dl, dm = (a.astype(np.float32).astype(float)
              for a in zero.directions(material)["interface"])
    empty = np.zeros((exp.nt, len(exp.receivers), 2))
    data, images = [], []
    total = np.zeros((2, exp.ny, exp.nx), dtype=np.float64)
    for coordinates in exp.sources:  # ascending physical indices 1, 2, 3
        shot = replace(exp, sources=(coordinates,))
        prepared = ref.trajectory(shot, source, fs)
        d = ref.products(shot, source, dl, dm, empty, fs, prepared)[0].astype(np.float32)
        # Cotangent is the rounded independent Born data, not Production output.
        products = ref.products(shot, source, np.zeros_like(dl), np.zeros_like(dm),
                                d.astype(float), fs, prepared)
        image = np.stack(products[1:3])
        assert (image[1][material.fluid].view(np.uint64) == 0).all()
        total += image
        data.append(d)
        images.append(image)
    total[1][material.fluid] = 0.0  # only the frozen restricted-space projection
    assert all(not np.array_equal(data[a], data[b]) for a in range(3) for b in range(a))
    activity = active_channels(total, exp)
    record("fixture", fs=fs, fw=fw, nx=exp.nx, ny=exp.ny, nt=exp.nt,
           dh=exp.dh, dt=exp.dt, sources=list(exp.sources), receivers=list(exp.receivers),
           source_fc=exp.fc, source_t0=exp.source_t0, source_amplitude=exp.source_amplitude,
           ascending_indices=[1, 2, 3], oracle="frozen restricted FP64 equations", **activity)
    return exp, fs, tuple(data), total, tuple(images)


def owner_for(exp, fs, data):
    owner = driver.RequestOwner(exp, [values.copy() for values in data])
    owner.request.free_surface = fs
    return owner


def run_driver(api, exp, fs, data):
    images, diagnostics = driver._run(api, owner_for(exp, fs, data))
    assert images[0].size == images[1].size == exp.nx * exp.ny
    return np.stack(images), diagnostics


def replay_diagnostics(exp, diagnostics, shots):
    segments = min(32, exp.nt)
    assert exp.nt % segments != 0
    assert diagnostics["shots_completed"] == shots
    assert diagnostics["segment_count"] == segments > 0
    assert diagnostics["checkpoint_count"] == segments - 1
    assert diagnostics["max_segment_length"] == (exp.nt + segments - 1) // segments
    assert diagnostics["initial_forward_steps"] == diagnostics["replayed_steps"] == exp.nt
    assert diagnostics["trajectory_bytes"] == 4 * exp.nt * exp.nx * exp.ny * 4
    assert 0 < diagnostics["peak_replay_storage_bytes"] < diagnostics["trajectory_bytes"]
    assert diagnostics["peak_replay_storage_bytes"] == sum(diagnostics[key] for key in (
        "checkpoint_bytes", "checkpoint_metadata_bytes", "checkpoint_pointer_bytes",
        "segment_schedule_bytes", "segment_operand_bytes"))


def test_three_shot_reference_decomposition_and_fresh_determinism(migration_library, experiment):
    exp, fs, data, expected, per_shot = experiment
    actual, diagnostics = run_driver(migration_library, exp, fs, data)
    metrics = reference_errors(actual, expected)
    activity = active_channels(actual, exp)
    replay_diagnostics(exp, diagnostics, 3)
    if exp.cpml:
        assert diagnostics["cpml_memory_peak"] > 0
    manual = np.zeros_like(actual)
    for index, coordinates in enumerate(exp.sources):
        shot = replace(exp, sources=(coordinates,))
        image, diag = run_driver(migration_library, shot, fs, [data[index]])
        reference_errors(image, per_shot[index])
        active_channels(image, exp)
        replay_diagnostics(shot, diag, 1)
        manual += image
    bit_equal(actual, manual)
    repeated, _ = run_driver(migration_library, exp, fs, data)
    bit_equal(repeated, actual)
    record("serial", fs=fs, fw=exp.fw, decomposition_bitwise=True,
           fresh_repeat_bitwise=True, diagnostics=diagnostics, **metrics, **activity)


def test_full_segmented_each_shot_and_sum_bitwise(replay_library, migration_library, experiment):
    exp, fs, data, expected, _ = experiment
    sums = [np.zeros_like(expected), np.zeros_like(expected)]
    diagnostics = []
    for shot_index in range(3):
        images = []
        for mode in ("FULL", "SEGMENTED"):
            operator = born.Operator(replay_library, exp, shot=shot_index, free_surface=fs)
            try:
                if mode == "SEGMENTED":
                    assert replay_library.denise_elastic_psv_born_set_replay_segments(
                        operator.context, min(32, exp.nt)) == 0
                    bounds = replay._bounds(replay_library, operator, 32)
                    assert bounds == [(s * exp.nt // 32, (s + 1) * exp.nt // 32) for s in range(32)]
                    assert {end - start for start, end in bounds} == {3, 4}
                operator.prepare()
                image = np.stack(operator.jt(data[shot_index]))
                active_channels(image, exp)
                d = replay.StorageDiagnostics()
                assert replay_library.denise_elastic_psv_born_storage_diagnostics(
                    operator.context, C.byref(d)) == 0
                assert d.segmented == int(mode == "SEGMENTED")
                if d.segmented:
                    assert d.segment_count == 32 and d.checkpoint_count == 31
                    assert d.initial_forward_steps == d.replayed_forward_steps_last == exp.nt
                    assert d.max_segment_length == 4
                diagnostics.append({name: getattr(d, name) for name, _ in d._fields_})
                images.append(image)
            finally:
                operator.close()
        bit_equal(images[0], images[1])
        for total, image in zip(sums, images):
            total += image
    bit_equal(sums[0], sums[1])
    actual, _ = run_driver(migration_library, exp, fs, data)
    bit_equal(actual, sums[1])
    reference_errors(sums[0], expected)
    record("replay", fs=fs, fw=exp.fw, per_shot_bitwise=True,
           accumulated_bitwise=True, driver_bitwise=True, diagnostics=diagnostics)


@pytest.mark.parametrize("fault", ("equal-index", "descending-index", "late-data", "late-source"))
def test_invalid_multishot_has_no_result_and_recovers_bitwise(migration_library, experiment, fault):
    exp, fs, data, _, _ = experiment
    canonical, _ = run_driver(migration_library, exp, fs, data)
    owner = owner_for(exp, fs, data)
    if fault.endswith("index"):
        owner.shots[2].physical_shot_index = 2 if fault == "equal-index" else 1
    elif fault == "late-data":
        owner.data[2][7, 1, 1] = np.nan
    else:
        owner.sources[2][7] = np.nan
    result = driver.MigrationResult()
    assert migration_library.denise_elastic_psv_migrate(C.byref(owner.request), C.byref(result)) != 0
    message = migration_library.denise_elastic_psv_migration_last_error().decode()
    assert ("strictly ascending" if fault.endswith("index") else "shot 3") in message
    assert not result.image_lambda_raw and not result.image_mu_raw
    assert result.cell_count == result.shots_completed == 0
    owner.shots[2].physical_shot_index = 3
    owner.data[2][:] = data[2]
    owner.sources[2][:] = ref.b._source_samples(exp).astype(np.float32)
    recovered = np.stack(driver._run(migration_library, owner)[0])
    bit_equal(recovered, canonical)
    record("transaction", fs=fs, fw=exp.fw, fault=fault,
           no_authoritative_result=True, corrected_bitwise=True, error=message)


@pytest.mark.integration
@pytest.mark.parametrize("top", TOPS)
def test_actual_mode2_distributed_multishot_and_pair_recovery(
        tmp_path, experiment, migration_library, denise_binary, mpiexec, top):
    exp, fs, data, reference, _ = experiment
    direct, _ = run_driver(migration_library, exp, fs, data)
    paths = driver._write_mode2_case(tmp_path, exp, data)
    inp = tmp_path / "denise.inp"
    text = inp.read_text().replace("NPROCX =1", f"NPROCX ={top[0]}")
    text = text.replace("NPROCY =1", f"NPROCY ={top[1]}")
    inp.write_text(text.replace("FREE_SURF =0", f"FREE_SURF ={fs}"))

    def execute(label):
        p = subprocess.run([mpiexec, "--oversubscribe", "-n", str(top[0] * top[1]),
                            str(denise_binary), "denise.inp", "workflow.inp"],
                           cwd=tmp_path, capture_output=True, text=True, timeout=120)
        log = p.stdout + p.stderr
        (tmp_path / (label + ".log")).write_text(log)
        return p.returncode, log

    status, log = execute("valid")
    assert status == 0, log
    assert "FLUID-2" not in log
    assert "chronological [time][receiver][vx,vy]" in log
    if top == (1, 1):
        assert "shots: 3 (strict ascending physical index)" in log
        assert "SEGMENTED" in log
        assert re.search(r"replay segments/checkpoints/max length:\s*32 / 31 / 4", log)
    else:
        assert f"{top[0]}x{top[1]}; 3 ascending shots" in log
        rows = re.findall(r"rank (\d+) FULL/replay/retained bytes: (\d+) / (\d+) / (\d+); backend (\w+)", log)
        assert {int(row[0]) for row in rows} == set(range(top[0] * top[1]))
        assert all(backend == "SEGMENTED" and int(retained) == int(estimate) < int(full)
                   for _, full, estimate, retained, backend in rows)
    blobs = tuple(paths[key].read_bytes() for key in ("lambda", "mu"))
    assert all(len(blob) == exp.nx * exp.ny * 8 for blob in blobs)
    actual = np.stack([np.frombuffer(blob, np.float64).reshape(exp.ny, exp.nx) for blob in blobs])
    activity = active_channels(actual, exp)
    independent = reference_errors(actual, reference)
    differences = [mpi.errors(a, b) for a, b in zip(actual, direct)]
    if top == (1, 1):
        bit_equal(actual, direct)
    else:
        assert all(metric["relative_l2"] <= 1e-12 for metric in differences), differences
        assert driver._relative(actual, direct) <= 1e-12
    status, repeated_log = execute("repeated")
    assert status == 0, repeated_log
    assert tuple(paths[key].read_bytes() for key in ("lambda", "mu")) == blobs
    # Late prepared data fail on both serial and collective distributed paths;
    # stale successful pairs must be removed, then corrected input must recover.
    late = tmp_path / "prepared/migration.vy.shot_3.bin"
    original = late.read_bytes()
    invalid = np.frombuffer(original, np.float32).copy()
    invalid[7 * len(exp.receivers) + 1] = np.nan
    late.write_bytes(invalid.tobytes())
    status, failed_log = execute("invalid-shot-three")
    assert status != 0, failed_log
    # Published distributed loader reports a collective setup/I/O failure,
    # whereas the one-rank adapter preserves the detailed nonfinite message.
    fragment = "NaN or Inf" if top == (1, 1) else "M9d2 collective MODE=2 failure"
    assert fragment in failed_log, failed_log
    assert all(not paths[key].exists() for key in ("lambda", "mu"))
    assert not list((tmp_path / "image").glob("*.tmp"))
    late.write_bytes(original)
    status, recovered_log = execute("recovered")
    assert status == 0, recovered_log
    assert tuple(paths[key].read_bytes() for key in ("lambda", "mu")) == blobs
    record("mode2_mpi", fs=fs, fw=exp.fw, top=list(top), serial_metrics=differences,
           combined_serial_relative=driver._relative(actual, direct),
           independent_metrics=independent, pair_removed=True,
           temporary_pair_absent=True, repeat_bitwise=True, recovery_bitwise=True,
           actual_distributed=top != (1, 1), **activity)
