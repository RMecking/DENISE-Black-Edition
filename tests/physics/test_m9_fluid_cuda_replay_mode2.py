"""FLUID-4E: compact CUDA fluid replay, requests, and MODE=2 gates."""
from __future__ import annotations

import ctypes as C
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from tests.physics import test_m9e1_cuda_elastic_psv_forward as e1
from tests.physics import test_m9e2_cuda_free_surface_born as e2
from tests.physics import test_m9e3_cuda_jt_migration as e3
from tests.physics import test_m9e4_cuda_replay_mode2 as e4
from tests.physics import test_m9c_elastic_psv_migration_driver as driver
from tests.physics import test_m9_fluid_cuda_forward as forward
from tests.physics import test_m9_fluid_cuda_j as fluid_j
from tests.physics import test_m9_fluid_cuda_jt as fluid_jt
from tests.physics.test_m9e2_cuda_free_surface_born import backends
from tests.physics.test_m9e3_cuda_jt_migration import jt
from tests.physics.test_m9e4_cuda_replay_mode2 import replay, cuda_mode2_binary
from tests.physics.test_m9_fluid_cuda_forward import fluid_cuda
from tests.physics.test_m9_fluid_cuda_j import restricted_cuda
from tests.physics.test_m9_fluid_cuda_jt import restricted_jt
from tests.physics.test_m9d1_elastic_psv_checkpoint_replay import StorageDiagnostics
from tests.utilities import m9_fluid_restricted_reference as ref

F, P, fp, check = e1.F, e1.P, e1.fp, e1.check
RECORDS = []


@pytest.fixture(scope="session")
def fluid_replay(restricted_jt):
    cpu, libs = restricted_jt
    cpu.denise_elastic_psv_born_checkpoint_roundtrip.argtypes = [P, C.c_int]
    yield cpu, libs
    path = os.environ.get("DENISE_FLUID4E_EVIDENCE")
    if path:
        Path(path).write_text(json.dumps(RECORDS, indent=2))


def signed_fluid(a):
    mask = a["m"] == 0
    y, x = np.indices(mask.shape)
    a["m"][mask] = np.where((x[mask] + y[mask]) & 1, -0.0, 0.0)
    return mask


def relative(actual, expected):
    metric = e3.metrics(actual, expected)
    assert metric["rel_l2"] <= 6e-5, metric
    return metric


def exact_positive_zero(image, mask):
    assert np.all(image[mask].view(np.uint64) == 0)


def expected_checkpoint_payload(lib, full, cfg, segments):
    maps = np.empty(5*(cfg.ny+4)*(cfg.nx+4), np.float32)
    profiles = np.empty(12*(cfg.nx+cfg.ny), np.float32)
    source = np.empty(cfg.nt, np.float32)
    geometry = np.empty(3*cfg.receiver_count, np.int32)
    check(lib, lib.denise_cuda_m9_test_static(
        full, fp(maps), fp(profiles), fp(source), e1.ip(geometry)))
    effective = min(segments, cfg.nt)
    payloads = []
    for checkpoint in range(effective-1):
        timestep = (checkpoint+1)*cfg.nt//effective - 1
        state = e4.probe(lib, full, cfg, 0, timestep, False)[0]
        parts = [state[:5].reshape(-1)]
        for kind in range(8):
            x_channel = kind in (0, 2, 4, 5)
            profile = (1 if kind in (0, 5) else
                       2 if kind in (1, 7) else
                       0 if kind in (2, 4) else 3)
            length = cfg.nx if x_channel else cfg.ny
            offset = (0 if profile == 0 else
                      3*cfg.nx if profile == 1 else
                      6*cfg.nx if profile == 2 else
                      6*cfg.nx + 3*cfg.ny)
            active = profiles[offset+length:offset+2*length] != 0
            psi = state[5+kind]
            if x_channel:
                assert np.count_nonzero(psi[:, ~active]) == 0
                parts.append(psi[:, active].reshape(-1))
            else:
                assert np.count_nonzero(psi[~active, :]) == 0
                parts.append(psi[active, :].T.reshape(-1))
        payloads.append(np.concatenate(parts))
    return np.concatenate(payloads) if payloads else np.empty(0, np.float32)


CASES = [
    ("nofma", "homogeneous", 0, 0, 1),
    ("fma", "horizontal", 0, 3, 2),
    ("nofma", "horizontal", 1, 0, 3),
    ("fma", "offset", 1, 3, 7),
    ("nofma", "vertical", 0, 0, 19),
    ("fma", "horizontal", 1, 3, 24),
]


@pytest.mark.parametrize("mode,pattern,fs,fw,segments", CASES)
def test_fluid_full_segmented_replay_matrix(
        fluid_replay, mode, pattern, fs, fw, segments):
    cpu, libs = fluid_replay
    lib = libs[mode]
    cfg, a, exp = forward.fixture(pattern, fs, fw, nt=19)
    fluid = signed_fluid(a)
    full = e3.create(lib, cfg)
    replay_context = e4.create(lib, cfg, segments)
    try:
        check(lib, lib.denise_cuda_m9_prepare(full))
        check(lib, lib.denise_cuda_m9_prepare(replay_context))
        full_data = e1.download(lib, full, cfg)[0]
        replay_data = np.empty_like(full_data)
        check(lib, lib.denise_cuda_m9_download(
            replay_context, fp(replay_data), F(), F(), F()))
        e4.assert_same(full_data, replay_data)

        diagnostics = e4.diag(lib, replay_context)
        effective = min(segments, cfg.nt)
        assert diagnostics.effective_segments == effective
        assert diagnostics.checkpoint_count == effective - 1
        assert diagnostics.max_segment_length == (cfg.nt + effective - 1)//effective
        assert diagnostics.initial_forward_steps == cfg.nt
        assert diagnostics.segment_operand_bytes == (
            16*((cfg.nt + effective - 1)//effective)*cfg.nx*cfg.ny)

        cpu_context = P()
        assert cpu.denise_elastic_psv_born_create(C.byref(cfg), C.byref(cpu_context)) == 0
        try:
            assert cpu.denise_elastic_psv_born_set_replay_segments(
                cpu_context, segments) == 0
            assert cpu.denise_elastic_psv_born_prepare(cpu_context, F()) == 0
            cpu_storage = StorageDiagnostics()
            assert cpu.denise_elastic_psv_born_storage_diagnostics(
                cpu_context, C.byref(cpu_storage)) == 0
            assert cpu_storage.checkpoint_payload_bytes == 4*diagnostics.checkpoint_values
            if cfg.nt > 1:
                assert cpu.denise_elastic_psv_born_checkpoint_roundtrip(
                    cpu_context, cfg.nt//2 - 1) == 0
        finally:
            cpu.denise_elastic_psv_born_destroy(C.byref(cpu_context))

        previous = 0
        for segment in range(effective):
            start, end = C.c_int(), C.c_int()
            check(lib, lib.denise_cuda_m9_segment_bounds(
                cfg.nt, segments, segment, C.byref(start), C.byref(end)))
            assert (start.value, end.value) == (
                segment*cfg.nt//effective, (segment+1)*cfg.nt//effective)
            assert start.value == previous and end.value > start.value
            previous = end.value
        assert previous == cfg.nt

        checkpoints = e4.payload(lib, replay_context)
        e4.assert_same(checkpoints, expected_checkpoint_payload(
            lib, full, cfg, segments))
        for segment in sorted({0, effective//2, effective-1}):
            start = segment*cfg.nt//effective
            end = (segment+1)*cfg.nt//effective
            restored = e4.probe(
                lib, replay_context, cfg, segment, -1, False)[0]
            wanted = e4.probe(lib, full, cfg, 0, start-1, False)[0]
            e4.assert_same(restored, wanted)
            e4.assert_same(restored, e4.probe(
                lib, replay_context, cfg, segment, -1, False)[0])
            for timestep in sorted({start, end-1}):
                replay_probe = e4.probe(
                    lib, replay_context, cfg, segment, timestep)
                full_probe = e4.probe(lib, full, cfg, 0, timestep)
                for actual, expected in zip(replay_probe, full_probe):
                    e4.assert_same(actual, expected)

        residual = np.random.default_rng(
            4100 + segments + 17*fs + fw).normal(
                size=full_data.shape).astype(np.float32)
        expected = e3.apply(lib, full, cfg, residual)
        actual = e3.apply(lib, replay_context, cfg, residual)
        e4.assert_same(actual, expected)
        e4.assert_same(actual, e3.apply(lib, replay_context, cfg, residual))
        e4.assert_same(checkpoints, e4.payload(lib, replay_context))
        diagnostics = e4.diag(lib, replay_context)
        assert diagnostics.replayed_forward_steps == cfg.nt
        exact_positive_zero(actual[1], fluid)
        assert np.linalg.norm(actual[0][fluid]) > 0
        if np.any(~fluid):
            near = (~fluid) & (np.roll(fluid, 1, 0) | np.roll(fluid, -1, 0) |
                               np.roll(fluid, 1, 1) | np.roll(fluid, -1, 1))
            assert np.linalg.norm(actual[1][near]) > 0

        cpu_image = e3.cpu_jt(cpu, cfg, residual)
        cpu_metrics = [relative(actual[k], cpu_image[k]) for k in (0, 1)]
        dl, dm = fluid_j.directions(a)["joint"]
        oracle_j, gl, gm, oracle_dot = ref.products(
            exp, a["source"], dl.astype(float), dm.astype(float),
            residual.astype(float), fs)
        oracle = np.stack((gl, gm))
        oracle_metrics = [relative(actual[k], oracle[k]) for k in (0, 1)]
        tangent = e2.gpu_j(lib, full, cfg, dl, dm)[0]
        lhs = float(np.sum(tangent.astype(float)*residual.astype(float)))
        rhs = float(np.sum(dl.astype(float)*actual[0]) +
                    np.sum(dm.astype(float)*actual[1]))
        scale = max(np.linalg.norm(tangent)*np.linalg.norm(residual),
                    np.hypot(np.linalg.norm(dl), np.linalg.norm(dm))*
                    np.linalg.norm(actual), 1e-300)
        assert abs(lhs-rhs) <= 7e-5*scale
        assert e1.metrics(tangent, oracle_j)["rel_l2"] <= 1e-5
        RECORDS.append({
            "replay": [mode, pattern, fs, fw, segments],
            "diagnostics": {name: getattr(diagnostics, name)
                            for name, _ in e4.Replay._fields_},
            "cpu": cpu_metrics, "independent": oracle_metrics,
            "oracle_dot": oracle_dot,
            "cuda_dot_scaled": abs(lhs-rhs)/scale,
            "checkpoint_bytes_immutable": True,
            "fluid_mu_positive_zero": True,
        })
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(replay_context)))
        check(lib, lib.denise_cuda_m9_destroy(C.byref(full)))
    assert e1.ledger(lib)[:3] == (0, 0, 0)


@pytest.mark.parametrize("nx,ny,nt,selected", [
    (5, 5, 41, False), (51, 43, 97, True)])
def test_fluid_automatic_full_and_segmented_selection(
        fluid_replay, nx, ny, nt, selected):
    _, libs = fluid_replay
    lib = libs["fma"]
    cfg, a = e2.fixture(0, nx=nx, ny=ny, nt=nt, cpml=False)
    a["m"][:max(1, ny//2)] = 0.0
    mask = signed_fluid(a)
    full = e3.create(lib, cfg)
    automatic = e4.create(lib, cfg, min(32, nt), 1)
    try:
        d = e4.diag(lib, automatic)
        assert bool(d.selected_replay) == selected
        check(lib, lib.denise_cuda_m9_prepare(full))
        check(lib, lib.denise_cuda_m9_prepare(automatic))
        residual = np.random.default_rng(4200 + nt).normal(
            size=(nt, cfg.receiver_count, 2)).astype(np.float32)
        expected = e3.apply(lib, full, cfg, residual)
        actual = e3.apply(lib, automatic, cfg, residual)
        e4.assert_same(actual, expected)
        exact_positive_zero(actual[1], mask)
        state, operands = e4.probe(lib, automatic, cfg, 0, 0)
        assert np.isfinite(state).all() and np.isfinite(operands).all()
        RECORDS.append({"automatic": selected, "shape": [nx, ny, nt],
                        "diagnostics": {name: getattr(d, name)
                                        for name, _ in e4.Replay._fields_}})
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(automatic)))
        check(lib, lib.denise_cuda_m9_destroy(C.byref(full)))
    assert e1.ledger(lib)[:3] == (0, 0, 0)


def replay_layout(lib, cfg):
    before = e1.ledger(lib)
    context = e4.create(lib, cfg, 7)
    try:
        replay_diagnostics = e4.diag(lib, context)
        forward_diagnostics = forward.diagnostics(lib, context)
        owned = e1.ledger(lib)
        return ([getattr(replay_diagnostics, name) for name, _ in e4.Replay._fields_],
                [forward_diagnostics.mandatory_bytes,
                 forward_diagnostics.owned_bytes,
                 forward_diagnostics.host_metadata_bytes],
                tuple(owned[k]-before[k] for k in range(3)))
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(context)))


@pytest.mark.parametrize("mode", ["nofma", "fma"])
def test_fluid_replay_layout_matches_solid(mode, fluid_replay):
    _, libs = fluid_replay
    lib = libs[mode]
    solid_cfg, solid = e2.fixture(1, nx=11, ny=9, nt=19, cpml=True)
    fluid_cfg, fluid = e2.fixture(1, nx=11, ny=9, nt=19, cpml=True)
    fluid["m"][:4] = 0.0
    signed_fluid(fluid)
    assert replay_layout(lib, solid_cfg) == replay_layout(lib, fluid_cfg)
    assert e1.ledger(lib)[:3] == (0, 0, 0)
    RECORDS.append({"resource_invariance": mode, "fluid_specific_bytes": 0})


def multishot_case(shots, *, nt=19, fs=1, fw=3):
    _, a, exp = forward.fixture("offset", fs, fw, nt=nt)
    positions = ((5, 6), (15, 8), (10, 14))[:shots]
    exp = replace(exp, sources=positions)
    data = [np.random.default_rng(4300+k).normal(
        size=(nt, len(exp.receivers), 2)).astype(np.float32)
        for k in range(shots)]
    owner = driver.RequestOwner(exp, data)
    owner.request.free_surface = fs
    for k, samples in enumerate(owner.sources):
        samples *= np.float32(1.0 + .125*k)
        samples[:] = np.roll(samples, k)
    return a, exp, data, owner


def independent_request(exp, owner, data, fs):
    image = np.zeros((2, exp.ny, exp.nx), np.float64)
    for k, position in enumerate(exp.sources):
        shot_exp = replace(exp, sources=(position,))
        _, gl, gm, _ = ref.products(
            shot_exp, owner.sources[k].astype(float),
            np.zeros_like(exp.lam), np.zeros_like(exp.mu),
            data[k].astype(float), fs)
        image += np.stack((gl, gm))
    return image


def request_run(lib, owner):
    result = driver.MigrationResult()
    rc = lib.denise_cuda_m9_migrate_request(
        C.byref(owner.request), C.byref(result))
    if rc:
        return rc, {}, None
    diagnostics = {name: getattr(result, name) for name, _ in
                   driver.MigrationResult._fields_
                   if name not in ("image_lambda_raw", "image_mu_raw")}
    image = np.stack((
        np.ctypeslib.as_array(result.image_lambda_raw,
                              shape=(result.cell_count,)).copy(),
        np.ctypeslib.as_array(result.image_mu_raw,
                              shape=(result.cell_count,)).copy()))
    lib.denise_cuda_m9_migration_result_destroy(C.byref(result))
    assert not result.image_lambda_raw and not result.image_mu_raw
    return rc, diagnostics, image


@pytest.mark.parametrize("shots", [1, 2, 3])
def test_ordered_fluid_request_fp64_accumulation(fluid_replay, shots):
    _, libs = fluid_replay
    lib = libs["fma" if shots == 2 else "nofma"]
    a, exp, data, owner = multishot_case(shots)
    rc, diagnostics, actual = request_run(lib, owner)
    assert rc == 0
    assert diagnostics["shots_completed"] == shots
    assert diagnostics["cell_count"] == exp.nx*exp.ny
    assert diagnostics["trajectory_bytes"] == 16*exp.nt*exp.nx*exp.ny
    assert diagnostics["maximum_shot_data_bytes"] == 8*exp.nt*len(exp.receivers)
    assert diagnostics["initial_forward_steps"] == exp.nt
    assert diagnostics["segment_count"] in (0, min(32, exp.nt))
    assert np.isfinite(actual).all()
    fluid = a["m"] == 0
    exact_positive_zero(actual[1].reshape(a["m"].shape), fluid)
    expected = independent_request(exp, owner, data, 1)
    metrics = [relative(actual[k].reshape(expected[k].shape), expected[k])
               for k in (0, 1)]

    manual = np.zeros_like(actual)
    for k, shot in enumerate(owner.shots):
        cfg, _ = e2.config_from_exp(replace(exp, sources=(exp.sources[k],)), 1)
        cfg.source_i = shot.source_i
        cfg.source_j = shot.source_j
        cfg.source_samples = fp(owner.sources[k])
        context = e3.create(lib, cfg)
        try:
            check(lib, lib.denise_cuda_m9_prepare(context))
            manual += e3.apply(lib, context, cfg, data[k]).reshape(manual.shape)
        finally:
            check(lib, lib.denise_cuda_m9_destroy(C.byref(context)))
    e4.assert_same(actual, manual)
    rc, _, repeated = request_run(lib, owner)
    assert rc == 0
    e4.assert_same(actual, repeated)
    assert e1.ledger(lib)[:3] == (0, 0, 0)
    RECORDS.append({"request_shots": shots, "independent": metrics,
                    "ordered_fp64_byte_identical": True,
                    "fluid_mu_positive_zero": True})


def test_fluid_request_failure_transaction_and_retry(fluid_replay):
    _, libs = fluid_replay
    lib = libs["nofma"]
    _, _, _, owner = multishot_case(2, nt=7, fs=1, fw=0)
    lib.denise_cuda_m9_replay_fault(0)
    rc, _, fresh = request_run(lib, owner)
    assert rc == 0
    sites = e1.ledger(lib)[3]
    assert e1.ledger(lib)[:3] == (0, 0, 0)
    shot2 = 0
    output_failures = 0
    for at in range(1, sites+1):
        lib.denise_cuda_m9_replay_fault(at)
        result = driver.MigrationResult()
        C.memset(C.byref(result), 0x7f, C.sizeof(result))
        rc = lib.denise_cuda_m9_migrate_request(
            C.byref(owner.request), C.byref(result))
        assert rc != 0
        assert not result.image_lambda_raw and not result.image_mu_raw
        assert result.cell_count == 0 and result.shots_completed == 0
        assert e1.ledger(lib)[:3] == (0, 0, 0)
        message = lib.denise_cuda_m9_migration_last_error().decode()
        shot2 += "shot 2" in message
        output_failures += "output" in message
    assert shot2 > 0 and output_failures > 0

    bad = driver.RequestOwner(replace(ref.fixture("horizontal", fs=1, fw=0, nt=7),
                                      sources=((5, 6),)),
                              [np.ones((7, 4, 2), np.float32)], source_type=5)
    result = driver.MigrationResult()
    lib.denise_cuda_m9_fault(0)
    assert lib.denise_cuda_m9_migrate_request(
        C.byref(bad.request), C.byref(result)) != 0
    assert not result.image_lambda_raw and not result.image_mu_raw
    assert e1.ledger(lib)[3] == 0

    lib.denise_cuda_m9_fault(0)
    rc, _, recovered = request_run(lib, owner)
    assert rc == 0
    e4.assert_same(fresh, recovered)
    assert e1.ledger(lib)[:3] == (0, 0, 0)
    RECORDS.append({"request_fault_sites": sites, "shot2_failures": shot2,
                    "publication_failures": output_failures,
                    "fresh_retry_byte_identical": True})


@pytest.mark.parametrize("mode", ["nofma", "fma"])
def test_fluid_replay_faults_and_recovery(fluid_replay, mode):
    _, libs = fluid_replay
    lib = libs[mode]
    cfg, a, _ = forward.fixture("horizontal", 1, 1, nt=7)
    signed_fluid(a)
    residual = np.ones((cfg.nt, cfg.receiver_count, 2), np.float32)
    counts = {}
    for operation in ("create", "migration", "probe", "checkpoints"):
        def attempt(at):
            lib.denise_cuda_m9_fault(0)
            context = P()
            if operation != "create":
                context = e4.create(lib, cfg, 3)
                check(lib, lib.denise_cuda_m9_prepare(context))
            lib.denise_cuda_m9_replay_fault(at)
            out = np.full((2, cfg.ny, cfg.nx), -211.25)
            saved = out.tobytes()
            if operation == "create":
                rc = lib.denise_cuda_m9_create_replay(
                    C.byref(cfg), C.byref(e1.Options()), 3, 0,
                    C.byref(context))
            elif operation == "migration":
                rc = lib.denise_cuda_m9_migrate(
                    context, fp(residual), residual.size,
                    e3.dp(out[0]), e3.dp(out[1]), cfg.nx*cfg.ny)
            elif operation == "probe":
                out = np.full((17, cfg.ny, cfg.nx), -211.25, np.float32)
                saved = out.tobytes()
                rc = lib.denise_cuda_m9_test_replay_probe(
                    context, 2, 6, fp(out[:13]), fp(out[13:]))
            else:
                d = e4.diag(lib, context)
                out = np.full(d.checkpoint_payload_bytes//4, -211.25,
                              np.float32)
                saved = out.tobytes()
                rc = lib.denise_cuda_m9_test_checkpoints(
                    context, fp(out), out.size)
            count = e1.ledger(lib)[3]
            if at:
                assert rc != 0 and out.tobytes() == saved
            else:
                check(lib, rc)
            check(lib, lib.denise_cuda_m9_destroy(C.byref(context)))
            assert e1.ledger(lib)[:3] == (0, 0, 0)
            return count
        count = attempt(0)
        for at in range(1, count+1):
            attempt(at)
        counts[operation] = count
    lib.denise_cuda_m9_fault(0)
    RECORDS.append({"fluid_replay_faults": mode, **counts})


MODE2_CASES = [
    ("homogeneous", 0, 0, 12, 1),
    ("horizontal", 1, 0, 43, 2),
    ("offset", 1, 3, 43, 1),
]


@pytest.mark.integration
@pytest.mark.optional_prerequisite
@pytest.mark.parametrize("pattern,fs,fw,nt,shots", MODE2_CASES)
def test_fluid_mode2_cpu_direct_independent(
        tmp_path, denise_binary, mpiexec, cuda_mode2_binary, fluid_replay,
        pattern, fs, fw, nt, shots):
    _, libs = fluid_replay
    lib = libs["fma"]
    _, a, base = forward.fixture(pattern, fs, fw, nt=nt)
    positions = ((5, 6), (15, 8))[:shots]
    exp = replace(base, sources=positions)
    data = [np.random.default_rng(4500+k).normal(
        size=(nt, len(exp.receivers), 2)).astype(np.float32)
        for k in range(shots)]
    owner = driver.RequestOwner(exp, data)
    owner.request.free_surface = fs
    rc, diagnostics, direct = request_run(lib, owner)
    assert rc == 0
    independent = independent_request(exp, owner, data, fs)
    paths = driver._write_mode2_case(tmp_path, exp, data)
    inp = tmp_path/"denise.inp"
    inp.write_text(inp.read_text().replace("FREE_SURF =0", f"FREE_SURF ={fs}"))
    prepared = {path: path.read_bytes() for path in (tmp_path/"prepared").glob("*")}

    cpu = driver._run_denise(tmp_path, denise_binary, mpiexec)
    assert cpu.returncode == 0 and "M9 MODE=2 backend: CPU-M9" in cpu.stdout
    cpu_image = np.stack([np.fromfile(paths[name], np.float64)
                          for name in ("lambda", "mu")])
    cuda = driver._run_denise(tmp_path, cuda_mode2_binary, mpiexec)
    assert cuda.returncode == 0, cuda.stdout
    assert "M9 MODE=2 backend: CUDA-M9e-4" in cuda.stdout
    assert "CPU-M9\n" not in cuda.stdout
    actual = np.stack([np.fromfile(paths[name], np.float64)
                       for name in ("lambda", "mu")])
    e4.assert_same(actual, direct)
    cpu_metrics = [relative(actual[k], cpu_image[k]) for k in (0, 1)]
    independent_metrics = [relative(
        actual[k].reshape(independent[k].shape), independent[k]) for k in (0, 1)]
    exact_positive_zero(actual[1].reshape(a["m"].shape), a["m"] == 0)
    assert all(path.read_bytes() == content for path, content in prepared.items())
    RECORDS.append({"mode2": [pattern, fs, fw, nt, shots],
                    "selected_replay": bool(diagnostics["segment_count"]),
                    "cpu": cpu_metrics, "independent": independent_metrics,
                    "direct_byte_identical": True,
                    "fluid_mu_positive_zero": True})


@pytest.mark.integration
@pytest.mark.optional_prerequisite
@pytest.mark.parametrize("fault", ["ranks", "unavailable", "shot2"])
def test_fluid_mode2_fail_closed(
        tmp_path, denise_binary, mpiexec, cuda_mode2_binary, fault):
    _, _, base = forward.fixture("horizontal", 1, 0, nt=12)
    exp = replace(base, sources=((5, 6), (15, 8)))
    data = [np.ones((exp.nt, len(exp.receivers), 2), np.float32),
            np.full((exp.nt, len(exp.receivers), 2), 2, np.float32)]
    paths = driver._write_mode2_case(tmp_path, exp, data)
    inp = tmp_path/"denise.inp"
    inp.write_text(inp.read_text().replace("FREE_SURF =0", "FREE_SURF =1"))
    env = os.environ.copy()
    ranks = 1
    if fault == "ranks":
        ranks = 2
        inp.write_text(inp.read_text().replace("NPROCX =1", "NPROCX =2"))
    elif fault == "unavailable":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        Path(str(paths["vx1"]).replace("shot_1", "shot_2")).write_bytes(b"bad")
    completed = subprocess.run(
        [mpiexec, "-n", str(ranks), str(cuda_mode2_binary),
         "denise.inp", "workflow.inp"], cwd=tmp_path, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert completed.returncode != 0, completed.stdout
    assert "M9 MODE=2 backend: CPU-M9" not in completed.stdout
    assert not paths["lambda"].exists() and not paths["mu"].exists()
    marker = {"ranks": "unsupported envelope",
              "unavailable": "migration failed", "shot2": "shot_2"}[fault]
    assert marker in completed.stdout
