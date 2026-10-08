"""FLUID-4B: real CUDA forward only; frozen fluid references stay unchanged."""
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
from tests.physics import test_m9c_elastic_psv_migration_driver as driver
from tests.physics.test_m9e2_cuda_free_surface_born import backends
from tests.physics.test_m9e3_cuda_jt_migration import jt
from tests.physics.test_m9e4_cuda_replay_mode2 import replay, cuda_mode2_binary
from tests.utilities import m9_fluid_restricted_reference as restricted
from tests.utilities import zero_shear_reference as frozen

F, P, fp, check = e1.F, e1.P, e1.fp, e1.check
RECORDS = []


@pytest.fixture(scope='session')
def fluid_cuda(replay):
    yield replay
    if os.environ.get('DENISE_FLUID4B_EVIDENCE'):
        Path(os.environ['DENISE_FLUID4B_EVIDENCE']).write_text(json.dumps(RECORDS, indent=2))


def fixture(pattern='horizontal', fs=0, fw=0, nt=120):
    exp = replace(restricted.fixture(pattern, fs=fs, fw=fw, nt=nt),
                  pml_reflection=1e-3, pml_power=2., pml_kmax=1.)
    cfg, a = e2.config_from_exp(exp, fs)
    a['source'][:] = frozen.prepared_source(nt, .1, fc=.5, t0=3).astype(np.float32)
    return cfg, a, exp


def identical(got, want):
    assert all(e1.same(x, y) for x, y in zip(got, want))


def diagnostics(lib, c):
    d = e1.Diagnostics()
    check(lib, lib.denise_cuda_m9_diagnostics(c, C.byref(d)))
    return d


def compare(label, got, want, *, exact=False, independent=False):
    values = {}
    for name, x, y in zip(('data', 'fields', 'psi', 'strain'), got, want):
        assert np.isfinite(x).all(), (label, name)
        z = e1.metrics(x, y)
        values[name] = z
        # Existing FLUID-3 independent bound; existing CUDA differential bound.
        assert z['rel_l2'] <= (1e-5 if independent else 2e-6), (label, name, z)
        if not independent:
            assert z['peak_normalized'] <= 8e-6, (label, name, z)
        if exact:
            assert e1.same(x, y), (label, name, z)
    RECORDS.append({'case': label, 'arrays': values})


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
@pytest.mark.parametrize('pattern,fs,fw', [
    ('homogeneous', 0, 0), ('horizontal', 0, 3),
    ('horizontal', 1, 0), ('offset', 1, 3), ('vertical', 0, 0)])
def test_fluid_forward_references(fluid_cuda, mode, pattern, fs, fw):
    cpu, libs = fluid_cuda
    lib = libs[mode]
    cfg, a, exp = fixture(pattern, fs, fw)
    c = e2.create(lib, cfg) if fs else e1.create(lib, cfg)
    try:
        check(lib, lib.denise_cuda_m9_prepare(c))
        got = e1.download(lib, c, cfg)
        label = f'{mode}/{pattern}/fs{fs}/fw{fw}'
        compare(label + '/CPU', got, e1.cpu_run(cpu, cfg), exact=mode == 'nofma')
        m = frozen.material(a['r'], a['l'], a['m'])
        ref = frozen.elastic_forward(m, nt=cfg.nt, fw=fw, free_surface=bool(fs), source=a['source'])
        _, tape, data = restricted.trajectory(exp, a['source'], fs)
        strain = np.asarray([t.strain for t in tape]) if fs else tape.strain
        compare(label + '/independent', got, (ref['data'], ref['state'], ref['memory'], strain), independent=True)
        assert e1.metrics(got[0], data)['rel_l2'] <= 1e-5
        assert np.linalg.norm(got[1][2:4]) > 0 and np.linalg.norm(got[3][:, :, a['m'] == 0]) > 0
        assert np.count_nonzero(got[1][4][m.corner == 0]) == 0
        if pattern == 'homogeneous':
            acoustic = frozen.acoustic_forward(source=a['source'])[0]
            z = e1.metrics(got[0], acoustic)
            assert z['rel_l2'] <= 1e-5
            assert e1.same(got[1][2], got[1][3]) and np.count_nonzero(got[1][4]) == 0
            RECORDS.append({'case': label + '/acoustic', **z})
        else:
            assert np.linalg.norm(got[1][4][m.corner > 0]) > 0
            assert np.linalg.norm(got[1][2:4, a['m'] > 0]) > 0
        if fw:
            assert np.linalg.norm(got[2]) > 0
        check(lib, lib.denise_cuda_m9_prepare(c))
        identical(got, e1.download(lib, c, cfg))
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
    assert e1.ledger(lib)[:3] == (0, 0, 0)


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
@pytest.mark.parametrize('fs', [0, 1])
def test_nonlinear_classification_and_recovery(fluid_cuda, mode, fs):
    cpu, libs = fluid_cuda
    lib = libs[mode]
    cfg, a, _ = fixture('horizontal', fs, 3, nt=64)
    # Both signs are present in the copied background.
    mask = a['m'] == 0
    a['m'][0] = -0.
    c = e2.create(lib, cfg)
    try:
        check(lib, lib.denise_cuda_m9_prepare(c))
        background = e1.download(lib, c, cfg)
        trials = []
        for kind in ('identical', 'fluid_lambda', 'solid_material', 'combined', 'zero_sign_flip'):
            l, m = a['l'].copy(), a['m'].copy()
            if kind in ('fluid_lambda', 'combined'):
                l[mask] *= np.float32(1.03)
            if kind in ('solid_material', 'combined'):
                l[~mask] *= np.float32(.98)
                m[~mask] *= np.float32(1.04)
            if kind == 'zero_sign_flip':
                m[mask] = -m[mask]
            trials.append((kind, l, m))
        for kind, l, m in trials:
            check(lib, lib.denise_cuda_m9_nonlinear(c, fp(l), fp(m)))
            got = e1.download(lib, c, cfg)
            trial = e1.Config.from_buffer_copy(cfg)
            setattr(trial, 'lambda', fp(l)); trial.mu = fp(m)
            # Fixed original CPML profiles, as specified by the nonlinear contract.
            maps = np.empty((5, cfg.ny+4, cfg.nx+4), np.float32)
            profiles = np.empty(6*(cfg.nx+cfg.ny), np.float32)
            assert cpu.m9e1_cpu_static(C.byref(cfg), fp(maps), fp(profiles)) == 0
            compare(f'nonlinear/{mode}/fs{fs}/{kind}', got,
                    e1.cpu_run(cpu, trial, profiles=profiles), exact=mode == 'nofma')
            check(lib, lib.denise_cuda_m9_prepare(c))
            identical(background, e1.download(lib, c, cfg))
        for kind, fluid, value in (
                ('fluid_subnormal', True, np.nextafter(np.float32(0), np.float32(1))),
                ('fluid_solid', True, np.float32(3.)),
                ('solid_positive_zero', False, np.float32(0)),
                ('solid_negative_zero', False, np.float32(-0.))):
            m = a['m'].copy()
            index = np.flatnonzero(mask.ravel() if fluid else ~mask.ravel())[0]
            m.flat[index] = value
            before = diagnostics(lib, c)
            owned = e1.ledger(lib)[:3]
            lib.denise_cuda_m9_fault(0)
            assert lib.denise_cuda_m9_nonlinear(c, fp(a['l']), fp(m)) != 0
            assert 'classification change' in lib.denise_cuda_m9_last_error().decode()
            assert e1.ledger(lib) == (*owned, 0)  # no allocation, copy or launch
            after = diagnostics(lib, c)
            assert bytes(before) == bytes(after)
            identical(background, e1.download(lib, c, cfg))
            check(lib, lib.denise_cuda_m9_nonlinear(c, fp(a['l']), fp(a['m'])))
            identical(background, e1.download(lib, c, cfg))
            check(lib, lib.denise_cuda_m9_prepare(c))
            identical(background, e1.download(lib, c, cfg))
            RECORDS.append({'classification_rejection': [mode, fs, kind], 'GPU_operations': 0, 'background_preserved': True})
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
    assert e1.ledger(lib)[:3] == (0, 0, 0)


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
def test_copied_ownership_and_subnormal_solid(fluid_cuda, mode):
    _, libs = fluid_cuda
    lib = libs[mode]
    for pattern in ('homogeneous', 'solid'):
        cfg, a, _ = fixture(pattern, nt=12)
        if pattern == 'solid':
            a['m'].flat[0] = np.nextafter(np.float32(0), np.float32(1))
        c = e2.create(lib, cfg)
        try:
            check(lib, lib.denise_cuda_m9_prepare(c))
            original = e1.download(lib, c, cfg)
            a['m'][:] = 3. if pattern == 'homogeneous' else 0.
            check(lib, lib.denise_cuda_m9_prepare(c))
            identical(original, e1.download(lib, c, cfg))
            zeros = np.zeros((cfg.ny, cfg.nx), np.float32)
            rc = lib.denise_cuda_m9_apply_j(c, fp(zeros), fp(zeros), zeros.size)
            check(lib, rc)
            born = e2.born_outputs(cfg)
            check(lib, lib.denise_cuda_m9_born_download(c, *map(fp, born)))
            assert all(np.isfinite(x).all() and np.count_nonzero(x) == 0 for x in born)
        finally:
            check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
        # Positive subnormal must also pass the migration/replay capability policy.
        if pattern == 'solid':
            a['m'][:] = 3.; a['m'].flat[0] = np.nextafter(np.float32(0), np.float32(1))
            for constructor in ('migration', 'replay'):
                c = P()
                args = [C.byref(cfg), C.byref(e1.Options())]
                if constructor == 'replay': args += [3, 1]
                check(lib, getattr(lib, 'denise_cuda_m9_create_' + constructor)(*args, C.byref(c)))
                check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
        assert e1.ledger(lib)[:3] == (0, 0, 0)


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
def test_later_capabilities_fail_before_mutation(fluid_cuda, mode):
    _, libs = fluid_cuda
    lib = libs[mode]
    cfg, a, exp = fixture('horizontal', fs=1, nt=12)
    c = e2.create(lib, cfg)
    try:
        check(lib, lib.denise_cuda_m9_prepare(c))
        bg = e1.download(lib, c, cfg)
        owned = e1.ledger(lib)[:3]
        zeros = np.zeros((cfg.ny, cfg.nx), np.float32)
        padded = np.full((5, cfg.ny+4, cfg.nx+4), 731., np.float32)
        q = np.full((4, cfg.ny, cfg.nx), 751., np.float32)
        state = np.full((13, cfg.ny, cfg.nx), 761., np.float32)
        sentinels = [v.tobytes() for v in (padded, q, state)]
        operations = [
            ('FLUID-4E', lambda: lib.denise_cuda_m9_test_replay_probe(c, 0, 0, fp(state), fp(q)))]
        for marker, operation in operations:
            before = bytes(diagnostics(lib, c))
            lib.denise_cuda_m9_fault(0)
            assert operation() != 0 and marker in lib.denise_cuda_m9_last_error().decode()
            assert e1.ledger(lib) == (*owned, 0)
            assert bytes(diagnostics(lib, c)) == before
            assert [v.tobytes() for v in (padded, q, state)] == sentinels
            identical(bg, e1.download(lib, c, cfg))
        born = e2.born_outputs(cfg)
        for x in born: x.fill(771.)
        before = [x.tobytes() for x in born]
        assert lib.denise_cuda_m9_born_download(c, *map(fp, born)) != 0
        assert [x.tobytes() for x in born] == before
        for _ in range(3):
            other = P()
            check(lib, lib.denise_cuda_m9_create_migration(C.byref(cfg), C.byref(e1.Options()), C.byref(other)))
            check(lib, lib.denise_cuda_m9_destroy(C.byref(other)))
            assert not other
            assert e1.ledger(lib)[:3] == owned
            for automatic in (0, 1):
                # S=1 is otherwise automatic FULL, not SEGMENTED.
                assert lib.denise_cuda_m9_create_replay(C.byref(cfg), C.byref(e1.Options()), 1, automatic, C.byref(other)) != 0
                assert not other and 'FLUID-4E' in lib.denise_cuda_m9_last_error().decode()
                assert e1.ledger(lib)[:3] == owned
        data = np.ones((cfg.nt, cfg.receiver_count, 2), np.float32)
        owner = driver.RequestOwner(exp, [data]); owner.request.free_surface = 1
        result = driver.MigrationResult()
        C.memset(C.byref(result), 0x7F, C.sizeof(result))
        lib.denise_cuda_m9_fault(0)
        assert lib.denise_cuda_m9_migrate_request(C.byref(owner.request), C.byref(result)) != 0
        assert 'FLUID-4E' in lib.denise_cuda_m9_migration_last_error().decode()
        assert bytes(result) == bytes(C.sizeof(result))
        assert e1.ledger(lib) == (*owned, 0)
        # Forward-only surface blocks remain available.
        for block in (0, 1, 2):
            check(lib, lib.denise_cuda_m9_test_surface(c, block, fp(padded), fp(q), F(), F(), F()))
            assert np.isfinite(padded).all()
        check(lib, lib.denise_cuda_m9_prepare(c))
        identical(bg, e1.download(lib, c, cfg))
        # FLUID-4C now opens legal restricted J and the tangent surface probe.
        check(lib, lib.denise_cuda_m9_apply_j(c, fp(zeros), fp(zeros), zeros.size))
        born = e2.born_outputs(cfg)
        check(lib, lib.denise_cuda_m9_born_download(c, *map(fp, born)))
        assert all(np.isfinite(x).all() and np.count_nonzero(x) == 0 for x in born)
        check(lib, lib.denise_cuda_m9_test_surface(c, 1, fp(padded), fp(q), fp(q), fp(zeros), fp(zeros)))
        RECORDS.append({'staged_capabilities': mode, 'replay_probe_GPU_operations': 0,
                        'migration_request_GPU_operations': 0, 'later_constructor_leaks': 0})
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
    assert e1.ledger(lib)[:3] == (0, 0, 0)


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
@pytest.mark.parametrize('full', [False, True])
def test_fluid_memory_and_repeated_ownership(fluid_cuda, mode, full):
    _, libs = fluid_cuda
    lib = libs[mode]
    sizes = []
    for pattern in ('solid', 'homogeneous'):
        cfg, a, _ = fixture(pattern, fs=int(full), fw=3, nt=12)
        for _ in range(3):
            c = e2.create(lib, cfg) if full else e1.create(lib, cfg)
            try:
                d = diagnostics(lib, c)
                sizes.append((d.mandatory_bytes, d.owned_bytes, d.host_metadata_bytes, e1.ledger(lib)[:3]))
                check(lib, lib.denise_cuda_m9_prepare(c))
            finally:
                check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
            assert not c and e1.ledger(lib)[:3] == (0, 0, 0)
    assert all(value == sizes[0] for value in sizes)
    RECORDS.append({'memory_comparison': [mode, full], 'mandatory_owned_metadata_ledger': sizes[0],
                    'fluid_specific_allocation': 0, 'repeated_destroy_ledger': [0, 0, 0]})


def test_private_fluid_query_host_sanitizers(tmp_path, repository_root):
    objects = []
    for name in ('host', 'migration'):
        obj = tmp_path / (name + '.o')
        subprocess.run(['cc', '-std=c99', '-O1', '-g', '-fsanitize=address,undefined',
                        '-fno-omit-frame-pointer', '-I' + str(repository_root / 'include'),
                        '-c', str(repository_root / f'src/CUDA/m9_elastic_{name}.c'),
                        '-o', str(obj)], check=True)
        objects.append(str(obj))
    executable = tmp_path / 'fluid-host-sanitizers'
    subprocess.run(['g++', '-std=c++14', '-O1', '-g', '-fsanitize=address,undefined',
                    '-fno-omit-frame-pointer', '-I' + str(repository_root / 'include'),
                    str(repository_root / 'tests/utilities/m9_fluid_cuda_host_harness.cpp'),
                    *objects, '-lm', '-o', str(executable)], check=True)
    env = os.environ.copy()
    env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
    result = subprocess.run([str(executable)], env=env, text=True, capture_output=True, check=True)
    assert 'FLUID4B_HOST_PASS' in result.stdout
    RECORDS.append({'host_ASan_UBSan': result.stdout.strip(), 'stderr': result.stderr})


@pytest.mark.integration
@pytest.mark.optional_prerequisite
def test_fluid_mode2_remains_closed(tmp_path, denise_binary, mpiexec, cuda_mode2_binary):
    _, _, exp = fixture('horizontal', fs=1, nt=12)
    data = [np.ones((exp.nt, len(exp.receivers), 2), np.float32)]
    paths = driver._write_mode2_case(tmp_path, exp, data)
    inp = tmp_path / 'denise.inp'
    inp.write_text(inp.read_text().replace('FREE_SURF =0', 'FREE_SURF =1'))
    result = driver._run_denise(tmp_path, cuda_mode2_binary, mpiexec)
    assert result.returncode != 0 and 'FLUID-4E' in result.stdout, result.stdout
    assert 'CPU-M9' not in result.stdout
    assert not paths['lambda'].exists() and not paths['mu'].exists()
    RECORDS.append({'MODE2_fluid': 'FLUID-4E rejection', 'log': result.stdout})
