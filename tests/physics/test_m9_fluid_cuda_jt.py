"""FLUID-4D restricted CUDA transpose and direct FULL raw migration gates."""
import ctypes as C
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.physics import test_m9e1_cuda_elastic_psv_forward as e1
from tests.physics import test_m9e2_cuda_free_surface_born as e2
from tests.physics import test_m9e3_cuda_jt_migration as e3
from tests.physics.test_m9e4_cuda_replay_mode2 import replay
from tests.physics import test_m9_fluid_cuda_forward as forward
from tests.physics import test_m9_fluid_cuda_j as fluid_j
from tests.physics.test_m9e2_cuda_free_surface_born import backends
from tests.physics.test_m9e3_cuda_jt_migration import jt
from tests.physics.test_m9_fluid_cuda_forward import fluid_cuda
from tests.utilities import m9_fluid_restricted_reference as ref
from tests.utilities import zero_shear_reference as zero

F, P, fp, check = e1.F, e1.P, e1.fp, e1.check
RECORDS = []


@pytest.fixture(scope='session')
def restricted_jt(fluid_cuda):
    yield fluid_cuda
    path = os.environ.get('DENISE_FLUID4D_EVIDENCE')
    if path:
        Path(path).write_text(json.dumps(RECORDS, indent=2))


def create(lib, cfg):
    return e3.create(lib, cfg)


def relative(actual, expected):
    metric = e3.metrics(actual, expected)
    assert metric['rel_l2'] <= 6e-5, metric
    return metric


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
@pytest.mark.parametrize('occupancy', list(itertools.product((0, 1), repeat=4)))
def test_rharmonic_all_corner_occupancies(restricted_jt, mode, occupancy):
    _, libs = restricted_jt
    lib = libs[mode]
    cfg, a = e2.fixture(0, nt=1, nx=7, ny=7, cpml=False)
    a['l'][:] = 4.; a['m'][:] = 3.; a['r'][:] = 1.
    points = ((2, 2), (2, 3), (3, 2), (3, 3))
    for n, (point, fluid) in enumerate(zip(points, occupancy)):
        if fluid:
            a['m'][point] = np.float32(-0. if n & 1 else 0.)
    c = create(lib, cfg)
    try:
        q = np.zeros((4, 7, 7))
        q[3, 2, 2] = -3.25
        got = e3.blocks(lib, c, cfg, 9, q=q)[3]
        expected_mu = zero.corner_vjp(zero.material(a['r'], a['l'], a['m']), q[3])
        expected = np.zeros_like(got); expected[1] = expected_mu
        np.testing.assert_array_equal(got, expected)
        assert np.isfinite(got).all()
        if any(occupancy):
            assert np.count_nonzero(got) == 0
        else:
            assert np.count_nonzero(got[1]) == 4
        RECORDS.append({'corner_occupancy': [mode, list(occupancy)],
                        'nonzero': int(np.count_nonzero(got[1]))})
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
def test_positive_subnormal_is_solid_in_rharmonic_and_projection(restricted_jt, mode):
    _, libs = restricted_jt
    lib = libs[mode]
    cfg, a = e2.fixture(0, nt=1, nx=7, ny=7, cpml=False)
    a['l'][:] = 4.; a['m'][:] = 3.; a['r'][:] = 1.
    tiny = np.nextafter(np.float32(0), np.float32(1))
    a['m'][2, 2] = tiny
    c = create(lib, cfg)
    try:
        q = np.zeros((4, 7, 7)); q[3, 2, 2] = tiny
        got = e3.blocks(lib, c, cfg, 9, q=q)[3][1]
        expected = zero.corner_vjp(zero.material(a['r'], a['l'], a['m']), q[3])
        np.testing.assert_array_equal(got, expected)
        assert got[2, 2] > 0 and got[2, 2].view(np.uint64) != 0
        maps = np.empty(5*(cfg.ny+4)*(cfg.nx+4), np.float32)
        profiles = np.empty(12*(cfg.nx+cfg.ny), np.float32)
        source = np.empty(cfg.nt, np.float32)
        geometry = np.empty(3*cfg.receiver_count, np.int32)
        check(lib, lib.denise_cuda_m9_test_static(c, fp(maps), fp(profiles), fp(source), e1.ip(geometry)))
        copied_mu = maps.reshape(5, cfg.ny+4, cfg.nx+4)[1, 4, 4]
        assert copied_mu.view(np.uint32) == tiny.view(np.uint32)
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
def test_direct_solid_mu_survives_mixed_corner_skip(restricted_jt, mode):
    _, libs = restricted_jt
    lib = libs[mode]
    cfg, a = e2.fixture(0, nt=1, nx=7, ny=7, cpml=False)
    a['l'][:] = 4.; a['m'][:] = 3.; a['r'][:] = 1.; a['m'][2, 3] = -0.
    c = create(lib, cfg)
    try:
        fields = np.zeros((5, 11, 11)); fields[2, 4, 4] = 2.
        bg = np.zeros((4, 7, 7), np.float32); bg[0, 2, 2] = 5.
        q = np.zeros((4, 7, 7)); q[3, 2, 2] = 7.
        got = e3.blocks(lib, c, cfg, 6, f=fields, bg=bg, q=q)[3]
        assert got[1, 2, 2] == 20.0
        assert np.count_nonzero(got[1]) == 1
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
@pytest.mark.parametrize('pattern,fs,fw', [
    ('homogeneous', 0, 0), ('horizontal', 0, 3),
    ('horizontal', 1, 0), ('offset', 1, 3), ('vertical', 0, 0)])
def test_full_restricted_jt_oracle_dot_projection_and_migration(
        restricted_jt, mode, pattern, fs, fw):
    cpu, libs = restricted_jt
    lib = libs[mode]
    cfg, a, exp = forward.fixture(pattern, fs, fw, nt=36)
    fluid = a['m'] == 0
    a['m'][fluid] = np.where(np.indices(a['m'].shape)[1][fluid] & 1, -0., 0.)
    dl, dm = fluid_j.directions(a)['joint']
    c = create(lib, cfg)
    try:
        check(lib, lib.denise_cuda_m9_prepare(c))
        j = e2.gpu_j(lib, c, cfg, dl, dm)[0]
        rng = np.random.default_rng(940 + fs + fw)
        residual = np.asarray(j + .03*rng.normal(size=j.shape), np.float32)
        got = e3.apply(lib, c, cfg, residual)
        cpu_image = e3.cpu_jt(cpu, cfg, residual)
        prepared = ref.trajectory(exp, a['source'], fs)
        oracle_j, gl, gm, oracle_dot = ref.products(
            exp, a['source'], dl.astype(float), dm.astype(float),
            residual.astype(float), fs, prepared)
        oracle_image = np.stack((gl, gm))
        record = {'case': f'{mode}/{pattern}/fs{fs}/fw{fw}',
                  'independent': [relative(got[k], oracle_image[k]) for k in (0, 1)],
                  'cpu': [relative(got[k], cpu_image[k]) for k in (0, 1)],
                  'oracle_dot': oracle_dot}
        lhs = float(np.sum(j.astype(float)*residual.astype(float)))
        rhs = float(np.sum(dl.astype(float)*got[0]) + np.sum(dm.astype(float)*got[1]))
        scale = max(np.linalg.norm(j)*np.linalg.norm(residual),
                    np.hypot(np.linalg.norm(dl), np.linalg.norm(dm))*np.linalg.norm(got), 1e-300)
        record['cuda_dot'] = {'absolute_residual': abs(lhs-rhs), 'scaled': abs(lhs-rhs)/scale}
        assert abs(lhs-rhs) <= 7e-5*scale, record['cuda_dot']
        assert e1.metrics(j, oracle_j)['rel_l2'] <= 1e-5
        assert np.all(got[1][fluid].view(np.uint64) == 0)
        assert np.linalg.norm(got[0][fluid]) > 0
        if np.any(~fluid):
            near = (~fluid) & (np.roll(fluid, 1, 0) | np.roll(fluid, -1, 0) |
                               np.roll(fluid, 1, 1) | np.roll(fluid, -1, 1))
            assert np.linalg.norm(got[1][near]) > 0
        repeated = e3.apply(lib, c, cfg, residual)
        assert got.tobytes() == repeated.tobytes()
        direct = e3.images(cfg)
        check(lib, lib.denise_cuda_m9_migrate(c, fp(residual), residual.size,
                                              e3.dp(direct[0]), e3.dp(direct[1]), fluid.size))
        assert got.tobytes() == direct.tobytes()
        RECORDS.append(record)
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
    assert e1.ledger(lib)[:3] == (0, 0, 0)


def test_restricted_material_basis_transpose(restricted_jt):
    _, libs = restricted_jt
    lib = libs['nofma']
    cfg, a, _ = forward.fixture('offset', 1, 0, nt=8)
    c = create(lib, cfg)
    rng = np.random.default_rng(944)
    residual = rng.normal(size=(cfg.nt, cfg.receiver_count, 2)).astype(np.float32)
    try:
        check(lib, lib.denise_cuda_m9_prepare(c))
        image = e3.apply(lib, c, cfg, residual)
        columns = 0; worst = 0.
        for channel, mask in ((0, np.ones_like(a['m'], bool)), (1, a['m'] != 0)):
            for point in zip(*np.nonzero(mask)):
                dl = np.zeros_like(a['m']); dm = np.zeros_like(a['m'])
                (dl if channel == 0 else dm)[point] = 1.
                column = e2.gpu_j(lib, c, cfg, dl, dm)[0]
                lhs = float(np.sum(column.astype(float)*residual.astype(float)))
                rhs = float(image[channel][point])
                scale = max(np.linalg.norm(column.astype(float))*np.linalg.norm(residual.astype(float)),
                            abs(rhs), 1e-300)
                error = abs(lhs-rhs)/scale
                assert error <= 7e-5, (channel, point, error)
                worst = max(worst, error); columns += 1
        assert columns == a['m'].size + np.count_nonzero(a['m'])
        assert np.all(image[1][a['m'] == 0].view(np.uint64) == 0)
        RECORDS.append({'restricted_basis_columns': columns, 'worst_scaled': worst})
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))


def test_invalid_residual_image_transaction_and_recovery(restricted_jt):
    _, libs = restricted_jt
    lib = libs['nofma']
    cfg, owner, _ = forward.fixture('horizontal', 0, 0, nt=4)
    c = create(lib, cfg); n = cfg.nx*cfg.ny
    out = np.full((2, cfg.ny, cfg.nx), -197.125); sentinel = out.tobytes()
    good = np.ones((cfg.nt, cfg.receiver_count, 2), np.float32)
    try:
        assert lib.denise_cuda_m9_image_download(c, e3.dp(out[0]), e3.dp(out[1]), n) != 0
        assert out.tobytes() == sentinel
        check(lib, lib.denise_cuda_m9_prepare(c))
        invalid = [good.copy() for _ in range(3)]
        invalid[0].flat[0] = np.nan; invalid[1].flat[0] = np.inf; invalid[2].flat[0] = -np.inf
        for value in invalid:
            assert lib.denise_cuda_m9_apply_jt(c, fp(value), value.size) != 0
            assert lib.denise_cuda_m9_image_download(c, e3.dp(out[0]), e3.dp(out[1]), n) != 0
            assert out.tobytes() == sentinel
            check(lib, lib.denise_cuda_m9_prepare(c))
        assert lib.denise_cuda_m9_apply_jt(c, fp(good), good.size-1) != 0
        assert lib.denise_cuda_m9_apply_jt(c, F(), good.size) != 0
        check(lib, lib.denise_cuda_m9_prepare(c)); image = e3.apply(lib, c, cfg, good)
        assert np.isfinite(image).all()
        assert lib.denise_cuda_m9_image_download(c, e3.D(), e3.dp(out[1]), n) != 0
        assert lib.denise_cuda_m9_image_download(c, e3.dp(out[0]), e3.dp(out[1]), n-1) != 0
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))


def test_projection_fault_is_transactional_and_recovers(restricted_jt):
    _, libs = restricted_jt
    lib = libs['nofma']
    cfg, owner, _ = forward.fixture('horizontal', 0, 0, nt=1)
    c = create(lib, cfg); residual = np.ones((1, cfg.receiver_count, 2), np.float32)
    try:
        def successful_sites(local_cfg):
            lib.denise_cuda_m9_fault(0); local = create(lib, local_cfg)
            try:
                check(lib, lib.denise_cuda_m9_prepare(local)); lib.denise_cuda_m9_fault(0)
                check(lib, lib.denise_cuda_m9_apply_jt(local, fp(residual), residual.size))
                return e1.ledger(lib)[3]
            finally:
                check(lib, lib.denise_cuda_m9_destroy(C.byref(local)))
        sites = successful_sites(cfg)
        # The checked projection is the final launch: gate + launch_check, then
        # the three end-event calls.  A matched solid context must omit exactly
        # those two projection operations.
        solid_cfg, solid_owner, _ = forward.fixture('solid', 0, 0, nt=1)
        assert sites == successful_sites(solid_cfg) + 2
        # Exhaust every fluid JT operation.  Together with the exact +2 delta,
        # this necessarily exercises both the fluid-only projection launch gate
        # and its checked launch result without depending on private call numbers.
        for at in range(1, sites+1):
            lib.denise_cuda_m9_fault(0)
            attempt = create(lib, cfg)
            try:
                check(lib, lib.denise_cuda_m9_prepare(attempt)); lib.denise_cuda_m9_fault(at)
                rc = lib.denise_cuda_m9_apply_jt(attempt, fp(residual), residual.size)
                assert rc != 0, (at, sites, e1.ledger(lib)[3])
                d = e3.JTDiagnostics(); check(lib, lib.denise_cuda_m9_adjoint_diagnostics(attempt, C.byref(d)))
                assert not d.valid
            finally:
                check(lib, lib.denise_cuda_m9_destroy(C.byref(attempt)))
        lib.denise_cuda_m9_fault(0); check(lib, lib.denise_cuda_m9_prepare(c))
        recovered = e3.apply(lib, c, cfg, residual)
        assert np.isfinite(recovered).all()
        RECORDS.append({'fluid_jt_fault_sites': sites, 'fluid_only_projection_operations': 2,
                        'recovered': True})
    finally:
        lib.denise_cuda_m9_fault(0)
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
def test_migration_layout_matches_solid_and_fluid(restricted_jt, mode):
    _, libs = restricted_jt
    lib = libs[mode]; values = []
    for pattern in ('solid', 'horizontal'):
        cfg, owner, _ = forward.fixture(pattern, 1, 3, nt=8)
        c = create(lib, cfg)
        try:
            d = forward.diagnostics(lib, c); jd = e3.JTDiagnostics()
            check(lib, lib.denise_cuda_m9_adjoint_diagnostics(c, C.byref(jd)))
            values.append((d.mandatory_bytes, d.owned_bytes, d.host_metadata_bytes,
                           jd.field_bytes, jd.cpml_bytes, jd.image_bytes, jd.workspace_bytes))
        finally:
            check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
    assert values[0] == values[1]
    assert e1.ledger(lib)[:3] == (0, 0, 0)
