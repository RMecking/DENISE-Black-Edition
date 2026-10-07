"""Restricted FLUID-4C J: unchanged FP64 equations plus real CPU/CUDA products."""
import ctypes as C
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from tests.physics import test_m9e1_cuda_elastic_psv_forward as e1
from tests.physics import test_m9e2_cuda_free_surface_born as e2
from tests.physics import test_m9_fluid_cuda_forward as forward
from tests.physics.test_m9e2_cuda_free_surface_born import backends
from tests.physics.test_m9e3_cuda_jt_migration import jt
from tests.physics.test_m9e4_cuda_replay_mode2 import replay
from tests.physics.test_m9_fluid_cuda_forward import fluid_cuda
from tests.utilities import zero_shear_reference as z
from tests.utilities import m9_fluid_restricted_reference as ref

F, P, fp, check = e1.F, e1.P, e1.fp, e1.check
RECORDS = []


@pytest.fixture(scope='session')
def restricted_cuda(fluid_cuda):
    yield fluid_cuda
    if os.environ.get('DENISE_FLUID4C_EVIDENCE'):
        Path(os.environ['DENISE_FLUID4C_EVIDENCE']).write_text(json.dumps(RECORDS, indent=2))


def directions(a):
    y, x = np.indices(a['m'].shape)
    dl = np.asarray(.013*a['l']*(np.sin(.79*x+.31*y)-.3*np.cos(.27*x-.7*y)), np.float32)
    dm = np.asarray(.017*a['m']*(np.cos(.53*x-.41*y)-.4*np.sin(.63*x+.17*y)), np.float32)
    # Include both legal direction zero signs.
    dm[a['m'] == 0] = -0.
    return {'lambda': (dl, np.zeros_like(dm)), 'solid_mu': (np.zeros_like(dl), dm), 'joint': (dl, dm)}


def reference_products(exp, source, dl, dm, fs, prepared):
    """Observe final locals of the unchanged isolated independent oracle.

    Profiling captures products that its public helper returns only as data.
    No expected wavefield is computed by CPU production or duplicated equations.
    """
    r, tapes, _ = prepared
    captured = {}
    if fs:
        code = r.SurfaceOperator.tangent.__code__
        def capture(frame, event, value):
            if event == 'return' and frame.f_code is code:
                v = frame.f_locals
                state = value[0]
                operands = np.asarray(v['dtape'].strain).copy()
                operands[3] += v['extra']
                captured.update(fields=state[:5].copy(), psi=state[5:].copy(), q=operands)
    else:
        code = r.born_forward.__code__
        def capture(frame, event, value):
            if event == 'return' and frame.f_code is code:
                v = frame.f_locals
                captured.update(fields=np.stack([v['fields'][n] for n in ('vx','vy','sxx','syy','sxy')]),
                    psi=np.stack([v['memories'][n] for n in ('sxx_x','sxy_y','sxy_x','syy_y','vxx','vyx','vxy','vyy')]),
                    q=np.stack([v[n] for n in ('qxx','qyx','qxy','qyy')]))
    previous = sys.getprofile()
    try:
        sys.setprofile(capture)
        data = r.born(exp, tapes, dl.astype(float), dm.astype(float)) if fs else r.born_forward(exp, tapes, dl.astype(float), dm.astype(float))
    finally:
        sys.setprofile(previous)
    assert set(captured) == {'fields', 'psi', 'q'}
    material = z.material(exp.rho, exp.lam, exp.mu)
    return data, captured['fields'], captured['psi'], captured['q'], z.corner_jvp(material, dm.astype(float))


def compare(label, got, expected, exact=False):
    record = {'case': label, 'arrays': {}}
    for name, actual, want in zip(('data', 'fields', 'psi', 'q', 'corner'), got, expected):
        assert np.isfinite(actual).all(), (label, name)
        metric = e1.metrics(actual, want)
        record['arrays'][name] = metric
        # Frozen M9e2 J receiver budgets and FLUID-3 applicable relative bound.
        assert metric['rel_l2'] <= 1e-5 and metric['peak_normalized'] <= 4e-5, (label, name, metric)
        if exact:
            assert e1.same(actual, want), (label, name, metric)
    RECORDS.append(record)


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
@pytest.mark.parametrize('pattern,fs,fw', [
    ('homogeneous',0,0), ('horizontal',0,3), ('horizontal',1,0), ('offset',1,3), ('vertical',0,0)])
def test_restricted_j_reference_and_centered_fd(restricted_cuda, mode, pattern, fs, fw):
    cpu, libs = restricted_cuda
    lib = libs[mode]
    cfg, a, exp = forward.fixture(pattern, fs, fw)
    prepared = ref.trajectory(exp, a['source'], fs)
    c = e2.create(lib, cfg)
    try:
        check(lib, lib.denise_cuda_m9_prepare(c))
        background = e1.download(lib, c, cfg)
        for kind, (dl, dm) in directions(a).items():
            got = e2.gpu_j(lib, c, cfg, dl, dm)
            label = f'{mode}/{pattern}/fs{fs}/fw{fw}/{kind}'
            compare(label + '/CPU', got, e2.cpu_j(cpu, cfg, dl, dm), exact=mode == 'nofma')
            expected = reference_products(exp, a['source'], dl, dm, fs, prepared)
            compare(label + '/independent', got, expected)
            repeated = e2.gpu_j(lib, c, cfg, dl, dm)
            forward.identical(got, repeated)
            forward.identical(background, e1.download(lib, c, cfg))
            # Solid-mu is the zero direction for homogeneous fluid; still tested.
            if np.linalg.norm(got[0]) == 0:
                assert kind == 'solid_mu' and pattern == 'homogeneous'
                continue
            nonlinear = []
            for sign in (1,-1):
                l = np.asarray(a['l'] + sign*.05*dl, np.float32)
                m = np.asarray(a['m'] + sign*.05*dm, np.float32)
                assert np.array_equal(m == 0, a['m'] == 0)
                check(lib, lib.denise_cuda_m9_nonlinear(c, fp(l), fp(m)))
                nonlinear.append(e1.download(lib, c, cfg)[0])
            finite_difference = (nonlinear[0].astype(float)-nonlinear[1].astype(float))/.1
            metric = e1.metrics(got[0], finite_difference)
            assert metric['rel_l2'] <= .004, (label, metric)
            RECORDS.append({'case': label + '/FD', 'epsilon': .05, **metric})
            check(lib, lib.denise_cuda_m9_prepare(c))
            forward.identical(background, e1.download(lib, c, cfg))
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
    assert e1.ledger(lib)[:3] == (0,0,0)


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
@pytest.mark.parametrize('occupancy', list(itertools.product((0,1), repeat=4)))
def test_all_corner_occupancies(restricted_cuda, mode, occupancy):
    cpu, libs = restricted_cuda
    lib = libs[mode]
    cfg, a = e2.fixture(0, nt=8, nx=7, ny=7, cpml=False)
    a['l'][:] = 4; a['m'][:] = 3; a['r'][:] = 1
    cfg.dh=1; cfg.dt=.05
    dm = np.zeros((7,7), np.float32)
    points = ((2,2),(2,3),(3,2),(3,3))
    for n, (point, fluid) in enumerate(zip(points, occupancy)):
        a['m'][point] = (-0. if n%2 else 0.) if fluid else (2,3,5,7)[n]
        dm[point] = (-0. if n%2 else 0.) if fluid else np.float32(.1*(n+1))
    dl = np.zeros_like(dm)
    c = e2.create(lib, cfg)
    try:
        check(lib, lib.denise_cuda_m9_prepare(c))
        got = e2.gpu_j(lib, c, cfg, dl, dm)
        compare(f'occupancy/{mode}/{occupancy}/CPU', got, e2.cpu_j(cpu,cfg,dl,dm), exact=mode=='nofma')
        material = z.material(a['r'],a['l'],a['m'])
        expected = z.corner_jvp(material, dm.astype(float))
        metric = e1.metrics(got[4],expected)
        assert metric['rel_l2'] <= 2e-6 and metric['peak_normalized'] <= 8e-6, metric
        if any(occupancy):
            assert got[4][2,2].view(np.uint32) == 0
        else:
            assert got[4][2,2] > 0
        zeros = material.corner == 0
        assert np.count_nonzero(got[4][zeros]) == 0 and not np.signbit(got[4][zeros]).any()
        RECORDS.append({'corner_occupancy': [mode,list(occupancy)], 'mixed_corner_positive_zero': bool(any(occupancy)), **metric})
    finally:
        check(lib, lib.denise_cuda_m9_destroy(C.byref(c)))
    assert e1.ledger(lib)[:3] == (0,0,0)


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
def test_positive_subnormal_background_nonzero_direction(restricted_cuda, mode):
    cpu, libs = restricted_cuda
    lib = libs[mode]
    cfg, a, _ = forward.fixture('solid', nt=12)
    a['m'][3,3] = np.nextafter(np.float32(0), np.float32(1))
    dl = np.zeros_like(a['m']); dm = np.zeros_like(dl)
    dm[3,3] = a['m'][3,3]
    c = e2.create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c))
        got = e2.gpu_j(lib,c,cfg,dl,dm)
        compare('subnormal/'+mode+'/CPU',got,e2.cpu_j(cpu,cfg,dl,dm),exact=mode=='nofma')
        assert got[4][3,3] > 0 and np.count_nonzero(got[4]) == 4
        expected = z.corner_jvp(z.material(a['r'],a['l'],a['m']),dm.astype(float)).astype(np.float32)
        assert e1.same(got[4],expected)
        # The original all-solid arithmetic still evaluates nonzero dH.
        RECORDS.append({'subnormal_background': mode, 'nonzero_direction_accepted': True,
                        'corner_nonzero_count': int(np.count_nonzero(got[4])), 'corner_oracle_bytes': True})
    finally:
        check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))


@pytest.mark.parametrize('mode', ['nofma', 'fma'])
@pytest.mark.parametrize('operation', ['J', 'surface'])
@pytest.mark.parametrize('invalid', ['fluid_mu', 'fluid_subnormal', 'fluid_negative_subnormal',
    'nan_lambda','nan_mu','inf_lambda','inf_mu','null_lambda','null_mu','count'])
def test_invalid_direction_transaction_and_recovery(restricted_cuda, mode, operation, invalid):
    _, libs = restricted_cuda
    lib = libs[mode]
    cfg, a, _ = forward.fixture('horizontal',fs=1,nt=12)
    c = e2.create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c))
        legal_l, legal_m = directions(a)['joint']
        good = e2.gpu_j(lib,c,cfg,legal_l,legal_m)
        dl, dm = legal_l.copy(), legal_m.copy()
        if invalid.startswith('fluid_'):
            value = np.float32(1) if invalid=='fluid_mu' else np.nextafter(np.float32(0),np.float32(1))
            dm.flat[0] = -value if invalid=='fluid_negative_subnormal' else value
            marker = 'requires dMu=0'
        elif invalid.startswith(('nan','inf')):
            (dl if invalid.endswith('lambda') else dm).flat[-1] = np.nan if invalid.startswith('nan') else np.inf
            marker = 'nonfinite'
        else:
            marker = 'arguments invalid' if operation=='surface' else 'exact direction'
        dlp = F() if invalid=='null_lambda' else fp(dl)
        dmp = F() if invalid=='null_mu' else fp(dm)
        padded = np.full((5,cfg.ny+4,cfg.nx+4),727.,np.float32)
        operands = np.full((4,cfg.ny,cfg.nx),737.,np.float32)
        before = padded.tobytes(), operands.tobytes()
        lib.denise_cuda_m9_fault(0)
        if operation=='J':
            rc=lib.denise_cuda_m9_apply_j(c,dlp,dmp,dl.size-1 if invalid=='count' else dl.size)
        else:
            # Surface has no count parameter; invalid block is its shape guard.
            rc=lib.denise_cuda_m9_test_surface(c,3 if invalid=='count' else 1,fp(padded),fp(operands),fp(operands),dlp,dmp)
        assert rc!=0 and marker in lib.denise_cuda_m9_last_error().decode()
        assert e1.ledger(lib)[3] == 0
        assert before == (padded.tobytes(),operands.tobytes())
        born = e2.born_outputs(cfg)
        for x in born: x.fill(747.)
        bytes_before = [x.tobytes() for x in born]
        assert lib.denise_cuda_m9_born_download(c,*map(fp,born))!=0
        assert bytes_before == [x.tobytes() for x in born]
        d=forward.diagnostics(lib,c)
        assert not d.prepared and not d.valid_steps
        check(lib,lib.denise_cuda_m9_prepare(c))
        recovered=e2.gpu_j(lib,c,cfg,legal_l,legal_m)
        forward.identical(good,recovered)
        RECORDS.append({'invalid_direction':[mode,operation,invalid],'tracked_operations':0,'stale_output_unavailable':True,'recovery_byte_identical':True})
    finally:
        check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
    assert e1.ledger(lib)[:3] == (0,0,0)


@pytest.mark.parametrize('mode',['nofma','fma'])
@pytest.mark.parametrize('pattern',['horizontal','solid'])
def test_direction_preflight_uses_copied_material(restricted_cuda,mode,pattern):
    cpu,libs=restricted_cuda;lib=libs[mode]
    cfg,a,_=forward.fixture(pattern,fs=1,nt=12)
    dl,dm=directions(a)['joint']
    c=e2.create(lib,cfg)
    try:
        check(lib,lib.denise_cuda_m9_prepare(c))
        expected=e2.cpu_j(cpu,cfg,dl,dm)
        before=e2.gpu_j(lib,c,cfg,dl,dm)
        owned=e1.ledger(lib)[:3]
        # Mutate every caller classification after the context has copied it.
        a['m'][:]=3. if pattern=='horizontal' else 0.
        check(lib,lib.denise_cuda_m9_prepare(c))
        got=e2.gpu_j(lib,c,cfg,dl,dm)
        forward.identical(before,got)
        compare('copied/'+mode+'/'+pattern+'/CPU',got,expected,exact=mode=='nofma')
        assert e1.ledger(lib)[:3]==owned
        if pattern=='horizontal':
            # Caller now claims solid, but the context-owned cell is fluid.
            dm.flat[0]=np.nextafter(np.float32(0),np.float32(1))
            lib.denise_cuda_m9_fault(0)
            assert lib.denise_cuda_m9_apply_j(c,fp(dl),fp(dm),dm.size)!=0
            assert 'requires dMu=0' in lib.denise_cuda_m9_last_error().decode()
            assert e1.ledger(lib)[3]==0
        RECORDS.append({'copied_direction_authority':[mode,pattern],'J_bytes_unchanged':True,'persistent_ownership_unchanged':True})
    finally:
        check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
    assert e1.ledger(lib)[:3]==(0,0,0)


@pytest.mark.parametrize('mode',['nofma','fma'])
@pytest.mark.parametrize('kind',['lambda','solid_mu','joint'])
def test_surface_tangent_primitive(restricted_cuda,mode,kind):
    cpu,libs=restricted_cuda;lib=libs[mode]
    cfg,a,exp=forward.fixture('vertical',fs=1,nt=12)
    dl,dm=directions(a)[kind]
    rng=np.random.default_rng(915)
    field=rng.normal(size=(5,cfg.ny,cfg.nx)).astype(np.float32)
    q=rng.normal(size=(4,cfg.ny,cfg.nx)).astype(np.float32)
    bg=rng.normal(size=q.shape).astype(np.float32)
    r=ref.module(1,a['source'])
    coeff=r.surface_coefficients(a['l'][0].astype(float),a['m'][0].astype(float))
    da=coeff[2]*dl[0]+coeff[3]*dm[0]
    fluid=a['m'][0]==0
    assert np.count_nonzero(da[fluid])==0
    assert np.count_nonzero(coeff[3][fluid])>0  # unrestricted partial remains intact
    c=e2.create(lib,cfg)
    try:
        padded=np.zeros((5,cfg.ny+4,cfg.nx+4),np.float32)
        padded[:,2:-2,2:-2]=field
        check(lib,lib.denise_cuda_m9_test_surface(c,1,fp(padded),fp(q),fp(bg),fp(dl),fp(dm)))
        h=float(np.float32(1)/np.float32(cfg.dt/cfg.dh))
        for component,name in ((0,'vx'),(1,'vy')):
            cpu_ghost=np.empty((2,cfg.nx),np.float32)
            assert cpu.m9e2_cpu_ghost(C.byref(cfg),component,fp(field[component]),fp(q),fp(bg),fp(dl),fp(dm),fp(cpu_ghost))==0
            assert e1.same(padded[component,:2,2:-2],cpu_ghost)
            extended=r.extend(field[component].astype(float),name,alpha=coeff[0].astype(np.float32).astype(float),
                qxx=q[0].astype(float),qyx=q[1].astype(float),h_over_dt=h)
            if component==1:
                for depth in (1,2):
                    extended[2-depth] += (2*depth-1)*h*da*bg[0,0]
            metric=e1.metrics(padded[component,:2,2:-2],extended[:2])
            assert metric['rel_l2']<=2e-6 and metric['peak_normalized']<=8e-6,metric
            RECORDS.append({'case':f'surface/{mode}/{kind}/{name}',**metric,'CPU_ghost_bytes':True})
        # Closure consumes restricted tangent operands; the full material
        # injection (including the unchanged generic partials) is tested above by J.
        padded[:,2:-2,2:-2]=field
        check(lib,lib.denise_cuda_m9_test_surface(c,2,fp(padded),fp(q),fp(bg),fp(dl),fp(dm)))
        expected=field[2,0].astype(float)+coeff[1].astype(np.float32).astype(float)*q[0,0]
        metric=e1.metrics(padded[2,2,2:-2],expected)
        assert metric['rel_l2']<=2e-6 and metric['peak_normalized']<=8e-6,metric
        assert np.count_nonzero(padded[3,2,2:-2])==0
        assert e1.same(padded[2,2,2:-2][fluid],field[2,0][fluid])
        RECORDS.append({'case':f'surface/{mode}/{kind}/closure',**metric})
    finally:
        check(lib,lib.denise_cuda_m9_destroy(C.byref(c)))
