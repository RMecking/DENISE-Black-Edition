"""Independent material/acoustic/closure gates; no production fluid support.

Future production adapters must supply real material/forward/J/JT products;
passing only these independent-reference self-checks is not production GREEN.
"""
import itertools

import numpy as np
import pytest

from tests.utilities.zero_shear_reference import (
    acoustic_forward, corner_jvp, corner_vjp, cpml, cpml_t, dense_columns,
    derivative, directions, elastic_forward, material, require_direction,
    require_source_row, require_trial, surface, y_with_surface,
    surface_ghosts, require_homogeneous_products,
)


LOCAL_DOT = 5e-13
HOMOGENEOUS_DATA_L2 = 5e-13
PHASE_SPEED_REL = 0.01
CPML_LATE_NORM_RATIO = 0.05


def fixture(shape=(8, 10), pattern="horizontal"):
    mu = 2+np.arange(np.prod(shape)).reshape(shape)/32
    if pattern == "horizontal":
        mu[:shape[0]//2] = 0
    elif pattern == "vertical":
        mu[:, :shape[1]//2] = 0
    elif pattern == "isolated":
        mu[3, 4] = 0
    elif pattern == "strip":
        mu[:, 3] = 0
    elif pattern == "pocket":
        mu[2:5, 2:5] = 0
    elif pattern == "edges":
        mu[0, 0] = mu[0, -1] = mu[-1, 0] = mu[-1, -1] = 0
        mu[0, 4] = mu[-1, 4] = mu[3, 0] = mu[3, -1] = 0
    return material(np.ones(shape), np.full(shape, 4.), mu)


@pytest.mark.parametrize("mask", list(itertools.product([False, True], repeat=4)))
def test_every_zero_corner_pattern_is_positive_zero_or_original_formula(mask):
    values = np.array([2., 3., 5., 7.])
    values[list(mask)] = 0
    m = material(np.array([[1., 2.], [3., 4.]]), np.ones((2, 2))*4, values.reshape(2, 2))
    expected = 0.0 if any(mask) else 4/(1/2+1/3+1/5+1/7)
    assert m.corner[0, 0] == expected
    assert not np.signbit(m.corner[0, 0])
    assert m.rx[0, 0] == 2/3 and m.ry[0, 0] == 2/4
    dm = np.where(m.fluid, 0, 0.1)
    if any(mask):
        np.testing.assert_array_equal(corner_jvp(m, dm), 0)
        np.testing.assert_array_equal(corner_vjp(m, np.ones((2, 2))), 0)


@pytest.mark.parametrize("pattern", ["horizontal", "vertical", "isolated", "strip", "pocket", "edges", "solid"])
def test_wrapped_contributors_and_corner_transpose(pattern):
    m = fixture(pattern=pattern)
    ny, nx = m.mu.shape
    for j in range(ny):
        for i in range(nx):
            block = m.mu[np.ix_([j, (j+1) % ny], [i, (i+1) % nx])]
            assert (m.corner[j, i] == 0) == bool(np.any(block == 0))
            if np.all(block > 0):
                v = block.ravel()
                assert m.corner[j, i] == 4/(1/v[0]+1/v[1]+1/v[2]+1/v[3])
    rng = np.random.default_rng(716)
    dm = rng.normal(size=m.mu.shape)*~m.fluid
    bar = rng.normal(size=m.mu.shape)
    jv, vb = corner_jvp(m, dm), corner_vjp(m, bar)
    scale = max(np.linalg.norm(jv)*np.linalg.norm(bar), np.linalg.norm(dm)*np.linalg.norm(vb))
    assert abs(np.vdot(jv, bar)-np.vdot(dm, vb)) <= LOCAL_DOT*scale
    np.testing.assert_array_equal(vb[m.fluid], 0.)
    assert not np.signbit(vb[m.fluid]).any()
    # Independent basis matrix, not only a quotient/dot self-check.
    columns = []
    solid_points = np.argwhere(~m.fluid)
    for p in solid_points:
        basis = np.zeros_like(dm)
        basis[tuple(p)] = 1
        columns.append(corner_jvp(m, basis).ravel())
    matrix = np.array(columns).T
    np.testing.assert_allclose(matrix.T@bar.ravel(), vb[~m.fluid], rtol=5e-13, atol=5e-13)


def test_canonical_classification_has_no_threshold_and_is_immutable():
    mu = np.ones((4, 4))
    mu[0, 0] = 0
    mu[1, 1] = np.float32(1e-40)  # Positive FP32 subnormal is still solid.
    m = material(np.ones_like(mu), np.ones_like(mu)*4, mu)
    assert m.fluid[0, 0] and not m.fluid[1, 1]
    assert m.corner[1, 1] > 0
    mu[0, 0] = 1
    assert m.fluid[0, 0]
    with pytest.raises(ValueError):
        m.fluid[0, 0] = False


def test_negative_mu_and_fluid_physical_envelope_not_general_validator_cleanup():
    for rho, lam, mu in ((1, 4, -1), (0, 4, 0), (1, 0, 0), (1, np.inf, 0)):
        with pytest.raises(ValueError):
            material(np.full((4, 4), rho), np.full((4, 4), lam), np.full((4, 4), mu))
    # A finite negative lambda in a physically valid solid remains permitted.
    m = material(np.ones((4, 4)), -np.ones((4, 4)), np.ones((4, 4)))
    assert np.all(m.lam+2*m.mu > 0)


def test_future_manifold_invalid_direction_and_class_change_reject_before_mutation():
    m = fixture()
    output = np.full(m.mu.shape, 37.)
    bad = np.zeros_like(m.mu)
    bad[0, 0] = 1
    with pytest.raises(ValueError, match="fluid dMu"):
        # Adapter MUST validate before touching any output; not production GREEN.
        require_direction(m, bad)
        output.fill(0)
    np.testing.assert_array_equal(output, 37.)
    for row, value in ((0, 1), (6, 0)):
        trial = m.mu.copy()
        trial[row, 0] = value
        with pytest.raises(ValueError, match="classification"):
            require_trial(m, trial)
    for dl, dm in directions(m).values():
        require_direction(m, dm)
        for eps in (-0.05, 0.05):
            trial = m.mu+eps*dm
            require_trial(m, trial)
            np.testing.assert_array_equal(trial[m.fluid], 0)
    columns = list(dense_columns(m))
    assert len(columns) == m.mu.size+np.count_nonzero(~m.fluid)
    assert all(channel != "mu" or not m.fluid[j, i] for channel, j, i in columns)


def test_mixed_corner_vjp_does_not_erase_adjacent_solid_direct_mu():
    m = fixture()
    mixed_bar = np.where(m.corner == 0, 1., 0.)
    np.testing.assert_array_equal(corner_vjp(m, mixed_bar), 0.)
    # Arbitrary direct constitutive operands: 2*(bar_sxx*ex+bar_syy*ey).
    direct = np.where(m.fluid, 0., 2*(0.7*0.3+0.4*0.2))
    assert np.all(direct[4] > 0)
    assert np.any(corner_vjp(m, np.ones_like(m.mu))[4] > 0)


def test_local_dense_centered_fd_enumerates_only_manifold_columns():
    m = fixture(shape=(6, 6))
    columns = list(dense_columns(m))
    fd, analytic = [], []
    for channel, j, i in columns:
        dl, dm = np.zeros_like(m.lam), np.zeros_like(m.mu)
        (dl if channel == "lambda" else dm)[j, i] = 1
        plus = material(m.rho, m.lam+0.05*dl, m.mu+0.05*dm)
        minus = material(m.rho, m.lam-0.05*dl, m.mu-0.05*dm)
        np.testing.assert_array_equal(plus.mu[m.fluid], 0.)
        np.testing.assert_array_equal(minus.mu[m.fluid], 0.)
        fd.append(((plus.corner-minus.corner)/0.1).ravel())
        analytic.append(corner_jvp(m, dm).ravel())
    relative = np.linalg.norm(np.array(fd)-analytic)/np.linalg.norm(analytic)
    assert relative <= 0.004
    print("FLUID_LOCAL_DENSE_FD", len(columns), relative)


def test_homogeneous_pressure_reference_nontrivial_finite_and_zero_shear():
    m = material(np.ones((20, 24)), np.full((20, 24), 4.), np.zeros((20, 24)))
    data, pressure, vx, vy = acoustic_forward()
    study = elastic_forward(m)
    assert np.isfinite(data).all() and np.linalg.norm(data[..., 0]) > 0 and np.linalg.norm(data[..., 1]) > 0
    relative = np.linalg.norm(study["data"]-data)/np.linalg.norm(data)
    assert relative <= HOMOGENEOUS_DATA_L2
    np.testing.assert_array_equal(m.corner, 0.)
    np.testing.assert_array_equal(study["state"][4], 0.)
    np.testing.assert_allclose(study["state"][2], -pressure, rtol=5e-13, atol=5e-13)
    np.testing.assert_array_equal(study["state"][2], study["state"][3])
    assert np.linalg.norm(study["strain"]) > 0
    print("FLUID_ACOUSTIC", "data_norm", np.linalg.norm(data), "relative", relative)


def test_homogeneous_fd4_acoustic_phase_has_declared_dispersion_and_speed():
    nx, ny, dt, nt = 24, 20, 0.1, 120
    kx, ky = 2*np.pi/nx, 2*np.pi/ny
    symbol = lambda k: 2*(9/8*np.sin(k/2)-1/24*np.sin(3*k/2))
    sx, sy = symbol(kx), symbol(ky)
    temporal = 2*np.hypot(sx, sy)
    omega = 2/dt*np.arcsin(dt*temporal/2)
    assert abs(omega/np.hypot(kx, ky)/2-1) <= PHASE_SPEED_REL
    y, x = np.indices((ny, nx))
    p = np.exp(1j*(kx*x+ky*y))
    vx = sx/temporal*np.exp(1j*(kx*(x+0.5)+ky*y+omega*dt/2))
    vy = sy/temporal*np.exp(1j*(kx*x+ky*(y+0.5)+omega*dt/2))
    expected = p*np.exp(-1j*omega*nt*dt)
    for _ in range(nt):
        vx -= dt*derivative(p, 1, True)
        vy -= dt*derivative(p, 0, True)
        p -= 4*dt*(derivative(vx, 1, False)+derivative(vy, 0, False))
    assert np.linalg.norm(p-expected)/np.linalg.norm(expected) < 5e-13


def test_fluid_free_surface_lambda_sensitivity_is_structurally_zero():
    lam = np.array([2., 4., 7.])
    d, alpha, a, da, d_a = surface(lam, np.zeros(3), np.array([1., -3., 2.]), np.zeros(3))
    np.testing.assert_array_equal(d, lam)
    for array, value in ((alpha, 1.), (a, 0.), (da, 0.), (d_a, 0.)):
        np.testing.assert_array_equal(array, value)
    # Alpha ghost and direct A paths contain all top-row lambda dependence.
    # Both derivatives vanish: raw gLambda[fluid top row] must be +0.
    require_source_row(1)
    with pytest.raises(ValueError, match="first surface row"):
        require_source_row(0)
    values = np.arange(80).reshape(8, 10).astype(float)
    normal = y_with_surface(values, "syy")
    assert np.isfinite(normal).all()
    yy = values.copy()
    yy[0] = 0
    gy, gs = surface_ghosts(yy, "syy"), surface_ghosts(values, "sxy")
    for k in (1, 2):
        np.testing.assert_array_equal(gy[2-k], -yy[k])
        np.testing.assert_array_equal(gs[2-k], -values[k-1])
    np.testing.assert_array_equal(9/16*(gs[1]+gs[2])-1/16*(gs[0]+gs[3]), 0)
    # Explicit polynomial vy extension enforces ey=-ex at the surface.
    ex = np.full_like(values, 0.2)
    ey = y_with_surface(values, "vy", alpha=np.ones(10), qxx=ex, dt=0.1)
    np.testing.assert_allclose(0.1*ey[0], -ex[0], atol=1e-14)


@pytest.mark.parametrize("k,a,b", [(1., 0., 1.), (1., -0.2, 0.7), (1.7, -0.13, 0.81)])
def test_scalar_cpml_transpose_against_explicit_matrix(k, a, b):
    p = (k, a, b)
    matrix = np.array([[1/k+a, b], [a, b]])
    x, z = np.array([0.37, -0.81]), np.array([-0.19, 0.51])
    f = np.array(cpml(*x, p))
    t = np.array(cpml_t(*z, p))
    np.testing.assert_allclose(f, matrix@x, atol=1e-16)
    np.testing.assert_allclose(t, matrix.T@z, atol=1e-16)
    scale = max(np.linalg.norm(f)*np.linalg.norm(z), np.linalg.norm(x)*np.linalg.norm(t))
    assert abs(f@z-x@t) <= LOCAL_DOT*scale


@pytest.mark.parametrize("interface,free_surface", [(False, False), (False, True), (True, False), (True, True)])
def test_fluid_cpml_forward_absorption_and_boundary_contract(interface, free_surface):
    rho, lam, mu = np.ones((64, 64)), np.full((64, 64), 4.), np.zeros((64, 64))
    if interface:
        rho[32:] = 1.5
        lam[32:], mu[32:] = 6.75, 3.375
    m = material(rho, lam, mu)
    run = elastic_forward(m, nt=800, fw=10, free_surface=free_surface)
    assert run["finite"] and run["peak_memory"] > 0 and np.linalg.norm(run["data"]) > 0
    assert run["late_energy_ratio"] <= CPML_LATE_NORM_RATIO
    # All-fluid includes fluid at left/right/bottom/top CPML; interface overlaps
    # side CPML. FS=1 disables top y memories, not side x memories.
    if free_surface:
        for key in ("yTrue", "yFalse"):
            k, a, b = run["profiles"][key]
            np.testing.assert_array_equal(k[:10], 1.)
            np.testing.assert_array_equal(a[:10], 0.)
            np.testing.assert_array_equal(b[:10], 1.)
        np.testing.assert_array_equal(run["memory"][[1, 3, 6, 7], :10], 0.)
        np.testing.assert_array_equal(run["state"][3, 0], 0.)
        np.testing.assert_array_equal(run["state"][2, 0], 0.)
    else:
        assert np.any(run["profiles"]["yTrue"][1][:10] != 0)
    assert np.any(run["profiles"]["xTrue"][1][:, :10] != 0)
    assert np.any(run["profiles"]["yTrue"][1][-10:] != 0)
    assert np.linalg.norm(run["strain"][:, m.fluid]) > 0
    if not interface:
        np.testing.assert_array_equal(run["state"][4], 0.)
    print("FLUID_CPML", interface, free_surface, run["late_energy_ratio"], run["peak_memory"])


def test_validator_relaxation_alone_is_insufficient():
    data, _, _, _ = acoustic_forward()
    m = material(np.ones((20, 24)), np.full((20, 24), 4.), np.zeros((20, 24)))
    zero = np.zeros_like(m.mu)
    require_homogeneous_products(m, data, zero, zero, data)
    with pytest.raises(AssertionError):
        require_homogeneous_products(m, np.zeros_like(data), zero, zero, data)
    with pytest.raises(AssertionError):
        require_homogeneous_products(m, data, np.ones_like(zero), zero, data)
    with pytest.raises(AssertionError):
        require_homogeneous_products(m, data, zero, np.ones_like(zero), data)
