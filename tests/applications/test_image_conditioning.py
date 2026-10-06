"""Analytic conditioning checks; no solver, compilation, network or asset IO."""
import numpy as np
import pytest

from tools.denise_case import image_conditioning as c


def test_transform_known_coefficients_pure_inputs_cancellation_and_water():
    lam = np.array([[1., -2.], [3., 4.]])
    mu = np.array([[3., 4.], [6., 0.]])
    vp = np.full((2, 2), 3.)
    vs = np.array([[0., 0.], [2., 2.]])
    rho = np.full((2, 2), 5.)
    originals = [v.copy() for v in (lam, mu, vp, vs, rho)]
    result = c.parameter_transform(lam, mu, vp, vs, rho)
    np.testing.assert_array_equal(result['vp'], [[30., -60.], [90., 120.]])
    np.testing.assert_array_equal(result['vs'], [[0., 0.], [0., -160.]])
    np.testing.assert_array_equal(result['lnvp'], result['vp'] * 3.)
    np.testing.assert_array_equal(result['lnvs'], result['vs'] * vs)
    pure_mu = c.parameter_transform(np.zeros((2, 2)), mu, vp, vs, rho)
    np.testing.assert_array_equal(pure_mu['vp'], np.zeros((2, 2)))
    np.testing.assert_array_equal(pure_mu['vs'], [[0., 0.], [120., 0.]])
    pure_lam = c.parameter_transform(lam, np.zeros((2, 2)), vp, vs, rho)
    np.testing.assert_array_equal(pure_lam['vs'], [[0., 0.], [-120., -160.]])
    for original, a in zip(originals, (lam, mu, vp, vs, rho)):
        np.testing.assert_array_equal(a, original)
    assert all(v.dtype == np.float64 for v in result.values())
    assert not any(np.shares_memory(v, lam) for v in result.values())


def test_transform_independent_cotangent_identity():
    # Directional changes of lambda and mu derived independently from their
    # fixed-density definitions, not from the production cotangent expression.
    rng = np.random.default_rng(17)
    shape = (9, 7)
    vp = rng.uniform(1500, 4500, shape)
    vs = rng.uniform(300, 2500, shape)
    vs[:2] = 0
    rho = rng.uniform(1000, 2800, shape)
    dl, dm, dvp, dvs = [rng.normal(size=shape) for _ in range(4)]
    # A complex-step directional derivative has no subtractive cancellation.
    h = 1.e-100
    vpc, vsc = vp + 1j * h * dvp, vs + 1j * h * dvs
    d_lambda = np.imag(rho * (vpc ** 2 - 2 * vsc ** 2)) / h
    d_mu = np.imag(rho * vsc ** 2) / h
    result = c.parameter_transform(dl, dm, vp, vs, rho)
    np.testing.assert_allclose(dl * d_lambda + dm * d_mu,
                               result['vp'] * dvp + result['vs'] * dvs,
                               rtol=3.e-14, atol=5.e-8)
    assert np.all(result['vs'][:2] == 0)
    assert np.all(result['lnvs'][:2] == 0)


@pytest.mark.parametrize('which,bad', [('vp', 0.), ('vs', -1.), ('rho', 0.), ('lambda', np.nan), ('mu', np.inf)])
def test_transform_invalid_inputs(which, bad):
    arrays = {k: np.ones((3, 4)) for k in ('lambda', 'mu', 'vp', 'vs', 'rho')}
    arrays[which][0, 0] = bad
    with pytest.raises(ValueError):
        c.parameter_transform(*(arrays[k] for k in ('lambda', 'mu', 'vp', 'vs', 'rho')))


def test_transform_shape_overflow_complex_and_tiny_physical_vs():
    a = np.ones((3, 4))
    with pytest.raises(ValueError):
        c.parameter_transform(a, a[:, :2], a, a, a)
    with pytest.raises(ValueError):
        c.parameter_transform(a.astype(complex), a, a, a, a)
    with pytest.raises(ValueError):
        c.parameter_transform(a * 1.e308, a, a * 2, a, a * 2)
    result = c.parameter_transform(a, a * 3, a, a * 1.e-100, a)
    np.testing.assert_array_equal(result['vs'], a * 2.e-100)


def test_polynomial_derivatives_independent_analytic_values():
    dh = 20.
    z, x = np.meshgrid(np.arange(7) * dh, np.arange(9) * dh, indexing='ij')
    image = 3 * x ** 2 + 5 * z ** 2 + 7 * x * z + 11 * x - 13 * z + 17
    before = image.copy()
    result = c.derivatives(image, dh)
    np.testing.assert_array_equal(result['dx'][:, 1:-1], (6 * x + 7 * z + 11)[:, 1:-1])
    np.testing.assert_array_equal(result['dz'][1:-1], (10 * z + 7 * x - 13)[1:-1])
    np.testing.assert_array_equal(result['laplacian'][1:-1, 1:-1], np.full((5, 7), 16.))
    for key in ('dx', 'dz', 'laplacian'):
        assert np.isnan(result[key][~result['valid'][key]]).all()
        assert np.isfinite(result[key][result['valid'][key]]).all()
    assert result['valid']['dx'].sum() == 7 * 7
    assert result['valid']['dz'].sum() == 5 * 9
    assert result['valid']['laplacian'].sum() == 5 * 7
    np.testing.assert_array_equal(image, before)


def test_interface_crossings_are_flagged_not_muted():
    image = np.zeros((7, 8)); image[3:] = 2
    water = np.zeros_like(image, dtype=bool); water[:3] = True
    result = c.derivatives(image, 2., water)
    assert not result['crosses_interface']['dx'].any()
    np.testing.assert_array_equal(np.flatnonzero(result['crosses_interface']['dz'][:, 3]), [2, 3])
    np.testing.assert_array_equal(result['dz'][2:4], np.full((2, 8), .5))
    assert result['crosses_interface']['laplacian'][2:4, 1:-1].all()
    assert not result['crosses_interface']['laplacian'][:, [0, -1]].any()
    assert result['dz'][2, 3] != 0  # derivative water need not retain restricted mu zero


def test_derivative_no_periodic_wrap_and_invalid_inputs():
    image = np.zeros((5, 6)); image[:, 0] = 10
    result = c.derivatives(image, 1.)
    assert np.isnan(result['dx'][:, -1]).all()
    assert np.all(result['dx'][:, -2] == 0)
    for invalid in (0., -1., np.nan, np.inf, True):
        with pytest.raises(ValueError):
            c.derivatives(image, invalid)
    with pytest.raises(ValueError):
        c.derivatives(image[:2], 1.)
    with pytest.raises(ValueError):
        c.derivatives(image, 1., np.zeros_like(image))


def test_radial_transfer_symbol_and_evenness():
    h = c.filter_transfer((80, 100), 1., (.02, .04, .1, .15))
    assert h[0, 0] == 0
    assert h[0, 2] == 0
    np.testing.assert_allclose(h[0, 3], .5, atol=1.e-15)
    assert h[0, 4] == 1
    assert h[0, 10] == 1
    np.testing.assert_allclose(h[0, 12], np.cos(np.pi * .02 / .1) ** 2, atol=1.e-15)
    assert h[0, 15] == 0
    for z in range(h.shape[0]):
        for x in range(h.shape[1]):
            assert h[z, x] == h[-z % h.shape[0], -x % h.shape[1]]
    assert h.min() == 0 and h.max() == 1


@pytest.mark.parametrize('mode,expected', [(1, 0.), (3, .5), (7, 1.), (12, np.cos(np.pi * .02 / .1) ** 2), (20, 0.)])
def test_bandpass_fourier_mode_independent_response(mode, expected):
    a = np.broadcast_to(np.cos(2 * np.pi * mode * np.arange(100) / 100), (80, 100)).copy()
    output = c.bandpass(a, 1., (.02, .04, .1, .15), padding=0, taper_cells=0)
    np.testing.assert_allclose(output, a * expected, atol=2.e-14, rtol=2.e-14)


def test_impulse_real_even_zero_mean_and_deterministic():
    impulse = np.zeros((32, 40)); impulse[16, 20] = 1
    output = c.bandpass(impulse, 1., (.02, .04, .1, .15), padding=0, taper_cells=0)
    np.testing.assert_array_equal(output, c.bandpass(impulse, 1., (.02, .04, .1, .15), padding=0, taper_cells=0))
    assert abs(output.sum()) < 1.e-15
    assert output[16, 20] > 0
    assert np.any(output < 0)  # ringing retained, not clipped
    for dz in range(16):
        np.testing.assert_allclose(output[16 + dz], output[16 - dz], atol=1.e-16)
    assert np.isrealobj(output)


def test_zero_padding_edge_response_and_taper_controls():
    edge = np.zeros((40, 50)); edge[:, 0] = 1
    before = edge.copy()
    no_pad = c.bandpass(edge, 1., (.02, .04, .1, .15), padding=0, taper_cells=0)
    pad = c.bandpass(edge, 1., (.02, .04, .1, .15), padding=64, taper_cells=0)
    larger = c.bandpass(edge, 1., (.02, .04, .1, .15), padding=128, taper_cells=0)
    # Prevent claiming zero leakage: padded kernels retain nonzero far response.
    assert np.max(np.abs(pad[:, -1])) > 0
    assert np.max(np.abs(pad[:, -1])) < np.max(np.abs(no_pad[:, -1]))
    assert np.max(np.abs(larger - pad)) < .01
    np.testing.assert_array_equal(c.bandpass(edge, 1., padding=64, taper_cells=10), np.zeros_like(edge))
    taper = c.edge_taper((40, 50), 10)
    assert taper[0].sum() == 0
    assert taper[:, 0].sum() == 0
    assert taper[10, 10] == 1
    np.testing.assert_array_equal(taper, taper[::-1, ::-1])
    np.testing.assert_array_equal(edge, before)


@pytest.mark.parametrize('kwargs', [dict(padding=-1), dict(padding=1.5), dict(taper_cells=30), dict(taper_cells=True), dict(corners=(0, 1, 1, 3)), dict(corners=(0, 1, 2, np.nan))])
def test_bandpass_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        c.bandpass(np.ones((40, 50)), 20., **kwargs)


def test_gain_fixed_known_depths_and_identity_without_mutation():
    z = np.array([20., 450., 650., 1300., 2600., 3480.])
    a = np.full((6, 3), 2.)
    np.testing.assert_array_equal(c.depth_gain(a, z), np.broadcast_to([2., 2., 2., 4., 6., 6.], (3, 6)).T)
    np.testing.assert_array_equal(c.depth_gain(a, z, p=0, cap=10), a)
    np.testing.assert_allclose(c.depth_gain(a, z, p=.5, cap=10)[:, 0], 2 * np.sqrt(np.maximum(z, 650) / 650), rtol=3.e-16)
    assert not np.shares_memory(c.depth_gain(a, z, p=0), a)
    np.testing.assert_array_equal(a, np.full((6, 3), 2.))


@pytest.mark.parametrize('kwargs', [dict(z0=0), dict(p=-1), dict(p=np.nan), dict(cap=.5)])
def test_gain_invalid(kwargs):
    with pytest.raises(ValueError):
        c.depth_gain(np.ones((3, 4)), [20, 40, 60], **kwargs)


def test_display_percentile_independently_expected_counts_and_mask_domain():
    a = np.array([[-4., -2., 0.], [1., 3., 10.]])
    scale = c.display_scale(a, percentile=50)
    assert scale['q'] == 2.5  # sorted absolute values 0,1,2,3,4,10
    assert scale['limits'] == [-2.5, 2.5]
    assert scale['method'] == 'linear'
    assert scale['saturated_count'] == 3
    assert scale['saturated_fraction'] == .5
    mask = np.zeros_like(a, dtype=bool); mask[0] = True
    scale = c.display_scale(a, percentile=100, mask=mask)
    assert scale['q'] == 4
    assert scale['scale_domain_count'] == 3
    assert scale['total_count'] == 6
    assert scale['saturated_count'] == 1


def test_zero_display_compression_zero_safe_and_nonzero_zero_q_rejected():
    zero = np.zeros((3, 5))
    scale = c.display_scale(zero)
    assert scale['q'] == 0 and scale['saturated_fraction'] == 0
    np.testing.assert_array_equal(c.signed_compress(zero, 0), zero)
    with pytest.raises(ValueError):
        c.signed_compress(np.ones((3, 5)), 0)


@pytest.mark.parametrize('ratio', [3, 10, 30])
def test_compression_independent_formula_odd_monotone_unsaturated(ratio):
    a = np.array([[-100., -10., -1., 0., 1., 10., 100.]])
    output = c.signed_compress(a, q=10., ratio=ratio)
    np.testing.assert_allclose(output, np.arcsinh(a / (10 / ratio)) / np.arcsinh(ratio), rtol=5.e-16, atol=1.e-16)
    np.testing.assert_array_equal(output, -output[:, ::-1])
    assert np.all(np.diff(output) > 0)
    assert output[0, 3] == 0
    np.testing.assert_allclose(output[0, [1, 5]], [-1, 1], atol=5.e-16)
    assert output[0, -1] > 1  # no hidden clipping


def test_compression_extreme_scale_stays_finite():
    output = c.signed_compress(np.array([[-1.e308, 0., 1.e308]]), q=1.e-300)
    assert np.isfinite(output).all()
    assert output[0, 0] == -output[0, 2]


def test_display_invalid_domains_and_compression_parameters():
    a = np.ones((3, 4))
    for percentile in (-1, 101, np.inf):
        with pytest.raises(ValueError):
            c.display_scale(a, percentile)
    for mask in (np.zeros_like(a, dtype=bool), np.ones((2, 4), dtype=bool), a):
        with pytest.raises(ValueError):
            c.display_scale(a, mask=mask)
    for q, ratio in ((-1, 10), (np.inf, 10), (1, 0), (1, -1)):
        with pytest.raises(ValueError):
            c.signed_compress(a, q, ratio)


@pytest.mark.parametrize('a', [np.ones(3), np.empty((0, 3)), np.array([[np.nan]]), np.array([[np.inf]]), np.array([[1j]])])
def test_all_operators_reject_invalid_images(a):
    operations = [lambda: c.derivatives(a, 20), lambda: c.bandpass(a, 20),
                  lambda: c.display_scale(a), lambda: c.signed_compress(a, 1),
                  lambda: c.depth_gain(a, [20])]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()
