"""Pure historical mapping/PSF and saved-array analysis, no numerical kernels."""
import copy

import numpy as np
import pytest

from tools.denise_case.campaign import encoded
from tools.denise_case.normal_operator import basis, coefficients, metrics, tangent, transpose
from tools.denise_case.normal_operator_analysis import aperture, compare, cosine, deep_cosine, reconstruct


def model():
    shape = (24, 50)
    vs = np.ones(shape); vs[0] = 0
    return {'vp': np.full(shape, 3.), 'vs': vs, 'rho': np.full(shape, 2.)}


def probe(models=None):
    return basis({'name': 'tiny', 'parameter': 'lnvp', 'x_m': 500, 'z_m': 300, 'tier': 1}, models or model(), 20)


def test_exact_cotangent_identity_coupling_and_water():
    m = model(); rng = np.random.default_rng(20261006)
    z = rng.normal(size=(2, 24, 50)); z[1, 0] = 0
    g = rng.normal(size=z.shape)
    native, exact = tangent(z, m, native_dtype=np.float64)
    assert np.array_equal(native, exact)
    assert np.sum(exact*g) == pytest.approx(np.sum(z*transpose(g, m)), rel=2e-14)
    assert np.all(transpose(g, m)[1, 0] == 0)
    assert np.all(transpose(np.stack([np.ones((24, 50)), np.zeros((24, 50))]), m)[1, 1:] < 0)
    z[1, 0, 1] = 1
    with pytest.raises(ValueError, match='frozen'):
        tangent(z, m)


@pytest.mark.parametrize('key,value', [('vs', -1), ('vp', 0), ('rho', 0), ('vp', np.nan)])
def test_no_floor_or_invalid_background(key, value):
    m = model(); m[key][5, 5] = value
    with pytest.raises(ValueError):
        coefficients(m)


def test_unit_basis_and_exact_grid_no_repair():
    z, native, exact, spec = probe()
    assert np.linalg.norm(z) == 1 and spec['row_col'] == [14, 24]
    assert native.dtype == np.float32 and exact.dtype == np.float64
    assert spec['normalization'] == 'discrete unit basis, Euclidean norm1'
    spec['x_m'] += 1
    with pytest.raises(ValueError, match='grid center'):
        basis(spec, model(), 20)


def test_predeclared_radius_tie_shell_zero_cross_talk_and_cuts():
    m = model(); z, _, _, spec = probe(m)
    response = 3*z
    response[1, *spec['row_col']] = 1
    ms = metrics(response, spec, m)
    same = ms['channels']['lnvp']
    assert same['r50_m'] == same['r90_m'] == 0
    assert same['regions']['target']['squared_norm_fraction'] == 1
    assert ms['cross_talk']['full_other_over_same_norm'] == 1/3
    assert ms['sampled_diagonal'] == 3
    assert same['secondary_half_height_lobe_count'] == 1
    assert ms['complete_grid_including_cpml'] and ms['labels_not_acceptance']
    # Two equal radius shells: threshold must select the whole radius, not cell count.
    response *= 0; response[0, 14, 23] = response[0, 14, 25] = 1
    ms = metrics(response, spec, m)
    assert ms['channels']['lnvp']['r90_m'] == 20
    zero = metrics(np.zeros_like(response), spec, m)
    assert zero['channels']['lnvp']['r90_m'] is None
    assert zero['cross_talk']['full_other_over_same_norm'] is None
    assert zero['labels'] == ['INDETERMINATE']


def test_non_top_connected_water_rejected():
    m = model(); m['vs'][3, 3] = 0
    with pytest.raises(ValueError, match='top-connected'):
        metrics(probe()[0], probe()[3], m)


def test_ascending_reduction_not_input_order_or_pairwise():
    items = [(i, np.full((2, 2, 2), {1: 1e16, 2: -1e16, 3: 1.}.get(i, 0.))) for i in range(1, 101)]
    sums = reconstruct(items[::-1])
    assert np.all(sums['full'] == 1)
    assert np.all(sums['block1'] == 1)
    assert np.all(sums['odd'] == 1e16)
    assert np.all(sums['even'] == -1e16)
    assert encoded(reconstruct_summary(sums)) == encoded(reconstruct_summary(reconstruct(items)))


def reconstruct_summary(sums):
    return {k: v.tolist() for k, v in sums.items()}


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'fp32', 'nan'])
def test_reduction_refuses_incomplete_or_modified_contract(bad):
    items = [(i, np.ones((2, 2, 2))) for i in range(1, 101)]
    if bad == 'missing': items.pop()
    if bad == 'duplicate': items[-1] = items[0]
    if bad == 'fp32': items[0] = (1, items[0][1].astype(np.float32))
    if bad == 'nan': items[0][1][0, 0, 0] = np.nan
    with pytest.raises(ValueError): reconstruct(items)


def test_aperture_frequency_semantics_and_display_independence():
    m = model(); z, _, _, spec = probe(m)
    fields = {g: z.copy() for g in ('full', 'odd', 'even', 'block1', 'block2', 'block3', 'block4', 'block5')}
    ms = metrics(z, spec, m)
    entry = {'groups': {g: {'metrics': ms} for g in fields}, 'sampled_diagonal': 1., 'summed_Jz_energy': 1.}
    before = encoded(entry)
    ap = aperture(fields, entry, m)
    assert ap['odd_even_full_cosine'] == 1
    assert all(b['cosine_full'] == 1 for b in ap['blocks'])
    other = copy.deepcopy(entry); other['sampled_diagonal'] = .1
    out = compare(entry, other)
    assert out['distinct_operators'] == ['H_15', 'H_5']
    assert out['H5_over_H15']['sampled_diagonal'] == .1
    assert 'dispersion-only' in out['interpretation_scope']
    other['groups']['full']['metrics']['probe']['name'] = 'other'
    with pytest.raises(ValueError, match='same probe'): compare(entry, other)
    assert encoded(entry) == before
    assert cosine(z, -z) == -1 and cosine(z, z*0) is None
    # Interface domination must not substitute for the frozen deep restriction.
    tall = {'vs': np.ones((174, 3))}; tall['vs'][:22] = 0
    a = np.zeros((2, 174, 3)); a[:, 22] = 1e6; a[:, 100] = 1
    b = a.copy(); b[:, 100] *= -1
    assert cosine(a, b) > .99
    # Stable scaled norms involve sqrt; the analytic -1 identity has FP64 roundoff.
    assert deep_cosine(a, b, tall, np.arange(1, 175)*20) == pytest.approx(-1, abs=2e-15)
