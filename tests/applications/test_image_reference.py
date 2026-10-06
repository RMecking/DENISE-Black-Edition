"""Independent small geometric fixtures; no solver or image-conditioned truth."""
import copy
import json

import numpy as np
import pytest

from tools.denise_case.image_reference import define_reference, evaluate_reference


def fixture():
    shape = (31, 41)
    true = dict(vp=np.full(shape, 2.), vs=np.ones(shape), rho=np.full(shape, 3.))
    smooth = {k: a.copy() for k, a in true.items()}
    # Independently drawn rectangular contrast. Centered stencil marks both
    # rows either side of its top edge, not one arbitrarily selected row.
    true['vp'][10:, 18:23] = 4.
    window = dict(name='fixed', x_min_m=100, x_max_m=4100,
                  z_min_m=100, z_max_m=3100, include_z_max=True)
    return true, smooth, [window]


def test_reference_fixed_independent_edge_and_contrast():
    true, smooth, windows = fixture()
    spec, arrays = define_reference(true, smooth, 100, windows=windows)
    assert arrays['edge_dz'][9, 20] == .01
    assert arrays['edge_dz'][10, 20] == .01
    assert arrays['edge_dx'][15, 17] == .01
    assert arrays['contrast_vp'][15, 20] == 2.
    assert arrays['contrast_vs'][15, 20] == 0.
    assert not arrays['evaluation_domain'][0].any()
    assert spec['windows'][0]['sample_count'] == 29 * 39
    assert spec['primary_edge'].startswith('hypot')
    assert all(not a.flags.writeable for a in arrays.values())
    json.dumps(spec, allow_nan=False)
    second, other = define_reference(true, smooth, 100, windows=windows)
    assert second == spec
    for key in arrays:
        np.testing.assert_array_equal(other[key], arrays[key])


def test_ridge_location_displaced_controls_and_no_polarity_requirement():
    true, smooth, windows = fixture()
    spec, arrays = define_reference(true, smooth, 100, windows=windows)
    image = np.zeros((31, 41))
    image[9:11, 18:23] = 7.
    result = evaluate_reference(image, spec, arrays)['windows']['fixed']
    assert result['observed']['ridge_nearest_edge_mean_m'] == 0.
    assert result['observed']['ridge_within_40m_fraction'] == 1.
    # At the two rectangle corners |grad| is sqrt(2)*.01 in row 10,
    # while the image tie selects row 9: (2*100+3*0)/5 = 40 metres.
    assert result['observed']['ridge_same_column_depth_error_mean_m'] == 40.
    assert result['displaced_reference_controls']['shift_+1000m']['ridge_nearest_edge_mean_m'] > 500
    assert result['displaced_reference_controls']['shift_-1000m']['ridge_nearest_edge_mean_m'] > 500
    inverse = evaluate_reference(-image, spec, arrays)['windows']['fixed']
    assert inverse == result
    scaled = evaluate_reference(image * 3, spec, arrays)['windows']['fixed']['observed']
    for key in ('ridge_nearest_edge_mean_m', 'ridge_within_40m_fraction', 'axial_gradient_cosine_mean'):
        assert scaled[key] == result['observed'][key]


def test_shifted_ridge_loses_localization():
    true, smooth, windows = fixture()
    spec, arrays = define_reference(true, smooth, 100, windows=windows)
    image = np.zeros((31, 41))
    image[6, 20] = 1
    value = evaluate_reference(image, spec, arrays)['windows']['fixed']['observed']
    assert value['ridge_nearest_edge_mean_m'] == 300
    assert value['ridge_same_column_depth_error_mean_m'] == 300
    assert value['ridge_within_40m_fraction'] == 0


def test_zero_image_and_zero_reference_are_undefined_not_epsilon_repaired():
    true, smooth, windows = fixture()
    spec, arrays = define_reference(smooth, smooth, 100, windows=windows)
    value = evaluate_reference(np.zeros((31, 41)), spec, arrays)
    observed = value['windows']['fixed']['observed']
    assert observed['target_edge_count'] == 0
    assert observed['image_ridge_count'] == 0
    assert observed['near_off_mean_abs_ratio'] is None
    assert observed['axial_gradient_cosine_mean'] is None
    assert observed['ridge_nearest_edge_mean_m'] is None
    json.dumps(value, allow_nan=False)


def test_frozen_reference_definition_and_array_identity_reject_tampering():
    true, smooth, windows = fixture()
    spec, arrays = define_reference(true, smooth, 100, windows=windows)
    changed = copy.deepcopy(spec)
    changed['edge_percentile'] = 95
    with pytest.raises(ValueError, match='definition identity'):
        evaluate_reference(np.ones((31, 41)), changed, arrays)
    changed_arrays = {k: a.copy() for k, a in arrays.items()}
    changed_arrays['target_edge'][4, 5] = ~changed_arrays['target_edge'][4, 5]
    with pytest.raises(ValueError, match='array identity'):
        evaluate_reference(np.ones((31, 41)), spec, changed_arrays)


def test_invalid_geometry_and_models():
    true, smooth, windows = fixture()
    with pytest.raises(ValueError, match='DH'):
        define_reference(true, smooth, 0, windows=windows)
    true['rho'][1, 1] = np.inf
    with pytest.raises(ValueError, match='finite'):
        define_reference(true, smooth, 100, windows=windows)


def test_invalid_derivative_cells_explicitly_excluded():
    true, smooth, windows = fixture()
    spec, arrays = define_reference(true, smooth, 100, windows=windows)
    image = np.ones((31, 41))
    image[4, 5] = np.nan
    observed = evaluate_reference(image, spec, arrays)['windows']['fixed']['observed']
    assert observed['nonfinite_excluded'] == 1
    assert observed['sample_count'] == 29 * 39 - 1


def test_axial_orientation_parallel_and_perpendicular_analytic_gradients():
    shape = (31, 41)
    row = np.arange(shape[0], dtype=float)[:, None]
    col = np.arange(shape[1], dtype=float)[None, :]
    true = dict(vp=np.broadcast_to(2. + row, shape).copy(),
                vs=np.ones(shape), rho=np.full(shape, 3.))
    window = dict(name='fixed', x_min_m=100, x_max_m=4100,
                  z_min_m=100, z_max_m=3100, include_z_max=True)
    spec, arrays = define_reference(true, true, 100, windows=[window])
    parallel = np.broadcast_to(row, shape)
    orthogonal = np.broadcast_to(col, shape)
    a = evaluate_reference(parallel, spec, arrays)['windows']['fixed']['observed']
    b = evaluate_reference(orthogonal, spec, arrays)['windows']['fixed']['observed']
    assert a['axial_gradient_cosine_mean'] == 1.
    assert b['axial_gradient_cosine_mean'] == 0.
    assert a['orientation_sample_count'] == b['orientation_sample_count'] == 29 * 39


def test_default_windows_are_physical_predeclared():
    true = dict(vp=np.full((174, 500), 2000.), vs=np.full((174, 500), 1000.),
                rho=np.full((174, 500), 2000.))
    for a in (true['vs'],):
        a[:22] = 0
    spec, arrays = define_reference(true, true)
    assert len(spec['windows']) == 10
    assert spec['windows'][0]['sample_count'] == 25 * 480
    assert spec['windows'][4]['sample_count'] == 15 * 480
    assert spec['windows'][5]['sample_count'] == 25 * 400
    assert not arrays['evaluation_domain'][:49].any()
