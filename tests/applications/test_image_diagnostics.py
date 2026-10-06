"""Independent analytic solver-free fixtures for A2.5A diagnostics."""
import json
import math

import numpy as np
import pytest

from tools.denise_case import image_diagnostics as d


def test_stats_analytic_signs_quantiles_and_zeros():
    s = d.stats([-3.,-0.,0.,4.,np.nan])
    assert s['count'] == 5 and s['finite_count'] == 4 and s['nonfinite_count'] == 1
    assert s['squared_norm'] == 25 and s['norm'] == 5 and s['rms'] == 2.5
    assert s['l1'] == 7 and s['abs_p50'] == 1.5
    assert s['positive'] == s['negative'] == s['negative_zero'] == 1
    assert s['zero'] == 2 and s['zero_fraction'] == .5
    assert s['abs_p90'] == pytest.approx(3.7)
    assert s['abs_p99'] == pytest.approx(3.97)
    json.dumps(s,allow_nan=False)


def test_underflow_representation_preserves_mathematical_scale_and_ratios():
    a = np.array([3e-200,4e-200])
    s = d.stats(a)
    assert s['norm'] == pytest.approx(5e-200,rel=1e-14,abs=0)
    assert s['squared_norm'] is None  # not the misleading physical zero
    assert s['squared_norm_representation']['log10'] == pytest.approx(math.log10(25)-400)
    assert d._ratio_square(d._square(a[:1]),d._square(a)) == pytest.approx(9/25)
    assert d.stats([0.,0.])['squared_norm'] == 0
    assert d._ratio_square(d._square([0]),d._square([0])) is None


def test_empty_and_overflow_statistics_are_json_safe():
    assert d.stats([])['rms'] is None
    s = d.stats([1e308,-1e308])
    assert s['squared_norm'] is None and s['l1'] is None
    json.dumps(s,allow_nan=False)


def test_generic_sloping_interface_and_exact_zero_classification():
    vs = np.array([[0,0,0],[1e-30,0,0],[2,3,0],[2,3,4.]])
    g = d.infer_water(vs,20)
    np.testing.assert_array_equal(g['last_water'],[0,1,2])
    np.testing.assert_array_equal(g['first_solid'],[1,2,3])
    np.testing.assert_array_equal(g['bottom_m'],[30,50,70])
    assert np.count_nonzero(g['water']) == 6


@pytest.mark.parametrize('vs',[
    [[0],[1],[0]], [[1],[0],[1]], [[0],[0],[0]], [[0],[-1],[1]], [[0],[np.nan],[1]],
])
def test_ambiguous_water_rejected(vs):
    with pytest.raises(ValueError):
        d.infer_water(vs,20)


def test_marmousi_fixed_coordinate_region_counts_without_hardcoded_inference():
    vs = np.ones((174,500)); vs[:22]=0
    ds = d.domains(vs,20)
    expected = {'water':22*500,'first_solid':500,'guard550':5*500,
                'guard650':10*500,'guard750':15*500,'shallow':17*500,
                '1000_1500':25*500,'1500_2000':25*500,'2000_2500':25*500,
                '2500_3000':25*500,'3000_3280':15*500,'bottom':10*500,
                'deep':115*500}
    for k,n in expected.items():
        assert np.count_nonzero(ds['vertical'][k]) == n
    assert np.count_nonzero(ds['lateral']['interior']) == 480*174
    assert np.count_nonzero(ds['lateral']['acquisition']) == 400*174
    assert np.all(d.infer_water(vs,20)['bottom_m'] == 450)


def test_depth_energy_analytic_fixture_and_no_epsilon():
    a = np.array([[0.,0.],[3.,4.],[0.,0.]])
    vs = np.array([[0.,0.],[1.,1.],[1.,1.]])
    before = a.copy()
    r = d.depth_metrics(a,vs)
    assert r['rows'][1]['squared_norm'] == 25
    assert r['row_squared_norm_fraction'] == [0,1,0]
    assert r['cumulative_squared_norm_fraction'] == [0,1,1]
    assert r['ratios']['full']['first_solid_over_solid_squared_norm'] == 1
    assert r['ratios']['full']['deep_over_shallow_rms'] is None
    assert r['regions']['acquisition']['full']['count'] == 0
    z = d.depth_metrics(np.zeros_like(a),vs)
    assert z['cumulative_squared_norm_fraction'] is None
    np.testing.assert_array_equal(a,before)
    json.dumps(r,allow_nan=False)


def test_extreme_dynamic_range_db_does_not_overflow():
    vs = np.ones((174,2)); vs[:22]=0
    a = np.zeros_like(vs); a[22]=1e300; a[60]=1e-300
    r = d.depth_metrics(a,vs)
    assert r['ratios']['full']['dynamic_range_deficit_db'] == 12000
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('scale',[1.,1e-200,1e200])
def test_correlation_scale_independent_analytic_answers(scale):
    a = np.array([1.,2.,3.])*scale
    assert d.correlation(a,2*a) == pytest.approx(1)
    assert d.correlation(a,-a,centered=True) == pytest.approx(-1)
    assert d.correlation(a,np.array([2.,0.,2.])*scale,centered=True) == pytest.approx(0)
    assert d.correlation(a,np.ones(3)*scale,centered=True) is None
    assert d.correlation(np.zeros(3),a) is None
    assert d.correlation(a,a,np.zeros(3,bool)) is None


def shots_fixture():
    return {i: np.array([[0.,0.],[float(i),-float(i)],[1.,-1.]]) for i in range(100,0,-1)}


def test_subset_membership_ascending_fp64_and_input_preservation():
    shots = shots_fixture()
    before = {i:a.copy() for i,a in shots.items()}
    sums = d.subset_sums(shots)
    assert list(sums) == ['odd','even','block01','block02','block03','block04','block05']
    assert sums['odd'][1,0] == 2500
    assert sums['even'][1,0] == 2550
    assert sums['block01'][1,0] == 210
    assert sums['block05'][1,0] == 1810
    assert all(a.dtype == np.float64 for a in sums.values())
    for i in shots:
        np.testing.assert_array_equal(shots[i],before[i])
    # Order-sensitive cancellation distinguishes ascending from insertion order.
    for a in shots.values():
        a[:]=0
    shots[1][:]=1e16; shots[3][:]=-1e16; shots[5][:]=1
    np.testing.assert_array_equal(d.subset_sums(shots)['odd'],np.ones((3,2)))


def test_subset_ids_fail_closed():
    s = shots_fixture(); del s[68]
    with pytest.raises(ValueError):
        d.subset_sums(s)
    s = shots_fixture(); s[68]=np.ones((4,2))
    with pytest.raises(ValueError):
        d.subset_sums(s)


def test_shot_coherent_fixture_concentration_semblance_and_68_retained():
    s = {i:np.array([[0.,0.],[1.,-1.],[2.,-2.]]) for i in range(1,101)}
    vs = np.array([[0.,0.],[1.,1.],[1.,1.]])
    r = d.shot_diagnostics(s,vs)
    reg = r['regions']['full']['solid']
    assert len(r['per_shot']) == 100 and '68' in r['per_shot']
    assert reg['odd_even'] == {'centered':1.,'uncentered':1.}
    assert len(reg['block_pairs']) == 10
    assert reg['contributions']['shot_semblance'] == pytest.approx(1)
    assert reg['contributions']['effective_contributor_count'] == pytest.approx(100)
    assert reg['shot68_observation']['not_rejected']
    assert reg['shot68_observation']['squared_norm_over_aggregate_squared_norm'] == pytest.approx(.0001)
    assert len(r['leave_one_out']['full']) == 100
    json.dumps(r,allow_nan=False)


def test_canceling_shots_show_negative_coherence_and_zero_semblance():
    a = np.array([[0.,0.],[1.,-1.],[2.,-2.]])
    s = {i:a*(1 if i%2 else -1) for i in range(1,101)}
    vs = np.array([[0.,0.],[1.,1.],[1.,1.]])
    reg = d.shot_diagnostics(s,vs)['regions']['full']['solid']
    assert reg['odd_even']['centered'] == pytest.approx(-1)
    assert reg['contributions']['shot_semblance'] == 0
    assert reg['shot68_observation']['squared_norm_over_aggregate_squared_norm'] is None


def test_shift_control_deterministic_undefined_null_and_effect_sizes():
    a = np.array([[1.,2.,0.,-1.],[4.,1.,-3.,2.]])
    mask = np.ones_like(a,bool)
    x = d.lateral_shift_control(a,a,mask,realizations=19)
    assert x == d.lateral_shift_control(a,a,mask,realizations=19)
    assert x['observed_centered'] == pytest.approx(1)
    assert len(x['control_centered']) == 19
    z = d.lateral_shift_control(a*0,a*0,mask,realizations=3)
    assert z['observed_centered'] is z['empirical_absolute_exceedance'] is None
    assert z['control_centered'] == [None]*3
    json.dumps(x,allow_nan=False)
