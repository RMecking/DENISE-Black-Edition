from __future__ import annotations

import pytest

from tests.utilities.taylor import analyze_taylor_remainders
from tests.utilities.taylor_derivative import (
    M61E_EPSILONS,
    analyze_taylor_derivative_consistency,
)


M52_RHO_GF1 = {
    "baseline": 7.92419149038913e-08,
    "gradient_product": -5.599540504001491e-06,
    "objectives": (
        3.8678596097356057e-08,
        5.512557598503896e-08,
        6.621580996742151e-08,
        7.248477842124544e-08,
        7.58030600817194e-08,
    ),
}
M61E_FS1_RHO_GF1 = {
    "baseline": 6.591655927815609e-08,
    "gradient_product": -4.725686718825262e-06,
    "objectives": (
        4.780640534243274e-08,
        4.961736174712694e-08,
        5.5938858445378715e-08,
        6.046735269197403e-08,
        6.307528309549969e-08,
    ),
}
M61E_FS0_RHO_GF1 = {
    "baseline": 4.135960652170961e-08,
    "gradient_product": -2.286177494300244e-06,
    "objectives": (
        3.7833730883863376e-08,
        3.527082080271422e-08,
        3.72234377864909e-08,
        3.90170898944099e-08,
        4.011831900109887e-08,
    ),
}


def _historical(fixture):
    return analyze_taylor_remainders(
        epsilons=M61E_EPSILONS,
        objectives=fixture["objectives"],
        baseline_objective=fixture["baseline"],
        gradient_directional_product=fixture["gradient_product"],
    )


def _derivative_v2(fixture, *, gradient_product=None, repeatability=0.0):
    return analyze_taylor_derivative_consistency(
        epsilons=M61E_EPSILONS,
        objectives=fixture["objectives"],
        baseline_objective=fixture["baseline"],
        gradient_directional_product=(
            fixture["gradient_product"]
            if gradient_product is None
            else gradient_product
        ),
        repeatability_difference=repeatability,
    )


def _analytic_fixture(polynomial, *, gradient_product=1.0):
    return {
        "baseline": 0.0,
        "gradient_product": gradient_product,
        "objectives": tuple(polynomial(epsilon) for epsilon in M61E_EPSILONS),
    }


def _assert_v2_contract(result):
    assert result["accepted"]
    assert all(result["acceptance"].values())
    assert set(result["acceptance"]) == {
        "resolved_nondegenerate_direction",
        "r1_fit_order",
        "first_three_q1_finite",
        "median_first_three_q1_order",
        "two_resolved_quadratic_steps",
    }
    assert result["diagnostics"]["acceptance_effect"] == "none"


def test_historical_m52_homogeneous_rho_gf1_passes_derivative_v2():
    assert _historical(M52_RHO_GF1)["accepted"]
    _assert_v2_contract(_derivative_v2(M52_RHO_GF1))


def test_free_surface_rho_gf1_cancellation_fixture_passes_derivative_v2():
    assert not _historical(M61E_FS1_RHO_GF1)["accepted"]
    result = _derivative_v2(M61E_FS1_RHO_GF1)
    _assert_v2_contract(result)
    assert result["slope_r1"] == pytest.approx(1.9973144757, abs=5.0e-11)
    assert result["median_first_three_q1"] == pytest.approx(
        1.9966913950, abs=5.0e-11
    )


def test_absorbing_top_shallow_rho_gf1_cancellation_fixture_passes_v2():
    assert not _historical(M61E_FS0_RHO_GF1)["accepted"]
    _assert_v2_contract(_derivative_v2(M61E_FS0_RHO_GF1))


def test_coarse_sign_reversal_and_zero_r0_are_diagnostic_only():
    fixture = _analytic_fixture(lambda epsilon: epsilon - 200.0 * epsilon**2)
    result = _derivative_v2(fixture)
    _assert_v2_contract(result)
    assert result["rows"][0]["delta_objective"] < 0.0
    assert result["rows"][0]["epsilon_gradient_dot_direction"] > 0.0
    assert result["rows"][1]["r0"] == 0.0
    diagnostics = result["diagnostics"]
    assert not diagnostics["signed_prediction_consistent"]
    assert diagnostics["slope_r0"] is None
    assert diagnostics["median_first_three_q0"] is None
    assert diagnostics["r1_over_r0_resolved_window"][1] is None


def test_nonstrict_eta_is_diagnostic_only_for_correct_derivative():
    fixture = _analytic_fixture(
        lambda epsilon: (
            epsilon
            + epsilon**2
            + 15.0 * epsilon**3
            - 7000.0 * epsilon**4
        )
    )
    result = _derivative_v2(fixture)
    _assert_v2_contract(result)
    assert result["diagnostics"]["eta_resolved_window"] == pytest.approx(
        (0.0045, 0.0045, 0.002484375, 0.001259765625),
        abs=5.0e-16,
    )
    assert not result["diagnostics"]["eta_strictly_decreasing"]
    assert result["pairwise_q1"][:3] == pytest.approx(
        (1.0, 1.857042046, 1.979727605), abs=5.0e-10
    )
    assert result["slope_r1"] == pytest.approx(1.636735100, abs=5.0e-10)


@pytest.mark.parametrize("wrong_gradient", (-1.0, 0.5))
def test_wrong_gradient_introduces_first_order_r1_and_fails(wrong_gradient):
    fixture = _analytic_fixture(lambda epsilon: epsilon + epsilon**2)
    result = _derivative_v2(fixture, gradient_product=wrong_gradient)
    assert result["acceptance"]["resolved_nondegenerate_direction"]
    assert result["acceptance"]["first_three_q1_finite"]
    assert not result["acceptance"]["r1_fit_order"]
    assert not result["accepted"]


def test_repeatability_guard_is_an_independent_acceptance_condition():
    fixture = _analytic_fixture(lambda epsilon: epsilon + epsilon**2)
    repeatability = abs(M61E_EPSILONS[0]) / 99.0
    result = _derivative_v2(fixture, repeatability=repeatability)
    assert not result["acceptance"]["resolved_nondegenerate_direction"]
    assert all(
        passed
        for name, passed in result["acceptance"].items()
        if name != "resolved_nondegenerate_direction"
    )
    assert not result["accepted"]


def test_zero_direction_guard_is_an_independent_acceptance_condition():
    fixture = _analytic_fixture(lambda epsilon: epsilon**2)
    result = _derivative_v2(fixture, gradient_product=0.0)
    assert not result["acceptance"]["resolved_nondegenerate_direction"]
    assert all(
        passed
        for name, passed in result["acceptance"].items()
        if name != "resolved_nondegenerate_direction"
    )
    assert result["diagnostics"]["eta_resolved_window"] == [None] * 4
    assert not result["accepted"]
