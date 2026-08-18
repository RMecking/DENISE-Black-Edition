from __future__ import annotations

import math
from array import array

from tests.cases.psv_fwi_taylor import (
    case_hashes,
    gradient_contributions,
    model_at_epsilon,
    taylor_cases,
)
from tests.physics.test_psv_fwi_gradient_audit import _objective
from tests.utilities.taylor import analyze_taylor_remainders


EPSILONS = (1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4)


def test_multiplicative_perturbation_uses_delta_m_equal_m0_times_p() -> None:
    case = taylor_cases()[0]
    epsilon = EPSILONS[0]
    model = model_at_epsilon(case, epsilon)
    expected = array(
        "f",
        (
            value * (1.0 + epsilon * weight)
            for value, weight in zip(case.background["vp"], case.direction["vp"])
        ),
    ).tolist()
    assert model["vp"] == expected
    assert case.delta_model["vp"] == tuple(
        value * weight
        for value, weight in zip(case.background["vp"], case.direction["vp"])
    )


def test_joint_product_sums_positive_psv_gradient_contributions_without_sign_flip() -> None:
    gradients = {
        "vp": (1.0, 2.0),
        "vs": (3.0, 4.0),
        "rho": (5.0, 6.0),
    }
    delta = {
        "vp": (0.5, 0.25),
        "vs": (0.2, 0.1),
        "rho": (0.05, 0.025),
    }
    result = gradient_contributions(gradients, delta)
    assert result["vp"] == 1.0
    assert math.isclose(result["vs"], 1.0)
    assert math.isclose(result["rho"], 0.4)
    assert math.isclose(result["total"], 2.4)
    assert result["total"] > 0.0  # PSV uses the stored positive objective gradient.


def test_independent_gf1_gf2_objectives_include_both_components() -> None:
    config = taylor_cases()[0].config
    count = config.receiver_count * config.samples_per_trace
    observed = {"x": [0.0] * count, "y": [0.0] * count}
    synthetic = {"x": [0.0] * count, "y": [0.0] * count}
    synthetic["x"][1] = 2.0
    synthetic["y"][1] = 3.0

    gf2_x = _objective(
        synthetic, observed, config=config, grad_form=2, data_components=3
    )
    gf2_y = _objective(
        synthetic, observed, config=config, grad_form=2, data_components=2
    )
    gf2_both = _objective(
        synthetic, observed, config=config, grad_form=2, data_components=1
    )
    assert gf2_x == 2.0
    assert gf2_y == 4.5
    assert gf2_both == 6.5 == gf2_x + gf2_y

    expected_gf1 = (
        0.5
        * (config.samples_per_trace - 1)
        * config.dt_s**2
        * (2.0**2 + 3.0**2)
    )
    gf1_both = _objective(
        synthetic, observed, config=config, grad_form=1, data_components=1
    )
    assert math.isclose(gf1_both, expected_gf1, rel_tol=1.0e-14)
    assert gf1_both > 0.0


def test_shared_analyzer_recovers_linear_and_quadratic_remainders() -> None:
    baseline = 2.0
    derivative = -0.65
    objectives = [
        baseline + derivative * epsilon + 1.75 * epsilon**2 - 0.2 * epsilon**3
        for epsilon in EPSILONS
    ]
    result = analyze_taylor_remainders(
        epsilons=EPSILONS,
        objectives=objectives,
        baseline_objective=baseline,
        gradient_directional_product=derivative,
    )
    assert 0.98 <= result["slope_r0"] <= 1.02
    assert 1.98 <= result["slope_r1"] <= 2.02
    assert result["accepted"] is True


def test_heterogeneous_holdout_is_positive_nonconstant_and_independent() -> None:
    case = taylor_cases()[-1]
    assert case.holdout is True
    hashes = case_hashes(case)
    for component in ("vp", "vs", "rho"):
        assert min(case.background[component]) > 0.0
        assert min(case.target[component]) > 0.0
        assert max(case.background[component]) > min(case.background[component])
        assert max(case.direction[component]) > min(case.direction[component])
        assert hashes["target"][component] != hashes["direction"][component]
    assert len(set(hashes["direction"].values())) == 3
    assert len(set(hashes["target"].values())) == 3
    assert all(
        value > 0.0
        for values in model_at_epsilon(case, EPSILONS[0]).values()
        for value in values
    )
