from __future__ import annotations

from tests.utilities.taylor import analyze_taylor_remainders


EPSILONS = (1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4)


def _objectives(*, baseline: float, linear: float, quadratic: float, cubic: float):
    return [
        baseline + linear * epsilon + quadratic * epsilon**2 + cubic * epsilon**3
        for epsilon in EPSILONS
    ]


def test_taylor_analyzer_recovers_first_and_second_order():
    result = analyze_taylor_remainders(
        epsilons=EPSILONS,
        objectives=_objectives(
            baseline=3.0, linear=-0.7, quadratic=2.5, cubic=-0.4
        ),
        baseline_objective=3.0,
        gradient_directional_product=-0.7,
    )
    assert 0.98 <= result["slope_r0"] <= 1.02
    assert 1.98 <= result["slope_r1"] <= 2.02
    assert result["r_squared_r0"] >= 0.9999
    assert result["r_squared_r1"] >= 0.9999
    assert result["accepted"] is True


def test_taylor_analyzer_rejects_wrong_gradient():
    result = analyze_taylor_remainders(
        epsilons=EPSILONS,
        objectives=_objectives(
            baseline=3.0, linear=-0.7, quadratic=2.5, cubic=-0.4
        ),
        baseline_objective=3.0,
        gradient_directional_product=-0.45,
    )
    assert result["accepted"] is False
    assert result["checks"]["slope_r1"] is False
