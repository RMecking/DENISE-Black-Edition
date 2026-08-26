from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


M61E_EPSILONS = (1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4)
M61E_FIT_POINTS = 4


def _optional_log_log_fit(
    x_values: Sequence[float], y_values: Sequence[float]
) -> dict[str, float] | None:
    if any(value <= 0.0 or not math.isfinite(value) for value in y_values):
        return None
    x = [math.log(value) for value in x_values]
    y = [math.log(value) for value in y_values]
    x_mean = math.fsum(x) / len(x)
    y_mean = math.fsum(y) / len(y)
    denominator = math.fsum((value - x_mean) ** 2 for value in x)
    slope = math.fsum(
        (left - x_mean) * (right - y_mean) for left, right in zip(x, y)
    ) / denominator
    intercept = y_mean - slope * x_mean
    residual_sum = math.fsum(
        (right - (intercept + slope * left)) ** 2
        for left, right in zip(x, y)
    )
    total_sum = math.fsum((value - y_mean) ** 2 for value in y)
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    return {"slope": slope, "r_squared": r_squared}


def _pairwise_orders(values: Sequence[float]) -> list[float | None]:
    orders = []
    for coarse, fine in zip(values, values[1:]):
        if min(coarse, fine) <= 0.0 or not all(
            math.isfinite(value) for value in (coarse, fine)
        ):
            orders.append(None)
        else:
            orders.append(math.log2(coarse / fine))
    return orders


def _finite_median(values: Sequence[float | None]) -> float | None:
    if not all(value is not None and math.isfinite(value) for value in values):
        return None
    return statistics.median(values)


def analyze_taylor_derivative_consistency(
    *,
    epsilons: Sequence[float],
    objectives: Sequence[float],
    baseline_objective: float,
    gradient_directional_product: float,
    repeatability_difference: float,
) -> dict[str, object]:
    """Evaluate the fixed M6.1e derivative-consistency contract.

    R0, coarse-window sign agreement, and eta monotonicity remain reported as
    asymptotic-quality diagnostics. They do not gate derivative acceptance:
    finite-epsilon cancellation can make R0 vanish or reverse delta-J, and eta
    need not decrease strictly at every coarse step for a correct derivative.
    """
    if tuple(epsilons) != M61E_EPSILONS:
        raise ValueError("M6.1e requires the fixed five-point epsilon ladder")
    if len(objectives) != len(epsilons):
        raise ValueError("Taylor objectives must cover the fixed epsilon ladder")
    if not math.isfinite(baseline_objective) or any(
        not math.isfinite(value) for value in objectives
    ):
        raise ValueError("Taylor objectives must be finite")
    if not math.isfinite(gradient_directional_product):
        raise ValueError("Taylor directional product must be finite")
    if not math.isfinite(repeatability_difference) or repeatability_difference < 0.0:
        raise ValueError("Repeatability difference must be finite and nonnegative")

    rows = []
    for epsilon, objective in zip(epsilons, objectives):
        delta = objective - baseline_objective
        prediction = epsilon * gradient_directional_product
        r0 = abs(delta)
        r1 = abs(delta - prediction)
        denominator = abs(prediction)
        rows.append(
            {
                "epsilon": epsilon,
                "objective": objective,
                "delta_objective": delta,
                "epsilon_gradient_dot_direction": prediction,
                "r0": r0,
                "r1": r1,
                "eta": r1 / denominator if denominator != 0.0 else None,
                "r1_over_r0": r1 / r0 if r0 != 0.0 else None,
                "r1_less_than_r0": r1 < r0,
            }
        )

    q0 = _pairwise_orders([row["r0"] for row in rows])
    q1 = _pairwise_orders([row["r1"] for row in rows])
    for index, row in enumerate(rows):
        row["q0_to_next"] = q0[index] if index < len(q0) else None
        row["q1_to_next"] = q1[index] if index < len(q1) else None

    window = rows[:M61E_FIT_POINTS]
    fit_r0 = _optional_log_log_fit(
        [row["epsilon"] for row in window], [row["r0"] for row in window]
    )
    fit_r1 = _optional_log_log_fit(
        [row["epsilon"] for row in window], [row["r1"] for row in window]
    )
    first_three_q0 = q0[:3]
    first_three_q1 = q1[:3]
    median_q0 = _finite_median(first_three_q0)
    median_q1 = _finite_median(first_three_q1)
    q1_finite = median_q1 is not None
    nonzero_delta_rows = [row for row in window if row["delta_objective"] != 0.0]
    signed_consistency = all(
        row["epsilon_gradient_dot_direction"] != 0.0
        and math.copysign(1.0, row["delta_objective"])
        == math.copysign(1.0, row["epsilon_gradient_dot_direction"])
        for row in nonzero_delta_rows
    )
    eta = [row["eta"] for row in window]
    eta_decreases = all(
        value is not None and math.isfinite(value) for value in eta
    ) and all(fine < coarse for coarse, fine in zip(eta, eta[1:]))
    resolved_direction = (
        gradient_directional_product != 0.0
        if repeatability_difference == 0.0
        else abs(M61E_EPSILONS[0] * gradient_directional_product)
        >= 100.0 * repeatability_difference
    )
    slope_r1 = fit_r1["slope"] if fit_r1 is not None else None

    acceptance = {
        "resolved_nondegenerate_direction": resolved_direction,
        "r1_fit_order": slope_r1 is not None and 1.6 <= slope_r1 <= 2.4,
        "first_three_q1_finite": q1_finite,
        "median_first_three_q1_order": q1_finite
        and 1.6 <= median_q1 <= 2.4,
        "two_resolved_quadratic_steps": q1_finite
        and sum(value > 1.5 for value in first_three_q1) >= 2,
    }
    diagnostics = {
        "pairwise_q0": q0,
        "slope_r0": fit_r0["slope"] if fit_r0 is not None else None,
        "r_squared_r0": fit_r0["r_squared"] if fit_r0 is not None else None,
        "median_first_three_q0": median_q0,
        "r1_over_r0_resolved_window": [row["r1_over_r0"] for row in window],
        "r1_less_than_r0_resolved_window": [
            row["r1_less_than_r0"] for row in window
        ],
        "signed_prediction_consistent": signed_consistency,
        "eta_resolved_window": eta,
        "eta_strictly_decreasing": eta_decreases,
        "acceptance_effect": "none",
    }
    return {
        "baseline_objective": baseline_objective,
        "gradient_directional_product": gradient_directional_product,
        "repeatability_difference": repeatability_difference,
        "fit_points": M61E_FIT_POINTS,
        "rows": rows,
        "pairwise_q1": q1,
        "slope_r1": slope_r1,
        "r_squared_r1": fit_r1["r_squared"] if fit_r1 is not None else None,
        "median_first_three_q1": median_q1,
        "acceptance": acceptance,
        "diagnostics": diagnostics,
        "accepted": all(acceptance.values()),
    }
