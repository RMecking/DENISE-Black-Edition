from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def _log_log_fit(x_values: Sequence[float], y_values: Sequence[float]) -> dict[str, float]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Taylor fit needs matching sequences with at least two points")
    if any(value <= 0.0 or not math.isfinite(value) for value in (*x_values, *y_values)):
        raise ValueError("Taylor fit values must be positive and finite")
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
        (right - (intercept + slope * left)) ** 2 for left, right in zip(x, y)
    )
    total_sum = math.fsum((value - y_mean) ** 2 for value in y)
    r_squared = 1.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


def analyze_taylor_remainders(
    *,
    epsilons: Sequence[float],
    objectives: Sequence[float],
    baseline_objective: float,
    gradient_directional_product: float,
    fit_points: int = 4,
) -> dict[str, object]:
    if len(epsilons) != len(objectives) or len(epsilons) < fit_points:
        raise ValueError("Taylor inputs do not cover the fixed fit window")
    if fit_points < 4:
        raise ValueError("M5.2 requires at least the predefined four-point window")
    if any(value <= 0.0 or not math.isfinite(value) for value in epsilons):
        raise ValueError("Taylor epsilons must be positive and finite")
    if any(not math.isfinite(value) for value in objectives):
        raise ValueError("Taylor objectives must be finite")
    for coarse, fine in zip(epsilons, epsilons[1:]):
        if not math.isclose(coarse / fine, 2.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Taylor epsilon ladder must use consecutive halving")

    rows = []
    for epsilon, objective in zip(epsilons, objectives):
        delta = objective - baseline_objective
        prediction = epsilon * gradient_directional_product
        rows.append(
            {
                "epsilon": epsilon,
                "objective": objective,
                "delta_objective": delta,
                "epsilon_gradient_dot_direction": prediction,
                "r0": abs(delta),
                "r1": abs(delta - prediction),
            }
        )

    q0 = []
    q1 = []
    for coarse, fine in zip(rows, rows[1:]):
        if min(coarse["r0"], fine["r0"], coarse["r1"], fine["r1"]) <= 0.0:
            q0.append(None)
            q1.append(None)
        else:
            q0.append(math.log2(coarse["r0"] / fine["r0"]))
            q1.append(math.log2(coarse["r1"] / fine["r1"]))
    for index, row in enumerate(rows):
        row["q0_to_next"] = q0[index] if index < len(q0) else None
        row["q1_to_next"] = q1[index] if index < len(q1) else None

    window = rows[:fit_points]
    fit_r0 = _log_log_fit(
        [row["epsilon"] for row in window], [row["r0"] for row in window]
    )
    fit_r1 = _log_log_fit(
        [row["epsilon"] for row in window], [row["r1"] for row in window]
    )
    first_three_q0 = q0[:3]
    first_three_q1 = q1[:3]
    finite_orders = all(value is not None and math.isfinite(value) for value in (
        *first_three_q0,
        *first_three_q1,
    ))
    ratios = [row["r1"] / row["r0"] for row in window]
    nonzero_signed_rows = [row for row in window if row["delta_objective"] != 0.0]
    sign_consistent = bool(nonzero_signed_rows) and all(
        math.copysign(1.0, row["delta_objective"])
        == math.copysign(1.0, row["epsilon_gradient_dot_direction"])
        for row in nonzero_signed_rows
    )
    checks = {
        "slope_r0": 0.75 <= fit_r0["slope"] <= 1.25,
        "slope_r1": 1.6 <= fit_r1["slope"] <= 2.4,
        "pairwise_orders_finite": finite_orders,
        "median_q0": finite_orders
        and 0.75 <= statistics.median(first_three_q0) <= 1.25,
        "median_q1": finite_orders
        and 1.6 <= statistics.median(first_three_q1) <= 2.4,
        "two_q1_above_1_5": finite_orders
        and sum(value > 1.5 for value in first_three_q1) >= 2,
        "first_order_improves": all(row["r1"] < row["r0"] for row in window),
        "relative_remainder_decreases": all(
            fine < coarse for coarse, fine in zip(ratios, ratios[1:])
        ),
        "signed_prediction_consistent": sign_consistent,
    }
    return {
        "baseline_objective": baseline_objective,
        "gradient_directional_product": gradient_directional_product,
        "fit_points": fit_points,
        "rows": rows,
        "pairwise_q0": q0,
        "pairwise_q1": q1,
        "slope_r0": fit_r0["slope"],
        "slope_r1": fit_r1["slope"],
        "r_squared_r0": fit_r0["r_squared"],
        "r_squared_r1": fit_r1["r_squared"],
        "median_first_three_q0": statistics.median(first_three_q0)
        if finite_orders
        else None,
        "median_first_three_q1": statistics.median(first_three_q1)
        if finite_orders
        else None,
        "r1_over_r0_resolved_window": ratios,
        "checks": checks,
        "accepted": all(checks.values()),
    }
