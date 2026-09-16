"""Independent contracts for the C7c-b1 single-step material bridge."""

from __future__ import annotations

from dataclasses import replace
import math

from tests.utilities.m63c_material_timestep_vjp_reference import (
    CASES,
    finite_difference_gradient,
)


BRIDGE_REFERENCE_RELATIVE_MAX = 3.0e-7


def split_cases():
    """Return deterministic L=1/2/3 and isolated-channel C7b inputs."""
    result = []
    for case in CASES:
        zero = (0.0,) * case.mechanisms
        result.extend(
            (
                case,
                replace(case, bar_sx=0.0, bar_sy=0.0, bar_rx=zero, bar_qy=zero),
                replace(case, bar_v=0.0, bar_sy=0.0, bar_qy=zero),
                replace(case, bar_v=0.0, bar_sx=0.0, bar_rx=zero),
            )
        )
    return tuple(result)


def independent_expected(case):
    """Five-point derivative of the local material-dependent objective."""
    return finite_difference_gradient(case)


def relative_l2(actual, expected):
    numerator = math.fsum((a - b) ** 2 for a, b in zip(actual, expected))
    denominator = math.fsum(b * b for b in expected)
    return math.sqrt(numerator / max(denominator, 1.0e-300))
