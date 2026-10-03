"""Physical reference and pre-implementation classification study.

Failures of the physical refinement gate are scientific HOLD evidence, not
an xfail and not permission to retune the physical coefficients/tolerance.
"""
import numpy as np
import pytest

from tests.utilities.fluid_interface_reference import (
    FLUID_CP, FLUID_RHO, SOLID_CP, SOLID_RHO, REFINEMENT_H,
    interface_measurement, slip_coefficients, require_interface_refinement,
)


def test_normal_reference_is_independent_impedance_solution():
    zf, zs = FLUID_RHO*FLUID_CP, SOLID_RHO*SOLID_CP
    values, matrix, rhs = slip_coefficients(0)
    np.testing.assert_allclose(values, [(zs-zf)/(zs+zf), 2/(zs+zf), 0], atol=1e-15)
    np.testing.assert_allclose(matrix@values, rhs, atol=1e-15)


def test_oblique_reference_has_slip_and_energy_conservation():
    theta = np.deg2rad(20)
    alpha = np.arcsin(SOLID_CP/FLUID_CP*np.sin(theta))
    beta = np.arcsin(1.5/FLUID_CP*np.sin(theta))
    values, matrix, rhs = slip_coefficients(20)
    np.testing.assert_allclose(matrix@values, rhs, atol=1e-15)
    r, p, s = values
    assert min(abs(r), abs(p), abs(s)) > 0.01
    fluid_vx = (1+r)*np.sin(theta)/(FLUID_RHO*FLUID_CP)
    solid_vx = p*np.sin(alpha)+s*np.cos(beta)
    assert abs(fluid_vx-solid_vx) > 0.01  # DO NOT impose welded continuity.
    flux = r*r + SOLID_RHO*(SOLID_CP*np.cos(alpha)*p*p+1.5*np.cos(beta)*s*s) / (
        np.cos(theta)/(FLUID_RHO*FLUID_CP))
    assert abs(flux-1) < 5e-14


@pytest.mark.parametrize("angle", [0, 20])
def test_unmodified_zero_corner_fd4_refines_to_physical_interface(angle):
    results = [interface_measurement(h, angle) for h in REFINEMENT_H]
    for row in results:
        print("FLUID_INTERFACE", row)
    require_interface_refinement(results)


def test_narrow_pulse_fourier_window_is_not_interface_error():
    first = interface_measurement(0.25, 20)
    extended = interface_measurement(0.25, 20, window=180, half_domain=360)
    scale = abs(first["expected"])
    residual = max(abs(first["measured"]-extended["measured"])/scale)
    print("FLUID_INTERFACE_WINDOW_RESIDUAL", residual)
    # Independently declared 0.0005: comfortably below 0.002 refinement floor.
    assert residual < 0.0005


def test_physical_gate_does_not_trust_adapter_claimed_error():
    expected, _, _ = slip_coefficients(20)
    rows = [dict(h=h, angle=20, peak=1, measured=expected.copy(), max_error=0) for h in REFINEMENT_H]
    require_interface_refinement(rows)
    rows[-1]["measured"] *= 1.1
    with pytest.raises(AssertionError):
        require_interface_refinement(rows)
