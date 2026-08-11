from __future__ import annotations

import cmath
import math

import pytest

from tests.utilities.effective_parameters import (
    EffectiveDeniseParameters,
    parse_effective_parameters,
    require_effective_parameters,
)
from tests.utilities.viscoelastic_rheology import (
    complex_shear_modulus,
    effective_q_from_transfer_slopes,
    linear_fit,
    rheology_prediction,
    transfer_spectrum,
    unwrap_phase,
)


EFFECTIVE_OUTPUT = """
 MODE=0: Only forward modeling is applied.
 PHYSICS=5: Solve 2D isotropic elastic SH problem.
 Number of relaxation mechanisms (L): 3
 The L relaxation frequencies are at:
     5.000000    10.000000    20.000000 Hz
"""


def test_parse_and_require_effective_denise_parameters():
    effective = parse_effective_parameters(EFFECTIVE_OUTPUT)
    assert effective == EffectiveDeniseParameters(0, 5, 3, (5.0, 10.0, 20.0))
    require_effective_parameters(
        effective,
        mode=0,
        physics=5,
        relaxation_frequencies_hz=(5.0, 10.0, 20.0),
    )


@pytest.mark.parametrize(
    "expected, message",
    (
        ({"mode": 1, "physics": 5, "relaxation_frequencies_hz": (5.0, 10.0, 20.0)}, "MODE"),
        ({"mode": 0, "physics": 1, "relaxation_frequencies_hz": (5.0, 10.0, 20.0)}, "PHYSICS"),
        ({"mode": 0, "physics": 5, "relaxation_frequencies_hz": (10.0,)}, "effective L"),
    ),
)
def test_effective_parameter_mismatch_is_a_normal_assertion(expected, message):
    with pytest.raises(AssertionError, match=message):
        require_effective_parameters(parse_effective_parameters(EFFECTIVE_OUTPUT), **expected)


def test_effective_parameter_parser_rejects_incomplete_echo():
    with pytest.raises(ValueError, match="FL"):
        parse_effective_parameters(
            "MODE=0: forward\nPHYSICS=5: SH\nNumber of relaxation mechanisms (L): 1\n"
        )


def test_l1_reference_frequency_effective_q_matches_implemented_mapping():
    modulus = complex_shear_modulus(
        frequency_hz=10.0,
        reference_shear_modulus_pa=8.0e9,
        qs_input=20.0,
        relaxation_frequencies_hz=(10.0,),
    )
    assert math.isclose(modulus.real, 8.0e9, rel_tol=1.0e-12)
    assert math.isclose(modulus.real / modulus.imag, 21.0, rel_tol=1.0e-12)


def test_rheology_predicts_attenuation_and_dispersive_phase():
    prediction = rheology_prediction(
        frequency_hz=10.0,
        vs_m_s=2000.0,
        density_kg_m3=2000.0,
        qs_input=50.0,
        relaxation_frequencies_hz=(10.0,),
    )
    assert prediction.log_amplitude_slope_per_m < 0.0
    assert prediction.phase_slope_rad_per_m > 0.0
    assert prediction.wavenumber_per_m.imag < 0.0
    observed_q = effective_q_from_transfer_slopes(
        frequency_hz=10.0,
        vs_m_s=2000.0,
        density_kg_m3=2000.0,
        log_amplitude_slope_per_m=prediction.log_amplitude_slope_per_m,
        phase_slope_rad_per_m=prediction.phase_slope_rad_per_m,
    )
    assert math.isclose(observed_q, prediction.effective_q, rel_tol=1.0e-12)


def test_transfer_spectrum_recovers_known_complex_ratio():
    dt_s = 0.001
    frequency_hz = 10.0
    ratio = 0.7 * cmath.exp(0.3j)
    elastic = [math.cos(2.0 * math.pi * frequency_hz * index * dt_s) for index in range(1001)]
    viscoelastic = [
        0.7 * math.cos(2.0 * math.pi * frequency_hz * index * dt_s + 0.3)
        for index in range(1001)
    ]
    sample = transfer_spectrum(
        viscoelastic,
        elastic,
        dt_s=dt_s,
        frequencies_hz=(frequency_hz,),
    )[0]
    assert math.isclose(abs(sample.value), abs(ratio), rel_tol=1.0e-6)
    assert math.isclose(cmath.phase(sample.value), cmath.phase(ratio), abs_tol=1.0e-6)


def test_unwrap_and_linear_fit_report_slope_residuals_and_quality():
    phases = unwrap_phase((2.8, -3.0, -2.5))
    assert phases[0] < phases[1] < phases[2]
    fit = linear_fit((1.0, 2.0, 3.0), (4.0, 6.0, 8.0))
    assert math.isclose(fit.slope, 2.0)
    assert fit.maximum_absolute_residual < 1.0e-12
    assert math.isclose(fit.r_squared, 1.0)
