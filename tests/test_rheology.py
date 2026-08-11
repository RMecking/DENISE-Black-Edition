from __future__ import annotations

import cmath
import json
import math

import pytest

from tests.utilities.effective_parameters import (
    EffectiveDeniseParameters,
    parse_effective_parameters,
    require_effective_parameters,
)
from tests.utilities.viscoelastic_rheology import (
    approximate_main_lobe_width_hz,
    complex_shear_modulus,
    discrete_rheology_prediction,
    effective_q_from_transfer_slopes,
    linear_fit,
    rheology_prediction,
    synthetic_rheology_pair,
    transfer_spectrum,
    unwrap_phase,
)
from tests.utilities.seismogram import time_interval


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


CALIBRATION_FREQUENCIES_HZ = (6.0, 8.0, 10.0, 12.0, 14.0)
CALIBRATION_OFFSETS_M = (400.0, 500.0, 600.0, 700.0, 800.0)


def _broadband_calibration(*, half_width_s, window_kind, tukey_alpha=0.2):
    dt_s = 0.0005
    transfer = {frequency: [] for frequency in CALIBRATION_FREQUENCIES_HZ}
    for offset in CALIBRATION_OFFSETS_M:
        center = 0.15 + offset / 2000.0
        elastic, viscoelastic = synthetic_rheology_pair(
            dt_s=dt_s,
            time_s=0.75,
            pulse_center_s=center,
            source_frequency_hz=10.0,
            distance_m=offset,
            vs_m_s=2000.0,
            density_kg_m3=2000.0,
            qs_input=50.0,
            relaxation_frequencies_hz=(10.0,),
        )
        start, stop = center - half_width_s, center + half_width_s
        elastic_window = time_interval(elastic, start_s=start, stop_s=stop, dt_s=dt_s)
        viscoelastic_window = time_interval(
            viscoelastic, start_s=start, stop_s=stop, dt_s=dt_s
        )
        samples = transfer_spectrum(
            viscoelastic_window,
            elastic_window,
            dt_s=dt_s,
            frequencies_hz=CALIBRATION_FREQUENCIES_HZ,
            window_kind=window_kind,
            tukey_alpha=tukey_alpha,
        )
        assert tuple(sample.frequency_hz for sample in samples) == CALIBRATION_FREQUENCIES_HZ
        for sample in samples:
            transfer[sample.frequency_hz].append(sample.value)
    report = {}
    for frequency, values in transfer.items():
        attenuation_fit = linear_fit(
            CALIBRATION_OFFSETS_M, [math.log(abs(value)) for value in values]
        )
        phase_fit = linear_fit(
            CALIBRATION_OFFSETS_M,
            unwrap_phase([cmath.phase(value) for value in values]),
        )
        theory = rheology_prediction(
            frequency_hz=frequency,
            vs_m_s=2000.0,
            density_kg_m3=2000.0,
            qs_input=50.0,
            relaxation_frequencies_hz=(10.0,),
        )
        phase_absolute_error = abs(phase_fit.slope - theory.phase_slope_rad_per_m)
        report[str(frequency)] = {
            "imposed_attenuation_slope_per_m": theory.log_amplitude_slope_per_m,
            "recovered_attenuation_slope_per_m": attenuation_fit.slope,
            "attenuation_relative_error": abs(
                attenuation_fit.slope - theory.log_amplitude_slope_per_m
            )
            / abs(theory.log_amplitude_slope_per_m),
            "imposed_phase_slope_rad_per_m": theory.phase_slope_rad_per_m,
            "recovered_phase_slope_rad_per_m": phase_fit.slope,
            "phase_absolute_error_rad_per_m": phase_absolute_error,
            "phase_relative_error": phase_absolute_error
            / abs(theory.phase_slope_rad_per_m),
        }
    return report


def test_short_hann_broadband_calibration_exposes_phase_bias():
    report = _broadband_calibration(half_width_s=0.11, window_kind="hann")
    print("OLD_HANN_CALIBRATION=" + json.dumps(report, sort_keys=True))
    assert report["8.0"]["phase_relative_error"] > 0.20
    assert report["12.0"]["phase_relative_error"] > 0.20


def test_long_tukey_broadband_calibration_recovers_known_transfer():
    report = _broadband_calibration(
        half_width_s=0.20, window_kind="tukey", tukey_alpha=0.2
    )
    print("LONG_TUKEY_CALIBRATION=" + json.dumps(report, sort_keys=True))
    assert approximate_main_lobe_width_hz(
        duration_s=0.40, kind="tukey", tukey_alpha=0.2
    ) < 6.0
    for row in report.values():
        assert row["attenuation_relative_error"] <= 0.05
    for row in report.values():
        assert row["phase_relative_error"] <= 0.05


def test_discrete_fd_rheology_is_negligibly_different_from_continuous():
    rows = {}
    for frequency in CALIBRATION_FREQUENCIES_HZ:
        continuous = rheology_prediction(
            frequency_hz=frequency,
            vs_m_s=2000.0,
            density_kg_m3=2000.0,
            qs_input=50.0,
            relaxation_frequencies_hz=(10.0,),
        )
        discrete = discrete_rheology_prediction(
            frequency_hz=frequency,
            vs_m_s=2000.0,
            density_kg_m3=2000.0,
            qs_input=50.0,
            relaxation_frequencies_hz=(10.0,),
            dt_s=0.0005,
            dh_m=10.0,
        )
        rows[str(frequency)] = {
            "continuous_attenuation": continuous.log_amplitude_slope_per_m,
            "discrete_attenuation": discrete.log_amplitude_slope_per_m,
            "continuous_phase": continuous.phase_slope_rad_per_m,
            "discrete_phase": discrete.phase_slope_rad_per_m,
        }
        assert abs(
            discrete.log_amplitude_slope_per_m - continuous.log_amplitude_slope_per_m
        ) <= 1.0e-6
        assert abs(
            discrete.phase_slope_rad_per_m - continuous.phase_slope_rad_per_m
        ) <= 1.0e-6
    print("DISCRETE_RHEOLOGY=" + json.dumps(rows, sort_keys=True))
