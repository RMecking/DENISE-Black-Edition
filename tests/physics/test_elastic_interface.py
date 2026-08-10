from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case as generate_homogeneous
from tests.cases.layered_psv import LayeredPSVConfig, generate_case as generate_layered
from tests.utilities.elastic_analytics import two_segment_ray, zoeppritz_p_coefficients
from tests.utilities.physics_run import run_psv_case
from tests.utilities.staggered_grid import (
    collocate_velocity_at_sxy,
    input_coordinate_for_field_position,
    input_field_position,
    sxy_collocation_stencil,
)
from tests.utilities.seismogram import (
    absolute_peak_index_in_interval,
    normalized_correlation,
    project_components,
    relative_amplitude_error,
    relative_l2,
    signal_energy,
    time_interval,
)


pytestmark = pytest.mark.integration
PEAK_HALF_WIDTH_S = 0.05
AMPLITUDE_RELATIVE_TOLERANCE = 0.15
PHASE_CORRELATION_MIN = 0.98
IDENTICAL_CONTROL_RATIO_MAX = 1.0e-3


def _run_layer(directory, *, repository_root, denise_binary, mpiexec, config, nprocx=1, nprocy=1):
    return run_psv_case(
        directory, repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=config, generator=generate_layered,
        nprocx=nprocx, nprocy=nprocy,
    )


def _run_homogeneous(directory, *, repository_root, denise_binary, mpiexec, config):
    return run_psv_case(
        directory, repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=config, generator=generate_homogeneous,
    )


def _window(trace, center, dt):
    return time_interval(
        trace, start_s=center - PEAK_HALF_WIDTH_S,
        stop_s=center + PEAK_HALF_WIDTH_S, dt_s=dt,
    )


def _pick(trace, center, dt):
    return (
        absolute_peak_index_in_interval(
            trace, start_s=center - PEAK_HALF_WIDTH_S,
            stop_s=center + PEAK_HALF_WIDTH_S, dt_s=dt,
        ) + 1
    ) * dt


def _normal_p_config():
    return LayeredPSVConfig(receivers_m=((1200.0, 700.0), (1200.0, 1600.0)))


def test_normal_p_interface_and_identical_medium_control(
    tmp_path, repository_root, denise_binary, mpiexec
):
    contrast = _normal_p_config()
    identical = replace(
        contrast, vp2_m_s=contrast.vp1_m_s, vs2_m_s=contrast.vs1_m_s,
        rho2_kg_m3=contrast.rho1_kg_m3,
    )
    source_physical = input_field_position(
        (contrast.source_x_m, contrast.source_y_m), contrast.dh_m, "sxx"
    )
    reflected_receiver_physical = input_field_position(
        contrast.receivers_m[0], contrast.dh_m, "vy"
    )
    physical_reflection_path = (
        contrast.interface_y_m - source_physical[1]
        + contrast.interface_y_m - reflected_receiver_physical[1]
    )
    calibration_source_input_y = 500.0
    calibration_source_physical_y = input_field_position(
        (1200.0, calibration_source_input_y), contrast.dh_m, "sxx"
    )[1]
    calibration_receiver_physical_y = calibration_source_physical_y + physical_reflection_path
    calibration_receiver_input_y = input_coordinate_for_field_position(
        calibration_receiver_physical_y, contrast.dh_m, axis="y", field="vy"
    )
    calibration = HomogeneousPSVConfig(
        nx=240, ny=240, time_s=contrast.time_s, dt_s=contrast.dt_s,
        vp_m_s=contrast.vp1_m_s, vs_m_s=contrast.vs1_m_s,
        density_kg_m3=contrast.rho1_kg_m3,
        source_x_m=1200.0, source_y_m=calibration_source_input_y,
        receivers_m=((1200.0, calibration_receiver_input_y),),
    )
    identity_reference = replace(
        calibration, source_y_m=contrast.source_y_m, receivers_m=contrast.receivers_m,
    )
    _, contrast_vy = _run_layer(
        tmp_path / "contrast", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=contrast,
    )
    _, identical_vy = _run_layer(
        tmp_path / "identical", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=identical,
    )
    _, calibration_vy = _run_homogeneous(
        tmp_path / "calibration", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=calibration,
    )
    _, identity_reference_vy = _run_homogeneous(
        tmp_path / "identity_reference", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=identity_reference,
    )
    nominal_reflection_path = (
        contrast.interface_y_m - contrast.source_y_m
        + contrast.interface_y_m - contrast.receivers_m[0][1]
    )
    reflection_path = physical_reflection_path
    calibration_source_physical = input_field_position(
        (calibration.source_x_m, calibration.source_y_m), calibration.dh_m, "sxx"
    )
    calibration_receiver_physical = input_field_position(
        calibration.receivers_m[0], calibration.dh_m, "vy"
    )
    calibration_path = math.dist(calibration_source_physical, calibration_receiver_physical)
    source_peak = 1.5 / contrast.source_frequency_hz
    reflection_peak = source_peak + reflection_path / contrast.vp1_m_s
    reflection = [a - b for a, b in zip(contrast_vy[0], identical_vy[0])]
    reflection_pick = _pick(reflection, reflection_peak, contrast.dt_s)
    calibration_pick = _pick(calibration_vy[0], reflection_peak, contrast.dt_s)
    reflection_window = _window(reflection, reflection_pick, contrast.dt_s)
    calibration_window = _window(calibration_vy[0], calibration_pick, contrast.dt_s)
    coefficients = zoeppritz_p_coefficients(
        0.0,
        vp1_m_s=contrast.vp1_m_s, vs1_m_s=contrast.vs1_m_s,
        rho1_kg_m3=contrast.rho1_kg_m3,
        vp2_m_s=contrast.vp2_m_s, vs2_m_s=contrast.vs2_m_s,
        rho2_kg_m3=contrast.rho2_kg_m3,
    )
    # Reflected P polarization points toward y_min, hence fixed vy is -Rpp.
    expected_window = [
        -coefficients["reflected_p_displacement"] * value
        for value in calibration_window
    ]
    amplitude_error = relative_amplitude_error(reflection_window, expected_window)
    correlation = normalized_correlation([reflection_window], [expected_window])
    identical_residual = [
        a - b for a, b in zip(identical_vy[0], identity_reference_vy[0])
    ]
    identical_window = _window(identical_residual, reflection_peak, contrast.dt_s)
    identical_ratio = math.sqrt(
        signal_energy(identical_window) / signal_energy(reflection_window)
    )

    transmission_receiver_physical = input_field_position(
        contrast.receivers_m[1], contrast.dh_m, "vy"
    )
    transmission_time = (
        (contrast.interface_y_m - source_physical[1]) / contrast.vp1_m_s
        + (transmission_receiver_physical[1] - contrast.interface_y_m) / contrast.vp2_m_s
    )
    transmission_peak = source_peak + transmission_time
    transmission_pick = _pick(contrast_vy[1], transmission_peak, contrast.dt_s)
    timing_tolerance = 2.0 * contrast.dt_s + 0.005 * reflection_path / contrast.vp1_m_s
    metrics = {
        "interface_y_m": contrast.interface_y_m,
        "source_input_m": [contrast.source_x_m, contrast.source_y_m],
        "source_physical_sxx_syy_m": source_physical,
        "reflection_receiver_input_m": contrast.receivers_m[0],
        "reflection_receiver_physical_vy_m": reflected_receiver_physical,
        "old_nominal_reflection_path_m": nominal_reflection_path,
        "reflection_path_m": reflection_path,
        "calibration_source_physical_sxx_syy_m": calibration_source_physical,
        "calibration_receiver_input_m": calibration.receivers_m[0],
        "calibration_receiver_physical_vy_m": calibration_receiver_physical,
        "calibration_path_m": calibration_path,
        "expected_reflection_peak_s": reflection_peak,
        "observed_reflection_peak_s": reflection_pick,
        "calibration_peak_s": calibration_pick,
        "expected_transmission_peak_s": transmission_peak,
        "observed_transmission_peak_s": transmission_pick,
        "timing_tolerance_s": timing_tolerance,
        "zoeppritz": coefficients,
        "expected_fixed_vy_reflection_coefficient": -coefficients["reflected_p_displacement"],
        "reflection_amplitude_error": amplitude_error,
        "reflection_phase_correlation": correlation,
        "identical_medium_to_contrast_reflection_l2_ratio": identical_ratio,
        "transmission_amplitude_is_diagnostic": True,
    }
    (tmp_path / "normal_p_interface_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert abs(reflection_pick - calibration_pick) <= timing_tolerance
    assert math.isclose(calibration_path, reflection_path, abs_tol=1.0e-12)
    assert amplitude_error <= AMPLITUDE_RELATIVE_TOLERANCE
    assert correlation >= PHASE_CORRELATION_MIN
    assert identical_ratio <= IDENTICAL_CONTROL_RATIO_MAX
    assert abs(
        (reflection_pick - transmission_pick) - (reflection_path / contrast.vp1_m_s - transmission_time)
    ) <= timing_tolerance


def test_normal_sv_interface_reflection(tmp_path, repository_root, denise_binary, mpiexec):
    contrast = LayeredPSVConfig(
        time_s=1.0, source_type=2, receivers_m=((1200.0, 700.0),)
    )
    identical = replace(
        contrast, vp2_m_s=contrast.vp1_m_s, vs2_m_s=contrast.vs1_m_s,
        rho2_kg_m3=contrast.rho1_kg_m3,
    )
    source_physical = input_field_position(
        (contrast.source_x_m, contrast.source_y_m), contrast.dh_m, "vx"
    )
    receiver_physical = input_field_position(contrast.receivers_m[0], contrast.dh_m, "vx")
    physical_path = (
        contrast.interface_y_m - source_physical[1]
        + contrast.interface_y_m - receiver_physical[1]
    )
    calibration_source_input_y = 500.0
    calibration_source_physical_y = input_field_position(
        (1200.0, calibration_source_input_y), contrast.dh_m, "vx"
    )[1]
    calibration_receiver_physical_y = calibration_source_physical_y + physical_path
    calibration_receiver_input_y = input_coordinate_for_field_position(
        calibration_receiver_physical_y, contrast.dh_m, axis="y", field="vx"
    )
    calibration = HomogeneousPSVConfig(
        nx=240, ny=240, time_s=contrast.time_s, dt_s=contrast.dt_s,
        vp_m_s=contrast.vp1_m_s, vs_m_s=contrast.vs1_m_s,
        density_kg_m3=contrast.rho1_kg_m3,
        source_x_m=1200.0, source_y_m=calibration_source_input_y, source_type=2,
        receivers_m=((1200.0, calibration_receiver_input_y),),
    )
    contrast_vx, _ = _run_layer(
        tmp_path / "contrast", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=contrast,
    )
    identical_vx, _ = _run_layer(
        tmp_path / "identical", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=identical,
    )
    calibration_vx, _ = _run_homogeneous(
        tmp_path / "calibration", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=calibration,
    )
    old_nominal_path = (
        2.0 * contrast.interface_y_m - contrast.source_y_m - contrast.receivers_m[0][1]
    )
    path = physical_path
    calibration_source_physical = input_field_position(
        (calibration.source_x_m, calibration.source_y_m), calibration.dh_m, "vx"
    )
    calibration_receiver_physical = input_field_position(
        calibration.receivers_m[0], calibration.dh_m, "vx"
    )
    calibration_path = math.dist(calibration_source_physical, calibration_receiver_physical)
    peak = 1.5 / contrast.source_frequency_hz + path / contrast.vs1_m_s
    reflection = [a - b for a, b in zip(contrast_vx[0], identical_vx[0])]
    reflection_pick = _pick(reflection, peak, contrast.dt_s)
    calibration_pick = _pick(calibration_vx[0], peak, contrast.dt_s)
    reflected_window = _window(reflection, reflection_pick, contrast.dt_s)
    calibration_window = _window(calibration_vx[0], calibration_pick, contrast.dt_s)
    z1 = contrast.rho1_kg_m3 * contrast.vs1_m_s
    z2 = contrast.rho2_kg_m3 * contrast.vs2_m_s
    coefficient = (z1 - z2) / (z1 + z2)
    expected = [coefficient * value for value in calibration_window]
    amplitude_error = relative_amplitude_error(reflected_window, expected)
    correlation = normalized_correlation([reflected_window], [expected])
    tolerance = 2.0 * contrast.dt_s + 0.005 * path / contrast.vs1_m_s
    metrics = {
        "interface_y_m": contrast.interface_y_m,
        "source_input_m": [contrast.source_x_m, contrast.source_y_m],
        "source_physical_vx_m": source_physical,
        "receiver_input_m": contrast.receivers_m[0],
        "receiver_physical_vx_m": receiver_physical,
        "old_nominal_reflection_path_m": old_nominal_path,
        "actual_staggered_reflection_path_m": path,
        "old_nominal_calibration_path_m": 1200.0,
        "calibration_receiver_input_m": calibration.receivers_m[0],
        "calibration_receiver_physical_vx_m": calibration_receiver_physical,
        "corrected_calibration_path_m": calibration_path,
        "shear_impedance_reflection_coefficient_fixed_vx": coefficient,
        "observed_reflection_peak_s": reflection_pick,
        "calibration_peak_s": calibration_pick,
        "observed_peak_difference_s": abs(reflection_pick - calibration_pick),
        "timing_tolerance_s": tolerance,
        "amplitude_error": amplitude_error,
        "phase_correlation": correlation,
    }
    (tmp_path / "normal_sv_interface_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert abs(reflection_pick - calibration_pick) <= tolerance
    assert math.isclose(calibration_path, path, abs_tol=1.0e-12)
    assert amplitude_error <= AMPLITUDE_RELATIVE_TOLERANCE
    assert correlation >= PHASE_CORRELATION_MIN


def test_oblique_p_interface_mode_conversion(tmp_path, repository_root, denise_binary, mpiexec):
    central_receiver = (1400.0, 700.0)
    contrast = LayeredPSVConfig(
        time_s=0.9, source_x_m=900.0, source_y_m=500.0,
        receivers_m=sxy_collocation_stencil(central_receiver, 10.0),
    )
    identical = replace(
        contrast, vp2_m_s=contrast.vp1_m_s, vs2_m_s=contrast.vs1_m_s,
        rho2_kg_m3=contrast.rho1_kg_m3,
    )
    contrast_vx, contrast_vy = _run_layer(
        tmp_path / "contrast", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=contrast,
    )
    identical_vx, identical_vy = _run_layer(
        tmp_path / "identical", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=identical,
    )
    contrast_collocated_x, contrast_collocated_y = collocate_velocity_at_sxy(
        contrast_vx[0], contrast_vx[1], contrast_vy[0], contrast_vy[2]
    )
    identical_collocated_x, identical_collocated_y = collocate_velocity_at_sxy(
        identical_vx[0], identical_vx[1], identical_vy[0], identical_vy[2]
    )
    residual_x = [a - b for a, b in zip(contrast_collocated_x, identical_collocated_x)]
    residual_y = [a - b for a, b in zip(contrast_collocated_y, identical_collocated_y)]
    source_input = (contrast.source_x_m, contrast.source_y_m)
    source = input_field_position(source_input, contrast.dh_m, "sxx")
    receiver = input_field_position(central_receiver, contrast.dh_m, "sxy")
    p_ray = two_segment_ray(
        source, receiver, boundary_y_m=contrast.interface_y_m,
        incident_velocity_m_s=contrast.vp1_m_s, outgoing_velocity_m_s=contrast.vp1_m_s,
    )
    sv_ray = two_segment_ray(
        source, receiver, boundary_y_m=contrast.interface_y_m,
        incident_velocity_m_s=contrast.vp1_m_s, outgoing_velocity_m_s=contrast.vs1_m_s,
    )
    source_peak = 1.5 / contrast.source_frequency_hz
    p_peak = source_peak + p_ray.travel_time_s
    sv_peak = source_peak + sv_ray.travel_time_s
    p_direction = (receiver[0] - p_ray.boundary_x_m, receiver[1] - contrast.interface_y_m)
    sv_direction = (receiver[0] - sv_ray.boundary_x_m, receiver[1] - contrast.interface_y_m)
    p_long, p_trans = project_components(residual_x, residual_y, p_direction)
    sv_long, sv_trans = project_components(residual_x, residual_y, sv_direction)
    p_pick = _pick(p_long, p_peak, contrast.dt_s)
    sv_pick = _pick(sv_trans, sv_peak, contrast.dt_s)
    p_ratio = signal_energy(_window(p_long, p_peak, contrast.dt_s)) / signal_energy(
        _window(p_trans, p_peak, contrast.dt_s)
    )
    sv_ratio = signal_energy(_window(sv_trans, sv_peak, contrast.dt_s)) / signal_energy(
        _window(sv_long, sv_peak, contrast.dt_s)
    )
    incidence_angle = math.atan2(
        abs(p_ray.boundary_x_m - source[0]), contrast.interface_y_m - source[1]
    )
    coefficients = zoeppritz_p_coefficients(
        incidence_angle,
        vp1_m_s=contrast.vp1_m_s, vs1_m_s=contrast.vs1_m_s,
        rho1_kg_m3=contrast.rho1_kg_m3,
        vp2_m_s=contrast.vp2_m_s, vs2_m_s=contrast.vs2_m_s,
        rho2_kg_m3=contrast.rho2_kg_m3,
    )
    tolerance = 2.0 * contrast.dt_s + 0.005 * sv_ray.travel_time_s
    metrics = {
        "interface_y_m": contrast.interface_y_m,
        "source_input_m": source_input,
        "source_physical_sxx_syy_m": source,
        "receiver_stencil_input_m": contrast.receivers_m,
        "collocated_receiver_physical_sxy_m": receiver,
        "p_ray": p_ray.__dict__, "sv_ray": sv_ray.__dict__,
        "expected_p_peak_s": p_peak, "observed_p_peak_s": p_pick,
        "expected_sv_peak_s": sv_peak, "observed_sv_peak_s": sv_pick,
        "expected_sv_minus_p_s": sv_ray.travel_time_s - p_ray.travel_time_s,
        "observed_sv_minus_p_s": sv_pick - p_pick,
        "sv_minus_p_error_s": abs(
            (sv_pick - p_pick) - (sv_ray.travel_time_s - p_ray.travel_time_s)
        ),
        "timing_tolerance_s": tolerance,
        "p_longitudinal_to_transverse_energy_ratio": p_ratio,
        "sv_transverse_to_longitudinal_energy_ratio": sv_ratio,
        "zoeppritz_amplitudes_diagnostic": coefficients,
        "raw_amplitude_assertion": False,
    }
    (tmp_path / "oblique_interface_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert p_peak < sv_peak
    assert abs((sv_pick - p_pick) - (sv_ray.travel_time_s - p_ray.travel_time_s)) <= tolerance
    assert p_ratio >= 10.0
    assert sv_ratio >= 5.0


def test_layered_interface_mpi_reproducibility(
    tmp_path, repository_root, denise_binary, mpiexec
):
    config = _normal_p_config()
    reference = _run_layer(
        tmp_path / "mpi_1x1", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config,
    )
    metrics = {}
    for label, nprocx, nprocy in (("2x1", 2, 1), ("1x2", 1, 2), ("2x2", 2, 2)):
        variant = _run_layer(
            tmp_path / f"mpi_{label}", repository_root=repository_root,
            denise_binary=denise_binary, mpiexec=mpiexec, config=config,
            nprocx=nprocx, nprocy=nprocy,
        )
        metrics[label] = {}
        for component, ref, candidate in zip(("vx", "vy"), reference, variant):
            rel = relative_l2(ref, candidate)
            corr = normalized_correlation(ref, candidate)
            metrics[label][component] = {"relative_l2": rel, "normalized_correlation": corr}
            assert rel <= 1.0e-5
            assert corr >= 0.999999
    (tmp_path / "layered_mpi_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
