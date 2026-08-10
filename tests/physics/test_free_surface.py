from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case
from tests.utilities.elastic_analytics import free_surface_p_coefficients, two_segment_ray
from tests.utilities.physics_run import run_psv_case
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
SURFACE_Y_M = 5.0
PEAK_HALF_WIDTH_S = 0.05
AMPLITUDE_RELATIVE_TOLERANCE = 0.15
PHASE_CORRELATION_MIN = 0.98
NEGATIVE_CONTROL_RATIO_MAX = 0.10


def _run(directory, *, repository_root, denise_binary, mpiexec, config, nprocx=1, nprocy=1):
    return run_psv_case(
        directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        generator=generate_case,
        nprocx=nprocx,
        nprocy=nprocy,
    )


def _normal_configs():
    free = HomogeneousPSVConfig(
        nx=240, ny=240, time_s=0.9, source_x_m=1200.0, source_y_m=700.0,
        receivers_m=((1200.0, 1100.0),), free_surface=True,
    )
    control = replace(free, free_surface=False)
    calibration = replace(
        free, ny=300, free_surface=False, source_y_m=400.0,
        receivers_m=((1200.0, 2190.0),),
    )
    return free, control, calibration


def _window(trace, center, dt):
    return time_interval(
        trace, start_s=center - PEAK_HALF_WIDTH_S,
        stop_s=center + PEAK_HALF_WIDTH_S, dt_s=dt,
    )


def test_psv_free_surface_normal_incidence_and_negative_control(
    tmp_path, repository_root, denise_binary, mpiexec
):
    free, control, calibration = _normal_configs()
    _, free_vy = _run(
        tmp_path / "free", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=free,
    )
    _, control_vy = _run(
        tmp_path / "absorbing_control", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=control,
    )
    _, calibration_vy = _run(
        tmp_path / "image_path_calibration", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=calibration,
    )
    direct_distance = 400.0
    reflection_distance = (free.source_y_m - SURFACE_Y_M) + (
        free.receivers_m[0][1] - SURFACE_Y_M
    )
    source_peak = 1.5 / free.source_frequency_hz
    direct_peak = source_peak + direct_distance / free.vp_m_s
    reflected_peak = source_peak + reflection_distance / free.vp_m_s
    timing_tolerance = 2.0 * free.dt_s + 0.005 * reflection_distance / free.vp_m_s

    direct_pick = (
        absolute_peak_index_in_interval(
            free_vy[0], start_s=direct_peak - PEAK_HALF_WIDTH_S,
            stop_s=direct_peak + PEAK_HALF_WIDTH_S, dt_s=free.dt_s,
        ) + 1
    ) * free.dt_s
    reflection = [left - right for left, right in zip(free_vy[0], control_vy[0])]
    reflection_pick = (
        absolute_peak_index_in_interval(
            reflection, start_s=reflected_peak - PEAK_HALF_WIDTH_S,
            stop_s=reflected_peak + PEAK_HALF_WIDTH_S, dt_s=free.dt_s,
        ) + 1
    ) * free.dt_s
    calibration_pick = (
        absolute_peak_index_in_interval(
            calibration_vy[0], start_s=reflected_peak - PEAK_HALF_WIDTH_S,
            stop_s=reflected_peak + PEAK_HALF_WIDTH_S, dt_s=free.dt_s,
        ) + 1
    ) * free.dt_s
    reflected_window = _window(reflection, reflection_pick, free.dt_s)
    control_window = _window(control_vy[0], reflected_peak, free.dt_s)
    calibration_window = _window(calibration_vy[0], calibration_pick, free.dt_s)
    expected = [-value for value in calibration_window]
    amplitude_error = relative_amplitude_error(reflected_window, expected)
    phase_correlation = normalized_correlation([reflected_window], [expected])
    negative_ratio = math.sqrt(signal_energy(control_window) / signal_energy(reflected_window))
    plane_wave = free_surface_p_coefficients(
        0.0, vp_m_s=free.vp_m_s, vs_m_s=free.vs_m_s,
        density_kg_m3=free.density_kg_m3,
    )
    metrics = {
        "surface_y_m": SURFACE_Y_M,
        "direct_distance_m": direct_distance,
        "reflection_distance_m": reflection_distance,
        "expected_direct_peak_s": direct_peak,
        "observed_direct_peak_s": direct_pick,
        "expected_reflected_peak_s": reflected_peak,
        "observed_reflected_peak_s": reflection_pick,
        "observed_image_path_calibration_peak_s": calibration_pick,
        "expected_reflection_minus_direct_s": (
            reflection_distance - direct_distance
        ) / free.vp_m_s,
        "observed_reflection_minus_direct_s": reflection_pick - direct_pick,
        "timing_tolerance_s": timing_tolerance,
        "expected_vy_polarity_relative_to_downward_calibration": -1,
        "plane_wave_coefficients": plane_wave,
        "reflection_calibration_amplitude_error": amplitude_error,
        "reflection_calibration_correlation": phase_correlation,
        "absorbing_control_to_reflection_l2_ratio": negative_ratio,
        "amplitude_tolerance": AMPLITUDE_RELATIVE_TOLERANCE,
        "phase_correlation_minimum": PHASE_CORRELATION_MIN,
        "negative_control_ratio_maximum": NEGATIVE_CONTROL_RATIO_MAX,
    }
    (tmp_path / "free_surface_normal_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert abs(
        (reflection_pick - direct_pick)
        - (reflection_distance - direct_distance) / free.vp_m_s
    ) <= timing_tolerance
    assert abs(reflection_pick - calibration_pick) <= timing_tolerance
    assert amplitude_error <= AMPLITUDE_RELATIVE_TOLERANCE
    assert phase_correlation >= PHASE_CORRELATION_MIN
    assert negative_ratio <= NEGATIVE_CONTROL_RATIO_MAX


def test_psv_free_surface_oblique_modes(tmp_path, repository_root, denise_binary, mpiexec):
    free = HomogeneousPSVConfig(
        nx=240, ny=240, time_s=1.05, source_x_m=900.0, source_y_m=700.0,
        receivers_m=((1400.0, 900.0),), free_surface=True,
    )
    control = replace(free, free_surface=False)
    free_vx, free_vy = _run(
        tmp_path / "free", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=free,
    )
    control_vx, control_vy = _run(
        tmp_path / "control", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=control,
    )
    residual_x = [a - b for a, b in zip(free_vx[0], control_vx[0])]
    residual_y = [a - b for a, b in zip(free_vy[0], control_vy[0])]
    source = (free.source_x_m, free.source_y_m)
    receiver = free.receivers_m[0]
    p_ray = two_segment_ray(
        source, receiver, boundary_y_m=SURFACE_Y_M,
        incident_velocity_m_s=free.vp_m_s, outgoing_velocity_m_s=free.vp_m_s,
    )
    sv_ray = two_segment_ray(
        source, receiver, boundary_y_m=SURFACE_Y_M,
        incident_velocity_m_s=free.vp_m_s, outgoing_velocity_m_s=free.vs_m_s,
    )
    source_peak = 1.5 / free.source_frequency_hz
    p_peak = source_peak + p_ray.travel_time_s
    sv_peak = source_peak + sv_ray.travel_time_s
    p_direction = (receiver[0] - p_ray.boundary_x_m, receiver[1] - SURFACE_Y_M)
    sv_direction = (receiver[0] - sv_ray.boundary_x_m, receiver[1] - SURFACE_Y_M)
    direct_direction = (receiver[0] - source[0], receiver[1] - source[1])
    direct_longitudinal, _ = project_components(free_vx[0], free_vy[0], direct_direction)
    p_longitudinal, p_transverse = project_components(residual_x, residual_y, p_direction)
    sv_longitudinal, sv_transverse = project_components(residual_x, residual_y, sv_direction)
    p_pick = (
        absolute_peak_index_in_interval(
            p_longitudinal, start_s=p_peak - PEAK_HALF_WIDTH_S,
            stop_s=p_peak + PEAK_HALF_WIDTH_S, dt_s=free.dt_s,
        ) + 1
    ) * free.dt_s
    sv_pick = (
        absolute_peak_index_in_interval(
            sv_transverse, start_s=sv_peak - PEAK_HALF_WIDTH_S,
            stop_s=sv_peak + PEAK_HALF_WIDTH_S, dt_s=free.dt_s,
        ) + 1
    ) * free.dt_s
    direct_travel = math.hypot(*direct_direction) / free.vp_m_s
    direct_peak = source_peak + direct_travel
    direct_pick = (
        absolute_peak_index_in_interval(
            direct_longitudinal, start_s=direct_peak - PEAK_HALF_WIDTH_S,
            stop_s=direct_peak + PEAK_HALF_WIDTH_S, dt_s=free.dt_s,
        ) + 1
    ) * free.dt_s
    p_long_e = signal_energy(_window(p_longitudinal, p_peak, free.dt_s))
    p_trans_e = signal_energy(_window(p_transverse, p_peak, free.dt_s))
    sv_long_e = signal_energy(_window(sv_longitudinal, sv_peak, free.dt_s))
    sv_trans_e = signal_energy(_window(sv_transverse, sv_peak, free.dt_s))
    incidence_angle = math.atan2(
        abs(p_ray.boundary_x_m - source[0]), source[1] - SURFACE_Y_M
    )
    coefficients = free_surface_p_coefficients(
        incidence_angle, vp_m_s=free.vp_m_s, vs_m_s=free.vs_m_s,
        density_kg_m3=free.density_kg_m3,
    )
    tolerance = 2.0 * free.dt_s + 0.005 * sv_ray.travel_time_s
    metrics = {
        "surface_y_m": SURFACE_Y_M,
        "p_ray": p_ray.__dict__,
        "sv_ray": sv_ray.__dict__,
        "expected_p_peak_s": p_peak,
        "observed_p_peak_s": p_pick,
        "expected_sv_peak_s": sv_peak,
        "observed_sv_peak_s": sv_pick,
        "expected_p_minus_direct_s": p_ray.travel_time_s - direct_travel,
        "observed_p_minus_direct_s": p_pick - direct_pick,
        "expected_sv_minus_p_s": sv_ray.travel_time_s - p_ray.travel_time_s,
        "observed_sv_minus_p_s": sv_pick - p_pick,
        "timing_tolerance_s": tolerance,
        "p_longitudinal_to_transverse_energy_ratio": p_long_e / p_trans_e,
        "sv_transverse_to_longitudinal_energy_ratio": sv_trans_e / sv_long_e,
        "plane_wave_coefficients_diagnostic": coefficients,
        "amplitudes_are_mandatory": False,
    }
    (tmp_path / "free_surface_oblique_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert p_peak < sv_peak
    assert abs((p_pick - direct_pick) - (p_ray.travel_time_s - direct_travel)) <= tolerance
    assert abs((sv_pick - p_pick) - (sv_ray.travel_time_s - p_ray.travel_time_s)) <= tolerance
    assert metrics["p_longitudinal_to_transverse_energy_ratio"] >= 10.0
    assert metrics["sv_transverse_to_longitudinal_energy_ratio"] >= 5.0


def test_psv_free_surface_mpi_reproducibility(
    tmp_path, repository_root, denise_binary, mpiexec
):
    config, _, _ = _normal_configs()
    reference = _run(
        tmp_path / "mpi_1x1", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config,
    )
    metrics = {}
    for label, nprocx, nprocy in (("2x1", 2, 1), ("1x2", 1, 2), ("2x2", 2, 2)):
        variant = _run(
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
    (tmp_path / "free_surface_mpi_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
