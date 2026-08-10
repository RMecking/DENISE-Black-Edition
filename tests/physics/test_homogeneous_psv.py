from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case, with_geometry
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import (
    absolute_peak_index_in_window,
    all_finite,
    fit_propagation_velocity,
    first_break_index,
    normalized_correlation,
    project_components,
    read_ascii_seismograms,
    relative_amplitude_error,
    relative_l2,
    signal_energy,
    source_pick_delay,
    time_window,
)


pytestmark = pytest.mark.integration


def _run_case(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    config: HomogeneousPSVConfig,
    nprocx: int = 1,
    nprocy: int = 1,
) -> tuple[list[list[float]], list[list[float]]]:
    generate_case(directory, config=config, nprocx=nprocx, nprocy=nprocy)
    metadata = config.as_metadata() | {"nprocx": nprocx, "nprocy": nprocy}
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration=metadata,
    )
    assert result.returncode == 0, result_summary(result)

    component_traces = []
    for filename in ("homogeneous_vx.asc.shot1", "homogeneous_vy.asc.shot1"):
        output = directory / "su" / filename
        assert output.is_file() and output.stat().st_size > 0
        traces = read_ascii_seismograms(output, config.receiver_count, config.samples_per_trace)
        assert all_finite(traces)
        component_traces.append(traces)
    flattened = [
        sample for traces in component_traces for trace in traces for sample in trace
    ]
    assert signal_energy(flattened) > 0.0
    return component_traces[0], component_traces[1]


def _velocity_metrics(
    config: HomogeneousPSVConfig,
    traces: list[list[float]],
    velocity_m_s: float,
) -> tuple[dict[str, object], float]:
    smoothing_samples = round(0.25 / config.source_frequency_hz / config.dt_s)
    source_delay = source_pick_delay(
        samples=config.samples_per_trace,
        dt=config.dt_s,
        frequency_hz=config.source_frequency_hz,
        smoothing_samples=smoothing_samples,
    )
    raw_picks = [
        (first_break_index(trace, smoothing_samples=smoothing_samples) + 1) * config.dt_s
        for trace in traces
    ]
    offsets = config.receiver_offsets_m()
    fit = fit_propagation_velocity(offsets, raw_picks)
    relative_error = abs(fit.velocity_m_s - velocity_m_s) / velocity_m_s
    receivers = []
    for receiver, offset, raw_pick, residual in zip(
        config.receivers_m, offsets, raw_picks, fit.residuals_s
    ):
        expected = offset / velocity_m_s
        observed = raw_pick - source_delay
        receivers.append(
            {
                "receiver_m": list(receiver),
                "offset_m": offset,
                "raw_pick_s": raw_pick,
                "expected_travel_time_s": expected,
                "source_delay_corrected_pick_s": observed,
                "absolute_travel_time_error_s": observed - expected,
                "fit_residual_s": residual,
            }
        )
    metrics = {
        "model_velocity_m_s": velocity_m_s,
        "fitted_velocity_m_s": fit.velocity_m_s,
        "relative_velocity_error": relative_error,
        "fitted_intercept_s": fit.intercept_s,
        "maximum_absolute_residual_s": fit.maximum_absolute_residual_s,
        "residual_tolerance_s": 2.0 * config.dt_s,
        "receivers": receivers,
    }
    return metrics, relative_error


def _sv_peak_velocity_metrics(
    config: HomogeneousPSVConfig, traces: list[list[float]]
) -> tuple[dict[str, object], float]:
    offsets = config.receiver_offsets_m()
    half_width = 0.5 / config.source_frequency_hz
    peak_delay = 1.5 / config.source_frequency_hz
    raw_picks = [
        (absolute_peak_index_in_window(
            trace,
            center_s=peak_delay + offset / config.vs_m_s,
            half_width_s=half_width,
            dt_s=config.dt_s,
        ) + 1) * config.dt_s
        for offset, trace in zip(offsets, traces)
    ]
    fit = fit_propagation_velocity(offsets, raw_picks)
    relative_error = abs(fit.velocity_m_s - config.vs_m_s) / config.vs_m_s
    receivers = [
        {
            "receiver_m": list(receiver),
            "offset_m": offset,
            "direct_arrival_peak_s": pick,
            "expected_travel_time_s": offset / config.vs_m_s,
            "expected_peak_s": peak_delay + offset / config.vs_m_s,
            "fit_residual_s": residual,
        }
        for receiver, offset, pick, residual in zip(
            config.receivers_m, offsets, raw_picks, fit.residuals_s
        )
    ]
    metrics = {
        "pick_method": "maximum absolute transverse velocity in analytical S peak +/- 0.5/f",
        "pick_window_half_width_s": half_width,
        "model_velocity_m_s": config.vs_m_s,
        "fitted_velocity_m_s": fit.velocity_m_s,
        "relative_velocity_error": relative_error,
        "fitted_intercept_s": fit.intercept_s,
        "maximum_absolute_residual_s": fit.maximum_absolute_residual_s,
        "residual_tolerance_s": 2.0 * config.dt_s,
        "receivers": receivers,
    }
    return metrics, relative_error


def test_homogeneous_psv_p_velocity(tmp_path, repository_root, denise_binary, mpiexec):
    config = HomogeneousPSVConfig()
    vx, _ = _run_case(
        tmp_path / "p_velocity", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config,
    )
    metrics, relative_error = _velocity_metrics(config, vx, config.vp_m_s)
    (tmp_path / "p_velocity" / "p_velocity_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert relative_error <= 0.01
    assert metrics["maximum_absolute_residual_s"] <= 2.0 * config.dt_s


def test_homogeneous_psv_sv_velocity(tmp_path, repository_root, denise_binary, mpiexec):
    base = HomogeneousPSVConfig()
    receivers = tuple(
        (base.source_x_m + 0.6 * offset, base.source_y_m + 0.8 * offset)
        for offset in range(200, 700, 100)
    )
    config = with_geometry(
        base,
        receivers_m=receivers,
        source_type=4,
        source_azimuth_deg=math.degrees(math.atan2(0.8, 0.6)),
    )
    vx, vy = _run_case(
        tmp_path / "sv_velocity", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config,
    )
    transverse = [
        project_components(trace_x, trace_y, (receiver[0] - config.source_x_m, receiver[1] - config.source_y_m))[1]
        for receiver, trace_x, trace_y in zip(config.receivers_m, vx, vy)
    ]
    metrics, relative_error = _sv_peak_velocity_metrics(config, transverse)
    (tmp_path / "sv_velocity" / "sv_velocity_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert relative_error <= 0.01
    assert metrics["maximum_absolute_residual_s"] <= 2.0 * config.dt_s


def test_homogeneous_psv_polarization(tmp_path, repository_root, denise_binary, mpiexec):
    base = HomogeneousPSVConfig()
    receiver = (1300.0, 1400.0)
    config = with_geometry(base, receivers_m=(receiver,), source_type=2)
    vx, vy = _run_case(
        tmp_path / "polarization", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config,
    )
    direction = (receiver[0] - config.source_x_m, receiver[1] - config.source_y_m)
    parallel, perpendicular = project_components(vx[0], vy[0], direction)
    distance = math.hypot(*direction)
    peak_delay = 1.5 / config.source_frequency_hz
    half_width = 0.30 / config.source_frequency_hz
    p_parallel = time_window(
        parallel, center_s=peak_delay + distance / config.vp_m_s,
        half_width_s=half_width, dt_s=config.dt_s,
    )
    p_perpendicular = time_window(
        perpendicular, center_s=peak_delay + distance / config.vp_m_s,
        half_width_s=half_width, dt_s=config.dt_s,
    )
    sv_parallel = time_window(
        parallel, center_s=peak_delay + distance / config.vs_m_s,
        half_width_s=half_width, dt_s=config.dt_s,
    )
    sv_perpendicular = time_window(
        perpendicular, center_s=peak_delay + distance / config.vs_m_s,
        half_width_s=half_width, dt_s=config.dt_s,
    )
    p_ratio = signal_energy(p_parallel) / signal_energy(p_perpendicular)
    sv_ratio = signal_energy(sv_perpendicular) / signal_energy(sv_parallel)
    metrics = {
        "direction_vector_m": list(direction),
        "distance_m": distance,
        "window_half_width_s": half_width,
        "p_parallel_energy": signal_energy(p_parallel),
        "p_perpendicular_energy": signal_energy(p_perpendicular),
        "p_longitudinal_dominance_ratio": p_ratio,
        "sv_parallel_energy": signal_energy(sv_parallel),
        "sv_perpendicular_energy": signal_energy(sv_perpendicular),
        "sv_transverse_dominance_ratio": sv_ratio,
    }
    (tmp_path / "polarization" / "polarization_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert p_ratio >= 10.0
    assert sv_ratio >= 10.0


def test_homogeneous_psv_source_symmetry(tmp_path, repository_root, denise_binary, mpiexec):
    base = HomogeneousPSVConfig()
    receivers = ((1500.0, 1000.0), (500.0, 1000.0))
    config = with_geometry(base, receivers_m=receivers, source_type=2)
    vx, vy = _run_case(
        tmp_path / "symmetry", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config,
    )
    center = 1.5 / config.source_frequency_hz + 500.0 / config.vp_m_s
    half_width = 0.30 / config.source_frequency_hz
    right_x = time_window(vx[0], center_s=center, half_width_s=half_width, dt_s=config.dt_s)
    left_x = time_window(vx[1], center_s=center, half_width_s=half_width, dt_s=config.dt_s)
    radial_right, _ = project_components(vx[0], vy[0], (500.0, 0.0))
    radial_left, _ = project_components(vx[1], vy[1], (-500.0, 0.0))
    smoothing_samples = round(0.25 / config.source_frequency_hz / config.dt_s)
    pick_difference = abs(
        first_break_index(radial_right, smoothing_samples=smoothing_samples)
        - first_break_index(radial_left, smoothing_samples=smoothing_samples)
    ) * config.dt_s
    metrics = {
        "expected_component_polarity": "Gxx vx retains polarity; ideal transverse vy remains zero",
        "pick_difference_s": pick_difference,
        "vx_correlation": normalized_correlation([right_x], [left_x]),
        "vx_relative_amplitude_error": relative_amplitude_error(right_x, left_x),
        "right_transverse_to_radial_energy_ratio": signal_energy(vy[0]) / signal_energy(vx[0]),
        "left_transverse_to_radial_energy_ratio": signal_energy(vy[1]) / signal_energy(vx[1]),
    }
    (tmp_path / "symmetry" / "symmetry_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert pick_difference <= config.dt_s
    assert metrics["vx_correlation"] >= 0.999
    assert metrics["vx_relative_amplitude_error"] <= 0.01
    assert metrics["right_transverse_to_radial_energy_ratio"] <= 1.0e-3
    assert metrics["left_transverse_to_radial_energy_ratio"] <= 1.0e-3


def test_homogeneous_psv_reciprocity(tmp_path, repository_root, denise_binary, mpiexec):
    base = HomogeneousPSVConfig()
    point_a = (700.0, 1000.0)
    point_b = (1300.0, 1000.0)
    config_ab = with_geometry(base, source_m=point_a, receivers_m=(point_b,), source_type=2)
    config_ba = with_geometry(base, source_m=point_b, receivers_m=(point_a,), source_type=2)
    vx_ab, _ = _run_case(
        tmp_path / "reciprocity_ab", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config_ab,
    )
    vx_ba, _ = _run_case(
        tmp_path / "reciprocity_ba", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config_ba,
    )
    center = 1.5 / base.source_frequency_hz + 600.0 / base.vp_m_s
    half_width = 0.30 / base.source_frequency_hz
    window_ab = time_window(vx_ab[0], center_s=center, half_width_s=half_width, dt_s=base.dt_s)
    window_ba = time_window(vx_ba[0], center_s=center, half_width_s=half_width, dt_s=base.dt_s)
    metrics = {
        "relation": "G_xx(B,A,t) = G_xx(A,B,t)",
        "normalized_correlation": normalized_correlation([window_ab], [window_ba]),
        "relative_l2": relative_l2([window_ab], [window_ba]),
        "relative_amplitude_error": relative_amplitude_error(window_ab, window_ba),
    }
    (tmp_path / "reciprocity_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert metrics["normalized_correlation"] >= 0.99999
    assert metrics["relative_l2"] <= 1.0e-4
    assert metrics["relative_amplitude_error"] <= 1.0e-4


def test_homogeneous_psv_mpi_reproducibility(
    tmp_path, repository_root, denise_binary, mpiexec
):
    base = HomogeneousPSVConfig()
    config = with_geometry(
        base,
        receivers_m=((1300.0, 1400.0), (1400.0, 1300.0), (700.0, 1400.0)),
        source_type=2,
    )
    reference = _run_case(
        tmp_path / "mpi_1x1", repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config,
    )
    metrics = {}
    for label, nprocx, nprocy in (("2x1", 2, 1), ("1x2", 1, 2), ("2x2", 2, 2)):
        variant = _run_case(
            tmp_path / f"mpi_{label}", repository_root=repository_root,
            denise_binary=denise_binary, mpiexec=mpiexec, config=config,
            nprocx=nprocx, nprocy=nprocy,
        )
        component_metrics = {}
        for name, reference_component, variant_component in zip(("vx", "vy"), reference, variant):
            rel_error = relative_l2(reference_component, variant_component)
            correlation = normalized_correlation(reference_component, variant_component)
            component_metrics[name] = {
                "relative_l2": rel_error,
                "normalized_correlation": correlation,
            }
            assert rel_error <= 1.0e-5
            assert correlation >= 0.999999
        metrics[label] = {"mpi_ranks": nprocx * nprocy, "components": component_metrics}
    (tmp_path / "psv_mpi_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
