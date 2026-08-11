from __future__ import annotations

import math
from array import array

import pytest

from tests.conftest import unavailable_dependency
from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case as generate_psv_case
from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case
from tests.cases.layered_psv import LayeredPSVConfig, generate_case as generate_layered_case
from tests.utilities.elastic_analytics import (
    free_surface_p_coefficients,
    two_segment_ray,
    zoeppritz_p_coefficients,
)
from tests.utilities.runner import executable_sha256
from tests.utilities.staggered_grid import (
    collocate_velocity_at_sxy,
    denise_grid_index,
    field_position,
    input_coordinate_for_field_position,
    input_field_position,
    sxy_collocation_stencil,
)
from tests.utilities.seismogram import (
    absolute_peak_index_in_interval,
    absolute_peak_index_in_window,
    cpml_reflection_metrics,
    fit_propagation_velocity,
    first_break_index,
    normalized_correlation,
    project_components,
    read_ascii_seismograms,
    relative_amplitude_error,
    relative_l2,
    ricker_wavelet,
    signal_energy,
    time_interval,
    time_window,
)
from tests.physics import test_viscoelastic_q as viscoelastic_q_tests


def test_case_generator_writes_expected_native_model_and_inputs(tmp_path):
    config = generate_case(tmp_path, nprocx=2, nprocy=1)
    expected_bytes = config.nx * config.ny * 4
    assert (tmp_path / "model" / "homogeneous.vs").stat().st_size == expected_bytes
    assert (tmp_path / "model" / "homogeneous.rho").stat().st_size == expected_bytes
    assert len((tmp_path / "receiver.dat").read_text(encoding="ascii").splitlines()) == config.receiver_count
    assert "NPROCX =2" in (tmp_path / "denise.inp").read_text(encoding="ascii")


def test_case_generator_rejects_incompatible_decomposition(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        generate_case(tmp_path, config=HomogeneousSHConfig(nx=201), nprocx=2)


def test_ascii_reader_reshapes_receiver_major_data(tmp_path):
    path = tmp_path / "seismograms.asc"
    path.write_text("1\n2\n3\n4\n5\n6\n", encoding="ascii")
    assert read_ascii_seismograms(path, 2, 3) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_first_break_picker_recovers_a_known_wavelet_delay():
    dt = 0.0005
    wavelet = ricker_wavelet(500, dt, 10.0)
    delay_samples = 137
    delayed = [0.0] * delay_samples + wavelet
    smoothing = round(0.25 / 10.0 / dt)
    source_pick = first_break_index(wavelet, smoothing_samples=smoothing)
    observed_pick = first_break_index(delayed, smoothing_samples=smoothing)
    assert observed_pick - source_pick == delay_samples


def test_comparison_metrics_have_expected_values():
    first = [[1.0, 2.0], [-1.0, 0.5]]
    assert relative_l2(first, first) == 0.0
    assert math.isclose(normalized_correlation(first, first), 1.0)


def test_velocity_fit_recovers_slope_and_free_intercept():
    offsets = [200.0, 300.0, 400.0, 500.0, 600.0]
    picks = [0.12 + offset / 2000.0 for offset in offsets]
    fit = fit_propagation_velocity(offsets, picks)
    assert math.isclose(fit.velocity_m_s, 2000.0)
    assert math.isclose(fit.intercept_s, 0.12)
    assert fit.maximum_absolute_residual_s < 1.0e-12


def test_missing_dependency_skips_in_development_mode():
    with pytest.raises(pytest.skip.Exception):
        unavailable_dependency("missing", required=False)


def test_missing_dependency_fails_in_verification_mode():
    with pytest.raises(pytest.fail.Exception):
        unavailable_dependency("missing", required=True)


def test_repaired_q_guards_are_not_xfailed_and_keep_the_sensitivity_threshold():
    guards = (
        viscoelastic_q_tests.test_sh_mode0_qs_sensitivity,
        viscoelastic_q_tests.test_psv_qp_input_sensitivity,
        viscoelastic_q_tests.test_psv_qs_input_sensitivity,
    )
    for guard in guards:
        assert not any(mark.name == "xfail" for mark in getattr(guard, "pytestmark", ()))

    with pytest.raises(viscoelastic_q_tests.KnownViscoelasticQDefect):
        viscoelastic_q_tests._require_q_sensitivity(0.0, "test fixture")
    viscoelastic_q_tests._require_q_sensitivity(1.0e-3, "test fixture")
    assert not isinstance(AssertionError("unrelated"), viscoelastic_q_tests.KnownViscoelasticQDefect)
    assert not isinstance(RuntimeError("unrelated"), viscoelastic_q_tests.KnownViscoelasticQDefect)


def test_executable_hash_uses_sha256(tmp_path):
    executable = tmp_path / "denise"
    executable.write_bytes(b"DENISE verification fixture\n")
    assert executable_sha256(executable) == "7066bfa05e8b9d79ea04630c63754c5e442c4cc93c9a43e4bbdfeb12fd84b7d0"


def test_psv_case_generator_writes_three_models_and_two_component_outputs(tmp_path):
    config = generate_psv_case(tmp_path)
    expected_bytes = config.nx * config.ny * 4
    assert (tmp_path / "model" / "homogeneous.vp").stat().st_size == expected_bytes
    assert (tmp_path / "model" / "homogeneous.vs").stat().st_size == expected_bytes
    assert (tmp_path / "model" / "homogeneous.rho").stat().st_size == expected_bytes
    parameters = (tmp_path / "denise.inp").read_text(encoding="ascii")
    assert "PHYSICS =1" in parameters
    assert "SEIS_FILE_VX =su/homogeneous_vx.asc" in parameters
    assert "SEIS_FILE_VY =su/homogeneous_vy.asc" in parameters
    assert math.isclose(config.courant_number, 0.12)
    assert math.isclose(config.conservative_s_wavelength_points, 7.2)


def test_psv_case_generator_rejects_incompatible_decomposition(tmp_path):
    with pytest.raises(ValueError, match="divisible"):
        generate_psv_case(tmp_path, config=HomogeneousPSVConfig(nx=201), nprocx=2)


def test_component_projection_recovers_parallel_and_perpendicular_motion():
    parallel, perpendicular = project_components([3.0, -3.0], [4.0, -4.0], (3.0, 4.0))
    assert parallel == [5.0, -5.0]
    assert all(abs(value) < 1.0e-12 for value in perpendicular)


def test_time_window_energy_and_relative_amplitude():
    trace = [0.0, 1.0, 2.0, 3.0, 0.0]
    window = time_window(trace, center_s=0.3, half_width_s=0.11, dt_s=0.1)
    assert window == [1.0, 2.0, 3.0]
    assert signal_energy(window) == 14.0
    assert relative_amplitude_error(window, [-1.0, -2.0, -3.0]) == 0.0


@pytest.mark.parametrize("start_s, stop_s", [(0.0, 0.3), (0.2, 0.6)])
def test_time_interval_rejects_clipped_analysis_windows(start_s, stop_s):
    with pytest.raises(ValueError, match="not fully contained"):
        time_interval([1.0, 2.0, 3.0, 4.0, 5.0], start_s=start_s, stop_s=stop_s, dt_s=0.1)


def test_absolute_peak_index_in_window_returns_global_index():
    trace = [0.0, 1.0, -4.0, 2.0, 9.0]
    assert absolute_peak_index_in_window(
        trace, center_s=0.2, half_width_s=0.11, dt_s=0.1
    ) == 2


def test_absolute_peak_interval_can_exclude_an_earlier_larger_peak():
    trace = [0.0, 9.0, 0.0, -4.0, 0.0]
    assert absolute_peak_index_in_interval(
        trace, start_s=0.3, stop_s=0.5, dt_s=0.1
    ) == 3


def test_cpml_reflection_metric_has_known_ratio_and_decibels():
    reference = [2.0, 0.0, 0.0, 0.0]
    compact = [2.0, 0.0, 0.2, 0.0]
    metrics = cpml_reflection_metrics(
        compact,
        reference,
        dt_s=0.1,
        direct_window_s=(0.1, 0.15),
        reflection_window_s=(0.25, 0.35),
    )
    assert metrics.direct_l2 == 2.0
    assert math.isclose(metrics.late_residual_l2, 0.2)
    assert math.isclose(metrics.reflection_ratio, 0.1)
    assert math.isclose(metrics.reflection_db, -20.0)


def test_free_surface_plane_wave_solution_recovers_normal_incidence_polarity():
    coefficients = free_surface_p_coefficients(
        0.0, vp_m_s=3000.0, vs_m_s=1800.0, density_kg_m3=2000.0
    )
    assert math.isclose(coefficients["reflected_p_displacement"], -1.0)
    assert abs(coefficients["reflected_sv_displacement"]) < 1.0e-12


def test_zoeppritz_solver_matches_normal_incidence_impedance_formula():
    coefficients = zoeppritz_p_coefficients(
        0.0,
        vp1_m_s=3000.0, vs1_m_s=1800.0, rho1_kg_m3=2000.0,
        vp2_m_s=3600.0, vs2_m_s=2100.0, rho2_kg_m3=2300.0,
    )
    expected = (2300.0 * 3600.0 - 2000.0 * 3000.0) / (
        2300.0 * 3600.0 + 2000.0 * 3000.0
    )
    assert math.isclose(coefficients["reflected_p_displacement"], expected)
    assert abs(coefficients["reflected_sv_displacement"]) < 1.0e-12


def test_two_segment_ray_obeys_snell_law():
    ray = two_segment_ray(
        (0.0, 600.0), (500.0, 900.0), boundary_y_m=5.0,
        incident_velocity_m_s=3000.0, outgoing_velocity_m_s=1800.0,
    )
    outgoing_p = abs(500.0 - ray.boundary_x_m) / (
        1800.0 * ray.outgoing_distance_m
    )
    assert math.isclose(ray.horizontal_slowness_s_m, outgoing_p, rel_tol=1.0e-10)


def test_layered_generator_writes_declared_row_assignment(tmp_path):
    config = LayeredPSVConfig(nx=2, ny=4, interface_upper_row=2)
    generate_layered_case(tmp_path, config=config)
    values = array("f")
    values.frombytes((tmp_path / "model" / "layered.vp").read_bytes())
    assert list(values) == [3000.0, 3000.0, 3600.0, 3600.0] * 2
    assert config.interface_y_m == 20.0


@pytest.mark.parametrize(
    "field, expected",
    [
        ("material", (15.0, 25.0)),
        ("sxx", (15.0, 25.0)),
        ("syy", (15.0, 25.0)),
        ("vx", (20.0, 25.0)),
        ("vy", (15.0, 30.0)),
        ("sxy", (20.0, 30.0)),
    ],
)
def test_staggered_field_positions(field, expected):
    assert field_position(2, 3, 10.0, field) == expected


def test_input_coordinates_are_rounded_before_staggered_mapping():
    assert denise_grid_index(24.9, 10.0) == 2
    assert denise_grid_index(25.0, 10.0) == 3
    assert input_field_position((20.0, 30.0), 10.0, "vx") == (20.0, 25.0)
    assert input_field_position((20.0, 30.0), 10.0, "vy") == (15.0, 30.0)


def test_inverse_field_coordinate_derives_equal_path_receiver_input():
    assert input_coordinate_for_field_position(1705.0, 10.0, axis="y", field="vx") == 1710.0
    assert input_coordinate_for_field_position(2180.0, 10.0, axis="y", field="vy") == 2180.0
    with pytest.raises(ValueError, match="not representable"):
        input_coordinate_for_field_position(1700.0, 10.0, axis="y", field="vx")


def test_sxy_collocation_stencil_has_required_neighbors():
    assert sxy_collocation_stencil((1400.0, 900.0), 10.0) == (
        (1400.0, 900.0), (1400.0, 910.0), (1410.0, 900.0)
    )


def test_sxy_collocation_recovers_constant_fields():
    assert collocate_velocity_at_sxy([2.0, 2.0], [2.0, 2.0], [-3.0, -3.0], [-3.0, -3.0]) == (
        [2.0, 2.0], [-3.0, -3.0]
    )


def test_sxy_collocation_recovers_linear_fields_at_midpoint():
    vx, vy = collocate_velocity_at_sxy(
        [1.0, 3.0], [3.0, 5.0], [10.0, 14.0], [14.0, 18.0]
    )
    assert vx == [2.0, 4.0]
    assert vy == [12.0, 16.0]
