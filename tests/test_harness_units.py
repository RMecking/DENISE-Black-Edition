from __future__ import annotations

import math

import pytest

from tests.conftest import unavailable_dependency
from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case as generate_psv_case
from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case
from tests.utilities.runner import executable_sha256
from tests.utilities.seismogram import (
    absolute_peak_index_in_window,
    fit_propagation_velocity,
    first_break_index,
    normalized_correlation,
    project_components,
    read_ascii_seismograms,
    relative_amplitude_error,
    relative_l2,
    ricker_wavelet,
    signal_energy,
    time_window,
)


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


def test_absolute_peak_index_in_window_returns_global_index():
    trace = [0.0, 1.0, -4.0, 2.0, 9.0]
    assert absolute_peak_index_in_window(
        trace, center_s=0.2, half_width_s=0.11, dt_s=0.1
    ) == 2
