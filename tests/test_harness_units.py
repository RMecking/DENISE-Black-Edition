from __future__ import annotations

import math

import pytest

from tests.conftest import unavailable_dependency
from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case
from tests.utilities.runner import executable_sha256
from tests.utilities.seismogram import (
    fit_propagation_velocity,
    first_break_index,
    normalized_correlation,
    read_ascii_seismograms,
    relative_l2,
    ricker_wavelet,
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
