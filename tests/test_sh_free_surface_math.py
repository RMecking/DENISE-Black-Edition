from __future__ import annotations

import hashlib
import math

import pytest

from tests.cases.sh_free_surface import generate_case, normal_scenario, oblique_scenario
from tests.utilities.sh_free_surface import (
    SUPPORTED_FD_ORDERS,
    SurfaceState,
    SurfaceTimes,
    arrival_tolerance,
    backward_staggered,
    center_shear_modulus,
    evaluate_production_acceptance,
    evaluate_reflection,
    evaluate_surface_boundary,
    extend_surface_state,
    finite_nonzero,
    forward_staggered,
    holberg_coefficients,
    impedance_scaled_velocity,
    native_vz_position,
    normalized_correlation,
    normalized_surface_residuals,
    numerical_dispersion,
    peak_time,
    relative_l2,
    required_ghost_rows,
    ricker_f95,
    signed_amplitude_alpha,
    stability_modulation_limit,
    staggered_symbol,
    surface_candidate_times,
    surface_contract_errors,
    surface_roundoff_limits,
)


def _interior(fd_order: int) -> tuple[dict[int, float], dict[int, float]]:
    m = fd_order // 2
    return (
        {index: 0.25 + 0.7 * index for index in range(1, m + 2)},
        {index: -0.4 + 0.9 * index for index in range(1, m + 2)},
    )


@pytest.mark.parametrize("fd_order", SUPPORTED_FD_ORDERS)
def test_full_surface_state_and_staggered_cancellation(fd_order):
    vz, syz = _interior(fd_order)
    state = extend_surface_state(vz, syz, fd_order)
    rows = required_ghost_rows(fd_order)
    coefficients = holberg_coefficients(fd_order)

    assert tuple(sorted(index for index in state.vz if index <= 0)) == rows.full_velocity_rows
    assert tuple(sorted(index for index in state.syz if index <= 0)) == rows.stress_rows
    assert state.syz[0] == 0.0
    assert forward_staggered(state.vz, 0, coefficients) == 0.0
    assert surface_contract_errors(state, fd_order) == []

    for k in range(1, rows.half_order + 1):
        assert state.vz[1 - k] == state.vz[k]
    for k in range(1, rows.half_order):
        assert state.syz[-k] == -state.syz[k]


@pytest.mark.parametrize("fd_order", SUPPORTED_FD_ORDERS)
def test_active_and_full_velocity_row_contracts_are_distinct(fd_order):
    rows = required_ghost_rows(fd_order)
    assert rows.full_velocity_rows == tuple(range(1 - rows.half_order, 1))
    expected_active = tuple(range(2 - rows.half_order, 1)) if rows.half_order > 1 else ()
    assert rows.active_velocity_rows == expected_active
    assert set(rows.active_velocity_rows) < set(rows.full_velocity_rows)


@pytest.mark.parametrize("fd_order", (4, 6, 8, 10, 12))
def test_missing_outer_velocity_ghost_is_detected(fd_order):
    vz, syz = _interior(fd_order)
    state = extend_surface_state(vz, syz, fd_order)
    del state.vz[1 - fd_order // 2]
    assert any("missing vz row" in error or "surface derivative missing" in error
               for error in surface_contract_errors(state, fd_order))


@pytest.mark.parametrize("fd_order", SUPPORTED_FD_ORDERS)
def test_odd_velocity_parity_is_detected(fd_order):
    vz, syz = _interior(fd_order)
    state = extend_surface_state(vz, syz, fd_order)
    state.vz[0] = -state.vz[1]
    assert any("vz parity" in error for error in surface_contract_errors(state, fd_order))


@pytest.mark.parametrize("fd_order", (4, 6, 8, 10, 12))
def test_even_stress_parity_is_detected(fd_order):
    vz, syz = _interior(fd_order)
    state = extend_surface_state(vz, syz, fd_order)
    state.syz[-1] = state.syz[1]
    assert any("syz parity" in error for error in surface_contract_errors(state, fd_order))


@pytest.mark.parametrize("fd_order", (4, 12))
def test_shifted_row_one_surface_is_detected(fd_order):
    vz, syz = _interior(fd_order)
    correct = extend_surface_state(vz, syz, fd_order)
    shifted = SurfaceState(vz=dict(correct.vz), syz=dict(correct.syz))
    shifted.syz[1] = 0.0
    for k in range(1, fd_order // 2):
        shifted.syz[1 - k] = -syz[1 + k]
    assert surface_contract_errors(shifted, fd_order)


@pytest.mark.parametrize("fd_order", (4, 12))
def test_omitted_high_order_extension_is_detected(fd_order):
    vz, syz = _interior(fd_order)
    omitted = SurfaceState(vz=dict(vz), syz=dict(syz) | {0: 0.0})
    errors = surface_contract_errors(omitted, fd_order)
    assert any("missing vz row" in error for error in errors)
    assert any("missing syz row" in error for error in errors)


def test_fdorder2_is_positive_coverage_not_legacy_discriminator():
    rows = required_ghost_rows(2)
    assert rows.active_velocity_rows == ()
    assert rows.full_velocity_rows == (0,)
    assert rows.stress_rows == (0,)
    assert math.isfinite(backward_staggered({0: 0.0, 1: 2.0}, 1, holberg_coefficients(2)))


@pytest.mark.parametrize("fd_order", SUPPORTED_FD_ORDERS)
def test_holberg_contract_has_exact_half_order_length(fd_order):
    for max_relative_error in range(5):
        assert len(holberg_coefficients(fd_order, max_relative_error)) == fd_order // 2


def test_native_geometry_and_equal_vector_calibration():
    for scenario in (normal_scenario(), oblique_scenario()):
        metadata = scenario.metadata()
        source = metadata["native_vz_coordinates_m"]["source"]
        receiver = metadata["native_vz_coordinates_m"]["receiver"]
        calibration_source = metadata["native_vz_coordinates_m"]["calibration_source"]
        calibration_receiver = metadata["native_vz_coordinates_m"]["calibration_receiver"]
        image = metadata["image_source_m"]
        reflection_vector = (receiver[0] - image[0], receiver[1] - image[1])
        calibration_vector = (
            calibration_receiver[0] - calibration_source[0],
            calibration_receiver[1] - calibration_source[1],
        )
        assert reflection_vector == calibration_vector
        assert metadata["image_distance_m"] == metadata["calibration_distance_m"]
        assert metadata["surface_y_m"] == 0.0
        assert metadata["expected_vz_reflection_coefficient"] == 1.0
        assert metadata["surface_location_resolved"] is True


def test_nominal_input_is_not_native_vz_coordinate():
    assert native_vz_position((1200.0, 700.0), 10.0) == (1195.0, 695.0)


def test_surface_candidate_discrimination_uses_actual_geometry():
    source = (1195.0, 695.0)
    receiver = (1195.0, 995.0)
    times = surface_candidate_times(source, receiver, dh_m=10.0, vs_m_s=2000.0)
    tolerance = arrival_tolerance(
        dt_s=0.0005,
        reference_distance_m=1690.0,
        calibration_distance_m=1690.0,
        vs_m_s=2000.0,
        differential_dispersion_s=0.0,
    )
    assert math.isclose(times.y0_s, 0.845)
    assert tolerance == 0.001
    assert tolerance < times.half_minimum_separation_s
    assert abs(times.y0_s - times.y_half_h_s) > tolerance
    assert abs(times.y0_s - times.y_h_s) > tolerance


def test_staggered_symbol_and_dispersion_approach_continuum():
    coefficients = holberg_coefficients(8)
    xi = 1.0e-6
    assert math.isclose(staggered_symbol(xi, coefficients), xi, rel_tol=1.0e-3)
    dispersion = numerical_dispersion(
        distance_m=1000.0,
        angle_rad=0.4,
        frequency_hz=10.0,
        vs_m_s=2000.0,
        dt_s=0.0005,
        dh_m=10.0,
        fd_order=8,
    )
    assert abs(dispersion.group_velocity_m_s / 2000.0 - 1.0) < 0.01


def test_ricker_f95_is_derived_from_source_spectrum():
    f95 = ricker_f95(8.0)
    assert 8.0 < f95 < 16.0
    assert math.isclose(ricker_f95(16.0), 2.0 * f95, rel_tol=1.0e-12)


@pytest.mark.parametrize("fd_order", SUPPORTED_FD_ORDERS)
def test_predeclared_surface_and_stability_limits(fd_order):
    coefficients = holberg_coefficients(fd_order)
    syz_limit, dplus_limit = surface_roundoff_limits(coefficients)
    assert syz_limit == 32.0 * 2.0**-24
    assert dplus_limit == 64.0 * 2.0**-24 * sum(map(abs, coefficients))
    assert stability_modulation_limit(
        dt_s=0.0005, f95_hz=ricker_f95(8.0), coefficients=coefficients
    ) > dplus_limit


def test_surface_normalization_uses_the_larger_declared_scale():
    syz, dplus = normalized_surface_residuals(
        max_abs_syz0=2.0,
        max_abs_dplus_vz0=0.4,
        max_abs_interior_stress=8.0,
        max_impedance_vz=10.0,
        max_abs_dx_vz=0.5,
        max_abs_vz=2.0,
        f95_hz=10.0,
        vs_m_s=20.0,
    )
    assert syz == 0.2
    assert math.isclose(dplus, 0.4 / (2.0 * math.pi))


def test_impedance_scale_respects_sh_invmat_parameterization():
    rho = 2000.0
    vs = 1800.0
    vz = -0.25
    mu = rho * vs * vs
    assert center_shear_modulus(rho=rho, pu=vs, invmat1=1) == mu
    assert center_shear_modulus(rho=rho, pu=mu, invmat1=3) == mu
    expected = rho * vs * abs(vz)
    assert impedance_scaled_velocity(
        rho=rho, pu=vs, vz=vz, invmat1=1
    ) == expected
    assert impedance_scaled_velocity(
        rho=rho, pu=mu, vz=vz, invmat1=3
    ) == expected
    with pytest.raises(ValueError, match="INVMAT1=2"):
        center_shear_modulus(rho=rho, pu=vs, invmat1=2)


def _valid_boundary_metrics() -> dict[str, float]:
    return {
        "normalized_physical_traction": 0.5e-6,
        "physical_traction_limit": 1.0e-6,
        "max_velocity_parity_residual": 0.0,
        "max_stress_parity_residual": 0.0,
        "normalized_image_closure": 0.5e-6,
        "image_closure_limit": 1.0e-6,
    }


def test_locked_surface_boundary_acceptance_passes_valid_state():
    acceptance = evaluate_surface_boundary(**_valid_boundary_metrics())
    assert acceptance.physical_traction
    assert acceptance.velocity_parity
    assert acceptance.stress_parity
    assert acceptance.image_closure
    assert acceptance.all_pass


@pytest.mark.parametrize(
    "metric, expected_failed_field",
    (
        ("normalized_physical_traction", "physical_traction"),
        ("max_velocity_parity_residual", "velocity_parity"),
        ("max_stress_parity_residual", "stress_parity"),
        ("normalized_image_closure", "image_closure"),
    ),
)
def test_locked_surface_boundary_acceptance_rejects_each_independent_defect(
    metric, expected_failed_field
):
    values = _valid_boundary_metrics()
    values[metric] = 2.0e-6
    acceptance = evaluate_surface_boundary(**values)
    assert getattr(acceptance, expected_failed_field) is False
    assert acceptance.all_pass is False


def test_locked_reflection_and_combined_production_acceptance():
    reflection = evaluate_reflection(
        timing_error_s=0.001,
        observed_propagation_s=0.845,
        surface_times=SurfaceTimes(y0_s=0.845, y_half_h_s=0.84, y_h_s=0.835),
        timing_tolerance_s=0.001,
        signed_amplitude_alpha_value=1.0,
        normalized_correlation_value=1.0,
        absorbing_l2_ratio=0.01,
    )
    boundary = evaluate_surface_boundary(**_valid_boundary_metrics())
    assert reflection.all_pass
    assert evaluate_production_acceptance(
        healthy=True, reflection=reflection, boundary=boundary
    ).all_pass
    assert not evaluate_production_acceptance(
        healthy=False, reflection=reflection, boundary=boundary
    ).all_pass


def test_locked_reflection_acceptance_preserves_signed_amplitude_guard():
    reflection = evaluate_reflection(
        timing_error_s=0.0,
        observed_propagation_s=0.845,
        surface_times=SurfaceTimes(y0_s=0.845, y_half_h_s=0.84, y_h_s=0.835),
        timing_tolerance_s=0.001,
        signed_amplitude_alpha_value=-1.0,
        normalized_correlation_value=-1.0,
        absorbing_l2_ratio=0.01,
    )
    assert reflection.signed_amplitude is False
    assert reflection.phase is False
    assert reflection.all_pass is False


def test_reflection_metrics_preserve_signed_polarity_and_detect_wrong_sign():
    calibration = [0.0, 1.0, -2.0, 1.0, 0.0]
    reflection = [0.0, 1.02, -2.04, 1.02, 0.0]
    wrong_sign = [-value for value in reflection]
    assert math.isclose(signed_amplitude_alpha(reflection, calibration), 1.02)
    assert normalized_correlation(reflection, calibration) == 1.0
    assert signed_amplitude_alpha(wrong_sign, calibration) < 0.0
    assert normalized_correlation(wrong_sign, calibration) == -1.0
    assert relative_l2(calibration, reflection) < 0.05
    assert finite_nonzero(reflection)


def test_peak_picker_uses_declared_window():
    trace = [0.0] * 20
    trace[9] = -4.0
    trace[14] = 9.0
    assert peak_time(trace, expected_s=1.0, half_width_s=0.2, dt_s=0.1) == 1.0


def test_case_generator_keeps_physics_identical_across_mpi_decompositions(tmp_path):
    scenario = oblique_scenario(fd_order=12)
    hashes = []
    metadata = []
    for label, nprocx, nprocy in (("1x1", 1, 1), ("2x1", 2, 1), ("1x2", 1, 2)):
        directory = tmp_path / label
        generate_case(
            directory, scenario=scenario, role="free_surface",
            nprocx=nprocx, nprocy=nprocy,
        )
        hashes.append(tuple(
            hashlib.sha256((directory / "model" / f"homogeneous.{suffix}").read_bytes()).hexdigest()
            for suffix in ("vs", "rho")
        ))
        metadata.append((directory / "case.json").read_text(encoding="utf-8"))
        assert (directory / "source.dat").read_text(encoding="ascii") == (
            tmp_path / "1x1" / "source.dat"
        ).read_text(encoding="ascii")
        assert (directory / "receiver.dat").read_text(encoding="ascii") == (
            tmp_path / "1x1" / "receiver.dat"
        ).read_text(encoding="ascii")
    assert hashes[0] == hashes[1] == hashes[2]
    assert '"nprocx": 2' in metadata[1]
    assert '"nprocy": 2' in metadata[2]
    assert scenario.metadata()["native_vz_coordinates_m"]["source"][0] < 1200.0
    assert scenario.metadata()["native_vz_coordinates_m"]["receiver"][0] > 1200.0
    assert scenario.metadata()["native_vz_coordinates_m"]["source"][1] < 1200.0
    assert scenario.metadata()["native_vz_coordinates_m"]["receiver"][1] < 1200.0
