from __future__ import annotations

import math

import pytest

from tests.utilities.sh_free_surface import holberg_coefficients
from tests.utilities.visco_sh_free_surface import (
    SUPPORTED_FDORDERS,
    gsls_split_step,
    staggered_forward_2d,
    translated_image_geometry,
    visco_surface_rows,
)
from tests.cases.visco_sh_free_surface import runtime_scenario
from tests.utilities.visco_sh_free_surface_runtime import (
    acceptance_metadata,
    constitutive_surface_state_accepts,
)


@pytest.mark.parametrize("fd_order", SUPPORTED_FDORDERS)
def test_visco_surface_ghost_row_contract(fd_order):
    rows = visco_surface_rows(fd_order)
    m = fd_order // 2
    assert rows.active_vz_ghosts == (() if m == 1 else tuple(range(2 - m, 1)))
    assert rows.full_vz_extension == tuple(range(1 - m, 1))
    assert rows.total_syz_extension == tuple(range(1 - m, 1))
    assert rows.minimum_q_ghosts == ()
    assert rows.full_q_extension == tuple(range(1 - m, 1))


@pytest.mark.parametrize("mechanisms", (1, 3))
def test_zero_surface_gsls_state_is_invariant_for_arbitrary_mechanisms(mechanisms):
    b = tuple(0.73 + 0.02 * index for index in range(mechanisms))
    c = tuple(0.91 - 0.03 * index for index in range(mechanisms))
    d = tuple(2.5 + index for index in range(mechanisms))
    state = gsls_split_step(
        total_stress=0.0,
        memory=(0.0,) * mechanisms,
        derivative=0.0,
        instantaneous_increment_coefficient=17.0,
        dt=0.0005,
        b=b,
        c=c,
        d=d,
    )
    assert state.total_stress == 0.0
    assert state.memory == (0.0,) * mechanisms


@pytest.mark.parametrize("fd_order", SUPPORTED_FDORDERS)
@pytest.mark.parametrize("mechanisms", (1, 3))
def test_discrete_drivers_and_gsls_states_have_the_derived_y_parity(
    fd_order, mechanisms
):
    coefficients = holberg_coefficients(fd_order)
    m = fd_order // 2
    values = {}
    for row in range(-m - 2, m + 4):
        mirrored_distance = abs(row - 0.5)
        for column in range(-m - 1, m + 4):
            values[(row, column)] = 0.25 * mirrored_distance**2 + 0.5 * column**2

    dx_positive = staggered_forward_2d(
        values, row=2, column=0, axis="x", coefficients=coefficients
    )
    dx_negative = staggered_forward_2d(
        values, row=-1, column=0, axis="x", coefficients=coefficients
    )
    dy_positive = staggered_forward_2d(
        values, row=1, column=0, axis="y", coefficients=coefficients
    )
    dy_negative = staggered_forward_2d(
        values, row=-1, column=0, axis="y", coefficients=coefficients
    )
    assert dx_negative == pytest.approx(dx_positive)
    assert dy_negative == pytest.approx(-dy_positive)

    b = tuple(0.8 - 0.03 * index for index in range(mechanisms))
    c = tuple(0.9 - 0.04 * index for index in range(mechanisms))
    d = tuple(1.0 + index for index in range(mechanisms))
    memory = tuple(0.1 * (index + 1) for index in range(mechanisms))
    coefficients = {
        "instantaneous_increment_coefficient": 7.0,
        "dt": 0.002,
        "b": b,
        "c": c,
        "d": d,
    }
    positive_r = gsls_split_step(
        total_stress=0.4,
        memory=memory,
        derivative=dx_positive,
        **coefficients,
    )
    negative_r = gsls_split_step(
        total_stress=0.4,
        memory=memory,
        derivative=dx_negative,
        **coefficients,
    )
    assert negative_r.total_stress == pytest.approx(positive_r.total_stress)
    assert negative_r.memory == pytest.approx(positive_r.memory)

    positive_q = gsls_split_step(
        total_stress=0.4,
        memory=memory,
        derivative=dy_positive,
        **coefficients,
    )
    negative_q = gsls_split_step(
        total_stress=-0.4,
        memory=tuple(-value for value in memory),
        derivative=dy_negative,
        **coefficients,
    )
    assert negative_q.total_stress == pytest.approx(-positive_q.total_stress)
    assert negative_q.memory == pytest.approx(tuple(-value for value in positive_q.memory))


def test_translated_image_problem_preserves_native_phase_and_both_path_lengths():
    geometry = translated_image_geometry(
        candidate_source=(605.0, 305.0),
        candidate_receiver=(1005.0, 205.0),
        reference_plane_y=1000.0,
        dh=10.0,
    )
    assert geometry.reference_direct_distance == pytest.approx(
        geometry.candidate_direct_distance
    )
    assert geometry.reference_image_distance == pytest.approx(
        geometry.candidate_image_distance
    )
    assert geometry.reference_real_source[1] / 10.0 - 0.5 == pytest.approx(130.0)
    assert geometry.reference_image_source[1] / 10.0 - 0.5 == pytest.approx(69.0)
    assert geometry.reference_receiver[1] / 10.0 - 0.5 == pytest.approx(120.0)
    assert math.isclose(geometry.reference_plane_y / 10.0, 100.0)


@pytest.mark.parametrize(
    ("source", "receiver", "plane", "message"),
    (
        ((600.0, 300.0), (1005.0, 205.0), 1000.0, "native vz"),
        ((605.0, 305.0), (1000.0, 200.0), 1000.0, "native vz"),
        ((605.0, 305.0), (1005.0, 205.0), 1005.0, "native syz"),
        ((605.0, 0.0), (1005.0, 205.0), 1000.0, "below y=0"),
        ((605.0, 305.0), (1005.0, -5.0), 1000.0, "below y=0"),
        ((605.0, 305.0), (1005.0, 205.0), 300.0, "mirrored source"),
    ),
)
def test_translated_image_geometry_rejects_non_native_or_unsafe_inputs(
    source, receiver, plane, message
):
    with pytest.raises(ValueError, match=message):
        translated_image_geometry(
            candidate_source=source,
            candidate_receiver=receiver,
            reference_plane_y=plane,
            dh=10.0,
        )


@pytest.mark.parametrize("geometry_name", ("normal", "oblique"))
@pytest.mark.parametrize("plane_y", (1200.0, 1600.0))
def test_runtime_geometry_is_native_translated_and_externally_uncontaminated(
    geometry_name, plane_y
):
    scenario = runtime_scenario(
        geometry=geometry_name, fd_order=12, reference_plane_y_m=plane_y
    )
    candidate = scenario.metadata("candidate")
    reference = scenario.metadata("reference_combined")
    assert candidate["candidate_direct_distance_m"] == pytest.approx(
        reference["reference_direct_distance_m"]
    )
    assert candidate["candidate_image_distance_m"] == pytest.approx(
        reference["reference_image_distance_m"]
    )
    assert reference["external_return_outside_comparison"] is True
    assert len(reference["nominal_denise_input_coordinates_m"]["sources"]) == 2
    for point in reference["native_vz_coordinates_m"]["sources"]:
        assert all(
            math.isclose(coordinate / 10.0 - 0.5, round(coordinate / 10.0 - 0.5))
            for coordinate in point
        )


def test_frozen_acceptance_is_independent_of_runtime_candidate():
    metadata = acceptance_metadata()
    assert metadata["rationale"]["frozen_before_candidate_execution"] is True
    assert metadata["waveform"] == {
        "relative_l2_max": 0.02,
        "normalized_correlation_min": 0.999,
        "signed_amplitude_error_max": 0.03,
        "arrival_lag_max_s": 0.001,
    }
    assert metadata["boundary"] == {
        "hard_keys": [
            "traction_residual",
            "dplus_vz_residual",
            "vz_parity_residual",
            "total_syz_parity_residual",
            "q_surface_residual",
        ],
        "hard_limits": {
            "traction_residual_max": 5.0e-6,
            "dplus_vz_residual_max": 5.0e-5,
            "vz_parity_residual_max": 2.0e-6,
            "total_syz_parity_residual_max": 2.0e-6,
            "q_surface_residual_max": 2.0e-6,
        },
        "diagnostic_only": {
            "q_parity_residual": {"acceptance_effect": "none"}
        },
    }
    assert metadata["stability"] == {
        "metric": "fixed post-source quarter max_abs_vz Q4/Q1",
        "q4_to_q1_max": 0.01,
        "source_off": {"quellart": 1, "n_order": 0, "n_off": 1257},
        "calibration_role": "finite-Q FD12 FREE_SURF=0 absorbing reference",
        "reference_calibration_q4_to_q1": 0.0003110815438951515,
    }


def test_constitutive_oracle_rejects_hidden_nonzero_q_at_zero_total_traction():
    assert constitutive_surface_state_accepts(
        total_syz0=0.0, q_surface=(0.0, 0.0), tolerance=1.0e-12
    )
    assert not constitutive_surface_state_accepts(
        total_syz0=0.0, q_surface=(0.0, 1.0e-5), tolerance=1.0e-12
    )
