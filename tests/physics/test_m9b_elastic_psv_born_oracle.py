"""Frozen independent elastic P/SV Born / Born-adjoint scientific oracle."""

from __future__ import annotations

import math
import platform
import sys

import numpy as np
import pytest

from tests.utilities.elastic_psv_born_reference import (
    U,
    alignment_separation,
    born_adjoint,
    born_directional_fd_errors,
    born_forward,
    cfl_diagnostic,
    deterministic_data,
    deterministic_model_direction,
    dot,
    dot_metrics,
    hash_f64,
    image_diagnostics,
    lambda_reflector,
    make_experiment,
    migrate_shot_stack,
    mu_compact_direction,
    nonlinear_forward,
    receiver_transpose_basis,
    roundoff_gamma,
)


# The executable production integration gate intentionally remains the M9b-1
# RED frontier.  This frozen contract is the target for the future production
# adapter; no historical RTM path is accepted as its implementation.
M9B1_PRODUCTION_GATE = {
    "operator": "J(m0)^T d_mig",
    "data_components": ("vx", "vy"),
    "model_channels": ("delta_lambda", "delta_mu"),
    "model_density_perturbation": False,
    "inner_products": "raw Euclidean arrays; no DT or DH^2 factor",
    "sign": "<J dm,d>_D = <dm,J^T d>_M",
    "outputs": ("image_lambda_raw", "image_mu_raw"),
    "preprocessing_inside_operator": (),
    "comparison": "independent FP64 reference; production tolerance separately reviewed",
}


# Publication-baseline fingerprints cover every element of each deterministic
# raw gather/image. They are reproducibility evidence for the declared
# canonical runtime; equations and numerical scientific gates remain authority.
ACTIVE_BORN_SHA256 = {
    "shot_1_lambda": "f104a2850d1d62d9c28619e568bbdd7973164d95d832fcf9c0535358ce48a3ec",
    "shot_2_lambda": "afde7d4e5972dd8d89af25b10b45877c296d9ce619366e6fef279ccc85da3a4c",
    "shot_1_mu": "33d6546cc08511df16c92775618047add117d0f8a087f0b8ce8f59050b66954a",
    "shot_2_mu": "5138b9f06d3225701dbf71dd451a94492c47f496eb7f394458ec5672e78daef7",
}
ACTIVE_NORMAL_SHA256 = {
    "image_lambda_raw": "c03d496e2dfc60f181387d7c8ef7ecfcfc31114cef6d9ce5466b69f7f45d1270",
    "image_mu_raw": "9135afe8873e71eb8b16dd3ddd148b845f87f46d82f5e4bf46292b64376829db",
}


def _is_canonical_publication_runtime(
        *, os_family: str = sys.platform,
        machine: str | None = None,
        numpy_version: str = np.__version__) -> bool:
    """Whether exact publication fingerprints are defined for this runtime."""
    if machine is None:
        machine = platform.machine()
    return (
        os_family.lower().startswith("linux")
        and machine.lower() in {"x86_64", "amd64"}
        and numpy_version == "2.5.3"
        and np.dtype(np.float64).itemsize == 8
        and np.finfo(np.float64).nmant == 52
    )


def _assert_publication_fingerprints(
        actual: dict[str, str], expected: dict[str, str]) -> None:
    """Fail closed: canonical publication payloads must match exactly."""
    assert actual == expected


def _canonical_publication_payloads() -> dict[str, str]:
    """Compute the six reviewed canonical reproducibility payload hashes."""
    exp = make_experiment("active_cpml")
    dlam = lambda_reflector(exp)
    dmu = mu_compact_direction(exp)
    zero = np.zeros_like(dmu)
    trajectories = tuple(
        nonlinear_forward(exp, shot, save_strain=True)
        for shot in range(len(exp.sources)))
    lambda_data = tuple(
        born_forward(exp, trajectory, dlam, zero)
        for trajectory in trajectories)
    mu_data = tuple(
        born_forward(exp, trajectory, np.zeros_like(dlam), dmu)
        for trajectory in trajectories)
    image = migrate_shot_stack(exp, trajectories, lambda_data)
    return {
        "shot_1_lambda": hash_f64(lambda_data[0]),
        "shot_2_lambda": hash_f64(lambda_data[1]),
        "shot_1_mu": hash_f64(mu_data[0]),
        "shot_2_mu": hash_f64(mu_data[1]),
        "image_lambda_raw": hash_f64(image.image_lambda_raw),
        "image_mu_raw": hash_f64(image.image_mu_raw),
    }


@pytest.fixture(scope="module")
def active_case():
    exp = make_experiment("active_cpml")
    dlam = lambda_reflector(exp)
    dmu = mu_compact_direction(exp)
    zero = np.zeros_like(dmu)
    trajectories = []
    lambda_data = []
    mu_data = []
    for shot in range(len(exp.sources)):
        trajectory = nonlinear_forward(exp, shot, save_strain=True)
        trajectories.append(trajectory)
        lambda_data.append(born_forward(exp, trajectory, dlam, zero))
        mu_data.append(born_forward(exp, trajectory, zero, dmu))
    return exp, dlam, dmu, tuple(trajectories), tuple(lambda_data), tuple(mu_data)


def _assert_dot_closes(metrics: dict[str, float]) -> None:
    assert metrics["absolute_residual"] <= metrics["absolute_ceiling"], metrics
    assert metrics["relative_residual"] <= metrics["relative_ceiling"], metrics
    assert metrics["pair_relative_residual"] <= metrics["pair_relative_ceiling"], metrics
    assert metrics["relative_ceiling"] < 2.0e-6, metrics
    # Negating the whole image must miss the identity by far more than roundoff.
    assert abs(metrics["lhs"] + metrics["rhs"]) > 100.0 * metrics["absolute_ceiling"], metrics


def test_roundoff_bound_fails_closed_when_nu_reaches_one():
    nwork = math.ceil(1.0 / U)
    with pytest.raises(ValueError, match=r"n\*u < 1"):
        roundoff_gamma(nwork)


def test_canonical_linux_publication_fingerprints():
    # Direct classification evidence, independent of the host running pytest.
    assert _is_canonical_publication_runtime(
        os_family="linux", machine="x86_64", numpy_version="2.5.3")
    assert not _is_canonical_publication_runtime(
        os_family="win32", machine="AMD64", numpy_version="2.5.3")

    if not _is_canonical_publication_runtime():
        pytest.skip(
            "Exact publication fingerprints are defined for the canonical "
            "Linux/x86_64/NumPy 2.5.3 FP64 runtime; scientific numerical "
            "gates remain authoritative on this runtime.")

    actual = _canonical_publication_payloads()
    expected = ACTIVE_BORN_SHA256 | ACTIVE_NORMAL_SHA256
    _assert_publication_fingerprints(actual, expected)

    # A simulated canonical mismatch must remain a hard failure, not a no-op.
    wrong = dict(expected)
    wrong["shot_1_lambda"] = "0" * 64
    with pytest.raises(AssertionError):
        _assert_publication_fingerprints(actual, wrong)


def test_reference_nonlinear_born_derivative_converges_for_each_channel(active_case):
    exp, dlam, dmu, trajectories, _, _ = active_case
    eps, lambda_errors = born_directional_fd_errors(
        exp, dlam, np.zeros_like(dmu), trajectory=trajectories[0])
    _, mu_errors = born_directional_fd_errors(
        exp, np.zeros_like(dlam), dmu, trajectory=trajectories[0])
    _, combined_errors = born_directional_fd_errors(
        exp, dlam, dmu, trajectory=trajectories[0])
    assert tuple(eps) == (1.0, 0.5, 0.25, 0.125)
    for errors in (lambda_errors, mu_errors, combined_errors):
        # Central differences converge quadratically, by 4x per halving,
        # before the tested range approaches FP64 saturation.
        assert np.all(errors[1:] < errors[:-1] / 3.9), errors
        assert errors[-1] < 2.0e-5, errors


@pytest.mark.parametrize("data_component", (0, 1, None), ids=("vx-only", "vy-only", "vx-plus-vy"))
def test_j_jt_dot_product_interior_microcase(data_component):
    exp = make_experiment("interior")
    assert cfl_diagnostic(exp)["leapfrog_ratio"] < 1.0
    trajectory = nonlinear_forward(exp, 0, save_strain=True)
    lam_dir = lambda_reflector(exp)
    mu_dir = mu_compact_direction(exp)
    random_lam, random_mu = deterministic_model_direction(exp)
    vectors = (
        (lam_dir, np.zeros_like(mu_dir)),
        (np.zeros_like(lam_dir), mu_dir),
        (random_lam, random_mu),
    )
    d = deterministic_data(exp, seed=117, component=data_component)
    for dlam, dmu in vectors:
        _assert_dot_closes(dot_metrics(exp, dlam, dmu, d, trajectory))


@pytest.mark.parametrize("data_component", (0, 1, None), ids=("vx-only", "vy-only", "vx-plus-vy"))
def test_j_jt_dot_product_with_active_cpml(active_case, data_component):
    exp, dlam, dmu, trajectories, _, _ = active_case
    random_lam, random_mu = deterministic_model_direction(exp, seed=191)
    vectors = (
        (dlam, np.zeros_like(dmu)),
        (np.zeros_like(dlam), dmu),
        (random_lam, random_mu),
    )
    d = deterministic_data(exp, seed=229, component=data_component)
    for dm_lam, dm_mu in vectors:
        _assert_dot_closes(dot_metrics(exp, dm_lam, dm_mu, d, trajectories[0]))


def test_cpml_energy_reaches_the_active_layer(active_case):
    exp, _, _, trajectories, _, _ = active_case
    assert exp.cpml and exp.fw == 8
    for trajectory in trajectories:
        assert trajectory.pml_memory_peak > 1.0
        assert trajectory.pml_wave_energy_peak > 1.0
        assert trajectory.pml_energy_fraction_peak > 0.5


@pytest.mark.parametrize("component", (0, 1), ids=("vx", "vy"))
def test_receiver_basis_transpose_preserves_location_component_time_and_sign(component):
    exp = make_experiment("interior")
    receiver, timestep = 2, 7
    vx_bar, vy_bar = receiver_transpose_basis(exp, receiver, timestep, component)
    assert vx_bar.shape == (exp.nt, exp.ny, exp.nx)
    assert vy_bar.shape == vx_bar.shape
    expected = np.zeros_like(vx_bar)
    ri, rj = exp.receivers[receiver]
    expected[timestep, rj - 1, ri - 1] = 1.0
    if component == 0:
        np.testing.assert_array_equal(vx_bar, expected)
        assert not np.any(vy_bar)
    else:
        np.testing.assert_array_equal(vy_bar, expected)
        assert not np.any(vx_bar)

    # The basis identifies receiver shift, time reversal/shift, component swap,
    # and global sign as distinct operators, not visually judged differences.
    ri2, rj2 = exp.receivers[receiver + 1]
    shifted_receiver = np.zeros_like(expected)
    shifted_receiver[timestep, rj2 - 1, ri2 - 1] = 1.0
    assert not np.array_equal(expected, shifted_receiver)
    shifted_time = np.zeros_like(expected)
    shifted_time[exp.nt - 1 - timestep, rj - 1, ri - 1] = 1.0
    assert not np.array_equal(expected, shifted_time)
    opposite = receiver_transpose_basis(exp, receiver, timestep, 1 - component)[component]
    assert not np.array_equal(expected, opposite)
    negative = receiver_transpose_basis(exp, receiver, timestep, component, sign=-1.0)[component]
    np.testing.assert_array_equal(negative, -expected)


def test_time_alignment_falsifies_both_one_sample_shifts(active_case):
    exp, _, _, trajectories, data, _ = active_case
    metrics = alignment_separation(exp, trajectories[0], data[0])
    assert metrics["minus_relative_separation"] > 0.05, metrics
    assert metrics["plus_relative_separation"] > 0.05, metrics
    assert metrics["minus_norm_ratio"] != pytest.approx(1.0, abs=1e-5), metrics
    assert metrics["plus_norm_ratio"] != pytest.approx(1.0, abs=1e-5), metrics


def test_lambda_mu_channel_swap_is_strongly_falsified(active_case):
    exp, dlam, dmu, trajectories, lambda_data, mu_data = active_case
    expected = lambda_data[0]
    swapped = born_forward(exp, trajectories[0], dmu, dlam)
    separation = float(np.linalg.norm(expected - swapped) / np.linalg.norm(expected))
    assert separation > 1.0
    assert not np.allclose(expected, mu_data[0], rtol=1e-2, atol=0.0)
    # Separate channel columns and the joint direction are frozen by the dot tests.
    image = born_adjoint(exp, trajectories[0], expected)
    assert abs(dot(dlam, image.image_lambda_raw)) > 0.0
    assert abs(dot(dmu, image.image_mu_raw)) > 0.0


def test_known_reflector_normal_response_matches_full_fd_assembly(active_case):
    exp, dlam, dmu, trajectories, lambda_data, _ = active_case
    image = migrate_shot_stack(exp, trajectories, lambda_data)

    diagnostics = image_diagnostics(exp, image)
    assert abs(diagnostics["peak_j"] - 34.0) <= 1.0, diagnostics
    assert diagnostics["lambda_energy_fraction_near_reflector"] > 0.25, diagnostics
    assert diagnostics["lateral_symmetry_relative_residual"] < 0.1, diagnostics

    # Independent full-image normal reference: nonlinear central differences
    # generate d_FD per shot; analytic J^T is then applied to that complete data.
    fd_data = []
    for shot in range(len(exp.sources)):
        plus = nonlinear_forward(exp, shot, lam=exp.lam + 0.25 * dlam,
                                 mu=exp.mu).data
        minus = nonlinear_forward(exp, shot, lam=exp.lam - 0.25 * dlam,
                                  mu=exp.mu).data
        fd_data.append((plus - minus) / 0.5)
    fd_error = math.sqrt(sum(float(np.sum((fd_data[k] - lambda_data[k])**2))
                             for k in range(len(fd_data))))
    born_norm = math.sqrt(sum(float(np.sum(d**2)) for d in lambda_data))
    fd_relative_error = fd_error / born_norm
    fd_image = migrate_shot_stack(exp, trajectories, fd_data)
    normal_difference = math.sqrt(
        float(np.sum((fd_image.image_lambda_raw - image.image_lambda_raw)**2))
        + float(np.sum((fd_image.image_mu_raw - image.image_mu_raw)**2)))
    normal_norm = math.sqrt(float(np.sum(image.image_lambda_raw**2)
                                  + np.sum(image.image_mu_raw**2)))
    assert fd_relative_error < 5.0e-6
    assert normal_difference / normal_norm < 2.0 * fd_relative_error


def test_two_shot_raw_migration_is_linear_and_ordered(active_case):
    exp, _, _, trajectories, lambda_data, _ = active_case
    shot1 = migrate_shot_stack(exp, trajectories[:1], lambda_data[:1])
    shot2 = migrate_shot_stack(exp, trajectories[1:], lambda_data[1:])
    stacked = migrate_shot_stack(exp, trajectories, lambda_data)
    for got, first, second in (
        (stacked.image_lambda_raw, shot1.image_lambda_raw, shot2.image_lambda_raw),
        (stacked.image_mu_raw, shot1.image_mu_raw, shot2.image_mu_raw),
    ):
        scale = max(float(np.linalg.norm(got)), np.finfo(float).tiny)
        rel = float(np.linalg.norm(got - (first + second)) / scale)
        gamma = (2 * exp.nt * U) / (1.0 - 2 * exp.nt * U)
        assert rel <= 16.0 * gamma, (rel, gamma)


def test_born_data_and_full_normal_image_repeat_bitwise(active_case):
    exp, dlam, dmu, trajectories, lambda_data, _ = active_case
    zero = np.zeros_like(dmu)
    for shot, trajectory in enumerate(trajectories):
        again = born_forward(exp, trajectory, dlam, zero)
        np.testing.assert_array_equal(again, lambda_data[shot])
    image1 = migrate_shot_stack(exp, trajectories, lambda_data)
    image2 = migrate_shot_stack(exp, trajectories, lambda_data)
    np.testing.assert_array_equal(image1.image_lambda_raw, image2.image_lambda_raw)
    np.testing.assert_array_equal(image1.image_mu_raw, image2.image_mu_raw)
    data = deterministic_data(exp, seed=229, component=1)
    dot1 = dot_metrics(exp, dlam, dmu, data, trajectories[0])
    dot2 = dot_metrics(exp, dlam, dmu, data, trajectories[0])
    assert dot1 == dot2


def test_future_production_red_frontier_is_explicit_not_xfailed():
    assert M9B1_PRODUCTION_GATE == {
        "operator": "J(m0)^T d_mig",
        "data_components": ("vx", "vy"),
        "model_channels": ("delta_lambda", "delta_mu"),
        "model_density_perturbation": False,
        "inner_products": "raw Euclidean arrays; no DT or DH^2 factor",
        "sign": "<J dm,d>_D = <dm,J^T d>_M",
        "outputs": ("image_lambda_raw", "image_mu_raw"),
        "preprocessing_inside_operator": (),
        "comparison": "independent FP64 reference; production tolerance separately reviewed",
    }
