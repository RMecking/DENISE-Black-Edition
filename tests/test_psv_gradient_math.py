from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from tests.utilities.psv_gradient import (
    PSVState,
    add_scaled,
    density_average_jvp,
    density_average_vjp,
    density_mass_jvp,
    density_mass_vjp,
    forward_step,
    harmonic_mu_jvp,
    harmonic_mu_vjp,
    material_fields,
    material_jvp,
    material_vjp,
    matrix_inner,
    matrix_norm,
    physical_parameter_chain,
    staggered_shear_modulus,
    state_inner,
    subtract_scaled,
    transpose_step,
)


Shape = tuple[int, int]


def _rng() -> random.Random:
    return random.Random(20260814)


def _random_matrix(rng: random.Random, shape: Shape, *, offset: float = 0.0, scale: float = 1.0) -> list[list[float]]:
    ny, nx = shape
    return [[offset + scale * rng.uniform(-1.0, 1.0) for _ in range(nx)] for _ in range(ny)]


def _random_state(rng: random.Random, shape: Shape) -> PSVState:
    return PSVState(*(_random_matrix(rng, shape) for _ in range(5)))


def _relative_dot_error(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0)


def test_full_state_step_transpose_closes_at_machine_precision() -> None:
    rng, shape = _rng(), (7, 9)
    state, adjoint = _random_state(rng, shape), _random_state(rng, shape)
    vp = _random_matrix(rng, shape, offset=2600.0, scale=100.0)
    vs = _random_matrix(rng, shape, offset=1400.0, scale=75.0)
    rho = _random_matrix(rng, shape, offset=1850.0, scale=50.0)
    fields = material_fields(vp, vs, rho)
    output = forward_step(state, dt_over_dh=2.5e-5, **fields)
    transposed = transpose_step(adjoint, dt_over_dh=2.5e-5, **fields)
    assert _relative_dot_error(state_inner(output, adjoint), state_inner(state, transposed)) < 2.0e-13


def test_four_cell_harmonic_mu_jvp_vjp_and_finite_difference() -> None:
    rng, shape = _rng(), (6, 8)
    mu = _random_matrix(rng, shape, offset=4.5e9, scale=5.0e8)
    delta = _random_matrix(rng, shape, scale=1.0e8)
    adjoint = _random_matrix(rng, shape)
    jvp, vjp = harmonic_mu_jvp(mu, delta), harmonic_mu_vjp(mu, adjoint)
    assert _relative_dot_error(matrix_inner(jvp, adjoint), matrix_inner(delta, vjp)) < 2.0e-13
    epsilon = 1.0e-5
    fd = subtract_scaled(
        staggered_shear_modulus(add_scaled(mu, delta, epsilon)),
        staggered_shear_modulus(add_scaled(mu, delta, -epsilon)),
        2.0 * epsilon,
    )
    difference = [[x - y for x, y in zip(row_x, row_y)] for row_x, row_y in zip(fd, jvp)]
    assert matrix_norm(difference) / matrix_norm(jvp) < 2.0e-9


def test_material_jvp_vjp_closes_including_staggered_shear() -> None:
    rng, shape = _rng(), (7, 8)
    vx, vy = _random_matrix(rng, shape), _random_matrix(rng, shape)
    mu = _random_matrix(rng, shape, offset=4.5e9, scale=5.0e8)
    delta_lam, delta_mu = _random_matrix(rng, shape), _random_matrix(rng, shape)
    adjoints = tuple(_random_matrix(rng, shape) for _ in range(3))
    jvp = material_jvp(vx, vy, delta_lam, delta_mu, mu=mu, dt_over_dh=4.0e-5)
    g_lam, g_mu = material_vjp(vx, vy, *adjoints, mu=mu, dt_over_dh=4.0e-5)
    lhs = sum(matrix_inner(value, adjoint) for value, adjoint in zip(jvp, adjoints))
    rhs = matrix_inner(delta_lam, g_lam) + matrix_inner(delta_mu, g_mu)
    assert _relative_dot_error(lhs, rhs) < 2.0e-13


def test_density_average_and_mass_jvp_vjp_close() -> None:
    rng, shape = _rng(), (7, 9)
    rho = _random_matrix(rng, shape, offset=1850.0, scale=50.0)
    delta = _random_matrix(rng, shape)
    adj_rx, adj_ry = _random_matrix(rng, shape), _random_matrix(rng, shape)
    delta_rx, delta_ry = density_average_jvp(rho, delta)
    g_rho = density_average_vjp(rho, adj_rx, adj_ry)
    assert _relative_dot_error(matrix_inner(delta_rx, adj_rx) + matrix_inner(delta_ry, adj_ry), matrix_inner(delta, g_rho)) < 2.0e-13
    state = _random_state(rng, shape)
    adj_vx, adj_vy = _random_matrix(rng, shape), _random_matrix(rng, shape)
    jvp_x, jvp_y = density_mass_jvp(state, rho, delta, dt_over_dh=4.0e-5)
    mass_vjp = density_mass_vjp(state, rho, adj_vx, adj_vy, dt_over_dh=4.0e-5)
    assert _relative_dot_error(matrix_inner(jvp_x, adj_vx) + matrix_inner(jvp_y, adj_vy), matrix_inner(delta, mass_vjp)) < 2.0e-13


def test_physical_vp_vs_rho_chain_matches_material_directional_derivative() -> None:
    rng, shape = _rng(), (6, 7)
    vp = _random_matrix(rng, shape, offset=2600.0, scale=100.0)
    vs = _random_matrix(rng, shape, offset=1400.0, scale=50.0)
    rho = _random_matrix(rng, shape, offset=1850.0, scale=50.0)
    g_lam, g_mu, g_rho_mass = (_random_matrix(rng, shape) for _ in range(3))
    d_vp, d_vs, d_rho = (_random_matrix(rng, shape) for _ in range(3))
    g_vp, g_vs, g_rho = physical_parameter_chain(g_lam, g_mu, g_rho_mass, vp=vp, vs=vs, rho=rho)
    ny, nx = shape
    d_lam = [[(vp[j][i] ** 2 - 2.0 * vs[j][i] ** 2) * d_rho[j][i] + 2.0 * rho[j][i] * vp[j][i] * d_vp[j][i] - 4.0 * rho[j][i] * vs[j][i] * d_vs[j][i] for i in range(nx)] for j in range(ny)]
    d_mu = [[vs[j][i] ** 2 * d_rho[j][i] + 2.0 * rho[j][i] * vs[j][i] * d_vs[j][i] for i in range(nx)] for j in range(ny)]
    material_product = matrix_inner(g_lam, d_lam) + matrix_inner(g_mu, d_mu) + matrix_inner(g_rho_mass, d_rho)
    physical_product = matrix_inner(g_vp, d_vp) + matrix_inner(g_vs, d_vs) + matrix_inner(g_rho, d_rho)
    assert _relative_dot_error(material_product, physical_product) < 2.0e-13


def test_m53_retained_diagnostic_provenance_is_closed() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    artifact = json.loads(
        (repository_root / "tests" / "m5.3_psv_gradient_audit.json").read_text(
            encoding="utf-8"
        )
    )
    patch = repository_root / "tests" / "m5.3_psv_instrumentation.patch"
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == artifact[
        "instrumentation_patch_sha256"
    ]
    text = patch.read_text(encoding="utf-8")
    assert "src/PSV/ass_gradPSV.c" in text
    assert "src/PSV/update_v_PML_PSV.c" in text
    assert "M53_PSV_DUMP_RAW" in text
    assert "M53_PSV_RECEIVER_METRIC" in text
    assert artifact["base_git_sha"] == "68d3bd68ff25ee7f225ade6bc88ec4beb6d6f96e"
    assert artifact["final_verdict"] == "MULTIPLE PSV GRADIENT DEFECTS IDENTIFIED"
    assert all(
        len(record["sha256"]) == 64
        for record in artifact["binary_provenance"].values()
    )
