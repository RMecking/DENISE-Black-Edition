"""Independent periodic-grid oracle for the elastic DENISE PSV interior step.

All matrices use ``matrix[y][x]``. Normal stresses, density, lambda, and cell
shear modulus live at cell centres. ``vx``/``rx`` are on x-faces,
``vy``/``ry`` on y-faces, and ``sxy``/``mu_xy`` at cell corners. Periodic
boundaries isolate the interior operator and make its transpose unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


Matrix = list[list[float]]


@dataclass(frozen=True)
class PSVState:
    vx: Matrix
    vy: Matrix
    sxx: Matrix
    syy: Matrix
    sxy: Matrix


def _shape(a: Matrix) -> tuple[int, int]:
    if not a or not a[0] or any(len(row) != len(a[0]) for row in a):
        raise ValueError("matrix must be non-empty and rectangular")
    return len(a), len(a[0])


def _matrix(shape: tuple[int, int], fn: Callable[[int, int], float]) -> Matrix:
    ny, nx = shape
    return [[float(fn(j, i)) for i in range(nx)] for j in range(ny)]


def _xf(a: Matrix) -> Matrix:
    ny, nx = _shape(a)
    return _matrix((ny, nx), lambda j, i: a[j][(i + 1) % nx] - a[j][i])


def _xb(a: Matrix) -> Matrix:
    ny, nx = _shape(a)
    return _matrix((ny, nx), lambda j, i: a[j][i] - a[j][(i - 1) % nx])


def _yf(a: Matrix) -> Matrix:
    ny, nx = _shape(a)
    return _matrix((ny, nx), lambda j, i: a[(j + 1) % ny][i] - a[j][i])


def _yb(a: Matrix) -> Matrix:
    ny, nx = _shape(a)
    return _matrix((ny, nx), lambda j, i: a[j][i] - a[(j - 1) % ny][i])


def _xf_t(a: Matrix) -> Matrix:
    ny, nx = _shape(a)
    return _matrix((ny, nx), lambda j, i: a[j][(i - 1) % nx] - a[j][i])


def _xb_t(a: Matrix) -> Matrix:
    ny, nx = _shape(a)
    return _matrix((ny, nx), lambda j, i: a[j][i] - a[j][(i + 1) % nx])


def _yf_t(a: Matrix) -> Matrix:
    ny, nx = _shape(a)
    return _matrix((ny, nx), lambda j, i: a[(j - 1) % ny][i] - a[j][i])


def _yb_t(a: Matrix) -> Matrix:
    ny, nx = _shape(a)
    return _matrix((ny, nx), lambda j, i: a[j][i] - a[(j + 1) % ny][i])


def matrix_inner(a: Matrix, b: Matrix) -> float:
    if _shape(a) != _shape(b):
        raise ValueError("matrix shapes differ")
    return sum(x * y for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


def staggered_inverse_density(rho: Matrix) -> tuple[Matrix, Matrix]:
    ny, nx = _shape(rho)
    rx = _matrix((ny, nx), lambda j, i: 2.0 / (rho[j][i] + rho[j][(i + 1) % nx]))
    ry = _matrix((ny, nx), lambda j, i: 2.0 / (rho[j][i] + rho[(j + 1) % ny][i]))
    return rx, ry


def staggered_shear_modulus(mu: Matrix) -> Matrix:
    ny, nx = _shape(mu)
    return _matrix(
        (ny, nx),
        lambda j, i: 4.0
        / (
            1.0 / mu[j][i]
            + 1.0 / mu[j][(i + 1) % nx]
            + 1.0 / mu[(j + 1) % ny][i]
            + 1.0 / mu[(j + 1) % ny][(i + 1) % nx]
        ),
    )


def material_fields(vp: Matrix, vs: Matrix, rho: Matrix) -> dict[str, Matrix]:
    shape = _shape(vp)
    if _shape(vs) != shape or _shape(rho) != shape:
        raise ValueError("physical model shapes differ")
    lam = _matrix(shape, lambda j, i: rho[j][i] * (vp[j][i] ** 2 - 2.0 * vs[j][i] ** 2))
    mu = _matrix(shape, lambda j, i: rho[j][i] * vs[j][i] ** 2)
    rx, ry = staggered_inverse_density(rho)
    return {"lam": lam, "mu": mu, "mu_xy": staggered_shear_modulus(mu), "rx": rx, "ry": ry}


def strain(vx: Matrix, vy: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    exx, eyy = _xb(vx), _yb(vy)
    vxy, vyx = _yf(vx), _xf(vy)
    shape = _shape(vx)
    gamma = _matrix(shape, lambda j, i: vxy[j][i] + vyx[j][i])
    return exx, eyy, gamma


def stress_divergence(state: PSVState) -> tuple[Matrix, Matrix]:
    sxx_x, sxy_y = _xf(state.sxx), _yb(state.sxy)
    syy_y, sxy_x = _yf(state.syy), _xb(state.sxy)
    shape = _shape(state.vx)
    return (
        _matrix(shape, lambda j, i: sxx_x[j][i] + sxy_y[j][i]),
        _matrix(shape, lambda j, i: syy_y[j][i] + sxy_x[j][i]),
    )


def forward_step(
    state: PSVState,
    *,
    lam: Matrix,
    mu: Matrix,
    mu_xy: Matrix,
    rx: Matrix,
    ry: Matrix,
    dt_over_dh: float,
) -> PSVState:
    shape = _shape(state.vx)
    div_x, div_y = stress_divergence(state)
    vx = _matrix(shape, lambda j, i: state.vx[j][i] + dt_over_dh * rx[j][i] * div_x[j][i])
    vy = _matrix(shape, lambda j, i: state.vy[j][i] + dt_over_dh * ry[j][i] * div_y[j][i])
    exx, eyy, gamma = strain(vx, vy)
    sxx = _matrix(shape, lambda j, i: state.sxx[j][i] + dt_over_dh * ((lam[j][i] + 2.0 * mu[j][i]) * exx[j][i] + lam[j][i] * eyy[j][i]))
    syy = _matrix(shape, lambda j, i: state.syy[j][i] + dt_over_dh * (lam[j][i] * exx[j][i] + (lam[j][i] + 2.0 * mu[j][i]) * eyy[j][i]))
    sxy = _matrix(shape, lambda j, i: state.sxy[j][i] + dt_over_dh * mu_xy[j][i] * gamma[j][i])
    return PSVState(vx, vy, sxx, syy, sxy)


def transpose_step(
    adjoint_out: PSVState,
    *,
    lam: Matrix,
    mu: Matrix,
    mu_xy: Matrix,
    rx: Matrix,
    ry: Matrix,
    dt_over_dh: float,
) -> PSVState:
    shape = _shape(adjoint_out.vx)
    adj_exx = _matrix(shape, lambda j, i: dt_over_dh * ((lam[j][i] + 2.0 * mu[j][i]) * adjoint_out.sxx[j][i] + lam[j][i] * adjoint_out.syy[j][i]))
    adj_eyy = _matrix(shape, lambda j, i: dt_over_dh * (lam[j][i] * adjoint_out.sxx[j][i] + (lam[j][i] + 2.0 * mu[j][i]) * adjoint_out.syy[j][i]))
    adj_gamma = _matrix(shape, lambda j, i: dt_over_dh * mu_xy[j][i] * adjoint_out.sxy[j][i])
    exx_t, gamma_y_t = _xb_t(adj_exx), _yf_t(adj_gamma)
    eyy_t, gamma_x_t = _yb_t(adj_eyy), _xf_t(adj_gamma)
    avx = _matrix(shape, lambda j, i: adjoint_out.vx[j][i] + exx_t[j][i] + gamma_y_t[j][i])
    avy = _matrix(shape, lambda j, i: adjoint_out.vy[j][i] + eyy_t[j][i] + gamma_x_t[j][i])
    weighted_x = _matrix(shape, lambda j, i: dt_over_dh * rx[j][i] * avx[j][i])
    weighted_y = _matrix(shape, lambda j, i: dt_over_dh * ry[j][i] * avy[j][i])
    sxx_t, syy_t = _xf_t(weighted_x), _yf_t(weighted_y)
    sxy_xt, sxy_yt = _yb_t(weighted_x), _xb_t(weighted_y)
    return PSVState(
        avx,
        avy,
        _matrix(shape, lambda j, i: adjoint_out.sxx[j][i] + sxx_t[j][i]),
        _matrix(shape, lambda j, i: adjoint_out.syy[j][i] + syy_t[j][i]),
        _matrix(shape, lambda j, i: adjoint_out.sxy[j][i] + sxy_xt[j][i] + sxy_yt[j][i]),
    )


def state_inner(a: PSVState, b: PSVState) -> float:
    return sum(matrix_inner(getattr(a, name), getattr(b, name)) for name in PSVState.__annotations__)


def harmonic_mu_jvp(mu: Matrix, delta_mu: Matrix) -> Matrix:
    shape = _shape(mu)
    ny, nx = shape
    mu_xy = staggered_shear_modulus(mu)
    return _matrix(
        shape,
        lambda j, i: sum(
            mu_xy[j][i] ** 2 * delta_mu[(j + dy) % ny][(i + dx) % nx]
            / (4.0 * mu[(j + dy) % ny][(i + dx) % nx] ** 2)
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))
        ),
    )


def harmonic_mu_vjp(mu: Matrix, adj_mu_xy: Matrix) -> Matrix:
    shape = _shape(mu)
    ny, nx = shape
    mu_xy = staggered_shear_modulus(mu)
    return _matrix(
        shape,
        lambda j, i: sum(
            adj_mu_xy[(j - dy) % ny][(i - dx) % nx]
            * mu_xy[(j - dy) % ny][(i - dx) % nx] ** 2
            / (4.0 * mu[j][i] ** 2)
            for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1))
        ),
    )


def material_jvp(vx: Matrix, vy: Matrix, delta_lam: Matrix, delta_mu: Matrix, *, mu: Matrix, dt_over_dh: float) -> tuple[Matrix, Matrix, Matrix]:
    shape = _shape(vx)
    exx, eyy, gamma = strain(vx, vy)
    delta_mu_xy = harmonic_mu_jvp(mu, delta_mu)
    return (
        _matrix(shape, lambda j, i: dt_over_dh * (delta_lam[j][i] * (exx[j][i] + eyy[j][i]) + 2.0 * delta_mu[j][i] * exx[j][i])),
        _matrix(shape, lambda j, i: dt_over_dh * (delta_lam[j][i] * (exx[j][i] + eyy[j][i]) + 2.0 * delta_mu[j][i] * eyy[j][i])),
        _matrix(shape, lambda j, i: dt_over_dh * delta_mu_xy[j][i] * gamma[j][i]),
    )


def material_vjp(vx: Matrix, vy: Matrix, adj_sxx: Matrix, adj_syy: Matrix, adj_sxy: Matrix, *, mu: Matrix, dt_over_dh: float) -> tuple[Matrix, Matrix]:
    shape = _shape(vx)
    exx, eyy, gamma = strain(vx, vy)
    g_lam = _matrix(shape, lambda j, i: dt_over_dh * (exx[j][i] + eyy[j][i]) * (adj_sxx[j][i] + adj_syy[j][i]))
    local_mu = _matrix(shape, lambda j, i: dt_over_dh * 2.0 * (exx[j][i] * adj_sxx[j][i] + eyy[j][i] * adj_syy[j][i]))
    staggered = harmonic_mu_vjp(mu, _matrix(shape, lambda j, i: dt_over_dh * gamma[j][i] * adj_sxy[j][i]))
    return g_lam, _matrix(shape, lambda j, i: local_mu[j][i] + staggered[j][i])


def density_average_jvp(rho: Matrix, delta_rho: Matrix) -> tuple[Matrix, Matrix]:
    shape = _shape(rho)
    ny, nx = shape
    rx, ry = staggered_inverse_density(rho)
    return (
        _matrix(shape, lambda j, i: -0.5 * rx[j][i] ** 2 * (delta_rho[j][i] + delta_rho[j][(i + 1) % nx])),
        _matrix(shape, lambda j, i: -0.5 * ry[j][i] ** 2 * (delta_rho[j][i] + delta_rho[(j + 1) % ny][i])),
    )


def density_average_vjp(rho: Matrix, adj_rx: Matrix, adj_ry: Matrix) -> Matrix:
    shape = _shape(rho)
    ny, nx = shape
    rx, ry = staggered_inverse_density(rho)
    edge_x = _matrix(shape, lambda j, i: -0.5 * rx[j][i] ** 2 * adj_rx[j][i])
    edge_y = _matrix(shape, lambda j, i: -0.5 * ry[j][i] ** 2 * adj_ry[j][i])
    return _matrix(shape, lambda j, i: edge_x[j][i] + edge_x[j][(i - 1) % nx] + edge_y[j][i] + edge_y[(j - 1) % ny][i])


def density_mass_jvp(state: PSVState, rho: Matrix, delta_rho: Matrix, *, dt_over_dh: float) -> tuple[Matrix, Matrix]:
    shape = _shape(rho)
    delta_rx, delta_ry = density_average_jvp(rho, delta_rho)
    div_x, div_y = stress_divergence(state)
    return (
        _matrix(shape, lambda j, i: dt_over_dh * delta_rx[j][i] * div_x[j][i]),
        _matrix(shape, lambda j, i: dt_over_dh * delta_ry[j][i] * div_y[j][i]),
    )


def density_mass_vjp(state: PSVState, rho: Matrix, adj_vx: Matrix, adj_vy: Matrix, *, dt_over_dh: float) -> Matrix:
    shape = _shape(rho)
    div_x, div_y = stress_divergence(state)
    return density_average_vjp(
        rho,
        _matrix(shape, lambda j, i: dt_over_dh * div_x[j][i] * adj_vx[j][i]),
        _matrix(shape, lambda j, i: dt_over_dh * div_y[j][i] * adj_vy[j][i]),
    )


def physical_parameter_chain(g_lam: Matrix, g_mu: Matrix, g_rho_mass: Matrix, *, vp: Matrix, vs: Matrix, rho: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    shape = _shape(vp)
    return (
        _matrix(shape, lambda j, i: 2.0 * rho[j][i] * vp[j][i] * g_lam[j][i]),
        _matrix(shape, lambda j, i: -4.0 * rho[j][i] * vs[j][i] * g_lam[j][i] + 2.0 * rho[j][i] * vs[j][i] * g_mu[j][i]),
        _matrix(shape, lambda j, i: (vp[j][i] ** 2 - 2.0 * vs[j][i] ** 2) * g_lam[j][i] + vs[j][i] ** 2 * g_mu[j][i] + g_rho_mass[j][i]),
    )


def add_scaled(a: Matrix, b: Matrix, scale: float) -> Matrix:
    shape = _shape(a)
    return _matrix(shape, lambda j, i: a[j][i] + scale * b[j][i])


def subtract_scaled(a: Matrix, b: Matrix, scale: float) -> Matrix:
    shape = _shape(a)
    return _matrix(shape, lambda j, i: (a[j][i] - b[j][i]) / scale)


def matrix_norm(a: Matrix) -> float:
    return matrix_inner(a, a) ** 0.5
