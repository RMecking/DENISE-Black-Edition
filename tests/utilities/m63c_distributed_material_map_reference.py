"""Independent distributed SH material-preparation map and exact transpose."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from tests.utilities.m63c_material_map_reference import (
    QMapping,
    harmonic,
    harmonic_jvp,
    harmonic_vjp,
    q_to_tau,
    q_to_tau_derivative,
    rhoi,
    rhoi_derivative,
)


C6B_MPI_DOT_RELATIVE_MAX = 5.0e-6
C6B_REFERENCE_RELATIVE_MAX = 5.0e-6
C6B_DOUBLE_DOT_RELATIVE_MAX = 5.0e-12
C6B_FD_RELATIVE_MAX = 5.0e-6


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


@dataclass(frozen=True)
class Layout:
    nx: int = 4
    ny: int = 3

    @property
    def cells(self):
        return (self.nx + 2) * (self.ny + 2)

    @property
    def owned(self):
        return self.nx * self.ny

    def index(self, j, i):
        return j * (self.nx + 2) + i

    def owned_index(self, j, i):
        return (j - 1) * self.nx + i - 1


def neighbours(rank, npx, npy):
    x, y = rank % npx, rank // npx
    left = y * npx + (x - 1) % npx
    right = y * npx + (x + 1) % npx
    top = ((y - 1) % npy) * npx + x
    bottom = ((y + 1) % npy) * npx + x
    return left, right, top, bottom


def zeros(ranks, layout):
    return [[0.0] * layout.cells for _ in range(ranks)]


def matcopy_forward(values, layout, npx, npy):
    """Actual V then H cyclic overwrite graph, including self neighbours."""
    ranks = npx * npy
    vertical = [list(field) for field in values]
    for rank in range(ranks):
        _, _, top, bottom = neighbours(rank, npx, npy)
        for i in range(1, layout.nx + 1):
            vertical[rank][layout.index(layout.ny + 1, i)] = values[bottom][layout.index(1, i)]
            vertical[rank][layout.index(0, i)] = values[top][layout.index(layout.ny, i)]
        # matcopy_SH's uncommunicated vertical buffer endpoints are local;
        # H overwrites them subsequently, but retaining the stage is exact.
        vertical[rank][layout.index(layout.ny + 1, 0)] = values[rank][layout.index(1, 0)]
        vertical[rank][layout.index(layout.ny + 1, layout.nx + 1)] = values[rank][layout.index(1, layout.nx + 1)]
        vertical[rank][layout.index(0, 0)] = values[rank][layout.index(layout.ny, 0)]
        vertical[rank][layout.index(0, layout.nx + 1)] = values[rank][layout.index(layout.ny, layout.nx + 1)]
    output = [list(field) for field in vertical]
    for rank in range(ranks):
        left, right, _, _ = neighbours(rank, npx, npy)
        for j in range(layout.ny + 2):
            output[rank][layout.index(j, layout.nx + 1)] = vertical[right][layout.index(j, 1)]
            output[rank][layout.index(j, 0)] = vertical[left][layout.index(j, layout.nx)]
    return output


def matcopy_transpose(bars, layout, npx, npy):
    """Exact V^T H^T; output-halo cotangents are consumed."""
    ranks = npx * npy
    horizontal = [list(field) for field in bars]
    for rank in range(ranks):
        left, right, _, _ = neighbours(rank, npx, npy)
        for j in range(layout.ny + 2):
            horizontal[right][layout.index(j, 1)] += bars[rank][layout.index(j, layout.nx + 1)]
            horizontal[left][layout.index(j, layout.nx)] += bars[rank][layout.index(j, 0)]
            horizontal[rank][layout.index(j, 0)] = 0.0
            horizontal[rank][layout.index(j, layout.nx + 1)] = 0.0
    output = [list(field) for field in horizontal]
    for rank in range(ranks):
        _, _, top, bottom = neighbours(rank, npx, npy)
        for i in range(1, layout.nx + 1):
            output[bottom][layout.index(1, i)] += horizontal[rank][layout.index(layout.ny + 1, i)]
            output[top][layout.index(layout.ny, i)] += horizontal[rank][layout.index(0, i)]
            output[rank][layout.index(0, i)] = 0.0
            output[rank][layout.index(layout.ny + 1, i)] = 0.0
    return output


def dot_fields(left, right):
    return math.fsum(a * b for lf, rf in zip(left, right) for a, b in zip(lf, rf))


def relative(left, right):
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def material_forward(invmat1, mapping, primary, rho_values, q_values,
                     layout, npx, npy):
    ranks = npx * npy
    tau = zeros(ranks, layout)
    for rank in range(ranks):
        for j in range(1, layout.ny + 1):
            for i in range(1, layout.nx + 1):
                k = layout.index(j, i)
                tau[rank][k] = q_to_tau(q_values[rank][k], mapping)
    post_primary = matcopy_forward(primary, layout, npx, npy)
    post_rho = matcopy_forward(rho_values, layout, npx, npy)
    post_tau = matcopy_forward(tau, layout, npx, npy)
    output = [[[0.0] * layout.owned for _ in range(5)] for _ in range(ranks)]
    for rank in range(ranks):
        mu = [post_rho[rank][k] * value * value for k, value in enumerate(post_primary[rank])] if invmat1 == 1 else post_primary[rank]
        for j in range(1, layout.ny + 1):
            for i in range(1, layout.nx + 1):
                o, c = layout.owned_index(j, i), layout.index(j, i)
                e, s, se = layout.index(j, i + 1), layout.index(j + 1, i), layout.index(j + 1, i + 1)
                output[rank][0][o] = harmonic(mu[c], mu[e])
                output[rank][1][o] = harmonic(mu[c], mu[s])
                output[rank][2][o] = 0.25 * (post_tau[rank][c] + post_tau[rank][e] + post_tau[rank][s] + post_tau[rank][se])
                output[rank][3][o] = post_tau[rank][c]
                output[rank][4][o] = rhoi(post_rho[rank][c])
    return output


def material_jvp(invmat1, mapping, primary, rho_values, q_values,
                 dprimary, drho, dq, layout, npx, npy):
    ranks = npx * npy
    tau, dtau = zeros(ranks, layout), zeros(ranks, layout)
    for rank in range(ranks):
        for j in range(1, layout.ny + 1):
            for i in range(1, layout.nx + 1):
                k = layout.index(j, i)
                tau[rank][k] = q_to_tau(q_values[rank][k], mapping)
                dtau[rank][k] = q_to_tau_derivative(q_values[rank][k], mapping) * dq[rank][k]
    p = matcopy_forward(primary, layout, npx, npy)
    r = matcopy_forward(rho_values, layout, npx, npy)
    t = matcopy_forward(tau, layout, npx, npy)
    dp = matcopy_forward(dprimary, layout, npx, npy)
    dr = matcopy_forward(drho, layout, npx, npy)
    dt = matcopy_forward(dtau, layout, npx, npy)
    output = [[[0.0] * layout.owned for _ in range(5)] for _ in range(ranks)]
    for rank in range(ranks):
        if invmat1 == 1:
            mu = [rv * pv * pv for pv, rv in zip(p[rank], r[rank])]
            dmu = [pv * pv * drv + 2.0 * rv * pv * dpv for pv, rv, dpv, drv in zip(p[rank], r[rank], dp[rank], dr[rank])]
        else:
            mu, dmu = p[rank], dp[rank]
        for j in range(1, layout.ny + 1):
            for i in range(1, layout.nx + 1):
                o, c = layout.owned_index(j, i), layout.index(j, i)
                e, s, se = layout.index(j, i + 1), layout.index(j + 1, i), layout.index(j + 1, i + 1)
                output[rank][0][o] = harmonic_jvp(mu[c], mu[e], dmu[c], dmu[e])
                output[rank][1][o] = harmonic_jvp(mu[c], mu[s], dmu[c], dmu[s])
                output[rank][2][o] = 0.25 * (dt[rank][c] + dt[rank][e] + dt[rank][s] + dt[rank][se])
                output[rank][3][o] = dt[rank][c]
                output[rank][4][o] = rhoi_derivative(r[rank][c]) * dr[rank][c]
    return output


def material_vjp(invmat1, mapping, primary, rho_values, q_values, bars,
                 layout, npx, npy):
    ranks = npx * npy
    tau = zeros(ranks, layout)
    for rank in range(ranks):
        for j in range(1, layout.ny + 1):
            for i in range(1, layout.nx + 1):
                k = layout.index(j, i)
                tau[rank][k] = q_to_tau(q_values[rank][k], mapping)
    p = matcopy_forward(primary, layout, npx, npy)
    r = matcopy_forward(rho_values, layout, npx, npy)
    t = matcopy_forward(tau, layout, npx, npy)
    bp, br, bt = zeros(ranks, layout), zeros(ranks, layout), zeros(ranks, layout)
    for rank in range(ranks):
        mu = [rv * pv * pv for pv, rv in zip(p[rank], r[rank])] if invmat1 == 1 else p[rank]
        bm = [0.0] * layout.cells
        for j in range(1, layout.ny + 1):
            for i in range(1, layout.nx + 1):
                o, c = layout.owned_index(j, i), layout.index(j, i)
                e, s, se = layout.index(j, i + 1), layout.index(j + 1, i), layout.index(j + 1, i + 1)
                lc, le = harmonic_vjp(mu[c], mu[e], bars[rank][0][o]); bm[c] += lc; bm[e] += le
                lc, ls = harmonic_vjp(mu[c], mu[s], bars[rank][1][o]); bm[c] += lc; bm[s] += ls
                for k in (c, e, s, se): bt[rank][k] += 0.25 * bars[rank][2][o]
                bt[rank][c] += bars[rank][3][o]
                br[rank][c] += rhoi_derivative(r[rank][c]) * bars[rank][4][o]
        for k in range(layout.cells):
            if invmat1 == 1:
                bp[rank][k] += 2.0 * r[rank][k] * p[rank][k] * bm[k]
                br[rank][k] += p[rank][k] * p[rank][k] * bm[k]
            else:
                bp[rank][k] += bm[k]
    bp = matcopy_transpose(bp, layout, npx, npy)
    br = matcopy_transpose(br, layout, npx, npy)
    bt = matcopy_transpose(bt, layout, npx, npy)
    bq = zeros(ranks, layout)
    for rank in range(ranks):
        for j in range(1, layout.ny + 1):
            for i in range(1, layout.nx + 1):
                k = layout.index(j, i)
                bq[rank][k] = q_to_tau_derivative(q_values[rank][k], mapping) * bt[rank][k]
    return bp, br, bq


def dot_outputs(left, right):
    return math.fsum(a * b for lr, rr in zip(left, right) for lf, rf in zip(lr, rr) for a, b in zip(lf, rf))
