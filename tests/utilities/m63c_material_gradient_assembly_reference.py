"""Independent C7c-a temporal reduction and distributed-map oracle."""

from __future__ import annotations

import math

from tests.utilities.m63c_distributed_material_map_reference import (
    material_vjp,
    matcopy_transpose,
    zeros,
)
from tests.utilities.m63c_material_map_reference import q_to_tau_derivative


C7CA_TEMPORAL_RELATIVE_MAX = 2.0e-13
C7CA_MPI_REFERENCE_RELATIVE_MAX = 7.0e-6
C7CA_LINEARITY_RELATIVE_MAX = 2.0e-12


def c6_channel_order(native):
    """C7b rhoi/mu_x/mu_y/tau_x/tau_y -> locked C6 output order."""
    return [
        [rank[1], rank[2], rank[3], rank[4], rank[0]]
        for rank in native
    ]


def temporal_accumulate(series, dt, dtinv):
    """Reduce time-major [step][rank][channel][point] contributions."""
    weight = dt * dtinv
    nsteps, ranks, channels, points = (
        len(series), len(series[0]), len(series[0][0]), len(series[0][0][0])
    )
    return [
        [
            [
                weight * math.fsum(series[n][rank][channel][point]
                                   for n in range(nsteps))
                for point in range(points)
            ]
            for channel in range(channels)
        ]
        for rank in range(ranks)
    ]


def distributed_gradient(invmat1, mapping, primary, rho_values, q_values,
                         native, layout, npx, npy):
    return material_vjp(
        invmat1, mapping, primary, rho_values, q_values,
        c6_channel_order(native),
        layout, npx, npy,
    )


def sum_mapped_per_step(invmat1, mapping, primary, rho_values, q_values,
                        series, dt, dtinv, layout, npx, npy):
    weight = dt * dtinv
    mapped = [
        material_vjp(
            invmat1, mapping, primary, rho_values, q_values,
            c6_channel_order(step),
            layout, npx, npy,
        )
        for step in series
    ]
    ranks = npx * npy
    output = [zeros(ranks, layout) for _ in range(3)]
    for field in range(3):
        for rank in range(ranks):
            for cell in range(layout.cells):
                output[field][rank][cell] = weight * math.fsum(
                    value[field][rank][cell] for value in mapped
                )
    return tuple(output)


def wrong_q_chain_before_matcopy(mapping, q_values, native, layout, npx, npy):
    """Wrong: apply receiving-rank local Q derivatives before matcopy^T."""
    native = c6_channel_order(native)
    ranks = npx * npy
    tau_owned = zeros(ranks, layout)
    post_bar_tau = zeros(ranks, layout)
    for rank in range(ranks):
        for j in range(1, layout.ny + 1):
            for i in range(1, layout.nx + 1):
                o = layout.owned_index(j, i)
                c = layout.index(j, i)
                e = layout.index(j, i + 1)
                s = layout.index(j + 1, i)
                se = layout.index(j + 1, i + 1)
                for cell in (c, e, s, se):
                    post_bar_tau[rank][cell] += 0.25 * native[rank][2][o]
                post_bar_tau[rank][c] += native[rank][3][o]
        for cell in range(layout.cells):
            post_bar_tau[rank][cell] *= q_to_tau_derivative(
                q_values[rank][cell], mapping
            )
    returned = matcopy_transpose(post_bar_tau, layout, npx, npy)
    for rank in range(ranks):
        for j in range(1, layout.ny + 1):
            for i in range(1, layout.nx + 1):
                cell = layout.index(j, i)
                tau_owned[rank][cell] = returned[rank][cell]
    return tau_owned
