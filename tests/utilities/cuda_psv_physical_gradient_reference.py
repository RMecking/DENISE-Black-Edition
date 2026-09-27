"""Independent M8e-2C4 native-to-physical gradient reference."""

from __future__ import annotations

import math
import struct


NATIVE_FIELDS = ("gf", "gg", "gfc", "gd", "ge", "gdc", "grx", "gry")
PHYSICAL_FIELDS = ("vp", "vs", "rho", "qp", "qs")
MATERIAL_FIELDS = (
    "prho", "ppi", "pu", "ptaus", "ptaup", "puipjp", "ptausipjp",
    "prip", "prjp",
)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def canonical_q_mapping(mode: int) -> tuple[float, float]:
    if mode == 0:
        return 0.0, 0.0
    frequency, fmin, fmax, df = f32(13.0), f32(2.0), f32(24.0), f32(0.5)
    count = math.floor((float(fmax) - float(fmin)) / float(df) + 1.0e-12) + 1
    sum_a = sum_ab = sum_aa = 0.0
    for index in range(count):
        sample = float(fmin) + index * float(df)
        omega = 2.0 * math.pi * sample
        theta = 1.0 / (2.0 * math.pi * float(frequency))
        omega_theta = omega * theta
        divisor = 1.0 + omega_theta * omega_theta
        a_term = omega_theta * omega_theta / divisor
        b_term = omega_theta / divisor
        a = 1.0 / b_term
        b = a_term / b_term
        sum_a += a
        sum_ab += a * b
        sum_aa += a * a
    return sum_a / sum_aa, -sum_ab / sum_aa


def q_derivative(tau: float, mode: int, per_q: float, offset: float) -> float:
    q = 2.0 / tau if mode == 0 else (1.0 / tau - offset) / per_q
    qf = f32(q)
    if mode == 0:
        return -2.0 / (float(qf) * float(qf))
    reconstructed_tau = 1.0 / (per_q * float(qf) + offset)
    return -per_q * reconstructed_tau * reconstructed_tau


def physical_gradient_reference(
    native: dict[str, tuple[float, ...]],
    material: dict[str, tuple[float, ...]],
    nx: int,
    ny: int,
    dt: float,
    eta: float,
    q_mode: int,
    per_q: float,
    offset: float,
) -> dict[str, tuple[float, ...]]:
    output = {field: [0.0] * (nx * ny) for field in PHYSICAL_FIELDS}
    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            q = (j - 1) * nx + i - 1
            rho, vp, vs = material["prho"][q], material["ppi"][q], material["pu"][q]
            ts, tp = material["ptaus"][q], material["ptaup"][q]
            modulus_s, modulus_p = rho * vs * vs, rho * vp * vp
            den_s, den_p = 1.0 + 0.5 * ts, 1.0 + 0.5 * tp
            gm = native["gf"][q] * dt * (1.0 + ts) / den_s + native["gd"][q] * eta * ts / den_s
            gp = native["gg"][q] * dt * (1.0 + tp) / den_p + native["ge"][q] * eta * tp / den_p
            gts = native["gf"][q] * dt * modulus_s * 0.5 / (den_s * den_s) + native["gd"][q] * eta * modulus_s / (den_s * den_s)
            gtp = native["gg"][q] * dt * modulus_p * 0.5 / (den_p * den_p) + native["ge"][q] * eta * modulus_p / (den_p * den_p)
            vp_gradient = gp * 2.0 * rho * vp
            vs_gradient = gm * 2.0 * rho * vs
            rho_gradient = gp * vp * vp + gm * vs * vs
            qs_tau_gradient = gts
            for sj in (j - 1, j):
                for si in (i - 1, i):
                    if si < 1 or sj < 1:
                        continue
                    source = (sj - 1) * nx + si - 1
                    h, corner_tau = material["puipjp"][source], material["ptausipjp"][source]
                    den_c = 1.0 + 0.5 * corner_tau
                    gh = native["gfc"][source] * dt * (1.0 + corner_tau) / den_c + native["gdc"][source] * eta * corner_tau / den_c
                    gt = native["gfc"][source] * dt * h * 0.5 / (den_c * den_c) + native["gdc"][source] * eta * h / (den_c * den_c)
                    weight = gh * h * h / (4.0 * modulus_s * modulus_s)
                    vs_gradient += weight * 2.0 * rho * vs
                    rho_gradient += weight * vs * vs
                    qs_tau_gradient += 0.25 * gt
            rx, ry = material["prip"][q], material["prjp"][q]
            rho_gradient += -0.5 * rx * rx * native["grx"][q]
            rho_gradient += -0.5 * ry * ry * native["gry"][q]
            if i > 1:
                source = q - 1
                value = material["prip"][source]
                rho_gradient += -0.5 * value * value * native["grx"][source]
            if j > 1:
                source = q - nx
                value = material["prjp"][source]
                rho_gradient += -0.5 * value * value * native["gry"][source]
            output["vp"][q] = vp_gradient
            output["vs"][q] = vs_gradient
            output["rho"][q] = rho_gradient
            output["qp"][q] = gtp * q_derivative(tp, q_mode, per_q, offset)
            output["qs"][q] = qs_tau_gradient * q_derivative(ts, q_mode, per_q, offset)
    return {field: tuple(values) for field, values in output.items()}
