"""Independent M6.3c-6a local SH material-map reference."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct


C6A_DOUBLE_DOT_RELATIVE_MAX = 5.0e-12
C6A_REFERENCE_RELATIVE_MAX = 5.0e-12
C6A_FD_RELATIVE_MAX = 5.0e-6


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


@dataclass(frozen=True)
class QMapping:
    mode: int
    sample_count: int = 0
    a: float = 0.0
    b: float = 0.0


def physical_mapping(frequencies, fmin, fmax, df):
    frequencies = tuple(f32(value) for value in frequencies)
    fmin, fmax, df = f32(fmin), f32(fmax), f32(df)
    count = int(math.floor((fmax - fmin) / df + 1.0e-12)) + 1
    sum_a = sum_ab = sum_aa = 0.0
    for index in range(count):
        omega = 2.0 * math.pi * (fmin + index * df)
        aa = bb = 0.0
        for frequency in frequencies:
            theta = 1.0 / (2.0 * math.pi * frequency)
            product = omega * theta
            divisor = 1.0 + product * product
            aa += product * product / divisor
            bb += product / divisor
        avalue = 1.0 / bb
        bvalue = aa / bb
        sum_a += avalue
        sum_ab += avalue * bvalue
        sum_aa += avalue * avalue
    return QMapping(1, count, sum_a / sum_aa, -sum_ab / sum_aa)


def q_to_tau(q, mapping):
    if mapping.mode == 0:
        return 2.0 / q
    inverse = mapping.a * q + mapping.b
    return 1.0 / inverse


def q_to_tau_derivative(q, mapping):
    tau = q_to_tau(q, mapping)
    return -2.0 / q**2 if mapping.mode == 0 else -mapping.a * tau**2


def harmonic(left, right):
    if left <= 0.0 or right <= 0.0:
        raise ValueError("harmonic VJP requires positive moduli")
    return 2.0 * left * right / (left + right)


def harmonic_jvp(left, right, dleft, dright):
    divisor = (left + right) ** 2
    return 2.0 * right**2 / divisor * dleft + 2.0 * left**2 / divisor * dright


def harmonic_vjp(left, right, bar):
    divisor = (left + right) ** 2
    return 2.0 * right**2 / divisor * bar, 2.0 * left**2 / divisor * bar


def rhoi(rho):
    return 0.0 if rho < 1.0e-4 else 1.0 / rho


def rhoi_derivative(rho):
    return 0.0 if rho < 1.0e-4 else -1.0 / rho**2


def forward(invmat1, mapping, primary, rho_values, q_values):
    if invmat1 == 1:
        mu = [rho * vs**2 for rho, vs in zip(rho_values, primary)]
    elif invmat1 == 3:
        mu = list(primary)
    else:
        raise ValueError("C6a supports INVMAT1 1 and 3 only")
    tau = [q_to_tau(q, mapping) for q in q_values]
    return [
        harmonic(mu[0], mu[1]),
        harmonic(mu[0], mu[2]),
        0.25 * math.fsum(tau),
        tau[0],
        rhoi(rho_values[0]),
    ]


def jvp(invmat1, mapping, primary, rho_values, q_values,
        dprimary, drho, dq):
    if invmat1 == 1:
        mu = [rho * vs**2 for rho, vs in zip(rho_values, primary)]
        dmu = [
            vs**2 * rd + 2.0 * rho * vs * vd
            for rho, vs, rd, vd in zip(rho_values, primary, drho, dprimary)
        ]
    else:
        mu, dmu = list(primary), list(dprimary)
    dtau = [q_to_tau_derivative(q, mapping) * value
            for q, value in zip(q_values, dq)]
    return [
        harmonic_jvp(mu[0], mu[1], dmu[0], dmu[1]),
        harmonic_jvp(mu[0], mu[2], dmu[0], dmu[2]),
        0.25 * math.fsum(dtau),
        dtau[0],
        rhoi_derivative(rho_values[0]) * drho[0],
    ]


def vjp(invmat1, mapping, primary, rho_values, q_values, bar_output):
    if invmat1 == 1:
        mu = [rho * vs**2 for rho, vs in zip(rho_values, primary)]
    else:
        mu = list(primary)
    bar_mu = [0.0] * 4
    left, right = harmonic_vjp(mu[0], mu[1], bar_output[0])
    bar_mu[0] += left
    bar_mu[1] += right
    left, right = harmonic_vjp(mu[0], mu[2], bar_output[1])
    bar_mu[0] += left
    bar_mu[2] += right
    bar_tau = [0.25 * bar_output[2]] * 4
    bar_tau[0] += bar_output[3]
    bar_q = [q_to_tau_derivative(q, mapping) * bar
             for q, bar in zip(q_values, bar_tau)]
    bar_rho = [0.0] * 4
    if invmat1 == 1:
        bar_primary = [2.0 * rho * vs * bar
                       for rho, vs, bar in zip(rho_values, primary, bar_mu)]
        bar_rho = [vs**2 * bar for vs, bar in zip(primary, bar_mu)]
    else:
        bar_primary = bar_mu
    bar_rho[0] += rhoi_derivative(rho_values[0]) * bar_output[4]
    return bar_primary, bar_rho, bar_q


def dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def relative(left, right):
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)
