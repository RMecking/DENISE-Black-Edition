"""Independent double-precision reference for SH-PRECOND-01.

Derived solely from ``docs/sh_precond_01_contract.md`` (section 2).  No
production source is consulted or imported.  Fields are nested lists indexed
``[j][i]`` with 0-based row ``j`` (y) and column ``i`` (x); the contract's
local 1-based cell ``(i, j)`` corresponds to ``[j - 1][i - 1]``.

The SH-OPT-SCALE-01 class statistic is re-derived here (not imported) so that
this reference depends on the two frozen contracts only.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from tests.utilities.sh_opt_scale_01_reference import f32

CLASSES = ("primary", "rho", "q")
Field = Sequence[Sequence[float]]


# --------------------------------------------------------------------------
# 2.1 energy accumulation
# --------------------------------------------------------------------------

def energy_exact(snapshots: Sequence[Field]) -> list[list[float]]:
    """Exact (correctly rounded double) sum of squared binary32 inputs."""
    ny, nx = len(snapshots[0]), len(snapshots[0][0])
    return [[math.fsum(float(s[j][i]) * float(s[j][i]) for s in snapshots)
             for i in range(nx)] for j in range(ny)]


def energy_bound(steps: int, exact: float) -> float:
    """Recursive-summation bound of contract 2.1: (n + 1) * 2^-24 * E_exact."""
    return (steps + 1) * 2.0 ** -24 * exact


# --------------------------------------------------------------------------
# 2.2 receiver-line term
# --------------------------------------------------------------------------

def receiver_term_quadrature(x: float, z: float, x_min: float, x_max: float,
                             intervals: int = 4096) -> float:
    """Composite Simpson quadrature of int dx_r / sqrt((x - x_r)^2 + z^2).

    Independent of the closed asinh form.  The integrand is analytic for
    z >= 1, so the panels are refined only near the integrand peak x_r = x.
    """
    def integrand(xr: float) -> float:
        return 1.0 / math.sqrt((x - xr) ** 2 + z * z)

    # Split at x when it lies inside the interval (peak of the integrand).
    cuts = [x_min, x_max]
    if x_min < x < x_max:
        cuts = [x_min, x, x_max]
    total = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        h = (b - a) / intervals
        acc = [integrand(a), integrand(b)]
        for k in range(1, intervals):
            acc.append((4.0 if k % 2 else 2.0) * integrand(a + k * h))
        total.append(math.fsum(acc) * h / 3.0)
    return math.fsum(total)


def receiver_term_status(term_is_null: bool, nx: int, ny: int, pos_x: int,
                         pos_y: int, x_min: float, x_max: float) -> int:
    if term_is_null or nx < 1 or ny < 1 or pos_x < 0 or pos_y < 0:
        return -1
    if not (math.isfinite(x_min) and math.isfinite(x_max)):
        return -1
    if not x_min < x_max:
        return -1
    return 0


def receiver_term_local(nx: int, ny: int, pos_x: int, pos_y: int,
                        x_min: float, x_max: float) -> list[list[float]]:
    """Reference term from quadrature, rounded to binary32."""
    return [[f32(receiver_term_quadrature(i + pos_x * nx, j + pos_y * ny,
                                          x_min, x_max))
             for i in range(1, nx + 1)] for j in range(1, ny + 1)]


# --------------------------------------------------------------------------
# 2.3 preconditioner
# --------------------------------------------------------------------------

def illumination_weight(mode: int, source_energy: float, receiver: float) -> float:
    if mode == 1:
        return float(source_energy)
    if mode == 2:
        return math.sqrt(float(source_energy)) * float(receiver)
    raise ValueError("mode must be 1 or 2")


def precond_status(mode: int, epsilon: float, source_energy: Field,
                   receiver: Field | None, mask: Field) -> int:
    """Data-dependent codes -12..-14 of contract 2.3 (structural -11 aside)."""
    if mode not in (1, 2) or not math.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        return -12
    count, w_max = 0, 0.0
    for j, mask_row in enumerate(mask):
        for i, m in enumerate(mask_row):
            if m not in (0.0, 1.0):
                return -13
            if m != 1.0:
                continue
            ws = float(source_energy[j][i])
            if not math.isfinite(ws) or ws < 0.0:
                return -13
            r = 0.0
            if mode == 2:
                r = float(receiver[j][i])
                if not math.isfinite(r) or r <= 0.0:
                    return -13
    for j, mask_row in enumerate(mask):
        for i, m in enumerate(mask_row):
            if m == 1.0:
                count += 1
                r = float(receiver[j][i]) if mode == 2 else 0.0
                w_max = max(w_max, illumination_weight(mode, source_energy[j][i], r))
    if count == 0 or not w_max > 0.0:
        return -14
    bound = 1.0 / (float(f32(epsilon)) * w_max)
    if not math.isfinite(bound) or abs(bound) > 3.4028234663852886e38:
        return -14
    return 0


def precond(mode: int, epsilon: float, source_energy: Field,
            receiver: Field | None, mask: Field) -> list[list[float]]:
    """P = 1 / (W + eps * W_max) on masked cells, exactly 0 elsewhere."""
    weights = [[illumination_weight(mode, source_energy[j][i],
                                    receiver[j][i] if mode == 2 else 0.0)
                if mask[j][i] == 1.0 else None
                for i in range(len(mask[0]))] for j in range(len(mask))]
    w_max = max(w for row in weights for w in row if w is not None)
    eps = float(f32(epsilon))
    return [[0.0 if w is None else f32(1.0 / (w + eps * w_max)) for w in row]
            for row in weights]


# --------------------------------------------------------------------------
# 2.4 preconditioned scaled direction (exact q = P * g in double)
# --------------------------------------------------------------------------

def _class_scale(mode: int, q: Field, base: Field, mask: Field) -> float:
    masked_q, masked_m = [], []
    for q_row, m_row, mask_row in zip(q, base, mask):
        for q_value, m_value, mask_value in zip(q_row, m_row, mask_row):
            if mask_value == 1.0:
                masked_q.append(float(q_value))
                masked_m.append(float(m_value))
    n = len(masked_q)
    if n == 0:
        raise ValueError("no masked cell")
    if mode == 1:
        norm = math.sqrt(math.fsum(v * v for v in masked_q) / n)
        reference = math.sqrt(math.fsum(v * v for v in masked_m) / n)
    elif mode == 2:
        norm = max(abs(v) for v in masked_q)
        reference = max(abs(v) for v in masked_m)
    else:
        raise ValueError("scaling mode must be 1 or 2")
    return reference / norm if norm > 0.0 else 0.0


def preconditioned_step(mode: int, weights: Mapping[str, float],
                        gradients: Mapping[str, Field], bases: Mapping[str, Field],
                        mask: Field, p_field: Field) -> dict[str, list[list[float]]]:
    """p_k = M * w_k * s_k * (P g_k) with s_k from the masked statistic of P g_k."""
    step: dict[str, list[list[float]]] = {}
    for name in CLASSES:
        q = [[float(p_field[j][i]) * float(gradients[name][j][i])
              for i in range(len(mask[0]))] for j in range(len(mask))]
        scale = _class_scale(mode, q, bases[name], mask)
        weight = float(weights[name])
        step[name] = [[0.0 if (mask[j][i] != 1.0 or weight == 0.0)
                       else f32(weight * scale * q[j][i])
                       for i in range(len(mask[0]))] for j in range(len(mask))]
    return step


def descent_dot(gradients: Mapping[str, Field], step: Mapping[str, Field]) -> float:
    return math.fsum(float(g) * float(p)
                     for name in CLASSES
                     for g_row, p_row in zip(gradients[name], step[name])
                     for g, p in zip(g_row, p_row))


# --------------------------------------------------------------------------
# 2.5 initial alpha
# --------------------------------------------------------------------------

def initial_alpha(eps_scale: float, adaptive: int, factor: float, iteration: int,
                  previous_alpha: float) -> float:
    eps_scale = f32(eps_scale)
    if adaptive == 0 or iteration <= 1:
        return eps_scale
    if not math.isfinite(previous_alpha) or previous_alpha <= 0.0:
        return eps_scale
    candidate = f32(float(f32(factor)) * float(f32(previous_alpha)))
    return candidate if candidate < eps_scale else eps_scale
