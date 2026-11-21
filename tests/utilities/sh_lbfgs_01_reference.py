"""Independent double-precision reference for SH-LBFGS-01.

Derived solely from ``docs/sh_lbfgs_01_contract.md`` (sections 2 and 3).  No
production source is consulted or imported.  Fields are nested lists indexed
``[j][i]`` with 0-based row ``j`` (y) and column ``i`` (x) over the *global*
grid; the contract's sums over the entry set E are global, so a reference over
the global grid is equivalent to any decomposition up to summation order.

Two independent algorithms are provided:

* ``LbfgsReference`` follows the two-loop recursion of contract 3.3;
* ``dense_direction`` builds the inverse-Hessian approximation H explicitly
  as a dense matrix by the BFGS recursion and returns ``H g``.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from tests.utilities.sh_opt_scale_01_reference import f32

CLASSES = ("primary", "rho", "q")
CURVATURE_THRESHOLD = 1.0e-8
DESCENT_THRESHOLD = 1.0e-8
MAX_MEMORY = 20

ACTION_NAMES = ("sd_empty", "sd_reset", "sd_fallback", "lbfgs")
PAIR_NAMES = ("none", "stored", "skipped")

Field = Sequence[Sequence[float]]


def _fsum(values) -> float:
    return math.fsum(values)


def sigma(bases: Sequence[Field], mask: Field) -> list[float]:
    """Contract 3.2: masked RMS of each class base model (double)."""
    cells = [(j, i) for j, row in enumerate(mask) for i, v in enumerate(row) if v == 1.0]
    n = len(cells)
    return [math.sqrt(_fsum(float(b[j][i]) ** 2 for j, i in cells) / n) for b in bases]


class LbfgsReference:
    """State machine of contract 3.3 over the global grid."""

    def __init__(self, memory: int) -> None:
        assert 1 <= memory <= MAX_MEMORY
        self.memory = memory
        self.sigma: Optional[list[float]] = None
        self.pairs: list[tuple[list[float], list[float]]] = []
        self.previous = None

    # -- 3.2 ---------------------------------------------------------------
    def begin_stage(self, bases: Sequence[Field], mask: Field) -> list[float]:
        self.sigma = sigma(bases, mask)
        self.pairs = []
        self.previous = None
        return list(self.sigma)

    # -- 3.3 ---------------------------------------------------------------
    def step(self, bases: Sequence[Field], grads: Sequence[Field], mask: Field,
             precond: Optional[Field], weights: Sequence[float],
             step_in: Sequence[Field]) -> dict:
        assert self.sigma is not None
        sig = self.sigma
        ny, nx = len(mask), len(mask[0])
        active = tuple(w > 0.0 for w in weights)
        assert any(active)
        cells = [(j, i) for j in range(ny) for i in range(nx) if mask[j][i] == 1.0]
        entries = [(c, j, i) for c in range(3) if active[c] for (j, i) in cells]
        n_m, n_a = len(cells), sum(active)

        def pval(j, i):
            return 1.0 if precond is None else float(precond[j][i])

        reset = False
        pair_code, curvature = 0, 0.0
        if self.previous is not None:
            prev_bases, prev_grads, prev_active = self.previous
            if prev_active != active:
                self.pairs = []
                reset = True
            else:
                s = [f32((float(bases[c][j][i]) - float(prev_bases[c][j][i])) / sig[c])
                     for (c, j, i) in entries]
                y = [f32(sig[c] * (float(grads[c][j][i]) - float(prev_grads[c][j][i])))
                     for (c, j, i) in entries]
                ss = _fsum(v * v for v in s)
                yy = _fsum(v * v for v in y)
                sy = _fsum(a * b for a, b in zip(s, y))
                if ss > 0.0 and yy > 0.0:
                    value = sy / math.sqrt(ss * yy)
                    curvature = value if math.isfinite(value) else 0.0
                if (ss > 0.0 and yy > 0.0 and math.isfinite(sy)
                        and sy > CURVATURE_THRESHOLD * math.sqrt(ss * yy)):
                    if len(self.pairs) == self.memory:
                        self.pairs.pop(0)
                    self.pairs.append((s, y))
                    pair_code = 1
                else:
                    pair_code = 2

        out = {"action": None, "pair": pair_code, "pairs_used": 0,
               "curvature": curvature, "gamma": 0.0, "descent": 0.0, "kappa": 0.0,
               "sigma": list(sig), "step": None}
        if not self.pairs:
            out["action"] = 1 if reset else 0
        else:
            gx = [sig[c] * float(grads[c][j][i]) for (c, j, i) in entries]
            dd = [float(weights[c]) * pval(j, i) for (c, j, i) in entries]
            d, gamma = two_loop(self.pairs, dd, gx)
            gd = _fsum(a * b for a, b in zip(gx, d))
            gdg = _fsum(w * a * a for w, a in zip(dd, gx))
            ddd = _fsum(a * a / w for w, a in zip(dd, d))
            out["gamma"] = gamma
            out["pairs_used"] = len(self.pairs)
            if gdg > 0.0 and ddd > 0.0:
                value = gd / math.sqrt(gdg * ddd)
                out["descent"] = value if math.isfinite(value) else 0.0
            if (math.isfinite(gd) and gdg > 0.0 and ddd > 0.0
                    and gd > DESCENT_THRESHOLD * math.sqrt(gdg * ddd)):
                kappa = 1.0 / math.sqrt(_fsum(v * v for v in d) / (n_a * n_m))
                out["kappa"] = kappa
                out["action"] = 3
                step = [[[0.0] * nx for _ in range(ny)] for _ in range(3)]
                for (c, j, i), v in zip(entries, d):
                    step[c][j][i] = f32(sig[c] * kappa * v)
                out["step"] = step
            else:
                out["action"] = 2
                self.pairs = []
        if out["step"] is None:
            out["step"] = [[list(row) for row in step_in[c]] for c in range(3)]
        self.previous = ([[list(r) for r in bases[c]] for c in range(3)],
                         [[list(r) for r in grads[c]] for c in range(3)], active)
        out["entries"] = entries
        return out


def two_loop(pairs, dd: Sequence[float], gx: Sequence[float]):
    """Contract 3.3 step 4: two-loop recursion; returns (d, gamma)."""
    rho = [1.0 / _fsum(a * b for a, b in zip(s, y)) for s, y in pairs]
    q = list(gx)
    alphas = [0.0] * len(pairs)
    for k in range(len(pairs) - 1, -1, -1):
        s, y = pairs[k]
        alphas[k] = rho[k] * _fsum(a * b for a, b in zip(s, q))
        q = [qv - alphas[k] * yv for qv, yv in zip(q, y)]
    s_n, y_n = pairs[-1]
    gamma = _fsum(a * b for a, b in zip(s_n, y_n)) / _fsum(w * v * v for w, v in zip(dd, y_n))
    r = [gamma * w * qv for w, qv in zip(dd, q)]
    for k in range(len(pairs)):
        s, y = pairs[k]
        beta = rho[k] * _fsum(a * b for a, b in zip(y, r))
        r = [rv + (alphas[k] - beta) * sv for rv, sv in zip(r, s)]
    return r, gamma


def dense_direction(pairs, dd: Sequence[float], gx: Sequence[float]) -> list[float]:
    """Independent algorithm: explicit dense H from gamma*D by the BFGS update.

    H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T, applied in O(n^2) form
    H+ = H - rho (s (H y)^T + (H y) s^T) + (rho^2 y^T H y + rho) s s^T.
    """
    n = len(gx)
    s_n, y_n = pairs[-1]
    gamma = math.fsum(a * b for a, b in zip(s_n, y_n)) / math.fsum(
        w * v * v for w, v in zip(dd, y_n))
    h = [[0.0] * n for _ in range(n)]
    for k in range(n):
        h[k][k] = gamma * dd[k]
    for s, y in pairs:
        rho = 1.0 / math.fsum(a * b for a, b in zip(s, y))
        hy = [math.fsum(h[r][c] * y[c] for c in range(n)) for r in range(n)]
        yhy = math.fsum(a * b for a, b in zip(y, hy))
        coef = rho * rho * yhy + rho
        for r in range(n):
            for c in range(n):
                h[r][c] += -rho * (s[r] * hy[c] + hy[r] * s[c]) + coef * s[r] * s[c]
    return [math.fsum(h[r][c] * gx[c] for c in range(n)) for r in range(n)]


def kappa_of(d: Sequence[float], n_active: int, n_masked: int) -> float:
    return 1.0 / math.sqrt(math.fsum(v * v for v in d) / (n_active * n_masked))


def class_fields_from_entries(entries, values, sig, scale, nx, ny):
    """Scatter E-ordered x-space values into m-space class fields (f32)."""
    step = [[[0.0] * nx for _ in range(ny)] for _ in range(3)]
    for (c, j, i), v in zip(entries, values):
        step[c][j][i] = f32(sig[c] * scale * v)
    return step


def solve_linear(a: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (double), for oracle O6."""
    n = len(b)
    m = [list(map(float, row)) + [float(v)] for row, v in zip(a, b)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - math.fsum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x
