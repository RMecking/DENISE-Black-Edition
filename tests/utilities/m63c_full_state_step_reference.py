"""Independent complete one-step viscoelastic SH state map and transpose."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from tests.utilities.m63c_mpi_free_surface_reference import (
    Layout,
    exchange_forward,
    exchange_transpose,
    surface_stress_forward,
    surface_stress_transpose,
    surface_velocity_forward,
    surface_velocity_transpose,
)


C5A_GLOBAL_DOT_RELATIVE_MAX = 1.0e-5
C5A_REFERENCE_RELATIVE_MAX = 5.0e-6
C5A_DOUBLE_REFERENCE_DOT_MAX = 5.0e-12


def f32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", float(value)))[0]


@dataclass(frozen=True)
class Case:
    name: str
    fdorder: int
    mechanisms: int
    fw: int
    free_surface: int
    nproc_x: int
    nproc_y: int
    boundary: int = 0
    nx: int = 14
    ny: int = 14
    dt: float = 0.0013
    dh: float = 7.5

    @property
    def layout(self) -> Layout:
        return Layout(self.nx, self.ny, self.fdorder)

    @property
    def ranks(self) -> int:
        return self.nproc_x * self.nproc_y


CASES = (
    Case("fs0_1x1_fw0", 2, 1, 0, 0, 1, 1),
    Case("fs0_1x2", 4, 3, 2, 0, 1, 2),
    Case("fs0_2x1", 6, 1, 2, 0, 2, 1),
    Case("fs1_1x1", 8, 3, 2, 1, 1, 1),
    Case("fs0_2x2", 10, 3, 2, 0, 2, 2),
    Case("periodic_x_2x1", 12, 1, 2, 0, 2, 1, boundary=1),
)


def _round(value, enabled):
    return f32(value) if enabled else value


def _field(case: Case, rank: int, group: int, dual: bool):
    shift = 0.39 if dual else -0.17
    values = []
    for j in range(case.layout.row_min, case.layout.row_max + 1):
        for i in range(case.layout.col_min, case.layout.col_max + 1):
            z = 0.021 * (rank + 1) + 0.009 * (group + 1) + 0.0013 * j - 0.0008 * i + shift
            values.append(f32(0.31 * math.sin(4.3 * z) + 0.19 * math.cos(2.7 * z)))
    return values


def _aux(case: Case, rank: int, group: int, dual: bool, cells: int):
    shift = 0.23 if dual else -0.11
    return [
        f32(0.13 * math.sin(0.071 * (k + 1) + 0.17 * rank + 0.09 * group + shift))
        for k in range(cells)
    ]


def copy_state(state):
    return {
        key: ([list(v) for v in value] if key in ("r", "q") else list(value))
        for key, value in state.items()
    }


def make_states(case: Case, dual=False):
    result = []
    for rank in range(case.ranks):
        state = {
            "vz": _field(case, rank, 0, dual),
            "sxz": _field(case, rank, 1, dual),
            "syz": _field(case, rank, 2, dual),
            "r": [_field(case, rank, 3 + l, dual) for l in range(case.mechanisms)],
            "q": [_field(case, rank, 3 + case.mechanisms + l, dual) for l in range(case.mechanisms)],
            "psi_sxz_x": _aux(case, rank, 20, dual, case.ny * 2 * case.fw),
            "psi_syz_y": _aux(case, rank, 21, dual, case.nx * 2 * case.fw),
            "psi_vzx": _aux(case, rank, 22, dual, case.ny * 2 * case.fw),
            "psi_vzy": _aux(case, rank, 23, dual, case.nx * 2 * case.fw),
        }
        result.append(state)
    return result


def coefficients(case: Case):
    hc = [0.0] + [f32(v) for v in (1.0, -0.041, 0.007, -0.0014, 0.00031, -0.00007)[: case.fdorder // 2]]
    bip = [f32(0.79 + 0.025 * l) for l in range(case.mechanisms)]
    bjm = [f32(0.77 + 0.021 * l) for l in range(case.mechanisms)]
    cip = [f32(0.91 - 0.018 * l) for l in range(case.mechanisms)]
    cjm = [f32(0.89 - 0.015 * l) for l in range(case.mechanisms)]
    cpml = {
        "K": [0.0] + [f32(1.11 + 0.017 * k) for k in range(1, 2 * case.fw + 1)],
        "Kh": [0.0] + [f32(1.08 + 0.013 * k) for k in range(1, 2 * case.fw + 1)],
        "a": [0.0] + [f32(-0.031 - 0.001 * k) for k in range(1, 2 * case.fw + 1)],
        "ah": [0.0] + [f32(-0.027 - 0.0008 * k) for k in range(1, 2 * case.fw + 1)],
        "b": [0.0] + [f32(0.82 + 0.006 * k) for k in range(1, 2 * case.fw + 1)],
        "bh": [0.0] + [f32(0.84 + 0.005 * k) for k in range(1, 2 * case.fw + 1)],
    }
    return hc, bip, bjm, cip, cjm, cpml


def material(case: Case, rank: int, j: int, i: int, l=None):
    if l is None:
        return (
            f32(0.00043 + 0.000002 * rank + 0.0000003 * j),
            f32(0.0047 + 0.00001 * i + 0.000006 * j),
            f32(0.0042 + 0.000008 * i + 0.000004 * j),
        )
    return (
        f32(0.16 + 0.004 * l + 0.0002 * i),
        f32(0.14 + 0.003 * l + 0.0002 * j),
    )


def _cpml_select(case, rank, axis, staggered, coordinate):
    if case.fw == 0:
        return None
    x = rank % case.nproc_x
    y = rank // case.nproc_x
    if axis == "x":
        if case.boundary:
            return None
        if x == 0 and coordinate <= case.fw:
            return coordinate
        if x == case.nproc_x - 1 and coordinate >= case.nx - case.fw + 1:
            return coordinate - case.nx + 2 * case.fw
    else:
        if y == 0 and not case.free_surface and coordinate <= case.fw:
            return coordinate
        if y == case.nproc_y - 1 and coordinate >= case.ny - case.fw + 1:
            return coordinate - case.ny + 2 * case.fw
    return None


def _cpml_index(case, axis, j, i, aux):
    return (j - 1) * (2 * case.fw) + (aux - 1) if axis == "x" else (aux - 1) * case.nx + (i - 1)


def _cpml_coeff(case, rank, axis, staggered, coordinate, cpml):
    aux = _cpml_select(case, rank, axis, staggered, coordinate)
    if aux is None:
        return None
    if axis == "x":
        names = ("Kh", "ah", "bh") if staggered else ("K", "a", "b")
    else:
        bottom = rank // case.nproc_x == case.nproc_y - 1 and coordinate >= case.ny - case.fw + 1
        names = ("Kh", "ah", "bh") if (staggered and bottom) else ("K", "a", "b")
    return aux, cpml[names[0]][aux], cpml[names[1]][aux], cpml[names[2]][aux]


def _exchange(states, case, kind, transpose=False, round_to_float=False):
    if kind == "velocity":
        fields = [[s["vz"]] for s in states]
    else:
        fields = [[s["sxz"], s["syz"]] for s in states]
    operation = exchange_transpose if transpose else exchange_forward
    if transpose:
        mapped = operation(fields, case.layout, case.nproc_x, case.nproc_y, case.boundary, kind, round_to_float=round_to_float)
    else:
        mapped = operation(fields, case.layout, case.nproc_x, case.nproc_y, case.boundary, kind)
    result = [copy_state(s) for s in states]
    for rank in range(case.ranks):
        if kind == "velocity":
            result[rank]["vz"] = mapped[rank][0]
        else:
            result[rank]["sxz"], result[rank]["syz"] = mapped[rank]
    return result


def _surface(states, case, kind, transpose=False, round_to_float=False):
    if not case.free_surface:
        return [copy_state(s) for s in states]
    result = [copy_state(s) for s in states]
    fields = [[s["vz"]] for s in states] if kind == "velocity" else [[s["sxz"], s["syz"]] for s in states]
    if kind == "velocity":
        mapped = surface_velocity_transpose(fields, case.layout, case.nproc_x, round_to_float=round_to_float) if transpose else surface_velocity_forward(fields, case.layout, case.nproc_x)
        for rank in range(case.ranks): result[rank]["vz"] = mapped[rank][0]
    else:
        mapped = surface_stress_transpose(fields, case.layout, case.nproc_x, round_to_float=round_to_float) if transpose else surface_stress_forward(fields, case.layout, case.nproc_x)
        for rank in range(case.ranks): result[rank]["sxz"], result[rank]["syz"] = mapped[rank]
    return result


def _add(values, index, value, rounded):
    values[index] = _round(values[index] + value, rounded)


def forward(states, signals, case: Case, *, rounded=False):
    out = [copy_state(s) for s in states]
    hc, bip, bjm, cip, cjm, cpml = coefficients(case)
    h, layout = case.fdorder // 2, case.layout
    for rank, state in enumerate(out):
        for j in range(1, case.ny + 1):
            for i in range(1, case.nx + 1):
                dx = sum(hc[m] * (state["sxz"][layout.index(j, i + m - 1)] - state["sxz"][layout.index(j, i - m)]) for m in range(1, h + 1))
                dy = sum(hc[m] * (state["syz"][layout.index(j + m - 1, i)] - state["syz"][layout.index(j - m, i)]) for m in range(1, h + 1))
                for axis, raw, key, staggered in (("x", dx, "psi_sxz_x", False), ("y", dy, "psi_syz_y", False)):
                    selected = _cpml_coeff(case, rank, axis, staggered, i if axis == "x" else j, cpml)
                    if selected:
                        aux, K, a, b = selected
                        ci = _cpml_index(case, axis, j, i, aux)
                        state[key][ci] = _round(b * state[key][ci] + a * raw, rounded)
                        if axis == "x": dx = _round(raw / K + state[key][ci], rounded)
                        else: dy = _round(raw / K + state[key][ci], rounded)
                rhoi, _, _ = material(case, rank, j, i)
                idx = layout.index(j, i)
                state["vz"][idx] = _round(state["vz"][idx] + case.dt * rhoi * (dx + dy) / case.dh, rounded)
        source_i, source_j = 3 + rank % 2, 4 + rank % 3
        state["vz"][layout.index(source_j, source_i)] = _round(state["vz"][layout.index(source_j, source_i)] + signals[rank], rounded)
    out = _exchange(out, case, "velocity")
    out = _surface(out, case, "velocity")
    for rank, state in enumerate(out):
        for j in range(1, case.ny + 1):
            for i in range(1, case.nx + 1):
                ex = sum(hc[m] * (state["vz"][layout.index(j, i + m)] - state["vz"][layout.index(j, i - (m - 1))]) for m in range(1, h + 1)) / case.dh
                ey = sum(hc[m] * (state["vz"][layout.index(j + m, i)] - state["vz"][layout.index(j - (m - 1), i)]) for m in range(1, h + 1)) / case.dh
                for axis, raw, key in (("x", ex, "psi_vzx"), ("y", ey, "psi_vzy")):
                    selected = _cpml_coeff(case, rank, axis, True, i if axis == "x" else j, cpml)
                    if selected:
                        aux, K, a, b = selected
                        ci = _cpml_index(case, axis, j, i, aux)
                        state[key][ci] = _round(b * state[key][ci] + a * raw, rounded)
                        if axis == "x": ex = _round(raw / K + state[key][ci], rounded)
                        else: ey = _round(raw / K + state[key][ci], rounded)
                idx = layout.index(j, i)
                rhoi, fx, fy = material(case, rank, j, i)
                del rhoi
                old_r = [state["r"][l][idx] for l in range(case.mechanisms)]
                old_q = [state["q"][l][idx] for l in range(case.mechanisms)]
                new_r, new_q = [], []
                for l in range(case.mechanisms):
                    dip, d = material(case, rank, j, i, l)
                    new_r.append(_round(bip[l] * (cip[l] * old_r[l] - dip * ex), rounded))
                    new_q.append(_round(bjm[l] * (cjm[l] * old_q[l] - d * ey), rounded))
                state["sxz"][idx] = _round(state["sxz"][idx] + fx * ex + 0.5 * case.dt * (sum(old_r) + sum(new_r)), rounded)
                state["syz"][idx] = _round(state["syz"][idx] + fy * ey + 0.5 * case.dt * (sum(old_q) + sum(new_q)), rounded)
                for l in range(case.mechanisms):
                    state["r"][l][idx], state["q"][l][idx] = new_r[l], new_q[l]
    out = _surface(out, case, "stress")
    out = _exchange(out, case, "stress")
    receivers = [out[rank]["vz"][layout.index(5 + rank % 2, 6 + rank % 3)] for rank in range(case.ranks)]
    return out, receivers


def transpose(bars, bar_receiver, case: Case, *, rounded=False):
    work = [copy_state(s) for s in bars]
    hc, bip, bjm, cip, cjm, cpml = coefficients(case)
    h, layout = case.fdorder // 2, case.layout
    for rank in range(case.ranks):
        _add(work[rank]["vz"], layout.index(5 + rank % 2, 6 + rank % 3), bar_receiver[rank], rounded)
    work = _exchange(work, case, "stress", transpose=True, round_to_float=rounded)
    work = _surface(work, case, "stress", transpose=True, round_to_float=rounded)
    prev = [copy_state(s) for s in work]
    for rank in range(case.ranks):
        prev[rank]["vz"] = [0.0] * layout.cells
    for rank, state in enumerate(work):
        for j in range(1, case.ny + 1):
            for i in range(1, case.nx + 1):
                idx = layout.index(j, i)
                bar_ex = bar_ey = 0.0
                for axis, stress, memory, F, b0, c0, key in (
                    ("x", "sxz", "r", material(case, rank, j, i)[1], bip, cip, "psi_vzx"),
                    ("y", "syz", "q", material(case, rank, j, i)[2], bjm, cjm, "psi_vzy"),
                ):
                    bar_s = state[stress][idx]
                    bar_e = F * bar_s
                    prev[rank][stress][idx] = _round(bar_s, rounded)
                    for l in range(case.mechanisms):
                        coeff, _other = material(case, rank, j, i, l) if axis == "x" else material(case, rank, j, i, l)[::-1]
                        C = -b0[l] * coeff
                        A = b0[l] * c0[l]
                        t = state[memory][l][idx] + 0.5 * case.dt * bar_s
                        prev[rank][memory][l][idx] = _round(A * t + 0.5 * case.dt * bar_s, rounded)
                        bar_e += C * t
                    selected = _cpml_coeff(case, rank, axis, True, i if axis == "x" else j, cpml)
                    if selected:
                        aux, K, a, b = selected
                        ci = _cpml_index(case, axis, j, i, aux)
                        tpsi = state[key][ci] + bar_e
                        prev[rank][key][ci] = _round(b * tpsi, rounded)
                        bar_raw = bar_e / K + a * tpsi
                    else:
                        bar_raw = bar_e
                    if axis == "x": bar_ex = bar_raw
                    else: bar_ey = bar_raw
                for m in range(1, h + 1):
                    scale = hc[m] / case.dh
                    _add(work[rank]["vz"], layout.index(j, i + m), scale * bar_ex, rounded)
                    _add(work[rank]["vz"], layout.index(j, i - (m - 1)), -scale * bar_ex, rounded)
                    _add(work[rank]["vz"], layout.index(j + m, i), scale * bar_ey, rounded)
                    _add(work[rank]["vz"], layout.index(j - (m - 1), i), -scale * bar_ey, rounded)
    work = _surface(work, case, "velocity", transpose=True, round_to_float=rounded)
    work = _exchange(work, case, "velocity", transpose=True, round_to_float=rounded)
    bar_signal = []
    for rank in range(case.ranks):
        prev[rank]["vz"] = list(work[rank]["vz"])
        source_i, source_j = 3 + rank % 2, 4 + rank % 3
        bar_signal.append(work[rank]["vz"][layout.index(source_j, source_i)])
        for j in range(1, case.ny + 1):
            for i in range(1, case.nx + 1):
                idx = layout.index(j, i)
                bar_v = work[rank]["vz"][idx]
                rhoi = material(case, rank, j, i)[0]
                factor = case.dt * rhoi / case.dh * bar_v
                for axis, key in (("x", "psi_sxz_x"), ("y", "psi_syz_y")):
                    selected = _cpml_coeff(case, rank, axis, False, i if axis == "x" else j, cpml)
                    if selected:
                        aux, K, a, b = selected
                        ci = _cpml_index(case, axis, j, i, aux)
                        tpsi = work[rank][key][ci] + factor
                        prev[rank][key][ci] = _round(b * tpsi, rounded)
                        raw = factor / K + a * tpsi
                    else:
                        raw = factor
                    for m in range(1, h + 1):
                        if axis == "x":
                            _add(prev[rank]["sxz"], layout.index(j, i + m - 1), hc[m] * raw, rounded)
                            _add(prev[rank]["sxz"], layout.index(j, i - m), -hc[m] * raw, rounded)
                        else:
                            _add(prev[rank]["syz"], layout.index(j + m - 1, i), hc[m] * raw, rounded)
                            _add(prev[rank]["syz"], layout.index(j - m, i), -hc[m] * raw, rounded)
    return prev, bar_signal


def state_dot(left, right):
    keys = ("vz", "sxz", "syz", "psi_sxz_x", "psi_syz_y", "psi_vzx", "psi_vzy")
    terms = []
    for a, b in zip(left, right):
        for key in keys:
            terms.extend(x * y for x, y in zip(a[key], b[key]))
        for key in ("r", "q"):
            for av, bv in zip(a[key], b[key]):
                terms.extend(x * y for x, y in zip(av, bv))
    return math.fsum(terms)


def relative_dot(lhs, rhs):
    return abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1.0e-300)
