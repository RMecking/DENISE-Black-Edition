"""Independent C7a forward material-observable trajectory reference."""

from __future__ import annotations

from dataclasses import dataclass

from tests.utilities.m63c_full_state_step_reference import (
    Case,
    _cpml_coeff,
    _cpml_index,
    _exchange,
    _round,
    _surface,
    coefficients,
    copy_state,
    material,
)


C7A_REFERENCE_RELATIVE_MAX = 5.0e-6


@dataclass(frozen=True)
class ObservableCase:
    case: Case
    nsteps: int

    @property
    def name(self):
        return f"{self.case.name}_n{self.nsteps}"


CASES = (
    ObservableCase(Case("fs0_1x1_fw0", 2, 1, 0, 0, 1, 1), 1),
    ObservableCase(Case("fs0_2x1", 4, 3, 2, 0, 2, 1), 2),
    ObservableCase(Case("fs1_1x2", 6, 1, 2, 1, 1, 2), 5),
    ObservableCase(Case("fs0_2x2", 8, 3, 2, 0, 2, 2), 6),
    ObservableCase(
        Case("periodic_x_2x1", 10, 1, 2, 0, 2, 1, boundary=1), 7
    ),
    ObservableCase(Case("fs1_1x1_fw0", 12, 3, 0, 1, 1, 1), 8),
)


def signal(case: Case, rank: int, step: int) -> float:
    """A timestep-distinguishing deterministic source signature."""
    return _round(
        (0.013 + 0.0021 * rank) * (step + 1)
        + (0.004 if step % 2 else -0.003),
        True,
    )


def forward_step_with_observables(states, signals, case: Case, *, rounded=True):
    """Apply the independent forward equations and return the three channels."""
    out = [copy_state(state) for state in states]
    hc, bip, bjm, cip, cjm, cpml = coefficients(case)
    h, layout = case.fdorder // 2, case.layout
    observables = []
    for rank, state in enumerate(out):
        qsum = [0.0] * (case.nx * case.ny)
        strain_x = [0.0] * (case.nx * case.ny)
        strain_y = [0.0] * (case.nx * case.ny)
        observables.append(
            {"qsum": qsum, "strain_x": strain_x, "strain_y": strain_y}
        )
        for j in range(1, case.ny + 1):
            for i in range(1, case.nx + 1):
                dx = sum(
                    hc[m]
                    * (
                        state["sxz"][layout.index(j, i + m - 1)]
                        - state["sxz"][layout.index(j, i - m)]
                    )
                    for m in range(1, h + 1)
                )
                dy = sum(
                    hc[m]
                    * (
                        state["syz"][layout.index(j + m - 1, i)]
                        - state["syz"][layout.index(j - m, i)]
                    )
                    for m in range(1, h + 1)
                )
                for axis, raw, key in (
                    ("x", dx, "psi_sxz_x"),
                    ("y", dy, "psi_syz_y"),
                ):
                    selected = _cpml_coeff(
                        case, rank, axis, False, i if axis == "x" else j, cpml
                    )
                    if selected:
                        aux, K, a, b = selected
                        ci = _cpml_index(case, axis, j, i, aux)
                        state[key][ci] = _round(b * state[key][ci] + a * raw, rounded)
                        corrected = _round(raw / K + state[key][ci], rounded)
                        if axis == "x":
                            dx = corrected
                        else:
                            dy = corrected
                owned = (j - 1) * case.nx + (i - 1)
                qsum[owned] = _round(dx + dy, rounded)
                rhoi = material(case, rank, j, i)[0]
                idx = layout.index(j, i)
                state["vz"][idx] = _round(
                    state["vz"][idx] + case.dt * rhoi * (dx + dy) / case.dh,
                    rounded,
                )
        source_i, source_j = 3 + rank % 2, 4 + rank % 3
        source_index = layout.index(source_j, source_i)
        state["vz"][source_index] = _round(
            state["vz"][source_index] + signals[rank], rounded
        )

    out = _surface(_exchange(out, case, "velocity"), case, "velocity")
    for rank, state in enumerate(out):
        for j in range(1, case.ny + 1):
            for i in range(1, case.nx + 1):
                ex = sum(
                    hc[m]
                    * (
                        state["vz"][layout.index(j, i + m)]
                        - state["vz"][layout.index(j, i - (m - 1))]
                    )
                    for m in range(1, h + 1)
                ) / case.dh
                ey = sum(
                    hc[m]
                    * (
                        state["vz"][layout.index(j + m, i)]
                        - state["vz"][layout.index(j - (m - 1), i)]
                    )
                    for m in range(1, h + 1)
                ) / case.dh
                for axis, raw, key in (
                    ("x", ex, "psi_vzx"),
                    ("y", ey, "psi_vzy"),
                ):
                    selected = _cpml_coeff(
                        case, rank, axis, True, i if axis == "x" else j, cpml
                    )
                    if selected:
                        aux, K, a, b = selected
                        ci = _cpml_index(case, axis, j, i, aux)
                        state[key][ci] = _round(b * state[key][ci] + a * raw, rounded)
                        corrected = _round(raw / K + state[key][ci], rounded)
                        if axis == "x":
                            ex = corrected
                        else:
                            ey = corrected
                owned = (j - 1) * case.nx + (i - 1)
                observables[rank]["strain_x"][owned] = ex
                observables[rank]["strain_y"][owned] = ey
                idx = layout.index(j, i)
                _, fx, fy = material(case, rank, j, i)
                old_r = [state["r"][l][idx] for l in range(case.mechanisms)]
                old_q = [state["q"][l][idx] for l in range(case.mechanisms)]
                new_r, new_q = [], []
                for l in range(case.mechanisms):
                    dip, d = material(case, rank, j, i, l)
                    new_r.append(_round(bip[l] * (cip[l] * old_r[l] - dip * ex), rounded))
                    new_q.append(_round(bjm[l] * (cjm[l] * old_q[l] - d * ey), rounded))
                state["sxz"][idx] = _round(
                    state["sxz"][idx]
                    + fx * ex
                    + 0.5 * case.dt * (sum(old_r) + sum(new_r)),
                    rounded,
                )
                state["syz"][idx] = _round(
                    state["syz"][idx]
                    + fy * ey
                    + 0.5 * case.dt * (sum(old_q) + sum(new_q)),
                    rounded,
                )
                for l in range(case.mechanisms):
                    state["r"][l][idx], state["q"][l][idx] = new_r[l], new_q[l]
    out = _exchange(_surface(out, case, "stress"), case, "stress")
    receivers = [
        out[rank]["vz"][layout.index(5 + rank % 2, 6 + rank % 3)]
        for rank in range(case.ranks)
    ]
    return out, receivers, observables


def forward_trajectory(states, observable_case: ObservableCase):
    current = [copy_state(state) for state in states]
    trajectory = [[] for _ in range(observable_case.case.ranks)]
    receivers = [[] for _ in range(observable_case.case.ranks)]
    for step in range(observable_case.nsteps):
        signals = [
            signal(observable_case.case, rank, step)
            for rank in range(observable_case.case.ranks)
        ]
        current, sampled, observables = forward_step_with_observables(
            current, signals, observable_case.case, rounded=True
        )
        for rank in range(observable_case.case.ranks):
            trajectory[rank].append(observables[rank])
            receivers[rank].append(sampled[rank])
    return current, receivers, trajectory
