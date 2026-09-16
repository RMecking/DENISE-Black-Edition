"""Independent multi-step reference for the fixed-material SH state map."""

from __future__ import annotations

from dataclasses import dataclass
import math

from tests.utilities.m63c_full_state_step_reference import (
    Case,
    copy_state,
    forward,
    make_states,
    relative_dot,
    state_dot,
    transpose,
)


C5B_GLOBAL_DOT_RELATIVE_MAX = 1.0e-5
C5B_REFERENCE_RELATIVE_MAX = 5.0e-6
C5B_DOUBLE_DOT_RELATIVE_MAX = 5.0e-12
C5B_N1_C5A_RELATIVE_MAX = 5.0e-7
C5B_N2_COMPOSITION_RELATIVE_MAX = 5.0e-7


@dataclass(frozen=True)
class MultiStepCase:
    base: Case
    nsteps: int

    @property
    def name(self) -> str:
        return f"{self.base.name}_n{self.nsteps}"


CASES = (
    MultiStepCase(Case("fs0_1x1_fw0", 2, 1, 0, 0, 1, 1), 1),
    MultiStepCase(Case("fs0_1x2", 4, 3, 2, 0, 1, 2), 2),
    MultiStepCase(Case("fs0_2x1", 6, 1, 2, 0, 2, 1), 5),
    MultiStepCase(Case("fs1_1x1", 8, 3, 2, 1, 1, 1), 8),
    MultiStepCase(Case("fs0_2x2", 10, 3, 2, 0, 2, 2), 12),
    MultiStepCase(
        Case("periodic_x_2x1", 12, 1, 2, 0, 2, 1, boundary=1), 5
    ),
)


def signal_series(case: MultiStepCase) -> list[list[float]]:
    return [
        [0.021 + 0.003 * rank + 0.0017 * n for rank in range(case.base.ranks)]
        for n in range(case.nsteps)
    ]


def receiver_dual_series(
    case: MultiStepCase, mode: str = "full", impulse: int | None = None
) -> list[list[float]]:
    result = []
    for n in range(case.nsteps):
        row = [-0.17 + 0.019 * rank - 0.0023 * n for rank in range(case.base.ranks)]
        if mode == "terminal":
            row = [0.0] * case.base.ranks
        elif mode == "impulse" and n != impulse:
            row = [0.0] * case.base.ranks
        result.append(row)
    return result


def zero_states(case: Case):
    result = make_states(case, dual=True)
    for state in result:
        for key in ("vz", "sxz", "syz", "psi_sxz_x", "psi_syz_y", "psi_vzx", "psi_vzy"):
            state[key] = [0.0] * len(state[key])
        for key in ("r", "q"):
            state[key] = [[0.0] * len(values) for values in state[key]]
    return result


def terminal_dual(case: MultiStepCase, mode: str = "full"):
    return zero_states(case.base) if mode in ("receiver", "impulse") else make_states(case.base, dual=True)


def forward_multi(states, signals, case: MultiStepCase, *, rounded=False):
    current = [copy_state(state) for state in states]
    receivers = []
    for n in range(case.nsteps):
        current, row = forward(current, signals[n], case.base, rounded=rounded)
        receivers.append(row)
    return current, receivers


def transpose_multi(bars, bar_receivers, case: MultiStepCase, *, rounded=False):
    current = [copy_state(state) for state in bars]
    bar_signals = [[0.0] * case.base.ranks for _ in range(case.nsteps)]
    for n in range(case.nsteps - 1, -1, -1):
        current, bar_signals[n] = transpose(
            current, bar_receivers[n], case.base, rounded=rounded
        )
    return current, bar_signals


def time_series_dot(left, right):
    return math.fsum(
        value * dual
        for left_row, right_row in zip(left, right)
        for value, dual in zip(left_row, right_row)
    )


def multi_step_dot(case: MultiStepCase, *, rounded=False):
    initial = make_states(case.base, dual=False)
    dual = terminal_dual(case)
    signals = signal_series(case)
    bar_receivers = receiver_dual_series(case)
    final, receivers = forward_multi(initial, signals, case, rounded=rounded)
    bar_initial, bar_signals = transpose_multi(
        dual, bar_receivers, case, rounded=rounded
    )
    lhs = state_dot(final, dual) + time_series_dot(receivers, bar_receivers)
    rhs = state_dot(initial, bar_initial) + time_series_dot(signals, bar_signals)
    return lhs, rhs, relative_dot(lhs, rhs)
