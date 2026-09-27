"""Independent M8e-2C3 native-gradient reference.

The reverse state is evolved only by the frozen M8e-2C0 reference.  This
module adds the CPU-authoritative correlations from visco_psv_exact_gradient.c
and consumes the six recorded forward operands in absolute-time order.
"""

from __future__ import annotations

from tests.utilities.visco_psv_one_step_adjoint_reference import (
    Fixture,
    stress_gsls_transpose,
    velocity_transpose,
)


NATIVE_FIELDS = ("gf", "gg", "gfc", "gd", "ge", "gdc", "grx", "gry")
OPERAND_FIELDS = ("fx", "fy", "vxx", "vyx", "vxy", "vyy")


def _inject_residual(fixture: Fixture, fields: dict[str, list[float]],
                     timestep: int, traces: tuple[tuple[float, ...], ...],
                     nt: int) -> None:
    if timestep <= 1:
        return
    modeled_vx, observed_vx, modeled_vy, observed_vy = traces
    for receiver, (j, i, *_unused) in enumerate(fixture.receivers):
        sample = receiver * nt + timestep - 1
        point = fixture.index(j, i)
        fields["avx"][point] += modeled_vx[sample] - observed_vx[sample]
        fields["avy"][point] += modeled_vy[sample] - observed_vy[sample]


def _stress_correlation(fixture: Fixture, fields: dict[str, list[float]],
                        operands: dict[str, tuple[float, ...]], timestep: int,
                        gradient: dict[str, list[float]]) -> None:
    cells = fixture.nx * fixture.ny
    time_offset = (timestep - 1) * cells
    dt2, b = fixture.dt * 0.5, fixture.bjm1
    for j in range(1, fixture.ny + 1):
        for i in range(1, fixture.nx + 1):
            q = (j - 1) * fixture.nx + i - 1
            p = fixture.index(j, i)
            o = time_offset + q
            xx, yx = operands["vxx"][o], operands["vyx"][o]
            xy, yy = operands["vxy"][o], operands["vyy"][o]
            div, shear = xx + yy, xy + yx
            lambda_r = fields["ar"][p] + dt2 * fields["asxy"][p]
            lambda_p = fields["ap"][p] + dt2 * fields["asxx"][p]
            lambda_q = fields["aq"][p] + dt2 * fields["asyy"][p]
            gradient["gfc"][q] += fields["asxy"][p] * shear
            gradient["gf"][q] += -2.0 * (
                fields["asxx"][p] * yy + fields["asyy"][p] * xx
            )
            gradient["gg"][q] += (
                fields["asxx"][p] + fields["asyy"][p]
            ) * div
            gradient["gdc"][q] += -b * lambda_r * shear
            gradient["gd"][q] += 2.0 * b * (
                lambda_p * yy + lambda_q * xx
            )
            gradient["ge"][q] += -b * (lambda_p + lambda_q) * div


def _velocity_correlation(fixture: Fixture, fields: dict[str, list[float]],
                          operands: dict[str, tuple[float, ...]], timestep: int,
                          gradient: dict[str, list[float]]) -> None:
    cells = fixture.nx * fixture.ny
    time_offset = (timestep - 1) * cells
    for j in range(1, fixture.ny + 1):
        for i in range(1, fixture.nx + 1):
            q = (j - 1) * fixture.nx + i - 1
            p = fixture.index(j, i)
            o = time_offset + q
            gradient["grx"][q] += (
                fields["avx"][p] * fixture.dt * operands["fx"][o] / fixture.dh
            )
            gradient["gry"][q] += (
                fields["avy"][p] * fixture.dt * operands["fy"][o] / fixture.dh
            )


def native_gradient_boundaries(
    fixture: Fixture,
    traces: tuple[tuple[float, ...], ...],
    operands: dict[str, tuple[float, ...]],
    nt: int,
    segments: int,
) -> list[dict[str, list[float]]]:
    """Return cumulative gradients after every descending segment boundary."""
    fields = {name: values.copy() for name, values in fixture.fields.items()}
    gradient = {name: [0.0] * (fixture.nx * fixture.ny)
                for name in NATIVE_FIELDS}
    boundaries = []
    for segment in range(segments - 1, -1, -1):
        begin = segment * nt // segments + 1
        end = (segment + 1) * nt // segments
        for timestep in range(end, begin - 1, -1):
            _inject_residual(fixture, fields, timestep, traces, nt)
            _stress_correlation(fixture, fields, operands, timestep, gradient)
            stress_gsls_transpose(fixture, fields)
            _velocity_correlation(fixture, fields, operands, timestep, gradient)
            velocity_transpose(fixture, fields)
        boundaries.append({name: values.copy()
                           for name, values in gradient.items()})
    return boundaries
