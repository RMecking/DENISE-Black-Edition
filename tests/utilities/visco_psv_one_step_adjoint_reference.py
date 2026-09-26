"""Independent FP64 reference for one exact viscoelastic P/SV reverse step.

This module is deliberately standalone: it uses only Python's standard
library, has no dependency on DENISE or CUDA, and does not import production
helpers.  The equations and staggered scatters are written in direct-index
form so the test can freeze every output field and separately check the
transpose identities against forward operators.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct


MAIN_FIELDS = ("avx", "avy", "asxx", "asyy", "asxy", "ar", "ap", "aq")
CPML_FIELDS = (
    "psxx", "psxyx", "psxyy", "psyy",
    "pvxx", "pvyx", "pvxy", "pvyy",
)
ALL_FIELDS = MAIN_FIELDS + CPML_FIELDS

PSXX, PSXYX, PSXYY, PSYY, PVXX, PVYX, PVXY, PVYY = CPML_FIELDS


def _float32(value: float) -> float:
    """Round fixture coefficients/observations to production's float type."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass(frozen=True)
class Fixture:
    name: str
    nx: int
    ny: int
    fw: int
    dt: float
    dh: float
    hc1: float
    hc2: float
    bjm1: float
    cjm1: float
    fields: dict[str, list[float]]
    material: dict[str, list[float]]
    cpml: dict[str, dict[str, list[float]]]
    receivers: tuple[tuple[int, int, float, float, float, float], ...]

    @property
    def pitch(self) -> int:
        return self.nx + 6

    @property
    def height(self) -> int:
        return self.ny + 6

    @property
    def count(self) -> int:
        return self.pitch * self.height

    def index(self, j: int, i: int) -> int:
        if not (-2 <= j <= self.ny + 3 and -2 <= i <= self.nx + 3):
            raise IndexError((j, i))
        return (j + 2) * self.pitch + i + 2


def _pattern(tag: int, j: int, i: int) -> float:
    integer = ((tag * 61 + (j + 11) * 17 + (i + 13) * 29
                + (j + 7) * (i + 3) * 5) % 257) - 128
    value = integer / 257.0 + tag * 0.00031 + j * 0.000017 - i * 0.000023
    return value + 0.019 if abs(value) < 0.01 else value


def _cell_array(fixture_shape: tuple[int, int], tag: int, *, float32: bool) -> list[float]:
    nx, ny = fixture_shape
    pitch = nx + 6
    values = [0.0] * (pitch * (ny + 6))
    for j in range(-2, ny + 4):
        for i in range(-2, nx + 4):
            value = _pattern(tag, j, i)
            values[(j + 2) * pitch + i + 2] = _float32(value) if float32 else value
    return values


def _cpml_h(coordinate: int, extent: int, fw: int) -> int:
    if coordinate <= fw:
        return coordinate
    if coordinate >= extent - fw + 1:
        return coordinate - extent + 2 * fw
    return 0


def build_fixture(name: str, nx: int, ny: int, fw: int) -> Fixture:
    if nx <= 2 * fw + 4 or ny <= 2 * fw + 4:
        raise ValueError("fixture must include CPML strips and an interior region")
    shape = (nx, ny)
    pitch = nx + 6
    count = pitch * (ny + 6)
    fields = {field: _cell_array(shape, tag, float32=False)
              for tag, field in enumerate(MAIN_FIELDS, start=1)}

    # Adjoint CPML variables use the exact solver's full cell indexing.  Only
    # cells belonging to that component's live CPML strip are populated.
    cpml_orientation = {
        PSXX: "x", PSXYX: "x", PSXYY: "y", PSYY: "y",
        PVXX: "x", PVYX: "x", PVXY: "y", PVYY: "y",
    }
    for tag, field in enumerate(CPML_FIELDS, start=9):
        values = [0.0] * count
        axis = cpml_orientation[field]
        extent = nx if axis == "x" else ny
        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                coordinate = i if axis == "x" else j
                if _cpml_h(coordinate, extent, fw):
                    values[(j + 2) * pitch + i + 2] = _pattern(tag, j, i)
        fields[field] = values

    material = {}
    coefficient_tags = {
        "prip": (0.19, 0.0021), "prjp": (0.23, 0.0017),
        "f": (0.31, 0.0013), "g": (0.57, 0.0019),
        "fipjp": (0.27, 0.0011), "d": (0.013, 0.00011),
        "e": (0.017, 0.00013), "dip": (0.009, 0.00017),
    }
    for tag, (field, (base, step)) in enumerate(coefficient_tags.items(), start=1):
        values = [0.0] * count
        for j in range(-2, ny + 4):
            for i in range(-2, nx + 4):
                n = (tag * 7 + (j + 3) * 5 + (i + 5) * 11
                     + (j + 2) * (i + 1) * 3) % 23
                values[(j + 2) * pitch + i + 2] = _float32(
                    base + step * n + 0.00003 * (j - i)
                )
        material[field] = values

    cpml = {}
    for group_tag, group in enumerate(("x", "x_half", "y", "y_half"), start=1):
        cpml[group] = {}
        for kind, base, step in (("K", 1.035, 0.0061),
                                 ("a", -0.023, 0.00037),
                                 ("b", 0.681, 0.0093)):
            cpml[group][kind] = [0.0] + [
                _float32(base + step * (group_tag * 2 + h)
                         + 0.00019 * ((h * group_tag) % 3))
                for h in range(1, 2 * fw + 1)
            ]

    receivers = (
        (2, 3, _float32(0.173), _float32(-0.091),
         _float32(-0.227), _float32(0.064)),
        (fw + 3, fw + 2, _float32(-0.118), _float32(0.057),
         _float32(0.209), _float32(-0.033)),
        (ny - 1, nx - 2, _float32(0.084), _float32(-0.141),
         _float32(-0.102), _float32(0.046)),
    )
    return Fixture(name, nx, ny, fw, _float32(0.0125), _float32(0.5),
                   _float32(1.125), _float32(-1.0 / 24.0),
                   _float32(0.873), _float32(0.917), fields, material,
                   cpml, receivers)


def _clone_fields(fields: dict[str, list[float]]) -> dict[str, list[float]]:
    return {name: values.copy() for name, values in fields.items()}


def inject_receiver_residuals(fixture: Fixture, fields: dict[str, list[float]],
                              timestep: int) -> None:
    if timestep <= 1:
        return
    for j, i, modeled_vx, observed_vx, modeled_vy, observed_vy in fixture.receivers:
        p = fixture.index(j, i)
        fields["avx"][p] += modeled_vx - observed_vx
        fields["avy"][p] += modeled_vy - observed_vy


def _reverse_cpml(fixture: Fixture, fields: dict[str, list[float]],
                  psi_name: str, p: int, coordinate: int, extent: int,
                  group: str, corrected: float) -> float:
    h = _cpml_h(coordinate, extent, fixture.fw)
    if not h:
        return corrected
    profile = fixture.cpml[group]
    combined = fields[psi_name][p] + corrected
    fields[psi_name][p] = profile["b"][h] * combined
    return corrected / profile["K"][h] + profile["a"][h] * combined


def _add_backward_x(fixture: Fixture, field: list[float], j: int, i: int,
                    value: float, scale: float = 1.0) -> None:
    h1, h2 = fixture.hc1, fixture.hc2
    field[fixture.index(j, i)] += scale * h1 * value
    field[fixture.index(j, i - 1)] -= scale * h1 * value
    field[fixture.index(j, i + 1)] += scale * h2 * value
    field[fixture.index(j, i - 2)] -= scale * h2 * value


def _add_forward_x(fixture: Fixture, field: list[float], j: int, i: int,
                   value: float, scale: float = 1.0) -> None:
    h1, h2 = fixture.hc1, fixture.hc2
    field[fixture.index(j, i + 1)] += scale * h1 * value
    field[fixture.index(j, i)] -= scale * h1 * value
    field[fixture.index(j, i + 2)] += scale * h2 * value
    field[fixture.index(j, i - 1)] -= scale * h2 * value


def _add_backward_y(fixture: Fixture, field: list[float], j: int, i: int,
                    value: float, scale: float = 1.0) -> None:
    h1, h2 = fixture.hc1, fixture.hc2
    field[fixture.index(j, i)] += scale * h1 * value
    field[fixture.index(j - 1, i)] -= scale * h1 * value
    field[fixture.index(j + 1, i)] += scale * h2 * value
    field[fixture.index(j - 2, i)] -= scale * h2 * value


def _add_forward_y(fixture: Fixture, field: list[float], j: int, i: int,
                   value: float, scale: float = 1.0) -> None:
    h1, h2 = fixture.hc1, fixture.hc2
    field[fixture.index(j + 1, i)] += scale * h1 * value
    field[fixture.index(j, i)] -= scale * h1 * value
    field[fixture.index(j + 2, i)] += scale * h2 * value
    field[fixture.index(j - 1, i)] -= scale * h2 * value


def stress_gsls_transpose(fixture: Fixture, fields: dict[str, list[float]]) -> None:
    """Transpose stress and L=1 GSLS recurrence into velocity/memory bars."""
    dt2 = fixture.dt * 0.5
    b, c = fixture.bjm1, fixture.cjm1
    material = fixture.material
    for j in range(1, fixture.ny + 1):
        for i in range(1, fixture.nx + 1):
            p = fixture.index(j, i)
            lam_r = fields["ar"][p] + dt2 * fields["asxy"][p]
            lam_p = fields["ap"][p] + dt2 * fields["asxx"][p]
            lam_q = fields["aq"][p] + dt2 * fields["asyy"][p]
            f, g = material["f"][p], material["g"][p]
            d, e = material["d"][p], material["e"][p]
            dip = material["dip"][p]
            ax = fields["asxx"][p] * g + fields["asyy"][p] * (g - 2.0 * f)
            ay = fields["asyy"][p] * g + fields["asxx"][p] * (g - 2.0 * f)
            ax += b * (-e * (lam_p + lam_q) + 2.0 * d * lam_q)
            ay += b * (-e * (lam_p + lam_q) + 2.0 * d * lam_p)
            shear = fields["asxy"][p] * material["fipjp"][p] - b * lam_r * dip
            fields["ar"][p] = dt2 * fields["asxy"][p] + b * c * lam_r
            fields["ap"][p] = dt2 * fields["asxx"][p] + b * c * lam_p
            fields["aq"][p] = dt2 * fields["asyy"][p] + b * c * lam_q

            xx = _reverse_cpml(fixture, fields, PVXX, p, i, fixture.nx,
                               "x", ax)
            yx = _reverse_cpml(fixture, fields, PVYX, p, i, fixture.nx,
                               "x_half", shear)
            xy = _reverse_cpml(fixture, fields, PVXY, p, j, fixture.ny,
                               "y_half", shear)
            yy = _reverse_cpml(fixture, fields, PVYY, p, j, fixture.ny,
                               "y", ay)
            _add_backward_x(fixture, fields["avx"], j, i, xx, 1.0 / fixture.dh)
            _add_forward_x(fixture, fields["avy"], j, i, yx, 1.0 / fixture.dh)
            _add_forward_y(fixture, fields["avx"], j, i, xy, 1.0 / fixture.dh)
            _add_backward_y(fixture, fields["avy"], j, i, yy, 1.0 / fixture.dh)


def velocity_transpose(fixture: Fixture, fields: dict[str, list[float]]) -> None:
    """Transpose velocity update into stress bars and four CPML histories."""
    material = fixture.material
    for j in range(1, fixture.ny + 1):
        for i in range(1, fixture.nx + 1):
            p = fixture.index(j, i)
            ax = fields["avx"][p] * fixture.dt * material["prip"][p] / fixture.dh
            ay = fields["avy"][p] * fixture.dt * material["prjp"][p] / fixture.dh
            xx = _reverse_cpml(fixture, fields, PSXX, p, i, fixture.nx,
                               "x_half", ax)
            yx = _reverse_cpml(fixture, fields, PSXYX, p, i, fixture.nx,
                               "x", ay)
            xy = _reverse_cpml(fixture, fields, PSXYY, p, j, fixture.ny,
                               "y", ax)
            yy = _reverse_cpml(fixture, fields, PSYY, p, j, fixture.ny,
                               "y_half", ay)
            _add_forward_x(fixture, fields["asxx"], j, i, xx)
            _add_backward_x(fixture, fields["asxy"], j, i, yx)
            _add_backward_y(fixture, fields["asxy"], j, i, xy)
            _add_forward_y(fixture, fields["asyy"], j, i, yy)


def one_reverse_step(fixture: Fixture, timestep: int) -> dict[str, list[float]]:
    fields = _clone_fields(fixture.fields)
    inject_receiver_residuals(fixture, fields, timestep)
    stress_gsls_transpose(fixture, fields)
    velocity_transpose(fixture, fields)
    return fields


def _forward_cpml(fixture: Fixture, old_fields: dict[str, list[float]],
                  new_fields: dict[str, list[float]], psi_name: str,
                  p: int, coordinate: int, extent: int, group: str,
                  derivative: float) -> float:
    h = _cpml_h(coordinate, extent, fixture.fw)
    if not h:
        return derivative
    profile = fixture.cpml[group]
    psi_new = profile["b"][h] * old_fields[psi_name][p] + profile["a"][h] * derivative
    new_fields[psi_name][p] = psi_new
    return derivative / profile["K"][h] + psi_new


def _dx_forward(fixture: Fixture, field: list[float], j: int, i: int) -> float:
    return (fixture.hc1 * (field[fixture.index(j, i + 1)] - field[fixture.index(j, i)])
            + fixture.hc2 * (field[fixture.index(j, i + 2)] - field[fixture.index(j, i - 1)]))


def _dx_backward(fixture: Fixture, field: list[float], j: int, i: int) -> float:
    return (fixture.hc1 * (field[fixture.index(j, i)] - field[fixture.index(j, i - 1)])
            + fixture.hc2 * (field[fixture.index(j, i + 1)] - field[fixture.index(j, i - 2)]))


def _dy_forward(fixture: Fixture, field: list[float], j: int, i: int) -> float:
    return (fixture.hc1 * (field[fixture.index(j + 1, i)] - field[fixture.index(j, i)])
            + fixture.hc2 * (field[fixture.index(j + 2, i)] - field[fixture.index(j - 1, i)]))


def _dy_backward(fixture: Fixture, field: list[float], j: int, i: int) -> float:
    return (fixture.hc1 * (field[fixture.index(j, i)] - field[fixture.index(j - 1, i)])
            + fixture.hc2 * (field[fixture.index(j + 1, i)] - field[fixture.index(j - 2, i)]))


def _bar_fields(fixture: Fixture, seed: int, *, cpml_only: bool = False) -> dict[str, list[float]]:
    result = {field: [0.0] * fixture.count for field in ALL_FIELDS}
    for field in MAIN_FIELDS:
        tag = ALL_FIELDS.index(field) + seed
        result[field] = _cell_array((fixture.nx, fixture.ny), tag, float32=False)
    if cpml_only:
        for field in CPML_FIELDS:
            result[field] = [0.0] * fixture.count
            axis = "x" if field in (PSXX, PSXYX, PVXX, PVYX) else "y"
            extent = fixture.nx if axis == "x" else fixture.ny
            for j in range(1, fixture.ny + 1):
                for i in range(1, fixture.nx + 1):
                    coordinate = i if axis == "x" else j
                    if _cpml_h(coordinate, extent, fixture.fw):
                        result[field][fixture.index(j, i)] = _pattern(
                            ALL_FIELDS.index(field) + seed, j, i
                        )
    return result


def _dot(left, right) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def _active_cpml_indices(fixture: Fixture, axis: str):
    if axis == "x":
        for j in range(1, fixture.ny + 1):
            for i in range(1, fixture.nx + 1):
                if _cpml_h(i, fixture.nx, fixture.fw):
                    yield fixture.index(j, i)
    else:
        for j in range(1, fixture.ny + 1):
            if _cpml_h(j, fixture.ny, fixture.fw):
                for i in range(1, fixture.nx + 1):
                    yield fixture.index(j, i)


def velocity_dot_product_error(fixture: Fixture) -> float:
    """Check <A_vel stress/psi, y> = <input, A_vel^T y> for FD4+CPML."""
    inputs = _bar_fields(fixture, 31)
    old_names = (PSXX, PSXYX, PSXYY, PSYY)
    old_fields = {name: values.copy() for name, values in inputs.items()}
    forward_fields = _clone_fields(inputs)
    vx_out, vy_out = [0.0] * fixture.count, [0.0] * fixture.count
    m = fixture.material
    for j in range(1, fixture.ny + 1):
        for i in range(1, fixture.nx + 1):
            p = fixture.index(j, i)
            xx = _forward_cpml(fixture, old_fields, forward_fields, PSXX, p, i,
                               fixture.nx, "x_half", _dx_forward(fixture, inputs["asxx"], j, i))
            yx = _forward_cpml(fixture, old_fields, forward_fields, PSXYX, p, i,
                               fixture.nx, "x", _dx_backward(fixture, inputs["asxy"], j, i))
            xy = _forward_cpml(fixture, old_fields, forward_fields, PSXYY, p, j,
                               fixture.ny, "y", _dy_backward(fixture, inputs["asxy"], j, i))
            yy = _forward_cpml(fixture, old_fields, forward_fields, PSYY, p, j,
                               fixture.ny, "y_half", _dy_forward(fixture, inputs["asyy"], j, i))
            vx_out[p] = fixture.dt * m["prip"][p] / fixture.dh * (xx + xy)
            vy_out[p] = fixture.dt * m["prjp"][p] / fixture.dh * (yx + yy)

    bars = _bar_fields(fixture, 79, cpml_only=True)
    left_terms = [vx_out[fixture.index(j, i)] * bars["avx"][fixture.index(j, i)]
                  + vy_out[fixture.index(j, i)] * bars["avy"][fixture.index(j, i)]
                  for j in range(1, fixture.ny + 1) for i in range(1, fixture.nx + 1)]
    for field in old_names:
        axis = "x" if field in (PSXX, PSXYX) else "y"
        for p in _active_cpml_indices(fixture, axis):
            left_terms.append(forward_fields[field][p] * bars[field][p])

    transposed = {field: [0.0] * fixture.count for field in ALL_FIELDS}
    transposed["avx"] = bars["avx"].copy()
    transposed["avy"] = bars["avy"].copy()
    for field in old_names:
        transposed[field] = bars[field].copy()
    velocity_transpose(fixture, transposed)
    right_terms = []
    for field in ("asxx", "asxy", "asyy"):
        right_terms.extend(a * b for a, b in zip(inputs[field], transposed[field]))
    for field in old_names:
        axis = "x" if field in (PSXX, PSXYX) else "y"
        right_terms.extend(old_fields[field][p] * transposed[field][p]
                           for p in _active_cpml_indices(fixture, axis))
    lhs, rhs = math.fsum(left_terms), math.fsum(right_terms)
    return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


def stress_gsls_dot_product_error(fixture: Fixture) -> float:
    """Check <A_stress/GSLS/CPML(v,mem),y> = <input,A^T y>."""
    inputs = {field: [0.0] * fixture.count for field in ALL_FIELDS}
    # Velocity, memory state, and incoming stress-bar fixtures are all
    # deterministic and independent of the one-step hash fixture.
    for field, tag in (("avx", 101), ("avy", 102), ("ar", 103),
                       ("ap", 104), ("aq", 105)):
        inputs[field] = _cell_array((fixture.nx, fixture.ny), tag, float32=False)
    for field in (PVXX, PVYX, PVXY, PVYY):
        axis = "x" if field in (PVXX, PVYX) else "y"
        extent = fixture.nx if axis == "x" else fixture.ny
        for j in range(1, fixture.ny + 1):
            for i in range(1, fixture.nx + 1):
                coordinate = i if axis == "x" else j
                if _cpml_h(coordinate, extent, fixture.fw):
                    inputs[field][fixture.index(j, i)] = _pattern(
                        ALL_FIELDS.index(field) + 107, j, i
                    )
    old_fields = _clone_fields(inputs)
    forward_fields = _clone_fields(inputs)
    stress_inc = {field: [0.0] * fixture.count for field in ("asxx", "asyy", "asxy")}
    memory_new = {field: [0.0] * fixture.count for field in ("ar", "ap", "aq")}
    m = fixture.material
    # prepare_update_s_visc_PSV assigns both etaip and etajm from peta, so
    # this supported L=1 CPU configuration has bip==bjm and cip==cjm.
    b, c, dt2 = fixture.bjm1, fixture.cjm1, fixture.dt * 0.5
    for j in range(1, fixture.ny + 1):
        for i in range(1, fixture.nx + 1):
            p = fixture.index(j, i)
            vxx = _forward_cpml(fixture, old_fields, forward_fields, PVXX, p, i,
                                fixture.nx, "x", _dx_backward(fixture, inputs["avx"], j, i) / fixture.dh)
            vyx = _forward_cpml(fixture, old_fields, forward_fields, PVYX, p, i,
                                fixture.nx, "x_half", _dx_forward(fixture, inputs["avy"], j, i) / fixture.dh)
            vxy = _forward_cpml(fixture, old_fields, forward_fields, PVXY, p, j,
                                fixture.ny, "y_half", _dy_forward(fixture, inputs["avx"], j, i) / fixture.dh)
            vyy = _forward_cpml(fixture, old_fields, forward_fields, PVYY, p, j,
                                fixture.ny, "y", _dy_backward(fixture, inputs["avy"], j, i) / fixture.dh)
            div, shear = vxx + vyy, vxy + vyx
            r_old, p_old, q_old = inputs["ar"][p], inputs["ap"][p], inputs["aq"][p]
            r_new = b * (r_old * c - m["dip"][p] * shear)
            p_new = b * (p_old * c - m["e"][p] * div + 2.0 * m["d"][p] * vyy)
            q_new = b * (q_old * c - m["e"][p] * div + 2.0 * m["d"][p] * vxx)
            memory_new["ar"][p], memory_new["ap"][p], memory_new["aq"][p] = r_new, p_new, q_new
            stress_inc["asxy"][p] = m["fipjp"][p] * shear + dt2 * (r_old + r_new)
            stress_inc["asxx"][p] = m["g"][p] * div - 2.0 * m["f"][p] * vyy + dt2 * (p_old + p_new)
            stress_inc["asyy"][p] = m["g"][p] * div - 2.0 * m["f"][p] * vxx + dt2 * (q_old + q_new)

    bars = _bar_fields(fixture, 131, cpml_only=True)
    # Distinct output bars for every stress, memory, and CPML field.
    for field, tag in (("asxx", 139), ("asyy", 140), ("asxy", 141),
                       ("ar", 142), ("ap", 143), ("aq", 144)):
        bars[field] = _cell_array((fixture.nx, fixture.ny), tag, float32=False)
    left_terms = []
    for field in ("asxx", "asyy", "asxy"):
        left_terms.extend(stress_inc[field][p] * bars[field][p]
                          for j in range(1, fixture.ny + 1)
                          for i in range(1, fixture.nx + 1)
                          for p in (fixture.index(j, i),))
    for field in ("ar", "ap", "aq"):
        left_terms.extend(memory_new[field][p] * bars[field][p]
                          for j in range(1, fixture.ny + 1)
                          for i in range(1, fixture.nx + 1)
                          for p in (fixture.index(j, i),))
    for field in (PVXX, PVYX, PVXY, PVYY):
        axis = "x" if field in (PVXX, PVYX) else "y"
        left_terms.extend(forward_fields[field][p] * bars[field][p]
                          for p in _active_cpml_indices(fixture, axis))

    transposed = {field: [0.0] * fixture.count for field in ALL_FIELDS}
    for field in ("asxx", "asyy", "asxy", "ar", "ap", "aq",
                  PVXX, PVYX, PVXY, PVYY):
        transposed[field] = bars[field].copy()
    stress_gsls_transpose(fixture, transposed)
    right_terms = []
    for field in ("avx", "avy"):
        right_terms.extend(a * b for a, b in zip(inputs[field], transposed[field]))
    for field in ("ar", "ap", "aq"):
        right_terms.extend(inputs[field][p] * transposed[field][p]
                           for j in range(1, fixture.ny + 1)
                           for i in range(1, fixture.nx + 1)
                           for p in (fixture.index(j, i),))
    for field in (PVXX, PVYX, PVXY, PVYY):
        axis = "x" if field in (PVXX, PVYX) else "y"
        right_terms.extend(old_fields[field][p] * transposed[field][p]
                           for p in _active_cpml_indices(fixture, axis))
    lhs, rhs = math.fsum(left_terms), math.fsum(right_terms)
    return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


def transpose_errors(fixture: Fixture) -> dict[str, float]:
    return {
        "stress_gsls_cpml": stress_gsls_dot_product_error(fixture),
        "velocity_cpml": velocity_dot_product_error(fixture),
    }
