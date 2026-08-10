from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RayPath:
    boundary_x_m: float
    incident_distance_m: float
    outgoing_distance_m: float
    travel_time_s: float
    horizontal_slowness_s_m: float


def _solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
    size = len(rhs)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-14:
            raise ValueError("Singular elastic coefficient system")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def two_segment_ray(
    source_m: tuple[float, float],
    receiver_m: tuple[float, float],
    *,
    boundary_y_m: float,
    incident_velocity_m_s: float,
    outgoing_velocity_m_s: float,
) -> RayPath:
    """Solve the stationary-time ray through a horizontal boundary."""
    xs, ys = source_m
    xr, yr = receiver_m
    vertical_in = abs(ys - boundary_y_m)
    vertical_out = abs(yr - boundary_y_m)
    if vertical_in == 0.0 or vertical_out == 0.0:
        raise ValueError("Ray endpoints must not lie on the boundary")
    low, high = sorted((xs, xr))
    if low == high:
        boundary_x = low
    else:
        def derivative(x: float) -> float:
            dx_in = x - xs
            dx_out = xr - x
            return (
                dx_in / (incident_velocity_m_s * math.hypot(dx_in, vertical_in))
                - dx_out / (outgoing_velocity_m_s * math.hypot(dx_out, vertical_out))
            )

        for _ in range(100):
            midpoint = 0.5 * (low + high)
            if derivative(midpoint) < 0.0:
                low = midpoint
            else:
                high = midpoint
        boundary_x = 0.5 * (low + high)
    incident_distance = math.hypot(boundary_x - xs, vertical_in)
    outgoing_distance = math.hypot(xr - boundary_x, vertical_out)
    horizontal_slowness = abs(boundary_x - xs) / (
        incident_velocity_m_s * incident_distance
    )
    return RayPath(
        boundary_x,
        incident_distance,
        outgoing_distance,
        incident_distance / incident_velocity_m_s
        + outgoing_distance / outgoing_velocity_m_s,
        horizontal_slowness,
    )


def _traction(
    polarization: tuple[float, float],
    slowness: tuple[float, float],
    *,
    vp_m_s: float,
    vs_m_s: float,
    density_kg_m3: float,
) -> tuple[float, float]:
    ux, uy = polarization
    px, py = slowness
    mu = density_kg_m3 * vs_m_s * vs_m_s
    lam = density_kg_m3 * (vp_m_s * vp_m_s - 2.0 * vs_m_s * vs_m_s)
    txy = mu * (py * ux + px * uy)
    tyy = lam * (px * ux + py * uy) + 2.0 * mu * py * uy
    return txy, tyy


def free_surface_p_coefficients(
    incidence_angle_rad: float,
    *,
    vp_m_s: float,
    vs_m_s: float,
    density_kg_m3: float,
) -> dict[str, float]:
    """Displacement coefficients for an upward incident P plane wave."""
    p = math.sin(incidence_angle_rad) / vp_m_s
    q_p = math.sqrt(1.0 / (vp_m_s * vp_m_s) - p * p)
    if p * vs_m_s >= 1.0:
        raise ValueError("Converted SV is evanescent")
    q_s = math.sqrt(1.0 / (vs_m_s * vs_m_s) - p * p)
    incident = (vp_m_s * p, -vp_m_s * q_p)
    reflected_p = (vp_m_s * p, vp_m_s * q_p)
    reflected_sv = (-vs_m_s * q_s, vs_m_s * p)
    ti = _traction(incident, (p, -q_p), vp_m_s=vp_m_s, vs_m_s=vs_m_s,
                   density_kg_m3=density_kg_m3)
    tp = _traction(reflected_p, (p, q_p), vp_m_s=vp_m_s, vs_m_s=vs_m_s,
                   density_kg_m3=density_kg_m3)
    ts = _traction(reflected_sv, (p, q_s), vp_m_s=vp_m_s, vs_m_s=vs_m_s,
                   density_kg_m3=density_kg_m3)
    rpp, rps = _solve_linear([[tp[0], ts[0]], [tp[1], ts[1]]], [-ti[0], -ti[1]])
    return {
        "horizontal_slowness_s_m": p,
        "reflected_p_displacement": rpp,
        "reflected_sv_displacement": rps,
        "reflected_sv_angle_rad": math.asin(p * vs_m_s),
    }


def zoeppritz_p_coefficients(
    incidence_angle_rad: float,
    *,
    vp1_m_s: float,
    vs1_m_s: float,
    rho1_kg_m3: float,
    vp2_m_s: float,
    vs2_m_s: float,
    rho2_kg_m3: float,
) -> dict[str, float]:
    """Solve displacement/traction continuity for downward incident P."""
    p = math.sin(incidence_angle_rad) / vp1_m_s
    velocities = (vp1_m_s, vs1_m_s, vp2_m_s, vs2_m_s)
    if any(abs(p * velocity) >= 1.0 for velocity in velocities):
        raise ValueError("Critical/evanescent angle is outside this real-valued solver")
    qp1, qs1, qp2, qs2 = (
        math.sqrt(1.0 / (velocity * velocity) - p * p) for velocity in velocities
    )
    incident = (vp1_m_s * p, vp1_m_s * qp1)
    rp = (vp1_m_s * p, -vp1_m_s * qp1)
    rs = (vs1_m_s * qs1, vs1_m_s * p)
    tp = (vp2_m_s * p, vp2_m_s * qp2)
    ts = (-vs2_m_s * qs2, vs2_m_s * p)
    ti_tr = _traction(incident, (p, qp1), vp_m_s=vp1_m_s, vs_m_s=vs1_m_s,
                      density_kg_m3=rho1_kg_m3)
    bases = (
        (rp, (p, -qp1), vp1_m_s, vs1_m_s, rho1_kg_m3, 1.0),
        (rs, (p, -qs1), vp1_m_s, vs1_m_s, rho1_kg_m3, 1.0),
        (tp, (p, qp2), vp2_m_s, vs2_m_s, rho2_kg_m3, -1.0),
        (ts, (p, qs2), vp2_m_s, vs2_m_s, rho2_kg_m3, -1.0),
    )
    columns = []
    for polarization, slowness, vp, vs, rho, sign in bases:
        traction = _traction(polarization, slowness, vp_m_s=vp, vs_m_s=vs,
                             density_kg_m3=rho)
        columns.append((sign * polarization[0], sign * polarization[1],
                        sign * traction[0], sign * traction[1]))
    matrix = [[columns[column][row] for column in range(4)] for row in range(4)]
    solution = _solve_linear(
        matrix, [-incident[0], -incident[1], -ti_tr[0], -ti_tr[1]]
    )
    return {
        "horizontal_slowness_s_m": p,
        "reflected_p_displacement": solution[0],
        "reflected_sv_displacement": solution[1],
        "transmitted_p_displacement": solution[2],
        "transmitted_sv_displacement": solution[3],
        "reflected_sv_angle_rad": math.asin(p * vs1_m_s),
        "transmitted_p_angle_rad": math.asin(p * vp2_m_s),
        "transmitted_sv_angle_rad": math.asin(p * vs2_m_s),
    }
