"""Independent slip-interface reference and FD4 consistency investigation.

No DENISE kernel or existing material helper is called. This diagnostic does
not define a replacement production interface stencil. Coordinates: y down,
interface y=0, fluid y<0, tensile-positive stress, exp(i*kx*x-i*omega*t).
"""

from functools import lru_cache
import math

import numpy as np


# Predeclared before any numerical interface execution (see companion doc).
REFINEMENT_H = (1.0, 0.5, 0.25, 0.125)
INTERFACE_MAX_ERROR = 0.02
REFINEMENT_MAX_RATIO = 0.8
REFINEMENT_NOISE_FLOOR = 0.002
FREQUENCY = 0.25
FLUID_RHO, FLUID_CP = 1.0, 2.0
SOLID_RHO, SOLID_CP, SOLID_CS = 1.5, 3.0, 1.5


def slip_coefficients(angle_degrees):
    """Return reflected pressure, transmitted P/SV particle-speed amplitudes.

    Incident pressure is 1. P polarization is (sin alpha, cos alpha); SV is
    (cos beta, -sin beta). Three equations impose only normal velocity,
    normal traction and zero shear traction. Tangential velocity is FREE.
    """
    theta = math.radians(angle_degrees)
    alpha = math.asin(SOLID_CP / FLUID_CP * math.sin(theta))
    beta = math.asin(SOLID_CS / FLUID_CP * math.sin(theta))
    mu = SOLID_RHO * SOLID_CS**2
    lam = SOLID_RHO * (SOLID_CP**2 - 2 * SOLID_CS**2)
    v = math.cos(theta) / (FLUID_RHO * FLUID_CP)
    cp_normal = (lam + 2*mu*math.cos(alpha)**2) / SOLID_CP
    cs_normal = 2*mu*math.sin(beta)*math.cos(beta) / SOLID_CS
    matrix = np.array([
        [1, -cp_normal, cs_normal],
        [v, math.cos(alpha), -math.sin(beta)],
        [0, -2*mu*math.sin(alpha)*math.cos(alpha)/SOLID_CP,
         -mu*math.cos(2*beta)/SOLID_CS],
    ])
    result = np.linalg.solve(matrix, [-1, v, 0])
    return result, matrix, np.array([-1, v, 0])


def fd4_symbol(k, h):
    return 2/h * (9/8*np.sin(k*h/2) - 1/24*np.sin(3*k*h/2))


def vertical_wavenumber(omega, kx, speed, h, dt):
    """Bulk discrete dispersion root; removes propagation phase, not error fit."""
    temporal = 2*math.sin(omega*dt/2)/dt
    target = math.sqrt((temporal/speed)**2 - fd4_symbol(kx, h)**2)
    lo, hi = 0.0, math.pi/h
    for _ in range(60):
        mid = (lo+hi)/2
        if fd4_symbol(mid, h) < target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def derivative_y(a, h, forward):
    if forward:
        return (9/8*(np.roll(a, -1, axis=0)-a)
                - 1/24*(np.roll(a, -2, axis=0)-np.roll(a, 1, axis=0)))/h
    return (9/8*(a-np.roll(a, 1, axis=0))
            - 1/24*(np.roll(a, -1, axis=0)-np.roll(a, 2, axis=0)))/h


def require_interface_refinement(results):
    """Frozen physical gate: future adapter supplies actual production results."""
    assert tuple(row["h"] for row in results) == REFINEMENT_H
    angle = results[0]["angle"]
    assert angle in (0, 20) and all(row["angle"] == angle for row in results)
    expected, _, _ = slip_coefficients(angle)
    scales = np.where(abs(expected) > 1e-12, abs(expected), 1/(FLUID_RHO*FLUID_CP))
    errors = []
    for row in results:
        measured = np.asarray(row["measured"])
        assert measured.shape == (3,) and np.isfinite(measured).all()
        assert np.isfinite(row["peak"]) and row["peak"] > 0
        # Do not trust a future adapter's claimed "error" or expected values.
        errors.append(float(max(abs(measured-expected)/scales)))
    assert errors[-1] <= INTERFACE_MAX_ERROR, results
    for coarse, fine in zip(errors, errors[1:]):
        assert fine <= max(REFINEMENT_MAX_RATIO*coarse, REFINEMENT_NOISE_FLOOR), results


@lru_cache(maxsize=None)
def interface_measurement(h, angle_degrees, *, window=140.0, half_domain=240.0,
                          pulse="narrow"):
    """Complex single-horizontal-Fourier-mode FD4 experiment + fluid control.

    This is an independent study of the *proposed unmodified* staggered
    zero-corner scheme, not the physical oracle. The physical oracle above
    is continuum traction/slip algebra. Geometry, source spectrum, Fourier
    window and material are fixed for all h. No production results are used.

    Domain [-240,240], source y=-64, observation y approximately +/-32.
    Source is narrow-band isotropic stress, after velocity/stress steps.
    Optional Ricker is a documented initial measurement diagnostic only.
    Window [0,140];
    earliest outer-wrap returning wave cannot reach an observation by 140.
    Homogeneous control divides out source spectrum (no geometric spreading
    for a plane mode). Both velocity components are sampled at their actual
    staggered positions/times, not treated as collocated.
    """
    dt = 0.04*h
    nt = round(window/dt)
    y = -half_domain + (np.arange(round(2*half_domain/h))+0.5)*h
    ny = len(y)
    omega = 2*math.pi*FREQUENCY
    kx = omega/FLUID_CP * math.sin(math.radians(angle_degrees))
    ikx = 1j*fd4_symbol(kx, h)
    # Column 0 is the homogeneous acoustic control; column 1 the interface.
    rho = np.ones((ny, 2))*FLUID_RHO
    lam = np.ones((ny, 2))*(FLUID_RHO*FLUID_CP**2)
    mu = np.zeros((ny, 2))
    solid = y > 0
    rho[solid, 1] = SOLID_RHO
    mu[solid, 1] = SOLID_RHO*SOLID_CS**2
    lam[solid, 1] = SOLID_RHO*(SOLID_CP**2-2*SOLID_CS**2)
    ry = 2/(rho+np.roll(rho, -1, axis=0))
    # Each pair occurs twice among four corner contributors in this plane case.
    other = np.roll(mu, -1, axis=0)
    hcorner = np.zeros_like(mu)
    active = (mu > 0) & (other > 0)
    hcorner[active] = 4/(2/mu[active]+2/other[active])
    vx, vy, xx, yy, xy = (np.zeros((ny, 2), complex) for _ in range(5))
    source = int(np.argmin(abs(y+64)))
    rf = int(np.argmin(abs(y+32)))
    rs = int(np.argmin(abs(y-32)))
    pressure = np.zeros((2,), complex)
    velocity = np.zeros((2,), complex)
    peak = 0.0
    late_pressure_peak = 0.0
    for n in range(nt):
        vx += dt/rho*(ikx*xx+derivative_y(xy, h, False))
        vy += dt*ry*(ikx*xy+derivative_y(yy, h, True))
        velocity += dt*np.array([vx[rs, 1], vy[rs, 1]])*np.exp(1j*omega*(n+0.5)*dt)
        ex, ey = ikx*vx, derivative_y(vy, h, False)
        xx += dt*(lam*(ex+ey)+2*mu*ex)
        yy += dt*(lam*(ex+ey)+2*mu*ey)
        xy += dt*hcorner*(derivative_y(vx, h, True)+ikx*vy)
        if pulse == "ricker":
            tau = math.pi*FREQUENCY*((n+1)*dt-16)
            source_increment = (1-2*tau*tau)*math.exp(-tau*tau)*dt
        elif pulse == "narrow":
            t = (n+1)*dt-40
            source_increment = math.exp(-0.5*(t/8)**2)*math.cos(omega*t)*dt
        else:
            raise ValueError("unknown source pulse")
        xx[source, :] += source_increment
        yy[source, :] += source_increment
        pressure += dt*(-yy[rf, :])*np.exp(1j*omega*(n+1)*dt)
        if (n+1)*dt > window-20:
            late_pressure_peak = max(late_pressure_peak, float(np.max(abs(yy[rf, :]))))
        peak = max(peak, float(np.max(abs(vx))), float(np.max(abs(vy))))
    kf = vertical_wavenumber(omega, kx, FLUID_CP, h, dt)
    kp = vertical_wavenumber(omega, kx, SOLID_CP, h, dt)
    ks = vertical_wavenumber(omega, kx, SOLID_CS, h, dt)
    incident = pressure[0]*np.exp(-1j*kf*y[rf])
    reflected = (pressure[1]-pressure[0])/pressure[0]*np.exp(2j*kf*y[rf])
    theta = math.radians(angle_degrees)
    alpha = math.asin(SOLID_CP/FLUID_CP*math.sin(theta))
    beta = math.asin(SOLID_CS/FLUID_CP*math.sin(theta))
    polarizations = np.array([
        [math.sin(alpha)*np.exp(1j*kp*y[rs]), math.cos(beta)*np.exp(1j*ks*y[rs])],
        [math.cos(alpha)*np.exp(1j*kp*(y[rs]+h/2)),
         -math.sin(beta)*np.exp(1j*ks*(y[rs]+h/2))],
    ])
    transmitted = np.linalg.solve(polarizations, velocity/incident)
    measured = np.array([reflected, *transmitted])
    expected, _, _ = slip_coefficients(angle_degrees)
    # Normalize nonzero modes individually. At normal incidence SV must be 0.
    scales = np.where(abs(expected) > 1e-12, abs(expected), 1/(FLUID_RHO*FLUID_CP))
    errors = abs(measured-expected)/scales
    return dict(h=h, angle=angle_degrees, measured=measured, expected=expected,
                errors=errors, max_error=float(max(errors)), peak=peak,
                finite=bool(np.isfinite(measured).all()), observation_y=(y[rf], y[rs]),
                window=window, half_domain=half_domain, pulse=pulse,
                late_pressure_peak=late_pressure_peak)
