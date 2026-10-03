"""FP64 zero-shear mathematics, independent of production and old M9 helpers.

Only background classification is canonical FP32. Material maps and waves
are then independently evaluated in FP64. No public production API is added.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Material:
    rho: np.ndarray
    lam: np.ndarray
    mu: np.ndarray
    fluid: np.ndarray
    rx: np.ndarray
    ry: np.ndarray
    corner: np.ndarray


def material(rho, lam, mu):
    arrays = [np.array(a, dtype=np.float32, copy=True).astype(np.float64)
              for a in (rho, lam, mu)]
    rho, lam, mu = arrays
    if rho.ndim != 2 or any(a.shape != rho.shape for a in arrays):
        raise ValueError("material shape")
    if np.any(mu < 0) or not np.isfinite(mu).all():
        raise ValueError("negative/nonfinite mu")
    fluid = mu == 0
    # This oracle's physical fixture envelope, NOT a new universal validator.
    if not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError("positive finite rho required")
    if not np.isfinite(lam).all() or np.any(lam[fluid] <= 0):
        raise ValueError("fluid needs finite positive compressional stiffness")
    ny, nx = rho.shape
    rx, ry, corner = (np.zeros_like(rho) for _ in range(3))
    for j in range(ny):
        for i in range(nx):
            ip, jp = (i+1) % nx, (j+1) % ny
            rx[j, i] = 2/(rho[j, i]+rho[j, ip])
            ry[j, i] = 2/(rho[j, i]+rho[jp, i])
            values = [mu[j, i], mu[j, ip], mu[jp, i], mu[jp, ip]]
            if all(v > 0 for v in values):
                corner[j, i] = 4/(1/values[0]+1/values[1]+1/values[2]+1/values[3])
    for a in (*arrays, fluid, rx, ry, corner):
        a.flags.writeable = False
    return Material(rho, lam, mu, fluid, rx, ry, corner)


def require_direction(m, dm):
    if dm.shape != m.mu.shape or not np.isfinite(dm).all() or np.any(dm[m.fluid] != 0):
        raise ValueError("nonzero fluid dMu is outside the fixed manifold")


def require_trial(m, mu):
    trial = np.asarray(mu, np.float32)
    if np.any(trial < 0) or not np.isfinite(trial).all() or not np.array_equal(trial == 0, m.fluid):
        raise ValueError("trial changes prepared fluid classification")


def corner_jvp(m, dm):
    require_direction(m, dm)
    out = np.zeros_like(m.mu)
    ny, nx = out.shape
    for j in range(ny):
        for i in range(nx):
            if m.corner[j, i] == 0:
                continue
            points = [(j, i), (j, (i+1) % nx), ((j+1) % ny, i), ((j+1) % ny, (i+1) % nx)]
            out[j, i] = m.corner[j, i]**2/4 * math.fsum(dm[p]/m.mu[p]**2 for p in points)
    return out


def corner_vjp(m, bar):
    out = np.zeros_like(m.mu)
    ny, nx = out.shape
    for j in range(ny):
        for i in range(nx):
            if m.corner[j, i] == 0:
                continue
            common = bar[j, i]*m.corner[j, i]**2/4
            for p in ((j, i), (j, (i+1) % nx), ((j+1) % ny, i), ((j+1) % ny, (i+1) % nx)):
                out[p] += common/m.mu[p]**2
    out[m.fluid] = 0.0
    return out


def directions(m):
    y, x = np.indices(m.mu.shape)
    dl = m.lam*(0.007*np.sin(0.3*x+0.4*y)+0.003)
    dm = m.mu*(0.009*np.cos(0.2*x-0.5*y)+0.002)
    near = np.zeros_like(m.fluid)
    for axis in (0, 1):
        near |= m.fluid != np.roll(m.fluid, 1, axis)
        near |= m.fluid != np.roll(m.fluid, -1, axis)
    zero = np.zeros_like(dl)
    return {"fluid_lambda": (dl*m.fluid, zero),
            "solid_lambda": (dl*~m.fluid, zero),
            "solid_mu": (zero, dm), "joint": (dl, dm),
            "interface": (dl*near, dm*near)}


def dense_columns(m):
    # All lambda; only solid mu. Fluid mu is NOT a zero derivative column.
    for channel, mask in (("lambda", np.ones_like(m.fluid)), ("mu", ~m.fluid)):
        for j, i in np.argwhere(mask):
            yield channel, int(j), int(i)


def surface(lam, mu, dl, dm):
    d = lam+2*mu
    alpha, a = lam/d, 4*mu*(lam+mu)/d
    da = (2*mu*dl-2*lam*dm)/d**2
    d_a = (4*mu**2*dl+4*(lam**2+2*lam*mu+2*mu**2)*dm)/d**2
    return d, alpha, a, da, d_a


def require_source_row(row):
    if row < 1:
        raise ValueError("source must be below first surface row")


def derivative(a, axis, forward, h=1):
    if forward:
        return (9/8*(np.roll(a, -1, axis)-a)
                - 1/24*(np.roll(a, -2, axis)-np.roll(a, 1, axis)))/h
    return (9/8*(a-np.roll(a, 1, axis))
            - 1/24*(np.roll(a, -1, axis)-np.roll(a, 2, axis)))/h


def profile(n, h, dt, fw, speed, half=False, free_top=False):
    """Independent scalar coordinate evaluation of M9 polynomial CPML law."""
    k, a, b = np.ones(n), np.zeros(n), np.ones(n)
    if fw == 0:
        return k, a, b
    thick = fw*h
    d0 = -3*speed*math.log(1e-3)/(2*thick)
    for i in range(n):
        x = (i+0.5*half)*h
        dist = max(thick-x if not free_top else -math.inf, x-((n-1)*h-thick))
        if dist < 0:
            continue
        u = dist/thick
        damp = d0*u*u
        alpha = max(0, math.pi*0.1*(1-u))
        b[i] = math.exp(-(damp+alpha)*dt)
        if damp > 1e-6:
            a[i] = damp*(b[i]-1)/(damp+alpha)
    return k, a, b


def cpml(q, old, p):
    k, a, b = p
    new = b*old+a*q
    return q/k+new, new


def cpml_t(bar_q, bar_new, p):
    k, a, b = p
    combined = bar_q+bar_new
    return bar_q/k+a*combined, b*combined


def prepared_source(nt, dt, fc=0.1, t0=15):
    t = np.arange(1, nt+1)*dt
    tau = math.pi*(t-t0)/(1.5/fc)
    wave = (1-4*tau*tau)*np.exp(-2*tau*tau)
    signal = np.zeros(nt+2)
    for n in range(1, nt+1):
        signal[n] = signal[n-1]+dt*wave[n-1]
    samples = np.empty(nt)
    samples[0] = signal[2]/dt
    samples[1:-1] = (signal[3:nt+1]-signal[1:nt-1])/dt
    samples[-1] = -signal[nt-1]/dt
    return samples


def acoustic_forward(nx=24, ny=20, dt=0.1, nt=120, source=None):
    """Pressure/velocity acoustic FD4; no elastic stresses or corner helpers."""
    pressure, vx, vy = (np.zeros((ny, nx)) for _ in range(3))
    if source is None:
        source = prepared_source(nt, dt, fc=0.5, t0=3)
    data = np.zeros((nt, 4, 2))
    receivers = [(ny//2, nx//2+3), (ny//2+3, nx//2), (ny//2-2, nx//2-2), (1, 1)]
    for n, load in enumerate(source):
        vx -= dt*derivative(pressure, 1, True)
        vy -= dt*derivative(pressure, 0, True)
        for r, p in enumerate(receivers):
            data[n, r] = vx[p], vy[p]
        pressure -= 4*dt*(derivative(vx, 1, False)+derivative(vy, 0, False))
        pressure[ny//2, nx//2] -= load
    return data, pressure, vx, vy


def surface_ghosts(a, kind, *, alpha=None, qxx=None, qyx=None, dt=0.1):
    ny = len(a)
    g = np.empty((ny+4, a.shape[1]))
    g[2:ny+2] = a
    for m in (1, 2):
        g[ny+1+m] = a[m-1]
        if kind == "syy":
            g[2-m] = -a[m]
        elif kind == "sxy":
            g[2-m] = -a[m-1]
        elif kind == "vy":
            g[2-m] = a[m-1]+(2*m-1)/dt*alpha*qxx[0]
        elif kind == "vx":
            w = np.array([35, -35, 21, -5])/16
            g[2-m] = a[m]+2*m/dt*(w@qyx[:4])
        else:
            raise ValueError(kind)
    return g


def y_with_surface(a, kind, **kwargs):
    ny = len(a)
    g = surface_ghosts(a, kind, **kwargs)
    offsets = ((1, 9/8), (0, -9/8), (2, -1/24), (-1, 1/24)) if kind in ("syy", "vx") else (
        (0, 9/8), (-1, -9/8), (1, -1/24), (-2, 1/24))
    return sum(w*g[2+o:2+o+ny] for o, w in offsets)


def require_homogeneous_products(m, data, corner, sxy, reference_data):
    """Future production supplies actual products; no empty/permanent RED hook."""
    assert np.isfinite(data).all() and np.linalg.norm(data) > 0
    assert np.linalg.norm(data-reference_data)/np.linalg.norm(reference_data) <= 1e-5
    np.testing.assert_array_equal(corner[m.corner == 0], 0.)
    assert not np.signbit(corner[m.corner == 0]).any()
    np.testing.assert_array_equal(sxy, 0.)


def elastic_forward(m, *, nt=120, dt=0.1, fw=0, free_surface=False, source=None):
    """Independent constitutive FD4 study with 8 scalar CPML states.

    Used to corroborate material/boundary algebra; the separate pressure-only
    acoustic solver is the homogeneous-fluid data authority. Not production J.
    DH=1, prepared source at (ny//2,nx//2), or y=20 for NY=64 interface cases.
    """
    ny, nx = m.mu.shape
    if free_surface and ny//2 == 0:
        raise ValueError("source must be below first surface row")
    speed = float(np.sqrt(np.max((m.lam+2*m.mu)/m.rho)))
    profiles = {}
    for axis, size in (("x", nx), ("y", ny)):
        for half in (False, True):
            p = profile(size, 1, dt, fw, speed, half, axis == "y" and free_surface)
            shape = (1, nx) if axis == "x" else (ny, 1)
            profiles[axis+str(half)] = tuple(a.reshape(shape) for a in p)
    state = np.zeros((5, ny, nx))
    memory = np.zeros((8, ny, nx))
    vx, vy, xx, yy, xy = state
    if source is None:
        source = prepared_source(nt, dt, fc=0.5 if nt == 120 else 0.1, t0=3 if nt == 120 else 15)
    data = np.zeros((nt, 4, 2))
    receivers = [(ny//2, nx//2+3), (ny//2+3, nx//2), (ny//2-2, nx//2-2), (1, 1)]
    source_j = 20 if ny == 64 else ny//2
    if free_surface:
        require_source_row(source_j)
    strains = np.zeros((4, ny, nx))
    peak_energy = 0.0
    late_energy = []
    peak_memory = 0.0
    def corrected(q, which, name):
        new, memory[which] = cpml(dt*q, memory[which], profiles[name])
        return new
    for n, load in enumerate(source):
        if free_surface:
            yy[0] = 0
            dxy = y_with_surface(xy, "sxy")
            dyy = y_with_surface(yy, "syy")
        else:
            dxy, dyy = derivative(xy, 0, False), derivative(yy, 0, True)
        vx += m.rx*(corrected(derivative(xx, 1, True), 0, "xTrue")+corrected(dxy, 1, "yFalse"))
        vy += m.ry*(corrected(derivative(xy, 1, False), 2, "xFalse")+corrected(dyy, 3, "yTrue"))
        for r, p in enumerate(receivers):
            data[n, r] = vx[p], vy[p]
        ex = corrected(derivative(vx, 1, False), 4, "xFalse")
        yx = corrected(derivative(vy, 1, True), 5, "xTrue")
        if free_surface:
            alpha = m.lam[0]/(m.lam[0]+2*m.mu[0])
            ey = corrected(y_with_surface(vy, "vy", alpha=alpha, qxx=ex, dt=dt), 7, "yFalse")
            x_y = corrected(y_with_surface(vx, "vx", qyx=yx, dt=dt), 6, "yTrue")
        else:
            ey = corrected(derivative(vy, 0, False), 7, "yFalse")
            x_y = corrected(derivative(vx, 0, True), 6, "yTrue")
        strains[:] = ex, yx, x_y, ey  # VXX,VYX,VXY,VYY; NEVER zero fluid strains.
        bulk_x = m.lam*(ex+ey)+2*m.mu*ex
        bulk_y = m.lam*(ex+ey)+2*m.mu*ey
        xy += m.corner*(yx+x_y)
        if free_surface:
            a = 4*m.mu[0]*(m.lam[0]+m.mu[0])/(m.lam[0]+2*m.mu[0])
            xx[1:] += bulk_x[1:]
            yy[1:] += bulk_y[1:]
            xx[0] += a*ex[0]
            yy[0] = 0
        else:
            xx += bulk_x
            yy += bulk_y
        xx[source_j, nx//2] += load
        yy[source_j, nx//2] += load
        # Positive diagnostic norm, not a claimed solid compliance energy.
        energy = float(np.sum(m.rho*(vx*vx+vy*vy)+(xx*xx+yy*yy+2*xy*xy)/(m.lam+2*m.mu)))
        peak_energy = max(peak_energy, energy)
        if n >= 3*nt//4:
            late_energy.append(energy)
        peak_memory = max(peak_memory, float(np.max(abs(memory))))
    return dict(data=data, state=state, strain=strains, memory=memory,
                peak_memory=peak_memory, finite=bool(np.isfinite(state).all() and np.isfinite(data).all()),
                late_energy_ratio=max(late_energy)/peak_energy, profiles=profiles)
