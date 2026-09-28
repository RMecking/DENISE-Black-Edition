"""Independent FP64 reference for the elastic P/SV Born pair.

The implementation uses only NumPy and direct staggered-grid equations.  It
does not import DENISE kernels.  Array axes are (y, x), with physical indices
reported in the DENISE one-based convention.  See the companion oracle tests
for the frozen experiments and numerical acceptance contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np


FD4 = (9.0 / 8.0, -1.0 / 24.0)
VXX, VYX, VXY, VYY = range(4)
U = np.finfo(np.float64).eps / 2.0


class PMLProfile(NamedTuple):
    kappa: np.ndarray
    a: np.ndarray
    b: np.ndarray


@dataclass(frozen=True)
class Experiment:
    name: str
    nx: int
    ny: int
    dh: float
    dt: float
    nt: int
    fw: int
    cpml: bool
    rho: np.ndarray
    lam: np.ndarray
    mu: np.ndarray
    sources: tuple[tuple[int, int], ...]  # (i,j), one-based DENISE indices
    receivers: tuple[tuple[int, int], ...]  # (i,j), one-based DENISE indices
    fc: float = 15.0
    source_t0: float = 0.1
    source_amplitude: float = 1.0e6
    pml_reflection: float = 1.0e-3
    pml_power: float = 2.0
    pml_kmax: float = 1.0
    pml_fpml: float = 15.0


@dataclass
class BackgroundRun:
    data: np.ndarray  # (time, receiver, [vx,vy])
    strain: np.ndarray | None  # (time, [VXX,VYX,VXY,VYY], y, x), post-CPML
    pml_memory_peak: float
    pml_wave_energy_peak: float
    pml_energy_fraction_peak: float


@dataclass
class BornImage:
    image_lambda_raw: np.ndarray
    image_mu_raw: np.ndarray


def make_experiment(name: str = "interior") -> Experiment:
    """Build either the boundary-isolated microcase or active-CPML fixture."""
    if name == "interior":
        nx, ny, fw, dt, nt, cpml = 41, 37, 0, 5.0e-4, 24, False
        sources = ((21, 19),)
        receivers = ((17, 19), (21, 19), (25, 19), (21, 16), (23, 22))
        source_t0 = 2.0e-3
    elif name == "active_cpml":
        nx, ny, fw, dt, nt, cpml = 64, 56, 8, 5.0e-4, 700, True
        # Pair indices about the true NX=64 centre (x=32.5 in one-based cells).
        sources = ((20, 12), (45, 12))
        receivers = tuple((i, 12) for i in (12, 16, 20, 24, 28, 37, 41, 45, 49, 53))
        source_t0 = 1.5 / 15.0
    else:
        raise ValueError(f"unknown experiment {name!r}")

    dh = 10.0
    rho = np.full((ny, nx), 2000.0, dtype=np.float64)
    vp, vs = 3000.0, 1700.0
    mu = rho * vs**2
    lam = rho * (vp**2 - 2.0 * vs**2)
    lam = np.full((ny, nx), lam, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    return Experiment(name, nx, ny, dh, dt, nt, fw, cpml,
                      rho, lam, mu, sources, receivers,
                      source_t0=source_t0)


def lambda_reflector(exp: Experiment) -> np.ndarray:
    """Declared positive, thin, laterally symmetric lambda-only direction."""
    out = np.zeros_like(exp.lam)
    if exp.name == "interior":
        j = exp.ny // 2
        left, right = exp.nx // 2 - 3, exp.nx // 2 + 4
    else:
        j, left, right = 33, 15, 49  # one-based y=34, x=16..49
    x = np.arange(exp.nx, dtype=np.float64)
    center = 0.5 * (left + right - 1)
    halfwidth = 0.5 * (right - left + 1)
    profile = np.zeros(exp.nx, dtype=np.float64)
    mask = (x >= left) & (x < right)
    profile[mask] = 0.5 * (1.0 + np.cos(np.pi * (x[mask] - center) / halfwidth))
    out[j, :] = 0.02 * exp.lam[j, :] * profile
    return out


def mu_compact_direction(exp: Experiment) -> np.ndarray:
    """A separate compact mu-only perturbation, with an asymmetric pattern."""
    out = np.zeros_like(exp.mu)
    if exp.name == "interior":
        j0, i0 = exp.ny // 2 - 2, exp.nx // 2 - 3
    else:
        j0, i0 = 28, 26  # one-based y=29..32, x=27..35
    for dj in range(4):
        for di in range(9):
            weight = 0.55 + 0.11 * di + 0.037 * dj + 0.013 * di * dj
            out[j0 + dj, i0 + di] = 0.02 * exp.mu[j0 + dj, i0 + di] * weight
    return out


def deterministic_model_direction(exp: Experiment, seed: int = 47) -> tuple[np.ndarray, np.ndarray]:
    """Non-symmetric direction for dot-product and parameter-channel gates."""
    y, x = np.mgrid[1:exp.ny + 1, 1:exp.nx + 1]
    lpat = (0.31 * np.sin(0.173 * x + 0.119 * y + 0.07 * seed)
            + 0.23 * np.cos(0.071 * x - 0.211 * y + 0.13 * seed)
            + 0.17 * np.sin(0.037 * x * y + 0.11 * seed))
    mpat = (0.29 * np.cos(0.137 * x + 0.191 * y + 0.17 * seed)
            - 0.21 * np.sin(0.223 * x - 0.053 * y + 0.09 * seed)
            + 0.16 * np.cos(0.031 * x * y + 0.19 * seed))
    return (exp.lam * 0.01 * lpat, exp.mu * 0.01 * mpat)


def deterministic_data(exp: Experiment, seed: int = 83,
                       component: int | None = None) -> np.ndarray:
    """Non-symmetric prepared migration data, not an FWI residual."""
    t = np.arange(1, exp.nt + 1, dtype=np.float64)[:, None]
    r = np.arange(1, len(exp.receivers) + 1, dtype=np.float64)[None, :]
    vx = np.sin(0.193 * t + 0.317 * r + 0.11 * seed) + 0.23 * np.cos(0.071 * t * r)
    vy = np.cos(0.157 * t - 0.271 * r + 0.07 * seed) - 0.19 * np.sin(0.043 * t * r)
    data = np.stack((vx, vy), axis=-1).astype(np.float64)
    if component is not None:
        if component not in (0, 1):
            raise ValueError("component must be 0 (vx) or 1 (vy)")
        data[..., 1 - component] = 0.0
    return data


def _shift_periodic(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """DENISE's single-rank halo exchange wraps edge values to the other side."""
    return np.roll(a, shift=(-dy, -dx), axis=(-2, -1))


def _stencil(a: np.ndarray, terms: tuple[tuple[int, int, float], ...]) -> np.ndarray:
    out = np.zeros_like(a)
    for dy, dx, weight in terms:
        out += weight * _shift_periodic(a, dy, dx)
    return out


def _stencil_transpose(a_bar: np.ndarray,
                       terms: tuple[tuple[int, int, float], ...]) -> np.ndarray:
    out = np.zeros_like(a_bar)
    for dy, dx, weight in terms:
        out += weight * _shift_periodic(a_bar, -dy, -dx)
    return out


_DX_BACK = ((0, 0, FD4[0]), (0, -1, -FD4[0]),
            (0, 1, FD4[1]), (0, -2, -FD4[1]))
_DX_FWD = ((0, 1, FD4[0]), (0, 0, -FD4[0]),
           (0, 2, FD4[1]), (0, -1, -FD4[1]))
_DY_BACK = ((0, 0, FD4[0]), (-1, 0, -FD4[0]),
            (1, 0, FD4[1]), (-2, 0, -FD4[1]))
_DY_FWD = ((1, 0, FD4[0]), (0, 0, -FD4[0]),
           (2, 0, FD4[1]), (-1, 0, -FD4[1]))


def _cpml_1d(n: int, dh: float, dt: float, fw: int, *, half: bool,
             damping_speed: float, reflection: float, power: float,
             kmax: float, fpml: float) -> PMLProfile:
    if fw == 0:
        one = np.ones(n, dtype=np.float64)
        return PMLProfile(one, np.zeros(n), one)
    thick = fw * dh
    x = np.arange(n, dtype=np.float64) * dh + (0.5 * dh if half else 0.0)
    left_distance = thick - x
    right_origin = (n - 1) * dh - thick
    right_distance = x - right_origin
    distance = np.maximum(left_distance, right_distance)
    active = distance >= 0.0
    normalized = np.zeros(n, dtype=np.float64)
    normalized[active] = distance[active] / thick
    d0 = -(power + 1.0) * damping_speed * math.log(reflection) / (2.0 * thick)
    damping = d0 * normalized**power
    kappa = 1.0 + (kmax - 1.0) * normalized**power
    alpha = np.maximum(0.0, math.pi * fpml * (1.0 - normalized))
    b = np.exp(-(damping / kappa + alpha) * dt)
    a = np.zeros(n, dtype=np.float64)
    nz = np.abs(damping) > 1.0e-6
    a[nz] = (damping[nz] * (b[nz] - 1.0)
             / (kappa[nz] * (damping[nz] + kappa[nz] * alpha[nz])))
    return PMLProfile(kappa, a, b)


def cpml_profiles(exp: Experiment) -> dict[str, PMLProfile]:
    """DENISE PML_pro profile equations on center and half-grid locations."""
    if not exp.cpml:
        x0 = _cpml_1d(exp.nx, exp.dh, exp.dt, 0, half=False,
                      damping_speed=3000.0, reflection=exp.pml_reflection,
                      power=exp.pml_power, kmax=exp.pml_kmax, fpml=exp.pml_fpml)
        xh = x0
        y0 = _cpml_1d(exp.ny, exp.dh, exp.dt, 0, half=False,
                      damping_speed=3000.0, reflection=exp.pml_reflection,
                      power=exp.pml_power, kmax=exp.pml_kmax, fpml=exp.pml_fpml)
        yh = y0
    else:
        vmax = float(np.sqrt(np.max((exp.lam + 2.0 * exp.mu) / exp.rho)))
        args = dict(damping_speed=vmax, reflection=exp.pml_reflection,
                    power=exp.pml_power, kmax=exp.pml_kmax, fpml=exp.pml_fpml)
        x0 = _cpml_1d(exp.nx, exp.dh, exp.dt, exp.fw, half=False, **args)
        xh = _cpml_1d(exp.nx, exp.dh, exp.dt, exp.fw, half=True, **args)
        y0 = _cpml_1d(exp.ny, exp.dh, exp.dt, exp.fw, half=False, **args)
        yh = _cpml_1d(exp.ny, exp.dh, exp.dt, exp.fw, half=True, **args)
    return {
        "x": PMLProfile(*(v[None, :] for v in x0)),
        "xh": PMLProfile(*(v[None, :] for v in xh)),
        "y": PMLProfile(*(v[:, None] for v in y0)),
        "yh": PMLProfile(*(v[:, None] for v in yh)),
    }


def _pml_forward(q: np.ndarray, psi: np.ndarray,
                 profile: PMLProfile) -> tuple[np.ndarray, np.ndarray]:
    psi_new = profile.b * psi + profile.a * q
    return q / profile.kappa + psi_new, psi_new


def _pml_transpose(qcorr_bar: np.ndarray, psi_new_bar: np.ndarray,
                   profile: PMLProfile) -> tuple[np.ndarray, np.ndarray]:
    total_new_bar = qcorr_bar + psi_new_bar
    q_bar = qcorr_bar / profile.kappa + profile.a * total_new_bar
    psi_old_bar = profile.b * total_new_bar
    return q_bar, psi_old_bar


def shear_corner_mu(mu: np.ndarray) -> np.ndarray:
    """Four-cell harmonic mean at DENISE's (i+1/2,j+1/2) shear point."""
    m10 = _shift_periodic(mu, 0, 1)
    m01 = _shift_periodic(mu, 1, 0)
    m11 = _shift_periodic(mu, 1, 1)
    return 4.0 / (1.0 / mu + 1.0 / m10 + 1.0 / m01 + 1.0 / m11)


def shear_corner_tangent(mu: np.ndarray, dmu: np.ndarray) -> np.ndarray:
    m10, m01, m11 = (_shift_periodic(mu, 0, 1), _shift_periodic(mu, 1, 0),
                     _shift_periodic(mu, 1, 1))
    dm10, dm01, dm11 = (_shift_periodic(dmu, 0, 1), _shift_periodic(dmu, 1, 0),
                        _shift_periodic(dmu, 1, 1))
    mcorner = 4.0 / (1.0 / mu + 1.0 / m10 + 1.0 / m01 + 1.0 / m11)
    return (mcorner**2 / 4.0) * (dmu / mu**2 + dm10 / m10**2
                                  + dm01 / m01**2 + dm11 / m11**2)


def shear_corner_transpose(mu: np.ndarray, corner_bar: np.ndarray) -> np.ndarray:
    """Exact transpose of shear_corner_tangent, including periodic halos."""
    m10, m01, m11 = (_shift_periodic(mu, 0, 1), _shift_periodic(mu, 1, 0),
                     _shift_periodic(mu, 1, 1))
    mcorner = 4.0 / (1.0 / mu + 1.0 / m10 + 1.0 / m01 + 1.0 / m11)
    common = corner_bar * mcorner**2 / 4.0
    out = np.zeros_like(mu)
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        neighbor_mu = _shift_periodic(mu, dy, dx)
        corner_contribution = common / neighbor_mu**2
        out += np.roll(corner_contribution, shift=(dy, dx), axis=(0, 1))
    return out


def _source_samples(exp: Experiment) -> np.ndarray:
    """Discrete production-style Ricker, one integrated signal, then psource.

    DENISE QUELLART=1 evaluates the Ricker at integer stress times, optionally
    integrates N_ORDER times in wavelet(), then psource() differentiates the
    integrated signal after update_s.  We freeze N_ORDER=1.  The endpoints and
    one-based sample indexing match psource(); source amplitude is in Pa.
    """
    t = np.arange(1, exp.nt + 1, dtype=np.float64) * exp.dt
    ts = 1.0 / exp.fc
    tau = math.pi * (t - 1.5 * ts) / (1.5 * ts)
    wave = ((1.0 - 4.0 * tau * tau) * np.exp(-2.0 * tau * tau)
            * exp.source_amplitude)
    signal = np.zeros(exp.nt + 2, dtype=np.float64)
    for n in range(1, exp.nt + 1):
        signal[n] = signal[n - 1] + wave[n - 1] * exp.dt
    out = np.zeros(exp.nt, dtype=np.float64)
    out[0] = signal[2] / exp.dt
    for n in range(2, exp.nt):
        out[n - 1] = (signal[n + 1] - signal[n - 1]) / exp.dt
    out[-1] = -signal[exp.nt - 1] / exp.dt
    # Make the fixture's declared shift explicit (production uses srcpos[4]).
    if exp.source_t0 != 1.5 * ts:
        tau = math.pi * (t - exp.source_t0) / (1.5 * ts)
        wave = ((1.0 - 4.0 * tau * tau) * np.exp(-2.0 * tau * tau)
                * exp.source_amplitude)
        signal.fill(0.0)
        for n in range(1, exp.nt + 1):
            signal[n] = signal[n - 1] + wave[n - 1] * exp.dt
        out[0] = signal[2] / exp.dt
        for n in range(2, exp.nt):
            out[n - 1] = (signal[n + 1] - signal[n - 1]) / exp.dt
        out[-1] = -signal[exp.nt - 1] / exp.dt
    return out


def _layer_mask(exp: Experiment) -> np.ndarray:
    if not exp.cpml or exp.fw == 0:
        return np.zeros((exp.ny, exp.nx), dtype=bool)
    mask = np.zeros((exp.ny, exp.nx), dtype=bool)
    mask[:, :exp.fw] = True
    mask[:, -exp.fw:] = True
    mask[:exp.fw, :] = True
    mask[-exp.fw:, :] = True
    return mask


def _zero_state(exp: Experiment) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    fields = {name: np.zeros((exp.ny, exp.nx), dtype=np.float64)
              for name in ("vx", "vy", "sxx", "syy", "sxy")}
    memories = {name: np.zeros((exp.ny, exp.nx), dtype=np.float64)
                for name in ("sxx_x", "sxy_y", "sxy_x", "syy_y",
                             "vxx", "vyy", "vxy", "vyx")}
    return fields, memories


def _density_faces(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (2.0 / (rho + _shift_periodic(rho, 0, 1)),
            2.0 / (rho + _shift_periodic(rho, 1, 0)))


def nonlinear_forward(exp: Experiment, shot: int = 0, *,
                      lam: np.ndarray | None = None,
                      mu: np.ndarray | None = None,
                      save_strain: bool = False) -> BackgroundRun:
    """One shot of the nonlinear velocity-stress reference forward map."""
    lam = exp.lam if lam is None else np.asarray(lam, dtype=np.float64)
    mu = exp.mu if mu is None else np.asarray(mu, dtype=np.float64)
    if lam.shape != exp.lam.shape or mu.shape != exp.mu.shape or np.any(mu <= 0.0):
        raise ValueError("invalid elastic material arrays")
    profiles = cpml_profiles(exp)
    fields, memories = _zero_state(exp)
    invrho_x, invrho_y = _density_faces(exp.rho)
    mu_corner = shear_corner_mu(mu)
    data = np.zeros((exp.nt, len(exp.receivers), 2), dtype=np.float64)
    strains = (np.zeros((exp.nt, 4, exp.ny, exp.nx), dtype=np.float64)
               if save_strain else None)
    source = _source_samples(exp)
    si, sj = exp.sources[shot]
    si, sj = si - 1, sj - 1
    layer = _layer_mask(exp)
    pml_memory_peak = wave_peak = frac_peak = 0.0
    scale = np.maximum(lam + 2.0 * mu, 1.0)

    for k in range(exp.nt):
        # Stress -> half-step velocity.  q includes DT/DH as in DENISE update_v.
        q_sxx_x = exp.dt / exp.dh * _stencil(fields["sxx"], _DX_FWD)
        q_sxy_y = exp.dt / exp.dh * _stencil(fields["sxy"], _DY_BACK)
        q_sxy_x = exp.dt / exp.dh * _stencil(fields["sxy"], _DX_BACK)
        q_syy_y = exp.dt / exp.dh * _stencil(fields["syy"], _DY_FWD)
        c_sxx_x, memories["sxx_x"] = _pml_forward(
            q_sxx_x, memories["sxx_x"], profiles["xh"])
        c_sxy_y, memories["sxy_y"] = _pml_forward(
            q_sxy_y, memories["sxy_y"], profiles["y"])
        c_sxy_x, memories["sxy_x"] = _pml_forward(
            q_sxy_x, memories["sxy_x"], profiles["x"])
        c_syy_y, memories["syy_y"] = _pml_forward(
            q_syy_y, memories["syy_y"], profiles["yh"])
        fields["vx"] += invrho_x * (c_sxx_x + c_sxy_y)
        fields["vy"] += invrho_y * (c_sxy_x + c_syy_y)

        # DENISE seismo_ssg(SEISMO=1) samples these direct velocity nodes.
        for r, (ri, rj) in enumerate(exp.receivers):
            data[k, r, 0] = fields["vx"][rj - 1, ri - 1]
            data[k, r, 1] = fields["vy"][rj - 1, ri - 1]

        # Half-step velocity -> integer stress, with the exact four FD4 strains.
        raw = (
            exp.dt / exp.dh * _stencil(fields["vx"], _DX_BACK),
            exp.dt / exp.dh * _stencil(fields["vy"], _DX_FWD),
            exp.dt / exp.dh * _stencil(fields["vx"], _DY_FWD),
            exp.dt / exp.dh * _stencil(fields["vy"], _DY_BACK),
        )
        corrected = (
            _pml_forward(raw[VXX], memories["vxx"], profiles["x"]),
            _pml_forward(raw[VYX], memories["vyx"], profiles["xh"]),
            _pml_forward(raw[VXY], memories["vxy"], profiles["yh"]),
            _pml_forward(raw[VYY], memories["vyy"], profiles["y"]),
        )
        qxx, memories["vxx"] = corrected[VXX]
        qyx, memories["vyx"] = corrected[VYX]
        qxy, memories["vxy"] = corrected[VXY]
        qyy, memories["vyy"] = corrected[VYY]
        if strains is not None:
            strains[k, VXX] = qxx
            strains[k, VYX] = qyx
            strains[k, VXY] = qxy
            strains[k, VYY] = qyy
        div = qxx + qyy
        fields["sxx"] += lam * div + 2.0 * mu * qxx
        fields["syy"] += lam * div + 2.0 * mu * qyy
        fields["sxy"] += mu_corner * (qyx + qxy)

        # Explosive psource insertion is after update_s and adds to both normals.
        fields["sxx"][sj, si] += source[k]
        fields["syy"][sj, si] += source[k]

        if exp.cpml:
            memory_sum = sum(float(np.dot(p[layer], p[layer]))
                             for p in memories.values())
            pml_memory_peak = max(pml_memory_peak, memory_sum)
            e = (0.5 * exp.rho * (fields["vx"]**2 + fields["vy"]**2)
                 + 0.5 * (fields["sxx"]**2 + fields["syy"]**2
                          + 2.0 * fields["sxy"]**2) / scale)
            in_energy = float(np.sum(e[layer]))
            all_energy = float(np.sum(e))
            wave_peak = max(wave_peak, in_energy)
            frac_peak = max(frac_peak, in_energy / max(all_energy, np.finfo(float).tiny))

    return BackgroundRun(data, strains, pml_memory_peak, wave_peak, frac_peak)


def born_forward(exp: Experiment, trajectory: BackgroundRun,
                 dlam: np.ndarray, dmu: np.ndarray) -> np.ndarray:
    """Analytic tangent-linear propagation; not a finite-difference quotient."""
    if trajectory.strain is None:
        raise ValueError("Born propagation needs full-storage strain operands")
    dlam = np.asarray(dlam, dtype=np.float64)
    dmu = np.asarray(dmu, dtype=np.float64)
    if dlam.shape != exp.lam.shape or dmu.shape != exp.mu.shape:
        raise ValueError("model directions have incorrect shape")
    profiles = cpml_profiles(exp)
    fields, memories = _zero_state(exp)
    invrho_x, invrho_y = _density_faces(exp.rho)
    mu_corner = shear_corner_mu(exp.mu)
    dmu_corner = shear_corner_tangent(exp.mu, dmu)
    data = np.zeros_like(trajectory.data)

    for k in range(exp.nt):
        q_sxx_x = exp.dt / exp.dh * _stencil(fields["sxx"], _DX_FWD)
        q_sxy_y = exp.dt / exp.dh * _stencil(fields["sxy"], _DY_BACK)
        q_sxy_x = exp.dt / exp.dh * _stencil(fields["sxy"], _DX_BACK)
        q_syy_y = exp.dt / exp.dh * _stencil(fields["syy"], _DY_FWD)
        c_sxx_x, memories["sxx_x"] = _pml_forward(q_sxx_x, memories["sxx_x"], profiles["xh"])
        c_sxy_y, memories["sxy_y"] = _pml_forward(q_sxy_y, memories["sxy_y"], profiles["y"])
        c_sxy_x, memories["sxy_x"] = _pml_forward(q_sxy_x, memories["sxy_x"], profiles["x"])
        c_syy_y, memories["syy_y"] = _pml_forward(q_syy_y, memories["syy_y"], profiles["yh"])
        fields["vx"] += invrho_x * (c_sxx_x + c_sxy_y)
        fields["vy"] += invrho_y * (c_sxy_x + c_syy_y)
        for r, (ri, rj) in enumerate(exp.receivers):
            data[k, r, 0] = fields["vx"][rj - 1, ri - 1]
            data[k, r, 1] = fields["vy"][rj - 1, ri - 1]

        raw = (
            exp.dt / exp.dh * _stencil(fields["vx"], _DX_BACK),
            exp.dt / exp.dh * _stencil(fields["vy"], _DX_FWD),
            exp.dt / exp.dh * _stencil(fields["vx"], _DY_FWD),
            exp.dt / exp.dh * _stencil(fields["vy"], _DY_BACK),
        )
        corrected = (
            _pml_forward(raw[VXX], memories["vxx"], profiles["x"]),
            _pml_forward(raw[VYX], memories["vyx"], profiles["xh"]),
            _pml_forward(raw[VXY], memories["vxy"], profiles["yh"]),
            _pml_forward(raw[VYY], memories["vyy"], profiles["y"]),
        )
        qxx, memories["vxx"] = corrected[VXX]
        qyx, memories["vyx"] = corrected[VYX]
        qxy, memories["vxy"] = corrected[VXY]
        qyy, memories["vyy"] = corrected[VYY]
        bg = trajectory.strain[k]
        div = qxx + qyy
        fields["sxx"] += exp.lam * div + 2.0 * exp.mu * qxx
        fields["syy"] += exp.lam * div + 2.0 * exp.mu * qyy
        fields["sxy"] += mu_corner * (qyx + qxy)
        fields["sxx"] += dlam * (bg[VXX] + bg[VYY]) + 2.0 * dmu * bg[VXX]
        fields["syy"] += dlam * (bg[VXX] + bg[VYY]) + 2.0 * dmu * bg[VYY]
        fields["sxy"] += dmu_corner * (bg[VYX] + bg[VXY])
    return data


def _inject_receiver_transpose(exp: Experiment, data_sample: np.ndarray,
                               vx_bar: np.ndarray, vy_bar: np.ndarray) -> None:
    """P^T for exact-node vx/vy samples; repeated receivers sum, as in P^T."""
    for r, (ri, rj) in enumerate(exp.receivers):
        vx_bar[rj - 1, ri - 1] += data_sample[r, 0]
        vy_bar[rj - 1, ri - 1] += data_sample[r, 1]


def born_adjoint(exp: Experiment, trajectory: BackgroundRun,
                 data: np.ndarray, *, strain_shift: int = 0) -> BornImage:
    """Analytic reverse of the tangent recurrence and receiver sampling P."""
    if trajectory.strain is None:
        raise ValueError("Born adjoint needs full-storage strain operands")
    data = np.asarray(data, dtype=np.float64)
    if data.shape != trajectory.data.shape:
        raise ValueError("migration data must have shape (time, receiver, 2)")
    profiles = cpml_profiles(exp)
    invrho_x, invrho_y = _density_faces(exp.rho)
    mu_corner = shear_corner_mu(exp.mu)
    bars = {name: np.zeros((exp.ny, exp.nx), dtype=np.float64)
            for name in ("vx", "vy", "sxx", "syy", "sxy")}
    psi_bar = {name: np.zeros((exp.ny, exp.nx), dtype=np.float64)
               for name in ("sxx_x", "sxy_y", "sxy_x", "syy_y",
                            "vxx", "vyy", "vxy", "vyx")}
    glambda = np.zeros_like(exp.lam)
    gmu = np.zeros_like(exp.mu)
    coeff = exp.dt / exp.dh

    for k in range(exp.nt - 1, -1, -1):
        # VJP of the stress update: these are the post-stress cotangents.
        source_k = k + strain_shift
        if 0 <= source_k < exp.nt:
            bg = trajectory.strain[source_k]
            glambda += (bars["sxx"] + bars["syy"]) * (bg[VXX] + bg[VYY])
            gmu += 2.0 * (bars["sxx"] * bg[VXX] + bars["syy"] * bg[VYY])
            gcorner = bars["sxy"] * (bg[VYX] + bg[VXY])
            gmu += shear_corner_transpose(exp.mu, gcorner)

        # Reverse update_s, including the stored recursive CPML states.
        qxx_bar = (exp.lam + 2.0 * exp.mu) * bars["sxx"] + exp.lam * bars["syy"]
        qyy_bar = exp.lam * bars["sxx"] + (exp.lam + 2.0 * exp.mu) * bars["syy"]
        qyx_bar = mu_corner * bars["sxy"]
        qxy_bar = mu_corner * bars["sxy"]
        qxx_bar, psi_bar["vxx"] = _pml_transpose(qxx_bar, psi_bar["vxx"], profiles["x"])
        qyx_bar, psi_bar["vyx"] = _pml_transpose(qyx_bar, psi_bar["vyx"], profiles["xh"])
        qxy_bar, psi_bar["vxy"] = _pml_transpose(qxy_bar, psi_bar["vxy"], profiles["yh"])
        qyy_bar, psi_bar["vyy"] = _pml_transpose(qyy_bar, psi_bar["vyy"], profiles["y"])
        bars["vx"] += coeff * _stencil_transpose(qxx_bar, _DX_BACK)
        bars["vy"] += coeff * _stencil_transpose(qyx_bar, _DX_FWD)
        bars["vx"] += coeff * _stencil_transpose(qxy_bar, _DY_FWD)
        bars["vy"] += coeff * _stencil_transpose(qyy_bar, _DY_BACK)

        # P^T is inserted at the post-update_v sample before reversing update_v.
        _inject_receiver_transpose(exp, data[k], bars["vx"], bars["vy"])

        # Reverse update_v.  Stress persistence gives identity paths; derivative
        # paths are scattered with exact FD4 transposes and the CPML VJP.
        q_sxx_x_bar = invrho_x * bars["vx"]
        q_sxy_y_bar = invrho_x * bars["vx"]
        q_sxy_x_bar = invrho_y * bars["vy"]
        q_syy_y_bar = invrho_y * bars["vy"]
        q_sxx_x_bar, psi_bar["sxx_x"] = _pml_transpose(
            q_sxx_x_bar, psi_bar["sxx_x"], profiles["xh"])
        q_sxy_y_bar, psi_bar["sxy_y"] = _pml_transpose(
            q_sxy_y_bar, psi_bar["sxy_y"], profiles["y"])
        q_sxy_x_bar, psi_bar["sxy_x"] = _pml_transpose(
            q_sxy_x_bar, psi_bar["sxy_x"], profiles["x"])
        q_syy_y_bar, psi_bar["syy_y"] = _pml_transpose(
            q_syy_y_bar, psi_bar["syy_y"], profiles["yh"])
        bars["sxx"] += coeff * _stencil_transpose(q_sxx_x_bar, _DX_FWD)
        bars["sxy"] += coeff * _stencil_transpose(q_sxy_y_bar, _DY_BACK)
        bars["sxy"] += coeff * _stencil_transpose(q_sxy_x_bar, _DX_BACK)
        bars["syy"] += coeff * _stencil_transpose(q_syy_y_bar, _DY_FWD)

    return BornImage(glambda, gmu)


def receiver_transpose_basis(exp: Experiment, receiver: int, timestep: int,
                             component: int, sign: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Single (receiver,time,component) basis application of P^T."""
    if not (0 <= receiver < len(exp.receivers) and 0 <= timestep < exp.nt):
        raise IndexError("receiver/time basis outside the experiment")
    if component not in (0, 1):
        raise ValueError("component must be 0 (vx) or 1 (vy)")
    sample = np.zeros((len(exp.receivers), 2), dtype=np.float64)
    sample[receiver, component] = sign
    vx = np.zeros((exp.nt, exp.ny, exp.nx), dtype=np.float64)
    vy = np.zeros_like(vx)
    _inject_receiver_transpose(exp, sample, vx[timestep], vy[timestep])
    return vx, vy


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """Deterministic high-accuracy discrete Euclidean inner product."""
    return math.fsum(float(x) * float(y) for x, y in zip(a.ravel(), b.ravel()))


def roundoff_gamma(nwork: int) -> float:
    """Return gamma_n for the frozen FP64 bound, failing closed at n*u >= 1."""
    nu = nwork * U
    if not nu < 1.0:
        raise ValueError(f"roundoff bound requires n*u < 1 (got {nu!r})")
    return nu / (1.0 - nu)


def dot_metrics(exp: Experiment, dlam: np.ndarray, dmu: np.ndarray,
                data: np.ndarray, trajectory: BackgroundRun) -> dict[str, float]:
    jdm = born_forward(exp, trajectory, dlam, dmu)
    image = born_adjoint(exp, trajectory, data)
    lhs = dot(jdm, data)
    rhs = dot(dlam, image.image_lambda_raw) + dot(dmu, image.image_mu_raw)
    lhs_scale = float(np.linalg.norm(jdm) * np.linalg.norm(data))
    rhs_scale = math.sqrt(float(np.sum(dlam * dlam) + np.sum(dmu * dmu))) * math.sqrt(
        float(np.sum(image.image_lambda_raw**2) + np.sum(image.image_mu_raw**2)))
    scale = max(lhs_scale, rhs_scale, np.finfo(float).tiny)
    nwork = 512 * exp.nt * exp.nx * exp.ny
    gamma = roundoff_gamma(nwork)
    ceiling = 8.0 * gamma * scale
    abs_residual = abs(lhs - rhs)
    pair_scale = max(abs(lhs), abs(rhs), np.finfo(float).tiny)
    pair_condition = (lhs_scale + rhs_scale) / pair_scale
    return {
        "lhs": lhs,
        "rhs": rhs,
        "absolute_residual": abs_residual,
        "relative_residual": abs_residual / scale,
        "pair_relative_residual": abs_residual / pair_scale,
        "scale": scale,
        "absolute_ceiling": ceiling,
        "relative_ceiling": ceiling / scale,
        "pair_relative_ceiling": 8.0 * gamma * pair_condition,
        "lhs_operand_norm_product": lhs_scale,
        "rhs_operand_norm_product": rhs_scale,
        "forward_data_norm": float(np.linalg.norm(jdm)),
        "migration_data_norm": float(np.linalg.norm(data)),
        "model_direction_norm": math.sqrt(float(np.sum(dlam * dlam) + np.sum(dmu * dmu))),
        "adjoint_image_norm": math.sqrt(float(np.sum(image.image_lambda_raw**2)
                                               + np.sum(image.image_mu_raw**2))),
        "pair_condition": pair_condition,
        "forward_amplification": float(np.linalg.norm(jdm) / max(np.linalg.norm(dlam) + np.linalg.norm(dmu), 1e-300)),
        "adjoint_amplification": float(math.sqrt(np.sum(image.image_lambda_raw**2) + np.sum(image.image_mu_raw**2))
                                       / max(np.linalg.norm(data), 1e-300)),
    }


def hash_f64(a: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(a, dtype="<f8").tobytes(order="C")).hexdigest()


def fd4_spectral_bound() -> float:
    """Supremum of the dimensionless staggered FD4 derivative symbol."""
    theta = np.linspace(0.0, math.pi, 200_001, dtype=np.float64)
    symbol = 2.0 * np.sin(theta / 2.0) * (FD4[0] + 2.0 * FD4[1] * np.cos(theta))
    return float(np.max(np.abs(symbol)))


def cfl_diagnostic(exp: Experiment) -> dict[str, float]:
    vmax = float(np.sqrt(np.max((exp.lam + 2.0 * exp.mu) / exp.rho)))
    omega_bound = vmax * math.sqrt(2.0) * fd4_spectral_bound() / exp.dh
    dt_limit = 2.0 / omega_bound
    return {"vmax": vmax, "omega_bound": omega_bound, "dt_limit": dt_limit,
            "leapfrog_ratio": exp.dt / dt_limit}


def born_directional_fd_errors(exp: Experiment, dlam: np.ndarray,
                               dmu: np.ndarray,
                               epsilons: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
                               trajectory: BackgroundRun | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Central nonlinear checks of J; the quotient does not construct J or J^T."""
    base = (nonlinear_forward(exp, 0, save_strain=True)
            if trajectory is None else trajectory)
    born = born_forward(exp, base, dlam, dmu)
    errors = []
    for eps in epsilons:
        plus = nonlinear_forward(exp, 0, lam=exp.lam + eps * dlam,
                                 mu=exp.mu + eps * dmu).data
        minus = nonlinear_forward(exp, 0, lam=exp.lam - eps * dlam,
                                  mu=exp.mu - eps * dmu).data
        fd = (plus - minus) / (2.0 * eps)
        errors.append(float(np.linalg.norm(fd - born) / max(np.linalg.norm(born), 1e-300)))
    return np.asarray(epsilons, dtype=np.float64), np.asarray(errors, dtype=np.float64)


def migrate_shot_stack(exp: Experiment,
                       trajectories: tuple[BackgroundRun, ...] | list[BackgroundRun],
                       data: tuple[np.ndarray, ...] | list[np.ndarray],
                       *, strain_shift: int = 0) -> BornImage:
    """Raw J^T over ordered independent shots; no shot normalization."""
    if len(trajectories) != len(data):
        raise ValueError("each shot trajectory needs one prepared data gather")
    glambda = np.zeros_like(exp.lam)
    gmu = np.zeros_like(exp.mu)
    for trajectory, shot_data in zip(trajectories, data):
        image = born_adjoint(exp, trajectory, shot_data, strain_shift=strain_shift)
        glambda += image.image_lambda_raw
        gmu += image.image_mu_raw
    return BornImage(glambda, gmu)


def reflector_mask(exp: Experiment, *, y_radius: int = 0) -> np.ndarray:
    """Boolean support of the declared primary lambda reflector, optionally dilated in y."""
    mask = lambda_reflector(exp) != 0.0
    if y_radius <= 0:
        return mask
    out = mask.copy()
    for dy in range(1, y_radius + 1):
        out |= np.roll(mask, dy, axis=0) | np.roll(mask, -dy, axis=0)
    return out


def image_diagnostics(exp: Experiment, image: BornImage) -> dict[str, float]:
    """Numerical reflector diagnostics; no visualization-based acceptance."""
    absolute = np.abs(image.image_lambda_raw)
    peak_j, peak_i = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    mask = reflector_mask(exp, y_radius=2)
    energy = image.image_lambda_raw**2
    fraction = float(np.sum(energy[mask]) / max(np.sum(energy), np.finfo(float).tiny))
    # The symmetric shot/reflector fixture is centered between cells 32 and 33;
    # exclude the outer eight-cell PML and compare the entire symmetric bulk.
    lo, hi = exp.fw, exp.nx - exp.fw
    crop = image.image_lambda_raw[:, lo:hi]
    symmetry = float(np.linalg.norm(crop - crop[:, ::-1])
                     / max(np.linalg.norm(crop), np.finfo(float).tiny))
    return {
        "peak_i": float(peak_i + 1),
        "peak_j": float(peak_j + 1),
        "lambda_energy_fraction_near_reflector": fraction,
        "lateral_symmetry_relative_residual": symmetry,
        "lambda_image_norm": float(np.linalg.norm(image.image_lambda_raw)),
        "mu_image_norm": float(np.linalg.norm(image.image_mu_raw)),
    }


def alignment_separation(exp: Experiment, trajectory: BackgroundRun,
                         data: np.ndarray) -> dict[str, float]:
    """Contrast the correct source-strain alignment with ±1-step alternatives."""
    good = born_adjoint(exp, trajectory, data, strain_shift=0)
    minus = born_adjoint(exp, trajectory, data, strain_shift=-1)
    plus = born_adjoint(exp, trajectory, data, strain_shift=1)
    ngood = math.sqrt(dot(good.image_lambda_raw, good.image_lambda_raw)
                      + dot(good.image_mu_raw, good.image_mu_raw))
    nminus = math.sqrt(dot(minus.image_lambda_raw, minus.image_lambda_raw)
                       + dot(minus.image_mu_raw, minus.image_mu_raw))
    nplus = math.sqrt(dot(plus.image_lambda_raw, plus.image_lambda_raw)
                      + dot(plus.image_mu_raw, plus.image_mu_raw))
    dm = math.sqrt(dot(good.image_lambda_raw - minus.image_lambda_raw,
                       good.image_lambda_raw - minus.image_lambda_raw)
                   + dot(good.image_mu_raw - minus.image_mu_raw,
                         good.image_mu_raw - minus.image_mu_raw))
    dp = math.sqrt(dot(good.image_lambda_raw - plus.image_lambda_raw,
                       good.image_lambda_raw - plus.image_lambda_raw)
                   + dot(good.image_mu_raw - plus.image_mu_raw,
                         good.image_mu_raw - plus.image_mu_raw))
    return {"minus_relative_separation": dm / max(ngood, 1e-300),
            "plus_relative_separation": dp / max(ngood, 1e-300),
            "minimum_separation": min(dm, dp) / max(ngood, 1e-300),
            "minus_norm_ratio": nminus / max(ngood, 1e-300),
            "plus_norm_ratio": nplus / max(ngood, 1e-300)}
