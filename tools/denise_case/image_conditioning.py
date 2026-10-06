"""Pure, solver-free A2.5A operators on immutable sensitivity images.

Every operator returns fresh FP64 arrays. Parameter transforms, derivatives,
band-pass and gain are DERIVED_NUMERICAL, not reflectivity or velocity updates.
Percentile scaling and signed compression are DISPLAY_ONLY and must never feed
amplitude/energy diagnostics. No operator masks, floors or repairs physical water.
"""
from __future__ import annotations

import math
import numpy as np

PRIMARY_CORNERS = (.0005, .001, .005, .00625)
SECONDARY_CORNERS = (.001, .002, .00625, .008333333333333333)
FFT_CONVENTION = 'numpy.fft.fft2/ifft2, norm=backward; cycles/metre; symmetric zero padding; original-grid crop'
TAPER_CONVENTION = 'separable symmetric sin^2(pi*i/(2*N)), i=0..N-1 from edge; interior=1; N=0 means no taper'


def _image(a, name='image'):
    original = np.asarray(a)
    if original.dtype.kind not in 'biuf':
        raise ValueError(f'{name} must be a real numeric array')
    value = np.asarray(original, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a nonempty finite 2-D image')
    return value


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be positive and finite')
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return result


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _finite_output(a, name):
    if not np.isfinite(a).all():
        raise ValueError(f'{name} overflowed FP64')
    return a


def parameter_transform(lam, mu, vp, vs, rho):
    """Fixed-density cotangent chain rule using the supplied background.

    lnVs is a frozen-zero extension at Vs=0, NOT physical log coordinates there.
    Positive-zero water Vs naturally gives zero; no water mask is applied.
    """
    arrays = [_image(v, n) for v, n in zip((lam, mu, vp, vs, rho), ('lambda', 'mu', 'Vp', 'Vs', 'rho'))]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('sensitivity/background shapes must match')
    lam, mu, vp, vs, rho = arrays
    if np.any(vp <= 0) or np.any(vs < 0) or np.any(rho <= 0):
        raise ValueError('Vp/rho must be positive; Vs must be nonnegative with exact physical zeros')
    with np.errstate(over='ignore', invalid='ignore'):
        gv = 2.0 * rho * vp * lam
        gs = 2.0 * rho * vs * (mu - 2.0 * lam)
        result = {'vp': gv, 'vs': gs, 'lnvp': vp * gv, 'lnvs': vs * gs}
    return {k: _finite_output(v, k) for k, v in result.items()}


def derivatives(a, dh, water=None):
    """Centered nonperiodic dx, depth-positive dz and five-point Laplacian.

    Only the required stencil boundary is invalid: dx at x edges, dz at z
    edges, Laplacian at all edges. Invalid samples are NaN plus explicit masks.
    Interface crossings are separate metadata; their computed values are retained.
    """
    a = _image(a)
    dh = _positive(dh, 'DH')
    if min(a.shape) < 3:
        raise ValueError('derivatives require at least three cells on each axis')
    valid = {k: np.zeros(a.shape, dtype=bool) for k in ('dx', 'dz', 'laplacian')}
    valid['dx'][:, 1:-1] = True
    valid['dz'][1:-1, :] = True
    valid['laplacian'][1:-1, 1:-1] = True
    result = {k: np.full(a.shape, np.nan, dtype=np.float64) for k in valid}
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        result['dx'][:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * dh)
        result['dz'][1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * dh)
        result['laplacian'][1:-1, 1:-1] = (
            (a[1:-1, 2:] - a[1:-1, 1:-1]) + (a[1:-1, :-2] - a[1:-1, 1:-1])
            + (a[2:, 1:-1] - a[1:-1, 1:-1]) + (a[:-2, 1:-1] - a[1:-1, 1:-1])) / (dh * dh)
    for key in valid:
        _finite_output(result[key][valid[key]], key)
    crosses = {k: np.zeros(a.shape, dtype=bool) for k in valid}
    if water is not None:
        water = np.asarray(water)
        if water.shape != a.shape or water.dtype.kind != 'b':
            raise ValueError('water must be a boolean image of matching shape')
        crosses['dx'][:, 1:-1] = ((water[:, 2:] != water[:, 1:-1]) | (water[:, :-2] != water[:, 1:-1]))
        crosses['dz'][1:-1, :] = ((water[2:, :] != water[1:-1, :]) | (water[:-2, :] != water[1:-1, :]))
        crosses['laplacian'] = (crosses['dx'] | crosses['dz']) & valid['laplacian']
    result['valid'] = valid
    result['crosses_interface'] = crosses
    return result


def filter_transfer(shape, dh, corners=PRIMARY_CORNERS):
    """Real-even radial Fourier multiplier; spatial cycles/metre, not Hz."""
    if len(shape) != 2 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 1 for n in shape):
        raise ValueError('shape must contain two positive integers')
    dh = _positive(dh, 'DH')
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4,) or not np.isfinite(corners).all() or corners[0] < 0 or not np.all(np.diff(corners) > 0):
        raise ValueError('four finite nonnegative strictly increasing filter corners required')
    k1, k2, k3, k4 = corners
    k = np.hypot(np.fft.fftfreq(shape[0], dh)[:, None], np.fft.fftfreq(shape[1], dh)[None, :])
    transfer = np.zeros(shape, dtype=np.float64)
    transfer[(k >= k2) & (k <= k3)] = 1.0
    lower = (k > k1) & (k < k2)
    upper = (k > k3) & (k < k4)
    transfer[lower] = np.sin(np.pi * (k[lower] - k1) / (2.0 * (k2 - k1))) ** 2
    transfer[upper] = np.cos(np.pi * (k[upper] - k3) / (2.0 * (k4 - k3))) ** 2
    return transfer


def edge_taper(shape, cells=10):
    """Declared symmetric separable cosine taper; no alteration of source array."""
    # The transfer validates shape without choosing an image-specific filter.
    filter_transfer(shape, 1.0)
    cells = _integer(cells, 'taper_cells')
    if cells * 2 > min(shape):
        raise ValueError('taper regions must not overlap')
    axes = []
    for length in shape:
        axis = np.ones(length, dtype=np.float64)
        if cells:
            ramp = np.sin(np.pi * np.arange(cells, dtype=np.float64) / (2.0 * cells)) ** 2
            axis[:cells] = ramp
            axis[-cells:] = ramp[::-1]
        axes.append(axis)
    return axes[0][:, None] * axes[1][None, :]


def bandpass(a, dh, corners=PRIMARY_CORNERS, padding=128, taper_cells=10):
    """Full unmasked image -> declared taper -> zero-pad -> FFT/H/IFFT -> crop.

    Padding reduces wrap contamination but cannot remove the noncompact filter
    kernel's leakage/ringing. No claim of recovered bandwidth is made.
    """
    a = _image(a)
    dh = _positive(dh, 'DH')
    padding = _integer(padding, 'padding')
    tapered = a * edge_taper(a.shape, taper_cells)
    padded = np.pad(tapered, ((padding, padding), (padding, padding)), mode='constant')
    transfer = filter_transfer(padded.shape, dh, corners)
    with np.errstate(over='ignore', invalid='ignore'):
        output = np.fft.ifft2(np.fft.fft2(padded, norm='backward') * transfer, norm='backward').real
    # Explicit stop indices make padding=0 the same well-defined crop operation.
    result = output[padding:padding + a.shape[0], padding:padding + a.shape[1]].copy()
    return _finite_output(result, 'bandpass')


def depth_gain(a, z, z0=650, p=1, cap=3):
    """Fixed diagnostic gain, never adaptive AGC or illumination compensation."""
    a = _image(a)
    z = np.asarray(z, dtype=np.float64)
    z0, cap = _positive(z0, 'z0'), _positive(cap, 'cap')
    if z.shape != (a.shape[0],) or not np.isfinite(z).all() or np.any(z < 0):
        raise ValueError('z must contain one finite nonnegative depth per row')
    if isinstance(p, (bool, np.bool_)):
        raise ValueError('p must be finite nonnegative and cap at least one')
    p = float(p)
    if not math.isfinite(p) or p < 0 or cap < 1:
        raise ValueError('p must be finite nonnegative and cap at least one')
    if p == 0 or cap == 1:
        gain = np.ones_like(z)
    else:
        # An overflowing positive power is capped exactly; no infinity enters
        # the returned numerical image. Ordinary values follow the declared
        # literal power/minimum expression rather than a log-roundtrip.
        with np.errstate(over='ignore'):
            gain = np.minimum(cap, (np.maximum(z, z0) / z0) ** p)
    with np.errstate(over='ignore', invalid='ignore'):
        return _finite_output(a * gain[:, None], 'depth gain')


def display_scale(a, percentile=99, mask=None):
    """Symmetric percentile limits ±q; no numerical amplitude modification.

    q uses NumPy percentile with linear interpolation on abs(source[scale domain]).
    Saturation is measured over the complete source, not only the scale domain.
    """
    a = _image(a)
    percentile = float(percentile)
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError('percentile must lie in [0,100]')
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.shape != a.shape or mask.dtype.kind != 'b':
            raise ValueError('display scale mask must be boolean with matching shape')
    if not np.any(mask):
        raise ValueError('display scale domain must be nonempty')
    q = float(np.percentile(np.abs(a[mask]), percentile, interpolation='linear'))
    return {'q': q, 'limits': [-q, q], 'percentile': percentile, 'method': 'linear',
            'saturated_fraction': float(np.count_nonzero(np.abs(a) > q) / a.size),
            'saturated_count': int(np.count_nonzero(np.abs(a) > q)),
            'scale_domain_count': int(np.count_nonzero(mask)), 'total_count': int(a.size)}


def signed_compress(a, q, ratio=10):
    """DISPLAY_ONLY asinh(a/(q/ratio))/asinh(ratio); odd and zero-preserving.

    q=0 is valid only for an exactly zero image. No arbitrary epsilon or floor.
    No saturation is performed; values outside ±q remain outside ±1.
    """
    a = _image(a)
    ratio = _positive(ratio, 'ratio')
    q = float(q)
    if not math.isfinite(q) or q < 0:
        raise ValueError('q must be finite nonnegative')
    if q == 0:
        if np.any(a != 0):
            raise ValueError('zero q cannot compress nonzero data')
        return a.copy()
    # asinh(x) ~ sign(x)*(log(abs(x))+log(2)) for large x. Work in
    # logarithms if scaling would overflow, without creating infinity/epsilon.
    result = np.zeros_like(a)
    nonzero = a != 0
    log_abs = np.log(np.abs(a[nonzero])) + math.log(ratio) - math.log(q)
    compressed = np.empty_like(log_abs)
    large = log_abs > 350
    compressed[large] = log_abs[large] + math.log(2)
    compressed[~large] = np.arcsinh(np.exp(log_abs[~large]))
    result[nonzero] = np.copysign(compressed / math.asinh(ratio), a[nonzero])
    return _finite_output(result, 'signed compression')
