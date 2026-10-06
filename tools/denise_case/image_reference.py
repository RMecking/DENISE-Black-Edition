"""Reference-only geometry evaluation; never an image-conditioning input.

Windows, thresholds and displacement controls are declared before an image is
provided. Geometry correspondence is descriptive, not proof of geology, and
does not imply matching sensitivity amplitude/polarity to model contrast.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


DEPTH_WINDOWS = ((1000, 1500), (1500, 2000), (2000, 2500),
                 (2500, 3000), (3000, 3280))


def _gradient(a, dh):
    """Centered physical-grid gradients, with invalid boundary values NaN."""
    dx = np.full(a.shape, np.nan, dtype=np.float64)
    dz = dx.copy()
    dx[1:-1, 1:-1] = (a[1:-1, 2:] - a[1:-1, :-2]) / (2 * dh)
    dz[1:-1, 1:-1] = (a[2:, 1:-1] - a[:-2, 1:-1]) / (2 * dh)
    return dx, dz


def _hash(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype='<f8').tobytes()).hexdigest()


def _models(values, shape=None):
    result = {}
    for key in ('vp', 'vs', 'rho'):
        a = np.asarray(values[key], dtype=np.float64)
        if a.ndim != 2 or min(a.shape) < 3 or not np.all(np.isfinite(a)):
            raise ValueError('reference models must be finite 2-D grids of at least 3x3')
        if shape is not None and a.shape != shape:
            raise ValueError('reference/background model shape mismatch')
        shape = a.shape
        if np.any(a < 0) or (key != 'vs' and np.any(a == 0)):
            raise ValueError('nonphysical reference/background model values')
        result[key] = a.copy()
    return result


def _window_mask(window, shape, dh):
    z = dh * np.arange(1, shape[0] + 1)
    x = dh * np.arange(1, shape[1] + 1)
    high = z <= window['z_max_m'] if window['include_z_max'] else z < window['z_max_m']
    return ((z >= window['z_min_m']) & high)[:, None] & (
        (x >= window['x_min_m']) & (x <= window['x_max_m']))[None, :]


def _ridges(a, mask):
    """Strongest absolute sample in each column, first depth on exact ties."""
    values = np.where(mask & np.isfinite(a), np.abs(a), 0.)
    rows = np.argmax(values, axis=0)
    cols = np.arange(a.shape[1])
    present = values[rows, cols] > 0
    return np.column_stack((rows[present], cols[present]))


def define_reference(reference: dict, background: dict, dh=20., *, windows=None):
    """Return ``(JSON spec, read-only arrays)`` without seeing migration images.

    Default fixed domains use non-lateral-CPML and acquisition-covered windows
    in five deeper depth bins. Optional windows are for independently declared
    fixtures/cases, not image-dependent selection. The primary model edge is
    |grad(true Vp)|; Vs/rho edges and all contrasts are separate references.
    """
    if not np.isfinite(dh) or dh <= 0:
        raise ValueError('DH must be finite and positive')
    true = _models(reference)
    shape = true['vp'].shape
    smooth = _models(background, shape)
    if windows is None:
        windows = [dict(name=f'{label}_{lo}_{hi}', x_min_m=xlo, x_max_m=xhi,
                        z_min_m=lo, z_max_m=hi, include_z_max=hi == 3280)
                   for label, xlo, xhi in (('interior', 220, 9800), ('acquisition', 800, 8780))
                   for lo, hi in DEPTH_WINDOWS]
    windows = [dict(w) for w in windows]
    if not windows or len({w['name'] for w in windows}) != len(windows):
        raise ValueError('reference windows must be nonempty with unique names')
    for w in windows:
        for key in ('x_min_m', 'x_max_m', 'z_min_m', 'z_max_m'):
            if not np.isfinite(w[key]):
                raise ValueError('window coordinates must be finite')
        if w['x_min_m'] > w['x_max_m'] or w['z_min_m'] >= w['z_max_m']:
            raise ValueError('invalid reference window bounds')
    arrays = {}
    for key in true:
        arrays[f'true_{key}'] = true[key]
        arrays[f'contrast_{key}'] = true[key] - smooth[key]
        dx, dz = _gradient(true[key], dh)
        arrays[f'edge_{key}'] = np.hypot(dx, dz)
        if key == 'vp':
            arrays['edge_dx'], arrays['edge_dz'] = dx, dz
    domain = np.zeros(shape, dtype=bool)
    for w in windows:
        domain |= _window_mask(w, shape, dh)
    domain &= (smooth['vs'] > 0) & np.isfinite(arrays['edge_vp'])
    values = arrays['edge_vp'][domain]
    threshold = float(np.percentile(values, 90, interpolation='linear')) if values.size else None
    target = domain & (arrays['edge_vp'] > 0)
    if threshold is not None:
        target &= arrays['edge_vp'] >= threshold
    arrays['evaluation_domain'], arrays['target_edge'] = domain, target
    for w in windows:
        mask = _window_mask(w, shape, dh) & domain
        w['sample_count'] = int(mask.sum())
        w['target_edge_count'] = int((mask & target).sum())
        w['reference_ridge_y_x'] = _ridges(arrays['edge_vp'], mask).tolist()
    spec = dict(schema='a25a-reference-v1', role='EVALUATION_ONLY', shape_y_x=list(shape),
                dh_m=float(dh), centers='(index+1)*DH; depth positive downward',
                primary_edge='hypot(centered_Dx_true_Vp, centered_Dz_true_Vp)',
                boundary='one-cell invalid; no periodic wrap', solid='smooth2 Vs > 0',
                edge_percentile=90, percentile_method='linear', edge_threshold=threshold,
                windows=windows, ridge_definition='per-column absolute maximum; first-depth tie',
                proximity_tolerance_m=40., lateral_control_offsets_m=[-1000., 1000.],
                controls='zero fill displaced reference; common lateral support; no wrap',
                reference_model_sha256={k: _hash(v) for k, v in true.items()},
                background_model_sha256={k: _hash(v) for k, v in smooth.items()},
                array_sha256={k: _hash(v) for k, v in arrays.items()},
                caveat='Geometry correspondence does not prove geological correctness; no amplitude or polarity match required.')
    spec['definition_sha256'] = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    for a in arrays.values():
        a.flags.writeable = False
    return spec, arrays


def _nearest(points, targets, dh):
    if not len(points) or not len(targets):
        return None
    output = []
    for chunk in np.array_split(points, max(1, (len(points) + 31) // 32)):
        differences = chunk[:, None, :] - targets[None, :, :]
        output.extend(np.sqrt(np.min(np.sum(differences * differences, axis=2), axis=1)) * dh)
    return np.asarray(output)


def _measure(image, dx, dz, domain, target, edge_dx, edge_dz, edge_strength, dh, tolerance):
    valid = domain & np.isfinite(image)
    targets = np.argwhere(target & valid)
    ridges = _ridges(image, valid)
    distances = _nearest(ridges, targets, dh)
    reference_ridges = _ridges(edge_strength, valid)
    reference_by_column = {int(x): int(y) for y, x in reference_ridges}
    vertical = [abs(int(y) - reference_by_column[int(x)]) * dh for y, x in ridges if int(x) in reference_by_column]
    near = np.zeros(image.shape, dtype=bool)
    radius = int(np.floor(tolerance / dh))
    for oy in range(-radius, radius + 1):
        for ox in range(-radius, radius + 1):
            if np.hypot(oy, ox) * dh <= tolerance:
                y0, y1 = max(0, oy), min(image.shape[0], image.shape[0] + oy)
                x0, x1 = max(0, ox), min(image.shape[1], image.shape[1] + ox)
                near[y0:y1, x0:x1] |= target[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    near &= valid
    away = valid & ~near
    nmean = float(np.mean(np.abs(image[near]))) if near.any() else None
    amean = float(np.mean(np.abs(image[away]))) if away.any() else None
    norm = np.hypot(dx, dz) * np.hypot(edge_dx, edge_dz)
    oriented = target & valid & np.isfinite(norm) & (norm > 0)
    cosine = np.abs((dx[oriented] * edge_dx[oriented] + dz[oriented] * edge_dz[oriented]) / norm[oriented])
    return dict(sample_count=int(valid.sum()), nonfinite_excluded=int((domain & ~np.isfinite(image)).sum()),
                target_edge_count=len(targets), image_ridge_count=len(ridges),
                ridge_nearest_edge_mean_m=None if distances is None else float(np.mean(distances)),
                ridge_within_40m_fraction=None if distances is None else float(np.mean(distances <= tolerance)),
                ridge_same_column_depth_error_mean_m=None if not vertical else float(np.mean(vertical)),
                near_edge_count=int(near.sum()), off_edge_count=int(away.sum()),
                mean_abs_near_edge=nmean, mean_abs_off_edge=amean,
                near_off_mean_abs_ratio=None if amean is None or amean == 0 or nmean is None else nmean / amean,
                axial_gradient_cosine_mean=None if not cosine.size else float(np.mean(np.minimum(cosine, 1))),
                orientation_sample_count=int(oriented.sum()))


def _shift(a, cells):
    shifted = np.zeros_like(a)
    if abs(cells) >= a.shape[1]:
        return shifted
    if cells > 0:
        shifted[:, cells:] = a[:, :-cells]
    elif cells < 0:
        shifted[:, :cells] = a[:, -cells:]
    else:
        shifted[:] = a
    return shifted


def evaluate_reference(image, spec, arrays):
    """Compare independently processed sensitivity with frozen model geometry.

    Undefined denominators/absent structures are JSON null, never epsilon
    repaired. Invalid derived boundary cells are counted and excluded explicitly.
    """
    image = np.asarray(image, dtype=np.float64)
    if list(image.shape) != spec['shape_y_x']:
        raise ValueError('image/reference shape mismatch')
    for key, digest in spec['array_sha256'].items():
        if _hash(arrays[key]) != digest:
            raise ValueError('frozen reference array identity mismatch')
    bound = {k: v for k, v in spec.items() if k != 'definition_sha256'}
    digest = hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if digest != spec['definition_sha256']:
        raise ValueError('frozen reference definition identity mismatch')
    dh = spec['dh_m']
    dx, dz = _gradient(image, dh)
    result = dict(reference_definition_sha256=spec['definition_sha256'],
                  role='DESCRIPTIVE_STRUCTURAL_EVALUATION', image_sha256=_hash(image),
                  caveat=spec['caveat'], windows={})
    shifts = [int(round(m / dh)) for m in spec['lateral_control_offsets_m']]
    margin = max(map(abs, shifts))
    common = np.zeros(image.shape, dtype=bool)
    if image.shape[1] > 2 * margin:
        common[:, margin:-margin] = True
    for w in spec['windows']:
        domain = _window_mask(w, image.shape, dh) & arrays['evaluation_domain']
        arguments = (arrays['target_edge'], arrays['edge_dx'], arrays['edge_dz'], arrays['edge_vp'], dh, spec['proximity_tolerance_m'])
        observed = _measure(image, dx, dz, domain, *arguments)
        control_domain = domain & common
        # Same valid source/evaluation support for every displaced reference.
        # Global grid margins alone do not account for restricted target domains.
        for cells in shifts:
            control_domain &= _shift(arrays['evaluation_domain'], cells)
        controls = {'unshifted_common_support': _measure(image, dx, dz, control_domain, *arguments)}
        for metres, cells in zip(spec['lateral_control_offsets_m'], shifts):
            shifted = tuple(_shift(arrays[k], cells) for k in ('target_edge', 'edge_dx', 'edge_dz', 'edge_vp'))
            controls[f'shift_{metres:+g}m'] = _measure(image, dx, dz, control_domain, *shifted, dh, spec['proximity_tolerance_m'])
        result['windows'][w['name']] = dict(observed=observed, displaced_reference_controls=controls)
    result['control_caveats'] = ('Two displaced layered-model references are weak specificity controls, not a geology null or p-value. '
                                'Axial orientation |cos| has isotropic baseline 2/pi, not zero; lateral shifts can preserve layering.')
    return result
