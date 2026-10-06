"""Solver-free sensitivity diagnostics; squared norms are not seismic energy.

All masks use sample centers (index+1)*DH. No mute, epsilon denominator,
normalization of the input, or shot rejection is performed. JSON ``None`` means
undefined, including norms whose square cannot be represented in binary64;
the exact power-of-two scale representation remains available in that case.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def _array(a):
    return np.asarray(a, dtype=np.float64)


def _finite_image(a):
    a = _array(a)
    if a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all():
        raise ValueError('expected nonempty finite two-dimensional image')
    return a


def _square(a):
    """Represent sum(a*a) without squaring tiny unscaled input values."""
    a = _array(a).ravel()
    scale = float(np.max(np.abs(a))) if a.size else 0.
    if scale == 0:
        return {'mantissa': 0., 'exponent2': 0, 'value': 0., 'log10': None,
                'scale': 0., 'scaled_sum_squares': 0.}
    v = a / scale
    ss = float(np.sum(v*v, dtype=np.float64))
    sm, se = math.frexp(scale)
    em, ee = math.frexp(sm*sm*ss)
    exponent = 2*se+ee
    try:
        value = math.ldexp(em, exponent)
    except OverflowError:
        value = None
    if value == 0 or (value is not None and not math.isfinite(value)):
        value = None
    return {'mantissa': em, 'exponent2': exponent, 'value': value,
            'log10': math.log10(em)+exponent*math.log10(2.),
            'scale': scale, 'scaled_sum_squares': ss}


def _ratio_square(numerator, denominator):
    if denominator['mantissa'] == 0:
        return None
    if numerator['mantissa'] == 0:
        return 0.
    try:
        v = math.ldexp(numerator['mantissa']/denominator['mantissa'],
                       numerator['exponent2']-denominator['exponent2'])
    except OverflowError:
        return None
    return v if math.isfinite(v) else None


def _safe_product(a, b):
    v = a*b
    return v if math.isfinite(v) else None


def _safe_ratio(a, b):
    if a is None or not b:
        return None
    v = a/b
    return v if math.isfinite(v) else None


def stats(a):
    """Unconditioned finite-sample statistics, with nonfinite counts explicit.

    Quantiles use NumPy's linear interpolation. Empty finite sets have undefined
    magnitude statistics. Nonfinite samples are counted and excluded, never
    filled; authenticated raw diagnostic entry points reject nonfinite images.
    """
    a = _array(a).ravel()
    finite = np.isfinite(a)
    v = a[finite]
    result = {'count': int(a.size), 'finite_count': int(v.size),
              'nonfinite_count': int(a.size-v.size), 'positive': int(np.sum(v > 0)),
              'negative': int(np.sum(v < 0)), 'zero': int(np.sum(v == 0)),
              'negative_zero': int(np.sum((v == 0) & np.signbit(v))),
              'zero_fraction': float(np.mean(v == 0)) if v.size else None,
              'statistics_domain': 'finite samples only', 'quantile_method': 'linear'}
    e = _square(v)
    result['squared_norm_representation'] = e
    result['squared_norm'] = e['value'] if v.size else None
    if not v.size:
        result.update({k: None for k in ('norm','rms','l1','abs_p50','abs_p90','abs_p99','max_abs')})
        return result
    scale, ss = e['scale'], e['scaled_sum_squares']
    result['norm'] = _safe_product(scale, math.sqrt(ss))
    result['rms'] = _safe_product(scale, math.sqrt(ss/v.size))
    result['l1'] = _safe_product(scale, float(np.sum(np.abs(v/scale)))) if scale else 0.
    p = np.quantile(np.abs(v), [.5,.9,.99], method='linear')
    result.update(abs_p50=float(p[0]), abs_p90=float(p[1]), abs_p99=float(p[2]), max_abs=scale)
    return result


def infer_water(vs, dh):
    """Infer each top-connected exact-zero water column and midpoint interface.

    Every column needs water at its top and a solid below it. All-water, no-top
    water, buried water pockets, negative Vs, or nonfinite Vs fail closed rather
    than being silently interpreted as a simple water/solid interface.
    """
    vs = _finite_image(vs)
    if not math.isfinite(dh) or dh <= 0 or np.any(vs < 0):
        raise ValueError('positive DH and nonnegative Vs required')
    water = vs == 0
    ny, nx = vs.shape
    if not water[0].all() or water.all(axis=0).any():
        raise ValueError('ambiguous water: each column requires top water and underlying solid')
    first = np.argmax(~water, axis=0)
    expected = np.arange(ny)[:,None] < first[None,:]
    if not np.array_equal(water, expected):
        raise ValueError('water must be top-connected, with no buried zero-Vs pockets')
    last = first-1
    return {'water': water, 'last_water': last, 'first_solid': first,
            'last_water_center_m': (last+1)*dh,
            'first_solid_center_m': (first+1)*dh,
            'bottom_m': (first+.5)*dh}


def domains(vs, dh):
    """Fixed physical domains; tiny fixtures may have empty named domains."""
    g = infer_water(vs, dh)
    ny, nx = np.shape(vs)
    x = (np.arange(nx)+1)*dh
    z = (np.arange(ny)+1)*dh
    broadcast = lambda m: np.broadcast_to(m, (ny,nx)).copy()
    lateral = {'full': np.ones((ny,nx),bool),
               'interior': broadcast(((x >= 220) & (x <= 9800))[None,:]),
               'acquisition': broadcast(((x >= 800) & (x <= 8780))[None,:])}
    water = g['water']
    solid = ~water
    vertical = {'full': np.ones((ny,nx),bool), 'water': water.copy(), 'solid': solid,
                'first_solid': np.arange(ny)[:,None] == g['first_solid'][None,:]}
    for limit in (550,650,750):
        vertical[f'guard{limit}'] = solid & (z[:,None] >= g['bottom_m'][None,:]) & (z[:,None] < limit)
    for name, lo, hi, inclusive in (
        ('shallow',650,1000,False), ('1000_1500',1000,1500,False),
        ('1500_2000',1500,2000,False), ('2000_2500',2000,2500,False),
        ('2500_3000',2500,3000,False), ('3000_3280',3000,3280,True),
        ('deep',1000,3280,True)):
        vertical[name] = solid & broadcast(((z >= lo) & ((z <= hi) if inclusive else (z < hi)))[:,None])
    vertical['bottom'] = solid & broadcast((z > 3280)[:,None])
    return {'lateral': lateral, 'vertical': vertical}


def depth_metrics(a, vs, dh=20.):
    """Raw row profiles, region counts/fractions and dynamic-range diagnostics."""
    a = _finite_image(a)
    if a.shape != np.shape(vs):
        raise ValueError('image/model shape mismatch')
    ds = domains(vs,dh)
    rows = [stats(row) for row in a]
    total = _square(a)
    rowfractions = [_ratio_square(r['squared_norm_representation'],total) for r in rows]
    cumulative = None if total['mantissa'] == 0 else np.cumsum(rowfractions).tolist()
    result = {'squared_norm_label': 'squared Euclidean sensitivity norm, not physical seismic energy',
              'depth_centers_m': ((np.arange(a.shape[0])+1)*dh).tolist(),
              'rows': rows, 'row_squared_norm_fraction': rowfractions,
              'cumulative_squared_norm_fraction': cumulative, 'regions': {}, 'ratios': {}}
    for lname, lm in ds['lateral'].items():
        result['regions'][lname] = region = {}
        solid = _square(a[lm & ds['vertical']['solid']])
        full = _square(a[lm])
        for vname, vm in ds['vertical'].items():
            s = stats(a[lm & vm])
            e = s['squared_norm_representation']
            s['fraction_of_solid_squared_norm'] = _ratio_square(e,solid)
            s['fraction_of_lateral_full_squared_norm'] = _ratio_square(e,full)
            region[vname] = s
        r = {f'{key}_over_solid_squared_norm': region[key]['fraction_of_solid_squared_norm']
             for key in ('first_solid','guard550','guard650','guard750')}
        shallow, deep = region['shallow']['rms'], region['deep']['rms']
        r['deep_over_shallow_rms'] = _safe_ratio(deep,shallow)
        maxi, deepmax = region['full']['max_abs'], region['deep']['max_abs']
        r['deep_max_over_full_max'] = _safe_ratio(deepmax,maxi)
        r['dynamic_range_deficit_db'] = 20*(math.log10(maxi)-math.log10(deepmax)) if maxi and deepmax else None
        result['ratios'][lname] = r
    return result


def correlation(a, b, mask=None, centered=False):
    """Cosine (or centered cosine); empty/zero-variance denominators are None."""
    a, b = _array(a), _array(b)
    if a.shape != b.shape:
        raise ValueError('correlation shape mismatch')
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != a.shape:
            raise ValueError('correlation mask shape mismatch')
        a, b = a[mask], b[mask]
    a, b = a.ravel(), b.ravel()
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite correlation samples')
    if not a.size:
        return None
    sa, sb = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
    if not sa or not sb:
        return None
    a, b = a/sa, b/sb
    if centered:
        a, b = a-np.mean(a), b-np.mean(b)
    aa, bb = float(np.sum(a*a)), float(np.sum(b*b))
    if not aa or not bb:
        return None
    return float(np.clip(float(np.sum(a*b))/math.sqrt(aa*bb), -1., 1.))


def _shots(shots):
    if set(shots) != set(range(1,101)) or any(type(k) is not int for k in shots):
        raise ValueError('exact physical shot IDs 1..100 required')
    ordered = {i: _finite_image(shots[i]) for i in range(1,101)}
    shape = ordered[1].shape
    if any(a.shape != shape for a in ordered.values()):
        raise ValueError('shot image shapes differ')
    return ordered


def subset_membership():
    return {'odd': list(range(1,101,2)), 'even': list(range(2,101,2)),
            **{f'block{j+1:02d}': list(range(j*20+1,(j+1)*20+1)) for j in range(5)}}


def _sum(shots, ids):
    total = np.zeros_like(shots[1],dtype=np.float64)
    for i in sorted(ids):
        total += shots[i]
    if not np.isfinite(total).all():
        raise ValueError('FP64 subset accumulation overflow')
    return total


def subset_sums(shots):
    """Ascending physical-ID, unweighted FP64 sums; never replace raw aggregate."""
    shots = _shots(shots)
    return {name: _sum(shots,ids) for name, ids in subset_membership().items()}


def _coherence(a,b,mask):
    return {'uncentered': correlation(a,b,mask), 'centered': correlation(a,b,mask,True)}


def _regional_shots(shots, aggregate, mask):
    scale = max(float(np.max(np.abs(a[mask]))) if mask.any() else 0. for a in shots.values())
    if not scale:
        return {'count': int(mask.sum()), 'sum_individual_squared_norm_scaled': 0.,
                'common_amplitude_scale': 0., 'contribution_fractions': {str(i): None for i in shots},
                'concentration_sum_fraction_squared': None, 'effective_contributor_count': None,
                'shot_semblance': None, 'aggregate_norm_over_sum_individual_norms': None}
    individual = np.array([float(np.sum((shots[i][mask]/scale)**2)) for i in shots])
    den = float(np.sum(individual))
    fractions = individual/den
    sumsq = float(np.sum((aggregate[mask]/scale)**2))
    concentration = float(np.sum(fractions*fractions))
    return {'count': int(mask.sum()), 'common_amplitude_scale': scale,
            'sum_individual_squared_norm_scaled': den,
            'contribution_fractions': {str(i): float(v) for i,v in zip(shots,fractions)},
            'concentration_sum_fraction_squared': concentration,
            'effective_contributor_count': 1./concentration,
            'shot_semblance': sumsq/(len(shots)*den),
            'aggregate_norm_over_sum_individual_norms': math.sqrt(sumsq)/float(np.sum(np.sqrt(individual))),
            'interpretation': 'descriptive shared-amplitude concentration and semblance; correlated shots, not independent evidence'}


def shot_diagnostics(shots, vs, dh=20.):
    """Raw coherence/contribution diagnostics including all physical shots.

    Odd/even interleaves share aperture and are not independent experiments.
    Five block pairs expose aperture disagreement. Shot 68 is observational,
    never rejected. Leave-one-out influences are descriptive, not uncertainty
    intervals or geological acceptance criteria.
    """
    shots = _shots(shots)
    if shots[1].shape != np.shape(vs):
        raise ValueError('shot/model shape mismatch')
    ds = domains(vs,dh)
    subsets = subset_sums(shots)
    aggregate = _sum(shots,range(1,101))
    blocks = [f'block{i:02d}' for i in range(1,6)]
    result = {'membership': subset_membership(), 'accumulation': 'ascending physical-ID unweighted FP64',
              'per_shot': {}, 'regions': {}, 'local_windows': [], 'leave_one_out': {},
              'caveat': 'Coherence is descriptive; not proof of geology or independent shot statistics.'}
    for i,a in shots.items():
        result['per_shot'][str(i)] = {'total': stats(a), 'regions': {}}
    for ln,lm in ds['lateral'].items():
        result['regions'][ln] = {}
        for vn,vm in ds['vertical'].items():
            mask = lm & vm
            region = {'count': int(mask.sum()), 'odd_even': _coherence(subsets['odd'],subsets['even'],mask),
                      'block_pairs': {f'{u}_{v}': _coherence(subsets[u],subsets[v],mask)
                                      for u,v in itertools.combinations(blocks,2)},
                      'contributions': _regional_shots(shots,aggregate,mask)}
            without68 = aggregate-shots[68]
            e = _ratio_square(_square(shots[68][mask]),_square(aggregate[mask]))
            region['shot68_observation'] = {'not_rejected': True,
                                           'squared_norm_over_aggregate_squared_norm': e,
                                           'aggregate_vs_without68': _coherence(aggregate,without68,mask)}
            result['regions'][ln][vn] = region
            for i,a in shots.items():
                s = stats(a[mask])
                result['per_shot'][str(i)]['regions'][f'{ln}/{vn}'] = {
                    k:s[k] for k in ('count','norm','rms','max_abs','squared_norm','squared_norm_representation')}
    # Fixed tiled windows, declared before images are inspected. No dip following.
    x = (np.arange(aggregate.shape[1])+1)*dh
    for vn in ('shallow','1000_1500','1500_2000','2000_2500','2500_3000','3000_3280'):
        for lo,hi in ((800,2400),(2400,4000),(4000,5600),(5600,7200),(7200,8780)):
            lateral = (x >= lo) & ((x <= hi) if hi == 8780 else (x < hi))
            mask = ds['vertical'][vn] & lateral[None,:]
            result['local_windows'].append({'x_interval_m': [lo,hi], 'last_endpoint_inclusive': hi == 8780,
                'vertical_domain': vn, 'count': int(mask.sum()),
                'odd_even': _coherence(subsets['odd'],subsets['even'],mask),
                'contributions': _regional_shots(shots,aggregate,mask)})
    for vn in ('full','deep'):
        mask = ds['vertical'][vn]
        result['leave_one_out'][vn] = {str(i): {
            'removed_shot_squared_norm_over_aggregate_squared_norm': _ratio_square(_square(a[mask]),_square(aggregate[mask])),
            'aggregate_vs_leave_one_out': _coherence(aggregate,aggregate-a,mask)} for i,a in shots.items()}
    return result


def lateral_shift_control(a,b,mask,realizations=1999,seed=250501):
    """Deterministic independent lateral cyclic-shift investigative controls.

    Zero shifts are excluded; wrap is intentional *only for this null control*.
    Correlated/nonstationary shots invalidate a confirmatory p-value reading.
    This supplies effect sizes and empirical exceedance, not a numerical oracle.
    Settings must be frozen before evaluating real images.
    """
    a,b = _finite_image(a),_finite_image(b)
    if a.shape != b.shape or a.shape[1] < 2 or realizations < 1:
        raise ValueError('shift controls need compatible width>=2 and positive realizations')
    mask = np.asarray(mask,dtype=bool)
    observed = correlation(a,b,mask,True)
    rng = np.random.Generator(np.random.PCG64(seed))
    children = rng.integers(0,2**63,size=2,dtype=np.int64)
    left = np.random.Generator(np.random.PCG64(int(children[0])))
    right = np.random.Generator(np.random.PCG64(int(children[1])))
    controls = [correlation(np.roll(a,int(left.integers(1,a.shape[1])),axis=1),
                            np.roll(b,int(right.integers(1,b.shape[1])),axis=1),mask,True)
                for _ in range(realizations)]
    valid = [v for v in controls if v is not None]
    return {'observed_centered': observed, 'control_centered': controls, 'realizations': realizations,
            'seed': seed, 'generator': 'PCG64', 'child_seeds': [int(v) for v in children],
            'cyclic_wrap_control_only': True, 'nonzero_independent_shifts': True,
            'control_median': float(np.median(valid)) if valid else None,
            'control_p95': float(np.quantile(valid,.95,method='linear')) if valid else None,
            'empirical_absolute_exceedance': ((1+sum(abs(v)>=abs(observed) for v in valid))/(1+len(valid)))
                if valid and observed is not None else None,
            'caveat': 'Investigative nonstationary correlated-shot control, not a confirmatory p-value or geological oracle.'}
