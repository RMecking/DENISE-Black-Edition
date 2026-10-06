"""Predeclared discrete-coordinate maps and PSF reductions; never physics fixes."""
import hashlib
import math
import numpy as np
from .campaign import encoded


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def coefficients(models):
    vp, vs, rho = (np.asarray(models[k], dtype=np.float64) for k in ('vp', 'vs', 'rho'))
    if vp.shape != vs.shape or vp.shape != rho.shape or not all(np.isfinite(a).all() for a in (vp,vs,rho)):
        raise ValueError('finite matching original background required')
    if (vp <= 0).any() or (rho <= 0).any() or (vs < 0).any():
        raise ValueError('physical background required, including exact Vs=0')
    return 2*rho*vp*vp, 2*rho*vs*vs


def tangent(log_probe, models, *, native_dtype=np.float32):
    z = np.asarray(log_probe, dtype=np.float64)
    av, ass = coefficients(models)
    if z.shape != (2,*av.shape) or not np.isfinite(z).all():
        raise ValueError('two-channel finite log probe required')
    if np.any(z[1][np.asarray(models['vs']) == 0] != 0):
        raise ValueError('lnVs tangent is frozen in water; do not mask/repair')
    exact = np.stack((av*z[0] - 2*ass*z[1], ass*z[1]))
    result = np.asarray(exact, dtype=native_dtype)
    if not np.isfinite(result).all():
        raise ValueError('native mapping overflow')
    return result, exact


def transpose(native_cotangent, models):
    g = np.asarray(native_cotangent, dtype=np.float64)
    av, ass = coefficients(models)
    if g.shape != (2,*av.shape) or not np.isfinite(g).all():
        raise ValueError('two-channel finite native cotangent required')
    return np.stack((av*g[0], ass*(g[1]-2*g[0])))


def basis(spec, models, dh):
    shape = np.shape(models['vs']); row = round(spec['z_m']/dh)-1; col = round(spec['x_m']/dh)-1
    if (row+1)*dh != spec['z_m'] or (col+1)*dh != spec['x_m'] or not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise ValueError('probe must be an exact valid grid center')
    channel = ('lnvp','lnvs').index(spec['parameter'])
    z = np.zeros((2,*shape), dtype=np.float64); z[channel,row,col] = 1.
    native, exact = tangent(z,models)
    metadata = {**spec, 'row_col': [row,col], 'channel': channel, 'normalization': 'discrete unit basis, Euclidean norm1',
                'log_probe_sha256': array_hash(z), 'native_probe_sha256': array_hash(native),
                'exact_native_mapping_sha256': array_hash(exact),
                'native_nonzero_values': native[:,row,col].astype(float).tolist(),
                'exact_native_nonzero_values': exact[:,row,col].tolist(),
                'mapping_rounding_relative_norm': norm(native.astype(float)-exact)/norm(exact)}
    return z, native, exact, metadata


def norm(a):
    a = np.asarray(a,dtype=np.float64); scale = float(np.max(np.abs(a))) if a.size else 0.
    return scale*float(np.sqrt(np.sum((a/scale)**2))) if scale else 0.


def ratio(a,b):
    return float(a/b) if b != 0 else None


def metrics(response, spec, models, dh=20, target_radius=200):
    """Every domain/rule is declared in contract.json before production probes."""
    a = np.asarray(response,dtype=np.float64)
    if a.shape != (2,*np.shape(models['vs'])) or not np.isfinite(a).all():
        raise ValueError('finite two-channel response required')
    ny,nx = a.shape[1:]; yy,xx = np.indices((ny,nx)); x=(xx+1)*dh; z=(yy+1)*dh
    dx=x-spec['x_m']; dz=z-spec['z_m']; radius=np.hypot(dx,dz); target=radius<=target_radius
    vs=np.asarray(models['vs']); water=vs==0
    if not water[0].all() or water.all(axis=0).any():
        raise ValueError('ambiguous water geometry')
    first=np.argmax(~water,axis=0)
    if not np.array_equal(water,yy<first[None,:]):
        raise ValueError('non-top-connected water')
    regions={'water':water,'first_solid':yy==first[None,:],
             'guard550':(~water)&(z>=450)&(z<550), 'guard650':(~water)&(z>=450)&(z<650),
             'guard750':(~water)&(z>=450)&(z<750),'shallow':(~water)&(z>=650)&(z<1000),
             'deep':(~water)&(z>=1000)&(z<=3280), 'target':target,
             'injection_depth_neighborhood':np.abs(dz)<=200,
             'shallower_off_target':dz < -200,'deeper_off_target':dz > 200,
             'outside_target':~target,'bottom_cpml':z>3280,
             'lateral_cpml':(x<220)|(x>9800),'interior':(x>=220)&(x<=9800)&(z<=3280),
             'acquisition':(x>=800)&(x<=8780)}
    answer={'definition':'unweighted squared Euclidean grid-coordinate cotangent norm, not seismic energy or continuum Hessian',
            'probe':spec,'channels':{},'row_depth_m':((np.arange(ny)+1)*dh).tolist(),
            'target_radius_m':target_radius,'complete_grid_including_cpml':True}
    for k,name in enumerate(('lnvp','lnvs')):
        field=a[k]; scale=float(np.max(np.abs(field))); e=(field/scale)**2 if scale else np.zeros_like(field)
        total=float(e.sum()); peak=np.unravel_index(np.argmax(np.abs(field)),field.shape)
        region_stats={key:{'count':int(mask.sum()),'norm':norm(field[mask]),'squared_norm_fraction':ratio(float(e[mask].sum()),total),
                           'max_abs':float(np.max(np.abs(field[mask]))) if mask.any() else None} for key,mask in regions.items()}
        if total:
            cx=float(np.sum(e*x)/total); cz=float(np.sum(e*z)/total)
            mx=float(np.sum(e*(x-cx)**2)/total); mz=float(np.sum(e*(z-cz)**2)/total)
            order=np.argsort(radius.ravel(),kind='stable'); cumulative=np.cumsum(e.ravel()[order]); radii=radius.ravel()[order]
            r50=float(radii[np.searchsorted(cumulative,.5*total)]); r90=float(radii[np.searchsorted(cumulative,.9*total)])
        else:
            cx=cz=mx=mz=r50=r90=None
        outside=np.where(~target,np.abs(field),-1.); side=np.unravel_index(np.argmax(outside),field.shape)
        local=np.where(radius<=400,np.abs(field),-1.); seed=np.unravel_index(np.argmax(local),field.shape)
        connected=np.zeros(field.shape,bool)
        if local[seed]>0:
            allowed=(radius<=400)&(np.abs(field)>=.5*local[seed]); todo=[seed]
            while todo:
                j,i=todo.pop()
                if not(0<=j<ny and 0<=i<nx) or connected[j,i] or not allowed[j,i]:continue
                connected[j,i]=True; todo.extend(((j-1,i),(j+1,i),(j,i-1),(j,i+1)))
        cutx=field[spec['row_col'][0]]; cutz=field[:,spec['row_col'][1]]
        answer['channels'][name]={'norm':norm(field),'scaled_squared_norm':total,'amplitude_scale':scale,
            'peak_abs':scale,'peak_row_col':[int(v) for v in peak],'peak_x_z_m':[int((peak[1]+1)*dh),int((peak[0]+1)*dh)],
            'peak_offset_m':float(radius[peak]) if total else None,'centroid_x_z_m':[cx,cz],
            'second_moments_m2':[mx,mz],'sigma_x_z_m':[math.sqrt(mx),math.sqrt(mz)] if total else None,
            'width_2sigma_x_z_m':[2*math.sqrt(mx),2*math.sqrt(mz)] if total else None,'r50_m':r50,'r90_m':r90,
            'regions':region_stats,'secondary_half_height_lobe_count':int(connected.sum()),
            'secondary_half_height_lobe_fraction':ratio(float(e[connected].sum()),total),
            'strongest_fixed_sidelobe_abs':float(outside[side]) if (~target).any() and total else None,
            'strongest_sidelobe_x_z_m':[int((side[1]+1)*dh),int((side[0]+1)*dh)] if (~target).any() and total else None,
            'fixed_sidelobe_over_main_norm':ratio(region_stats['outside_target']['norm'],region_stats['target']['norm']),
            'row_rms':(scale*np.sqrt(np.mean(e,axis=1))).tolist(),
            'row_squared_norm_fraction':[ratio(float(v),total) for v in e.sum(axis=1)],
            'horizontal_cut':cutx.tolist(),'vertical_cut':cutz.tolist(),
            'horizontal_sign_changes':int(np.count_nonzero(np.sign(cutx[1:])*np.sign(cutx[:-1])<0)),
            'vertical_sign_changes':int(np.count_nonzero(np.sign(cutz[1:])*np.sign(cutz[:-1])<0))}
    same=answer['channels'][spec['parameter']]; other=answer['channels']['lnvs' if spec['parameter']=='lnvp' else 'lnvp']
    answer['cross_talk']={'full_other_over_same_norm':ratio(other['norm'],same['norm']),
        'target_other_over_same_norm':ratio(other['regions']['target']['norm'],same['regions']['target']['norm'])}
    answer['sampled_diagonal']=float(a[spec['channel'],*spec['row_col']])
    labels=[]
    if same['norm']:
        if same['r90_m']<=200 and same['peak_offset_m']<=40:labels.append('WELL LOCALIZED')
        elif same['peak_offset_m']<=200:labels.append('BROAD BUT TARGET-CENTERED')
        if same['regions']['guard650']['norm']>same['regions']['target']['norm']:labels.append('INTERFACE-DOMINATED')
        if other['norm']>same['norm']:labels.append('STRONG PARAMETER CROSS-TALK')
        if same['r90_m']>1000 or same['peak_offset_m']>400:labels.append('STRONG SIDELOBES / NONLOCAL RESPONSE')
    answer['labels']=labels or ['INDETERMINATE'];answer['labels_not_acceptance']=True
    return answer
