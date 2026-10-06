"""Accepted saved-product reductions/comparison; no runtime imports."""

import numpy as np

from .normal_operator import norm, ratio

def aperture(fields, entry, models):
    full=fields['full'];odd=fields['odd'];even=fields['even']
    spec=entry['groups']['full']['metrics']['probe'];same=spec['channel']
    m=models;depth=entry['groups']['full']['metrics']['row_depth_m']
    comparisons={'odd_even_full_cosine':cosine(odd,even),'odd_even_same_cosine':cosine(odd[same],even[same]),
        'odd_even_deep_cosine':deep_cosine(odd,even,m,depth),
        'deep_cosine_domain':'Previously frozen deep solid partition:1000<=z<=3280, full x, exact original Vs!=0; derived cosine only, response arrays unchanged.',
        'blocks':[],'block_pairs':[]}
    blocks=[]
    for i in range(1,6):
        field=fields[f'block{i}'];blocks.append(field)
        ms=entry['groups'][f'block{i}']['metrics'];c=ms['channels'][spec['parameter']]
        comparisons['blocks'].append({'block':i,'cosine_full':cosine(field,full),
          'deep_cosine_full':deep_cosine(field,full,m,depth),'r90_m':c['r90_m'],
          'peak_x_z_m':c['peak_x_z_m'],'peak_offset_m':c['peak_offset_m'],'target_fraction':c['regions']['target']['squared_norm_fraction'],
          'interface_fraction':c['regions']['guard650']['squared_norm_fraction'],'cross_talk':ms['cross_talk']['full_other_over_same_norm'],
          'target_norm':c['regions']['target']['norm'],'same_norm':c['norm']})
    for i in range(5):
        for j in range(i+1,5):comparisons['block_pairs'].append({'blocks':[i+1,j+1],
            'cosine':cosine(blocks[i],blocks[j]),'deep_cosine':deep_cosine(blocks[i],blocks[j],m,depth)})
    return comparisons

GROUPS=['full','odd','even','block1','block2','block3','block4','block5']

def groups_for(shot):
    if shot not in range(1,101): raise ValueError('physical shot')
    return ['full','odd' if shot%2 else 'even',f'block{(shot-1)//20+1}']

def reconstruct(items):
    ids=[shot for shot,h in items]
    if len(ids)!=100 or set(ids)!=set(range(1,101)): raise ValueError('100 unique physical shots required')
    sums={g:np.zeros_like(items[0][1],dtype=np.float64) for g in GROUPS}
    for shot,h in sorted(items,key=lambda row:row[0]):
        if h.dtype!=np.float64 or not np.isfinite(h).all(): raise ValueError('saved precision/finite')
        for g in groups_for(shot): sums[g]+=h
    return sums

def compact(entry):
    ms=entry['groups']['full']['metrics'];c=ms['channels'][ms['probe']['parameter']]
    return {'sampled_diagonal':entry['sampled_diagonal'],'summed_Jz_energy':entry['summed_Jz_energy'],
        'peak_x_z_m':c['peak_x_z_m'],'peak_offset_m':c['peak_offset_m'],
        'r50_m':c['r50_m'],'r90_m':c['r90_m'],'width_2sigma_x_z_m':c['width_2sigma_x_z_m'],
        'target_fraction':c['regions']['target']['squared_norm_fraction'],
        'guard650_fraction':c['regions']['guard650']['squared_norm_fraction'],
        'guard_over_target_norm':ratio(c['regions']['guard650']['norm'],c['regions']['target']['norm']),
        'guard_over_target_squared_norm':ratio(c['regions']['guard650']['squared_norm_fraction'],c['regions']['target']['squared_norm_fraction']),
        'outside_over_target_norm':c['fixed_sidelobe_over_main_norm'],
        'strongest_fixed_sidelobe_abs':c['strongest_fixed_sidelobe_abs'],
        'strongest_sidelobe_x_z_m':c['strongest_sidelobe_x_z_m'],
        'full_cross_talk_norm':ms['cross_talk']['full_other_over_same_norm'],
        'target_cross_talk_norm':ms['cross_talk']['target_other_over_same_norm'],
        'partitions':{k:c['regions'][k]['squared_norm_fraction'] for k in ('water','first_solid','guard550','guard650','guard750','shallow','deep','bottom_cpml','lateral_cpml')},
        'labels':ms['labels'],'definition':ms['definition']}

def compare(a,b):
    ca,cb=compact(a),compact(b)
    if a['groups']['full']['metrics']['probe']!=b['groups']['full']['metrics']['probe']: raise ValueError('same probe required')
    return {'distinct_operators':['H_15','H_5'],'H15':ca,'H5':cb,
        'H5_over_H15':{k:ratio(cb[k],ca[k]) for k in ('sampled_diagonal','summed_Jz_energy','r50_m','r90_m','target_fraction',
            'guard650_fraction','outside_over_target_norm','full_cross_talk_norm','target_cross_talk_norm')},
        'target_fraction_difference':cb['target_fraction']-ca['target_fraction'],
        'guard_fraction_difference':cb['guard650_fraction']-ca['guard650_fraction'],
        'interpretation_scope':'bandwidth/discretization sensitivity of the measured operator; not a dispersion-only attribution'}

def deep_cosine(a,b,models,row_depth_m):
    """Diagnostic restriction to the already-frozen deep solid partition.

    Never changes a response, a numerical dot-product contract, or a window.
    """
    depth=np.asarray(row_depth_m)[:,None]
    support=(np.asarray(models['vs'])!=0)&(depth>=1000)&(depth<=3280)
    return cosine(np.asarray(a)[:,support],np.asarray(b)[:,support])

def cosine(a, b):
    denominator = norm(a)*norm(b)
    return float(np.sum(np.asarray(a,np.float64)*np.asarray(b,np.float64),dtype=np.float64))/denominator if denominator else None
