"""Unconditioned raw-image views and reproducible application-only QC evidence."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from .fluid2 import DEFAULT_CASE, settings, physical, image_checks, identity
from .rtm import read_raw_image
from .generate import _write_exact, _json_bytes


def render(name, case=DEFAULT_CASE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root, repo, cfg, accepted, binary=settings(case)
    run=root/'runs'/name
    metadata=json.loads((run/'run_metadata.json').read_text())
    if not metadata['complete']:
        raise ValueError('cannot present failed/incomplete run as successful QC')
    vp,vs,rho,_,_=physical(accepted)
    images={k:read_raw_image(run/f'raw/migration.image_{k}_raw.bin',500,174) for k in ('lambda','mu')}
    before={k:identity(run/f'raw/migration.image_{k}_raw.bin') for k in images}
    qc=image_checks(images['lambda'],images['mu'],vs)
    out=root/'qc'/name
    out.mkdir(parents=True,exist_ok=True)
    display={}
    fig,axes=plt.subplots(3,2,figsize=(14,10),constrained_layout=True)
    for col,(key,a) in enumerate(images.items()):
        maximum=float(np.max(np.abs(a)))
        p99=float(np.percentile(np.abs(a),99))
        display[key]={'linear_vmax':maximum,'p99_vmax':p99,'display_only':True}
        for row,limit,label in [(0,maximum,'full raw linear'),(1,p99,'P99 display clip')]:
            ax=axes[row,col]
            im=ax.imshow(a,origin='upper',extent=[10,10010,3490,10],aspect='auto',
                         cmap='seismic',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.axhline(450,color='k',linewidth=.6)
            ax.set(xlabel='x (m)',ylabel='depth (m)',title=f'g{key}: {label}; no raw modification')
            fig.colorbar(im,ax=ax)
        ax=axes[2,col]
        crop=a[17:28]
        limit=float(np.max(np.abs(crop)))
        im=ax.imshow(crop,origin='upper',extent=[10,10010,570,350],aspect='auto',
                     cmap='seismic',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.axhline(450,color='k',linewidth=.8)
        ax.set(xlabel='x (m)',ylabel='depth (m)',title=f'g{key}: interface rows 17..27, raw linear')
        fig.colorbar(im,ax=ax)
    fig.suptitle(f'CPU FLUID-2: physical shots {metadata["authoritative_shots"]}; water gMu exact +0')
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=140);plt.close(fig)
    _write_exact(out/'raw_and_interface.png',stream.getvalue())
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,(key,a) in zip(axes,images.items()):
        rows=qc['interface']['profiles'][key]
        ax.plot([r['depth_center_m'] for r in rows],[r['rms'] for r in rows],'-o',label='row RMS')
        ax.plot([r['depth_center_m'] for r in rows],[r['max_abs'] for r in rows],'-s',label='row max abs')
        ax.axvline(450,color='k',linewidth=.8)
        ax.set(xlabel='depth sample center (m)',ylabel='raw image units',title=key)
        ax.legend()
    stream=io.BytesIO();fig.savefig(stream,format='png',dpi=140);plt.close(fig)
    _write_exact(out/'interface_profiles.png',stream.getvalue())
    if before!={k:identity(run/f'raw/migration.image_{k}_raw.bin') for k in images}:
        raise ValueError('raw images changed during display QC')
    record={'run':name,'raw_identities':before,'qc':qc,'display':display,
            'plots':{p.name:identity(p) for p in sorted(out.glob('*.png'))},
            'no_conditioning_no_mask':True,'FLUID3_acceptance_claimed':False}
    _write_exact(out/'qc.json',_json_bytes(record))
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name');parser.add_argument('--case',type=Path,default=DEFAULT_CASE)
    args=parser.parse_args()
    result=render(args.name,args.case)
    print(json.dumps({'run':args.name,'plots':result['plots'],'water_mu':result['qc']['water_mu']},indent=2))
