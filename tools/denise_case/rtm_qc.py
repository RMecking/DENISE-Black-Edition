"""Display-only scientific plots and portable RTM reporting; raw bytes immutable."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np

from .config import CaseConfig
from .coordinates import coordinate_contract
from .generate import _json_bytes, _write_exact
from .geometry import read_sources, read_receivers
from .model_io import read_grid, sha256_file
from .report import _embedded_png, _pre
from .rtm import raw_inventory, read_raw_image, require_lane


def write_derived(run: Path, name: str, values: np.ndarray) -> Path:
    """Only exclusive/idempotent writes under derived/, never raw or path traversal."""
    if not re.fullmatch(r'[a-z][a-z0-9_]*\.npy', name):
        raise ValueError('safe derived .npy filename required')
    root = (run / 'derived').resolve()
    if root == (run / 'raw').resolve() or not root.is_relative_to(run.resolve()):
        raise ValueError('derived directory aliases raw or leaves the run')
    target = root / name
    if not target.resolve().is_relative_to(root):
        raise ValueError('derived target leaves derived directory')
    import io
    buf = io.BytesIO()
    np.save(buf, np.asarray(values), allow_pickle=False)
    _write_exact(target, buf.getvalue())
    return target


def rtm_qc(config: CaseConfig, run_name: str = 'full_rtm') -> dict:
    require_lane(config)
    if not re.fullmatch(r'[a-z][a-z0-9_]*', run_name):
        raise ValueError('safe run name required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run = config.root / 'runs' / run_name
    output = config.root / 'qc/rtm' / run_name
    output.mkdir(parents=True, exist_ok=True)
    execution = json.loads((run / 'run_metadata.json').read_text())
    raw = raw_inventory(config, run) if execution['returncode'] == 0 else {}
    nx, ny, dh = config.require('grid.nx'), config.require('grid.ny'), config.require('grid.dh_m')
    contract = coordinate_contract(nx, ny, dh)
    extent = np.asarray(contract['imshow_extent_m']) / 1000
    sources = read_sources(config.root / 'input/geometry/sources.dat')
    receivers = read_receivers(config.root / 'input/geometry/receivers.dat')
    figures = []

    def plot(values, title, name, *, lo=None, hi=None, diverging=False, units=''):
        target = output / name
        if target.exists():
            raise FileExistsError(f'refusing to replace QC artifact {target}')
        fig, ax = plt.subplots(figsize=(12,4.8), constrained_layout=True)
        image = ax.imshow(values, origin='upper', extent=extent, aspect='equal',
                          cmap='RdBu_r' if diverging else 'viridis', vmin=lo, vmax=hi)
        ax.scatter([s.x_m/1000 for s in sources],[s.y_m/1000 for s in sources],s=7,c='red',label='100 sources',zorder=3)
        ax.scatter([r.x_m/1000 for r in receivers],[r.y_m/1000 for r in receivers],s=3,c='black',label='400 OBC receivers',zorder=3)
        ax.set(xlabel='Horizontal x (km)', ylabel='Depth (km)', title=title)
        ax.legend(loc='lower left',fontsize=7)
        fig.colorbar(image,ax=ax,label=units)
        fig.savefig(target,dpi=150)
        plt.close(fig)
        figures.append({'path':target.relative_to(config.root).as_posix(),'title':title,'sha256':sha256_file(target)})

    for component in ('vp','vs','rho'):
        values = read_grid(config.root / 'input/models' / f'smooth2.{component}', nx, ny)
        plot(values, f'Smooth2 background {component}; nominal DENISE sample coordinates',f'background_{component}.png',units='kg/m³' if component=='rho' else 'm/s')
    for component in raw:
        path = run / 'raw' / f'migration.image_{component}_raw.bin'
        a = read_raw_image(path,nx,ny)
        plot(a,f'Raw FP64 {component} JT image — full-range linear display only',f'{component}_raw_linear.png',lo=float(a.min()),hi=float(a.max()),diverging=True,units='raw JT amplitude (unconditioned)')
        clip = float(np.percentile(np.abs(a),99)) or 1.0
        plot(a,f'Raw {component} values — DERIVED DISPLAY: symmetric ±P99={clip:.6g}',f'{component}_display_p99.png',lo=-clip,hi=clip,diverging=True,units='raw JT amplitude; colour saturation only')
    if raw and raw_inventory(config,run) != raw:
        raise RuntimeError('raw image identity changed during display-only QC')
    result = {'lane':config.require('case.id'),'run_name':run_name,'coordinates':contract,'raw_images':raw,'figures':figures,
              'status':'raw RTM QC' if raw else 'BLOCKED: background-only QC; no migration images exist',
              'raw_vs_derived':'All PNGs are derived display artifacts; numerical raw FP64 files are unchanged. P99 affects colour only.',
              'acquisition':{'source_count':len(sources),'receiver_count':len(receivers),'receiver_x_range_m':[receivers[0].x_m,receivers[-1].x_m],'receiver_spacing_m':20},
              'limitations':'No beauty-based acceptance; no conditioning, illumination compensation or density image. CUDA CPML peak unsampled.'}
    _write_exact(output / 'rtm_qc.json',_json_bytes(result))
    return result


def rtm_report(config: CaseConfig) -> Path:
    require_lane(config)
    read = lambda relative: json.loads((config.root / relative).read_text())
    validation = read('generated/true_forward/validation.json')
    audit = read('runs/forward_audit.json')
    build = read('runs/cuda_build/receipt.json')
    preflight = read('runs/preflight/run_metadata.json')
    complete = (config.root / 'runs/full_rtm/run_metadata.json').is_file()
    full = read('runs/full_rtm/run_metadata.json') if complete else {'status':'NOT EXECUTED: failed bounded preflight'}
    bridge = read('runs/full_rtm/provenance.json') if complete else read('runs/preflight/provenance.json')
    stage = 'full_rtm' if complete else 'preflight'
    qc = read(f'qc/rtm/{stage}/rtm_qc.json')
    observations = read(f'qc/rtm/{stage}/observations.json')
    forward_qc = read('qc/data/data_qc.json')
    figures = ''.join(_embedded_png(config.root/f['path'],f['title']) for f in qc['figures'])
    for component, component_qc in forward_qc['components'].items():
        for shot, details in component_qc['shots'].items():
            figures += _embedded_png(config.root/details['gather_figure'],f'FD4 true-data {component}, shot {shot}; display-only P99 clipping')
            figures += _embedded_png(config.root/details['trace_overlay_figure'],f'FD4 true-data {component}, shot {shot}, selected receiver traces')
    sections = [('Manifest / frozen identities',config.data),('FD4 dispersion/CFL and warnings',validation),
                ('FD4 true forward execution',read('generated/true_forward/run_metadata.json')),
                ('All-trace data validation / identities',audit),('CUDA frozen-core build / environment',build),
                ('Bounded preflight',preflight),('Full 100-shot execution / raw images',full),
                ('Representation bridge / local-to-authoritative shot identity',bridge),('QC metadata / coordinate convention',qc),
                ('Observable structure and numerical limitations',observations)]
    body = "<!doctype html><html><head><meta charset='utf-8'><title>A2-M9e4-FD4 Marmousi-II</title><style>body{font:15px system-ui;max-width:1200px;margin:2rem auto;padding:1rem}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f5f6;padding:1rem}.warning{background:#fff1cc;padding:1rem}</style></head><body><h1>Marmousi-II A2-M9e4-FD4</h1>"
    body += '<p>Separate matched Taylor-FD4 benchmark, not an FD8 RTM or a claim of FD8 numerical equivalence. A0/A1-FD8 is preserved byte-for-byte. Physical source/filter and acquisition are preserved; discretization matches the frozen M9e4 Taylor-FD4 contract.</p>'
    if not complete:
        body += '<p class="warning">BLOCKED: the unchanged smooth2 water layer has Vs=0 / mu=0 in 11,000 cells. Frozen M9e4 requires strictly positive mu; preflight rejected the model before GPU migration. No shear floor, core patch or full RTM was applied.</p>'
    body += ''.join("<p class='warning'>"+html.escape(w)+"</p>" for w in validation['warnings'])
    body += figures + ''.join('<h2>'+html.escape(title)+'</h2>'+_pre(value) for title,value in sections)
    body += '</body></html>\n'
    target = config.root / 'qc/report/index.html'
    _write_exact(target,body.encode())
    return target
