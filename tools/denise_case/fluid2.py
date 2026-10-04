"""Canonical CPU-only FLUID-2 application integration; no numerical operator here.

Reuse the accepted FD4 bridge and data read-only. The historical CUDA lane and its
positive-mu guard remain historical: this lane explicitly accepts physical zero
mu under the new CPU contract, without passing any fluid classification to core.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_case
from .generate import _copy_exact, _write_exact, _json_bytes, _generator_identity
from .geometry import read_receivers, read_sources
from .model_io import read_grid, sha256_file
from .rtm import (lame_parameters, stress_source, validated_component,
                  read_raw_image, verify_baseline, mode2_records)
from .seismic import read_su
from .coordinates import coordinate_contract

CORE_SHA = 'f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d'
DEFAULT_CASE = Path(__file__).resolve().parents[2] / 'applications/marmousi2_fluid2/case.json'


def git(repo, *args):
    # Windows-created checkout on DrvFS: trust only this exact task directory,
    # for this read-only invocation, without changing global Git configuration.
    return subprocess.check_output(['git', '-c', f'safe.directory={Path(repo).resolve()}',
                                    '-c', 'core.safecrlf=false', *args], cwd=repo, text=True).strip()


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256_file(path)}


def settings(path=DEFAULT_CASE):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    repo = path.parents[2]
    if cfg['core_sha'] != CORE_SHA or cfg['backend'] != 'CPU-M9':
        raise ValueError('canonical CPU-M9 FLUID-2 only; CUDA fluid dispatch forbidden')
    binary = (path.parent / cfg['executable']).resolve()
    if binary != (repo / 'bin/denise').resolve():
        raise ValueError('CPU executable must be this canonical checkout/bin/denise')
    accepted = load_case(path.parent / cfg['accepted_case'])
    # The old manifest describes the accepted DATA, not the migration executable.
    mode2_records(accepted)  # freezes Taylor FD4 and the existing 122-record bridge
    if accepted.root == path.parent:
        raise ValueError('historical accepted case must remain separate and read-only')
    return path.parent, repo, cfg, accepted, binary


def require_core(repo, *, authorized_sha=CORE_SHA, remote=False):
    """Validate the explicitly frozen LOCAL execution identity, always offline.

    ``remote`` is retained for old callers, but never initiates network access.
    Binary/build/input identities remain separate checks in build/run/resume.
    A locally present moving origin ref is observation only, not authorization.
    """
    if not isinstance(authorized_sha, str) or not re.fullmatch(r'[0-9a-f]{40}', authorized_sha):
        raise ValueError('explicit full authorized execution SHA required')
    head = git(repo, 'rev-parse', 'HEAD')
    if head != authorized_sha:
        raise ValueError('local HEAD differs from authorized frozen execution SHA')
    dirty = git(repo, 'diff', authorized_sha, '--name-only', '--',
                'src', 'include', 'tests/physics', 'tests/utilities')
    untracked = git(repo, 'ls-files', '--others', '--exclude-standard', '--',
                    'src', 'include', 'tests/physics', 'tests/utilities')
    if dirty or untracked or git(repo, 'diff', '--cached', '--name-only'):
        raise ValueError('production/core tests or index changed')
    try:
        origin = git(repo, 'rev-parse', '--verify', '--quiet', 'refs/remotes/origin/modernization')
    except subprocess.CalledProcessError:
        origin = None
    return {'head': head, 'authorized_core_sha': authorized_sha,
            'observed_repository_state': {'origin_modernization': origin},
            'live_remote_checked': False, 'legacy_remote_argument': bool(remote),
            'branch': git(repo, 'branch', '--show-current') or '(detached)',
            'core_diff': dirty, 'staged_diff': ''}


def material_check(lam, mu, rho):
    if lam.ndim != 2 or lam.shape != mu.shape or lam.shape != rho.shape:
        raise ValueError('matching 2-D lambda/mu/rho required')
    if not all(np.isfinite(a).all() for a in (lam, mu, rho)) or np.any(mu < 0) or np.any(rho <= 0):
        raise ValueError('invalid canonical CPU material')
    if np.any(lam.astype(np.float64) + 2 * mu.astype(np.float64) <= 0):
        raise ValueError('nonpositive longitudinal modulus')
    return {'supported': True, 'zero_mu_cells': int(np.count_nonzero(mu == 0)),
            'required': 'finite FP32 lambda, nonnegative mu, positive rho/lambda+2mu', 'model_modified': False}


def physical(accepted):
    nx, ny = accepted.require('grid.nx'), accepted.require('grid.ny')
    vp, vs, rho = [read_grid(accepted.root / f'input/models/smooth2.{k}', nx, ny) for k in ('vp', 'vs', 'rho')]
    lam, mu = lame_parameters(vp, vs, rho)
    material_check(lam, mu, rho)
    return vp, vs, rho, lam, mu


def verify(path=DEFAULT_CASE):
    root, repo, cfg, accepted, binary = settings(path)
    state = require_core(repo, remote=True)
    baseline = verify_baseline(accepted)
    previous = json.loads((accepted.root / 'runs/final_verification.json').read_text())
    forward = accepted.root / 'generated/true_forward'
    for name, expected in [('runs/forward_audit.json', previous['FD4_data']['audit_sha256']),
                           ('generated/true_forward/run_metadata.json', previous['FD4_data']['metadata_sha256']),
                           ('generated/true_forward/provenance.json', previous['FD4_data']['provenance_sha256']),
                           ('generated/true_forward/denise.inp', previous['FD4_data']['resolved_input_sha256'])]:
        if sha256_file(accepted.root / name) != expected:
            raise ValueError(f'accepted FD4 provenance changed: {name}')
    if accepted.sha256 != previous['FD4_data']['manifest_canonical_sha256']:
        raise ValueError('accepted FD4 manifest changed')
    audit = json.loads((accepted.root / 'runs/forward_audit.json').read_text())
    sources = read_sources(accepted.root / 'input/geometry/sources.dat')
    receivers = read_receivers(accepted.root / 'input/geometry/receivers.dat')
    nt = round(accepted.require('time.time_s') / accepted.require('time.dt_s'))
    for entry in audit['files']:
        data = forward / entry['path'].replace('\\', '/')
        if sha256_file(data) != entry['sha256'] or data.stat().st_size != entry['bytes']:
            raise ValueError(f'accepted data changed: {data}')
        validated_component(data, entry['component'], sources[entry['shot']-1], receivers, nt, accepted.require('time.dt_s'))
    # Freeze ALL historical products, including reports, failed-run evidence and
    # source wavelets, not just the accepted SU receiver data.
    preserved = {}
    for old_root in (accepted.root, accepted.resolve('rtm.baseline_root')):
        preserved[old_root.name] = {p.relative_to(old_root).as_posix(): identity(p)
                                  for p in sorted(old_root.rglob('*')) if p.is_file()}
    snapshot = root / 'runs/preserved_inputs.json'
    _write_exact(snapshot, _json_bytes(preserved))
    # MODE=0 files are unchanged; the diff comprises only M9 Born/JT and CUDA
    # host/forward capability guards, never the classical FD_PSV data generator.
    differences = git(repo, 'diff', accepted.require('provenance.denise_core_sha'), CORE_SHA,
                      '--name-only', '--', 'src', 'include').splitlines()
    allowed = {'src/CUDA/m9_elastic_forward.cu', 'src/CUDA/m9_elastic_host.c',
               'src/PSV/elastic_psv_born.c', 'src/PSV/elastic_psv_born_mpi.c',
               'src/PSV/elastic_psv_migration_mode2.c'}
    if not set(differences) <= allowed:
        raise ValueError('direct-forward source identity changed: reassess data reuse')
    vp, vs, rho, lam, mu = physical(accepted)
    water = vs == 0  # QC selector from authoritative source Vs; never sent to core
    if not np.all(mu[water] == 0):
        raise ValueError('source physical water no longer exact zero')
    rows = np.where(np.any(water, axis=1))[0]
    result = {'repository': state, 'manifest': identity(Path(path)), 'accepted_manifest_sha256': accepted.sha256,
              'FD8_unchanged': True, 'FD8_files': len(baseline['baseline_files']),
              'FD4_all_200_components_validated': len(audit['files']) == 200,
              'accepted_audit': identity(accepted.root/'runs/forward_audit.json'),
              'preservation_snapshot': identity(snapshot), 'classical_forward_sources_unchanged': True,
              'core_source_changes_since_data_run': differences,
              'physical_water': {'cells': int(water.sum()), 'vs_range': [float(vs[water].min()),float(vs[water].max())],
                                 'mu_range': [float(mu[water].min()),float(mu[water].max())],
                                 'rows_zero_based': [int(rows.min()), int(rows.max())],
                                 'depth_sample_centers_m': [float((rows.min()+1)*20),float((rows.max()+1)*20)]},
              'previous_blocker': previous['preflight'], 'coordinate_contract': coordinate_contract(500,174,20)}
    _write_exact(root/'runs/verification.json', _json_bytes(result))
    return result


def build(path=DEFAULT_CASE):
    root, repo, cfg, accepted, binary = settings(path)
    require_core(repo, remote=True)
    out = root / 'runs/cpu_build'
    if (out / 'receipt.json').exists():
        raise FileExistsError('preserve build receipt; no rebuild over accepted executable')
    started = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    records = []
    for index, command in enumerate((['make','-C','libcseife','-j4'], ['make','-C','src','-j4','denise'])):
        p = subprocess.run(command, cwd=repo, capture_output=True, text=True)
        _write_exact(out/f'{index}_stdout.txt', p.stdout.encode())
        _write_exact(out/f'{index}_stderr.txt', p.stderr.encode())
        records.append({'command': command, 'returncode': p.returncode})
        if p.returncode:
            _write_exact(out/'failure.json', _json_bytes({'commands':records}))
            raise RuntimeError('CPU build failed; inspect build logs')
    libs = subprocess.check_output(['ldd',str(binary)],text=True)
    if 'cudart' in libs.lower():
        raise ValueError('CPU executable links CUDA')
    tracked = git(repo,'ls-files','src','include','tests/physics','tests/utilities').splitlines()
    result = {'core_sha': CORE_SHA, 'commands': records, 'binary': identity(binary),
              'started_utc': started, 'ended_utc': datetime.now(timezone.utc).isoformat(),
              'runtime_seconds': time.perf_counter()-clock, 'linked_libraries': libs,
              'compiler': subprocess.check_output(['mpicc','--version'],text=True),
              'mpi': subprocess.check_output(['mpiexec','--version'],text=True),
              'core_and_oracle_source_sha256': {p:sha256_file(repo/p) for p in tracked},
              'application_generator': _generator_identity(CORE_SHA)}
    require_core(repo)
    _write_exact(out/'receipt.json',_json_bytes(result))
    return result


def prepare(shots, name, path=DEFAULT_CASE):
    root, repo, cfg, accepted, binary = settings(path)
    if shots != sorted(set(shots)) or not shots or any(s not in range(1,101) for s in shots):
        raise ValueError('ascending unique physical shots 1..100 required')
    if not name.replace('_','').isalnum() or '/' in name or '\\' in name:
        raise ValueError('safe isolated run name required')
    run = root/'runs'/name
    if (run/'attempt.json').exists() or (run/'run_metadata.json').exists() or ((run/'raw').exists() and any((run/'raw').iterdir())):
        raise FileExistsError('no relaunch or overwrite of attempted/raw run')
    old = accepted.root/'runs/preflight'
    old_provenance = json.loads((old/'provenance.json').read_text())
    original = old_provenance['background_original']
    vp, vs, rho, lam, mu = physical(accepted)
    for k, a in zip(('lam','mu','rho'), (lam,mu,rho)):
        expected = old_provenance['background_prepared'][k]
        p = old/expected['path']
        if sha256_file(p) != expected['sha256'] or a.T.astype('<f4').tobytes() != p.read_bytes():
            raise ValueError('physical material changed or x-major mismatch')
        _copy_exact(p,run/expected['path'])
    for k, expected in original.items():
        if sha256_file(accepted.root/expected['path']) != expected['sha256']:
            raise ValueError('physical source model changed')
    sources = read_sources(accepted.root/'input/geometry/sources.dat')
    receivers = read_receivers(accepted.root/'input/geometry/receivers.dat')
    geometry = str(len(shots))+'\n'+''.join(f'{sources[s-1].x_m:.17g} 0 {sources[s-1].y_m:.17g} 0 0 1 0 1\n' for s in shots)
    _write_exact(run/'source/sources.dat',geometry.encode('ascii'))
    _copy_exact(accepted.root/'input/geometry/receivers.dat',run/'receiver/receivers.dat')
    # Preserve the exact old input including comments: all scientific records are
    # identical. Number of physical shots is specified in sources.dat only.
    _copy_exact(old/'denise.inp',run/'denise.inp')
    _copy_exact(old/'workflow.inp',run/'workflow.inp')
    forward = accepted.root/'generated/true_forward'
    audit = json.loads((accepted.root/'runs/forward_audit.json').read_text())
    known = {e['path']:e for e in audit['files']}
    nt, dt = 3000, .002
    entries = []
    for local, shot in enumerate(shots,1):
        entry = {'authoritative_shot':shot, 'core_shot_index':local, 'components':{}}
        for component in ('vx','vy'):
            data = forward/f'data/observed_{component}.su.shot{shot}'
            if sha256_file(data) != known[data.relative_to(forward).as_posix()]['sha256']:
                raise ValueError('direct data differs from accepted audit')
            matrix = validated_component(data,component,sources[shot-1],receivers,nt,dt)
            target = run/f'prepared/observed.{component}.shot_{local}.bin'
            _write_exact(target,matrix.tobytes(order='C'))
            entry['components'][component] = {'original_SU':identity(data),'prepared':identity(target),
                                               'shape_time_receiver':list(matrix.shape),'component':component}
        matches = list((forward/'model').glob(f'true_source_signal.*.su.shot{shot}'))
        if len(matches) != 1:
            raise ValueError('exactly one saved filtered source required')
        traces = read_su(matches[0],component='source')
        if len(traces)!=1 or traces[0].samples.size!=nt or traces[0].dt_s!=dt:
            raise ValueError('saved source sample/time mismatch')
        target = run/f'prepared/source.shot_{local}.bin'
        _write_exact(target,stress_source(traces[0].samples,dt).astype('<f4').tobytes())
        entry['source'] = {'saved_filtered_wavelet':identity(matches[0]),'prepared':identity(target)}
        entries.append(entry)
    replay_equal = None
    if shots == [1,50,100]:
        relevant = [old/'denise.inp', old/'workflow.inp', old/'source/sources.dat', old/'receiver/receivers.dat']
        relevant += list((old/'model').glob('*')) + list((old/'prepared').glob('*'))
        replay_equal = all(p.read_bytes()==(run/p.relative_to(old)).read_bytes() for p in relevant)
        if not replay_equal:
            raise ValueError('exact previously blocked scientific request changed')
    for name2 in ('raw','data','log'):
        (run/name2).mkdir(parents=True,exist_ok=True)
    _write_exact(run/'provenance.json',_json_bytes({'core_sha':CORE_SHA,'backend':'CPU-M9',
                 'manifest':identity(Path(path)), 'accepted_manifest_sha256':accepted.sha256,
                 'exact_previous_request_byte_identity':replay_equal, 'bridge':entries,
                 'inputs':{p.relative_to(run).as_posix():identity(p) for p in sorted(run.rglob('*')) if p.is_file()},
                 'generator':_generator_identity(CORE_SHA), 'model_conversion':'unchanged FP32 x-major/y-fastest',
                 'data_conversion':'chronological [time,receiver] FP32, vx/vy separate, no resampling',
                 'source_conversion':'psource.c FP32 endpoint/interior difference / DT, no 1/2'}))
    return run


def stats(a):
    return {'l2':float(np.linalg.norm(a)), 'min':float(a.min()), 'max':float(a.max()),
            'max_abs':float(np.max(np.abs(a))), 'nonzero_count':int(np.count_nonzero(a))}


def image_checks(lam_image, mu_image, authoritative_vs):
    if lam_image.shape != mu_image.shape or lam_image.shape != authoritative_vs.shape:
        raise ValueError('raw image/source model shape mismatch')
    if not all(np.isfinite(a).all() for a in (lam_image,mu_image)):
        raise ValueError('raw images nonfinite')
    water = authoritative_vs == 0  # diagnostic only; never modify raw image
    solid = ~water
    wm = mu_image[water]
    if np.any(wm != 0) or np.any(np.signbit(wm)):
        raise ValueError('raw water gMu must be exact +0, before any display/processing')
    interior = water.copy(); interior[0] = False
    if not np.any(lam_image[interior] != 0) or not np.any(lam_image[solid] != 0) or not np.any(mu_image[solid] != 0):
        raise ValueError('expected interior-fluid lambda / solid lambda/mu sensitivity absent')
    result = {'water_lambda':stats(lam_image[water]), 'water_mu':stats(wm),
              'solid_lambda':stats(lam_image[solid]), 'solid_mu':stats(mu_image[solid]),
              'water_mu_exact_positive_zero':True, 'post_mask_applied':False,
              'water_interior_lambda':stats(lam_image[interior]),
              'interpretation':'Water gMu=+0 is the restricted parameter-space result (fluid mu frozen), not absence of active wavefield/adjoint or lambda sensitivity.'}
    rows = np.where(np.any(water,axis=1))[0]
    boundary = int(rows.max())
    profiles = {}
    for key, a in [('lambda',lam_image),('mu',mu_image)]:
        selected = range(max(0,boundary-4),min(a.shape[0],boundary+6))
        profiles[key] = [{'row_zero_based':y,'depth_center_m':(y+1)*20,
                          **stats(a[y]), 'rms':float(np.sqrt(np.mean(a[y]*a[y]))),
                          'neighbor_difference_l2':float(np.linalg.norm(np.diff(a[y]))),
                          'alternating_projection_fraction':float(abs(np.sum(a[y]*((-1.)**np.arange(a.shape[1]))))/(np.sum(np.abs(a[y])) or 1))}
                         for y in selected]
    result['interface'] = {'last_water_row':boundary, 'first_solid_row':boundary+1,
                            'profiles':profiles, 'finite':True,
                            'acceptance_scope':'application layout/finite/profile QC only; NOT FLUID-3 interface refinement acceptance'}
    return result


def run(shots, name, path=DEFAULT_CASE):
    root, repo, cfg, accepted, binary = settings(path)
    require_core(repo,remote=True)
    verification = json.loads((root/'runs/verification.json').read_text())
    if not verification['FD4_all_200_components_validated'] or not verification['FD8_unchanged']:
        raise ValueError('accepted data/baseline verification required before solver execution')
    receipt_path = root/'runs/cpu_build/receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt['core_sha']!=CORE_SHA or sha256_file(binary)!=receipt['binary']['sha256']:
        raise ValueError('CPU binary/build receipt identity mismatch')
    if any(sha256_file(repo/p)!=h for p,h in receipt['core_and_oracle_source_sha256'].items()):
        raise ValueError('frozen build sources or core oracles changed')
    execution = prepare(shots,name,path)
    command = ['mpiexec','-np','1',str(binary),'denise.inp','workflow.inp']
    started = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    # Durable launch marker survives interruption before final metadata exists.
    # Interrupted/failed shots remain preserved and cannot silently relaunch.
    _write_exact(execution/'attempt.json',_json_bytes({'core_sha':CORE_SHA,'command':command,
                 'authoritative_shots':shots,'started_utc':started,'binary_sha256':sha256_file(binary)}))
    p = subprocess.run(command,cwd=execution,capture_output=True,text=True)
    for key, text in [('stdout',p.stdout),('stderr',p.stderr)]:
        _write_exact(execution/f'{key}.txt',text.encode())
    result = {'core_sha':CORE_SHA,'backend':'CPU-M9','binary_sha256':sha256_file(binary),
              'build_receipt_sha256':sha256_file(receipt_path), 'command':command,'cwd':str(execution),
              'started_utc':started,'ended_utc':datetime.now(timezone.utc).isoformat(),
              'runtime_seconds':time.perf_counter()-clock,'returncode':p.returncode,
              'authoritative_shots':shots,'provenance_sha256':sha256_file(execution/'provenance.json'),
              'raw_images':None,'qc':None,'complete':False}
    try:
        if p.returncode:
            raise RuntimeError(f'CPU MODE=2 failed with exit {p.returncode}; STOP, no core repair')
        if 'M9 MODE=2 backend: CPU-M9' not in p.stdout:
            raise ValueError('CPU migration backend marker missing')
        images = {}
        for k in ('lambda','mu'):
            target = execution/f'raw/migration.image_{k}_raw.bin'
            images[k]=read_raw_image(target,500,174)
        result['raw_images']={k:{**identity(execution/f'raw/migration.image_{k}_raw.bin'),
                                   'shape_y_x':[174,500], 'dtype':'<f8','order':'row-major [y,x]',
                                   'finite':True,**stats(a)} for k,a in images.items()}
        result['qc']=image_checks(images['lambda'],images['mu'],physical(accepted)[1])
        result['complete']=True
    except (ValueError,RuntimeError,FileNotFoundError) as error:
        result['application_error']=str(error)
    _write_exact(execution/'run_metadata.json',_json_bytes(result))
    require_core(repo)
    if not result['complete']:
        raise RuntimeError(result['application_error'])
    return result


def plan(path=DEFAULT_CASE):
    root, repo, cfg, accepted, binary = settings(path)
    # No implicit long campaign. Each CPU process is one complete physical shot;
    # only verified successful metadata can be resumed/skipped.
    batches = [list(range(first,min(first+cfg['batch_size'],101))) for first in range(1,101,cfg['batch_size'])]
    result = {'core_sha':CORE_SHA,'batches':batches,'run_name_pattern':'shot_001..shot_100',
              'resume_policy':'skip only validated complete metadata with identical executable/input/raw hashes; never relaunch an attempted run',
              'raw_retention':'per-shot FP64 lambda/mu pairs, no post-mask or conditioning',
              'aggregation':'optional separate FP64 unweighted sum in ascending physical-shot order; require all 100 verified pairs',
              'expected_raw_bytes_per_shot':1392000,'estimated_prepared_bytes_100':961200000,
              'command_template':'python -m tools.denise_case.fluid2 batch --shots 1,2,3,4,5',
              'no_automatic_monolithic_run':True}
    _write_exact(root/'runs/campaign_plan.json',_json_bytes(result))
    return result


def batch(shots,path=DEFAULT_CASE):
    root, repo, cfg, accepted, binary = settings(path)
    require_core(repo,remote=True)
    if shots != sorted(set(shots)) or not shots or len(shots)>cfg['batch_size']:
        raise ValueError('one deterministic ascending batch of at most five physical shots')
    results=[]; newly_executed=0
    for shot in shots:
        name=f'shot_{shot:03d}'
        metadata=root/'runs'/name/'run_metadata.json'
        if metadata.exists():
            old=json.loads(metadata.read_text())
            if not old.get('complete') or old['core_sha']!=CORE_SHA or old['authoritative_shots']!=[shot] or old['binary_sha256']!=sha256_file(binary):
                raise ValueError('attempted shot not safely resumable: preserve and report HOLD')
            prov=metadata.parent/'provenance.json'
            if sha256_file(prov)!=old['provenance_sha256']:
                raise ValueError('completed-shot provenance changed')
            provenance=json.loads(prov.read_text())
            if sha256_file(Path(path))!=provenance['manifest']['sha256'] or accepted.sha256!=provenance['accepted_manifest_sha256']:
                raise ValueError('completed-shot application/accepted manifest changed')
            for entry in provenance['inputs'].values():
                if sha256_file(Path(entry['path']))!=entry['sha256']:
                    raise ValueError('completed-shot input changed')
            for entry in old['raw_images'].values():
                if sha256_file(Path(entry['path']))!=entry['sha256']:
                    raise ValueError('completed-shot raw pair changed')
            arrays=[read_raw_image(Path(old['raw_images'][k]['path']),500,174) for k in ('lambda','mu')]
            image_checks(*arrays,physical(accepted)[1])
            results.append(old)
        else:
            results.append(run([shot],name,path))
            newly_executed+=1
    summary={'shots':shots,'completed':len(results),'failed':0,
             'per_shot_metadata':{str(s):identity(root/'runs'/f'shot_{s:03d}'/'run_metadata.json') for s in shots}}
    batch_name='_'.join(f'{s:03d}' for s in shots)
    _write_exact(root/'runs/batches'/f'batch_{batch_name}.json',_json_bytes(summary))
    return {**summary,'newly_executed':newly_executed,'resumed':len(results)-newly_executed,'results':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['verify','build','run','plan','batch'])
    parser.add_argument('--case',type=Path,default=DEFAULT_CASE)
    parser.add_argument('--shots',default='1,50,100')
    parser.add_argument('--name',default='blocker_replay')
    args=parser.parse_args()
    if args.action in ('verify','build','plan'):
        result=globals()[args.action](args.case)
    else:
        shots=[int(s) for s in args.shots.split(',')]
        result=run(shots,args.name,args.case) if args.action=='run' else batch(shots,args.case)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
