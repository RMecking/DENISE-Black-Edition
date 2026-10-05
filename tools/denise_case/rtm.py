"""Isolated matched-FD4 application bridge; numerical operators stay in the core."""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import CaseConfig
from .generate import _copy_exact, _generator_identity, _json_bytes, _parameter_records, _write_exact
from .geometry import read_receivers, read_sources
from .model_io import read_grid, sha256_file
from .seismic import read_su


CORE_SHA = "f315d157c2e817480990c97a3de45c4a23e465db"
BENCHMARK_SHA = "931131e8dc53649320a5620befe751ad32d2f7bd"


def require_lane(config: CaseConfig) -> None:
    if config.require("case.id") != "A2-M9e4-FD4":
        raise ValueError("FD4 operations require the separate A2-M9e4-FD4 lane")
    if config.require("physics.fd_order") != 4 or config.require("physics.max_relative_error") != 0:
        raise ValueError("matched M9e4 requires Taylor FD4, MAX_RELATIVE_ERROR=0")
    if config.require("provenance.denise_core_sha") != CORE_SHA or config.require("provenance.benchmark_sha") != BENCHMARK_SHA:
        raise ValueError("frozen core/reference identity mismatch")
    if config.root.resolve() == config.resolve("rtm.baseline_root"):
        raise ValueError("FD4 and FD8 roots must be distinct")


def _identity(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _baseline_files(root: Path) -> list[Path]:
    files = [root / "case.yaml", root / "provenance.yaml"]
    for directory in ("input", "generated/true_forward", "qc/models", "qc/geometry", "qc/data", "qc/report"):
        files.extend(p for p in (root / directory).rglob("*") if p.is_file())
    if (root / "qc/validation.json").is_file():
        files.append(root / "qc/validation.json")
    return sorted(set(files))


def freeze_baseline(config: CaseConfig) -> dict:
    """Capture original FD8 data, input, metadata, provenance and QC read-only."""
    require_lane(config)
    target = config.root / "runs" / "fd8_baseline_identity.json"
    if target.exists():
        return verify_baseline(config)
    repo = config.root.parents[1]
    git = lambda *args: subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    if git("rev-parse", "HEAD") != CORE_SHA or git("-C", "DENISE-Benchmark", "rev-parse", "HEAD") != BENCHMARK_SHA:
        raise ValueError("live frozen repository identity mismatch")
    if git("diff", "--name-only", "--", "src", "include", "tests/physics", "tests/utilities") or git("diff", "--cached", "--name-only"):
        raise ValueError("numerical or staged changes present")
    base = config.resolve("rtm.baseline_root")
    paths = _baseline_files(base)
    core_files = subprocess.check_output(["git", "ls-files", "src", "include"], cwd=repo, text=True).splitlines()
    result = {"core_sha": CORE_SHA, "benchmark_sha": BENCHMARK_SHA, "branch": git("branch", "--show-current"),
              "baseline_files": [_identity(p, base) for p in paths],
              "core_source_sha256": {p: sha256_file(repo / p) for p in core_files},
              "cpu_binary": _identity(config.resolve("backend.executable"), repo)}
    _write_exact(target, _json_bytes(result))
    return result


def verify_baseline(config: CaseConfig) -> dict:
    require_lane(config)
    record = json.loads((config.root / "runs/fd8_baseline_identity.json").read_text())
    base = config.resolve("rtm.baseline_root")
    expected = {r["path"]: r for r in record["baseline_files"]}
    actual = {p.relative_to(base).as_posix(): _identity(p, base) for p in _baseline_files(base)}
    if actual != expected:
        raise ValueError("protected A0/A1-FD8 artifacts changed")
    repo = config.root.parents[1]
    if any(sha256_file(repo / p) != h for p, h in record["core_source_sha256"].items()):
        raise ValueError("frozen core source changed")
    return record


def lame_parameters(vp: np.ndarray, vs: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FP32 expression/order from update_s_elastic_PML_PSV.c INVMAT1=1."""
    vp, vs, rho = (np.asarray(a, dtype=np.float32) for a in (vp, vs, rho))
    if vp.shape != vs.shape or vp.shape != rho.shape or vp.ndim != 2:
        raise ValueError("Vp/Vs/rho shapes must match and be 2-D")
    if not all(np.isfinite(a).all() for a in (vp, vs, rho)) or np.any(rho <= 0) or np.any(vs < 0) or np.any(vp <= 0):
        raise ValueError("invalid physical model")
    mu = (rho * vs) * vs
    lam = rho * (vp * vp - (np.float32(2) * vs) * vs)
    if not np.isfinite(lam).all() or not np.isfinite(mu).all() or np.any(lam + 2 * mu <= 0):
        raise ValueError("invalid lambda/mu conversion")
    return lam, mu


def stress_source(signal: np.ndarray, dt_s: float) -> np.ndarray:
    """Exact chronological stress injection of src/psource.c, no extra 1/2."""
    values = np.asarray(signal, dtype=np.float32)
    if values.ndim != 1 or values.size < 2 or dt_s <= 0 or not np.isfinite(values).all():
        raise ValueError("finite source vector with >=2 samples and positive DT required")
    out = np.empty_like(values)
    dt = np.float32(dt_s)
    out[0] = values[1] / dt
    out[1:-1] = (values[2:] - values[:-2]) / dt
    out[-1] = -values[-2] / dt
    if not np.isfinite(out).all():
        raise ValueError("nonfinite prepared source")
    return out


def material_envelope(lam: np.ndarray, mu: np.ndarray, rho: np.ndarray) -> dict:
    """Read-only frozen M9e4 material assessment; never invent a shear floor."""
    lam, mu, rho = (np.asarray(a) for a in (lam, mu, rho))
    if lam.ndim != 2 or lam.shape != mu.shape or lam.shape != rho.shape:
        raise ValueError('matching 2-D lambda/mu/rho required')
    finite = all(np.isfinite(a).all() for a in (lam, mu, rho))
    bad = (mu <= 0) | (rho <= 0) | ~np.isfinite(lam) | ~np.isfinite(mu) | ~np.isfinite(rho)
    bad |= lam.astype(np.float64) + 2 * mu.astype(np.float64) <= 0
    indices = np.argwhere(bad)
    return {'supported':bool(finite and not bad.any()), 'invalid_cells':int(bad.sum()),
            'zero_mu_cells':int((mu == 0).sum()), 'mu_min':float(mu.min()),
            'first_invalid_y_x':indices[0].tolist() if len(indices) else None,
            'required':'frozen M9e4: finite lambda, strictly positive mu/rho and positive lambda+2*mu',
            'model_modified':False}


def validated_component(path: Path, component: str, source, receivers: list, nt: int, dt: float) -> np.ndarray:
    """Resolve each trace by authoritative coordinates; never filename ordering."""
    traces = read_su(path, component=component,
                     shot_lookup={(source.x_m, source.y_m): source.ordinal},
                     receiver_lookup={(r.x_m, r.y_m): r.ordinal for r in receivers})
    if len(traces) != len(receivers):
        raise ValueError(f"{path}: expected {len(receivers)} receivers")
    by_receiver = {}
    for trace in traces:
        if trace.receiver_ordinal not in range(1, len(receivers) + 1) or trace.receiver_ordinal in by_receiver:
            raise ValueError(f"{path}: invalid or duplicate receiver identity")
        receiver = receivers[trace.receiver_ordinal - 1]
        if (trace.shot_id != source.ordinal or trace.source_xy_m != (source.x_m, source.y_m)
                or trace.receiver_xy_m != (receiver.x_m, receiver.y_m)
                or trace.header['dt_us'] == 0 or trace.dt_s != dt or trace.samples.size != nt
                or not np.isfinite(trace.samples).all()):
            raise ValueError(f"{path}: inconsistent SU component/geometry/time/finite contract")
        by_receiver[trace.receiver_ordinal] = trace.samples
    # Output disk order is [time][authoritative receiver ordinal].
    return np.stack([by_receiver[r.ordinal] for r in receivers], axis=1).astype('<f4', copy=False)


def forward_audit(config: CaseConfig) -> dict:
    require_lane(config)
    run = config.root / 'generated/true_forward'
    sources = read_sources(config.root / 'input/geometry/sources.dat')
    receivers = read_receivers(config.root / 'input/geometry/receivers.dat')
    nt = round(config.require('time.time_s') / config.require('time.dt_s'))
    dt = config.require('time.dt_s')
    metadata = json.loads((run / 'run_metadata.json').read_text())
    if metadata['returncode'] != 0 or metadata.get('dry_run'):
        raise ValueError('FD4 forward did not complete')
    expected = {r['path'].replace('\\', '/'): {**r, 'path': r['path'].replace('\\', '/')}
                for r in metadata['output_inventory']}
    names = {f'observed_{comp}.su.shot{s.ordinal}' for s in sources for comp in ('vx', 'vy')}
    if {p.name for p in (run / 'data').glob('*') if p.is_file()} != names:
        raise ValueError('unexpected or incomplete FD4 data discovery')
    results = []
    for source in sources:
        for component in ('vx', 'vy'):
            path = run / 'data' / f'observed_{component}.su.shot{source.ordinal}'
            matrix = validated_component(path, component, source, receivers, nt, dt)
            identity = _identity(path, run)
            if identity != expected.get(identity['path']):
                raise ValueError('FD4 observed data no longer matches execution inventory')
            results.append({**identity, 'shot': source.ordinal, 'component': component, 'finite': True,
                            'receiver_count': matrix.shape[1], 'samples': matrix.shape[0], 'dt_s': dt})
    result = {'lane': 'A2-M9e4-FD4', 'shots': len(sources), 'receivers': len(receivers), 'files': results,
              'bounded_QC_shots': [1, 50, 100], 'source_and_receiver_coordinates': 'all traces validated'}
    _write_exact(config.root / 'runs/forward_audit.json', _json_bytes(result))
    return result


def mode2_records(config: CaseConfig) -> list[str]:
    require_lane(config)
    replacements = {'MODE': '2', 'NPROCX': '1', 'NPROCY': '1', 'FD_ORDER': '4', 'MAX_RELATIVE_ERROR': '0',
                    'QUELLART': '3', 'INVMAT1': '3', 'DTINV': '1', 'NDT': '1',
                    'MFILE': 'model/background', 'WRITE_STF': '0'}
    records = []
    for line in _parameter_records(config):
        key = line.split(' =')[0]
        records.append(f'{key} ={replacements[key]}' if key in replacements else line)
    records += ['Q_PARAMETERIZATION =0', 'Q_APPROX_FMIN =0', 'Q_APPROX_FMAX =0', 'Q_APPROX_DF =0',
                'MIGRATION_SOURCE_PREFIX =prepared/source', 'MIGRATION_DATA_PREFIX =prepared/observed',
                'MIGRATION_IMAGE_PREFIX =raw/migration']
    assert len(records) == 122
    return records


def materialize_rtm(config: CaseConfig, shots: list[int], run_name: str) -> Path:
    require_lane(config)
    if not re.fullmatch(r'[a-z][a-z0-9_]*', run_name) or shots != sorted(set(shots)) or not shots:
        raise ValueError('safe run name and strictly ascending unique shots required')
    forward_audit(config)
    run = config.root / 'runs' / run_name
    # The core removes stale final pairs on failure; application never relaunches over a raw result.
    if (run / 'raw').exists() and any((run / 'raw').iterdir()):
        raise FileExistsError('refusing to overwrite a raw RTM result directory')
    for name in ('model', 'source', 'receiver', 'prepared', 'raw', 'data', 'log'):
        (run / name).mkdir(parents=True, exist_ok=True)
    nx, ny = config.require('grid.nx'), config.require('grid.ny')
    arrays = [read_grid(config.root / 'input/models' / f'smooth2.{k}', nx, ny) for k in ('vp', 'vs', 'rho')]
    lam, mu = lame_parameters(*arrays)
    background = {}
    for key, a in zip(('lam', 'mu', 'rho'), (lam, mu, arrays[2])):
        path = run / 'model' / f'background.{key}'
        _write_exact(path, np.asarray(a.T, dtype='<f4', order='C').tobytes())
        if not np.array_equal(read_grid(path, nx, ny), a):
            raise ValueError('model representation roundtrip failed')
        background[key] = _identity(path, run)
    sources = read_sources(config.root / 'input/geometry/sources.dat')
    receivers = read_receivers(config.root / 'input/geometry/receivers.dat')
    if any(s < 1 or s > len(sources) for s in shots):
        raise ValueError('shot outside authoritative geometry')
    geometry = str(len(shots)) + '\n'
    for shot in shots:
        s = sources[shot - 1]
        if s.source_type != 1:
            raise ValueError('only explosive sources supported')
        geometry += f'{s.x_m:.17g} 0 {s.y_m:.17g} 0 0 1 0 1\n'
    _write_exact(run / 'source/sources.dat', geometry.encode('ascii'))
    _copy_exact(config.root / 'input/geometry/receivers.dat', run / 'receiver/receivers.dat')
    forward = config.root / 'generated/true_forward'
    nt, dt = round(config.require('time.time_s') / config.require('time.dt_s')), config.require('time.dt_s')
    bridge = []
    for local, shot in enumerate(shots, 1):
        s = sources[shot - 1]
        entry = {'core_shot_index': local, 'authoritative_shot': shot, 'components': {}}
        for component in ('vx', 'vy'):
            path = forward / 'data' / f'observed_{component}.su.shot{shot}'
            matrix = validated_component(path, component, s, receivers, nt, dt)
            target = run / 'prepared' / f'observed.{component}.shot_{local}.bin'
            _write_exact(target, matrix.tobytes(order='C'))
            if target.read_bytes() != matrix.tobytes(order='C'):
                raise ValueError('time-major component roundtrip failed')
            entry['components'][component] = {'original_SU': _identity(path, forward), 'prepared': _identity(target, run),
                                               'shape_time_receiver': list(matrix.shape), 'dtype': '<f4', 'resampling': False}
        matches = list((forward / 'model').glob(f'true_source_signal.*.su.shot{shot}'))
        if len(matches) != 1:
            raise ValueError(f'expected exactly one saved filtered source for shot {shot}')
        ts = read_su(matches[0], component='source')
        if len(ts) != 1 or ts[0].samples.size != nt or ts[0].dt_s != dt:
            raise ValueError('invalid saved physical wavelet')
        source = stress_source(ts[0].samples, dt)
        target = run / 'prepared' / f'source.shot_{local}.bin'
        _write_exact(target, source.astype('<f4').tobytes())
        if not np.array_equal(np.fromfile(target, dtype='<f4'), source):
            raise ValueError('source roundtrip failed')
        entry['source'] = {'saved_filtered_wavelet': _identity(matches[0], forward), 'stress_increments': _identity(target, run),
                           'transform': 'psource.c endpoint/interior chronological FP32 difference / FP32 DT; no 1/2 factor'}
        bridge.append(entry)
    text = '# Application MODE=2; frozen M9e4 Taylor FD4\n' + ''.join(f'# record {i}\n{line}\n' for i, line in enumerate(mode2_records(config), 1))
    _write_exact(run / 'denise.inp', text.encode('ascii'))
    _write_exact(run / 'workflow.inp', b'# Unused MODE=2 workflow placeholder\n')
    _write_exact(run / 'resolved_case.json', config.canonical_bytes())
    provenance = {'lane': config.require('case.id'), 'core_sha': CORE_SHA, 'benchmark_sha': BENCHMARK_SHA,
                  'manifest_sha256': config.sha256, 'generator': _generator_identity(CORE_SHA),
                  'model_conversion': 'FP32 lambda=rho*(vp*vp-(2*vs)*vs); mu=(rho*vs)*vs; disk x-major/y-fastest',
                  'background_original': {k: _identity(config.root / 'input/models' / f'smooth2.{k}', config.root) for k in ('vp','vs','rho')},
                  'background_prepared': background, 'geometry_original': {k: _identity(config.root / 'input/geometry' / f'{k}.dat', config.root) for k in ('sources','receivers')},
                  'source_geometry_prepared': _identity(run / 'source/sources.dat', run),
                  'receiver_geometry_prepared': _identity(run / 'receiver/receivers.dat', run),
                  'resolved_input': _identity(run / 'denise.inp', run), 'shot_count': len(shots), 'bridge': bridge}
    _write_exact(run / 'provenance.json', _json_bytes(provenance))
    return run


def read_raw_image(path: Path, nx: int, ny: int) -> np.ndarray:
    if nx <= 0 or ny <= 0 or path.stat().st_size != nx * ny * 8:
        raise ValueError(f'invalid FP64 raw image size: expected {nx * ny * 8} bytes')
    values = np.fromfile(path, dtype='<f8').reshape(ny, nx)
    if not np.isfinite(values).all():
        raise ValueError('raw image contains nonfinite values')
    return values


def raw_inventory(config: CaseConfig, run: Path) -> dict:
    result = {}
    for key in ('lambda', 'mu'):
        path = run / 'raw' / f'migration.image_{key}_raw.bin'
        a = read_raw_image(path, config.require('grid.nx'), config.require('grid.ny'))
        result[key] = {**_identity(path, run), 'dtype': 'little-endian FP64', 'order': 'row-major [depth_y,x]',
                       'shape_y_x': list(a.shape), 'finite': True, 'min': float(a.min()), 'max': float(a.max()),
                       'mean': float(a.mean()), 'rms': float(np.sqrt(np.mean(a*a))),
                       'nonzero_count': int(np.count_nonzero(a)), 'abs_p99': float(np.percentile(np.abs(a),99))}
    return result


def _probe(command: list[str]) -> dict:
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
        return {'command': command, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': command, 'unavailable': str(error)}


def cuda_build_spec(*, nvcc: str | Path | None = None, cuda_archs: str | None = None,
                    environ=None) -> dict:
    """Pure command specification: explicit > environment > make/PATH defaults.

    No compiler execution, GPU detection or machine-directory search. Omitted
    overrides leave the frozen repository Makefile contract intact.
    """
    env = os.environ if environ is None else environ
    compiler = str(nvcc) if nvcc is not None else env.get('NVCC')
    archs = cuda_archs if cuda_archs is not None else env.get('CUDA_ARCHS')
    if compiler is not None and (not compiler.strip() or any(c in compiler for c in '\r\n\0$#')):
        raise ValueError('NVCC must be a nonempty compiler path/name without make expansion')
    if archs is not None:
        archs = ' '.join(str(archs).split())
        if not re.fullmatch(r'[1-9][0-9]{1,2}(?:[af])?(?: [1-9][0-9]{1,2}(?:[af])?)*', archs):
            raise ValueError('CUDA architectures must be space-separated architecture identifiers')
    command = ['make', '-C', 'src', 'denise_cuda']
    if compiler is not None:
        command.append(f'NVCC={compiler}')
    if archs is not None:
        command.append(f'CUDA_ARCHS={archs}')
    command.append('-j4')
    return {'command': command,
            'compiler_override': compiler, 'architecture_override': archs,
            'compiler_selection': 'explicit' if nvcc is not None else 'environment' if compiler is not None else 'make/PATH default',
            'architecture_selection': 'explicit' if cuda_archs is not None else 'environment' if archs is not None else 'repository make default',
            'toolkit_probe': [compiler if compiler is not None else 'nvcc', '--version']}


def build_cuda(config: CaseConfig, *, nvcc: str | Path | None = None,
               cuda_archs: str | None = None) -> dict:
    require_lane(config)
    verify_baseline(config)
    repo = config.root.parents[1]
    output = config.root / 'runs/cuda_build'
    if (output / 'receipt.json').exists():
        raise FileExistsError('preserve existing build evidence; use recorded binary')
    spec = cuda_build_spec(nvcc=nvcc, cuda_archs=cuda_archs)
    command = spec['command']
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    p = subprocess.run(command, cwd=repo, capture_output=True, text=True, check=False)
    for name, data in (('stdout.txt',p.stdout), ('stderr.txt',p.stderr)):
        _write_exact(output / name, data.encode())
    result = {'command': command, 'cwd': str(repo), 'core_sha': CORE_SHA, 'started_utc': started,
              'ended_utc': datetime.now(timezone.utc).isoformat(), 'runtime_seconds': time.perf_counter()-clock,
              'returncode': p.returncode, 'platform': platform.platform(),
              'build_selection': spec, 'toolkit': _probe(spec['toolkit_probe']),
              'compiler': _probe(['mpicc','--version']), 'mpi': _probe(['mpiexec','--version']),
              'gpu': _probe(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv']),
              'driver_support': _probe(['nvidia-smi'])}
    binary = config.resolve('rtm.executable')
    if p.returncode == 0:
        result['binary'] = _identity(binary, repo)
        result['linked_libraries'] = _probe(['ldd', str(binary)])
    _write_exact(output / 'receipt.json', _json_bytes(result))
    verify_baseline(config)
    if p.returncode:
        raise RuntimeError('CUDA build failed; inspect preserved build stdout/stderr/receipt')
    return result


def execute_rtm(config: CaseConfig, shots: list[int], run_name: str, *, timeout: float | None = None) -> dict:
    require_lane(config)
    verify_baseline(config)
    nx, ny = config.require('grid.nx'), config.require('grid.ny')
    physical = [read_grid(config.root / 'input/models' / f'smooth2.{k}', nx, ny) for k in ('vp','vs','rho')]
    lam, mu = lame_parameters(*physical)
    assessment = material_envelope(lam, mu, physical[2])
    if not assessment['supported']:
        raise ValueError(f"frozen M9e4 material envelope rejects {assessment['zero_mu_cells']} zero-mu cells; STOP, no model floor or core patch")
    if run_name == 'full_rtm':
        gate = json.loads((config.root / 'runs/preflight/run_metadata.json').read_text())
        if gate.get('returncode') != 0 or not gate.get('raw_images') or gate.get('authoritative_shots') != [1,50,100]:
            raise ValueError('full RTM requires successful bounded CUDA preflight')
        if shots != list(range(1,101)):
            raise ValueError('full benchmark requires exactly 100 shots')
    run = materialize_rtm(config, shots, run_name)
    if (run / 'run_metadata.json').exists():
        raise FileExistsError('execution metadata exists; refusing to relaunch')
    binary = config.resolve('rtm.executable')
    receipt = json.loads((config.root / 'runs/cuda_build/receipt.json').read_text())
    if sha256_file(binary) != receipt['binary']['sha256']:
        raise ValueError('CUDA executable differs from frozen-core build receipt')
    command = ['mpiexec', '-np', '1', str(binary), 'denise.inp', 'workflow.inp']
    started = datetime.now(timezone.utc).isoformat(); clock = time.perf_counter()
    try:
        p = subprocess.run(command, cwd=run, capture_output=True, text=True, timeout=timeout, check=False)
        rc, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as error:
        rc = -1
        stdout = error.stdout.decode(errors='replace') if isinstance(error.stdout, bytes) else (error.stdout or '')
        stderr = 'Timed out\n' + (error.stderr.decode(errors='replace') if isinstance(error.stderr, bytes) else (error.stderr or ''))
    for name, data in (('stdout.txt',stdout), ('stderr.txt',stderr)):
        _write_exact(run / name, data.encode())
    result = {'lane':config.require('case.id'), 'core_sha':CORE_SHA, 'benchmark_sha':BENCHMARK_SHA,
              'command':command, 'cwd':str(run), 'started_utc':started, 'ended_utc':datetime.now(timezone.utc).isoformat(),
              'runtime_seconds':time.perf_counter()-clock, 'returncode':rc, 'authoritative_shots':shots,
              'binary_sha256':sha256_file(binary), 'build_receipt_sha256':sha256_file(config.root / 'runs/cuda_build/receipt.json'),
              'provenance_sha256':sha256_file(run / 'provenance.json'), 'raw_images':None}
    if rc == 0:
        if 'M9 MODE=2 backend: CUDA-M9e-4' not in stdout:
            result['application_error'] = 'missing intended CUDA backend marker'
        else:
            try:
                result['raw_images'] = raw_inventory(config, run)
            except (ValueError, FileNotFoundError) as error:
                result['application_error'] = str(error)
    _write_exact(run / 'run_metadata.json', _json_bytes(result))
    verify_baseline(config)
    if rc or not result['raw_images']:
        raise RuntimeError('CUDA MODE=2 preflight/run failed; STOP and preserve evidence; no core repair')
    return result
