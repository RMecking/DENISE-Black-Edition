"""SH-S3: source time function estimation in the exact viscoelastic SH FWI
(docs/sh_s3_stf_contract.md, evidence section 7a; criteria frozen 2026-11-23).

Unit level: the estimator against an independent dense least-squares solution.
Driver level (the 32x32 fixture of the SH-S2 suite, real exact path):
  A1  the logged objective against an independent offline Phi (impulse-forward
      traces of a MODE-0 run, the legacy filter, the independent weights and a
      dense solve);
  A2  the directional derivative of the raw gradient against fourth-order
      central differences of the offline Phi with the frozen eps;
  A3  at the true model with same-solver data, B v equals B w_true;
  A5  identical Phi and v for 1 and 2 ranks;
  A6  multi-stage run with accepted steps, path equality, .stf output and
      reload; INV_STF = 3 keeps the signal of the stage;
  configuration aborts; INV_STF = 0 leaves the base behaviour (A4: the S-2 and
  L-BFGS suites, plus the bitwise base comparison recorded in the candidate
  report).
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import shutil
import struct
import subprocess

import numpy as np
import pytest

from tests.utilities.fwi_gradient import read_su_float_samples
from tests.test_sh_s2_data_operators import (  # noqa: F401  (fixtures are used by name)
    CELLS, DT, MODES, NS, PICKS, RECEIVER_GRID, TIME, WORKFLOW_HEADER,
    _direction, _f32, _filter, _misfit_rows, _parameter as _s2_parameter, _read_model,
    _reference_weights, _run, _signal_file, _write_model, harness, s2_case)

NREC = 6
EPS_STF = 1.0e-3
NW_DEFAULT = int(round(0.5 * TIME / DT))

# ----------------------------------------------------------------- unit level

STF_SOURCES = ("tests/utilities/sh_s3_stf_harness.c", "src/SH/visco_sh_exact_stf.c")


@pytest.fixture(scope="module")
def stf_harness(tmp_path_factory: pytest.TempPathFactory, repository_root: Path) -> Path:
    compiler = shutil.which("mpicc")
    assert compiler
    exe = tmp_path_factory.mktemp("sh_s3_unit") / "sh_s3_stf_harness"
    command = [compiler, "-std=c99", "-O2", "-fcommon",
               "-I", str(repository_root / "include"), "-I", str(repository_root / "libcseife"),
               *(str(repository_root / s) for s in STF_SOURCES), "-o", str(exe), "-lfftw3", "-lm"]
    done = subprocess.run(command, cwd=repository_root, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert done.returncode == 0, done.stdout
    return exe


def dense_estimate(g: np.ndarray, d: np.ndarray, w, nw: int, eps_rel: float | None = None,
                   eps: float | None = None, w_est=None, v_fixed=None):
    """Independent oracle: explicit convolution matrices, dense normal
    equations, numpy solve.  Sample index n (0-based) is sample n+1; n = 0 is
    not in the objective."""
    ntr, nt = g.shape
    w2 = np.ones((ntr, nt)) if w is None else np.asarray(w, dtype=np.float64) ** 2
    w2 = w2.copy()
    w2[:, 0] = 0.0
    # estimation weights (INV_STF = 3 with OFFSETC_STF); Phi always uses w
    we2 = w2 if w_est is None else np.asarray(w_est, dtype=np.float64) ** 2
    we2 = we2.copy()
    we2[:, 0] = 0.0
    a = np.zeros((nw, nw))
    b = np.zeros(nw)
    mats = []
    for i in range(ntr):
        c = np.zeros((nt, nw))
        for j in range(nw):
            c[j:, j] = g[i, :nt - j]
        cw = c * we2[i][:, None]
        a += c.T @ cw
        b += cw.T @ d[i]
        mats.append(c)
    if eps is None:
        eps = eps_rel * np.trace(a) / nw
    v = np.linalg.solve(a + eps * np.eye(nw), b) if v_fixed is None else np.asarray(v_fixed)
    dterm = 0.5 * sum(float(np.sum(w2[i] * (mats[i] @ v - d[i]) ** 2)) for i in range(ntr))
    return v, eps, dterm, 0.5 * eps * float(v @ v), np.trace(a) / nw


def _run_harness(exe: Path, tmp: Path, g, d, w, nw, eps_rel):
    ntr, nt = g.shape
    src, dst = tmp / "stf_in.bin", tmp / "stf_out.bin"
    with src.open("wb") as f:
        f.write(struct.pack("=4i", ntr, nt, nw, 0 if w is None else 1))
        f.write(struct.pack("=d", eps_rel))
        f.write(np.asarray(g, dtype="=f4").tobytes())
        f.write(np.asarray(d, dtype="=f4").tobytes())
        if w is not None:
            f.write(np.asarray(w, dtype="=f8").tobytes())
    done = subprocess.run([str(exe), str(src), str(dst)], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert done.returncode == 0, done.stdout
    raw = dst.read_bytes()
    status, iters = struct.unpack_from("=2i", raw, 0)
    md, eps, relres, dterm, eterm = struct.unpack_from("=5d", raw, 8)
    v = np.frombuffer(raw, "=f8", nw, 48)
    return dict(status=status, iterations=iters, mean_diag=md, eps=eps, relres=relres,
                dterm=dterm, eterm=eterm, v=v, stdout=done.stdout)


def _synthetic_case(seed: int, ntr: int, nt: int, nw_true: int):
    rng = np.random.default_rng(seed)
    t = np.arange(nt)
    g = np.zeros((ntr, nt), dtype=np.float32)
    for i in range(ntr):
        delay = 20 + 15 * i
        g[i] = (np.exp(-((t - delay) / 6.0) ** 2) * np.cos(0.3 * (t - delay))
                + 0.3 * np.exp(-((t - delay - 40) / 9.0) ** 2)).astype(np.float32)
    k = np.arange(nw_true)
    v_true = np.exp(-((k - min(12, nw_true // 2)) / 4.0) ** 2) * np.cos(0.5 * k)
    d = np.zeros((ntr, nt))
    for i in range(ntr):
        d[i] = np.convolve(g[i].astype(np.float64), v_true)[:nt]
    d = (d + 1e-3 * rng.standard_normal(d.shape) * np.abs(d).max()).astype(np.float32)
    return g, d


UNIT_CASES = {"unweighted_nw_eq_nt": (0, 120, 120, False), "weighted": (1, 150, 60, True),
              "single_sample": (2, 100, 1, True), "unweighted_short": (3, 200, 40, False)}


@pytest.mark.parametrize("name", sorted(UNIT_CASES))
def test_estimator_matches_dense_least_squares(stf_harness: Path, tmp_path: Path, name: str) -> None:
    seed, nt, nw, weighted = UNIT_CASES[name]
    g, d = _synthetic_case(seed, 5, nt, min(nw, 30))
    w = None
    if weighted:
        n = np.arange(nt)
        w = np.stack([np.where(n < 30 + 10 * i, np.exp(-2e-3 * (30 + 10 * i - n) ** 2), 1.0)
                      for i in range(5)])
        w[2] = 0.0                                   # a muted trace
        w[:, -1] = 0.0
    got = _run_harness(stf_harness, tmp_path, g, d, w, nw, EPS_STF)
    assert got["status"] == 0, got["stdout"]
    v, eps, dterm, eterm, md = dense_estimate(g.astype(np.float64), d.astype(np.float64), w, nw,
                                              eps_rel=EPS_STF)
    rel_v = np.linalg.norm(got["v"] - v) / np.linalg.norm(v)
    print(f"S3-UNIT {name} iterations={got['iterations']} relres={got['relres']:.2e} rel_v={rel_v:.2e}")
    assert abs(got["mean_diag"] - md) <= 1e-12 * md
    assert abs(got["eps"] - eps) <= 1e-12 * eps
    assert rel_v <= 1e-8
    assert abs(got["dterm"] - dterm) <= 1e-9 * (dterm + eterm)
    assert abs(got["eterm"] - eterm) <= 1e-9 * (dterm + eterm)


def test_estimator_rejects_invalid_input(stf_harness: Path, tmp_path: Path) -> None:
    g, d = _synthetic_case(4, 3, 50, 10)
    got = _run_harness(stf_harness, tmp_path, g, d, None, 60, EPS_STF)    # nw > nt
    assert got["status"] != 0
    got = _run_harness(stf_harness, tmp_path, np.zeros_like(g), d, None, 10, EPS_STF)  # eps = 0
    assert got["status"] != 0


# --------------------------------------------------------------- driver level

def _row(*, inv_stf=1, eps_stf=EPS_STF, offsetc_stf=0.0, pro=0.01, time_filt=0, fc_low=0.0,
         fc_high=20.0, order=2, timewin=0, gamma=1.0, twm=0.0, twp=0.0, mute=0,
         offsetc=10.0) -> str:
    values = [pro, time_filt, fc_low, fc_high, order, timewin, gamma, twm, twp, 0, 1, 1, 1,
              0, 0.5, 0.5, 0, 2, 0, inv_stf, offsetc_stf, eps_stf, 0, mute, offsetc, 1, 1, 0, 0, 0]
    return " ".join(str(v) for v in values)


def _parameter(path: Path, *, stf_length: float | None = None, **kwargs) -> None:
    _s2_parameter(path, **kwargs)
    if stf_length is not None:
        with path.open("a", encoding="ascii", newline="\n") as f:
            f.write(f"(STF_LENGTH) = {stf_length}\n")


def _workflow(case: dict, name: str, rows: list[str]) -> Path:
    path = case["root"] / f"wf3_{name}.inp"
    path.write_text(WORKFLOW_HEADER + "\n" + "\n".join(rows) + "\n", encoding="ascii")
    return path


def _fwi(case: dict, tag: str, model: str, workflow: Path, ranks: int = 1, itermax: int = 1,
         stf_length: float | None = None, lbfgs: int = 0):
    parameter = case["root"] / f"{tag}.inp"
    _parameter(parameter, mode=1, model=model, output=f"model/{tag}", nprocx=ranks,
               itermax=itermax, lbfgs=lbfgs, stf_length=stf_length)
    run = _run(case, parameter, workflow, ranks)
    assert run.returncode == 0 and "MPI_ABORT" not in run.stdout, run.stdout
    return _misfit_rows(case), run


def _read_stf(case: dict, tag: str, stage: int, it: int, nw: int) -> np.ndarray:
    raw = (case["root"] / f"model/{tag}_stage_{stage}_it_{it}.stf").read_bytes()
    assert len(raw) == 4 * nw          # one shot
    return np.frombuffer(raw, "=f4").astype(np.float64)


def _ops(name: str) -> dict:
    return dict(MODES[name])


def _impulse_traces(case: dict, harness: Path, tmp: Path, model: str, ops: dict, tag: str):
    """Receiver traces of F(m; B e_1) from a MODE-0 run (QUELLART = 3 file)."""
    root = case["root"]
    impulse = [1.0] + [0.0] * (NS - 1)
    if ops.get("time_filt", 0) == 1:
        _, (impulse,) = _filter(harness, tmp, [impulse], order=ops["order"], fc_high=ops["fc_high"],
                                fc_low=ops.get("fc_low", 0.0), dt=DT, tag=f"imp_{tag}")
    _signal_file(root / f"source/imp_{tag}_shot_1.dat", impulse)
    parameter = root / f"imp_{tag}.inp"
    _s2_parameter(parameter, mode=0, model=model, quellart=3, seis_vy=f"su/imp_{tag}_y.su")
    text = parameter.read_text(encoding="ascii").replace(
        "(SIGNAL_FILE) = source/signal.dat\n", f"(SIGNAL_FILE) = source/imp_{tag}\n")
    parameter.write_text(text, encoding="ascii")
    run = _run(case, parameter, root / "wf_plain.inp")
    assert run.returncode == 0, run.stdout
    raw = read_su_float_samples(root / f"su/imp_{tag}_y.su.shot1", NREC, NS)
    return np.array(raw, dtype=np.float64).reshape(NREC, NS)


def _observed_and_weights(case: dict, harness: Path, tmp: Path, ops: dict, tag: str):
    raw = read_su_float_samples(case["root"] / "su/observed_y.su.shot1", NREC, NS)
    observed = [raw[i * NS:(i + 1) * NS] for i in range(NREC)]
    filtered = ops.get("time_filt", 0) == 1
    if filtered:
        _, observed = _filter(harness, tmp, observed, order=ops["order"], fc_high=ops["fc_high"],
                              fc_low=ops.get("fc_low", 0.0), dt=DT, tag=f"obs_{tag}")
    weighted = filtered or ops.get("timewin", 0) != 0 or ops.get("mute", 0) != 0
    _, weights = _reference_weights(NS, DT, 10.0, ops.get("timewin", 0), ops.get("gamma", 1.0),
                                    ops.get("twm", 0.0), ops.get("twp", 0.0), ops.get("mute", 0),
                                    ops.get("offsetc", 10.0), 160.0, 160.0, RECEIVER_GRID, PICKS,
                                    mask_last=1 if filtered else 0)
    return (np.array(observed, dtype=np.float64),
            np.array(weights, dtype=np.float64) if weighted else None)


A1_CASES = [("base", 1), ("lowpass", 1), ("window1", 1), ("mute_far", 1), ("combined", 1),
            ("combined", 2), ("bandpass", 1), ("mute_near", 1)]      # last two: review C1


@pytest.mark.integration
@pytest.mark.parametrize("name,ranks", A1_CASES)
def test_objective_matches_the_independent_offline_phi(s2_case: dict, harness: Path, tmp_path: Path,
                                                       name: str, ranks: int) -> None:
    """A1: J logged by the exact path (INV_STF = 1, default support) equals the
    offline Phi = D(v) + eps/2 |v|^2 of a dense solve on MODE-0 impulse traces.
    Tolerances as SH-S2 R1: 2e-6 without filter, 1e-4 with filter."""
    ops = _ops(name)
    workflow = _workflow(s2_case, f"a1_{name}", [_row(**ops)])
    rows, run = _fwi(s2_case, f"a1_{name}_{ranks}", "model/start", workflow, ranks)
    assert "Exact SH STF: stage 1 INV_STF 1" in run.stdout
    j_log = float(rows[0][4])
    g = _impulse_traces(s2_case, harness, tmp_path, "model/start", ops, f"a1_{name}")
    d, w = _observed_and_weights(s2_case, harness, tmp_path, ops, f"a1_{name}")
    v, eps, dterm, eterm, _ = dense_estimate(g, d, w, NW_DEFAULT, eps_rel=_f32(EPS_STF))
    phi = dterm + eterm
    eps_log = float(re.search(r"Exact SH STF: stage 1 shot 1 eps (\S+)", run.stdout).group(1))
    v_out = _read_stf(s2_case, f"a1_{name}_{ranks}", 1, 1, NW_DEFAULT)
    rel = abs(j_log - phi) / phi
    rel_v = np.linalg.norm(v_out - v) / np.linalg.norm(v)
    print(f"S3-A1 {name} ranks={ranks} J_log={j_log:.7e} Phi_ref={phi:.9e} rel={rel:.2e} "
          f"eps_rel_err={abs(eps_log - eps) / eps:.2e} rel_v={rel_v:.2e} eps_term/Phi={eterm / phi:.2e}")
    tol = 1.0e-4 if ops.get("time_filt", 0) == 1 else 2.0e-6
    assert rel <= tol
    assert abs(eps_log - eps) <= 1e-5 * eps
    assert rel_v <= 1e-4


A2_CASES = [("base", 1), ("lowpass", 1), ("combined", 1), ("combined", 2),
            ("bandpass", 1), ("bandpass", 2), ("mute_near", 1)]      # last three: review C1


@pytest.mark.integration
@pytest.mark.parametrize("name,ranks", A2_CASES)
def test_gradient_matches_central_differences_of_phi(s2_case: dict, harness: Path, tmp_path: Path,
                                                     name: str, ranks: int) -> None:
    """A2: <g, dm> of the raw exact gradient (variable projection) equals the
    fourth-order central difference of the offline Phi with eps frozen at the
    unperturbed model (contract 2).  Same direction, steps and tolerance (5e-3)
    as SH-S2, steps 0.4 and 0.2 (test correction 2026-11-23, contract 7a: under
    variable projection the binary32 noise of Phi dominates below 0.2); the
    error curve over 0.8/0.4/0.2/0.1 is printed as evidence."""
    root = s2_case["root"]
    ops = _ops(name)
    workflow = _workflow(s2_case, f"a2_{name}", [_row(pro=0.0, **ops)])
    tag = f"a2_{name}_{ranks}"
    _fwi(s2_case, tag, "model/start", workflow, ranks)
    grad = {s: _read_model(root / f"model/{tag}_stage_1_it_1.grad_{s}") for s in ("vs", "rho", "qs")}
    direction = _direction()
    d_ad = math.fsum(g * d for s in ("vs", "rho", "qs") for g, d in zip(grad[s], direction[s]))
    d_obs, w = _observed_and_weights(s2_case, harness, tmp_path, ops, tag)
    g0 = _impulse_traces(s2_case, harness, tmp_path, "model/start", ops, f"{tag}_0")
    _, eps0, _, _, _ = dense_estimate(g0, d_obs, w, NW_DEFAULT, eps_rel=_f32(EPS_STF))
    curve = _fd_curve(s2_case, harness, tmp_path, tag, ops, d_ad,
                      lambda g: sum(dense_estimate(g, d_obs, w, NW_DEFAULT, eps=eps0)[2:4]))
    print(f"S3-FD {name} ranks={ranks} D_ad={d_ad:.6e} curve={curve}")
    assert max(curve[0.4], curve[0.2]) <= 5.0e-3, curve


FD_STEPS = (0.8, 0.4, 0.2, 0.1)


def _fd_curve(case: dict, harness: Path, tmp: Path, tag: str, ops: dict, d_ad: float, phi_of,
              direction: dict | None = None, values: dict | None = None) -> dict:
    """Relative error of the fourth-order central difference of phi_of(g(m0 +
    x dm)) against d_ad for every step in FD_STEPS; the differences are
    stored in `values` (if given) for further comparisons."""
    root = case["root"]
    base = {s: _read_model(root / f"model/start.{s}") for s in ("vs", "rho", "qs")}
    direction = _direction() if direction is None else direction
    cache = {}

    def phi(x: float) -> float:
        key = int(round(x * 1000))
        if key not in cache:
            name = f"p3_{tag}_{key + 5000}"
            for s in ("vs", "rho", "qs"):
                _write_model(root / f"model/{name}.{s}",
                             [b + x * dd for b, dd in zip(base[s], direction[s])])
            cache[key] = phi_of(_impulse_traces(case, harness, tmp, f"model/{name}", ops, name))
        return cache[key]

    out = {}
    for eps in FD_STEPS:
        d_fd = (8.0 * (phi(eps) - phi(-eps)) - (phi(2 * eps) - phi(-2 * eps))) / (12.0 * eps)
        out[eps] = abs(d_fd - d_ad) / abs(d_ad)
        if values is not None:
            values[eps] = d_fd
    return out


@pytest.mark.integration
@pytest.mark.parametrize("name", ["base", "lowpass"])
def test_gradient_check_rejects_wrong_gradients(s2_case: dict, harness: Path, tmp_path: Path,
                                                name: str) -> None:
    """Falsification control of A2 (external audit 2026-11-23), fixed before
    its first run: with the A2 steps 0.4/0.2 and tolerance 5e-3,
      W1  the raw gradient of the known-source path (INV_STF = 0, Ricker) and
      W2  the correct gradient with one class block scaled by 1.05
    must fail the A2 criterion; W2 is required to fail for every class block
    that carries at least 20 % of <g, dm> (predicted error >= 1e-2) and is
    reported for the others.  The binary32 rounding floor of Phi is measured
    by a line fit over steps +-1e-3 .. +-5e-3 and reported as the predicted
    noise term of the fourth-order difference, sigma * sqrt(130) / (12 h)."""
    root = s2_case["root"]
    ops = _ops(name)
    tag = f"fz_{name}"
    wf_stf = _workflow(s2_case, f"{tag}_stf", [_row(pro=0.0, **ops)])
    wf_known = _workflow(s2_case, f"{tag}_known", [_row(pro=0.0, inv_stf=0, **ops)])
    _fwi(s2_case, f"{tag}_stf", "model/start", wf_stf)
    _fwi(s2_case, f"{tag}_known", "model/start", wf_known)
    direction = _direction()
    classes = ("vs", "rho", "qs")
    grads = {k: {s: _read_model(root / f"model/{tag}_{k}_stage_1_it_1.grad_{s}") for s in classes}
             for k in ("stf", "known")}
    part = {s: math.fsum(g * d for g, d in zip(grads["stf"][s], direction[s])) for s in classes}
    d_ad = sum(part.values())
    d_known = math.fsum(g * d for s in classes for g, d in zip(grads["known"][s], direction[s]))
    d_obs, w = _observed_and_weights(s2_case, harness, tmp_path, ops, tag)
    g0 = _impulse_traces(s2_case, harness, tmp_path, "model/start", ops, f"{tag}_0")
    _, eps0, _, _, _ = dense_estimate(g0, d_obs, w, NW_DEFAULT, eps_rel=_f32(EPS_STF))
    phi_of = lambda g: sum(dense_estimate(g, d_obs, w, NW_DEFAULT, eps=eps0)[2:4])
    base = {s: _read_model(root / f"model/start.{s}") for s in classes}
    phis = {}
    for x in (0.4, 0.2, -0.2, -0.4, 0.8, -0.8):
        key = f"{tag}_{int(round(x * 1000)) + 5000}"
        for s in classes:
            _write_model(root / f"model/{key}.{s}", [b + x * dd for b, dd in zip(base[s], direction[s])])
        phis[x] = phi_of(_impulse_traces(s2_case, harness, tmp_path, f"model/{key}", ops, key))

    def fd(h):
        return (8.0 * (phis[h] - phis[-h]) - (phis[2 * h] - phis[-2 * h])) / (12.0 * h)

    def fails(d_test):
        return max(abs(fd(h) - d_test) / abs(d_test) for h in (0.4, 0.2)) > 5.0e-3

    correct = max(abs(fd(h) - d_ad) / abs(d_ad) for h in (0.4, 0.2))
    assert correct <= 5.0e-3
    w1 = max(abs(fd(h) - d_known) / abs(d_known) for h in (0.4, 0.2))
    print(f"S3-FALSIFY {name} correct={correct:.2e} W1 known-source d={d_known:.6e} "
          f"vs d_ad={d_ad:.6e}: max rel={w1:.2e}")
    assert fails(d_known)
    for s in classes:
        share = abs(part[s]) / abs(d_ad)
        d_w2 = d_ad + 0.05 * part[s]
        w2 = max(abs(fd(h) - d_w2) / abs(d_w2) for h in (0.4, 0.2))
        print(f"S3-FALSIFY {name} W2 {s} x1.05 share={share:.3f} max rel={w2:.2e} "
              f"fails={fails(d_w2)}")
        if share >= 0.2:
            assert fails(d_w2)
    # rounding floor of Phi (reported)
    xs = [k * 1.0e-3 for k in (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)]
    vals = []
    for x in xs:
        key = f"{tag}_n{int(round(x * 1e4)) + 500}"
        for s in classes:
            _write_model(root / f"model/{key}.{s}", [b + x * dd for b, dd in zip(base[s], direction[s])])
        vals.append(phi_of(_impulse_traces(s2_case, harness, tmp_path, f"model/{key}", ops, key)))
    coef = np.polyfit(xs, vals, 1)
    sigma = float(np.std(np.array(vals) - np.polyval(coef, xs), ddof=2))
    floor = {h: sigma * math.sqrt(130.0) / (12.0 * h) / abs(d_ad) for h in (0.4, 0.2, 0.1)}
    print(f"S3-NOISE {name} Phi0={phi_of(g0):.9e} sigma={sigma:.2e} (rel {sigma / phi_of(g0):.2e}) "
          f"predicted rel noise term: h0.4 {floor[0.4]:.2e} h0.2 {floor[0.2]:.2e} h0.1 {floor[0.1]:.2e}")


def _program_ricker() -> np.ndarray:
    """QUELLART 1 of src/wavelet.c (fc 20 Hz, tshift 0, amplitude 1), binary32."""
    fc = np.float32(20.0)
    ts = np.float32(1.0) / fc
    out = []
    for nt in range(1, NS + 1):
        t = np.float32(nt) * np.float32(DT)
        tau = np.float32(math.pi) * (t - np.float32(1.5) * ts) / (np.float32(1.5) * ts)
        out.append(float(np.float32((np.float32(1.0) - np.float32(4.0) * tau * tau)
                                    * np.exp(np.float32(-2.0) * tau * tau))))
    return np.array(out)


@pytest.mark.integration
@pytest.mark.parametrize("name", ["base", "lowpass"])
def test_true_model_recovers_the_true_source(s2_case: dict, harness: Path, tmp_path: Path,
                                             name: str) -> None:
    """A3: at the true model with same-solver data (QUELLART 1 Ricker), B v
    equals B w_true within relative L2 error 1e-3 (threshold frozen by the
    Owner, F9); the support 0.17 s contains w_true (|w| < 1e-11 beyond)."""
    ops = _ops(name)
    nw = int(round(0.17 / DT))
    workflow = _workflow(s2_case, f"a3_{name}", [_row(**ops)])
    tag = f"a3_{name}"
    _fwi(s2_case, tag, "model/true", workflow, 1, stf_length=0.17)
    v = _read_stf(s2_case, tag, 1, 1, nw)
    w_true = _program_ricker()
    assert np.max(np.abs(w_true[nw:])) < 1e-11
    v_full = np.concatenate([v, np.zeros(NS - nw)])
    if ops.get("time_filt", 0) == 1:
        _, (bv, bw) = _filter(harness, tmp_path, [list(v_full), list(w_true)], order=ops["order"],
                              fc_high=ops["fc_high"], fc_low=0.0, dt=DT, tag=f"a3_{name}")
        bv, bw = np.array(bv[:-1]), np.array(bw[:-1])     # legacy last sample unfiltered
    else:
        bv, bw = v_full, w_true
    rel = np.linalg.norm(bv - bw) / np.linalg.norm(bw)
    print(f"S3-A3 {name} rel_L2(Bv, Bw_true)={rel:.3e}")
    assert rel <= 1.0e-3


@pytest.mark.integration
def test_decomposition_gives_identical_phi_and_signal(s2_case: dict) -> None:
    """A5: the estimate is computed on MYID 0 from the gathered global traces;
    Phi and v agree for 1 and 2 ranks within 1e-6 (relative)."""
    workflow = _workflow(s2_case, "a5", [_row(**_ops("combined"))])
    out = {}
    for ranks in (1, 2):
        rows, _ = _fwi(s2_case, f"a5_{ranks}", "model/start", workflow, ranks)
        out[ranks] = (float(rows[0][4]), _read_stf(s2_case, f"a5_{ranks}", 1, 1, NW_DEFAULT))
    rel_j = abs(out[1][0] - out[2][0]) / out[1][0]
    rel_v = np.linalg.norm(out[1][1] - out[2][1]) / np.linalg.norm(out[1][1])
    print(f"S3-A5 rel_J={rel_j:.2e} rel_v={rel_v:.2e} bitwise_v={np.array_equal(out[1][1], out[2][1])}")
    assert rel_j <= 1e-6 and rel_v <= 1e-6


@pytest.mark.integration
def test_multi_stage_run_with_estimated_source(s2_case: dict) -> None:
    """A6: three filter stages with window, INV_STF = 1 and L-BFGS: accepted
    descending steps, J_base(k+1) = J_selected(k) within a stage (objective and
    objective/gradient paths use the same estimator), one .stf file per
    gradient evaluation, and a reload of stage 2 reproduces its first J."""
    root = s2_case["root"]
    rows = [_row(pro=0.0, time_filt=1, fc_high=fc, order=4, timewin=1, gamma=2000.0, twm=0.01,
                 twp=0.02) for fc in (8.0, 12.0, 18.0)]
    workflow = _workflow(s2_case, "a6", rows)
    misfit, run = _fwi(s2_case, "a6", "model/start", workflow, 1, itermax=2, lbfgs=1)
    assert len(misfit) == 6 and [int(r[-1]) for r in misfit] == [1, 1, 2, 2, 3, 3]
    for r in misfit:
        assert float(r[0]) > 0.0 and float(r[6]) < float(r[4])
    for current, following in ((misfit[0], misfit[1]), (misfit[2], misfit[3]), (misfit[4], misfit[5])):
        assert following[4] == current[6], (current, following)
    for stage in (1, 2, 3):
        assert f"Exact SH STF: stage {stage} INV_STF 1" in run.stdout
        assert f"Exact SH STF: stage {stage} shot 1 eps" in run.stdout
        for it in (1, 2):
            v = _read_stf(s2_case, "a6", stage, it, NW_DEFAULT)
            assert np.all(np.isfinite(v)) and np.any(v != 0.0)
    reload_wf = _workflow(s2_case, "a6_reload", [rows[1]])
    misfit2, _ = _fwi(s2_case, "a6_reload", "model/a6_stage_1_it_2", reload_wf, 1, lbfgs=1)
    assert abs(float(misfit2[0][4]) - float(misfit[2][4])) <= 1.0e-6 * abs(float(misfit[2][4]))


@pytest.mark.integration
def test_fixed_per_stage_mode_keeps_the_signal(s2_case: dict) -> None:
    """INV_STF = 3 (O1): the signal of the stage's first evaluation is used for
    every later evaluation of the stage; steps are accepted."""
    workflow = _workflow(s2_case, "o1", [_row(pro=0.0, inv_stf=3, **_ops("lowpass"))])
    misfit, run = _fwi(s2_case, "o1", "model/start", workflow, 1, itermax=2, lbfgs=1)
    assert "Exact SH STF: stage 1 INV_STF 3" in run.stdout
    assert len(misfit) == 2 and misfit[1][4] == misfit[0][6]
    v1 = _read_stf(s2_case, "o1", 1, 1, NW_DEFAULT)
    v2 = _read_stf(s2_case, "o1", 1, 2, NW_DEFAULT)
    assert np.array_equal(v1, v2)
    workflow2 = _workflow(s2_case, "o2", [_row(pro=0.0, inv_stf=1, **_ops("lowpass"))])
    _fwi(s2_case, "o2", "model/start", workflow2, 1, itermax=2, lbfgs=1)
    assert np.array_equal(_read_stf(s2_case, "o2", 1, 1, NW_DEFAULT), v1)
    assert not np.array_equal(_read_stf(s2_case, "o2", 1, 2, NW_DEFAULT), v1)


@pytest.mark.integration
def test_known_source_path_is_unchanged(s2_case: dict) -> None:
    """INV_STF = 0: no SH-S3 line, no .stf output, J as the S-2 suite's path."""
    workflow = _workflow(s2_case, "off", [_row(inv_stf=0)])
    _, run = _fwi(s2_case, "off", "model/start", workflow)
    assert "Exact SH STF" not in run.stdout
    assert not list((s2_case["root"] / "model").glob("off_stage_*.stf"))


STF_ABORTS = {
    "inv_stf_2": dict(inv_stf=2),
    "inv_stf_4": dict(inv_stf=4),
    "eps_zero": dict(eps_stf=0.0),
    "eps_negative": dict(eps_stf=-1.0e-3),
    "offsetc_stf": dict(offsetc_stf=25.0),               # INV_STF = 1 needs the weights of Phi
    "offsetc_stf_negative": dict(inv_stf=3, offsetc_stf=-1.0),
}


@pytest.mark.integration
@pytest.mark.parametrize("name,ranks", [(n, 1) for n in STF_ABORTS] + [("inv_stf_2", 2),
                                                                      ("short_support", 1)])
def test_stf_configuration_aborts_before_the_first_iteration(s2_case: dict, name: str,
                                                             ranks: int) -> None:
    root = s2_case["root"]
    workflow = _workflow(s2_case, f"abort_{name}", [_row(**STF_ABORTS.get(name, {}))])
    parameter = root / f"abort3_{name}_{ranks}.inp"
    _parameter(parameter, mode=1, model="model/start", output=f"model/abort3_{name}", nprocx=ranks,
               stf_length=1.0e-5 if name == "short_support" else None)
    run = _run(s2_case, parameter, workflow, ranks)
    assert run.returncode != 0 or "MPI_ABORT was invoked" in run.stdout, run.stdout
    assert "TDFWI ITERATION" not in run.stdout, run.stdout
    assert not (root / "log/misfit").exists()
    for s, raw in s2_case["start"].items():
        assert (root / f"model/start.{s}").read_bytes() == raw
    assert not list((root / "model").glob(f"abort3_{name}*"))


# ------------------------------------------- INV_STF = 3 with OFFSETC_STF

OFFSETC_STF = 45.0       # keeps the receivers at 20 m and 40 m offset (4 of 6)


def _estimation_weights(w, ntr: int = NREC):
    offsets = [abs(x - 160.0) for x in (100, 120, 140, 180, 200, 220)]
    full = np.ones((ntr, NS)) if w is None else np.array(w, dtype=np.float64)
    keep = np.array([1.0 if _f32(o) < OFFSETC_STF else 0.0 for o in offsets])
    assert keep.sum() == 4
    return full * keep[:, None]


@pytest.mark.integration
# not "combined": its far-offset mute (OFFSETC 45) already removes the traces
# that OFFSETC_STF excludes, so the restriction would not change the estimate
@pytest.mark.parametrize("name,ranks", [("base", 1), ("lowpass", 1), ("window1", 2)])
def test_fixed_signal_with_offset_restricted_estimation(s2_case: dict, harness: Path, tmp_path: Path,
                                                        name: str, ranks: int) -> None:
    """A1/A2 for INV_STF = 3 with OFFSETC_STF (Owner decision 2026-11-23): the
    signal is estimated from the traces with offset < OFFSETC_STF only; Phi
    uses every trace with the fixed signal.  A1: J against the offline dense
    oracle (tolerances as A1).  A2: gradient of the fixed-signal objective
    against central differences (steps 0.4/0.2, tolerance 5e-3; curve
    recorded)."""
    ops = _ops(name)
    workflow = _workflow(s2_case, f"oc_{name}",
                         [_row(pro=0.0, inv_stf=3, offsetc_stf=OFFSETC_STF, **ops)])
    tag = f"oc_{name}_{ranks}"
    rows, run = _fwi(s2_case, tag, "model/start", workflow, ranks)
    assert "Exact SH STF: stage 1 shot 1 eps" in run.stdout and "traces 4 of 6" in run.stdout
    j_log = float(rows[0][4])
    g0 = _impulse_traces(s2_case, harness, tmp_path, "model/start", ops, f"{tag}_0")
    d_obs, w = _observed_and_weights(s2_case, harness, tmp_path, ops, tag)
    w_est = _estimation_weights(w)
    v0, eps0, dterm, eterm, _ = dense_estimate(g0, d_obs, w, NW_DEFAULT, eps_rel=_f32(EPS_STF),
                                               w_est=w_est)
    rel = abs(j_log - (dterm + eterm)) / (dterm + eterm)
    v_out = _read_stf(s2_case, tag, 1, 1, NW_DEFAULT)
    rel_v = np.linalg.norm(v_out - v0) / np.linalg.norm(v0)
    _, _, d_all, _, _ = dense_estimate(g0, d_obs, w, NW_DEFAULT, eps_rel=_f32(EPS_STF))
    assert abs(d_all - dterm) > 1e-6 * d_all        # the restriction changes the estimate
    root = s2_case["root"]
    grad = {s: _read_model(root / f"model/{tag}_stage_1_it_1.grad_{s}") for s in ("vs", "rho", "qs")}
    d_ad = math.fsum(gv * dv for s in ("vs", "rho", "qs") for gv, dv in zip(grad[s], _direction()[s]))
    curve = _fd_curve(s2_case, harness, tmp_path, tag, ops, d_ad,
                      lambda g: sum(dense_estimate(g, d_obs, w, NW_DEFAULT, eps=eps0, v_fixed=v0)[2:4]))
    print(f"S3-OC {name} ranks={ranks} J_log={j_log:.7e} Phi_ref={dterm + eterm:.9e} rel={rel:.2e} "
          f"rel_v={rel_v:.2e} D_ad={d_ad:.6e} curve={curve}")
    assert rel <= (1.0e-4 if ops.get("time_filt", 0) == 1 else 2.0e-6)
    assert rel_v <= 1e-4
    assert max(curve[0.4], curve[0.2]) <= 5.0e-3, curve


@pytest.mark.integration
def test_fixed_signal_with_offset_restriction_multi_stage(s2_case: dict) -> None:
    """A6 for INV_STF = 3 with OFFSETC_STF: two stages with L-BFGS, accepted
    steps, path equality, signal fixed within a stage and re-estimated in the
    next one, reload of stage 2 reproduces its first J."""
    rows = [_row(pro=0.0, inv_stf=3, offsetc_stf=OFFSETC_STF, time_filt=1, fc_high=fc, order=4)
            for fc in (8.0, 12.0)]
    workflow = _workflow(s2_case, "oc_stages", rows)
    misfit, run = _fwi(s2_case, "oc_stages", "model/start", workflow, 1, itermax=2, lbfgs=1)
    assert len(misfit) == 4 and [int(r[-1]) for r in misfit] == [1, 1, 2, 2]
    for r in misfit:
        assert float(r[0]) > 0.0 and float(r[6]) < float(r[4])
    assert misfit[1][4] == misfit[0][6] and misfit[3][4] == misfit[2][6]
    assert run.stdout.count("traces 4 of 6") == 2
    s1 = [_read_stf(s2_case, "oc_stages", 1, it, NW_DEFAULT) for it in (1, 2)]
    s2 = [_read_stf(s2_case, "oc_stages", 2, it, NW_DEFAULT) for it in (1, 2)]
    assert np.array_equal(s1[0], s1[1]) and np.array_equal(s2[0], s2[1])
    assert not np.array_equal(s1[0], s2[0])
    reload_wf = _workflow(s2_case, "oc_reload", [rows[1]])
    misfit2, _ = _fwi(s2_case, "oc_reload", "model/oc_stages_stage_1_it_2", reload_wf, 1, lbfgs=1)
    assert abs(float(misfit2[0][4]) - float(misfit[2][4])) <= 1.0e-6 * abs(float(misfit[2][4]))


# ------------------------------------- review 2026-11-23 (V-1 to V-4: C2, C3, C5, C7)
# Taken over from the independent review's counter-checks with unchanged
# tolerances (docs/sh_s3_stf_contract.md, section 7a).

def _q_only_direction() -> dict:
    phi = [math.exp(-((x - 16) ** 2 + (y - 20) ** 2) / (2 * 2.5 ** 2)) for x in range(32) for y in range(32)]
    return {"vs": [0.0] * CELLS, "rho": [0.0] * CELLS, "qs": [20.0 * p for p in phi]}


@pytest.mark.integration
@pytest.mark.parametrize("name", ["lowpass", "base"])
def test_q_only_direction_gradient_and_falsification(s2_case: dict, harness: Path, tmp_path: Path,
                                                     name: str) -> None:
    """V-4 / C2: A2 with a direction only in Q (closes the reported limit of
    the Q block); the Q block x 1.05 must fail at the A2 tolerance."""
    root = s2_case["root"]
    ops = _ops(name)
    qdir = _q_only_direction()
    tag = f"v4q_{name}"
    _fwi(s2_case, tag, "model/start", _workflow(s2_case, tag, [_row(pro=0.0, **ops)]), 1)
    grad = _read_model(root / f"model/{tag}_stage_1_it_1.grad_qs")
    d_ad = math.fsum(g * d for g, d in zip(grad, qdir["qs"]))
    d_obs, w = _observed_and_weights(s2_case, harness, tmp_path, ops, tag)
    g0 = _impulse_traces(s2_case, harness, tmp_path, "model/start", ops, f"{tag}_0")
    _, eps0, _, _, _ = dense_estimate(g0, d_obs, w, NW_DEFAULT, eps_rel=_f32(EPS_STF))
    values = {}
    curve = _fd_curve(s2_case, harness, tmp_path, tag, ops, d_ad,
                      lambda g: sum(dense_estimate(g, d_obs, w, NW_DEFAULT, eps=eps0)[2:4]),
                      direction=qdir, values=values)
    w2 = {h: abs(values[h] - 1.05 * d_ad) / abs(1.05 * d_ad) for h in values}
    print(f"S3-V4Q {name} D_ad={d_ad:.6e} curve={curve} W2(Q x1.05)={w2}")
    assert max(curve[0.4], curve[0.2]) <= 5.0e-3
    assert max(w2[0.4], w2[0.2]) > 5.0e-3


@pytest.mark.integration
@pytest.mark.parametrize("mode", [1, 3])
def test_eps_and_signal_frozen_away_from_the_stage_start(s2_case: dict, harness: Path, tmp_path: Path,
                                                         mode: int) -> None:
    """V-2 / C3: J of iteration 2 (a model != stage start) equals the offline
    Phi with eps frozen at the stage start (INV_STF = 1) or with the fixed
    stage-start signal (INV_STF = 3); the alternative (eps re-frozen or signal
    re-estimated) differs by more than ten times the observed deviation."""
    ops = _ops("lowpass")
    tag = f"v2_{mode}"
    misfit, run = _fwi(s2_case, tag, "model/start", _workflow(s2_case, tag, [_row(pro=0.0, inv_stf=mode, **ops)]),
                       1, itermax=2)
    assert run.stdout.count("Exact SH STF: stage 1 shot 1 eps") == 1
    j2 = float(misfit[1][4])
    d_obs, w = _observed_and_weights(s2_case, harness, tmp_path, ops, tag)
    g0 = _impulse_traces(s2_case, harness, tmp_path, "model/start", ops, f"{tag}_0")
    v0, eps0, _, _, _ = dense_estimate(g0, d_obs, w, NW_DEFAULT, eps_rel=_f32(EPS_STF))
    g1 = _impulse_traces(s2_case, harness, tmp_path, f"model/{tag}_stage_1_it_1", ops, f"{tag}_1")
    if mode == 1:
        frozen = sum(dense_estimate(g1, d_obs, w, NW_DEFAULT, eps=eps0)[2:4])
    else:
        frozen = sum(dense_estimate(g1, d_obs, w, NW_DEFAULT, eps=eps0, v_fixed=v0)[2:4])
    alternative = sum(dense_estimate(g1, d_obs, w, NW_DEFAULT, eps_rel=_f32(EPS_STF))[2:4])
    v2 = _read_stf(s2_case, tag, 1, 2, NW_DEFAULT)
    print(f"S3-V2 mode={mode} J_it2={j2:.7e} Phi_frozen={frozen:.9e} rel={abs(j2 - frozen) / frozen:.2e} "
          f"alternative rel={abs(j2 - alternative) / alternative:.2e} "
          f"v_it2 vs v0 rel={np.linalg.norm(v2 - v0) / np.linalg.norm(v0):.2e}")
    assert abs(j2 - frozen) <= 1e-4 * frozen
    assert abs(frozen - alternative) > 10 * abs(j2 - frozen)


TWO_SOURCES = ((160.0, 160.0), (150.0, 120.0))


@pytest.fixture(scope="module")
def two_shot_case(tmp_path_factory: pytest.TempPathFactory, denise_binary: Path, mpiexec: str) -> dict:
    root = tmp_path_factory.mktemp("sh_s3_two_shots")
    for name in ("model", "source", "receiver", "su", "log", "picks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    true_vs = [1900.0 if 18 <= x <= 22 and 18 <= y <= 22 else 2000.0 for x in range(32) for y in range(32)]
    for prefix, vs in (("true", true_vs), ("start", [2000.0] * CELLS)):
        _write_model(root / f"model/{prefix}.vs", vs)
        _write_model(root / f"model/{prefix}.rho", [2000.0] * CELLS)
        _write_model(root / f"model/{prefix}.qs", [80.0] * CELLS)
    (root / "source/one.dat").write_text(
        "2\n" + "".join(f"{x:g} 0 {y:g} 0 20 1 0 1\n" for x, y in TWO_SOURCES), encoding="ascii")
    (root / "receiver/line.dat").write_text(
        "100 160\n120 160\n140 160\n180 160\n200 160\n220 160\n", encoding="ascii")
    for s in (1, 2):
        (root / f"picks/pick_{s}.dat").write_text("".join(f"{p:.6f}\n" for p in PICKS), encoding="ascii")
    case = {"root": root, "binary": denise_binary, "mpiexec": mpiexec}
    wf = root / "wf_plain.inp"
    wf.write_text(WORKFLOW_HEADER + "\n" + _row(inv_stf=0) + "\n", encoding="ascii")
    _s2_parameter(root / "forward.inp", mode=0, model="model/true")
    run = _run(case, root / "forward.inp", wf)
    assert run.returncode == 0, run.stdout
    assert (root / "su/observed_y.su.shot2").exists()
    return case


def _fwi2(case: dict, tag: str, model: str, workflow: Path, ranks: int = 1, itermax: int = 1, lbfgs: int = 0):
    parameter = case["root"] / f"{tag}.inp"
    _parameter(parameter, mode=1, model=model, output=f"model/{tag}", nprocx=ranks, itermax=itermax, lbfgs=lbfgs)
    text = parameter.read_text(encoding="ascii")
    old, new = "(TESTSHOTS) = 1, 1, 1\n", "(TESTSHOTS) = 1, 2, 1\n"
    assert text.count(old) == 1
    parameter.write_text(text.replace(old, new), encoding="ascii")
    run = _run(case, parameter, workflow, ranks)
    assert run.returncode == 0 and "MPI_ABORT" not in run.stdout, run.stdout[-2000:]
    return _misfit_rows(case), run


def _two_shot_impulse(case: dict, harness: Path, tmp: Path, model: str, ops: dict, tag: str):
    root = case["root"]
    impulse = [1.0] + [0.0] * (NS - 1)
    if ops.get("time_filt", 0) == 1:
        _, (impulse,) = _filter(harness, tmp, [impulse], order=ops["order"], fc_high=ops["fc_high"],
                                fc_low=ops.get("fc_low", 0.0), dt=DT, tag=f"imp_{tag}")
    for s in (1, 2):
        _signal_file(root / f"source/imp_{tag}_shot_{s}.dat", impulse)
    parameter = root / f"imp_{tag}.inp"
    _s2_parameter(parameter, mode=0, model=model, quellart=3, seis_vy=f"su/imp_{tag}_y.su")
    text = parameter.read_text(encoding="ascii").replace(
        "(SIGNAL_FILE) = source/signal.dat\n", f"(SIGNAL_FILE) = source/imp_{tag}\n")
    parameter.write_text(text, encoding="ascii")
    run = _run(case, parameter, root / "wf_plain.inp")
    assert run.returncode == 0, run.stdout
    return [np.array(read_su_float_samples(root / f"su/imp_{tag}_y.su.shot{s}", NREC, NS),
                     dtype=np.float64).reshape(NREC, NS) for s in (1, 2)]


def _two_shot_observed(case: dict, harness: Path, tmp: Path, ops: dict, tag: str):
    out = []
    for s in (1, 2):
        raw = read_su_float_samples(case["root"] / f"su/observed_y.su.shot{s}", NREC, NS)
        obs = [raw[i * NS:(i + 1) * NS] for i in range(NREC)]
        if ops.get("time_filt", 0) == 1:
            _, obs = _filter(harness, tmp, obs, order=ops["order"], fc_high=ops["fc_high"],
                             fc_low=ops.get("fc_low", 0.0), dt=DT, tag=f"obs_{tag}_{s}")
        out.append(np.array(obs, dtype=np.float64))
    w = None                      # base and lowpass only: weights = last-sample mask
    if ops.get("time_filt", 0) == 1:
        w = np.ones((NREC, NS))
        w[:, -1] = 0.0
    return out, w


def _read_stf2(case: dict, tag: str, stage: int, it: int, nw: int) -> np.ndarray:
    raw = (case["root"] / f"model/{tag}_stage_{stage}_it_{it}.stf").read_bytes()
    assert len(raw) == 2 * 4 * nw
    return np.frombuffer(raw, "=f4").astype(np.float64).reshape(2, nw)


@pytest.mark.integration
@pytest.mark.parametrize("name,ranks", [("lowpass", 1), ("lowpass", 2), ("base", 1)])
def test_two_shots_objective_and_signals(two_shot_case: dict, harness: Path, tmp_path: Path,
                                         name: str, ranks: int) -> None:
    """V-1 / C5: two shots, A1 per-shot sum, one .stf row and one eps per shot."""
    ops = _ops(name)
    tag = f"v1_{name}_{ranks}"
    rows, run = _fwi2(two_shot_case, tag, "model/start", _workflow(two_shot_case, tag, [_row(pro=0.0, **ops)]),
                      ranks)
    j_log = float(rows[0][4])
    gs = _two_shot_impulse(two_shot_case, harness, tmp_path, "model/start", ops, tag)
    ds, w = _two_shot_observed(two_shot_case, harness, tmp_path, ops, tag)
    est = [dense_estimate(g, d, w, NW_DEFAULT, eps_rel=_f32(EPS_STF)) for g, d in zip(gs, ds)]
    phi = sum(e[2] + e[3] for e in est)
    v_out = _read_stf2(two_shot_case, tag, 1, 1, NW_DEFAULT)
    rel_v = [np.linalg.norm(v_out[s] - est[s][0]) / np.linalg.norm(est[s][0]) for s in (0, 1)]
    eps_log = [float(re.search(rf"Exact SH STF: stage 1 shot {s} eps (\S+)", run.stdout).group(1))
               for s in (1, 2)]
    print(f"S3-V1 {name} ranks={ranks} J_log={j_log:.7e} Phi={phi:.9e} rel={abs(j_log - phi) / phi:.2e} "
          f"rel_v={rel_v}")
    tol = 1e-4 if ops.get("time_filt", 0) == 1 else 2e-6
    assert abs(j_log - phi) <= tol * phi
    assert max(rel_v) <= 1e-4
    assert all(abs(a - b[1]) <= 1e-5 * b[1] for a, b in zip(eps_log, est))
    assert not np.allclose(v_out[0], v_out[1])


@pytest.mark.integration
def test_two_shots_gradient(two_shot_case: dict, harness: Path, tmp_path: Path) -> None:
    """V-1 / C5: A2 with two shots (lowpass, steps 0.4/0.2, tolerance 5e-3)."""
    ops = _ops("lowpass")
    root = two_shot_case["root"]
    tag = "v1g"
    _fwi2(two_shot_case, tag, "model/start", _workflow(two_shot_case, tag, [_row(pro=0.0, **ops)]), 1)
    direction = _direction()
    grad = {s: _read_model(root / f"model/{tag}_stage_1_it_1.grad_{s}") for s in ("vs", "rho", "qs")}
    d_ad = math.fsum(g * d for s in ("vs", "rho", "qs") for g, d in zip(grad[s], direction[s]))
    ds, w = _two_shot_observed(two_shot_case, harness, tmp_path, ops, tag)
    g0 = _two_shot_impulse(two_shot_case, harness, tmp_path, "model/start", ops, f"{tag}_0")
    eps0 = [dense_estimate(g, d, w, NW_DEFAULT, eps_rel=_f32(EPS_STF))[1] for g, d in zip(g0, ds)]
    base = {s: _read_model(root / f"model/start.{s}") for s in ("vs", "rho", "qs")}
    cache = {}

    def phi(x: float) -> float:
        key = int(round(x * 1000))
        if key not in cache:
            name = f"v1p_{key + 5000}"
            for s in ("vs", "rho", "qs"):
                _write_model(root / f"model/{name}.{s}", [b + x * dd for b, dd in zip(base[s], direction[s])])
            gs = _two_shot_impulse(two_shot_case, harness, tmp_path, f"model/{name}", ops, name)
            cache[key] = sum(sum(dense_estimate(g, d, w, NW_DEFAULT, eps=e)[2:4]) for g, d, e in zip(gs, ds, eps0))
        return cache[key]

    curve = {}
    for h in FD_STEPS:
        d_fd = (8.0 * (phi(h) - phi(-h)) - (phi(2 * h) - phi(-2 * h))) / (12.0 * h)
        curve[h] = abs(d_fd - d_ad) / abs(d_ad)
    print(f"S3-V1-A2 D_ad={d_ad:.6e} curve={curve}")
    assert max(curve[0.4], curve[0.2]) <= 5.0e-3


@pytest.mark.integration
def test_two_shots_fixed_mode(two_shot_case: dict) -> None:
    """V-1 / C5: INV_STF = 3 keeps both signals over two iterations; the shots differ."""
    ops = _ops("lowpass")
    tag = "v1o1"
    misfit, _ = _fwi2(two_shot_case, tag, "model/start", _workflow(two_shot_case, tag,
                      [_row(pro=0.0, inv_stf=3, **ops)]), 1, itermax=2, lbfgs=1)
    assert misfit[1][4] == misfit[0][6]
    v1 = _read_stf2(two_shot_case, tag, 1, 1, NW_DEFAULT)
    v2 = _read_stf2(two_shot_case, tag, 1, 2, NW_DEFAULT)
    assert np.array_equal(v1, v2) and not np.allclose(v1[0], v1[1])


@pytest.mark.integration
@pytest.mark.parametrize("seq", [(1, 0), (0, 1), (1, 3), (3, 1)])
def test_mixed_stage_sequences(s2_case: dict, seq) -> None:
    """V-3 / C7: stage 2 of a two-stage run equals a single-stage run of the
    stage-2 row from the stage-1 output model (state reset per stage, eps
    frozen at the stage-2 start, known source restored for INV_STF = 0)."""
    a, b = seq
    tag = f"v3_{a}{b}"
    row1 = _row(pro=0.0, inv_stf=a, time_filt=1, fc_high=10.0, order=4)
    row2 = _row(pro=0.0, inv_stf=b, time_filt=1, fc_high=15.0, order=4)
    misfit, _ = _fwi(s2_case, tag, "model/start", _workflow(s2_case, tag, [row1, row2]), 1)
    assert [int(r[-1]) for r in misfit] == [1, 2]
    ref, _ = _fwi(s2_case, f"{tag}_ref", f"model/{tag}_stage_1_it_1", _workflow(s2_case, f"{tag}_ref", [row2]), 1)
    stf2 = list((s2_case["root"] / "model").glob(f"{tag}_stage_2_it_*.stf"))
    assert misfit[1][4] == ref[0][4] and misfit[1][6] == ref[0][6]
    assert (len(stf2) == 1) == (b != 0)


@pytest.mark.integration
def test_runtime_check_and_solver_log(s2_case: dict) -> None:
    """V-5 / V-7.5: every gradient evaluation logs D, E, the PCG iterations and
    the true relative residual (<= 1e-11), and the runtime check of s = g * v
    (true forward with B v against the estimator's D) agrees within 1e-5."""
    tag = "v5"
    _, run = _fwi(s2_case, tag, "model/start", _workflow(s2_case, tag, [_row(pro=0.0, **_ops("combined"))]), 1)
    line = re.search(r"Exact SH STF: stage 1 it 1 shot 1 D (\S+) E (\S+) eps \S+ max\|v\| \S+ \|v\| \S+ "
                     r"pcg_it (\d+) relres (\S+)", run.stdout)
    assert line and int(line.group(3)) > 0 and float(line.group(4)) <= 1e-11
    check = re.search(r"Exact SH STF: shot 1 D_forward (\S+) D_estimator (\S+) rel (\S+)", run.stdout)
    assert check and float(check.group(3)) <= 1e-5, check and check.group(0)


# ------------------------------- review 2026-11-23, V-3 rest (C4, C6)

@pytest.mark.integration
def test_support_clipped_to_nt(s2_case: dict, harness: Path, tmp_path: Path) -> None:
    """V-3 / C4: STF_LENGTH > TIME clips the support to N_w = NT; A1 and A2
    (lowpass, steps 0.4/0.2, tolerance 5e-3)."""
    root = s2_case["root"]
    ops = _ops("lowpass")
    tag = "v3c4"
    rows, run = _fwi(s2_case, tag, "model/start", _workflow(s2_case, tag, [_row(pro=0.0, **ops)]), 1,
                     stf_length=0.3)
    assert f"N_w {NS}" in run.stdout
    j_log = float(rows[0][4])
    d_obs, w = _observed_and_weights(s2_case, harness, tmp_path, ops, tag)
    g0 = _impulse_traces(s2_case, harness, tmp_path, "model/start", ops, f"{tag}_0")
    v0, eps0, dt0, et0, _ = dense_estimate(g0, d_obs, w, NS, eps_rel=_f32(EPS_STF))
    v_out = _read_stf(s2_case, tag, 1, 1, NS)
    rel = abs(j_log - (dt0 + et0)) / (dt0 + et0)
    grad = {s: _read_model(root / f"model/{tag}_stage_1_it_1.grad_{s}") for s in ("vs", "rho", "qs")}
    d_ad = math.fsum(g * d for s in ("vs", "rho", "qs") for g, d in zip(grad[s], _direction()[s]))
    curve = _fd_curve(s2_case, harness, tmp_path, tag, ops, d_ad,
                      lambda g: sum(dense_estimate(g, d_obs, w, NS, eps=eps0)[2:4]))
    print(f"S3-V3C4 nw={NS} J_log={j_log:.7e} Phi={dt0 + et0:.9e} rel={rel:.2e} "
          f"rel_v={np.linalg.norm(v_out - v0) / np.linalg.norm(v0):.2e} curve={curve}")
    assert rel <= 1e-4
    assert np.linalg.norm(v_out - v0) / np.linalg.norm(v0) <= 1e-4
    assert max(curve[0.4], curve[0.2]) <= 5.0e-3


REC2 = {1: (100, 120, 140, 180, 200, 220), 2: (100, 120, 140, 180)}


def _readrec2(parameter: Path) -> None:
    text = parameter.read_text(encoding="ascii")
    for old, new in (("(READREC) = 1\n", "(READREC) = 2\n"),
                     ("(TESTSHOTS) = 1, 1, 1\n", "(TESTSHOTS) = 1, 2, 1\n")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    parameter.write_text(text, encoding="ascii")


@pytest.fixture(scope="module")
def readrec2_case(tmp_path_factory: pytest.TempPathFactory, denise_binary: Path, mpiexec: str) -> dict:
    root = tmp_path_factory.mktemp("sh_s3_readrec2")
    for name in ("model", "source", "receiver", "su", "log", "picks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    true_vs = [1900.0 if 18 <= x <= 22 and 18 <= y <= 22 else 2000.0 for x in range(32) for y in range(32)]
    for prefix, vs in (("true", true_vs), ("start", [2000.0] * CELLS)):
        _write_model(root / f"model/{prefix}.vs", vs)
        _write_model(root / f"model/{prefix}.rho", [2000.0] * CELLS)
        _write_model(root / f"model/{prefix}.qs", [80.0] * CELLS)
    (root / "source/one.dat").write_text(
        "2\n" + "".join(f"{x:g} 0 {y:g} 0 20 1 0 1\n" for x, y in TWO_SOURCES), encoding="ascii")
    for s, xs in REC2.items():
        (root / f"receiver/line_shot_{s}.dat").write_text("".join(f"{x} 160\n" for x in xs), encoding="ascii")
    case = {"root": root, "binary": denise_binary, "mpiexec": mpiexec}
    wf = root / "wf_plain.inp"
    wf.write_text(WORKFLOW_HEADER + "\n" + _row(inv_stf=0) + "\n", encoding="ascii")
    _s2_parameter(root / "forward.inp", mode=0, model="model/true")
    _readrec2(root / "forward.inp")
    run = _run(case, root / "forward.inp", wf)
    assert run.returncode == 0, run.stdout[-3000:]
    return case


@pytest.mark.integration
@pytest.mark.parametrize("ranks", [1, 2])
def test_readrec2_two_shots(readrec2_case: dict, harness: Path, tmp_path: Path, ranks: int) -> None:
    """V-3 / C6: READREC = 2 (6 and 4 receivers per shot), lowpass, A1 sum and
    per-shot signals against the offline dense oracle."""
    case, root = readrec2_case, readrec2_case["root"]
    ops = _ops("lowpass")
    tag = f"v3c6_{ranks}"
    parameter = root / f"{tag}.inp"
    _parameter(parameter, mode=1, model="model/start", output=f"model/{tag}", nprocx=ranks)
    _readrec2(parameter)
    run = _run(case, parameter, _workflow(case, tag, [_row(pro=0.0, **ops)]), ranks)
    assert run.returncode == 0 and "MPI_ABORT" not in run.stdout, run.stdout[-3000:]
    assert "traces 6 of 6" in run.stdout and "traces 4 of 4" in run.stdout
    j_log = float(_misfit_rows(case)[0][4])
    impulse = [1.0] + [0.0] * (NS - 1)
    _, (impulse,) = _filter(harness, tmp_path, [impulse], order=ops["order"], fc_high=ops["fc_high"],
                            fc_low=0.0, dt=DT, tag=f"imp_{tag}")
    for s in (1, 2):
        _signal_file(root / f"source/imp_{tag}_shot_{s}.dat", impulse)
    ip = root / f"imp_{tag}.inp"
    _s2_parameter(ip, mode=0, model="model/start", quellart=3, seis_vy=f"su/imp_{tag}_y.su")
    ip.write_text(ip.read_text(encoding="ascii").replace(
        "(SIGNAL_FILE) = source/signal.dat\n", f"(SIGNAL_FILE) = source/imp_{tag}\n"), encoding="ascii")
    _readrec2(ip)
    r = _run(case, ip, root / "wf_plain.inp")
    assert r.returncode == 0, r.stdout[-3000:]
    phi, vs = 0.0, []
    for s in (1, 2):
        nrec = len(REC2[s])
        g = np.array(read_su_float_samples(root / f"su/imp_{tag}_y.su.shot{s}", nrec, NS)).reshape(nrec, NS)
        raw = read_su_float_samples(root / f"su/observed_y.su.shot{s}", nrec, NS)
        _, obs = _filter(harness, tmp_path, [raw[i * NS:(i + 1) * NS] for i in range(nrec)],
                         order=ops["order"], fc_high=ops["fc_high"], fc_low=0.0, dt=DT, tag=f"obs_{tag}_{s}")
        w = np.ones((nrec, NS))
        w[:, -1] = 0.0
        e = dense_estimate(g.astype(np.float64), np.array(obs, dtype=np.float64), w, NW_DEFAULT,
                           eps_rel=_f32(EPS_STF))
        phi += e[2] + e[3]
        vs.append(e[0])
    v_out = np.frombuffer((root / f"model/{tag}_stage_1_it_1.stf").read_bytes(), "=f4").astype(np.float64)
    v_out = v_out.reshape(2, NW_DEFAULT)
    rel_v = [float(np.linalg.norm(v_out[s] - vs[s]) / np.linalg.norm(vs[s])) for s in (0, 1)]
    print(f"S3-V3C6 ranks={ranks} J_log={j_log:.7e} Phi={phi:.9e} rel={abs(j_log - phi) / phi:.2e} rel_v={rel_v}")
    assert abs(j_log - phi) <= 1e-4 * phi and max(rel_v) <= 1e-4
