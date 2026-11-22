"""SH-S2: frequency stages, time window and offset mute in the exact
viscoelastic SH FWI (docs/sh_s2_data_operators_contract.md).

Unit level: the stage filter is the legacy apply_tdfilt (its unfiltered last
sample is a documented legacy property); the data weights against an
independent formula; error codes.
Driver level (32x32 fixture, real exact path): directional derivative of the
raw gradient against central differences of J for every mode; the logged J
against an independent offline reference; the stage filter commutes with the
forward solver on samples 1..NT-1; a multi-stage run with accepted steps, path
equality and reload consistency; configuration aborts.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess

import pytest

from tests.utilities.fwi_gradient import read_su_float_samples
from tests.utilities.sh_fwi_fixture import write_sh_parameter
from tests.utilities.sh_s2_link import S2_LINK_SOURCES

# ----------------------------------------------------------------- unit level

HARNESS_SOURCES = (
    "tests/utilities/sh_s2_data_operators_harness.c",
    "src/SH/visco_sh_exact_data_operators.c",
    "src/apply_tdfilt.c",
    "src/timedomain_filt.c",
    "libcseife/cseife.c", "libcseife/cseife_deriv.c", "libcseife/cseife_gauss.c",
    "libcseife/cseife_rekfl.c", "libcseife/cseife_rfk.c", "libcseife/cseife_tides.c",
)


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory, repository_root: Path) -> Path:
    compiler = shutil.which("mpicc")
    assert compiler
    exe = tmp_path_factory.mktemp("sh_s2_unit") / "sh_s2_harness"
    command = [compiler, "-std=c99", "-O2", "-fcommon",
               "-I", str(repository_root / "include"), "-I", str(repository_root / "libcseife"),
               *(str(repository_root / s) for s in HARNESS_SOURCES), "-o", str(exe), "-lm"]
    done = subprocess.run(command, cwd=repository_root, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert done.returncode == 0, done.stdout
    return exe


def _f32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


def _write_traces(path: Path, traces: list[list[float]]) -> None:
    path.write_text("".join(f"{_f32(v):.9e}\n" for tr in traces for v in tr), encoding="ascii")


def _read_traces(path: Path, ntr: int, ns: int) -> list[list[float]]:
    values = [_f32(float(x)) for x in path.read_text(encoding="ascii").split()]
    assert len(values) == ntr * ns
    return [values[i * ns:(i + 1) * ns] for i in range(ntr)]


def _filter(harness: Path, tmp: Path, traces, *, order, fc_high, fc_low, dt,
            mode="legacy", tag="x"):
    src, dst = tmp / f"{tag}_in.txt", tmp / f"{tag}_out.txt"
    _write_traces(src, traces)
    done = subprocess.run([str(harness), mode, str(src), str(dst), str(len(traces)),
                           str(len(traces[0])), str(order), repr(fc_high), repr(fc_low), repr(dt)],
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert done.returncode == 0, done.stdout
    status = int(re.search(r"status (-?\d+)", done.stdout).group(1))
    if status != 0:
        return status, None
    return 0, _read_traces(dst, len(traces), len(traces[0]))


def _random_traces(seed: int, ntr: int, ns: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[_f32(rng.gauss(0.0, 1.0)) for _ in range(ns)] for _ in range(ntr)]


FILTER_CASES = [  # order, fc_high, fc_low, dt
    (4, 12.0, 0.0, 5.0e-4), (3, 12.0, 0.0, 5.0e-4), (6, 15.0, 6.0, 5.0e-4),
]


@pytest.mark.parametrize("order,fc_high,fc_low,dt", FILTER_CASES)
def test_legacy_stage_filter_leaves_the_last_sample_unfiltered(harness: Path, tmp_path: Path,
                                                               order, fc_high, fc_low, dt) -> None:
    """Owner decision 2026-11-21: the exact path uses the legacy apply_tdfilt
    unchanged (as legacy and the PSV strand).  Documented legacy property: the
    last sample of every trace stays unfiltered; samples 1..ns-1 equal the
    filtered values of a one sample longer trace (causal filter)."""
    ns = 300
    traces = _random_traces(order, 3, ns)
    _, got = _filter(harness, tmp_path, traces, order=order, fc_high=fc_high, fc_low=fc_low, dt=dt)
    _, longer = _filter(harness, tmp_path, [tr + [7.0] for tr in traces], order=order,
                        fc_high=fc_high, fc_low=fc_low, dt=dt, tag="l")
    for g, ll, raw in zip(got, longer, traces):
        assert g[-1] == raw[-1]
        assert g[:-1] == ll[:ns - 1]
        assert g[:-1] != raw[:-1]


def test_exact_path_filters_source_and_observed_with_apply_tdfilt(repository_root: Path) -> None:
    glue = re.sub(r"\s+", "", (repository_root / "src/SH/exact_sh_data_ops.h").read_text())
    assert "apply_tdfilt(signals,nsrc_local,ns,ORDER,FC_END,FC_START);" in glue
    assert "apply_tdfilt(observed_global,nrec_global,ns,ORDER,FC_END,FC_START);" in glue
    assert "visco_sh_exact_filter_traces" not in glue
    for wrapper in ("src/SH/obj_sh_visc_exact.c", "src/SH/grad_obj_sh_visc_exact.c"):
        source = re.sub(r"\s+", "", (repository_root / wrapper).read_text())
        read = source.index("inseis(ishot,request->legacy_fwi_seismogram->sectionread")
        filt = source.index("exact_sh_s2_filter_shot(request->acquisition->signals")
        mapped = source.index("_map_observed(", filt)
        assert read < filt < mapped       # filter on the global observed traces, then map


def _reference_weights(ns, dt, dh, timewin, gamma, twm, twp, mute, offsetc, sx, sy,
                       receivers, picks, mask_last=0):
    """Independent formula of contract 2 (double precision); mask_last
    (TIME_FILT = 1) sets the weight of the last sample to 0."""
    dt, dh, gamma, twm, twp, offsetc = (_f32(v) for v in (dt, dh, gamma, twm, twp, offsetc))
    sx, sy = _f32(sx), _f32(sy)
    out, kept = [], 0
    for g, gx, gy in receivers:
        # binary32 as legacy calc_res: xr = recpos*DH, offset = (float)sqrt(dx*dx+dy*dy)
        dx, dy = _f32(sx - _f32(gx * dh)), _f32(sy - _f32(gy * dh))
        off = _f32(math.sqrt(_f32(_f32(dx * dx) + _f32(dy * dy))))
        if (mute == 1 and off >= offsetc) or (mute == 2 and off <= offsetc):
            out.append([0.0] * ns)
            continue
        kept += 1
        if timewin == 0:
            out.append([1.0] * (ns - 1) + [0.0 if mask_last else 1.0])
            continue
        p = _f32(picks[g - 1]) if timewin == 1 else 0.0
        lo, hi = p - twm, p + twp
        row = []
        for k in range(1, ns + 1):
            t = k * dt
            d = t - hi if t > hi else (t - lo if t < lo else 0.0)
            row.append(1.0 if d == 0.0 else math.exp(-gamma * d * d))
        if mask_last:
            row[-1] = 0.0
        out.append(row)
    return kept, out


WEIGHT_RECEIVERS = [(1, 10, 16), (2, 12, 16), (4, 18, 16), (5, 20, 16), (6, 22, 16)]
WEIGHT_PICKS = [0.081, 0.071, 0.061, 0.061, 0.071, 0.081]
WEIGHT_CASES = {
    "window1": dict(timewin=1, gamma=2000.0, twm=0.01, twp=0.02, mute=0, offsetc=0.0),
    "window1_no_pre": dict(timewin=1, gamma=500.0, twm=0.0, twp=0.0, mute=0, offsetc=0.0),
    "mute_far": dict(timewin=0, gamma=0.0, twm=0.0, twp=0.0, mute=1, offsetc=45.0),
    "mute_near": dict(timewin=0, gamma=0.0, twm=0.0, twp=0.0, mute=2, offsetc=45.0),
    "window1_mute_far": dict(timewin=1, gamma=2000.0, twm=0.01, twp=0.02, mute=1, offsetc=45.0),
}


def _run_weights(harness: Path, tmp: Path, case: dict, *, picks_prefix: str | None,
                 ns=400, dt=5.0e-4, dh=10.0, sx=160.0, sy=160.0, receivers=WEIGHT_RECEIVERS,
                 mask_last=0):
    stdin = "".join(f"{g} {gx} {gy}\n" for g, gx, gy in receivers)
    args = [str(harness), "weights", str(ns), repr(dt), repr(dh), str(case["timewin"]),
            repr(case["gamma"]), repr(case["twm"]), repr(case["twp"]), str(case["mute"]),
            repr(case["offsetc"]), repr(sx), repr(sy), picks_prefix or "-", "6", str(mask_last)]
    done = subprocess.run(args, input=stdin, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    assert done.returncode == 0, done.stdout
    return done.stdout.splitlines()


@pytest.mark.parametrize("mask_last", [0, 1])
@pytest.mark.parametrize("name", sorted(WEIGHT_CASES))
def test_weights_match_the_independent_formula(harness: Path, tmp_path: Path, name: str,
                                               mask_last: int) -> None:
    """mask_last = 1 is TIME_FILT = 1: the last sample (left unfiltered by the
    legacy apply_tdfilt) has weight 0 (Owner decision 2026-11-21)."""
    case = WEIGHT_CASES[name]
    prefix = None
    if case["timewin"] == 1:
        prefix = str(tmp_path / "picks_")
        (tmp_path / "picks_1.dat").write_text("".join(f"{p}\n" for p in WEIGHT_PICKS),
                                              encoding="ascii")
    lines = _run_weights(harness, tmp_path, case, picks_prefix=prefix, mask_last=mask_last)
    kept = int(lines[0].split()[1])
    got = [[float(v) for v in ln.split()] for ln in lines[1:]]
    ref_kept, ref = _reference_weights(400, 5.0e-4, 10.0, case["timewin"], case["gamma"],
                                       case["twm"], case["twp"], case["mute"], case["offsetc"],
                                       160.0, 160.0, WEIGHT_RECEIVERS, WEIGHT_PICKS,
                                       mask_last=mask_last)
    assert kept == ref_kept and len(got) == len(ref)
    if mask_last:
        assert all(row[-1] == 0.0 for row in got)
    for g_row, r_row in zip(got, ref):
        for g, r in zip(g_row, r_row):
            assert abs(g - r) <= 1.0e-12 * max(1.0, abs(r))
            if r in (0.0, 1.0):
                assert g == r                     # mask and flat window exact
    if case["mute"]:
        assert 0 < kept < len(WEIGHT_RECEIVERS)
    if name == "window1":                         # the pre-pick flank takes effect (TWIN-)
        assert min(ref[0][:100]) < 1.0e-3 and max(ref[0]) == 1.0


@pytest.mark.parametrize("mute", [1, 2])
def test_offset_ties_follow_the_legacy_binary32_offset(harness: Path, tmp_path: Path,
                                                       mute: int) -> None:
    """Review N1: DH 0.1, receiver index 50, source at 0 and OFFSETC 5: the
    binary32 offset of legacy calc_res decides the tie, not a double offset."""
    case = dict(timewin=0, gamma=0.0, twm=0.0, twp=0.0, mute=mute, offsetc=5.0)
    receivers = [(1, 50, 0), (2, 49, 0), (3, 51, 0)]
    lines = _run_weights(harness, tmp_path, case, picks_prefix=None, ns=4, dh=0.1, sx=0.0,
                         sy=0.0, receivers=receivers)
    kept, ref = _reference_weights(4, 5.0e-4, 0.1, 0, 0.0, 0.0, 0.0, mute, 5.0, 0.0, 0.0,
                                   receivers, WEIGHT_PICKS)
    assert int(lines[0].split()[1]) == kept
    assert [[float(v) for v in ln.split()] for ln in lines[1:]] == ref
    # the tie is exercised: in binary32 the offset is exactly OFFSETC, while a
    # double offset from the binary32 DH would lie above it
    assert _f32(50 * _f32(0.1)) == 5.0 and 50 * _f32(0.1) > 5.0
    assert ref[0] == [0.0] * 4          # offset == OFFSETC is muted in both modes


def test_pick_reader_errors(harness: Path, tmp_path: Path) -> None:
    def run(*args):
        done = subprocess.run([str(harness), "picks", *args], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return int(re.search(r"status (-?\d+)", done.stdout).group(1)), done.stdout
    (tmp_path / "p_3.dat").write_text("0.1\n0.2\n", encoding="ascii")
    assert run(str(tmp_path / "p_"), "3", "2")[0] == 0
    assert run(str(tmp_path / "p_"), "3", "3")[0] == -4           # too few values
    assert run(str(tmp_path / "p_"), "4", "2")[0] == -3           # missing file
    (tmp_path / "p_5.dat").write_text("0.1\nnan\n", encoding="ascii")
    assert run(str(tmp_path / "p_"), "5", "2")[0] == -4           # non-finite


@pytest.mark.parametrize("change", [dict(timewin=2), dict(timewin=3), dict(gamma=-1.0),
                                    dict(twm=-0.01), dict(twp=-0.01), dict(mute=3),
                                    dict(mute=1, offsetc=0.0)])
def test_weights_reject_invalid_parameters(harness: Path, tmp_path: Path, change: dict) -> None:
    """TIMEWIN 2 is not supported in the exact path (fail closed); a constant
    window is a pick file with equal times."""
    (tmp_path / "picks_1.dat").write_text("".join(f"{p}\n" for p in WEIGHT_PICKS), encoding="ascii")
    case = dict(WEIGHT_CASES["window1"], **change)
    lines = _run_weights(harness, tmp_path, case, picks_prefix=str(tmp_path / "picks_"))
    assert lines == ["status -1"]


# --------------------------------------------------------------- driver level

N = 32
CELLS = N * N
TIME = 0.2
DT = 5.0e-4
NS = int(round(TIME / DT))
WORKFLOW_HEADER = (
    "PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ INV_VP_ITER "
    "INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP WD_DAMP1 EPRECOND "
    "LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE OFFSET_MUTE OFFSETC "
    "SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER")
TRAILING = (("MIGRATION_SOURCE_PREFIX", "migration/unused_source"),
            ("MIGRATION_DATA_PREFIX", "migration/unused_data"),
            ("MIGRATION_IMAGE_PREFIX", "migration/unused_image"),
            ("OPT_STEP_SCALING", "1"), ("OPT_WEIGHT_PRIMARY", "1.0"), ("OPT_WEIGHT_RHO", "1.0"),
            ("OPT_WEIGHT_Q", "1.0"), ("OPT_DIAG_OUT", "1"), ("OPT_PRECOND", "0"),
            ("OPT_PRECOND_EPS", "0.005"), ("OPT_ALPHA_ADAPT", "0"), ("OPT_ALPHA_ADAPT_FACTOR", "2.0"),
            ("OPT_LBFGS", "0"), ("OPT_LBFGS_M", "5"))
PICKS = [0.075 + abs(x - 160.0) / 2000.0 for x in (100, 120, 140, 180, 200, 220)]
MODES = {  # workflow S-2 columns
    "base": {},
    "lowpass": dict(time_filt=1, fc_high=12.0, order=4),
    "bandpass": dict(time_filt=1, fc_low=6.0, fc_high=15.0, order=3),
    "window1": dict(timewin=1, gamma=2000.0, twm=0.01, twp=0.02),
    "mute_far": dict(mute=1, offsetc=45.0),
    "mute_near": dict(mute=2, offsetc=45.0),
    "combined": dict(time_filt=1, fc_high=12.0, order=4, timewin=1, gamma=2000.0,
                     twm=0.01, twp=0.02, mute=1, offsetc=45.0),
}


def _row(*, pro=0.01, time_filt=0, fc_low=0.0, fc_high=20.0, order=2, timewin=0, gamma=1.0,
         twm=0.0, twp=0.0, mute=0, offsetc=10.0) -> str:
    values = [pro, time_filt, fc_low, fc_high, order, timewin, gamma, twm, twp, 0, 1, 1, 1,
              0, 0.5, 0.5, 0, 2, 0, 0, 0, 0.1, 0, mute, offsetc, 1, 1, 0, 0, 0]
    return " ".join(str(v) for v in values)


def _write_model(path: Path, values) -> None:
    path.write_bytes(struct.pack(f"={CELLS}f", *values))


def _read_model(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    assert len(raw) == 4 * CELLS, path
    return struct.unpack(f"={CELLS}f", raw)


def _parameter(path: Path, *, mode: int, model: str, output: str = "model/out",
               itermax: int = 1, nprocx: int = 1, quellart: int = 1,
               seis_vy: str = "su/observed_y.su", lbfgs: int = 0) -> None:
    write_sh_parameter(path, mode=mode, model_prefix=model, grad_method=0, inv_mod_out=1,
                       inv_model_file=output, itermax=itermax, nprocx=nprocx, eps_scale=0.01)
    text = path.read_text(encoding="ascii")
    for old, new in (("(TIME) = 0.08\n", f"(TIME) = {TIME}\n"),
                     ("(QUELLART) = 1\n", f"(QUELLART) = {quellart}\n"),
                     ("(SEIS_FILE_VY) = su/observed_y.su\n", f"(SEIS_FILE_VY) = {seis_vy}\n"),
                     ("(PICKS_FILE) = picks/unused\n", "(PICKS_FILE) = picks/pick_\n"),
                     ("(MIN_ITER) = 1\n", f"(MIN_ITER) = {itermax}\n")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    trailing = dict(TRAILING)
    trailing["OPT_LBFGS"] = str(lbfgs)
    text += "".join(f"({k}) = {trailing[k]}\n" for k, _ in TRAILING)
    path.write_text(text, encoding="ascii", newline="\n")


def _run(case: dict, parameter: Path, workflow: Path, ranks: int = 1):
    misfit = case["root"] / "log/misfit"
    if misfit.exists():
        misfit.unlink()
    return subprocess.run([case["mpiexec"], "--oversubscribe", "-n", str(ranks),
                           str(case["binary"]), str(parameter), str(workflow)],
                          cwd=case["root"], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=900)


def _misfit_rows(case: dict) -> list[list[str]]:
    path = case["root"] / "log/misfit"
    assert path.exists()
    return [ln.split() for ln in path.read_text().splitlines() if ln.strip()]


@pytest.fixture(scope="module")
def s2_case(tmp_path_factory: pytest.TempPathFactory, denise_binary: Path, mpiexec: str) -> dict:
    root = tmp_path_factory.mktemp("sh_s2_case")
    for name in ("model", "source", "receiver", "su", "log", "picks"):
        (root / name).mkdir(parents=True, exist_ok=True)
    true_vs = [1900.0 if 18 <= x <= 22 and 18 <= y <= 22 else 2000.0
               for x in range(N) for y in range(N)]
    for prefix, vs in (("true", true_vs), ("start", [2000.0] * CELLS)):
        _write_model(root / f"model/{prefix}.vs", vs)
        _write_model(root / f"model/{prefix}.rho", [2000.0] * CELLS)
        _write_model(root / f"model/{prefix}.qs", [80.0] * CELLS)
    (root / "source/one.dat").write_text("1\n160 0 160 0 20 1 0 1\n", encoding="ascii")
    (root / "receiver/line.dat").write_text(
        "100 160\n120 160\n140 160\n180 160\n200 160\n220 160\n", encoding="ascii")
    (root / "picks/pick_1.dat").write_text("".join(f"{p:.6f}\n" for p in PICKS), encoding="ascii")
    case = {"root": root, "binary": denise_binary, "mpiexec": mpiexec}
    workflow = root / "wf_plain.inp"
    workflow.write_text(WORKFLOW_HEADER + "\n" + _row() + "\n", encoding="ascii")
    forward = root / "forward.inp"
    _parameter(forward, mode=0, model="model/true")
    run = _run(case, forward, workflow)
    assert run.returncode == 0, run.stdout
    case["start"] = {s: (root / f"model/start.{s}").read_bytes() for s in ("vs", "rho", "qs")}
    return case


def _direction() -> dict[str, list[float]]:
    """Smooth interior bump between source/receivers and the slow patch."""
    phi = [math.exp(-((x - 16) ** 2 + (y - 20) ** 2) / (2 * 2.5 ** 2))
           for x in range(N) for y in range(N)]
    return {"vs": [40.0 * p for p in phi], "rho": [40.0 * p for p in phi],
            "qs": [8.0 * p for p in phi]}


def _fwi_j_and_gradient(case: dict, tag: str, model: str, workflow: Path, ranks: int,
                        want_gradient: bool):
    parameter = case["root"] / f"{tag}.inp"
    output = f"model/{tag}"
    _parameter(parameter, mode=1, model=model, output=output, nprocx=ranks)
    run = _run(case, parameter, workflow, ranks)
    assert run.returncode == 0 and "MPI_ABORT" not in run.stdout, run.stdout
    rows = _misfit_rows(case)
    j = float(rows[0][4])
    grad = None
    if want_gradient:
        grad = {s: _read_model(case["root"] / f"{output}_stage_1_it_1.grad_{s}")
                for s in ("vs", "rho", "qs")}
    return j, grad, run


@pytest.mark.integration
@pytest.mark.parametrize("name,ranks", [(n, 1) for n in MODES] +
                         [("window1", 2), ("combined", 2)])
def test_gradient_matches_central_differences(s2_case: dict, name: str, ranks: int) -> None:
    """Contract 5: d/de J(m + e*dm) by central differences equals <g, dm> of the
    raw exact gradient for the filtered, windowed and masked objective."""
    root = s2_case["root"]
    workflow = root / f"wf_{name}.inp"
    workflow.write_text(WORKFLOW_HEADER + "\n" + _row(pro=0.0, **MODES[name]) + "\n", encoding="ascii")
    j0, grad, run = _fwi_j_and_gradient(s2_case, f"g_{name}_{ranks}", "model/start", workflow,
                                        ranks, True)
    has_ops = bool(MODES[name])
    assert ("Exact SH data operators: stage 1" in run.stdout) == has_ops
    direction = _direction()
    base = {s: _read_model(root / f"model/start.{s}") for s in ("vs", "rho", "qs")}
    d_ad = math.fsum(g * d for s in ("vs", "rho", "qs") for g, d in zip(grad[s], direction[s]))
    assert abs(d_ad) > 0.0
    errors = []
    # Direction peak 2 % of Vs/rho and 10 % of Q at eps = 1.  Fourth-order central
    # difference (points +-eps, +-2 eps): the misfit of this fixture carries a
    # relative float noise of about 1.5e-5, so second-order differences cannot
    # make both the O(eps^2) truncation and the noise term small.
    for eps in (0.2, 0.1):
        js = {}
        for step in (2, 1, -1, -2):
            tag = f"{name}_{ranks}_{step}"
            prefix = root / f"model/p_{tag}"
            for s in ("vs", "rho", "qs"):
                _write_model(prefix.with_suffix(f".{s}"),
                             [b + step * eps * d for b, d in zip(base[s], direction[s])])
            js[step], _, _ = _fwi_j_and_gradient(s2_case, f"f_{tag}", f"model/p_{tag}",
                                                 workflow, ranks, False)
        d_fd = (8.0 * (js[1] - js[-1]) - (js[2] - js[-2])) / (12.0 * eps)
        errors.append(abs(d_fd - d_ad) / abs(d_ad))
    print(f"S2-FD {name} ranks={ranks} J0={j0:.6e} D_ad={d_ad:.6e} rel_errors={errors}")
    assert max(errors) <= 5.0e-3, errors


@pytest.mark.integration
def test_data_operators_change_the_objective(s2_case: dict) -> None:
    """Positive control: each operator changes J at the start model, and the
    far-offset mute removes data (smaller J)."""
    root = s2_case["root"]
    values = {}
    for name in ("base", "lowpass", "window1", "mute_far"):
        workflow = root / f"wfj_{name}.inp"
        workflow.write_text(WORKFLOW_HEADER + "\n" + _row(**MODES[name]) + "\n", encoding="ascii")
        values[name], _, _ = _fwi_j_and_gradient(s2_case, f"j_{name}", "model/start", workflow, 1, False)
    assert len(set(values.values())) == 4, values
    assert values["mute_far"] < values["base"] and values["window1"] < values["base"]


def _signal_file(path: Path, values: list[float]) -> None:
    path.write_text("".join(f"{_f32(v):.9e}\n" for v in values), encoding="ascii")


@pytest.mark.integration
def test_stage_filter_commutes_with_the_forward_solver(s2_case: dict, harness: Path,
                                                       tmp_path: Path) -> None:
    """Contract 2 (Koehn semantics): F(m; B w) = B F(m; w) up to rounding on
    samples 1..NT-1, with B the legacy apply_tdfilt used by the exact path.
    The last sample differs by construction (legacy leaves it unfiltered in
    B F(m; w)); this is the documented legacy property, not a claim of S-2."""
    root = s2_case["root"]
    ts, fc = 1.0 / 20.0, 20.0
    ricker = []
    for nt in range(1, NS + 1):
        t = nt * DT
        tau = math.pi * (t - 1.5 * ts) / (1.5 * ts)
        ricker.append((1.0 - 4.0 * tau * tau) * math.exp(-2.0 * tau * tau))
    _, (filtered,) = _filter(harness, tmp_path, [ricker], order=4, fc_high=12.0, fc_low=0.0, dt=DT)
    workflow = root / "wf_plain.inp"
    outputs = {}
    for tag, signal in (("raw", ricker), ("filtered", filtered)):
        _signal_file(root / f"source/sig_{tag}_shot_1.dat", signal)
        parameter = root / f"fwd_{tag}.inp"
        _parameter(parameter, mode=0, model="model/true", quellart=3, seis_vy=f"su/fwd_{tag}_y.su")
        text = parameter.read_text(encoding="ascii").replace(
            "(SIGNAL_FILE) = source/signal.dat\n", f"(SIGNAL_FILE) = source/sig_{tag}\n")
        parameter.write_text(text, encoding="ascii")
        run = _run(s2_case, parameter, workflow)
        assert run.returncode == 0, run.stdout
        samples = read_su_float_samples(root / f"su/fwd_{tag}_y.su.shot1", 6, NS)
        outputs[tag] = [samples[i * NS:(i + 1) * NS] for i in range(6)]
    _, b_of_raw = _filter(harness, tmp_path, outputs["raw"], order=4, fc_high=12.0, fc_low=0.0,
                          dt=DT, tag="bf")
    scale = max(abs(v) for tr in outputs["filtered"] for v in tr)
    worst = max(abs(a - b) for ta, tb in zip(outputs["filtered"], b_of_raw)
                for a, b in zip(ta[:-1], tb[:-1]))
    assert all(tb[-1] == tr[-1] for tb, tr in zip(b_of_raw, outputs["raw"]))   # legacy last sample
    print(f"S2-COMMUTE scale={scale:.6e} max_abs_diff={worst:.3e}")
    assert scale > 0.0 and worst <= 1.0e-5 * scale


@pytest.mark.integration
def test_multi_stage_run_accepts_steps_and_reloads_consistently(s2_case: dict) -> None:
    """Contract 1/3: three frequency stages with L-BFGS; each stage prints its
    operators, starts a fresh L-BFGS memory, accepts descending steps; a run of
    stage 2 alone from the written stage-1 model reproduces stage 2's first J."""
    root = s2_case["root"]
    rows = [_row(pro=0.0, time_filt=1, fc_high=fc, order=4, timewin=1, gamma=2000.0,
                 twm=0.01, twp=0.02) for fc in (8.0, 12.0, 18.0)]
    workflow = root / "wf_stages.inp"
    workflow.write_text(WORKFLOW_HEADER + "\n" + "\n".join(rows) + "\n", encoding="ascii")
    parameter = root / "stages.inp"
    _parameter(parameter, mode=1, model="model/start", output="model/stages", itermax=2, lbfgs=1)
    run = _run(s2_case, parameter, workflow)
    assert run.returncode == 0 and "MPI_ABORT" not in run.stdout, run.stdout
    for stage, fc in ((1, 8.0), (2, 12.0), (3, 18.0)):
        line = re.search(rf"Exact SH data operators: stage {stage} TIME_FILT 1 FC_low (\S+) "
                         rf"FC_high (\S+) ORDER 4 TIMEWIN 1", run.stdout)
        assert line and float(line.group(2)) == fc
        assert f"Exact SH L-BFGS: stage {stage} sigma" in run.stdout
        assert f"Exact SH L-BFGS: stage {stage} iteration" in run.stdout
    misfit = _misfit_rows(s2_case)
    assert len(misfit) == 6 and [int(r[-1]) for r in misfit] == [1, 1, 2, 2, 3, 3]
    for r in misfit:
        assert float(r[0]) > 0.0 and float(r[6]) < float(r[4])     # accepted, J decreases
    # Review R2 (path equality): within a stage, J_base of the next iteration
    # (objective/gradient path) equals the selected J of the line search
    # (objective path) at the same accepted model, as printed.
    for current, following in ((misfit[0], misfit[1]), (misfit[2], misfit[3]),
                               (misfit[4], misfit[5])):
        assert following[4] == current[6], (current, following)
    stage2_first_j = misfit[2][4]
    reload_wf = root / "wf_stage2.inp"
    reload_wf.write_text(WORKFLOW_HEADER + "\n" + rows[1] + "\n", encoding="ascii")
    parameter2 = root / "stage2_reload.inp"
    _parameter(parameter2, mode=1, model="model/stages_stage_1_it_2", output="model/reload",
               itermax=1, lbfgs=1)
    run2 = _run(s2_case, parameter2, reload_wf)
    assert run2.returncode == 0, run2.stdout
    j_reload = float(_misfit_rows(s2_case)[0][4])
    assert abs(j_reload - float(stage2_first_j)) <= 1.0e-6 * abs(float(stage2_first_j))


PRODUCTION_FILTER_MODE = "legacy"   # the exact path calls the legacy apply_tdfilt
RECEIVER_GRID = [(g, x // 10, 16) for g, x in enumerate((100, 120, 140, 180, 200, 220), start=1)]


@pytest.mark.integration
@pytest.mark.parametrize("name,ranks", [("window1", 1), ("window1", 2), ("mute_far", 1),
                                        ("mute_near", 1), ("lowpass", 1), ("bandpass", 1),
                                        ("combined", 1), ("combined", 2)])
def test_objective_matches_an_independent_reference(s2_case: dict, harness: Path,
                                                    tmp_path: Path, name: str, ranks: int) -> None:
    """Review R1 / plan T3: the logged J of the exact path equals
    J_ref = 1/2 sum_{traces, k=2..NT} (w (s - d))^2 computed offline from a
    MODE-0 forward run at the same model, the harness filter on source and
    observed data, and the independent weight formula.  Tolerances fixed
    before the first run: 2e-6 relative without filter (7-digit log, forward
    paths), 1e-4 with filter (offline Ricker in double)."""
    root = s2_case["root"]
    ops = MODES[name]
    filtered = ops.get("time_filt", 0) == 1
    workflow = root / f"wfr_{name}.inp"
    workflow.write_text(WORKFLOW_HEADER + "\n" + _row(**ops) + "\n", encoding="ascii")
    j_log, _, _ = _fwi_j_and_gradient(s2_case, f"r_{name}_{ranks}", "model/start", workflow,
                                      ranks, False)
    filter_args = dict(order=ops.get("order", 2), fc_high=ops.get("fc_high", 20.0),
                       fc_low=ops.get("fc_low", 0.0), dt=DT, mode=PRODUCTION_FILTER_MODE)
    parameter = root / f"ref_fwd_{name}.inp"
    if filtered:
        ts = 1.0 / 20.0
        ricker = [(1.0 - 4.0 * tau * tau) * math.exp(-2.0 * tau * tau)
                  for tau in (math.pi * (nt * DT - 1.5 * ts) / (1.5 * ts) for nt in range(1, NS + 1))]
        _, (source,) = _filter(harness, tmp_path, [ricker], tag="src", **filter_args)
        _signal_file(root / f"source/ref_{name}_shot_1.dat", source)
        _parameter(parameter, mode=0, model="model/start", quellart=3,
                   seis_vy=f"su/ref_{name}_y.su")
        text = parameter.read_text(encoding="ascii").replace(
            "(SIGNAL_FILE) = source/signal.dat\n", f"(SIGNAL_FILE) = source/ref_{name}\n")
        parameter.write_text(text, encoding="ascii")
    else:
        _parameter(parameter, mode=0, model="model/start", seis_vy=f"su/ref_{name}_y.su")
    run = _run(s2_case, parameter, root / "wf_plain.inp")
    assert run.returncode == 0, run.stdout
    raw_s = read_su_float_samples(root / f"su/ref_{name}_y.su.shot1", 6, NS)
    raw_d = read_su_float_samples(root / "su/observed_y.su.shot1", 6, NS)
    synthetic = [raw_s[i * NS:(i + 1) * NS] for i in range(6)]
    observed = [raw_d[i * NS:(i + 1) * NS] for i in range(6)]
    if filtered:
        _, observed = _filter(harness, tmp_path, observed, tag="obs", **filter_args)
    _, weights = _reference_weights(NS, DT, 10.0, ops.get("timewin", 0), ops.get("gamma", 1.0),
                                    ops.get("twm", 0.0), ops.get("twp", 0.0), ops.get("mute", 0),
                                    ops.get("offsetc", 10.0), 160.0, 160.0, RECEIVER_GRID, PICKS,
                                    mask_last=1 if filtered else 0)
    j_ref = 0.5 * math.fsum((w[k] * _f32(s[k] - d[k])) ** 2
                            for s, d, w in zip(synthetic, observed, weights) for k in range(1, NS))
    rel = abs(j_log - j_ref) / j_ref
    print(f"S2-JREF {name} ranks={ranks} J_log={j_log:.7e} J_ref={j_ref:.9e} rel={rel:.3e}")
    assert rel <= (1.0e-4 if filtered else 2.0e-6), (j_log, j_ref)


ABORTS = {
    "time_filt_2": dict(time_filt=2),
    "timewin_2": dict(timewin=2, gamma=1.0, twp=0.05),
    "fc_high_nyquist": dict(time_filt=1, fc_high=1000.0, order=4),
    "fc_low_above_high": dict(time_filt=1, fc_low=15.0, fc_high=12.0, order=4),
    "order_0": dict(time_filt=1, fc_high=12.0, order=0),
    "timewin_3": dict(timewin=3, gamma=1.0),
    "gamma_negative": dict(timewin=1, gamma=-1.0, twp=0.05),
    "twin_plus_negative": dict(timewin=1, gamma=1.0, twp=-0.01),
    "twin_minus_negative": dict(timewin=1, gamma=1.0, twm=-0.01),
    "mute_3": dict(mute=3),
    "offsetc_zero": dict(mute=1, offsetc=0.0),
}


@pytest.mark.integration
@pytest.mark.parametrize("name,ranks", [(n, 1) for n in ABORTS] + [("fc_high_nyquist", 2),
                                                                   ("missing_picks", 1),
                                                                   ("missing_picks", 2)])
def test_configuration_aborts_before_the_first_iteration(s2_case: dict, name: str,
                                                         ranks: int) -> None:
    root = s2_case["root"]
    workflow = root / f"wfa_{name}.inp"
    options = ABORTS.get(name, dict(timewin=1, gamma=1.0, twp=0.02))
    workflow.write_text(WORKFLOW_HEADER + "\n" + _row(**options) + "\n", encoding="ascii")
    parameter = root / f"abort_{name}_{ranks}.inp"
    _parameter(parameter, mode=1, model="model/start", output=f"model/abort_{name}", nprocx=ranks)
    if name == "missing_picks":
        text = parameter.read_text(encoding="ascii").replace(
            "(PICKS_FILE) = picks/pick_\n", "(PICKS_FILE) = picks/none_\n")
        parameter.write_text(text, encoding="ascii")
    run = _run(s2_case, parameter, workflow, ranks)
    assert run.returncode != 0 or "MPI_ABORT was invoked" in run.stdout, run.stdout
    assert "TDFWI ITERATION" not in run.stdout, run.stdout
    assert not (root / "log/misfit").exists()
    for s, raw in s2_case["start"].items():
        assert (root / f"model/start.{s}").read_bytes() == raw
    assert not list((root / "model").glob(f"abort_{name}*"))
