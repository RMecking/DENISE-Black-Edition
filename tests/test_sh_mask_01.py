"""SH-MASK: update mask from file and source taper in the exact viscoelastic
SH FWI (task packet SH-MASKE, 2026-11-26).

Records reused from the legacy interface: SWS_TAPER_FILE (78) with TFILE (79)
selects a 0/1 update mask; SWS_TAPER_GRAD_SOURCES (73) with SRTSHAPE 3 (75),
SRTRADIUS (76) and FILTSIZE (77) selects the source taper T, a factor of the
search-direction metric.  Driver level on the 32x32 SH-LBFGS-01 fixture:

M1  switches off: no new output, taper records not validated;
M2  effective mask and T equal an independent formula, every decomposition;
M3  fixed cells stay bitwise at the start model over iterations and a stage
    change, steps there are exactly 0, the printed diagnostic is 0;
M4  a restart at the stage boundary is bitwise equal to the continuous run;
M5  the raw gradient is bitwise unchanged by the switches;
M6  the L-BFGS direction equals the reference whose pairs live on the
    effective mask only and whose initial metric is T * P;
M7  directional derivative of J along the masked, tapered direction
    (central differences, descent);
M8  decompositions: mask, taper and fixed cells bitwise; models, gradient,
    step and log within the decomposition spread of the base (DECOMP_TOL);
M9  configuration aborts before the first iteration, naming the violated
    condition of the mask file;
M10 two sources: T is the minimum over the sources.
"""

from __future__ import annotations

import math
from pathlib import Path
import re
import struct

import pytest

from tests.test_sh_lbfgs_01 import (  # noqa: F401  (driver_case is used by name)
    DRIVER_CELLS, DRIVER_FW, DRIVER_N, MODEL_SUFFIXES, SCALED_EPS, STEP_TOLERANCE,
    _files, _grids, _lbfgs_log, _read_native, _run_denise, _to_grid, _workflow_row,
    _write_native_model, _write_parameter, _write_workflow, driver_case, ref, ref32)

N = DRIVER_N
DH = 10.0
SOURCE = (16, 16)                 # iround(160 / 10), global 1-based (I, J)
GRAD_SUFFIXES = ("grad_vs", "grad_rho", "grad_qs")
STEP_SUFFIXES = ("step_vs", "step_rho", "step_qs")
MASK_LINE = re.compile(
    r"Exact SH update mask: stage (\d+) iteration (\d+) fixed_change_max (\S+) "
    r"taper_share (\S+) max_change (\S+) at x (\S+) y (\S+)")


def _file_mask() -> list[list[float]]:
    """[J-1][I-1]: fixed top band J <= 10 (a 'deck') and a block I <= 12, J >= 24."""
    return [[0.0 if (j <= 10 or (i <= 12 and j >= 24)) else 1.0
             for i in range(1, N + 1)] for j in range(1, N + 1)]


def _write_mask_files(root: Path, name: str, grid, rho_grid=None, cells=DRIVER_CELLS) -> str:
    (root / "taper").mkdir(exist_ok=True)
    for suffix, g in (("u", grid), ("rho", rho_grid if rho_grid is not None else grid)):
        values = [g[y][x] for x in range(N) for y in range(N)][:cells]
        (root / f"taper/{name}_{suffix}.bin").write_bytes(struct.pack(f"={len(values)}f", *values))
    return f"taper/{name}"


def _taper_reference(radius: float, filtsize: int, sources=(SOURCE,)) -> list[list[float]]:
    out = []
    for j in range(1, N + 1):
        row = []
        for i in range(1, N + 1):
            t = 1.0
            for si, sj in sources:
                di, dj = i - si, j - sj
                if abs(di) <= filtsize and abs(dj) <= filtsize:
                    ts = 0.0
                else:
                    r = DH * math.hypot(di, dj)
                    ts = 1.0 if r >= radius else math.sin(math.pi * r / (2.0 * radius)) ** 2
                t = min(t, ts)
            row.append(ref32.f32(t))
        out.append(row)
    return out


def _effective_mask(file_mask=None, taper=None) -> list[list[float]]:
    cpml = ref32.update_mask(N, N, 0, 0, 1, 1, DRIVER_FW, 0, 0)
    return [[0.0 if (cpml[j][i] == 0.0 or (file_mask is not None and file_mask[j][i] == 0.0)
                     or (taper is not None and taper[j][i] == 0.0)) else 1.0
             for i in range(N)] for j in range(N)]


def _patch(path: Path, values: dict) -> None:
    text = path.read_text(encoding="ascii")
    for name, value in values.items():
        new, count = re.subn(rf"^\({name}\) = .*$", f"({name}) = {value}", text, flags=re.M)
        assert count == 1, name
        text = new
    path.write_text(text, encoding="ascii", newline="\n")


def _fwi(case: dict, name: str, *, taper: dict | None = None, itermax: int = 1,
         nprocx: int = 1, nprocy: int = 1, model_prefix: str = "model/start",
         workflow: Path | None = None, scaling: int = 1, precond: int = 0,
         lbfgs: int = 0, eps_scale: float = SCALED_EPS):
    output = f"model/{name}"
    parameter = case["root"] / f"{name}.inp"
    _write_parameter(parameter, records=133, mode=1, model_prefix=model_prefix,
                     grad_method=0, inv_mod_out=1, inv_model_file=output, itermax=itermax,
                     nprocx=nprocx, nprocy=nprocy, eps_scale=eps_scale, scaling=scaling,
                     diag=1, precond=precond, lbfgs=lbfgs)
    if taper:
        _patch(parameter, taper)
    misfit = case["root"] / "log/misfit"
    if misfit.exists():
        misfit.unlink()
    if workflow is None:
        workflow = case["root"] / "wf_mask_one.inp"
        _write_workflow(workflow, [_workflow_row(pro=0.0)])
    run = _run_denise(case["binary"], case["mpiexec"], case["root"], parameter, workflow,
                      ranks=nprocx * nprocy)
    return run, output


def _ok(run) -> None:
    assert run.returncode == 0 and "MPI_ABORT" not in run.stdout, run.stdout[-4000:]


def _misfit(case: dict) -> list[list[str]]:
    return [ln.split() for ln in (case["root"] / "log/misfit").read_text().splitlines()
            if ln.strip()]


def _switches(case: dict, *, mask: str | None = None, radius: float | None = None,
              filtsize: int = 0) -> dict:
    out = {}
    if mask is not None:
        out.update({"SWS_TAPER_FILE": 1, "TFILE": mask})
    if radius is not None:
        out.update({"SWS_TAPER_GRAD_SOURCES": 1, "SRTSHAPE": 3, "SRTRADIUS": radius,
                    "FILTSIZE": filtsize})
    return out


def _start(case: dict) -> list[tuple[float, ...]]:
    return [_read_native(case["start"].with_suffix(f".{s}")) for s in MODEL_SUFFIXES]


# M1 --------------------------------------------------------------------------

@pytest.mark.integration
def test_m1_switches_off_ignore_taper_records(driver_case: dict) -> None:
    """With records 73 and 78 at 0 the other taper records are not validated and
    nothing new is printed or written."""
    plain, out_a = _fwi(driver_case, "m1_plain", itermax=2, precond=2, lbfgs=1)
    _ok(plain)
    odd, out_b = _fwi(driver_case, "m1_odd", itermax=2, precond=2, lbfgs=1,
                      taper={"SRTSHAPE": 7, "SRTRADIUS": -1.0, "FILTSIZE": -3,
                             "TFILE": "taper/missing"})
    _ok(odd)
    assert "Exact SH update mask" not in plain.stdout + odd.stdout
    root = driver_case["root"]
    files_a = sorted(p.name.replace("m1_plain", "X") for p in (root / "model").glob("m1_plain_*"))
    files_b = sorted(p.name.replace("m1_odd", "X") for p in (root / "model").glob("m1_odd_*"))
    assert files_a == files_b and not any("mask" in f or "taper" in f for f in files_a)
    for f in files_a:
        assert (root / "model" / f.replace("X", "m1_plain")).read_bytes() == \
            (root / "model" / f.replace("X", "m1_odd")).read_bytes(), f


# M2 --------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("variant", ["file", "taper_f0", "taper_f1", "both"])
@pytest.mark.parametrize("procs", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_m2_mask_and_taper_fields_match_reference(driver_case: dict, variant: str,
                                                  procs) -> None:
    root = driver_case["root"]
    fmask = _file_mask() if variant in ("file", "both") else None
    mask_name = _write_mask_files(root, "m2", _file_mask()) if fmask else None
    radius = None if variant == "file" else 55.0
    filtsize = 1 if variant == "taper_f1" else 0
    run, out = _fwi(driver_case, f"m2_{variant}_{procs[0]}x{procs[1]}", nprocx=procs[0],
                    nprocy=procs[1], taper=_switches(driver_case, mask=mask_name,
                                                     radius=radius, filtsize=filtsize))
    _ok(run)
    taper = _taper_reference(radius, filtsize) if radius is not None else None
    want_mask = _effective_mask(fmask, taper)
    got_mask = _grids(driver_case, out, 1, ("update_mask",))[0]
    assert got_mask == want_mask
    free = int(sum(sum(r) for r in want_mask))
    assert f"free_cells {free} of {N * N}" in run.stdout
    if taper is not None:
        got = _grids(driver_case, out, 1, ("taper",))[0]
        for j in range(N):
            for i in range(N):
                w = taper[j][i]
                assert abs(got[j][i] - w) <= 1.2e-7 * max(abs(w), 1e-30), (i, j, got[j][i], w)
        assert any(0.0 < v < 1.0 for row in got for v in row)
    else:
        assert not _files(driver_case, out, 1, ["taper"])["taper"].exists()


# M3, M4 ----------------------------------------------------------------------

def _two_stage(case: dict, name: str, switches: dict, **kw):
    wf = case["root"] / "wf_mask_two.inp"
    _write_workflow(wf, [_workflow_row(pro=0.0), _workflow_row(pro=0.0)])
    return _fwi(case, name, itermax=2, workflow=wf, taper=switches, **kw)


@pytest.mark.integration
@pytest.mark.parametrize("precond,lbfgs", [(0, 0), (2, 1), (0, 1)])
def test_m3_fixed_cells_stay_bitwise_over_iterations_and_stages(driver_case: dict,
                                                                precond: int,
                                                                lbfgs: int) -> None:
    root = driver_case["root"]
    mask_name = _write_mask_files(root, "m3", _file_mask())
    switches = _switches(driver_case, mask=mask_name, radius=55.0)
    run, out = _two_stage(driver_case, f"m3_{precond}_{lbfgs}", switches, precond=precond,
                          lbfgs=lbfgs)
    _ok(run)
    eff = _effective_mask(_file_mask(), _taper_reference(55.0, 0))
    start = [_to_grid(v) for v in _start(driver_case)]
    changed_free = 0
    for stage in (1, 2):
        for it in (1, 2):
            models = _grids(driver_case, out, it, MODEL_SUFFIXES, stage)
            steps = _grids(driver_case, out, it, STEP_SUFFIXES, stage)
            for c in range(3):
                for j in range(N):
                    for i in range(N):
                        if eff[j][i] == 0.0:
                            assert struct.pack("=f", models[c][j][i]) == \
                                struct.pack("=f", start[c][j][i]), (stage, it, c, i, j)
                            assert steps[c][j][i] == 0.0, (stage, it, c, i, j)
                        elif models[c][j][i] != start[c][j][i]:
                            changed_free += 1
    assert changed_free > 0
    lines = MASK_LINE.findall(run.stdout)
    assert [(int(s), int(k)) for s, k, *_ in lines] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    for _, _, fixed, share, maxc, x, y in lines:
        assert float(fixed) == 0.0
        assert 0.0 <= float(share) <= 1.0 and float(maxc) > 0.0
    for r in _misfit(driver_case):
        assert float(r[5]) < float(r[4])          # accepted < base (columns 6 / 5)
    if lbfgs:
        actions = [l["action"] for l in _lbfgs_log(run)]
        assert "lbfgs" in actions, actions


@pytest.mark.integration
def test_m4_restart_at_stage_boundary_is_bitwise(driver_case: dict) -> None:
    root = driver_case["root"]
    mask_name = _write_mask_files(root, "m4", _file_mask())
    switches = _switches(driver_case, mask=mask_name, radius=55.0)
    run, out = _two_stage(driver_case, "m4_cont", switches, precond=2, lbfgs=1)
    _ok(run)
    cont_rows = _misfit(driver_case)
    reload, rout = _fwi(driver_case, "m4_reload", itermax=2, taper=switches, precond=2,
                        lbfgs=1, model_prefix=f"{out}_stage_1_it_2")
    _ok(reload)
    rows = _misfit(driver_case)
    assert [r[:-1] for r in rows] == [r[:-1] for r in cont_rows[2:]]
    suffixes = MODEL_SUFFIXES + GRAD_SUFFIXES + STEP_SUFFIXES + (
        "energy", "precond", "update_mask", "taper")
    for it in (1, 2):
        a = _files(driver_case, out, it, suffixes, stage=2)
        b = _files(driver_case, rout, it, suffixes, stage=1)
        for s in suffixes:
            assert a[s].read_bytes() == b[s].read_bytes(), (it, s)


# M5 --------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("precond", [0, 2])
def test_m5_raw_gradient_unchanged(driver_case: dict, precond: int) -> None:
    root = driver_case["root"]
    mask_name = _write_mask_files(root, "m5", _file_mask())
    off, oout = _fwi(driver_case, f"m5_off_{precond}", precond=precond)
    _ok(off)
    j_off = _misfit(driver_case)[0][4]
    on, nout = _fwi(driver_case, f"m5_on_{precond}", precond=precond,
                    taper=_switches(driver_case, mask=mask_name, radius=55.0))
    _ok(on)
    a = _files(driver_case, oout, 1, GRAD_SUFFIXES + (("energy",) if precond else ()))
    b = _files(driver_case, nout, 1, GRAD_SUFFIXES + (("energy",) if precond else ()))
    for s in a:
        assert a[s].read_bytes() == b[s].read_bytes(), s
    assert _misfit(driver_case)[0][4] == j_off          # same base J
    assert _files(driver_case, oout, 1, ["step_vs"])["step_vs"].read_bytes() != \
        _files(driver_case, nout, 1, ["step_vs"])["step_vs"].read_bytes()


# M6 --------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("precond", [2, 0])
def test_m6_lbfgs_pairs_on_effective_mask_and_metric_t_times_p(driver_case: dict,
                                                               precond: int) -> None:
    root = driver_case["root"]
    mask_name = _write_mask_files(root, "m6", _file_mask())
    run, out = _fwi(driver_case, f"m6_{precond}", itermax=2, precond=precond, lbfgs=1,
                    taper=_switches(driver_case, mask=mask_name, radius=55.0))
    _ok(run)
    log = _lbfgs_log(run)
    assert [(l["iteration"], l["action"], l["pair"]) for l in log] == \
        [(1, "sd_empty", "none"), (2, "lbfgs", "stored")]
    taper = _taper_reference(55.0, 0)
    eff = _effective_mask(_file_mask(), taper)
    assert _grids(driver_case, out, 1, ("update_mask",))[0] == eff
    t_grid = _grids(driver_case, out, 1, ("taper",))[0]
    start = [_to_grid(v) for v in _start(driver_case)]
    m2 = _grids(driver_case, out, 1, MODEL_SUFFIXES)
    g1 = _grids(driver_case, out, 1, GRAD_SUFFIXES)
    g2 = _grids(driver_case, out, 2, GRAD_SUFFIXES)
    if precond:
        p1 = _grids(driver_case, out, 1, ("precond",))[0]
        p2 = _grids(driver_case, out, 2, ("precond",))[0]
        for j in range(N):                       # D = T * P on free cells, 0 elsewhere
            for i in range(N):
                if eff[j][i] == 0.0:
                    assert p1[j][i] == 0.0 and p2[j][i] == 0.0
    else:
        p1 = p2 = t_grid
    zeros = [[[0.0] * N for _ in range(N)] for _ in range(3)]
    model = ref.LbfgsReference(5)
    model.begin_stage(start, eff)
    assert model.step(start, g1, eff, p1, (1.0, 1.0, 1.0), zeros)["action"] == 0
    second = model.step(m2, g2, eff, p2, (1.0, 1.0, 1.0), zeros)
    assert second["action"] == 3
    got = _grids(driver_case, out, 2, STEP_SUFFIXES)
    for c in range(3):
        scale = max(abs(v) for row in second["step"][c] for v in row)
        worst = max(abs(got[c][j][i] - second["step"][c][j][i])
                    for j in range(N) for i in range(N))
        assert worst <= STEP_TOLERANCE * scale, (c, worst, scale)
        assert all(got[c][j][i] == 0.0 for j in range(N) for i in range(N) if eff[j][i] == 0.0)
    # Discrimination: pairs on the CPML mask alone give a different direction.
    other = ref.LbfgsReference(5)
    cpml = _effective_mask()
    other.begin_stage(start, cpml)
    other.step(start, g1, cpml, [[v if v > 0.0 else 1.0 for v in row] for row in p1],
               (1.0, 1.0, 1.0), zeros)
    alt = other.step(m2, g2, cpml, [[v if v > 0.0 else 1.0 for v in row] for row in p2],
                     (1.0, 1.0, 1.0), zeros)
    worst = max(abs(got[0][j][i] - alt["step"][0][j][i]) for j in range(N) for i in range(N))
    assert worst > 1.0e-3 * max(abs(v) for row in alt["step"][0] for v in row)


# M7 --------------------------------------------------------------------------

def _j_at(case: dict, name: str, prefix: str) -> float:
    run, _ = _fwi(case, name, model_prefix=prefix)
    _ok(run)
    return float(_misfit(case)[0][4])


@pytest.mark.integration
@pytest.mark.parametrize("precond", [0, 2])
def test_m7_directional_derivative_along_masked_tapered_direction(driver_case: dict,
                                                                  precond: int) -> None:
    """d/dh J(m - h p) at h = 0 equals -<g, p> < 0 for the direction p the
    driver builds with mask and taper (fourth-order central differences)."""
    root = driver_case["root"]
    mask_name = _write_mask_files(root, "m7", _file_mask())
    run, out = _fwi(driver_case, f"m7_{precond}", precond=precond,
                    taper=_switches(driver_case, mask=mask_name, radius=55.0))
    _ok(run)
    g = [_read_native(f) for f in _files(driver_case, out, 1, GRAD_SUFFIXES).values()]
    p = [_read_native(f) for f in _files(driver_case, out, 1, STEP_SUFFIXES).values()]
    base = _start(driver_case)
    gp = math.fsum(a * b for gc, pc in zip(g, p) for a, b in zip(gc, pc))
    assert gp > 0.0                                   # descent for m - h p
    h0 = 8.0 / max(abs(v) for v in p[0])              # largest Vs change 8 m/s at h0
    errors = []
    for h in (h0, 0.5 * h0):
        js = {}
        for k in (2, 1, -1, -2):
            prefix = root / f"model/m7p_{precond}_{k}"
            for s, bc, pc in zip(MODEL_SUFFIXES, base, p):
                _write_native_model(prefix.with_suffix(f".{s}"),
                                    [b - k * h * d for b, d in zip(bc, pc)])
            js[k] = _j_at(driver_case, f"m7j_{precond}_{k}", f"model/m7p_{precond}_{k}")
        d_fd = (8.0 * (js[1] - js[-1]) - (js[2] - js[-2])) / (12.0 * h)
        errors.append(abs(d_fd + gp) / gp)
        assert d_fd < 0.0
    print(f"M7 precond={precond} <g,p>={gp:.6e} rel_errors={errors}")
    assert max(errors) <= 5.0e-3, errors


# M8 --------------------------------------------------------------------------

# The exact SH FWI of the base c76ce8ba is not bitwise reproducible across MPI
# decompositions: with all switches off, the raw gradient of iteration 1 (same
# start model) differs between 1x1 and 2x1 / 1x2 / 2x2 in its last FP32 bits;
# models then differ by about 1 ulp in a few cells and the differences grow in
# iteration 2 (independent review 2026-11-26, section 7, and a base-binary
# measurement on this 32x32 fixture, 3 optimizer configurations x 4
# decompositions x 2 iterations, relative to the 1x1 run):
#   raw gradient and step, iteration 1          <= 3.6e-7 of the field maximum
#   raw gradient and step, iteration 2          <= 1.3e-4
#   models after iteration 1 and 2              <= 9.6e-8 of the class maximum
#   logged J (columns 5-8)                      <= 4.1e-5 of the start J
#   logged step lengths (columns 1-4)           <= 4.5e-5 relative
# The package claim is therefore checked bitwise only where it is the
# package's own: effective mask, taper, fixed cells at the start value, and a
# zero change on fixed cells in the diagnostics.  Everything that inherits the
# base's decomposition spread is checked with a tolerance about ten times the
# base measurement.  This corrects a test that demanded more than the base
# guarantees; it does not weaken a package claim.
DECOMP_TOL = {"grad1": 1.0e-6, "grad2": 1.0e-3, "model": 1.0e-6, "j": 4.0e-4,
              "alpha": 4.0e-4}


def _rel_field(a: Path, b: Path) -> float:
    x, y = _read_native(a), _read_native(b)
    scale = max(abs(v) for v in x)
    assert scale > 0.0, a
    return max(abs(u - v) for u, v in zip(x, y)) / scale


@pytest.mark.integration
@pytest.mark.parametrize("config", ["max_sd", "rms_precond_lbfgs"])
def test_m8_decompositions(driver_case: dict, config: str) -> None:
    """1x1, 2x1, 1x2, 2x2: mask, taper and fixed cells bitwise; the rest within
    the decomposition spread of the base (see DECOMP_TOL)."""
    root = driver_case["root"]
    mask_name = _write_mask_files(root, "m8", _file_mask())
    switches = _switches(driver_case, mask=mask_name, radius=55.0, filtsize=1)
    options = (dict(scaling=2) if config == "max_sd"
               else dict(scaling=1, precond=2, lbfgs=1))
    eff = _effective_mask(_file_mask(), _taper_reference(55.0, 1))
    start = [struct.pack(f"={DRIVER_CELLS}f", *v) for v in _start(driver_case)]
    outs = {}
    for procs in ((1, 1), (2, 1), (1, 2), (2, 2)):
        run, out = _fwi(driver_case, f"m8_{config}_{procs[0]}x{procs[1]}", itermax=2,
                        nprocx=procs[0], nprocy=procs[1], taper=switches, **options)
        _ok(run)
        lines = MASK_LINE.findall(run.stdout)
        assert len(lines) == 2 and all(float(l[2]) == 0.0 for l in lines), lines
        outs[procs] = (out, _misfit(driver_case))
    ref_out, ref_rows = outs[(1, 1)]
    j1 = float(ref_rows[0][4])
    for procs, (out, rows) in outs.items():
        assert len(rows) == len(ref_rows) == 2
        for r0, r in zip(ref_rows, rows):
            for c in range(4, 8):
                assert abs(float(r[c]) - float(r0[c])) <= DECOMP_TOL["j"] * j1, (procs, c)
            for c in range(0, 4):
                assert abs(float(r[c]) - float(r0[c])) <= \
                    DECOMP_TOL["alpha"] * abs(float(r0[c])), (procs, c)
        for it in (1, 2):
            a = _files(driver_case, ref_out, it, MODEL_SUFFIXES + GRAD_SUFFIXES +
                       STEP_SUFFIXES + ("update_mask", "taper"))
            b = _files(driver_case, out, it, MODEL_SUFFIXES + GRAD_SUFFIXES +
                       STEP_SUFFIXES + ("update_mask", "taper"))
            for s in ("update_mask", "taper"):                      # bitwise
                assert a[s].read_bytes() == b[s].read_bytes(), (procs, it, s)
            for c, s in enumerate(MODEL_SUFFIXES):                   # fixed cells bitwise
                got = b[s].read_bytes()
                for x in range(N):
                    for y in range(N):
                        if eff[y][x] == 0.0:
                            k = 4 * (x * N + y)
                            assert got[k:k + 4] == start[c][k:k + 4], (procs, it, s, x, y)
                assert _rel_field(a[s], b[s]) <= DECOMP_TOL["model"], (procs, it, s)
            for s in GRAD_SUFFIXES + STEP_SUFFIXES:
                tol = DECOMP_TOL["grad1"] if it == 1 else DECOMP_TOL["grad2"]
                assert _rel_field(a[s], b[s]) <= tol, (procs, it, s)


# N4 (review): T is the minimum over several sources -----------------------------

TWO_SOURCES = ((160.0, 160.0), (220.0, 200.0))   # grid (16, 16) and (22, 20), radii overlap


@pytest.mark.integration
@pytest.mark.parametrize("procs", [(1, 1), (2, 2)])
def test_m10_taper_is_minimum_over_sources(driver_case: dict, procs) -> None:
    root = driver_case["root"]
    (root / "source/two.dat").write_text(
        "2\n" + "".join(f"{x:g} 0 {y:g} 0 20 1 0 1\n" for x, y in TWO_SOURCES),
        encoding="ascii")
    two = {"SOURCE_FILE": "source/two.dat", "SEIS_FILE_VY": "su/two_y.su",
           "DATA_DIR": "su/two", "TESTSHOTS": "1, 2, 1"}
    if not (root / "su/two_y.su.shot2").exists():
        forward = root / "two_forward.inp"
        from tests.utilities.sh_fwi_fixture import write_sh_parameter
        write_sh_parameter(forward)
        _patch(forward, two)
        _ok(_run_denise(driver_case["binary"], driver_case["mpiexec"], root, forward,
                        driver_case["workflows"]["one"]))
        assert (root / "su/two_y.su.shot2").exists()
    run, out = _fwi(driver_case, f"m10_{procs[0]}x{procs[1]}", nprocx=procs[0],
                    nprocy=procs[1], taper={**two, **_switches(driver_case, radius=55.0)})
    _ok(run)
    sources = tuple((int(round(x / DH)), int(round(y / DH))) for x, y in TWO_SOURCES)
    taper = _taper_reference(55.0, 0, sources)
    single = [_taper_reference(55.0, 0, (s,)) for s in sources]
    # The minimum is exercised: cells where both single-source tapers are < 1
    # and differ, and cells reached only by the second source.
    both = [(i, j) for j in range(N) for i in range(N)
            if single[0][j][i] < 1.0 and single[1][j][i] < 1.0
            and single[0][j][i] != single[1][j][i]]
    assert len(both) > 5
    got = _grids(driver_case, out, 1, ("taper",))[0]
    for j in range(N):
        for i in range(N):
            w = taper[j][i]
            assert abs(got[j][i] - w) <= 1.2e-7 * max(abs(w), 1e-30), (i, j, got[j][i], w)
    for i, j in both:
        assert got[j][i] == min(single[0][j][i], single[1][j][i]) or \
            abs(got[j][i] - min(single[0][j][i], single[1][j][i])) <= 1.2e-7
    assert _grids(driver_case, out, 1, ("update_mask",))[0] == _effective_mask(None, taper)
    assert got[sources[1][1] - 1][sources[1][0] - 1] == 0.0      # second source cell fixed


# M9 --------------------------------------------------------------------------

def _abort_cases(root: Path) -> dict:
    good = _write_mask_files(root, "m9_good", _file_mask())
    half = [[0.5 if (i, j) == (20, 20) else v for i, v in enumerate(row)]
            for j, row in enumerate(_file_mask())]
    rho_other = [[0.0 if (i, j) == (20, 20) else v for i, v in enumerate(row)]
                 for j, row in enumerate(_file_mask())]
    return {
        "file_without_scaling": ({"SWS_TAPER_FILE": 1, "TFILE": good}, 0),
        "taper_without_scaling": ({"SWS_TAPER_GRAD_SOURCES": 1, "SRTSHAPE": 3,
                                   "SRTRADIUS": 55.0, "FILTSIZE": 0}, 0),
        "file_missing": ({"SWS_TAPER_FILE": 1, "TFILE": "taper/none"}, 1),
        "file_short": ({"SWS_TAPER_FILE": 1,
                        "TFILE": _write_mask_files(root, "m9_short", _file_mask(),
                                                   cells=DRIVER_CELLS - 1)}, 1),
        "file_half_value": ({"SWS_TAPER_FILE": 1,
                             "TFILE": _write_mask_files(root, "m9_half", half)}, 1),
        "files_differ": ({"SWS_TAPER_FILE": 1,
                          "TFILE": _write_mask_files(root, "m9_diff", _file_mask(),
                                                     rho_grid=rho_other)}, 1),
        "file_all_fixed": ({"SWS_TAPER_FILE": 1,
                            "TFILE": _write_mask_files(root, "m9_zero",
                                                       [[0.0] * N for _ in range(N)])}, 1),
        "file_value_two": ({"SWS_TAPER_FILE": 2, "TFILE": good}, 1),
        "shape_erf": ({"SWS_TAPER_GRAD_SOURCES": 1, "SRTSHAPE": 1, "SRTRADIUS": 55.0,
                       "FILTSIZE": 0}, 1),
        "radius_below_dh": ({"SWS_TAPER_GRAD_SOURCES": 1, "SRTSHAPE": 3, "SRTRADIUS": 9.5,
                             "FILTSIZE": 0}, 1),
        "filtsize_negative": ({"SWS_TAPER_GRAD_SOURCES": 1, "SRTSHAPE": 3,
                               "SRTRADIUS": 55.0, "FILTSIZE": -1}, 1),
        "per_shot_taper": ({"SWS_TAPER_CIRCULAR_PER_SHOT": 1}, 1),
    }


@pytest.mark.integration
@pytest.mark.parametrize("name", ["file_without_scaling", "taper_without_scaling",
                                  "file_missing", "file_short", "file_half_value",
                                  "files_differ", "file_all_fixed", "file_value_two",
                                  "shape_erf", "radius_below_dh", "filtsize_negative",
                                  "per_shot_taper"])
@pytest.mark.parametrize("ranks", [1, 2])
def test_m9_configuration_aborts_before_first_iteration(driver_case: dict, name: str,
                                                        ranks: int) -> None:
    switches, scaling = _abort_cases(driver_case["root"])[name]
    run, out = _fwi(driver_case, f"m9_{name}_{ranks}", nprocx=ranks, scaling=scaling,
                    taper=switches)
    assert run.returncode != 0 or "MPI_ABORT was invoked" in run.stdout, run.stdout[-3000:]
    assert "TDFWI ITERATION" not in run.stdout, run.stdout[-3000:]
    assert not (driver_case["root"] / "log/misfit").exists()
    assert not list((driver_case["root"] / "model").glob(f"{Path(out).name}_*"))
    if name in ABORT_REASONS:                     # review N5: the violated condition
        assert ABORT_REASONS[name] in run.stdout, run.stdout[-3000:]


ABORT_REASONS = {
    "file_missing": "rejected: a file is missing or cannot be read (status -2)",
    "file_short": "rejected: a file does not hold exactly NXG x NYG float32 values (status -3)",
    "file_half_value": "rejected: a value is not exactly 0 or 1 (status -4)",
    "files_differ": "rejected: the _u.bin and _rho.bin files differ (status -5)",
    "file_all_fixed": "the update mask leaves no free cell",
}
