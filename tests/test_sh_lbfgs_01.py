"""SH-LBFGS-01 acceptance oracles O1-O12.

Written before the implementation from the frozen contract
``docs/sh_lbfgs_01_contract.md`` (SHA-256 pinned below).  Revision 2 applies
the independent oracle review (READY WITH REQUIRED CORRECTIONS) of the r1 set.  Expected values
come only from that contract and from the independent reference
``tests/utilities/sh_lbfgs_01_reference.py``; production code is exercised
exclusively through its public C API and the compiled driver.

Oracle map (contract section 5):
  O1  identity of records 131 vs 133 with L-BFGS off       -> test_o1_*
  O2  non-interference, iteration-1 identity, branch reached -> test_o2_*
  O3  two-loop closed form (unit and driver)                 -> test_o3_*
  O4  dense BFGS matrix                                      -> test_o4_*
  O5  secant condition                                       -> test_o5_*
  O6  quadratic finite termination H = A^-1                  -> test_o6_*
  O7  safeguards, resets, ring buffer, stage start           -> test_o7_*
  O8  projection on mask and active classes                  -> test_o8_*
  O9  reparametrisation invariance                           -> test_o9_*
  O10 MPI decompositions                                     -> test_o10_*
  O11 error paths, configuration aborts, records             -> test_o11_*
  O12 descent sign                                           -> test_o12_*
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess

import pytest

from tests.utilities import sh_lbfgs_01_reference as ref
from tests.utilities import sh_opt_scale_01_reference as ref32
from tests.utilities.sh_fwi_fixture import write_sh_parameter


CONTRACT_PATH = "docs/sh_lbfgs_01_contract.md"
CONTRACT_SHA256 = "36e8761be1eb9e677e4494f5983109e2e7a136a7a7ed6298b3cd243fc89d780f"

# Contract section 5 tolerances; never widen these.
STEP_TOLERANCE = 1.0e-6          # |got - want| <= tol * max|want| per class field
SCALAR_TOLERANCE = 1.0e-7        # report scalars, relative
QUADRATIC_TOLERANCE = 1.0e-4     # O6, relative to max
PRINTED_TOLERANCE = 2.0e-6       # printed %e scalars, relative

CASE_MAGIC = 0x31464231
RESULT_MAGIC = 0x52464231
OUTPUT_SENTINEL = -98765.25
REPORT_SENTINEL = -7777
NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _launcher() -> str:
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert launcher, "an MPI launcher is required for the SH-LBFGS-01 harness"
    return launcher


def _rel_close(left: float, right: float, tol: float) -> bool:
    if left == 0.0 or right == 0.0:
        return left == right
    return abs(left - right) <= tol * max(abs(left), abs(right))


def _field(ny: int, nx: int, function) -> list[list[float]]:
    return [[ref32.f32(function(j, i)) for i in range(nx)] for j in range(ny)]


def _const(ny: int, nx: int, value: float) -> list[list[float]]:
    return [[ref32.f32(value)] * nx for _ in range(ny)]


def _frame_mask(ny: int, nx: int, width: int) -> list[list[float]]:
    return [[0.0 if (i < width or i >= nx - width or j < width or j >= ny - width)
             else 1.0 for i in range(nx)] for j in range(ny)]


def _copy3(fields):
    return [[list(row) for row in f] for f in fields]


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory, repository_root: Path) -> Path:
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the SH-LBFGS-01 harness"
    output = tmp_path_factory.mktemp("sh_lbfgs_01") / "harness"
    command = [
        compiler, "-std=c99", "-O2", "-fcommon", "-I", str(repository_root / "include"),
        str(repository_root / "tests/utilities/sh_lbfgs_01_harness.c"),
        str(repository_root / "src/SH/visco_sh_exact_lbfgs.c"),
        "-o", str(output), "-lm",
    ]
    run = subprocess.run(command, cwd=repository_root, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert run.returncode == 0, run.stdout
    return output


def _pack(grid, nxg: int, nyg: int) -> bytes:
    assert len(grid) == nyg and all(len(row) == nxg for row in grid)
    return struct.pack(f"={nxg * nyg}f", *(v for row in grid for v in row))


def _run(harness: Path, workdir: Path, *, stage, iterations, nxg: int, nyg: int,
         nprocx: int = 1, nprocy: int = 1, memory: int = 5, has_precond: bool = False,
         fault: int = 0, fault_rank: int = -1, fault_iter: int = 0, skip_begin: int = 0,
         op: int = 1) -> dict:
    """Run one case; returns collective flags (asserted equal over ranks)."""
    workdir.mkdir(parents=True, exist_ok=True)
    payload = bytearray(struct.pack(
        "=13i", CASE_MAGIC, op, nxg, nyg, nprocx, nprocy, memory, len(iterations),
        int(has_precond), fault, fault_rank, fault_iter, skip_begin))
    if op == 1:
        bases, mask = stage
        for grid in list(bases) + [mask]:
            payload += _pack(grid, nxg, nyg)
        for it in iterations:
            payload += struct.pack("=i3f", it.get("kind", 0), *it["weights"])
            for grid in list(it["base"]) + list(it["grad"]) + list(it["step_in"]):
                payload += _pack(grid, nxg, nyg)
            payload += _pack(it["mask"], nxg, nyg)
            if has_precond:
                payload += _pack(it["precond"], nxg, nyg)
    case = workdir / "case.bin"
    case.write_bytes(bytes(payload))
    ranks = nprocx * nprocy
    run = subprocess.run(
        [_launcher(), "--oversubscribe", "-n", str(ranks), str(harness), str(case),
         str(workdir / "result")],
        cwd=workdir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=300)
    assert run.returncode == 0, run.stdout
    per_rank = []
    for rank in range(ranks):
        raw = (workdir / f"result.{rank}").read_bytes()
        head = struct.unpack("=9i", raw[:36])
        magic, file_rank, pos_x, pos_y, nx, ny, niter, init_status, begin_status = head
        assert magic == RESULT_MAGIC and file_rank == rank
        offset, its = 36, []
        count = (nx + 2) * (ny + 2)
        for _ in range(niter):
            flags = struct.unpack("=8i", raw[offset:offset + 32]); offset += 32
            scalars = struct.unpack("=7d", raw[offset:offset + 56]); offset += 56
            arrays = []
            for _c in range(3):
                vals = struct.unpack(f"={count}f", raw[offset:offset + 4 * count])
                offset += 4 * count
                arrays.append([list(vals[j * (nx + 2):(j + 1) * (nx + 2)])
                               for j in range(ny + 2)])
            its.append({"flags": flags, "scalars": scalars, "arrays": arrays})
        assert offset == len(raw)
        per_rank.append({"pos": (pos_x, pos_y), "nx": nx, "ny": ny, "init": init_status,
                         "begin": begin_status, "its": its})
    # Collective decisions: every rank reports the same status, flags and scalars.
    first = per_rank[0]
    for other in per_rank[1:]:
        assert other["init"] == first["init"] and other["begin"] == first["begin"]
        for a, b in zip(first["its"], other["its"]):
            assert a["flags"][:5] == b["flags"][:5], (a["flags"], b["flags"])
            assert all((x == y) or (math.isnan(x) and math.isnan(y))
                       for x, y in zip(a["scalars"], b["scalars"]))
    result = {"init": first["init"], "begin": first["begin"], "its": []}
    for k in range(len(first["its"])):
        steps, halos = [], True
        for c in range(3):
            out = [[None] * nxg for _ in range(nyg)]
            for r in per_rank:
                nx, ny = r["nx"], r["ny"]
                px, py = r["pos"]
                arr = r["its"][k]["arrays"][c]
                for j in range(ny + 2):
                    for i in range(nx + 2):
                        if j in (0, ny + 1) or i in (0, nx + 1):
                            halos &= arr[j][i] == OUTPUT_SENTINEL
                        else:
                            out[py * ny + j - 1][px * nx + i - 1] = arr[j][i]
            steps.append(out)
        flags = first["its"][k]["flags"]
        result["its"].append({
            "status": flags[0], "begin": flags[1], "action": flags[2], "pair": flags[3],
            "pairs_used": flags[4],
            "step_unchanged": all(r["its"][k]["flags"][5] for r in per_rank),
            "inputs_unchanged": all(r["its"][k]["flags"][6] for r in per_rank),
            "report_untouched": all(r["its"][k]["flags"][7] for r in per_rank),
            "curvature": first["its"][k]["scalars"][0],
            "gamma": first["its"][k]["scalars"][1],
            "descent": first["its"][k]["scalars"][2],
            "kappa": first["its"][k]["scalars"][3],
            "sigma": list(first["its"][k]["scalars"][4:7]),
            "step": steps, "halos": halos})
    return result


# ---------------------------------------------------------------------------
# Synthetic sequences
# ---------------------------------------------------------------------------

NXG, NYG = 12, 6
CLASS_CENTRES = (2000.0, 1800.0, 60.0)


def _spd_block(rng: random.Random) -> list[list[float]]:
    """Random SPD 3x3 block with eigenvalues in [0.5, 3]."""
    a = [[rng.uniform(-1.0, 1.0) for _ in range(3)] for _ in range(3)]
    b = [[sum(a[r][k] * a[c][k] for k in range(3)) * 0.3 for c in range(3)] for r in range(3)]
    for r in range(3):
        b[r][r] += 0.5
    return b


class QuadraticSource:
    """g_c = (B z)_c / t_c with z = (m - m*)/t per cell and an SPD block B per cell.

    Pairs from this source have s.y = dz^T B dz > 0 (independently of sigma).
    """

    def __init__(self, seed: int, nxg: int = NXG, nyg: int = NYG) -> None:
        rng = random.Random(seed)
        self.rng = rng
        self.nxg, self.nyg = nxg, nyg
        self.t = CLASS_CENTRES
        self.blocks = [[_spd_block(rng) for _ in range(nxg)] for _ in range(nyg)]
        self.target = [_field(nyg, nxg, lambda j, i, c=c: self.t[c] * (1 + rng.uniform(-0.1, 0.1)))
                       for c in range(3)]

    def start(self):
        rng = self.rng
        return [_field(self.nyg, self.nxg,
                       lambda j, i, c=c: self.t[c] * (1 + rng.uniform(-0.15, 0.15)))
                for c in range(3)]

    def grad(self, base):
        out = [[[0.0] * self.nxg for _ in range(self.nyg)] for _ in range(3)]
        for j in range(self.nyg):
            for i in range(self.nxg):
                z = [(base[c][j][i] - self.target[c][j][i]) / self.t[c] for c in range(3)]
                b = self.blocks[j][i]
                for c in range(3):
                    out[c][j][i] = ref32.f32(sum(b[c][k] * z[k] for k in range(3)) / self.t[c])
        return out

    def move(self, base, scale: float = 0.01):
        rng = self.rng
        return [_field(self.nyg, self.nxg,
                       lambda j, i, c=c: base[c][j][i] * (1 + scale * rng.uniform(-1, 1)))
                for c in range(3)]


def _random_field(rng: random.Random, ny: int, nx: int, lo: float, hi: float):
    return _field(ny, nx, lambda j, i: rng.uniform(lo, hi))


def _sequence(seed: int, niter: int, *, weights=((1.0, 1.0, 1.0),), precond: bool = False,
              mask=None, nxg: int = NXG, nyg: int = NYG):
    """Stage + iterations from the quadratic source; weights cycle per iteration."""
    src = QuadraticSource(seed, nxg, nyg)
    rng = random.Random(seed + 1000)
    mask = mask if mask is not None else _frame_mask(nyg, nxg, 1)
    base = src.start()
    stage = (_copy3(base), mask)
    its = []
    for k in range(niter):
        it = {"kind": 0, "weights": tuple(weights[min(k, len(weights) - 1)]),
              "base": _copy3(base), "grad": src.grad(base),
              "step_in": [_random_field(rng, nyg, nxg, -1.0, 1.0) for _ in range(3)],
              "mask": mask}
        if precond:
            it["precond"] = _random_field(rng, nyg, nxg, 0.2, 5.0)
        its.append(it)
        base = src.move(base)
    return stage, its


def _reference(stage, iterations, memory: int, has_precond: bool):
    model = ref.LbfgsReference(memory)
    model.begin_stage(stage[0], stage[1])
    outs = []
    for it in iterations:
        if it.get("kind", 0) == 1:
            model.begin_stage(it["base"], it["mask"])
        outs.append(model.step(it["base"], it["grad"], it["mask"],
                               it["precond"] if has_precond else None, it["weights"],
                               it["step_in"]))
    return outs, model


def _assert_step_close(got, want, tol: float = STEP_TOLERANCE) -> None:
    for c in range(3):
        scale = max(abs(v) for row in want[c] for v in row)
        for j, row in enumerate(want[c]):
            for i, w in enumerate(row):
                g = got[c][j][i]
                if w == 0.0:
                    assert g == 0.0, (c, j, i, g)
                else:
                    assert abs(g - w) <= tol * scale, (c, j, i, g, w, scale)


def _assert_matches_reference(result, outs, iterations) -> None:
    for k, (got, want) in enumerate(zip(result["its"], outs)):
        assert got["status"] == 0, (k, got["status"])
        assert (got["action"], got["pair"], got["pairs_used"]) == \
            (want["action"], want["pair"], want["pairs_used"]), (k, got, want)
        for name in ("curvature", "gamma", "descent", "kappa"):
            assert _rel_close(got[name], want[name], SCALAR_TOLERANCE), (k, name, got[name],
                                                                         want[name])
        for a, b in zip(got["sigma"], want["sigma"]):
            assert _rel_close(a, b, SCALAR_TOLERANCE)
        assert got["halos"] and got["inputs_unchanged"]
        if want["action"] == 3:
            _assert_step_close(got["step"], want["step"])
        else:
            assert got["step_unchanged"], k
            assert got["step"] == [[list(r) for r in f] for f in iterations[k]["step_in"]]


# ---------------------------------------------------------------------------
# Contract identity and declarations
# ---------------------------------------------------------------------------

def test_contract_identity(repository_root: Path) -> None:
    digest = hashlib.sha256((repository_root / CONTRACT_PATH).read_bytes()).hexdigest()
    assert digest == CONTRACT_SHA256


def test_api_declared_in_header_and_makefile(repository_root: Path) -> None:
    header = (repository_root / "include/fd.h").read_text(encoding="utf-8", errors="replace")
    for name in ("visco_sh_exact_lbfgs_init", "visco_sh_exact_lbfgs_free",
                 "visco_sh_exact_lbfgs_begin_stage", "visco_sh_exact_lbfgs_step",
                 "visco_sh_exact_lbfgs_get_sigma",
                 "struct visco_sh_exact_lbfgs_request", "struct visco_sh_exact_lbfgs_report",
                 "VISCO_SH_EXACT_LBFGS_MAX_MEMORY"):
        assert name in header, name
    makefile = (repository_root / "src/Makefile").read_text(encoding="utf-8", errors="replace")
    assert "visco_sh_exact_lbfgs.c" in makefile


# ---------------------------------------------------------------------------
# O3 / O12: two-loop closed form and descent (unit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("memory,precond,weights", [
    (1, False, ((1.0, 1.0, 1.0),)),
    (3, True, ((1.0, 1.0, 1.0),)),
    (5, True, ((1.0, 0.5, 2.0),)),
    (5, False, ((1.0, 0.0, 1.0),)),
])
def test_o3_o12_two_loop_matches_reference(harness: Path, tmp_path: Path, memory: int,
                                           precond: bool, weights) -> None:
    stage, its = _sequence(11 + memory, 8, weights=weights, precond=precond)
    result = _run(harness, tmp_path, stage=stage, iterations=its, nxg=NXG, nyg=NYG,
                  memory=memory, has_precond=precond)
    outs, _ = _reference(stage, its, memory, precond)
    assert [o["action"] for o in outs][:2] == [0, 3]
    assert sum(o["action"] == 3 for o in outs) >= 6
    _assert_matches_reference(result, outs, its)
    # O12: descent sign of every action-3 output (subtractive convention).
    for got, want, it in zip(result["its"], outs, its):
        if want["action"] == 3:
            dot = math.fsum(float(it["grad"][c][j][i]) * float(got["step"][c][j][i])
                            for c in range(3) for j in range(NYG) for i in range(NXG)
                            if it["mask"][j][i] == 1.0 and it["weights"][c] > 0.0)
            assert dot > 0.0


def test_o3_sigma_is_masked_rms_of_stage_base(harness: Path, tmp_path: Path) -> None:
    stage, its = _sequence(5, 2)
    result = _run(harness, tmp_path, stage=stage, iterations=its, nxg=NXG, nyg=NYG)
    want = ref.sigma(stage[0], stage[1])
    for got in result["its"]:
        for a, b in zip(got["sigma"], want):
            assert _rel_close(a, b, SCALAR_TOLERANCE)


# ---------------------------------------------------------------------------
# O4: dense BFGS matrix (independent algorithm)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precond,weights", [(False, (1.0, 1.0, 1.0)),
                                             (True, (1.0, 0.25, 3.0)),
                                             (True, (0.0, 1.0, 1.0))])
def test_o4_direction_equals_dense_bfgs(harness: Path, tmp_path: Path, precond: bool,
                                        weights) -> None:
    nxg, nyg = 4, 3
    stage, its = _sequence(23, 5, weights=(weights,), precond=precond,
                           mask=[[1.0] * nxg for _ in range(nyg)], nxg=nxg, nyg=nyg)
    result = _run(harness, tmp_path, stage=stage, iterations=its, nxg=nxg, nyg=nyg,
                  memory=5, has_precond=precond)
    model = ref.LbfgsReference(5)
    sig = model.begin_stage(stage[0], stage[1])
    checked = 0
    for k, it in enumerate(its):
        out = model.step(it["base"], it["grad"], it["mask"],
                         it["precond"] if precond else None, it["weights"], it["step_in"])
        got = result["its"][k]
        assert got["action"] == out["action"]
        if out["action"] != 3:
            continue
        entries = out["entries"]
        gx = [sig[c] * float(it["grad"][c][j][i]) for (c, j, i) in entries]
        dd = [float(it["weights"][c]) * (float(it["precond"][j][i]) if precond else 1.0)
              for (c, j, i) in entries]
        d = ref.dense_direction(model.pairs if out["action"] == 3 else [], dd, gx)
        kappa = ref.kappa_of(d, sum(w > 0 for w in it["weights"]), nxg * nyg)
        want = ref.class_fields_from_entries(entries, d, sig, kappa, nxg, nyg)
        _assert_step_close(got["step"], want)
        checked += 1
    assert checked >= 3


# ---------------------------------------------------------------------------
# O5: secant condition
# ---------------------------------------------------------------------------

def test_o5_newest_pair_satisfies_secant(harness: Path, tmp_path: Path) -> None:
    nxg, nyg = 6, 4
    rng = random.Random(77)
    mask = [[1.0] * nxg for _ in range(nyg)]
    m1 = [_field(nyg, nxg, lambda j, i, c=c: CLASS_CENTRES[c] * (1 + rng.uniform(-0.1, 0.1)))
          for c in range(3)]
    sig = ref.sigma(m1, mask)
    s_t = [[[rng.uniform(-0.01, 0.01) for _ in range(nxg)] for _ in range(nyg)]
           for _ in range(3)]
    y_t = [[[s_t[c][j][i] * (1.0 + 0.2 * rng.uniform(-1, 1)) for i in range(nxg)]
            for j in range(nyg)] for c in range(3)]
    m2 = [_field(nyg, nxg, lambda j, i, c=c: m1[c][j][i] + sig[c] * s_t[c][j][i])
          for c in range(3)]
    g1 = [_field(nyg, nxg, lambda j, i, c=c: y_t[c][j][i] / sig[c]) for c in range(3)]
    g2 = [_field(nyg, nxg, lambda j, i, c=c: 2.0 * g1[c][j][i]) for c in range(3)]
    zeros = [_const(nyg, nxg, 0.0) for _ in range(3)]
    its = [{"weights": (1.0, 1.0, 1.0), "base": m1, "grad": g1, "step_in": zeros,
            "mask": mask},
           {"weights": (1.0, 1.0, 1.0), "base": m2, "grad": g2, "step_in": zeros,
            "mask": mask}]
    result = _run(harness, tmp_path, stage=(m1, mask), iterations=its, nxg=nxg, nyg=nyg)
    got = result["its"][1]
    assert (got["action"], got["pair"], got["pairs_used"]) == (3, 1, 1)
    # Actual stored s from the binary32 inputs (contract 3.3 step 2).
    s = [[[ref32.f32((float(m2[c][j][i]) - float(m1[c][j][i])) / sig[c])
           for i in range(nxg)] for j in range(nyg)] for c in range(3)]
    px = [[[got["step"][c][j][i] / sig[c] for i in range(nxg)] for j in range(nyg)]
          for c in range(3)]
    norm = lambda f: math.sqrt(math.fsum(v * v for c in f for row in c for v in row))
    sn, pn = norm(s), norm(px)
    worst = max(abs(px[c][j][i] / pn - s[c][j][i] / sn)
                for c in range(3) for j in range(nyg) for i in range(nxg))
    scale = max(abs(s[c][j][i] / sn) for c in range(3) for j in range(nyg) for i in range(nxg))
    assert worst <= STEP_TOLERANCE * scale, (worst, scale)


# ---------------------------------------------------------------------------
# O6: quadratic finite termination (H = A^-1 after n conjugate pairs)
# ---------------------------------------------------------------------------

def _orthonormal(rng: random.Random, n: int) -> list[list[float]]:
    vectors = []
    while len(vectors) < n:
        v = [rng.gauss(0.0, 1.0) for _ in range(n)]
        for u in vectors:
            d = math.fsum(a * b for a, b in zip(u, v))
            v = [a - d * b for a, b in zip(v, u)]
        nv = math.sqrt(math.fsum(a * a for a in v))
        if nv > 1e-3:
            vectors.append([a / nv for a in v])
    return vectors


def test_o6_quadratic_twelve_conjugate_pairs_give_inverse(harness: Path,
                                                          tmp_path: Path) -> None:
    nxg = nyg = 2
    n = 3 * nxg * nyg
    rng = random.Random(4242)
    q = _orthonormal(rng, n)
    lam = [1.0 + 3.0 * k / (n - 1) for k in range(n)]           # condition number 4
    a = [[math.fsum(q[k][r] * lam[k] * q[k][c] for k in range(n)) for c in range(n)]
         for r in range(n)]
    # s_i = 0.01 * A^-1/2 q_i (A-conjugate), y_i = A s_i.
    inv_sqrt = [[math.fsum(q[k][r] * q[k][c] / math.sqrt(lam[k]) for k in range(n))
                 for c in range(n)] for r in range(n)]
    mask = [[1.0] * nxg for _ in range(nyg)]
    # Entry order of the contract: class-major, then row j, then column i.
    order = [(c, j, i) for c in range(3) for j in range(nyg) for i in range(nxg)]
    m = [_field(nyg, nxg, lambda j, i, c=c: CLASS_CENTRES[c]) for c in range(3)]
    sig = ref.sigma(m, mask)
    g = [_field(nyg, nxg, lambda j, i: rng.uniform(-1e-5, 1e-5)) for _ in range(3)]
    zeros = [_const(nyg, nxg, 0.0) for _ in range(3)]
    its, bases, grads = [], [m], [g]
    for k in range(n):
        s_vec = [0.01 * math.fsum(inv_sqrt[r][c] * q[k][c] for c in range(n)) for r in range(n)]
        mb, gb = _copy3(bases[-1]), _copy3(grads[-1])
        m_new = _copy3(mb)
        for (c, j, i), sv in zip(order, s_vec):
            m_new[c][j][i] = ref32.f32(mb[c][j][i] + sig[c] * sv)
        # Gradient update from the actual binary32 step (y = A s_actual).
        s_act = [(float(m_new[c][j][i]) - float(mb[c][j][i])) / sig[c] for (c, j, i) in order]
        y_vec = [math.fsum(a[r][cc] * s_act[cc] for cc in range(n)) for r in range(n)]
        g_new = _copy3(gb)
        for (c, j, i), yv in zip(order, y_vec):
            g_new[c][j][i] = ref32.f32(gb[c][j][i] + yv / sig[c])
        bases.append(m_new)
        grads.append(g_new)
    for mb, gb in zip(bases, grads):
        its.append({"weights": (1.0, 1.0, 1.0), "base": mb, "grad": gb, "step_in": zeros,
                    "mask": mask})
    result = _run(harness, tmp_path, stage=(m, mask), iterations=its, nxg=nxg, nyg=nyg,
                  memory=n)
    final = result["its"][-1]
    assert final["action"] == 3 and final["pairs_used"] == n
    assert all(r["pair"] == 1 for r in result["its"][1:])
    gx = [sig[c] * float(grads[-1][c][j][i]) for (c, j, i) in order]
    want = ref.solve_linear(a, gx)
    px = [final["step"][c][j][i] / sig[c] for (c, j, i) in order]
    pn = math.sqrt(math.fsum(v * v for v in px))
    wn = math.sqrt(math.fsum(v * v for v in want))
    worst = max(abs(x / pn - w / wn) for x, w in zip(px, want))
    assert worst <= QUADRATIC_TOLERANCE * max(abs(w / wn) for w in want), worst


# ---------------------------------------------------------------------------
# O7: safeguards (unit)
# ---------------------------------------------------------------------------

def _it(base, grad, mask, weights=(1.0, 1.0, 1.0), step_in=None, kind=0, precond=None):
    ny, nx = len(mask), len(mask[0])
    rng = random.Random(len(base[0][0]) * 31 + int(abs(grad[0][1][1]) * 1e9) % 997)
    it = {"kind": kind, "weights": weights, "base": base, "grad": grad, "mask": mask,
          "step_in": step_in or [_random_field(rng, ny, nx, -1, 1) for _ in range(3)]}
    if precond is not None:
        it["precond"] = precond
    return it


def test_o7_negative_curvature_and_zero_step_are_skipped(harness: Path,
                                                         tmp_path: Path) -> None:
    src = QuadraticSource(31)
    mask = _frame_mask(NYG, NXG, 1)
    m1 = src.start(); m2 = src.move(m1); m3 = src.move(m2)
    g1, g2 = src.grad(m1), src.grad(m2)
    g3_good = src.grad(m3)
    # Negative curvature at iteration 3: reverse the gradient change.
    g3 = [_field(NYG, NXG, lambda j, i, c=c: 2 * g2[c][j][i] - g3_good[c][j][i])
          for c in range(3)]
    g4 = [_field(NYG, NXG, lambda j, i, c=c: g3[c][j][i] * 0.5) for c in range(3)]
    its = [_it(m1, g1, mask), _it(m2, g2, mask), _it(m3, g3, mask), _it(m3, g4, mask)]
    result = _run(harness, tmp_path, stage=(m1, mask), iterations=its, nxg=NXG, nyg=NYG)
    flags = [(r["action"], r["pair"], r["pairs_used"]) for r in result["its"]]
    assert flags[0] == (0, 0, 0)
    assert flags[1] == (3, 1, 1)
    assert flags[2][1:] == (2, 1)            # negative curvature skipped, memory kept
    assert result["its"][2]["curvature"] < 0.0
    assert flags[3][1:] == (2, 1)            # s == 0 skipped
    assert result["its"][3]["curvature"] == 0.0
    outs, _ = _reference((m1, mask), its, 5, False)
    _assert_matches_reference(result, outs, its)


def test_o7_active_set_change_resets_memory(harness: Path, tmp_path: Path) -> None:
    stage, its = _sequence(41, 6, weights=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                           (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    result = _run(harness, tmp_path, stage=stage, iterations=its, nxg=NXG, nyg=NYG)
    flags = [(r["action"], r["pair"], r["pairs_used"]) for r in result["its"]]
    assert flags[:5] == [(0, 0, 0), (3, 1, 1), (3, 1, 2), (1, 0, 0), (3, 1, 1)]
    assert all(result["its"][k]["step_unchanged"] for k in (0, 3))
    # Inactive classes are exactly zero in L-BFGS output (O8 as well).
    for k in (1, 2):
        for c in (1, 2):
            assert all(v == 0.0 for row in result["its"][k]["step"][c] for v in row)
    outs, _ = _reference(stage, its, 5, False)
    _assert_matches_reference(result, outs, its)


def test_o7_zero_gradient_falls_back_and_clears_memory(harness: Path,
                                                       tmp_path: Path) -> None:
    src = QuadraticSource(51)
    mask = _frame_mask(NYG, NXG, 1)
    m1 = src.start(); g1 = src.grad(m1)
    m2 = src.move(m1); g2 = src.grad(m2)
    sig = ref.sigma(m1, mask)
    # m3 = m2 - t * sigma^2 * g2 gives s = -t sigma g2 and, with g3 = 0, y = -sigma g2.
    t = 1.0e-3
    m3 = [_field(NYG, NXG, lambda j, i, c=c: m2[c][j][i] - t * sig[c] ** 2 * g2[c][j][i])
          for c in range(3)]
    g3 = [_const(NYG, NXG, 0.0) for _ in range(3)]
    m4 = src.move(m3); g4 = src.grad(m4)
    its = [_it(m1, g1, mask), _it(m2, g2, mask), _it(m3, g3, mask), _it(m4, g4, mask)]
    result = _run(harness, tmp_path, stage=(m1, mask), iterations=its, nxg=NXG, nyg=NYG)
    flags = [(r["action"], r["pair"], r["pairs_used"]) for r in result["its"]]
    assert flags[2] == (2, 1, 2)                 # stored, then fallback with 2 pairs
    assert result["its"][2]["step_unchanged"]
    assert result["its"][2]["descent"] == 0.0
    assert flags[3] == (3, 1, 1)                 # memory was cleared
    outs, _ = _reference((m1, mask), its, 5, False)
    _assert_matches_reference(result, outs, its)


def test_o7_ring_buffer_keeps_newest_pairs(harness: Path, tmp_path: Path) -> None:
    stage, its = _sequence(61, 5, precond=True)
    result = _run(harness, tmp_path, stage=stage, iterations=its, nxg=NXG, nyg=NYG,
                  memory=2, has_precond=True)
    assert [r["pairs_used"] for r in result["its"]] == [0, 1, 2, 2, 2]
    outs, _ = _reference(stage, its, 2, True)
    _assert_matches_reference(result, outs, its)


def test_o7_begin_stage_clears_memory(harness: Path, tmp_path: Path) -> None:
    stage, its = _sequence(71, 5)
    its[3]["kind"] = 1
    result = _run(harness, tmp_path, stage=stage, iterations=its, nxg=NXG, nyg=NYG)
    flags = [(r["action"], r["pair"], r["pairs_used"]) for r in result["its"]]
    assert flags[3] == (0, 0, 0) and result["its"][3]["begin"] == 0
    assert flags[4] == (3, 1, 1)
    want = ref.sigma(its[3]["base"], its[3]["mask"])
    for a, b in zip(result["its"][4]["sigma"], want):
        assert _rel_close(a, b, SCALAR_TOLERANCE)
    outs, _ = _reference(stage, its, 5, False)
    _assert_matches_reference(result, outs, its)


def test_o7_curvature_threshold_is_discriminated(harness: Path, tmp_path: Path) -> None:
    """RC-5: y == 0 skipped (curvature 0); cos ~ 2e-9 skipped; cos ~ 1e-6 stored."""
    rng = random.Random(181)
    mask = _frame_mask(NYG, NXG, 1)
    cells = [(j, i) for j in range(NYG) for i in range(NXG) if mask[j][i] == 1.0]
    star = (2, 5)
    m = [QuadraticSource(181).start()]
    for _ in range(3):                     # only class 0 moves: s is zero on rho and q
        nxt = _copy3(m[-1])
        for (j, i) in cells:
            nxt[0][j][i] = ref32.f32(m[-1][0][j][i] * (1 + 0.01 * rng.uniform(0.5, 1.0)))
        m.append(nxt)
    sig = ref.sigma(m[0], mask)

    def s_vec(a, b):
        return {(j, i): ref32.f32((float(b[0][j][i]) - float(a[0][j][i])) / sig[0])
                for (j, i) in cells}

    def with_pattern(g_old, s_now, cosine):
        """y on class q (orthogonal to s) plus one tiny class-0 entry giving cos."""
        g_new = _copy3(g_old)
        yq = {c: rng.uniform(0.5, 1.0) * 1e-2 for c in cells}
        ynorm = math.sqrt(math.fsum(v * v for v in yq.values()))
        snorm = math.sqrt(math.fsum(v * v for v in s_now.values()))
        for (j, i), v in yq.items():
            g_new[2][j][i] = ref32.f32(g_old[2][j][i] + v / sig[2])
        y_star = cosine * snorm * ynorm / s_now[star]
        g_new[0][star[0]][star[1]] = ref32.f32(g_old[0][star[0]][star[1]] + y_star / sig[0])
        return g_new

    g1 = [_const(NYG, NXG, 0.0) for _ in range(3)]
    g2 = _copy3(g1)                                        # y == 0
    g3 = with_pattern(g2, s_vec(m[1], m[2]), 2.0e-9)
    g4 = with_pattern(g3, s_vec(m[2], m[3]), 1.0e-6)
    its = [_it(m[k], g, mask) for k, g in enumerate((g1, g2, g3, g4))]
    outs, _ = _reference((m[0], mask), its, 5, False)
    # Preconditions of the construction, from the reference (binary32 pairs).
    assert outs[1]["curvature"] == 0.0
    assert 1.0e-9 < outs[2]["curvature"] < 5.0e-9, outs[2]["curvature"]
    assert 1.0e-7 < outs[3]["curvature"] < 1.0e-5, outs[3]["curvature"]
    result = _run(harness, tmp_path, stage=(m[0], mask), iterations=its, nxg=NXG, nyg=NYG)
    flags = [(r["action"], r["pair"], r["pairs_used"]) for r in result["its"]]
    assert flags == [(0, 0, 0), (0, 2, 0), (0, 2, 0), (3, 1, 1)]
    assert result["its"][1]["curvature"] == 0.0
    _assert_matches_reference(result, outs, its)


def test_o11_begin_stage_ignores_unmasked_nan(harness: Path, tmp_path: Path) -> None:
    stage, its = _sequence(191, 3)
    bases, mask = stage
    dirty = _copy3(bases)
    for j in range(NYG):
        for i in range(NXG):
            if mask[j][i] == 0.0:
                for c in range(3):
                    dirty[c][j][i] = NAN
    a = _run(harness, tmp_path / "clean", stage=stage, iterations=its, nxg=NXG, nyg=NYG)
    b = _run(harness, tmp_path / "dirty", stage=(dirty, mask), iterations=its, nxg=NXG,
             nyg=NYG)
    assert b["begin"] == 0
    for ra, rb in zip(a["its"], b["its"]):
        assert ra["sigma"] == rb["sigma"]
        assert ra["step"] == rb["step"]


# ---------------------------------------------------------------------------
# O8: projection
# ---------------------------------------------------------------------------

def test_o8_unmasked_and_inactive_values_have_no_influence(harness: Path,
                                                           tmp_path: Path) -> None:
    weights = ((1.0, 0.0, 1.0),)
    stage, clean = _sequence(81, 5, weights=weights, precond=True,
                             mask=_frame_mask(NYG, NXG, 2))
    dirty = []
    for it in clean:
        d = {k: (_copy3(v) if k in ("base", "grad") else v) for k, v in it.items()}
        d["precond"] = [list(r) for r in it["precond"]]
        for j in range(NYG):
            for i in range(NXG):
                if it["mask"][j][i] == 0.0:
                    for c in range(3):
                        d["base"][c][j][i] = NAN
                        d["grad"][c][j][i] = INF
                    d["precond"][j][i] = NAN
                d["grad"][1][j][i] = NAN            # inactive class gradient
        dirty.append(d)
    a = _run(harness, tmp_path / "clean", stage=stage, iterations=clean, nxg=NXG, nyg=NYG,
             has_precond=True)
    b = _run(harness, tmp_path / "dirty", stage=stage, iterations=dirty, nxg=NXG, nyg=NYG,
             has_precond=True)
    for ra, rb in zip(a["its"], b["its"]):
        assert rb["status"] == 0
        assert (ra["action"], ra["pair"], ra["pairs_used"]) == \
            (rb["action"], rb["pair"], rb["pairs_used"])
        assert ra["step"] == rb["step"]
        if ra["action"] == 3:
            for c in range(3):
                for j in range(NYG):
                    for i in range(NXG):
                        if c == 1 or clean[0]["mask"][j][i] == 0.0:
                            assert ra["step"][c][j][i] == 0.0


# ---------------------------------------------------------------------------
# O9: reparametrisation invariance
# ---------------------------------------------------------------------------

def test_o9_class_reparametrisation_scales_only_that_class(harness: Path,
                                                          tmp_path: Path) -> None:
    stage, its = _sequence(91, 5, precond=True)
    scale = 1024.0
    scaled_stage = (_copy3(stage[0]), stage[1])
    scaled_stage[0][1] = [[v * scale for v in row] for row in stage[0][1]]
    scaled = []
    for it in its:
        s = dict(it)
        s["base"] = _copy3(it["base"]); s["grad"] = _copy3(it["grad"])
        s["base"][1] = [[v * scale for v in row] for row in it["base"][1]]
        s["grad"][1] = [[v / scale for v in row] for row in it["grad"][1]]
        scaled.append(s)
    a = _run(harness, tmp_path / "a", stage=stage, iterations=its, nxg=NXG, nyg=NYG,
             has_precond=True)
    b = _run(harness, tmp_path / "b", stage=scaled_stage, iterations=scaled, nxg=NXG,
             nyg=NYG, has_precond=True)
    checked = 0
    for ra, rb in zip(a["its"], b["its"]):
        assert (ra["action"], ra["pair"], ra["pairs_used"]) == \
            (rb["action"], rb["pair"], rb["pairs_used"])
        if ra["action"] != 3:
            continue
        unscaled = [rb["step"][0], [[v / scale for v in row] for row in rb["step"][1]],
                    rb["step"][2]]
        _assert_step_close(unscaled, ra["step"])
        checked += 1
    assert checked >= 3


# ---------------------------------------------------------------------------
# O10: MPI decompositions (unit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nprocx,nprocy", [(2, 1), (1, 2), (3, 2)])
def test_o10_decompositions_agree(harness: Path, tmp_path: Path, nprocx: int,
                                  nprocy: int) -> None:
    stage, its = _sequence(101, 6, weights=((1.0, 0.5, 1.0),), precond=True)
    serial = _run(harness, tmp_path / "serial", stage=stage, iterations=its, nxg=NXG,
                  nyg=NYG, has_precond=True)
    split = _run(harness, tmp_path / "split", stage=stage, iterations=its, nxg=NXG,
                 nyg=NYG, nprocx=nprocx, nprocy=nprocy, has_precond=True)
    for a, b in zip(serial["its"], split["its"]):
        assert (a["action"], a["pair"], a["pairs_used"]) == \
            (b["action"], b["pair"], b["pairs_used"])
        for name in ("curvature", "gamma", "descent", "kappa"):
            assert _rel_close(a[name], b[name], SCALAR_TOLERANCE)
        assert b["halos"]
        if a["action"] == 3:
            _assert_step_close(b["step"], a["step"])
        else:
            assert a["step"] == b["step"]
    outs, _ = _reference(stage, its, 5, True)
    _assert_matches_reference(split, outs, its)


# ---------------------------------------------------------------------------
# O11: error paths (unit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nx_fault,memory,want", [(False, 0, -31), (False, 21, -31),
                                                  (False, 20, 0), (False, 1, 0),
                                                  (True, 5, -31)])
def test_o11_init_errors_are_collective(harness: Path, tmp_path: Path, nx_fault: bool,
                                        memory: int, want: int) -> None:
    result = _run(harness, tmp_path, stage=None, iterations=[], nxg=4, nyg=2, nprocx=2,
                  memory=memory, fault=2 if nx_fault else 0, fault_rank=1, op=2)
    assert result["init"] == want


def test_o11_init_null_state(harness: Path, tmp_path: Path) -> None:
    result = _run(harness, tmp_path, stage=None, iterations=[], nxg=4, nyg=2, memory=5,
                  fault=1, op=2)
    assert result["init"] == -31


def test_o11_failed_init_makes_later_calls_fail(harness: Path, tmp_path: Path) -> None:
    stage, its = _sequence(111, 1)
    result = _run(harness, tmp_path, stage=stage, iterations=its, nxg=NXG, nyg=NYG,
                  memory=0)
    assert result["init"] == -31 and result["begin"] == -31
    assert result["its"][0]["status"] == -31


STEP_FAULTS = {1: -31, 2: -31, 3: -31, 4: -31, 5: -33, 6: -33, 7: -33, 8: -33, 9: -33,
               10: -33, 11: -31}


def _assert_rejected(r, code: int) -> None:
    assert r["status"] == code, r["status"]
    assert r["step_unchanged"] and r["report_untouched"] and r["inputs_unchanged"]


@pytest.mark.parametrize("fault", sorted(STEP_FAULTS))
@pytest.mark.parametrize("ranks", [1, 2])
def test_o11_step_structural_errors_are_collective_and_stateless(
        harness: Path, tmp_path: Path, fault: int, ranks: int) -> None:
    stage, clean = _sequence(121, 4, precond=True)
    faulty = clean[:2] + [clean[2]] + clean[2:]      # iteration 2 is the failing call
    result = _run(harness, tmp_path / "faulty", stage=stage, iterations=faulty, nxg=NXG,
                  nyg=NYG, nprocx=ranks, has_precond=True, fault=fault,
                  fault_rank=ranks - 1, fault_iter=2)
    _assert_rejected(result["its"][2], STEP_FAULTS[fault])
    reference = _run(harness, tmp_path / "clean", stage=stage, iterations=clean, nxg=NXG,
                     nyg=NYG, nprocx=ranks, has_precond=True)
    # State unchanged: the following calls equal the run without the failed call.
    for a, b in zip(result["its"][3:], reference["its"][2:]):
        assert (a["action"], a["pair"], a["pairs_used"]) == \
            (b["action"], b["pair"], b["pairs_used"])
        assert a["step"] == b["step"]


def _corrupt(it, *, rank_cell=(1, 10), **changes):
    out = dict(it)
    for key in ("base", "grad"):
        out[key] = _copy3(it[key])
    out["mask"] = [list(r) for r in it["mask"]]
    if "precond" in it:
        out["precond"] = [list(r) for r in it["precond"]]
    j, i = rank_cell                                   # a masked cell owned by rank 1 of 2
    for name, value in changes.items():
        if name == "weights":
            out["weights"] = value
        elif name == "mask":
            out["mask"][j][i] = value
        elif name == "precond":
            out["precond"][j][i] = value
        else:
            key, c = name.rsplit("_", 1)
            out[key][int(c)][j][i] = value
    return out


DATA_FAULTS = {
    "weight_nan": ({"weights": (1.0, NAN, 1.0)}, -34),
    "weight_negative": ({"weights": (1.0, -0.5, 1.0)}, -34),
    "weight_inf": ({"weights": (INF, 1.0, 1.0)}, -34),
    "mask_half": ({"mask": 0.5}, -34),
    "base_zero": ({"base_1": 0.0}, -34),
    "base_negative_q": ({"base_2": -1.0}, -34),
    "base_nan": ({"base_0": NAN}, -34),
    "grad_inf_active": ({"grad_0": INF}, -34),
    "grad_nan_active": ({"grad_2": NAN}, -34),
    "precond_zero": ({"precond": 0.0}, -34),
    "precond_nan": ({"precond": NAN}, -34),
    "no_active_class": ({"weights": (0.0, 0.0, 0.0)}, -36),
    "mask_and_no_active": ({"weights": (0.0, 0.0, 0.0), "mask": 2.0}, -34),
}


@pytest.mark.parametrize("name", sorted(DATA_FAULTS))
def test_o11_step_data_errors_on_one_rank(harness: Path, tmp_path: Path, name: str) -> None:
    changes, code = DATA_FAULTS[name]
    stage, clean = _sequence(131, 4, precond=True)
    bad = _corrupt(clean[2], **changes)
    faulty = clean[:2] + [bad] + clean[2:]
    result = _run(harness, tmp_path / "faulty", stage=stage, iterations=faulty, nxg=NXG,
                  nyg=NYG, nprocx=2, has_precond=True)
    _assert_rejected(result["its"][2], code)
    reference = _run(harness, tmp_path / "clean", stage=stage, iterations=clean, nxg=NXG,
                     nyg=NYG, nprocx=2, has_precond=True)
    for a, b in zip(result["its"][3:], reference["its"][2:]):
        assert (a["action"], a["pair"], a["pairs_used"]) == \
            (b["action"], b["pair"], b["pairs_used"])
        assert a["step"] == b["step"]


@pytest.mark.parametrize("fault,changes,code", [
    (5, {"weights": (1.0, NAN, 1.0)}, -33),           # -33 before -34
    (4, {"mask": 0.5}, -31),                          # -31 before -34
    (9, {"weights": (0.0, 0.0, 0.0)}, -33),           # -33 before -36
])
def test_o11_step_precedence(harness: Path, tmp_path: Path, fault: int, changes,
                             code: int) -> None:
    stage, clean = _sequence(141, 3, precond=True)
    faulty = clean[:2] + [_corrupt(clean[2], **changes)]
    result = _run(harness, tmp_path, stage=stage, iterations=faulty, nxg=NXG, nyg=NYG,
                  nprocx=2, has_precond=True, fault=fault, fault_rank=0, fault_iter=2)
    _assert_rejected(result["its"][2], code)


def test_o11_step_before_begin_stage(harness: Path, tmp_path: Path) -> None:
    stage, its = _sequence(151, 1)
    result = _run(harness, tmp_path, stage=stage, iterations=its, nxg=NXG, nyg=NYG,
                  skip_begin=1)
    assert result["init"] == 0
    _assert_rejected(result["its"][0], -31)


@pytest.mark.parametrize("fault,code", [(101, -31), (102, -33), (103, -33)])
def test_o11_begin_stage_structural_errors_keep_state(harness: Path, tmp_path: Path,
                                                      fault: int, code: int) -> None:
    stage, its = _sequence(161, 5)
    its_kind = [dict(it) for it in its]
    its_kind[3]["kind"] = 1
    result = _run(harness, tmp_path / "faulty", stage=stage, iterations=its_kind, nxg=NXG,
                  nyg=NYG, nprocx=2, fault=fault, fault_rank=1, fault_iter=3)
    assert result["its"][3]["begin"] == code
    clean = _run(harness, tmp_path / "clean", stage=stage, iterations=its, nxg=NXG,
                 nyg=NYG, nprocx=2)
    for a, b in zip(result["its"][3:], clean["its"][3:]):
        assert a["status"] == 0
        assert (a["action"], a["pair"], a["pairs_used"]) == \
            (b["action"], b["pair"], b["pairs_used"])
        assert a["step"] == b["step"]


@pytest.mark.parametrize("change,code", [("mask_invalid", -34), ("base_zero", -34),
                                         ("base_inf", -34), ("empty_mask", -35)])
def test_o11_begin_stage_data_errors(harness: Path, tmp_path: Path, change: str,
                                     code: int) -> None:
    stage, its = _sequence(171, 4)
    bad = dict(its[2])
    bad["kind"] = 2
    bad["base"] = _copy3(its[2]["base"])
    bad["mask"] = [list(r) for r in its[2]["mask"]]
    if change == "mask_invalid":
        bad["mask"][2][9] = 0.25
    elif change == "base_zero":
        bad["base"][0][2][9] = 0.0
    elif change == "base_inf":
        bad["base"][2][2][9] = INF
    else:
        bad["mask"] = [[0.0] * NXG for _ in range(NYG)]
    faulty = its[:2] + [bad] + its[2:]
    result = _run(harness, tmp_path / "faulty", stage=stage, iterations=faulty, nxg=NXG,
                  nyg=NYG, nprocx=2)
    assert result["its"][2]["begin"] == code
    clean = _run(harness, tmp_path / "clean", stage=stage, iterations=its, nxg=NXG,
                 nyg=NYG, nprocx=2)
    for a, b in zip(result["its"][3:], clean["its"][2:]):
        assert (a["action"], a["pair"], a["pairs_used"]) == \
            (b["action"], b["pair"], b["pairs_used"])
        assert a["step"] == b["step"]


# ===========================================================================
# Driver level (32x32 SH-PRECOND-01 fixture)
# ===========================================================================

DRIVER_N = 32
DRIVER_CELLS = DRIVER_N * DRIVER_N
DRIVER_FW = 6
SCALED_EPS = 0.01
WORKFLOW_HEADER = (
    "PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ INV_VP_ITER "
    "INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP WD_DAMP1 EPRECOND "
    "LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE OFFSET_MUTE OFFSETC "
    "SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER"
)
TRAILING_RECORDS = (
    ("MIGRATION_SOURCE_PREFIX", "migration/unused_source"),
    ("MIGRATION_DATA_PREFIX", "migration/unused_data"),
    ("MIGRATION_IMAGE_PREFIX", "migration/unused_image"),
    ("OPT_STEP_SCALING", None), ("OPT_WEIGHT_PRIMARY", "1.0"), ("OPT_WEIGHT_RHO", "1.0"),
    ("OPT_WEIGHT_Q", "1.0"), ("OPT_DIAG_OUT", None),
    ("OPT_PRECOND", None), ("OPT_PRECOND_EPS", None), ("OPT_ALPHA_ADAPT", None),
    ("OPT_ALPHA_ADAPT_FACTOR", None), ("OPT_LBFGS", None), ("OPT_LBFGS_M", None),
)
MODEL_SUFFIXES = ("vs", "rho", "qs")
DIAG_SUFFIXES = ("grad_vs", "grad_rho", "grad_qs", "step_vs", "step_rho", "step_qs")
PRECOND_SUFFIXES = ("energy", "precond")
LOG_PATTERN = re.compile(
    r"Exact SH L-BFGS: stage (\d+) iteration (\d+) action (\w+) pair (\w+) pairs (\d+) "
    r"curvature (\S+) gamma (\S+) descent (\S+) kappa (\S+)")
SIGMA_PATTERN = re.compile(r"Exact SH L-BFGS: stage (\d+) sigma (\S+) (\S+) (\S+)")


def _write_parameter(path: Path, *, records: int, scaling: object = 0, diag: object = 0,
                     precond: object = 0, precond_eps: object = 0.005, adapt: object = 0,
                     factor: object = 2.0, lbfgs: object = 0, lbfgs_m: object = 5,
                     **fixture: object) -> None:
    """Records 1-119 via the frozen fixture, then trailing records 120..``records``."""
    assert 127 <= records <= 133
    write_sh_parameter(path, **fixture)
    text = path.read_text(encoding="ascii")
    values = {"OPT_STEP_SCALING": scaling, "OPT_DIAG_OUT": diag, "OPT_PRECOND": precond,
              "OPT_PRECOND_EPS": precond_eps, "OPT_ALPHA_ADAPT": adapt,
              "OPT_ALPHA_ADAPT_FACTOR": factor, "OPT_LBFGS": lbfgs, "OPT_LBFGS_M": lbfgs_m}
    extra = []
    for name, default in TRAILING_RECORDS[:records - 119]:
        extra.append(f"({name}) = {values.get(name, default) if default is None else default}\n")
    path.write_text(text + "".join(extra), encoding="ascii", newline="\n")


def _workflow_row(pro: float = 0.01, inv_rho: int = 0, inv_qs: int = 0) -> str:
    values = [pro, 0, 0, 20, 2, 0, 1, 0, 0, 0, 0, inv_rho, inv_qs,
              0, 0.5, 0.5, 0, 2, 0, 0, 0, 0.1, 0, 0, 10, 1, 1, 0, 0, 0]
    return " ".join(str(v) for v in values)


def _write_workflow(path: Path, rows: list[str]) -> None:
    path.write_text(WORKFLOW_HEADER + "\n" + "\n".join(rows) + "\n", encoding="ascii")


def _write_native_model(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"={DRIVER_CELLS}f", *values))


def _read_native(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    assert len(raw) == 4 * DRIVER_CELLS, path
    return struct.unpack(f"={DRIVER_CELLS}f", raw)


def _to_grid(values) -> list[list[float]]:
    """Merged model files are x-major: value(x, y) = values[x * NY + y]."""
    return [[values[x * DRIVER_N + y] for x in range(DRIVER_N)] for y in range(DRIVER_N)]


def _run_denise(denise_binary: Path, mpiexec: str, root: Path, parameter: Path,
                workflow: Path, ranks: int = 1) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [mpiexec, "--oversubscribe", "-n", str(ranks), str(denise_binary), str(parameter),
         str(workflow)], cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=900)


@pytest.fixture(scope="module")
def driver_case(tmp_path_factory: pytest.TempPathFactory, denise_binary: Path,
                mpiexec: str) -> dict:
    """32x32 viscoelastic SH case: Vs 2000 with a slow patch, Q=80, observed data."""
    root = tmp_path_factory.mktemp("sh_lbfgs_case")
    for name in ("model", "source", "receiver", "su", "log"):
        (root / name).mkdir(parents=True, exist_ok=True)
    true_vs, start_vs = [], []
    for x in range(DRIVER_N):
        for y in range(DRIVER_N):
            true_vs.append(1900.0 if 18 <= x <= 22 and 18 <= y <= 22 else 2000.0)
            start_vs.append(2000.0)
    for prefix, vs in ((root / "model/true", true_vs), (root / "model/start", start_vs)):
        _write_native_model(prefix.with_suffix(".vs"), vs)
        _write_native_model(prefix.with_suffix(".rho"), [2000.0] * DRIVER_CELLS)
        _write_native_model(prefix.with_suffix(".qs"), [80.0] * DRIVER_CELLS)
    # INVMAT1 = 3 start model (mu = rho * vs^2) for the rev-3 guard oracles.
    mu_prefix = root / "model/start_mu"
    _write_native_model(mu_prefix.with_suffix(".mu"), [2000.0 * 2000.0 ** 2] * DRIVER_CELLS)
    _write_native_model(mu_prefix.with_suffix(".vs"), start_vs)
    _write_native_model(mu_prefix.with_suffix(".rho"), [2000.0] * DRIVER_CELLS)
    _write_native_model(mu_prefix.with_suffix(".qs"), [80.0] * DRIVER_CELLS)
    (root / "source/one.dat").write_text("1\n160 0 160 0 20 1 0 1\n", encoding="ascii")
    (root / "receiver/line.dat").write_text(
        "100 160\n120 160\n140 160\n180 160\n200 160\n220 160\n", encoding="ascii")
    workflows = {"one": root / "workflow.inp", "delayed": root / "workflow_delayed.inp",
                 "two": root / "workflow_two.inp", "nostop": root / "workflow_nostop.inp"}
    _write_workflow(workflows["one"], [_workflow_row()])
    _write_workflow(workflows["delayed"], [_workflow_row(pro=0.0, inv_rho=3, inv_qs=3)])
    _write_workflow(workflows["nostop"], [_workflow_row(pro=0.0)])
    _write_workflow(workflows["two"], [_workflow_row(), _workflow_row()])
    forward = root / "true_forward.inp"
    write_sh_parameter(forward)
    run = _run_denise(denise_binary, mpiexec, root, forward, workflows["one"])
    assert run.returncode == 0, run.stdout
    start = root / "model/start"
    return {"root": root, "workflows": workflows, "start": start,
            "start_bytes": {s: start.with_suffix(s).read_bytes()
                            for s in (".vs", ".rho", ".qs")},
            "binary": denise_binary, "mpiexec": mpiexec}


def _fwi(case: dict, name: str, *, records: int, itermax: int = 1, nprocx: int = 1,
         nprocy: int = 1, model_prefix: str = "model/start", eps_scale: float = SCALED_EPS,
         workflow: str = "one", **options) -> tuple[subprocess.CompletedProcess[str], str]:
    output = f"model/{name}"
    parameter = case["root"] / f"{name}.inp"
    _write_parameter(parameter, records=records, mode=1, model_prefix=model_prefix,
                     grad_method=0, inv_mod_out=1, inv_model_file=output,
                     itermax=itermax, nprocx=nprocx, nprocy=nprocy, eps_scale=eps_scale,
                     **options)
    misfit = case["root"] / "log/misfit"
    if misfit.exists():
        misfit.unlink()
    run = _run_denise(case["binary"], case["mpiexec"], case["root"], parameter,
                      case["workflows"][workflow], ranks=nprocx * nprocy)
    return run, output


def _files(case: dict, output: str, iteration: int, suffixes, stage: int = 1):
    prefix = f"{case['root'] / output}_stage_{stage}_it_{iteration}"
    return {s: Path(f"{prefix}.{s}") for s in suffixes}


def _log_lines(run) -> list[str]:
    return [line.strip() for line in run.stdout.splitlines()
            if re.search(r"opteps_vp|L2t\[1\]|epst1\[1\]", line)]


def _lbfgs_log(run) -> list[dict]:
    out = []
    for m in LOG_PATTERN.finditer(run.stdout):
        out.append({"stage": int(m.group(1)), "iteration": int(m.group(2)),
                    "action": m.group(3), "pair": m.group(4), "pairs": int(m.group(5)),
                    "curvature": float(m.group(6)), "gamma": float(m.group(7)),
                    "descent": float(m.group(8)), "kappa": float(m.group(9))})
    return out


def _sigma_log(run) -> list[tuple[int, list[float]]]:
    return [(int(m.group(1)), [float(m.group(k)) for k in (2, 3, 4)])
            for m in SIGMA_PATTERN.finditer(run.stdout)]


def _driver_mask() -> list[list[float]]:
    return ref32.update_mask(DRIVER_N, DRIVER_N, 0, 0, 1, 1, DRIVER_FW, 0, 0)


def _grids(case: dict, output: str, iteration: int, suffixes, stage: int = 1):
    files = _files(case, output, iteration, suffixes, stage)
    return [_to_grid(_read_native(files[s])) for s in suffixes]


def _ulp32(value: float) -> float:
    bits = ref32.f32_bits(abs(value))
    return struct.unpack("=f", struct.pack("=I", bits + 1))[0] - abs(value)


def _assert_configuration_abort(run, case: dict, output: str) -> None:
    """Section 4: a collective abort before the first iteration, no model output."""
    assert run.returncode != 0 or "MPI_ABORT was invoked" in run.stdout, run.stdout
    assert "TDFWI ITERATION" not in run.stdout, run.stdout
    assert not (case["root"] / "log/misfit").exists()
    for suffix, raw in case["start_bytes"].items():
        assert case["start"].with_suffix(suffix).read_bytes() == raw
    assert not list((case["root"] / "model").glob(f"{Path(output).name}*"))


# O1 -------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("variant", ["scaling0", "scaling1", "precond_adapt"])
def test_o1_lbfgs_off_is_bitwise_identical_to_131_records(driver_case: dict,
                                                          variant: str) -> None:
    options = {"scaling0": {"scaling": 0}, "scaling1": {"scaling": 1},
               "precond_adapt": {"scaling": 1, "precond": 2, "adapt": 1}}[variant]
    eps = 1.0e6 if variant == "scaling0" else SCALED_EPS
    runs = {}
    for records, extra in ((131, {}), (133, {"lbfgs": 0, "lbfgs_m": 7})):
        run, output = _fwi(driver_case, f"o1_{variant}_r{records}", records=records,
                           itermax=2, diag=1, eps_scale=eps, **options, **extra)
        assert run.returncode == 0, run.stdout
        runs[records] = (run, output, (driver_case["root"] / "log/misfit").read_bytes())
    suffixes = MODEL_SUFFIXES + DIAG_SUFFIXES + (PRECOND_SUFFIXES
                                                 if variant == "precond_adapt" else ())
    for iteration in (1, 2):
        for suffix in suffixes:
            left = _files(driver_case, runs[131][1], iteration, [suffix])[suffix]
            right = _files(driver_case, runs[133][1], iteration, [suffix])[suffix]
            assert left.read_bytes() == right.read_bytes(), (iteration, suffix)
    assert _log_lines(runs[131][0]) == _log_lines(runs[133][0])
    assert runs[131][2] == runs[133][2]
    assert "Exact SH L-BFGS" not in runs[133][0].stdout


# O2 / O3 (driver) -----------------------------------------------------------

@pytest.mark.integration
def test_o2_o3_driver_lbfgs_iteration_two_closed_form(driver_case: dict) -> None:
    common = dict(records=133, itermax=2, scaling=1, diag=1, precond=2)
    off, off_out = _fwi(driver_case, "o2_off", lbfgs=0, **common)
    assert off.returncode == 0, off.stdout
    on, on_out = _fwi(driver_case, "o2_on", lbfgs=1, **common)
    assert on.returncode == 0, on.stdout
    for suffix in MODEL_SUFFIXES + DIAG_SUFFIXES + PRECOND_SUFFIXES:
        assert _files(driver_case, off_out, 1, [suffix])[suffix].read_bytes() == \
            _files(driver_case, on_out, 1, [suffix])[suffix].read_bytes(), suffix
    for suffix in ("grad_vs", "grad_rho", "grad_qs") + PRECOND_SUFFIXES:
        assert _files(driver_case, off_out, 2, [suffix])[suffix].read_bytes() == \
            _files(driver_case, on_out, 2, [suffix])[suffix].read_bytes(), suffix
    assert _files(driver_case, off_out, 2, ["step_vs"])["step_vs"].read_bytes() != \
        _files(driver_case, on_out, 2, ["step_vs"])["step_vs"].read_bytes()
    log = _lbfgs_log(on)
    assert [(l["iteration"], l["action"], l["pair"], l["pairs"]) for l in log] == \
        [(1, "sd_empty", "none", 0), (2, "lbfgs", "stored", 1)]
    # O3 driver: reference from the written files.
    mask = _driver_mask()
    start = [_to_grid(_read_native(driver_case["start"].with_suffix(f".{s}")))
             for s in MODEL_SUFFIXES]
    m2 = _grids(driver_case, on_out, 1, MODEL_SUFFIXES)
    g1 = _grids(driver_case, on_out, 1, ("grad_vs", "grad_rho", "grad_qs"))
    g2 = _grids(driver_case, on_out, 2, ("grad_vs", "grad_rho", "grad_qs"))
    p1 = _grids(driver_case, on_out, 1, ("precond",))[0]
    p2 = _grids(driver_case, on_out, 2, ("precond",))[0]
    model = ref.LbfgsReference(5)
    sig = model.begin_stage(start, mask)
    zeros = [[[0.0] * DRIVER_N for _ in range(DRIVER_N)] for _ in range(3)]
    first = model.step(start, g1, mask, p1, (1.0, 1.0, 1.0), zeros)
    assert first["action"] == 0
    second = model.step(m2, g2, mask, p2, (1.0, 1.0, 1.0), zeros)
    assert second["action"] == 3
    got = _grids(driver_case, on_out, 2, ("step_vs", "step_rho", "step_qs"))
    for c in range(3):
        scale = max(abs(v) for row in second["step"][c] for v in row)
        worst = max(abs(got[c][j][i] - second["step"][c][j][i])
                    for j in range(DRIVER_N) for i in range(DRIVER_N))
        assert worst <= STEP_TOLERANCE * scale, (c, worst, scale)
    for name in ("curvature", "gamma", "descent", "kappa"):
        assert _rel_close(log[1][name], second[name], PRINTED_TOLERANCE), \
            (name, log[1][name], second[name])
    (stage_no, printed_sigma), = _sigma_log(on)
    assert stage_no == 1
    for a, b in zip(printed_sigma, sig):
        assert _rel_close(a, b, PRINTED_TOLERANCE)
    # RC-4: the L-BFGS direction reaches the trial (m_2 = m_1 - alpha_2 * p_2).
    for suffix in MODEL_SUFFIXES:
        assert _files(driver_case, off_out, 2, [suffix])[suffix].read_bytes() != \
            _files(driver_case, on_out, 2, [suffix])[suffix].read_bytes(), suffix
    alphas = [float(v) for v in re.findall(r"opteps_vp = (\S+)", on.stdout)]
    assert len(alphas) == 2
    alpha2 = alphas[1]
    m3 = _grids(driver_case, on_out, 2, MODEL_SUFFIXES)
    for c in range(3):
        for j in range(DRIVER_N):
            for i in range(DRIVER_N):
                p = got[c][j][i]
                want_m = ref32.f32(m2[c][j][i] - ref32.f32(alpha2 * p))
                bound = 2.0 * _ulp32(m3[c][j][i]) + 1.0e-6 * abs(alpha2 * p)
                assert abs(m3[c][j][i] - want_m) <= bound, (c, j, i, m3[c][j][i], want_m)


# O7 (driver): delayed classes and stage start ----------------------------------

@pytest.mark.integration
def test_o7_driver_class_activation_resets_memory(driver_case: dict) -> None:
    run, output = _fwi(driver_case, "o7_delayed", records=133, itermax=4, scaling=1,
                       diag=1, precond=2, lbfgs=1, workflow="delayed")
    assert run.returncode == 0, run.stdout
    log = _lbfgs_log(run)
    assert [(l["iteration"], l["action"], l["pair"], l["pairs"]) for l in log] == [
        (1, "sd_empty", "none", 0), (2, "lbfgs", "stored", 1),
        (3, "sd_reset", "none", 0), (4, "lbfgs", "stored", 1)]
    for suffix in ("step_rho", "step_qs"):
        assert all(v == 0.0 for v in _read_native(_files(driver_case, output, 2,
                                                         [suffix])[suffix]))


@pytest.mark.integration
def test_o7_driver_second_stage_starts_empty(driver_case: dict) -> None:
    run, output = _fwi(driver_case, "o7_two_stages", records=133, itermax=2, scaling=1,
                       precond=2, lbfgs=1, workflow="two")
    assert run.returncode == 0, run.stdout
    log = _lbfgs_log(run)
    stage2 = [l for l in log if l["stage"] == 2]
    assert stage2 and stage2[0]["iteration"] == 1 and stage2[0]["action"] == "sd_empty"
    sigmas = dict(_sigma_log(run))
    assert set(sigmas) == {1, 2}
    last = max(l["iteration"] for l in log if l["stage"] == 1)
    final = _grids(driver_case, output, last, MODEL_SUFFIXES, stage=1)
    want = ref.sigma(final, _driver_mask())
    for a, b in zip(sigmas[2], want):
        assert _rel_close(a, b, PRINTED_TOLERANCE)


# O10 (driver) -----------------------------------------------------------------

@pytest.mark.integration
def test_o10_driver_decompositions_agree(driver_case: dict) -> None:
    outputs, logs = {}, {}
    for nprocx, nprocy in ((1, 1), (2, 1), (1, 2)):
        run, output = _fwi(driver_case, f"o10_{nprocx}x{nprocy}", records=133, itermax=2,
                           nprocx=nprocx, nprocy=nprocy, scaling=1, diag=1, precond=2,
                           lbfgs=1)
        assert run.returncode == 0, run.stdout
        outputs[(nprocx, nprocy)] = output
        logs[(nprocx, nprocy)] = [(l["action"], l["pair"], l["pairs"])
                                  for l in _lbfgs_log(run)]
    assert logs[(1, 1)] == logs[(2, 1)] == logs[(1, 2)]
    for suffix in ("step_vs", "step_rho", "step_qs"):
        serial = _read_native(_files(driver_case, outputs[(1, 1)], 2, [suffix])[suffix])
        scale = max(abs(v) for v in serial)
        assert scale > 0.0, suffix
        for key in ((2, 1), (1, 2)):
            other = _read_native(_files(driver_case, outputs[key], 2, [suffix])[suffix])
            worst = max(abs(a - b) for a, b in zip(serial, other))
            assert worst <= STEP_TOLERANCE * scale, (suffix, key, worst, scale)


# O11 (driver): configuration aborts and records -------------------------------

CONFIG_ABORTS = {
    "lbfgs_mode": ({"lbfgs": 2, "scaling": 1}, 1),
    "lbfgs_negative": ({"lbfgs": -1, "scaling": 1}, 1),
    "lbfgs_without_scaling": ({"lbfgs": 1, "scaling": 0}, 1),
    "lbfgs_max_scaling": ({"lbfgs": 1, "scaling": 2}, 1),
    "lbfgs_without_scaling_two_ranks": ({"lbfgs": 1, "scaling": 0}, 2),
    "memory_zero_off": ({"lbfgs": 0, "lbfgs_m": 0}, 1),
    "memory_large": ({"lbfgs": 1, "scaling": 1, "lbfgs_m": 21}, 1),
    "memory_negative_two_ranks": ({"lbfgs": 1, "scaling": 1, "lbfgs_m": -3}, 2),
    "lbfgs_invmat3": ({"lbfgs": 1, "scaling": 1, "invmat1": 3,
                       "model_prefix": "model/start_mu"}, 1),
    "lbfgs_invmat3_two_ranks": ({"lbfgs": 1, "scaling": 1, "invmat1": 3,
                                 "model_prefix": "model/start_mu"}, 2),
}


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(CONFIG_ABORTS))
def test_o11_configuration_aborts(driver_case: dict, name: str) -> None:
    options, ranks = CONFIG_ABORTS[name]
    run, output = _fwi(driver_case, f"o11_abort_{name}", records=133, nprocx=ranks,
                       **dict(options))
    _assert_configuration_abort(run, driver_case, output)


@pytest.mark.integration
def test_o11_invmat3_without_lbfgs_reaches_the_first_iteration(driver_case: dict) -> None:
    """Positive control for the INVMAT1 abort (contract rev 3): only L-BFGS is refused."""
    run, _ = _fwi(driver_case, "o11_invmat3_off", records=133, scaling=1, lbfgs=0,
                  invmat1=3, model_prefix="model/start_mu")
    assert "TDFWI ITERATION 1" in run.stdout, run.stdout


@pytest.mark.integration
def test_o11_record_132_alone_uses_default_memory(driver_case: dict) -> None:
    short, short_out = _fwi(driver_case, "o11_records_132", records=132, itermax=2,
                            scaling=1, diag=1, precond=2, lbfgs=1)
    assert short.returncode == 0, short.stdout
    full, full_out = _fwi(driver_case, "o11_records_133", records=133, itermax=2,
                          scaling=1, diag=1, precond=2, lbfgs=1, lbfgs_m=5)
    assert full.returncode == 0, full.stdout
    for iteration in (1, 2):
        for suffix in MODEL_SUFFIXES + ("step_vs", "step_rho", "step_qs"):
            assert _files(driver_case, short_out, iteration, [suffix])[suffix].read_bytes() \
                == _files(driver_case, full_out, iteration, [suffix])[suffix].read_bytes()
    assert _lbfgs_log(short) == _lbfgs_log(full)


@pytest.mark.integration
def test_o11_memory_length_reaches_the_unit(driver_case: dict) -> None:
    results = {}
    for memory in (1, 5):
        run, _ = _fwi(driver_case, f"o11_memory_{memory}", records=133, itermax=3,
                      scaling=1, precond=2, lbfgs=1, lbfgs_m=memory, workflow="nostop")
        assert run.returncode == 0, run.stdout
        results[memory] = _lbfgs_log(run)
    assert [l["pair"] for l in results[5]] == ["none", "stored", "stored"]
    assert results[5][2]["pairs"] == 2
    assert results[1][2]["pairs"] == 1
