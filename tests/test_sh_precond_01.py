"""SH-PRECOND-01 acceptance oracles O1-O10.

Written before the implementation from the frozen contract
``docs/sh_precond_01_contract.md`` revision 2 (SHA-256 pinned below).
Revision 2 applies the independent oracle review (HOLD) of the r1 set.
Expected values come only from that contract and from the independent
reference ``tests/utilities/sh_precond_01_reference.py``; production code is
exercised exclusively through its public C API and the compiled driver.

Oracle map (contract section 5):
  O1  identity of records 127 vs 131 with controls off     -> test_o1_*
  O2  no interference with objective / raw gradient        -> test_o2_*
  O3  energy accumulation bound; no trial leakage          -> test_o3_*
  O4  receiver-line term vs quadrature                     -> test_o4_*
  O5  closed forms of P and p (unit and driver)            -> test_o5_*
  O6  descent sign                                         -> test_o6_*
  O7  invariance under W -> c W                            -> test_o7_*
  O8  MPI decompositions                                   -> test_o8_*
  O9  error paths, initial alpha, configuration, records   -> test_o9_*
  O10 reported evidence only (printed in test_o2_o5_*)
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

from tests.utilities import sh_opt_scale_01_reference as ref32
from tests.utilities import sh_precond_01_reference as ref
from tests.utilities.sh_fwi_fixture import write_sh_parameter


CONTRACT_PATH = "docs/sh_precond_01_contract.md"
CONTRACT_SHA256 = "5404af51f3818926556f0476c508f7cd0583d264a31a5233596aacc0f7219cea"

# Contract section 5 tolerances; never widen these.
ULP_TOLERANCE = 4
RELATIVE_TOLERANCE = 1.0e-6

CASE_MAGIC = 0x31435250
RESULT_MAGIC = 0x52435250
OUTPUT_SENTINEL = -98765.25
CLASSES = ref.CLASSES
NXG, NYG = 12, 10


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _launcher() -> str:
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert launcher, "an MPI launcher is required for the SH-PRECOND-01 harness"
    return launcher


def _relative_close(left: float, right: float, tol: float = RELATIVE_TOLERANCE) -> bool:
    if left == 0.0 or right == 0.0:
        return left == right
    return abs(left - right) <= tol * max(abs(left), abs(right))


def _grid(function, nx: int = NXG, ny: int = NYG) -> list[list[float]]:
    return [[ref32.f32(function(j, i)) for i in range(nx)] for j in range(ny)]


def _frame_mask(width: int, nx: int = NXG, ny: int = NYG) -> list[list[float]]:
    return [[0.0 if (i < width or i >= nx - width or j < width or j >= ny - width)
             else 1.0 for i in range(nx)] for j in range(ny)]


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory, repository_root: Path) -> Path:
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the SH-PRECOND-01 harness"
    output = tmp_path_factory.mktemp("sh_precond_01") / "harness"
    command = [
        compiler, "-std=c99", "-O2", "-fcommon", "-I", str(repository_root / "include"),
        str(repository_root / "tests/utilities/sh_precond_01_harness.c"),
        str(repository_root / "src/SH/visco_sh_exact_precond.c"),
        str(repository_root / "src/SH/visco_sh_exact_step_scaling.c"),
        "-o", str(output), "-lm",
    ]
    run = subprocess.run(command, cwd=repository_root, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert run.returncode == 0, run.stdout
    return output


def _run(harness: Path, workdir: Path, *, op: int, nxg: int = NXG, nyg: int = NYG,
         nprocx: int = 1, nprocy: int = 1, mode: int = 0, scaling_mode: int = 0,
         fault: int = 0, fault_rank: int = -1, nsteps: int = 0,
         override_rank: int = -1, override_value: int = 0, epsilon: float = 0.005,
         x_min: float = 0.0, x_max: float = 0.0, weights=(1.0, 1.0, 1.0),
         fields: list[list[list[float]]] = (), tail: bytes = b"") -> list[dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    payload = bytearray(struct.pack(
        "=13i", CASE_MAGIC, op, nxg, nyg, nprocx, nprocy, mode, scaling_mode, fault,
        fault_rank, nsteps, override_rank, override_value))
    payload += struct.pack("=3d", epsilon, x_min, x_max)
    payload += struct.pack("=3f", *weights)
    for grid in fields:
        assert len(grid) == nyg and all(len(row) == nxg for row in grid)
        payload += struct.pack(f"={nxg * nyg}f", *(v for row in grid for v in row))
    payload += tail
    case = workdir / "case.bin"
    case.write_bytes(bytes(payload))
    ranks = nprocx * nprocy
    run = subprocess.run(
        [_launcher(), "--oversubscribe", "-n", str(ranks), str(harness), str(case),
         str(workdir / "result")],
        cwd=workdir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120)
    assert run.returncode == 0, run.stdout
    results = []
    for rank in range(ranks):
        raw = (workdir / f"result.{rank}").read_bytes()
        magic, file_rank, status, pos_x, pos_y, nx, ny, unchanged, nout = \
            struct.unpack("=9i", raw[:36])
        assert magic == RESULT_MAGIC and file_rank == rank
        count = (nx + 2) * (ny + 2)
        values = struct.unpack(f"={nout * count}f", raw[36:])
        arrays = []
        for k in range(nout):
            local = values[k * count:(k + 1) * count]
            arrays.append([list(local[j * (nx + 2):(j + 1) * (nx + 2)])
                           for j in range(ny + 2)])
        results.append({"status": status, "pos": (pos_x, pos_y), "nx": nx, "ny": ny,
                        "unchanged": bool(unchanged), "arrays": arrays})
    return results


def _assemble(results: list[dict], index: int, nxg: int = NXG, nyg: int = NYG):
    """Global own-cell field from per-rank arrays; also checks halos untouched."""
    out = [[None] * nxg for _ in range(nyg)]
    halos = True
    for r in results:
        nx, ny = r["nx"], r["ny"]
        px, py = r["pos"]
        arr = r["arrays"][index]
        for j in range(ny + 2):
            for i in range(nx + 2):
                if j in (0, ny + 1) or i in (0, nx + 1):
                    halos &= arr[j][i] == OUTPUT_SENTINEL
                else:
                    out[py * ny + j - 1][px * nx + i - 1] = arr[j][i]
    return out, halos


def _untouched(results: list[dict], index: int) -> bool:
    return all(v == OUTPUT_SENTINEL for r in results for row in r["arrays"][index]
               for v in row)


# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

def test_contract_identity(repository_root: Path) -> None:
    digest = hashlib.sha256((repository_root / CONTRACT_PATH).read_bytes()).hexdigest()
    assert digest == CONTRACT_SHA256


def test_api_declared_in_header_and_makefile(repository_root: Path) -> None:
    header = re.sub(r"\s+", " ", (repository_root / "include/fd.h").read_text(encoding="utf-8"))
    for pattern in (
        r"struct visco_sh_exact_precond_request \{",
        r"void visco_sh_exact_accumulate_energy\(",
        r"int visco_sh_exact_build_receiver_line_term\(",
        r"int visco_sh_exact_build_illumination_precond\(",
        r"int visco_sh_exact_build_preconditioned_step\(",
        r"float visco_sh_exact_initial_alpha\(",
    ):
        assert re.search(pattern, header), pattern
    makefile = (repository_root / "src/Makefile").read_text(encoding="utf-8")
    assert "visco_sh_exact_precond.c" in makefile


# ---------------------------------------------------------------------------
# O3: energy accumulation (2.1)
# ---------------------------------------------------------------------------

def _snapshots(seed: int, steps: int) -> list[list[list[float]]]:
    rng = random.Random(seed)
    return [_grid(lambda j, i: rng.uniform(-1.0, 1.0) * 10.0 ** rng.uniform(-6, 3))
            for _ in range(steps)]


@pytest.mark.parametrize("steps", [1, 7, 64])
def test_o3_energy_accumulation_meets_summation_bound(harness: Path, tmp_path: Path,
                                                      steps: int) -> None:
    snapshots = _snapshots(31 + steps, steps)
    results = _run(harness, tmp_path, op=1, nsteps=steps, fields=snapshots)
    energy, halos = _assemble(results, 0)
    assert halos
    exact = ref.energy_exact(snapshots)
    for j in range(NYG):
        for i in range(NXG):
            assert abs(energy[j][i] - exact[j][i]) <= ref.energy_bound(steps, exact[j][i]), (j, i)


@pytest.mark.parametrize("fault", [1, 2, 3])
def test_o3_o9_energy_accumulation_degenerate_arguments_are_no_ops(
        harness: Path, tmp_path: Path, fault: int) -> None:
    snapshots = _snapshots(5, 3)
    results = _run(harness, tmp_path, op=1, nsteps=3, fault=fault, fields=snapshots)
    energy, halos = _assemble(results, 0)
    assert halos
    assert all(v == 0.0 for row in energy for v in row)


# ---------------------------------------------------------------------------
# O4: receiver-line term (2.2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pos,extent", [((0, 0), (3.0, 9.0)), ((2, 1), (10.0, 22.0)),
                                        ((0, 3), (-4.0, 40.0))])
def test_o4_receiver_term_matches_quadrature(harness: Path, tmp_path: Path, pos,
                                             extent) -> None:
    nx, ny = 6, 5
    results = _run(harness, tmp_path, op=2, nxg=nx, nyg=ny, mode=pos[0],
                   scaling_mode=pos[1], x_min=extent[0], x_max=extent[1])
    assert results[0]["status"] == 0
    assert results[0]["pos"] == pos
    # Op 2 returns the local block of rank position ``pos`` on a single rank.
    term, halos = _assemble([{**results[0], "pos": (0, 0)}], 0, nx, ny)
    assert halos
    expected = ref.receiver_term_local(nx, ny, pos[0], pos[1], *extent)
    for j in range(ny):
        for i in range(nx):
            assert term[j][i] > 0.0
            assert _relative_close(term[j][i], expected[j][i]), (j, i, term[j][i], expected[j][i])


@pytest.mark.parametrize("fault,pos,extent", [
    (1, (0, 0), (1.0, 5.0)), (2, (0, 0), (1.0, 5.0)), (0, (-1, 0), (1.0, 5.0)),
    (0, (0, -1), (1.0, 5.0)), (0, (0, 0), (5.0, 5.0)), (0, (0, 0), (6.0, 5.0)),
    (0, (0, 0), (math.nan, 5.0)), (0, (0, 0), (1.0, math.inf)),
], ids=["null", "null_row", "pos_x", "pos_y", "equal", "reversed", "nan", "inf"])
def test_o9_receiver_term_errors(harness: Path, tmp_path: Path, fault, pos, extent) -> None:
    results = _run(harness, tmp_path, op=2, nxg=4, nyg=3, mode=pos[0],
                   scaling_mode=pos[1], fault=fault, x_min=extent[0], x_max=extent[1])
    assert results[0]["status"] == -1
    assert _untouched(results, 0)


# ---------------------------------------------------------------------------
# O5 / O7 / O8 / O9: preconditioner (2.3)
# ---------------------------------------------------------------------------

def _precond_fields(seed: int, mask_width: int = 2):
    rng = random.Random(seed)
    ws = _grid(lambda j, i: rng.uniform(0.0, 1.0) * 10.0 ** rng.uniform(-5, 2))
    rr = _grid(lambda j, i: rng.uniform(0.05, 3.0))
    return ws, rr, _frame_mask(mask_width)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("epsilon", [0.005, 0.2, 1.0])
def test_o5_precond_matches_closed_form(harness: Path, tmp_path: Path, mode: int,
                                        epsilon: float) -> None:
    ws, rr, mask = _precond_fields(7 + mode)
    results = _run(harness, tmp_path, op=3, mode=mode, epsilon=epsilon,
                   fields=[ws, rr, mask])
    assert [r["status"] for r in results] == [0]
    assert results[0]["unchanged"]
    got, halos = _assemble(results, 0)
    assert halos
    want = ref.precond(mode, epsilon, ws, rr, mask)
    w_max = max(ref.illumination_weight(mode, ws[j][i], rr[j][i])
                for j in range(NYG) for i in range(NXG) if mask[j][i] == 1.0)
    for j in range(NYG):
        for i in range(NXG):
            if mask[j][i] == 0.0:
                assert got[j][i] == 0.0 and math.copysign(1.0, got[j][i]) > 0
            else:
                assert ref32.ulp_distance_f32(got[j][i], want[j][i]) <= ULP_TOLERANCE
                assert 0.0 < got[j][i] <= ref32.f32(1.0 / (ref32.f32(epsilon) * w_max)) * (1 + 1e-6)


def test_o5_precond_ignores_invalid_values_outside_mask(harness: Path, tmp_path: Path) -> None:
    ws, rr, mask = _precond_fields(17)
    ws[0][0] = math.nan
    ws[0][1] = -3.0
    rr[0][0] = -1.0
    rr[1][0] = math.inf
    results = _run(harness, tmp_path, op=3, mode=2, fields=[ws, rr, mask])
    assert results[0]["status"] == 0
    got, _ = _assemble(results, 0)
    assert got[0][0] == 0.0 and got[0][1] == 0.0 and got[1][0] == 0.0


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("decomposition", [(2, 1), (1, 2), (3, 2)])
def test_o8_precond_decompositions_agree(harness: Path, tmp_path: Path, mode: int,
                                         decomposition) -> None:
    ws, rr, mask = _precond_fields(23)
    serial, _ = _assemble(_run(harness, tmp_path / "serial", op=3, mode=mode,
                               fields=[ws, rr, mask]), 0)
    results = _run(harness, tmp_path / "split", op=3, mode=mode, nprocx=decomposition[0],
                   nprocy=decomposition[1], fields=[ws, rr, mask])
    assert all(r["status"] == 0 for r in results)
    parallel, halos = _assemble(results, 0)
    assert halos
    for j in range(NYG):
        for i in range(NXG):
            assert _relative_close(parallel[j][i], serial[j][i]), (j, i)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("c", [1.0e-6, 3.0, 1.0e8])
def test_o7_direction_invariant_under_weight_scaling(harness: Path, tmp_path: Path,
                                                     mode: int, c: float) -> None:
    ws, rr, mask = _precond_fields(41)
    if mode == 1:
        scaled = ([[ref32.f32(v * c) for v in row] for row in ws], rr)
    else:
        scaled = (ws, [[ref32.f32(v * c) for v in row] for row in rr])
    base_p, _ = _assemble(_run(harness, tmp_path / "base", op=3, mode=mode,
                               fields=[ws, rr, mask]), 0)
    scaled_p, _ = _assemble(_run(harness, tmp_path / "scaled", op=3, mode=mode,
                                 fields=[*scaled, mask]), 0)
    step_case = _step_case(1, seed=43)
    base_step = _run_step(harness, tmp_path / "base_step", step_case, base_p)
    scaled_step = _run_step(harness, tmp_path / "scaled_step", step_case, scaled_p)
    for name in CLASSES:
        for j in range(NYG):
            for i in range(NXG):
                assert _relative_close(scaled_step[name][j][i], base_step[name][j][i]), \
                    (name, j, i, scaled_step[name][j][i], base_step[name][j][i])


# Error codes of 2.3 --------------------------------------------------------

PRECOND_STRUCTURAL = {1: -11, 2: -11, 3: -11, 4: -11, 5: -11, 7: -11}


@pytest.mark.parametrize("fault,code", sorted(PRECOND_STRUCTURAL.items()))
@pytest.mark.parametrize("fault_rank", [-1, 1])
def test_o9_precond_structural_errors_are_collective(harness: Path, tmp_path: Path,
                                                     fault: int, code: int,
                                                     fault_rank: int) -> None:
    ws, rr, mask = _precond_fields(3)
    results = _run(harness, tmp_path, op=3, mode=1, nprocx=2, fault=fault,
                   fault_rank=fault_rank, fields=[ws, rr, mask])
    assert [r["status"] for r in results] == [code, code]
    assert _untouched(results, 0)


def test_o9_precond_receiver_term_null_only_matters_in_mode_2(harness: Path,
                                                              tmp_path: Path) -> None:
    ws, rr, mask = _precond_fields(4)
    ok = _run(harness, tmp_path / "m1", op=3, mode=1, fault=6, fields=[ws, rr, mask])
    assert ok[0]["status"] == 0
    bad = _run(harness, tmp_path / "m2", op=3, mode=2, fault=6, fields=[ws, rr, mask])
    assert bad[0]["status"] == -11 and _untouched(bad, 0)


@pytest.mark.parametrize("mode,epsilon", [(0, 0.005), (3, 0.005), (1, 0.0), (1, -0.1),
                                          (1, 1.5), (1, math.nan), (2, math.inf)])
def test_o9_precond_invalid_mode_or_epsilon_returns_minus_12(harness: Path, tmp_path: Path,
                                                            mode: int, epsilon: float) -> None:
    ws, rr, mask = _precond_fields(5)
    results = _run(harness, tmp_path, op=3, mode=mode, epsilon=epsilon, fields=[ws, rr, mask])
    assert results[0]["status"] == -12 and _untouched(results, 0)


def test_o9_precond_mode_override_on_one_rank_is_collective(harness: Path,
                                                           tmp_path: Path) -> None:
    ws, rr, mask = _precond_fields(6)
    results = _run(harness, tmp_path, op=3, mode=1, nprocx=2, override_rank=1,
                   override_value=4, fields=[ws, rr, mask])
    assert [r["status"] for r in results] == [-12, -12] and _untouched(results, 0)


@pytest.mark.parametrize("mode,field,value", [
    (1, 0, -1.0), (1, 0, math.nan), (1, 0, math.inf), (2, 1, 0.0), (2, 1, -2.0),
    (2, 1, math.nan), (1, 2, 0.5), (1, 2, -0.0 + 2.0)])
def test_o9_precond_invalid_masked_data_returns_minus_13(harness: Path, tmp_path: Path,
                                                        mode: int, field: int,
                                                        value: float) -> None:
    fields = list(_precond_fields(8))
    fields[field] = [row[:] for row in fields[field]]
    fields[field][5][9] = value          # rank 1 of a 2x1 split, masked cell
    results = _run(harness, tmp_path, op=3, mode=mode, nprocx=2, fields=fields)
    assert [r["status"] for r in results] == [-13, -13] and _untouched(results, 0)


def test_o9_precond_empty_mask_or_no_illumination_returns_minus_14(harness: Path,
                                                                 tmp_path: Path) -> None:
    ws, rr, mask = _precond_fields(9)
    empty = _run(harness, tmp_path / "empty", op=3, mode=1, nprocx=2,
                 fields=[ws, rr, _frame_mask(6)])
    assert [r["status"] for r in empty] == [-14, -14] and _untouched(empty, 0)
    dark = _run(harness, tmp_path / "dark", op=3, mode=2, nprocx=2,
                fields=[_grid(lambda j, i: 0.0), rr, mask])
    assert [r["status"] for r in dark] == [-14, -14] and _untouched(dark, 0)


def test_o9_precond_overflow_returns_minus_14(harness: Path, tmp_path: Path) -> None:
    """2.3: (float)(1/(eps*W_max)) not finite -> -14 (binary32 overflow guard)."""
    _, rr, mask = _precond_fields(11)
    tiny = _grid(lambda j, i: 1.0e-40)                 # subnormal binary32
    assert ref.precond_status(1, 0.005, tiny, rr, mask) == -14
    results = _run(harness, tmp_path, op=3, mode=1, nprocx=2, fields=[tiny, rr, mask])
    assert [r["status"] for r in results] == [-14, -14] and _untouched(results, 0)


def test_o9_precond_precedence_smallest_code_wins(harness: Path, tmp_path: Path) -> None:
    ws, rr, mask = _precond_fields(10)
    ws = [row[:] for row in ws]
    ws[5][2] = -1.0                       # -13 on rank 0
    results = _run(harness, tmp_path / "a", op=3, mode=1, epsilon=2.0, nprocx=2,
                   fields=[ws, rr, mask])  # -12 everywhere
    assert [r["status"] for r in results] == [-12, -12]
    results = _run(harness, tmp_path / "b", op=3, mode=1, epsilon=2.0, nprocx=2,
                   fault=3, fault_rank=1, fields=[ws, rr, mask])  # -11 on rank 1
    assert [r["status"] for r in results] == [-11, -11] and _untouched(results, 0)


# ---------------------------------------------------------------------------
# O5 / O6 / O8 / O9: preconditioned step (2.4)
# ---------------------------------------------------------------------------

def _step_case(scaling_mode: int, seed: int = 11, weights=(1.0, 0.5, 2.0),
               mask_width: int = 2) -> dict:
    rng = random.Random(seed)
    fields = {
        "g_primary": _grid(lambda j, i: rng.uniform(-1, 1) * 1e-3),
        "g_rho": _grid(lambda j, i: rng.uniform(-1, 1) * 1e-6),
        "g_q": _grid(lambda j, i: rng.uniform(-1, 1) * 1e-9),
        "m_primary": _grid(lambda j, i: rng.uniform(150, 450)),
        "m_rho": _grid(lambda j, i: rng.uniform(1600, 2100)),
        "m_q": _grid(lambda j, i: rng.uniform(15, 60)),
        "mask": _frame_mask(mask_width),
    }
    return {"mode": scaling_mode, "weights": weights, "fields": fields}


def _p_field(seed: int, mask) -> list[list[float]]:
    rng = random.Random(seed)
    return [[ref32.f32(rng.uniform(0.0, 1.0) * 10.0 ** rng.uniform(-3, 3))
             if mask[j][i] == 1.0 else 0.0 for i in range(NXG)] for j in range(NYG)]


STEP_FIELDS = ("g_primary", "g_rho", "g_q", "m_primary", "m_rho", "m_q", "mask")


def _run_step_raw(harness: Path, workdir: Path, case: dict, p_field, **options) -> list[dict]:
    fields = [case["fields"][name] for name in STEP_FIELDS] + [p_field]
    return _run(harness, workdir, op=4, scaling_mode=case["mode"],
                weights=case["weights"], fields=fields, **options)


def _run_step(harness: Path, workdir: Path, case: dict, p_field, **options) -> dict:
    results = _run_step_raw(harness, workdir, case, p_field, **options)
    assert all(r["status"] == 0 for r in results), [r["status"] for r in results]
    assert all(r["unchanged"] for r in results)
    step = {}
    for k, name in enumerate(CLASSES):
        step[name], halos = _assemble(results, k)
        assert halos
    return step


def _expected_step(case: dict, p_field) -> dict:
    f = case["fields"]
    return ref.preconditioned_step(
        case["mode"], dict(zip(CLASSES, case["weights"])),
        {n: f[f"g_{n}"] for n in CLASSES}, {n: f[f"m_{n}"] for n in CLASSES},
        f["mask"], p_field)


def _assert_step_closed_form(step: dict, case: dict, p_field) -> None:
    want = _expected_step(case, p_field)
    mask = case["fields"]["mask"]
    weights = dict(zip(CLASSES, case["weights"]))
    for name in CLASSES:
        for j in range(NYG):
            for i in range(NXG):
                got = step[name][j][i]
                if mask[j][i] == 0.0 or weights[name] == 0.0 or want[name][j][i] == 0.0:
                    assert got == 0.0, (name, j, i, got)
                else:
                    assert ref32.ulp_distance_f32(got, want[name][j][i]) <= ULP_TOLERANCE, \
                        (name, j, i, got, want[name][j][i])


@pytest.mark.parametrize("scaling_mode", [1, 2])
@pytest.mark.parametrize("weights", [(1.0, 0.5, 2.0), (1.0, 0.0, 1.0)], ids=["w", "zero_rho"])
def test_o5_preconditioned_step_matches_closed_form(harness: Path, tmp_path: Path,
                                                    scaling_mode: int, weights) -> None:
    case = _step_case(scaling_mode, weights=weights)
    p_field = _p_field(13, case["fields"]["mask"])
    step = _run_step(harness, tmp_path, case, p_field)
    _assert_step_closed_form(step, case, p_field)


def test_o5_zero_masked_gradient_class_gives_zero_step(harness: Path, tmp_path: Path) -> None:
    case = _step_case(1)
    case["fields"]["g_q"] = _grid(lambda j, i: 0.0)
    p_field = _p_field(14, case["fields"]["mask"])
    step = _run_step(harness, tmp_path, case, p_field)
    assert all(v == 0.0 for row in step["q"] for v in row)
    _assert_step_closed_form(step, case, p_field)


def test_o5_unit_precond_reproduces_scaled_step(harness: Path, tmp_path: Path) -> None:
    """P == 1 on the mask: p equals the SH-OPT-SCALE-01 closed form."""
    case = _step_case(1)
    ones = [[1.0 if m == 1.0 else 0.0 for m in row] for row in case["fields"]["mask"]]
    step = _run_step(harness, tmp_path, case, ones)
    f = case["fields"]
    expected = ref32.scaled_step(1, dict(zip(CLASSES, case["weights"])),
                                 {n: f[f"g_{n}"] for n in CLASSES},
                                 {n: f[f"m_{n}"] for n in CLASSES}, f["mask"])
    for name in CLASSES:
        for j in range(NYG):
            for i in range(NXG):
                assert ref32.ulp_distance_f32(step[name][j][i], expected[name][j][i]) <= \
                    ULP_TOLERANCE


@pytest.mark.parametrize("scaling_mode", [1, 2])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_o6_preconditioned_step_is_a_descent_direction(harness: Path, tmp_path: Path,
                                                       scaling_mode: int, seed: int) -> None:
    case = _step_case(scaling_mode, seed=100 + seed)
    p_field = _p_field(200 + seed, case["fields"]["mask"])
    step = _run_step(harness, tmp_path, case, p_field)
    f = case["fields"]
    dot = ref.descent_dot({n: f[f"g_{n}"] for n in CLASSES}, step)
    assert dot > 0.0


@pytest.mark.parametrize("decomposition", [(2, 1), (1, 2), (3, 2)])
@pytest.mark.parametrize("scaling_mode", [1, 2])
def test_o8_preconditioned_step_decompositions_agree(harness: Path, tmp_path: Path,
                                                     decomposition, scaling_mode: int) -> None:
    case = _step_case(scaling_mode, seed=55)
    p_field = _p_field(56, case["fields"]["mask"])
    serial = _run_step(harness, tmp_path / "serial", case, p_field)
    parallel = _run_step(harness, tmp_path / "split", case, p_field,
                         nprocx=decomposition[0], nprocy=decomposition[1])
    for name in CLASSES:
        for j in range(NYG):
            for i in range(NXG):
                assert _relative_close(parallel[name][j][i], serial[name][j][i])


STEP_ERRORS = {1: -1, 2: -2, 3: -3, 4: -15, 5: -16, 6: -16, 7: -16}


def _assert_step_rejected(results: list[dict], code: int, *, work_untouched: bool) -> None:
    """O9: identical code everywhere, no step written; step-1 codes also keep work."""
    assert [r["status"] for r in results] == [code] * len(results)
    assert all(r["unchanged"] for r in results)
    for k in range(3):
        assert _untouched(results, k), f"step array {k} written"
    if work_untouched:
        for k in range(3, 6):
            assert _untouched(results, k), f"work array {k - 3} written"


@pytest.mark.parametrize("fault,code", sorted(STEP_ERRORS.items()))
@pytest.mark.parametrize("fault_rank", [-1, 1])
def test_o9_preconditioned_step_structural_errors(harness: Path, tmp_path: Path, fault: int,
                                                  code: int, fault_rank: int) -> None:
    case = _step_case(1)
    p_field = _p_field(15, case["fields"]["mask"])
    results = _run_step_raw(harness, tmp_path, case, p_field, nprocx=2, fault=fault,
                            fault_rank=fault_rank)
    _assert_step_rejected(results, code, work_untouched=True)


@pytest.mark.parametrize("scaling_mode", [0, 3])
def test_o9_preconditioned_step_requires_scaling_mode_1_or_2(harness: Path, tmp_path: Path,
                                                            scaling_mode: int) -> None:
    case = _step_case(1)
    case["mode"] = scaling_mode
    results = _run_step_raw(harness, tmp_path, case, _p_field(16, case["fields"]["mask"]))
    _assert_step_rejected(results, -15, work_untouched=True)


def test_o9_preconditioned_step_mode_override_is_collective(harness: Path,
                                                           tmp_path: Path) -> None:
    case = _step_case(1)
    results = _run_step_raw(harness, tmp_path, case, _p_field(17, case["fields"]["mask"]),
                            nprocx=2, override_rank=0, override_value=0)
    _assert_step_rejected(results, -15, work_untouched=True)


@pytest.mark.parametrize("value,where", [(-1.0, (5, 9)), (math.nan, (5, 9)),
                                         (math.inf, (0, 0)), (-0.5, (0, 0))],
                         ids=["neg_masked", "nan_masked", "inf_unmasked", "neg_unmasked"])
def test_o9_invalid_precond_returns_minus_17(harness: Path, tmp_path: Path, value: float,
                                             where) -> None:
    case = _step_case(1)
    p_field = [row[:] for row in _p_field(18, case["fields"]["mask"])]
    p_field[where[0]][where[1]] = value
    results = _run_step_raw(harness, tmp_path, case, p_field, nprocx=2)
    _assert_step_rejected(results, -17, work_untouched=True)


def test_o9_negative_zero_precond_is_accepted(harness: Path, tmp_path: Path) -> None:
    """2.4: -17 is P < 0.0f; -0.0f compares equal to 0 and is accepted."""
    case = _step_case(1)
    p_field = [row[:] for row in _p_field(21, case["fields"]["mask"])]
    p_field[0][0] = -0.0
    p_field[5][5] = -0.0
    step = _run_step(harness, tmp_path, case, p_field)
    _assert_step_closed_form(step, case, p_field)


def test_o9_preconditioned_step_precedence(harness: Path, tmp_path: Path) -> None:
    case = _step_case(1)
    p_field = _p_field(22, case["fields"]["mask"])
    bad_p = [row[:] for row in p_field]
    bad_p[5][2] = -1.0                                   # -17 on rank 0
    # -15 (rank 0: scaling mode 0) wins over -16 (rank 1: precond NULL).
    results = _run_step_raw(harness, tmp_path / "a", case, p_field, nprocx=2,
                            override_rank=0, override_value=0, fault=5, fault_rank=1)
    _assert_step_rejected(results, -15, work_untouched=True)
    # -16 (rank 1: work_rho NULL) wins over -17 (rank 0: negative P).
    results = _run_step_raw(harness, tmp_path / "b", case, bad_p, nprocx=2, fault=6,
                            fault_rank=1)
    _assert_step_rejected(results, -16, work_untouched=True)
    # -17 wins over the inner SH-OPT-SCALE-01 codes -5 / -8.
    inner = _step_case(1)
    inner["fields"]["g_rho"] = [row[:] for row in inner["fields"]["g_rho"]]
    inner["fields"]["g_rho"][4][8] = math.inf            # would be -8 (rank 1)
    inner["fields"]["mask"] = [row[:] for row in inner["fields"]["mask"]]
    inner["fields"]["mask"][3][3] = 0.5                  # would be -5 (rank 0)
    results = _run_step_raw(harness, tmp_path / "c", inner, bad_p, nprocx=2)
    _assert_step_rejected(results, -17, work_untouched=True)


def test_o9_scaled_step_codes_pass_through(harness: Path, tmp_path: Path) -> None:
    """After step-1 validation, SH-OPT-SCALE-01 codes pass through (-8, -6, -5)."""
    case = _step_case(1)
    case["fields"]["g_rho"] = [row[:] for row in case["fields"]["g_rho"]]
    case["fields"]["g_rho"][4][4] = math.inf
    results = _run_step_raw(harness, tmp_path, case, _p_field(19, case["fields"]["mask"]),
                            nprocx=2)
    _assert_step_rejected(results, -8, work_untouched=False)
    weights_bad = _step_case(1, weights=(0.0, 0.0, 0.0))
    results = _run_step_raw(harness, tmp_path / "w", weights_bad,
                            _p_field(20, weights_bad["fields"]["mask"]))
    _assert_step_rejected(results, -6, work_untouched=False)
    mask_bad = _step_case(1)
    mask_bad["fields"]["mask"] = [row[:] for row in mask_bad["fields"]["mask"]]
    mask_bad["fields"]["mask"][3][3] = 0.5
    results = _run_step_raw(harness, tmp_path / "m", mask_bad,
                            _p_field(23, _step_case(1)["fields"]["mask"]))
    _assert_step_rejected(results, -5, work_untouched=False)


# ---------------------------------------------------------------------------
# O9: initial alpha (2.5)
# ---------------------------------------------------------------------------

ALPHA_CASES = [
    (0.02, 0, 2.0, 5, 1e-4), (0.02, 1, 2.0, 1, 1e-4), (0.02, 1, 2.0, 0, 1e-4),
    (0.02, 1, 2.0, 2, 1e-4), (0.02, 1, 2.0, 2, 0.5), (0.02, 1, 2.0, 2, 0.01),
    (0.02, 1, 3.0, 7, 3.3e-3), (0.02, 1, 2.0, 2, 0.0), (0.02, 1, 2.0, 2, -1e-3),
    (0.02, 1, 2.0, 2, math.nan), (0.02, 1, 2.0, 2, math.inf), (5.864293e-03, 1, 2.0, 3, 2.9e-4),
    (1.0, 1, 1.0, 4, 0.75),
]


def test_o9_initial_alpha_all_branches(harness: Path, tmp_path: Path) -> None:
    tail = b"".join(struct.pack("=fifif", e, a, c, k, p) for e, a, c, k, p in ALPHA_CASES)
    results = _run(harness, tmp_path, op=5, nsteps=len(ALPHA_CASES), tail=tail)
    got = results[0]["arrays"][0][1][1:len(ALPHA_CASES) + 1]
    for value, (e, a, c, k, p) in zip(got, ALPHA_CASES):
        want = ref.initial_alpha(e, a, c, k, p)
        assert ref32.f32_bits(value) == ref32.f32_bits(want), ((e, a, c, k, p), value, want)


# ---------------------------------------------------------------------------
# Driver (real bin/denise, 32x32 physical-Q SH-FWI case)
# ---------------------------------------------------------------------------

DRIVER_N = 32
DRIVER_CELLS = DRIVER_N * DRIVER_N
DRIVER_FW = 6
SCALED_EPS = 0.01
# Receiver columns of the fixture (x = 100..220 m, DH = 10 m, REFREC 0).
RECEIVER_X_MIN, RECEIVER_X_MAX = 10.0, 22.0
SOURCE_CELL = (16, 16)       # (column, row), 1-based, x = y = 160 m
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
    ("OPT_ALPHA_ADAPT_FACTOR", None),
)


def _write_parameter(path: Path, *, records: int, scaling: object = 0, diag: object = 0,
                     precond: object = 0, precond_eps: object = 0.005, adapt: object = 0,
                     factor: object = 2.0, **fixture: object) -> None:
    """Records 1-119 via the frozen fixture, then trailing records 120..``records``."""
    assert 127 <= records <= 131
    write_sh_parameter(path, **fixture)
    text = path.read_text(encoding="ascii")
    values = {"OPT_STEP_SCALING": scaling, "OPT_DIAG_OUT": diag, "OPT_PRECOND": precond,
              "OPT_PRECOND_EPS": precond_eps, "OPT_ALPHA_ADAPT": adapt,
              "OPT_ALPHA_ADAPT_FACTOR": factor}
    extra = []
    for name, default in TRAILING_RECORDS[:records - 119]:
        extra.append(f"({name}) = {values.get(name, default) if default is None else default}\n")
    path.write_text(text + "".join(extra), encoding="ascii", newline="\n")


def _write_workflow(path: Path, *, pro: float = 0.01) -> None:
    values = [pro, 0, 0, 20, 2, 0, 1, 0, 0, 0, 0, 0, 0,
              0, 0.5, 0.5, 0, 2, 0, 0, 0, 0.1, 0, 0, 10, 1, 1, 0, 0, 0]
    path.write_text(f"{WORKFLOW_HEADER}\n{' '.join(str(v) for v in values)}\n",
                    encoding="ascii")


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


def _build_case(root: Path, denise_binary: Path, mpiexec: str, receivers: str) -> dict:
    """32x32 viscoelastic SH case: Vs 2000 with a slow patch, Q=80, observed data."""
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
    (root / "source/one.dat").write_text("1\n160 0 160 0 20 1 0 1\n", encoding="ascii")
    (root / "receiver/line.dat").write_text(receivers, encoding="ascii")
    workflow = root / "workflow.inp"
    _write_workflow(workflow)
    forward = root / "true_forward.inp"
    write_sh_parameter(forward)
    run = _run_denise(denise_binary, mpiexec, root, forward, workflow)
    assert run.returncode == 0, run.stdout
    start = root / "model/start"
    return {"root": root, "workflow": workflow, "start": start,
            "start_bytes": {s: start.with_suffix(s).read_bytes()
                            for s in (".vs", ".rho", ".qs")},
            "binary": denise_binary, "mpiexec": mpiexec}


@pytest.fixture(scope="module")
def driver_case(tmp_path_factory: pytest.TempPathFactory, denise_binary: Path,
                mpiexec: str) -> dict:
    return _build_case(tmp_path_factory.mktemp("sh_precond_case"), denise_binary, mpiexec,
                       "100 160\n120 160\n140 160\n180 160\n200 160\n220 160\n")


@pytest.fixture(scope="module")
def single_column_case(tmp_path_factory: pytest.TempPathFactory, denise_binary: Path,
                       mpiexec: str) -> dict:
    """Two receivers in one column (x = 140 m): x_min == x_max for mode 2."""
    return _build_case(tmp_path_factory.mktemp("sh_precond_single_column"), denise_binary,
                       mpiexec, "140 160\n140 200\n")


def _fwi(case: dict, name: str, *, records: int, itermax: int = 1, nprocx: int = 1,
         nprocy: int = 1, model_prefix: str = "model/start", eps_scale: float = SCALED_EPS,
         **options) -> tuple[subprocess.CompletedProcess[str], str]:
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
                      case["workflow"], ranks=nprocx * nprocy)
    return run, output


def _files(case: dict, output: str, iteration: int, suffixes) -> dict[str, Path]:
    prefix = f"{case['root'] / output}_stage_1_it_{iteration}"
    return {s: Path(f"{prefix}.{s}") for s in suffixes}


MODEL_SUFFIXES = ("vs", "rho", "qs")
DIAG_SUFFIXES = ("grad_vs", "grad_rho", "grad_qs", "step_vs", "step_rho", "step_qs")
PRECOND_SUFFIXES = ("energy", "precond")


def _log_lines(run: subprocess.CompletedProcess[str]) -> list[str]:
    return [line.strip() for line in run.stdout.splitlines()
            if re.search(r"opteps_vp|L2t\[1\]|epst1\[1\]", line)]


def _driver_mask() -> list[list[float]]:
    return ref32.update_mask(DRIVER_N, DRIVER_N, 0, 0, 1, 1, DRIVER_FW, 0, 0)


def _closed_form_receiver_term() -> list[list[float]]:
    """R of contract 2.2 in global grid coordinates (x = i, z = j, 1-based)."""
    return [[ref32.f32(math.asinh((RECEIVER_X_MAX - (i + 1)) / (j + 1))
                       - math.asinh((RECEIVER_X_MIN - (i + 1)) / (j + 1)))
             for i in range(DRIVER_N)] for j in range(DRIVER_N)]


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
@pytest.mark.parametrize("scaling", [0, 1])
def test_o1_controls_off_are_bitwise_identical_to_127_records(driver_case: dict,
                                                              scaling: int) -> None:
    runs = {}
    for records, extra in ((127, {}), (131, {"precond": 0, "precond_eps": 0.25, "adapt": 0,
                                             "factor": 3.5})):
        run, output = _fwi(driver_case, f"o1_s{scaling}_r{records}", records=records,
                           itermax=2, scaling=scaling, diag=1,
                           eps_scale=SCALED_EPS if scaling else 1.0e6, **extra)
        assert run.returncode == 0, run.stdout
        runs[records] = (run, output, (driver_case["root"] / "log/misfit").read_bytes())
    for iteration in (1, 2):
        for suffix in MODEL_SUFFIXES + DIAG_SUFFIXES:
            left = _files(driver_case, runs[127][1], iteration, [suffix])[suffix]
            right = _files(driver_case, runs[131][1], iteration, [suffix])[suffix]
            assert left.read_bytes() == right.read_bytes(), (iteration, suffix)
        for suffix in PRECOND_SUFFIXES:
            assert not _files(driver_case, runs[131][1], iteration, [suffix])[suffix].exists()
    assert _log_lines(runs[127][0]) == _log_lines(runs[131][0])
    assert runs[127][2] == runs[131][2]
    assert "Exact SH initial alpha" not in runs[131][0].stdout


# O2 / O5 / O10 ---------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("precond,nprocx", [(1, 1), (2, 1), (2, 2)],
                         ids=["source", "receiver_line", "receiver_line_two_ranks"])
def test_o2_o5_driver_precond_does_not_touch_gradient_and_obeys_closed_form(
        driver_case: dict, precond: int, nprocx: int) -> None:
    base, base_out = _fwi(driver_case, f"o2_off_{precond}_{nprocx}", records=131,
                          nprocx=nprocx, scaling=1, diag=1, precond=0)
    assert base.returncode == 0, base.stdout
    pre, pre_out = _fwi(driver_case, f"o2_on_{precond}_{nprocx}", records=131,
                        nprocx=nprocx, scaling=1, diag=1, precond=precond,
                        precond_eps=0.005)
    assert pre.returncode == 0, pre.stdout
    # O2: raw gradients bitwise unchanged; printed J_base equal; the direction changes.
    for suffix in ("grad_vs", "grad_rho", "grad_qs"):
        assert _files(driver_case, base_out, 1, [suffix])[suffix].read_bytes() == \
            _files(driver_case, pre_out, 1, [suffix])[suffix].read_bytes(), suffix
    l2 = lambda run: [l for l in _log_lines(run) if "L2t[1]" in l]
    assert l2(base)[0].split("L2t[2]")[0] == l2(pre)[0].split("L2t[2]")[0]
    assert _files(driver_case, base_out, 1, ["step_vs"])["step_vs"].read_bytes() != \
        _files(driver_case, pre_out, 1, ["step_vs"])["step_vs"].read_bytes()
    # O5: written P and p obey the closed forms computed from written inputs.
    files = _files(driver_case, pre_out, 1, DIAG_SUFFIXES + PRECOND_SUFFIXES)
    for path in files.values():
        assert path.stat().st_size == 4 * DRIVER_CELLS
        assert list(path.parent.glob(f"{path.name}.*.*")) == []
    mask = _driver_mask()
    energy = _to_grid(_read_native(files["energy"]))
    receiver = _closed_form_receiver_term() if precond == 2 else None
    p_got = _to_grid(_read_native(files["precond"]))
    p_want = ref.precond(precond, 0.005, energy, receiver, mask)
    for j in range(DRIVER_N):
        for i in range(DRIVER_N):
            if mask[j][i] == 0.0:
                assert p_got[j][i] == 0.0
            else:
                assert ref32.ulp_distance_f32(p_got[j][i], p_want[j][i]) <= ULP_TOLERANCE, \
                    (j, i, p_got[j][i], p_want[j][i])
    gradients = {n: _to_grid(_read_native(files[f"grad_{s}"]))
                 for n, s in zip(CLASSES, MODEL_SUFFIXES)}
    bases = {n: _to_grid(_read_native(driver_case["start"].with_suffix(f".{s}")))
             for n, s in zip(CLASSES, MODEL_SUFFIXES)}
    want = ref.preconditioned_step(1, dict(zip(CLASSES, (1.0, 1.0, 1.0))), gradients,
                                   bases, mask, p_got)
    for n, s in zip(CLASSES, MODEL_SUFFIXES):
        got = _to_grid(_read_native(files[f"step_{s}"]))
        for j in range(DRIVER_N):
            for i in range(DRIVER_N):
                if mask[j][i] == 0.0 or want[n][j][i] == 0.0:
                    assert got[j][i] == 0.0
                else:
                    assert ref32.ulp_distance_f32(got[j][i], want[n][j][i]) <= ULP_TOLERANCE
    # O10: evidence only, reported without assertion (contract section 5).
    masked = [p_got[j][i] for j in range(DRIVER_N) for i in range(DRIVER_N) if mask[j][i]]
    source_p = p_got[SOURCE_CELL[1] - 1][SOURCE_CELL[0] - 1]
    print(f"O10 precond={precond} nprocx={nprocx}: P(source)={source_p:.4e} "
          f"masked mean={sum(masked) / len(masked):.4e} "
          f"below_mean={source_p < sum(masked) / len(masked)}")


@pytest.mark.integration
def test_o5_no_precond_diagnostics_without_diag_out(driver_case: dict) -> None:
    run, output = _fwi(driver_case, "o5_no_diag", records=131, scaling=1, diag=0, precond=2)
    assert run.returncode == 0, run.stdout
    for suffix in PRECOND_SUFFIXES + DIAG_SUFFIXES:
        assert not _files(driver_case, output, 1, [suffix])[suffix].exists(), suffix
    assert _files(driver_case, output, 1, ["vs"])["vs"].exists()


# O8 (driver) -----------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("precond", [1, 2])
def test_o8_driver_decompositions_agree(driver_case: dict, precond: int) -> None:
    outputs = {}
    for nprocx, nprocy in ((1, 1), (2, 1), (1, 2)):
        run, output = _fwi(driver_case, f"o8_p{precond}_{nprocx}x{nprocy}", records=131,
                           nprocx=nprocx, nprocy=nprocy, scaling=1, diag=1,
                           precond=precond)
        assert run.returncode == 0, run.stdout
        outputs[(nprocx, nprocy)] = output
    for suffix in PRECOND_SUFFIXES + ("step_vs", "step_rho", "step_qs"):
        serial = _read_native(_files(driver_case, outputs[(1, 1)], 1, [suffix])[suffix])
        scale = max(abs(v) for v in serial)
        assert scale > 0.0, suffix
        for key in ((2, 1), (1, 2)):
            other = _read_native(_files(driver_case, outputs[key], 1, [suffix])[suffix])
            worst = max(abs(a - b) for a, b in zip(serial, other))
            assert worst <= RELATIVE_TOLERANCE * scale, (suffix, key, worst, scale)


# O3 (driver) -------------------------------------------------------------------

@pytest.mark.integration
def test_o3_trial_runs_do_not_leak_into_energy(driver_case: dict) -> None:
    first, first_out = _fwi(driver_case, "o3_two", records=131, itermax=2, scaling=1,
                            diag=1, precond=1)
    assert first.returncode == 0, first.stdout
    restart_prefix = f"{first_out}_stage_1_it_1"
    second, second_out = _fwi(driver_case, "o3_restart", records=131, itermax=1, scaling=1,
                              diag=1, precond=1, model_prefix=restart_prefix)
    assert second.returncode == 0, second.stdout
    a = _files(driver_case, first_out, 2, ["energy"])["energy"].read_bytes()
    b = _files(driver_case, second_out, 1, ["energy"])["energy"].read_bytes()
    assert a == b
    e1 = _to_grid(_read_native(_files(driver_case, first_out, 1, ["energy"])["energy"]))
    assert all(v >= 0.0 for row in e1 for v in row)
    assert max(v for row in e1 for v in row) > 0.0


# O9 (driver): initial alpha, configuration aborts, record order ----------------

ADAPT_EPS = 0.05
ADAPT_FACTOR = 3.0


def _initial_alphas(run) -> list[tuple[int, float]]:
    return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(
        r"Exact SH initial alpha: iteration (\d+) alpha_start (\S+)", run.stdout)]


@pytest.mark.integration
def test_o9_driver_adaptive_initial_alpha_reaches_the_line_search(driver_case: dict) -> None:
    adaptive, adaptive_out = _fwi(driver_case, "o9_adapt_on", records=131, itermax=3,
                                  scaling=1, precond=2, adapt=1, factor=ADAPT_FACTOR,
                                  eps_scale=ADAPT_EPS)
    assert adaptive.returncode == 0, adaptive.stdout
    fixed, fixed_out = _fwi(driver_case, "o9_adapt_off", records=131, itermax=3,
                            scaling=1, precond=2, adapt=0, factor=ADAPT_FACTOR,
                            eps_scale=ADAPT_EPS)
    assert fixed.returncode == 0, fixed.stdout
    starts = _initial_alphas(adaptive)
    selected = [float(m.group(1)) for m in re.finditer(r"opteps_vp = (\S+)", adaptive.stdout)]
    assert [k for k, _ in starts] == [1, 2, 3]
    assert len(selected) == 3
    assert _relative_close(starts[0][1], ADAPT_EPS)
    branch_min = False
    for (k, start), previous in zip(starts[1:], selected[:-1]):
        expected = min(ADAPT_EPS, ADAPT_FACTOR * previous)
        assert _relative_close(start, expected, tol=2.0e-6), (k, start, expected)
        branch_min |= ADAPT_FACTOR * previous < ADAPT_EPS * (1.0 - 1.0e-5)
    assert branch_min, "the min branch of contract 2.5 was not exercised"
    assert "Exact SH initial alpha" not in fixed.stdout
    # Wiring: iteration 1 is identical; iteration 2 differs exactly when alpha_start
    # of iteration 2 is below EPS_SCALE.
    for suffix in MODEL_SUFFIXES:
        assert _files(driver_case, adaptive_out, 1, [suffix])[suffix].read_bytes() == \
            _files(driver_case, fixed_out, 1, [suffix])[suffix].read_bytes(), suffix
    second_differs = _files(driver_case, adaptive_out, 2, ["vs"])["vs"].read_bytes() != \
        _files(driver_case, fixed_out, 2, ["vs"])["vs"].read_bytes()
    second_below = starts[1][1] < ADAPT_EPS * (1.0 - 1.0e-5)
    assert second_differs == second_below, (starts, selected)


CONFIG_ABORTS = {
    "precond_mode": ({"precond": 3, "scaling": 1}, 1),
    "precond_negative": ({"precond": -1, "scaling": 1}, 1),
    "precond_without_scaling": ({"precond": 1, "scaling": 0}, 1),
    "precond_without_scaling_two_ranks": ({"precond": 2, "scaling": 0}, 2),
    "eps_zero": ({"precond": 1, "scaling": 1, "precond_eps": 0.0}, 1),
    "eps_large": ({"precond": 1, "scaling": 1, "precond_eps": 1.5}, 1),
    "eps_nan": ({"precond": 1, "scaling": 1, "precond_eps": "nan"}, 1),
    "eps_inf_two_ranks": ({"precond": 1, "scaling": 1, "precond_eps": "inf"}, 2),
    "eps_off_still_validated": ({"precond": 0, "scaling": 0, "precond_eps": -1.0}, 1),
    "adapt_mode": ({"adapt": 2}, 1),
    "factor_small": ({"adapt": 1, "factor": 0.5}, 1),
    "factor_nan": ({"adapt": 1, "factor": "nan"}, 1),
    "factor_inf_two_ranks": ({"adapt": 1, "factor": "inf"}, 2),
}


@pytest.mark.integration
@pytest.mark.parametrize("name", sorted(CONFIG_ABORTS))
def test_o9_configuration_aborts(driver_case: dict, name: str) -> None:
    options, ranks = CONFIG_ABORTS[name]
    run, output = _fwi(driver_case, f"o9_abort_{name}", records=131, nprocx=ranks,
                       **dict(options))
    _assert_configuration_abort(run, driver_case, output)


@pytest.mark.integration
def test_o9_receiver_line_mode_requires_two_receiver_columns(
        single_column_case: dict) -> None:
    # Positive control: the same case runs with the source-energy mode.
    run, output = _fwi(single_column_case, "o9_single_column_mode1", records=131,
                       scaling=1, precond=1)
    assert run.returncode == 0, run.stdout
    assert _files(single_column_case, output, 1, ["vs"])["vs"].exists()
    for ranks in (1, 2):
        run, output = _fwi(single_column_case, f"o9_single_column_mode2_{ranks}",
                           records=131, nprocx=ranks, scaling=1, precond=2)
        _assert_configuration_abort(run, single_column_case, output)


@pytest.mark.integration
@pytest.mark.parametrize("records", [128, 129, 130])
def test_o9_trailing_records_are_read_in_order(driver_case: dict, records: int) -> None:
    """A file with 128..130 records carries the first of 128..131; the rest default."""
    run, output = _fwi(driver_case, f"o9_records_{records}", records=records, scaling=1,
                       diag=1, precond=1, precond_eps=0.5, adapt=1, factor=1.0)
    assert run.returncode == 0, run.stdout
    assert ("Exact SH initial alpha" in run.stdout) == (records >= 130)
    full, full_out = _fwi(driver_case, f"o9_records_full_{records}", records=131, scaling=1,
                          diag=1, precond=1,
                          precond_eps=0.5 if records >= 129 else 0.005,
                          adapt=1 if records >= 130 else 0, factor=2.0)
    assert full.returncode == 0, full.stdout
    for suffix in ("precond", "step_vs", "vs"):
        assert _files(driver_case, output, 1, [suffix])[suffix].read_bytes() == \
            _files(driver_case, full_out, 1, [suffix])[suffix].read_bytes(), suffix
