"""SH-MASK-01 unit level: the non-collective public API of
src/SH/visco_sh_exact_update_mask.c through a small mpicc harness
(tests/utilities/sh_mask_01_harness.c) against the independent reference
tests/utilities/sh_mask_01_reference.py (contract sections 3 and 4).

These tests need no DENISE binary and no MPI launcher, so they run in the
non-physics selection of the hosted CI.  The driver-level tests of the
package stay in tests/test_sh_mask_01.py (integration).

  U1  contract identity, API in header and Makefile
  U2  mask file: AND with the CPML mask on several decompositions; rejections
      with the mask untouched
  U3  source taper: fixed square, sin^2 ramp, minimum over sources, rounding
      of the source cell; rejections with nothing written
  U4  metric product D = T * P bitwise; rejections with the metric untouched
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tests.utilities import sh_mask_01_reference as ref

CONTRACT_PATH = "docs/sh_mask_01_contract.md"
CONTRACT_SHA256 = "d1fe8d2f644550c121154876884d4f0956513c3baf90ef4e81920d073fbf6d6f"
NXG, NYG = 12, 9                       # non-square, so a transposed index is caught
DECOMPOSITIONS = [(1, 1), (2, 1), (1, 3), (3, 3), (4, 3)]
TAPER_SENTINEL = -12345.5
SIN2_RELATIVE_TOLERANCE = 1.0e-6       # 0 < T < 1 only; T = 0 and T = 1 are exact


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory, repository_root: Path) -> Path:
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the SH-MASK-01 unit harness"
    output = tmp_path_factory.mktemp("sh_mask_01_unit") / "harness"
    command = [compiler, "-std=c99", "-O2", "-fcommon", "-I", str(repository_root / "include"),
               str(repository_root / "tests/utilities/sh_mask_01_harness.c"),
               str(repository_root / "src/SH/visco_sh_exact_update_mask.c"),
               "-o", str(output), "-lm"]
    run = subprocess.run(command, cwd=repository_root, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert run.returncode == 0, run.stdout
    return output


def _call(harness: Path, workdir: Path, *, op: int, nx: int, ny: int, pos_x: int = 0,
          pos_y: int = 0, nxg: int = NXG, nyg: int = NYG, tfile: str = "", sources=(),
          filtsize: int = 0, fault: int = 0, dh: float = 1.0, radius: float = 1.0,
          fields=()) -> tuple[int, bool, list[list[list[float]]]]:
    workdir.mkdir(parents=True, exist_ok=True)
    name = tfile.encode()
    assert len(name) < 256
    payload = struct.pack("=10i", op, nx, ny, pos_x, pos_y, nxg, nyg, len(sources), filtsize, fault)
    payload += struct.pack("=2f", dh, radius) + name + bytes(256 - len(name))
    for sx, sy in sources:
        payload += struct.pack("=2f", sx, sy)
    for local in fields:
        assert len(local) == ny and all(len(row) == nx for row in local)
        payload += struct.pack(f"={nx * ny}f", *(v for row in local for v in row))
    case, result = workdir / "case.bin", workdir / "result.bin"
    case.write_bytes(payload)
    run = subprocess.run([str(harness), str(case), str(result)], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert run.returncode == 0, run.stdout
    raw = result.read_bytes()
    status, halo, nfields = struct.unpack("=3i", raw[:12])
    count = nx * ny
    assert len(raw) == 12 + 4 * nfields * count
    values = struct.unpack(f"={nfields * count}f", raw[12:])
    out = [[list(values[k * count + j * nx:k * count + (j + 1) * nx]) for j in range(ny)]
           for k in range(nfields)]
    return status, bool(halo), out


def _cut(global_field, nx, ny, px, py):
    """Local [j][i] block of a global field indexed [J-1][I-1]."""
    return [row[px * nx:(px + 1) * nx] for row in global_field[py * ny:(py + 1) * ny]]


def _bits(values) -> list[int]:
    return [struct.unpack("=I", struct.pack("=f", v))[0] for row in values for v in row]


def _cpml(width: int = 1) -> list[list[float]]:
    return [[0.0 if (x < width or x >= NXG - width or y >= NYG - width) else 1.0
             for x in range(NXG)] for y in range(NYG)]


SAME = object()


def _write_mask(base: Path, u_values, rho_values=SAME) -> str:
    """Writes <base>_u.bin and <base>_rho.bin; None leaves a file out."""
    base.parent.mkdir(parents=True, exist_ok=True)
    rho_values = u_values if rho_values is SAME else rho_values
    for suffix, values in (("_u.bin", u_values), ("_rho.bin", rho_values)):
        if values is not None:
            Path(str(base) + suffix).write_bytes(struct.pack(f"={len(values)}f", *values))
    return str(base)


def _file_pattern(seed: int) -> list[float]:
    """Global x-outer file values: random 0/1 with an asymmetric block."""
    rng = random.Random(seed)
    values = []
    for x in range(NXG):
        for y in range(NYG):
            values.append(0.0 if (rng.random() < 0.3 or (x >= 8 and y <= 2)) else 1.0)
    return values


# ----------------------------------------------------------------------------- U1

def test_u1_contract_identity(repository_root: Path) -> None:
    digest = hashlib.sha256((repository_root / CONTRACT_PATH).read_bytes()).hexdigest()
    assert digest == CONTRACT_SHA256


def test_u1_api_declared_in_header_and_makefile(repository_root: Path) -> None:
    header = re.sub(r"\s+", " ", (repository_root / "include/fd.h").read_text(encoding="utf-8"))
    for pattern in (r"struct visco_sh_exact_mask_report \{",
                    r"int visco_sh_exact_apply_mask_file\(",
                    r"int visco_sh_exact_build_source_taper\(",
                    r"int visco_sh_exact_apply_taper_metric\(",
                    r"int visco_sh_exact_mask_change_report\("):
        assert re.search(pattern, header), pattern
    assert "visco_sh_exact_update_mask.c" in (repository_root / "src/Makefile").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------- U2

@pytest.mark.parametrize("nprocx,nprocy", DECOMPOSITIONS)
@pytest.mark.parametrize("seed", [1, 2])
def test_u2_mask_file_and_cpml_on_decompositions(harness: Path, tmp_path: Path, seed: int,
                                                 nprocx: int, nprocy: int) -> None:
    values = _file_pattern(seed)
    tfile = _write_mask(tmp_path / "mask" / "m", values)
    status, grid = ref.read_global_mask(tfile, NXG, NYG)
    assert status == ref.OK
    cpml = _cpml()
    expected = ref.apply_global_mask(cpml, grid)
    assert expected != cpml                                  # the file fixes extra cells
    nx, ny = NXG // nprocx, NYG // nprocy
    for py in range(nprocy):
        for px in range(nprocx):
            status, halo, (mask,) = _call(harness, tmp_path / f"r{px}{py}", op=1, nx=nx, ny=ny,
                                          pos_x=px, pos_y=py, tfile=tfile,
                                          fields=[_cut(cpml, nx, ny, px, py)])
            assert status == 0 and halo
            assert _bits(mask) == _bits(_cut(expected, nx, ny, px, py)), (px, py)


def _bad_files(base: Path, case: str) -> str:
    good = _file_pattern(3)
    if case == "missing_u":
        return _write_mask(base, None, good)
    if case == "missing_rho":
        return _write_mask(base, good, None)
    if case == "short":
        return _write_mask(base, good[:-1])
    if case == "long":
        return _write_mask(base, good + [1.0])
    if case == "transposed_size":                            # NYG x (NXG+1) values
        return _write_mask(base, [1.0] * (NYG * (NXG + 1)))
    if case in ("half", "two", "nan", "minus_one"):
        bad = list(good)
        bad[17] = {"half": 0.5, "two": 2.0, "nan": math.nan, "minus_one": -1.0}[case]
        return _write_mask(base, bad)
    if case in ("u_value", "rho_value"):                     # a bad value in one file only
        bad = list(good)
        bad[5] = 0.25
        return _write_mask(base, bad, good) if case == "u_value" else _write_mask(base, good, bad)
    if case == "differ":
        bad = list(good)
        bad[40] = 1.0 - bad[40]
        return _write_mask(base, good, bad)
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["missing_u", "missing_rho", "short", "long", "transposed_size",
                                  "half", "two", "nan", "minus_one", "u_value", "rho_value",
                                  "differ"])
def test_u2_mask_file_rejections_leave_the_mask_untouched(harness: Path, tmp_path: Path,
                                                          case: str) -> None:
    tfile = _bad_files(tmp_path / "mask" / "m", case)
    expected_status, _ = ref.read_global_mask(tfile, NXG, NYG)
    assert expected_status != ref.OK
    cpml = _cpml()
    nx, ny = NXG // 2, NYG // 3
    local = _cut(cpml, nx, ny, 1, 1)
    status, halo, (mask,) = _call(harness, tmp_path / "run", op=1, nx=nx, ny=ny, pos_x=1, pos_y=1,
                                  tfile=tfile, fields=[local])
    assert status == expected_status and halo
    assert _bits(mask) == _bits(local)


@pytest.mark.parametrize("change", ["null_mask", "null_row", "null_tfile", "pos_x", "pos_y",
                                    "nxg_small", "negative_pos"])
def test_u2_mask_file_invalid_arguments(harness: Path, tmp_path: Path, change: str) -> None:
    tfile = _write_mask(tmp_path / "mask" / "m", _file_pattern(4))
    nx, ny = NXG // 2, NYG // 3
    local = _cut(_cpml(), nx, ny, 0, 0)
    kwargs = dict(op=1, nx=nx, ny=ny, tfile=tfile, fields=[local])
    kwargs.update({"null_mask": dict(fault=1), "null_row": dict(fault=2), "null_tfile": dict(fault=3),
                   "pos_x": dict(pos_x=2), "pos_y": dict(pos_y=3), "nxg_small": dict(nxg=nx - 1),
                   "negative_pos": dict(pos_x=-1)}[change])
    status, halo, (mask,) = _call(harness, tmp_path / "run", **kwargs)
    assert status == ref.BAD_ARGUMENTS and halo
    assert _bits(mask) == _bits(local)


# ----------------------------------------------------------------------------- U3

TAPER_CASES = {
    # name: (dh, radius, filtsize, sources)
    "one_source_dh10": (10.0, 45.0, 1, [(55.0, 42.0)]),
    "half_cell_rounding": (10.0, 30.0, 0, [(35.0, 25.0)]),
    "two_sources_min": (10.0, 40.0, 1, [(30.0, 30.0), (90.0, 60.0)]),
    "radius_equals_dh": (10.0, 10.0, 0, [(60.0, 40.0)]),
    "filtsize_2_dh_0p1": (0.1, 0.5, 2, [(0.45, 0.35)]),
    "source_outside_grid": (10.0, 50.0, 1, [(-20.0, 130.0), (60.0, 50.0)]),
}


@pytest.mark.parametrize("nprocx,nprocy", DECOMPOSITIONS)
@pytest.mark.parametrize("name", sorted(TAPER_CASES))
def test_u3_source_taper_matches_the_reference(harness: Path, tmp_path: Path, name: str,
                                               nprocx: int, nprocy: int) -> None:
    dh, radius, filtsize, sources = TAPER_CASES[name]
    taper = ref.global_taper(NXG, NYG, sources, dh, radius, filtsize)
    flat = [v for row in taper for v in row]
    if name not in ("source_outside_grid",):
        assert 0.0 in flat                                   # a fixed square exists
    if name != "radius_equals_dh":
        assert any(0.0 < v < 1.0 for v in flat)              # the sin^2 ramp is exercised
    cpml = _cpml()
    expected_mask = [[0.0 if taper[y][x] == 0.0 else cpml[y][x] for x in range(NXG)]
                     for y in range(NYG)]
    nx, ny = NXG // nprocx, NYG // nprocy
    for py in range(nprocy):
        for px in range(nprocx):
            status, halo, (got_taper, got_mask) = _call(
                harness, tmp_path / f"r{px}{py}", op=2, nx=nx, ny=ny, pos_x=px, pos_y=py, dh=dh,
                radius=radius, filtsize=filtsize, sources=sources,
                fields=[_cut(cpml, nx, ny, px, py)])
            assert status == 0 and halo
            want = _cut(taper, nx, ny, px, py)
            for j in range(ny):
                for i in range(nx):
                    t, r = got_taper[j][i], want[j][i]
                    if r in (0.0, 1.0):
                        assert t == r, (name, px, py, i, j, t, r)
                    else:
                        assert abs(t - r) <= SIN2_RELATIVE_TOLERANCE * r, (name, i, j, t, r)
            assert _bits(got_mask) == _bits(_cut(expected_mask, nx, ny, px, py))


def test_u3_source_cell_rounding_is_half_up_on_the_float_quotient() -> None:
    """Reference self-check of the contract rule iround(x_s/DH)."""
    assert ref.source_cell(35.0, 10.0) == 4 and ref.source_cell(25.0, 10.0) == 3
    assert ref.source_cell(34.9, 10.0) == 3 and ref.source_cell(-20.0, 10.0) == -2


TAPER_REJECTIONS = {
    "radius_below_dh": dict(radius=9.99),
    "negative_filtsize": dict(filtsize=-1),
    "zero_dh": dict(dh=0.0),
    "negative_dh": dict(dh=-10.0),
    "nan_dh": dict(dh=math.nan),
    "inf_radius": dict(radius=math.inf),
    "nan_radius": dict(radius=math.nan),
    "no_source": dict(sources=[]),
    "nan_source": dict(sources=[(30.0, 30.0), (math.nan, 20.0)]),
    "inf_source": dict(sources=[(30.0, math.inf)]),
    "null_mask": dict(fault=1),
    "null_row": dict(fault=2),
    "null_srcpos": dict(fault=3),
}


@pytest.mark.parametrize("name", sorted(TAPER_REJECTIONS))
def test_u3_taper_rejections_write_nothing(harness: Path, tmp_path: Path, name: str) -> None:
    args = dict(dh=10.0, radius=30.0, filtsize=1, sources=[(30.0, 30.0)])
    args.update(TAPER_REJECTIONS[name])
    if "fault" not in args:
        assert not ref.taper_arguments_valid(args["sources"], args["dh"], args["radius"],
                                             args["filtsize"])
    nx, ny = NXG // 2, NYG // 3
    local = _cut(_cpml(), nx, ny, 1, 0)
    status, halo, (got_taper, got_mask) = _call(harness, tmp_path / "run", op=2, nx=nx, ny=ny,
                                                pos_x=1, pos_y=0, fields=[local], **args)
    assert status == ref.BAD_ARGUMENTS and halo
    assert all(v == TAPER_SENTINEL for row in got_taper for v in row)
    assert _bits(got_mask) == _bits(local)


# ----------------------------------------------------------------------------- U4

def _metric_fields(seed: int, nx: int, ny: int):
    rng = random.Random(seed)
    special = [0.0, -0.0, 1.0, 1.0e-42, -3.5e-39, 3.0e38, -1.0e-30]
    metric = [[ref.f32(rng.choice(special) if rng.random() < 0.2 else
                       rng.uniform(-1.0, 1.0) * 10.0 ** rng.uniform(-20, 20))
               for _ in range(nx)] for _ in range(ny)]
    taper = [[ref.f32(rng.choice([0.0, 1.0, rng.random(), 1.0e-7]))
              for _ in range(nx)] for _ in range(ny)]
    return metric, taper


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_u4_metric_is_the_float32_product(harness: Path, tmp_path: Path, seed: int) -> None:
    nx, ny = 7, 5
    metric, taper = _metric_fields(seed, nx, ny)
    expected = [[ref.metric_product(taper[j][i], metric[j][i]) for i in range(nx)]
                for j in range(ny)]
    assert expected != metric
    status, halo, (got,) = _call(harness, tmp_path / "run", op=3, nx=nx, ny=ny,
                                 fields=[metric, taper])
    assert status == 0 and halo
    assert _bits(got) == _bits(expected)


@pytest.mark.parametrize("fault", [1, 2, 3])
def test_u4_metric_rejections_leave_the_metric_untouched(harness: Path, tmp_path: Path,
                                                         fault: int) -> None:
    nx, ny = 7, 5
    metric, taper = _metric_fields(9, nx, ny)
    status, halo, (got,) = _call(harness, tmp_path / "run", op=3, nx=nx, ny=ny, fault=fault,
                                 fields=[metric, taper])
    assert status == ref.BAD_ARGUMENTS and halo
    assert _bits(got) == _bits(metric)
