"""FLUID-3C1: focused native-FP32 reader I/O and collective transaction gates."""
from pathlib import Path
import json
import os
import subprocess
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture
def tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("f")

@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    output = tmp_path_factory.mktemp("elastic-input-build")
    built = {}
    def build(opt="O3"):
        if opt not in built:
            exe = output / opt
            source = os.environ.get("DENISE_ELASTIC_READER_SOURCE",
                                    str(ROOT / "src/PSV/readmod_elastic_PSV.c"))
            subprocess.run([
                "mpicc", "-std=c99", "-" + opt, "-fcommon", "-Wall", "-Wextra", "-Werror",
                "-I" + str(ROOT / "include"),
                str(ROOT / "tests/utilities/legacy_elastic_input_preflight.c"),
                source, "-Wl,--wrap=malloc", "-Wl,--wrap=fopen",
                "-Wl,--wrap=fclose", "-lm", "-o", str(exe),
            ], check=True)
            built[opt] = exe
        return built[opt]
    return build

def model(directory, mode):
    # Short paths respect the unchanged legacy MFILE ABI (74 bytes).
    prefix = directory / "m"
    suffix = (".vp", ".vs", ".rho") if mode == 1 else (".lam", ".mu", ".rho")
    x, y = np.indices((8, 6))
    shear = (1 + x / 8 + y / 16).astype("=f4")
    shear[:3] = 0.0
    shear[0, 0] = -0.0
    primary = (4 + x / 4 + y / 8).astype("=f4")
    if mode == 3:
        primary[6:] = -0.5  # preserve canonical negative-lambda solid envelope
    density = (1 + (x + y) / 16).astype("=f4")
    arrays = (primary, shear, density)
    paths = [Path(str(prefix) + s) for s in suffix]
    for path, array in zip(paths, arrays):
        path.write_bytes(array.tobytes())
    return prefix, paths, arrays

def run(exe, directory, mode, prefix, topology=(1, 1), bad_rank=-1, bad_prefix=None, fail_alloc=-1):
    out = directory / "out"
    out.mkdir(exist_ok=True)
    px, py = topology
    command = ["mpiexec", "--oversubscribe", "-n", str(px * py), str(exe),
               str(mode), str(px), str(py), str(prefix), str(out),
               str(bad_rank), str(bad_prefix or prefix), str(fail_alloc)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    records = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert len(records) == px * py, result.stdout + result.stderr
    return result, records, out

def rejected(result, records):
    assert result.returncode != 0, result.stdout + result.stderr
    assert all(r["rejected"] and r["unchanged_closed"] and r["same_error"] for r in records)
    assert "Elastic model preflight:" in result.stderr

@pytest.mark.parametrize("opt", ["O0", "O3"])
@pytest.mark.parametrize("mode", [1, 3])
@pytest.mark.parametrize("topology", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_valid_raw_bits_ownership_and_output(builds, tmp_path, opt, mode, topology):
    prefix, _, arrays = model(tmp_path, mode)
    result, records, out = run(builds(opt), tmp_path, mode, prefix, topology)
    assert result.returncode == 0, result.stderr
    px, py = topology
    nx, ny = 8 // px, 6 // py
    for record in records:
        rank = record["rank"]
        assert not record["rejected"] and record["halo"] and record["closed"]
        assert record["writes"] == 3 and record["merges"] == (3 if rank == 0 else 0)
        x, y = rank % px, rank // px
        expected = b"".join(a[x*nx:(x+1)*nx, y*ny:(y+1)*ny].copy().tobytes() for a in arrays)
        assert (out / f"{rank}.bin").read_bytes() == expected

@pytest.mark.parametrize("mode", [1, 3])
@pytest.mark.parametrize("field", [0, 1, 2])
@pytest.mark.parametrize("damage", ["missing", "empty", "short_float", "partial_float", "trailing_byte",
                                  "extra_float", "nan", "pos_inf", "neg_inf"])
def test_malformed_files(builds, tmp_path, mode, field, damage):
    prefix, paths, _ = model(tmp_path, mode)
    path = paths[field]
    data = path.read_bytes()
    if damage == "missing":
        path.unlink()
    elif damage == "empty":
        path.write_bytes(b"")
    elif damage == "short_float":
        path.write_bytes(data[:-4])
    elif damage == "partial_float":
        path.write_bytes(data[:-1])
    elif damage == "trailing_byte":
        path.write_bytes(data + b"x")
    elif damage == "extra_float":
        path.write_bytes(data + np.float32(1).tobytes())
    else:
        value = {"nan": np.nan, "pos_inf": np.inf, "neg_inf": -np.inf}[damage]
        path.write_bytes(data[:-4] + np.float32(value).tobytes())
    result, records, _ = run(builds(), tmp_path, mode, prefix)
    rejected(result, records)
    assert str(path) in result.stderr


def replace_cell(paths, arrays, cell, triple):
    for path, array, value in zip(paths, arrays, triple):
        array[cell] = value
        path.write_bytes(array.tobytes())

def accepted(result, records, out, arrays, topology=(1, 1)):
    assert result.returncode == 0, result.stdout + result.stderr
    px, py = topology
    nx, ny = 8 // px, 6 // py
    for record in records:
        rank = record["rank"]
        assert not record["rejected"] and record["halo"] and record["closed"]
        assert record["writes"] == 3 and record["merges"] == (3 if rank == 0 else 0)
        x, y = rank % px, rank // px
        expected = b"".join(a[x*nx:(x+1)*nx, y*ny:(y+1)*ny].copy().tobytes() for a in arrays)
        assert (out / f"{rank}.bin").read_bytes() == expected

TINY = np.nextafter(np.float32(0), np.float32(1))
MAX = np.finfo(np.float32).max
INVALID = [
    (1, "vp_zero", (0.0, 1.0, 1.0)),
    (1, "vp_negative_zero", (-0.0, 1.0, 1.0)),
    (1, "vp_negative", (-1.0, 1.0, 1.0)),
    (1, "vs_negative", (1.0, -1.0, 1.0)),
    (1, "vs_negative_subnormal", (1.0, -TINY, 1.0)),
    (1, "rho_zero", (1.0, 1.0, 0.0)),
    (1, "rho_negative_zero", (1.0, 1.0, -0.0)),
    (1, "rho_negative", (1.0, 1.0, -1.0)),
    (3, "mu_negative", (1.0, -1.0, 1.0)),
    (3, "mu_negative_subnormal", (1.0, -TINY, 1.0)),
    (3, "rho_zero", (1.0, 1.0, 0.0)),
    (3, "rho_negative_zero", (1.0, 1.0, -0.0)),
    (3, "rho_negative", (1.0, 1.0, -1.0)),
    (3, "fluid_lambda_zero", (0.0, 0.0, 1.0)),
    (3, "fluid_lambda_negative_zero", (-0.0, -0.0, 1.0)),
    (3, "fluid_lambda_negative", (-1.0, 0.0, 1.0)),
    (3, "solid_compressional_zero", (-2.0, 1.0, 1.0)),
    (3, "solid_compressional_negative", (-3.0, 1.0, 1.0)),
    (3, "solid_compressional_zero_large", (-MAX, MAX / np.float32(2), 1.0)),
]
VALID = [
    (1, "fluid_positive_zero", (1.0, 0.0, 1.0)),
    (1, "fluid_negative_zero", (1.0, -0.0, 1.0)),
    (1, "solid_derived_negative_lambda", (1.0, 1.0, 1.0)),
    (1, "solid_vs_above_vp", (1.0, 2.0, 1.0)),
    (1, "positive_subnormal_rho", (1.0, 1.0, TINY)),
    (1, "positive_subnormal_vp", (TINY, 0.0, 1.0)),
    (1, "positive_subnormal_vs", (1.0, TINY, 1.0)),
    (3, "fluid_positive_zero", (1.0, 0.0, 1.0)),
    (3, "fluid_negative_zero", (1.0, -0.0, 1.0)),
    (3, "solid_negative_lambda", (-1.0, 1.0, 1.0)),
    (3, "solid_just_positive_compressional", (np.nextafter(np.float32(-2), np.float32(0)), 1.0, 1.0)),
    (3, "solid_subnormal_compressional", (-TINY, TINY, 1.0)),
    (3, "positive_subnormal_rho", (1.0, 1.0, TINY)),
    (3, "fluid_subnormal_lambda", (TINY, 0.0, 1.0)),
    (3, "widened_fp32_intermediate_overflow", (-MAX, MAX, 1.0)),
]

@pytest.mark.parametrize("mode,name,triple", INVALID, ids=[c[1] + "-" + str(c[0]) for c in INVALID])
def test_physical_invalid_before_mutation(builds, tmp_path, mode, name, triple):
    prefix, paths, arrays = model(tmp_path, mode)
    replace_cell(paths, arrays, (7, 5), triple)
    result, records, _ = run(builds(), tmp_path, mode, prefix)
    rejected(result, records)
    assert "x=8 y=6" in result.stderr
    assert "inadmissible material" in result.stderr or "lambda+2mu must be positive" in result.stderr

@pytest.mark.parametrize("opt", ["O0", "O3"])
@pytest.mark.parametrize("mode,name,triple", VALID, ids=[c[1] + "-" + str(c[0]) for c in VALID])
def test_normative_valid_boundary_raw_bits(builds, tmp_path, opt, mode, name, triple):
    prefix, paths, arrays = model(tmp_path, mode)
    replace_cell(paths, arrays, (7, 5), triple)
    result, records, out = run(builds(opt), tmp_path, mode, prefix)
    accepted(result, records, out, arrays)

@pytest.mark.parametrize("mode,triple", [(1, (1.0, -1.0, 1.0)), (3, (-2.0, 1.0, 1.0))])
def test_rank_local_physical_invalid_is_collective(builds, tmp_path, mode, triple):
    prefix, _, _ = model(tmp_path, mode)
    bad = tmp_path / "bad"
    bad.mkdir()
    bad_prefix, paths, arrays = model(bad, mode)
    # Bad rank 3 must reject even though the invalid point belongs to rank 0.
    replace_cell(paths, arrays, (0, 0), triple)
    result, records, _ = run(builds(), tmp_path, mode, prefix, (2, 2),
                             bad_rank=3, bad_prefix=bad_prefix)
    rejected(result, records)
    assert "x=1 y=1" in result.stderr

@pytest.mark.parametrize("topology", [(2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("damage", ["missing", "tail", "nonfinite", "allocation"])
def test_rank_local_failure_is_collective(builds, tmp_path, topology, damage):
    prefix, _, _ = model(tmp_path, 3)
    bad = tmp_path / "bad"
    bad.mkdir()
    bad_prefix, paths, _ = model(bad, 3)
    if damage == "missing":
        paths[0].unlink()
    elif damage == "tail":
        paths[2].write_bytes(paths[2].read_bytes() + b"x")
    elif damage == "nonfinite":
        data = paths[1].read_bytes()
        # First global point belongs to rank 0; failing reader is last rank.
        paths[1].write_bytes(np.float32(np.nan).tobytes() + data[4:])
    rank = topology[0] * topology[1] - 1
    result, records, _ = run(builds(), tmp_path, 3, prefix, topology,
                             bad_rank=rank, bad_prefix=bad_prefix,
                             fail_alloc=rank if damage == "allocation" else -1)
    rejected(result, records)
