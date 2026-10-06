"""FLUID-3C2: actual diagnostic, authenticated BASE, exact quantities before printing.

No propagation, CUDA, or broader physics tests. DENISE_ELASTIC_CHECKFD_BASE_SOURCE
can supply the exact BASE text in Windows-managed WSL worktrees.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = "8a000c1d4cf1ec7912ca3719f19a9e0799a90c4e"
BASE_BLOB = "bb01d94a62ea1c5be75d58e659689c1d3d8ccac7"
TOPOLOGIES = [(1, 1), (2, 1), (1, 2), (2, 2)]
CASES = ["solid", "horizontal", "vertical", "irregular", "fluid", "p_controls", "tiny"]

def bits(value):
    return int(np.float32(value).view(np.uint32))

def from_bits(value):
    return np.uint32(value).view(np.float32).item()

@pytest.fixture(scope="module")
def builders(tmp_path_factory):
    directory = tmp_path_factory.mktemp("elastic-diagnostic-build")
    supplied = os.environ.get("DENISE_ELASTIC_CHECKFD_BASE_SOURCE")
    if supplied:
        baseline = Path(supplied).read_bytes()
    else:
        baseline = subprocess.run(["git", "show", f"{BASE}:src/checkfd_ssg_elastic.c"],
                                  cwd=ROOT, capture_output=True, check=True).stdout
    baseline = baseline.replace(b"\r\n", b"\n")
    actual_blob = hashlib.sha1(b"blob " + str(len(baseline)).encode() + b"\0" + baseline).hexdigest()
    assert actual_blob == BASE_BLOB, "comparison/negative control must use authentic canonical BASE"
    base_source = directory / "base.c"
    base_source.write_bytes(baseline)
    executables = {}
    def build(opt):
        if opt not in executables:
            flags = ["mpicc", "-std=c99", "-" + opt, "-fcommon", "-Wall", "-Wextra", "-Werror",
                     "-Wno-error=maybe-uninitialized",
                     "-U_FORTIFY_SOURCE", "-I" + str(ROOT / "include")]
            obj = directory / f"base-{opt}.o"
            subprocess.run(flags + ["-Dcheckfd_ssg_elastic=checkfd_ssg_elastic_BASE", "-c",
                                    str(base_source), "-o", str(obj)], check=True)
            exe = directory / opt
            subprocess.run(flags + [str(ROOT / "tests/utilities/elastic_propagating_speed_diagnostic.c"),
                                    str(ROOT / "src/checkfd_ssg_elastic.c"), str(obj),
                                    "-Wl,--wrap=fprintf", "-lm", "-o", str(exe)], check=True)
            executables[opt] = exe
        return executables[opt]
    return build

def data(case, mode):
    x, y = np.indices((8, 8))
    vp = (3.0 + 0.5 * ((2*x + y) % 3)).astype("=f4")
    vs = (1.0 + 0.25 * ((x + 2*y) % 4)).astype("=f4")
    rho = (1.0 + ((x + y) % 2)).astype("=f4")
    fluid = np.zeros((8, 8), dtype=bool)
    if case in ("horizontal", "p_controls"):
        fluid = y < 4
    elif case == "vertical":
        fluid = x < 4
    elif case == "irregular":
        fluid = y < 3 + x % 3
    elif case == "fluid":
        fluid[:] = True
    vs[fluid] = 0.0
    vs[fluid & (x % 2 == 0)] = -0.0
    if case == "p_controls":
        vp[0, 0] = 0.5
        vp[7, 7] = 0.75  # valid negative solid lambda in mode 3
    if mode == 1:
        if case == "tiny":
            vs[0, 0] = np.ldexp(np.float32(1), -140)
        primary, shear = vp.copy(), vs.copy()
    else:
        shear = (rho * vs * vs).astype("=f4")
        shear[fluid & (x % 2 == 0)] = -0.0
        primary = (rho * (vp * vp - np.float32(2)*vs*vs)).astype("=f4")
        if case == "tiny":
            rho[0, 0] = 1.0
            shear[0, 0] = np.ldexp(np.float32(1), -140)
            primary[0, 0] = 9.0
            vs[0, 0] = np.ldexp(np.float32(1), -70)
        elif case == "quotient_underflow":
            rho[0, 0] = 2.0
            shear[0, 0] = np.ldexp(np.float32(1), -149)
            primary[0, 0] = 18.0
            vs[0, 0] = np.ldexp(np.float32(1), -75)
        elif case == "positive_base_rounding":
            rho[0, 0] = 23.0
            shear[0, 0] = 1.0
            primary[0, 0] = 205.0  # (lambda+2mu)/rho == 9 exactly
            # Authentic BASE sqrt(FP32(1/23)); widening this positive quotient
            # would yield the adjacent lower FP32 speed, which is forbidden.
            vs[0, 0] = np.uint32(1045791950).view(np.float32)
    if case == "high_speed":
        vp[:] = np.ldexp(np.float32(1), 32)
        vs[:] = np.ldexp(np.float32(1), 31)
        if mode == 1:
            primary, shear = vp.copy(), vs.copy()
        else:
            shear = rho * vs * vs
            primary = rho * (vp * vp - np.float32(2)*vs*vs)
    arrays = np.stack((primary, shear, rho)).astype("=f4")
    return arrays, vp, vs

def run(builders, tmp_path, case, mode, topology, opt):
    arrays, vp, vs = data(case, mode)
    path = tmp_path / "model.bin"
    path.write_bytes(arrays.tobytes())
    px, py = topology
    out = tmp_path / "logs"
    out.mkdir()
    command = ["mpiexec", "--oversubscribe", "-n", str(px*py), str(builders(opt)),
               str(mode), str(px), str(py), str(path), str(out),
               str(2 if case == "high_speed" else int(case in ("tiny", "quotient_underflow")))]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    records = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    assert len(records) == 1, completed.stdout + completed.stderr
    record = records[0]
    assert record["material_unchanged"] == 1
    assert record["base"]["seen"] == record["candidate"]["seen"] == 31
    evidence = os.environ.get("DENISE_FLUID3C2_EVIDENCE")
    if evidence:
        with Path(evidence).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"case": case, "mode": mode, "topology": topology,
                                     "optimization": opt, "quantities": record,
                                     "decoded": {name: {k: from_bits(v) for k, v in q.items() if k.endswith("_bits")}
                                                 for name, q in record.items() if name in ("base", "candidate")},
                                     "command": command,
                                     "root_log": (out / "candidate-0.txt").read_text(),
                                     "base_log": (out / "base-0.txt").read_text()}, sort_keys=True) + "\n")
    return record, out, vp, vs

def verify(record, out, vp, vs, topology, case):
    candidate, base = record["candidate"], record["base"]
    # Prescribed independent phase-speed dataset; no production reduction used.
    positives = vs[vs > 0]
    minimum = min(float(vp.min()), float(positives.min())) if positives.size else float(vp.min())
    assert candidate["cmin_bits"] == bits(minimum)
    assert np.isfinite(from_bits(candidate["cmin_bits"])) and from_bits(candidate["cmin_bits"]) > 0
    assert candidate["dh_bits"] == bits(np.float32(minimum) / np.float32(1.5))
    assert candidate["wavelength_bits"] == bits(np.float32(minimum) / np.float32(0.25))
    assert np.isfinite(from_bits(candidate["dh_bits"])) and from_bits(candidate["dh_bits"]) > 0
    assert candidate["grid_warnings"] == 0
    # BASE maximum selection, stability formula, and passed FD coefficients unchanged.
    assert candidate["cmax_bits"] == base["cmax_bits"]
    assert candidate["dt_bits"] == base["dt_bits"]
    assert candidate["cmax_bits"] == bits(max(float(vp.max()), float(vs.max())))
    if case in ("solid", "tiny", "positive_base_rounding"):
        assert candidate["cmin_bits"] == base["cmin_bits"]
        assert candidate["dh_bits"] == base["dh_bits"]
        assert candidate["wavelength_bits"] == base["wavelength_bits"]
    if case in ("horizontal", "vertical", "irregular", "fluid", "p_controls", "quotient_underflow"):
        # Authentic BASE may retain -0 from signed-zero fluid input.
        # Both signs are exact zero; keep their exact bits in the evidence.
        assert from_bits(base["cmin_bits"]) == 0.0
        assert from_bits(base["dh_bits"]) == 0.0
        assert from_bits(base["wavelength_bits"]) == 0.0
        assert base["grid_warnings"] == 1
    root = (out / "candidate-0.txt").read_text()
    assert "minimum positive propagating phase velocity" in root.lower()
    assert "(of S-waves)" not in root and "Vs_min=" not in root
    assert ("No positive S-wave branch exists globally" in root) == (positives.size == 0)
    assert ("Minimum positive S-wave phase velocity=" in root) == (positives.size > 0)
    # Every local fluid-only tile prints 'none' rather than its reduction sentinel.
    px, py = topology
    nx, ny = 8 // px, 8 // py
    for rank in range(px*py):
        x, y = rank % px, rank // px
        local = vs[x*nx:(x+1)*nx, y*ny:(y+1)*ny]
        text = (out / f"candidate-{rank}.txt").read_text()
        assert ("none" in text) == (not np.any(local > 0))

@pytest.mark.parametrize("opt", ["O0", "O3"])
@pytest.mark.parametrize("mode", [1, 3])
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("case", CASES)
def test_prescribed_propagating_branches(builders, tmp_path, case, mode, topology, opt):
    record, out, vp, vs = run(builders, tmp_path, case, mode, topology, opt)
    verify(record, out, vp, vs, topology, case)

@pytest.mark.parametrize("opt", ["O0", "O3"])
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_approved_positive_mu_quotient_underflow(builders, tmp_path, topology, opt):
    record, out, vp, vs = run(builders, tmp_path, "quotient_underflow", 3, topology, opt)
    verify(record, out, vp, vs, topology, "quotient_underflow")

@pytest.mark.parametrize("opt", ["O0", "O3"])
@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_already_positive_base_rounding_is_authoritative(builders, tmp_path, topology, opt):
    record, out, vp, vs = run(builders, tmp_path, "positive_base_rounding", 3, topology, opt)
    verify(record, out, vp, vs, topology, "positive_base_rounding")
    assert record["candidate"]["cmin_bits"] == record["base"]["cmin_bits"] == 1045791950

@pytest.mark.parametrize("mode", [1, 3])
def test_infinity_sentinel_does_not_clip_physical_minimum(builders, tmp_path, mode):
    record, out, vp, vs = run(builders, tmp_path, "high_speed", mode, (1, 1), "O3")
    verify(record, out, vp, vs, (1, 1), "high_speed")
    assert record["base"]["cmin_bits"] == bits(1e9)
    assert record["candidate"]["cmin_bits"] == bits(np.ldexp(np.float32(1), 31))
