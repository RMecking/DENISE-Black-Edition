"""Private opt-in FD8 Born J/J^T integration smoke (T1a); not a scientific accuracy gate.

The private FD8 Born J/J^T path (src/PSV/elastic_psv_fd8_*_private.h, enabled only
inside tests/utilities/m9l2a_fd8_state_harness.c via DENISE_M9L2A_PRIVATE) was
landed byte-identically from the Case-3-qualified candidate. Its scientific
evidence is the M9L-RTM-REFERENCE-02 Case-3 record (Stage N, F3, P) transferred by
source and native-output identity, not this test.

This smoke only checks, on a tiny homogeneous grid with CPML off and on, that the
private translation unit builds with the frozen candidate flags, loads, and that
prepare/J/J^T run consistently. Frozen pre-output criterion (G-T5): all statuses 0,
all outputs finite, outputs not identically zero, dot identity
|<Jq,w> - <q,J^T w>| <= 2^-30 (||Jq|| ||w|| + ||q|| ||J^T w||), and failure paths
return nonzero status (or a null handle).
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "tests" / "utilities" / "m9l5p_physical_harness.c"
# Frozen candidate build flags (M9L-RTM-POINT-C3 candidate/build-binding.json).
CANDIDATE_FLAGS = [
    "-std=c99", "-O2", "-g", "-Wall", "-Wextra", "-Werror", "-pedantic",
    "-fno-fast-math", "-ffp-contract=off", "-fexcess-precision=standard",
    "-frounding-math", "-fno-inline",
]
SCENARIOS = ("dot_no_cpml", "dot_cpml", "failure_paths")
DOT_FACTOR = 2.0 ** -30

CLIENT = r'''
import ctypes as C, json, math, sys
lib = C.CDLL(sys.argv[1]); scenario = sys.argv[2]
F = C.POINTER(C.c_float); D = C.POINTER(C.c_double); I = C.POINTER(C.c_int)
lib.m9l5p_create.argtypes = [C.c_int] * 4 + [C.c_float, F] + [C.c_int] * 3 + [I, I]
lib.m9l5p_create.restype = C.c_void_p
lib.m9l3_test_prepare.argtypes = [C.c_void_p, F, C.c_size_t]
lib.m9l3_test_j.argtypes = [C.c_void_p, D, D, D, C.c_size_t]
lib.m9l3_test_jt.argtypes = [C.c_void_p, D, D, D, C.c_size_t]
lib.m9l3_test_destroy.argtypes = [C.POINTER(C.c_void_p)]
lib.m9l3_test_destroy.restype = None
NX, NT, NR = 21, 40, 3
CELLS, COUNT = NX * NX, NT * NR * 2
source = (C.c_float * NT)(*([0.0, 1e-3, 3e-3, 2e-3, -1e-3, -3e-3, -1e-3] + [0.0] * (NT - 7)))
ri = (C.c_int * NR)(6, 10, 14)
rj = (C.c_int * NR)(6, 6, 6)

def create(nx, fw):
    return C.c_void_p(lib.m9l5p_create(nx, nx, NT, fw, 0.125, source, 10, 5, NR, ri, rj))

def lcg(n, seed, scale):
    out, s = [], seed
    for _ in range(n):
        s = (1103515245 * s + 12345) % (1 << 31)
        out.append(scale * (s / float(1 << 30) - 1.0))
    return out

def norm(v):
    return math.sqrt(sum(x * x for x in v))

result = {"scenario": scenario}
if scenario in ("dot_no_cpml", "dot_cpml"):
    h = create(NX, 0 if scenario == "dot_no_cpml" else 4)
    assert h.value, "create failed"
    try:
        bg = (C.c_float * COUNT)()
        result["prepare"] = lib.m9l3_test_prepare(h, bg, COUNT)
        dl = (C.c_double * CELLS)(*lcg(CELLS, 11, 1.0 / 32))
        dm = (C.c_double * CELLS)(*lcg(CELLS, 23, 1.0 / 32))
        d = (C.c_double * COUNT)()
        result["j"] = lib.m9l3_test_j(h, dl, dm, d, COUNT)
        w = (C.c_double * COUNT)(*lcg(COUNT, 37, 1e-3))
        gl, gm = (C.c_double * CELLS)(), (C.c_double * CELLS)()
        result["jt"] = lib.m9l3_test_jt(h, w, gl, gm, CELLS)
    finally:
        lib.m9l3_test_destroy(C.byref(h))
    result["destroyed"] = h.value is None
    vals = {"background": list(bg), "j": list(d), "jt": list(gl) + list(gm)}
    result["finite"] = all(math.isfinite(x) for v in vals.values() for x in v)
    result["nonzero"] = {k: any(x != 0.0 for x in v) for k, v in vals.items()}
    lhs = math.fsum(a * b for a, b in zip(d, w))
    rhs = math.fsum(a * b for a, b in zip(dl, gl)) + math.fsum(a * b for a, b in zip(dm, gm))
    scale = norm(d) * norm(w) + math.sqrt(norm(dl) ** 2 + norm(dm) ** 2) * norm(vals["jt"])
    result.update(lhs=lhs, rhs=rhs, residual=abs(lhs - rhs), scale=scale)
else:
    result["create_small_grid_null"] = not create(8, 0).value
    result["null_handle_j"] = lib.m9l3_test_j(None, None, None, None, 0)
    h = create(NX, 4)
    assert h.value, "create failed"
    try:
        dl = (C.c_double * CELLS)(); dm = (C.c_double * CELLS)()
        d = (C.c_double * COUNT)(); w = (C.c_double * COUNT)()
        gl, gm = (C.c_double * CELLS)(), (C.c_double * CELLS)()
        result["j_unprepared"] = lib.m9l3_test_j(h, dl, dm, d, COUNT)
        result["jt_unprepared"] = lib.m9l3_test_jt(h, w, gl, gm, CELLS)
        bg = (C.c_float * COUNT)()
        result["prepare_short"] = lib.m9l3_test_prepare(h, bg, COUNT - 1)
        result["prepare"] = lib.m9l3_test_prepare(h, bg, COUNT)
        result["j_short"] = lib.m9l3_test_j(h, dl, dm, d, COUNT - 1)
        dl[3] = float("nan")
        result["j_nonfinite"] = lib.m9l3_test_j(h, dl, dm, d, COUNT)
    finally:
        lib.m9l3_test_destroy(C.byref(h))
print(json.dumps(result))
'''


@pytest.fixture(scope="module")
def private_library(tmp_path_factory):
    compiler = shutil.which("gcc")
    assert compiler, "gcc is required for the private FD8 Born J/J^T integration smoke"
    work = tmp_path_factory.mktemp("m9l-fd8-born-private-smoke")
    library = work / "lib_m9l_fd8_born_private.so"
    subprocess.run(
        [compiler, *CANDIDATE_FLAGS, "-I" + str(ROOT / "include"), "-shared", "-fPIC",
         str(HARNESS), "-lm", "-o", str(library)],
        check=True, cwd=ROOT, timeout=300)
    return library


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_private_fd8_born_j_jt_smoke(private_library, scenario):
    proc = subprocess.run([sys.executable, "-I", "-c", CLIENT, str(private_library), scenario],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    r = json.loads(proc.stdout)
    if scenario == "failure_paths":
        assert r["create_small_grid_null"]
        for key in ("null_handle_j", "j_unprepared", "jt_unprepared", "prepare_short", "j_short", "j_nonfinite"):
            assert r[key] != 0, key
        assert r["prepare"] == 0
        return
    assert (r["prepare"], r["j"], r["jt"]) == (0, 0, 0)
    assert r["destroyed"]
    assert r["finite"]
    assert all(r["nonzero"].values()), r["nonzero"]
    assert r["scale"] > 0.0
    assert r["residual"] <= DOT_FACTOR * r["scale"], r
