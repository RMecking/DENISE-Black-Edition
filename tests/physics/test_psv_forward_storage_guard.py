"""Legacy forward-wavefield storage on the exact visco-PSV FWI path (review K-4).

alloc_fwiPSV leaves the five forward_prop_* vectors at one element for
MODE == 1 && L > 0; every other MODE/L keeps the full NX*NY*NT length.
psv() refuses an adjoint call (mode 1) without the full storage.  The
production path never reaches that guard, so a harness linked against the
objects of `make denise` (without denise.o) calls psv() directly.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

# Integration test: under --require-denise a missing build (skip) must fail.
pytestmark = pytest.mark.integration

GUARD = "PSV adjoint needs the forward wavefield storage"


@pytest.fixture(scope="module")
def harness(repository_root, tmp_path_factory):
    src = repository_root / "src"
    listing = subprocess.run(["make", "-pn", "-C", str(src), "denise"],
                             capture_output=True, text=True, check=False).stdout
    match = re.search(r"^DENISE\s*:?=\s*(.+)$", listing, re.MULTILINE)
    assert match, "DENISE source list not found in the Makefile"
    objects = [src / Path(name).with_suffix(".o").name for name in match.group(1).split()
               if Path(name).name != "denise.c"]
    missing = [str(o) for o in objects if not o.exists()]
    if missing:
        pytest.skip(f"production objects not built (run make denise): {missing[:3]}")
    out = tmp_path_factory.mktemp("storage-guard") / "harness"
    compiler = os.environ.get("MPICC", "mpicc")
    subprocess.run([compiler, "-O2", "-w", "-fcommon", "-fno-stack-protector",
                    "-D_FORTIFY_SOURCE=0", "-I" + str(repository_root / "include"),
                    "-I" + str(repository_root / "libcseife"),
                    str(repository_root / "tests/utilities/psv_forward_storage_guard_harness.c"),
                    *map(str, objects), "-L" + str(repository_root / "libcseife"),
                    "-lm", "-lcseife", "-lfftw3", "-lstdc++", "-o", str(out)], check=True)
    return out


def call(harness, *args):
    launcher = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
    flags = shlex.split(os.environ.get("MPIEXEC_FLAGS", ""))
    return subprocess.run([*launcher, *flags, "-n", "1", str(harness), *args],
                          capture_output=True, text=True, timeout=60)


def test_adjoint_without_full_storage_fails_closed(harness):
    result = call(harness, "short")
    assert result.returncode != 0, result.stdout
    assert GUARD in result.stdout, result.stdout + result.stderr
    assert "guard did not fire" not in result.stdout


@pytest.mark.parametrize("mode,l_value,reduced", [
    (1, 1, True),    # exact visco-PSV FWI: one element
    (1, 0, False),   # elastic FWI: the exact elastic adjoint reads the storage
    (2, 1, False),   # RTM keeps the full storage
    (2, 0, False),
    (0, 1, False),
])
def test_allocation_condition(harness, mode, l_value, reduced):
    result = call(harness, "length", str(mode), str(l_value))
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["length"] == (1 if reduced else report["full"]), report
