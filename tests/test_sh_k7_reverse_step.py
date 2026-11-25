"""K-7: bitwise speed-up of the exact viscoelastic SH reverse step
(docs/sh_k7_reverse_speedup_contract.md; criteria fixed before candidate
output).

The production reverse step is compared with the frozen base step
(tests/utilities/sh_k7_reference, git blob a13d88a0 of b24161a1) over several
consecutive steps.  Every float and double of the adjoint state, the source
cotangent and the material output must be bitwise equal, with -0.0/+0.0
cells, a zero block, duplicate receiver and source cells, CPML, free surface,
1/2/2x2 ranks, FD 2..12, subdomains down to one cell (dense path) and two
cells (cell-kernel path) wide, different x/y relaxation weights and CPML
profiles, a state that is -0.0 in every word, and the material capture
without plan, with the per-shot plan, and with the plan accumulating into
the shot sum.  Injected non-finite or invalid inputs make both
implementations fail.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

REFERENCE = "tests/utilities/sh_k7_reference/visco_sh_full_state_adjoint_step_base.c"
REFERENCE_BLOB = "a13d88a0c28775b5f46aa85615c66013fdd502f2"
STEPS = 4
SOURCES = [
    "tests/utilities/sh_k7_reverse_step_harness.c",
    "tests/utilities/sh_k7_reference/k7_reference_step.c",
    "src/SH/visco_sh_full_state_adjoint_step.c",
    "src/SH/update_v_PML_SH.c",
    "src/SH/update_s_visc_PML_SH.c",
    "src/SH/exchange_v_SH.c",
    "src/SH/exchange_s_SH.c",
    "src/SH/surface_elastic_SH.c",
    "src/SH/visco_sh_gsls_vjp.c",
    "src/SH/visco_sh_material_vjp.c",
    "src/SH/visco_sh_material_timestep_vjp.c",
    "src/SH/visco_sh_material_observable.c",
    "src/SH/update_s_visc_PML_SH_adjoint.c",
    "src/SH/update_v_PML_SH_adjoint.c",
    "src/SH/exchange_v_SH_adjoint.c",
    "src/SH/exchange_s_SH_adjoint.c",
    "src/SH/surface_elastic_SH_adjoint.c",
]
# name, npx, npy, boundary, free surface, FD order, L, FW, nx, ny
CASES = (
    ("1x1_fd2_l1_fw0", 1, 1, 0, 0, 2, 1, 0, 10, 12),
    ("1x1_fd4_l4_fw3_fs", 1, 1, 0, 1, 4, 4, 3, 10, 12),
    ("1x1_fd12_l1_fw2", 1, 1, 0, 0, 12, 1, 2, 14, 15),
    ("2x1_fd4_l2_fw2", 2, 1, 0, 0, 4, 2, 2, 9, 11),
    ("1x2_fd6_l3_fw2_fs", 1, 2, 0, 1, 6, 3, 2, 10, 9),
    ("2x2_fd8_l2_fw3", 2, 2, 0, 0, 8, 2, 3, 9, 10),
    ("2x2_fd4_l4_fw2_fs_boundary", 2, 2, 1, 1, 4, 4, 2, 8, 9),
    # subdomains one cell wide or high take the dense-patch path
    ("1x1_fd4_l2_fw0_nx1", 1, 1, 0, 0, 4, 2, 0, 1, 6),
    ("2x1_fd2_l1_fw0_ny1_fs", 2, 1, 0, 1, 2, 1, 0, 5, 1),
    # the narrowest subdomains on the cell-kernel path
    ("2x1_fd8_l2_fw2_nx2", 2, 1, 0, 0, 8, 2, 2, 2, 7),
    ("1x2_fd4_l3_fw2_ny2_fs", 1, 2, 0, 1, 4, 3, 2, 6, 2),
    ("2x2_fd2_l1_fw1_nx2_ny2", 2, 2, 0, 0, 2, 1, 1, 2, 2),
)
# every adjoint state word and receiver cotangent -0.0 (valid input)
NEGZERO_CASES = (CASES[1], CASES[5], CASES[10], CASES[11])
INJECTIONS = ("nan_sxz", "inf_receiver", "nan_halo_vz", "nan_qsum", "neg_mu")


@pytest.fixture(scope="module", params=["-O3", "-O2"])
def k7_harness(request, tmp_path_factory, repository_root: Path):
    compiler = shutil.which("mpicc")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert compiler and launcher
    executable = tmp_path_factory.mktemp("sh_k7") / f"sh_k7{request.param}"
    command = [compiler, "-std=c99", request.param, "-fcommon",
               "-I", str(repository_root / "include"), "-I", str(repository_root / "src/SH")]
    command.extend(str(repository_root / source) for source in SOURCES)
    command.extend(["-o", str(executable), "-lm"])
    result = subprocess.run(command, cwd=repository_root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert result.returncode == 0, result.stdout
    return launcher, executable


def _run(k7_harness, case, inject="none"):
    launcher, executable = k7_harness
    _, npx, npy, boundary, fs, fd, mechanisms, fw, nx, ny = case
    command = [launcher, "--oversubscribe", "-n", str(npx * npy), str(executable),
               str(npx), str(npy), str(boundary), str(fs), str(fd), str(mechanisms), str(fw),
               str(nx), str(ny), str(STEPS), inject]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, result.stdout
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]


def test_reference_is_the_frozen_base_step(repository_root: Path) -> None:
    data = (repository_root / REFERENCE).read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    assert blob == REFERENCE_BLOB


def _assert_bitwise(records, case) -> None:
    assert [r["variant"] for r in records] == [0, 1, 2, 3]
    ranks = case[1] * case[2]
    for record in records:
        assert record["status_ref"] == 0 and record["status_cand"] == 0, record
        assert record["steps_compared"] == STEPS * ranks, record
        assert record["state_words_diff"] == 0, record
        assert record["signal_diff"] == 0, record
        assert record["native_diff"] == 0, record


@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
def test_reverse_step_is_bitwise_equal_to_the_base(k7_harness, case) -> None:
    records = _run(k7_harness, case)
    _assert_bitwise(records, case)
    print("K7-STEP " + case[0] + " " + json.dumps(records))


@pytest.mark.parametrize("case", NEGZERO_CASES, ids=lambda case: case[0])
def test_all_negative_zero_state_is_bitwise_equal(k7_harness, case) -> None:
    records = _run(k7_harness, case, "all_negzero")
    _assert_bitwise(records, case)
    print("K7-NEGZERO " + case[0] + " " + json.dumps(records))


@pytest.mark.parametrize("case,inject",
                         [(CASES[1], inject) for inject in INJECTIONS] +
                         [(CASES[3], inject) for inject in INJECTIONS[:3]],
                         ids=lambda value: value if isinstance(value, str) else value[0])
def test_invalid_inputs_fail_in_both(k7_harness, case, inject) -> None:
    """One rank: fixed and material variants.  Two ranks: the fixed variant
    only, because a failing distributed material step aborts collectively."""
    records = _run(k7_harness, case, inject)
    assert records, inject
    for record in records:
        assert record["failed"] == 3, record
    print("K7-INJECT " + case[0] + " " + json.dumps(records))
