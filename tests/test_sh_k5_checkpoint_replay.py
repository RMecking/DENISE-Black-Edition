"""K-5: checkpoint/replay of the exact viscoelastic SH forward history
(docs/sh_k5_checkpoint_replay_contract.md, evidence section 3a; criteria
fixed before candidate output).

C1  every output of exact SH FWI runs is bytewise equal for full storage
    (SH_EXACT_SEGMENTS 0) and segmented storage (several S, automatic S),
    over FD order, mechanisms, free surface, decompositions, S-2 operators,
    INV_STF = 1, L-BFGS with two stages and reload, and two shots;
C3  verify mode: 0 mismatches, every replayed record compared;
C6  with full storage a history larger than the node memory stops before
    iteration 1 with the record-135 message; values < -1 are rejected.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest

from tests.test_sh_s2_data_operators import (  # noqa: F401  (fixtures are used by name)
    MODES, NS, WORKFLOW_HEADER, _misfit_rows, _parameter as _s2_parameter, harness, s2_case)
from tests.test_sh_s3_stf import _row, two_shot_case  # noqa: F401


def _parameter(path: Path, *, segments: int, fdorder: int = 4, mechanisms: int = 1,
               free_surface: int = 0, nprocy: int = 1, time: float | None = None,
               testshots2: bool = False, **kwargs) -> None:
    _s2_parameter(path, **kwargs)
    text = path.read_text(encoding="ascii")
    changes = [("(FD_ORDER) = 4\n", f"(FD_ORDER) = {fdorder}\n"),
               ("(L) = 1\n", f"(L) = {mechanisms}\n"),
               ("(FREE_SURF) = 0\n", f"(FREE_SURF) = {free_surface}\n"),
               ("(NPROCY) = 1\n", f"(NPROCY) = {nprocy}\n")]
    if mechanisms > 1:
        freqs = " ".join(str(5.0 * 2 ** l) for l in range(mechanisms))
        changes.append(("(FL) = 20.0\n", f"(FL) = {freqs}\n"))
    if time is not None:
        changes.append(("(TIME) = 0.2\n", f"(TIME) = {time}\n"))
    if testshots2:
        changes.append(("(TESTSHOTS) = 1, 1, 1\n", "(TESTSHOTS) = 1, 2, 1\n"))
    for old, new in changes:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    text += "(STF_LENGTH) = -1.0\n" + f"(SH_EXACT_SEGMENTS) = {segments}\n"
    path.write_text(text, encoding="ascii", newline="\n")


def _run(case: dict, tag: str, rows: list[str], *, segments: int, ranks_x: int = 1,
         ranks_y: int = 1, itermax: int = 1, lbfgs: int = 0, model: str = "model/start",
         verify: bool = False, **kwargs):
    root = case["root"]
    workflow = root / f"wfk5_{tag}.inp"
    workflow.write_text(WORKFLOW_HEADER + "\n" + "\n".join(rows) + "\n", encoding="ascii")
    parameter = root / f"k5_{tag}.inp"
    _parameter(parameter, segments=segments, mode=1, model=model, output=f"model/{tag}",
               nprocx=ranks_x, nprocy=ranks_y, itermax=itermax, lbfgs=lbfgs, **kwargs)
    misfit = root / "log/misfit"
    if misfit.exists():
        misfit.unlink()
    env = dict(os.environ)
    if verify:
        env["DENISE_SH_EXACT_SEGMENTS_VERIFY"] = "1"
    run = subprocess.run([case["mpiexec"], "--oversubscribe", "-n", str(ranks_x * ranks_y),
                          str(case["binary"]), str(parameter), str(workflow)],
                         cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         timeout=1200, env=env)
    return run


def _outputs(case: dict, tag: str) -> dict[str, bytes]:
    root = case["root"]
    files = {p.name.replace(tag, "TAG", 1): p.read_bytes()
             for p in (root / "model").glob(f"{tag}_*") if p.is_file()}
    files["misfit"] = (root / "log/misfit").read_bytes()
    return files


def _exact_lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines()
            if ln.startswith("Exact SH") and not ln.startswith(("Exact SH replay", "Exact SH history"))]


def _compare(case: dict, name: str, rows: list[str], segments_list, **kw):
    full = _run(case, f"{name}_full", rows, segments=0, **kw)
    assert full.returncode == 0 and "MPI_ABORT" not in full.stdout, full.stdout[-3000:]
    ref = _outputs(case, f"{name}_full")
    assert len(ref) > 3
    for s in segments_list:
        tag = f"{name}_s{'auto' if s == -1 else s}"
        run = _run(case, tag, rows, segments=s, **kw)
        assert run.returncode == 0 and "MPI_ABORT" not in run.stdout, run.stdout[-3000:]
        got = _outputs(case, tag)
        assert sorted(got) == sorted(ref), (s, sorted(set(got) ^ set(ref)))
        differing = [k for k in ref if ref[k] != got[k]]
        assert not differing, (s, differing)
        assert _exact_lines(run.stdout) == _exact_lines(full.stdout)
        replay = re.findall(r"Exact SH replay: segments (\d+) steps (\d+)", run.stdout)
        assert replay, "no replay line"
        if s > 0:
            assert int(replay[0][0]) == min(s, NS)
        history = re.search(r"Exact SH history: SH_EXACT_SEGMENTS (-?\d+) segments (\d+)", run.stdout)
        assert history and int(history.group(1)) == s
        print(f"K5-C1 {name} S={s} segments={replay[0][0]} files={len(got)} bytewise equal")


ROW = _row(pro=0.0, inv_stf=0)


@pytest.mark.integration
def test_segments_bitwise_equal_fd4_l1(s2_case: dict) -> None:
    _compare(s2_case, "c1a", [ROW], (1, 2, 7, -1, NS))


@pytest.mark.integration
def test_segments_bitwise_equal_fd2_l4_free_surface(s2_case: dict) -> None:
    _compare(s2_case, "c1b", [ROW], (7, -1), fdorder=2, mechanisms=4, free_surface=1)


@pytest.mark.integration
def test_segments_bitwise_equal_free_surface_fd4_fd8(s2_case: dict) -> None:
    """Free surface with FD >= 4: the stress stencil reads the mirrored vz row,
    so a replay without the surface velocity operator would differ (review H1)."""
    _compare(s2_case, "c1i", [ROW], (5, -1), fdorder=4, mechanisms=4, free_surface=1,
             ranks_x=2)
    _compare(s2_case, "c1j", [ROW], (3, -1), fdorder=8, mechanisms=2, free_surface=1,
             ranks_x=2, ranks_y=2)


@pytest.mark.integration
def test_segments_bitwise_equal_fd8_two_ranks(s2_case: dict) -> None:
    _compare(s2_case, "c1c", [ROW], (3, -1), fdorder=8, ranks_x=2)


@pytest.mark.integration
def test_segments_bitwise_equal_two_by_two_ranks(s2_case: dict) -> None:
    _compare(s2_case, "c1d", [ROW], (-1,), ranks_x=2, ranks_y=2)


@pytest.mark.integration
def test_segments_bitwise_equal_s2_operators_and_stf(s2_case: dict) -> None:
    rows = [_row(pro=0.0, inv_stf=1, **MODES["combined"])]
    _compare(s2_case, "c1e", rows, (5, -1), ranks_x=2)


@pytest.mark.integration
def test_segments_bitwise_equal_lbfgs_stages_and_reload(s2_case: dict) -> None:
    rows = [_row(pro=0.0, inv_stf=0, time_filt=1, fc_high=fc, order=4) for fc in (8.0, 14.0)]
    _compare(s2_case, "c1f", rows, (4, -1), itermax=2, lbfgs=1)
    _compare(s2_case, "c1g", [rows[1]], (-1,), lbfgs=1, model="model/c1f_full_stage_1_it_2")


@pytest.mark.integration
def test_segments_bitwise_equal_two_shots(two_shot_case: dict) -> None:
    _compare(two_shot_case, "c1h", [_row(pro=0.0, inv_stf=1, **MODES["lowpass"])], (6, -1),
             testshots2=True)


@pytest.mark.integration
def test_verify_mode_finds_no_mismatch(s2_case: dict) -> None:
    run = _run(s2_case, "c3", [ROW], segments=7, verify=True)
    assert run.returncode == 0 and "MPI_ABORT" not in run.stdout, run.stdout[-3000:]
    lines = re.findall(r"Exact SH replay: segments (\d+) steps (\d+) max_segment \d+ .* "
                       r"replayed_steps (\d+) replay_seconds \S+ verify (\d) compared (\d+) mismatches (\d+)",
                       run.stdout)
    assert lines
    for segments, steps, replayed, verify, compared, mismatches in lines:
        assert verify == "1" and int(mismatches) == 0
        # the last segment stays in memory from the forward run; every other
        # segment (steps 0 .. first[S-1]-1, first[k] = floor(k N / S)) is replayed
        s, n = int(segments), int(steps)
        assert int(replayed) == ((s - 1) * n) // s and int(compared) == 3 * int(replayed)
    print(f"K5-C3 lines={len(lines)} {lines[0]}")


@pytest.mark.integration
def test_full_storage_beyond_node_memory_stops_before_iteration(s2_case: dict) -> None:
    """C6: 3 x 33 x 33 floats x NT with TIME 4000 s (8e6 steps) is about 100 GiB."""
    run = _run(s2_case, "c6", [ROW], segments=0, time=4000.0)
    assert run.returncode != 0 or "MPI_ABORT was invoked" in run.stdout
    assert "SH_EXACT_SEGMENTS" in run.stdout and "parameter record 135" in run.stdout
    assert "TDFWI ITERATION" not in run.stdout


@pytest.mark.integration
def test_invalid_segment_value_is_rejected(s2_case: dict) -> None:
    run = _run(s2_case, "c6b", [ROW], segments=-2)
    assert run.returncode != 0 or "MPI_ABORT was invoked" in run.stdout
    assert "SH_EXACT_SEGMENTS (parameter record 135) must be" in run.stdout
    assert "TDFWI ITERATION" not in run.stdout
