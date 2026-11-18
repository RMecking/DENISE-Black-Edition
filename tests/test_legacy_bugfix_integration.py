"""Focused regressions for the pinned DENISE legacy bugfix group."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from tests.utilities.legacy_bugfix.extract_fd_source_regions import write_fragments
from tests.utilities.legacy_bugfix.science_oracle import read_science
from tests.utilities.legacy_bugfix.su_reference import fixture as writer_fixture

TEST_DIR = Path(__file__).parent
HARNESS = TEST_DIR / "utilities/legacy_bugfix"
FIXTURES = TEST_DIR / "fixtures/legacy_bugfix"
PINS = json.loads((HARNESS / "reference-pins.json").read_text(encoding="ascii"))
WRITER_CASES = json.loads((HARNESS / "cases.json").read_text(encoding="ascii"))
WRITER_FAILURES = (
    "zero", "over-local", "over-global", "over-vector", "over-source",
    "stride-zero", "mode", "short-input", "time-overflow", "interval-overflow",
    "open", "header", "payload", "close",
)
WRITERS = ("outseis.c", "outseis_glob.c", "outseis_vector.c", "output_source_signal.c")
SH_SOURCES = (
    "src/wavelet.c", "src/rd_sour.c", "src/splitsrc.c", "src/splitrec.c",
    "src/SH/update_v_PML_SH.c", "src/SH/update_s_elastic_PML_SH.c", "src/seismo_ssg.c",
)
DRIVERS = (
    "src/saveseis.c", "src/saveseis_glob.c", "src/SH/saveseis_glob_SH.c",
    "src/AC/FD_AC.c", "src/AC/grad_obj_ac.c", "src/PSV/FD_PSV.c",
    "src/PSV/grad_obj_psv.c", "src/SH/FD_SH.c", "src/SH/grad_obj_sh.c",
    "src/AC/FWI_AC.c", "src/PSV/FWI_PSV.c", "src/SH/FWI_SH.c", "src/SH/FWI_SH_visc.c",
)
MEMORY_FAILURES = {
    "bad-row": "invalid Numerical Recipes array bounds",
    "bad-column": "invalid Numerical Recipes array bounds",
    "bad-depth": "invalid Numerical Recipes array bounds",
    "bad-values": "invalid Numerical Recipes array bounds",
    "bad-product": "Numerical Recipes array extent overflow",
    "bad-bytes": "Numerical Recipes array storage overflow",
}
OWNERSHIP = (
    ("base", "single", "signals=1/0 geometry=1/0 shots=1"),
    ("base", "simultaneous", "signals=1/0 geometry=1/0 shots=1"),
    ("base", "repeat", "signals=12/0 geometry=6/0 shots=12"),
    ("base", "leak-probe", "signals=2/0 geometry=2/0 shots=2"),
    ("candidate", "single", "signals=1/1 geometry=1/1 shots=1"),
    ("candidate", "simultaneous", "signals=1/1 geometry=1/1 shots=1"),
    ("candidate", "empty", "signals=1/1 geometry=0/0 shots=1"),
    ("candidate", "repeat", "signals=12/12 geometry=6/6 shots=12"),
)
WRAP_SOURCE = "-Wl,--wrap=splitsrc,--wrap=wavelet,--wrap=free_matrix"
SANITIZER_FLAGS = (
    "-g", "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-fno-pie", "-no-pie",
)
SANITIZERS = ("-fsanitize=address,undefined", *SANITIZER_FLAGS)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_command(argv, cwd, *, kind="test", sanitizer=None, expected=0, timeout=None):
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_THREAD_LIMIT="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               ASAN_OPTIONS=f"detect_leaks={int(sanitizer != 'reader')}:halt_on_error=1:exitcode=85",
               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1",
               LSAN_OPTIONS="use_stacks=0:use_registers=0" if sanitizer == "cleanup" else "",
               OMPI_MCA_ess_singleton_isolated="1")
    result = subprocess.run(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=timeout or (60 if kind == "build" else 30),
    )
    assert result.returncode == expected, (
        f"{argv}: exit {result.returncode}, expected {expected}\n{result.stdout}"
    )
    return result.stdout


@pytest.fixture(scope="module")
def compiler():
    command = shutil.which("mpicc")
    assert command, "DENISE focused C regression requires mpicc"
    return command


@pytest.fixture(scope="module")
def reference_contract():
    assert len(WRITER_CASES) == 13 and sum(case["legacy_compare"] for case in WRITER_CASES) == 7
    fixture_names = {p.name for p in FIXTURES.iterdir() if p.is_file() and not p.name.startswith(".")}
    assert fixture_names == PINS["fixtures"].keys()
    for name, expected in PINS["fixtures"].items():
        assert sha256((FIXTURES / name).read_bytes()) == expected, name
    for name, expected in PINS["case_contract"].items():
        assert sha256((HARNESS / name).read_bytes()) == expected, name


@pytest.fixture(scope="module")
def historical_tree(repository_root, tmp_path_factory):
    """Read exact baseline sources from the history or a verified private export."""
    supplied_export = os.environ.get("DENISE_BUGFIX_BASE_SOURCE")
    target = tmp_path_factory.mktemp("bugfix-base")
    for relative, expected in PINS["historical_sources"].items():
        if supplied_export:
            payload = (Path(supplied_export) / relative).read_bytes()
        else:
            result = subprocess.run(
                ["git", "cat-file", "blob", f"{PINS['base']}:{relative}"],
                cwd=repository_root, capture_output=True, timeout=10,
            )
            assert result.returncode == 0, f"Missing pinned base {relative}: {result.stderr!r}"
            payload = result.stdout
        assert sha256(payload) == expected, relative
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return target


def compile_harness(compiler, directory, includes, sources, *, flags=(), name="check"):
    binary = directory / name
    run_command(
        [compiler, "-std=c99", "-O2", "-fcommon", "-ffp-contract=off",
         *(f"-I{include}" for include in includes), *flags, *sources, "-o", binary, "-lm"],
        directory, kind="build",
    )
    return binary


def check_output(actual, reference, count, groups):
    data = actual.read_bytes()
    assert len(data) == count * 4 and data == reference.read_bytes(), actual
    values = struct.unpack(f"<{count}f", data)
    assert all(math.isfinite(x) for x in values)
    assert all(any(values[start:end]) for start, end in groups)


def test_frozen_fixture_material(reference_contract):
    assert sum(case["traces"] * case["samples"] for case in WRITER_CASES) == 364637


def test_active_driver_syntax(compiler, repository_root, historical_tree, tmp_path):
    run_command(
        [compiler, "-std=c99", "-O2", "-fcommon", "-ffp-contract=off",
         f"-I{repository_root / 'include'}", "-fsyntax-only",
         *(repository_root / source for source in DRIVERS), historical_tree / "src/SH/FD_SH.c"],
        tmp_path, kind="build",
    )


@pytest.fixture(scope="module", params=("canary", "asan"))
def su_reader_binary(request, compiler, repository_root, tmp_path_factory):
    mode = request.param
    directory = tmp_path_factory.mktemp(f"su-reader-{mode}")
    flags = ("-O1", "-fsanitize=address", *SANITIZER_FLAGS) if mode == "asan" else ()
    binary = compile_harness(compiler, directory, [repository_root / "include"],
                             [HARNESS / "su_reader.c", repository_root / "src/inseis.c"],
                             flags=flags)
    return mode, binary


@pytest.fixture(scope="module")
def reader_inputs(tmp_path_factory, reference_contract):
    directory = tmp_path_factory.mktemp("su-reader-inputs")
    for name in PINS["fixtures"]:
        if name.startswith("fixture_y.su.shot"):
            shutil.copyfile(FIXTURES / name, directory / name)
    return directory


@pytest.mark.parametrize("shot", range(1, 10))
def test_su_reader_bounds(su_reader_binary, reader_inputs, shot):
    mode, binary = su_reader_binary
    output = run_command([binary, mode, str(shot)], reader_inputs,
                         sanitizer="reader" if mode == "asan" else None,
                         expected=0 if shot <= 3 else 42)
    marker = "SU OK" if shot <= 3 else "EXPECTED SU ERROR"
    assert output.strip() == f"{marker} {shot}"


def test_su_sample_one_scientific_delta(compiler, repository_root, reader_inputs, tmp_path):
    sources = (HARNESS / "science-sources.txt").read_text(encoding="ascii").splitlines()
    binary = compile_harness(
        compiler, tmp_path, [repository_root / "include", TEST_DIR / "utilities"],
        [HARNESS / "su_sh_delta.c", repository_root / "src/inseis.c",
         *(repository_root / name for name in sources)],
    )
    for shot in (21, 22):
        name = f"fixture_y.su.shot{shot}"
        shutil.copyfile(reader_inputs / name, tmp_path / name)
    run_command([binary, "bridge_m1_physical"], tmp_path, timeout=22)
    output = tmp_path / "science.bin"
    read_science(output)
    assert output.read_bytes() == (FIXTURES / "science.bin").read_bytes()


@pytest.fixture(scope="module")
def writer_inputs(reference_contract, tmp_path_factory):
    directory = tmp_path_factory.mktemp("su-writer-fixtures")
    hashes = PINS["writer_recipe_artifacts"]
    generated = set()
    for case in WRITER_CASES:
        for suffix, contents in zip((".input", ".expected"), writer_fixture(case)):
            name = case["name"] + suffix
            assert sha256(contents) == hashes[name], name
            (directory / name).write_bytes(contents)
            generated.add(name)
    assert generated == hashes.keys()
    return directory


@pytest.fixture(scope="module")
def writer_binaries(compiler, repository_root, historical_tree, tmp_path_factory):
    directory = tmp_path_factory.mktemp("su-writer-build")
    result = {}
    for mode in ("old", "normal", "asan"):
        tree = historical_tree if mode == "old" else repository_root
        flags = (("-fsanitize=address,undefined,float-cast-overflow", *SANITIZER_FLAGS)
                 if mode == "asan" else ())
        result[mode] = compile_harness(
            compiler, directory, [repository_root / "include"],
            [HARNESS / "su_writer.c", repository_root / "src/inseis.c",
             *(tree / "src" / name for name in WRITERS)], flags=flags, name=mode,
        )
    return result


def writer_records(case, actual, independent, *, historical):
    count, traces = case["samples"], case["traces"]
    stride = 240 + 4 * count
    assert len(actual) == len(independent) == traces * stride
    if not historical:
        assert actual == independent, case["name"]
    header = ((slice(0, 4), slice(28, 30), slice(114, 118), slice(180, 184))
              if case["producer"] == "SOURCE" else (slice(0, 210),))
    for row in range(traces):
        start = row * stride
        actual_row, expected_row = actual[start:start+stride], independent[start:start+stride]
        assert struct.unpack_from("<H", actual_row, 114)[0] == count
        assert actual_row[240:] == expected_row[240:]
        for field in header:
            assert actual_row[field] == expected_row[field]


@pytest.mark.parametrize("mode", ("old", "normal", "asan"))
def test_su_writer_records(writer_binaries, writer_inputs, tmp_path, mode):
    shutil.copytree(writer_inputs, tmp_path / "fixtures")
    output = run_command([writer_binaries[mode], "historical" if mode == "old" else mode, mode],
                         tmp_path, sanitizer="writer" if mode == "asan" else None)
    cases = [case for case in WRITER_CASES if mode != "old" or case["legacy_compare"]]
    assert [line for line in output.splitlines() if line.startswith("PASS ")] == [
        "PASS " + case["name"] for case in cases
    ]
    for case in cases:
        name = case["name"]
        writer_records(case, (tmp_path / f"{mode}-{name}_y.su.shot1").read_bytes(),
                       (writer_inputs / f"{name}.expected").read_bytes(), historical=mode == "old")


@pytest.mark.parametrize("failure", WRITER_FAILURES)
def test_su_writer_errors(writer_binaries, tmp_path, failure):
    log = run_command([writer_binaries["asan"], "negative", failure], tmp_path,
                      sanitizer="writer", expected=42)
    assert log.startswith("EXPECTED: ") and len(log.splitlines()) == 1, log


@pytest.fixture(scope="module")
def memory_binaries(compiler, repository_root, historical_tree, tmp_path_factory):
    directory = tmp_path_factory.mktemp("legacy-memory-build")
    sh = [repository_root / source for source in SH_SOURCES]
    sh.append(repository_root / "src/SH/alloc_seisSHfwi.c")
    binaries = {}
    for mode, tree in (("base", historical_tree), ("candidate", repository_root)):
        binaries[mode] = compile_harness(
            compiler, directory, [tree / "include"],
            [HARNESS / "memory_sh.c", tree / "src/util.c", *sh],
            flags=("-DMEMORY_SH_STANDALONE",), name=mode,
        )
    binaries["sanitized"] = compile_harness(
        compiler, directory, [repository_root / "include"],
        [HARNESS / "memory_cases.c", HARNESS / "memory_sh.c", repository_root / "src/util.c", *sh],
        flags=SANITIZERS, name="sanitized",
    )
    return binaries


@pytest.mark.parametrize("case", ("history", "alloc", *MEMORY_FAILURES, "empty-sh", "positive-sh"))
def test_legacy_history_and_allocators(memory_binaries, reference_contract, tmp_path, case):
    diagnostic = MEMORY_FAILURES.get(case)
    log = run_command([memory_binaries["sanitized"], case], tmp_path,
                      sanitizer="memory", expected=42 if diagnostic else 0)
    if diagnostic:
        assert diagnostic in log.splitlines(), log
    else:
        expected = {"history": "PASS history (103 assertions)", "alloc": "PASS alloc (194 assertions)",
                    "empty-sh": "PASS SH empty", "positive-sh": "PASS SH positive"}
        assert expected[case] in log.splitlines(), log
    if case == "positive-sh":
        check_output(tmp_path / "sh-sanitized.bin", FIXTURES / "fd2-single.bin", 510,
                     ((0, 10), (10, 30), (30, 510)))


@pytest.mark.parametrize("mode", ("base", "candidate"))
def test_legacy_allocator_fp32_reference(memory_binaries, reference_contract, tmp_path, mode):
    file = tmp_path / "positive.bin"
    run_command([memory_binaries[mode], file], tmp_path)
    check_output(file, FIXTURES / "fd2-single.bin", 510, ((0, 10), (10, 30), (30, 510)))


@pytest.fixture(scope="module")
def cleanup_binaries(compiler, repository_root, historical_tree, tmp_path_factory):
    directory = tmp_path_factory.mktemp("fd-sh-build")
    sh = [repository_root / source for source in SH_SOURCES if source != "src/splitrec.c"]
    flags = [*SANITIZERS, WRAP_SOURCE]
    result = {}
    for mode, tree, is_candidate in (
        ("base", historical_tree, False), ("candidate", repository_root, True),
        ("bridge", repository_root, True),
    ):
        extracted = directory / f"fd-{mode}"
        write_fragments(tree / "src/SH/FD_SH.c", extracted, candidate=is_candidate)
        source = "fd_sh_io_bridge.c" if mode == "bridge" else "fd_cleanup.c"
        sources = [HARNESS / source, repository_root / "src/util.c", *sh]
        extra_flags = []
        if mode == "bridge":
            sources += [repository_root / "src/output_source_signal.c", repository_root / "src/inseis.c"]
            extra_flags.append("-Wl,--wrap=output_source_signal")
        result[mode] = compile_harness(
            compiler, directory, [repository_root / "include", extracted], sources,
            flags=[*flags, *extra_flags, f"-DHISTORICAL={int(not is_candidate)}"], name=mode,
        )
    return result


def cleanup_case(binary, case, directory, counts):
    positive = case in ("single", "simultaneous")
    output = directory / f"{case}.bin"
    probe = case == "leak-probe"
    log = run_command([binary, case, output if positive else "-"], directory,
                      sanitizer="cleanup", expected=42 if probe else 0)
    lines = [line.strip() for line in log.splitlines()]
    assert counts in lines, log
    if probe:
        assert "168 byte(s) leaked in 4 allocation(s)" in log
        assert "__wrap_wavelet" in log and "__wrap_splitsrc" in log
    else:
        assert "Sanitizer" not in log and "runtime error:" not in log, log
    if positive:
        source_words = 10 if case == "single" else 30
        count = source_words + 20 + 4 * 12 * 10
        check_output(output, FIXTURES / f"fd2-{case}.bin", count,
                     ((0, source_words), (source_words, source_words + 20), (source_words + 20, count)))
    return lines


@pytest.mark.parametrize("mode,case,counts", OWNERSHIP,
                         ids=[f"{mode}-{case}" for mode, case, _ in OWNERSHIP])
def test_fd_sh_shot_ownership(cleanup_binaries, reference_contract, tmp_path, mode, case, counts):
    cleanup_case(cleanup_binaries[mode], case, tmp_path, counts)


BRIDGE_CASES = [(case, counts) for mode, case, counts in OWNERSHIP if mode == "candidate"]


@pytest.mark.parametrize("case,counts", BRIDGE_CASES, ids=[case for case, _ in BRIDGE_CASES])
def test_fd_sh_su_lifecycle(cleanup_binaries, reference_contract, tmp_path, case, counts):
    lines = cleanup_case(cleanup_binaries["bridge"], case, tmp_path, counts)
    shot_numbers = (1,) if case == "single" else tuple(range(1, 12, 2)) if case == "repeat" else ()
    assert [line for line in lines if line.startswith("SU_ROUNDTRIP ")] == [
        f"SU_ROUNDTRIP shot={shot} words=10" for shot in shot_numbers
    ]

