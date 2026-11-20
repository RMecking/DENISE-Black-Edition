"""SH-OPT-SCALE-01 acceptance oracles O1-O9.

Independent oracles written before the implementation from the frozen
contract ``docs/sh_opt_scale_01_contract.md`` revision 2 (SHA-256 below).
Expected values come only from that contract and from the independent
closed-form reference ``tests/utilities/sh_opt_scale_01_reference.py``;
production code is exercised exclusively through its public C API and the
compiled driver.  The d4693e1 optimizer boundary is the frozen O1 reference.

Oracle map (contract section 5):
  O1  mode 0 / NULL bitwise identity       -> test_o1_*
  O2  modes 1/2 closed form (<= 4 ulp)     -> test_o2_*
  O3  descent sign                         -> test_o3_*
  O4  unit invariance (<= 1e-6 rel.)       -> test_o4_*
  O5  MPI 1x1/2x1/1x2 (<= 1e-6 rel.)       -> test_o5_*
  O6  CPML mask + trial state (bitwise)    -> test_o6_*
  O7  driver delayed classes (bitwise)     -> test_o7_*
  O8  error paths -4..-8 and mask -1       -> test_o8_*
  O9  parameter records 119/122/123..127  -> test_o9_*
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
from typing import Callable

import pytest

from tests.utilities import sh_opt_scale_01_reference as ref
from tests.utilities.sh_fwi_fixture import write_sh_parameter


CONTRACT_PATH = "docs/sh_opt_scale_01_contract.md"
CONTRACT_SHA256 = "233b8dde539d79ef211b4fd94b0a8c6f702200573c8bcf96cf4149272e973eaf"
BASE_COMMIT = "d4693e1165c6d5d0cfe0bb8adc4cd4a79e4166fa"

# Contract section 5 tolerances; never widen these.
ULP_TOLERANCE = 4
RELATIVE_TOLERANCE = 1.0e-6

LEGACY_SYMBOL = "visco_sh_exact_build_steepest_subtractive_step"
SCALED_SYMBOL = "visco_sh_exact_build_scaled_steepest_step"
MASK_SYMBOL = "visco_sh_exact_build_update_mask"
LEGACY_SOURCE = "src/SH/visco_sh_exact_optimizer_boundary.c"
STEP_SCALING_SOURCE = "src/SH/visco_sh_exact_step_scaling.c"
CASE_MAGIC = 0x32534F53
RESULT_MAGIC = 0x52534F53
NULL_SCALING = 10
BOUNDARY_FAULTS = {"null_boundary": (1, -1), "nx_zero": (2, -2), "ny_zero": (3, -2),
                   "null_gradient": (4, -3), "null_step": (5, -3)}
OUTPUT_SENTINEL = -98765.25
MASK_SENTINEL = 7.0
NOT_CALLED = 2**31 - 1
FIELD_ORDER = ("g_primary", "g_rho", "g_q", "m_primary", "m_rho", "m_q", "mask")
CLASSES = ref.CLASSES
NXG, NYG = 12, 10


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _read(repository_root: Path, relative_path: str) -> str:
    return (repository_root / relative_path).read_text(encoding="utf-8")


def _strip_c_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", _strip_c_comments(text))


def _defining_source(repository_root: Path, symbol: str) -> Path:
    """Locate the single production translation unit defining a public symbol."""
    pattern = re.compile(r"\bint\s+" + re.escape(symbol) + r"\s*\([^;{}]*\)\s*\{")
    hits = []
    source_root = repository_root / "src"
    for path in sorted(source_root.rglob("*.c")):
        if "debug" in path.relative_to(source_root).parts:
            continue
        text = _strip_c_comments(path.read_text(encoding="utf-8", errors="replace"))
        if pattern.search(text):
            hits.append(path)
    assert len(hits) == 1, f"expected one production definition of {symbol}, found {hits}"
    return hits[0]


def _launcher() -> str:
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert launcher, "an MPI launcher is required for the SH-OPT-SCALE-01 harnesses"
    return launcher


def _compile_harness(
    repository_root: Path, output: Path, harness: str, sources: list[Path],
) -> Path:
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the SH-OPT-SCALE-01 harnesses"
    unique: list[Path] = []
    for source in sources:
        if source not in unique:
            unique.append(source)
    command = [
        compiler, "-std=c99", "-O2", "-fcommon",
        "-I", str(repository_root / "include"),
        str(repository_root / "tests" / "utilities" / harness),
        *(str(source) for source in unique),
        "-o", str(output), "-lm",
    ]
    result = subprocess.run(
        command, cwd=repository_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    return output


def _run_mpi(
    executable: Path, ranks: int, arguments: list[str], cwd: Path, timeout: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_launcher(), "--oversubscribe", "-n", str(ranks), str(executable), *arguments],
        cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def _grid(function: Callable[[int, int], float], nx: int = NXG, ny: int = NYG) -> list[list[float]]:
    return [[ref.f32(function(j, i)) for i in range(nx)] for j in range(ny)]


def _frame_mask(width: int, nx: int = NXG, ny: int = NYG) -> list[list[float]]:
    return _grid(
        lambda j, i: 0.0 if (i < width or i >= nx - width or j < width or j >= ny - width) else 1.0,
        nx, ny,
    )


def _random_gradient(rng: random.Random, scale: float) -> list[list[float]]:
    return _grid(lambda j, i: rng.choice((-1.0, 1.0)) * scale * rng.uniform(0.05, 1.0))


def _physical_bases(rng: random.Random) -> dict[str, list[list[float]]]:
    return {
        "m_primary": _grid(lambda j, i: rng.uniform(1500.0, 2500.0)),
        "m_rho": _grid(lambda j, i: rng.uniform(1800.0, 2600.0)),
        "m_q": _grid(lambda j, i: rng.uniform(20.0, 100.0)),
    }


def _case(mode: int, weights: tuple[float, float, float], **fields: list[list[float]]) -> dict:
    assert set(fields) == set(FIELD_ORDER)
    ny = len(fields["mask"])
    nx = len(fields["mask"][0])
    return {
        "mode": mode, "nx": nx, "ny": ny,
        "weights": tuple(ref.f32(float(value)) for value in weights),
        "fields": fields,
    }


def _standard_case(mode: int, seed: int = 11, weights=(1.0, 0.5, 2.0)) -> dict:
    rng = random.Random(seed)
    return _case(
        mode, weights,
        g_primary=_random_gradient(rng, 3.0e-4),
        g_rho=_random_gradient(rng, 2.0e-7),
        g_q=_random_gradient(rng, 5.0e-2),
        mask=_frame_mask(2),
        **_physical_bases(rng),
    )


def _with_field(case: dict, name: str, j: int, i: int, value: float) -> dict:
    fields = {key: [row[:] for row in grid] for key, grid in case["fields"].items()}
    fields[name][j][i] = value
    return {**case, "fields": fields}


def _expected(case: dict) -> dict[str, list[list[float]]]:
    fields = case["fields"]
    weights = dict(zip(CLASSES, case["weights"]))
    gradients = {name: fields[f"g_{name}"] for name in CLASSES}
    bases = {name: fields[f"m_{name}"] for name in CLASSES}
    return ref.scaled_step(case["mode"], weights, gradients, bases, fields["mask"])


def _write_case(
    path: Path, case: dict, nprocx: int, nprocy: int, *,
    null_target: int, null_rank: int, callers: int, boundary_fault: int,
    reference: int, mode_override_rank: int, mode_override: int,
) -> None:
    nx, ny = case["nx"], case["ny"]
    payload = bytearray(struct.pack(
        "=13i", CASE_MAGIC, nx, ny, nprocx, nprocy, case["mode"],
        null_target, null_rank, callers, boundary_fault, reference,
        mode_override_rank, mode_override,
    ))
    payload += struct.pack("=3f", *case["weights"])
    for name in FIELD_ORDER:
        grid = case["fields"][name]
        assert len(grid) == ny and all(len(row) == nx for row in grid)
        payload += struct.pack(f"={nx * ny}f", *(value for row in grid for value in row))
    path.write_bytes(bytes(payload))


def _run_boundary(
    harness: Path, workdir: Path, case: dict, *, nprocx: int = 1, nprocy: int = 1,
    null_target: int = 0, null_rank: int = -1, callers: int = 0,
    boundary_fault: int = 0, reference: int = 0, mode_override_rank: int = -1,
    mode_override: int = 0, timeout: float = 120.0,
) -> dict:
    """Run the scaled-step harness and return statuses plus the assembled direction."""
    workdir.mkdir(parents=True, exist_ok=True)
    case_path = workdir / "case.bin"
    prefix = workdir / "result"
    _write_case(case_path, case, nprocx, nprocy, null_target=null_target,
                null_rank=null_rank, callers=callers, boundary_fault=boundary_fault,
                reference=reference, mode_override_rank=mode_override_rank,
                mode_override=mode_override)
    run = _run_mpi(harness, nprocx * nprocy, [str(case_path), str(prefix)], workdir, timeout)
    assert run.returncode == 0, run.stdout
    nxg, nyg = case["nx"], case["ny"]
    step = {name: [[None] * nxg for _ in range(nyg)] for name in CLASSES}
    statuses, unchanged, halos_untouched, untouched = [], [], True, []
    reference_statuses, equal_reference = [], []
    for rank in range(nprocx * nprocy):
        raw = Path(f"{prefix}.{rank}").read_bytes()
        header = struct.unpack("=10i", raw[:40])
        (magic, file_rank, status, pos_x, pos_y, nx, ny, inputs_unchanged,
         reference_status, outputs_equal_reference) = header
        assert magic == RESULT_MAGIC and file_rank == rank
        assert (pos_x, pos_y) == (rank % nprocx, rank // nprocx)
        count = (nx + 2) * (ny + 2)
        assert len(raw) == 40 + 12 * count
        values = struct.unpack(f"={3 * count}f", raw[40:])
        statuses.append(status)
        unchanged.append(bool(inputs_unchanged))
        reference_statuses.append(reference_status)
        equal_reference.append(bool(outputs_equal_reference))
        untouched.append(all(value == OUTPUT_SENTINEL for value in values))
        for k, name in enumerate(CLASSES):
            local = values[k * count:(k + 1) * count]
            for j in range(ny + 2):
                for i in range(nx + 2):
                    value = local[j * (nx + 2) + i]
                    if j in (0, ny + 1) or i in (0, nx + 1):
                        halos_untouched &= value == OUTPUT_SENTINEL
                    else:
                        step[name][pos_y * ny + j - 1][pos_x * nx + i - 1] = value
    return {"statuses": statuses, "inputs_unchanged": unchanged,
            "halos_untouched": halos_untouched, "untouched": untouched, "step": step,
            "reference_statuses": reference_statuses, "equal_reference": equal_reference}


def _assert_closed_form(result: dict, case: dict) -> None:
    """O2: <= 4 ulp against the double reference; masked/zero-weight cells exactly 0."""
    expected = _expected(case)
    mask = case["fields"]["mask"]
    weights = dict(zip(CLASSES, case["weights"]))
    for name in CLASSES:
        for j in range(case["ny"]):
            for i in range(case["nx"]):
                got = result["step"][name][j][i]
                want = expected[name][j][i]
                where = f"{name}[{j}][{i}]"
                assert got is not None and math.isfinite(got), where
                if mask[j][i] == 0.0 or weights[name] == 0.0 or want == 0.0:
                    assert got == 0.0, f"{where}: expected exact 0.0f, got {got!r}"
                else:
                    distance = ref.ulp_distance_f32(got, want)
                    assert distance <= ULP_TOLERANCE, (
                        f"{where}: got {got!r}, reference {want!r}, {distance} ulp")


def _assert_success(result: dict) -> None:
    assert result["statuses"] == [0] * len(result["statuses"])
    assert all(result["inputs_unchanged"])
    assert result["halos_untouched"]


def _assert_rejected(result: dict, code: int) -> None:
    """O8: identical code on every rank and no output word written anywhere."""
    assert result["statuses"] == [code] * len(result["statuses"])
    assert all(result["untouched"]), "step arrays modified on a non-zero return"
    assert all(result["inputs_unchanged"])


def _relative_close(left: float, right: float) -> bool:
    if left == 0.0 or right == 0.0:
        return left == right
    return abs(left - right) <= RELATIVE_TOLERANCE * max(abs(left), abs(right))


# ---------------------------------------------------------------------------
# Compiled harness fixtures
# ---------------------------------------------------------------------------

def _step_scaling_source(repository_root: Path) -> Path:
    return repository_root / STEP_SCALING_SOURCE


@pytest.fixture(scope="module")
def boundary_harness(tmp_path_factory: pytest.TempPathFactory, repository_root: Path) -> Path:
    output = tmp_path_factory.mktemp("sh_opt_scale_boundary") / "boundary"
    # The d4693e1 unit is linked as the frozen O1 reference implementation.
    return _compile_harness(
        repository_root, output, "sh_opt_scale_01_boundary_harness.c",
        [_step_scaling_source(repository_root), repository_root / LEGACY_SOURCE],
    )


@pytest.fixture(scope="module")
def mask_harness(tmp_path_factory: pytest.TempPathFactory, repository_root: Path) -> Path:
    output = tmp_path_factory.mktemp("sh_opt_scale_mask") / "mask"
    # Contract section 2: the new unit links stand-alone (harness provides err()).
    return _compile_harness(
        repository_root, output, "sh_opt_scale_01_mask_harness.c",
        [_step_scaling_source(repository_root)],
    )


@pytest.fixture(scope="module")
def trial_mask_harness(tmp_path_factory: pytest.TempPathFactory, repository_root: Path) -> Path:
    output = tmp_path_factory.mktemp("sh_opt_scale_trial_mask") / "trial_mask"
    return _compile_harness(
        repository_root, output, "sh_opt_scale_01_trial_mask_harness.c",
        [
            _step_scaling_source(repository_root),
            repository_root / LEGACY_SOURCE,
            repository_root / "src" / "SH" / "visco_sh_exact_trial_state.c",
            repository_root / "src" / "q_parameterization.c",
        ],
    )


# ---------------------------------------------------------------------------
# Contract identity and public API (prerequisites of every oracle)
# ---------------------------------------------------------------------------

def test_contract_identity(repository_root: Path) -> None:
    data = (repository_root / CONTRACT_PATH).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == CONTRACT_SHA256


def _struct_members(header: str, name: str) -> list[tuple[str, str]]:
    text = _strip_c_comments(header)
    match = re.search(r"struct\s+" + re.escape(name) + r"\s*\{(.*?)\}\s*;", text, re.DOTALL)
    assert match, f"struct {name} not found"
    members = []
    for declaration in match.group(1).split(";"):
        declaration = " ".join(declaration.split())
        if not declaration:
            continue
        first, *rest = (part.strip() for part in declaration.split(","))
        head = re.fullmatch(r"(?P<type>[A-Za-z_][\w ]*?)\s*(?P<stars>\**)\s*(?P<name>\w+)", first)
        assert head, declaration
        base_type = head.group("type").strip()
        members.append((base_type + head.group("stars"), head.group("name")))
        for part in rest:
            tail = re.fullmatch(r"(?P<stars>\**)\s*(?P<name>\w+)", part)
            assert tail, declaration
            members.append((base_type + tail.group("stars"), tail.group("name")))
    return members


def test_api_step_scaling_struct_has_contract_members(repository_root: Path) -> None:
    members = _struct_members(_read(repository_root, "include/fd.h"),
                              "visco_sh_exact_step_scaling")
    assert members == [
        ("int", "scaling_mode"),
        ("float**", "base_primary"), ("float**", "base_rho"), ("float**", "base_q"),
        ("float**", "update_mask"),
        ("float", "weight_primary"), ("float", "weight_rho"), ("float", "weight_q"),
    ]


def test_api_prototypes_and_translation_unit(repository_root: Path) -> None:
    header = _compact(_read(repository_root, "include/fd.h"))
    assert re.search(
        r"intvisco_sh_exact_build_scaled_steepest_step\("
        r"conststructvisco_sh_exact_optimizer_boundary\*\w*,"
        r"conststructvisco_sh_exact_step_scaling\*\w*\);", header)
    assert re.search(
        r"intvisco_sh_exact_build_update_mask\(float\*\*\w*,(?:int\w*,){8}int\w*\);", header)
    unit = repository_root / STEP_SCALING_SOURCE
    assert unit.is_file()
    for symbol in (SCALED_SYMBOL, MASK_SYMBOL):
        assert _defining_source(repository_root, symbol) == unit
    assert _defining_source(repository_root, LEGACY_SYMBOL) == repository_root / LEGACY_SOURCE
    assert "visco_sh_exact_step_scaling.c" in _read(repository_root, "src/Makefile")
    # Contract section 2: the unit itself does not call err().
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', _strip_c_comments(unit.read_text(encoding="utf-8")))
    assert not re.search(r"\berr\s*\(", code)


def test_driver_dispatches_legacy_and_scaled_steps(repository_root: Path) -> None:
    driver = _compact(_read(repository_root, "src/SH/FWI_SH_visc.c"))
    for call in ("visco_sh_exact_build_steepest_subtractive_step(",
                 "visco_sh_exact_build_scaled_steepest_step(",
                 "visco_sh_exact_build_update_mask("):
        assert call in driver


# ---------------------------------------------------------------------------
# O1: mode 0 / NULL scaling is the d4693e1 bitwise identity p = g_raw
# ---------------------------------------------------------------------------

# SHA-256 (LF-normalized) of `git show d4693e1:<path>` content, frozen so the
# check also works where the base commit cannot be resolved (shallow clones,
# a Windows worktree gitdir seen from WSL).  Whenever git can resolve the
# base, the frozen digests are re-derived from it as a cross-check.
BASE_LEGACY_SOURCE_SHA256 = "a6310aa17936bf5f029a38ae36fe1392a663c2bf93b1a37026cb0b22ee4a2630"
BASE_LEGACY_HEADER_BLOCK_SHA256 = (
    "39fe7de106b409d2c8769febe85f97244bc07995eacd85341223f544c0e3ee9f",  # struct
    "59ed8290c25474e559fbf647ee0e902033674562e319f0f48e39c95107970059",  # prototype
)


def _git_base_blob(repository_root: Path, relative_path: str) -> bytes | None:
    git = shutil.which("git")
    if git is None:
        return None
    result = subprocess.run(
        [git, "show", f"{BASE_COMMIT}:{relative_path}"], cwd=repository_root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    return result.stdout.replace(b"\r\n", b"\n")


def _legacy_header_digests(header: bytes) -> tuple[str, ...]:
    digests = []
    for pattern in (rb"struct visco_sh_exact_optimizer_boundary \{.*?\};",
                    rb"int visco_sh_exact_build_steepest_subtractive_step\([^;]*\);"):
        found = re.findall(pattern, header, flags=re.DOTALL)
        assert len(found) == 1, pattern
        digests.append(hashlib.sha256(found[0]).hexdigest())
    return tuple(digests)


def test_o1_legacy_boundary_unit_and_struct_are_byte_identical_to_base(
    repository_root: Path,
) -> None:
    base_source = _git_base_blob(repository_root, LEGACY_SOURCE)
    if base_source is not None:
        assert hashlib.sha256(base_source).hexdigest() == BASE_LEGACY_SOURCE_SHA256
    base_header = _git_base_blob(repository_root, "include/fd.h")
    if base_header is not None:
        assert _legacy_header_digests(base_header) == BASE_LEGACY_HEADER_BLOCK_SHA256
    current_source = (repository_root / LEGACY_SOURCE).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(current_source).hexdigest() == BASE_LEGACY_SOURCE_SHA256
    current_header = (repository_root / "include/fd.h").read_bytes().replace(b"\r\n", b"\n")
    assert _legacy_header_digests(current_header) == BASE_LEGACY_HEADER_BLOCK_SHA256


def _special_gradient_case() -> dict:
    rng = random.Random(1)
    case = _standard_case(0, seed=1)
    specials = (-0.0, 1.0e-40, -1.0e-40, 1.17549435e-38, 3.0e38, -3.0e38, 0.0, -2.5)
    for index, value in enumerate(specials):
        for name in ("g_primary", "g_rho", "g_q"):
            case = _with_field(case, name, index % NYG, (3 * index + rng.randrange(2)) % NXG,
                               ref.f32(value))
    return case


def _assert_identity(result: dict, case: dict, ranks: list[int] | None = None,
                     nprocx: int = 1, nprocy: int = 1) -> None:
    nx, ny = case["nx"] // nprocx, case["ny"] // nprocy
    for name in CLASSES:
        gradient = case["fields"][f"g_{name}"]
        for j in range(case["ny"]):
            for i in range(case["nx"]):
                rank = (j // ny) * nprocx + (i // nx)
                if ranks is not None and rank not in ranks:
                    continue
                got = result["step"][name][j][i]
                assert ref.f32_bits(got) == ref.f32_bits(gradient[j][i]), (
                    f"{name}[{j}][{i}] not bitwise equal to the raw gradient")


IDENTITY_VARIANTS = {"mode0": 0, "null_scaling": NULL_SCALING}


@pytest.mark.parametrize("variant", list(IDENTITY_VARIANTS))
def test_o1_identity_is_bitwise_raw_gradient_and_d4693e1(
    boundary_harness: Path, tmp_path: Path, variant: str,
) -> None:
    case = _special_gradient_case()
    result = _run_boundary(boundary_harness, tmp_path / variant, case,
                           null_target=IDENTITY_VARIANTS[variant], reference=1)
    _assert_success(result)
    _assert_identity(result, case)
    assert result["reference_statuses"] == [0]
    assert result["equal_reference"] == [True]


@pytest.mark.parametrize("null_target", [0, 9], ids=["poisoned_values", "null_new_members"])
def test_o1_mode0_reads_no_other_scaling_member(
    boundary_harness: Path, tmp_path: Path, null_target: int,
) -> None:
    case = _special_gradient_case()
    nan = float("nan")
    poisoned = _case(
        0, (nan, -1.0, float("inf")),
        g_primary=case["fields"]["g_primary"], g_rho=case["fields"]["g_rho"],
        g_q=case["fields"]["g_q"],
        m_primary=_grid(lambda j, i: nan), m_rho=_grid(lambda j, i: float("inf")),
        m_q=_grid(lambda j, i: nan), mask=_grid(lambda j, i: 0.5),
    )
    result = _run_boundary(boundary_harness, tmp_path / "poison", poisoned,
                           null_target=null_target, reference=1)
    _assert_success(result)
    _assert_identity(result, poisoned)
    assert result["equal_reference"] == [True]


@pytest.mark.parametrize("variant", list(IDENTITY_VARIANTS))
@pytest.mark.parametrize("fault", list(BOUNDARY_FAULTS))
def test_o1_identity_keeps_d4693e1_return_codes(
    boundary_harness: Path, tmp_path: Path, variant: str, fault: str,
) -> None:
    fault_id, code = BOUNDARY_FAULTS[fault]
    result = _run_boundary(boundary_harness, tmp_path / f"{variant}_{fault}",
                           _special_gradient_case(), null_target=IDENTITY_VARIANTS[variant],
                           boundary_fault=fault_id, reference=1)
    assert result["statuses"] == [code]
    assert result["reference_statuses"] == [code]
    assert all(result["untouched"]) and result["equal_reference"] == [True]
    assert all(result["inputs_unchanged"])


@pytest.mark.parametrize("variant", list(IDENTITY_VARIANTS))
def test_o1_identity_is_not_collective(
    boundary_harness: Path, tmp_path: Path, variant: str,
) -> None:
    case = _special_gradient_case()
    try:
        result = _run_boundary(boundary_harness, tmp_path / "rank0_only", case,
                               nprocx=2, callers=1, null_target=IDENTITY_VARIANTS[variant],
                               reference=1, timeout=60.0)
    except subprocess.TimeoutExpired:
        pytest.fail("mode 0 / NULL scaling blocked when only one rank called it")
    assert result["statuses"] == [0, NOT_CALLED]
    assert result["equal_reference"][0] is True
    assert all(result["inputs_unchanged"])
    _assert_identity(result, case, ranks=[0], nprocx=2)


# ---------------------------------------------------------------------------
# O2: modes 1/2 equal the closed form of contract section 2
# ---------------------------------------------------------------------------

def _o2_cases(mode: int) -> dict[str, dict]:
    rng = random.Random(100 + mode)
    bases = _physical_bases(rng)
    frame = _frame_mask(2)
    g = {
        "g_primary": _random_gradient(rng, 3.0e-4),
        "g_rho": _random_gradient(rng, 2.0e-7),
        "g_q": _random_gradient(rng, 5.0e-2),
    }
    zero = _grid(lambda j, i: 0.0)
    outside_only = _grid(lambda j, i: 0.0 if frame[j][i] == 1.0 else rng.uniform(1.0, 9.0))
    single = _grid(lambda j, i: 1.0 if (j, i) == (4, 7) else 0.0)
    irregular = _grid(lambda j, i: float(rng.random() < 0.5))
    irregular[3][3] = 1.0
    outlier_g = _grid(lambda j, i: (1.0e3 if frame[j][i] == 0.0 else 1.0)
                      * rng.uniform(-1.0, 1.0) * 1.0e-3)
    outlier_m = _grid(lambda j, i: 1.0e6 if frame[j][i] == 0.0 else rng.uniform(10.0, 20.0))
    wide = _grid(lambda j, i: rng.choice((-1.0, 1.0)) * 10.0 ** rng.uniform(-9.0, 3.0))
    mixed_m = _grid(lambda j, i: rng.uniform(-50.0, 120.0))
    return {
        "frame_mask_weighted": _case(mode, (1.0, 0.5, 2.0), mask=frame, **g, **bases),
        "zero_gradient_class": _case(mode, (1.0, 0.5, 2.0), mask=frame,
                                     g_primary=g["g_primary"], g_rho=zero, g_q=g["g_q"], **bases),
        "zero_weight_class": _case(mode, (1.0, 1.0, 0.0), mask=frame, **g, **bases),
        "gradient_only_outside_mask": _case(mode, (1.0, 0.7, 1.3), mask=frame,
                                            g_primary=g["g_primary"], g_rho=g["g_rho"],
                                            g_q=outside_only, **bases),
        "full_mask": _case(mode, (0.25, 3.0, 1.0), mask=_grid(lambda j, i: 1.0), **g, **bases),
        "single_active_cell": _case(mode, (1.0, 1.0, 1.0), mask=single, **g, **bases),
        "irregular_mask": _case(mode, (0.1, 0.2, 0.3), mask=irregular, **g, **bases),
        "outliers_outside_mask": _case(mode, (1.0, 1.0, 1.0), mask=frame,
                                       g_primary=outlier_g, g_rho=outlier_g, g_q=outlier_g,
                                       m_primary=outlier_m, m_rho=outlier_m, m_q=outlier_m),
        "mixed_sign_wide_range": _case(mode, (1.5, 0.75, 1.0), mask=irregular,
                                       g_primary=wide, g_rho=g["g_rho"], g_q=wide,
                                       m_primary=bases["m_primary"], m_rho=mixed_m,
                                       m_q=mixed_m),
    }


O2_CASE_NAMES = (
    "frame_mask_weighted", "zero_gradient_class", "zero_weight_class",
    "gradient_only_outside_mask", "full_mask", "single_active_cell",
    "irregular_mask", "outliers_outside_mask", "mixed_sign_wide_range",
)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("case_name", O2_CASE_NAMES)
def test_o2_scaled_direction_matches_closed_form(
    boundary_harness: Path, tmp_path: Path, mode: int, case_name: str,
) -> None:
    case = _o2_cases(mode)[case_name]
    result = _run_boundary(boundary_harness, tmp_path / case_name, case)
    _assert_success(result)
    _assert_closed_form(result, case)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
def test_o2_o8_non_finite_values_outside_mask_are_ignored_and_zeroed(
    boundary_harness: Path, tmp_path: Path, mode: int,
) -> None:
    case = _standard_case(mode)
    frame = case["fields"]["mask"]
    assert frame[0][0] == 0.0 and frame[NYG - 1][NXG - 1] == 0.0
    case = _with_field(case, "g_primary", 0, 0, float("nan"))
    case = _with_field(case, "g_q", NYG - 1, NXG - 1, float("inf"))
    case = _with_field(case, "m_rho", 0, 5, float("nan"))
    result = _run_boundary(boundary_harness, tmp_path / "outside_non_finite", case)
    _assert_success(result)
    _assert_closed_form(result, case)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
def test_o2_negative_zero_counts_as_zero(
    boundary_harness: Path, tmp_path: Path, mode: int,
) -> None:
    case = _standard_case(mode, weights=(-0.0, 0.5, 2.0))
    signed_mask = [[-0.0 if value == 0.0 else value for value in row]
                   for row in case["fields"]["mask"]]
    signed = {**case, "fields": {**case["fields"], "mask": signed_mask}}
    plain = {**case, "weights": (0.0, *case["weights"][1:])}
    signed_result = _run_boundary(boundary_harness, tmp_path / "signed", signed)
    plain_result = _run_boundary(boundary_harness, tmp_path / "plain", plain)
    _assert_success(signed_result)
    _assert_success(plain_result)
    _assert_closed_form(signed_result, signed)
    for name in CLASSES:
        for j in range(NYG):
            for i in range(NXG):
                left = signed_result["step"][name][j][i]
                right = plain_result["step"][name][j][i]
                assert left == right, f"{name}[{j}][{i}]: {left!r} vs {right!r}"


# ---------------------------------------------------------------------------
# O3: descent property sum g.p > 0 (= 0 only for all-zero weighted gradients)
# ---------------------------------------------------------------------------

def _o3_random_case(mode: int, seed: int) -> dict:
    rng = random.Random(seed)
    mask = _grid(lambda j, i: float(rng.random() < 0.6))
    mask[rng.randrange(NYG)][rng.randrange(NXG)] = 1.0
    weights = [rng.choice((0.0, rng.uniform(0.1, 3.0))) for _ in CLASSES]
    if all(weight == 0.0 for weight in weights):
        weights[rng.randrange(3)] = rng.uniform(0.1, 3.0)

    def nonzero(scale: float) -> list[list[float]]:
        return _grid(lambda j, i: rng.choice((-1.0, 1.0)) * scale * rng.uniform(0.01, 1.0))

    return _case(mode, tuple(weights), mask=mask, g_primary=nonzero(1.0e-4),
                 g_rho=nonzero(1.0e-7), g_q=nonzero(1.0e-1), **_physical_bases(rng))


def _class_dots(result: dict, case: dict) -> dict[str, float]:
    return {name: ref.class_dot(case["fields"][f"g_{name}"], result["step"][name])
            for name in CLASSES}


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("seed", range(8))
def test_o3_random_masked_gradients_give_strict_descent(
    boundary_harness: Path, tmp_path: Path, mode: int, seed: int,
) -> None:
    case = _o3_random_case(mode, 1000 * mode + seed)
    result = _run_boundary(boundary_harness, tmp_path / f"descent_{seed}", case)
    _assert_success(result)
    dots = _class_dots(result, case)
    for name, weight in zip(CLASSES, case["weights"]):
        assert dots[name] >= 0.0
        if weight > 0.0:
            assert dots[name] > 0.0
    assert ref.descent_dot({n: case["fields"][f"g_{n}"] for n in CLASSES}, result["step"]) > 0.0


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
def test_o3_zero_only_when_weighted_masked_gradients_vanish(
    boundary_harness: Path, tmp_path: Path, mode: int,
) -> None:
    rng = random.Random(77 + mode)
    frame = _frame_mask(2)
    outside = _grid(lambda j, i: 0.0 if frame[j][i] == 1.0 else rng.uniform(-5.0, 5.0))
    everywhere = _random_gradient(rng, 1.0)
    # primary/rho carry weight but have non-zero gradients only outside M;
    # q has non-zero masked gradients but zero weight.
    case = _case(mode, (1.0, 2.0, 0.0), mask=frame, g_primary=outside, g_rho=outside,
                 g_q=everywhere, **_physical_bases(rng))
    result = _run_boundary(boundary_harness, tmp_path / "zero_descent", case)
    _assert_success(result)
    assert all(value == 0.0 for name in CLASSES for row in result["step"][name] for value in row)
    assert ref.descent_dot({n: case["fields"][f"g_{n}"] for n in CLASSES}, result["step"]) == 0.0

    # A single tiny non-zero masked gradient of a weighted class restores > 0.
    tiny = _with_field(case, "g_rho", 4, 5, ref.f32(1.0e-20))
    assert tiny["fields"]["mask"][4][5] == 1.0
    result = _run_boundary(boundary_harness, tmp_path / "single_tiny", tiny)
    _assert_success(result)
    assert ref.descent_dot({n: tiny["fields"][f"g_{n}"] for n in CLASSES}, result["step"]) > 0.0
    _assert_closed_form(result, tiny)


# ---------------------------------------------------------------------------
# O4: unit invariance of the relative update alpha * p_k / m_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize(
    "factors", [(1.0e3, 1.0e-2, 37.5), (2.0 ** -7, 2.0 ** 9, 1.0), (0.3, 3.0e4, 1.0e-3)],
    ids=["mixed", "powers_of_two", "wide"],
)
def test_o4_relative_update_is_unit_invariant(
    boundary_harness: Path, tmp_path: Path, mode: int, factors: tuple[float, float, float],
) -> None:
    alpha = 0.037
    case = _standard_case(mode, seed=404, weights=(1.0, 0.6, 1.7))
    fields = dict(case["fields"])
    for name, factor in zip(CLASSES, factors):
        fields[f"m_{name}"] = [[ref.f32(factor * value) for value in row]
                               for row in case["fields"][f"m_{name}"]]
        fields[f"g_{name}"] = [[ref.f32(value / factor) for value in row]
                               for row in case["fields"][f"g_{name}"]]
    scaled = {**case, "fields": fields}
    original = _run_boundary(boundary_harness, tmp_path / "original", case)
    transformed = _run_boundary(boundary_harness, tmp_path / "scaled", scaled)
    _assert_success(original)
    _assert_success(transformed)
    for name in CLASSES:
        for j in range(NYG):
            for i in range(NXG):
                left = alpha * original["step"][name][j][i] / case["fields"][f"m_{name}"][j][i]
                right = alpha * transformed["step"][name][j][i] / fields[f"m_{name}"][j][i]
                assert _relative_close(left, right), f"{name}[{j}][{i}]: {left!r} vs {right!r}"


# ---------------------------------------------------------------------------
# O5: MPI decompositions 1x1, 2x1, 1x2 give the same direction
# ---------------------------------------------------------------------------

def _o5_cases(mode: int) -> dict[str, dict]:
    left_masked = _grid(lambda j, i: 0.0 if i < NXG // 2 else 1.0)
    top_masked = _grid(lambda j, i: 0.0 if j < NYG // 2 else 1.0)
    base = _standard_case(mode, seed=505)
    return {
        "frame": base,
        "rank_local_zero_mask_x": {**base, "fields": {**base["fields"], "mask": left_masked}},
        "rank_local_zero_mask_y": {**base, "fields": {**base["fields"], "mask": top_masked}},
    }


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("case_name", ["frame", "rank_local_zero_mask_x", "rank_local_zero_mask_y"])
def test_o5_decompositions_agree(
    boundary_harness: Path, tmp_path: Path, mode: int, case_name: str,
) -> None:
    case = _o5_cases(mode)[case_name]
    results = {}
    for nprocx, nprocy in ((1, 1), (2, 1), (1, 2)):
        result = _run_boundary(boundary_harness, tmp_path / f"{nprocx}x{nprocy}", case,
                               nprocx=nprocx, nprocy=nprocy)
        _assert_success(result)
        _assert_closed_form(result, case)
        results[(nprocx, nprocy)] = result
    reference = results[(1, 1)]["step"]
    for decomposition in ((2, 1), (1, 2)):
        step = results[decomposition]["step"]
        for name in CLASSES:
            for j in range(NYG):
                for i in range(NXG):
                    assert _relative_close(step[name][j][i], reference[name][j][i]), (
                        f"{decomposition} {name}[{j}][{i}]")


# ---------------------------------------------------------------------------
# O6: CPML update mask (section 3) and bitwise-frozen masked trial cells
# ---------------------------------------------------------------------------

def _mask_configurations() -> list[tuple[int, ...]]:
    configurations = []
    for nx, ny in ((7, 5), (4, 9), (1, 1), (13, 13)):
        for nprocx in (1, 2, 3):
            for nprocy in (1, 2, 3):
                for pos_x in range(nprocx):
                    for pos_y in range(nprocy):
                        for fw in (0, 1, 2, 3, 6):
                            for free_surf in (0, 1):
                                for boundary in (0, 1):
                                    configurations.append(
                                        (nx, ny, pos_x, pos_y, nprocx, nprocy, fw,
                                         free_surf, boundary, 0))
    return configurations


def _run_mask(harness: Path, workdir: Path, configurations: list[tuple[int, ...]]) -> list[dict]:
    workdir.mkdir(parents=True, exist_ok=True)
    config = workdir / "mask_configurations.txt"
    config.write_text("".join(" ".join(map(str, item)) + "\n" for item in configurations),
                      encoding="ascii")
    run = _run_mpi(harness, 1, [str(config)], workdir)
    assert run.returncode == 0, run.stdout
    records = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    assert len(records) == len(configurations), run.stdout
    return records


def test_o6_update_mask_matches_cpml_definition(mask_harness: Path, tmp_path: Path) -> None:
    configurations = _mask_configurations()
    records = _run_mask(mask_harness, tmp_path / "mask", configurations)
    for configuration, record in zip(configurations, records):
        nx, ny, pos_x, pos_y, nprocx, nprocy, fw, free_surf, boundary, _ = configuration
        assert record["status"] == 0, configuration
        expected = ref.update_mask(nx, ny, pos_x, pos_y, nprocx, nprocy, fw, free_surf, boundary)
        cols = record["cols"]
        assert (record["rows"], cols) == (ny + 2, nx + 2)
        values = record["values"]
        for j in range(ny + 2):
            for i in range(nx + 2):
                value = values[j * cols + i]
                if j in (0, ny + 1) or i in (0, nx + 1):
                    assert value == MASK_SENTINEL, (configuration, j, i)
                else:
                    assert value == expected[j - 1][i - 1], (configuration, j, i)


def test_o6_masked_cells_are_bitwise_frozen_in_trial_state(trial_mask_harness: Path,
                                                           tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    run = _run_mpi(trial_mask_harness, 1, [], tmp_path)
    assert run.returncode == 0, run.stdout
    records = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]
    assert len(records) == 1, run.stdout
    record = records[0]
    expected_masked = 0
    for _mode in (1, 2):
        for pos_y in (0, 1):
            for pos_x in (0, 1):
                for free_surf in (0, 1):
                    for boundary in (0, 1):
                        mask = ref.update_mask(10, 9, pos_x, pos_y, 2, 2, 3, free_surf, boundary)
                        expected_masked += sum(row.count(0.0) for row in mask)
    assert record["configurations"] == 32
    assert record["mask_failures"] == 0
    assert record["boundary_failures"] == 0
    assert record["trial_failures"] == 0
    assert record["masked_cells_checked"] == expected_masked * record["alphas"]
    assert record["masked_cells_checked"] > 0
    assert record["masked_step_nonzero"] == 0
    assert record["masked_mismatches"] == 0
    assert record["unmasked_changed"] > 0


# ---------------------------------------------------------------------------
# O8: error paths leave every output array untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_mode,nprocx", [(3, 1), (-1, 1), (7, 1), (3, 2)])
def test_o8_invalid_scaling_mode_returns_minus_4(
    boundary_harness: Path, tmp_path: Path, bad_mode: int, nprocx: int,
) -> None:
    case = {**_standard_case(1), "mode": bad_mode}
    result = _run_boundary(boundary_harness, tmp_path / "mode", case, nprocx=nprocx)
    _assert_rejected(result, -4)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("null_target", range(1, 9), ids=[
    "base_primary", "base_rho", "base_q", "update_mask",
    "row_base_primary", "row_base_rho", "row_base_q", "row_update_mask"])
def test_o8_null_base_or_mask_returns_minus_5(
    boundary_harness: Path, tmp_path: Path, mode: int, null_target: int,
) -> None:
    result = _run_boundary(boundary_harness, tmp_path / "null", _standard_case(mode),
                           null_target=null_target)
    _assert_rejected(result, -5)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("value", [0.5, 2.0, -1.0, float("nan"), float("inf")],
                         ids=["half", "two", "minus_one", "nan", "inf"])
def test_o8_non_binary_mask_value_returns_minus_5(
    boundary_harness: Path, tmp_path: Path, mode: int, value: float,
) -> None:
    case = _with_field(_standard_case(mode), "mask", 5, 6, value)
    result = _run_boundary(boundary_harness, tmp_path / "mask_value", case)
    _assert_rejected(result, -5)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("weights", [
    (-1.0e-3, 1.0, 1.0), (1.0, float("nan"), 1.0), (1.0, 1.0, float("inf")),
    (1.0, float("-inf"), 1.0), (0.0, 0.0, 0.0), (-0.0, 0.0, -0.0),
], ids=["negative", "nan", "inf", "minus_inf", "all_zero", "all_signed_zero"])
def test_o8_invalid_weights_return_minus_6(
    boundary_harness: Path, tmp_path: Path, mode: int, weights: tuple[float, float, float],
) -> None:
    case = {**_standard_case(mode), "weights": tuple(ref.f32(w) for w in weights)}
    result = _run_boundary(boundary_harness, tmp_path / "weights", case)
    _assert_rejected(result, -6)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("nprocx,nprocy", [(1, 1), (2, 1), (1, 2)])
def test_o8_empty_global_mask_returns_minus_7(
    boundary_harness: Path, tmp_path: Path, mode: int, nprocx: int, nprocy: int,
) -> None:
    base = _standard_case(mode)
    case = {**base, "fields": {**base["fields"], "mask": _grid(lambda j, i: 0.0)}}
    result = _run_boundary(boundary_harness, tmp_path / "empty", case,
                           nprocx=nprocx, nprocy=nprocy)
    _assert_rejected(result, -7)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("field,value", [
    ("g_primary", float("nan")), ("g_rho", float("inf")), ("g_q", float("-inf")),
    ("m_primary", float("-inf")), ("m_rho", float("nan")), ("m_q", float("inf")),
])
def test_o8_non_finite_masked_value_returns_minus_8(
    boundary_harness: Path, tmp_path: Path, mode: int, field: str, value: float,
) -> None:
    base = _standard_case(mode)
    assert base["fields"]["mask"][5][6] == 1.0
    case = _with_field(base, field, 5, 6, value)
    result = _run_boundary(boundary_harness, tmp_path / "non_finite", case)
    _assert_rejected(result, -8)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("violation,code", [
    ("mask_value", -5), ("null_row", -5), ("non_finite_gradient", -8),
    ("non_finite_base", -8),
])
def test_o8_violation_on_one_rank_is_collective(
    boundary_harness: Path, tmp_path: Path, mode: int, violation: str, code: int,
) -> None:
    # 2x1 split of the 12x10 grid: rank 1 owns columns 6..11.
    case = _standard_case(mode)
    assert case["fields"]["mask"][4][8] == 1.0
    null_target, null_rank = 0, -1
    if violation == "mask_value":
        case = _with_field(case, "mask", 4, 8, 0.25)
    elif violation == "null_row":
        null_target, null_rank = 8, 1
    elif violation == "non_finite_gradient":
        case = _with_field(case, "g_q", 4, 8, float("nan"))
    else:
        case = _with_field(case, "m_primary", 4, 8, float("inf"))
    result = _run_boundary(boundary_harness, tmp_path / violation, case, nprocx=2,
                           null_target=null_target, null_rank=null_rank)
    _assert_rejected(result, code)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("weights,field", [
    ((1.0, 0.0, 1.0), "g_rho"), ((1.0, 1.0, 0.0), "m_q"), ((0.0, 1.0, 1.0), "g_primary"),
], ids=["zero_weight_gradient", "zero_weight_base", "zero_weight_primary"])
def test_o8_minus_8_is_independent_of_weight(
    boundary_harness: Path, tmp_path: Path, mode: int,
    weights: tuple[float, float, float], field: str,
) -> None:
    case = _with_field(_standard_case(mode, weights=weights), field, 5, 6, float("nan"))
    assert case["fields"]["mask"][5][6] == 1.0
    result = _run_boundary(boundary_harness, tmp_path / "weightless_nan", case)
    _assert_rejected(result, -8)


def _precedence_case(mode: int, violations: tuple[str, ...]) -> tuple[dict, int]:
    """Apply several simultaneous violations; returns (case, null_target)."""
    case = _standard_case(mode)
    null_target = 0
    for violation in violations:
        if violation == "bad_mode":
            case = {**case, "mode": 3}
        elif violation == "null_mask":
            null_target = 4
        elif violation == "null_row":
            null_target = 6
        elif violation == "mask_value":
            case = _with_field(case, "mask", 5, 6, 0.5)
        elif violation == "negative_weight":
            case = {**case, "weights": (ref.f32(-1.0), *case["weights"][1:])}
        elif violation == "zero_weights":
            case = {**case, "weights": (0.0, 0.0, 0.0)}
        elif violation == "empty_mask":
            case = {**case, "fields": {**case["fields"], "mask": _grid(lambda j, i: 0.0)}}
        elif violation == "nan_masked_gradient":
            case = _with_field(case, "g_q", 4, 4, float("nan"))
        elif violation == "nan_masked_base":
            case = _with_field(case, "m_rho", 4, 4, float("nan"))
        else:
            raise AssertionError(violation)
    return case, null_target


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("violations,code", [
    (("bad_mode", "null_mask"), -4),
    (("bad_mode", "negative_weight", "empty_mask", "nan_masked_gradient"), -4),
    (("null_mask", "negative_weight"), -5),
    (("null_row", "nan_masked_gradient"), -5),
    (("mask_value", "zero_weights", "nan_masked_base"), -5),
    (("negative_weight", "empty_mask"), -6),
    (("zero_weights", "nan_masked_base"), -6),
    (("negative_weight", "nan_masked_gradient"), -6),
], ids=lambda value: "+".join(value) if isinstance(value, tuple) else str(value))
def test_o8_smallest_applicable_code_wins(
    boundary_harness: Path, tmp_path: Path, mode: int, violations: tuple[str, ...], code: int,
) -> None:
    case, null_target = _precedence_case(mode, violations)
    result = _run_boundary(boundary_harness, tmp_path / "precedence", case,
                           null_target=null_target)
    _assert_rejected(result, code)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
def test_o8_precedence_is_collective_across_ranks(
    boundary_harness: Path, tmp_path: Path, mode: int,
) -> None:
    # Rank 0 (columns 0..5) sees only a -8 condition, rank 1 only a -5 one.
    case = _with_field(_standard_case(mode), "g_primary", 4, 3, float("nan"))
    case = _with_field(case, "mask", 4, 8, 2.0)
    result = _run_boundary(boundary_harness, tmp_path / "split_precedence", case, nprocx=2)
    _assert_rejected(result, -5)
    # Rank 0 additionally loses its mask (-5) while rank 1 has an invalid mode (-4).
    try:
        result = _run_boundary(boundary_harness, tmp_path / "split_mode", case, nprocx=2,
                               null_target=4, null_rank=0, mode_override_rank=1,
                               mode_override=3, timeout=60.0)
    except subprocess.TimeoutExpired:
        pytest.fail("an invalid mode on one rank was not checked collectively")
    _assert_rejected(result, -4)


@pytest.mark.parametrize("mode", [1, 2], ids=["rms", "max"])
@pytest.mark.parametrize("bad_mode", [3, -1])
def test_o8_invalid_mode_on_one_rank_is_collective(
    boundary_harness: Path, tmp_path: Path, mode: int, bad_mode: int,
) -> None:
    try:
        result = _run_boundary(boundary_harness, tmp_path / "rank_mode", _standard_case(mode),
                               nprocx=2, mode_override_rank=1, mode_override=bad_mode,
                               timeout=60.0)
    except subprocess.TimeoutExpired:
        pytest.fail("an invalid mode on one rank was not checked collectively")
    _assert_rejected(result, -4)


def test_o8_update_mask_invalid_arguments_return_minus_1(mask_harness: Path,
                                                        tmp_path: Path) -> None:
    configurations = [
        (5, 4, 0, 0, 1, 1, 2, 0, 0, 1),     # NULL mask
        (0, 4, 0, 0, 1, 1, 2, 0, 0, 0),     # nx < 1
        (-3, 4, 0, 0, 1, 1, 2, 0, 0, 0),
        (5, 0, 0, 0, 1, 1, 2, 0, 0, 0),     # ny < 1
        (5, 4, 0, 0, 1, 1, -1, 0, 0, 0),    # fw < 0
        (5, 4, 0, 0, 0, 1, 2, 0, 0, 0),     # nprocx < 1
        (5, 4, 0, 0, 1, 0, 2, 0, 0, 0),     # nprocy < 1
        (5, 4, -1, 0, 2, 1, 2, 0, 0, 0),    # pos_x < 0
        (5, 4, 2, 0, 2, 1, 2, 0, 0, 0),     # pos_x > nprocx - 1
        (5, 4, 0, -1, 1, 2, 2, 0, 0, 0),    # pos_y < 0
        (5, 4, 0, 2, 1, 2, 2, 0, 0, 0),     # pos_y > nprocy - 1
    ]
    records = _run_mask(mask_harness, tmp_path / "mask_invalid", configurations)
    for configuration, record in zip(configurations, records):
        nx, ny, pos_x, pos_y, nprocx, nprocy, fw = configuration[:7]
        assert ref.update_mask_status(nx, ny, pos_x, pos_y, nprocx, nprocy, fw,
                                      bool(configuration[9])) == -1
        assert record["status"] == -1, configuration
        assert all(value == MASK_SENTINEL for value in record["values"]), configuration


# ---------------------------------------------------------------------------
# Driver fixture for O7 / O9 (real bin/denise, 32x32 physical-Q SH-FWI case)
# ---------------------------------------------------------------------------

DRIVER_N = 32
DRIVER_CELLS = DRIVER_N * DRIVER_N
DRIVER_FW = 6
WORKFLOW_HEADER = (
    "PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ INV_VP_ITER "
    "INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP WD_DAMP1 EPRECOND "
    "LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE OFFSET_MUTE OFFSETC "
    "SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER"
)
MIGRATION_RECORDS = (
    ("MIGRATION_SOURCE_PREFIX", "migration/unused_source"),
    ("MIGRATION_DATA_PREFIX", "migration/unused_data"),
    ("MIGRATION_IMAGE_PREFIX", "migration/unused_image"),
)
OPT_RECORD_NAMES = (
    "OPT_STEP_SCALING", "OPT_WEIGHT_PRIMARY", "OPT_WEIGHT_RHO", "OPT_WEIGHT_Q",
    "OPT_DIAG_OUT",
)
# A dimensionless initial alpha for modes 1/2 (relative class change 1 %).
SCALED_EPS = 0.01


def _write_parameter(
    path: Path, *, records: int, opt_step_scaling: object = 0,
    opt_weights: tuple[object, object, object] = (1.0, 1.0, 1.0),
    opt_diag_out: object = 0, **fixture: object,
) -> None:
    """Write 119, 122 or 123..127 positional records (1-119 via the frozen fixture).

    Records 123..127 are optional trailing records (contract section 4); a file
    with ``records`` = 122 + n carries the first n of them in order.
    """
    assert records in (119, 122, 123, 124, 125, 126, 127)
    write_sh_parameter(path, **fixture)
    text = path.read_text(encoding="ascii")
    assert text.endswith("\n")
    extra: list[tuple[str, object]] = []
    if records >= 122:
        extra.extend(MIGRATION_RECORDS)
    if records >= 123:
        values = (opt_step_scaling, *opt_weights, opt_diag_out)
        extra.extend(list(zip(OPT_RECORD_NAMES, values))[:records - 122])
    path.write_text(text + "".join(f"({name}) = {value}\n" for name, value in extra),
                    encoding="ascii", newline="\n")


def _parameter_records(path: Path) -> list[tuple[str, str]]:
    records = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        assert separator == "="
        records.append((name.strip().strip("()"), value.strip()))
    return records


def _write_workflow(path: Path, *, pro: float = 0.01, inv_vs_iter: int = 0,
                    inv_rho_iter: int = 0, inv_qs_iter: int = 0) -> None:
    values = [pro, 0, 0, 20, 2, 0, 1, 0, 0, 0, inv_vs_iter, inv_rho_iter, inv_qs_iter,
              0, 0.5, 0.5, 0, 2, 0, 0, 0, 0.1, 0, 0, 10, 1, 1, 0, 0, 0]
    assert len(values) == len(WORKFLOW_HEADER.split()) == 30
    path.write_text(f"{WORKFLOW_HEADER}\n{' '.join(str(v) for v in values)}\n", encoding="ascii")


def _write_native_model(path: Path, values: list[float]) -> None:
    assert len(values) == DRIVER_CELLS
    path.write_bytes(struct.pack(f"={DRIVER_CELLS}f", *values))


def _read_native_model(path: Path) -> tuple[float, ...]:
    raw = path.read_bytes()
    assert len(raw) == 4 * DRIVER_CELLS, path
    return struct.unpack(f"={DRIVER_CELLS}f", raw)


def _run_denise(denise_binary: Path, mpiexec: str, root: Path, parameter: Path,
                workflow: Path, ranks: int = 1) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [mpiexec, "--oversubscribe", "-n", str(ranks), str(denise_binary),
         str(parameter), str(workflow)],
        cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900,
    )


def _prepare_driver_case(tmp_path: Path, denise_binary: Path, mpiexec: str, *,
                         invmat1: int = 1) -> dict:
    """32x32 viscoelastic SH case: true Q=80, start Q=70 in a central patch."""
    root = tmp_path / "sh_opt_scale_case"
    for name in ("model", "source", "receiver", "su", "log"):
        (root / name).mkdir(parents=True, exist_ok=True)
    primary_suffix = ".vs" if invmat1 == 1 else ".mu"
    primary_value = 2000.0 if invmat1 == 1 else 2000.0 * 2000.0 * 2000.0
    true_q, start_q = [], []
    for x in range(DRIVER_N):
        for y in range(DRIVER_N):
            true_q.append(80.0)
            start_q.append(70.0 if 14 <= x <= 18 and 14 <= y <= 18 else 80.0)
    for prefix, qs in ((root / "model" / "true", true_q), (root / "model" / "start", start_q)):
        _write_native_model(prefix.with_suffix(primary_suffix), [primary_value] * DRIVER_CELLS)
        _write_native_model(prefix.with_suffix(".rho"), [2000.0] * DRIVER_CELLS)
        _write_native_model(prefix.with_suffix(".qs"), qs)
    (root / "source" / "one.dat").write_text("1\n160 0 160 0 20 1 0 1\n", encoding="ascii")
    (root / "receiver" / "line.dat").write_text(
        "100 160\n120 160\n140 160\n180 160\n200 160\n220 160\n", encoding="ascii")
    workflow = root / "workflow.inp"
    _write_workflow(workflow)
    forward = root / "true_forward.inp"
    write_sh_parameter(forward, invmat1=invmat1)
    run = _run_denise(denise_binary, mpiexec, root, forward, workflow)
    assert run.returncode == 0, run.stdout
    assert (root / "su" / "observed_y.su.shot1").is_file()
    start = root / "model" / "start"
    suffixes = (primary_suffix, ".rho", ".qs")
    return {
        "root": root, "workflow": workflow, "invmat1": invmat1,
        "primary_suffix": primary_suffix, "suffixes": suffixes, "start": start,
        "start_bytes": {s: start.with_suffix(s).read_bytes() for s in suffixes},
    }


def _fwi_parameter(case: dict, name: str, *, records: int, output: str,
                   itermax: int = 1, nprocx: int = 1, eps_scale: float = 1.0e6,
                   **options: object) -> Path:
    parameter = case["root"] / f"{name}.inp"
    _write_parameter(
        parameter, records=records, mode=1, model_prefix="model/start", grad_method=0,
        invmat1=case["invmat1"], inv_mod_out=1, inv_model_file=output, itermax=itermax,
        nprocx=nprocx, eps_scale=eps_scale, **options,
    )
    return parameter


def _iteration_files(case: dict, output: str, iteration: int) -> dict[str, Path]:
    prefix = f"{case['root'] / output}_stage_1_it_{iteration}"
    return {suffix: Path(f"{prefix}{suffix}") for suffix in case["suffixes"]}


def _diag_files(case: dict, output: str, iteration: int) -> dict[str, Path]:
    prefix = f"{case['root'] / output}_stage_1_it_{iteration}"
    primary = case["primary_suffix"][1:]
    return {
        f"{kind}_{name}": Path(f"{prefix}.{kind}_{suffix}")
        for kind in ("grad", "step")
        for name, suffix in zip(CLASSES, (primary, "rho", "qs"))
    }


def _assert_start_unchanged(case: dict) -> None:
    for suffix, raw in case["start_bytes"].items():
        assert case["start"].with_suffix(suffix).read_bytes() == raw


def _assert_configuration_abort(run: subprocess.CompletedProcess[str], case: dict,
                                output: str) -> None:
    # Open MPI may mask MPI_Abort's status; accept its abort transcript then.
    assert run.returncode != 0 or "MPI_ABORT was invoked" in run.stdout, run.stdout
    _assert_start_unchanged(case)
    assert not list((case["root"] / "model").glob(f"{Path(output).name}*"))


def _to_grid(values: tuple[float, ...]) -> list[list[float]]:
    """Merged model files are x-major: value(x, y) = values[x * NY + y]."""
    return [[values[x * DRIVER_N + y] for x in range(DRIVER_N)] for y in range(DRIVER_N)]


def _driver_mask() -> list[list[float]]:
    # FW=6, FREE_SURF=0, BOUNDARY=0 for the fixture; the global mask is the
    # 1x1 local mask because each decomposition divides the grid exactly.
    return ref.update_mask(DRIVER_N, DRIVER_N, 0, 0, 1, 1, DRIVER_FW, 0, 0)


# ---------------------------------------------------------------------------
# O9: parameter records 119 / 122 / 123..127
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("records", [119, 122, 123, 124, 125, 126, 127])
def test_o9_parameter_writer_emits_positional_records(tmp_path: Path, records: int) -> None:
    path = tmp_path / "parameters.inp"
    _write_parameter(path, records=records, opt_step_scaling=2,
                     opt_weights=(0.25, 0.5, 4.0), opt_diag_out=1)
    parsed = _parameter_records(path)
    assert len(parsed) == records
    assert parsed[115][0] == "Q_PARAMETERIZATION_MODE"
    assert parsed[118][0] == "Q_APPROX_DF"
    if records >= 122:
        assert [name for name, _ in parsed[119:122]] == [n for n, _ in MIGRATION_RECORDS]
    expected_optional = [
        ("OPT_STEP_SCALING", "2"), ("OPT_WEIGHT_PRIMARY", "0.25"),
        ("OPT_WEIGHT_RHO", "0.5"), ("OPT_WEIGHT_Q", "4.0"), ("OPT_DIAG_OUT", "1"),
    ]
    assert parsed[122:] == expected_optional[:max(0, records - 122)]


@pytest.mark.integration
def test_o9_o1_record_variants_keep_mode0_behaviour(
    tmp_path: Path, denise_binary: Path, mpiexec: str,
) -> None:
    """119, 122, 123/124 (mode 0, trailing defaults) and explicit-default 127
    records give bitwise identical models; defaults write no diagnostics;
    mode-0 diagnostics show p == g_raw bitwise."""
    case = _prepare_driver_case(tmp_path, denise_binary, mpiexec)
    variants = {
        "r119": dict(records=119),
        "r122": dict(records=122),
        "r123": dict(records=123, opt_step_scaling=0),
        "r124": dict(records=124, opt_step_scaling=0, opt_weights=(1.0, 1.0, 1.0)),
        "r127": dict(records=127),
        "r127_diag": dict(records=127, opt_diag_out=1),
    }
    models = {}
    for name, options in variants.items():
        output = f"model/{name}"
        parameter = _fwi_parameter(case, name, output=output, **options)
        run = _run_denise(denise_binary, mpiexec, case["root"], parameter, case["workflow"])
        assert run.returncode == 0, run.stdout
        assert "TDFWI ITERATION 1" in run.stdout
        models[name] = {s: p.read_bytes() for s, p in _iteration_files(case, output, 1).items()}
        diagnostics = _diag_files(case, output, 1)
        if name == "r127_diag":
            for path in diagnostics.values():
                assert path.stat().st_size == 4 * DRIVER_CELLS
                assert list(path.parent.glob(f"{path.name}.*.*")) == []
            for name_k in CLASSES:
                assert (diagnostics[f"step_{name_k}"].read_bytes()
                        == diagnostics[f"grad_{name_k}"].read_bytes())
        else:
            assert not any(path.exists() for path in diagnostics.values())
            assert not list((case["root"] / "model").glob(f"{name}_stage_1_it_1.grad_*"))
            assert not list((case["root"] / "model").glob(f"{name}_stage_1_it_1.step_*"))
    for name in variants:
        assert models[name] == models["r119"], name
    # The accepted start-model perturbation produced a real update.
    assert models["r119"][".qs"] != case["start_bytes"][".qs"]
    _assert_start_unchanged(case)


@pytest.mark.integration
def test_o9_missing_trailing_records_keep_defaults(
    tmp_path: Path, denise_binary: Path, mpiexec: str,
) -> None:
    """Shorter files equal a 127-record file carrying the documented defaults."""
    case = _prepare_driver_case(tmp_path, denise_binary, mpiexec)
    pairs = {
        "r123_rms": (dict(records=123, opt_step_scaling=1),
                     dict(records=127, opt_step_scaling=1, opt_weights=(1.0, 1.0, 1.0),
                          opt_diag_out=0)),
        "r125_max": (dict(records=125, opt_step_scaling=2, opt_weights=(1.0, 0.0, 1.0)),
                     dict(records=127, opt_step_scaling=2, opt_weights=(1.0, 0.0, 1.0),
                          opt_diag_out=0)),
        "r126_rms": (dict(records=126, opt_step_scaling=1, opt_weights=(1.0, 0.0, 0.0)),
                     dict(records=127, opt_step_scaling=1, opt_weights=(1.0, 0.0, 0.0),
                          opt_diag_out=0)),
    }
    models = {}
    for name, (short, full) in pairs.items():
        for label, options in (("short", short), ("full", full)):
            output = f"model/{name}_{label}"
            parameter = _fwi_parameter(case, f"{name}_{label}", output=output,
                                       eps_scale=SCALED_EPS, **options)
            run = _run_denise(denise_binary, mpiexec, case["root"], parameter,
                              case["workflow"])
            assert run.returncode == 0, run.stdout
            assert not any(path.exists() for path in _diag_files(case, output, 1).values())
            models[(name, label)] = {s: p.read_bytes()
                                     for s, p in _iteration_files(case, output, 1).items()}
        assert models[(name, "short")] == models[(name, "full")], name
    # Records 123..126 were read in order: zero-weight classes stay at the
    # start model, which the identity direction of mode 0 would not guarantee.
    assert models[("r125_max", "short")][".rho"] == case["start_bytes"][".rho"]
    assert models[("r126_rms", "short")][".rho"] == case["start_bytes"][".rho"]
    assert models[("r126_rms", "short")][".qs"] == case["start_bytes"][".qs"]
    _assert_start_unchanged(case)


CONFIGURATION_ABORTS = {
    "scaling_7": dict(opt_step_scaling=7),
    "scaling_minus_1": dict(opt_step_scaling=-1),
    "diag_2": dict(opt_diag_out=2),
    "mode0_weight_primary": dict(opt_weights=(0.5, 1.0, 1.0)),
    "mode0_weight_rho": dict(opt_weights=(1.0, 2.0, 1.0)),
    "mode0_weight_q": dict(opt_weights=(1.0, 1.0, 0.0)),
    "mode1_negative_weight": dict(opt_step_scaling=1, opt_weights=(1.0, -0.5, 1.0)),
    "mode1_nan_weight": dict(opt_step_scaling=1, opt_weights=(1.0, 1.0, "nan")),
    "mode2_inf_weight": dict(opt_step_scaling=2, opt_weights=("inf", 1.0, 1.0)),
    "mode1_all_zero_weights": dict(opt_step_scaling=1, opt_weights=(0.0, 0.0, 0.0)),
    "mode1_negative_inv_iter": dict(opt_step_scaling=1, workflow=dict(inv_rho_iter=-1)),
    "scaling_7_two_ranks": dict(opt_step_scaling=7, nprocx=2),
}


@pytest.mark.integration
@pytest.mark.parametrize("name", list(CONFIGURATION_ABORTS))
def test_o9_127_record_controls_abort_configuration(
    tmp_path: Path, denise_binary: Path, mpiexec: str, name: str,
) -> None:
    options = dict(CONFIGURATION_ABORTS[name])
    workflow_options = options.pop("workflow", {})
    nprocx = options.pop("nprocx", 1)
    case = _prepare_driver_case(tmp_path, denise_binary, mpiexec)
    _write_workflow(case["workflow"], **workflow_options)
    output = "model/must_not_be_accepted"
    parameter = _fwi_parameter(case, name, records=127, output=output, nprocx=nprocx,
                               eps_scale=SCALED_EPS, **options)
    run = _run_denise(denise_binary, mpiexec, case["root"], parameter, case["workflow"],
                      ranks=nprocx)
    _assert_configuration_abort(run, case, output)
    assert "TDFWI ITERATION" not in run.stdout


@pytest.mark.integration
def test_o9_mode0_keeps_existing_delayed_class_guard(
    tmp_path: Path, denise_binary: Path, mpiexec: str,
) -> None:
    case = _prepare_driver_case(tmp_path, denise_binary, mpiexec)
    _write_workflow(case["workflow"], inv_qs_iter=2)
    output = "model/must_not_be_accepted"
    parameter = _fwi_parameter(case, "mode0_guard", records=127, output=output)
    run = _run_denise(denise_binary, mpiexec, case["root"], parameter, case["workflow"])
    assert ("Exact viscoelastic SH FWI: INV_QS_ITER=2 at stage 1; supported values: 0 or 1."
            in run.stdout)
    _assert_configuration_abort(run, case, output)
    assert "TDFWI ITERATION" not in run.stdout


@pytest.mark.integration
def test_o9_all_effective_weights_zero_aborts_collectively(
    tmp_path: Path, denise_binary: Path, mpiexec: str,
) -> None:
    # Section 4: abort at the start of iteration 1, before its gradient.  With
    # OPT_DIAG_OUT=1 no gradient/step diagnostic and no model may be written.
    case = _prepare_driver_case(tmp_path, denise_binary, mpiexec)
    _write_workflow(case["workflow"], inv_vs_iter=3)
    output = "model/must_not_be_accepted"
    parameter = _fwi_parameter(case, "all_effective_zero", records=127, output=output,
                               itermax=2, nprocx=2, eps_scale=SCALED_EPS,
                               opt_step_scaling=1, opt_weights=(1.0, 0.0, 0.0),
                               opt_diag_out=1)
    run = _run_denise(denise_binary, mpiexec, case["root"], parameter, case["workflow"],
                      ranks=2)
    _assert_configuration_abort(run, case, output)
    assert "TDFWI ITERATION 2" not in run.stdout


@pytest.mark.integration
@pytest.mark.parametrize("weights,changed", [
    ((1.0, 0.0, 0.0), "primary"), ((0.0, 1.0, 0.0), "rho"), ((0.0, 0.0, 1.0), "q"),
], ids=["primary_only", "rho_only", "q_only"])
def test_o9_weight_records_select_their_classes(
    tmp_path: Path, denise_binary: Path, mpiexec: str,
    weights: tuple[float, float, float], changed: str,
) -> None:
    case = _prepare_driver_case(tmp_path, denise_binary, mpiexec)
    output = f"model/weights_{changed}"
    parameter = _fwi_parameter(case, f"weights_{changed}", records=127, output=output,
                               eps_scale=SCALED_EPS, opt_step_scaling=1, opt_weights=weights)
    run = _run_denise(denise_binary, mpiexec, case["root"], parameter, case["workflow"])
    assert run.returncode == 0, run.stdout
    files = _iteration_files(case, output, 1)
    for name, suffix in zip(CLASSES, case["suffixes"]):
        raw = files[suffix].read_bytes()
        if name == changed:
            assert raw != case["start_bytes"][suffix], f"{name} did not change"
        else:
            assert raw == case["start_bytes"][suffix], f"{name} changed with zero weight"
    _assert_start_unchanged(case)


# ---------------------------------------------------------------------------
# O7: delayed rho/Q classes keep the iteration-1 model bitwise
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_o7_delayed_rho_and_q_are_bitwise_frozen_while_vs_changes(
    tmp_path: Path, denise_binary: Path, mpiexec: str,
) -> None:
    case = _prepare_driver_case(tmp_path, denise_binary, mpiexec)
    _write_workflow(case["workflow"], inv_vs_iter=0, inv_rho_iter=3, inv_qs_iter=3)
    output = "model/o7_delayed"
    parameter = _fwi_parameter(case, "o7_delayed", records=127, output=output, itermax=2,
                               eps_scale=SCALED_EPS, opt_step_scaling=1,
                               opt_weights=(1.0, 1.0, 1.0), opt_diag_out=0)
    run = _run_denise(denise_binary, mpiexec, case["root"], parameter, case["workflow"])
    assert run.returncode == 0, run.stdout
    assert "TDFWI ITERATION 2" in run.stdout
    mask = _driver_mask()
    start_vs = _to_grid(_read_native_model(case["start"].with_suffix(".vs")))
    for iteration in (1, 2):
        files = _iteration_files(case, output, iteration)
        # INV_RHO_ITER = INV_QS_ITER = 3 > iteration: effective weight 0.
        assert files[".rho"].read_bytes() == case["start_bytes"][".rho"]
        assert files[".qs"].read_bytes() == case["start_bytes"][".qs"]
        vs = _to_grid(_read_native_model(files[".vs"]))
        # O6 at driver level: CPML cells (M = 0) keep the start Vs bitwise.
        for y in range(DRIVER_N):
            for x in range(DRIVER_N):
                if mask[y][x] == 0.0:
                    assert ref.f32_bits(vs[y][x]) == ref.f32_bits(start_vs[y][x]), (x, y)
    assert _iteration_files(case, output, 1)[".vs"].read_bytes() != case["start_bytes"][".vs"]
    _assert_start_unchanged(case)


# ---------------------------------------------------------------------------
# Driver diagnostics (section 4): written direction obeys the O2 closed form
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.parametrize("mode,invmat1,nprocx", [(1, 1, 2), (2, 1, 1), (1, 3, 1)],
                         ids=["rms_vs_two_ranks", "max_vs", "rms_mu"])
def test_o2_o6_driver_diagnostics_match_closed_form(
    tmp_path: Path, denise_binary: Path, mpiexec: str, mode: int, invmat1: int, nprocx: int,
) -> None:
    case = _prepare_driver_case(tmp_path, denise_binary, mpiexec, invmat1=invmat1)
    output = "model/diag"
    weights = (1.0, 0.5, 2.0)
    parameter = _fwi_parameter(case, "diag", records=127, output=output, nprocx=nprocx,
                               eps_scale=SCALED_EPS, opt_step_scaling=mode,
                               opt_weights=weights, opt_diag_out=1)
    run = _run_denise(denise_binary, mpiexec, case["root"], parameter, case["workflow"],
                      ranks=nprocx)
    assert run.returncode == 0, run.stdout
    files = _diag_files(case, output, 1)
    for path in files.values():
        assert path.stat().st_size == 4 * DRIVER_CELLS
        assert list(path.parent.glob(f"{path.name}.*.*")) == []
    mask = _driver_mask()
    fields = {"mask": mask}
    for name, suffix in zip(CLASSES, case["suffixes"]):
        fields[f"m_{name}"] = _to_grid(_read_native_model(case["start"].with_suffix(suffix)))
        fields[f"g_{name}"] = _to_grid(_read_native_model(files[f"grad_{name}"]))
    driver_case = _case(mode, weights, **fields)
    observed = {"step": {name: _to_grid(_read_native_model(files[f"step_{name}"]))
                         for name in CLASSES}}
    _assert_closed_form(observed, driver_case)
