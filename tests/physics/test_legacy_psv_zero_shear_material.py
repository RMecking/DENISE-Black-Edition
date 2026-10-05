"""FLUID-3B: actual legacy material routines, fenv and frozen BASE identity.

Only helper/material maps are verified here. No wave/J/JT/CUDA claim.
For Windows-created WSL worktrees, supply DENISE_AV_MUE_BASE_SOURCE containing
the exact BASE source; ordinary Git checkouts obtain it directly with git show.
DENISE_AV_MUE_SOURCE is solely an explicit negative-control source override.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

BASE = "54c69297a6c1d5fef94c396b49451ebaadcec2ab"
BASE_BLOB = "359b7b5c1853ca812e6738854fa74fc70e7bd96d"


def record(**values):
    destination = os.environ.get("DENISE_FLUID3B_EVIDENCE")
    if destination:
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(values, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def helper_builder(repository_root, tmp_path_factory):
    directory = tmp_path_factory.mktemp("legacy-zero-shear-build")
    supplied = os.environ.get("DENISE_AV_MUE_BASE_SOURCE")
    if supplied:
        data = Path(supplied).read_bytes()
    else:
        data = subprocess.run(
            ["git", "show", f"{BASE}:src/av_mue.c"], cwd=repository_root,
            capture_output=True, check=True,
        ).stdout
    # Git text checkout newline conversion must not change the frozen content.
    data = data.replace(b"\r\n", b"\n")
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    assert blob == BASE_BLOB, "all-solid reference must be the exact canonical BASE"
    baseline = directory / "av_mue_base.c"
    baseline.write_bytes(data)
    source = Path(os.environ.get("DENISE_AV_MUE_SOURCE",
                                  str(repository_root / "src/av_mue.c")))
    compiler = os.environ.get("MPICC", "mpicc")
    built = {}

    def build(opt):
        if opt in built:
            return built[opt]
        flags = [compiler, "-std=c99", f"-{opt}", "-fcommon", "-ffunction-sections",
                 "-fno-stack-protector", "-D_FORTIFY_SOURCE=0",
                 "-I" + str(repository_root / "include")]
        base_object = directory / f"baseline-{opt}.o"
        subprocess.run(flags + ["-Dav_mue=av_mue_base", "-c", str(baseline),
                                "-o", str(base_object)], check=True)
        executable = directory / f"helper-{opt}"
        command = flags + [
            str(repository_root / "tests/utilities/legacy_psv_zero_shear_material.c"),
            str(source), str(repository_root / "src/PSV/matcopy_elastic_PSV.c"),
            str(repository_root / "src/util.c"), str(base_object),
            "-Wl,--gc-sections", "-lm", "-o", str(executable),
        ]
        subprocess.run(command, check=True)
        record(kind="build", optimization=opt, command=command,
               baseline_blob=blob, source=str(source),
               input_sha256={str(path.relative_to(repository_root)):
                   hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in [repository_root / "src/av_mue.c",
                                repository_root / "tests/utilities/legacy_psv_zero_shear_material.c",
                                Path(__file__), repository_root / "src/PSV/matcopy_elastic_PSV.c",
                                repository_root / "src/util.c"]})
        built[opt] = executable
        return executable
    return build


@pytest.fixture(scope="module", params=["O0", "O3"])
def scalar_helper(request, helper_builder):
    return request.param, helper_builder(request.param)


def local(helper, mode, kind, index):
    opt, executable = helper
    result = subprocess.run([str(executable), "local", str(mode), kind, str(index)],
                            capture_output=True, text=True, check=True, timeout=10)
    values = json.loads(result.stdout)
    record(kind=kind, optimization=opt, index=index, **values)
    return values


@pytest.mark.parametrize("mode", [1, 3])
@pytest.mark.parametrize("occupancy", range(16))
def test_occupancy(scalar_helper, mode, occupancy):
    values = local(scalar_helper, mode, "occupancy", occupancy)
    assert values["divide"] == 0, values
    assert values["invalid"] == 0, values
    if occupancy != 15:
        assert values["bits"] == 0, values  # exact positive zero, including sign bit
    else:
        assert values["bits"] == values["base_bits"] > 0, values


@pytest.mark.parametrize("mode", [1, 3])
@pytest.mark.parametrize("case", range(4), ids=["asymmetric", "wide", "small", "subnormal"])
def test_all_solid_bit_identity_and_small_classification(scalar_helper, mode, case):
    values = local(scalar_helper, mode, "solid", case)
    assert all(0 < value < 0x7F800000 for value in values["mu_bits"]), values
    assert values["bits"] == values["base_bits"] > 0, values
    assert not values["divide"] and not values["invalid"], values
    assert not values["base_divide"] and not values["base_invalid"], values
    if case == 3:
        assert all(value < 0x00800000 for value in values["mu_bits"]), values


def test_positive_vs_converted_fp32_zero_is_fluid(scalar_helper):
    values = local(scalar_helper, 1, "underflow", 0)
    assert values["mu_bits"][0] == 0, values
    assert values["bits"] == 0, values
    assert not values["divide"] and not values["invalid"], values


@pytest.mark.parametrize("mode", [1, 3])
def test_signed_zero_contributor_returns_positive_zero(scalar_helper, mode):
    values = local(scalar_helper, mode, "signed_zero", 0)
    assert values["bits"] == 0, values
    assert not values["divide"] and not values["invalid"], values


@pytest.mark.parametrize("opt,mode", [("O0", 1), ("O0", 3), ("O3", 3)])
def test_explicit_base_negative_control(helper_builder, opt, mode):
    # At O3 GCC can elide the BASE velocity branch's unused reciprocals.
    # Candidate fenv requirements above remain strict for BOTH branches at
    # BOTH levels; these controls select observable, unrepaired BASE failures.
    values = local((opt, helper_builder(opt)), mode, "occupancy", 14)
    assert values["base_divide"] == 1 and values["base_invalid"] == 0, values
    assert values["divide"] == 0 and values["invalid"] == 0, values
    assert values["bits"] == values["base_bits"] == 0, values


def global_material(mode, pattern):
    y, x = np.indices((8, 8))
    if pattern == "horizontal":
        fluid = y < 4
    elif pattern == "vertical":
        fluid = x < 4
    else:
        fluid = y < 4 + (x >= 4)
    rho = (1 + .125 * ((x + 2 * y) % 5)).astype(np.float32)
    primary = np.where(fluid, 0, 1 + .125 * x + .0625 * y).astype(np.float32)
    mu = (rho * primary * primary).astype(np.float32) if mode == 1 else primary
    return mu


def frozen_corner_map(mu):
    # Independent exact four-contributor specification. No production output
    # enters this construction; reciprocals exist only on strictly solid corners.
    contributors = [mu, np.roll(mu, -1, 1), np.roll(mu, -1, 0),
                    np.roll(np.roll(mu, -1, 0), -1, 1)]
    solid = np.logical_and.reduce([entry > 0 for entry in contributors])
    result = np.zeros(mu.shape, dtype=np.float32)
    a, b, c, d = [entry[solid].astype(np.float64) for entry in contributors]
    result[solid] = (4.0 / (1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)).astype(np.float32)
    return result


@pytest.mark.parametrize("topology", [(1, 1), (2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("pattern", ["horizontal", "vertical", "offset"])
@pytest.mark.parametrize("mode", [1, 3])
def test_actual_material_halos(helper_builder, mpiexec, tmp_path, topology, pattern, mode):
    executable = helper_builder("O3")
    output = tmp_path / "corners.bin"
    px, py = topology
    env = os.environ.copy()
    env["OMPI_MCA_rmaps_base_oversubscribe"] = "1"
    command = [mpiexec, "-n", str(px * py), str(executable), "halo",
               str(px), str(py), str(mode), pattern, str(output)]
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    values = json.loads(run.stdout)
    assert not values["halo_mismatch"], values
    assert not values["divide"] and not values["invalid"], values
    actual = np.fromfile(output, dtype=np.float32).reshape(8, 8)
    expected = frozen_corner_map(global_material(mode, pattern))
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    record(kind="halo", topology=topology, pattern=pattern, mode=mode,
           optimization="O3", cells=64, bit_identical=True, command=command, **values)
