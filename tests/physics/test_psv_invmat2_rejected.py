from __future__ import annotations

import hashlib
import json
import os
from array import array
from pathlib import Path

import pytest

from tests.cases.psv_fwi_gradient import (
    PSVFWIGradientConfig,
    baseline_model,
    generate_case,
)
from tests.utilities.runner import result_summary, run_denise


pytestmark = pytest.mark.integration
DIAGNOSTIC = (
    "PSV INVMAT1=2 (Zp/Zs/rho impedance parameterization) is unsupported: "
    "the legacy PSV model-input/file contract is undefined. Use PSV "
    "INVMAT1=1 or INVMAT1=3 where applicable."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_record(directory: Path, key: str, value: str, *, leading_space: bool = False) -> None:
    path = directory / "denise.inp"
    lines = path.read_text(encoding="ascii").splitlines()
    matches = [
        index for index, line in enumerate(lines)
        if not line.lstrip().startswith("#")
        and line.split("=", 1)[0].strip() == key
    ]
    assert len(matches) == 1, (key, matches)
    lines[matches[0]] = f"{' ' if leading_space else ''}{key} ={value}"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_grid(path: Path, values: list[float]) -> None:
    with path.open("wb") as stream:
        array("f", values).tofile(stream)


def _configure_invmat2(directory: Path, *, viscoelastic: bool,
                       config: PSVFWIGradientConfig) -> None:
    _set_record(directory, "INVMAT1", "2")
    _set_record(directory, "READMOD", "0")
    if viscoelastic:
        _set_record(directory, "L", "1", leading_space=True)
        _set_record(directory, "FL", "10.0")
        _write_grid(directory / "model/current.qp", [80.0] * config.cell_count)
        _write_grid(directory / "model/current.qs", [50.0] * config.cell_count)


def _configure_invmat3(directory: Path, model: dict[str, list[float]]) -> None:
    _set_record(directory, "INVMAT1", "3")
    lam = [rho * (vp * vp - 2.0 * vs * vs)
           for vp, vs, rho in zip(model["vp"], model["vs"], model["rho"])]
    mu = [rho * vs * vs for vs, rho in zip(model["vs"], model["rho"])]
    _write_grid(directory / "model/current.lam", lam)
    _write_grid(directory / "model/current.mu", mu)


def _run(directory: Path, *, repository_root: Path, denise_binary: Path,
         mpiexec: str, config: PSVFWIGradientConfig, role: str,
         ranks: int = 1):
    return run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=ranks,
        configuration=config.as_metadata() | {"role": role, "ranks": ranks},
        timeout_seconds=90.0,
    )


@pytest.mark.parametrize("viscoelastic", [False, True], ids=["elastic", "viscoelastic"])
def test_psv_invmat2_rejection_is_deterministic(
    tmp_path: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    viscoelastic: bool,
) -> None:
    config = PSVFWIGradientConfig()
    model = baseline_model(config)
    returncodes: list[int] = []

    for repetition in range(3):
        directory = tmp_path / f"run_{repetition}"
        generate_case(directory, model=model, config=config, mode=0)
        _configure_invmat2(directory, viscoelastic=viscoelastic, config=config)
        result = _run(
            directory, repository_root=repository_root,
            denise_binary=denise_binary, mpiexec=mpiexec, config=config,
            role=f"invmat2_{'visco' if viscoelastic else 'elastic'}_{repetition}",
        )
        returncodes.append(result.returncode)
        output = result.stdout_path.read_text(encoding="utf-8", errors="replace")
        output += result.stderr_path.read_text(encoding="utf-8", errors="replace")
        assert DIAGNOSTIC in output, result_summary(result)
        assert not list((directory / "su").glob("*.su*"))

    assert len(set(returncodes)) == 1
    assert returncodes[0] != 0


@pytest.mark.parametrize("viscoelastic", [False, True], ids=["elastic", "viscoelastic"])
def test_psv_invmat2_two_rank_rejection_terminates_cleanly(
    tmp_path: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    viscoelastic: bool,
) -> None:
    config = PSVFWIGradientConfig()
    directory = tmp_path / "run"
    generate_case(
        directory,
        model=baseline_model(config),
        config=config,
        mode=0,
        nprocx=2,
        nprocy=1,
    )
    _configure_invmat2(directory, viscoelastic=viscoelastic, config=config)
    result = _run(
        directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role=f"invmat2_{'visco' if viscoelastic else 'elastic'}_two_rank",
        ranks=2,
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    output = result.stdout_path.read_text(encoding="utf-8", errors="replace")
    output += result.stderr_path.read_text(encoding="utf-8", errors="replace")

    assert metadata["timed_out"] is False, result_summary(result)
    assert result.returncode != 0, result_summary(result)
    assert DIAGNOSTIC in output, result_summary(result)
    assert not list((directory / "su").glob("*.su*"))


def test_supported_psv_forward_parameterizations_remain_operational(
    tmp_path: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
) -> None:
    config = PSVFWIGradientConfig()
    model = baseline_model(config)
    binaries = [("repaired", denise_binary)]
    configured_base = os.environ.get("M541A_BASE_DENISE_BIN")
    if configured_base:
        binaries.append(("base", Path(configured_base).resolve(strict=True)))
    hashes: dict[str, dict[int, list[dict[str, str]]]] = {
        label: {1: [], 3: []} for label, _ in binaries
    }

    for label, binary in binaries:
        for parameterization in (1, 3):
            for repetition in range(2):
                directory = tmp_path / label / f"invmat{parameterization}_{repetition}"
                generate_case(directory, model=model, config=config, mode=0)
                if parameterization == 3:
                    _configure_invmat3(directory, model)
                result = _run(
                    directory, repository_root=repository_root,
                    denise_binary=binary, mpiexec=mpiexec, config=config,
                    role=f"{label}_invmat{parameterization}_supported_{repetition}",
                )
                assert result.returncode == 0, result_summary(result)
                outputs = {
                    component: _sha256(directory / f"su/synthetic_{component}.su.shot1")
                    for component in ("x", "y")
                }
                assert all(
                    (directory / f"su/synthetic_{component}.su.shot1").stat().st_size > 0
                    for component in ("x", "y")
                )
                hashes[label][parameterization].append(outputs)

    assert hashes["repaired"][1][0] == hashes["repaired"][1][1]
    assert hashes["repaired"][3][0] == hashes["repaired"][3][1]
    if configured_base:
        assert hashes["base"][1] == hashes["repaired"][1]
        assert hashes["base"][3] == hashes["repaired"][3]
