"""Active exact-viscoelastic P/SV FWI lifecycle acceptance test."""

from __future__ import annotations

import json
import math
import shutil
from array import array
from pathlib import Path

import pytest

from tests.physics.test_visco_psv_physical_gradient_oracle import (
    PHYSICAL_FIELDS,
    ViscoPSVOracleConfig,
    _objective,
    _reference_model,
    _run_forward,
    _target_model,
    _write_case,
)
from tests.utilities.runner import result_summary, run_denise


pytestmark = pytest.mark.integration


def _read_float_grid(path: Path, cell_count: int) -> array:
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, cell_count)
        assert not stream.read(1), f"unexpected trailing data in {path}"
    assert len(values) == cell_count
    return values


def _point_model_at(parameter_file: Path, prefix: str) -> None:
    text = parameter_file.read_text(encoding="ascii")
    old = "MFILE =model/current"
    assert text.count(old) == 1
    parameter_file.write_text(text.replace(old, f"MFILE ={prefix}"), encoding="ascii")


def test_active_exact_visco_psv_step_persists_and_reloads(
    tmp_path, repository_root, denise_binary, mpiexec, monkeypatch
):
    config = ViscoPSVOracleConfig()
    base_model = _reference_model(config)

    truth_directory = tmp_path / "truth"
    _write_case(
        truth_directory,
        config=config,
        model=_target_model(base_model, config),
        mode=0,
    )
    observed = _run_forward(
        truth_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
    )

    fwi_directory = tmp_path / "fwi"
    _write_case(
        fwi_directory,
        config=config,
        model=base_model,
        mode=1,
        observed=truth_directory,
    )
    monkeypatch.delenv("DENISE_PSV_EXACT_VISCO_GRADIENT", raising=False)
    result = run_denise(
        repository_root=repository_root,
        case_directory=fwi_directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration={"oracle": "M7c-2", "mode": 1, "role": "active_fwi"},
        timeout_seconds=90.0,
    )
    assert result.returncode == 0, result_summary(result)

    report = json.loads(
        (fwi_directory / "jacobian" / "gradient.active_fwi.json").read_text(
            encoding="utf-8"
        )
    )
    assert math.isfinite(report["base_objective"])
    assert math.isfinite(report["accepted_alpha"])
    assert report["accepted_alpha"] > 0.0
    assert report["accepted_objective"] < report["base_objective"]
    assert report["trials"]
    assert any(
        math.isclose(
            trial["alpha"], report["accepted_alpha"], rel_tol=0.0, abs_tol=1.0e-15
        )
        and math.isclose(
            trial["objective"],
            report["accepted_objective"],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for trial in report["trials"]
    )
    assert set(report["update_norms"]) == set(PHYSICAL_FIELDS)
    assert all(
        math.isfinite(report["update_norms"][name])
        and report["update_norms"][name] > 0.0
        for name in PHYSICAL_FIELDS
    )

    persisted_prefix = fwi_directory / report["persisted_prefix"]
    accepted = {
        name: _read_float_grid(
            Path(f"{persisted_prefix}.{name}"), config.cell_count
        )
        for name in PHYSICAL_FIELDS
    }
    initial = {
        name: _read_float_grid(
            fwi_directory / "model" / f"current.{name}", config.cell_count
        )
        for name in PHYSICAL_FIELDS
    }
    assert all(math.isfinite(value) for values in accepted.values() for value in values)
    assert accepted["qp"] != initial["qp"]
    assert accepted["qs"] != initial["qs"]

    reload_directory = tmp_path / "reload"
    _write_case(
        reload_directory,
        config=config,
        model=accepted,
        mode=0,
    )
    reload_prefix = reload_directory / "model" / "accepted"
    for name in PHYSICAL_FIELDS:
        shutil.copyfile(
            Path(f"{persisted_prefix}.{name}"), Path(f"{reload_prefix}.{name}")
        )
    _point_model_at(reload_directory / "denise.inp", "model/accepted")
    reloaded_synthetic = _run_forward(
        reload_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
    )

    reader_suffix = {
        "vp": "pi",
        "vs": "mu",
        "rho": "rho",
        "qp": "qp",
        "qs": "qs",
    }
    for name in PHYSICAL_FIELDS:
        reloaded = _read_float_grid(
            Path(f"{reload_prefix}.fdveps.{reader_suffix[name]}"),
            config.cell_count,
        )
        assert reloaded == accepted[name]

    reloaded_objective = _objective(reloaded_synthetic, observed)
    assert math.isclose(
        reloaded_objective,
        report["accepted_objective"],
        rel_tol=2.0e-6,
        abs_tol=1.0e-10,
    )

    (tmp_path / "m7c2_active_fwi_report.json").write_text(
        json.dumps(
            {
                **report,
                "accepted_fields_finite": True,
                "reload_matches": {name: True for name in PHYSICAL_FIELDS},
                "reloaded_objective": reloaded_objective,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
