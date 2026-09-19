"""M8d-1B distributed exact-viscoelastic P/SV Active-FWI gate."""

from __future__ import annotations

import json
import math
import shutil
from array import array
from dataclasses import replace
from pathlib import Path

import pytest

from tests.physics.test_visco_psv_active_fwi import _point_model_at
from tests.physics.test_visco_psv_physical_gradient_oracle import (
    PHYSICAL_FIELDS,
    ViscoPSVOracleConfig,
    _objective,
    _reference_model,
    _target_model,
    _write_case,
)
from tests.utilities.fwi_gradient import read_su_float_samples
from tests.utilities.runner import result_summary, run_denise


pytestmark = pytest.mark.integration


def _read_floats(path: Path, count: int) -> array:
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, count)
        assert not stream.read(1), f"unexpected trailing bytes in {path}"
    assert len(values) == count
    return values


def _spike_non_root_maxima(model, config: ViscoPSVOracleConfig):
    result = {field: list(values) for field, values in model.items()}
    index = (config.nx - 3) * config.ny + (config.ny - 3)
    for field in PHYSICAL_FIELDS:
        result[field][index] = max(result[field]) * 1.05
    return result


def _run_active(
    directory: Path,
    *,
    config: ViscoPSVOracleConfig,
    model,
    observed_directory: Path,
    nprocx: int,
    nprocy: int,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    invalid_rank: int | None = None,
):
    ranks = nprocx * nprocy
    _write_case(
        directory,
        config=config,
        model=model,
        mode=1,
        observed=observed_directory,
        nprocx=nprocx,
        nprocy=nprocy,
    )
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("DENISE_PSV_EXACT_VISCO_GRADIENT", "1")
        environment.setenv("DENISE_PSV_EXACT_SEGMENTS", "32")
        environment.delenv(
            "DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY", raising=False
        )
        if invalid_rank is None:
            environment.delenv(
                "DENISE_PSV_EXACT_TEST_INVALID_TRIAL_RANK", raising=False
            )
        else:
            environment.setenv(
                "DENISE_PSV_EXACT_TEST_INVALID_TRIAL_RANK", str(invalid_rank)
            )
        result = run_denise(
            repository_root=repository_root,
            case_directory=directory,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            ranks=ranks,
            configuration={
                "gate": "M8d-1B",
                "decomposition": [nprocx, nprocy],
                "invalid_rank": invalid_rank,
            },
            timeout_seconds=240.0,
        )
    assert result.returncode == 0, result_summary(result)
    report = json.loads(
        (directory / "jacobian" / "gradient.active_fwi.json").read_text(
            encoding="utf-8"
        )
    )
    persisted_prefix = directory / report["persisted_prefix"]
    return {
        "directory": directory,
        "report": report,
        "gradients": {
            field: _read_floats(
                directory / "jacobian" / f"gradient.raw.{field}",
                config.cell_count,
            )
            for field in PHYSICAL_FIELDS
        },
        "accepted": {
            field: _read_floats(
                Path(f"{persisted_prefix}.{field}"), config.cell_count
            )
            for field in PHYSICAL_FIELDS
        },
        "persisted_prefix": persisted_prefix,
    }


def _reload(
    directory: Path,
    *,
    source_run,
    observed,
    config: ViscoPSVOracleConfig,
    nprocx: int,
    nprocy: int,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
):
    _write_case(
        directory,
        config=config,
        model=source_run["accepted"],
        mode=0,
        nprocx=nprocx,
        nprocy=nprocy,
    )
    reload_prefix = directory / "model" / "accepted"
    for field in PHYSICAL_FIELDS:
        shutil.copyfile(
            Path(f"{source_run['persisted_prefix']}.{field}"),
            Path(f"{reload_prefix}.{field}"),
        )
    _point_model_at(directory / "denise.inp", "model/accepted")
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration={
            "gate": "M8d-1B",
            "role": "accepted_reload",
            "decomposition": [nprocx, nprocy],
        },
        timeout_seconds=120.0,
    )
    assert result.returncode == 0, result_summary(result)
    suffix = {"vp": "pi", "vs": "mu", "rho": "rho", "qp": "qp", "qs": "qs"}
    reloaded = {
        field: _read_floats(
            Path(f"{reload_prefix}.fdveps.{suffix[field]}"), config.cell_count
        )
        for field in PHYSICAL_FIELDS
    }
    synthetic = {
        component: read_su_float_samples(
            directory / "su" / f"synthetic_{component}.su.shot1",
            config.receiver_count,
            config.samples_per_trace,
        )
        for component in ("x", "y")
    }
    return reloaded, _objective(synthetic, observed)


def _trial_projection(report, key: str):
    return [trial[key] for trial in report["trials"]]


def _changed_in_every_rank(initial, accepted, config, nprocx, nprocy, field):
    width = config.nx // nprocx
    height = config.ny // nprocy
    for px in range(nprocx):
        for py in range(nprocy):
            changed = False
            for ix in range(px * width, (px + 1) * width):
                for iy in range(py * height, (py + 1) * height):
                    index = ix * config.ny + iy
                    if initial[field][index] != accepted[field][index]:
                        changed = True
                        break
                if changed:
                    break
            assert changed, f"{field} unchanged on rank tile {(px, py)}"


@pytest.fixture(scope="module")
def distributed_active_runs(tmp_path_factory, repository_root, denise_binary, mpiexec):
    root = tmp_path_factory.mktemp("m8d1b_distributed_active_fwi")
    config = replace(
        ViscoPSVOracleConfig(),
        source_x_m=240.0,
        source_y_m=220.0,
        receivers_m=(
            (240.0, 220.0),
            (250.0, 140.0),
            (290.0, 180.0),
            (330.0, 230.0),
        ),
    )
    model = _spike_non_root_maxima(_reference_model(config), config)
    observed_directory = root / "observed_truth"
    _write_case(
        observed_directory,
        config=config,
        model=_target_model(model, config),
        mode=0,
    )
    truth = run_denise(
        repository_root=repository_root,
        case_directory=observed_directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration={"gate": "M8d-1B", "role": "observed_truth"},
        timeout_seconds=120.0,
    )
    assert truth.returncode == 0, result_summary(truth)
    observed = {
        component: read_su_float_samples(
            observed_directory / "su" / f"synthetic_{component}.su.shot1",
            config.receiver_count,
            config.samples_per_trace,
        )
        for component in ("x", "y")
    }
    layouts = {
        "one": (1, 1),
        "horizontal": (2, 1),
        "vertical": (1, 2),
        "four": (2, 2),
    }
    runs = {
        name: _run_active(
            root / name,
            config=config,
            model=model,
            observed_directory=observed_directory,
            nprocx=nprocx,
            nprocy=nprocy,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
        )
        for name, (nprocx, nprocy) in layouts.items()
    }
    reloads = {
        name: _reload(
            root / f"{name}_reload",
            source_run=runs[name],
            observed=observed,
            config=config,
            nprocx=nprocx,
            nprocy=nprocy,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
        )
        for name, (nprocx, nprocy) in layouts.items()
    }
    return {
        "root": root,
        "config": config,
        "model": model,
        "observed_directory": observed_directory,
        "runs": runs,
        "reloads": reloads,
    }


def test_complete_distributed_active_lifecycle_is_decomposition_invariant(
    distributed_active_runs,
):
    config = distributed_active_runs["config"]
    model = distributed_active_runs["model"]
    runs = distributed_active_runs["runs"]
    reloads = distributed_active_runs["reloads"]
    reference = runs["one"]
    reference_report = reference["report"]
    for name, nprocx, nprocy in (
        ("horizontal", 2, 1),
        ("vertical", 1, 2),
        ("four", 2, 2),
    ):
        candidate = runs[name]
        report = candidate["report"]
        assert report["decomposition"] == [nprocx, nprocy]
        assert report["collective_communicator"] == "SHOT_COMM"
        assert report["model_maxima"] == reference_report["model_maxima"]
        assert report["gradient_maxima"] == reference_report["gradient_maxima"]
        assert any(rank != 0 for rank in report["model_max_owner_ranks"].values())
        assert _trial_projection(report, "alpha") == _trial_projection(
            reference_report, "alpha"
        )
        assert _trial_projection(report, "valid") == _trial_projection(
            reference_report, "valid"
        )
        assert _trial_projection(report, "accepted") == _trial_projection(
            reference_report, "accepted"
        )
        assert report["accepted_alpha"] == reference_report["accepted_alpha"]
        assert report["decision_consensus"] is True
        assert report["line_search_control_identical"] is True
        assert report["trial_forwards_min"] == report["trial_forwards_max"]
        assert report["trial_forwards_forward_only"] is True
        assert report["global_field_replication"] is False
        assert math.isclose(
            report["base_objective"],
            reference_report["base_objective"],
            rel_tol=1.0e-14,
            abs_tol=0.0,
        )
        assert math.isclose(
            report["accepted_objective"],
            reference_report["accepted_objective"],
            rel_tol=1.0e-14,
            abs_tol=0.0,
        )
        for trial, reference_trial in zip(
            report["trials"], reference_report["trials"], strict=True
        ):
            assert math.isclose(
                trial["objective"],
                reference_trial["objective"],
                rel_tol=1.0e-14,
                abs_tol=0.0,
            )
        for field in PHYSICAL_FIELDS:
            assert math.isclose(
                report["update_norms"][field],
                reference_report["update_norms"][field],
                rel_tol=1.0e-14,
                abs_tol=0.0,
            )
        for field in PHYSICAL_FIELDS:
            assert candidate["gradients"][field].tobytes() == reference["gradients"][field].tobytes()
            assert candidate["accepted"][field].tobytes() == reference["accepted"][field].tobytes()
            assert reloads[name][0][field].tobytes() == candidate["accepted"][field].tobytes()
        for field in ("qp", "qs"):
            _changed_in_every_rank(
                model, candidate["accepted"], config, nprocx, nprocy, field
            )
        assert math.isclose(
            reloads[name][1],
            report["accepted_objective"],
            rel_tol=2.0e-6,
            abs_tol=1.0e-10,
        )
    for field in PHYSICAL_FIELDS:
        assert reloads["one"][0][field].tobytes() == reference["accepted"][field].tobytes()
    assert math.isclose(
        reloads["one"][1],
        reference_report["accepted_objective"],
        rel_tol=2.0e-6,
        abs_tol=1.0e-10,
    )


def test_non_root_global_maximum_is_used(distributed_active_runs):
    runs = distributed_active_runs["runs"]
    reference = runs["one"]["report"]
    for name in ("horizontal", "vertical", "four"):
        report = runs[name]["report"]
        assert report["model_maxima"] == reference["model_maxima"]
        assert all(rank != 0 for rank in report["model_max_owner_ranks"].values())


def test_single_rank_invalid_trial_is_rejected_collectively(
    distributed_active_runs, repository_root, denise_binary, mpiexec
):
    state = distributed_active_runs
    run = _run_active(
        state["root"] / "invalid_rank_one",
        config=state["config"],
        model=state["model"],
        observed_directory=state["observed_directory"],
        nprocx=2,
        nprocy=1,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        invalid_rank=1,
    )
    report = run["report"]
    first, second = report["trials"][:2]
    assert first["valid"] is False
    assert first["invalid_rank_count"] == 1
    assert first["forward_executed"] is False
    assert first["accepted"] is False
    assert second["alpha"] == first["alpha"] / 2.0
    assert second["valid"] is True
    assert second["forward_executed"] is True
    assert second["accepted"] is True
    assert report["trial_forwards_min"] == report["trial_forwards_max"] == 1
    assert report["line_search_control_identical"] is True
