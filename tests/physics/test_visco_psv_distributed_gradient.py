"""M8d-1A distributed exact-viscoelastic P/SV gradient gate."""

from __future__ import annotations

import json
import math
from array import array
from dataclasses import replace
from pathlib import Path

import pytest

from tests.physics.test_visco_psv_physical_gradient_oracle import (
    PHYSICAL_FIELDS,
    ViscoPSVOracleConfig,
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


def _run_gradient(
    directory: Path,
    *,
    config: ViscoPSVOracleConfig,
    model,
    observed_directory: Path,
    nprocx: int,
    nprocy: int,
    segments: int,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
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
        environment.setenv("DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY", "1")
        environment.setenv("DENISE_PSV_EXACT_SEGMENTS", str(segments))
        result = run_denise(
            repository_root=repository_root,
            case_directory=directory,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            ranks=ranks,
            configuration={
                "gate": "M8d-1A",
                "decomposition": [nprocx, nprocy],
                "segments": segments,
            },
            timeout_seconds=240.0,
        )
    assert result.returncode == 0, result_summary(result)
    gradients = {
        field: _read_floats(
            directory / "jacobian" / f"gradient.raw.{field}", config.cell_count
        )
        for field in PHYSICAL_FIELDS
    }
    traces = {
        component: read_su_float_samples(
            directory / "su" / f"synthetic_{component}.su.shot1.it1",
            config.receiver_count,
            config.samples_per_trace,
        )
        for component in ("x", "y")
    }
    if ranks == 1:
        storage = json.loads(
            (directory / "jacobian" / "gradient.segmented_gradient.json").read_text(
                encoding="utf-8"
            )
        )
        distributed = None
    else:
        storage = [
            json.loads(
                (directory / "jacobian" / f"gradient.segmented_gradient.rank{rank}.json").read_text(
                    encoding="utf-8"
                )
            )
            for rank in range(ranks)
        ]
        distributed = json.loads(
            (directory / "jacobian" / "gradient.distributed_gradient.json").read_text(
                encoding="utf-8"
            )
        )
    return {
        "directory": directory,
        "gradients": gradients,
        "traces": traces,
        "storage": storage,
        "distributed": distributed,
    }


def _mismatch_diagnostic(reference: array, candidate: array, config, nprocx, nprocy):
    mismatches = []
    max_absolute = 0.0
    max_relative = 0.0
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if expected == actual:
            continue
        ix, iy = divmod(index, config.ny)
        mismatches.append((ix, iy))
        difference = abs(float(actual) - float(expected))
        max_absolute = max(max_absolute, difference)
        max_relative = max(
            max_relative,
            difference / max(abs(float(expected)), abs(float(actual)), 1.0e-30),
        )
    x_interfaces = {part * (config.nx // nprocx) for part in range(1, nprocx)}
    y_interfaces = {part * (config.ny // nprocy) for part in range(1, nprocy)}
    confined = all(
        ix in x_interfaces
        or ix + 1 in x_interfaces
        or iy in y_interfaces
        or iy + 1 in y_interfaces
        for ix, iy in mismatches
    )
    return {
        "mismatch_count": len(mismatches),
        "max_absolute_difference": max_absolute,
        "max_relative_difference": max_relative,
        "confined_to_partition_interfaces": confined,
    }


def _require_byte_identical(reference, candidate, config, nprocx, nprocy):
    failures = {}
    for field in PHYSICAL_FIELDS:
        if reference[field].tobytes() != candidate[field].tobytes():
            failures[field] = _mismatch_diagnostic(
                reference[field], candidate[field], config, nprocx, nprocy
            )
    assert not failures, json.dumps(failures, indent=2, sort_keys=True)


@pytest.fixture(scope="module")
def distributed_gradient_runs(
    tmp_path_factory, repository_root, denise_binary, mpiexec
):
    root = tmp_path_factory.mktemp("m8d1a_distributed_exact_gradient")
    # Source and first receiver lie exactly on both 2x2 decomposition lines.
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
    model = _reference_model(config)
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
        configuration={"gate": "M8d-1A", "role": "observed_truth"},
        timeout_seconds=120.0,
    )
    assert truth.returncode == 0, result_summary(truth)

    layouts = {
        "one": (1, 1, 32),
        "horizontal": (2, 1, 32),
        "vertical": (1, 2, 32),
        "four": (2, 2, 32),
        "horizontal_segments_5": (2, 1, 5),
    }
    runs = {
        name: _run_gradient(
            root / name,
            config=config,
            model=model,
            observed_directory=observed_directory,
            nprocx=nprocx,
            nprocy=nprocy,
            segments=segments,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
        )
        for name, (nprocx, nprocy, segments) in layouts.items()
    }
    return {"root": root, "config": config, "model": model, "runs": runs}


def test_distributed_gradients_traces_objective_and_memory_are_exact(
    distributed_gradient_runs,
):
    config = distributed_gradient_runs["config"]
    runs = distributed_gradient_runs["runs"]
    reference = runs["one"]
    for name, nprocx, nprocy in (
        ("horizontal", 2, 1),
        ("vertical", 1, 2),
        ("four", 2, 2),
    ):
        candidate = runs[name]
        for component in ("x", "y"):
            assert candidate["traces"][component] == reference["traces"][component]
        report = candidate["distributed"]
        assert report["decomposition"] == [nprocx, nprocy]
        assert math.isclose(
            report["global_objective"],
            reference["storage"]["global_objective"],
            rel_tol=1.0e-14,
            abs_tol=0.0,
        )
        assert report["source_owners"] == 1
        assert report["receiver_ownership_valid"] is True
        assert report["checkpoint_count_min"] == report["checkpoint_count_max"] == 31
        assert report["replayed_forward_steps_min"] == report["replayed_forward_steps_max"] == config.samples_per_trace
        assert report["global_trajectory_replication"] is False
        _require_byte_identical(
            reference["gradients"], candidate["gradients"], config, nprocx, nprocy
        )
        for rank_report in candidate["storage"]:
            assert rank_report["local_grid"] == [config.nx // nprocx, config.ny // nprocy]
            assert rank_report["checkpoint_count"] == 31
            assert rank_report["replayed_forward_steps"] == config.samples_per_trace
            assert rank_report["combined_working_set_bytes"] < reference["storage"]["combined_working_set_bytes"]


def test_interface_sensitivity_and_segment_schedule_are_exact(
    distributed_gradient_runs,
):
    config = distributed_gradient_runs["config"]
    runs = distributed_gradient_runs["runs"]
    reference = runs["one"]
    _require_byte_identical(
        reference["gradients"],
        runs["horizontal_segments_5"]["gradients"],
        config,
        2,
        1,
    )
    for field in PHYSICAL_FIELDS:
        values = reference["gradients"][field]
        crossing_cells = [
            abs(values[ix * config.ny + iy])
            for ix in (config.nx // 2 - 1, config.nx // 2)
            for iy in range(config.ny)
        ] + [
            abs(values[ix * config.ny + iy])
            for ix in range(config.nx)
            for iy in (config.ny // 2 - 1, config.ny // 2)
        ]
        assert any(math.isfinite(value) and value > 0.0 for value in crossing_cells), field


def test_distributed_active_fwi_remains_fail_closed(
    tmp_path,
    repository_root,
    denise_binary,
    mpiexec,
):
    config = ViscoPSVOracleConfig()
    model = _reference_model(config)
    observed = tmp_path / "observed"
    _write_case(observed, config=config, model=_target_model(model, config), mode=0)
    truth = run_denise(
        repository_root=repository_root,
        case_directory=observed,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration={"gate": "M8d-1A", "role": "fail_closed_truth"},
        timeout_seconds=120.0,
    )
    assert truth.returncode == 0, result_summary(truth)
    case = tmp_path / "active_distributed"
    _write_case(
        case,
        config=config,
        model=model,
        mode=1,
        observed=observed,
        nprocx=2,
        nprocy=1,
    )
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("DENISE_PSV_EXACT_VISCO_GRADIENT", "1")
        environment.delenv("DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY", raising=False)
        result = run_denise(
            repository_root=repository_root,
            case_directory=case,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            ranks=2,
            configuration={"gate": "M8d-1A", "role": "active_fail_closed"},
            timeout_seconds=60.0,
        )
    assert result.returncode != 0
    output = result.stdout_path.read_text(errors="replace") + result.stderr_path.read_text(errors="replace")
    assert "Distributed exact-visco active FWI requires M8d-1B" in output
