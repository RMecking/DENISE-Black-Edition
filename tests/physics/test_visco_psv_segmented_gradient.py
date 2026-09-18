"""M8c-2 segmented exact-viscoelastic P/SV gradient acceptance gate."""

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
from tests.utilities.fwi_gradient import read_su_float_samples
from tests.utilities.runner import result_summary, run_denise


pytestmark = pytest.mark.integration


def _point_model_at(parameter_file: Path, prefix: str) -> None:
    text = parameter_file.read_text(encoding="ascii")
    old = "MFILE =model/current"
    assert text.count(old) == 1
    parameter_file.write_text(text.replace(old, f"MFILE ={prefix}"), encoding="ascii")


def _read_grid(path: Path, count: int) -> array:
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, count)
        assert not stream.read(1)
    assert len(values) == count
    return values


def _active_run(
    directory: Path,
    *,
    config: ViscoPSVOracleConfig,
    model,
    observed_directory: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    monkeypatch,
    segments: int | None,
    full_storage: bool,
):
    _write_case(
        directory, config=config, model=model, mode=1, observed=observed_directory
    )
    monkeypatch.setenv("DENISE_PSV_EXACT_VISCO_GRADIENT", "1")
    if full_storage:
        monkeypatch.setenv("DENISE_PSV_EXACT_FULL_STORAGE_REFERENCE", "1")
    else:
        monkeypatch.delenv("DENISE_PSV_EXACT_FULL_STORAGE_REFERENCE", raising=False)
    if segments is None:
        monkeypatch.delenv("DENISE_PSV_EXACT_SEGMENTS", raising=False)
    else:
        monkeypatch.setenv("DENISE_PSV_EXACT_SEGMENTS", str(segments))
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration={
            "gate": "M8c-2",
            "storage": "full" if full_storage else "segmented",
            "segments": segments if segments is not None else "default",
        },
        timeout_seconds=180.0,
    )
    assert result.returncode == 0, result_summary(result)
    active = json.loads(
        (directory / "jacobian" / "gradient.active_fwi.json").read_text(
            encoding="utf-8"
        )
    )
    storage = json.loads(
        (directory / "jacobian" / "gradient.segmented_gradient.json").read_text(
            encoding="utf-8"
        )
    )
    return result, active, storage


def _reload(
    root: Path,
    *,
    source_directory: Path,
    active_report,
    config: ViscoPSVOracleConfig,
    observed,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
):
    prefix = source_directory / active_report["persisted_prefix"]
    accepted = {
        name: _read_grid(Path(f"{prefix}.{name}"), config.cell_count)
        for name in PHYSICAL_FIELDS
    }
    _write_case(root, config=config, model=accepted, mode=0)
    reload_prefix = root / "model" / "accepted"
    for name in PHYSICAL_FIELDS:
        shutil.copyfile(Path(f"{prefix}.{name}"), Path(f"{reload_prefix}.{name}"))
    _point_model_at(root / "denise.inp", "model/accepted")
    synthetic = _run_forward(
        root,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
    )
    suffix = {"vp": "pi", "vs": "mu", "rho": "rho", "qp": "qp", "qs": "qs"}
    for name in PHYSICAL_FIELDS:
        assert Path(f"{reload_prefix}.fdveps.{suffix[name]}").read_bytes() == Path(
            f"{prefix}.{name}"
        ).read_bytes()
    return synthetic, _objective(synthetic, observed)


def _assert_same_production_result(
    reference: Path,
    candidate: Path,
    reference_report,
    candidate_report,
    config: ViscoPSVOracleConfig,
) -> None:
    for name in PHYSICAL_FIELDS:
        assert (reference / "jacobian" / f"gradient.raw.{name}").read_bytes() == (
            candidate / "jacobian" / f"gradient.raw.{name}"
        ).read_bytes()
    for component in ("x", "y"):
        filename = f"synthetic_{component}.su.shot1.it1"
        assert read_su_float_samples(
            reference / "su" / filename,
            config.receiver_count,
            config.samples_per_trace,
        ) == read_su_float_samples(
            candidate / "su" / filename,
            config.receiver_count,
            config.samples_per_trace,
        )
    assert candidate_report == reference_report
    reference_prefix = reference / reference_report["persisted_prefix"]
    candidate_prefix = candidate / candidate_report["persisted_prefix"]
    for name in PHYSICAL_FIELDS:
        assert Path(f"{reference_prefix}.{name}").read_bytes() == Path(
            f"{candidate_prefix}.{name}"
        ).read_bytes()


def test_segmented_exact_gradient_is_byte_identical_and_bounded(
    tmp_path, repository_root, denise_binary, mpiexec, monkeypatch
):
    config = ViscoPSVOracleConfig()
    model = _reference_model(config)
    truth = tmp_path / "truth"
    _write_case(truth, config=config, model=_target_model(model, config), mode=0)
    observed = _run_forward(
        truth,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
    )

    full = tmp_path / "full_storage"
    full_result, full_active, full_storage = _active_run(
        full,
        config=config,
        model=model,
        observed_directory=truth,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        monkeypatch=monkeypatch,
        segments=None,
        full_storage=True,
    )
    assert full_storage["storage_mode"] == "full_storage_reference"
    assert full_storage["full_storage_bytes_allocated"] == 6 * config.cell_count * (
        config.samples_per_trace + 1
    ) * 4

    timings = {"full_storage": full_result.runtime_seconds}
    segmented_runs = []
    for label, segments in (
        ("segments_5", 5),
        ("segments_10", 10),
        ("segments_20", 20),
        ("segments_default_32", None),
        ("segments_above_nt", config.samples_per_trace + 17),
    ):
        directory = tmp_path / label
        result, active, storage = _active_run(
            directory,
            config=config,
            model=model,
            observed_directory=truth,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            monkeypatch=monkeypatch,
            segments=segments,
            full_storage=False,
        )
        _assert_same_production_result(full, directory, full_active, active, config)
        expected_segments = min(segments or 32, config.samples_per_trace)
        assert storage["storage_mode"] == "segmented"
        assert storage["segment_count"] == expected_segments
        assert storage["checkpoint_count"] == expected_segments - 1
        assert storage["replayed_forward_steps"] == config.samples_per_trace
        assert storage["full_storage_bytes_allocated"] == 0
        assert storage["max_segment_length"] == math.ceil(
            config.samples_per_trace / expected_segments
        )
        boundaries = storage["segment_boundaries"]
        assert boundaries[0][0] == 1
        assert boundaries[-1][1] == config.samples_per_trace
        assert all(left[1] + 1 == right[0] for left, right in zip(boundaries, boundaries[1:]))
        timings[label] = result.runtime_seconds
        segmented_runs.append((label, directory, active, storage))

    default_label, default_directory, default_active, default_storage = segmented_runs[3]
    assert default_label == "segments_default_32"
    checkpoint_each = (8 * (config.nx + 6) * (config.ny + 6) + 8 * 8 * (config.nx + config.ny)) * 4
    expected_legacy = 6 * config.cell_count * (config.samples_per_trace + 1) * 4
    expected_segment = 6 * config.cell_count * 19 * 4
    assert default_storage["requested_segments"] == 32
    assert default_storage["checkpoint_bytes_each"] == checkpoint_each
    assert default_storage["checkpoint_bytes_total"] == 31 * checkpoint_each
    assert default_storage["segment_buffer_bytes"] == expected_segment
    assert default_storage["combined_working_set_bytes"] == 31 * checkpoint_each + expected_segment
    assert default_storage["legacy_six_field_bytes"] == expected_legacy
    assert math.isclose(
        default_storage["reduction_factor"],
        expected_legacy / (31 * checkpoint_each + expected_segment),
        rel_tol=0.0,
        abs_tol=1.0e-14,
    )
    representative = default_storage["representative_grids"]
    assert 1.0 < representative["500x500x5000"]["combined_gib"] < 1.3
    assert 27.0 < representative["500x500x5000"]["legacy_gib"] < 29.0
    assert 4.0 < representative["1000x1000x5000"]["combined_gib"] < 5.0
    assert 110.0 < representative["1000x1000x5000"]["legacy_gib"] < 113.0

    above_storage = segmented_runs[-1][3]
    assert above_storage["segment_count_clamped"] is True
    assert above_storage["segment_count"] == config.samples_per_trace
    assert above_storage["max_segment_length"] == 1

    full_reload, full_reload_objective = _reload(
        tmp_path / "reload_full",
        source_directory=full,
        active_report=full_active,
        config=config,
        observed=observed,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
    )
    segmented_reload, segmented_reload_objective = _reload(
        tmp_path / "reload_segmented",
        source_directory=default_directory,
        active_report=default_active,
        config=config,
        observed=observed,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
    )
    assert segmented_reload == full_reload
    assert segmented_reload_objective == full_reload_objective
    assert math.isclose(
        segmented_reload_objective,
        default_active["accepted_objective"],
        rel_tol=2.0e-6,
        abs_tol=1.0e-10,
    )

    (tmp_path / "m8c2_segmented_gradient_report.json").write_text(
        json.dumps(
            {
                "byte_identical_fields": list(PHYSICAL_FIELDS),
                "segment_counts": [5, 10, 20, 32, config.samples_per_trace],
                "timings_seconds": timings,
                "runtime_ratio_segmented32_to_full": timings["segments_default_32"]
                / timings["full_storage"],
                "memory": default_storage,
                "reload_objective": segmented_reload_objective,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
