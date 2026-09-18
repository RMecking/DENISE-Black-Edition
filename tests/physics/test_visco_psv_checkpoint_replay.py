"""M8c-1 byte-identical replay gate for exact one-rank viscoelastic P/SV."""

from __future__ import annotations

import json

import pytest

from tests.physics.test_visco_psv_physical_gradient_oracle import (
    ViscoPSVOracleConfig,
    _reference_model,
    _run_forward,
    _target_model,
    _write_case,
)
from tests.utilities.runner import result_summary, run_denise


pytestmark = pytest.mark.integration


def _snapshot_path(directory, label: str, family: str):
    return directory / "jacobian" / f"gradient.checkpoint_replay.{label}.{family}.bin"


def _assert_snapshot_bytes(directory, *, reference: str, candidate: str, family: str, expected_bytes: int):
    reference_path = _snapshot_path(directory, reference, family)
    candidate_path = _snapshot_path(directory, candidate, family)
    assert reference_path.stat().st_size == expected_bytes
    assert candidate_path.stat().st_size == expected_bytes
    assert reference_path.read_bytes() == candidate_path.read_bytes()


def test_exact_visco_psv_checkpoint_replay_is_byte_identical(
    tmp_path, repository_root, denise_binary, mpiexec, monkeypatch
):
    """Capture after complete timestep c, restore, then execute c+1..NT.

    The production boundary follows source injection, velocity/stress updates,
    GSLS and CPML recurrences, exchanges, receiver sampling, and the existing
    full trajectory recorder.  The replay therefore starts at exactly c+1.
    """
    config = ViscoPSVOracleConfig()
    model = _reference_model(config)
    observed = tmp_path / "observed_truth"
    restarted = tmp_path / "checkpoint_replay"

    _write_case(
        observed,
        config=config,
        model=_target_model(model, config),
        mode=0,
    )
    _run_forward(
        observed,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
    )
    assert not list((observed / "jacobian").glob("*checkpoint_replay*"))
    _write_case(
        restarted,
        config=config,
        model=model,
        mode=1,
        observed=observed,
    )

    monkeypatch.setenv("DENISE_PSV_EXACT_VISCO_GRADIENT", "1")
    monkeypatch.setenv("DENISE_PSV_CHECKPOINT_REPLAY_TEST", "1")
    result = run_denise(
        repository_root=repository_root,
        case_directory=restarted,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration={"gate": "M8c-1", "mode": 1},
        timeout_seconds=120.0,
    )
    assert result.returncode == 0, result_summary(result)

    report_path = restarted / "jacobian" / "gradient.checkpoint_replay.json"
    report = json.loads(report_path.read_text(encoding="ascii"))
    checkpoint = config.samples_per_trace // 2
    replay_steps = config.samples_per_trace - checkpoint
    expected_operand_values = replay_steps * config.nx * config.ny
    expected_receiver_values = 2 * replay_steps * config.receiver_count
    expected_payload_bytes = (
        8 * (config.nx + 6) * (config.ny + 6)
        + 8 * 8 * (config.nx + config.ny)
    ) * 4
    full_count = (config.nx + 6) * (config.ny + 6)
    primary_bytes = 5 * full_count * 4
    gsls_bytes = 3 * full_count * 4
    cpml_x_count = config.ny * 2 * 8
    cpml_y_count = config.nx * 2 * 8
    cpml_bytes = 4 * (cpml_x_count + cpml_y_count) * 4

    assert report["checkpoint_timestep"] == checkpoint
    assert report["resume_first_timestep"] == checkpoint + 1
    assert report["resume_last_timestep"] == config.samples_per_trace
    assert report["replay_steps"] == replay_steps
    assert report["payload_bytes"] == expected_payload_bytes
    assert report["live_snapshot_full_extent"] == [config.nx + 6, config.ny + 6]
    assert report["live_snapshot_full_elements_per_field"] == full_count
    assert report["live_snapshot_primary_elements"] == 5 * full_count
    assert report["live_snapshot_gsls_elements"] == 3 * full_count
    assert report["live_snapshot_cpml_x_extent"] == [config.ny, 16]
    assert report["live_snapshot_cpml_y_extent"] == [16, config.nx]
    assert report["live_snapshot_cpml_elements"] == 4 * (cpml_x_count + cpml_y_count)

    # Independent raw-byte observations: the C hook traverses live pointers
    # directly and neither packs nor compares checkpoint payloads.
    _assert_snapshot_bytes(
        restarted, reference="t300_reference", candidate="t300_restored",
        family="primary", expected_bytes=primary_bytes,
    )
    _assert_snapshot_bytes(
        restarted, reference="t300_reference", candidate="t300_restored",
        family="gsls", expected_bytes=gsls_bytes,
    )
    _assert_snapshot_bytes(
        restarted, reference="t300_reference", candidate="t300_restored",
        family="cpml", expected_bytes=cpml_bytes,
    )
    _assert_snapshot_bytes(
        restarted, reference="t600_reference", candidate="t600_replayed",
        family="primary", expected_bytes=primary_bytes,
    )
    _assert_snapshot_bytes(
        restarted, reference="t600_reference", candidate="t600_replayed",
        family="gsls", expected_bytes=gsls_bytes,
    )
    _assert_snapshot_bytes(
        restarted, reference="t600_reference", candidate="t600_replayed",
        family="cpml", expected_bytes=cpml_bytes,
    )

    source = (repository_root / "src" / "PSV" / "psv.c").read_text(encoding="utf-8")
    hook = source.split("static void write_live_state_snapshot", 1)[1].split("void psv(", 1)[0]
    assert "visco_psv_checkpoint" not in hook

    # Recorder order is VXX, VYX, VXY, VYY, FX, FY.  Every cell of every
    # replayed timestep is compared with memcmp semantics in the C hook.
    assert report["operand_compared_per_field"] == expected_operand_values
    assert report["operand_compared"] == [expected_operand_values] * 6
    assert report["operand_mismatches"] == [0] * 6
    assert report["receiver_values_compared"] == expected_receiver_values
    assert report["receiver_mismatches"] == 0
    assert report["source_timing_equal"] is True
    assert report["receiver_timing_equal"] is True
    assert report["bit_identical_replay"] is True
