from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.cases.homogeneous_viscoelastic import (
    ViscoelasticPSVConfig,
    ViscoelasticSHConfig,
    generate_viscoelastic_psv_case,
    generate_viscoelastic_sh_case,
)
from tests.utilities.attenuation import peak_absolute, root_mean_square, spectral_band_rms
from tests.utilities.physics_run import run_psv_case
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import (
    all_finite,
    normalized_correlation,
    read_ascii_seismograms,
    relative_l2,
)


pytestmark = pytest.mark.integration


def _run_sh(directory: Path, *, repository_root: Path, denise_binary: Path, mpiexec: str, config):
    generate_viscoelastic_sh_case(directory, config=config)
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration=config.as_metadata() | {"nprocx": 1, "nprocy": 1},
    )
    assert result.returncode == 0, result_summary(result)
    output = directory / "su" / "homogeneous_vz.asc.shot1"
    assert output.is_file() and output.stat().st_size > 0
    traces = read_ascii_seismograms(output, config.receiver_count, config.samples_per_trace)
    assert all_finite(traces)
    return traces


def _amplitude_metrics(trace: list[float], config: ViscoelasticSHConfig) -> dict[str, float]:
    return {
        "peak_absolute": peak_absolute(trace),
        "rms": root_mean_square(trace),
        "spectral_band_rms_5_15_hz": spectral_band_rms(
            trace, dt_s=config.dt_s, lower_hz=5.0, upper_hz=15.0
        ),
    }


def _comparison_metrics(high: list[list[float]], low: list[list[float]], config) -> dict[str, object]:
    high_amplitude = _amplitude_metrics(high[-1], config)
    low_amplitude = _amplitude_metrics(low[-1], config)
    return {
        "relative_l2": relative_l2(high, low),
        "normalized_correlation": normalized_correlation(high, low),
        "far_receiver_high_q_amplitudes": high_amplitude,
        "far_receiver_low_q_amplitudes": low_amplitude,
        "far_receiver_relative_amplitude_differences": {
            name: abs(high_amplitude[name] - low_amplitude[name]) / high_amplitude[name]
            for name in high_amplitude
        },
    }


def _case_model_hashes(directory: Path) -> dict[str, str]:
    return json.loads((directory / "case.json").read_text(encoding="utf-8"))["model_sha256"]


def _run_psv(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    config: ViscoelasticPSVConfig,
) -> tuple[list[list[float]], list[list[float]]]:
    return run_psv_case(
        directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        generator=generate_viscoelastic_psv_case,
    )


def test_sh_qs_200_repeatability(tmp_path, repository_root, denise_binary, mpiexec):
    config = ViscoelasticSHConfig(qs=200.0)
    first_directory = tmp_path / "sh_qs_200_first"
    repeat_directory = tmp_path / "sh_qs_200_repeat"
    first = _run_sh(
        first_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
    )
    repeat = _run_sh(
        repeat_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
    )
    repeat_rel_l2 = relative_l2(first, repeat)
    repeat_correlation = normalized_correlation(first, repeat)
    metrics = {
        "qs": config.qs,
        "first_model_sha256": _case_model_hashes(first_directory),
        "repeat_model_sha256": _case_model_hashes(repeat_directory),
        "relative_l2": repeat_rel_l2,
        "normalized_correlation": repeat_correlation,
    }
    (tmp_path / "sh_qs_200_repeatability_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert metrics["first_model_sha256"] == metrics["repeat_model_sha256"]
    assert repeat_rel_l2 <= 1.0e-12
    assert repeat_correlation >= 1.0 - 1.0e-12


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect: PHYSICS=SH, MODE=0, L>0 viscoelastic forward modelling "
        "ignores Qs because the forward path uses the elastic stress update"
    ),
)
def test_sh_mode0_qs_sensitivity(tmp_path, repository_root, denise_binary, mpiexec):
    base = ViscoelasticSHConfig()
    traces = {
        qs: _run_sh(
            tmp_path / f"sh_qs_{int(qs)}",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=replace(base, qs=qs),
        )
        for qs in (20.0, 50.0, 200.0)
    }
    amplitudes = {str(int(qs)): _amplitude_metrics(data[-1], base) for qs, data in traces.items()}
    model_hashes = {
        str(int(qs)): _case_model_hashes(tmp_path / f"sh_qs_{int(qs)}")
        for qs in traces
    }
    sensitivity_rel_l2 = relative_l2(traces[200.0], traces[20.0])
    metrics = {
        "receiver_offset_m": base.receiver_offsets_m()[-1],
        "amplitudes_by_qs": amplitudes,
        "model_sha256_by_qs": model_hashes,
        "q200_vs_q20_relative_l2": sensitivity_rel_l2,
    }
    (tmp_path / "sh_q_sensitivity_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    qs_hashes = {model_hashes[label]["qs"] for label in ("20", "50", "200")}
    assert len(qs_hashes) == 3, "Qs input models must have distinct SHA-256 hashes"
    assert sensitivity_rel_l2 >= 1.0e-3


@pytest.mark.xfail(
    strict=True,
    reason="Known defect: readmod_visc_PSV.c overwrites input Qp with 30.0",
)
def test_psv_qp_input_sensitivity(tmp_path, repository_root, denise_binary, mpiexec):
    base = ViscoelasticPSVConfig()
    p_low = replace(base, qp=20.0, qs=100.0, source_type=2)
    p_high = replace(base, qp=200.0, qs=100.0, source_type=2)
    low_directory = tmp_path / "psv_qp_20"
    high_directory = tmp_path / "psv_qp_200"
    p_low_vx, _ = _run_psv(
        low_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=p_low,
    )
    p_high_vx, _ = _run_psv(
        high_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=p_high,
    )
    metrics = _comparison_metrics(p_high_vx, p_low_vx, p_high) | {
        "input_values": [20.0, 200.0],
        "fixed_qs": 100.0,
        "model_sha256": {
            "low": _case_model_hashes(low_directory),
            "high": _case_model_hashes(high_directory),
        },
    }
    (tmp_path / "psv_qp_sensitivity_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert metrics["model_sha256"]["low"]["qp"] != metrics["model_sha256"]["high"]["qp"]
    assert metrics["relative_l2"] >= 1.0e-3


@pytest.mark.xfail(
    strict=True,
    reason="Known defect: readmod_visc_PSV.c overwrites input Qs with 30.0",
)
def test_psv_qs_input_sensitivity(tmp_path, repository_root, denise_binary, mpiexec):
    base = ViscoelasticPSVConfig()
    vertical_receivers = tuple(
        (base.source_x_m, base.source_y_m + offset) for offset in range(200, 700, 100)
    )
    s_low = replace(
        base, qp=100.0, qs=20.0, source_type=2, receivers_m=vertical_receivers
    )
    s_high = replace(
        base, qp=100.0, qs=200.0, source_type=2, receivers_m=vertical_receivers
    )
    low_directory = tmp_path / "psv_qs_20"
    high_directory = tmp_path / "psv_qs_200"
    s_low_vx, _ = _run_psv(
        low_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=s_low,
    )
    s_high_vx, _ = _run_psv(
        high_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=s_high,
    )
    metrics = _comparison_metrics(s_high_vx, s_low_vx, s_high) | {
        "input_values": [20.0, 200.0],
        "fixed_qp": 100.0,
        "model_sha256": {
            "low": _case_model_hashes(low_directory),
            "high": _case_model_hashes(high_directory),
        },
    }
    (tmp_path / "psv_qs_sensitivity_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert metrics["model_sha256"]["low"]["qs"] != metrics["model_sha256"]["high"]["qs"]
    assert metrics["relative_l2"] >= 1.0e-3
