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


def test_sh_qs_sensitivity_and_repeatability(tmp_path, repository_root, denise_binary, mpiexec):
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
    repeat = _run_sh(
        tmp_path / "sh_qs_200_repeat",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=replace(base, qs=200.0),
    )
    amplitudes = {str(int(qs)): _amplitude_metrics(data[-1], base) for qs, data in traces.items()}
    model_hashes = {
        str(int(qs)): json.loads((tmp_path / f"sh_qs_{int(qs)}" / "case.json").read_text())["model_sha256"]
        for qs in traces
    }
    repeat_rel_l2 = relative_l2(traces[200.0], repeat)
    repeat_correlation = normalized_correlation(traces[200.0], repeat)
    sensitivity_rel_l2 = relative_l2(traces[200.0], traces[20.0])
    metrics = {
        "receiver_offset_m": base.receiver_offsets_m()[-1],
        "amplitudes_by_qs": amplitudes,
        "model_sha256_by_qs": model_hashes,
        "q200_vs_q20_relative_l2": sensitivity_rel_l2,
        "q200_repeat_relative_l2": repeat_rel_l2,
        "q200_repeat_normalized_correlation": repeat_correlation,
    }
    (tmp_path / "sh_q_sensitivity_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert repeat_rel_l2 <= 1.0e-12
    assert repeat_correlation >= 1.0 - 1.0e-12
    failures = [
        f"{metric} is not ordered Qs20 < Qs50 < Qs200"
        for metric in ("peak_absolute", "rms", "spectral_band_rms_5_15_hz")
        if not amplitudes["20"][metric] < amplitudes["50"][metric] < amplitudes["200"][metric]
    ]
    if sensitivity_rel_l2 < 1.0e-3:
        failures.append(f"Qs200/Qs20 relative L2 {sensitivity_rel_l2:.6g} is below 1e-3")
    assert not failures, "; ".join(failures)


def test_psv_qp_and_qs_inputs_affect_waveforms(tmp_path, repository_root, denise_binary, mpiexec):
    base = ViscoelasticPSVConfig(qp=20.0, qs=20.0)
    p_low = replace(base, qp=20.0, qs=100.0, source_type=2)
    p_high = replace(base, qp=200.0, qs=100.0, source_type=2)
    vertical_receivers = tuple((base.source_x_m, base.source_y_m + offset) for offset in range(200, 700, 100))
    s_low = replace(base, qp=100.0, qs=20.0, source_type=2, receivers_m=vertical_receivers)
    s_high = replace(base, qp=100.0, qs=200.0, source_type=2, receivers_m=vertical_receivers)

    def run(label: str, config: ViscoelasticPSVConfig):
        return run_psv_case(
            tmp_path / label,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=config,
            generator=generate_viscoelastic_psv_case,
        )

    p_low_vx, _ = run("psv_qp_20", p_low)
    p_high_vx, _ = run("psv_qp_200", p_high)
    s_low_vx, _ = run("psv_qs_20", s_low)
    s_high_vx, _ = run("psv_qs_200", s_high)
    metrics = {
        "qp": _comparison_metrics(p_high_vx, p_low_vx, p_high) | {
            "input_values": [20.0, 200.0],
            "fixed_qs": 100.0,
            "model_sha256": {
                "low": json.loads((tmp_path / "psv_qp_20" / "case.json").read_text())["model_sha256"],
                "high": json.loads((tmp_path / "psv_qp_200" / "case.json").read_text())["model_sha256"],
            },
        },
        "qs": _comparison_metrics(s_high_vx, s_low_vx, s_high) | {
            "input_values": [20.0, 200.0],
            "fixed_qp": 100.0,
            "model_sha256": {
                "low": json.loads((tmp_path / "psv_qs_20" / "case.json").read_text())["model_sha256"],
                "high": json.loads((tmp_path / "psv_qs_200" / "case.json").read_text())["model_sha256"],
            },
        },
    }
    (tmp_path / "psv_q_sensitivity_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failures = [
        f"P/SV waveform is insensitive to {name} input (relative L2={metrics[name]['relative_l2']:.6g})"
        for name in ("qp", "qs")
        if metrics[name]["relative_l2"] < 1.0e-3
    ]
    assert not failures, "; ".join(failures)
