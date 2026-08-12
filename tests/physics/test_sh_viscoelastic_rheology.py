from __future__ import annotations

import cmath
import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case as generate_elastic_sh
from tests.cases.homogeneous_viscoelastic import (
    ViscoelasticSHConfig,
    generate_viscoelastic_sh_case,
)
from tests.utilities.attenuation import peak_absolute, root_mean_square
from tests.utilities.effective_parameters import (
    EffectiveDeniseParameters,
    read_effective_parameters,
    require_effective_parameters,
)
from tests.utilities.runner import RunResult, result_summary, run_denise
from tests.utilities.qstd_reference import qstd_quality_factor, target_q_to_tau
from tests.utilities.seismogram import (
    absolute_peak_index_in_interval,
    all_finite,
    normalized_correlation,
    read_ascii_seismograms,
    relative_l2,
    time_interval,
)
from tests.utilities.viscoelastic_rheology import (
    approximate_main_lobe_width_hz,
    discrete_rheology_prediction,
    effective_q_from_transfer_slopes,
    linear_fit,
    rheology_prediction,
    transfer_spectrum,
    unwrap_phase,
)


pytestmark = pytest.mark.integration


# Predeclared before examining M4.2 results. At 10 Hz, DH=10 m gives 20
# points per S wavelength for the eighth-order stencil and DT=0.5 ms gives 200
# samples per period. The tolerances allow residual source/receiver and FD
# effects while remaining much tighter than the Q=50 signal differences.
HIGH_Q_VALUES = (50.0, 200.0, 1000.0)
HIGH_Q_DIRECT_RELATIVE_L2_MAX = 0.025
HIGH_Q_CORRELATION_MIN = 0.999
SPECTRAL_FREQUENCIES_HZ = (6.0, 8.0, 10.0, 12.0, 14.0)
ATTENUATION_FREQUENCIES_HZ = (8.0, 10.0, 12.0)
PHASE_FREQUENCIES_HZ = (6.0, 14.0)
PHASE_DIAGNOSTIC_FREQUENCIES_HZ = (8.0, 12.0)
DIRECT_WINDOW_HALF_WIDTH_S = 0.11
SPECTRAL_WINDOW_HALF_WIDTH_S = 0.20
SPECTRAL_WINDOW_KIND = "tukey"
SPECTRAL_TUKEY_ALPHA = 0.2
ATTENUATION_SLOPE_RELATIVE_TOLERANCE = 0.15
PHASE_SLOPE_RELATIVE_TOLERANCE = 0.20
SLOPE_R_SQUARED_MIN = 0.95
LOG_AMPLITUDE_RESIDUAL_MAX = 0.02
PHASE_RESIDUAL_MAX_RAD = 0.02
EFFECTIVE_Q_RELATIVE_TOLERANCE = 0.20
HISTORICAL_L4_RELAXATION_FREQUENCIES_HZ = (2.7105, 12.2792, 68.1930, 265.2297)
HISTORICAL_L4_OPTIMIZED_TAU = 0.0386
HISTORICAL_L4_TARGET_Q = 30.0
HISTORICAL_L4_COMPENSATING_QS_FILE = 2.0 / HISTORICAL_L4_OPTIMIZED_TAU
HISTORICAL_L4_Q_RELATIVE_TOLERANCE = 0.10


BASE_CONFIG = HomogeneousSHConfig(
    nx=260,
    ny=140,
    time_s=0.75,
    source_x_m=600.0,
    source_y_m=700.0,
    receiver_x_m=(1000.0, 1100.0, 1200.0, 1300.0, 1400.0),
    receiver_y_m=700.0,
)


@dataclass(frozen=True)
class SHRun:
    directory: Path
    config: HomogeneousSHConfig
    traces: list[list[float]]
    effective: EffectiveDeniseParameters
    model_sha256: dict[str, str]
    result: RunResult


@dataclass(frozen=True)
class M42Runs:
    root: Path
    elastic: SHRun
    viscoelastic: dict[float, SHRun]
    q200_repeat: SHRun


@dataclass(frozen=True)
class HistoricalL4Runs:
    root: Path
    nominal_qs30: SHRun
    compensating_qs: SHRun
    physical_qs30: SHRun


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float32_values(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(struct.unpack("f", struct.pack("f", value))[0] for value in values)


def _viscoelastic_config(qs: float, frequencies_hz=(10.0,)) -> ViscoelasticSHConfig:
    return ViscoelasticSHConfig(
        **asdict(BASE_CONFIG),
        qs=qs,
        relaxation_frequencies_hz=tuple(frequencies_hz),
    )


def _physical_q_config(qs: float, frequencies_hz) -> ViscoelasticSHConfig:
    return ViscoelasticSHConfig(
        **asdict(BASE_CONFIG),
        qs=qs,
        relaxation_frequencies_hz=tuple(frequencies_hz),
        q_parameterization_mode=1,
        q_approx_fmin_hz=5.0,
        q_approx_fmax_hz=120.0,
        q_approx_df_hz=5.0,
    )


def _run_sh_case(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    config: HomogeneousSHConfig,
    relaxation_frequencies_hz: tuple[float, ...],
) -> SHRun:
    if relaxation_frequencies_hz:
        assert isinstance(config, ViscoelasticSHConfig)
        generate_viscoelastic_sh_case(directory, config=config)
    else:
        generate_elastic_sh(directory, config=config)
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
    assert len(traces) == config.receiver_count
    assert all(len(trace) == config.samples_per_trace for trace in traces)
    assert all_finite(traces)

    effective = read_effective_parameters(result.stdout_path)
    require_effective_parameters(
        effective,
        mode=0,
        physics=5,
        relaxation_frequencies_hz=_float32_values(relaxation_frequencies_hz),
        q_parameterization_mode=int(getattr(config, "q_parameterization_mode", 0)),
        q_approx_fmin_hz=(getattr(config, "q_approx_fmin_hz", None) or None),
        q_approx_fmax_hz=(getattr(config, "q_approx_fmax_hz", None) or None),
        q_approx_df_hz=(getattr(config, "q_approx_df_hz", None) or None),
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["returncode"] == 0
    assert metadata["mpi_ranks"] == 1
    assert len(metadata["executable"]["sha256"]) == 64
    model = directory / "model" / "homogeneous"
    names = ("vs", "rho", "qs") if relaxation_frequencies_hz else ("vs", "rho")
    model_hashes = {name: _sha256(model.with_suffix(f".{name}")) for name in names}
    assert all(len(value) == 64 for value in model_hashes.values())
    return SHRun(directory, config, traces, effective, model_hashes, result)


@pytest.fixture(scope="module")
def m42_runs(tmp_path_factory, repository_root, denise_binary, mpiexec) -> M42Runs:
    root = tmp_path_factory.mktemp("m42_sh_rheology")
    elastic = _run_sh_case(
        root / "elastic",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=BASE_CONFIG,
        relaxation_frequencies_hz=(),
    )
    viscoelastic = {
        qs: _run_sh_case(
            root / f"qs_{int(qs)}",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=_viscoelastic_config(qs),
            relaxation_frequencies_hz=(10.0,),
        )
        for qs in HIGH_Q_VALUES
    }
    repeat = _run_sh_case(
        root / "qs_200_repeat",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=_viscoelastic_config(200.0),
        relaxation_frequencies_hz=(10.0,),
    )
    return M42Runs(root, elastic, viscoelastic, repeat)


@pytest.fixture(scope="module")
def historical_l4_runs(
    tmp_path_factory, repository_root, denise_binary, mpiexec
) -> HistoricalL4Runs:
    root = tmp_path_factory.mktemp("m42_historical_l4")
    nominal = _run_sh_case(
        root / "qs_30",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=_viscoelastic_config(
            HISTORICAL_L4_TARGET_Q, HISTORICAL_L4_RELAXATION_FREQUENCIES_HZ
        ),
        relaxation_frequencies_hz=HISTORICAL_L4_RELAXATION_FREQUENCIES_HZ,
    )
    compensating = _run_sh_case(
        root / "qs_compensating",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=_viscoelastic_config(
            HISTORICAL_L4_COMPENSATING_QS_FILE,
            HISTORICAL_L4_RELAXATION_FREQUENCIES_HZ,
        ),
        relaxation_frequencies_hz=HISTORICAL_L4_RELAXATION_FREQUENCIES_HZ,
    )
    physical = _run_sh_case(
        root / "physical_qs_30",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=_physical_q_config(
            HISTORICAL_L4_TARGET_Q, HISTORICAL_L4_RELAXATION_FREQUENCIES_HZ
        ),
        relaxation_frequencies_hz=HISTORICAL_L4_RELAXATION_FREQUENCIES_HZ,
    )
    return HistoricalL4Runs(root, nominal, compensating, physical)


def _direct_interval(config: HomogeneousSHConfig, offset_m: float) -> tuple[float, float]:
    center = 1.5 / config.source_frequency_hz + offset_m / config.vs_m_s
    return center - DIRECT_WINDOW_HALF_WIDTH_S, center + DIRECT_WINDOW_HALF_WIDTH_S


def _direct_windows(run: SHRun) -> list[list[float]]:
    return [
        time_interval(
            trace,
            start_s=_direct_interval(run.config, offset)[0],
            stop_s=_direct_interval(run.config, offset)[1],
            dt_s=run.config.dt_s,
        )
        for trace, offset in zip(run.traces, run.config.receiver_offsets_m())
    ]


def _frequency_transfer_by_receiver(
    viscoelastic: SHRun,
    elastic: SHRun,
    *,
    half_width_s: float = SPECTRAL_WINDOW_HALF_WIDTH_S,
    window_kind: str = SPECTRAL_WINDOW_KIND,
    tukey_alpha: float = SPECTRAL_TUKEY_ALPHA,
) -> dict[float, list[complex]]:
    result = {frequency: [] for frequency in SPECTRAL_FREQUENCIES_HZ}
    for offset, visco_trace, elastic_trace in zip(
        elastic.config.receiver_offsets_m(), viscoelastic.traces, elastic.traces
    ):
        center = 1.5 / elastic.config.source_frequency_hz + offset / elastic.config.vs_m_s
        start, stop = center - half_width_s, center + half_width_s
        visco_window = time_interval(
            visco_trace, start_s=start, stop_s=stop, dt_s=elastic.config.dt_s
        )
        elastic_window = time_interval(
            elastic_trace, start_s=start, stop_s=stop, dt_s=elastic.config.dt_s
        )
        samples = transfer_spectrum(
            visco_window,
            elastic_window,
            dt_s=elastic.config.dt_s,
            frequencies_hz=SPECTRAL_FREQUENCIES_HZ,
            window_kind=window_kind,
            tukey_alpha=tukey_alpha,
        )
        assert tuple(sample.frequency_hz for sample in samples) == SPECTRAL_FREQUENCIES_HZ
        for sample in samples:
            result[sample.frequency_hz].append(sample.value)
    return result


def _slope_metrics(
    viscoelastic: SHRun,
    elastic: SHRun,
    *,
    half_width_s: float = SPECTRAL_WINDOW_HALF_WIDTH_S,
    window_kind: str = SPECTRAL_WINDOW_KIND,
    tukey_alpha: float = SPECTRAL_TUKEY_ALPHA,
) -> dict[str, dict[str, object]]:
    offsets = elastic.config.receiver_offsets_m()
    aperture_m = max(offsets) - min(offsets)
    transfer = _frequency_transfer_by_receiver(
        viscoelastic,
        elastic,
        half_width_s=half_width_s,
        window_kind=window_kind,
        tukey_alpha=tukey_alpha,
    )
    metrics: dict[str, dict[str, object]] = {}
    qs = float(getattr(viscoelastic.config, "qs"))
    frequencies = tuple(getattr(viscoelastic.config, "relaxation_frequencies_hz"))
    tau_override = None
    if int(getattr(viscoelastic.config, "q_parameterization_mode", 0)) == 1:
        tau_override = target_q_to_tau(
            target_q=qs,
            relaxation_frequencies_hz=frequencies,
            fmin_hz=float(getattr(viscoelastic.config, "q_approx_fmin_hz")),
            fmax_hz=float(getattr(viscoelastic.config, "q_approx_fmax_hz")),
            df_hz=float(getattr(viscoelastic.config, "q_approx_df_hz")),
        )
    for frequency in SPECTRAL_FREQUENCIES_HZ:
        values = transfer[frequency]
        amplitude_fit = linear_fit(offsets, [math_log_abs(value) for value in values])
        unwrapped = unwrap_phase([complex_phase(value) for value in values])
        phase_fit = linear_fit(offsets, unwrapped)
        theory = rheology_prediction(
            frequency_hz=frequency,
            vs_m_s=elastic.config.vs_m_s,
            density_kg_m3=elastic.config.density_kg_m3,
            qs_input=qs,
            relaxation_frequencies_hz=frequencies,
            tau_override=tau_override,
        )
        discrete_theory = discrete_rheology_prediction(
            frequency_hz=frequency,
            vs_m_s=elastic.config.vs_m_s,
            density_kg_m3=elastic.config.density_kg_m3,
            qs_input=qs,
            relaxation_frequencies_hz=frequencies,
            dt_s=elastic.config.dt_s,
            dh_m=elastic.config.dh_m,
            tau_override=tau_override,
        )
        observed_q = effective_q_from_transfer_slopes(
            frequency_hz=frequency,
            vs_m_s=elastic.config.vs_m_s,
            density_kg_m3=elastic.config.density_kg_m3,
            log_amplitude_slope_per_m=amplitude_fit.slope,
            phase_slope_rad_per_m=phase_fit.slope,
        )
        metrics[str(frequency)] = {
            "frequency_hz": frequency,
            "log_amplitudes": [math_log_abs(value) for value in values],
            "unwrapped_phase_rad": list(unwrapped),
            "attenuation_fit": asdict(amplitude_fit),
            "phase_fit": asdict(phase_fit),
            "theoretical_log_amplitude_slope_per_m": theory.log_amplitude_slope_per_m,
            "theoretical_phase_slope_rad_per_m": theory.phase_slope_rad_per_m,
            "discrete_theoretical_log_amplitude_slope_per_m": (
                discrete_theory.log_amplitude_slope_per_m
            ),
            "discrete_theoretical_phase_slope_rad_per_m": (
                discrete_theory.phase_slope_rad_per_m
            ),
            "theoretical_phase_accumulation_across_aperture_rad": (
                theory.phase_slope_rad_per_m * aperture_m
            ),
            "continuous_vs_discrete_attenuation_absolute_difference_per_m": abs(
                theory.log_amplitude_slope_per_m
                - discrete_theory.log_amplitude_slope_per_m
            ),
            "continuous_vs_discrete_phase_absolute_difference_rad_per_m": abs(
                theory.phase_slope_rad_per_m - discrete_theory.phase_slope_rad_per_m
            ),
            "observed_effective_q": observed_q,
            "theoretical_effective_q": theory.effective_q,
        }
    return metrics


def math_log_abs(value: complex) -> float:
    return math.log(abs(value))


def complex_phase(value: complex) -> float:
    return cmath.phase(value)


def _relative_error(observed: float, expected: float) -> float:
    return abs(observed - expected) / abs(expected)


def test_m42_effective_parameters_and_exact_repeatability(m42_runs):
    assert m42_runs.elastic.effective == EffectiveDeniseParameters(0, 5, 0, ())
    for run in m42_runs.viscoelastic.values():
        assert run.effective == EffectiveDeniseParameters(0, 5, 1, (10.0,))
    reference = m42_runs.viscoelastic[200.0]
    repeat = m42_runs.q200_repeat
    metrics = {
        "effective_parameters": {
            "elastic": asdict(m42_runs.elastic.effective),
            **{str(int(qs)): asdict(run.effective) for qs, run in m42_runs.viscoelastic.items()},
        },
        "relative_l2": relative_l2(reference.traces, repeat.traces),
        "normalized_correlation": normalized_correlation(reference.traces, repeat.traces),
        "model_sha256_equal": reference.model_sha256 == repeat.model_sha256,
    }
    (m42_runs.root / "effective_parameters_repeatability.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert metrics["model_sha256_equal"]
    assert metrics["relative_l2"] <= 1.0e-12
    assert metrics["normalized_correlation"] >= 1.0 - 1.0e-12


def test_m42_high_q_converges_monotonically_to_elastic(m42_runs):
    elastic_windows = _direct_windows(m42_runs.elastic)
    rows = []
    for qs in HIGH_Q_VALUES:
        run = m42_runs.viscoelastic[qs]
        windows = _direct_windows(run)
        far_start, far_stop = _direct_interval(run.config, run.config.receiver_offsets_m()[-1])
        elastic_peak = absolute_peak_index_in_interval(
            m42_runs.elastic.traces[-1],
            start_s=far_start,
            stop_s=far_stop,
            dt_s=run.config.dt_s,
        )
        visco_peak = absolute_peak_index_in_interval(
            run.traces[-1], start_s=far_start, stop_s=far_stop, dt_s=run.config.dt_s
        )
        far_transfer = _frequency_transfer_by_receiver(run, m42_runs.elastic)
        rows.append(
            {
                "qs": qs,
                "complete_relative_l2": relative_l2(run.traces, m42_runs.elastic.traces),
                "direct_window_relative_l2": relative_l2(windows, elastic_windows),
                "direct_window_correlation": normalized_correlation(windows, elastic_windows),
                "far_peak_time_difference_s": (visco_peak - elastic_peak) * run.config.dt_s,
                "far_peak_absolute": peak_absolute(windows[-1]),
                "far_rms": root_mean_square(windows[-1]),
                "far_spectral_amplitude_ratio": {
                    str(frequency): abs(far_transfer[frequency][-1])
                    for frequency in SPECTRAL_FREQUENCIES_HZ
                },
                "far_spectral_phase_rad": {
                    str(frequency): complex_phase(far_transfer[frequency][-1])
                    for frequency in SPECTRAL_FREQUENCIES_HZ
                },
            }
        )
    metrics = {"criteria": {
        "strict_monotonic_complete_and_direct_relative_l2": True,
        "qs_1000_direct_relative_l2_max": HIGH_Q_DIRECT_RELATIVE_L2_MAX,
        "qs_1000_direct_correlation_min": HIGH_Q_CORRELATION_MIN,
    }, "rows": rows}
    (m42_runs.root / "high_q_convergence.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete_errors = [row["complete_relative_l2"] for row in rows]
    direct_errors = [row["direct_window_relative_l2"] for row in rows]
    assert all(left > right for left, right in zip(complete_errors, complete_errors[1:]))
    assert all(left > right for left, right in zip(direct_errors, direct_errors[1:]))
    assert rows[-1]["direct_window_relative_l2"] <= HIGH_Q_DIRECT_RELATIVE_L2_MAX
    assert rows[-1]["direct_window_correlation"] >= HIGH_Q_CORRELATION_MIN


def test_m42_distance_attenuation_and_phase_match_l1_rheology(m42_runs):
    run = m42_runs.viscoelastic[50.0]
    old_hann_metrics = _slope_metrics(
        run,
        m42_runs.elastic,
        half_width_s=DIRECT_WINDOW_HALF_WIDTH_S,
        window_kind="hann",
    )
    metrics = {
        "receiver_distances_m": run.config.receiver_offsets_m(),
        "receiver_aperture_m": (
            max(run.config.receiver_offsets_m()) - min(run.config.receiver_offsets_m())
        ),
        "old_hann_diagnostic": {
            "window_duration_s": 2.0 * DIRECT_WINDOW_HALF_WIDTH_S,
            "approximate_main_lobe_width_hz": approximate_main_lobe_width_hz(
                duration_s=2.0 * DIRECT_WINDOW_HALF_WIDTH_S, kind="hann"
            ),
            "frequencies": old_hann_metrics,
        },
        "calibrated_estimator": {
            "window_kind": SPECTRAL_WINDOW_KIND,
            "tukey_alpha": SPECTRAL_TUKEY_ALPHA,
            "window_duration_s": 2.0 * SPECTRAL_WINDOW_HALF_WIDTH_S,
            "approximate_main_lobe_width_hz": approximate_main_lobe_width_hz(
                duration_s=2.0 * SPECTRAL_WINDOW_HALF_WIDTH_S,
                kind=SPECTRAL_WINDOW_KIND,
                tukey_alpha=SPECTRAL_TUKEY_ALPHA,
            ),
            "mandatory_phase_frequencies_hz": list(PHASE_FREQUENCIES_HZ),
            "diagnostic_phase_frequencies_hz": list(
                PHASE_DIAGNOSTIC_FREQUENCIES_HZ
            ),
            "phase_slope_relative_tolerance": PHASE_SLOPE_RELATIVE_TOLERANCE,
        },
        "usable_frequency_range_hz": [min(SPECTRAL_FREQUENCIES_HZ), max(SPECTRAL_FREQUENCIES_HZ)],
        "frequencies": _slope_metrics(run, m42_runs.elastic),
    }
    (m42_runs.root / "distance_attenuation_phase_l1.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for frequency in ATTENUATION_FREQUENCIES_HZ:
        row = metrics["frequencies"][str(frequency)]
        fit = row["attenuation_fit"]
        values = row["log_amplitudes"]
        assert all(left > right for left, right in zip(values, values[1:]))
        assert fit["slope"] < 0.0
        assert fit["r_squared"] >= SLOPE_R_SQUARED_MIN
        assert max(abs(value) for value in fit["residuals"]) <= LOG_AMPLITUDE_RESIDUAL_MAX
        assert _relative_error(
            fit["slope"], row["theoretical_log_amplitude_slope_per_m"]
        ) <= ATTENUATION_SLOPE_RELATIVE_TOLERANCE
    for frequency in PHASE_FREQUENCIES_HZ:
        row = metrics["frequencies"][str(frequency)]
        fit = row["phase_fit"]
        theory = row["theoretical_phase_slope_rad_per_m"]
        values = row["unwrapped_phase_rad"]
        assert fit["slope"] * theory > 0.0
        assert all(
            (right - left) * theory > 0.0 for left, right in zip(values, values[1:])
        )
        assert fit["r_squared"] >= SLOPE_R_SQUARED_MIN
        assert max(abs(value) for value in fit["residuals"]) <= PHASE_RESIDUAL_MAX_RAD
        assert _relative_error(fit["slope"], theory) <= PHASE_SLOPE_RELATIVE_TOLERANCE


def test_m42_historical_l4_external_qs_parameterization(
    m42_runs, historical_l4_runs, repository_root
):
    frequencies = HISTORICAL_L4_RELAXATION_FREQUENCIES_HZ
    for run in (
        historical_l4_runs.nominal_qs30,
        historical_l4_runs.compensating_qs,
        historical_l4_runs.physical_qs30,
    ):
        require_effective_parameters(
            run.effective,
            mode=0,
            physics=5,
            relaxation_frequencies_hz=_float32_values(frequencies),
            q_parameterization_mode=run.config.q_parameterization_mode,
            q_approx_fmin_hz=run.config.q_approx_fmin_hz or None,
            q_approx_fmax_hz=run.config.q_approx_fmax_hz or None,
            q_approx_df_hz=run.config.q_approx_df_hz or None,
        )
    assert historical_l4_runs.nominal_qs30.config.qs == HISTORICAL_L4_TARGET_Q
    assert math.isclose(
        historical_l4_runs.compensating_qs.config.qs,
        HISTORICAL_L4_COMPENSATING_QS_FILE,
        rel_tol=1.0e-15,
    )
    reader_source = (repository_root / "src" / "SH" / "readmod_visc_SH.c").read_text(
        encoding="utf-8"
    )
    psv_reader_source = (repository_root / "src" / "PSV" / "readmod_visc_PSV.c").read_text(
        encoding="utf-8"
    )
    assert "taus[jj][ii]=q_to_tau(qs, &q_mapping);" in reader_source
    assert "taus[jj][ii]=q_to_tau(qs, &q_mapping);" in psv_reader_source
    assert "taup[jj][ii]=q_to_tau(qp, &q_mapping);" in psv_reader_source

    metrics = {
        "relaxation_frequencies_hz": list(frequencies),
        "historical_optimized_tau": HISTORICAL_L4_OPTIMIZED_TAU,
        "target_q": HISTORICAL_L4_TARGET_Q,
        "reader_mapping_source": "shared q_to_tau used by SH and P/SV external readers",
        "runs": {},
    }
    for name, run in (
        ("nominal_qs30", historical_l4_runs.nominal_qs30),
        ("compensating_qs", historical_l4_runs.compensating_qs),
        ("physical_qs30", historical_l4_runs.physical_qs30),
    ):
        if run.config.q_parameterization_mode == 1:
            reader_tau = target_q_to_tau(
                target_q=run.config.qs,
                relaxation_frequencies_hz=frequencies,
                fmin_hz=run.config.q_approx_fmin_hz,
                fmax_hz=run.config.q_approx_fmax_hz,
                df_hz=run.config.q_approx_df_hz,
            )
        else:
            reader_tau = 2.0 / run.config.qs
        slopes = _slope_metrics(run, m42_runs.elastic)
        rows = {}
        for frequency in SPECTRAL_FREQUENCIES_HZ:
            row = slopes[str(frequency)]
            intended_q = qstd_quality_factor(
                frequency_hz=frequency,
                relaxation_frequencies_hz=frequencies,
                tau=HISTORICAL_L4_OPTIMIZED_TAU,
            )
            reader_q = qstd_quality_factor(
                frequency_hz=frequency,
                relaxation_frequencies_hz=frequencies,
                tau=reader_tau,
            )
            rows[str(frequency)] = row | {
                "historical_intended_q": intended_q,
                "current_reader_theoretical_q": reader_q,
                "observed_vs_historical_q_relative_error": _relative_error(
                    row["observed_effective_q"], intended_q
                ),
                "observed_vs_reader_q_relative_error": _relative_error(
                    row["observed_effective_q"], reader_q
                ),
            }
        metrics["runs"][name] = {
            "qs_file_value": run.config.qs,
            "reader_tau": reader_tau,
            "q_parameterization_mode": run.config.q_parameterization_mode,
            "effective_parameters": asdict(run.effective),
            "model_sha256": run.model_sha256,
            "frequencies": rows,
        }
    (historical_l4_runs.root / "historical_l4_parameterization.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    nominal_rows = metrics["runs"]["nominal_qs30"]["frequencies"]
    compensating_rows = metrics["runs"]["compensating_qs"]["frequencies"]
    physical_rows = metrics["runs"]["physical_qs30"]["frequencies"]
    assert metrics["runs"]["nominal_qs30"]["reader_tau"] == 2.0 / 30.0
    assert math.isclose(
        metrics["runs"]["compensating_qs"]["reader_tau"],
        HISTORICAL_L4_OPTIMIZED_TAU,
        rel_tol=1.0e-15,
    )
    assert abs(metrics["runs"]["physical_qs30"]["reader_tau"] - 0.0386) < 3.0e-4
    for row in nominal_rows.values():
        assert row["observed_vs_reader_q_relative_error"] <= HISTORICAL_L4_Q_RELATIVE_TOLERANCE
        assert row["observed_vs_historical_q_relative_error"] >= 0.25
        assert _relative_error(
            row["attenuation_fit"]["slope"],
            row["theoretical_log_amplitude_slope_per_m"],
        ) <= ATTENUATION_SLOPE_RELATIVE_TOLERANCE
        assert _relative_error(
            row["phase_fit"]["slope"], row["theoretical_phase_slope_rad_per_m"]
        ) <= PHASE_SLOPE_RELATIVE_TOLERANCE
    for row in compensating_rows.values():
        assert row["observed_vs_reader_q_relative_error"] <= HISTORICAL_L4_Q_RELATIVE_TOLERANCE
        assert row["observed_vs_historical_q_relative_error"] <= HISTORICAL_L4_Q_RELATIVE_TOLERANCE
        assert _relative_error(
            row["attenuation_fit"]["slope"],
            row["theoretical_log_amplitude_slope_per_m"],
        ) <= ATTENUATION_SLOPE_RELATIVE_TOLERANCE
        assert _relative_error(
            row["phase_fit"]["slope"], row["theoretical_phase_slope_rad_per_m"]
        ) <= PHASE_SLOPE_RELATIVE_TOLERANCE
    for row in physical_rows.values():
        assert row["observed_vs_reader_q_relative_error"] <= HISTORICAL_L4_Q_RELATIVE_TOLERANCE
        assert row["observed_vs_historical_q_relative_error"] <= HISTORICAL_L4_Q_RELATIVE_TOLERANCE
        assert _relative_error(
            row["attenuation_fit"]["slope"],
            row["theoretical_log_amplitude_slope_per_m"],
        ) <= ATTENUATION_SLOPE_RELATIVE_TOLERANCE
        assert _relative_error(
            row["phase_fit"]["slope"], row["theoretical_phase_slope_rad_per_m"]
        ) <= PHASE_SLOPE_RELATIVE_TOLERANCE

    equivalence_l2 = relative_l2(
        historical_l4_runs.physical_qs30.traces,
        historical_l4_runs.compensating_qs.traces,
    )
    equivalence_correlation = normalized_correlation(
        historical_l4_runs.physical_qs30.traces,
        historical_l4_runs.compensating_qs.traces,
    )
    metrics["physical_vs_compensating_equivalence"] = {
        "relative_l2": equivalence_l2,
        "normalized_correlation": equivalence_correlation,
    }
    (historical_l4_runs.root / "historical_l4_parameterization.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert equivalence_l2 <= 0.005
    assert equivalence_correlation >= 0.999


@pytest.mark.extended
def test_m42_multiple_relaxation_mechanisms_change_rheology(
    m42_runs, repository_root, denise_binary, mpiexec
):
    frequencies = (5.0, 10.0, 20.0)
    run = _run_sh_case(
        m42_runs.root / "qs_50_l3",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=_viscoelastic_config(50.0, frequencies),
        relaxation_frequencies_hz=frequencies,
    )
    assert run.effective == EffectiveDeniseParameters(0, 5, 3, frequencies)
    l1 = m42_runs.viscoelastic[50.0]
    slope_metrics = _slope_metrics(run, m42_runs.elastic)
    metrics = {
        "effective_parameters": asdict(run.effective),
        "l3_vs_l1_relative_l2": relative_l2(run.traces, l1.traces),
        "receiver_distances_m": run.config.receiver_offsets_m(),
        "frequencies": slope_metrics,
    }
    (m42_runs.root / "multiple_relaxation_l3.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert metrics["l3_vs_l1_relative_l2"] >= 1.0e-3
    for row in slope_metrics.values():
        attenuation_fit = row["attenuation_fit"]
        phase_fit = row["phase_fit"]
        assert _relative_error(
            attenuation_fit["slope"], row["theoretical_log_amplitude_slope_per_m"]
        ) <= ATTENUATION_SLOPE_RELATIVE_TOLERANCE
        assert _relative_error(
            phase_fit["slope"], row["theoretical_phase_slope_rad_per_m"]
        ) <= PHASE_SLOPE_RELATIVE_TOLERANCE
        assert _relative_error(
            row["observed_effective_q"], row["theoretical_effective_q"]
        ) <= EFFECTIVE_Q_RELATIVE_TOLERANCE
