from __future__ import annotations

import cmath
import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pytest

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case as generate_elastic
from tests.cases.homogeneous_viscoelastic import (
    ViscoelasticPSVConfig,
    generate_viscoelastic_psv_case,
)
from tests.utilities.effective_parameters import (
    EffectiveDeniseParameters,
    read_effective_parameters,
    require_effective_parameters,
)
from tests.utilities.qstd_reference import target_q_to_tau
from tests.utilities.runner import RunResult, result_summary, run_denise
from tests.utilities.seismogram import (
    all_finite,
    normalized_correlation,
    read_ascii_seismograms,
    relative_l2,
    signal_energy,
    time_interval,
)
from tests.utilities.staggered_grid import input_field_position
from tests.utilities.viscoelastic_rheology import (
    discrete_rheology_prediction,
    linear_fit,
    rheology_prediction,
    transfer_spectrum,
    unwrap_phase,
)


pytestmark = pytest.mark.integration


RELAXATION_FREQUENCIES_HZ = (2.7105, 12.2792, 68.1930, 265.2297)
Q_APPROX_FMIN_HZ = 5.0
Q_APPROX_FMAX_HZ = 120.0
Q_APPROX_DF_HZ = 5.0
SPECTRAL_FREQUENCIES_HZ = (8.0, 10.0, 12.0, 14.0)
SPECTRAL_WINDOW_HALF_WIDTH_S = 0.09
SPECTRAL_TUKEY_ALPHA = 0.2

# The synthetic known-answer gate is 5%. These predeclared black-box limits
# add margin for eighth-order grid dispersion and staggered source/receiver
# sampling while remaining below the earlier M4.2 upper bounds of 15%/20%.
ATTENUATION_RELATIVE_TOLERANCE = 0.10
PHASE_RELATIVE_TOLERANCE = 0.12
FIT_R_SQUARED_MIN = 0.98
# A relative phase-slope error becomes ill-conditioned when the theoretical
# phase change across the complete receiver aperture is only a few hundredths
# of a radian.  Keep those bins as diagnostics, but require at least 0.07 rad
# of predicted accumulation before using a bin as a quantitative phase guard.
MIN_PHASE_ACCUMULATION_RAD = 0.07
CROSS_RELATIVE_L2_MAX = 5.0e-3
MPI_RELATIVE_L2_MAX = 1.0e-5
MPI_CORRELATION_MIN = 0.999999


BASE = HomogeneousPSVConfig(
    nx=300,
    ny=180,
    time_s=0.9,
    dt_s=0.0004,
    source_x_m=900.0,
    source_y_m=900.0,
    receivers_m=tuple((x, 900.0) for x in (1300.0, 1400.0, 1500.0, 1600.0, 1700.0)),
    source_frequency_hz=10.0,
)


@dataclass(frozen=True)
class PSVRun:
    directory: Path
    mode: str
    config: HomogeneousPSVConfig
    traces: list[list[float]]
    effective: EffectiveDeniseParameters
    model_sha256: dict[str, str]
    result: RunResult
    nprocx: int
    nprocy: int


@dataclass(frozen=True)
class ModeRuns:
    elastic: PSVRun
    q50: PSVRun
    q50_repeat: PSVRun
    q200: PSVRun
    q1000: PSVRun
    cross: PSVRun


@dataclass(frozen=True)
class M43Runs:
    root: Path
    p: ModeRuns
    sv: ModeRuns


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float32_values(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(struct.unpack("f", struct.pack("f", value))[0] for value in values)


def _config(mode: str, *, qp: float | None = None, qs: float | None = None):
    source_type = 1 if mode == "P" else 3
    base = replace(BASE, source_type=source_type, source_azimuth_deg=0.0)
    if qp is None or qs is None:
        return base
    return ViscoelasticPSVConfig(
        **asdict(base),
        qp=qp,
        qs=qs,
        relaxation_frequencies_hz=RELAXATION_FREQUENCIES_HZ,
        q_parameterization_mode=1,
        q_approx_fmin_hz=Q_APPROX_FMIN_HZ,
        q_approx_fmax_hz=Q_APPROX_FMAX_HZ,
        q_approx_df_hz=Q_APPROX_DF_HZ,
    )


def _run(
    directory: Path,
    *,
    mode: str,
    config: HomogeneousPSVConfig,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    nprocx: int = 1,
    nprocy: int = 1,
) -> PSVRun:
    viscoelastic = isinstance(config, ViscoelasticPSVConfig)
    generator = generate_viscoelastic_psv_case if viscoelastic else generate_elastic
    generator(directory, config=config, nprocx=nprocx, nprocy=nprocy)
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration=config.as_metadata()
        | {"mode_under_test": mode, "nprocx": nprocx, "nprocy": nprocy},
    )
    assert result.returncode == 0, result_summary(result)
    component = "vx" if mode == "P" else "vy"
    output = directory / "su" / f"homogeneous_{component}.asc.shot1"
    assert output.is_file() and output.stat().st_size > 0
    traces = read_ascii_seismograms(output, config.receiver_count, config.samples_per_trace)
    assert len(traces) == config.receiver_count
    assert all(len(trace) == config.samples_per_trace for trace in traces)
    assert all_finite(traces)
    assert signal_energy(sample for trace in traces for sample in trace) > 0.0

    effective = read_effective_parameters(result.stdout_path)
    frequencies = _float32_values(RELAXATION_FREQUENCIES_HZ) if viscoelastic else ()
    require_effective_parameters(
        effective,
        mode=0,
        physics=1,
        relaxation_frequencies_hz=frequencies,
        q_parameterization_mode=1 if viscoelastic else 0,
        q_approx_fmin_hz=Q_APPROX_FMIN_HZ if viscoelastic else None,
        q_approx_fmax_hz=Q_APPROX_FMAX_HZ if viscoelastic else None,
        q_approx_df_hz=Q_APPROX_DF_HZ if viscoelastic else None,
    )
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["returncode"] == 0
    assert metadata["mpi_ranks"] == nprocx * nprocy
    assert len(metadata["executable"]["sha256"]) == 64
    model = directory / "model" / "homogeneous"
    names = ("vp", "vs", "rho", "qp", "qs") if viscoelastic else ("vp", "vs", "rho")
    hashes = {name: _sha256(model.with_suffix(f".{name}")) for name in names}
    return PSVRun(directory, mode, config, traces, effective, hashes, result, nprocx, nprocy)


def _mode_runs(root, mode, repository_root, denise_binary, mpiexec) -> ModeRuns:
    if mode == "P":
        values = {
            "q50": (50.0, 1000.0),
            "q200": (200.0, 1000.0),
            "q1000": (1000.0, 1000.0),
            "cross": (50.0, 50.0),
        }
    else:
        values = {
            "q50": (1000.0, 50.0),
            "q200": (1000.0, 200.0),
            "q1000": (1000.0, 1000.0),
            "cross": (50.0, 50.0),
        }
    runs = {
        name: _run(
            root / name,
            mode=mode,
            config=_config(mode, qp=qp, qs=qs),
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
        )
        for name, (qp, qs) in values.items()
    }
    repeat = _run(
        root / "q50_repeat",
        mode=mode,
        config=_config(mode, qp=values["q50"][0], qs=values["q50"][1]),
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
    )
    elastic = _run(
        root / "elastic",
        mode=mode,
        config=_config(mode),
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
    )
    return ModeRuns(elastic, runs["q50"], repeat, runs["q200"], runs["q1000"], runs["cross"])


@pytest.fixture(scope="module")
def m43_runs(tmp_path_factory, repository_root, denise_binary, mpiexec) -> M43Runs:
    root = tmp_path_factory.mktemp("m43_psv_rheology")
    return M43Runs(
        root,
        _mode_runs(root / "p", "P", repository_root, denise_binary, mpiexec),
        _mode_runs(root / "sv", "SV", repository_root, denise_binary, mpiexec),
    )


def _velocity(config: HomogeneousPSVConfig, mode: str) -> float:
    return config.vp_m_s if mode == "P" else config.vs_m_s


def _native_geometry(config: HomogeneousPSVConfig, mode: str) -> dict[str, object]:
    field = "vx" if mode == "P" else "vy"
    source_field = "sxx" if config.source_type == 1 else field
    source_input = (config.source_x_m, config.source_y_m)
    source_physical = input_field_position(source_input, config.dh_m, source_field)
    receivers_physical = [
        input_field_position(receiver, config.dh_m, field) for receiver in config.receivers_m
    ]
    offsets = [receiver[0] - source_physical[0] for receiver in receivers_physical]
    return {
        "source_type": config.source_type,
        "source_native_field": source_field,
        "native_component": field,
        "projection": "axis-aligned receiver component with source and receiver positions evaluated on their native staggered fields",
        "source_input_m": list(source_input),
        "source_physical_m": list(source_physical),
        "receiver_input_m": [list(value) for value in config.receivers_m],
        "receiver_physical_m": [list(value) for value in receivers_physical],
        "offsets_m": offsets,
    }


def _direct_windows(run: PSVRun) -> list[list[float]]:
    velocity = _velocity(run.config, run.mode)
    offsets = _native_geometry(run.config, run.mode)["offsets_m"]
    return [
        time_interval(
            trace,
            start_s=1.5 / run.config.source_frequency_hz + offset / velocity - 0.09,
            stop_s=1.5 / run.config.source_frequency_hz + offset / velocity + 0.09,
            dt_s=run.config.dt_s,
        )
        for trace, offset in zip(run.traces, offsets)
    ]


def _slope_metrics(viscoelastic: PSVRun, elastic: PSVRun) -> dict[str, object]:
    mode = viscoelastic.mode
    velocity = _velocity(elastic.config, mode)
    geometry = _native_geometry(elastic.config, mode)
    offsets = geometry["offsets_m"]
    transfers = {frequency: [] for frequency in SPECTRAL_FREQUENCIES_HZ}
    for offset, visco_trace, elastic_trace in zip(offsets, viscoelastic.traces, elastic.traces):
        center = 1.5 / elastic.config.source_frequency_hz + offset / velocity
        samples = transfer_spectrum(
            time_interval(
                visco_trace,
                start_s=center - SPECTRAL_WINDOW_HALF_WIDTH_S,
                stop_s=center + SPECTRAL_WINDOW_HALF_WIDTH_S,
                dt_s=elastic.config.dt_s,
            ),
            time_interval(
                elastic_trace,
                start_s=center - SPECTRAL_WINDOW_HALF_WIDTH_S,
                stop_s=center + SPECTRAL_WINDOW_HALF_WIDTH_S,
                dt_s=elastic.config.dt_s,
            ),
            dt_s=elastic.config.dt_s,
            frequencies_hz=SPECTRAL_FREQUENCIES_HZ,
            window_kind="tukey",
            tukey_alpha=SPECTRAL_TUKEY_ALPHA,
        )
        assert tuple(sample.frequency_hz for sample in samples) == SPECTRAL_FREQUENCIES_HZ
        for sample in samples:
            transfers[sample.frequency_hz].append(sample.value)

    quality_factor = viscoelastic.config.qp if mode == "P" else viscoelastic.config.qs
    tau = target_q_to_tau(
        target_q=quality_factor,
        relaxation_frequencies_hz=RELAXATION_FREQUENCIES_HZ,
        fmin_hz=Q_APPROX_FMIN_HZ,
        fmax_hz=Q_APPROX_FMAX_HZ,
        df_hz=Q_APPROX_DF_HZ,
    )
    rows = {}
    for frequency, values in transfers.items():
        attenuation = linear_fit(offsets, [math.log(abs(value)) for value in values])
        phase = linear_fit(offsets, unwrap_phase([cmath.phase(value) for value in values]))
        continuous = rheology_prediction(
            frequency_hz=frequency,
            vs_m_s=velocity,
            density_kg_m3=elastic.config.density_kg_m3,
            qs_input=quality_factor,
            relaxation_frequencies_hz=RELAXATION_FREQUENCIES_HZ,
            tau_override=tau,
        )
        theory = discrete_rheology_prediction(
            frequency_hz=frequency,
            vs_m_s=velocity,
            density_kg_m3=elastic.config.density_kg_m3,
            qs_input=quality_factor,
            relaxation_frequencies_hz=RELAXATION_FREQUENCIES_HZ,
            dt_s=elastic.config.dt_s,
            dh_m=elastic.config.dh_m,
            tau_override=tau,
        )
        phase_accumulation = abs(theory.phase_slope_rad_per_m) * (
            max(offsets) - min(offsets)
        )
        rows[str(frequency)] = {
            "frequency_hz": frequency,
            "theoretical_attenuation_slope_per_m": theory.log_amplitude_slope_per_m,
            "observed_attenuation_slope_per_m": attenuation.slope,
            "attenuation_relative_error": abs(
                attenuation.slope - theory.log_amplitude_slope_per_m
            ) / abs(theory.log_amplitude_slope_per_m),
            "attenuation_r_squared": attenuation.r_squared,
            "theoretical_phase_slope_rad_per_m": theory.phase_slope_rad_per_m,
            "observed_phase_slope_rad_per_m": phase.slope,
            "phase_relative_error": abs(phase.slope - theory.phase_slope_rad_per_m)
            / abs(theory.phase_slope_rad_per_m),
            "theoretical_phase_accumulation_rad": phase_accumulation,
            "phase_is_quantitative": phase_accumulation
            >= MIN_PHASE_ACCUMULATION_RAD,
            "phase_r_squared": phase.r_squared,
            "continuous_attenuation_slope_per_m": continuous.log_amplitude_slope_per_m,
            "continuous_phase_slope_rad_per_m": continuous.phase_slope_rad_per_m,
            "attenuation_fit_residuals": list(attenuation.residuals),
            "phase_fit_residuals_rad": list(phase.residuals),
        }
    metadata = json.loads(viscoelastic.result.metadata_path.read_text(encoding="utf-8"))
    return {
        "mode": mode,
        "Qp": viscoelastic.config.qp,
        "Qs": viscoelastic.config.qs,
        "taup": target_q_to_tau(
            target_q=viscoelastic.config.qp,
            relaxation_frequencies_hz=RELAXATION_FREQUENCIES_HZ,
            fmin_hz=Q_APPROX_FMIN_HZ,
            fmax_hz=Q_APPROX_FMAX_HZ,
            df_hz=Q_APPROX_DF_HZ,
        ),
        "taus": target_q_to_tau(
            target_q=viscoelastic.config.qs,
            relaxation_frequencies_hz=RELAXATION_FREQUENCIES_HZ,
            fmin_hz=Q_APPROX_FMIN_HZ,
            fmax_hz=Q_APPROX_FMAX_HZ,
            df_hz=Q_APPROX_DF_HZ,
        ),
        "L": len(RELAXATION_FREQUENCIES_HZ),
        "FL_hz": list(RELAXATION_FREQUENCIES_HZ),
        "Q_approximation_band_hz": [Q_APPROX_FMIN_HZ, Q_APPROX_FMAX_HZ, Q_APPROX_DF_HZ],
        "source_frequency_hz": viscoelastic.config.source_frequency_hz,
        "geometry": geometry,
        "frequencies": rows,
        "provenance": metadata,
    }


@pytest.mark.parametrize("mode", ("P", "SV"))
def test_m43_quantitative_attenuation_and_phase(m43_runs: M43Runs, mode: str):
    runs = m43_runs.p if mode == "P" else m43_runs.sv
    metrics = _slope_metrics(runs.q50, runs.elastic)
    (m43_runs.root / f"{mode.lower()}_quantitative_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for row in metrics["frequencies"].values():
        assert row["attenuation_relative_error"] <= ATTENUATION_RELATIVE_TOLERANCE
        assert row["attenuation_r_squared"] >= FIT_R_SQUARED_MIN
        if row["phase_is_quantitative"]:
            assert row["phase_relative_error"] <= PHASE_RELATIVE_TOLERANCE
            assert row["phase_r_squared"] >= FIT_R_SQUARED_MIN


@pytest.mark.parametrize("mode", ("P", "SV"))
def test_m43_repeatability_cross_sensitivity_and_negative_control(m43_runs: M43Runs, mode: str):
    runs = m43_runs.p if mode == "P" else m43_runs.sv
    assert runs.q50.traces == runs.q50_repeat.traces
    cross_l2 = relative_l2(_direct_windows(runs.q50), _direct_windows(runs.cross))
    cross_correlation = normalized_correlation(
        _direct_windows(runs.q50), _direct_windows(runs.cross)
    )
    negative_l2 = relative_l2(_direct_windows(runs.q50), _direct_windows(runs.elastic))
    metrics = {
        "mode": mode,
        "constitutive_cross_effect_expected": 0.0,
        "cross_relative_l2": cross_l2,
        "cross_normalized_correlation": cross_correlation,
        "moderate_q_vs_elastic_relative_l2": negative_l2,
        "repeatability_relative_l2": relative_l2(runs.q50.traces, runs.q50_repeat.traces),
    }
    (m43_runs.root / f"{mode.lower()}_cross_repeatability.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert cross_l2 <= CROSS_RELATIVE_L2_MAX
    assert cross_correlation >= 0.99999
    assert negative_l2 >= 0.01


@pytest.mark.parametrize("mode", ("P", "SV"))
def test_m43_high_q_converges_monotonically_to_elastic(m43_runs: M43Runs, mode: str):
    runs = m43_runs.p if mode == "P" else m43_runs.sv
    errors = {
        q: relative_l2(_direct_windows(run), _direct_windows(runs.elastic))
        for q, run in ((50, runs.q50), (200, runs.q200), (1000, runs.q1000))
    }
    correlations = {
        q: normalized_correlation(_direct_windows(run), _direct_windows(runs.elastic))
        for q, run in ((50, runs.q50), (200, runs.q200), (1000, runs.q1000))
    }
    (m43_runs.root / f"{mode.lower()}_high_q_convergence.json").write_text(
        json.dumps({"relative_l2": errors, "correlation": correlations}, indent=2) + "\n",
        encoding="utf-8",
    )
    assert errors[50] > errors[200] > errors[1000]
    assert correlations[50] < correlations[200] < correlations[1000]
    assert errors[1000] <= 0.025


@pytest.mark.parametrize(("mode", "nprocx", "nprocy"), (("P", 2, 1), ("SV", 1, 2)))
def test_m43_selected_mpi_reproducibility(
    tmp_path, repository_root, denise_binary, mpiexec, m43_runs, mode, nprocx, nprocy
):
    reference = (m43_runs.p if mode == "P" else m43_runs.sv).q50
    variant = _run(
        tmp_path / f"{mode.lower()}_{nprocx}x{nprocy}",
        mode=mode,
        config=reference.config,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        nprocx=nprocx,
        nprocy=nprocy,
    )
    rel = relative_l2(reference.traces, variant.traces)
    corr = normalized_correlation(reference.traces, variant.traces)
    metrics = {
        "mode": mode,
        "reference_decomposition": [1, 1],
        "variant_decomposition": [nprocx, nprocy],
        "relative_l2": rel,
        "normalized_correlation": corr,
        "effective_parameters_equal": reference.effective == variant.effective,
    }
    (tmp_path / f"{mode.lower()}_mpi_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    assert metrics["effective_parameters_equal"]
    assert rel <= MPI_RELATIVE_L2_MAX
    assert corr >= MPI_CORRELATION_MIN


@pytest.mark.extended
@pytest.mark.parametrize(
    ("mode", "nprocx", "nprocy"),
    (("P", 1, 2), ("P", 2, 2), ("SV", 2, 1), ("SV", 2, 2)),
)
def test_m43_extended_mpi_matrix(
    tmp_path, repository_root, denise_binary, mpiexec, m43_runs, mode, nprocx, nprocy
):
    reference = (m43_runs.p if mode == "P" else m43_runs.sv).q50
    variant = _run(
        tmp_path / f"extended_{mode.lower()}_{nprocx}x{nprocy}",
        mode=mode,
        config=reference.config,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        nprocx=nprocx,
        nprocy=nprocy,
    )
    assert relative_l2(reference.traces, variant.traces) <= MPI_RELATIVE_L2_MAX
    assert normalized_correlation(reference.traces, variant.traces) >= MPI_CORRELATION_MIN
