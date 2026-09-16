"""M7c physical Vp/Vs/rho/Qp/Qs finite-difference acceptance oracle.

The reference objective is the production P/SV ``calc_res`` L2 path with
``LNORM=2`` and ``GRAD_FORM=2``: both vx and vy receivers contribute
``r = synthetic - observed`` (sample one is explicitly zeroed by production),
and ``J = 0.5 * sum(r**2)``.  There is no dt factor, component weighting, or
normalization denominator.  The production residual files are time-reversed
only for adjoint injection; this oracle evaluates the unreversed physical
receiver samples before that storage convention.
"""

from __future__ import annotations

import json
import math
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, _parameter_lines
from tests.utilities.fwi_gradient import read_su_float_samples
from tests.utilities.runner import result_summary, run_denise


pytestmark = pytest.mark.integration

EPSILONS = (0.02, 0.01, 0.005)
PHYSICAL_FIELDS = ("vp", "vs", "rho", "qp", "qs")
# The measured float32 central-FD windows differ by at most 0.25 percent.
# Two percent leaves a conservative factor-eight margin without obscuring a
# channel error or a missing physical chain-rule term.
GREEN_RELATIVE_ERROR_MAX = 0.02


@dataclass(frozen=True)
class ViscoPSVOracleConfig:
    """Small interior, heterogeneous, one-rank L=1 P/SV experiment."""

    nx: int = 48
    ny: int = 44
    dh_m: float = 10.0
    time_s: float = 0.24
    dt_s: float = 0.0004
    vp_m_s: float = 2900.0
    vs_m_s: float = 1550.0
    rho_kg_m3: float = 1950.0
    qp: float = 38.0
    qs: float = 23.0
    source_x_m: float = 180.0
    source_y_m: float = 170.0
    source_frequency_hz: float = 13.0
    receivers_m: tuple[tuple[float, float], ...] = (
        (250.0, 140.0), (290.0, 180.0), (330.0, 210.0), (370.0, 250.0)
    )

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny

    @property
    def samples_per_trace(self) -> int:
        return round(self.time_s / self.dt_s)

    @property
    def receiver_count(self) -> int:
        return len(self.receivers_m)


def _replace(records: list[str], key: str, value: str) -> None:
    matches = [i for i, record in enumerate(records) if record.split("=", 1)[0].strip() == key]
    assert len(matches) == 1, f"expected exactly one {key} record"
    records[matches[0]] = f"{key} ={value}"


def _base_parameter_records(config: ViscoPSVOracleConfig, *, mode: int) -> list[str]:
    base = HomogeneousPSVConfig(
        nx=config.nx, ny=config.ny, dh_m=config.dh_m, time_s=config.time_s,
        dt_s=config.dt_s, vp_m_s=config.vp_m_s, vs_m_s=config.vs_m_s,
        density_kg_m3=config.rho_kg_m3, source_x_m=config.source_x_m,
        source_y_m=config.source_y_m, source_frequency_hz=config.source_frequency_hz,
        source_type=4, receivers_m=config.receivers_m, fd_order=4,
        absorbing_width_gridpoints=8, damping_velocity_m_s=config.vp_m_s,
        pml_frequency_hz=config.source_frequency_hz,
    )
    records = _parameter_lines(base, 1, 1)
    overrides = {
        "MODE": str(mode), "MFILE": "model/current", "L": "1", "FL": "10.0",
        "SEIS_FORMAT": "1", "SEIS_FILE_VX": "su/synthetic_x.su",
        "SEIS_FILE_VY": "su/synthetic_y.su", "ITERMAX": "1",
        "JACOBIAN": "jacobian/gradient", "DATA_DIR": "observed",
        "INVMAT1": "1", "GRAD_FORM": "2", "QUELLTYPB": "1",
        "GRAD_METHOD": "1", "NLBFGS": "1", "DTINV": "1",
        "INV_MOD_OUT": "0", "MODEL_FILTER": "0", "GRAD_FILTER": "0",
    }
    for key, value in overrides.items():
        _replace(records, key, value)
    # read_par consumes the first character of every non-comment line before
    # scanning it. The one-character L key therefore needs this sentinel space.
    records[24] = " L =1"
    return records


def _workflow() -> str:
    return (
        "PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ "
        "INV_VP_ITER INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP "
        "WD_DAMP1 EPRECOND LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE "
        "OFFSET_MUTE OFFSETC SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER\n"
        "0.01 0 0.0 0.0 4 0 0.0 0.0 0.0 1 1 1 0 0 0.0 0.0 0 2 0 0 0.0 "
        "0.0 0 0 0.0 1.0 1.0 0 0.0 0\n"
    )


def _write_float_grid(path: Path, values: Sequence[float]) -> None:
    with path.open("wb") as stream:
        array("f", values).tofile(stream)


def _write_case(directory: Path, *, config: ViscoPSVOracleConfig, model: Mapping[str, Sequence[float]], mode: int, observed: Path | None = None) -> None:
    for name in ("model", "su", "log", "snap", "wavelet", "jacobian", "taper", "picked_times", "trace_kill", "gravity", "observed"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    for name in PHYSICAL_FIELDS:
        values = model[name]
        assert len(values) == config.cell_count
        _write_float_grid(directory / "model" / f"current.{name}", values)
    (directory / "source.dat").write_text(f"1\n{config.source_x_m} 0.0 {config.source_y_m} 0.0 {config.source_frequency_hz} 1.0 20.0 4\n", encoding="ascii")
    (directory / "receiver.dat").write_text("".join(f"{x} {y}\n" for x, y in config.receivers_m), encoding="ascii")
    records = _base_parameter_records(config, mode=mode)
    (directory / "denise.inp").write_text("# M7c viscoelastic P/SV physical gradient oracle\n" + "".join(f"# positional parameter {index:03d}\n{record}\n" for index, record in enumerate(records, start=1)), encoding="ascii")
    (directory / "workflow.inp").write_text(_workflow(), encoding="ascii")
    if observed is not None:
        for component in ("x", "y"):
            (directory / "observed" / f"synthetic_{component}.su.shot1").write_bytes((observed / "su" / f"synthetic_{component}.su.shot1").read_bytes())


def _pattern(config: ViscoPSVOracleConfig, *, phase: float, cx: float, cy: float) -> list[float]:
    values: list[float] = []
    for ix in range(config.nx):
        x = (ix + 0.5) / config.nx
        for iy in range(config.ny):
            y = (iy + 0.5) / config.ny
            envelope = math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.055)
            values.append(envelope * (0.65 + 0.35 * math.sin(5.0 * x + 4.0 * y + phase)))
    maximum = max(values)
    return [value / maximum for value in values]


def _reference_model(config: ViscoPSVOracleConfig) -> dict[str, list[float]]:
    references = {"vp": config.vp_m_s, "vs": config.vs_m_s, "rho": config.rho_kg_m3, "qp": config.qp, "qs": config.qs}
    amplitudes = {"vp": 0.045, "vs": 0.050, "rho": 0.035, "qp": 0.16, "qs": 0.20}
    patterns = {name: _pattern(config, phase=phase, cx=cx, cy=cy) for name, phase, cx, cy in (("vp", 0.1, 0.52, 0.52), ("vs", 1.1, 0.48, 0.57), ("rho", 2.2, 0.55, 0.45), ("qp", 2.9, 0.45, 0.47), ("qs", 4.0, 0.58, 0.58))}
    return {name: [references[name] * (1.0 + amplitudes[name] * value) for value in patterns[name]] for name in PHYSICAL_FIELDS}


def _directions(config: ViscoPSVOracleConfig) -> dict[str, dict[str, list[float]]]:
    fields = {name: _pattern(config, phase=phase, cx=cx, cy=cy) for name, phase, cx, cy in (("vp", 0.7, 0.51, 0.49), ("vs", 1.8, 0.46, 0.55), ("rho", 2.7, 0.55, 0.46), ("qp", 3.5, 0.43, 0.50), ("qs", 4.6, 0.58, 0.56))}
    return {name: {name: fields[name]} for name in PHYSICAL_FIELDS} | {"joint": fields}


def _target_model(model: Mapping[str, Sequence[float]], config: ViscoPSVOracleConfig) -> dict[str, list[float]]:
    target = {name: _pattern(config, phase=phase, cx=cx, cy=cy) for name, phase, cx, cy in (("vp", 0.3, 0.55, 0.54), ("vs", 1.5, 0.47, 0.49), ("rho", 2.4, 0.52, 0.44), ("qp", 3.2, 0.44, 0.57), ("qs", 4.4, 0.59, 0.48))}
    fraction = {"vp": 0.025, "vs": 0.028, "rho": 0.020, "qp": 0.10, "qs": 0.12}
    return {name: [value * (1.0 + fraction[name] * weight) for value, weight in zip(model[name], target[name])] for name in PHYSICAL_FIELDS}


def _perturb(model: Mapping[str, Sequence[float]], direction: Mapping[str, Sequence[float]], epsilon: float) -> dict[str, list[float]]:
    result = {name: list(values) for name, values in model.items()}
    for name, weights in direction.items():
        result[name] = [value * (1.0 + epsilon * weight) for value, weight in zip(model[name], weights)]
    assert all(value > 0.0 and math.isfinite(value) for values in result.values() for value in values)
    return result


def _run_forward(directory: Path, *, repository_root: Path, denise_binary: Path, mpiexec: str, config: ViscoPSVOracleConfig) -> dict[str, list[float]]:
    result = run_denise(repository_root=repository_root, case_directory=directory, denise_binary=denise_binary, mpiexec=mpiexec, ranks=1, configuration={"oracle": "M7c", "mode": 0}, timeout_seconds=90.0)
    assert result.returncode == 0, result_summary(result)
    return {component: read_su_float_samples(directory / "su" / f"synthetic_{component}.su.shot1", config.receiver_count, config.samples_per_trace) for component in ("x", "y")}


def _objective(synthetic: Mapping[str, Sequence[float]], observed: Mapping[str, Sequence[float]]) -> float:
    residual = [sample - data for component in ("x", "y") for sample, data in zip(synthetic[component], observed[component])]
    samples_per_trace = len(synthetic["x"]) // 4
    for component in ("x", "y"):
        offset = 0 if component == "x" else len(synthetic["x"])
        for trace in range(4):
            residual[offset + trace * samples_per_trace] = 0.0
    return 0.5 * math.fsum(value * value for value in residual)


def _central_fd(rows: list[dict[str, float]]) -> dict[str, object]:
    derivatives = [row["derivative"] for row in rows]
    stable = derivatives[-2:]
    spread = abs(stable[1] - stable[0]) / max(abs(stable[0]), abs(stable[1]), 1.0e-30)
    return {"rows": rows, "stable_window": [rows[-2]["epsilon"], rows[-1]["epsilon"]], "derivative": math.fsum(stable) / 2.0, "relative_window_spread": spread}


def _require_future_physical_products(
    fd: Mapping[str, Mapping[str, object]], products: Mapping[str, float]
) -> None:
    """Frozen unscaled GREEN criterion for the repaired physical interface.

    The M7c repair must pass its actual Vp/Vs/rho/Qp/Qs directional products
    here, including ``joint``.  It must not alter the FD reference, physical
    directions, sign convention, or introduce a fitted scale or time shift.
    """
    required = set(PHYSICAL_FIELDS) | {"joint"}
    assert set(products) == required
    for name in required:
        reference = float(fd[name]["derivative"])
        analytic = products[name]
        assert math.isfinite(analytic)
        assert analytic * reference > 0.0, f"{name} gradient has wrong sign"
        assert abs(analytic - reference) / max(abs(analytic), abs(reference), 1.0e-30) <= GREEN_RELATIVE_ERROR_MAX


@pytest.fixture(scope="module")
def m7c_fd_measurements(tmp_path_factory, repository_root, denise_binary, mpiexec):
    root = tmp_path_factory.mktemp("m7c_visco_psv_physical_gradient")
    config = ViscoPSVOracleConfig()
    model = _reference_model(config)
    observed_directory = root / "observed_truth"
    _write_case(observed_directory, config=config, model=_target_model(model, config), mode=0)
    observed = _run_forward(observed_directory, repository_root=repository_root, denise_binary=denise_binary, mpiexec=mpiexec, config=config)
    results: dict[str, dict[str, object]] = {}
    for name, direction in _directions(config).items():
        rows: list[dict[str, float]] = []
        for epsilon in EPSILONS:
            objectives: dict[float, float] = {}
            for sign in (-1.0, 1.0):
                directory = root / "fd" / name / f"eps_{sign * epsilon:+.4f}"
                _write_case(directory, config=config, model=_perturb(model, direction, sign * epsilon), mode=0)
                objectives[sign] = _objective(_run_forward(directory, repository_root=repository_root, denise_binary=denise_binary, mpiexec=mpiexec, config=config), observed)
            rows.append({"epsilon": epsilon, "j_plus": objectives[1.0], "j_minus": objectives[-1.0], "derivative": (objectives[1.0] - objectives[-1.0]) / (2.0 * epsilon)})
        results[name] = _central_fd(rows)
    report = {"objective_contract": "J=0.5*sum_{vx,vy,traces,samples>1}((synthetic-observed)^2); unweighted and unnormalized", "fd": results, "proposed_green_relative_error_max": GREEN_RELATIVE_ERROR_MAX}
    (root / "m7c_fd_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"root": root, "config": config, "model": model, "observed_directory": observed_directory, "results": results}


def test_physical_viscoelastic_psv_central_differences_are_stable_and_distinct(m7c_fd_measurements):
    results = m7c_fd_measurements["results"]
    for name in (*PHYSICAL_FIELDS, "joint"):
        entry = results[name]
        assert math.isfinite(entry["derivative"])
        assert abs(entry["derivative"]) > 1.0e-14, f"{name} direction is numerically insensitive"
        assert entry["relative_window_spread"] <= GREEN_RELATIVE_ERROR_MAX
    qp, qs = results["qp"]["derivative"], results["qs"]["derivative"]
    assert not math.isclose(qp, qs, rel_tol=0.02, abs_tol=1.0e-14)


@pytest.mark.xfail(strict=True, reason="M7c current baseline: viscoelastic P/SV exposes no Qp/Qs physical gradient channel")
def test_current_viscoelastic_psv_gradient_baseline_is_incomplete(m7c_fd_measurements, repository_root, denise_binary, mpiexec):
    """Frozen RED baseline: today's production FWI has no physical Qp/Qs output."""
    config = m7c_fd_measurements["config"]
    directory = m7c_fd_measurements["root"] / "current_fwi"
    _write_case(directory, config=config, model=m7c_fd_measurements["model"], mode=1, observed=m7c_fd_measurements["observed_directory"])
    result = run_denise(repository_root=repository_root, case_directory=directory, denise_binary=denise_binary, mpiexec=mpiexec, ranks=1, configuration={"oracle": "M7c", "mode": 1, "role": "current_gradient_baseline"}, timeout_seconds=90.0)
    assert result.returncode == 0, result_summary(result)
    # Current FWI executes transient Vp/Vs/rho correlations but persists none
    # of their raw directional products and has no Qp/Qs channel at all. The
    # future repair must wire actual physical directional products into
    # _require_future_physical_products without changing the frozen FD
    # reference, sign/scale contract, tolerance, or directions.
    produced = {path.name for path in (directory / "jacobian").glob("*")}
    assert any("qp" in name.lower() for name in produced)
    assert any("qs" in name.lower() for name in produced)
