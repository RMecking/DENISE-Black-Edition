from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.cases.visco_sh_free_surface import runtime_scenario
from tests.physics.test_visco_sh_free_surface import _run
from tests.utilities.sh_free_surface_runtime import (
    denise_ricker_reference,
    post_source_quarters,
)
from tests.utilities.visco_sh_free_surface_runtime import STABILITY_Q4_TO_Q1_MAX


pytestmark = [pytest.mark.integration, pytest.mark.extended]


def _scenario():
    scenario = runtime_scenario(fd_order=12, qs=50.0)
    return replace(
        scenario,
        candidate=replace(scenario.candidate, time_s=2.6005),
    )


def _quarter_metrics(run: dict[str, object]) -> dict[str, object]:
    config = run["case"]["numerics"]
    source = denise_ricker_reference(
        nt=round(config["time_s"] / config["dt_s"]),
        dt_s=config["dt_s"],
        frequency_hz=config["source_frequency_hz"],
        amplitude=1.0,
        timeshift_s=0.0,
        quellart=1,
        n_order=0,
    )
    quarters = post_source_quarters(
        nt=round(config["time_s"] / config["dt_s"]), n_off=source.n_off
    )
    series = run["diagnostic"]["max_abs_vz_series"]

    def maximum(bounds: tuple[int, int]) -> float:
        first, last = bounds
        return max(series[first - 1 : last])

    maxima = [maximum(bounds) for bounds in quarters.inclusive_bounds]
    return {
        "source_definition": {"quellart": 1, "n_order": 0},
        "n_off": source.n_off,
        "nominal_t_off_s": source.n_off * config["dt_s"],
        "quarter_size": quarters.quarter_size,
        "quarter_bounds": quarters.inclusive_bounds,
        "quarter_max_abs_vz": maxima,
        "q4_to_q1": maxima[3] / maxima[0],
    }


def test_m62b_finite_q_fd12_fixed_quarter_stability(
    tmp_path, repository_root, mpiexec
):
    instrumented = os.environ.get("M62B_INSTRUMENTED_DENISE")
    assert instrumented, "M62B_INSTRUMENTED_DENISE is required"
    results = {}
    for role in ("absorbing", "candidate"):
        run = _run(
            tmp_path / role,
            repository_root=repository_root,
            denise_binary=Path(instrumented),
            mpiexec=mpiexec,
            scenario=_scenario(),
            role=role,
            diagnostic=True,
            retain_diagnostic_series=True,
        )
        metrics = _quarter_metrics(run)
        assert metrics["n_off"] == 1257
        assert metrics["quarter_bounds"] == (
            (1258, 2243),
            (2244, 3229),
            (3230, 4215),
            (4216, 5201),
        )
        assert metrics["q4_to_q1"] <= STABILITY_Q4_TO_Q1_MAX
        results[role] = {
            **metrics,
            "returncode": run["run"]["returncode"],
            "executable_sha256": run["run"]["executable"]["sha256"],
        }
    report = {
        "acceptance": {
            "metric": "fixed post-source quarter max_abs_vz Q4/Q1",
            "q4_to_q1_max": STABILITY_Q4_TO_Q1_MAX,
            "frozen_from_candidate": False,
            "calibration_role": "absorbing",
        },
        "results": results,
    }
    (tmp_path / "m6.2b_stability_live_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("M62B_STABILITY " + json.dumps(report, sort_keys=True))
