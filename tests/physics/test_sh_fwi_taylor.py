from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from tests.cases.sh_fwi_gradient import SHFWIGradientConfig
from tests.cases.sh_fwi_taylor import (
    generate_taylor_fwi_case,
    generate_taylor_observed_case,
)
from tests.utilities.fwi_gradient import (
    directional_derivative,
    gaussian_direction,
    l2_objective_from_reversed_residual_su,
    read_float_grid,
)
from tests.utilities.runner import executable_sha256, result_summary, run_denise
from tests.utilities.taylor import analyze_taylor_remainders


pytestmark = [pytest.mark.integration, pytest.mark.extended]

BASE_SHA = "91bbfdc772e4d1d7973428145e5c2aa005c419a8"
EPSILONS = (1.0e-2, 5.0e-3, 2.5e-3, 1.25e-3, 6.25e-4)


@dataclass(frozen=True)
class TaylorCase:
    name: str
    config: SHFWIGradientConfig
    vs_background: tuple[float, ...]
    rho_background: tuple[float, ...]
    target_vs: tuple[float, ...]
    target_rho: tuple[float, ...]
    p_vs: tuple[float, ...]
    p_rho: tuple[float, ...]
    delta_vs: tuple[float, ...]
    delta_rho: tuple[float, ...]
    active_vs: bool
    active_rho: bool
    holdout: bool = False


def _gaussian(
    config: SHFWIGradientConfig, *, x_m: float, y_m: float, sigma_m: float
) -> tuple[float, ...]:
    return tuple(
        gaussian_direction(
            nx=config.nx,
            ny=config.ny,
            dh_m=config.dh_m,
            center_x_m=x_m,
            center_y_m=y_m,
            sigma_m=sigma_m,
        )
    )


def _scaled_target(
    background: tuple[float, ...], direction: tuple[float, ...]
) -> tuple[float, ...]:
    return tuple(
        value * (1.0 + 0.02 * component)
        for value, component in zip(background, direction)
    )


def _zero(config: SHFWIGradientConfig) -> tuple[float, ...]:
    return (0.0,) * config.cell_count


def _cases() -> tuple[TaylorCase, ...]:
    homogeneous = SHFWIGradientConfig()
    homogeneous_vs = tuple(homogeneous.background_vs())
    homogeneous_rho = (homogeneous.density_kg_m3,) * homogeneous.cell_count
    target_520_400_70 = _gaussian(
        homogeneous, x_m=520.0, y_m=400.0, sigma_m=70.0
    )
    direction_560_440_80 = _gaussian(
        homogeneous, x_m=560.0, y_m=440.0, sigma_m=80.0
    )
    zero = _zero(homogeneous)

    case_a = TaylorCase(
        name="homogeneous_vs_only",
        config=homogeneous,
        vs_background=homogeneous_vs,
        rho_background=homogeneous_rho,
        target_vs=_scaled_target(homogeneous_vs, target_520_400_70),
        target_rho=homogeneous_rho,
        p_vs=direction_560_440_80,
        p_rho=zero,
        delta_vs=tuple(
            value * component
            for value, component in zip(homogeneous_vs, direction_560_440_80)
        ),
        delta_rho=zero,
        active_vs=True,
        active_rho=False,
    )
    case_b = TaylorCase(
        name="homogeneous_rho_only",
        config=homogeneous,
        vs_background=homogeneous_vs,
        rho_background=homogeneous_rho,
        target_vs=homogeneous_vs,
        target_rho=_scaled_target(homogeneous_rho, target_520_400_70),
        p_vs=zero,
        p_rho=direction_560_440_80,
        delta_vs=zero,
        delta_rho=tuple(
            value * component
            for value, component in zip(homogeneous_rho, direction_560_440_80)
        ),
        active_vs=False,
        active_rho=True,
    )

    joint_target_vs = _gaussian(
        homogeneous, x_m=520.0, y_m=400.0, sigma_m=70.0
    )
    joint_target_rho = _gaussian(
        homogeneous, x_m=610.0, y_m=480.0, sigma_m=65.0
    )
    joint_p_vs = _gaussian(
        homogeneous, x_m=560.0, y_m=430.0, sigma_m=80.0
    )
    joint_p_rho = _gaussian(
        homogeneous, x_m=600.0, y_m=500.0, sigma_m=60.0
    )
    case_c = TaylorCase(
        name="homogeneous_joint",
        config=homogeneous,
        vs_background=homogeneous_vs,
        rho_background=homogeneous_rho,
        target_vs=_scaled_target(homogeneous_vs, joint_target_vs),
        target_rho=_scaled_target(homogeneous_rho, joint_target_rho),
        p_vs=joint_p_vs,
        p_rho=joint_p_rho,
        delta_vs=tuple(
            value * component
            for value, component in zip(homogeneous_vs, joint_p_vs)
        ),
        delta_rho=tuple(
            value * component
            for value, component in zip(homogeneous_rho, joint_p_rho)
        ),
        active_vs=True,
        active_rho=True,
    )

    heterogeneous = replace(homogeneous, background_contrast_fraction=0.10)
    heterogeneous_vs = tuple(heterogeneous.background_vs())
    heterogeneous_rho = tuple(
        heterogeneous.density_kg_m3
        * (
            1.0
            + 0.1
            * math.sin(2.0 * math.pi * (ix - 0.5) / heterogeneous.nx)
            * math.sin(2.0 * math.pi * (iy - 0.5) / heterogeneous.ny)
        )
        for ix in range(1, heterogeneous.nx + 1)
        for iy in range(1, heterogeneous.ny + 1)
    )
    holdout_target_vs = _gaussian(
        heterogeneous, x_m=500.0, y_m=380.0, sigma_m=65.0
    )
    holdout_target_rho = _gaussian(
        heterogeneous, x_m=640.0, y_m=460.0, sigma_m=70.0
    )
    holdout_p_vs = _gaussian(
        heterogeneous, x_m=590.0, y_m=470.0, sigma_m=75.0
    )
    holdout_p_rho = _gaussian(
        heterogeneous, x_m=540.0, y_m=520.0, sigma_m=65.0
    )
    case_d = TaylorCase(
        name="heterogeneous_joint_holdout",
        config=heterogeneous,
        vs_background=heterogeneous_vs,
        rho_background=heterogeneous_rho,
        target_vs=_scaled_target(heterogeneous_vs, holdout_target_vs),
        target_rho=_scaled_target(heterogeneous_rho, holdout_target_rho),
        p_vs=holdout_p_vs,
        p_rho=holdout_p_rho,
        delta_vs=tuple(
            value * component
            for value, component in zip(heterogeneous_vs, holdout_p_vs)
        ),
        delta_rho=tuple(
            value * component
            for value, component in zip(heterogeneous_rho, holdout_p_rho)
        ),
        active_vs=True,
        active_rho=True,
        holdout=True,
    )
    return case_a, case_b, case_c, case_d


def _sequence_sha256(values: tuple[float, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_payload_sha256(
    path: Path, *, trace_count: int, samples_per_trace: int
) -> str:
    raw = path.read_bytes()
    trace_size = 240 + 4 * samples_per_trace
    assert len(raw) == trace_count * trace_size
    digest = hashlib.sha256()
    for trace in range(trace_count):
        start = trace * trace_size + 240
        digest.update(raw[start : start + 4 * samples_per_trace])
    return digest.hexdigest()


def _objective(directory: Path, config: SHFWIGradientConfig) -> float:
    return l2_objective_from_reversed_residual_su(
        directory / "su" / "synthetic_y.su.shot1.it1",
        len(config.receiver_x_m),
        round(config.time_s / config.dt_s),
    )


def _gradient(
    directory: Path, config: SHFWIGradientConfig, component: str
) -> list[float]:
    stored = read_float_grid(
        directory / "jacobian" / f"gradient_p_{component}.old",
        config.cell_count,
    )
    # descent() stores -dJ/dm. This is the sole convention conversion.
    gradient = [-value for value in stored]
    assert all(math.isfinite(value) for value in gradient)
    return gradient


def _run(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    config: SHFWIGradientConfig,
    role: str,
):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration=config.as_metadata() | {"role": role, "m5.2": True},
        timeout_seconds=120.0,
    )
    assert result.returncode == 0, result_summary(result)
    return result


def _direction_metadata(case: TaylorCase) -> dict[str, object]:
    return {
        "p_vs_sha256": _sequence_sha256(case.p_vs),
        "p_rho_sha256": _sequence_sha256(case.p_rho),
        "delta_vs_sha256": _sequence_sha256(case.delta_vs),
        "delta_rho_sha256": _sequence_sha256(case.delta_rho),
        "p_vs_max_abs": max(abs(value) for value in case.p_vs),
        "p_rho_max_abs": max(abs(value) for value in case.p_rho),
    }


def _write_report(repository_root: Path, report: dict[str, object]) -> None:
    path = repository_root / "tests" / "m5.2_sh_taylor_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _evaluate_case_form(
    *,
    root: Path,
    case: TaylorCase,
    grad_form: int,
    observed_su: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
) -> dict[str, object]:
    baseline_directories = []
    baseline_objectives = []
    baseline_payload_hashes = []
    returncodes = []
    for repeat in (1, 2):
        directory = root / f"form{grad_form}" / f"baseline_repeat{repeat}"
        generate_taylor_fwi_case(
            directory,
            config=case.config,
            observed_su=observed_su,
            vs_background=case.vs_background,
            rho_background=case.rho_background,
            delta_vs=case.delta_vs,
            delta_rho=case.delta_rho,
            epsilon=0.0,
            grad_form=grad_form,
            active_vs=case.active_vs,
            active_rho=case.active_rho,
        )
        result = _run(
            directory,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=case.config,
            role=f"m5.2_{case.name}_form{grad_form}_baseline_repeat{repeat}",
        )
        returncodes.append(result.returncode)
        baseline_directories.append(directory)
        baseline_objectives.append(_objective(directory, case.config))
        baseline_payload_hashes.append(
            _sample_payload_sha256(
                directory / "su" / "synthetic_y.su.shot1.it1",
                trace_count=len(case.config.receiver_x_m),
                samples_per_trace=round(case.config.time_s / case.config.dt_s),
            )
        )

    repeatability_difference = abs(baseline_objectives[0] - baseline_objectives[1])
    repeatability_tolerance = max(1.0e-18, 1.0e-12 * abs(baseline_objectives[0]))
    assert repeatability_difference <= repeatability_tolerance
    assert baseline_payload_hashes[0] == baseline_payload_hashes[1]

    gradient_directory = baseline_directories[0]
    g_vs = (
        _gradient(gradient_directory, case.config, "u") if case.active_vs else []
    )
    g_rho = (
        _gradient(gradient_directory, case.config, "rho") if case.active_rho else []
    )
    g_vs_dot = (
        directional_derivative(g_vs, case.delta_vs) if case.active_vs else 0.0
    )
    g_rho_dot = (
        directional_derivative(g_rho, case.delta_rho) if case.active_rho else 0.0
    )
    g_total = g_vs_dot + g_rho_dot
    predicted_largest_change = abs(EPSILONS[0] * g_total)
    if repeatability_difference == 0.0:
        assert g_total != 0.0, "DEGENERATE TAYLOR DIRECTION"
    else:
        assert predicted_largest_change >= 100.0 * repeatability_difference, (
            "DEGENERATE TAYLOR DIRECTION"
        )

    objectives = []
    for epsilon in EPSILONS:
        directory = root / f"form{grad_form}" / f"epsilon_{epsilon:.8f}"
        generate_taylor_fwi_case(
            directory,
            config=case.config,
            observed_su=observed_su,
            vs_background=case.vs_background,
            rho_background=case.rho_background,
            delta_vs=case.delta_vs,
            delta_rho=case.delta_rho,
            epsilon=epsilon,
            grad_form=grad_form,
            active_vs=case.active_vs,
            active_rho=case.active_rho,
        )
        result = _run(
            directory,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=case.config,
            role=f"m5.2_{case.name}_form{grad_form}_epsilon_{epsilon:.8f}",
        )
        returncodes.append(result.returncode)
        objectives.append(_objective(directory, case.config))

    analysis = analyze_taylor_remainders(
        epsilons=EPSILONS,
        objectives=objectives,
        baseline_objective=baseline_objectives[0],
        gradient_directional_product=g_total,
    )
    contribution_denominator = abs(g_vs_dot) + abs(g_rho_dot)
    result = {
        "case": case.name,
        "grad_form": grad_form,
        "holdout": case.holdout,
        "objective_definition": (
            "0.5*sum(samples^2) from reversed residual "
            "su/synthetic_y.su.shot1.it1"
        ),
        "baseline_model_hashes": {
            "current.vs": _file_sha256(
                gradient_directory / "model" / "current.vs"
            ),
            "current.rho": _file_sha256(
                gradient_directory / "model" / "current.rho"
            ),
        },
        "direction_hashes": _direction_metadata(case),
        "baseline_repeatability": {
            "j0_run1": baseline_objectives[0],
            "j0_run2": baseline_objectives[1],
            "absolute_difference": repeatability_difference,
            "acceptance_tolerance": repeatability_tolerance,
            "sample_payload_sha256": baseline_payload_hashes,
        },
        "gradient_contributions": {
            "g_vs_dot_delta_vs": g_vs_dot,
            "g_rho_dot_delta_rho": g_rho_dot,
            "g_total_dot_p": g_total,
            "vs_fractional_magnitude": (
                abs(g_vs_dot) / contribution_denominator
                if contribution_denominator
                else 0.0
            ),
            "rho_fractional_magnitude": (
                abs(g_rho_dot) / contribution_denominator
                if contribution_denominator
                else 0.0
            ),
            "largest_epsilon_prediction_magnitude": predicted_largest_change,
        },
        "analysis": analysis,
        "smallest_point_floor_diagnostic": {
            "epsilon": EPSILONS[-1],
            "r1": analysis["rows"][-1]["r1"],
            "q1_from_previous": analysis["pairwise_q1"][-1],
            "inside_acceptance_fit_window": False,
        },
        "returncodes": returncodes,
        "accepted": analysis["accepted"],
    }
    assert all(code == 0 for code in returncodes)
    return result


def test_m52_formal_elastic_sh_taylor_verification(
    tmp_path, repository_root, denise_binary, mpiexec
):
    actual_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.strip()
    assert actual_sha == BASE_SHA
    cases = _cases()
    report: dict[str, object] = {
        "milestone": "M5.2 formal elastic SH Taylor verification",
        "base_git_sha": BASE_SHA,
        "branch": "codex/m5.2-sh-taylor-verification",
        "denise_executable_sha256": executable_sha256(denise_binary.resolve()),
        "epsilon_ladder": list(EPSILONS),
        "fit_window": list(EPSILONS[:4]),
        "gradient_convention": (
            "gradient=-gradient_p_*.old; unweighted cell sum; no fitted scale"
        ),
        "predeclared_directions": {
            case.name: _direction_metadata(case) for case in cases
        },
        "results": [],
    }

    for case_index, case in enumerate(cases):
        if case.holdout:
            homogeneous_failures = [
                row for row in report["results"] if not row["accepted"]
            ]
            if homogeneous_failures:
                report["verdict"] = "HETEROGENEOUS HOLDOUT NOT RUN"
                _write_report(repository_root, report)
                pytest.fail(json.dumps(homogeneous_failures, indent=2))

        case_root = tmp_path / case.name
        observed_directory = case_root / "observed"
        generate_taylor_observed_case(
            observed_directory,
            config=case.config,
            target_vs=case.target_vs,
            target_rho=case.target_rho,
        )
        observed_result = _run(
            observed_directory,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=case.config,
            role=f"m5.2_{case.name}_observed",
        )
        observed_su = observed_directory / "su" / "synthetic_y.su.shot1"
        assert observed_result.returncode == 0
        assert observed_su.is_file() and observed_su.stat().st_size > 0

        for grad_form in (1, 2):
            row = _evaluate_case_form(
                root=case_root,
                case=case,
                grad_form=grad_form,
                observed_su=observed_su,
                repository_root=repository_root,
                denise_binary=denise_binary,
                mpiexec=mpiexec,
            )
            report["results"].append(row)
            _write_report(repository_root, report)

        case_failures = [
            row
            for row in report["results"]
            if row["case"] == case.name and not row["accepted"]
        ]
        if case_failures:
            report["verdict"] = "TAYLOR CASE FAILED"
            _write_report(repository_root, report)
            pytest.fail(json.dumps(case_failures, indent=2))

    report["verdict"] = "M5.2 SH TAYLOR VERIFIED"
    _write_report(repository_root, report)
    assert all(row["accepted"] for row in report["results"])
