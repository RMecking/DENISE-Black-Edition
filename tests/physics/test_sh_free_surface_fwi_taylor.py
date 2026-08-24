from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from tests.cases.sh_free_surface_fwi import set_surface_case, surface_fwi_config
from tests.cases.sh_fwi_taylor import (
    generate_taylor_fwi_case,
    generate_taylor_observed_case,
)
from tests.physics.test_sh_fwi_production_gradient import (
    RHO_EPSILONS,
    VS_EPSILONS,
    _accept,
    _fd_metrics,
    _gradient,
)
from tests.physics.test_sh_fwi_taylor import EPSILONS, _objective
from tests.utilities.fwi_gradient import directional_derivative, gaussian_direction
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.taylor_derivative import analyze_taylor_derivative_consistency


pytestmark = [pytest.mark.integration, pytest.mark.extended]


@dataclass(frozen=True)
class SurfaceTaylorCase:
    name: str
    config: object
    vs_background: tuple[float, ...]
    rho_background: tuple[float, ...]
    target_vs: tuple[float, ...]
    target_rho: tuple[float, ...]
    delta_vs: tuple[float, ...]
    delta_rho: tuple[float, ...]
    active_vs: bool
    active_rho: bool
    holdout: bool = False


def _gaussian(config, *, x_m: float, y_m: float, sigma_m: float) -> tuple[float, ...]:
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


def _scaled(background, direction):
    return tuple(
        value * (1.0 + 0.02 * component)
        for value, component in zip(background, direction)
    )


def _cases() -> tuple[SurfaceTaylorCase, ...]:
    config = surface_fwi_config()
    vs = tuple(config.background_vs())
    rho = (config.density_kg_m3,) * config.cell_count
    zero = (0.0,) * config.cell_count
    target_vs_p = _gaussian(config, x_m=500.0, y_m=190.0, sigma_m=55.0)
    target_rho_p = _gaussian(config, x_m=610.0, y_m=230.0, sigma_m=55.0)
    p_vs = _gaussian(config, x_m=560.0, y_m=210.0, sigma_m=80.0)
    p_rho = _gaussian(config, x_m=530.0, y_m=180.0, sigma_m=70.0)
    delta_vs = tuple(value * component for value, component in zip(vs, p_vs))
    delta_rho = tuple(value * component for value, component in zip(rho, p_rho))

    homogeneous = (
        SurfaceTaylorCase(
            name="homogeneous_vs_only",
            config=config,
            vs_background=vs,
            rho_background=rho,
            target_vs=_scaled(vs, target_vs_p),
            target_rho=rho,
            delta_vs=delta_vs,
            delta_rho=zero,
            active_vs=True,
            active_rho=False,
        ),
        SurfaceTaylorCase(
            name="homogeneous_rho_only",
            config=config,
            vs_background=vs,
            rho_background=rho,
            target_vs=vs,
            target_rho=_scaled(rho, target_rho_p),
            delta_vs=zero,
            delta_rho=delta_rho,
            active_vs=False,
            active_rho=True,
        ),
        SurfaceTaylorCase(
            name="homogeneous_joint",
            config=config,
            vs_background=vs,
            rho_background=rho,
            target_vs=_scaled(vs, target_vs_p),
            target_rho=_scaled(rho, target_rho_p),
            delta_vs=delta_vs,
            delta_rho=delta_rho,
            active_vs=True,
            active_rho=True,
        ),
    )

    heterogeneous_config = replace(config, background_contrast_fraction=0.10)
    heterogeneous_vs = tuple(heterogeneous_config.background_vs())
    heterogeneous_rho = tuple(
        heterogeneous_config.density_kg_m3
        * (
            1.0
            + 0.10
            * math.sin(2.0 * math.pi * (ix - 0.5) / heterogeneous_config.nx)
            * math.sin(2.0 * math.pi * (iy - 0.5) / heterogeneous_config.ny)
        )
        for ix in range(1, heterogeneous_config.nx + 1)
        for iy in range(1, heterogeneous_config.ny + 1)
    )
    holdout_target_vs = _gaussian(
        heterogeneous_config, x_m=480.0, y_m=180.0, sigma_m=60.0
    )
    holdout_target_rho = _gaussian(
        heterogeneous_config, x_m=630.0, y_m=240.0, sigma_m=60.0
    )
    holdout_p_vs = _gaussian(
        heterogeneous_config, x_m=590.0, y_m=220.0, sigma_m=75.0
    )
    holdout_p_rho = _gaussian(
        heterogeneous_config, x_m=520.0, y_m=170.0, sigma_m=65.0
    )
    holdout = SurfaceTaylorCase(
        name="heterogeneous_joint_holdout",
        config=heterogeneous_config,
        vs_background=heterogeneous_vs,
        rho_background=heterogeneous_rho,
        target_vs=_scaled(heterogeneous_vs, holdout_target_vs),
        target_rho=_scaled(heterogeneous_rho, holdout_target_rho),
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
    return *homogeneous, holdout


def _run(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    config,
    role: str,
    free_surface: bool = True,
):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration=config.as_metadata()
        | {
            "role": role,
            "milestone": "M6.1e",
            "free_surface": int(free_surface),
        },
        timeout_seconds=120.0,
    )
    assert result.returncode == 0, result_summary(result)
    return result


def _evaluate(
    *,
    root: Path,
    case: SurfaceTaylorCase,
    grad_form: int,
    observed_su: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    free_surface: bool = True,
    require_same_direction_fd: bool = False,
) -> dict[str, object]:
    baseline_objectives = []
    baseline_directories = []
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
        set_surface_case(
            directory,
            free_surface=free_surface,
            role=(
                f"m61e_{case.name}_fs{int(free_surface)}_form{grad_form}"
                f"_baseline_repeat{repeat}"
            ),
        )
        result = _run(
            directory,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=case.config,
            role=(
                f"m61e_{case.name}_fs{int(free_surface)}_form{grad_form}"
                f"_baseline_repeat{repeat}"
            ),
            free_surface=free_surface,
        )
        returncodes.append(result.returncode)
        baseline_directories.append(directory)
        baseline_objectives.append(_objective(directory, case.config))
    repeatability_difference = abs(
        baseline_objectives[1] - baseline_objectives[0]
    )

    gradient_directory = baseline_directories[0]
    g_vs_dot = (
        directional_derivative(
            _gradient(gradient_directory, case.config, "u"), case.delta_vs
        )
        if case.active_vs
        else 0.0
    )
    g_rho_dot = (
        directional_derivative(
            _gradient(gradient_directory, case.config, "rho"), case.delta_rho
        )
        if case.active_rho
        else 0.0
    )
    g_total = g_vs_dot + g_rho_dot

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
        set_surface_case(
            directory,
            free_surface=free_surface,
            role=(
                f"m61e_{case.name}_fs{int(free_surface)}_form{grad_form}"
                f"_epsilon_{epsilon:.8f}"
            ),
        )
        result = _run(
            directory,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=case.config,
            role=(
                f"m61e_{case.name}_fs{int(free_surface)}_form{grad_form}"
                f"_epsilon_{epsilon:.8f}"
            ),
            free_surface=free_surface,
        )
        returncodes.append(result.returncode)
        objectives.append(_objective(directory, case.config))

    analysis = analyze_taylor_derivative_consistency(
        epsilons=EPSILONS,
        objectives=objectives,
        baseline_objective=baseline_objectives[0],
        gradient_directional_product=g_total,
        repeatability_difference=repeatability_difference,
    )
    same_direction_fd = None
    if require_same_direction_fd:
        assert case.active_vs != case.active_rho, (
            "Same-direction M5.1 FD pairing applies to parameter-only cases"
        )
        same_direction_fd = _same_case_fd(
            root / f"form{grad_form}" / "same_direction_fd",
            case=case,
            observed_su=observed_su,
            gradient_directory=gradient_directory,
            grad_form=grad_form,
            free_surface=free_surface,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
        )
    accepted = analysis["accepted"] and (
        same_direction_fd is None or same_direction_fd["accepted"]
    )
    row = {
        "case": case.name,
        "grad_form": grad_form,
        "holdout": case.holdout,
        "free_surface": int(free_surface),
        "j0": baseline_objectives[0],
        "baseline_repeat_j0": baseline_objectives[1],
        "repeatability_difference": repeatability_difference,
        "baseline_exactly_repeatable": repeatability_difference == 0.0,
        "g_vs_dot_delta_vs": g_vs_dot,
        "g_rho_dot_delta_rho": g_rho_dot,
        "g_total_dot_p": g_total,
        "analysis": analysis,
        "taylor_derivative_accepted": analysis["accepted"],
        "same_direction_fd": same_direction_fd,
        "same_direction_fd_accepted": (
            None if same_direction_fd is None else same_direction_fd["accepted"]
        ),
        "returncodes": returncodes,
        "accepted": accepted,
    }
    assert all(code == 0 for code in returncodes)
    print("M61E_TAYLOR_ROW " + json.dumps(row, sort_keys=True))
    return row


def _same_case_fd(
    root: Path,
    *,
    case: SurfaceTaylorCase,
    observed_su: Path,
    gradient_directory: Path,
    grad_form: int,
    free_surface: bool,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
) -> dict[str, object]:
    if case.active_vs == case.active_rho:
        raise ValueError("Same-direction FD requires exactly one active parameter")
    parameter = "vs" if case.active_vs else "rho"
    epsilons = VS_EPSILONS if case.active_vs else RHO_EPSILONS
    objectives = {}
    returncodes = []
    for epsilon in epsilons:
        for sign in (-1.0, 1.0):
            signed = sign * epsilon
            directory = root / f"{signed:+.7f}"
            generate_taylor_fwi_case(
                directory,
                config=case.config,
                observed_su=observed_su,
                vs_background=case.vs_background,
                rho_background=case.rho_background,
                delta_vs=case.delta_vs,
                delta_rho=case.delta_rho,
                epsilon=signed,
                grad_form=grad_form,
                active_vs=case.active_vs,
                active_rho=case.active_rho,
            )
            role = (
                f"m61e_exact_{parameter}_gf{grad_form}"
                f"_fs{int(free_surface)}_fd_{signed:+.7f}"
            )
            set_surface_case(directory, free_surface=free_surface, role=role)
            result = _run(
                directory,
                repository_root=repository_root,
                denise_binary=denise_binary,
                mpiexec=mpiexec,
                config=case.config,
                role=role,
                free_surface=free_surface,
            )
            returncodes.append(result.returncode)
            objectives[signed] = _objective(directory, case.config)
    fd = _fd_metrics(objectives, epsilons)
    product = directional_derivative(
        _gradient(gradient_directory, case.config, "u" if case.active_vs else "rho"),
        case.delta_vs if case.active_vs else case.delta_rho,
    )
    result = {
        "parameter": parameter,
        "grad_form": grad_form,
        "free_surface": int(free_surface),
        "fd_diagnostics": fd,
        "returncodes": returncodes,
    }
    result.update(_accept(fd, product))
    print("M61E_EXACT_CASE_FD " + json.dumps(result, sort_keys=True))
    return result


def test_03_formal_free_surface_taylor_gate(
    tmp_path, repository_root, denise_binary, mpiexec
):
    report = {
        "epsilon_ladder": list(EPSILONS),
        "gradient_convention": "gradient=-gradient_p_*.old; unweighted cell sum; no fitted scale",
        "results": [],
    }
    for case in _cases():
        if case.holdout:
            homogeneous = [row for row in report["results"] if not row["holdout"]]
            assert len(homogeneous) == 6
            assert all(row["accepted"] for row in homogeneous), homogeneous

        root = tmp_path / case.name
        observed_directory = root / "observed"
        generate_taylor_observed_case(
            observed_directory,
            config=case.config,
            target_vs=case.target_vs,
            target_rho=case.target_rho,
        )
        set_surface_case(
            observed_directory,
            free_surface=True,
            role=f"m61e_{case.name}_observed",
        )
        _run(
            observed_directory,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=case.config,
            role=f"m61e_{case.name}_observed",
        )
        observed_su = observed_directory / "su" / "synthetic_y.su.shot1"
        assert observed_su.is_file() and observed_su.stat().st_size > 0

        for grad_form in (1, 2):
            row = _evaluate(
                root=root,
                case=case,
                grad_form=grad_form,
                observed_su=observed_su,
                repository_root=repository_root,
                denise_binary=denise_binary,
                mpiexec=mpiexec,
                require_same_direction_fd=(case.active_vs != case.active_rho),
            )
            report["results"].append(row)
            assert row["accepted"], row

    assert len(report["results"]) == 8
    assert sum(not row["holdout"] for row in report["results"]) == 6
    assert sum(row["holdout"] for row in report["results"]) == 2
    print("M61E_TAYLOR " + json.dumps(report, sort_keys=True))
