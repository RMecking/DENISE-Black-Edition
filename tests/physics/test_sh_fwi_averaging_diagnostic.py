from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from tests.cases.sh_fwi_gradient import (
    SHFWIGradientConfig,
    generate_forward_observed_case,
    generate_fwi_case,
)
from tests.utilities.fwi_gradient import (
    central_difference,
    directional_derivative,
    flat_top_direction,
    gaussian_direction,
    l2_objective_from_reversed_residual_su,
    read_float_grid,
)
from tests.utilities.runner import executable_sha256, result_summary, run_denise


pytestmark = [pytest.mark.integration, pytest.mark.extended]

EPSILONS = (0.015, 0.0075, 0.00375)


def _run(directory, *, repository_root, denise_binary, mpiexec, config, role, fwi=False):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration=config.as_metadata() | {"role": role, "nprocx": 1, "nprocy": 1},
        timeout_seconds=12.0 if fwi else 120.0,
    )
    if result.returncode != 0:
        residual = directory / "su" / "synthetic_y.su.shot1.it1"
        gradient = directory / "jacobian" / "gradient_p_u.old"
        assert fwi and residual.is_file() and gradient.is_file(), result_summary(result)
    return result


def _objective(directory: Path, config: SHFWIGradientConfig) -> float:
    return l2_objective_from_reversed_residual_su(
        directory / "su" / "synthetic_y.su.shot1.it1",
        len(config.receiver_x_m),
        round(config.time_s / config.dt_s),
    )


def _prepare_background(tmp_path, repository_root, denise_binary, mpiexec, config):
    observed_dir = tmp_path / "observed"
    generate_forward_observed_case(observed_dir, config=config)
    observed_run = _run(
        observed_dir,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role="m5c_observed_target",
    )
    observed = observed_dir / "su" / "synthetic_y.su.shot1"
    assert observed.is_file() and observed.stat().st_size > 0, result_summary(observed_run)

    gradients = {}
    for grad_form in (1, 2):
        center_dir = tmp_path / f"form{grad_form}_center"
        generate_fwi_case(
            center_dir,
            observed_su=observed,
            epsilon_fraction=0.0,
            grad_form=grad_form,
            config=config,
        )
        _run(
            center_dir,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=config,
            role=f"m5c_center_form_{grad_form}",
            fwi=True,
        )
        stored = read_float_grid(
            center_dir / "jacobian" / "gradient_p_u.old", config.cell_count
        )
        gradients[grad_form] = [-value for value in stored]
    return observed, gradients


def _gradient_products(
    gradient: Sequence[float], direction: Sequence[float], config: SHFWIGradientConfig,
    grad_form: int
) -> dict[str, float | None]:
    background = config.background_vs()
    physical_direction = [vs * value for vs, value in zip(background, direction)]
    raw = directional_derivative(gradient, physical_direction)
    if grad_form == 1:
        vs6_corrected = directional_derivative(
            [value / vs**6 for value, vs in zip(gradient, background)], physical_direction
        )
        state_corrected = directional_derivative(
            [
                value * config.dtinv / (config.density_kg_m3 * vs**6)
                for value, vs in zip(gradient, background)
            ],
            physical_direction,
        )
    else:
        vs6_corrected = None
        state_corrected = directional_derivative(
            [
                value
                * config.dtinv
                / (config.dt_s * config.density_kg_m3**2 * vs**2)
                for value, vs in zip(gradient, background)
            ],
            physical_direction,
        )
    return {
        "g_dot_p_raw": raw,
        "g_dot_p_vs_minus_6_only": vs6_corrected,
        "g_dot_p_m5b_corrected": state_corrected,
    }


def _direction_rows(
    tmp_path, repository_root, denise_binary, mpiexec, config, observed, gradients,
    directions: Sequence[tuple[str, Sequence[float], dict[str, object]]]
):
    rows = []
    for name, direction, description in directions:
        for grad_form in (1, 2):
            products = _gradient_products(gradients[grad_form], direction, config, grad_form)
            epsilon_rows = []
            for epsilon in EPSILONS:
                objectives = {}
                for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                    run_dir = tmp_path / name / f"form{grad_form}_{side}_{epsilon:g}"
                    generate_fwi_case(
                        run_dir,
                        observed_su=observed,
                        epsilon_fraction=sign * epsilon,
                        grad_form=grad_form,
                        config=config,
                        direction=direction,
                    )
                    _run(
                        run_dir,
                        repository_root=repository_root,
                        denise_binary=denise_binary,
                        mpiexec=mpiexec,
                        config=config,
                        role=f"m5c_{name}_form_{grad_form}",
                        fwi=True,
                    )
                    objectives[side] = _objective(run_dir, config)
                derivative = central_difference(objectives["plus"], objectives["minus"], epsilon)
                epsilon_rows.append(
                    {
                        "epsilon_fraction": epsilon,
                        "j_plus": objectives["plus"],
                        "j_minus": objectives["minus"],
                        "d_fd": derivative,
                        "relative_to_previous": None,
                    }
                )
            for previous, current in zip(epsilon_rows, epsilon_rows[1:]):
                current["relative_to_previous"] = abs(current["d_fd"] - previous["d_fd"]) / max(
                    abs(current["d_fd"]), abs(previous["d_fd"])
                )
            derivative = epsilon_rows[-1]["d_fd"]
            corrected = products["g_dot_p_m5b_corrected"]
            assert isinstance(corrected, float) and corrected != 0.0
            rows.append(
                {
                    "direction": name,
                    "description": description,
                    "grad_form": grad_form,
                    **products,
                    "d_fd_plateau": derivative,
                    "k_residual_fd_over_corrected_gradient": derivative / corrected,
                    "epsilons": epsilon_rows,
                }
            )
    assert all(
        row["epsilons"][-1]["relative_to_previous"] is not None
        and row["epsilons"][-1]["relative_to_previous"] <= 0.01
        for row in rows
    ), json.dumps(rows, indent=2)
    return rows


def _write_output(repository_root, denise_binary, config, prediction, rows, filename):
    output = {
        "git_sha": "47f3ab93ae7b27433980dafeb77646c9f5a6940a",
        "binary_path": str(denise_binary.resolve()),
        "binary_sha256": executable_sha256(denise_binary.resolve()),
        "prediction_recorded_before_runs": prediction,
        "m5b_correction": {
            "form1": "g_prod * DTINV / (rho * Vs^6), applied pointwise",
            "form2": "g_prod * DTINV / (DT * rho^2 * Vs^2), applied pointwise",
        },
        "configuration": config.as_metadata(),
        "rows": rows,
    }
    (repository_root / "tests" / filename).write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


def test_sh_fwi_averaging_gaussian_width_sweep(
    tmp_path, repository_root, denise_binary, mpiexec
):
    config = SHFWIGradientConfig()
    prediction = (
        "If the omitted staggered-material transpose dominates the residual, "
        "abs(K-1) will decrease as Gaussian sigma increases."
    )
    observed, gradients = _prepare_background(
        tmp_path, repository_root, denise_binary, mpiexec, config
    )
    directions = []
    for sigma_m in (20.0, 40.0, 80.0, 120.0):
        directions.append(
            (
                f"gaussian_sigma_{sigma_m:g}m",
                gaussian_direction(
                    nx=config.nx,
                    ny=config.ny,
                    dh_m=config.dh_m,
                    center_x_m=config.anomaly_x_m,
                    center_y_m=config.anomaly_y_m,
                    sigma_m=sigma_m,
                ),
                {"family": "gaussian", "sigma_m": sigma_m, "sigma_over_dh": sigma_m / config.dh_m},
            )
        )
    rows = _direction_rows(
        tmp_path, repository_root, denise_binary, mpiexec, config, observed, gradients, directions
    )
    _write_output(
        repository_root,
        denise_binary,
        config,
        prediction,
        rows,
        "m5c_gaussian_width_diagnostics.json",
    )


def test_sh_fwi_averaging_flat_top(
    tmp_path, repository_root, denise_binary, mpiexec
):
    config = SHFWIGradientConfig()
    prediction = (
        "A broad plateau should make local and neighbor-backprojected directional "
        "products more alike, moving K closer to one than the M5b sigma=70 m Gaussian."
    )
    observed, gradients = _prepare_background(
        tmp_path, repository_root, denise_binary, mpiexec, config
    )
    direction = flat_top_direction(
        nx=config.nx,
        ny=config.ny,
        dh_m=config.dh_m,
        center_x_m=config.anomaly_x_m,
        center_y_m=config.anomaly_y_m,
        half_width_x_m=120.0,
        half_width_y_m=100.0,
        taper_m=80.0,
    )
    rows = _direction_rows(
        tmp_path,
        repository_root,
        denise_binary,
        mpiexec,
        config,
        observed,
        gradients,
        [("flat_top", direction, {"family": "flat_top", "half_width_m": [120.0, 100.0], "taper_m": 80.0})],
    )
    _write_output(
        repository_root, denise_binary, config, prediction, rows, "m5c_flat_top_diagnostics.json"
    )


def test_sh_fwi_averaging_smooth_heterogeneous_background(
    tmp_path, repository_root, denise_binary, mpiexec
):
    config = replace(SHFWIGradientConfig(), background_contrast_fraction=0.20)
    prediction = (
        "If the exact harmonic-mean Jacobian is missing, a smooth 20% total Vs "
        "contrast should change K relative to the homogeneous sigma=80 m result, "
        "with a larger mismatch expected as neighboring derivative weights depart from one half."
    )
    observed, gradients = _prepare_background(
        tmp_path, repository_root, denise_binary, mpiexec, config
    )
    direction = gaussian_direction(
        nx=config.nx,
        ny=config.ny,
        dh_m=config.dh_m,
        center_x_m=config.anomaly_x_m,
        center_y_m=config.anomaly_y_m,
        sigma_m=80.0,
    )
    rows = _direction_rows(
        tmp_path,
        repository_root,
        denise_binary,
        mpiexec,
        config,
        observed,
        gradients,
        [("heterogeneous_gaussian_sigma_80m", direction, {"family": "gaussian", "sigma_m": 80.0})],
    )
    _write_output(
        repository_root,
        denise_binary,
        config,
        prediction,
        rows,
        "m5c_heterogeneous_diagnostics.json",
    )
