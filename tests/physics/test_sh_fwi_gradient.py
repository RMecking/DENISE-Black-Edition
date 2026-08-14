from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.cases.sh_fwi_gradient import (
    SHFWIGradientConfig,
    generate_forward_observed_case,
    generate_fwi_case,
)
from tests.utilities.fwi_gradient import (
    central_difference,
    directional_derivative,
    parse_initial_objective,
    read_float_grid,
)
from tests.utilities.runner import result_summary, run_denise


pytestmark = pytest.mark.integration


def _run(directory, *, repository_root, denise_binary, mpiexec, config, role):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration=config.as_metadata() | {"role": role, "nprocx": 1, "nprocy": 1},
        timeout_seconds=120.0,
    )
    assert result.returncode == 0, result_summary(result)
    return result


@pytest.mark.parametrize("grad_form", [1, 2])
def test_sh_vs_gradient_matches_central_finite_difference(
    tmp_path, repository_root, denise_binary, mpiexec, grad_form
):
    config = SHFWIGradientConfig()
    observed_dir = tmp_path / "observed"
    generate_forward_observed_case(observed_dir, config=config)
    observed_result = _run(
        observed_dir,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role="observed_target",
    )
    observed = observed_dir / "su" / "synthetic_y.su.shot1"
    assert observed.is_file() and observed.stat().st_size > 0, result_summary(observed_result)

    center_dir = tmp_path / f"form{grad_form}_center"
    generate_fwi_case(
        center_dir, observed_su=observed, epsilon_fraction=0.0, grad_form=grad_form, config=config
    )
    center_result = _run(
        center_dir,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role=f"gradient_form_{grad_form}",
    )
    center_stdout = center_result.stdout_path.read_text(encoding="utf-8", errors="replace")
    center_objective = parse_initial_objective(center_stdout)
    assert math.isfinite(center_objective) and center_objective > 0.0

    # store_PCG_SH writes gradp_u=-dJ/dVs before the PCG recursion. With every
    # optional conditioning switch off, negating this file recovers the pristine
    # post-chain-rule derivative produced by ass_gradSH.
    stored_direction = read_float_grid(
        center_dir / "jacobian" / "gradient_p_u.old", config.cell_count
    )
    gradient = [-value for value in stored_direction]
    physical_direction = [config.vs_m_s * value for value in config.direction()]
    gradient_dot_p = directional_derivative(gradient, physical_direction)
    assert math.isfinite(gradient_dot_p) and gradient_dot_p != 0.0

    rows = []
    # Stay away from +40 m/s, which is the exact target and therefore has a
    # zero gradient that makes DENISE's unrelated line-search normalization
    # divide by zero after the objective of interest has already been logged.
    for epsilon in (0.015, 0.0075, 0.00375):
        objectives = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            run_dir = tmp_path / f"form{grad_form}_{label}_{epsilon:g}"
            generate_fwi_case(
                run_dir,
                observed_su=observed,
                epsilon_fraction=sign * epsilon,
                grad_form=grad_form,
                config=config,
            )
            result = _run(
                run_dir,
                repository_root=repository_root,
                denise_binary=denise_binary,
                mpiexec=mpiexec,
                config=config,
                role=f"finite_difference_form_{grad_form}",
            )
            objectives[label] = parse_initial_objective(
                result.stdout_path.read_text(encoding="utf-8", errors="replace")
            )
        fd = central_difference(objectives["plus"], objectives["minus"], epsilon)
        ratio = fd / gradient_dot_p
        relative_error = abs(fd - gradient_dot_p) / max(abs(fd), abs(gradient_dot_p))
        rows.append(
            {
                "epsilon_m_s": epsilon,
                "j_plus": objectives["plus"],
                "j_minus": objectives["minus"],
                "central_fd": fd,
                "gradient_dot_p": gradient_dot_p,
                "ratio_fd_over_gradient": ratio,
                "relative_error": relative_error,
            }
        )

    resolved = [row for row in rows if row["relative_error"] <= 0.08]
    metrics = {
        "grad_form": grad_form,
        "objective_at_background": center_objective,
        "model_inner_product": "sum_ij g_ij p_ij (no extra DH^2)",
        "direction_normalization": "delta_Vs=Vs_background*p; epsilon is dimensionless",
        "rows": rows,
    }
    (tmp_path / f"gradient_form_{grad_form}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert resolved, json.dumps(metrics, indent=2)
    best = min(rows, key=lambda row: row["relative_error"])
    assert best["ratio_fd_over_gradient"] > 0.0, json.dumps(metrics, indent=2)

    repeat_dir = tmp_path / f"form{grad_form}_repeat"
    generate_fwi_case(
        repeat_dir, observed_su=observed, epsilon_fraction=0.0, grad_form=grad_form, config=config
    )
    repeat = _run(
        repeat_dir,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role=f"repeat_form_{grad_form}",
    )
    repeat_objective = parse_initial_objective(
        repeat.stdout_path.read_text(encoding="utf-8", errors="replace")
    )
    assert repeat_objective == center_objective
    assert read_float_grid(
        repeat_dir / "jacobian" / "gradient_p_u.old", config.cell_count
    ) == stored_direction
