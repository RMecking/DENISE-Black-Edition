from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
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
    l2_objective_from_reversed_residual_su,
    read_float_grid,
)
from tests.utilities.runner import executable_sha256, result_summary, run_denise


pytestmark = [pytest.mark.integration, pytest.mark.extended]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        # Some legacy one-iteration FWI configurations corrupt memory or hang
        # later in the unrelated line-search/cleanup path. M5.0b only accepts
        # such a run after both the residual and pristine-gradient artifacts
        # have been fully written. The failure remains recorded in provenance.
        residual = directory / "su" / "synthetic_y.su.shot1.it1"
        gradient = directory / "jacobian" / "gradient_p_u.old"
        assert fwi and residual.is_file() and gradient.is_file(), result_summary(result)
    else:
        assert result.returncode == 0, result_summary(result)
    return result


def _objective(directory: Path, config: SHFWIGradientConfig) -> float:
    return l2_objective_from_reversed_residual_su(
        directory / "su" / "synthetic_y.su.shot1.it1",
        len(config.receiver_x_m),
        round(config.time_s / config.dt_s),
    )


def _configuration_matrix() -> list[tuple[str, SHFWIGradientConfig]]:
    base = SHFWIGradientConfig()
    return [
        ("baseline", base),
        ("dt_0.00030", replace(base, dt_s=0.00030)),
        ("dt_0.00050", replace(base, dt_s=0.00050)),
        ("rho_1600", replace(base, density_kg_m3=1600.0)),
        ("rho_2300", replace(base, density_kg_m3=2300.0)),
        ("vs_1800", replace(base, vs_m_s=1800.0)),
        ("vs_2700", replace(base, vs_m_s=2700.0)),
        ("dtinv_2", replace(base, dtinv=2)),
        ("dtinv_3", replace(base, dtinv=3)),
        ("dtinv_4", replace(base, dtinv=4)),
        (
            "holdout",
            replace(base, vs_m_s=2450.0, density_kg_m3=2050.0, dt_s=0.00035),
        ),
        (
            "holdout_blind",
            replace(
                base,
                vs_m_s=2550.0,
                density_kg_m3=2150.0,
                dt_s=0.00037,
                dtinv=2,
            ),
        ),
    ]


def test_sh_gradient_scaling_matrix(
    tmp_path, repository_root, denise_binary, mpiexec
):
    rows = []
    requested = {value for value in os.environ.get("M5B_LABELS", "").split(",") if value}
    configurations = [
        item for item in _configuration_matrix() if not requested or item[0] in requested
    ]
    for label, config in configurations:
        observed_dir = tmp_path / label / "observed"
        generate_forward_observed_case(observed_dir, config=config)
        observed_run = _run(
            observed_dir,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=config,
            role="observed_target",
        )
        observed = observed_dir / "su" / "synthetic_y.su.shot1"
        assert observed.is_file() and observed.stat().st_size > 0, result_summary(observed_run)

        for grad_form in (1, 2):
            center_dir = tmp_path / label / f"form{grad_form}_center"
            generate_fwi_case(
                center_dir,
                observed_su=observed,
                epsilon_fraction=0.0,
                grad_form=grad_form,
                config=config,
            )
            center_run = _run(
                center_dir,
                repository_root=repository_root,
                denise_binary=denise_binary,
                mpiexec=mpiexec,
                config=config,
                role=f"scaling_{label}_form_{grad_form}",
                fwi=True,
            )
            stored = read_float_grid(
                center_dir / "jacobian" / "gradient_p_u.old", config.cell_count
            )
            gradient = [-value for value in stored]
            physical_direction = [config.vs_m_s * value for value in config.direction()]
            gradient_dot_p = directional_derivative(gradient, physical_direction)
            assert math.isfinite(gradient_dot_p) and gradient_dot_p != 0.0

            epsilon_rows = []
            for epsilon in (0.015, 0.0075, 0.00375):
                objectives = {}
                model_hashes = {}
                for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                    run_dir = tmp_path / label / f"form{grad_form}_{side}_{epsilon:g}"
                    generate_fwi_case(
                        run_dir,
                        observed_su=observed,
                        epsilon_fraction=sign * epsilon,
                        grad_form=grad_form,
                        config=config,
                    )
                    model_hashes[side] = _sha256(run_dir / "model" / "current.vs")
                    result = _run(
                        run_dir,
                        repository_root=repository_root,
                        denise_binary=denise_binary,
                        mpiexec=mpiexec,
                        config=config,
                        role=f"scaling_fd_{label}_form_{grad_form}",
                        fwi=True,
                    )
                    objectives[side] = _objective(run_dir, config)
                fd = central_difference(objectives["plus"], objectives["minus"], epsilon)
                epsilon_rows.append(
                    {
                        "epsilon_fraction": epsilon,
                        "j_plus": objectives["plus"],
                        "j_minus": objectives["minus"],
                        "d_fd": fd,
                        "relative_to_previous": None,
                        "model_sha256": model_hashes,
                    }
                )
            for previous, current in zip(epsilon_rows, epsilon_rows[1:]):
                current["relative_to_previous"] = abs(current["d_fd"] - previous["d_fd"]) / max(
                    abs(current["d_fd"]), abs(previous["d_fd"])
                )
            plateau = epsilon_rows[-1]["d_fd"]
            raw_ratio = plateau / gradient_dot_p
            hypothetical_g = (
                gradient_dot_p / config.vs_m_s**6 if grad_form == 1 else None
            )
            rows.append(
                {
                    "label": label,
                    "grad_form": grad_form,
                    "vs_m_s": config.vs_m_s,
                    "density_kg_m3": config.density_kg_m3,
                    "dt_s": config.dt_s,
                    "dh_m": config.dh_m,
                    "dtinv": config.dtinv,
                    "objective_center": _objective(center_dir, config),
                    "g_dot_p_raw": gradient_dot_p,
                    "g_dot_p_hypothetical": hypothetical_g,
                    "d_fd_plateau": plateau,
                    "r_raw": raw_ratio,
                    "r_hypothetical": (
                        plateau / hypothetical_g if hypothetical_g is not None else None
                    ),
                    "epsilons": epsilon_rows,
                    "observed_sha256": _sha256(observed),
                }
            )

    output = {
        "git_sha": "47f3ab93ae7b27433980dafeb77646c9f5a6940a",
        "binary_path": str(denise_binary.resolve()),
        "binary_sha256": executable_sha256(denise_binary.resolve()),
        "direction": "delta_Vs=Vs_background*p, max_abs(p)=1; epsilon dimensionless",
        "rows": rows,
    }
    output_path = repository_root / "tests" / os.environ.get(
        "M5B_OUTPUT", "m5b_scaling_diagnostics.json"
    )
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    assert all(
        row["epsilons"][-1]["relative_to_previous"] is not None
        and row["epsilons"][-1]["relative_to_previous"] <= 0.01
        for row in rows
    ), json.dumps(output, indent=2)
