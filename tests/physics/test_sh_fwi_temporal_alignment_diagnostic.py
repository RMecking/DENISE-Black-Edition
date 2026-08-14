from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
from pathlib import Path

import pytest

from tests.cases.sh_fwi_gradient import (
    SHFWIGradientConfig,
    generate_forward_observed_case,
    generate_fwi_case,
)
from tests.physics.test_sh_fwi_component_diagnostic import (
    _correlation,
    _exact_gradient,
    _fd_reference,
    _recover_correlation,
    _relative_l2,
    _run,
)
from tests.physics.test_sh_fwi_averaging_diagnostic import _objective
from tests.utilities.fwi_gradient import (
    directional_derivative,
    flat_top_direction,
    gaussian_direction,
    read_float_grid,
)
from tests.utilities.runner import executable_sha256, result_summary


pytestmark = [pytest.mark.integration, pytest.mark.extended]


MANDATORY_PRE_ABSOLUTE_RESIDUAL_MAX = 5.0e-5
MANDATORY_POST_ABSOLUTE_RESIDUAL_MIN = 1.0e-3
MANDATORY_FD_RELATIVE_CHANGE_MAX = 5.0e-5
HOLDOUT_PRE_ABSOLUTE_RESIDUAL_MAX = 5.0e-5


def _required_binary(temporal: str, component: str) -> Path:
    variable = f"M5E_{temporal.upper()}_{component.upper()}_BIN"
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"{variable} is required for M5.0e", pytrace=False)
    path = Path(value).resolve()
    assert path.is_file(), f"Missing diagnostic binary: {path}"
    return path


def _compiler_version() -> str:
    result = subprocess.run(
        ["mpicc", "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return result.stdout.strip()


def _directions(config: SHFWIGradientConfig) -> dict[str, list[float]]:
    return {
        "gaussian_sigma_20m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
            sigma_m=20.0,
        ),
        "gaussian_sigma_80m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
            sigma_m=80.0,
        ),
        "flat_top": flat_top_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
            half_width_x_m=120.0, half_width_y_m=100.0, taper_m=80.0,
        ),
    }


def test_sh_discrete_temporal_state_alignment(
    tmp_path, repository_root, mpiexec
):
    binaries = {
        temporal: {
            component: _required_binary(temporal, component)
            for component in ("xy", "x", "y")
        }
        for temporal in ("post", "pre")
    }
    config = SHFWIGradientConfig()
    observed_dir = tmp_path / "observed"
    generate_forward_observed_case(observed_dir, config=config)
    observed_run = _run(
        observed_dir,
        repository_root=repository_root,
        binary=binaries["post"]["xy"],
        mpiexec=mpiexec,
        config=config,
        role="m5e_observed_post_xy",
    )
    observed = observed_dir / "su" / "synthetic_y.su.shot1"
    assert observed.is_file() and observed.stat().st_size > 0, result_summary(observed_run)

    identities = []
    exact_gradients: dict[int, dict[str, list[float]]] = {}
    run_metadata = []
    # Form 2 is deliberately completed and validated before Form 1.
    for grad_form in (2, 1):
        exact_gradients[grad_form] = {}
        for temporal in ("post", "pre"):
            gradients = {}
            for component, binary in binaries[temporal].items():
                run_dir = tmp_path / f"form{grad_form}_{temporal}_{component}"
                generate_fwi_case(
                    run_dir,
                    observed_su=observed,
                    epsilon_fraction=0.0,
                    grad_form=grad_form,
                    config=config,
                )
                result = _run(
                    run_dir,
                    repository_root=repository_root,
                    binary=binary,
                    mpiexec=mpiexec,
                    config=config,
                    role=f"m5e_form_{grad_form}_{temporal}_{component}",
                    fwi=True,
                )
                stored = read_float_grid(
                    run_dir / "jacobian" / "gradient_p_u.old", config.cell_count
                )
                gradients[component] = [-value for value in stored]
                run_metadata.append(json.loads(result.metadata_path.read_text(encoding="utf-8")))

            correlations = {
                component: _recover_correlation(values, config, grad_form)
                for component, values in gradients.items()
            }
            component_sum = [
                x + y for x, y in zip(correlations["x"], correlations["y"])
            ]
            identity = {
                "grad_form": grad_form,
                "temporal_mode": temporal,
                "relative_l2": _relative_l2(correlations["xy"], component_sum),
                "normalized_correlation": _correlation(correlations["xy"], component_sum),
            }
            identities.append(identity)
            assert identity["relative_l2"] <= 2.0e-6, json.dumps(identity, indent=2)
            assert identity["normalized_correlation"] >= 0.999999999, json.dumps(identity, indent=2)
            exact_gradients[grad_form][temporal] = _exact_gradient(
                correlations["x"], correlations["y"], config, grad_form
            )

    rows = []
    for grad_form in (2, 1):
        for direction_name, direction in _directions(config).items():
            physical_direction = [
                vs * value for vs, value in zip(config.background_vs(), direction)
            ]
            products = {
                temporal: directional_derivative(
                    exact_gradients[grad_form][temporal], physical_direction
                )
                for temporal in ("post", "pre")
            }
            fd = _fd_reference(repository_root, direction_name, grad_form)
            derivative = fd["five_point_fine"]
            row = {
                "direction": direction_name,
                "grad_form": grad_form,
                "d_fd": derivative,
                "fd_diagnostics": fd,
                "post_exact_vjp_g_dot_p": products["post"],
                "pre_exact_vjp_g_dot_p": products["pre"],
                "k_post": derivative / products["post"],
                "k_pre": derivative / products["pre"],
                "post_absolute_residual": abs(derivative / products["post"] - 1.0),
                "pre_absolute_residual": abs(derivative / products["pre"] - 1.0),
                "theoretical_selection": "pre",
            }
            row["fd_stable"] = (
                fd["five_point_relative_change"]
                < MANDATORY_FD_RELATIVE_CHANGE_MAX
            )
            assert row["fd_stable"], json.dumps(row, indent=2)
            assert (
                row["pre_absolute_residual"]
                < MANDATORY_PRE_ABSOLUTE_RESIDUAL_MAX
            ), json.dumps(row, indent=2)
            assert (
                row["pre_absolute_residual"] < row["post_absolute_residual"]
            ), json.dumps(row, indent=2)
            assert (
                row["post_absolute_residual"]
                > MANDATORY_POST_ABSOLUTE_RESIDUAL_MIN
            ), json.dumps(row, indent=2)
            rows.append(row)

    pre_closes = all(
        row["pre_absolute_residual"] < MANDATORY_PRE_ABSOLUTE_RESIDUAL_MAX
        for row in rows
    )
    assert pre_closes
    holdout_prediction = (
        "For the previously unseen sigma=60 m Gaussian centered at "
        "(550 m, 520 m), PRE plus exact material VJP predicts K approximately "
        "one without any fitted scalar."
    )
    holdout_rows = []
    holdout_direction = gaussian_direction(
        nx=config.nx, ny=config.ny, dh_m=config.dh_m,
        center_x_m=550.0, center_y_m=520.0, sigma_m=60.0,
    )
    physical_holdout = [
        vs * value for vs, value in zip(config.background_vs(), holdout_direction)
    ]
    for grad_form in (2, 1):
        objectives = {}
        epsilon_rows = []
        for epsilon in (0.015, 0.0075, 0.00375):
            values = {"epsilon_fraction": epsilon}
            for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                run_dir = tmp_path / "holdout" / f"form{grad_form}_{side}_{epsilon:g}"
                generate_fwi_case(
                    run_dir,
                    observed_su=observed,
                    epsilon_fraction=sign * epsilon,
                    grad_form=grad_form,
                    config=config,
                    direction=holdout_direction,
                )
                result = _run(
                    run_dir,
                    repository_root=repository_root,
                    binary=binaries["post"]["xy"],
                    mpiexec=mpiexec,
                    config=config,
                    role=f"m5e_holdout_form_{grad_form}_{side}",
                    fwi=True,
                )
                objective = _objective(run_dir, config)
                objectives[sign * epsilon] = objective
                values[f"j_{side}"] = objective
                run_metadata.append(
                    json.loads(result.metadata_path.read_text(encoding="utf-8"))
                )
            epsilon_rows.append(values)

        def five_point(h: float) -> float:
            return (
                -objectives[2.0 * h]
                + 8.0 * objectives[h]
                - 8.0 * objectives[-h]
                + objectives[-2.0 * h]
            ) / (12.0 * h)

        coarse = five_point(0.0075)
        fine = five_point(0.00375)
        uncertainty = abs(fine - coarse) / max(abs(fine), abs(coarse))
        assert uncertainty < 5.0e-5
        exact_product = directional_derivative(
            exact_gradients[grad_form]["pre"], physical_holdout
        )
        holdout_row = {
            "grad_form": grad_form,
            "direction": {
                "family": "gaussian",
                "sigma_m": 60.0,
                "center_x_m": 550.0,
                "center_y_m": 520.0,
            },
            # This prediction is literal provenance: the hold-out does not fit or
            # update any calibration scalar.
            "prediction_recorded_before_evaluation": holdout_prediction,
            "epsilon_objectives": epsilon_rows,
            "five_point_coarse": coarse,
            "five_point_fine": fine,
            "five_point_relative_change": uncertainty,
            "pre_exact_vjp_g_dot_p": exact_product,
            "k_pre": fine / exact_product,
            "absolute_residual": abs(fine / exact_product - 1.0),
        }
        assert (
            holdout_row["absolute_residual"]
            < HOLDOUT_PRE_ABSOLUTE_RESIDUAL_MAX
        ), json.dumps(holdout_row, indent=2)
        holdout_rows.append(holdout_row)
    assert len(holdout_rows) == 2
    holdout_run = True
    assert holdout_run
    provenance = {
        temporal: {
            component: {
                "path": str(binary),
                "sha256": executable_sha256(binary),
                "temporal_mode": temporal,
                "component_mode": component,
            }
            for component, binary in components.items()
        }
        for temporal, components in binaries.items()
    }
    output = {
        "git_base_sha": "47f3ab93ae7b27433980dafeb77646c9f5a6940a",
        "reconstructed_patch_sha256": hashlib.sha256(
            (
                repository_root / "tests" / "m5e_sh_temporal_instrumentation.patch"
            ).read_bytes()
        ).hexdigest(),
        "compiler": "mpicc",
        "compiler_version": _compiler_version(),
        "prediction_recorded_before_runs": (
            "The exact material derivative requires the adjoint stress after reverse "
            "update_v and residual injection but before reverse update_s. PRE must "
            "outperform POST consistently without calibration."
        ),
        "forward_indexing": {
            "NT_5_DTINV_1": [
                {"backward_nt": value, "forward_nt": 6 - value, "stored_block": 6 - value}
                for value in range(1, 6)
            ],
            "NT_8_DTINV_3": [
                {"backward_nt": 2, "forward_nt": 7, "stored_block": 3},
                {"backward_nt": 5, "forward_nt": 4, "stored_block": 2},
                {"backward_nt": 8, "forward_nt": 1, "stored_block": 1},
            ],
            "conclusion": "current DTINV_help and imat reverse mapping is correct",
        },
        "residual_injection": {
            "location": "end of update_v_PML_SH(sw=1), before update_s_elastic_PML_SH",
            "degree_of_freedom": "native integer-grid pvz/vz receiver degree of freedom",
            "conclusion": "consistent with adding C^T r to adjoint velocity before V_n^T",
        },
        "component_identity": identities,
        "rows": rows,
        "pre_closes_all_mandatory_cases": pre_closes,
        "holdout_prediction": holdout_prediction,
        "holdout_run": holdout_run,
        "holdout_rows": holdout_rows,
        "holdout_reason": None,
        "binaries": provenance,
        "run_metadata": run_metadata,
    }
    output_path = Path(
        os.environ.get(
            "M5E_REPRO_DIAGNOSTICS_PATH",
            tmp_path / "m5e_temporal_alignment_reproduction.json",
        )
    ).resolve()
    historical_path = (
        repository_root / "tests" / "m5e_temporal_alignment_diagnostics.json"
    ).resolve()
    assert output_path != historical_path, "Historical M5.0e run data is immutable"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
