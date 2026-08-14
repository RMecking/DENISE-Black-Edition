from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from tests.cases.sh_fwi_density import (
    generate_density_case,
    generate_density_observed_case,
)
from tests.cases.sh_fwi_gradient import SHFWIGradientConfig
from tests.physics.test_sh_fwi_component_diagnostic import (
    _correlation,
    _relative_l2,
    _run,
    _rows_to_xmajor,
    _xmajor_to_rows,
)
from tests.physics.test_sh_fwi_averaging_diagnostic import _objective
from tests.utilities.fwi_gradient import (
    directional_derivative,
    compliance_average_vjp,
    flat_top_direction,
    gaussian_direction,
    harmonic_mean,
    harmonic_mean_vjp,
    read_float_grid,
    read_su_float_samples,
)


pytestmark = [pytest.mark.integration, pytest.mark.extended]

EPSILONS = (0.0075, 0.00375, 0.001875)
PRE_RECORDED_PREDICTION = (
    "The exact inverse-density term uses B_VELOCITY: after reverse update_v "
    "and receiver injection, before reverse update_s. No fitted rho, DT, or DH "
    "factor is permitted."
)


def _binary(name: str) -> Path:
    variable = f"M5F_{name.upper()}_BIN"
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"{variable} is required for M5.0f", pytrace=False)
    path = Path(value).resolve()
    assert path.is_file(), path
    return path


def _directions(config: SHFWIGradientConfig):
    return {
        "gaussian_sigma_25m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
            sigma_m=25.0,
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


def _five_point(objectives: dict[float, float], h: float) -> float:
    return (
        -objectives[2.0 * h]
        + 8.0 * objectives[h]
        - 8.0 * objectives[-h]
        + objectives[-2.0 * h]
    ) / (12.0 * h)


def _exact_density_material_gradient(
    x_values, y_values, config, grad_form, density_values
):
    x_full = _xmajor_to_rows(x_values, config.nx, config.ny)
    y_full = _xmajor_to_rows(y_values, config.nx, config.ny)
    x_native = [row[:-1] for row in x_full]
    y_native = y_full[:-1]
    density = _xmajor_to_rows(density_values, config.nx, config.ny)
    vs = _xmajor_to_rows(config.background_vs(), config.nx, config.ny)
    if grad_form == 2:
        mu = [
            [density[j][i] * vs[j][i] ** 2 for i in range(config.nx)]
            for j in range(config.ny)
        ]
        mu_x, mu_y = harmonic_mean(mu)
        gmu_x = [
            [
                config.dtinv * x_native[j][i]
                / (config.dt_s * mu_x[j][i])
                for i in range(config.nx - 1)
            ]
            for j in range(config.ny)
        ]
        gmu_y = [
            [
                config.dtinv * y_native[j][i]
                / (config.dt_s * mu_y[j][i])
                for i in range(config.nx)
            ]
            for j in range(config.ny - 1)
        ]
        gmu = harmonic_mean_vjp(mu, gmu_x, gmu_y)
        result = [
            [vs[j][i] ** 2 * gmu[j][i] for i in range(config.nx)]
            for j in range(config.ny)
        ]
    else:
        gc_x = [
            [config.dtinv * value for value in row] for row in x_native
        ]
        gc_y = [
            [config.dtinv * value for value in row] for row in y_native
        ]
        gc = compliance_average_vjp((config.ny, config.nx), gc_x, gc_y)
        result = [
            [
                gc[j][i] / (density[j][i] ** 2 * vs[j][i] ** 2)
                for i in range(config.nx)
            ]
            for j in range(config.ny)
        ]
    return _rows_to_xmajor(result)


def _run_case(directory, repository_root, binary, mpiexec, config, role, fwi):
    return _run(
        directory,
        repository_root=repository_root,
        binary=binary,
        mpiexec=mpiexec,
        config=config,
        role=role,
        fwi=fwi,
    )


def test_exact_discrete_sh_density_gradient(
    tmp_path, repository_root, mpiexec
):
    binaries = {
        name: _binary(name)
        for name in (
            "legacy_post_xy",
            "legacy_b_xy",
            "exact_metric_legacy_xy",
            "exact_b_pre_xy",
            "exact_b_integrated_pre_xy",
            "exact_b_pre_x",
            "exact_b_pre_y",
        )
    }
    config = SHFWIGradientConfig()
    directions = _directions(config)
    target_direction = gaussian_direction(
        nx=config.nx, ny=config.ny, dh_m=config.dh_m,
        center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
        sigma_m=70.0,
    )

    observed_dir = tmp_path / "observed"
    generate_density_observed_case(
        observed_dir, config=config, direction=target_direction
    )
    observed_run = _run_case(
        observed_dir, repository_root, binaries["legacy_post_xy"], mpiexec,
        config, "m5f_observed_density_target", False,
    )
    observed = observed_dir / "su" / "synthetic_y.su.shot1"
    assert observed.is_file() and observed.stat().st_size > 0

    baseline_samples = {}
    for parameterization in ("T", "R", "M"):
        directory = tmp_path / "baseline" / parameterization
        generate_density_case(
            directory,
            config=config,
            parameterization=parameterization,
            epsilon_fraction=0.0,
            direction=target_direction,
            grad_form=2,
            mode=0,
        )
        _run_case(
            directory, repository_root, binaries["legacy_post_xy"], mpiexec,
            config, f"m5f_baseline_{parameterization}", False,
        )
        baseline_samples[parameterization] = read_su_float_samples(
            directory / "su" / "synthetic_y.su.shot1",
            len(config.receiver_x_m),
            round(config.time_s / config.dt_s),
        )
    baseline_mismatch = {
        parameterization: _relative_l2(
            baseline_samples["T"], baseline_samples[parameterization]
        )
        for parameterization in ("R", "M")
    }
    assert max(baseline_mismatch.values()) <= 1.0e-7, baseline_mismatch

    rows = []
    component_identity = []
    legacy_comparison = []
    exact_gradients = {}
    for grad_form in (2, 1):
        center_r = tmp_path / "gradients" / f"form{grad_form}_r_exact"
        generate_density_case(
            center_r, config=config, parameterization="R", epsilon_fraction=0.0,
            direction=target_direction, grad_form=grad_form, mode=1,
            observed_su=observed,
        )
        kinetic_binary = (
            binaries["exact_b_pre_xy"] if grad_form == 2
            else binaries["exact_b_integrated_pre_xy"]
        )
        _run_case(
            center_r, repository_root, kinetic_binary, mpiexec,
            config, f"m5f_form{grad_form}_r_exact", True,
        )
        exact_r = [
            -config.dtinv * value for value in read_float_grid(
                center_r / "jacobian" / "gradient_p_rho.old", config.cell_count
            )
        ]

        temporal_gradients = {}
        for name in ("legacy_post_xy", "legacy_b_xy", "exact_metric_legacy_xy"):
            directory = tmp_path / "gradients" / f"form{grad_form}_{name}"
            generate_density_case(
                directory, config=config, parameterization="R",
                epsilon_fraction=0.0, direction=target_direction,
                grad_form=grad_form, mode=1, observed_su=observed,
            )
            _run_case(
                directory, repository_root, binaries[name], mpiexec, config,
                f"m5f_form{grad_form}_{name}", True,
            )
            temporal_gradients[name] = [
                -value for value in read_float_grid(
                    directory / "jacobian" / "gradient_p_rho.old",
                    config.cell_count,
                )
            ]

        material_components = {}
        for component in ("xy", "x", "y"):
            directory = tmp_path / "gradients" / f"form{grad_form}_material_{component}"
            generate_density_case(
                directory, config=config, parameterization="M",
                epsilon_fraction=0.0, direction=target_direction,
                grad_form=grad_form, mode=1, observed_su=observed,
            )
            _run_case(
                directory, repository_root,
                binaries[f"exact_b_pre_{component}"], mpiexec, config,
                f"m5f_form{grad_form}_material_{component}", True,
            )
            material_components[component] = [
                -value for value in read_float_grid(
                    directory / "jacobian" / "gradient_p_u.old",
                    config.cell_count,
                )
            ]

        correlations = material_components
        component_sum = [
            x + y for x, y in zip(correlations["x"], correlations["y"])
        ]
        identity = {
            "grad_form": grad_form,
            "relative_l2": _relative_l2(correlations["xy"], component_sum),
            "normalized_correlation": _correlation(
                correlations["xy"], component_sum
            ),
        }
        assert identity["relative_l2"] <= 2.0e-6, identity
        assert identity["normalized_correlation"] >= 0.999999999, identity
        component_identity.append(identity)
        exact_m = _exact_density_material_gradient(
            correlations["x"], correlations["y"], config, grad_form,
            [config.density_kg_m3] * config.cell_count,
        )
        exact_t = [left + right for left, right in zip(exact_r, exact_m)]
        exact_gradients[grad_form] = {"R": exact_r, "M": exact_m, "T": exact_t}

        for direction_name, direction in directions.items():
            physical_direction = [
                config.density_kg_m3 * value for value in direction
            ]
            fd = {}
            fd_stability = {}
            for parameterization in ("R", "M", "T"):
                objectives = {}
                for epsilon in EPSILONS:
                    for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                        signed = sign * epsilon
                        directory = (
                            tmp_path / "fd" / f"form{grad_form}" / direction_name
                            / parameterization / f"{side}_{epsilon:g}"
                        )
                        generate_density_case(
                            directory, config=config,
                            parameterization=parameterization,
                            epsilon_fraction=signed, direction=direction,
                            grad_form=grad_form, mode=1, observed_su=observed,
                        )
                        _run_case(
                            directory, repository_root,
                            binaries["legacy_post_xy"], mpiexec, config,
                            f"m5f_fd_{parameterization}_{direction_name}_form{grad_form}",
                            True,
                        )
                        objectives[signed] = _objective(directory, config)
                coarse = _five_point(objectives, 0.00375)
                fine = _five_point(objectives, 0.001875)
                fd[parameterization] = fine
                fd_stability[parameterization] = abs(fine - coarse) / max(
                    abs(fine), abs(coarse)
                )

            products = {
                key: directional_derivative(value, physical_direction)
                for key, value in exact_gradients[grad_form].items()
            }
            decomposition = fd["T"] - fd["R"] - fd["M"]
            decomposition_scale = max(
                abs(fd["T"]), abs(fd["R"]), abs(fd["M"])
            )
            decomposition_relative = abs(decomposition) / decomposition_scale
            row = {
                "direction": direction_name,
                "grad_form": grad_form,
                "d_r_fd": fd["R"],
                "d_m_fd": fd["M"],
                "d_t_fd": fd["T"],
                "g_r_dot_p": products["R"],
                "g_m_dot_p": products["M"],
                "g_t_dot_p": products["T"],
                "k_r": fd["R"] / products["R"],
                "k_m": fd["M"] / products["M"],
                "k_t": fd["T"] / products["T"],
                "fd_stability": fd_stability,
                "fd_decomposition_residual": decomposition,
                "fd_decomposition_relative": decomposition_relative,
                "acceptance_ceiling": {
                    key: max(
                        5.0e-5,
                        2.0 * fd_stability[key],
                        decomposition_relative if key == "T" else 0.0,
                    )
                    for key in ("R", "M", "T")
                },
            }
            rows.append(row)
            legacy_comparison.append(
                {
                    "direction": direction_name,
                    "grad_form": grad_form,
                    "legacy_post_g_r_dot_p": directional_derivative(
                        temporal_gradients["legacy_post_xy"], physical_direction
                    ),
                    "legacy_metric_b_g_r_dot_p": directional_derivative(
                        temporal_gradients["legacy_b_xy"], physical_direction
                    ),
                    "exact_metric_legacy_state_g_r_dot_p": directional_derivative(
                        temporal_gradients["exact_metric_legacy_xy"], physical_direction
                    ),
                    "exact_metric_b_g_r_dot_p": products["R"],
                }
            )

    output = {
        "base_git_sha": "47f3ab93ae7b27433980dafeb77646c9f5a6940a",
        "prediction_recorded_before_runs": PRE_RECORDED_PREDICTION,
        "source_audit": {
            "forward_prop_rho_z": "R*D_sigma/DH before v advance; no DT; no source",
            "source_density_derivative": "none for QUELLTYP=1 direct vz injection",
            "legacy_pvzp1_form2": "A velocity before reverse update_v",
            "legacy_pvzp1_form1": "DT accumulation of A velocities",
            "B_VELOCITY": "after reverse update_v and residual injection",
            "C_velocity": "identical to B because reverse update_s does not alter velocity",
            "state_metric_mapping": (
                "lambda_v=rho*y_v; legacy lambda correlation requires /rho, "
                "exact-metric y correlation does not"
            ),
            "exact_kinetic_form2": "-DT * y_v(B) * forward_prop_rho_z",
            "exact_kinetic_form1": (
                "-DT * time_integral(y_v(B)) * forward_prop_rho_z"
            ),
        },
        "baseline_relative_l2": baseline_mismatch,
        "component_identity": component_identity,
        "rows": rows,
        "legacy_comparison": legacy_comparison,
        "heterogeneous_run": False,
        "dtinv_secondary_run": False,
    }
    output_path = repository_root / "tests" / "m5f_density_gradient_diagnostics.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    for row in rows:
        assert abs(row["k_r"] - 1.0) < row["acceptance_ceiling"]["R"], row
        assert abs(row["k_m"] - 1.0) < row["acceptance_ceiling"]["M"], row
        assert abs(row["k_t"] - 1.0) < row["acceptance_ceiling"]["T"], row
        scale = max(abs(row["d_t_fd"]), abs(row["d_r_fd"]), abs(row["d_m_fd"]))
        decomposition_ceiling = max(
            5.0e-5, 2.0 * max(row["fd_stability"].values())
        )
        assert abs(row["fd_decomposition_residual"]) / scale < decomposition_ceiling, row


def test_exact_discrete_sh_density_gradient_heterogeneous(
    tmp_path, repository_root, mpiexec
):
    binaries = {
        name: _binary(name)
        for name in (
            "legacy_post_xy",
            "exact_b_pre_xy",
            "exact_b_integrated_pre_xy",
            "exact_b_pre_x",
            "exact_b_pre_y",
        )
    }
    config = SHFWIGradientConfig()
    density_background = [
        config.density_kg_m3
        * (
            1.0
            + 0.1
            * math.sin(2.0 * math.pi * (ix - 0.5) / config.nx)
            * math.sin(2.0 * math.pi * (iy - 0.5) / config.ny)
        )
        for ix in range(1, config.nx + 1)
        for iy in range(1, config.ny + 1)
    ]
    target_direction = gaussian_direction(
        nx=config.nx, ny=config.ny, dh_m=config.dh_m,
        center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
        sigma_m=70.0,
    )
    directions = {
        "broad_gaussian_sigma_80m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
            sigma_m=80.0,
        ),
        "shifted_holdout_sigma_60m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=610.0, center_y_m=520.0, sigma_m=60.0,
        ),
    }
    observed_dir = tmp_path / "observed"
    generate_density_observed_case(
        observed_dir,
        config=config,
        direction=target_direction,
        density_background=density_background,
    )
    _run_case(
        observed_dir, repository_root, binaries["legacy_post_xy"], mpiexec,
        config, "m5f_heterogeneous_observed", False,
    )
    observed = observed_dir / "su" / "synthetic_y.su.shot1"
    assert observed.is_file() and observed.stat().st_size > 0

    baseline_samples = {}
    for parameterization in ("T", "R", "M"):
        directory = tmp_path / "baseline" / parameterization
        generate_density_case(
            directory, config=config, parameterization=parameterization,
            epsilon_fraction=0.0, direction=target_direction, grad_form=2,
            mode=0, density_background=density_background,
        )
        _run_case(
            directory, repository_root, binaries["legacy_post_xy"], mpiexec,
            config, f"m5f_heterogeneous_baseline_{parameterization}", False,
        )
        baseline_samples[parameterization] = read_su_float_samples(
            directory / "su" / "synthetic_y.su.shot1",
            len(config.receiver_x_m), round(config.time_s / config.dt_s),
        )
    baseline_mismatch = {
        parameterization: _relative_l2(
            baseline_samples["T"], baseline_samples[parameterization]
        )
        for parameterization in ("R", "M")
    }
    assert max(baseline_mismatch.values()) <= 1.0e-7, baseline_mismatch

    rows = []
    identities = []
    for grad_form in (2, 1):
        center_r = tmp_path / "gradient" / f"form{grad_form}_r"
        generate_density_case(
            center_r, config=config, parameterization="R", epsilon_fraction=0.0,
            direction=target_direction, grad_form=grad_form, mode=1,
            observed_su=observed, density_background=density_background,
        )
        kinetic_binary = (
            binaries["exact_b_pre_xy"] if grad_form == 2
            else binaries["exact_b_integrated_pre_xy"]
        )
        _run_case(
            center_r, repository_root, kinetic_binary, mpiexec, config,
            f"m5f_heterogeneous_form{grad_form}_r", True,
        )
        exact_r = [
            -config.dtinv * value for value in read_float_grid(
                center_r / "jacobian" / "gradient_p_rho.old", config.cell_count
            )
        ]
        components = {}
        for component in ("xy", "x", "y"):
            directory = tmp_path / "gradient" / f"form{grad_form}_m_{component}"
            generate_density_case(
                directory, config=config, parameterization="M",
                epsilon_fraction=0.0, direction=target_direction,
                grad_form=grad_form, mode=1, observed_su=observed,
                density_background=density_background,
            )
            _run_case(
                directory, repository_root, binaries[f"exact_b_pre_{component}"],
                mpiexec, config,
                f"m5f_heterogeneous_form{grad_form}_m_{component}", True,
            )
            components[component] = [
                -value for value in read_float_grid(
                    directory / "jacobian" / "gradient_p_u.old", config.cell_count
                )
            ]
        component_sum = [x + y for x, y in zip(components["x"], components["y"])]
        identity = {
            "grad_form": grad_form,
            "relative_l2": _relative_l2(components["xy"], component_sum),
            "normalized_correlation": _correlation(components["xy"], component_sum),
        }
        assert identity["relative_l2"] <= 2.0e-6, identity
        assert identity["normalized_correlation"] >= 0.999999999, identity
        identities.append(identity)
        exact_m = _exact_density_material_gradient(
            components["x"], components["y"], config, grad_form,
            density_background,
        )
        exact = {
            "R": exact_r,
            "M": exact_m,
            "T": [left + right for left, right in zip(exact_r, exact_m)],
        }

        for direction_name, direction in directions.items():
            physical_direction = [
                density * value for density, value in zip(density_background, direction)
            ]
            fd = {}
            stability = {}
            for parameterization in ("R", "M", "T"):
                objectives = {}
                for epsilon in EPSILONS:
                    for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                        signed = sign * epsilon
                        directory = (
                            tmp_path / "fd" / f"form{grad_form}" / direction_name
                            / parameterization / f"{side}_{epsilon:g}"
                        )
                        generate_density_case(
                            directory, config=config,
                            parameterization=parameterization,
                            epsilon_fraction=signed, direction=direction,
                            grad_form=grad_form, mode=1, observed_su=observed,
                            density_background=density_background,
                        )
                        _run_case(
                            directory, repository_root, binaries["legacy_post_xy"],
                            mpiexec, config,
                            f"m5f_heterogeneous_fd_{parameterization}_{direction_name}_form{grad_form}",
                            True,
                        )
                        objectives[signed] = _objective(directory, config)
                coarse = _five_point(objectives, 0.00375)
                fine = _five_point(objectives, 0.001875)
                fd[parameterization] = fine
                stability[parameterization] = abs(fine - coarse) / max(
                    abs(fine), abs(coarse)
                )
            products = {
                key: directional_derivative(values, physical_direction)
                for key, values in exact.items()
            }
            decomposition = fd["T"] - fd["R"] - fd["M"]
            scale = max(abs(fd["T"]), abs(fd["R"]), abs(fd["M"]))
            decomposition_relative = abs(decomposition) / scale
            acceptance = {
                key: max(
                    5.0e-5,
                    2.0 * stability[key],
                    decomposition_relative if key == "T" else 0.0,
                )
                for key in ("R", "M", "T")
            }
            row = {
                "direction": direction_name,
                "grad_form": grad_form,
                "d_r_fd": fd["R"], "g_r_dot_p": products["R"],
                "k_r": fd["R"] / products["R"],
                "d_m_fd": fd["M"], "g_m_dot_p": products["M"],
                "k_m": fd["M"] / products["M"],
                "d_t_fd": fd["T"], "g_t_dot_p": products["T"],
                "k_t": fd["T"] / products["T"],
                "fd_stability": stability,
                "fd_decomposition_residual": decomposition,
                "fd_decomposition_relative": decomposition_relative,
                "acceptance_ceiling": acceptance,
            }
            rows.append(row)
            assert abs(row["k_r"] - 1.0) < acceptance["R"], row
            assert abs(row["k_m"] - 1.0) < acceptance["M"], row
            assert abs(row["k_t"] - 1.0) < acceptance["T"], row
            assert decomposition_relative < max(
                5.0e-5, 2.0 * max(stability.values())
            ), row

    output = {
        "base_git_sha": "47f3ab93ae7b27433980dafeb77646c9f5a6940a",
        "density_background": "rho0*(1+0.1*sin(2pi*x)*sin(2pi*y))",
        "receiver_metric": "spatial R*C^T*r; no global post-hoc scaling",
        "baseline_relative_l2": baseline_mismatch,
        "component_identity": identities,
        "rows": rows,
        "holdout_run": True,
        "no_fitted_scalar": True,
    }
    (repository_root / "tests" / "m5f_density_gradient_heterogeneous.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )


def test_exact_discrete_sh_density_gradient_dtinv3(
    tmp_path, repository_root, mpiexec
):
    from dataclasses import replace

    binaries = {
        name: _binary(name)
        for name in (
            "legacy_post_xy", "exact_b_pre_xy", "exact_b_pre_x", "exact_b_pre_y"
        )
    }
    config = replace(SHFWIGradientConfig(), dtinv=3)
    target_direction = gaussian_direction(
        nx=config.nx, ny=config.ny, dh_m=config.dh_m,
        center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
        sigma_m=70.0,
    )
    direction = gaussian_direction(
        nx=config.nx, ny=config.ny, dh_m=config.dh_m,
        center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
        sigma_m=80.0,
    )
    observed_dir = tmp_path / "observed"
    generate_density_observed_case(
        observed_dir, config=config, direction=target_direction
    )
    _run_case(
        observed_dir, repository_root, binaries["legacy_post_xy"], mpiexec,
        config, "m5f_dtinv3_observed", False,
    )
    observed = observed_dir / "su" / "synthetic_y.su.shot1"

    center_r = tmp_path / "gradient_r"
    generate_density_case(
        center_r, config=config, parameterization="R", epsilon_fraction=0.0,
        direction=target_direction, grad_form=2, mode=1, observed_su=observed,
    )
    _run_case(
        center_r, repository_root, binaries["exact_b_pre_xy"], mpiexec,
        config, "m5f_dtinv3_r", True,
    )
    exact_r = [
        -config.dtinv * value for value in read_float_grid(
            center_r / "jacobian" / "gradient_p_rho.old", config.cell_count
        )
    ]
    components = {}
    for component in ("x", "y"):
        directory = tmp_path / f"gradient_m_{component}"
        generate_density_case(
            directory, config=config, parameterization="M", epsilon_fraction=0.0,
            direction=target_direction, grad_form=2, mode=1, observed_su=observed,
        )
        _run_case(
            directory, repository_root, binaries[f"exact_b_pre_{component}"],
            mpiexec, config, f"m5f_dtinv3_m_{component}", True,
        )
        components[component] = [
            -value for value in read_float_grid(
                directory / "jacobian" / "gradient_p_u.old", config.cell_count
            )
        ]
    exact_m = _exact_density_material_gradient(
        components["x"], components["y"], config, 2,
        [config.density_kg_m3] * config.cell_count,
    )
    exact = {
        "R": exact_r,
        "M": exact_m,
        "T": [left + right for left, right in zip(exact_r, exact_m)],
    }
    fd = {}
    stability = {}
    for parameterization in ("R", "M", "T"):
        objectives = {}
        for epsilon in EPSILONS:
            for sign, side in ((1.0, "plus"), (-1.0, "minus")):
                signed = sign * epsilon
                directory = tmp_path / "fd" / parameterization / f"{side}_{epsilon:g}"
                generate_density_case(
                    directory, config=config, parameterization=parameterization,
                    epsilon_fraction=signed, direction=direction, grad_form=2,
                    mode=1, observed_su=observed,
                )
                _run_case(
                    directory, repository_root, binaries["legacy_post_xy"],
                    mpiexec, config, f"m5f_dtinv3_fd_{parameterization}", True,
                )
                objectives[signed] = _objective(directory, config)
        coarse = _five_point(objectives, 0.00375)
        fine = _five_point(objectives, 0.001875)
        fd[parameterization] = fine
        stability[parameterization] = abs(fine - coarse) / max(abs(fine), abs(coarse))
    physical_direction = [config.density_kg_m3 * value for value in direction]
    products = {
        key: directional_derivative(values, physical_direction)
        for key, values in exact.items()
    }
    decomposition = fd["T"] - fd["R"] - fd["M"]
    decomposition_relative = abs(decomposition) / max(
        abs(fd["T"]), abs(fd["R"]), abs(fd["M"])
    )
    result = {
        "dtinv": 3,
        "quadrature": "DT*DTINV rectangular sampling",
        "d_fd": fd,
        "g_dot_p": products,
        "k": {key: fd[key] / products[key] for key in ("R", "M", "T")},
        "fd_stability": stability,
        "decomposition_relative": decomposition_relative,
    }
    (repository_root / "tests" / "m5f_density_gradient_dtinv3.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for key in ("R", "M", "T"):
        ceiling = max(
            5.0e-5,
            2.0 * stability[key],
            decomposition_relative if key == "T" else 0.0,
        )
        assert abs(result["k"][key] - 1.0) < ceiling, result
    assert decomposition_relative < max(5.0e-5, 2.0 * max(stability.values())), result
