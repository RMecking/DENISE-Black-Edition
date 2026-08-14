from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Sequence

import pytest

from tests.cases.sh_fwi_gradient import (
    SHFWIGradientConfig,
    generate_forward_observed_case,
    generate_fwi_case,
)
from tests.utilities.fwi_gradient import (
    compliance_average_vjp,
    directional_derivative,
    flat_top_direction,
    gaussian_direction,
    harmonic_mean,
    harmonic_mean_vjp,
    read_float_grid,
)
from tests.utilities.runner import executable_sha256, result_summary, run_denise


pytestmark = [pytest.mark.integration, pytest.mark.extended]


def _required_binary(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} is required for the temporary M5.0d diagnostic", pytrace=False)
    path = Path(value).resolve()
    assert path.is_file(), f"Missing diagnostic binary: {path}"
    return path


def _run(directory, *, repository_root, binary, mpiexec, config, role, fwi=False):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=binary,
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


def _xmajor_to_rows(values: Sequence[float], nx: int, ny: int) -> list[list[float]]:
    if len(values) != nx * ny:
        raise ValueError("grid size mismatch")
    return [[values[i * ny + j] for i in range(nx)] for j in range(ny)]


def _rows_to_xmajor(rows: Sequence[Sequence[float]]) -> list[float]:
    ny = len(rows)
    nx = len(rows[0])
    return [rows[j][i] for i in range(nx) for j in range(ny)]


def _relative_l2(reference: Sequence[float], candidate: Sequence[float]) -> float:
    difference = math.fsum((a - b) ** 2 for a, b in zip(reference, candidate))
    norm = math.fsum(value * value for value in reference)
    return math.sqrt(difference / norm)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = math.fsum(a * b for a, b in zip(left, right))
    return numerator / math.sqrt(
        math.fsum(value * value for value in left)
        * math.fsum(value * value for value in right)
    )


def _recover_correlation(
    production_gradient: Sequence[float], config: SHFWIGradientConfig, grad_form: int
) -> list[float]:
    if grad_form == 1:
        factor = -config.density_kg_m3 / (2.0 * config.dt_s * config.vs_m_s**3)
    else:
        factor = -1.0 / (
            2.0 * config.density_kg_m3 * config.vs_m_s * config.dt_s
        )
    return [factor * value for value in production_gradient]


def _exact_gradient(
    cx_values: Sequence[float],
    cy_values: Sequence[float],
    config: SHFWIGradientConfig,
    grad_form: int,
) -> list[float]:
    cx_full = _xmajor_to_rows(cx_values, config.nx, config.ny)
    cy_full = _xmajor_to_rows(cy_values, config.nx, config.ny)
    cx = [row[:-1] for row in cx_full]
    cy = cy_full[:-1]
    vs = _xmajor_to_rows(config.background_vs(), config.nx, config.ny)
    density = config.density_kg_m3

    if grad_form == 1:
        gcx = [
            [config.dt_s * config.dtinv / density * value for value in row]
            for row in cx
        ]
        gcy = [
            [config.dt_s * config.dtinv / density * value for value in row]
            for row in cy
        ]
        gc = compliance_average_vjp((config.ny, config.nx), gcx, gcy)
        gvs = [
            [-2.0 / (density * vs[j][i] ** 3) * gc[j][i] for i in range(config.nx)]
            for j in range(config.ny)
        ]
    else:
        mu = [
            [density * vs[j][i] ** 2 for i in range(config.nx)]
            for j in range(config.ny)
        ]
        mu_x, mu_y = harmonic_mean(mu)
        gmu_x = [
            [
                -config.dtinv / (density * mu_x[j][i]) * cx[j][i]
                for i in range(config.nx - 1)
            ]
            for j in range(config.ny)
        ]
        gmu_y = [
            [
                -config.dtinv / (density * mu_y[j][i]) * cy[j][i]
                for i in range(config.nx)
            ]
            for j in range(config.ny - 1)
        ]
        gmu = harmonic_mean_vjp(mu, gmu_x, gmu_y)
        gvs = [
            [2.0 * density * vs[j][i] * gmu[j][i] for i in range(config.nx)]
            for j in range(config.ny)
        ]
    return _rows_to_xmajor(gvs)


def _five_point(rows: Sequence[dict[str, object]], h: float) -> float:
    by_epsilon = {float(row["epsilon_fraction"]): row for row in rows}
    outer = by_epsilon[2.0 * h]
    inner = by_epsilon[h]
    return (
        -float(outer["j_plus"])
        + 8.0 * float(inner["j_plus"])
        - 8.0 * float(inner["j_minus"])
        + float(outer["j_minus"])
    ) / (12.0 * h)


def _fd_reference(repository_root: Path, direction: str, grad_form: int) -> dict[str, float]:
    filename = (
        "m5c_flat_top_diagnostics.json"
        if direction == "flat_top"
        else "m5c_gaussian_width_diagnostics.json"
    )
    document = json.loads((repository_root / "tests" / filename).read_text(encoding="utf-8"))
    source = next(
        row
        for row in document["rows"]
        if row["direction"] == direction and row["grad_form"] == grad_form
    )
    d5_coarse = _five_point(source["epsilons"], 0.0075)
    d5_fine = _five_point(source["epsilons"], 0.00375)
    return {
        "central_fine": float(source["d_fd_plateau"]),
        "central_relative_change": float(source["epsilons"][-1]["relative_to_previous"]),
        "five_point_coarse": d5_coarse,
        "five_point_fine": d5_fine,
        "five_point_relative_change": abs(d5_fine - d5_coarse)
        / max(abs(d5_fine), abs(d5_coarse)),
    }


def _compiler_version() -> str:
    result = subprocess.run(
        ["mpicc", "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def test_component_resolved_sh_staggered_gradient(
    tmp_path, repository_root, mpiexec
):
    binaries = {
        mode: _required_binary(f"M5D_{mode.upper()}_BIN") for mode in ("xy", "x", "y")
    }
    config = SHFWIGradientConfig()
    observed_dir = tmp_path / "observed"
    generate_forward_observed_case(observed_dir, config=config)
    observed_run = _run(
        observed_dir,
        repository_root=repository_root,
        binary=binaries["xy"],
        mpiexec=mpiexec,
        config=config,
        role="m5d_observed_xy",
    )
    observed = observed_dir / "su" / "synthetic_y.su.shot1"
    assert observed.is_file() and observed.stat().st_size > 0, result_summary(observed_run)

    gradients: dict[int, dict[str, list[float]]] = {}
    run_metadata = []
    for grad_form in (1, 2):
        gradients[grad_form] = {}
        for mode, binary in binaries.items():
            run_dir = tmp_path / f"form{grad_form}_{mode}"
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
                role=f"m5d_form_{grad_form}_{mode}",
                fwi=True,
            )
            stored = read_float_grid(
                run_dir / "jacobian" / "gradient_p_u.old", config.cell_count
            )
            gradients[grad_form][mode] = [-value for value in stored]
            run_metadata.append(json.loads(result.metadata_path.read_text(encoding="utf-8")))

    identities = []
    rows = []
    directions = {
        "gaussian_sigma_20m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m, sigma_m=20.0,
        ),
        "gaussian_sigma_80m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m, sigma_m=80.0,
        ),
        "flat_top": flat_top_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
            half_width_x_m=120.0, half_width_y_m=100.0, taper_m=80.0,
        ),
    }

    for grad_form in (1, 2):
        correlations = {
            mode: _recover_correlation(values, config, grad_form)
            for mode, values in gradients[grad_form].items()
        }
        component_sum = [
            x + y for x, y in zip(correlations["x"], correlations["y"])
        ]
        identity = {
            "grad_form": grad_form,
            "relative_l2": _relative_l2(correlations["xy"], component_sum),
            "normalized_correlation": _correlation(correlations["xy"], component_sum),
        }
        identities.append(identity)
        assert identity["relative_l2"] <= 2.0e-6, json.dumps(identity, indent=2)
        assert identity["normalized_correlation"] >= 0.999999999, json.dumps(identity, indent=2)

        exact = _exact_gradient(
            correlations["x"], correlations["y"], config, grad_form
        )
        for direction_name, direction in directions.items():
            physical_direction = [
                vs * value for vs, value in zip(config.background_vs(), direction)
            ]
            production = directional_derivative(
                gradients[grad_form]["xy"], physical_direction
            )
            if grad_form == 1:
                corrected_field = [
                    value * config.dtinv
                    / (config.density_kg_m3 * vs**6)
                    for value, vs in zip(
                        gradients[grad_form]["xy"], config.background_vs()
                    )
                ]
            else:
                corrected_field = [
                    value * config.dtinv
                    / (config.dt_s * config.density_kg_m3**2 * vs**2)
                    for value, vs in zip(
                        gradients[grad_form]["xy"], config.background_vs()
                    )
                ]
            local = directional_derivative(corrected_field, physical_direction)
            exact_product = directional_derivative(exact, physical_direction)
            fd = _fd_reference(repository_root, direction_name, grad_form)
            derivative = fd["five_point_fine"]
            rows.append(
                {
                    "direction": direction_name,
                    "grad_form": grad_form,
                    "d_fd": derivative,
                    "fd_diagnostics": fd,
                    "production_g_dot_p": production,
                    "m5b_corrected_local_g_dot_p": local,
                    "exact_vjp_g_dot_p": exact_product,
                    "k_production": derivative / production,
                    "k_local": derivative / local,
                    "k_exact_vjp": derivative / exact_product,
                    "absolute_local_residual": abs(derivative / local - 1.0),
                    "absolute_exact_vjp_residual": abs(derivative / exact_product - 1.0),
                }
            )

    build_commands = {
        "xy": "make -C src -B sh.o CFLAGS='-O3 -w -fno-stack-protector -D_FORTIFY_SOURCE=0 -fcommon'; make -C src denise",
        "x": "CFLAGS='-O3 -w -fno-stack-protector -D_FORTIFY_SOURCE=0 -fcommon -DM5_SH_GRAD_X_ONLY' make -e -C src -B sh.o; make -C src denise",
        "y": "CFLAGS='-O3 -w -fno-stack-protector -D_FORTIFY_SOURCE=0 -fcommon -DM5_SH_GRAD_Y_ONLY' make -e -C src -B sh.o; make -C src denise",
    }
    provenance = {
        mode: {
            "path": str(binary),
            "sha256": executable_sha256(binary),
            "build_command": build_commands[mode],
        }
        for mode, binary in binaries.items()
    }
    output = {
        "git_base_sha": "47f3ab93ae7b27433980dafeb77646c9f5a6940a",
        "git_dirty_during_instrumented_run": True,
        "source_patch_sha256": os.environ.get("M5D_PATCH_SHA256"),
        "compiler": "mpicc",
        "compiler_version": _compiler_version(),
        "compile_flags": "-O3 -w -fno-stack-protector -D_FORTIFY_SOURCE=0 -fcommon plus component macro",
        "binaries": provenance,
        "component_identity": identities,
        "rows": rows,
        "run_metadata": run_metadata,
        "finite_difference_source": (
            "Reused M5.0c objective evaluations; fine five-point derivative uses "
            "h=0.00375 and coarse comparison uses h=0.0075."
        ),
    }
    (repository_root / "tests" / "m5d_component_diagnostics.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
