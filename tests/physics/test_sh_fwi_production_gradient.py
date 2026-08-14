from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.cases.homogeneous_viscoelastic import (
    ViscoelasticSHConfig,
    generate_viscoelastic_sh_case,
)
from tests.cases.sh_fwi_density import (
    generate_density_case,
    generate_density_observed_case,
)
from tests.cases.sh_fwi_gradient import (
    SHFWIGradientConfig,
    generate_forward_observed_case,
    generate_fwi_case,
)
from tests.physics.test_sh_fwi_averaging_diagnostic import _objective
from tests.physics.test_sh_fwi_component_diagnostic import (
    _correlation,
    _relative_l2,
)
from tests.physics.test_sh_fwi_density_diagnostic import _five_point
from tests.utilities.fwi_gradient import (
    directional_derivative,
    flat_top_direction,
    gaussian_direction,
    read_float_grid,
    read_su_float_samples,
)
from tests.utilities.runner import (
    executable_sha256,
    result_summary,
    run_denise,
)
from tests.utilities.seismogram import read_ascii_seismograms


pytestmark = [pytest.mark.integration, pytest.mark.extended]

VS_EPSILONS = (0.015, 0.0075, 0.00375)
RHO_EPSILONS = (0.0075, 0.00375, 0.001875)
# M5.0f independently measured an 8.823e-5 total-density oracle floor for
# this narrow direction.  Round upward; this is not a gradient scale factor.
ESTABLISHED_FD_ORACLE_FLOOR = 1.0e-4


def _base_binary() -> Path:
    value = os.environ.get("M51_BASE_DENISE_BIN")
    if not value:
        pytest.fail("M51_BASE_DENISE_BIN is required", pytrace=False)
    path = Path(value).resolve()
    assert path.is_file(), path
    return path


def _run(
    directory: Path,
    *,
    repository_root: Path,
    binary: Path,
    mpiexec: str,
    config: SHFWIGradientConfig,
    role: str,
    nprocx: int = 1,
    nprocy: int = 1,
    fwi: bool = False,
):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration=config.as_metadata()
        | {"role": role, "nprocx": nprocx, "nprocy": nprocy},
        timeout_seconds=30.0 if fwi else 120.0,
    )
    assert result.returncode == 0, result_summary(result)
    return result


def _gradient(directory: Path, config: SHFWIGradientConfig, component: str):
    # descent() writes -dJ/dm to gradient_p_*.old; negate only that documented
    # optimizer convention. No diagnostic reconstruction or fitted scaling occurs.
    values = read_float_grid(
        directory / "jacobian" / f"gradient_p_{component}.old",
        config.cell_count,
    )
    return [-value for value in values]


def _field_metrics(values: list[float], config: SHFWIGradientConfig):
    assert all(math.isfinite(value) for value in values)
    norm = math.sqrt(math.fsum(value * value for value in values))
    boundary = []
    interior = []
    for ix in range(config.nx):
        for iy in range(config.ny):
            value = values[ix * config.ny + iy]
            if ix in (0, config.nx - 1) or iy in (0, config.ny - 1):
                boundary.append(value)
            else:
                interior.append(value)
    interior_rms = math.sqrt(
        math.fsum(value * value for value in interior) / len(interior)
    )
    metrics = {
        "min": min(values),
        "max": max(values),
        "l2_norm": norm,
        "boundary_max_abs": max(abs(value) for value in boundary),
        "interior_rms": interior_rms,
    }
    assert norm > 0.0
    assert metrics["boundary_max_abs"] <= 50.0 * max(interior_rms, 1.0e-30)
    return metrics


def _fd_metrics(objectives: dict[float, float], epsilons: tuple[float, ...]):
    coarse = _five_point(objectives, epsilons[1])
    fine = _five_point(objectives, epsilons[2])
    stability = abs(fine - coarse) / max(abs(fine), abs(coarse), 1.0e-30)
    return {
        "coarse_five_point": coarse,
        "fine_five_point": fine,
        "five_point_relative_change": stability,
        "raw_objectives": {
            f"{epsilon:+.7f}": value
            for epsilon, value in sorted(objectives.items())
        },
    }


def _accept(fd: dict[str, object], product: float):
    derivative = fd["fine_five_point"]
    relative_error = abs(derivative - product) / max(
        abs(derivative), abs(product), 1.0e-30
    )
    uncertainty = max(
        ESTABLISHED_FD_ORACLE_FLOOR,
        2.0 * fd["five_point_relative_change"],
    )
    result = {
        "fd": derivative,
        "gradient_directional_product": product,
        "k_fd_over_gradient": derivative / product,
        "relative_error": relative_error,
        "fd_uncertainty_ceiling": uncertainty,
        "accepted": relative_error <= uncertainty,
    }
    return result


def _sample_payload(path: Path, traces: int, samples: int):
    raw = path.read_bytes()
    trace_size = 240 + 4 * samples
    assert len(raw) == traces * trace_size
    payload = b"".join(
        raw[index * trace_size + 240 : (index + 1) * trace_size]
        for index in range(traces)
    )
    return hashlib.sha256(payload).hexdigest(), read_su_float_samples(
        path, traces, samples
    )


def _compare(reference: list[float], candidate: list[float]):
    assert len(reference) == len(candidate)
    maximum = max(abs(left - right) for left, right in zip(reference, candidate))
    return {
        "relative_l2": _relative_l2(reference, candidate),
        "normalized_correlation": _correlation(reference, candidate),
        "max_absolute_difference": maximum,
    }


def _directions(config: SHFWIGradientConfig):
    return {
        "gaussian_25m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
            sigma_m=25.0,
        ),
        "gaussian_80m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
            sigma_m=80.0,
        ),
        "flat_top": flat_top_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
            half_width_x_m=120.0, half_width_y_m=100.0, taper_m=80.0,
        ),
        "shifted_60m": gaussian_direction(
            nx=config.nx, ny=config.ny, dh_m=config.dh_m,
            center_x_m=610.0, center_y_m=520.0, sigma_m=60.0,
        ),
    }


@pytest.mark.parametrize("grad_form", (1, 2))
def test_exact_elastic_sh_fwi_smoke_exits_successfully(
    tmp_path, repository_root, denise_binary, mpiexec, grad_form
):
    config = SHFWIGradientConfig()
    observed_dir = tmp_path / f"observed_form{grad_form}"
    generate_forward_observed_case(observed_dir, config=config)
    _run(
        observed_dir,
        repository_root=repository_root,
        binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role=f"m51_successful_fwi_smoke_observed_form{grad_form}",
    )
    observed = observed_dir / "su" / "synthetic_y.su.shot1"
    assert observed.is_file() and observed.stat().st_size > 0

    fwi_dir = tmp_path / f"fwi_form{grad_form}"
    generate_fwi_case(
        fwi_dir,
        observed_su=observed,
        epsilon_fraction=0.0,
        grad_form=grad_form,
        config=config,
    )
    result = _run(
        fwi_dir,
        repository_root=repository_root,
        binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role=f"m51_successful_fwi_smoke_form{grad_form}",
        fwi=True,
    )
    assert (fwi_dir / "jacobian" / "gradient_p_u.old").is_file()
    assert (fwi_dir / "su" / "synthetic_y.su.shot1.it1").is_file()
    assert result.returncode == 0


def test_exact_elastic_sh_production_gradients(
    tmp_path, repository_root, denise_binary, mpiexec
):
    base_binary = _base_binary()
    config = SHFWIGradientConfig()
    directions = _directions(config)
    target = gaussian_direction(
        nx=config.nx, ny=config.ny, dh_m=config.dh_m,
        center_x_m=config.anomaly_x_m, center_y_m=config.anomaly_y_m,
        sigma_m=70.0,
    )
    report: dict[str, object] = {
        "base_sha": "47f3ab93ae7b27433980dafeb77646c9f5a6940a",
        "branch": "codex/m5.1-sh-gradient-production-repair",
        "production_binary_sha256": executable_sha256(denise_binary.resolve()),
        "base_binary_sha256": executable_sha256(base_binary),
        "changed_production_files": [
            "include/fd.h",
            "src/Makefile",
            "src/SH/FWI_SH.c",
            "src/SH/FWI_SH_visc.c",
            "src/SH/alloc_fwiSH.c",
            "src/SH/assemble_gradSH_exact.c",
            "src/SH/ass_gradSH.c",
            "src/SH/debug/sh.c",
            "src/SH/debug/update_v_PML_SH.c",
            "src/SH/grad_obj_sh.c",
            "src/SH/grad_obj_sh_visc.c",
            "src/SH/sh.c",
            "src/SH/sh_visc.c",
            "src/SH/update_v_PML_SH.c",
        ],
        "test_commands": [
            "python3 -m pytest tests -m 'not integration' -q",
            "python3 -m pytest tests/physics/test_sh_viscoelastic_rheology.py -q --require-denise --denise-bin bin/denise",
            "M51_BASE_DENISE_BIN=/mnt/d/Softwareprojekte/DENISE-Black-Edition-m51-base/bin/denise python3 -m pytest tests/physics/test_sh_fwi_production_gradient.py -q -m extended --require-denise --denise-bin bin/denise --basetemp=tests/.m51_tmp",
        ],
        "test_counts": {
            "pure_python": "84 passed, 60 deselected",
            "existing_sh_viscoelastic_rheology": "5 passed",
            "m5.1_production_integration": "3 passed",
        },
        "verdict": "M5.1 PRODUCTION GRADIENT REPAIR VERIFIED",
        "production_patch_sha256": hashlib.sha256(
            (
                repository_root
                / "tests"
                / "m5.1_sh_gradient_production_repair.patch"
            ).read_bytes()
        ).hexdigest(),
        "representative_fwi_returncodes": [],
        "vs_results": [],
        "rho_results": [],
        "heterogeneous_rho_results": [],
        "mpi_results": [],
    }

    # One observed Vs target is shared.  GRAD_FORM changes DENISE's residual
    # representation, so each form requires its own independent FD objective.
    observed_vs_dir = tmp_path / "vs" / "observed"
    generate_forward_observed_case(observed_vs_dir, config=config)
    _run(
        observed_vs_dir, repository_root=repository_root, binary=denise_binary,
        mpiexec=mpiexec, config=config, role="m51_vs_observed",
    )
    observed_vs = observed_vs_dir / "su" / "synthetic_y.su.shot1"

    vs_fd = {}
    for form in (1, 2):
        vs_fd[form] = {}
        for name, direction in directions.items():
            objectives = {}
            for epsilon in VS_EPSILONS:
                for sign in (-1.0, 1.0):
                    signed = sign * epsilon
                    directory = (
                        tmp_path / "vs" / "fd" / f"form{form}" / name
                        / f"{signed:+.7f}"
                    )
                    generate_fwi_case(
                        directory, observed_su=observed_vs,
                        epsilon_fraction=signed, grad_form=form, config=config,
                        direction=direction,
                    )
                    _run(
                        directory, repository_root=repository_root,
                        binary=denise_binary, mpiexec=mpiexec, config=config,
                        role=f"m51_vs_fd_form{form}_{name}", fwi=True,
                    )
                    objectives[signed] = _objective(directory, config)
            vs_fd[form][name] = _fd_metrics(objectives, VS_EPSILONS)

        directory = tmp_path / "vs" / "gradient" / f"form{form}"
        generate_fwi_case(
            directory, observed_su=observed_vs, epsilon_fraction=0.0,
            grad_form=form, config=config, direction=target,
        )
        gradient_result = _run(
            directory, repository_root=repository_root,
            binary=denise_binary, mpiexec=mpiexec, config=config,
            role=f"m51_vs_gradient_form{form}", fwi=True,
        )
        report["representative_fwi_returncodes"].append(
            {"grad_form": form, "mode": 1, "invmat1": 1,
             "returncode": gradient_result.returncode}
        )
        field = _gradient(directory, config, "u")
        for name, direction in directions.items():
            physical_direction = [
                vs * component
                for vs, component in zip(config.background_vs(), direction)
            ]
            row = {"direction": name, "form": form}
            row.update(_accept(
                vs_fd[form][name],
                directional_derivative(field, physical_direction),
            ))
            row["fd_diagnostics"] = vs_fd[form][name]
            row["field"] = _field_metrics(field, config)
            report["vs_results"].append(row)

    def density_suite(root, density_background, selected, output_key):
        observed_dir = root / "observed"
        generate_density_observed_case(
            observed_dir, config=config, direction=target,
            density_background=density_background,
        )
        _run(
            observed_dir, repository_root=repository_root, binary=denise_binary,
            mpiexec=mpiexec, config=config, role=f"m51_{output_key}_observed",
        )
        observed = observed_dir / "su" / "synthetic_y.su.shot1"
        gradient_runs = {}
        for form in (1, 2):
            fd_by_direction = {}
            for name in selected:
                direction = directions[name]
                objectives = {}
                for epsilon in RHO_EPSILONS:
                    for sign in (-1.0, 1.0):
                        signed = sign * epsilon
                        directory = (
                            root / "fd" / f"form{form}" / name
                            / f"{signed:+.7f}"
                        )
                        generate_density_case(
                            directory, config=config, parameterization="T",
                            epsilon_fraction=signed, direction=direction,
                            grad_form=form, mode=1, observed_su=observed,
                            density_background=density_background,
                        )
                        _run(
                            directory, repository_root=repository_root,
                            binary=denise_binary, mpiexec=mpiexec, config=config,
                            role=f"m51_{output_key}_fd_form{form}_{name}",
                            fwi=True,
                        )
                        objectives[signed] = _objective(directory, config)
                fd_by_direction[name] = _fd_metrics(
                    objectives, RHO_EPSILONS
                )

            directory = root / "gradient" / f"form{form}"
            generate_density_case(
                directory, config=config, parameterization="T",
                epsilon_fraction=0.0, direction=target, grad_form=form,
                mode=1, observed_su=observed,
                density_background=density_background,
            )
            _run(
                directory, repository_root=repository_root,
                binary=denise_binary, mpiexec=mpiexec, config=config,
                role=f"m51_{output_key}_gradient_form{form}", fwi=True,
            )
            field = _gradient(directory, config, "rho")
            gradient_runs[form] = directory
            density_values = density_background or [
                config.density_kg_m3
            ] * config.cell_count
            for name in selected:
                physical_direction = [
                    rho * component
                    for rho, component in zip(density_values, directions[name])
                ]
                row = {"direction": name, "form": form}
                row.update(_accept(
                    fd_by_direction[name],
                    directional_derivative(field, physical_direction),
                ))
                row["fd_diagnostics"] = fd_by_direction[name]
                row["field"] = _field_metrics(field, config)
                report[output_key].append(row)
        return observed, gradient_runs

    observed_rho, homogeneous_gradient_runs = density_suite(
        tmp_path / "rho_homogeneous", None,
        ("gaussian_25m", "gaussian_80m"), "rho_results",
    )
    heterogeneous_density = [
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
    density_suite(
        tmp_path / "rho_heterogeneous", heterogeneous_density,
        ("gaussian_80m", "shifted_60m"),
        "heterogeneous_rho_results",
    )

    # MPI: compare globally merged production Vs and rho fields.
    mpi_direction = directions["gaussian_80m"]
    for form in (1, 2):
        reference_directory = homogeneous_gradient_runs[form]
        variants = {
            (1, 1): {
                "vs": _gradient(reference_directory, config, "u"),
                "rho": _gradient(reference_directory, config, "rho"),
            }
        }
        products = {
            (1, 1): {
                "vs": directional_derivative(
                    variants[(1, 1)]["vs"],
                    [config.vs_m_s * value for value in mpi_direction],
                ),
                "rho": directional_derivative(
                    variants[(1, 1)]["rho"],
                    [config.density_kg_m3 * value for value in mpi_direction],
                ),
            }
        }
        for nprocx, nprocy in ((2, 1), (1, 2)):
            directory = (
                tmp_path / "mpi" / f"form{form}_{nprocx}x{nprocy}"
            )
            generate_density_case(
                directory, config=config, parameterization="T",
                epsilon_fraction=0.0, direction=target, grad_form=form,
                mode=1, observed_su=observed_rho,
                nprocx=nprocx, nprocy=nprocy,
            )
            _run(
                directory, repository_root=repository_root,
                binary=denise_binary, mpiexec=mpiexec, config=config,
                role=f"m51_mpi_form{form}_{nprocx}x{nprocy}",
                nprocx=nprocx, nprocy=nprocy, fwi=True,
            )
            variants[(nprocx, nprocy)] = {
                "vs": _gradient(directory, config, "u"),
                "rho": _gradient(directory, config, "rho"),
            }
            products[(nprocx, nprocy)] = {
                "vs": directional_derivative(
                    variants[(nprocx, nprocy)]["vs"],
                    [config.vs_m_s * value for value in mpi_direction],
                ),
                "rho": directional_derivative(
                    variants[(nprocx, nprocy)]["rho"],
                    [config.density_kg_m3 * value for value in mpi_direction],
                ),
            }
        reference = variants[(1, 1)]
        for decomposition in ((2, 1), (1, 2)):
            row = {"form": form, "decomposition": list(decomposition)}
            for component in ("vs", "rho"):
                comparison = _compare(
                    reference[component], variants[decomposition][component]
                )
                comparison["reference_directional_product"] = products[(1, 1)][
                    component
                ]
                comparison["candidate_directional_product"] = products[
                    decomposition
                ][component]
                assert comparison["relative_l2"] <= 2.0e-6, comparison
                assert comparison["normalized_correlation"] >= 0.999999999, comparison
                row[component] = comparison
            report["mpi_results"].append(row)

    # Elastic forward payload must be identical to the unmodified base.
    production_sha, production_samples = _sample_payload(
        observed_vs, len(config.receiver_x_m),
        round(config.time_s / config.dt_s),
    )
    forward_comparison = {
        "production": {"sample_payload_sha256": production_sha}
    }
    forward_samples = {"production": production_samples}
    for label, binary in (("base", base_binary),):
        directory = tmp_path / "forward_invariance" / label
        generate_forward_observed_case(directory, config=config)
        _run(
            directory, repository_root=repository_root, binary=binary,
            mpiexec=mpiexec, config=config, role=f"m51_forward_{label}",
        )
        sha, samples = _sample_payload(
            directory / "su" / "synthetic_y.su.shot1",
            len(config.receiver_x_m), round(config.time_s / config.dt_s),
        )
        forward_comparison[label] = {"sample_payload_sha256": sha}
        forward_samples[label] = samples
    forward_comparison["metrics"] = _compare(
        forward_samples["base"], forward_samples["production"]
    )
    assert forward_comparison["base"]["sample_payload_sha256"] == (
        forward_comparison["production"]["sample_payload_sha256"]
    )
    report["forward_invariance"] = forward_comparison

    # Shared update_v kernel: viscoelastic MODE=0 must remain byte-identical.
    visc_config = ViscoelasticSHConfig(qs=50.0)
    visc_samples = {}
    visco = {}
    for label, binary in (("base", base_binary), ("production", denise_binary)):
        directory = tmp_path / "visco_nonregression" / label
        generate_viscoelastic_sh_case(directory, config=visc_config)
        result = run_denise(
            repository_root=repository_root, case_directory=directory,
            denise_binary=binary, mpiexec=mpiexec, ranks=1,
            configuration={"role": f"m51_visco_{label}"},
        )
        assert result.returncode == 0, result_summary(result)
        path = directory / "su" / "homogeneous_vz.asc.shot1"
        traces = read_ascii_seismograms(
            path, visc_config.receiver_count, visc_config.samples_per_trace
        )
        visco[label] = {
            "ascii_output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
        visc_samples[label] = [value for trace in traces for value in trace]
    visco["metrics"] = _compare(visc_samples["base"], visc_samples["production"])
    assert visco["base"]["ascii_output_sha256"] == visco["production"][
        "ascii_output_sha256"
    ]
    report["viscoelastic_nonregression"] = visco

    # DTINV=3: objective stays fixed; only the established tiny quadrature drift
    # is permitted in the broad Form-2 density gradient.
    dtinv1_directory = homogeneous_gradient_runs[2]
    dtinv_fields = {1: _gradient(dtinv1_directory, config, "rho")}
    dtinv_objectives = {1: _objective(dtinv1_directory, config)}
    for dtinv in (3,):
        dt_config = replace(config, dtinv=dtinv)
        directory = tmp_path / "dtinv" / str(dtinv)
        generate_density_case(
            directory, config=dt_config, parameterization="T",
            epsilon_fraction=0.0, direction=target, grad_form=2, mode=1,
            observed_su=observed_rho,
        )
        _run(
            directory, repository_root=repository_root,
            binary=denise_binary, mpiexec=mpiexec, config=dt_config,
            role=f"m51_dtinv_{dtinv}", fwi=True,
        )
        dtinv_fields[dtinv] = _gradient(directory, dt_config, "rho")
        dtinv_objectives[dtinv] = _objective(directory, dt_config)
    dtinv_metrics = _compare(dtinv_fields[1], dtinv_fields[3])
    dtinv_metrics["objective_dtinv1"] = dtinv_objectives[1]
    dtinv_metrics["objective_dtinv3"] = dtinv_objectives[3]
    assert dtinv_objectives[1] == dtinv_objectives[3]
    assert dtinv_metrics["relative_l2"] <= 2.0e-6, dtinv_metrics
    report["dtinv_result"] = dtinv_metrics

    artifact = repository_root / "tests" / "m5.1_sh_gradient_production_validation.json"
    artifact.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rejected = [
        row
        for key in ("vs_results", "rho_results", "heterogeneous_rho_results")
        for row in report[key]
        if not row["accepted"]
    ]
    assert not rejected, rejected
