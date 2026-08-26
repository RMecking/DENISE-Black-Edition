from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.cases.sh_free_surface_fwi import (
    set_surface_case,
    symbolic_surface_operators,
    surface_fwi_config,
    surface_reflection_timing,
)
from tests.cases.sh_fwi_density import (
    generate_density_case,
    generate_density_observed_case,
)
from tests.cases.sh_fwi_gradient import (
    generate_forward_observed_case,
    generate_fwi_case,
)
from tests.cases.sh_fwi_taylor import (
    generate_taylor_fwi_case,
    generate_taylor_observed_case,
)
from tests.physics.test_sh_fwi_component_diagnostic import (
    _correlation,
    _relative_l2,
)
from tests.physics.test_sh_fwi_production_gradient import (
    RHO_EPSILONS,
    VS_EPSILONS,
    _accept,
    _compare,
    _fd_metrics,
    _gradient,
)
from tests.physics.test_sh_fwi_taylor import _objective
from tests.utilities.fwi_gradient import (
    directional_derivative,
    gaussian_direction,
    read_su_float_samples,
)
from tests.utilities.runner import result_summary, run_denise


def _run(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    config,
    role: str,
    nprocx: int = 1,
    nprocy: int = 1,
    free_surface: bool = True,
):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration=config.as_metadata()
        | {
            "role": role,
            "milestone": "M6.1e",
            "free_surface": int(free_surface),
            "nprocx": nprocx,
            "nprocy": nprocy,
        },
        timeout_seconds=120.0,
    )
    assert result.returncode == 0, result_summary(result)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _direction(config, *, x_m: float = 550.0, y_m: float = 210.0, sigma_m: float = 80.0):
    return gaussian_direction(
        nx=config.nx,
        ny=config.ny,
        dh_m=config.dh_m,
        center_x_m=x_m,
        center_y_m=y_m,
        sigma_m=sigma_m,
    )


def _configure(directory: Path, *, role: str, free_surface: bool = True, nprocx=1, nprocy=1):
    set_surface_case(
        directory,
        free_surface=free_surface,
        role=role,
        nprocx=nprocx,
        nprocy=nprocy,
    )


def _waveform(path: Path, config) -> list[float]:
    return read_su_float_samples(
        path,
        len(config.receiver_x_m),
        round(config.time_s / config.dt_s),
    )


def test_surface_reflection_timing_uses_native_vz_coordinates():
    timing = surface_reflection_timing(surface_fwi_config())
    expected = {
        "source_native_x_m": 255.0,
        "source_native_y_m": 195.0,
        "receiver_native_x_m": 735.0,
        "receiver_native_y_m": 195.0,
        "image_path_distance_m": 618.4658438426491,
        "continuum_travel_time_s": 0.2688981929750648,
        "source_peak_delay_s": 0.125,
        "predicted_reflection_peak_s": 0.3938981929750648,
        "post_arrival_margin_s": 0.2061018070249352,
    }
    for key, value in expected.items():
        assert timing[key] == pytest.approx(value, rel=0.0, abs=1.0e-15)


@pytest.mark.parametrize("fd_order", (2, 4, 6, 8, 10, 12))
def test_symbolic_surface_derivatives_are_exact_negative_transposes(fd_order):
    d_plus, d_minus, core_depth = symbolic_surface_operators(fd_order)
    for velocity_row in range(1, core_depth + 1):
        for stress_column in range(1, core_depth + 1):
            actual = d_minus.get((velocity_row, stress_column), {})
            expected = {
                coefficient: -multiplicity
                for coefficient, multiplicity in d_plus.get(
                    (stress_column, velocity_row), {}
                ).items()
            }
            assert actual == expected, {
                "fd_order": fd_order,
                "velocity_row": velocity_row,
                "stress_column": stress_column,
                "d_minus": actual,
                "negative_transpose_d_plus": expected,
            }


@pytest.mark.parametrize(("free_surface", "expected"), ((True, 1), (False, 0)))
def test_run_provenance_records_requested_free_surface(
    monkeypatch, tmp_path, free_surface, expected
):
    captured = {}

    def fake_run_denise(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "tests.physics.test_sh_free_surface_fwi_gradient.run_denise",
        fake_run_denise,
    )
    _run(
        tmp_path,
        repository_root=tmp_path,
        denise_binary=tmp_path / "denise",
        mpiexec="mpiexec",
        config=surface_fwi_config(),
        role="m61e_pure_provenance_contract",
        free_surface=free_surface,
    )
    assert captured["configuration"]["free_surface"] == expected


@pytest.mark.integration
@pytest.mark.extended
def test_01_surface_coupled_geometry_and_waveform_diagnostic(
    tmp_path, repository_root, denise_binary, mpiexec
):
    config = surface_fwi_config()
    timing = surface_reflection_timing(config)
    assert timing["predicted_reflection_peak_s"] < config.time_s - 0.10, timing

    traces = {}
    hashes = {}
    for free_surface in (True, False):
        directory = tmp_path / ("free_surface" if free_surface else "absorbing_top")
        generate_forward_observed_case(directory, config=config)
        _configure(
            directory,
            role=f"m61e_waveform_diagnostic_fs{int(free_surface)}",
            free_surface=free_surface,
        )
        _run(
            directory,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=config,
            role=f"m61e_waveform_diagnostic_fs{int(free_surface)}",
            free_surface=free_surface,
        )
        path = directory / "su" / "synthetic_y.su.shot1"
        assert path.is_file() and path.stat().st_size > 0
        traces[free_surface] = _waveform(path, config)
        hashes[free_surface] = _sha256(path)

    diagnostic = {
        "timing": timing,
        "free_surface": 1,
        "free_surface_vs_absorbing_top": {
            "relative_l2": _relative_l2(traces[True], traces[False]),
            "normalized_correlation": _correlation(traces[True], traces[False]),
            "free_surface_sha256": hashes[True],
            "absorbing_top_sha256": hashes[False],
            "acceptance": None,
            "purpose": "surface-coupling diagnostic only; no waveform threshold",
        },
    }
    print("M61E_GEOMETRY " + json.dumps(diagnostic, sort_keys=True))


@pytest.mark.integration
@pytest.mark.extended
def test_02_direct_free_surface_gradient_gate(
    tmp_path, repository_root, denise_binary, mpiexec
):
    config = surface_fwi_config()
    direction = _direction(config)
    report = {"vs": [], "rho": []}

    observed_vs_directory = tmp_path / "vs" / "observed"
    generate_forward_observed_case(observed_vs_directory, config=config)
    _configure(observed_vs_directory, role="m61e_vs_observed")
    _run(
        observed_vs_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role="m61e_vs_observed",
    )
    observed_vs = observed_vs_directory / "su" / "synthetic_y.su.shot1"

    for grad_form in (1, 2):
        objectives = {}
        for epsilon in VS_EPSILONS:
            for sign in (-1.0, 1.0):
                signed = sign * epsilon
                directory = tmp_path / "vs" / f"form{grad_form}" / f"{signed:+.7f}"
                generate_fwi_case(
                    directory,
                    observed_su=observed_vs,
                    epsilon_fraction=signed,
                    grad_form=grad_form,
                    config=config,
                    direction=direction,
                )
                _configure(directory, role=f"m61e_vs_fd_form{grad_form}_{signed:+.7f}")
                _run(
                    directory,
                    repository_root=repository_root,
                    denise_binary=denise_binary,
                    mpiexec=mpiexec,
                    config=config,
                    role=f"m61e_vs_fd_form{grad_form}_{signed:+.7f}",
                )
                objectives[signed] = _objective(directory, config)
        fd = _fd_metrics(objectives, VS_EPSILONS)
        gradient_directory = tmp_path / "vs" / f"gradient_form{grad_form}"
        generate_fwi_case(
            gradient_directory,
            observed_su=observed_vs,
            epsilon_fraction=0.0,
            grad_form=grad_form,
            config=config,
            direction=direction,
        )
        _configure(gradient_directory, role=f"m61e_vs_gradient_form{grad_form}")
        _run(
            gradient_directory,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=config,
            role=f"m61e_vs_gradient_form{grad_form}",
        )
        physical_direction = [
            value * component
            for value, component in zip(config.background_vs(), direction)
        ]
        row = {"grad_form": grad_form, "fd_diagnostics": fd}
        row.update(
            _accept(
                fd,
                directional_derivative(
                    _gradient(gradient_directory, config, "u"), physical_direction
                ),
            )
        )
        report["vs"].append(row)

    observed_rho_directory = tmp_path / "rho" / "observed"
    generate_density_observed_case(
        observed_rho_directory,
        config=config,
        direction=config.direction(),
    )
    _configure(observed_rho_directory, role="m61e_rho_observed")
    _run(
        observed_rho_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role="m61e_rho_observed",
    )
    observed_rho = observed_rho_directory / "su" / "synthetic_y.su.shot1"

    for grad_form in (1, 2):
        objectives = {}
        for epsilon in RHO_EPSILONS:
            for sign in (-1.0, 1.0):
                signed = sign * epsilon
                directory = tmp_path / "rho" / f"form{grad_form}" / f"{signed:+.7f}"
                generate_density_case(
                    directory,
                    config=config,
                    parameterization="T",
                    epsilon_fraction=signed,
                    direction=direction,
                    grad_form=grad_form,
                    mode=1,
                    observed_su=observed_rho,
                )
                _configure(directory, role=f"m61e_rho_fd_form{grad_form}_{signed:+.7f}")
                _run(
                    directory,
                    repository_root=repository_root,
                    denise_binary=denise_binary,
                    mpiexec=mpiexec,
                    config=config,
                    role=f"m61e_rho_fd_form{grad_form}_{signed:+.7f}",
                )
                objectives[signed] = _objective(directory, config)
        fd = _fd_metrics(objectives, RHO_EPSILONS)
        gradient_directory = tmp_path / "rho" / f"gradient_form{grad_form}"
        generate_density_case(
            gradient_directory,
            config=config,
            parameterization="T",
            epsilon_fraction=0.0,
            direction=direction,
            grad_form=grad_form,
            mode=1,
            observed_su=observed_rho,
        )
        _configure(gradient_directory, role=f"m61e_rho_gradient_form{grad_form}")
        _run(
            gradient_directory,
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            config=config,
            role=f"m61e_rho_gradient_form{grad_form}",
        )
        physical_direction = [config.density_kg_m3 * value for value in direction]
        row = {"grad_form": grad_form, "fd_diagnostics": fd}
        row.update(
            _accept(
                fd,
                directional_derivative(
                    _gradient(gradient_directory, config, "rho"), physical_direction
                ),
            )
        )
        report["rho"].append(row)

    print("M61E_DIRECT_FD " + json.dumps(report, sort_keys=True))
    rejected = [row for rows in report.values() for row in rows if not row["accepted"]]
    assert not rejected, rejected


def _joint_fields(config):
    vs_background = tuple(config.background_vs())
    rho_background = (config.density_kg_m3,) * config.cell_count
    target_vs_p = _direction(config, x_m=500.0, y_m=190.0, sigma_m=55.0)
    target_rho_p = _direction(config, x_m=610.0, y_m=230.0, sigma_m=55.0)
    p_vs = tuple(_direction(config, x_m=560.0, y_m=210.0, sigma_m=80.0))
    p_rho = tuple(_direction(config, x_m=530.0, y_m=180.0, sigma_m=70.0))
    target_vs = tuple(
        value * (1.0 + config.target_fraction * component)
        for value, component in zip(vs_background, target_vs_p)
    )
    target_rho = tuple(
        value * (1.0 + config.target_fraction * component)
        for value, component in zip(rho_background, target_rho_p)
    )
    delta_vs = tuple(value * component for value, component in zip(vs_background, p_vs))
    delta_rho = tuple(value * component for value, component in zip(rho_background, p_rho))
    return vs_background, rho_background, target_vs, target_rho, delta_vs, delta_rho


@pytest.mark.integration
@pytest.mark.extended
def test_04_free_surface_mpi_and_dtinv_holdouts(
    tmp_path, repository_root, denise_binary, mpiexec
):
    config = surface_fwi_config()
    (
        vs_background,
        rho_background,
        target_vs,
        target_rho,
        delta_vs,
        delta_rho,
    ) = _joint_fields(config)
    observed_directory = tmp_path / "observed"
    generate_taylor_observed_case(
        observed_directory,
        config=config,
        target_vs=target_vs,
        target_rho=target_rho,
    )
    _configure(observed_directory, role="m61e_mpi_observed")
    _run(
        observed_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=config,
        role="m61e_mpi_observed",
    )
    observed = observed_directory / "su" / "synthetic_y.su.shot1"
    report = {"mpi": [], "dtinv": {}}

    references = {}
    for grad_form in (1, 2):
        variants = {}
        for nprocx, nprocy in ((1, 1), (2, 1), (1, 2)):
            directory = tmp_path / "mpi" / f"form{grad_form}_{nprocx}x{nprocy}"
            generate_taylor_fwi_case(
                directory,
                config=config,
                observed_su=observed,
                vs_background=vs_background,
                rho_background=rho_background,
                delta_vs=delta_vs,
                delta_rho=delta_rho,
                epsilon=0.0,
                grad_form=grad_form,
                active_vs=True,
                active_rho=True,
            )
            _configure(
                directory,
                role=f"m61e_mpi_form{grad_form}_{nprocx}x{nprocy}",
                nprocx=nprocx,
                nprocy=nprocy,
            )
            _run(
                directory,
                repository_root=repository_root,
                denise_binary=denise_binary,
                mpiexec=mpiexec,
                config=config,
                role=f"m61e_mpi_form{grad_form}_{nprocx}x{nprocy}",
                nprocx=nprocx,
                nprocy=nprocy,
            )
            variants[(nprocx, nprocy)] = {
                "directory": directory,
                "vs": _gradient(directory, config, "u"),
                "rho": _gradient(directory, config, "rho"),
                "input_hashes": {
                    name: _sha256(directory / name)
                    for name in (
                        "model/current.vs",
                        "model/current.rho",
                        "source.dat",
                        "receiver.dat",
                        "observed_y.su.shot1",
                    )
                },
            }
        reference = variants[(1, 1)]
        references[grad_form] = reference
        for decomposition in ((2, 1), (1, 2)):
            candidate = variants[decomposition]
            assert candidate["input_hashes"] == reference["input_hashes"]
            row = {"grad_form": grad_form, "decomposition": list(decomposition)}
            for component, direction in (("vs", delta_vs), ("rho", delta_rho)):
                metrics = _compare(reference[component], candidate[component])
                reference_product = directional_derivative(reference[component], direction)
                candidate_product = directional_derivative(candidate[component], direction)
                product_relative_error = abs(candidate_product - reference_product) / max(
                    abs(reference_product), abs(candidate_product), 1.0e-30
                )
                metrics.update(
                    {
                        "reference_directional_product": reference_product,
                        "candidate_directional_product": candidate_product,
                        "directional_product_relative_error": product_relative_error,
                        "byte_identical": candidate[component] == reference[component],
                    }
                )
                assert metrics["relative_l2"] <= 2.0e-6, metrics
                assert metrics["normalized_correlation"] >= 0.999999999, metrics
                assert product_relative_error <= 2.0e-6, metrics
                row[component] = metrics
            row["input_hashes"] = candidate["input_hashes"]
            report["mpi"].append(row)

    dtinv1 = references[2]
    dtinv3_config = replace(config, dtinv=3)
    dtinv3_directory = tmp_path / "dtinv3"
    generate_taylor_fwi_case(
        dtinv3_directory,
        config=dtinv3_config,
        observed_su=observed,
        vs_background=vs_background,
        rho_background=rho_background,
        delta_vs=delta_vs,
        delta_rho=delta_rho,
        epsilon=0.0,
        grad_form=2,
        active_vs=True,
        active_rho=True,
    )
    _configure(dtinv3_directory, role="m61e_dtinv3")
    _run(
        dtinv3_directory,
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        config=dtinv3_config,
        role="m61e_dtinv3",
    )
    objective1 = _objective(dtinv1["directory"], config)
    objective3 = _objective(dtinv3_directory, dtinv3_config)
    assert objective1 == objective3
    report["dtinv"] = {"objective_dtinv1": objective1, "objective_dtinv3": objective3}
    for component in ("vs", "rho"):
        candidate = _gradient(
            dtinv3_directory,
            dtinv3_config,
            "u" if component == "vs" else "rho",
        )
        metrics = _compare(dtinv1[component], candidate)
        assert metrics["relative_l2"] <= 2.0e-6, metrics
        report["dtinv"][component] = metrics
    print("M61E_MPI_DTINV " + json.dumps(report, sort_keys=True))
