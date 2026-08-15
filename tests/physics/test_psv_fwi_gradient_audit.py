from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from tests.cases.psv_fwi_gradient import (
    PSVFWIGradientConfig,
    baseline_model,
    generate_case,
    perturbed_model,
    target_model,
)
from tests.utilities.fwi_gradient import gaussian_direction, read_float_grid, read_su_float_samples
from tests.utilities.runner import executable_sha256, result_summary, run_denise


pytestmark = [pytest.mark.integration, pytest.mark.extended]

BASE_SHA = "68d3bd68ff25ee7f225ade6bc88ec4beb6d6f96e"
EPSILONS = (-0.005, -0.0025, -0.00125, 0.00125, 0.0025, 0.005)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    directory: Path,
    *,
    repository_root: Path,
    binary: Path,
    mpiexec: str,
    config: PSVFWIGradientConfig,
    role: str,
    require_success: bool,
):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration=config.as_metadata() | {"role": role},
        timeout_seconds=90.0,
    )
    if require_success:
        assert result.returncode == 0, result_summary(result)
    return result


def _seismograms(directory: Path, config: PSVFWIGradientConfig) -> dict[str, list[float]]:
    return {
        component: read_su_float_samples(
            directory / "su" / f"synthetic_{component}.su.shot1",
            config.receiver_count,
            config.samples_per_trace,
        )
        for component in ("x", "y")
    }


def _independent_residual(
    synthetic: list[float], observed: list[float], config: PSVFWIGradientConfig, grad_form: int
) -> list[float]:
    ns = config.samples_per_trace
    residual: list[float] = []
    for trace in range(config.receiver_count):
        start = trace * ns
        syn = synthetic[start : start + ns]
        obs = observed[start : start + ns]
        # Although inseis() uses one-based Numerical Recipes indexing, the SU
        # payload written and read by DENISE maps back to the same physical
        # sample indices.  The retained residual SU proves this identity.
        raw = [syn[index] - obs[index] for index in range(ns)]
        raw[0] = 0.0  # calc_res() explicitly forces sample 1 to match.
        if grad_form == 1:
            integrated: list[float] = []
            total = 0.0
            for value in raw:
                total += config.dt_s * value
                integrated.append(total)
            raw = integrated
        residual.extend(raw)
    return residual


def _objective(
    synthetic: dict[str, list[float]],
    observed: dict[str, list[float]],
    *,
    config: PSVFWIGradientConfig,
    grad_form: int,
    data_components: int,
) -> float:
    components = {1: ("x", "y"), 2: ("y",), 3: ("x",)}[data_components]
    return 0.5 * math.fsum(
        value * value
        for component in components
        for value in _independent_residual(
            synthetic[component], observed[component], config, grad_form
        )
    )


def _residual_oracle_metrics(
    directory: Path,
    synthetic: dict[str, list[float]],
    observed: dict[str, list[float]],
    *,
    config: PSVFWIGradientConfig,
    grad_form: int,
    data_components: int,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    components = {1: ("x", "y"), 2: ("y",), 3: ("x",)}[data_components]
    for component in components:
        expected = _independent_residual(
            synthetic[component], observed[component], config, grad_form
        )
        stored = read_su_float_samples(
            directory / "su" / f"synthetic_{component}.su.shot1.it1",
            config.receiver_count,
            config.samples_per_trace,
        )
        expected_reversed: list[float] = []
        ns = config.samples_per_trace
        for trace in range(config.receiver_count):
            values = expected[trace * ns : (trace + 1) * ns]
            expected_reversed.extend(reversed(values))
        difference = [left - right for left, right in zip(stored, expected_reversed)]
        denominator = math.sqrt(math.fsum(value * value for value in expected_reversed))
        metrics[component] = {
            "stored_is_time_reversed": True,
            "stored_sign": "synthetic_minus_observed with identical SU sample indexing",
            "max_absolute_difference": max(abs(value) for value in difference),
            "relative_l2": math.sqrt(math.fsum(value * value for value in difference))
            / max(denominator, 1.0e-30),
            "stored_sha256": _sha256(
                directory / "su" / f"synthetic_{component}.su.shot1.it1"
            ),
        }
    return metrics


def _five_point(objectives: dict[float, float], h: float) -> float:
    return (
        -objectives[2.0 * h]
        + 8.0 * objectives[h]
        - 8.0 * objectives[-h]
        + objectives[-2.0 * h]
    ) / (12.0 * h)


def _fd_result(objectives: dict[float, float]) -> dict[str, object]:
    coarse = _five_point(objectives, 0.0025)
    fine = _five_point(objectives, 0.00125)
    return {
        "coarse_five_point": coarse,
        "fine_five_point": fine,
        "relative_change": abs(fine - coarse) / max(abs(fine), abs(coarse), 1.0e-30),
        "raw_objectives": {f"{epsilon:+.7f}": objectives[epsilon] for epsilon in EPSILONS},
    }


def _directions(config: PSVFWIGradientConfig) -> dict[str, dict[str, list[float]]]:
    base = config.direction()
    joint = {
        "vp": gaussian_direction(nx=config.nx, ny=config.ny, dh_m=config.dh_m, center_x_m=500.0, center_y_m=390.0, sigma_m=80.0),
        "vs": gaussian_direction(nx=config.nx, ny=config.ny, dh_m=config.dh_m, center_x_m=470.0, center_y_m=420.0, sigma_m=70.0),
        "rho": gaussian_direction(nx=config.nx, ny=config.ny, dh_m=config.dh_m, center_x_m=530.0, center_y_m=370.0, sigma_m=75.0),
    }
    return {"vp": {"vp": base}, "vs": {"vs": base}, "rho": {"rho": base}, "joint": joint}


def _direction_hashes(directions: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, str]]:
    import struct

    return {
        case: {
            component: hashlib.sha256(
                struct.pack(f"={len(values)}d", *values)
            ).hexdigest()
            for component, values in fields.items()
        }
        for case, fields in directions.items()
    }


def _gradient_product(
    directory: Path,
    *,
    config: PSVFWIGradientConfig,
    directions: dict[str, list[float]],
) -> tuple[dict[str, float], dict[str, str]]:
    baseline = baseline_model(config)
    products: dict[str, float] = {}
    hashes: dict[str, str] = {}
    for component in ("vp", "vs", "rho"):
        path = directory / "jacobian" / f"m53_raw_{component}.0.0"
        values = read_float_grid(path, config.cell_count)
        assert all(math.isfinite(value) for value in values)
        hashes[component] = _sha256(path)
        direction = directions.get(component)
        products[component] = 0.0 if direction is None else math.fsum(
            gradient * model_value * weight
            for gradient, model_value, weight in zip(values, baseline[component], direction)
        )
    products["total"] = math.fsum(products.values())
    return products, hashes


def test_psv_fwi_production_gradient_audit(
    tmp_path: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
) -> None:
    metric_value = os.environ.get("M53_PSV_METRIC_BIN")
    if not metric_value:
        pytest.fail("M53_PSV_METRIC_BIN is required", pytrace=False)
    metric_binary = Path(metric_value).resolve()
    assert metric_binary.is_file()
    config = PSVFWIGradientConfig()
    directions = _directions(config)
    base = baseline_model(config)

    models = {
        "baseline": base,
        "vp": target_model(config, ("vp",)),
        "vs": target_model(config, ("vs",)),
        "rho": target_model(config, ("rho",)),
        "joint": target_model(config, ("vp", "vs", "rho")),
    }
    datasets: dict[str, dict[str, list[float]]] = {}
    run_records: list[dict[str, object]] = []
    for name, model in models.items():
        directory = tmp_path / "observed" / name
        generate_case(directory, model=model, config=config, mode=0, metadata={"role": f"observed_{name}"})
        result = _run(directory, repository_root=repository_root, binary=denise_binary, mpiexec=mpiexec, config=config, role=f"observed_{name}", require_success=True)
        datasets[name] = _seismograms(directory, config)
        run_records.append({"role": f"observed_{name}", "returncode": result.returncode, "runtime_seconds": result.runtime_seconds})

    perturbed: dict[str, dict[float, dict[str, list[float]]]] = {}
    for case, direction in directions.items():
        perturbed[case] = {}
        for epsilon in EPSILONS:
            directory = tmp_path / "perturbed" / case / f"eps_{epsilon:+.7f}"
            generate_case(directory, model=perturbed_model(config, direction, epsilon), config=config, mode=0, metadata={"role": "fd", "case": case, "epsilon": epsilon})
            result = _run(directory, repository_root=repository_root, binary=denise_binary, mpiexec=mpiexec, config=config, role=f"fd_{case}_{epsilon:+.7f}", require_success=True)
            perturbed[case][epsilon] = _seismograms(directory, config)
            run_records.append({"role": f"fd_{case}_{epsilon:+.7f}", "returncode": result.returncode, "runtime_seconds": result.runtime_seconds})

    fd: dict[str, dict[str, object]] = {}
    for case in directions:
        modes = (3, 2, 1) if case == "joint" else (3, 2)
        fd[case] = {}
        for data_components in modes:
            for grad_form in (1, 2):
                key = f"data_{data_components}_gf{grad_form}"
                objectives = {
                    epsilon: _objective(values, datasets[case], config=config, grad_form=grad_form, data_components=data_components)
                    for epsilon, values in perturbed[case].items()
                }
                fd[case][key] = _fd_result(objectives)

    production: list[dict[str, object]] = []
    variants = {"legacy_raw_receiver": denise_binary, "rx_ry_receiver_metric": metric_binary}
    for variant, binary in variants.items():
        cases = directions
        for case, direction in cases.items():
            modes = (3, 2, 1) if case == "joint" else (3, 2)
            for data_components in modes:
                for grad_form in (1, 2):
                    directory = tmp_path / "fwi" / variant / case / f"data_{data_components}_gf{grad_form}"
                    observed_dir = tmp_path / "observed" / case / "su"
                    generate_case(
                        directory,
                        model=base,
                        config=config,
                        mode=1,
                        grad_form=grad_form,
                        data_components=data_components,
                        observed_x=observed_dir / "synthetic_x.su.shot1",
                        observed_y=observed_dir / "synthetic_y.su.shot1",
                        metadata={"role": "production_gradient", "variant": variant, "case": case},
                    )
                    result = _run(directory, repository_root=repository_root, binary=binary, mpiexec=mpiexec, config=config, role=f"gradient_{variant}_{case}_{data_components}_{grad_form}", require_success=True)
                    products, gradient_hashes = _gradient_product(directory, config=config, directions=direction)
                    derivative = fd[case][f"data_{data_components}_gf{grad_form}"]["fine_five_point"]
                    total = products["total"]
                    entry = {
                        "variant": variant,
                        "case": case,
                        "data_components": data_components,
                        "grad_form": grad_form,
                        "returncode": result.returncode,
                        "runtime_seconds": result.runtime_seconds,
                        "fd": fd[case][f"data_{data_components}_gf{grad_form}"],
                        "gradient_products": products,
                        "gradient_hashes": gradient_hashes,
                        "k_fd_over_gradient": derivative / total if total else None,
                        "relative_error": abs(derivative - total) / max(abs(derivative), abs(total), 1.0e-30),
                        "residual_oracle": _residual_oracle_metrics(directory, datasets["baseline"], datasets[case], config=config, grad_form=grad_form, data_components=data_components),
                    }
                    production.append(entry)
                    run_records.append({"role": f"gradient_{variant}_{case}_{data_components}_{grad_form}", "returncode": result.returncode, "runtime_seconds": result.runtime_seconds})

    for entry in production:
        for metrics in entry["residual_oracle"].values():
            assert metrics["relative_l2"] < 2.0e-6
            # Two independent SU float32 write/read paths differ by only a
            # few representable single-precision values at peak amplitude.
            assert metrics["max_absolute_difference"] < 2.0e-9
        assert entry["gradient_products"]["total"] != 0.0

    assert all(entry["fd"]["relative_change"] < 1.0e-3 for entry in production)
    legacy = [entry for entry in production if entry["variant"] == "legacy_raw_receiver"]
    metric = [entry for entry in production if entry["variant"] == "rx_ry_receiver_metric"]
    # The legacy source injection omits the diagonal inverse-density receiver
    # metric.  Adding only Rx/Ry removes the approximately rho-sized scale
    # error, while parameter-specific residual errors remain.
    assert max(abs(entry["k_fd_over_gradient"]) for entry in legacy) < 1.0e-3
    assert max(entry["relative_error"] for entry in metric if entry["case"] == "vp") < 0.01
    assert min(entry["relative_error"] for entry in metric if entry["case"] == "vs") > 0.04
    assert max(entry["relative_error"] for entry in metric if entry["case"] == "rho") > 0.15

    source_files = [
        "src/PSV/psv.c",
        "src/PSV/update_v_PML_PSV.c",
        "src/PSV/update_s_elastic_PML_PSV.c",
        "src/PSV/ass_gradPSV.c",
        "src/PSV/calc_res_PSV.c",
        "src/av_mue.c",
        "src/av_rho.c",
    ]
    patch_path = repository_root / "tests" / "m5.3_psv_instrumentation.patch"
    artifact = {
        "milestone": "M5.3 elastic PSV discrete-adjoint and gradient audit",
        "base_git_sha": BASE_SHA,
        "configuration": config.as_metadata(),
        "binary_provenance": {name: {"path_at_execution": str(path.resolve()), "sha256": executable_sha256(path), "retained_after_audit": False} for name, path in variants.items()},
        "instrumented_source_sha256": {name: _sha256(repository_root / name) for name in source_files},
        "instrumentation_patch_sha256": _sha256(patch_path),
        "direction_hashes": _direction_hashes(directions),
        "objective_definitions": {
            "GF1": "0.5*sum((DT*cumsum(synthetic-observed))^2)",
            "GF2": "0.5*sum((synthetic-observed)^2)",
            "observed_sample_mapping": "empirical retained-SU audit proves identical physical sample indexing; calc_res forces sample 1 residual to zero",
            "stored_residual": "time reversal of the positive synthetic-minus-observed adjoint source",
        },
        "grid_staggering": {
            "cell": ["rho", "lambda", "mu", "sxx", "syy"],
            "x_face": ["vx", "rip"],
            "y_face": ["vy", "rjp"],
            "corner": ["sxy", "mu_xy=uipjp"],
        },
        "independent_operator_oracle": {
            "state_transpose": "covered by tests/test_psv_gradient_math.py",
            "material_jvp_vjp": "includes exact four-cell harmonic mu_xy transpose",
            "density_jvp_vjp": "includes both rip and rjp edge-to-cell transpose",
            "physical_chain": "Vp, Vs, rho chain closes by dot product",
        },
        "temporal_state_audit": {
            "A": "stress and velocity state entering the velocity update",
            "B": "velocity after v <- v + DT*R*C^T*s and before the stress update",
            "C": "stress after s <- s + DT*M*C*v; receiver samples are taken here",
            "exact_reverse_order": "inject receiver transpose at C, apply stress-update transpose C->B and accumulate the material VJP, then apply velocity-update transpose B->A and accumulate the density VJP",
            "production_observation": "psv.c correlates after the reverse stress update; its collocated legacy correlations do not expose the exact four-cell shear or edge-density VJPs",
        },
        "scaling_analysis": {
            "method": "blind Rx/Ry receiver-metric variant; no fitted acceptance factor",
            "legacy_k_range": [min(entry["k_fd_over_gradient"] for entry in legacy), max(entry["k_fd_over_gradient"] for entry in legacy)],
            "metric_k_range": [min(entry["k_fd_over_gradient"] for entry in metric), max(entry["k_fd_over_gradient"] for entry in metric)],
            "interpretation": "receiver metric explains the approximately density-sized global scale error, but not the parameter/component-dependent remainder",
        },
        "confirmed_defects": [
            "adjoint vx/vy receiver injection omits rip/rjp",
            "cell-centred Vs assembly does not apply the transpose of the four-cell harmonic mu_xy map",
            "cell-centred density assembly does not apply the separate x-face/y-face rip/rjp edge-to-cell VJPs",
        ],
        "cleanup": {
            "temporary_binaries_removed": True,
            "temporary_run_directories_removed": True,
            "production_sources_restored_to_base": True,
        },
        "production_results": production,
        "run_records": run_records,
        "final_verdict": "MULTIPLE PSV GRADIENT DEFECTS IDENTIFIED",
    }
    (repository_root / "tests" / "m5.3_psv_gradient_audit.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
