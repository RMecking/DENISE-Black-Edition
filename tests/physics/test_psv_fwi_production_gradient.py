from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from tests.cases.psv_fwi_gradient import (
    PSVFWIGradientConfig,
    baseline_model,
    generate_case,
    perturbed_model,
    target_model,
)
from tests.physics.test_psv_fwi_gradient_audit import (
    EPSILONS,
    _directions,
    _fd_result,
    _objective,
    _seismograms,
)
from tests.utilities.fwi_gradient import read_float_grid, read_su_float_samples
from tests.utilities.runner import result_summary, run_denise


pytestmark = [pytest.mark.integration, pytest.mark.extended]
BASE_SHA = "b4a4ea95d4dc36d64b377923fb9526cfc06dc631"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(directory: Path, *, repository_root: Path, binary: Path, mpiexec: str,
         config: PSVFWIGradientConfig, role: str, nprocx: int = 1,
         nprocy: int = 1):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=binary,
        mpiexec=mpiexec,
        ranks=nprocx*nprocy,
        configuration=config.as_metadata() | {
            "role": role, "nprocx": nprocx, "nprocy": nprocy
        },
        timeout_seconds=90.0,
    )
    assert result.returncode == 0, result_summary(result)
    return result


def _gradient(directory: Path, config: PSVFWIGradientConfig,
              component: str) -> list[float]:
    suffix = {"vp": "", "vs": "_u", "rho": "_rho"}[component]
    # The first PSV L-BFGS iteration writes the positive objective gradient
    # after store_LBFGS_PSV's documented C_parameter normalization.
    values = read_float_grid(
        directory / "jacobian" / f"gradient_p{suffix}.old",
        config.cell_count,
    )
    assert all(math.isfinite(value) for value in values)
    normalization = {
        "vp": config.vp_m_s,
        "vs": config.vs_m_s,
        "rho": config.density_kg_m3,
    }[component]
    return [value/normalization for value in values]


def _product(directory: Path, config: PSVFWIGradientConfig,
             directions: dict[str, list[float]]) -> dict[str, float]:
    model = baseline_model(config)
    result: dict[str, float] = {}
    for component in ("vp", "vs", "rho"):
        direction = directions.get(component)
        result[component] = 0.0 if direction is None else math.fsum(
            g*m*d for g, m, d in zip(
                _gradient(directory, config, component),
                model[component], direction,
            )
        )
    result["total"] = math.fsum(result.values())
    return result


def _field_comparison(reference: list[float], candidate: list[float]) -> dict[str, float]:
    difference = [right-left for left, right in zip(reference, candidate)]
    ref_norm = math.sqrt(math.fsum(value*value for value in reference))
    cand_norm = math.sqrt(math.fsum(value*value for value in candidate))
    dot = math.fsum(left*right for left, right in zip(reference, candidate))
    return {
        "relative_l2": math.sqrt(math.fsum(value*value for value in difference))
        / max(ref_norm, 1.0e-30),
        "max_absolute_difference": max(abs(value) for value in difference),
        "normalized_correlation": dot/max(ref_norm*cand_norm, 1.0e-30),
    }


def test_psv_fwi_production_gradient_primary_gate(
    tmp_path: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
) -> None:
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
    observed: dict[str, dict[str, list[float]]] = {}
    run_records: list[dict[str, object]] = []
    for name, model in models.items():
        directory = tmp_path / "observed" / name
        generate_case(directory, model=model, config=config, mode=0)
        result = _run(directory, repository_root=repository_root, binary=denise_binary,
                      mpiexec=mpiexec, config=config, role=f"observed_{name}")
        run_records.append({"role": f"observed_{name}", "returncode": result.returncode})
        observed[name] = _seismograms(directory, config)

    perturbed: dict[str, dict[float, dict[str, list[float]]]] = {}
    for case, direction in directions.items():
        perturbed[case] = {}
        for epsilon in EPSILONS:
            directory = tmp_path / "fd" / case / f"eps_{epsilon:+.7f}"
            generate_case(directory, model=perturbed_model(config, direction, epsilon),
                          config=config, mode=0)
            result = _run(directory, repository_root=repository_root, binary=denise_binary,
                          mpiexec=mpiexec, config=config,
                          role=f"fd_{case}_{epsilon:+.7f}")
            run_records.append({"role": f"fd_{case}_{epsilon:+.7f}",
                                "returncode": result.returncode})
            perturbed[case][epsilon] = _seismograms(directory, config)

    rows: list[dict[str, object]] = []
    gradient_directories: dict[tuple[str, int, int], Path] = {}
    for case, direction in directions.items():
        components = (3, 2, 1) if case == "joint" else (3, 2)
        for data_components in components:
            for grad_form in (1, 2):
                objectives = {
                    epsilon: _objective(
                        values, observed[case], config=config,
                        grad_form=grad_form, data_components=data_components,
                    )
                    for epsilon, values in perturbed[case].items()
                }
                fd = _fd_result(objectives)
                directory = tmp_path / "fwi" / case / (
                    f"data_{data_components}_gf{grad_form}"
                )
                observed_dir = tmp_path / "observed" / case / "su"
                generate_case(
                    directory, model=base, config=config, mode=1,
                    grad_form=grad_form, data_components=data_components,
                    observed_x=observed_dir / "synthetic_x.su.shot1",
                    observed_y=observed_dir / "synthetic_y.su.shot1",
                )
                result = _run(directory, repository_root=repository_root,
                              binary=denise_binary, mpiexec=mpiexec, config=config,
                              role=f"gradient_{case}_{data_components}_{grad_form}")
                gradient_directories[(case, data_components, grad_form)] = directory
                run_records.append({
                    "role": f"gradient_{case}_{data_components}_{grad_form}",
                    "returncode": result.returncode,
                })
                product = _product(directory, config, direction)
                derivative = float(fd["fine_five_point"])
                relative_error = abs(derivative-product["total"])/max(
                    abs(derivative), abs(product["total"]), 1.0e-30
                )
                assert float(fd["relative_change"]) < 1.0e-3
                assert relative_error <= max(
                    5.0e-4, 5.0*float(fd["relative_change"])
                ), (case, data_components, grad_form, derivative, product,
                    relative_error, fd)
                assert relative_error <= 2.5e-3
                rows.append({
                    "case": case,
                    "data_components": data_components,
                    "grad_form": grad_form,
                    "fd": fd,
                    "gradient_products": product,
                    "relative_error": relative_error,
                    "returncode": result.returncode,
                })

    # Staggered VJPs cross both horizontal and vertical MPI seams.  Compare
    # merged physical fields and the joint directional product against 1x1.
    mpi_results: list[dict[str, object]] = []
    joint_direction = directions["joint"]
    joint_observed = tmp_path / "observed" / "joint" / "su"
    for grad_form in (1, 2):
        reference_directory = gradient_directories[("joint", 1, grad_form)]
        reference_fields = {
            component: _gradient(reference_directory, config, component)
            for component in ("vp", "vs", "rho")
        }
        reference_product = _product(reference_directory, config, joint_direction)
        for nprocx, nprocy in ((2, 1), (1, 2)):
            directory = tmp_path / "mpi" / f"{nprocx}x{nprocy}" / f"gf{grad_form}"
            generate_case(
                directory, model=base, config=config, mode=1,
                grad_form=grad_form, data_components=1,
                observed_x=joint_observed / "synthetic_x.su.shot1",
                observed_y=joint_observed / "synthetic_y.su.shot1",
                nprocx=nprocx, nprocy=nprocy,
            )
            result = _run(
                directory, repository_root=repository_root, binary=denise_binary,
                mpiexec=mpiexec, config=config,
                role=f"mpi_{nprocx}x{nprocy}_gf{grad_form}",
                nprocx=nprocx, nprocy=nprocy,
            )
            comparisons = {
                component: _field_comparison(
                    reference_fields[component], _gradient(directory, config, component)
                )
                for component in ("vp", "vs", "rho")
            }
            candidate_product = _product(directory, config, joint_direction)
            product_relative_difference = abs(
                candidate_product["total"]-reference_product["total"]
            )/max(abs(reference_product["total"]), 1.0e-30)
            assert all(value["relative_l2"] < 2.0e-6 for value in comparisons.values()), (
                nprocx, nprocy, grad_form, comparisons, reference_product,
                candidate_product, product_relative_difference,
            )
            assert product_relative_difference < 2.0e-6, (
                nprocx, nprocy, grad_form, comparisons, reference_product,
                candidate_product, product_relative_difference,
            )
            mpi_results.append({
                "decomposition": f"{nprocx}x{nprocy}", "grad_form": grad_form,
                "fields": comparisons,
                "reference_gradient_products": reference_product,
                "candidate_gradient_products": candidate_product,
                "product_relative_difference": product_relative_difference,
                "returncode": result.returncode,
            })
            run_records.append({"role": f"mpi_{nprocx}x{nprocy}_gf{grad_form}",
                                "returncode": result.returncode})

    # DTINV is storage quadrature only: forward/residual data remain identical;
    # the sampled gradient may move only within the established FD floor.
    dtinv_results: list[dict[str, object]] = []
    dt3 = replace(config, dtinv=3)
    dt3_forward = tmp_path / "dtinv3" / "forward"
    generate_case(dt3_forward, model=base, config=dt3, mode=0)
    dt3_forward_result = _run(
        dt3_forward, repository_root=repository_root, binary=denise_binary,
        mpiexec=mpiexec, config=dt3, role="dtinv3_forward",
    )
    forward_hash_match = all(
        _sha256(dt3_forward / "su" / f"synthetic_{component}.su.shot1")
        == _sha256(tmp_path / "observed" / "baseline" / "su" /
                   f"synthetic_{component}.su.shot1")
        for component in ("x", "y")
    )
    assert forward_hash_match
    run_records.append({"role": "dtinv3_forward",
                        "returncode": dt3_forward_result.returncode})
    for grad_form in (1, 2):
        reference_directory = gradient_directories[("joint", 1, grad_form)]
        directory = tmp_path / "dtinv3" / f"gf{grad_form}"
        generate_case(
            directory, model=base, config=dt3, mode=1, grad_form=grad_form,
            data_components=1,
            observed_x=joint_observed / "synthetic_x.su.shot1",
            observed_y=joint_observed / "synthetic_y.su.shot1",
        )
        result = _run(directory, repository_root=repository_root, binary=denise_binary,
                      mpiexec=mpiexec, config=dt3, role=f"dtinv3_gf{grad_form}")
        fields = {
            component: _field_comparison(
                _gradient(reference_directory, config, component),
                _gradient(directory, dt3, component),
            )
            for component in ("vp", "vs", "rho")
        }
        reference_product = _product(reference_directory, config, joint_direction)
        candidate_product = _product(directory, dt3, joint_direction)
        product_relative_difference = abs(
            candidate_product["total"]-reference_product["total"]
        )/max(abs(reference_product["total"]), 1.0e-30)
        residual_payload = {}
        for component in ("x", "y"):
            reference_samples = read_su_float_samples(
                reference_directory / "su" / f"synthetic_{component}.su.shot1.it1",
                config.receiver_count, config.samples_per_trace,
            )
            candidate_samples = read_su_float_samples(
                directory / "su" / f"synthetic_{component}.su.shot1.it1",
                config.receiver_count, config.samples_per_trace,
            )
            residual_payload[component] = {
                "max_absolute_difference": max(
                    abs(left-right) for left, right in
                    zip(reference_samples, candidate_samples)
                ),
                "reference_sha256": _sha256(
                    reference_directory / "su" /
                    f"synthetic_{component}.su.shot1.it1"
                ),
                "candidate_sha256": _sha256(
                    directory / "su" / f"synthetic_{component}.su.shot1.it1"
                ),
            }
        residual_payload_match = all(
            value["max_absolute_difference"] == 0.0
            for value in residual_payload.values()
        )
        assert forward_hash_match and residual_payload_match
        assert product_relative_difference < 2.5e-3
        dtinv_results.append({
            "dtinv": 3, "grad_form": grad_form, "fields": fields,
            "reference_gradient_products": reference_product,
            "candidate_gradient_products": candidate_product,
            "product_relative_difference": product_relative_difference,
            "forward_hash_match": forward_hash_match,
            "residual_payload_match": residual_payload_match,
            "residual_files": residual_payload,
            "returncode": result.returncode,
        })
        run_records.append({"role": f"dtinv3_gf{grad_form}",
                            "returncode": result.returncode})

    artifact = {
        "milestone": "M5.4 elastic PSV production gradient repair",
        "base_sha": BASE_SHA,
        "working_tree_head": BASE_SHA,
        "binary_sha256": _sha256(denise_binary),
        "configuration": config.as_metadata(),
        "objective_definitions": {
            "GF1": "0.5*sum((DT*cumsum(synthetic-observed))^2)",
            "GF2": "0.5*sum((synthetic-observed)^2)",
        },
        "primary_fd_results": rows,
        "mpi_results": mpi_results,
        "dtinv_results": dtinv_results,
        "run_records": run_records,
        "primary_gate_verdict": "M5.4 PRIMARY GRADIENT GATE PASSED",
    }
    (repository_root / "tests/m5.4_psv_gradient_production_validation.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
