from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tests.cases.psv_fwi_gradient import (
    PSVFWIGradientConfig,
    generate_case,
    heterogeneous_direction,
    heterogeneous_model,
    heterogeneous_perturbed_model,
    heterogeneous_target_model,
)
from tests.physics.test_psv_fwi_gradient_audit import (
    EPSILONS,
    _fd_result,
    _objective,
    _seismograms,
)
from tests.physics.test_psv_fwi_production_gradient import (
    _field_comparison,
    _gradient,
    _run,
)


pytestmark = [pytest.mark.integration, pytest.mark.extended]


def _product(
    directory: Path,
    config: PSVFWIGradientConfig,
    model: dict[str, list[float]],
    direction: dict[str, list[float]],
) -> dict[str, float]:
    result = {
        component: math.fsum(
            gradient * value * weight
            for gradient, value, weight in zip(
                _gradient(directory, config, component),
                model[component], direction[component],
            )
        )
        for component in ("vp", "vs", "rho")
    }
    result["total"] = math.fsum(result.values())
    return result


def test_psv_fwi_heterogeneous_fd_and_mpi_holdout(
    tmp_path: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
) -> None:
    config = PSVFWIGradientConfig()
    model = heterogeneous_model(config)
    direction = heterogeneous_direction(config)
    target = heterogeneous_target_model(config, model)
    run_records: list[dict[str, object]] = []

    observed_directory = tmp_path / "observed"
    generate_case(observed_directory, model=target, config=config, mode=0)
    result = _run(
        observed_directory, repository_root=repository_root,
        binary=denise_binary, mpiexec=mpiexec, config=config,
        role="heterogeneous_observed",
    )
    run_records.append({"role": "heterogeneous_observed", "returncode": result.returncode})
    observed = _seismograms(observed_directory, config)

    perturbed: dict[float, dict[str, list[float]]] = {}
    for epsilon in EPSILONS:
        directory = tmp_path / "fd" / f"eps_{epsilon:+.7f}"
        generate_case(
            directory,
            model=heterogeneous_perturbed_model(model, direction, epsilon),
            config=config,
            mode=0,
        )
        result = _run(
            directory, repository_root=repository_root, binary=denise_binary,
            mpiexec=mpiexec, config=config, role=f"heterogeneous_fd_{epsilon:+.7f}",
        )
        run_records.append({"role": f"heterogeneous_fd_{epsilon:+.7f}",
                            "returncode": result.returncode})
        perturbed[epsilon] = _seismograms(directory, config)

    fd_rows: list[dict[str, object]] = []
    mpi_rows: list[dict[str, object]] = []
    observed_x = observed_directory / "su/synthetic_x.su.shot1"
    observed_y = observed_directory / "su/synthetic_y.su.shot1"
    for grad_form in (1, 2):
        objectives = {
            epsilon: _objective(
                values, observed, config=config, grad_form=grad_form,
                data_components=1,
            )
            for epsilon, values in perturbed.items()
        }
        fd = _fd_result(objectives)
        reference = tmp_path / "gradient" / "1x1" / f"gf{grad_form}"
        generate_case(
            reference, model=model, config=config, mode=1,
            grad_form=grad_form, data_components=1,
            observed_x=observed_x, observed_y=observed_y,
        )
        result = _run(
            reference, repository_root=repository_root, binary=denise_binary,
            mpiexec=mpiexec, config=config,
            role=f"heterogeneous_1x1_gf{grad_form}",
        )
        run_records.append({"role": f"heterogeneous_1x1_gf{grad_form}",
                            "returncode": result.returncode})
        product = _product(reference, config, model, direction)
        derivative = float(fd["fine_five_point"])
        relative_error = abs(derivative - product["total"]) / max(
            abs(derivative), abs(product["total"]), 1.0e-30
        )
        assert float(fd["relative_change"]) < 1.0e-3
        assert relative_error <= max(5.0e-4, 5.0 * float(fd["relative_change"]))
        assert relative_error <= 2.5e-3
        fd_rows.append({
            "grad_form": grad_form,
            "data_components": 1,
            "fd": fd,
            "gradient_products": product,
            "relative_error": relative_error,
            "returncode": result.returncode,
        })

        reference_fields = {
            component: _gradient(reference, config, component)
            for component in ("vp", "vs", "rho")
        }
        for nprocx, nprocy in ((2, 1), (1, 2)):
            directory = tmp_path / "gradient" / f"{nprocx}x{nprocy}" / f"gf{grad_form}"
            generate_case(
                directory, model=model, config=config, mode=1,
                grad_form=grad_form, data_components=1,
                observed_x=observed_x, observed_y=observed_y,
                nprocx=nprocx, nprocy=nprocy,
            )
            result = _run(
                directory, repository_root=repository_root, binary=denise_binary,
                mpiexec=mpiexec, config=config,
                role=f"heterogeneous_{nprocx}x{nprocy}_gf{grad_form}",
                nprocx=nprocx, nprocy=nprocy,
            )
            run_records.append({
                "role": f"heterogeneous_{nprocx}x{nprocy}_gf{grad_form}",
                "returncode": result.returncode,
            })
            fields = {
                component: _field_comparison(
                    reference_fields[component],
                    _gradient(directory, config, component),
                )
                for component in ("vp", "vs", "rho")
            }
            candidate = _product(directory, config, model, direction)
            product_relative_difference = abs(
                candidate["total"] - product["total"]
            ) / max(abs(product["total"]), 1.0e-30)
            assert all(value["relative_l2"] < 2.0e-6 for value in fields.values())
            assert product_relative_difference < 2.0e-6
            mpi_rows.append({
                "decomposition": f"{nprocx}x{nprocy}",
                "grad_form": grad_form,
                "fields": fields,
                "reference_gradient_products": product,
                "candidate_gradient_products": candidate,
                "product_relative_difference": product_relative_difference,
                "returncode": result.returncode,
            })

    artifact_path = repository_root / "tests/m5.4_psv_gradient_production_validation.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["heterogeneous_holdout"] = {
        "description": (
            "smooth positive Vp/Vs/rho current model and independent joint "
            "direction crossing the 2x1 and 1x2 seams"
        ),
        "configuration": config.as_metadata(),
        "fd_results": fd_rows,
        "mpi_results": mpi_rows,
    }
    artifact["run_records"].extend(run_records)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
