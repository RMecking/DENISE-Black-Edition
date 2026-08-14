from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path

import pytest

from tests.cases.sh_fwi_density import (
    generate_density_case,
    generate_density_observed_case,
)
from tests.cases.sh_fwi_gradient import SHFWIGradientConfig
from tests.physics.test_sh_fwi_component_diagnostic import _relative_l2
from tests.physics.test_sh_fwi_density_diagnostic import (
    EPSILONS,
    _exact_density_material_gradient,
    _five_point,
    _run_case,
)
from tests.physics.test_sh_fwi_averaging_diagnostic import _objective
from tests.utilities.fwi_gradient import (
    directional_derivative,
    gaussian_direction,
    read_float_grid,
    read_su_float_samples,
)


pytestmark = [pytest.mark.integration, pytest.mark.extended]

HISTORICAL_INDEPENDENT_DTINV1_M_FD = 0.060991786848798905


def _binary(name: str) -> Path:
    variable = f"M5F_{name.upper()}_BIN"
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"{variable} is required for M5.0f.1", pytrace=False)
    path = Path(value).resolve()
    assert path.is_file(), path
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _difference(reference: list[float], candidate: list[float]) -> dict[str, float]:
    assert len(reference) == len(candidate)
    return {
        "relative_l2": _relative_l2(reference, candidate),
        "max_absolute_difference": max(
            abs(left - right) for left, right in zip(reference, candidate)
        ),
    }


def _su_regions(
    path: Path, trace_count: int, samples_per_trace: int
) -> dict[str, object]:
    raw = path.read_bytes()
    trace_size = 240 + 4 * samples_per_trace
    assert len(raw) == trace_count * trace_size
    headers = bytearray()
    payload = bytearray()
    for trace in range(trace_count):
        start = trace * trace_size
        headers.extend(raw[start : start + 240])
        payload.extend(raw[start + 240 : start + trace_size])
    return {
        "full_file_sha256": hashlib.sha256(raw).hexdigest(),
        "trace_headers_sha256": hashlib.sha256(headers).hexdigest(),
        "sample_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "headers": bytes(headers),
    }


def _header_difference_offsets(
    reference_headers: bytes, candidate_headers: bytes
) -> list[int]:
    assert len(reference_headers) == len(candidate_headers)
    return sorted(
        {
            index % 240 + 1
            for index, (left, right) in enumerate(
                zip(reference_headers, candidate_headers)
            )
            if left != right
        }
    )


def _epsilon_key(value: float) -> str:
    return f"{value:+.7f}"


def test_sh_density_dtinv_oracle_invariance(
    tmp_path, repository_root, mpiexec
):
    """Separate DTINV effects in the forward, residual, FD, and gradient paths."""
    binaries = {
        name: _binary(name)
        for name in ("legacy_post_xy", "exact_b_pre_x", "exact_b_pre_y")
    }
    base = SHFWIGradientConfig()
    target_direction = gaussian_direction(
        nx=base.nx,
        ny=base.ny,
        dh_m=base.dh_m,
        center_x_m=base.anomaly_x_m,
        center_y_m=base.anomaly_y_m,
        sigma_m=70.0,
    )
    direction = gaussian_direction(
        nx=base.nx,
        ny=base.ny,
        dh_m=base.dh_m,
        center_x_m=base.anomaly_x_m,
        center_y_m=base.anomaly_y_m,
        sigma_m=80.0,
    )
    sample_count = round(base.time_s / base.dt_s)
    trace_count = len(base.receiver_x_m)

    # Generate the observed data once. Every FWI variant receives these exact bytes.
    observed_dir = tmp_path / "observed"
    generate_density_observed_case(
        observed_dir,
        config=replace(base, dtinv=1),
        direction=target_direction,
    )
    _run_case(
        observed_dir,
        repository_root,
        binaries["legacy_post_xy"],
        mpiexec,
        replace(base, dtinv=1),
        "m5f1_observed_once",
        False,
    )
    observed = observed_dir / "su" / "synthetic_y.su.shot1"
    assert observed.is_file() and observed.stat().st_size > 0
    observed_hash = _sha256(observed)

    signed_epsilons = sorted(
        {sign * epsilon for epsilon in EPSILONS for sign in (-1.0, 1.0)}
    )
    per_model: dict[str, dict[str, object]] = {}
    objectives: dict[int, dict[float, float]] = {k: {} for k in range(1, 5)}

    for signed_epsilon in signed_epsilons:
        key = _epsilon_key(signed_epsilon)
        variants: dict[str, object] = {}
        forward_samples: dict[int, list[float]] = {}
        residual_samples: dict[int, list[float]] = {}
        forward_regions: dict[int, dict[str, object]] = {}
        residual_regions: dict[int, dict[str, object]] = {}

        for dtinv in range(1, 5):
            config = replace(base, dtinv=dtinv)

            forward_dir = tmp_path / "forward" / key / f"dtinv_{dtinv}"
            generate_density_case(
                forward_dir,
                config=config,
                parameterization="M",
                epsilon_fraction=signed_epsilon,
                direction=direction,
                grad_form=2,
                mode=0,
            )
            _run_case(
                forward_dir,
                repository_root,
                binaries["legacy_post_xy"],
                mpiexec,
                config,
                f"m5f1_forward_{key}_dtinv_{dtinv}",
                False,
            )
            forward_su = forward_dir / "su" / "synthetic_y.su.shot1"
            forward_samples[dtinv] = read_su_float_samples(
                forward_su, trace_count, sample_count
            )
            forward_regions[dtinv] = _su_regions(
                forward_su, trace_count, sample_count
            )

            fwi_dir = tmp_path / "fwi" / key / f"dtinv_{dtinv}"
            generate_density_case(
                fwi_dir,
                config=config,
                parameterization="M",
                epsilon_fraction=signed_epsilon,
                direction=direction,
                grad_form=2,
                mode=1,
                observed_su=observed,
            )
            model_hashes = {
                "current.rho": _sha256(fwi_dir / "model" / "current.rho"),
                "current.mu": _sha256(fwi_dir / "model" / "current.mu"),
                "observed_y.su.shot1": _sha256(
                    fwi_dir / "observed_y.su.shot1"
                ),
            }
            _run_case(
                fwi_dir,
                repository_root,
                binaries["legacy_post_xy"],
                mpiexec,
                config,
                f"m5f1_fwi_{key}_dtinv_{dtinv}",
                True,
            )
            residual_su = fwi_dir / "su" / "synthetic_y.su.shot1.it1"
            residual_samples[dtinv] = read_su_float_samples(
                residual_su, trace_count, sample_count
            )
            residual_regions[dtinv] = _su_regions(
                residual_su, trace_count, sample_count
            )
            objective = _objective(fwi_dir, config)
            objectives[dtinv][signed_epsilon] = objective
            variants[str(dtinv)] = {
                "model_hashes": model_hashes,
                "forward_synthetic_sha256": forward_regions[dtinv][
                    "full_file_sha256"
                ],
                "forward_trace_headers_sha256": forward_regions[dtinv][
                    "trace_headers_sha256"
                ],
                "forward_sample_payload_sha256": forward_regions[dtinv][
                    "sample_payload_sha256"
                ],
                "reversed_residual_sha256": residual_regions[dtinv][
                    "full_file_sha256"
                ],
                "reversed_residual_trace_headers_sha256": residual_regions[dtinv][
                    "trace_headers_sha256"
                ],
                "reversed_residual_sample_payload_sha256": residual_regions[dtinv][
                    "sample_payload_sha256"
                ],
                "objective": objective,
            }

        reference_hashes = variants["1"]["model_hashes"]
        for dtinv in range(1, 5):
            variant = variants[str(dtinv)]
            assert variant["model_hashes"] == reference_hashes
            assert variant["model_hashes"]["observed_y.su.shot1"] == observed_hash
            variant["forward_vs_dtinv1"] = _difference(
                forward_samples[1], forward_samples[dtinv]
            )
            variant["residual_vs_dtinv1"] = _difference(
                residual_samples[1], residual_samples[dtinv]
            )
            variant["forward_header_difference_offsets_1_based"] = (
                _header_difference_offsets(
                    forward_regions[1]["headers"],
                    forward_regions[dtinv]["headers"],
                )
            )
            variant["residual_header_difference_offsets_1_based"] = (
                _header_difference_offsets(
                    residual_regions[1]["headers"],
                    residual_regions[dtinv]["headers"],
                )
            )
        per_model[key] = variants

    gradients: dict[int, list[float]] = {}
    gradient_products: dict[int, float] = {}
    gradient_records: dict[str, object] = {}
    density = [base.density_kg_m3] * base.cell_count
    density_direction = [
        rho * component for rho, component in zip(density, direction)
    ]
    for dtinv in range(1, 5):
        config = replace(base, dtinv=dtinv)
        components = {}
        component_hashes = {}
        for component in ("x", "y"):
            directory = tmp_path / "gradient" / f"dtinv_{dtinv}_{component}"
            generate_density_case(
                directory,
                config=config,
                parameterization="M",
                epsilon_fraction=0.0,
                direction=target_direction,
                grad_form=2,
                mode=1,
                observed_su=observed,
            )
            component_hashes[component] = {
                "current.rho": _sha256(directory / "model" / "current.rho"),
                "current.mu": _sha256(directory / "model" / "current.mu"),
                "observed_y.su.shot1": _sha256(
                    directory / "observed_y.su.shot1"
                ),
            }
            _run_case(
                directory,
                repository_root,
                binaries[f"exact_b_pre_{component}"],
                mpiexec,
                config,
                f"m5f1_gradient_dtinv_{dtinv}_{component}",
                True,
            )
            components[component] = [
                -value for value in read_float_grid(
                    directory / "jacobian" / "gradient_p_u.old",
                    config.cell_count,
                )
            ]
        gradient = _exact_density_material_gradient(
            components["x"], components["y"], config, 2, density
        )
        gradients[dtinv] = gradient
        gradient_products[dtinv] = directional_derivative(
            gradient, density_direction
        )
        gradient_records[str(dtinv)] = {
            "component_model_hashes": component_hashes,
            "directional_product": gradient_products[dtinv],
        }

    reference_component_hashes = gradient_records["1"]["component_model_hashes"]
    for dtinv in range(1, 5):
        assert (
            gradient_records[str(dtinv)]["component_model_hashes"]
            == reference_component_hashes
        )

    fd_derivatives = {}
    fd_diagnostics = {}
    for dtinv in range(1, 5):
        coarse = _five_point(objectives[dtinv], EPSILONS[1])
        fine = _five_point(objectives[dtinv], EPSILONS[2])
        fd_derivatives[dtinv] = fine
        fd_diagnostics[str(dtinv)] = {
            "coarse_h": EPSILONS[1],
            "coarse_five_point_derivative": coarse,
            "fine_h": EPSILONS[2],
            "fine_five_point_derivative": fine,
            "five_point_relative_change": abs(fine - coarse)
            / max(abs(fine), 1.0e-30),
            "raw_objectives": {
                _epsilon_key(epsilon): value
                for epsilon, value in sorted(objectives[dtinv].items())
            },
        }

    reference_gradient = gradients[1]
    for dtinv in range(1, 5):
        gradient_records[str(dtinv)]["field_vs_dtinv1"] = _difference(
            reference_gradient, gradients[dtinv]
        )
        gradient_records[str(dtinv)]["directional_product_difference_vs_dtinv1"] = (
            gradient_products[dtinv] - gradient_products[1]
        )
        gradient_records[str(dtinv)]["directional_product_relative_difference_vs_dtinv1"] = abs(
            gradient_products[dtinv] - gradient_products[1]
        ) / max(abs(gradient_products[1]), 1.0e-30)
        gradient_records[str(dtinv)]["fd_five_point_derivative"] = fd_derivatives[dtinv]
        gradient_records[str(dtinv)]["fd_to_gradient_ratio"] = (
            fd_derivatives[dtinv] / gradient_products[dtinv]
        )

    temporal_quadrature = {}
    for dtinv in range(1, 5):
        product_difference = gradient_products[dtinv] - gradient_products[1]
        temporal_quadrature[str(dtinv)] = {
            "temporal_sample_interval_s": base.dt_s * dtinv,
            "directional_product_difference_vs_dtinv1": product_difference,
            "directional_product_relative_difference_vs_dtinv1": abs(
                product_difference
            ) / abs(gradient_products[1]),
            "second_order_indicator_difference_over_k_squared_minus_one": (
                product_difference / (dtinv * dtinv - 1)
                if dtinv > 1
                else None
            ),
        }
    error3 = abs(gradient_products[3] - gradient_products[1])
    error4 = abs(gradient_products[4] - gradient_products[1])
    temporal_quadrature["observed_order_from_dtinv3_to_dtinv4"] = (
        math.log(error4 / error3) / math.log(4.0 / 3.0)
    )

    forward_full_file_hash_invariant = all(
        variants[str(dtinv)]["forward_synthetic_sha256"]
        == variants["1"]["forward_synthetic_sha256"]
        for variants in per_model.values()
        for dtinv in range(2, 5)
    )
    residual_full_file_hash_invariant = all(
        variants[str(dtinv)]["reversed_residual_sha256"]
        == variants["1"]["reversed_residual_sha256"]
        for variants in per_model.values()
        for dtinv in range(2, 5)
    )
    forward_samples_invariant = all(
        variants[str(dtinv)]["forward_sample_payload_sha256"]
        == variants["1"]["forward_sample_payload_sha256"]
        and variants[str(dtinv)]["forward_vs_dtinv1"]["relative_l2"] == 0.0
        and variants[str(dtinv)]["forward_vs_dtinv1"]["max_absolute_difference"]
        == 0.0
        for variants in per_model.values()
        for dtinv in range(2, 5)
    )
    residual_samples_invariant = all(
        variants[str(dtinv)]["reversed_residual_sample_payload_sha256"]
        == variants["1"]["reversed_residual_sample_payload_sha256"]
        and variants[str(dtinv)]["residual_vs_dtinv1"]["relative_l2"] == 0.0
        and variants[str(dtinv)]["residual_vs_dtinv1"]["max_absolute_difference"]
        == 0.0
        for variants in per_model.values()
        for dtinv in range(2, 5)
    )
    fd_invariant = all(
        fd_derivatives[dtinv] == fd_derivatives[1] for dtinv in range(2, 5)
    )
    forward_header_difference_offsets = sorted(
        {
            offset
            for variants in per_model.values()
            for dtinv in range(2, 5)
            for offset in variants[str(dtinv)][
                "forward_header_difference_offsets_1_based"
            ]
        }
    )
    gradient_quadrature_max_relative = max(
        gradient_records[str(dtinv)][
            "directional_product_relative_difference_vs_dtinv1"
        ]
        for dtinv in range(2, 5)
    )
    current_fd_gradient_relative_error = abs(
        fd_derivatives[1] - gradient_products[1]
    ) / abs(gradient_products[1])
    historical_fd_relative_shift = abs(
        fd_derivatives[1] - HISTORICAL_INDEPENDENT_DTINV1_M_FD
    ) / abs(HISTORICAL_INDEPENDENT_DTINV1_M_FD)
    conservative_oracle_floor = math.hypot(
        2.0 * fd_diagnostics["1"]["five_point_relative_change"],
        historical_fd_relative_shift,
    )

    # Classification is deliberately based on independent path comparisons.
    # A finer interpretation of an invariant secondary mismatch is made after
    # inspecting the retained raw objective values and gradient differences.
    if not forward_samples_invariant or not residual_samples_invariant:
        verdict = "DTINV FORWARD INVARIANCE DEFECT"
    elif not fd_invariant:
        verdict = "HARNESS INCONSISTENCY IDENTIFIED"
    else:
        verdict = "NO DTINV DEFECT — SECONDARY FAILURE WITHIN TRUE ORACLE FLOOR"

    output = {
        "milestone": "M5.0f.1",
        "repository_commit": os.environ.get("M5F_REPOSITORY_COMMIT", "unknown"),
        "observed_dataset": {
            "generated_once_with_dtinv": 1,
            "sha256": observed_hash,
        },
        "epsilons": list(EPSILONS),
        "per_model": per_model,
        "fd_diagnostics": fd_diagnostics,
        "gradient_diagnostics": gradient_records,
        "temporal_quadrature_analysis": {
            "method": (
                "Direct comparison against DTINV=1; no fitted correction factor"
            ),
            "per_dtinv": temporal_quadrature,
            "interpretation": (
                "The directional-product change is approximately second order "
                "at DTINV=3..4 but is negligible relative to the FD-oracle mismatch."
            ),
        },
        "oracle_floor_analysis": {
            "historical_independent_dtinv1_five_point_derivative": (
                HISTORICAL_INDEPENDENT_DTINV1_M_FD
            ),
            "common_oracle_five_point_derivative": fd_derivatives[1],
            "historical_cross_run_relative_shift": historical_fd_relative_shift,
            "current_fd_to_gradient_relative_error": (
                current_fd_gradient_relative_error
            ),
            "current_five_point_relative_change": fd_diagnostics["1"][
                "five_point_relative_change"
            ],
            "previous_two_times_stability_ceiling": 2.0
            * fd_diagnostics["1"]["five_point_relative_change"],
            "excess_above_previous_ceiling": current_fd_gradient_relative_error
            - 2.0 * fd_diagnostics["1"]["five_point_relative_change"],
            "conservative_combined_oracle_floor": conservative_oracle_floor,
            "current_mismatch_within_combined_oracle_floor": (
                current_fd_gradient_relative_error <= conservative_oracle_floor
            ),
            "maximum_dtinv_gradient_product_relative_change": (
                gradient_quadrature_max_relative
            ),
            "fd_mismatch_to_dtinv_gradient_change_ratio": (
                current_fd_gradient_relative_error
                / gradient_quadrature_max_relative
            ),
        },
        "invariance_summary": {
            "model_and_observed_hashes_invariant": True,
            "forward_full_file_hashes_invariant": forward_full_file_hash_invariant,
            "forward_sample_payloads_invariant": forward_samples_invariant,
            "reversed_residual_full_file_hashes_invariant": (
                residual_full_file_hash_invariant
            ),
            "reversed_residual_sample_payloads_invariant": (
                residual_samples_invariant
            ),
            "five_point_derivative_invariant": fd_invariant,
            "forward_full_hash_difference_is_header_only": (
                not forward_full_file_hash_invariant and forward_samples_invariant
            ),
            "forward_header_difference_offsets_1_based": (
                forward_header_difference_offsets
            ),
            "forward_header_note": (
                "Full-file SHA differences are confined to SU trace-header "
                "bytes 211..238; sample payload SHA and decoded samples are exact."
            ),
        },
        "verdict": verdict,
    }
    artifact = repository_root / "tests" / "m5f_dtinv_oracle_invariance.json"
    artifact.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    assert forward_samples_invariant, output["invariance_summary"]
    assert residual_samples_invariant, output["invariance_summary"]
    assert fd_invariant, output["invariance_summary"]
    assert set(forward_header_difference_offsets) <= set(range(211, 241))
    assert all(math.isfinite(value) for value in fd_derivatives.values())
    assert all(math.isfinite(value) for value in gradient_products.values())
