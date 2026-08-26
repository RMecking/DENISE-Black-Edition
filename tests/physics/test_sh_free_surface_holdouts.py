"""M6.1d forward-only elastic SH free-surface hold-outs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from tests.cases.sh_free_surface_holdouts import (
    SHFreeSurfaceHoldout,
    corner_holdout,
    corner_wide_reference,
    generate_holdout_case,
    heterogeneous_holdout,
)
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import read_ascii_seismograms
from tests.utilities.sh_free_surface import (
    evaluate_surface_boundary,
    finite_nonzero,
    holberg_coefficients,
    normalized_correlation,
    normalized_surface_residuals,
    relative_l2,
    stability_modulation_limit,
    surface_roundoff_limits,
)
from tests.utilities.sh_free_surface_runtime import (
    denise_ricker_reference,
    post_source_quarters,
)


pytestmark = [pytest.mark.integration, pytest.mark.extended]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_diagnostics(path: Path, *, config, metadata) -> tuple[dict[str, object], list[float]]:
    assert path.is_file(), "M6.1d requires the frozen diagnostic instrumentation"
    with path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    assert len(source) == config.samples_per_trace
    rows = [
        {
            "timestep": int(row["timestep"]),
            "max_abs_syz0": float(row["max_abs_syz0"]),
            "max_abs_dplus_vz0": float(row["max_abs_dplus_vz0"]),
            "max_vz_parity_residual": float(row["max_vz_parity_residual"]),
            "max_syz_parity_residual": float(row["max_syz_parity_residual"]),
            "max_abs_interior_stress": float(row["max_abs_interior_stress"]),
            "max_impedance_vz": float(row["max_impedance_vz"]),
            "max_abs_dx_vz": float(row["max_abs_dx_vz"]),
            "max_abs_vz": float(row["max_abs_vz"]),
            "centered_energy": float(row["centered_energy"]),
        }
        for row in source
    ]
    assert [row["timestep"] for row in rows] == list(
        range(1, config.samples_per_trace + 1)
    )
    assert all(
        math.isfinite(value)
        for row in rows
        for key, value in row.items()
        if key != "timestep"
    )

    maxima = {
        key: max(row[key] for row in rows)
        for key in (
            "max_abs_syz0",
            "max_abs_dplus_vz0",
            "max_vz_parity_residual",
            "max_syz_parity_residual",
            "max_abs_interior_stress",
            "max_impedance_vz",
            "max_abs_dx_vz",
            "max_abs_vz",
        )
    }
    normalized_syz, normalized_dplus = normalized_surface_residuals(
        max_abs_syz0=maxima["max_abs_syz0"],
        max_abs_dplus_vz0=maxima["max_abs_dplus_vz0"],
        max_abs_interior_stress=maxima["max_abs_interior_stress"],
        max_impedance_vz=maxima["max_impedance_vz"],
        max_abs_dx_vz=maxima["max_abs_dx_vz"],
        max_abs_vz=maxima["max_abs_vz"],
        f95_hz=metadata["source_spectrum"]["f95_hz"],
        vs_m_s=config.vs_m_s,
    )
    syz_limit, dplus_limit = surface_roundoff_limits(
        holberg_coefficients(config.fd_order)
    )
    report = maxima | {
        "normalized_physical_traction_residual": normalized_syz,
        "normalized_image_closure_residual": normalized_dplus,
        "physical_traction_limit": syz_limit,
        "image_closure_limit": dplus_limit,
        "minimum_centered_energy": min(row["centered_energy"] for row in rows),
        "maximum_centered_energy": max(row["centered_energy"] for row in rows),
    }
    return report, [row["centered_energy"] for row in rows]


def _stability(energies: list[float], *, config, metadata) -> dict[str, float | int]:
    reference = denise_ricker_reference(
        nt=config.samples_per_trace,
        dt_s=config.dt_s,
        frequency_hz=config.source_frequency_hz,
        amplitude=1.0,
        timeshift_s=0.0,
        quellart=1,
        n_order=0,
    )
    assert reference.n_off == 1257
    quarters = post_source_quarters(nt=config.samples_per_trace, n_off=reference.n_off)

    def maximum(first: int, last: int) -> float:
        return max(energies[first - 1 : last])

    active_max = maximum(1, reference.n_off)
    post_max = maximum(reference.n_off + 1, config.samples_per_trace)
    q1_max = maximum(*quarters.inclusive_bounds[0])
    q4_max = maximum(*quarters.inclusive_bounds[3])
    delta_e = stability_modulation_limit(
        dt_s=config.dt_s,
        f95_hz=metadata["source_spectrum"]["f95_hz"],
        coefficients=holberg_coefficients(config.fd_order),
    )
    assert post_max <= (1.0 + delta_e) * active_max
    assert q4_max <= (1.0 + delta_e) * q1_max
    return {
        "n_off": reference.n_off,
        "n_post": config.samples_per_trace - reference.n_off,
        "delta_e": delta_e,
        "active_max": active_max,
        "post_max": post_max,
        "q1_max": q1_max,
        "q4_max": q4_max,
        "post_to_active": post_max / active_max,
        "q4_to_q1": q4_max / q1_max,
    }


def _run_holdout(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    holdout: SHFreeSurfaceHoldout,
    nprocx: int = 1,
    nprocy: int = 1,
) -> dict[str, object]:
    config = generate_holdout_case(
        directory, holdout=holdout, nprocx=nprocx, nprocy=nprocy
    )
    metadata = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration=metadata,
        timeout_seconds=300.0,
    )
    assert result.returncode == 0, result_summary(result)
    output = directory / "su" / "homogeneous_vz.asc.shot1"
    assert output.is_file() and output.stat().st_size > 0
    traces = read_ascii_seismograms(
        output, config.receiver_count, config.samples_per_trace
    )
    assert len(traces) == config.receiver_count
    assert all(finite_nonzero(trace) for trace in traces)

    diagnostic, energies = _parse_diagnostics(
        directory / "m61b_diagnostics.csv", config=config, metadata=metadata
    )
    stability = _stability(energies, config=config, metadata=metadata)
    boundary = None
    if holdout.free_surface:
        accepted = evaluate_surface_boundary(
            normalized_physical_traction=diagnostic[
                "normalized_physical_traction_residual"
            ],
            physical_traction_limit=diagnostic["physical_traction_limit"],
            max_velocity_parity_residual=diagnostic[
                "max_vz_parity_residual"
            ],
            max_stress_parity_residual=diagnostic[
                "max_syz_parity_residual"
            ],
            normalized_image_closure=diagnostic[
                "normalized_image_closure_residual"
            ],
            image_closure_limit=diagnostic["image_closure_limit"],
        )
        assert accepted.all_pass, asdict(accepted)
        boundary = asdict(accepted) | {"all_pass": accepted.all_pass}

    return {
        "holdout": holdout.name,
        "free_surface": holdout.free_surface,
        "fd_order": config.fd_order,
        "decomposition": [nprocx, nprocy],
        "returncode": result.returncode,
        "input_sha256": metadata["input_sha256"],
        "seismogram_sha256": _sha256(output),
        "traces": traces,
        "diagnostics": diagnostic,
        "boundary": boundary,
        "stability": stability,
    }


def _public_run(run: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in run.items() if key != "traces"}


def test_strong_near_surface_heterogeneity(
    tmp_path, repository_root, denise_binary, mpiexec
):
    fd4 = _run_holdout(
        tmp_path / "fd4_1x1",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        holdout=heterogeneous_holdout(4),
    )
    variants = {"1x1": (1, 1), "2x1": (2, 1), "1x2": (1, 2)}
    fd12 = {
        label: _run_holdout(
            tmp_path / f"fd12_{label}",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            holdout=heterogeneous_holdout(12),
            nprocx=shape[0],
            nprocy=shape[1],
        )
        for label, shape in variants.items()
    }
    assert fd12["1x1"]["input_sha256"] == fd12["2x1"]["input_sha256"] == fd12["1x2"]["input_sha256"]

    comparisons = {}
    for label in ("2x1", "1x2"):
        reference = fd12["1x1"]["traces"][0]
        candidate = fd12[label]["traces"][0]
        comparisons[label] = {
            "relative_l2": relative_l2(reference, candidate),
            "normalized_correlation": normalized_correlation(reference, candidate),
            "byte_identical": fd12["1x1"]["seismogram_sha256"]
            == fd12[label]["seismogram_sha256"],
        }
        assert comparisons[label]["relative_l2"] <= 1.0e-6
        assert comparisons[label]["normalized_correlation"] >= 0.999999

    absorbing = _run_holdout(
        tmp_path / "fd12_absorbing",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        holdout=heterogeneous_holdout(12, free_surface=False),
    )
    assert absorbing["boundary"] is None
    report = {
        "scope": "boundary-state correctness, health/stability, and MPI only",
        "new_wavefield_oracle": None,
        "fd4": _public_run(fd4),
        "fd12": {label: _public_run(run) for label, run in fd12.items()},
        "mpi_comparisons": comparisons,
        "free_surface_0_control": _public_run(absorbing),
    }
    (tmp_path / "m61d_heterogeneous.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("M61D_HETEROGENEOUS " + json.dumps(report, sort_keys=True))


def test_lateral_cpml_free_surface_corners(
    tmp_path, repository_root, denise_binary, mpiexec
):
    cases = {}
    comparisons = {}
    for side in ("left", "right"):
        cases[side] = {
            "1x1": _run_holdout(
                tmp_path / f"{side}_1x1",
                repository_root=repository_root,
                denise_binary=denise_binary,
                mpiexec=mpiexec,
                holdout=corner_holdout(side),
            ),
            "2x1": _run_holdout(
                tmp_path / f"{side}_2x1",
                repository_root=repository_root,
                denise_binary=denise_binary,
                mpiexec=mpiexec,
                holdout=corner_holdout(side),
                nprocx=2,
            ),
        }
        assert cases[side]["1x1"]["input_sha256"] == cases[side]["2x1"]["input_sha256"]
        reference = cases[side]["1x1"]["traces"][0]
        candidate = cases[side]["2x1"]["traces"][0]
        comparisons[side] = {
            "relative_l2": relative_l2(reference, candidate),
            "normalized_correlation": normalized_correlation(reference, candidate),
            "byte_identical": cases[side]["1x1"]["seismogram_sha256"]
            == cases[side]["2x1"]["seismogram_sha256"],
        }
        assert comparisons[side]["relative_l2"] <= 1.0e-6
        assert comparisons[side]["normalized_correlation"] >= 0.999999

    wide_references = {
        side: _run_holdout(
            tmp_path / f"{side}_wide_reference",
            repository_root=repository_root,
            denise_binary=denise_binary,
            mpiexec=mpiexec,
            holdout=corner_wide_reference(side),
        )
        for side in ("left", "right")
    }
    wide_domain_diagnostics = {}
    for side in ("left", "right"):
        narrow_trace = cases[side]["1x1"]["traces"][0]
        wide_trace = wide_references[side]["traces"][0]
        wide_domain_diagnostics[side] = {
            "relative_l2": relative_l2(narrow_trace, wide_trace),
            "normalized_correlation": normalized_correlation(
                narrow_trace, wide_trace
            ),
            "acceptance": None,
            "purpose": "diagnostic only; no frozen wide-domain tolerance",
        }

    absorbing = _run_holdout(
        tmp_path / "left_absorbing",
        repository_root=repository_root,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        holdout=corner_holdout("left", free_surface=False),
    )
    assert absorbing["boundary"] is None
    report = {
        "scope": "exact boundary, health/stability, and MPI only",
        "wide_domain_waveform_tolerance": None,
        "left_right_symmetry_oracle": None,
        "cases": {
            side: {label: _public_run(run) for label, run in variants.items()}
            for side, variants in cases.items()
        },
        "mpi_comparisons": comparisons,
        "wide_references": {
            side: _public_run(run) for side, run in wide_references.items()
        },
        "wide_domain_diagnostics": wide_domain_diagnostics,
        "free_surface_0_control": _public_run(absorbing),
    }
    (tmp_path / "m61d_corners.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("M61D_CORNERS " + json.dumps(report, sort_keys=True))
