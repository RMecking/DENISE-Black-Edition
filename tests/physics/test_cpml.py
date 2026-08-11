from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from tests.cases.cpml import (
    CPMLPair,
    corner_sh_pair,
    normal_p_pair,
    normal_sh_pair,
    normal_sv_pair,
    oblique_sh_pair,
)
from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case as generate_psv
from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case as generate_sh
from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import (
    all_finite,
    cpml_reflection_metrics,
    normalized_correlation,
    read_ascii_seismograms,
    relative_l2,
    signal_energy,
    time_interval,
)


pytestmark = pytest.mark.integration

# Paired runs have different global loop bounds and single-precision models.
# Requiring <0.1% direct-wave mismatch verifies equivalence while remaining
# much stricter than the 3.16% amplitude ratio represented by -30 dB.
DIRECT_PAIR_RELATIVE_L2_MAX = 1.0e-3
DIRECT_PAIR_CORRELATION_MIN = 0.999999


def _run(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    config: HomogeneousSHConfig | HomogeneousPSVConfig,
    component: str,
    nprocx: int = 1,
    nprocy: int = 1,
) -> list[float]:
    if isinstance(config, HomogeneousSHConfig):
        generate_sh(directory, config=config, nprocx=nprocx, nprocy=nprocy)
    else:
        generate_psv(directory, config=config, nprocx=nprocx, nprocy=nprocy)
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration=config.as_metadata() | {"nprocx": nprocx, "nprocy": nprocy},
    )
    assert result.returncode == 0, result_summary(result)
    output = directory / "su" / f"homogeneous_{component}.asc.shot1"
    assert output.is_file() and output.stat().st_size > 0
    traces = read_ascii_seismograms(output, config.receiver_count, config.samples_per_trace)
    assert all_finite(traces)
    assert signal_energy(traces[0]) > 0.0
    return traces[0]


def _measure_pair(
    directory: Path,
    pair: CPMLPair,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
) -> dict[str, object]:
    compact = _run(
        directory / "compact", repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=pair.compact, component=pair.component,
    )
    reference = _run(
        directory / "reference", repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=pair.reference, component=pair.component,
    )
    measured = cpml_reflection_metrics(
        compact, reference, dt_s=pair.compact.dt_s,
        direct_window_s=pair.direct_window_s, reflection_window_s=pair.reflection_window_s,
    )
    compact_direct = time_interval(
        compact, start_s=pair.direct_window_s[0], stop_s=pair.direct_window_s[1],
        dt_s=pair.compact.dt_s,
    )
    reference_direct = time_interval(
        reference, start_s=pair.direct_window_s[0], stop_s=pair.direct_window_s[1],
        dt_s=pair.compact.dt_s,
    )
    direct_relative_l2 = relative_l2([reference_direct], [compact_direct])
    direct_correlation = normalized_correlation([reference_direct], [compact_direct])
    metrics = {
        "case": pair.name,
        "component": pair.component,
        "acceptance_db": pair.acceptance_db,
        "direct_window_s": list(pair.direct_window_s),
        "reflection_window_s": list(pair.reflection_window_s),
        "compact_to_reference_translation_m": list(pair.compact_to_reference_translation_m),
        "direct_relative_l2": direct_relative_l2,
        "direct_normalized_correlation": direct_correlation,
        "direct_relative_l2_tolerance": DIRECT_PAIR_RELATIVE_L2_MAX,
        "direct_correlation_tolerance": DIRECT_PAIR_CORRELATION_MIN,
        **asdict(measured),
    }
    (directory / "cpml_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert direct_relative_l2 <= DIRECT_PAIR_RELATIVE_L2_MAX
    assert direct_correlation >= DIRECT_PAIR_CORRELATION_MIN
    return metrics


def test_sh_cpml_normal_incidence_all_sides(
    tmp_path, repository_root, denise_binary, mpiexec
):
    metrics = {}
    for side in ("left", "right", "y_min", "y_max"):
        pair = normal_sh_pair(side)
        result = _measure_pair(
            tmp_path / side, pair, repository_root=repository_root,
            denise_binary=denise_binary, mpiexec=mpiexec,
        )
        metrics[side] = result
        assert result["reflection_db"] <= pair.acceptance_db

    symmetry_tolerance_db = 3.0
    symmetry = {
        "left_right_difference_db": abs(
            metrics["left"]["reflection_db"] - metrics["right"]["reflection_db"]
        ),
        "y_min_y_max_difference_db": abs(
            metrics["y_min"]["reflection_db"] - metrics["y_max"]["reflection_db"]
        ),
        "tolerance_db": symmetry_tolerance_db,
    }
    (tmp_path / "sh_side_symmetry.json").write_text(
        json.dumps(symmetry, indent=2) + "\n", encoding="utf-8"
    )
    assert symmetry["left_right_difference_db"] <= symmetry_tolerance_db
    assert symmetry["y_min_y_max_difference_db"] <= symmetry_tolerance_db


@pytest.mark.parametrize("pair_factory", [oblique_sh_pair, corner_sh_pair], ids=["oblique", "corner"])
def test_sh_cpml_non_normal_incidence(
    tmp_path, repository_root, denise_binary, mpiexec, pair_factory
):
    pair = pair_factory()
    metrics = _measure_pair(
        tmp_path / pair.name, pair, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    assert metrics["reflection_db"] <= pair.acceptance_db


@pytest.mark.parametrize("axis", ["x", "y"])
def test_psv_cpml_p_normal_incidence(
    tmp_path, repository_root, denise_binary, mpiexec, axis
):
    pair = normal_p_pair(axis)
    metrics = _measure_pair(
        tmp_path / pair.name, pair, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    assert metrics["reflection_db"] <= pair.acceptance_db


def test_psv_cpml_sv_normal_incidence(tmp_path, repository_root, denise_binary, mpiexec):
    pair = normal_sv_pair()
    metrics = _measure_pair(
        tmp_path / pair.name, pair, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    assert metrics["reflection_db"] <= pair.acceptance_db


def test_sh_cpml_disabled_negative_control(tmp_path, repository_root, denise_binary, mpiexec):
    pair = normal_sh_pair("right")
    reference = _run(
        tmp_path / "reference", repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=pair.reference, component=pair.component,
    )
    enabled = _run(
        tmp_path / "enabled", repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=pair.compact, component=pair.component,
    )
    disabled_config = replace(pair.compact, absorbing_width_gridpoints=0)
    disabled = _run(
        tmp_path / "disabled", repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=disabled_config, component=pair.component,
    )
    kwargs = {
        "dt_s": pair.compact.dt_s,
        "direct_window_s": pair.direct_window_s,
        "reflection_window_s": pair.reflection_window_s,
    }
    enabled_metrics = cpml_reflection_metrics(enabled, reference, **kwargs)
    disabled_metrics = cpml_reflection_metrics(disabled, reference, **kwargs)
    metrics = {
        "enabled": asdict(enabled_metrics),
        "disabled": asdict(disabled_metrics),
        "minimum_degradation_db": 15.0,
        "observed_degradation_db": disabled_metrics.reflection_db - enabled_metrics.reflection_db,
    }
    (tmp_path / "negative_control_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    assert enabled_metrics.reflection_db <= pair.acceptance_db
    assert metrics["observed_degradation_db"] >= metrics["minimum_degradation_db"]


def test_sh_cpml_mpi_reproducibility(tmp_path, repository_root, denise_binary, mpiexec):
    pair = normal_sh_pair("right")
    reference = _run(
        tmp_path / "mpi_1x1", repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=pair.compact, component=pair.component,
    )
    metrics = {}
    for label, nprocx, nprocy in (("2x1", 2, 1), ("1x2", 1, 2), ("2x2", 2, 2)):
        variant = _run(
            tmp_path / f"mpi_{label}", repository_root=repository_root,
            denise_binary=denise_binary, mpiexec=mpiexec, config=pair.compact,
            component=pair.component, nprocx=nprocx, nprocy=nprocy,
        )
        rel_error = relative_l2([reference], [variant])
        correlation = normalized_correlation([reference], [variant])
        metrics[label] = {
            "mpi_ranks": nprocx * nprocy,
            "relative_l2": rel_error,
            "normalized_correlation": correlation,
        }
        assert rel_error <= 1.0e-5
        assert correlation >= 0.999999
    (tmp_path / "cpml_mpi_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
