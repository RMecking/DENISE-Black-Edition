from __future__ import annotations

from pathlib import Path
from typing import Callable

from tests.utilities.runner import result_summary, run_denise
from tests.utilities.seismogram import all_finite, read_ascii_seismograms, signal_energy


def run_psv_case(
    directory: Path,
    *,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
    config,
    generator: Callable,
    nprocx: int = 1,
    nprocy: int = 1,
) -> tuple[list[list[float]], list[list[float]]]:
    generator(directory, config=config, nprocx=nprocx, nprocy=nprocy)
    metadata = config.as_metadata() | {"nprocx": nprocx, "nprocy": nprocy}
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=denise_binary,
        mpiexec=mpiexec,
        ranks=nprocx * nprocy,
        configuration=metadata,
    )
    assert result.returncode == 0, result_summary(result)
    components = []
    for name in ("vx", "vy"):
        output = directory / "su" / f"homogeneous_{name}.asc.shot1"
        assert output.is_file() and output.stat().st_size > 0
        traces = read_ascii_seismograms(output, config.receiver_count, config.samples_per_trace)
        assert all_finite(traces)
        components.append(traces)
    assert signal_energy(
        sample for component in components for trace in component for sample in trace
    ) > 0.0
    return components[0], components[1]
