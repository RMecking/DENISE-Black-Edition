# Test layout

- `cases/`: deterministic input/model/geometry generators.
- `physics/`: analytical black-box physics checks.
- `regression/`: future stored-reference regression cases.
- `utilities/`: subprocess, metadata, seismogram, picking, and metric helpers.
- `conftest.py`: executable and MPI launcher discovery.

The physics suite contains independent homogeneous elastic SH and P/SV cases.
The P/SV suite covers P and SV velocity, projected polarization, source
symmetry, Gxx reciprocity, and both-component MPI reproducibility.
The CPML suite uses paired compact/reference domains to measure reflections at
all four sides for SH, P and SV waves, including oblique, corner, disabled-CPML,
and MPI-decomposition checks.
M3 adds P/SV free-surface and elastic-interface cases. The current M3 branch
intentionally retains a failing normal-SV interface timing assertion for
independent review; see `docs/verification.md`. SH free-surface propagation is
not claimed because the current SH stepping path does not apply that boundary.

See `docs/verification.md` for installation, commands, tolerances, generated
artifacts, and failure inspection.

Use `--require-denise` for verification or CI. Development runs may omit the
flag to allow explicit skips when the executable or MPI launcher is unavailable.
