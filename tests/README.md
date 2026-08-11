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
M3 adds P/SV free-surface and elastic-interface cases. Its staggered-grid
geometry helper maps each injected or sampled field to its physical position
and collocates oblique `vx`/`vy` measurements before vector projection. SH
free-surface propagation is not claimed because the current SH stepping path
does not apply that boundary. See `docs/verification.md` for the review history.
M4 adds homogeneous viscoelastic Q-input generators, one SH repeatability test,
and independent SH `MODE=0` Qs, P/SV Qp, and P/SV Qs sensitivity guards. M4.1
repairs those three defects, so all guards are now normal mandatory tests with
the original relative-L2 threshold of `1e-3`; their former strict-XFAIL markers
have been removed. Crashes, missing outputs, NaN/Inf, shape errors, hash
failures, and insufficient Q sensitivity are normal failures. The SH result
does not cover the separate viscoelastic FWI path. Mandatory mode still rejects
every integration-test skip.

See `docs/verification.md` for installation, commands, tolerances, generated
artifacts, and failure inspection.

Use `--require-denise` for verification or CI. Development runs may omit the
flag to allow explicit skips when the executable or MPI launcher is unavailable.
