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

M4.2 adds quantitative SH-only rheology verification. Mandatory cases parse
DENISE's own stdout echo to require effective `MODE=0`, `PHYSICS=5`, and exact
`L`/`FL`; test elastic/high-Q convergence; fit transfer-function attenuation
and phase versus distance; and retain an exact repeat. The optional
`extended` marker covers the first multiple-relaxation (`L>1`) experiment.
Run mandatory physics with `-m 'not extended'`; run the additional experiment
with `-m extended`. Review rework demonstrated with a broadband synthetic
known-answer test that the original 0.22 s Hann gate biased the phase slopes.
The production estimator now uses a receiver-centred 0.40 s Tukey gate with
`alpha=0.2`; the old estimator remains in the run metrics as a diagnostic.
Frequencies 6 and 14 Hz are the mandatory dispersion pair because their
theoretical phase accumulation across the 400 m aperture is better conditioned
than at 8/12 Hz. All five frequencies remain reported, and the quantitative
phase tolerance remains 20%. The calibrated L=1 suite is green; the generic
L=3 experiment remains unexecuted pending independent review.

The M4.2 L>1 follow-up supersedes the arbitrary generic L=3 case with a
historical production-like L=4 audit using `FL=(2.7105,12.2792,68.1930,
265.2297) Hz` and optimized `tau=0.0386`. Pure-Python tests independently
reproduce the recovered MATLAB `qstd.m` definition and cross-check it against
the complex-modulus implementation. The mandatory black-box diagnostic
compares nominal `.qs=30` with the compensating `.qs=2/0.0386`. It preserves a
confirmed external-Q parameterization inconsistency as review evidence; it does
not alter the model readers or solver. The former generic L=3 case remains
marked `extended` and unexecuted.

See `docs/verification.md` for installation, commands, tolerances, generated
artifacts, and failure inspection.

Use `--require-denise` for verification or CI. Development runs may omit the
flag to allow explicit skips when the executable or MPI launcher is unavailable.
