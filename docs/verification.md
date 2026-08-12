# DENISE verification harness

## Purpose

The `tests/` harness exercises DENISE as an external MPI program. It generates
inputs, runs the unmodified executable, reads seismograms back into Python, and
checks numerical behaviour without depending on C solver internals. M0 provides
a homogeneous elastic SH case and M1 adds independent homogeneous isotropic
elastic P/SV baselines. These are regression baselines for later refactoring,
and M2 adds quantitative paired-domain CPML reflection tests. These are
regression baselines for later refactoring, not a complete physics validation
suite. Python 3.9 or newer is required.

Install Python and build dependencies, then build DENISE:

```bash
sudo apt install build-essential openmpi-bin libopenmpi-dev libfftw3-dev python3-pytest
make -C libcseife
make -C src denise
```

Run all tests from the repository root:

```bash
python3 -m pytest -v
```

For release verification and CI, make DENISE and MPI mandatory:

```bash
python3 -m pytest -v --require-denise
```

Run one physics family or point at a non-default executable:

```bash
python3 -m pytest tests/physics/test_homogeneous_sh.py -vv -s
python3 -m pytest tests/physics/test_homogeneous_psv.py -vv -s --require-denise
DENISE_BIN=/other/bin/denise python3 -m pytest tests/physics/test_homogeneous_sh.py -vv
```

`MPIEXEC` changes the launcher and `MPIEXEC_FLAGS` supplies extra launcher
arguments. In the default development mode, an absent `bin/denise` or launcher
skips integration tests with an explicit reason. With `--require-denise`, either
condition fails the run, and any skipped integration test is converted to a
failure. CI always uses this mandatory mode.

## Homogeneous SH case

`tests/cases/homogeneous_sh.py` generates every run in a fresh directory:

| Quantity | Value |
| --- | ---: |
| Grid | 200 x 120 |
| Grid spacing | 10 m |
| Simulation time | 0.55 s |
| Time step | 0.0005 s |
| Vs | 2000 m/s |
| Density | 2000 kg/m3 |
| Source | (700 m, 600 m) |
| Ricker frequency | 10 Hz |
| Receivers | x=900, 1000, 1100, 1200, 1300 m; y=600 m |
| FD order / coefficients | 8 / Holberg 0.1% setting |
| CPML | 15 grid points, 2000 m/s damping velocity, 10 Hz |
| MPI decompositions | 1 x 1, 2 x 1, 1 x 2, and 2 x 2 |

The Courant number `Vs * dt / dh` is 0.1. Even using 2.5 times the nominal
Ricker frequency as a conservative bandwidth estimate, the shortest wavelength
has eight grid points. These choices are comfortably inside the stability and
dispersion checks printed by DENISE for the eighth-order Holberg operator.
Sources and receivers are outside the CPML, and the analysis window ends before
boundary returns can overtake the direct arrivals.

The model files are native single-precision floats in DENISE's x-major/y-inner
order. Geometry and the 115-record positional parameter file are generated as
text. Seismograms use `SEIS_FORMAT=2` so the reader has no dependency on
Seismic Unix or ObsPy.

## Checks and tolerances

Program-health checks require return code zero, a non-empty output file with
the expected number of samples, and finite values throughout.

For the travel-time check, each trace is converted to a moving mean of absolute
amplitude over one quarter of the nominal source period. The first sample at
5% of that trace's smoothed maximum is the observed first break. The same picker
is applied to the exact discretized DENISE Ricker wavelet; its causal source
delay is subtracted before comparison with `distance / Vs`.

The diagnostic absolute tolerance is:

```text
2 * dt + 0.25 / source_frequency = 0.026 s
```

Two samples cover time discretization and indexing. A quarter period covers
the finite-bandwidth resolution of an onset pick; the spatial discretization
is already constrained by the Holberg 0.1% setting and eight points per
conservative shortest wavelength. This tolerance was defined from sampling and
bandwidth, not from the measured result.

The propagation-velocity assertion is deliberately stronger. Raw first-break
picks at all receiver offsets are fit by ordinary least squares to
`t_pick(r) = t0 + r / Vs_fit`. The free intercept removes the constant source
wavelet and picker delay from the tested slope. The test requires the relative
error in `Vs_fit` to be at most 1%, and it also requires the maximum fit residual
to be no more than two timesteps (1 ms).

The 1% velocity tolerance follows from the numerical resolution. Picks are
quantized at 0.5 ms over a 0.2 s differential travel-time aperture; a one-sample
end-to-end perturbation is about 0.25% in slope. All source and receiver
coordinates lie exactly on the 10 m grid, so geometry introduces no rounding
error in this case. The eighth-order Holberg 0.1% setting and eight grid points
per conservative shortest wavelength keep expected FD dispersion well below
1%. The remaining margin covers picker quantization and small finite-bandwidth
or decomposition effects without inheriting the permissive 26 ms absolute
first-break tolerance.

MPI reproducibility compares complete 2 x 1, 1 x 2, and 2 x 2 seismogram arrays
against the 1 x 1 reference. Every variant requires relative L2 error at most
`1e-5` and normalized correlation at least `0.999999`. The four-rank 2 x 2 case
remains in the quick suite because the complete local suite takes only a few
seconds. Bitwise identity is neither tested nor required.

## Homogeneous elastic P/SV cases

`tests/cases/homogeneous_psv.py` generates separate `vp`, `vs`, and `rho`
single-precision model files plus every geometry and positional configuration
file needed by each run. No generated model, seismogram, or runtime artifact is
committed.

| Quantity | Value |
| --- | ---: |
| Grid | 200 x 200 |
| Grid spacing | 10 m |
| Simulation time | 0.55 s |
| Time step | 0.0004 s |
| Vp / Vs | 3000 / 1800 m/s |
| Density | 2000 kg/m3 |
| Default source | (1000 m, 1000 m) |
| Ricker frequency | 10 Hz |
| FD order / coefficients | 8 / Holberg 0.1% setting |
| CPML | 15 grid points, 3000 m/s damping velocity, 10 Hz |
| MPI decompositions | 1 x 1, 2 x 1, 1 x 2, and 2 x 2 |

The `Vp * dt / dh` Courant number is 0.12. Taking 2.5 times the nominal
frequency as a conservative bandwidth gives 7.2 grid points per shortest S
wavelength. DENISE's stability and dispersion checks accept this FD8/Holberg
configuration. Sources and receivers are on exact grid coordinates and at
least 250 m inside the non-CPML domain. Direct-arrival analysis windows finish
before relevant boundary returns can contaminate them.

### P and SV velocity

The P case uses a type-1 explosive source and five horizontal `vx` receivers at
200, 300, 400, 500, and 600 m offset. Its first-break picker is the same 5% of
quarter-period-smoothed amplitude picker used by the SH case. Raw picks are fit
to `t_pick = t0 + r / Vp_fit`; the fit must be within 1% of 3000 m/s and every
residual within `2*dt = 0.8 ms`.

The SV case uses the same five offsets on the 3-4-5 direction `(0.6, 0.8)`.
A type-4 rotated point force has direction `(-0.8, 0.6)`, exactly transverse
to those rays according to DENISE's `(-sin(azimuth), cos(azimuth))` convention.
Each recorded `(vx, vy)` vector is projected onto the ray-transverse direction.
The pick is the maximum absolute transverse velocity in a predeclared interval
from `1.5/f + r/Vp + 0.25/f` through `1.5/f + r/Vs + 0.5/f`. Thus every lower
bound lies 25 ms after the predicted P main-lobe peak; in particular, the
200 m P peak cannot enter the S search despite the modes being separated by
only 44.4 ms there. The upper bound retains 50 ms after the predicted S peak.
The interval comes entirely from analytical velocities and source bandwidth,
not from observed picks. The peak picks are independently fit to
`t_pick = t0 + r / Vs_fit`, again with a 1% velocity limit and `2*dt` residual
limit. Thus P and SV fits cannot pass if both modes propagate at one velocity.

At `dt=0.4 ms`, a one-sample end-to-end perturbation over either aperture is
substantially less than 1% slope error. Coordinates introduce no rounding, and
FD8/Holberg dispersion at 7.2 or more points per conservative wavelength is
expected below the 1% regression limit. The tolerance follows these numerical
resolutions, not the observed fitted values.

### Polarization

An x-directed type-2 point force and a receiver on the non-axis-aligned
`(300 m, 400 m)` ray excite both modes. For unit ray vector `n`, each velocity
sample is projected as `v_parallel = n dot v` and
`v_perpendicular = (-n_y, n_x) dot v`. Energy is the sum of squared samples in
`+/- 0.3/f` (30 ms) windows around `1.5/f + r/Vp` and `1.5/f + r/Vs`.
The P longitudinal/transverse energy ratio and SV transverse/longitudinal ratio
must each be at least 10. This tests vector polarization rather than merely
requiring a named component to be non-zero.

### Symmetry and reciprocity

For symmetry, an x-directed point force is recorded in `vx` at receivers 500 m
to either side of the source. For `Gxx`, both source-force and recording axes
are fixed, so opposite-ray traces retain polarity. Direct-P pick times must
agree within one timestep, correlation must be at least 0.999, and relative
amplitude error at most 1%. The ideal transverse `vy` component is zero;
staggered-grid leakage energy must remain below 0.1% of `vx` energy on each
side.

Reciprocity exchanges an x-force source and x-velocity receiver at
`A=(700 m,1000 m)` and `B=(1300 m,1000 m)`. The tested tensor relation is
`Gxx(B,A,t) = Gxx(A,B,t)`, not an arbitrary cross-component equality. In the
direct-P window, normalized correlation must be at least 0.99999, relative L2
at most `1e-4`, and relative amplitude error at most `1e-4`.

### P/SV MPI and observed baseline

The MPI case records both `vx` and `vy` for three non-axis-aligned receivers.
Each 2 x 1, 1 x 2, and 2 x 2 component array is compared with 1 x 1 using the
unchanged M0.1 limits: relative L2 at most `1e-5` and normalized correlation at
least `0.999999`.

The observed local baseline is distinct from the analytical requirements:

| Metric | Observed value |
| --- | ---: |
| P fitted velocity / relative error | 2990.43 m/s / 0.319% |
| P maximum fit residual | 0.160 ms |
| SV fitted velocity / relative error | 1810.28 m/s / 0.571% |
| SV maximum fit residual | 0.480 ms |
| P longitudinal dominance | 30.82 |
| SV transverse dominance | 210.55 |
| Symmetry pick difference | 0.000 ms |
| Symmetry correlation / amplitude error | 1.0 / 2.9e-9 |
| Reciprocity correlation / relative L2 | 1.0 / 2.61e-8 |
| All P/SV MPI comparisons | relative L2 0.0, correlation 1.0 |

Six P/SV integration tests take approximately 25 seconds in the documented
WSL development environment. Exact metrics are written as JSON beside the run
artifacts.

## CPML reflection verification

M2 measures boundary energy with paired black-box runs. The compact run lets a
wave encounter the selected 15-cell CPML and return to the receiver. Its
homogeneous reference enlarges the tested dimension from 120 to 360 cells so
the external reference boundary cannot return during analysis. Material,
`DH`, `DT`, FD8/Holberg coefficients, source function, source/receiver offsets,
and grid phase are identical. Because DENISE fixes its coordinate origin and
does not accept a negative model origin, left/`y_min` references cannot literally
retain their input coordinate numbers while moving those boundaries outward.
Every reference therefore translates source and receiver together by exactly
120 grid cells in the enlarged dimension. This is a change of computational
origin only: physical geometry and integer staggered-grid placement are
preserved. Each metrics file records that translation.

The paired direct windows must agree to relative L2 at most `1e-3` and
correlation at least `0.999999`. The L2 limit allows 0.1% single-precision
difference from different global loop bounds while remaining over 30 times
tighter than the 3.16% amplitude ratio represented by the normal-incidence
acceptance limit. Observed direct mismatch is also reported; it is not included
in the late residual definition.

For traces `d_compact` and `d_reference`, M2 computes

```text
d_reflection(t) = d_compact(t) - d_reference(t)
R = ||d_reflection||_2,late / ||d_reference||_2,direct
R_dB = 20 log10(R)
```

Both linear ratio and amplitude dB are stored. A more negative value is better.
Normal-incidence SH, P, and SV require `R_dB <= -30 dB`. The harder oblique and
corner cases use a predeclared `-25 dB` limit. Opposite SH sides may differ by
at most 3 dB. These are acceptance criteria, distinct from the observed values
below.

### Geometry and analytical windows

Normal cases use a 1200 m tested dimension and 2400 m transverse dimension.
The source and receiver are 100 m apart and face the selected boundary. The
inner and outer compact-boundary image paths are 1200 and 1500 m. Direct
windows span `1.5/f + r_direct/V +/- 0.75/f`; late windows span from
`1.5/f + r_inner/V - 0.5/f` through
`1.5/f + r_outer/V + 0.5/f`. Consequently SH uses direct `[0.125,0.275] s`
and late `[0.700,0.950] s`; P uses direct approximately
`[0.1083,0.2583] s` and late `[0.500,0.700] s`; SV uses direct approximately
`[0.1306,0.2806] s` and late `[0.7667,1.0333] s`. The nearest transverse
boundary returns occur after each simulation ends.

The oblique SH ray has direct vector `(-100,-600) m`; its right-boundary image
paths are 1341.64 and 1615.55 m, giving late `[0.77082,1.00777] s`.
The diagonal corner case has direct vector `(-100,-100) m`. Its conservative
late interval `[0.40414,0.83640] s` starts at the earliest inner side-PML image
path `sqrt(600^2+100^2) m` and ends after the outer-corner path
`sqrt(900^2+900^2) m`. It therefore includes the transition from individual
side damping into the region where x and y CPML overlap, rather than claiming
to isolate a mathematically pure corner reflection.

SH records `vz`. Explosive P tests record the longitudinal component (`vx` for
x incidence and `vy` for y incidence). The SV test uses a y-directed point
force on an x ray; the ray-transverse projection reduces exactly to recorded
`vy`, so P and SV metrics remain separate.

### CPML negative control, MPI, and observed baseline

The negative control sets the existing input parameter `FW=0`. Existing DENISE
branches guard CPML allocation and updates with `FW>0`, so this disables the
absorbing layer without source changes or solver instrumentation. The disabled
case must be at least 15 dB worse than the enabled case. This threshold tests
sensitivity rather than CPML quality; the enabled run must independently pass
the `-30 dB` criterion.

The representative right-going SH compact case runs as 1x1, 2x1, 1x2, and
2x2. In x-decomposed runs the outgoing wave and boundary return cross the
internal x interface; the source lies on the y partition line in y-decomposed
runs. Complete `vz` traces retain the M0/M1 limits: relative L2 at most `1e-5`
and normalized correlation at least `0.999999`.

Observed WSL baseline (acceptance values above were declared independently):

| Case | Reflection ratio | Level |
| --- | ---: | ---: |
| SH left normal | 0.000344021 | -69.27 dB |
| SH right normal | 0.000317903 | -69.95 dB |
| SH `y_min` normal | 0.000344021 | -69.27 dB |
| SH `y_max` normal | 0.000317903 | -69.95 dB |
| SH oblique right | 0.000730314 | -62.73 dB |
| SH corner/overlap | 0.001310140 | -57.65 dB |
| P normal x | 0.0003053 | -70.31 dB |
| P normal y | 0.0003049 | -70.31 dB |
| SV normal x | 0.0006380 | -63.90 dB |
| SH right, `FW=0` | 0.254429 | -11.89 dB |

Left/right and `y_min`/`y_max` differences are both 0.686 dB. Here `y_min` is
DENISE's physical upper side and the side on which `FREE_SURF=1` replaces CPML;
`y_max` is the physical lower side. Disabling CPML
degrades the metric by 58.07 dB. All three non-reference MPI layouts currently
have relative L2 `0.0` and correlation `1.0` (within floating display). The
eight CPML pytest cases, comprising 25 DENISE runs, take about 70 seconds in
the documented WSL environment and remain in the mandatory quick suite.

## M3 free-surface and elastic-interface verification

The first M3 review found that two timing failures were primarily test-geometry
errors: input coordinates had been treated as though all staggered fields were
collocated. The rework corrects physical coordinates and vector collocation
without changing any solver file or relaxing any tolerance.

### Capability audit and coordinate conventions

The active elastic P/SV loop calls `surface_elastic_PML_PSV(1, ...)`. That
routine states and implements the stress-free surface at `y=0.5*DH`, or 5 m for
the M3 grid. All analytical surface paths use that location. In contrast, the
free-surface blocks in both elastic and viscoelastic SH stepping functions are
commented out. M3 therefore makes no SH free-surface claim; this is an existing
capability gap, not repaired here.

All analytical intervals now must lie wholly between the first DENISE sample
at `dt` and the final sample at `NT*dt`. `time_interval()` raises instead of
silently clipping either edge, and `time_window()` delegates to that strict
implementation.

DENISE first converts positive input coordinates to one-based integer grid
indices with `iround(coordinate/DH)`, where `iround(x)=floor(x+0.5)`. For grid
indices `(i,j)`, the verification harness uses these physical positions:

| Field | Physical position |
| --- | --- |
| material parameters, `sxx`, `syy` | `((i-0.5) DH, (j-0.5) DH)` |
| `vx` | `(i DH, (j-0.5) DH)` |
| `vy` | `((i-0.5) DH, j DH)` |
| `sxy` | `(i DH, j DH)` |

Explosive sources are therefore located on the `sxx`/`syy` grid, x-force
sources on the `vx` grid, and each recorded velocity component on its native
grid. For oblique tests, three receiver inputs are used. Native `vx` is averaged
between the central and `+DH` y receivers, while native `vy` is averaged between
the central and `+DH` x receivers. Both resulting components occupy the central
`sxy` point before longitudinal/transverse projection. Pure-Python tests verify
all field mappings, nearest-index conversion, inverse coordinate construction,
the receiver stencil, and exact collocation of constant and linear fields.

The generated layered model assigns one-based rows 1 through 120 to the upper
medium and rows 121 through 240 to the lower medium. At `DH=10 m`, the adjacent
material centres are at 1195 m and 1205 m, so the nominal continuum interface
remains exactly `y=1200 m`. This is distinct from the staggered field positions
and from material averaging on those positions. DENISE's `av_rho()`
arithmetically averages adjacent densities onto velocity positions, while
`av_mue()` harmonically averages the four surrounding shear moduli onto the
`sxy` position. These averages are not described as moving the interface; an
effective shift would require a separate convergence experiment.

### Independent analytical methods and predeclared criteria

The Python-only analytical module constructs displacement polarizations and
tractions from isotropic Lamé parameters, then solves the two free-surface
traction equations or the four displacement/traction interface equations by
Gaussian elimination. Unit tests recover the normal free-surface P coefficient
`-1`, zero converted SV at normal incidence, and the normal impedance formula.
Stationary two-segment rays are solved independently and unit-tested against
Snell's law.

Normal reflected amplitudes use equal-path homogeneous calibration runs, so
source wavelet, two-dimensional spreading, and Green-function phase are not
blindly compared with plane-wave coefficients. Acceptance was declared as
15% relative amplitude error, correlation at least 0.98, and timing
`2*dt + 0.5%` of propagation time. Negative controls require the unwanted
residual to be at most `1e-3` of the contrast reflection for identical layers
and at most 0.10 of the free-surface reflection for an absorbing upper side.
Oblique plane-wave amplitudes are diagnostic only; mandatory checks use
Snell-law timing differences and projected polarization.

### Corrected observed results

For the normal free surface, source input `(1200,700) m` is the physical stress
point `(1195,695) m`; receiver input `(1200,1100) m` is the physical `vy` point
`(1195,1100) m`; and the surface is at 5 m. Direct and reflected paths are thus
405 m and 1785 m. Their analytical difference is exactly 1380 m / 3000 m/s =
0.4600 s, equal to the observed peak difference. The calibration was corrected
from input y=2190 m to 2180 m, giving an exact physical 1785 m path. Reflection
and calibration peaks both occur at 0.7536 s; amplitude error is 0.1681% and
correlation 0.999821. The `FREE_SURF=0` residual ratio is `5.45e-4`.

The collocated oblique free-surface receiver is `(1400,900) m`, and the physical
stress source is `(895,695) m`. Reflected-P minus direct error is 0.0275 ms and
converted-SV minus P error is 0.0459 ms, against the unchanged 4.604 ms
tolerance. Energy ratios are 1181.37 for P longitudinal/transverse and 757.17
for SV transverse/longitudinal.

For the normal P interface, the physical stress source is `(1195,495) m` and
the reflected `vy` receiver is `(1195,700) m`. The reflected path is 1205 m,
not the old nominal 1200 m, and the existing calibration is confirmed to have
the same 1205 m physical path. Peaks differ by 0.4 ms; amplitude error is
0.3689%, correlation 0.999790, and the identical-medium residual is zero.

For normal SV, x-force input y=500 m is physical `vx` y=495 m and receiver
input y=700 m is physical `vx` y=695 m. The reflection path is 1210 m. The old
calibration path was 1200 m; deriving an equal-path receiver through the
geometry helper gives input y=1710 m, physical y=1705 m, and a 1210 m path.
Reflection and calibration peaks are 0.8144 s and 0.8128 s, a 1.6 ms difference
within the unchanged 4.161 ms tolerance. Amplitude error is 2.833% and phase
correlation 0.997505.

For the oblique interface, the physical stress source is `(895,495) m` and the
collocated receiver is `(1400,700) m`. Analytical P and converted-SV travel
times are 0.435514 s and 0.553084 s. Their analytical difference is 0.117571 s
versus 0.118800 s observed: 1.229 ms error within the unchanged 3.565 ms
tolerance. P and SV polarization energy ratios are 140.46 and 614.93.

Each run writes `stdout.txt`, `stderr.txt`, and `run_metadata.json` into its
pytest temporary directory. Metadata includes the absolute executed binary
path and SHA-256, repository commit and dirty state, MPI ranks, command,
return code, runtime, and MPI version. Local Makefile variables and the current
compiler wrapper are recorded separately as local build context and are
explicitly not claimed as provenance for an externally supplied executable.
Physics tests additionally write velocity, polarization, symmetry, reciprocity,
and MPI metrics JSON files. Use pytest's
`--basetemp=/path/to/artifacts` option to retain them at a known location:

```bash
python3 -m pytest tests/physics/test_homogeneous_sh.py -vv --basetemp=/tmp/denise-m0
```

On failure, inspect the pytest assertion first, then the generated stdout,
stderr, metadata, parameter file, geometry, and seismogram in that run folder.

## GitHub Actions

`.github/workflows/verification.yml` runs on pull requests targeting
`modernization` and by manual dispatch. It installs the Linux toolchain,
OpenMPI, FFTW, and pytest; builds both `libcseife` and `bin/denise`; runs the
pure-Python unit tests; then runs all physics tests with `--require-denise`.
The physics step sets `MPIEXEC_FLAGS=--oversubscribe` because the hosted runner
provides fewer CPU slots than the four MPI ranks used by the small 2 x 2 case;
this changes process placement only, not the DENISE model or decomposition.
Missing dependencies, build failures, skipped integration tests, and physics
assertion failures therefore fail the job.

## M4 viscoelastic Q-input verification (review stop)

M4 begins with input sensitivity rather than fitting an assumed constant-Q
law. The generated homogeneous cases retain the established M0/M1 grids,
sources, receivers, FD order, and CPML settings. Their parameter files
nominally contained `L =1` and `FL=10 Hz`, and they wrote native
single-precision `.qs`, `.qp`, `.vs`, `.vp`, and `.rho` models. As documented
in the erratum below, the positional parser did not actually assign that
one-character `L` record, so the effective runtime value remained `L=0`.
Each `case.json` records the requested numerical Q values, relaxation
frequencies, MPI decomposition, and SHA-256 of every model file.

### Retrospective correction: M4 erratum

The original M4 interpretation overstated the black-box evidence. Although
the generated parameter files visibly contained `L =1`, `read_par.c` consumes
the first character of each non-comment positional record before calling
`fscanf`. For multi-character labels this is hidden because the parsed label
is not validated. For the one-character `L` label, however, `fscanf` could not
perform the integer assignment and the zero-initialized global value remained
`L=0`. The nominal M4 viscoelastic black-box runs therefore executed the
elastic paths.

Consequently, the identical Q=20, Q=50, and Q=200 seismograms observed in M4
were not an isolated executable demonstration of either the SH forward-routing
defect or the P/SV Q override. They showed that the nominal cases were
Q-insensitive, but the unactivated viscoelastic path was a confounding cause.
No statement below should be read as claiming that M4 dynamically isolated or
exercised either old-code defect.

Both defects were nevertheless real and independently established by source
inspection:

- For SH `MODE=0`, the `L>0` setup read and prepared viscoelastic material and
  memory-variable data, but `FD_SH.c` still called `sh()`, whose timestep used
  the elastic stress update rather than the available viscoelastic path.
- `readmod_visc_PSV.c` read Qp and Qs from their model files and then replaced
  both values with `30.0` before forming `taup` and `taus`.

M4.1 corrects both layers of the problem. The compact verification generator
uses a leading whitespace character so the existing positional parser really
assigns `L>0`, while the two independently source-identified solver defects
are repaired. The M4.1 runs are therefore the first runs in this verification
history that execute these Q-sensitivity guards with the viscoelastic path
actually active. Their non-zero relative-L2 values, reported below, confirm
the repaired Q-sensitive state. The whitespace workaround remains local to
the compact harness; `read_par.c` is unchanged, and general parser
modernization is a separate future task.

### Implemented source semantics

The following is a transcription of the existing implementation, not an
assumption based on parameter names. A non-zero `L` selects the viscoelastic
model reader and allocates `L` memory variables. For mechanism `l`, DENISE
defines the stress-relaxation time and dimensionless timestep ratio as

```text
theta_l = 1 / (2*pi*FL[l])
eta_l   = DT / theta_l
```

The readers map input quality factors to dimensionless relaxation strengths,
not to `theta_l`:

```text
taus = 2 / Qs
taup = 2 / Qp
```

The modulus-reference angular frequency is `omega_ref = 2*pi*FL[1]`. With
`S = sum_l (omega_ref^2*theta_l^2)/(1+omega_ref^2*theta_l^2)`, the
velocity-input P/SV path forms

```text
mu = rho*Vs^2 / (1 + S*taus)
pi = rho*Vp^2 / (1 + S*taup)
```

and the SH path applies the analogous denominator to harmonically averaged
`rho*Vs^2`. The stress coefficients contain `DT*(1+L*tau)`. Each memory
variable uses trapezoidal factors `b=1/(1+eta/2)`, `c=1-eta/2` and coupling
`modulus*eta*tau`. In SH the memory terms couple to the two shear derivatives.
In P/SV, `r` couples to `vxy+vyx`; `p` and `q` couple to `vxx+vyy` plus their
respective shear corrections. Each stress update adds half the old memory sum,
advances the memory variables, then adds half the new sum.

### Acceptance criteria

Changing Q by a factor of ten must produce a complete-seismogram relative L2
of at least `1e-3`. This is deliberately far above ASCII rounding noise and
MPI roundoff. For SH, peak, whole-trace RMS, and 5--15 Hz spectral RMS at the
600 m receiver are retained as diagnostics; their physical ordering is not an
M4 defect-guard assertion and will be validated only after the solver repair.
An identical-Q repeat must have relative L2 no greater than `1e-12` and
correlation within `1e-12` of unity. For P/SV, direct-P `vx` compares `Qp=20`
with `Qp=200` at fixed `Qs=100`; transverse-SV `vx` on the vertical receiver
line compares `Qs=20` with `Qs=200` at fixed `Qp=100`.

### Historical observations and source-inspected defects

The SH Qs=20, 50, and 200 outputs were identical: relative L2 for Qs=200
versus Qs=20 was `0.0`; peak amplitude was `0.1396774`, RMS
`0.03474107875259459`, and 5--15 Hz spectral RMS `10.44032870046328` in all
three runs. The Qs=200 repeat was exactly reproducible (relative L2 `0.0`,
correlation `1.0`). Because the effective runtime value was `L=0`, these
black-box results did not dynamically exercise or isolate the SH routing
defect. Independently, source inspection found that `FD_SH.c` read and prepared
viscoelastic arrays when `L>0` but still called `sh()`, whose forward timestep
used `update_s_elastic_PML_SH` rather than the available viscoelastic stress
update. This source finding is restricted to `PHYSICS=SH`, `MODE=0`, `L>0`.
The separate `FWI_SH_visc()` path used by `MODE=1` was not tested, so M4 makes
no claim about viscoelastic SH FWI.

P/SV independently produced identical direct-P results for Qp=20 and 200 and
identical transverse-SV results for Qs=20 and 200. Both complete-seismogram
relative L2 values were `0.0`; correlations were unity to floating-point
display precision and all recorded peak/RMS/spectral amplitude differences
were zero. The differing Q model hashes in each generated `case.json` prove
that materially different files were supplied, but the effective `L=0` means
these black-box results did not dynamically reach or isolate the P/SV Q
override. Independently, source inspection showed that `readmod_visc_PSV.c`
read the values from those files and then overwrote them with `qp = 30.0` and
`qs = 30.0` before forming `taup` and `taus`.

The three independent sensitivity assertions are retained as executable known-
defect regressions using `pytest.mark.xfail(strict=True)`: SH MODE=0 Qs, P/SV
Qp, and P/SV Qs. The SH Qs=200 repeatability check is a separate normal passing
test. Each marker also specifies `raises=KnownViscoelasticQDefect`. Program
health, output, shape, finite-sample, model-hash, and other ordinary assertions
therefore remain normal failures; only relative L2 below the unchanged `1e-3`
threshold raises the accepted defect exception. A pure-Python harness test
checks all three marker configurations and verifies that unrelated
`AssertionError` and runtime errors are not instances of that exception.

With each defect still present, the dedicated exception is reported as
`XFAIL`, so mandatory CI remains green. A repair, intentional or accidental,
raises no defect exception and produces `XPASS(strict)`, making CI red until
the corresponding marker is deliberately removed after review. The
`--require-denise` harness continues to reject real integration-test skips;
only declared xfails are exempt from that skip-to-failure conversion.

Green M4 CI therefore meant that all functioning M0--M3 tests and SH
repeatability passed and that the three declared known-defect guards reported
`XFAIL`. In retrospect, those XFAIL results must not be described as dynamic
reproduction or isolation of the two source-identified defects because the
viscoelastic path was not active. They recorded the Q-insensitive nominal M4
state under the then-current harness. Per the M4 stop condition, no
elastic/high-Q limit,
distance-dependent attenuation, spectral phase/dispersion, multiple-mechanism,
Qp/Qs-independence, or viscoelastic MPI acceptance result is claimed. Those
checks must wait for separately authorized solver fixes and a rerun of M4.

Run the review evidence with:

```bash
python3 -m pytest tests/test_attenuation.py -q
python3 -m pytest tests/physics/test_viscoelastic_q.py --require-denise -vv \
  --basetemp=/tmp/denise-m4
```

For the historical M4 revision, the second command reported one pass and three
strict xfails. That result is retained as history but, per the erratum, is not
an isolated executable proof of the source-inspected defects. Inspect
`sh_qs_200_repeatability_metrics.json`,
`sh_q_sensitivity_metrics.json`, `psv_qp_sensitivity_metrics.json`,
`psv_qs_sensitivity_metrics.json`, each case's `case.json`, and the standard run
provenance files under the retained base directory.

## M4.1 viscoelastic Q defect repairs

M4.1 activates the existing viscoelastic implementations without changing
their equations. In P/SV, `readmod_visc_PSV.c` now retains the Qp and Qs values
read from the supplied model files instead of replacing both with `30.0`. In
SH forward modelling, `FD_SH.c` selects the existing `sh_visc()` path when
`L>0` and preserves the existing `sh()` path when `L=0`.

The SH routing does not allocate the FWI data structure. `alloc_SH()` already
owns the `pr`, `pp`, and `pq` memory variables for `L>0`, `dealloc_SH()` frees
them, and the existing CPML allocation and exchange path is shared. Within
`MODE=0`, `sh_visc()` does not use the FWI-only `Rxz`, `Ryz`, stored forward
wavefields, gradients, or preconditioners. Consequently this repair makes no
claim about and does not modify viscoelastic SH FWI.

The test generator also accounts for a positional-parser peculiarity in
`read_par.c`: it consumes the first character of a non-comment record before
`fscanf`. Multi-character labels tolerate that behavior because the label is
not validated, but the one-character `L` record does not. A leading space on
the generated `L` record ensures DENISE actually receives `L>0`; without it,
the parser silently retained the zero-initialized default and the nominal M4
viscoelastic cases ran elastically.

Before removing the three M4 markers, a focused run produced three
`XPASS(strict)` failures with all original assertions intact. After marker
removal, the accepted complete-seismogram relative-L2 sensitivities are:

```text
SH Qs=200 versus Qs=20:       0.23835335436049154
P/SV Qp=200 versus Qp=20:     0.16035696010216646
P/SV Qs=200 versus Qs=20:     0.2571877051507964
```

These values exceed the unchanged `1e-3` guard independently. They establish
input sensitivity, not a complete validation of the rheology: no new assertion
on amplitude ordering, high-Q convergence, distance dependence, dispersion,
multiple relaxation mechanisms, viscoelastic MPI behavior, or FWI is added in
M4.1.

Run the repaired checks with:

```bash
python3 -m pytest tests -m 'not integration' -q
python3 -m pytest tests/physics -v --require-denise
```

The mandatory physics run must contain no XFAIL, XPASS, or SKIP results.

## M4.2 quantitative SH rheology verification (failing review stop)

M4.2 does not change any constitutive equation. It adds an independent parser
for DENISE's stdout parameter echo and requires every M4.2 SH run to report the
requested effective `MODE`, `PHYSICS`, `L`, and every `FL` value. The initial
mandatory runs confirmed `MODE=0`, `PHYSICS=5`; the elastic reference reported
`L=0` with no frequencies, and Qs=50, 200, and 1000 plus the Qs=200 repeat each
reported `L=1`, `FL=(10 Hz)`. This verifies runtime values rather than merely
the generated input text. The legacy leading-whitespace compatibility for `L`
is retained and `read_par.c` remains unchanged.

### Implemented SH rheology

For input `tau=2/Qs` and mechanism times
`theta_l=1/(2*pi*FL_l)`, the code first normalizes the shear modulus supplied
as `mu_ref=rho*Vs^2` at `omega_ref=2*pi*FL_1`:

```text
S_ref = sum_l (omega_ref*theta_l)^2 / (1+(omega_ref*theta_l)^2)
mu_R  = mu_ref / (1 + tau*S_ref)
```

The corresponding continuous generalized-Maxwell modulus transcribed from the
stress and memory-variable updates is

```text
mu*(omega) = mu_R * [1 + tau * sum_l i*omega*theta_l/(1+i*omega*theta_l)]
Q_eff(omega) = Re(mu*) / Im(mu*)
```

Thus the input Qs is not an ideal frequency-independent Q. For `L=1` at its
reference frequency, this implementation gives `Q_eff=Qs+1`. The discrete
memory update uses

```text
b_l = 1/(1+DT/(2*theta_l))
c_l = 1-DT/(2*theta_l)
R_l[n+1] = b_l * (c_l*R_l[n] - mu_R*(DT/theta_l)*tau*strain_rate[n])
```

and the stress increment uses half the old and half the new memory sum. For the
continuous comparison, `k(omega)=omega*sqrt(rho/mu*)`; the predicted slopes are
`d log|H|/dr=Im(k)` and `d arg(H)/dr=-(Re(k)-omega/Vs)` under the transform
convention used by the harness.

### Predeclared criteria and geometry

The homogeneous case uses Vs=2000 m/s, density=2000 kg/m3, DH=10 m,
DT=0.5 ms, FD order 8, a 10 Hz Ricker source, and receivers at 400, 500,
600, 700, and 800 m offset. Hann-tapered direct windows have half-width
0.11 s. Frequencies 6--14 Hz are diagnostic; 8, 10, and 12 Hz are the
mandatory attenuation band, while 8 and 12 Hz are the mandatory phase checks.
Before executing the cases, the test fixed 15% attenuation-slope and 20%
phase-slope relative tolerances, minimum R-squared 0.95, maximum log-amplitude
residual 0.02, and maximum phase residual 0.02 rad. Qs=1000 must have direct
relative L2 no greater than 0.025 and correlation at least 0.999. Complete and
direct errors must decrease strictly for Qs=50, 200, and 1000.

### Observed mandatory results

Effective parameters and the exact Qs=200 repeat passed; repeat relative L2 is
0.0 and correlation is 1.0000000000000002. High-Q convergence also passed:

| Qs | complete relative L2 | direct relative L2 | direct correlation | far peak shift (s) |
|---:|---:|---:|---:|---:|
| 50 | 0.2206403506 | 0.2206519456 | 0.9957438019 | -0.0015 |
| 200 | 0.0535405683 | 0.0535073574 | 0.9997144746 | -0.0005 |
| 1000 | 0.0107910885 | 0.0106103968 | 0.9999883611 | 0.0 |

Distance attenuation is strictly increasing at every evaluated receiver and
the mandatory attenuation slopes pass their predeclared comparison:

| f (Hz) | observed dlog-amplitude/dr (1/m) | theory (1/m) | relative error | R-squared |
|---:|---:|---:|---:|---:|
| 8 | -2.7100371e-4 | -2.4189429e-4 | 12.03% | 0.99999978 |
| 10 | -3.0718816e-4 | -3.0792529e-4 | 0.24% | 0.99999998 |
| 12 | -3.4272501e-4 | -3.6153704e-4 | 5.20% | 0.99999946 |

The additional viscoelastic phase shift is monotonic with distance and has the
theoretical sign with high fit quality, but the quantitative phase-slope gate
fails:

| f (Hz) | observed phase slope (rad/m) | theory (rad/m) | relative error | R-squared |
|---:|---:|---:|---:|---:|
| 8 | -1.3358138e-5 | -5.0777534e-5 | **73.69% FAIL** | 0.99772719 |
| 12 | 5.5830878e-5 | 7.1683734e-5 | **22.11% FAIL** | 0.99977533 |

The mandatory result is therefore two passing tests and one failing test. Per
the M4.2 stop rule, neither the solver nor the 20% tolerance was changed. The
failure may represent an analytical-model mismatch, discrete/staggered FD or
source/receiver phase effects, or a constitutive discrepancy; M4.2 does not
choose among those explanations without independent review. The `L=3`,
`FL=(5,10,20 Hz)` extended experiment was deliberately not run after the
mandatory failure, so no L=1 versus L>1 result is claimed.

Mandatory and extended commands are respectively:

```bash
MPIEXEC_FLAGS=--oversubscribe python3 -m pytest tests/physics \
  -m 'not extended' -v --require-denise
MPIEXEC_FLAGS=--oversubscribe python3 -m pytest \
  tests/physics/test_sh_viscoelastic_rheology.py -m extended -v --require-denise
```

The first command is intentionally red at this review stop because the failing
physics assertion is preserved. Extended execution must wait for review.

### M4.2 review rework: calibrated spectral estimator

Independent review identified that the original phase estimator had not been
validated for a broadband transient. Its 0.22 s fully Hann-weighted gate has an
approximate zero-to-zero main-lobe width of 18.18 Hz, much broader than the
6--14 Hz analysis band. It therefore convolves contributions from both sides
of the 10 Hz reference frequency, where the theoretical dispersive phase
changes sign.

A pure-Python known-answer test now constructs elastic 10 Hz broadband Ricker
traces at the same DT=0.5 ms and 400--800 m offsets as M4.2. For each receiver,
the FFT is multiplied by the analytical transfer
`exp((attenuation_slope(f)+i*phase_slope(f))*r)`, transformed back to time, and
processed by the production transfer-spectrum, phase-unwrapping, and linear-fit
pipeline. This calibration reproduced the short-Hann bias:

| f (Hz) | imposed attenuation | recovered attenuation | relative error | imposed phase | recovered phase | phase relative error | phase absolute error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | -1.6531012e-4 | -2.2335312e-4 | 35.11% | -8.5406063e-5 | -4.4660129e-5 | 47.71% | 4.0745933e-5 |
| 8 | -2.4189429e-4 | -2.6208256e-4 | 8.35% | -5.0777534e-5 | -1.9032633e-5 | 62.52% | 3.1744901e-5 |
| 10 | -3.0792529e-4 | -3.0022988e-4 | 2.50% | 4.5281318e-6 | 1.3218570e-5 | 191.92% | 8.6904383e-6 |
| 12 | -3.6153704e-4 | -3.3815986e-4 | 6.47% | 7.1683734e-5 | 5.2650912e-5 | 26.55% | 1.9032822e-5 |
| 14 | -4.0394503e-4 | -3.7454979e-4 | 7.28% | 1.4476839e-4 | 9.9765459e-5 | 31.09% | 4.5002935e-5 |

The replacement is a receiver-centred 0.40 s Tukey gate with alpha=0.2. Its
0.32 s flat top contains the complete direct pulse while the tapered ends
suppress truncation. Its approximate main-lobe width is 5.56 Hz. The same
known-answer test then recovered all five attenuation and phase slopes within
2.07% (the largest relative phase error occurs at 10 Hz where the expected
slope is nearly zero):

| f (Hz) | imposed attenuation | recovered attenuation | relative error | imposed phase | recovered phase | phase relative error | phase absolute error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | -1.6531012e-4 | -1.6512886e-4 | 0.110% | -8.5406063e-5 | -8.5647863e-5 | 0.283% | 2.4180030e-7 |
| 8 | -2.4189429e-4 | -2.4209712e-4 | 0.084% | -5.0777534e-5 | -5.0741116e-5 | 0.072% | 3.6417720e-8 |
| 10 | -3.0792529e-4 | -3.0778428e-4 | 0.046% | 4.5281318e-6 | 4.6217310e-6 | 2.067% | 9.3599243e-8 |
| 12 | -3.6153704e-4 | -3.6156974e-4 | 0.009% | 7.1683734e-5 | 7.1526739e-5 | 0.219% | 1.5699541e-7 |
| 14 | -4.0394503e-4 | -4.0404039e-4 | 0.024% | 1.4476839e-4 | 1.4491086e-4 | 0.098% | 1.4246974e-7 |

The receiver aperture is 400 m. Continuous-theory phase accumulation across
it is only -0.02031 rad at 8 Hz and +0.02867 rad at 12 Hz, compared with
-0.03416 rad at 6 Hz and +0.05791 rad at 14 Hz. The calibrated gate works at
all five frequencies, but 6 and 14 Hz were therefore predeclared as the revised
mandatory phase pair before rerunning DENISE. The former 8/12 Hz results remain
in the JSON as `old_hann_diagnostic`, and calibrated 8/12 Hz values remain
available as diagnostics. The phase tolerance remains 20%.

The discrete diagnostic uses DT=0.5 ms, DH=10 m, the FDORDER=8 Holberg 0.1%
coefficients `(1.2257,-0.099537,0.018063,-0.0026274)`, the staggered spatial
symbol, leapfrog temporal symbol, and the exact harmonic response of the
trapezoidal memory recurrence. Continuous and discrete slopes are:

| f (Hz) | continuous attenuation | discrete attenuation | continuous phase | discrete phase |
|---:|---:|---:|---:|---:|
| 6 | -1.6531012e-4 | -1.6540096e-4 | -8.5406063e-5 | -8.5448222e-5 |
| 8 | -2.4189429e-4 | -2.4195329e-4 | -5.0777534e-5 | -5.0778426e-5 |
| 10 | -3.0792529e-4 | -3.0789385e-4 | 4.5281318e-6 | 4.5505380e-6 |
| 12 | -3.6153704e-4 | -3.6137185e-4 | 7.1683734e-5 | 7.1691318e-5 |
| 14 | -4.0394503e-4 | -4.0362677e-4 | 1.4476839e-4 | 1.4471976e-4 |

The largest phase-slope correction is 4.86e-8 rad/m, orders of magnitude below
the original 8 Hz discrepancy of 3.74e-5 rad/m. Ordinary temporal/spatial FD
dispersion therefore cannot explain the original failure.

After calibration passed, the unchanged Qs=50 DENISE cases were rerun. The
calibrated results are:

| f (Hz) | observed attenuation | continuous theory | relative error | observed phase | continuous theory | relative error | R-squared phase |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | -1.6482496e-4 | -1.6531012e-4 | 0.29% | -8.4301770e-5 | -8.5406063e-5 | 1.29% | 0.99999999 |
| 8 | -2.4162393e-4 | -2.4189429e-4 | 0.11% | -5.1395515e-5 | -5.0777534e-5 | 1.22% | 0.99999967 |
| 10 | -3.0822426e-4 | -3.0792529e-4 | 0.10% | 4.8804297e-6 | 4.5281318e-6 | 7.78% | 0.99999875 |
| 12 | -3.6075679e-4 | -3.6153704e-4 | 0.22% | 7.1723699e-5 | 7.1683734e-5 | 0.06% | 1.00000000 |
| 14 | -4.0393781e-4 | -4.0394503e-4 | 0.002% | 1.4429393e-4 | 1.4476839e-4 | 0.33% | 1.00000000 |

Both mandatory phase frequencies pass the unchanged 20% criterion, the
8/10/12 Hz attenuation checks pass the unchanged 15% criterion, and effective
parameters, exact Qs=200 repeatability, and high-Q convergence remain green.
The original failure is therefore attributed to the uncalibrated short-Hann
measurement, not to DENISE rheology. Per the review stop, the L=3 experiment
remains unexecuted pending independent review.

## M4.2 L>1 follow-up: Qs-to-GSLS parameterization audit

This diagnostic supersedes the arbitrary L=3 experiment with the recovered
production-like L=4 parameterization
`FL=(2.7105,12.2792,68.1930,265.2297) Hz`, optimized `tau=0.0386`, and target
Qs approximately 30. No source, model-reader, parser, or solver equation is
changed. The recovered `scripts/qapprox.m` and `scripts/qstd.m` were used as
read-only historical references and remain outside the committed patch.

### Independent qstd reference and historical curve

The Python reference in `tests/utilities/qstd_reference.py` implements the
recovered MATLAB expression independently of the complex-modulus code. With
`theta_l=1/(2*pi*FL_l)`, it evaluates

```text
Q(omega) = [1 + sum_l omega^2*theta_l^2*tau/(1+omega^2*theta_l^2)]
           / [sum_l omega*theta_l*tau/(1+omega^2*theta_l^2)].
```

For L=1, an independent L=3 set, and the historical L=4 set, qstd agrees with
`Re(M*)/Im(M*)` from the existing generalized-Maxwell implementation within a
relative tolerance of `2e-15` at 5, 8, 10, 20, 60, and 120 Hz.

The historical optimized L=4 curve is:

| f (Hz) | Q, optimized tau=0.0386 | Q, direct tau=2/30 | Q, compensating `.qs=2/0.0386` |
|---:|---:|---:|---:|
| 5 | 31.19376 | 18.51194 | 31.19376 |
| 10 | 29.52250 | 17.71033 | 29.52250 |
| 20 | 29.97808 | 18.17428 | 29.97808 |
| 40 | 30.14829 | 18.44415 | 30.14829 |
| 60 | 29.75869 | 18.30982 | 29.75869 |
| 80 | 29.92265 | 18.48958 | 29.92265 |
| 100 | 30.41250 | 18.85424 | 30.41250 |
| 120 | 31.02579 | 19.28497 | 31.02579 |

Dense 0.1 Hz sampling over 5--120 Hz gives:

| case | tau | Q minimum (frequency) | Q maximum (frequency) | mean Q | RMS deviation from 30 | relative RMS |
|---|---:|---:|---:|---:|---:|---:|
| A: optimized GSLS | 0.0386 | 29.42008 (12.0 Hz) | 31.19376 (5.0 Hz) | 30.11691 | 0.39028 | 1.301% |
| B: `.qs=30` reader mapping | 0.0666667 | 17.69007 (11.1 Hz) | 19.28497 (120 Hz) | 18.49196 | 11.51344 | 38.378% |
| C: `.qs=51.81347` compensation | 0.0386 | 29.42008 (12.0 Hz) | 31.19376 (5.0 Hz) | 30.11691 | 0.39028 | 1.301% |

Case B remains comparably flat (range 1.595) but is centred near Q=18.5 rather
than Q=30. Flatness and absolute Q level are therefore distinct properties.

### Source parameter-semantics audit

The relevant paths are:

| path | model origin | input and conversion | global TAU | L dependence |
|---|---|---|---|---|
| `src/SH/readmod_visc_SH.c:11,51-52,94` | external | reads `.qs` as "Qs-values" and sets `taus=2/qs` | not used | none in Q-to-tau mapping |
| `src/PSV/readmod_visc_PSV.c:11,56-63,86-94,118-119` | external | reads `.qp/.qs` as Qp/Qs and sets `taup=2/qp`, `taus=2/qs` | not used | none in Q-to-tau mapping |
| `src/models/model.c:17,41-42,75-76` and `hh.c`, `FLnodes_visc.c`, `model_grad_visc.c`, `TOAST_bench_mod1_DEN_visc.c` | internal | assigns global `TAU` directly to both `taus` and `taup` | used directly | no automatic target-Q fit |
| `src/models/model_ainos_visc.c:50-54` and the P02/P20/plexiglas examples | internal | local variables named Qp/Qs are converted with `2/Q` | declared but not used for the conversion | none |
| `src/SH/FD_SH.c:172`, `FD_grad_SH.c:271`, `FWI_SH_visc.c:316` | external SH forward/gradient/FWI | all call the same SH reader | not used | inherit reader mapping |
| `src/PSV/FD_PSV.c:204-207`, `FWI_PSV.c:338-341`, `RTM_PSV.c:248-249` | external or internal P/SV | reader for `READMOD`, otherwise selected internal `model()` | path dependent | inherit selected mapping |

Global `TAU` is declared in `include/globvar.h:12`, read positionally in
`src/read_par.c:178`, and printed by `src/write_par.c:220`. For external models
it does not override or rescale `.qs/.qp`. The preparation routines apply the
expected L/FL-dependent reference-modulus normalization and use the resulting
`taus/taup` in every mechanism, but contain no inverse mapping from physical
target Q to optimized common tau. No missing L-dependent Q scaling was found.

FWI naming reinforces the ambiguity: `src/read_par_inv.c:51,157-158` calls the
inverted quantity Qs, while SH FWI stores and updates `ptaus`; the explicit
comment in `src/SH/FD_grad_SH.c:229` states `tau_s = 2/Qs`. Thus SH, P/SV,
forward, RTM, and FWI are internally consistent in using relaxation strength,
but the external interface exposes the files as physical Qp/Qs.

### Black-box L=4 results

Before the DENISE runs, the calibrated 0.40 s Tukey estimator was tested with
synthetic L=4 transfers for both tau values. Across 6--14 Hz its maximum
attenuation error was 0.183% and maximum phase error 0.707%. The black-box gate
was then fixed at 10% for recovered effective Q, retaining the accepted 15%
attenuation and 20% phase-slope limits.

Both runs reported effective `MODE=0`, `PHYSICS=5`, `L=4`, and runtime
`FL=(2.710500,12.279200,68.193001,265.229706) Hz`. The final digits are the
correct C Float32 representations printed to six decimals, not altered input
frequencies. Case metadata records `.qs=30` and `.qs=51.8134715`; the two Qs
model SHA-256 hashes differ. The source guard independently requires the SH
reader statement `taus[jj][ii]=2.0/qs`.

Nominal `.qs=30` (reader tau=0.0666667):

| f (Hz) | attenuation observed/theory | phase observed/theory | Q observed | Q reader theory | Q historical | historical error |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | -5.07378e-4 / -5.08275e-4 | 3.06817e-4 / 3.07212e-4 | 18.2594 | 18.2268 | 30.6271 | 40.38% |
| 8 | -6.87812e-4 / -6.87426e-4 | 5.46596e-4 / 5.46892e-4 | 17.8587 | 17.8686 | 29.8916 | 40.26% |
| 10 | -8.62244e-4 / -8.63122e-4 | 8.19560e-4 / 8.19205e-4 | 17.7282 | 17.7103 | 29.5225 | 39.95% |
| 12 | -1.03177e-3 / -1.03258e-3 | 1.11457e-3 / 1.11624e-3 | 17.7149 | 17.7002 | 29.4201 | 39.79% |
| 14 | -1.19538e-3 / -1.19553e-3 | 1.43094e-3 / 1.43111e-3 | 17.7842 | 17.7819 | 29.4847 | 39.68% |

Compensating `.qs=2/0.0386=51.8134715` (reader tau=0.0386):

| f (Hz) | attenuation observed/theory | phase observed/theory | Q observed | Q historical | relative Q error |
|---:|---:|---:|---:|---:|---:|
| 6 | -3.04087e-4 / -3.04751e-4 | 1.77187e-4 / 1.77319e-4 | 30.6942 | 30.6271 | 0.219% |
| 8 | -4.15167e-4 / -4.14949e-4 | 3.18494e-4 / 3.18788e-4 | 29.8763 | 29.8916 | 0.051% |
| 10 | -5.23369e-4 / -5.23780e-4 | 4.80724e-4 / 4.80445e-4 | 29.5454 | 29.5225 | 0.078% |
| 12 | -6.28665e-4 / -6.29346e-4 | 6.56665e-4 / 6.57577e-4 | 29.4527 | 29.4201 | 0.111% |
| 14 | -7.31294e-4 / -7.31294e-4 | 8.45605e-4 / 8.45998e-4 | 29.4849 | 29.4847 | 0.001% |

Both cases match their current-reader attenuation, phase, and Q predictions.
Only the compensating file reproduces the historical qstd target. This is
strong evidence that propagation and memory-variable updates implement the
declared GSLS response, while the external Qs-to-tau mapping uses the L=1-style
`2/Qs` approximation for every L.

### Classification and review stop

The evidence supports **Outcome 2: parameterization inconsistency / likely
defect**, not a propagation-kernel defect. External files, diagnostics, and FWI
controls are named and described as physical Qp/Qs, but for L>1 the reader does
not perform the optimized Q-target-to-common-tau mapping required by the
recovered qapprox/qstd workflow. Internal model paths that accept global TAU
already operate directly in relaxation-strength semantics.

That classification was the M4.2 review stop. M4.2.1, documented below, adds
the separately reviewed opt-in repair while preserving the audit as historical
evidence. The generic L=3 experiment remains deselected.

## M4.2.1: explicit physical-Q parameterization

M4.2.1 resolves the accepted M4.2 input defect without silently changing old
projects. Existing 115-record parameter files select mode 0 by default and
retain the historical external-reader mapping `taus=2/Qs`, `taup=2/Qp`.
Physical-Q semantics are opt-in through four optional positional records after
`RTM_SHOT`:

```text
Q_PARAMETERIZATION_MODE =1
Q_APPROX_FMIN =5.0
Q_APPROX_FMAX =120.0
Q_APPROX_DF =5.0
```

Mode 1 interprets `.qp/.qs` as requested physical quality factors. `FL` and
`L` define the global GSLS shape. The approximation grid is explicit, linear,
inclusive, equally weighted, and deterministic. This reproduces the historical
`qapprox.m` sampling `fmin:df:fmax`; it is not inferred from the source,
timestep, Nyquist frequency, or FL extrema. DENISE echoes the mode, band,
sampling interval, L, and FL at runtime.

For each sampled angular frequency, define

```text
A_i = sum_l (omega_i theta_l)^2 / (1 + (omega_i theta_l)^2)
B_i = sum_l (omega_i theta_l)   / (1 + (omega_i theta_l)^2)
a_i = 1/B_i
b_i = A_i/B_i
```

Then `Q_i=a_i/tau+b_i`. With `y=1/tau`, minimizing the recovered unweighted
qstd residual `sum_i (a_i*y+b_i-Qtarget)^2` gives

```text
y = sum_i a_i (Qtarget-b_i) / sum_i a_i^2
tau = 1/y.
```

The C implementation precomputes the affine coefficients of `y(Q)` once per
reader call. Per-cell conversion needs no optimizer or allocation. One
implementation in `src/q_parameterization.c` is shared by the external SH and
P/SV readers. Propagation, memory-variable, CPML, FWI-gradient, and RTM kernels
are unchanged.

For historical `FL=(2.7105,12.2792,68.1930,265.2297) Hz`, Q=30, and the
accepted M4.2 band sampled as 5:5:120 Hz, the fixed-FL conversion gives
`tau=0.0388444710`, only `0.0002444710` above the recovered joint-fit value
0.0386. Independent scalar minimization of the original qstd residual agrees
with the analytical result within relative `1e-8`. Tests also cover targets
20, 30, 50, 100, and 200, positivity, monotonic decreasing tau with increasing
Q, and the finite-band L=1 relationship to the approximate `2/Q` rule.

The mandatory L=4 black-box comparison uses:

```text
legacy .qs=30             -> tau=0.0666667 -> effective Q about 18
legacy .qs=51.8134715     -> tau=0.0386    -> effective Q about 30
physical-Q .qs=30         -> tau=0.0388445 -> effective Q about 30
```

Changing from mode 0 to mode 1 changes attenuation and dispersion for L>1 and
is therefore intentionally not backward-compatible unless explicitly selected.
Global `TAU` retains its historical use by internal direct-tau model generators
and is not overloaded with physical-Q semantics.

### FWI semantic boundary

The readers initialize `ptaus/ptaup` according to the selected external-input
mode for forward-model initialization. Source inspection shows that the legacy
SH attenuation-inversion path copies, differentiates, scales, updates, bounds,
and writes the internal `ptaus` field directly. P/SV likewise stores internal
`ptaus/ptaup` material fields. Names such as `INV_QS_ITER`, `QSLOWERLIM`, and
`QSUPPERLIM` do not establish physical-Q optimization semantics, and no
Q-to-tau chain rule is applied.

Attenuation/Q inversion is therefore an **explicitly deferred, currently
unverified legacy capability**. The M4--M4.2.1 harness verifies forward
viscoelastic propagation and external Q-to-initial-tau parameterization only;
it has not dynamically verified attenuation gradients, optimization updates,
bounds, recovered models, or convergence for SH or P/SV FWI. M4.2.1 neither
implements nor repairs attenuation inversion and makes no claim that mode 0 or
mode 1 provides a physically consistent Q inversion.

When `Q_PARAMETERIZATION_MODE=1` is combined with viscoelastic FWI and the
workflow schedules Qs inversion within `ITERMAX`, DENISE now emits a rank-zero
runtime warning that physical-Q conversion applies only to the initial tau
fields and that the legacy inversion has no Q-to-tau chain rule. A separate,
independently designed and verified FWI parameter-semantics milestone is
required before physical-Q attenuation inversion can be considered supported.
