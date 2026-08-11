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
sources, receivers, FD order, and CPML settings. They set `L=1`, `FL=10 Hz`
and write native single-precision `.qs`, `.qp`, `.vs`, `.vp`, and `.rho`
models. Each `case.json` records the numerical Q values, relaxation
frequencies, MPI decomposition, and SHA-256 of every model file.

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

### Observed baseline and defects

The SH Qs=20, 50, and 200 outputs were identical: relative L2 for Qs=200
versus Qs=20 was `0.0`; peak amplitude was `0.1396774`, RMS
`0.03474107875259459`, and 5--15 Hz spectral RMS `10.44032870046328` in all
three runs. The Qs=200 repeat was exactly reproducible (relative L2 `0.0`,
correlation `1.0`). Source inspection explains the failed sensitivity:
`FD_SH.c` reads and prepares viscoelastic arrays when `L>0`, but `sh.c` calls
`update_s_elastic_PML_SH` unconditionally in the forward timestep and never
calls the available viscoelastic SH stress update. This finding is restricted
to `PHYSICS=SH`, `MODE=0`, `L>0`. The separate `FWI_SH_visc()` path used by
`MODE=1` was not tested, so M4 makes no claim about viscoelastic SH FWI.

P/SV independently produced identical direct-P results for Qp=20 and 200 and
identical transverse-SV results for Qs=20 and 200. Both complete-seismogram
relative L2 values were `0.0`; correlations were unity to floating-point
display precision and all recorded peak/RMS/spectral amplitude differences
were zero. The differing Q model hashes in each generated `case.json` prove
that materially different files were supplied. In `readmod_visc_PSV.c`, the
values read from those files are overwritten by `qp = 30.0` and `qs = 30.0`
before `taup` and `taus` are formed.

The three independent sensitivity assertions are retained as executable known-
defect regressions using `pytest.mark.xfail(strict=True)`: SH MODE=0 Qs, P/SV
Qp, and P/SV Qs. The SH Qs=200 repeatability check is a separate normal passing
test. With each defect still present, its assertion fails internally and is
reported as `XFAIL`, so mandatory CI remains green. A repair, intentional or
accidental, produces `XPASS(strict)` and makes CI red until the corresponding
marker is deliberately removed after review. The `--require-denise` harness
continues to reject real integration-test skips; only declared xfails are
exempt from that skip-to-failure conversion.

Green M4 CI therefore means that all functioning M0--M3 tests and SH
repeatability pass, while the three confirmed M4 defects still reproduce
exactly as documented. It does not mean that the defects have been repaired.
Per the M4 stop condition, no elastic/high-Q limit,
distance-dependent attenuation, spectral phase/dispersion, multiple-mechanism,
Qp/Qs-independence, or viscoelastic MPI acceptance result is claimed. Those
checks must wait for separately authorized solver fixes and a rerun of M4.

Run the review evidence with:

```bash
python3 -m pytest tests/test_attenuation.py -q
python3 -m pytest tests/physics/test_viscoelastic_q.py --require-denise -vv \
  --basetemp=/tmp/denise-m4
```

The second command is expected to report one pass and three strict xfails while
the defects remain. Inspect `sh_qs_200_repeatability_metrics.json`,
`sh_q_sensitivity_metrics.json`, `psv_qp_sensitivity_metrics.json`,
`psv_qs_sensitivity_metrics.json`, each case's `case.json`, and the standard run
provenance files under the retained base directory.
