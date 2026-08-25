# Developer verification guide

This guide is the practical entry point for building and testing DENISE. It
describes what the current suite actually checks; it does not claim that
DENISE is globally verified. See [verification.md](verification.md) for the
detailed scientific history, numerical results, and provenance.

## Build and prerequisites

The supported developer environment is Ubuntu or WSL with GNU Make, GCC,
OpenMPI, FFTW 3 development files, Python 3, and pytest. The detailed legacy
build and execution contract is documented in
[current_build_and_execution.md](current_build_and_execution.md).

From the repository root:

```bash
make -C libcseife
make -C src denise
```

This creates `libcseife/libcseife.a` and `bin/denise`. No root privileges are
used by the build or test commands.

Retained `tests/m5*.patch` and `tests/m5*.json` provenance artifacts are
byte-hashed. `.gitattributes` disables end-of-line translation for them so the
same QUICK checks run from native Linux and Windows-hosted WSL worktrees. Do
not normalize or reformat those historical artifacts.

Tracked `*.sh` files are forced to LF by `.gitattributes`. This keeps their
shebangs directly executable in Linux and WSL even when the host Git setting
uses CRLF conversion. In an existing Windows worktree where the runner was
already materialized with CRLF, first confirm it has no local change and then
rematerialize only that file:

```bash
git diff --quiet -- scripts/run_verification.sh && \
  rm -- scripts/run_verification.sh && \
  git -c core.autocrlf=false restore --worktree -- scripts/run_verification.sh
```

After pulling this attribute into an existing Windows worktree whose artifacts
were already translated, rematerialize only those retained files once:

```bash
git -c core.autocrlf=false restore --worktree -- \
  ':(glob)tests/m5*.patch' ':(glob)tests/m5*.json'
```

## Official test levels

| Level | Purpose | DENISE required? | Selection |
|---|---|---:|---|
| QUICK | Fast harness, analytical, math, and provenance checks | No | all tests not marked `integration` |
| MANDATORY | Normal developer and pull-request gate | Yes | build, QUICK, then non-`extended` physics |
| EXTENDED | Expensive scientific diagnostics, broad MPI matrices, production-gradient FD checks, and Taylor tests | Yes | all physics marked `extended` |
| TARGETED | Minimum feedback for a changed subsystem | Depends | explicit test files listed below |

The suite uses only the registered `integration` and `extended` markers.
`extended` physics tests are also integration tests. The current M6.1
collection contains 284 tests: 195 selected by QUICK, 56 selected by the
MANDATORY physics command, and 42 selected by EXTENDED. These selection counts
are not additive: nine pure configuration/mathematics tests live under
`tests/physics/` and are selected by both QUICK and the non-extended physics
command. The counts were recomputed with pytest collection at the M6.1 closure
head rather than carried forward from M5.6/M6.0.

### Copy-pasteable commands

QUICK:

```bash
python3 -m pytest tests -m 'not integration' -q
```

MANDATORY physics (after building):

```bash
MPIEXEC_FLAGS=--oversubscribe \
python3 -m pytest tests/physics \
  -m 'not extended' \
  --require-denise -v
```

EXTENDED physics (after building):

```bash
MPIEXEC_FLAGS=--oversubscribe \
python3 -m pytest tests/physics \
  -m extended \
  --require-denise -v
```

Run one test, for example the homogeneous SH travel-time check:

```bash
MPIEXEC_FLAGS=--oversubscribe \
python3 -m pytest \
  tests/physics/test_homogeneous_sh.py::test_homogeneous_elastic_sh_travel_times \
  --require-denise -v
```

The equivalent developer runner is:

```bash
./scripts/run_verification.sh quick
./scripts/run_verification.sh mandatory
./scripts/run_verification.sh extended
```

`mandatory` and `extended` build `libcseife` and DENISE first. The runner uses
`MPIEXEC_FLAGS=--oversubscribe` unless the caller already supplied a value,
prints each command, and returns the failing command's status.

### Development skips versus mandatory failures

Without `--require-denise`, an integration test explicitly reports `SKIP` when
`bin/denise` or the configured MPI launcher is unavailable. This is convenient
while developing pure-Python helpers. A skip does not demonstrate physics
correctness.

With `--require-denise`, a missing executable, missing MPI launcher, or any
other integration-test skip is a `FAIL`. CI and review-grade verification must
use this option. A declared XFAIL, if one is ever present, remains distinct
from a dependency skip; the current mandatory suite is expected to have no
skips.

## Verification coverage matrix

Status words refer only to evidence in the current repository:

- **VERIFIED**: quantitative or mathematical checks directly exercise the
  stated behavior.
- **PARTIALLY VERIFIED**: strong evidence exists for a defined subset, with
  material gaps stated explicitly.
- **FORWARD VERIFIED ONLY**: propagation is checked, but inversion semantics
  are not.
- **UNVERIFIED / NOT COVERED**: no current targeted evidence supports a claim.
- **QUARANTINED**: the path is deliberately rejected before solver execution.

| Physics / feature | Forward | Boundary | MPI | Viscoelastic | Gradient FD | Taylor | End-to-end inversion | Status / limitations |
|---|---|---|---|---|---|---|---|---|
| SH elastic | VERIFIED: homogeneous velocity; flat free-surface normal/oblique reflection and forward hold-outs | VERIFIED: CPML and flat grid-aligned free surface at FDORDER 2/4/6/8/10/12 | VERIFIED: selected homogeneous, CPML, free-surface, corner, and FWI 1x1/2x1/1x2 decompositions | n/a | VERIFIED: Vs/rho, including heterogeneous/DTINV diagnostics and FREE_SURF=1 direct FD | VERIFIED: Vs/rho production gradients and FREE_SURF=1 GF1/GF2 Taylor closure | PARTIALLY VERIFIED: successful GF1/GF2 FWI smokes and gradient closure, not optimizer convergence | **VERIFIED within the listed elastic flat-surface scope**; not a claim for viscoelastic free surfaces, topography, anisotropy, attenuation inversion, or optimizer convergence |
| SH viscoelastic | VERIFIED: Q sensitivity, attenuation, phase, high-Q convergence | NOT COVERED by a separate visco boundary oracle | NOT COVERED beyond single-rank execution | VERIFIED for reviewed forward GSLS/Q mappings | UNVERIFIED for attenuation parameters | NOT COVERED | UNVERIFIED; attenuation update path appears incomplete | **FORWARD VERIFIED ONLY** |
| PSV elastic | VERIFIED: P/SV velocity, polarization, symmetry, reciprocity, interfaces | VERIFIED: CPML, free surface, elastic interfaces | VERIFIED: forward, boundaries, heterogeneous gradient seams | n/a | VERIFIED: physical Vp/Vs/rho, homogeneous and heterogeneous | VERIFIED: five predefined GF1/GF2 cases | PARTIALLY VERIFIED: successful gradient runs, not optimizer convergence | **VERIFIED within INVMAT1=1 scope** |
| PSV viscoelastic | VERIFIED: Qp/Qs sensitivity, attenuation, phase, high-Q convergence | NOT COVERED by a separate visco boundary oracle | VERIFIED for selected P/SV decompositions; broader matrix is extended | VERIFIED for reviewed forward GSLS/Q mappings | UNVERIFIED for attenuation parameters | NOT COVERED | UNVERIFIED | **FORWARD VERIFIED ONLY** |
| Acoustic | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | **UNVERIFIED** by the current harness |
| VTI | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | **UNVERIFIED**; preservation in a source diff is not a physics test |
| TTI | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | **UNVERIFIED**; preservation in a source diff is not a physics test |
| RTM | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | NOT COVERED | **UNVERIFIED**; no black-box RTM acceptance test |
| PSV `INVMAT1=1` (Vp/Vs/rho) | VERIFIED | VERIFIED in elastic PSV tests | VERIFIED | Forward viscoelastic VERIFIED | VERIFIED | VERIFIED | PARTIALLY VERIFIED | **VERIFIED for elastic physical gradients and listed forward paths** |
| PSV `INVMAT1=2` (Zp/Zs/rho) | Rejected | n/a | Rejection VERIFIED at 1 and 2 ranks | Rejected | NOT COVERED | NOT COVERED | NOT COVERED | **QUARANTINED**: legacy file/input contract is undefined |
| PSV `INVMAT1=3` (lambda/mu/rho) | VERIFIED by deterministic forward smoke only | NOT COVERED | NOT COVERED | NOT COVERED | UNVERIFIED | NOT COVERED | NOT COVERED | **FORWARD VERIFIED ONLY**; no gradient claim |
| Q / attenuation inversion | Forward Q-to-GSLS initialization VERIFIED | NOT COVERED | NOT COVERED | Forward rheology VERIFIED | UNVERIFIED | NOT COVERED | UNVERIFIED and apparently incomplete | **UNVERIFIED legacy capability**; no Q-to-tau chain rule |

CPML tests cover SH and elastic PSV incidence, negative controls, and selected
MPI decompositions. Elastic-interface tests remain PSV-only. Free-surface
coverage includes the PSV cases plus the M6.1 flat grid-aligned elastic SH
boundary, reflection, stability, hold-out, gradient, and Taylor tests. M6.1
does not cover a viscoelastic SH free surface, topography, an anisotropic free
surface, attenuation/Q inversion, or optimizer convergence.
Provenance tests lock retained M5 production patches and diagnostic artifacts;
they establish reproducibility, not additional physics coverage.

## What should I run after my change?

Every production change should finish with QUICK and MANDATORY. The targeted
column provides earlier feedback; it does not replace the mandatory gate.

| Changed subsystem | Minimum targeted tests | Mandatory suite | Extended guidance |
|---|---|---|---|
| PSV forward kernel | `test_homogeneous_psv.py`, `test_cpml.py`, `test_free_surface.py`, `test_elastic_interface.py` | Required | Run PSV rheology extended MPI if exchange/visco code is touched; no acoustic/VTI/TTI inference |
| PSV FWI / gradient | QUICK: `test_psv_gradient_math.py`, `test_psv_gradient_production_math.py`, `test_psv_taylor_math.py` | Required | Required: production gradient, heterogeneous hold-out, protected paths, and PSV Taylor files |
| SH forward kernel | `test_homogeneous_sh.py`, SH selections in `test_cpml.py`; for a flat elastic free-surface change also run `test_sh_free_surface.py`, `test_sh_free_surface_runtime.py`, and `test_sh_free_surface_holdouts.py` | Required | Run the full M6.1 free-surface families for boundary/CPML interaction changes and SH rheology for visco changes; do not infer viscoelastic or topographic free-surface coverage |
| SH FWI / gradient | QUICK: `test_sh_fwi_gradient_math.py`, `test_sh_density_gradient_math.py`, `test_sh_taylor_math.py`; mandatory `test_sh_fwi_gradient.py`; for FREE_SURF=1 run `test_sh_free_surface_fwi_gradient.py` and `test_sh_free_surface_fwi_taylor.py` | Required | Required: production gradient and SH Taylor relevant to the changed boundary; use retained diagnostics when temporal, density, averaging, or scaling code changes |
| Viscoelastic / rheology | QUICK: `test_attenuation.py`, `test_rheology.py`; physics: `test_viscoelastic_q.py`, SH and PSV rheology files | Required | Run extended multi-mechanism SH and PSV MPI cases when their paths are affected; attenuation inversion remains unverified |
| CPML | `test_cpml.py` | Required | No separate extended CPML suite exists; add a new oracle for uncovered physics families |
| Free surface | PSV: `test_free_surface.py`; elastic SH: `test_sh_free_surface.py`, `test_sh_free_surface_runtime.py`, and `test_sh_free_surface_holdouts.py` | Required | For SH adjoint/FWI changes additionally run `test_sh_free_surface_fwi_gradient.py` and `test_sh_free_surface_fwi_taylor.py`; M6.1 is limited to a flat grid-aligned elastic SH surface |
| MPI exchange | MPI cases in homogeneous SH/PSV, CPML, free-surface, interface, and rheology tests | Required | Run extended PSV rheology MPI matrix and heterogeneous gradient seam tests relevant to the changed exchange path |
| Model reading / parameterization | QUICK harness, attenuation, rheology, and `test_psv_invmat2_contract.py`; physics `test_viscoelastic_q.py` and `test_psv_invmat2_rejected.py` | Required | Run protected paths and quantitative rheology for affected modes; do not reinterpret quarantined `INVMAT1=2` |
| Objective / residual | QUICK SH/PSV gradient math and Taylor math; mandatory `test_sh_fwi_gradient.py` | Required | Run both SH and PSV production-gradient/Taylor tests for affected GF forms |
| Optimizer / line search | No adequate targeted correctness test exists | Required, but insufficient alone | Add an independent end-to-end optimization test; current gradient and smoke tests do not verify convergence |
| Documentation only | `test_verification_workflow.py` and runner `quick` | QUICK required | Physics not normally required unless commands, selection, or scientific claims changed |

Examples of explicit targeted invocation:

```bash
MPIEXEC_FLAGS=--oversubscribe \
python3 -m pytest tests/physics/test_cpml.py --require-denise -v

MPIEXEC_FLAGS=--oversubscribe \
python3 -m pytest \
  tests/physics/test_psv_fwi_production_gradient.py \
  tests/physics/test_psv_fwi_heterogeneous_holdout.py \
  tests/physics/test_psv_fwi_taylor.py \
  -m extended --require-denise -v
```

## Adding a test

1. Put deterministic model/geometry/input generation in `tests/cases/` and
   reusable readers, metrics, or analytical oracles in `tests/utilities/`.
2. Put black-box DENISE execution in `tests/physics/` and mark it
   `integration`. Add `extended` only when it is genuinely too expensive or is
   an additional matrix/diagnostic rather than the normal gate.
3. Keep pure mathematical and harness-contract tests directly under `tests/`.
4. Use the shared `denise_binary`, `mpiexec`, and `repository_root` fixtures.
   Run DENISE through the existing runner utilities so return codes, commands,
   runtime, executable hash, git state, MPI ranks, and logs are retained.
5. Use an independent analytical, physical, finite-difference, transpose, or
   Taylor oracle. Do not derive both expected and observed values from the same
   production formula.
6. Choose tolerances before inspecting results and justify them using temporal
   sampling, spatial sampling, discretization, bandwidth, or oracle error.
7. Prove that missing outputs, nonzero returns, NaN/Inf, and mandatory skips
   fail normally. Add a negative control when it materially strengthens the
   claim.
8. Update this matrix and the detailed scientific log without broadening the
   verification claim beyond the new evidence.

## Historical SH comparison

[`RMecking/DENISE-SH`](https://github.com/RMecking/DENISE-SH) is a
**HISTORICAL COMPARATIVE IMPLEMENTATION — NOT AN ORACLE**. Future SH work may
use it for source-genealogy or comparative analysis, but correctness must still
be established with independent physical or mathematical tests. M5.6 imports
or copies no DENISE-SH code.
