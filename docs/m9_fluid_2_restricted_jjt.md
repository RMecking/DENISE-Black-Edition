# M9-FLUID-2 restricted zero-shear CPU J/JT candidate

Status: M9-FLUID-2 READY FOR INDEPENDENT SCIENTIFIC VERIFICATION.
Local implementation evidence only; no publication or milestone closure.
Date: 2026-10-04.

## Provenance and scope

BASE / HEAD: `d66437ebe37bff82d95e24c1bff6d79f4e29660e`.
Branch: `codex/m9-fluid-2-restricted-jjt`.
Live `origin/modernization` matched BASE before work and at final verification.
Worktree: `C:\Users\rebme\.codex\worktrees\m9-fluid-2-restricted-jjt\DENISE-Black-Edition`.

Production changes are confined to:
- `src/PSV/elastic_psv_born.c`
- `src/PSV/elastic_psv_born_mpi.c`

Added tests/support:
- `tests/physics/test_m9_fluid_restricted_jjt.py`
- `tests/physics/test_m9_fluid_restricted_jjt_mpi.py`
- `tests/physics/test_m9_fluid_restricted_material.py`
- `tests/physics/test_m9_fluid_restricted_mode2.py`
- `tests/utilities/m9_fluid_restricted_material.c`
- `tests/utilities/m9_fluid_restricted_mpi.c`
- `tests/utilities/m9_fluid_restricted_reference.py`

This document is the tenth candidate path. The external freeze manifest records
each path's RAW SHA256, unfiltered and canonical filtered Git blobs, attributes,
EOL state and Git-derived statistics. File-by-file identities are authoritative;
no aggregate path-set hash is used. The six FLUID-1A files and existing FLUID-1B
test files remain unchanged.

## Frozen graph and implementation

Fluid is exactly copied background FP32 mu == 0; signed zero is accepted.
Positive subnormal mu is solid. There is no epsilon, mask or Vs floor. Existing
fixed-density inverse face maps are unchanged.

Serial J validates every copied fluid cell's dMu before zeroing caller data,
allocating tangent state, reconstructing operands or changing replay diagnostics.
Any nonzero fluid dMu rejects. A subsequent valid J succeeds bit-identically.
Caller mutation cannot change the context's copied classification.

MPI performs the equivalent owned-tile restriction inside compact input
validation, then collective agreement, before allocation or material/tangent
halo work. Tests put one invalid dMu on exactly one real owning CPU MPI rank;
all ranks reject, sentinel data and complete diagnostics remain unchanged, and
valid J recovers. Both fluid-to-solid and solid-to-fluid nonlinear trials are
also tested on one owner, with collective rejection before mutation and valid
nonlinear/J recovery.

The corner tangent and harmonic VJP inspect the four actual copied physical
background contributors (MPI uses canonical material halos). If any is fluid,
dH is +0 and the corner contributes no VJP to any contributor, including adjacent
solids. The branch precedes every division. All-solid derivative expressions,
precision and accumulation order are retained verbatim. Tests expose actual
private production products and compare with unchanged FLUID-1A material JVP/VJP
for all 16 occupancy patterns; mixed-corner VJP is exactly +0 everywhere.

JT projects only the final raw material mu image on copied fluid cells to +0,
inside the serial operator / MPI owned-output operation. This is the transpose
of the restricted parameter injection. It does not project state, adjoint state,
fluid strains or adjacent solid images. Direct solid normal/surface terms and
all-solid-corner contributions remain active. Adjacent solid gMu is explicitly
nonzero in the wave tests.

Interior fluid gLambda remains nonzero. The exact existing free-surface graph
is retained: fluid alpha=1, A=0, restricted d-alpha=d-A=0. Raw top-row fluid
gLambda and every raw fluid gMu are checked for exact positive zero. No surface
stencil or CPML coefficient/transpose is altered; FS1 still disables top-y CPML.
Actual fluid strain operands remain nonzero.

Checkpoint timing, payload, FULL/replay selection and deterministic schedule
are unchanged. Static classification comes from copied mu, not checkpoint data.
Serial FULL and segments 1/3/7 give bit-identical F/J/JT, including uneven NT=120,
S=7. MPI FULL/segments=3 likewise match bitwise. Production MODE=2 selects its
existing 32-segment replay backend in the bounded water cases, including the
uneven schedule with max length 4, and performs exactly NT replayed steps.

CPU MODE=2 requires no adapter or file change: its existing driver now reaches
the opened restricted JT. Model files remain x-major FP32 .lam/.mu/.rho;
prepared source and chronological direct vx/vy data semantics stay unchanged.
Raw lambda/mu images remain row-major FP64 with existing names and no weighting.
There is no density image, coordinate transform or application mask.

## Independent evidence

The new test-only reference adapter leaves the original reference files intact.
It combines frozen zero-shear material maps/JVP/VJP with the already independent
M9 FP64 state/CPML tangent and reverse equations. FS uses the original independently
matrix-checked surface operator with coefficients supplied from the frozen
zero-shear surface law. FS0 forward operands are evaluated using frozen zero-shear
FD4 equations; no production states or J/JT outputs create expected values.
FP64 reference dots independently satisfy 5e-13 times operand scale.

Actual production accuracy and dots use the unchanged
`_production_dot_metrics` / `_assert_production_dot_closes` absolute bound,
separate J ceiling 1e-5 and JT ceiling 6e-5. The relative dot quotient is diagnostic,
not the gate. Dense centered FD retains epsilon=0.05 and ceiling=0.004.

| Measured fluid maximum | Value |
|---|---:|
| J vs independent FP64, relative L2 | 1.1172817631592228e-6 |
| raw JT vs independent FP64, relative L2 | 7.882905564638122e-7 |
| absolute production dot residual | 6.886124814808880e-8 |
| pair-relative dot residual (diagnostic) | 1.132417489665050e-5 |
| operand-scale dot residual (diagnostic) | 1.6724828189968044e-8 |
| restricted directional FD relative L2 | 0.0013045128094537485 |
| dense every-allowed-column FD relative L2 | 0.0008153767479744056 |

Serial geometries include homogeneous fluid, horizontal interface, offset
interface, FS0/FS1, FW=0/3, FULL/replay. Fluid-lambda, solid-lambda, solid-mu,
joint and localized interface directions combine vx-only, vy-only and both
cotangents; zero directions are excluded only where a phase is absent.
FS0 maxima: J 1.0683522125533339e-6, JT 7.882905564638122e-7,
FD 0.0013045128094537485.
FS1 maxima: J 1.1172817631592228e-6, JT 7.858389764662235e-7,
FD 0.0013013082721751751.
Dense wave FD enumerates all 64 lambda and 24 solid-mu columns of each 8x8
fixture (88 columns), FS0 and FS1+CPML. Aggregate dense errors are
0.0006366560851764496 and 0.0008153767479744056, respectively.
Homogeneous F also passes the unchanged separate pressure-only acoustic
authority; H, shear state and normal-stress equality are checked explicitly.

The MPI matrix uses actual production owned halos and ADD transpose on 1x1,
2x1, 1x2 and 2x2, horizontal/vertical/offset interfaces, FS0/FS1 and CPML.
F and J are bit-identical to serial on every decomposition. Raw JT relative
differences to serial remain below 1e-12:

| CPU MPI topology | Max raw gMu relative difference to serial |
|---|---:|
| 1x1 | 9.441387881451563e-16 |
| 2x1 | 1.2553553603127265e-15 |
| 1x2 | 1.058664990216551e-15 |
| 2x2 | 1.1263018560886127e-15 |

Each topology's max independent J error is 6.514275904883080e-7,
max JT error <=6.477889594089085e-7 and max FD error
0.00016627787517118562. Interface/corner material maps and CPML profiles are
checked by the canonical production MPI harness; no Python halo reconstruction
substitutes for actual numerical execution.

Bounded Marmousi-II-shaped top water layer: 24x20, 11 water rows, mu exactly
zero, lambda=4/rho=1 in water, solid lambda=6.75/mu=3.375/rho=1.5 below.
The prepared explosive source is in water next to the interface; a receiver is
below the interface. Independent restricted Born data drive real CPU MODE=2.
API images agree with independent FP64 to <=2.544894402176219e-7.
Interior water lambda image norms are 0.12427977164671597 (FS0) and
0.1150927939127376 (FS1+CPML); solid mu norms are 0.036591317966435555
and 0.03600273402032018. Fluid mu is exact +0. Four executable runs cover
1x1 FS0, 2x1 FS0+CPML, 1x2 FS1 and 2x2 FS1+CPML. File images differ from the
serial CPU driver by at most 2.866511271135863e-16 relative L2. A negative mu
trial removes the existing final pair, and subsequent valid execution recovers
bit-identically. No FLUID-2 capability rejection remains on CPU.

## All-solid and CUDA boundary

Eight serial and eight 2x2 CPU MPI comparisons against exact BASE source
artifacts are bit-identical (FS0/FS1, CPML off/on, FULL/replay), including F, J,
raw JT, actual strain products and material maps; MPI also checks nonlinear
plus/minus and four stored strain samples. BASE source RAW SHA256 values:
serial f76dce894c3e761bf0f911c827d40f1304fedfba7c3cf8f8a91f5ac36a163eb1;
MPI e9fdc9f1f357eb170c44996bcc9117ff4b2fab7e1f16012cb3da6667f7e396c7.
The existing canonical Linux/NumPy 2.5.3 six FP64 fingerprints pass unchanged,
as do four representative original all-solid J/JT/reference/dot gates.

No CUDA production path changes. The existing m9_host_create fluid guard
rejects constructors before GPU numerical work, and nonlinear fluid trial
guard precedes invalidation of a prepared solid context. Opening CPU JT cannot
bypass these guards. Actual published forward/full/migration/replay constructors
and migrate_request reject valid physical fluid with explicit FLUID-4; requests
return no raw image pointers and retain no owned ledger resources. Failed fluid
trial preserves all prepared solid GPU products byte-for-byte. Twelve original
all-solid CUDA full/replay J/JT/dot/migration gates pass for local FMA/non-FMA.
No NESH execution, no additional GPU, no unpublished M9e-5 access.

## Verification inventory and historical capability tests

197 distinct focused gates pass:
- 45 immutable FLUID-1A oracle/interface gates.
- 44 still-valid FLUID-1B gates (41 serial forward/material/boundary plus three
  collective negative-material MPI gates).
- 24 new serial wave/transaction/FD/replay/acoustic gates.
- 25 new corner/copied-classification/serial BASE gates.
- 28 new restricted MPI and MPI BASE gates.
- Six new CPU MODE=2 API/executable gates.
- 20 local CUDA gates (eight fluid boundary, twelve all-solid).
- Five existing relevant solid oracle/publication-fingerprint gates.

The initial historical run had 86 passes and one obsolete failure:
`test_copied_fp32_classification_no_epsilon_or_caller_mask` additionally expects
zero-direction fluid J to fail. Its copied classification assertions passed;
its old capability expectation is deliberately superseded. The four old
homogeneous forward tests likewise combine valid acoustic F with the old
FLUID-2 J/JT rejection; the old MPI forward harness and two old MODE=2 fluid
tests also assert that superseded capability boundary. They remain unmodified
historical tests. The final scoped selection excludes those expectations and
replaces them with new actual acoustic F, copied classification, positive J/JT,
owned MPI and positive MODE=2 evidence. The two NumPy warnings in legacy invalid
material fixture construction (zero rho / negative lambda) remain expected.
No new unexpected production or scientific failures were observed. This task
does not repair the unrelated repository-wide hosted-CI baseline.

Evidence directory:
`C:\Users\rebme\.codex\visualizations\2026\09\09\01a08808-9623-7c91-97c3-68cf57cc7dd7\m9-fluid-2`.
It contains command logs, per-product metrics, exact BASE artifacts, reproduction
instructions and the final manifest/work report.

Local runtime: user vialon; Ubuntu WSL2 kernel 6.18.40.1; GCC13; Open MPI4.1.6;
NumPy2.5.3; CUDA12.8.93; RTX3070 Laptop 8192MiB, sm86, driver591.44.
Normal CPU `make -C libcseife -j4; make -C src -j4 denise` succeeds.
CUDA libraries are opt-in and built by the existing published test fixture.

## Required scope flags and repository state

| Contract | Changed / added |
|---|---|
| Tolerances | NO |
| Frozen oracles | NO |
| Checkpoint format/architecture | NO |
| Public API/ABI | NO |
| Density derivative/image | NO |
| Parameter transform/optimizer | NO |
| CUDA fluid numerics | NO |
| Interface classification-C evidence | NO |
| M9e-5 touched/accessed/copied | NO |
| Applications worktree/setup touched | NO |
| Commit / push / PR / staging | NO |

`git diff --check` passes. Nothing staged. Candidate HEAD remains BASE.
The primary F: worktree's pre-existing seven dirty/untracked paths are unchanged.
Only the ten listed candidate paths are changed/added in the isolated worktree.
The candidate remains local for independent scientific verification; no
acceptance, publication or FLUID-2 closure is claimed.
