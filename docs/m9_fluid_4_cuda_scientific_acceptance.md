# FLUID-4 — CUDA zero-shear scientific acceptance ledger

WORKER: DENISE — DOCS-AUDIT. Documentation/evidence synthesis, 2026-10-08.

**FLUID-4F documentation candidate: NOT CLOSED / NOT PUBLISHED.**
Canonical BASE: `c582623111468cd6b4bee3a7aa4e4df5355ee00c`.
The published 4B–4E capabilities below are accepted within their stated
envelope; this candidate does not create a new merge or acceptance authority.
Next owner is independent SCIENTIFIC-VERIFICATION for cross-slice review.
External acceptance, Content Lock and publication authorization remain separate.

## Published authority and supporting verification

| Slice | Canonical normal merge | PR | Accepted scope |
|---|---|---|---|
| 4B | `e0ae634f368f41d3d7954227bd2485906da5bc12` | [#97](https://github.com/RMecking/DENISE-Black-Edition/pull/97) | Forward / classification-preserving nonlinear |
| 4C | `97798b883def24de12a4c022ed4f0deaf67ad311` | [#98](https://github.com/RMecking/DENISE-Black-Edition/pull/98) | Restricted Born J / surface tangent |
| 4D | `8787fe25088727785ecfd46b5d62c92e7ee3f47c` | [#99](https://github.com/RMecking/DENISE-Black-Edition/pull/99) | Restricted JT / direct FULL raw migration |
| 4E | `c582623111468cd6b4bee3a7aa4e4df5355ee00c` | [#100](https://github.com/RMecking/DENISE-Black-Edition/pull/100) | Replay / ordered multishot / actual MODE=2 |

GitHub identifies all four PRs as merged into `modernization`; each normal
merge's first parent is the preceding canonical milestone, and its second
parent is the published feature head. Feature heads are respectively
`d498b12c2261ba28a086e364c3f1c51ddd5d0828`,
`3814c61c4e7156d0c3061f92505d7fc16bfddf37`,
`2ef28620755ab1109c9dc0c8ec0300d9e73eaf99`, and
`1a185944fccdd4fd825f4c7a4cdf759abd015eeb`.
4B's first parent is `501389d156f55cb662e34b30363b516b288a77ea`.

FLUID-4A is the frozen source-audit/implementation contract at that first
parent, not separate Production capability or GPU execution evidence.
The [original zero-shear contract](m9_fluid_1a_zero_shear_contract.md), blob
`21bed107fe6d6b869f65718870f98de968f5b11e`, is unchanged and authoritative.
The independent references `tests/utilities/zero_shear_reference.py` and
`tests/utilities/m9_fluid_restricted_reference.py` have unchanged blobs
`60d9ed521432af94e65515d5844865c7cd764c44` and
`1a3e230f36637ddbccd53e2b379ae49a53bd0ab3`.

CPU FLUID-3F acceptance remains **CLOSED / VERIFICATION-ONLY** at
`5263bf1b257ab8b9c8ca77a2514cb27be0db61c5`: 602 focused CPU tests passed.
Its tested Cartesian MPI support is not reopened, widened or replaced by
GPU results. See the [status ledger](modernization_status.md).

### External report identifiers (not invented repository evidence paths)

Supporting reports belong to `DENISE — SCIENTIFIC-VERIFICATION`, thread
`01a09e28-ff24-7410-99b2-2f6db6b21459`. Their actual external artifact root is
`C:/Users/rebme/.codex/visualizations/2026/09/14/01a09e28-ff24-7410-99b2-2f6db6b21459/`.
Identifiers below mean subdirectory plus `REPORT.md` under that root; these
are local external records, not GitHub-hosted or committed evidence files.

| ID | Report artifact | RAW report SHA256 |
|---|---|---|
| B | `fluid-4b-verification/REPORT.md` | `aeb75d6ab76d1cad389b6b1ef88cc862f91fdec6be90b119256ff4ada75938a8` |
| C | `fluid-4c-verification/REPORT.md` | `6e22b59babc54640f2c7c78e9a4525bc35b74c427db7eabc7dc8739572202bbb` |
| D | `fluid-4d-verification/REPORT.md` | `9876100a913a9b926561df28dd56a499c3ab58326e13007f5c5accc97a0d2be2` |
| E | `fluid-4e-verification/REPORT.md` | `444b3b9a8907638b774a893ddce1f3b917f5da28527fa7ddfb6a06b7c597c734` |

B/C reports are dated 2026-10-07; D/E 2026-10-08. Each reports scientific
verification PASS / READY FOR CONTENT LOCK for an exact uncommitted snapshot,
not publication authority. Their file-identity tables match their subsequently
published milestone blobs. Later stage guard/test changes do not retroactively
change an earlier report's scope. C/D/E report hashes also match their retained
manifests; B is identified by its actual report bytes and source-identity table,
not by an invented seal. 4A is `fluid-4a-audit/REPORT.md` at the same root.

## Scientific envelope and capability matrix

Modern M9 isotropic **elastic P/SV**, physical exact-zero-shear fluid,
restricted fixed-density lambda/mu parameter space, one MPI rank and one
NVIDIA CUDA GPU. Published FD4 discretization, source/receiver semantics,
precision, boundary ordering and admissible material conditions are preserved.
The MODE=2 adapter uses `L=0`, `INVMAT1=3`, `FDORDER=4`, `NDT=DTINV=1`,
`BOUNDARY=0`, `READMOD=1`, `READREC=SRCREC=1`, `RUN_MULTIPLE_SHOTS=1`,
`SEISMO=1`, `QUELLART=3`, `INV_STF=0` and its canonical envelope validator.
This does not certify arbitrary uses of legacy `INVMAT1=3`.

| Capability | Accepted evidence | Boundary of claim |
|---|---|---|
| Forward / nonlinear | B independent full products, acoustic comparison, ownership and trial checks | Trials preserve copied classification; no topology-changing linearization |
| Fluid/solid interfaces | B/C/D representative horizontal, vertical and offset fixtures | No new general geometry/topography or arbitrary contrast certification |
| Restricted FULL J | C independent FP64 products, all 16 corner occupancies, centered FD | Lambda everywhere; mu only in solid; no density column |
| Restricted JT / direct FULL migration | D independent raw images, restricted dot/basis, internal projection | Raw physical Euclidean lambda/mu images, not transformed FWI gradients |
| FULL/SEGMENTED replay JT | E complete checkpoints, reconstruction, raw images and dot checks | Existing J/nonlinear restrictions through SEGMENTED contexts remain |
| Ordered multishot / actual CUDA MODE=2 | E independent sums, direct request equality and real files | Canonical acquisition, ascending physical-shot order, one rank/GPU |
| Flat FREE_SURF / CPML | Representative FS0/FW0, FS0/FW3, FS1/FW0, FS1/FW3 cases | No opposing same-axis CPML overlap or unrestricted grid/FD/CPML claim |
| All-solid preservation | Published M9e1–M9e4 regression gates and D canonical differential | Existing bounds/byte contracts preserved, not universal bit equality |

Representative principal wave fixtures use NX=24, NY=20, DH=1, DT=0.1;
B/C use NT=120 and D NT=36. E additionally verifies NT=19,23,43,97,
FULL/SEGMENTED selection and representative small/large storage fixtures.
These are finite published tests, not certification of every accepted input.
Prepared explosive source samples and direct vx/vy receivers obey the unchanged
canonical geometry/validation and timestep ordering. Existing first-surface-row
source rejection remains. No arbitrary source physics or receiver type follows.

## Material, restricted tangent and transpose

Fluid means **copied FP32 `mu == 0.0f`**, not epsilon, Vs floor, harmonic-value
threshold or caller-supplied mask. Both +0 and -0 are fluid; the smallest positive
FP32 subnormal (bits `0x00000001`) is solid, as verified on the real device.
Classification is owned background state, unaffected by caller buffer mutation.
Valid fluid retains finite positive rho/lambda and compressional stiffness.

Density, CPML profiles, acquisition and classification stay fixed in
linearization. `dMu[fluid] == 0` accepts either zero sign; **any nonzero** fluid
dMu, including a subnormal, rejects before tangent output/device mutation.
All dLambda columns and all solid dMu columns remain active. Invalid/nonfinite
directions are not silently projected into the manifold.

JT implements the restricted Euclidean transpose `<Jv,d> = <v,JT d>` with
the existing raw sign/scale contract: no fitted sign, DT/DH weighting, time
shift or normalization. After reverse/direct/surface/corner accumulation,
the internal device mu image is projected only at fluid cells, **before image
validity/publication**. Every fluid raw gMu has FP64 positive-zero bits
`0x0000000000000000`. Download is not a postprocessing projection. Fluid
interior gLambda and neighboring solid gMu remain nonzero where exercised.
No dRho/gRho or continuously changing material-topology claim is made.

### Harmonic corner graph

The frozen four-contributor harmonic rule branches before reciprocal/division:
any fluid contributor makes H exact +0 and restricted dH exact +0. Its VJP
contributes zero to **all four** owners, including solids. Final fluid-gMu
projection alone would leave erroneous adjacent-solid corner contributions
and is therefore insufficient. CUDA corner J and reverse `rharmonic` follow
the same copied contributor topology as CPU/reference, including wrap corners.
All 16 occupancies, both zero signs and actual subnormal solid behavior were
verified; all-solid expression/accumulation ordering remains unchanged.
Solid normal/surface sensitivity and incident all-solid corners remain active.
Fluid strains/normal waves are not masked or disconnected.

### Surface and absorbing boundary

At a valid fluid surface the existing closure gives alpha=1 and A=0.
Restricted dMu=0 makes their directional variations zero for fluid dLambda;
**generic unconstrained mu surface partials do not vanish**. Those partials
are retained in the reverse graph and fluid mu is finally restricted as above.
The frozen closure's top-row fluid raw gLambda is structurally +0; this is not
zeroing interior fluid lambda sensitivity. Incoming syy projection, stress
mirrors, velocity ghosts, source ordering and four strain operands stay frozen.
Only the representative flat surface and published non-overlap CPML envelope
are verified, not topography, arbitrary interface geometry or opposing same-axis
CPML overlap. Active CPML memory is directly evidenced, not inferred from a
diagnostic peak field that happens to be zero.

## Numerical evidence and unchanged acceptance budgets

Relative L2 is `||actual-reference||2 / ||reference||2` over the complete
specified products, not a masked fluid/solid domain. Grouped B products and
per-component C/D/E products are distinguished. All quantities below are
dimensionless errors, except an absolute dot residual in the native operand
units. Peak-normalized error is a separate metric, not interchangeable with L2.

| Stage / metric | Independently verified maximum | Unchanged ceiling | Source |
|---|---:|---|---|
| B independent grouped Forward L2 | `1.1860788932519425e-6` | `1e-5` | B §7; `evidence/fluid4b.json` |
| B independent per-component Forward L2 supplement | `1.6658035453615385e-6` | applicable frozen Forward bounds | B §7; `evidence/independent_checks.json` |
| B CPU/CUDA grouped Forward L2 | `8.780158734318872e-7` | `2e-6`, peak `8e-6` | B §7 |
| B separate acoustic receiver L2, nofma / fma | `3.969807336628684e-7` / `2.8839319322006683e-7` | production Forward `1e-5` | B §8 |
| C independent J L2 (four operands worst) | `1.3632655334428713e-6` | `1e-5`; peak `4e-5` | C item24; `fluid4c-evidence.json` |
| C centered restricted directional FD | `2.3325093689846599e-4` | `0.004`, fixed epsilon `0.05` | C item27 |
| D independent raw-image component L2 | `2.634019044792765e-7` | `6e-5` | D item20; `fluid4d-evidence.json` |
| D CPU/raw-image component L2 | `1.567794368750281e-7` | `6e-5` | D item21 |
| D restricted J/JT scaled discrepancy | `1.3581933283063122e-8` | `7e-5` times the defined scale | D item22 |
| D restricted J/JT absolute residual | `6.408388172874334e-11` | same scale-dependent absolute inequality, not a new fixed ceiling | D item22 |
| D restricted-basis scaled transpose | `1.1377945764234613e-7` | `7e-5` | D item23; 708 allowed nofma columns |
| D complete reverse-step relative L2 | `1.5253405999024321e-16` | `5e-13` | D item29; independent dense graph |
| E replay / independent image component L2 | `3.258226964114538e-7` | `6e-5` | E item14; `fluid4e-evidence.json` |
| E 1/2/3-shot request / independent component L2 | `6.060075475508585e-7` | `6e-5` | E item14 |
| E actual MODE=2 / independent component L2 | `4.718339550379426e-7` | `6e-5` | E item14 |
| E replay J/JT scaled discrepancy | `6.54028270847857e-9` | `7e-5` | E item13 |

Published CUDA D/E dot tests require
`abs(lhs-rhs) <= 7e-5 * max(||Jv||2*||d||2, hypot(||dLambda||2,||dMu||2)*||image||2, 1e-300)`.
Both sums use complete legal restricted directions and complete raw images.
Do not substitute a cancellation-sensitive quotient or a relaxed generic bound.
The frozen independent local FP64 dot bound remains `5e-13*operand_scale`.
Its separate Production-reference absolute gate remains
`1e-5*||Jref v||2*||d||2 + oracle_absolute_bound + 6e-5*||v||2*||JTref d||2`;
the CUDA scaled check does not replace that reference methodology.
Centered FD uses prescribed epsilon=0.05 on the established fixture, fixed
rho/classification/CPML, not a one-sided fluid-to-solid derivative or tuned step.
The homogeneous all-fluid solid-mu zero direction is a zero control, not a
nontrivial FD quotient. No tolerance, oracle or acceptance rule was changed.

## Exact invariants and provenance classes

- Positive-zero corner H/dH and mixed-corner VJP contributions are exact.
- Fluid internal raw gMu is exact FP64 +0; fluid lambda/adjacent solid mu
  activity is separately tested to exclude globally zero false positives.
- B nofma CPU/CUDA data, five fields, eight memories and four strain-tape
  channels are byte-identical for the five tested fluid fixtures. C equivalent
  nofma Born data/fields/memories/operands/corners and surface ghosts match CPU
  bytes. FMA instead uses published bounds; D JT is **not** claimed CPU-byte-equal.
- D direct migration equals prepare/JT/download byte-for-byte within each build.
- E FULL/SEGMENTED data, all 13 state channels, four strains and raw images
  are byte-identical within each build; repeated JT leaves checkpoints immutable.
- E ordered explicit per-shot CUDA sums equal request bytes; fresh requests
  and recovery retries reproduce accepted outputs for tested fixtures.

Independent mathematical FP64 references supply expected physics/material
products from prescribed inputs; they do not consume Production outputs.
CPU Production is a secondary differential comparator. Same-backend CUDA
equality checks determinism/replay but is not an independent physics oracle.
Real GPU execution used fresh fma/nofma CUDA 12.8 (NVCC12.8.93), sm_86,
RTX 3070 Laptop, driver591.44; no fast-math/FTZ change was introduced.

| Stage | Focused independent regression PASS | Scope |
|---|---:|---|
| B | 220 | New B plus four inherited all-solid CUDA modules |
| C | 314 | Adds restricted J |
| D | 365 | Adds restricted JT |
| E | 387 | Adds replay/request/MODE=2 |

These totals overlap: **do not add them**. Latest eight modules have
387 PASS, zero failures/errors/skips/XFAIL: inherited M9e1/e2/e3/e4
25/58/32/81, fluid B/C/D/E 24/94/51/22. External supplements close identified
coverage gaps and are not relabelled as additional durable pytest counts.

Hosted repository CI is a separate portability/publication gate. Its existing
Ubuntu workflow installs CPU dependencies, not CUDA hardware/toolchain; CUDA
suites skipped under their existing policy. The [PR100 Hosted run](https://github.com/RMecking/DENISE-Black-Edition/actions/runs/37766548874)
passed, but did **not** independently rerun GPU physics. This documentation task
does not run Hosted CI or rerun the 387 CUDA tests.

## Replay, checkpoints and storage

Automatic SEGMENTED selection requires retained storage **strictly smaller**
than FULL; ties select FULL. E independently tested equality 7x7x5/S=3:
3,920 bytes equals 3,920 bytes, selecting FULL. Its 51x43x97 case selects
SEGMENTED at 1,501,800 versus FULL 3,403,536 bytes. S=1 also falls back to FULL.
Existing segment boundaries cover [0,NT) once, effective E=min(S,NT), E-1
checkpoints and maximum length ceil(NT/E); uneven boundaries are checked.

Checkpoint capture is post-timestep, with metadata time=end-1 and canonical
version/dimension checks before restore. Payload retains five physical fields
and eight sparse CPML channels with actual support; inactive support is zero.
Banks stay immutable through probes and repeated JT. Restore/regeneration
reproduces complete physical/CPML state and all four strain channels, not a
fluid-masked approximation. No fluid-specific persistent host/device masks,
buffers, checkpoint payload or layout change: matched fluid/solid storage
ledgers agree. Replay JT is supported; J and nonlinear operations through
SEGMENTED replay contexts remain subject to their existing restrictions.

## Multishot, files and failure authority

Canonical source/receiver validation precedes work. Physical shot indices are
strictly ascending; both raw images accumulate in FP64 `sum += work` order,
without averaging, scaling or reordering. Each shot context is released before
the next; private sums become authoritative only after successful completion.
Actual CUDA MODE=2 emits the canonical lambda/mu raw file pair with no CPU
fallback. Complete arrays match direct CUDA requests; exact fluid gMu survives
accumulation and output. These are **raw migration images**, not an automatic
FWI objective/gradient, model update, optimizer or geological interpretation.

Accepted fault evidence includes invalid model/trial/direction/residual,
checkpoint capture/restore/regeneration, JT reverse/projection, shot2 after
shot1 success, image-download and final output allocation faults, ordinary
cleanup, empty unpublished results and successful fresh retries. The E
supplement uses actual auto-SEGMENTED shot2 faults as well as the smaller FULL
campaign; replay-filtered fault counts alone do not prove reverse coverage.
Ordinary failure exposes no partial final file pair or shot1 result. Invalid
buffers retain sentinels; unsupported backend/input combinations fail closed.

**A genuinely unrecoverable CUDA cleanup failure may leave resource handles
retained.** E's shared-cudart one-shot cudaFree injection proves recoverable
failure/retry only. Normal scientific runs use the static-cudart builds; the
supplement is not a repair or certification of permanent driver/device failures.
See E items21–25 and `extra-evidence.json`, `cleanup-evidence.json`,
`final-checks.json`; D items14–16 establish internal projection authority.

## Explicit exclusions and future capabilities

- Multi-rank CUDA fluid, multi-GPU fluid and M9e-5 completion.
- Arbitrary grid/FD/CPML outside published tests; opposing same-axis CPML
  overlap; topography/general geometry and arbitrary source/receiver physics.
- Changing material classification during linearization, fluid dMu,
  density derivatives/images, viscoelastic or anisotropic fluid.
- Blanket legacy MODE=0 fluid support; active MODE=1 fluid FWI
  objective/gradient; fluid-preserving optimizer, line search or model update;
  parameter-transformation gradients; FWI persistence/restart.
- Reflection-FWI/RWI and full general-purpose RTM formulation certification.

These are future work, not missing prerequisites that reopen the bounded
published CUDA capability. CPU FLUID-3F's tested MPI matrix remains separate.
No performance/scalability, arbitrary contrast or optimizer-convergence result
is inferred from the scientific regression counts.

## Reproduction and durable regression paths

Use the immutable 4E BASE and a separate build/output location on a CUDA-capable
Linux/WSL system. This is a reproduction recipe, **not tests run by DOCS-AUDIT**.
Build with the published options, including fma/nofma, CPU comparator and real
CUDA executable; the independent reports retain full expanded commands/hashes.

```sh
make -C libcseife -j4
make -C src -j4 denise denise_cuda cuda_m9_elastic_psv cuda_m9_elastic_psv_nofma \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCHS=86 M9_CUDA_BUILD_DIR="$cuda_build"
cc -std=c99 -O3 -Wall -Wextra -Werror -pedantic -fPIC -shared -Iinclude \
  tests/utilities/m9e2_cuda_free_surface_born_harness.c -lm -o "$cuda_build/cpu_e2.so"
DENISE_M9E2_BUILD="$cuda_build" PYTHONDONTWRITEBYTECODE=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest \
  tests/physics/test_m9e1_cuda_elastic_psv_forward.py \
  tests/physics/test_m9e2_cuda_free_surface_born.py \
  tests/physics/test_m9e3_cuda_jt_migration.py \
  tests/physics/test_m9e4_cuda_replay_mode2.py \
  tests/physics/test_m9_fluid_cuda_forward.py \
  tests/physics/test_m9_fluid_cuda_j.py \
  tests/physics/test_m9_fluid_cuda_jt.py \
  tests/physics/test_m9_fluid_cuda_replay_mode2.py \
  -q -ra --require-denise -p no:cacheprovider --basetemp "$pytest_output"
```

Set `cuda_build` and `pytest_output` to fresh absolute external directories
before use. Real executable fixtures use `mpiexec -n 1 bin/denise_cuda` with
their generated input/workflow files. Merely reading evidence JSON or getting a
dependency skip is not reproduction of GPU verification. Supplemental checkpoint,
subnormal, equality, shot2 reverse/projection/cleanup and MODE=2 rejection checks
are preserved in E's external scripts, not falsely claimed as repository tests.

## Candidate disposition

This ledger preserves published science, source identities, original limits and
known gaps; it changes no Production, test, oracle, ABI, workflow or governance.
FLUID-4F can be reported **DOCUMENTATION CANDIDATE GREEN — FINAL CUDA FLUID
ACCEPTANCE** only after documentation-health/path checks and evidence-to-claim
cross-checks pass. It is not CLOSED/CANONICAL. Freeze the reviewed candidate
unstaged, then obtain independent read-only cross-slice review; any source gap
or new scientific inconsistency triggers HOLD, not silent claim expansion.
