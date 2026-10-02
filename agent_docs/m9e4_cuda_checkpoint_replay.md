# M9e-4 compact CUDA checkpoint/replay and MODE=2

Implementation base / HEAD: `15683fc4c1666b60973be5fcfb2ef0f63cb43420`.
Branch: `codex/m9e4-cuda-replay-mode2`. This is an uncommitted candidate;
acceptance and publication belong to the user and independent review.

## C0: canonical contracts inspected before editing

CPU `elastic_psv_born.c` supplies the M9d1 segmentation and post-timestep
checkpoint boundary. M9d1 H1 compares complete retained differential storage,
rejects invalid estimates, and selects replay strictly below FULL. CPU
`elastic_psv_migration.c` uses `min(32, NT)` and ordered FP64 shot accumulation.
M9d3 proves that FREE_SURF=1 adds no persistent surface state. Derived mirrors,
COPY halos, velocity ghosts and alpha/A are reconstructed; their checkpointing
would change the canonical state definition. CPU production and its oracles
are unchanged.

M9e-3 retains four FP32 corrected strain operands per cell and timestep. Its
reverse graph uses FP64 fields, memories, operand bars and raw images. Only
the host loop supplying background operands changes here; all reverse kernels
and their accumulation order remain canonical. GPU replay calls the same
`extended_step` kernels as FULL. FS0 also retains the canonical source/sample
ordering and arithmetic.

MODE=2 dispatches directly from `physics_PSV.c` into
`denise_elastic_psv_migration_mode2`, bypassing the M8 `psv()` dispatcher.
The insertion point is the existing clean adapter's call to the migration
request driver, after its shared input loading/validation and before any
output writing. The optional build compiles that same adapter with
`DENISE_ENABLE_CUDA_M9_MODE2`. Ordinary `denise` compiles it without the macro.
M8 MODE=0/1 sources, objects and compilation flags are unchanged.

## Additive ABI and backend selection

Existing public structures, including `denise_cuda_m9_options`, retain their
sizes. The opaque private context may grow. Existing create/create_full/
create_migration constructors retain their FULL behavior and device budgets.

New `denise_cuda_m9_create_replay(config, options, S, automatic, out)` uses
forced replay for `automatic=0`, strict automatic selection for `automatic=1`.
New diagnostics, checked estimator, bounds and transactional test probes are
separate functions/structures. The clean request adapter has a separate
`denise_cuda_m9_migrate_request` entry point using the existing CPU request/
result ABI. Its result destructor uses ordinary free(), as does the shared
CPU destructor. Output image ownership passes to the caller only on success.

Replay supports prepare/JT/image download/migration. It explicitly rejects J,
nonlinear evaluation, FULL trajectory downloads and legacy FULL test-evolve/
trajectory-roundtrip APIs. The existing FULL J/JT path remains available.
Replay background receiver data can be downloaded; transient forward workspace
after JT is not misrepresented as the final NT state. New probes expose the
requested restored/recomputed state instead.

## Boundary, payload and deterministic schedule

For positive S, let E=min(S,T), K=E-1, M=ceil(T/E), T=NT, N=NX*NY.
Segment s covers `[floor(s*T/E), floor((s+1)*T/E))` using checked integer
arithmetic. Segments ascend and exactly cover `[0,T)`. The first segment
starts from zero without a synthetic initial checkpoint.

Checkpoint s is captured after the complete timestep `end[s]-1`, including
source insertion. Restoring it resumes at `start[s+1]`. Metadata validates
version, completed time, dimensions, FW, free surface, CPML and value count.
Metadata and schedules are contiguous host ownership; no checkpoint pointer
table is allocated.

Let a[k] be the number of coordinates with nonzero actual built CPML profile
coefficient a for channel k. Profiles are `(XH,Y,X,YH,X,XH,YH,Y)`.
The value count per checkpoint is

`V = 5*N + NY*(a[0]+a[2]+a[4]+a[5]) + NX*(a[1]+a[3]+a[6]+a[7])`.

Payload is exactly V FP32 values: five physical fields, then eight active
memories. X memories traverse row then active column; Y memories traverse
column then active row, matching CPU logical packing. The built mask handles
the extra half-grid right coordinate and FS1 top-Y identity correctly. With
CPML disabled, V=5N. Capture rejects nonzero omitted inactive memories.
Restore clears padded fields/memories then copies physical/active values.
Ghosts, mirrors, Q operands, material maps, profiles, sources, geometry,
alpha/A, images and reverse state are not checkpointed.

Initial prepare performs T forward steps and captures K checkpoints. Its
four-operand buffer has only M timesteps. JT visits segments in descending
order, restores their start, regenerates each segment with absolute source/
sample time and relative operand index, then consumes its operands descending.
The FP64 reverse state persists across segments. Every completed JT replays
exactly T steps. Checkpoints remain immutable, including across repeated JT.

FS1 replay preserves projection, stress COPY/mirrors, momentum/CPML, receiver
sample, velocity COPY, horizontal corrected operands, material-dependent
velocity ghosts, vertical corrected operands, constitutive update, top A
closure and source insertion. There is no additional surface state.

## Complete memory accounting

All following sizes are requested allocation bytes, including explicit
alignment and new retained allocator prefix, not undocumented driver/malloc
bookkeeping. On this tested LP64 build, M9Replay is 272 bytes, checkpoint
record is 40 bytes, and tracked host allocation prefix is 16 bytes.

Differential FULL storage is `F=16*T*N`.

Replay retained storage is

`R = 4*K*V + (272+16+40*K) + 0 + 8*E + 16*M*N + C`,

where `C=(8-(4*K*V)%8)%8`. The terms are checkpoint device payload, complete
host owner/record metadata including its allocation prefix, allocated pointer
table (zero), host schedule, device segment operands, and device payload
alignment. These are actual allocations. The owner contains its small pointer
members; these are already counted in 272 bytes, not a separate pointer table.

Automatic selection is replay iff R<F. Tie is FULL. Invalid/overflow estimates
fail closed and clear all output diagnostics; no allocation is attempted by
the estimator. Forced replay may deliberately use more memory, including S=1.
MODE=2 uses the canonical target min(32,T), without a new input parameter.

For padded P=(NX+4)*(NY+4), receiver count r, the common forward/J storage,
excluding FULL/replay operands, is

`B = 124*P + 24*(NX+NY) + 4*T + 12*r + 16*T*r + 28*N`.

The common FP64 reverse arena is `40*P+112*N+8*T*r`, preceded by
`A=(8-B%8)%8`. FULL device ownership is `B+F+A+40*P+112*N+8*T*r`;
replay ownership is `B+16*M*N+4*K*V+C+A+40*P+112*N+8*T*r`.
Both have one cudaMalloc, the existing two events, and the same fixed J/JT
workspaces. No hidden checkpoint or segment cudaMalloc exists. Arena
arithmetic, payload, metadata and schedule formulas use checked size arithmetic.

The private canonical host bridge retains the same static model/source/
profile/geometry preparation as M9e-3, never a CPU forward tape. The canonical
request validator is compiled privately and invoked directly; its CPU
migration function is never invoked by the CUDA request adapter.

Transactional host scratch is temporary and separate from retained R:
image download 16N bytes +16-byte prefix; checkpoint download 4KV+16;
state-only probe 52N+16; state/Q probe 68N+16; background data download 8Tr+16.
Existing static upload scratch is 20P+24(NX+NY)+12r plus three prefixes.
Request sum/work scratch is 32N plus two prefixes; successful published images
are 16N ordinary caller-owned bytes, temporarily concurrent with sum/work.
The ledger excludes those published caller-owned buffers. The shared file
adapter also owns its existing loaded model, geometry, source and prepared
data buffers; these are neither a hidden GPU tape nor checkpoint retention.

## MODE=2 file and failure contract

`denise` remains CPU/MPI, without CUDA linkage. Optional `denise_cuda` chooses
CUDA for PHYSICS=1/MODE=2; log markers identify CPU-M9 and CUDA-M9e-4. There is
no CUDA-init failure fallback. More than one MPI rank or a non-1x1 topology
fails with the explicit CUDA-M9e-4 unsupported-envelope diagnostic.

The same adapter loads x-major `.lam/.mu/.rho` models into the row-major
operator, neutral source geometry, prepared `.shot_s.bin` source samples,
and chronological `.vx/.vy.shot_s.bin` data. It retains canonical geometry
validation and prepared-source restrictions. Outputs keep canonical
`.image_lambda_raw.bin` and `.image_mu_raw.bin` names/layout: raw FP64
row-major lambda/mu images, without filtering/scaling/conditioning.

Shots execute in strictly ascending physical index with their own source and
data and common model, using create/execute/destroy per shot. Images accumulate
in deterministic host FP64 order. A shot failure returns no result, destroys
ownership, and reports its physical index and first meaningful error. Final
files are written only after all shots succeed through the existing temporary
pair publication/cleanup. Existing failed-run semantics remove stale final
pairs. A clean rerun equals a fresh run. CPML peak monitoring remains a CPU-only
diagnostic; CUDA explicitly labels it unsampled, rather than claiming a zero
physical peak.

Required envelope: elastic P/SV, L=0, INVMAT1=3, FDORDER=4, NDT=DTINV=1,
BOUNDARY=0, FS0/1, fixed rho, direct vx/vy, INV_STF=0, explosive prepared source,
one rank/GPU. No MPI multi-GPU, viscoelasticity, density/Q/Vp/Vs images, source
inversion, FWI residual construction or image conditioning is claimed.

## Verification and provenance

Durable suite: `tests/physics/test_m9e4_cuda_replay_mode2.py`. It checks both
CUDA arithmetic builds, FS0/FS1 and CPML off/on, S=1,2,3,7,32,NT,>NT, several
nondivisible schedules, CPU logical payload parity, bitwise data/restored
fields/all-eight memories/Q4/images, repeatability, checkpoint immutability,
frozen-J/replay-JT dot closure, CPU image ceiling 6e-5, complete metadata,
strict automatic selection, pure SIZE_MAX overflow, executable routing,
multi-shot order and transactional failures. No oracle or tolerance is changed.

New-operation fault injection is scoped to retained host allocation, schedule,
capture/restore, replay timesteps and migration request/publication. Legacy
fault counting remains unchanged. The complete M9e-1/2/3 suites are also run
because shared owner creation/destruction changes; their existing old fault
matrix is therefore justified. No unrelated M8/SH suite is run.

The host ASan/UBSan harness exercises actual estimator/metadata allocation,
canonical preparation and request failure cleanup without claiming GPU
execution. Its unavailable-backend stubs are explicit. Shared MODE=2 I/O is
also instrumented and its relevant CPU integration subset runs. That MPI
process uses ASan leak detection disabled for external MPI ownership; the
isolated host harness separately enables leak detection. CUDA host-side UBSan
exercises actual GPU contexts/fault paths and the lifecycle harness. Neither
host sanitizer claims device memory instrumentation.

Compute-Sanitizer reaches the real replay lifecycle but reports failure to
initialize the WDDM debugger interface and device-not-supported errors.
Classification: **ENVIRONMENT / DRIVER LIMITATION**, never PASS (even if its
process exit code is zero). No Administrator command is executed.

The consolidated external report records exact counts, numerical/performance
diagnostics, raw logs, environment, SHA256/raw/canonical blobs and EOL/attributes
for every candidate path. Because publication is prohibited, evidence is tied
to the unchanged base SHA plus that complete content manifest, not a fabricated
candidate commit SHA. Independent scientific verification remains required.
