# M9-FLUID-1B CPU zero-shear forward and capability boundaries

Implementation BASE/HEAD: `51fdf128d25615c34afcb4336e59c2e48eb43316`.
Branch: `codex/m9-fluid-1b-cpu-forward`. Uncommitted candidate.

## Frozen contract and production scope

The six canonical FLUID-1A authority files are immutable and retain their BASE
Git blobs. Actual production products are passed to their independent helpers;
neither expected values nor acceptance ceilings are modified.

Five production files change: serial and MPI CPU Born contexts, MODE=2 CFL
validation, the published CUDA host preparation bridge and CUDA nonlinear host
preflight. No legacy PSV, spatial operator, CUDA numerical kernel, migration
file convention, public ABI or checkpoint layout changes.

Classification is exactly copied canonical FP32 background mu==0 versus mu>0.
It is derived from the existing immutable context-owned mu array; there is no
new mask, allocation, threshold or Vs input. Thus memory/lifecycle accounting
and checkpoint payload remain unchanged. Negative mu still rejects.
Fluid requires finite positive rho and lambda (D=lambda at mu0). Additional
validity checks apply only to fluid; the unrelated all-solid validator envelope,
including valid negative-lambda solids, is retained.

Both constructor maps and forward/nonlinear corner construction choose exact
positive zero before any reciprocal if any of the four physical contributors
is zero. Otherwise the original harmonic expression and ordering execute
unchanged. The two existing arithmetic density-face maps are unchanged.

No special fluid state equation is introduced: normal stresses naturally become
lambda times divergence, and zero corner shear stiffness makes shear increments
zero. Nonzero fluid strains are retained in FULL and replay.

Nonlinear trials must preserve background zero/nonzero classification exactly,
in both directions, before output or state work. MPI validation agrees invalid
material/classification collectively before trial material halo transport.
Same-class trials keep fixed rho/CPML/acquisition and recover deterministically
after a rejected request. A new topology needs a new context.

Serial J/JT scan the copied background before output zeroing, trajectory/replay
mutation or singular tangent/transpose arithmetic. MPI performs the collective
capability check before such work, including an unprepared fluid context and
fluid present on only one rank. Diagnostics explicitly require FLUID-2.
The complete fluid J/JT API is blocked, even for lambda-only/zero directions.

MODE=2 recognizes fluid material and the existing cp-squared/CFL formula,
including unchanged FD4 bound29/12. It then fails transactionally through the
J/JT capability guard; it publishes no raw images. Existing migration request
cleanup supplies this boundary without a new migration algorithm.

The published CUDA host bridge includes CPU static construction, so CPU changes
would otherwise admit fluid automatically. Its narrow host guard rejects mu0
with FLUID-4. CUDA nonlinear trial preflight rejects before invalidating an
already prepared solid context. Constructors FULL/forward/migration/replay all
remain blocked. No GPU material/tangent/reverse/surface/replay numerics change.
The frozen unpublished M9e-5 worktree is not inspected or touched.

## Actual production verification

Local environment: Ubuntu24.04.5 LTS/WSL2 kernel6.18.40.1, GCC13.3.0,
OpenMPI4.1.6, NumPy2.5.3, published single RTX3070 Laptop CUDA12.8 toolchain.
Ordinary CPU executable builds and has no CUDA dependency. Narrow single-GPU
CUDA libraries compile for the boundary tests; no multi-GPU or NESH work.

| Gate | Result |
| --- | --- |
| 0/1/2/3/4 zero contributors | All16 occupancy patterns: +0 for any fluid, exact canonical solid result otherwise |
| Wrapping/density | Horizontal/vertical/isolated/strip/pocket/edge/solid maps equal frozen maps after FP32 rounding; +0 sign checked |
| Validator-only mutant | Actual accepted fluid constructor with restored reciprocal expression raises FE_DIVBYZERO/INVALID; killed |
| Homogeneous fluid F | Frozen24x20/DH1/DT.1/NT120/rho1/lambda4/mu0: relL2 3.9564635148e-7 <=1e-5 |
| Homogeneous state | Finite, equal normal channels, sxy0, nonzero strains (norm.77564937); actual prepare data identical to inspected production timestep run |
| FULL/replay fluid F | S1/3/7, checkpoint continuation and strain access work; J/JT remain blocked without changing diagnostics/outputs |
| Nonlinear trials | Valid same class; fluid->solid and solid->fluid reject; invalid mu/lambda reject; sentinels and prepared context retained; subsequent valid result bitwise |
| Negative/invalid material | Serial negative mu, zero rho, fluid nonpositive/Inf lambda, Inf rho reject; one-rank-negative MPI collectively rejects |
| Fluid J/JT | Serial/MPI outputs unchanged, no tangent/reverse path; post-failure F succeeds; fluid on only one rank still blocks every rank |
| FS/CPML data | Four64x64/NT800 homogeneous/interface FS0/1 cases: max relL2 7.0336010221e-7 <=1e-5 |
| CPML late norm | .0000002902034/.0024319724/.0000006167254/.0230393109, each <=.05 |
| FS graph | Actual fluid first-surface-row source rejects; top normal projections preserved; FS1 top-y memories0; side/bottom memories active |
| Preliminary horizontal interface | Mixed corner row+0; first solid corner row>0; finite P response, solid off-axis shear/curl, no fluid shear stress artifact |
| Interface comparison | Point-shot actual data relL2 4.4804399432e-7; full final state relL2 1.5459640093e-6 vs frozen constitutive reference |
| CPU MPI actual halos | 2x1/1x2/2x2 horizontal/vertical/one-rank-fluid, FS0/1; added2x2 CPML/interface cases; gathered maps and F bitwise serial |
| MPI independent data error | Maximum4.8767078499e-7 <=1e-5; FULL/replay and collective failure/recovery pass |
| MODE=2 | Valid fluid passes material/CFL, explicitly fails FLUID-2; no pair output; subsequent solid request publishes normally (1/2ranks, FS0/1) |
| CUDA boundary | Six gates pass on published fma/nofma libraries; fluid construction blocked; trial failure retains solid prepared data/state/trajectory |
| All-solid exact BASE | Eight FS0/1 x CPML0/3 x FULL/replay3 cases: maps,F,J,JT,nonlinear,state,strains bitwise BASE |
| Relevant all-solid CPU | 137 frozen F/J/JT/CPML/FS/replay/MPI gates pass; final MPI ordering rechecked by10 overlapping relevant gates |
| Relevant published CUDA | 44 F/nonlinear/J/JT/FS/CPML/replay gates pass |
| Canonical fingerprints | Existing six Linux/NumPy2.5.3 fingerprints pass unchanged |
| Frozen FLUID-1A suites | All45 oracle/interface tests pass unchanged |

New local tests:75 CPU/MPI plus1 fluid-source boundary plus6 CUDA=82 pass.
Relevant existing gates:137 CPU+44CUDA; immutable oracle45 and fingerprint1.
Total distinct pytest gates309 pass; eight additional BASE identity comparisons.
The final10MPI repeat gates overlap the137 and are not added to that count.
Two expected warnings in deliberately invalid test-fixture damping-speed
calculation are not production singular arithmetic; actual material/forward
FE_DIVBYZERO/INVALID guards pass. No sanitizer or performance claim is made.

The preliminary interface result supports bounded classification B. It is not
a point-source-to-plane-coefficient amplitude comparison. No coherent-source
API is added and the frozen2% physical refinement gate is not weakened or
claimed as completed production refinement; that campaign remains FLUID-3.
No evidence requires classification C or a new spatial coupling graph.

## Candidate freeze and boundaries

The report and external manifest record exact raw SHA256, unfiltered/canonical
Git blobs and EOL/attributes for every candidate file, with canonical six-file
authority identity and Git-derived stats. HEAD alone denotes BASE, not the
uncommitted tested candidate. Nothing staged; diff check passes. No commit,
push or PR. Primary dirty checkout and other worktrees are preserved.

Tolerance changes: NO. Oracle changes: NO. Checkpoint format changed: NO.
Public API changed: NO. M9e-5 touched: NO.

Fluid Born/JT/migration remains FLUID-2. Exhaustive interface/CPML/transpose
refinement remains FLUID-3. CUDA fluid numerics remains FLUID-4.

M9-FLUID-1B READY FOR INDEPENDENT SCIENTIFIC VERIFICATION
