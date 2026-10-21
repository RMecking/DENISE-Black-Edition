# DENISE — M9L: Modern Elastic FD8 Extension & Legacy Replacement

**Current status (2026-10-20):** bounded Taylor8 Forward Foundation published;
private R&D partly independently qualified; full FD8 workflow not integrated.
See the [current R&D status and evidence](m9l_rd_progress.md).
No numerical work, publication or new oracle acceptance is authorized here.
The original planning text below is retained as the **2026-10-08 historical
A2/Holberg proposal**, including its then-current kickoff/publication wording;
it is not the execution status of later Taylor8 work.
**Prepared:** 2026-10-08.
**Repository authority inspected:** `RMecking/DENISE-Black-Edition`, `modernization@4eded044e370461644b359039662ca6c1bd4477f`.
**Roadmap source:** `docs/modernization_status.md`, Git blob `ce8269d273e8d76a56c375b20f615b046ce34eb3`.
**Precedence:** Complete current in-flight Modernization work and existing acceptance/publication gates before implementing M9L. Existing merged scientific claims retain only their original tested envelopes.

## Strategic decision

This and the subsequent original planning sections describe the dated plan,
not a retrospective qualification of Holberg by Taylor8 results. The current
reconciliation and dependencies follow at the end of this document.

Choose **A2**, extension of the verified **modern M9 CPU elastic P/SV operator** to the FD8/Holberg family, not a long-lived second exact-adjoint implementation inside the legacy FD8 reverse-time path. Preserve M9 FD4 behavior; the modern FD4 P/SV J/JT/CPML/MPI/Free-Surface/checkpointing infrastructure and documented contracts are reusable *architectural* assets, not automatic evidence of FD8 correctness. CUDA/accelerator extension is not a prerequisite for FD8 science. Plan the modern FD8 path as an explicitly selected operator with unsupported configurations failing closed.

**Scientific priority:** demonstrably correct and usable forward modelling, FWI and RTM in declared configurations before new research features, GPU performance optimization or RWI/M10. Do not treat a successful J/JT dot product alone as proof of physical fidelity or full FWI/RTM correctness.

**Legacy preservation:** The original and corrected legacy FD8 source, binaries, inputs and frozen histories shall be kept for scientific reproduction. Upon accepted replacement, mark *deprecated*, document limitations and route/compatibility, and archive immutably. **Do not delete legacy FD8.** No deprecation before replacement is accepted across the declared workflows/configurations.

## Transferred benchmark evidence — authentication pending in M9L-0

The following A3.0 conclusions are transferred handover evidence, not newly
independently authenticated results from this documentation task. M9L-0 must
locate and authenticate original/corrected sources, binaries, inputs, manifests
and evidence before relying on them. Gate C does not conclusively explain or
repair historical Check 7. Corrected Forward does not independently certify
reflection suppression. No benchmark was modified or rerun.

## Why this belongs to modernization

The independent A3.0 physical-water benchmark identified a legacy `abs(float)` CPML coefficient defect; isolated eight-guard `fabsf` Gate A coefficient verification PASSED, and Gate B bounded corrected forward propagation PASSED. Reflection quality remained unresolved. Gate C identified actual elastic FD8/L0 reverse CPML A-vs-A^T behavior and STOPPED before full gradient runs. Follow-up design proved a CPML-only adjoint patch insufficient: missing full field and eight memory cotangents, reverse composition, halo transpose, and time-aligned source/receiver/material dependencies. No complete FD8 gradient correctness/failure rate was measured; historical Check 7 remains **FAIL/BLOCKED/STOP**, Checks 8–9 not executed. Historic evidence is frozen and does not automatically certify a new operator.

The existing modern M9 FD4 restricted, fixed-density `lambda/mu` J/JT (including specific zero-shear fluid, MPI, free-surface, CUDA subsets) and M7 exact-FWI achievements demonstrate reusable *design patterns*, but differ in operator, material and supported-envelope details. Never replace legacy FD8 numerical identity with M9 FD4 results by implication.

## Proposed phased work packages

| Milestone | Implementation / scope | Mandatory scientific exit |
|---|---|---|
| **M9L-0 — Architecture & independent verification freeze** | Read-only contract comparison: corrected legacy FD8 versus modern M9 FD4; decide compatibility/dependencies, initial FD8 scope, independent oracles, run budgets and fail-closed limits. | Reviewed target discrete graph, precision/material/CPML semantics, agreed validation plan and smallest first implementation slice. **No implementation.** |
| **M9L-1 — FD8 nonlinear forward** | Add isolated FD8/Holberg stencil selection to modern CPU M9 P/SV. Initially homogeneous/all-solid, one rank, nonperiodic, FS=0, then corrected CPML. | Independently checked stencil symbol/dispersion, analytic or manufactured tests, mesh/time convergence, CPML comparisons, retained FD4 regressions. |
| **M9L-2 — Exact FD8 state J/JT** | Exact tangent and full augmented-state transpose (five elastic fields + eight CPML memories), time-order, receiver/source transposes, checkpoint consistency. | Kernel/full-step/two-step dot products; full-time J/JT; independent finite-difference J, seeded CPML memories, precision budget. No implicit FD4 oracle inheritance. |
| **M9L-3 — Material VJPs and physical water** | Verified lambda/mu with fixed density first; deliberate extension to rho and Vp/Vs/rho chain if approved; zero-shear fluid/solid interface and admissible parameter directions. | Independent gradient direction FD, exact zero-shear constraints, correct harmonic averaging and chain rule, all-solid preservation. Do **not** presume fluid `dMu` or rho directions supported in M9. |
| **M9L-4 — Active FWI** | Explicit modern FD8 selection for objective, gradient, accepted model update, persistence/reload; retain legacy selector. | End-to-end signed gradient and descent checks, multi-iteration controlled reconstruction and model-error improvement, stage/precision accounting. |
| **M9L-5 — RTM scientific integration** | Integrate modern FD8 Born adjoint to raw migration; freeze imaging condition, wave-mode/content convention, acquisition and any post-processing separately. | J^T dot identity **and** independent point-scatterer/planar-reflector imaging controls, spatial positioning, relative amplitudes and artifact diagnostics. |
| **M9L-6 — Supported-envelope expansion** | FD8 halo width and transpose, 1x1→MPI test topologies, flat free surface and explicit boundary variants, as prioritized. | Per-configuration MPI/FS/CPML dot identities and physical validations; no blanket support claim. |
| **M9L-7 — Controlled legacy deprecation/archive** | Route supported workflows to accepted modern FD8 path; document unsupported cases and archival reproduction. | Formal replacement coverage, scientific acceptance, publication, migration guide, immutable legacy archive. **No deletion.** |

M9L-4 and M9L-5 may be partly parallel **only after** the necessary M9L-2/M9L-3 gates. M9L-6 may be pulled forward for explicitly required fixtures, but no premature scientific PASS. M10 Reflection-FWI/RWI stays downstream and need not set the M9L schedule.

## Mandatory independent verification matrix (freeze in M9L-0)

1. **Operator and precision contract:** FD8 optimized Holberg coefficient source, staggering, halo widths, `DT/DH`, source/receiver timing, binary32 operation order, material averaging and parameter maps; record analytic stencil Fourier symbol. Optimized Holberg weights must not be assumed to give a nominal asymptotic order solely because labeled FD8.
2. **FD4-versus-FD8 comparison:** same continuum PDE, physical coordinates, source distribution and observable, with decreasing `DH` and separately controlled `DT`; compare velocities, phases, P/SV polarization, arrival times and energy norms. Do not demand equality on a common coarse grid; expect convergence, quantify an independently justified error budget. Test FD4 baseline unchanged for its frozen scope.
3. **Independent physical references:** analytic homogeneous plane waves/Green functions where suitable, known layered interface reflection/transmission (P/SV and acoustic–elastic interface, including true zero-shear water), and manufactured solutions for smooth bulk operators. Source/receiver conventions, 2D geometry and temporal convolution must match.
4. **External cross-code:** e.g. SPECFEM2D, as an *optional independently sourced comparator*, after comparing equations, boundary treatment, attenuation, source strength/time representation, receiver quantity and numerical convergence. A third code does not replace analytic or internal identities.
5. **Corrected CPML:** coefficient-threshold oracle (the A/B eight-guard correction), full/half strips and corners; isolated incident/return wave using compact CPML domain versus much larger reference domain (same discretization, unaffected common interior, reflection-free observation interval); vary wave incidence, CPML thickness and damping. Report reflection error independent of direct/water-interface scattering. Earlier Gate B reflection quality is **UNRESOLVED**.
6. **Discrete mathematical identities:** local forward/J/JT, CPML temporal-memory transposes, complete timestep and multi-timestep augmented-state dot products, parameter map VJPs and actual full-objective gradient central differences. Fix norms, tolerances, algebraic FP64 contractions and admissible FP32 epsilon ranges **before** outputs; no sign/time shift/empirical scaling fits.
7. **Scientific application gates:** FWI tests with independent truth and a controlled multi-iteration reconstruction (misfit, model error, accepted-trial and reload identities), and RTM Born point-scatterer/planar-reflector reference tests plus a physically motivated imaging-condition audit. A decrease in misfit alone or a J^T identity alone is insufficient.

Prioritize analytic/dispersion/MMS and algebraic gates locally before committing to large external-solver campaigns. Establish a low-cost bounded reference matrix first, add physically complex and MPI cases in later authorized slices. A reference built from old FD8 is historical/diagnostic; a corrected legacy FD8 forward is a **comparison**, never sole source of truth.

## Publication/governance boundary

- Finish the currently active Modernization package before M9L-0 begins; do not rewrite existing M9/FLUID milestone claims or source trees.
- Freeze actual `modernization` HEAD anew at kickoff; the SHA in this proposal records the inspected repository, not a forever-locked base.
- Keep A3.0 historical original binaries, manifests and `Check 7 FAIL/BLOCKED/STOP` immutable; no automatic Checks 8–9 or A3.0b.
- Each work package follows the existing Codex worker / independent SCIENTIFIC-VERIFICATION / content lock / publication authority governance, preserving no auto commit/push/merge.
- A later explicit task may authorize **M9L-0 only**: read-only architecture/verification-contract audit, no numerical source edit, no solver campaign. This roadmap does not authorize even that task to begin.

## M9L-0 acceptance deliverables

1. Exact live branch SHA and provenance for M9 source, legacy BASE, Gate A/B/C, and feasibility report.
2. Machine-readable FD4 vs corrected-FD8 operator-difference inventory; stable, corrected and intentionally different semantics separately identified.
3. Explicit supported-envelope and parameterization matrix including FD8 one-rank/no-FS first, future MPI/FS, fluid fixed-density restrictions and rho roadmap.
4. Frozen independent validation catalog, reference implementations/physics, precision and cost bounds, scientific PASS/FAIL/UNRESOLVED policy, no post-hoc fitting.
5. Reuse-vs-reimplement decision, architecture risks, test dependencies, minimal M9L-1 scope and blockers.
6. Scientific review recommendation `M9L-0 — ARCHITECTURE CONTRACT READY` or `M9L-0 — BLOCKED`, with immutable evidence; no implied implementation/publication authority.

## Integration sources and provenance boundary

This candidate integrates the complete user-approved handover proposals
`M9L_ROADMAP.md`, `ROADMAP_STATUS_INSERT.md` and
`DENISE_HAUPTCHAT_UEBERGABE.md` from `C:/Users/rebme/Downloads/`.
The user explicitly confirmed the unsuffixed files in place of the task's
`(1)` filenames. They are proposals, not previously published repository content.
RAW SHA256 identities:

| Proposal | RAW SHA256 |
|---|---|
| M9L_ROADMAP.md | `6c7526fa6609a2df8a11f66a7300511fd7930f2d21021e18feea2680cbeb6267` |
| ROADMAP_STATUS_INSERT.md | `49e7483f42e250701f7a355c35f658ff62a692857be1b000a8560667310120ec` |
| DENISE_HAUPTCHAT_UEBERGABE.md | `c21ad68f7aa235e7133d564f1a464dca5de889f63a54457eb9953c8ca7a65ee1` |

The [modernization ledger](modernization_status.md) records FLUID-4F POST-MERGE
PASS — CLOSED / CANONICAL at `4eded044e370461644b359039662ca6c1bd4477f`,
[PR #101](https://github.com/RMecking/DENISE-Black-Edition/pull/101).
This is a recorded milestone identity, not a permanently current branch HEAD.
The [CUDA-fluid acceptance ledger](m9_fluid_4_cuda_scientific_acceptance.md)
retains the earlier candidate's scientific scope/evidence; its historical
candidate wording is not silently rewritten here. Modern restricted raw
migration does not certify generic legacy RTM or active fluid FWI.
M8e remains planned/deferred, not completed; already published M8 and M9
scientific evidence keeps its original envelope. The new priority is correct
Forward, exact FWI and scientifically valid RTM before additional performance/
features and M10 Reflection-FWI/RWI.

Transferred benchmark reference identities (claimed by the handover; **pending
authentication in M9L-0**, not reverified by this roadmap):

| Artifact | Transferred identity |
|---|---|
| Legacy BASE | `eac30a641b240e3f80f83fb3208947f9b15a68cf` |
| Original executable SHA256 | `498b3da66556c450038b6378744534d4ee39d5fe13487200bec97af84a03cdf2` |
| Gate-A corrected PML_pro.c RAW SHA256 | `b58439f1ad85693a076aabf0075760a80f2610c80cef003c4d0890e9c56954d7` |
| Gate-B executable SHA256 | `282abc53054be61b9a3eedb1065971a4db57df13ed29a0d66579b49e0455126d` |
| Gate-C final manifest SHA256 | `5903df0d5aa3783abef4249dfe010b546a6372c5f9790b1b78a4057bccc83f64` |
| CPML-ELASTIC-ADJOINT-01 final manifest SHA256 | `96486442dde5abed998b40a4dcb74ae497bcfc5470e37eca6dae08c76bc241c5` |

No artifact path is invented. The handover says the evidence paths are retained
in the benchmark project; M9L-0 must resolve them. Historical Check 7's reported
Taylor R1=2.6192195948155774 exceeds its fixed 2.5 ceiling: FAIL/BLOCKED/STOP.
Checks 8–9/A3.0b were not executed. These reported numerical observations are
not a newly accepted oracle or a causal explanation of Check 7.

## Current Taylor8 research path — reconciled 2026-10-20

The historical A2 architecture direction remains distinct from coefficient
qualification. The published additive Taylor8 homogeneous-solid Forward
Foundation is PR #103, merge `02b46e8fe5528054ac4643f836b9261d5dd27832`;
the inspected integration HEAD is `02f4b96c813e1fe0e333d9257488eb0efa5e1164`.
The subsequent private research lineage is state/source adjoint (M9L-2A),
fixed-profile augmented CPML transpose (2B), fixed-density lambda/mu material
J/JT (3), restricted functional FWI (4), corrective sidecar controls (4S2),
GN diagnosis and corrected adaptive controller (4G/4G-D3/4G.1), and raw
migration/physical diagnostics (5/5P/5P1). These are separate frozen candidates
with restricted review verdicts, not one released end-to-end product.
The [R&D overview](m9l_rd_progress.md) binds their original independent reports
and keeps the failed frozen 4G trajectory and 5P-P3 amplitude gate visible.

REAL-01 preparation/design is not native qualification. REAL-02A adds four
small SI/toy material configurations; 02B qualifies limited private basis
storage, not material-tape streaming or large SI FWI. Corrected 02C-S0 binds
the CPML adapter and controller but has no controller-run/GN/FWI qualification.
02D's partial three-shot diagnosis stopped after its sole shot-2 recovery also
failed before execution; no complete cross-image diagnosis or P3 closure exists.

### Next dependencies and decision points (not execution permission)

| Workstream | Current status / required next evidence |
|---|---|
| REAL-03A — R1 SI sensitivity preflight | D1, 2026-10-21: independent **SCOPED PREFLIGHT PASS**, four shots F4/J16/JT4, 24 sweeps, both original pairs PASS; approximately 131.414 native seconds, observed VmPeak maximum 360058880 bytes. No GN/FWI or general operator/long-time/dispersion/reconstruction qualification; call allowances consumed. |
| REAL-03B — GN SI precision/scaling contract | Independent **SCOPED DESIGN/PRECISION REVIEW PASS**; retain GN system and regularization metric. Proposed residual/uncertainty contract needs private implementation and qualification; no changed S/alpha/stopping or native GN execution. |
| REAL-04A — material-tape storage / SI-GN integration | **PLANNED/AUTHORIZED BY TASK PACKET**, no authenticated execution result/PASS included. Qualify history ownership/order/replay and resource bounds with small fixtures; 02B basis storage alone does not supply this capability. |
| REAL-04B — same-system GN precision diagnostics | **PLANNED/AUTHORIZED BY TASK PACKET**, no authenticated execution result/PASS included. Implement separate diagnostics/conditional certification, preserving system/metric/solver rules; no native campaign authorized. |
| CPML-S1, then CPML-active FWI | Proposed beyond accepted S0. Separate execution authority, fixed-profile scientific/resource gates and actual controller-run evidence are required. S0 PASS is not S1 permission. |
| RTM diagnosis | 02D INCOMPLETE/HOLD; recovery authority exhausted. Resolve remaining diagnosis through a separately authorized bounded package; keep 5P-P3 HOLD and 5P1 resource NONCOMPLIANCE. |
| Controlled R1 IC0/DN1 FWI, then broader physical references/R2/R3 | Only after relevant preflight, precision, storage, boundary/reference and independent application gates. No large-FWI or reference-generation permission follows from this plan. |

REAL-03A's scoped native PASS and REAL-03B's design PASS are different gates,
not full workflow qualification or authority to orchestrate another worker.
LEGACY-CPML-01 is separate: proven float-abs guards in both legacy profile
generators and suspected viscoelastic SH top-staggering investigation, without
a completed new repair/physical qualification. Historical A3.0/Check-7 HOLDs
remain, and no legacy finding automatically qualifies or indicts modern
Taylor8. Exact D1 source identities are in the R&D overview.
Source reviews, frozen criteria,
resource limits and explicit packets govern each dependent step. No fresh
attempt is authorized by unused time after a consumed permit. Full CPML-FWI,
final RTM qualification, MPI/free-surface/fluid/density expansion and accepted
legacy replacement remain open. M10 stays downstream. No historical HOLD,
oracle, acceptance threshold or legacy archive is rewritten by this update.
