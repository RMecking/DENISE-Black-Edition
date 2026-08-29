# DENISE Modernization Status

## Purpose

This document is the concise modernization ledger. It records published and
locked checkpoints, the active milestone, the next planned checkpoint,
durable acceptance criteria, intentionally unresolved RED evidence, and the
roadmap state. It does not replace detailed design/audit documents or Git
history.

## Source-of-truth hierarchy

1. Published Git objects and exact commit SHAs are authoritative for what was
   committed.
2. Frozen/versioned tests, validation artifacts, and acceptance contracts are
   authoritative for the mathematical or verification claims they define.
3. Detailed milestone design and audit documents provide rationale and
   context.
4. This ledger summarizes those sources.
5. Chat and Codex reports are working context, not authoritative publication
   evidence.

## Repository workflow

- Integration branch: `modernization`
- Current feature branch: `codex/m6.3c-visco-sh-discrete-adjoint-gradient`
- Status current through locked implementation checkpoint:
  `M6.3c-4 @ 6281219308731bd5e251a3226a372506cd137ba1`
- Current milestone: **M6.3c — exact discrete viscoelastic SH
  adjoint/gradient repair**

## Locked M6.3 checkpoints

| Checkpoint | Published SHA | Role |
| --- | --- | --- |
| M6.3a | `a4d2ca176f518a8414aa95aef256265dd89fa567` | Viscoelastic SH FWI attenuation-inversion audit/design |
| M6.3b | `8c1bfc9c5a5c9f39396b9be5030464f683d3ab5d` | Frozen typed RED attenuation-FWI oracles |
| M6.3c-0 | `b1ec01b4fe14cb7dee7f0b92bbe0592e8b16f07a` | Frozen M6.3c discrete-adjoint acceptance contract |
| M6.3c-0a | `006c7e0df47d0f7fcd04d242a12058cc3622dc7c` | Acceptance-contract nomenclature clarification only |
| M6.3c-1 | `0782cc30e011f62065436435c22a6516c79a9045` | Exact local viscoelastic GSLS VJP |
| M6.3c-1p | `22a2dec5bb415f87f2534fc71b8203885b2939dd` | Anchor M6.3b provenance guard to its historical publication snapshot |
| M6.3c-2 | `cd7ffc82f6e6e87483c868d509d24b734bbd9f9f` | Exact stress-side spatial derivative and CPML transpose |
| M6.3c-3 | `fa58413fdb2ce0f1c5cca7af8fe2e50dc4ee8696` | Exact velocity-side adjoint primitives |
| M6.3c-4 | `6281219308731bd5e251a3226a372506cd137ba1` | Exact MPI-exchange and free-surface adjoint primitives |

M6.3c-2 composes the locked C1 GSLS VJP with the exact staggered FD
transpose and stress-side CPML temporal-state transpose. Its coverage includes
FDORDER 2/4/6/8/10/12, CPML left/right/top/bottom/corner cases, and
FREE_SURF top-CPML selection behavior.

Locked local thresholds:

```text
C2_DOUBLE_DOT_RELATIVE_MAX = 5.0e-12
C2_DOUBLE_REFERENCE_RELATIVE_MAX = 5.0e-12
```

Publication-gate evidence maxima:

| Measurement | Maximum |
| --- | ---: |
| Standalone CPML dot residual | `1.632922442287044e-16` |
| Standalone spatial dot residual | `3.866498983147064e-16` |
| Full stress-side block dot residual | `3.900895271804678e-14` |
| C vs independent-reference relative error | `2.7209690178180662e-16` |

M6.3c-3 provides the locked velocity-side discrete-adjoint primitives for
the velocity-update transpose, velocity-side CPML temporal-state transpose,
receiver-sampling transpose, and source-injection transpose. It does not yet
close or activate the full global production adjoint. M6.3c-4 provides the
exact transposes for MPI velocity and stress exchange and for velocity and
stress free-surface completion. Its verification includes actual multi-rank
MPI dot-product tests and comparisons with an independent reference. Full-state
adjoint integration, material-map/Q-gradient work, and active-path unification
remain later checkpoints.

## Open integration risks / preconditions

The current local adjoint CPML helpers do not treat simultaneous CPML
activation on both sides of the same axis as a normal case and reject such
same-axis overlap configurations. The current production forward code can in
principle execute both corresponding CPML `if` branches sequentially when the
opposing CPML activation regions overlap. The existing geometry check only
verifies that `FW` does not exceed the minimum local domain dimension; it does
not guarantee that opposing CPML activation regions on an axis are disjoint.

Before a final global-exactness claim, one of the following must therefore be
decided and verified explicitly:

1. prohibit such domain-decomposition geometries as a production
   precondition; or
2. implement the exact sequential transpose of the overlapping forward CPML
   operations.

This remains an open integration/precondition issue, not a retrospective
M6.3c-2 publication blocker. It is not repaired in this documentation-only
checkpoint.

## Frozen M6.3 provenance hashes

| Artifact | SHA-256 |
| --- | --- |
| M6.3a audit | `03c757210f0b86db5be82cd0dbe3f6650ce9115c2933926ccda9c5f2f1bca28a` |
| M6.3b validation | `15a8b21077f03e902d2edc735941442b384935431b749540401a0d018e5e0552` |
| M6.3b instrumentation | `84a821686303b9b8166ec884b381348900e7158f074dc57259d12142a0d991cd` |

## Frozen M6.3c acceptance

- Local Python/C operator-transpose closure must be at machine-precision
  scale where applicable.
- The eventual production global float32 adjoint-dot relative residual must
  be at most `1e-5`.
- Directional Q/tau gradient relative disagreement must be at most `5e-3`
  for the DTINV=1 gate.
- Temporal quadrature must contain the independently verified `DT * DTINV`
  factor.
- No exactness claim is made for DTINV>1 until separately demonstrated.
- The viscoelastic base objective and zero-step trial objective must use the
  same physics and agree to relative `1e-12` or better.
- No fitted sign, temporal shift, or empirical scale factor is allowed.

## Known intentional RED/XFAIL evidence

Two M6.3b baseline XFAILs remain intentionally frozen until the corresponding
later production defects are repaired:

1. the production global adjoint-dot defect;
2. the disconnected/incomplete Q/tau gradient baseline.

These are historical RED evidence, not current test-suite regressions. Later
post-repair GREEN tests do not rewrite this frozen baseline.

## Roadmap

### Done / locked

- C0 acceptance contract
- C1 local GSLS VJP
- C2 stress-side spatial derivative and CPML transpose
- C3 velocity-side transpose and receiver-sampling/source-injection transpose
  primitives
- C4 MPI-exchange and free-surface transpose primitives

### Next

- C5 full-state viscoelastic SH adjoint integration, without active FWI-path
  switch

### Planned

- C6 material-map VJPs, including `av_mu^T` / `av_tau^T` and MPI seam/corner
  handling
- C7 production parameter gradients and temporal quadrature
- C8 active-path unification; remove elastic-base versus visco-trial physics
  split

### Follow-up

- M6.3d optimizer/model update, bounds, and output integration

## Update policy

- Update this ledger after an independently verified substantive
  modernization publication lock or a material roadmap decision, not for
  every local/uncommitted iteration.
- A documentation-only commit that updates this ledger or its review policy
  does not itself require another self-referential ledger update.
- Published SHAs must never be silently rewritten.
- If a checkpoint is superseded or corrected, record the new checkpoint
  rather than altering history.
- Detailed test logs belong in verification evidence, not in this concise
  ledger.
