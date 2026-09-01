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
  `M6.3c-C7c-r1 @ fe60e9b858585421f3dbaefca77e53e419b81e20`
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
| M6.3c-5a | `52fcc03c8bbdb2fbae3c40c6b7fc9cf67d2c1e54` | Exact full-state transpose of one fixed-material viscoelastic SH timestep |
| M6.3c-5b | `39be93f1c817ced7489d036f22001cf8437434e3` | Exact reverse-time composition of the fixed-material viscoelastic SH adjoint over the full time axis |
| M6.3c-6a | `e855d1b2feb9dc468ad3af3303727e5a52ce3007` | Exact local SH material-map VJPs and physical parameter-chain verification |
| M6.3c-6b | `8a708de5c9a03a9c3d22bf199d3697f047ca7d5a` | Exact distributed SH material-map transpose across MPI seams and corners |
| M6.3c-7a | `8f711dfbe1bb32af34120a5cb80800082ce76e41` | Exact forward material-observable trajectory from the viscoelastic SH forward path |
| M6.3c-7b | `f128dfa0a563f334116da1c58c746da9eaf2b6fa` | Exact local per-timestep native material sensitivities |
| M6.3c-7c-a | `77a20dd8d81de6f444e3292f438626bc0b2a48a3` | Temporal reduction and exact distributed mapping of prescribed per-timestep sensitivities to owned physical gradients |
| M6.3c-7c-b1 | `f67daaff71f98b1f7ef048821175b56e9ea73ac8` | Exact single-step bridge from real forward observables and aligned adjoint cotangents to native material sensitivities |
| M6.3c-7c-b2 | `dc37e0602c92c3bfc76600f2a32eac691ca69941` | Historical multi-step reverse-time material-gradient assembly under the then-current temporal contract |
| M6.3c-7d-a RED | `dd787f01f3db32bcff4c83ce5328c615fda0b19a` | Real-objective directional-FD falsification exposing an erroneous extra `DT` scaling in the assembled material gradient |
| M6.3c-C7c-r1 | `fe60e9b858585421f3dbaefca77e53e419b81e20` | Correct discrete-objective temporal material-gradient normalization for `DTINV==1` |

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
MPI dot-product tests and comparisons with an independent reference.

M6.3c-5a composes the locked C1--C4 primitives into the exact transpose of one
complete fixed-material viscoelastic SH propagation timestep. The propagated
state comprises `vz`, `sxz`, `syz`, the GSLS memory variables `r` and `q`, and
the stress-side and velocity-side CPML states. In reverse order, the composed
operator covers receiver sampling, stress MPI exchange, free-surface stress
completion, the viscoelastic GSLS/stress update with spatial-derivative and
CPML transposes, free-surface velocity completion, velocity MPI exchange, the
velocity update, and source injection.

C5a is not the active production FWI adjoint. It provides neither a reverse-
time driver over the full time axis nor a switch of `grad_obj_sh` or the
existing FWI path to the new operator. Material gradients and the complete
`mu`/`rho`/`tau`/`Q` chain, optimizer integration, and model update remain
later work.

M6.3c-5b composes the locked C5a single-step operator over multiple timesteps
in reverse temporal order and propagates the complete cotangent state backward
over the full time axis. Receiver cotangents are injected at their associated
timesteps, while source cotangents are returned for every chronological
forward timestep. The initial-state cotangent is placed unambiguously in the
designated output state for both even and odd timestep counts. C5b remains a
fixed-material operator: it accumulates no material gradients and is not yet
connected to the active SH FWI path.

M6.3c-6a closes the local material-parameter transpose for the viscoelastic
SH path. It provides the exact harmonic-average VJP used by `av_mu_SH`, the
`av_tau` transpose, the piecewise `rho -> rhoi` VJP, velocity-update
coefficient sensitivity with respect to `rhoi`, and the exact legacy and
physical-Q `Q -> tau` derivatives. The complete local 2x2 parameter map is
verified by dot-product tests, an independent analytic reference, and finite
differences for the audited `INVMAT1==1` (`Vs`, `rho`, `Q`) and `INVMAT1==3`
(`mu`, `rho`, `Q`) material modes. C6a is local only: it does not transpose
the distributed `matcopy_SH` operation or accumulate production gradients
over time.

M6.3c-6b implements the exact transpose of the production `matcopy_SH`
material exchange. It transposes the actual vertical-then-horizontal cyclic
forward exchange in horizontal-then-vertical reverse order, accumulates
returned cotangents into their source cells, consumes overwritten halo
cotangents, and preserves diagonal-corner provenance. Verification covers
self-neighbour and multi-rank MPI topologies and composes the locked C6a local
VJPs for `INVMAT1==1` and `INVMAT1==3` with legacy and physical Q. All
material channels, including `bar_Q`, are independently checked with
channel-specific normalization. C6b reproduces the existing cyclic
`matcopy_SH` topology exactly; it does not redesign its forward boundary
semantics.

C6 is complete. C6a and C6b together close the spatial material-parameter
transpose from staggered viscoelastic SH coefficient sensitivities to owned
`Vs`/`rho`/`Q` parameters for `INVMAT1==1` and `mu`/`rho`/`Q` parameters for
`INVMAT1==3`, including local nonlinear maps, staggered averaging, MPI seams,
and MPI corners. This is not yet the production FWI gradient because temporal
integration with the forward trajectory remains to be implemented.

M6.3c-7a provides the exact forward material-observable trajectory required
by the later material VJP. It passively captures the corrected stress
divergence `qsum` at the velocity update and the CPML-corrected strains
`strain_x` and `strain_y` at the viscoelastic constitutive update. These are
the three frozen forward-observable sampling contracts in the real
viscoelastic SH forward timestep. C7a neither assembles a material gradient
nor changes the active FWI or gradient paths.

M6.3c-7b combines the locked C7a observables `qsum`, `strain_x`, and
`strain_y` with the time-aligned reverse-time cotangents at the outputs of
the material-dependent velocity and constitutive updates. For one physical
timestep it returns the native sensitivities `bar_rhoi`, `bar_mu_x`,
`bar_mu_y`, `bar_tau_x`, and `bar_tau_y`. The density contribution is exactly
`(DT / DH) * qsum * bar_v_post-velocity`, with no additional `rhoi` factor.
The stress and memory contributions reuse the locked C1 GSLS VJP rather than
deriving a second GSLS adjoint.

C7b performs no multi-step temporal accumulation or `DT * DTINV` gradient
quadrature, no C6 mapping to owned physical `Vs`/`mu`, `rho`, and `Q`
parameters, no model-level tau-to-Q composition, no objective directional-FD
verification, and no integration into the active FWI or line-search path.

M6.3c-7c-a independently validated that its then-assumed temporal reduction
and exact distributed mapping were implemented as specified for prescribed
C7b per-timestep native sensitivities. That historical checkpoint applied
the assumed outer weight `DT * DTINV` before the exact C6b distributed
material transpose and locked C6a native-to-physical mapping. C7d-a later
falsified that temporal assumption against the real discrete objective;
C7c-r1 corrects the verified `DTINV==1` path without rewriting the published
C7c-a commit or its historical evidence.

C7c-b is complete. C7c-b1 bridges the locked real C7a observables `qsum`,
`strain_x`, and `strain_y` with the exactly time-aligned cotangents of the
locked C5 reverse step. Through the locked C7b VJP it produces the five
native sensitivities `g_rhoi`, `g_mu_x`, `g_mu_y`, `g_tau_x`, and `g_tau_y`
for one physical timestep without changing the fixed-material C5 state
transpose.

C7c-b2 composes that bridge over the real reverse-time trajectory. At each
physical reverse timestep it uses the corresponding C7a observable set and
the time-aligned C5 cotangents to produce the five C7b per-step material
VJPs. For the corrected `DTINV==1` contract, C7c-r1 sums those contributions
directly over the discrete timesteps, with no additional temporal `DT`
scaling, and then applies the locked distributed C6 material transpose
exactly once. The result is an owned `Vs`/`rho`/`Q` gradient for
`INVMAT1==1` or an owned `mu`/`rho`/`Q` gradient for `INVMAT1==3`. Q remains
mapped only after `matcopy_SH_adjoint` through the corresponding locked
tau/Q chain rule.

For a supplied receiver-cotangent series and its corresponding real C7a
forward-observable trajectory, the multi-step viscoelastic SH reverse-time
adjoint therefore assembles the directly summed and distributed owned
`Vs`/`mu`, `rho`, and `Q` material gradient for `DTINV==1`, while preserving
the locked fixed-material state transpose. No end-to-end objective-gradient
weighting has been established for `DTINV>1`; neither `DTINV` nor
`DT * DTINV` may be inferred as the correct weight without separate proof.

C7d-a now validates the objective directional derivative for one frozen
`DTINV==1`, `INVMAT1==3`, heterogeneous mu-only configuration. It does not
yet validate Vs, rho, Q, combined directions, the broader MPI/boundary
matrix, or `DTINV>1`. The assembled gradient is not yet connected to the
active SH FWI, line-search, optimizer, model-update, or end-to-end
Q-inversion workflow. C7d-b and C8 have not started.

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
- For the verified `DTINV==1` discrete SH L2 objective, material
  sensitivities are accumulated as the direct sum of the exact discrete
  per-timestep VJPs, with no additional temporal `DT` scaling.
- No exactness claim is made for DTINV>1 until separately demonstrated.
- The viscoelastic base objective and zero-step trial objective must use the
  same physics and agree to relative `1e-12` or better.
- No fitted sign, temporal shift, or empirical scale factor is allowed.

## Resolved falsification evidence

C7d-a at `dd787f01f3db32bcff4c83ce5328c615fda0b19a` froze a real
single-rank, heterogeneous mu-only `INVMAT1==3`, `DTINV==1`, `LNORM==2`
objective-directional-FD experiment. The production contract was
`J = 0.5 * sum_n r[n]^2`, with `r[n] = synthetic[n] - observed[n]` and
receiver cotangent `bar_receiver[n] = r[n]`; neither objective nor receiver
cotangent contained an extra `DT`. Before C7c-r1, the complete predefined
epsilon series reproducibly yielded `D_ad / D_fd ~= DT`, with
`DT = 0.0013`, without a fitted sign, scale, or time shift.

C7c-r1 at `fe60e9b858585421f3dbaefca77e53e419b81e20` removed that
erroneous outer `DT` scaling. It distinguishes the operator-level `dt`
factors already contained in the discrete C7b update Jacobians from the
objective sample weighting: for the verified `DTINV==1` path,
`g_total = sum_n g_step[n]`.

The unchanged C7d-a acceptance gate then produced these relative errors:

| Epsilon | Relative error |
| ---: | ---: |
| `1.0e-2` | `1.2750705797569411e-05` |
| `3.0e-3` | `1.7147288560645724e-04` |
| `1.0e-3` | `2.7921168943080390e-04` |
| `3.0e-4` | `4.5648788994939954e-04` |

The maximum `4.5648788994939954e-04` is below the frozen `5e-3` limit.
Thus, for this verified configuration, the exact assembled mu material
gradient agrees with the independently evaluated central finite-difference
derivative of the real discrete receiver-data objective without fitted sign,
scale, or time shift. C7d-a remains a published falsification checkpoint,
not a permanent XFAIL and not a complete C7d matrix.

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
- C5a exact full-state transpose of one fixed-material viscoelastic SH
  propagation timestep
- C5b full reverse-time fixed-material viscoelastic SH adjoint driver
- C6a exact local material-map VJPs and physical parameter-chain verification
- C6b exact distributed material-map transpose across MPI seams and corners
- C7a exact forward material-observable trajectory
- C7b exact local per-timestep native material sensitivities
- C7c-a temporal reduction and exact distributed mapping of prescribed
  per-timestep sensitivities to owned physical gradients
- C7c-b1 exact single-step bridge from real forward observables and aligned
  adjoint cotangents to native material sensitivities
- C7c-b2 multi-step reverse-time assembly of temporally reduced distributed
  `Vs`/`mu`, `rho`, and `Q` material gradients for `DTINV==1`
- C7d-a real-objective mu directional-FD falsification checkpoint
- C7c-r1 corrected discrete-objective temporal gradient normalization and
  successful rerun of the frozen C7d-a gate

### Next

- **M6.3c-7d-b broadened end-to-end objective directional-FD validation**:
  for `DTINV==1`, compare
  `[J(m + eps*dm) - J(m - eps*dm)] / (2*eps)` with `grad(J)^T dm` for
  separate and combined `mu`, `rho`, and `Q` directions for `INVMAT1==3`
  and `Vs`, `rho`, and `Q` directions for `INVMAT1==1`. Coverage must include
  legacy and physical Q mapping, representative MPI topologies, free surface,
  and representative CPML, while retaining the `5e-3` relative-error target
  and prohibiting fitted sign, scale, or time shift. C7d-b has not started.

### Planned

- C8 active-path unification; remove elastic-base versus visco-trial physics
  split; not started

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
