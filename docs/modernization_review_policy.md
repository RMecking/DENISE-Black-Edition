# DENISE Modernization Review Policy

## Purpose

This document defines stable working and review rules for DENISE
modernization. The rules are independent of any single milestone or
implementation detail.

The root [`AGENTS.md`](../AGENTS.md) defines the repository-wide operating
model, worker roles, decision rights, safety boundaries, and publication
authority. This policy specializes the modernization review/locking process and
does not override `AGENTS.md`, frozen scientific contracts, or explicit task
packets.

## Roles

- `DENISE — NUMERICS` owns normal numerical implementation, solver-local
  production work, performance work, and scoped implementation evidence.
- `DENISE — NUMERICS-SPECIALIST` is an on-demand independent research/advice
  role for a hard, bounded numerical question. It does not become a second
  production owner; accepted ideas return to `NUMERICS` for integration.
- `DENISE — SCIENTIFIC-VERIFICATION` independently reviews the exact candidate
  state and evidence. It reports findings and does not silently repair
  production code.
- `DENISE — TEST-ORACLES` creates or maintains durable independent scientific
  checks when a new behavior needs an independent target or a real oracle/
  coverage gap exists. It is not an automatic stage after ordinary
  optimization work.
- `DENISE — DOCS-AUDIT` maintains durable scientific, provenance, milestone,
  and context documentation when requested.
- `DENISE — REPO-OPS` performs mechanical Git/GitHub publication operations
  only after scientific/content acceptance and an explicit Content Lock /
  publication authorization; it does not make scientific/product decisions or
  changes.

There is no permanent Codex orchestrator. Keep each worker on a stable role and
configuration through a task. Delegate a hard bounded numerical question to
`NUMERICS-SPECIALIST` instead of reconfiguring `NUMERICS` and discarding useful
context.

## Git and GitHub

Git objects and GitHub publication state are the authoritative record of
published work.

## Tests and oracles

Tests and oracles are authoritative evidence only for the specific contract
they define. An oracle must remain independent of the implementation property
it is meant to check.

## Branch safety

- Never modify `modernization` directly for feature work.
- Use dedicated feature branches.
- Never force-push.
- Merge to `modernization` only under explicit publication authority after the
  locked candidate passes its required gates.
- Do not silently rebase or amend already locked commits.
- Preserve exact published SHAs.
- Normal GitHub merge commit is the default; do not rebase, amend, force-push,
  or squash locked/published work.

## Review lifecycle

The default lifecycle is:

1. Define the work package with exact base, scope, exclusions, frozen
   scientific behavior, and current publication authority.
2. `NUMERICS` investigates/implements and produces focused evidence.
3. If a genuinely hard bounded numerical question blocks progress,
   `NUMERICS-SPECIALIST` may investigate it independently; `NUMERICS` integrates
   any accepted approach.
4. Run focused verification and relevant regressions required by the task and
   the affected scientific contract.
5. `SCIENTIFIC-VERIFICATION` independently reviews the exact candidate state,
   scientific equivalence/correctness, scope, and adequacy of evidence. It does
   not repair the candidate during verification.
6. A failed verification returns to implementation/research. Call
   `TEST-ORACLES` only when independent evidence is actually missing, a new
   scientific behavior needs a frozen target, or verification identifies a
   concrete oracle/coverage gap.
7. Apply bounded hardening when needed. If it is test-, harness-, tooling-, or
   CI-only, a narrow independent review is sufficient when Production and the
   previously verified numerical candidate are byte-identical, scientific
   assertions/oracles and tolerances are unchanged, no skip/XFAIL hides a
   failure, and independent narrow delta verification passes.
8. After independent scientific/content acceptance, freeze a **Content Lock**
   for the exact candidate. Record, as applicable, exact BASE SHA, candidate
   identity, authorized paths, expected commit structure/message policy,
   relevant frozen blobs/hashes, required Hosted-CI evidence, merge method, and
   fail-closed conditions. Local test success or worker completion alone does
   not create this lock.
9. An explicit publication authorization may cover one operation or **Full
   Conditional Publication**: deterministic stage/review/commit/verification,
   push/remote verification, PR creation/update and verification, required
   Hosted CI, conditional normal merge, and post-merge identity verification.
   These remain distinct Git/GitHub operations but require no separate
   conversational approvals while all frozen conditions hold. `REPO-OPS` is
   mechanical only and cannot change the locked content or repair scientific
   failures.
10. Obtain the default single risk-appropriate Hosted verification on the final
    locked PR candidate before merge. A real content failure reopens its
    relevant scientific/content gate; an infrastructure-only failure may have
    a bounded retry or separately authorized infrastructure-only hardening if
    the frozen candidate is unchanged and the cause is demonstrated.
11. If all conditions remain satisfied, merge using the normal GitHub merge
    commit method, then verify PR state, merge SHA and ordered parents, tested
    tree identity, live `modernization`, exact changed paths, and applicable CI
    evidence. Only then declare the checkpoint **CLOSED/CANONICAL**.

Do not routinely spend Hosted-CI runs after every implementation/hardening
iteration, merely because a branch was pushed, repeatedly on identical locked
content, on extra platforms without material platform-specific risk, or
post-merge when the normal merge tree is exactly the already-tested tree.
Additional runs require a concrete reason (for example a Hosted-only failure,
compiler/toolchain portability, workflow-specific or materially
platform-specific behavior, or release-critical validation). Test-suite
partitioning may remove duplicate execution only when exact collection-set
equivalence proves coverage is preserved. Never remove required node IDs, hide
empty-selection failures with `|| true`, weaken markers/assertions, or change
scientific acceptance semantics to save runtime.

On any unexpected base move, candidate/scope/commit/remote mismatch,
conflict/non-mergeability, real Hosted-CI failure, numerical/scientific
discrepancy, oracle/tolerance/acceptance mismatch, tested-candidate/merge-tree
mismatch, a final PR head different from the tested PR head, synthetic
parent/tree mismatch, or post-merge identity/scope mismatch, `REPO-OPS` stops
and reports the concrete HOLD. It does not rebase, amend, force-push, weaken
tests, or autonomously repair scientific/product content.
ChatGPT/Product Owner re-enters before closure only for a defined HOLD, genuinely
new scientific/product evidence, or a required change to locked scope. A
dependent milestone does not start before the current milestone closes.

## Uncommitted review artifacts

- Uncommitted production changes must be reviewable as exact current diffs or
  complete files tied to the verified base; after commit, exact Git object/blob
  identity should be checked against the accepted candidate.
- Transient patch files, raw pytest output, and worker reports normally are not
  committed merely as transport artifacts.
- After publication, Git is the canonical patch history.

## Permanent verification artifacts

Commit permanent test material when it forms part of the product or
verification contract. This may include regression tests, mathematical
reference implementations, frozen validation JSON, acceptance contracts, and
intentional provenance/instrumentation artifacts. An important oracle must
not exist solely in chat or as an ephemeral upload.

## Frozen artifacts

- Frozen baseline artifacts are immutable by default.
- A necessary lifecycle or provenance repair is a separate, narrowly scoped
  maintenance checkpoint.
- Never make an old RED oracle green by silently rewriting its evidence.
- Distinguish historical RED baselines from post-repair GREEN tests.

## Historical code

Historical DENISE and DENISE-SH may be used for genealogy, intent discovery,
and architecture comparison. They are not automatically numerical oracles.
The current Black Edition discrete operator together with explicitly frozen
contracts defines adjoint correctness.

## No fitting

Discrete-adjoint and gradient work prohibits:

- fitted signs;
- fitted time shifts;
- empirical gradient scaling;
- replacing the exact transpose with a merely similar continuous-adjoint
  expression;
- assuming halo, free-surface, or stateful CPML maps are self-adjoint without
  proof.

## Scope discipline

- Each checkpoint has explicit in-scope and out-of-scope work.
- Defer unrelated cleanup and refactoring.
- Report blockers rather than silently broadening scope.
- A mathematically distinct repair normally receives its own checkpoint.

## Provenance guards

Historical provenance assertions must be anchored to the historical
publication SHA they describe. Do not require a production diff from an old
audit to an ever-moving current HEAD to remain empty when later milestones
are expected to add production code.

## Status ledger

[`docs/modernization_status.md`](modernization_status.md) is updated after
substantive modernization publication locks or material roadmap decisions. A
documentation-only status/policy maintenance commit does not itself require
another ledger update. The ledger is the first repository document to consult
for concise current modernization state. It summarizes, but does not override,
Git history or frozen evidence.
