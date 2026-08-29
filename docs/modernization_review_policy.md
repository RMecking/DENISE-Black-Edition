# DENISE Modernization Review Policy

## Purpose

This document defines stable working and review rules for DENISE
modernization. The rules are independent of any single milestone or
implementation detail.

## Roles

### Codex

- Implements narrowly scoped work packages.
- Executes required focused and regression verification.
- Reports exact state, diffs, and results.
- Stops at review gates.

### Independent reviewer

- Defines or reviews checkpoint scope and acceptance.
- Inspects mathematical and implementation evidence.
- Checks scope creep.
- Grants explicit commit and publication gates.
- Independently verifies published GitHub state.

### Git and GitHub

Git objects and GitHub publication state are the authoritative record of
published work.

### Tests and oracles

Tests and oracles are authoritative evidence only for the specific contract
they define.

## Branch safety

- Never modify `modernization` directly for feature work.
- Use dedicated feature branches.
- Do not force-push unless explicitly authorized for an exceptional reason.
- Do not merge to `modernization` without explicit approval.
- Do not silently rebase or amend already locked commits.
- Preserve exact published SHAs.

## Review gates

The normal sequence is:

1. Define the work package with exact scope and exclusions.
2. Codex implements locally.
3. Run focused verification and relevant regressions.
4. Before commit, review production changes from the actual current patch or
   complete files.
5. The reviewer grants explicit commit/push approval.
6. Codex commits only approved paths and pushes normally.
7. The reviewer independently checks GitHub for the SHA, parent, message,
   exact file list, statistics, relevant file modes, and remote branch head.
8. Only then is the checkpoint **LOCKED**.

Local test success alone does not lock a checkpoint.

## Uncommitted review artifacts

- Uncommitted production changes must be reviewable as exact current diffs or
  complete files.
- Transient patch files, raw pytest output, and Codex reports normally are not
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
