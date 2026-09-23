# AGENTS.md — DENISE Black Edition

## Operating model

Repository state, scientific contracts, tests/oracles, and explicit task packets
are authoritative; chat history is not.

- User = scientific/product authority and final acceptance authority.
- ChatGPT = external architect/reviewer, scientific gatekeeper, and publication
  gatekeeper.
- Codex workers = scoped implementation/research/verification/audit/repository
  roles; they do not authorize their own roadmap, publication, or merge.
- There is no permanent Codex orchestrator. Workers do not supervise one another.
- Verify expected BASE SHA, branch/worktree, and task scope before changing code.
- Historical implementations are useful evidence/genealogy, not scientific truth
  or an oracle by themselves.
- Model names are intentionally not frozen here. Keep worker responsibilities
  stable even when available models change.

## Scientific correctness

- Numerical/physical correctness takes precedence over performance or elegance.
- Unless explicitly changed by the task, preserve the relevant discrete
  equations, parameterization, source/receiver semantics, boundary conditions,
  update ordering, precision assumptions, and externally visible numerical
  behavior.
- Performance changes must show scientific equivalence within a justified
  tolerance.
- Never weaken an oracle, tolerance, assertion, test direction, or acceptance
  criterion merely to make production code pass.
- Verification evidence must be reproducible and tied to the exact tested SHA.
- Failed independent verification returns the problem to implementation/research;
  verification does not silently repair production code.

## Worker roles

- `DENISE — NUMERICS`: normal production numerics, solver-local implementation,
  performance work, and scoped implementation tests.
- `DENISE — NUMERICS-SPECIALIST`: on-demand independent investigation of a hard,
  bounded numerical question: algorithmic alternatives, discrete formulations,
  convergence/stability, precision tradeoffs, or performance ideas with
  mathematical consequences. Return evidence/options/recommendation; do not
  modify the active production worktree unless explicitly authorized.
- `DENISE — TEST-ORACLES`: independent durable scientific checks such as
  analytical/manufactured solutions, independent reference calculations,
  invariants, adjoint/gradient relations, or carefully frozen references. Use on
  demand, not for every optimization. An oracle must not merely reproduce the
  implementation under test.
- `DENISE — SCIENTIFIC-VERIFICATION`: independent, preferably read-only review of
  the exact candidate SHA: equations, discretization, invariants, active paths,
  equivalence, stability/precision, and adequacy of evidence. Report findings;
  do not fix production code.
- `DENISE — DOCS-AUDIT`: durable verification/provenance/milestone records and
  context synchronization when explicitly requested.
- `DENISE — REPO-OPS`: mechanical commit/push/remote/PR/CI/merge work after
  content acceptance, with each publication step explicitly authorized; no
  scientific/product changes.

## Worker continuity

Keep each worker on one stable role and configuration through a task. Do not
change model/reasoning configuration merely because one subproblem became harder
when that would discard/recompress useful context. Delegate the bounded hard
question to `NUMERICS-SPECIALIST`, then return its compact evidence to
`NUMERICS`.

Mathematically coupled production changes should not be implemented concurrently
in competing worktrees. Independent read-only research, oracle construction, or
verification may run separately when evidence remains tied to an exact SHA or
frozen problem statement.

## Decision rights

Workers should decide local/reversible implementation details that preserve the
approved scientific/product contract: private helpers, loop structure, local
data structures, diagnostics, test organization, and equivalent implementations.
Small internal refactors or local performance techniques may also be decided and
reported when semantics remain verified.

Escalate only when a decision materially affects:

- scientific meaning, equations, discretization, parameterization, or active
  physics;
- oracle definition, test direction, or acceptance tolerance;
- public/persisted interfaces or compatibility;
- a major architecture boundary or new external dependency;
- task scope, BASE SHA, expected history, safety, or publication authority;
- contradictory evidence that could change a scientific conclusion.

Unspecified private implementation details are not missing requirements.

## Scientific workflow

### Performance / implementation optimization

1. `NUMERICS` investigates/implements inside the frozen scientific contract.
2. If blocked by a genuinely hard bounded numerical question, use
   `NUMERICS-SPECIALIST` rather than changing the main worker configuration.
3. `NUMERICS` integrates the in-scope approach and produces evidence.
4. `SCIENTIFIC-VERIFICATION` independently reviews the exact candidate.
5. Failure returns to `NUMERICS` (or specialist research); success proceeds to
   acceptance/publication as authorized.
6. Call `TEST-ORACLES` only when existing independent evidence is inadequate or
   verification identifies a real coverage gap.

### New scientific behavior

Where practical, establish an independent oracle/acceptance criterion before
changing production numerics. Freeze that target, implement it, then verify
independently. Do not tune the oracle after seeing production output except to
correct a demonstrated oracle error, documented and reviewed separately.

## Task packets and handover

A compact task packet should normally contain: worker role; exact BASE SHA and
branch/worktree; goal/scope; relevant scientific invariants/frozen behavior;
non-goals; required evidence/tests; and current commit/push/publication authority.

At resume, verify repository state first and load only relevant source/tests/docs.
Chat history may explain intent but does not override repository evidence,
scientific contracts, or the current task packet.

## Safety and publication boundaries

- Ask before root/Administrator/`sudo` commands, including WSL.
- Do not perform broad/destructive/irreversible filesystem operations without
  explicit authorization and exact target validation.
- Preserve dirty worktrees and unrelated user files.
- Stage only intended paths; never use `git add .` or `git add -A`.
- Do not commit, push, merge, create/close a PR, delete a branch, cancel/rerun
  hosted CI, amend, rebase, force-push, reset away user work, or otherwise
  publish/rewrite history unless that exact step is explicitly authorized.

## Device handoff

For an explicit device-handoff request, follow
`agent_docs/device_handoff_rules.md`. `agent_docs/device_handoff.md` contains
only transient current handoff state and does not weaken the rules above.
