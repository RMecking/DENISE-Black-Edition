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
- Governance must survive individual Codex sessions and ChatGPT chats. At start
  or resume, obtain operating policy from the current repository state, this
  `AGENTS.md`, applicable canonical project/review documents, frozen
  task/scientific contracts, and the explicit current task packet. Chat history
  is supplementary context, not governance authority.

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
  scientific/content acceptance and explicit Content Lock / publication
  authorization; no scientific or product decisions or changes.

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
   scientific/content acceptance and a candidate-specific Content Lock before
   publication authorization.
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

## Content lock and conditional publication

Scientific/content acceptance authorizes only the exact candidate that was
reviewed. It does not authorize later, arbitrary modifications. Before
publication, an explicit Content Lock / publication packet records, as
applicable: exact BASE SHA; candidate/content identity; authorized paths;
expected commit structure and message policy; relevant frozen blobs/hashes;
required Hosted-CI evidence; merge method; and fail-closed conditions. Passing
local tests or worker completion alone does not create a Content Lock or grant
publication authority.

Publication authority may cover one named operation or, under **Full Conditional
Publication**, the complete declared deterministic chain:

`stage -> verify staged content -> commit -> verify commit -> push -> verify remote -> create/update PR -> verify PR -> required Hosted CI -> conditional normal GitHub merge -> post-merge verification`

The operations remain distinct Git/GitHub actions, but do not require separate
conversational approval round-trips while the frozen conditions remain true.
`REPO-OPS` executes mechanics only; it does not decide scientific acceptance,
change product content, or repair a scientific/content failure. Normal GitHub
merge commit is the default. Full Conditional Publication never permits force
push, hidden history rewriting, unauthorized scope, rebase/amend of locked
work, or a different merge method unless that method was explicitly frozen.

`REPO-OPS` fails closed and stops on an unexpected BASE/`modernization` move,
candidate identity mismatch, unauthorized path/diff, unexpected commit, remote
branch change, conflict/non-mergeability, real Hosted-CI failure, scientific or
numerical discrepancy, oracle/tolerance/acceptance mismatch, tested candidate
or merge-tree mismatch, a final PR head different from the tested PR head,
synthetic merge/tree/parent mismatch, or post-merge tree/parent/scope mismatch.
On HOLD it does not rebase, amend, force-push, weaken tests, or autonomously
change scientific/product content; it reports the concrete blocker.

## Test hardening and Hosted-CI economy

GitHub-hosted CI is a scarce publication resource even if a runner currently
appears unmetered. Prefer focused local tests, worker self-verification,
independent local scientific verification, local portability reproduction, and
targeted external harnesses during development. Default to one risk-appropriate
Hosted verification on the final locked PR candidate before merge. Do not run
Hosted CI after every iteration or hardening edit, merely because a branch was
pushed, repeatedly for identical locked content, on extra platforms without a
material platform-specific risk, or post-merge when the normal merge tree is
exactly the already-tested tree. Additional Hosted runs require a concrete
reason such as closing a Hosted-only failure, compiler/toolchain portability,
workflow-specific or materially platform-specific behavior, or release-critical
validation.

An infrastructure-only failure (for example runner cancellation, transient
service failure, external timeout, or download/package infrastructure failure)
may receive a bounded retry or separately authorized infrastructure-only
hardening without reopening scientific acceptance, provided candidate identity
remains frozen and the infrastructure cause is demonstrated. A real numerical,
scientific, assertion, oracle, build-content, or unexplained behavioral failure
reopens the relevant content/hardening gate; `REPO-OPS` does not repair it.

A bounded test-, harness-, tooling-, or CI-only hardening delta need not repeat
full scientific verification when Production is byte-identical, the verified
numerical candidate and scientific assertions/oracles are unchanged, tolerances
are not widened, no skip/XFAIL hides a failure, and independent narrow delta
verification passes. The existing Content Lock may then be extended to include
that verified delta. Test-suite partitioning should avoid duplicate execution
when exact collection-set equivalence proves coverage is preserved; CI savings
must not remove required node IDs, hide empty-selection failures (including with
`|| true`), weaken markers/assertions, or alter scientific acceptance semantics.

## Safety and publication boundaries

- Ask before root/Administrator/`sudo` commands, including WSL.
- Do not perform broad/destructive/irreversible filesystem operations without
  explicit authorization and exact target validation.
- Preserve dirty worktrees and unrelated user files.
- Stage only intended paths; never use `git add .` or `git add -A`.
- No publication or history-changing operation is allowed unless covered by
  explicit publication authority. Authority may be narrow single-step authority
  or Full Conditional Publication covering the complete declared mechanical
  chain. It never authorizes force-push, hidden history rewriting, unauthorized
  scope, or autonomous scientific repairs. Do not amend/rebase locked or
  published work, reset away user work, or alter frozen candidate content.
- Normal GitHub merge commit is the default. A dependent milestone does not
  start until the current milestone is closed.

## Device handoff

For an explicit device-handoff request, follow
`agent_docs/device_handoff_rules.md`. `agent_docs/device_handoff.md` contains
only transient current handoff state and does not weaken the rules above.
