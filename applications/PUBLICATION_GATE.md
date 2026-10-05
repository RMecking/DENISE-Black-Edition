# Marmousi-II A0/A1/A2 publication-candidate assessment

Historical publication-preparation assessment: the BLOCKED result and original
35-file audit below are retained. B1/B2 are addressed by the separately audited
[reproducibility hardening](HARDENING_REPORT.md), which supplies the current
39-file candidate inventory. Do not mistake this historical assessment for
the new verdict or rewrite its original local audit receipts.

Verdict: **BLOCKED** (reproduction/packaging, not rejection of the accepted A2 campaign).
No staging, commit, push or PR is authorized or performed.

## 1. Checkout and frozen scientific identity

Candidate: detached `app-marmousi2-fluid2` execution checkout, HEAD
`f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d`. The historical
`app-marmousi2-a01` remains on `codex/app-marmousi2-a01` at
`f315d157c2e817480990c97a3de45c4a23e465db`; it was not edited.
Accepted A2 execution core is exactly the first SHA. Observed integration-branch
state is recorded separately in the local audit receipt, never as a completed
benchmark acceptance condition. No refs were changed or fetched.

The accepted CPU binary SHA-256 remains
`9292c21197a59ae4bf842d18c1c1608517ea8409f5fa1860b8f45715fdd5ac73`.
Application Review has accepted the 100-shot campaign under the separate frozen
closure. The earlier unblock/HOLD reports remain historical evidence, not the
current status. Neither publication nor final broadband-accuracy acceptance
follows automatically from Application Review.

## 2. Intended publication files and exact exclusions

The deterministic [publication inventory](publication_candidate/runs/publication_inventory.json)
is the proposed commit allowlist: repository-relative path, added/modified
status against HEAD, eight-category role, source/test/docs/config designation,
byte count and SHA-256. No numerical production/core oracle or M9e-5 file is
included. The inventory is local audit evidence, not part of its own commit.

Proposed boundary: root `.gitignore`, `requirements-applications.txt`,
18 reusable `tools/denise_case/*.py` modules, three unchanged application-test
files, the three cases' `.gitignore`/manifest/README files, consolidated
`applications/README.md`, this assessment and the publication-audit ignore file.
Exactly 35 files are proposed; only `.gitignore` is a tracked modification.

The separate [exact exclusion inventory](publication_candidate/runs/excluded_inventory.json)
lists ignored evidence/cache/build paths and immutable historical originals.
Two visible untracked files are explicitly NOT intended for publication:

- `applications/marmousi2_fluid2/A2_REPORT.md`: generated two-shot unblock report,
  historically correct, but superseded as current status and contains local paths.
- `tools/denise_case/fluid2_report.py`: workstation-specific old gate/report helper,
  not a reusable current 100-shot campaign reporter.

Also exclude all raw lambda/mu images, SU data, copied benchmark models/geometry,
prepared inputs, execution metadata/provenance, batch and aggregate receipts,
HOLD/closure evidence, generated PNG/embedded HTML, local campaign/audit recipes,
test caches/temporary fixtures and compiled binaries/libraries/objects.
Exclusion from Git does NOT authorize deleting accepted evidence.

## 3. Layout and changes made by this packaging task

Keep `marmousi2`, `marmousi2_fd4`, `marmousi2_fluid2`, reusable
`tools/denise_case` and `tests/applications` separate. Their scientific roles
are different; renaming/consolidation would obscure frozen path provenance.
The consolidated guide provides the current names and historical aliases.

Two historical manifests and their per-case ignore policies were copied
byte-identically into the candidate; the original checkout was not modified.
Added a consolidated guide, three short case READMEs, this assessment and
`publication_candidate/.gitignore`. Local audit helpers/receipts are ignored.
No existing application implementation, tests, configurations, accepted reports
or scientific evidence were edited. The root ignore modification predates this
task and remains unchanged.

## 4. Ignore policy and source visibility

The existing root rule `!tools/denise_case/model_io.py` remains necessary and
resolved; the blanket `model_*` dataset rule still works. All 35 proposed source,
test, docs and configuration paths must be visible to Git. Source visibility
and positive dataset-ignore checks are recorded in the audit.

Each case ignores generated inputs/data/runs/QC; publication audit receipts
are separately ignored. Only the root `.gitignore` appears in the tracked diff,
with two pre-existing additions: the model_io exception and `.tmp-app-fluid2-*/`.
Do not use blanket `git add .`, `git add -A`, or force-add evidence; any eventual
staging must use the independently accepted exact allowlist.

## 5. Provenance and stale-reference findings

The completed-benchmark policy is documented in the guide: freeze execution
core, binary/build, scientific inputs and result/metadata identities. Later
`origin/modernization` movement is observation only. Before NEW numerical work,
the explicitly authorized execution SHA and binary/input contract must still
be verified; no pre-execution identity check was bypassed by this task.

Finding B1: `tools/denise_case/fluid2.py:60` still requires local origin equality;
the following remote check also requires live origin equality. The accepted
closure does not invoke those legacy moving-origin completion checks, but the
new-run CLI remains unusable at the advanced origin state. Its exact HEAD
check also means a future application publication commit is not automatically
an authorized numerical execution checkout. This needs a separately reviewed
application identity-boundary correction, not a ref reset or campaign rerun.

Finding B2: full-campaign aggregate/QC/report orchestration remains in ignored
`runs/campaign_100` helpers. Some depend on the excluded historical report helper.
The reusable public CLI provides deterministic individual/batch execution and
single-shot QC, but not a complete standalone 100-shot aggregate/report workflow.
Those essential reproducibility sources must be deliberately promoted/reviewed
before a source-only publication can claim full campaign reproduction.

The unchanged CPU manifest also requires the named historical sibling checkout,
its accepted FD4 direct data and gate/preflight/baseline receipts. Copied manifests
alone cannot replace that audit/data bundle. This dependency is documented,
not silently rewritten. Windows/WSL absolute paths and a workstation username
remain only in excluded historical generated evidence and the old report helper;
they are not sanitized out of correct execution provenance. The optional
historical CUDA helper fixes `/usr/local/cuda/bin/nvcc` as
the recorded historical toolkit location and `CUDA_ARCHS=86`; this helper also
probes the fixed toolkit path. It is a host-specific legacy build recipe, not a
portable toolchain selector or an assertion of CUDA fluid capability.

Raw/display isolation, the 15 Hz versus Taylor-FD4 advisory, no final broadband
accuracy claim, no CUDA fluid claim, no FLUID-3/4 acceptance, shot 68's retained
norm observation and the preserved first-solid-row contribution remain explicit.

## 6. Verification and complete diff evidence

Full unchanged application suite: **36/36 PASS**. All solver/build subprocesses
were forbidden by a Git-only subprocess guard. Existing deterministic
configuration/MODE=2 generation, overwrite protection, shot isolation/resume
and source-visibility tests were included. Additional CLI template generation
is performed twice in disposable fixtures and required to be byte-identical.
No assertions, tolerances or numerical oracle semantics were changed.

The initial frozen protection check passed for 4061 identities, including the
accepted closure, 535 core/oracle identities, all 27 original candidate identities
and the immutable scientific products. Post-test protection and final whitespace/
visibility checks are recorded in [verification](publication_candidate/runs/verification.json)
and [repository review](publication_candidate/runs/review.json). The complete
application test output is [here](publication_candidate/runs/application_tests.json).

These receipts provide modified tracked files, untracked intended files, exact
excluded paths, any unexpected files and line-specific stale-reference findings.
`git diff --check`, staged `git diff --check`, empty index/core/oracle diffs and
untracked intended-file whitespace checks must all pass. No scientific product
is regenerated; synthetic test fixtures are disposable and isolated.

Solver executions **0**; binary rebuilds **0**; observed-data/migration-product
regenerations **0**; historical execution metadata modifications **0**;
core production/oracle modifications **0**; M9e-5 candidate files accessed **NO**;
stage/commit/push/PR **0**. Worktree inventory names do not constitute reading
or modifying unrelated candidate files.

## 7. Proposed commit and remaining blockers

Only after B1/B2 are resolved and the independent Publication Gate accepts the
revised exact inventory, propose one application-only source/docs/config/test/
ignore commit. Keep scientific evidence out of that commit and separately
retained with its immutable receipts; no historical worktree merge or core
change belongs in this boundary.

Proposed message:
`docs(applications): package frozen Marmousi-II A0/A1/A2 workflows`

Remaining blockers are B1/B2 above. Resolving them must preserve the accepted
products and pre-execution frozen identity guarantees; this audit does not
authorize new numerics, weakening scientific checks or publishing anything.

BLOCKED
