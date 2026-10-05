# Marmousi-II publication reproducibility hardening

MARMousi-II PUBLICATION REPRODUCIBILITY HARDENING — READY FOR PUBLICATION GATE

This is a source-candidate handoff, not publication or new scientific acceptance.
It resolves exactly B1/B2 in the retained [previous assessment](PUBLICATION_GATE.md).
The accepted physical-water campaign and all historical execution/HOLD/closure
evidence remain immutable. No numerical campaign is authorized by this report.

## 1. Repository, checkout and HEAD

Canonical candidate: `DENISE-Benchmark/app-marmousi2-fluid2`, detached HEAD
`f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d` (the authorized execution core).
Locally observed `origin/modernization` at this audit:
`7de3198e5379b6e9e5c6fa6d2bc8fa07afab951b`. This is observation only.
No remote contact or ref write occurred. The historical A0/A1 checkout remains
separate and unchanged; no M9e-5 candidate file was accessed.

## 2. Exact source/documentation changes in this task

Modified relative to the previous local publication candidate:

- `tools/denise_case/fluid2.py`: only the frozen-core gate and its import.
- `applications/README.md`: implemented gate/aggregation/report usage.
- `applications/PUBLICATION_GATE.md`: historical-assessment pointer only;
  its original BLOCKED assessment and audit receipts are preserved.

Added:

- `tools/denise_case/campaign.py`: reusable solver-free post-processing API/CLI.
- `tests/applications/test_frozen_core.py`: ten gate tests.
- `tests/applications/test_campaign.py`: 23 post-processing tests.
- `applications/HARDENING_REPORT.md`: this handoff.

No other previous publication source/config/test file was intentionally edited.
The pre-existing root `.gitignore` modification is retained unchanged.
Ignored task helpers and receipts are separately inventoried under
`publication_candidate/runs/hardening/`; disposable verification products are
under `.tmp-app-fluid2-hardening-*`. These are not publication sources or accepted
scientific products. The exact per-file publication/exclusion inventories are
linked in section 13; no staging has occurred.

## 3. Previous require_core semantics

The old gate required HEAD and local/live `origin/modernization` to equal the
frozen core, including a remote lookup for callers requesting `remote=True`.
An integration-branch advance could therefore block a valid frozen execution
checkout. The original source identity is retained in the new protection receipt,
not substituted into historical execution provenance.

## 4. Corrected frozen-core semantics and caller audit

`require_core(repo, authorized_sha=CORE_SHA, remote=False)` requires an explicit
full SHA, exact local HEAD equality, no tracked or visible untracked changes in
`src`, `include`, `tests/physics`, `tests/utilities`, and an empty index. Wrong
HEAD/authorization or protected source edits fail. Local origin is optional
`observed_repository_state`, never a gate; `remote=True` is compatibility-only
and never initiates network access. All verify/build/run/plan/batch callers were
audited. Separate binary/build/source/input and resume checks remain intact;
run provenance still records the explicitly frozen A2 `core_sha`.

## 5. Moving-origin/offline proof

Tests pass for equal, advanced and unavailable local origin, and reject wrong
HEAD, wrong explicit SHA, non-full-SHA authorization, tracked/untracked core
changes, oracle changes and nonempty index. A strict allowed-command mock rejects
any network/ref-mutation call. Actual frozen-core validation also passes with
the advanced locally observed origin recorded above; no fetch is performed.

## 6. Aggregation API/CLI and deterministic contract

API: `tools.denise_case.campaign.aggregate_campaign`.
CLI: `python -m tools.denise_case.campaign aggregate` with explicit
`--runs-root`, `--output`, `--shots`, `--shape`, optional `--campaign-receipt`
and `--vs-model`; full implemented examples are in the [guide](README.md).

Completed singleton CPU metadata identifies physical shots, not glob order.
Duplicates/missing IDs, incomplete products, inconsistent frozen core/binary,
provenance/raw hash changes, wrong FP64 size/layout/shape and nonfinite images
fail closed. Lambda and mu are separate ascending physical-shot FP64 unweighted
sums. No average, normalization, smoothing, mute, illumination weighting,
preconditioning or parameter transform. Raw/derived isolation and exclusive
fresh outputs prevent overwrite. JSON receipts record input hashes, order,
output hashes, configuration, and optional source-mask/campaign receipt hashes.

## 7. Reporting API/CLI

API: `tools.denise_case.campaign.report_campaign`.
CLI: `python -m tools.denise_case.campaign report`, additionally requiring
`--aggregate-receipt` and accepting repeated `--qc-record` arguments. It validates
campaign metadata/raw identities and aggregate order/operation/configuration/
input/output identities, including the accepted historical aggregate receipt.
It produces deterministic JSON/HTML with source-Vs-recomputed QC or explicitly
labeled retained hash-bound QC. Execution, aggregation and reporting are separate;
aggregation/reporting never launch DENISE.

## 8. Portability and scientific reporting

Products resolve by paths relative to caller-selected run/aggregate directories,
not absolute historical metadata paths. Generated receipts/reports contain
relative product paths and content hashes, not the workstation layout. Synthetic
relocation and actual report scans pass. The accepted manifest's historical FD4
audit/data-bundle dependency remains explicit, unchanged and separately acquired;
this hardening does not redistribute models/data or rewrite that manifest.

Reports retain physical-water restricted parameter-space interpretation, exact
raw water `gMu=+0` without negative-zero bits or post-masking, shot 68 as an
observation rather than rejection, and the strong first-solid-row contribution.
The 15 Hz low-pass exceeds the frozen Taylor-FD4 advisory (~5.50625 Hz); no final
broadband migration accuracy, CUDA fluid support or FLUID-3/4 acceptance is claimed.
Descriptive norm values are not converted into new acceptance thresholds.

## 9. Accepted 100-shot reconciliation

PASS: exactly physical IDs 1 through 100, all retained metadata/provenance and
raw lambda/mu identities, and ascending order agree with the accepted campaign.
Reproduction wrote only a fresh temporary aggregate. Reporting consumed the
accepted aggregate receipt unchanged; two fresh reports were byte-identical and
free of local absolute paths. All subprocesses were forbidden during this
post-processing audit. [Reconciliation receipt](publication_candidate/runs/hardening/reconciliation.json).

## 10. Exact aggregate hash comparison

Both reproduced and accepted values are identical:

| Raw output | SHA-256 |
|---|---|
| Lambda | `8e979872d9ec18f0fe7df4f9088a5722d707a32e6bd1c7af54153821dea8448a` |
| Mu | `f311970f856ff1a3239f056a74d4984b05249efab01e4a2f7ec4b84b3fbbee46` |

No accepted aggregate was overwritten or modified.

## 11. Application/config verification

Complete suite: 69/69 PASS (36 retained tests unchanged, 33 new tests), with a
Git-only subprocess guard; solver/build subprocesses are forbidden. CLI template
generation was performed twice in isolated fixtures and was byte-identical with
equal canonical configuration hashes. No existing assertion/tolerance/oracle
was changed. Initial sandbox ACL failures were infrastructure failures; a new
two-water-row synthetic fixture corrected an insufficient interior-water mask
without weakening QC. The final guarded suite passes in 33.22 seconds.
[Tests](publication_candidate/runs/hardening/application_tests.json),
[manifest/config check](publication_candidate/runs/hardening/manifest_generation.json).

## 12. Diff and index verification

`git diff --check`: PASS; staged diff check: PASS, index empty.
Every intended untracked publication file also passes a no-index whitespace
check. Production/core/oracle tracked diff against the frozen core is empty.
No stage, commit, push or PR; no fetch, reset, rebase or ref rewrite.

## 13. Regenerated publication inventory

The [new exact allowlist](publication_candidate/runs/hardening/publication_inventory.json)
contains 39 reviewed candidate files, including the new API, tests and this report;
all are Git-visible. It records relative paths, added/modified status, role,
source/test/docs/config designation, byte count and SHA-256. The prior 35-file
inventory remains historical and unchanged. The model_io visibility exception
and model-data ignore behavior pass. No unexpected visible file, core/oracle or
M9e-5 file enters the boundary. The
[exclusion inventory](publication_candidate/runs/hardening/excluded_inventory.json)
retains the exact ignored product/evidence/cache/helper paths separately.
[Audit checks](publication_candidate/runs/hardening/review.json).
Independent publication-gate acceptance of this exact candidate is still required.

## 14. Immutability and safety accounting

Before/after identity verification preserves all 4060 protected identities other
than the single explicitly authorized application source edit (`fluid2.py`).
This includes all accepted scientific products, executable, 535 core/oracle
sources, original unchanged application files, model/data/input and historical
HOLD/closure records. The original executed fluid2 source hash is NOT rewritten
in any old receipt. [Protection](publication_candidate/runs/hardening/after.json).

- 0 solver executions; 0 binary rebuilds.
- 0 accepted raw-image or aggregate modifications.
- 0 per-shot execution metadata modifications; 0 core/oracle modifications.
- 0 ref rewrites/fetch/reset/rebase; M9e-5 untouched.
- 0 stage/commit/push/PR.

Temporary aggregate reproduction is post-processing verification, not a shot run
or modification of the accepted evidence. No accepted evidence was deleted.

## 15. Remaining blockers and authority

No remaining blocker within the two explicitly authorized hardening items.
Scientific caveats, external data/audit dependencies, source-review and final
publication authority remain unchanged. READY FOR PUBLICATION GATE does not
authorize publication, new numerical execution or scientific-scope expansion.
