# DENISE practical FWI/RTM research priorities

> **Repository note (2026-11-21).** Imported unchanged in substance from the
> private working copy as the Owner's standing research direction. The
> dated status table below reflects 2026-10-25; current planning is in the
> [integration roadmap](roadmap.md). The research journal referred to below
> is kept privately and is not part of this repository.

**RESEARCH-PRIORITIES-01 — Owner direction, 2026-10-25.**
DENISE Black Edition is an experimental scientific development project, not
currently a software product being driven to product maturity. The top goal is
**functioning, scientifically traceable FWI/RTM on increasingly realistic seismic
problems**. Practical applications and mathematical/numerical development
inform one another; tests and reviews evaluate claims and investigate real
problems, not impose an administrative sequence on all exploration.

## P0 — Practical application capability

- Actual multistep FWI with model updates, data fit and reconstruction comparison.
- Actual RTM with reflected data, known structures, migration images and
  physically meaningful QC metrics.
- Begin with controlled synthetic models; progressively extend to heterogeneous
  and geologically more realistic models. Record acquisition, model/parameter,
  boundary and precision limits rather than generalizing a small success.

## P1 — Scientific development

- New FD/adjoint/FWI/RTM methods and numerical improvements.
- Algorithmic investigation of problems found in applications.
- Try new methods through meaningful experiments, with explicit hypotheses and
  feedback to implementation. Development can proceed alongside applications.

## P2 — Supporting foundational checks

- Operator regressions, convergence and small mathematical reference checks.
- Targeted investigations prompted by new findings.
- Performance profiling/benchmarking tied to actual workloads and scientifically
  comparable work, not automatic application-readiness claims from tiny timings.

P2 can run in parallel. It is **not automatically a prerequisite for every
P0/P1 attempt**. Genuine scientific/reference dependencies, source ownership,
resource conflicts and explicit execution gates still apply to affected steps.

## Iterative workflow and quality boundary

Hypothesis -> implementation -> practical experiment -> scientific assessment
-> feedback -> further development. Independent streams are encouraged;
new findings may change priorities or supersede ongoing approaches. There is
no general sequential milestone barrier.

Exploratory experiments may fail or remain inconclusive. They need not earn a
PASS or pass a complete scientific certification chain. Label them EXPLORATORY
or PRELIMINARY; never retroactively promote them into accepted evidence.
Substantial scientific results still require independent assessment. References,
observables, uncertainties and tolerances for acceptance claims are fixed
before candidate results; no post-hoc fitting or weakening. Focused delta
reviews suffice for bounded corrections unless the scientific argument changes.
All specific execution bans, Owner decisions, hard task budgets, frozen
references and spent-permit boundaries remain. Integration/publication retains
its own Content Lock and explicit Git authority. This direction starts no run.

## Dated current state and active research packets

The following is the Owner's status supplied with RESEARCH-PRIORITIES-01,
not a new numerical qualification by DOCS-AUDIT. Original exact-candidate reviews
remain authoritative for their tested envelopes; no new SHA or coverage is
invented here.

| Track | Status at this direction |
|---|---|
| R&D-TEST-01 | Independently accepted small GN functional demonstration; not general large-FWI or physical recovery qualification. |
| R&D-GROUP-01 | Independently accepted FD8 group-velocity evidence in its restricted scope; not blanket angle/boundary/FWI/RTM qualification. |
| Modern FD8 FWI/RTM | General integrated application path not yet demonstrated. |
| Earlier RTM tests | Relevant localization/full-image problems remain open. |
| REAL-06B / R1 | Complete large GN execution and corresponding qualification remain open. |

Active new research packets: **APP-FWI-01**, **APP-RTM-01**, **SCI-RTM-01**.
Their exact packets, not this list, govern permitted work, resources and gates.
The next practical focus is controlled multistep FWI/model/data comparisons and
reflected-data RTM/image QC, with method work driven by observed problems.
No workers are launched or messaged and no permits are transferred here.

## Handover minimum

Start new main chats and substantial research tasks from these priorities and
the research journal (kept privately). It records substantial application
experiments, observed status, insights/refuted hypotheses, limitations,
development consequences and direct completion/review references (or an explicit
missing review). Synthesize related experiments and update the dated scientific
view/open questions; do not log every small test. Append dated corrections to
historical entries. Editorial maintenance grants no new Scientific PASS, Owner
gate, execution permit or publication authority.

Every future chat/worker handover states the current practical research question,
ongoing experiments, scientific development streams and concrete next application
steps. Include exact source/worktree/data/evidence identities, results and
limits, open risks and remaining execution/publication authority. Old test
matrices are background/reference, not automatic priority setters. Preserve
historical evidence when an approach is superseded; a changed priority is not
retroactive acceptance, a lifted STOP or a new numerical permit.

See the [review policy](modernization_review_policy.md)
and [historical milestone ledger](modernization_status.md).
