# Device Handoff / Rechnerwechsel — durable rules

## Purpose

A device handoff transports an already active DENISE task between machines. It
is not a scientific acceptance, review, publication, PR, or merge gate.

The root [`AGENTS.md`](../AGENTS.md) remains authoritative. These rules only
specialize device-handoff mechanics and never weaken repository safety,
scientific verification, worker-role, or publication boundaries.

`agent_docs/device_handoff.md` is transient current state. When no handoff is
active it should say so rather than preserving an obsolete task as current.

## Authority and triggers

The instructions `Gerätewechsel vorbereiten` / `Prepare device handoff` and
`Gerätewechsel fortsetzen` / `Resume device handoff` are explicit handoff
triggers only. They do not authorize a new task, scientific acceptance, PR,
merge, integration-branch push, rebase, amend, force-push, or milestone-status
change.

A handoff may create and push a clearly labelled transport/WIP checkpoint on
the existing task branch only when that is necessary to move uncommitted task
work and the user explicitly triggered handoff preparation.

## Preparation

Before changing anything, verify at least:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
git fetch origin --prune
```

Do not continue automatically if the working tree contains unrelated user work
or if uncommitted work is on `modernization`, `main`, or another integration
branch.

Before staging, inspect:

```bash
git diff --name-status
git diff --stat
git diff --check
```

If anything is already staged, also inspect the cached equivalents. Preserve
all unrelated or ambiguous changes. Never use `git add .` or `git add -A`.

Update `agent_docs/device_handoff.md` with the exact current transport state,
including:

- repository, task branch, integration branch, exact HEAD/base/remote state;
- active task and frozen contracts/invariants;
- allowed scope and explicit non-goals;
- completed and unfinished work;
- changed, untracked, excluded, and unrelated files;
- verification already completed and still pending;
- exact next authorized action and actions still not authorized.

If a transport commit is necessary, use an unmistakable WIP message such as:

```text
wip(handoff): <TASK-ID> device transfer checkpoint
```

A transport commit is not implementation completion or scientific acceptance.
Push only the current task branch. Verify local and remote HEAD equality after
the push. Then stop.

## Resume

Before restoring the transported task, protect the destination machine's local
state:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
```

If unrelated local changes exist, stop rather than overwriting, stashing,
resetting, deleting, or automatically committing them.

Fetch remote state and restore only the branch documented in
`agent_docs/device_handoff.md`, using fast-forward-only behavior. Do not merge
or rebase to repair a mismatch.

Before new work, reread:

1. root `AGENTS.md` and any applicable nested `AGENTS.md`;
2. `agent_docs/device_handoff.md`;
3. the active task packet;
4. relevant canonical scientific/project documentation.

Verify branch, HEAD, remote HEAD, expected base SHA, working-tree state, scope,
and next authorized step. If they do not match the recorded handoff, stop and
report the discrepancy.

Continue with the same worker role/configuration and original task scope. A
machine change does not reopen completed work, change scientific contracts, or
bypass independent verification/publication gates.

## Publication boundary

A handoff push carries state only. Any later commit, publication, PR, CI action,
or merge remains separately governed by the root `AGENTS.md`, the active task
packet, and explicit authorization.
