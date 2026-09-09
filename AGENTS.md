# Repository operating rules

## Personal safety and publication boundaries

- Ask the user before every command that would run as root, Administrator, or through `sudo`, including inside WSL. Prefer the normal user whenever possible.
- Do not perform destructive recursive deletion, broad cleanup, or irreversible filesystem operations without explicit user authorization and an exact target check. Cleanup of a narrowly identified temporary artifact created by the current task is allowed after validating its resolved path.
- Do not commit, push, merge, create or close a pull request, delete a branch, cancel or rerun a hosted workflow, or publish externally unless the current user request explicitly authorizes it.
- Preserve dirty worktrees and unrelated user files. Stage only explicitly intended paths; never use broad staging.

## Device handoff

The detailed, mandatory device-handoff procedure is maintained in [`agent_docs/device_handoff_rules.md`](agent_docs/device_handoff_rules.md). It applies whenever a user explicitly requests preparation or continuation of a device handoff.

These handoff rules supplement the safety and publication boundaries above. They do not override, narrow, or weaken them. If any rule conflicts, the stricter existing repository or user instruction prevails. `agent_docs/device_handoff.md` is reserved for the transient, current handoff state and is separate from the durable rules file.
