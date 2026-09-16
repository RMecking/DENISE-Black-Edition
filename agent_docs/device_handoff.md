# Device Handoff / Rechnerwechsel

## Repository

- Repository: `RMecking/DENISE-Black-Edition`
- Arbeitsbranch: `codex/m6.3c-visco-sh-discrete-adjoint-gradient`
- Integrationsbranch: `modernization`
- Exact Base SHA: `0a79bd3b7198aca4ccb5819631a1a9bf7bd5f169`
- HEAD vor dem Transport-Checkpoint: `fdec89ec789174ee27178cd44ef17a5a773fe288`
- HEAD nach dem Transport-Checkpoint: dieser Handoff-Commit; nach der Veröffentlichung mit `git rev-parse HEAD` verifizieren
- Erwarteter Remote-Branch: `origin/codex/m6.3c-visco-sh-discrete-adjoint-gradient`
- Bestätigter Remote-HEAD vor dem Checkpoint: `fdec89ec789174ee27178cd44ef17a5a773fe288`

## Aktiver Task

- Task/Milestone: `M6.3c` — exact discrete viscoelastic SH adjoint/gradient repair
- Ziel: die eingefrorene diskrete viskoelastische-SH-Objective-/Adjoint-/Gradient-Kette innerhalb des bestehenden M6.3c-Scope weiterzuführen.
- Relevante Contracts/Invariants: `tests/m6.3c_acceptance_contract.json`, die eingefrorenen M6.3b-Oracles, die M6.3c-C8-Active-Path-Inventur sowie die exakte Trial-Objective-Komposition aus der aktuellen Branch-Historie.
- Erlaubter Scope: der bestehende M6.3c-Branch und seine Task-Dokumentation, Tests und Produktionsdateien gemäß Task Packet.
- Non-goals: kein Arbeiten direkt auf `modernization`, keine automatische Aktivierung außerhalb des Task Scopes, kein PR, kein Merge, kein Rebase, kein Amend und kein Force-Push.

## Aktueller Stand

- Der kanonische Remote-Arbeitsbranch steht bei `fdec89e` (`test(sh): verify exact trial objective runtime`).
- Der ursprüngliche lokale Checkout stand bei `67f1040`; er hatte keine eigenen, nicht veröffentlichten Commits und war gegenüber dem Remote-Branch veraltet.
- Für den Handoff wurde deshalb der bestätigte Remote-Stand `fdec89e` als Grundlage verwendet; der veraltete lokale Stand wird nicht transportiert.
- Noch unfertige Arbeiten und offene Entscheidungen: aus diesem Handoff-Check nicht neu bewertet; zuerst auf dem Laptop die Branch-Historie, das Task Packet und die kanonischen Projektdokumente lesen.
- Nächster sinnvoller Schritt: auf dem Laptop `Gerätewechsel fortsetzen`, den Remote-HEAD und den sauberen Working Tree verifizieren und anschließend den im Task Packet dokumentierten nächsten M6.3c-Schritt auswählen.

## Working Tree und Scope

- Der Handoff-Clone war vor Erstellung dieser Datei sauber.
- `agent_docs/device_handoff.md` ist die einzige Datei dieses Transport-Checkpoints.
- Die im ursprünglichen Checkout untracked vorhandenen `AGENTS.md` und `agent_docs/device_handoff_rules.md` gehören zur separat veröffentlichten Workflow-Infrastruktur und werden nicht in den M6.3c-Branch gemischt. Der Workflow-Branch ist `docs/codex-workflow-infrastructure` bei `7ad9713ec62ad8857be12082cc597f209f2fb57f`.
- Uncommitted M6.3c-Fachdateien waren im ursprünglichen Checkout nicht vorhanden.

## Verification

- `git ls-remote` bestätigte `origin/codex/m6.3c-visco-sh-discrete-adjoint-gradient` bei `fdec89ec789174ee27178cd44ef17a5a773fe288`.
- `git merge-base` mit `origin/modernization` ergab `0a79bd3b7198aca4ccb5819631a1a9bf7bd5f169`.
- Der Handoff-Clone wurde vom bestätigten Remote-Branch ausgecheckt; vor dieser Datei war `git status --short` sauber.
- Im ursprünglichen Checkout scheiterte `git fetch origin --prune` an fehlender Schreibberechtigung für `.git/FETCH_HEAD`; die Remote-Tips wurden deshalb zusätzlich mit `git ls-remote` verifiziert.
- In diesem Transport-Checkpoint wurden keine Builds oder Tests ausgeführt. Der Checkpoint ist keine vollständige fachliche Verification.

## Resume Contract

- Auf dem Laptop muss von `origin/codex/m6.3c-visco-sh-discrete-adjoint-gradient` bei dem bestätigten Remote-HEAD fortgesetzt werden.
- Vor Änderungen: `git fetch origin --prune`, Branch/HEAD/Remote-HEAD/Working-Tree prüfen, danach Root-`AGENTS.md`, geltende Workflow-Regeln, diese Datei, Task Packet und relevante Projektdokumente lesen.
- Der Task wurde an einem normalen Transport-Gate gestoppt, nicht als fachlich abgeschlossen oder review-fertig.
- Als Nächstes ist nur die Wiederaufnahmeprüfung und danach der dokumentierte M6.3c-Task-Schritt autorisiert.
- Nicht autorisiert sind PR, Merge, Push auf `modernization`, Rebase, Amend, Force-Push, Milestone-Statusänderung oder ungeprüfte Publication.
