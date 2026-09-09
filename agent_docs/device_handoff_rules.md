# Device Handoff / Rechnerwechsel

## Zweck

Ein Wechsel zwischen PC und Laptop soll mit möglichst wenig manueller Git- und Codex-Arbeit möglich sein.

Der GitHub-Stand ist die Transportebene zwischen den Rechnern. Ein Gerätewechsel darf jedoch niemals versehentlich als Review-, Publication-, PR- oder Merge-Gate interpretiert werden.

Die normalen Projektregeln bleiben vollständig gültig. Insbesondere haben die Root-`AGENTS.md`, projektspezifische Regeln und ausdrückliche Benutzeranweisungen Vorrang; diese Datei ergänzt sie nur.

`agent_docs/device_handoff.md` ist davon getrennt. Sie enthält ausschließlich den transienten, aktuellen Handoff-Zustand und ist keine dauerhafte Regeldatei.

## 1. Grundprinzipien

* GitHub und der aktuelle Repository-Zustand sind autoritativ.
* Chat-/Thread-Historie ist nicht autoritativ.
* `AGENTS.md` und die kanonischen Projektdokumente müssen nach einem Rechnerwechsel erneut gelesen werden.
* Niemals direkt auf `main`, `modernization` oder einem anderen Integrationsbranch arbeiten.
* Kein Rebase.
* Kein Amend veröffentlichter Commits.
* Kein Force-Push.
* Kein automatischer PR.
* Kein automatischer Merge.
* Kein `git add .`.
* Kein `git add -A`.
* Fremde, nutzereigene oder nicht zum aktuellen Task gehörende Änderungen niemals ungefragt stagen, committen, löschen oder überschreiben.
* Veröffentlichte Checkpoints bleiben unverändert. Spätere Korrekturen erfolgen additiv.

`latest_session_work.md` und `project_progress.md` sind keine Dateien für transienten Rechnerwechsel-WIP. Sie dokumentieren weiterhin nur dafür vorgesehene stabile bzw. verifizierte Projektzustände.

## 2. Expliziter Trigger

Die Benutzeranweisung `Gerätewechsel vorbereiten` oder sinngleich `Prepare device handoff` ist eine explizite Autorisierung für genau folgende zusätzliche Aktionen:

1. aktuellen Arbeitszustand analysieren;
2. einen Transport-Handoff erzeugen;
3. die für den aktuellen Task vorgesehenen Dateien explizit stagen;
4. bei vorhandenen uncommitted Task-Änderungen einen eindeutig als Transport/WIP gekennzeichneten Checkpoint-Commit erzeugen;
5. ausschließlich den aktuellen Arbeitsbranch zu `origin` pushen;
6. Local/Remote-Synchronität verifizieren.

Diese Autorisierung gilt nicht für:

* PR-Erstellung;
* Merge;
* Push auf Integrationsbranches;
* Rebase;
* Amend;
* Force-Push;
* Beginn eines neuen Tasks;
* Änderung des Milestone-Status;
* Interpretation des Transport-Commits als fachliche Acceptance oder Review-Gate.

Nach erfolgreicher Vorbereitung ist STOP.

## 3. Transport-Handoff-Datei

Für Rechnerwechsel wird `agent_docs/device_handoff.md` verwendet. Falls das Projekt eine andere kanonische Stelle für Agent-Dokumentation definiert, darf der entsprechende projektspezifische Pfad verwendet werden.

Diese Datei ist transient und darf nicht mit dem dauerhaften Projektfortschritt verwechselt werden.

Sie muss mindestens enthalten:

### Repository

* Repository
* aktueller Arbeitsbranch
* Integrationsbranch
* HEAD vor dem Transport-Checkpoint
* HEAD nach dem Transport-Checkpoint
* erwarteter Remote-Branch
* bestätigter Remote-HEAD

### Aktiver Task

* Task-/Milestone-ID
* Ziel des aktuellen Arbeitspakets
* Exact Base SHA, sofern für das Task Packet definiert
* relevante Contracts/Invariants
* erlaubter Scope
* explizite Non-goals

### Aktueller Stand

* bereits erledigte Arbeiten
* noch unfertige Arbeiten
* bekannte Probleme oder offene Entscheidungen
* exakt nächster sinnvoller Arbeitsschritt

### Working Tree

* geänderte Task-Dateien
* neu hinzugekommene Task-Dateien
* absichtlich nicht übertragene Dateien
* fremde oder nutzereigene Änderungen
* untracked Dateien und deren Behandlung

### Verification

* bereits ausgeführte Tests/Checks
* deren Resultate
* noch nicht ausgeführte Tests
* bekannte nicht verifizierte Teile

Ein Transport-Checkpoint darf niemals als vollständige Verification dargestellt werden.

### Resume Contract

Explizit festhalten:

* von welchem Branch fortgesetzt werden muss;
* welcher Remote-HEAD erwartet wird;
* ob der Task fachlich mitten in der Implementierung oder an einem normalen Gate gestoppt wurde;
* welche Aktion als Nächstes autorisiert ist;
* welche Aktionen ausdrücklich noch nicht autorisiert sind.

## 4. Vorbereitung des Gerätewechsels

Bei `Gerätewechsel vorbereiten`:

### A. Repository prüfen

Zuerst mindestens prüfen:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
```

Außerdem Remote aktualisieren:

```bash
git fetch origin --prune
```

Prüfen, dass nicht versehentlich auf dem Integrationsbranch gearbeitet wird.

Falls aktuell `main`, `modernization` oder ein anderer geschützter Integrationsbranch ausgecheckt ist und dort uncommitted Änderungen vorhanden sind: STOP. Keine automatische Commit-/Push-Aktion durchführen. Den Zustand klar melden.

### B. Scope bestimmen

Vor dem Staging:

```bash
git diff --name-status
git diff --stat
git diff --check
```

Falls bereits staged:

```bash
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
```

Task-fremde Dateien identifizieren und vom Transport ausschließen. Nie allein aufgrund ihres Vorhandenseins annehmen, dass untracked Dateien zum Task gehören.

### C. Handoff aktualisieren

`agent_docs/device_handoff.md` mit dem tatsächlichen Zustand aktualisieren. Keine erfundenen oder erwarteten Ergebnisse eintragen.

### D. Explizites Staging

Nur die tatsächlich zum aktiven Task gehörenden Dateien sowie `agent_docs/device_handoff.md` explizit stagen. Niemals `git add .` oder `git add -A` verwenden.

Danach erneut prüfen:

```bash
git status --short
git diff --cached --name-status
git diff --cached --stat
git diff --cached --check
```

### E. Transport-Commit

Nur falls für die Übertragung notwendig, einen eindeutig gekennzeichneten Commit erzeugen:

```text
wip(handoff): <TASK-ID> device transfer checkpoint
```

Der Commit bedeutet ausschließlich, dass dieser Git-Zustand für die Fortsetzung auf einem anderen Rechner transportierbar gemacht wurde. Er bedeutet nicht: implementation complete, tests complete, review passed, ready for PR oder ready for merge.

### F. Push

Nur den aktuellen Arbeitsbranch pushen, beispielsweise mit `git push -u origin <current-branch>` oder bei vorhandenem Upstream mit `git push`. Keinen anderen Branch pushen.

### G. Remote verifizieren

Nach dem Push mindestens vergleichen:

```bash
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git ls-remote --heads origin <current-branch>
git status --short
```

Local HEAD und Remote HEAD müssen identisch sein. Falls nicht: STOP. Nicht durch Force-Push, Rebase oder Amend lösen.

## 5. Abschlussbericht vor Rechnerwechsel

Nach erfolgreicher Vorbereitung einen kompakten Bericht ausgeben:

```text
DEVICE HANDOFF READY

Project:
Branch:
Local HEAD:
Remote HEAD:
Base SHA:
Working tree:
Transport commit:
Push:
Verification already completed:
Verification still pending:
Excluded/unrelated files:
Next authorized action:
Not authorized:
```

Danach STOP. Keine weitere Implementierung beginnen.

## 6. Fortsetzen auf einem anderen Rechner

Die Benutzeranweisung `Gerätewechsel fortsetzen` oder `Resume device handoff` autorisiert Codex dazu, den zuletzt vorbereiteten Transportzustand wiederherzustellen und den bereits begonnenen Task fortzusetzen. Sie autorisiert keinen neuen Task und kein Publication-Gate.

### A. Lokalen Zustand schützen

Zuerst:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
```

Falls auf diesem Rechner lokale, nicht übertragene Änderungen existieren: STOP. Nichts überschreiben, staschen, löschen oder automatisch committen.

### B. Remote laden

```bash
git fetch origin --prune
```

Den in `agent_docs/device_handoff.md` angegebenen Arbeitsbranch verwenden. Falls er lokal bereits existiert, `git switch <branch>` und `git pull --ff-only`; falls er lokal noch nicht existiert, `git switch --track origin/<branch>`. Kein Merge-Pull und kein Rebase.

### C. Zustand verifizieren

Prüfen:

```bash
git rev-parse HEAD
git rev-parse --abbrev-ref HEAD
git status --short
git rev-parse origin/<branch>
```

Anschließend erneut lesen: Root-`AGENTS.md`, alle geltenden untergeordneten `AGENTS.md`, `agent_docs/device_handoff.md`, relevantes Task Packet und relevante kanonische Projekt-/Milestone-Dokumentation. Der tatsächliche Git-Zustand hat Vorrang vor Chat-Historie.

### D. Handoff validieren

Vor Änderungen müssen Branch, HEAD, Remote HEAD, erwarteter Base SHA, Working Tree, Scope und nächster autorisierter Schritt übereinstimmen bzw. eindeutig sein. Falls ein Punkt nicht stimmt: STOP und Abweichung melden. Nicht durch Merge, Rebase, Reset oder Force-Push reparieren.

## 7. Fortsetzung des Tasks

Wenn alle Handoff-Prüfungen PASS sind:

* exakt beim dokumentierten nächsten Schritt fortsetzen;
* keine bereits erledigte Arbeit unnötig wiederholen;
* bestehende Implementierung nicht neu entwerfen, sofern der Handoff keinen entsprechenden Befund enthält;
* innerhalb des ursprünglichen Task Scopes arbeiten;
* normale Self-Gates des Projekts anwenden;
* Build, targeted tests, Regressionen, `git diff --check`, Scope- und Statusprüfungen wie im Task Packet vorgeschrieben durchführen.

Der Gerätewechsel hebt kein bestehendes Review- oder Publication-Gate auf.

## 8. Abschluss eines transportierten WIP-Tasks

Der `wip(handoff)`-Commit darf niemals nachträglich amended oder force-gepusht werden. Er ist ein unveränderlicher technischer Checkpoint. Weitere Arbeit erfolgt über additive Commits.

Vor einem späteren PR-/Publication-Gate gelten wieder vollständig die normalen Projektregeln. Gesamt-Diff gegen die autorisierte Base, vollständige Verification, Scope, User-/Review-Gate sowie Commit-, Push-, PR-, CI- und Merge-Gates bleiben getrennt. Der frühere Transport-Push stellt keinerlei Freigabe für diese Schritte dar.

## 9. Kein dauerhafter Workflow-State aus dem Chat ableiten

Nach einem Gerätewechsel niemals Aussagen aus einem vorherigen Chat als alleinige technische Grundlage verwenden.

Autoritativ sind:

1. GitHub / Repository;
2. aktueller Branch und SHA;
3. `AGENTS.md`;
4. Task Packet;
5. kanonische Projektdokumente;
6. `device_handoff.md` für den transienten Transportzustand.

Chatverläufe dürfen zusätzlichen Kontext liefern, aber keine widersprechenden Repository-Fakten überschreiben.

## 10. Ziel

Für einen normalen Rechnerwechsel sollen idealerweise nur zwei Anweisungen nötig sein:

Auf Rechner A: `Gerätewechsel vorbereiten.`

Auf Rechner B: `Gerätewechsel fortsetzen.`

Codex übernimmt dabei Git-Inspektion, Handoff-Dokumentation, explizites Staging, Transport-Checkpoint, Push, Remote-Verifikation und Wiederaufnahmeprüfung innerhalb der oben definierten Sicherheitsgrenzen.
