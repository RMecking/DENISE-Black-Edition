# DENISE Black Edition — Integrations-Roadmap

**Stand:** 2026-11-22 · **Basis:** `modernization@b5df20b` · **Status:** aus dem
Projekt-Audit vom 21.11.2026; die Owner-Entscheidungen E-1 bis E-17 sind vom
21. bis 23.11.2026 getroffen (Abschnitt 9). Veröffentlicht mit PR #114; nach einem
externen Architektur-Audit vom selben Tag präzisiert (Abschnitt 10).
Dieses Dokument gibt Orientierung. Es erteilt keine Ausführungs-, Commit- oder
Publikationsfreigabe und ändert keinen bestehenden Scientific Contract, HOLD
oder Content Lock ([AGENTS.md](../AGENTS.md) gilt unverändert).

Es ersetzt als Planungsdokument die Abschnitte „Current high-level roadmap" im
[Ledger](modernization_status.md) und die Planungsteile der
[M9L-Roadmap](m9l_modern_elastic_fd8_roadmap.md). Beide bleiben als historische
Evidenz erhalten.

## 1. Ziel

Eine **als Gesamtprogramm benutzbare** moderne (visko-)elastische
PSV-/SH-FWI mit Reflexions-FWI und RTM: ein `denise`-Binary, gesteuert wie
Köhns Original über Parameterdatei und Workflow-Datei, lauffähig auf Laptop,
Rechner4 (GPU) und Cluster, und nach jedem Integrationsschritt an denselben
realistischen Benchmarks gegen Köhns Original gemessen.

Maßstab für Fortschritt ist ab jetzt: **Was kann das veröffentlichte Programm
auf der Benchmark-Leiter (Abschnitt 4)?** Private Kandidaten, Referenzen und
Berichte zählen erst, wenn sie dort ankommen.

## 2. Audit: Stand am 21.11.2026

### 2.1 Was das veröffentlichte Programm kann

| Strang | Im Programm benutzbar | Was für einen Köhn-artigen Workflow fehlt |
|---|---|---|
| Legacy-Workflow (Köhn) | Elastische PSV/AC/VTI/TTI/SH-FWI mit Mehrschuss, Frequenzstufen, L-BFGS/PCG, STF. Forward und FWI (`INVMAT1=3`) lokal bitgleich zu Köhn `b07d81d`. | Bekannte Schwächen des Originals (approximierte Gradienten, CPML-Guard). |
| Exakte visko-SH-FWI (M6.3c, PR #107/#108/#110) | Mehrschuss, MPI, freie Oberfläche, CPML, physikalisches Q, Klassenskalierung, Beleuchtungs-Vorkonditionierer. L-BFGS als PR #112 offen. | `TIME_FILT`, `TIMEWIN`, `INV_STF`, `OFFSET_MUTE`, `NORMALIZE` werden abgewiesen. Damit keine Frequenzstufen, keine Zeitfenster, keine Quellsignalschätzung. Ganze Vorwärtsgeschichte im RAM. |
| Exakte visko-PSV-FWI (M7/M8, CUDA) | Exakte Gradienten für Vp/Vs/ρ/Qp/Qs, MPI-verteilt, eine GPU (lokal 24-fach gegenüber einem CPU-Kern). | Nur **eine Quelle**, FD4, L=1, keine freie Oberfläche, kein Filter, kein STF, nur Steepest Descent. Jede andere visko-PSV-FWI bricht kontrolliert ab. Eine visko-PSV-FWI wie bei Köhn ist im Programm derzeit **nicht** ausführbar. |
| Moderner elastischer PSV-Born-Operator (M9, FD4) | J/Jᵀ, Checkpoint-Replay, MPI, freie Oberfläche, eingeschränktes Fluid, CUDA; Rohmigration `MODE=2`; Marmousi-II-Kampagne mit 100 Schüssen. | Nur rohes Jᵀ-Bild: keine Beleuchtungskompensation, keine LSRTM, keine FWI auf diesem Operator. |
| FD8 | Taylor8-Forward für homogenen Festkörper (PR #103). | Alles außer Forward. |

### 2.2 Was nur privat existiert

| Strang | Ort | Stand | Bewertung |
|---|---|---|---|
| Holberg8/PHYSICAL01 mit Fluid/Festkörper-Kontakt (K1/A2), CPU und CUDA | `F:`, Rechner4 | Marmousi-II, 5+2 Schüsse, 2,5→5 Hz, L-BFGS. Mit A2-Randabschluss sinkt der tiefe Vs-Fehler nach 60 Updates um 10,6 % (K1: 1,1 %). Legacy bleibt bei gleicher Iterationszahl vergleichbar oder besser. | **Anwendungsnah und wertvoll**, aber dritte Operatorfamilie außerhalb des Repos. Seit 15.11. ohne Fortsetzung. |
| Taylor8-M9L-Linie (J/Jᵀ, GN/LSQR, REAL-0x, RTM Case 3, T1a) | `F:`, Dropbox, Worktree `t1a-fd8-born-private` | J/Jᵀ auf Kleinstgittern qualifiziert, R1-GN-Schritte auf 4 Schüssen, Case 3 nach mehrtägiger Referenzkette PASS, T1a S1/S2 PASS (gestaged). | **Abgedriftet.** Seit Ende Oktober überwiegend Speicher-, Präzisions- und Einschließungsnachweise auf 9×9 bis 49×49. Kein Anwendungsergebnis. |
| Reflexions-FWI: Python-SH-Prototyp (FD2, Variable Projection, 38 Tests) | Worktree `heuristic-shaw-e5e226` | Born, Jᵀ, Tomographie-Kernel, LSRTM, VP-Schleife funktionsfähig. Codex fand zwei echte Fehler (W statt W² im inneren Bildsolver, 0/0-Pfad). | Guter Start, nie auf einem Testmodell als Inversion ausgewertet. |
| Reflexions-FWI: Codex RFWI-C-FD4 | `F:\…\DENISE-Benchmark\rfwi-c-fd4-*`, rund 50 Berichte in drei Tagen | 73×81-Träger. Rohwellenbank fertig, Gradientenreferenz **INCOMPLETE**, Kandidatenvergleich **HOLD**. Zuletzt Unsicherheitszerlegung, Replay-Profile und ein CPU-Wächter. | **Abgedriftet.** Nach drei Tagen kein einziges RFWI-Update. Die Arbeit zertifiziert eine Referenz auf 10⁻¹³, bevor die Methode einmal gelaufen ist. |
| SH-Praxis | Rechner4, Dropbox | LVZ-Pilot r2→r5: Datenfit −90 %, Modell in der Tiefe kaum verbessert, Diagnose „optimiererbegrenzt" → L-BFGS. synth-A-Hohlraum: 5 von 79 Wahrheitsschüssen. | **Auf Kurs.** |
| Kernel-Performance, SH-Replay-Speicher | Worktree `heuristic-shaw-e5e226` | visko-PSV-Forward bis 2,1-fach bei bytegleichen Seismogrammen; SH-Spitzenspeicher 3,6 GB → 112 MB. | Fertige Prototypen, nicht integriert. |

### 2.3 Antwort auf die Leitfrage

- **SH-FWI: ja, zielgerichtet.** Pilot → Diagnose → Codepaket → Merge in je
  einem Tag (PR #108, #110, #112). Das ist das Muster für alle Stränge.
- **Visko-PSV-FWI im Programm: Stillstand seit September.** Der exakte Pfad ist
  korrekt und schnell, aber auf einen Schuss ohne Filter beschränkt. Niemand
  arbeitet an seiner Erweiterung. Das ist die größte Lücke zum Ziel.
- **Elastische PSV (PHYSICAL01/A2/CUDA): echter Anwendungsfortschritt, aber
  am Programm vorbei.**
- **Taylor8/M9L und Codex-RFWI: in Testumgebungen verstrickt.** Der Aufwand
  fließt in Nachweisschärfe auf Spielzeuggittern, nicht in Fähigkeiten.
- **RTM: Fundament vorhanden (M9), Abbildungsqualität seit A2.5 nicht
  weiterentwickelt.** Die privaten LSRTM-Versuche vom Oktober zeigten Nutzen
  und wurden nicht übernommen.

Strukturelle Ursachen:

1. **Fünf Operatorfamilien** (Legacy-SSG, M9-FD4, Taylor8, Holberg8/PHYSICAL01,
   Python-SH-Träger). Jede hat eigene Adjungierte, Tests und Beweisketten.
2. **Kein Integrationsziel pro Paket.** Pakete enden mit einem Bericht statt
   mit einer Programmfähigkeit.
3. **Nachweisaufwand ohne Verhältnis zum Risiko.** Explorative Versuche
   durchlaufen Verträge, Permits, Ledger und unabhängige Reviews je Teilschritt.
   Die Forschungsprioritäten vom 25.10. und die Lockerung vom 19.11. wirken im
   Codex-Strang nicht.
4. **Der tatsächliche Stand steht nicht im Repo.** Forschungsjournal und
   Forschungsprioritäten liegen unversioniert nur auf `F:`. Die Repo-Dokumente
   stehen auf dem 21.10.

## 3. Leitplanken

1. **Programm zuerst.** Jedes Arbeitspaket nennt vorab die Fähigkeit, die
   danach per Parameterdatei im `denise`-Binary wählbar ist, und die Stufe der
   Benchmark-Leiter, auf der sie gezeigt wird.
2. **Höchstens zwei Implementierungsstränge gleichzeitig** (einer SH, einer
   PSV/RTM), dazu höchstens eine Forschungs-Sandbox.
3. **Sandbox mit Frist.** Eine Sandbox-Frage läuft höchstens eine Woche.
   Danach: integrieren, parken oder verwerfen. Sandbox-Arbeit braucht Notiz und
   SHA, keine Verträge, Permits, Ledger oder Einzelreviews.
4. **Angemessene Prüfung für Integration.** Punktprodukttest (J/Jᵀ),
   Gradient gegen zentrale Differenzen, Regression gegen den vorherigen Stand,
   Ergebnis auf der Benchmark-Leiter, ein unabhängiger Review des Diffs.
   Intervall-Einschließungen und Nachweise unterhalb der FP32-Auflösung sind
   kein Integrationskriterium. Die Prüfungen gelten für den jeweils geänderten
   wissenschaftlichen Pfad. Punktprodukt- und Gradiententest prüfen nur die
   Konsistenz von Forward und Adjoint, nicht die Physik des Forwards: Ändert
   ein Paket Operator, Rand oder Material, kommt genau ein unabhängiger
   physikalischer Referenzfall hinzu (Ankunft/Dispersion, Grenzflächen-
   reflexion, Absorption oder Bildamplitude/-ort), vorab auf höchstens einer
   Seite festgelegt. Reine Workflow-Pakete brauchen das nicht.
5. **Eine Operatorfamilie je Aufgabe.** Nichtlineare FWI: der
   Velocity-Stress-Kern des Programms mit exaktem Adjoint (SH und PSV, CPU-MPI
   und CUDA). Linearisierte Aufgaben (RTM, LSRTM, RFWI-Innenproblem): der
   M9-Born-Operator. FD8, A2-Kontakt und Taylor/Holberg werden Optionen dieser
   Kerne, keine eigenen Programme. Weitere Lösungswege können später
   hinzukommen; zunächst genügt ein Weg je FWI/RTM (E-1). Der Codevergleich
   K-6 (21.11.2026) hat die Wahl des Velocity-Stress-Kerns als Träger der
   nichtlinearen FWI bestätigt: Der private Holberg8/A2-Pfad ist eine
   elastische Erweiterung des M9-Born-Operators, kein zweiter FWI-Kern. FD8
   und der Wasserkontakt kommen über einen begrenzten Umbau in den
   Programmkern (Strang PSV, E-9).
6. **Legacy bleibt Vergleichsbasis**, wird nicht gelöscht und nicht verändert.

## 4. Benchmark-Leiter

Feste Eingaben, nach jedem Integrationsstand unverändert gerechnet, jeweils
gegen Köhn `b07d81d` und gegen den vorherigen Stand. Kennzahlen: Datenmisfit
(Training/Heldout), Modellfehler gesamt und nach Tiefe, Laufzeit, Speicher.

| Stufe | Fall | Vorhanden |
|---|---|---|
| B1 | SH, LVZ-Pilotmodell (Freeze r6) | Eingaben, Legacy- und Modern-Läufe. **Erste Zahl (r6, 21.11., Stand `67dd7ae`):** Vs-Modellfehler relativ zum Start nach 60 Iterationen 0,53–0,57 mit L-BFGS gegen 0,82–0,83 mit steilstem Abstieg; Fehler in der Lockerungszone 0,23–0,25 gegen 0,57. Eine Frequenzstufe. **Mehrstufig (r7, 22.11., Stand `4ca93d7`):** zwei gefilterte Stufen und eine Vollbandstufe zu je 20 Iterationen, ohne und mit Pickfenster. Vs-Modellfehler 0,58 und 0,60 (r6: 0,57). Das vorab festgelegte Nicht-Verschlechterungs-Kriterium gegen r6 ist am Q-Fehler **verfehlt** (0,39 und 0,29 bei Schwelle 0,24); alle Kriterien zur Korrektheit von Filter, Fenster, Mute und Reload sind erfüllt. In B1 startet Q im ausgewerteten Bereich am wahren Wert; der Q-Fehler misst dort nur Übersprechen. **Folgelauf r7b (22.11., gleicher Code):** Q in den gefilterten Stufen nicht aktualisiert, sonst wie r7; alle Kriterien erfüllt, Q-Fehler 0,12 und 0,15 (besser als r6 mit 0,20), Vs-Modellfehler unverändert 0,58 und 0,60. Für S-3 kommt eine Variante mit falschem Start-Q hinzu. |
| B2 | SH, synth-A-Hohlraum (Paper III) | 5 von 79 Wahrheitsschüssen |
| B3 | PSV, Marmousi-II OBC, 5+2 Schüsse, 2,5/3,5/5 Hz, elastisch und mit Q. Der Fall hat Wasserschicht, freie Oberfläche und FD8 und ist mit dem exakten Pfad erst nach P-3, P-4 und P-5 rechenbar. | Banken, Legacy- und private Modern-Ergebnisse |
| B3-S | PSV, Marmousi-II **ohne Wasserschicht** (E-13): Wasserzeilen abgeschnitten, CPML an allen vier Seiten, FD4, L=1, homogenes Qp = Qs = 80, invertiert werden Vp und Vs, eine gemeinsame Datenbank für beide Programme. PSV-Stufe der Leiter, bis B3 rechenbar ist. Nicht mit früheren Marmousi-Läufen vergleichbar. Lauf nach E-16: Quellen und Empfänger unterhalb der oberen CPML, 5 Trainings- und 2 Kontrollschüsse, 2,5/3,5/5 Hz, je höchstens 25 Iterationen, Kriterien vorab eingefroren. | Kleine Fassung gerechnet (22.11., Kandidat lokal): CPU und CUDA nach acht Schritten bytegleich, Datenmisfit −18 % und −14 %, Modellfehler praktisch unverändert. Der Aufbau war ungeeignet (Quellen und Empfänger in der CPML, 4 Iterationen je Stufe). **Lauf nach E-16 (23.11.):** 75 Iterationen in 153 s auf der GPU; Trainings- und Kontrollmisfit sinken in allen drei Stufen, CPU und GPU bytegleich, Forward gegen Köhn bei gleicher Dämpfung 4e-7 bis 5e-7. Das Kriterium zur Modellverbesserung ist **verfehlt** (Vs im oberen Band −0,18 % statt mindestens −2 %). Im verwendeten Paket ist Vs genau Vp/√3; für P-2 wird eine Fassung mit unabhängigem Vs festgelegt. |
| B4 | RTM/LSRTM/RFWI, Marmousi-II, 100 Schüsse (A2-Kampagne), Variante ohne tiefe Frequenzen und mit kurzen Offsets | A2-Daten und Rohbild |
| B5 | Felddaten: ein SH-Profil der Ownerin. Für PSV/RTM bleibt Marmousi-II (B3/B4) der Referenzfall; ein PSV-Felddatensatz ist vorerst nicht vorgesehen (E-4). | Profil von der Ownerin zu benennen |

## 5. Plan

Zeitangaben sind Schätzungen aus dem bisherigen Tempo, keine Zusagen.

### Phase 0 — Konsolidieren (etwa eine Woche)

| Paket | Inhalt |
|---|---|
| K-1 | **Erledigt 21.11.:** PR #111 (CUDA-Link, −0.0-Gradienten), #112 (SH-L-BFGS) und #113 (Ledger) sind gemergt. |
| K-2 | **Nicht gelandet, geparkt:** T1a (privater FD8-Born-Pfad) wurde als PR #115 vorgelegt; die Hosted-CI war rot, der PR ist ohne Merge geschlossen, der Zweig `claude/t1a-fd8-born-private` bleibt erhalten. Die zwei roten Tests (`test_m63c_distributed_objective_directional_fd`, Fall `b2_corner-2-2-3-physical-4-0-0`; `test_psv_viscoelastic_rheology::test_m43_selected_mpi_reproducibility[SV-1-2]`) scheitern beim Auswerten der MPI-Programmausgabe. Ursache offen; ein Zusammenhang mit dem Kandidaten ist weder belegt noch ausgeschlossen. Die Case-3-Qualifikation des eingefrorenen Kandidaten bleibt bestehen. Die Taylor8-Linie ruht (Abschnitt 7). |
| K-3 | **Erledigt 21.11.:** RFWI-C-FD4 mit Statusnotiz geschlossen (Referenz INCOMPLETE, geparkt). Die zwei Quellenbefunde (W², 0/0) sind im Prototyp behoben (RFWI-SANDBOX-01). |
| K-4 | Forschungsprioritäten ins Repo übernehmen (`docs/research_priorities.md`). Das Forschungsjournal bleibt auf Owner-Entscheidung privat. Ledger auf den Stand der gemergten PRs bringen. |
| K-5 | Fertige Performance-Prototypen integrieren: Allokator-Versatz, visko-PSV-Zeilenschleife (beide bytegleich), SH-Checkpoint/Replay. |
| K-6 | **Erledigt 21.11.:** lesender Codevergleich PHYSICAL01/Holberg8 gegen den Programmkern. Im Inneren passen beide zusammen (Staggering, Update-Reihenfolge, Holberg8-Koeffizienten, CPML-Rekursion, Checkpoint/Replay, Zielfunktion). Physikalisch verschieden sind Viskoelastizität und Q (nur Programm), der Wasserkontakt mit doppeltem Grenzknoten (nur privat), die Ränder und die CPML-Profilwerte. Der private Pfad ist elastisch in (λ, μ) bei fester Dichte, läuft auf einem Rang mit fester Geometrie und stammt aus `src/PSV/elastic_psv_born.c`. Die A2-Tabelle ist an Holberg8 gebunden. Alles Lesebefund, nichts gebaut oder gerechnet. Folgen: Strang PSV und E-9 bis E-12. |

### Strang SH — vom Pilot zu Felddaten

| Paket | Fähigkeit im Programm | Nachweis |
|---|---|---|
| S-1 | **Erledigt 21.11.:** L-BFGS und Vs-zuerst-Hierarchie (PR #112) | B1: Pilot r6 besteht erstmals alle Pilotkriterien; L-BFGS erreicht in Iteration 9 das Niveau, für das steilster Abstieg 60 Iterationen brauchte. Restproblem: Dichte-Crosstalk im Arm mit freier Dichte (vorgemerkt für S-4). |
| S-2 | **Integriert 22.11. (PR #120, `b5df20b`), mit begrenztem Anspruch (E-15):** Frequenzstufen (`TIME_FILT`), Zeitfenster um Picks (`TIMEWIN` = 1 mit `TWIN-` und `TWIN+`) und Offset-Mute im exakten Pfad, nach den Festlegungen E-7 und E-14 | Gradient gegen zentrale Differenzen mit aktivem Filter, zweiseitigem Fenster und Maske; bitgleich mit allen Schaltern aus; B1 mehrstufig gegen r6 (r7: Korrektheit belegt, Kriterium zur Q-Rekonstruktion verfehlt, keine Aussage dazu); B2 offen |
| S-3 | Quellsignalschätzung (`INV_STF`) je Schuss und Stufe; alternativ quellunabhängiger Misfit | B2 mit Ricker-Wahrheit und Spike-Start. Vorab zu klären (Befund aus r7): Der erste Schritt jeder Stufe gibt jeder aktiven Parameterklasse dieselbe relative Schrittgröße, obwohl die Daten auf Q nur mit 1–4 % der Empfindlichkeit von Vs reagieren; daraus entsteht der größte Teil des Q-Übersprechens. Eine Änderung dieser Skalierung ist Produktionsverhalten und eine Owner-Entscheidung. |
| S-4 | Spurnormierung, robuste Misfits (normierte Kreuzkorrelation, Envelope) | B2, dann B5 |
| S-5 | Erste Felddaten-SH-FWI | B5 |

### Strang PSV — exakte visko-FWI als Köhn-Workflow

| Paket | Fähigkeit im Programm | Nachweis |
|---|---|---|
| P-1 | Mehrschuss und Frequenzstufen im exakten visko-PSV-Pfad, CPU-MPI und CUDA. Stand 21.11.: Kandidat lokal; Mehrschuss-Gradient gleich Summe der Einzelschüsse, Gradient gegen zentrale Differenzen für alle fünf Klassen, vier MPI-Aufteilungen bytegleich, zwei Stufen mit akzeptierten Schritten. CPU und GPU sind bytegleich, wenn CUDA ohne FMA-Kontraktion baut (E-13); der Laufzeitverlust beträgt 4,3 %, der Build ohne FMA ist damit Standard. Stand 22.11.: letztes Sample nach E-14 umgesetzt (im PSV-Testfall 10,6 % der Zielfunktion). Stand 23.11.: Forward-Vergleich nach E-16 erfüllt, B3-S-Lauf ohne nennenswerte Modellverbesserung (siehe B3-S): 97 % der Vs-Änderung liegen an der Empfängerzeile, weil die Schrittlänge sich nach dem größten Gradientenwert richtet und der exakte Pfad weder Quell-Taper noch Vorkonditionierung zulässt (P-2). Der unabhängige Diff-Review vom 23.11. lautet FAIL (unter anderem zugelassene Fenster- und Integrationsschalter ohne passende Adjoint-Quelle); Korrekturen beauftragt. Kandidat lokal. | B3-S mit Q-Hintergrund; gegen Köhn ein Forward-Vergleich bei gleicher Dämpfung (E-16) |
| P-2 | L-BFGS mit Klassenskalierung und Beleuchtungs-Vorkonditionierer (Bausteine aus dem SH-Strang wiederverwenden) | B3: Modellfehler nach Tiefe |
| P-3 | Freie Oberfläche (eben) und STF im exakten Pfad | Landmodell, B3 mit freier Oberfläche |
| P-4 | FD-Ordnung 8 (Holberg) als Option des exakten Pfads, **vor** dem Wasserkontakt, weil die A2-Tabelle an Holberg8 gebunden ist (E-9). Bauweise: die vertikale und horizontale Ableitung im exakten Pfad werden ein austauschbarer Baustein (Innenstencil plus optionale Randtabelle), der Adjoint-Stencil eine Schleife über die Halbordnung; CPU und CUDA. In drei Schritten: erst die Kapselung mit FD4 bitgleich, dann FD8 mit eigener Transponierter, danach der Kontakt (P-5). | Punktprodukt und Gradient für FD8; **FD4 bitgleich zum Stand davor**; ein Dispersions-Referenzfall; B3 bei 5 Hz, Kosten je Wellenlänge |
| P-5 | Fluid/Festkörper-Kontakt (A2-Abschluss) als Option des Programmkerns, zunächst **ebener Meeresboden** (E-10). Neu herzuleiten und zu prüfen: Transponierte der Randoperatoren im Programmlayout, Zusammenspiel mit den Speichervariablen, ρ- und Q-Gradient in den Grenzzeilen, fluidfeste Gradientenabbildung, MPI über die Kontaktzeilen, Transponierte des OBC-Empfängers. | Punktprodukt und Gradient um die Grenze für alle fünf Klassen; „Kontakt aus" bitgleich zum Stand davor; Referenzfall Wasserschicht über Halbraum; B3: Tiefenwirkung Vs wie im privaten A2-Vergleich |
| P-6 | **Topographie und variabler Meeresboden** (Owner-Anforderung vom 21.11., wichtig): Landdaten mit Geländerelief und nicht ebene Meeresböden im exakten Pfad. Vorab eine Sandbox zur Methodenwahl, siehe unten. | Referenzfall mit geneigter und gestufter Oberfläche gegen eine unabhängige Lösung; Gradient gegen zentrale Differenzen; Landmodell mit Relief |

P-1 ist das wichtigste Einzelpaket der Roadmap. Es macht aus einem
verifizierten Ein-Schuss-Demonstrator eine benutzbare FWI.

**Gefilterte Zielfunktion (E-7).** Für P-1 gilt Köhns `TIME_FILT`-Semantik:
Der Stufenfilter B wirkt einmal auf das Quellsignal und einmal auf die
eingelesenen beobachteten Daten. Zielfunktion je Stufe:
Φ(m) = ½ Σ_s ‖F_s(m; B·s) − B·d_s‖². Der bestehende exakte Adjoint liefert
dafür bereits den exakten Gradienten; eine Filter-Transponierte ist nicht
nötig. Wirkt später ein Operator im Datenraum (Zeitfenster T, Offset-Maske M in
S-2), lautet die Zielfunktion ½ Σ_s ‖M T (F_s − B·d_s)‖² mit der Reihenfolge
Filter, Fenster, Maske, und die Adjoint-Quelle enthält (M T)ᵀ(M T) angewandt
auf das Residuum. Filterung des Quellsignals und Filterung im Datenraum sind
verschiedene diskrete Experimente und werden nicht stillschweigend getauscht.

**Präzisierungen nach der Prüfung des Legacy-Codes (E-14), für PSV und SH gleich.**

- *Letztes Zeitsample:* Köhns Filterroutine erreicht das letzte Sample jeder
  Spur nicht. In einem kleinen SH-Testfall mit 12-Hz-Tiefpass trug dieses
  eine ungefilterte Sample rund 80 % der Zielfunktion. Die Filterroutine
  bleibt unverändert; bei aktivem `TIME_FILT` geht das letzte Zeitsample aber
  weder in die Zielfunktion noch in die Adjoint-Quelle ein. Ohne Filter
  ändert sich nichts.
- *Vergleich mit Köhn:* Der Misfit wird für beide Programme mit derselben
  Auswertung ohne das letzte Sample gemessen. Köhns internes J enthält das
  Artefakt weiter und wird nicht direkt übernommen.
- *Zeitfenster:* Immer um die Zeiten der Pickdatei (`TIMEWIN` = 1); ein
  konstantes Fenster entsteht über eine Pickdatei mit gleichen Zeiten.
  `TIMEWIN` = 2 bleibt im exakten Pfad abgewiesen. `TWIN-` und `TWIN+` sind
  Zeiten in Sekunden als Abstand zum Pick: volles Gewicht zwischen
  Pick − `TWIN-` und Pick + `TWIN+`, davor und danach Gauß-Abfall mit `GAMMA`
  (Form aus der DENISE-layerstrip-Fassung der Ownerin von 2021). Im
  Legacy-Pfad wirkt `TWIN-` weiterhin nicht.
- *Fenster im Gradienten:* exakt, also mit dem Quadrat des Gewichts. Köhns
  Adjoint-Quelle gewichtet nur einfach und passt damit nicht zu seiner
  eigenen Zielfunktion; das wird im exakten Pfad nicht übernommen.
- *Später:* Eine Korrektur des Filter-Versatzes für alle Pfade wäre ein
  eigenes Bugfix-Paket, weil sie die Bitgleichheit zu Köhn aufhebt.

**Reihenfolge und Bauweise nach K-6 (E-9, E-10).** FD8 kommt vor dem
Wasserkontakt. Beides wird über einen begrenzten Umbau eingeführt: Die
räumliche Ableitung im exakten Pfad wird ein austauschbarer Baustein, so dass
FD-Ordnung und Randabschluss Daten statt kopierter Code-Zweige sind und die
Transponierte einmal je Baustein hergeleitet wird. Das greift in den einzigen
verifizierten exakten PSV-Pfad ein; ohne Nachweis, dass FD4 danach bitgleich
rechnet, wird nichts gemergt. Der private Pfad bleibt Vergleichsrechnung für
den elastischen Grenzfall, nicht Träger und nicht Orakel; sein Jᵀ belegt nichts
für den Programmkern.

**Entwurfsauflage für P-4 und P-5.** Der Baustein darf einen ebenen
Abschluss in fester Gitterzeile nicht fest verdrahten. Er muss so angelegt
sein, dass die Lage der Grenzfläche später je Spalte variieren kann (P-6).

**Topographie (P-6).** Vor dem Bau klärt eine Sandbox von höchstens einer
Woche die Methode, zuerst für SH, weil der SH-Feldbenchmark B5 sie am ehesten
braucht. Kandidaten: (i) Vakuum-Formulierung über die Materialmittelung, deren
Adjoint aus den vorhandenen Materialabbildungen folgt; (ii) Bildmethode auf
gestufter Oberfläche; (iii) gekrümmtes oder vertikal gestrecktes Gitter;
(iv) eingebettete Randbedingung. Kriterien: Genauigkeit der Oberflächenwellen
je Gitterpunkte pro Wellenlänge, exakter Adjoint mit vertretbarem Aufwand,
MPI- und GPU-Tauglichkeit. Ob das SH-Feldprofil Relief hat, legt fest, ob
die SH-Topographie vor S-5 fertig sein muss. Vorläufige Rangfolge nach dem
externen Audit vom 22.11.: (i) nur als Negativkontrolle, (ii) als Basislinie,
(iv) als bevorzugter Kandidat, (iii) als langfristige Option.

**Offene Punkte aus K-6.**

- *Fluidzellen im exakten Pfad:* Die Gradientenabbildung teilt durch μ² der
  Zelle; für Zellen mit Vs = 0 wäre das 0/0 (Lesebefund, nicht ausgeführt).
  P-1 prüft das Verhalten früh und repariert es nicht selbst; bis P-5 rechnet
  B3 ohne Wasserschicht oder mit fester, aus der Inversion ausgenommener
  Wasserschicht.
- *Dichteversatz an der Grenzfläche (Zeilen 22/23 der privaten Modelle):*
  Ob Konvention oder Fehler, ist ungeklärt. Wird vor P-5 lesend an den
  Modelldateien geprüft, bevor B3-Datenbanken neu erzeugt werden (E-11).
- *`abs()`-Wächter im CPML-Profil (`PML_pro.c`):* bleibt im Legacy-Pfad
  unverändert, damit die Bitgleichheit zu Köhn hält, und wird dokumentiert.
  Ein korrigiertes Profil nur für die exakten Pfade kommt frühestens mit P-4,
  nicht während P-1 läuft (E-12).
- *Privater Pfad im Strang Abbildung:* Als Erweiterung des M9-Born-Operators
  wäre er der günstige Weg zu elastischer RTM/LSRTM mit Wasserschicht und
  FD8. Zu bewerten, wenn A-1 ansteht.

### Strang Abbildung — RTM → LSRTM → RFWI auf einem Operator

Das Innenproblem der Reflexions-FWI ist eine LSRTM. Die drei Stufen bauen
deshalb aufeinander auf und laufen alle auf dem M9-Born-Operator.

**Geltungsbereich.** Der M9-Born-Operator ist **elastisch**, FD4, in (λ, μ)
bei fester Dichte. Die exakte PSV-FWI des PSV-Strangs ist **viskoelastisch**
in (Vp, Vs, ρ, Qp, Qs). J und Jᵀ des M9-Operators sind nicht die Ableitung des
viskoelastischen Forwards. A-1 bis A-3 sind deshalb ausdrücklich eine
**elastische** RTM/LSRTM/RFWI mit fester Dichte und eigenem elastischem
Zielfunktional: Hintergrundmodellierung, Born-Linearisierung und reduzierter
Gradient beziehen sich auf denselben M9-Graphen. Der elastische Hintergrund
wird nicht mit dem visko-PSV-Gradienten verkettet. Eine viskoelastische RFWI
braucht einen zum visko-Forward passenden Born-Pfad und ist ein eigener,
späterer Integrationsschritt (nach I-3).

| Paket | Fähigkeit im Programm | Nachweis |
|---|---|---|
| A-1 | RTM-Modus mit Beleuchtungskompensation und Impedanz-Parametrisierung des Bildes (unterdrückt langwellige Rückstreuartefakte) | B4 gegen A2.5-Rohbild; dazu ein eigener, vorab festgelegter Punkt- oder Reflektorfall für Bildort und relative Amplitude |
| A-2 | LSRTM (CGLS, beleuchtungs-vorkonditioniert, 10–20 Iterationen) | B4: Auflösung, Amplituden, Quell-/Zielenergie |
| A-3 | Elastische RFWI v1 auf M9 mit fester Dichte auf der **reduzierten Zielfunktion** (E-8): Das Bild ist die LSRTM-Lösung aus A-2 zum jeweiligen Hintergrund, und jeder Hintergrundschritt wird mit neu geschätztem Bild geprüft. Der Gradient ohne impliziten Term ist der Standard, solange innere und äußere Zielfunktion gleich sind; der implizite Term (Abhängigkeit des Bildes vom Hintergrund) wird als Schalter gebaut und ist Pflicht, sobald sie verschieden sind (andere Gewichte, starke Dämpfung, laufzeitartiger Misfit für den Hintergrund). Hintergrund-Update mit L-BFGS. | Gradient gegen zentrale Differenzen mit auskonvergiertem Bild bei gleicher Zielfunktion und mit implizitem Term bei verschiedener; **Nullkontrolle** als festes Kriterium (derselbe Lauf mit wahrem Hintergrund gleich Startmodell: das Ergebnis muss klar über diesem Drift liegen); B4-Variante ohne tiefe Frequenzen, Start mit falschem Hintergrund; Vergleich mit konventioneller FWI |
| A-4 | SH-Born-Tangente im SH-Kern und SH-RFWI mit demselben Treiber | Reflektor+Linse-Test aus dem Prototyp, dann LVZ und Hohlraum |
| A-5 | SH-RTM/RFWI auf dem Feldprofil | B5 |

**Sandbox vor A-3 (RFWI-SANDBOX-01, 21.11.2026, EXPLORATORY).** Der
Python-SH-Prototyp hat den Reflektor+Linse-Fall als Inversion gerechnet
(73×81 Knoten, eine Geometrie, rauschfrei, gleicher Löser für Daten und
Inversion). Ein Nachlauf mit gleichem Optimierer auf beiden Seiten
(RFWI-SANDBOX-01b) hat den ersten Befund abgeschwächt. Laufzeitfehler des
Hintergrunds in ms, Start 0,273, je 12 Hintergrundschritte:

| Formulierung, Optimierer | Linsenfall | Nullkontrolle | Datenmisfit (relativ) |
|---|---:|---:|---:|
| Variable Projektion, L-BFGS | 0,038 | 0,017 | 2e-5 |
| Variable Projektion, steilster Abstieg | 0,050 (schwankt 0,016–0,058) | 0,026 | 6e-5 |
| Alternierend, steilster Abstieg | 0,055 | 0,054 | 6e-4 |
| Alternierend, L-BFGS (Speicher je Bild geleert) | 0,129 | 0,049 | 7e-4 |

Beide Formulierungen bauen rund 80 % des Laufzeitfehlers ab. Mit steilstem
Abstieg auf beiden Seiten ist der Laufzeitfehler im Linsenfall praktisch
gleich; der Vorsprung der variablen Projektion im ersten Lauf kam zum Teil vom
L-BFGS. Für die variable Projektion sprechen danach: eine Drift in der
Nullkontrolle, die halb bis ein Drittel so groß ist, ein 10- bis 30-fach
kleinerer Datenmisfit, ein festes Zielfunktional mit prüfbarem Gradienten
(Übereinstimmung mit zentralen Differenzen auf 10⁻⁹; ohne impliziten Term 40 %
daneben) und die Verträglichkeit mit L-BFGS, das der alternierenden Form
schadet. Dagegen steht die zwei- bis dreifache Rechenzeit je Schritt. Die
konventionelle FWI senkt auf reinen Reflexionsdaten den Misfit um drei
Größenordnungen, ohne den Hintergrund zu verbessern. Je Variante lief nur ein
Lauf; Unterschiede von 0,01 ms liegen in der Schwankung zwischen Schritten.
Folgerungen für A-3:

1. Maßstab ist die reduzierte Zielfunktion (E-8): Ein Hintergrundschritt gilt
   nur, wenn der Datenfehler mit neu geschätztem Bild gefallen ist. Eine
   Mischform (Schritt wie alternierend, dann Bild neu schätzen und Schritt
   prüfen) erreichte im Testfall das Niveau der variablen Projektion mit
   L-BFGS (0,037 ms, Nullkontrolle 0,020 ms) bei 13 Bildschätzungen für 12
   Schritte.
2. Der implizite Term war im Testfall klein (2 bis 18 % des expliziten
   Anteils), weil innere und äußere Zielfunktion gleich waren. Bei
   verschiedenen Zielfunktionen lag er in den Prototyp-Tests bei 13 bis
   700 %; dort ist er Pflicht, und die Mischform ist dort nicht geprüft.
3. Die Nullkontrolle ist festes Prüfkriterium: Der Restfehler eines Verfahrens
   wird gegen seine eigene Drift im Kontrollfall gelesen.
4. Reflexionen legen die Laufzeit fest, nicht die Tiefenverteilung der
   Geschwindigkeit (der Geschwindigkeitsfehler im Zielbereich unterscheidet
   sich zwischen den Verfahren nicht). Regularisierung oder Skalentrennung
   zwischen Bild und Hintergrund ist vor dem Bau von A-3 festzulegen.

### Integrationsstände

| Stand | Inhalt | Schätzung |
|---|---|---|
| I-1 | Phase 0, S-1, S-2, P-1, P-2. Neues Cluster-Testpaket mit Parameterdateien für B1–B3. | 3–4 Wochen |
| I-2 | S-3, S-4, P-3, A-1, A-2. | weitere 4–5 Wochen |
| I-3 | P-4 (FD8), A-3, A-4, S-5. | weitere 5–7 Wochen |
| I-4 | P-5 (Wasserkontakt, ebener Meeresboden), P-6 (Topographie, variabler Meeresboden), A-5, danach Anisotropie (VTI vor TTI). | offen |

## 6. Forschungsaspekte mit Aussicht

Jeweils an ein Paket gebunden, damit sie im Programm landen.

- **Exakter visko-Adjoint als Alleinstellung** (P-1). Qp/Qs-Inversion mit
  physikalischem Q steht mit P-1 zur Verfügung. Q-kompensierte RTM und LSRTM
  wären mit einem exakten visko-Adjoint stabil formulierbar, ohne die
  instabile Amplitudenverstärkung üblicher Q-RTM; sie setzen aber einen
  viskoelastischen Born-Pfad voraus, den es noch nicht gibt (nach I-3).
- **A2-Fluid/Festkörper-Abschluss** (P-5). Der private Befund (tiefer Vs-Fehler
  −10,6 % statt −1,1 %) ist ein eigenständiges, publizierbares Ergebnis.
- **Mehrparameter-Kopplung** (P-2). Die Diagnose vom 2.11. zeigt 49 %
  Vp-Blockanteil in der Vs-L-BFGS-Richtung. Kandidaten: Block-Vorkonditionierung
  je Parameterklasse, Ip/Is/ρ-Parametrisierung, später abgeschnittenes
  Gauss-Newton (der private GN/LSQR-Code ist dafür wiederverwendbar).
- **Skalentrennung in der RFWI** (A-3). Geschwindigkeit für die Kinematik,
  Impedanz für die Reflektivität (Zhou, Brossier, Operto, Virieux 2015);
  Pseudozeit-Formulierung, damit Reflektoren beim Hintergrund-Update konsistent
  bleiben (Plessix 2013); laufzeitartige Misfits (Korrelation, dynamisches
  Matching) für den Tomographieanteil, weil auch L2-RFWI Zyklen überspringt.
- **Hohlraum und Lockerungszone** (S-4, A-4). SH-RTM der rückgestreuten
  Love-Wellen lokalisiert den Streuer; die FWI rekonstruiert danach die glatte
  LVZ. Für den Hohlraum selbst: TV-Regularisierung oder Level-Set statt
  glatter Updates.
- **Felddaten-Robustheit** (S-3, S-4). Quellunabhängige Misfits
  (Faltung mit Referenzspur), Graph-Space Optimal Transport für
  oberflächenwellendominierte Landdaten.

## 7. Geparkt

Nicht verworfen, aber erst nach I-3 wieder zu bewerten:

- Taylor8-Zertifizierungslinie nach T1a (REAL-06B, ZMR, Case-3-Folgen, R2/R3).
  Wiederaufnahme nur über P-4: als FD8-Option der Programmkerne mit wählbaren
  Koeffizienten (Taylor oder Holberg), wenn B3/B4 Dispersion oder Kosten als
  Engpass zeigen. Der T1a-Code für J/Jᵀ wird dann wiederverwendet; die
  Einschließungskette wird nicht fortgesetzt.
- Intervall-Einschließungen und Referenzbanken für RFWI-C-FD4.
- K1-Nullmoden-Analyse, sofern P-5 den A2-Abschluss übernimmt.
- MPI-Overlap, Legacy-FD8-Ablösung (M9L-7), 3D, anisotrope FWI.
- Unsicherheitsquantifizierung und ML-Priors (Abschnitt 8).

Historische HOLDs bleiben unverändert bestehen. Sie halten keine Arbeit dieser
Roadmap auf, sperren aber weiterhin ihre ursprünglichen Aussagen und können
Voraussetzungen neuer Aussagen offen lassen:

| HOLD | Wirkung auf diese Roadmap |
|---|---|
| M9L-4G (GN, τ=0,0001, Line Search erschöpft) | Sperrt den eingefrorenen GN-Erfolg, nicht SD/L-BFGS in P-1/P-2. Bei Wiederverwendung des GN-Codes bleibt der HOLD sichtbar. |
| M9L-5P-P3 (relative Vollbildamplitude, 10 %) | Stammt aus dem privaten Taylor8-Pfad und belegt keinen Fehler des M9-FD4-Operators. Er zeigt aber, dass Bildamplituden nicht qualifiziert sind: A-1 braucht einen eigenen Amplituden- und Ortsnachweis. Ein J/Jᵀ-Punktprodukt oder ein gutes kompensiertes Bild ist kein nachträglicher P3-PASS. |
| REAL-02D (Dreischussbild unvollständig) | Kein Beleg für einen Operatorfehler, aber auch kein Mehrschuss-RTM-Nachweis für diesen Pfad. |
| Legacy Check 7 / A3.0 (FD8/L0-Adjoint, CPML) | Widerlegt weder den M9-FD4- noch den exakten visko-FD4-Pfad. Relevant für K-6, P-4/P-5 und jede Wiederverwendung des Legacy-FD8- oder CPML-Adjoints. |

## 8. Ausblick: aktuelle Forschungstrends

- **FWI-Imaging und Mehrparameter-FWI.** Die Industrie invertiert
  Geschwindigkeit und Reflektivität gemeinsam und nutzt das FWI-Reflektivitäts-
  bild als Alternative zur LSRTM, seit 2025 auch elastisch
  ([TGS, EAGE 2026](https://www.tgs.com/hubfs/Technical%20Library/EAGE%202026/MPEFWI_eage26_submit_main.pdf),
  [TGS, Elastic FWI 2025](https://www.tgs.com/articles/elastic-full-waveform-inversion-enhance-imaging-for-legacy-and-modern-acquisition)).
  Das entspricht dem Strang Abbildung (A-2/A-3). Die Quellen sind
  Herstellerangaben.
- **Land-FWI.** Offene Punkte sind schwaches Signal bei tiefen Frequenzen und
  die Quellsignalschätzung; „Dynamic Matching" stabilisiert die Updates
  ([TGS Land-MP-FWI, EAGE 2026](https://www.tgs.com/hubfs/Technical%20Library/EAGE%202026/EAGE_2026_MPFWI-land_firstdraft_f.pdf)).
  Elastische Landfälle kombinieren Tauchwellen-FWI mit Refraktions-/Reflexions-
  tomographie ([CGG](https://www.cgg.com/resources/technical-content/technical-abstract/elastic-fwi-application-land-data-set-middle-east)).
  Oberflächennah: Laufzeit- und Oberflächenwellen-Inversion als Start für
  elastische FWI ([Boise State, 2025](https://experts.boisestate.edu/en/publications/improving-resolution-of-near-surface-structure-imaging-based-on-e-2/)).
- **Misfits gegen Cycle-Skipping.** Graph-Space Optimal Transport für
  3D-elastische Landdaten ([HAL](https://hal.archives-ouvertes.fr/LJK/hal-02325580v1)),
  neue Transportmetriken für vorzeichenbehaftete Signale
  ([HV-Geometrie, arXiv 2025](https://arxiv.org/html/2505.01817v2)),
  erweiterte Formulierungen (Extended Source, WRI/ADMM).
- **Bayes und generative Priors.** Diffusionsmodelle als Prior oder
  Regularisierer und für Posterior-Sampling
  ([DiffusionInv](https://arxiv.org/html/2505.03138v1),
  [Posterior-Sampling, KAUST 2025](https://arxiv.org/html/2512.12797v1)).
  Interessant für Unsicherheitsangaben, setzt aber eine schnelle, robuste FWI
  voraus und ist deshalb geparkt.

## 9. Owner-Entscheidungen

Entschieden von der Ownerin am 2026-11-21.

| ID | Frage | Entscheidung |
|---|---|---|
| E-1 | Eine Operatorfamilie je Aufgabe (Leitplanke 5)? | Ja. Zunächst ein Weg je FWI/RTM; weitere Lösungen können später hinzukommen. Details nach K-6. |
| E-2 | Codex-Strang RFWI-C-FD4 schließen und Codex auf P-1 oder A-1/A-2 ansetzen? | Ja. |
| E-3 | Taylor8-Linie nach T1a ruhen lassen? | Ja. Wiederaufnahme nur über P-4 (Abschnitt 7). |
| E-4 | Welche Felddaten für B5? | SH-Profil der Ownerin. Für PSV bleibt Marmousi-II der Referenzfall. |
| E-5 | Leitplanken 2–4 als Arbeitsregel in `AGENTS.md` übernehmen? | Ja. |
| E-6 | Reihenfolge RFWI: PSV auf M9 zuerst, SH danach (A-3 vor A-4)? | Ja. |
| E-7 | Filterstelle für Frequenzstufen in P-1? | Köhns Semantik: Filter auf Quellsignal und beobachtete Daten (Strang PSV). |
| E-8 | Formulierung der RFWI in A-3? | Reduzierte Zielfunktion (variable Projektion) statt der am 10.11. gewählten alternierenden Form, auf Grundlage von RFWI-SANDBOX-01 und 01b. Präzisiert am 21.11.: Annahmeprüfung mit neu geschätztem Bild immer; impliziter Term als Schalter, Pflicht bei verschiedenen Zielfunktionen. |
| E-9 | Reihenfolge und Bauweise nach K-6? | FD8 (P-4) vor Wasserkontakt (P-5); begrenzter Umbau mit austauschbarem Ableitungsbaustein; der private Pfad wird nicht Träger. |
| E-10 | Genügt für den Wasserkontakt ein ebener Meeresboden? | Für das Benchmarking vorerst ja. Variable Meeresböden und vor allem Landdaten mit Topographie sind als eigenes Paket P-6 verlangt. |
| E-11 | Dichteversatz an der Grenzfläche? | Noch nicht entschieden; erst lesend klären. |
| E-12 | `abs()`-Wächter im CPML-Profil? | Legacy unverändert und dokumentiert; korrigiertes Profil nur für die exakten Pfade, frühestens mit P-4. |
| E-13 | Benchmark und CUDA-Build für P-1? | Festkörperfassung B3-S als PSV-Stufe der Leiter. `--fmad=false` wird Standard für die CUDA-Objekte des exakten PSV-Pfads, wenn der gemessene Laufzeitverlust höchstens 15 % beträgt; die Toleranz bleibt unverändert. Gemessen 4,3 %. Ergänzt am 23.11.: Der Build ohne FMA gilt für den ganzen geteilten CUDA-PSV-Kern, also auch für den allgemeinen visko-PSV-Forward auf der GPU. |
| E-14 | Filter-Spurende und Zeitfenster im exakten Pfad? | Letztes Zeitsample bei aktivem Filter aus Zielfunktion und Adjoint-Quelle; Vergleich mit Köhn entsprechend ausgewertet. Fenster nur um Picks, `TWIN-` als Zeit vor dem Pick mit Gauß-Flanke, Fenster exakt im Gradienten. |
| E-15 | S-2 integrieren, obwohl r7 das Kriterium K6 verfehlt hat? (22.11.) | Ja, mit begrenztem Anspruch: r7 bleibt nicht bestanden, K6 wird nicht umgedeutet. S-2 ist als Filter-, Fenster- und Mute-Funktion angenommen, ohne Aussage zur Q-Rekonstruktion. Ein Folgelauf mit festem Q in den gefilterten Stufen (r7b) ist Diagnose vor S-3, keine Merge-Bedingung; er hat am 22.11. alle Kriterien erfüllt, ändert die Bewertung von r7 aber nicht. Empfohlenes Stufenschema bis S-3: Q erst in der Vollbandstufe freigeben. Künftig getrennt und jeweils vorab festgelegt: Kriterien zur Fähigkeit und Kriterien zur Rekonstruktionsgüte. |
| E-16 | B3-S-Lauf und Vergleich mit Köhn für P-1? (22.11.) | Akquisition unterhalb der CPML, drei Stufen zu je höchstens 25 Iterationen, Kriterien vor dem Lauf eingefroren (Misfit sinkt je Stufe für Training und Kontrolle, Vs-Modellfehler sinkt, im oberen Tiefenband um mindestens 2 % des Startfehlers). Der Inversionsvergleich mit Köhn entfällt, weil sein Modellleser eine andere Dämpfung setzt und der Vergleich nichts aussagte; an seine Stelle tritt ein Forward-Vergleich bei gleicher Dämpfung mit vorab notierter Erwartung. |
| E-17 | Quellsignalschätzung S-3 und Reihenfolge im SH-Strang? (23.11.) | Variable Projektion: Das Quellsignal wird bei jeder Auswertung der Zielfunktion neu geschätzt, als gewichtetes Kleinste-Quadrate-Problem mit denselben Gewichten wie die Zielfunktion; Köhns einmalige Schätzung je Stufe bleibt als Schalter. Ein Signal je Schuss, begrenzte Länge als Parameter, Regularisierung als Teil der Zielfunktion. Nachweis auf B1 mit unbekanntem Signal, weil B2 im exakten Pfad rund 355 GB je Schuss bräuchte. Parallel ein Eingabepaket B1 mit falschem Start-Q, darin die Klassengewichtung für Q als Arm. K-5 (Checkpoint/Replay für SH) folgt direkt nach S-3, vor S-4; es ist Voraussetzung für B2 und für Felddaten. Vp und Vs werden in den PSV-Benchmarks künftig unabhängig betrachtet. |

## 10. Externes Audit vom 21.11.2026

Ein externes Architektur- und Wissenschaftsaudit (Urteil: CONDITIONAL GO) hat
die Richtung bestätigt und Präzisierungen empfohlen. Die Ownerin hat sie als
Empfehlung behandelt und Folgendes übernommen:

- Kernwahl E-1 vorläufig bis K-6; K-6 vorgezogen und am selben Tag erledigt:
  Kernwahl bestätigt (Leitplanke 5, Phase 0).
- Filterstelle vorab von der Ownerin festgelegt statt vom Worker gewählt (E-7).
  Abweichend vom Audit-Vorschlag (Filter im Datenraum mit Transponierter) gilt
  für P-1 Köhns Semantik; die Datenraum-Formel gilt für Fenster und Maske.
- Integrations-Gate risikoadaptiv ergänzt, auf einen Referenzfall je Paket
  begrenzt (Leitplanke 4).
- Strang Abbildung ausdrücklich elastisch abgegrenzt.
- Wirkung der historischen HOLDs je Befund benannt (Abschnitt 7).
- Rolle „Project Coordination" in `AGENTS.md`.

### Audits vom 22.11.2026

Zwei weitere externe Audits (beide CONDITIONAL GO) begleiteten S-2 und P-1.
Übernommen:

- S-2 wurde nicht vor dem Lauf auf der Benchmark-Leiter gemergt; r7 lief auf
  dem offenen PR-Kopf mit vorab eingefrorenen Kriterien und einem Mute-Fall.
- Nach dem verfehlten K6: Integration nur durch ausdrückliche
  Owner-Entscheidung mit begrenztem Anspruch (E-15); Fähigkeit und
  Rekonstruktionsgüte künftig als getrennte Kriterien.
- Hypothesen zum Q-Übersprechen getrennt prüfen, nicht im selben Lauf mischen.
- Rangfolge der Topographie-Methoden und Stufung von P-4 (Strang PSV).
- Auftragspakete nennen vollständige, von einem Kommando ausgegebene Hashes.
- Beim Forward-Vergleich mit Köhn Relaxationsfrequenz, Quellsignal und
  Randschicht einzeln abgleichen; gleiches Q allein genügt nicht.

Nächster vorgesehener Audit-Zeitpunkt: wenn der B3-S-Lauf nach E-16
vorliegt, vor dem Content Lock für P-1.
