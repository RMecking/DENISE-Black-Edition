# DENISE Black Edition — Integrations-Roadmap

**Stand:** 2026-11-21 · **Basis:** `modernization@86fd2bf` · **Status:** aus dem
Projekt-Audit vom 21.11.2026; die Owner-Entscheidungen E-1 bis E-8 sind am
21.11.2026 getroffen (Abschnitt 9). Veröffentlicht mit PR #114; nach einem
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
   hinzukommen; zunächst genügt ein Weg je FWI/RTM (E-1). Die Wahl des
   Velocity-Stress-Kerns als Träger gilt für P-1 bis P-3 und ist **vorläufig
   bis K-6**: Erst der Codevergleich mit dem privaten Holberg8/A2-Pfad
   entscheidet, ob dessen Bausteine als Option in diesen Kern passen, als
   eigenes Modul bleiben oder einen begrenzten Umbau verlangen.
6. **Legacy bleibt Vergleichsbasis**, wird nicht gelöscht und nicht verändert.

## 4. Benchmark-Leiter

Feste Eingaben, nach jedem Integrationsstand unverändert gerechnet, jeweils
gegen Köhn `b07d81d` und gegen den vorherigen Stand. Kennzahlen: Datenmisfit
(Training/Heldout), Modellfehler gesamt und nach Tiefe, Laufzeit, Speicher.

| Stufe | Fall | Vorhanden |
|---|---|---|
| B1 | SH, LVZ-Pilotmodell (Freeze r5) | Eingaben, Legacy- und Modern-Läufe |
| B2 | SH, synth-A-Hohlraum (Paper III) | 5 von 79 Wahrheitsschüssen |
| B3 | PSV, Marmousi-II OBC, 5+2 Schüsse, 2,5/3,5/5 Hz, elastisch und mit Q | Banken, Legacy- und private Modern-Ergebnisse |
| B4 | RTM/LSRTM/RFWI, Marmousi-II, 100 Schüsse (A2-Kampagne), Variante ohne tiefe Frequenzen und mit kurzen Offsets | A2-Daten und Rohbild |
| B5 | Felddaten: ein SH-Profil der Ownerin. Für PSV/RTM bleibt Marmousi-II (B3/B4) der Referenzfall; ein PSV-Felddatensatz ist vorerst nicht vorgesehen (E-4). | Profil von der Ownerin zu benennen |

## 5. Plan

Zeitangaben sind Schätzungen aus dem bisherigen Tempo, keine Zusagen.

### Phase 0 — Konsolidieren (etwa eine Woche)

| Paket | Inhalt |
|---|---|
| K-1 | **Erledigt 21.11.:** PR #111 (CUDA-Link, −0.0-Gradienten), #112 (SH-L-BFGS) und #113 (Ledger) sind gemergt. |
| K-2 | **Nicht gelandet, geparkt:** T1a (privater FD8-Born-Pfad) wurde als PR #115 vorgelegt; die Hosted-CI war rot, der PR ist ohne Merge geschlossen, der Zweig `claude/t1a-fd8-born-private` bleibt erhalten. Die zwei roten Tests (`test_m63c_distributed_objective_directional_fd`, Fall `b2_corner-2-2-3-physical-4-0-0`; `test_psv_viscoelastic_rheology::test_m43_selected_mpi_reproducibility[SV-1-2]`) scheitern beim Auswerten der MPI-Programmausgabe. Ursache offen; ein Zusammenhang mit dem Kandidaten ist weder belegt noch ausgeschlossen. Die Case-3-Qualifikation des eingefrorenen Kandidaten bleibt bestehen. Die Taylor8-Linie ruht (Abschnitt 7). |
| K-3 | RFWI-C-FD4 mit Statusnotiz schließen: Referenz INCOMPLETE, geparkt. Die zwei Quellenbefunde (W², 0/0) in den Prototyp übernehmen. |
| K-4 | Forschungsprioritäten ins Repo übernehmen (`docs/research_priorities.md`). Das Forschungsjournal bleibt auf Owner-Entscheidung privat. Ledger auf den Stand der gemergten PRs bringen. |
| K-5 | Fertige Performance-Prototypen integrieren: Allokator-Versatz, visko-PSV-Zeilenschleife (beide bytegleich), SH-Checkpoint/Replay. |
| K-6 | Lesender Codevergleich PHYSICAL01/Holberg8 gegen den Programmkern, **jetzt** und nicht erst vor P-4, damit P-1 bis P-3 nicht auf einer ungeprüften Kernwahl aufbauen. Verglichen werden Felder und Staggering, Update-, Quell- und Empfängerzeitpunkte, FD-Koeffizienten, elastische und viskoelastische Zustandsgrößen samt Q-Abbildung, Material- und Dichtemittelung, A2-Kontakt, CPML-Profile und -Speicher, Halos, Forward- und Adjoint-Tape, FP32-/GPU-Operationen, Parameter- und Modellformat. Ergebnis je Zeile: identisch, kompatibel nach Adapter, Physik-Differenz oder unbekannt; dazu eine Entscheidungsvorlage. Kein Rückport von Adjungierten aus einer anderen Diskretisierung ohne eigenen Nachweis. |

### Strang SH — vom Pilot zu Felddaten

| Paket | Fähigkeit im Programm | Nachweis |
|---|---|---|
| S-1 | L-BFGS und Vs-zuerst-Hierarchie (PR #112) | B1: Pilot r6, LVZ in der Tiefe |
| S-2 | Frequenzstufen (`TIME_FILT`), Zeitfenster, Offset-Mute im exakten Pfad; Filter und Fenster als lineare Operatoren mit ihrer Transponierten im Adjoint-Quellterm | B1 mehrstufig, B2 |
| S-3 | Quellsignalschätzung (`INV_STF`) je Schuss und Stufe; alternativ quellunabhängiger Misfit | B2 mit Ricker-Wahrheit und Spike-Start |
| S-4 | Spurnormierung, robuste Misfits (normierte Kreuzkorrelation, Envelope) | B2, dann B5 |
| S-5 | Erste Felddaten-SH-FWI | B5 |

### Strang PSV — exakte visko-FWI als Köhn-Workflow

| Paket | Fähigkeit im Programm | Nachweis |
|---|---|---|
| P-1 | Mehrschuss und Frequenzstufen im exakten visko-PSV-Pfad, CPU-MPI und CUDA | B3 elastisch und mit Q gegen Legacy |
| P-2 | L-BFGS mit Klassenskalierung und Beleuchtungs-Vorkonditionierer (Bausteine aus dem SH-Strang wiederverwenden) | B3: Modellfehler nach Tiefe |
| P-3 | Freie Oberfläche und STF im exakten Pfad | Landmodell, B3 mit freier Oberfläche |
| P-4 | Fluid/Festkörper-Kontakt (A2-Abschluss) aus PHYSICAL01 in den Programmkern | B3: Tiefenwirkung Vs wie im privaten A2-Vergleich |
| P-5 | FD-Ordnung 8 als Option des exakten Pfads, wenn B3 bei 5 Hz und darüber Dispersion oder Kosten als Engpass zeigt | B3 höhere Stufen, Kosten je Wellenlänge |

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
| A-3 | Elastische RFWI v1 auf M9 mit fester Dichte als **variable Projektion** (E-8): das Bild ist die LSRTM-Lösung aus A-2 zum jeweiligen Hintergrund, die Zielfunktion hängt nur vom Hintergrund ab, der reduzierte Gradient enthält den impliziten Term (Abhängigkeit des Bildes vom Hintergrund); Hintergrund-Update mit L-BFGS. Die alternierende Form (ein Hintergrundschritt bei festem Bild) läuft als billigere Vergleichsvariante mit. | Reduzierter Gradient gegen zentrale Differenzen; **Nullkontrolle** als festes Kriterium (derselbe Lauf mit wahrem Hintergrund gleich Startmodell: das Ergebnis muss klar über diesem Drift liegen); B4-Variante ohne tiefe Frequenzen, Start mit falschem Hintergrund; Vergleich mit konventioneller FWI |
| A-4 | SH-Born-Tangente im SH-Kern und SH-RFWI mit demselben Treiber | Reflektor+Linse-Test aus dem Prototyp, dann LVZ und Hohlraum |
| A-5 | SH-RTM/RFWI auf dem Feldprofil | B5 |

**Sandbox vor A-3 (RFWI-SANDBOX-01, 21.11.2026, EXPLORATORY).** Der
Python-SH-Prototyp hat den Reflektor+Linse-Fall als Inversion gerechnet
(73×81 Knoten, eine Geometrie, rauschfrei, gleicher Löser für Daten und
Inversion). Befund: Auf reinen Reflexionsdaten liegt nur die variable
Projektion klar über ihrer Nullkontrolle (Laufzeitfehler des Hintergrunds
0,273 → 0,037 ms, Nullkontrolle 0,015 ms). Die alternierende Form erreicht
0,055 ms bei einer Nullkontrolle von 0,054 ms. Die konventionelle FWI senkt den
Misfit um drei Größenordnungen, ohne den Hintergrund zu verbessern. Der
reduzierte Gradient stimmt mit zentralen Differenzen auf 10⁻⁹ überein, ohne
impliziten Term liegt er 40 % daneben. Folgerungen für A-3:

1. Variable Projektion mit implizitem Term ist die Zielformulierung (E-8).
2. Die Nullkontrolle ist festes Prüfkriterium.
3. Reflexionen legen die Laufzeit fest, nicht die Tiefenverteilung der
   Geschwindigkeit. Regularisierung oder Skalentrennung zwischen Bild und
   Hintergrund ist vor dem Bau von A-3 festzulegen.

Vorbehalt: In der Sandbox liefen die beiden Formulierungen mit verschiedenen
Optimierern. Ein Wiederholungslauf mit gleichem Optimierer ist beauftragt
(RFWI-SANDBOX-01b); sein Ergebnis wird hier nachgetragen.

### Integrationsstände

| Stand | Inhalt | Schätzung |
|---|---|---|
| I-1 | Phase 0, S-1, S-2, P-1, P-2. Neues Cluster-Testpaket mit Parameterdateien für B1–B3. | 3–4 Wochen |
| I-2 | S-3, S-4, P-3, A-1, A-2. | weitere 4–5 Wochen |
| I-3 | P-4, A-3, A-4, S-5. | weitere 5–7 Wochen |
| I-4 | P-5, A-5, danach Anisotropie (VTI vor TTI). | offen |

## 6. Forschungsaspekte mit Aussicht

Jeweils an ein Paket gebunden, damit sie im Programm landen.

- **Exakter visko-Adjoint als Alleinstellung** (P-1). Qp/Qs-Inversion mit
  physikalischem Q steht mit P-1 zur Verfügung. Q-kompensierte RTM und LSRTM
  wären mit einem exakten visko-Adjoint stabil formulierbar, ohne die
  instabile Amplitudenverstärkung üblicher Q-RTM; sie setzen aber einen
  viskoelastischen Born-Pfad voraus, den es noch nicht gibt (nach I-3).
- **A2-Fluid/Festkörper-Abschluss** (P-4). Der private Befund (tiefer Vs-Fehler
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
  Wiederaufnahme nur über P-5: als FD8-Option der Programmkerne mit wählbaren
  Koeffizienten (Taylor oder Holberg), wenn B3/B4 Dispersion oder Kosten als
  Engpass zeigen. Der T1a-Code für J/Jᵀ wird dann wiederverwendet; die
  Einschließungskette wird nicht fortgesetzt.
- Intervall-Einschließungen und Referenzbanken für RFWI-C-FD4.
- K1-Nullmoden-Analyse, sofern P-4 den A2-Abschluss übernimmt.
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
| E-3 | Taylor8-Linie nach T1a ruhen lassen? | Ja. Wiederaufnahme nur über P-5 (Abschnitt 7). |
| E-4 | Welche Felddaten für B5? | SH-Profil der Ownerin. Für PSV bleibt Marmousi-II der Referenzfall. |
| E-5 | Leitplanken 2–4 als Arbeitsregel in `AGENTS.md` übernehmen? | Ja. |
| E-6 | Reihenfolge RFWI: PSV auf M9 zuerst, SH danach (A-3 vor A-4)? | Ja. |
| E-7 | Filterstelle für Frequenzstufen in P-1? | Köhns Semantik: Filter auf Quellsignal und beobachtete Daten (Strang PSV). |
| E-8 | Formulierung der RFWI in A-3? | Variable Projektion mit implizitem Term statt der am 10.11. gewählten alternierenden Form, auf Grundlage von RFWI-SANDBOX-01; Wiederholung mit gleichem Optimierer beauftragt. |

## 10. Externes Audit vom 21.11.2026

Ein externes Architektur- und Wissenschaftsaudit (Urteil: CONDITIONAL GO) hat
die Richtung bestätigt und Präzisierungen empfohlen. Die Ownerin hat sie als
Empfehlung behandelt und Folgendes übernommen:

- Kernwahl E-1 vorläufig bis K-6; K-6 vorgezogen (Leitplanke 5, Phase 0).
- Filterstelle vorab von der Ownerin festgelegt statt vom Worker gewählt (E-7).
  Abweichend vom Audit-Vorschlag (Filter im Datenraum mit Transponierter) gilt
  für P-1 Köhns Semantik; die Datenraum-Formel gilt für Fenster und Maske.
- Integrations-Gate risikoadaptiv ergänzt, auf einen Referenzfall je Paket
  begrenzt (Leitplanke 4).
- Strang Abbildung ausdrücklich elastisch abgegrenzt.
- Wirkung der historischen HOLDs je Befund benannt (Abschnitt 7).
- Rolle „Project Coordination" in `AGENTS.md`.

Nächster vorgesehener Audit-Zeitpunkt: wenn der Kandidatenbericht zu P-1 und
das Ergebnis des SH-Pilots r6 vorliegen, vor dem nächsten Content Lock.
