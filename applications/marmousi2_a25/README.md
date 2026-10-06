# Marmousi-II A2.5 publication candidate

This is an offline consolidation of accepted A2.5A image diagnostics and A2.5B
normal-operator evidence, not a new experiment or publication authorization.
Read [scientific interpretation](SCIENTIFIC_REPORT.md), [frozen claims](claims.json),
[operator contract](normal_operator_contract.json) and [evidence index](evidence.json).
The index is a lightweight set of exact external receipt hashes, not the campaign.
`h15_inventory.json` preserves 2129 relative file identities from the hash-bound
historical serial preparation, including all original H15 execution receipts.

## Architecture and identities

Publication/integration BASE: `6c348d490862277d6c2c5d017bf2e4029ce54169`.
A0/A1/A2 canonical merge: `54c69297a6c1d5fef94c396b49451ebaadcec2ab`.
Historical scientific Core: `f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d`.
The current dirty candidate has no separate commit SHA: its source hashes bind
new report receipts. Do not call the BASE SHA the new tooling SHA.

`tools/denise_case/image_*` holds reusable loading, transforms, conditioning,
reference evaluation and reporting. `normal_operator.py` preserves the accepted
pure PSF/tangent/transpose arithmetic; `normal_operator_analysis.py` preserves
ascending native reductions, aperture and frequency diagnostics.
`a25_publication.py` is the case-specific offline evidence adapter and report
orchestration, inside the existing Applications infrastructure.
Canonical JSON encoding is shared with `campaign.py`. A2.5A staged Vp/Vs/log
multiplications and A2.5B coefficient-first transpose intentionally keep their
respective accepted FP64 operation ordering; these are not interchangeable
bitwise implementations. No numerical algorithm was changed for cleanup.

The historical driver, C bridge, source construction and runtime supervision
remain external experiment-specific tooling. See
[experiment boundary](../../experiments/marmousi2_a25b/README.md).
They are not imported into the general Applications API.

## External layout and offline reproduction

Supply five explicit logical roots. Their actual filesystem locations are not
persisted in reports and need not match the original workstation:

| Argument / root | Required relative contents |
| --- | --- |
| `--evidence` / A25B | `H15/shots/001..100/<probe>/{receipt.json,h_native.npy}`, `H5/shots/...`, `H15/sums/<probe>/<group>/{h_native.npy,h_log.npy,metrics.json}`, H5 equivalent; exact identity/closure/source/build receipts listed in `evidence.json` |
| `--a25a-evidence` / A25A | accepted `receipt.json` and every relative output it binds (597 output files; 598 including receipt) |
| `--core` / CORE | frozen historical checkout; 667 Core/oracle source hashes authenticated from the numerical identity record |
| `--runs` / RUNS | accepted `shot_001..100` inputs/provenance/metadata/images and `campaign_100/{summary.json,aggregate/receipt.json}` |
| `--models` / MODELS | original `smooth2.{vp,vs,rho}` and evaluation-only `true.{vp,vs,rho}` |

From the repository root, with Python/NumPy already available (no network):

```text
python -m tools.denise_case a25-publication --evidence <A25B> --a25a-evidence <A25A> --core <CORE> --runs <RUNS> --models <MODELS> --output <fresh-external-output>
python -B -m pytest -p no:cacheprovider tests/applications
```

Flow: authenticate external receipts and their products; authenticate all
accepted raw images/models/inputs; reconstruct 88 ascending native/log group
pairs from 1100 saved native H fields; reproduce exact predeclared PSF metrics,
aperture diagnostics and H15/H5 comparisons; regenerate A2.5A diagnostics and
render the consolidated HTML/JSON; rehash protected inputs afterwards.
No DENISE, J/JT/HVP, MPI, build, CUDA or FWI is launched.

Missing/mismatched artifacts fail, with no substitute or solver fallback. An
interrupted output is preserved; select a fresh directory on restart. Output
may neither overlap evidence nor the candidate repository. Existing products
are never overwritten. A2.5A alone also supports explicit logical root overrides:

```text
python -m tools.denise_case image-condition --case applications/marmousi2_a25a/case.json --runs <RUNS> --models <MODELS> --output <fresh-external-output>
```

Historical A2.5A receipt/content ID remain authoritative references. Relocation,
the new tooling and the updated cross-reference change the new derivation ID,
not the accepted campaign identity. Historical derived arrays are compared
bytewise to regenerated arrays during this publication gate. Same environment
is needed for bitwise FFT/report reproduction; no cross-platform bitwise promise.
Metrics consume raw/derived arrays, never rendered or normalized pixels.

Original Jz hashes and Euclidean norm attestations survive in sealed per-shot
receipts; complete campaign Jz arrays are not retained in these product
directories. The available later H5 Jz arrays are checked too. Missing historical
Jz arrays are explicitly attested only, never silently reconstructed or newly
computed. This does not weaken the mandatory saved H-field/metadata gate.

## Excluded from Git

All 100 raw JT shot products, raw aggregates, 900 H15/200 H5 fields, source/data
arrays, full H columns, any Jz arrays, large PNG/SVG collections, process dumps,
build binaries and redundant reproduction outputs remain external. Frozen
contracts, exact receipt/content hashes, source genealogy, compact full-precision
comparison metrics, reusable code and solver-free fixtures/tests belong in Git.
Historical absolute paths in external execution receipts are not rewritten;
they are read for provenance only, never copied into permanent reports.

## A3.0 handover

Next scientific step requires separate authority: **A3.0 — Controlled Marine FWI
Benchmark**, simple controlled geometry before Marmousi-II FWI. Carry forward
restricted physical-water parameter space, coupling/nonuniform sensitivity,
explicit gradient scaling/preconditioning diagnostics, frequency/discretization
awareness and the rejection of naive diagonal RTM normalization. No full A3
design or FWI run is part of this candidate.
