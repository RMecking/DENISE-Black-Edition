# M8d MPI overlap evaluation and no-go

## Scope and canonical state

This record closes M8d at
`origin/modernization@360771ff86520af40ef0d146c426d1764023573b`. The
canonical merge contains M8d-1A distributed exact viscoelastic gradients,
M8d-1B distributed exact-visco Active-FWI, the M8d-2A frozen blocking MPI
baseline, and M8d-2B1a region-safe P/SV stress kernels. M8d-2A and M8d-2B1a
are closed successfully; the B1a canonical merge is
`360771ff86520af40ef0d146c426d1764023573b`.

M8d-2B1a is the merged prerequisite for safe rectangular stress updates: it
separated the requested stress-update rectangle from full local physical-domain
dimensions while preserving CPML/GSLS semantics. It does not implement
asynchronous MPI, V overlap, or S overlap.

The experimental M8d-2B1b/B1c asynchronous V-exchange production
implementation was not committed, was not merged, and is not part of the
supported canonical `modernization` production code. Its raw benchmark
artifacts are also deliberately not canonical. This document preserves the
engineering decision, not experimental source or raw JSON artifacts.

## Correctness result

The experimental B1b V-overlap path was scientifically correct. Blocking and
async active V halos were byte-identical. Elastic and `L=1` OFF/ON traces were
byte-identical for 2x1, 1x2, and 2x2 decompositions at FD4 and FD8. The M8b
fast path, M8c-1, M8c-2, M8d-1A, and M8d-1B gates passed; exact operand
mismatches were `[0,0,0,0,0,0]` and receiver mismatches were zero. The
free-surface, periodic, and undersized fallbacks also passed.

The no-go is therefore performance-driven, not a correctness failure.

## Performance evidence on the tested OpenMPI/WSL runtime

The same-binary paired B1b result was neutral/inconclusive:

| Case | OFF median | ON median | Paired speedup | Runtime reduction | ON faster |
| --- | ---: | ---: | ---: | ---: | ---: |
| `L=1`, 2x2 | 0.500402355 s | 0.510663819 s | 0.995104x | -0.4920% | 3/7 |
| Elastic, 2x2 | 0.278101692 s | 0.288465881 s | 0.947497x | -5.5412% | 2/7 |

The `L=1` final exposed V wait was 0.063105199 s. These measurements did not
demonstrate a reproducible end-to-end benefit.

### B1c progress-pumping experiment

`MPI_Testall` polling materially changed MPI progress for `L=1`, 2x2:

| Polling | Final V wait | Polling overhead | Completion before final wait |
| --- | ---: | ---: | ---: |
| No poll / Bands=1 | about 0.123585 s | 0 | not observed by polling |
| Bands=2 | about 0.023179 s | about 1.3–2.3 ms | n/a |
| Bands=4 | about 0.002307 s | about 1.3–2.3 ms | 22,700 / 27,500 rank-timestep observations (about 82.5%) |
| Bands=8 | about 0.000337 s | about 1.3–2.3 ms | 26,438 / 27,500 rank-timestep observations (about 96.1%) |

Thus explicit polling clearly drove OpenMPI progress on the tested WSL
runtime. It does not establish behavior for another MPI implementation,
native Linux, or an HPC system.

### Bands=2 confirmation

Bands=2 was selected as the lowest useful polling frequency by the screening
stability/overhead tradeoff. The independent paired confirmation did not
reproduce a runtime win:

| Case | OFF median | Bands=2 ON median | Paired speedup | Runtime reduction | ON faster |
| --- | ---: | ---: | ---: | ---: |
| `L=1`, 2x2 | 0.491805492 s | 0.509304895 s | 0.965641x | -3.5582% | 2/7 |
| Elastic, 2x2 | 0.282701036 s | 0.295462134 s | 1.017661x | +1.7355% | 5/7 |

For the primary `L=1` case, final wait was 0.014218 s and polling overhead
was 0.001326 s. The Elastic row remains neutral/inconclusive because the
independent medians and paired behavior disagree; it is not a proven speedup.

Reduced exposed wait did not translate reproducibly to reduced timestep or
runtime cost. Plausible, individually unproven contributors are pack/unpack
cost, MPI request management, region/kernel splitting, rank imbalance,
single-host/WSL scheduling variability, and MPI runtime characteristics.

## Decision and deferred work

**M8d-2B1b: NO-GO on current MPI runtime.** Do not merge B1b/B1c production
code, enable nonblocking V overlap by default, or proceed to M8d-2B2
S-overlap on this runtime. Retain canonical blocking V/S exchange and the
merged B1a region-safe kernels.

M8d-2B2 is **deferred**, not implemented or failed. Ordinary nonblocking V
overlap did not show reproducible benefit, while S overlap has a smaller safe
same-timestep compute window and greater scheduling complexity.

The work may be revisited on a materially different environment, especially a
native Linux HPC node, alternative MPI implementation, runtime with
demonstrated asynchronous progress, or a distributed/GPU architecture with
natural compute/transfer overlap. A revisit should reuse B1a region-safe
kernels, the M8d-2A paired/interleaved same-binary methodology, and
exposed-wait instrumentation concepts. The unmerged B1b/B1c source is not
part of the supported codebase.

## Milestone closure

**M8d CLOSED.** Distributed exact-visco P/SV gradient and physical-Q
Active-FWI were delivered; checkpoint/recompute scalability was delivered in
M8c; blocking MPI performance was characterized; region-safe stress kernels
were delivered; and nonblocking overlap was investigated and rejected/deferred
on the tested runtime. This is a completed engineering decision, not an
unresolved blocker.

The next performance/compute milestone is **M8e — GPU / accelerator
feasibility and design**. Its detailed design is outside this record.
