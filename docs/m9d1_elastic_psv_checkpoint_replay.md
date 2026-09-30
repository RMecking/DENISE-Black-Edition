# M9d1 elastic P/SV checkpoint/replay

M9d1 changes only the retained background storage used by the M9b-1 Born
operator. The FD4 forward map, tangent map `J`, reverse-coded transpose `J^T`,
receiver/source timing, material maps, harmonic-mu map, CPML recurrences and
raw Euclidean image definition are unchanged.

## Storage modes and API

New contexts retain the historical full strain trajectory by default. This is
the differential reference mode. Calling
`denise_elastic_psv_born_set_replay_segments(context, segments)` before
`prepare()` selects segmented checkpoint/replay. Counts below one fail; counts
above `NT` clamp to `NT`. Changing the policy after preparation fails.

`denise_elastic_psv_born_estimate_replay_storage()` computes the exact retained
complete logical retained size from the context's built CPML profiles before any
replay allocation. The same internal calculation drives allocation accounting,
the public estimate, diagnostics and production selection.
`denise_elastic_psv_born_storage_diagnostics()` reports the
selected schedule, retained storage, working-state sizes and forward-step
counts. `denise_elastic_psv_born_get_segment_bounds()` exposes the schedule for
verification. `denise_elastic_psv_born_checkpoint_roundtrip()` is a diagnostic
that captures a real state, restores it into clean storage, advances one full
timestep and compares every persistent value bitwise.

Production M9c requests `min(32, NT)` segments. It selects segmented mode only
when the complete pre-allocation estimate is strictly smaller than the historical
`4*NT*NX*NY*sizeof(float)` trajectory. The existing `trajectory_bytes` result
field keeps that historical meaning. New result fields report actual replay
storage and recomputation. Equality selects FULL, with no hysteresis or threshold.

The retained metric includes each checkpoint payload, the contiguous checkpoint
object array (`(S-1)*sizeof(struct elastic_checkpoint)`), both schedule arrays
(`S*sizeof(int)` each), and the reusable operand buffer. There is no separately
allocated checkpoint pointer table: payload pointers belong to the checkpoint
objects and are already included in their `sizeof`. No other replay-only heap
allocations remain after preparation.

Common context fields exist in both modes and are excluded from this comparison,
as are shared static model data, transient forward/tangent/adjoint working
state, stack variables and allocator-private bookkeeping. All figures are
logical program-requested bytes, not allocator slab overhead.

`checkpoint_payload_bytes` means one payload; `checkpoint_bytes` remains the
sum of payload allocations. `checkpoint_metadata_bytes`,
`checkpoint_pointer_bytes` (zero here) and `segment_schedule_bytes` report the
remaining components. `retained_replay_bytes` / M9c `peak_replay_storage_bytes`
include all these components plus operands. Before preparation, diagnostics
count only already allocated schedule arrays; the estimate is prospective.

## Timing and persistent state

One background timestep remains:

1. update velocity;
2. sample receivers;
3. compute CPML-corrected `VXX`, `VYX`, `VXY`, `VYY`;
4. expose the four operands;
5. update stress;
6. inject the source into `SXX` and `SYY`;
7. update the CPML peak diagnostic.

A checkpoint labelled `k` is captured after step 7. Restore therefore resumes
at `k+1`. Segment zero starts from exact zero state.

Each checkpoint stores the five persistent float32 fields `VX`, `VY`, `SXX`,
`SYY`, `SXY`, followed by the active values of all eight CPML memories `PSXX`,
`PSXYY`, `PSXYX`, `PSYY`, `PVXX`, `PVYX`, `PVXY`, `PVYY`. Scratch strains are
regenerated. Static model maps and CPML coefficients remain context-owned and
are never duplicated per checkpoint.

CPML values are packed only at coordinates where the matching profile's
float32 recurrence coefficient `a` is nonzero. All omitted memory values are
proved exactly zero during capture and reconstructed as zero during restore.
For the non-staggered profiles this is `2*FW` coordinates. The frozen half-grid
profile has one additional active cell at the right boundary because its
coordinate is shifted by `0.5*DH`; retaining that cell is required for bitwise
replay. With `FW=0`, no CPML value is stored.

Checkpoint metadata contains layout version, timestep, `NX`, `NY`, `FW`, the
effective `FDORDER=4` layout and CPML-enabled state. Restore rejects a mismatch.

## Schedule and replay

For `S` segments, the deterministic zero-based schedule is

```text
start[s] = floor(s * NT / S)
end[s]   = floor((s + 1) * NT / S)
```

Checkpoint `s` is captured after `end[s]-1` for every segment except the last,
so the checkpoint count is `S-1`. One reusable operand buffer holds
`4*max_segment_length*NX*NY` float32 values.

`J` replays segments chronologically while one tangent state continues across
all boundaries. `J^T` allocates one adjoint state, visits segments in reverse,
replays each background segment from its preceding checkpoint, and applies the
unchanged reverse graph in descending timestep order. `copy_strain()` replays
the segment containing the requested timestep.

Preparation performs `NT` initial forward steps. Each segmented `J` or `J^T`
performs exactly `NT` replayed forward steps. This one-level schedule does not
implement Revolve or retain an all-segment cache.

## Canonical active-CPML footprint

For `NX=64`, `NY=56`, `NT=700`, `FW=8`, and 32 segments:

```text
historical full strains       40,140,800 bytes
checkpoint payload               103,360 bytes
checkpoint count                      31
all checkpoints                3,204,160 bytes
checkpoint object array            1,488 bytes (31 * sizeof=48 on local ABI)
checkpoint pointer table               0 bytes
segment start/end arrays             256 bytes
maximum segment length                22 steps
segment operand buffer         1,261,568 bytes
complete retained replay       4,467,472 bytes
ratio to full trajectory        0.111295...
```

For `NX=NY=5`, `NT=41`, `FW=0`, `S=32`, full storage is 16,400 bytes.
Replay comprises 15,500 payload bytes (31 * 500), 1,488 object bytes,
256 schedule bytes and 800 operand bytes: 18,044 total. The former incomplete
16,300-byte estimate incorrectly chose replay; the complete estimate chooses
FULL. A 6-by-73, NT=41 case chooses SEGMENTED with exactly 8 bytes advantage
(287,320 versus 287,328). Synthetic equal-byte selection chooses FULL.

Forced S=1 on the 5-by-5 fixture retains 16,408 bytes (full operands plus an
8-byte schedule), and forced S=NT=41 retains 22,648 bytes. Both estimates
are larger than full storage. The forced modes remain available for testing.

H1 tests exercise the production arithmetic with near-SIZE_MAX inputs before
allocation, including object multiplication, schedule multiplication/addition,
operand/payload multiplication, total addition, full trajectory overflow and
invalid estimates. No pointer table exists; its hypothetical multiplication
is checked through the same checked-product primitive.

The transient forward working state is 258,048 bytes and the adjoint working
state is 430,080 bytes for this fixture. These are reported separately and are
not included in retained replay storage.

The M9d1 gates require bitwise equality between full and segmented operands,
Born data, lambda images and mu images for interior and active-CPML cases. The
frozen M9b-0 oracle ceilings remain `1e-5` for `J` and `6e-5` for `J^T`.

M9d1 does not add MPI, free-surface support, CUDA, disk checkpoints,
compression, Revolve, wave-mode decomposition or image conditioning.

## M9d2 distributed replay

The serial replay API and complete H1 accounting remain unchanged. M9d2 adds
owned-state checkpoints and local active CPML strips, regenerates ghosts on
restore, and selects replay only when every rank wins the complete byte
comparison. Equality or any rank loss selects FULL collectively. See
[the M9d2 contract](m9d2_elastic_psv_mpi.md) for the distributed ledger and gates.
