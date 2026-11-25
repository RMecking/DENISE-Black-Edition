# K-5 contract: checkpoint/replay of the exact viscoelastic SH forward history

Status: interface and behaviour contract of roadmap package K-5.
- Owner decisions: E-17 ("K-5 direkt nach S-3 einplanen") and the K-5 release of 2026-11-24 (F1–F4 plus the no-overflow requirement).
- Base: `modernization` at `83a7a329`.
- Origin: the exploratory prototype `sh_replay_prototype.patch` of 2026-11-12.
- Model: the segmented replay of the exact P-SV paths (`docs/m9d1_elastic_psv_checkpoint_replay.md`).

## 1. Capability

The exact viscoelastic SH gradient (`FWI_SH_visc.c`, MODE 1, PHYSICS 5) needs, per shot and process, the three forward records `qsum`, `strain_x` and `strain_y` of every time step. Until K-5 it kept all NT steps in memory, 3 (NX+1)(NY+1) NT floats; a B2-size shot (1249×593, 40 000 steps) needs 332 GB. With K-5 the history can be kept in segments instead.

- **Forward:** before the first step of every segment k ≥ 1, the forward state is stored as a checkpoint.
- **Reverse:** the reverse loop reads the records from the last step to the first. When it reaches a segment whose records are not in memory, it restores the checkpoint of that segment and recomputes the segment's forward steps.
- **Memory:** one segment of records plus S − 1 checkpoints, instead of NT records.
- **Cost:** (S − 1)/S of one additional forward per gradient; the last segment stays in memory from the forward run and is never recomputed.

**Parameter record 135 `SH_EXACT_SEGMENTS`** (integer, optional; when the record is absent, 0 applies):

| Value | Meaning |
| --- | --- |
| 0 | Full storage, as before. **Default.** |
| S > 0 | S segments, clipped to NT. S = 1 keeps the whole history as one segment: no checkpoint, no replay, the memory of full storage. |
| −1 | Automatic: S = round(√(NT · 3(NX+1)(NY+1) / c)), where c is the number of floats of one checkpoint. This minimises one segment of records plus S − 1 checkpoints. |
| < −1 | Rejected. |

**Checkpoint content.** vz, σxz and σyz with halos, r and q for the L mechanisms, and the four CPML memory strips (ψ_σxz,x, ψ_vz,x, ψ_σyz,y, ψ_vz,y). Every other forward array is a temporary that each step overwrites before it is read.

**Legacy arrays (F3).** The exact driver no longer allocates the legacy history arrays `forward_prop_*`:
- They are (4 + 2L) × NX·NY × NT floats, written by `sh_visc.c` in the gradient forward and never read by the exact path.
- Their length `NXNYI*NTDTINV` is an `int` and overflows above 2³¹ elements.
- `alloc_fwiSH_exact` allocates one element each, and `sh_visc.c` skips the legacy store whenever a trajectory is captured.
- The legacy SH FWI (`FWI_SH.c`, `FD_grad_SH.c`) is unchanged.

**No-overflow check (Owner requirement).**
- Before any allocation, the driver computes the history bytes of every process in double precision for the chosen mode. It sums them over the processes of a node (`MPI_COMM_TYPE_SHARED`).
- If the sum exceeds the node's physical memory, the run stops collectively. The message names `SH_EXACT_SEGMENTS` (record 135).
- MYID 0 prints `Exact SH history: SH_EXACT_SEGMENTS … segments … history_GiB_per_process … full_storage_GiB_per_process … node_GiB … physical_GiB … ranks_on_node …`.

**Diagnostics.**
- Per shot, a segmented run prints `Exact SH replay: segments … steps … max_segment … record_MB … checkpoint_MB … full_record_MB … replayed_steps … replay_seconds … verify … compared … mismatches …`.
- The environment variable `DENISE_SH_EXACT_SEGMENTS_VERIFY=1` (diagnostic, also kept in full) compares every recomputed record with the stored one.

## 2. Scientific contract

- **Bitwise, no tolerance.** The replay repeats the forward step exactly: same kernels and order, same source samples, same CPML, same MPI exchanges and free-surface operators, starting from a bitwise copy of the state. Hence the records, the objective, the raw gradient, the source energy (preconditioner), the L-BFGS path and all written models are bytewise equal to full storage, for every S. A difference is a defect, not a tolerance case.
- **Unaffected by construction:**
  - S-2: filter, window, mute and last sample act on source, data and adjoint source, not on the history.
  - S-3: the impulse forward has no trajectory (mode 2). The gradient forward with B v̂ is replayed like any source.
  - Reload, line search, L-BFGS: model space.
- **MPI:** the checkpoints are process-local. The segment boundaries depend only on NT and S, so every process recomputes the same segment at the same reverse step; the replay is collective and exchanges halos as the forward does.
- **Errors:** checkpoints are allocated with the trajectory, at its collectively checked allocation, so no allocation can fail during the run. A replay logic error ends the gradient with an error status on every process.
- **Configuration:** only the exact driver (`FWI_SH_visc.c`) reads record 135. After the memory check it sets the effective segment count once, through `sh_visc_use_exact_segments` (`sh_visc.c`), for every following exact gradient trajectory. The observable module (`visco_sh_material_observable.c`) uses no parameter globals; a caller that never sets a count, such as the unit harnesses, keeps full storage.
- **Limitation:** one segmented trajectory per process at a time, which is how the exact gradient runs, one shot after the other.

## 3. Evidence (integration gate; criteria fixed before candidate output)

**(a) Capability**, local:

- **C1:** with `SH_EXACT_SEGMENTS` ∈ {1, 2, 7, −1, NT}, all outputs of exact SH FWI runs are bytewise equal to full storage: misfit log, models, gradient, step and energy files, seismograms where written, `.stf` files. Covered:
  - FD order 2/4/8;
  - L = 1 and 4;
  - free surface, with FD 2, 4 and 8;
  - 1, 2 and 2×2 ranks;
  - two shots;
  - S-2 operators;
  - `INV_STF` = 1;
  - L-BFGS over two stages with reload.
- **C2:** a B1 single step (S-QF stage 1, 1 iteration, 4 ranks) with `SH_EXACT_SEGMENTS` = −1 is bytewise equal to full storage.
- **C3:** verify mode reports 0 mismatches, and as many comparisons as replayed records.
- **C4:** with `SH_EXACT_SEGMENTS` = 0, all outputs are bitwise equal to the base `83a7a329` (S-2 workflows, as in S-3 A4). The S-1/S-2/S-3/L-BFGS suites pass, and the legacy SH paths are unchanged.
- **C5:** memory and time on B1 (peak RSS and wall, segmented against full).
- **C6:** with `SH_EXACT_SEGMENTS` = 0, a configuration whose full history exceeds the node memory stops before iteration 1 with the record-135 message. Record values < −1 are rejected.
- **C7:** the complete Hosted CI selection passes locally in a copy with `.git` before the lock (non-physics and physics, as `.github/workflows/verification.yml`).

**(b) Benchmark.** One B2-size single-shot gradient on the project's 30 GB test workstation (homogeneous 1249×593 model, 40 000 steps, L = 4, 4 ranks, `SH_EXACT_SEGMENTS` = −1):
- peak RSS ≤ 12 GiB;
- wall time measured;
- the history line reports the expected segment count and memory.

K-5 changes neither operator, boundary nor material, so there is no physical reference case. B2 itself is a follow-up package.
