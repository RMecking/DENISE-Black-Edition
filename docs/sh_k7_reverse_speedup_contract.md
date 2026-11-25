# K-7 contract: bitwise speed-up of the exact viscoelastic SH reverse step

Status: criteria contract of roadmap package K-7, fixed before any candidate output.
- Owner decisions of 2026-11-25: performance target ≤ 20 min per iteration; ±0 shortcuts only if every output stays bytewise equal; a host profile run before the implementation; no OpenMP.
- Base: `modernization` at `b24161a1` (after K-5).

## 1. Capability

The exact viscoelastic SH gradient (`FWI_SH_visc.c`, MODE 1, PHYSICS 5) becomes faster. Its results are unchanged. There is no new switch and no new parameter record.

- The reverse step is the full-state adjoint with the material VJP, per time step and shot.
- On the B2-size single shot of K-5 it costs about 17 of the about 22 forward equivalents of one iteration.
- Shown on the benchmark ladder at B1 (single step, 4 ranks, locally) and at B2 size (single shot, 4 ranks, on the project's 30 GB test workstation).

## 2. Scientific contract

**Bitwise, no tolerance, for every output:**
- misfit log, models, gradients, step, energy and preconditioner files, `.stf` files;
- the `Exact SH` lines on stdout, except measured times such as `replay_seconds`.

A difference is a defect, not a tolerance case.

**Allowed changes.** The same floating-point operations in the same order and precision (`double` in the reverse step); only ballast is removed:
- per-step allocations and full-field buffers whose only effect is a copy, such as the receiver and source transposes;
- work whose result is discarded, such as material branches evaluated with zero inputs and then dropped;
- time-invariant terms computed once per shot instead of once per step and cell (GSLS derivatives, CPML selection), with identical expressions and precision;
- checks on time-invariant data (configuration, material, coefficients, CPML profiles) moved from the cell loop to once per shot.

**Build.** Bitwise equality is claimed for the repository's build flags (`src/Makefile`: `-O3` without `-march`, hence no fused multiply-add). A build that allows FMA contraction may contract base and candidate differently. The ±0 argument (dense patch scatter replaced by its cross contributions plus one `+= 0.0f` pass) also assumes IEEE round-to-nearest without flush-to-zero or denormals-are-zero; builds that set them (for example Intel `-ftz` or `-fast`, as in the commented-out lines of `src/Makefile`) are not covered by the bitwise claim.

**Zeros.** Dropping an addition of +0.0 can leave a −0.0 that used to become +0.0. Such a change is allowed only if the bitwise tests of §4 confirm it over all fixtures; otherwise the addition stays.

**Errors.**
- Every input that made a run fail before still makes it fail, with an error status on every process, before any output is written.
- Checks on time-dependent values (adjoint fields, records) stay per value.
- Only the phase named in the error message may change.

**Not touched:**
- forward propagation, K-5 checkpoint/replay, S-2, S-3, L-BFGS, legacy SH and PSV;
- the public helpers in `include/fd.h`: signatures and results unchanged;
- the unit harnesses and their source-text contracts.

**Interface addition.** `struct visco_sh_material_adjoint_step_context` gets two optional fields; zero keeps the previous behaviour:
- `plan`: the time-invariant GSLS derivatives of every cell, computed once per shot by `visco_sh_reverse_material_plan_create` with the same helper, `visco_sh_gsls_local_derivatives`, as before;
- `accumulate`: the step adds its material output to `native_output` instead of overwriting it. The shot sum then receives the same double additions in the same order as through the former per-step fields.

**Not in this package**, named as a later option needing an Owner decision:
- non-bitwise kernels (gather instead of scatter, `float`, OpenMP);
- shot distribution over MPI colours (K-8).

## 3. Scope

- `src/SH/visco_sh_full_state_adjoint_step.c`, `src/SH/visco_sh_reverse_time_material_gradient.c`;
- the local `*_adjoint_point` and GSLS helpers only where needed, with unchanged public behaviour;
- tests, this contract. The step oracle `tests/test_sh_k7_reverse_step.py` compares the production step with a verbatim copy of the base step (git blob `a13d88a0`) bit by bit; see A4.

Profiling timers stay in a sandbox build and are not part of the candidate.

## 4. Criteria (fixed before candidate output)

**(a) Correctness**, base `b24161a1` binary against candidate binary:

- **A1, fixture matrix:** a two-binary comparison script; every output file and every `Exact SH` line bytewise equal. Covered:
  - K-5 on (S ∈ {1, 7, −1}) and off;
  - S-2: filter, time window, offset mute;
  - S-3: `INV_STF` 1 and 3;
  - L-BFGS over two stages with reload;
  - 1, 2 and 2×2 ranks, FD 2/4/8, L = 1 and 4, free surface on and off.
- **A2:** the S-2 workflow comparison (A4 script of K-5) is bitwise equal.
- **A3:** the B1 single step (S-QF stage 1, 1 iteration, 4 ranks) is bitwise equal, 16 of 16 files and all `Exact SH` lines. Wall times are reported.
- **A4:** the SH suites pass unchanged (M6.3c unit and harness tests, S-1, S-2, S-3, L-BFGS, K-5); no existing test or tolerance is changed. The new step oracle passes: zero differing bits over several steps, with ±0 cells, duplicate receiver and source cells, CPML, free surface, 1/2/2×2 ranks, FD 2–12, every material variant, and both `-O3` and `-O2`; injected invalid inputs make base and candidate fail alike.
- **A5:** the full Hosted-CI selection runs locally on the exact lock tree, with tree IDs logged. Its failure set is identical to the base.
- **A6:** one independent review of the diff before the performance run.

**(b) Performance**, B2-size single shot on the test workstation under the conditions of the K-5 run (same recipe and inputs, 4 ranks, `SH_EXACT_SEGMENTS` −1, ITERMAX 1, STEPMAX 3):

- **P1:** the FWI iteration takes **at most 20 min** (1200 s) of wall time. Reference: 36.4 min in the K-5 run.
- **P2:** every output hash equals the K-5 run (`OUTPUTS.sha256`). A difference is a HOLD; it is diagnosed, for example as a toolchain change on the host, with no tolerance.
