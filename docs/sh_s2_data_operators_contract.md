# SH-S2 contract: frequency stages, time window and offset mute in the exact viscoelastic SH FWI

Status: interface and behaviour contract of roadmap package S-2 (`docs/roadmap.md`, strand SH).
- Owner decisions: 2026-11-21. The objective follows E-7. Legacy semantics were checked before implementation. The final decisions are: legacy stage filter, `TIMEWIN` 1 only, exact window gradient, and `TWIN-` takes effect.
- Base: `modernization` at `69da72d`.

## 1. Capability

The exact viscoelastic SH FWI (MODE 1, PHYSICS 5, physical Q) accepts the following workflow controls, which it previously rejected:

- `TIME_FILT` 0/1, with `FC_low`, `FC_high` and `ORDER`;
- `TIMEWIN` 0/1, with `GAMMA`, `TWIN-`, `TWIN+` and the first-arrival pick files `PICKS_FILE<shot>.dat`;
- `OFFSET_MUTE` 0/1/2, with `OFFSETC`.

Each value is set per workflow stage (one workflow row per stage). There are no new parameter records. With all three controls at 0, the driver behaves bitwise as at the base.

`TIMEWIN` 2 (the legacy constant window) is rejected (fail closed). A constant window is a pick file with equal times.

## 2. Objective

For every stage, Phi(m) = 1/2 sum_s || M_s T_s ( F_s(m; B w_s) - B d_s ) ||^2. The sum runs over the samples k = 2..NT, as at the base. The operators act in the order filter, window, mask.

- **B (stage filter, Koehn semantics).** B is the legacy `apply_tdfilt`, called unchanged on the source signal `w_s` and on the global observed traces `d_s`. These are the same calls and the same filter as in legacy and in the PSV strand.
  - Synthetics and residual are not filtered, so no transpose of B is needed: the existing exact adjoint gives the exact gradient of Phi.
  - B is a causal, single-pass `cseife` Butterworth filter: low pass at t0 = 1/`FC_high`, then high pass at t0 = 1/`FC_low` if `FC_low` > 0, with order `ORDER`.
  - The source signal is filtered after the `QUELLART` = 6 spike filter.
  - **Known legacy property:** `apply_tdfilt` leaves the last sample of every trace unfiltered, in the source signal and in the observed data. B is therefore not an exact LTI operator at k = NT, and F(m; B w) = B F(m; w) holds on samples 1..NT−1 only.
    - At k = NT the residual would compare a filtered synthetic sample with an unfiltered observed sample. On the 32x32 test fixture with a 12 Hz low pass, that single sample carried about 80 % of Phi (J 4.07e-3 instead of 7.16e-4).
    - **Owner decision 2026-11-21:** with `TIME_FILT` = 1, the mask M has weight 0 at k = NT for every trace. That sample enters neither the objective nor the adjoint source. With noise-free data this gives Phi(m_true) = 0, and the gradient stays exact.
    - Legacy's internal J still contains the artefact. Comparisons with legacy evaluate the misfit of both codes without the last sample, from the seismograms, instead of taking legacy's own J.
    - A fix of the filter offset for all paths is a separate, later bugfix package.
- **T (time window).** T is diagonal. Sample k has time t_k = k·DT (legacy `time_window` convention).
  - p is the first-arrival pick of the receiver, from `PICKS_FILE<shot>.dat`: one value per global receiver, in seconds.
  - t_lo = p − `TWIN-` and t_hi = p + `TWIN+`, where `TWIN-` and `TWIN+` are lengths in seconds.
  - Weights: w = 1 on [t_lo, t_hi]; w = exp(−`GAMMA`·(t − t_hi)²) after it; w = exp(−`GAMMA`·(t − t_lo)²) before it.
  - If t_lo ≤ 0, no sample lies before t_lo and the window is one-sided, as in legacy. There is no separate "off" value.
  - The front flank was confirmed by the Owner on 2026-11-21 as the form of an earlier Owner modification of the legacy window (2021): the same Gaussian with `GAMMA` as after the window. In legacy and in DENISE-SH, `TWIN-` is read but has no effect.
- **M (data mask).** M is diagonal and 0/1: per trace for the offset mute, and at sample k = NT for every trace when `TIME_FILT` = 1. The offset |(recpos_x·DH, recpos_y·DH) − (srcpos_x, srcpos_y)| is computed in binary32, as in legacy `calc_res`, so that ties at `OFFSETC` are decided identically.
  - Mode 1 removes traces with offset ≥ `OFFSETC`; mode 2 removes traces with offset ≤ `OFFSETC`.
  - The source position is that of source column `ishot`.
- **Adjoint source.** (M T)^T (M T) r = w² ⊙ r. The objective uses (w r)², with w = w_T·w_M.

### Deliberate deviations from legacy (genealogy, not oracle)

| Item | Legacy | Here |
| --- | --- | --- |
| Window in the adjoint | w·r, which does not match the legacy J = ½(w r)² | w²·r, the exact gradient |
| `TWIN-` | read, no effect (code commented out) | front flank as confirmed by the Owner: flat window between t_lo and t_hi, Gaussian with `GAMMA` before t_lo |
| `TIMEWIN` 2 | constant window | rejected; use a pick file with equal times |
| Last sample with `TIME_FILT` = 1 | in J (unfiltered observed sample) | weight 0 in J and in the adjoint source |

## 3. Validation (collective, before the first iteration of each stage)

The driver aborts collectively if any of the following holds:

- `TIME_FILT` ∉ {0, 1}, `TIMEWIN` ∉ {0, 1} or `OFFSET_MUTE` ∉ {0, 1, 2};
- with `TIME_FILT` = 1: `ORDER` < 1, or not 0 < `FC_high` < 1/(2·DT), or 0 < `FC_low` but not `FC_low` < `FC_high`;
- with `TIMEWIN` = 1: `GAMMA`, `TWIN-` or `TWIN+` negative or non-finite;
- with `OFFSET_MUTE` ≠ 0: `OFFSETC` ≤ 0 or non-finite;
- with `TIMEWIN` = 1 and `READREC` = 1: a pick file of some shot is missing or holds fewer finite values than there are receivers.

With `READREC` = 2 the receiver count is known only per shot. The pick-file check then happens at the first objective evaluation, as a collective abort inside iteration 1.

The multi-shot wrappers repeat the range checks. A per-shot failure is reported collectively and the wrapper returns an error, which the driver turns into a collective abort. As before S-2, the gradient outputs may then be partly accumulated.

Every stage with a control ≠ 0 prints on MYID 0 one line `Exact SH data operators: stage <n> TIME_FILT … OFFSETC …`. With `TIME_FILT` = 1, the misfit log has 11 columns, as in legacy.

## 4. Code

- `src/SH/visco_sh_exact_data_operators.c`: the pick reader and the window and mask weights. These functions are local, use no MPI and do not call `err()`.
- `src/SH/exact_sh_data_ops.h`: the per-shot glue (the `apply_tdfilt` calls and the weights) and the configuration check.
- Callers: both multi-shot wrappers (objective and objective/gradient, so the line-search trials are covered) and both shot functions (weighted objective and adjoint seed).

## 5. Evidence (integration gate, `AGENTS.md`)

`tests/test_sh_s2_data_operators.py`:

- **Unit:**
  - the documented last-sample property of `apply_tdfilt`;
  - that the exact path calls `apply_tdfilt` on source and global observed data before the mapping;
  - the weights against an independent Python formula, including binary32 offset ties, the pre-pick flank and the masked last sample;
  - the error codes, including `TIMEWIN` 2.
- **Driver, 32x32 fixture:**
  - Directional derivative of the raw gradient against fourth-order central differences of J, for every mode and for a combination, on 1 and 2 ranks, with the m63c tolerance 5e-3. The step sizes are eps 0.2 and 0.1 of a smooth direction that peaks at 2 % of Vs and rho and 10 % of Q.
  - The logged J against an independent offline J_ref = ½ Σ (w (s − d))². J_ref is computed from a MODE 0 forward run, `apply_tdfilt` and the reference weights. Tolerance: 2e-6 relative without the filter, 1e-4 with it; on 1 and 2 ranks.
  - F(m; B w) = B F(m; w) on samples 1..NT−1, using `QUELLART` = 3 signal files.
  - A multi-stage run with accepted steps and path equality: within a stage, J_base(k+1) equals the selected J(k). The run also checks reload consistency.
  - Configuration aborts.
- **Regression:** the existing suite passes unchanged, and with all controls at 0 the outputs are bitwise equal to the base.
