# SH-S3 contract: source time function estimation in the exact viscoelastic SH FWI

Status: interface and behaviour contract of roadmap package S-3 (`docs/roadmap.md`, strand SH).
- Owner decisions: 2026-11-23 (F1–F9 of the S-3 package proposal). The objective is the variable projection of the SH-S2 objective over a source signal of limited, causal support.
- Base: `modernization` at `b5df20b`.
- Legacy semantics (`stf.c`, `stf_sh.c`) were read before implementation; they are genealogy, not an oracle (section 6).

## 1. Capability

The exact viscoelastic SH FWI (MODE 1, PHYSICS 5, physical Q) accepts the workflow control `INV_STF` (column 20), set per stage:

| `INV_STF` | Meaning |
| --- | --- |
| 0 | known source signal, as at the base |
| 1 | **variable projection (O2):** the source signal of every shot is re-estimated at every objective evaluation, including line-search trials |
| 3 | **fixed per stage (O1):** the source signal of every shot is estimated once, at the first objective evaluation of the stage, and then held fixed for that stage (comparison with the legacy workflow). With `OFFSETC_STF` > 0 the estimation uses only the traces with offset < `OFFSETC_STF` (section 2) |

- `INV_STF` 2 (legacy: STA/LTA window applied to the observed data only) is rejected (fail closed).
- With `INV_STF` = 0 in every stage, the driver behaves bitwise as at the base.
- The estimated signals are written per gradient evaluation (section 5).

## 2. Objective

Notation per shot s:
- receivers i;
- samples k = 1..NT with t_k = k·DT;
- the objective sums k = 2..NT, as at the base;
- ω_{s,i}(k) is the SH-S2 data weight (window T, mask M, and weight 0 at k = NT under `TIME_FILT` = 1). It is 1 when no data operator is on.
- B is the stage filter (legacy `apply_tdfilt`, E-7), acting on the source signal and on the observed data.

**Support (Owner decision F3, 2026-11-23).**
- The unknown is the source signal **before** the stage filter: v_s ∈ ℝ^{N_w}.
- It occupies samples k = 1..N_w, with v_s(k) = 0 for k > N_w.
- Sample 1 is the first injection step, i.e. the source time of the program. The support is therefore causal by construction.
- The forward source is B v_s, zero beyond N_w before filtering. B is causal (single-pass recursive Butterworth), so B v_s is causal as well and its filter tail is represented exactly. No acausal lead is needed or possible: the program has no sample before the source time.
- Both edges of the support are hard. There is no taper.
- N_w = `round(STF_LENGTH / DT)`, clipped to 1..NT. `STF_LENGTH` is parameter record 134, in seconds.
  - Default: `STF_LENGTH` < 0 means 0.5·`TIME`.
  - On B1: 0.25 s, against a 20 Hz Ricker that holds 99.999 % of its energy within 0.12 s.

**Objective.** Let g_{s,i} be the receiver traces of F_s(m; B e_1), with e_1 the unit impulse at sample 1. Then:
- s_{s,i}(v)(k) = Σ_{j=1}^{min(k,N_w)} v(j) g_{s,i}(k − j + 1). This is the synthetic of source B v, by linearity and time invariance of the discrete forward. It is exact on k = 1..NT−1 up to binary32 rounding; at k = NT the legacy filter's unfiltered last sample is masked by ω.
- D_s(v; m) = ½ Σ_i Σ_{k=2}^{NT} ( ω_{s,i}(k) ( s_{s,i}(v)(k) − (B d_{s,i})(k) ) )²
- Φ(m) = Σ_s Φ_s(m), with
  - O2: Φ_s(m) = min_v [ D_s(v; m) + ½ ε_s ‖v‖² ];
  - O1: Φ_s(m) = D_s(v̂_s; m) + ½ ε_s ‖v̂_s‖², where v̂_s is the minimiser at the stage's first evaluation.
  - O1 with `OFFSETC_STF` > 0 (Owner decision 2026-11-23, legacy practice: estimate the wavelet from the near offsets before dispersive surface waves appear): v̂_s minimises D_s^est(v) + ½ ε_s ‖v‖², where D_s^est uses the weights ω multiplied by 0 for every trace with offset ≥ `OFFSETC_STF`. The offset is |(recpos_x·DH, recpos_y·DH) − (srcpos_x, srcpos_y)| in metres, computed in binary32 as legacy `stf.c` and `calc_res`. Φ_s keeps every trace and the weights ω. Because v̂_s is fixed for the stage, the existing adjoint is the exact gradient of Φ.

**Regularisation (F3).**
- ε_s = `EPS_STF` · (1/N_w) · trace(A_s), with A_s the normal matrix Σ_i C_iᵀ Ω_i² C_i of the estimation (with the offset restriction, if any) at the stage's **first** evaluation, i.e. `EPS_STF` (workflow column 22) times the mean energy of the weighted basis traces. Value for B1: see the candidate report (Owner decision before B1-STF).
- ε_s is frozen per shot and stage, so Φ is one fixed function of m within a stage. Reload at a stage start reproduces it. A restart from an iteration model in the middle of a stage freezes ε_s at that model and therefore evaluates a different Φ than the original run (review V-7.2).

**Estimator (F2).**
- v̂_s solves (A_s + ε_s I) v = Σ_i C_iᵀ Ω_i² B d_{s,i} exactly: the weighted time-domain least-squares problem with the same weights as Φ.
- Solver: conjugate gradients in double precision, with FFT convolutions and a circulant (Wiener-type) preconditioner.
- Converged means that the **true** relative residual |b − (A + ε I) v| / |b| ≤ 1e-11. It is recomputed after the iterations, and PCG restarts from it (at most 5 restarts, 5000 iterations). Otherwise the evaluation fails collectively (review V-5; the earlier acceptance up to 100·tol is removed).
- It is computed on MYID 0 from the gathered global traces and broadcast, so every rank uses the identical v̂_s, independent of the domain decomposition.

**Gradient.**
- O2: because ∂(D_s + ½ ε_s‖v‖²)/∂v = 0 at v̂_s and ε_s does not depend on m, ∇Φ is the existing exact adjoint gradient of ½ Σ ‖ω (F_s(m; B v̂_s) − B d_s)‖² with v̂_s held fixed (variable projection). No new adjoint is needed.
- O1: the same expression is exact because v̂_s is fixed.

**Cost per evaluation and shot.**
- One additional forward of B e_1 (receiver traces only, no trajectory, no energy accumulation).
- A line-search trial needs no further forward: Φ_s follows from g and v̂_s.

## 3. Validation (collective, before the first iteration of each stage)

The driver aborts collectively if a stage has `INV_STF` ∉ {0, 1, 3}, or if `INV_STF` ≠ 0 and one of the following holds:
- `EPS_STF` ≤ 0 or non-finite;
- `OFFSETC_STF` < 0 or non-finite;
- `INV_STF` = 1 and `OFFSETC_STF` ≠ 0 (variable projection needs the estimation to use exactly the weights of Φ, F2);
- `NORMALIZE` ≠ 0;
- N_w < 1.

`QUELLART`, `SIGNAL_FILE` and the `SOURCE_FILE` columns TD (delay), FC and amplitude do not enter the estimated source (review V-7.1). Every support starts at sample 1, so a source delay TD > 0 must fit into `STF_LENGTH`. `QUELLART` is reported in the stage line.

If `OFFSETC_STF`, `OFFSET_MUTE` or `TIMEWIN` leave no trace or weight for the estimation of a shot, ε_s = 0 and the first evaluation of the stage aborts collectively with the message `… no trace (or no weight) left for the source estimation …` (review V-7.6). This is fail-closed, but it is detected only at the first evaluation, not in the validation before iteration 1.

Every stage with `INV_STF` ≠ 0 prints on MYID 0 the line `Exact SH STF: stage <n> INV_STF <v> STF_LENGTH <s> N_w <n> EPS_STF <e>`. At the stage's first evaluation it prints one line per shot, `Exact SH STF: stage <n> shot <s> eps <ε_s> traces <used> of <total>` (the effective ε_s, which must be recorded with every run: old workflows carry `EPS_STF` 0.1).

After every gradient evaluation it prints one line per shot: `Exact SH STF: stage <s> it <k> shot <i> D <data term> E <ε term> eps <ε_s> max|v| … |v| … pcg_it <iterations> relres <true relative residual>`. A fixed signal (`INV_STF` = 3 after the first evaluation) prints `pcg_it 0`. The same evaluation also prints the runtime check of s = g ∗ v: `Exact SH STF: shot <i> D_forward <data term of the true forward with B v̂> D_estimator <D> rel <relative difference>` (review V-7.5).

## 4. Code

- `src/SH/visco_sh_exact_stf.c`: the estimator (local, no MPI, no `err()`): normal-matrix trace, preconditioned CG, objective terms.
- `src/SH/visco_sh_exact_stf_shot.c`: the per-shot glue in both multi-shot wrappers:
  - impulse forward;
  - gather to MYID 0;
  - estimate and broadcast;
  - source B v̂;
  - the stage state (ε_s, O1 signals);
  - the output.
- Driver: stage validation and the stage start; output after every gradient evaluation.
- `read_par.c`: record 134 `STF_LENGTH`.

## 5. Output

After every gradient evaluation (iteration k of stage s), MYID 0 writes `<INV_MODELFILE>_stage_<s>_it_<k>.stf`:
- little-endian binary32, nshots rows of N_w samples;
- row s holds the v̂_s used in that evaluation, i.e. the signal before the stage filter.
- The output exists only for gradient evaluations. The signal at the model accepted in the last iteration of a stage is not written; the next stage estimates its own signal at that model (review V-7.4).

## 6. Deliberate deviations from legacy (genealogy, not oracle)

| Item | Legacy (`stf.c`) | Here |
| --- | --- | --- |
| When | iteration 1 of every stage, then fixed | every evaluation (O2) or the stage's first evaluation (O1) |
| Estimation weights | own mutes (`OFFSETC_STF`), STA/LTA window on the observed data only (`INV_STF` 2), normalisation; objective window not used | exactly the weights of Φ; with `INV_STF` = 3 optionally restricted to offsets < `OFFSETC_STF` (same threshold, comparison and unit as legacy; negative values are rejected instead of meaning "off") |
| Estimator | frequency-domain Wiener with circular division, truncated to ns samples, real part | exact weighted least squares in the time domain on the causal support |
| Regularisation | ε·ntr·Ebar, where Ebar accumulates running partial sums (trace-order dependent) | ε_s = `EPS_STF` · trace(A_s)/N_w, frozen per stage, part of Φ |
| Source after estimation | `QUELLART` switched to 3 globally; later stages with `INV_STF` = 0 re-filter the last estimate | per stage; `INV_STF` = 0 uses the configured source |
| Storage | text, 7 digits, re-read | double in memory, binary32 output |

## 7. Evidence (integration gate, `AGENTS.md`; criteria frozen before candidate output)

**(a) Capability**, local, with the S-2 tolerances:
- A1: Φ against an independent offline computation (dense least squares from the impulse-forward traces and the observed data);
- A2: the directional derivative of the gradient against fourth-order central differences of Φ, for Vs, ρ and Q, with filter, window and mask, on 1 and 2 ranks; steps 0.4 and 0.2 of the S-2 direction, tolerance 5e-3. **Test correction (Owner decision 2026-11-23):** the steps were 0.2 and 0.1 as in S-2. Under variable projection the directional derivative of the unfiltered case is only about 1e-3 of Φ, so the binary32 noise of Φ dominates below 0.2. The error curve over 0.8/0.4/0.2/0.1 is recorded for every case. Also for `INV_STF` = 3 with `OFFSETC_STF`;
- A3: at the true model with same-solver data, B v̂ = B w_true within relative L2 error ≤ 1e-3, with a support that contains w_true;
- A4: `INV_STF` = 0 is bitwise equal to the base, and the S-2 and L-BFGS suites pass;
- A5: identical Φ and v̂ for different domain decompositions;
- A6: reload ΔJ = 0, and a multi-stage run with `.stf` output; also `INV_STF` = 3 with `OFFSETC_STF`.

No physical reference case: S-3 changes neither operator, boundary nor material.

**(b) Reconstruction on B1-STF** against r7b M_V, with the same iteration budget (thresholds frozen before output):
- P-V ≤ P-V(r7b) + 0.03;
- ρ error, Q error and J_hold each ≤ 1.1 × r7b.
