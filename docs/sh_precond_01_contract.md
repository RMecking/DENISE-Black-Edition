# SH-PRECOND-01 contract: illumination preconditioning and adaptive initial step at the exact SH optimizer boundary

Status: frozen interface/behaviour contract for oracle construction.

- **Revision:** 2 (2026-11-20); see section 6.
- **Owner decisions:** E-1..E-4 "Ja, Empfehlungen" and E-5 "Hier umsetzen", both 2026-11-20.
- **Base:** `8ca5738e8c1880e9f6bdd6a87f7c080b505fb304`. The code is identical to `eed08e1`, SH-OPT-SCALE-01.
- **Plan:** `/DENISE/DENISE-SH-PRECOND-01-PLAN-20261120.md`.
- **Predecessor contract:** `docs/sh_opt_scale_01_contract.md` revision 2. It stays in force unchanged.

## 1. Scope and frozen behaviour

**What changes:**

- On request, the search direction of the exact viscoelastic SH FWI is preconditioned by a positive diagonal operator P built from the forward wavefield energy of the current base model.
- On request, the initial trial step of the unchanged line search is derived from the previously selected step.

**What stays unchanged:**

- forward modelling and material preparation;
- the Q->tau mapping;
- the exact C5/C7 reverse and the raw gradients;
- the objective;
- the line-search algorithm `step_length_est_sh_visc_exact.c`;
- the trial construction `m - alpha*p` with rejection outside bounds;
- the SH-OPT-SCALE-01 class scaling, mask, weights and delayed classes;
- every existing M6.3/SH-MPI/C8c/SH-OPT-SCALE-01 contract.

`EPRECOND` remains required to be 0 on the exact path.

**Identity default:** With `OPT_PRECOND == 0` and `OPT_ALPHA_ADAPT == 0`, the driver behaves exactly as at the base, bitwise. This covers models, logs of J and alpha, and diagnostics.

## 2. API (`include/fd.h`, new translation unit `src/SH/visco_sh_exact_precond.c`)

**Common conventions:**

- Arrays are 1-based `[j][i]`, with `j = 1..ny` and `i = 1..nx`, local to each rank.
- Only own cells are read or written; halos are never touched.
- The new unit does not call `err()`.
- It links stand-alone with `mpicc -std=c99 -O2 -fcommon ... -lm` together with `src/SH/visco_sh_exact_step_scaling.c` and a harness that provides `err()`.
- The unit is listed in `src/Makefile`.

### 2.1 Energy accumulation

```c
void visco_sh_exact_accumulate_energy(float **energy, float **field,
                                      int nx, int ny);
```

- For every own cell it performs `energy[j][i] += field[j][i] * field[j][i]` in binary32.
- It is local, not collective, and does nothing if a pointer is NULL or `nx < 1` or `ny < 1`.
- After n calls on zero-initialised energy, the oracle requires `|E - E_exact| <= (n + 1) * 2^-24 * E_exact`, where E_exact is the exact sum of the squared binary32 inputs. All terms are non-negative, so this is the classical recursive-summation bound. It holds with or without fused multiply-add.

### 2.2 Receiver-line term (Plessix & Mulder type)

```c
int visco_sh_exact_build_receiver_line_term(float **term, int nx, int ny,
        int pos_x, int pos_y, double x_min, double x_max);
```

- Local and not collective. It requires a uniform decomposition, as DENISE does.
- **Coordinates:** For own cell (i, j), use global grid coordinates `x = i + pos_x*nx` and `z = j + pos_y*ny`, in grid units.
- **Value:** `term[j][i] = (float)( asinh((x_max - x)/z) - asinh((x_min - x)/z) )`, evaluated in double.
- **Meaning:** This equals ∫_{x_min}^{x_max} dx_r / sqrt((x - x_r)^2 + z^2) for a continuous receiver line at depth 0, one grid step above row 1. It has the same structure as legacy `EPRECOND == 3`, with the model extent replaced by the receiver extent. The receiver depth is ignored, as in legacy. The value is dimensionless, so DH cancels.
- **Return values:**
  - `0` on success.
  - `-1` if `term` or a row is NULL, `nx < 1`, `ny < 1`, `pos_x < 0`, `pos_y < 0`, `x_min` or `x_max` is non-finite, or `!(x_min < x_max)`.
  - On error, `term` is untouched.

### 2.3 Preconditioner

```c
struct visco_sh_exact_precond_request {
   int precond_mode;        /* 1 source energy, 2 source energy x receiver line */
   int nx, ny;
   float epsilon;           /* stabilisation, 0 < epsilon <= 1 */
   float **source_energy;   /* W_s >= 0 (2.1, summed over all shots) */
   float **receiver_term;   /* R > 0 (2.2); read in mode 2 only */
   float **update_mask;     /* exactly 0.0f or 1.0f (SH-OPT-SCALE-01 mask) */
   float **precond;         /* output P */
};

int visco_sh_exact_build_illumination_precond(
        const struct visco_sh_exact_precond_request *request);
```

**Behaviour:** Collective over `MPI_COMM_WORLD`, and every rank must call it.

For masked cells (M = 1), compute in double:

- mode 1: `W = W_s`;
- mode 2: `W = sqrt(W_s) * R`;
- `W_max` = the global maximum of W over masked cells, via `MPI_Allreduce` with `MPI_MAX`.

Then:

- masked cells: `P = (float)(1.0 / (W + (double)epsilon * W_max))`;
- unmasked cells: `P = 0.0f` exactly.

Consequences:

- masked cells satisfy `0 < P <= 1/(epsilon*W_max)`;
- the amplification relative to the best-illuminated cell is at most `(1 + epsilon)/epsilon`;
- P is invariant to `W -> c*W` up to the global factor `1/c`.

**Return values:** All ranks return the same value. The decision is collective and is made before any output is written; on a non-zero return `precond` is untouched. If several conditions apply, the first applicable code in the listed order wins:

- `-11` NULL request, `nx < 1` or `ny < 1`, or a NULL `source_energy`, `update_mask` or `precond` array or row. In mode 2 this also covers a NULL `receiver_term` array or row.
- `-12` `precond_mode` is not 1 or 2, or `epsilon` is non-finite, `<= 0` or `> 1`.
- `-13` A mask value is not exactly 0.0f/1.0f, a masked `W_s` is non-finite or `< 0`, or in mode 2 a masked R is non-finite or `<= 0`.
- `-14` The global masked cell count is 0, or `W_max` is not `> 0` (no illumination), or `(float)(1.0 / ((double)epsilon * W_max))` is not finite (P would overflow binary32).
- `-19` An MPI reduction failed; this is independent of the order above.

### 2.4 Preconditioned scaled direction

```c
int visco_sh_exact_build_preconditioned_step(
        const struct visco_sh_exact_optimizer_boundary *boundary,
        const struct visco_sh_exact_step_scaling *scaling,
        float **precond, float **work_primary, float **work_rho,
        float **work_q);
```

**Behaviour:** Collective over `MPI_COMM_WORLD`.

1. Validate collectively; the first applicable code in this order wins:
   - `-1`, `-2`, `-3` as in SH-OPT-SCALE-01: NULL boundary; `nx < 1` or `ny < 1`; NULL gradient or step array.
   - `-15` `scaling` is NULL, or `scaling_mode` is not 1 or 2.
   - `-16` `precond` or a work array, or one of their rows, is NULL.
   - `-17` An own cell of `precond` is non-finite or satisfies `P < 0.0f`. `-0.0f` is accepted, because it compares equal to 0.
   - `-19` An MPI failure, independent of the order above.

   Codes of step 3 can only occur after this validation passed.

   On these codes, neither the step nor the work arrays are written.
2. For every own cell and class k, set `work_k[j][i] = (float)((double)P[j][i] * (double)g_k[j][i])`.
3. Return `visco_sh_exact_build_scaled_steepest_step(&b, scaling)`, where `b` is a copy of `*boundary` whose `grad_raw_*` members point to the work arrays.

   Its codes `-4..-9` and its no-write-on-error rule therefore apply unchanged. The work arrays may have been written at that point.

**Closed form:** For each class k:

- `q_k = P * g_k`;
- `s_k = r_k / N_k(M*q_k)`, using the RMS (mode 1) or MAX (mode 2) statistic of SH-OPT-SCALE-01 over masked cells, with `s_k = 0` if `N_k = 0`;
- `p_k = M * w_k * s_k * q_k`.

The step still realises a relative masked RMS or max class change `alpha*w_k`, so `alpha` stays dimensionless.

**Descent:** `sum g.p = sum M w_k s_k P g_k^2 >= 0`. It is strictly positive as soon as some class with `w_k > 0` and `r_k > 0` has a masked cell with `P > 0` and `g_k != 0`. P from 2.3 is `> 0` on every masked cell, so with it this reduces to a non-zero masked gradient of such a class.

### 2.5 Initial step of the line search

```c
float visco_sh_exact_initial_alpha(float eps_scale, int adaptive,
        float factor, int iteration, float previous_alpha);
```

The function is pure, local and needs no MPI. It returns `eps_scale` if any of these holds:

- `adaptive == 0`;
- `iteration <= 1`;
- `previous_alpha` is non-finite or `<= 0`.

Otherwise it computes `c = (float)((double)factor * (double)previous_alpha)` and returns `c < eps_scale ? c : eps_scale`.

## 3. `sh_visc_with_material_trajectory`

Inside the time loop, directly after the block that calls `visco_sh_material_observable_end_step()`, the forward loop applies the 2.1 arithmetic to `(Ws, (*waveSH).pvz, NX, NY)`, but only when `trajectory != NULL` and `Ws != NULL`.

It does so through the header helper `static inline void visco_sh_exact_energy_step(float **energy, float **field, int nx, int ny)` in `include/fd.h`. `visco_sh_exact_accumulate_energy` also delegates to this helper after its argument checks, so both use one definition of the arithmetic. Because the helper is inline, `sh_visc.c` gains no link-time dependency on the new translation unit, and every existing harness that links `sh_visc.c` stays link-compatible.

This means energy is accumulated only in the exact gradient forward run (mode 0 with trajectory), at every time step `nt = 1..NT` (DTINV = 1 on the exact path).

The function is otherwise unchanged:

- The legacy `EPRECOND` accumulation stays as it is, and it is inactive on the exact path.
- With `trajectory == NULL` (legacy `sh_visc`, objective and trial runs) nothing changes.

## 4. Driver controls (`denise.inp`, positional records)

Records 1-127 are unchanged. The new records are optional, read only if present and in order; missing trailing records keep their defaults. A file with 119, 122 or 127 records behaves exactly as at the base.

| Record | Name | Type | Default | Meaning |
| ---: | --- | --- | --- | --- |
| 128 | `OPT_PRECOND` | int | 0 | 0 off, 1 source energy, 2 source energy x receiver line |
| 129 | `OPT_PRECOND_EPS` | float | 0.005 | epsilon of 2.3 |
| 130 | `OPT_ALPHA_ADAPT` | int | 0 | 1: initial alpha after 2.5 |
| 131 | `OPT_ALPHA_ADAPT_FACTOR` | float | 2.0 | factor c of 2.5 |

### Configuration aborts

These aborts are collective, happen before the first iteration and write no model output. They apply in the exact SH FWI driver (`FWI_SH_visc.c`, MODE = 1, physical Q):

- `OPT_PRECOND` not in {0, 1, 2};
- `OPT_PRECOND > 0` with `OPT_STEP_SCALING == 0`;
- `OPT_PRECOND_EPS` non-finite, `<= 0` or `> 1`;
- `OPT_ALPHA_ADAPT` not in {0, 1};
- `OPT_ALPHA_ADAPT_FACTOR` non-finite or `< 1`;
- `OPT_PRECOND == 2` while the global receiver columns do not satisfy `x_min < x_max`.

### `OPT_PRECOND > 0`

- **Setup:**
  - An energy field and a P field are allocated, plus three work fields.
  - In mode 2, the receiver-line term is built once, with `x_min`/`x_max` = min/max of the global receiver columns `acq.recpos[1][1..ntr_glob]`.
- **In every iteration:**
  1. Before the gradient evaluation, every own energy cell is set to 0.0f and passed as `source_energy` of the gradient request.
  2. The trial-objective requests of the line search carry `source_energy = NULL`.
  3. After the gradient, P is built with the SH-OPT-SCALE-01 update mask.
  4. `visco_sh_exact_build_preconditioned_step` is called with the same scaling struct, including effective weights and delayed classes, that would otherwise go to `visco_sh_exact_build_scaled_steepest_step`.
- **No interference:** The objective, the raw gradients and every forward or adjoint wavefield are bitwise identical to `OPT_PRECOND == 0`.

### `OPT_PRECOND == 0`

The driver dispatches exactly as at the base, and `source_energy` stays NULL.

### `OPT_ALPHA_ADAPT`

- The line search receives `initial_alpha = visco_sh_exact_initial_alpha(EPS_SCALE, OPT_ALPHA_ADAPT, OPT_ALPHA_ADAPT_FACTOR, iter, previous)`.
- `previous` is the alpha selected in the previous iteration of the same workflow stage. It is reset to 0 at each stage start.
- With `OPT_ALPHA_ADAPT == 0` this is `EPS_SCALE` bitwise.
- With `OPT_ALPHA_ADAPT == 1`, MYID 0 prints `Exact SH initial alpha: iteration <k> alpha_start <%e>` to stdout once per iteration.

### `OPT_DIAG_OUT == 1` and `OPT_PRECOND > 0`

In addition to the SH-OPT-SCALE-01 files, the driver writes `.energy` (W_s) and `.precond` (P). They use the same prefix and merge pattern, and are written after the direction has been built.

## 5. Acceptance oracles (to be implemented before the production code)

| ID | Content | Tolerance |
| --- | --- | --- |
| O1 | Identity: 2-iteration driver runs with 127 records and with 131 records (`OPT_PRECOND = OPT_ALPHA_ADAPT = 0`, non-default `OPT_PRECOND_EPS` and factor) give bitwise equal models, diagnostics and J/alpha logs, for `OPT_STEP_SCALING` 0 and 1. All existing SH-OPT-SCALE-01, M6.3 and SH-MPI tests pass unmodified (verified by running them), and so do the SH-OPT-SCALE-01 record variants 119/122/127. | bitwise |
| O2 | No interference: `OPT_PRECOND = 2` vs `0` (both `OPT_STEP_SCALING = 1`). The iteration-1 raw gradient files are bitwise equal and the steps differ. J_base agrees in the printed log, `%e` with 7 significant digits; bitwise equality of the objective is carried by the bitwise equal raw gradients and the unchanged objective code path. | bitwise (gradients); printed J equal |
| O3 | Accumulation (2.1): synthetic snapshot sequences meet the summation bound. Driver: `.energy` of iteration 2 of a 2-iteration run equals `.energy` of iteration 1 of a run started from the accepted iteration-1 model, so the trials do not leak into W_s. | bound of 2.1; driver bitwise |
| O4 | Receiver term (2.2) against independent numerical quadrature of the line integral; error codes -1. | <= 1e-6 relative |
| O5 | Closed forms: P (2.3) and p (2.4) against the independent double reference on synthetic fields, including mask, weights and a zero class; unmasked cells exactly 0. Driver diagnostics: `.precond` obeys 2.3 computed from `.energy` and, in mode 2, from R evaluated with the closed asinh form of 2.2 in double; `.step_*` obey 2.4 computed from `.grad_*`, the start model and `.precond`. `.energy`/`.precond` are absent with `OPT_DIAG_OUT = 0`. | <= 4 ulp; 0 exact |
| O6 | Descent: `sum g.p > 0` for random non-zero masked g with some `w_k > 0`. | sign exact |
| O7 | Invariance: `W_s -> c*W_s` (mode 1) or `R -> c*R` (mode 2) leaves p unchanged. | <= 1e-6 relative |
| O8 | MPI: unit level, 1x1, 2x1, 1x2 and 3x2 give the same P and p. Driver: 1x1, 2x1 and 1x2 give the same `.energy`, `.precond` and `.step_*`. | unit <= 1e-6 relative; driver `abs diff <= 1e-6 * max abs` per field |
| O9 | Error paths: the codes of 2.2-2.4, collective, with the listed precedence and with neither output nor (for 2.4 step 1) work arrays written; initial alpha (2.5) for all branches; configuration aborts of section 4, which must happen before the first iteration (no `TDFWI ITERATION` line, no misfit log), including a 2-rank case; a positive control for the receiver-column abort; records 128..131 read in order; the adaptive initial alpha reaches the line search (iteration-2 model differs from `OPT_ALPHA_ADAPT = 0` exactly when alpha_start < `EPS_SCALE`, iteration 1 bitwise equal) and its min branch is exercised. | exact; printed alpha <= 2e-6 relative |
| O10 | Evidence, not acceptance: in the driver fixture, P of iteration 1 at the source cell is below the masked mean of P, for both modes. | reported only (no assertion) |

## 6. Revision log

- Rev. 1 (2026-11-20): initial frozen contract.
- Rev. 2 (2026-11-20), after the independent oracle review (SCIENTIFIC-VERIFICATION, verdict HOLD), and after the r1 oracle hashes had been pinned before any production file existed.
  - **Section 3:** the forward hook uses the shared header-inline helper instead of a call into the new translation unit. Otherwise about eight existing harnesses that link `sh_visc.c` with fixed source lists would fail to link. The arithmetic of 2.1 is unchanged.
  - **2.3:** a new `-14` condition for a P that would overflow binary32.
  - **2.3 and 2.4:** precedence is worded as "first applicable code in the listed order", and `-19` is independent of that order.
  - **2.4:** `-17` is specified as `P < 0.0f` (`-0.0f` accepted), and the descent statement is qualified.
  - **O2:** the J comparison is stated as a printed-log comparison.
  - **O5:** the mode-2 P check uses the closed asinh form, so it needs no tolerance beyond 4 ulp, and the absence of diagnostics with `OPT_DIAG_OUT = 0` is checked.
  - **O8:** a driver-level comparison is added.
  - **O9:** strengthened with location-specific aborts, a 2-rank abort, NaN/inf, a positive control, work-array no-write, the 2.4 precedence, and the wiring and min branch of the adaptive alpha.
  - **O10:** report only.
  - No formula or existing tolerance is changed.
