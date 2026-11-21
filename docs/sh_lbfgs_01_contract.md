# SH-LBFGS-01 contract: limited-memory BFGS direction at the exact SH optimizer boundary

Status: frozen interface and behaviour contract for oracle construction.

- **Revision:** 3 (2026-11-21); see section 7.
- **Owner decisions:** E-1..E-7 "Ja, Empfehlungen" and E-8 "Hier umsetzen", both 2026-11-21.
- **Base:** `86fd2bf6097890f99966b7e630a8c4aae3823354` (`modernization`, PR #110, SH-PRECOND-01).
- **Plan:** `/DENISE/DENISE-SH-LBFGS-01-PLAN-20261121.md`.
- **Predecessor contracts:** `docs/sh_opt_scale_01_contract.md` revision 2 and `docs/sh_precond_01_contract.md` revision 2. Both stay in force unchanged.

## 1. Scope and frozen behaviour

**What changes:** On request, the search direction of the exact viscoelastic SH FWI is replaced by a limited-memory BFGS direction. It is built in class-scaled variables, with the SH-PRECOND-01 diagonal as initial inverse-Hessian metric, curvature and descent safeguards, and deterministic memory resets.

**What stays unchanged:**

- forward modelling and material preparation;
- the Q->tau mapping;
- the exact C5/C7 reverse and the raw gradients;
- the objective;
- the line-search algorithm `step_length_est_sh_visc_exact.c`, including the SH-PRECOND-01 initial alpha;
- the trial construction `m - alpha*p` with base-cell rejection outside bounds;
- the SH-OPT-SCALE-01 mask, weights and delayed classes;
- the SH-PRECOND-01 preconditioner;
- every existing M6.3/SH-MPI/C8c/SH-OPT-SCALE-01/SH-PRECOND-01 contract.

`GRAD_METHOD` remains required to be 0 on the exact path; the legacy L-BFGS (`LBFGS.c`, `GRAD_METHOD == 2`, `NLBFGS`) is neither used nor changed.

**Identity default:** With `OPT_LBFGS == 0`, the driver behaves exactly as at the base, bitwise. This covers models, logs of J and alpha, stdout lines other than the new ones, and diagnostics.

## 2. Notation

- Classes `c in {primary, rho, q}` (primary = Vs; `OPT_LBFGS == 1` requires `INVMAT1 == 1`, see section 4).
- Arrays are 1-based `[j][i]`, with `j = 1..ny` and `i = 1..nx`, local to each rank. Only own cells are read or written; halos are never touched.
- `M` is the SH-OPT-SCALE-01 update mask (exactly 0.0f or 1.0f).
- `w_c >= 0` are the effective class weights of the iteration (SH-OPT-SCALE-01: `OPT_WEIGHT_*`, 0 before `INV_*_ITER`).
- `A = {c : w_c > 0}` is the active class set.
- The entry set is `E = {(c, j, i) : M[j][i] == 1 and c in A}`, taken globally over all ranks.
- `n_M` is the global count of masked cells, and `|A|` is the number of active classes.
- Every sum `sum_E` is a global sum in double over E, formed by `MPI_Allreduce` with `MPI_SUM`; the summation order is unspecified.
- `P` is the SH-PRECOND-01 preconditioner of the current iteration, or `P == 1` on every cell if none is supplied.

## 3. API (`include/fd.h`, new translation unit `src/SH/visco_sh_exact_lbfgs.c`)

**Common conventions:**

- The unit does not call `err()`.
- It links stand-alone with `mpicc -std=c99 -O2 -fcommon ... -lm` together with a harness that provides `err()`. It depends on no other DENISE translation unit.
- It is listed in `src/Makefile`.
- All functions except `visco_sh_exact_lbfgs_free` and `visco_sh_exact_lbfgs_get_sigma` are collective over `MPI_COMM_WORLD`, and every rank must call them.
- All ranks return the same value. The error decision is collective and is made before any output or state is written: on a non-zero return, neither the caller's output arrays, nor `*report`, nor the state are changed. On success, the report fields are identical on all ranks (they derive from `MPI_Allreduce` results).
- If several conditions apply, the first applicable code in the listed order wins; `-39` (MPI failure) is independent of that order.

```c
#define VISCO_SH_EXACT_LBFGS_MAX_MEMORY 20

struct visco_sh_exact_lbfgs { ... };   /* opaque; members are private */
```

The caller zero-initialises the state (`memset(&state, 0, sizeof(state))`) before `visco_sh_exact_lbfgs_init`. Only the functions of this section read or write its members.

### 3.1 Life cycle

```c
int visco_sh_exact_lbfgs_init(struct visco_sh_exact_lbfgs *lbfgs,
                              int nx, int ny, int memory);
void visco_sh_exact_lbfgs_free(struct visco_sh_exact_lbfgs *lbfgs);
```

**`visco_sh_exact_lbfgs_init`** allocates storage for `memory` pairs (s, y) of three float fields each, the previous base model and gradient (three float fields each), and double work storage.

- After success the state is initialised, has no stage and holds no pair.
- **Return codes:**
  - `-31` `lbfgs` NULL, `nx < 1`, `ny < 1`, or `memory` not in `1..VISCO_SH_EXACT_LBFGS_MAX_MEMORY`;
  - `-32` an allocation failed on some rank (every rank then releases what it allocated, and the state is left zero);
  - `-39` MPI failure.

**`visco_sh_exact_lbfgs_free`** is local. It releases all storage and zeroes the state. It does nothing for NULL or for a zero state.

### 3.2 Stage start

```c
int visco_sh_exact_lbfgs_begin_stage(struct visco_sh_exact_lbfgs *lbfgs,
        float **base_primary, float **base_rho, float **base_q,
        float **update_mask);
```

**Behaviour:** Freezes the class scales and clears the memory.

```
sigma_c = sqrt( sum over global masked cells of (double)m_c^2 / n_M )   for all three classes
```

On success:

- `sigma` is stored;
- the pair count is set to 0;
- the "previous" record is cleared;
- the state becomes stage-ready.

**Return codes:**

- `-31` `lbfgs` NULL or not initialised;
- `-33` a NULL array or row among the four inputs;
- `-34` a mask value not exactly 0.0f/1.0f, or a masked base value (any class) non-finite or `<= 0`;
- `-35` `n_M == 0`, or some `sigma_c` not finite or not `> 0`;
- `-39` MPI failure.

Values in unmasked cells are not read for validation and do not influence sigma.

```c
void visco_sh_exact_lbfgs_get_sigma(const struct visco_sh_exact_lbfgs *lbfgs,
                                    double sigma[3]);
```

`visco_sh_exact_lbfgs_get_sigma` is local and read-only. It writes the frozen `sigma_c` of the current stage, or three zeros if `lbfgs` is NULL or not stage-ready.

### 3.3 Direction

```c
struct visco_sh_exact_lbfgs_request {
   int nx, ny;                                    /* must equal init */
   float **base_primary, **base_rho, **base_q;    /* current base m_k */
   float **grad_primary, **grad_rho, **grad_q;    /* raw gradient g_k at m_k */
   float **update_mask;                           /* M */
   float **precond;                               /* P_k, or NULL: P == 1 */
   float weight_primary, weight_rho, weight_q;    /* effective w_c */
   float **step_primary, **step_rho, **step_q;    /* in: fallback direction; out */
};

struct visco_sh_exact_lbfgs_report {
   int action;          /* 0 sd_empty, 1 sd_reset, 2 sd_fallback, 3 lbfgs */
   int pair;            /* 0 none, 1 stored, 2 skipped */
   int pairs_used;      /* pairs in the two-loop (actions 2 and 3), else 0 */
   double curvature;    /* s.y/(|s||y|) of the candidate pair, else 0 */
   double gamma;        /* actions 2 and 3, else 0 */
   double descent;      /* D-cosine (step 4), actions 2 and 3, else 0 */
   double kappa;        /* action 3, else 0 */
   double sigma[3];     /* frozen sigma_c of the stage */
};

int visco_sh_exact_lbfgs_step(struct visco_sh_exact_lbfgs *lbfgs,
        const struct visco_sh_exact_lbfgs_request *request,
        struct visco_sh_exact_lbfgs_report *report);
```

**Input:** The step arrays hold, on input, the direction the driver would use without L-BFGS (SH-OPT-SCALE-01 or SH-PRECOND-01). This is the fallback direction.

**Validation** (collective, before anything is written):

- `-31` `lbfgs` NULL, not initialised or not stage-ready; `request` or `report` NULL; `request->nx`/`ny` different from init;
- `-33` a NULL array or row among base, grad, mask and step; or, if `precond != NULL`, a NULL row of `precond`;
- `-34` any of:
  - a weight non-finite or `< 0`;
  - a mask value not exactly 0.0f/1.0f;
  - a masked base value (any class) non-finite or `<= 0`;
  - a masked gradient value of an active class non-finite;
  - if `precond != NULL`, a masked P non-finite or `<= 0`;
- `-36` `A` is empty (all weights 0);
- `-39` MPI failure.

Values in unmasked cells and gradients of inactive classes are not read for validation and do not influence any result. The mask must be the same in all calls of a stage; this is not validated.

**Algorithm on success** (all arithmetic in double unless stated):

1. **Active set.** `A_k = {c : w_c > 0}`.

2. **Pair.**
   - **No previous record:** `pair = 0`.
   - **Previous record with `A_prev != A_k`:** clear the memory, `pair = 0`, and mark *reset*.
   - **Previous record with `A_prev == A_k`:** form the candidate pair on E. It is zero outside E.
     - `s(e) = (float)( ((double)m_k(e) - (double)m_prev(e)) / sigma_c )`
     - `y(e) = (float)( sigma_c * ((double)g_k(e) - (double)g_prev(e)) )`
     - Then `ss = sum_E s^2`, `yy = sum_E y^2` and `sy = sum_E s*y`, all from the stored binary32 values.
     - The pair is **stored** (`pair = 1`) if and only if `ss > 0`, `yy > 0`, `sy` is finite and `sy > 1e-8 * sqrt(ss*yy)`. If the memory is full, the oldest pair is discarded first. Otherwise the pair is **skipped** (`pair = 2`) and the memory is unchanged.
     - `curvature = sy/sqrt(ss*yy)` if `ss > 0`, `yy > 0` and the value is finite; otherwise 0.

3. **No pairs.** If the memory holds no pair, set `action = 1` if reset was marked in step 2, else `action = 0`. The step arrays are not written.

4. **Two-loop recursion.** Otherwise, with the stored pairs `i = 1..n` (oldest to newest):
   - `rho_i = 1/(s_i.y_i)`.
   - `g_x(e) = sigma_c * (double)g_k(e)`.
   - `D(e) = (double)w_c * (double)P[j][i]`.
   - **First loop:** `q = g_x`. For `i = n..1`: `a_i = rho_i * s_i.q`, then `q -= a_i * y_i`.
   - **Scaling:** `gamma = s_n.y_n / sum_E D*y_n^2`, using the newest pair; `r = gamma * D * q`.
   - **Second loop:** for `i = 1..n`: `b = rho_i * y_i.r`, then `r += (a_i - b) * s_i`.
   - `d = r`.
   - **Descent test:** `gd = sum_E g_x*d`, `gDg = sum_E D*g_x^2`, `dDd = sum_E d^2/D`.
     - `descent = gd/sqrt(gDg*dDd)` if `gDg > 0`, `dDd > 0` and the value is finite; else 0.
     - The test passes if and only if `gd` is finite, `gDg > 0`, `dDd > 0` and `gd > 1e-8 * sqrt(gDg*dDd)`.
     - If it fails: `action = 2`, clear the memory, and do not write the step arrays.

5. **Output (action 3).**
   - `kappa = 1/sqrt( sum_E d^2 / (|A| * n_M) )`.
   - For `e = (c, j, i)` in E: `step_c[j][i] = (float)(sigma_c * kappa * d(e))`.
   - Every other own cell of the three step arrays (unmasked cells and inactive classes) is set to `0.0f`.

6. **Previous record** (every successful call, all actions): copy `m_k` and `g_k` of all own cells, set `A_prev = A_k`, and mark the record valid.

`report` is fully written on success, including `sigma`.

**Consequences:**

- **Conventions:** The direction follows the subtractive convention of the trial (`m - alpha*p`). With action 3, the double-precision direction satisfies `kappa * sum_E g_x*d > 0`, so it is a descent direction. After rounding p to binary32 (relative error up to `2^-24` per entry), Cauchy-Schwarz bounds the change of `sum g.p` by `2^-24 * kappa * sqrt(gDg*dDd)`; the sign of the binary32 `sum g.p` is therefore guaranteed for `descent > 2^-24` (about `6e-8`), not for every `descent` in `(1e-8, 6e-8]`. The threshold 1e-8 is the value of the approved plan (4.3) and is kept; the line search remains the final safeguard: it accepts a step only if J strictly decreases, so a non-descent binary32 step cannot be accepted; the line search then fails and the run aborts collectively, as at the base.
- **Step size:** The masked RMS of `sigma_c^-1 * p` over E equals 1. `alpha` therefore stays a dimensionless relative change, and `EPS_SCALE` and the SH-PRECOND-01 initial alpha keep their meaning. With unit weights this matches the fallback direction, where every active class has relative RMS change `alpha`. With non-unit weights, `OPT_WEIGHT_*` only shape the initial metric `H0`; unlike the fallback direction they no longer set each class's share of the step.
- **Positive definiteness:** Every stored pair has `s.y > 0` and `D > 0` on E, so the recursion is positive definite. The descent test only guards rounding and the degenerate case `g_x == 0` on E.
- **Fallback:** Actions 0, 1 and 2 leave the fallback direction bitwise untouched. The first iteration of a stage and the first iteration after a reset are therefore bitwise identical to the run without L-BFGS, given the same base and gradient.
- **Secant:** The newest pair satisfies `H y_n = s_n` for the operator `H` with `d = H g_x`.
- **Reparametrisation invariance:** `m_c -> a*m_c` together with `g_c -> g_c/a` (a > 0) leaves `x`, `g_x`, `s`, `y` and `d` unchanged and scales `p_c` by `a`, up to rounding.

## 4. Driver controls (`denise.inp`, positional records)

Records 1-131 are unchanged. The new records are optional, read only if present and in order; missing trailing records keep their defaults. A file with 119, 122, 127 or 131 records behaves exactly as at the base.

| Record | Name | Type | Default | Meaning |
| ---: | --- | --- | --- | --- |
| 132 | `OPT_LBFGS` | int | 0 | 0 off, 1 L-BFGS direction after section 3 |
| 133 | `OPT_LBFGS_M` | int | 5 | memory length (pairs) |

### Configuration aborts

These aborts are collective, happen before the first iteration and write no model output. They apply in the exact SH FWI driver (`FWI_SH_visc.c`, MODE = 1, physical Q):

- `OPT_LBFGS` not in {0, 1};
- `OPT_LBFGS_M` not in `1..20`, which is validated also with `OPT_LBFGS == 0`;
- `OPT_LBFGS == 1` with `OPT_STEP_SCALING != 1`;
- `OPT_LBFGS == 1` with `INVMAT1 != 1`.

All `OPT_PRECOND` modes (0, 1, 2) and `OPT_ALPHA_ADAPT` values are allowed with L-BFGS.

### `OPT_LBFGS == 1`

- **Setup:** After the update mask has been built, the state is zero-initialised and `visco_sh_exact_lbfgs_init(NX, NY, OPT_LBFGS_M)` is called. A failure aborts collectively.
- **Stage start:** At the start of every workflow stage, before its first iteration, `visco_sh_exact_lbfgs_begin_stage` is called with the authoritative base model and the update mask. A failure aborts collectively. MYID 0 then prints, with the values from `visco_sh_exact_lbfgs_get_sigma`:

  `Exact SH L-BFGS: stage <n> sigma <%e> <%e> <%e>`
- **Every iteration:**
  1. The fallback direction is built exactly as without L-BFGS, into `exact_step_*`.
  2. `visco_sh_exact_lbfgs_step` is called with:
     - the base;
     - the raw gradients;
     - the mask;
     - `precond` = the SH-PRECOND-01 P of this iteration if `OPT_PRECOND > 0`, else NULL;
     - the effective weights;
     - `exact_step_*`.

     A non-zero return aborts collectively.
  3. MYID 0 prints:

     `Exact SH L-BFGS: stage <n> iteration <k> action <sd_empty|sd_reset|sd_fallback|lbfgs> pair <none|stored|skipped> pairs <%d> curvature <%e> gamma <%e> descent <%e> kappa <%e>`
  4. With `OPT_DIAG_OUT == 1`, the `.step_*` diagnostics are written after this call, so they hold the final direction.
- **At the end:** the state is freed.
- **Unchanged:** Everything else, including the line search, the trial, the alpha logs and the model output, is unchanged.

### `OPT_LBFGS == 0`

No L-BFGS function is called, and no `Exact SH L-BFGS` line is printed.

## 5. Acceptance oracles (to be implemented before the production code)

**Unit level:** harness `tests/utilities/sh_lbfgs_01_harness.c`, independent reference `tests/utilities/sh_lbfgs_01_reference.py`. Tolerances:

- step fields: `|got - want| <= 1e-6 * max|want|` per class field, with zero entries exactly 0;
- report scalars: `<= 1e-7` relative;
- flags: exact.

**Driver level:** the 32x32 SH-PRECOND-01 fixture (Vs patch, Q 80, 6 receivers, one source).

| ID | Content | Tolerance |
| --- | --- | --- |
| O1 | Identity: 2-iteration driver runs with 131 records and with 133 records (`OPT_LBFGS = 0`, `OPT_LBFGS_M = 7`) give bitwise equal models, diagnostics, misfit logs and J/alpha log lines. Cases: (a) `OPT_STEP_SCALING` 0; (b) `OPT_STEP_SCALING` 1; (c) `OPT_STEP_SCALING` 1 with `OPT_PRECOND` 2 and `OPT_ALPHA_ADAPT` 1. No `Exact SH L-BFGS` line appears. All existing SH-OPT-SCALE-01, SH-PRECOND-01, M6.3 and SH-MPI tests pass unmodified (verified by running them). | bitwise |
| O2 | Non-interference: `OPT_LBFGS = 1` vs `0` (scaling 1, precond 2, diag 1, 2 iterations). Iteration-1 model, gradient, step, energy and precond files are bitwise equal. Iteration-2 gradient, energy and precond files are bitwise equal. The iteration-2 step differs. Log: iteration 1 `action sd_empty pair none`, iteration 2 `action lbfgs pair stored pairs 1`. The fixture is required to store this pair (positive control of the L-BFGS branch). The L-BFGS direction reaches the trial: the iteration-2 models differ between on and off, and every masked cell of the iteration-2 model of the L-BFGS run equals `(float)(m_1 - (float)(alpha_2 * p_2))` from the iteration-1 model, the written iteration-2 step and the printed selected `alpha_2` (`opteps_vp`), with no rejected cell in the fixture. | bitwise; flags exact; trial `<= 2 ulp(m_2) + 1e-6 * abs(alpha_2 * p_2)` per cell |
| O3 | Two-loop closed form (unit): random pair sequences (memory 1, 3, 5; with and without P; weights including a zero class) against the reference recursion of section 3.3. Driver: the iteration-2 `.step_*` files equal the reference computed from the written files, namely start model, iteration-1 model, iteration-1 and iteration-2 `.grad_*`, iteration-2 `.precond`, the mask, weights 1/1/1 and `sigma` from the start model. The printed curvature, gamma, descent and kappa agree with the reference. | unit tolerances; driver `abs diff <= 1e-6 * max abs` per field, printed scalars `<= 2e-6` relative |
| O4 | Dense BFGS (unit): the direction equals `kappa * sigma * H g_x`, where H is built explicitly as a dense matrix by the recursion `H <- (I - rho s y^T) H (I - rho y s^T) + rho s s^T` from `gamma * D` with the same stored pairs. This is an algorithm independent of the two-loop. | unit step tolerance |
| O5 | Secant (unit): with `g_prev = y_n/sigma` and `g_k = 2 y_n/sigma` (per class), the direction `p/sigma` is parallel to `s_n`. | `<= 1e-6` relative to max |
| O6 | Quadratic (unit): 2x2 grid, 3 classes, all masked, n = 12 variables, memory 12. Pairs `y_i = A s_i` with 12 A-conjugate `s_i` for an SPD A with condition number `<= 4`. After 12 stored pairs, the direction for the gradient of the 13th call is parallel to `A^-1 g_x` (finite-termination identity `H = A^-1`). | `<= 1e-4` relative to max (binary32 models and gradients perturb conjugacy at about `1e-5`) |
| O7 | Safeguards (unit): (a) a pair with `s.y < 0` is skipped and the memory is kept (`pairs_used` unchanged); (b) `s == 0` is skipped; (b') `y == 0` with `s != 0` is skipped with curvature 0, a pair with `0 < s.y/(|s||y|) ~ 2e-9` is skipped, and a pair with `s.y/(|s||y|) ~ 1e-6` is stored (threshold discrimination); (c) a change of `A` gives `sd_reset`, and the next iteration uses one pair; (d) `g_x == 0` on E with memory gives `sd_fallback` and clears the memory; (e) the ring buffer with memory 2 after three stored pairs uses the newest two; (f) for actions 0, 1 and 2 the step arrays, halos included, are bitwise unchanged. Driver: `INV_RHO_ITER = INV_QS_ITER = 3`, `PRO = 0` (no misfit stop), 4 iterations, gives `sd_empty`, `lbfgs` (with exactly zero rho and q steps), `sd_reset`, then `lbfgs pairs 1`. A second workflow stage starts with `sd_empty`, and its printed sigma equals the RMS of the last stage-1 model. | flags exact; sigma `<= 2e-6` relative |
| O8 | Projection (unit): unmasked cells and inactive classes are exactly 0 in the output. NaN in unmasked cells and in gradients of inactive classes is accepted and has no influence. | exact |
| O9 | Invariance (unit): `m_c -> 1024*m_c`, `g_c -> g_c/1024` for one class scales only that class's output by 1024. The power of two makes the input scaling exact in binary32. | `<= 1e-6` relative to max |
| O10 | MPI: unit level 1x1, 2x1, 1x2 and 3x2; driver level 1x1, 2x1 and 1x2 (iteration-2 `.step_*` and the log flags). | unit tolerances; driver `abs diff <= 1e-6 * max abs` per field; flags exact |
| O11 | Error paths: the codes of section 3, collective (fault on one rank), with the listed precedence, and with neither outputs nor state changed. "State unchanged" is shown by a following successful call giving the same result as without the failed call. The configuration aborts of section 4 happen before the first iteration (no `TDFWI ITERATION` line, no misfit log), including 2-rank cases. Records 132..133 are read in order. `OPT_LBFGS_M` reaches the unit: with `PRO = 0`, memory 1 vs 5 gives `pairs 1` vs `pairs 2` in iteration 3; the fixture is required to store both pairs (positive control). `INVMAT1 = 3` with `OPT_LBFGS = 1` aborts (1 and 2 ranks); positive control: the same configuration with `OPT_LBFGS = 0` reaches `TDFWI ITERATION 1`. Unit level also covers a `ny` mismatch (`-31`) and NaN in unmasked cells at stage start (accepted, no influence on sigma). | exact |
| O12 | Descent (unit): `sum g.p > 0` for every action-3 output in random sequences. | sign exact |

The `-32` allocation failure of 3.1 cannot be forced without fault injection into the production unit. It is verified by review, not by test.

## 6. Non-goals

The following are not part of this package:

- Wolfe line search or gradients at trial points;
- nonlinear CG or PCG;
- Gauss-Newton or full Hessians;
- multiscale schemes;
- tapers;
- changes to gradients or physics;
- GPU work, reflection FWI, M9L;
- legacy L-BFGS.

## 7. Revision log

- Rev. 1 (2026-11-21): initial frozen contract.
- Rev. 2 (2026-11-21), after the independent oracle review (SCIENTIFIC-VERIFICATION, verdict READY WITH REQUIRED CORRECTIONS), and after the r1 oracle hashes had been pinned.
  - **RC-1, 3.2:** a local read-only accessor `visco_sh_exact_lbfgs_get_sigma`, so that the driver can print the stage-start sigma line; the opaque-state rule now refers to the functions of section 3.
  - **RC-2, 3.3 consequences:** the descent statement is qualified for the binary32 step. The threshold 1e-8 of the approved plan is unchanged.
  - **RC-3, section 3:** a failed call does not write `*report`.
  - **RC-4, O2:** the L-BFGS direction is shown to reach the trial.
  - **RC-5, O7:** threshold discrimination for the curvature rule.
  - **Notes:** report fields identical on all ranks (N-6); the role of the weights with L-BFGS (N-3); `PRO = 0` and declared positive controls for the multi-iteration driver tests (N-4); `ny` mismatch and NaN at stage start (N-7); wording (N-8); O6 tolerance rationale (N-5).
  - **RC-6** (r2 delta review): `visco_sh_exact_lbfgs_get_sigma` is named as local in the common conventions; the line-search safeguard wording is made explicit.
  - No formula, threshold or existing tolerance is changed.
- Rev. 3 (2026-11-21), Owner decision "Guard ergänzen" on SCIENTIFIC-VERIFICATION note N-3: `OPT_LBFGS == 1` requires `INVMAT1 == 1` (configuration abort, section 4; oracle O11). No other change.
