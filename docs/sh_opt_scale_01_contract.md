# SH-OPT-SCALE-01 contract: class scaling and update mask at the exact SH optimizer boundary

Status: frozen interface/behaviour contract for oracle construction (Owner
decisions E-1..E-6, 2026-11-19). Revision 2 (2026-11-19): API moved to a
separate struct/function/translation unit after the independent oracle review
found that an existing harness allocates `struct visco_sh_exact_optimizer_boundary`
without zero-initialisation (see section 6). Base: `d4693e1165c6d5d0cfe0bb8adc4cd4a79e4166fa`.
Plan: `/DENISE/DENISE-SH-OPT-SCALE-01-PLAN-20261119.md`.

## 1. Scope and frozen behaviour

Only the construction of the search direction `p` at the exact viscoelastic SH
optimizer boundary changes. Unchanged: forward modelling, material preparation,
Q->tau mapping, exact C5/C7 reverse and raw gradients, objective, line search
(`step_length_est_sh_visc_exact.c`), trial construction `m - alpha*p` with
rejection outside bounds, and every existing M6.3/SH-MPI/C8c contract.

With `scaling_mode == 0` (the zero-initialised default) the direction is the
current identity `p = g_raw`, bitwise.

## 2. Scaled-step API (include/fd.h, new translation unit `src/SH/visco_sh_exact_step_scaling.c`)

`struct visco_sh_exact_optimizer_boundary` and
`visco_sh_exact_build_steepest_subtractive_step` (file
`src/SH/visco_sh_exact_optimizer_boundary.c`) stay **byte-identical** to
`d4693e1`. New, in `include/fd.h`:

```c
struct visco_sh_exact_step_scaling {
   int scaling_mode;                 /* 0 identity, 1 RMS, 2 MAX            */
   float **base_primary;             /* current base model, local NY x NX   */
   float **base_rho;
   float **base_q;
   float **update_mask;              /* 1.0f or 0.0f per local cell          */
   float weight_primary, weight_rho, weight_q;  /* effective class weights   */
};

int visco_sh_exact_build_scaled_steepest_step(
        const struct visco_sh_exact_optimizer_boundary *boundary,
        const struct visco_sh_exact_step_scaling *scaling);
```

Both new functions (this one and the mask builder of section 3) are defined in
`src/SH/visco_sh_exact_step_scaling.c`, listed in `src/Makefile`, and link
stand-alone with `mpicc -std=c99 -O2 -fcommon ... -lm` plus a harness that
provides `err()` (the unit itself does not call `err()`).

- `scaling == NULL` or `scaling->scaling_mode == 0`: identical to
  `visco_sh_exact_build_steepest_subtractive_step(boundary)` (same return codes,
  p = g bitwise); no other member of `scaling` is read; not collective.
- Mode 1 or 2: **collective over `MPI_COMM_WORLD`**; every rank must call it.
  Arrays are 1-based `[j][i]`, `j=1..ny`, `i=1..nx` (local).
- For each class k in {primary, rho, q} with gradient g_k, base model m_k,
  weight w_k and mask M (all sums/maxima global over all ranks and over cells
  with M = 1):
  - n = global count of cells with M = 1.
  - Mode 1 (RMS): N_k = sqrt(sum M*g_k^2 / n), r_k = sqrt(sum M*m_k^2 / n).
  - Mode 2 (MAX): N_k = max over M=1 of |g_k|, r_k = max over M=1 of |m_k|.
  - s_k = r_k / N_k if N_k > 0, else s_k = 0.
  - p_k(cell) = M(cell) * w_k * s_k * g_k(cell); evaluated in double, stored
    as float. Cells with M = 0 and classes with w_k = 0 get exactly 0.0f.
  - Sums and maxima are accumulated in double and reduced with `MPI_Allreduce`
    (`MPI_SUM` / `MPI_MAX`); results may differ between decompositions only by
    reduction order.
- Consequence: with the step `alpha*p_k`, the masked RMS (mode 1) or masked max
  (mode 2) of the class-k change equals `alpha * w_k * r_k`, i.e. relative to
  the masked RMS/max of the base model it is `alpha * w_k`. `alpha` is
  dimensionless.
- Descent: p = D g with D diagonal and D >= 0, hence sum g.p >= 0, and > 0 if
  some class with w_k > 0 has a non-zero masked gradient.
- Return values (identical on all ranks in modes 1/2; the decision is made
  collectively before any output array is written):
  - `-1` NULL boundary; `-2` nx<1 or ny<1; `-3` NULL gradient/step array
    (unchanged meanings);
  - `-4` scaling_mode not in {0,1,2} (checked collectively when called on
    several ranks);
  - `-5` modes 1/2 only: NULL base or mask array (or NULL row), or a mask value
    other than exactly 0.0f/1.0f;
  - `-6` modes 1/2 only: a weight that is negative or non-finite, or all three
    weights zero;
  - `-7` modes 1/2 only: global n == 0;
  - `-8` modes 1/2 only: a non-finite masked gradient or masked base value in
    any of the three classes, independent of its weight (fail closed);
  - `-9` MPI reduction failure.
  On a non-zero return the step arrays are not modified. If several
  conditions are violated, the smallest-numbered applicable code among -4..-8
  in the order -4, -5, -6, -7, -8 is returned. Values `-0.0f` count as 0.
  Only own cells `j=1..ny, i=1..nx` are read (no halo).

## 3. Update mask builder

```c
int visco_sh_exact_build_update_mask(float **mask, int nx, int ny,
        int pos_x, int pos_y, int nprocx, int nprocy,
        int fw, int free_surf, int boundary);
```

Local, not collective. For local cell (i, j), `mask[j][i] = 0.0f` exactly when
it lies in the CPML as used by `update_v_PML_SH.c`/`update_s_*_PML_SH.c`:

- `(!boundary) && pos_x == 0 && i <= fw`, or
- `(!boundary) && pos_x == nprocx-1 && i >= nx-fw+1`, or
- `pos_y == 0 && !free_surf && j <= fw`, or
- `pos_y == nprocy-1 && j >= ny-fw+1`;

otherwise `1.0f`. Returns 0, or -1 for NULL mask / nx<1 / ny<1 / fw<0 /
nprocx<1 / nprocy<1 / pos_x not in [0,nprocx-1] / pos_y not in [0,nprocy-1]
(mask untouched on error).

## 4. Driver controls (`denise.inp`, positional records)

Records 1-119 and the optional 120-122 are unchanged. New optional records,
read only if present and in order; missing trailing records keep their
defaults (a file with 119 or 122 records behaves exactly as at `d4693e1`):

| Record | Name | Type | Default | Meaning |
| ---: | --- | --- | --- | --- |
| 123 | `OPT_STEP_SCALING` | int | 0 | 0 identity, 1 RMS, 2 MAX |
| 124 | `OPT_WEIGHT_PRIMARY` | float | 1.0 | w for Vs (INVMAT1=1) or mu (INVMAT1=3) |
| 125 | `OPT_WEIGHT_RHO` | float | 1.0 | w for rho |
| 126 | `OPT_WEIGHT_Q` | float | 1.0 | w for physical Q |
| 127 | `OPT_DIAG_OUT` | int | 0 | 1: write raw gradient and step per iteration |

Exact SH FWI driver (`FWI_SH_visc.c`, MODE=1, physical Q):

- `OPT_STEP_SCALING` must be 0, 1 or 2 and `OPT_DIAG_OUT` 0 or 1; otherwise
  configuration abort.
- `OPT_STEP_SCALING == 0`: behaviour identical to `d4693e1`, including the
  existing guard `INV_VS_ITER, INV_RHO_ITER, INV_QS_ITER in {0,1}` (validated,
  not used); the three weights must then equal 1.0, otherwise configuration
  abort. `OPT_DIAG_OUT` may be 1 (output only).
- `OPT_STEP_SCALING in {1,2}`:
  - `INV_VS_ITER`, `INV_RHO_ITER`, `INV_QS_ITER` must be integers >= 0;
    weights finite and >= 0, not all zero.
  - In iteration `iter` (1-based) the effective weight of class k is
    `OPT_WEIGHT_k` if `iter >= INV_k_ITER`, else 0 (`INV_k_ITER = 0` or 1:
    active from iteration 1). If all effective weights are zero in an
    iteration, the run aborts collectively at the start of that iteration,
    before its gradient is computed.
  - "Configuration abort" means a collective abort before the first
    iteration (like the existing `exact_sh_require_config_*` guards), with
    no model output.
  - The update mask is built once from `FW`, `FREE_SURF`, `BOUNDARY`, `POS`,
    `NPROCX`, `NPROCY` and passed with the current authoritative base model.
- `OPT_DIAG_OUT == 1`: after the direction is built, write per iteration, with
  the existing `writemod`/`mergemod` pattern and prefix
  `<INV_MODELFILE>_stage_<s>_it_<k>`, the merged files `.grad_<vs|mu>`,
  `.grad_rho`, `.grad_qs` (raw gradients) and `.step_<vs|mu>`, `.step_rho`,
  `.step_qs` (direction p), removing the per-rank parts like
  `model_it_out_SH_visc`.
- With `OPT_STEP_SCALING == 0` the driver calls
  `visco_sh_exact_build_steepest_subtractive_step` exactly as at `d4693e1`;
  otherwise it calls `visco_sh_exact_build_scaled_steepest_step`.
- The line search, trial state, objective and Q->tau regeneration are called
  exactly as before; only the step arrays differ.

## 5. Acceptance oracles (to be implemented independently, before code)

| ID | Content | Tolerance |
| --- | --- | --- |
| O1 | mode 0 (and `scaling == NULL`): p bitwise equal to g; `visco_sh_exact_optimizer_boundary.c` and the existing struct byte-identical to `d4693e1`; the existing SH/M6.3/SH-MPI tests pass unmodified (verified by running them, not by a hash freeze of test files) | bitwise |
| O2 | modes 1/2 on synthetic fields (incl. mask, weights, zero gradient class): p equals the closed form of section 2 computed independently in double | <= 4 ulp (float); masked/zero-weight cells exactly 0 |
| O3 | descent: sum over cells of g.p > 0 for random non-zero masked g with some w_k > 0; = 0 only if all masked g of weighted classes are zero | sign exact |
| O4 | unit invariance: m_k -> c*m_k, g_k -> g_k/c (c > 0) leaves (alpha*p_k)/m_k unchanged | <= 1e-6 relative |
| O5 | MPI: 1x1, 2x1, 1x2 (same global fields) give the same p | <= 1e-6 relative |
| O6 | trial: with the CPML mask, cells with M=0 are bitwise unchanged in the trial state for any alpha | bitwise |
| O7 | driver: mode 1 with `INV_RHO_ITER = INV_QS_ITER = 3`, `ITERMAX >= 2`: rho and Q of the iteration-1 model bitwise equal the start model while Vs changes | bitwise |
| O8 | error paths: codes -4..-8 of section 2 and the mask builder's -1 returned without modifying output arrays | exact |
| O9 | parser/driver: files with 119/122 records keep defaults and identical behaviour; 127 records are read in order | exact |

## 6. Revision log

- Rev. 2 (2026-11-19), after the independent oracle review (TEST-ORACLES):
  the existing harness `tests/utilities/m63c_c8c_optimizer_boundary_harness.c`
  declares the boundary struct without zero-initialisation, so appended
  members would be indeterminate there and could break O1. The scaling inputs
  therefore move to the new `struct visco_sh_exact_step_scaling` and function
  `visco_sh_exact_build_scaled_steepest_step` in a new translation unit;
  the existing struct, function and file stay byte-identical. Further
  clarifications: code order -4..-8, -8 independent of weight, -0.0f counts as
  0, own cells only, trailing records 123..127 optional in order,
  configuration-abort timing, O1 by running existing tests (no hash freeze of
  test files). Formulas, tolerances and all other oracles are unchanged.
