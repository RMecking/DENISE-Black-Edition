# SH-MASK-01: update mask from file and source taper in the exact visco-SH FWI

Status: candidate. Interface: variant A, decided by the Owner on 2026-11-26
(legacy records, values between 0 and 1 rejected, both mask files equal,
taper as minimum over the sources, `FILTSIZE` with its legacy meaning). Base
`c76ce8ba7f43d18e3c9c9ffe1ffbbe75afb559b0`. Tests: `tests/test_sh_mask_01.py`.

**Difference from the legacy paths.** In the legacy SH/PSV paths these records
multiply the gradient. In the exact SH path the gradient stays unchanged: the
mask selects the free parameters and the source taper acts only on the search
direction and the initial L-BFGS metric.

## 1. What changes and what does not

Only the set of free parameters and the diagonal metric of the search direction
at the exact SH optimizer boundary change. Unchanged: forward and adjoint
recurrences, checkpoint/replay, the reverse step, the raw gradient, the
objective with its filters, windows and mutes, the source estimation, the line
search and the trial `m - alpha*p` with base-cell rejection.

With `SWS_TAPER_FILE = 0` and `SWS_TAPER_GRAD_SOURCES = 0` every output is
bytewise identical to the base, and the records 75-77 and 79 are not
validated.

## 2. Records (legacy positions, no new record)

| Record | Name | Exact SH meaning |
| ---: | --- | --- |
| 73 | `SWS_TAPER_GRAD_SOURCES` | 0 or 1: source taper T as a metric factor |
| 74 | `SWS_TAPER_CIRCULAR_PER_SHOT` | must be 0 (rejected as before) |
| 75 | `SRTSHAPE` | with record 73 = 1: must be 3 (sin^2) |
| 76 | `SRTRADIUS` | R in metres, finite and `>= DH` |
| 77 | `FILTSIZE` | `>= 0`: half width of the fixed square around each source |
| 78 | `SWS_TAPER_FILE` | 0 or 1: update mask from file |
| 79 | `TFILE` | basename of `<TFILE>_u.bin` and `<TFILE>_rho.bin` |

Both switches require `OPT_STEP_SCALING` 1 or 2 (the cell mask of
SH-OPT-SCALE-01 exists only there).

## 3. Update mask

`<TFILE>_u.bin` and `<TFILE>_rho.bin`: float32 in native byte order, the
global grid `NXG x NYG`, x outer, y inner (the layout of the model files),
size checked exactly. Each value is exactly 0 (fixed) or 1 (free); both files
are equal, because the optimizer has one cell set for all classes. The file
mask is combined with the CPML mask by logical AND. The effective mask `M`
enters the class scaling (SH-OPT-SCALE-01), the preconditioner and its
`W_max` (SH-PRECOND-01), and the L-BFGS entry set, sigma and pairs
(SH-LBFGS-01). Steps on fixed cells are exactly zero, so these cells keep
their start value in every iteration and stage.

## 4. Source taper

For each source s with grid indices `I_s = iround(x_s/DH)`,
`J_s = iround(y_s/DH)` (as the forward injection) and global cell `(I, J)`:

    T_s = 0                       |I-I_s| <= FILTSIZE and |J-J_s| <= FILTSIZE
    T_s = sin^2(pi r / (2 R))     0 < r < R,  r = DH sqrt((I-I_s)^2 + (J-J_s)^2)
    T_s = 1                       r >= R

in double, `T = min_s T_s`, stored as float. Cells with `T = 0` are fixed
(`M = 0`), because the metric must be positive on free cells.

T is a factor of the diagonal metric `D = T * P` (`P = 1` without
preconditioner, otherwise SH-PRECOND-01; product in double, stored as float):

- fallback direction: SH-OPT-SCALE-01 scaling of `D * g` over the free cells;
- L-BFGS: pairs from the raw gradient on the free cells, initial metric
  `w * D`; the descent test of SH-LBFGS-01 is unchanged.

`D` is diagonal and non-negative, so every direction is a descent direction
of the unchanged objective.

## 5. Rejections (collective, before iteration 1, no model output)

Either switch with `OPT_STEP_SCALING = 0`; a switch outside {0, 1}; a mask
file that is missing, has the wrong size, a value other than 0 or 1, or
differs between `_u` and `_rho`; no free cell; `SRTSHAPE != 3`;
`SRTRADIUS` not finite or `< DH`; `FILTSIZE < 0`;
`SWS_TAPER_CIRCULAR_PER_SHOT != 0`.

## 6. Diagnostics

With a switch on, stdout carries one setup line (`free_cells`) and per
iteration

    Exact SH update mask: stage s iteration k fixed_change_max .. taper_share .. max_change .. at x .. y ..

`fixed_change_max` is the largest |change| on fixed cells over all classes
(must be 0); `taper_share` the L2 share of the primary-class change on cells
with `T < 1` (-1 without taper); `max_change` and its position (global cell
times `DH`) the largest primary-class change. With `OPT_DIAG_OUT = 1` the
fields `.update_mask` and `.taper` are written per iteration.
