/* SH-OPT-SCALE-01: class-scaled steepest-descent direction and CPML update
 * mask at the exact viscoelastic SH optimizer boundary.
 * Contract: docs/sh_opt_scale_01_contract.md (revision 2).
 * The raw-gradient identity direction of
 * visco_sh_exact_build_steepest_subtractive_step stays unchanged; this unit
 * only adds a positive diagonal class scaling and a cell mask. */
#include "fd.h"

/* Priorities of the collective error decision (contract section 2):
 * -1..-3 (existing structural checks), then -4, -5, -6, -7, -8. */
#define SCALING_OK_PRIORITY 99

static int scaling_mask_value_is_valid(float value) {
    return (value == 0.0f) || (value == 1.0f);
}

static int scaling_local_priority(
        const struct visco_sh_exact_optimizer_boundary *b,
        const struct visco_sh_exact_step_scaling *s, long *masked_cells) {
    float **grad[3], **base[3];
    float weights[3];
    int i, j, k;

    *masked_cells = 0;
    if (b == NULL) return 1;
    if ((b->nx < 1) || (b->ny < 1)) return 2;
    if ((b->grad_raw_primary == NULL) || (b->grad_raw_rho == NULL) ||
            (b->grad_raw_q == NULL) || (b->optimizer_step_primary == NULL) ||
            (b->optimizer_step_rho == NULL) || (b->optimizer_step_q == NULL))
        return 3;
    if ((s->scaling_mode != 1) && (s->scaling_mode != 2)) return 4;
    if ((s->base_primary == NULL) || (s->base_rho == NULL) ||
            (s->base_q == NULL) || (s->update_mask == NULL)) return 5;
    grad[0] = b->grad_raw_primary; grad[1] = b->grad_raw_rho;
    grad[2] = b->grad_raw_q;
    base[0] = s->base_primary; base[1] = s->base_rho; base[2] = s->base_q;
    for (j = 1; j <= b->ny; ++j) {
        if (s->update_mask[j] == NULL) return 5;
        for (k = 0; k < 3; ++k)
            if ((base[k][j] == NULL) || (grad[k][j] == NULL)) return 5;
        for (i = 1; i <= b->nx; ++i)
            if (!scaling_mask_value_is_valid(s->update_mask[j][i])) return 5;
    }
    weights[0] = s->weight_primary; weights[1] = s->weight_rho;
    weights[2] = s->weight_q;
    for (k = 0; k < 3; ++k)
        if (!isfinite(weights[k]) || (weights[k] < 0.0f)) return 6;
    if ((weights[0] == 0.0f) && (weights[1] == 0.0f) && (weights[2] == 0.0f))
        return 6;
    for (j = 1; j <= b->ny; ++j) {
        for (i = 1; i <= b->nx; ++i) {
            if (s->update_mask[j][i] != 1.0f) continue;
            ++(*masked_cells);
            for (k = 0; k < 3; ++k)
                if (!isfinite(grad[k][j][i]) || !isfinite(base[k][j][i]))
                    return 8;
        }
    }
    return SCALING_OK_PRIORITY;
}

int visco_sh_exact_build_scaled_steepest_step(
        const struct visco_sh_exact_optimizer_boundary *boundary,
        const struct visco_sh_exact_step_scaling *scaling) {
    float **grad[3], **base[3], **step[3];
    double local_red[7], global_red[7], scale[3], norm, ref;
    float weights[3];
    long masked_cells = 0;
    int local_priority, global_priority, i, j, k;

    /* Mode 0 (or no scaling): the d4693e1 identity direction p = g_raw with
     * the return codes of visco_sh_exact_build_steepest_subtractive_step.
     * Repeated here so that this unit links stand-alone (contract section 2). */
    if ((scaling == NULL) || (scaling->scaling_mode == 0)) {
        if (boundary == NULL) return -1;
        if ((boundary->nx < 1) || (boundary->ny < 1)) return -2;
        if ((boundary->grad_raw_primary == NULL) ||
                (boundary->grad_raw_rho == NULL) ||
                (boundary->grad_raw_q == NULL) ||
                (boundary->optimizer_step_primary == NULL) ||
                (boundary->optimizer_step_rho == NULL) ||
                (boundary->optimizer_step_q == NULL)) return -3;
        for (j = 1; j <= boundary->ny; ++j) {
            for (i = 1; i <= boundary->nx; ++i) {
                boundary->optimizer_step_primary[j][i] =
                        boundary->grad_raw_primary[j][i];
                boundary->optimizer_step_rho[j][i] =
                        boundary->grad_raw_rho[j][i];
                boundary->optimizer_step_q[j][i] =
                        boundary->grad_raw_q[j][i];
            }
        }
        return 0;
    }

    /* Collective validation before any output array is written. */
    local_priority = scaling_local_priority(boundary, scaling, &masked_cells);
    if (MPI_Allreduce(&local_priority, &global_priority, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -9;
    local_red[0] = (double)masked_cells;
    if (MPI_Allreduce(local_red, global_red, 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -9;
    if ((global_priority >= 1) && (global_priority <= 6))
        return -global_priority;
    if (!(global_red[0] > 0.0)) return -7;
    if (global_priority == 8) return -8;

    grad[0] = boundary->grad_raw_primary; grad[1] = boundary->grad_raw_rho;
    grad[2] = boundary->grad_raw_q;
    step[0] = boundary->optimizer_step_primary;
    step[1] = boundary->optimizer_step_rho;
    step[2] = boundary->optimizer_step_q;
    base[0] = scaling->base_primary; base[1] = scaling->base_rho;
    base[2] = scaling->base_q;
    weights[0] = scaling->weight_primary; weights[1] = scaling->weight_rho;
    weights[2] = scaling->weight_q;

    /* Masked class statistics in double: [n, |g|_k (3), |m|_k (3)]. */
    for (k = 0; k < 7; ++k) local_red[k] = 0.0;
    for (j = 1; j <= boundary->ny; ++j) {
        for (i = 1; i <= boundary->nx; ++i) {
            if (scaling->update_mask[j][i] != 1.0f) continue;
            local_red[0] += 1.0;
            for (k = 0; k < 3; ++k) {
                double g = (double)grad[k][j][i];
                double m = (double)base[k][j][i];
                if (scaling->scaling_mode == 1) {
                    local_red[1 + k] += g * g;
                    local_red[4 + k] += m * m;
                } else {
                    if (fabs(g) > local_red[1 + k]) local_red[1 + k] = fabs(g);
                    if (fabs(m) > local_red[4 + k]) local_red[4 + k] = fabs(m);
                }
            }
        }
    }
    if (MPI_Allreduce(&local_red[0], &global_red[0], 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -9;
    if (MPI_Allreduce(&local_red[1], &global_red[1], 6, MPI_DOUBLE,
                      (scaling->scaling_mode == 1) ? MPI_SUM : MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -9;

    for (k = 0; k < 3; ++k) {
        if (scaling->scaling_mode == 1) {
            norm = sqrt(global_red[1 + k] / global_red[0]);
            ref = sqrt(global_red[4 + k] / global_red[0]);
        } else {
            norm = global_red[1 + k];
            ref = global_red[4 + k];
        }
        scale[k] = (norm > 0.0) ? ref / norm : 0.0;
    }

    for (j = 1; j <= boundary->ny; ++j) {
        for (i = 1; i <= boundary->nx; ++i) {
            for (k = 0; k < 3; ++k) {
                if ((scaling->update_mask[j][i] != 1.0f) ||
                        (weights[k] == 0.0f) || (scale[k] == 0.0)) {
                    step[k][j][i] = 0.0f;
                } else {
                    step[k][j][i] = (float)((double)weights[k] * scale[k] *
                                            (double)grad[k][j][i]);
                }
            }
        }
    }
    return 0;
}

int visco_sh_exact_build_update_mask(float **mask, int nx, int ny,
        int pos_x, int pos_y, int nprocx, int nprocy,
        int fw, int free_surf, int boundary) {
    int i, j, cpml;

    if ((mask == NULL) || (nx < 1) || (ny < 1) || (fw < 0) ||
            (nprocx < 1) || (nprocy < 1) || (pos_x < 0) ||
            (pos_x > nprocx - 1) || (pos_y < 0) || (pos_y > nprocy - 1))
        return -1;
    for (j = 1; j <= ny; ++j)
        if (mask[j] == NULL) return -1;
    for (j = 1; j <= ny; ++j) {
        for (i = 1; i <= nx; ++i) {
            /* Same CPML regions as update_v_PML_SH.c / update_s_*_PML_SH.c. */
            cpml = ((!boundary) && (pos_x == 0) && (i <= fw)) ||
                   ((!boundary) && (pos_x == nprocx - 1) && (i >= nx - fw + 1)) ||
                   ((pos_y == 0) && (!free_surf) && (j <= fw)) ||
                   ((pos_y == nprocy - 1) && (j >= ny - fw + 1));
            mask[j][i] = cpml ? 0.0f : 1.0f;
        }
    }
    return 0;
}
