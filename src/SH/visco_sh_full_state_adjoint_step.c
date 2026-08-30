/* Exact transpose of one fixed-material viscoelastic SH propagation step.
 *
 * The routine composes the locked C1--C4 point, sampling, halo-copy and
 * free-surface transposes.  It is intentionally not connected to the legacy
 * viscoelastic SH dispatcher
 * or the legacy FWI path.  bar_next_work is consumed as mutable reverse
 * workspace; bar_prev is overwritten on the complete relevant state range.
 */

#include "fd.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#define M63C_MAX_PATCH 13
#define M63C_PATCH_CELLS (M63C_MAX_PATCH * M63C_MAX_PATCH)

static int state_complete(
        const struct visco_sh_full_state *state, int fw) {
    if ((state == NULL) || (state->vz == NULL) || (state->sxz == NULL) ||
            (state->syz == NULL) || (state->r == NULL) ||
            (state->q == NULL)) return 0;
    if ((fw > 0) && ((state->psi_sxz_x == NULL) ||
            (state->psi_syz_y == NULL) || (state->psi_vzx == NULL) ||
            (state->psi_vzy == NULL))) return 0;
    return 1;
}

static int state_distinct(
        const struct visco_sh_full_state *next,
        const struct visco_sh_full_state *prev, int fw) {
    if ((next == prev) || (next->vz == prev->vz) ||
            (next->sxz == prev->sxz) || (next->syz == prev->syz) ||
            (next->r == prev->r) || (next->q == prev->q)) return 0;
    if ((fw > 0) && ((next->psi_sxz_x == prev->psi_sxz_x) ||
            (next->psi_syz_y == prev->psi_syz_y) ||
            (next->psi_vzx == prev->psi_vzx) ||
            (next->psi_vzy == prev->psi_vzy))) return 0;
    return 1;
}

static void copy_field(
        float **dst, float **src, int row_min, int row_max,
        int col_min, int col_max) {
    int i, j;
    for (j = row_min; j <= row_max; ++j)
        for (i = col_min; i <= col_max; ++i) dst[j][i] = src[j][i];
}

static void zero_field(
        float **field, int row_min, int row_max,
        int col_min, int col_max) {
    int i, j;
    for (j = row_min; j <= row_max; ++j)
        for (i = col_min; i <= col_max; ++i) field[j][i] = 0.0f;
}

static void copy_memory(
        float ***dst, float ***src, int row_min, int row_max,
        int col_min, int col_max, int mechanisms) {
    int i, j, l;
    for (j = row_min; j <= row_max; ++j)
        for (i = col_min; i <= col_max; ++i)
            for (l = 1; l <= mechanisms; ++l)
                dst[j][i][l] = src[j][i][l];
}

static int flatten_field(
        float **field, int row_min, int row_max, int col_min, int col_max,
        double *flat) {
    int i, j, offset = 0;
    if ((field == NULL) || (flat == NULL)) return -1;
    for (j = row_min; j <= row_max; ++j)
        for (i = col_min; i <= col_max; ++i)
            flat[offset++] = field[j][i];
    return 0;
}

static void unflatten_field(
        float **field, int row_min, int row_max, int col_min, int col_max,
        const double *flat) {
    int i, j, offset = 0;
    for (j = row_min; j <= row_max; ++j)
        for (i = col_min; i <= col_max; ++i)
            field[j][i] = (float)flat[offset++];
}

static int receiver_transpose(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *work, int row_min, int row_max,
        int col_min, int col_max) {
    double *flat = NULL;
    int *x = NULL, *y = NULL;
    int cells, k, rows, stride, status = -1;

    if (cfg->nrec == 0) return 0;
    rows = row_max - row_min + 1;
    stride = col_max - col_min + 1;
    cells = rows * stride;
    flat = (double *)calloc((size_t)cells, sizeof(double));
    x = (int *)calloc((size_t)cfg->nrec, sizeof(int));
    y = (int *)calloc((size_t)cfg->nrec, sizeof(int));
    if ((flat == NULL) || (x == NULL) || (y == NULL)) goto cleanup;
    if (flatten_field(work->vz, row_min, row_max, col_min, col_max,
                      flat) != 0) goto cleanup;
    for (k = 0; k < cfg->nrec; ++k) {
        x[k] = cfg->rec_x[k] - col_min;
        y[k] = cfg->rec_y[k] - row_min;
    }
    status = visco_sh_receiver_velocity_sampling_vjp(
            cfg->nrec, x, y, cfg->bar_receiver, flat, rows, stride);
    if (status == 0)
        unflatten_field(work->vz, row_min, row_max, col_min, col_max,
                        flat);
cleanup:
    free(flat);
    free(x);
    free(y);
    return status;
}

static int source_transpose(
        const struct visco_sh_full_step_config *cfg,
        float **bar_after, float **bar_before, int row_min, int row_max,
        int col_min, int col_max, double *bar_signal) {
    double *after = NULL, *before = NULL;
    int *x = NULL, *y = NULL;
    int cells, k, rows, stride, status = -1;

    rows = row_max - row_min + 1;
    stride = col_max - col_min + 1;
    cells = rows * stride;
    after = (double *)calloc((size_t)cells, sizeof(double));
    before = (double *)calloc((size_t)cells, sizeof(double));
    if ((after == NULL) || (before == NULL)) goto cleanup;
    if (cfg->nsrc > 0) {
        x = (int *)calloc((size_t)cfg->nsrc, sizeof(int));
        y = (int *)calloc((size_t)cfg->nsrc, sizeof(int));
        if ((x == NULL) || (y == NULL)) goto cleanup;
    }
    if (flatten_field(bar_after, row_min, row_max, col_min, col_max,
                      after) != 0) goto cleanup;
    for (k = 0; k < cfg->nsrc; ++k) {
        x[k] = cfg->src_x[k] - col_min;
        y[k] = cfg->src_y[k] - row_min;
    }
    status = visco_sh_velocity_source_injection_vjp(
            rows, stride, after, before, cfg->nsrc, x, y,
            cfg->source_type, bar_signal);
    if (status == 0)
        unflatten_field(bar_before, row_min, row_max, col_min, col_max,
                        before);
cleanup:
    free(after);
    free(before);
    free(x);
    free(y);
    return status;
}

static int validate_config(const struct visco_sh_full_step_config *cfg) {
    int k;
    if ((cfg == NULL) || (cfg->nx < 1) || (cfg->ny < 1) ||
            (cfg->fdorder < 2) || (cfg->fdorder > 12) ||
            ((cfg->fdorder % 2) != 0) || (cfg->mechanisms < 1) ||
            (cfg->fw < 0) || !(cfg->dt > 0.0f) || !(cfg->dh > 0.0f) ||
            (cfg->hc == NULL) || (cfg->rhoi == NULL) ||
            (cfg->fipjp == NULL) || (cfg->f == NULL) ||
            (cfg->bip == NULL) || (cfg->bjm == NULL) ||
            (cfg->cip == NULL) || (cfg->cjm == NULL) ||
            (cfg->dip == NULL) || (cfg->d == NULL) ||
            (cfg->nproc_x < 1) || (cfg->nproc_y < 1) ||
            (cfg->pos[1] < 0) || (cfg->pos[1] >= cfg->nproc_x) ||
            (cfg->pos[2] < 0) || (cfg->pos[2] >= cfg->nproc_y) ||
            (cfg->nrec < 0) || (cfg->nsrc < 0)) return -1;
    if ((cfg->nrec > 0) && ((cfg->rec_x == NULL) ||
            (cfg->rec_y == NULL) || (cfg->bar_receiver == NULL))) return -1;
    if ((cfg->nsrc > 0) && ((cfg->src_x == NULL) ||
            (cfg->src_y == NULL) || (cfg->source_type == NULL))) return -1;
    if ((cfg->fw > 0) && ((cfg->K_x == NULL) || (cfg->a_x == NULL) ||
            (cfg->b_x == NULL) || (cfg->K_x_half == NULL) ||
            (cfg->a_x_half == NULL) || (cfg->b_x_half == NULL) ||
            (cfg->K_y == NULL) || (cfg->a_y == NULL) ||
            (cfg->b_y == NULL) || (cfg->K_y_half == NULL) ||
            (cfg->a_y_half == NULL) || (cfg->b_y_half == NULL))) return -1;
    for (k = 1; k <= cfg->fdorder / 2; ++k)
        if (!isfinite(cfg->hc[k])) return -1;

    /* Locked C2/C3 selectors reject simultaneous opposing CPML operations.
     * C5a exposes that restriction instead of claiming an overlap transpose. */
    if ((cfg->fw > 0) && (!cfg->boundary) && (cfg->nproc_x == 1) &&
            (cfg->nx <= 2 * cfg->fw)) return -2;
    if ((cfg->fw > 0) && (!cfg->free_surface) &&
            (cfg->nproc_y == 1) && (cfg->ny <= 2 * cfg->fw)) return -2;
    return 0;
}

static int reverse_stress_block(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *work,
        struct visco_sh_full_state *prev) {
    double patch[M63C_PATCH_CELLS];
    double *bar_rx_next = NULL, *bar_qy_next = NULL;
    double *bar_rx_prev = NULL, *bar_qy_prev = NULL;
    double *ax = NULL, *ay = NULL;
    double *cx = NULL, *cy = NULL, *zero_l = NULL;
    double strain[2] = {0.0, 0.0}, bar_stress[2], forward_f[2];
    double f_zero[2] = {0.0, 0.0}, bar_stress_prev[2];
    double bar_psi_next[2], bar_psi_prev[2], g_tau[2], g_modulus[2];
    double cpml_K[2], cpml_a[2], cpml_b[2];
    int cpml_active[2], aux_x, aux_y;
    int h = cfg->fdorder / 2, patch_size = 2 * h + 1;
    int i, j, l, pi, pj, status = -1;

    bar_rx_next = (double *)calloc((size_t)cfg->mechanisms, sizeof(double));
    bar_qy_next = (double *)calloc((size_t)cfg->mechanisms, sizeof(double));
    bar_rx_prev = (double *)calloc((size_t)cfg->mechanisms, sizeof(double));
    bar_qy_prev = (double *)calloc((size_t)cfg->mechanisms, sizeof(double));
    ax = (double *)calloc((size_t)cfg->mechanisms, sizeof(double));
    ay = (double *)calloc((size_t)cfg->mechanisms, sizeof(double));
    cx = (double *)calloc((size_t)cfg->mechanisms, sizeof(double));
    cy = (double *)calloc((size_t)cfg->mechanisms, sizeof(double));
    zero_l = (double *)calloc((size_t)cfg->mechanisms, sizeof(double));
    if ((bar_rx_next == NULL) || (bar_qy_next == NULL) ||
            (bar_rx_prev == NULL) || (bar_qy_prev == NULL) || (ax == NULL) ||
            (ay == NULL) || (cx == NULL) || (cy == NULL) ||
            (zero_l == NULL)) goto cleanup;
    for (l = 0; l < cfg->mechanisms; ++l) {
        ax[l] = (double)cfg->bip[l + 1] * cfg->cip[l + 1];
        ay[l] = (double)cfg->bjm[l + 1] * cfg->cjm[l + 1];
    }

    for (j = 1; j <= cfg->ny; ++j) {
        for (i = 1; i <= cfg->nx; ++i) {
            status = visco_sh_stress_cpml_select_x(
                    i, cfg->nx, cfg->fw, cfg->boundary, cfg->pos[1],
                    cfg->nproc_x, cfg->K_x_half, cfg->a_x_half,
                    cfg->b_x_half, &cpml_active[0], &aux_x,
                    &cpml_K[0], &cpml_a[0], &cpml_b[0]);
            if (status != 0) goto cleanup;
            status = visco_sh_stress_cpml_select_y(
                    j, cfg->ny, cfg->fw, cfg->free_surface, cfg->pos[2],
                    cfg->nproc_y, cfg->K_y, cfg->a_y, cfg->b_y,
                    cfg->K_y_half, cfg->a_y_half, cfg->b_y_half,
                    &cpml_active[1], &aux_y, &cpml_K[1], &cpml_a[1],
                    &cpml_b[1]);
            if (status != 0) goto cleanup;

            bar_stress[0] = work->sxz[j][i];
            bar_stress[1] = work->syz[j][i];
            forward_f[0] = cfg->fipjp[j][i];
            forward_f[1] = cfg->f[j][i];
            for (l = 0; l < cfg->mechanisms; ++l) {
                bar_rx_next[l] = work->r[j][i][l + 1];
                bar_qy_next[l] = work->q[j][i][l + 1];
                bar_rx_prev[l] = 0.0;
                bar_qy_prev[l] = 0.0;
                cx[l] = -(double)cfg->bip[l + 1] *
                        cfg->dip[j][i][l + 1];
                cy[l] = -(double)cfg->bjm[l + 1] *
                        cfg->d[j][i][l + 1];
                prev->r[j][i][l + 1] = 0.0f;
                prev->q[j][i][l + 1] = 0.0f;
            }
            bar_psi_next[0] = cpml_active[0] ?
                    work->psi_vzx[j][aux_x] : 0.0;
            bar_psi_next[1] = cpml_active[1] ?
                    work->psi_vzy[aux_y][i] : 0.0;
            bar_psi_prev[0] = bar_psi_prev[1] = 0.0;
            bar_stress_prev[0] = bar_stress_prev[1] = 0.0;
            g_tau[0] = g_tau[1] = 0.0;
            g_modulus[0] = g_modulus[1] = 0.0;
            for (pi = 0; pi < M63C_PATCH_CELLS; ++pi) patch[pi] = 0.0;

            status = update_s_visc_PML_SH_adjoint_point(
                    cfg->fdorder, cfg->mechanisms, cfg->dh, cfg->dt,
                    cfg->hc, cpml_active, cpml_K, cpml_a, cpml_b, strain,
                    bar_stress, bar_rx_next, bar_qy_next,
                    forward_f, ax, ay, cx, cy,
                    f_zero, f_zero, zero_l, zero_l, zero_l, zero_l,
                    bar_psi_next, bar_stress_prev,
                    bar_rx_prev, bar_qy_prev, bar_psi_prev, patch,
                    patch_size, patch_size, h, h, g_tau, g_modulus);
            if (status != 0) goto cleanup;
            prev->sxz[j][i] = (float)bar_stress_prev[0];
            prev->syz[j][i] = (float)bar_stress_prev[1];
            for (l = 0; l < cfg->mechanisms; ++l) {
                prev->r[j][i][l + 1] = (float)bar_rx_prev[l];
                prev->q[j][i][l + 1] = (float)bar_qy_prev[l];
            }
            if (cpml_active[0])
                prev->psi_vzx[j][aux_x] = (float)bar_psi_prev[0];
            if (cpml_active[1])
                prev->psi_vzy[aux_y][i] = (float)bar_psi_prev[1];
            if ((g_tau[0] != 0.0) || (g_tau[1] != 0.0) ||
                    (g_modulus[0] != 0.0) || (g_modulus[1] != 0.0)) {
                status = -3;
                goto cleanup;
            }
            for (pj = 0; pj < patch_size; ++pj)
                for (pi = 0; pi < patch_size; ++pi)
                    work->vz[j + pj - h][i + pi - h] +=
                            (float)patch[pj * patch_size + pi];
        }
    }
    status = 0;
cleanup:
    free(bar_rx_next);
    free(bar_qy_next);
    free(bar_rx_prev);
    free(bar_qy_prev);
    free(ax);
    free(ay);
    free(cx);
    free(cy);
    free(zero_l);
    return status;
}

static int reverse_velocity_block(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *work,
        struct visco_sh_full_state *prev) {
    double patch_x[M63C_PATCH_CELLS], patch_y[M63C_PATCH_CELLS];
    double bar_psi_next[2], bar_psi_prev[2], cpml_K[2];
    double cpml_a[2], cpml_b[2], bar_vz_prev;
    int cpml_active[2], aux_x, aux_y;
    int h = cfg->fdorder / 2, patch_size = 2 * h + 1;
    int i, j, pi, pj, status;

    for (j = 1; j <= cfg->ny; ++j) {
        for (i = 1; i <= cfg->nx; ++i) {
            status = visco_sh_velocity_cpml_select_x(
                    i, cfg->nx, cfg->fw, cfg->boundary, cfg->pos[1],
                    cfg->nproc_x, cfg->K_x, cfg->a_x, cfg->b_x,
                    &cpml_active[0], &aux_x, &cpml_K[0], &cpml_a[0],
                    &cpml_b[0]);
            if (status != 0) return status;
            status = visco_sh_velocity_cpml_select_y(
                    j, cfg->ny, cfg->fw, cfg->free_surface, cfg->pos[2],
                    cfg->nproc_y, cfg->K_y, cfg->a_y, cfg->b_y,
                    &cpml_active[1], &aux_y, &cpml_K[1], &cpml_a[1],
                    &cpml_b[1]);
            if (status != 0) return status;
            bar_psi_next[0] = cpml_active[0] ?
                    work->psi_sxz_x[j][aux_x] : 0.0;
            bar_psi_next[1] = cpml_active[1] ?
                    work->psi_syz_y[aux_y][i] : 0.0;
            bar_psi_prev[0] = bar_psi_prev[1] = 0.0;
            bar_vz_prev = 0.0;
            for (pi = 0; pi < M63C_PATCH_CELLS; ++pi) {
                patch_x[pi] = 0.0;
                patch_y[pi] = 0.0;
            }
            status = update_v_PML_SH_adjoint_point(
                    cfg->fdorder, cfg->dt, cfg->dh, cfg->rhoi[j][i],
                    cfg->hc, cpml_active, cpml_K, cpml_a, cpml_b,
                    work->vz[j][i], bar_psi_next, &bar_vz_prev,
                    bar_psi_prev, patch_x, patch_y, patch_size, patch_size,
                    h, h);
            if (status != 0) return status;
            prev->vz[j][i] = (float)bar_vz_prev;
            if (cpml_active[0])
                prev->psi_sxz_x[j][aux_x] = (float)bar_psi_prev[0];
            if (cpml_active[1])
                prev->psi_syz_y[aux_y][i] = (float)bar_psi_prev[1];
            for (pj = 0; pj < patch_size; ++pj)
                for (pi = 0; pi < patch_size; ++pi) {
                    prev->sxz[j + pj - h][i + pi - h] +=
                            (float)patch_x[pj * patch_size + pi];
                    prev->syz[j + pj - h][i + pi - h] +=
                            (float)patch_y[pj * patch_size + pi];
                }
        }
    }
    return 0;
}

int visco_sh_full_state_adjoint_step(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *bar_next_work,
        struct visco_sh_full_state *bar_prev,
        double *bar_signal) {
    int h, row_min, row_max, col_min, col_max, status;

    status = validate_config(cfg);
    if (status != 0) return status;
    if (!state_complete(bar_next_work, cfg->fw) ||
            !state_complete(bar_prev, cfg->fw) ||
            !state_distinct(bar_next_work, bar_prev, cfg->fw) ||
            ((cfg->nsrc > 0) && (bar_signal == NULL))) return -1;

    h = cfg->fdorder / 2;
    row_min = -h;
    row_max = cfg->ny + h + 1;
    col_min = 1 - h;
    col_max = cfg->nx + h;

    zero_field(bar_prev->vz, row_min, row_max, col_min, col_max);
    zero_field(bar_prev->sxz, row_min, row_max, col_min, col_max);
    zero_field(bar_prev->syz, row_min, row_max, col_min, col_max);
    copy_memory(bar_prev->r, bar_next_work->r,
                row_min, row_max, col_min, col_max, cfg->mechanisms);
    copy_memory(bar_prev->q, bar_next_work->q,
                row_min, row_max, col_min, col_max, cfg->mechanisms);
    if (cfg->fw > 0) {
        copy_field(bar_prev->psi_sxz_x, bar_next_work->psi_sxz_x,
                   1, cfg->ny, 1, 2 * cfg->fw);
        copy_field(bar_prev->psi_syz_y, bar_next_work->psi_syz_y,
                   1, 2 * cfg->fw, 1, cfg->nx);
        copy_field(bar_prev->psi_vzx, bar_next_work->psi_vzx,
                   1, cfg->ny, 1, 2 * cfg->fw);
        copy_field(bar_prev->psi_vzy, bar_next_work->psi_vzy,
                   1, 2 * cfg->fw, 1, cfg->nx);
    }
    if (cfg->nsrc > 0) {
        int source;
        for (source = 0; source < cfg->nsrc; ++source)
            bar_signal[source] = 0.0;
    }

    status = receiver_transpose(
            cfg, bar_next_work, row_min, row_max, col_min, col_max);
    if (status != 0) return status;
    status = exchange_s_SH_adjoint(
            bar_next_work->sxz, bar_next_work->syz, cfg->nx, cfg->ny,
            cfg->fdorder, cfg->boundary, cfg->pos, cfg->nproc_x,
            cfg->nproc_y, cfg->index, cfg->comm);
    if (status != MPI_SUCCESS) return status;
    if (cfg->free_surface && (cfg->pos[2] == 0))
        surface_elastic_SH_stress_adjoint(
                bar_next_work->syz, cfg->nx, h);

    /* The stress halo/surface maps precede the constitutive update in
     * reverse.  Only now is their transformed identity branch the input to
     * the stress-point VJP. */
    copy_field(bar_prev->sxz, bar_next_work->sxz,
               row_min, row_max, col_min, col_max);
    copy_field(bar_prev->syz, bar_next_work->syz,
               row_min, row_max, col_min, col_max);

    status = reverse_stress_block(cfg, bar_next_work, bar_prev);
    if (status != 0) return status;

    if (cfg->free_surface && (cfg->pos[2] == 0))
        surface_elastic_SH_velocity_adjoint(
                bar_next_work->vz, cfg->nx, h);
    status = exchange_v_SH_adjoint(
            bar_next_work->vz, cfg->nx, cfg->ny, cfg->fdorder,
            cfg->boundary, cfg->pos, cfg->nproc_x, cfg->nproc_y,
            cfg->index, cfg->comm);
    if (status != MPI_SUCCESS) return status;

    /* Source^T and velocity-update^T both consume this same post-source
     * cotangent.  source_transpose copies it to prev; physical cells are then
     * overwritten by the exact velocity point VJP, whose identity term uses
     * the unchanged work value. */
    status = source_transpose(
            cfg, bar_next_work->vz, bar_prev->vz, row_min, row_max,
            col_min, col_max, bar_signal);
    if (status != 0) return status;
    return reverse_velocity_block(cfg, bar_next_work, bar_prev);
}
