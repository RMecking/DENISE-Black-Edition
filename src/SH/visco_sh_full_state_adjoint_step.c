/* Exact transpose of one fixed-material viscoelastic SH propagation step.
 *
 * The routine composes the locked C1--C4 point, sampling, halo-copy and
 * free-surface transposes.  It is intentionally not connected to the legacy
 * viscoelastic SH dispatcher
 * or the legacy FWI path.  bar_next_work is consumed as mutable reverse
 * workspace; bar_prev is overwritten on the complete relevant state range.
 *
 * K-7: for subdomains with nx >= 2 and ny >= 2 the stress and velocity
 * blocks use the cell kernels stress_block_cells and velocity_block_cells,
 * exact transcriptions of update_s_visc_PML_SH_adjoint_point and
 * update_v_PML_SH_adjoint_point with the dense patch scatter replaced by its
 * cross contributions plus add_positive_zero.  Narrower subdomains keep the
 * point functions with the dense patch.  Every float and double result is
 * bitwise that of the base step (tests/test_sh_k7_reverse_step.py against
 * the frozen copy in tests/utilities/sh_k7_reference).  The +-0 argument
 * assumes IEEE round-to-nearest without flush-to-zero or denormals-are-zero.
 */

#ifndef M63C_FULL_STATE_ADJOINT_EMBEDDED
#include "fd.h"
#endif

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "exact_sh_collective.h"

#define M63C_MAX_PATCH 13
#define M63C_PATCH_CELLS (M63C_MAX_PATCH * M63C_MAX_PATCH)

#if defined(__GNUC__)
extern int visco_sh_material_timestep_vjp(
        const struct visco_sh_material_timestep_vjp_input *input,
        struct visco_sh_material_timestep_vjp_output *output)
        __attribute__((weak));
#endif

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

/* No longer called by the step itself (K-7); kept for the M6.3c-7c-b1 bridge
 * harness, which embeds this file and composes the step from its helpers. */
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

/* K-7: the receiver and source transposes act on the field directly instead
 * of on field-size double copies.  A float survives the float -> double ->
 * float round trip unchanged, so only the receiver cells change: their
 * cotangents are added in double, in receiver order, and rounded once, as
 * visco_sh_receiver_velocity_sampling_vjp did on the double copy.  On any
 * invalid receiver the field stays unchanged, as before. */
static int receiver_transpose(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *work, int row_min, int row_max,
        int col_min, int col_max) {
    int k, m, x, y;
    double value;

    if (cfg->nrec == 0) return 0;
    if ((work == NULL) || (work->vz == NULL) || (cfg->rec_x == NULL) ||
            (cfg->rec_y == NULL) || (cfg->bar_receiver == NULL)) return -1;
    for (k = 0; k < cfg->nrec; ++k) {
        x = cfg->rec_x[k];
        y = cfg->rec_y[k];
        if ((x < col_min) || (x > col_max) || (y < row_min) ||
                (y > row_max) || !isfinite(cfg->bar_receiver[k])) return -1;
    }
    for (k = 0; k < cfg->nrec; ++k) {
        x = cfg->rec_x[k];
        y = cfg->rec_y[k];
        for (m = 0; m < k; ++m)
            if ((cfg->rec_x[m] == x) && (cfg->rec_y[m] == y)) break;
        if (m < k) continue;            /* cell already done with receiver m */
        value = work->vz[y][x];
        for (m = k; m < cfg->nrec; ++m)
            if ((cfg->rec_x[m] == x) && (cfg->rec_y[m] == y))
                value += cfg->bar_receiver[m];
        work->vz[y][x] = (float)value;
    }
    return 0;
}

/* bar_before = (float)(0.0 + (double)bar_after) on the whole range, as
 * visco_sh_velocity_source_injection_vjp did on zeroed double copies; the
 * source cotangent of type-1 sources accumulates the after value. */
static int source_transpose(
        const struct visco_sh_full_step_config *cfg,
        float **bar_after, float **bar_before, int row_min, int row_max,
        int col_min, int col_max, double *bar_signal) {
    int i, j, k, x, y;

    if ((bar_after == NULL) || (bar_before == NULL) ||
            (bar_after == bar_before)) return -1;
    if ((cfg->nsrc > 0) && ((cfg->src_x == NULL) || (cfg->src_y == NULL) ||
            (cfg->source_type == NULL) || (bar_signal == NULL))) return -1;
    for (j = row_min; j <= row_max; ++j)
        for (i = col_min; i <= col_max; ++i)
            if (!isfinite(bar_after[j][i])) return -1;
    for (k = 0; k < cfg->nsrc; ++k) {
        x = cfg->src_x[k];
        y = cfg->src_y[k];
        if ((x < col_min) || (x > col_max) || (y < row_min) ||
                (y > row_max)) return -1;
    }
    for (k = 0; k < cfg->nsrc; ++k)
        if (cfg->source_type[k] == 1)
            bar_signal[k] += (double)bar_after[cfg->src_y[k]][cfg->src_x[k]];
    for (j = row_min; j <= row_max; ++j)
        for (i = col_min; i <= col_max; ++i)
            bar_before[j][i] = (float)(0.0 + (double)bar_after[j][i]);
    return 0;
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

static int validate_material_context(
        const struct visco_sh_full_step_config *cfg,
        const struct visco_sh_material_adjoint_step_context *material) {
    const struct visco_sh_native_material_gradient_fields *output;

    if (material == NULL) return 0;
#if defined(__GNUC__)
    if (visco_sh_material_timestep_vjp == NULL) return -1;
#endif
    output = material->native_output;
    if ((material->observable == NULL) ||
            (material->observable->qsum == NULL) ||
            (material->observable->strain_x == NULL) ||
            (material->observable->strain_y == NULL) ||
            (material->mu_x == NULL) || (material->tau_x == NULL) ||
            (material->mu_y == NULL) || (material->tau_y == NULL) ||
            !isfinite(material->reference_sum) ||
            (material->eta_x == NULL) || (material->eta_y == NULL) ||
            (output == NULL) || (output->g_rhoi == NULL) ||
            (output->g_mu_x == NULL) || (output->g_mu_y == NULL) ||
            (output->g_tau_x == NULL) || (output->g_tau_y == NULL)) return -1;
    if ((cfg->bip == NULL) || (cfg->bjm == NULL)) return -1;
    return 0;
}

static void zero_native_material_output(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_native_material_gradient_fields *output) {
    int i, j;
    for (j = 1; j <= cfg->ny; ++j) {
        for (i = 1; i <= cfg->nx; ++i) {
            output->g_rhoi[j][i] = 0.0;
            output->g_mu_x[j][i] = 0.0;
            output->g_mu_y[j][i] = 0.0;
            output->g_tau_x[j][i] = 0.0;
            output->g_tau_y[j][i] = 0.0;
        }
    }
}

/* K-7 per-shot plan: the time-invariant GSLS derivatives F_tau, F_modulus,
 * C_tau,l and C_modulus,l of both branches of every owned cell, computed by
 * the same visco_sh_gsls_local_derivatives call the per-step capture used to
 * make (visco_sh_material_timestep_vjp), from the same double arguments.
 * Layout per cell: f_tau_x, f_mod_x, f_tau_y, f_mod_y, then c_tau_x[L],
 * c_mod_x[L], c_tau_y[L], c_mod_y[L]. */
struct visco_sh_reverse_material_plan {
    int nx, ny, mechanisms, stride;
    float dt;
    double reference_sum;
    float **mu_x, **tau_x, **mu_y, **tau_y;
    const float *eta_x, *eta_y, *bip, *bjm;
    double *coef;
};

static int material_derivatives(
        int mechanisms, double dt, double modulus, double tau,
        double reference_sum, const double *eta, const double *b,
        double *f_tau, double *f_modulus, double *c_tau, double *c_modulus) {
    int l, status;
    status = visco_sh_gsls_local_derivatives(
            mechanisms, dt, modulus, tau, reference_sum, eta, b,
            f_tau, f_modulus, c_tau, c_modulus);
    if (status != 0) return status;
    /* the finiteness checks visco_sh_gsls_local_vjp applies to them */
    if (!isfinite(*f_tau) || !isfinite(*f_modulus)) return -1;
    for (l = 0; l < mechanisms; ++l)
        if (!isfinite(c_tau[l]) || !isfinite(c_modulus[l])) return -1;
    return 0;
}

void visco_sh_reverse_material_plan_free(
        struct visco_sh_reverse_material_plan *plan) {
    if (plan == NULL) return;
    free(plan->coef);
    free(plan);
}

int visco_sh_reverse_material_plan_create(
        const struct visco_sh_full_step_config *cfg,
        const struct visco_sh_material_adjoint_step_context *material,
        struct visco_sh_reverse_material_plan **out) {
    struct visco_sh_reverse_material_plan *plan;
    double *eta_x, *eta_y, *b_x, *b_y, *cell;
    int i, j, l, L, status = 0;

    if (out == NULL) return -1;
    *out = NULL;
    if ((cfg == NULL) || (material == NULL) || (cfg->nx < 1) ||
            (cfg->ny < 1) || (cfg->mechanisms < 1) || (cfg->bip == NULL) ||
            (cfg->bjm == NULL) || (material->mu_x == NULL) ||
            (material->tau_x == NULL) || (material->mu_y == NULL) ||
            (material->tau_y == NULL) || (material->eta_x == NULL) ||
            (material->eta_y == NULL)) return -1;
    L = cfg->mechanisms;
    plan = (struct visco_sh_reverse_material_plan *)calloc(1, sizeof(*plan));
    eta_x = (double *)calloc((size_t)4 * L, sizeof(double));
    if ((plan == NULL) || (eta_x == NULL)) {
        free(plan);
        free(eta_x);
        return MPI_ERR_NO_MEM;
    }
    eta_y = eta_x + L;
    b_x = eta_x + 2 * L;
    b_y = eta_x + 3 * L;
    plan->nx = cfg->nx;
    plan->ny = cfg->ny;
    plan->mechanisms = L;
    plan->stride = 4 + 4 * L;
    plan->dt = cfg->dt;
    plan->reference_sum = material->reference_sum;
    plan->mu_x = material->mu_x;
    plan->tau_x = material->tau_x;
    plan->mu_y = material->mu_y;
    plan->tau_y = material->tau_y;
    plan->eta_x = material->eta_x;
    plan->eta_y = material->eta_y;
    plan->bip = cfg->bip;
    plan->bjm = cfg->bjm;
    plan->coef = (double *)malloc(
            (size_t)cfg->nx * cfg->ny * plan->stride * sizeof(double));
    if (plan->coef == NULL) status = MPI_ERR_NO_MEM;
    for (l = 0; l < L; ++l) {
        eta_x[l] = material->eta_x[l + 1];
        eta_y[l] = material->eta_y[l + 1];
        b_x[l] = cfg->bip[l + 1];
        b_y[l] = cfg->bjm[l + 1];
    }
    for (j = 1; (status == 0) && (j <= cfg->ny); ++j)
        for (i = 1; (status == 0) && (i <= cfg->nx); ++i) {
            cell = plan->coef +
                    ((size_t)(j - 1) * cfg->nx + (i - 1)) * plan->stride;
            status = material_derivatives(
                    L, cfg->dt, material->mu_x[j][i], material->tau_x[j][i],
                    material->reference_sum, eta_x, b_x, &cell[0], &cell[1],
                    cell + 4, cell + 4 + L);
            if (status == 0)
                status = material_derivatives(
                        L, cfg->dt, material->mu_y[j][i],
                        material->tau_y[j][i], material->reference_sum,
                        eta_y, b_y, &cell[2], &cell[3], cell + 4 + 2 * L,
                        cell + 4 + 3 * L);
        }
    free(eta_x);
    if (status != 0) {
        visco_sh_reverse_material_plan_free(plan);
        return status;
    }
    *out = plan;
    return 0;
}

static int plan_matches(
        const struct visco_sh_reverse_material_plan *plan,
        const struct visco_sh_full_step_config *cfg,
        const struct visco_sh_material_adjoint_step_context *material) {
    return (plan->nx == cfg->nx) && (plan->ny == cfg->ny) &&
            (plan->mechanisms == cfg->mechanisms) && (plan->dt == cfg->dt) &&
            (plan->reference_sum == material->reference_sum) &&
            (plan->mu_x == material->mu_x) && (plan->tau_x == material->tau_x) &&
            (plan->mu_y == material->mu_y) && (plan->tau_y == material->tau_y) &&
            (plan->eta_x == material->eta_x) &&
            (plan->eta_y == material->eta_y) && (plan->bip == cfg->bip) &&
            (plan->bjm == cfg->bjm);
}

/* Native material sensitivities of the stress update, the stress part of
 * visco_sh_material_timestep_vjp evaluated per owned cell: per branch
 *   tau = F_tau s + sum_l C_tau,l t_l,  modulus = F_mod s + sum_l C_mod,l t_l,
 *   t_l = bar_memory_l + dt/2 s,  g = 0.0 + strain * value,
 * with the same double operations in the same order.  Terms the former
 * capture computed and discarded (g_rhoi with a zero cotangent, the strain
 * and state cotangents) are not formed; the checks of their inputs that the
 * same step repeats on identical values (bar stress and memory, F, C in
 * reverse_stress_block; the material derivatives) are not repeated here. */
static int capture_material_stress_cotangents(
        const struct visco_sh_full_step_config *cfg,
        const struct visco_sh_full_state *work,
        const struct visco_sh_material_adjoint_step_context *material) {
    const struct visco_sh_reverse_material_plan *plan = material->plan;
    struct visco_sh_native_material_gradient_fields *out =
            material->native_output;
    double *scratch = NULL, *eta_x, *eta_y, *b_x, *b_y, *coef;
    double dt = cfg->dt, dh = cfg->dh, half_dt, s, strain, t, tau_v, mod_v;
    double g_mu_x, g_mu_y, g_tau_x, g_tau_y;
    const double *c_tau, *c_mod;
    float *rx, *qy;
    int i, j, l, L = cfg->mechanisms, status = -1;

    if ((L < 1) || !(dt > 0.0) || !(dh > 0.0) || !isfinite(dt) ||
            !isfinite(dh) || !isfinite(dt / dh)) return -1;
    if ((plan != NULL) && !plan_matches(plan, cfg, material)) return -1;
    scratch = (double *)calloc((size_t)8 * L + 4, sizeof(double));
    if (scratch == NULL) return -1;
    eta_x = scratch;
    eta_y = scratch + L;
    b_x = scratch + 2 * L;
    b_y = scratch + 3 * L;
    for (l = 0; l < L; ++l) {
        eta_x[l] = material->eta_x[l + 1];
        eta_y[l] = material->eta_y[l + 1];
        b_x[l] = cfg->bip[l + 1];
        b_y[l] = cfg->bjm[l + 1];
    }
    half_dt = 0.5 * dt;
    for (j = 1; j <= cfg->ny; ++j) {
        for (i = 1; i <= cfg->nx; ++i) {
            if (plan != NULL) {
                coef = plan->coef +
                        ((size_t)(j - 1) * cfg->nx + (i - 1)) * plan->stride;
            } else {
                coef = scratch + 4 * L;     /* 4 + 4L doubles */
                status = material_derivatives(
                        L, dt, material->mu_x[j][i], material->tau_x[j][i],
                        material->reference_sum, eta_x, b_x, &coef[0],
                        &coef[1], coef + 4, coef + 4 + L);
                if (status == 0)
                    status = material_derivatives(
                            L, dt, material->mu_y[j][i],
                            material->tau_y[j][i], material->reference_sum,
                            eta_y, b_y, &coef[2], &coef[3], coef + 4 + 2 * L,
                            coef + 4 + 3 * L);
                if (status != 0) goto cleanup;
            }
            if (!isfinite(material->observable->qsum[j][i]) ||
                    !isfinite(material->observable->strain_x[j][i]) ||
                    !isfinite(material->observable->strain_y[j][i])) {
                status = -1;
                goto cleanup;
            }
            /* x branch: sxz with memory r */
            s = work->sxz[j][i];
            strain = material->observable->strain_x[j][i];
            rx = work->r[j][i];
            c_tau = coef + 4;
            c_mod = coef + 4 + L;
            tau_v = coef[0] * s;
            mod_v = coef[1] * s;
            for (l = 0; l < L; ++l) {
                t = (double)rx[l + 1] + half_dt * s;
                tau_v += c_tau[l] * t;
                mod_v += c_mod[l] * t;
            }
            g_tau_x = 0.0 + strain * tau_v;
            g_mu_x = 0.0 + strain * mod_v;
            /* y branch: syz with memory q */
            s = work->syz[j][i];
            strain = material->observable->strain_y[j][i];
            qy = work->q[j][i];
            c_tau = coef + 4 + 2 * L;
            c_mod = coef + 4 + 3 * L;
            tau_v = coef[2] * s;
            mod_v = coef[3] * s;
            for (l = 0; l < L; ++l) {
                t = (double)qy[l + 1] + half_dt * s;
                tau_v += c_tau[l] * t;
                mod_v += c_mod[l] * t;
            }
            g_tau_y = 0.0 + strain * tau_v;
            g_mu_y = 0.0 + strain * mod_v;
            if (material->accumulate) {
                out->g_mu_x[j][i] += g_mu_x;
                out->g_mu_y[j][i] += g_mu_y;
                out->g_tau_x[j][i] += g_tau_x;
                out->g_tau_y[j][i] += g_tau_y;
            } else {
                out->g_mu_x[j][i] = g_mu_x;
                out->g_mu_y[j][i] = g_mu_y;
                out->g_tau_x[j][i] = g_tau_x;
                out->g_tau_y[j][i] = g_tau_y;
            }
        }
    }
    status = 0;
cleanup:
    free(scratch);
    return status;
}

/* Native density sensitivity of the velocity update,
 * visco_sh_velocity_rhoi_vjp(dt, dh, qsum, 0.0, bar_v) =
 * (dt / dh) * (qsum + 0.0) * bar_v, per owned cell.  The two GSLS branches
 * the former capture evaluated here had zero cotangents and were discarded;
 * their input checks are the ones the stress capture and the stress update
 * of the same step apply to identical values. */
static int capture_material_velocity_cotangent(
        const struct visco_sh_full_step_config *cfg,
        const struct visco_sh_full_state *work,
        const struct visco_sh_material_adjoint_step_context *material) {
    struct visco_sh_native_material_gradient_fields *out =
            material->native_output;
    double dt = cfg->dt, dh = cfg->dh, ratio, g;
    int i, j;

    if ((cfg->mechanisms < 1) || !(dt > 0.0) || !(dh > 0.0) ||
            !isfinite(dh)) return -1;
    ratio = dt / dh;
    for (j = 1; j <= cfg->ny; ++j) {
        for (i = 1; i <= cfg->nx; ++i) {
            g = ratio * ((double)material->observable->qsum[j][i] + 0.0) *
                    (double)work->vz[j][i];
            if (!isfinite(g)) return -1;
            if (material->accumulate) out->g_rhoi[j][i] += g;
            else out->g_rhoi[j][i] = g;
        }
    }
    return 0;
}

/* CPML selection depends on i or j alone: it is evaluated once per column
 * and row of the step, with the same selector functions. */
struct cpml_axis {
    int *active, *aux;
    double *K, *a, *b;
};

static int cpml_axis_alloc(struct cpml_axis *axis, int n) {
    axis->active = (int *)calloc((size_t)2 * (n + 1), sizeof(int));
    axis->K = (double *)calloc((size_t)3 * (n + 1), sizeof(double));
    if ((axis->active == NULL) || (axis->K == NULL)) {
        free(axis->active);
        free(axis->K);
        axis->active = NULL;
        axis->K = NULL;
        return -1;
    }
    axis->aux = axis->active + (n + 1);
    axis->a = axis->K + (n + 1);
    axis->b = axis->K + 2 * (n + 1);
    return 0;
}

static void cpml_axis_free(struct cpml_axis *axis) {
    free(axis->active);
    free(axis->K);
}

/* Adds +0.0f to every cell of rows 1-h..ny+h, columns 1-h..nx+h.  This is
 * the only effect of the patch entries outside the stencil cross in the
 * dense patch scatter: x + (+0.0f) == x for every x except -0.0f, which
 * becomes +0.0f, and once a running sum is not -0.0f no addition can make it
 * -0.0f again.  Hence a cell that receives at least one such +0.0f at any
 * time ends as the sum of its cross contributions with -0.0f replaced by
 * +0.0f.  For nx >= 2 and ny >= 2 every cell of the range lies outside the
 * cross of at least one window, for both the stress stencil and the two
 * velocity stencils. */
static void add_positive_zero(float **field, int nx, int ny, int h) {
    int i, j;
    float *row;
    for (j = 1 - h; j <= ny + h; ++j) {
        row = field[j];
        for (i = 1 - h; i <= nx + h; ++i) row[i] += 0.0f;
    }
}

/* Owned-cell loop of reverse_stress_block for nx, ny >= 2: an exact
 * transcription of update_s_visc_PML_SH_adjoint_point for the inputs this
 * block passes (strain 0, f_tau = f_modulus = 0, c_tau = c_modulus = 0),
 * with the patch scatter split into its cross contributions and
 * add_positive_zero.  Every double operation, its operands and its order
 * are those of visco_sh_gsls_local_vjp, visco_sh_stress_cpml_local_vjp and
 * visco_sh_stress_spatial_local_vjp:
 *   t_l = m_l + dt/2 s,  bar_m_l = 0.0 + (a_l t_l + dt/2 s),
 *   e = 0.0 + (F s + sum_l C_l t_l),  bar_s = 0.0 + s,
 *   CPML: t = psi + e, bar_psi = 0.0 + b t, raw = 0.0 + (e/K + a t),
 *   inactive: raw = 0.0 + e,
 *   patch: x line +-(hc_m/dh) raw_x, y line +-(hc_m/dh) raw_y, centre
 *   (0.0 - s_1 raw_x) - s_1 raw_y.
 * The zero-weight material terms g = 0.0 + 0.0 * (0 s + sum 0 t_l) are
 * nonzero exactly when some t_l is not finite; that case keeps status -3,
 * as in the base.  With the value checks passed, t_l comes from finite
 * floats and cannot overflow in double, so -3 is not reached in practice.
 * The checks are the same value checks; time-invariant ones (dt, dh, hc,
 * a_l, CPML profiles in the selectors) are made once per call. */
static int stress_block_cells(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *work, struct visco_sh_full_state *prev,
        const struct cpml_axis *sx, const struct cpml_axis *sy,
        const double *ax, const double *ay) {
    double scale[M63C_MAX_PATCH];
    double dt = cfg->dt, dh = cfg->dh, half_dt;
    double s0, s1, v0, v1, m, c, t, e0, e1, tp, r0, r1, centre;
    int h = cfg->fdorder / 2, L = cfg->mechanisms;
    int i, j, l, k, bad;
    float *rx, *qy, *px, *qp, *prx, *pqy, *row;

    if (!isfinite(dt) || !(dt > 0.0) || !(dh > 0.0) || !isfinite(dh))
        return -1;
    for (k = 1; k <= h; ++k) {
        if (!isfinite(cfg->hc[k])) return -1;
        scale[k] = (double)cfg->hc[k] / dh;
    }
    for (l = 0; l < L; ++l)
        if (!isfinite(ax[l]) || !isfinite(ay[l])) return -1;
    half_dt = 0.5 * dt;

    for (j = 1; j <= cfg->ny; ++j) {
        for (i = 1; i <= cfg->nx; ++i) {
            s0 = work->sxz[j][i];
            s1 = work->syz[j][i];
            v0 = cfg->fipjp[j][i];
            v1 = cfg->f[j][i];
            if (!isfinite(s0) || !isfinite(v0) || !isfinite(s1) ||
                    !isfinite(v1)) return -1;
            rx = work->r[j][i];
            qy = work->q[j][i];
            px = cfg->dip[j][i];
            qp = cfg->d[j][i];
            prx = prev->r[j][i];
            pqy = prev->q[j][i];
            bad = 0;
            /* x branch: sxz, memory r */
            v0 = v0 * s0;
            for (l = 0; l < L; ++l) {
                m = rx[l + 1];
                c = -(double)cfg->bip[l + 1] * px[l + 1];
                if (!isfinite(m) || !isfinite(c)) return -1;
                t = m + half_dt * s0;
                prx[l + 1] = (float)(0.0 + (ax[l] * t + half_dt * s0));
                v0 += c * t;
                bad |= !isfinite(t);
            }
            e0 = 0.0 + v0;
            /* y branch: syz, memory q */
            v1 = v1 * s1;
            for (l = 0; l < L; ++l) {
                m = qy[l + 1];
                c = -(double)cfg->bjm[l + 1] * qp[l + 1];
                if (!isfinite(m) || !isfinite(c)) return -1;
                t = m + half_dt * s1;
                pqy[l + 1] = (float)(0.0 + (ay[l] * t + half_dt * s1));
                v1 += c * t;
                bad |= !isfinite(t);
            }
            e1 = 0.0 + v1;
            /* CPML memory transposes */
            if (!isfinite(e0) || !isfinite(e1)) return -1;
            if (sx->active[i]) {
                tp = (double)work->psi_vzx[j][sx->aux[i]];
                if (!isfinite(tp)) return -1;
                tp = tp + e0;
                prev->psi_vzx[j][sx->aux[i]] = (float)(0.0 + sx->b[i] * tp);
                r0 = 0.0 + (e0 / sx->K[i] + sx->a[i] * tp);
            } else {
                r0 = 0.0 + e0;
            }
            if (sy->active[j]) {
                tp = (double)work->psi_vzy[sy->aux[j]][i];
                if (!isfinite(tp)) return -1;
                tp = tp + e1;
                prev->psi_vzy[sy->aux[j]][i] = (float)(0.0 + sy->b[j] * tp);
                r1 = 0.0 + (e1 / sy->K[j] + sy->a[j] * tp);
            } else {
                r1 = 0.0 + e1;
            }
            if (!isfinite(r0) || !isfinite(r1)) return -1;
            prev->sxz[j][i] = (float)(0.0 + s0);
            prev->syz[j][i] = (float)(0.0 + s1);
            if (bad) return -3;
            /* cross contributions of the spatial transpose */
            row = work->vz[j];
            centre = (0.0 - scale[1] * r0) - scale[1] * r1;
            for (k = 1; k <= h; ++k) {
                row[i + k] += (float)(0.0 + scale[k] * r0);
                work->vz[j + k][i] += (float)(0.0 + scale[k] * r1);
            }
            for (k = 2; k <= h; ++k) {
                row[i - (k - 1)] += (float)(0.0 - scale[k] * r0);
                work->vz[j - (k - 1)][i] += (float)(0.0 - scale[k] * r1);
            }
            row[i] += (float)centre;
        }
    }
    add_positive_zero(work->vz, cfg->nx, cfg->ny, h);
    return 0;
}

/* Owned-cell loop of reverse_velocity_block for nx, ny >= 2: an exact
 * transcription of update_v_PML_SH_adjoint_point,
 *   alpha = dt rhoi / dh,  bar_v = 0.0 + v,  q = alpha v,
 *   CPML as in the stress block with q in place of e,
 *   patch_x[c+(m-1)] = 0.0 + hc_m raw_x, patch_x[c-m] = 0.0 - hc_m raw_x,
 *   patch_y likewise along y,
 * with the patch scatter split into its line contributions and
 * add_positive_zero on sxz and syz. */
static int velocity_block_cells(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *work, struct visco_sh_full_state *prev,
        const struct cpml_axis *vx, const struct cpml_axis *vy) {
    double scale[M63C_MAX_PATCH];
    double dt = cfg->dt, dh = cfg->dh, v, q, tp, r0, r1, alpha;
    int h = cfg->fdorder / 2;
    int i, j, k;
    float rhoi, *row;

    if (!(dt > 0.0) || !(dh > 0.0) || !isfinite(dt) || !isfinite(dh))
        return -1;
    for (k = 1; k <= h; ++k) {
        if (!isfinite(cfg->hc[k])) return -1;
        scale[k] = (double)cfg->hc[k];
    }
    for (j = 1; j <= cfg->ny; ++j) {
        for (i = 1; i <= cfg->nx; ++i) {
            rhoi = cfg->rhoi[j][i];
            v = work->vz[j][i];
            if (!(rhoi > 0.0f) || !isfinite(rhoi) || !isfinite(v)) return -1;
            alpha = dt * (double)rhoi / dh;
            q = alpha * v;
            if (!isfinite(q)) return -1;
            if (vx->active[i]) {
                tp = (double)work->psi_sxz_x[j][vx->aux[i]];
                if (!isfinite(tp)) return -1;
                tp = tp + q;
                prev->psi_sxz_x[j][vx->aux[i]] = (float)(0.0 + vx->b[i] * tp);
                r0 = 0.0 + (q / vx->K[i] + vx->a[i] * tp);
            } else {
                r0 = 0.0 + q;
            }
            if (vy->active[j]) {
                tp = (double)work->psi_syz_y[vy->aux[j]][i];
                if (!isfinite(tp)) return -1;
                tp = tp + q;
                prev->psi_syz_y[vy->aux[j]][i] = (float)(0.0 + vy->b[j] * tp);
                r1 = 0.0 + (q / vy->K[j] + vy->a[j] * tp);
            } else {
                r1 = 0.0 + q;
            }
            if (!isfinite(r0) || !isfinite(r1)) return -1;
            prev->vz[j][i] = (float)(0.0 + v);
            row = prev->sxz[j];
            for (k = 1; k <= h; ++k) {
                row[i + (k - 1)] += (float)(0.0 + scale[k] * r0);
                row[i - k] += (float)(0.0 - scale[k] * r0);
                prev->syz[j + (k - 1)][i] += (float)(0.0 + scale[k] * r1);
                prev->syz[j - k][i] += (float)(0.0 - scale[k] * r1);
            }
        }
    }
    add_positive_zero(prev->sxz, cfg->nx, cfg->ny, h);
    add_positive_zero(prev->syz, cfg->nx, cfg->ny, h);
    return 0;
}

static int reverse_stress_block(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *work,
        struct visco_sh_full_state *prev) {
    double patch[M63C_PATCH_CELLS];
    struct cpml_axis sx = {0}, sy = {0};
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
    int patch_cells = patch_size * patch_size;
    int i, j, l, pi, pj, status = -1;
    float *row, *rx, *qy, *px, *qp;

    bar_rx_next = (double *)calloc((size_t)9 * cfg->mechanisms, sizeof(double));
    if ((bar_rx_next == NULL) || (cpml_axis_alloc(&sx, cfg->nx) != 0) ||
            (cpml_axis_alloc(&sy, cfg->ny) != 0)) goto cleanup;
    bar_qy_next = bar_rx_next + cfg->mechanisms;
    bar_rx_prev = bar_rx_next + 2 * cfg->mechanisms;
    bar_qy_prev = bar_rx_next + 3 * cfg->mechanisms;
    ax = bar_rx_next + 4 * cfg->mechanisms;
    ay = bar_rx_next + 5 * cfg->mechanisms;
    cx = bar_rx_next + 6 * cfg->mechanisms;
    cy = bar_rx_next + 7 * cfg->mechanisms;
    zero_l = bar_rx_next + 8 * cfg->mechanisms;
    for (l = 0; l < cfg->mechanisms; ++l) {
        ax[l] = (double)cfg->bip[l + 1] * cfg->cip[l + 1];
        ay[l] = (double)cfg->bjm[l + 1] * cfg->cjm[l + 1];
    }
    for (i = 1; i <= cfg->nx; ++i) {
        status = visco_sh_stress_cpml_select_x(
                i, cfg->nx, cfg->fw, cfg->boundary, cfg->pos[1],
                cfg->nproc_x, cfg->K_x_half, cfg->a_x_half,
                cfg->b_x_half, &sx.active[i], &sx.aux[i],
                &sx.K[i], &sx.a[i], &sx.b[i]);
        if (status != 0) goto cleanup;
    }
    for (j = 1; j <= cfg->ny; ++j) {
        status = visco_sh_stress_cpml_select_y(
                j, cfg->ny, cfg->fw, cfg->free_surface, cfg->pos[2],
                cfg->nproc_y, cfg->K_y, cfg->a_y, cfg->b_y,
                cfg->K_y_half, cfg->a_y_half, cfg->b_y_half,
                &sy.active[j], &sy.aux[j], &sy.K[j], &sy.a[j], &sy.b[j]);
        if (status != 0) goto cleanup;
    }
    if ((cfg->nx >= 2) && (cfg->ny >= 2)) {
        status = stress_block_cells(cfg, work, prev, &sx, &sy, ax, ay);
        goto cleanup;
    }

    /* Subdomains narrower than two cells: point VJP with the dense patch. */
    for (j = 1; j <= cfg->ny; ++j) {
        cpml_active[1] = sy.active[j];
        aux_y = sy.aux[j];
        cpml_K[1] = sy.K[j];
        cpml_a[1] = sy.a[j];
        cpml_b[1] = sy.b[j];
        for (i = 1; i <= cfg->nx; ++i) {
            cpml_active[0] = sx.active[i];
            aux_x = sx.aux[i];
            cpml_K[0] = sx.K[i];
            cpml_a[0] = sx.a[i];
            cpml_b[0] = sx.b[i];

            bar_stress[0] = work->sxz[j][i];
            bar_stress[1] = work->syz[j][i];
            forward_f[0] = cfg->fipjp[j][i];
            forward_f[1] = cfg->f[j][i];
            rx = work->r[j][i];
            qy = work->q[j][i];
            px = cfg->dip[j][i];
            qp = cfg->d[j][i];
            for (l = 0; l < cfg->mechanisms; ++l) {
                bar_rx_next[l] = rx[l + 1];
                bar_qy_next[l] = qy[l + 1];
                bar_rx_prev[l] = 0.0;
                bar_qy_prev[l] = 0.0;
                cx[l] = -(double)cfg->bip[l + 1] * px[l + 1];
                cy[l] = -(double)cfg->bjm[l + 1] * qp[l + 1];
            }
            bar_psi_next[0] = cpml_active[0] ?
                    work->psi_vzx[j][aux_x] : 0.0;
            bar_psi_next[1] = cpml_active[1] ?
                    work->psi_vzy[aux_y][i] : 0.0;
            bar_psi_prev[0] = bar_psi_prev[1] = 0.0;
            bar_stress_prev[0] = bar_stress_prev[1] = 0.0;
            g_tau[0] = g_tau[1] = 0.0;
            g_modulus[0] = g_modulus[1] = 0.0;
            /* the point VJP writes only inside the patch_size^2 window */
            for (pi = 0; pi < patch_cells; ++pi) patch[pi] = 0.0;

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
            for (pj = 0; pj < patch_size; ++pj) {
                row = work->vz[j + pj - h] + (i - h);
                for (pi = 0; pi < patch_size; ++pi)
                    row[pi] += (float)patch[pj * patch_size + pi];
            }
        }
    }
    status = 0;
cleanup:
    free(bar_rx_next);
    cpml_axis_free(&sx);
    cpml_axis_free(&sy);
    return status;
}

static int reverse_velocity_block(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *work,
        struct visco_sh_full_state *prev) {
    double patch_x[M63C_PATCH_CELLS], patch_y[M63C_PATCH_CELLS];
    double bar_psi_next[2], bar_psi_prev[2], cpml_K[2];
    double cpml_a[2], cpml_b[2], bar_vz_prev;
    struct cpml_axis vx = {0}, vy = {0};
    int cpml_active[2], aux_x, aux_y;
    int h = cfg->fdorder / 2, patch_size = 2 * h + 1;
    int patch_cells = patch_size * patch_size;
    int i, j, pi, pj, status = -1;
    float *row_x, *row_y;

    if ((cpml_axis_alloc(&vx, cfg->nx) != 0) ||
            (cpml_axis_alloc(&vy, cfg->ny) != 0)) goto cleanup;
    for (i = 1; i <= cfg->nx; ++i) {
        status = visco_sh_velocity_cpml_select_x(
                i, cfg->nx, cfg->fw, cfg->boundary, cfg->pos[1],
                cfg->nproc_x, cfg->K_x, cfg->a_x, cfg->b_x,
                &vx.active[i], &vx.aux[i], &vx.K[i], &vx.a[i], &vx.b[i]);
        if (status != 0) goto cleanup;
    }
    for (j = 1; j <= cfg->ny; ++j) {
        status = visco_sh_velocity_cpml_select_y(
                j, cfg->ny, cfg->fw, cfg->free_surface, cfg->pos[2],
                cfg->nproc_y, cfg->K_y, cfg->a_y, cfg->b_y,
                &vy.active[j], &vy.aux[j], &vy.K[j], &vy.a[j], &vy.b[j]);
        if (status != 0) goto cleanup;
    }
    if ((cfg->nx >= 2) && (cfg->ny >= 2)) {
        status = velocity_block_cells(cfg, work, prev, &vx, &vy);
        goto cleanup;
    }

    /* Subdomains narrower than two cells: point VJP with the dense patch. */
    for (j = 1; j <= cfg->ny; ++j) {
        cpml_active[1] = vy.active[j];
        aux_y = vy.aux[j];
        cpml_K[1] = vy.K[j];
        cpml_a[1] = vy.a[j];
        cpml_b[1] = vy.b[j];
        for (i = 1; i <= cfg->nx; ++i) {
            cpml_active[0] = vx.active[i];
            aux_x = vx.aux[i];
            cpml_K[0] = vx.K[i];
            cpml_a[0] = vx.a[i];
            cpml_b[0] = vx.b[i];
            bar_psi_next[0] = cpml_active[0] ?
                    work->psi_sxz_x[j][aux_x] : 0.0;
            bar_psi_next[1] = cpml_active[1] ?
                    work->psi_syz_y[aux_y][i] : 0.0;
            bar_psi_prev[0] = bar_psi_prev[1] = 0.0;
            bar_vz_prev = 0.0;
            /* the point VJP writes only inside the patch_size^2 window */
            for (pi = 0; pi < patch_cells; ++pi) {
                patch_x[pi] = 0.0;
                patch_y[pi] = 0.0;
            }
            status = update_v_PML_SH_adjoint_point(
                    cfg->fdorder, cfg->dt, cfg->dh, cfg->rhoi[j][i],
                    cfg->hc, cpml_active, cpml_K, cpml_a, cpml_b,
                    work->vz[j][i], bar_psi_next, &bar_vz_prev,
                    bar_psi_prev, patch_x, patch_y, patch_size, patch_size,
                    h, h);
            if (status != 0) goto cleanup;
            prev->vz[j][i] = (float)bar_vz_prev;
            if (cpml_active[0])
                prev->psi_sxz_x[j][aux_x] = (float)bar_psi_prev[0];
            if (cpml_active[1])
                prev->psi_syz_y[aux_y][i] = (float)bar_psi_prev[1];
            for (pj = 0; pj < patch_size; ++pj) {
                row_x = prev->sxz[j + pj - h] + (i - h);
                row_y = prev->syz[j + pj - h] + (i - h);
                for (pi = 0; pi < patch_size; ++pi) {
                    row_x[pi] += (float)patch_x[pj * patch_size + pi];
                    row_y[pi] += (float)patch_y[pj * patch_size + pi];
                }
            }
        }
    }
    status = 0;
cleanup:
    cpml_axis_free(&vx);
    cpml_axis_free(&vy);
    return status;
}

/* A peer can already be in a halo exchange when a local point operation
 * fails. Terminate that distributed material step through its communicator. */
static int exact_sh_reverse_step_failure(const struct visco_sh_full_step_config *cfg,
        const struct visco_sh_material_adjoint_step_context *material,
        const char *phase, int status) {
    if (cfg != NULL && material != NULL &&
            (cfg->nproc_x > 1 || cfg->nproc_y > 1))
        return exact_sh_abort_phase(cfg->comm, phase, status);
    return status;
}

static int visco_sh_full_state_adjoint_step_internal(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *bar_next_work,
        struct visco_sh_full_state *bar_prev,
        double *bar_signal,
        const struct visco_sh_material_adjoint_step_context *material) {
    int h, row_min, row_max, col_min, col_max, status;

    status = validate_config(cfg);
    if (status != 0) return exact_sh_reverse_step_failure(cfg, material,
            "reverse step configuration", status);
    status = validate_material_context(cfg, material);
    if (status != 0) return exact_sh_reverse_step_failure(cfg, material,
            "reverse step material capture", status);
    if (!state_complete(bar_next_work, cfg->fw) ||
            !state_complete(bar_prev, cfg->fw) ||
            !state_distinct(bar_next_work, bar_prev, cfg->fw) ||
            ((cfg->nsrc > 0) && (bar_signal == NULL)))
        return exact_sh_reverse_step_failure(cfg, material, "reverse step workspace", -1);

    h = cfg->fdorder / 2;
    row_min = -h;
    row_max = cfg->ny + h + 1;
    col_min = 1 - h;
    col_max = cfg->nx + h;

    /* K-7: bar_prev->vz is overwritten on the whole range by
     * source_transpose and bar_prev->sxz/syz by the copies before the stress
     * update, with nothing reading them in between, so they are not zeroed
     * first.  The memory variables of owned cells are overwritten by
     * reverse_stress_block; only the halo ring is copied. */
    copy_memory(bar_prev->r, bar_next_work->r,
                row_min, 0, col_min, col_max, cfg->mechanisms);
    copy_memory(bar_prev->r, bar_next_work->r,
                cfg->ny + 1, row_max, col_min, col_max, cfg->mechanisms);
    copy_memory(bar_prev->r, bar_next_work->r,
                1, cfg->ny, col_min, 0, cfg->mechanisms);
    copy_memory(bar_prev->r, bar_next_work->r,
                1, cfg->ny, cfg->nx + 1, col_max, cfg->mechanisms);
    copy_memory(bar_prev->q, bar_next_work->q,
                row_min, 0, col_min, col_max, cfg->mechanisms);
    copy_memory(bar_prev->q, bar_next_work->q,
                cfg->ny + 1, row_max, col_min, col_max, cfg->mechanisms);
    copy_memory(bar_prev->q, bar_next_work->q,
                1, cfg->ny, col_min, 0, cfg->mechanisms);
    copy_memory(bar_prev->q, bar_next_work->q,
                1, cfg->ny, cfg->nx + 1, col_max, cfg->mechanisms);
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
    if ((material != NULL) && !material->accumulate)
        zero_native_material_output(cfg, material->native_output);

    status = receiver_transpose(
            cfg, bar_next_work, row_min, row_max, col_min, col_max);
    if (status != 0) return exact_sh_reverse_step_failure(cfg, material,
            "receiver transpose", status);
    status = exchange_s_SH_adjoint(
            bar_next_work->sxz, bar_next_work->syz, cfg->nx, cfg->ny,
            cfg->fdorder, cfg->boundary, cfg->pos, cfg->nproc_x,
            cfg->nproc_y, cfg->index, cfg->comm);
    if (status != MPI_SUCCESS) return exact_sh_reverse_step_failure(cfg, material,
            "reverse stress halo", status);
    if (cfg->free_surface && (cfg->pos[2] == 0))
        surface_elastic_SH_stress_adjoint(
                bar_next_work->syz, cfg->nx, h);

    if (material != NULL) {
        status = capture_material_stress_cotangents(
                cfg, bar_next_work, material);
        if (status != 0) return exact_sh_reverse_step_failure(cfg, material,
                "reverse stress capture", status);
    }

    /* The stress halo/surface maps precede the constitutive update in
     * reverse.  Only now is their transformed identity branch the input to
     * the stress-point VJP. */
    copy_field(bar_prev->sxz, bar_next_work->sxz,
               row_min, row_max, col_min, col_max);
    copy_field(bar_prev->syz, bar_next_work->syz,
               row_min, row_max, col_min, col_max);

    status = reverse_stress_block(cfg, bar_next_work, bar_prev);
    if (status != 0) return exact_sh_reverse_step_failure(cfg, material,
            "reverse stress update", status);

    if (cfg->free_surface && (cfg->pos[2] == 0))
        surface_elastic_SH_velocity_adjoint(
                bar_next_work->vz, cfg->nx, h);
    status = exchange_v_SH_adjoint(
            bar_next_work->vz, cfg->nx, cfg->ny, cfg->fdorder,
            cfg->boundary, cfg->pos, cfg->nproc_x, cfg->nproc_y,
            cfg->index, cfg->comm);
    if (status != MPI_SUCCESS) return exact_sh_reverse_step_failure(cfg, material,
            "reverse velocity halo", status);

    /* Source^T and velocity-update^T both consume this same post-source
     * cotangent.  source_transpose copies it to prev; physical cells are then
     * overwritten by the exact velocity point VJP, whose identity term uses
     * the unchanged work value. */
    status = source_transpose(
            cfg, bar_next_work->vz, bar_prev->vz, row_min, row_max,
            col_min, col_max, bar_signal);
    if (status != 0) return exact_sh_reverse_step_failure(cfg, material,
            "source transpose", status);
    if (material != NULL) {
        status = capture_material_velocity_cotangent(
                cfg, bar_next_work, material);
        if (status != 0) return exact_sh_reverse_step_failure(cfg, material,
                "reverse velocity capture", status);
    }
    return reverse_velocity_block(cfg, bar_next_work, bar_prev);
}

int visco_sh_full_state_adjoint_step(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *bar_next_work,
        struct visco_sh_full_state *bar_prev,
        double *bar_signal) {
    return visco_sh_full_state_adjoint_step_internal(
            cfg, bar_next_work, bar_prev, bar_signal, NULL);
}

int visco_sh_full_state_adjoint_step_material(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *bar_next_work,
        struct visco_sh_full_state *bar_prev,
        double *bar_signal,
        const struct visco_sh_material_adjoint_step_context *material) {
    int status;

    if (material == NULL) return -1;
    status = visco_sh_full_state_adjoint_step_internal(
            cfg, bar_next_work, bar_prev, bar_signal, material);
    /* In a distributed material step every earlier failure has already
     * aborted at its own phase, so a nonzero status reaching this point
     * comes from the final reverse velocity update. */
    if (status != 0) return exact_sh_reverse_step_failure(cfg, material,
            "reverse velocity update", status);
    return 0;
}
