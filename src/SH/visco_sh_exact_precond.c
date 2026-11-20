/* SH-PRECOND-01: illumination preconditioning (diagonal pseudo-Hessian
 * approximation) and adaptive initial line-search step at the exact
 * viscoelastic SH optimizer boundary.
 * Contract: docs/sh_precond_01_contract.md (revision 2).
 * Gradient, objective, line-search algorithm and trial construction stay
 * unchanged; this unit only builds a positive diagonal P and applies it in
 * front of the SH-OPT-SCALE-01 class scaling. It does not call err(). */
#include "fd.h"

/* 2.1: energy[j][i] += field^2 on own cells (binary32). */
void visco_sh_exact_accumulate_energy(float **energy, float **field,
                                      int nx, int ny) {
    int j;

    if ((energy == NULL) || (field == NULL) || (nx < 1) || (ny < 1)) return;
    for (j = 1; j <= ny; ++j) {
        if ((energy[j] == NULL) || (field[j] == NULL)) return;
    }
    visco_sh_exact_energy_step(energy, field, nx, ny);
}

/* 2.2: receiver-line term in global grid coordinates (local, not collective). */
int visco_sh_exact_build_receiver_line_term(float **term, int nx, int ny,
        int pos_x, int pos_y, double x_min, double x_max) {
    int i, j;
    double x, z;

    if ((term == NULL) || (nx < 1) || (ny < 1) || (pos_x < 0) || (pos_y < 0))
        return -1;
    if (!isfinite(x_min) || !isfinite(x_max) || !(x_min < x_max)) return -1;
    for (j = 1; j <= ny; ++j)
        if (term[j] == NULL) return -1;
    for (j = 1; j <= ny; ++j) {
        z = (double)(j + pos_y * ny);
        for (i = 1; i <= nx; ++i) {
            x = (double)(i + pos_x * nx);
            term[j][i] = (float)(asinh((x_max - x) / z) -
                                 asinh((x_min - x) / z));
        }
    }
    return 0;
}

static int precond_rows_valid(float **field, int ny) {
    int j;

    if (field == NULL) return 0;
    for (j = 1; j <= ny; ++j)
        if (field[j] == NULL) return 0;
    return 1;
}

static double precond_weight(int mode, float source_energy, float receiver) {
    if (mode == 1) return (double)source_energy;
    return sqrt((double)source_energy) * (double)receiver;
}

/* Local priority of contract 2.3 (11..14 or 99 = ok); also the local masked
 * count and the local maximum of W over masked cells. */
static int precond_local_priority(
        const struct visco_sh_exact_precond_request *r,
        long *count, double *w_max) {
    int i, j;
    float m, ws, rr;
    double w;

    *count = 0;
    *w_max = 0.0;
    if (r == NULL) return 11;
    if ((r->nx < 1) || (r->ny < 1)) return 11;
    if (!precond_rows_valid(r->source_energy, r->ny) ||
            !precond_rows_valid(r->update_mask, r->ny) ||
            !precond_rows_valid(r->precond, r->ny)) return 11;
    if ((r->precond_mode == 2) &&
            !precond_rows_valid(r->receiver_term, r->ny)) return 11;
    if ((r->precond_mode != 1) && (r->precond_mode != 2)) return 12;
    if (!isfinite(r->epsilon) || (r->epsilon <= 0.0f) ||
            (r->epsilon > 1.0f)) return 12;
    for (j = 1; j <= r->ny; ++j) {
        for (i = 1; i <= r->nx; ++i) {
            m = r->update_mask[j][i];
            if ((m != 0.0f) && (m != 1.0f)) return 13;
            if (m != 1.0f) continue;
            ws = r->source_energy[j][i];
            if (!isfinite(ws) || (ws < 0.0f)) return 13;
            rr = 0.0f;
            if (r->precond_mode == 2) {
                rr = r->receiver_term[j][i];
                if (!isfinite(rr) || (rr <= 0.0f)) return 13;
            }
            w = precond_weight(r->precond_mode, ws, rr);
            if (w > *w_max) *w_max = w;
            ++(*count);
        }
    }
    return 99;
}

/* 2.3: P = 1 / (W + eps * W_max) on masked cells, 0 elsewhere (collective). */
int visco_sh_exact_build_illumination_precond(
        const struct visco_sh_exact_precond_request *request) {
    int local_priority, global_priority, i, j;
    long local_count;
    double local_red[2], global_red[2], local_max, global_max, w, eps;

    local_priority = precond_local_priority(request, &local_count, &local_max);
    if (MPI_Allreduce(&local_priority, &global_priority, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -19;
    if (global_priority != 99) {
        /* No statistics are meaningful; still keep every rank in step. */
        return -global_priority;
    }
    local_red[0] = (double)local_count;
    local_red[1] = local_max;
    if (MPI_Allreduce(&local_red[0], &global_red[0], 1, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -19;
    if (MPI_Allreduce(&local_red[1], &global_max, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -19;
    global_red[1] = global_max;
    if (!(global_red[0] > 0.0) || !(global_red[1] > 0.0)) return -14;

    eps = (double)request->epsilon;
    /* Every masked P is bounded by 1/(eps*W_max); reject a binary32 overflow. */
    if (!isfinite((float)(1.0 / (eps * global_red[1])))) return -14;
    for (j = 1; j <= request->ny; ++j) {
        for (i = 1; i <= request->nx; ++i) {
            if (request->update_mask[j][i] != 1.0f) {
                request->precond[j][i] = 0.0f;
                continue;
            }
            w = precond_weight(request->precond_mode,
                               request->source_energy[j][i],
                               (request->precond_mode == 2)
                                   ? request->receiver_term[j][i] : 0.0f);
            request->precond[j][i] = (float)(1.0 / (w + eps * global_red[1]));
        }
    }
    return 0;
}

/* Local priority of contract 2.4 (1, 2, 3, 15, 16, 17 or 99 = ok). */
static int preconditioned_local_priority(
        const struct visco_sh_exact_optimizer_boundary *b,
        const struct visco_sh_exact_step_scaling *s, float **precond,
        float **work_primary, float **work_rho, float **work_q) {
    int i, j;
    float p;

    if (b == NULL) return 1;
    if ((b->nx < 1) || (b->ny < 1)) return 2;
    if ((b->grad_raw_primary == NULL) || (b->grad_raw_rho == NULL) ||
            (b->grad_raw_q == NULL) || (b->optimizer_step_primary == NULL) ||
            (b->optimizer_step_rho == NULL) || (b->optimizer_step_q == NULL))
        return 3;
    if ((s == NULL) || ((s->scaling_mode != 1) && (s->scaling_mode != 2)))
        return 15;
    if (!precond_rows_valid(precond, b->ny) ||
            !precond_rows_valid(work_primary, b->ny) ||
            !precond_rows_valid(work_rho, b->ny) ||
            !precond_rows_valid(work_q, b->ny)) return 16;
    for (j = 1; j <= b->ny; ++j) {
        for (i = 1; i <= b->nx; ++i) {
            p = precond[j][i];
            if (!isfinite(p) || (p < 0.0f)) return 17;
        }
    }
    return 99;
}

/* 2.4: q_k = P * g_k, then the unchanged SH-OPT-SCALE-01 scaled step. */
int visco_sh_exact_build_preconditioned_step(
        const struct visco_sh_exact_optimizer_boundary *boundary,
        const struct visco_sh_exact_step_scaling *scaling,
        float **precond, float **work_primary, float **work_rho,
        float **work_q) {
    struct visco_sh_exact_optimizer_boundary preconditioned;
    int local_priority, global_priority, i, j;
    double p;

    local_priority = preconditioned_local_priority(
            boundary, scaling, precond, work_primary, work_rho, work_q);
    if (MPI_Allreduce(&local_priority, &global_priority, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -19;
    if (global_priority != 99) return -global_priority;

    for (j = 1; j <= boundary->ny; ++j) {
        for (i = 1; i <= boundary->nx; ++i) {
            p = (double)precond[j][i];
            work_primary[j][i] =
                    (float)(p * (double)boundary->grad_raw_primary[j][i]);
            work_rho[j][i] = (float)(p * (double)boundary->grad_raw_rho[j][i]);
            work_q[j][i] = (float)(p * (double)boundary->grad_raw_q[j][i]);
        }
    }
    preconditioned = *boundary;
    preconditioned.grad_raw_primary = work_primary;
    preconditioned.grad_raw_rho = work_rho;
    preconditioned.grad_raw_q = work_q;
    return visco_sh_exact_build_scaled_steepest_step(&preconditioned, scaling);
}

/* 2.5: initial step of the unchanged line search (pure, local). */
float visco_sh_exact_initial_alpha(float eps_scale, int adaptive,
        float factor, int iteration, float previous_alpha) {
    float candidate;

    if ((adaptive == 0) || (iteration <= 1)) return eps_scale;
    if (!isfinite(previous_alpha) || (previous_alpha <= 0.0f)) return eps_scale;
    candidate = (float)((double)factor * (double)previous_alpha);
    return (candidate < eps_scale) ? candidate : eps_scale;
}
