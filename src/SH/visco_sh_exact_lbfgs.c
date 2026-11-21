/* SH-LBFGS-01: limited-memory BFGS direction at the exact viscoelastic SH
 * optimizer boundary.
 * Contract: docs/sh_lbfgs_01_contract.md (revision 1), section 3.
 *
 * Variables are class-scaled, x_c = m_c / sigma_c, with sigma_c frozen per
 * workflow stage.  The initial inverse-Hessian metric is gamma * D with the
 * diagonal D = w_c * P (SH-PRECOND-01 preconditioner, or 1).  Pairs (s, y)
 * are stored in binary32; every reduction and the recursion run in double.
 * The unit never calls err() and depends on no other DENISE translation
 * unit. */
#include "fd.h"

#define LBFGS_OK_PRIORITY 99
#define LBFGS_CURVATURE_THRESHOLD 1.0e-8
#define LBFGS_DESCENT_THRESHOLD 1.0e-8

/* ---------------------------------------------------------------------- */
/* storage layout                                                          */
/* ---------------------------------------------------------------------- */

static size_t lbfgs_cells(const struct visco_sh_exact_lbfgs *l) {
    return (size_t)l->nx * (size_t)l->ny;
}

static size_t lbfgs_index(const struct visco_sh_exact_lbfgs *l, int j, int i) {
    return (size_t)(j - 1) * (size_t)l->nx + (size_t)(i - 1);
}

/* slot 0..memory-1, which 0 = s, 1 = y, class 0..2 */
static float *lbfgs_pair_field(const struct visco_sh_exact_lbfgs *l, int slot,
                               int which, int c) {
    return l->pairs + ((size_t)(slot * 2 + which) * 3 + (size_t)c) * lbfgs_cells(l);
}

/* which 0 = previous base, 1 = previous gradient */
static float *lbfgs_previous_field(const struct visco_sh_exact_lbfgs *l,
                                   int which, int c) {
    return l->previous + ((size_t)which * 3 + (size_t)c) * lbfgs_cells(l);
}

/* which 0 = q, 1 = r (double work vectors over all own cells and classes) */
static double *lbfgs_work_field(const struct visco_sh_exact_lbfgs *l, int which,
                                int c) {
    return l->work + ((size_t)which * 3 + (size_t)c) * lbfgs_cells(l);
}

static int lbfgs_slot(const struct visco_sh_exact_lbfgs *l, int k) {
    return (l->first + k) % l->memory;
}

static int lbfgs_mask_value_is_valid(float value) {
    return (value == 0.0f) || (value == 1.0f);
}

static int lbfgs_sum(double *values, int count) {
    double local[8], global[8];
    int k;

    for (k = 0; k < count; ++k) local[k] = values[k];
    if (MPI_Allreduce(local, global, count, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -39;
    for (k = 0; k < count; ++k) values[k] = global[k];
    return 0;
}

static int lbfgs_rows_valid(float **field, int ny) {
    int j;

    if (field == NULL) return 0;
    for (j = 1; j <= ny; ++j)
        if (field[j] == NULL) return 0;
    return 1;
}

/* ---------------------------------------------------------------------- */
/* 3.1 life cycle                                                          */
/* ---------------------------------------------------------------------- */

void visco_sh_exact_lbfgs_free(struct visco_sh_exact_lbfgs *lbfgs) {
    if (lbfgs == NULL) return;
    free(lbfgs->pairs);
    free(lbfgs->previous);
    free(lbfgs->work);
    free(lbfgs->rho);
    free(lbfgs->sy);
    free(lbfgs->alpha);
    memset(lbfgs, 0, sizeof(*lbfgs));
}

int visco_sh_exact_lbfgs_init(struct visco_sh_exact_lbfgs *lbfgs, int nx, int ny,
                              int memory) {
    int local_priority = LBFGS_OK_PRIORITY, global_priority;
    int local_failure = 0, any_failure = 0;
    size_t cells;

    if ((lbfgs == NULL) || (nx < 1) || (ny < 1) || (memory < 1) ||
            (memory > VISCO_SH_EXACT_LBFGS_MAX_MEMORY)) local_priority = 31;
    if (MPI_Allreduce(&local_priority, &global_priority, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -39;
    if (global_priority != LBFGS_OK_PRIORITY) return -global_priority;

    memset(lbfgs, 0, sizeof(*lbfgs));
    cells = (size_t)nx * (size_t)ny;
    lbfgs->pairs = (float *)malloc((size_t)memory * 6 * cells * sizeof(float));
    lbfgs->previous = (float *)malloc(6 * cells * sizeof(float));
    lbfgs->work = (double *)malloc(6 * cells * sizeof(double));
    lbfgs->rho = (double *)malloc((size_t)memory * sizeof(double));
    lbfgs->sy = (double *)malloc((size_t)memory * sizeof(double));
    lbfgs->alpha = (double *)malloc((size_t)memory * sizeof(double));
    local_failure = (lbfgs->pairs == NULL) || (lbfgs->previous == NULL) ||
                    (lbfgs->work == NULL) || (lbfgs->rho == NULL) ||
                    (lbfgs->sy == NULL) || (lbfgs->alpha == NULL);
    if (MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) {
        visco_sh_exact_lbfgs_free(lbfgs);
        return -39;
    }
    if (any_failure) {
        visco_sh_exact_lbfgs_free(lbfgs);
        return -32;
    }
    lbfgs->nx = nx;
    lbfgs->ny = ny;
    lbfgs->memory = memory;
    lbfgs->initialised = 1;
    return 0;
}

/* ---------------------------------------------------------------------- */
/* 3.2 stage start                                                         */
/* ---------------------------------------------------------------------- */

static int begin_local_priority(const struct visco_sh_exact_lbfgs *l,
                                float **base[3], float **mask, double *sums) {
    int i, j, c;

    sums[0] = sums[1] = sums[2] = sums[3] = 0.0;
    if ((l == NULL) || !l->initialised) return 31;
    if (!lbfgs_rows_valid(mask, l->ny)) return 33;
    for (c = 0; c < 3; ++c)
        if (!lbfgs_rows_valid(base[c], l->ny)) return 33;
    for (j = 1; j <= l->ny; ++j) {
        for (i = 1; i <= l->nx; ++i) {
            if (!lbfgs_mask_value_is_valid(mask[j][i])) return 34;
            if (mask[j][i] != 1.0f) continue;
            sums[0] += 1.0;
            for (c = 0; c < 3; ++c) {
                double m = (double)base[c][j][i];
                if (!isfinite(base[c][j][i]) || !(base[c][j][i] > 0.0f)) return 34;
                sums[1 + c] += m * m;
            }
        }
    }
    return LBFGS_OK_PRIORITY;
}

int visco_sh_exact_lbfgs_begin_stage(struct visco_sh_exact_lbfgs *lbfgs,
        float **base_primary, float **base_rho, float **base_q,
        float **update_mask) {
    float **base[3];
    double sums[4], sigma[3];
    int local_priority, global_priority, c;

    base[0] = base_primary; base[1] = base_rho; base[2] = base_q;
    local_priority = begin_local_priority(lbfgs, base, update_mask, sums);
    if (MPI_Allreduce(&local_priority, &global_priority, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -39;
    if (global_priority != LBFGS_OK_PRIORITY) return -global_priority;
    if (lbfgs_sum(sums, 4) != 0) return -39;
    if (!(sums[0] > 0.0)) return -35;
    for (c = 0; c < 3; ++c) {
        sigma[c] = sqrt(sums[1 + c] / sums[0]);
        if (!isfinite(sigma[c]) || !(sigma[c] > 0.0)) return -35;
    }
    for (c = 0; c < 3; ++c) lbfgs->sigma[c] = sigma[c];
    lbfgs->count = 0;
    lbfgs->first = 0;
    lbfgs->have_previous = 0;
    lbfgs->stage_ready = 1;
    return 0;
}

void visco_sh_exact_lbfgs_get_sigma(const struct visco_sh_exact_lbfgs *lbfgs,
                                    double sigma[3]) {
    int c;

    for (c = 0; c < 3; ++c)
        sigma[c] = ((lbfgs != NULL) && lbfgs->stage_ready) ? lbfgs->sigma[c] : 0.0;
}

/* ---------------------------------------------------------------------- */
/* 3.3 direction                                                           */
/* ---------------------------------------------------------------------- */

struct lbfgs_view {
    float **base[3], **grad[3], **step[3], **mask, **precond;
    double weight[3];
    int active[3];
};

static int step_local_priority(const struct visco_sh_exact_lbfgs *l,
        const struct visco_sh_exact_lbfgs_request *r,
        const struct visco_sh_exact_lbfgs_report *report, struct lbfgs_view *v,
        double *masked_cells) {
    float weights[3];
    int i, j, c, any_active;

    *masked_cells = 0.0;
    if ((l == NULL) || !l->initialised || !l->stage_ready) return 31;
    if ((r == NULL) || (report == NULL)) return 31;
    if ((r->nx != l->nx) || (r->ny != l->ny)) return 31;

    v->base[0] = r->base_primary; v->base[1] = r->base_rho; v->base[2] = r->base_q;
    v->grad[0] = r->grad_primary; v->grad[1] = r->grad_rho; v->grad[2] = r->grad_q;
    v->step[0] = r->step_primary; v->step[1] = r->step_rho; v->step[2] = r->step_q;
    v->mask = r->update_mask;
    v->precond = r->precond;
    for (c = 0; c < 3; ++c) {
        if (!lbfgs_rows_valid(v->base[c], l->ny) ||
                !lbfgs_rows_valid(v->grad[c], l->ny) ||
                !lbfgs_rows_valid(v->step[c], l->ny)) return 33;
    }
    if (!lbfgs_rows_valid(v->mask, l->ny)) return 33;
    if ((v->precond != NULL) && !lbfgs_rows_valid(v->precond, l->ny)) return 33;

    weights[0] = r->weight_primary; weights[1] = r->weight_rho;
    weights[2] = r->weight_q;
    any_active = 0;
    for (c = 0; c < 3; ++c) {
        if (!isfinite(weights[c]) || (weights[c] < 0.0f)) return 34;
        v->weight[c] = (double)weights[c];
        v->active[c] = (weights[c] > 0.0f);
        any_active |= v->active[c];
    }
    for (j = 1; j <= l->ny; ++j) {
        for (i = 1; i <= l->nx; ++i) {
            if (!lbfgs_mask_value_is_valid(v->mask[j][i])) return 34;
            if (v->mask[j][i] != 1.0f) continue;
            *masked_cells += 1.0;
            for (c = 0; c < 3; ++c) {
                if (!isfinite(v->base[c][j][i]) || !(v->base[c][j][i] > 0.0f))
                    return 34;
                if (v->active[c] && !isfinite(v->grad[c][j][i])) return 34;
            }
            if ((v->precond != NULL) &&
                    (!isfinite(v->precond[j][i]) || !(v->precond[j][i] > 0.0f)))
                return 34;
        }
    }
    if (!any_active) return 36;
    return LBFGS_OK_PRIORITY;
}

/* x-space value of the candidate pair at an entry (binary32, contract 3.3/2). */
static float pair_s(const struct visco_sh_exact_lbfgs *l, const struct lbfgs_view *v,
                    int c, int j, int i) {
    double now = (double)v->base[c][j][i];
    double before = (double)lbfgs_previous_field(l, 0, c)[lbfgs_index(l, j, i)];
    return (float)((now - before) / l->sigma[c]);
}

static float pair_y(const struct visco_sh_exact_lbfgs *l, const struct lbfgs_view *v,
                    int c, int j, int i) {
    double now = (double)v->grad[c][j][i];
    double before = (double)lbfgs_previous_field(l, 1, c)[lbfgs_index(l, j, i)];
    return (float)(l->sigma[c] * (now - before));
}

static int entry_in_e(const struct lbfgs_view *v, int c, int j, int i) {
    return v->active[c] && (v->mask[j][i] == 1.0f);
}

static double diag_d(const struct lbfgs_view *v, int c, int j, int i) {
    return v->weight[c] * ((v->precond != NULL) ? (double)v->precond[j][i] : 1.0);
}

/* sum over E of a(e)*b(e) for two stored binary32 vectors */
static int dot_float_double(const struct visco_sh_exact_lbfgs *l,
                            const struct lbfgs_view *v, int slot, int which,
                            int work, double *out) {
    double value = 0.0;
    int i, j, c;
    size_t k;

    for (c = 0; c < 3; ++c) {
        const float *a = lbfgs_pair_field(l, slot, which, c);
        const double *b = lbfgs_work_field(l, work, c);
        for (j = 1; j <= l->ny; ++j)
            for (i = 1; i <= l->nx; ++i) {
                if (!entry_in_e(v, c, j, i)) continue;
                k = lbfgs_index(l, j, i);
                value += (double)a[k] * b[k];
            }
    }
    *out = value;
    return lbfgs_sum(out, 1);
}

static void axpy_pair(const struct visco_sh_exact_lbfgs *l, const struct lbfgs_view *v,
                      int slot, int which, int work, double factor) {
    int i, j, c;
    size_t k;

    for (c = 0; c < 3; ++c) {
        const float *a = lbfgs_pair_field(l, slot, which, c);
        double *b = lbfgs_work_field(l, work, c);
        for (j = 1; j <= l->ny; ++j)
            for (i = 1; i <= l->nx; ++i) {
                if (!entry_in_e(v, c, j, i)) continue;
                k = lbfgs_index(l, j, i);
                b[k] += factor * (double)a[k];
            }
    }
}

static void store_previous(struct visco_sh_exact_lbfgs *l, const struct lbfgs_view *v) {
    int i, j, c;

    for (c = 0; c < 3; ++c) {
        float *b = lbfgs_previous_field(l, 0, c);
        float *g = lbfgs_previous_field(l, 1, c);
        for (j = 1; j <= l->ny; ++j)
            for (i = 1; i <= l->nx; ++i) {
                b[lbfgs_index(l, j, i)] = v->base[c][j][i];
                g[lbfgs_index(l, j, i)] = v->grad[c][j][i];
            }
        l->previous_active[c] = v->active[c];
    }
    l->have_previous = 1;
}

int visco_sh_exact_lbfgs_step(struct visco_sh_exact_lbfgs *lbfgs,
        const struct visco_sh_exact_lbfgs_request *request,
        struct visco_sh_exact_lbfgs_report *report) {
    struct visco_sh_exact_lbfgs_report out;
    struct lbfgs_view v;
    double masked_cells, sums[4], n_active, gamma, kappa, yDy, beta;
    int local_priority, global_priority, reset = 0, i, j, c, k, slot, n;
    size_t idx;

    memset(&v, 0, sizeof(v));
    local_priority = step_local_priority(lbfgs, request, report, &v, &masked_cells);
    if (MPI_Allreduce(&local_priority, &global_priority, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -39;
    if (global_priority != LBFGS_OK_PRIORITY) return -global_priority;
    if (lbfgs_sum(&masked_cells, 1) != 0) return -39;
    n_active = (double)(v.active[0] + v.active[1] + v.active[2]);

    memset(&out, 0, sizeof(out));
    for (c = 0; c < 3; ++c) out.sigma[c] = lbfgs->sigma[c];

    /* Step 2: pair. */
    if (lbfgs->have_previous) {
        if ((lbfgs->previous_active[0] != v.active[0]) ||
                (lbfgs->previous_active[1] != v.active[1]) ||
                (lbfgs->previous_active[2] != v.active[2])) {
            lbfgs->count = 0;
            lbfgs->first = 0;
            reset = 1;
        } else {
            double ss = 0.0, yy = 0.0, sy = 0.0, s, y;
            sums[0] = sums[1] = sums[2] = 0.0;
            for (c = 0; c < 3; ++c)
                for (j = 1; j <= lbfgs->ny; ++j)
                    for (i = 1; i <= lbfgs->nx; ++i) {
                        if (!entry_in_e(&v, c, j, i)) continue;
                        s = (double)pair_s(lbfgs, &v, c, j, i);
                        y = (double)pair_y(lbfgs, &v, c, j, i);
                        sums[0] += s * s;
                        sums[1] += y * y;
                        sums[2] += s * y;
                    }
            if (lbfgs_sum(sums, 3) != 0) return -39;
            ss = sums[0]; yy = sums[1]; sy = sums[2];
            if ((ss > 0.0) && (yy > 0.0)) {
                double value = sy / sqrt(ss * yy);
                out.curvature = isfinite(value) ? value : 0.0;
            }
            if ((ss > 0.0) && (yy > 0.0) && isfinite(sy) &&
                    (sy > LBFGS_CURVATURE_THRESHOLD * sqrt(ss * yy))) {
                if (lbfgs->count == lbfgs->memory) {
                    lbfgs->first = (lbfgs->first + 1) % lbfgs->memory;
                    lbfgs->count -= 1;
                }
                slot = lbfgs_slot(lbfgs, lbfgs->count);
                for (c = 0; c < 3; ++c) {
                    float *sf = lbfgs_pair_field(lbfgs, slot, 0, c);
                    float *yf = lbfgs_pair_field(lbfgs, slot, 1, c);
                    for (j = 1; j <= lbfgs->ny; ++j)
                        for (i = 1; i <= lbfgs->nx; ++i) {
                            idx = lbfgs_index(lbfgs, j, i);
                            if (entry_in_e(&v, c, j, i)) {
                                sf[idx] = pair_s(lbfgs, &v, c, j, i);
                                yf[idx] = pair_y(lbfgs, &v, c, j, i);
                            } else {
                                sf[idx] = 0.0f;
                                yf[idx] = 0.0f;
                            }
                        }
                }
                lbfgs->sy[slot] = sy;
                lbfgs->rho[slot] = 1.0 / sy;
                lbfgs->count += 1;
                out.pair = 1;
            } else {
                out.pair = 2;
            }
        }
    }

    /* Step 3: no pairs. */
    if (lbfgs->count == 0) {
        out.action = reset ? 1 : 0;
        store_previous(lbfgs, &v);
        *report = out;
        return 0;
    }

    /* Step 4: two-loop recursion with q = work 0 and r = work 1. */
    n = lbfgs->count;
    for (c = 0; c < 3; ++c) {
        double *q = lbfgs_work_field(lbfgs, 0, c);
        for (j = 1; j <= lbfgs->ny; ++j)
            for (i = 1; i <= lbfgs->nx; ++i) {
                idx = lbfgs_index(lbfgs, j, i);
                q[idx] = entry_in_e(&v, c, j, i)
                         ? lbfgs->sigma[c] * (double)v.grad[c][j][i] : 0.0;
            }
    }
    for (k = n - 1; k >= 0; --k) {
        double sq;
        slot = lbfgs_slot(lbfgs, k);
        if (dot_float_double(lbfgs, &v, slot, 0, 0, &sq) != 0) return -39;
        lbfgs->alpha[k] = lbfgs->rho[slot] * sq;
        axpy_pair(lbfgs, &v, slot, 1, 0, -lbfgs->alpha[k]);
    }
    slot = lbfgs_slot(lbfgs, n - 1);
    yDy = 0.0;
    for (c = 0; c < 3; ++c) {
        const float *yf = lbfgs_pair_field(lbfgs, slot, 1, c);
        for (j = 1; j <= lbfgs->ny; ++j)
            for (i = 1; i <= lbfgs->nx; ++i) {
                if (!entry_in_e(&v, c, j, i)) continue;
                idx = lbfgs_index(lbfgs, j, i);
                yDy += diag_d(&v, c, j, i) * (double)yf[idx] * (double)yf[idx];
            }
    }
    if (lbfgs_sum(&yDy, 1) != 0) return -39;
    gamma = lbfgs->sy[slot] / yDy;
    for (c = 0; c < 3; ++c) {
        const double *q = lbfgs_work_field(lbfgs, 0, c);
        double *r = lbfgs_work_field(lbfgs, 1, c);
        for (j = 1; j <= lbfgs->ny; ++j)
            for (i = 1; i <= lbfgs->nx; ++i) {
                idx = lbfgs_index(lbfgs, j, i);
                r[idx] = entry_in_e(&v, c, j, i)
                         ? gamma * diag_d(&v, c, j, i) * q[idx] : 0.0;
            }
    }
    for (k = 0; k < n; ++k) {
        double yr;
        slot = lbfgs_slot(lbfgs, k);
        if (dot_float_double(lbfgs, &v, slot, 1, 1, &yr) != 0) return -39;
        beta = lbfgs->rho[slot] * yr;
        axpy_pair(lbfgs, &v, slot, 0, 1, lbfgs->alpha[k] - beta);
    }

    /* Descent test and normalisation sums: gd, gDg, dDd, d.d */
    sums[0] = sums[1] = sums[2] = sums[3] = 0.0;
    for (c = 0; c < 3; ++c) {
        const double *r = lbfgs_work_field(lbfgs, 1, c);
        for (j = 1; j <= lbfgs->ny; ++j)
            for (i = 1; i <= lbfgs->nx; ++i) {
                double g, d, dd;
                if (!entry_in_e(&v, c, j, i)) continue;
                idx = lbfgs_index(lbfgs, j, i);
                g = lbfgs->sigma[c] * (double)v.grad[c][j][i];
                d = r[idx];
                dd = diag_d(&v, c, j, i);
                sums[0] += g * d;
                sums[1] += dd * g * g;
                sums[2] += d * d / dd;
                sums[3] += d * d;
            }
    }
    if (lbfgs_sum(sums, 4) != 0) return -39;
    out.gamma = gamma;
    out.pairs_used = n;
    if ((sums[1] > 0.0) && (sums[2] > 0.0)) {
        double value = sums[0] / sqrt(sums[1] * sums[2]);
        out.descent = isfinite(value) ? value : 0.0;
    }
    if (!(isfinite(sums[0]) && (sums[1] > 0.0) && (sums[2] > 0.0) &&
          (sums[0] > LBFGS_DESCENT_THRESHOLD * sqrt(sums[1] * sums[2])))) {
        out.action = 2;
        lbfgs->count = 0;
        lbfgs->first = 0;
        store_previous(lbfgs, &v);
        *report = out;
        return 0;
    }

    /* Step 5: global normalisation and output. */
    kappa = 1.0 / sqrt(sums[3] / (n_active * masked_cells));
    out.kappa = kappa;
    out.action = 3;
    for (c = 0; c < 3; ++c) {
        const double *r = lbfgs_work_field(lbfgs, 1, c);
        for (j = 1; j <= lbfgs->ny; ++j)
            for (i = 1; i <= lbfgs->nx; ++i) {
                idx = lbfgs_index(lbfgs, j, i);
                v.step[c][j][i] = entry_in_e(&v, c, j, i)
                        ? (float)(lbfgs->sigma[c] * kappa * r[idx]) : 0.0f;
            }
    }
    store_previous(lbfgs, &v);
    *report = out;
    return 0;
}
