/* Inactive collective physical-Q material preparation. */
#include "fd.h"
#include <float.h>
#include <limits.h>

/* Check the arithmetic used by init_q_tau_mapping before calling its
 * abort-on-invalid API. No target storage is touched here. */
static int preparation_mapping(struct q_tau_mapping *mapping) {
    extern int L, Q_PARAMETERIZATION_MODE;
    extern float *FL, Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF;
    double count, sum_a = 0.0, sum_ab = 0.0, sum_aa = 0.0;
    int n, l, samples;
    if (Q_PARAMETERIZATION_MODE != Q_PARAMETERIZATION_PHYSICAL ||
            !isfinite(Q_APPROX_FMIN) || !isfinite(Q_APPROX_FMAX) ||
            !isfinite(Q_APPROX_DF) || !(Q_APPROX_FMIN > 0.0f) ||
            Q_APPROX_FMAX < Q_APPROX_FMIN || !(Q_APPROX_DF > 0.0f))
        return -1;
    count = floor(((double)Q_APPROX_FMAX - Q_APPROX_FMIN) /
                  Q_APPROX_DF + 1.0e-12) + 1.0;
    if (!isfinite(count) || count < 1 || count >= INT_MAX) return -1;
    samples = (int)count;
    for (n = 0; n < samples; ++n) {
        double omega = 2.0 * PI * ((double)Q_APPROX_FMIN +
                                  n * (double)Q_APPROX_DF);
        double A = 0.0, B = 0.0;
        for (l = 1; l <= L; ++l) {
            double theta = 1.0 / (2.0 * PI * FL[l]);
            double x = omega * theta;
            A += x * x / (1.0 + x * x);
            B += x / (1.0 + x * x);
        }
        if (!(B > 0.0) || !isfinite(B) || !isfinite(A)) return -1;
        sum_a += 1.0 / B;
        sum_ab += (1.0 / B) * (A / B);
        sum_aa += (1.0 / B) * (1.0 / B);
    }
    if (!(sum_aa > 0.0) || !isfinite(sum_aa) ||
            !isfinite(sum_a) || !isfinite(sum_ab)) return -1;
    init_q_tau_mapping(mapping, Q_PARAMETERIZATION_PHYSICAL, L, FL,
                      Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF);
    return !isfinite(mapping->inverse_tau_per_q) ||
           !(mapping->inverse_tau_per_q > 0.0) ||
           !isfinite(mapping->inverse_tau_offset) ? -1 : 0;
}

static int preparation_validate(
        const struct visco_sh_exact_material_preparation_request *r,
        struct q_tau_mapping *mapping) {
    extern int NX, NY, L, INVMAT1;
    extern float DT, *FL;
    extern FILE *FP;
    struct matSH *t;
    int i, j, l;
    if (r == NULL || r->target == NULL || NX < 1 || NY < 1 || L < 1 ||
            (INVMAT1 != 1 && INVMAT1 != 3) || FP == NULL ||
            !isfinite(DT) || !(DT > 0.0f) || FL == NULL ||
            r->mechanisms != L || r->dt != DT ||
            r->frequencies_hz == NULL || r->peta == NULL ||
            r->primary == NULL || r->rho == NULL || r->physical_q == NULL)
        return -1;
    t = r->target;
    if (t->pu == NULL || t->prho == NULL || t->pqs == NULL ||
            t->ptaus == NULL || t->puip == NULL || t->pujp == NULL ||
            t->prhoi == NULL || t->ptausipjp == NULL || t->f == NULL ||
            t->fipjp == NULL || t->d == NULL || t->dip == NULL ||
            t->peta == NULL || t->etaip == NULL || t->etajm == NULL ||
            t->bip == NULL || t->bjm == NULL ||
            t->cip == NULL || t->cjm == NULL) return -1;
    for (l = 1; l <= L; ++l) {
        float theta, eta;
        if (!isfinite(FL[l]) || !(FL[l] > 0.0f) ||
                r->frequencies_hz[l] != FL[l]) return -1;
        theta = 1.0 / (2.0 * PI * FL[l]);
        eta = DT / theta;
        if (!isfinite(theta) || !(theta > 0.0f) ||
                !isfinite(eta) || !(eta > 0.0f) ||
                !isfinite(r->peta[l]) ||
                fabs((double)r->peta[l] - eta) >
                    4.0 * FLT_EPSILON * eta) return -1;
    }
    if (preparation_mapping(mapping) != 0) return -1;
    for (j = 0; j <= NY + 1; ++j)
        if (t->pu[j] == NULL || t->prho[j] == NULL ||
                t->ptaus[j] == NULL) return -1;
    for (j = 1; j <= NY; ++j) {
        if (r->primary[j] == NULL || r->rho[j] == NULL ||
                r->physical_q[j] == NULL || t->pqs[j] == NULL ||
                t->puip[j] == NULL || t->pujp[j] == NULL ||
                t->prhoi[j] == NULL || t->ptausipjp[j] == NULL ||
                t->f[j] == NULL || t->fipjp[j] == NULL ||
                t->d[j] == NULL || t->dip[j] == NULL) return -1;
        for (i = 1; i <= NX; ++i) {
            float p = r->primary[j][i], rho = r->rho[j][i];
            float q = r->physical_q[j][i], tau;
            double inverse_tau;
            /* Vs may be zero (the averaging helper handles zero shear).
             * Direct mu must be positive for its reciprocal averaging. */
            if (!isfinite(p) || (INVMAT1 == 1 ? p < 0.0f : p <= 0.0f) ||
                    !isfinite(rho) || !(rho > 0.0f) ||
                    !isfinite(q) || !(q > 0.0f) ||
                    t->d[j][i] == NULL || t->dip[j][i] == NULL) return -1;
            if (INVMAT1 == 1 && !isfinite(rho * p * p)) return -1;
            inverse_tau = mapping->inverse_tau_per_q * q +
                          mapping->inverse_tau_offset;
            if (!(inverse_tau > 0.0) || !isfinite(inverse_tau)) return -1;
            tau = (float)(1.0 / inverse_tau);
            if (!isfinite(tau) || !(tau > 0.0f)) return -1;
        }
    }
    return 0;
}

int visco_sh_exact_prepare_visco_material(
        const struct visco_sh_exact_material_preparation_request *request) {
    extern int NX, NY, L;
    struct q_tau_mapping mapping;
    struct matSH *t;
    int i, j, l, invalid, any_invalid;
    invalid = preparation_validate(request, &mapping) != 0;
    if (MPI_Allreduce(&invalid, &any_invalid, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (any_invalid) return -1;

    t = request->target;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i) {
            t->pu[j][i] = request->primary[j][i];
            t->prho[j][i] = request->rho[j][i];
            t->pqs[j][i] = request->physical_q[j][i];
            t->ptaus[j][i] = q_to_tau(t->pqs[j][i], &mapping);
        }
    for (l = 1; l <= L; ++l) t->peta[l] = request->peta[l];
    matcopy_SH(t->prho, t->pu, t->ptaus);
    av_mu_SH(t->pu, t->puip, t->pujp, t->prho);
    inv_rho_SH(t->prho, t->prhoi);
    av_tau(t->ptaus, t->ptausipjp);
    prepare_update_s_visc_SH(t->etajm, t->etaip, t->peta, t->fipjp,
            t->pujp, t->puip, t->prho, t->ptaus, t->ptausipjp,
            t->f, t->g, t->bip, t->bjm, t->cip, t->cjm,
            t->dip, t->d, t->e);
    return 0;
}
