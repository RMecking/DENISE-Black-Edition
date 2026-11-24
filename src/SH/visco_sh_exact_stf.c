/* SH-S3: source time function estimator of the exact viscoelastic SH FWI.
 *
 * For one shot with basis traces g_i (receiver traces of F(m; B e_1), e_1 the
 * unit impulse at sample 1), filtered observed traces d_i and SH-S2 data
 * weights w_i, the source signal v (samples 1..nw, before the stage filter)
 * minimises
 *
 *   D(v) + eps/2 |v|^2,   D(v) = 1/2 sum_i sum_{k=2..nt} (w_i(k) (s_i(k) - d_i(k)))^2,
 *   s_i(k) = sum_{j=1..min(k,nw)} v(j) g_i(k-j+1),
 *
 * i.e. (A + eps I) v = b with A = sum_i C_i^T W_i^2 C_i, b = sum_i C_i^T W_i^2 d_i.
 * Solved exactly (to a relative residual tolerance) by preconditioned
 * conjugate gradients in double precision; C_i and C_i^T are applied with
 * zero-padded FFT convolutions (no wrap-around for nfft >= nt + nw - 1); the
 * preconditioner is the circulant approximation sum_i mean(w_i^2) |G_i|^2 + eps.
 * Contract: docs/sh_s3_stf_contract.md.
 *
 * Arrays are 0-based: g[i*nt + n] is sample n+1, w[i*nt + n] the weight of
 * sample n+1 (layout of the SH-S2 weights); sample 1 (n = 0) never enters the
 * objective.  Local, no MPI, no err(). */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>
#include "fd.h"

#define STF_MAX_RESTARTS 5

struct stf_work {
    int ntr, nt, nw, nfft, nf;
    double *w2;              /* ntr*nt effective squared weights */
    fftw_complex *G;         /* ntr*nf spectra of the basis traces */
    double *lambda;          /* nf preconditioner spectrum (without eps) */
    double *buf;             /* nfft real buffer */
    fftw_complex *spec;      /* nf */
    fftw_complex *acc;       /* nf */
    fftw_plan fwd, inv;
};

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

static double weight2(const struct visco_sh_exact_stf_problem *p, int i, int n) {
    double w;
    if (n == 0) return 0.0;
    if (p->w == NULL) return 1.0;
    w = p->w[(size_t)i * p->nt + n];
    return w * w;
}

static int problem_valid(const struct visco_sh_exact_stf_problem *p) {
    return (p != NULL) && (p->ntr >= 1) && (p->nt >= 2) && (p->nw >= 1) &&
           (p->nw <= p->nt) && (p->g != NULL) && (p->d != NULL);
}

static void work_free(struct stf_work *ws) {
    if (ws->fwd) fftw_destroy_plan(ws->fwd);
    if (ws->inv) fftw_destroy_plan(ws->inv);
    free(ws->w2);
    free(ws->lambda);
    if (ws->G) fftw_free(ws->G);
    if (ws->buf) fftw_free(ws->buf);
    if (ws->spec) fftw_free(ws->spec);
    if (ws->acc) fftw_free(ws->acc);
    memset(ws, 0, sizeof(*ws));
}

static int work_init(struct stf_work *ws, const struct visco_sh_exact_stf_problem *p) {
    int i, n, f;
    memset(ws, 0, sizeof(*ws));
    ws->ntr = p->ntr; ws->nt = p->nt; ws->nw = p->nw;
    ws->nfft = next_pow2(p->nt + p->nw - 1);
    ws->nf = ws->nfft / 2 + 1;
    ws->w2 = (double *)malloc((size_t)p->ntr * p->nt * sizeof(double));
    ws->lambda = (double *)calloc((size_t)ws->nf, sizeof(double));
    ws->G = (fftw_complex *)fftw_malloc((size_t)p->ntr * ws->nf * sizeof(fftw_complex));
    ws->buf = (double *)fftw_malloc((size_t)ws->nfft * sizeof(double));
    ws->spec = (fftw_complex *)fftw_malloc((size_t)ws->nf * sizeof(fftw_complex));
    ws->acc = (fftw_complex *)fftw_malloc((size_t)ws->nf * sizeof(fftw_complex));
    if (!ws->w2 || !ws->lambda || !ws->G || !ws->buf || !ws->spec || !ws->acc) {
        work_free(ws);
        return -2;
    }
    ws->fwd = fftw_plan_dft_r2c_1d(ws->nfft, ws->buf, ws->spec, FFTW_ESTIMATE);
    ws->inv = fftw_plan_dft_c2r_1d(ws->nfft, ws->spec, ws->buf, FFTW_ESTIMATE);
    if (!ws->fwd || !ws->inv) {
        work_free(ws);
        return -2;
    }
    for (i = 0; i < p->ntr; ++i) {
        double mean = 0.0;
        for (n = 0; n < p->nt; ++n) {
            double v = weight2(p, i, n);
            ws->w2[(size_t)i * p->nt + n] = v;
            mean += v;
        }
        mean /= (double)p->nt;
        memset(ws->buf, 0, (size_t)ws->nfft * sizeof(double));
        for (n = 0; n < p->nt; ++n) ws->buf[n] = (double)p->g[(size_t)i * p->nt + n];
        fftw_execute(ws->fwd);
        memcpy(ws->G + (size_t)i * ws->nf, ws->spec, (size_t)ws->nf * sizeof(fftw_complex));
        for (f = 0; f < ws->nf; ++f) {
            double re = ws->spec[f][0], im = ws->spec[f][1];
            ws->lambda[f] += mean * (re * re + im * im);
        }
    }
    return 0;
}

/* acc += conj(G_i) * FFT(W_i^2 r_i), r_i given in buf[0..nt-1] */
static void accumulate_adjoint(struct stf_work *ws, int i) {
    int n, f;
    const double *w2 = ws->w2 + (size_t)i * ws->nt;
    const fftw_complex *G = ws->G + (size_t)i * ws->nf;
    for (n = 0; n < ws->nt; ++n) ws->buf[n] *= w2[n];
    for (n = ws->nt; n < ws->nfft; ++n) ws->buf[n] = 0.0;
    fftw_execute(ws->fwd);
    for (f = 0; f < ws->nf; ++f) {
        double gr = G[f][0], gi = G[f][1], sr = ws->spec[f][0], si = ws->spec[f][1];
        ws->acc[f][0] += gr * sr + gi * si;
        ws->acc[f][1] += gr * si - gi * sr;
    }
}

/* out[0..nw-1] = first nw samples of IFFT(acc)/nfft */
static void finish_adjoint(struct stf_work *ws, double *out) {
    int j;
    memcpy(ws->spec, ws->acc, (size_t)ws->nf * sizeof(fftw_complex));
    fftw_execute(ws->inv);
    for (j = 0; j < ws->nw; ++j) out[j] = ws->buf[j] / (double)ws->nfft;
}

static void zero_acc(struct stf_work *ws) {
    memset(ws->acc, 0, (size_t)ws->nf * sizeof(fftw_complex));
}

/* spec_v = FFT(v padded) kept in vspec */
static void spectrum_of(struct stf_work *ws, const double *v, fftw_complex *vspec) {
    int j;
    memset(ws->buf, 0, (size_t)ws->nfft * sizeof(double));
    for (j = 0; j < ws->nw; ++j) ws->buf[j] = v[j];
    fftw_execute(ws->fwd);
    memcpy(vspec, ws->spec, (size_t)ws->nf * sizeof(fftw_complex));
}

/* buf[0..nt-1] = (C_i v)(n) from vspec */
static void synthetic(struct stf_work *ws, int i, const fftw_complex *vspec) {
    int f, n;
    const fftw_complex *G = ws->G + (size_t)i * ws->nf;
    for (f = 0; f < ws->nf; ++f) {
        double gr = G[f][0], gi = G[f][1], vr = vspec[f][0], vi = vspec[f][1];
        ws->spec[f][0] = gr * vr - gi * vi;
        ws->spec[f][1] = gr * vi + gi * vr;
    }
    fftw_execute(ws->inv);
    for (n = 0; n < ws->nt; ++n) ws->buf[n] /= (double)ws->nfft;
}

static void apply_normal(struct stf_work *ws, const double *x, double eps, double *y,
                         fftw_complex *vspec) {
    int i, j;
    spectrum_of(ws, x, vspec);
    zero_acc(ws);
    for (i = 0; i < ws->ntr; ++i) {
        synthetic(ws, i, vspec);
        accumulate_adjoint(ws, i);
    }
    finish_adjoint(ws, y);
    for (j = 0; j < ws->nw; ++j) y[j] += eps * x[j];
}

static void apply_precond(struct stf_work *ws, const double *r, double eps, double *z) {
    int j, f;
    memset(ws->buf, 0, (size_t)ws->nfft * sizeof(double));
    for (j = 0; j < ws->nw; ++j) ws->buf[j] = r[j];
    fftw_execute(ws->fwd);
    for (f = 0; f < ws->nf; ++f) {
        double l = ws->lambda[f] + eps;
        ws->spec[f][0] /= l;
        ws->spec[f][1] /= l;
    }
    fftw_execute(ws->inv);
    for (j = 0; j < ws->nw; ++j) z[j] = ws->buf[j] / (double)ws->nfft;
}

static double dot(const double *a, const double *b, int n) {
    double s = 0.0;
    int j;
    for (j = 0; j < n; ++j) s += a[j] * b[j];
    return s;
}

double visco_sh_exact_stf_mean_diag(const struct visco_sh_exact_stf_problem *p) {
    /* trace(A)/nw = (1/nw) sum_i sum_m g_i(m)^2 sum_{j=0}^{min(nw-1,nt-1-m)} w2_i(m+j) */
    double total = 0.0, *pre;
    int i, m, n;
    if (!problem_valid(p)) return -1.0;
    pre = (double *)malloc((size_t)(p->nt + 1) * sizeof(double));
    if (pre == NULL) return -1.0;
    for (i = 0; i < p->ntr; ++i) {
        pre[0] = 0.0;
        for (n = 0; n < p->nt; ++n) pre[n + 1] = pre[n] + weight2(p, i, n);
        for (m = 0; m < p->nt; ++m) {
            double g = (double)p->g[(size_t)i * p->nt + m];
            int hi = m + p->nw;
            if (hi > p->nt) hi = p->nt;
            total += g * g * (pre[hi] - pre[m]);
        }
    }
    free(pre);
    return total / (double)p->nw;
}

int visco_sh_exact_stf_terms(const struct visco_sh_exact_stf_problem *p, const double *v,
                             double eps, double *data_term, double *eps_term) {
    struct stf_work ws;
    fftw_complex *vspec;
    double dterm = 0.0;
    int i, n, status;
    if (!problem_valid(p) || (v == NULL) || !(eps >= 0.0)) return -1;
    status = work_init(&ws, p);
    if (status != 0) return status;
    vspec = (fftw_complex *)fftw_malloc((size_t)ws.nf * sizeof(fftw_complex));
    if (vspec == NULL) { work_free(&ws); return -2; }
    spectrum_of(&ws, v, vspec);
    for (i = 0; i < p->ntr; ++i) {
        synthetic(&ws, i, vspec);
        for (n = 1; n < p->nt; ++n) {
            double r = ws.buf[n] - (double)p->d[(size_t)i * p->nt + n];
            dterm += 0.5 * ws.w2[(size_t)i * p->nt + n] * r * r;
        }
    }
    fftw_free(vspec);
    work_free(&ws);
    *data_term = dterm;
    *eps_term = 0.5 * eps * dot(v, v, p->nw);
    return 0;
}

int visco_sh_exact_stf_solve(const struct visco_sh_exact_stf_problem *p, double eps,
                             double tol, int maxit, double *v, int *iterations,
                             double *relres) {
    struct stf_work ws;
    fftw_complex *vspec = NULL;
    double *b = NULL, *r = NULL, *z = NULL, *q = NULL, *Ap = NULL;
    double bnorm, rz, rz_new, alpha, beta, rn;
    int i, n, it = 0, status, nw, restart;

    if (!problem_valid(p) || (v == NULL) || !(eps > 0.0) || !isfinite(eps) ||
            !(tol > 0.0) || (maxit < 1)) return -1;
    status = work_init(&ws, p);
    if (status != 0) return status;
    nw = p->nw;
    vspec = (fftw_complex *)fftw_malloc((size_t)ws.nf * sizeof(fftw_complex));
    b = (double *)malloc((size_t)nw * sizeof(double));
    r = (double *)malloc((size_t)nw * sizeof(double));
    z = (double *)malloc((size_t)nw * sizeof(double));
    q = (double *)malloc((size_t)nw * sizeof(double));
    Ap = (double *)malloc((size_t)nw * sizeof(double));
    if (!vspec || !b || !r || !z || !q || !Ap) { status = -2; goto done; }

    /* b = sum_i C_i^T W_i^2 d_i */
    zero_acc(&ws);
    for (i = 0; i < p->ntr; ++i) {
        for (n = 0; n < p->nt; ++n) ws.buf[n] = (double)p->d[(size_t)i * p->nt + n];
        accumulate_adjoint(&ws, i);
    }
    finish_adjoint(&ws, b);
    bnorm = sqrt(dot(b, b, nw));
    for (n = 0; n < nw; ++n) v[n] = 0.0;
    if (!(bnorm > 0.0)) {
        status = isfinite(bnorm) ? 0 : -3;
        if (iterations) *iterations = 0;
        if (relres) *relres = 0.0;
        goto done;
    }
    /* PCG with restarts from the true residual: the solution is accepted only
     * if the true relative residual |b - (A + eps I) v| / |b| <= tol (review
     * V-5); each restart runs until the recursive residual is <= tol/2. */
    for (n = 0; n < nw; ++n) r[n] = b[n];
    rn = 1.0;
    for (restart = 0; restart <= STF_MAX_RESTARTS; ++restart) {
        if (restart > 0) {
            apply_normal(&ws, v, eps, Ap, vspec);
            for (n = 0; n < nw; ++n) r[n] = b[n] - Ap[n];
            rn = sqrt(dot(r, r, nw)) / bnorm;
            if (!isfinite(rn)) { status = -3; goto done; }
            if (rn <= tol) break;
        }
        apply_precond(&ws, r, eps, z);
        for (n = 0; n < nw; ++n) q[n] = z[n];
        rz = dot(r, z, nw);
        while (it < maxit) {
            double pAp, rr;
            apply_normal(&ws, q, eps, Ap, vspec);
            pAp = dot(q, Ap, nw);
            if (!(pAp > 0.0) || !isfinite(pAp)) { status = -3; goto done; }
            alpha = rz / pAp;
            for (n = 0; n < nw; ++n) { v[n] += alpha * q[n]; r[n] -= alpha * Ap[n]; }
            ++it;
            rr = sqrt(dot(r, r, nw)) / bnorm;
            if (rr <= 0.5 * tol) break;
            apply_precond(&ws, r, eps, z);
            rz_new = dot(r, z, nw);
            beta = rz_new / rz;
            rz = rz_new;
            for (n = 0; n < nw; ++n) q[n] = z[n] + beta * q[n];
        }
        if (it >= maxit) break;
    }
    /* true residual of the returned solution */
    apply_normal(&ws, v, eps, Ap, vspec);
    for (n = 0; n < nw; ++n) Ap[n] -= b[n];
    rn = sqrt(dot(Ap, Ap, nw)) / bnorm;
    if (iterations) *iterations = it;
    if (relres) *relres = rn;
    status = (isfinite(rn) && (rn <= tol)) ? 0 : -4;
done:
    if (vspec) fftw_free(vspec);
    free(b); free(r); free(z); free(q); free(Ap);
    work_free(&ws);
    return status;
}
