/* SH-S3: per-shot source time function estimation in the exact viscoelastic
 * SH multi-shot wrappers (objective and objective/gradient).
 *
 * Per shot: impulse source e_1 (sample 1), stage filter B on it and on the
 * observed traces (SH-S2), impulse forward (mode 2: receiver traces only, no
 * trajectory, no energy), gather of the basis traces and weights to MYID 0,
 * estimation there (src/SH/visco_sh_exact_stf.c), broadcast of the signal v,
 * and the forward source B v.  The shot objective is D(v) + eps/2 |v|^2.
 * Per stage: eps of every shot (frozen at its first evaluation) and, for
 * INV_STF = 3, the fixed signal of every shot.
 * Contract: docs/sh_s3_stf_contract.md. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fd.h"
#include "exact_sh_receivers.h"
#include "exact_sh_data_ops.h"
#include "exact_sh_collective.h"

#define STF_TOL 1.0e-11
#define STF_MAXIT 5000

static struct {
    int stage, mode, nshots, nw;
    double eps_rel;
    float offsetc;      /* OFFSETC_STF of the stage (INV_STF = 3), 0 = off */
    double *eps;        /* [nshots+1], < 0: not yet set in this stage */
    double *v_fixed;    /* [(nshots+1)*nw], INV_STF = 3 */
    int *fixed_set;     /* [nshots+1] */
    double *v_last;     /* [(nshots+1)*nw], last signal per shot (output) */
    int *last_set;
    double *d_last, *e_last;   /* [nshots+1] data term D and eps term E (log) */
    double *res_last;          /* [nshots+1] true relative residual of the last solve */
    int *it_last;              /* [nshots+1] PCG iterations of the last solve (0: fixed) */
    double dterm_shot;         /* D of the last prepared shot, every rank */
    int max_iterations;
} stfs;

static void stf_free(void) {
    free(stfs.eps); free(stfs.v_fixed); free(stfs.fixed_set);
    free(stfs.v_last); free(stfs.last_set);
    free(stfs.d_last); free(stfs.e_last); free(stfs.res_last); free(stfs.it_last);
    memset(&stfs, 0, sizeof(stfs));
}

int visco_sh_exact_stf_nw(void) {
    extern float STF_LENGTH, TIME, DT;
    extern int NT;
    double length = (STF_LENGTH < 0.0f) ? 0.5 * (double)TIME : (double)STF_LENGTH;
    double n;
    /* the driver validates the stages before it sets NT = iround(TIME/DT) */
    int nt = iround(TIME / DT);
    if (!isfinite(length) || !(DT > 0.0f) || (nt < 1)) return 0;
    if ((NT > 0) && (NT != nt)) return 0;
    n = floor(length / (double)DT + 0.5);
    if (n < 1.0) return 0;
    if (n > (double)nt) n = (double)nt;
    return (int)n;
}

/* Per-stage validation of the SH-S3 controls (local; caller decides
 * collectively).  Returns 1 if valid. */
int visco_sh_exact_stf_config_valid(void) {
    extern int INV_STF, NORMALIZE;
    extern float EPS_STF, OFFSETC_STF;
    if (INV_STF == 0) return 1;
    if ((INV_STF != 1) && (INV_STF != 3)) return 0;
    if (!isfinite(EPS_STF) || !(EPS_STF > 0.0f)) return 0;
    /* OFFSETC_STF (m): 0 = off; > 0 restricts the estimation to traces with
     * offset < OFFSETC_STF (legacy stf.c), only with the fixed signal of
     * INV_STF = 3; variable projection needs the weights of Phi. */
    if (!isfinite(OFFSETC_STF) || (OFFSETC_STF < 0.0f)) return 0;
    if ((INV_STF == 1) && (OFFSETC_STF != 0.0f)) return 0;
    if (NORMALIZE != 0) return 0;
    if (visco_sh_exact_stf_nw() < 1) return 0;
    return 1;
}

int visco_sh_exact_stf_active(void) {
    return stfs.mode != 0;
}

/* Called on every rank at the start of each workflow stage (after the
 * stage's workflow row was read).  Local status. */
int visco_sh_exact_stf_stage_begin(int stage, int nshots) {
    extern int INV_STF;
    extern float EPS_STF, OFFSETC_STF;
    int s;
    stf_free();
    if (INV_STF == 0) return 0;
    if (!visco_sh_exact_stf_config_valid() || (nshots < 1)) return -1;
    stfs.stage = stage;
    stfs.mode = INV_STF;
    stfs.nshots = nshots;
    stfs.nw = visco_sh_exact_stf_nw();
    stfs.eps_rel = (double)EPS_STF;
    stfs.offsetc = (INV_STF == 3) ? OFFSETC_STF : 0.0f;
    stfs.eps = (double *)malloc((size_t)(nshots + 1) * sizeof(double));
    stfs.fixed_set = (int *)calloc((size_t)(nshots + 1), sizeof(int));
    stfs.last_set = (int *)calloc((size_t)(nshots + 1), sizeof(int));
    stfs.v_fixed = (double *)calloc((size_t)(nshots + 1) * stfs.nw, sizeof(double));
    stfs.v_last = (double *)calloc((size_t)(nshots + 1) * stfs.nw, sizeof(double));
    stfs.d_last = (double *)calloc((size_t)(nshots + 1), sizeof(double));
    stfs.e_last = (double *)calloc((size_t)(nshots + 1), sizeof(double));
    stfs.res_last = (double *)calloc((size_t)(nshots + 1), sizeof(double));
    stfs.it_last = (int *)calloc((size_t)(nshots + 1), sizeof(int));
    if (!stfs.eps || !stfs.fixed_set || !stfs.last_set || !stfs.v_fixed || !stfs.v_last ||
            !stfs.d_last || !stfs.e_last || !stfs.res_last || !stfs.it_last) {
        stf_free();
        return -2;
    }
    for (s = 0; s <= nshots; ++s) stfs.eps[s] = -1.0;
    return 0;
}

/* MYID 0 writes <prefix>_stage_<s>_it_<k>.stf (nshots rows of nw binary32).
 * Local status (call on every rank; only MYID 0 writes). */
int visco_sh_exact_stf_write(const char *prefix, int stage, int iteration) {
    extern int MYID;
    char name[STRING_SIZE2];
    FILE *fp;
    float *row;
    int s, j, status = 0;
    if (stfs.mode == 0 || MYID != 0) return 0;
    snprintf(name, sizeof(name), "%s_stage_%d_it_%d.stf", prefix, stage, iteration);
    row = (float *)malloc((size_t)stfs.nw * sizeof(float));
    fp = fopen(name, "wb");
    if ((row == NULL) || (fp == NULL)) status = -3;
    for (s = 1; (status == 0) && (s <= stfs.nshots); ++s) {
        double vmax = 0.0, vnorm = 0.0;
        for (j = 0; j < stfs.nw; ++j) {
            double x = stfs.last_set[s] ? stfs.v_last[(size_t)s * stfs.nw + j] : 0.0;
            row[j] = (float)x;
            if (fabs(x) > vmax) vmax = fabs(x);
            vnorm += x * x;
        }
        if (fwrite(row, sizeof(float), (size_t)stfs.nw, fp) != (size_t)stfs.nw) status = -3;
        /* data misfit and regularisation term separately, signal amplitude */
        printf("Exact SH STF: stage %d it %d shot %d D %e E %e eps %e max|v| %e |v| %e "
               "pcg_it %d relres %e\n",
               stage, iteration, s, stfs.d_last[s], stfs.e_last[s], stfs.eps[s], vmax,
               sqrt(vnorm), stfs.it_last[s], stfs.res_last[s]);
    }
    if (fp != NULL && fclose(fp) != 0) status = -3;
    free(row);
    return status;
}

int visco_sh_exact_stf_max_iterations(void) {
    return stfs.max_iterations;
}

/* Data term D of the shot prepared last (identical on every rank). */
double visco_sh_exact_stf_shot_data_term(void) {
    return stfs.dterm_shot;
}

/* Source, observed data, weights and shot objective of one shot with an
 * estimated source signal.  Replaces the wavelet/filter/weights block of the
 * wrappers.  On return 0: acquisition->signals = B v (local sources),
 * sectionread = B d (global), *data_weight as SH-S2, *phi_shot = D + eps term.
 * Collective status. */
int visco_sh_exact_stf_prepare_shot(
        const struct visco_sh_exact_multi_shot_request *request, int ishot,
        int nshots, int nsrc_local, int nrec_local, int nrec_global,
        double **data_weight, double *phi_shot) {
    extern int MYID, TIME_FILT, ORDER;
    extern float FC_START, FC_END;
    struct acq *acq = request->acquisition;
    float **section = request->legacy_fwi_seismogram->sectionread;
    int ns = request->ns, nw = stfs.nw, weighted = exact_sh_s2_weighted();
    size_t size = (size_t)(nrec_global > 0 ? nrec_global : 1) * (size_t)ns;
    float *g_loc = NULL, *g_glob = NULL, *d_glob = NULL;
    double *w_loc = NULL, *w_glob = NULL, *v = NULL;
    double head[4] = {0.0, 0.0, 0.0, 0.0};
    int status = 0, i, k, n;

    *data_weight = NULL;
    *phi_shot = 0.0;
    if ((stfs.mode == 0) || (ishot < 1) || (ishot > stfs.nshots) || (nshots != stfs.nshots))
        status = -1;
    status = exact_sh_collective_status(MPI_COMM_WORLD, "SH-S3 stage state", status);
    if (status != 0) return status;

    /* impulse e_1, B on it and on the observed traces, weights.  The signal
     * matrix has the layout of wavelet() (freed by the wrappers); QUELLART,
     * SIGNAL_FILE and the amplitude column do not enter the source. */
    acq->signals = matrix(1, nsrc_local, 1, ns);
    for (k = 1; k <= nsrc_local; ++k) {
        for (n = 1; n <= ns; ++n) acq->signals[k][n] = 0.0f;
        acq->signals[k][1] = 1.0f;
    }
    inseis(ishot, section, nrec_global, ns, 2, request->iter);
    status = exact_sh_s2_filter_shot(acq->signals, nsrc_local, section, nrec_global, ns);
    if (status == 0)
        status = exact_sh_s2_shot_weights(acq, ishot, nrec_local, nrec_global, ns, data_weight);
    status = exact_sh_collective_status(MPI_COMM_WORLD, "SH-S3 impulse source and data operators", status);
    if (status != 0) return status;

    /* impulse forward: receiver traces only */
    sh_visc(request->wave, request->pml, request->material, request->fwi,
            request->mpi, request->seismogram, request->legacy_fwi_seismogram,
            acq, request->hc, ishot, nshots, nsrc_local, ns, nrec_local,
            NULL, NULL, request->hin, request->dtinv_help, 2,
            request->request_send, request->request_receive);

    /* gather basis traces and weights to MYID 0 (one owner per receiver) */
    g_loc = (float *)calloc(size, sizeof(float));
    if (weighted) w_loc = (double *)calloc(size, sizeof(double));
    v = (double *)calloc((size_t)nw, sizeof(double));
    if ((g_loc == NULL) || (weighted && (w_loc == NULL)) || (v == NULL)) status = -2;
    if ((status == 0) && (MYID == 0)) {
        g_glob = (float *)calloc(size, sizeof(float));
        d_glob = (float *)malloc(size * sizeof(float));
        if (weighted) w_glob = (double *)calloc(size, sizeof(double));
        if ((g_glob == NULL) || (d_glob == NULL) || (weighted && (w_glob == NULL))) status = -2;
    }
    status = exact_sh_collective_status(MPI_COMM_WORLD, "SH-S3 gather buffers", status);
    if (status != 0) goto done;
    for (i = 1; i <= nrec_local; ++i) {
        int gi = acq->recpos_loc[3][i];
        for (n = 1; n <= ns; ++n) {
            g_loc[(size_t)(gi - 1) * ns + (n - 1)] = request->seismogram->sectionvz[i][n];
            if (weighted)
                w_loc[(size_t)(gi - 1) * ns + (n - 1)] = (*data_weight)[(size_t)(i - 1) * ns + (n - 1)];
        }
    }
    if (MPI_Reduce(g_loc, g_glob, (int)size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD) != MPI_SUCCESS ||
            (weighted && MPI_Reduce(w_loc, w_glob, (int)size, MPI_DOUBLE, MPI_SUM, 0,
                                    MPI_COMM_WORLD) != MPI_SUCCESS)) {
        status = exact_sh_abort_phase(MPI_COMM_WORLD, "SH-S3 gather", -2);
        goto done;
    }

    if (MYID == 0) {
        struct visco_sh_exact_stf_problem p, pe;
        double dterm = 0.0, eterm = 0.0, relres = 0.0, *w_est = NULL;
        int iterations = 0, local = 0, kept;
        for (i = 1; i <= nrec_global; ++i)
            for (n = 1; n <= ns; ++n)
                d_glob[(size_t)(i - 1) * ns + (n - 1)] = section[i][n];
        p.ntr = nrec_global; p.nt = ns; p.nw = nw;
        p.g = g_glob; p.d = d_glob; p.w = w_glob;
        pe = p;                       /* estimation problem */
        kept = nrec_global;
        if (stfs.offsetc > 0.0f) {
            /* INV_STF = 3 with OFFSETC_STF: estimation from the traces with
             * offset < OFFSETC_STF only (legacy stf.c; binary32 offset as
             * legacy calc_res); Phi keeps every trace and the weights of Phi. */
            extern float DH;
            w_est = (double *)malloc(size * sizeof(double));
            if (w_est == NULL) local = -2;
            kept = 0;
            for (i = 1; (local == 0) && (i <= nrec_global); ++i) {
                float xr = (float)acq->recpos[1][i] * DH, yr = (float)acq->recpos[2][i] * DH;
                float dx = acq->srcpos[1][ishot] - xr, dy = acq->srcpos[2][ishot] - yr;
                float offset = (float)sqrt((double)(dx * dx + dy * dy));
                int use = offset < stfs.offsetc;
                kept += use;
                for (n = 0; n < ns; ++n)
                    w_est[(size_t)(i - 1) * ns + n] = use ? (w_glob ? w_glob[(size_t)(i - 1) * ns + n] : 1.0) : 0.0;
            }
            pe.w = w_est;
        }
        if ((local == 0) && (stfs.eps[ishot] < 0.0)) {
            double md = visco_sh_exact_stf_mean_diag(&pe);
            stfs.eps[ishot] = stfs.eps_rel * md;
            printf("Exact SH STF: stage %d shot %d eps %e traces %d of %d\n", stfs.stage, ishot,
                   stfs.eps[ishot], kept, nrec_global);
            if (!(stfs.eps[ishot] > 0.0) || !isfinite(stfs.eps[ishot])) {
                local = -5;
                fprintf(stderr, "Exact SH STF: stage %d shot %d: no trace (or no weight) "
                        "left for the source estimation (OFFSETC_STF, OFFSET_MUTE, "
                        "TIMEWIN); eps = %e\n", stfs.stage, ishot, stfs.eps[ishot]);
            }
        }
        if ((local == 0) && (stfs.mode == 3) && stfs.fixed_set[ishot]) {
            memcpy(v, stfs.v_fixed + (size_t)ishot * nw, (size_t)nw * sizeof(double));
        } else if (local == 0) {
            local = visco_sh_exact_stf_solve(&pe, stfs.eps[ishot], STF_TOL, STF_MAXIT, v,
                                             &iterations, &relres);
            if (local != 0)
                fprintf(stderr, "Exact SH STF: stage %d shot %d estimation failed "
                        "(status %d, iterations %d, relative residual %e)\n",
                        stfs.stage, ishot, local, iterations, relres);
            if ((local == 0) && (stfs.mode == 3)) {
                memcpy(stfs.v_fixed + (size_t)ishot * nw, v, (size_t)nw * sizeof(double));
                stfs.fixed_set[ishot] = 1;
            }
            if (iterations > stfs.max_iterations) stfs.max_iterations = iterations;
        }
        if (local == 0)
            local = visco_sh_exact_stf_terms(&p, v, stfs.eps[ishot], &dterm, &eterm);
        if (local == 0) {
            memcpy(stfs.v_last + (size_t)ishot * nw, v, (size_t)nw * sizeof(double));
            stfs.last_set[ishot] = 1;
            stfs.d_last[ishot] = dterm;
            stfs.e_last[ishot] = eterm;
            stfs.res_last[ishot] = relres;
            stfs.it_last[ishot] = iterations;
        }
        head[0] = (double)local; head[1] = dterm; head[2] = eterm; head[3] = (double)iterations;
        free(w_est);
    }
    if ((MPI_Bcast(head, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) ||
            (MPI_Bcast(v, nw, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS)) {
        status = exact_sh_abort_phase(MPI_COMM_WORLD, "SH-S3 broadcast", -2);
        goto done;
    }
    status = (int)head[0];
    if (status != 0) goto done;     /* identical on every rank */

    /* forward source B v */
    for (k = 1; k <= nsrc_local; ++k) {
        for (n = 1; n <= ns; ++n) acq->signals[k][n] = 0.0f;
        for (n = 1; n <= nw; ++n) acq->signals[k][n] = (float)v[n - 1];
    }
    if ((TIME_FILT == 1) && (nsrc_local > 0))
        apply_tdfilt(acq->signals, nsrc_local, ns, ORDER, FC_END, FC_START);
    *phi_shot = head[1] + head[2];
    stfs.dterm_shot = head[1];

done:
    free(g_loc); free(w_loc); free(v);
    free(g_glob); free(d_glob); free(w_glob);
    return status;
}
