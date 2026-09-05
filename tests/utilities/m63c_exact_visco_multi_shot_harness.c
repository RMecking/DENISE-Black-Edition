/* C8b2-b2 orchestration harness with an analytic locked-shot stand-in. */

#include "fd.h"

int DTINV, LNORM, GRAD_FORM, N_ORDER, TIMEWIN, OFFSET_MUTE, TRKILL;
int NX, NY, NT, L, INVMAT1, SEISMO, EPRECOND;
int SWS_TAPER_CIRCULAR_PER_SHOT, TIME_FILT, INV_STF, QUELLTYPB, READREC;
int RUN_MULTIPLE_SHOTS, QUELLTYP, QUELLART, ORDER_SPIKE;
int MYID, POS[3], NPROCX, NPROCY, IENDX, IENDY;
float FC_SPIKE_1, FC_SPIKE_2, DH;
FILE *FP;

static int current_source_id;
static int bridge_calls;
static int observed_mapping_failures;
static int source_flow_failures;

float **matrix(int nrl, int nrh, int ncl, int nch) {
    int r, rows = nrh - nrl + 1, cols = nch - ncl + 1;
    float **base = calloc((size_t)rows, sizeof(*base));
    float *data = calloc((size_t)rows * cols, sizeof(*data));
    float **view;
    if (!base || !data) MPI_Abort(MPI_COMM_WORLD, 971);
    view = base - nrl;
    for (r = nrl; r <= nrh; ++r)
        view[r] = data + (size_t)(r - nrl) * cols - ncl;
    return view;
}

void free_matrix(float **m, int nrl, int nrh, int ncl, int nch) {
    (void)nrh; (void)nch;
    if (m == NULL) return;
    free(m[nrl] + ncl); free(m + nrl);
}

static int **harness_imatrix(int nrl, int nrh, int ncl, int nch) {
    int r, rows = nrh - nrl + 1, cols = nch - ncl + 1;
    int **base = calloc((size_t)rows, sizeof(*base));
    int *data = calloc((size_t)rows * cols, sizeof(*data));
    int **view;
    if (!base || !data) MPI_Abort(MPI_COMM_WORLD, 972);
    view = base - nrl;
    for (r = nrl; r <= nrh; ++r)
        view[r] = data + (size_t)(r - nrl) * cols - ncl;
    return view;
}

void free_imatrix(int **m, int nrl, int nrh, int ncl, int nch) {
    (void)nrh; (void)nch;
    if (m == NULL) return;
    free(m[nrl] + ncl); free(m + nrl);
}

int *ivector(int nl, int nh) {
    int *base = calloc((size_t)(nh - nl + 1), sizeof(*base));
    if (!base) MPI_Abort(MPI_COMM_WORLD, 973);
    return base - nl;
}

void free_ivector(int *v, int nl, int nh) {
    (void)nh;
    if (v != NULL) free(v + nl);
}

int **receiver(FILE *fp, int *ntr, int ishot) {
    int **rec = harness_imatrix(1, 3, 1, 1);
    (void)fp; (void)ishot;
    *ntr = 1;
    rec[1][1] = NPROCX * NX - 3;
    rec[2][1] = NY / 2;
    rec[3][1] = 1;
    return rec;
}

int **splitrec(int **recpos, int *ntr_loc, int ntr, int *recswitch) {
    int owner, **local = NULL;
    (void)ntr;
    owner = (recpos[1][1] - 1) / NX;
    *ntr_loc = POS[1] == owner ? 1 : 0;
    recswitch[1] = *ntr_loc;
    if (*ntr_loc) {
        local = harness_imatrix(1, 3, 1, 1);
        local[1][1] = (recpos[1][1] - 1) % NX + 1;
        local[2][1] = recpos[2][1];
        local[3][1] = 1;
    }
    return local;
}

float **splitsrc(float **srcpos, int *nsrc_loc, int nsrc) {
    int s, count = 0, k = 0;
    float **local = NULL;
    current_source_id = iround(srcpos[3][1]);
    for (s = 1; s <= nsrc; ++s)
        if ((iround(srcpos[1][s] / DH) - 1) / NX == POS[1]) ++count;
    *nsrc_loc = count;
    if (count == 0) return NULL;
    local = matrix(1, 8, 1, count);
    for (s = 1; s <= nsrc; ++s) {
        int owner = (iround(srcpos[1][s] / DH) - 1) / NX;
        int row;
        if (owner != POS[1]) continue;
        ++k;
        for (row = 1; row <= 8; ++row) local[row][k] = srcpos[row][s];
        local[1][k] = (float)((iround(srcpos[1][s] / DH) - 1) % NX + 1);
        local[2][k] = (float)((iround(srcpos[2][s] / DH) - 1) % NY + 1);
    }
    return local;
}

float **wavelet(float **srcpos_loc, int nsrc_loc, int ishot) {
    float **signals;
    int s, n;
    (void)ishot;
    if (nsrc_loc == 0) return NULL;
    signals = matrix(1, nsrc_loc, 1, NT);
    for (s = 1; s <= nsrc_loc; ++s)
        for (n = 1; n <= NT; ++n)
            signals[s][n] = (float)(0.01 * srcpos_loc[3][s] * (n + 2));
    return signals;
}

void apply_tdfilt(float **section, int ntr, int ns, int order,
                  float fc2, float fc1) {
    (void)section; (void)ntr; (void)ns; (void)order; (void)fc2; (void)fc1;
    MPI_Abort(MPI_COMM_WORLD, 974);
}

void inseis(int comp, float **section, int ntr, int ns, int sws, int iter) {
    int r, n;
    (void)comp; (void)sws; (void)iter;
    for (r = 1; r <= ntr; ++r)
        for (n = 1; n <= ns; ++n)
            section[r][n] = (float)(0.02 * current_source_id + 0.001 * n);
}

void alloc_seisSH(int ntr, int ns, struct seisSH *seis) {
    seis->sectionvz = ntr ? matrix(1, ntr, 1, ns) : NULL;
}

void alloc_seisSHfwi(int ntr, int ntr_glob, int ns, struct seisSHfwi *seis) {
    seis->sectionread = matrix(1, ntr_glob, 1, ns);
    seis->sectionvzdata = ntr ? matrix(1, ntr, 1, ns) : NULL;
    seis->sectionvzdiff = ntr ? matrix(1, ntr, 1, ns) : NULL;
    seis->sectionvzdiffold = ntr ? matrix(1, ntr, 1, ns) : NULL;
}

static double primary_direction(int gj, int gi) {
    return 0.11 + 0.002 * gi - 0.001 * gj;
}

static double rho_direction(int gj, int gi) {
    return -0.07 + 0.0015 * gj + 0.0003 * gi;
}

static double q_direction(int gj, int gi) {
    return 0.09 - 0.0007 * gi + 0.0004 * gj;
}

int visco_sh_exact_objective_gradient_shot(
        const struct visco_sh_exact_shot_request *request,
        struct visco_sh_exact_shot_result *result) {
    double local_observed = 0.0, observed = 0.0;
    double local_signal = 0.0, signal = 0.0, local_objective = 0.0;
    int i, j, n, s;
    ++bridge_calls;
    for (i = 1; i <= request->nrec_local; ++i)
        for (n = 1; n <= request->ns; ++n) {
            double expected = 0.02 * current_source_id + 0.001 * n;
            if (request->observed_vz[i][n] != (float)expected)
                ++observed_mapping_failures;
            local_observed += request->observed_vz[i][n];
        }
    for (s = 1; s <= request->nsrc_local; ++s)
        for (n = 1; n <= request->ns; ++n) {
            double expected = 0.01 *
                request->acquisition->srcpos_loc[3][s] * (n + 2);
            if (request->acquisition->signals[s][n] != (float)expected)
                ++source_flow_failures;
            local_signal += request->acquisition->signals[s][n];
        }
    MPI_Allreduce(&local_observed, &observed, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&local_signal, &signal, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i) {
            int gi = POS[1] * NX + i, gj = POS[2] * NY + j;
            double ap = 0.004 + 0.00003 * gi + 0.00001 * current_source_id;
            double ar = -0.003 + 0.00002 * gj;
            double aq = 0.002 + 0.000001 * (gi + gj);
            double target = 0.01 * observed + 0.005 * signal;
            double z = ap * request->material->pu[j][i] +
                       ar * request->material->prho[j][i] +
                       aq * request->material->pqs[j][i] - target;
            local_objective += 0.5 * z * z;
            request->grad_primary[j][i] = (float)(ap * z);
            request->grad_rho[j][i] = (float)(ar * z);
            request->grad_q[j][i] = (float)(aq * z);
        }
    MPI_Allreduce(&local_objective, &result->objective, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    return 0;
}

struct run_storage {
    struct waveSH wave;
    struct waveSH_PML pml;
    struct matSH material;
    struct fwiSH fwi;
    struct mpiPSV mpi;
    struct seisSH seismogram;
    struct seisSHfwi legacy;
    struct acq acquisition;
    float **primary, **rho, **q, **grad_primary, **grad_rho, **grad_q;
    int *dtinv_help;
};

static void fill_model(struct run_storage *run, double epsilon) {
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i) {
            int gi = POS[1] * NX + i, gj = POS[2] * NY + j;
            run->primary[j][i] = (float)(2.1 + 0.002 * gi +
                    epsilon * primary_direction(gj, gi));
            run->rho[j][i] = (float)(1.8 + 0.001 * gj +
                    epsilon * rho_direction(gj, gi));
            run->q[j][i] = (float)(35.0 + 0.02 * (gi + gj) +
                    epsilon * q_direction(gj, gi));
        }
}

static void init_run(struct run_storage *run, int readrec) {
    int global_receiver_x = NPROCX * NX - 3;
    int receiver_owner = (global_receiver_x - 1) / NX;
    memset(run, 0, sizeof(*run));
    run->primary = matrix(1, NY, 1, NX);
    run->rho = matrix(1, NY, 1, NX);
    run->q = matrix(1, NY, 1, NX);
    run->grad_primary = matrix(1, NY, 1, NX);
    run->grad_rho = matrix(1, NY, 1, NX);
    run->grad_q = matrix(1, NY, 1, NX);
    run->material.pu = run->primary;
    run->material.prho = run->rho;
    run->material.pqs = run->q;
    run->acquisition.srcpos = matrix(1, 8, 1, 2);
    run->acquisition.srcpos1 = matrix(1, 8, 1, 1);
    run->dtinv_help = ivector(1, NT);
    run->acquisition.srcpos[1][1] = 3.0f * DH;
    run->acquisition.srcpos[2][1] = (NY / 2.0f) * DH;
    run->acquisition.srcpos[3][1] = 1.0f;
    run->acquisition.srcpos[8][1] = 1.0f;
    run->acquisition.srcpos[1][2] = (NPROCX * NX - 4.0f) * DH;
    run->acquisition.srcpos[2][2] = (NY / 2.0f) * DH;
    run->acquisition.srcpos[3][2] = 2.0f;
    run->acquisition.srcpos[8][2] = 1.0f;
    if (!readrec) {
        run->acquisition.recpos = harness_imatrix(1, 3, 1, 1);
        run->acquisition.recpos[1][1] = global_receiver_x;
        run->acquisition.recpos[2][1] = NY / 2;
        run->acquisition.recpos[3][1] = 1;
        run->acquisition.recswitch = ivector(1, 1);
        run->acquisition.recpos_loc = splitrec(
                run->acquisition.recpos, &(int){0}, 1,
                run->acquisition.recswitch);
        if (POS[1] == receiver_owner) {
            run->seismogram.sectionvz = matrix(1, 1, 1, NT);
            run->legacy.sectionvzdata = matrix(1, 1, 1, NT);
            run->legacy.sectionvzdiff = matrix(1, 1, 1, NT);
            run->legacy.sectionvzdiffold = matrix(1, 1, 1, NT);
        }
        run->legacy.sectionread = matrix(1, 1, 1, NT);
    }
}

static void release_run(struct run_storage *run, int readrec) {
    int receiver_owner = ((NPROCX * NX - 3) - 1) / NX;
    if (!readrec) {
        if (POS[1] == receiver_owner) {
            free_matrix(run->seismogram.sectionvz, 1, 1, 1, NT);
            free_matrix(run->legacy.sectionvzdata, 1, 1, 1, NT);
            free_matrix(run->legacy.sectionvzdiff, 1, 1, 1, NT);
            free_matrix(run->legacy.sectionvzdiffold, 1, 1, 1, NT);
            free_imatrix(run->acquisition.recpos_loc, 1, 3, 1, 1);
        }
        free_matrix(run->legacy.sectionread, 1, 1, 1, NT);
        free_ivector(run->acquisition.recswitch, 1, 1);
        free_imatrix(run->acquisition.recpos, 1, 3, 1, 1);
    }
    free_ivector(run->dtinv_help, 1, NT);
    free_matrix(run->acquisition.srcpos1, 1, 8, 1, 1);
    free_matrix(run->acquisition.srcpos, 1, 8, 1, 2);
    free_matrix(run->grad_q, 1, NY, 1, NX);
    free_matrix(run->grad_rho, 1, NY, 1, NX);
    free_matrix(run->grad_primary, 1, NY, 1, NX);
    free_matrix(run->q, 1, NY, 1, NX);
    free_matrix(run->rho, 1, NY, 1, NX);
    free_matrix(run->primary, 1, NY, 1, NX);
}

static int aggregate(struct run_storage *run, int nsrc, double *objective) {
    struct visco_sh_exact_multi_shot_request request;
    struct visco_sh_exact_multi_shot_result result;
    int receiver_owner = ((NPROCX * NX - 3) - 1) / NX;
    memset(&request, 0, sizeof(request)); memset(&result, 0, sizeof(result));
    request.wave = &run->wave; request.pml = &run->pml;
    request.material = &run->material; request.fwi = &run->fwi;
    request.mpi = &run->mpi; request.seismogram = &run->seismogram;
    request.legacy_fwi_seismogram = &run->legacy;
    request.acquisition = &run->acquisition; request.hc = (float[]){0, 1};
    request.iter = 1; request.nsrc = nsrc; request.ns = NT;
    request.nrec_local = POS[1] == receiver_owner;
    request.nrec_global = 1; request.hin = 1;
    request.dtinv_help = run->dtinv_help;
    request.grad_primary = run->grad_primary;
    request.grad_rho = run->grad_rho; request.grad_q = run->grad_q;
    if (visco_sh_exact_objective_gradient(&request, &result) != 0) return -1;
    if (result.shot_count != (RUN_MULTIPLE_SHOTS ? nsrc : 1)) return -1;
    *objective = result.objective;
    return 0;
}

static double contraction(struct run_storage *run) {
    double local = 0.0, global;
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i) {
            int gi = POS[1] * NX + i, gj = POS[2] * NY + j;
            local += run->grad_primary[j][i] * primary_direction(gj, gi);
            local += run->grad_rho[j][i] * rho_direction(gj, gi);
            local += run->grad_q[j][i] * q_direction(gj, gi);
        }
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

static void copy_gradient(float **dst, float **src) {
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i) dst[j][i] = src[j][i];
}

int main(int argc, char **argv) {
    const double epsilons[] = {1.0e-2, 3.0e-3};
    struct run_storage run;
    float **base_primary, **base_rho, **base_q;
    double objective, repeat_objective, d_ad, repeat_d_ad;
    double single_objective_sum = 0.0, max_gradient_sum_error = 0.0;
    int rank, size, readrec, e, i, j, s, saved_calls;
    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size); MYID = rank; FP = stderr;
    if (argc != 2) MPI_Abort(MPI_COMM_WORLD, 975);
    NPROCX = size; NPROCY = 1; POS[1] = rank; POS[2] = 0;
    NX = 7; NY = 6; IENDX = NX; IENDY = NY; NT = 8; L = 2; DH = 5.0f;
    INVMAT1 = strstr(argv[1], "m3") ? 3 : 1;
    readrec = strstr(argv[1], "readrec1") == NULL; READREC = readrec ? 2 : 1;
    RUN_MULTIPLE_SHOTS = 1; DTINV = 1; LNORM = 2; GRAD_FORM = 2;
    N_ORDER = TIMEWIN = OFFSET_MUTE = TRKILL = 0; SEISMO = 1;
    EPRECOND = SWS_TAPER_CIRCULAR_PER_SHOT = TIME_FILT = INV_STF = 0;
    QUELLTYPB = 1; QUELLART = 1; ORDER_SPIKE = 0;
    init_run(&run, readrec); fill_model(&run, 0.0);
    if (aggregate(&run, 2, &objective) != 0) MPI_Abort(MPI_COMM_WORLD, 976);
    d_ad = contraction(&run);
    base_primary = matrix(1, NY, 1, NX); base_rho = matrix(1, NY, 1, NX);
    base_q = matrix(1, NY, 1, NX);
    copy_gradient(base_primary, run.grad_primary);
    copy_gradient(base_rho, run.grad_rho); copy_gradient(base_q, run.grad_q);

    fill_model(&run, 0.0);
    if (aggregate(&run, 2, &repeat_objective) != 0)
        MPI_Abort(MPI_COMM_WORLD, 977);
    repeat_d_ad = contraction(&run);
    for (s = 1; s <= 2; ++s) {
        float **all = run.acquisition.srcpos;
        float **one = matrix(1, 8, 1, 1);
        double one_objective;
        int row;
        for (row = 1; row <= 8; ++row) one[row][1] = all[row][s];
        run.acquisition.srcpos = one; RUN_MULTIPLE_SHOTS = 0;
        fill_model(&run, 0.0);
        if (aggregate(&run, 1, &one_objective) != 0)
            MPI_Abort(MPI_COMM_WORLD, 978);
        single_objective_sum += one_objective;
        for (j = 1; j <= NY; ++j)
            for (i = 1; i <= NX; ++i) {
                if (s == 1) {
                    base_primary[j][i] -= run.grad_primary[j][i];
                    base_rho[j][i] -= run.grad_rho[j][i];
                    base_q[j][i] -= run.grad_q[j][i];
                } else {
                    max_gradient_sum_error = fmax(max_gradient_sum_error,
                        fabs(base_primary[j][i] - run.grad_primary[j][i]));
                    max_gradient_sum_error = fmax(max_gradient_sum_error,
                        fabs(base_rho[j][i] - run.grad_rho[j][i]));
                    max_gradient_sum_error = fmax(max_gradient_sum_error,
                        fabs(base_q[j][i] - run.grad_q[j][i]));
                }
            }
        run.acquisition.srcpos = all; free_matrix(one, 1, 8, 1, 1);
    }
    RUN_MULTIPLE_SHOTS = 1;
    MPI_Allreduce(MPI_IN_PLACE, &max_gradient_sum_error, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    saved_calls = bridge_calls;
    {
        float sentinel_primary = run.grad_primary[1][1];
        float sentinel_rho = run.grad_rho[1][1];
        float sentinel_q = run.grad_q[1][1];
        EPRECOND = 1;
        if (aggregate(&run, 2, &repeat_objective) == 0 ||
                bridge_calls != saved_calls ||
                run.grad_primary[1][1] != sentinel_primary ||
                run.grad_rho[1][1] != sentinel_rho ||
                run.grad_q[1][1] != sentinel_q)
            MPI_Abort(MPI_COMM_WORLD, 979);
    }
    EPRECOND = 0;
    if (rank == 0)
        printf("{\"case\":\"%s\",\"ranks\":%d,\"invmat1\":%d,"
               "\"q_mode\":\"%s\",\"shots\":2,\"objective\":%.17g,"
               "\"single_objective_sum\":%.17g,\"D_ad\":%.17g,"
               "\"repeat_objective\":%.17g,\"repeat_D_ad\":%.17g,"
               "\"max_gradient_sum_error\":%.17g,\"bridge_calls\":%d,"
               "\"observed_mapping_failures\":%d,"
               "\"source_flow_failures\":%d,\"precondition_rejected\":true,"
               "\"precondition_outputs_unchanged\":true,"
               "\"source1_owner\":0,\"receiver_owner\":%d,"
               "\"cross_rank_activation\":%s}\n",
               argv[1], size, INVMAT1, INVMAT1 == 1 ? "physical" : "legacy",
               objective, single_objective_sum, d_ad, repeat_objective,
               repeat_d_ad, max_gradient_sum_error, bridge_calls,
               observed_mapping_failures, source_flow_failures, NPROCX - 1,
               NPROCX > 1 ? "true" : "false");
    for (e = 0; e < 2; ++e) {
        double jp, jm, dfd, rel;
        fill_model(&run, epsilons[e]);
        if (aggregate(&run, 2, &jp) != 0) MPI_Abort(MPI_COMM_WORLD, 980);
        fill_model(&run, -epsilons[e]);
        if (aggregate(&run, 2, &jm) != 0) MPI_Abort(MPI_COMM_WORLD, 981);
        dfd = (jp - jm) / (2.0 * epsilons[e]);
        rel = fabs(dfd - d_ad) / fmax(fmax(fabs(dfd), fabs(d_ad)), 1.0e-300);
        if (rank == 0)
            printf("{\"case\":\"%s\",\"epsilon\":%.17g,"
                   "\"J_plus\":%.17g,\"J_minus\":%.17g,"
                   "\"D_fd\":%.17g,\"D_ad\":%.17g,"
                   "\"relative_error\":%.17g}\n",
                   argv[1], epsilons[e], jp, jm, dfd, d_ad, rel);
    }
    free_matrix(base_q, 1, NY, 1, NX); free_matrix(base_rho, 1, NY, 1, NX);
    free_matrix(base_primary, 1, NY, 1, NX); release_run(&run, readrec);
    MPI_Finalize(); return 0;
}
