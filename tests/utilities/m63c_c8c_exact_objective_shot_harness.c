/* Direct runtime oracle for C8c B3a's inactive exact SH objective-only path. */

#ifndef M63C_DIRECTIONAL_SUPPORT
#define M63C_DIRECTIONAL_SUPPORT "m63c_objective_directional_fd_harness.c"
#endif
#include M63C_DIRECTIONAL_SUPPORT

float TSNAP1, TSNAP2, TSNAPINC, *FL, Q_APPROX_FMIN, Q_APPROX_FMAX;
float Q_APPROX_DF, FC_SPIKE_1, FC_SPIKE_2;
int Q_PARAMETERIZATION_MODE, NDT, SEISMO, IDXI, IDYI, DTINV, SNAP;
int INV_STF, EPRECOND, NTDTINV, NXNYI, NT, QUELLART, ORDER_SPIKE;
int LNORM, N_ORDER, TIMEWIN, OFFSET_MUTE, TRKILL;

float *vector(int nl, int nh) {
    float *base = (float *)calloc((size_t)(nh - nl + 1), sizeof(float));
    if (base == NULL) MPI_Abort(MPI_COMM_WORLD, 970);
    return base - nl;
}

void free_vector(float *v, int nl, int nh) { (void)nh; free(v + nl); }

float ***f3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh) {
    int j, i, rows = nrh - nrl + 1, cols = nch - ncl + 1, depth = ndh - ndl + 1;
    float ***row_base = (float ***)calloc((size_t)rows, sizeof(float **));
    float **col_base = (float **)calloc((size_t)rows * cols, sizeof(float *));
    float *data = (float *)calloc((size_t)rows * cols * depth, sizeof(float));
    float ***view;
    if (!row_base || !col_base || !data) MPI_Abort(MPI_COMM_WORLD, 971);
    view = row_base - nrl;
    for (j = nrl; j <= nrh; ++j) {
        view[j] = col_base + (j - nrl) * cols - ncl;
        for (i = ncl; i <= nch; ++i)
            view[j][i] = data + ((j - nrl) * cols + i - ncl) * depth - ndl;
    }
    return view;
}

void free_f3tensor(float ***t, int nrl, int nrh, int ncl, int nch, int ndl, int ndh) {
    (void)nrh; (void)nch; (void)ndh;
    free(t[nrl][ncl] + ndl); free(t[nrl] + ncl); free(t + nrl);
}

static int **test_imatrix(int nrl, int nrh, int ncl, int nch) {
    int j, rows = nrh - nrl + 1, cols = nch - ncl + 1;
    int **base = (int **)calloc((size_t)rows, sizeof(int *));
    int *data = (int *)calloc((size_t)rows * cols, sizeof(int));
    int **view;
    if (!base || !data) MPI_Abort(MPI_COMM_WORLD, 972);
    view = base - nrl;
    for (j = nrl; j <= nrh; ++j) view[j] = data + (j - nrl) * cols - ncl;
    return view;
}

void snap(FILE *fp, int nt, int nsnap, float **vx, float **vy, float **sxx,
          float **syy, float **u, float **pi, float *hc) {
    (void)fp; (void)nt; (void)nsnap; (void)vx; (void)vy; (void)sxx;
    (void)syy; (void)u; (void)pi; (void)hc; MPI_Abort(MPI_COMM_WORLD, 973);
}

void eprecond_SH(float **W, float **vz) {
    (void)W; (void)vz; MPI_Abort(MPI_COMM_WORLD, 974);
}

struct bridge_storage {
    struct waveSH wave;
    struct waveSH_PML pml;
    struct fwiSH fwi;
    struct mpiPSV mpi;
    struct seisSH seismogram;
    struct seisSHfwi legacy;
    struct acq acquisition;
    int **recpos;
    float **observed;
    int *dtinv_help;
};

static void bridge_storage_init(struct bridge_storage *shot, int nsteps,
        int source_i, int source_j, int receiver_i, int receiver_j,
        float **srcpos, float **signals, float **bl, float **br, float **bt,
        float **bb, const float *K, const float *Kh, const float *a,
        const float *ah, const float *b, const float *bh) {
    int k;
    memset(shot, 0, sizeof(*shot)); alloc_SH(&shot->wave, &shot->pml);
    for (k = 1; k <= 2 * FW; ++k) {
        shot->pml.K_x[k] = shot->pml.K_y[k] = K[k];
        shot->pml.K_x_half[k] = shot->pml.K_y_half[k] = Kh[k];
        shot->pml.a_x[k] = shot->pml.a_y[k] = a[k];
        shot->pml.a_x_half[k] = shot->pml.a_y_half[k] = ah[k];
        shot->pml.b_x[k] = shot->pml.b_y[k] = b[k];
        shot->pml.b_x_half[k] = shot->pml.b_y_half[k] = bh[k];
    }
    shot->mpi.bufferlef_to_rig = bl; shot->mpi.bufferrig_to_lef = br;
    shot->mpi.buffertop_to_bot = bt; shot->mpi.bufferbot_to_top = bb;
    shot->acquisition.srcpos_loc = srcpos; shot->acquisition.signals = signals;
    shot->recpos = test_imatrix(1, 2, 1, 1);
    shot->recpos[1][1] = receiver_i; shot->recpos[2][1] = receiver_j;
    shot->acquisition.recpos_loc = shot->recpos;
    shot->seismogram.sectionvz = matrix(1, 1, 1, nsteps);
    shot->observed = matrix(1, 1, 1, nsteps);
    shot->dtinv_help = (int *)calloc((size_t)nsteps + 1, sizeof(int));
    if (shot->dtinv_help == NULL) MPI_Abort(MPI_COMM_WORLD, 975);
    (void)source_i; (void)source_j;
}

static void bridge_bind_material(struct matSH *material, struct field *rho,
        struct field *rhoi, struct field *primary, struct field *mu_x,
        struct field *mu_y, struct field *q, struct field *tau_y,
        struct field *tau_x, struct field *fipjp, struct field *f,
        struct volume *dip, struct volume *d, struct volume *pp, float *eta,
        float *bip, float *bjm, float *cip, float *cjm) {
    memset(material, 0, sizeof(*material));
    material->prho = rho->v; material->prhoi = rhoi->v; material->pu = primary->v;
    material->puip = mu_x->v; material->pujp = mu_y->v; material->pqs = q->v;
    material->ptaus = tau_y->v; material->ptausipjp = tau_x->v;
    material->fipjp = fipjp->v; material->f = f->v; material->dip = dip->v;
    material->d = d->v; material->e = pp->v; material->etaip = eta;
    material->etajm = eta; material->bip = bip; material->bjm = bjm;
    material->cip = cip; material->cjm = cjm;
}

static int adjoint_calls, forward_calls;

void __real_sh_visc(struct waveSH *, struct waveSH_PML *, struct matSH *,
        struct fwiSH *, struct mpiPSV *, struct seisSH *, struct seisSHfwi *,
        struct acq *, float *, int, int, int, int, int, float **, float **,
        int, int *, int, MPI_Request *, MPI_Request *);

void __wrap_sh_visc(struct waveSH *wave, struct waveSH_PML *pml,
        struct matSH *material, struct fwiSH *fwi, struct mpiPSV *mpi,
        struct seisSH *seismogram, struct seisSHfwi *legacy,
        struct acq *acquisition, float *hc, int ishot, int nshots,
        int nsrc_local, int ns, int nrec_local, float **source_energy,
        float **receiver_energy, int hin, int *dtinv_help, int mode,
        MPI_Request *send, MPI_Request *receive) {
    ++forward_calls;
    __real_sh_visc(wave, pml, material, fwi, mpi, seismogram, legacy,
            acquisition, hc, ishot, nshots, nsrc_local, ns, nrec_local,
            source_energy, receiver_energy, hin, dtinv_help, mode, send,
            receive);
}

int __real_visco_sh_reverse_time_adjoint_material(
        const struct visco_sh_full_step_config *, int, const double *,
        struct visco_sh_full_state *, struct visco_sh_full_state *,
        struct visco_sh_full_state *, double *,
        const struct visco_sh_reverse_time_material_context *);

int __wrap_visco_sh_reverse_time_adjoint_material(
        const struct visco_sh_full_step_config *config, int steps,
        const double *receiver, struct visco_sh_full_state *terminal,
        struct visco_sh_full_state *initial, struct visco_sh_full_state *scratch,
        double *signal, const struct visco_sh_reverse_time_material_context *material) {
    ++adjoint_calls;
    return __real_visco_sh_reverse_time_adjoint_material(config, steps, receiver,
            terminal, initial, scratch, signal, material);
}

static void die(const char *message) {
    fprintf(stderr, "objective-shot oracle failure: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 981);
}

static void prepare_globals(int free_surface, int steps) {
    MYID = 0; FP = stderr; NPROCX = NPROCY = 1;
    NX = 18; NY = 20; FDORDER = 4; L = 2; FW = 0;
    FREE_SURF = free_surface; BOUNDARY = 0; INVMAT1 = 1;
    DT = 0.0013f; DH = 7.5f; GRAD_FORM = 2; ADJ_SIGN = 1; MODE = 0;
    QUELLTYP = QUELLTYPB = QUELLART = 1; topology(0);
    NT = steps; DTINV = NDT = 1; SEISMO = 1; SNAP = 0; INV_STF = 0;
    EPRECOND = 0; LNORM = 2; N_ORDER = 0; TIMEWIN = OFFSET_MUTE = TRKILL = 0;
    IDXI = IDYI = 1; NTDTINV = NT; NXNYI = NX * NY;
    TSNAP1 = TSNAP2 = TSNAPINC = 1.0f;
}

static void bind_objective_request(
        struct visco_sh_exact_objective_shot_request *request,
        struct bridge_storage *shot, struct matSH *material, float *hc,
        MPI_Request *requests) {
    memset(request, 0, sizeof(*request));
    request->wave = &shot->wave; request->pml = &shot->pml;
    request->material = material; request->fwi = &shot->fwi;
    request->mpi = &shot->mpi; request->seismogram = &shot->seismogram;
    request->legacy_fwi_seismogram = &shot->legacy;
    request->acquisition = &shot->acquisition; request->hc = hc;
    request->ishot = request->nshots = request->nsrc_local = 1;
    request->ns = NT; request->nrec_local = 1; request->hin = 1;
    request->dtinv_help = shot->dtinv_help; request->observed_vz = shot->observed;
    request->request_send = requests; request->request_receive = requests;
}

static void bind_gradient_request(struct visco_sh_exact_shot_request *request,
        struct bridge_storage *shot, struct matSH *material, float *hc,
        MPI_Request *requests, struct field *primary, struct field *rho,
        struct field *q) {
    memset(request, 0, sizeof(*request));
    request->wave = &shot->wave; request->pml = &shot->pml;
    request->material = material; request->fwi = &shot->fwi;
    request->mpi = &shot->mpi; request->seismogram = &shot->seismogram;
    request->legacy_fwi_seismogram = &shot->legacy;
    request->acquisition = &shot->acquisition; request->hc = hc;
    request->ishot = request->nshots = request->nsrc_local = 1;
    request->ns = NT; request->nrec_local = 1; request->hin = 1;
    request->dtinv_help = shot->dtinv_help; request->observed_vz = shot->observed;
    request->grad_primary = primary->v; request->grad_rho = rho->v;
    request->grad_q = q->v; request->request_send = requests;
    request->request_receive = requests;
}

static int expect_preflight(
        const struct visco_sh_exact_objective_shot_request *request) {
    struct visco_sh_exact_objective_shot_result result;
    result.objective = -789123.25;
    forward_calls = 0;
    return visco_sh_exact_objective_shot(request, &result) != 0 &&
            result.objective == -789123.25 && forward_calls == 0;
}

static int run_preflight_contract(
        const struct visco_sh_exact_objective_shot_request *valid) {
    struct visco_sh_exact_objective_shot_request request;
    struct visco_sh_exact_objective_shot_result result;
    struct acq acquisition;
    int saved;

    result.objective = -789123.25; forward_calls = 0;
    if (visco_sh_exact_objective_shot(NULL, &result) == 0 ||
            result.objective != -789123.25 || forward_calls != 0) return 0;
    if (visco_sh_exact_objective_shot(valid, NULL) == 0) return 0;
#define BAD(statement) do { request = *valid; statement; if (!expect_preflight(&request)) return 0; } while (0)
    BAD(request.wave = NULL); BAD(request.ns = NT - 1);
    BAD(request.nrec_local = -1); BAD(request.nsrc_local = -1);
    saved = INVMAT1; INVMAT1 = 2; if (!expect_preflight(valid)) return 0; INVMAT1 = saved;
    saved = DTINV; DTINV = 2; if (!expect_preflight(valid)) return 0; DTINV = saved;
    saved = LNORM; LNORM = 1; if (!expect_preflight(valid)) return 0; LNORM = saved;
    saved = N_ORDER; N_ORDER = 1; if (!expect_preflight(valid)) return 0; N_ORDER = saved;
    saved = TIMEWIN; TIMEWIN = 1; if (!expect_preflight(valid)) return 0; TIMEWIN = saved;
    saved = OFFSET_MUTE; OFFSET_MUTE = 1; if (!expect_preflight(valid)) return 0; OFFSET_MUTE = saved;
    saved = TRKILL; TRKILL = 1; if (!expect_preflight(valid)) return 0; TRKILL = saved;
    saved = SEISMO; SEISMO = 0; if (!expect_preflight(valid)) return 0; SEISMO = saved;
    BAD(request.observed_vz = NULL);
    request = *valid; acquisition = *valid->acquisition;
    request.acquisition = &acquisition; acquisition.recpos_loc = NULL;
    if (!expect_preflight(&request)) return 0;
    request = *valid; acquisition = *valid->acquisition;
    request.acquisition = &acquisition; acquisition.srcpos_loc = NULL;
    if (!expect_preflight(&request)) return 0;
    request = *valid; acquisition = *valid->acquisition;
    request.acquisition = &acquisition; acquisition.signals = NULL;
    if (!expect_preflight(&request)) return 0;
#undef BAD
    return 1;
}

static void run_case(int free_surface) {
    const int steps = 48, source_i = 5, receiver_i = 11;
    const int source_j = free_surface ? 3 : 9;
    const int receiver_j = free_surface ? 3 : 9;
    struct directional_case test = {"objective", 1, Q_PARAMETERIZATION_PHYSICAL,
            DIRECTION_PRIMARY | DIRECTION_RHO | DIRECTION_Q,
            1, 1, free_surface, 0, steps, source_i, source_j,
            receiver_i, receiver_j};
    struct field rhoi, fipjp, f, primary, rho, q, mu, tau, mu_x, mu_y;
    struct field tau_x, tau_y, grad_primary, grad_rho, grad_q;
    struct volume dip, d, pp;
    struct q_tau_mapping mapping;
    struct matSH material;
    struct bridge_storage truth, only, gradient;
    struct visco_sh_exact_objective_shot_request only_request, truth_request;
    struct visco_sh_exact_shot_request gradient_request;
    struct visco_sh_exact_objective_shot_result only_result, truth_result;
    struct visco_sh_exact_shot_result gradient_result;
    float *hc, *bip, *bjm, *cip, *cjm, *eta, *frequencies;
    float *K, *Kh, *a, *ah, *b, *bh;
    float **srcpos, **signals, *src_storage, *signal_storage;
    float **bl, **br, **bt, **bb, *sl, *sr, *st, *sb;
    MPI_Request requests[4];
    double reference_sum = 0.0, manual = 0.0, including_first = 0.0;
    double relative, manual_relative, included_difference;
    int h, i, j, l, n, objective_adjoint, gradient_adjoint, objective_forward;

    prepare_globals(free_surface, steps); h = FDORDER / 2;
#define FIELD(name) name = field_new(-h, NY + h + 1, 1 - h, NX + h)
    FIELD(rhoi); FIELD(fipjp); FIELD(f); FIELD(primary); FIELD(rho); FIELD(q);
    FIELD(mu); FIELD(tau); FIELD(mu_x); FIELD(mu_y); FIELD(tau_x); FIELD(tau_y);
    FIELD(grad_primary); FIELD(grad_rho); FIELD(grad_q);
#undef FIELD
    dip = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    d = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    pp = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    hc = (float *)calloc(3, sizeof(float)); hc[1] = 1.0f; hc[2] = -0.041f;
    bip = (float *)calloc(3, sizeof(float)); bjm = (float *)calloc(3, sizeof(float));
    cip = (float *)calloc(3, sizeof(float)); cjm = (float *)calloc(3, sizeof(float));
    eta = (float *)calloc(3, sizeof(float)); frequencies = (float *)calloc(3, sizeof(float));
    frequencies[1] = 4.0f; frequencies[2] = 9.0f; FL = frequencies;
    Q_PARAMETERIZATION_MODE = Q_PARAMETERIZATION_PHYSICAL;
    Q_APPROX_FMIN = 2.0f; Q_APPROX_FMAX = 18.0f; Q_APPROX_DF = 0.5f;
    init_q_tau_mapping(&mapping, Q_PARAMETERIZATION_MODE, L, FL,
            Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF);
    for (l = 1; l <= L; ++l) {
        double theta = 1.0 / (2.0 * PI * frequencies[l]);
        double x = 2.0 * PI * frequencies[1] * theta;
        eta[l] = (float)(DT / theta); reference_sum += x * x / (1.0 + x * x);
        bip[l] = bjm[l] = 1.0f / (1.0f + 0.5f * eta[l]);
        cip[l] = cjm[l] = 1.0f - 0.5f * eta[l];
    }
    K = (float *)calloc(1, sizeof(float)); Kh = (float *)calloc(1, sizeof(float));
    a = (float *)calloc(1, sizeof(float)); ah = (float *)calloc(1, sizeof(float));
    b = (float *)calloc(1, sizeof(float)); bh = (float *)calloc(1, sizeof(float));
    srcpos = buffer_new(8, 1, &src_storage); signals = buffer_new(1, steps, &signal_storage);
    srcpos[1][1] = (float)source_i; srcpos[2][1] = (float)source_j; srcpos[8][1] = 1.0f;
    for (n = 0; n < steps; ++n) {
        double x = (n - 9.0) / 3.1;
        signals[1][n + 1] = (float)(0.014 * (1.0 - 2.0 * x * x) * exp(-x * x));
    }
    bl = buffer_new(NY, 2 * (h + 1), &sl); br = buffer_new(NY, 2 * (h + 1), &sr);
    bt = buffer_new(NX, 2 * (h + 1), &st); bb = buffer_new(NX, 2 * (h + 1), &sb);
    bridge_storage_init(&truth, steps, source_i, source_j, receiver_i, receiver_j,
            srcpos, signals, bl, br, bt, bb, K, Kh, a, ah, b, bh);
    bridge_storage_init(&only, steps, source_i, source_j, receiver_i, receiver_j,
            srcpos, signals, bl, br, bt, bb, K, Kh, a, ah, b, bh);
    bridge_storage_init(&gradient, steps, source_i, source_j, receiver_i, receiver_j,
            srcpos, signals, bl, br, bt, bb, K, Kh, a, ah, b, bh);
    bridge_bind_material(&material, &rho, &rhoi, &primary, &mu_x, &mu_y, &q,
            &tau_y, &tau_x, &fipjp, &f, &dip, &d, &pp, eta, bip, bjm, cip, cjm);

    prepare_model(0.0, 1, &test, &mapping, &primary, &rho, &q, &mu, &tau,
            &rhoi, &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip, &d,
            eta, bip, bjm, reference_sum);
    bind_objective_request(&truth_request, &truth, &material, hc, requests);
    if (visco_sh_exact_objective_shot(&truth_request, &truth_result) != 0) die("truth forward");
    for (n = 1; n <= NT; ++n) {
        only.observed[1][n] = truth.seismogram.sectionvz[1][n];
        gradient.observed[1][n] = truth.seismogram.sectionvz[1][n];
    }
    only.observed[1][1] += 3.25f; gradient.observed[1][1] += 3.25f;

    prepare_model(0.0, 0, &test, &mapping, &primary, &rho, &q, &mu, &tau,
            &rhoi, &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip, &d,
            eta, bip, bjm, reference_sum);
    bind_objective_request(&only_request, &only, &material, hc, requests);
    adjoint_calls = forward_calls = 0;
    if (visco_sh_exact_objective_shot(&only_request, &only_result) != 0) die("objective only");
    objective_adjoint = adjoint_calls; objective_forward = forward_calls;
    for (n = 1; n < NT; ++n) {
        double residual = only.seismogram.sectionvz[1][n + 1] - only.observed[1][n + 1];
        manual += 0.5 * residual * residual;
    }
    for (n = 1; n <= NT; ++n) {
        double residual = only.seismogram.sectionvz[1][n] - only.observed[1][n];
        including_first += 0.5 * residual * residual;
    }

    prepare_model(0.0, 0, &test, &mapping, &primary, &rho, &q, &mu, &tau,
            &rhoi, &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip, &d,
            eta, bip, bjm, reference_sum);
    bind_gradient_request(&gradient_request, &gradient, &material, hc, requests,
            &grad_primary, &grad_rho, &grad_q);
    adjoint_calls = 0;
    if (visco_sh_exact_objective_gradient_shot(&gradient_request, &gradient_result) != 0)
        die("objective gradient");
    gradient_adjoint = adjoint_calls;
    if (!run_preflight_contract(&only_request)) die("preflight transactionality");

    relative = fabs(only_result.objective - gradient_result.objective) /
            fmax(fabs(gradient_result.objective), 1.0);
    manual_relative = fabs(only_result.objective - manual) / fmax(fabs(manual), 1.0);
    included_difference = fabs(including_first - only_result.objective);
    printf("{\"free_surface\":%d,\"objective_only\":%.17g,"
            "\"objective_gradient\":%.17g,\"relative_difference\":%.17g,"
            "\"manual_relative_difference\":%.17g,\"included_first_difference\":%.17g,"
            "\"adjoint_calls_objective_only\":%d,\"adjoint_calls_gradient\":%d,"
            "\"forward_calls_objective_only\":%d}\n", free_surface,
            only_result.objective, gradient_result.objective, relative, manual_relative,
            included_difference, objective_adjoint, gradient_adjoint, objective_forward);
    if (free_surface == 1)
        printf("{\"preflight_transactional\":true,\"preflight_forward_calls\":0}\n");
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    MPI_Init(&argc, &argv);
    run_case(0); run_case(1);
    MPI_Finalize();
    return 0;
}
