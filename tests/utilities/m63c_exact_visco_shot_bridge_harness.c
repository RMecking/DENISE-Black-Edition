/* M6.3c-8b2-b1: execute the inactive exact-shot bridge around the real
 * trajectory-aware production sh_visc() forward and the locked C7 reverse. */

#ifndef M63C_LOCKED_OBJECTIVE_HARNESS
#define M63C_LOCKED_OBJECTIVE_HARNESS "m63c_objective_directional_fd_harness.c"
#endif
#include M63C_LOCKED_OBJECTIVE_HARNESS

float TSNAP1, TSNAP2, TSNAPINC, *FL, Q_APPROX_FMIN, Q_APPROX_FMAX;
float Q_APPROX_DF, FC_SPIKE_1, FC_SPIKE_2;
int Q_PARAMETERIZATION_MODE, NDT, SEISMO, IDXI, IDYI, DTINV, SNAP;
int INV_STF, EPRECOND, NTDTINV, NXNYI, NT, QUELLART, ORDER_SPIKE;
int LNORM, N_ORDER, TIMEWIN, OFFSET_MUTE, TRKILL;

float *vector(int nl, int nh) {
    float *base = (float *)calloc((size_t)(nh - nl + 1), sizeof(float));
    if (base == NULL) MPI_Abort(MPI_COMM_WORLD, 940);
    return base - nl;
}

void free_vector(float *v, int nl, int nh) {
    (void)nh;
    free(v + nl);
}

float ***f3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh) {
    int j, i, rows = nrh - nrl + 1, cols = nch - ncl + 1;
    int depth = ndh - ndl + 1;
    float ***row_base = (float ***)calloc((size_t)rows, sizeof(float **));
    float **col_base = (float **)calloc((size_t)rows * cols, sizeof(float *));
    float *data = (float *)calloc((size_t)rows * cols * depth, sizeof(float));
    float ***view;
    if (!row_base || !col_base || !data) MPI_Abort(MPI_COMM_WORLD, 941);
    view = row_base - nrl;
    for (j = nrl; j <= nrh; ++j) {
        view[j] = col_base + (j - nrl) * cols - ncl;
        for (i = ncl; i <= nch; ++i)
            view[j][i] = data + ((j - nrl) * cols + i - ncl) * depth - ndl;
    }
    return view;
}

void free_f3tensor(float ***t, int nrl, int nrh, int ncl, int nch,
                   int ndl, int ndh) {
    (void)nrh; (void)nch; (void)ndh;
    free(t[nrl][ncl] + ndl);
    free(t[nrl] + ncl);
    free(t + nrl);
}

static int **test_imatrix(int nrl, int nrh, int ncl, int nch) {
    int j, rows = nrh - nrl + 1, cols = nch - ncl + 1;
    int **base = (int **)calloc((size_t)rows, sizeof(int *));
    int *data = (int *)calloc((size_t)rows * cols, sizeof(int));
    int **view;
    if (!base || !data) MPI_Abort(MPI_COMM_WORLD, 942);
    view = base - nrl;
    for (j = nrl; j <= nrh; ++j)
        view[j] = data + (j - nrl) * cols - ncl;
    return view;
}

static void test_free_imatrix(int **m, int nrl, int ncl) {
    free(m[nrl] + ncl);
    free(m + nrl);
}

void snap(FILE *fp, int nt, int nsnap, float **vx, float **vy, float **sxx,
          float **syy, float **u, float **pi, float *hc) {
    (void)fp; (void)nt; (void)nsnap; (void)vx; (void)vy; (void)sxx;
    (void)syy; (void)u; (void)pi; (void)hc;
    MPI_Abort(MPI_COMM_WORLD, 943);
}

void eprecond_SH(float **W, float **vz) {
    (void)W; (void)vz;
    MPI_Abort(MPI_COMM_WORLD, 944);
}

static const struct directional_case bridge_cases[] = {
    {"bridge_m1_physical", 1, Q_PARAMETERIZATION_PHYSICAL,
     DIRECTION_PRIMARY | DIRECTION_RHO | DIRECTION_Q,
     1, 1, 0, 0, 48, 5, 9, 11, 9},
    {"bridge_m3_legacy", 3, Q_PARAMETERIZATION_LEGACY,
     DIRECTION_PRIMARY | DIRECTION_RHO | DIRECTION_Q,
     1, 1, 0, 0, 48, 5, 9, 11, 9},
    {"bridge_mpi_physical", 1, Q_PARAMETERIZATION_PHYSICAL,
     DIRECTION_PRIMARY | DIRECTION_RHO | DIRECTION_Q,
     2, 1, 0, 0, 64, 6, 10, 24, 10},
};

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
    double *cotangent;
    int *dtinv_help;
};

static const struct directional_case *find_bridge_case(const char *name) {
    size_t k;
    for (k = 0; k < sizeof(bridge_cases) / sizeof(bridge_cases[0]); ++k)
        if (strcmp(name, bridge_cases[k].name) == 0) return &bridge_cases[k];
    return NULL;
}

static void bridge_storage_init(
        struct bridge_storage *shot, int nsteps, int nsrc_local,
        int nrec_local, int source_i, int source_j, int receiver_i,
        int receiver_j, float **srcpos, float **signals,
        float **bl, float **br, float **bt, float **bb,
        const float *K, const float *Kh, const float *a, const float *ah,
        const float *b, const float *bh) {
    int k;
    memset(shot, 0, sizeof(*shot));
    alloc_SH(&shot->wave, &shot->pml);
    if (FW > 0) {
        for (k = 1; k <= 2 * FW; ++k) {
            shot->pml.K_x[k] = shot->pml.K_y[k] = K[k];
            shot->pml.K_x_half[k] = shot->pml.K_y_half[k] = Kh[k];
            shot->pml.a_x[k] = shot->pml.a_y[k] = a[k];
            shot->pml.a_x_half[k] = shot->pml.a_y_half[k] = ah[k];
            shot->pml.b_x[k] = shot->pml.b_y[k] = b[k];
            shot->pml.b_x_half[k] = shot->pml.b_y_half[k] = bh[k];
        }
    }
    shot->mpi.bufferlef_to_rig = bl;
    shot->mpi.bufferrig_to_lef = br;
    shot->mpi.buffertop_to_bot = bt;
    shot->mpi.bufferbot_to_top = bb;
    shot->acquisition.srcpos_loc = nsrc_local ? srcpos : NULL;
    shot->acquisition.signals = nsrc_local ? signals : NULL;
    if (nrec_local) {
        shot->recpos = test_imatrix(1, 2, 1, 1);
        shot->recpos[1][1] = receiver_i;
        shot->recpos[2][1] = receiver_j;
        shot->acquisition.recpos_loc = shot->recpos;
        shot->seismogram.sectionvz = matrix(1, 1, 1, nsteps);
        shot->observed = matrix(1, 1, 1, nsteps);
        shot->cotangent = (double *)calloc((size_t)nsteps, sizeof(double));
        if (!shot->cotangent) MPI_Abort(MPI_COMM_WORLD, 945);
    }
    shot->dtinv_help = (int *)calloc((size_t)nsteps + 1, sizeof(int));
    if (!shot->dtinv_help) MPI_Abort(MPI_COMM_WORLD, 946);
}

static void bridge_storage_release(
        struct bridge_storage *shot, int nsteps, int nrec_local) {
    if (nrec_local) {
        free(shot->cotangent);
        free_matrix(shot->observed, 1, 1, 1, nsteps);
        free_matrix(shot->seismogram.sectionvz, 1, 1, 1, nsteps);
        test_free_imatrix(shot->recpos, 1, 1);
    }
    free(shot->dtinv_help);
    dealloc_SH(&shot->wave, &shot->pml);
}

static void bridge_bind_material(
        struct matSH *material, struct field *rho, struct field *rhoi,
        struct field *primary, struct field *mu_x, struct field *mu_y,
        struct field *q, struct field *tau_y, struct field *tau_x,
        struct field *fipjp, struct field *f, struct volume *dip,
        struct volume *d, struct volume *pp, float *eta, float *bip,
        float *bjm, float *cip, float *cjm) {
    memset(material, 0, sizeof(*material));
    material->prho = rho->v; material->prhoi = rhoi->v;
    material->pu = primary->v; material->puip = mu_x->v;
    material->pujp = mu_y->v; material->pqs = q->v;
    material->ptaus = tau_y->v; material->ptausipjp = tau_x->v;
    material->fipjp = fipjp->v; material->f = f->v;
    material->dip = dip->v; material->d = d->v; material->e = pp->v;
    material->etaip = eta; material->etajm = eta;
    material->bip = bip; material->bjm = bjm;
    material->cip = cip; material->cjm = cjm;
}

static int bridge_run(
        struct bridge_storage *shot, struct matSH *material,
        int nsteps, int nsrc_local, int nrec_local, float *hc,
        float **grad_primary, float **grad_rho, float **grad_q,
        MPI_Request *request, double *objective) {
    struct visco_sh_exact_shot_request bridge;
    struct visco_sh_exact_shot_result result;
    memset(&bridge, 0, sizeof(bridge));
    memset(&result, 0, sizeof(result));
    bridge.wave = &shot->wave; bridge.pml = &shot->pml;
    bridge.material = material; bridge.fwi = &shot->fwi;
    bridge.mpi = &shot->mpi; bridge.seismogram = &shot->seismogram;
    bridge.legacy_fwi_seismogram = &shot->legacy;
    bridge.acquisition = &shot->acquisition; bridge.hc = hc;
    bridge.ishot = 1; bridge.nshots = 1; bridge.nsrc_local = nsrc_local;
    bridge.ns = nsteps; bridge.nrec_local = nrec_local; bridge.hin = 1;
    bridge.dtinv_help = shot->dtinv_help; bridge.observed_vz = shot->observed;
    bridge.receiver_cotangent = shot->cotangent;
    bridge.grad_primary = grad_primary; bridge.grad_rho = grad_rho;
    bridge.grad_q = grad_q; bridge.request_send = request;
    bridge.request_receive = request;
    if (visco_sh_exact_objective_gradient_shot(&bridge, &result) != 0)
        return -1;
    *objective = result.objective;
    return 0;
}

static double bridge_directional_contraction(
        const struct directional_case *test, struct field *grad_primary,
        struct field *grad_rho, struct field *grad_q) {
    double local = 0.0, global = 0.0;
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i) {
            int gi = POS[1] * NX + i, gj = POS[2] * NY + j;
            local += grad_primary->v[j][i] * primary_direction(test, gj, gi);
            local += grad_rho->v[j][i] * rho_direction(gj, gi);
            local += grad_q->v[j][i] * q_direction(gj, gi);
        }
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

int main(int argc, char **argv) {
    const double epsilons[2] = {1.0e-2, 3.0e-3};
    const struct directional_case *test;
    struct field rhoi, fipjp, f, primary, rho, q, mu, tau, mu_x, mu_y;
    struct field tau_x, tau_y, grad_primary, grad_rho, grad_q, diag[6], dummy;
    struct volume dip, d, pp;
    struct q_tau_mapping mapping;
    struct matSH material;
    struct bridge_storage shot;
    float *hc, *bip, *bjm, *cip, *cjm, *eta, *frequencies;
    float *K, *Kh, *a, *ah, *b, *bh;
    float **srcpos, **signals, *src_storage, *signal_storage;
    float **bl, **br, **bt, **bb, *sl, *sr, *st, *sb;
    MPI_Request request[4];
    void *bsend;
    int bsize = 1 << 20;
    int rank, size, h, i, j, k, l, n, e, nsteps;
    int source_i, source_j, receiver_i, receiver_j;
    int source_owner, receiver_owner, nsrc_local, nrec_local;
    double reference_sum = 0.0, objective_truth, objective_base;
    double objective_repeat, d_ad, d_repeat, max_cotangent_error = 0.0;
    double objective_reference_local = 0.0, objective_reference = 0.0;
    double first_cotangent = 0.0, remote_trace_local = 0.0, remote_trace = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 2 || !(test = find_bridge_case(argv[1])))
        MPI_Abort(MPI_COMM_WORLD, 947);
    MYID = rank; FP = stderr; NPROCX = test->nproc_x; NPROCY = test->nproc_y;
    if (size != NPROCX * NPROCY) MPI_Abort(MPI_COMM_WORLD, 948);
    NX = 18; NY = 20; FDORDER = 4; L = 2; FW = test->fw;
    FREE_SURF = test->free_surface; BOUNDARY = 0; INVMAT1 = test->invmat1;
    DT = 0.0013f; DH = 7.5f; GRAD_FORM = 2; ADJ_SIGN = 1; MODE = 0;
    QUELLTYP = 1; QUELLTYPB = 1; QUELLART = 1; topology(rank);
    NT = test->nsteps; nsteps = NT; DTINV = 1; NDT = 1; SEISMO = 1;
    SNAP = 0; INV_STF = 0; EPRECOND = 0; LNORM = 2; N_ORDER = 0;
    TIMEWIN = 0; OFFSET_MUTE = 0; TRKILL = 0; IDXI = IDYI = 1;
    NTDTINV = NT; NXNYI = NX * NY; TSNAP1 = TSNAP2 = TSNAPINC = 1.0f;
    source_i = test->source_i; source_j = test->source_j;
    receiver_i = test->receiver_i; receiver_j = test->receiver_j;
    source_owner = ((source_j - 1) / NY) * NPROCX + (source_i - 1) / NX;
    receiver_owner = ((receiver_j - 1) / NY) * NPROCX + (receiver_i - 1) / NX;
    nsrc_local = rank == source_owner; nrec_local = rank == receiver_owner;
    source_i -= POS[1] * NX; source_j -= POS[2] * NY;
    receiver_i -= POS[1] * NX; receiver_j -= POS[2] * NY;
    h = FDORDER / 2; bsend = malloc((size_t)bsize);
    MPI_Buffer_attach(bsend, bsize);

#define FIELD(name) name = field_new(-h, NY + h + 1, 1 - h, NX + h)
    FIELD(rhoi); FIELD(fipjp); FIELD(f); FIELD(primary); FIELD(rho); FIELD(q);
    FIELD(mu); FIELD(tau); FIELD(mu_x); FIELD(mu_y); FIELD(tau_x); FIELD(tau_y);
    FIELD(grad_primary); FIELD(grad_rho); FIELD(grad_q); FIELD(dummy);
    for (k = 0; k < 6; ++k) FIELD(diag[k]);
#undef FIELD
    dip = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    d = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    pp = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    hc = (float *)calloc(3, sizeof(float)); hc[1] = 1.0f; hc[2] = -0.041f;
    bip = (float *)calloc(3, sizeof(float)); bjm = (float *)calloc(3, sizeof(float));
    cip = (float *)calloc(3, sizeof(float)); cjm = (float *)calloc(3, sizeof(float));
    eta = (float *)calloc(3, sizeof(float)); frequencies = (float *)calloc(3, sizeof(float));
    frequencies[1] = 4.0f; frequencies[2] = 9.0f; FL = frequencies;
    Q_PARAMETERIZATION_MODE = test->qmode; Q_APPROX_FMIN = 2.0f;
    Q_APPROX_FMAX = 18.0f; Q_APPROX_DF = 0.5f;
    init_q_tau_mapping(&mapping, Q_PARAMETERIZATION_MODE, L, FL,
                       Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF);
    for (l = 1; l <= L; ++l) {
        double theta = 1.0 / (2.0 * PI * frequencies[l]);
        double x = 2.0 * PI * frequencies[1] * theta;
        eta[l] = (float)(DT / theta); reference_sum += x * x / (1.0 + x * x);
        bip[l] = bjm[l] = 1.0f / (1.0f + 0.5f * eta[l]);
        cip[l] = cjm[l] = 1.0f - 0.5f * eta[l];
    }
    K = (float *)calloc((size_t)(2 * FW + 1), sizeof(float));
    Kh = (float *)calloc((size_t)(2 * FW + 1), sizeof(float));
    a = (float *)calloc((size_t)(2 * FW + 1), sizeof(float));
    ah = (float *)calloc((size_t)(2 * FW + 1), sizeof(float));
    b = (float *)calloc((size_t)(2 * FW + 1), sizeof(float));
    bh = (float *)calloc((size_t)(2 * FW + 1), sizeof(float));
    initialize_cpml_coefficients(FW, K, Kh, a, ah, b, bh);
    srcpos = buffer_new(8, 1, &src_storage);
    signals = buffer_new(1, nsteps, &signal_storage);
    srcpos[1][1] = (float)source_i; srcpos[2][1] = (float)source_j;
    srcpos[8][1] = 1.0f;
    for (n = 0; n < nsteps; ++n) {
        double x = (n - 9.0) / 3.1;
        signals[1][n + 1] = (float)(0.014 * (1.0 - 2.0 * x * x) * exp(-x * x));
    }
    bl = buffer_new(NY, 2 * (h + 1), &sl); br = buffer_new(NY, 2 * (h + 1), &sr);
    bt = buffer_new(NX, 2 * (h + 1), &st); bb = buffer_new(NX, 2 * (h + 1), &sb);
    bridge_storage_init(&shot, nsteps, nsrc_local, nrec_local, source_i,
                        source_j, receiver_i, receiver_j, srcpos, signals,
                        bl, br, bt, bb, K, Kh, a, ah, b, bh);
    bridge_bind_material(&material, &rho, &rhoi, &primary, &mu_x, &mu_y, &q,
                         &tau_y, &tau_x, &fipjp, &f, &dip, &d, &pp,
                         eta, bip, bjm, cip, cjm);

    prepare_model(0.0, 1, test, &mapping, &primary, &rho, &q, &mu, &tau,
                  &rhoi, &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip,
                  &d, eta, bip, bjm, reference_sum);
    if (bridge_run(&shot, &material, nsteps, nsrc_local, nrec_local, hc,
                   grad_primary.v, grad_rho.v, grad_q.v, request,
                   &objective_truth) != 0) MPI_Abort(MPI_COMM_WORLD, 949);
    if (nrec_local)
        for (n = 1; n <= nsteps; ++n)
            shot.observed[1][n] = shot.seismogram.sectionvz[1][n];

    prepare_model(0.0, 0, test, &mapping, &primary, &rho, &q, &mu, &tau,
                  &rhoi, &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip,
                  &d, eta, bip, bjm, reference_sum);
    if (bridge_run(&shot, &material, nsteps, nsrc_local, nrec_local, hc,
                   grad_primary.v, grad_rho.v, grad_q.v, request,
                   &objective_base) != 0) MPI_Abort(MPI_COMM_WORLD, 950);
    d_ad = bridge_directional_contraction(test, &grad_primary, &grad_rho, &grad_q);
    if (nrec_local) {
        first_cotangent = shot.cotangent[0];
        for (n = 0; n < nsteps; ++n) {
            double residual = n == 0 ? 0.0 :
                    shot.seismogram.sectionvz[1][n + 1] - shot.observed[1][n + 1];
            max_cotangent_error = fmax(max_cotangent_error,
                                       fabs(shot.cotangent[n] - residual));
            objective_reference_local += 0.5 * residual * residual;
            if (receiver_owner != source_owner)
                remote_trace_local = fmax(remote_trace_local,
                        fabs(shot.seismogram.sectionvz[1][n + 1]));
        }
    }
    MPI_Allreduce(&objective_reference_local, &objective_reference, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_cotangent_error, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &first_cotangent, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&remote_trace_local, &remote_trace, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    prepare_model(0.0, 0, test, &mapping, &primary, &rho, &q, &mu, &tau,
                  &rhoi, &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip,
                  &d, eta, bip, bjm, reference_sum);
    if (bridge_run(&shot, &material, nsteps, nsrc_local, nrec_local, hc,
                   grad_primary.v, grad_rho.v, grad_q.v, request,
                   &objective_repeat) != 0) MPI_Abort(MPI_COMM_WORLD, 951);
    d_repeat = bridge_directional_contraction(test, &grad_primary, &grad_rho, &grad_q);
    if (rank == 0)
        printf("{\"case\":\"%s\",\"invmat1\":%d,\"q_mode\":\"%s\","
               "\"truth_generation_objective\":%.17g,\"objective\":%.17g,"
               "\"objective_reference\":%.17g,"
               "\"D_ad\":%.17g,\"repeat_objective\":%.17g,\"repeat_D_ad\":%.17g,"
               "\"max_cotangent_error\":%.17g,\"first_cotangent\":%.17g,"
               "\"nproc_x\":%d,\"nproc_y\":%d,\"source_owner\":%d,"
               "\"receiver_owner\":%d,\"remote_trace\":%.17g}\n",
               test->name, test->invmat1,
               test->qmode == Q_PARAMETERIZATION_LEGACY ? "legacy" : "physical",
               objective_truth, objective_base, objective_reference, d_ad,
               objective_repeat, d_repeat, max_cotangent_error,
               first_cotangent, NPROCX, NPROCY, source_owner,
               receiver_owner, remote_trace);

    for (e = 0; e < 2; ++e) {
        double ep = epsilons[e], jp, jm, dfd, rel;
        prepare_model(ep, 0, test, &mapping, &primary, &rho, &q, &mu, &tau,
                      &rhoi, &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f,
                      &dip, &d, eta, bip, bjm, reference_sum);
        if (bridge_run(&shot, &material, nsteps, nsrc_local, nrec_local, hc,
                       grad_primary.v, grad_rho.v, grad_q.v, request, &jp) != 0)
            MPI_Abort(MPI_COMM_WORLD, 952);
        prepare_model(-ep, 0, test, &mapping, &primary, &rho, &q, &mu, &tau,
                      &rhoi, &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f,
                      &dip, &d, eta, bip, bjm, reference_sum);
        if (bridge_run(&shot, &material, nsteps, nsrc_local, nrec_local, hc,
                       grad_primary.v, grad_rho.v, grad_q.v, request, &jm) != 0)
            MPI_Abort(MPI_COMM_WORLD, 953);
        dfd = (jp - jm) / (2.0 * ep);
        rel = fabs(dfd - d_ad) / fmax(fmax(fabs(dfd), fabs(d_ad)), 1.0e-300);
        if (rank == 0)
            printf("{\"case\":\"%s\",\"epsilon\":%.17g,\"J_plus\":%.17g,"
                   "\"J_minus\":%.17g,\"D_fd\":%.17g,\"D_ad\":%.17g,"
                   "\"relative_error\":%.17g}\n",
                   test->name, ep, jp, jm, dfd, d_ad, rel);
    }
    bridge_storage_release(&shot, nsteps, nrec_local);
    MPI_Buffer_detach(&bsend, &bsize); free(bsend); MPI_Finalize();
    return 0;
}
