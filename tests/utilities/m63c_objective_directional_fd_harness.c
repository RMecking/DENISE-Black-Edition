/* M6.3c-7d-a: real forward objective versus the locked C7c-b gradient.
 *
 * Frozen objective contract (LNORM=2, GRAD_FORM=2, SH vz data):
 *   r[n] = synthetic[n] - observed[n]
 *   J = 0.5 * sum_n r[n]^2
 *   dJ/dsynthetic[n] = r[n]
 *   bar_receiver[n] = r[n] in chronological order for C5/C7c-b.
 * Production calc_res stores the same residual in reverse-time array order;
 * that storage convention is not an extra sign or time shift.  The objective
 * and receiver cotangent contain no DT or DTINV.  The locked C7c-b material
 * assembly directly sums the discrete per-step VJPs at DTINV == 1.
 */

#define main m63c_locked_c5a_harness_main
#include "m63c_full_state_step_harness.c"
#undef main

int INVMAT1;

float **matrix(int nrl, int nrh, int ncl, int nch) {
    int row, rows = nrh - nrl + 1, cols = nch - ncl + 1;
    float **base = (float **)malloc((size_t)rows * sizeof(float *));
    float *data = (float *)calloc((size_t)rows * cols, sizeof(float));
    float **view;
    if (base == NULL || data == NULL) MPI_Abort(MPI_COMM_WORLD, 900);
    view = base - nrl;
    for (row = nrl; row <= nrh; ++row)
        view[row] = data + (row - nrl) * cols - ncl;
    return view;
}

void free_matrix(float **m, int nrl, int nrh, int ncl, int nch) {
    (void)nrh; (void)nch;
    free(m[nrl] + ncl); free(m + nrl);
}

void err(char text[]) {
    fprintf(stderr, "%s\n", text); MPI_Abort(MPI_COMM_WORLD, 900);
}

static void forward_step(
        struct owned_state *state, int nt, float **srcpos, float **signals,
        struct field diag[6], struct field *dummy, struct field *rhoi,
        struct field *fipjp, struct field *f, struct volume *dip,
        struct volume *d, struct volume *pp, float *hc, float *bip,
        float *bjm, float *cip, float *cjm, float *K, float *Kh,
        float *a, float *ah, float *b, float *bh, float **bl, float **br,
        float **bt, float **bb, MPI_Request *request) {
    update_v_PML_SH(1, NX, 1, NY, nt, state->view.vz, diag[0].v,
                    diag[1].v, diag[2].v, state->view.sxz, state->view.syz,
                    dummy->v, rhoi->v, srcpos, signals, 1, dummy->v, hc,
                    0, 0, 0, K, a, b, Kh, ah, bh, K, a, b, Kh, ah, bh,
                    state->view.psi_sxz_x, state->view.psi_syz_y);
    exchange_v_SH(state->view.vz, bl, br, bt, bb, request, request);
    if (FREE_SURF && POS[2] == 0)
        surface_elastic_SH_velocity(state->view.vz, NX, FDORDER / 2);
    update_s_visc_PML_SH(1, NX, 1, NY, state->view.vz, diag[3].v,
                         diag[4].v, state->view.syz, state->view.sxz,
                         dummy->v, dummy->v, dummy->v, hc, 0, state->view.r,
                         pp->v, state->view.q, fipjp->v, f->v, dummy->v,
                         bip, bjm, cip, cjm, d->v, pp->v, dip->v,
                         K, a, b, Kh, ah, bh, K, a, b, Kh, ah, bh,
                         state->view.psi_vzx, state->view.psi_vzy, NULL, 0);
    if (FREE_SURF && POS[2] == 0)
        surface_elastic_SH_stress(state->view.syz, NX, FDORDER / 2);
    exchange_s_SH(state->view.sxz, state->view.syz, bl, br, bt, bb,
                  request, request);
}

enum direction_kind {
    DIRECTION_PRIMARY = 1,
    DIRECTION_RHO = 2,
    DIRECTION_Q = 4
};

struct directional_case {
    const char *name;
    int invmat1;
    int qmode;
    int directions;
};

static const struct directional_case cases[] = {
    {"m3_mu", 3, Q_PARAMETERIZATION_PHYSICAL, DIRECTION_PRIMARY},
    {"m3_rho", 3, Q_PARAMETERIZATION_PHYSICAL, DIRECTION_RHO},
    {"m3_q_legacy", 3, Q_PARAMETERIZATION_LEGACY, DIRECTION_Q},
    {"m3_q_physical", 3, Q_PARAMETERIZATION_PHYSICAL, DIRECTION_Q},
    {"m3_combined_legacy", 3, Q_PARAMETERIZATION_LEGACY,
     DIRECTION_PRIMARY | DIRECTION_RHO | DIRECTION_Q},
    {"m3_combined_physical", 3, Q_PARAMETERIZATION_PHYSICAL,
     DIRECTION_PRIMARY | DIRECTION_RHO | DIRECTION_Q},
    {"m1_vs", 1, Q_PARAMETERIZATION_PHYSICAL, DIRECTION_PRIMARY},
    {"m1_rho", 1, Q_PARAMETERIZATION_PHYSICAL, DIRECTION_RHO},
    {"m1_q_legacy", 1, Q_PARAMETERIZATION_LEGACY, DIRECTION_Q},
    {"m1_q_physical", 1, Q_PARAMETERIZATION_PHYSICAL, DIRECTION_Q},
    {"m1_combined_legacy", 1, Q_PARAMETERIZATION_LEGACY,
     DIRECTION_PRIMARY | DIRECTION_RHO | DIRECTION_Q},
    {"m1_combined_physical", 1, Q_PARAMETERIZATION_PHYSICAL,
     DIRECTION_PRIMARY | DIRECTION_RHO | DIRECTION_Q}
};

static const struct directional_case *find_case(const char *name) {
    size_t index;
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index)
        if (strcmp(name, cases[index].name) == 0) return &cases[index];
    return NULL;
}

static double base_mu_value(int j, int i) {
    return 4.55e9 * (1.0 + 0.035 * sin(0.31 * i + 0.17 * j)
                    + 0.018 * cos(0.13 * i - 0.29 * j));
}

static double base_vs_value(int j, int i) {
    return 1460.0 * (1.0 + 0.021 * sin(0.29 * i + 0.13 * j)
                    - 0.014 * cos(0.17 * i - 0.31 * j));
}

static double base_rho_value(int j, int i) {
    return 2130.0 + 1.8 * i + 1.1 * j
           + 4.0 * sin(0.21 * i + 0.09 * j);
}

static double base_q_value(int j, int i) {
    return 31.0 + 0.19 * i + 0.14 * j
           + 0.23 * sin(0.16 * i - 0.11 * j);
}

static double primary_direction(const struct directional_case *test,
                                int j, int i) {
    double pattern = 0.11 + 0.23 * sin(0.41 * i + 0.19 * j)
                     - 0.14 * cos(0.27 * i - 0.33 * j);
    double base = test->invmat1 == 1 ? base_vs_value(j, i)
                                     : base_mu_value(j, i);
    return base * pattern;
}

static double rho_direction(int j, int i) {
    double pattern = -0.08 + 0.17 * cos(0.37 * i + 0.23 * j)
                     + 0.12 * sin(0.19 * i - 0.41 * j);
    return base_rho_value(j, i) * pattern;
}

static double q_direction(int j, int i) {
    double pattern = 0.16 - 0.21 * sin(0.33 * i + 0.28 * j)
                     + 0.09 * cos(0.47 * i - 0.15 * j);
    return base_q_value(j, i) * pattern;
}

static void prepare_model(
        double epsilon, int truth, const struct directional_case *test,
        const struct q_tau_mapping *mapping, struct field *primary,
        struct field *rho, struct field *q, struct field *mu,
        struct field *tau, struct field *rhoi, struct field *mu_x,
        struct field *mu_y, struct field *tau_x, struct field *tau_y,
        struct field *fipjp, struct field *f, struct volume *dip,
        struct volume *d, const float *eta, const float *bip,
        const float *bjm, double reference_sum) {
    int i, j, l;
    for (j = 1; j <= NY; ++j) {
        for (i = 1; i <= NX; ++i) {
            double primary_base = test->invmat1 == 1 ? base_vs_value(j, i)
                                                     : base_mu_value(j, i);
            double rho_base = base_rho_value(j, i);
            double q_base = base_q_value(j, i);
            double truth_primary = primary_base *
                    (0.026 * cos(0.22 * i + 0.37 * j)
                     - 0.012 * sin(0.35 * i - 0.16 * j));
            double truth_rho = rho_base *
                    (-0.014 * sin(0.18 * i + 0.32 * j)
                     + 0.009 * cos(0.39 * i - 0.21 * j));
            double truth_q = q_base *
                    (0.031 * cos(0.26 * i + 0.14 * j)
                     + 0.017 * sin(0.43 * i - 0.24 * j));
            double primary_delta = (test->directions & DIRECTION_PRIMARY)
                    ? epsilon * primary_direction(test, j, i) : 0.0;
            double rho_delta = (test->directions & DIRECTION_RHO)
                    ? epsilon * rho_direction(j, i) : 0.0;
            double q_delta = (test->directions & DIRECTION_Q)
                    ? epsilon * q_direction(j, i) : 0.0;
            primary->v[j][i] = (float)(primary_base +
                    (truth ? truth_primary : primary_delta));
            rho->v[j][i] = (float)(rho_base +
                    (truth ? truth_rho : rho_delta));
            q->v[j][i] = (float)(q_base +
                    (truth ? truth_q : q_delta));
            mu->v[j][i] = test->invmat1 == 1
                    ? rho->v[j][i] * primary->v[j][i] * primary->v[j][i]
                    : primary->v[j][i];
            rhoi->v[j][i] = 1.0f / rho->v[j][i];
            tau->v[j][i] = q_to_tau(q->v[j][i], mapping);
        }
    }
    /* Reproduce the production material graph in native parameter space:
     * INVMAT1=1 exchanges Vs, while INVMAT1=3 exchanges mu. */
    matcopy_SH(rho->v, primary->v, tau->v);
    av_mu_SH(primary->v, mu_x->v, mu_y->v, rho->v);
    av_tau(tau->v, tau_x->v);
    for (j = 1; j <= NY; ++j) {
        for (i = 1; i <= NX; ++i) {
            double relaxed_x, relaxed_y;
            tau_y->v[j][i] = tau->v[j][i];
            relaxed_x = mu_x->v[j][i] /
                    (1.0 + reference_sum * tau_x->v[j][i]);
            relaxed_y = mu_y->v[j][i] /
                    (1.0 + reference_sum * tau_y->v[j][i]);
            fipjp->v[j][i] = (float)(DT * relaxed_x *
                    (1.0 + L * tau_x->v[j][i]));
            f->v[j][i] = (float)(DT * relaxed_y *
                    (1.0 + L * tau_y->v[j][i]));
            for (l = 1; l <= L; ++l) {
                dip->v[j][i][l] = (float)(relaxed_x * eta[l] *
                                                tau_x->v[j][i]);
                d->v[j][i][l] = (float)(relaxed_y * eta[l] *
                                               tau_y->v[j][i]);
            }
        }
    }
    (void)bip;
    (void)bjm;
}

static int forward_trace(
        int nsteps, struct visco_sh_material_observable_trajectory *trajectory,
        struct field *rhoi, struct field *fipjp, struct field *f,
        struct volume *dip, struct volume *d, struct volume *pp,
        struct field diag[6], struct field *dummy, float **srcpos,
        float **signals, float *hc, float *bip, float *bjm, float *cip,
        float *cjm, float *K, float *Kh, float *a, float *ah, float *b,
        float *bh, float **bl, float **br, float **bt, float **bb,
        MPI_Request *request, int receiver_i, int receiver_j, double *trace) {
    struct owned_state state = state_new(FDORDER / 2, FW, L);
    int n;
    memset(pp->data, 0, (size_t)pp->nrows * pp->ncols *
                         (pp->mechanisms + 1) * sizeof(float));
    for (n = 0; n < nsteps; ++n) {
        if (trajectory != NULL &&
                visco_sh_material_observable_begin_step(trajectory, n) != 0)
            return -1;
        forward_step(&state, n + 1, srcpos, signals, diag, dummy, rhoi,
                     fipjp, f, dip, d, pp, hc, bip, bjm, cip, cjm,
                     K, Kh, a, ah, b, bh, bl, br, bt, bb, request);
        if (trajectory != NULL) visco_sh_material_observable_end_step();
        trace[n] = state.view.vz[receiver_j][receiver_i];
    }
    state_free(&state, FW);
    return 0;
}

static double objective(const double *synthetic, const double *observed,
                        int nsteps) {
    double value = 0.0;
    int n;
    for (n = 0; n < nsteps; ++n) {
        double residual = (n == 0) ? 0.0 : synthetic[n] - observed[n];
        value += 0.5 * residual * residual;
    }
    return value;
}

int main(int argc, char **argv) {
    const double epsilons[4] = {1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4};
    const struct directional_case *test;
    struct field rhoi, fipjp, f, primary, rho, q, mu, tau, mu_x, mu_y;
    struct field tau_x, tau_y, grad_primary, grad_rho, grad_q, diag[6], dummy;
    struct volume dip, d, pp;
    struct visco_sh_material_observable_trajectory trajectory;
    struct visco_sh_reverse_time_material_context material;
    struct visco_sh_full_step_config cfg;
    struct owned_state terminal, initial, scratch;
    struct q_tau_mapping mapping;
    float *hc, *bip, *bjm, *cip, *cjm, *eta, *frequencies;
    float *K, *Kh, *a, *ah, *b, *bh;
    float **srcpos, **signals, *src_storage, *signal_storage;
    float **bl, **br, **bt, **bb, *sl, *sr, *st, *sb;
    double *observed, *synthetic, *plus, *minus, *bar_receiver, *bar_signal;
    MPI_Request request[4];
    void *bsend;
    int bsize = 1 << 20;
    int rank, size, h, i, j, k, l, n, e, status;
    int nsteps = 48, receiver_i = 11, receiver_j = 9;
    int rec_x[1], rec_y[1], src_x[1], src_y[1], src_type[1];
    double reference_sum = 0.0, d_ad = 0.0, direction_norm = 0.0;
    double d_primary = 0.0, d_rho = 0.0, d_q = 0.0;
    double j_base, max_trace = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 1 || argc != 2) MPI_Abort(MPI_COMM_WORLD, 901);
    test = find_case(argv[1]);
    if (test == NULL) MPI_Abort(MPI_COMM_WORLD, 901);
    MYID = rank; FP = stderr; NPROCX = NPROCY = 1; NX = 18; NY = 20;
    FDORDER = 4; L = 2; FW = 0; FREE_SURF = 0; BOUNDARY = 0;
    INVMAT1 = test->invmat1; DT = 0.0013f; DH = 7.5f; GRAD_FORM = 2;
    ADJ_SIGN = 1; MODE = 0; QUELLTYPB = 1; topology(rank); h = FDORDER / 2;
    bsend = malloc((size_t)bsize); MPI_Buffer_attach(bsend, bsize);

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
    frequencies[1] = 4.0f; frequencies[2] = 9.0f;
    init_q_tau_mapping(&mapping, test->qmode, L,
                       frequencies, 2.0f, 18.0f, 0.5f);
    for (l = 1; l <= L; ++l) {
        double theta = 1.0 / (2.0 * PI * frequencies[l]);
        double x = 2.0 * PI * 6.0 * theta;
        eta[l] = (float)(DT / theta);
        reference_sum += x * x / (1.0 + x * x);
        bip[l] = bjm[l] = 1.0f / (1.0f + 0.5f * eta[l]);
        cip[l] = cjm[l] = 1.0f - 0.5f * eta[l];
    }
    K = (float *)calloc(1, sizeof(float)); Kh = (float *)calloc(1, sizeof(float));
    a = (float *)calloc(1, sizeof(float)); ah = (float *)calloc(1, sizeof(float));
    b = (float *)calloc(1, sizeof(float)); bh = (float *)calloc(1, sizeof(float));
    srcpos = buffer_new(8, 1, &src_storage);
    signals = buffer_new(1, nsteps, &signal_storage);
    srcpos[1][1] = 5.0f; srcpos[2][1] = 9.0f; srcpos[8][1] = 1.0f;
    for (n = 0; n < nsteps; ++n) {
        double x = (n - 9.0) / 3.1;
        signals[1][n + 1] = (float)(0.014 * (1.0 - 2.0 * x * x) * exp(-x * x));
    }
    bl = buffer_new(NY, 2 * (h + 1), &sl); br = buffer_new(NY, 2 * (h + 1), &sr);
    bt = buffer_new(NX, 2 * (h + 1), &st); bb = buffer_new(NX, 2 * (h + 1), &sb);
    observed = (double *)calloc((size_t)nsteps, sizeof(double));
    synthetic = (double *)calloc((size_t)nsteps, sizeof(double));
    plus = (double *)calloc((size_t)nsteps, sizeof(double));
    minus = (double *)calloc((size_t)nsteps, sizeof(double));
    bar_receiver = (double *)calloc((size_t)nsteps, sizeof(double));
    bar_signal = (double *)calloc((size_t)nsteps, sizeof(double));

    prepare_model(0.0, 1, test, &mapping, &primary, &rho, &q, &mu, &tau, &rhoi,
                  &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip, &d,
                  eta, bip, bjm, reference_sum);
    if (forward_trace(nsteps, NULL, &rhoi, &fipjp, &f, &dip, &d, &pp,
                      diag, &dummy, srcpos, signals, hc, bip, bjm, cip, cjm,
                      K, Kh, a, ah, b, bh, bl, br, bt, bb, request,
                      receiver_i, receiver_j, observed) != 0)
        MPI_Abort(MPI_COMM_WORLD, 902);

    prepare_model(0.0, 0, test, &mapping, &primary, &rho, &q, &mu, &tau, &rhoi,
                  &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip, &d,
                  eta, bip, bjm, reference_sum);
    memset(&trajectory, 0, sizeof(trajectory));
    if (visco_sh_material_observable_trajectory_init(
                &trajectory, NX, NY, nsteps, 1, FW, FREE_SURF, BOUNDARY,
                NPROCX, NPROCY) != 0) MPI_Abort(MPI_COMM_WORLD, 903);
    if (forward_trace(nsteps, &trajectory, &rhoi, &fipjp, &f, &dip, &d,
                      &pp, diag, &dummy, srcpos, signals, hc, bip, bjm,
                      cip, cjm, K, Kh, a, ah, b, bh, bl, br, bt, bb,
                      request, receiver_i, receiver_j, synthetic) != 0)
        MPI_Abort(MPI_COMM_WORLD, 904);
    for (n = 0; n < nsteps; ++n) {
        bar_receiver[n] = (n == 0) ? 0.0 : synthetic[n] - observed[n];
        max_trace = fmax(max_trace, fabs(synthetic[n]));
    }
    j_base = objective(synthetic, observed, nsteps);

    terminal = state_new(h, FW, L); initial = state_new(h, FW, L);
    scratch = state_new(h, FW, L);
    rec_x[0] = receiver_i; rec_y[0] = receiver_j;
    src_x[0] = (int)srcpos[1][1]; src_y[0] = (int)srcpos[2][1]; src_type[0] = 1;
    memset(&cfg, 0, sizeof(cfg));
    cfg.nx = NX; cfg.ny = NY; cfg.fdorder = FDORDER; cfg.mechanisms = L;
    cfg.fw = FW; cfg.free_surface = FREE_SURF; cfg.boundary = BOUNDARY;
    memcpy(cfg.pos, POS, sizeof(POS)); memcpy(cfg.index, INDEX, sizeof(INDEX));
    cfg.nproc_x = NPROCX; cfg.nproc_y = NPROCY; cfg.dt = DT; cfg.dh = DH;
    cfg.hc = hc; cfg.rhoi = rhoi.v; cfg.fipjp = fipjp.v; cfg.f = f.v;
    cfg.bip = bip; cfg.bjm = bjm; cfg.cip = cip; cfg.cjm = cjm;
    cfg.dip = dip.v; cfg.d = d.v; cfg.K_x = K; cfg.a_x = a; cfg.b_x = b;
    cfg.K_x_half = Kh; cfg.a_x_half = ah; cfg.b_x_half = bh;
    cfg.K_y = K; cfg.a_y = a; cfg.b_y = b; cfg.K_y_half = Kh;
    cfg.a_y_half = ah; cfg.b_y_half = bh; cfg.nrec = 1;
    cfg.rec_x = rec_x; cfg.rec_y = rec_y; cfg.nsrc = 1;
    cfg.src_x = src_x; cfg.src_y = src_y; cfg.source_type = src_type;
    cfg.comm = MPI_COMM_WORLD;
    memset(&material, 0, sizeof(material)); material.trajectory = &trajectory;
    material.mu_x = mu_x.v; material.tau_x = tau_x.v;
    material.mu_y = mu_y.v; material.tau_y = tau_y.v;
    material.reference_sum = reference_sum; material.eta_x = eta;
    material.eta_y = eta; material.invmat1 = test->invmat1;
    material.mapping = &mapping;
    material.primary_post = primary.v; material.rho_post = rho.v;
    material.owned_q = q.v; material.grad_primary = grad_primary.v;
    material.grad_rho = grad_rho.v; material.grad_q = grad_q.v;
    status = visco_sh_reverse_time_adjoint_material(
            &cfg, nsteps, bar_receiver, &terminal.view, &initial.view,
            &scratch.view, bar_signal, &material);
    if (status != 0) {
        fprintf(stderr, "C7d-b1 material adjoint status=%d case=%s\n",
                status, test->name);
        MPI_Abort(MPI_COMM_WORLD, 905);
    }
    for (j = 1; j <= NY; ++j) {
        for (i = 1; i <= NX; ++i) {
            if (test->directions & DIRECTION_PRIMARY) {
                double direction = primary_direction(test, j, i);
                d_primary += grad_primary.v[j][i] * direction;
                direction_norm += direction * direction;
            }
            if (test->directions & DIRECTION_RHO) {
                double direction = rho_direction(j, i);
                d_rho += grad_rho.v[j][i] * direction;
                direction_norm += direction * direction;
            }
            if (test->directions & DIRECTION_Q) {
                double direction = q_direction(j, i);
                d_q += grad_q.v[j][i] * direction;
                direction_norm += direction * direction;
            }
        }
    }
    d_ad = d_primary + d_rho + d_q;
    direction_norm = sqrt(direction_norm);

    printf("{\"case\":\"%s\",\"invmat1\":%d,\"q_mode\":\"%s\","
           "\"directions\":%d,\"contract\":{\"lnorm\":2,"
           "\"grad_form\":2,\"quelltypb\":1,"
           "\"residual\":\"synthetic-observed\",\"objective\":\"0.5*sum(r^2)\","
           "\"receiver_cotangent\":\"r_chronological\",\"objective_dt_factor\":0,"
           "\"receiver_dt_factor\":0,\"material_quadrature\":\"discrete_sum_once\","
           "\"dtinv\":1},\"J_base\":%.17g,\"D_ad\":%.17g,"
           "\"D_primary\":%.17g,\"D_rho\":%.17g,\"D_Q\":%.17g,"
           "\"direction_norm\":%.17g,\"max_trace\":%.17g}\n",
           test->name, test->invmat1,
           test->qmode == Q_PARAMETERIZATION_LEGACY ? "legacy" : "physical",
           test->directions, j_base, d_ad, d_primary, d_rho, d_q,
           direction_norm, max_trace);

    for (e = 0; e < 4; ++e) {
        double ep = epsilons[e], jp, jm, dfd, rel;
        prepare_model(ep, 0, test, &mapping, &primary, &rho, &q, &mu, &tau, &rhoi,
                      &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip, &d,
                      eta, bip, bjm, reference_sum);
        forward_trace(nsteps, NULL, &rhoi, &fipjp, &f, &dip, &d, &pp,
                      diag, &dummy, srcpos, signals, hc, bip, bjm, cip, cjm,
                      K, Kh, a, ah, b, bh, bl, br, bt, bb, request,
                      receiver_i, receiver_j, plus);
        prepare_model(-ep, 0, test, &mapping, &primary, &rho, &q, &mu, &tau, &rhoi,
                      &mu_x, &mu_y, &tau_x, &tau_y, &fipjp, &f, &dip, &d,
                      eta, bip, bjm, reference_sum);
        forward_trace(nsteps, NULL, &rhoi, &fipjp, &f, &dip, &d, &pp,
                      diag, &dummy, srcpos, signals, hc, bip, bjm, cip, cjm,
                      K, Kh, a, ah, b, bh, bl, br, bt, bb, request,
                      receiver_i, receiver_j, minus);
        jp = objective(plus, observed, nsteps); jm = objective(minus, observed, nsteps);
        dfd = (jp - jm) / (2.0 * ep);
        rel = fabs(dfd - d_ad) / fmax(fmax(fabs(dfd), fabs(d_ad)), 1.0e-300);
        printf("{\"case\":\"%s\",\"epsilon\":%.17g,"
               "\"J_plus\":%.17g,\"J_minus\":%.17g,"
               "\"D_fd\":%.17g,\"D_ad\":%.17g,\"D_ad_over_D_fd\":%.17g,"
               "\"relative_error\":%.17g}\n", test->name, ep, jp, jm,
               dfd, d_ad, d_ad / dfd, rel);
    }
    visco_sh_material_observable_trajectory_release(&trajectory);
    MPI_Buffer_detach(&bsend, &bsize); free(bsend); MPI_Finalize();
    return 0;
}
