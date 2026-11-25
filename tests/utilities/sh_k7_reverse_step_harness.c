/* K-7 bitwise harness (docs/sh_k7_reverse_speedup_contract.md): the
 * production exact SH reverse step against the frozen base step
 * (sh_k7_reference/k7_reference_step.c), over several consecutive steps.
 * Inputs contain -0.0/+0.0 cells and a zero block, duplicate receiver and
 * source cells and a source of another type.  Variants: fixed material,
 * material capture without plan, with the per-shot plan, and with the plan
 * accumulating into the shot sum.  Every float and double of the state,
 * the source cotangent and the material output is compared by its bits.
 *
 * The x and y directions get different relaxation weights (eta_x, eta_y)
 * and different CPML profiles, so that a swap of the two is detected.
 *
 * usage: npx npy boundary fs fd L fw nx ny steps inject
 *   inject: none | all_negzero | nan_sxz | inf_receiver | nan_halo_vz |
 *           nan_qsum | neg_mu
 *   all_negzero is a valid input: every adjoint state word and receiver
 *   cotangent is -0.0.  The others are error injections; they run the
 *   material variants only with one rank, where a failing step returns
 *   instead of aborting. */

#define main m63c_locked_c5a_harness_main
#include "m63c_full_state_step_harness.c"
#undef main

int k7ref_full_state_adjoint_step(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *bar_next_work,
        struct visco_sh_full_state *bar_prev, double *bar_signal);
int k7ref_full_state_adjoint_step_material(
        const struct visco_sh_full_step_config *cfg,
        struct visco_sh_full_state *bar_next_work,
        struct visco_sh_full_state *bar_prev, double *bar_signal,
        const struct visco_sh_material_adjoint_step_context *material);

#define K7_NREC 4
#define K7_NSRC 3

struct dgrid { double **v, *data; };

static struct dgrid dgrid_new(void) {
    struct dgrid g; int j;
    double **rows = (double **)calloc((size_t)NY + 1, sizeof(double *));
    g.data = (double *)calloc((size_t)NY * NX, sizeof(double));
    if (!rows || !g.data) MPI_Abort(MPI_COMM_WORLD, 701);
    for (j = 1; j <= NY; ++j) rows[j] = g.data + (size_t)(j - 1) * NX - 1;
    g.v = rows;
    return g;
}
static void dgrid_free(struct dgrid *g) { free(g->data); free(g->v); }

struct native { struct dgrid c[5]; struct visco_sh_native_material_gradient_fields f; };
static void native_new(struct native *n) {
    int k; for (k = 0; k < 5; ++k) n->c[k] = dgrid_new();
    n->f.g_rhoi = n->c[0].v; n->f.g_mu_x = n->c[1].v; n->f.g_mu_y = n->c[2].v;
    n->f.g_tau_x = n->c[3].v; n->f.g_tau_y = n->c[4].v;
}
static void native_free(struct native *n) { int k; for (k = 0; k < 5; ++k) dgrid_free(&n->c[k]); }
static void native_fill(struct native *n, double value) {
    int k, p; for (k = 0; k < 5; ++k) for (p = 0; p < NX * NY; ++p) n->c[k].data[p] = value;
}

static long differ_words(const void *a, const void *b, size_t count, size_t width) {
    const unsigned char *x = (const unsigned char *)a, *y = (const unsigned char *)b;
    long d = 0; size_t k;
    for (k = 0; k < count; ++k) d += memcmp(x + k * width, y + k * width, width) != 0;
    return d;
}
static long field_diff(const struct field *a, const struct field *b) {
    return differ_words(a->data, b->data, (size_t)a->nrows * a->ncols, sizeof(float));
}
static long volume_diff(const struct volume *a, const struct volume *b) {
    return differ_words(a->data, b->data,
                        (size_t)a->nrows * a->ncols * (a->mechanisms + 1), sizeof(float));
}
static long state_diff(const struct owned_state *a, const struct owned_state *b, int fw) {
    long d = field_diff(&a->vz, &b->vz) + field_diff(&a->sxz, &b->sxz) +
             field_diff(&a->syz, &b->syz) + volume_diff(&a->r, &b->r) + volume_diff(&a->q, &b->q);
    if (fw) d += field_diff(&a->psi_sxz_x, &b->psi_sxz_x) + field_diff(&a->psi_syz_y, &b->psi_syz_y) +
                 field_diff(&a->psi_vzx, &b->psi_vzx) + field_diff(&a->psi_vzy, &b->psi_vzy);
    return d;
}
static long native_diff(const struct native *a, const struct native *b) {
    long d = 0; int k;
    for (k = 0; k < 5; ++k) d += differ_words(a->c[k].data, b->c[k].data, (size_t)NX * NY, sizeof(double));
    return d;
}

static void zero_pattern(float *data, size_t n, int seed) {
    size_t k;
    for (k = 0; k < n; ++k) {
        if ((k + (size_t)seed) % 7 == 0) data[k] = -0.0f;
        else if ((k + (size_t)seed) % 11 == 0) data[k] = 0.0f;
    }
}
/* -0.0/+0.0 cells everywhere and a block of -0.0 in rows 1..2, so that
 * whole stencils see only zeros of both signs. */
static void inject_zeros(struct owned_state *s, int fw, int mechanisms) {
    int i, j, l;
    zero_pattern(s->vz.data, (size_t)s->vz.nrows * s->vz.ncols, 1);
    zero_pattern(s->sxz.data, (size_t)s->sxz.nrows * s->sxz.ncols, 2);
    zero_pattern(s->syz.data, (size_t)s->syz.nrows * s->syz.ncols, 3);
    zero_pattern(s->r.data, (size_t)s->r.nrows * s->r.ncols * (mechanisms + 1), 4);
    zero_pattern(s->q.data, (size_t)s->q.nrows * s->q.ncols * (mechanisms + 1), 5);
    if (fw) {
        zero_pattern(s->psi_sxz_x.data, (size_t)s->psi_sxz_x.nrows * s->psi_sxz_x.ncols, 6);
        zero_pattern(s->psi_syz_y.data, (size_t)s->psi_syz_y.nrows * s->psi_syz_y.ncols, 7);
        zero_pattern(s->psi_vzx.data, (size_t)s->psi_vzx.nrows * s->psi_vzx.ncols, 8);
        zero_pattern(s->psi_vzy.data, (size_t)s->psi_vzy.nrows * s->psi_vzy.ncols, 9);
    }
    for (j = 1; j <= 2 && j <= NY; ++j)
        for (i = 1; i <= (NX + 1) / 2; ++i) {
            s->view.vz[j][i] = s->view.sxz[j][i] = s->view.syz[j][i] = -0.0f;
            for (l = 1; l <= mechanisms; ++l) s->view.r[j][i][l] = s->view.q[j][i][l] = -0.0f;
        }
}

static void fill_negzero(float *data, size_t n) {
    size_t k; for (k = 0; k < n; ++k) data[k] = -0.0f;
}
/* every word of the adjoint state, halo and memory variables included */
static void all_negzero(struct owned_state *s, int fw, int mechanisms) {
    fill_negzero(s->vz.data, (size_t)s->vz.nrows * s->vz.ncols);
    fill_negzero(s->sxz.data, (size_t)s->sxz.nrows * s->sxz.ncols);
    fill_negzero(s->syz.data, (size_t)s->syz.nrows * s->syz.ncols);
    fill_negzero(s->r.data, (size_t)s->r.nrows * s->r.ncols * (mechanisms + 1));
    fill_negzero(s->q.data, (size_t)s->q.nrows * s->q.ncols * (mechanisms + 1));
    if (fw) {
        fill_negzero(s->psi_sxz_x.data, (size_t)s->psi_sxz_x.nrows * s->psi_sxz_x.ncols);
        fill_negzero(s->psi_syz_y.data, (size_t)s->psi_syz_y.nrows * s->psi_syz_y.ncols);
        fill_negzero(s->psi_vzx.data, (size_t)s->psi_vzx.nrows * s->psi_vzx.ncols);
        fill_negzero(s->psi_vzy.data, (size_t)s->psi_vzy.nrows * s->psi_vzy.ncols);
    }
}

struct impl { struct owned_state buf[3]; double signal[K7_NSRC]; struct native out, sum; int status; };

int main(int argc, char **argv) {
    struct owned_state dual;
    struct field rhoi, fipjp, f, mu_x, tau_x, mu_y, tau_y, qsum, strain_x, strain_y;
    struct volume dip, d;
    struct visco_sh_full_step_config cfg, step_cfg;
    struct visco_sh_material_observable_step observable;
    struct visco_sh_material_adjoint_step_context ctx_ref, ctx_cand;
    struct visco_sh_reverse_material_plan *plan = NULL;
    struct impl ref, cand;
    float *hc, *bip, *bjm, *cip, *cjm, *K, *Kh, *a, *ah, *b, *bh, *eta, *eta_y;
    float *Ky, *Kyh, *ay, *ayh, *by, *byh;
    double *bar_receiver;
    int rec_x[K7_NREC], rec_y[K7_NREC], src_x[K7_NSRC], src_y[K7_NSRC], src_type[K7_NSRC];
    int rank, size, h, i, j, l, k, n, steps, variant, first_variant, last_variant, cur, nxt, last;
    const char *inject;
    long local[4], total[4];

    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    MYID = rank; FP = stderr;
    if (argc != 12) { if (rank == 0) fprintf(stderr, "usage: npx npy boundary fs fd L fw nx ny steps inject\n"); MPI_Abort(MPI_COMM_WORLD, 702); }
    NPROCX = atoi(argv[1]); NPROCY = atoi(argv[2]); BOUNDARY = atoi(argv[3]); FREE_SURF = atoi(argv[4]);
    FDORDER = atoi(argv[5]); L = atoi(argv[6]); FW = atoi(argv[7]); NX = atoi(argv[8]); NY = atoi(argv[9]);
    steps = atoi(argv[10]); inject = argv[11];
    if (size != NPROCX * NPROCY || steps < 1) MPI_Abort(MPI_COMM_WORLD, 703);
    topology(rank); h = FDORDER / 2; DT = 0.0013f; DH = 7.5f;

    rhoi = field_new(-h, NY + h + 1, 1 - h, NX + h); fipjp = field_new(-h, NY + h + 1, 1 - h, NX + h);
    f = field_new(-h, NY + h + 1, 1 - h, NX + h);
    dip = volume_new(-h, NY + h + 1, 1 - h, NX + h, L); d = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    for (j = -h; j <= NY + h + 1; ++j) for (i = 1 - h; i <= NX + h; ++i) {
        rhoi.v[j][i] = (float)(0.00043 + 0.000002 * rank + 0.0000003 * j);
        fipjp.v[j][i] = (float)(0.0047 + 0.00001 * i + 0.000006 * j);
        f.v[j][i] = (float)(0.0042 + 0.000008 * i + 0.000004 * j);
        for (l = 1; l <= L; ++l) {
            dip.v[j][i][l] = (float)(0.16 + 0.004 * (l - 1) + 0.0002 * i);
            d.v[j][i][l] = (float)(0.14 + 0.003 * (l - 1) + 0.0002 * j);
        }
    }
    hc = (float *)calloc((size_t)h + 1, sizeof(float)); bip = (float *)calloc((size_t)L + 1, sizeof(float));
    bjm = (float *)calloc((size_t)L + 1, sizeof(float)); cip = (float *)calloc((size_t)L + 1, sizeof(float));
    cjm = (float *)calloc((size_t)L + 1, sizeof(float)); eta = (float *)calloc((size_t)L + 1, sizeof(float));
    eta_y = (float *)calloc((size_t)L + 1, sizeof(float));
    { double values[6] = {1.0, -0.041, 0.007, -0.0014, 0.00031, -0.00007}; for (k = 1; k <= h; ++k) hc[k] = (float)values[k - 1]; }
    for (l = 1; l <= L; ++l) {
        bip[l] = (float)(0.79 + 0.025 * (l - 1)); bjm[l] = (float)(0.77 + 0.021 * (l - 1));
        cip[l] = (float)(0.91 - 0.018 * (l - 1)); cjm[l] = (float)(0.89 - 0.015 * (l - 1));
        eta[l] = (float)(0.031 * l); eta_y[l] = (float)(0.027 * l + 0.002);
    }
    K = (float *)calloc((size_t)2 * FW + 1, sizeof(float)); Kh = (float *)calloc((size_t)2 * FW + 1, sizeof(float));
    a = (float *)calloc((size_t)2 * FW + 1, sizeof(float)); ah = (float *)calloc((size_t)2 * FW + 1, sizeof(float));
    b = (float *)calloc((size_t)2 * FW + 1, sizeof(float)); bh = (float *)calloc((size_t)2 * FW + 1, sizeof(float));
    for (k = 1; k <= 2 * FW; ++k) {
        K[k] = (float)(1.11 + 0.017 * k); Kh[k] = (float)(1.08 + 0.013 * k);
        a[k] = (float)(-0.031 - 0.001 * k); ah[k] = (float)(-0.027 - 0.0008 * k);
        b[k] = (float)(0.82 + 0.006 * k); bh[k] = (float)(0.84 + 0.005 * k);
    }
    Ky = (float *)calloc((size_t)2 * FW + 1, sizeof(float)); Kyh = (float *)calloc((size_t)2 * FW + 1, sizeof(float));
    ay = (float *)calloc((size_t)2 * FW + 1, sizeof(float)); ayh = (float *)calloc((size_t)2 * FW + 1, sizeof(float));
    by = (float *)calloc((size_t)2 * FW + 1, sizeof(float)); byh = (float *)calloc((size_t)2 * FW + 1, sizeof(float));
    for (k = 1; k <= 2 * FW; ++k) {
        Ky[k] = (float)(1.19 + 0.023 * k); Kyh[k] = (float)(1.15 + 0.019 * k);
        ay[k] = (float)(-0.043 - 0.0013 * k); ayh[k] = (float)(-0.038 - 0.0011 * k);
        by[k] = (float)(0.76 + 0.009 * k); byh[k] = (float)(0.79 + 0.007 * k);
    }
    mu_x = field_new(1, NY, 1, NX); tau_x = field_new(1, NY, 1, NX); mu_y = field_new(1, NY, 1, NX);
    tau_y = field_new(1, NY, 1, NX); qsum = field_new(1, NY, 1, NX); strain_x = field_new(1, NY, 1, NX);
    strain_y = field_new(1, NY, 1, NX);
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i) {
        mu_x.v[j][i] = (float)(1.7e7 * (1.0 + 0.01 * i + 0.003 * j + 0.02 * rank));
        mu_y.v[j][i] = (float)(1.6e7 * (1.0 + 0.008 * i + 0.004 * j + 0.01 * rank));
        tau_x.v[j][i] = (float)(0.04 + 0.0007 * i + 0.0003 * j);
        tau_y.v[j][i] = (float)(0.05 + 0.0005 * i + 0.0004 * j);
        qsum.v[j][i] = deterministic(rank, 30, j, i, 0);
        strain_x.v[j][i] = deterministic(rank, 31, j, i, 1);
        strain_y.v[j][i] = deterministic(rank, 32, j, i, 0);
    }
    zero_pattern(qsum.data, (size_t)NX * NY, 3); zero_pattern(strain_x.data, (size_t)NX * NY, 4);
    zero_pattern(strain_y.data, (size_t)NX * NY, 5);
    observable.qsum = qsum.v; observable.strain_x = strain_x.v; observable.strain_y = strain_y.v;

    /* receivers 0 and 1 share a cell (their cotangents meet in double before
     * rounding); sources 0 and 1 share a cell, source 2 is not of type 1 */
    rec_x[0] = rec_x[1] = 2 + rank % 2; rec_y[0] = rec_y[1] = 3;
    rec_x[2] = NX; rec_y[2] = 1; rec_x[3] = 1 + NX / 2; rec_y[3] = NY;
    src_x[0] = src_x[1] = 3; src_y[0] = src_y[1] = 2 + rank % 2; src_type[0] = src_type[1] = 1;
    src_x[2] = NX - 1 > 0 ? NX - 1 : 1; src_y[2] = NY; src_type[2] = 2;
    bar_receiver = (double *)calloc((size_t)steps * K7_NREC, sizeof(double));
    for (n = 0; n < steps; ++n) for (k = 0; k < K7_NREC; ++k)
        bar_receiver[n * K7_NREC + k] = (n == 1 && k == 2) ? -0.0 : (n == 2 && k == 3) ? 0.0 :
                                        -0.17 + 0.019 * rank + 0.031 * k - 0.007 * n;
    if (!strcmp(inject, "inf_receiver")) bar_receiver[(steps - 1) * K7_NREC + 1] = INFINITY;

    memset(&cfg, 0, sizeof(cfg));
    cfg.nx = NX; cfg.ny = NY; cfg.fdorder = FDORDER; cfg.mechanisms = L; cfg.fw = FW;
    cfg.free_surface = FREE_SURF; cfg.boundary = BOUNDARY; memcpy(cfg.pos, POS, sizeof(POS));
    memcpy(cfg.index, INDEX, sizeof(INDEX)); cfg.nproc_x = NPROCX; cfg.nproc_y = NPROCY;
    cfg.dt = DT; cfg.dh = DH; cfg.hc = hc; cfg.rhoi = rhoi.v; cfg.fipjp = fipjp.v; cfg.f = f.v;
    cfg.bip = bip; cfg.bjm = bjm; cfg.cip = cip; cfg.cjm = cjm; cfg.dip = dip.v; cfg.d = d.v;
    cfg.K_x = K; cfg.a_x = a; cfg.b_x = b; cfg.K_x_half = Kh; cfg.a_x_half = ah; cfg.b_x_half = bh;
    cfg.K_y = Ky; cfg.a_y = ay; cfg.b_y = by; cfg.K_y_half = Kyh; cfg.a_y_half = ayh; cfg.b_y_half = byh;
    cfg.nrec = K7_NREC; cfg.rec_x = rec_x; cfg.rec_y = rec_y; cfg.nsrc = K7_NSRC;
    cfg.src_x = src_x; cfg.src_y = src_y; cfg.source_type = src_type; cfg.comm = MPI_COMM_WORLD;

    dual = state_new(h, FW, L); state_initialize(&dual, rank, 1, FW, L); inject_zeros(&dual, FW, L);
    if (!strcmp(inject, "all_negzero")) {
        all_negzero(&dual, FW, L);
        for (k = 0; k < steps * K7_NREC; ++k) bar_receiver[k] = -0.0;
    }
    if (!strcmp(inject, "nan_sxz")) dual.view.sxz[1 + NY / 2][1 + NX / 2] = NAN;
    if (!strcmp(inject, "nan_halo_vz")) dual.view.vz[NY + 1][NX + h] = NAN;
    if (!strcmp(inject, "nan_qsum")) qsum.v[1 + NY / 2][1] = NAN;
    if (!strcmp(inject, "neg_mu")) mu_y.v[NY][NX] = -1.0f;
    for (k = 0; k < 3; ++k) { ref.buf[k] = state_new(h, FW, L); cand.buf[k] = state_new(h, FW, L); }
    native_new(&ref.out); native_new(&ref.sum); native_new(&cand.out); native_new(&cand.sum);

    memset(&ctx_ref, 0, sizeof(ctx_ref));
    ctx_ref.observable = &observable; ctx_ref.mu_x = mu_x.v; ctx_ref.tau_x = tau_x.v;
    ctx_ref.mu_y = mu_y.v; ctx_ref.tau_y = tau_y.v; ctx_ref.reference_sum = 0.37;
    ctx_ref.eta_x = eta; ctx_ref.eta_y = eta_y; ctx_ref.native_output = &ref.out.f;

    first_variant = 0; last_variant = 3;
    for (variant = first_variant; variant <= last_variant; ++variant) {
        if (variant > 0 && size > 1 && strcmp(inject, "none") && strcmp(inject, "all_negzero"))
            continue;   /* would abort */
        if (variant == 0 && (!strcmp(inject, "nan_qsum") || !strcmp(inject, "neg_mu"))) continue;
        for (k = 0; k < 3; ++k) {
            state_initialize(&ref.buf[k], rank, k, FW, L); state_initialize(&cand.buf[k], rank, k, FW, L);
        }
        state_copy(&ref.buf[0], &dual, FW); state_copy(&cand.buf[0], &dual, FW);
        native_fill(&ref.out, 3.5); native_fill(&cand.out, 3.5);
        native_fill(&ref.sum, 0.0); native_fill(&cand.sum, 0.0);
        ctx_cand = ctx_ref; ctx_cand.native_output = (variant == 3) ? &cand.sum.f : &cand.out.f;
        plan = NULL;
        ref.status = cand.status = 0;
        if (variant >= 2) {
            cand.status = visco_sh_reverse_material_plan_create(&cfg, &ctx_cand, &plan);
            ctx_cand.plan = plan; ctx_cand.accumulate = (variant == 3);
        }
        local[0] = local[1] = local[2] = 0; local[3] = 0;
        cur = 0;
        for (n = steps - 1; n >= 0; --n) {
            nxt = (n == 0) ? 2 : (cur == 1 ? 0 : 1);
            step_cfg = cfg; step_cfg.bar_receiver = bar_receiver + (size_t)n * K7_NREC;
            if (ref.status == 0)
                ref.status = (variant == 0) ?
                    k7ref_full_state_adjoint_step(&step_cfg, &ref.buf[cur].view, &ref.buf[nxt].view, ref.signal) :
                    k7ref_full_state_adjoint_step_material(&step_cfg, &ref.buf[cur].view, &ref.buf[nxt].view, ref.signal, &ctx_ref);
            if (cand.status == 0)
                cand.status = (variant == 0) ?
                    visco_sh_full_state_adjoint_step(&step_cfg, &cand.buf[cur].view, &cand.buf[nxt].view, cand.signal) :
                    visco_sh_full_state_adjoint_step_material(&step_cfg, &cand.buf[cur].view, &cand.buf[nxt].view, cand.signal, &ctx_cand);
            if (ref.status != 0 || cand.status != 0) break;
            if (variant == 3)
                for (k = 0; k < 5; ++k) for (i = 0; i < NX * NY; ++i) ref.sum.c[k].data[i] += ref.out.c[k].data[i];
            local[0] += state_diff(&ref.buf[nxt], &cand.buf[nxt], FW) + state_diff(&ref.buf[cur], &cand.buf[cur], FW);
            local[1] += differ_words(ref.signal, cand.signal, K7_NSRC, sizeof(double));
            if (variant == 1 || variant == 2) local[2] += native_diff(&ref.out, &cand.out);
            if (variant == 3) local[2] += native_diff(&ref.sum, &cand.sum);
            local[3] += 1;
            cur = nxt;
        }
        visco_sh_reverse_material_plan_free(plan);
        MPI_Allreduce(local, total, 4, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        last = (ref.status != 0) + 2 * (cand.status != 0);
        if (rank == 0)
            printf("{\"variant\":%d,\"inject\":\"%s\",\"steps_compared\":%ld,\"state_words_diff\":%ld,"
                   "\"signal_diff\":%ld,\"native_diff\":%ld,\"status_ref\":%d,\"status_cand\":%d,"
                   "\"failed\":%d}\n", variant, inject, total[3], total[0], total[1], total[2],
                   ref.status, cand.status, last);
        fflush(stdout);
    }
    MPI_Finalize();
    return 0;
}
