/* Direct runtime oracle for the inactive C8c B3b exact objective wrapper. */

#ifndef M63C_B3A_SUPPORT
#define M63C_B3A_SUPPORT "m63c_c8c_exact_objective_shot_harness.c"
#endif
#include M63C_B3A_SUPPORT

int RUN_MULTIPLE_SHOTS, READREC, SWS_TAPER_CIRCULAR_PER_SHOT, TIME_FILT;
int QUELLTYPB, COLOR, NCOLORS, NSHOT1, NSHOT2, NSHOTS, MYID_SHOT;
int IENDX, IENDY;
float TS;
char SIGNAL_FILE[STRING_SIZE], DATA_DIR[STRING_SIZE], DATA_DIR_T0[STRING_SIZE];
char SEIS_FILE_P[STRING_SIZE], SEIS_FILE_VX[STRING_SIZE], SEIS_FILE_VY[STRING_SIZE];
float TIME, REFREC[4];

float *rd_sour(int *nts, FILE *fp) {
    (void)nts; (void)fp; MPI_Abort(MPI_COMM_WORLD, 989); return NULL;
}

static int objective_calls, objective_cardinality[4], inject_failure_on_call;
static int inseis_calls, inseis_components[4], splitsrc_calls;
static int gradient_shot_calls, material_vjp_calls;
static double objective_values[4];

float **fmatrix(int nrl, int nrh, int ncl, int nch) {
    return matrix(nrl, nrh, ncl, nch);
}

int *ivector(int nl, int nh) {
    int *base = (int *)calloc((size_t)(nh - nl + 1), sizeof(*base));
    if (base == NULL) MPI_Abort(MPI_COMM_WORLD, 990);
    return base - nl;
}

void free_ivector(int *values, int nl, int nh) {
    (void)nh;
    if (values != NULL) free(values + nl);
}

void free_imatrix(int **values, int nrl, int nrh, int ncl, int nch) {
    (void)nrh; (void)nch;
    if (values != NULL) { free(values[nrl] + ncl); free(values + nrl); }
}

/* The B3b wrapper's READREC=2 plumbing is intentionally unreachable in this
 * fixed-geometry oracle.  These are only link seams for that dormant branch. */
int **receiver(FILE *fp, int *nrec, int ishot) {
    (void)fp; (void)nrec; (void)ishot; MPI_Abort(MPI_COMM_WORLD, 991); return NULL;
}

int **splitrec(int **recpos, int *nrec_local, int nrec_global, int *switches) {
    (void)recpos; (void)nrec_local; (void)nrec_global; (void)switches;
    MPI_Abort(MPI_COMM_WORLD, 992); return NULL;
}

void alloc_seisSH(int nrec, int ns, struct seisSH *seismogram) {
    (void)nrec; (void)ns; (void)seismogram; MPI_Abort(MPI_COMM_WORLD, 993);
}

void alloc_seisSHfwi(int nrec, int nrec_global, int ns, struct seisSHfwi *legacy) {
    (void)nrec; (void)nrec_global; (void)ns; (void)legacy;
    MPI_Abort(MPI_COMM_WORLD, 994);
}

void apply_tdfilt(float **section, int ntr, int ns, int order,
                  float high, float low) {
    (void)section; (void)ntr; (void)ns; (void)order; (void)high; (void)low;
    MPI_Abort(MPI_COMM_WORLD, 995);
}

/* Deterministic observed-data seam; production inseis is file I/O only. */
void inseis(int component, float **section, int ntr, int ns, int sws, int iter) {
    int i, n;
    (void)sws; (void)iter;
    if (inseis_calls < 4) inseis_components[inseis_calls] = component;
    ++inseis_calls;
    for (i = 1; i <= ntr; ++i)
        for (n = 1; n <= ns; ++n)
            section[i][n] = (float)(0.00017 * n + 0.0013 * component);
}

float **__real_splitsrc(float **srcpos, int *nsrc_local, int nsrc);

float **__wrap_splitsrc(float **srcpos, int *nsrc_local, int nsrc) {
    ++splitsrc_calls;
    return __real_splitsrc(srcpos, nsrc_local, nsrc);
}

int __real_visco_sh_exact_objective_shot(
        const struct visco_sh_exact_objective_shot_request *,
        struct visco_sh_exact_objective_shot_result *);

int __wrap_visco_sh_exact_objective_shot(
        const struct visco_sh_exact_objective_shot_request *request,
        struct visco_sh_exact_objective_shot_result *result) {
    int slot = objective_calls++;
    if (slot < 4) objective_cardinality[slot] = request->nsrc_local;
    if (inject_failure_on_call == objective_calls) return -77;
    if (__real_visco_sh_exact_objective_shot(request, result) != 0) {
        fprintf(stderr, "wrapped objective shot failed: sources=%d receivers=%d ns=%d\n",
                request->nsrc_local, request->nrec_local, request->ns);
        return -1;
    }
    if (slot < 4) objective_values[slot] = result->objective;
    return 0;
}

int __real_visco_sh_exact_objective_gradient_shot(
        const struct visco_sh_exact_shot_request *,
        struct visco_sh_exact_shot_result *);

int __wrap_visco_sh_exact_objective_gradient_shot(
        const struct visco_sh_exact_shot_request *request,
        struct visco_sh_exact_shot_result *result) {
    ++gradient_shot_calls;
    return __real_visco_sh_exact_objective_gradient_shot(request, result);
}

int __real_visco_sh_distributed_material_gradient_vjp(
        int, const struct q_tau_mapping *, float **, float **, float **,
        const struct visco_sh_native_material_gradient_fields *,
        float **, float **, float **);

int __wrap_visco_sh_distributed_material_gradient_vjp(
        int invmat1, const struct q_tau_mapping *mapping,
        float **primary_post, float **rho_post, float **owned_q,
        const struct visco_sh_native_material_gradient_fields *native,
        float **grad_primary, float **grad_rho, float **grad_q) {
    ++material_vjp_calls;
    return __real_visco_sh_distributed_material_gradient_vjp(
            invmat1, mapping, primary_post, rho_post, owned_q, native,
            grad_primary, grad_rho, grad_q);
}

struct multi_fixture {
    struct field rhoi, fipjp, f, primary, rho, q, mu, tau, mu_x, mu_y;
    struct field tau_x, tau_y, grad_primary, grad_rho, grad_q;
    struct volume dip, d, pp;
    struct q_tau_mapping mapping;
    struct matSH material;
    struct bridge_storage storage;
    float **srcpos, **srcpos1;
    int **recpos;
    int *dtinv_help;
    float *hc, *bip, *bjm, *cip, *cjm, *eta, *frequencies;
    float *K, *Kh, *a, *ah, *b, *bh;
    float **bl, **br, **bt, **bb, *sl, *sr, *st, *sb;
    double reference_sum;
};

static void reset_observation_storage(struct multi_fixture *fixture) {
    int n;
    for (n = 1; n <= NT; ++n) {
        fixture->storage.legacy.sectionread[1][n] = 0.0f;
        fixture->storage.legacy.sectionvzdata[1][n] = 0.0f;
        fixture->storage.seismogram.sectionvz[1][n] = 0.0f;
    }
}

static void setup_fixture(struct multi_fixture *fixture) {
    const int h = 2;
    int l;
    memset(fixture, 0, sizeof(*fixture));
    prepare_globals(0, 48);
    IENDX = NX; IENDY = NY;
    RUN_MULTIPLE_SHOTS = 1; READREC = 1; SWS_TAPER_CIRCULAR_PER_SHOT = 0;
    TIME_FILT = 0; INV_STF = 0; QUELLTYPB = 1; QUELLART = 1;
    EPRECOND = 0; INVMAT1 = 1; ORDER_SPIKE = 0; TS = 1.0f;
    Q_PARAMETERIZATION_MODE = Q_PARAMETERIZATION_PHYSICAL;
    Q_APPROX_FMIN = 2.0f; Q_APPROX_FMAX = 18.0f; Q_APPROX_DF = 0.5f;
#define FIELD(name) fixture->name = field_new(-h, NY + h + 1, 1 - h, NX + h)
    FIELD(rhoi); FIELD(fipjp); FIELD(f); FIELD(primary); FIELD(rho); FIELD(q);
    FIELD(mu); FIELD(tau); FIELD(mu_x); FIELD(mu_y); FIELD(tau_x); FIELD(tau_y);
    FIELD(grad_primary); FIELD(grad_rho); FIELD(grad_q);
#undef FIELD
    fixture->dip = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    fixture->d = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    fixture->pp = volume_new(-h, NY + h + 1, 1 - h, NX + h, L);
    fixture->hc = (float *)calloc(3, sizeof(float)); fixture->hc[1] = 1.0f;
    fixture->hc[2] = -0.041f;
    fixture->bip = (float *)calloc(3, sizeof(float));
    fixture->bjm = (float *)calloc(3, sizeof(float));
    fixture->cip = (float *)calloc(3, sizeof(float));
    fixture->cjm = (float *)calloc(3, sizeof(float));
    fixture->eta = (float *)calloc(3, sizeof(float));
    fixture->frequencies = (float *)calloc(3, sizeof(float));
    fixture->frequencies[1] = 4.0f; fixture->frequencies[2] = 9.0f; FL = fixture->frequencies;
    init_q_tau_mapping(&fixture->mapping, Q_PARAMETERIZATION_MODE, L, FL,
            Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF);
    for (l = 1; l <= L; ++l) {
        double theta = 1.0 / (2.0 * PI * fixture->frequencies[l]);
        double x = 2.0 * PI * fixture->frequencies[1] * theta;
        fixture->eta[l] = (float)(DT / theta);
        fixture->reference_sum += x * x / (1.0 + x * x);
        fixture->bip[l] = fixture->bjm[l] = 1.0f / (1.0f + 0.5f * fixture->eta[l]);
        fixture->cip[l] = fixture->cjm[l] = 1.0f - 0.5f * fixture->eta[l];
    }
    fixture->K = (float *)calloc(1, sizeof(float)); fixture->Kh = (float *)calloc(1, sizeof(float));
    fixture->a = (float *)calloc(1, sizeof(float)); fixture->ah = (float *)calloc(1, sizeof(float));
    fixture->b = (float *)calloc(1, sizeof(float)); fixture->bh = (float *)calloc(1, sizeof(float));
    fixture->bl = buffer_new(NY, 2 * (h + 1), &fixture->sl);
    fixture->br = buffer_new(NY, 2 * (h + 1), &fixture->sr);
    fixture->bt = buffer_new(NX, 2 * (h + 1), &fixture->st);
    fixture->bb = buffer_new(NX, 2 * (h + 1), &fixture->sb);
    fixture->srcpos = matrix(1, 8, 1, 2); fixture->srcpos1 = matrix(1, 8, 1, 1);
    fixture->srcpos[1][1] = 5.0f * DH; fixture->srcpos[2][1] = 9.0f * DH;
    fixture->srcpos[4][1] = 0.0f; fixture->srcpos[5][1] = 20.0f; fixture->srcpos[6][1] = 0.015f; fixture->srcpos[8][1] = 1.0f;
    fixture->srcpos[1][2] = 11.0f * DH; fixture->srcpos[2][2] = 9.0f * DH;
    fixture->srcpos[4][2] = 0.004f; fixture->srcpos[5][2] = 17.0f; fixture->srcpos[6][2] = 0.011f; fixture->srcpos[8][2] = 1.0f;
    fixture->recpos = test_imatrix(1, 3, 1, 1);
    fixture->recpos[1][1] = 15; fixture->recpos[2][1] = 9; fixture->recpos[3][1] = 1;
    fixture->dtinv_help = (int *)calloc((size_t)NT + 1, sizeof(int));
    bridge_storage_init(&fixture->storage, NT, 5, 9, 15, 9,
            NULL, NULL, fixture->bl, fixture->br, fixture->bt, fixture->bb,
            fixture->K, fixture->Kh, fixture->a, fixture->ah, fixture->b, fixture->bh);
    free_imatrix(fixture->storage.recpos, 1, 2, 1, 1);
    fixture->storage.recpos = fixture->recpos;
    fixture->storage.acquisition.recpos = fixture->recpos;
    fixture->storage.acquisition.recpos_loc = fixture->recpos;
    fixture->storage.acquisition.srcpos = fixture->srcpos;
    fixture->storage.acquisition.srcpos1 = fixture->srcpos1;
    free(fixture->storage.dtinv_help); fixture->storage.dtinv_help = fixture->dtinv_help;
    fixture->storage.legacy.sectionread = matrix(1, 1, 1, NT);
    fixture->storage.legacy.sectionvzdata = matrix(1, 1, 1, NT);
    fixture->storage.legacy.sectionvzdiff = matrix(1, 1, 1, NT);
    fixture->storage.legacy.sectionvzdiffold = matrix(1, 1, 1, NT);
    bridge_bind_material(&fixture->material, &fixture->rho, &fixture->rhoi,
            &fixture->primary, &fixture->mu_x, &fixture->mu_y, &fixture->q,
            &fixture->tau_y, &fixture->tau_x, &fixture->fipjp, &fixture->f,
            &fixture->dip, &fixture->d, &fixture->pp, fixture->eta, fixture->bip,
            fixture->bjm, fixture->cip, fixture->cjm);
}

static void prepare_fixture_model(struct multi_fixture *fixture) {
    struct directional_case test = {"multi", 1, Q_PARAMETERIZATION_PHYSICAL,
            DIRECTION_PRIMARY | DIRECTION_RHO | DIRECTION_Q, 1, 1, 0, 0,
            NT, 5, 9, 15, 9};
    prepare_model(0.0, 0, &test, &fixture->mapping, &fixture->primary,
            &fixture->rho, &fixture->q, &fixture->mu, &fixture->tau,
            &fixture->rhoi, &fixture->mu_x, &fixture->mu_y, &fixture->tau_x,
            &fixture->tau_y, &fixture->fipjp, &fixture->f, &fixture->dip,
            &fixture->d, fixture->eta, fixture->bip, fixture->bjm,
            fixture->reference_sum);
    reset_observation_storage(fixture);
}

static void bind_multi_request(struct visco_sh_exact_multi_shot_request *request,
        struct multi_fixture *fixture, int gradients, int nsrc) {
    memset(request, 0, sizeof(*request));
    request->wave = &fixture->storage.wave; request->pml = &fixture->storage.pml;
    request->material = &fixture->material; request->fwi = &fixture->storage.fwi;
    request->mpi = &fixture->storage.mpi; request->seismogram = &fixture->storage.seismogram;
    request->legacy_fwi_seismogram = &fixture->storage.legacy;
    request->acquisition = &fixture->storage.acquisition; request->hc = fixture->hc;
    request->iter = 1; request->nsrc = nsrc; request->ns = NT; request->nrec_local = 1;
    request->nrec_global = 1; request->hin = 1; request->dtinv_help = fixture->dtinv_help;
    if (gradients) { request->grad_primary = fixture->grad_primary.v;
        request->grad_rho = fixture->grad_rho.v; request->grad_q = fixture->grad_q.v; }
}

static int run_objective(struct multi_fixture *fixture,
        struct visco_sh_exact_multi_shot_result *result, int nsrc) {
    struct visco_sh_exact_multi_shot_request request;
    bind_multi_request(&request, fixture, 0, nsrc);
    return visco_sh_exact_objective(&request, result);
}

static int run_gradient(struct multi_fixture *fixture,
        struct visco_sh_exact_multi_shot_result *result, int nsrc) {
    struct visco_sh_exact_multi_shot_request request;
    bind_multi_request(&request, fixture, 1, nsrc);
    return visco_sh_exact_objective_gradient(&request, result);
}

static void reset_objective_observation(void) {
    objective_calls = 0; inject_failure_on_call = 0;
    memset(objective_cardinality, 0, sizeof(objective_cardinality));
    memset(objective_values, 0, sizeof(objective_values));
    inseis_calls = splitsrc_calls = gradient_shot_calls = material_vjp_calls = 0;
    memset(inseis_components, 0, sizeof(inseis_components));
}

static int preflight_is_transactional(struct multi_fixture *fixture) {
    struct visco_sh_exact_multi_shot_request request;
    struct visco_sh_exact_multi_shot_result result;
    struct acq acquisition;
    int saved;
    bind_multi_request(&request, fixture, 0, 2);
    result.objective = -444.0; result.shot_count = -77; forward_calls = 0;
    if (visco_sh_exact_objective(NULL, &result) == 0 || result.objective != -444.0 || result.shot_count != -77 || forward_calls != 0) return 0;
    if (visco_sh_exact_objective(&request, NULL) == 0) return 0;
#define BAD(statement) do { result.objective = -444.0; result.shot_count = -77; request = (struct visco_sh_exact_multi_shot_request){0}; bind_multi_request(&request, fixture, 0, 2); statement; forward_calls = 0; if (visco_sh_exact_objective(&request, &result) == 0 || result.objective != -444.0 || result.shot_count != -77 || forward_calls != 0) return 0; } while (0)
    BAD(request.nsrc = 0); BAD(request.ns = NT - 1); BAD(request.nrec_local = -1);
    BAD(request.acquisition->srcpos = NULL);
    request = (struct visco_sh_exact_multi_shot_request){0}; bind_multi_request(&request, fixture, 0, 2);
    acquisition = *request.acquisition; request.acquisition = &acquisition; acquisition.recpos = NULL;
    result.objective = -444.0; result.shot_count = -77; forward_calls = 0;
    if (visco_sh_exact_objective(&request, &result) == 0 || result.objective != -444.0 || result.shot_count != -77 || forward_calls != 0) return 0;
    saved = INVMAT1; INVMAT1 = 2; result.objective = -444.0; result.shot_count = -77; forward_calls = 0;
    if (visco_sh_exact_objective(&request, &result) == 0 || result.objective != -444.0 || result.shot_count != -77 || forward_calls != 0) return 0;
    INVMAT1 = saved;
#undef BAD
    return 1;
}

static void fill_owned(float **field, float value) {
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i) field[j][i] = value;
}

static int owned_is_value(float **field, float value) {
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i)
            if (field[j][i] != value) return 0;
    return 1;
}

static int owned_is_finite(float **field) {
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i)
            if (!isfinite(field[j][i])) return 0;
    return 1;
}

static double owned_signature(float **field) {
    int i, j;
    double sum = 0.0;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i)
            sum += field[j][i] * (double)(37 * j + i);
    return sum;
}

#if 0
int main(int argc, char **argv) {
    struct multi_fixture fixture;
    struct visco_sh_exact_multi_shot_result separate, simultaneous, gradient;
    int separate_adjoint, simultaneous_adjoint, gradient_separate_adjoint, gradient_simultaneous_adjoint;
    int preflight, intermediate, cleanup;
    double j1, j2, separate_gradient_rel, simultaneous_gradient_rel, simultaneous_sum_rel;
    MPI_Init(&argc, &argv); setup_fixture(&fixture);

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 1; reset_objective_observation();
    adjoint_calls = forward_calls = 0; memset(&separate, 0, sizeof(separate));
    if (run_objective(&fixture, &separate) != 0) die("separate objective");
    j1 = objective_values[0]; j2 = objective_values[1]; separate_adjoint = adjoint_calls;
    if (objective_calls != 2 || objective_cardinality[0] != 1 || objective_cardinality[1] != 1) die("separate source plan");

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 1; adjoint_calls = 0; memset(&gradient, 0, sizeof(gradient));
    if (run_gradient(&fixture, &gradient) != 0) die("separate gradient");
    separate_gradient_rel = fabs(separate.objective - gradient.objective) / fmax(fabs(gradient.objective), 1.0);
    gradient_separate_adjoint = adjoint_calls;

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 0; reset_objective_observation();
    adjoint_calls = forward_calls = 0; memset(&simultaneous, 0, sizeof(simultaneous));
    if (run_objective(&fixture, &simultaneous) != 0) die("simultaneous objective");
    simultaneous_adjoint = adjoint_calls;
    if (objective_calls != 1 || objective_cardinality[0] != 2) die("simultaneous source plan");

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 0; adjoint_calls = 0; memset(&gradient, 0, sizeof(gradient));
    if (run_gradient(&fixture, &gradient) != 0) die("simultaneous gradient");
    simultaneous_gradient_rel = fabs(simultaneous.objective - gradient.objective) / fmax(fabs(gradient.objective), 1.0);
    gradient_simultaneous_adjoint = adjoint_calls;
    simultaneous_sum_rel = fabs(simultaneous.objective - (j1 + j2)) / fmax(fabs(j1 + j2), 1.0e-300);

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 1; reset_objective_observation(); inject_failure_on_call = 2;
    { struct visco_sh_exact_multi_shot_result failed = {-991.0, -992};
      intermediate = run_objective(&fixture, &failed) != 0 && failed.objective == -991.0 && failed.shot_count == -992 &&
          fixture.storage.acquisition.signals == NULL && fixture.storage.acquisition.srcpos_loc == NULL; }
    inject_failure_on_call = 0;
    cleanup = fixture.storage.acquisition.signals == NULL && fixture.storage.acquisition.srcpos_loc == NULL;
    preflight = preflight_is_transactional(&fixture);

    printf("{\"separate_cardinalities\":[%d,%d],\"simultaneous_cardinalities\":[%d],"
            "\"separate_calls\":2,\"simultaneous_calls\":1,\"separate_shot_count\":%d,"
            "\"simultaneous_shot_count\":%d,\"j1\":%.17g,\"j2\":%.17g,"
            "\"j_separate\":%.17g,\"j_simultaneous\":%.17g,"
            "\"separate_gradient_relative_difference\":%.17g,"
            "\"simultaneous_gradient_relative_difference\":%.17g,"
            "\"simultaneous_independent_relative_difference\":%.17g,"
            "\"objective_adjoint_separate\":%d,\"objective_adjoint_simultaneous\":%d,"
            "\"gradient_adjoint_separate\":%d,\"gradient_adjoint_simultaneous\":%d,"
            "\"gradient_pointers_null\":true,\"preflight_transactional\":%s,"
            "\"intermediate_failure_transactional\":%s,\"resource_cleanup\":%s}\n",
            objective_cardinality[0], objective_cardinality[1], 2, separate.shot_count,
            simultaneous.shot_count, j1, j2, separate.objective, simultaneous.objective,
            separate_gradient_rel, simultaneous_gradient_rel, simultaneous_sum_rel,
            separate_adjoint, simultaneous_adjoint, gradient_separate_adjoint,
            gradient_simultaneous_adjoint, preflight ? "true" : "false",
            intermediate ? "true" : "false", cleanup ? "true" : "false");
    MPI_Finalize(); return 0;
}
#endif

int main(int argc, char **argv) {
    struct multi_fixture fixture;
    struct visco_sh_exact_multi_shot_result separate, separate_gradient;
    struct visco_sh_exact_multi_shot_result one_source, one_source_gradient;
    struct visco_sh_exact_multi_shot_result rejected_objective, rejected_gradient;
    int objective_rejection, gradient_rejection, objective_sentinel, gradient_sentinels;
    int rejected_inseis, rejected_splitsrc, rejected_objective_shot, rejected_gradient_shot, rejected_vjp;
    int separate_objective_calls, separate_gradient_calls, separate_vjp_calls;
    int separate_obs1, separate_obs2, separate_card1, separate_card2;
    int one_objective_calls, one_gradient_calls, one_vjp_calls, one_obs, one_card;
    int objective_adjoint, gradients_finite, repeatable, preflight, intermediate, cleanup, real_chain;
    double j1, j2, separate_rel, one_rel, signature_first, signature_second;

    MPI_Init(&argc, &argv); setup_fixture(&fixture);

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 0; reset_objective_observation();
    rejected_objective.objective = 321.125; rejected_objective.shot_count = 71;
    objective_rejection = run_objective(&fixture, &rejected_objective, 2);
    objective_sentinel = rejected_objective.objective == 321.125 && rejected_objective.shot_count == 71;
    rejected_inseis = inseis_calls; rejected_splitsrc = splitsrc_calls;
    rejected_objective_shot = objective_calls; rejected_gradient_shot = gradient_shot_calls;
    rejected_vjp = material_vjp_calls;

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 0; reset_objective_observation();
    fill_owned(fixture.grad_primary.v, 17.25f); fill_owned(fixture.grad_rho.v, -9.5f); fill_owned(fixture.grad_q.v, 3.75f);
    rejected_gradient.objective = -123.75; rejected_gradient.shot_count = 37;
    gradient_rejection = run_gradient(&fixture, &rejected_gradient, 2);
    gradient_sentinels = rejected_gradient.objective == -123.75 && rejected_gradient.shot_count == 37 &&
            owned_is_value(fixture.grad_primary.v, 17.25f) && owned_is_value(fixture.grad_rho.v, -9.5f) &&
            owned_is_value(fixture.grad_q.v, 3.75f) && inseis_calls == 0 && splitsrc_calls == 0 &&
            objective_calls == 0 && gradient_shot_calls == 0 && material_vjp_calls == 0;

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 1; reset_objective_observation();
    adjoint_calls = forward_calls = 0; memset(&separate, 0, sizeof(separate));
    if (run_objective(&fixture, &separate, 2) != 0) die("separate objective");
    j1 = objective_values[0]; j2 = objective_values[1];
    separate_objective_calls = objective_calls; separate_obs1 = inseis_components[0]; separate_obs2 = inseis_components[1];
    separate_card1 = objective_cardinality[0]; separate_card2 = objective_cardinality[1]; objective_adjoint = adjoint_calls;

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 1; reset_objective_observation();
    adjoint_calls = 0; memset(&separate_gradient, 0, sizeof(separate_gradient));
    if (run_gradient(&fixture, &separate_gradient, 2) != 0) die("separate gradient");
    separate_rel = fabs(separate.objective - separate_gradient.objective) / fmax(fabs(separate_gradient.objective), 1.0);
    separate_gradient_calls = gradient_shot_calls; separate_vjp_calls = material_vjp_calls;

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 0; reset_objective_observation();
    adjoint_calls = forward_calls = 0; memset(&one_source, 0, sizeof(one_source));
    if (run_objective(&fixture, &one_source, 1) != 0) die("one-source objective");
    one_objective_calls = objective_calls; one_obs = inseis_components[0]; one_card = objective_cardinality[0];

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 0; reset_objective_observation();
    adjoint_calls = 0; memset(&one_source_gradient, 0, sizeof(one_source_gradient));
    if (run_gradient(&fixture, &one_source_gradient, 1) != 0) die("one-source gradient");
    one_rel = fabs(one_source.objective - one_source_gradient.objective) / fmax(fabs(one_source_gradient.objective), 1.0);
    one_gradient_calls = gradient_shot_calls; one_vjp_calls = material_vjp_calls;
    gradients_finite = owned_is_finite(fixture.grad_primary.v) && owned_is_finite(fixture.grad_rho.v) && owned_is_finite(fixture.grad_q.v);
    signature_first = owned_signature(fixture.grad_primary.v) + owned_signature(fixture.grad_rho.v) + owned_signature(fixture.grad_q.v);

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 0; reset_objective_observation();
    memset(&one_source_gradient, 0, sizeof(one_source_gradient));
    if (run_gradient(&fixture, &one_source_gradient, 1) != 0) die("one-source repeat");
    signature_second = owned_signature(fixture.grad_primary.v) + owned_signature(fixture.grad_rho.v) + owned_signature(fixture.grad_q.v);
    repeatable = fabs(signature_first - signature_second) <= 1.0e-20;

    prepare_fixture_model(&fixture); RUN_MULTIPLE_SHOTS = 1; reset_objective_observation(); inject_failure_on_call = 2;
    { struct visco_sh_exact_multi_shot_result failed = {-991.0, -992};
      intermediate = run_objective(&fixture, &failed, 2) != 0 && failed.objective == -991.0 && failed.shot_count == -992 &&
          fixture.storage.acquisition.signals == NULL && fixture.storage.acquisition.srcpos_loc == NULL; }
    inject_failure_on_call = 0;
    cleanup = fixture.storage.acquisition.signals == NULL && fixture.storage.acquisition.srcpos_loc == NULL;
    preflight = preflight_is_transactional(&fixture);
    real_chain = separate_gradient_calls >= 2 && separate_vjp_calls >= 2 && one_gradient_calls >= 1 && one_vjp_calls >= 1 && adjoint_calls >= 1;

    printf("{\"rejected_objective_return\":%d,\"rejected_objective_sentinel_unchanged\":%s,"
            "\"rejected_gradient_return\":%d,\"rejected_gradient_sentinels_unchanged\":%s,"
            "\"rejection_inseis_calls\":%d,\"rejection_splitsrc_calls\":%d,"
            "\"rejection_objective_shot_calls\":%d,\"rejection_gradient_shot_calls\":%d,\"rejection_material_vjp_calls\":%d,"
            "\"separate_cardinalities\":[%d,%d],\"separate_observed_indices\":[%d,%d],"
            "\"separate_objective_shot_calls\":%d,\"separate_gradient_shot_calls\":%d,\"separate_shot_count\":%d,"
            "\"j1\":%.17g,\"j2\":%.17g,\"j_separate\":%.17g,\"separate_gradient_relative_difference\":%.17g,\"separate_material_vjp_calls\":%d,"
            "\"one_source_cardinalities\":[%d],\"one_source_observed_indices\":[%d],"
            "\"one_source_objective_shot_calls\":%d,\"one_source_gradient_shot_calls\":%d,\"one_source_shot_count\":%d,"
            "\"one_source_gradient_relative_difference\":%.17g,\"one_source_material_vjp_calls\":%d,"
            "\"objective_adjoint_calls\":%d,\"owned_gradients_finite\":%s,\"gradient_repeatable\":%s,"
            "\"real_production_gradient_chain\":%s,\"preflight_transactional\":%s,\"intermediate_failure_transactional\":%s,\"resource_cleanup\":%s}\n",
            objective_rejection, objective_sentinel ? "true" : "false", gradient_rejection, gradient_sentinels ? "true" : "false",
            rejected_inseis, rejected_splitsrc, rejected_objective_shot, rejected_gradient_shot, rejected_vjp,
            separate_card1, separate_card2, separate_obs1, separate_obs2, separate_objective_calls, separate_gradient_calls,
            separate.shot_count, j1, j2, separate.objective, separate_rel, separate_vjp_calls,
            one_card, one_obs, one_objective_calls, one_gradient_calls, one_source.shot_count, one_rel, one_vjp_calls,
            objective_adjoint, gradients_finite ? "true" : "false", repeatable ? "true" : "false",
            real_chain ? "true" : "false", preflight ? "true" : "false", intermediate ? "true" : "false", cleanup ? "true" : "false");
    MPI_Finalize(); return 0;
}
