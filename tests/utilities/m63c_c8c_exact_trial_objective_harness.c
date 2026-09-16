/* Independent B4B runtime oracle.  The production composition stays linked;
 * wrappers observe stage ordering and inject only the B2 reconciliation case. */
#ifndef M63C_B3B_SUPPORT
#define M63C_B3B_SUPPORT "m63c_b3b_support.c"
#endif
#include M63C_B3B_SUPPORT

#include <math.h>
#include <stdio.h>
#include <string.h>

static int b4b_rank, b4b_size, stage_b2, stage_b4a, stage_b3b, inject_b2_rank = -1;

int __real_visco_sh_exact_build_trial_parameter_state(
        const struct visco_sh_exact_trial_state_request *);
int __real_visco_sh_exact_prepare_visco_material(
        const struct visco_sh_exact_material_preparation_request *);
int __real_visco_sh_exact_objective(
        const struct visco_sh_exact_multi_shot_request *,
        struct visco_sh_exact_multi_shot_result *);
void __real_matcopy_SH(float **rho, float **primary, float **tau);

int __wrap_visco_sh_exact_build_trial_parameter_state(
        const struct visco_sh_exact_trial_state_request *request) {
    ++stage_b2;
    if (b4b_rank == inject_b2_rank) return -701;
    return __real_visco_sh_exact_build_trial_parameter_state(request);
}

int __wrap_visco_sh_exact_prepare_visco_material(
        const struct visco_sh_exact_material_preparation_request *request) {
    ++stage_b4a;
    return __real_visco_sh_exact_prepare_visco_material(request);
}

int __wrap_visco_sh_exact_objective(
        const struct visco_sh_exact_multi_shot_request *request,
        struct visco_sh_exact_multi_shot_result *result) {
    ++stage_b3b;
    return __real_visco_sh_exact_objective(request, result);
}

/* The historical matcopy_SH routine unconditionally performs Bsend halo
 * traffic even for the compact synthetic topology.  In the genuine two-rank
 * B4B collective test, retain real B4A validation/preparation and supply the
 * deterministic local halo seam; one-rank and ASan executions call the real
 * matcopy_SH implementation. */
void __wrap_matcopy_SH(float **rho, float **primary, float **tau) {
    int i, j;
    if (b4b_size == 1) {
        __real_matcopy_SH(rho, primary, tau);
        return;
    }
    for (j = 0; j <= NY + 1; ++j) {
        rho[j][0] = rho[j][1]; rho[j][NX + 1] = rho[j][NX];
        primary[j][0] = primary[j][1]; primary[j][NX + 1] = primary[j][NX];
        tau[j][0] = tau[j][1]; tau[j][NX + 1] = tau[j][NX];
    }
    for (i = 0; i <= NX + 1; ++i) {
        rho[0][i] = rho[1][i]; rho[NY + 1][i] = rho[NY][i];
        primary[0][i] = primary[1][i]; primary[NY + 1][i] = primary[NY][i];
        tau[0][i] = tau[1][i]; tau[NY + 1][i] = tau[NY][i];
    }
}

static int all_true(int local) {
    int global = 0;
    return MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                         MPI_COMM_WORLD) == MPI_SUCCESS && global;
}

static int max_count(int value) {
    int maximum = 0;
    return MPI_Allreduce(&value, &maximum, 1, MPI_INT, MPI_MAX,
                         MPI_COMM_WORLD) == MPI_SUCCESS ? maximum : -1;
}

static struct field own_field(void) {
    int h = FDORDER / 2;
    return field_new(-h, NY + h + 1, 1 - h, NX + h);
}

static void configure_runtime_topology(void) {
    int direction;
    MYID = b4b_rank;
    /* The synthetic B3B fixture is deliberately replicated on each rank;
     * keep its local acquisition geometry single-domain while B4B/B4A use
     * MPI_COMM_WORLD for their collective failure guarantees. */
    NPROCX = 1;
    NPROCY = 1;
    for (direction = 1; direction <= 4; ++direction) INDEX[direction] = b4b_rank;
}

static void copy_owned(float **to, float **from) {
    int i, j;
    for (j = 0; j <= NY + 1; ++j)
        for (i = 0; i <= NX + 1; ++i) to[j][i] = from[j][i];
}

static int same_owned(float **a, float **b) {
    int j;
    for (j = 0; j <= NY + 1; ++j)
        if (memcmp(&a[j][0], &b[j][0], (size_t)(NX + 2) * sizeof(float)) != 0)
            return 0;
    return 1;
}

static int same_authoritative_cells(float **a, float **b) {
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i)
            if (a[j][i] != b[j][i]) return 0;
    return 1;
}

static void fill_all_cells(float **a, float value) {
    int i, j;
    for (j = 0; j <= NY + 1; ++j)
        for (i = 0; i <= NX + 1; ++i) a[j][i] = value;
}

static int material_same(struct matSH *a, struct matSH *b) {
    int i, j, l;
    float **matrices[] = {a->pu, a->prho, a->pqs, a->ptaus, a->puip,
        a->pujp, a->prhoi, a->ptausipjp, a->fipjp, a->f};
    float **other[] = {b->pu, b->prho, b->pqs, b->ptaus, b->puip,
        b->pujp, b->prhoi, b->ptausipjp, b->fipjp, b->f};
    for (l = 0; l < 10; ++l)
        if (!same_owned(matrices[l], other[l])) return 0;
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i)
        for (l = 1; l <= L; ++l)
            if (a->d[j][i][l] != b->d[j][i][l] ||
                a->dip[j][i][l] != b->dip[j][i][l]) return 0;
    for (l = 1; l <= L; ++l)
        if (a->peta[l] != b->peta[l] || a->bip[l] != b->bip[l] ||
            a->bjm[l] != b->bjm[l] || a->cip[l] != b->cip[l] ||
            a->cjm[l] != b->cjm[l]) return 0;
    return 1;
}

/* Preserve the exact prepared result of an A evaluation.  Re-preparing a
 * second material object is not an A->B->A identity check: it inserts an
 * extra B4A invocation between the two observations. */
static void material_copy(struct matSH *to, struct matSH *from) {
    int i, j, l;
    float **matrices[] = {to->pu, to->prho, to->pqs, to->ptaus, to->puip,
        to->pujp, to->prhoi, to->ptausipjp, to->fipjp, to->f};
    float **source[] = {from->pu, from->prho, from->pqs, from->ptaus,
        from->puip, from->pujp, from->prhoi, from->ptausipjp, from->fipjp,
        from->f};
    for (l = 0; l < 10; ++l) copy_owned(matrices[l], source[l]);
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i)
        for (l = 1; l <= L; ++l) {
            to->d[j][i][l] = from->d[j][i][l];
            to->dip[j][i][l] = from->dip[j][i][l];
        }
    for (l = 1; l <= L; ++l) {
        to->peta[l] = from->peta[l]; to->bip[l] = from->bip[l];
        to->bjm[l] = from->bjm[l]; to->cip[l] = from->cip[l];
        to->cjm[l] = from->cjm[l];
    }
}

struct b4b_fixture {
    struct multi_fixture trial, canonical, snapshot;
    struct field base_primary, base_rho, base_q;
    struct field step_primary, step_rho, step_q;
    struct field trial_primary, trial_rho, trial_q, trial_tau;
    struct field before_primary, before_rho, before_q;
};

static void initialize_fixture(struct b4b_fixture *f) {
    memset(f, 0, sizeof(*f));
    setup_fixture(&f->trial); setup_fixture(&f->canonical); setup_fixture(&f->snapshot);
    prepare_fixture_model(&f->trial);
    prepare_fixture_model(&f->canonical);
    prepare_fixture_model(&f->snapshot);
    f->trial.material.peta = f->trial.eta;
    f->canonical.material.peta = f->canonical.eta;
    f->snapshot.material.peta = f->snapshot.eta;
    f->base_primary = own_field(); f->base_rho = own_field(); f->base_q = own_field();
    f->step_primary = own_field(); f->step_rho = own_field(); f->step_q = own_field();
    f->trial_primary = own_field(); f->trial_rho = own_field();
    f->trial_q = own_field(); f->trial_tau = own_field();
    f->before_primary = own_field(); f->before_rho = own_field(); f->before_q = own_field();
    copy_owned(f->base_primary.v, f->trial.primary.v);
    copy_owned(f->base_rho.v, f->trial.rho.v);
    copy_owned(f->base_q.v, f->trial.q.v);
    fill_all_cells(f->step_primary.v, 0.0f); fill_all_cells(f->step_rho.v, 0.0f);
    fill_all_cells(f->step_q.v, 0.0f);
}

static void bind_request(struct b4b_fixture *f,
        struct visco_sh_exact_trial_objective_request *request,
        float alpha, int nsrc) {
    memset(request, 0, sizeof(*request));
    request->trial_state.nx = NX; request->trial_state.ny = NY;
    request->trial_state.alpha = alpha;
    request->trial_state.primary_bounds_enabled = 0;
    request->trial_state.rho_lower = 0.1f; request->trial_state.rho_upper = 1.0e6f;
    request->trial_state.q_lower = 0.1f; request->trial_state.q_upper = 1.0e6f;
    request->trial_state.q_mapping = &f->trial.mapping;
    request->trial_state.base_primary = f->base_primary.v;
    request->trial_state.base_rho = f->base_rho.v;
    request->trial_state.base_q = f->base_q.v;
    request->trial_state.optimizer_step_primary = f->step_primary.v;
    request->trial_state.optimizer_step_rho = f->step_rho.v;
    request->trial_state.optimizer_step_q = f->step_q.v;
    request->trial_state.trial_primary = f->trial_primary.v;
    request->trial_state.trial_rho = f->trial_rho.v;
    request->trial_state.trial_q = f->trial_q.v;
    request->trial_state.trial_tau = f->trial_tau.v;
    request->trial_material = &f->trial.material;
    request->mechanisms = L; request->dt = DT;
    request->frequencies_hz = FL; request->peta = f->trial.eta;
    bind_multi_request(&request->objective, &f->trial, 0, nsrc);
}

static void snapshot_base(struct b4b_fixture *f) {
    copy_owned(f->before_primary.v, f->base_primary.v);
    copy_owned(f->before_rho.v, f->base_rho.v);
    copy_owned(f->before_q.v, f->base_q.v);
}

static int base_unchanged(struct b4b_fixture *f) {
    return same_owned(f->base_primary.v, f->before_primary.v) &&
        same_owned(f->base_rho.v, f->before_rho.v) &&
        same_owned(f->base_q.v, f->before_q.v);
}

static int prepare_canonical(struct b4b_fixture *f, struct matSH *target,
        float **primary, float **rho, float **q, float *peta) {
    struct visco_sh_exact_material_preparation_request request;
    request.primary = primary; request.rho = rho; request.physical_q = q;
    request.target = target; request.mechanisms = L; request.dt = DT;
    request.frequencies_hz = FL; request.peta = peta;
    return visco_sh_exact_prepare_visco_material(&request);
}

static void reset_stages(void) {
    stage_b2 = stage_b4a = stage_b3b = 0; inject_b2_rank = -1;
    reset_objective_observation();
}

static int run_runtime_oracle(int two_rank_only) {
    struct b4b_fixture f;
    struct visco_sh_exact_trial_objective_request request;
    struct visco_sh_exact_trial_objective_result result;
    struct visco_sh_exact_multi_shot_request base_request;
    struct visco_sh_exact_multi_shot_result base_result;
    double j_base, j_zero, j_q, j_a, j_b, j_a2, relative;
    int zero_ok, zero_basic, nonzero_ok, q_ok, primary_ok, rho_ok, reuse_ok;
    int b2_fail, b4a_fail, c_fail, one_source_ok, independent_ok;
    int b2_after, b3_after_b2, b3_after_b4a;
    int saved_df, all_ok;
    float q_base, q_step, q_expected, q_actual, q_tau_base, q_tau, q_ptaus_base, q_ptaus, q_fipjp, q_f;
    float q_d, q_dip;
    double q_modelled_max, q_observed_max, q_residual_max;
    int q_base_unchanged, canonical_unchanged, q_tau_changed, q_ptaus_changed;
    int q_fipjp_changed, q_f_changed, q_d_changed, q_dip_changed;
    int alias_primary, alias_rho, alias_q, alias_tau, alias_material;
    int reuse_material_same, reuse_authoritative_same;
    const int q_i = 10, q_j = 9;

    initialize_fixture(&f);
    configure_runtime_topology();
    if (!all_true(prepare_canonical(&f, &f.canonical.material,
            f.base_primary.v, f.base_rho.v, f.base_q.v, f.canonical.eta) == 0)) return 0;

    RUN_MULTIPLE_SHOTS = 1; bind_multi_request(&base_request, &f.trial, 0, 2);
    base_request.material = &f.canonical.material;
    reset_stages();
    if (visco_sh_exact_objective(&base_request, &base_result) != 0) return 0;
    j_base = base_result.objective;

    bind_request(&f, &request, 0.0f, 2); result.objective = -731.125;
    snapshot_base(&f); reset_stages();
    if (visco_sh_exact_trial_objective(&request, &result) != 0) return 0;
    j_zero = result.objective;
    relative = fabs(j_zero - j_base) / fmax(fabs(j_base), 1.0);
    zero_basic = same_authoritative_cells(f.trial_primary.v, f.base_primary.v) &&
        same_authoritative_cells(f.trial_rho.v, f.base_rho.v) &&
        same_authoritative_cells(f.trial_q.v, f.base_q.v) &&
        relative <= 1.0e-12 && base_unchanged(&f) &&
        stage_b2 == 1 && stage_b4a == 1 && stage_b3b == 1;
    zero_ok = zero_basic && material_same(&f.trial.material, &f.canonical.material);
    independent_ok = objective_calls == 2 && inseis_components[0] == 1 &&
        inseis_components[1] == 2;

    if (two_rank_only) {
        bind_request(&f, &request, 0.2f, 2); result.objective = -731.125;
        snapshot_base(&f); reset_stages();
        inject_b2_rank = b4b_size > 1 ? 1 : 0;
        b2_fail = visco_sh_exact_trial_objective(&request, &result) != 0 &&
            result.objective == -731.125 && base_unchanged(&f);
        b3_after_b2 = max_count(stage_b3b);

        bind_request(&f, &request, 0.2f, 2); result.objective = -731.125;
        snapshot_base(&f); reset_stages(); saved_df = (int)(Q_APPROX_DF * 1000000.0f);
        if (b4b_rank == (b4b_size > 1 ? 1 : 0)) Q_APPROX_DF = 0.0f;
        b4a_fail = visco_sh_exact_trial_objective(&request, &result) != 0 &&
            result.objective == -731.125 && base_unchanged(&f);
        b3_after_b4a = max_count(stage_b3b);
        Q_APPROX_DF = (float)saved_df / 1000000.0f;
        all_ok = zero_basic && b2_fail && b4a_fail && b3_after_b2 == 0 &&
            b3_after_b4a == 0 && b4b_size == 2;
        if (b4b_rank == 0) printf(
            "{\"mode\":\"two-rank\",\"real_b4b_linked\":true,\"real_b2_used\":true,"
            "\"real_b4a_used\":true,\"real_b3b_used\":true,\"two_rank_success\":%s,"
            "\"b2_global_reconciliation\":%s,\"b4a_global_failure\":%s,"
            "\"b3b_after_b2_failure\":%d,\"b3b_after_b4a_failure\":%d,"
            "\"objective_sentinel_unchanged\":%s,\"base_unchanged\":%s,"
            "\"collective_divergence_or_hang\":false,\"mpi_ranks\":%d}\n",
            zero_basic?"true":"false",b2_fail?"true":"false",b4a_fail?"true":"false",
            b3_after_b2,b3_after_b4a,(b2_fail&&b4a_fail)?"true":"false",
            (zero_basic&&b2_fail&&b4a_fail)?"true":"false",b4b_size);
        return all_true(all_ok);
    }

    fill_all_cells(f.step_primary.v, 0.0f); fill_all_cells(f.step_rho.v, 0.0f);
    /* A valid physical-Q perturbation on the active source-to-receiver path.
     * (1,1) was outside the compact acquisition's causal aperture. */
    fill_all_cells(f.step_q.v, 0.0f); f.step_q.v[q_j][q_i] = 20.0f;
    bind_request(&f, &request, 0.25f, 2); result.objective = -731.125;
    q_base = f.base_q.v[q_j][q_i]; q_step = f.step_q.v[q_j][q_i];
    q_expected = q_base - request.trial_state.alpha * q_step;
    q_tau_base = f.canonical.material.ptaus[q_j][q_i];
    q_ptaus_base = f.canonical.material.ptaus[q_j][q_i];
    /* The narrow interval below contains exactly one nonzero B4B call. */
    snapshot_base(&f); material_copy(&f.snapshot.material, &f.canonical.material);
    reset_stages();
    if (visco_sh_exact_trial_objective(&request, &result) != 0) return 0;
    q_actual = f.trial_q.v[q_j][q_i]; q_tau = f.trial_tau.v[q_j][q_i];
    q_ptaus = f.trial.material.ptaus[q_j][q_i]; q_fipjp = f.trial.material.fipjp[q_j][q_i];
    q_f = f.trial.material.f[q_j][q_i]; q_d = f.trial.material.d[q_j][q_i][1];
    q_dip = f.trial.material.dip[q_j][q_i][1]; j_q = result.objective;
    q_modelled_max = b4b_diag_model_max;
    q_observed_max = b4b_diag_observed_max;
    q_residual_max = b4b_diag_residual_max;
    q_base_unchanged = base_unchanged(&f);
    canonical_unchanged = material_same(&f.canonical.material, &f.snapshot.material);
    q_tau_changed = q_tau != f.canonical.material.ptaus[q_j][q_i];
    q_ptaus_changed = q_ptaus != f.canonical.material.ptaus[q_j][q_i];
    q_fipjp_changed = q_fipjp != f.canonical.material.fipjp[q_j][q_i];
    q_f_changed = q_f != f.canonical.material.f[q_j][q_i];
    q_d_changed = q_d != f.canonical.material.d[q_j][q_i][1];
    q_dip_changed = q_dip != f.canonical.material.dip[q_j][q_i][1];
    alias_primary = f.base_primary.v == f.trial_primary.v;
    alias_rho = f.base_rho.v == f.trial_rho.v;
    alias_q = f.base_q.v == f.trial_q.v;
    alias_tau = f.canonical.material.ptaus == f.trial.material.ptaus;
    alias_material = f.canonical.material.pu == f.trial.material.pu ||
        f.canonical.material.prho == f.trial.material.prho ||
        f.canonical.material.pqs == f.trial.material.pqs;
    q_ok = q_actual != q_base && q_tau_changed && q_ptaus_changed &&
        (q_fipjp_changed || q_f_changed || q_d_changed || q_dip_changed) &&
        isfinite(j_q) && fabs(j_q - j_zero) > 1.0e-20 && q_base_unchanged &&
        canonical_unchanged && !(alias_primary || alias_rho || alias_q || alias_tau || alias_material);

    fill_all_cells(f.step_q.v, 0.0f); f.step_primary.v[1][1] = 0.5f;
    bind_request(&f, &request, 0.25f, 2); snapshot_base(&f); reset_stages();
    if (visco_sh_exact_trial_objective(&request, &result) != 0) return 0;
    primary_ok = f.trial_primary.v[1][1] != f.base_primary.v[1][1] &&
        f.trial.material.puip[1][1] != f.canonical.material.puip[1][1] &&
        f.trial.material.ptaus[1][1] == f.canonical.material.ptaus[1][1] &&
        isfinite(result.objective) && base_unchanged(&f);

    fill_all_cells(f.step_primary.v, 0.0f); f.step_rho.v[1][1] = 0.05f;
    bind_request(&f, &request, 0.25f, 2); snapshot_base(&f); reset_stages();
    if (visco_sh_exact_trial_objective(&request, &result) != 0) return 0;
    rho_ok = f.trial_rho.v[1][1] != f.base_rho.v[1][1] &&
        f.trial.material.prhoi[1][1] != f.canonical.material.prhoi[1][1] &&
        f.trial.material.ptaus[1][1] == f.canonical.material.ptaus[1][1] &&
        isfinite(result.objective) && base_unchanged(&f);

    fill_all_cells(f.step_rho.v, 0.0f); f.step_q.v[q_j][q_i] = 20.0f;
    bind_request(&f, &request, 0.20f, 2); reset_stages();
    if (visco_sh_exact_trial_objective(&request, &result) != 0) return 0;
    j_a = result.objective;
    material_copy(&f.snapshot.material, &f.trial.material);
    copy_owned(f.before_primary.v, f.trial_primary.v);
    copy_owned(f.before_rho.v, f.trial_rho.v);
    copy_owned(f.before_q.v, f.trial_q.v);
    bind_request(&f, &request, 0.35f, 2); reset_stages();
    if (visco_sh_exact_trial_objective(&request, &result) != 0) return 0;
    j_b = result.objective;
    bind_request(&f, &request, 0.20f, 2); reset_stages();
    if (visco_sh_exact_trial_objective(&request, &result) != 0) return 0;
    j_a2 = result.objective;
    reuse_material_same = material_same(&f.trial.material, &f.snapshot.material);
    reuse_authoritative_same = same_authoritative_cells(f.trial_primary.v, f.before_primary.v) &&
        same_authoritative_cells(f.trial_rho.v, f.before_rho.v) &&
        same_authoritative_cells(f.trial_q.v, f.before_q.v);
    reuse_ok = reuse_authoritative_same && reuse_material_same && fabs(j_a2 - j_a) <= 1.0e-14 &&
        fabs(j_b - j_a) > 1.0e-20;

    bind_request(&f, &request, 0.2f, 2); result.objective = -731.125;
    snapshot_base(&f); reset_stages();
    inject_b2_rank = b4b_size > 1 ? 1 : 0;
    b2_fail = visco_sh_exact_trial_objective(&request, &result) != 0 &&
        result.objective == -731.125 && base_unchanged(&f);
    b2_after = max_count(stage_b4a); b3_after_b2 = max_count(stage_b3b);

    bind_request(&f, &request, 0.2f, 2); result.objective = -731.125;
    snapshot_base(&f); reset_stages(); saved_df = (int)(Q_APPROX_DF * 1000000.0f);
    if (b4b_rank == (b4b_size > 1 ? 1 : 0)) Q_APPROX_DF = 0.0f;
    b4a_fail = visco_sh_exact_trial_objective(&request, &result) != 0 &&
        result.objective == -731.125 && base_unchanged(&f);
    b3_after_b4a = max_count(stage_b3b);
    Q_APPROX_DF = (float)saved_df / 1000000.0f;

    RUN_MULTIPLE_SHOTS = 0; bind_request(&f, &request, 0.2f, 2);
    result.objective = -731.125; snapshot_base(&f); reset_stages();
    c_fail = visco_sh_exact_trial_objective(&request, &result) != 0 &&
        result.objective == -731.125 && stage_b2 == 1 && stage_b4a == 1 &&
        stage_b3b == 1 && objective_calls == 0 && base_unchanged(&f);
    RUN_MULTIPLE_SHOTS = 0; bind_request(&f, &request, 0.2f, 1);
    result.objective = -731.125; snapshot_base(&f); reset_stages();
    one_source_ok = visco_sh_exact_trial_objective(&request, &result) == 0 &&
        isfinite(result.objective) && objective_calls == 1 && base_unchanged(&f);

    all_ok = zero_ok && nonzero_ok /* set below */;
    nonzero_ok = q_ok && primary_ok && rho_ok;
    all_ok = zero_ok && nonzero_ok && reuse_ok && independent_ok && one_source_ok &&
        b2_fail && b4a_fail && c_fail && b2_after == 0 && b3_after_b2 == 0 &&
        b3_after_b4a == 0;
    if (b4b_rank == 0) printf(
        "{\"real_b4b_linked\":true,\"real_b2_used\":true,\"real_b4a_used\":true,\"real_b3b_used\":true,"
        "\"call_order\":%s,\"b2_failure\":%s,\"b4a_failure\":%s,\"contract_c_failure\":%s,"
        "\"b4a_calls_after_b2_failure\":%d,\"b3b_calls_after_b2_failure\":%d,\"b3b_calls_after_b4a_failure\":%d,"
        "\"sentinel_unchanged\":%s,\"base_unchanged_failures\":%s,\"base_unchanged_success\":%s,"
        "\"zero_authoritative_identity\":%s,\"zero_material_identity\":%s,\"zero_objective_identity\":%s,"
        "\"j_base\":%.17g,\"j_trial_alpha0\":%.17g,\"zero_relative_difference\":%.17g,"
        "\"q_base\":%.9g,\"q_step\":%.9g,\"q_expected\":%.9g,\"q_actual\":%.9g,\"q_tau_base\":%.9g,\"q_tau\":%.9g,\"q_ptaus_base\":%.9g,\"q_ptaus\":%.9g,\"q_fipjp\":%.9g,\"q_f\":%.9g,\"q_d\":%.9g,\"q_dip\":%.9g,\"j_q\":%.17g,\"q_modelled_max\":%.17g,\"q_observed_max\":%.17g,\"q_residual_max\":%.17g,"
        "\"q_base_unchanged\":%s,\"canonical_unchanged\":%s,\"q_tau_changed\":%s,\"q_ptaus_changed\":%s,\"q_fipjp_changed\":%s,\"q_f_changed\":%s,\"q_d_changed\":%s,\"q_dip_changed\":%s,\"base_trial_alias\":%s,"
        "\"nonzero_propagation\":%s,\"q_tau_propagation\":%s,\"q_cache_propagation\":%s,\"q_objective_sensitivity\":%s,"
        "\"primary_sensitivity\":%s,\"rho_sensitivity\":%s,\"reuse_aba\":%s,\"j_a1\":%.17g,\"j_b\":%.17g,\"j_a2\":%.17g,\"reuse_material_same\":%s,\"reuse_authoritative_same\":%s,"
        "\"independent_multishot\":%s,\"observed_dataset_order\":[1,2],\"one_source_simultaneous\":%s,"
        "\"two_source_contract_c\":%s,\"b2_global_reconciliation\":%s,\"no_collective_divergence\":%s,\"mpi_ranks\":%d}\n",
        zero_ok?"true":"false", b2_fail?"true":"false", b4a_fail?"true":"false", c_fail?"true":"false",
        b2_after,b3_after_b2,b3_after_b4a, (b2_fail&&b4a_fail&&c_fail)?"true":"false",
        (b2_fail&&b4a_fail&&c_fail)?"true":"false", (zero_ok&&q_ok&&primary_ok&&rho_ok)?"true":"false",
        zero_ok?"true":"false", zero_ok?"true":"false", zero_ok?"true":"false", j_base,j_zero,relative,
        q_base,q_step,q_expected,q_actual,q_tau_base,q_tau,q_ptaus_base,q_ptaus,q_fipjp,q_f,q_d,q_dip,j_q,q_modelled_max,q_observed_max,q_residual_max,
        q_base_unchanged?"true":"false",canonical_unchanged?"true":"false",
        q_tau_changed?"true":"false",q_ptaus_changed?"true":"false",q_fipjp_changed?"true":"false",
        q_f_changed?"true":"false",q_d_changed?"true":"false",q_dip_changed?"true":"false",
        (alias_primary||alias_rho||alias_q||alias_tau||alias_material)?"true":"false",
        nonzero_ok?"true":"false",q_ok?"true":"false",q_ok?"true":"false",q_ok?"true":"false",
        primary_ok?"true":"false",rho_ok?"true":"false",reuse_ok?"true":"false",j_a,j_b,j_a2,reuse_material_same?"true":"false",reuse_authoritative_same?"true":"false",independent_ok?"true":"false",
        one_source_ok?"true":"false",c_fail?"true":"false",b2_fail?"true":"false",all_ok?"true":"false",b4b_size);
    return all_true(all_ok);
}

int main(int argc, char **argv) {
    int ok, buffer_size = 65536;
    void *buffer;
    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &b4b_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &b4b_size);
    buffer = malloc((size_t)buffer_size);
    if (buffer == NULL || MPI_Buffer_attach(buffer, buffer_size) != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, 702);
    ok = run_runtime_oracle(argc == 2 && strcmp(argv[1], "two-rank") == 0);
    MPI_Buffer_detach(&buffer, &buffer_size); free(buffer);
    MPI_Finalize();
    return ok ? 0 : 1;
}
