/* Independent B5A runtime oracle.  It embeds the B4B fixture only for its
 * compact real-production experiment; B5A and B4B remain separately linked
 * production symbols. */
#ifndef M63C_B4B_SUPPORT
#define M63C_B4B_SUPPORT "m63c_c8c_exact_trial_objective_harness.c"
#endif

#define main m63c_b4b_embedded_main
#include M63C_B4B_SUPPORT
#undef main

#include <float.h>

int __real_visco_sh_exact_trial_objective(
        const struct visco_sh_exact_trial_objective_request *,
        struct visco_sh_exact_trial_objective_result *);

static int line_rank, line_size, trial_calls, legacy_calls;
static float trial_alpha[16], trial_q[16], trial_tau[16], trial_ptaus[16];
static double trial_objective[16];
static const int q_i = 10, q_j = 9;

/* These wrappers turn an accidental legacy route into an observable failure.
 * The real B4B composition is still reached through __real below. */
float __wrap_calc_mat_change_test_SH_visc(void) { ++legacy_calls; return -1.0f; }
void __wrap_obj_sh(void) { ++legacy_calls; }
double __wrap_grad_obj_sh(void) { ++legacy_calls; return NAN; }
void __wrap_ass_gradSH_visc(void) { ++legacy_calls; }

int __wrap_visco_sh_exact_trial_objective(
        const struct visco_sh_exact_trial_objective_request *request,
        struct visco_sh_exact_trial_objective_result *result) {
    int slot = trial_calls++;
    int status = __real_visco_sh_exact_trial_objective(request, result);
    if (slot < 16) {
        trial_alpha[slot] = request->trial_state.alpha;
        trial_objective[slot] = status == 0 ? result->objective : NAN;
        if (status == 0) {
            trial_q[slot] = request->trial_state.trial_q[q_j][q_i];
            trial_tau[slot] = request->trial_state.trial_tau[q_j][q_i];
            trial_ptaus[slot] = request->trial_material->ptaus[q_j][q_i];
        }
    }
    return status;
}

static int all_true_line(int local) {
    int global = 0;
    return MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                         MPI_COMM_WORLD) == MPI_SUCCESS && global;
}

static int same_float_line(float value) {
    float lo, hi;
    return MPI_Allreduce(&value, &lo, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS &&
        MPI_Allreduce(&value, &hi, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS && lo == hi;
}

static int same_double_line(double value) {
    double lo, hi;
    return MPI_Allreduce(&value, &lo, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS &&
        MPI_Allreduce(&value, &hi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS && lo == hi;
}

static int same_int_line(int value) {
    int lo, hi;
    return MPI_Allreduce(&value, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS &&
        MPI_Allreduce(&value, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD) == MPI_SUCCESS && lo == hi;
}

static int same_alpha(float left, float right) {
    float scale = fmaxf(1.0f, fmaxf(fabsf(left), fabsf(right)));
    return fabsf(left - right) <= 4.0f * FLT_EPSILON * scale;
}

static void reset_line_observation(void) {
    int i;
    trial_calls = legacy_calls = 0;
    for (i = 0; i < 16; ++i) {
        trial_alpha[i] = trial_q[i] = trial_tau[i] = trial_ptaus[i] = NAN;
        trial_objective[i] = NAN;
    }
    reset_stages();
}

static int run_success(void) {
    struct b4b_fixture f;
    struct visco_sh_exact_line_search_request request;
    struct visco_sh_exact_line_search_result result;
    struct visco_sh_exact_multi_shot_request base_request;
    struct visco_sh_exact_multi_shot_result base_result;
    double policy_values[4], selected_objective_expected;
    float policy_alphas[4], policy_selected;
    int i, distinct_alpha = 0, distinct_state = 0, q_propagates;
    int base_ok, alias, scalar_ok, trial_reuse, sign_ok, mpi_ok;

    initialize_fixture(&f);
    configure_runtime_topology();
    if (!all_true_line(prepare_canonical(&f, &f.canonical.material,
            f.base_primary.v, f.base_rho.v, f.base_q.v, f.canonical.eta) == 0)) return 0;

    RUN_MULTIPLE_SHOTS = 1;
    bind_multi_request(&base_request, &f.trial, 0, 2);
    base_request.material = &f.canonical.material;
    if (visco_sh_exact_objective(&base_request, &base_result) != 0) return 0;

    fill_all_cells(f.step_primary.v, 0.0f);
    fill_all_cells(f.step_rho.v, 0.0f);
    fill_all_cells(f.step_q.v, 0.0f);
    /* This owned cell lies on the compact fixture's effective aperture. */
    f.step_q.v[q_j][q_i] = 20.0f;
    memset(&request, 0, sizeof(request));
    bind_request(&f, &request.trial_objective, 0.0f, 2);
    request.base_objective = base_result.objective;
    request.initial_alpha = 0.2f;
    request.scale_factor = 2.0f;
    request.max_retries = 3;

    snapshot_base(&f);
    material_copy(&f.snapshot.material, &f.canonical.material);
    result.selected_alpha = -91.25f;
    result.selected_objective = -92.25;
    result.candidate_count = -93;
    result.bracketed = -94;
    reset_line_observation();
    if (step_length_est_sh_visc_exact(&request, &result) != 0) return 0;

    for (i = 1; i < trial_calls; ++i) {
        if (trial_alpha[i] != trial_alpha[0]) distinct_alpha = 1;
        if (trial_q[i] != trial_q[0]) distinct_state = 1;
    }
    sign_ok = trial_calls >= 2;
    for (i = 0; i < trial_calls; ++i) {
        float expected = f.base_q.v[q_j][q_i] - trial_alpha[i] * f.step_q.v[q_j][q_i];
        sign_ok = sign_ok && isfinite(trial_objective[i]) && trial_q[i] == expected;
    }
    q_propagates = trial_calls > 0 && trial_q[0] != f.base_q.v[q_j][q_i] &&
        trial_tau[0] != f.canonical.material.ptaus[q_j][q_i] &&
        trial_ptaus[0] != f.canonical.material.ptaus[q_j][q_i] && isfinite(trial_objective[0]);
    base_ok = base_unchanged(&f) && material_same(&f.canonical.material, &f.snapshot.material);
    alias = f.base_primary.v == f.trial_primary.v || f.base_rho.v == f.trial_rho.v ||
        f.base_q.v == f.trial_q.v || f.canonical.material.pu == f.trial.material.pu ||
        f.canonical.material.pqs == f.trial.material.pqs ||
        f.canonical.material.ptaus == f.trial.material.ptaus;
    trial_reuse = distinct_alpha && distinct_state && sign_ok;

    /* The helper retains its final growth candidate in objectives[3], whether
     * that candidate brackets a minimum or reaches the retry limit. */
    policy_values[1] = request.base_objective;
    policy_values[2] = trial_objective[0];
    policy_values[3] = trial_objective[trial_calls - 1];
    policy_alphas[1] = 0.0f;
    policy_alphas[2] = trial_alpha[0];
    policy_alphas[3] = trial_alpha[trial_calls - 1];
    policy_selected = calc_opt_step(policy_values, policy_alphas, 1);
    selected_objective_expected = same_alpha(policy_selected, policy_alphas[1]) ? policy_values[1] :
        (same_alpha(policy_selected, policy_alphas[2]) ? policy_values[2] :
         (same_alpha(policy_selected, policy_alphas[3]) ? policy_values[3] : NAN));
    scalar_ok = trial_calls >= 2 && trial_objective[0] < request.base_objective &&
        same_alpha(result.selected_alpha, policy_selected) && isfinite(result.selected_objective) &&
        (isfinite(selected_objective_expected) ? result.selected_objective == selected_objective_expected :
         (trial_calls >= 3 && result.selected_objective == trial_objective[trial_calls - 1]));
    mpi_ok = same_int_line(trial_calls) && same_float_line(result.selected_alpha) &&
        same_double_line(result.selected_objective);
    for (i = 0; i < trial_calls; ++i)
        mpi_ok = mpi_ok && same_float_line(trial_alpha[i]) && same_double_line(trial_objective[i]);

    if (line_rank == 0) {
        printf("{\"mode\":\"success\",\"real_b5a_linked\":true,\"real_b4b_used\":true,"
               "\"legacy_path_used\":%s,\"candidate_count\":%d,\"b4b_calls\":%d,\"candidate_alphas\":[",
               legacy_calls ? "true" : "false", result.candidate_count, trial_calls);
        for (i = 0; i < trial_calls; ++i) printf("%s%.9g", i ? "," : "", trial_alpha[i]);
        printf("],\"candidate_objectives\":[");
        for (i = 0; i < trial_calls; ++i) printf("%s%.17g", i ? "," : "", trial_objective[i]);
        printf(
        "],\"alpha0\":%.9g,\"objective0\":%.17g,\"alpha1\":%.9g,\"objective1\":%.17g,"
        "\"scalar_policy_result\":%.9g,\"selected_alpha\":%.9g,\"scalar_policy_consistent\":%s,"
        "\"base_unchanged\":%s,\"base_trial_alias\":%s,\"trial_reuse\":%s,"
        "\"physical_q_propagation\":%s,\"subtractive_sign\":%s,\"base_model_committed\":false,"
        "\"mpi_ranks\":%d,\"mpi_identical\":%s}\n",
        trial_alpha[0], trial_objective[0], trial_alpha[1], trial_objective[1],
        policy_selected, result.selected_alpha, scalar_ok ? "true" : "false",
        base_ok ? "true" : "false", alias ? "true" : "false", trial_reuse ? "true" : "false",
        q_propagates ? "true" : "false", sign_ok ? "true" : "false", line_size,
        mpi_ok ? "true" : "false");
    }
    return all_true_line(scalar_ok && base_ok && !alias && trial_reuse && q_propagates &&
                         sign_ok && legacy_calls == 0 && mpi_ok);
}

static int run_failure(int rank_sensitive) {
    struct b4b_fixture f;
    struct visco_sh_exact_line_search_request request;
    struct visco_sh_exact_line_search_result result, sentinel;
    struct visco_sh_exact_multi_shot_request base_request;
    struct visco_sh_exact_multi_shot_result base_result;
    int status, base_ok, result_ok, failure_ok, mpi_ok;

    initialize_fixture(&f);
    configure_runtime_topology();
    if (!all_true_line(prepare_canonical(&f, &f.canonical.material,
            f.base_primary.v, f.base_rho.v, f.base_q.v, f.canonical.eta) == 0)) return 0;
    RUN_MULTIPLE_SHOTS = 1;
    bind_multi_request(&base_request, &f.trial, 0, 2);
    base_request.material = &f.canonical.material;
    if (visco_sh_exact_objective(&base_request, &base_result) != 0) return 0;
    fill_all_cells(f.step_primary.v, 0.0f); fill_all_cells(f.step_rho.v, 0.0f);
    fill_all_cells(f.step_q.v, 0.0f); f.step_q.v[q_j][q_i] = 20.0f;
    memset(&request, 0, sizeof(request));
    bind_request(&f, &request.trial_objective, 0.0f, rank_sensitive ? 2 : 2);
    request.base_objective = base_result.objective;
    request.initial_alpha = 0.2f; request.scale_factor = 2.0f; request.max_retries = 3;
    if (!rank_sensitive) RUN_MULTIPLE_SHOTS = 0; /* Contract C: nsrc > 1. */
    snapshot_base(&f); material_copy(&f.snapshot.material, &f.canonical.material);
    sentinel.selected_alpha = -901.0f; sentinel.selected_objective = -902.0;
    sentinel.candidate_count = -903; sentinel.bracketed = -904; result = sentinel;
    reset_line_observation();
    if (rank_sensitive) inject_b2_rank = line_size > 1 ? 1 : 0;
    status = step_length_est_sh_visc_exact(&request, &result);
    base_ok = base_unchanged(&f) && material_same(&f.canonical.material, &f.snapshot.material);
    result_ok = memcmp(&result, &sentinel, sizeof(result)) == 0;
    failure_ok = status != 0 && trial_calls == 1 && result_ok && base_ok && legacy_calls == 0;
    mpi_ok = all_true_line(failure_ok) && same_float_line(trial_alpha[0]);
    if (line_rank == 0) printf(
        "{\"mode\":\"%s\",\"failure\":%s,\"fail_closed\":%s,"
        "\"result_sentinel_unchanged\":%s,\"base_unchanged\":%s,\"b4b_calls\":%d,"
        "\"legacy_fallback\":%s,\"mpi_ranks\":%d,\"mpi_identical_failure\":%s}\n",
        rank_sensitive ? "mpi-failure" : "contract-c", status != 0 ? "true" : "false",
        failure_ok ? "true" : "false", result_ok ? "true" : "false", base_ok ? "true" : "false",
        trial_calls, legacy_calls ? "true" : "false", line_size, mpi_ok ? "true" : "false");
    RUN_MULTIPLE_SHOTS = 1;
    return mpi_ok;
}

int main(int argc, char **argv) {
    int ok, buffer_size = 65536;
    void *buffer;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &line_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &line_size);
    /* The embedded fixture normally initializes these in its own main(). */
    b4b_rank = line_rank;
    b4b_size = line_size;
    buffer = malloc((size_t)buffer_size);
    if (buffer == NULL || MPI_Buffer_attach(buffer, buffer_size) != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, 711);
    ok = argc == 2 && strcmp(argv[1], "contract-c") == 0 ? run_failure(0) :
        (argc == 2 && strcmp(argv[1], "mpi-failure") == 0 ? run_failure(1) : run_success());
    MPI_Buffer_detach(&buffer, &buffer_size);
    free(buffer);
    MPI_Finalize();
    return ok ? 0 : 1;
}
