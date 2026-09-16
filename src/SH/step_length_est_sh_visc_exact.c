/* Inactive exact physical-Q line search for viscoelastic SH FWI. */

#include "fd.h"

#include <float.h>
#include <math.h>

static int exact_line_search_preflight(
        const struct visco_sh_exact_line_search_request *request,
        const struct visco_sh_exact_line_search_result *result) {
    if ((request == NULL) || (result == NULL)) return -1;
    if (!isfinite(request->base_objective) ||
            !isfinite(request->initial_alpha) ||
            !isfinite(request->scale_factor)) return -1;
    if ((request->initial_alpha <= 0.0f) ||
            (request->scale_factor <= 1.0f) ||
            (request->max_retries < 0)) return -1;
    return 0;
}

static int exact_line_search_scalars_match(
        const struct visco_sh_exact_line_search_request *request) {
    double double_values[3], double_min[3], double_max[3];
    int retries_min, retries_max;

    double_values[0] = request->base_objective;
    double_values[1] = (double)request->initial_alpha;
    double_values[2] = (double)request->scale_factor;
    if (MPI_Allreduce(double_values, double_min, 3, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (MPI_Allreduce(double_values, double_max, 3, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (MPI_Allreduce(&request->max_retries, &retries_min, 1, MPI_INT,
                      MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (MPI_Allreduce(&request->max_retries, &retries_max, 1, MPI_INT,
                      MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if ((double_min[0] != double_max[0]) ||
            (double_min[1] != double_max[1]) ||
            (double_min[2] != double_max[2]) ||
            (retries_min != retries_max)) return -1;
    return 0;
}

static int exact_line_search_evaluate(
        const struct visco_sh_exact_line_search_request *request,
        float alpha, double *objective) {
    struct visco_sh_exact_trial_objective_request trial;
    struct visco_sh_exact_trial_objective_result trial_result;
    double objective_min, objective_max;
    int status, local_failure, any_failure;

    trial = request->trial_objective;
    trial.trial_state.alpha = alpha;
    status = visco_sh_exact_trial_objective(&trial, &trial_result);
    local_failure = (status != 0) || !isfinite(trial_result.objective);
    if (MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (any_failure) return (status == -2) ? -2 : -1;

    if (MPI_Allreduce(&trial_result.objective, &objective_min, 1, MPI_DOUBLE,
                      MPI_MIN, MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (MPI_Allreduce(&trial_result.objective, &objective_max, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (objective_min != objective_max) return -1;
    *objective = objective_min;
    return 0;
}

static int exact_line_search_same_alpha(float left, float right) {
    float scale = fmaxf(1.0f, fmaxf(fabsf(left), fabsf(right)));
    return fabsf(left - right) <= (4.0f * FLT_EPSILON * scale);
}

int step_length_est_sh_visc_exact(
        const struct visco_sh_exact_line_search_request *request,
        struct visco_sh_exact_line_search_result *result) {
    struct visco_sh_exact_line_search_result completed;
    double objectives[4];
    float alphas[4], candidate, selected, selected_min, selected_max;
    int local_failure, any_failure, status, retries;

    local_failure = exact_line_search_preflight(request, result) != 0;
    if (MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (any_failure) return -1;
    status = exact_line_search_scalars_match(request);
    if (status != 0) return status;

    objectives[1] = request->base_objective;
    alphas[1] = 0.0f;
    candidate = request->initial_alpha;
    completed.candidate_count = 0;
    completed.bracketed = 0;

    /* Preserve the legacy scalar policy: shrink until the first improving
     * candidate is found. */
    retries = 0;
    for (;;) {
        status = exact_line_search_evaluate(request, candidate, &objectives[2]);
        if (status != 0) return status;
        completed.candidate_count++;
        alphas[2] = candidate;
        if (objectives[2] < objectives[1]) break;
        if (retries >= request->max_retries) return -1;
        candidate /= request->scale_factor;
        if (!isfinite(candidate) || (candidate <= 0.0f)) return -1;
        retries++;
    }

    /* Grow from the improving point until the minimum is bracketed.  At the
     * retry limit the final descending candidate is retained, matching the
     * existing SH policy before parabolic selection. */
    candidate = alphas[2] + (alphas[2] / request->scale_factor);
    retries = 0;
    for (;;) {
        status = exact_line_search_evaluate(request, candidate, &objectives[3]);
        if (status != 0) return status;
        completed.candidate_count++;
        alphas[3] = candidate;
        if (objectives[2] < objectives[3]) {
            completed.bracketed = 1;
            break;
        }
        if (retries >= request->max_retries) break;
        candidate += candidate / request->scale_factor;
        if (!isfinite(candidate) || (candidate <= alphas[3])) return -1;
        retries++;
    }

    selected = calc_opt_step(objectives, alphas, 1);
    local_failure = !isfinite(selected) || (selected < 0.0f);
    if (MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (any_failure) return -1;
    if (MPI_Allreduce(&selected, &selected_min, 1, MPI_FLOAT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (MPI_Allreduce(&selected, &selected_max, 1, MPI_FLOAT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (selected_min != selected_max) return -1;
    selected = selected_min;
    if (exact_line_search_same_alpha(selected, alphas[1])) {
        completed.selected_objective = objectives[1];
    } else if (exact_line_search_same_alpha(selected, alphas[2])) {
        completed.selected_objective = objectives[2];
    } else if (exact_line_search_same_alpha(selected, alphas[3])) {
        completed.selected_objective = objectives[3];
    } else {
        status = exact_line_search_evaluate(
                request, selected, &completed.selected_objective);
        if (status != 0) return status;
        completed.candidate_count++;
    }
    completed.selected_alpha = selected;

    /* Transactional publication: failures above leave caller output intact. */
    *result = completed;
    return 0;
}
