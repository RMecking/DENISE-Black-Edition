/* Inactive until a later C8c integration gate. */

#include "fd.h"

static int trial_state_preflight(
        const struct visco_sh_exact_trial_state_request *request) {
    if (request == NULL) return -1;
    if ((request->nx < 1) || (request->ny < 1)) return -2;
    if (!isfinite(request->alpha) || (request->alpha < 0.0f)) return -3;
    if (request->q_mapping == NULL) return -4;
    if (request->q_mapping->mode != Q_PARAMETERIZATION_PHYSICAL) return -5;
    if (!isfinite(request->rho_lower) || !isfinite(request->rho_upper) ||
            (request->rho_lower > request->rho_upper)) return -6;
    if (!isfinite(request->q_lower) || !isfinite(request->q_upper) ||
            !(request->q_lower > 0.0f) || !(request->q_upper > 0.0f) ||
            (request->q_lower > request->q_upper)) return -7;
    if (request->primary_bounds_enabled &&
            (!isfinite(request->primary_lower) ||
             !isfinite(request->primary_upper) ||
             (request->primary_lower > request->primary_upper))) return -8;
    if ((request->base_primary == NULL) || (request->base_rho == NULL) ||
            (request->base_q == NULL) ||
            (request->optimizer_step_primary == NULL) ||
            (request->optimizer_step_rho == NULL) ||
            (request->optimizer_step_q == NULL) ||
            (request->trial_primary == NULL) ||
            (request->trial_rho == NULL) || (request->trial_q == NULL) ||
            (request->trial_tau == NULL)) return -9;
    return 0;
}

static float trial_candidate(float base, float step, float alpha,
                             int bounds_enabled, float lower, float upper) {
    float candidate;

    if (alpha == 0.0f) return base;
    candidate = base - alpha * step;
    if (!isfinite(candidate) || !(candidate > 0.0f)) return base;
    if (bounds_enabled && ((candidate < lower) || (candidate > upper)))
        return base;
    return candidate;
}

static int trial_state_rows_are_valid(
        const struct visco_sh_exact_trial_state_request *request) {
    int j;

    for (j = 1; j <= request->ny; ++j) {
        if ((request->base_primary[j] == NULL) ||
                (request->base_rho[j] == NULL) ||
                (request->base_q[j] == NULL) ||
                (request->optimizer_step_primary[j] == NULL) ||
                (request->optimizer_step_rho[j] == NULL) ||
                (request->optimizer_step_q[j] == NULL) ||
                (request->trial_primary[j] == NULL) ||
                (request->trial_rho[j] == NULL) ||
                (request->trial_q[j] == NULL) ||
                (request->trial_tau[j] == NULL)) return -10;
    }
    return 0;
}

static int trial_state_inputs_are_valid(
        const struct visco_sh_exact_trial_state_request *request) {
    int i, j;

    for (j = 1; j <= request->ny; ++j) {
        for (i = 1; i <= request->nx; ++i) {
            if (!isfinite(request->base_primary[j][i]) ||
                    !isfinite(request->base_rho[j][i]) ||
                    !isfinite(request->base_q[j][i]) ||
                    !isfinite(request->optimizer_step_primary[j][i]) ||
                    !isfinite(request->optimizer_step_rho[j][i]) ||
                    !isfinite(request->optimizer_step_q[j][i])) return -11;
            if (!(request->base_primary[j][i] > 0.0f) ||
                    !(request->base_rho[j][i] > 0.0f) ||
                    !(request->base_q[j][i] > 0.0f)) return -12;
        }
    }
    return 0;
}

static int trial_state_tau_is_valid(
        const struct visco_sh_exact_trial_state_request *request) {
    float q_value, tau_value;
    int i, j;

    for (j = 1; j <= request->ny; ++j) {
        for (i = 1; i <= request->nx; ++i) {
            q_value = trial_candidate(request->base_q[j][i],
                    request->optimizer_step_q[j][i], request->alpha, 1,
                    request->q_lower, request->q_upper);
            tau_value = q_to_tau(q_value, request->q_mapping);
            if (!isfinite(tau_value) || !(tau_value > 0.0f)) return -13;
        }
    }
    return 0;
}

int visco_sh_exact_build_trial_parameter_state(
        const struct visco_sh_exact_trial_state_request *request) {
    float primary_value, rho_value, q_value, tau_value;
    int i, j, status;

    status = trial_state_preflight(request);
    if (status != 0) return status;
    status = trial_state_rows_are_valid(request);
    if (status != 0) return status;
    status = trial_state_inputs_are_valid(request);
    if (status != 0) return status;
    status = trial_state_tau_is_valid(request);
    if (status != 0) return status;

    /* Subtractive candidates use base-cell rejection, not clipping.  Q is
     * optimizer-owned physical state; tau is derived solver state only. */
    for (j = 1; j <= request->ny; ++j) {
        for (i = 1; i <= request->nx; ++i) {
            primary_value = trial_candidate(request->base_primary[j][i],
                    request->optimizer_step_primary[j][i], request->alpha,
                    request->primary_bounds_enabled,
                    request->primary_lower, request->primary_upper);
            rho_value = trial_candidate(request->base_rho[j][i],
                    request->optimizer_step_rho[j][i], request->alpha, 1,
                    request->rho_lower, request->rho_upper);
            q_value = trial_candidate(request->base_q[j][i],
                    request->optimizer_step_q[j][i], request->alpha, 1,
                    request->q_lower, request->q_upper);
            tau_value = q_to_tau(q_value, request->q_mapping);

            request->trial_primary[j][i] = primary_value;
            request->trial_rho[j][i] = rho_value;
            request->trial_q[j][i] = q_value;
            request->trial_tau[j][i] = tau_value;
        }
    }
    return 0;
}
