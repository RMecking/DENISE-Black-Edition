/* Inactive exact viscoelastic SH trial-objective composition boundary. */

#include "fd.h"

static int trial_objective_matrix_is_base(
        float **matrix,
        const struct visco_sh_exact_trial_state_request *trial) {
    return (matrix == trial->base_primary) || (matrix == trial->base_rho) ||
           (matrix == trial->base_q);
}

static int trial_objective_matrix_is_trial(
        float **matrix,
        const struct visco_sh_exact_trial_state_request *trial) {
    return (matrix == trial->trial_primary) ||
           (matrix == trial->trial_rho) || (matrix == trial->trial_q) ||
           (matrix == trial->trial_tau);
}

static int trial_objective_preflight(
        const struct visco_sh_exact_trial_objective_request *request,
        const struct visco_sh_exact_trial_objective_result *result) {
    const struct visco_sh_exact_trial_state_request *trial;
    const struct matSH *material;

    if ((request == NULL) || (result == NULL) ||
            (request->trial_material == NULL)) return -1;
    trial = &request->trial_state;
    material = request->trial_material;
    if ((trial->base_primary == NULL) || (trial->base_rho == NULL) ||
            (trial->base_q == NULL) ||
            (trial->optimizer_step_primary == NULL) ||
            (trial->optimizer_step_rho == NULL) ||
            (trial->optimizer_step_q == NULL) ||
            (trial->trial_primary == NULL) || (trial->trial_rho == NULL) ||
            (trial->trial_q == NULL) || (trial->trial_tau == NULL) ||
            (trial->q_mapping == NULL) ||
            (request->frequencies_hz == NULL) ||
            (request->peta == NULL) ||
            (request->objective.material != material)) return -1;

    /* B2 output and B4A target storage must never overwrite Base or alias
     * each other. Deeper row/allocation validity remains owned by B2/B4A. */
    if (trial_objective_matrix_is_base(trial->trial_primary, trial) ||
            trial_objective_matrix_is_base(trial->trial_rho, trial) ||
            trial_objective_matrix_is_base(trial->trial_q, trial) ||
            trial_objective_matrix_is_base(trial->trial_tau, trial) ||
            (trial->trial_primary == trial->trial_rho) ||
            (trial->trial_primary == trial->trial_q) ||
            (trial->trial_primary == trial->trial_tau) ||
            (trial->trial_rho == trial->trial_q) ||
            (trial->trial_rho == trial->trial_tau) ||
            (trial->trial_q == trial->trial_tau)) return -1;
    if (trial_objective_matrix_is_base(material->pu, trial) ||
            trial_objective_matrix_is_base(material->prho, trial) ||
            trial_objective_matrix_is_base(material->pqs, trial) ||
            trial_objective_matrix_is_base(material->ptaus, trial) ||
            trial_objective_matrix_is_trial(material->pu, trial) ||
            trial_objective_matrix_is_trial(material->prho, trial) ||
            trial_objective_matrix_is_trial(material->pqs, trial) ||
            trial_objective_matrix_is_trial(material->ptaus, trial)) return -1;
    return 0;
}

int visco_sh_exact_trial_objective(
        const struct visco_sh_exact_trial_objective_request *request,
        struct visco_sh_exact_trial_objective_result *result) {
    struct visco_sh_exact_material_preparation_request material_request;
    struct visco_sh_exact_multi_shot_result objective_result;
    int local_failure, any_failure, status;

    local_failure = trial_objective_preflight(request, result) != 0;
    if (MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (any_failure) return -1;

    status = visco_sh_exact_build_trial_parameter_state(
            &request->trial_state);
    local_failure = status != 0;
    if (MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (any_failure) return -1;

    material_request.primary = request->trial_state.trial_primary;
    material_request.rho = request->trial_state.trial_rho;
    material_request.physical_q = request->trial_state.trial_q;
    material_request.target = request->trial_material;
    material_request.mechanisms = request->mechanisms;
    material_request.dt = request->dt;
    material_request.frequencies_hz = request->frequencies_hz;
    material_request.peta = request->peta;
    status = visco_sh_exact_prepare_visco_material(&material_request);
    if (status != 0) return status;

    status = visco_sh_exact_objective(
            &request->objective, &objective_result);
    if (status != 0) return status;

    result->objective = objective_result.objective;
    return 0;
}
