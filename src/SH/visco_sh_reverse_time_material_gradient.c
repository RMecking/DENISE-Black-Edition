/* Material-aware companion for the locked fixed-material reverse driver. */

#include "fd.h"

#include <stddef.h>

static int unsupported_cpml_overlap_material(
        const struct visco_sh_full_step_config *cfg) {
    if ((cfg->fw > 0) && (!cfg->boundary) && (cfg->nproc_x == 1) &&
            (cfg->nx <= 2 * cfg->fw)) return 1;
    if ((cfg->fw > 0) && (!cfg->free_surface) &&
            (cfg->nproc_y == 1) && (cfg->ny <= 2 * cfg->fw)) return 1;
    return 0;
}
static int state_complete_for_material_driver(
        const struct visco_sh_full_state *state, int fw) {
    if ((state == NULL) || (state->vz == NULL) || (state->sxz == NULL) ||
            (state->syz == NULL) || (state->r == NULL) ||
            (state->q == NULL)) return 0;
    if ((fw > 0) && ((state->psi_sxz_x == NULL) ||
            (state->psi_syz_y == NULL) || (state->psi_vzx == NULL) ||
            (state->psi_vzy == NULL))) return 0;
    return 1;
}

static int states_distinct_for_material_driver(
        const struct visco_sh_full_state *left,
        const struct visco_sh_full_state *right, int fw) {
    if ((left == right) || (left->vz == right->vz) ||
            (left->sxz == right->sxz) || (left->syz == right->syz) ||
            (left->r == right->r) || (left->q == right->q)) return 0;
    if ((fw > 0) && ((left->psi_sxz_x == right->psi_sxz_x) ||
            (left->psi_syz_y == right->psi_syz_y) ||
            (left->psi_vzx == right->psi_vzx) ||
            (left->psi_vzy == right->psi_vzy))) return 0;
    return 1;
}

struct c7cb2_double_field {
    double **row;
    double *data;
};

static int c7cb2_field_allocate(
        struct c7cb2_double_field *field, int nx, int ny) {
    int j;
    field->row = (double **)calloc((size_t)(ny + 2), sizeof(double *));
    field->data = (double *)calloc(
            (size_t)(nx + 2) * (ny + 2), sizeof(double));
    if ((field->row == NULL) || (field->data == NULL)) {
        free(field->data);
        free(field->row);
        field->data = NULL;
        field->row = NULL;
        return -1;
    }
    for (j = 0; j <= ny + 1; ++j)
        field->row[j] = field->data + (size_t)j * (nx + 2);
    return 0;
}

static void c7cb2_field_release(struct c7cb2_double_field *field) {
    free(field->data);
    free(field->row);
}

static int c7cb2_material_preflight(
        const struct visco_sh_full_step_config *cfg, int nsteps,
        const struct visco_sh_reverse_time_material_context *material) {
    extern int NX, NY;
    int i, j, l, n;
    if ((material == NULL) || (material->trajectory == NULL) ||
            (material->trajectory->steps == NULL) ||
            (material->trajectory->nsteps != nsteps) ||
            (material->trajectory->nx != cfg->nx) ||
            (material->trajectory->ny != cfg->ny) ||
            (material->trajectory->dtinv != 1) ||
            (NX != cfg->nx) || (NY != cfg->ny) ||
            (material->mu_x == NULL) || (material->tau_x == NULL) ||
            (material->mu_y == NULL) || (material->tau_y == NULL) ||
            (material->eta_x == NULL) || (material->eta_y == NULL) ||
            ((material->invmat1 != 1) && (material->invmat1 != 3)) ||
            (material->mapping == NULL) ||
            (material->primary_post == NULL) ||
            (material->rho_post == NULL) || (material->owned_q == NULL) ||
            (material->grad_primary == NULL) ||
            (material->grad_rho == NULL) || (material->grad_q == NULL))
        return -1;
    if (!isfinite(material->reference_sum)) return -1;
    for (l = 1; l <= cfg->mechanisms; ++l)
        if (!isfinite(material->eta_x[l]) ||
                !isfinite(material->eta_y[l])) return -1;
    for (n = 0; n < nsteps; ++n)
        if ((material->trajectory->steps[n].qsum == NULL) ||
                (material->trajectory->steps[n].strain_x == NULL) ||
                (material->trajectory->steps[n].strain_y == NULL)) return -1;
    for (j = 0; j <= cfg->ny + 1; ++j)
        for (i = 0; i <= cfg->nx + 1; ++i)
            if (!(material->primary_post[j][i] > 0.0f) ||
                    !(material->rho_post[j][i] > 0.0f) ||
                    !isfinite(material->primary_post[j][i]) ||
                    !isfinite(material->rho_post[j][i])) return -1;
    for (j = 1; j <= cfg->ny; ++j)
        for (i = 1; i <= cfg->nx; ++i)
            if (!isfinite(material->owned_q[j][i]) ||
                    !isfinite(q_to_tau_derivative(
                        material->owned_q[j][i], material->mapping))) return -1;
    return 0;
}

int visco_sh_reverse_time_adjoint_material(
        const struct visco_sh_full_step_config *base_config,
        int nsteps,
        const double *bar_receiver_series,
        struct visco_sh_full_state *bar_terminal_work,
        struct visco_sh_full_state *bar_initial,
        struct visco_sh_full_state *scratch,
        double *bar_signal_series,
        const struct visco_sh_reverse_time_material_context *material) {
    struct visco_sh_full_step_config step_config;
    struct visco_sh_full_state *current, *previous;
    struct visco_sh_material_adjoint_step_context step_material;
    struct visco_sh_native_material_gradient_fields step_native, sum_native;
    struct c7cb2_double_field step_fields[5] = {{0}}, sum_fields[5] = {{0}};
    struct visco_sh_material_timestep_vjp_output *unweighted = NULL;
    struct visco_sh_material_timestep_vjp_output *weighted = NULL;
    int i, j, n, source, channel, point, points, status = -1;

    if ((base_config == NULL) || (nsteps < 1) ||
            ((base_config->nrec > 0) && (bar_receiver_series == NULL)) ||
            ((base_config->nsrc > 0) && (bar_signal_series == NULL)))
        return -1;
    if (!state_complete_for_material_driver(bar_terminal_work, base_config->fw) ||
            !state_complete_for_material_driver(bar_initial, base_config->fw) ||
            !state_complete_for_material_driver(scratch, base_config->fw) ||
            !states_distinct_for_material_driver(
                bar_terminal_work, bar_initial, base_config->fw) ||
            !states_distinct_for_material_driver(
                bar_terminal_work, scratch, base_config->fw) ||
            !states_distinct_for_material_driver(
                bar_initial, scratch, base_config->fw)) return -1;
    if (unsupported_cpml_overlap_material(base_config)) return -2;
    if (c7cb2_material_preflight(base_config, nsteps, material) != 0)
        return -1;

    for (channel = 0; channel < 5; ++channel)
        if ((c7cb2_field_allocate(
                    &step_fields[channel], base_config->nx,
                    base_config->ny) != 0) ||
                (c7cb2_field_allocate(
                    &sum_fields[channel], base_config->nx,
                    base_config->ny) != 0)) {
            status = MPI_ERR_NO_MEM;
            goto cleanup;
        }
    points = base_config->nx * base_config->ny;
    unweighted = (struct visco_sh_material_timestep_vjp_output *)calloc(
            (size_t)points, sizeof(*unweighted));
    weighted = (struct visco_sh_material_timestep_vjp_output *)calloc(
            (size_t)points, sizeof(*weighted));
    if ((unweighted == NULL) || (weighted == NULL)) {
        status = MPI_ERR_NO_MEM;
        goto cleanup;
    }
    step_native.g_rhoi = step_fields[0].row;
    step_native.g_mu_x = step_fields[1].row;
    step_native.g_mu_y = step_fields[2].row;
    step_native.g_tau_x = step_fields[3].row;
    step_native.g_tau_y = step_fields[4].row;
    sum_native.g_rhoi = sum_fields[0].row;
    sum_native.g_mu_x = sum_fields[1].row;
    sum_native.g_mu_y = sum_fields[2].row;
    sum_native.g_tau_x = sum_fields[3].row;
    sum_native.g_tau_y = sum_fields[4].row;

    for (n = 0; n < nsteps; ++n)
        for (source = 0; source < base_config->nsrc; ++source)
            bar_signal_series[(size_t)n * base_config->nsrc + source] = 0.0;
    current = bar_terminal_work;
    for (n = nsteps - 1; n >= 0; --n) {
        step_config = *base_config;
        step_config.bar_receiver = (base_config->nrec > 0) ?
                bar_receiver_series + (size_t)n * base_config->nrec : NULL;
        previous = (n == 0) ? bar_initial :
                ((current == scratch) ? bar_terminal_work : scratch);
        memset(&step_material, 0, sizeof(step_material));
        step_material.observable = &material->trajectory->steps[n];
        step_material.mu_x = material->mu_x;
        step_material.tau_x = material->tau_x;
        step_material.mu_y = material->mu_y;
        step_material.tau_y = material->tau_y;
        step_material.reference_sum = material->reference_sum;
        step_material.eta_x = material->eta_x;
        step_material.eta_y = material->eta_y;
        step_material.native_output = &step_native;
        status = visco_sh_full_state_adjoint_step_material(
                &step_config, current, previous,
                (base_config->nsrc > 0) ?
                    bar_signal_series + (size_t)n * base_config->nsrc : NULL,
                &step_material);
        if (status != 0) goto cleanup;
        for (j = 1; j <= base_config->ny; ++j)
            for (i = 1; i <= base_config->nx; ++i) {
                sum_native.g_rhoi[j][i] += step_native.g_rhoi[j][i];
                sum_native.g_mu_x[j][i] += step_native.g_mu_x[j][i];
                sum_native.g_mu_y[j][i] += step_native.g_mu_y[j][i];
                sum_native.g_tau_x[j][i] += step_native.g_tau_x[j][i];
                sum_native.g_tau_y[j][i] += step_native.g_tau_y[j][i];
            }
        current = previous;
    }

    for (j = 1; j <= base_config->ny; ++j)
        for (i = 1; i <= base_config->nx; ++i) {
            point = (j - 1) * base_config->nx + i - 1;
            unweighted[point].g_rhoi = sum_native.g_rhoi[j][i];
            unweighted[point].g_mu_x = sum_native.g_mu_x[j][i];
            unweighted[point].g_mu_y = sum_native.g_mu_y[j][i];
            unweighted[point].g_tau_x = sum_native.g_tau_x[j][i];
            unweighted[point].g_tau_y = sum_native.g_tau_y[j][i];
        }
    status = visco_sh_temporal_native_gradient_accumulate(
            1, points, material->trajectory->dtinv,
            unweighted, weighted);
    if (status != 0) goto cleanup;
    for (j = 1; j <= base_config->ny; ++j)
        for (i = 1; i <= base_config->nx; ++i) {
            point = (j - 1) * base_config->nx + i - 1;
            sum_native.g_rhoi[j][i] = weighted[point].g_rhoi;
            sum_native.g_mu_x[j][i] = weighted[point].g_mu_x;
            sum_native.g_mu_y[j][i] = weighted[point].g_mu_y;
            sum_native.g_tau_x[j][i] = weighted[point].g_tau_x;
            sum_native.g_tau_y[j][i] = weighted[point].g_tau_y;
        }
    status = visco_sh_distributed_material_gradient_vjp(
            material->invmat1, material->mapping, material->primary_post,
            material->rho_post, material->owned_q, &sum_native,
            material->grad_primary, material->grad_rho, material->grad_q);

cleanup:
    free(unweighted);
    free(weighted);
    for (channel = 0; channel < 5; ++channel) {
        c7cb2_field_release(&step_fields[channel]);
        c7cb2_field_release(&sum_fields[channel]);
    }
    return status;
}
