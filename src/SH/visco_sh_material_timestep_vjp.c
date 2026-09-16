/* Exact native material sensitivity contribution of one viscoelastic SH
 * physical timestep.  This composes locked C1/C6a helpers and deliberately
 * performs neither temporal accumulation nor physical-parameter mapping. */

#include "fd.h"

static int branch_vjp(
        int mechanisms, double dt, double strain, double bar_s_next,
        const double *bar_memory_next, double modulus, double tau,
        double reference_sum, const double *eta, const double *b,
        double forward_f, const double *forward_a, const double *forward_c,
        double *g_modulus, double *g_tau) {
    double f_tau, f_modulus;
    double bar_s_prev = 0.0, bar_strain = 0.0;
    double c_tau[mechanisms], c_modulus[mechanisms];
    double bar_memory_prev[mechanisms];
    int status;

    memset(bar_memory_prev, 0, sizeof(bar_memory_prev));

    status = visco_sh_gsls_local_derivatives(
            mechanisms, dt, modulus, tau, reference_sum, eta, b,
            &f_tau, &f_modulus, c_tau, c_modulus);
    if (status == 0)
        status = visco_sh_gsls_local_vjp(
                mechanisms, dt, strain, bar_s_next, bar_memory_next,
                forward_f, forward_a, forward_c, f_tau, f_modulus,
                c_tau, c_modulus, &bar_s_prev, bar_memory_prev,
                &bar_strain, g_tau, g_modulus);

    return status;
}

int visco_sh_material_timestep_vjp(
        const struct visco_sh_material_timestep_vjp_input *input,
        struct visco_sh_material_timestep_vjp_output *output) {
    int status;

    if ((input == NULL) || (output == NULL) || (input->mechanisms < 1) ||
            !(input->dt > 0.0) || !(input->dh > 0.0) ||
            (input->bar_r_next == NULL) || (input->bar_q_next == NULL) ||
            (input->eta_x == NULL) || (input->b_x == NULL) ||
            (input->eta_y == NULL) || (input->b_y == NULL) ||
            (input->forward_a_x == NULL) ||
            (input->forward_c_x == NULL) ||
            (input->forward_a_y == NULL) ||
            (input->forward_c_y == NULL)) return -1;

    memset(output, 0, sizeof(*output));
    output->g_rhoi = visco_sh_velocity_rhoi_vjp(
            input->dt, input->dh, input->qsum, 0.0,
            input->bar_v_post_velocity);
    if (!isfinite(output->g_rhoi)) return -1;

    status = branch_vjp(
            input->mechanisms, input->dt, input->strain_x,
            input->bar_sxz_next, input->bar_r_next, input->mu_x,
            input->tau_x, input->reference_sum, input->eta_x, input->b_x,
            input->forward_f_x, input->forward_a_x, input->forward_c_x,
            &output->g_mu_x, &output->g_tau_x);
    if (status != 0) return status;

    status = branch_vjp(
            input->mechanisms, input->dt, input->strain_y,
            input->bar_syz_next, input->bar_q_next, input->mu_y,
            input->tau_y, input->reference_sum, input->eta_y, input->b_y,
            input->forward_f_y, input->forward_a_y, input->forward_c_y,
            &output->g_mu_y, &output->g_tau_y);
    if (status != 0) return status;

    return 0;
}
