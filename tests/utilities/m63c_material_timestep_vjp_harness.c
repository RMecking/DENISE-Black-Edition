/* Flat test-only adapter for the isolated C7b production helper. */

#include "fd.h"

int m63c_material_timestep_vjp_harness(
        int mechanisms, double dt, double dh, double qsum,
        double strain_x, double strain_y, double bar_v_post_velocity,
        double bar_sxz_next, double bar_syz_next,
        const double *bar_r_next, const double *bar_q_next,
        double mu_x, double tau_x, double mu_y, double tau_y,
        double reference_sum, const double *eta_x, const double *b_x,
        const double *eta_y, const double *b_y,
        double forward_f_x, double forward_f_y,
        const double *forward_a_x, const double *forward_c_x,
        const double *forward_a_y, const double *forward_c_y,
        double output[5]) {
    struct visco_sh_material_timestep_vjp_input input;
    struct visco_sh_material_timestep_vjp_output result;
    int status;

    if (output == NULL) return -1;
    memset(&input, 0, sizeof(input));
    input.mechanisms = mechanisms;
    input.dt = dt;
    input.dh = dh;
    input.qsum = qsum;
    input.strain_x = strain_x;
    input.strain_y = strain_y;
    input.bar_v_post_velocity = bar_v_post_velocity;
    input.bar_sxz_next = bar_sxz_next;
    input.bar_syz_next = bar_syz_next;
    input.bar_r_next = bar_r_next;
    input.bar_q_next = bar_q_next;
    input.mu_x = mu_x;
    input.tau_x = tau_x;
    input.mu_y = mu_y;
    input.tau_y = tau_y;
    input.reference_sum = reference_sum;
    input.eta_x = eta_x;
    input.b_x = b_x;
    input.eta_y = eta_y;
    input.b_y = b_y;
    input.forward_f_x = forward_f_x;
    input.forward_f_y = forward_f_y;
    input.forward_a_x = forward_a_x;
    input.forward_c_x = forward_c_x;
    input.forward_a_y = forward_a_y;
    input.forward_c_y = forward_c_y;
    status = visco_sh_material_timestep_vjp(&input, &result);
    if (status != 0) return status;
    output[0] = result.g_rhoi;
    output[1] = result.g_mu_x;
    output[2] = result.g_mu_y;
    output[3] = result.g_tau_x;
    output[4] = result.g_tau_y;
    return 0;
}
