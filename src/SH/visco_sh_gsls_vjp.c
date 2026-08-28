/* Exact local reverse/VJP for the SH GSLS constitutive recurrence.
 *
 * This helper is deliberately independent of grid geometry, MPI, CPML,
 * free-surface handling, source/receiver state, and optimizer storage.  The
 * caller supplies the actual forward coefficients F, A_l=b_l*c_l and
 * C_l=-b_l*D_l prepared by prepare_update_s_visc_SH.  Consequently the state
 * transpose cannot drift from the coefficients used by the forward step.
 */

#include "fd.h"

#include <math.h>
#include <stddef.h>

int visco_sh_gsls_local_derivatives(
        int mechanisms, double dt, double unrelaxed_modulus, double tau,
        double reference_sum, const double *eta, const double *b,
        double *f_tau, double *f_modulus, double *c_tau,
        double *c_modulus) {
    double denominator, relaxed, relaxed_tau, relaxed_modulus;
    int l;

    if ((mechanisms < 1) || !(dt > 0.0) || !(unrelaxed_modulus > 0.0) ||
            !(tau > 0.0) || !(reference_sum >= 0.0) ||
            (eta == NULL) || (b == NULL) || (f_tau == NULL) ||
            (f_modulus == NULL) || (c_tau == NULL) ||
            (c_modulus == NULL)) return -1;
    if (!isfinite(dt) || !isfinite(unrelaxed_modulus) || !isfinite(tau) ||
            !isfinite(reference_sum)) return -1;

    denominator = 1.0 + reference_sum * tau;
    if (!(denominator > 0.0) || !isfinite(denominator)) return -1;
    relaxed = unrelaxed_modulus / denominator;
    relaxed_tau = -unrelaxed_modulus * reference_sum /
                  (denominator * denominator);
    relaxed_modulus = 1.0 / denominator;

    *f_tau = dt * (relaxed_tau * (1.0 + mechanisms * tau) +
                   mechanisms * relaxed);
    *f_modulus = dt * relaxed_modulus * (1.0 + mechanisms * tau);

    for (l = 0; l < mechanisms; ++l) {
        if (!(eta[l] > 0.0) || !(b[l] > 0.0) ||
                !isfinite(eta[l]) || !isfinite(b[l])) return -1;
        c_tau[l] = -b[l] * eta[l] * (relaxed_tau * tau + relaxed);
        c_modulus[l] = -b[l] * eta[l] * relaxed_modulus * tau;
    }

    return 0;
}

int visco_sh_gsls_local_vjp(
        int mechanisms, double dt, double strain, double bar_s_next,
        const double *bar_r_next, double forward_f,
        const double *forward_a, const double *forward_c,
        double f_tau, double f_modulus, const double *c_tau,
        const double *c_modulus, double *bar_s_prev,
        double *bar_r_prev, double *bar_strain, double *g_tau,
        double *g_modulus) {
    double half_dt, strain_value, tau_value, modulus_value, t_value;
    int l;

    if ((mechanisms < 1) || !(dt > 0.0) ||
            (bar_r_next == NULL) || (forward_a == NULL) ||
            (forward_c == NULL) || (c_tau == NULL) ||
            (c_modulus == NULL) || (bar_s_prev == NULL) ||
            (bar_r_prev == NULL) || (bar_strain == NULL) ||
            (g_tau == NULL) || (g_modulus == NULL)) return -1;
    if (!isfinite(dt) || !isfinite(strain) || !isfinite(bar_s_next) ||
            !isfinite(forward_f) || !isfinite(f_tau) ||
            !isfinite(f_modulus)) return -1;

    half_dt = 0.5 * dt;
    strain_value = forward_f * bar_s_next;
    tau_value = f_tau * bar_s_next;
    modulus_value = f_modulus * bar_s_next;
    *bar_s_prev += bar_s_next;

    for (l = 0; l < mechanisms; ++l) {
        if (!isfinite(bar_r_next[l]) || !isfinite(forward_a[l]) ||
                !isfinite(forward_c[l]) || !isfinite(c_tau[l]) ||
                !isfinite(c_modulus[l])) return -1;
        t_value = bar_r_next[l] + half_dt * bar_s_next;
        bar_r_prev[l] += forward_a[l] * t_value +
                         half_dt * bar_s_next;
        strain_value += forward_c[l] * t_value;
        tau_value += c_tau[l] * t_value;
        modulus_value += c_modulus[l] * t_value;
    }

    *bar_strain += strain_value;
    *g_tau += strain * tau_value;
    *g_modulus += strain * modulus_value;
    return 0;
}
