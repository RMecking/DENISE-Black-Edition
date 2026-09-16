/* Exact local VJPs for the SH material-preparation graph. */

#include "fd.h"

enum {
    C6A_CENTER = 0,
    C6A_EAST = 1,
    C6A_SOUTH = 2,
    C6A_SOUTHEAST = 3,
    C6A_MU_X = 0,
    C6A_MU_Y = 1,
    C6A_TAU_X = 2,
    C6A_TAU_Y = 3,
    C6A_RHOI = 4
};

double visco_sh_harmonic_pair(double left, double right) {
    if (!(left > 0.0) || !(right > 0.0))
        return NAN;
    return 2.0 * left * right / (left + right);
}

int visco_sh_harmonic_pair_vjp(double left, double right, double bar_value,
                               double *bar_left, double *bar_right) {
    double denominator;
    if (bar_left == NULL || bar_right == NULL) return -1;
    if (!(left > 0.0) || !(right > 0.0) ||
        !isfinite(left) || !isfinite(right)) return -2;
    denominator = left + right;
    *bar_left += bar_value * 2.0 * right * right
               / (denominator * denominator);
    *bar_right += bar_value * 2.0 * left * left
                / (denominator * denominator);
    return 0;
}

void visco_sh_av_tau_local_vjp(double bar_tau_x, double bar_tau_y,
                               double bar_tau_cells[4]) {
    int cell;
    if (bar_tau_cells == NULL) return;
    for (cell = 0; cell < 4; ++cell)
        bar_tau_cells[cell] += 0.25 * bar_tau_x;
    bar_tau_cells[C6A_CENTER] += bar_tau_y;
}

double visco_sh_rhoi_value(double rho) {
    return rho < 1.0e-4 ? 0.0 : 1.0 / rho;
}

double visco_sh_rhoi_vjp(double rho, double bar_rhoi) {
    return rho < 1.0e-4 ? 0.0 : -bar_rhoi / (rho * rho);
}

double visco_sh_velocity_rhoi_vjp(double dt, double dh,
                                  double corrected_qx, double corrected_qy,
                                  double bar_v_next) {
    if (!(dh > 0.0) || !isfinite(dh)) return NAN;
    return (dt / dh) * (corrected_qx + corrected_qy) * bar_v_next;
}

static int material_cells(int invmat1, const double primary[4],
                          const double rho[4], double mu[4]) {
    int cell;
    if (invmat1 != 1 && invmat1 != 3) return -3;
    for (cell = 0; cell < 4; ++cell) {
        if (!(primary[cell] > 0.0) || !(rho[cell] > 0.0) ||
            !isfinite(primary[cell]) || !isfinite(rho[cell])) return -2;
        mu[cell] = invmat1 == 1
                 ? rho[cell] * primary[cell] * primary[cell]
                 : primary[cell];
        if (!(mu[cell] > 0.0) || !isfinite(mu[cell])) return -2;
    }
    return 0;
}

static int tau_value(double q, const struct q_tau_mapping *mapping,
                     double *tau) {
    double inverse_tau;
    if (mapping == NULL || tau == NULL) return -1;
    if (!(q > 0.0) || !isfinite(q)) return -2;
    if (mapping->mode == Q_PARAMETERIZATION_LEGACY) {
        *tau = 2.0 / q;
        return 0;
    }
    if (mapping->mode != Q_PARAMETERIZATION_PHYSICAL) return -3;
    inverse_tau = mapping->inverse_tau_per_q * q
                + mapping->inverse_tau_offset;
    if (!(inverse_tau > 0.0) || !isfinite(inverse_tau)) return -2;
    *tau = 1.0 / inverse_tau;
    return 0;
}

int visco_sh_material_patch_forward(
        int invmat1, const struct q_tau_mapping *mapping,
        const double primary[4], const double rho[4], const double q[4],
        double output[5]) {
    double mu[4], tau[4];
    int cell, status;
    if (primary == NULL || rho == NULL || q == NULL || output == NULL)
        return -1;
    status = material_cells(invmat1, primary, rho, mu);
    if (status != 0) return status;
    for (cell = 0; cell < 4; ++cell) {
        status = tau_value(q[cell], mapping, &tau[cell]);
        if (status != 0) return status;
    }
    output[C6A_MU_X] = visco_sh_harmonic_pair(mu[C6A_CENTER], mu[C6A_EAST]);
    output[C6A_MU_Y] = visco_sh_harmonic_pair(mu[C6A_CENTER], mu[C6A_SOUTH]);
    output[C6A_TAU_X] = 0.25 * (tau[C6A_CENTER] + tau[C6A_EAST]
                              + tau[C6A_SOUTH] + tau[C6A_SOUTHEAST]);
    output[C6A_TAU_Y] = tau[C6A_CENTER];
    output[C6A_RHOI] = visco_sh_rhoi_value(rho[C6A_CENTER]);
    return 0;
}

int visco_sh_material_patch_vjp(
        int invmat1, const struct q_tau_mapping *mapping,
        const double primary[4], const double rho[4], const double q[4],
        const double bar_output[5], double bar_primary[4],
        double bar_rho[4], double bar_q[4]) {
    double mu[4], bar_mu[4] = {0.0, 0.0, 0.0, 0.0};
    double bar_tau[4] = {0.0, 0.0, 0.0, 0.0};
    double tau, derivative;
    int cell, status;
    if (primary == NULL || rho == NULL || q == NULL || bar_output == NULL ||
        bar_primary == NULL || bar_rho == NULL || bar_q == NULL) return -1;
    status = material_cells(invmat1, primary, rho, mu);
    if (status != 0) return status;
    for (cell = 0; cell < 4; ++cell) {
        bar_primary[cell] = 0.0;
        bar_rho[cell] = 0.0;
        bar_q[cell] = 0.0;
    }
    status = visco_sh_harmonic_pair_vjp(
            mu[C6A_CENTER], mu[C6A_EAST], bar_output[C6A_MU_X],
            &bar_mu[C6A_CENTER], &bar_mu[C6A_EAST]);
    if (status != 0) return status;
    status = visco_sh_harmonic_pair_vjp(
            mu[C6A_CENTER], mu[C6A_SOUTH], bar_output[C6A_MU_Y],
            &bar_mu[C6A_CENTER], &bar_mu[C6A_SOUTH]);
    if (status != 0) return status;
    visco_sh_av_tau_local_vjp(bar_output[C6A_TAU_X],
                              bar_output[C6A_TAU_Y], bar_tau);
    for (cell = 0; cell < 4; ++cell) {
        status = tau_value(q[cell], mapping, &tau);
        if (status != 0) return status;
        derivative = mapping->mode == Q_PARAMETERIZATION_LEGACY
                   ? -2.0 / (q[cell] * q[cell])
                   : -mapping->inverse_tau_per_q * tau * tau;
        bar_q[cell] = derivative * bar_tau[cell];
        if (invmat1 == 1) {
            bar_primary[cell] = 2.0 * rho[cell] * primary[cell]
                              * bar_mu[cell];
            bar_rho[cell] = primary[cell] * primary[cell] * bar_mu[cell];
        } else {
            bar_primary[cell] = bar_mu[cell];
        }
    }
    bar_rho[C6A_CENTER] += visco_sh_rhoi_vjp(
            rho[C6A_CENTER], bar_output[C6A_RHOI]);
    return 0;
}
