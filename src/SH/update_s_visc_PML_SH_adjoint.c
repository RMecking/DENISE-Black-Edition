/* Exact local transpose of the viscoelastic SH stress-side block.
 *
 * This file does not alter or dispatch the production reverse-time loop.  It
 * provides the point operation needed by that future loop: the locked local
 * GSLS VJP, followed by the exact CPML-state transpose and the scatter form
 * of the staggered spatial-derivative transpose.
 *
 * CPML convention: bar_psi_next is an incoming next-time adjoint and
 * bar_psi_prev is a distinct accumulating output.  The direct contribution
 * of corrected strain is combined before the recursion is transposed.
 */

#include "fd.h"

#include <math.h>
#include <stddef.h>

static int finite_cpml(double K, double a, double b) {
    return (K > 0.0) && isfinite(K) && isfinite(a) && isfinite(b);
}

static void inactive_cpml(
        int *active, int *aux_index, double *K, double *a, double *b) {
    *active = 0;
    *aux_index = -1;
    *K = 1.0;
    *a = 0.0;
    *b = 0.0;
}

int visco_sh_stress_cpml_select_x(
        int i, int nx2, int fw, int boundary, int pos_x, int nproc_x,
        const float *K_x_half, const float *a_x_half,
        const float *b_x_half, int *active, int *aux_index,
        double *K, double *a, double *b) {
    int index = -1;

    if ((active == NULL) || (aux_index == NULL) || (K == NULL) ||
            (a == NULL) || (b == NULL) || (nproc_x < 1) ||
            (pos_x < 0) || (pos_x >= nproc_x)) return -1;
    inactive_cpml(active, aux_index, K, a, b);
    if (fw <= 0) return 0;
    if ((K_x_half == NULL) || (a_x_half == NULL) ||
            (b_x_half == NULL)) return -1;

    if ((!boundary) && (pos_x == 0) && (i <= fw)) index = i;
    if ((!boundary) && (pos_x == nproc_x - 1) &&
            (i >= nx2 - fw + 1)) {
        if (index >= 0) return -2;
        index = i - nx2 + 2 * fw;
    }
    if (index < 0) return 0;
    if ((index < 1) || (index > 2 * fw) ||
            !finite_cpml(K_x_half[index], a_x_half[index],
                         b_x_half[index])) return -1;

    *active = 1;
    *aux_index = index;
    *K = K_x_half[index];
    *a = a_x_half[index];
    *b = b_x_half[index];
    return 0;
}

int visco_sh_stress_cpml_select_y(
        int j, int ny2, int fw, int free_surface, int pos_y, int nproc_y,
        const float *K_y, const float *a_y, const float *b_y,
        const float *K_y_half, const float *a_y_half,
        const float *b_y_half, int *active, int *aux_index,
        double *K, double *a, double *b) {
    int index = -1;
    int use_half = 0;

    if ((active == NULL) || (aux_index == NULL) || (K == NULL) ||
            (a == NULL) || (b == NULL) || (nproc_y < 1) ||
            (pos_y < 0) || (pos_y >= nproc_y)) return -1;
    inactive_cpml(active, aux_index, K, a, b);
    if (fw <= 0) return 0;

    if ((pos_y == 0) && (!free_surface) && (j <= fw)) index = j;
    if ((pos_y == nproc_y - 1) && (j >= ny2 - fw + 1)) {
        if (index >= 0) return -2;
        index = j - ny2 + 2 * fw;
        use_half = 1;
    }
    if (index < 0) return 0;
    if ((index < 1) || (index > 2 * fw)) return -1;

    if (use_half) {
        if ((K_y_half == NULL) || (a_y_half == NULL) ||
                (b_y_half == NULL) ||
                !finite_cpml(K_y_half[index], a_y_half[index],
                             b_y_half[index])) return -1;
        *K = K_y_half[index];
        *a = a_y_half[index];
        *b = b_y_half[index];
    } else {
        if ((K_y == NULL) || (a_y == NULL) || (b_y == NULL) ||
                !finite_cpml(K_y[index], a_y[index],
                             b_y[index])) return -1;
        *K = K_y[index];
        *a = a_y[index];
        *b = b_y[index];
    }
    *active = 1;
    *aux_index = index;
    return 0;
}

int visco_sh_stress_cpml_local_vjp(
        int active, double K, double a, double b, double bar_e,
        double bar_psi_next, double *bar_e_raw, double *bar_psi_prev) {
    double t_psi;

    if ((bar_e_raw == NULL) || (bar_psi_prev == NULL) ||
            !isfinite(bar_e) || !isfinite(bar_psi_next)) return -1;
    if (!active) {
        if (bar_psi_next != 0.0) return -1;
        *bar_e_raw += bar_e;
        return 0;
    }
    if (!finite_cpml(K, a, b)) return -1;

    t_psi = bar_psi_next + bar_e;
    *bar_psi_prev += b * t_psi;
    *bar_e_raw += bar_e / K + a * t_psi;
    return 0;
}

int visco_sh_stress_spatial_local_vjp(
        int fdorder, double dh, const float *hc, double bar_e_raw_x,
        double bar_e_raw_y, double *bar_vz_patch, int patch_rows,
        int patch_stride, int center_row, int center_col) {
    double scale;
    int half_order, m;

    if ((fdorder < 2) || (fdorder > 12) || (fdorder % 2 != 0) ||
            !(dh > 0.0) || !isfinite(dh) || (hc == NULL) ||
            !isfinite(bar_e_raw_x) || !isfinite(bar_e_raw_y) ||
            (bar_vz_patch == NULL)) return -1;
    half_order = fdorder / 2;
    if ((patch_rows < 2 * half_order + 1) ||
            (patch_stride < 2 * half_order + 1) ||
            (center_row < half_order) || (center_col < half_order) ||
            (center_row + half_order >= patch_rows) ||
            (center_col + half_order >= patch_stride)) return -1;

    for (m = 1; m <= half_order; ++m) {
        if (!isfinite(hc[m])) return -1;
        scale = (double)hc[m] / dh;
        bar_vz_patch[center_row * patch_stride + center_col + m] +=
                scale * bar_e_raw_x;
        bar_vz_patch[center_row * patch_stride + center_col - (m - 1)] -=
                scale * bar_e_raw_x;
        bar_vz_patch[(center_row + m) * patch_stride + center_col] +=
                scale * bar_e_raw_y;
        bar_vz_patch[(center_row - (m - 1)) * patch_stride + center_col] -=
                scale * bar_e_raw_y;
    }
    return 0;
}

int update_s_visc_PML_SH_adjoint_point(
        int fdorder, int mechanisms, double dh, double dt,
        const float *hc, const int cpml_active[2],
        const double cpml_K[2], const double cpml_a[2],
        const double cpml_b[2], const double strain[2],
        const double bar_stress_next[2],
        const double *bar_memory_x_next,
        const double *bar_memory_y_next, const double forward_f[2],
        const double *forward_a_x, const double *forward_a_y,
        const double *forward_c_x, const double *forward_c_y,
        const double f_tau[2], const double f_modulus[2],
        const double *c_tau_x, const double *c_tau_y,
        const double *c_modulus_x, const double *c_modulus_y,
        const double bar_psi_next[2], double bar_stress_prev[2],
        double *bar_memory_x_prev, double *bar_memory_y_prev,
        double bar_psi_prev[2], double *bar_vz_patch, int patch_rows,
        int patch_stride, int center_row, int center_col,
        double g_tau[2], double g_modulus[2]) {
    const double *bar_memory_next[2];
    const double *forward_a[2], *forward_c[2];
    const double *c_tau[2], *c_modulus[2];
    double *bar_memory_prev[2];
    double bar_e[2] = {0.0, 0.0};
    double bar_e_raw[2] = {0.0, 0.0};
    int axis, status;

    if ((mechanisms < 1) || (cpml_active == NULL) ||
            (cpml_K == NULL) || (cpml_a == NULL) || (cpml_b == NULL) ||
            (strain == NULL) || (bar_stress_next == NULL) ||
            (forward_f == NULL) || (f_tau == NULL) ||
            (f_modulus == NULL) || (bar_psi_next == NULL) ||
            (bar_stress_prev == NULL) || (bar_psi_prev == NULL) ||
            (g_tau == NULL) || (g_modulus == NULL) ||
            (bar_psi_next == bar_psi_prev)) return -1;

    bar_memory_next[0] = bar_memory_x_next;
    bar_memory_next[1] = bar_memory_y_next;
    bar_memory_prev[0] = bar_memory_x_prev;
    bar_memory_prev[1] = bar_memory_y_prev;
    forward_a[0] = forward_a_x;
    forward_a[1] = forward_a_y;
    forward_c[0] = forward_c_x;
    forward_c[1] = forward_c_y;
    c_tau[0] = c_tau_x;
    c_tau[1] = c_tau_y;
    c_modulus[0] = c_modulus_x;
    c_modulus[1] = c_modulus_y;

    for (axis = 0; axis < 2; ++axis) {
        status = visco_sh_gsls_local_vjp(
                mechanisms, dt, strain[axis], bar_stress_next[axis],
                bar_memory_next[axis], forward_f[axis], forward_a[axis],
                forward_c[axis], f_tau[axis], f_modulus[axis], c_tau[axis],
                c_modulus[axis], &bar_stress_prev[axis],
                bar_memory_prev[axis], &bar_e[axis], &g_tau[axis],
                &g_modulus[axis]);
        if (status != 0) return status;
        status = visco_sh_stress_cpml_local_vjp(
                cpml_active[axis], cpml_K[axis], cpml_a[axis],
                cpml_b[axis], bar_e[axis], bar_psi_next[axis],
                &bar_e_raw[axis], &bar_psi_prev[axis]);
        if (status != 0) return status;
    }

    return visco_sh_stress_spatial_local_vjp(
            fdorder, dh, hc, bar_e_raw[0], bar_e_raw[1], bar_vz_patch,
            patch_rows, patch_stride, center_row, center_col);
}
