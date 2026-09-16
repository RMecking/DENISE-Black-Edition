/* Exact local transpose primitives for the SH velocity-side update.
 *
 * These routines intentionally do not dispatch a reverse-time solver.  They
 * expose the pointwise velocity/CPML/spatial transpose and the distinct
 * Euclidean receiver and physical-source operator transposes needed by a
 * later global adjoint driver.
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

int visco_sh_velocity_cpml_select_x(
        int i, int nx2, int fw, int boundary, int pos_x, int nproc_x,
        const float *K_x, const float *a_x, const float *b_x,
        int *active, int *aux_index, double *K, double *a, double *b) {
    int index = -1;

    if ((active == NULL) || (aux_index == NULL) || (K == NULL) ||
            (a == NULL) || (b == NULL) || (nproc_x < 1) ||
            (pos_x < 0) || (pos_x >= nproc_x)) return -1;
    inactive_cpml(active, aux_index, K, a, b);
    if (fw <= 0) return 0;
    if ((K_x == NULL) || (a_x == NULL) || (b_x == NULL)) return -1;

    if ((!boundary) && (pos_x == 0) && (i <= fw)) index = i;
    if ((!boundary) && (pos_x == nproc_x - 1) &&
            (i >= nx2 - fw + 1)) {
        if (index >= 0) return -2;
        index = i - nx2 + 2 * fw;
    }
    if (index < 0) return 0;
    if ((index < 1) || (index > 2 * fw) ||
            !finite_cpml(K_x[index], a_x[index], b_x[index])) return -1;

    *active = 1;
    *aux_index = index;
    *K = K_x[index];
    *a = a_x[index];
    *b = b_x[index];
    return 0;
}

int visco_sh_velocity_cpml_select_y(
        int j, int ny2, int fw, int free_surface, int pos_y, int nproc_y,
        const float *K_y, const float *a_y, const float *b_y,
        int *active, int *aux_index, double *K, double *a, double *b) {
    int index = -1;

    if ((active == NULL) || (aux_index == NULL) || (K == NULL) ||
            (a == NULL) || (b == NULL) || (nproc_y < 1) ||
            (pos_y < 0) || (pos_y >= nproc_y)) return -1;
    inactive_cpml(active, aux_index, K, a, b);
    if (fw <= 0) return 0;
    if ((K_y == NULL) || (a_y == NULL) || (b_y == NULL)) return -1;

    if ((pos_y == 0) && (!free_surface) && (j <= fw)) index = j;
    if ((pos_y == nproc_y - 1) && (j >= ny2 - fw + 1)) {
        if (index >= 0) return -2;
        index = j - ny2 + 2 * fw;
    }
    if (index < 0) return 0;
    if ((index < 1) || (index > 2 * fw) ||
            !finite_cpml(K_y[index], a_y[index], b_y[index])) return -1;

    *active = 1;
    *aux_index = index;
    *K = K_y[index];
    *a = a_y[index];
    *b = b_y[index];
    return 0;
}

int visco_sh_velocity_cpml_local_vjp(
        int active, double K, double a, double b, double bar_q,
        double bar_psi_next, double *bar_d_raw, double *bar_psi_prev) {
    double t_psi;

    if ((bar_d_raw == NULL) || (bar_psi_prev == NULL) ||
            !isfinite(bar_q) || !isfinite(bar_psi_next)) return -1;
    if (!active) {
        if (bar_psi_next != 0.0) return -1;
        *bar_d_raw += bar_q;
        return 0;
    }
    if (!finite_cpml(K, a, b)) return -1;

    t_psi = bar_psi_next + bar_q;
    *bar_psi_prev += b * t_psi;
    *bar_d_raw += bar_q / K + a * t_psi;
    return 0;
}

int visco_sh_velocity_spatial_local_vjp(
        int fdorder, const float *hc, double bar_dx, double bar_dy,
        double *bar_sxz_patch, double *bar_syz_patch, int patch_rows,
        int patch_stride, int center_row, int center_col) {
    double scale;
    int half_order, m;

    if ((fdorder < 2) || (fdorder > 12) || (fdorder % 2 != 0) ||
            (hc == NULL) || !isfinite(bar_dx) || !isfinite(bar_dy) ||
            (bar_sxz_patch == NULL) || (bar_syz_patch == NULL)) return -1;
    half_order = fdorder / 2;
    if ((patch_rows < 2 * half_order + 1) ||
            (patch_stride < 2 * half_order + 1) ||
            (center_row < half_order) || (center_col < half_order) ||
            (center_row + half_order >= patch_rows) ||
            (center_col + half_order >= patch_stride)) return -1;

    for (m = 1; m <= half_order; ++m) {
        if (!isfinite(hc[m])) return -1;
        scale = (double)hc[m];
        bar_sxz_patch[center_row * patch_stride + center_col + (m - 1)] +=
                scale * bar_dx;
        bar_sxz_patch[center_row * patch_stride + center_col - m] -=
                scale * bar_dx;
        bar_syz_patch[(center_row + (m - 1)) * patch_stride + center_col] +=
                scale * bar_dy;
        bar_syz_patch[(center_row - m) * patch_stride + center_col] -=
                scale * bar_dy;
    }
    return 0;
}

int update_v_PML_SH_adjoint_point(
        int fdorder, double dt, double dh, float rhoi, const float *hc,
        const int cpml_active[2], const double cpml_K[2],
        const double cpml_a[2], const double cpml_b[2],
        double bar_vz_next, const double bar_psi_next[2],
        double *bar_vz_prev, double bar_psi_prev[2],
        double *bar_sxz_patch, double *bar_syz_patch, int patch_rows,
        int patch_stride, int center_row, int center_col) {
    double alpha, bar_q, bar_raw[2] = {0.0, 0.0};
    int axis, status;

    if (!(dt > 0.0) || !(dh > 0.0) || !isfinite(dt) || !isfinite(dh) ||
            !(rhoi > 0.0f) || !isfinite(rhoi) || (hc == NULL) ||
            (cpml_active == NULL) || (cpml_K == NULL) ||
            (cpml_a == NULL) || (cpml_b == NULL) ||
            !isfinite(bar_vz_next) || (bar_psi_next == NULL) ||
            (bar_vz_prev == NULL) || (bar_psi_prev == NULL) ||
            (bar_psi_next == bar_psi_prev)) return -1;

    alpha = dt * (double)rhoi / dh;
    *bar_vz_prev += bar_vz_next;
    bar_q = alpha * bar_vz_next;
    for (axis = 0; axis < 2; ++axis) {
        status = visco_sh_velocity_cpml_local_vjp(
                cpml_active[axis], cpml_K[axis], cpml_a[axis],
                cpml_b[axis], bar_q, bar_psi_next[axis],
                &bar_raw[axis], &bar_psi_prev[axis]);
        if (status != 0) return status;
    }
    return visco_sh_velocity_spatial_local_vjp(
            fdorder, hc, bar_raw[0], bar_raw[1], bar_sxz_patch,
            bar_syz_patch, patch_rows, patch_stride, center_row, center_col);
}

int visco_sh_receiver_velocity_sampling_vjp(
        int nrec, const int *rec_x, const int *rec_y,
        const double *bar_data, double *bar_vz, int rows, int stride) {
    int receiver, x, y;

    if ((nrec < 0) || (rows < 1) || (stride < 1) ||
            (bar_vz == NULL)) return -1;
    if ((nrec > 0) && ((rec_x == NULL) || (rec_y == NULL) ||
            (bar_data == NULL))) return -1;
    for (receiver = 0; receiver < nrec; ++receiver) {
        x = rec_x[receiver];
        y = rec_y[receiver];
        if ((x < 0) || (x >= stride) || (y < 0) || (y >= rows) ||
                !isfinite(bar_data[receiver])) return -1;
        bar_vz[y * stride + x] += bar_data[receiver];
    }
    return 0;
}

int visco_sh_velocity_source_injection_vjp(
        int rows, int stride, const double *bar_vz_after,
        double *bar_vz_before, int nsrc, const int *src_x,
        const int *src_y, const int *source_type, double *bar_signal) {
    int cell, cells, source, x, y;

    if ((rows < 1) || (stride < 1) || (nsrc < 0) ||
            (bar_vz_after == NULL) || (bar_vz_before == NULL) ||
            (bar_vz_after == bar_vz_before)) return -1;
    if ((nsrc > 0) && ((src_x == NULL) || (src_y == NULL) ||
            (source_type == NULL) || (bar_signal == NULL))) return -1;
    cells = rows * stride;
    for (cell = 0; cell < cells; ++cell) {
        if (!isfinite(bar_vz_after[cell])) return -1;
        bar_vz_before[cell] += bar_vz_after[cell];
    }
    for (source = 0; source < nsrc; ++source) {
        x = src_x[source];
        y = src_y[source];
        if ((x < 0) || (x >= stride) || (y < 0) || (y >= rows)) return -1;
        if (source_type[source] == 1) {
            bar_signal[source] += bar_vz_after[y * stride + x];
        }
    }
    return 0;
}
