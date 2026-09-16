/* Inactive until a later C8c integration gate. */

#include "fd.h"

int visco_sh_exact_build_steepest_subtractive_step(
        const struct visco_sh_exact_optimizer_boundary *boundary) {
    int i, j;

    if (boundary == NULL) return -1;
    if ((boundary->nx < 1) || (boundary->ny < 1)) return -2;
    if ((boundary->grad_raw_primary == NULL) ||
            (boundary->grad_raw_rho == NULL) ||
            (boundary->grad_raw_q == NULL) ||
            (boundary->optimizer_step_primary == NULL) ||
            (boundary->optimizer_step_rho == NULL) ||
            (boundary->optimizer_step_q == NULL)) return -3;

    /* Baseline subtractive adapter: p = g_raw, while the model trajectory
     * direction is -p.  The third channel is the physical-Q derivative. */
    for (j = 1; j <= boundary->ny; ++j) {
        for (i = 1; i <= boundary->nx; ++i) {
            boundary->optimizer_step_primary[j][i] =
                    boundary->grad_raw_primary[j][i];
            boundary->optimizer_step_rho[j][i] =
                    boundary->grad_raw_rho[j][i];
            boundary->optimizer_step_q[j][i] =
                    boundary->grad_raw_q[j][i];
        }
    }

    return 0;
}
