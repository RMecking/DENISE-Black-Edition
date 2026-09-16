/* Exact reverse-time composition of the fixed-material viscoelastic SH
 * full-state transpose step.  This unit is intentionally not connected to
 * the active SH FWI dispatcher and does not accumulate material gradients.
 */

#include "fd.h"

#include <stddef.h>

static int unsupported_cpml_overlap(
        const struct visco_sh_full_step_config *cfg) {
    if ((cfg->fw > 0) && (!cfg->boundary) && (cfg->nproc_x == 1) &&
            (cfg->nx <= 2 * cfg->fw)) return 1;
    if ((cfg->fw > 0) && (!cfg->free_surface) &&
            (cfg->nproc_y == 1) && (cfg->ny <= 2 * cfg->fw)) return 1;
    return 0;
}

static int state_complete_for_driver(
        const struct visco_sh_full_state *state, int fw) {
    if ((state == NULL) || (state->vz == NULL) || (state->sxz == NULL) ||
            (state->syz == NULL) || (state->r == NULL) ||
            (state->q == NULL)) return 0;
    if ((fw > 0) && ((state->psi_sxz_x == NULL) ||
            (state->psi_syz_y == NULL) || (state->psi_vzx == NULL) ||
            (state->psi_vzy == NULL))) return 0;
    return 1;
}

static int states_distinct_for_driver(
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

int visco_sh_reverse_time_adjoint(
        const struct visco_sh_full_step_config *base_config,
        int nsteps,
        const double *bar_receiver_series,
        struct visco_sh_full_state *bar_terminal_work,
        struct visco_sh_full_state *bar_initial,
        struct visco_sh_full_state *scratch,
        double *bar_signal_series) {
    struct visco_sh_full_step_config step_config;
    struct visco_sh_full_state *current, *previous;
    int n, source, status;

    if ((base_config == NULL) || (nsteps < 1) ||
            ((base_config->nrec > 0) && (bar_receiver_series == NULL)) ||
            ((base_config->nsrc > 0) && (bar_signal_series == NULL)))
        return -1;
    if (!state_complete_for_driver(bar_terminal_work, base_config->fw) ||
            !state_complete_for_driver(bar_initial, base_config->fw) ||
            !state_complete_for_driver(scratch, base_config->fw) ||
            !states_distinct_for_driver(
                bar_terminal_work, bar_initial, base_config->fw) ||
            !states_distinct_for_driver(
                bar_terminal_work, scratch, base_config->fw) ||
            !states_distinct_for_driver(
                bar_initial, scratch, base_config->fw)) return -1;

    /* Match the locked C5a restriction before consuming any cotangent. */
    if (unsupported_cpml_overlap(base_config)) return -2;

    /* This time series is an output of the driver.  Establish deterministic
     * overwrite semantics here instead of relying on the locked step VJP's
     * current handling of its additive source-cotangent output. */
    for (n = 0; n < nsteps; ++n)
        for (source = 0; source < base_config->nsrc; ++source)
            bar_signal_series[(size_t)n * base_config->nsrc + source] = 0.0;

    current = bar_terminal_work;
    for (n = nsteps - 1; n >= 0; --n) {
        step_config = *base_config;
        step_config.bar_receiver = (base_config->nrec > 0) ?
                bar_receiver_series + (size_t)n * base_config->nrec : NULL;

        /* For all non-final reverse iterations, alternate the two mutable
         * workspaces.  The last transpose always writes bar_initial, making
         * result ownership independent of odd/even nsteps. */
        if (n == 0)
            previous = bar_initial;
        else
            previous = (current == scratch) ?
                    bar_terminal_work : scratch;

        status = visco_sh_full_state_adjoint_step(
                &step_config, current, previous,
                (base_config->nsrc > 0) ?
                    bar_signal_series + (size_t)n * base_config->nsrc : NULL);
        if (status != 0) return status;
        current = previous;
    }
    return 0;
}
