/* SH-OPT-SCALE-01 O6 harness: CPML mask -> scaled direction -> trial state.
 *
 * Contract: docs/sh_opt_scale_01_contract.md (revision 2), sections 1-3, O6.
 * For every local block position of a 2x2 decomposition, every FREE_SURF and
 * BOUNDARY flag and both scaling modes, the production mask builder, the
 * production scaled optimizer boundary and the unchanged production
 * trial-state builder are composed.  For a range of alphas every cell with
 * M = 0 must keep the Base primary/rho/Q bitwise; its Tau must be the
 * production Q->tau image of the unchanged Base Q.  Only counters are
 * reported; the Python oracle evaluates them.
 */

#include "fd.h"

#include <stdint.h>

#define NXL 10
#define NYL 9
#define FWL 3
#define SENTINEL (-98765.25f)
#define ALPHA_COUNT 9

void err(char err_text[]) {
    fprintf(stderr, "unexpected err(): %s\n", err_text);
    exit(91);
}

static uint32_t lcg_state = 20261119u;

static float uniform(float lower, float upper) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lower + (upper - lower) * ((float)(lcg_state >> 8) / 16777216.0f);
}

static float **allocate_local(float value) {
    float **matrix = (float **)malloc((size_t)(NYL + 2) * sizeof(*matrix));
    int i, j;

    if (matrix == NULL) exit(4);
    for (j = 0; j < NYL + 2; ++j) {
        matrix[j] = (float *)malloc((size_t)(NXL + 2) * sizeof(float));
        if (matrix[j] == NULL) exit(4);
        for (i = 0; i < NXL + 2; ++i) matrix[j][i] = value;
    }
    return matrix;
}

static int same_bits(float left, float right) {
    return memcmp(&left, &right, sizeof(float)) == 0;
}

int main(int argc, char **argv) {
    const float alphas[ALPHA_COUNT] = {
        0.0f, 1.0e-6f, 1.0e-3f, 0.05f, 0.5f, 1.0f, 3.0f, 1.0e3f, 1.0e30f};
    float fl[2] = {0.0f, 10.0f};
    struct q_tau_mapping mapping;
    struct visco_sh_exact_optimizer_boundary boundary;
    struct visco_sh_exact_step_scaling scaling;
    struct visco_sh_exact_trial_state_request trial;
    float **mask = allocate_local(SENTINEL);
    float **base[3], **grad[3], **step[3], **out[4];
    long configurations = 0, mask_failures = 0, boundary_failures = 0;
    long trial_failures = 0, masked_checked = 0, masked_mismatches = 0;
    long masked_step_nonzero = 0, unmasked_changed = 0;
    long min_masked = -1, masked_here;
    int mode, pos_x, pos_y, free_surf, boundary_flag, a, i, j, k, status;

    MPI_Init(&argc, &argv);
    init_q_tau_mapping(&mapping, Q_PARAMETERIZATION_PHYSICAL, 1, fl, 1.0f, 10.0f, 1.0f);
    for (k = 0; k < 3; ++k) {
        base[k] = allocate_local(SENTINEL);
        grad[k] = allocate_local(SENTINEL);
        step[k] = allocate_local(SENTINEL);
    }
    for (k = 0; k < 4; ++k) out[k] = allocate_local(SENTINEL);

    for (mode = 1; mode <= 2; ++mode)
    for (pos_y = 0; pos_y <= 1; ++pos_y)
    for (pos_x = 0; pos_x <= 1; ++pos_x)
    for (free_surf = 0; free_surf <= 1; ++free_surf)
    for (boundary_flag = 0; boundary_flag <= 1; ++boundary_flag) {
        configurations++;
        status = visco_sh_exact_build_update_mask(mask, NXL, NYL, pos_x, pos_y,
                2, 2, FWL, free_surf, boundary_flag);
        if (status != 0) { mask_failures++; continue; }

        masked_here = 0;
        for (j = 1; j <= NYL; ++j) {
            for (i = 1; i <= NXL; ++i) {
                float boost = (mask[j][i] == 0.0f) ? 50.0f : 1.0f;
                if (mask[j][i] == 0.0f) masked_here++;
                base[0][j][i] = uniform(1500.0f, 2500.0f);
                base[1][j][i] = uniform(1800.0f, 2600.0f);
                base[2][j][i] = uniform(30.0f, 90.0f);
                grad[0][j][i] = boost * uniform(-3.0e-4f, 3.0e-4f) + 1.0e-6f;
                grad[1][j][i] = boost * uniform(-2.0e-6f, 2.0e-6f) - 1.0e-8f;
                grad[2][j][i] = boost * uniform(-5.0e-2f, 5.0e-2f) + 1.0e-4f;
            }
        }
        if (min_masked < 0 || masked_here < min_masked) min_masked = masked_here;

        memset(&boundary, 0, sizeof(boundary));
        boundary.nx = NXL;
        boundary.ny = NYL;
        boundary.grad_raw_primary = grad[0];
        boundary.grad_raw_rho = grad[1];
        boundary.grad_raw_q = grad[2];
        boundary.optimizer_step_primary = step[0];
        boundary.optimizer_step_rho = step[1];
        boundary.optimizer_step_q = step[2];
        memset(&scaling, 0, sizeof(scaling));
        scaling.scaling_mode = mode;
        scaling.base_primary = base[0];
        scaling.base_rho = base[1];
        scaling.base_q = base[2];
        scaling.update_mask = mask;
        scaling.weight_primary = 1.0f;
        scaling.weight_rho = 0.7f;
        scaling.weight_q = 1.3f;
        status = visco_sh_exact_build_scaled_steepest_step(&boundary, &scaling);
        if (status != 0) { boundary_failures++; continue; }
        for (j = 1; j <= NYL; ++j)
            for (i = 1; i <= NXL; ++i)
                if (mask[j][i] == 0.0f)
                    for (k = 0; k < 3; ++k)
                        if (step[k][j][i] != 0.0f) masked_step_nonzero++;

        for (a = 0; a < ALPHA_COUNT; ++a) {
            memset(&trial, 0, sizeof(trial));
            trial.nx = NXL;
            trial.ny = NYL;
            trial.alpha = alphas[a];
            trial.primary_bounds_enabled = 1;
            trial.primary_lower = 1.0f;
            trial.primary_upper = 1.0e5f;
            trial.rho_lower = 1.0f;
            trial.rho_upper = 1.0e5f;
            trial.q_lower = 1.0f;
            trial.q_upper = 1.0e4f;
            trial.q_mapping = &mapping;
            trial.base_primary = base[0];
            trial.base_rho = base[1];
            trial.base_q = base[2];
            trial.optimizer_step_primary = step[0];
            trial.optimizer_step_rho = step[1];
            trial.optimizer_step_q = step[2];
            trial.trial_primary = out[0];
            trial.trial_rho = out[1];
            trial.trial_q = out[2];
            trial.trial_tau = out[3];
            status = visco_sh_exact_build_trial_parameter_state(&trial);
            if (status != 0) { trial_failures++; continue; }
            for (j = 1; j <= NYL; ++j) {
                for (i = 1; i <= NXL; ++i) {
                    if (mask[j][i] == 0.0f) {
                        masked_checked++;
                        if (!same_bits(out[0][j][i], base[0][j][i]) ||
                                !same_bits(out[1][j][i], base[1][j][i]) ||
                                !same_bits(out[2][j][i], base[2][j][i]) ||
                                !same_bits(out[3][j][i], q_to_tau(base[2][j][i], &mapping)))
                            masked_mismatches++;
                    } else if (alphas[a] > 0.0f && alphas[a] <= 1.0f) {
                        if (!same_bits(out[0][j][i], base[0][j][i]) ||
                                !same_bits(out[1][j][i], base[1][j][i]) ||
                                !same_bits(out[2][j][i], base[2][j][i]))
                            unmasked_changed++;
                    }
                }
            }
        }
    }

    printf("{\"configurations\":%ld,\"alphas\":%d,\"mask_failures\":%ld,"
           "\"boundary_failures\":%ld,\"trial_failures\":%ld,"
           "\"masked_cells_checked\":%ld,\"masked_mismatches\":%ld,"
           "\"masked_step_nonzero\":%ld,\"unmasked_changed\":%ld,"
           "\"min_masked_per_configuration\":%ld}\n",
           configurations, ALPHA_COUNT, mask_failures, boundary_failures,
           trial_failures, masked_checked, masked_mismatches,
           masked_step_nonzero, unmasked_changed, min_masked);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
