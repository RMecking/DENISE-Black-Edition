/* Direct contract harness for the inactive C8c physical-Q trial-state helper. */

#include "fd.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NX 2
#define NY 2
#define WIDTH (NX + 2)
#define HEIGHT (NY + 2)
#define OUTPUT_SENTINEL (-98765.25f)
#define INPUT_HALO_SENTINEL (31415.75f)

#define REQUIRE(condition, description) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "trial-state contract failed: %s\n", description); \
            return 1; \
        } \
    } while (0)

/* The real q_parameterization implementation only reaches err on an invalid
 * mapping/input; all exercised mapping calls are valid. */
void err(char err_text[]) {
    fprintf(stderr, "unexpected q-mapping error: %s\n", err_text);
    exit(91);
}

static float **allocate_matrix(float value) {
    float **matrix = (float **)malloc((size_t)HEIGHT * sizeof(*matrix));
    int i, j;

    if (matrix == NULL) return NULL;
    for (j = 0; j < HEIGHT; ++j) {
        matrix[j] = (float *)malloc((size_t)WIDTH * sizeof(*matrix[j]));
        if (matrix[j] == NULL) {
            while (j-- > 0) free(matrix[j]);
            free(matrix);
            return NULL;
        }
        for (i = 0; i < WIDTH; ++i) matrix[j][i] = value;
    }
    return matrix;
}

static void release_matrix(float **matrix) {
    int j;

    if (matrix == NULL) return;
    for (j = 0; j < HEIGHT; ++j) free(matrix[j]);
    free(matrix);
}

static void fill_matrix(float **matrix, float value) {
    int i, j;

    for (j = 0; j < HEIGHT; ++j)
        for (i = 0; i < WIDTH; ++i) matrix[j][i] = value;
}

static void copy_matrix(float **target, float **source) {
    int j;

    for (j = 0; j < HEIGHT; ++j)
        memcpy(target[j], source[j], (size_t)WIDTH * sizeof(float));
}

static int matrices_equal(float **left, float **right) {
    int j;

    for (j = 0; j < HEIGHT; ++j)
        if (memcmp(left[j], right[j], (size_t)WIDTH * sizeof(float)) != 0)
            return 0;
    return 1;
}

static int matrix_is_sentinel(float **matrix) {
    int i, j;

    for (j = 0; j < HEIGHT; ++j)
        for (i = 0; i < WIDTH; ++i)
            if (matrix[j][i] != OUTPUT_SENTINEL) return 0;
    return 1;
}

static int halos_are_sentinel(float **matrix) {
    int i, j;

    for (j = 0; j < HEIGHT; ++j)
        for (i = 0; i < WIDTH; ++i)
            if ((j == 0) || (j == NY + 1) || (i == 0) || (i == NX + 1))
                if (matrix[j][i] != OUTPUT_SENTINEL) return 0;
    return 1;
}

struct fixture {
    struct visco_sh_exact_trial_state_request request;
    struct q_tau_mapping mapping;
    float **base_primary, **base_rho, **base_q;
    float **step_primary, **step_rho, **step_q;
    float **trial_primary, **trial_rho, **trial_q, **trial_tau;
    float **base_primary_before, **base_rho_before, **base_q_before;
    float **step_primary_before, **step_rho_before, **step_q_before;
    struct q_tau_mapping mapping_before;
};

static int fixture_allocate(struct fixture *fixture) {
    float fl[2] = {0.0f, 10.0f};

    memset(fixture, 0, sizeof(*fixture));
    fixture->base_primary = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->base_rho = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->base_q = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->step_primary = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->step_rho = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->step_q = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->trial_primary = allocate_matrix(OUTPUT_SENTINEL);
    fixture->trial_rho = allocate_matrix(OUTPUT_SENTINEL);
    fixture->trial_q = allocate_matrix(OUTPUT_SENTINEL);
    fixture->trial_tau = allocate_matrix(OUTPUT_SENTINEL);
    fixture->base_primary_before = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->base_rho_before = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->base_q_before = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->step_primary_before = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->step_rho_before = allocate_matrix(INPUT_HALO_SENTINEL);
    fixture->step_q_before = allocate_matrix(INPUT_HALO_SENTINEL);
    if (!fixture->base_primary || !fixture->base_rho || !fixture->base_q ||
            !fixture->step_primary || !fixture->step_rho || !fixture->step_q ||
            !fixture->trial_primary || !fixture->trial_rho || !fixture->trial_q ||
            !fixture->trial_tau || !fixture->base_primary_before ||
            !fixture->base_rho_before || !fixture->base_q_before ||
            !fixture->step_primary_before || !fixture->step_rho_before ||
            !fixture->step_q_before) return 0;
    init_q_tau_mapping(&fixture->mapping, Q_PARAMETERIZATION_PHYSICAL, 1,
            fl, 1.0f, 10.0f, 1.0f);
    fixture->request.nx = NX;
    fixture->request.ny = NY;
    fixture->request.alpha = 1.0f;
    fixture->request.primary_bounds_enabled = 1;
    fixture->request.primary_lower = 1.0f;
    fixture->request.primary_upper = 1000.0f;
    fixture->request.rho_lower = 1.0f;
    fixture->request.rho_upper = 10000.0f;
    fixture->request.q_lower = 1.0f;
    fixture->request.q_upper = 1000.0f;
    fixture->request.q_mapping = &fixture->mapping;
    fixture->request.base_primary = fixture->base_primary;
    fixture->request.base_rho = fixture->base_rho;
    fixture->request.base_q = fixture->base_q;
    fixture->request.optimizer_step_primary = fixture->step_primary;
    fixture->request.optimizer_step_rho = fixture->step_rho;
    fixture->request.optimizer_step_q = fixture->step_q;
    fixture->request.trial_primary = fixture->trial_primary;
    fixture->request.trial_rho = fixture->trial_rho;
    fixture->request.trial_q = fixture->trial_q;
    fixture->request.trial_tau = fixture->trial_tau;
    return 1;
}

static void fixture_release(struct fixture *fixture) {
    release_matrix(fixture->base_primary);
    release_matrix(fixture->base_rho);
    release_matrix(fixture->base_q);
    release_matrix(fixture->step_primary);
    release_matrix(fixture->step_rho);
    release_matrix(fixture->step_q);
    release_matrix(fixture->trial_primary);
    release_matrix(fixture->trial_rho);
    release_matrix(fixture->trial_q);
    release_matrix(fixture->trial_tau);
    release_matrix(fixture->base_primary_before);
    release_matrix(fixture->base_rho_before);
    release_matrix(fixture->base_q_before);
    release_matrix(fixture->step_primary_before);
    release_matrix(fixture->step_rho_before);
    release_matrix(fixture->step_q_before);
}

static void reset_valid_case(struct fixture *fixture) {
    int i, j;

    fixture->mapping.mode = Q_PARAMETERIZATION_PHYSICAL;
    fill_matrix(fixture->base_primary, INPUT_HALO_SENTINEL);
    fill_matrix(fixture->base_rho, INPUT_HALO_SENTINEL);
    fill_matrix(fixture->base_q, INPUT_HALO_SENTINEL);
    fill_matrix(fixture->step_primary, INPUT_HALO_SENTINEL);
    fill_matrix(fixture->step_rho, INPUT_HALO_SENTINEL);
    fill_matrix(fixture->step_q, INPUT_HALO_SENTINEL);
    for (j = 1; j <= NY; ++j) {
        for (i = 1; i <= NX; ++i) {
            fixture->base_primary[j][i] = 20.0f + (float)(3 * j + i);
            fixture->base_rho[j][i] = 200.0f + (float)(7 * j + 2 * i);
            fixture->base_q[j][i] = 50.0f + (float)(5 * j + i);
            fixture->step_primary[j][i] = 0.0f;
            fixture->step_rho[j][i] = 0.0f;
            fixture->step_q[j][i] = 0.0f;
        }
    }
    fixture->request.nx = NX;
    fixture->request.ny = NY;
    fixture->request.alpha = 1.0f;
    fixture->request.primary_bounds_enabled = 1;
    fixture->request.primary_lower = 1.0f;
    fixture->request.primary_upper = 1000.0f;
    fixture->request.rho_lower = 1.0f;
    fixture->request.rho_upper = 10000.0f;
    fixture->request.q_lower = 1.0f;
    fixture->request.q_upper = 1000.0f;
    fixture->request.q_mapping = &fixture->mapping;
    fixture->request.base_primary = fixture->base_primary;
    fixture->request.base_rho = fixture->base_rho;
    fixture->request.base_q = fixture->base_q;
    fixture->request.optimizer_step_primary = fixture->step_primary;
    fixture->request.optimizer_step_rho = fixture->step_rho;
    fixture->request.optimizer_step_q = fixture->step_q;
    fixture->request.trial_primary = fixture->trial_primary;
    fixture->request.trial_rho = fixture->trial_rho;
    fixture->request.trial_q = fixture->trial_q;
    fixture->request.trial_tau = fixture->trial_tau;
    fill_matrix(fixture->trial_primary, OUTPUT_SENTINEL);
    fill_matrix(fixture->trial_rho, OUTPUT_SENTINEL);
    fill_matrix(fixture->trial_q, OUTPUT_SENTINEL);
    fill_matrix(fixture->trial_tau, OUTPUT_SENTINEL);
}

static void snapshot_inputs(struct fixture *fixture) {
    copy_matrix(fixture->base_primary_before, fixture->base_primary);
    copy_matrix(fixture->base_rho_before, fixture->base_rho);
    copy_matrix(fixture->base_q_before, fixture->base_q);
    copy_matrix(fixture->step_primary_before, fixture->step_primary);
    copy_matrix(fixture->step_rho_before, fixture->step_rho);
    copy_matrix(fixture->step_q_before, fixture->step_q);
    fixture->mapping_before = fixture->mapping;
}

static int inputs_unchanged(const struct fixture *fixture) {
    return matrices_equal(fixture->base_primary, fixture->base_primary_before) &&
            matrices_equal(fixture->base_rho, fixture->base_rho_before) &&
            matrices_equal(fixture->base_q, fixture->base_q_before) &&
            matrices_equal(fixture->step_primary, fixture->step_primary_before) &&
            matrices_equal(fixture->step_rho, fixture->step_rho_before) &&
            matrices_equal(fixture->step_q, fixture->step_q_before) &&
            memcmp(&fixture->mapping, &fixture->mapping_before,
                    sizeof(fixture->mapping)) == 0;
}

static int output_halos_untouched(const struct fixture *fixture) {
    return halos_are_sentinel(fixture->trial_primary) &&
            halos_are_sentinel(fixture->trial_rho) &&
            halos_are_sentinel(fixture->trial_q) &&
            halos_are_sentinel(fixture->trial_tau);
}

static int outputs_unchanged(const struct fixture *fixture) {
    return matrix_is_sentinel(fixture->trial_primary) &&
            matrix_is_sentinel(fixture->trial_rho) &&
            matrix_is_sentinel(fixture->trial_q) &&
            matrix_is_sentinel(fixture->trial_tau);
}

static int successful_call(struct fixture *fixture) {
    int i, j;

    snapshot_inputs(fixture);
    if (visco_sh_exact_build_trial_parameter_state(&fixture->request) != 0) return 0;
    if (!inputs_unchanged(fixture) || !output_halos_untouched(fixture)) return 0;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i)
            if (fixture->trial_tau[j][i] !=
                    q_to_tau(fixture->trial_q[j][i], &fixture->mapping)) return 0;
    return 1;
}

static int transactional_failure(struct fixture *fixture,
        const struct visco_sh_exact_trial_state_request *request) {
    fill_matrix(fixture->trial_primary, OUTPUT_SENTINEL);
    fill_matrix(fixture->trial_rho, OUTPUT_SENTINEL);
    fill_matrix(fixture->trial_q, OUTPUT_SENTINEL);
    fill_matrix(fixture->trial_tau, OUTPUT_SENTINEL);
    return visco_sh_exact_build_trial_parameter_state(request) != 0 &&
            outputs_unchanged(fixture);
}

static int test_normal_and_zero_step(struct fixture *fixture) {
    int i, j;

    reset_valid_case(fixture);
    fixture->request.alpha = 0.25f;
    fixture->step_primary[1][1] = 1.25f;
    fixture->step_primary[1][2] = -2.50f;
    fixture->step_primary[2][1] = -0.75f;
    fixture->step_primary[2][2] = 3.00f;
    fixture->step_rho[1][1] = -4.0f;
    fixture->step_rho[1][2] = 5.5f;
    fixture->step_rho[2][1] = 2.25f;
    fixture->step_rho[2][2] = -1.5f;
    fixture->step_q[1][1] = 7.0f;
    fixture->step_q[1][2] = -3.5f;
    fixture->step_q[2][1] = -1.25f;
    fixture->step_q[2][2] = 4.75f;
    if (!successful_call(fixture)) return 0;
    for (j = 1; j <= NY; ++j) {
        for (i = 1; i <= NX; ++i) {
            if (fixture->trial_primary[j][i] != fixture->base_primary[j][i] -
                    fixture->request.alpha * fixture->step_primary[j][i]) return 0;
            if (fixture->trial_rho[j][i] != fixture->base_rho[j][i] -
                    fixture->request.alpha * fixture->step_rho[j][i]) return 0;
            if (fixture->trial_q[j][i] != fixture->base_q[j][i] -
                    fixture->request.alpha * fixture->step_q[j][i]) return 0;
        }
    }

    reset_valid_case(fixture);
    fixture->request.alpha = 0.0f;
    fixture->step_primary[1][1] = 99.0f;
    fixture->step_rho[1][2] = -77.0f;
    fixture->step_q[2][1] = 44.0f;
    if (!successful_call(fixture)) return 0;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i)
            if (fixture->trial_primary[j][i] != fixture->base_primary[j][i] ||
                    fixture->trial_rho[j][i] != fixture->base_rho[j][i] ||
                    fixture->trial_q[j][i] != fixture->base_q[j][i]) return 0;
    return 1;
}

static void prepare_rejection_case(struct fixture *fixture) {
    reset_valid_case(fixture);
    fixture->base_primary[1][1] = 20.0f;
    fixture->base_rho[1][1] = 200.0f;
    fixture->base_q[1][1] = 50.0f;
    fixture->request.alpha = 1.0f;
    fixture->request.primary_bounds_enabled = 1;
    fixture->request.primary_lower = 10.0f;
    fixture->request.primary_upper = 30.0f;
    fixture->request.rho_lower = 150.0f;
    fixture->request.rho_upper = 250.0f;
    fixture->request.q_lower = 20.0f;
    fixture->request.q_upper = 80.0f;
}

static int test_reject_to_base_and_no_clipping(struct fixture *fixture) {
    prepare_rejection_case(fixture);
    fixture->step_primary[1][1] = 25.0f;
    if (!successful_call(fixture) || fixture->trial_primary[1][1] != 20.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->step_primary[1][1] = 15.0f;
    if (!successful_call(fixture) || fixture->trial_primary[1][1] != 20.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->step_primary[1][1] = -15.0f;
    if (!successful_call(fixture) || fixture->trial_primary[1][1] != 20.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->request.alpha = 2.0f;
    fixture->step_primary[1][1] = -FLT_MAX;
    if (!successful_call(fixture) || fixture->trial_primary[1][1] != 20.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->request.primary_bounds_enabled = 0;
    fixture->request.primary_lower = 999.0f;
    fixture->request.primary_upper = 1000.0f;
    fixture->step_primary[1][1] = -40.0f;
    if (!successful_call(fixture) || fixture->trial_primary[1][1] != 60.0f) return 0;

    prepare_rejection_case(fixture);
    fixture->step_rho[1][1] = 250.0f;
    if (!successful_call(fixture) || fixture->trial_rho[1][1] != 200.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->step_rho[1][1] = 60.0f;
    if (!successful_call(fixture) || fixture->trial_rho[1][1] != 200.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->step_rho[1][1] = -60.0f;
    if (!successful_call(fixture) || fixture->trial_rho[1][1] != 200.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->request.alpha = 2.0f;
    fixture->step_rho[1][1] = -FLT_MAX;
    if (!successful_call(fixture) || fixture->trial_rho[1][1] != 200.0f) return 0;

    prepare_rejection_case(fixture);
    fixture->step_q[1][1] = 60.0f;
    if (!successful_call(fixture) || fixture->trial_q[1][1] != 50.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->step_q[1][1] = 45.0f;
    if (!successful_call(fixture) || fixture->trial_q[1][1] != 50.0f ||
            fixture->trial_q[1][1] == 20.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->step_q[1][1] = -40.0f;
    if (!successful_call(fixture) || fixture->trial_q[1][1] != 50.0f) return 0;
    prepare_rejection_case(fixture);
    fixture->request.alpha = 2.0f;
    fixture->step_q[1][1] = -FLT_MAX;
    if (!successful_call(fixture) || fixture->trial_q[1][1] != 50.0f) return 0;
    return 1;
}

static int test_transactional_failures(struct fixture *fixture) {
    float ***pointers[] = {
        &fixture->request.base_primary, &fixture->request.base_rho,
        &fixture->request.base_q, &fixture->request.optimizer_step_primary,
        &fixture->request.optimizer_step_rho, &fixture->request.optimizer_step_q,
        &fixture->request.trial_primary, &fixture->request.trial_rho,
        &fixture->request.trial_q, &fixture->request.trial_tau,
    };
    int index;

    reset_valid_case(fixture);
    if (!transactional_failure(fixture, NULL)) return 0;
    reset_valid_case(fixture);
    fixture->request.nx = 0;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.ny = 0;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.alpha = -0.1f;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.alpha = NAN;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.alpha = INFINITY;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.q_mapping = NULL;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->mapping.mode = Q_PARAMETERIZATION_LEGACY;
    if (!transactional_failure(fixture, &fixture->request)) return 0;

    for (index = 0; index < 10; ++index) {
        float **saved;
        reset_valid_case(fixture);
        saved = *pointers[index];
        *pointers[index] = NULL;
        if (!transactional_failure(fixture, &fixture->request)) return 0;
        *pointers[index] = saved;
    }

    reset_valid_case(fixture);
    fixture->request.rho_lower = NAN;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.rho_upper = INFINITY;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.rho_lower = 3.0f;
    fixture->request.rho_upper = 2.0f;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.q_lower = 0.0f;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.q_upper = 0.0f;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.q_lower = 9.0f;
    fixture->request.q_upper = 8.0f;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.primary_lower = NAN;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.primary_upper = INFINITY;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->request.primary_lower = 30.0f;
    fixture->request.primary_upper = 10.0f;
    if (!transactional_failure(fixture, &fixture->request)) return 0;

    reset_valid_case(fixture);
    fixture->base_primary[1][1] = NAN;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->base_rho[1][1] = NAN;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->base_q[1][1] = NAN;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->step_primary[1][1] = NAN;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->step_rho[1][1] = NAN;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->step_q[1][1] = NAN;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->base_primary[1][1] = 0.0f;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->base_rho[1][1] = 0.0f;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    reset_valid_case(fixture);
    fixture->base_q[1][1] = 0.0f;
    if (!transactional_failure(fixture, &fixture->request)) return 0;
    return 1;
}

int main(void) {
    struct fixture fixture;

    REQUIRE(fixture_allocate(&fixture), "matrix allocation and physical mapping");
    REQUIRE(test_normal_and_zero_step(&fixture), "normal and zero-step contracts");
    REQUIRE(test_reject_to_base_and_no_clipping(&fixture),
            "reject-to-base and no-clipping contracts");
    REQUIRE(test_transactional_failures(&fixture),
            "structural and numerical transactional failures");
    printf("{\"normal_subtractive\":true,\"zero_step\":true,"
            "\"physical_q\":true,\"reject_to_base\":true,"
            "\"no_clipping\":true,\"input_immutable\":true,"
            "\"halos_untouched\":true,\"transactional_failures\":true,"
            "\"legacy_mode_fail_closed\":true,"
            "\"tau_return_failure_unreachable\":true}\n");
    fixture_release(&fixture);
    return 0;
}
