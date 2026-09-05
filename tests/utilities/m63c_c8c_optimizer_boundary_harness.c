/* Direct contract harness for the inactive C8c exact optimizer boundary. */

#include "fd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NX 3
#define NY 2
#define WIDTH (NX + 2)
#define HEIGHT (NY + 2)
#define OUTPUT_SENTINEL (-98765.25f)
#define RAW_HALO_SENTINEL (31415.75f)

#define REQUIRE(condition, description) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "optimizer-boundary contract failed: %s\n", description); \
            return 1; \
        } \
    } while (0)

static float **allocate_matrix(float value) {
    int i, j;
    float **matrix = (float **)malloc((size_t)HEIGHT * sizeof(*matrix));

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

static void release_harness_matrix(float **matrix) {
    int j;

    if (matrix == NULL) return;
    for (j = 0; j < HEIGHT; ++j) free(matrix[j]);
    free(matrix);
}

static void fill_output_sentinel(float **matrix) {
    int i, j;

    for (j = 0; j < HEIGHT; ++j)
        for (i = 0; i < WIDTH; ++i) matrix[j][i] = OUTPUT_SENTINEL;
}

static int matrix_equals(float **left, float **right) {
    int j;

    for (j = 0; j < HEIGHT; ++j)
        if (memcmp(left[j], right[j], (size_t)WIDTH * sizeof(float)) != 0)
            return 0;
    return 1;
}

static int output_is_all_sentinel(float **matrix) {
    int i, j;

    for (j = 0; j < HEIGHT; ++j)
        for (i = 0; i < WIDTH; ++i)
            if (matrix[j][i] != OUTPUT_SENTINEL) return 0;
    return 1;
}

static int output_halos_are_sentinel(float **matrix) {
    int i, j;

    for (j = 0; j < HEIGHT; ++j)
        for (i = 0; i < WIDTH; ++i)
            if ((j == 0) || (j == NY + 1) || (i == 0) || (i == NX + 1))
                if (matrix[j][i] != OUTPUT_SENTINEL) return 0;
    return 1;
}

static void initialize_raw_fields(
        float **primary, float **rho, float **q) {
    int i, j;

    for (j = 0; j < HEIGHT; ++j) {
        for (i = 0; i < WIDTH; ++i) {
            primary[j][i] = RAW_HALO_SENTINEL;
            rho[j][i] = RAW_HALO_SENTINEL;
            q[j][i] = RAW_HALO_SENTINEL;
        }
    }
    primary[1][1] = 1.25f;
    primary[1][2] = -2.50f;
    primary[1][3] = 3.75f;
    primary[2][1] = -4.00f;
    primary[2][2] = 5.50f;
    primary[2][3] = -6.25f;

    rho[1][1] = -11.0f;
    rho[1][2] = 12.5f;
    rho[1][3] = -13.75f;
    rho[2][1] = 14.25f;
    rho[2][2] = -15.5f;
    rho[2][3] = 16.0f;

    q[1][1] = 101.0f;
    q[1][2] = -0.375f;
    q[1][3] = 88.125f;
    q[2][1] = -77.5f;
    q[2][2] = 0.0625f;
    q[2][3] = -64.25f;
}

static int copied_owned_cells_are_exact(
        const struct visco_sh_exact_optimizer_boundary *boundary) {
    int i, j;

    for (j = 1; j <= NY; ++j) {
        for (i = 1; i <= NX; ++i) {
            if (boundary->optimizer_step_primary[j][i] !=
                    boundary->grad_raw_primary[j][i]) return 0;
            if (boundary->optimizer_step_rho[j][i] !=
                    boundary->grad_raw_rho[j][i]) return 0;
            if (boundary->optimizer_step_q[j][i] !=
                    boundary->grad_raw_q[j][i]) return 0;
        }
    }
    return 1;
}

static double raw_dot_optimizer_step(
        const struct visco_sh_exact_optimizer_boundary *boundary) {
    double dot = 0.0;
    int i, j;

    for (j = 1; j <= NY; ++j) {
        for (i = 1; i <= NX; ++i) {
            dot += (double)boundary->grad_raw_primary[j][i] *
                    boundary->optimizer_step_primary[j][i];
            dot += (double)boundary->grad_raw_rho[j][i] *
                    boundary->optimizer_step_rho[j][i];
            dot += (double)boundary->grad_raw_q[j][i] *
                    boundary->optimizer_step_q[j][i];
        }
    }
    return dot;
}

static int invalid_cases_fail_closed(
        struct visco_sh_exact_optimizer_boundary *boundary,
        float **out_primary, float **out_rho, float **out_q) {
    float ***matrix_fields[] = {
        &boundary->grad_raw_primary,
        &boundary->grad_raw_rho,
        &boundary->grad_raw_q,
        &boundary->optimizer_step_primary,
        &boundary->optimizer_step_rho,
        &boundary->optimizer_step_q,
    };
    int original_nx = boundary->nx;
    int original_ny = boundary->ny;
    int index;

    fill_output_sentinel(out_primary);
    fill_output_sentinel(out_rho);
    fill_output_sentinel(out_q);
    if (visco_sh_exact_build_steepest_subtractive_step(NULL) == 0) return 0;
    if (!output_is_all_sentinel(out_primary) || !output_is_all_sentinel(out_rho) ||
            !output_is_all_sentinel(out_q)) return 0;

    boundary->nx = 0;
    fill_output_sentinel(out_primary);
    fill_output_sentinel(out_rho);
    fill_output_sentinel(out_q);
    if (visco_sh_exact_build_steepest_subtractive_step(boundary) == 0) return 0;
    if (!output_is_all_sentinel(out_primary) || !output_is_all_sentinel(out_rho) ||
            !output_is_all_sentinel(out_q)) return 0;
    boundary->nx = original_nx;

    boundary->ny = 0;
    fill_output_sentinel(out_primary);
    fill_output_sentinel(out_rho);
    fill_output_sentinel(out_q);
    if (visco_sh_exact_build_steepest_subtractive_step(boundary) == 0) return 0;
    if (!output_is_all_sentinel(out_primary) || !output_is_all_sentinel(out_rho) ||
            !output_is_all_sentinel(out_q)) return 0;
    boundary->ny = original_ny;

    for (index = 0; index < 6; ++index) {
        float **saved = *matrix_fields[index];
        *matrix_fields[index] = NULL;
        fill_output_sentinel(out_primary);
        fill_output_sentinel(out_rho);
        fill_output_sentinel(out_q);
        if (visco_sh_exact_build_steepest_subtractive_step(boundary) == 0) return 0;
        if (!output_is_all_sentinel(out_primary) ||
                !output_is_all_sentinel(out_rho) ||
                !output_is_all_sentinel(out_q)) return 0;
        *matrix_fields[index] = saved;
    }
    return 1;
}

int main(void) {
    float **raw_primary = allocate_matrix(RAW_HALO_SENTINEL);
    float **raw_rho = allocate_matrix(RAW_HALO_SENTINEL);
    float **raw_q = allocate_matrix(RAW_HALO_SENTINEL);
    float **raw_primary_before = allocate_matrix(RAW_HALO_SENTINEL);
    float **raw_rho_before = allocate_matrix(RAW_HALO_SENTINEL);
    float **raw_q_before = allocate_matrix(RAW_HALO_SENTINEL);
    float **out_primary = allocate_matrix(OUTPUT_SENTINEL);
    float **out_rho = allocate_matrix(OUTPUT_SENTINEL);
    float **out_q = allocate_matrix(OUTPUT_SENTINEL);
    struct visco_sh_exact_optimizer_boundary boundary;
    double dot;
    int i, j;

    REQUIRE(raw_primary && raw_rho && raw_q && raw_primary_before && raw_rho_before &&
            raw_q_before && out_primary && out_rho && out_q, "matrix allocation");
    initialize_raw_fields(raw_primary, raw_rho, raw_q);
    for (j = 0; j < HEIGHT; ++j) {
        for (i = 0; i < WIDTH; ++i) {
            raw_primary_before[j][i] = raw_primary[j][i];
            raw_rho_before[j][i] = raw_rho[j][i];
            raw_q_before[j][i] = raw_q[j][i];
        }
    }
    boundary.nx = NX;
    boundary.ny = NY;
    boundary.grad_raw_primary = raw_primary;
    boundary.grad_raw_rho = raw_rho;
    boundary.grad_raw_q = raw_q;
    boundary.optimizer_step_primary = out_primary;
    boundary.optimizer_step_rho = out_rho;
    boundary.optimizer_step_q = out_q;

    REQUIRE(visco_sh_exact_build_steepest_subtractive_step(&boundary) == 0,
            "successful boundary call");
    REQUIRE(copied_owned_cells_are_exact(&boundary), "component-wise raw copy");
    REQUIRE(matrix_equals(raw_primary, raw_primary_before) &&
            matrix_equals(raw_rho, raw_rho_before) &&
            matrix_equals(raw_q, raw_q_before), "raw-gradient immutability");
    REQUIRE(output_halos_are_sentinel(out_primary) &&
            output_halos_are_sentinel(out_rho) &&
            output_halos_are_sentinel(out_q), "owned-cell-only writes");
    REQUIRE(out_q[1][1] == 101.0f && out_q[1][2] == -0.375f &&
            out_q[2][3] == -64.25f, "physical-Q identity");
    dot = raw_dot_optimizer_step(&boundary);
    REQUIRE(dot > 0.0, "positive raw-gradient/optimizer-step dot product");
    REQUIRE(invalid_cases_fail_closed(&boundary, out_primary, out_rho, out_q),
            "invalid inputs fail closed without output writes");

    printf("{\"successful_copy\":true,\"raw_immutable\":true,"
            "\"halos_untouched\":true,\"physical_q_identity\":true,"
            "\"dot_product\":%.17g,\"invalid_cases\":9}\n", dot);

    release_harness_matrix(raw_primary);
    release_harness_matrix(raw_rho);
    release_harness_matrix(raw_q);
    release_harness_matrix(raw_primary_before);
    release_harness_matrix(raw_rho_before);
    release_harness_matrix(raw_q_before);
    release_harness_matrix(out_primary);
    release_harness_matrix(out_rho);
    release_harness_matrix(out_q);
    return 0;
}
