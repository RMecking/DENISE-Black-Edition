/* SH-OPT-SCALE-01 data-driven harness for the scaled exact SH step.
 *
 * Contract: docs/sh_opt_scale_01_contract.md (revision 2), section 2
 * (oracles O1-O5, O8).  The harness only distributes global input fields,
 * calls the public visco_sh_exact_build_scaled_steepest_step(), optionally
 * calls the unchanged visco_sh_exact_build_steepest_subtractive_step() on
 * separate output arrays as the d4693e1 reference, and records raw results.
 * All expectations are evaluated by the Python oracle.
 *
 * Usage: mpiexec -n <NPROCX*NPROCY> harness <case.bin> <output_prefix>
 *
 * Case file (native int32/float32):
 *   int32  magic, nxg, nyg, nprocx, nprocy, scaling_mode,
 *          null_target, null_rank, callers, boundary_fault, reference,
 *          mode_override_rank, mode_override
 *   float  weight_primary, weight_rho, weight_q
 *   float  7 global fields [nyg][nxg] (row j outer, column i inner):
 *          g_primary, g_rho, g_q, m_primary, m_rho, m_q, mask
 * null_target:    0 none; 1..4 base_primary/base_rho/base_q/update_mask NULL;
 *                 5..8 last owned row of those arrays NULL; 9 all four NULL;
 *                 10 the scaling pointer itself NULL.
 * null_rank:      rank to which null_target applies (-1: every rank).
 * callers:        0 every rank calls; 1 only rank 0 calls.
 * boundary_fault: 0 none; 1 boundary NULL; 2 nx=0; 3 ny=0;
 *                 4 grad_raw_q NULL; 5 optimizer_step_rho NULL.
 * reference:      1 also call the d4693e1 function on separate outputs.
 * mode_override_rank/mode_override: rank (-1 none) that uses another mode.
 *
 * Output file <output_prefix>.<rank>:
 *   int32  magic, rank, status, pos_x, pos_y, nx, ny, inputs_unchanged,
 *          reference_status, outputs_equal_reference
 *   float  step_primary, step_rho, step_q, each (ny+2)*(nx+2) incl. halos.
 */

#include "fd.h"

#include <limits.h>
#include <stdint.h>

#define CASE_MAGIC 0x32534F53
#define RESULT_MAGIC 0x52534F53
#define HEADER_INTS 13
#define FIELD_COUNT 7
#define OUTPUT_SENTINEL (-98765.25f)
#define MASK_HALO_POISON (0.5f)
#define NOT_CALLED INT_MAX

void err(char err_text[]) {
    fprintf(stderr, "unexpected err(): %s\n", err_text);
    exit(91);
}

static float **allocate_local(int nx, int ny, float value) {
    float **matrix = (float **)malloc((size_t)(ny + 2) * sizeof(*matrix));
    int i, j;

    if (matrix == NULL) return NULL;
    for (j = 0; j < ny + 2; ++j) {
        matrix[j] = (float *)malloc((size_t)(nx + 2) * sizeof(float));
        if (matrix[j] == NULL) return NULL;
        for (i = 0; i < nx + 2; ++i) matrix[j][i] = value;
    }
    return matrix;
}

static void release_local(float **matrix, int ny) {
    int j;

    if (matrix == NULL) return;
    for (j = 0; j < ny + 2; ++j) free(matrix[j]);
    free(matrix);
}

static void copy_local(float **target, float **source, int nx, int ny) {
    int j;

    for (j = 0; j < ny + 2; ++j)
        memcpy(target[j], source[j], (size_t)(nx + 2) * sizeof(float));
}

static int local_equal(float **left, float **right, int nx, int ny) {
    int j;

    for (j = 0; j < ny + 2; ++j)
        if (memcmp(left[j], right[j], (size_t)(nx + 2) * sizeof(float)) != 0)
            return 0;
    return 1;
}

static int read_exact(FILE *stream, void *target, size_t size, size_t count) {
    return fread(target, size, count, stream) == count;
}

static void apply_boundary_fault(struct visco_sh_exact_optimizer_boundary *boundary,
                                 int fault) {
    if (fault == 2) boundary->nx = 0;
    if (fault == 3) boundary->ny = 0;
    if (fault == 4) boundary->grad_raw_q = NULL;
    if (fault == 5) boundary->optimizer_step_rho = NULL;
}

int main(int argc, char **argv) {
    int32_t header[HEADER_INTS];
    float weights[3];
    float *global = NULL;
    float **fields[FIELD_COUNT], **before[FIELD_COUNT];
    float **steps[3], **reference_steps[3];
    float *saved_rows[4] = {NULL, NULL, NULL, NULL};
    float ***row_targets[4];
    struct visco_sh_exact_optimizer_boundary boundary, reference_boundary;
    struct visco_sh_exact_step_scaling scaling;
    const struct visco_sh_exact_step_scaling *scaling_argument = &scaling;
    int rank, size, nxg, nyg, nprocx, nprocy, nx, ny, pos_x, pos_y;
    int null_target, null_rank, callers, fault, reference, status, unchanged;
    int reference_status = NOT_CALLED, equal_reference = 0, k, i, j;
    size_t cells;
    char output_path[4096];
    FILE *stream;
    int32_t result_header[10];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 3) {
        fprintf(stderr, "usage: harness <case.bin> <output_prefix>\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    stream = fopen(argv[1], "rb");
    if (stream == NULL || !read_exact(stream, header, sizeof(int32_t), HEADER_INTS) ||
            header[0] != CASE_MAGIC || !read_exact(stream, weights, sizeof(float), 3)) {
        fprintf(stderr, "invalid case file\n");
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    nxg = header[1];
    nyg = header[2];
    nprocx = header[3];
    nprocy = header[4];
    null_target = header[6];
    null_rank = header[7];
    callers = header[8];
    fault = header[9];
    reference = header[10];
    if (nxg < 1 || nyg < 1 || nprocx < 1 || nprocy < 1 ||
            nprocx * nprocy != size || nxg % nprocx != 0 || nyg % nprocy != 0) {
        fprintf(stderr, "case decomposition does not match the MPI job\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    cells = (size_t)nxg * (size_t)nyg;
    global = (float *)malloc(cells * FIELD_COUNT * sizeof(float));
    if (global == NULL || !read_exact(stream, global, sizeof(float), cells * FIELD_COUNT)) {
        fprintf(stderr, "truncated case file\n");
        MPI_Abort(MPI_COMM_WORLD, 5);
    }
    fclose(stream);

    pos_x = rank % nprocx;
    pos_y = rank / nprocx;
    nx = nxg / nprocx;
    ny = nyg / nprocy;

    /* Halos are poisoned: only owned cells j=1..ny, i=1..nx may be read. */
    for (k = 0; k < FIELD_COUNT; ++k) {
        fields[k] = allocate_local(nx, ny, (k == 6) ? MASK_HALO_POISON : NAN);
        before[k] = allocate_local(nx, ny, 0.0f);
        if (fields[k] == NULL || before[k] == NULL) MPI_Abort(MPI_COMM_WORLD, 6);
        for (j = 1; j <= ny; ++j) {
            for (i = 1; i <= nx; ++i) {
                size_t gj = (size_t)(pos_y * ny + j - 1);
                size_t gi = (size_t)(pos_x * nx + i - 1);
                fields[k][j][i] = global[(size_t)k * cells + gj * (size_t)nxg + gi];
            }
        }
        copy_local(before[k], fields[k], nx, ny);
    }
    for (k = 0; k < 3; ++k) {
        steps[k] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        reference_steps[k] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        if (steps[k] == NULL || reference_steps[k] == NULL) MPI_Abort(MPI_COMM_WORLD, 6);
    }

    memset(&boundary, 0, sizeof(boundary));
    boundary.nx = nx;
    boundary.ny = ny;
    boundary.grad_raw_primary = fields[0];
    boundary.grad_raw_rho = fields[1];
    boundary.grad_raw_q = fields[2];
    boundary.optimizer_step_primary = steps[0];
    boundary.optimizer_step_rho = steps[1];
    boundary.optimizer_step_q = steps[2];
    apply_boundary_fault(&boundary, fault);

    reference_boundary = boundary;
    reference_boundary.optimizer_step_primary = reference_steps[0];
    reference_boundary.optimizer_step_rho =
            (fault == 5) ? NULL : reference_steps[1];
    reference_boundary.optimizer_step_q = reference_steps[2];

    memset(&scaling, 0, sizeof(scaling));
    scaling.scaling_mode = (header[11] == rank) ? header[12] : header[5];
    scaling.base_primary = fields[3];
    scaling.base_rho = fields[4];
    scaling.base_q = fields[5];
    scaling.update_mask = fields[6];
    scaling.weight_primary = weights[0];
    scaling.weight_rho = weights[1];
    scaling.weight_q = weights[2];

    row_targets[0] = &fields[3];
    row_targets[1] = &fields[4];
    row_targets[2] = &fields[5];
    row_targets[3] = &fields[6];
    if (null_target != 0 && (null_rank < 0 || null_rank == rank)) {
        if (null_target == 1) scaling.base_primary = NULL;
        if (null_target == 2) scaling.base_rho = NULL;
        if (null_target == 3) scaling.base_q = NULL;
        if (null_target == 4) scaling.update_mask = NULL;
        if (null_target >= 5 && null_target <= 8) {
            float **matrix = *row_targets[null_target - 5];
            saved_rows[null_target - 5] = matrix[ny];
            matrix[ny] = NULL;
        }
        if (null_target == 9) {
            scaling.base_primary = NULL;
            scaling.base_rho = NULL;
            scaling.base_q = NULL;
            scaling.update_mask = NULL;
        }
        if (null_target == 10) scaling_argument = NULL;
    }

    if (callers == 0 || rank == 0)
        status = visco_sh_exact_build_scaled_steepest_step(
                (fault == 1) ? NULL : &boundary, scaling_argument);
    else
        status = NOT_CALLED;

    for (k = 0; k < 4; ++k)
        if (saved_rows[k] != NULL) (*row_targets[k])[ny] = saved_rows[k];

    if (reference && (callers == 0 || rank == 0)) {
        reference_status = visco_sh_exact_build_steepest_subtractive_step(
                (fault == 1) ? NULL : &reference_boundary);
        equal_reference = 1;
        for (k = 0; k < 3; ++k)
            if (!local_equal(steps[k], reference_steps[k], nx, ny)) equal_reference = 0;
    }

    unchanged = 1;
    for (k = 0; k < FIELD_COUNT; ++k)
        if (!local_equal(fields[k], before[k], nx, ny)) unchanged = 0;

    snprintf(output_path, sizeof(output_path), "%s.%d", argv[2], rank);
    stream = fopen(output_path, "wb");
    if (stream == NULL) MPI_Abort(MPI_COMM_WORLD, 7);
    result_header[0] = RESULT_MAGIC;
    result_header[1] = rank;
    result_header[2] = status;
    result_header[3] = pos_x;
    result_header[4] = pos_y;
    result_header[5] = nx;
    result_header[6] = ny;
    result_header[7] = unchanged;
    result_header[8] = reference_status;
    result_header[9] = equal_reference;
    fwrite(result_header, sizeof(int32_t), 10, stream);
    for (k = 0; k < 3; ++k)
        for (j = 0; j < ny + 2; ++j)
            fwrite(steps[k][j], sizeof(float), (size_t)(nx + 2), stream);
    fclose(stream);

    for (k = 0; k < FIELD_COUNT; ++k) {
        release_local(fields[k], ny);
        release_local(before[k], ny);
    }
    for (k = 0; k < 3; ++k) {
        release_local(steps[k], ny);
        release_local(reference_steps[k], ny);
    }
    free(global);
    MPI_Finalize();
    return 0;
}
