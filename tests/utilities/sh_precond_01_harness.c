/* SH-PRECOND-01 data-driven harness for the public API of contract
 * docs/sh_precond_01_contract.md (revision 2), section 2.
 *
 * The harness only distributes global input fields, calls one public
 * function and records raw results; every expectation is evaluated by the
 * Python oracle (tests/test_sh_precond_01.py).
 *
 * Usage: mpiexec -n <NPROCX*NPROCY> harness <case.bin> <output_prefix>
 *
 * Case file (native int32 / float64 / float32):
 *   int32   magic, op, nxg, nyg, nprocx, nprocy, mode, scaling_mode,
 *           fault, fault_rank, nsteps, override_rank, override_value
 *   float64 epsilon, x_min, x_max
 *   float32 weight_primary, weight_rho, weight_q
 *   float32 fields [nyg][nxg] (row j outer, column i inner), depending on op:
 *     op 1 (accumulate, 1 rank): nsteps snapshot fields
 *     op 2 (receiver term, 1 rank): none (pos_x = mode, pos_y = scaling_mode)
 *     op 3 (preconditioner): source_energy, receiver_term, mask
 *     op 4 (preconditioned step): g_primary, g_rho, g_q, m_primary, m_rho,
 *           m_q, mask, precond
 *     op 5 (initial alpha, 1 rank): nsteps tuples float32 eps, int32 adaptive,
 *           float32 factor, int32 iteration, float32 previous
 *
 * fault (applies on fault_rank, or on every rank if fault_rank == -1):
 *   op 1: 1 energy NULL, 2 field NULL, 3 nx = 0
 *   op 2: 1 term NULL, 2 last row NULL
 *   op 3: 1 request NULL, 2 nx = 0, 3 source_energy NULL, 4 mask NULL,
 *         5 precond NULL, 6 receiver_term NULL, 7 last source_energy row NULL
 *   op 4: 1 boundary NULL, 2 nx = 0, 3 grad_raw_q NULL, 4 scaling NULL,
 *         5 precond NULL, 6 work_rho NULL, 7 last precond row NULL
 * override_rank/override_value: op 3 rank (-1 none) using another
 *   precond_mode; op 4 rank using another scaling_mode.
 *
 * Output file <output_prefix>.<rank>:
 *   int32  magic, rank, status, pos_x, pos_y, nx, ny, inputs_unchanged,
 *          nout
 *   float  nout arrays of (ny+2)*(nx+2) values incl. halos
 *          (op 1: energy; op 2: term (local block at pos; pos is the
 *           requested rank position); op 3: precond; op 4: 3 steps then
 *           3 work arrays;
 *           op 5: nsteps alphas, nx = nsteps, ny = 1, no halos).
 */

#include "fd.h"

#include <stdint.h>

#define CASE_MAGIC 0x31435250
#define RESULT_MAGIC 0x52435250
#define HEADER_INTS 13
#define OUTPUT_SENTINEL (-98765.25f)
#define MAX_FIELDS 8

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

static int local_equal(float **left, float **right, int nx, int ny) {
    int j;

    for (j = 0; j < ny + 2; ++j)
        if (memcmp(left[j], right[j], (size_t)(nx + 2) * sizeof(float)) != 0)
            return 0;
    return 1;
}

static void copy_local(float **target, float **source, int nx, int ny) {
    int j;

    for (j = 0; j < ny + 2; ++j)
        memcpy(target[j], source[j], (size_t)(nx + 2) * sizeof(float));
}

static void scatter(float **local, const float *global, int nxg, int nx,
                    int ny, int pos_x, int pos_y) {
    int i, j;

    for (j = 1; j <= ny; ++j)
        for (i = 1; i <= nx; ++i)
            local[j][i] = global[(size_t)(pos_y * ny + j - 1) * nxg +
                                 (size_t)(pos_x * nx + i - 1)];
}

static void fail(const char *message) {
    fprintf(stderr, "harness: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static void write_result(const char *prefix, int rank, int status, int pos_x,
                         int pos_y, int nx, int ny, int unchanged, int nout,
                         float ***arrays) {
    char path[4096];
    int32_t header[9];
    FILE *stream;
    int k, j;

    snprintf(path, sizeof(path), "%s.%d", prefix, rank);
    stream = fopen(path, "wb");
    if (stream == NULL) fail("cannot open output");
    header[0] = RESULT_MAGIC; header[1] = rank; header[2] = status;
    header[3] = pos_x; header[4] = pos_y; header[5] = nx; header[6] = ny;
    header[7] = unchanged; header[8] = nout;
    fwrite(header, sizeof(int32_t), 9, stream);
    for (k = 0; k < nout; ++k)
        for (j = 0; j < ny + 2; ++j)
            fwrite(arrays[k][j], sizeof(float), (size_t)(nx + 2), stream);
    fclose(stream);
}

int main(int argc, char **argv) {
    int32_t header[HEADER_INTS];
    double scalars[3];
    float weights[3];
    float *global = NULL;
    float **fields[MAX_FIELDS], **before[MAX_FIELDS], **outputs[3];
    int rank, size, op, nxg, nyg, nprocx, nprocy, nx, ny, pos_x, pos_y;
    int mode, scaling_mode, fault, fault_rank, nsteps, override_rank;
    int override_value, faulty, nfields = 0, status = 0, unchanged = 1, k, j;
    size_t cells;
    FILE *stream;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 3) fail("usage");
    stream = fopen(argv[1], "rb");
    if (stream == NULL) fail("cannot open case");
    if (fread(header, sizeof(int32_t), HEADER_INTS, stream) != HEADER_INTS ||
            header[0] != CASE_MAGIC) fail("bad header");
    if (fread(scalars, sizeof(double), 3, stream) != 3) fail("bad scalars");
    if (fread(weights, sizeof(float), 3, stream) != 3) fail("bad weights");
    op = header[1]; nxg = header[2]; nyg = header[3];
    nprocx = header[4]; nprocy = header[5]; mode = header[6];
    scaling_mode = header[7]; fault = header[8]; fault_rank = header[9];
    nsteps = header[10]; override_rank = header[11];
    override_value = header[12];
    faulty = (fault != 0) && ((fault_rank == -1) || (fault_rank == rank));

    /* ---------------- op 5: initial alpha (pure, one rank) ---------------- */
    if (op == 5) {
        float **alphas = allocate_local(nsteps, 1, OUTPUT_SENTINEL);
        float **row_view[1];
        for (k = 0; k < nsteps; ++k) {
            float eps_scale, factor, previous;
            int32_t adaptive, iteration;
            if (fread(&eps_scale, 4, 1, stream) != 1 ||
                    fread(&adaptive, 4, 1, stream) != 1 ||
                    fread(&factor, 4, 1, stream) != 1 ||
                    fread(&iteration, 4, 1, stream) != 1 ||
                    fread(&previous, 4, 1, stream) != 1) fail("bad tuple");
            alphas[1][k + 1] = visco_sh_exact_initial_alpha(
                    eps_scale, adaptive, factor, iteration, previous);
        }
        fclose(stream);
        row_view[0] = alphas;
        write_result(argv[2], rank, 0, 0, 0, nsteps, 1, 1, 1, row_view);
        MPI_Finalize();
        return 0;
    }

    if (op == 1 || op == 2 || op == 5) {
        if (size != 1) fail("op needs one rank");
        nprocx = nprocy = 1;
    }
    if (nprocx * nprocy != size) fail("rank count");
    if (nxg % nprocx || nyg % nprocy) fail("decomposition");
    nx = nxg / nprocx; ny = nyg / nprocy;
    pos_x = rank % nprocx; pos_y = rank / nprocx;
    cells = (size_t)nxg * (size_t)nyg;

    if (op == 1) nfields = nsteps;
    if (op == 3) nfields = 3;
    if (op == 4) nfields = 8;
    if (nfields > MAX_FIELDS && op != 1) fail("fields");
    global = (float *)malloc(cells * sizeof(float));
    if (global == NULL) fail("memory");

    /* ---------------- op 1: energy accumulation ---------------- */
    if (op == 1) {
        float **energy = allocate_local(nx, ny, 0.0f);
        float **snapshot = allocate_local(nx, ny, OUTPUT_SENTINEL);
        float **view[1];
        int i;
        /* Halos of the energy carry a sentinel; own cells start at 0. */
        for (j = 0; j < ny + 2; ++j)
            for (i = 0; i < nx + 2; ++i)
                energy[j][i] = (j == 0 || j == ny + 1 || i == 0 || i == nx + 1)
                               ? OUTPUT_SENTINEL : 0.0f;
        for (k = 0; k < nsteps; ++k) {
            if (fread(global, sizeof(float), cells, stream) != cells)
                fail("bad snapshot");
            scatter(snapshot, global, nxg, nx, ny, 0, 0);
            visco_sh_exact_accumulate_energy(
                    (faulty && fault == 1) ? NULL : energy,
                    (faulty && fault == 2) ? NULL : snapshot,
                    (faulty && fault == 3) ? 0 : nx, ny);
        }
        fclose(stream);
        view[0] = energy;
        write_result(argv[2], rank, 0, 0, 0, nx, ny, 1, 1, view);
        MPI_Finalize();
        return 0;
    }

    /* ---------------- op 2: receiver-line term ---------------- */
    if (op == 2) {
        float **term = allocate_local(nx, ny, OUTPUT_SENTINEL);
        float *saved = NULL;
        float **view[1];
        fclose(stream);
        if (faulty && fault == 2) { saved = term[ny]; term[ny] = NULL; }
        status = visco_sh_exact_build_receiver_line_term(
                (faulty && fault == 1) ? NULL : term, nx, ny, mode,
                scaling_mode, scalars[1], scalars[2]);
        if (saved != NULL) term[ny] = saved;
        view[0] = term;
        write_result(argv[2], rank, status, mode, scaling_mode, nx, ny, 1, 1,
                     view);
        MPI_Finalize();
        return 0;
    }

    for (k = 0; k < nfields; ++k) {
        if (fread(global, sizeof(float), cells, stream) != cells)
            fail("bad field");
        fields[k] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        before[k] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        if (fields[k] == NULL || before[k] == NULL) fail("memory");
        scatter(fields[k], global, nxg, nx, ny, pos_x, pos_y);
        copy_local(before[k], fields[k], nx, ny);
    }
    fclose(stream);

    /* ---------------- op 3: preconditioner ---------------- */
    if (op == 3) {
        struct visco_sh_exact_precond_request request;
        float *saved = NULL;
        memset(&request, 0, sizeof(request));
        outputs[0] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        request.precond_mode = (rank == override_rank) ? override_value : mode;
        request.nx = (faulty && fault == 2) ? 0 : nx;
        request.ny = ny;
        request.epsilon = (float)scalars[0];
        request.source_energy = (faulty && fault == 3) ? NULL : fields[0];
        request.receiver_term = (faulty && fault == 6) ? NULL : fields[1];
        request.update_mask = (faulty && fault == 4) ? NULL : fields[2];
        request.precond = (faulty && fault == 5) ? NULL : outputs[0];
        if (faulty && fault == 7) {
            saved = fields[0][ny]; fields[0][ny] = NULL;
        }
        status = visco_sh_exact_build_illumination_precond(
                (faulty && fault == 1) ? NULL : &request);
        if (saved != NULL) fields[0][ny] = saved;
        for (k = 0; k < nfields; ++k)
            unchanged &= local_equal(fields[k], before[k], nx, ny);
        write_result(argv[2], rank, status, pos_x, pos_y, nx, ny, unchanged,
                     1, outputs);
        MPI_Finalize();
        return 0;
    }

    /* ---------------- op 4: preconditioned step ---------------- */
    if (op == 4) {
        struct visco_sh_exact_optimizer_boundary boundary;
        struct visco_sh_exact_step_scaling scaling;
        float **work[3];
        float *saved = NULL;
        memset(&boundary, 0, sizeof(boundary));
        memset(&scaling, 0, sizeof(scaling));
        for (k = 0; k < 3; ++k) {
            outputs[k] = allocate_local(nx, ny, OUTPUT_SENTINEL);
            work[k] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        }
        boundary.nx = (faulty && fault == 2) ? 0 : nx;
        boundary.ny = ny;
        boundary.grad_raw_primary = fields[0];
        boundary.grad_raw_rho = fields[1];
        boundary.grad_raw_q = (faulty && fault == 3) ? NULL : fields[2];
        boundary.optimizer_step_primary = outputs[0];
        boundary.optimizer_step_rho = outputs[1];
        boundary.optimizer_step_q = outputs[2];
        scaling.scaling_mode = (rank == override_rank) ? override_value
                                                       : scaling_mode;
        scaling.base_primary = fields[3];
        scaling.base_rho = fields[4];
        scaling.base_q = fields[5];
        scaling.update_mask = fields[6];
        scaling.weight_primary = weights[0];
        scaling.weight_rho = weights[1];
        scaling.weight_q = weights[2];
        if (faulty && fault == 7) {
            saved = fields[7][ny]; fields[7][ny] = NULL;
        }
        status = visco_sh_exact_build_preconditioned_step(
                (faulty && fault == 1) ? NULL : &boundary,
                (faulty && fault == 4) ? NULL : &scaling,
                (faulty && fault == 5) ? NULL : fields[7],
                work[0], (faulty && fault == 6) ? NULL : work[1], work[2]);
        if (saved != NULL) fields[7][ny] = saved;
        for (k = 0; k < nfields; ++k)
            unchanged &= local_equal(fields[k], before[k], nx, ny);
        {
            /* Steps first, then the three work arrays (contract 2.4 step 1). */
            float **all[6];
            for (k = 0; k < 3; ++k) { all[k] = outputs[k]; all[3 + k] = work[k]; }
            write_result(argv[2], rank, status, pos_x, pos_y, nx, ny,
                         unchanged, 6, all);
        }
        MPI_Finalize();
        return 0;
    }

    fail("unknown op");
    return 1;
}
