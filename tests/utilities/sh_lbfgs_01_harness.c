/* SH-LBFGS-01 data-driven harness for the public API of contract
 * docs/sh_lbfgs_01_contract.md (revision 2), section 3.
 *
 * The harness only distributes global input fields, calls the public
 * functions in the order given by the case file and records raw results;
 * every expectation is evaluated by the Python oracle
 * (tests/test_sh_lbfgs_01.py).
 *
 * Usage: mpiexec -n <NPROCX*NPROCY> harness <case.bin> <output_prefix>
 *
 * Case file (native int32 / float32):
 *   int32   magic, op, nxg, nyg, nprocx, nprocy, memory, niter, has_precond,
 *           fault, fault_rank, fault_iter, skip_begin
 *   op 2 (init only): nothing else.
 *   op 1 (sequence):
 *     float32 fields [nyg][nxg] (row j outer, column i inner):
 *             stage base_primary, base_rho, base_q, mask
 *     per iteration:
 *       int32   kind (0 step; 1 begin_stage with this iteration's base, then
 *               step; 2 begin_stage only)
 *       float32 weight_primary, weight_rho, weight_q
 *       fields  base x3, grad x3, step_in x3, mask, precond (if has_precond)
 *
 * fault (on fault_rank, or every rank if fault_rank == -1):
 *   op 2: 1 lbfgs NULL, 2 nx = 0
 *   op 1, applied to the call of iteration fault_iter (fault_iter == -1:
 *   the initial begin_stage):
 *     step:        1 lbfgs NULL, 2 request NULL, 3 report NULL,
 *                  4 request nx + 1, 5 base_rho NULL, 6 grad_q NULL,
 *                  7 mask NULL, 8 step_primary NULL,
 *                  9 last row of grad_primary NULL,
 *                  10 last row of precond NULL (has_precond),
 *                  11 request ny + 1
 *     begin_stage: 101 lbfgs NULL, 102 base_q NULL, 103 last row of mask NULL
 *   skip_begin = 1: no initial begin_stage.
 *
 * Output file <output_prefix>.<rank>:
 *   int32  magic, rank, pos_x, pos_y, nx, ny, niter, init_status,
 *          begin_status
 *   per iteration:
 *     int32  status, begin_status, action, pair, pairs_used,
 *            step_unchanged, inputs_unchanged, report_untouched
 *     double curvature, gamma, descent, kappa, sigma[3]
 *     float  3 step arrays of (ny+2)*(nx+2) values incl. halos
 */

#include "fd.h"

#include <stdint.h>

#define CASE_MAGIC 0x31464231
#define RESULT_MAGIC 0x52464231
#define HEADER_INTS 13
#define OUTPUT_SENTINEL (-98765.25f)
#define REPORT_SENTINEL (-7777)

void err(char err_text[]) {
    fprintf(stderr, "unexpected err(): %s\n", err_text);
    exit(91);
}

static void fail(const char *message) {
    fprintf(stderr, "harness: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static float **allocate_local(int nx, int ny, float value) {
    float **matrix = (float **)malloc((size_t)(ny + 2) * sizeof(*matrix));
    int i, j;

    if (matrix == NULL) fail("memory");
    for (j = 0; j < ny + 2; ++j) {
        matrix[j] = (float *)malloc((size_t)(nx + 2) * sizeof(float));
        if (matrix[j] == NULL) fail("memory");
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

static void read_field(FILE *stream, float *global, size_t cells, float **local,
                       int nxg, int nx, int ny, int pos_x, int pos_y) {
    int i, j;

    if (fread(global, sizeof(float), cells, stream) != cells) fail("bad field");
    for (j = 1; j <= ny; ++j)
        for (i = 1; i <= nx; ++i)
            local[j][i] = global[(size_t)(pos_y * ny + j - 1) * nxg +
                                 (size_t)(pos_x * nx + i - 1)];
}

static int fault_here(int fault, int fault_rank, int rank, int code) {
    return (fault == code) && ((fault_rank == -1) || (fault_rank == rank));
}

int main(int argc, char **argv) {
    int32_t header[HEADER_INTS], out_header[9];
    int rank, size, op, nxg, nyg, nprocx, nprocy, memory, niter, has_precond;
    int fault, fault_rank, fault_iter, skip_begin, nx, ny, pos_x, pos_y;
    int init_status, begin_status = 0, k, c, j;
    size_t cells;
    float *global;
    float **stage_base[3], **stage_mask;
    float **base[3], **grad[3], **step[3], **mask, **precond = NULL;
    float **base_copy[3], **grad_copy[3], **step_copy[3], **mask_copy,
          **precond_copy = NULL;
    struct visco_sh_exact_lbfgs lbfgs;
    char path[4096];
    FILE *stream, *out;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 3) fail("usage");
    stream = fopen(argv[1], "rb");
    if (stream == NULL) fail("cannot open case");
    if (fread(header, sizeof(int32_t), HEADER_INTS, stream) != HEADER_INTS ||
            header[0] != CASE_MAGIC) fail("bad header");
    op = header[1]; nxg = header[2]; nyg = header[3];
    nprocx = header[4]; nprocy = header[5]; memory = header[6];
    niter = header[7]; has_precond = header[8]; fault = header[9];
    fault_rank = header[10]; fault_iter = header[11]; skip_begin = header[12];
    if (nprocx * nprocy != size) fail("rank count");
    if (nxg % nprocx || nyg % nprocy) fail("decomposition");
    nx = nxg / nprocx; ny = nyg / nprocy;
    pos_x = rank % nprocx; pos_y = rank / nprocx;
    cells = (size_t)nxg * (size_t)nyg;

    memset(&lbfgs, 0, sizeof(lbfgs));
    init_status = visco_sh_exact_lbfgs_init(
            fault_here(fault, fault_rank, rank, 1) && op == 2 ? NULL : &lbfgs,
            (op == 2 && fault_here(fault, fault_rank, rank, 2)) ? 0 : nx, ny,
            memory);

    snprintf(path, sizeof(path), "%s.%d", argv[2], rank);
    out = fopen(path, "wb");
    if (out == NULL) fail("cannot open output");

    if (op == 2) {
        out_header[0] = RESULT_MAGIC; out_header[1] = rank; out_header[2] = pos_x;
        out_header[3] = pos_y; out_header[4] = nx; out_header[5] = ny;
        out_header[6] = 0; out_header[7] = init_status; out_header[8] = 0;
        fwrite(out_header, sizeof(int32_t), 9, out);
        fclose(out);
        fclose(stream);
        visco_sh_exact_lbfgs_free(&lbfgs);
        visco_sh_exact_lbfgs_free(&lbfgs);   /* second free must be a no-op */
        visco_sh_exact_lbfgs_free(NULL);
        MPI_Finalize();
        return 0;
    }
    if (op != 1) fail("op");

    global = (float *)malloc(cells * sizeof(float));
    if (global == NULL) fail("memory");
    for (c = 0; c < 3; ++c) {
        stage_base[c] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        base[c] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        grad[c] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        step[c] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        base_copy[c] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        grad_copy[c] = allocate_local(nx, ny, OUTPUT_SENTINEL);
        step_copy[c] = allocate_local(nx, ny, OUTPUT_SENTINEL);
    }
    stage_mask = allocate_local(nx, ny, OUTPUT_SENTINEL);
    mask = allocate_local(nx, ny, OUTPUT_SENTINEL);
    mask_copy = allocate_local(nx, ny, OUTPUT_SENTINEL);
    if (has_precond) {
        precond = allocate_local(nx, ny, OUTPUT_SENTINEL);
        precond_copy = allocate_local(nx, ny, OUTPUT_SENTINEL);
    }
    for (c = 0; c < 3; ++c)
        read_field(stream, global, cells, stage_base[c], nxg, nx, ny, pos_x, pos_y);
    read_field(stream, global, cells, stage_mask, nxg, nx, ny, pos_x, pos_y);

    if (!skip_begin) {
        float *saved = NULL;
        int f = (fault_iter == -1) ? fault : 0;

        if (fault_here(f, fault_rank, rank, 103)) {
            saved = stage_mask[ny]; stage_mask[ny] = NULL;
        }
        begin_status = visco_sh_exact_lbfgs_begin_stage(
                fault_here(f, fault_rank, rank, 101) ? NULL : &lbfgs,
                stage_base[0], stage_base[1],
                fault_here(f, fault_rank, rank, 102) ? NULL : stage_base[2],
                stage_mask);
        if (saved != NULL) stage_mask[ny] = saved;
    }

    out_header[0] = RESULT_MAGIC; out_header[1] = rank; out_header[2] = pos_x;
    out_header[3] = pos_y; out_header[4] = nx; out_header[5] = ny;
    out_header[6] = niter; out_header[7] = init_status;
    out_header[8] = begin_status;
    fwrite(out_header, sizeof(int32_t), 9, out);

    for (k = 0; k < niter; ++k) {
        int32_t kind, flags[8];
        float weights[3];
        double scalars[7];
        struct visco_sh_exact_lbfgs_request request;
        struct visco_sh_exact_lbfgs_report report;
        int f = (fault_iter == k) ? fault : 0, status = 0, iter_begin = 0;
        int inputs_unchanged = 1, report_untouched;
        float *saved_row = NULL, **saved_owner = NULL;

        if (fread(&kind, sizeof(int32_t), 1, stream) != 1) fail("bad kind");
        if (fread(weights, sizeof(float), 3, stream) != 3) fail("bad weights");
        for (c = 0; c < 3; ++c)
            read_field(stream, global, cells, base[c], nxg, nx, ny, pos_x, pos_y);
        for (c = 0; c < 3; ++c)
            read_field(stream, global, cells, grad[c], nxg, nx, ny, pos_x, pos_y);
        for (c = 0; c < 3; ++c)
            read_field(stream, global, cells, step[c], nxg, nx, ny, pos_x, pos_y);
        read_field(stream, global, cells, mask, nxg, nx, ny, pos_x, pos_y);
        if (has_precond)
            read_field(stream, global, cells, precond, nxg, nx, ny, pos_x, pos_y);
        for (c = 0; c < 3; ++c) {
            copy_local(base_copy[c], base[c], nx, ny);
            copy_local(grad_copy[c], grad[c], nx, ny);
            copy_local(step_copy[c], step[c], nx, ny);
        }
        copy_local(mask_copy, mask, nx, ny);
        if (has_precond) copy_local(precond_copy, precond, nx, ny);

        if (kind == 1 || kind == 2) {
            float *saved = NULL;

            if (fault_here(f, fault_rank, rank, 103)) {
                saved = mask[ny]; mask[ny] = NULL;
            }
            iter_begin = visco_sh_exact_lbfgs_begin_stage(
                    fault_here(f, fault_rank, rank, 101) ? NULL : &lbfgs,
                    base[0], base[1],
                    fault_here(f, fault_rank, rank, 102) ? NULL : base[2], mask);
            if (saved != NULL) mask[ny] = saved;
        }

        memset(&report, 0, sizeof(report));
        report.action = report.pair = report.pairs_used = REPORT_SENTINEL;
        report.curvature = report.gamma = report.descent = report.kappa =
                (double)REPORT_SENTINEL;
        report.sigma[0] = report.sigma[1] = report.sigma[2] =
                (double)REPORT_SENTINEL;

        if (kind != 2) {
            memset(&request, 0, sizeof(request));
            request.nx = fault_here(f, fault_rank, rank, 4) ? nx + 1 : nx;
            request.ny = fault_here(f, fault_rank, rank, 11) ? ny + 1 : ny;
            request.base_primary = base[0];
            request.base_rho = fault_here(f, fault_rank, rank, 5) ? NULL : base[1];
            request.base_q = base[2];
            request.grad_primary = grad[0];
            request.grad_rho = grad[1];
            request.grad_q = fault_here(f, fault_rank, rank, 6) ? NULL : grad[2];
            request.update_mask = fault_here(f, fault_rank, rank, 7) ? NULL : mask;
            request.precond = has_precond ? precond : NULL;
            request.weight_primary = weights[0];
            request.weight_rho = weights[1];
            request.weight_q = weights[2];
            request.step_primary = fault_here(f, fault_rank, rank, 8) ? NULL : step[0];
            request.step_rho = step[1];
            request.step_q = step[2];
            if (fault_here(f, fault_rank, rank, 9)) {
                saved_owner = grad[0]; saved_row = grad[0][ny]; grad[0][ny] = NULL;
            }
            if (fault_here(f, fault_rank, rank, 10) && has_precond) {
                saved_owner = precond; saved_row = precond[ny]; precond[ny] = NULL;
            }
            status = visco_sh_exact_lbfgs_step(
                    fault_here(f, fault_rank, rank, 1) ? NULL : &lbfgs,
                    fault_here(f, fault_rank, rank, 2) ? NULL : &request,
                    fault_here(f, fault_rank, rank, 3) ? NULL : &report);
            if (saved_owner != NULL) saved_owner[ny] = saved_row;
        }

        for (c = 0; c < 3; ++c) {
            inputs_unchanged &= local_equal(base[c], base_copy[c], nx, ny);
            inputs_unchanged &= local_equal(grad[c], grad_copy[c], nx, ny);
        }
        inputs_unchanged &= local_equal(mask, mask_copy, nx, ny);
        if (has_precond)
            inputs_unchanged &= local_equal(precond, precond_copy, nx, ny);
        report_untouched = (report.action == REPORT_SENTINEL) &&
                (report.pair == REPORT_SENTINEL) &&
                (report.pairs_used == REPORT_SENTINEL) &&
                (report.kappa == (double)REPORT_SENTINEL) &&
                (report.sigma[0] == (double)REPORT_SENTINEL);

        flags[0] = status; flags[1] = iter_begin; flags[2] = report.action;
        flags[3] = report.pair; flags[4] = report.pairs_used;
        flags[5] = local_equal(step[0], step_copy[0], nx, ny) &&
                   local_equal(step[1], step_copy[1], nx, ny) &&
                   local_equal(step[2], step_copy[2], nx, ny);
        flags[6] = inputs_unchanged;
        flags[7] = report_untouched;
        scalars[0] = report.curvature; scalars[1] = report.gamma;
        scalars[2] = report.descent; scalars[3] = report.kappa;
        scalars[4] = report.sigma[0]; scalars[5] = report.sigma[1];
        scalars[6] = report.sigma[2];
        fwrite(flags, sizeof(int32_t), 8, out);
        fwrite(scalars, sizeof(double), 7, out);
        for (c = 0; c < 3; ++c)
            for (j = 0; j < ny + 2; ++j)
                fwrite(step[c][j], sizeof(float), (size_t)(nx + 2), out);
    }
    fclose(out);
    fclose(stream);
    visco_sh_exact_lbfgs_free(&lbfgs);
    MPI_Finalize();
    return 0;
}
