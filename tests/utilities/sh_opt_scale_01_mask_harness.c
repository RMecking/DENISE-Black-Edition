/* SH-OPT-SCALE-01 harness for the CPML update-mask builder.
 *
 * Contract: docs/sh_opt_scale_01_contract.md (revision 2), section 3
 * (oracles O6, O8).  Linked against src/SH/visco_sh_exact_step_scaling.c
 * alone, which the contract requires to be stand-alone linkable.
 * Reads one configuration per line from <config.txt>:
 *   nx ny pos_x pos_y nprocx nprocy fw free_surf boundary null_mask
 * and prints one JSON object per configuration with the return code and the
 * complete local array including its halo.  The array is prefilled with a
 * sentinel; invalid nx/ny use a 1x1 interior so "untouched" stays checkable.
 */

#include "fd.h"

#define MASK_SENTINEL (7.0f)

void err(char err_text[]) {
    fprintf(stderr, "unexpected err(): %s\n", err_text);
    exit(91);
}

int main(int argc, char **argv) {
    int nx, ny, pos_x, pos_y, nprocx, nprocy, fw, free_surf, boundary, null_mask;
    int rows, cols, status, i, j;
    float **mask;
    FILE *stream;

    MPI_Init(&argc, &argv);
    if (argc != 2) {
        fprintf(stderr, "usage: harness <config.txt>\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    stream = fopen(argv[1], "r");
    if (stream == NULL) MPI_Abort(MPI_COMM_WORLD, 3);
    while (fscanf(stream, "%d %d %d %d %d %d %d %d %d %d", &nx, &ny, &pos_x,
                  &pos_y, &nprocx, &nprocy, &fw, &free_surf, &boundary,
                  &null_mask) == 10) {
        cols = ((nx >= 1) ? nx : 1) + 2;
        rows = ((ny >= 1) ? ny : 1) + 2;
        mask = (float **)malloc((size_t)rows * sizeof(*mask));
        if (mask == NULL) MPI_Abort(MPI_COMM_WORLD, 4);
        for (j = 0; j < rows; ++j) {
            mask[j] = (float *)malloc((size_t)cols * sizeof(float));
            if (mask[j] == NULL) MPI_Abort(MPI_COMM_WORLD, 4);
            for (i = 0; i < cols; ++i) mask[j][i] = MASK_SENTINEL;
        }
        status = visco_sh_exact_build_update_mask(null_mask ? NULL : mask,
                nx, ny, pos_x, pos_y, nprocx, nprocy, fw, free_surf, boundary);
        printf("{\"status\":%d,\"rows\":%d,\"cols\":%d,\"values\":[", status, rows, cols);
        for (j = 0; j < rows; ++j)
            for (i = 0; i < cols; ++i)
                printf("%s%.9g", (j == 0 && i == 0) ? "" : ",", (double)mask[j][i]);
        printf("]}\n");
        for (j = 0; j < rows; ++j) free(mask[j]);
        free(mask);
    }
    fclose(stream);
    fflush(stdout);
    MPI_Finalize();
    return 0;
}
