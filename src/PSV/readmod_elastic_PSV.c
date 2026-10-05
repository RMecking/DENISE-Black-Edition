/* Read the legacy native-FP32 elastic model in global x-major order. */
#include "fd.h"
#include <limits.h>
#include <stdint.h>

void readmod_elastic_PSV(float **rho, float **pi, float **u) {
    extern int NX, NY, NXG, NYG, POS[3], MYID, INVMAT1, WRITEMOD;
    extern char MFILE[STRING_SIZE];
    extern FILE *FP;
    FILE *streams[3] = {NULL, NULL, NULL};
    const char *suffix[3];
    char paths[3][STRING_SIZE + 5], filename[STRING_SIZE2], error[512] = "";
    float *owned = NULL, values[3];
    size_t cells = 0, slot;
    int i, j, k, rank, failed_rank, first_failed_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(FP, "\n...reading model information from modell-files...\n");
    if (!memchr(MFILE, '\0', sizeof(MFILE))) {
        snprintf(error, sizeof(error), "Elastic model preflight: unterminated model filename");
        goto finish;
    }
    if (INVMAT1 != 1 && INVMAT1 != 3) {
        snprintf(error, sizeof(error), "Elastic model preflight: unsupported INVMAT1=%d", INVMAT1);
        goto finish;
    }
    if (NX <= 0 || NY <= 0 || NXG < NX || NYG < NY || POS[1] < 0 || POS[2] < 0
        || POS[1] > (NXG - NX) / NX || POS[2] > (NYG - NY) / NY
        || (size_t)NX > SIZE_MAX / (size_t)NY
        || (size_t)NX * (size_t)NY > SIZE_MAX / (3 * sizeof(float))) {
        snprintf(error, sizeof(error), "Elastic model preflight: invalid model dimensions or ownership");
        goto finish;
    }
    cells = (size_t)NX * (size_t)NY;
    owned = malloc(3 * cells * sizeof(float));
    if (!owned) {
        snprintf(error, sizeof(error), "Elastic model preflight: cannot allocate local input buffer");
        goto finish;
    }
    suffix[0] = INVMAT1 == 1 ? ".vp" : ".lam";
    suffix[1] = INVMAT1 == 1 ? ".vs" : ".mu";
    suffix[2] = ".rho";
    for (k = 0; k < 3; ++k) {
        snprintf(paths[k], sizeof(paths[k]), "%s%s", MFILE, suffix[k]);
        fprintf(FP, "\t %s\n", paths[k]);
        streams[k] = fopen(paths[k], "rb");
        if (!streams[k]) {
            snprintf(error, sizeof(error), "Elastic model preflight: cannot open %s", paths[k]);
            goto finish;
        }
    }

    /* Every rank validates the entire input, including cells it does not own.
       Keep caller arrays unchanged until all ranks have accepted the files. */
    for (i = 0; i < NXG; ++i) {
        for (j = 0; j < NYG; ++j) {
            for (k = 0; k < 3; ++k) {
                if (fread(&values[k], sizeof(float), 1, streams[k]) != 1) {
                    snprintf(error, sizeof(error), "Elastic model preflight: short/read failure in %s at x=%d y=%d",
                             paths[k], i + 1, j + 1);
                    goto finish;
                }
                if (!isfinite(values[k])) {
                    snprintf(error, sizeof(error), "Elastic model preflight: nonfinite value in %s at x=%d y=%d",
                             paths[k], i + 1, j + 1);
                    goto finish;
                }
            }
            if (!(values[2] > 0.0f) || values[1] < 0.0f
                || (INVMAT1 == 1 && !(values[0] > 0.0f))) {
                k = !(values[2] > 0.0f) ? 2 : (values[1] < 0.0f ? 1 : 0);
                snprintf(error, sizeof(error), "Elastic model preflight: inadmissible material in %s at x=%d y=%d",
                         paths[k], i + 1, j + 1);
                goto finish;
            }
            /* Check the raw lambda/mu contract in widened arithmetic. Negative
               solid lambda and exact signed-zero fluid mu remain admissible. */
            if (INVMAT1 == 3 && !((double)values[0] + 2.0 * (double)values[1] > 0.0)) {
                snprintf(error, sizeof(error), "Elastic model preflight: lambda+2mu must be positive in %s / %s at x=%d y=%d",
                         paths[0], paths[1], i + 1, j + 1);
                goto finish;
            }
            if (POS[1] == i / NX && POS[2] == j / NY) {
                slot = (size_t)(i - POS[1] * NX) * (size_t)NY + (size_t)(j - POS[2] * NY);
                for (k = 0; k < 3; ++k) owned[(size_t)k * cells + slot] = values[k];
            }
        }
    }
    for (k = 0; k < 3; ++k) {
        int extra = fgetc(streams[k]);
        if (extra != EOF || ferror(streams[k])) {
            snprintf(error, sizeof(error), "Elastic model preflight: trailing bytes/read failure in %s", paths[k]);
            goto finish;
        }
    }

finish:
    for (k = 0; k < 3; ++k) {
        if (streams[k] && fclose(streams[k]) != 0 && !error[0])
            snprintf(error, sizeof(error), "Elastic model preflight: close failure in %s", paths[k]);
    }
    /* Choose one reproducible diagnostic; no rank may enter model output or
       downstream material work while another rank rejects its input. */
    failed_rank = error[0] ? rank : INT_MAX;
    MPI_Allreduce(&failed_rank, &first_failed_rank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (first_failed_rank != INT_MAX) {
        MPI_Bcast(error, sizeof(error), MPI_CHAR, first_failed_rank, MPI_COMM_WORLD);
        free(owned);
        err(error);
        return;
    }
    for (i = 1; i <= NX; ++i) {
        for (j = 1; j <= NY; ++j) {
            slot = (size_t)(i - 1) * (size_t)NY + (size_t)(j - 1);
            pi[j][i] = owned[slot];
            u[j][i] = owned[cells + slot];
            rho[j][i] = owned[2 * cells + slot];
        }
    }
    free(owned);

    /* Preserve the legacy output names and raw parameter representation. */
    if (WRITEMOD) {
        snprintf(filename, sizeof(filename), "%s.denise.pi", MFILE);
        writemod(filename, pi, 3);
        MPI_Barrier(MPI_COMM_WORLD);
        if (MYID == 0) mergemod(filename, 3);
        snprintf(filename, sizeof(filename), "%s.denise.mu", MFILE);
        writemod(filename, u, 3);
        MPI_Barrier(MPI_COMM_WORLD);
        if (MYID == 0) mergemod(filename, 3);
        snprintf(filename, sizeof(filename), "%s.denise.rho", MFILE);
        writemod(filename, rho, 3);
        MPI_Barrier(MPI_COMM_WORLD);
        if (MYID == 0) mergemod(filename, 3);
    }
}




