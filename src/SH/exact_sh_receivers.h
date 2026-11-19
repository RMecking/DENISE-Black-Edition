#ifndef DENISE_EXACT_SH_RECEIVERS_H
#define DENISE_EXACT_SH_RECEIVERS_H

#include "exact_sh_collective.h"

static inline int exact_sh_global_receivers_valid(int **positions, int count) {
    extern int NX, NY, NPROCX, NPROCY;
    int i;
    if (count < 1 || positions == NULL || positions[1] == NULL ||
            positions[2] == NULL) return 0;
    for (i = 1; i <= count; ++i)
        if (positions[1][i] < 1 || positions[1][i] > (long long)NX * NPROCX ||
                positions[2][i] < 1 || positions[2][i] > (long long)NY * NPROCY)
            return 0;
    return 1;
}

static inline int exact_sh_local_receivers_valid(int **positions, int count,
        float **synthetic, float **observed) {
    extern int NX, NY;
    int i;
    if (count == 0) return 1;
    if (count < 0 || positions == NULL || positions[1] == NULL ||
            positions[2] == NULL || synthetic == NULL || observed == NULL)
        return 0;
    for (i = 1; i <= count; ++i)
        if (positions[1][i] < 1 || positions[1][i] > NX ||
                positions[2][i] < 1 || positions[2][i] > NY ||
                synthetic[i] == NULL || observed[i] == NULL) return 0;
    return 1;
}

/* Match splitrec's ordered local-to-global mapping before any trace I/O. */
static inline int exact_sh_receiver_status(const struct acq *acquisition,
        int nrec_local, int nrec_global, float **synthetic, float **observed,
        float **read_buffer) {
    extern int NX, NY, POS[3];
    int global, local = 0;
    int **positions = acquisition->recpos;
    int **owned = acquisition->recpos_loc;
    if (!exact_sh_global_receivers_valid(positions, nrec_global) ||
            nrec_local < 0 || nrec_local > nrec_global || read_buffer == NULL)
        return -1;
    if (!exact_sh_local_receivers_valid(owned, nrec_local, synthetic, observed) ||
            (nrec_local > 0 && owned[3] == NULL)) return -1;
    for (global = 1; global <= nrec_global; ++global) {
        int x = positions[1][global] - 1;
        int y = positions[2][global] - 1;
        if (read_buffer[global] == NULL) return -1;
        if (x / NX != POS[1] || y / NY != POS[2]) continue;
        if (++local > nrec_local || owned[3][local] != global ||
                owned[1][local] != x % NX + 1 || owned[2][local] != y % NY + 1)
            return -1;
    }
    return local == nrec_local ? 0 : -1;
}

/* Count-only comparisons before choosing shot loop/receiver branches. */
static inline int exact_sh_check_schedule(int nshots, int readrec, int ns) {
    int local[6] = {nshots, -nshots, readrec, -readrec, ns, -ns};
    int minimum[6];
    if (MPI_Allreduce(local, minimum, 6, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD) != MPI_SUCCESS)
        return exact_sh_abort_phase(MPI_COMM_WORLD, "shot schedule exchange", -2);
    if (minimum[0] != -minimum[1] || minimum[2] != -minimum[3] ||
            minimum[4] != -minimum[5]) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Exact SH: inconsistent shot schedule on rank %d "
                "(shots %d, READREC %d, samples %d)\n", rank, nshots, readrec, ns);
        return -1;
    }
    return 0;
}

#endif
