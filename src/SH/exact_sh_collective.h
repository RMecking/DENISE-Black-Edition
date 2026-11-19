#ifndef DENISE_EXACT_SH_COLLECTIVE_H
#define DENISE_EXACT_SH_COLLECTIVE_H

static inline int exact_sh_abort_phase(MPI_Comm comm, const char *phase,
                                       int status) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    fprintf(stderr, "Exact SH: %s aborted on rank %d (status %d)\n",
            phase, rank, status);
    fflush(stderr);
    MPI_Abort(comm, 1);
    return status;
}

/* The exact SH driver and the material transpose use MPI_COMM_WORLD.
 * Report the failing rank before peers are released from this phase. */
static inline int exact_sh_collective_status(MPI_Comm comm, const char *phase,
                                            int local_status) {
    int codes[2] = {local_status < 0 ? local_status : 0,
                    local_status > 0 ? -local_status : 0};
    int shared[2];
    if (local_status != 0) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        fprintf(stderr, "Exact SH: %s failed on rank %d (status %d)\n",
                phase, rank, local_status);
        fflush(stderr);
    }
    if (MPI_Allreduce(codes, shared, 2, MPI_INT, MPI_MIN, comm) != MPI_SUCCESS)
        return exact_sh_abort_phase(comm, phase, -2);
    return shared[0] != 0 ? shared[0] : -shared[1];
}

#endif
