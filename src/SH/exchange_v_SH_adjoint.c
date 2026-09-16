/*------------------------------------------------------------------------
 * Exact Euclidean transpose of the SH halo-copy field map.
 *----------------------------------------------------------------------*/

#include "fd.h"

static int exchange_pair(
        float **bar_field, int nx, int ny, int depth, int vertical,
        int negative_active, int positive_active, int negative_rank,
        int positive_rank, MPI_Comm comm){

    float *send_negative, *send_positive, *recv_negative, *recv_positive;
    int count, coordinate, layer, offset, status_code;
    MPI_Status status;

    if (depth <= 0 || (!negative_active && !positive_active)) return MPI_SUCCESS;

    count = (vertical ? nx : ny) * depth;
    send_negative = (float *)calloc((size_t)count, sizeof(float));
    send_positive = (float *)calloc((size_t)count, sizeof(float));
    recv_negative = (float *)calloc((size_t)count, sizeof(float));
    recv_positive = (float *)calloc((size_t)count, sizeof(float));
    if (!send_negative || !send_positive || !recv_negative || !recv_positive){
        free(send_negative);
        free(send_positive);
        free(recv_negative);
        free(recv_positive);
        MPI_Abort(comm, 91);
        return 91;
    }

    offset = 0;
    for (coordinate = 1; coordinate <= (vertical ? nx : ny); ++coordinate){
        for (layer = 1; layer <= depth; ++layer){
            if (vertical){
                if (negative_active){
                    send_negative[offset] = bar_field[1-layer][coordinate];
                    bar_field[1-layer][coordinate] = 0.0f;
                }
                if (positive_active){
                    send_positive[offset] = bar_field[ny+layer][coordinate];
                    bar_field[ny+layer][coordinate] = 0.0f;
                }
            } else {
                if (negative_active){
                    send_negative[offset] = bar_field[coordinate][1-layer];
                    bar_field[coordinate][1-layer] = 0.0f;
                }
                if (positive_active){
                    send_positive[offset] = bar_field[coordinate][nx+layer];
                    bar_field[coordinate][nx+layer] = 0.0f;
                }
            }
            ++offset;
        }
    }

    status_code = MPI_Sendrecv(
            send_negative, count, MPI_FLOAT,
            negative_active ? negative_rank : MPI_PROC_NULL, vertical ? 605 : 602,
            recv_negative, count, MPI_FLOAT,
            positive_active ? positive_rank : MPI_PROC_NULL, vertical ? 605 : 602,
            comm, &status);
    if (status_code != MPI_SUCCESS) goto cleanup;

    status_code = MPI_Sendrecv(
            send_positive, count, MPI_FLOAT,
            positive_active ? positive_rank : MPI_PROC_NULL, vertical ? 606 : 601,
            recv_positive, count, MPI_FLOAT,
            negative_active ? negative_rank : MPI_PROC_NULL, vertical ? 606 : 601,
            comm, &status);
    if (status_code != MPI_SUCCESS) goto cleanup;

    offset = 0;
    for (coordinate = 1; coordinate <= (vertical ? nx : ny); ++coordinate){
        for (layer = 1; layer <= depth; ++layer){
            if (vertical){
                if (positive_active)
                    bar_field[ny-layer+1][coordinate] += recv_negative[offset];
                if (negative_active)
                    bar_field[layer][coordinate] += recv_positive[offset];
            } else {
                if (positive_active)
                    bar_field[coordinate][nx-layer+1] += recv_negative[offset];
                if (negative_active)
                    bar_field[coordinate][layer] += recv_positive[offset];
            }
            ++offset;
        }
    }

cleanup:
    free(send_negative);
    free(send_positive);
    free(recv_negative);
    free(recv_positive);
    return status_code;
}

int visco_sh_exchange_field_adjoint(
        float **bar_field, int nx, int ny, int vertical_depth,
        int horizontal_depth, int boundary, const int pos[3],
        int nproc_x, int nproc_y, const int index[5], MPI_Comm comm){

    int status_code;

    if (!bar_field || !pos || !index || nx <= 0 || ny <= 0 ||
            vertical_depth < 0 || horizontal_depth < 0 ||
            nproc_x <= 0 || nproc_y <= 0) return 1;

    status_code = exchange_pair(
            bar_field, nx, ny, vertical_depth, 1,
            pos[2] != 0, pos[2] != nproc_y-1,
            index[3], index[4], comm);
    if (status_code != MPI_SUCCESS) return status_code;

    return exchange_pair(
            bar_field, nx, ny, horizontal_depth, 0,
            boundary || pos[1] != 0,
            boundary || pos[1] != nproc_x-1,
            index[1], index[2], comm);
}

int exchange_v_SH_adjoint(
        float **bar_vz, int nx, int ny, int fdorder, int boundary,
        const int pos[3], int nproc_x, int nproc_y,
        const int index[5], MPI_Comm comm){

    if (fdorder < 2 || (fdorder % 2) != 0) return 1;
    return visco_sh_exchange_field_adjoint(
            bar_vz, nx, ny, fdorder/2+1, fdorder/2, boundary,
            pos, nproc_x, nproc_y, index, comm);
}
