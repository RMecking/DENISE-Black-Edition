/* Exact transpose of the two-stage cyclic SH material halo exchange. */

#include "fd.h"

static void pack_cell(float *buffer, int offset,
                      float **rho, float **u, float **taus,
                      int j, int i) {
    buffer[3 * offset] = rho[j][i];
    buffer[3 * offset + 1] = u[j][i];
    buffer[3 * offset + 2] = taus[j][i];
}

static void add_cell(const float *buffer, int offset,
                     float **rho, float **u, float **taus,
                     int j, int i) {
    rho[j][i] += buffer[3 * offset];
    u[j][i] += buffer[3 * offset + 1];
    taus[j][i] += buffer[3 * offset + 2];
}

static void clear_cell(float **rho, float **u, float **taus, int j, int i) {
    rho[j][i] = 0.0f;
    u[j][i] = 0.0f;
    taus[j][i] = 0.0f;
}

int matcopy_SH_adjoint(float **bar_rho, float **bar_u, float **bar_taus) {
    extern int NX, NY, INDEX[5];
    float *send_right, *recv_left, *send_left, *recv_right;
    float *send_bottom, *recv_top, *send_top, *recv_bottom;
    MPI_Status status;
    int horizontal_count = 3 * (NY + 2);
    int vertical_count = 3 * NX;
    int i, j, status_code = MPI_SUCCESS;

    if (bar_rho == NULL || bar_u == NULL || bar_taus == NULL)
        return MPI_ERR_ARG;

    send_right = (float *)malloc((size_t)horizontal_count * sizeof(float));
    recv_left = (float *)malloc((size_t)horizontal_count * sizeof(float));
    send_left = (float *)malloc((size_t)horizontal_count * sizeof(float));
    recv_right = (float *)malloc((size_t)horizontal_count * sizeof(float));
    send_bottom = (float *)malloc((size_t)vertical_count * sizeof(float));
    recv_top = (float *)malloc((size_t)vertical_count * sizeof(float));
    send_top = (float *)malloc((size_t)vertical_count * sizeof(float));
    recv_bottom = (float *)malloc((size_t)vertical_count * sizeof(float));
    if (send_right == NULL || recv_left == NULL || send_left == NULL ||
        recv_right == NULL || send_bottom == NULL || recv_top == NULL ||
        send_top == NULL || recv_bottom == NULL) {
        status_code = MPI_ERR_NO_MEM;
        goto cleanup;
    }

    /* H^T: right-halo bars return right, left-halo bars return left. */
    for (j = 0; j <= NY + 1; ++j) {
        pack_cell(send_right, j, bar_rho, bar_u, bar_taus, j, NX + 1);
        pack_cell(send_left, j, bar_rho, bar_u, bar_taus, j, 0);
    }
    status_code = MPI_Sendrecv(send_right, horizontal_count, MPI_FLOAT,
            INDEX[2], 6311, recv_left, horizontal_count, MPI_FLOAT,
            INDEX[1], 6311, MPI_COMM_WORLD, &status);
    if (status_code != MPI_SUCCESS) goto cleanup;
    status_code = MPI_Sendrecv(send_left, horizontal_count, MPI_FLOAT,
            INDEX[1], 6312, recv_right, horizontal_count, MPI_FLOAT,
            INDEX[2], 6312, MPI_COMM_WORLD, &status);
    if (status_code != MPI_SUCCESS) goto cleanup;
    for (j = 0; j <= NY + 1; ++j) {
        clear_cell(bar_rho, bar_u, bar_taus, j, 0);
        clear_cell(bar_rho, bar_u, bar_taus, j, NX + 1);
        add_cell(recv_left, j, bar_rho, bar_u, bar_taus, j, 1);
        add_cell(recv_right, j, bar_rho, bar_u, bar_taus, j, NX);
    }

    /* V^T: bottom-halo bars return down, top-halo bars return up. */
    for (i = 1; i <= NX; ++i) {
        pack_cell(send_bottom, i - 1, bar_rho, bar_u, bar_taus, NY + 1, i);
        pack_cell(send_top, i - 1, bar_rho, bar_u, bar_taus, 0, i);
    }
    status_code = MPI_Sendrecv(send_bottom, vertical_count, MPI_FLOAT,
            INDEX[4], 6313, recv_top, vertical_count, MPI_FLOAT,
            INDEX[3], 6313, MPI_COMM_WORLD, &status);
    if (status_code != MPI_SUCCESS) goto cleanup;
    status_code = MPI_Sendrecv(send_top, vertical_count, MPI_FLOAT,
            INDEX[3], 6314, recv_bottom, vertical_count, MPI_FLOAT,
            INDEX[4], 6314, MPI_COMM_WORLD, &status);
    if (status_code != MPI_SUCCESS) goto cleanup;
    for (i = 1; i <= NX; ++i) {
        clear_cell(bar_rho, bar_u, bar_taus, 0, i);
        clear_cell(bar_rho, bar_u, bar_taus, NY + 1, i);
        add_cell(recv_top, i - 1, bar_rho, bar_u, bar_taus, 1, i);
        add_cell(recv_bottom, i - 1, bar_rho, bar_u, bar_taus, NY, i);
    }

cleanup:
    free(send_right);
    free(recv_left);
    free(send_left);
    free(recv_right);
    free(send_bottom);
    free(recv_top);
    free(send_top);
    free(recv_bottom);
    return status_code;
}
