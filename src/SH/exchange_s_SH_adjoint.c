/*------------------------------------------------------------------------
 * Exact Euclidean transpose of the SH stress halo-copy field map.
 *----------------------------------------------------------------------*/

#include "fd.h"

int exchange_s_SH_adjoint(
        float **bar_sxz, float **bar_syz, int nx, int ny, int fdorder,
        int boundary, const int pos[3], int nproc_x, int nproc_y,
        const int index[5], MPI_Comm comm){

    int status_code;

    if (!bar_sxz || !bar_syz || fdorder < 2 || (fdorder % 2) != 0)
        return 1;

    status_code = visco_sh_exchange_field_adjoint(
            bar_syz, nx, ny, fdorder/2+1, 0, boundary,
            pos, nproc_x, nproc_y, index, comm);
    if (status_code != MPI_SUCCESS) return status_code;

    return visco_sh_exchange_field_adjoint(
            bar_sxz, nx, ny, 0, fdorder/2, boundary,
            pos, nproc_x, nproc_y, index, comm);
}
