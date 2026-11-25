/* Review K-4: psv() must refuse an adjoint call (mode 1) when the legacy
 * forward-wavefield storage was not allocated in full, as on the exact
 * visco-PSV FWI path (alloc_fwiPSV with MODE == 1 && L > 0).
 *
 * Linked against the production objects of `make denise` without denise.o.
 * argv[1]: "short"  -> forward_prop_length = 1, mode 1: must abort with the
 *                      guard message before touching any field;
 *          "length" -> prints the length alloc_fwiPSV sets for the MODE and
 *                      L given in argv[2], argv[3] (no wave propagation). */
#include "fd.h"
#include "globvar.h"

int main(int argc, char **argv) {
    struct wavePSV wave; struct wavePSV_PML pml; struct matPSV mat;
    struct fwiPSV fwi; struct mpiPSV mpi; struct seisPSV seis;
    struct seisPSVfwi seisfwi; struct acq acq;
    MPI_Request send[4], recv[4];
    int help[8] = {0};

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MYID);
    memset(&wave, 0, sizeof(wave)); memset(&pml, 0, sizeof(pml));
    memset(&mat, 0, sizeof(mat)); memset(&fwi, 0, sizeof(fwi));
    memset(&mpi, 0, sizeof(mpi)); memset(&seis, 0, sizeof(seis));
    memset(&seisfwi, 0, sizeof(seisfwi)); memset(&acq, 0, sizeof(acq));
    FDORDER = 4; NX = 8; NY = 8; NT = 6; NTDTINV = 6; IDXI = IDYI = 1;
    NXNYI = NX * NY; INVMAT1 = 1; GRAD_FORM = 2; FW = 2;

    if (argc >= 2 && strcmp(argv[1], "short") == 0) {
        MODE = 1; L = 1;
        fwi.forward_prop_length = 1;
        psv(&wave, &pml, &mat, &fwi, &mpi, &seis, &seisfwi, &acq, NULL, 1, 1, 0,
            NT, 0, NULL, NULL, 1, help, 1, send, recv);
        printf("guard did not fire\n");
    } else if (argc >= 4 && strcmp(argv[1], "length") == 0) {
        MODE = atoi(argv[2]); L = atoi(argv[3]);
        alloc_fwiPSV(&fwi);
        printf("{\"mode\": %d, \"L\": %d, \"length\": %zu, \"full\": %zu}\n",
               MODE, L, fwi.forward_prop_length, (size_t)NXNYI * (size_t)NTDTINV);
    }
    MPI_Finalize();
    return 0;
}
