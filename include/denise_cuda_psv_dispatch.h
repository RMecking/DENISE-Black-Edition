#ifndef DENISE_CUDA_PSV_DISPATCH_H
#define DENISE_CUDA_PSV_DISPATCH_H

#include "denise_cuda_psv_forward.h"

struct wavePSV;
struct wavePSV_PML;
struct matPSV;
struct seisPSV;
struct acq;

enum denise_psv_backend {
    DENISE_PSV_BACKEND_CPU = 0,
    DENISE_PSV_BACKEND_CUDA = 1
};

/* Classifies the provisional selector and validates every deterministic B2A
 * CUDA precondition available at psv() entry.  This call does not mutate
 * solver state or allocate device memory. */
int denise_cuda_psv_backend_preflight(
        int nsrc_local,
        int ntr,
        int mode,
        enum denise_psv_backend *backend);

/* Returns 0 for the existing CPU path, 1 after a completed CUDA forward run,
 * and -1 for a preflighted CUDA request that failed during execution. */
int denise_cuda_psv_dispatch(
        struct wavePSV *wave,
        struct wavePSV_PML *pml,
        struct matPSV *material,
        struct seisPSV *seismogram,
        struct acq *acquisition,
        float *hc,
        int nsrc_local,
        int ntr,
        int mode,
        enum denise_psv_backend backend);

const char *denise_cuda_psv_dispatch_last_error(void);
int denise_cuda_psv_dispatch_last_stats(
        struct denise_cuda_psv_forward_stats *stats);

#endif
