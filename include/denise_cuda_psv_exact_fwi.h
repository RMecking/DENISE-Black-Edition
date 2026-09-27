#ifndef DENISE_CUDA_PSV_EXACT_FWI_H
#define DENISE_CUDA_PSV_EXACT_FWI_H

#include "denise_cuda_psv_forward.h"

struct visco_psv_exact_fwi_request;

/* These hooks are linked only into the optional CUDA executable. */
int denise_cuda_psv_exact_fwi_selected(void);
int denise_cuda_psv_exact_fwi_original_forward(
        const struct denise_cuda_psv_forward_config *config,
        const struct denise_cuda_psv_forward_host *host,
        float **sectionvx,
        float **sectionvy);
int denise_cuda_psv_exact_fwi_finish(
        const struct visco_psv_exact_fwi_request *request);
void denise_cuda_psv_exact_fwi_trial_complete(
        const struct denise_cuda_psv_forward_stats *stats);
void denise_cuda_psv_exact_fwi_trial_failed(const char *message);
const char *denise_cuda_psv_exact_fwi_last_error(void);

#endif
