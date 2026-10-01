#ifndef DENISE_CUDA_M9_ELASTIC_H
#define DENISE_CUDA_M9_ELASTIC_H
/* INTERNAL verification ABI: isolated M9 elastic forward, never MODE=2. */
#include "denise_elastic_psv_born.h"
#ifdef __cplusplus
extern "C" {
#endif
struct denise_cuda_m9;
struct denise_cuda_m9_options { int device; size_t cap_bytes; size_t reserve_bytes; };
struct denise_cuda_m9_diagnostics {
 size_t mandatory_bytes,usable_budget,remaining_budget,owned_bytes;
 size_t model_bytes,wavefield_bytes,cpml_bytes,profile_bytes,source_bytes;
 size_t receiver_bytes,trajectory_bytes,workspace_bytes,host_metadata_bytes;
 int valid_steps,prepared,visible_devices,runtime_version,major,minor;
 float elapsed_ms;
};
const char *denise_cuda_m9_last_error(void);
int denise_cuda_m9_create(const struct denise_elastic_psv_born_config *,
                         const struct denise_cuda_m9_options *,struct denise_cuda_m9 **);
int denise_cuda_m9_prepare(struct denise_cuda_m9 *);
int denise_cuda_m9_nonlinear(struct denise_cuda_m9 *,const float *,const float *);
/* Data [t][receiver][vx,vy], fields [5][y][x], psi [8][y][x], operands [t][4][y][x]. */
int denise_cuda_m9_download(struct denise_cuda_m9 *,float *data,float *fields,float *psi,float *operands);
int denise_cuda_m9_diagnostics(const struct denise_cuda_m9 *,struct denise_cuda_m9_diagnostics *);
int denise_cuda_m9_destroy(struct denise_cuda_m9 **);
/* Candidate-owned fault/ledger diagnostics. at=0 disables injection; at>0 fails
 * one indexed operation before execution. Cleanup is never fault-injected. */
void denise_cuda_m9_fault(size_t at);
void denise_cuda_m9_ledger(size_t *device_bytes,size_t *host_bytes,size_t *events,size_t *calls);
/* Test-only operations invalidate prepared state. Arbitrary physical/psi state,
 * optional packed profiles, and actual production halos are exposed explicitly. */
int denise_cuda_m9_test_step(struct denise_cuda_m9 *,const float *state13,const float *profiles);
int denise_cuda_m9_test_halo(struct denise_cuda_m9 *,int velocity,float *five_padded);
int denise_cuda_m9_test_static(struct denise_cuda_m9 *,float *maps,float *profiles,float *source,int *geometry);
/* Pure copy layout gate: overwrite/download FULL operands, invalidate results. */
int denise_cuda_m9_test_trajectory_roundtrip(struct denise_cuda_m9 *,float *operands);
#ifdef __cplusplus
}
#endif
#endif
