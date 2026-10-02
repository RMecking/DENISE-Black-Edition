#ifndef DENISE_CUDA_M9_ELASTIC_H
#define DENISE_CUDA_M9_ELASTIC_H
/* Additive internal M9 elastic CUDA ABI. Existing structure sizes are frozen. */
#include "denise_elastic_psv_born.h"
#ifdef __cplusplus
extern "C" {
#endif
struct denise_cuda_m9;
struct denise_cuda_m9_options;
struct denise_cuda_m9_replay_diagnostics {
 size_t checkpoint_values,checkpoint_payload_bytes,checkpoint_metadata_bytes;
 size_t checkpoint_pointer_bytes,segment_schedule_bytes,segment_operand_bytes;
 size_t alignment_bytes,retained_bytes,full_operand_bytes,owned_device_bytes;
 size_t initial_forward_steps,replayed_forward_steps;
 int requested_segments,effective_segments,checkpoint_count,max_segment_length,selected_replay;
};
/* automatic=0 forces replay; automatic=1 uses the strict differential-storage rule. */
int denise_cuda_m9_create_replay(const struct denise_elastic_psv_born_config *,
 const struct denise_cuda_m9_options *,int segments,int automatic,struct denise_cuda_m9 **);
int denise_cuda_m9_replay_diagnostics(const struct denise_cuda_m9 *,struct denise_cuda_m9_replay_diagnostics *);
/* Pure checked estimator: active[k] counts coordinates where the built profile a != 0. */
int denise_cuda_m9_estimate_replay(size_t nx,size_t ny,size_t nt,size_t segments,
 const size_t active[8],struct denise_cuda_m9_replay_diagnostics *);
int denise_cuda_m9_segment_bounds(int nt,int segments,int segment,int *start,int *end);
/* Transactional test probes. time=-1 denotes the restored segment-start state.
 * FULL uses uninterrupted canonical propagation in its already-owned J workspace.
 * Replay restores the requested checkpoint then regenerates up to time. */
int denise_cuda_m9_test_replay_probe(struct denise_cuda_m9 *,int segment,int time,float *state13,float *q4);
int denise_cuda_m9_test_checkpoints(struct denise_cuda_m9 *,float *,size_t values);
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
/* Additive FULL-J constructor: accepts FREE_SURF=0/1. The legacy constructor
 * retains its M9e-1 FREE_SURF=0 contract and original device budget. */
int denise_cuda_m9_create_full(const struct denise_elastic_psv_born_config *,
                             const struct denise_cuda_m9_options *,struct denise_cuda_m9 **);
struct denise_cuda_m9_born_diagnostics {
 size_t physical_bytes,cpml_bytes,direction_bytes,corner_bytes,operand_bytes,data_bytes;
 int valid;
 float elapsed_ms;
};
/* Additive FULL J/JT arena. Legacy constructors/budgets remain unchanged. */
int denise_cuda_m9_create_migration(const struct denise_elastic_psv_born_config *,
                                  const struct denise_cuda_m9_options *,struct denise_cuda_m9 **);
struct denise_cuda_m9_adjoint_diagnostics {
 size_t field_bytes,cpml_bytes,image_bytes,data_bytes,workspace_bytes,alignment_bytes;
 int valid;
 float elapsed_ms;
};
int denise_cuda_m9_apply_jt(struct denise_cuda_m9 *,const float *,size_t values);
int denise_cuda_m9_image_download(struct denise_cuda_m9 *,double *,double *,size_t cells);
int denise_cuda_m9_migrate(struct denise_cuda_m9 *,const float *,size_t values,double *,double *,size_t cells);
int denise_cuda_m9_adjoint_diagnostics(const struct denise_cuda_m9 *,struct denise_cuda_m9_adjoint_diagnostics *);
/* Test-only reverse blocks, transactional in/out padded fields, compact psi8,
 * q4, images2. Modes: 0 halo(first field), 1 CPML(kind,q0), 2 FD(kind,field),
 * 3 surface ghost(field=0/1), 4 mirror(field=3/4), 5 projection,
 * 6 material images/q, 7 receiver(time), 8 surface A closure, 9 harmonic(q3),
 * 10 complete source-free reverse step (residual time zero).
 * The background is compact q4; residual chronological FP32. */
int denise_cuda_m9_test_reverse(struct denise_cuda_m9 *,int mode,int kind,int field,
 double *padded,double *psi,double *q,double *images,const float *background,const float *residual);
int denise_cuda_m9_apply_j(struct denise_cuda_m9 *,const float *delta_lambda,
                          const float *delta_mu,size_t cells);
/* Transactional optional outputs: data [t][r][2], fields [5][y][x], psi [8],
 * final tangent operands [4][y][x], harmonic corner tangent [y][x]. */
int denise_cuda_m9_born_download(struct denise_cuda_m9 *,float *,float *,float *,float *,float *);
int denise_cuda_m9_born_diagnostics(const struct denise_cuda_m9 *,struct denise_cuda_m9_born_diagnostics *);
int denise_cuda_m9_prepare(struct denise_cuda_m9 *);
int denise_cuda_m9_nonlinear(struct denise_cuda_m9 *,const float *,const float *);
/* Data [t][receiver][vx,vy], fields [5][y][x], psi [8][y][x], operands [t][4][y][x]. */
int denise_cuda_m9_download(struct denise_cuda_m9 *,float *data,float *fields,float *psi,float *operands);
int denise_cuda_m9_diagnostics(const struct denise_cuda_m9 *,struct denise_cuda_m9_diagnostics *);
int denise_cuda_m9_destroy(struct denise_cuda_m9 **);
/* Candidate-owned fault/ledger diagnostics. at=0 disables injection; at>0 fails
 * one indexed operation before execution. Cleanup is never fault-injected. */
void denise_cuda_m9_fault(size_t at);
/* Scoped new-operation fault counter; legacy denise_cuda_m9_fault disables it. */
void denise_cuda_m9_replay_fault(size_t at);
void denise_cuda_m9_ledger(size_t *device_bytes,size_t *host_bytes,size_t *events,size_t *calls);
/* Test-only operations invalidate prepared state. Arbitrary physical/psi state,
 * optional packed profiles, and actual production halos are exposed explicitly. */
int denise_cuda_m9_test_step(struct denise_cuda_m9 *,const float *state13,const float *profiles);
int denise_cuda_m9_test_evolve(struct denise_cuda_m9 *,const float *state13);
/* Pure local production blocks; mode 0 projection/mirrors, 1 velocity ghosts,
 * 2 A closure. Physical/halo fields [5][ny+4][nx+4], compact q/bg [4][ny][nx]. */
int denise_cuda_m9_test_surface(struct denise_cuda_m9 *,int mode,float *padded,
                               const float *q,const float *bg,const float *dl,const float *dm);
int denise_cuda_m9_test_halo(struct denise_cuda_m9 *,int velocity,float *five_padded);
int denise_cuda_m9_test_static(struct denise_cuda_m9 *,float *maps,float *profiles,float *source,int *geometry);
/* Pure copy layout gate: overwrite/download FULL operands, invalidate results. */
int denise_cuda_m9_test_trajectory_roundtrip(struct denise_cuda_m9 *,float *operands);
#ifdef __cplusplus
}
#endif
#endif
