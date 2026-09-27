#ifndef DENISE_CUDA_PSV_FORWARD_H
#define DENISE_CUDA_PSV_FORWARD_H

#include "denise_cuda_psv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Frozen M8e-1B2A solver-level envelope. CUDA runtime types stay private. */
struct denise_cuda_psv_forward;

/* M8c partitions NT into min(requested_segments,NT) nonempty segments. */

struct denise_cuda_psv_forward_config {
    struct denise_cuda_psv_fd4_l1_config core;
    int nt;
    int ntr;
    int global_mode;
    int source_count;
    int source_type;
    int seismo;
    int ndt;
    int snapshots;
    int inv_stf;
};

/* CPU adapters retain DENISE's one-based matrix conventions. */
struct denise_cuda_psv_forward_host {
    struct denise_cuda_psv_fd4_l1_host core;
    float **source_positions;
    float **source_signals;
    int **receiver_positions;
    float **sectionvx;
    float **sectionvy;
};

struct denise_cuda_psv_forward_stats {
    size_t b1_core_bytes;
    size_t source_signal_bytes;
    size_t source_geometry_bytes;
    size_t receiver_geometry_bytes;
    size_t trace_bytes;
    size_t workspace_bytes;
    size_t total_mandatory_bytes;
    size_t usable_budget_bytes;
    size_t remaining_budget_bytes;
    size_t checkpoint_bytes; /* Required C = 4 [8F + 4X + 4Y]; reserve is opt-in. */
    size_t checkpoint_d2d_calls;
    size_t checkpoint_d2d_bytes;
    size_t checkpoint_capture_calls;
    size_t checkpoint_restore_calls;
    size_t segment_bank_bytes; /* Includes the initial-state seed plus S-1 boundaries. */
    size_t segment_operand_bytes; /* 6 * max_segment_length * NX * NY * sizeof(float). */
    size_t segment_d2d_calls;
    int segment_count;
    int max_segment_length;
    size_t h2d_transfer_calls;
    size_t d2h_transfer_calls;
    size_t h2d_bytes;
    size_t d2h_bytes;
    size_t full_grid_h2d_per_timestep;
    size_t full_grid_d2h_per_timestep;
    size_t source_sample_h2d_per_timestep;
    size_t receiver_sample_d2h_per_timestep;
    unsigned long long timesteps;
    size_t forward_synchronization_calls;
    size_t resident_event_records;
    size_t resident_elapsed_queries;
    size_t profile_event_records;
    size_t profile_elapsed_queries;
    int profiling_enabled;
    float velocity_kernel_ms;
    float stress_kernel_ms;
    float source_kernel_ms;
    float receiver_kernel_ms;
    float resident_timestep_ms;
    float context_setup_ms;
    float initial_upload_ms;
    float trace_download_ms;
    float mutable_download_ms;
    float total_forward_ms;
};

/* FP64 exact-visco P/SV adjoint state. Main fields use the full halo layout
 * (NX+6)*(NY+6). X CPML fields use NY*(2*FW); Y fields use (2*FW)*NX. */
struct denise_cuda_psv_adjoint_host {
    double *avx, *avy, *asxx, *asyy, *asxy, *ar, *ap, *aq;
    double *psxx, *psxyx, *pvxx, *pvyx;
    double *psxyy, *psyy, *pvxy, *pvyy;
};

struct denise_cuda_psv_adjoint_stats {
    size_t main_state_bytes;
    size_t cpml_state_bytes;
    size_t workspace_bytes;
    size_t residual_bytes;
    size_t total_bytes;
    size_t remaining_budget_bytes;
    size_t initial_h2d_calls;
    size_t diagnostic_d2h_calls;
    /* Synchronous small-vector H2D copies performed before kernel launch. */
    size_t residual_h2d_calls;
    size_t residual_h2d_bytes;
    size_t step_allocation_calls;
    size_t step_full_state_h2d_calls;
    size_t step_full_state_d2h_calls;
    /* Explicit device/kernel synchronization calls, excluding H2D copies. */
    size_t step_blocking_sync_calls;
    unsigned long long steps;
    size_t kernel_launches;
};

/* M8e-2C2 persistent segmented reverse-sweep accounting. Observed traces use
 * the forward trace layout: receiver-major [receiver][timestep-1]. */
struct denise_cuda_psv_adjoint_sweep_stats {
    size_t observed_trace_bytes;
    size_t total_incremental_bytes;
    size_t remaining_budget_bytes;
    size_t observed_h2d_calls;
    size_t observed_h2d_bytes;
    size_t reverse_segments;
    unsigned long long reverse_timesteps;
    size_t replay_synchronization_calls;
    size_t reverse_segment_synchronization_calls;
    size_t per_step_residual_h2d_calls;
    size_t per_step_full_state_h2d_calls;
    size_t per_step_full_state_d2h_calls;
    size_t per_step_allocation_calls;
    size_t per_step_free_calls;
    size_t per_step_blocking_sync_calls;
    int next_reverse_segment;
    int prepared;
    int complete;
    int invalid;
};

/* M8e-2C3 native exact-visco gradients. Each host pointer addresses exactly
 * NX*NY doubles. The frozen field order is GF,GG,GFC,GD,GE,GDC,GRX,GRY and
 * compact interior element (i,j), with one-based solver indices, is stored at
 * (j-1)*NX+(i-1). */
struct denise_cuda_psv_native_gradient_host {
    double *gf, *gg, *gfc, *gd, *ge, *gdc, *grx, *gry;
};

struct denise_cuda_psv_native_gradient_stats {
    size_t native_gradient_bytes;
    size_t remaining_budget_bytes;
    size_t allocation_calls;
    size_t zero_calls;
    size_t diagnostic_d2h_calls;
    size_t diagnostic_d2h_bytes;
    size_t per_step_h2d_calls;
    size_t per_step_d2h_calls;
    size_t per_step_allocation_calls;
    size_t per_step_free_calls;
    size_t per_step_blocking_sync_calls;
    unsigned long long accumulated_timesteps;
    size_t correlation_kernel_launches;
    int prepared;
};

/* M8e-2C4 authoritative material inputs for the exact native-to-physical
 * mapping. Each matrix keeps DENISE's one-based [j][i] convention. rip/rjp
 * remain the already-resident authoritative face-density inputs. The Q
 * coefficients must be copied from the CPU q_tau_mapping produced by
 * init_q_tau_mapping(); CUDA does not duplicate the fitting procedure. */
struct denise_cuda_psv_physical_material_host {
    float **prho, **ppi, **pu, **ptaus, **ptaup;
    float **puipjp, **ptausipjp;
    float eta;
    int q_parameterization_mode;
    double inverse_tau_per_q;
    double inverse_tau_offset;
};

/* Compact field order Vp,Vs,rho,Qp,Qs. Each pointer addresses exactly
 * NX*NY doubles with (i,j) at (j-1)*NX+(i-1). */
struct denise_cuda_psv_physical_gradient_host {
    double *vp, *vs, *rho, *qp, *qs;
};

struct denise_cuda_psv_physical_gradient_stats {
    size_t material_bytes;
    size_t physical_gradient_bytes;
    size_t alignment_bytes;
    size_t validation_bytes;
    size_t total_bytes;
    size_t remaining_budget_bytes;
    size_t allocation_calls;
    size_t material_h2d_calls;
    size_t material_h2d_bytes;
    size_t zero_calls;
    size_t map_kernel_launches;
    size_t map_synchronization_calls;
    size_t validation_d2h_calls;
    size_t validation_d2h_bytes;
    size_t diagnostic_d2h_calls;
    size_t diagnostic_d2h_bytes;
    size_t per_step_h2d_calls;
    size_t per_step_d2h_calls;
    size_t per_step_allocation_calls;
    size_t per_step_free_calls;
    size_t per_step_blocking_sync_calls;
    int prepared;
    int mapped;
    int invalid;
};

const char *denise_cuda_psv_forward_last_error(void);

int denise_cuda_psv_forward_required_bytes(
        const struct denise_cuda_psv_forward_config *config,
        struct denise_cuda_psv_forward_stats *plan);

int denise_cuda_psv_forward_create(
        const struct denise_cuda_psv_forward_config *config,
        const struct denise_cuda_psv_forward_host *host,
        struct denise_cuda_psv_forward **context);

int denise_cuda_psv_forward_run(
        struct denise_cuda_psv_forward *context);

/* Inclusive, absolute, one-based timesteps [begin,end]. The next range must
 * begin immediately after the last completed timestep. Source and receiver
 * indices remain absolute. The same context survives all ranges. */
int denise_cuda_psv_forward_run_range(
        struct denise_cuda_psv_forward *context,int begin,int end);

/* One device-resident checkpoint per context. Reserve before the first range
 * for a fail-closed budget gate. Capture only at a completed range boundary.
 * Restore sets the next executable timestep to completed_timestep+1. The
 * checkpoint is bound to this persistent context and freed with it. */
/* Receiver samples after the restored boundary are stale until the resumed
 * ranges overwrite them; no trace data is part of the checkpoint payload. */
int denise_cuda_psv_forward_checkpoint_reserve(
        struct denise_cuda_psv_forward *context);
int denise_cuda_psv_forward_checkpoint_capture(
        struct denise_cuda_psv_forward *context,int completed_timestep);
int denise_cuda_psv_forward_checkpoint_restore(
        struct denise_cuda_psv_forward *context,int *next_timestep);

/* Optional M8c-compatible segmentation. Call prepare before any propagation:
 * requested_segments=0 selects the M8c default of 32; otherwise it must be
 * positive. S=min(requested_segments,NT), start[k]=floor(k*NT/S),
 * end[k]=floor((k+1)*NT/S), and segment k runs [start[k]+1,end[k]].
 * S-1 interior boundary states and one initial-state seed remain on device.
 * Run each segment in order with run_range, then capture each interior end.
 * Recording is opt-in for an uninterrupted diagnostic reference; replay
 * records the same six operands automatically into a bounded device buffer.
 * Download is diagnostic and legal only after a complete recorded segment.
 * Operand layout is field-major [FX,FY,VXX,VYX,VXY,VYY], then time, y, x. */
int denise_cuda_psv_forward_segments_prepare(
        struct denise_cuda_psv_forward *context,int requested_segments);
int denise_cuda_psv_forward_segment_bounds(
        const struct denise_cuda_psv_forward *context,int segment,
        int *begin,int *end);
int denise_cuda_psv_forward_segment_capture(
        struct denise_cuda_psv_forward *context,int segment);
int denise_cuda_psv_forward_segment_record_next(
        struct denise_cuda_psv_forward *context,int segment);
int denise_cuda_psv_forward_segment_replay(
        struct denise_cuda_psv_forward *context,int segment);
int denise_cuda_psv_forward_segment_download_operands(
        struct denise_cuda_psv_forward *context,int segment,
        float *host_fields,size_t float_capacity);

int denise_cuda_psv_adjoint_required_bytes(
        const struct denise_cuda_psv_forward_config *config,
        struct denise_cuda_psv_adjoint_stats *plan);
int denise_cuda_psv_adjoint_prepare(
        struct denise_cuda_psv_forward *context,
        const struct denise_cuda_psv_adjoint_host *initial);
/* timestep is the absolute production sample. Sample 1 injects no residual.
 * Later samples synchronously copy four NREC float vectors before launching
 * the exact resident reverse state operator, so caller ownership may end when
 * this function returns. Calls evolve the current device state. */
int denise_cuda_psv_adjoint_step(
        struct denise_cuda_psv_forward *context,int timestep,
        const float *modeled_vx,const float *observed_vx,
        const float *modeled_vy,const float *observed_vy);
int denise_cuda_psv_adjoint_download(
        struct denise_cuda_psv_forward *context,
        struct denise_cuda_psv_adjoint_host *host);
int denise_cuda_psv_adjoint_get_stats(
        const struct denise_cuda_psv_forward *context,
        struct denise_cuda_psv_adjoint_stats *stats);

/* Plan and run the M8e-2C2 state-only segmented reverse sweep. The prepare
 * call requires a completed original segmented forward trajectory, a complete
 * checkpoint bank, and a prepared adjoint state. observed_vx/observed_vy each
 * contain NTR*NT floats in receiver-major [receiver][timestep-1] order and are
 * uploaded exactly once. Reverse segment calls must follow S-1,...,0. Replay
 * leaves the six bounded operands resident as
 * [field][timestep-begin(segment)][y][x]; 2C2 does not consume them yet. */
int denise_cuda_psv_adjoint_sweep_required_bytes(
        const struct denise_cuda_psv_forward_config *config,
        struct denise_cuda_psv_adjoint_sweep_stats *plan);
int denise_cuda_psv_adjoint_sweep_prepare(
        struct denise_cuda_psv_forward *context,
        const float *observed_vx,const float *observed_vy,
        size_t float_count_per_component);
int denise_cuda_psv_adjoint_reverse_segment(
        struct denise_cuda_psv_forward *context,int segment);
int denise_cuda_psv_adjoint_reverse_sweep(
        struct denise_cuda_psv_forward *context);
int denise_cuda_psv_adjoint_sweep_get_stats(
        const struct denise_cuda_psv_forward *context,
        struct denise_cuda_psv_adjoint_sweep_stats *stats);

/* Explicitly allocate, zero, and arm the eight compact FP64 native fields.
 * Preparation is legal exactly once after sweep preparation and before the
 * first reverse segment. Download is diagnostic and never occurs implicitly. */
int denise_cuda_psv_native_gradient_required_bytes(
        const struct denise_cuda_psv_forward_config *config,
        struct denise_cuda_psv_native_gradient_stats *plan);
int denise_cuda_psv_native_gradient_prepare(
        struct denise_cuda_psv_forward *context);
int denise_cuda_psv_native_gradient_download(
        struct denise_cuda_psv_forward *context,
        struct denise_cuda_psv_native_gradient_host *host,
        size_t elements_per_field);
int denise_cuda_psv_native_gradient_get_stats(
        const struct denise_cuda_psv_forward *context,
        struct denise_cuda_psv_native_gradient_stats *stats);

/* Allocate/upload the persistent physical mapping state before the first
 * reverse segment. Mapping is legal once, only after the reverse sweep has
 * completed, and performs no host-side reconstruction. Downloads are
 * diagnostic and never implicit. */
int denise_cuda_psv_physical_gradient_required_bytes(
        const struct denise_cuda_psv_forward_config *config,
        struct denise_cuda_psv_physical_gradient_stats *plan);
int denise_cuda_psv_physical_gradient_prepare(
        struct denise_cuda_psv_forward *context,
        const struct denise_cuda_psv_physical_material_host *material);
int denise_cuda_psv_physical_gradient_map(
        struct denise_cuda_psv_forward *context);
int denise_cuda_psv_physical_gradient_download(
        struct denise_cuda_psv_forward *context,
        struct denise_cuda_psv_physical_gradient_host *host,
        size_t elements_per_field);
int denise_cuda_psv_physical_gradient_get_stats(
        const struct denise_cuda_psv_forward *context,
        struct denise_cuda_psv_physical_gradient_stats *stats);

int denise_cuda_psv_forward_download_traces(
        struct denise_cuda_psv_forward *context,
        float **sectionvx,
        float **sectionvy);

int denise_cuda_psv_forward_download_mutable(
        struct denise_cuda_psv_forward *context,
        struct denise_cuda_psv_fd4_l1_host *host);

int denise_cuda_psv_forward_get_stats(
        const struct denise_cuda_psv_forward *context,
        struct denise_cuda_psv_forward_stats *stats);

int denise_cuda_psv_forward_destroy(
        struct denise_cuda_psv_forward **context);

#ifdef __cplusplus
}
#endif

#endif
