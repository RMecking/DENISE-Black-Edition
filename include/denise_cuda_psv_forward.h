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
