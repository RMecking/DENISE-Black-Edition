#ifndef DENISE_CUDA_PSV_FORWARD_H
#define DENISE_CUDA_PSV_FORWARD_H

#include "denise_cuda_psv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Frozen M8e-1B2A solver-level envelope. CUDA runtime types stay private. */
struct denise_cuda_psv_forward;

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
