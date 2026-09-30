#ifndef DENISE_ELASTIC_PSV_MIGRATION_H
#define DENISE_ELASTIC_PSV_MIGRATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Clean M9c elastic P/SV migration contract.
 *
 * Models and images are contiguous row-major [ny][nx]. Receiver coordinates
 * are zero-based local cell indices. Prepared migration data is chronological
 * and time-major at ((time * receiver_count + receiver) * 2 + component), with
 * component zero vx and component one vy. No residual construction, scaling,
 * filtering, image conditioning, DT, or DH weighting occurs here.
 */
struct denise_elastic_psv_migration_shot {
    int physical_shot_index;
    int source_type;
    int source_i;
    int source_j;
    const float *source_samples;
    int receiver_count;
    const int *receiver_i;
    const int *receiver_j;
    const float *migration_data;
};

struct denise_elastic_psv_migration_request {
    int nx;
    int ny;
    int nt;
    int fw;
    float dh;
    float dt;

    int l;
    int invmat1;
    int fdorder;
    int ndt;
    int dtinv;
    int free_surface;
    int boundary;
    int mpi_size;
    int receiver_components;
    int inv_stf;

    const float *lambda;
    const float *mu;
    const float *rho;

    int cpml_enabled;
    float pml_reflection;
    float pml_power;
    float pml_kmax;
    float pml_fpml;
    float pml_damping_speed;

    int shot_count;
    const struct denise_elastic_psv_migration_shot *shots;
};

struct denise_elastic_psv_migration_result {
    double *image_lambda_raw;
    double *image_mu_raw;
    size_t cell_count;
    int shots_completed;
    float cpml_memory_peak;
    size_t trajectory_bytes;
    size_t global_image_bytes;
    size_t maximum_shot_data_bytes;
    size_t checkpoint_payload_bytes;
    size_t checkpoint_bytes;
    size_t segment_operand_bytes;
    size_t peak_replay_storage_bytes; /* Complete logical retained bytes. */
    size_t forward_working_bytes;
    size_t adjoint_working_bytes;
    size_t initial_forward_steps;
    size_t replayed_steps;
    int segment_count;
    int checkpoint_count;
    int max_segment_length;
    size_t checkpoint_metadata_bytes;
    size_t checkpoint_pointer_bytes;
    size_t segment_schedule_bytes;
};

const char *denise_elastic_psv_migration_last_error(void);

int denise_elastic_psv_migration_pack_components(
    const float *vx,
    const float *vy,
    int nt,
    int receiver_count,
    float *time_major_vx_vy);

int denise_elastic_psv_migrate(
    const struct denise_elastic_psv_migration_request *request,
    struct denise_elastic_psv_migration_result *result);

void denise_elastic_psv_migration_result_destroy(
    struct denise_elastic_psv_migration_result *result);

/* Active single-rank MODE=2 executable entry point. */
int denise_elastic_psv_migration_mode2(void);
const char *denise_elastic_psv_migration_mode2_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
