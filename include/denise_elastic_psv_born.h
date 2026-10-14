#ifndef DENISE_ELASTIC_PSV_BORN_H
#define DENISE_ELASTIC_PSV_BORN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * M9b-1 elastic P/SV Born operator.
 *
 * Every model and image is a contiguous row-major [ny][nx] array. Receiver
 * coordinates are zero-based cell indices. Data is time-major with offset
 * ((time * receiver_count + receiver) * 2 + component), where component zero
 * is vx and component one is vy. The single-rank halo semantics are periodic,
 * matching DENISE's one-rank exchange. No DT or DH weighting is part of either
 * Euclidean inner product.
 */

enum denise_elastic_psv_born_strain_component {
    DENISE_ELASTIC_PSV_BORN_VXX = 0,
    DENISE_ELASTIC_PSV_BORN_VYX = 1,
    DENISE_ELASTIC_PSV_BORN_VXY = 2,
    DENISE_ELASTIC_PSV_BORN_VYY = 3
};

struct denise_elastic_psv_born_config {
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

    const float *lambda;
    const float *mu;
    const float *rho;

    int source_i;
    int source_j;
    const float *source_samples;

    int receiver_count;
    const int *receiver_i;
    const int *receiver_j;

    int cpml_enabled;
    float pml_reflection;
    float pml_power;
    float pml_kmax;
    float pml_fpml;
    float pml_damping_speed;
};

struct denise_elastic_psv_born;

struct denise_elastic_psv_born_storage_diagnostics {
    int segmented;
    int segment_count;
    int checkpoint_count;
    int max_segment_length;
    size_t checkpoint_payload_bytes;
    size_t checkpoint_bytes; /* Sum of payloads only; metadata below. */
    size_t segment_operand_bytes;
    size_t retained_replay_bytes;
    size_t forward_working_bytes;
    size_t adjoint_working_bytes;
    size_t initial_forward_steps;
    size_t replayed_forward_steps_last;
    size_t checkpoint_metadata_bytes;
    size_t checkpoint_pointer_bytes;
    size_t segment_schedule_bytes;
};

const char *denise_elastic_psv_born_last_error(void);

int denise_elastic_psv_born_create(
    const struct denise_elastic_psv_born_config *config,
    struct denise_elastic_psv_born **context);

/* Select deterministic segmented checkpoint/replay before prepare(). */
int denise_elastic_psv_born_set_replay_segments(
    struct denise_elastic_psv_born *context,
    int segment_count);

/* Complete logical requested retained bytes, including objects/schedules. */
int denise_elastic_psv_born_estimate_replay_storage(
    const struct denise_elastic_psv_born *context,
    int segment_count,
    size_t *retained_bytes);

int denise_elastic_psv_born_get_segment_bounds(
    const struct denise_elastic_psv_born *context,
    int segment,
    int *start,
    int *end_exclusive);

int denise_elastic_psv_born_storage_diagnostics(
    const struct denise_elastic_psv_born *context,
    struct denise_elastic_psv_born_storage_diagnostics *diagnostics);

/* Test/diagnostic proof of post-timestep checkpoint sufficiency. */
int denise_elastic_psv_born_checkpoint_roundtrip(
    struct denise_elastic_psv_born *context,
    int timestep);

/* Run and retain the fixed-background trajectory and four post-CPML strains. */
int denise_elastic_psv_born_prepare(
    struct denise_elastic_psv_born *context,
    float *background_data);

/* Apply J to cell-centred delta-lambda/delta-mu. */
int denise_elastic_psv_born_apply_j(
    struct denise_elastic_psv_born *context,
    const float *delta_lambda,
    const float *delta_mu,
    float *born_data);

/* Apply the exact reverse-coded discrete transpose to prepared vx/vy data. */
int denise_elastic_psv_born_apply_jt(
    struct denise_elastic_psv_born *context,
    const float *migration_data,
    double *image_lambda_raw,
    double *image_mu_raw);

/* Production nonlinear map used by the dense finite-difference gate. */
int denise_elastic_psv_born_nonlinear(
    struct denise_elastic_psv_born *context,
    const float *lambda,
    const float *mu,
    float *receiver_data);

/* Copy one retained operand slot as [VXX,VYX,VXY,VYY][ny][nx]. */
int denise_elastic_psv_born_copy_strain(
    const struct denise_elastic_psv_born *context,
    int timestep,
    float *four_strains);

int denise_elastic_psv_born_is_prepared(
    const struct denise_elastic_psv_born *context);

float denise_elastic_psv_born_cpml_memory_peak(
    const struct denise_elastic_psv_born *context);

void denise_elastic_psv_born_destroy(
    struct denise_elastic_psv_born **context);


/* Additive CPU, one-rank homogeneous-solid Taylor8 Forward only.
 * Native ABI v1: exact sizeof/version/identity checks; arrays have the same
 * row-major and direct time-major vx/vy layout as above. boundary=0 in base
 * is neutral here: the explicit policy below selects zero FIELD extension.
 * The old Born API remains periodic FD4. No casts between opaque handles.
 * Create copies model/source/receiver inputs; nonlinear holds copied rho fixed.
 * Calls are externally serialized. Readable lengths and nonoverlapping writable
 * outputs are caller obligations. Publish K=NT*R*2 floats only on full success;
 * capacity is in floats, the tail is untouched and errors preserve outputs.
 */
enum denise_elastic_psv_fd8_family_v1 {
    DENISE_ELASTIC_PSV_FD8_TAYLOR8_V1 = 1
};
enum denise_elastic_psv_fd8_field_boundary_v1 {
    DENISE_ELASTIC_PSV_FD8_ZERO_FIELD_V1 = 1
};
enum denise_elastic_psv_fd8_material_boundary_v1 {
    DENISE_ELASTIC_PSV_FD8_REPLICATED_MATERIAL_V1 = 1
};
struct denise_elastic_psv_forward_fd8_config_v1 {
    size_t struct_size;
    unsigned int abi_version;
    struct denise_elastic_psv_born_config base;
    int coefficient_family;
    int field_boundary_policy;
    int material_boundary_policy;
};
struct denise_elastic_psv_forward_fd8;
int denise_elastic_psv_forward_fd8_create_v1(
    const struct denise_elastic_psv_forward_fd8_config_v1 *config,
    struct denise_elastic_psv_forward_fd8 **context);
int denise_elastic_psv_forward_fd8_nonlinear_v1(
    struct denise_elastic_psv_forward_fd8 *context,
    const float *lambda, const float *mu,
    float *receiver_data, size_t receiver_capacity_floats);
const char *denise_elastic_psv_forward_fd8_last_error_v1(void);
void denise_elastic_psv_forward_fd8_destroy_v1(
    struct denise_elastic_psv_forward_fd8 **context);

#ifdef __cplusplus
}
#endif

#endif
