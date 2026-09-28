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

const char *denise_elastic_psv_born_last_error(void);

int denise_elastic_psv_born_create(
    const struct denise_elastic_psv_born_config *config,
    struct denise_elastic_psv_born **context);

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

#ifdef __cplusplus
}
#endif

#endif
