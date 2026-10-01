#include "denise_elastic_psv_migration.h"
#include "denise_elastic_psv_born.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char migration_error[512];

static int migration_fail(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(migration_error, sizeof(migration_error), format, arguments);
    va_end(arguments);
    return -1;
}

const char *denise_elastic_psv_migration_last_error(void) {
    return migration_error[0] ? migration_error : "no elastic P/SV migration error";
}

static int checked_product(size_t left, size_t right, size_t *product) {
    if (left != 0 && right > SIZE_MAX / left) return -1;
    *product = left * right;
    return 0;
}

static int full_trajectory_size(size_t cells, size_t nt, size_t *bytes) {
    size_t values;
    if (!bytes) return -1;
    *bytes = 0;
    if (!cells || !nt || checked_product(cells, 4u, &values) != 0
        || checked_product(values, nt, &values) != 0
        || checked_product(values, sizeof(float), bytes) != 0) return -1;
    return 0;
}

static int select_replay_backend(int estimate_status, size_t replay_bytes,
        size_t full_bytes, int *selected) {
    if (!selected) return migration_fail("M9c backend selection result is null");
    *selected = 0;
    if (estimate_status != 0 || !replay_bytes || !full_bytes)
        return migration_fail("M9c backend selection requires valid storage estimates");
    *selected = replay_bytes < full_bytes;
    return 0;
}

static int finite_array(const float *values, size_t count) {
    size_t index;
    if (!values) return 0;
    for (index = 0; index < count; ++index)
        if (!isfinite(values[index])) return 0;
    return 1;
}

int denise_elastic_psv_migration_pack_components(
        const float *vx, const float *vy, int nt, int receiver_count,
        float *packed) {
    size_t samples;
    int time, receiver;
    migration_error[0] = '\0';
    if (!vx || !vy || !packed)
        return migration_fail("M9c migration-data adapter received a null pointer");
    if (nt < 1 || receiver_count < 1
        || checked_product((size_t)nt, (size_t)receiver_count, &samples) != 0)
        return migration_fail("M9c migration-data dimensions overflow size_t");
    if (!finite_array(vx, samples))
        return migration_fail("M9c vx migration data contains NaN or Inf");
    if (!finite_array(vy, samples))
        return migration_fail("M9c vy migration data contains NaN or Inf");
    for (time = 0; time < nt; ++time)
        for (receiver = 0; receiver < receiver_count; ++receiver) {
            size_t component = (size_t)time * (size_t)receiver_count + (size_t)receiver;
            size_t target = component * 2u;
            packed[target] = vx[component];
            packed[target + 1u] = vy[component];
        }
    return 0;
}

static int validate_request(
        const struct denise_elastic_psv_migration_request *request,
        size_t *cells, size_t *trajectory_bytes, size_t *image_bytes,
        size_t *maximum_data_bytes) {
    size_t values, bytes;
    int shot_index, previous_shot = 0;
    if (!request) return migration_fail("M9c migration request is null");
    if (request->l != 0) return migration_fail("M9c requires L=0 (got %d)", request->l);
    if (request->invmat1 != 3)
        return migration_fail("M9c requires INVMAT1=3 lambda/mu/rho semantics");
    if (request->fdorder != 4)
        return migration_fail("M9c requires FDORDER=4 (got %d)", request->fdorder);
    if (request->ndt != 1 || request->dtinv != 1)
        return migration_fail("M9c requires NDT=DTINV=1");
    if (request->free_surface != 0 && request->free_surface != 1)
        return migration_fail("M9 migration FREE_SURF must be 0 or 1");
    if (request->boundary != 0)
        return migration_fail("M9c does not support BOUNDARY");
    if (request->mpi_size != 1)
        return migration_fail("M9c requires one MPI rank");
    if (request->receiver_components != 2)
        return migration_fail("M9c requires direct vx/vy receiver components");
    if (request->inv_stf != 0)
        return migration_fail("M9c requires INV_STF=0 and a prepared source signal");
    if (request->nx < 5 || request->ny < 5 || request->nt < 1
        || !(request->dh > 0.0f) || !(request->dt > 0.0f))
        return migration_fail("M9c grid and time dimensions are invalid");
    if (request->shot_count < 1 || !request->shots)
        return migration_fail("M9c requires at least one physical shot");
    if (!request->lambda || !request->mu || !request->rho)
        return migration_fail("M9c model pointer is null");
    if (checked_product((size_t)request->nx, (size_t)request->ny, cells) != 0
        || *cells > (size_t)INT32_MAX)
        return migration_fail("M9c NX*NY overflows the supported flat index range");
    if (full_trajectory_size(*cells, (size_t)request->nt,
                             trajectory_bytes) != 0)
        return migration_fail("M9c 4*NT*NX*NY trajectory size overflows size_t");
    if (checked_product(*cells, 2u, &values) != 0
        || checked_product(values, sizeof(double), image_bytes) != 0)
        return migration_fail("M9c double image sizes overflow size_t");
    *maximum_data_bytes = 0;
    for (shot_index = 0; shot_index < request->shot_count; ++shot_index) {
        const struct denise_elastic_psv_migration_shot *shot = &request->shots[shot_index];
        int receiver;
        if (shot->physical_shot_index <= previous_shot)
            return migration_fail("M9c shots must be in strictly ascending physical-shot order");
        previous_shot = shot->physical_shot_index;
        if (shot->source_type != 1)
            return migration_fail("M9c shot %d has unsupported source type %d; explosive type 1 is required",
                                  shot->physical_shot_index, shot->source_type);
        if (shot->source_i < 0 || shot->source_i >= request->nx
            || shot->source_j < 0 || shot->source_j >= request->ny)
            return migration_fail("M9c shot %d source is outside the grid",
                                  shot->physical_shot_index);
        if (request->free_surface && shot->source_j == 0)
            return migration_fail("M9 free surface rejects explosive source at j=1");
        if (shot->receiver_count < 1 || !shot->receiver_i || !shot->receiver_j)
            return migration_fail("M9c shot %d receiver geometry is invalid",
                                  shot->physical_shot_index);
        if (!finite_array(shot->source_samples, (size_t)request->nt))
            return migration_fail("M9c shot %d prepared source contains NaN or Inf",
                                  shot->physical_shot_index);
        if (checked_product((size_t)request->nt, (size_t)shot->receiver_count, &values) != 0
            || checked_product(values, 2u, &values) != 0)
            return migration_fail("M9c shot %d NT*NREC*2 data size overflows size_t",
                                  shot->physical_shot_index);
        if (!finite_array(shot->migration_data, values))
            return migration_fail("M9c shot %d migration data contains NaN or Inf",
                                  shot->physical_shot_index);
        if (checked_product(values, sizeof(float), &bytes) != 0)
            return migration_fail("M9c shot %d migration-data byte size overflows size_t",
                                  shot->physical_shot_index);
        if (bytes > *maximum_data_bytes) *maximum_data_bytes = bytes;
        for (receiver = 0; receiver < shot->receiver_count; ++receiver)
            if (shot->receiver_i[receiver] < 0 || shot->receiver_i[receiver] >= request->nx
                || shot->receiver_j[receiver] < 0 || shot->receiver_j[receiver] >= request->ny)
                return migration_fail("M9c shot %d receiver %d is outside the grid",
                                      shot->physical_shot_index, receiver);
    }
    return 0;
}

void denise_elastic_psv_migration_result_destroy(
        struct denise_elastic_psv_migration_result *result) {
    if (!result) return;
    free(result->image_lambda_raw);
    free(result->image_mu_raw);
    memset(result, 0, sizeof(*result));
}

int denise_elastic_psv_migrate(
        const struct denise_elastic_psv_migration_request *request,
        struct denise_elastic_psv_migration_result *result) {
    struct denise_elastic_psv_born *context = NULL;
    double *lambda_sum = NULL, *mu_sum = NULL, *shot_lambda = NULL, *shot_mu = NULL;
    size_t cells = 0, trajectory_bytes = 0, image_bytes = 0, maximum_data_bytes = 0;
    int shot_index;
    int production_segments;
    float cpml_peak = 0.0f;
    migration_error[0] = '\0';
    if (!result) return migration_fail("M9c migration result is null");
    memset(result, 0, sizeof(*result));
    if (validate_request(request, &cells, &trajectory_bytes, &image_bytes,
                         &maximum_data_bytes) != 0)
        return -1;
    production_segments = request->nt < 32 ? request->nt : 32;
    lambda_sum = calloc(cells, sizeof(double));
    mu_sum = calloc(cells, sizeof(double));
    shot_lambda = malloc(cells * sizeof(double));
    shot_mu = malloc(cells * sizeof(double));
    if (!lambda_sum || !mu_sum || !shot_lambda || !shot_mu) {
        migration_fail("M9c could not allocate double image accumulators");
        goto failure;
    }
    for (shot_index = 0; shot_index < request->shot_count; ++shot_index) {
        const struct denise_elastic_psv_migration_shot *shot = &request->shots[shot_index];
        struct denise_elastic_psv_born_config config;
        struct denise_elastic_psv_born_storage_diagnostics storage;
        size_t replay_storage_estimate;
        int estimate_status, replay_selected;
        float shot_peak;
        size_t cell;
        memset(&config, 0, sizeof(config));
        config.nx=request->nx; config.ny=request->ny; config.nt=request->nt;
        config.fw=request->fw; config.dh=request->dh; config.dt=request->dt;
        config.l=request->l; config.invmat1=request->invmat1;
        config.fdorder=request->fdorder; config.ndt=request->ndt;
        config.dtinv=request->dtinv; config.free_surface=request->free_surface;
        config.boundary=request->boundary; config.mpi_size=request->mpi_size;
        config.receiver_components=request->receiver_components;
        config.lambda=request->lambda; config.mu=request->mu; config.rho=request->rho;
        config.source_i=shot->source_i; config.source_j=shot->source_j;
        config.source_samples=shot->source_samples;
        config.receiver_count=shot->receiver_count;
        config.receiver_i=shot->receiver_i; config.receiver_j=shot->receiver_j;
        config.cpml_enabled=request->cpml_enabled;
        config.pml_reflection=request->pml_reflection;
        config.pml_power=request->pml_power; config.pml_kmax=request->pml_kmax;
        config.pml_fpml=request->pml_fpml;
        config.pml_damping_speed=request->pml_damping_speed;
        if (denise_elastic_psv_born_create(&config, &context) != 0) {
            migration_fail("M9c shot %d operator creation failed: %s",
                           shot->physical_shot_index,
                           denise_elastic_psv_born_last_error());
            goto failure;
        }
        estimate_status = denise_elastic_psv_born_estimate_replay_storage(
                context, production_segments, &replay_storage_estimate);
        if (select_replay_backend(estimate_status, replay_storage_estimate,
                                  trajectory_bytes, &replay_selected) != 0) {
            migration_fail("M9c shot %d replay estimate failed: %s",
                           shot->physical_shot_index,
                           denise_elastic_psv_born_last_error());
            goto failure;
        }
        if (replay_selected
            && denise_elastic_psv_born_set_replay_segments(
                   context, production_segments) != 0) {
            migration_fail("M9c shot %d replay policy failed: %s",
                           shot->physical_shot_index,
                           denise_elastic_psv_born_last_error());
            goto failure;
        }
        if (denise_elastic_psv_born_prepare(context, NULL) != 0) {
            migration_fail("M9c shot %d background preparation failed: %s",
                           shot->physical_shot_index,
                           denise_elastic_psv_born_last_error());
            goto failure;
        }
        shot_peak = denise_elastic_psv_born_cpml_memory_peak(context);
        if (shot_peak > cpml_peak) cpml_peak = shot_peak;
        if (denise_elastic_psv_born_apply_jt(
                context, shot->migration_data, shot_lambda, shot_mu) != 0) {
            migration_fail("M9c shot %d adjoint application failed: %s",
                           shot->physical_shot_index,
                           denise_elastic_psv_born_last_error());
            goto failure;
        }
        if (denise_elastic_psv_born_storage_diagnostics(context, &storage) != 0) {
            migration_fail("M9c shot %d storage diagnostics failed: %s",
                           shot->physical_shot_index,
                           denise_elastic_psv_born_last_error());
            goto failure;
        }
        result->checkpoint_payload_bytes = storage.checkpoint_payload_bytes;
        result->checkpoint_bytes = storage.checkpoint_bytes;
        result->checkpoint_metadata_bytes = storage.checkpoint_metadata_bytes;
        result->checkpoint_pointer_bytes = storage.checkpoint_pointer_bytes;
        result->segment_schedule_bytes = storage.segment_schedule_bytes;
        result->segment_operand_bytes = storage.segment_operand_bytes;
        result->peak_replay_storage_bytes = storage.retained_replay_bytes;
        result->forward_working_bytes = storage.forward_working_bytes;
        result->adjoint_working_bytes = storage.adjoint_working_bytes;
        result->initial_forward_steps = storage.initial_forward_steps;
        result->replayed_steps = storage.replayed_forward_steps_last;
        result->segment_count = storage.segment_count;
        result->checkpoint_count = storage.checkpoint_count;
        result->max_segment_length = storage.max_segment_length;
        for (cell = 0; cell < cells; ++cell) {
            lambda_sum[cell] += shot_lambda[cell];
            mu_sum[cell] += shot_mu[cell];
        }
        denise_elastic_psv_born_destroy(&context);
    }
    free(shot_lambda);
    free(shot_mu);
    result->image_lambda_raw = lambda_sum;
    result->image_mu_raw = mu_sum;
    result->cell_count = cells;
    result->shots_completed = request->shot_count;
    result->cpml_memory_peak = cpml_peak;
    result->trajectory_bytes = trajectory_bytes;
    result->global_image_bytes = image_bytes;
    result->maximum_shot_data_bytes = maximum_data_bytes;
    return 0;

failure:
    denise_elastic_psv_born_destroy(&context);
    free(lambda_sum); free(mu_sum); free(shot_lambda); free(shot_mu);
    memset(result, 0, sizeof(*result));
    return -1;
}
