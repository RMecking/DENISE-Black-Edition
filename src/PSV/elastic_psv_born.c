#include "denise_elastic_psv_born.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum { VX, VY, SXX, SYY, SXY, FIELD_COUNT };
enum { PSXX, PSXYY, PSXYX, PSYY, PVXX, PVYX, PVXY, PVYY, PSI_COUNT };
enum { PROFILE_X, PROFILE_XH, PROFILE_Y, PROFILE_YH, PROFILE_COUNT };
enum { DX_BACK, DX_FWD, DY_BACK, DY_FWD };

struct pml_profile {
    float *kappa;
    float *a;
    float *b;
    int length;
};

enum { ELASTIC_CHECKPOINT_LAYOUT_VERSION = 1 };

struct elastic_checkpoint {
    int layout_version;
    int timestep;
    int nx;
    int ny;
    int fw;
    int fdorder;
    int cpml_enabled;
    size_t value_count;
    float *values;
};

struct denise_elastic_psv_born {
    int nx, ny, nt, fw, receiver_count, source_i, source_j, cpml_enabled;
    int free_surface;
    size_t cells, data_count;
    float dh, dt, coefficient;
    float *lambda, *mu, *rho, *invrho_x, *invrho_y, *mu_corner;
    float *source_samples;
    int *receiver_i, *receiver_j;
    struct pml_profile profile[PROFILE_COUNT];
    float *strain;
    int replay_segments;
    int checkpoint_count;
    int max_segment_length;
    int *segment_start;
    int *segment_end;
    struct elastic_checkpoint *checkpoint;
    float *segment_strain;
    size_t checkpoint_payload_values;
    size_t checkpoint_payload_bytes;
    size_t checkpoint_bytes;
    size_t segment_operand_bytes;
    size_t retained_replay_bytes;
    size_t replayed_forward_steps_last;
    float cpml_memory_peak;
    int prepared;
};

static char born_error[384];

static int fail(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(born_error, sizeof(born_error), format, arguments);
    va_end(arguments);
    return -1;
}

const char *denise_elastic_psv_born_last_error(void) {
    return born_error[0] ? born_error : "no elastic P/SV Born error";
}

static void *checked_calloc(size_t count, size_t width) {
    if (width && count > ((size_t)-1) / width) return NULL;
    return calloc(count, width);
}

static int checked_product(size_t left, size_t right, size_t *product) {
    if (left != 0 && right > SIZE_MAX / left) return -1;
    *product = left * right;
    return 0;
}

static int checked_sum(size_t left, size_t right, size_t *sum) {
    if (right > SIZE_MAX - left) return -1;
    *sum = left + right;
    return 0;
}

static int wrap(int value, int length) {
    if (value < 0) return value + length;
    if (value >= length) return value - length;
    return value;
}

static size_t cell(const struct denise_elastic_psv_born *c, int j, int i) {
    return (size_t)j * (size_t)c->nx + (size_t)i;
}

static void copy_floats(float *destination, const float *source, size_t count) {
    memcpy(destination, source, count * sizeof(*destination));
}

static float derivative_at(const struct denise_elastic_psv_born *c,
                           const float *field, int j, int i, int kind) {
    const float a = 9.0f / 8.0f;
    const float b = -1.0f / 24.0f;
    int im2 = wrap(i - 2, c->nx), im1 = wrap(i - 1, c->nx);
    int ip1 = wrap(i + 1, c->nx), ip2 = wrap(i + 2, c->nx);
    int jm2 = wrap(j - 2, c->ny), jm1 = wrap(j - 1, c->ny);
    int jp1 = wrap(j + 1, c->ny), jp2 = wrap(j + 2, c->ny);
    if (kind == DX_BACK)
        return a * (field[cell(c,j,i)] - field[cell(c,j,im1)])
             + b * (field[cell(c,j,ip1)] - field[cell(c,j,im2)]);
    if (kind == DX_FWD)
        return a * (field[cell(c,j,ip1)] - field[cell(c,j,i)])
             + b * (field[cell(c,j,ip2)] - field[cell(c,j,im1)]);
    if (kind == DY_BACK)
        return a * (field[cell(c,j,i)] - field[cell(c,jm1,i)])
             + b * (field[cell(c,jp1,i)] - field[cell(c,jm2,i)]);
    return a * (field[cell(c,jp1,i)] - field[cell(c,j,i)])
         + b * (field[cell(c,jp2,i)] - field[cell(c,jm1,i)]);
}

static void derivative_transpose_add(const struct denise_elastic_psv_born *c,
                                     double *destination, const double *bar,
                                     int kind, double scale) {
    const double a = 9.0 / 8.0;
    const double b = -1.0 / 24.0;
    int i, j;
    for (j = 0; j < c->ny; ++j) for (i = 0; i < c->nx; ++i) {
        size_t p = cell(c,j,i);
        double value = scale * bar[p];
        if (kind == DX_BACK) {
            destination[p] += a * value;
            destination[cell(c,j,wrap(i-1,c->nx))] -= a * value;
            destination[cell(c,j,wrap(i+1,c->nx))] += b * value;
            destination[cell(c,j,wrap(i-2,c->nx))] -= b * value;
        } else if (kind == DX_FWD) {
            destination[cell(c,j,wrap(i+1,c->nx))] += a * value;
            destination[p] -= a * value;
            destination[cell(c,j,wrap(i+2,c->nx))] += b * value;
            destination[cell(c,j,wrap(i-1,c->nx))] -= b * value;
        } else if (kind == DY_BACK) {
            destination[p] += a * value;
            destination[cell(c,wrap(j-1,c->ny),i)] -= a * value;
            destination[cell(c,wrap(j+1,c->ny),i)] += b * value;
            destination[cell(c,wrap(j-2,c->ny),i)] -= b * value;
        } else {
            destination[cell(c,wrap(j+1,c->ny),i)] += a * value;
            destination[p] -= a * value;
            destination[cell(c,wrap(j+2,c->ny),i)] += b * value;
            destination[cell(c,wrap(j-1,c->ny),i)] -= b * value;
        }
    }
}

static const struct pml_profile *profile_for(
        const struct denise_elastic_psv_born *c, int psi) {
    if (psi == PSXX || psi == PVYX) return &c->profile[PROFILE_XH];
    if (psi == PSXYY || psi == PVYY) return &c->profile[PROFILE_Y];
    if (psi == PSXYX || psi == PVXX) return &c->profile[PROFILE_X];
    return &c->profile[PROFILE_YH];
}

static int profile_index(int psi, int j, int i) {
    return (psi == PSXX || psi == PSXYX || psi == PVXX || psi == PVYX) ? i : j;
}

static float pml_forward(const struct denise_elastic_psv_born *c, int psi_kind,
                         int j, int i, float q, float *memory) {
    const struct pml_profile *profile = profile_for(c, psi_kind);
    int k = profile_index(psi_kind, j, i);
    float next = profile->b[k] * *memory + profile->a[k] * q;
    *memory = next;
    return q / profile->kappa[k] + next;
}

static double pml_transpose(const struct denise_elastic_psv_born *c, int psi_kind,
                            int j, int i, double corrected_bar,
                            double *next_memory_bar) {
    const struct pml_profile *profile = profile_for(c, psi_kind);
    int k = profile_index(psi_kind, j, i);
    double total = corrected_bar + *next_memory_bar;
    double qbar = corrected_bar / (double)profile->kappa[k]
                + (double)profile->a[k] * total;
    *next_memory_bar = (double)profile->b[k] * total;
    return qbar;
}

static int build_profile(struct pml_profile *profile, int n, float dh, float dt,
                         int fw, int half, float speed, float reflection,
                         float power, float kmax, float fpml) {
    int i;
    double thick = (double)fw * dh;
    profile->length = n;
    profile->kappa = checked_calloc((size_t)n, sizeof(float));
    profile->a = checked_calloc((size_t)n, sizeof(float));
    profile->b = checked_calloc((size_t)n, sizeof(float));
    if (!profile->kappa || !profile->a || !profile->b) return -1;
    for (i = 0; i < n; ++i) {
        double x, left, right_origin, right, distance, normalized = 0.0;
        double damping = 0.0, kappa = 1.0, alpha = 0.0, b = 1.0, a = 0.0;
        if (fw > 0) {
            x = (double)i * dh + (half ? 0.5 * dh : 0.0);
            left = thick - x;
            right_origin = (double)(n - 1) * dh - thick;
            right = x - right_origin;
            distance = left > right ? left : right;
            if (distance >= 0.0) normalized = distance / thick;
            if (distance >= 0.0) {
                double d0 = -(power + 1.0) * speed * log(reflection) / (2.0 * thick);
                damping = d0 * pow(normalized, power);
                kappa = 1.0 + (kmax - 1.0) * pow(normalized, power);
                alpha = M_PI * fpml * (1.0 - normalized);
                if (alpha < 0.0) alpha = 0.0;
                b = exp(-(damping / kappa + alpha) * dt);
                if (fabs(damping) > 1.0e-6)
                    a = damping * (b - 1.0)
                      / (kappa * (damping + kappa * alpha));
            }
        }
        profile->kappa[i] = (float)kappa;
        profile->a[i] = (float)a;
        profile->b[i] = (float)b;
    }
    return 0;
}

static void free_profile(struct pml_profile *profile) {
    free(profile->kappa); free(profile->a); free(profile->b);
    memset(profile, 0, sizeof(*profile));
}

static int validate_config(const struct denise_elastic_psv_born_config *q) {
    int p;
    size_t cells;
    if (!q) return fail("elastic P/SV Born configuration is null");
    if (q->l != 0) return fail("elastic P/SV Born requires L=0 (got %d)", q->l);
    if (q->invmat1 != 3)
        return fail("elastic P/SV Born requires INVMAT1=3 lambda/mu/rho material semantics");
    if (q->fdorder != 4) return fail("elastic P/SV Born requires FDORDER=4 (got %d)", q->fdorder);
    if (q->ndt != 1 || q->dtinv != 1)
        return fail("elastic P/SV Born requires NDT=DTINV=1");
    if (q->free_surface != 0 && q->free_surface != 1) return fail("elastic P/SV Born FREE_SURF must be 0 or 1");
    if (q->free_surface && q->source_j == 0) return fail("elastic free surface rejects explosive source at j=1");
    if (q->free_surface && q->fw >= q->ny-3) return fail("elastic CPML overlaps the surface closure");
    if (q->boundary != 0) return fail("elastic P/SV Born does not support BOUNDARY");
    if (q->mpi_size != 1) return fail("elastic P/SV Born requires one MPI rank");
    if (q->receiver_components != 2)
        return fail("elastic P/SV Born requires direct vx/vy receiver layout");
    if (q->nx < 5 || q->ny < 5 || q->nt < 1 || q->receiver_count < 1)
        return fail("elastic P/SV Born grid/time/receiver dimensions are invalid");
    if (!(q->dh > 0.0f) || !(q->dt > 0.0f))
        return fail("elastic P/SV Born requires positive DH and DT");
    if (q->fw < 0 || 2 * q->fw >= q->nx || 2 * q->fw >= q->ny)
        return fail("elastic P/SV Born FW is invalid for this grid");
    if (!!q->cpml_enabled != (q->fw > 0))
        return fail("elastic P/SV Born CPML flag and FW disagree");
    if (!q->lambda || !q->mu || !q->rho || !q->source_samples
        || !q->receiver_i || !q->receiver_j)
        return fail("elastic P/SV Born required input pointer is null");
    if (q->source_i < 0 || q->source_i >= q->nx || q->source_j < 0 || q->source_j >= q->ny)
        return fail("elastic P/SV Born source is outside the local grid");
    if ((size_t)q->nx > SIZE_MAX / (size_t)q->ny)
        return fail("elastic P/SV Born grid size overflows size_t");
    cells = (size_t)q->nx * (size_t)q->ny;
    if (cells > (size_t)INT_MAX)
        return fail("elastic P/SV Born grid exceeds the supported flat index range");
    if ((size_t)q->nt > SIZE_MAX / (4u * cells))
        return fail("elastic P/SV Born full-storage size overflows size_t");
    if ((size_t)q->receiver_count > SIZE_MAX / (2u * (size_t)q->nt))
        return fail("elastic P/SV Born data size overflows size_t");
    for (p = 0; p < (int)cells; ++p)
        if (!(q->rho[p] > 0.0f) || !(q->mu[p] >= 0.0f) || !isfinite(q->lambda[p])
            || (q->mu[p] == 0.0f && (!isfinite(q->rho[p])
                || !(q->lambda[p] > 0.0f)))
            || (q->free_surface && !((double)q->lambda[p]+2.0*q->mu[p]>0.0)))
            return fail("elastic P/SV Born material arrays contain an invalid cell");
    for (p = 0; p < q->receiver_count; ++p)
        if (q->receiver_i[p] < 0 || q->receiver_i[p] >= q->nx
            || q->receiver_j[p] < 0 || q->receiver_j[p] >= q->ny)
            return fail("elastic P/SV Born receiver %d is outside the local grid", p);
    if (q->cpml_enabled && (!(q->pml_reflection > 0.0f) || !(q->pml_reflection < 1.0f)
        || !(q->pml_power > 0.0f) || !(q->pml_kmax >= 1.0f)
        || !(q->pml_fpml >= 0.0f) || !(q->pml_damping_speed > 0.0f)))
        return fail("elastic P/SV Born CPML parameters are invalid");
    return 0;
}

static int allocate_context_arrays(struct denise_elastic_psv_born *c) {
    c->lambda = checked_calloc(c->cells, sizeof(float));
    c->mu = checked_calloc(c->cells, sizeof(float));
    c->rho = checked_calloc(c->cells, sizeof(float));
    c->invrho_x = checked_calloc(c->cells, sizeof(float));
    c->invrho_y = checked_calloc(c->cells, sizeof(float));
    c->mu_corner = checked_calloc(c->cells, sizeof(float));
    c->source_samples = checked_calloc((size_t)c->nt, sizeof(float));
    c->receiver_i = checked_calloc((size_t)c->receiver_count, sizeof(int));
    c->receiver_j = checked_calloc((size_t)c->receiver_count, sizeof(int));
    return (!c->lambda || !c->mu || !c->rho || !c->invrho_x || !c->invrho_y
            || !c->mu_corner || !c->source_samples || !c->receiver_i
            || !c->receiver_j) ? -1 : 0;
}

static void build_material_maps(struct denise_elastic_psv_born *c) {
    int i, j;
    for (j = 0; j < c->ny; ++j) for (i = 0; i < c->nx; ++i) {
        int ip = wrap(i + 1, c->nx), jp = wrap(j + 1, c->ny);
        size_t p = cell(c,j,i);
        double m00 = c->mu[p], m10 = c->mu[cell(c,j,ip)];
        double m01 = c->mu[cell(c,jp,i)], m11 = c->mu[cell(c,jp,ip)];
        c->invrho_x[p] = (float)(2.0 / ((double)c->rho[p] + c->rho[cell(c,j,ip)]));
        c->invrho_y[p] = (float)(2.0 / ((double)c->rho[p] + c->rho[cell(c,jp,i)]));
        c->mu_corner[p] = (m00 == 0.0 || m10 == 0.0 || m01 == 0.0 || m11 == 0.0)
            ? 0.0f : (float)(4.0 / (1.0/m00 + 1.0/m10 + 1.0/m01 + 1.0/m11));
    }
}

int denise_elastic_psv_born_create(
        const struct denise_elastic_psv_born_config *q,
        struct denise_elastic_psv_born **output) {
    struct denise_elastic_psv_born *c;
    int fw;
    born_error[0] = '\0';
    if (!output) return fail("elastic P/SV Born output context pointer is null");
    *output = NULL;
    if (validate_config(q) != 0) return -1;
    c = checked_calloc(1, sizeof(*c));
    if (!c) return fail("out of memory allocating elastic P/SV Born context");
    c->nx=q->nx; c->ny=q->ny; c->nt=q->nt; c->fw=q->fw;
    c->receiver_count=q->receiver_count; c->source_i=q->source_i; c->source_j=q->source_j;
    c->cpml_enabled=q->cpml_enabled; c->dh=q->dh; c->dt=q->dt;
    c->free_surface=q->free_surface;
    c->coefficient=q->dt/q->dh; c->cells=(size_t)c->nx*c->ny;
    c->data_count=(size_t)c->nt*(size_t)c->receiver_count*2;
    if (allocate_context_arrays(c) != 0) {
        denise_elastic_psv_born_destroy(&c);
        return fail("out of memory allocating elastic P/SV Born arrays");
    }
    copy_floats(c->lambda,q->lambda,c->cells); copy_floats(c->mu,q->mu,c->cells);
    copy_floats(c->rho,q->rho,c->cells); copy_floats(c->source_samples,q->source_samples,(size_t)c->nt);
    memcpy(c->receiver_i,q->receiver_i,(size_t)c->receiver_count*sizeof(int));
    memcpy(c->receiver_j,q->receiver_j,(size_t)c->receiver_count*sizeof(int));
    build_material_maps(c);
    fw = c->cpml_enabled ? c->fw : 0;
    if (build_profile(&c->profile[PROFILE_X],c->nx,c->dh,c->dt,fw,0,q->pml_damping_speed,
                      q->pml_reflection,q->pml_power,q->pml_kmax,q->pml_fpml) != 0
        || build_profile(&c->profile[PROFILE_XH],c->nx,c->dh,c->dt,fw,1,q->pml_damping_speed,
                         q->pml_reflection,q->pml_power,q->pml_kmax,q->pml_fpml) != 0
        || build_profile(&c->profile[PROFILE_Y],c->ny,c->dh,c->dt,fw,0,q->pml_damping_speed,
                         q->pml_reflection,q->pml_power,q->pml_kmax,q->pml_fpml) != 0
        || build_profile(&c->profile[PROFILE_YH],c->ny,c->dh,c->dt,fw,1,q->pml_damping_speed,
                         q->pml_reflection,q->pml_power,q->pml_kmax,q->pml_fpml) != 0) {
        denise_elastic_psv_born_destroy(&c);
        return fail("out of memory allocating elastic P/SV Born CPML profiles");
    }
    if(c->free_surface) {
        int k,j;
        /* The frozen graph decays arbitrary inactive x-memory cotangents;
         * reachable inactive forward memories still remain exactly zero. */
        if(c->fw>0)for(k=PROFILE_X;k<=PROFILE_XH;k++)for(j=0;j<c->nx;j++)
            if(c->profile[k].a[j]==0.0f && c->profile[k].b[j]==1.0f)
                c->profile[k].b[j]=(float)exp(-M_PI*q->pml_fpml*c->dt);
        for(k=PROFILE_Y;k<=PROFILE_YH;k++)for(j=0;j<c->ny-1-c->fw;j++) {
            c->profile[k].kappa[j]=1.0f;c->profile[k].a[j]=0.0f;c->profile[k].b[j]=1.0f;
        }
    }
    *output = c;
    return 0;
}

static int allocate_forward(float **fields, float **psi, float **scratch, size_t cells) {
    int k;
    for (k=0;k<FIELD_COUNT;k++) fields[k]=NULL;
    for (k=0;k<PSI_COUNT;k++) psi[k]=NULL;
    for (k=0;k<4;k++) scratch[k]=NULL;
    for (k=0;k<FIELD_COUNT;k++) if (!(fields[k]=checked_calloc(cells,sizeof(float)))) return -1;
    for (k=0;k<PSI_COUNT;k++) if (!(psi[k]=checked_calloc(cells,sizeof(float)))) return -1;
    for (k=0;k<4;k++) if (!(scratch[k]=checked_calloc(cells,sizeof(float)))) return -1;
    return 0;
}

static void free_forward(float **fields, float **psi, float **scratch) {
    int k;
    for(k=0;k<FIELD_COUNT;k++) free(fields[k]);
    for(k=0;k<PSI_COUNT;k++) free(psi[k]);
    for(k=0;k<4;k++) free(scratch[k]);
}

static int psi_is_x_oriented(int kind) {
    return kind == PSXX || kind == PSXYX || kind == PVXX || kind == PVYX;
}

static void free_checkpoint_payloads(struct denise_elastic_psv_born *c) {
    int index;
    if (!c) return;
    if (c->checkpoint)
        for (index = 0; index < c->checkpoint_count; ++index)
            free(c->checkpoint[index].values);
    free(c->checkpoint);
    free(c->segment_strain);
    c->checkpoint = NULL;
    c->segment_strain = NULL;
    c->checkpoint_payload_values = 0;
    c->checkpoint_payload_bytes = 0;
    c->checkpoint_bytes = 0;
    c->segment_operand_bytes = 0;
    c->retained_replay_bytes = 0;
}

static void free_replay_storage(struct denise_elastic_psv_born *c,
                                int clear_schedule) {
    if (!c) return;
    free_checkpoint_payloads(c);
    if (clear_schedule) {
        free(c->segment_start);
        free(c->segment_end);
        c->segment_start = NULL;
        c->segment_end = NULL;
        c->replay_segments = 0;
        c->checkpoint_count = 0;
        c->max_segment_length = 0;
    }
}

static int calculate_schedule_bytes(size_t count, size_t *bytes) {
    size_t array_bytes;
    if (!bytes) return -1;
    *bytes = 0;
    if (checked_product(count, sizeof(int), &array_bytes) != 0
        || checked_sum(array_bytes, array_bytes, bytes) != 0) return -1;
    return 0;
}

int denise_elastic_psv_born_set_replay_segments(
        struct denise_elastic_psv_born *c, int segment_count) {
    int segment, maximum = 0;
    int *start = NULL, *end = NULL;
    size_t schedule_bytes;
    born_error[0] = '\0';
    if (!c) return fail("elastic P/SV replay context is null");
    if (c->prepared)
        return fail("elastic P/SV replay policy must be selected before prepare");
    if (segment_count < 1)
        return fail("elastic P/SV replay requires at least one segment");
    if (segment_count > c->nt) segment_count = c->nt;
    if (calculate_schedule_bytes((size_t)segment_count, &schedule_bytes) != 0)
        return fail("elastic P/SV replay schedule size overflows size_t");
    start = checked_calloc((size_t)segment_count, sizeof(*start));
    end = checked_calloc((size_t)segment_count, sizeof(*end));
    if (!start || !end) {
        free(start); free(end);
        return fail("out of memory allocating elastic P/SV replay schedule");
    }
    for (segment = 0; segment < segment_count; ++segment) {
        int length;
        start[segment] = (int)(((uint64_t)segment * (uint64_t)c->nt)
                               / (uint64_t)segment_count);
        end[segment] = (int)(((uint64_t)(segment + 1) * (uint64_t)c->nt)
                             / (uint64_t)segment_count);
        length = end[segment] - start[segment];
        if (length > maximum) maximum = length;
    }
    free_replay_storage(c, 1);
    c->segment_start = start;
    c->segment_end = end;
    c->replay_segments = segment_count;
    c->checkpoint_count = segment_count - 1;
    c->max_segment_length = maximum;
    return 0;
}

int denise_elastic_psv_born_get_segment_bounds(
        const struct denise_elastic_psv_born *c, int segment,
        int *start, int *end_exclusive) {
    if (!c || !start || !end_exclusive)
        return fail("elastic P/SV segment query received a null pointer");
    if (c->replay_segments < 1 || !c->segment_start || !c->segment_end)
        return fail("elastic P/SV segment query requires segmented mode");
    if (segment < 0 || segment >= c->replay_segments)
        return fail("elastic P/SV segment query is outside the schedule");
    *start = c->segment_start[segment];
    *end_exclusive = c->segment_end[segment];
    return 0;
}

static int checkpoint_payload_values(const struct denise_elastic_psv_born *c,
                                     size_t *values) {
    size_t fields, strips = 0;
    int kind;
    if (checked_product(5u, c->cells, &fields) != 0)
        return fail("elastic P/SV checkpoint field size overflows size_t");
    if (c->cpml_enabled) {
        for (kind = 0; kind < PSI_COUNT; ++kind) {
            const struct pml_profile *profile = profile_for(c, kind);
            size_t active = 0, kind_values;
            int coordinate;
            for (coordinate = 0; coordinate < profile->length; ++coordinate)
                if (profile->a[coordinate] != 0.0f) ++active;
            if (checked_product(active,
                                psi_is_x_oriented(kind)
                                    ? (size_t)c->ny : (size_t)c->nx,
                                &kind_values) != 0
                || checked_sum(strips, kind_values, &strips) != 0)
                return fail("elastic P/SV checkpoint CPML strip size overflows size_t");
        }
    }
    if (checked_sum(fields, strips, values) != 0)
        return fail("elastic P/SV checkpoint payload size overflows size_t");
    return 0;
}

/* Logical requested bytes only; common context fields and allocator overhead
 * are not replay-only allocations. Checkpoints are one object array, not a
 * pointer table; each object's payload is allocated separately. */
struct replay_storage_estimate {
    size_t payload_values;
    size_t payload_bytes;
    size_t checkpoint_bytes;
    size_t checkpoint_metadata_bytes;
    size_t checkpoint_pointer_bytes;
    size_t segment_schedule_bytes;
    size_t operand_bytes;
    size_t retained_bytes;
};

static int calculate_retained_replay_sizes(size_t cells, size_t payload_values,
        size_t segment_count, size_t max_segment_length,
        struct replay_storage_estimate *estimate) {
    struct replay_storage_estimate e;
    size_t operand_values;
    if (!estimate) return fail("elastic P/SV replay size result is null");
    memset(estimate, 0, sizeof(*estimate));
    memset(&e, 0, sizeof(e));
    if (!cells || !payload_values || !segment_count || !max_segment_length)
        return fail("elastic P/SV replay size inputs are invalid");
    e.payload_values = payload_values;
    if (checked_product(payload_values, sizeof(float), &e.payload_bytes) != 0
        || checked_product(segment_count - 1u, e.payload_bytes,
                           &e.checkpoint_bytes) != 0
        || checked_product(segment_count - 1u, sizeof(struct elastic_checkpoint),
                           &e.checkpoint_metadata_bytes) != 0
        || calculate_schedule_bytes(segment_count, &e.segment_schedule_bytes) != 0
        || checked_product(max_segment_length, 4u, &operand_values) != 0
        || checked_product(operand_values, cells, &operand_values) != 0
        || checked_product(operand_values, sizeof(float), &e.operand_bytes) != 0
        || checked_sum(e.checkpoint_bytes, e.checkpoint_metadata_bytes,
                       &e.retained_bytes) != 0
        || checked_sum(e.retained_bytes, e.checkpoint_pointer_bytes,
                       &e.retained_bytes) != 0
        || checked_sum(e.retained_bytes, e.segment_schedule_bytes,
                       &e.retained_bytes) != 0
        || checked_sum(e.retained_bytes, e.operand_bytes,
                       &e.retained_bytes) != 0)
        return fail("elastic P/SV retained replay storage overflows size_t");
    *estimate = e;
    return 0;
}

static int calculate_replay_sizes(const struct denise_elastic_psv_born *c,
        int segment_count, int max_segment_length,
        struct replay_storage_estimate *estimate) {
    size_t payload_values;
    if (checkpoint_payload_values(c, &payload_values) != 0) return -1;
    return calculate_retained_replay_sizes(c->cells, payload_values,
            (size_t)segment_count, (size_t)max_segment_length, estimate);
}

int denise_elastic_psv_born_estimate_replay_storage(
        const struct denise_elastic_psv_born *c, int segment_count,
        size_t *retained_bytes) {
    struct replay_storage_estimate estimate;
    int max_segment_length;
    if (!retained_bytes)
        return fail("elastic P/SV replay estimate received a null pointer");
    *retained_bytes = 0;
    if (!c) return fail("elastic P/SV replay estimate received a null pointer");
    if (segment_count < 1)
        return fail("elastic P/SV replay estimate requires at least one segment");
    if (segment_count > c->nt) segment_count = c->nt;
    max_segment_length = c->nt / segment_count
                       + (c->nt % segment_count != 0);
    if (calculate_replay_sizes(c, segment_count, max_segment_length,
                               &estimate) != 0) return -1;
    *retained_bytes = estimate.retained_bytes;
    return 0;
}

static int allocate_replay_storage(struct denise_elastic_psv_born *c) {
    struct replay_storage_estimate estimate;
    size_t operand_values;
    int index;
    free_checkpoint_payloads(c);
    if (calculate_replay_sizes(c, c->replay_segments, c->max_segment_length,
                               &estimate) != 0) return -1;
    operand_values = estimate.operand_bytes / sizeof(float);
    if (c->checkpoint_count > 0) {
        c->checkpoint = checked_calloc((size_t)c->checkpoint_count,
                                       sizeof(*c->checkpoint));
        if (!c->checkpoint)
            return fail("out of memory allocating elastic P/SV checkpoints");
        for (index = 0; index < c->checkpoint_count; ++index) {
            c->checkpoint[index].values = checked_calloc(estimate.payload_values,
                                                         sizeof(float));
            if (!c->checkpoint[index].values) {
                free_checkpoint_payloads(c);
                return fail("out of memory allocating elastic P/SV checkpoint payload");
            }
        }
    }
    c->segment_strain = checked_calloc(operand_values, sizeof(float));
    if (!c->segment_strain) {
        free_checkpoint_payloads(c);
        return fail("out of memory allocating elastic P/SV segment operands");
    }
    c->checkpoint_payload_values = estimate.payload_values;
    c->checkpoint_payload_bytes = estimate.payload_bytes;
    c->checkpoint_bytes = estimate.checkpoint_bytes;
    c->segment_operand_bytes = estimate.operand_bytes;
    c->retained_replay_bytes = estimate.retained_bytes;
    return 0;
}

static int checkpoint_capture(const struct denise_elastic_psv_born *c,
                              struct elastic_checkpoint *checkpoint,
                              int timestep, float **fields, float **psi) {
    size_t offset = 0, p;
    int kind, i, j;
    if (checkpoint_payload_values(c, &checkpoint->value_count) != 0) return -1;
    checkpoint->layout_version = ELASTIC_CHECKPOINT_LAYOUT_VERSION;
    checkpoint->timestep = timestep;
    checkpoint->nx = c->nx; checkpoint->ny = c->ny; checkpoint->fw = c->fw;
    checkpoint->fdorder = 4; checkpoint->cpml_enabled = c->cpml_enabled;
    for (kind = 0; kind < FIELD_COUNT; ++kind) {
        copy_floats(checkpoint->values + offset, fields[kind], c->cells);
        offset += c->cells;
    }
    if (c->cpml_enabled) for (kind = 0; kind < PSI_COUNT; ++kind) {
        const struct pml_profile *profile = profile_for(c, kind);
        if (psi_is_x_oriented(kind)) {
            for (j = 0; j < c->ny; ++j)
                for (i = 0; i < c->nx; ++i)
                    if (profile->a[i] != 0.0f)
                        checkpoint->values[offset++] = psi[kind][cell(c,j,i)];
            for (j = 0; j < c->ny; ++j)
                for (i = 0; i < c->nx; ++i) if (profile->a[i] == 0.0f) {
                    p = cell(c,j,i);
                    if (psi[kind][p] != 0.0f)
                        return fail("elastic P/SV x-CPML interior memory %d at (%d,%d) is nonzero (%g)",
                                    kind,i,j,(double)psi[kind][p]);
                }
        } else {
            for (i = 0; i < c->nx; ++i)
                for (j = 0; j < c->ny; ++j)
                    if (profile->a[j] != 0.0f)
                        checkpoint->values[offset++] = psi[kind][cell(c,j,i)];
            for (i = 0; i < c->nx; ++i)
                for (j = 0; j < c->ny; ++j) if (profile->a[j] == 0.0f) {
                    p = cell(c,j,i);
                    if (psi[kind][p] != 0.0f)
                        return fail("elastic P/SV y-CPML interior memory %d at (%d,%d) is nonzero (%g)",
                                    kind,i,j,(double)psi[kind][p]);
                }
        }
    }
    if (offset != checkpoint->value_count)
        return fail("elastic P/SV checkpoint capture layout mismatch");
    return 0;
}

static int checkpoint_restore(const struct denise_elastic_psv_born *c,
                              const struct elastic_checkpoint *checkpoint,
                              int expected_timestep, float **fields,
                              float **psi) {
    size_t offset = 0, expected_values;
    int kind, i, j;
    if (checkpoint_payload_values(c, &expected_values) != 0) return -1;
    if (!checkpoint || checkpoint->layout_version != ELASTIC_CHECKPOINT_LAYOUT_VERSION
        || checkpoint->timestep != expected_timestep
        || checkpoint->nx != c->nx || checkpoint->ny != c->ny
        || checkpoint->fw != c->fw || checkpoint->fdorder != 4
        || checkpoint->cpml_enabled != c->cpml_enabled
        || checkpoint->value_count != expected_values)
        return fail("elastic P/SV checkpoint metadata is incompatible");
    for (kind = 0; kind < FIELD_COUNT; ++kind) {
        copy_floats(fields[kind], checkpoint->values + offset, c->cells);
        offset += c->cells;
    }
    for (kind = 0; kind < PSI_COUNT; ++kind)
        memset(psi[kind], 0, c->cells * sizeof(float));
    if (c->cpml_enabled) for (kind = 0; kind < PSI_COUNT; ++kind) {
        const struct pml_profile *profile = profile_for(c, kind);
        if (psi_is_x_oriented(kind)) {
            for (j = 0; j < c->ny; ++j)
                for (i = 0; i < c->nx; ++i)
                    if (profile->a[i] != 0.0f)
                        psi[kind][cell(c,j,i)] = checkpoint->values[offset++];
        } else {
            for (i = 0; i < c->nx; ++i)
                for (j = 0; j < c->ny; ++j)
                    if (profile->a[j] != 0.0f)
                        psi[kind][cell(c,j,i)] = checkpoint->values[offset++];
        }
    }
    if (offset != checkpoint->value_count)
        return fail("elastic P/SV checkpoint restore layout mismatch");
    return 0;
}

/* M9d3: ghosts are algebraic scratch, never persistent checkpoint state. */
struct surface_material { double alpha, a, al, am, bl, bm; };

static struct surface_material surface_material(float lambda, float mu) {
    struct surface_material s;
    double l=lambda,m=mu,d=l+2.0*m,dd=d*d;
    s.alpha=(float)(l/d); s.a=(float)(4.0*m*(l+m)/d);
    s.al=2.0*m/dd; s.am=-2.0*l/dd;
    s.bl=4.0*m*m/dd; s.bm=4.0*(l*l+2.0*l*m+2.0*m*m)/dd;
    return s;
}

static float surface_value(const struct denise_elastic_psv_born *c,
                           const float *field,int j,int i,int kind,
                           float **q,const float *lambda,const float *mu,
                           const float *bg,const float *dl,const float *dm) {
    int m,r;
    float h=1.0f/c->coefficient;
    static const float w[4]={35.0f/16.0f,-35.0f/16.0f,21.0f/16.0f,-5.0f/16.0f};
    if(j>=0)return field[cell(c,wrap(j,c->ny),i)];
    m=-j;
    if(kind==SYY)return -field[cell(c,m,i)];
    if(kind==SXY)return -field[cell(c,m-1,i)];
    if(kind==VX) {
        float slope=0.0f;
        for(r=0;r<4;r++)slope+=w[r]*q[1][cell(c,r,i)];
        return field[cell(c,m,i)]+(2.0f*m*h)*slope;
    } else {
        size_t p=cell(c,0,i);
        struct surface_material s=surface_material(lambda[p],mu[p]);
        float slope=(float)s.alpha*q[0][p];
        if(bg) slope+=(float)(s.al*dl[p]+s.am*dm[p])*bg[p];
        return field[cell(c,m-1,i)]+((2.0f*m-1.0f)*h)*slope;
    }
}

static float surface_derivative_y(const struct denise_elastic_psv_born *c,
                                 const float *field,int j,int i,int derivative,
                                 int kind,float **q,const float *lambda,const float *mu,
                                 const float *bg,const float *dl,const float *dm) {
    int o=derivative==DY_FWD?1:0;
    float f0=surface_value(c,field,j+o,i,kind,q,lambda,mu,bg,dl,dm);
    float f1=surface_value(c,field,j+o-1,i,kind,q,lambda,mu,bg,dl,dm);
    float f2=surface_value(c,field,j+o+1,i,kind,q,lambda,mu,bg,dl,dm);
    float f3=surface_value(c,field,j+o-2,i,kind,q,lambda,mu,bg,dl,dm);
    return (9.0f/8.0f)*(f0-f1)+(-1.0f/24.0f)*(f2-f3);
}

static void surface_project(const struct denise_elastic_psv_born *c,float *syy) {
    int i;
    for(i=0;i<c->nx;i++)syy[cell(c,0,i)]=0.0f;
}

static void surface_strains(struct denise_elastic_psv_born *c,float **f,float **psi,
                            float **q,const float *lambda,const float *mu,
                            const float *bg,const float *dl,const float *dm) {
    int i,j;
    /* Correct x operands once, before any velocity ghost consumes them. */
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        size_t p=cell(c,j,i);
        q[0][p]=pml_forward(c,PVXX,j,i,c->coefficient*derivative_at(c,f[VX],j,i,DX_BACK),&psi[PVXX][p]);
        q[1][p]=pml_forward(c,PVYX,j,i,c->coefficient*derivative_at(c,f[VY],j,i,DX_FWD),&psi[PVYX][p]);
    }
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        size_t p=cell(c,j,i);
        q[2][p]=pml_forward(c,PVXY,j,i,c->coefficient*surface_derivative_y(c,f[VX],j,i,DY_FWD,VX,q,lambda,mu,bg,dl,dm),&psi[PVXY][p]);
        q[3][p]=pml_forward(c,PVYY,j,i,c->coefficient*surface_derivative_y(c,f[VY],j,i,DY_BACK,VY,q,lambda,mu,bg,dl,dm),&psi[PVYY][p]);
    }
}

static void surface_stress(const struct denise_elastic_psv_born *c,float **f,
                           float **q,const float *lambda,const float *mu,
                           const float *corner) {
    int i,j;
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        size_t p=cell(c,j,i);
        if(j==0) {
            struct surface_material s=surface_material(lambda[p],mu[p]);
            f[SXX][p]+=(float)s.a*q[0][p]; f[SYY][p]=0.0f;
        } else {
            float div=q[0][p]+q[3][p];
            f[SXX][p]+=lambda[p]*div+2.0f*mu[p]*q[0][p];
            f[SYY][p]+=lambda[p]*div+2.0f*mu[p]*q[3][p];
        }
        f[SXY][p]+=corner[p]*(q[1][p]+q[2][p]);
    }
}


static void update_velocity(struct denise_elastic_psv_born *c, float **f, float **psi) {
    int i,j;
    for(j=0;j<c->ny;j++) for(i=0;i<c->nx;i++) {
        size_t p=cell(c,j,i);
        float qxx=c->coefficient*derivative_at(c,f[SXX],j,i,DX_FWD);
        float qxyy=c->coefficient*(c->free_surface?surface_derivative_y(c,f[SXY],j,i,DY_BACK,SXY,NULL,NULL,NULL,NULL,NULL,NULL):derivative_at(c,f[SXY],j,i,DY_BACK));
        float qxyx=c->coefficient*derivative_at(c,f[SXY],j,i,DX_BACK);
        float qyy=c->coefficient*(c->free_surface?surface_derivative_y(c,f[SYY],j,i,DY_FWD,SYY,NULL,NULL,NULL,NULL,NULL,NULL):derivative_at(c,f[SYY],j,i,DY_FWD));
        qxx=pml_forward(c,PSXX,j,i,qxx,&psi[PSXX][p]);
        qxyy=pml_forward(c,PSXYY,j,i,qxyy,&psi[PSXYY][p]);
        qxyx=pml_forward(c,PSXYX,j,i,qxyx,&psi[PSXYX][p]);
        qyy=pml_forward(c,PSYY,j,i,qyy,&psi[PSYY][p]);
        f[VX][p]+=c->invrho_x[p]*(qxx+qxyy);
        f[VY][p]+=c->invrho_y[p]*(qxyx+qyy);
    }
}

static void corrected_strains(struct denise_elastic_psv_born *c, float **f,
                              float **psi, float **q) {
    int i,j;
    for(j=0;j<c->ny;j++) for(i=0;i<c->nx;i++) {
        size_t p=cell(c,j,i);
        q[0][p]=pml_forward(c,PVXX,j,i,c->coefficient*derivative_at(c,f[VX],j,i,DX_BACK),&psi[PVXX][p]);
        q[1][p]=pml_forward(c,PVYX,j,i,c->coefficient*derivative_at(c,f[VY],j,i,DX_FWD),&psi[PVYX][p]);
        q[2][p]=pml_forward(c,PVXY,j,i,c->coefficient*derivative_at(c,f[VX],j,i,DY_FWD),&psi[PVXY][p]);
        q[3][p]=pml_forward(c,PVYY,j,i,c->coefficient*derivative_at(c,f[VY],j,i,DY_BACK),&psi[PVYY][p]);
    }
}

static void update_stress(const struct denise_elastic_psv_born *c, float **f,
                          float **q, const float *lambda, const float *mu,
                          const float *corner) {
    size_t p;
    for(p=0;p<c->cells;p++) {
        float div=q[0][p]+q[3][p];
        f[SXX][p]+=lambda[p]*div+2.0f*mu[p]*q[0][p];
        f[SYY][p]+=lambda[p]*div+2.0f*mu[p]*q[3][p];
        f[SXY][p]+=corner[p]*(q[1][p]+q[2][p]);
    }
}

static float memory_peak(const struct denise_elastic_psv_born *c, float **psi) {
    int k; size_t p; float peak=0.0f;
    for(k=0;k<PSI_COUNT;k++) for(p=0;p<c->cells;p++) {
        float x=fabsf(psi[k][p]); if(x>peak) peak=x;
    }
    return peak;
}

static void forward_timestep(struct denise_elastic_psv_born *c,
                             float **f, float **psi, float **q,
                             const float *lambda, const float *mu,
                             const float *corner, int timestep,
                             float *data, float *strain,
                             int retain_metrics) {
    int component, receiver;
    size_t p;
    if(c->free_surface)surface_project(c,f[SYY]);
    update_velocity(c,f,psi);
    if (data) for(receiver=0;receiver<c->receiver_count;receiver++) {
        p=cell(c,c->receiver_j[receiver],c->receiver_i[receiver]);
        data[((size_t)timestep*c->receiver_count+receiver)*2]=f[VX][p];
        data[((size_t)timestep*c->receiver_count+receiver)*2+1]=f[VY][p];
    }
    if(c->free_surface)surface_strains(c,f,psi,q,lambda,mu,NULL,NULL,NULL);
    else corrected_strains(c,f,psi,q);
    if(strain) for(component=0;component<4;component++)
        copy_floats(strain+(size_t)component*c->cells,q[component],c->cells);
    if(c->free_surface)surface_stress(c,f,q,lambda,mu,corner);
    else update_stress(c,f,q,lambda,mu,corner);
    p=cell(c,c->source_j,c->source_i);
    f[SXX][p]+=c->source_samples[timestep];
    f[SYY][p]+=c->source_samples[timestep];
    if(retain_metrics) {
        float peak=memory_peak(c,psi);
        if(peak>c->cpml_memory_peak)c->cpml_memory_peak=peak;
    }
}

static int run_nonlinear(struct denise_elastic_psv_born *c, const float *lambda,
                         const float *mu, float *data, float *strain,
                         int retain_metrics, int capture_checkpoints) {
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4],*corner=NULL;
    int k,r,next_checkpoint=0; size_t p;
    if(allocate_forward(f,psi,q,c->cells)!=0) {
        free_forward(f,psi,q); return fail("out of memory allocating elastic P/SV forward state");
    }
    corner=checked_calloc(c->cells,sizeof(float));
    if(!corner) { free_forward(f,psi,q); return fail("out of memory allocating elastic shear map"); }
    for(k=0;k<c->ny;k++) for(r=0;r<c->nx;r++) {
        int ip=wrap(r+1,c->nx),jp=wrap(k+1,c->ny); p=cell(c,k,r);
        corner[p]=(mu[p]==0.0f || mu[cell(c,k,ip)]==0.0f
                    || mu[cell(c,jp,r)]==0.0f || mu[cell(c,jp,ip)]==0.0f)
                    ? 0.0f : (float)(4.0/(1.0/mu[p]+1.0/mu[cell(c,k,ip)]
                    +1.0/mu[cell(c,jp,r)]+1.0/mu[cell(c,jp,ip)]));
    }
    if(data) memset(data,0,c->data_count*sizeof(float));
    if(retain_metrics)c->cpml_memory_peak=0.0f;
    for(k=0;k<c->nt;k++) {
        float *slot = strain ? strain + (size_t)k*4*c->cells : NULL;
        forward_timestep(c,f,psi,q,lambda,mu,corner,k,data,slot,retain_metrics);
        if(capture_checkpoints && next_checkpoint<c->checkpoint_count
           && k==c->segment_end[next_checkpoint]-1) {
            if(checkpoint_capture(c,&c->checkpoint[next_checkpoint],k,f,psi)!=0) {
                free(corner); free_forward(f,psi,q); return -1;
            }
            ++next_checkpoint;
        }
    }
    if(capture_checkpoints && next_checkpoint!=c->checkpoint_count) {
        free(corner); free_forward(f,psi,q);
        return fail("elastic P/SV checkpoint schedule was not fully captured");
    }
    free(corner); free_forward(f,psi,q); return 0;
}

int denise_elastic_psv_born_prepare(struct denise_elastic_psv_born *c,
                                    float *background_data) {
    float *temporary=NULL;
    if(!c)return fail("elastic P/SV Born context is null");
    c->prepared=0; c->replayed_forward_steps_last=0;
    free(c->strain); c->strain=NULL;
    free_checkpoint_payloads(c);
    if(c->replay_segments>0) {
        if(allocate_replay_storage(c)!=0)return -1;
    } else {
        c->strain=checked_calloc((size_t)c->nt*4*c->cells,sizeof(float));
        if(!c->strain)return fail("out of memory allocating full elastic P/SV operand storage");
    }
    if(!background_data) {
        temporary=checked_calloc(c->data_count,sizeof(float));
        if(!temporary) {
            free(c->strain); c->strain=NULL; free_checkpoint_payloads(c);
            return fail("out of memory allocating background data");
        }
        background_data=temporary;
    }
    if(run_nonlinear(c,c->lambda,c->mu,background_data,c->strain,1,
                     c->replay_segments>0)!=0) {
        free(temporary); free(c->strain); c->strain=NULL;
        free_checkpoint_payloads(c); return -1;
    }
    free(temporary); c->prepared=1; return 0;
}

static void harmonic_tangent(const struct denise_elastic_psv_born *c,
                             const float *dmu, float *out) {
    int i,j;
    for(j=0;j<c->ny;j++) for(i=0;i<c->nx;i++) {
        int ip=wrap(i+1,c->nx),jp=wrap(j+1,c->ny); size_t p=cell(c,j,i);
        size_t p10=cell(c,j,ip),p01=cell(c,jp,i),p11=cell(c,jp,ip);
        double h=c->mu_corner[p];
        /* Fixed classification: a mixed/fluid corner is constant zero. */
        if(c->mu[p]==0.0f || c->mu[p10]==0.0f
           || c->mu[p01]==0.0f || c->mu[p11]==0.0f) {
            out[p]=0.0f; continue;
        }
        out[p]=(float)(0.25*h*h*(dmu[p]/((double)c->mu[p]*c->mu[p])
             +dmu[p10]/((double)c->mu[p10]*c->mu[p10])
             +dmu[p01]/((double)c->mu[p01]*c->mu[p01])
             +dmu[p11]/((double)c->mu[p11]*c->mu[p11])));
    }
}

static int replay_segment(struct denise_elastic_psv_born *c, int segment) {
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4];
    int k, start, end;
    if(segment<0||segment>=c->replay_segments)
        return fail("elastic P/SV replay segment is outside the schedule");
    start=c->segment_start[segment]; end=c->segment_end[segment];
    if(allocate_forward(f,psi,q,c->cells)!=0) {
        free_forward(f,psi,q);
        return fail("out of memory allocating elastic P/SV replay state");
    }
    if(segment>0 && checkpoint_restore(c,&c->checkpoint[segment-1],
                                       start-1,f,psi)!=0) {
        free_forward(f,psi,q); return -1;
    }
    for(k=start;k<end;k++)
        forward_timestep(c,f,psi,q,c->lambda,c->mu,c->mu_corner,k,NULL,
                         c->segment_strain+(size_t)(k-start)*4*c->cells,0);
    c->replayed_forward_steps_last+=(size_t)(end-start);
    free_forward(f,psi,q);
    return 0;
}

static const float *background_slot(const struct denise_elastic_psv_born *c,
                                    int segment, int timestep) {
    if(c->replay_segments>0)
        return c->segment_strain
             +(size_t)(timestep-c->segment_start[segment])*4*c->cells;
    return c->strain+(size_t)timestep*4*c->cells;
}

int denise_elastic_psv_born_apply_j(struct denise_elastic_psv_born *c,
                                    const float *dlam,const float *dmu,float *data) {
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4],*dcorner=NULL;
    int k,r,segment,segment_count; size_t p;
    if(!c||!dlam||!dmu||!data)return fail("elastic P/SV Born J received a null pointer");
    for(p=0;p<c->cells;p++)if(c->mu[p]==0.0f && dmu[p]!=0.0f)
        return fail("nonzero fluid dMu is outside the restricted tangent space");
    if(!c->prepared)return fail("elastic P/SV Born J requires a prepared background trajectory");
    memset(data,0,c->data_count*sizeof(float));
    if(allocate_forward(f,psi,q,c->cells)!=0) { free_forward(f,psi,q); return fail("out of memory allocating Born tangent state"); }
    dcorner=checked_calloc(c->cells,sizeof(float));
    if(!dcorner){free_forward(f,psi,q);return fail("out of memory allocating harmonic-mu tangent");}
    harmonic_tangent(c,dmu,dcorner);
    c->replayed_forward_steps_last=0;
    segment_count=c->replay_segments>0?c->replay_segments:1;
    for(segment=0;segment<segment_count;segment++) {
      int start=c->replay_segments>0?c->segment_start[segment]:0;
      int end=c->replay_segments>0?c->segment_end[segment]:c->nt;
      if(c->replay_segments>0 && replay_segment(c,segment)!=0) {
          memset(data,0,c->data_count*sizeof(float));
          free(dcorner);free_forward(f,psi,q);return -1;
      }
      for(k=start;k<end;k++) {
        const float *bg=background_slot(c,segment,k);
        if(c->free_surface)surface_project(c,f[SYY]);
        update_velocity(c,f,psi);
        for(r=0;r<c->receiver_count;r++) {
            p=cell(c,c->receiver_j[r],c->receiver_i[r]);
            data[((size_t)k*c->receiver_count+r)*2]=f[VX][p];
            data[((size_t)k*c->receiver_count+r)*2+1]=f[VY][p];
        }
        if(c->free_surface) {
            surface_strains(c,f,psi,q,c->lambda,c->mu,bg,dlam,dmu);
            surface_stress(c,f,q,c->lambda,c->mu,c->mu_corner);
        } else {
            corrected_strains(c,f,psi,q); update_stress(c,f,q,c->lambda,c->mu,c->mu_corner);
        }
        for(p=0;p<c->cells;p++) {
            float xx=bg[p],yx=bg[c->cells+p],xy=bg[2*c->cells+p],yy=bg[3*c->cells+p];
            float div=xx+yy;
            if(c->free_surface && p<(size_t)c->nx) {
                struct surface_material s=surface_material(c->lambda[p],c->mu[p]);
                f[SXX][p]+=(float)(s.bl*dlam[p]+s.bm*dmu[p])*xx;
                f[SYY][p]=0.0f;
            } else {
                f[SXX][p]+=dlam[p]*div+2.0f*dmu[p]*xx;
                f[SYY][p]+=dlam[p]*div+2.0f*dmu[p]*yy;
            }
            f[SXY][p]+=dcorner[p]*(yx+xy);
        }
      }
    }
    free(dcorner);free_forward(f,psi,q);return 0;
}

static void harmonic_transpose_add(const struct denise_elastic_psv_born *c,
                                   const double *corner_bar,double *gmu) {
    int i,j;
    for(j=0;j<c->ny;j++) for(i=0;i<c->nx;i++) {
        int ip=wrap(i+1,c->nx),jp=wrap(j+1,c->ny); size_t p=cell(c,j,i);
        size_t p10=cell(c,j,ip),p01=cell(c,jp,i),p11=cell(c,jp,ip);
        double common;
        /* No harmonic VJP to ANY contributor of a mixed/fluid corner. */
        if(c->mu[p]==0.0f || c->mu[p10]==0.0f
           || c->mu[p01]==0.0f || c->mu[p11]==0.0f)continue;
        common=0.25*(double)c->mu_corner[p]*c->mu_corner[p]*corner_bar[p];
        gmu[p]+=common/((double)c->mu[p]*c->mu[p]);
        gmu[p10]+=common/((double)c->mu[p10]*c->mu[p10]);
        gmu[p01]+=common/((double)c->mu[p01]*c->mu[p01]);
        gmu[p11]+=common/((double)c->mu[p11]*c->mu[p11]);
    }
}

static int allocate_adjoint(double **bars,double **psi,double **q,size_t cells) {
    int k; for(k=0;k<FIELD_COUNT;k++)bars[k]=NULL;for(k=0;k<PSI_COUNT;k++)psi[k]=NULL;
    *q=NULL;
    for(k=0;k<FIELD_COUNT;k++)if(!(bars[k]=checked_calloc(cells,sizeof(double))))return -1;
    for(k=0;k<PSI_COUNT;k++)if(!(psi[k]=checked_calloc(cells,sizeof(double))))return -1;
    if(!(*q=checked_calloc(cells,sizeof(double))))return -1;
    return 0;
}

static void free_adjoint(double **bars,double **psi,double *q) {
    int k;for(k=0;k<FIELD_COUNT;k++)free(bars[k]);for(k=0;k<PSI_COUNT;k++)free(psi[k]);free(q);
}

static void reverse_pml_field(const struct denise_elastic_psv_born *c,int kind,
                              double *q,double *psi) {
    int i,j;for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++){size_t p=cell(c,j,i);q[p]=pml_transpose(c,kind,j,i,q[p],&psi[p]);}
}

static void surface_value_transpose(const struct denise_elastic_psv_born *c,
                                    double *out,int row,int i,int kind,double v,
                                    double *xx,double *yx,const float *bg,
                                    double *gl,double *gm) {
    static const double w[4]={35.0/16.0,-35.0/16.0,21.0/16.0,-5.0/16.0};
    double h=(float)(1.0f/c->coefficient);
    int r;
    if(row>=0)out[cell(c,wrap(row,c->ny),i)]+=v;
    else {
        int m=-row;
        size_t p=cell(c,0,i);
        if(kind==SYY)out[cell(c,m,i)]-=v;
        else if(kind==SXY)out[cell(c,m-1,i)]-=v;
        else if(kind==VX) {
            out[cell(c,m,i)]+=v;
            for(r=0;r<4;r++)yx[cell(c,r,i)]+=2.0*m*h*w[r]*v;
        } else {
            struct surface_material s=surface_material(c->lambda[p],c->mu[p]);
            double factor=(2.0*m-1.0)*h*v;
            out[cell(c,m-1,i)]+=v;
            xx[p]+=factor*s.alpha;
            if(bg) {gl[p]+=s.al*bg[p]*factor; gm[p]+=s.am*bg[p]*factor;}
        }
    }
}

static void surface_y_transpose(const struct denise_elastic_psv_born *c,double *out,
                                const double *bar,int derivative,int kind,
                                double *xx,double *yx,const float *bg,
                                double *gl,double *gm) {
    static const double fd[4]={9.0/8.0,-9.0/8.0,-1.0/24.0,1.0/24.0};
    int i,j,k,o=derivative==DY_FWD?1:0;
    int offsets[4]={o,o-1,o+1,o-2};
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++)for(k=0;k<4;k++) {
        int row=j+offsets[k];
        double v=(double)c->coefficient*fd[k]*bar[cell(c,j,i)];
        surface_value_transpose(c,out,row,i,kind,v,xx,yx,bg,gl,gm);
    }
}

static void surface_reverse_step(struct denise_elastic_psv_born *c,double **bar,
                                 double **psi,double *q,double *gcorner,
                                 double *xybar,const float *bg,double *gl,double *gm,
                                 const float *data,int timestep) {
    double *xx=xybar,*yx=xybar+c->cells;
    size_t p;
    int i,j,r;
    for(i=0;i<c->nx;i++)bar[SYY][cell(c,0,i)]=0.0;
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        double sx,sy,ss,bx,by,bxy,byx;
        p=cell(c,j,i); sx=bar[SXX][p];sy=bar[SYY][p];ss=bar[SXY][p];
        bx=bg[p];byx=bg[c->cells+p];bxy=bg[2*c->cells+p];by=bg[3*c->cells+p];
        if(j==0) {
            struct surface_material s=surface_material(c->lambda[p],c->mu[p]);
            gl[p]+=s.bl*bx*sx;gm[p]+=s.bm*bx*sx;
            xx[p]=s.a*sx;
        } else {
            gl[p]+=(sx+sy)*(bx+by);gm[p]+=2.0*(sx*bx+sy*by);
            xx[p]=((double)c->lambda[p]+2.0*c->mu[p])*sx+c->lambda[p]*sy;
        }
        yx[p]=(double)c->mu_corner[p]*ss;
        gcorner[p]=ss*(byx+bxy);
    }
    harmonic_transpose_add(c,gcorner,gm);
    for(p=0;p<c->cells;p++)q[p]=(double)c->mu_corner[p]*bar[SXY][p];
    reverse_pml_field(c,PVXY,q,psi[PVXY]);
    surface_y_transpose(c,bar[VX],q,DY_FWD,VX,xx,yx,bg,gl,gm);
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        p=cell(c,j,i);
        q[p]=j==0?0.0:c->lambda[p]*bar[SXX][p]+((double)c->lambda[p]+2.0*c->mu[p])*bar[SYY][p];
    }
    reverse_pml_field(c,PVYY,q,psi[PVYY]);
    surface_y_transpose(c,bar[VY],q,DY_BACK,VY,xx,yx,bg,gl,gm);
    reverse_pml_field(c,PVXX,xx,psi[PVXX]);derivative_transpose_add(c,bar[VX],xx,DX_BACK,c->coefficient);
    reverse_pml_field(c,PVYX,yx,psi[PVYX]);derivative_transpose_add(c,bar[VY],yx,DX_FWD,c->coefficient);
    for(r=0;r<c->receiver_count;r++) {
        p=cell(c,c->receiver_j[r],c->receiver_i[r]);
        bar[VX][p]+=data[((size_t)timestep*c->receiver_count+r)*2];
        bar[VY][p]+=data[((size_t)timestep*c->receiver_count+r)*2+1];
    }
    for(p=0;p<c->cells;p++)q[p]=(double)c->invrho_x[p]*bar[VX][p];
    reverse_pml_field(c,PSXX,q,psi[PSXX]);derivative_transpose_add(c,bar[SXX],q,DX_FWD,c->coefficient);
    for(p=0;p<c->cells;p++)q[p]=(double)c->invrho_x[p]*bar[VX][p];
    reverse_pml_field(c,PSXYY,q,psi[PSXYY]);
    surface_y_transpose(c,bar[SXY],q,DY_BACK,SXY,NULL,NULL,NULL,NULL,NULL);
    for(p=0;p<c->cells;p++)q[p]=(double)c->invrho_y[p]*bar[VY][p];
    reverse_pml_field(c,PSXYX,q,psi[PSXYX]);derivative_transpose_add(c,bar[SXY],q,DX_BACK,c->coefficient);
    for(p=0;p<c->cells;p++)q[p]=(double)c->invrho_y[p]*bar[VY][p];
    reverse_pml_field(c,PSYY,q,psi[PSYY]);
    surface_y_transpose(c,bar[SYY],q,DY_FWD,SYY,NULL,NULL,NULL,NULL,NULL);
    for(i=0;i<c->nx;i++)bar[SYY][cell(c,0,i)]=0.0;
}


int denise_elastic_psv_born_apply_jt(struct denise_elastic_psv_born *c,
                                     const float *data,double *glam,double *gmu) {
    double *bar[FIELD_COUNT],*psi[PSI_COUNT],*q=NULL,*gcorner=NULL,*surface_q=NULL;
    int k,r,segment,segment_count;size_t p;
    if(!c||!data||!glam||!gmu)return fail("elastic P/SV Born Jt received a null pointer");

    if(!c->prepared)return fail("elastic P/SV Born Jt requires a prepared background trajectory");
    memset(glam,0,c->cells*sizeof(double));memset(gmu,0,c->cells*sizeof(double));
    if(allocate_adjoint(bar,psi,&q,c->cells)!=0){free_adjoint(bar,psi,q);return fail("out of memory allocating Born adjoint state");}
    gcorner=checked_calloc(c->cells,sizeof(double));
    if(!gcorner){free_adjoint(bar,psi,q);return fail("out of memory allocating harmonic-mu transpose");}
    if(c->free_surface) {
        surface_q=checked_calloc(2*c->cells,sizeof(double));
        if(!surface_q) {free(gcorner);free_adjoint(bar,psi,q);return fail("out of memory allocating surface reverse scratch");}
    }
    c->replayed_forward_steps_last=0;
    segment_count=c->replay_segments>0?c->replay_segments:1;
    for(segment=segment_count-1;segment>=0;segment--) {
      int start=c->replay_segments>0?c->segment_start[segment]:0;
      int end=c->replay_segments>0?c->segment_end[segment]:c->nt;
      if(c->replay_segments>0 && replay_segment(c,segment)!=0) {
          memset(glam,0,c->cells*sizeof(double));memset(gmu,0,c->cells*sizeof(double));
          free(surface_q);free(gcorner);free_adjoint(bar,psi,q);return -1;
      }
      for(k=end-1;k>=start;k--) {
        const float *bg=background_slot(c,segment,k);
        memset(gcorner,0,c->cells*sizeof(double));
        if(c->free_surface) {
            surface_reverse_step(c,bar,psi,q,gcorner,surface_q,bg,glam,gmu,data,k);
            continue;
        }
        for(p=0;p<c->cells;p++) {
            double xx=bg[p],yx=bg[c->cells+p],xy=bg[2*c->cells+p],yy=bg[3*c->cells+p];
            glam[p]+=(bar[SXX][p]+bar[SYY][p])*(xx+yy);
            gmu[p]+=2.0*(bar[SXX][p]*xx+bar[SYY][p]*yy);
            gcorner[p]=bar[SXY][p]*(yx+xy);
        }
        harmonic_transpose_add(c,gcorner,gmu);
        for(p=0;p<c->cells;p++)q[p]=((double)c->lambda[p]+2.0*c->mu[p])*bar[SXX][p]+c->lambda[p]*bar[SYY][p];
        reverse_pml_field(c,PVXX,q,psi[PVXX]);derivative_transpose_add(c,bar[VX],q,DX_BACK,c->coefficient);
        for(p=0;p<c->cells;p++)q[p]=(double)c->mu_corner[p]*bar[SXY][p];
        reverse_pml_field(c,PVYX,q,psi[PVYX]);derivative_transpose_add(c,bar[VY],q,DX_FWD,c->coefficient);
        for(p=0;p<c->cells;p++)q[p]=(double)c->mu_corner[p]*bar[SXY][p];
        reverse_pml_field(c,PVXY,q,psi[PVXY]);derivative_transpose_add(c,bar[VX],q,DY_FWD,c->coefficient);
        for(p=0;p<c->cells;p++)q[p]=(double)c->lambda[p]*bar[SXX][p]+((double)c->lambda[p]+2.0*c->mu[p])*bar[SYY][p];
        reverse_pml_field(c,PVYY,q,psi[PVYY]);derivative_transpose_add(c,bar[VY],q,DY_BACK,c->coefficient);
        for(r=0;r<c->receiver_count;r++) {
            p=cell(c,c->receiver_j[r],c->receiver_i[r]);
            bar[VX][p]+=data[((size_t)k*c->receiver_count+r)*2];
            bar[VY][p]+=data[((size_t)k*c->receiver_count+r)*2+1];
        }
        for(p=0;p<c->cells;p++)q[p]=(double)c->invrho_x[p]*bar[VX][p];
        reverse_pml_field(c,PSXX,q,psi[PSXX]);derivative_transpose_add(c,bar[SXX],q,DX_FWD,c->coefficient);
        for(p=0;p<c->cells;p++)q[p]=(double)c->invrho_x[p]*bar[VX][p];
        reverse_pml_field(c,PSXYY,q,psi[PSXYY]);derivative_transpose_add(c,bar[SXY],q,DY_BACK,c->coefficient);
        for(p=0;p<c->cells;p++)q[p]=(double)c->invrho_y[p]*bar[VY][p];
        reverse_pml_field(c,PSXYX,q,psi[PSXYX]);derivative_transpose_add(c,bar[SXY],q,DX_BACK,c->coefficient);
        for(p=0;p<c->cells;p++)q[p]=(double)c->invrho_y[p]*bar[VY][p];
        reverse_pml_field(c,PSYY,q,psi[PSYY]);derivative_transpose_add(c,bar[SYY],q,DY_FWD,c->coefficient);
      }
    }
    /* Transpose of the restricted material injection, owned by this operator.
       Wave/adjoint state and adjacent solid sensitivities are unchanged. */
    for(p=0;p<c->cells;p++)if(c->mu[p]==0.0f)gmu[p]=0.0;
    free(surface_q);free(gcorner);free_adjoint(bar,psi,q);return 0;
}

int denise_elastic_psv_born_nonlinear(struct denise_elastic_psv_born *c,
                                      const float *lambda,const float *mu,float *data) {
    size_t p;
    if(!c||!lambda||!mu||!data)return fail("elastic P/SV nonlinear map received a null pointer");
    for(p=0;p<c->cells;p++) {
        if((mu[p]==0.0f)!=(c->mu[p]==0.0f))
            return fail("nonlinear trial changes prepared fluid classification");
        if(!(mu[p]>=0.0f)||!isfinite(lambda[p])
            ||(mu[p]==0.0f && (!(lambda[p]>0.0f)||!isfinite(c->rho[p])))
            ||(c->free_surface && !((double)lambda[p]+2.0*mu[p]>0.0)))
            return fail("elastic P/SV nonlinear material is invalid");
    }
    return run_nonlinear(c,lambda,mu,data,NULL,0,0);
}

int denise_elastic_psv_born_copy_strain(const struct denise_elastic_psv_born *c,
                                        int timestep,float *out) {
    struct denise_elastic_psv_born *mutable_context;
    int segment;
    if(!c||!out)return fail("elastic P/SV strain copy received a null pointer");
    if(!c->prepared)return fail("elastic P/SV strain copy requires a prepared trajectory");
    if(timestep<0||timestep>=c->nt)return fail("elastic P/SV strain timestep is outside the trajectory");
    if(c->replay_segments<1) {
        copy_floats(out,c->strain+(size_t)timestep*4*c->cells,4*c->cells);
        return 0;
    }
    for(segment=0;segment<c->replay_segments;segment++)
        if(timestep>=c->segment_start[segment] && timestep<c->segment_end[segment])
            break;
    if(segment==c->replay_segments)
        return fail("elastic P/SV strain timestep is absent from replay schedule");
    mutable_context=(struct denise_elastic_psv_born *)c;
    mutable_context->replayed_forward_steps_last=0;
    if(replay_segment(mutable_context,segment)!=0)return -1;
    copy_floats(out,background_slot(c,segment,timestep),4*c->cells);
    return 0;
}

int denise_elastic_psv_born_storage_diagnostics(
        const struct denise_elastic_psv_born *c,
        struct denise_elastic_psv_born_storage_diagnostics *d) {
    size_t values;
    if(!c||!d)return fail("elastic P/SV storage diagnostics received a null pointer");
    memset(d,0,sizeof(*d));
    d->segmented=c->replay_segments>0;
    d->segment_count=c->replay_segments;
    d->checkpoint_count=c->checkpoint_count;
    d->max_segment_length=c->max_segment_length;
    if(c->segment_strain) {
        struct replay_storage_estimate estimate;
        if(calculate_replay_sizes(c,c->replay_segments,c->max_segment_length,
                                  &estimate)!=0)return -1;
        d->checkpoint_payload_bytes=estimate.payload_bytes;
        d->checkpoint_bytes=estimate.checkpoint_bytes;
        d->segment_operand_bytes=estimate.operand_bytes;
        d->retained_replay_bytes=estimate.retained_bytes;
        d->checkpoint_metadata_bytes=estimate.checkpoint_metadata_bytes;
        d->checkpoint_pointer_bytes=estimate.checkpoint_pointer_bytes;
        d->segment_schedule_bytes=estimate.segment_schedule_bytes;
    } else if(c->replay_segments>0) {
        if(calculate_schedule_bytes((size_t)c->replay_segments,
                                    &d->segment_schedule_bytes)!=0)
            return fail("elastic P/SV replay schedule size overflows size_t");
        d->retained_replay_bytes=d->segment_schedule_bytes;
    }
    if(checked_product(18u,c->cells,&values)!=0
       || checked_product(values,sizeof(float),&d->forward_working_bytes)!=0
       || checked_product(c->free_surface?17u:15u,c->cells,&values)!=0
       || checked_product(values,sizeof(double),&d->adjoint_working_bytes)!=0)
        return fail("elastic P/SV working-state diagnostic overflows size_t");
    d->initial_forward_steps=c->prepared?(size_t)c->nt:0;
    d->replayed_forward_steps_last=c->replayed_forward_steps_last;
    return 0;
}

int denise_elastic_psv_born_checkpoint_roundtrip(
        struct denise_elastic_psv_born *c, int timestep) {
    float *reference[FIELD_COUNT],*reference_psi[PSI_COUNT],*reference_q[4];
    float *restored[FIELD_COUNT],*restored_psi[PSI_COUNT],*restored_q[4];
    struct elastic_checkpoint checkpoint;
    size_t values;
    int k,kind,status=-1;
    memset(&checkpoint,0,sizeof(checkpoint));
    if(!c)return fail("elastic P/SV checkpoint roundtrip context is null");
    if(timestep<0||timestep>=c->nt-1)
        return fail("elastic P/SV checkpoint roundtrip needs a following timestep");
    if(checkpoint_payload_values(c,&values)!=0)return -1;
    checkpoint.values=checked_calloc(values,sizeof(float));
    if(!checkpoint.values)return fail("out of memory allocating checkpoint roundtrip payload");
    if(allocate_forward(reference,reference_psi,reference_q,c->cells)!=0) {
        free_forward(reference,reference_psi,reference_q);
        free(checkpoint.values);
        return fail("out of memory allocating uninterrupted checkpoint state");
    }
    if(allocate_forward(restored,restored_psi,restored_q,c->cells)!=0) {
        free_forward(reference,reference_psi,reference_q);
        free_forward(restored,restored_psi,restored_q);
        free(checkpoint.values);
        return fail("out of memory allocating restored checkpoint state");
    }
    for(k=0;k<=timestep;k++)
        forward_timestep(c,reference,reference_psi,reference_q,c->lambda,c->mu,
                         c->mu_corner,k,NULL,NULL,0);
    if(checkpoint_capture(c,&checkpoint,timestep,reference,reference_psi)!=0)
        goto cleanup;
    forward_timestep(c,reference,reference_psi,reference_q,c->lambda,c->mu,
                     c->mu_corner,timestep+1,NULL,NULL,0);
    if(checkpoint_restore(c,&checkpoint,timestep,restored,restored_psi)!=0)
        goto cleanup;
    forward_timestep(c,restored,restored_psi,restored_q,c->lambda,c->mu,
                     c->mu_corner,timestep+1,NULL,NULL,0);
    for(kind=0;kind<FIELD_COUNT;kind++)
        if(memcmp(reference[kind],restored[kind],c->cells*sizeof(float))!=0) {
            fail("elastic P/SV restored field differs after continued timestep");
            goto cleanup;
        }
    for(kind=0;kind<PSI_COUNT;kind++)
        if(memcmp(reference_psi[kind],restored_psi[kind],c->cells*sizeof(float))!=0) {
            fail("elastic P/SV restored CPML memory differs after continued timestep");
            goto cleanup;
        }
    status=0;
cleanup:
    free_forward(reference,reference_psi,reference_q);
    free_forward(restored,restored_psi,restored_q);
    free(checkpoint.values);
    return status;
}

int denise_elastic_psv_born_is_prepared(const struct denise_elastic_psv_born *c) {
    return c ? c->prepared : 0;
}

float denise_elastic_psv_born_cpml_memory_peak(const struct denise_elastic_psv_born *c) {
    return c ? c->cpml_memory_peak : 0.0f;
}

void denise_elastic_psv_born_destroy(struct denise_elastic_psv_born **pointer) {
    struct denise_elastic_psv_born *c;int k;
    if(!pointer||!*pointer)return;
    c=*pointer;
    free(c->lambda);free(c->mu);free(c->rho);free(c->invrho_x);free(c->invrho_y);free(c->mu_corner);
    free(c->source_samples);free(c->receiver_i);free(c->receiver_j);free(c->strain);
    free_replay_storage(c,1);
    for(k=0;k<PROFILE_COUNT;k++)free_profile(&c->profile[k]);
    free(c);*pointer=NULL;
}
