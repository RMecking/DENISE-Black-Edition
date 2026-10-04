#include "denise_elastic_psv_born_mpi.h"

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

struct denise_elastic_psv_born_mpi {
    int nx, ny, nt, fw, receiver_count, source_i, source_j, cpml_enabled;
    int free_surface;
    size_t cells, owned_cells, data_count;
    int gx, gy, ox, oy, rank, size, px, py, left, right, top, bottom;
    int source_owned, global_receivers;
    int *receiver_ordinal;
    MPI_Comm comm;
    double *send_buffer, *receive_buffer;
    size_t halo_capacity;
    size_t forward_halo_bytes, adjoint_halo_bytes, material_halo_bytes;
    size_t forward_stress_exchanges,forward_velocity_exchanges;
    size_t adjoint_stress_exchanges,adjoint_velocity_exchanges,material_transposes;
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

const char *denise_elastic_psv_born_mpi_last_error(void) {
    return born_error[0] ? born_error : "no elastic P/SV Born error";
}

static void *checked_calloc(size_t count, size_t width) {
    if (width && count > ((size_t)-1) / width) return NULL;
    return calloc(count ? count : 1u, width);
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

static size_t cell(const struct denise_elastic_psv_born_mpi *c, int j, int i) {
    return (size_t)(j+2)*(size_t)(c->nx+4)+(size_t)(i+2);
}
static int owned(const struct denise_elastic_psv_born_mpi *c, size_t p) {
    size_t y=p/(size_t)(c->nx+4),x=p%(size_t)(c->nx+4);
    return x>=2 && x<(size_t)c->nx+2 && y>=2 && y<(size_t)c->ny+2;
}
static size_t compact(const struct denise_elastic_psv_born_mpi *c,size_t p) {
    return (p/(size_t)(c->nx+4)-2)*(size_t)c->nx+p%(size_t)(c->nx+4)-2;
}
static int agree(const struct denise_elastic_psv_born_mpi *c,int bad) {
    int any=0;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,c->comm);
    return any?fail("MPI elastic P/SV collective failure (including a remote rank)"):0;
}
static void pack(const struct denise_elastic_psv_born_mpi *c,float *out,const float *in) {
    int j;
    for(j=0;j<c->ny;j++)memcpy(out+(size_t)j*c->nx,in+cell(c,j,0),(size_t)c->nx*sizeof(float));
}
static void unpack(const struct denise_elastic_psv_born_mpi *c,float *out,const float *in) {
    int j;
    for(j=0;j<c->ny;j++)memcpy(out+cell(c,j,0),in+(size_t)j*c->nx,(size_t)c->nx*sizeof(float));
}
/* Staged x then y copy includes diagonal material ghosts. Its transpose
 * reverses y then x, clears ghosts, and adds once into the owner. */
#define DEFINE_HALO(TYPE,SUFFIX,DATATYPE) \
static void halo_##SUFFIX(struct denise_elastic_psv_born_mpi *c,TYPE *a,int transpose) { \
    TYPE *send=(TYPE *)c->send_buffer,*recv=(TYPE *)c->receive_buffer; \
    int pass,axis,side,j,i,h,n; \
    for(pass=0;pass<2;pass++) { \
        axis=transpose?1-pass:pass; \
        for(side=0;side<2;side++) { \
            int to,from; \
            n=0; \
            if(axis==0) { \
                to=side?c->left:c->right; from=side?c->right:c->left; \
                if(transpose){int t=to;to=from;from=t;} \
                for(j=0;j<c->ny;j++)for(h=0;h<2;h++) { \
                    i=transpose?(side?c->nx+h:h-2):(side?h:c->nx-2+h); \
                    send[n++]=a[cell(c,j,i)]; \
                    if(transpose)a[cell(c,j,i)]=0; \
                } \
            } else { \
                to=side?c->top:c->bottom; from=side?c->bottom:c->top; \
                if(transpose){int t=to;to=from;from=t;} \
                for(h=0;h<2;h++)for(i=-2;i<c->nx+2;i++) { \
                    j=transpose?(side?c->ny+h:h-2):(side?h:c->ny-2+h); \
                    send[n++]=a[cell(c,j,i)]; \
                    if(transpose)a[cell(c,j,i)]=0; \
                } \
            } \
            MPI_Sendrecv(send,n,DATATYPE,to,axis*2+side,recv,n,DATATYPE,from,axis*2+side,c->comm,MPI_STATUS_IGNORE); \
            n=0; \
            if(axis==0)for(j=0;j<c->ny;j++)for(h=0;h<2;h++){ \
                i=transpose?(side?h:c->nx-2+h):(side?c->nx+h:h-2); \
                if(transpose)a[cell(c,j,i)]+=recv[n++];else a[cell(c,j,i)]=recv[n++]; \
            } \
            else for(h=0;h<2;h++)for(i=-2;i<c->nx+2;i++){ \
                j=transpose?(side?h:c->ny-2+h):(side?c->ny+h:h-2); \
                if(transpose)a[cell(c,j,i)]+=recv[n++];else a[cell(c,j,i)]=recv[n++]; \
            } \
        } \
    } \
}
DEFINE_HALO(float,float,MPI_FLOAT)
DEFINE_HALO(double,double,MPI_DOUBLE)
#undef DEFINE_HALO

static void wave_halo(struct denise_elastic_psv_born_mpi *c,float **f,
                      int first,int count,int unused) {
    int k;
    (void)unused;
    for(k=first;k<first+count;k++)halo_float(c,f[k],0);
    if(first==VX)c->forward_velocity_exchanges++;else c->forward_stress_exchanges++;
    c->forward_halo_bytes+=(size_t)count*4u*(size_t)(c->nx+c->ny+4)*sizeof(float);
}
static int owned_difference(const struct denise_elastic_psv_born_mpi *c,
                            const float *a,const float *b) {
    int j;
    for(j=0;j<c->ny;j++)if(memcmp(a+cell(c,j,0),b+cell(c,j,0),
                               (size_t)c->nx*sizeof(float)))return 1;
    return 0;
}
static void copy_floats(float *destination, const float *source, size_t count) {
    memcpy(destination, source, count * sizeof(*destination));
}

static float derivative_at(const struct denise_elastic_psv_born_mpi *c,
                           const float *field, int j, int i, int kind) {
    const float a = 9.0f / 8.0f;
    const float b = -1.0f / 24.0f;
    int im2 = (i - 2), im1 = (i - 1);
    int ip1 = (i + 1), ip2 = (i + 2);
    int jm2 = (j - 2), jm1 = (j - 1);
    int jp1 = (j + 1), jp2 = (j + 2);
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

static void derivative_transpose_add(const struct denise_elastic_psv_born_mpi *c,
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
            destination[cell(c,j,(i-1))] -= a * value;
            destination[cell(c,j,(i+1))] += b * value;
            destination[cell(c,j,(i-2))] -= b * value;
        } else if (kind == DX_FWD) {
            destination[cell(c,j,(i+1))] += a * value;
            destination[p] -= a * value;
            destination[cell(c,j,(i+2))] += b * value;
            destination[cell(c,j,(i-1))] -= b * value;
        } else if (kind == DY_BACK) {
            destination[p] += a * value;
            destination[cell(c,(j-1),i)] -= a * value;
            destination[cell(c,(j+1),i)] += b * value;
            destination[cell(c,(j-2),i)] -= b * value;
        } else {
            destination[cell(c,(j+1),i)] += a * value;
            destination[p] -= a * value;
            destination[cell(c,(j+2),i)] += b * value;
            destination[cell(c,(j-1),i)] -= b * value;
        }
    }
}

static const struct pml_profile *profile_for(
        const struct denise_elastic_psv_born_mpi *c, int psi) {
    if (psi == PSXX || psi == PVYX) return &c->profile[PROFILE_XH];
    if (psi == PSXYY || psi == PVYY) return &c->profile[PROFILE_Y];
    if (psi == PSXYX || psi == PVXX) return &c->profile[PROFILE_X];
    return &c->profile[PROFILE_YH];
}

static int profile_index(int psi, int j, int i) {
    return (psi == PSXX || psi == PSXYX || psi == PVXX || psi == PVYX) ? i : j;
}

static float pml_forward(const struct denise_elastic_psv_born_mpi *c, int psi_kind,
                         int j, int i, float q, float *memory) {
    const struct pml_profile *profile = profile_for(c, psi_kind);
    int k = profile_index(psi_kind, j, i);
    float next = profile->b[k] * *memory + profile->a[k] * q;
    *memory = next;
    return q / profile->kappa[k] + next;
}

static double pml_transpose(const struct denise_elastic_psv_born_mpi *c, int psi_kind,
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

static int allocate_context_arrays(struct denise_elastic_psv_born_mpi *c) {
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

static void build_material_maps(struct denise_elastic_psv_born_mpi *c) {
    int i, j;
    for (j = 0; j < c->ny; ++j) for (i = 0; i < c->nx; ++i) {
        int ip = (i + 1), jp = (j + 1);
        size_t p = cell(c,j,i);
        double m00 = c->mu[p], m10 = c->mu[cell(c,j,ip)];
        double m01 = c->mu[cell(c,jp,i)], m11 = c->mu[cell(c,jp,ip)];
        c->invrho_x[p] = (float)(2.0 / ((double)c->rho[p] + c->rho[cell(c,j,ip)]));
        c->invrho_y[p] = (float)(2.0 / ((double)c->rho[p] + c->rho[cell(c,jp,i)]));
        c->mu_corner[p] = (m00 == 0.0 || m10 == 0.0 || m01 == 0.0 || m11 == 0.0)
            ? 0.0f : (float)(4.0 / (1.0/m00 + 1.0/m10 + 1.0/m01 + 1.0/m11));
    }
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

static void free_checkpoint_payloads(struct denise_elastic_psv_born_mpi *c) {
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

static void free_replay_storage(struct denise_elastic_psv_born_mpi *c,
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

static int core_set_replay_segments(
        struct denise_elastic_psv_born_mpi *c, int segment_count) {
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

int denise_elastic_psv_born_mpi_get_segment_bounds(
        const struct denise_elastic_psv_born_mpi *c, int segment,
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

static int checkpoint_payload_values(const struct denise_elastic_psv_born_mpi *c,
                                     size_t *values) {
    size_t fields, strips = 0;
    int kind;
    if (checked_product(5u, c->owned_cells, &fields) != 0)
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

static int calculate_replay_sizes(const struct denise_elastic_psv_born_mpi *c,
        int segment_count, int max_segment_length,
        struct replay_storage_estimate *estimate) {
    size_t payload_values;
    if (checkpoint_payload_values(c, &payload_values) != 0) return -1;
    return calculate_retained_replay_sizes(c->owned_cells, payload_values,
            (size_t)segment_count, (size_t)max_segment_length, estimate);
}

int denise_elastic_psv_born_mpi_estimate_replay_storage(
        const struct denise_elastic_psv_born_mpi *c, int segment_count,
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

static int allocate_replay_storage(struct denise_elastic_psv_born_mpi *c) {
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

static int checkpoint_capture(const struct denise_elastic_psv_born_mpi *c,
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
        pack(c,checkpoint->values+offset,fields[kind]);
        offset+=c->owned_cells;
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

static int checkpoint_restore(const struct denise_elastic_psv_born_mpi *c,
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
        memset(fields[kind],0,c->cells*sizeof(float));
        unpack(c,fields[kind],checkpoint->values+offset);
        offset+=c->owned_cells;
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

static float surface_value(const struct denise_elastic_psv_born_mpi *c,
                           const float *field,int j,int i,int kind,
                           float **q,const float *lambda,const float *mu,
                           const float *bg,const float *dl,const float *dm) {
    int m,r;
    float h=1.0f/c->coefficient;
    static const float w[4]={35.0f/16.0f,-35.0f/16.0f,21.0f/16.0f,-5.0f/16.0f};
    if(j>=0 || c->oy!=0)return field[cell(c,j,i)];
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
        if(bg) slope+=(float)(s.al*dl[p]+s.am*dm[p])*bg[compact(c,p)];
        return field[cell(c,m-1,i)]+((2.0f*m-1.0f)*h)*slope;
    }
}

static float surface_derivative_y(const struct denise_elastic_psv_born_mpi *c,
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

static void surface_project(const struct denise_elastic_psv_born_mpi *c,float *syy) {
    int i;
    if(c->oy==0)for(i=0;i<c->nx;i++)syy[cell(c,0,i)]=0.0f;
}

static void surface_strains(struct denise_elastic_psv_born_mpi *c,float **f,float **psi,
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

static void surface_stress(const struct denise_elastic_psv_born_mpi *c,float **f,
                           float **q,const float *lambda,const float *mu,
                           const float *corner) {
    int i,j;
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        size_t p=cell(c,j,i);
        if(c->oy==0 && j==0) {
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


static void update_velocity(struct denise_elastic_psv_born_mpi *c, float **f, float **psi) {
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

static void corrected_strains(struct denise_elastic_psv_born_mpi *c, float **f,
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

static void update_stress(const struct denise_elastic_psv_born_mpi *c, float **f,
                          float **q, const float *lambda, const float *mu,
                          const float *corner) {
    size_t p;
    for(p=0;p<c->cells;p++) if(owned(c,p)) {
        float div=q[0][p]+q[3][p];
        f[SXX][p]+=lambda[p]*div+2.0f*mu[p]*q[0][p];
        f[SYY][p]+=lambda[p]*div+2.0f*mu[p]*q[3][p];
        f[SXY][p]+=corner[p]*(q[1][p]+q[2][p]);
    }
}

static float memory_peak(const struct denise_elastic_psv_born_mpi *c, float **psi) {
    int k; size_t p; float peak=0.0f;
    for(k=0;k<PSI_COUNT;k++) for(p=0;p<c->cells;p++) if(owned(c,p)) {
        float x=fabsf(psi[k][p]); if(x>peak) peak=x;
    }
    return peak;
}

static void forward_timestep(struct denise_elastic_psv_born_mpi *c,
                             float **f, float **psi, float **q,
                             const float *lambda, const float *mu,
                             const float *corner, int timestep,
                             float *data, float *strain,
                             int retain_metrics) {
    int component, receiver;
    size_t p;
    if(c->free_surface)surface_project(c,f[SYY]);
    wave_halo(c,f,SXX,3,0);
    update_velocity(c,f,psi);
    if (data) for(receiver=0;receiver<c->receiver_count;receiver++) {
        p=cell(c,c->receiver_j[receiver],c->receiver_i[receiver]);
        data[((size_t)timestep*c->receiver_count+receiver)*2]=f[VX][p];
        data[((size_t)timestep*c->receiver_count+receiver)*2+1]=f[VY][p];
    }
    wave_halo(c,f,VX,2,0);
    if(c->free_surface)surface_strains(c,f,psi,q,lambda,mu,NULL,NULL,NULL);
    else corrected_strains(c,f,psi,q);
    if(strain) for(component=0;component<4;component++)
        pack(c,strain+(size_t)component*c->owned_cells,q[component]);
    if(c->free_surface)surface_stress(c,f,q,lambda,mu,corner);
    else update_stress(c,f,q,lambda,mu,corner);
    if(c->source_owned) {
        p=cell(c,c->source_j,c->source_i);
        f[SXX][p]+=c->source_samples[timestep];
        f[SYY][p]+=c->source_samples[timestep];
    }
    if(retain_metrics) {
        float peak=memory_peak(c,psi);
        if(peak>c->cpml_memory_peak)c->cpml_memory_peak=peak;
    }
}

static int run_nonlinear(struct denise_elastic_psv_born_mpi *c, const float *lambda,
                         const float *mu, float *data, float *strain,
                         int retain_metrics, int capture_checkpoints) {
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4],*corner=NULL;
    int k,r,next_checkpoint=0,status=-1; size_t p;
    if(agree(c,allocate_forward(f,psi,q,c->cells)!=0)!=0) {
        fail("out of memory allocating elastic P/SV forward state"); goto cleanup;
    }
    corner=checked_calloc(c->cells,sizeof(float));
    if(agree(c,!corner)!=0) { fail("out of memory allocating elastic shear map"); goto cleanup; }
    for(k=0;k<c->ny;k++) for(r=0;r<c->nx;r++) {
        int ip=(r+1),jp=(k+1); p=cell(c,k,r);
        corner[p]=(mu[p]==0.0f || mu[cell(c,k,ip)]==0.0f
                    || mu[cell(c,jp,r)]==0.0f || mu[cell(c,jp,ip)]==0.0f)
                    ? 0.0f : (float)(4.0/(1.0/mu[p]+1.0/mu[cell(c,k,ip)]
                    +1.0/mu[cell(c,jp,r)]+1.0/mu[cell(c,jp,ip)]));
    }
    if(data) memset(data,0,c->data_count*sizeof(float));
    if(retain_metrics)c->cpml_memory_peak=0.0f;
    for(k=0;k<c->nt;k++) {
        float *slot = strain ? strain + (size_t)k*4*c->owned_cells : NULL;
        forward_timestep(c,f,psi,q,lambda,mu,corner,k,data,slot,retain_metrics);
        if(capture_checkpoints && next_checkpoint<c->checkpoint_count
           && k==c->segment_end[next_checkpoint]-1) {
            if(agree(c,checkpoint_capture(c,&c->checkpoint[next_checkpoint],k,f,psi)!=0)!=0) {
                goto cleanup;
            }
            ++next_checkpoint;
        }
    }
    if(capture_checkpoints && next_checkpoint!=c->checkpoint_count) {
        fail("elastic P/SV checkpoint schedule was not fully captured"); goto cleanup;
    }
    status=0;
cleanup:
    free(corner); free_forward(f,psi,q); return status;
}

static int core_prepare(struct denise_elastic_psv_born_mpi *c,
                                    float *background_data) {
    float *temporary=NULL;
    if(!c)return fail("elastic P/SV Born context is null");
    c->prepared=0; c->replayed_forward_steps_last=0;
    free(c->strain); c->strain=NULL;
    free_checkpoint_payloads(c);
    if(c->replay_segments>0) {
        if(agree(c,allocate_replay_storage(c)!=0)!=0)goto failure;
    } else {
        c->strain=checked_calloc((size_t)c->nt*4*c->owned_cells,sizeof(float));
        if(agree(c,!c->strain)!=0){fail("out of memory allocating full elastic P/SV operand storage");goto failure;}
    }
    {
        temporary=checked_calloc(c->data_count,sizeof(float));
        if(agree(c,!temporary)!=0) {
            fail("out of memory allocating background data"); goto failure;
        }
        if(!background_data)background_data=temporary;
    }
    if(run_nonlinear(c,c->lambda,c->mu,background_data,c->strain,1,
                     c->replay_segments>0)!=0) {
        goto failure;
    }
    free(temporary); c->prepared=1; return 0;
failure:
    free(temporary); free(c->strain); c->strain=NULL;
    free_checkpoint_payloads(c); return -1;
}

static void harmonic_tangent(const struct denise_elastic_psv_born_mpi *c,
                             const float *dmu, float *out) {
    int i,j;
    for(j=0;j<c->ny;j++) for(i=0;i<c->nx;i++) {
        int ip=(i+1),jp=(j+1); size_t p=cell(c,j,i);
        size_t p10=cell(c,j,ip),p01=cell(c,jp,i),p11=cell(c,jp,ip);
        double h=c->mu_corner[p];
        out[p]=(float)(0.25*h*h*(dmu[p]/((double)c->mu[p]*c->mu[p])
             +dmu[p10]/((double)c->mu[p10]*c->mu[p10])
             +dmu[p01]/((double)c->mu[p01]*c->mu[p01])
             +dmu[p11]/((double)c->mu[p11]*c->mu[p11])));
    }
}

static int replay_segment(struct denise_elastic_psv_born_mpi *c, int segment) {
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4];
    int k, start, end;
    if(segment<0||segment>=c->replay_segments)
        return fail("elastic P/SV replay segment is outside the schedule");
    start=c->segment_start[segment]; end=c->segment_end[segment];
    if(agree(c,allocate_forward(f,psi,q,c->cells)!=0)!=0) {
        free_forward(f,psi,q);
        return fail("out of memory allocating elastic P/SV replay state");
    }
    if(agree(c,segment>0 && checkpoint_restore(c,&c->checkpoint[segment-1],
                                       start-1,f,psi)!=0)!=0) {
        free_forward(f,psi,q); return -1;
    }
    for(k=start;k<end;k++)
        forward_timestep(c,f,psi,q,c->lambda,c->mu,c->mu_corner,k,NULL,
                         c->segment_strain+(size_t)(k-start)*4*c->owned_cells,0);
    c->replayed_forward_steps_last+=(size_t)(end-start);
    free_forward(f,psi,q);
    return 0;
}

static const float *background_slot(const struct denise_elastic_psv_born_mpi *c,
                                    int segment, int timestep) {
    if(c->replay_segments>0)
        return c->segment_strain
             +(size_t)(timestep-c->segment_start[segment])*4*c->owned_cells;
    return c->strain+(size_t)timestep*4*c->owned_cells;
}

static int core_apply_j(struct denise_elastic_psv_born_mpi *c,
                                    const float *dlam,const float *dmu,float *data) {
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4],*dcorner=NULL;
    int k,r,segment,segment_count,status=-1; size_t p;
    if(!c||!dlam||!dmu||!data)return fail("elastic P/SV Born J received a null pointer");
    if(!c->prepared)return fail("elastic P/SV Born J requires a prepared background trajectory");
    memset(data,0,c->data_count*sizeof(float));
    if(agree(c,allocate_forward(f,psi,q,c->cells)!=0)!=0) { fail("out of memory allocating Born tangent state"); goto cleanup; }
    dcorner=checked_calloc(c->cells,sizeof(float));
    if(agree(c,!dcorner)!=0){fail("out of memory allocating harmonic-mu tangent");goto cleanup;}
    harmonic_tangent(c,dmu,dcorner);
    c->replayed_forward_steps_last=0;
    segment_count=c->replay_segments>0?c->replay_segments:1;
    for(segment=0;segment<segment_count;segment++) {
      int start=c->replay_segments>0?c->segment_start[segment]:0;
      int end=c->replay_segments>0?c->segment_end[segment]:c->nt;
      if(c->replay_segments>0 && replay_segment(c,segment)!=0) {
          memset(data,0,c->data_count*sizeof(float));
          goto cleanup;
      }
      for(k=start;k<end;k++) {
        const float *bg=background_slot(c,segment,k);
        if(c->free_surface)surface_project(c,f[SYY]);
        wave_halo(c,f,SXX,3,0);
    update_velocity(c,f,psi);
        for(r=0;r<c->receiver_count;r++) {
            p=cell(c,c->receiver_j[r],c->receiver_i[r]);
            data[((size_t)k*c->receiver_count+r)*2]=f[VX][p];
            data[((size_t)k*c->receiver_count+r)*2+1]=f[VY][p];
        }
        wave_halo(c,f,VX,2,0);
        if(c->free_surface) {
            surface_strains(c,f,psi,q,c->lambda,c->mu,bg,dlam,dmu);
            surface_stress(c,f,q,c->lambda,c->mu,c->mu_corner);
        } else {
            corrected_strains(c,f,psi,q); update_stress(c,f,q,c->lambda,c->mu,c->mu_corner);
        }
        for(p=0;p<c->cells;p++) if(owned(c,p)) {
            float xx=bg[compact(c,p)],yx=bg[c->owned_cells+compact(c,p)],xy=bg[2*c->owned_cells+compact(c,p)],yy=bg[3*c->owned_cells+compact(c,p)];
            float div=xx+yy;
            if(c->free_surface && c->oy==0 && compact(c,p)<(size_t)c->nx) {
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
    status=0;
cleanup:
    free(dcorner);free_forward(f,psi,q);return status;
}

static void harmonic_transpose_add(const struct denise_elastic_psv_born_mpi *c,
                                   const double *corner_bar,double *gmu) {
    int i,j;
    for(j=0;j<c->ny;j++) for(i=0;i<c->nx;i++) {
        int ip=(i+1),jp=(j+1); size_t p=cell(c,j,i);
        size_t p10=cell(c,j,ip),p01=cell(c,jp,i),p11=cell(c,jp,ip);
        double common=0.25*(double)c->mu_corner[p]*c->mu_corner[p]*corner_bar[p];
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

static void reverse_pml_field(const struct denise_elastic_psv_born_mpi *c,int kind,
                              double *q,double *psi) {
    int i,j;for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++){size_t p=cell(c,j,i);q[p]=pml_transpose(c,kind,j,i,q[p],&psi[p]);}
}

static void surface_value_transpose(const struct denise_elastic_psv_born_mpi *c,
                                    double *out,int row,int i,int kind,double v,
                                    double *xx,double *yx,const float *bg,
                                    double *gl,double *gm) {
    static const double w[4]={35.0/16.0,-35.0/16.0,21.0/16.0,-5.0/16.0};
    double h=(float)(1.0f/c->coefficient);
    int r;
    if(row>=0 || c->oy!=0)out[cell(c,row,i)]+=v;
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
            if(bg) {gl[p]+=s.al*bg[compact(c,p)]*factor; gm[p]+=s.am*bg[compact(c,p)]*factor;}
        }
    }
}

static void surface_y_transpose(const struct denise_elastic_psv_born_mpi *c,double *out,
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

static void surface_reverse_step(struct denise_elastic_psv_born_mpi *c,double **bar,
                                 double **psi,double *q,double *gcorner,
                                 double *xybar,const float *bg,double *gl,double *gm,
                                 const float *data,int timestep) {
    double *xx=xybar,*yx=xybar+c->cells;
    size_t p;
    int i,j,r;
    if(c->oy==0)for(i=0;i<c->nx;i++)bar[SYY][cell(c,0,i)]=0.0;
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        double sx,sy,ss,bx,by,bxy,byx;
        p=cell(c,j,i); sx=bar[SXX][p];sy=bar[SYY][p];ss=bar[SXY][p];
        bx=bg[compact(c,p)];byx=bg[c->owned_cells+compact(c,p)];bxy=bg[2*c->owned_cells+compact(c,p)];by=bg[3*c->owned_cells+compact(c,p)];
        if(c->oy==0 && j==0) {
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
    halo_double(c,gm,1);
    c->material_transposes++;
    c->material_halo_bytes+=4u*(size_t)(c->nx+c->ny+4)*sizeof(double);
    for(p=0;p<c->cells;p++)if(owned(c,p))q[p]=(double)c->mu_corner[p]*bar[SXY][p];
    reverse_pml_field(c,PVXY,q,psi[PVXY]);
    surface_y_transpose(c,bar[VX],q,DY_FWD,VX,xx,yx,bg,gl,gm);
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        p=cell(c,j,i);
        q[p]=(c->oy==0 && j==0)?0.0:c->lambda[p]*bar[SXX][p]+((double)c->lambda[p]+2.0*c->mu[p])*bar[SYY][p];
    }
    reverse_pml_field(c,PVYY,q,psi[PVYY]);
    surface_y_transpose(c,bar[VY],q,DY_BACK,VY,xx,yx,bg,gl,gm);
    reverse_pml_field(c,PVXX,xx,psi[PVXX]);derivative_transpose_add(c,bar[VX],xx,DX_BACK,c->coefficient);
    reverse_pml_field(c,PVYX,yx,psi[PVYX]);derivative_transpose_add(c,bar[VY],yx,DX_FWD,c->coefficient);
    for(r=VX;r<=VY;r++)halo_double(c,bar[r],1);
    c->adjoint_velocity_exchanges++;
    for(r=0;r<c->receiver_count;r++) {
        p=cell(c,c->receiver_j[r],c->receiver_i[r]);
        bar[VX][p]+=data[((size_t)timestep*c->receiver_count+r)*2];
        bar[VY][p]+=data[((size_t)timestep*c->receiver_count+r)*2+1];
    }
    for(p=0;p<c->cells;p++)if(owned(c,p))q[p]=(double)c->invrho_x[p]*bar[VX][p];
    reverse_pml_field(c,PSXX,q,psi[PSXX]);derivative_transpose_add(c,bar[SXX],q,DX_FWD,c->coefficient);
    for(p=0;p<c->cells;p++)if(owned(c,p))q[p]=(double)c->invrho_x[p]*bar[VX][p];
    reverse_pml_field(c,PSXYY,q,psi[PSXYY]);
    surface_y_transpose(c,bar[SXY],q,DY_BACK,SXY,NULL,NULL,NULL,NULL,NULL);
    for(p=0;p<c->cells;p++)if(owned(c,p))q[p]=(double)c->invrho_y[p]*bar[VY][p];
    reverse_pml_field(c,PSXYX,q,psi[PSXYX]);derivative_transpose_add(c,bar[SXY],q,DX_BACK,c->coefficient);
    for(p=0;p<c->cells;p++)if(owned(c,p))q[p]=(double)c->invrho_y[p]*bar[VY][p];
    reverse_pml_field(c,PSYY,q,psi[PSYY]);
    surface_y_transpose(c,bar[SYY],q,DY_FWD,SYY,NULL,NULL,NULL,NULL,NULL);
    for(r=SXX;r<=SXY;r++)halo_double(c,bar[r],1);
    c->adjoint_stress_exchanges++;
    c->adjoint_halo_bytes+=5u*4u*(size_t)(c->nx+c->ny+4)*sizeof(double);
    if(c->oy==0)for(i=0;i<c->nx;i++)bar[SYY][cell(c,0,i)]=0.0;
}

static int core_apply_jt(struct denise_elastic_psv_born_mpi *c,
                                     const float *data,double *glam,double *gmu) {
    double *bar[FIELD_COUNT],*psi[PSI_COUNT],*q=NULL,*gcorner=NULL,*surface_q=NULL;
    int k,r,segment,segment_count,status=-1;size_t p;
    if(!c||!data||!glam||!gmu)return fail("elastic P/SV Born Jt received a null pointer");
    if(!c->prepared)return fail("elastic P/SV Born Jt requires a prepared background trajectory");
    memset(glam,0,c->cells*sizeof(double));memset(gmu,0,c->cells*sizeof(double));
    if(agree(c,allocate_adjoint(bar,psi,&q,c->cells)!=0)!=0){fail("out of memory allocating Born adjoint state");goto cleanup;}
    gcorner=checked_calloc(c->cells,sizeof(double));
    if(agree(c,!gcorner)!=0){fail("out of memory allocating harmonic-mu transpose");goto cleanup;}
    if(c->free_surface) {
        surface_q=checked_calloc(2*c->cells,sizeof(double));
        if(agree(c,!surface_q)!=0) {fail("out of memory allocating surface reverse scratch");goto cleanup;}
    }
    c->replayed_forward_steps_last=0;
    segment_count=c->replay_segments>0?c->replay_segments:1;
    for(segment=segment_count-1;segment>=0;segment--) {
      int start=c->replay_segments>0?c->segment_start[segment]:0;
      int end=c->replay_segments>0?c->segment_end[segment]:c->nt;
      if(c->replay_segments>0 && replay_segment(c,segment)!=0) {
          memset(glam,0,c->cells*sizeof(double));memset(gmu,0,c->cells*sizeof(double));
          goto cleanup;
      }
      for(k=end-1;k>=start;k--) {
        const float *bg=background_slot(c,segment,k);
        memset(gcorner,0,c->cells*sizeof(double));
        if(c->free_surface) {
            surface_reverse_step(c,bar,psi,q,gcorner,surface_q,bg,glam,gmu,data,k);
            continue;
        }
        for(p=0;p<c->cells;p++) if(owned(c,p)) {
            double xx=bg[compact(c,p)],yx=bg[c->owned_cells+compact(c,p)],xy=bg[2*c->owned_cells+compact(c,p)],yy=bg[3*c->owned_cells+compact(c,p)];
            glam[p]+=(bar[SXX][p]+bar[SYY][p])*(xx+yy);
            gmu[p]+=2.0*(bar[SXX][p]*xx+bar[SYY][p]*yy);
            gcorner[p]=bar[SXY][p]*(yx+xy);
        }
        harmonic_transpose_add(c,gcorner,gmu);
        halo_double(c,gmu,1);
        c->material_transposes++;
        c->material_halo_bytes+=4u*(size_t)(c->nx+c->ny+4)*sizeof(double);
        for(p=0;p<c->cells;p++) if(owned(c,p))q[p]=((double)c->lambda[p]+2.0*c->mu[p])*bar[SXX][p]+c->lambda[p]*bar[SYY][p];
        reverse_pml_field(c,PVXX,q,psi[PVXX]);derivative_transpose_add(c,bar[VX],q,DX_BACK,c->coefficient);
        for(p=0;p<c->cells;p++) if(owned(c,p))q[p]=(double)c->mu_corner[p]*bar[SXY][p];
        reverse_pml_field(c,PVYX,q,psi[PVYX]);derivative_transpose_add(c,bar[VY],q,DX_FWD,c->coefficient);
        for(p=0;p<c->cells;p++) if(owned(c,p))q[p]=(double)c->mu_corner[p]*bar[SXY][p];
        reverse_pml_field(c,PVXY,q,psi[PVXY]);derivative_transpose_add(c,bar[VX],q,DY_FWD,c->coefficient);
        for(p=0;p<c->cells;p++) if(owned(c,p))q[p]=(double)c->lambda[p]*bar[SXX][p]+((double)c->lambda[p]+2.0*c->mu[p])*bar[SYY][p];
        reverse_pml_field(c,PVYY,q,psi[PVYY]);derivative_transpose_add(c,bar[VY],q,DY_BACK,c->coefficient);
        for(r=VX;r<=VY;r++)halo_double(c,bar[r],1);
        c->adjoint_velocity_exchanges++;
        for(r=0;r<c->receiver_count;r++) {
            p=cell(c,c->receiver_j[r],c->receiver_i[r]);
            bar[VX][p]+=data[((size_t)k*c->receiver_count+r)*2];
            bar[VY][p]+=data[((size_t)k*c->receiver_count+r)*2+1];
        }
        for(p=0;p<c->cells;p++) if(owned(c,p))q[p]=(double)c->invrho_x[p]*bar[VX][p];
        reverse_pml_field(c,PSXX,q,psi[PSXX]);derivative_transpose_add(c,bar[SXX],q,DX_FWD,c->coefficient);
        for(p=0;p<c->cells;p++) if(owned(c,p))q[p]=(double)c->invrho_x[p]*bar[VX][p];
        reverse_pml_field(c,PSXYY,q,psi[PSXYY]);derivative_transpose_add(c,bar[SXY],q,DY_BACK,c->coefficient);
        for(p=0;p<c->cells;p++) if(owned(c,p))q[p]=(double)c->invrho_y[p]*bar[VY][p];
        reverse_pml_field(c,PSXYX,q,psi[PSXYX]);derivative_transpose_add(c,bar[SXY],q,DX_BACK,c->coefficient);
        for(p=0;p<c->cells;p++) if(owned(c,p))q[p]=(double)c->invrho_y[p]*bar[VY][p];
        reverse_pml_field(c,PSYY,q,psi[PSYY]);derivative_transpose_add(c,bar[SYY],q,DY_FWD,c->coefficient);
        for(r=SXX;r<=SXY;r++)halo_double(c,bar[r],1);
        c->adjoint_stress_exchanges++;
        c->adjoint_halo_bytes+=5u*4u*(size_t)(c->nx+c->ny+4)*sizeof(double);
      }
    }
    status=0;
cleanup:
    free(surface_q);free(gcorner);free_adjoint(bar,psi,q);return status;
}

static int core_nonlinear(struct denise_elastic_psv_born_mpi *c,
                                      const float *lambda,const float *mu,float *data) {
    if(!c||!lambda||!mu||!data)return fail("elastic P/SV nonlinear map received a null pointer");
    /* Collective validation, including classification, precedes compact halo
     * transport in the public nonlinear entry point. */
    return run_nonlinear(c,lambda,mu,data,NULL,0,0);
}

static int core_copy_strain(const struct denise_elastic_psv_born_mpi *c,
                                        int timestep,float *out) {
    struct denise_elastic_psv_born_mpi *mutable_context;
    int segment;
    if(!c||!out)return fail("elastic P/SV strain copy received a null pointer");
    if(!c->prepared)return fail("elastic P/SV strain copy requires a prepared trajectory");
    if(timestep<0||timestep>=c->nt)return fail("elastic P/SV strain timestep is outside the trajectory");
    if(c->replay_segments<1) {
        copy_floats(out,c->strain+(size_t)timestep*4*c->owned_cells,4*c->owned_cells);
        return 0;
    }
    for(segment=0;segment<c->replay_segments;segment++)
        if(timestep>=c->segment_start[segment] && timestep<c->segment_end[segment])
            break;
    if(segment==c->replay_segments)
        return fail("elastic P/SV strain timestep is absent from replay schedule");
    mutable_context=(struct denise_elastic_psv_born_mpi *)c;
    mutable_context->replayed_forward_steps_last=0;
    if(replay_segment(mutable_context,segment)!=0)return -1;
    copy_floats(out,background_slot(c,segment,timestep),4*c->owned_cells);
    return 0;
}

int denise_elastic_psv_born_mpi_storage_diagnostics(
        const struct denise_elastic_psv_born_mpi *c,
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

static int core_checkpoint_roundtrip(
        struct denise_elastic_psv_born_mpi *c, int timestep) {
    float *reference[FIELD_COUNT]={NULL},*reference_psi[PSI_COUNT]={NULL},*reference_q[4]={NULL};
    float *restored[FIELD_COUNT]={NULL},*restored_psi[PSI_COUNT]={NULL},*restored_q[4]={NULL};
    struct elastic_checkpoint checkpoint;
    size_t values;
    int k,kind,status=-1;
    memset(&checkpoint,0,sizeof(checkpoint));
    if(!c)return fail("elastic P/SV checkpoint roundtrip context is null");
    if(timestep<0||timestep>=c->nt-1)
        return fail("elastic P/SV checkpoint roundtrip needs a following timestep");
    if(checkpoint_payload_values(c,&values)!=0)return -1;
    checkpoint.values=checked_calloc(values,sizeof(float));
    if(agree(c,!checkpoint.values)!=0){fail("out of memory allocating checkpoint roundtrip payload");goto cleanup;}
    if(agree(c,allocate_forward(reference,reference_psi,reference_q,c->cells)!=0)!=0) {
        fail("out of memory allocating uninterrupted checkpoint state"); goto cleanup;
    }
    if(agree(c,allocate_forward(restored,restored_psi,restored_q,c->cells)!=0)!=0) {
        fail("out of memory allocating restored checkpoint state"); goto cleanup;
    }
    for(k=0;k<=timestep;k++)
        forward_timestep(c,reference,reference_psi,reference_q,c->lambda,c->mu,
                         c->mu_corner,k,NULL,NULL,0);
    if(agree(c,checkpoint_capture(c,&checkpoint,timestep,reference,reference_psi)!=0)!=0)
        goto cleanup;
    forward_timestep(c,reference,reference_psi,reference_q,c->lambda,c->mu,
                     c->mu_corner,timestep+1,NULL,NULL,0);
    if(agree(c,checkpoint_restore(c,&checkpoint,timestep,restored,restored_psi)!=0)!=0)
        goto cleanup;
    if(c->free_surface) {
        size_t p;
        for(kind=0;kind<FIELD_COUNT;kind++)for(p=0;p<c->cells;p++)
            if(!owned(c,p))restored[kind][p]=NAN;
    }
    forward_timestep(c,restored,restored_psi,restored_q,c->lambda,c->mu,
                     c->mu_corner,timestep+1,NULL,NULL,0);
    for(kind=0;kind<FIELD_COUNT;kind++)
        if(owned_difference(c,reference[kind],restored[kind])) {
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

int denise_elastic_psv_born_mpi_is_prepared(const struct denise_elastic_psv_born_mpi *c) {
    return c ? c->prepared : 0;
}

float denise_elastic_psv_born_mpi_cpml_memory_peak(const struct denise_elastic_psv_born_mpi *c) {
    return c ? c->cpml_memory_peak : 0.0f;
}

void denise_elastic_psv_born_mpi_destroy(struct denise_elastic_psv_born_mpi **pointer) {
    struct denise_elastic_psv_born_mpi *c;int k;
    if(!pointer||!*pointer)return;
    c=*pointer;
    free(c->lambda);free(c->mu);free(c->rho);free(c->invrho_x);free(c->invrho_y);free(c->mu_corner);
    free(c->source_samples);free(c->receiver_i);free(c->receiver_j);free(c->strain);
    free_replay_storage(c,1);
    for(k=0;k<PROFILE_COUNT;k++)free_profile(&c->profile[k]);
    free(c->receiver_ordinal);free(c->send_buffer);free(c->receive_buffer);
    if(c->comm!=MPI_COMM_NULL)MPI_Comm_free(&c->comm);
    free(c);*pointer=NULL;
}

static int same_int(MPI_Comm comm,int value) {
    int lo,hi;
    MPI_Allreduce(&value,&lo,1,MPI_INT,MPI_MIN,comm);
    MPI_Allreduce(&value,&hi,1,MPI_INT,MPI_MAX,comm);
    return lo==hi;
}
static uint64_t signature(const struct denise_elastic_psv_born_config *q) {
    uint64_t h=UINT64_C(1469598103934665603);
    const unsigned char *p;size_t k;int r;
#define HASH(V) do {p=(const unsigned char *)&(V);for(k=0;k<sizeof(V);k++){h^=p[k];h*=UINT64_C(1099511628211);}}while(0)
    HASH(q->nx);HASH(q->ny);HASH(q->nt);HASH(q->fw);HASH(q->dh);HASH(q->dt);
    HASH(q->l);HASH(q->invmat1);HASH(q->fdorder);HASH(q->ndt);HASH(q->dtinv);
    HASH(q->free_surface);HASH(q->boundary);HASH(q->mpi_size);HASH(q->receiver_components);
    HASH(q->source_i);HASH(q->source_j);HASH(q->receiver_count);HASH(q->cpml_enabled);
    HASH(q->pml_reflection);HASH(q->pml_power);HASH(q->pml_kmax);HASH(q->pml_fpml);HASH(q->pml_damping_speed);
    for(r=0;r<q->receiver_count;r++){HASH(q->receiver_i[r]);HASH(q->receiver_j[r]);}
    for(r=0;r<q->nt;r++)HASH(q->source_samples[r]);
#undef HASH
    return h;
}

int denise_elastic_psv_born_mpi_create(
        const struct denise_elastic_psv_born_mpi_config *cfg,
        struct denise_elastic_psv_born_mpi **output) {
    struct denise_elastic_psv_born_mpi *c=NULL;
    const struct denise_elastic_psv_born_config *q;
    MPI_Comm comm;
    int rank,size,bad=0,any,r,i,k,lnx=0,lny=0,owner_count;
    unsigned long long hash,lo,hi;
    size_t n;
    if(!cfg || cfg->communicator==MPI_COMM_NULL)return fail("MPI configuration/communicator is null");
    comm=cfg->communicator;q=&cfg->global;
    MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);
    if(output)*output=NULL;else bad=1;
    if(!same_int(comm,cfg->nprocx)||!same_int(comm,cfg->nprocy))bad=1;
    if(cfg->nprocx<1||cfg->nprocy<1 || (int64_t)cfg->nprocx*cfg->nprocy!=size)bad=1;
    if(q->nx<5||q->ny<5||q->nt<1||q->receiver_count<1||q->mpi_size!=size)bad=1;
    if(q->l!=0||q->invmat1!=3||q->fdorder!=4||q->ndt!=1||q->dtinv!=1||(q->free_surface!=0&&q->free_surface!=1)||q->boundary||q->receiver_components!=2)bad=1;
    if(!isfinite(q->dh)||!isfinite(q->dt)||!(q->dh>0)||!(q->dt>0))bad=1;
    if(cfg->nprocx>0&&cfg->nprocy>0){
        lnx=q->nx/cfg->nprocx;lny=q->ny/cfg->nprocy;
        if(q->nx%cfg->nprocx||q->ny%cfg->nprocy||lnx<2||lny<(q->free_surface?4:2))bad=1;
    }
    /* Two owned layers supply the FD4 +/-2 one-neighbor exchange. */
    if(lnx>INT_MAX-4||lny>INT_MAX-4 || (uint64_t)(lnx+4)*(uint64_t)(lny+4)>INT_MAX)bad=1;
    if(q->fw<0||2LL*q->fw>=q->nx||2LL*q->fw>=q->ny||!!q->cpml_enabled!=(q->fw>0))bad=1;
    if(q->free_surface && (q->source_j==0 || q->fw>=q->ny-3))bad=1;
    if(q->cpml_enabled&&(!(q->pml_reflection>0)||!(q->pml_reflection<1)||!(q->pml_power>0)||!(q->pml_kmax>=1)||!(q->pml_fpml>=0)||!(q->pml_damping_speed>0)))bad=1;
    if(!q->lambda||!q->mu||!q->rho||!q->source_samples||!q->receiver_i||!q->receiver_j)bad=1;
    if(q->source_i<0||q->source_i>=q->nx||q->source_j<0||q->source_j>=q->ny)bad=1;
    if((uint64_t)q->nt*(uint64_t)q->receiver_count*2>INT_MAX)bad=1;
    {
        size_t full_values,full_bytes;
        if(lnx>0&&lny>0&&(checked_product((size_t)lnx*(size_t)lny,4u,&full_values)!=0
            ||checked_product(full_values,(size_t)q->nt,&full_values)!=0
            ||checked_product(full_values,sizeof(float),&full_bytes)!=0))bad=1;
    }
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm);
    if(any)return fail("MPI elastic configuration/topology/FD4 local minimum invalid");
    n=(size_t)lnx*lny;
    for(k=0;k<q->nt;k++)if(!isfinite(q->source_samples[k]))bad=1;
    for(r=0;r<q->receiver_count;r++)if(q->receiver_i[r]<0||q->receiver_i[r]>=q->nx||q->receiver_j[r]<0||q->receiver_j[r]>=q->ny)bad=1;
    for(n=0;n<(size_t)lnx*lny;n++)if(!isfinite(q->lambda[n])||!isfinite(q->mu[n])||!isfinite(q->rho[n])||!(q->mu[n]>=0)||!(q->rho[n]>0)||(q->mu[n]==0 && !(q->lambda[n]>0))||(q->free_surface&&!((double)q->lambda[n]+2.0*q->mu[n]>0.0)))bad=1;
    hash=(unsigned long long)signature(q);
    MPI_Allreduce(&hash,&lo,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,comm);
    MPI_Allreduce(&hash,&hi,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,comm);
    if(lo!=hi)bad=1;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm);
    if(any)return fail("MPI geometry/source/config mismatch or invalid model");
    c=checked_calloc(1,sizeof(*c));bad=!c;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,comm);
    if(any){free(c);return fail("MPI context allocation failed");}
    c->comm=MPI_COMM_NULL;
    if(MPI_Comm_dup(comm,&c->comm)!=MPI_SUCCESS){MPI_Abort(comm,70);free(c);return fail("MPI communicator duplication failed");}
    /* A transport/protocol error cannot safely be recovered by entering a
     * different collective. Recoverable local setup errors use agree(). */
    MPI_Comm_set_errhandler(c->comm,MPI_ERRORS_ARE_FATAL);
    c->rank=rank;c->size=size;c->px=cfg->nprocx;c->py=cfg->nprocy;
    c->nx=lnx;c->ny=lny;c->gx=q->nx;c->gy=q->ny;
    c->ox=(rank%c->px)*lnx;c->oy=(rank/c->px)*lny;
    c->left=(rank/c->px)*c->px+(rank%c->px+c->px-1)%c->px;
    c->right=(rank/c->px)*c->px+(rank%c->px+1)%c->px;
    c->top=((rank/c->px+c->py-1)%c->py)*c->px+rank%c->px;
    c->bottom=((rank/c->px+1)%c->py)*c->px+rank%c->px;
    c->cells=(size_t)(lnx+4)*(lny+4);c->owned_cells=(size_t)lnx*lny;
    c->nt=q->nt;c->fw=q->fw;c->dh=q->dh;c->dt=q->dt;c->coefficient=q->dt/q->dh;
    c->cpml_enabled=q->cpml_enabled;c->global_receivers=q->receiver_count;
    c->free_surface=q->free_surface;
    c->source_owned=q->source_i/lnx+c->px*(q->source_j/lny)==rank;
    c->source_i=q->source_i-c->ox;c->source_j=q->source_j-c->oy;
    c->receiver_count=q->receiver_count;
    c->halo_capacity=2u*(size_t)(lnx+4)>2u*(size_t)lny?2u*(size_t)(lnx+4):2u*(size_t)lny;
    c->send_buffer=checked_calloc(c->halo_capacity,sizeof(double));
    c->receive_buffer=checked_calloc(c->halo_capacity,sizeof(double));
    c->receiver_ordinal=checked_calloc((size_t)q->receiver_count,sizeof(int));
    bad=allocate_context_arrays(c)!=0||!c->send_buffer||!c->receive_buffer||!c->receiver_ordinal;
    if(agree(c,bad)!=0)goto failure;
    unpack(c,c->lambda,q->lambda);unpack(c,c->mu,q->mu);unpack(c,c->rho,q->rho);
    halo_float(c,c->mu,0);halo_float(c,c->rho,0);
    c->material_halo_bytes=2u*4u*(size_t)(c->nx+c->ny+4)*sizeof(float);
    copy_floats(c->source_samples,q->source_samples,(size_t)c->nt);
    c->receiver_count=0;
    for(r=0;r<q->receiver_count;r++)if(q->receiver_i[r]/lnx+c->px*(q->receiver_j[r]/lny)==rank){
        i=c->receiver_count++;c->receiver_i[i]=q->receiver_i[r]-c->ox;c->receiver_j[i]=q->receiver_j[r]-c->oy;c->receiver_ordinal[i]=r;
    }
    c->data_count=(size_t)c->nt*c->receiver_count*2;
    MPI_Allreduce(&c->source_owned,&owner_count,1,MPI_INT,MPI_SUM,c->comm);
    if(agree(c,owner_count!=1)!=0)goto failure;
    MPI_Allreduce(&c->receiver_count,&owner_count,1,MPI_INT,MPI_SUM,c->comm);
    if(agree(c,owner_count!=q->receiver_count)!=0)goto failure;
    build_material_maps(c);
    for(k=0;k<PROFILE_COUNT;k++){
        struct pml_profile global={NULL,NULL,NULL,0};
        int nx=k<2?c->gx:c->gy,len=k<2?lnx:lny,offset=k<2?c->ox:c->oy;
        bad=build_profile(&global,nx,c->dh,c->dt,c->fw,k%2,q->pml_damping_speed,q->pml_reflection,q->pml_power,q->pml_kmax,q->pml_fpml)!=0;
        c->profile[k].length=len;
        c->profile[k].kappa=checked_calloc((size_t)len,sizeof(float));
        c->profile[k].a=checked_calloc((size_t)len,sizeof(float));
        c->profile[k].b=checked_calloc((size_t)len,sizeof(float));
        bad=bad||!c->profile[k].kappa||!c->profile[k].a||!c->profile[k].b;
        if(agree(c,bad)!=0){free_profile(&global);goto failure;}
        if(c->free_surface && k<PROFILE_Y && c->fw>0) {
            int j;
            for(j=0;j<c->gx;j++)if(global.a[j]==0.0f && global.b[j]==1.0f)
                global.b[j]=(float)exp(-M_PI*q->pml_fpml*c->dt);
        }
        if(c->free_surface && k>=PROFILE_Y) {
            int j;
            for(j=0;j<c->gy-1-c->fw;j++) {global.kappa[j]=1.0f;global.a[j]=0.0f;global.b[j]=1.0f;}
        }
        copy_floats(c->profile[k].kappa,global.kappa+offset,(size_t)len);
        copy_floats(c->profile[k].a,global.a+offset,(size_t)len);
        copy_floats(c->profile[k].b,global.b+offset,(size_t)len);
        free_profile(&global);
    }
    *output=c;return 0;
failure:
    denise_elastic_psv_born_mpi_destroy(&c);return -1;
}

static int preflight(struct denise_elastic_psv_born_mpi *c,int bad) {
    if(!c)return fail("MPI context is null (collective caller contract violation)");
    return agree(c,bad);
}
int denise_elastic_psv_born_mpi_prepare(struct denise_elastic_psv_born_mpi *c,float *data) {
    if(!c)return fail("MPI context is null");
    return agree(c,core_prepare(c,data)!=0);
}
int denise_elastic_psv_born_mpi_set_replay_segments(struct denise_elastic_psv_born_mpi *c,int segments) {
    if(!c)return fail("MPI context is null");
    if(preflight(c,!same_int(c->comm,segments)||segments<1||c->prepared)!=0)return -1;
    return agree(c,core_set_replay_segments(c,segments)!=0);
}
int denise_elastic_psv_born_mpi_select_backend(struct denise_elastic_psv_born_mpi *c,int segments) {
    size_t replay=0,full;int benefit,all;
    if(!c)return fail("MPI context is null");
    if(preflight(c,!same_int(c->comm,segments)||segments<1||c->prepared)!=0)return -1;
    if(agree(c,denise_elastic_psv_born_mpi_estimate_replay_storage(c,segments,&replay)!=0)!=0)return -1;
    full=(size_t)c->nt*4*c->owned_cells*sizeof(float);
    benefit=replay<full;
    MPI_Allreduce(&benefit,&all,1,MPI_INT,MPI_MIN,c->comm);
    if(all)return denise_elastic_psv_born_mpi_set_replay_segments(c,segments);
    free_replay_storage(c,1);return 0;
}
static int compact_inputs(struct denise_elastic_psv_born_mpi *c,const float *a,const float *b,float **ha,float **hb,int material) {
    size_t p;int bad=!a||!b;
    *ha=NULL;*hb=NULL;
    if(preflight(c,bad)!=0)return -1;
    for(p=0;p<c->owned_cells;p++) {
        size_t background=cell(c,(int)(p/c->nx),(int)(p%c->nx));
        if(!isfinite(a[p])||!isfinite(b[p]))bad=1;
        if(material && (!(b[p]>=0.0f)||(b[p]==0.0f)!=(c->mu[background]==0.0f)
            ||(b[p]==0.0f && !(a[p]>0.0f))
            ||(c->free_surface && !((double)a[p]+2.0*b[p]>0.0))))bad=1;
    }
    if(agree(c,bad)!=0)return fail("MPI nonlinear/direction material invalid or changes prepared fluid classification");
    *ha=checked_calloc(c->cells,sizeof(float));*hb=checked_calloc(c->cells,sizeof(float));
    if(agree(c,!*ha||!*hb)!=0){free(*ha);free(*hb);*ha=*hb=NULL;return -1;}
    unpack(c,*ha,a);unpack(c,*hb,b);halo_float(c,*hb,0);
    c->material_halo_bytes+=4u*(size_t)(c->nx+c->ny+4)*sizeof(float);return 0;
}
int denise_elastic_psv_born_mpi_apply_j(struct denise_elastic_psv_born_mpi *c,const float *a,const float *b,float *data) {
    float *ha,*hb;int status;
    if(!c)return fail("MPI context is null");
    {
        size_t p;int fluid=0;
        for(p=0;p<c->cells;p++)if(owned(c,p)&&c->mu[p]==0.0f)fluid=1;
        if(agree(c,fluid)!=0)return fail("fluid MPI J requires FLUID-2 restricted tangent implementation");
    }
    if(preflight(c,!c->prepared||(!data&&c->data_count))!=0)return -1;
    if(compact_inputs(c,a,b,&ha,&hb,0)!=0)return -1;
    status=core_apply_j(c,ha,hb,data?data:ha);free(ha);free(hb);return agree(c,status!=0);
}
int denise_elastic_psv_born_mpi_nonlinear(struct denise_elastic_psv_born_mpi *c,const float *a,const float *b,float *data) {
    float *ha,*hb;int status;
    if(preflight(c,!c||(!data&&c->data_count))!=0)return -1;
    if(compact_inputs(c,a,b,&ha,&hb,1)!=0)return -1;
    status=core_nonlinear(c,ha,hb,data?data:ha);free(ha);free(hb);return agree(c,status!=0);
}
int denise_elastic_psv_born_mpi_apply_jt(struct denise_elastic_psv_born_mpi *c,const float *data,double *a,double *b) {
    double *ha,*hb;size_t p;int status,bad=0;float empty=0;
    if(!c)return fail("MPI context is null");
    for(p=0;p<c->cells;p++)if(owned(c,p)&&c->mu[p]==0.0f)bad=1;
    if(agree(c,bad)!=0)return fail("fluid MPI JT requires FLUID-2 restricted tangent implementation");
    if(preflight(c,!c->prepared||!a||!b||(!data&&c->data_count))!=0)return -1;
    for(p=0;p<c->data_count;p++)if(!isfinite(data[p]))bad=1;
    if(agree(c,bad)!=0)return -1;
    ha=checked_calloc(c->cells,sizeof(double));hb=checked_calloc(c->cells,sizeof(double));
    if(agree(c,!ha||!hb)!=0){free(ha);free(hb);return -1;}
    status=core_apply_jt(c,data?data:&empty,ha,hb);
    for(p=0;p<c->cells;p++)if(owned(c,p)){a[compact(c,p)]=ha[p];b[compact(c,p)]=hb[p];}
    free(ha);free(hb);return agree(c,status!=0);
}
int denise_elastic_psv_born_mpi_copy_strain(const struct denise_elastic_psv_born_mpi *c,int timestep,float *out) {
    if(!c)return fail("MPI context is null");
    if(preflight((struct denise_elastic_psv_born_mpi *)c,!same_int(c->comm,timestep)||!out||!c->prepared||timestep<0||timestep>=c->nt)!=0)return -1;
    return agree(c,core_copy_strain(c,timestep,out)!=0);
}
int denise_elastic_psv_born_mpi_checkpoint_roundtrip(struct denise_elastic_psv_born_mpi *c,int t) {
    if(!c)return fail("MPI context is null");
    if(preflight(c,!same_int(c->comm,t)||t<0||t>=c->nt-1)!=0)return -1;
    return agree(c,core_checkpoint_roundtrip(c,t)!=0);
}
int denise_elastic_psv_born_mpi_local_receivers(const struct denise_elastic_psv_born_mpi *c){return c?c->receiver_count:-1;}
int denise_elastic_psv_born_mpi_receiver_ordinal(const struct denise_elastic_psv_born_mpi *c,int r){return c&&r>=0&&r<c->receiver_count?c->receiver_ordinal[r]:-1;}

static int transfer_data(struct denise_elastic_psv_born_mpi *c,const float *input,float *output,int scatter) {
    float *buffer;int *ordinals;int rank,nrec,t,r,k;
    if(preflight(c,scatter?(!input&&c->rank==0)||(!output&&c->data_count):(!input&&c->data_count)||(!output&&c->rank==0))!=0)return -1;
    buffer=checked_calloc(c->rank==0?(size_t)c->nt*c->global_receivers*2:c->data_count,sizeof(float));
    ordinals=checked_calloc((size_t)c->global_receivers,sizeof(int));
    if(agree(c,!buffer||!ordinals)!=0){free(buffer);free(ordinals);return -1;}
    for(rank=0;rank<c->size;rank++){
        nrec=c->rank==rank?c->receiver_count:0;
        MPI_Bcast(&nrec,1,MPI_INT,rank,c->comm);
        if(c->rank==rank)memcpy(ordinals,c->receiver_ordinal,(size_t)nrec*sizeof(int));
        MPI_Bcast(ordinals,nrec,MPI_INT,rank,c->comm);
        if(scatter){
            if(c->rank==0)for(t=0;t<c->nt;t++)for(r=0;r<nrec;r++)for(k=0;k<2;k++)buffer[((size_t)t*nrec+r)*2+k]=input[((size_t)t*c->global_receivers+ordinals[r])*2+k];
            if(rank==0){if(c->rank==0&&nrec)memcpy(output,buffer,(size_t)c->nt*nrec*2*sizeof(float));}
            else {if(c->rank==0)MPI_Send(buffer,c->nt*nrec*2,MPI_FLOAT,rank,10,c->comm);if(c->rank==rank)MPI_Recv(output,c->nt*nrec*2,MPI_FLOAT,0,10,c->comm,MPI_STATUS_IGNORE);}
        }else{
            if(c->rank==rank&&nrec)memcpy(buffer,input,(size_t)c->nt*nrec*2*sizeof(float));
            if(rank!=0){if(c->rank==rank)MPI_Send(buffer,c->nt*nrec*2,MPI_FLOAT,0,10,c->comm);if(c->rank==0)MPI_Recv(buffer,c->nt*nrec*2,MPI_FLOAT,rank,10,c->comm,MPI_STATUS_IGNORE);}
            if(c->rank==0)for(t=0;t<c->nt;t++)for(r=0;r<nrec;r++)for(k=0;k<2;k++)output[((size_t)t*c->global_receivers+ordinals[r])*2+k]=buffer[((size_t)t*nrec+r)*2+k];
        }
    }
    free(buffer);free(ordinals);return 0;
}
int denise_elastic_psv_born_mpi_gather_data(struct denise_elastic_psv_born_mpi *c,const float *local,float *global){return transfer_data(c,local,global,0);}
int denise_elastic_psv_born_mpi_scatter_data(struct denise_elastic_psv_born_mpi *c,const float *global,float *local){return transfer_data(c,global,local,1);}
int denise_elastic_psv_born_mpi_gather_image(struct denise_elastic_psv_born_mpi *c,const double *local,double *global){
    double *tile;int rank,j,ox,oy;
    if(preflight(c,!local||(!global&&c->rank==0))!=0)return -1;
    tile=checked_calloc(c->owned_cells,sizeof(double));
    if(agree(c,!tile)!=0){free(tile);return -1;}
    for(rank=0;rank<c->size;rank++){
        if(c->rank==rank)memcpy(tile,local,c->owned_cells*sizeof(double));
        if(rank){if(c->rank==rank)MPI_Send(tile,(int)c->owned_cells,MPI_DOUBLE,0,11,c->comm);if(c->rank==0)MPI_Recv(tile,(int)c->owned_cells,MPI_DOUBLE,rank,11,c->comm,MPI_STATUS_IGNORE);}
        if(c->rank==0){ox=rank%c->px*c->nx;oy=rank/c->px*c->ny;for(j=0;j<c->ny;j++)memcpy(global+(size_t)(oy+j)*c->gx+ox,tile+(size_t)j*c->nx,(size_t)c->nx*sizeof(double));}
    }
    free(tile);return 0;
}
int denise_elastic_psv_born_mpi_copy_maps(const struct denise_elastic_psv_born_mpi *c,float *x,float *y,float *mu){
    if(!c||!x||!y||!mu)return fail("MPI map output is null");
    pack(c,x,c->invrho_x);pack(c,y,c->invrho_y);pack(c,mu,c->mu_corner);return 0;
}
int denise_elastic_psv_born_mpi_copy_profile(const struct denise_elastic_psv_born_mpi *c,int kind,float *k,float *a,float *b){
    const struct pml_profile *p;
    if(!c||kind<0||kind>=4||!k||!a||!b)return fail("MPI profile query is invalid");
    p=&c->profile[kind];copy_floats(k,p->kappa,(size_t)p->length);copy_floats(a,p->a,(size_t)p->length);copy_floats(b,p->b,(size_t)p->length);return 0;
}
int denise_elastic_psv_born_mpi_diagnostics(struct denise_elastic_psv_born_mpi *c,struct denise_elastic_psv_born_mpi_diagnostics *d){
    unsigned long long value;size_t replay=0;int status;
    if(preflight(c,!d)!=0)return -1;
    memset(d,0,sizeof(*d));
    status=denise_elastic_psv_born_mpi_storage_diagnostics(c,&d->replay);
    status|=denise_elastic_psv_born_mpi_estimate_replay_storage(c,c->nt<32?c->nt:32,&replay);
    if(agree(c,status!=0)!=0)return -1;
    d->rank=c->rank;d->size=c->size;d->local_nx=c->nx;d->local_ny=c->ny;d->offset_x=c->ox;d->offset_y=c->oy;d->local_receivers=c->receiver_count;d->source_owned=c->source_owned;
    d->full_bytes=(size_t)c->nt*4*c->owned_cells*sizeof(float);d->replay_estimate=replay;
    d->retained_backend_bytes=c->replay_segments?d->replay.retained_replay_bytes:(c->strain?d->full_bytes:0);
    d->model_bytes=6*c->cells*sizeof(float);d->halo_buffer_bytes=2*c->halo_capacity*sizeof(double);d->local_data_bytes=c->data_count*sizeof(float);
    d->forward_halo_bytes_per_step=5u*4u*(size_t)(c->nx+c->ny+4)*sizeof(float);
    d->adjoint_halo_bytes_per_step=5u*4u*(size_t)(c->nx+c->ny+4)*sizeof(double);
    d->material_bytes=c->material_halo_bytes;
    d->forward_stress_exchanges=c->forward_stress_exchanges;d->forward_velocity_exchanges=c->forward_velocity_exchanges;
    d->adjoint_stress_exchanges=c->adjoint_stress_exchanges;d->adjoint_velocity_exchanges=c->adjoint_velocity_exchanges;d->material_transposes=c->material_transposes;
    d->forward_halo_bytes=c->forward_halo_bytes;d->adjoint_halo_bytes=c->adjoint_halo_bytes;
    value=(unsigned long long)d->full_bytes;
    MPI_Allreduce(&value,&d->full_min,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,c->comm);MPI_Allreduce(&value,&d->full_max,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,c->comm);MPI_Allreduce(&value,&d->full_sum,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,c->comm);
    value=(unsigned long long)replay;
    MPI_Allreduce(&value,&d->replay_min,1,MPI_UNSIGNED_LONG_LONG,MPI_MIN,c->comm);MPI_Allreduce(&value,&d->replay_max,1,MPI_UNSIGNED_LONG_LONG,MPI_MAX,c->comm);MPI_Allreduce(&value,&d->replay_sum,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,c->comm);
    return 0;
}
