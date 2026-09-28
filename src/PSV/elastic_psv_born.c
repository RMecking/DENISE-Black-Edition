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

struct denise_elastic_psv_born {
    int nx, ny, nt, fw, receiver_count, source_i, source_j, cpml_enabled;
    size_t cells, data_count;
    float dh, dt, coefficient;
    float *lambda, *mu, *rho, *invrho_x, *invrho_y, *mu_corner;
    float *source_samples;
    int *receiver_i, *receiver_j;
    struct pml_profile profile[PROFILE_COUNT];
    float *strain;
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
    if (q->free_surface != 0) return fail("elastic P/SV Born does not support FREE_SURF");
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
        if (!(q->rho[p] > 0.0f) || !(q->mu[p] > 0.0f) || !isfinite(q->lambda[p]))
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
        c->mu_corner[p] = (float)(4.0 / (1.0/m00 + 1.0/m10 + 1.0/m01 + 1.0/m11));
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

static void update_velocity(struct denise_elastic_psv_born *c, float **f, float **psi) {
    int i,j;
    for(j=0;j<c->ny;j++) for(i=0;i<c->nx;i++) {
        size_t p=cell(c,j,i);
        float qxx=c->coefficient*derivative_at(c,f[SXX],j,i,DX_FWD);
        float qxyy=c->coefficient*derivative_at(c,f[SXY],j,i,DY_BACK);
        float qxyx=c->coefficient*derivative_at(c,f[SXY],j,i,DX_BACK);
        float qyy=c->coefficient*derivative_at(c,f[SYY],j,i,DY_FWD);
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

static int run_nonlinear(struct denise_elastic_psv_born *c, const float *lambda,
                         const float *mu, float *data, float *strain,
                         int retain_metrics) {
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4],*corner=NULL;
    int k,r; size_t p;
    if(allocate_forward(f,psi,q,c->cells)!=0) {
        free_forward(f,psi,q); return fail("out of memory allocating elastic P/SV forward state");
    }
    corner=checked_calloc(c->cells,sizeof(float));
    if(!corner) { free_forward(f,psi,q); return fail("out of memory allocating elastic shear map"); }
    for(k=0;k<c->ny;k++) for(r=0;r<c->nx;r++) {
        int ip=wrap(r+1,c->nx),jp=wrap(k+1,c->ny); p=cell(c,k,r);
        corner[p]=(float)(4.0/(1.0/mu[p]+1.0/mu[cell(c,k,ip)]
                    +1.0/mu[cell(c,jp,r)]+1.0/mu[cell(c,jp,ip)]));
    }
    memset(data,0,c->data_count*sizeof(float));
    if(retain_metrics)c->cpml_memory_peak=0.0f;
    for(k=0;k<c->nt;k++) {
        update_velocity(c,f,psi);
        for(r=0;r<c->receiver_count;r++) {
            p=cell(c,c->receiver_j[r],c->receiver_i[r]);
            data[((size_t)k*c->receiver_count+r)*2]=f[VX][p];
            data[((size_t)k*c->receiver_count+r)*2+1]=f[VY][p];
        }
        corrected_strains(c,f,psi,q);
        if(strain) for(r=0;r<4;r++)
            copy_floats(strain+(((size_t)k*4+r)*c->cells),q[r],c->cells);
        update_stress(c,f,q,lambda,mu,corner);
        p=cell(c,c->source_j,c->source_i);
        f[SXX][p]+=c->source_samples[k]; f[SYY][p]+=c->source_samples[k];
        if(retain_metrics) { float peak=memory_peak(c,psi); if(peak>c->cpml_memory_peak)c->cpml_memory_peak=peak; }
    }
    free(corner); free_forward(f,psi,q); return 0;
}

int denise_elastic_psv_born_prepare(struct denise_elastic_psv_born *c,
                                    float *background_data) {
    float *temporary=NULL;
    if(!c)return fail("elastic P/SV Born context is null");
    c->prepared=0; free(c->strain); c->strain=NULL;
    c->strain=checked_calloc((size_t)c->nt*4*c->cells,sizeof(float));
    if(!c->strain)return fail("out of memory allocating full elastic P/SV operand storage");
    if(!background_data) {
        temporary=checked_calloc(c->data_count,sizeof(float));
        if(!temporary) { free(c->strain); c->strain=NULL; return fail("out of memory allocating background data"); }
        background_data=temporary;
    }
    if(run_nonlinear(c,c->lambda,c->mu,background_data,c->strain,1)!=0) {
        free(temporary); free(c->strain); c->strain=NULL; return -1;
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
        out[p]=(float)(0.25*h*h*(dmu[p]/((double)c->mu[p]*c->mu[p])
             +dmu[p10]/((double)c->mu[p10]*c->mu[p10])
             +dmu[p01]/((double)c->mu[p01]*c->mu[p01])
             +dmu[p11]/((double)c->mu[p11]*c->mu[p11])));
    }
}

int denise_elastic_psv_born_apply_j(struct denise_elastic_psv_born *c,
                                    const float *dlam,const float *dmu,float *data) {
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4],*dcorner=NULL;
    int k,r; size_t p;
    if(!c||!dlam||!dmu||!data)return fail("elastic P/SV Born J received a null pointer");
    if(!c->prepared)return fail("elastic P/SV Born J requires a prepared background trajectory");
    if(allocate_forward(f,psi,q,c->cells)!=0) { free_forward(f,psi,q); return fail("out of memory allocating Born tangent state"); }
    dcorner=checked_calloc(c->cells,sizeof(float));
    if(!dcorner){free_forward(f,psi,q);return fail("out of memory allocating harmonic-mu tangent");}
    harmonic_tangent(c,dmu,dcorner); memset(data,0,c->data_count*sizeof(float));
    for(k=0;k<c->nt;k++) {
        const float *bg=c->strain+(size_t)k*4*c->cells;
        update_velocity(c,f,psi);
        for(r=0;r<c->receiver_count;r++) {
            p=cell(c,c->receiver_j[r],c->receiver_i[r]);
            data[((size_t)k*c->receiver_count+r)*2]=f[VX][p];
            data[((size_t)k*c->receiver_count+r)*2+1]=f[VY][p];
        }
        corrected_strains(c,f,psi,q); update_stress(c,f,q,c->lambda,c->mu,c->mu_corner);
        for(p=0;p<c->cells;p++) {
            float xx=bg[p],yx=bg[c->cells+p],xy=bg[2*c->cells+p],yy=bg[3*c->cells+p];
            float div=xx+yy;
            f[SXX][p]+=dlam[p]*div+2.0f*dmu[p]*xx;
            f[SYY][p]+=dlam[p]*div+2.0f*dmu[p]*yy;
            f[SXY][p]+=dcorner[p]*(yx+xy);
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

static void reverse_pml_field(const struct denise_elastic_psv_born *c,int kind,
                              double *q,double *psi) {
    int i,j;for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++){size_t p=cell(c,j,i);q[p]=pml_transpose(c,kind,j,i,q[p],&psi[p]);}
}

int denise_elastic_psv_born_apply_jt(struct denise_elastic_psv_born *c,
                                     const float *data,double *glam,double *gmu) {
    double *bar[FIELD_COUNT],*psi[PSI_COUNT],*q=NULL,*gcorner=NULL;
    int k,r;size_t p;
    if(!c||!data||!glam||!gmu)return fail("elastic P/SV Born Jt received a null pointer");
    if(!c->prepared)return fail("elastic P/SV Born Jt requires a prepared background trajectory");
    if(allocate_adjoint(bar,psi,&q,c->cells)!=0){free_adjoint(bar,psi,q);return fail("out of memory allocating Born adjoint state");}
    gcorner=checked_calloc(c->cells,sizeof(double));
    if(!gcorner){free_adjoint(bar,psi,q);return fail("out of memory allocating harmonic-mu transpose");}
    memset(glam,0,c->cells*sizeof(double));memset(gmu,0,c->cells*sizeof(double));
    for(k=c->nt-1;k>=0;k--) {
        const float *bg=c->strain+(size_t)k*4*c->cells;
        memset(gcorner,0,c->cells*sizeof(double));
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
    free(gcorner);free_adjoint(bar,psi,q);return 0;
}

int denise_elastic_psv_born_nonlinear(struct denise_elastic_psv_born *c,
                                      const float *lambda,const float *mu,float *data) {
    size_t p;
    if(!c||!lambda||!mu||!data)return fail("elastic P/SV nonlinear map received a null pointer");
    for(p=0;p<c->cells;p++)if(!(mu[p]>0.0f)||!isfinite(lambda[p]))return fail("elastic P/SV nonlinear material is invalid");
    return run_nonlinear(c,lambda,mu,data,NULL,0);
}

int denise_elastic_psv_born_copy_strain(const struct denise_elastic_psv_born *c,
                                        int timestep,float *out) {
    if(!c||!out)return fail("elastic P/SV strain copy received a null pointer");
    if(!c->prepared)return fail("elastic P/SV strain copy requires a prepared trajectory");
    if(timestep<0||timestep>=c->nt)return fail("elastic P/SV strain timestep is outside the trajectory");
    copy_floats(out,c->strain+(size_t)timestep*4*c->cells,4*c->cells);return 0;
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
    for(k=0;k<PROFILE_COUNT;k++)free_profile(&c->profile[k]);
    free(c);*pointer=NULL;
}
