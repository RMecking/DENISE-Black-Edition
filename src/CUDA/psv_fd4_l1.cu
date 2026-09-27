#include "denise_cuda_psv.h"
#include "denise_cuda_psv_forward.h"
#include "denise_cuda_backend.h"
#include <cuda_runtime.h>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>

namespace {
thread_local char psv_last_error[1024] = "";

struct device_fields {
    float *vx, *vy, *sxx, *syy, *sxy;
    float *r1, *p1, *q1;
    float *rip, *rjp, *fipjp, *f, *g, *dip1, *d1, *e1;
    float *psi_sxx_x, *psi_sxy_x, *psi_vxx, *psi_vyx;
    float *psi_syy_y, *psi_sxy_y, *psi_vyy, *psi_vxy;
    float *K_x, *a_x, *b_x, *K_x_half, *a_x_half, *b_x_half;
    float *K_y, *a_y, *b_y, *K_y_half, *a_y_half, *b_y_half;
};

struct adjoint_fields {
    double *avx, *avy, *asxx, *asyy, *asxy, *ar, *ap, *aq;
    double *psxx, *psxyx, *pvxx, *pvyx;
    double *psxyy, *psyy, *pvxy, *pvyy;
    double *wxx, *wyx, *wxy, *wyy;
};

struct native_gradient_fields {
    double *gf, *gg, *gfc, *gd, *ge, *gdc, *grx, *gry;
};

struct physical_material_fields {
    float *prho, *ppi, *pu, *ptaus, *ptaup, *puipjp, *ptausipjp;
};

struct physical_gradient_fields {
    double *vp, *vs, *rho, *qp, *qs;
};

struct denise_cuda_psv_fd4_l1_impl {
    denise_cuda_psv_fd4_l1_config config;
    device_fields fields;
    float *storage;
    size_t full_nx, full_ny, full_elements;
    size_t x_cpml_elements, y_cpml_elements;
    denise_cuda_psv_fd4_l1_stats stats;
};

int contract_failure(const char *operation, const char *file, int line,
                     const char *format, ...) {
    char detail[640];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(detail, sizeof(detail), format, arguments);
    va_end(arguments);
    std::snprintf(psv_last_error, sizeof(psv_last_error),
                  "CUDA PSV FD4/L1 contract failure during %s at %s:%d: %s",
                  operation, file, line, detail);
    std::fprintf(stderr, "%s\n", psv_last_error);
    return -1;
}

int cuda_failure(const char *operation, cudaError_t status,
                 const char *file, int line) {
    std::snprintf(psv_last_error, sizeof(psv_last_error),
                  "CUDA PSV FD4/L1 failure during %s at %s:%d: %s (%s)",
                  operation, file, line,
                  cudaGetErrorName(status), cudaGetErrorString(status));
    std::fprintf(stderr, "%s\n", psv_last_error);
    return -1;
}

#define PSV_CUDA_CALL(operation, expression)                                  \
    do {                                                                       \
        const cudaError_t psv_cuda_status = (expression);                      \
        if (psv_cuda_status != cudaSuccess)                                    \
            return cuda_failure((operation), psv_cuda_status,                  \
                                __FILE__, __LINE__);                            \
    } while (0)

bool checked_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return false;
    *out = a + b;
    return true;
}
bool checked_mul(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

int validate_config(const denise_cuda_psv_fd4_l1_config *c,
                    const char *operation) {
    if (!c) return contract_failure(operation, __FILE__, __LINE__,
                                    "configuration is null");
    if (c->fw < 1 || c->nx < 2 * c->fw + 5 || c->ny < 2 * c->fw + 5)
        return contract_failure(operation, __FILE__, __LINE__,
                                "grid %dx%d with FW=%d has no FD4 bulk",
                                c->nx, c->ny, c->fw);
    if (c->fdorder != 4 || c->mechanisms != 1)
        return contract_failure(operation, __FILE__, __LINE__,
                                "only FDORDER=4 and L=1 are supported");
    if (c->mpi_ranks_x != 1 || c->mpi_ranks_y != 1)
        return contract_failure(operation, __FILE__, __LINE__,
                                "only a single MPI rank is supported");
    if (c->boundary != 0 || c->free_surface != 0 || c->mode != 0)
        return contract_failure(operation, __FILE__, __LINE__,
                                "requires BOUNDARY=0 FREE_SURF=0 mode=0");
    if (!(c->dt > 0.0f) || !(c->dh > 0.0f))
        return contract_failure(operation, __FILE__, __LINE__,
                                "DT and DH must be positive");
    return 0;
}

int calculate_sizes(const denise_cuda_psv_fd4_l1_config *c,
                    size_t *fnx, size_t *fny, size_t *full,
                    size_t *x, size_t *y, size_t *bytes) {
    if (validate_config(c, "required_bytes") != 0) return -1;
    size_t fw2, elements, part;
    if (!checked_add((size_t)c->nx, 6, fnx) ||
        !checked_add((size_t)c->ny, 6, fny) ||
        !checked_mul(*fnx, *fny, full) ||
        !checked_mul((size_t)c->fw, 2, &fw2) ||
        !checked_mul((size_t)c->ny, fw2, x) ||
        !checked_mul(fw2, (size_t)c->nx, y) ||
        !checked_mul(*full, 16, &elements) ||
        !checked_mul(*x, 4, &part) || !checked_add(elements, part, &elements) ||
        !checked_mul(*y, 4, &part) || !checked_add(elements, part, &elements) ||
        !checked_mul(fw2, 12, &part) || !checked_add(elements, part, &elements) ||
        !checked_mul(elements, sizeof(float), bytes))
        return contract_failure("required_bytes", __FILE__, __LINE__,
                                "mandatory state size overflows size_t");
    return 0;
}

int validate_host(const denise_cuda_psv_fd4_l1_host *h) {
    if (!h) return contract_failure("create", __FILE__, __LINE__,
                                    "host adapter is null");
#define RM(m, row) if (!(h->m) || !(h->m[row])) return contract_failure(       \
    "create", __FILE__, __LINE__, "host field " #m " is null")
#define RT(m, row, col) if (!(h->m) || !(h->m[row]) || !(h->m[row][col]))      \
    return contract_failure("create", __FILE__, __LINE__,                    \
                            "host tensor " #m " is null")
#define RV(m) if (!(h->m)) return contract_failure(                            \
    "create", __FILE__, __LINE__, "host vector " #m " is null")
    RM(vx,-2); RM(vy,-2); RM(sxx,-2); RM(syy,-2); RM(sxy,-2);
    RT(r,-2,-2); RT(p,-2,-2); RT(q,-2,-2);
    RM(rip,-2); RM(rjp,-2); RM(fipjp,-2); RM(f,-2); RM(g,-2);
    RT(dip,-2,-2); RT(d,-2,-2); RT(e,-2,-2);
    RM(psi_sxx_x,1); RM(psi_sxy_x,1); RM(psi_vxx,1); RM(psi_vyx,1);
    RM(psi_syy_y,1); RM(psi_sxy_y,1); RM(psi_vyy,1); RM(psi_vxy,1);
    RV(K_x); RV(a_x); RV(b_x); RV(K_x_half); RV(a_x_half); RV(b_x_half);
    RV(K_y); RV(a_y); RV(b_y); RV(K_y_half); RV(a_y_half); RV(b_y_half);
#undef RM
#undef RT
#undef RV
    return 0;
}

float *take(float **cursor, size_t n) {
    float *result = *cursor;
    *cursor += n;
    return result;
}

void assign_slices(denise_cuda_psv_fd4_l1_impl *c) {
    float *p = c->storage;
    size_t f=c->full_elements, x=c->x_cpml_elements, y=c->y_cpml_elements;
    size_t n=(size_t)(2*c->config.fw);
    device_fields &d=c->fields;
#define T(member,count) d.member=take(&p,(count))
    T(vx,f); T(vy,f); T(sxx,f); T(syy,f); T(sxy,f);
    T(r1,f); T(p1,f); T(q1,f); T(rip,f); T(rjp,f); T(fipjp,f);
    T(f,f); T(g,f); T(dip1,f); T(d1,f); T(e1,f);
    T(psi_sxx_x,x); T(psi_sxy_x,x); T(psi_vxx,x); T(psi_vyx,x);
    T(psi_syy_y,y); T(psi_sxy_y,y); T(psi_vyy,y); T(psi_vxy,y);
    T(K_x,n); T(a_x,n); T(b_x,n); T(K_x_half,n); T(a_x_half,n); T(b_x_half,n);
    T(K_y,n); T(a_y,n); T(b_y,n); T(K_y_half,n); T(a_y_half,n); T(b_y_half,n);
#undef T
}

cudaError_t h2d(float *device, const float *host, size_t n,
                denise_cuda_psv_fd4_l1_stats *s) {
    size_t bytes=n*sizeof(float);
    cudaError_t rc=cudaMemcpy(device,host,bytes,cudaMemcpyHostToDevice);
    if (rc==cudaSuccess) { ++s->h2d_transfer_calls; s->h2d_bytes+=bytes; }
    return rc;
}
cudaError_t d2h(float *host, const float *device, size_t n,
                denise_cuda_psv_fd4_l1_stats *s) {
    size_t bytes=n*sizeof(float);
    cudaError_t rc=cudaMemcpy(host,device,bytes,cudaMemcpyDeviceToHost);
    if (rc==cudaSuccess) { ++s->d2h_transfer_calls; s->d2h_bytes+=bytes; }
    return rc;
}
int upload_initial(denise_cuda_psv_fd4_l1_impl *c,
                   const denise_cuda_psv_fd4_l1_host *h) {
    size_t f=c->full_elements, x=c->x_cpml_elements, y=c->y_cpml_elements;
    size_t n=(size_t)(2*c->config.fw);
    device_fields &d=c->fields;
    cudaEvent_t start=nullptr, stop=nullptr;
    cudaError_t rc=cudaEventCreate(&start);
    if (rc!=cudaSuccess) return cuda_failure("upload event create",rc,__FILE__,__LINE__);
    rc=cudaEventCreate(&stop);
    if (rc!=cudaSuccess) { cudaEventDestroy(start); return cuda_failure("upload event create",rc,__FILE__,__LINE__); }
    rc=cudaEventRecord(start); if(rc!=cudaSuccess) goto fail;
#define UM(dev,host,row,col,count) do { rc=h2d((dev),&(host)[row][col],(count),&c->stats); if(rc!=cudaSuccess) goto fail; } while(0)
#define UT(dev,host,row,col,count) do { rc=h2d((dev),&(host)[row][col][1],(count),&c->stats); if(rc!=cudaSuccess) goto fail; } while(0)
#define UV(dev,host) do { rc=h2d((dev),&(host)[1],n,&c->stats); if(rc!=cudaSuccess) goto fail; } while(0)
    UM(d.vx,h->vx,-2,-2,f); UM(d.vy,h->vy,-2,-2,f);
    UM(d.sxx,h->sxx,-2,-2,f); UM(d.syy,h->syy,-2,-2,f); UM(d.sxy,h->sxy,-2,-2,f);
    UT(d.r1,h->r,-2,-2,f); UT(d.p1,h->p,-2,-2,f); UT(d.q1,h->q,-2,-2,f);
    UM(d.rip,h->rip,-2,-2,f); UM(d.rjp,h->rjp,-2,-2,f);
    UM(d.fipjp,h->fipjp,-2,-2,f); UM(d.f,h->f,-2,-2,f); UM(d.g,h->g,-2,-2,f);
    UT(d.dip1,h->dip,-2,-2,f); UT(d.d1,h->d,-2,-2,f); UT(d.e1,h->e,-2,-2,f);
    UM(d.psi_sxx_x,h->psi_sxx_x,1,1,x); UM(d.psi_sxy_x,h->psi_sxy_x,1,1,x);
    UM(d.psi_vxx,h->psi_vxx,1,1,x); UM(d.psi_vyx,h->psi_vyx,1,1,x);
    UM(d.psi_syy_y,h->psi_syy_y,1,1,y); UM(d.psi_sxy_y,h->psi_sxy_y,1,1,y);
    UM(d.psi_vyy,h->psi_vyy,1,1,y); UM(d.psi_vxy,h->psi_vxy,1,1,y);
    UV(d.K_x,h->K_x); UV(d.a_x,h->a_x); UV(d.b_x,h->b_x);
    UV(d.K_x_half,h->K_x_half); UV(d.a_x_half,h->a_x_half); UV(d.b_x_half,h->b_x_half);
    UV(d.K_y,h->K_y); UV(d.a_y,h->a_y); UV(d.b_y,h->b_y);
    UV(d.K_y_half,h->K_y_half); UV(d.a_y_half,h->a_y_half); UV(d.b_y_half,h->b_y_half);
#undef UM
#undef UT
#undef UV
    rc=cudaEventRecord(stop); if(rc!=cudaSuccess) goto fail;
    rc=cudaEventSynchronize(stop); if(rc!=cudaSuccess) goto fail;
    rc=cudaEventElapsedTime(&c->stats.upload_ms,start,stop); if(rc!=cudaSuccess) goto fail;
    cudaEventDestroy(stop); cudaEventDestroy(start); return 0;
fail:
    cudaEventDestroy(stop); cudaEventDestroy(start);
    return cuda_failure("initial state upload",rc,__FILE__,__LINE__);
}

__device__ __forceinline__ size_t fi(int i,int j,size_t pitch) {
    return (size_t)(j+2)*pitch+(size_t)(i+2);
}
__device__ __forceinline__ size_t xi(int h,int j,int fw) {
    return (size_t)(j-1)*(size_t)(2*fw)+(size_t)(h-1);
}
__device__ __forceinline__ size_t yi(int i,int h,int nx) {
    return (size_t)(h-1)*(size_t)nx+(size_t)(i-1);
}

/* Two regular 2-D kernels retain compact CPML storage. Boundary predicates
 * select the four CPML strips/corners while bulk threads follow the same
 * instruction stream without host/device state movement between kernels. */
__global__ void velocity_fd4(device_fields d,int nx,int ny,int fw,size_t pitch,
                             float dt,float dh,float hc1,float hc2,
                             float *operands,int slot,size_t stride) {
    int i=1+(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int j=1+(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(i>nx||j>ny) return;
#define A(m,jj,ii) d.m[fi((ii),(jj),pitch)]
    float sxx_x=hc1*(A(sxx,j,i+1)-A(sxx,j,i))+hc2*(A(sxx,j,i+2)-A(sxx,j,i-1));
    float sxy_x=hc1*(A(sxy,j,i)-A(sxy,j,i-1))+hc2*(A(sxy,j,i+1)-A(sxy,j,i-2));
    float sxy_y=hc1*(A(sxy,j,i)-A(sxy,j-1,i))+hc2*(A(sxy,j+1,i)-A(sxy,j-2,i));
    float syy_y=hc1*(A(syy,j+1,i)-A(syy,j,i))+hc2*(A(syy,j+2,i)-A(syy,j-1,i));
#define XCPML(hh) do { size_t m=xi((hh),j,fw),c=(size_t)((hh)-1); \
    d.psi_sxx_x[m]=d.b_x_half[c]*d.psi_sxx_x[m]+d.a_x_half[c]*sxx_x; \
    sxx_x=sxx_x/d.K_x_half[c]+d.psi_sxx_x[m]; \
    d.psi_sxy_x[m]=d.b_x[c]*d.psi_sxy_x[m]+d.a_x[c]*sxy_x; \
    sxy_x=sxy_x/d.K_x[c]+d.psi_sxy_x[m]; } while(0)
#define YCPML(hh) do { size_t m=yi(i,(hh),nx),c=(size_t)((hh)-1); \
    d.psi_syy_y[m]=d.b_y_half[c]*d.psi_syy_y[m]+d.a_y_half[c]*syy_y; \
    syy_y=syy_y/d.K_y_half[c]+d.psi_syy_y[m]; \
    d.psi_sxy_y[m]=d.b_y[c]*d.psi_sxy_y[m]+d.a_y[c]*sxy_y; \
    sxy_y=sxy_y/d.K_y[c]+d.psi_sxy_y[m]; } while(0)
    if(i<=fw) XCPML(i);
    if(i>=nx-fw+1) XCPML(i-nx+2*fw);
    if(j<=fw) YCPML(j);
    if(j>=ny-fw+1) YCPML(j-ny+2*fw);
#undef XCPML
#undef YCPML
    size_t q=fi(i,j,pitch);
    float fx=sxx_x+sxy_y,fy=sxy_x+syy_y;
    if(operands) {
        size_t cell=(size_t)(j-1)*(size_t)nx+(size_t)(i-1);
        size_t offset=(size_t)slot*(size_t)nx*(size_t)ny+cell;
        operands[offset]=fx;
        operands[stride+offset]=fy;
    }
    d.vx[q]+=dt*d.rip[q]*fx/dh;
    d.vy[q]+=dt*d.rjp[q]*fy/dh;
#undef A
}

__global__ void stress_fd4_l1(device_fields d,int nx,int ny,int fw,size_t pitch,
                              float dt,float dh,float hc1,float hc2,
                              float bip1,float bjm1,float cip1,float cjm1,
                              float *operands,int slot,size_t stride) {
    int i=1+(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int j=1+(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(i>nx||j>ny) return;
#define A(m,jj,ii) d.m[fi((ii),(jj),pitch)]
    float dhi=1.0f/dh,dth=dt/2.0f;
    float vxx=(hc1*(A(vx,j,i)-A(vx,j,i-1))+hc2*(A(vx,j,i+1)-A(vx,j,i-2)))*dhi;
    float vyx=(hc1*(A(vy,j,i+1)-A(vy,j,i))+hc2*(A(vy,j,i+2)-A(vy,j,i-1)))*dhi;
    float vxy=(hc1*(A(vx,j+1,i)-A(vx,j,i))+hc2*(A(vx,j+2,i)-A(vx,j-1,i)))*dhi;
    float vyy=(hc1*(A(vy,j,i)-A(vy,j-1,i))+hc2*(A(vy,j+1,i)-A(vy,j-2,i)))*dhi;
#define XCPML(hh) do { size_t m=xi((hh),j,fw),c=(size_t)((hh)-1); \
    d.psi_vxx[m]=d.b_x[c]*d.psi_vxx[m]+d.a_x[c]*vxx; \
    vxx=vxx/d.K_x[c]+d.psi_vxx[m]; \
    d.psi_vyx[m]=d.b_x_half[c]*d.psi_vyx[m]+d.a_x_half[c]*vyx; \
    vyx=vyx/d.K_x_half[c]+d.psi_vyx[m]; } while(0)
#define YCPML(hh) do { size_t m=yi(i,(hh),nx),c=(size_t)((hh)-1); \
    d.psi_vyy[m]=d.b_y[c]*d.psi_vyy[m]+d.a_y[c]*vyy; \
    d.psi_vxy[m]=d.b_y_half[c]*d.psi_vxy[m]+d.a_y_half[c]*vxy; \
    vyy=vyy/d.K_y[c]+d.psi_vyy[m]; \
    vxy=vxy/d.K_y_half[c]+d.psi_vxy[m]; } while(0)
    if(i<=fw) XCPML(i);
    if(i>=nx-fw+1) XCPML(i-nx+2*fw);
    if(j<=fw) YCPML(j);
    if(j>=ny-fw+1) YCPML(j-ny+2*fw);
#undef XCPML
#undef YCPML
    size_t q=fi(i,j,pitch);
    if(operands) {
        size_t cell=(size_t)(j-1)*(size_t)nx+(size_t)(i-1);
        size_t offset=(size_t)slot*(size_t)nx*(size_t)ny+cell;
        operands[2*stride+offset]=vxx;
        operands[3*stride+offset]=vyx;
        operands[4*stride+offset]=vxy;
        operands[5*stride+offset]=vyy;
    }
    float oldr=d.r1[q],oldp=d.p1[q],oldq=d.q1[q];
    float sxy_acc=d.sxy[q];
    float sxx_acc=d.sxx[q];
    float syy_acc=d.syy[q];
    sxy_acc+=d.fipjp[q]*(vxy+vyx)+dth*oldr;
    sxx_acc+=d.g[q]*(vxx+vyy)-(2.0*d.f[q]*vyy)+dth*oldp;
    syy_acc+=d.g[q]*(vxx+vyy)-(2.0*d.f[q]*vxx)+dth*oldq;
    float newr=bip1*(oldr*cip1-d.dip1[q]*(vxy+vyx));
    float newp=bjm1*(oldp*cjm1-d.e1[q]*(vxx+vyy)+(2.0f*d.d1[q]*vyy));
    float newq=bjm1*(oldq*cjm1-d.e1[q]*(vxx+vyy)+(2.0f*d.d1[q]*vxx));
    d.r1[q]=newr; d.p1[q]=newp; d.q1[q]=newq;
    sxy_acc+=dth*newr; sxx_acc+=dth*newp; syy_acc+=dth*newq;
    d.sxy[q]=sxy_acc; d.sxx[q]=sxx_acc; d.syy[q]=syy_acc;
#undef A
}

int launch_velocity(denise_cuda_psv_fd4_l1_impl *c,float *operands=nullptr,
                    int slot=0,size_t stride=0) {
    dim3 block(32,4),grid((c->config.nx+31)/32,(c->config.ny+3)/4);
    velocity_fd4<<<grid,block>>>(c->fields,c->config.nx,c->config.ny,c->config.fw,
        c->full_nx,c->config.dt,c->config.dh,c->config.hc1,c->config.hc2,
        operands,slot,stride);
    PSV_CUDA_CALL("velocity kernel launch",cudaGetLastError());
    return 0;
}
int timed_velocity(denise_cuda_psv_fd4_l1_impl *c,cudaEvent_t a,cudaEvent_t b,float *ms) {
    PSV_CUDA_CALL("velocity event start",cudaEventRecord(a));
    if(launch_velocity(c)!=0) return -1;
    PSV_CUDA_CALL("velocity event stop",cudaEventRecord(b));
    PSV_CUDA_CALL("velocity kernel completion",cudaEventSynchronize(b));
    PSV_CUDA_CALL("velocity elapsed",cudaEventElapsedTime(ms,a,b));
    return 0;
}
int launch_stress(denise_cuda_psv_fd4_l1_impl *c,float *operands=nullptr,
                  int slot=0,size_t stride=0) {
    dim3 block(32,4),grid((c->config.nx+31)/32,(c->config.ny+3)/4);
    stress_fd4_l1<<<grid,block>>>(c->fields,c->config.nx,c->config.ny,c->config.fw,
        c->full_nx,c->config.dt,c->config.dh,c->config.hc1,c->config.hc2,
        c->config.bip1,c->config.bjm1,c->config.cip1,c->config.cjm1,
        operands,slot,stride);
    PSV_CUDA_CALL("stress kernel launch",cudaGetLastError());
    return 0;
}
int timed_stress(denise_cuda_psv_fd4_l1_impl *c,cudaEvent_t a,cudaEvent_t b,float *ms) {
    PSV_CUDA_CALL("stress event start",cudaEventRecord(a));
    if(launch_stress(c)!=0) return -1;
    PSV_CUDA_CALL("stress event stop",cudaEventRecord(b));
    PSV_CUDA_CALL("stress kernel completion",cudaEventSynchronize(b));
    PSV_CUDA_CALL("stress elapsed",cudaEventElapsedTime(ms,a,b));
    return 0;
}

__device__ __forceinline__ double adjoint_cpml_x(
        double corrected,double *psi,int h,int j,int fw,
        const float *K,const float *a,const float *b) {
    if(!h) return corrected;
    size_t m=xi(h,j,fw),c=(size_t)(h-1);
    double combined=psi[m]+corrected;
    psi[m]=(double)b[c]*combined;
    return corrected/(double)K[c]+(double)a[c]*combined;
}

__device__ __forceinline__ double adjoint_cpml_y(
        double corrected,double *psi,int h,int i,int nx,
        const float *K,const float *a,const float *b) {
    if(!h) return corrected;
    size_t m=yi(i,h,nx),c=(size_t)(h-1);
    double combined=psi[m]+corrected;
    psi[m]=(double)b[c]*combined;
    return corrected/(double)K[c]+(double)a[c]*combined;
}

__global__ void adjoint_inject_receivers(
        adjoint_fields a,size_t pitch,const int *receiver_i,
        const int *receiver_j,const float *residual,int ntr,int timestep) {
    int r=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(r>=ntr||timestep<=1) return;
    size_t p=fi(receiver_i[r],receiver_j[r],pitch);
    a.avx[p]+=(double)residual[r]-(double)residual[ntr+r];
    a.avy[p]+=(double)residual[2*ntr+r]-(double)residual[3*ntr+r];
}

__global__ void adjoint_inject_resident_traces(
        adjoint_fields a,size_t pitch,const int *receiver_i,
        const int *receiver_j,const float *modeled_vx,const float *modeled_vy,
        const float *observed_vx,const float *observed_vy,
        int ntr,int nt,int timestep) {
    int r=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(r>=ntr||timestep<=1) return;
    size_t p=fi(receiver_i[r],receiver_j[r],pitch);
    size_t sample=(size_t)r*(size_t)nt+(size_t)(timestep-1);
    a.avx[p]+=(double)modeled_vx[sample]-(double)observed_vx[sample];
    a.avy[p]+=(double)modeled_vy[sample]-(double)observed_vy[sample];
}

__global__ void adjoint_inject_resident_production_residual(
        adjoint_fields a,size_t pitch,const int *receiver_i,
        const int *receiver_j,const float *residual_vx,
        const float *residual_vy,int ntr,int nt,int timestep) {
    int r=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(r>=ntr||timestep<=1) return;
    size_t p=fi(receiver_i[r],receiver_j[r],pitch);
    size_t sample=(size_t)r*(size_t)nt+(size_t)(timestep-1);
    a.avx[p]+=(double)residual_vx[sample];
    a.avy[p]+=(double)residual_vy[sample];
}

/* Phase B: receiver residuals are already injected, while stress and memory
 * cotangents are still the incoming state for this reverse timestep. */
__global__ void native_gradient_stress_correlation(
        adjoint_fields a,native_gradient_fields g,const float *operands,
        size_t operand_stride,size_t slot,int nx,int ny,size_t pitch,
        float dt,float bjm1) {
    int i=1+(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int j=1+(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(i>nx||j>ny) return;
    size_t q=(size_t)(j-1)*(size_t)nx+(size_t)(i-1);
    size_t p=fi(i,j,pitch);
    size_t o=slot*(size_t)nx*(size_t)ny+q;
    double xx=(double)operands[2*operand_stride+o];
    double yx=(double)operands[3*operand_stride+o];
    double xy=(double)operands[4*operand_stride+o];
    double yy=(double)operands[5*operand_stride+o];
    double div=xx+yy,shear=xy+yx;
    double dt2=(double)dt*0.5,b=(double)bjm1;
    double lambda_r=a.ar[p]+dt2*a.asxy[p];
    double lambda_p=a.ap[p]+dt2*a.asxx[p];
    double lambda_q=a.aq[p]+dt2*a.asyy[p];
    g.gfc[q]+=a.asxy[p]*shear;
    g.gf[q]+=-2.0*(a.asxx[p]*yy+a.asyy[p]*xx);
    g.gg[q]+=(a.asxx[p]+a.asyy[p])*div;
    g.gdc[q]+=-b*lambda_r*shear;
    g.gd[q]+=2.0*b*(lambda_p*yy+lambda_q*xx);
    g.ge[q]+=-b*(lambda_p+lambda_q)*div;
}

/* Phase D: stress transpose and gather have updated avx/avy; the velocity
 * transpose has not yet mutated the reverse state. */
__global__ void native_gradient_velocity_correlation(
        adjoint_fields a,native_gradient_fields g,const float *operands,
        size_t operand_stride,size_t slot,int nx,int ny,size_t pitch,
        float dt,float dh) {
    int i=1+(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int j=1+(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(i>nx||j>ny) return;
    size_t q=(size_t)(j-1)*(size_t)nx+(size_t)(i-1);
    size_t p=fi(i,j,pitch);
    size_t o=slot*(size_t)nx*(size_t)ny+q;
    g.grx[q]+=a.avx[p]*(double)dt*(double)operands[o]/(double)dh;
    g.gry[q]+=a.avy[p]*(double)dt*
              (double)operands[operand_stride+o]/(double)dh;
}

__global__ void adjoint_stress_local(
        device_fields d,adjoint_fields a,int nx,int ny,int fw,size_t pitch,
        float dt,float bjm1,float cjm1) {
    int i=1+(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int j=1+(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(i>nx||j>ny) return;
    size_t p=fi(i,j,pitch);
    double dt2=(double)dt*0.5,b=(double)bjm1,c=(double)cjm1;
    double lam_r=a.ar[p]+dt2*a.asxy[p];
    double lam_p=a.ap[p]+dt2*a.asxx[p];
    double lam_q=a.aq[p]+dt2*a.asyy[p];
    double f=(double)d.f[p],g=(double)d.g[p];
    double ax=a.asxx[p]*g+a.asyy[p]*(g-2.0*f);
    double ay=a.asyy[p]*g+a.asxx[p]*(g-2.0*f);
    ax+=b*(-(double)d.e1[p]*(lam_p+lam_q)+2.0*(double)d.d1[p]*lam_q);
    ay+=b*(-(double)d.e1[p]*(lam_p+lam_q)+2.0*(double)d.d1[p]*lam_p);
    double shear=a.asxy[p]*(double)d.fipjp[p]-b*lam_r*(double)d.dip1[p];
    a.ar[p]=dt2*a.asxy[p]+b*c*lam_r;
    a.ap[p]=dt2*a.asxx[p]+b*c*lam_p;
    a.aq[p]=dt2*a.asyy[p]+b*c*lam_q;
    int hx=i<=fw?i:(i>=nx-fw+1?i-nx+2*fw:0);
    int hy=j<=fw?j:(j>=ny-fw+1?j-ny+2*fw:0);
    a.wxx[p]=adjoint_cpml_x(ax,a.pvxx,hx,j,fw,d.K_x,d.a_x,d.b_x);
    a.wyx[p]=adjoint_cpml_x(shear,a.pvyx,hx,j,fw,
                            d.K_x_half,d.a_x_half,d.b_x_half);
    a.wxy[p]=adjoint_cpml_y(shear,a.pvxy,hy,i,nx,
                            d.K_y_half,d.a_y_half,d.b_y_half);
    a.wyy[p]=adjoint_cpml_y(ay,a.pvyy,hy,i,nx,d.K_y,d.a_y,d.b_y);
}

__device__ __forceinline__ double adjoint_workspace(
        const double *w,int i,int j,int nx,int ny,size_t pitch) {
    return (i>=1&&i<=nx&&j>=1&&j<=ny)?w[fi(i,j,pitch)]:0.0;
}

__global__ void adjoint_stress_gather(
        adjoint_fields a,int nx,int ny,size_t pitch,float dh,float hc1,float hc2) {
    size_t q=(size_t)(blockIdx.x*blockDim.x+threadIdx.x);
    size_t full=pitch*(size_t)(ny+6);
    if(q>=full) return;
    int j=(int)(q/pitch)-2,i=(int)(q%pitch)-2;
#define W(m,ii,jj) adjoint_workspace(a.m,(ii),(jj),nx,ny,pitch)
    double h1=(double)hc1,h2=(double)hc2;
    double x=h1*(W(wxx,i,j)-W(wxx,i+1,j))+
             h2*(W(wxx,i-1,j)-W(wxx,i+2,j));
    double xy=h1*(W(wxy,i,j-1)-W(wxy,i,j))+
              h2*(W(wxy,i,j-2)-W(wxy,i,j+1));
    double yx=h1*(W(wyx,i-1,j)-W(wyx,i,j))+
              h2*(W(wyx,i-2,j)-W(wyx,i+1,j));
    double y=h1*(W(wyy,i,j)-W(wyy,i,j+1))+
             h2*(W(wyy,i,j-1)-W(wyy,i,j+2));
    a.avx[q]+=(x+xy)/(double)dh;
    a.avy[q]+=(yx+y)/(double)dh;
#undef W
}

__global__ void adjoint_velocity_local(
        device_fields d,adjoint_fields a,int nx,int ny,int fw,size_t pitch,
        float dt,float dh) {
    int i=1+(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int j=1+(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(i>nx||j>ny) return;
    size_t p=fi(i,j,pitch);
    double ax=a.avx[p]*(double)dt*(double)d.rip[p]/(double)dh;
    double ay=a.avy[p]*(double)dt*(double)d.rjp[p]/(double)dh;
    int hx=i<=fw?i:(i>=nx-fw+1?i-nx+2*fw:0);
    int hy=j<=fw?j:(j>=ny-fw+1?j-ny+2*fw:0);
    a.wxx[p]=adjoint_cpml_x(ax,a.psxx,hx,j,fw,
                            d.K_x_half,d.a_x_half,d.b_x_half);
    a.wyx[p]=adjoint_cpml_x(ay,a.psxyx,hx,j,fw,d.K_x,d.a_x,d.b_x);
    a.wxy[p]=adjoint_cpml_y(ax,a.psxyy,hy,i,nx,d.K_y,d.a_y,d.b_y);
    a.wyy[p]=adjoint_cpml_y(ay,a.psyy,hy,i,nx,
                            d.K_y_half,d.a_y_half,d.b_y_half);
}

__global__ void adjoint_velocity_gather(
        adjoint_fields a,int nx,int ny,size_t pitch,float hc1,float hc2) {
    size_t q=(size_t)(blockIdx.x*blockDim.x+threadIdx.x);
    size_t full=pitch*(size_t)(ny+6);
    if(q>=full) return;
    int j=(int)(q/pitch)-2,i=(int)(q%pitch)-2;
#define W(m,ii,jj) adjoint_workspace(a.m,(ii),(jj),nx,ny,pitch)
    double h1=(double)hc1,h2=(double)hc2;
    a.asxx[q]+=h1*(W(wxx,i-1,j)-W(wxx,i,j))+
               h2*(W(wxx,i-2,j)-W(wxx,i+1,j));
    a.asxy[q]+=h1*(W(wyx,i,j)-W(wyx,i+1,j))+
               h2*(W(wyx,i-1,j)-W(wyx,i+2,j))+
               h1*(W(wxy,i,j)-W(wxy,i,j+1))+
               h2*(W(wxy,i,j-1)-W(wxy,i,j+2));
    a.asyy[q]+=h1*(W(wyy,i,j-1)-W(wyy,i,j))+
               h2*(W(wyy,i,j-2)-W(wyy,i,j+1));
#undef W
}

}  // namespace

struct denise_cuda_psv_fd4_l1 { denise_cuda_psv_fd4_l1_impl impl; };

struct denise_cuda_psv_forward {
    denise_cuda_psv_fd4_l1 *core;
    denise_cuda_psv_forward_config config;
    int *source_xy;
    float *source_signal;
    int *receiver_i;
    int *receiver_j;
    float *trace_vx;
    float *trace_vy;
    float *checkpoint_storage;
    denise_cuda_psv_forward_config checkpoint_config;
    int checkpoint_completed_timestep;
    int next_timestep;
    bool checkpoint_valid;
    float *segment_storage;
    float *segment_operands;
    denise_cuda_psv_forward_config segment_config;
    int segment_count;
    int segment_captured;
    int segment_record_armed;
    int segment_record_ready;
    bool segment_original_complete;
    double *adjoint_storage;
    float *adjoint_residual;
    adjoint_fields adjoint;
    denise_cuda_psv_adjoint_stats adjoint_stats;
    float *adjoint_observed_storage;
    float *adjoint_observed_vx;
    float *adjoint_observed_vy;
    int adjoint_resident_source_kind;
    denise_cuda_psv_adjoint_sweep_stats adjoint_sweep_stats;
    int adjoint_sweep_state;
    double *native_gradient_storage;
    native_gradient_fields native_gradient;
    denise_cuda_psv_native_gradient_stats native_gradient_stats;
    unsigned char *physical_gradient_storage;
    physical_material_fields physical_material;
    physical_gradient_fields physical_gradient;
    int *physical_invalid_device;
    float physical_eta;
    int physical_q_mode;
    double physical_q_per;
    double physical_q_offset;
    denise_cuda_psv_physical_gradient_stats physical_gradient_stats;
    bool receivers_unique;
    bool profiling_enabled;
    denise_cuda_psv_forward_stats stats;
};

namespace {

enum {
    ADJOINT_SWEEP_UNPREPARED=0,
    ADJOINT_SWEEP_READY=1,
    ADJOINT_SWEEP_COMPLETE=2,
    ADJOINT_SWEEP_INVALID=3
};

enum {
    ADJOINT_RESIDENT_SOURCE_NONE=0,
    ADJOINT_RESIDENT_SOURCE_OBSERVED=1,
    ADJOINT_RESIDENT_SOURCE_PRODUCTION_RESIDUAL=2
};

int calculate_adjoint_sweep_plan(
        const denise_cuda_psv_forward_config *config,
        denise_cuda_psv_adjoint_sweep_stats *plan) {
    if(!config||!plan)
        return contract_failure("adjoint sweep required bytes",__FILE__,__LINE__,
                                "configuration or output is null");
    if(validate_config(&config->core,"adjoint sweep required bytes")!=0)
        return -1;
    if(config->nt<1||config->ntr<1)
        return contract_failure("adjoint sweep required bytes",__FILE__,__LINE__,
                                "NT and NTR must be positive");
    size_t samples,component_bytes,total;
    if(!checked_mul((size_t)config->ntr,(size_t)config->nt,&samples)||
       !checked_mul(samples,sizeof(float),&component_bytes)||
       !checked_mul(component_bytes,2,&total))
        return contract_failure("adjoint sweep required bytes",__FILE__,__LINE__,
                                "observed trace size overflows size_t");
    std::memset(plan,0,sizeof(*plan));
    plan->observed_trace_bytes=total;
    plan->total_incremental_bytes=total;
    plan->next_reverse_segment=-1;
    return 0;
}

int calculate_native_gradient_plan(
        const denise_cuda_psv_forward_config *config,
        denise_cuda_psv_native_gradient_stats *plan) {
    if(!config||!plan)
        return contract_failure("native gradient required bytes",__FILE__,__LINE__,
                                "configuration or output is null");
    if(validate_config(&config->core,"native gradient required bytes")!=0)
        return -1;
    size_t cells,fields;
    if(!checked_mul((size_t)config->core.nx,(size_t)config->core.ny,&cells)||
       !checked_mul(cells,8,&fields)||
       !checked_mul(fields,sizeof(double),&fields))
        return contract_failure("native gradient required bytes",__FILE__,__LINE__,
                                "native gradient size overflows size_t");
    std::memset(plan,0,sizeof(*plan));
    plan->native_gradient_bytes=fields;
    return 0;
}

int calculate_physical_gradient_plan(
        const denise_cuda_psv_forward_config *config,
        denise_cuda_psv_physical_gradient_stats *plan) {
    if(!config||!plan)
        return contract_failure("physical gradient required bytes",__FILE__,__LINE__,
                                "configuration or output is null");
    if(validate_config(&config->core,"physical gradient required bytes")!=0)
        return -1;
    size_t cells,material_elements,gradient_elements;
    size_t material_bytes,gradient_bytes,total;
    if(!checked_mul((size_t)config->core.nx,(size_t)config->core.ny,&cells)||
       !checked_mul(cells,7,&material_elements)||
       !checked_mul(material_elements,sizeof(float),&material_bytes)||
       !checked_mul(cells,5,&gradient_elements)||
       !checked_mul(gradient_elements,sizeof(double),&gradient_bytes))
        return contract_failure("physical gradient required bytes",__FILE__,__LINE__,
                                "physical mapping size overflows size_t");
    size_t alignment=(sizeof(double)-material_bytes%sizeof(double))%sizeof(double);
    if(!checked_add(material_bytes,alignment,&total)||
       !checked_add(total,gradient_bytes,&total)||
       !checked_add(total,sizeof(int),&total))
        return contract_failure("physical gradient required bytes",__FILE__,__LINE__,
                                "physical mapping total overflows size_t");
    std::memset(plan,0,sizeof(*plan));
    plan->material_bytes=material_bytes;
    plan->physical_gradient_bytes=gradient_bytes;
    plan->alignment_bytes=alignment;
    plan->validation_bytes=sizeof(int);
    plan->total_bytes=total;
    return 0;
}

void assign_native_gradient_slices(denise_cuda_psv_forward *f) {
    size_t cells=(size_t)f->config.core.nx*(size_t)f->config.core.ny;
    double *p=f->native_gradient_storage;
#define G(member) f->native_gradient.member=p; p+=cells
    G(gf); G(gg); G(gfc); G(gd); G(ge); G(gdc); G(grx); G(gry);
#undef G
}

bool valid_native_gradient_host(const denise_cuda_psv_native_gradient_host *h) {
    return h&&h->gf&&h->gg&&h->gfc&&h->gd&&h->ge&&h->gdc&&h->grx&&h->gry;
}

void assign_physical_gradient_slices(
        denise_cuda_psv_forward *f,
        const denise_cuda_psv_physical_gradient_stats &plan) {
    size_t cells=(size_t)f->config.core.nx*(size_t)f->config.core.ny;
    float *m=(float*)f->physical_gradient_storage;
#define M(member) f->physical_material.member=m; m+=cells
    M(prho); M(ppi); M(pu); M(ptaus); M(ptaup); M(puipjp); M(ptausipjp);
#undef M
    unsigned char *gradient_begin=f->physical_gradient_storage+
        plan.material_bytes+plan.alignment_bytes;
    double *g=(double*)gradient_begin;
#define P(member) f->physical_gradient.member=g; g+=cells
    P(vp); P(vs); P(rho); P(qp); P(qs);
#undef P
    f->physical_invalid_device=(int*)g;
}

bool valid_physical_material_host(
        const denise_cuda_psv_physical_material_host *h) {
    return h&&h->prho&&h->ppi&&h->pu&&h->ptaus&&h->ptaup&&
        h->puipjp&&h->ptausipjp&&h->prho[1]&&h->ppi[1]&&h->pu[1]&&
        h->ptaus[1]&&h->ptaup[1]&&h->puipjp[1]&&h->ptausipjp[1];
}

bool valid_physical_gradient_host(
        const denise_cuda_psv_physical_gradient_host *h) {
    return h&&h->vp&&h->vs&&h->rho&&h->qp&&h->qs;
}

__device__ double physical_q_derivative(double tau,int mode,
        double inverse_tau_per_q,double inverse_tau_offset,int *invalid) {
    double q=mode==0?2.0/tau:
        (1.0/tau-inverse_tau_offset)/inverse_tau_per_q;
    float qf=(float)q; /* Match q_to_tau_derivative(float, mapping). */
    if(!(qf>0.0f)||!isfinite(qf)) {
        atomicExch(invalid,1);
        return 0.0;
    }
    if(mode==0) return -2.0/((double)qf*(double)qf);
    double inverse_tau=inverse_tau_per_q*(double)qf+inverse_tau_offset;
    if(!(inverse_tau>0.0)||!isfinite(inverse_tau)) {
        atomicExch(invalid,1);
        return 0.0;
    }
    double reconstructed_tau=1.0/inverse_tau;
    return -inverse_tau_per_q*reconstructed_tau*reconstructed_tau;
}

/* One thread owns one output cell. Gathering incoming corner and face terms
 * removes atomics while preserving the CPU operator's global-boundary clips. */
__global__ void physical_gradient_map_kernel(
        native_gradient_fields n,physical_material_fields m,
        physical_gradient_fields out,device_fields resident,
        int nx,int ny,size_t pitch,float dt,float eta,int q_mode,
        double inverse_tau_per_q,double inverse_tau_offset,int *invalid) {
    int i=1+(int)(blockIdx.x*blockDim.x+threadIdx.x);
    int j=1+(int)(blockIdx.y*blockDim.y+threadIdx.y);
    if(i>nx||j>ny) return;
    size_t q=(size_t)(j-1)*(size_t)nx+(size_t)(i-1);
    double rho=(double)m.prho[q],vp=(double)m.ppi[q],vs=(double)m.pu[q];
    double ts=(double)m.ptaus[q],tp=(double)m.ptaup[q];
    double modulus_s=rho*vs*vs,modulus_p=rho*vp*vp;
    double den_s=1.0+0.5*ts,den_p=1.0+0.5*tp;
    double gM=n.gf[q]*(double)dt*(1.0+ts)/den_s+
        n.gd[q]*(double)eta*ts/den_s;
    double gP=n.gg[q]*(double)dt*(1.0+tp)/den_p+
        n.ge[q]*(double)eta*tp/den_p;
    double gts=n.gf[q]*(double)dt*modulus_s*0.5/(den_s*den_s)+
        n.gd[q]*(double)eta*modulus_s/(den_s*den_s);
    double gtp=n.gg[q]*(double)dt*modulus_p*0.5/(den_p*den_p)+
        n.ge[q]*(double)eta*modulus_p/(den_p*den_p);
    double vp_gradient=gP*2.0*rho*vp;
    double vs_gradient=gM*2.0*rho*vs;
    double rho_gradient=gP*vp*vp+gM*vs*vs;
    double qp_tau_gradient=gtp,qs_tau_gradient=gts;

    for(int sj=j-1;sj<=j;++sj) for(int si=i-1;si<=i;++si) {
        if(si<1||sj<1) continue;
        size_t source=(size_t)(sj-1)*(size_t)nx+(size_t)(si-1);
        double H=(double)m.puipjp[source],T=(double)m.ptausipjp[source];
        double den_c=1.0+0.5*T;
        double gH=n.gfc[source]*(double)dt*(1.0+T)/den_c+
            n.gdc[source]*(double)eta*T/den_c;
        double gT=n.gfc[source]*(double)dt*H*0.5/(den_c*den_c)+
            n.gdc[source]*(double)eta*H/(den_c*den_c);
        double weight=gH*H*H/(4.0*modulus_s*modulus_s);
        vs_gradient+=weight*2.0*rho*vs;
        rho_gradient+=weight*vs*vs;
        qs_tau_gradient+=0.25*gT;
    }

    size_t face=fi(i,j,pitch);
    double rx=(double)resident.rip[face];
    double ry=(double)resident.rjp[face];
    rho_gradient+=-0.5*rx*rx*n.grx[q];
    rho_gradient+=-0.5*ry*ry*n.gry[q];
    if(i>1) {
        size_t left=q-1;
        double value=(double)resident.rip[fi(i-1,j,pitch)];
        rho_gradient+=-0.5*value*value*n.grx[left];
    }
    if(j>1) {
        size_t above=q-(size_t)nx;
        double value=(double)resident.rjp[fi(i,j-1,pitch)];
        rho_gradient+=-0.5*value*value*n.gry[above];
    }

    out.vp[q]=vp_gradient;
    out.vs[q]=vs_gradient;
    out.rho[q]=rho_gradient;
    out.qp[q]=qp_tau_gradient*physical_q_derivative(
        tp,q_mode,inverse_tau_per_q,inverse_tau_offset,invalid);
    out.qs[q]=qs_tau_gradient*physical_q_derivative(
        ts,q_mode,inverse_tau_per_q,inverse_tau_offset,invalid);
}

int launch_adjoint_operator(denise_cuda_psv_forward *f,int timestep,
                            bool resident_traces,int segment_begin) {
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    cudaError_t rc=cudaSuccess;
    dim3 block(32,4),grid((c.config.nx+31)/32,(c.config.ny+3)/4);
    size_t full=c.full_elements;
    int linear_block=256;
    int linear_grid=(int)((full+(size_t)linear_block-1)/(size_t)linear_block);
    int receiver_grid=(f->config.ntr+127)/128;
    if(resident_traces&&
       f->adjoint_resident_source_kind==ADJOINT_RESIDENT_SOURCE_OBSERVED)
        adjoint_inject_resident_traces<<<receiver_grid,128>>>(
            f->adjoint,c.full_nx,f->receiver_i,f->receiver_j,
            f->trace_vx,f->trace_vy,f->adjoint_observed_vx,
            f->adjoint_observed_vy,f->config.ntr,f->config.nt,timestep);
    else if(resident_traces&&
            f->adjoint_resident_source_kind==
                ADJOINT_RESIDENT_SOURCE_PRODUCTION_RESIDUAL)
        adjoint_inject_resident_production_residual<<<receiver_grid,128>>>(
            f->adjoint,c.full_nx,f->receiver_i,f->receiver_j,
            f->adjoint_observed_vx,f->adjoint_observed_vy,
            f->config.ntr,f->config.nt,timestep);
    else
        adjoint_inject_receivers<<<receiver_grid,128>>>(f->adjoint,c.full_nx,
            f->receiver_i,f->receiver_j,f->adjoint_residual,
            f->config.ntr,timestep);
    rc=cudaGetLastError(); if(rc!=cudaSuccess) goto fail;
    if(f->native_gradient_storage) {
        size_t stride=(size_t)f->stats.max_segment_length*
                      (size_t)c.config.nx*(size_t)c.config.ny;
        size_t slot=(size_t)(timestep-segment_begin);
        native_gradient_stress_correlation<<<grid,block>>>(
            f->adjoint,f->native_gradient,f->segment_operands,stride,slot,
            c.config.nx,c.config.ny,c.full_nx,c.config.dt,c.config.bjm1);
        rc=cudaGetLastError(); if(rc!=cudaSuccess) goto fail;
    }
    adjoint_stress_local<<<grid,block>>>(c.fields,f->adjoint,c.config.nx,
        c.config.ny,c.config.fw,c.full_nx,c.config.dt,c.config.bjm1,
        c.config.cjm1);
    rc=cudaGetLastError(); if(rc!=cudaSuccess) goto fail;
    adjoint_stress_gather<<<linear_grid,linear_block>>>(f->adjoint,c.config.nx,
        c.config.ny,c.full_nx,c.config.dh,c.config.hc1,c.config.hc2);
    rc=cudaGetLastError(); if(rc!=cudaSuccess) goto fail;
    if(f->native_gradient_storage) {
        size_t stride=(size_t)f->stats.max_segment_length*
                      (size_t)c.config.nx*(size_t)c.config.ny;
        size_t slot=(size_t)(timestep-segment_begin);
        native_gradient_velocity_correlation<<<grid,block>>>(
            f->adjoint,f->native_gradient,f->segment_operands,stride,slot,
            c.config.nx,c.config.ny,c.full_nx,c.config.dt,c.config.dh);
        rc=cudaGetLastError(); if(rc!=cudaSuccess) goto fail;
    }
    adjoint_velocity_local<<<grid,block>>>(c.fields,f->adjoint,c.config.nx,
        c.config.ny,c.config.fw,c.full_nx,c.config.dt,c.config.dh);
    rc=cudaGetLastError(); if(rc!=cudaSuccess) goto fail;
    adjoint_velocity_gather<<<linear_grid,linear_block>>>(f->adjoint,c.config.nx,
        c.config.ny,c.full_nx,c.config.hc1,c.config.hc2);
    rc=cudaGetLastError(); if(rc!=cudaSuccess) goto fail;
    ++f->adjoint_stats.steps;
    f->adjoint_stats.kernel_launches+=f->native_gradient_storage?7:5;
    if(f->native_gradient_storage) {
        ++f->native_gradient_stats.accumulated_timesteps;
        f->native_gradient_stats.correlation_kernel_launches+=2;
    }
    return 0;
fail:
    return cuda_failure("adjoint kernel launch",rc,__FILE__,__LINE__);
}

}  // namespace

extern "C" {
const char *denise_cuda_psv_fd4_l1_last_error(void) { return psv_last_error; }

int denise_cuda_psv_fd4_l1_required_bytes(
        const denise_cuda_psv_fd4_l1_config *c,size_t *bytes) {
    psv_last_error[0]='\0';
    if(!bytes) return contract_failure("required_bytes",__FILE__,__LINE__,"output is null");
    size_t fnx,fny,full,x,y;
    return calculate_sizes(c,&fnx,&fny,&full,&x,&y,bytes);
}

int denise_cuda_psv_fd4_l1_create(const denise_cuda_psv_fd4_l1_config *cfg,
        const denise_cuda_psv_fd4_l1_host *initial,
        denise_cuda_psv_fd4_l1 **output) {
    psv_last_error[0]='\0';
    if(!output) return contract_failure("create",__FILE__,__LINE__,"context output is null");
    *output=nullptr;
    if(validate_config(cfg,"create")!=0||validate_host(initial)!=0) return -1;
    size_t bytes,fnx,fny,full,x,y;
    if(calculate_sizes(cfg,&fnx,&fny,&full,&x,&y,&bytes)!=0) return -1;
    denise_cuda_device_info info;
    if(denise_cuda_select_device(cfg->logical_device,cfg->safety_reserve_bytes,
                                 cfg->user_cap_bytes,&info)!=0)
        return contract_failure("create device selection",__FILE__,__LINE__,
                                "%s",denise_cuda_last_error());
    if(bytes>info.usable_budget_bytes)
        return contract_failure("create aggregate VRAM budget",__FILE__,__LINE__,
            "mandatory state %zu exceeds usable budget %zu",bytes,info.usable_budget_bytes);
    denise_cuda_psv_fd4_l1 *ctx=new(std::nothrow) denise_cuda_psv_fd4_l1();
    if(!ctx) return contract_failure("create",__FILE__,__LINE__,"host allocation failed");
    std::memset(ctx,0,sizeof(*ctx));
    denise_cuda_psv_fd4_l1_impl &c=ctx->impl;
    c.config=*cfg; c.full_nx=fnx; c.full_ny=fny; c.full_elements=full;
    c.x_cpml_elements=x; c.y_cpml_elements=y;
    c.stats.mandatory_state_bytes=bytes;
    c.stats.usable_budget_bytes=info.usable_budget_bytes;
    c.stats.remaining_budget_bytes=info.usable_budget_bytes-bytes;
    cudaError_t rc=cudaMalloc((void**)&c.storage,bytes);
    if(rc!=cudaSuccess) { delete ctx; return cuda_failure("aggregate state cudaMalloc",rc,__FILE__,__LINE__); }
    assign_slices(&c);
    if(upload_initial(&c,initial)!=0) { cudaFree(c.storage); delete ctx; return -1; }
    *output=ctx;
    return 0;
}

int denise_cuda_psv_fd4_l1_step(denise_cuda_psv_fd4_l1 *ctx,int count) {
    psv_last_error[0]='\0';
    if(!ctx||count<1) return contract_failure("step",__FILE__,__LINE__,
                                             "context is null or count is invalid");
    cudaEvent_t a=nullptr,b=nullptr;
    PSV_CUDA_CALL("kernel event create",cudaEventCreate(&a));
    cudaError_t rc=cudaEventCreate(&b);
    if(rc!=cudaSuccess) { cudaEventDestroy(a); return cuda_failure("kernel event create",rc,__FILE__,__LINE__); }
    for(int k=0;k<count;++k) {
        float vm=0.0f,sm=0.0f;
        if(timed_velocity(&ctx->impl,a,b,&vm)!=0||timed_stress(&ctx->impl,a,b,&sm)!=0) {
            cudaEventDestroy(b); cudaEventDestroy(a); return -1;
        }
        ctx->impl.stats.velocity_kernel_ms+=vm;
        ctx->impl.stats.stress_kernel_ms+=sm;
        ctx->impl.stats.combined_kernel_ms+=vm+sm;
        ++ctx->impl.stats.timesteps;
    }
    cudaEventDestroy(b); cudaEventDestroy(a); return 0;
}

int denise_cuda_psv_fd4_l1_download(denise_cuda_psv_fd4_l1 *ctx,
                                    denise_cuda_psv_fd4_l1_host *h) {
    psv_last_error[0]='\0';
    if(!ctx||!h) return contract_failure("download",__FILE__,__LINE__,
                                         "context or output is null");
    denise_cuda_psv_fd4_l1_impl &c=ctx->impl;
    size_t f=c.full_elements,x=c.x_cpml_elements,y=c.y_cpml_elements;
    device_fields &d=c.fields;
    cudaEvent_t a=nullptr,b=nullptr;
    cudaError_t rc=cudaEventCreate(&a);
    if(rc!=cudaSuccess) return cuda_failure("download event create",rc,__FILE__,__LINE__);
    rc=cudaEventCreate(&b);
    if(rc!=cudaSuccess) { cudaEventDestroy(a); return cuda_failure("download event create",rc,__FILE__,__LINE__); }
    rc=cudaEventRecord(a); if(rc!=cudaSuccess) goto fail;
#define DM(host,dev,row,col,count) do { rc=d2h(&(host)[row][col],(dev),(count),&c.stats); if(rc!=cudaSuccess) goto fail; } while(0)
#define DT(host,dev,row,col,count) do { rc=d2h(&(host)[row][col][1],(dev),(count),&c.stats); if(rc!=cudaSuccess) goto fail; } while(0)
    DM(h->vx,d.vx,-2,-2,f); DM(h->vy,d.vy,-2,-2,f);
    DM(h->sxx,d.sxx,-2,-2,f); DM(h->syy,d.syy,-2,-2,f); DM(h->sxy,d.sxy,-2,-2,f);
    DT(h->r,d.r1,-2,-2,f); DT(h->p,d.p1,-2,-2,f); DT(h->q,d.q1,-2,-2,f);
    DM(h->rip,d.rip,-2,-2,f); DM(h->rjp,d.rjp,-2,-2,f);
    DM(h->fipjp,d.fipjp,-2,-2,f); DM(h->f,d.f,-2,-2,f); DM(h->g,d.g,-2,-2,f);
    DT(h->dip,d.dip1,-2,-2,f); DT(h->d,d.d1,-2,-2,f); DT(h->e,d.e1,-2,-2,f);
    DM(h->psi_sxx_x,d.psi_sxx_x,1,1,x); DM(h->psi_sxy_x,d.psi_sxy_x,1,1,x);
    DM(h->psi_vxx,d.psi_vxx,1,1,x); DM(h->psi_vyx,d.psi_vyx,1,1,x);
    DM(h->psi_syy_y,d.psi_syy_y,1,1,y); DM(h->psi_sxy_y,d.psi_sxy_y,1,1,y);
    DM(h->psi_vyy,d.psi_vyy,1,1,y); DM(h->psi_vxy,d.psi_vxy,1,1,y);
    { size_t n=(size_t)(2*c.config.fw);
#define DV(host,dev) do { rc=d2h(&(host)[1],(dev),n,&c.stats); if(rc!=cudaSuccess) goto fail; } while(0)
        DV(h->K_x,d.K_x); DV(h->a_x,d.a_x); DV(h->b_x,d.b_x);
        DV(h->K_x_half,d.K_x_half); DV(h->a_x_half,d.a_x_half); DV(h->b_x_half,d.b_x_half);
        DV(h->K_y,d.K_y); DV(h->a_y,d.a_y); DV(h->b_y,d.b_y);
        DV(h->K_y_half,d.K_y_half); DV(h->a_y_half,d.a_y_half); DV(h->b_y_half,d.b_y_half);
#undef DV
    }
#undef DM
#undef DT
    rc=cudaEventRecord(b); if(rc!=cudaSuccess) goto fail;
    rc=cudaEventSynchronize(b); if(rc!=cudaSuccess) goto fail;
    { float ms=0.0f; rc=cudaEventElapsedTime(&ms,a,b); if(rc!=cudaSuccess) goto fail; c.stats.download_ms+=ms; }
    cudaEventDestroy(b); cudaEventDestroy(a); return 0;
fail:
    cudaEventDestroy(b); cudaEventDestroy(a);
    return cuda_failure("state download",rc,__FILE__,__LINE__);
}

int denise_cuda_psv_fd4_l1_download_mutable(
        denise_cuda_psv_fd4_l1 *ctx,
        denise_cuda_psv_fd4_l1_host *h) {
    psv_last_error[0]='\0';
    if(!ctx||!h) return contract_failure("mutable download",__FILE__,__LINE__,
                                         "context or output is null");
    if(!h->vx||!h->vy||!h->sxx||!h->syy||!h->sxy||
       !h->r||!h->p||!h->q||
       !h->psi_sxx_x||!h->psi_sxy_x||!h->psi_syy_y||!h->psi_sxy_y||
       !h->psi_vxx||!h->psi_vyx||!h->psi_vyy||!h->psi_vxy)
        return contract_failure("mutable download",__FILE__,__LINE__,
                                "one or more mutable host fields are null");
    denise_cuda_psv_fd4_l1_impl &c=ctx->impl;
    size_t f=c.full_elements,x=c.x_cpml_elements,y=c.y_cpml_elements;
    device_fields &d=c.fields;
    cudaEvent_t a=nullptr,b=nullptr;
    cudaError_t rc=cudaEventCreate(&a);
    if(rc!=cudaSuccess) return cuda_failure("mutable download event create",rc,__FILE__,__LINE__);
    rc=cudaEventCreate(&b);
    if(rc!=cudaSuccess) { cudaEventDestroy(a); return cuda_failure("mutable download event create",rc,__FILE__,__LINE__); }
    rc=cudaEventRecord(a); if(rc!=cudaSuccess) goto fail;
#define DMM(host,dev,row,col,count) do { rc=d2h(&(host)[row][col],(dev),(count),&c.stats); if(rc!=cudaSuccess) goto fail; } while(0)
#define DMT(host,dev,row,col,count) do { rc=d2h(&(host)[row][col][1],(dev),(count),&c.stats); if(rc!=cudaSuccess) goto fail; } while(0)
    DMM(h->vx,d.vx,-2,-2,f); DMM(h->vy,d.vy,-2,-2,f);
    DMM(h->sxx,d.sxx,-2,-2,f); DMM(h->syy,d.syy,-2,-2,f); DMM(h->sxy,d.sxy,-2,-2,f);
    DMT(h->r,d.r1,-2,-2,f); DMT(h->p,d.p1,-2,-2,f); DMT(h->q,d.q1,-2,-2,f);
    DMM(h->psi_sxx_x,d.psi_sxx_x,1,1,x); DMM(h->psi_sxy_x,d.psi_sxy_x,1,1,x);
    DMM(h->psi_vxx,d.psi_vxx,1,1,x); DMM(h->psi_vyx,d.psi_vyx,1,1,x);
    DMM(h->psi_syy_y,d.psi_syy_y,1,1,y); DMM(h->psi_sxy_y,d.psi_sxy_y,1,1,y);
    DMM(h->psi_vyy,d.psi_vyy,1,1,y); DMM(h->psi_vxy,d.psi_vxy,1,1,y);
#undef DMM
#undef DMT
    rc=cudaEventRecord(b); if(rc!=cudaSuccess) goto fail;
    rc=cudaEventSynchronize(b); if(rc!=cudaSuccess) goto fail;
    { float ms=0.0f; rc=cudaEventElapsedTime(&ms,a,b); if(rc!=cudaSuccess) goto fail; c.stats.download_ms+=ms; }
    cudaEventDestroy(b); cudaEventDestroy(a); return 0;
fail:
    cudaEventDestroy(b); cudaEventDestroy(a);
    return cuda_failure("mutable state download",rc,__FILE__,__LINE__);
}

int denise_cuda_psv_fd4_l1_get_stats(const denise_cuda_psv_fd4_l1 *ctx,
                                     denise_cuda_psv_fd4_l1_stats *stats) {
    psv_last_error[0]='\0';
    if(!ctx||!stats) return contract_failure("get_stats",__FILE__,__LINE__,
                                             "context or output is null");
    *stats=ctx->impl.stats; return 0;
}

int denise_cuda_psv_fd4_l1_destroy(denise_cuda_psv_fd4_l1 **handle) {
    psv_last_error[0]='\0';
    if(!handle) return 0;
    denise_cuda_psv_fd4_l1 *ctx=*handle; *handle=nullptr;
    if(!ctx) return 0;
    cudaError_t rc=ctx->impl.storage?cudaFree(ctx->impl.storage):cudaSuccess;
    delete ctx;
    if(rc!=cudaSuccess) return cuda_failure("destroy cudaFree",rc,__FILE__,__LINE__);
    return 0;
}

}  // extern "C"

namespace {

int validate_forward_config(const denise_cuda_psv_forward_config *c,
                            const char *operation) {
    if(!c) return contract_failure(operation,__FILE__,__LINE__,
                                   "forward configuration is null");
    if(validate_config(&c->core,operation)!=0) return -1;
    if(c->nt<2||c->nt==INT_MAX||c->ntr<1)
        return contract_failure(operation,__FILE__,__LINE__,
                                "NT must be in [2,INT_MAX) and receiver count positive");
    if(c->global_mode!=0||c->source_count!=1||c->source_type!=1||
       c->seismo!=1||c->ndt!=1||c->snapshots!=0||c->inv_stf!=0)
        return contract_failure(operation,__FILE__,__LINE__,
            "requires MODE=0 one QUELLTYP=1 source SEISMO=1 NDT=1 SNAP=0 INV_STF=0");
    return 0;
}

int calculate_forward_plan(const denise_cuda_psv_forward_config *c,
                           denise_cuda_psv_forward_stats *plan) {
    if(!plan) return contract_failure("forward required bytes",__FILE__,__LINE__,
                                      "plan output is null");
    std::memset(plan,0,sizeof(*plan));
    if(validate_forward_config(c,"forward required bytes")!=0) return -1;
    size_t fnx,fny,full,x,y,total,part,samples,checkpoint_elements;
    if(calculate_sizes(&c->core,&fnx,&fny,&full,&x,&y,&plan->b1_core_bytes)!=0)
        return -1;
    if(!checked_mul(full,8,&checkpoint_elements)||
       !checked_mul(x,4,&part)||!checked_add(checkpoint_elements,part,&checkpoint_elements)||
       !checked_mul(y,4,&part)||!checked_add(checkpoint_elements,part,&checkpoint_elements)||
       !checked_mul(checkpoint_elements,sizeof(float),&plan->checkpoint_bytes))
        return contract_failure("forward required bytes",__FILE__,__LINE__,
                                "checkpoint size overflows size_t");
    if(!checked_mul((size_t)c->nt,sizeof(float),&plan->source_signal_bytes)||
       !checked_mul((size_t)2,sizeof(int),&plan->source_geometry_bytes)||
       !checked_mul((size_t)c->ntr,2,&part)||
       !checked_mul(part,sizeof(int),&plan->receiver_geometry_bytes)||
       !checked_mul((size_t)c->ntr,(size_t)c->nt,&samples)||
       !checked_mul(samples,2,&part)||
       !checked_mul(part,sizeof(float),&plan->trace_bytes))
        return contract_failure("forward required bytes",__FILE__,__LINE__,
                                "forward state size overflows size_t");
    total=plan->b1_core_bytes;
    if(!checked_add(total,plan->source_signal_bytes,&total)||
       !checked_add(total,plan->source_geometry_bytes,&total)||
       !checked_add(total,plan->receiver_geometry_bytes,&total)||
       !checked_add(total,plan->trace_bytes,&total)||
       !checked_add(total,plan->workspace_bytes,&total))
        return contract_failure("forward required bytes",__FILE__,__LINE__,
                                "aggregate forward state size overflows size_t");
    plan->total_mandatory_bytes=total;
    return 0;
}

int calculate_adjoint_plan(const denise_cuda_psv_forward_config *c,
                           denise_cuda_psv_adjoint_stats *plan) {
    if(!plan) return contract_failure("adjoint required bytes",__FILE__,__LINE__,
                                      "plan output is null");
    std::memset(plan,0,sizeof(*plan));
    if(validate_forward_config(c,"adjoint required bytes")!=0) return -1;
    size_t fnx,fny,full,x,y,unused,part,total;
    if(calculate_sizes(&c->core,&fnx,&fny,&full,&x,&y,&unused)!=0) return -1;
    if(!checked_mul(full,8*sizeof(double),&plan->main_state_bytes)||
       !checked_add(x,y,&part)||!checked_mul(part,4*sizeof(double),
                                             &plan->cpml_state_bytes)||
       !checked_mul(full,4*sizeof(double),&plan->workspace_bytes)||
       !checked_mul((size_t)c->ntr,4*sizeof(float),&plan->residual_bytes))
        return contract_failure("adjoint required bytes",__FILE__,__LINE__,
                                "adjoint state size overflows size_t");
    total=plan->main_state_bytes;
    if(!checked_add(total,plan->cpml_state_bytes,&total)||
       !checked_add(total,plan->workspace_bytes,&total)||
       !checked_add(total,plan->residual_bytes,&total))
        return contract_failure("adjoint required bytes",__FILE__,__LINE__,
                                "adjoint aggregate size overflows size_t");
    plan->total_bytes=total;
    return 0;
}

bool valid_adjoint_host(const denise_cuda_psv_adjoint_host *h) {
    return h&&h->avx&&h->avy&&h->asxx&&h->asyy&&h->asxy&&h->ar&&h->ap&&h->aq&&
        h->psxx&&h->psxyx&&h->pvxx&&h->pvyx&&
        h->psxyy&&h->psyy&&h->pvxy&&h->pvyy;
}

void assign_adjoint_slices(denise_cuda_psv_forward *f) {
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    double *p=f->adjoint_storage;
    size_t n=c.full_elements,x=c.x_cpml_elements,y=c.y_cpml_elements;
#define A(member,count) f->adjoint.member=p; p+=(count)
    A(avx,n); A(avy,n); A(asxx,n); A(asyy,n); A(asxy,n);
    A(ar,n); A(ap,n); A(aq,n);
    A(psxx,x); A(psxyx,x); A(pvxx,x); A(pvyx,x);
    A(psxyy,y); A(psyy,y); A(pvxy,y); A(pvyy,y);
    A(wxx,n); A(wyx,n); A(wxy,n); A(wyy,n);
#undef A
    f->adjoint_residual=(float*)p;
}

cudaError_t raw_h2d(void *device,const void *host,size_t bytes,
                    denise_cuda_psv_fd4_l1_stats *stats) {
    cudaError_t rc=cudaMemcpy(device,host,bytes,cudaMemcpyHostToDevice);
    if(rc==cudaSuccess) { ++stats->h2d_transfer_calls; stats->h2d_bytes+=bytes; }
    return rc;
}

__global__ void explosive_source_fd4_l1(device_fields d,size_t pitch,
                                        const int *source_xy,
                                        const float *signal,int nt,int total_nt,
                                        float dt) {
    if(blockIdx.x||threadIdx.x) return;
    float amp;
    if(nt==1) amp=signal[1]/dt;
    else if(nt<total_nt) amp=(signal[nt]-signal[nt-2])/dt;
    else amp=-signal[nt-2]/dt;
    size_t q=fi(source_xy[0],source_xy[1],pitch);
    d.sxx[q]+=amp;
    d.syy[q]+=amp;
}

__global__ void sample_velocity_receivers(device_fields d,size_t pitch,
                                           const int *receiver_i,
                                           const int *receiver_j,
                                           float *trace_vx,float *trace_vy,
                                           int ntr,int nt,int total_nt) {
    int receiver=(int)(blockIdx.x*blockDim.x+threadIdx.x);
    if(receiver>=ntr) return;
    size_t q=fi(receiver_i[receiver],receiver_j[receiver],pitch);
    size_t sample=(size_t)receiver*(size_t)total_nt+(size_t)(nt-1);
    trace_vx[sample]=d.vx[q];
    trace_vy[sample]=d.vy[q];
}

int launch_source(denise_cuda_psv_forward *f,int nt) {
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    explosive_source_fd4_l1<<<1,1>>>(c.fields,c.full_nx,f->source_xy,
        f->source_signal,nt,f->config.nt,f->config.core.dt);
    PSV_CUDA_CALL("source kernel launch",cudaGetLastError());
    return 0;
}

int launch_receivers(denise_cuda_psv_forward *f,int nt) {
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    int blocks=(f->config.ntr+127)/128;
    sample_velocity_receivers<<<blocks,128>>>(c.fields,c.full_nx,
        f->receiver_i,f->receiver_j,f->trace_vx,f->trace_vy,
        f->config.ntr,nt,f->config.nt);
    PSV_CUDA_CALL("receiver kernel launch",cudaGetLastError());
    return 0;
}

void destroy_event_array(cudaEvent_t *events,size_t count) {
    if(!events) return;
    for(size_t i=0;i<count;++i) if(events[i]) cudaEventDestroy(events[i]);
    delete[] events;
}

void refresh_forward_stats(denise_cuda_psv_forward *f) {
    const denise_cuda_psv_fd4_l1_stats &core=f->core->impl.stats;
    f->stats.h2d_transfer_calls=core.h2d_transfer_calls;
    f->stats.d2h_transfer_calls=core.d2h_transfer_calls;
    f->stats.h2d_bytes=core.h2d_bytes;
    f->stats.d2h_bytes=core.d2h_bytes;
    f->stats.timesteps=core.timesteps;
    f->stats.velocity_kernel_ms=core.velocity_kernel_ms;
    f->stats.stress_kernel_ms=core.stress_kernel_ms;
}

/* The checkpoint never leaves its context. This guard also catches accidental
 * changes to the private configuration before restore. Compare floating
 * parameters by representation, since this is an exact replay contract. */
bool checkpoint_compatible(const denise_cuda_psv_forward_config &a,
                           const denise_cuda_psv_forward_config &b) {
    const denise_cuda_psv_fd4_l1_config &x=a.core,&y=b.core;
    return x.nx==y.nx&&x.ny==y.ny&&x.fw==y.fw&&
        x.fdorder==y.fdorder&&x.mechanisms==y.mechanisms&&
        x.mpi_ranks_x==y.mpi_ranks_x&&x.mpi_ranks_y==y.mpi_ranks_y&&
        x.boundary==y.boundary&&x.free_surface==y.free_surface&&
        x.mode==y.mode&&x.logical_device==y.logical_device&&
        std::memcmp(&x.dt,&y.dt,sizeof(float))==0&&
        std::memcmp(&x.dh,&y.dh,sizeof(float))==0&&
        std::memcmp(&x.hc1,&y.hc1,sizeof(float))==0&&
        std::memcmp(&x.hc2,&y.hc2,sizeof(float))==0&&
        std::memcmp(&x.bip1,&y.bip1,sizeof(float))==0&&
        std::memcmp(&x.bjm1,&y.bjm1,sizeof(float))==0&&
        std::memcmp(&x.cip1,&y.cip1,sizeof(float))==0&&
        std::memcmp(&x.cjm1,&y.cjm1,sizeof(float))==0&&
        a.nt==b.nt&&a.ntr==b.ntr&&a.global_mode==b.global_mode&&
        a.source_count==b.source_count&&a.source_type==b.source_type&&
        a.seismo==b.seismo&&a.ndt==b.ndt&&a.snapshots==b.snapshots&&
        a.inv_stf==b.inv_stf;
}

int checkpoint_copy_to(denise_cuda_psv_forward *f,float *storage,bool capture,
                       const char *operation) {
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    device_fields &d=c.fields;
    float *cursor=storage;
    const size_t full=c.full_elements,x=c.x_cpml_elements,y=c.y_cpml_elements;
#define COPY(member,n) do {                                                  \
    const size_t bytes=(n)*sizeof(float);                                     \
    cudaError_t rc=cudaMemcpyAsync(capture?cursor:d.member,                   \
        capture?d.member:cursor,bytes,cudaMemcpyDeviceToDevice);              \
    if(rc!=cudaSuccess) return cuda_failure(operation,rc,__FILE__,__LINE__);  \
    cursor+=(n); ++f->stats.checkpoint_d2d_calls;                              \
    f->stats.checkpoint_d2d_bytes+=bytes;                                      \
} while(0)
    COPY(vx,full); COPY(vy,full); COPY(sxx,full); COPY(syy,full); COPY(sxy,full);
    COPY(r1,full); COPY(p1,full); COPY(q1,full);
    COPY(psi_sxx_x,x); COPY(psi_sxy_x,x); COPY(psi_vxx,x); COPY(psi_vyx,x);
    COPY(psi_syy_y,y); COPY(psi_sxy_y,y); COPY(psi_vyy,y); COPY(psi_vxy,y);
#undef COPY
    return 0;
}

int checkpoint_copy(denise_cuda_psv_forward *f,bool capture) {
    return checkpoint_copy_to(f,f->checkpoint_storage,capture,
        capture?"checkpoint capture D2D":"checkpoint restore D2D");
}

void release_forward_partial(denise_cuda_psv_forward *f) {
    if(!f) return;
    if(f->checkpoint_storage) cudaFree(f->checkpoint_storage);
    if(f->segment_storage) cudaFree(f->segment_storage);
    if(f->adjoint_storage) cudaFree(f->adjoint_storage);
    if(f->adjoint_observed_storage) cudaFree(f->adjoint_observed_storage);
    if(f->native_gradient_storage) cudaFree(f->native_gradient_storage);
    if(f->physical_gradient_storage) cudaFree(f->physical_gradient_storage);
    if(f->core) {
        if(f->core->impl.storage) cudaFree(f->core->impl.storage);
        delete f->core;
    }
    delete f;
}

}  // namespace

extern "C" {

const char *denise_cuda_psv_forward_last_error(void) { return psv_last_error; }

int denise_cuda_psv_forward_required_bytes(
        const denise_cuda_psv_forward_config *config,
        denise_cuda_psv_forward_stats *plan) {
    psv_last_error[0]='\0';
    return calculate_forward_plan(config,plan);
}

int denise_cuda_psv_forward_create(
        const denise_cuda_psv_forward_config *cfg,
        const denise_cuda_psv_forward_host *host,
        denise_cuda_psv_forward **output) {
    psv_last_error[0]='\0';
    if(!output) return contract_failure("forward create",__FILE__,__LINE__,
                                        "context output is null");
    *output=nullptr;
    if(!host) return contract_failure("forward create",__FILE__,__LINE__,
                                      "host adapter is null");
    denise_cuda_psv_forward_stats plan;
    if(calculate_forward_plan(cfg,&plan)!=0||validate_host(&host->core)!=0) return -1;
    if(!host->source_positions||!host->source_positions[1]||!host->source_positions[2]||
       !host->source_signals||!host->source_signals[1]||
       !host->receiver_positions||!host->receiver_positions[1]||!host->receiver_positions[2]||
       !host->sectionvx||!host->sectionvy)
        return contract_failure("forward create",__FILE__,__LINE__,
                                "source, receiver, or trace host adapter is null");
    int source_i=(int)host->source_positions[1][1];
    int source_j=(int)host->source_positions[2][1];
    if(source_i<1||source_i>cfg->core.nx||source_j<1||source_j>cfg->core.ny)
        return contract_failure("forward create",__FILE__,__LINE__,
                                "source coordinate (%d,%d) is outside physical grid",source_i,source_j);
    for(int r=1;r<=cfg->ntr;++r) {
        int i=host->receiver_positions[1][r],j=host->receiver_positions[2][r];
        if(i<1||i>cfg->core.nx||j<1||j>cfg->core.ny)
            return contract_failure("forward create",__FILE__,__LINE__,
                "receiver %d coordinate (%d,%d) is outside physical grid",r,i,j);
    }
    bool receivers_unique=true;
    for(int r=1;r<=cfg->ntr&&receivers_unique;++r)
        for(int s=r+1;s<=cfg->ntr;++s)
            if(host->receiver_positions[1][r]==host->receiver_positions[1][s]&&
               host->receiver_positions[2][r]==host->receiver_positions[2][s]) {
                receivers_unique=false; break;
            }
    const char *profile_selector=std::getenv("DENISE_CUDA_PROFILE");
    bool profiling_enabled=false;
    if(profile_selector&&profile_selector[0]&&std::strcmp(profile_selector,"0")!=0) {
        if(std::strcmp(profile_selector,"1")!=0)
            return contract_failure("forward create",__FILE__,__LINE__,
                "unknown DENISE_CUDA_PROFILE='%s' (expected 0 or 1)",profile_selector);
        profiling_enabled=true;
    }
    const std::chrono::steady_clock::time_point setup_start=
        std::chrono::steady_clock::now();
    denise_cuda_device_info info;
    if(denise_cuda_select_device(cfg->core.logical_device,
            cfg->core.safety_reserve_bytes,cfg->core.user_cap_bytes,&info)!=0)
        return contract_failure("forward create device selection",__FILE__,__LINE__,
                                "%s",denise_cuda_last_error());
    if(plan.total_mandatory_bytes>info.usable_budget_bytes)
        return contract_failure("forward create aggregate VRAM budget",__FILE__,__LINE__,
            "mandatory forward state %zu exceeds usable budget %zu",
            plan.total_mandatory_bytes,info.usable_budget_bytes);

    denise_cuda_psv_forward *f=new(std::nothrow) denise_cuda_psv_forward();
    denise_cuda_psv_fd4_l1 *core=new(std::nothrow) denise_cuda_psv_fd4_l1();
    if(!f||!core) { delete f; delete core; return contract_failure(
        "forward create",__FILE__,__LINE__,"host allocation failed"); }
    std::memset(f,0,sizeof(*f)); std::memset(core,0,sizeof(*core));
    f->core=core; f->config=*cfg; f->profiling_enabled=profiling_enabled;
    f->receivers_unique=receivers_unique;
    f->next_timestep=1;
    f->stats=plan; f->stats.profiling_enabled=profiling_enabled?1:0;
    f->stats.usable_budget_bytes=info.usable_budget_bytes;
    f->stats.remaining_budget_bytes=info.usable_budget_bytes-plan.total_mandatory_bytes;

    size_t fnx,fny,full,x,y,core_bytes;
    if(calculate_sizes(&cfg->core,&fnx,&fny,&full,&x,&y,&core_bytes)!=0) {
        release_forward_partial(f); return -1;
    }
    denise_cuda_psv_fd4_l1_impl &c=core->impl;
    c.config=cfg->core; c.full_nx=fnx; c.full_ny=fny; c.full_elements=full;
    c.x_cpml_elements=x; c.y_cpml_elements=y;
    c.stats.mandatory_state_bytes=core_bytes;
    c.stats.usable_budget_bytes=info.usable_budget_bytes;
    c.stats.remaining_budget_bytes=info.usable_budget_bytes-plan.total_mandatory_bytes;
    cudaError_t rc=cudaMalloc((void**)&c.storage,plan.total_mandatory_bytes);
    if(rc!=cudaSuccess) { release_forward_partial(f); return cuda_failure(
        "aggregate forward cudaMalloc",rc,__FILE__,__LINE__); }
    assign_slices(&c);
    f->stats.context_setup_ms=(float)std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-setup_start).count();
    unsigned char *cursor=(unsigned char*)c.storage+core_bytes;
    f->source_xy=(int*)cursor; cursor+=plan.source_geometry_bytes;
    f->source_signal=(float*)cursor; cursor+=plan.source_signal_bytes;
    f->receiver_i=(int*)cursor; cursor+=(size_t)cfg->ntr*sizeof(int);
    f->receiver_j=(int*)cursor; cursor+=(size_t)cfg->ntr*sizeof(int);
    size_t trace_elements=(size_t)cfg->ntr*(size_t)cfg->nt;
    f->trace_vx=(float*)cursor; cursor+=trace_elements*sizeof(float);
    f->trace_vy=(float*)cursor;

    if(upload_initial(&c,&host->core)!=0) { release_forward_partial(f); return -1; }
    cudaEvent_t a=nullptr,b=nullptr;
    rc=cudaEventCreate(&a);
    if(rc!=cudaSuccess) { release_forward_partial(f); return cuda_failure(
        "forward upload event create",rc,__FILE__,__LINE__); }
    rc=cudaEventCreate(&b);
    if(rc!=cudaSuccess) { cudaEventDestroy(a); release_forward_partial(f); return cuda_failure(
        "forward upload event create",rc,__FILE__,__LINE__); }
    rc=cudaEventRecord(a);
    int source_xy[2]={source_i,source_j};
    if(rc==cudaSuccess) rc=raw_h2d(f->source_xy,source_xy,sizeof(source_xy),&c.stats);
    if(rc==cudaSuccess) rc=raw_h2d(f->source_signal,&host->source_signals[1][1],
                                   plan.source_signal_bytes,&c.stats);
    if(rc==cudaSuccess) rc=raw_h2d(f->receiver_i,&host->receiver_positions[1][1],
                                   (size_t)cfg->ntr*sizeof(int),&c.stats);
    if(rc==cudaSuccess) rc=raw_h2d(f->receiver_j,&host->receiver_positions[2][1],
                                   (size_t)cfg->ntr*sizeof(int),&c.stats);
    if(rc==cudaSuccess) rc=cudaMemset(f->trace_vx,0,plan.trace_bytes);
    if(rc==cudaSuccess) rc=cudaEventRecord(b);
    if(rc==cudaSuccess) rc=cudaEventSynchronize(b);
    float extra_upload_ms=0.0f;
    if(rc==cudaSuccess) rc=cudaEventElapsedTime(&extra_upload_ms,a,b);
    cudaEventDestroy(b); cudaEventDestroy(a);
    if(rc!=cudaSuccess) { release_forward_partial(f); return cuda_failure(
        "forward source/receiver upload",rc,__FILE__,__LINE__); }
    f->stats.initial_upload_ms=c.stats.upload_ms+extra_upload_ms;
    refresh_forward_stats(f);
    *output=f;
    return 0;
}

int denise_cuda_psv_forward_run_range(denise_cuda_psv_forward *f,
                                      int begin,int end) {
    psv_last_error[0]='\0';
    if(!f) return contract_failure("forward range",__FILE__,__LINE__,"context is null");
    if(begin<1||begin!=f->next_timestep||end<begin||end>f->config.nt)
        return contract_failure("forward range",__FILE__,__LINE__,
            "expected absolute begin=%d and 1<=begin<=end<=NT=%d; got [%d,%d]",
            f->next_timestep,f->config.nt,begin,end);
    if(f->segment_record_armed) {
        int k=f->segment_record_armed-1;
        int expected_begin=(int)(((long long)k*f->config.nt)/f->segment_count)+1;
        int expected_end=(int)(((long long)(k+1)*f->config.nt)/f->segment_count);
        if(begin!=expected_begin||end!=expected_end)
            return contract_failure("segment recording range",__FILE__,__LINE__,
                "segment %d requires [%d,%d], got [%d,%d]",
                k,expected_begin,expected_end,begin,end);
    }
    const size_t count=(size_t)end-(size_t)begin+1;
    size_t event_count;
    if(f->profiling_enabled) {
        size_t phase_events;
        if(!checked_mul(count,4,&phase_events)||
           !checked_add(phase_events,1,&event_count))
            return contract_failure("forward run",__FILE__,__LINE__,
                                    "profiling event count overflows size_t");
    } else event_count=2;
    cudaEvent_t *events=new(std::nothrow) cudaEvent_t[event_count]();
    if(!events) return contract_failure("forward run",__FILE__,__LINE__,
                                        "profiling event allocation failed");
    cudaError_t rc=cudaSuccess;
    size_t created=0;
    for(;created<event_count;++created) {
        rc=cudaEventCreate(&events[created]);
        if(rc!=cudaSuccess) {
            destroy_event_array(events,event_count);
            return cuda_failure("forward kernel event create",rc,__FILE__,__LINE__);
        }
    }
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    float *operands=f->segment_record_armed?f->segment_operands:nullptr;
    size_t operand_stride=(size_t)f->stats.max_segment_length*
        (size_t)f->config.core.nx*(size_t)f->config.core.ny;
    rc=cudaEventRecord(events[0]);
    if(rc!=cudaSuccess) goto fail;
    if(f->profiling_enabled) ++f->stats.profile_event_records;
    else ++f->stats.resident_event_records;
    for(int nt=begin;nt<=end;++nt) {
        size_t h2d_before=c.stats.h2d_transfer_calls;
        size_t d2h_before=c.stats.d2h_transfer_calls;
        size_t event=((size_t)nt-(size_t)begin)*4;
        if(launch_velocity(&c,operands,nt-begin,operand_stride)!=0) goto launch_fail;
        if(f->profiling_enabled) {
            rc=cudaEventRecord(events[event+1]); if(rc!=cudaSuccess) goto fail;
            ++f->stats.profile_event_records;
        }
        if(launch_stress(&c,operands,nt-begin,operand_stride)!=0) goto launch_fail;
        if(f->profiling_enabled) {
            rc=cudaEventRecord(events[event+2]); if(rc!=cudaSuccess) goto fail;
            ++f->stats.profile_event_records;
        }
        if(launch_source(f,nt)!=0) goto launch_fail;
        if(f->profiling_enabled) {
            rc=cudaEventRecord(events[event+3]); if(rc!=cudaSuccess) goto fail;
            ++f->stats.profile_event_records;
        }
        if(launch_receivers(f,nt)!=0) goto launch_fail;
        if(f->profiling_enabled) {
            rc=cudaEventRecord(events[event+4]); if(rc!=cudaSuccess) goto fail;
            ++f->stats.profile_event_records;
        }
        ++c.stats.timesteps;
        if(c.stats.h2d_transfer_calls!=h2d_before||
           c.stats.d2h_transfer_calls!=d2h_before) {
            contract_failure("forward timestep residency",__FILE__,__LINE__,
                "host/device transfer detected during timestep %d",nt);
            goto fail;
        }
    }
    if(!f->profiling_enabled) {
        rc=cudaEventRecord(events[1]); if(rc!=cudaSuccess) goto fail;
        ++f->stats.resident_event_records;
    }
    rc=cudaEventSynchronize(events[event_count-1]); if(rc!=cudaSuccess) goto fail;
    ++f->stats.forward_synchronization_calls;
    { float elapsed=0.0f;
      rc=cudaEventElapsedTime(&elapsed,events[0],events[event_count-1]);
      if(rc==cudaSuccess) f->stats.resident_timestep_ms+=elapsed; }
    if(rc!=cudaSuccess) goto fail;
    ++f->stats.resident_elapsed_queries;
    if(f->profiling_enabled) {
        for(int nt=begin;nt<=end;++nt) {
            size_t event=((size_t)nt-(size_t)begin)*4;
            float vm=0.0f,sm=0.0f,qm=0.0f,rm=0.0f;
            rc=cudaEventElapsedTime(&vm,events[event],events[event+1]);
            if(rc==cudaSuccess) rc=cudaEventElapsedTime(&sm,events[event+1],events[event+2]);
            if(rc==cudaSuccess) rc=cudaEventElapsedTime(&qm,events[event+2],events[event+3]);
            if(rc==cudaSuccess) rc=cudaEventElapsedTime(&rm,events[event+3],events[event+4]);
            if(rc!=cudaSuccess) goto fail;
            f->stats.profile_elapsed_queries+=4;
            c.stats.velocity_kernel_ms+=vm; c.stats.stress_kernel_ms+=sm;
            c.stats.combined_kernel_ms+=vm+sm;
            f->stats.source_kernel_ms+=qm; f->stats.receiver_kernel_ms+=rm;
        }
    }
    destroy_event_array(events,event_count);
    if(f->segment_record_armed) {
        f->segment_record_ready=f->segment_record_armed;
        f->segment_record_armed=0;
    }
    f->next_timestep=end+1;
    refresh_forward_stats(f);
    return 0;
fail:
    f->next_timestep=0; /* a partially launched range cannot be retried */
    f->segment_record_armed=0; f->segment_record_ready=0;
    destroy_event_array(events,event_count);
    if(rc!=cudaSuccess) return cuda_failure("forward timestep",rc,__FILE__,__LINE__);
    return -1;
launch_fail:
    f->next_timestep=0;
    f->segment_record_armed=0; f->segment_record_ready=0;
    destroy_event_array(events,event_count);
    return -1;
}

int denise_cuda_psv_forward_checkpoint_reserve(denise_cuda_psv_forward *f) {
    psv_last_error[0]='\0';
    if(!f) return contract_failure("checkpoint reserve",__FILE__,__LINE__,
                                   "context is null");
    if(f->checkpoint_storage) return 0;
    const size_t bytes=f->stats.checkpoint_bytes;
    if(bytes>f->stats.remaining_budget_bytes)
        return contract_failure("checkpoint reserve budget",__FILE__,__LINE__,
            "checkpoint %zu exceeds remaining usable budget %zu",
            bytes,f->stats.remaining_budget_bytes);
    float *storage=nullptr;
    cudaError_t rc=cudaMalloc((void**)&storage,bytes);
    if(rc!=cudaSuccess) return cuda_failure("checkpoint reserve cudaMalloc",rc,
                                           __FILE__,__LINE__);
    f->checkpoint_storage=storage;
    f->stats.remaining_budget_bytes-=bytes;
    f->core->impl.stats.remaining_budget_bytes=f->stats.remaining_budget_bytes;
    return 0;
}

int denise_cuda_psv_forward_checkpoint_capture(denise_cuda_psv_forward *f,
                                                int completed_timestep) {
    psv_last_error[0]='\0';
    if(!f||!f->checkpoint_storage)
        return contract_failure("checkpoint capture",__FILE__,__LINE__,
                                "context or reserved device storage is missing");
    if(completed_timestep<1||completed_timestep>=f->config.nt||
       completed_timestep!=f->next_timestep-1)
        return contract_failure("checkpoint capture",__FILE__,__LINE__,
            "completed timestep %d is not the current complete range boundary %d",
            completed_timestep,f->next_timestep-1);
    f->checkpoint_valid=false;
    if(checkpoint_copy(f,true)!=0) return -1;
    f->checkpoint_config=f->config;
    f->checkpoint_completed_timestep=completed_timestep;
    f->checkpoint_valid=true;
    ++f->stats.checkpoint_capture_calls;
    return 0;
}

int denise_cuda_psv_forward_checkpoint_restore(denise_cuda_psv_forward *f,
                                                int *next_timestep) {
    psv_last_error[0]='\0';
    if(!f||!f->checkpoint_storage||!f->checkpoint_valid||!next_timestep)
        return contract_failure("checkpoint restore",__FILE__,__LINE__,
                                "context, captured checkpoint, or output is missing");
    if(!checkpoint_compatible(f->config,f->checkpoint_config))
        return contract_failure("checkpoint restore",__FILE__,__LINE__,
                                "checkpoint configuration is incompatible");
    if(checkpoint_copy(f,false)!=0) {
        f->next_timestep=0; /* partial device copy: fail closed */
        return -1;
    }
    f->next_timestep=f->checkpoint_completed_timestep+1;
    *next_timestep=f->next_timestep;
    ++f->stats.checkpoint_restore_calls;
    return 0;
}

int denise_cuda_psv_forward_segments_prepare(denise_cuda_psv_forward *f,
                                               int requested_segments) {
    psv_last_error[0]='\0';
    if(!f||requested_segments<0||f->segment_storage||f->next_timestep!=1||
       f->core->impl.stats.timesteps)
        return contract_failure("segment prepare",__FILE__,__LINE__,
            "requires an untouched context, nonnegative count, and no existing bank");
    if(!requested_segments) requested_segments=32;
    int count=requested_segments<f->config.nt?requested_segments:f->config.nt;
    int max_length=(int)(((long long)f->config.nt+count-1)/count);
    size_t bank,operand_elements,operand_bytes,total,cells;
    if(!checked_mul((size_t)f->config.core.nx,(size_t)f->config.core.ny,&cells)||
       !checked_mul((size_t)count,f->stats.checkpoint_bytes,&bank)||
       !checked_mul((size_t)max_length,cells,&operand_elements)||
       !checked_mul(operand_elements,6,&operand_elements)||
       !checked_mul(operand_elements,sizeof(float),&operand_bytes)||
       !checked_add(bank,operand_bytes,&total))
        return contract_failure("segment prepare",__FILE__,__LINE__,
                                "bank or operand size overflows size_t");
    if(total>f->stats.remaining_budget_bytes)
        return contract_failure("segment prepare budget",__FILE__,__LINE__,
            "bank %zu plus operands %zu exceeds remaining usable budget %zu",
            bank,operand_bytes,f->stats.remaining_budget_bytes);
    float *storage=nullptr;
    cudaError_t rc=cudaMalloc((void**)&storage,total);
    if(rc!=cudaSuccess) return cuda_failure("segment aggregate cudaMalloc",rc,
                                           __FILE__,__LINE__);
    f->segment_storage=storage;
    f->segment_operands=(float*)((unsigned char*)storage+bank);
    f->segment_config=f->config;
    f->segment_count=count;
    f->stats.segment_count=count;
    f->stats.max_segment_length=max_length;
    f->stats.segment_bank_bytes=bank;
    f->stats.segment_operand_bytes=operand_bytes;
    f->stats.remaining_budget_bytes-=total;
    f->core->impl.stats.remaining_budget_bytes=f->stats.remaining_budget_bytes;
    /* Slot zero is the actual initial state; M8c assumes a zero initial state.
     * This extra seed keeps the CUDA API exact for any valid uploaded state. */
    if(checkpoint_copy_to(f,storage,true,"segment initial-state D2D")!=0) {
        cudaFree(storage); f->segment_storage=nullptr; f->segment_operands=nullptr;
        f->stats.remaining_budget_bytes+=total;
        f->core->impl.stats.remaining_budget_bytes=f->stats.remaining_budget_bytes;
        f->stats.segment_count=0; f->stats.max_segment_length=0;
        f->stats.segment_bank_bytes=0; f->stats.segment_operand_bytes=0;
        f->segment_count=0;
        return -1;
    }
    f->stats.segment_d2d_calls+=16;
    return 0;
}

int denise_cuda_psv_forward_segment_bounds(const denise_cuda_psv_forward *f,
                                            int segment,int *begin,int *end) {
    psv_last_error[0]='\0';
    if(!f||!f->segment_storage||segment<0||segment>=f->segment_count||
       !begin||!end)
        return contract_failure("segment bounds",__FILE__,__LINE__,
                                "invalid prepared context, segment, or outputs");
    *begin=(int)(((long long)segment*f->config.nt)/f->segment_count)+1;
    *end=(int)(((long long)(segment+1)*f->config.nt)/f->segment_count);
    return 0;
}

int denise_cuda_psv_forward_segment_capture(denise_cuda_psv_forward *f,
                                             int segment) {
    psv_last_error[0]='\0';
    int begin,end;
    if(denise_cuda_psv_forward_segment_bounds(f,segment,&begin,&end)!=0)
        return -1;
    if(segment>=f->segment_count-1||segment!=f->segment_captured||
       f->next_timestep!=end+1||
       !checkpoint_compatible(f->config,f->segment_config))
        return contract_failure("segment capture",__FILE__,__LINE__,
            "expected next uncaptured interior boundary at timestep %d",end);
    float *slot=f->segment_storage+(size_t)(segment+1)*
        (f->stats.checkpoint_bytes/sizeof(float));
    if(checkpoint_copy_to(f,slot,true,"segment boundary D2D")!=0) return -1;
    ++f->segment_captured;
    f->stats.segment_d2d_calls+=16;
    return 0;
}

int denise_cuda_psv_forward_segment_record_next(denise_cuda_psv_forward *f,
                                                 int segment) {
    psv_last_error[0]='\0';
    int begin,end;
    if(denise_cuda_psv_forward_segment_bounds(f,segment,&begin,&end)!=0)
        return -1;
    if(f->next_timestep!=begin||f->segment_record_armed||
       !checkpoint_compatible(f->config,f->segment_config))
        return contract_failure("segment record",__FILE__,__LINE__,
            "segment %d cannot record from next timestep %d",segment,f->next_timestep);
    f->segment_record_ready=0;
    f->segment_record_armed=segment+1;
    return 0;
}

int denise_cuda_psv_forward_segment_replay(denise_cuda_psv_forward *f,
                                            int segment) {
    psv_last_error[0]='\0';
    int begin,end;
    if(denise_cuda_psv_forward_segment_bounds(f,segment,&begin,&end)!=0)
        return -1;
    if(f->segment_captured!=f->segment_count-1||
       (!f->segment_original_complete&&f->next_timestep!=f->config.nt+1)||
       f->segment_record_armed||
       !checkpoint_compatible(f->config,f->segment_config))
        return contract_failure("segment replay",__FILE__,__LINE__,
            "forward trajectory, checkpoint bank, or configuration is incomplete");
    f->segment_original_complete=true;
    float *slot=f->segment_storage+(size_t)segment*
        (f->stats.checkpoint_bytes/sizeof(float));
    if(checkpoint_copy_to(f,slot,false,"segment replay restore D2D")!=0) {
        f->next_timestep=0; return -1;
    }
    f->stats.segment_d2d_calls+=16;
    f->next_timestep=begin;
    if(denise_cuda_psv_forward_segment_record_next(f,segment)!=0) return -1;
    return denise_cuda_psv_forward_run_range(f,begin,end);
}

int denise_cuda_psv_forward_segment_download_operands(
        denise_cuda_psv_forward *f,int segment,float *host_fields,
        size_t float_capacity) {
    psv_last_error[0]='\0';
    int begin,end;
    if(denise_cuda_psv_forward_segment_bounds(f,segment,&begin,&end)!=0)
        return -1;
    size_t cells=(size_t)f->config.core.nx*(size_t)f->config.core.ny;
    size_t field_elements=(size_t)(end-begin+1)*cells;
    size_t stride=(size_t)f->stats.max_segment_length*cells;
    if(!host_fields||float_capacity<6*field_elements||
       f->segment_record_ready!=segment+1)
        return contract_failure("segment operand download",__FILE__,__LINE__,
                                "recorded segment or host capacity is invalid");
    for(int k=0;k<6;++k) {
        cudaError_t rc=d2h(host_fields+(size_t)k*field_elements,
            f->segment_operands+(size_t)k*stride,field_elements,
            &f->core->impl.stats);
        if(rc!=cudaSuccess) return cuda_failure("segment diagnostic D2H",rc,
                                               __FILE__,__LINE__);
    }
    refresh_forward_stats(f);
    return 0;
}

int denise_cuda_psv_adjoint_required_bytes(
        const denise_cuda_psv_forward_config *config,
        denise_cuda_psv_adjoint_stats *plan) {
    psv_last_error[0]='\0';
    return calculate_adjoint_plan(config,plan);
}

static int adjoint_prepare_impl(
        denise_cuda_psv_forward *f,
        const denise_cuda_psv_adjoint_host *initial,
        bool zero_initial) {
    psv_last_error[0]='\0';
    if(!f||(!zero_initial&&!valid_adjoint_host(initial))||f->adjoint_storage)
        return contract_failure("adjoint prepare",__FILE__,__LINE__,
            "context/initial state is invalid or adjoint is already prepared");
    if(!f->receivers_unique)
        return contract_failure("adjoint prepare receivers",__FILE__,__LINE__,
                                "duplicate receiver cells are unsupported");
    denise_cuda_psv_adjoint_stats plan;
    if(calculate_adjoint_plan(&f->config,&plan)!=0) return -1;
    if(plan.total_bytes>f->stats.remaining_budget_bytes)
        return contract_failure("adjoint prepare budget",__FILE__,__LINE__,
            "adjoint foundation %zu exceeds remaining usable budget %zu",
            plan.total_bytes,f->stats.remaining_budget_bytes);
    double *storage=nullptr;
    cudaError_t rc=cudaMalloc((void**)&storage,plan.total_bytes);
    if(rc!=cudaSuccess) return cuda_failure("adjoint aggregate cudaMalloc",rc,
                                           __FILE__,__LINE__);
    f->adjoint_storage=storage;
    f->adjoint_stats=plan;
    f->adjoint_stats.remaining_budget_bytes=
        f->stats.remaining_budget_bytes-plan.total_bytes;
    assign_adjoint_slices(f);
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    if(zero_initial) {
        rc=cudaMemset(storage,0,plan.total_bytes);
        if(rc!=cudaSuccess) goto fail;
        ++f->adjoint_stats.initial_zero_calls;
    } else {
    size_t n=c.full_elements,x=c.x_cpml_elements,y=c.y_cpml_elements;
#define AU(member,count) do { rc=cudaMemcpy(f->adjoint.member,initial->member, \
    (count)*sizeof(double),cudaMemcpyHostToDevice);                           \
    if(rc!=cudaSuccess) goto fail; ++f->adjoint_stats.initial_h2d_calls; } while(0)
    AU(avx,n); AU(avy,n); AU(asxx,n); AU(asyy,n); AU(asxy,n);
    AU(ar,n); AU(ap,n); AU(aq,n);
    AU(psxx,x); AU(psxyx,x); AU(pvxx,x); AU(pvyx,x);
    AU(psxyy,y); AU(psyy,y); AU(pvxy,y); AU(pvyy,y);
#undef AU
    rc=cudaMemset(f->adjoint.wxx,0,plan.workspace_bytes);
    if(rc!=cudaSuccess) goto fail;
    }
    f->stats.remaining_budget_bytes-=plan.total_bytes;
    c.stats.remaining_budget_bytes=f->stats.remaining_budget_bytes;
    return 0;
fail:
    cudaFree(storage);
    f->adjoint_storage=nullptr; f->adjoint_residual=nullptr;
    std::memset(&f->adjoint,0,sizeof(f->adjoint));
    std::memset(&f->adjoint_stats,0,sizeof(f->adjoint_stats));
    return cuda_failure("adjoint initial upload",rc,__FILE__,__LINE__);
}

int denise_cuda_psv_adjoint_prepare(
        denise_cuda_psv_forward *f,
        const denise_cuda_psv_adjoint_host *initial) {
    return adjoint_prepare_impl(f,initial,false);
}

int denise_cuda_psv_adjoint_prepare_zero(
        denise_cuda_psv_forward *f) {
    return adjoint_prepare_impl(f,nullptr,true);
}

int denise_cuda_psv_adjoint_step(
        denise_cuda_psv_forward *f,int timestep,
        const float *modeled_vx,const float *observed_vx,
        const float *modeled_vy,const float *observed_vy) {
    psv_last_error[0]='\0';
    if(!f||!f->adjoint_storage||timestep<1||timestep>f->config.nt)
        return contract_failure("adjoint step",__FILE__,__LINE__,
                                "context is unprepared or timestep is invalid");
    if(f->adjoint_sweep_state!=ADJOINT_SWEEP_UNPREPARED)
        return contract_failure("adjoint step",__FILE__,__LINE__,
                                "standalone stepping cannot alter a prepared sweep");
    cudaError_t rc=cudaSuccess;
    size_t bytes=(size_t)f->config.ntr*sizeof(float);
    if(timestep>1) {
        if(!modeled_vx||!observed_vx||!modeled_vy||!observed_vy)
            return contract_failure("adjoint residual upload",__FILE__,__LINE__,
                                    "sample >1 requires four receiver vectors");
        const float *input[4]={modeled_vx,observed_vx,modeled_vy,observed_vy};
        for(int k=0;k<4;++k) {
            /* Copy pageable caller storage before kernel execution so the
             * input lifetime ends with this call's host-side copy. */
            rc=cudaMemcpy(f->adjoint_residual+(size_t)k*f->config.ntr,
                          input[k],bytes,cudaMemcpyHostToDevice);
            if(rc!=cudaSuccess) return cuda_failure("adjoint residual H2D",rc,
                                                    __FILE__,__LINE__);
            ++f->adjoint_stats.residual_h2d_calls;
            f->adjoint_stats.residual_h2d_bytes+=bytes;
        }
    }
    return launch_adjoint_operator(f,timestep,false,0);
}

int denise_cuda_psv_adjoint_download(
        denise_cuda_psv_forward *f,denise_cuda_psv_adjoint_host *host) {
    psv_last_error[0]='\0';
    if(!f||!f->adjoint_storage||!valid_adjoint_host(host))
        return contract_failure("adjoint download",__FILE__,__LINE__,
                                "context is unprepared or output is invalid");
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    size_t n=c.full_elements,x=c.x_cpml_elements,y=c.y_cpml_elements;
    cudaError_t rc=cudaSuccess;
#define AD(member,count) do { rc=cudaMemcpy(host->member,f->adjoint.member, \
    (count)*sizeof(double),cudaMemcpyDeviceToHost);                           \
    if(rc!=cudaSuccess) return cuda_failure("adjoint diagnostic D2H",rc,     \
                                            __FILE__,__LINE__);               \
    ++f->adjoint_stats.diagnostic_d2h_calls; } while(0)
    AD(avx,n); AD(avy,n); AD(asxx,n); AD(asyy,n); AD(asxy,n);
    AD(ar,n); AD(ap,n); AD(aq,n);
    AD(psxx,x); AD(psxyx,x); AD(pvxx,x); AD(pvyx,x);
    AD(psxyy,y); AD(psyy,y); AD(pvxy,y); AD(pvyy,y);
#undef AD
    return 0;
}

int denise_cuda_psv_adjoint_get_stats(
        const denise_cuda_psv_forward *f,denise_cuda_psv_adjoint_stats *stats) {
    psv_last_error[0]='\0';
    if(!f||!f->adjoint_storage||!stats)
        return contract_failure("adjoint get stats",__FILE__,__LINE__,
                                "context is unprepared or output is null");
    *stats=f->adjoint_stats;
    return 0;
}

int denise_cuda_psv_adjoint_sweep_required_bytes(
        const denise_cuda_psv_forward_config *config,
        denise_cuda_psv_adjoint_sweep_stats *plan) {
    psv_last_error[0]='\0';
    return calculate_adjoint_sweep_plan(config,plan);
}

static int adjoint_sweep_prepare_impl(
        denise_cuda_psv_forward *f,const float *source_vx,
        const float *source_vy,size_t float_count_per_component,
        bool production_residual) {
    psv_last_error[0]='\0';
    const char *kind=production_residual?"production residual":"observed traces";
    if(!f||!source_vx||!source_vy)
        return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
                                "context or %s are null",kind);
    if(f->adjoint_sweep_state!=ADJOINT_SWEEP_UNPREPARED||
       f->adjoint_observed_storage)
        return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
                                "sweep is already prepared or invalid");
    size_t expected=0;
    if(!checked_mul((size_t)f->config.ntr,(size_t)f->config.nt,&expected))
        return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
                                "%s count overflows size_t",kind);
    if(float_count_per_component!=expected)
        return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
            "each %s component must contain exactly NTR*NT=%zu floats",
            kind,expected);
    if(!f->segment_storage)
        return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
                                "segment bank is not prepared");
    if(f->segment_captured!=f->segment_count-1)
        return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
                                "segment checkpoint bank is incomplete");
    if(f->next_timestep!=f->config.nt+1||f->segment_record_armed)
        return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
                                "original forward trajectory is incomplete");
    if(!checkpoint_compatible(f->config,f->segment_config))
        return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
                                "segment configuration changed");
    if(!f->adjoint_storage)
        return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
                                "adjoint state is not prepared");
    denise_cuda_psv_adjoint_sweep_stats plan;
    if(calculate_adjoint_sweep_plan(&f->config,&plan)!=0) return -1;
    if(plan.total_incremental_bytes>f->stats.remaining_budget_bytes)
        return contract_failure("adjoint sweep prepare budget",__FILE__,__LINE__,
            "%s %zu exceed remaining usable budget %zu",kind,
            plan.total_incremental_bytes,f->stats.remaining_budget_bytes);
    float *storage=nullptr;
    cudaError_t rc=cudaMalloc((void**)&storage,plan.total_incremental_bytes);
    if(rc!=cudaSuccess)
        return cuda_failure("adjoint sweep source cudaMalloc",rc,
                            __FILE__,__LINE__);
    const size_t component_bytes=plan.observed_trace_bytes/2;
    rc=cudaMemcpy(storage,source_vx,component_bytes,cudaMemcpyHostToDevice);
    if(rc!=cudaSuccess) goto fail;
    if(production_residual) {
        ++plan.production_residual_h2d_calls;
        plan.production_residual_h2d_bytes+=component_bytes;
        const char *fail=getenv(
            "DENISE_CUDA_PSV_TEST_FAIL_PRODUCTION_RESIDUAL_AFTER_FIRST_COPY");
        if(fail&&fail[0]=='1'&&fail[1]=='\0') {
            cudaFree(storage);
            return contract_failure("adjoint sweep prepare",__FILE__,__LINE__,
                "injected production residual upload failure after first copy");
        }
    } else {
        ++plan.observed_h2d_calls; plan.observed_h2d_bytes+=component_bytes;
    }
    rc=cudaMemcpy(storage+expected,source_vy,component_bytes,
                  cudaMemcpyHostToDevice);
    if(rc!=cudaSuccess) goto fail;
    if(production_residual) {
        ++plan.production_residual_h2d_calls;
        plan.production_residual_h2d_bytes+=component_bytes;
    } else {
        ++plan.observed_h2d_calls; plan.observed_h2d_bytes+=component_bytes;
    }
    f->adjoint_observed_storage=storage;
    f->adjoint_observed_vx=storage;
    f->adjoint_observed_vy=storage+expected;
    f->adjoint_resident_source_kind=production_residual?
        ADJOINT_RESIDENT_SOURCE_PRODUCTION_RESIDUAL:
        ADJOINT_RESIDENT_SOURCE_OBSERVED;
    f->stats.remaining_budget_bytes-=plan.total_incremental_bytes;
    f->core->impl.stats.remaining_budget_bytes=f->stats.remaining_budget_bytes;
    plan.remaining_budget_bytes=f->stats.remaining_budget_bytes;
    plan.next_reverse_segment=f->segment_count-1;
    plan.prepared=1;
    f->adjoint_sweep_stats=plan;
    f->adjoint_sweep_state=ADJOINT_SWEEP_READY;
    f->segment_original_complete=true;
    return 0;
fail:
    cudaFree(storage);
    return cuda_failure("adjoint sweep source H2D",rc,__FILE__,__LINE__);
}

int denise_cuda_psv_adjoint_sweep_prepare(
        denise_cuda_psv_forward *f,const float *observed_vx,
        const float *observed_vy,size_t float_count_per_component) {
    return adjoint_sweep_prepare_impl(f,observed_vx,observed_vy,
                                      float_count_per_component,false);
}

int denise_cuda_psv_adjoint_production_residual_prepare(
        denise_cuda_psv_forward *f,const float *production_residual_vx,
        const float *production_residual_vy,
        size_t float_count_per_component) {
    return adjoint_sweep_prepare_impl(f,production_residual_vx,
        production_residual_vy,float_count_per_component,true);
}

int denise_cuda_psv_adjoint_reverse_segment(
        denise_cuda_psv_forward *f,int segment) {
    psv_last_error[0]='\0';
    if(!f||f->adjoint_sweep_state!=ADJOINT_SWEEP_READY)
        return contract_failure("adjoint reverse segment",__FILE__,__LINE__,
                                "sweep is unprepared, complete, or invalid");
    if(segment!=f->adjoint_sweep_stats.next_reverse_segment)
        return contract_failure("adjoint reverse segment",__FILE__,__LINE__,
            "expected segment %d, got %d",
            f->adjoint_sweep_stats.next_reverse_segment,segment);
    int begin,end;
    if(denise_cuda_psv_forward_segment_bounds(f,segment,&begin,&end)!=0)
        return -1;
    size_t syncs_before=f->stats.forward_synchronization_calls;
    if(denise_cuda_psv_forward_segment_replay(f,segment)!=0) {
        f->adjoint_sweep_state=ADJOINT_SWEEP_INVALID;
        f->adjoint_sweep_stats.invalid=1;
        f->adjoint_sweep_stats.prepared=0;
        return -1;
    }
    if(f->segment_record_ready!=segment+1) {
        f->adjoint_sweep_state=ADJOINT_SWEEP_INVALID;
        f->adjoint_sweep_stats.invalid=1;
        f->adjoint_sweep_stats.prepared=0;
        return contract_failure("adjoint reverse segment",__FILE__,__LINE__,
                                "replayed operand buffer is not ready");
    }
    f->adjoint_sweep_stats.replay_synchronization_calls+=
        f->stats.forward_synchronization_calls-syncs_before;
    for(int timestep=end;timestep>=begin;--timestep) {
        if(launch_adjoint_operator(f,timestep,true,begin)!=0) {
            f->adjoint_sweep_state=ADJOINT_SWEEP_INVALID;
            f->adjoint_sweep_stats.invalid=1;
            f->adjoint_sweep_stats.prepared=0;
            return -1;
        }
    }
    /* One segment-boundary synchronization makes the checkpointable API's
     * success result authoritative and catches asynchronous execution errors.
     * It is deliberately outside the timestep operator. */
    cudaError_t rc=cudaDeviceSynchronize();
    if(rc!=cudaSuccess) {
        f->adjoint_sweep_state=ADJOINT_SWEEP_INVALID;
        f->adjoint_sweep_stats.invalid=1;
        f->adjoint_sweep_stats.prepared=0;
        return cuda_failure("adjoint reverse segment synchronization",rc,
                            __FILE__,__LINE__);
    }
    ++f->adjoint_sweep_stats.reverse_segment_synchronization_calls;
    ++f->adjoint_sweep_stats.reverse_segments;
    f->adjoint_sweep_stats.reverse_timesteps+=(unsigned long long)(end-begin+1);
    f->adjoint_sweep_stats.next_reverse_segment=segment-1;
    if(segment==0) {
        f->adjoint_sweep_state=ADJOINT_SWEEP_COMPLETE;
        f->adjoint_sweep_stats.prepared=0;
        f->adjoint_sweep_stats.complete=1;
    }
    return 0;
}

int denise_cuda_psv_adjoint_reverse_sweep(denise_cuda_psv_forward *f) {
    psv_last_error[0]='\0';
    if(!f||f->adjoint_sweep_state!=ADJOINT_SWEEP_READY)
        return contract_failure("adjoint reverse sweep",__FILE__,__LINE__,
                                "sweep is unprepared, complete, or invalid");
    while(f->adjoint_sweep_state==ADJOINT_SWEEP_READY) {
        int segment=f->adjoint_sweep_stats.next_reverse_segment;
        if(denise_cuda_psv_adjoint_reverse_segment(f,segment)!=0) return -1;
    }
    return f->adjoint_sweep_state==ADJOINT_SWEEP_COMPLETE?0:-1;
}

int denise_cuda_psv_adjoint_sweep_get_stats(
        const denise_cuda_psv_forward *f,
        denise_cuda_psv_adjoint_sweep_stats *stats) {
    psv_last_error[0]='\0';
    if(!f||!stats||f->adjoint_sweep_state==ADJOINT_SWEEP_UNPREPARED)
        return contract_failure("adjoint sweep get stats",__FILE__,__LINE__,
                                "sweep is unprepared or output is null");
    *stats=f->adjoint_sweep_stats;
    return 0;
}

int denise_cuda_psv_native_gradient_required_bytes(
        const denise_cuda_psv_forward_config *config,
        denise_cuda_psv_native_gradient_stats *plan) {
    psv_last_error[0]='\0';
    return calculate_native_gradient_plan(config,plan);
}

int denise_cuda_psv_native_gradient_prepare(denise_cuda_psv_forward *f) {
    psv_last_error[0]='\0';
    if(!f)
        return contract_failure("native gradient prepare",__FILE__,__LINE__,
                                "context is null");
    if(f->native_gradient_storage||f->native_gradient_stats.prepared)
        return contract_failure("native gradient prepare",__FILE__,__LINE__,
                                "native gradient is already prepared");
    if(f->adjoint_sweep_state!=ADJOINT_SWEEP_READY||
       f->adjoint_sweep_stats.reverse_segments!=0||
       f->adjoint_sweep_stats.next_reverse_segment!=f->segment_count-1)
        return contract_failure("native gradient prepare",__FILE__,__LINE__,
            "a valid unstarted segmented reverse sweep is required");
    denise_cuda_psv_native_gradient_stats plan;
    if(calculate_native_gradient_plan(&f->config,&plan)!=0) return -1;
    if(plan.native_gradient_bytes>f->stats.remaining_budget_bytes)
        return contract_failure("native gradient prepare budget",__FILE__,__LINE__,
            "native gradients %zu exceed remaining usable budget %zu",
            plan.native_gradient_bytes,f->stats.remaining_budget_bytes);
    double *storage=nullptr;
    cudaError_t rc=cudaMalloc((void**)&storage,plan.native_gradient_bytes);
    if(rc!=cudaSuccess)
        return cuda_failure("native gradient cudaMalloc",rc,__FILE__,__LINE__);
    ++plan.allocation_calls;
    rc=cudaMemset(storage,0,plan.native_gradient_bytes);
    if(rc!=cudaSuccess) {
        cudaError_t free_rc=cudaFree(storage);
        if(free_rc!=cudaSuccess)
            return cuda_failure("native gradient failed-prepare cudaFree",free_rc,
                                __FILE__,__LINE__);
        return cuda_failure("native gradient zero",rc,__FILE__,__LINE__);
    }
    ++plan.zero_calls;
    f->native_gradient_storage=storage;
    assign_native_gradient_slices(f);
    f->stats.remaining_budget_bytes-=plan.native_gradient_bytes;
    f->core->impl.stats.remaining_budget_bytes=f->stats.remaining_budget_bytes;
    plan.remaining_budget_bytes=f->stats.remaining_budget_bytes;
    plan.prepared=1;
    f->native_gradient_stats=plan;
    return 0;
}

int denise_cuda_psv_native_gradient_download(
        denise_cuda_psv_forward *f,denise_cuda_psv_native_gradient_host *host,
        size_t elements_per_field) {
    psv_last_error[0]='\0';
    size_t cells;
    if(!f||!f->native_gradient_storage||!valid_native_gradient_host(host))
        return contract_failure("native gradient download",__FILE__,__LINE__,
                                "context is unprepared or output is invalid");
    if(!checked_mul((size_t)f->config.core.nx,(size_t)f->config.core.ny,&cells)||
       elements_per_field!=cells)
        return contract_failure("native gradient download",__FILE__,__LINE__,
            "each field must contain exactly NX*NY=%zu doubles",cells);
    const double *device[8]={f->native_gradient.gf,f->native_gradient.gg,
        f->native_gradient.gfc,f->native_gradient.gd,f->native_gradient.ge,
        f->native_gradient.gdc,f->native_gradient.grx,f->native_gradient.gry};
    double *output[8]={host->gf,host->gg,host->gfc,host->gd,host->ge,
        host->gdc,host->grx,host->gry};
    size_t bytes=cells*sizeof(double);
    for(int field=0;field<8;++field) {
        cudaError_t rc=cudaMemcpy(output[field],device[field],bytes,
                                  cudaMemcpyDeviceToHost);
        if(rc!=cudaSuccess)
            return cuda_failure("native gradient diagnostic D2H",rc,
                                __FILE__,__LINE__);
        ++f->native_gradient_stats.diagnostic_d2h_calls;
        f->native_gradient_stats.diagnostic_d2h_bytes+=bytes;
    }
    return 0;
}

int denise_cuda_psv_native_gradient_get_stats(
        const denise_cuda_psv_forward *f,
        denise_cuda_psv_native_gradient_stats *stats) {
    psv_last_error[0]='\0';
    if(!f||!f->native_gradient_storage||!stats)
        return contract_failure("native gradient get stats",__FILE__,__LINE__,
                                "context is unprepared or output is null");
    *stats=f->native_gradient_stats;
    return 0;
}

int denise_cuda_psv_physical_gradient_required_bytes(
        const denise_cuda_psv_forward_config *config,
        denise_cuda_psv_physical_gradient_stats *plan) {
    psv_last_error[0]='\0';
    return calculate_physical_gradient_plan(config,plan);
}

int denise_cuda_psv_physical_gradient_prepare(
        denise_cuda_psv_forward *f,
        const denise_cuda_psv_physical_material_host *material) {
    psv_last_error[0]='\0';
    if(!f||!valid_physical_material_host(material))
        return contract_failure("physical gradient prepare",__FILE__,__LINE__,
                                "context or material adapter is invalid");
    if(f->physical_gradient_storage||f->physical_gradient_stats.prepared)
        return contract_failure("physical gradient prepare",__FILE__,__LINE__,
                                "physical gradient is already prepared");
    if(!f->native_gradient_storage||!f->native_gradient_stats.prepared)
        return contract_failure("physical gradient prepare",__FILE__,__LINE__,
                                "native gradient must be prepared first");
    if(f->adjoint_sweep_state!=ADJOINT_SWEEP_READY||
       f->adjoint_sweep_stats.reverse_segments!=0||
       f->adjoint_sweep_stats.next_reverse_segment!=f->segment_count-1)
        return contract_failure("physical gradient prepare",__FILE__,__LINE__,
            "a valid unstarted segmented reverse sweep is required");
    if(!std::isfinite((double)material->eta)||
       (material->q_parameterization_mode!=0&&
        material->q_parameterization_mode!=1))
        return contract_failure("physical gradient prepare",__FILE__,__LINE__,
                                "eta or Q parameterization mode is invalid");
    if(material->q_parameterization_mode==1&&
       (!std::isfinite(material->inverse_tau_per_q)||
        !std::isfinite(material->inverse_tau_offset)||
        material->inverse_tau_per_q<=0.0))
        return contract_failure("physical gradient prepare",__FILE__,__LINE__,
                                "physical-Q mapping coefficients are invalid");
    denise_cuda_psv_physical_gradient_stats plan;
    if(calculate_physical_gradient_plan(&f->config,&plan)!=0) return -1;
    if(plan.total_bytes>f->stats.remaining_budget_bytes)
        return contract_failure("physical gradient prepare budget",__FILE__,__LINE__,
            "material and physical gradients %zu exceed remaining usable budget %zu",
            plan.total_bytes,f->stats.remaining_budget_bytes);

    unsigned char *storage=nullptr;
    cudaError_t rc=cudaMalloc((void**)&storage,plan.total_bytes);
    if(rc!=cudaSuccess)
        return cuda_failure("physical gradient aggregate cudaMalloc",rc,
                            __FILE__,__LINE__);
    ++plan.allocation_calls;
    size_t cells=(size_t)f->config.core.nx*(size_t)f->config.core.ny;
    size_t field_bytes=cells*sizeof(float);
    size_t row_bytes=(size_t)f->config.core.nx*sizeof(float);
    size_t host_pitch=(size_t)(f->config.core.nx+6)*sizeof(float);
    const float *source[7]={&material->prho[1][1],&material->ppi[1][1],
        &material->pu[1][1],&material->ptaus[1][1],&material->ptaup[1][1],
        &material->puipjp[1][1],&material->ptausipjp[1][1]};
    float *destination=(float*)storage;
    for(int field=0;field<7;++field) {
        rc=cudaMemcpy2D(destination,row_bytes,source[field],host_pitch,
                        row_bytes,(size_t)f->config.core.ny,
                        cudaMemcpyHostToDevice);
        if(rc!=cudaSuccess) goto fail;
        ++plan.material_h2d_calls;
        plan.material_h2d_bytes+=field_bytes;
        destination+=cells;
    }
    rc=cudaMemset(storage+plan.material_bytes+plan.alignment_bytes,0,
                  plan.physical_gradient_bytes+plan.validation_bytes);
    if(rc!=cudaSuccess) goto fail;
    ++plan.zero_calls;

    f->physical_gradient_storage=storage;
    assign_physical_gradient_slices(f,plan);
    f->physical_eta=material->eta;
    f->physical_q_mode=material->q_parameterization_mode;
    f->physical_q_per=material->inverse_tau_per_q;
    f->physical_q_offset=material->inverse_tau_offset;
    f->stats.remaining_budget_bytes-=plan.total_bytes;
    f->core->impl.stats.remaining_budget_bytes=f->stats.remaining_budget_bytes;
    plan.remaining_budget_bytes=f->stats.remaining_budget_bytes;
    plan.prepared=1;
    f->physical_gradient_stats=plan;
    return 0;
fail:
    {
        cudaError_t free_rc=cudaFree(storage);
        if(free_rc!=cudaSuccess)
            return cuda_failure("physical gradient failed-prepare cudaFree",free_rc,
                                __FILE__,__LINE__);
    }
    return cuda_failure("physical gradient material upload/zero",rc,
                        __FILE__,__LINE__);
}

int denise_cuda_psv_physical_gradient_map(denise_cuda_psv_forward *f) {
    psv_last_error[0]='\0';
    if(!f||!f->physical_gradient_storage||
       !f->physical_gradient_stats.prepared)
        return contract_failure("physical gradient map",__FILE__,__LINE__,
                                "physical gradient is unprepared");
    if(f->physical_gradient_stats.mapped||f->physical_gradient_stats.invalid)
        return contract_failure("physical gradient map",__FILE__,__LINE__,
                                "physical gradient is already mapped or invalid");
    if(f->adjoint_sweep_state!=ADJOINT_SWEEP_COMPLETE||
       !f->adjoint_sweep_stats.complete)
        return contract_failure("physical gradient map",__FILE__,__LINE__,
                                "a complete reverse sweep is required");
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    dim3 block(32,4),grid((c.config.nx+31)/32,(c.config.ny+3)/4);
    physical_gradient_map_kernel<<<grid,block>>>(
        f->native_gradient,f->physical_material,f->physical_gradient,c.fields,
        c.config.nx,c.config.ny,c.full_nx,c.config.dt,f->physical_eta,
        f->physical_q_mode,f->physical_q_per,f->physical_q_offset,
        f->physical_invalid_device);
    cudaError_t rc=cudaGetLastError();
    if(rc!=cudaSuccess) goto fail;
    ++f->physical_gradient_stats.map_kernel_launches;
    rc=cudaDeviceSynchronize();
    ++f->physical_gradient_stats.map_synchronization_calls;
    if(rc!=cudaSuccess) goto fail;
    {
        int invalid=0;
        rc=cudaMemcpy(&invalid,f->physical_invalid_device,sizeof(invalid),
                      cudaMemcpyDeviceToHost);
        if(rc!=cudaSuccess) goto fail;
        ++f->physical_gradient_stats.validation_d2h_calls;
        f->physical_gradient_stats.validation_d2h_bytes+=sizeof(invalid);
        if(invalid) {
            f->physical_gradient_stats.invalid=1;
            return contract_failure("physical gradient Q validation",
                __FILE__,__LINE__,
                "reconstructed Qp/Qs is non-positive or non-finite");
        }
    }
    f->physical_gradient_stats.mapped=1;
    return 0;
fail:
    f->physical_gradient_stats.invalid=1;
    return cuda_failure("physical gradient map",rc,__FILE__,__LINE__);
}

int denise_cuda_psv_physical_gradient_download(
        denise_cuda_psv_forward *f,
        denise_cuda_psv_physical_gradient_host *host,
        size_t elements_per_field) {
    psv_last_error[0]='\0';
    size_t cells;
    if(!f||!f->physical_gradient_storage||
       !f->physical_gradient_stats.mapped||
       f->physical_gradient_stats.invalid||
       !valid_physical_gradient_host(host))
        return contract_failure("physical gradient download",__FILE__,__LINE__,
                                "mapping is incomplete/invalid or output is invalid");
    if(!checked_mul((size_t)f->config.core.nx,(size_t)f->config.core.ny,&cells)||
       elements_per_field!=cells)
        return contract_failure("physical gradient download",__FILE__,__LINE__,
            "each field must contain exactly NX*NY=%zu doubles",cells);
    const double *device[5]={f->physical_gradient.vp,f->physical_gradient.vs,
        f->physical_gradient.rho,f->physical_gradient.qp,f->physical_gradient.qs};
    double *output[5]={host->vp,host->vs,host->rho,host->qp,host->qs};
    size_t bytes=cells*sizeof(double);
    for(int field=0;field<5;++field) {
        cudaError_t rc=cudaMemcpy(output[field],device[field],bytes,
                                  cudaMemcpyDeviceToHost);
        if(rc!=cudaSuccess)
            return cuda_failure("physical gradient diagnostic D2H",rc,
                                __FILE__,__LINE__);
        ++f->physical_gradient_stats.diagnostic_d2h_calls;
        f->physical_gradient_stats.diagnostic_d2h_bytes+=bytes;
    }
    return 0;
}

int denise_cuda_psv_physical_gradient_get_stats(
        const denise_cuda_psv_forward *f,
        denise_cuda_psv_physical_gradient_stats *stats) {
    psv_last_error[0]='\0';
    if(!f||!f->physical_gradient_storage||!stats)
        return contract_failure("physical gradient get stats",__FILE__,__LINE__,
                                "context is unprepared or output is null");
    *stats=f->physical_gradient_stats;
    return 0;
}

int denise_cuda_psv_forward_run(denise_cuda_psv_forward *f) {
    psv_last_error[0]='\0';
    if(!f) return contract_failure("forward run",__FILE__,__LINE__,"context is null");
    if(f->core->impl.stats.timesteps||f->next_timestep!=1)
        return contract_failure("forward run",__FILE__,__LINE__,"context has already run");
    size_t event_count;
    if(f->profiling_enabled) {
        size_t phase_events;
        if(!checked_mul((size_t)f->config.nt,4,&phase_events)||
           !checked_add(phase_events,1,&event_count))
            return contract_failure("forward run",__FILE__,__LINE__,
                                    "profiling event count overflows size_t");
    } else event_count=2;
    cudaEvent_t *events=new(std::nothrow) cudaEvent_t[event_count]();
    if(!events) return contract_failure("forward run",__FILE__,__LINE__,
                                        "profiling event allocation failed");
    cudaError_t rc=cudaSuccess;
    size_t created=0;
    for(;created<event_count;++created) {
        rc=cudaEventCreate(&events[created]);
        if(rc!=cudaSuccess) {
            destroy_event_array(events,event_count);
            return cuda_failure("forward kernel event create",rc,__FILE__,__LINE__);
        }
    }
    denise_cuda_psv_fd4_l1_impl &c=f->core->impl;
    rc=cudaEventRecord(events[0]);
    if(rc!=cudaSuccess) goto fail;
    if(f->profiling_enabled) ++f->stats.profile_event_records;
    else ++f->stats.resident_event_records;
    for(int nt=1;nt<=f->config.nt;++nt) {
        size_t h2d_before=c.stats.h2d_transfer_calls;
        size_t d2h_before=c.stats.d2h_transfer_calls;
        size_t event=(size_t)(nt-1)*4;
        if(launch_velocity(&c)!=0) goto launch_fail;
        if(f->profiling_enabled) {
            rc=cudaEventRecord(events[event+1]); if(rc!=cudaSuccess) goto fail;
            ++f->stats.profile_event_records;
        }
        if(launch_stress(&c)!=0) goto launch_fail;
        if(f->profiling_enabled) {
            rc=cudaEventRecord(events[event+2]); if(rc!=cudaSuccess) goto fail;
            ++f->stats.profile_event_records;
        }
        if(launch_source(f,nt)!=0) goto launch_fail;
        if(f->profiling_enabled) {
            rc=cudaEventRecord(events[event+3]); if(rc!=cudaSuccess) goto fail;
            ++f->stats.profile_event_records;
        }
        if(launch_receivers(f,nt)!=0) goto launch_fail;
        if(f->profiling_enabled) {
            rc=cudaEventRecord(events[event+4]); if(rc!=cudaSuccess) goto fail;
            ++f->stats.profile_event_records;
        }
        ++c.stats.timesteps;
        if(c.stats.h2d_transfer_calls!=h2d_before||
           c.stats.d2h_transfer_calls!=d2h_before) {
            contract_failure("forward timestep residency",__FILE__,__LINE__,
                "host/device transfer detected during timestep %d",nt);
            goto fail;
        }
    }
    if(!f->profiling_enabled) {
        rc=cudaEventRecord(events[1]); if(rc!=cudaSuccess) goto fail;
        ++f->stats.resident_event_records;
    }
    rc=cudaEventSynchronize(events[event_count-1]); if(rc!=cudaSuccess) goto fail;
    ++f->stats.forward_synchronization_calls;
    rc=cudaEventElapsedTime(&f->stats.resident_timestep_ms,
                            events[0],events[event_count-1]);
    if(rc!=cudaSuccess) goto fail;
    ++f->stats.resident_elapsed_queries;
    if(f->profiling_enabled) {
        for(int nt=1;nt<=f->config.nt;++nt) {
            size_t event=(size_t)(nt-1)*4;
            float vm=0.0f,sm=0.0f,qm=0.0f,rm=0.0f;
            rc=cudaEventElapsedTime(&vm,events[event],events[event+1]);
            if(rc==cudaSuccess) rc=cudaEventElapsedTime(&sm,events[event+1],events[event+2]);
            if(rc==cudaSuccess) rc=cudaEventElapsedTime(&qm,events[event+2],events[event+3]);
            if(rc==cudaSuccess) rc=cudaEventElapsedTime(&rm,events[event+3],events[event+4]);
            if(rc!=cudaSuccess) goto fail;
            f->stats.profile_elapsed_queries+=4;
            c.stats.velocity_kernel_ms+=vm; c.stats.stress_kernel_ms+=sm;
            c.stats.combined_kernel_ms+=vm+sm;
            f->stats.source_kernel_ms+=qm; f->stats.receiver_kernel_ms+=rm;
        }
    }
    destroy_event_array(events,event_count);
    f->next_timestep=f->config.nt+1;
    refresh_forward_stats(f);
    return 0;
fail:
    destroy_event_array(events,event_count);
    if(rc!=cudaSuccess) return cuda_failure("forward timestep",rc,__FILE__,__LINE__);
    return -1;
launch_fail:
    destroy_event_array(events,event_count);
    return -1;
}

int denise_cuda_psv_forward_download_traces(denise_cuda_psv_forward *f,
                                             float **sectionvx,
                                             float **sectionvy) {
    psv_last_error[0]='\0';
    if(!f||!sectionvx||!sectionvy)
        return contract_failure("trace download",__FILE__,__LINE__,"output is null");
    size_t elements=(size_t)f->config.ntr*(size_t)f->config.nt;
    cudaEvent_t a=nullptr,b=nullptr;
    cudaError_t rc=cudaEventCreate(&a);
    if(rc==cudaSuccess) rc=cudaEventCreate(&b);
    if(rc!=cudaSuccess) { if(a) cudaEventDestroy(a); return cuda_failure(
        "trace download event create",rc,__FILE__,__LINE__); }
    rc=cudaEventRecord(a);
    if(rc==cudaSuccess) rc=d2h(&sectionvx[1][1],f->trace_vx,elements,&f->core->impl.stats);
    if(rc==cudaSuccess) rc=d2h(&sectionvy[1][1],f->trace_vy,elements,&f->core->impl.stats);
    if(rc==cudaSuccess) rc=cudaEventRecord(b);
    if(rc==cudaSuccess) rc=cudaEventSynchronize(b);
    float ms=0.0f;
    if(rc==cudaSuccess) rc=cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(b); cudaEventDestroy(a);
    if(rc!=cudaSuccess) return cuda_failure("trace download",rc,__FILE__,__LINE__);
    f->stats.trace_download_ms+=ms; refresh_forward_stats(f); return 0;
}

int denise_cuda_psv_forward_download_mutable(
        denise_cuda_psv_forward *f,
        denise_cuda_psv_fd4_l1_host *host) {
    psv_last_error[0]='\0';
    if(!f) return contract_failure("forward mutable download",__FILE__,__LINE__,
                                   "context is null");
    float before=f->core->impl.stats.download_ms;
    if(denise_cuda_psv_fd4_l1_download_mutable(f->core,host)!=0) return -1;
    f->stats.mutable_download_ms+=f->core->impl.stats.download_ms-before;
    refresh_forward_stats(f); return 0;
}

int denise_cuda_psv_forward_get_stats(const denise_cuda_psv_forward *f,
                                       denise_cuda_psv_forward_stats *stats) {
    psv_last_error[0]='\0';
    if(!f||!stats) return contract_failure("forward get stats",__FILE__,__LINE__,
                                           "context or output is null");
    denise_cuda_psv_forward *mutable_f=const_cast<denise_cuda_psv_forward*>(f);
    refresh_forward_stats(mutable_f); *stats=mutable_f->stats; return 0;
}

int denise_cuda_psv_forward_destroy(denise_cuda_psv_forward **handle) {
    psv_last_error[0]='\0';
    if(!handle) return 0;
    denise_cuda_psv_forward *f=*handle; *handle=nullptr;
    if(!f) return 0;
    cudaError_t checkpoint_rc=f->checkpoint_storage?
        cudaFree(f->checkpoint_storage):cudaSuccess;
    cudaError_t segment_rc=f->segment_storage?
        cudaFree(f->segment_storage):cudaSuccess;
    cudaError_t adjoint_rc=f->adjoint_storage?
        cudaFree(f->adjoint_storage):cudaSuccess;
    cudaError_t observed_rc=f->adjoint_observed_storage?
        cudaFree(f->adjoint_observed_storage):cudaSuccess;
    cudaError_t gradient_rc=f->native_gradient_storage?
        cudaFree(f->native_gradient_storage):cudaSuccess;
    cudaError_t physical_rc=f->physical_gradient_storage?
        cudaFree(f->physical_gradient_storage):cudaSuccess;
    int result=denise_cuda_psv_fd4_l1_destroy(&f->core);
    delete f;
    if(checkpoint_rc!=cudaSuccess)
        return cuda_failure("checkpoint destroy cudaFree",checkpoint_rc,
                            __FILE__,__LINE__);
    if(segment_rc!=cudaSuccess)
        return cuda_failure("segment destroy cudaFree",segment_rc,
                            __FILE__,__LINE__);
    if(adjoint_rc!=cudaSuccess)
        return cuda_failure("adjoint destroy cudaFree",adjoint_rc,
                            __FILE__,__LINE__);
    if(observed_rc!=cudaSuccess)
        return cuda_failure("adjoint observed destroy cudaFree",observed_rc,
                            __FILE__,__LINE__);
    if(gradient_rc!=cudaSuccess)
        return cuda_failure("native gradient destroy cudaFree",gradient_rc,
                            __FILE__,__LINE__);
    if(physical_rc!=cudaSuccess)
        return cuda_failure("physical gradient destroy cudaFree",physical_rc,
                            __FILE__,__LINE__);
    return result;
}

}  // extern "C"
