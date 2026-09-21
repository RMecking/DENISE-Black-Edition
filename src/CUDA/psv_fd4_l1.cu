#include "denise_cuda_psv.h"
#include "denise_cuda_backend.h"
#include <cuda_runtime.h>
#include <climits>
#include <cstdarg>
#include <cstdint>
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
                             float dt,float dh,float hc1,float hc2) {
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
    d.vx[q]+=dt*d.rip[q]*(sxx_x+sxy_y)/dh;
    d.vy[q]+=dt*d.rjp[q]*(sxy_x+syy_y)/dh;
#undef A
}

__global__ void stress_fd4_l1(device_fields d,int nx,int ny,int fw,size_t pitch,
                              float dt,float dh,float hc1,float hc2,
                              float bip1,float bjm1,float cip1,float cjm1) {
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
    float oldr=d.r1[q],oldp=d.p1[q],oldq=d.q1[q];
    d.sxy[q]+=d.fipjp[q]*(vxy+vyx)+dth*oldr;
    d.sxx[q]+=d.g[q]*(vxx+vyy)-(2.0*d.f[q]*vyy)+dth*oldp;
    d.syy[q]+=d.g[q]*(vxx+vyy)-(2.0*d.f[q]*vxx)+dth*oldq;
    float newr=bip1*(oldr*cip1-d.dip1[q]*(vxy+vyx));
    float newp=bjm1*(oldp*cjm1-d.e1[q]*(vxx+vyy)+(2.0*d.d1[q]*vyy));
    float newq=bjm1*(oldq*cjm1-d.e1[q]*(vxx+vyy)+(2.0*d.d1[q]*vxx));
    d.r1[q]=newr; d.p1[q]=newp; d.q1[q]=newq;
    d.sxy[q]+=dth*newr; d.sxx[q]+=dth*newp; d.syy[q]+=dth*newq;
#undef A
}

int timed_velocity(denise_cuda_psv_fd4_l1_impl *c,cudaEvent_t a,cudaEvent_t b,float *ms) {
    dim3 block(16,16),grid((c->config.nx+15)/16,(c->config.ny+15)/16);
    PSV_CUDA_CALL("velocity event start",cudaEventRecord(a));
    velocity_fd4<<<grid,block>>>(c->fields,c->config.nx,c->config.ny,c->config.fw,
        c->full_nx,c->config.dt,c->config.dh,c->config.hc1,c->config.hc2);
    PSV_CUDA_CALL("velocity kernel launch",cudaGetLastError());
    PSV_CUDA_CALL("velocity event stop",cudaEventRecord(b));
    PSV_CUDA_CALL("velocity kernel completion",cudaEventSynchronize(b));
    PSV_CUDA_CALL("velocity elapsed",cudaEventElapsedTime(ms,a,b));
    return 0;
}
int timed_stress(denise_cuda_psv_fd4_l1_impl *c,cudaEvent_t a,cudaEvent_t b,float *ms) {
    dim3 block(16,16),grid((c->config.nx+15)/16,(c->config.ny+15)/16);
    PSV_CUDA_CALL("stress event start",cudaEventRecord(a));
    stress_fd4_l1<<<grid,block>>>(c->fields,c->config.nx,c->config.ny,c->config.fw,
        c->full_nx,c->config.dt,c->config.dh,c->config.hc1,c->config.hc2,
        c->config.bip1,c->config.bjm1,c->config.cip1,c->config.cjm1);
    PSV_CUDA_CALL("stress kernel launch",cudaGetLastError());
    PSV_CUDA_CALL("stress event stop",cudaEventRecord(b));
    PSV_CUDA_CALL("stress kernel completion",cudaEventSynchronize(b));
    PSV_CUDA_CALL("stress elapsed",cudaEventElapsedTime(ms,a,b));
    return 0;
}

}  // namespace

struct denise_cuda_psv_fd4_l1 { denise_cuda_psv_fd4_l1_impl impl; };

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
