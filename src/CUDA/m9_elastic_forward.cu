/* Isolated M9 elastic FP32 forward. No M8e numerical code or CUDA dispatch. */
#include "denise_cuda_m9_elastic.h"
#include "m9_elastic_host.h"
#ifndef DENISE_M9_HOST_SANITIZER_ONLY
#include <cuda_runtime.h>
#endif
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
static_assert(sizeof(float)==4 && sizeof(int)==4,"M9 verification ABI requires FP32/int32");

static char error_text[512];
static size_t fault_at, calls, device_owned, host_owned, events_owned;
static int error(const char *fmt, ...) {
    if (!error_text[0]) {
        va_list ap; va_start(ap,fmt); vsnprintf(error_text,sizeof(error_text),fmt,ap); va_end(ap);
    }
    return -1;
}
static void begin() { error_text[0]=0; }
static bool gate(const char *name) {
    ++calls;
    if (fault_at && calls==fault_at) { error("injected M9 CUDA failure at %s (%zu)",name,calls); return false; }
    return true;
}
#ifndef DENISE_M9_HOST_SANITIZER_ONLY
static bool checked(cudaError_t rc,const char *name) {
    if(rc!=cudaSuccess) { error("M9 CUDA %s: %s",name,cudaGetErrorString(rc)); return false; }
    return true;
}
#define CUDA(call) (gate(#call) && checked((call),#call))
#endif
union HostHeader { size_t bytes; long double alignment; void *pointer; };
extern "C" void *m9_host_calloc(size_t count,size_t width) {
    if(width && count>SIZE_MAX/width) { error("M9 host allocation overflow"); return NULL; }
    size_t bytes=count*width;
    if(bytes>SIZE_MAX-sizeof(HostHeader) || !gate("host allocation")) return NULL;
    HostHeader *p=(HostHeader *)calloc(1,bytes+sizeof(HostHeader));
    if(!p) { error("M9 host allocation unavailable (%zu bytes)",bytes); return NULL; }
    p->bytes=bytes;host_owned+=bytes;return p+1;
}
extern "C" void m9_host_free(void *p) {
    if(p) { HostHeader *h=(HostHeader *)p-1;host_owned-=h->bytes;free(h); }
}
extern "C" const char *denise_cuda_m9_last_error(void) {return error_text[0]?error_text:"no M9 CUDA error";}
extern "C" void denise_cuda_m9_fault(size_t at) {fault_at=at;calls=0;begin();}
extern "C" void denise_cuda_m9_ledger(size_t *d,size_t *h,size_t *e,size_t *n) {
    if(d)*d=device_owned;
    if(h)*h=host_owned;
    if(e)*e=events_owned;
    if(n)*n=calls;
}
#ifndef DENISE_M9_HOST_SANITIZER_ONLY
static bool mul(size_t a,size_t b,size_t &v) { if(a && b>SIZE_MAX/a)return false;v=a*b;return true; }
static bool add(size_t a,size_t b,size_t &v) { if(b>SIZE_MAX-a)return false;v=a+b;return true; }

/* Kernel argument view: host metadata only; all pointers refer to one arena. */
struct View {
    int nx,ny,w,nt,nr,si,sj;
    size_t cells,padded;
    float coefficient;
    float *field,*psi,*map,*profile,*source,*data,*strain;
    int *geometry;
};
struct denise_cuda_m9 {
    View v;
    struct denise_cuda_m9_diagnostics d;
    m9_host *host;
    void *arena;
    cudaEvent_t start,end;
    int device,valid;
};
__device__ static float fd(const float *f,size_t p,int w,int kind) {
    const float a=9.0f/8.0f,b=-1.0f/24.0f;
    if(kind==0)return a*(f[p]-f[p-1])+b*(f[p+1]-f[p-2]);
    if(kind==1)return a*(f[p+1]-f[p])+b*(f[p+2]-f[p-1]);
    if(kind==2)return a*(f[p]-f[p-w])+b*(f[p+w]-f[p-2*w]);
    return a*(f[p+w]-f[p])+b*(f[p+2*w]-f[p-w]);
}
__device__ static float pml(View v,int kind,int i,int j,size_t p,float q) {
    int prof=(kind==0 || kind==5)?1:((kind==1 || kind==7)?2:((kind==2 || kind==4)?0:3));
    int n=prof<2?v.nx:v.ny,k=prof<2?i:j;
    size_t offset=prof==0?0:(prof==1?3*(size_t)v.nx:(prof==2?6*(size_t)v.nx:6*(size_t)v.nx+3*(size_t)v.ny));
    const float *a=v.profile+offset;
    float *memory=v.psi+(size_t)kind*v.padded+p;
    float next=a[2*n+k]*(*memory)+a[n+k]*q;
    *memory=next;return q/a[k]+next;
}
__global__ static void halo_x(View v,int first,int count) {
    size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=(size_t)count*v.ny*4)return;
    int h=t%4,j=(t/4)%v.ny,k=t/(4*(size_t)v.ny)+first;
    int dst=h<2?h:v.nx+h,src=h<2?v.nx+h:h;
    float *f=v.field+(size_t)k*v.padded+(size_t)(j+2)*v.w;
    f[dst]=f[src];
}
__global__ static void halo_y(View v,int first,int count) {
    size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=(size_t)count*v.w*4)return;
    int h=t%4,i=(t/4)%v.w,k=t/(4*(size_t)v.w)+first;
    int dst=h<2?h:v.ny+h,src=h<2?v.ny+h:h;
    float *f=v.field+(size_t)k*v.padded;
    f[(size_t)dst*v.w+i]=f[(size_t)src*v.w+i];
}
__global__ static void momentum(View v) {
    size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.cells)return;
    int i=t%v.nx,j=t/v.nx;size_t p=(size_t)(j+2)*v.w+i+2;
    const float *sxx=v.field+2*v.padded,*syy=v.field+3*v.padded,*sxy=v.field+4*v.padded;
    float xx=pml(v,0,i,j,p,v.coefficient*fd(sxx,p,v.w,1));
    float xy=pml(v,1,i,j,p,v.coefficient*fd(sxy,p,v.w,2));
    float yx=pml(v,2,i,j,p,v.coefficient*fd(sxy,p,v.w,0));
    float yy=pml(v,3,i,j,p,v.coefficient*fd(syy,p,v.w,3));
    v.field[p]+=v.map[2*v.padded+p]*(xx+xy);
    v.field[v.padded+p]+=v.map[3*v.padded+p]*(yx+yy);
}
__global__ static void sample(View v,int timestep) {
    int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=v.nr)return;
    size_t p=(size_t)(v.geometry[3*r+1]+2)*v.w+v.geometry[3*r]+2;
    size_t o=((size_t)timestep*v.nr+v.geometry[3*r+2])*2;
    v.data[o]=v.field[p];v.data[o+1]=v.field[v.padded+p];
}
__global__ static void strains(View v,int timestep) {
    size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.cells)return;
    int i=t%v.nx,j=t/v.nx;size_t p=(size_t)(j+2)*v.w+i+2;
    float *q=v.strain+(size_t)timestep*4*v.cells;
    q[t]=pml(v,4,i,j,p,v.coefficient*fd(v.field,p,v.w,0));
    q[v.cells+t]=pml(v,5,i,j,p,v.coefficient*fd(v.field+v.padded,p,v.w,1));
    q[2*v.cells+t]=pml(v,6,i,j,p,v.coefficient*fd(v.field,p,v.w,3));
    q[3*v.cells+t]=pml(v,7,i,j,p,v.coefficient*fd(v.field+v.padded,p,v.w,2));
}
__global__ static void stress(View v,int timestep) {
    size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.cells)return;
    size_t p=(t/v.nx+2)*v.w+t%v.nx+2;
    const float *q=v.strain+(size_t)timestep*4*v.cells;
    float div=q[t]+q[3*v.cells+t],l=v.map[p],m=v.map[v.padded+p];
    v.field[2*v.padded+p]+=l*div+2.0f*m*q[t];
    v.field[3*v.padded+p]+=l*div+2.0f*m*q[3*v.cells+t];
    v.field[4*v.padded+p]+=v.map[4*v.padded+p]*(q[v.cells+t]+q[2*v.cells+t]);
}
__global__ static void source(View v,int timestep) {
    size_t p=(size_t)(v.sj+2)*v.w+v.si+2;
    v.field[2*v.padded+p]+=v.source[timestep];v.field[3*v.padded+p]+=v.source[timestep];
}
static void invalidate(denise_cuda_m9 *c) { if(c){c->valid=0;c->d.valid_steps=0;c->d.prepared=0;} }
static bool select(denise_cuda_m9 *c) { return c && CUDA(cudaSetDevice(c->device)); }
static bool launch_check() { return CUDA(cudaGetLastError()); }
static unsigned blocks(size_t n) { return (unsigned)((n+127)/128); }
static bool halos(denise_cuda_m9 *c,int velocity) {
    View v=c->v;int first=velocity?0:2,count=velocity?2:3;
    if(!gate("halo X launch"))return false;
    halo_x<<<blocks((size_t)count*v.ny*4),128>>>(v,first,count);if(!launch_check())return false;
    if(!gate("halo Y launch"))return false;
    halo_y<<<blocks((size_t)count*v.w*4),128>>>(v,first,count);return launch_check();
}
static bool step(denise_cuda_m9 *c,int t) {
    View v=c->v;
    if(!halos(c,0) || !gate("momentum launch"))return false;
    momentum<<<blocks(v.cells),128>>>(v);if(!launch_check() || !gate("sample launch"))return false;
    sample<<<blocks(v.nr),128>>>(v,t);if(!launch_check() || !halos(c,1) || !gate("strain launch"))return false;
    strains<<<blocks(v.cells),128>>>(v,t);if(!launch_check() || !gate("stress launch"))return false;
    stress<<<blocks(v.cells),128>>>(v,t);if(!launch_check() || !gate("source launch"))return false;
    source<<<1,1>>>(v,t);return launch_check() && CUDA(cudaDeviceSynchronize());
}
static bool zero(denise_cuda_m9 *c) {
    View v=c->v;
    return CUDA(cudaMemset(v.field,0,c->d.wavefield_bytes+c->d.cpml_bytes)) &&
           CUDA(cudaMemset(v.data,0,(size_t)v.nt*v.nr*2*sizeof(float))) &&
           CUDA(cudaMemset(v.strain,0,c->d.trajectory_bytes));
}
static bool upload(denise_cuda_m9 *c,const m9_host *h,bool fixed) {
    float *maps=(float *)m9_host_calloc(c->d.model_bytes,1);
    float *profiles=fixed?(float *)m9_host_calloc(c->d.profile_bytes,1):NULL;
    int *geometry=fixed?(int *)m9_host_calloc((size_t)c->v.nr*3,sizeof(int)):NULL;
    bool ok=maps && (!fixed || (profiles && geometry));
    if(ok) {
        m9_host_maps(h,maps);
        ok=CUDA(cudaMemcpy(c->v.map,maps,c->d.model_bytes,cudaMemcpyHostToDevice));
    }
    if(ok && fixed) {
        const denise_elastic_psv_born_config *q=m9_host_config(h);
        m9_host_profiles(h,profiles);
        for(int r=0;r<q->receiver_count;r++){geometry[3*r]=q->receiver_i[r];geometry[3*r+1]=q->receiver_j[r];geometry[3*r+2]=r;}
        ok=CUDA(cudaMemcpy(c->v.profile,profiles,c->d.profile_bytes,cudaMemcpyHostToDevice)) &&
           CUDA(cudaMemcpy(c->v.source,q->source_samples,c->d.source_bytes,cudaMemcpyHostToDevice)) &&
           CUDA(cudaMemcpy(c->v.geometry,geometry,(size_t)q->receiver_count*3*sizeof(int),cudaMemcpyHostToDevice));
    }
    m9_host_free(maps);m9_host_free(profiles);m9_host_free(geometry);return ok;
}
static int finish_failure(denise_cuda_m9 *c) { invalidate(c);return error("M9 CUDA operation failed"); }
extern "C" int denise_cuda_m9_destroy(denise_cuda_m9 **out) {
    if(!out || !*out)return 0;denise_cuda_m9 *c=*out;
    /* Cleanup is not injected; retain handles when the runtime rejects cleanup. */
    if(!checked(cudaSetDevice(c->device),"destroy device"))return -1;
    if(c->start){if(!checked(cudaEventDestroy(c->start),"destroy start event"))return -1;c->start=NULL;--events_owned;}
    if(c->end){if(!checked(cudaEventDestroy(c->end),"destroy end event"))return -1;c->end=NULL;--events_owned;}
    if(c->arena){if(!checked(cudaFree(c->arena),"free arena"))return -1;device_owned-=c->d.owned_bytes;c->arena=NULL;}
    m9_host_destroy(&c->host);m9_host_free(c);*out=NULL;return 0;
}
extern "C" int denise_cuda_m9_create(const denise_elastic_psv_born_config *q,
    const denise_cuda_m9_options *options,denise_cuda_m9 **out) {
    begin();if(!out)return error("M9 CUDA output context is null");*out=NULL;
    if(!q)return error("M9 CUDA configuration is null");
    if(q->free_surface!=0)return error("M9 CUDA requires FREE_SURF=0 before mutation");
    if(q->nx<5 || q->ny<5 || q->nt<1 || q->receiver_count<1 || q->nx>INT_MAX-4 || q->ny>INT_MAX-4)
        return error("M9 CUDA dimensions invalid or overflow");
    if(q->fw<0 || q->fw>(q->nx-1)/2 || q->fw>(q->ny-1)/2 || !std::isfinite(q->dh) || !std::isfinite(q->dt))
        return error("M9 CUDA FW/DH/DT invalid");
    size_t cells,padded,trajectory,data_count,tmp,total=0,prof_count;
    if(!mul(q->nx,q->ny,cells) || cells>INT_MAX || !mul(q->nx+4,q->ny+4,padded) ||
       !mul(cells,4,trajectory) || !mul(trajectory,q->nt,trajectory) || !mul(trajectory,4,trajectory) ||
       !mul(q->nt,q->receiver_count,data_count) || !mul(data_count,8,data_count) ||
       !add(q->nx,q->ny,prof_count) || !mul(prof_count,24,prof_count))return error("M9 CUDA size overflow");
    size_t wave,psi,model,src,geom;
    if(!mul(padded,20,wave) || !mul(padded,32,psi) || !mul(padded,20,model) ||
       !mul(q->nt,4,src) || !mul(q->receiver_count,12,geom))return error("M9 CUDA size overflow");
    const size_t parts[]={wave,psi,model,prof_count,src,geom,data_count,trajectory};
    for(size_t k=0;k<8;k++){if(!add(total,parts[k],tmp))return error("M9 CUDA size overflow");total=tmp;}
    int count=0,device=options?options->device:0,version=0;cudaDeviceProp prop;
    if(!CUDA(cudaGetDeviceCount(&count)))return -1;
    if(count==0 || device<0 || device>=count)return error("M9 CUDA backend/device unavailable (visible=%d, requested=%d)",count,device);
    if(!CUDA(cudaSetDevice(device)) || !CUDA(cudaGetDeviceProperties(&prop,device)) || !CUDA(cudaRuntimeGetVersion(&version)))return -1;
    size_t avail=0,capacity=0;
    if(!CUDA(cudaMemGetInfo(&avail,&capacity)))return -1;
    size_t reserve=options?options->reserve_bytes:0,usable=avail>reserve?avail-reserve:0;
    if(options && options->cap_bytes && usable>options->cap_bytes)usable=options->cap_bytes;
    if(total>usable)return error("M9 CUDA mandatory budget %zu exceeds usable budget %zu",total,usable);
    denise_cuda_m9 *c=(denise_cuda_m9 *)m9_host_calloc(1,sizeof(*c));if(!c)return -1;
    c->device=device;c->d.mandatory_bytes=total;c->d.usable_budget=usable;c->d.remaining_budget=usable-total;
    c->d.model_bytes=model;c->d.wavefield_bytes=wave;c->d.cpml_bytes=psi;c->d.profile_bytes=prof_count;
    c->d.source_bytes=src;c->d.receiver_bytes=geom+data_count;c->d.trajectory_bytes=trajectory;
    c->d.host_metadata_bytes=sizeof(*c)+m9_host_metadata_bytes();c->d.visible_devices=count;c->d.runtime_version=version;c->d.major=prop.major;c->d.minor=prop.minor;
    *out=c; /* Partial ownership remains destroyable if a real cleanup error occurs. */
    if(m9_host_create(q,&c->host)) { error("M9 canonical preparation: %s",m9_host_error());goto failure; }
    if(!CUDA(cudaMalloc(&c->arena,total)))goto failure;
    c->d.owned_bytes=total;device_owned+=total;
    {
        char *p=(char *)c->arena;View &v=c->v;
        v.nx=q->nx;v.ny=q->ny;v.w=q->nx+4;v.nt=q->nt;v.nr=q->receiver_count;v.si=q->source_i;v.sj=q->source_j;
        v.cells=cells;v.padded=padded;v.coefficient=q->dt/q->dh;
        v.field=(float *)p;p+=wave;v.psi=(float *)p;p+=psi;v.map=(float *)p;p+=model;
        v.profile=(float *)p;p+=prof_count;v.source=(float *)p;p+=src;v.geometry=(int *)p;p+=geom;
        v.data=(float *)p;p+=data_count;v.strain=(float *)p;
    }
    if(!CUDA(cudaEventCreate(&c->start)))goto failure;++events_owned;
    if(!CUDA(cudaEventCreate(&c->end)))goto failure;++events_owned;
    if(!upload(c,c->host,true) || !zero(c) || !CUDA(cudaDeviceSynchronize()))goto failure;
    return 0;
failure:
    invalidate(c);denise_cuda_m9_destroy(out);return -1;
}
static int run(denise_cuda_m9 *c,const m9_host *model,int prepared) {
    if(!select(c))return finish_failure(c);invalidate(c);
    /* Profiles/rho/source/geometry remain the original canonical background.
     * Only supplied lambda/mu and their harmonic corner map change. */
    if(!upload(c,c->host,true) || (model!=c->host && !upload(c,model,false)) ||
       !zero(c) || !CUDA(cudaEventRecord(c->start)))return finish_failure(c);
    for(int t=0;t<c->v.nt;t++)if(!step(c,t))return finish_failure(c);
    if(!CUDA(cudaEventRecord(c->end)) || !CUDA(cudaEventSynchronize(c->end)) ||
       !CUDA(cudaEventElapsedTime(&c->d.elapsed_ms,c->start,c->end)))return finish_failure(c);
    c->valid=1;c->d.valid_steps=c->v.nt;c->d.prepared=prepared;return 0;
}
extern "C" int denise_cuda_m9_prepare(denise_cuda_m9 *c) {
    begin();if(!c)return error("M9 CUDA context is null");return run(c,c->host,1);
}
extern "C" int denise_cuda_m9_nonlinear(denise_cuda_m9 *c,const float *l,const float *m) {
    begin();if(!c || !l || !m)return finish_failure(c);invalidate(c);
    denise_elastic_psv_born_config q=*m9_host_config(c->host);q.lambda=l;q.mu=m;
    m9_host *h=NULL;
    if(m9_host_create(&q,&h))return error("M9 canonical nonlinear material: %s",m9_host_error());
    int rc=run(c,h,0);m9_host_destroy(&h);return rc;
}
static bool copy_out(void *out,const void *in,size_t bytes) {return CUDA(cudaMemcpy(out,in,bytes,cudaMemcpyDeviceToHost));}
extern "C" int denise_cuda_m9_download(denise_cuda_m9 *c,float *data,float *fields,float *psi,float *operands) {
    begin();if(!c || !c->valid)return error("M9 CUDA output unavailable: no completed valid run");
    if(!select(c))return finish_failure(c);
    size_t sizes[4]={(size_t)c->v.nt*c->v.nr*8,5*c->v.cells*4,8*c->v.cells*4,c->d.trajectory_bytes};
    void *outs[4]={data,fields,psi,operands};float *scratch[4]={NULL,NULL,NULL,NULL};bool ok=true;
    for(int k=0;k<4 && ok;k++)if(outs[k]) {scratch[k]=(float *)m9_host_calloc(sizes[k],1);ok=scratch[k]!=NULL;}
    if(ok && data)ok=copy_out(scratch[0],c->v.data,sizes[0]);
    for(int k=1;k<=2 && ok;k++)if(outs[k])for(int f=0;f<(k==1?5:8) && ok;f++)for(int j=0;j<c->v.ny && ok;j++)
        ok=copy_out(scratch[k]+(size_t)f*c->v.cells+(size_t)j*c->v.nx,
                    (k==1?c->v.field:c->v.psi)+(size_t)f*c->v.padded+(size_t)(j+2)*c->v.w+2,(size_t)c->v.nx*4);
    if(ok && operands)ok=copy_out(scratch[3],c->v.strain,sizes[3]);
    if(ok)for(int k=0;k<4;k++)if(outs[k])memcpy(outs[k],scratch[k],sizes[k]);
    for(int k=0;k<4;k++)m9_host_free(scratch[k]);return ok?0:finish_failure(c);
}
extern "C" int denise_cuda_m9_diagnostics(const denise_cuda_m9 *c,struct denise_cuda_m9_diagnostics *d) {
    if(!c || !d)return error("M9 CUDA diagnostics argument is null");*d=c->d;return 0;
}
extern "C" int denise_cuda_m9_test_step(denise_cuda_m9 *c,const float *state,const float *profiles) {
    begin();if(!c || !state)return finish_failure(c);invalidate(c);
    if(!select(c) || !upload(c,c->host,true) || !zero(c))return finish_failure(c);
    bool ok=true;
    for(int f=0;f<13 && ok;f++)for(int j=0;j<c->v.ny && ok;j++)
        ok=CUDA(cudaMemcpy((f<5?c->v.field:c->v.psi)+(size_t)(f<5?f:f-5)*c->v.padded+(size_t)(j+2)*c->v.w+2,
                          state+(size_t)f*c->v.cells+(size_t)j*c->v.nx,(size_t)c->v.nx*4,cudaMemcpyHostToDevice));
    if(ok && profiles)ok=CUDA(cudaMemcpy(c->v.profile,profiles,c->d.profile_bytes,cudaMemcpyHostToDevice));
    if(!ok || !step(c,0))return finish_failure(c);c->valid=1;c->d.valid_steps=1;return 0;
}
extern "C" int denise_cuda_m9_test_halo(denise_cuda_m9 *c,int velocity,float *padded) {
    begin();if(!c || !padded || (velocity!=0 && velocity!=1))return finish_failure(c);invalidate(c);
    float *tmp=(float *)m9_host_calloc(c->d.wavefield_bytes,1);bool ok=tmp && select(c);
    if(ok)ok=CUDA(cudaMemcpy(c->v.field,padded,c->d.wavefield_bytes,cudaMemcpyHostToDevice)) && halos(c,velocity) &&
             CUDA(cudaDeviceSynchronize()) && copy_out(tmp,c->v.field,c->d.wavefield_bytes);
    if(ok)memcpy(padded,tmp,c->d.wavefield_bytes);m9_host_free(tmp);return ok?0:finish_failure(c);
}
extern "C" int denise_cuda_m9_test_static(denise_cuda_m9 *c,float *maps,float *profiles,float *source,int *geometry) {
    begin();if(!c || !maps || !profiles || !source || !geometry)return error("M9 static download argument is null");
    size_t n=c->d.model_bytes+c->d.profile_bytes+c->d.source_bytes+(size_t)c->v.nr*12;
    char *tmp=(char *)m9_host_calloc(n,1);bool ok=tmp && select(c);size_t o=0;
    if(ok)ok=copy_out(tmp,c->v.map,c->d.model_bytes);o+=c->d.model_bytes;
    if(ok)ok=copy_out(tmp+o,c->v.profile,c->d.profile_bytes);o+=c->d.profile_bytes;
    if(ok)ok=copy_out(tmp+o,c->v.source,c->d.source_bytes);o+=c->d.source_bytes;
    if(ok)ok=copy_out(tmp+o,c->v.geometry,(size_t)c->v.nr*12);
    if(ok){o=0;memcpy(maps,tmp,c->d.model_bytes);o+=c->d.model_bytes;memcpy(profiles,tmp+o,c->d.profile_bytes);
        o+=c->d.profile_bytes;memcpy(source,tmp+o,c->d.source_bytes);o+=c->d.source_bytes;memcpy(geometry,tmp+o,(size_t)c->v.nr*12);}
    m9_host_free(tmp);return ok?0:finish_failure(c);
}
extern "C" int denise_cuda_m9_test_trajectory_roundtrip(denise_cuda_m9 *c,float *operands) {
    begin();if(!c || !operands)return finish_failure(c);invalidate(c);
    float *tmp=(float *)m9_host_calloc(c->d.trajectory_bytes,1);
    bool ok=tmp && select(c) && CUDA(cudaMemcpy(c->v.strain,operands,c->d.trajectory_bytes,cudaMemcpyHostToDevice)) &&
        copy_out(tmp,c->v.strain,c->d.trajectory_bytes);
    if(ok)memcpy(operands,tmp,c->d.trajectory_bytes);
    m9_host_free(tmp);return ok?0:finish_failure(c);
}
#endif /* Host-only sanitizer compiles the actual tracked allocation code. */
