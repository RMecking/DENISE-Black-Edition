/* Private test translation unit: actual production routines, no reference loop. */
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
static long fail_after=-1;
static struct {void *pointer;size_t bytes;} allocations[512];
static size_t native_current,native_peak,native_live,native_live_peak;
static void *test_calloc(size_t n,size_t w) {
    void *p;size_t k;
    if(fail_after==0)return NULL;
    if(fail_after>0)--fail_after;
    p=calloc(n,w);
    if(p)for(k=0;k<512;k++)if(!allocations[k].pointer) {
        allocations[k].pointer=p;allocations[k].bytes=n*w;
        native_current+=n*w;++native_live;
        if(native_current>native_peak)native_peak=native_current;
        if(native_live>native_live_peak)native_live_peak=native_live;
        break;
    }
    return p;
}
static void test_free(void *p) {
    size_t k;
    if(p)for(k=0;k<512;k++)if(allocations[k].pointer==p) {
        native_current-=allocations[k].bytes;--native_live;
        allocations[k].pointer=NULL;allocations[k].bytes=0;break;
    }
    free(p);
}
static float *test_initial,*test_fields,*test_operands;
static size_t test_cells;
static void test_initialize(float **f);
static void test_snapshot(float **f,float **q,int time,int stage);
#define calloc test_calloc
#define free test_free
#define DENISE_M9L1A_INITIALIZE(c,f) do {if((c)->fd8)test_initialize(f);} while(0)
#define DENISE_M9L1A_SNAPSHOT(c,f,q,t,stage) do {if((c)->fd8)test_snapshot(f,q,t,stage);} while(0)
#include "../../src/PSV/elastic_psv_born.c"
#undef calloc
#undef free
static void test_initialize(float **f) {
    int k;size_t p;
    if(!test_initial)return;
    for(k=0;k<5;k++)for(p=0;p<test_cells;p++)f[k][p]=test_initial[(size_t)k*test_cells+p];
}
static void test_snapshot(float **f,float **q,int time,int stage) {
    int k;
    if(test_fields)for(k=0;k<5;k++)memcpy(test_fields+(((size_t)time*4+(size_t)stage)*5+(size_t)k)*test_cells,
                                         f[k],test_cells*sizeof(float));
    if(stage==1&&test_operands)for(k=0;k<4;k++)memcpy(test_operands+((size_t)time*4+(size_t)k)*test_cells,
                                                  q[k],test_cells*sizeof(float));
}
int denise_fd8_test_capture(struct denise_elastic_psv_forward_fd8 *h,float *initial,
                           float *data,float *fields,float *operands) {
    int result;size_t p;
    test_initial=initial;test_fields=fields;test_operands=operands;test_cells=h->core->cells;
    /* Reject nonfinite private initialization without public arbitrary-state API. */
    if(initial)for(p=0;p<test_cells*5;p++)if(!isfinite(initial[p])) {
        test_initial=NULL;test_fields=NULL;test_operands=NULL;
        return fd8_fail("nonfinite private initial state");
    }
    result=denise_elastic_psv_forward_fd8_nonlinear_v1(h,h->core->lambda,h->core->mu,data,h->core->data_count);
    test_initial=NULL;test_fields=NULL;test_operands=NULL;return result;
}
int denise_fd8_test_derivative(struct denise_elastic_psv_forward_fd8 *h,const float *input,
                              int kind,float *output) {
    int i,j;size_t p;
    if(!h||!input||!output||kind<0||kind>3)return -1;
    h->fault=0;
    for(p=0;p<h->core->cells;p++)if(!isfinite(input[p]))return -1;
    for(j=0;j<h->core->ny;j++)for(i=0;i<h->core->nx;i++)
        output[cell(h->core,j,i)]=derivative_at(h->core,input,j,i,kind);
    return h->fault?-1:0;
}
int denise_fd8_test_maps(int nx,int ny,float *mu,float *rho,float *maps) {
    struct denise_elastic_psv_born c;int fault=0;
    if(nx<1||ny<1)return -1;
    memset(&c,0,sizeof(c));c.nx=nx;c.ny=ny;c.cells=(size_t)nx*(size_t)ny;
    c.fd8=1;c.fd8_fault=&fault;c.mu=mu;c.rho=rho;c.invrho_x=maps;c.invrho_y=maps+c.cells;
    c.mu_corner=maps+2*c.cells;build_material_maps(&c);return fault?-1:0;
}
void denise_fd8_test_fail_allocation(long count) {fail_after=count;}
void denise_fd8_test_reset_native_peak(void) {native_peak=native_current;native_live_peak=native_live;}
size_t denise_fd8_test_native_bytes(int which) {
    return which==0?native_current:(which==1?native_peak:native_live_peak);
}
int denise_fd8_test_set_round(int mode) {
    int modes[4]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    return mode<0||mode>3?-1:fesetround(modes[mode]);
}
unsigned denise_fd8_test_mxcsr(unsigned value,int write) {
#if defined(__SSE__)
    unsigned old=_mm_getcsr();if(write)_mm_setcsr(value);return old;
#else
    (void)value;(void)write;return 0;
#endif
}
size_t denise_fd8_test_abi(int which,int member) {
    const size_t offsets[]={offsetof(struct denise_elastic_psv_born_config,nx),
        offsetof(struct denise_elastic_psv_born_config,ny),offsetof(struct denise_elastic_psv_born_config,nt),
        offsetof(struct denise_elastic_psv_born_config,fw),offsetof(struct denise_elastic_psv_born_config,dh),
        offsetof(struct denise_elastic_psv_born_config,dt),offsetof(struct denise_elastic_psv_born_config,l),
        offsetof(struct denise_elastic_psv_born_config,invmat1),offsetof(struct denise_elastic_psv_born_config,fdorder),
        offsetof(struct denise_elastic_psv_born_config,ndt),offsetof(struct denise_elastic_psv_born_config,dtinv),
        offsetof(struct denise_elastic_psv_born_config,free_surface),offsetof(struct denise_elastic_psv_born_config,boundary),
        offsetof(struct denise_elastic_psv_born_config,mpi_size),offsetof(struct denise_elastic_psv_born_config,receiver_components),
        offsetof(struct denise_elastic_psv_born_config,lambda),offsetof(struct denise_elastic_psv_born_config,mu),
        offsetof(struct denise_elastic_psv_born_config,rho),offsetof(struct denise_elastic_psv_born_config,source_i),
        offsetof(struct denise_elastic_psv_born_config,source_j),
        offsetof(struct denise_elastic_psv_born_config,source_samples),
        offsetof(struct denise_elastic_psv_born_config,receiver_count),offsetof(struct denise_elastic_psv_born_config,receiver_i),
        offsetof(struct denise_elastic_psv_born_config,receiver_j),offsetof(struct denise_elastic_psv_born_config,cpml_enabled),
        offsetof(struct denise_elastic_psv_born_config,pml_reflection),offsetof(struct denise_elastic_psv_born_config,pml_power),
        offsetof(struct denise_elastic_psv_born_config,pml_kmax),offsetof(struct denise_elastic_psv_born_config,pml_fpml),
        offsetof(struct denise_elastic_psv_born_config,pml_damping_speed)};
    const size_t new_offsets[]={offsetof(struct denise_elastic_psv_forward_fd8_config_v1,struct_size),
        offsetof(struct denise_elastic_psv_forward_fd8_config_v1,abi_version),
        offsetof(struct denise_elastic_psv_forward_fd8_config_v1,base),
        offsetof(struct denise_elastic_psv_forward_fd8_config_v1,coefficient_family),
        offsetof(struct denise_elastic_psv_forward_fd8_config_v1,field_boundary_policy),
        offsetof(struct denise_elastic_psv_forward_fd8_config_v1,material_boundary_policy)};
    if(which==0)return sizeof(struct denise_elastic_psv_born_config);
    if(which==1)return sizeof(struct denise_elastic_psv_forward_fd8_config_v1);
    if(which==2&&member>=0&&(size_t)member<sizeof(offsets)/sizeof(offsets[0]))return offsets[member];
    if(which==3&&member>=0&&member<6)return new_offsets[member];
    if(which==4){struct aligned_old {char pad;struct denise_elastic_psv_born_config value;};return offsetof(struct aligned_old,value);}
    if(which==5){struct aligned_new {char pad;struct denise_elastic_psv_forward_fd8_config_v1 value;};return offsetof(struct aligned_new,value);}
    return (size_t)-1;
}

/* The unchanged canonical Born prefix is inspected only by this private
 * harness, on its own or baseline C handle. No public layout is exposed. */
void denise_fd4_test_copy_maps(const struct denise_elastic_psv_born *c,float *out) {
    const float *maps[6]={c->lambda,c->mu,c->rho,c->invrho_x,c->invrho_y,c->mu_corner};
    int k;for(k=0;k<6;k++)memcpy(out+(size_t)k*c->cells,maps[k],c->cells*sizeof(float));
}

#ifdef DENISE_M9L1A_SANITIZER_MAIN
int main(void) {
    struct denise_elastic_psv_forward_fd8_config_v1 cfg;
    struct denise_elastic_psv_forward_fd8 *handle=NULL;
    float lambda[81],mu[81],rho[81],source[1]={0},data[2]={-1,-1};
    int ri=4,rj=4,k;long failure;
    memset(&cfg,0,sizeof(cfg));
    cfg.struct_size=sizeof(cfg);cfg.abi_version=1;cfg.coefficient_family=1;
    cfg.field_boundary_policy=1;cfg.material_boundary_policy=1;
    cfg.base.nx=cfg.base.ny=9;cfg.base.nt=1;cfg.base.dh=1;cfg.base.dt=0.025f;
    cfg.base.invmat1=3;cfg.base.fdorder=8;cfg.base.ndt=cfg.base.dtinv=1;
    cfg.base.mpi_size=1;cfg.base.receiver_components=2;cfg.base.receiver_count=1;
    cfg.base.lambda=lambda;cfg.base.mu=mu;cfg.base.rho=rho;cfg.base.source_samples=source;
    cfg.base.source_i=cfg.base.source_j=4;cfg.base.receiver_i=&ri;cfg.base.receiver_j=&rj;
    for(k=0;k<81;k++){lambda[k]=6.75f;mu[k]=3.375f;rho[k]=1.5f;}
    for(failure=0;failure<23;failure++) {
        denise_fd8_test_fail_allocation(failure);
        if(denise_elastic_psv_forward_fd8_create_v1(&cfg,&handle)!=-1||handle)return 1;
    }
    denise_fd8_test_fail_allocation(-1);
    if(denise_elastic_psv_forward_fd8_create_v1(&cfg,&handle))return 2;
    for(failure=0;failure<19;failure++) {
        denise_fd8_test_fail_allocation(failure);data[0]=data[1]=-1;
        if(denise_elastic_psv_forward_fd8_nonlinear_v1(handle,lambda,mu,data,2)!=-1
           ||data[0]!=-1||data[1]!=-1)return 3;
    }
    denise_fd8_test_fail_allocation(-1);
    if(denise_elastic_psv_forward_fd8_nonlinear_v1(handle,lambda,mu,data,2)
       ||data[0]!=0||data[1]!=0)return 4;
    denise_elastic_psv_forward_fd8_destroy_v1(&handle);
    if(handle)return 5;
    puts("Foundation sanitizer lifecycle/failure/bounds PASS");return 0;
}
#endif
