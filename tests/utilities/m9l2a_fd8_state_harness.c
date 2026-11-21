/* Private test translation unit. No analytical FIELD padding or snapshot hook.
 * Normal public compilation does not export these symbols or the private API.
 */
#include <stddef.h>
#include <stdlib.h>
static long m9l2a_fail_after=-1;
static void *m9l2a_calloc(size_t n,size_t w) {
    if(m9l2a_fail_after==0)return NULL;
    if(m9l2a_fail_after>0)--m9l2a_fail_after;
    return calloc(n,w);
}
#define calloc m9l2a_calloc
#define DENISE_M9L2A_PRIVATE 1
#include "../../src/PSV/elastic_psv_born.c"
#undef calloc

void *m9l2a_test_create(int nx,int ny,int nt,float lam,float mu,float rho,float dt,
        const float *source,int si,int sj,int nr,const int *ri,const int *rj) {
    struct denise_elastic_psv_forward_fd8_config_v1 cfg;
    struct denise_elastic_psv_forward_fd8 *h=NULL;
    float *material;size_t p,n;
    if(nx<9||ny<9||nt<1||checked_product((size_t)nx,(size_t)ny,&n)
       ||n>INT_MAX||n>SIZE_MAX/(3*sizeof(float)))return NULL;
    material=checked_calloc(3*n,sizeof(float));if(!material)return NULL;
    for(p=0;p<n;p++){material[p]=lam;material[n+p]=mu;material[2*n+p]=rho;}
    memset(&cfg,0,sizeof(cfg));cfg.struct_size=sizeof(cfg);cfg.abi_version=1;
    cfg.coefficient_family=1;cfg.field_boundary_policy=1;cfg.material_boundary_policy=1;
    cfg.base.nx=nx;cfg.base.ny=ny;cfg.base.nt=nt;cfg.base.dh=1.0f;cfg.base.dt=dt;
    cfg.base.invmat1=3;cfg.base.fdorder=8;cfg.base.ndt=cfg.base.dtinv=1;
    cfg.base.mpi_size=1;cfg.base.receiver_components=2;cfg.base.receiver_count=nr;
    cfg.base.lambda=material;cfg.base.mu=material+n;cfg.base.rho=material+2*n;
    cfg.base.source_samples=source;cfg.base.source_i=si;cfg.base.source_j=sj;
    cfg.base.receiver_i=ri;cfg.base.receiver_j=rj;
    if(denise_elastic_psv_forward_fd8_create_v1(&cfg,&h))h=NULL;
    free(material);return h;
}
void m9l2a_test_destroy(void **h) {
    struct denise_elastic_psv_forward_fd8 *v=(struct denise_elastic_psv_forward_fd8 *)*h;
    denise_elastic_psv_forward_fd8_destroy_v1(&v);*h=NULL;
}
int m9l2a_test_forward(void *h,int begin,int end,const float *initial,
        float *data,size_t data_count,float *final,size_t state_count) {
    return m9l2a_forward(h,begin,end,initial,data,data_count,final,state_count);
}
int m9l2a_test_transpose(void *h,int begin,int end,const double *terminal,
        const double *data,double *initial,size_t state_count,double *source,size_t source_count) {
    return m9l2a_transpose(h,begin,end,terminal,data,initial,state_count,source,source_count);
}
int m9l2a_test_derivative(void *handle,int kind,const float *input,float *out) {
    struct denise_elastic_psv_forward_fd8 *h=handle;
    int i,j;
    if(m9l2a_context(h))return -1;
    if(kind<0||kind>3||!out||!m9l2a_finite_f(input,h->core->cells))return -1;
    for(j=0;j<h->core->ny;j++)for(i=0;i<h->core->nx;i++)
        out[cell(h->core,j,i)]=fd8_derivative(h->core,input,j,i,kind);
    return h->fault?-1:0;
}
int m9l2a_test_derivative_transpose(void *h,int kind,const double *input,
                                   double *out,size_t count) {
    return m9l2a_derivative_transpose(h,kind,input,out,count);
}
/* Snapshot is in-memory and tied to the same live immutable context. No disk
 * checkpoint format or public ABI is introduced. Return checkpoint-restored
 * next step and payload values, so the caller independently checks chronology.
 */
int m9l2a_test_checkpoint(void *h,int next,const float *input,float *output,
                         size_t count,int *restored_next) {
    struct m9l2a_checkpoint *q=NULL;int status;
    status=m9l2a_checkpoint_capture(h,next,input,&q);
    if(!status)status=m9l2a_checkpoint_restore(h,q,output,count,restored_next);
    m9l2a_checkpoint_destroy(&q);return status;
}
void m9l2a_test_fail_allocation(long n) {m9l2a_fail_after=n;}
int m9l2a_test_round(int mode) {
    int a[4]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    return mode<0||mode>3?-1:fesetround(a[mode]);
}

/* Prospective standalone UBSan lifecycle/negative-capacity entry. No valid
 * Forward or reverse timestep is reachable; this is NOT an executed test.
 */
#ifdef DENISE_M9L2A_SANITIZER_MAIN
int main(void) {
    float signal[1]={0},initial[405]={0},trace[2]={-987,-987},end[405];
    double terminal[405]={0},residual[2]={0},bar[405],source_bar[1]={-987};
    int i=0,j=0,k,status=0;void *h;
    h=m9l2a_test_create(9,9,1,1.5f,0.75f,2.0f,0.03125f,signal,0,0,1,&i,&j);
    if(!h)return 1;
    for(k=0;k<405;k++){end[k]=-987;bar[k]=-987;}
    if(m9l2a_test_forward(h,0,1,initial,trace,2,end,404)!=-1)status=2;
    if(m9l2a_test_transpose(h,0,1,terminal,residual,bar,404,source_bar,1)!=-1)status=3;
    for(k=0;k<405;k++)if(end[k]!=-987||bar[k]!=-987)status=4;
    if(trace[0]!=-987||trace[1]!=-987||source_bar[0]!=-987)status=5;
    m9l2a_test_destroy(&h);if(h)status=6;
    return status;
}
#endif
