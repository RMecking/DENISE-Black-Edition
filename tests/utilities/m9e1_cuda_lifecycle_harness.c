/* Standalone candidate lifecycle/host sanitizer and compute-sanitizer view. */
#include "denise_cuda_m9_elastic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(int rc) {
    if(rc)fprintf(stderr,"%s\n",denise_cuda_m9_last_error());
    return rc;
}
int main(int argc,char **argv) {
    float l[63],m[63],rho[63],samples[3]={1.25f,-3.5f,.875f};
    int ri[3]={6,2,6},rj[3]={3,5,3};
    struct denise_elastic_psv_born_config cfg;
    struct denise_cuda_m9_options opt={0,0,0};
    struct denise_cuda_m9 *ctx=NULL;
    float data[18],fields[315],psi[504],operands[756],state[819];
    size_t db,hb,ev,calls,n,at;int k;
    memset(&cfg,0,sizeof(cfg));
    for(k=0;k<63;k++){l[k]=5e9f+k*1e6f;m[k]=3e9f+k*3e5f;rho[k]=2000+k;}
    cfg.nx=9;cfg.ny=7;cfg.nt=3;cfg.fw=2;cfg.dh=10;cfg.dt=.0005f;
    cfg.invmat1=3;cfg.fdorder=4;cfg.ndt=cfg.dtinv=cfg.mpi_size=1;cfg.receiver_components=2;
    cfg.lambda=l;cfg.mu=m;cfg.rho=rho;cfg.source_i=4;cfg.source_j=3;cfg.source_samples=samples;
    cfg.receiver_count=3;cfg.receiver_i=ri;cfg.receiver_j=rj;cfg.cpml_enabled=1;
    cfg.pml_reflection=.001f;cfg.pml_power=2;cfg.pml_kmax=1.3f;cfg.pml_fpml=15;cfg.pml_damping_speed=2500;
    denise_cuda_m9_fault(0);
    if(check(denise_cuda_m9_create(&cfg,&opt,&ctx)))return 1;
    denise_cuda_m9_ledger(&db,&hb,&ev,&n);
    if(check(denise_cuda_m9_prepare(ctx)) || check(denise_cuda_m9_download(ctx,data,fields,psi,operands)))return 2;
    if(check(denise_cuda_m9_nonlinear(ctx,l,m)) || check(denise_cuda_m9_download(ctx,data,fields,psi,operands)))return 3;
    for(k=0;k<819;k++)state[k]=(k%17-8)*.001f;
    if(check(denise_cuda_m9_test_step(ctx,state,NULL)) || check(denise_cuda_m9_download(ctx,data,fields,psi,operands)))return 4;
    if(check(denise_cuda_m9_destroy(&ctx)))return 5;
    /* Every new creation allocation and upload/event/setup call under ASan. */
    if(argc>1 && strcmp(argv[1],"faults")==0)for(at=1;at<=n;at++) {
        denise_cuda_m9_fault(at);
        if(denise_cuda_m9_create(&cfg,&opt,&ctx)==0)return 6;
        if(!strstr(denise_cuda_m9_last_error(),"injected"))return 7;
        if(check(denise_cuda_m9_destroy(&ctx)))return 8;
        denise_cuda_m9_ledger(&db,&hb,&ev,&calls);
        if(db || hb || ev)return 9;
    }
    denise_cuda_m9_fault(0);denise_cuda_m9_ledger(&db,&hb,&ev,&calls);
    if(db || hb || ev)return 10;
    printf("M9E1_LIFECYCLE_PASS creation_calls=%zu device=%zu host=%zu events=%zu\n",n,db,hb,ev);
    return 0;
}
