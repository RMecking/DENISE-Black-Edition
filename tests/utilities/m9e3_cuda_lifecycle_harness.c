/* Real new CUDA JT/migration ownership, transactional outputs, repeatability.
 * Optional fault sweep arms after prepare for the new JT/download phases. */
#include "denise_cuda_m9_elastic.h"
#include <stdio.h>
#include <string.h>
static int check(int rc) {if(rc)fprintf(stderr,"%s\n",denise_cuda_m9_last_error());return rc;}
int main(int argc,char **argv) {
    float l[63],m[63],rho[63],samples[3]={1.25f,-3.5f,.875f},residual[18];
    double gl[63],gm[63],first[126];int ri[3]={8,0,8},rj[3]={0,1,0};
    struct denise_elastic_psv_born_config q;
    struct denise_cuda_m9_options opt={0,0,0};
    struct denise_cuda_m9 *c=NULL;
    struct denise_cuda_m9_adjoint_diagnostics d;
    size_t db,hb,ev,calls,sites[2],at;int s,k,phase,rc;
    (void)argv;memset(&q,0,sizeof(q));
    for(k=0;k<63;k++){l[k]=5e9f+k*1e6f;m[k]=3e9f+k*3e5f;rho[k]=2000+k;}
    for(k=0;k<18;k++)residual[k]=(k%7-3)*.125f;
    q.nx=9;q.ny=7;q.nt=3;q.fw=2;q.dh=10;q.dt=.0005f;q.invmat1=3;q.fdorder=4;
    q.ndt=q.dtinv=q.mpi_size=1;q.receiver_components=2;q.lambda=l;q.mu=m;q.rho=rho;
    q.source_i=1;q.source_j=1;q.source_samples=samples;q.receiver_count=3;q.receiver_i=ri;q.receiver_j=rj;
    q.cpml_enabled=1;q.pml_reflection=.001f;q.pml_power=2;q.pml_kmax=1.3f;q.pml_fpml=15;q.pml_damping_speed=2500;
    for(s=0;s<2;s++) {
        q.free_surface=s;denise_cuda_m9_fault(0);
        if(check(denise_cuda_m9_create_migration(&q,&opt,&c)) || check(denise_cuda_m9_prepare(c)))return 1;
        denise_cuda_m9_fault(0);if(check(denise_cuda_m9_apply_jt(c,residual,18)))return 2;
        denise_cuda_m9_ledger(&db,&hb,&ev,&sites[0]);
        denise_cuda_m9_fault(0);if(check(denise_cuda_m9_image_download(c,gl,gm,63)))return 3;
        denise_cuda_m9_ledger(&db,&hb,&ev,&sites[1]);
        memcpy(first,gl,sizeof(gl));memcpy(first+63,gm,sizeof(gm));
        if(check(denise_cuda_m9_migrate(c,residual,18,gl,gm,63)))return 4;
        if(memcmp(first,gl,sizeof(gl)) || memcmp(first+63,gm,sizeof(gm)))return 5;
        if(check(denise_cuda_m9_destroy(&c)))return 6;
        if(argc>1)for(phase=0;phase<2;phase++)for(at=1;at<=sites[phase];at++) {
            denise_cuda_m9_fault(0);
            if(check(denise_cuda_m9_create_migration(&q,&opt,&c)) || check(denise_cuda_m9_prepare(c)))return 7;
            if(phase && check(denise_cuda_m9_apply_jt(c,residual,18)))return 8;
            for(k=0;k<63;k++)gl[k]=gm[k]=-91;
            denise_cuda_m9_fault(at);
            rc=phase?denise_cuda_m9_image_download(c,gl,gm,63):denise_cuda_m9_apply_jt(c,residual,18);
            if(!rc || !strstr(denise_cuda_m9_last_error(),"injected"))return 9;
            if(check(denise_cuda_m9_adjoint_diagnostics(c,&d)) || d.valid)return 10;
            for(k=0;k<63;k++)if(gl[k]!=-91 || gm[k]!=-91)return 11;
            if(check(denise_cuda_m9_destroy(&c)))return 12;
            denise_cuda_m9_ledger(&db,&hb,&ev,&calls);if(db || hb || ev)return 13;
        }
        denise_cuda_m9_fault(0);denise_cuda_m9_ledger(&db,&hb,&ev,&calls);if(db || hb || ev)return 14;
        printf("M9E3_CUDA_LIFECYCLE_PASS surface=%d new_jt_sites=%zu image_sites=%zu ownership=0,0,0\n",s,sites[0],sites[1]);
    }
    return 0;
}
