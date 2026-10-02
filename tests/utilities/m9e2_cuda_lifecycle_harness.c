/* Real CUDA FULL-J lifecycle, including both surface contracts. */
#include "denise_cuda_m9_elastic.h"
#include <stdio.h>
#include <string.h>
static int check(int rc) {if(rc)fprintf(stderr,"%s\n",denise_cuda_m9_last_error());return rc;}
int main(int argc,char **argv) {
    float l[63],m[63],rho[63],dl[63],dm[63],samples[3]={1.25f,-3.5f,.875f};
    int ri[3]={8,0,8},rj[3]={0,1,0},s,k,phase;
    struct denise_elastic_psv_born_config q;
    struct denise_cuda_m9_options opt={0,0,0};
    struct denise_cuda_m9 *c=NULL;
    struct denise_cuda_m9_diagnostics d;
    struct denise_cuda_m9_born_diagnostics bd;
    float data[18],f[315],psi[504],qj[252],corner[63],bg[756],repeat[756];
    size_t db,hb,ev,calls,sites[4],at;int rc;
    (void)argv;memset(&q,0,sizeof(q));
    for(k=0;k<63;k++){l[k]=5e9f+k*1e6f;m[k]=3e9f+k*3e5f;rho[k]=2000+k;dl[k]=(k%7-3)*1e7f;dm[k]=(k%5-2)*1e7f;}
    q.nx=9;q.ny=7;q.nt=3;q.fw=2;q.dh=10;q.dt=.0005f;q.invmat1=3;q.fdorder=4;
    q.ndt=q.dtinv=q.mpi_size=1;q.receiver_components=2;q.lambda=l;q.mu=m;q.rho=rho;
    q.source_i=1;q.source_j=1;q.source_samples=samples;q.receiver_count=3;q.receiver_i=ri;q.receiver_j=rj;
    q.cpml_enabled=1;q.pml_reflection=.001f;q.pml_power=2;q.pml_kmax=1.3f;q.pml_fpml=15;q.pml_damping_speed=2500;
    for(s=0;s<2;s++) {
        q.free_surface=s;denise_cuda_m9_fault(0);
        if(check(denise_cuda_m9_create_full(&q,&opt,&c)))return 1;
        denise_cuda_m9_ledger(&db,&hb,&ev,&sites[0]);
        denise_cuda_m9_fault(0);if(check(denise_cuda_m9_prepare(c)))return 2;
        denise_cuda_m9_ledger(&db,&hb,&ev,&sites[1]);
        if(check(denise_cuda_m9_download(c,NULL,NULL,NULL,bg)))return 3;
        denise_cuda_m9_fault(0);if(check(denise_cuda_m9_apply_j(c,dl,dm,63)))return 4;
        denise_cuda_m9_ledger(&db,&hb,&ev,&sites[2]);
        denise_cuda_m9_fault(0);if(check(denise_cuda_m9_born_download(c,data,f,psi,qj,corner)))return 5;
        denise_cuda_m9_ledger(&db,&hb,&ev,&sites[3]);
        if(check(denise_cuda_m9_apply_j(c,dl,dm,63)) || check(denise_cuda_m9_download(c,NULL,NULL,NULL,repeat)))return 6;
        if(memcmp(bg,repeat,sizeof(bg)))return 7;
        if(check(denise_cuda_m9_destroy(&c)))return 8;
        if(argc>1)for(phase=0;phase<4;phase++)for(at=1;at<=sites[phase];at++) {
            denise_cuda_m9_fault(0);
            if(phase>0 && check(denise_cuda_m9_create_full(&q,&opt,&c)))return 9;
            if(phase>1 && check(denise_cuda_m9_prepare(c)))return 10;
            if(phase>2 && check(denise_cuda_m9_apply_j(c,dl,dm,63)))return 11;
            for(k=0;k<18;k++)data[k]=-91;
            denise_cuda_m9_fault(at);
            rc=phase==0?denise_cuda_m9_create_full(&q,&opt,&c):
               (phase==1?denise_cuda_m9_prepare(c):(phase==2?denise_cuda_m9_apply_j(c,dl,dm,63):
                denise_cuda_m9_born_download(c,data,f,psi,qj,corner)));
            if(!rc || !strstr(denise_cuda_m9_last_error(),"injected"))return 12;
            if(c) {
                if(check(denise_cuda_m9_diagnostics(c,&d)) || check(denise_cuda_m9_born_diagnostics(c,&bd)))return 13;
                if(d.prepared || bd.valid)return 14;
            }
            if(phase==3)for(k=0;k<18;k++)if(data[k]!=-91)return 15;
            if(check(denise_cuda_m9_destroy(&c)))return 16;
            denise_cuda_m9_ledger(&db,&hb,&ev,&calls);if(db || hb || ev)return 17;
        }
        denise_cuda_m9_fault(0);denise_cuda_m9_ledger(&db,&hb,&ev,&calls);if(db || hb || ev)return 18;
        printf("M9E2_LIFECYCLE_PASS surface=%d sites=%zu,%zu,%zu,%zu ownership=%zu,%zu,%zu\n",s,sites[0],sites[1],sites[2],sites[3],db,hb,ev);
    }
    return 0;
}
