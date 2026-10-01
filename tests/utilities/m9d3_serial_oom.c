#include "denise_elastic_psv_born.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

enum { CAPACITY=2048,NX=8,NY=8,NT=32,NREC=5 };
struct entry {void *pointer;size_t bytes;};
static struct entry live[CAPACITY];
static size_t bytes,count;
static int armed,at,index_call,injected,invalid;
void *__real_calloc(size_t,size_t);
void __real_free(void *);
void *__wrap_calloc(size_t n,size_t width) {
    void *p;int i;
    if(armed && ++index_call==at){injected++;return NULL;}
    p=__real_calloc(n,width);if(!p)return NULL;
    for(i=0;i<CAPACITY;i++)if(!live[i].pointer) {
        live[i].pointer=p;live[i].bytes=n*width;bytes+=n*width;count++;return p;
    }
    abort();
}
void __wrap_free(void *p) {
    int i;if(!p)return;
    for(i=0;i<CAPACITY;i++)if(live[i].pointer==p) {
        bytes-=live[i].bytes;count--;live[i].pointer=NULL;__real_free(p);return;
    }
    invalid++;
}
static void require(int ok) {if(!ok){fprintf(stderr,"serial ledger failure index=%d injected=%d bytes=%lu count=%lu invalid=%d\n",at,injected,(unsigned long)bytes,(unsigned long)count,invalid);exit(1);}}
static int trial(int phase,int segments,int failure) {
    struct denise_elastic_psv_born_config cfg;
    struct denise_elastic_psv_born *c=NULL;
    float lam[NX*NY],mu[NX*NY],rho[NX*NY],dl[NX*NY],dm[NX*NY],source[NT],data[NT*NREC*2];
    double gl[NX*NY],gm[NX*NY];
    int ri[NREC]={0,3,4,7,1},rj[NREC]={0,1,0,1,6};
    int k,status,calls;
    require(bytes==0 && count==0 && invalid==0);
    memset(&cfg,0,sizeof(cfg));
    for(k=0;k<NX*NY;k++){lam[k]=6.44e9f;mu[k]=5.78e9f;rho[k]=2000;dl[k]=10000;dm[k]=-5000;}
    for(k=0;k<NT;k++)source[k]=(float)(k+1)*.01f;
    for(k=0;k<NT*NREC*2;k++)data[k]=(float)(k+1)*.001f;
    cfg.nx=NX;cfg.ny=NY;cfg.nt=NT;cfg.fw=2;cfg.dh=10;cfg.dt=.0004f;
    cfg.lambda=lam;cfg.mu=mu;cfg.rho=rho;cfg.source_i=3;cfg.source_j=1;cfg.source_samples=source;
    cfg.receiver_count=NREC;cfg.receiver_i=ri;cfg.receiver_j=rj;cfg.invmat1=3;cfg.fdorder=4;
    cfg.ndt=cfg.dtinv=1;cfg.free_surface=1;cfg.mpi_size=1;cfg.receiver_components=2;
    cfg.cpml_enabled=1;cfg.pml_reflection=.001f;cfg.pml_power=2;cfg.pml_kmax=1;cfg.pml_fpml=15;cfg.pml_damping_speed=3000;
    armed=0;
    if(phase!=0){require(denise_elastic_psv_born_create(&cfg,&c)==0);
        if(segments)require(denise_elastic_psv_born_set_replay_segments(c,segments)==0);
        require(denise_elastic_psv_born_prepare(c,NULL)==0);}
    at=failure;index_call=0;injected=0;armed=1;
    if(phase==0)status=denise_elastic_psv_born_create(&cfg,&c);
    else if(phase==1)status=denise_elastic_psv_born_prepare(c,data);
    else if(phase==2)status=denise_elastic_psv_born_apply_j(c,dl,dm,data);
    else if(phase==3)status=denise_elastic_psv_born_apply_jt(c,data,gl,gm);
    else status=denise_elastic_psv_born_checkpoint_roundtrip(c,13);
    armed=0;calls=index_call;
    require(failure?(status!=0 && injected==1):(status==0 && injected==0));
    if(phase==1)require(denise_elastic_psv_born_is_prepared(c)==!failure);
    denise_elastic_psv_born_destroy(&c);
    require(!c && bytes==0 && count==0 && invalid==0);
    return calls;
}
int main(void) {
    int phase,segments,k,n,total=0;
    for(segments=0;segments<=3;segments+=3)for(phase=0;phase<5;phase++) {
        n=trial(phase,segments,0);
        for(k=1;k<=n;k++){trial(phase,segments,k);total++;}
        printf("SERIAL_SURFACE_OOM phase=%d segments=%d calls=%d zero_outstanding=YES\n",phase,segments,n);
    }
    printf("SERIAL_SURFACE_OOM failures=%d PASS\n",total);return 0;
}
