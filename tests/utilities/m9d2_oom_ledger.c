/* Candidate allocation observer, not a numerical oracle. Link only this
 * helper and the Born MPI module with --wrap=calloc,--wrap=free. MPI/libc
 * shared-library allocations are deliberately outside the requested-byte ledger.
 * No production fault switch or fixture-dependent allocation index is added. */
#include "denise_elastic_psv_born_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

enum { CAPACITY=8192, NX=24, NY=20, NT=10, RECEIVERS=2 };
struct allocation { void *pointer; size_t bytes; };
static struct allocation live[CAPACITY];
static size_t live_bytes,live_count;
static int rank,counting,allocation_index,fail_index,fail_rank,injected,bad_free;
void *__real_calloc(size_t,size_t);
void __real_free(void *);
void *__wrap_calloc(size_t count,size_t width) {
    void *pointer;int i;
    if(counting) {
        allocation_index++;
        if(rank==fail_rank&&allocation_index==fail_index){injected++;return NULL;}
    }
    pointer=__real_calloc(count,width);
    if(!pointer)return NULL;
    for(i=0;i<CAPACITY;i++)if(!live[i].pointer) {
        live[i].pointer=pointer;live[i].bytes=count*width;
        live_bytes+=live[i].bytes;live_count++;return pointer;
    }
    MPI_Abort(MPI_COMM_WORLD,80);return NULL;
}
void __wrap_free(void *pointer) {
    int i;if(!pointer)return;
    for(i=0;i<CAPACITY;i++)if(live[i].pointer==pointer) {
        live_bytes-=live[i].bytes;live_count--;
        live[i].pointer=NULL;live[i].bytes=0;__real_free(pointer);return;
    }
    /* Fail the ledger gate without passing an invalid/double free to libc. */
    bad_free++;
}
static void require(int condition) {
    if(!condition)MPI_Abort(MPI_COMM_WORLD,81);
}
static int clean_float(const float *a,size_t n) {
    size_t i;int zero=1,untouched=1;
    for(i=0;i<n;i++){zero&=a[i]==0;untouched&=a[i]==777.0f;}
    return zero||untouched;
}
static int clean_double(const double *a,size_t n) {
    size_t i;int zero=1,untouched=1;
    for(i=0;i<n;i++){zero&=a[i]==0;untouched&=a[i]==777.0;}
    return zero||untouched;
}
struct result {
    unsigned long long bytes,allocations;
    int status,calls,injected,invalid_free,prepared,output_ok,destroyed;
};
static struct result run_case(const char *phase,int px,int py,int segments,
                              int target,int index) {
    struct denise_elastic_psv_born_mpi *context=NULL;
    struct denise_elastic_psv_born_mpi_config cfg;
    struct result result;
    float lam[NX*NY],mu[NX*NY],rho[NX*NY],dl[NX*NY],dm[NX*NY];
    float source[NT],data[NT*RECEIVERS*2],output[NT*RECEIVERS*2];
    double gl[NX*NY],gm[NX*NY];
    int ri[RECEIVERS]={3,21},rj[RECEIVERS]={3,17};
    int i,nx=NX/px,ny=NY/py,receivers,status;
    size_t cells=(size_t)nx*ny,data_count;
    require(live_count==0&&live_bytes==0&&bad_free==0);
    memset(&cfg,0,sizeof(cfg));memset(&result,0,sizeof(result));
    for(i=0;i<(int)cells;i++) {
        lam[i]=2.0e9f;mu[i]=3.0e9f;rho[i]=2000.0f;
        dl[i]=100000.0f;dm[i]=-50000.0f;gl[i]=gm[i]=777.0;
    }
    for(i=0;i<NT;i++)source[i]=(float)(i+1)*.001f;
    for(i=0;i<NT*RECEIVERS*2;i++){data[i]=(float)(i+1)*.0001f;output[i]=777.0f;}
    cfg.communicator=MPI_COMM_WORLD;cfg.nprocx=px;cfg.nprocy=py;
    cfg.global.nx=NX;cfg.global.ny=NY;cfg.global.nt=NT;
    cfg.global.dh=10;cfg.global.dt=.0005f;cfg.global.source_i=12;cfg.global.source_j=10;
    cfg.global.lambda=lam;cfg.global.mu=mu;cfg.global.rho=rho;cfg.global.source_samples=source;
    cfg.global.receiver_count=RECEIVERS;cfg.global.receiver_i=ri;cfg.global.receiver_j=rj;
    cfg.global.l=0;cfg.global.invmat1=3;cfg.global.fdorder=4;cfg.global.ndt=1;cfg.global.dtinv=1;
    cfg.global.mpi_size=px*py;cfg.global.receiver_components=2;
    /* Replay fixture includes CPML and interfaces; the confirmed FULL fixture
     * remains FW=0 and reproduces the original 1536/3072/4800-byte sites. */
    if(segments){cfg.global.fw=3;cfg.global.cpml_enabled=1;
        cfg.global.pml_damping_speed=2500;cfg.global.pml_reflection=.001f;
        cfg.global.pml_power=2;cfg.global.pml_kmax=1;cfg.global.pml_fpml=15;}
    counting=0;require(denise_elastic_psv_born_mpi_create(&cfg,&context)==0);
    if(segments)require(denise_elastic_psv_born_mpi_set_replay_segments(context,segments)==0);
    receivers=denise_elastic_psv_born_mpi_local_receivers(context);
    data_count=(size_t)NT*receivers*2;
    /* Re-prepare from a successful prepared state, so failed Prepare must
     * invalidate the old success state as well as release its new temporaries. */
    require(denise_elastic_psv_born_mpi_prepare(context,NULL)==0);
    allocation_index=0;fail_rank=target;fail_index=index;injected=0;counting=1;
    if(!strcmp(phase,"prepare"))status=denise_elastic_psv_born_mpi_prepare(context,output);
    else if(!strcmp(phase,"j"))status=denise_elastic_psv_born_mpi_apply_j(context,dl,dm,output);
    else if(!strcmp(phase,"jt"))status=denise_elastic_psv_born_mpi_apply_jt(context,data,gl,gm);
    else {require(!strcmp(phase,"checkpoint"));status=denise_elastic_psv_born_mpi_checkpoint_roundtrip(context,4);}
    counting=0;result.status=status;result.calls=allocation_index;result.injected=injected;
    result.prepared=denise_elastic_psv_born_mpi_is_prepared(context);
    result.output_ok=1;
    if(index) {
        if(!strcmp(phase,"prepare")||!strcmp(phase,"j"))result.output_ok=clean_float(output,data_count);
        if(!strcmp(phase,"jt"))result.output_ok=clean_double(gl,cells)&&clean_double(gm,cells);
    }
    denise_elastic_psv_born_mpi_destroy(&context);
    result.destroyed=context==NULL;result.bytes=live_bytes;result.allocations=live_count;result.invalid_free=bad_free;
    return result;
}
static int record(FILE *file,const char *phase,int target,int index,int size,
                  struct result local,int *first) {
    struct result all[4];int i,bad=0,global;
    MPI_Gather(&local,(int)sizeof(local),MPI_BYTE,all,(int)sizeof(local),MPI_BYTE,0,MPI_COMM_WORLD);
    bad=local.bytes!=0||local.allocations!=0||local.invalid_free!=0||!local.destroyed||!local.output_ok;
    bad|=index?(local.status==0||local.injected!=(rank==target)):(local.status!=0||local.injected!=0);
    if(!strcmp(phase,"prepare"))bad|=local.prepared!=(index==0);
    else bad|=local.prepared!=1;
    MPI_Allreduce(&bad,&global,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(rank==0) {
        fprintf(file,"%s{\"target\":%d,\"index\":%d,\"ranks\":[",*first?"":",",target,index);*first=0;
        for(i=0;i<size;i++)fprintf(file,"%s{\"rank\":%d,\"status\":%d,\"calls\":%d,\"injected\":%d,\"outstanding_bytes\":%llu,\"outstanding_allocations\":%llu,\"invalid_free\":%d,\"prepared\":%d,\"output_ok\":%d,\"destroyed\":%d}",i?",":"",i,all[i].status,all[i].calls,all[i].injected,all[i].bytes,all[i].allocations,all[i].invalid_free,all[i].prepared,all[i].output_ok,all[i].destroyed);
        fprintf(file,"],\"pass\":%s}",global?"false":"true");fflush(file);
    }
    return global;
}
int main(int argc,char **argv) {
    FILE *file=NULL;int size,px,py,segments,target,index,first=1,bad=0,t,k,counts[4];
    struct result result;
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc==2&&!strcmp(argv[1],"control")){MPI_Finalize();return 0;}
    require(argc==8);px=atoi(argv[2]);py=atoi(argv[3]);segments=atoi(argv[4]);
    target=atoi(argv[5]);index=atoi(argv[6]);require(px*py==size&&size<=4);
    if(rank==0){file=fopen(argv[7],"w");require(file!=NULL);fprintf(file,"{\"phase\":\"%s\",\"topology\":[%d,%d],\"segments\":%d,\"cases\":[",argv[1],px,py,segments);}
    if(index==-1) {
        result=run_case(argv[1],px,py,segments,0,0);
        bad=record(file,argv[1],0,0,size,result,&first);
        MPI_Allgather(&result.calls,1,MPI_INT,counts,1,MPI_INT,MPI_COMM_WORLD);
        for(t=0;t<size&&!bad;t++)for(k=1;k<=counts[t]&&!bad;k++) {
            result=run_case(argv[1],px,py,segments,t,k);
            bad=record(file,argv[1],t,k,size,result,&first);
        }
    } else {
        result=run_case(argv[1],px,py,segments,target,index);
        bad=record(file,argv[1],target,index,size,result,&first);
    }
    if(rank==0){fprintf(file,"],\"pass\":%s}\n",bad?"false":"true");fclose(file);}
    MPI_Finalize();
    /* Do not keep leaked candidate addresses reachable through the observer
     * when using this helper to demonstrate the pre-repair LeakSanitizer stack. */
    memset(live,0,sizeof(live));return bad?1:0;
}
