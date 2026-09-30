#include "denise_elastic_psv_born_mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
/* Including the immutable serial implementation here exposes material/profile
 * maps for a direct slice comparison. This is a test executable, never DENISE. */
#include "../../src/PSV/elastic_psv_born.c"

static int rank;
static int allocation_count,fail_at;
void *__real_calloc(size_t,size_t);
void *__wrap_calloc(size_t count,size_t width){
    if(fail_at && ++allocation_count==fail_at)return NULL;
    return __real_calloc(count,width);
}
static void arm(const char *phase){
    const char *requested=getenv("M9D2_FAIL_PHASE"),*value=getenv("M9D2_FAIL_AT"),*which=getenv("M9D2_FAIL_RANK");
    allocation_count=0;fail_at=requested&&!strcmp(requested,phase)&&rank==(which?atoi(which):1)?(value?atoi(value):1):0;
}
static int expected_failure(int status,const char *phase){
    const char *requested=getenv("M9D2_FAIL_PHASE");int failure=status!=0,all;
    if(!requested||strcmp(requested,phase))return 0;
    MPI_Allreduce(&failure,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!all)MPI_Abort(MPI_COMM_WORLD,19);
    if(rank==0)printf("M9D2_EXPECTED_COLLECTIVE_FAILURE\n");
    return 1;
}
static void check(int status,const char *phase) {
    if(status){fprintf(stderr,"rank %d %s: %s\n",rank,phase,denise_elastic_psv_born_mpi_last_error());MPI_Abort(MPI_COMM_WORLD,3);}
}
static void rd(FILE *f,void *p,size_t n,size_t width){if(fread(p,width,n,f)!=n)MPI_Abort(MPI_COMM_WORLD,4);}
static void wr(const char *dir,const char *name,const void *p,size_t n,size_t width){
    char path[2048];FILE *f;
    if(rank)return;
    snprintf(path,sizeof(path),"%s/%s",dir,name);f=fopen(path,"wb");
    if(!f||fwrite(p,width,n,f)!=n||fclose(f))MPI_Abort(MPI_COMM_WORLD,5);
}
static void scatter(float *global,float *local,int gx,int nx,int ny,int px,int size){
    float *packed=NULL;int r,j;
    if(rank==0){packed=malloc((size_t)nx*ny*size*sizeof(float));for(r=0;r<size;r++)for(j=0;j<ny;j++)memcpy(packed+(size_t)r*nx*ny+(size_t)j*nx,global+(size_t)(r/px*ny+j)*gx+r%px*nx,(size_t)nx*sizeof(float));}
    MPI_Scatter(packed,nx*ny,MPI_FLOAT,local,nx*ny,MPI_FLOAT,0,MPI_COMM_WORLD);free(packed);
}
int main(int argc,char **argv){
    int meta[7],size,px,py,s,r,k,j,nx,ny,nrec,segments;
    float params[7],*global[5]={NULL,NULL,NULL,NULL,NULL},*local[5],*source,*data=NULL,*lData,*gData=NULL,*maps[3];
    int *ri,*rj;
    double *gl,*gm,*assembled=NULL;
    struct denise_elastic_psv_born_mpi_config cfg;
    struct denise_elastic_psv_born_mpi *c=NULL;
    struct denise_elastic_psv_born *serial=NULL;
    struct denise_elastic_psv_born_config serialcfg;
    struct denise_elastic_psv_born_mpi_diagnostics diag,*all_diag=NULL;
    FILE *f;size_t cells,dc,lc;double start,elapsed;
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc!=6)MPI_Abort(MPI_COMM_WORLD,2);
    px=atoi(argv[3]);py=atoi(argv[4]);segments=atoi(argv[5]);
    if(rank==0){f=fopen(argv[1],"rb");if(!f)MPI_Abort(MPI_COMM_WORLD,2);rd(f,meta,7,sizeof(int));rd(f,params,7,sizeof(float));}else f=NULL;
    MPI_Bcast(meta,7,MPI_INT,0,MPI_COMM_WORLD);MPI_Bcast(params,7,MPI_FLOAT,0,MPI_COMM_WORLD);
    cells=(size_t)meta[0]*meta[1];nrec=meta[6];dc=(size_t)meta[2]*nrec*2;
    nx=meta[0]/px;ny=meta[1]/py;lc=(size_t)nx*ny;
    for(k=0;k<5;k++){local[k]=calloc(lc,sizeof(float));if(rank==0){global[k]=malloc(cells*sizeof(float));rd(f,global[k],cells,sizeof(float));}scatter(global[k],local[k],meta[0],nx,ny,px,size);}
    source=malloc((size_t)meta[2]*sizeof(float));ri=malloc((size_t)nrec*sizeof(int));rj=malloc((size_t)nrec*sizeof(int));
    if(rank==0){rd(f,source,(size_t)meta[2],sizeof(float));rd(f,ri,(size_t)nrec,sizeof(int));rd(f,rj,(size_t)nrec,sizeof(int));data=malloc(dc*sizeof(float));rd(f,data,dc,sizeof(float));fclose(f);gData=malloc(dc*sizeof(float));assembled=malloc(cells*sizeof(double));all_diag=malloc((size_t)size*sizeof(diag));}
    MPI_Bcast(source,meta[2],MPI_FLOAT,0,MPI_COMM_WORLD);MPI_Bcast(ri,nrec,MPI_INT,0,MPI_COMM_WORLD);MPI_Bcast(rj,nrec,MPI_INT,0,MPI_COMM_WORLD);
    memset(&cfg,0,sizeof(cfg));cfg.communicator=MPI_COMM_WORLD;cfg.nprocx=px;cfg.nprocy=py;
    cfg.global.nx=meta[0];cfg.global.ny=meta[1];cfg.global.nt=meta[2];cfg.global.fw=meta[3];cfg.global.source_i=meta[4];cfg.global.source_j=meta[5];cfg.global.receiver_count=nrec;
    cfg.global.dh=params[0];cfg.global.dt=params[1];cfg.global.pml_damping_speed=params[2];cfg.global.pml_reflection=params[3];cfg.global.pml_power=params[4];cfg.global.pml_kmax=params[5];cfg.global.pml_fpml=params[6];
    cfg.global.l=0;cfg.global.invmat1=3;cfg.global.fdorder=4;cfg.global.ndt=cfg.global.dtinv=1;cfg.global.mpi_size=size;cfg.global.receiver_components=2;cfg.global.cpml_enabled=meta[3]>0;
    cfg.global.lambda=local[0];cfg.global.mu=local[1];cfg.global.rho=local[2];cfg.global.source_samples=source;cfg.global.receiver_i=ri;cfg.global.receiver_j=rj;
    if(getenv("M9D2_INVALID")&&rank==1){
        const char *fault=getenv("M9D2_INVALID");
        if(!strcmp(fault,"topology"))cfg.nprocy++;
        else if(!strcmp(fault,"divisibility"))cfg.global.nx++;
        else if(!strcmp(fault,"source"))cfg.global.source_i++;
        else if(!strcmp(fault,"receiver"))ri[0]=-1;
        else if(!strcmp(fault,"model"))local[1][0]=-1;
        else if(!strcmp(fault,"free_surface"))cfg.global.free_surface=1;
        else if(!strcmp(fault,"boundary"))cfg.global.boundary=1;
        else if(!strcmp(fault,"size"))cfg.global.mpi_size++;
        else if(!strcmp(fault,"minimum"))cfg.global.nx=px;
    }
    arm("create");s=denise_elastic_psv_born_mpi_create(&cfg,&c);fail_at=0;
    if(expected_failure(s,"create")){MPI_Finalize();return 0;}
    check(s,"create");
    if(rank==0){serialcfg=cfg.global;serialcfg.mpi_size=1;serialcfg.lambda=global[0];serialcfg.mu=global[1];serialcfg.rho=global[2];if(denise_elastic_psv_born_create(&serialcfg,&serial))MPI_Abort(MPI_COMM_WORLD,6);}
    for(k=0;k<3;k++)maps[k]=malloc(lc*sizeof(float));
    check(denise_elastic_psv_born_mpi_copy_maps(c,maps[0],maps[1],maps[2]),"maps");
    gl=malloc(lc*sizeof(double));gm=malloc(lc*sizeof(double));
    for(k=0;k<3;k++){
        for(s=0;s<(int)lc;s++)gl[s]=maps[k][s];
        check(denise_elastic_psv_born_mpi_gather_image(c,gl,assembled),"gather maps");
        if(rank==0){float *ref=k==0?serial->invrho_x:k==1?serial->invrho_y:serial->mu_corner;for(s=0;s<(int)cells;s++)if(assembled[s]!=(double)ref[s])MPI_Abort(MPI_COMM_WORLD,7);}
    }
    for(k=0;k<4;k++){
        int len=k<2?nx:ny,offset=k<2?(rank%px)*nx:(rank/px)*ny;
        float *ka=malloc((size_t)len*sizeof(float)),*a=malloc((size_t)len*sizeof(float)),*b=malloc((size_t)len*sizeof(float));
        struct pml_profile profile={NULL,NULL,NULL,0};
        check(denise_elastic_psv_born_mpi_copy_profile(c,k,ka,a,b),"profile");
        build_profile(&profile,k<2?meta[0]:meta[1],params[0],params[1],meta[3],k%2,params[2],params[3],params[4],params[5],params[6]);
        if(memcmp(ka,profile.kappa+offset,(size_t)len*sizeof(float))||memcmp(a,profile.a+offset,(size_t)len*sizeof(float))||memcmp(b,profile.b+offset,(size_t)len*sizeof(float)))MPI_Abort(MPI_COMM_WORLD,8);
        free_profile(&profile);free(ka);free(a);free(b);
    }
    if(segments<0)check(denise_elastic_psv_born_mpi_select_backend(c,-segments),"select");
    else if(segments)check(denise_elastic_psv_born_mpi_set_replay_segments(c,segments),"segments");
    r=denise_elastic_psv_born_mpi_local_receivers(c);lData=calloc((size_t)meta[2]*(r?r:1)*2,sizeof(float));
    start=MPI_Wtime();
    arm("prepare");s=denise_elastic_psv_born_mpi_prepare(c,lData);fail_at=0;
    if(expected_failure(s,"prepare")){denise_elastic_psv_born_mpi_destroy(&c);MPI_Finalize();return 0;}
    check(s,"prepare");
    check(denise_elastic_psv_born_mpi_gather_data(c,lData,gData),"gather background");wr(argv[2],"background.bin",gData,dc,sizeof(float));
    if(rank==0){denise_elastic_psv_born_prepare(serial,gData);wr(argv[2],"serial_background.bin",gData,dc,sizeof(float));}
    check(denise_elastic_psv_born_mpi_checkpoint_roundtrip(c,meta[2]/2),"roundtrip");
    arm("j");s=denise_elastic_psv_born_mpi_apply_j(c,local[3],local[4],lData);fail_at=0;
    if(expected_failure(s,"j")){denise_elastic_psv_born_mpi_destroy(&c);MPI_Finalize();return 0;}
    check(s,"J");
    check(denise_elastic_psv_born_mpi_gather_data(c,lData,gData),"gather J");wr(argv[2],"j.bin",gData,dc,sizeof(float));
    if(rank==0){denise_elastic_psv_born_apply_j(serial,global[3],global[4],gData);wr(argv[2],"serial_j.bin",gData,dc,sizeof(float));}
    check(denise_elastic_psv_born_mpi_scatter_data(c,data,lData),"scatter data");
    arm("jt");s=denise_elastic_psv_born_mpi_apply_jt(c,lData,gl,gm);fail_at=0;
    if(expected_failure(s,"jt")){denise_elastic_psv_born_mpi_destroy(&c);MPI_Finalize();return 0;}
    check(s,"JT");
    check(denise_elastic_psv_born_mpi_gather_image(c,gl,assembled),"gather lambda");wr(argv[2],"gl.bin",assembled,cells,sizeof(double));
    check(denise_elastic_psv_born_mpi_gather_image(c,gm,assembled),"gather mu");wr(argv[2],"gm.bin",assembled,cells,sizeof(double));
    elapsed=MPI_Wtime()-start;
    /* JT->JT, JT->J, J->JT and A->B->A reuse must match this same context. */
    {
        double *saved_l=malloc(lc*sizeof(double)),*saved_m=malloc(lc*sizeof(double));
        float *saved_j=malloc((size_t)meta[2]*(r?r:1)*2*sizeof(float));
        int repeat=getenv("M9D2_REPEAT")?atoi(getenv("M9D2_REPEAT")):1;
        memcpy(saved_l,gl,lc*sizeof(double));memcpy(saved_m,gm,lc*sizeof(double));
        {
            struct denise_elastic_psv_born_mpi *fresh=NULL;
            check(denise_elastic_psv_born_mpi_create(&cfg,&fresh),"fresh context");
            if(segments<0)check(denise_elastic_psv_born_mpi_select_backend(fresh,-segments),"fresh select");
            else if(segments)check(denise_elastic_psv_born_mpi_set_replay_segments(fresh,segments),"fresh segments");
            check(denise_elastic_psv_born_mpi_prepare(fresh,NULL),"fresh prepare");
            check(denise_elastic_psv_born_mpi_apply_jt(fresh,lData,gl,gm),"prepare->JT");
            if(memcmp(saved_l,gl,lc*sizeof(double))||memcmp(saved_m,gm,lc*sizeof(double)))MPI_Abort(MPI_COMM_WORLD,15);
            check(denise_elastic_psv_born_mpi_apply_jt(fresh,lData,gl,gm),"prepare->JT->JT");
            if(memcmp(saved_l,gl,lc*sizeof(double))||memcmp(saved_m,gm,lc*sizeof(double)))MPI_Abort(MPI_COMM_WORLD,16);
            denise_elastic_psv_born_mpi_destroy(&fresh);
        }
        for(s=0;s<repeat;s++){
            check(denise_elastic_psv_born_mpi_apply_jt(c,lData,gl,gm),"repeat JT");
            if(memcmp(saved_l,gl,lc*sizeof(double))||memcmp(saved_m,gm,lc*sizeof(double)))MPI_Abort(MPI_COMM_WORLD,12);
            check(denise_elastic_psv_born_mpi_apply_j(c,local[3],local[4],saved_j),"JT->J");
            check(denise_elastic_psv_born_mpi_apply_jt(c,lData,gl,gm),"J->JT");
            if(memcmp(saved_l,gl,lc*sizeof(double))||memcmp(saved_m,gm,lc*sizeof(double)))MPI_Abort(MPI_COMM_WORLD,13);
            for(k=0;k<meta[2]*r*2;k++)lData[k]=-lData[k];
            check(denise_elastic_psv_born_mpi_apply_jt(c,lData,gl,gm),"dataset B");
            for(k=0;k<meta[2]*r*2;k++)lData[k]=-lData[k];
        }
        check(denise_elastic_psv_born_mpi_apply_jt(c,lData,gl,gm),"dataset A");
        if(memcmp(saved_l,gl,lc*sizeof(double))||memcmp(saved_m,gm,lc*sizeof(double)))MPI_Abort(MPI_COMM_WORLD,14);
        free(saved_l);free(saved_m);free(saved_j);
    }
    /* Dense nonlinear centered finite difference, no sign/scale/time fit. */
    for(s=0;s<2;s++){
        float *la=malloc(lc*sizeof(float)),*mu=malloc(lc*sizeof(float));
        for(k=0;k<(int)lc;k++){la[k]=local[0][k]+(s?-.05f:.05f)*local[3][k];mu[k]=local[1][k]+(s?-.05f:.05f)*local[4][k];}
        check(denise_elastic_psv_born_mpi_nonlinear(c,la,mu,lData),"dense FD");
        check(denise_elastic_psv_born_mpi_gather_data(c,lData,gData),"gather FD");wr(argv[2],s?"minus.bin":"plus.bin",gData,dc,sizeof(float));free(la);free(mu);
    }
    /* Strain assembly and rank-coded unique-owner gather proof. */
    {
        float *strains=malloc(4*lc*sizeof(float));
        check(denise_elastic_psv_born_mpi_copy_strain(c,meta[2]/2,strains),"copy strain");
        for(k=0;k<4;k++){
            char name[64];for(s=0;s<(int)lc;s++)gl[s]=strains[(size_t)k*lc+s];
            check(denise_elastic_psv_born_mpi_gather_image(c,gl,assembled),"gather strain");snprintf(name,sizeof(name),"strain%d.bin",k);wr(argv[2],name,assembled,cells,sizeof(double));
        }
        for(s=0;s<(int)lc;s++)gl[s]=rank*1000000+s;
        check(denise_elastic_psv_born_mpi_gather_image(c,gl,assembled),"rank-coded gather");wr(argv[2],"rank_codes.bin",assembled,cells,sizeof(double));free(strains);
    }
    if(rank==0){double *sl=malloc(cells*sizeof(double)),*sm=malloc(cells*sizeof(double));denise_elastic_psv_born_apply_jt(serial,data,sl,sm);wr(argv[2],"serial_gl.bin",sl,cells,sizeof(double));wr(argv[2],"serial_gm.bin",sm,cells,sizeof(double));free(sl);free(sm);}
    check(denise_elastic_psv_born_mpi_diagnostics(c,&diag),"diagnostics");
    MPI_Gather(&diag,(int)sizeof(diag),MPI_BYTE,all_diag,(int)sizeof(diag),MPI_BYTE,0,MPI_COMM_WORLD);
    if(rank==0){char path[2048];snprintf(path,sizeof(path),"%s/diagnostics.json",argv[2]);f=fopen(path,"w");fprintf(f,"{\"elapsed\":%.9g,\"ranks\":[",elapsed);for(s=0;s<size;s++){struct denise_elastic_psv_born_mpi_diagnostics *d=&all_diag[s];fprintf(f,"%s{\"rank\":%d,\"full\":%lu,\"estimate\":%lu,\"retained\":%lu,\"segmented\":%d,\"payload\":%lu,\"metadata\":%lu,\"schedule\":%lu,\"operand\":%lu,\"replayed\":%lu,\"local_data\":%lu,\"forward_halo\":%lu,\"adjoint_halo\":%lu,\"material\":%lu}",s?",":"",s,(unsigned long)d->full_bytes,(unsigned long)d->replay_estimate,(unsigned long)d->retained_backend_bytes,d->replay.segmented,(unsigned long)d->replay.checkpoint_bytes,(unsigned long)d->replay.checkpoint_metadata_bytes,(unsigned long)d->replay.segment_schedule_bytes,(unsigned long)d->replay.segment_operand_bytes,(unsigned long)d->replay.replayed_forward_steps_last,(unsigned long)d->local_data_bytes,(unsigned long)d->forward_halo_bytes_per_step,(unsigned long)d->adjoint_halo_bytes_per_step,(unsigned long)d->material_bytes);}fprintf(f,"]}\n");fclose(f);}
    denise_elastic_psv_born_mpi_destroy(&c);if(rank==0)denise_elastic_psv_born_destroy(&serial);
    for(k=0;k<5;k++){free(global[k]);free(local[k]);}for(k=0;k<3;k++)free(maps[k]);
    free(gl);free(gm);free(assembled);free(data);free(lData);free(gData);free(source);free(ri);free(rj);free(all_diag);
    (void)j;MPI_Finalize();return 0;
}
