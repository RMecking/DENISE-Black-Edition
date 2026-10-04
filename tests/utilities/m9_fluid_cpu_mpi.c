/* Actual CPU MPI fluid F and collective operator boundary harness. */
#include "../../src/PSV/elastic_psv_born_mpi.c"
#include <fenv.h>
static int rank;
static void require(int ok) { if(!ok){fprintf(stderr,"rank%d: %s\n",rank,denise_elastic_psv_born_mpi_last_error());MPI_Abort(MPI_COMM_WORLD,31);} }
static void read_values(FILE *f,void *p,size_t n,size_t s) { require(fread(p,s,n,f)==n); }
static void write_values(const char *dir,const char *name,const void *p,size_t n,size_t s) {
    char path[2048];FILE *f;
    if(rank)return;
    snprintf(path,sizeof(path),"%s/%s",dir,name);f=fopen(path,"wb");require(f!=NULL);
    require(fwrite(p,s,n,f)==n);require(fclose(f)==0);
}
int main(int argc,char **argv) {
    int h[7],size,px,py,segments,k,nx,ny,j,i,nr,invalid,owner,candidate;float p[7];
    float *global[5],*local[5],*src,*data,*again,*out,*q;
    int *ri,*rj;size_t n,gn,dc,fluid_cell;FILE *f;
    double *images,*assembled;
    struct denise_elastic_psv_born_mpi_config cfg;
    struct denise_elastic_psv_born_mpi *c=NULL;
    struct denise_elastic_psv_born_storage_diagnostics before,after;
    struct denise_elastic_psv_born_mpi_diagnostics work_before,work_after;
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    require(argc==6);px=atoi(argv[3]);py=atoi(argv[4]);segments=atoi(argv[5]);
    f=fopen(argv[1],"rb");require(f!=NULL);read_values(f,h,7,sizeof(int));read_values(f,p,7,sizeof(float));
    nx=h[0]/px;ny=h[1]/py;n=(size_t)nx*ny;gn=(size_t)h[0]*h[1];
    for(k=0;k<5;k++){
        global[k]=malloc(gn*4);local[k]=malloc(n*4);require(global[k]&&local[k]);
        read_values(f,global[k],gn,4);
        for(j=0;j<ny;j++)for(i=0;i<nx;i++)
            local[k][(size_t)j*nx+i]=global[k][(size_t)(rank/px*ny+j)*h[0]+rank%px*nx+i];
    }
    src=malloc((size_t)h[2]*4);ri=malloc((size_t)h[6]*sizeof(int));rj=malloc((size_t)h[6]*sizeof(int));require(src&&ri&&rj);
    read_values(f,src,h[2],4);read_values(f,ri,h[6],sizeof(int));read_values(f,rj,h[6],sizeof(int));fclose(f);
    memset(&cfg,0,sizeof(cfg));cfg.communicator=MPI_COMM_WORLD;cfg.nprocx=px;cfg.nprocy=py;
    cfg.global.nx=h[0];cfg.global.ny=h[1];cfg.global.nt=h[2];cfg.global.fw=h[3];
    cfg.global.source_i=h[4];cfg.global.source_j=h[5];cfg.global.receiver_count=h[6];
    cfg.global.dh=p[0];cfg.global.dt=p[1];cfg.global.pml_damping_speed=p[2];cfg.global.pml_reflection=p[3];
    cfg.global.pml_power=p[4];cfg.global.pml_kmax=p[5];cfg.global.pml_fpml=p[6];
    cfg.global.invmat1=3;cfg.global.fdorder=4;cfg.global.ndt=cfg.global.dtinv=1;
    cfg.global.receiver_components=2;cfg.global.mpi_size=size;cfg.global.cpml_enabled=h[3]>0;
    cfg.global.free_surface=getenv("FLUID_FS")?atoi(getenv("FLUID_FS")):0;
    cfg.global.lambda=local[0];cfg.global.mu=local[1];cfg.global.rho=local[2];
    cfg.global.source_samples=src;cfg.global.receiver_i=ri;cfg.global.receiver_j=rj;
    invalid=getenv("FLUID_NEGATIVE")!=NULL;
    if(invalid&&rank==0)local[1][0]=-1;
    feclearexcept(FE_ALL_EXCEPT);
    k=denise_elastic_psv_born_mpi_create(&cfg,&c);
    if(invalid){require(k!=0&&c==NULL);if(!rank)puts("FLUID_MPI_NEGATIVE_COLLECTIVE_PASS");goto cleanup;}
    require(k==0);
    if(segments)require(denise_elastic_psv_born_mpi_set_replay_segments(c,segments)==0);
    nr=denise_elastic_psv_born_mpi_local_receivers(c);dc=(size_t)h[2]*nr*2;
    data=calloc(dc+1,4);again=calloc(dc+1,4);out=malloc(((size_t)h[2]*h[6]*2+1)*4);
    images=malloc(n*2*sizeof(double));assembled=malloc(gn*sizeof(double));q=malloc(4*n*4);
    require(data&&again&&out&&images&&assembled&&q);
    /* Valid pointers/directions, but no background has been prepared. The
       public MPI preflight uses a collective diagnostic for this lifecycle
       failure; do not expect the removed FLUID-2 capability diagnostic. */
    require(!denise_elastic_psv_born_mpi_is_prepared(c));
    require(denise_elastic_psv_born_mpi_diagnostics(c,&work_before)==0);
    for(k=0;k<(int)dc;k++)again[k]=37;
    for(k=0;k<(int)(2*n);k++)images[k]=41;
    require(denise_elastic_psv_born_mpi_apply_j(c,local[3],local[4],again)!=0);
    require(strcmp(denise_elastic_psv_born_mpi_last_error(),"MPI elastic P/SV collective failure (including a remote rank)")==0);
    require(denise_elastic_psv_born_mpi_apply_jt(c,data,images,images+n)!=0);
    require(strcmp(denise_elastic_psv_born_mpi_last_error(),"MPI elastic P/SV collective failure (including a remote rank)")==0);
    for(k=0;k<(int)dc;k++)require(again[k]==37);
    for(k=0;k<(int)(2*n);k++)require(images[k]==41);
    require(denise_elastic_psv_born_mpi_diagnostics(c,&work_after)==0);
    require(memcmp(&work_before,&work_after,sizeof(work_before))==0);
    require(!denise_elastic_psv_born_mpi_is_prepared(c));
    if(!rank)puts("FLUID_MPI_UNPREPARED_LIFECYCLE_NO_WORK_PASS");
    for(k=0;k<3;k++) {
        float *map=k==0?c->invrho_x:k==1?c->invrho_y:c->mu_corner;size_t z;
        for(z=0;z<c->cells;z++)if(owned(c,z))images[compact(c,z)]=map[z];
        require(denise_elastic_psv_born_mpi_gather_image(c,images,assembled)==0);
        write_values(argv[2],k==0?"rx.bin":k==1?"ry.bin":"corner.bin",assembled,gn,sizeof(double));
    }
    require(denise_elastic_psv_born_mpi_prepare(c,data)==0);
    require(denise_elastic_psv_born_mpi_gather_data(c,data,out)==0);
    write_values(argv[2],"data.bin",out,(size_t)h[2]*h[6]*2,4);
    require(denise_elastic_psv_born_mpi_copy_strain(c,h[2]-1,q)==0);
    require(denise_elastic_psv_born_mpi_checkpoint_roundtrip(c,h[2]/2)==0);
    require(fetestexcept(FE_DIVBYZERO|FE_INVALID)==0);
    /* Prepared restricted J/JT are now supported on these same old fixtures. */
    for(k=0;k<(int)n;k++) {
        local[3][k]=0.001f*local[0][k];
        local[4][k]=local[1][k]==0.0f?0.0f:0.001f*local[1][k];
    }
    require(denise_elastic_psv_born_mpi_apply_j(c,local[3],local[4],again)==0);
    require(strstr(denise_elastic_psv_born_mpi_last_error(),"FLUID-2")==NULL);
    for(k=0;k<(int)dc;k++)require(isfinite(again[k]));
    require(denise_elastic_psv_born_mpi_apply_jt(c,data,images,images+n)==0);
    require(strstr(denise_elastic_psv_born_mpi_last_error(),"FLUID-2")==NULL);
    for(k=0;k<(int)n;k++) {
        require(isfinite(images[k])&&isfinite(images[n+k]));
        if(local[1][k]==0.0f)require(images[n+k]==0.0&&!signbit(images[n+k]));
    }
    require(denise_elastic_psv_born_mpi_is_prepared(c));
    require(denise_elastic_psv_born_mpi_storage_diagnostics(c,&after)==0);
    require(after.initial_forward_steps==(size_t)h[2]);
    require(after.replayed_forward_steps_last==(segments?(size_t)h[2]:0));
    require(fetestexcept(FE_DIVBYZERO|FE_INVALID)==0);
    if(!rank)puts("FLUID_MPI_PREPARED_RESTRICTED_J_JT_PASS");
    /* Exactly one owned physical fluid cell is invalid, not every fluid J. */
    fluid_cell=n;candidate=size;
    for(k=0;k<(int)n;k++)if(local[1][k]==0.0f&&fluid_cell==n) {
        fluid_cell=(size_t)k;candidate=rank;
    }
    MPI_Allreduce(&candidate,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    require(owner<size);
    if(rank==owner)local[4][fluid_cell]=1.0f;
    for(k=0;k<=(int)dc;k++)again[k]=37;
    require(denise_elastic_psv_born_mpi_storage_diagnostics(c,&before)==0);
    require(denise_elastic_psv_born_mpi_diagnostics(c,&work_before)==0);
    require(denise_elastic_psv_born_mpi_apply_j(c,local[3],local[4],again)!=0);
    require(strstr(denise_elastic_psv_born_mpi_last_error(),"nonzero fluid dMu")!=NULL);
    for(k=0;k<=(int)dc;k++)require(again[k]==37);
    require(denise_elastic_psv_born_mpi_storage_diagnostics(c,&after)==0);
    require(memcmp(&before,&after,sizeof(before))==0);
    require(denise_elastic_psv_born_mpi_diagnostics(c,&work_after)==0);
    require(memcmp(&work_before,&work_after,sizeof(work_before))==0);
    if(rank==owner)local[4][fluid_cell]=0.0f;
    require(denise_elastic_psv_born_mpi_apply_j(c,local[3],local[4],again)==0);
    for(k=0;k<(int)dc;k++)require(isfinite(again[k]));
    if(!rank)puts("FLUID_MPI_SINGLE_OWNER_INVALID_DMU_TRANSACTION_RECOVERY_PASS");
    /* One-rank-only invalid trial; all ranks reject before material transport. */
    for(k=0;k<(int)dc;k++)again[k]=37;
    require(denise_elastic_psv_born_mpi_storage_diagnostics(c,&before)==0);
    if(rank==0)local[1][0]=local[1][0]==0?1:0;
    require(denise_elastic_psv_born_mpi_nonlinear(c,local[0],local[1],again)!=0);
    for(k=0;k<(int)dc;k++)require(again[k]==37);
    require(denise_elastic_psv_born_mpi_storage_diagnostics(c,&after)==0);
    require(memcmp(&before,&after,sizeof(before))==0);
    for(j=0;j<ny;j++)for(i=0;i<nx;i++)local[1][(size_t)j*nx+i]=global[1][(size_t)(rank/px*ny+j)*h[0]+rank%px*nx+i];
    require(denise_elastic_psv_born_mpi_nonlinear(c,local[0],local[1],again)==0);
    require(memcmp(data,again,dc*4)==0);
    require(denise_elastic_psv_born_mpi_prepare(c,again)==0);
    require(memcmp(data,again,dc*4)==0);
    if(!rank)puts("FLUID_MPI_FORWARD_MAPS_BOUNDARY_RECOVERY_PASS");
    free(q);free(data);free(again);free(out);free(images);free(assembled);
cleanup:
    denise_elastic_psv_born_mpi_destroy(&c);
    for(k=0;k<5;k++){free(global[k]);free(local[k]);}
    free(src);free(ri);free(rj);MPI_Finalize();return 0;
}
