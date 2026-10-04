/* Focused FLUID-2 transaction adapter around the canonical MPI harness.
 * Actual production public J/nonlinear execute; all expected values remain
 * in the independent Python references. No alternative numerical/halo graph.
 */
#include "denise_elastic_psv_born_mpi.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int fluid2_j(struct denise_elastic_psv_born_mpi *,const float *,const float *,float *);
#define denise_elastic_psv_born_mpi_apply_j fluid2_j
#include "m9d2_elastic_psv_mpi_harness.c"
#undef denise_elastic_psv_born_mpi_apply_j

static int fluid2_j(struct denise_elastic_psv_born_mpi *c,const float *a,const float *b,float *data) {
    static int tested;
    struct denise_elastic_psv_born_mpi_diagnostics d,before,after;
    size_t p,n;
    int bad_rank=2147483647,owner,failed,all,local_bad;
    float *trial,*background_lam,*background_mu;
    if(!getenv("FLUID2_TRANSACTIONS") || tested)
        return denise_elastic_psv_born_mpi_apply_j(c,a,b,data);
    tested=1;
    check(denise_elastic_psv_born_mpi_diagnostics(c,&d),"diagnostics");
    n=(size_t)d.local_nx*d.local_ny;
    background_lam=malloc(n*sizeof(float));background_mu=malloc(n*sizeof(float));
    /* Find one actual fluid owner from the harness's copied owned background.
       A maps-zero corner is not sufficient to classify a physical cell. */
    {
        const char *input=getenv("FLUID2_INPUT");
        int meta[7],i,j;float params[7],*global_mu,*global_lam;
        FILE *f=fopen(input,"rb");
        if(!f)MPI_Abort(MPI_COMM_WORLD,41);
        rd(f,meta,7,sizeof(int));rd(f,params,7,sizeof(float));
        global_lam=malloc((size_t)meta[0]*meta[1]*sizeof(float));
        rd(f,global_lam,(size_t)meta[0]*meta[1],sizeof(float));
        global_mu=malloc((size_t)meta[0]*meta[1]*sizeof(float));
        rd(f,global_mu,(size_t)meta[0]*meta[1],sizeof(float));fclose(f);
        for(j=0;j<d.local_ny;j++)for(i=0;i<d.local_nx;i++) {
            size_t local=(size_t)j*d.local_nx+i;
            size_t global=(size_t)(d.offset_y+j)*meta[0]+d.offset_x+i;
            background_lam[local]=global_lam[global];
            background_mu[local]=global_mu[global];
        }
        p=n;
        for(j=0;j<d.local_ny;j++)for(i=0;i<d.local_nx;i++)
            if(global_mu[(size_t)(d.offset_y+j)*meta[0]+d.offset_x+i]==0.0f && p==n)
                p=(size_t)j*d.local_nx+i;
        if(p<n)bad_rank=d.rank;
        MPI_Allreduce(&bad_rank,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        free(global_mu);free(global_lam);
    }
    if(owner==2147483647)MPI_Abort(MPI_COMM_WORLD,42);
    trial=malloc(n*sizeof(float));memcpy(trial,b,n*sizeof(float));
    if(d.rank==owner)trial[p]=1.0f;
    for(n=0;n<d.local_data_bytes/sizeof(float);n++)data[n]=37.0f;
    check(denise_elastic_psv_born_mpi_diagnostics(c,&before),"before");
    failed=denise_elastic_psv_born_mpi_apply_j(c,a,trial,data)!=0;
    MPI_Allreduce(&failed,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!all)MPI_Abort(MPI_COMM_WORLD,43);
    check(denise_elastic_psv_born_mpi_diagnostics(c,&after),"after");
    local_bad=memcmp(&before,&after,sizeof(before))!=0;
    for(n=0;n<d.local_data_bytes/sizeof(float);n++)if(data[n]!=37.0f)local_bad=1;
    MPI_Allreduce(&local_bad,&all,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(all)MPI_Abort(MPI_COMM_WORLD,44);
    {
        int phase;
        for(phase=0;phase<2;phase++) {
            size_t cells=(size_t)d.local_nx*d.local_ny;
            int candidate=2147483647;
            p=cells;
            memcpy(trial,background_mu,cells*sizeof(float));
            for(n=0;n<cells;n++)if((background_mu[n]==0.0f)==(phase==0) && p==cells)p=n;
            if(p<cells)candidate=d.rank;
            MPI_Allreduce(&candidate,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
            if(owner==2147483647)MPI_Abort(MPI_COMM_WORLD,45);
            if(d.rank==owner)trial[p]=phase==0?1.0f:0.0f;
            for(n=0;n<d.local_data_bytes/sizeof(float);n++)data[n]=37.0f;
            check(denise_elastic_psv_born_mpi_diagnostics(c,&before),"trial before");
            failed=denise_elastic_psv_born_mpi_nonlinear(c,background_lam,trial,data)!=0;
            MPI_Allreduce(&failed,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
            if(!all)MPI_Abort(MPI_COMM_WORLD,46);
            check(denise_elastic_psv_born_mpi_diagnostics(c,&after),"trial after");
            local_bad=memcmp(&before,&after,sizeof(before))!=0;
            for(n=0;n<d.local_data_bytes/sizeof(float);n++)if(data[n]!=37.0f)local_bad=1;
            MPI_Allreduce(&local_bad,&all,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
            if(all)MPI_Abort(MPI_COMM_WORLD,47);
            check(denise_elastic_psv_born_mpi_nonlinear(c,background_lam,background_mu,data),"trial recovery");
        }
    }
    free(trial);free(background_lam);free(background_mu);
    if(!d.rank)printf("FLUID2_MPI_BOTH_TOPOLOGY_TRIALS_TRANSACTION_RECOVERY_PASS\n");
    if(!d.rank)printf("FLUID2_MPI_INVALID_SINGLE_OWNER_BEFORE_MUTATION_PASS\n");
    return denise_elastic_psv_born_mpi_apply_j(c,a,b,data);
}
