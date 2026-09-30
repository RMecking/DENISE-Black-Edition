#include "denise_elastic_psv_migration_mpi.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>
static int status_all(MPI_Comm comm,int local) {
    int any;MPI_Allreduce(&local,&any,1,MPI_INT,MPI_MAX,comm);return any?-1:0;
}
int denise_elastic_psv_migrate_mpi(
        const struct denise_elastic_psv_migration_request *q,MPI_Comm comm,int px,int py,
        struct denise_elastic_psv_migration_result *out,
        struct denise_elastic_psv_born_mpi_diagnostics *diagnostics) {
    struct denise_elastic_psv_born_mpi *c=NULL;
    struct denise_elastic_psv_born_mpi_config config;
    struct denise_elastic_psv_born_mpi_diagnostics d;
    double *sl=NULL,*sm=NULL;size_t cells=0,p;int s,previous=0,size,bad=0,min,max;
    if(comm==MPI_COMM_NULL)return -1;
    MPI_Comm_size(comm,&size);
    if(out)memset(out,0,sizeof(*out));
    bad=!q||!out||!diagnostics;
    if(status_all(comm,bad))return -1;
    {
        int values[6]={px,py,q->nx,q->ny,q->nt,q->mpi_size},lo[6],hi[6],k;
        MPI_Allreduce(values,lo,6,MPI_INT,MPI_MIN,comm);MPI_Allreduce(values,hi,6,MPI_INT,MPI_MAX,comm);
        for(k=0;k<6;k++)if(lo[k]!=hi[k])bad=1;
        if(status_all(comm,bad))return -1;
    }
    bad=px<1||py<1||(long long)px*py!=size||q->nx<5||q->ny<5||q->nt<1||q->inv_stf!=0||q->shot_count<1||!q->shots;
    if(px>0&&py>0)bad|=q->nx%px!=0||q->ny%py!=0||q->nx/px<2||q->ny/py<2;
    if(status_all(comm,bad))return -1;
    MPI_Allreduce(&q->shot_count,&min,1,MPI_INT,MPI_MIN,comm);MPI_Allreduce(&q->shot_count,&max,1,MPI_INT,MPI_MAX,comm);
    if(min!=max)return -1;
    cells=(size_t)(q->nx/px)*(q->ny/py);
    if(status_all(comm,cells>(size_t)INT_MAX))return -1;
    out->image_lambda_raw=calloc(cells,sizeof(double));out->image_mu_raw=calloc(cells,sizeof(double));
    sl=malloc(cells*sizeof(double));sm=malloc(cells*sizeof(double));
    if(status_all(comm,!out->image_lambda_raw||!out->image_mu_raw||!sl||!sm))goto failure;
    for(s=0;s<q->shot_count;s++){
        const struct denise_elastic_psv_migration_shot *shot=&q->shots[s];
        bad=shot->physical_shot_index<=previous||shot->source_type!=1;
        MPI_Allreduce(&shot->physical_shot_index,&min,1,MPI_INT,MPI_MIN,comm);MPI_Allreduce(&shot->physical_shot_index,&max,1,MPI_INT,MPI_MAX,comm);
        bad|=min!=max;
        if(status_all(comm,bad))goto failure;
        previous=shot->physical_shot_index;
        memset(&config,0,sizeof(config));config.communicator=comm;config.nprocx=px;config.nprocy=py;
        config.global.nx=q->nx;config.global.ny=q->ny;config.global.nt=q->nt;config.global.fw=q->fw;config.global.dh=q->dh;config.global.dt=q->dt;
        config.global.l=q->l;config.global.invmat1=q->invmat1;config.global.fdorder=q->fdorder;config.global.ndt=q->ndt;config.global.dtinv=q->dtinv;
        config.global.free_surface=q->free_surface;config.global.boundary=q->boundary;config.global.mpi_size=q->mpi_size;config.global.receiver_components=q->receiver_components;
        config.global.lambda=q->lambda;config.global.mu=q->mu;config.global.rho=q->rho;config.global.cpml_enabled=q->cpml_enabled;config.global.pml_reflection=q->pml_reflection;
        config.global.pml_power=q->pml_power;config.global.pml_kmax=q->pml_kmax;config.global.pml_fpml=q->pml_fpml;config.global.pml_damping_speed=q->pml_damping_speed;
        config.global.source_i=shot->source_i;config.global.source_j=shot->source_j;config.global.source_samples=shot->source_samples;
        config.global.receiver_count=shot->receiver_count;config.global.receiver_i=shot->receiver_i;config.global.receiver_j=shot->receiver_j;
        if(denise_elastic_psv_born_mpi_create(&config,&c)||denise_elastic_psv_born_mpi_select_backend(c,q->nt<32?q->nt:32)||denise_elastic_psv_born_mpi_prepare(c,NULL)||denise_elastic_psv_born_mpi_apply_jt(c,shot->migration_data,sl,sm)||denise_elastic_psv_born_mpi_diagnostics(c,&d))goto failure;
        for(p=0;p<cells;p++){out->image_lambda_raw[p]+=sl[p];out->image_mu_raw[p]+=sm[p];}
        *diagnostics=d;
        out->shots_completed=s+1;out->cell_count=cells;out->trajectory_bytes=d.full_bytes;
        out->checkpoint_payload_bytes=d.replay.checkpoint_payload_bytes;out->checkpoint_bytes=d.replay.checkpoint_bytes;out->checkpoint_metadata_bytes=d.replay.checkpoint_metadata_bytes;out->checkpoint_pointer_bytes=d.replay.checkpoint_pointer_bytes;out->segment_schedule_bytes=d.replay.segment_schedule_bytes;
        out->segment_operand_bytes=d.replay.segment_operand_bytes;out->peak_replay_storage_bytes=d.replay.retained_replay_bytes;out->forward_working_bytes=d.replay.forward_working_bytes;out->adjoint_working_bytes=d.replay.adjoint_working_bytes;
        out->initial_forward_steps=d.replay.initial_forward_steps;out->replayed_steps=d.replay.replayed_forward_steps_last;out->segment_count=d.replay.segment_count;out->checkpoint_count=d.replay.checkpoint_count;out->max_segment_length=d.replay.max_segment_length;
        out->maximum_shot_data_bytes=d.local_data_bytes;out->global_image_bytes=2*cells*sizeof(double);
        denise_elastic_psv_born_mpi_destroy(&c);
    }
    free(sl);free(sm);return 0;
failure:
    denise_elastic_psv_born_mpi_destroy(&c);free(sl);free(sm);denise_elastic_psv_migration_result_destroy(out);return -1;
}
