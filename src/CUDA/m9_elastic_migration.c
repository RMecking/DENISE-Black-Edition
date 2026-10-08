#include "denise_cuda_m9_migration.h"
#include "denise_cuda_m9_elastic.h"
#include "m9_elastic_host.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char request_error[768];
const char *denise_cuda_m9_migration_last_error(void){return request_error[0]?request_error:"no M9 CUDA migration error";}
void denise_cuda_m9_migration_result_destroy(struct denise_elastic_psv_migration_result *r){
 if(r){free(r->image_lambda_raw);free(r->image_mu_raw);memset(r,0,sizeof(*r));}
}
int denise_cuda_m9_migrate_request(const struct denise_elastic_psv_migration_request *q,
 struct denise_elastic_psv_migration_result *out) {
 struct denise_cuda_m9 *c=NULL;struct denise_elastic_psv_migration_result r;
 struct denise_cuda_m9_options options={0,0,0};
 double *work=NULL,*sum=NULL,*published=NULL;
 size_t cells=0,trajectory=0,images=0,data=0,k;int shot;
 request_error[0]=0;m9_host_begin();if(!out){snprintf(request_error,sizeof(request_error),"M9 CUDA result null");return -1;}
 memset(out,0,sizeof(*out));memset(&r,0,sizeof(r));
 if(m9_host_validate_migration(q,&cells,&trajectory,&images,&data)){
  snprintf(request_error,sizeof(request_error),"M9 CUDA request: %s",m9_host_migration_error());return -1;
 }
 sum=m9_host_calloc(images,1);work=m9_host_calloc(images,1);
 if(!sum || !work){snprintf(request_error,sizeof(request_error),"M9 CUDA shot %d: %s",q->shots[0].physical_shot_index,denise_cuda_m9_last_error());goto failure;}
 for(shot=0;shot<q->shot_count;shot++) {
  const struct denise_elastic_psv_migration_shot *s=q->shots+shot;
  struct denise_elastic_psv_born_config cfg;
  struct denise_cuda_m9_replay_diagnostics d;
  struct denise_cuda_m9_diagnostics f;
  struct denise_cuda_m9_adjoint_diagnostics a;
  memset(&cfg,0,sizeof(cfg));
  if(m9_host_gate("migration shot begin")){snprintf(request_error,sizeof(request_error),"M9 CUDA shot %d: %s",s->physical_shot_index,denise_cuda_m9_last_error());goto failure;}
  cfg.nx=q->nx;cfg.ny=q->ny;cfg.nt=q->nt;cfg.fw=q->fw;cfg.dh=q->dh;cfg.dt=q->dt;
  cfg.l=q->l;cfg.invmat1=q->invmat1;cfg.fdorder=q->fdorder;cfg.ndt=q->ndt;cfg.dtinv=q->dtinv;
  cfg.free_surface=q->free_surface;cfg.boundary=q->boundary;cfg.mpi_size=q->mpi_size;cfg.receiver_components=q->receiver_components;
  cfg.lambda=q->lambda;cfg.mu=q->mu;cfg.rho=q->rho;
  cfg.source_i=s->source_i;cfg.source_j=s->source_j;cfg.source_samples=s->source_samples;
  cfg.receiver_count=s->receiver_count;cfg.receiver_i=s->receiver_i;cfg.receiver_j=s->receiver_j;
  cfg.cpml_enabled=q->cpml_enabled;cfg.pml_reflection=q->pml_reflection;cfg.pml_power=q->pml_power;
  cfg.pml_kmax=q->pml_kmax;cfg.pml_fpml=q->pml_fpml;cfg.pml_damping_speed=q->pml_damping_speed;
  if(denise_cuda_m9_create_replay(&cfg,&options,q->nt<32?q->nt:32,1,&c) ||
     denise_cuda_m9_migrate(c,s->migration_data,(size_t)q->nt*s->receiver_count*2,work,work+cells,cells)) {
   snprintf(request_error,sizeof(request_error),"M9 CUDA shot %d: %s",s->physical_shot_index,denise_cuda_m9_last_error());goto failure;
  }
  denise_cuda_m9_replay_diagnostics(c,&d);denise_cuda_m9_diagnostics(c,&f);denise_cuda_m9_adjoint_diagnostics(c,&a);
  r.checkpoint_payload_bytes=d.selected_replay?d.checkpoint_values*4:0;
  r.checkpoint_bytes=d.selected_replay?d.checkpoint_payload_bytes:0;
  r.checkpoint_metadata_bytes=d.selected_replay?d.checkpoint_metadata_bytes:0;
  r.checkpoint_pointer_bytes=d.selected_replay?d.checkpoint_pointer_bytes:0;
  r.segment_schedule_bytes=d.selected_replay?d.segment_schedule_bytes:0;
  r.segment_operand_bytes=d.selected_replay?d.segment_operand_bytes:0;
  r.peak_replay_storage_bytes=d.selected_replay?d.retained_bytes:0;
  r.segment_count=d.selected_replay?d.effective_segments:0;
  r.checkpoint_count=d.selected_replay?d.checkpoint_count:0;
  r.max_segment_length=d.selected_replay?d.max_segment_length:0;
  r.initial_forward_steps=q->nt;r.replayed_steps=d.replayed_forward_steps;
  r.forward_working_bytes=f.wavefield_bytes+f.cpml_bytes;
  r.adjoint_working_bytes=a.field_bytes+a.cpml_bytes+a.workspace_bytes;
  for(k=0;k<cells*2;k++)sum[k]+=work[k];
  if(denise_cuda_m9_destroy(&c)) {snprintf(request_error,sizeof(request_error),"M9 CUDA shot %d cleanup: %s",s->physical_shot_index,denise_cuda_m9_last_error());goto failure;}
 }
 /* Publish only after all shots and cleanup succeed. Standard ownership is
  * deliberate: the shared canonical result destructor uses free(). */
 published=m9_host_publish(cells,sizeof(double));
 if(published)r.image_mu_raw=m9_host_publish(cells,sizeof(double));
 if(!published || !r.image_mu_raw){snprintf(request_error,sizeof(request_error),"M9 CUDA output: %s",denise_cuda_m9_last_error());goto failure;}
 memcpy(published,sum,cells*sizeof(double));memcpy(r.image_mu_raw,sum+cells,cells*sizeof(double));
 r.image_lambda_raw=published;r.cell_count=cells;r.shots_completed=q->shot_count;
 r.trajectory_bytes=trajectory;r.global_image_bytes=images;r.maximum_shot_data_bytes=data;
 m9_host_free(sum);m9_host_free(work);*out=r;return 0;
failure:
 denise_cuda_m9_destroy(&c);m9_host_free(sum);m9_host_free(work);free(published);free(r.image_mu_raw);return -1;
}
