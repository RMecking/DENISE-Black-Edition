/* Actual checked estimator, retained metadata owner, canonical preparation /
 * validation and request failure cleanup. No CUDA driver in the host ASan gate. */
#define DENISE_M9_HOST_SANITIZER_ONLY
#include <initializer_list>
#include "../../src/CUDA/m9_elastic_forward.cu"
#include "denise_cuda_m9_migration.h"
/* Explicit unavailable-backend stubs: only the host failure path is claimed. */
extern "C" int denise_cuda_m9_create_replay(const denise_elastic_psv_born_config *,const denise_cuda_m9_options *,int,int,denise_cuda_m9 **out){*out=NULL;return error("host sanitizer intentionally has no CUDA backend");}
extern "C" int denise_cuda_m9_destroy(denise_cuda_m9 **out){*out=NULL;return 0;}
extern "C" int denise_cuda_m9_migrate(denise_cuda_m9 *,const float *,size_t,double *,double *,size_t){return -1;}
extern "C" int denise_cuda_m9_replay_diagnostics(const denise_cuda_m9 *,struct denise_cuda_m9_replay_diagnostics *){return -1;}
extern "C" int denise_cuda_m9_diagnostics(const denise_cuda_m9 *,struct denise_cuda_m9_diagnostics *){return -1;}
extern "C" int denise_cuda_m9_adjoint_diagnostics(const denise_cuda_m9 *,struct denise_cuda_m9_adjoint_diagnostics *){return -1;}
int main(){
 size_t active[8]={},tested=0;struct denise_cuda_m9_replay_diagnostics d={};
 for(size_t nt: {size_t(1),size_t(17),size_t(41),size_t(67)})for(size_t segments:{size_t(1),size_t(2),size_t(3),size_t(7),size_t(32),size_t(99)}){
  if(denise_cuda_m9_estimate_replay(11,9,nt,segments,active,&d))return 1;
  M9Replay *p=(M9Replay *)m9_host_calloc(d.checkpoint_metadata_bytes-sizeof(HostHeader)+d.segment_schedule_bytes,1);
  if(!p)return 2;p->d=d;p->records=(M9Checkpoint *)(p+1);p->schedule=(int *)(p->records+d.checkpoint_count);
  int last=0;
  for(int s=0;s<d.effective_segments;s++){
   if(denise_cuda_m9_segment_bounds((int)nt,(int)segments,s,p->schedule+2*s,p->schedule+2*s+1))return 3;
   if(p->schedule[2*s]!=last)return 4;last=p->schedule[2*s+1];
   if(s<d.checkpoint_count){p->records[s].values=d.checkpoint_values;p->records[s].time=last-1;}
  }
  if(last!=(int)nt)return 5;m9_host_free(p);++tested;
 }
 const size_t bad[][4]={{SIZE_MAX,9,41,32},{11,SIZE_MAX,41,32},{11,9,SIZE_MAX,32},{SIZE_MAX/8,1,41,32},{11,9,41,0}};
 for(const auto &x:bad){memset(&d,0xa5,sizeof(d));if(!denise_cuda_m9_estimate_replay(x[0],x[1],x[2],x[3],active,&d))return 6;
  const unsigned char *bytes=(const unsigned char *)&d;for(size_t j=0;j<sizeof(d);j++)if(bytes[j])return 7;}
 float l[49],m[49],rho[49],source[7]={1,2,-1,0,1,0,0},data[28]={};int ri[2]={1,5},rj[2]={1,3};
 for(int k=0;k<49;k++){l[k]=5e9;m[k]=3e9;rho[k]=2000;}
 denise_elastic_psv_born_config cfg={};cfg.nx=cfg.ny=7;cfg.nt=7;cfg.dh=10;cfg.dt=.0004;
 cfg.invmat1=3;cfg.fdorder=4;cfg.ndt=cfg.dtinv=cfg.mpi_size=1;cfg.receiver_components=2;
 cfg.lambda=l;cfg.mu=m;cfg.rho=rho;cfg.source_i=cfg.source_j=1;cfg.source_samples=source;
 cfg.receiver_count=2;cfg.receiver_i=ri;cfg.receiver_j=rj;cfg.cpml_enabled=1;cfg.fw=1;
 cfg.pml_reflection=.001;cfg.pml_power=2;cfg.pml_kmax=1.3;cfg.pml_fpml=15;cfg.pml_damping_speed=2500;
 for(int surface=0;surface<2;surface++){
  cfg.free_surface=surface;denise_cuda_m9_fault(0);m9_host *h=NULL;
  if(m9_host_create(&cfg,&h))return 8;size_t sites=calls;m9_host_destroy(&h);
  for(size_t at=1;at<=sites;at++){denise_cuda_m9_fault(at);if(!m9_host_create(&cfg,&h))return 9;m9_host_destroy(&h);if(host_owned)return 10;}
 }
 denise_elastic_psv_migration_shot shot={1,1,1,1,source,2,ri,rj,data};
 denise_elastic_psv_migration_request q={};q.nx=q.ny=7;q.nt=7;q.dh=10;q.dt=.0004;q.invmat1=3;q.fdorder=4;
 q.ndt=q.dtinv=q.mpi_size=1;q.receiver_components=2;q.lambda=l;q.mu=m;q.rho=rho;q.shot_count=1;q.shots=&shot;
 for(size_t at=0;at<=3;at++){
  denise_cuda_m9_replay_fault(at);denise_elastic_psv_migration_result result={};
  if(!denise_cuda_m9_migrate_request(&q,&result) || result.image_lambda_raw || result.image_mu_raw || host_owned)return 11;
 }
 if(device_owned || host_owned || events_owned)return 12;
 printf("M9E4_HOST_REPLAY_PASS schedules=%zu overflow=5 metadata=%zu record=%zu ownership=0,0,0\n",tested,sizeof(M9Replay),sizeof(M9Checkpoint));return 0;
}
