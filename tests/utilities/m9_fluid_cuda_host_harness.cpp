/* Narrow FLUID-4B host ASan/UBSan gate: copied classification, canonical maps,
 * allocation failure cleanup and pre-shot migration request rejection. */
#define DENISE_M9_HOST_SANITIZER_ONLY
#include "../../src/CUDA/m9_elastic_forward.cu"
#include "denise_cuda_m9_migration.h"
#include <limits>

/* This host gate never substitutes for the separate real CUDA physics tests. */
extern "C" int denise_cuda_m9_create_replay(const denise_elastic_psv_born_config *,const denise_cuda_m9_options *,int,int,denise_cuda_m9 **out){*out=NULL;return error("unexpected replay in host fluid gate");}
extern "C" int denise_cuda_m9_destroy(denise_cuda_m9 **out){*out=NULL;return 0;}
extern "C" int denise_cuda_m9_migrate(denise_cuda_m9 *,const float *,size_t,double *,double *,size_t){return -1;}
extern "C" int denise_cuda_m9_replay_diagnostics(const denise_cuda_m9 *,struct denise_cuda_m9_replay_diagnostics *){return -1;}
extern "C" int denise_cuda_m9_diagnostics(const denise_cuda_m9 *,struct denise_cuda_m9_diagnostics *){return -1;}
extern "C" int denise_cuda_m9_adjoint_diagnostics(const denise_cuda_m9 *,struct denise_cuda_m9_adjoint_diagnostics *){return -1;}

int main(){
 float l[49],m[49],rho[49],source[7]={1,0,0,0,0,0,0},data[28]={};
 int ri[2]={1,5},rj[2]={1,3};
 denise_elastic_psv_born_config cfg={};
 cfg.nx=cfg.ny=7;cfg.nt=7;cfg.dh=1;cfg.dt=.1f;
 cfg.invmat1=3;cfg.fdorder=4;cfg.ndt=cfg.dtinv=cfg.mpi_size=1;cfg.receiver_components=2;
 cfg.lambda=l;cfg.mu=m;cfg.rho=rho;cfg.source_i=cfg.source_j=1;cfg.source_samples=source;
 cfg.receiver_count=2;cfg.receiver_i=ri;cfg.receiver_j=rj;
 size_t fault_sites=0;
 for(int fs=0;fs<2;fs++)for(int kind=0;kind<3;kind++){
  for(int k=0;k<49;k++){l[k]=4;rho[k]=1;m[k]=3;}
  m[48]=kind==0?0.f:kind==1?-0.f:std::numeric_limits<float>::denorm_min();
  cfg.free_surface=fs;denise_cuda_m9_fault(0);m9_host *h=NULL;
  if(m9_host_create(&cfg,&h))return 1;
  if(m9_host_has_fluid(h)!=(kind<2))return 2;
  size_t sites=calls;
  m[48]=kind<2?3.f:0.f;
  if(m9_host_has_fluid(h)!=(kind<2))return 3;
  float maps[5*11*11]={};m9_host_maps(h,maps);
  for(float value:maps)if(!std::isfinite(value))return 4;
  m9_host_destroy(&h);if(h || host_owned)return 5;
  m[48]=kind==0?0.f:kind==1?-0.f:std::numeric_limits<float>::denorm_min();
  for(size_t at=1;at<=sites;at++){
   denise_cuda_m9_fault(at);
   if(!m9_host_create(&cfg,&h))return 6;
   m9_host_destroy(&h);
   if(h || host_owned || !strstr(denise_cuda_m9_last_error(),"injected"))return 7;
   ++fault_sites;
  }
 }
 denise_elastic_psv_migration_shot shot={1,1,1,1,source,2,ri,rj,data};
 denise_elastic_psv_migration_request q={};q.nx=q.ny=7;q.nt=7;q.dh=1;q.dt=.1f;
 q.invmat1=3;q.fdorder=4;q.ndt=q.dtinv=q.mpi_size=1;q.receiver_components=2;
 q.lambda=l;q.mu=m;q.rho=rho;q.shot_count=1;q.shots=&shot;
 for(int sign=0;sign<2;sign++){
  m[48]=sign?-0.f:0.f;denise_cuda_m9_fault(0);
  denise_elastic_psv_migration_result result;memset(&result,0x7f,sizeof(result));
  if(!denise_cuda_m9_migrate_request(&q,&result) || !strstr(denise_cuda_m9_migration_last_error(),"FLUID-4E"))return 8;
  const unsigned char *bytes=(const unsigned char *)&result;
  for(size_t k=0;k<sizeof(result);k++)if(bytes[k])return 9;
  if(calls || device_owned || host_owned || events_owned)return 10;
 }
 printf("FLUID4B_HOST_PASS signed_zero=2 subnormal=solid copied_ownership=6 allocation_fault_sites=%zu request_empty=2 ledger=0,0,0\n",fault_sites);
 return 0;
}
