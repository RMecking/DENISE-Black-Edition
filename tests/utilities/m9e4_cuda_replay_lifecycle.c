/* Real compact replay owner for CUDA host UBSan / Compute-Sanitizer. */
#include "denise_cuda_m9_elastic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int check(int rc){if(rc)fprintf(stderr,"%s\n",denise_cuda_m9_last_error());return rc;}
int main(void){
 float l[63],m[63],rho[63],samples[7]={1.25f,-3.5f,.875f,0,1,0,-1},residual[42];
 double gl[63],gm[63],first[126];int ri[3]={8,0,8},rj[3]={0,1,0};
 struct denise_elastic_psv_born_config q;struct denise_cuda_m9_options opt={0,0,0};
 struct denise_cuda_m9 *full=NULL,*c=NULL;struct denise_cuda_m9_replay_diagnostics d;
 size_t db,hb,ev,calls;int surface,k;
 memset(&q,0,sizeof(q));
 for(k=0;k<63;k++){l[k]=5e9f+k*1e6f;m[k]=3e9f+k*3e5f;rho[k]=2000+k;}
 for(k=0;k<42;k++)residual[k]=(k%7-3)*.125f;
 q.nx=9;q.ny=7;q.nt=7;q.fw=1;q.dh=10;q.dt=.0005f;q.invmat1=3;q.fdorder=4;
 q.ndt=q.dtinv=q.mpi_size=1;q.receiver_components=2;q.lambda=l;q.mu=m;q.rho=rho;
 q.source_i=1;q.source_j=1;q.source_samples=samples;q.receiver_count=3;q.receiver_i=ri;q.receiver_j=rj;
 q.cpml_enabled=1;q.pml_reflection=.001f;q.pml_power=2;q.pml_kmax=1.3f;q.pml_fpml=15;q.pml_damping_speed=2500;
 for(surface=0;surface<2;surface++){
  q.free_surface=surface;denise_cuda_m9_fault(0);
  if(check(denise_cuda_m9_create_migration(&q,&opt,&full)) || check(denise_cuda_m9_create_replay(&q,&opt,3,0,&c)))return 1;
  if(check(denise_cuda_m9_migrate(full,residual,42,gl,gm,63)))return 2;
  memcpy(first,gl,sizeof(gl));memcpy(first+63,gm,sizeof(gm));
  if(check(denise_cuda_m9_migrate(c,residual,42,gl,gm,63)))return 3;
  if(memcmp(first,gl,sizeof(gl)) || memcmp(first+63,gm,sizeof(gm)))return 4;
  if(check(denise_cuda_m9_replay_diagnostics(c,&d)))return 5;
  float *before=malloc(d.checkpoint_payload_bytes),*after=malloc(d.checkpoint_payload_bytes);
  if(!before || !after)return 6;
  if(check(denise_cuda_m9_test_checkpoints(c,before,d.checkpoint_payload_bytes/4)))return 7;
  for(k=0;k<3;k++){
   float state[13*63],operand[4*63];int start,end;
   if(check(denise_cuda_m9_segment_bounds(7,3,k,&start,&end)) ||
      check(denise_cuda_m9_test_replay_probe(c,k,end-1,state,operand)))return 8;
  }
  if(check(denise_cuda_m9_apply_jt(c,residual,42)) || check(denise_cuda_m9_image_download(c,gl,gm,63)))return 9;
  if(memcmp(first,gl,sizeof(gl)) || memcmp(first+63,gm,sizeof(gm)))return 10;
  if(check(denise_cuda_m9_test_checkpoints(c,after,d.checkpoint_payload_bytes/4)) || memcmp(before,after,d.checkpoint_payload_bytes))return 11;
  free(before);free(after);
  if(check(denise_cuda_m9_destroy(&c)) || check(denise_cuda_m9_destroy(&full)))return 12;
  denise_cuda_m9_ledger(&db,&hb,&ev,&calls);if(db || hb || ev)return 13;
  printf("M9E4_CUDA_REPLAY_LIFECYCLE_PASS surface=%d steps=7 ownership=0,0,0\n",surface);
 }
 return 0;
}
