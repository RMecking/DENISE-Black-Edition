/* Host scheduling and compact FP32 state copies only. Numerical kernels are
 * the canonical M9e-3 extended_step and reverse_step, unchanged. */
extern "C" int denise_cuda_m9_create_replay(const denise_elastic_psv_born_config *q,
 const denise_cuda_m9_options *o,int segments,int automatic,denise_cuda_m9 **out) {
 begin();if(!out)return error("M9 replay context output null");*out=NULL;
 if(!q || segments<1 || (automatic!=0 && automatic!=1))return error("M9 replay configuration invalid");
 m9_host *h=NULL;M9Replay *p=NULL;size_t active[8];struct denise_cuda_m9_replay_diagnostics d={};
 if(m9_host_create(q,&h))return error("M9 replay canonical preparation: %s",m9_host_error());
 m9_host_active_counts(h,active);
 int rc=denise_cuda_m9_estimate_replay(q->nx,q->ny,q->nt,segments,active,&d);
 if(rc)goto done;
 if(!automatic || d.selected_replay) {
  p=(M9Replay *)m9_host_calloc(d.checkpoint_metadata_bytes-sizeof(HostHeader)+d.segment_schedule_bytes,1);
  if(!p){rc=-1;goto done;}
  d.selected_replay=1;p->d=d;
  p->records=(M9Checkpoint *)(p+1);
  p->schedule=(int *)(p->records+d.checkpoint_count);
  size_t offset=5*(size_t)q->nx*q->ny;
  for(int k=0;k<8;k++) {
   p->active[k]=active[k];p->offset[k]=offset;
   offset+=active[k]*((k==0 || k==2 || k==4 || k==5)?q->ny:q->nx);
  }
  for(int s=0;s<d.effective_segments;s++) {
   if(!gate("replay segment schedule") || denise_cuda_m9_segment_bounds(q->nt,segments,s,p->schedule+2*s,p->schedule+2*s+1)){rc=-1;goto done;}
   if(s<d.checkpoint_count) {
    M9Checkpoint &r=p->records[s];r.version=1;r.time=p->schedule[2*s+1]-1;
    r.nx=q->nx;r.ny=q->ny;r.fw=q->fw;r.surface=q->free_surface;r.cpml=q->cpml_enabled;r.values=d.checkpoint_values;
   }
  }
 }
 rc=create(q,o,out,true,true,&p,&h);
 if(!rc){(*out)->replay_diag=d;(*out)->replay_diag.owned_device_bytes=(*out)->d.owned_bytes;}
done:
 m9_host_free(p);m9_host_destroy(&h);return rc;
}
extern "C" int denise_cuda_m9_replay_diagnostics(const denise_cuda_m9 *c,struct denise_cuda_m9_replay_diagnostics *out) {
 if(!c || !out)return error("M9 replay diagnostics null");*out=c->replay_diag;
 out->owned_device_bytes=c->d.owned_bytes;
 if(!out->full_operand_bytes)out->full_operand_bytes=c->d.trajectory_bytes;
 return 0;
}
__global__ static void checkpoint_fields(View v,float *payload,bool restore) {
 size_t k=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(k>=5*v.cells)return;
 size_t t=k%v.cells,p=(k/v.cells)*v.padded+(t/v.nx+2)*v.w+t%v.nx+2;
 if(restore)v.field[p]=payload[k];else payload[k]=v.field[p];
}
/* One thread per orthogonal row/column, scan actual a != 0 coordinates.
 * X channels: row then active i; Y channels: column then active j, CPU parity. */
__global__ static void checkpoint_psi(View v,int kind,size_t active,size_t offset,float *payload,bool restore,int *invalid) {
 bool x=kind==0 || kind==2 || kind==4 || kind==5;
 int orth=blockIdx.x*blockDim.x+threadIdx.x;if(orth>=(x?v.ny:v.nx))return;
 int prof=(kind==0 || kind==5)?1:((kind==1 || kind==7)?2:((kind==2 || kind==4)?0:3));
 int n=x?v.nx:v.ny;
 size_t po=prof==0?0:(prof==1?3*(size_t)v.nx:(prof==2?6*(size_t)v.nx:6*(size_t)v.nx+3*(size_t)v.ny));
 const float *a=v.profile+po+n;size_t index=offset+(size_t)orth*active;
 for(int coord=0;coord<n;coord++) {
  int i=x?coord:orth,j=x?orth:coord;
  float *memory=v.psi+(size_t)kind*v.padded+(size_t)(j+2)*v.w+i+2;
  if(a[coord]!=0.0f) {if(restore)*memory=payload[index];else payload[index]=*memory;++index;}
  else if(!restore && *memory!=0.0f)atomicExch(invalid,1);
 }
}
static bool checkpoint_copy(denise_cuda_m9 *c,int checkpoint,bool restore) {
 M9Replay &p=*c->replay;View v=c->v;
 if(checkpoint<0 || checkpoint>=p.d.checkpoint_count)return error("M9 checkpoint index invalid")==0;
 const M9Checkpoint &r=p.records[checkpoint];const auto *cfg=m9_host_config(c->host);
 if(r.version!=1 || r.time!=p.schedule[2*checkpoint+1]-1 || r.values!=p.d.checkpoint_values ||
    r.nx!=v.nx || r.ny!=v.ny || r.fw!=cfg->fw || r.surface!=v.surface || r.cpml!=cfg->cpml_enabled)
  return error("M9 checkpoint metadata invalid")==0;
 float *payload=p.payload+(size_t)checkpoint*p.d.checkpoint_values;
 int *flag=(int *)c->full->corner;int invalid=0;
 if(restore && !CUDA(cudaMemset(v.field,0,c->d.wavefield_bytes+c->d.cpml_bytes)))return false;
 if(!CUDA(cudaMemset(flag,0,sizeof(int))) || !gate(restore?"checkpoint restore fields":"checkpoint capture fields"))return false;
 checkpoint_fields<<<blocks(5*v.cells),128>>>(v,payload,restore);if(!launch_check())return false;
 for(int k=0;k<8;k++) {
  if(!gate(restore?"checkpoint restore CPML":"checkpoint capture CPML"))return false;
  checkpoint_psi<<<blocks((k==0 || k==2 || k==4 || k==5)?v.ny:v.nx),128>>>(v,k,p.active[k],p.offset[k],payload,restore,flag);
  if(!launch_check())return false;
 }
 if(!CUDA(cudaDeviceSynchronize()) || !CUDA(cudaMemcpy(&invalid,flag,sizeof(int),cudaMemcpyDeviceToHost)))return false;
 if(invalid)return error("M9 compact checkpoint omitted nonzero inactive CPML state")==0;
 return true;
}
static bool replay_prepare(denise_cuda_m9 *c) {
 M9Replay &p=*c->replay;c->replay_diag.initial_forward_steps=0;c->replay_diag.replayed_forward_steps=0;
 for(int s=0;s<p.d.effective_segments;s++) {
  int start=p.schedule[2*s],end=p.schedule[2*s+1];
  for(int t=start;t<end;t++) {
   if(!gate("replay initial timestep") || !extended_step(c->v,t-start,t,NULL,NULL))return false;
   ++c->replay_diag.initial_forward_steps;
  }
  if(s<p.d.checkpoint_count && !checkpoint_copy(c,s,false))return false;
 }
 return true;
}
static bool replay_restore(denise_cuda_m9 *c,int segment) {
 if(segment)return checkpoint_copy(c,segment-1,true);
 return CUDA(cudaMemset(c->v.field,0,c->d.wavefield_bytes+c->d.cpml_bytes));
}
static bool replay_segment(denise_cuda_m9 *c,int s) {
 if(!replay_restore(c,s))return false;M9Replay &p=*c->replay;
 int start=p.schedule[2*s],end=p.schedule[2*s+1];
 for(int t=start;t<end;t++) {
  if(!gate("replay regenerated timestep") || !extended_step(c->v,t-start,t,NULL,NULL))return false;
  ++c->replay_diag.replayed_forward_steps;
 }
 return true;
}
extern "C" int denise_cuda_m9_test_checkpoints(denise_cuda_m9 *c,float *out,size_t values) {
 begin();if(!c || !c->replay || !c->d.prepared || !out ||
  values!=c->replay->d.checkpoint_payload_bytes/4)return error("M9 checkpoint output contract invalid");
 float *tmp=(float *)m9_host_calloc(values,4);
 bool ok=tmp && select(c) && CUDA(cudaMemcpy(tmp,c->replay->payload,values*4,cudaMemcpyDeviceToHost));
 if(ok)memcpy(out,tmp,values*4);m9_host_free(tmp);return ok?0:finish_failure(c);
}
extern "C" int denise_cuda_m9_test_replay_probe(denise_cuda_m9 *c,int segment,int time,float *state,float *q) {
 begin();if(!c || !c->full || !c->d.prepared || !c->valid || !state || time< -1 || time>=c->v.nt)
  return error("M9 replay probe invalid");
 View v=c->full->v;int start=0,end=c->v.nt;bool ok=select(c);
 if(c->replay) {
  if(segment<0 || segment>=c->replay->d.effective_segments)return error("M9 probe segment invalid");
  start=c->replay->schedule[2*segment];end=c->replay->schedule[2*segment+1];v=c->v;
  if(time!=-1 && (time<start || time>=end))return error("M9 probe time outside segment");
  if(ok)ok=replay_restore(c,segment);
 } else {
  if(ok)ok=CUDA(cudaMemset(v.field,0,c->d.wavefield_bytes+c->d.cpml_bytes));
  c->full->d.valid=0;
 }
 if(time==-1 && q)return error("M9 restored-state probe has no operands");
 for(int t=start;t<=time && ok;t++)ok=extended_step(v,c->replay?t-start:0,t,NULL,NULL);
 size_t bytes=52*v.cells+(q?16*v.cells:0);
 float *tmp=ok?(float *)m9_host_calloc(bytes,1):NULL;ok=ok && tmp;
 for(int f=0;f<13 && ok;f++)for(int j=0;j<v.ny && ok;j++)
  ok=CUDA(cudaMemcpy(tmp+(size_t)f*v.cells+(size_t)j*v.nx,
   (f<5?v.field:v.psi)+(size_t)(f<5?f:f-5)*v.padded+(size_t)(j+2)*v.w+2,(size_t)v.nx*4,cudaMemcpyDeviceToHost));
 if(ok && q)ok=CUDA(cudaMemcpy(tmp+13*v.cells,v.strain+(c->replay?(size_t)(time-start)*4*v.cells:0),16*v.cells,cudaMemcpyDeviceToHost));
 if(ok){memcpy(state,tmp,52*v.cells);if(q)memcpy(q,tmp+13*v.cells,16*v.cells);}
 m9_host_free(tmp);return ok?0:finish_failure(c);
}
