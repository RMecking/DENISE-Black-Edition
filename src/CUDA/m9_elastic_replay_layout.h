/* Included after checked arithmetic; usable without a CUDA driver. */
struct M9Checkpoint {
 int version,time,nx,ny,fw,surface,cpml;
 size_t values;
};
struct M9Replay {
 struct denise_cuda_m9_replay_diagnostics d;
 size_t active[8],offset[8];
 M9Checkpoint *records;
 int *schedule;
 float *payload;
};
extern "C" int denise_cuda_m9_segment_bounds(int nt,int segments,int segment,int *start,int *end) {
 begin();if(start)*start=0;if(end)*end=0;
 if(nt<1 || segments<1 || !start || !end)return error("M9 replay schedule invalid");
 int effective=segments<nt?segments:nt;
 if(segment<0 || segment>=effective)return error("M9 replay segment out of range");
 *start=(int)((uint64_t)segment*nt/effective);
 *end=(int)((uint64_t)(segment+1)*nt/effective);return 0;
}
extern "C" int denise_cuda_m9_estimate_replay(size_t nx,size_t ny,size_t nt,size_t segments,
 const size_t active[8],struct denise_cuda_m9_replay_diagnostics *out) {
 begin();if(!out)return error("M9 replay estimate output null");memset(out,0,sizeof(*out));
 struct denise_cuda_m9_replay_diagnostics d={};size_t n=0,v=0,term=0,metadata=0;
 if(!nx || !ny || !nt || !segments || nt>INT_MAX || segments>INT_MAX || !active ||
    !mul(nx,ny,n) || !mul(n,5,v))return error("M9 replay estimate dimensions invalid/overflow");
 size_t e=segments<nt?segments:nt,k=e-1,m=nt/e+(nt%e!=0);
 for(int c=0;c<8;c++) {
  bool x=c==0 || c==2 || c==4 || c==5;
  if(active[c]>(x?nx:ny) || !mul(active[c],x?ny:nx,term) || !add(v,term,v))
   return error("M9 replay active payload overflow");
 }
 d.checkpoint_values=v;
 if(!mul(v,4,term) || !mul(term,k,d.checkpoint_payload_bytes) ||
    !mul(k,sizeof(M9Checkpoint),metadata) || !add(metadata,sizeof(M9Replay),metadata) ||
    !add(metadata,sizeof(HostHeader),d.checkpoint_metadata_bytes) ||
    !mul(e,2*sizeof(int),d.segment_schedule_bytes) || !mul(n,16,term) ||
    !mul(term,m,d.segment_operand_bytes) || !mul(term,nt,d.full_operand_bytes))
  return error("M9 replay storage overflow");
 /* Payload and FULL tape are multiples of four. Pad payload to eight so the
  * common reverse-arena alignment remains identical in both backends. */
 d.alignment_bytes=(8-d.checkpoint_payload_bytes%8)%8;
 const size_t parts[]={d.checkpoint_payload_bytes,d.checkpoint_metadata_bytes,
  d.segment_schedule_bytes,d.segment_operand_bytes,d.alignment_bytes};
 for(size_t i=0;i<5;i++)if(!add(d.retained_bytes,parts[i],d.retained_bytes))return error("M9 replay retained overflow");
 d.requested_segments=(int)segments;d.effective_segments=(int)e;
 d.checkpoint_count=(int)k;d.max_segment_length=(int)m;
 d.selected_replay=d.retained_bytes<d.full_operand_bytes;*out=d;return 0;
}
