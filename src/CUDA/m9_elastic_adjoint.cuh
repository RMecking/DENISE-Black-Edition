/* M9e-3 exact CPU-contract FP64 reverse. Included only by the isolated CUDA
 * translation unit. No atomics: each row/column/owner has one writer. */
__device__ static size_t rp(View v,int j,int i) {return (size_t)(j+2)*v.w+i+2;}
__global__ static void rproject(View v,FullJT r) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<v.nx)r.field[3*v.padded+rp(v,0,i)]=0;
}
__global__ static void rhalo_y(View v,FullJT r,int first,int count) {
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=(size_t)count*v.w)return;
 int i=t%v.w,k=t/v.w+first;double *f=r.field+k*v.padded;
 for(int h=0;h<4;h++) {
  int dst=h<2?h:v.ny+h,src=h<2?v.ny+h:h;
  f[(size_t)src*v.w+i]+=f[(size_t)dst*v.w+i];f[(size_t)dst*v.w+i]=0;
 }
}
__global__ static void rhalo_x(View v,FullJT r,int first,int count) {
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
 if(t>=(size_t)count*v.ny)return;
 int j=t%v.ny,k=t/v.ny+first;double *f=r.field+k*v.padded+(size_t)(j+2)*v.w;
 for(int h=0;h<4;h++) {
  int dst=h<2?h:v.nx+h,src=h<2?v.nx+h:h;
  f[src]+=f[dst];f[dst]=0;
 }
}
/* FD transpose writes unwrapped ghosts. Halo transpose consumes them in
 * reverse COPY order Y then X. Surface ghosts are consumed before Y. */
__global__ static void rfd(View v,FullJT r,const double *q,int kind,int field) {
 int lane=blockIdx.x*blockDim.x+threadIdx.x;
 bool x=kind<2;int n=x?v.ny:v.nx;if(lane>=n)return;
 double *f=r.field+field*v.padded;
 const double w[4]={9.0/8,-9.0/8,-1.0/24,1.0/24};
 int o=kind%2,offset[4]={o,o-1,o+1,o-2};
 for(int a=0;a<(x?v.nx:v.ny);a++) {
  int i=x?a:lane,j=x?lane:a;double z=q[(size_t)j*v.nx+i];
  for(int k=0;k<4;k++) {
   size_t p=rp(v,j+(x?0:offset[k]),i+(x?offset[k]:0));
   double value=x?(double)v.coefficient*z*w[k]:(double)v.coefficient*w[k]*z;
   f[p]+=value;
  }
 }
}
__global__ static void rghost(View v,FullJT r,int field,const float *bg) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=v.nx)return;
 double *f=r.field+field*v.padded;
 double h=(float)(1.0f/v.coefficient);
 const double w[4]={35.0/16,-35.0/16,21.0/16,-5.0/16};
 Surface s=material(v,i);
 for(int m=1;m<=2;m++) {
  size_t p=rp(v,-m,i);double z=f[p];f[p]=0;
  if(field==0) {
   f[rp(v,m,i)]+=z;
   for(int a=0;a<4;a++)r.q[v.cells+(size_t)a*v.nx+i]+=2.0*m*h*w[a]*z;
  } else {
   double factor=(2.0*m-1)*h*z;f[rp(v,m-1,i)]+=z;r.q[i]+=factor*s.alpha;
   if(bg){r.images[i]+=s.al*bg[i]*factor;r.images[v.cells+i]+=s.am*bg[i]*factor;}
  }
 }
}
__global__ static void rmirror(View v,FullJT r,int field) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=v.w)return;
 double *f=r.field+field*v.padded;
 for(int m=1;m<=2;m++) {
  size_t g=(size_t)(2-m)*v.w+i,p=(size_t)(2+m-(field==4))*v.w+i;
  f[p]-=f[g];f[g]=0;
 }
}
__global__ static void rcpml(View v,FullJT r,int kind,double *q) {
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.cells)return;
 int i=t%v.nx,j=t/v.nx;
 int prof=(kind==0 || kind==5)?1:((kind==1 || kind==7)?2:((kind==2 || kind==4)?0:3));
 int n=prof<2?v.nx:v.ny,k=prof<2?i:j;
 size_t off=prof==0?0:(prof==1?3*(size_t)v.nx:(prof==2?6*(size_t)v.nx:6*(size_t)v.nx+3*(size_t)v.ny));
 const float *p=v.profile+off;double *psi=r.psi+kind*v.cells+t;
 double total=__dadd_rn(q[t],*psi);
 q[t]=__dadd_rn(q[t]/(double)p[k],__dmul_rn((double)p[n+k],total));
 *psi=__dmul_rn((double)p[2*n+k],total);
}
__global__ static void rmaterial(View v,FullJT r,const float *bg) {
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.cells)return;
 size_t p=rp(v,t/v.nx,t%v.nx);
 double sx=r.field[2*v.padded+p],sy=r.field[3*v.padded+p],ss=r.field[4*v.padded+p];
 double l=v.map[p],m=v.map[v.padded+p],xx=bg[t],yy=bg[3*v.cells+t];
 if(v.surface && t<(size_t)v.nx) {
  Surface s=material(v,t);r.images[t]+=s.bl*xx*sx;r.images[v.cells+t]+=s.bm*xx*sx;
  r.q[t]=0; /* A homogeneous closure is the separately gated rclosure stage. */
 } else {
  r.images[t]+=(sx+sy)*(xx+yy);r.images[v.cells+t]+=2.0*(sx*xx+sy*yy);
  r.q[t]=(l+2.0*m)*sx+l*sy;
 }
 r.q[v.cells+t]=(double)v.map[4*v.padded+p]*ss;
 r.q[3*v.cells+t]=ss*((double)bg[v.cells+t]+bg[2*v.cells+t]);
}
__global__ static void rharmonic(View v,FullJT r) {
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.cells)return;
 int i=t%v.nx,j=t/v.nx,im=(i+v.nx-1)%v.nx,jm=(j+v.ny-1)%v.ny;
 size_t a[4]={t,(size_t)j*v.nx+im,(size_t)jm*v.nx+i,(size_t)jm*v.nx+im};
 /* CPU scatters in ascending corner order. Gather in the same order. */
 for(int b=1;b<4;b++){size_t z=a[b];int k=b;while(k && a[k-1]>z){a[k]=a[k-1];k--;}a[k]=z;}
 double mu=v.map[v.padded+rp(v,j,i)];
 for(int k=0;k<4;k++) {
  size_t p=rp(v,a[k]/v.nx,a[k]%v.nx);double h=v.map[4*v.padded+p];
  double common=0.25*h*h*r.q[3*v.cells+a[k]];
  r.images[v.cells+t]+=common/(mu*mu);
 }
}
__global__ static void rclosure(View v,FullJT r) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=v.nx)return;
 r.q[i]+=material(v,i).a*r.field[2*v.padded+rp(v,0,i)];
 r.field[3*v.padded+rp(v,0,i)]=0;
}
__global__ static void rload(View v,FullJT r,int kind,double *q) {
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.cells)return;
 size_t p=rp(v,t/v.nx,t%v.nx);
 double sx=r.field[2*v.padded+p],sy=r.field[3*v.padded+p],ss=r.field[4*v.padded+p];
 double l=v.map[p],m=v.map[v.padded+p];
 if(kind<4)q[t]=(double)v.map[(kind<2?2:3)*v.padded+p]*r.field[(kind<2?0:1)*v.padded+p];
 else if(kind==4)q[t]=(l+2.0*m)*sx+l*sy;
 else if(kind==7)q[t]=(v.surface && t<(size_t)v.nx)?0:l*sx+(l+2.0*m)*sy;
 else q[t]=(double)v.map[4*v.padded+p]*ss;
}
__global__ static void rreceiver(View v,FullJT r,int time) {
 size_t t=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(t>=v.cells)return;
 int i=t%v.nx,j=t/v.nx;size_t p=rp(v,j,i);
 /* One owner, receiver list order, hence deterministic duplicate ADD. */
 for(int k=0;k<v.nr;k++)if(v.geometry[3*k]==i && v.geometry[3*k+1]==j) {
  size_t o=((size_t)time*v.nr+v.geometry[3*k+2])*2;
  r.field[p]+=r.data[o];r.field[v.padded+p]+=r.data[o+1];
 }
}
#define RLAUNCH(name,grid,...) do { if(!gate(#name " reverse launch"))return false; \
 name<<<blocks(grid),128>>>(__VA_ARGS__);if(!launch_check())return false; } while(0)
static bool rhalos(View v,FullJT r,int first,int count) {
 RLAUNCH(rhalo_y,(size_t)count*v.w,v,r,first,count);
 RLAUNCH(rhalo_x,(size_t)count*v.ny,v,r,first,count);return true;
}
static bool rderivative(View v,FullJT r,double *q,int kind,int field,const float *bg) {
 RLAUNCH(rfd,kind<2?v.ny:v.nx,v,r,q,kind,field);
 if(v.surface && kind>=2) {
  if(field<2){RLAUNCH(rghost,v.nx,v,r,field,bg);}
  else if(field>=3){RLAUNCH(rmirror,v.w,v,r,field);}
 }
 return rhalos(v,r,field,1);
}
static bool rchannel(View v,FullJT r,int kind,int fd_kind,int field,const float *bg,bool saved=false) {
 double *q=saved?r.q+(kind==5?v.cells:0):r.q+2*v.cells;
 if(!saved){RLAUNCH(rload,v.cells,v,r,kind,q);}
 RLAUNCH(rcpml,v.cells,v,r,kind,q);
 return rderivative(v,r,q,fd_kind,field,bg);
}
static bool reverse_step(View v,FullJT r,const float *bg,int time) {
 if(v.surface){RLAUNCH(rproject,v.nx,v,r);}
 RLAUNCH(rmaterial,v.cells,v,r,bg);
 if(v.surface){RLAUNCH(rclosure,v.nx,v,r);}
 RLAUNCH(rharmonic,v.cells,v,r);
 if(v.surface) {
  if(!rchannel(v,r,6,3,0,bg) || !rchannel(v,r,7,2,1,bg) ||
     !rchannel(v,r,4,0,0,bg,true) || !rchannel(v,r,5,1,1,bg,true))return false;
 } else if(!rchannel(v,r,4,0,0,NULL) || !rchannel(v,r,5,1,1,NULL) ||
           !rchannel(v,r,6,3,0,NULL) || !rchannel(v,r,7,2,1,NULL))return false;
 RLAUNCH(rreceiver,v.cells,v,r,time);
 if(!rchannel(v,r,0,1,2,NULL) || !rchannel(v,r,1,2,4,NULL) ||
    !rchannel(v,r,2,0,4,NULL) || !rchannel(v,r,3,3,3,NULL))return false;
 if(v.surface){RLAUNCH(rproject,v.nx,v,r);}
 return CUDA(cudaDeviceSynchronize());
}
extern "C" int denise_cuda_m9_apply_jt(denise_cuda_m9 *c,const float *data,size_t values) {
 begin();
 if(!c || !c->jt || !c->d.prepared || !c->valid || !data ||
    values!=(size_t)c->v.nt*c->v.nr*2) {
  error("M9 CUDA JT requires FULL migration prepared context and exact residual count");return finish_failure(c);
 }
 FullJT &r=*c->jt;r.d.valid=0;
 for(size_t k=0;k<values;k++)if(!std::isfinite(data[k])) {
  error("M9 CUDA nonfinite receiver adjoint");return finish_failure(c);
 }
 if(!select(c) || !CUDA(cudaMemset(r.field,0,r.d.field_bytes+r.d.cpml_bytes+r.d.image_bytes+r.d.workspace_bytes)) ||
    !CUDA(cudaMemcpy(r.data,data,r.d.data_bytes,cudaMemcpyHostToDevice)) ||
    !CUDA(cudaEventRecord(c->start)))return finish_failure(c);
 for(int t=c->v.nt-1;t>=0;t--)
  if(!reverse_step(c->v,r,c->v.strain+(size_t)t*4*c->v.cells,t))return finish_failure(c);
 if(!CUDA(cudaEventRecord(c->end)) || !CUDA(cudaEventSynchronize(c->end)) ||
    !CUDA(cudaEventElapsedTime(&r.d.elapsed_ms,c->start,c->end)))return finish_failure(c);
 r.d.valid=1;return 0;
}
extern "C" int denise_cuda_m9_image_download(denise_cuda_m9 *c,double *l,double *m,size_t cells) {
 begin();
 if(!c || !c->jt || !c->jt->d.valid || !l || !m || cells!=c->v.cells) {
  error("M9 CUDA raw image unavailable or invalid buffers/count");return finish_failure(c);
 }
 double *tmp=(double *)m9_host_calloc(c->jt->d.image_bytes,1);
 bool ok=tmp && select(c) && copy_out(tmp,c->jt->images,c->jt->d.image_bytes);
 if(ok){memcpy(l,tmp,8*cells);memcpy(m,tmp+cells,8*cells);}
 m9_host_free(tmp);return ok?0:finish_failure(c);
}
extern "C" int denise_cuda_m9_migrate(denise_cuda_m9 *c,const float *data,size_t values,double *l,double *m,size_t cells) {
 begin();
 if(!c || !c->jt || !data || !l || !m || cells!=c->v.cells ||
    values!=(size_t)c->v.nt*c->v.nr*2) {
  error("M9 CUDA migration input/output contract invalid");return finish_failure(c);
 }
 if(!gate("FULL migration wrapper") || denise_cuda_m9_prepare(c) ||
    denise_cuda_m9_apply_jt(c,data,values))return finish_failure(c);
 return denise_cuda_m9_image_download(c,l,m,cells);
}
extern "C" int denise_cuda_m9_adjoint_diagnostics(const denise_cuda_m9 *c,struct denise_cuda_m9_adjoint_diagnostics *d) {
 if(!c || !c->jt || !d)return error("M9 CUDA JT diagnostics require migration context");
 *d=c->jt->d;return 0;
}
/* Actual production blocks exposed for independent dense/basis gates. */
extern "C" int denise_cuda_m9_test_reverse(denise_cuda_m9 *c,int mode,int kind,int field,
 double *fields,double *psi,double *q,double *images,const float *bg,const float *data) {
 begin();
 if(!c || !c->jt || !fields || !psi || !q || !images || !bg || !data ||
    mode<0 || mode>10 || kind<0 || kind>7 || field<0 || field>4 ||
    (mode==2 && kind>3) || (mode==3 && (!c->v.surface || field>1)) ||
    (mode==4 && field<3))return finish_failure(c);
 invalidate(c);FullJT r=*c->jt;View v=c->v;
 size_t sizes[4]={r.d.field_bytes,r.d.cpml_bytes,r.d.workspace_bytes,r.d.image_bytes};
 void *src[4]={fields,psi,q,images},*dst[4]={r.field,r.psi,r.q,r.images};
 size_t total=sizes[0]+sizes[1]+sizes[2]+sizes[3];char *tmp=(char *)m9_host_calloc(total,1);
 bool ok=tmp && select(c);
 for(int k=0;k<4 && ok;k++)ok=CUDA(cudaMemcpy(dst[k],src[k],sizes[k],cudaMemcpyHostToDevice));
 if(ok)ok=CUDA(cudaMemcpy(v.strain,bg,16*v.cells,cudaMemcpyHostToDevice)) &&
          CUDA(cudaMemcpy(r.data,data,r.d.data_bytes,cudaMemcpyHostToDevice));
 /* RLAUNCH returns bool from a lambda, so failure remains transactional. */
 if(ok)ok=[&]() {
  if(mode==0)return rhalos(v,r,field,1);
  if(mode==1){RLAUNCH(rcpml,v.cells,v,r,kind,r.q);}
  if(mode==2)return rderivative(v,r,r.q,kind,field,v.strain);
  if(mode==3){RLAUNCH(rghost,v.nx,v,r,field,v.strain);}
  if(mode==4){RLAUNCH(rmirror,v.w,v,r,field);}
  if(mode==5){RLAUNCH(rproject,v.nx,v,r);}
  if(mode==6){
   RLAUNCH(rmaterial,v.cells,v,r,v.strain);
   if(v.surface){RLAUNCH(rclosure,v.nx,v,r);}
   RLAUNCH(rharmonic,v.cells,v,r);
  }
  if(mode==7){RLAUNCH(rreceiver,v.cells,v,r,0);}
  if(mode==8){RLAUNCH(rclosure,v.nx,v,r);}
  if(mode==9){RLAUNCH(rharmonic,v.cells,v,r);}
  if(mode==10)return reverse_step(v,r,v.strain,0);
  return true;
 }();
 if(ok)ok=CUDA(cudaDeviceSynchronize());size_t off=0;
 for(int k=0;k<4 && ok;k++){ok=copy_out(tmp+off,dst[k],sizes[k]);off+=sizes[k];}
 if(ok){off=0;for(int k=0;k<4;k++){memcpy(src[k],tmp+off,sizes[k]);off+=sizes[k];}}
 m9_host_free(tmp);return ok?0:finish_failure(c);
}
#undef RLAUNCH
