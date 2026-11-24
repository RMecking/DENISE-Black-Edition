/* R5 timing-only private link. Original arguments/operations are untouched. */
#define denise_cuda_psv_dispatch r5_dispatch_impl
#include "../src/PSV/psv_cuda_dispatch.c"
#undef denise_cuda_psv_dispatch
static void r5_time(double t,int result,int mode) {
 const char *p=getenv("DENISE_PSV_R5_TIMING_FILE");FILE *f;
 if(!p || !(f=fopen(p,"a"))) MPI_Abort(MPI_COMM_WORLD,93);
 fprintf(f,"dispatch,%.17g,%d,%d\n",t,result,mode);fclose(f);
}
int denise_cuda_psv_dispatch(struct wavePSV *w,struct wavePSV_PML *p,struct matPSV *m,struct seisPSV *s,struct acq *a,float *hc,int ns,int nr,int mode,enum denise_psv_backend b) {
 double t=MPI_Wtime();int r=r5_dispatch_impl(w,p,m,s,a,hc,ns,nr,mode,b);r5_time(MPI_Wtime()-t,r,mode);return r;
}
