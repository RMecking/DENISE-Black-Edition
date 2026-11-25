/* R5 timing-only private link. No residual/model/operator change. */
#define denise_cuda_psv_exact_fwi_finish r5_finish_impl
#include "../src/PSV/psv_cuda_exact_fwi.c"
#undef denise_cuda_psv_exact_fwi_finish
int denise_cuda_psv_exact_fwi_finish(const struct visco_psv_exact_fwi_request *r) {
 const char *p=getenv("DENISE_PSV_R5_TIMING_FILE");FILE *f;double t=MPI_Wtime();int v=r5_finish_impl(r);t=MPI_Wtime()-t;
 if(!p || !(f=fopen(p,"a"))) MPI_Abort(MPI_COMM_WORLD,93);
 fprintf(f,"finish,%.17g,%d,0\n",t,v);fclose(f);return v;
}
