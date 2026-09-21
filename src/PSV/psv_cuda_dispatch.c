#include "fd.h"
#include "denise_cuda_psv_dispatch.h"
#include "denise_cuda_backend.h"

#include <stdarg.h>

static char dispatch_error[1024];
static struct denise_cuda_psv_forward_stats dispatch_stats;
static int dispatch_stats_valid;

static int dispatch_failure(const char *format, ...) {
    va_list arguments;
    va_start(arguments,format);
    vsnprintf(dispatch_error,sizeof(dispatch_error),format,arguments);
    va_end(arguments);
    fprintf(stderr,"CUDA PSV forward dispatch failure: %s\n",dispatch_error);
    return -1;
}

static int validate_cuda_envelope(int nsrc_local,int ntr,int mode) {
    extern int FDORDER,L,NPROCX,NPROCY,BOUNDARY,FREE_SURF;
    extern int MODE,QUELLTYP,SEISMO,NDT,SNAP,INV_STF;
    if(mode!=0||MODE!=0||L!=1||FDORDER!=4||NPROCX!=1||NPROCY!=1||
       BOUNDARY!=0||FREE_SURF!=0||nsrc_local!=1||QUELLTYP!=1||
       SEISMO!=1||NDT!=1||SNAP!=0||INV_STF!=0||ntr<1)
        return dispatch_failure(
            "requested CUDA backend violates frozen envelope: mode=%d MODE=%d L=%d FDORDER=%d topology=%dx%d BOUNDARY=%d FREE_SURF=%d sources=%d QUELLTYP=%d SEISMO=%d NDT=%d SNAP=%d INV_STF=%d ntr=%d",
            mode,MODE,L,FDORDER,NPROCX,NPROCY,BOUNDARY,FREE_SURF,nsrc_local,
            QUELLTYP,SEISMO,NDT,SNAP,INV_STF,ntr);
    return 0;
}

int denise_cuda_psv_backend_preflight(
        int nsrc_local,int ntr,int mode,enum denise_psv_backend *backend) {
    const char *selector=getenv("DENISE_PSV_BACKEND");
    int visible_devices=0;
    dispatch_error[0]='\0'; dispatch_stats_valid=0;
    memset(&dispatch_stats,0,sizeof(dispatch_stats));
    if(!backend) return dispatch_failure("backend preflight output is null");
    *backend=DENISE_PSV_BACKEND_CPU;
    if(!selector||!selector[0]||strcmp(selector,"cpu")==0) return 0;
    if(strcmp(selector,"cuda")!=0)
        return dispatch_failure("unknown DENISE_PSV_BACKEND='%s'",selector);
    if(validate_cuda_envelope(nsrc_local,ntr,mode)!=0) return -1;
    if(denise_cuda_visible_device_count(&visible_devices)!=0)
        return dispatch_failure("CUDA device preflight failed: %s",
                                denise_cuda_last_error());
    if(visible_devices<1)
        return dispatch_failure("explicit CUDA request has no visible CUDA device");
    *backend=DENISE_PSV_BACKEND_CUDA;
    return 0;
}

static void bind_core_host(struct denise_cuda_psv_fd4_l1_host *host,
                           struct wavePSV *wave,
                           struct wavePSV_PML *pml,
                           struct matPSV *material) {
    memset(host,0,sizeof(*host));
    host->vx=wave->pvx; host->vy=wave->pvy;
    host->sxx=wave->psxx; host->syy=wave->psyy; host->sxy=wave->psxy;
    host->r=wave->pr; host->p=wave->pp; host->q=wave->pq;
    host->rip=material->prip; host->rjp=material->prjp;
    host->fipjp=material->fipjp; host->f=material->f; host->g=material->g;
    host->dip=material->dip; host->d=material->d; host->e=material->e;
    host->K_x=pml->K_x; host->a_x=pml->a_x; host->b_x=pml->b_x;
    host->K_x_half=pml->K_x_half; host->a_x_half=pml->a_x_half;
    host->b_x_half=pml->b_x_half;
    host->K_y=pml->K_y; host->a_y=pml->a_y; host->b_y=pml->b_y;
    host->K_y_half=pml->K_y_half; host->a_y_half=pml->a_y_half;
    host->b_y_half=pml->b_y_half;
    host->psi_sxx_x=pml->psi_sxx_x; host->psi_sxy_x=pml->psi_sxy_x;
    host->psi_syy_y=pml->psi_syy_y; host->psi_sxy_y=pml->psi_sxy_y;
    host->psi_vxx=pml->psi_vxx; host->psi_vyx=pml->psi_vyx;
    host->psi_vyy=pml->psi_vyy; host->psi_vxy=pml->psi_vxy;
}

static void write_dispatch_report(const struct denise_cuda_psv_forward_stats *s) {
    const char *path=getenv("DENISE_CUDA_PSV_REPORT_FILE");
    FILE *stream;
    if(!path||!path[0]) return;
    stream=fopen(path,"w");
    if(!stream) {
        dispatch_failure("cannot open CUDA report file %s",path);
        return;
    }
    fprintf(stream,
        "{\n"
        "  \"schema\": \"denise.cuda_psv_forward.v1\",\n"
        "  \"b1_core_bytes\": %zu,\n"
        "  \"source_signal_bytes\": %zu,\n"
        "  \"source_geometry_bytes\": %zu,\n"
        "  \"receiver_geometry_bytes\": %zu,\n"
        "  \"trace_bytes\": %zu,\n"
        "  \"workspace_bytes\": %zu,\n"
        "  \"total_mandatory_bytes\": %zu,\n"
        "  \"usable_budget_bytes\": %zu,\n"
        "  \"remaining_budget_bytes\": %zu,\n"
        "  \"h2d_transfer_calls\": %zu,\n"
        "  \"d2h_transfer_calls\": %zu,\n"
        "  \"h2d_bytes\": %zu,\n"
        "  \"d2h_bytes\": %zu,\n"
        "  \"full_grid_h2d_per_timestep\": %zu,\n"
        "  \"full_grid_d2h_per_timestep\": %zu,\n"
        "  \"source_sample_h2d_per_timestep\": %zu,\n"
        "  \"receiver_sample_d2h_per_timestep\": %zu,\n"
        "  \"timesteps\": %llu,\n"
        "  \"velocity_kernel_ms\": %.9g,\n"
        "  \"stress_kernel_ms\": %.9g,\n"
        "  \"source_kernel_ms\": %.9g,\n"
        "  \"receiver_kernel_ms\": %.9g,\n"
        "  \"resident_timestep_ms\": %.9g,\n"
        "  \"initial_upload_ms\": %.9g,\n"
        "  \"trace_download_ms\": %.9g,\n"
        "  \"mutable_download_ms\": %.9g,\n"
        "  \"total_forward_ms\": %.9g\n"
        "}\n",
        s->b1_core_bytes,s->source_signal_bytes,s->source_geometry_bytes,
        s->receiver_geometry_bytes,s->trace_bytes,s->workspace_bytes,
        s->total_mandatory_bytes,s->usable_budget_bytes,s->remaining_budget_bytes,
        s->h2d_transfer_calls,s->d2h_transfer_calls,s->h2d_bytes,s->d2h_bytes,
        s->full_grid_h2d_per_timestep,s->full_grid_d2h_per_timestep,
        s->source_sample_h2d_per_timestep,s->receiver_sample_d2h_per_timestep,
        s->timesteps,s->velocity_kernel_ms,s->stress_kernel_ms,
        s->source_kernel_ms,s->receiver_kernel_ms,s->resident_timestep_ms,
        s->initial_upload_ms,s->trace_download_ms,s->mutable_download_ms,
        s->total_forward_ms);
    fclose(stream);
}

const char *denise_cuda_psv_dispatch_last_error(void) { return dispatch_error; }

int denise_cuda_psv_dispatch_last_stats(
        struct denise_cuda_psv_forward_stats *stats) {
    if(!stats||!dispatch_stats_valid) return -1;
    *stats=dispatch_stats;
    return 0;
}

int denise_cuda_psv_dispatch(
        struct wavePSV *wave,
        struct wavePSV_PML *pml,
        struct matPSV *material,
        struct seisPSV *seismogram,
        struct acq *acquisition,
        float *hc,
        int nsrc_local,
        int ntr,
        int mode,
        enum denise_psv_backend backend) {
    extern float DT,DH;
    extern int NX,NY,NT,FW,FDORDER,L,NPROCX,NPROCY,BOUNDARY,FREE_SURF;
    extern int MODE,QUELLTYP,SEISMO,NDT,SNAP,INV_STF;
    struct denise_cuda_psv_forward_config config;
    struct denise_cuda_psv_forward_host host;
    struct denise_cuda_psv_forward *context=NULL;
    double start;
    int result=-1;

    dispatch_error[0]='\0'; dispatch_stats_valid=0;
    memset(&dispatch_stats,0,sizeof(dispatch_stats));
    if(backend==DENISE_PSV_BACKEND_CPU) return 0;
    if(backend!=DENISE_PSV_BACKEND_CUDA)
        return dispatch_failure("dispatch received invalid preflight backend %d",
                                (int)backend);

    memset(&config,0,sizeof(config)); memset(&host,0,sizeof(host));
    config.core.nx=NX; config.core.ny=NY; config.core.fw=FW;
    config.core.fdorder=FDORDER; config.core.mechanisms=L;
    config.core.mpi_ranks_x=NPROCX; config.core.mpi_ranks_y=NPROCY;
    config.core.boundary=BOUNDARY; config.core.free_surface=FREE_SURF;
    config.core.mode=mode; config.core.logical_device=0;
    config.core.dt=DT; config.core.dh=DH;
    config.core.hc1=hc[1]; config.core.hc2=hc[2];
    config.core.bip1=material->bip[1]; config.core.bjm1=material->bjm[1];
    config.core.cip1=material->cip[1]; config.core.cjm1=material->cjm[1];
    config.core.safety_reserve_bytes=(size_t)64*1024*1024;
    config.nt=NT; config.ntr=ntr; config.global_mode=MODE;
    config.source_count=nsrc_local; config.source_type=QUELLTYP;
    config.seismo=SEISMO; config.ndt=NDT; config.snapshots=SNAP;
    config.inv_stf=INV_STF;
    bind_core_host(&host.core,wave,pml,material);
    host.source_positions=acquisition->srcpos_loc;
    host.source_signals=acquisition->signals;
    host.receiver_positions=acquisition->recpos_loc;
    host.sectionvx=seismogram->sectionvx;
    host.sectionvy=seismogram->sectionvy;

    start=MPI_Wtime();
    if(denise_cuda_psv_forward_create(&config,&host,&context)!=0) goto fail;
    if(denise_cuda_psv_forward_run(context)!=0) goto fail;
    if(denise_cuda_psv_forward_download_traces(
            context,seismogram->sectionvx,seismogram->sectionvy)!=0) goto fail;
    if(denise_cuda_psv_forward_download_mutable(context,&host.core)!=0) goto fail;
    if(denise_cuda_psv_forward_get_stats(context,&dispatch_stats)!=0) goto fail;
    dispatch_stats.total_forward_ms=(float)((MPI_Wtime()-start)*1000.0);
    dispatch_stats_valid=1;
    write_dispatch_report(&dispatch_stats);
    result=1;
    goto cleanup;
fail:
    snprintf(dispatch_error,sizeof(dispatch_error),"%s",
             denise_cuda_psv_forward_last_error());
cleanup:
    if(denise_cuda_psv_forward_destroy(&context)!=0&&result>0) {
        snprintf(dispatch_error,sizeof(dispatch_error),"%s",
                 denise_cuda_psv_forward_last_error());
        result=-1;
    }
    return result;
}
