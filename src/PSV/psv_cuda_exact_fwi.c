#include "fd.h"
#include "denise_cuda_psv_exact_fwi.h"

#include <stdarg.h>
#include <stdint.h>

extern int NX,NY,NT,L,Q_PARAMETERIZATION_MODE;
extern float *FL,Q_APPROX_FMIN,Q_APPROX_FMAX,Q_APPROX_DF;
extern char JACOBIAN[STRING_SIZE];

enum { CUDA_PHYSICAL_FIELDS=5 };

static struct {
    struct denise_cuda_psv_forward *context;
    struct denise_cuda_psv_forward_config config;
    struct denise_cuda_psv_forward_stats forward;
    struct denise_cuda_psv_adjoint_stats adjoint;
    struct denise_cuda_psv_adjoint_sweep_stats sweep;
    struct denise_cuda_psv_native_gradient_stats native;
    struct denise_cuda_psv_physical_gradient_stats physical;
    int active,complete,segments,q_mode,trial_forwards,trial_failures;
    size_t trial_trace_d2h_calls,trial_trace_d2h_bytes;
    size_t context_destroy_calls;
    float *raw_modeled_test;
    size_t raw_modeled_test_elements;
    double base_objective;
    uint64_t field_hash[CUDA_PHYSICAL_FIELDS];
    uint64_t download_fp32_hash[CUDA_PHYSICAL_FIELDS];
    uint64_t production_matrix_hash[CUDA_PHYSICAL_FIELDS];
    double field_sum[CUDA_PHYSICAL_FIELDS];
    double field_absmax[CUDA_PHYSICAL_FIELDS];
    size_t field_nonzero[CUDA_PHYSICAL_FIELDS];
    int field_finite[CUDA_PHYSICAL_FIELDS];
} cuda_fwi;

static char cuda_fwi_error[1024];

static int adapter_failure(const char *format, ...) {
    va_list arguments;
    va_start(arguments,format);
    vsnprintf(cuda_fwi_error,sizeof(cuda_fwi_error),format,arguments);
    va_end(arguments);
    fprintf(stderr,"CUDA exact-visco P/SV FWI failure: %s\n",cuda_fwi_error);
    return -1;
}

static int requested_segments(void) {
    const char *value=getenv("DENISE_PSV_EXACT_SEGMENTS");
    char *end=NULL;
    long parsed;
    if(!value||!value[0]) return 32;
    parsed=strtol(value,&end,10);
    if(!end||*end||parsed<1||parsed>2147483647L) return -1;
    return (int)parsed;
}

static void preserve_cuda_error(const char *operation) {
    snprintf(cuda_fwi_error,sizeof(cuda_fwi_error),"%s: %s",operation,
             denise_cuda_psv_forward_last_error());
}

static int destroy_context(void) {
    if(!cuda_fwi.context) return 0;
    if(denise_cuda_psv_forward_destroy(&cuda_fwi.context)!=0) {
        preserve_cuda_error("context destruction");
        return -1;
    }
    ++cuda_fwi.context_destroy_calls;
    return 0;
}

static uint64_t hash_bytes(const void *data,size_t bytes) {
    const unsigned char *p=(const unsigned char *)data;
    uint64_t hash=UINT64_C(1469598103934665603);
    size_t k;
    for(k=0;k<bytes;++k) { hash^=p[k]; hash*=UINT64_C(1099511628211); }
    return hash;
}

static uint64_t hash_float_value(uint64_t hash,float value) {
    const unsigned char *p=(const unsigned char *)&value;
    size_t k;
    for(k=0;k<sizeof(value);++k) { hash^=p[k]; hash*=UINT64_C(1099511628211); }
    return hash;
}

static void summarize_fields(double *field[CUDA_PHYSICAL_FIELDS],size_t cells) {
    int f;
    size_t p;
    for(f=0;f<CUDA_PHYSICAL_FIELDS;++f) {
        cuda_fwi.field_hash[f]=hash_bytes(field[f],cells*sizeof(double));
        cuda_fwi.field_sum[f]=cuda_fwi.field_absmax[f]=0.0;
        cuda_fwi.field_nonzero[f]=0;
        cuda_fwi.field_finite[f]=1;
        for(p=0;p<cells;++p) {
            double value=field[f][p],absolute=fabs(value);
            if(!isfinite(value)) cuda_fwi.field_finite[f]=0;
            if(value!=0.0) ++cuda_fwi.field_nonzero[f];
            cuda_fwi.field_sum[f]+=value;
            if(absolute>cuda_fwi.field_absmax[f])
                cuda_fwi.field_absmax[f]=absolute;
        }
    }
}

static void write_report(void) {
    const char *path=getenv("DENISE_CUDA_PSV_EXACT_FWI_REPORT_FILE");
    static const char *name[CUDA_PHYSICAL_FIELDS]={"vp","vs","rho","qp","qs"};
    FILE *stream;
    int f;
    if(!path||!path[0]) return;
    stream=fopen(path,"w");
    if(!stream) {
        fprintf(stderr,"CUDA exact-visco P/SV FWI report open failed: %s\n",path);
        return;
    }
    fprintf(stream,
        "{\n"
        "  \"schema\": \"denise.cuda_psv_exact_fwi.v1\",\n"
        "  \"complete\": %s,\n"
        "  \"segments\": %d,\n"
        "  \"q_parameterization_mode\": %d,\n"
        "  \"base_objective\": %.17g,\n"
        "  \"trial_forward_count\": %d,\n"
        "  \"trial_failure_count\": %d,\n"
        "  \"trial_trace_d2h_calls\": %zu,\n"
        "  \"trial_trace_d2h_bytes\": %zu,\n"
        "  \"context_destroy_calls\": %zu,\n"
        "  \"original_h2d_calls\": %zu,\n"
        "  \"original_h2d_bytes\": %zu,\n"
        "  \"segment_d2d_calls\": %zu,\n"
        "  \"checkpoint_d2d_bytes\": %zu,\n"
        "  \"modeled_trace_d2h_calls\": %zu,\n"
        "  \"modeled_trace_d2h_bytes\": %zu,\n"
        "  \"observed_trace_h2d_calls\": %zu,\n"
        "  \"observed_trace_h2d_bytes\": %zu,\n"
        "  \"production_residual_h2d_calls\": %zu,\n"
        "  \"production_residual_h2d_bytes\": %zu,\n"
        "  \"physical_gradient_d2h_calls\": %zu,\n"
        "  \"physical_gradient_d2h_bytes\": %zu,\n"
        "  \"adjoint_initial_h2d_calls\": %zu,\n"
        "  \"adjoint_initial_zero_calls\": %zu,\n"
        "  \"reverse_timesteps\": %llu,\n"
        "  \"reverse_segments\": %zu,\n"
        "  \"forward_synchronization_calls\": %zu,\n"
        "  \"replay_synchronization_calls\": %zu,\n"
        "  \"reverse_segment_synchronization_calls\": %zu,\n"
        "  \"per_step_h2d_calls\": %zu,\n"
        "  \"per_step_d2h_calls\": %zu,\n"
        "  \"per_step_allocation_calls\": %zu,\n"
        "  \"per_step_free_calls\": %zu,\n"
        "  \"per_step_blocking_sync_calls\": %zu,\n"
        "  \"native_allocation_calls\": %zu,\n"
        "  \"physical_allocation_calls\": %zu,\n"
        "  \"map_kernel_launches\": %zu,\n"
        "  \"map_synchronization_calls\": %zu,\n"
        "  \"fields\": {\n",
        cuda_fwi.complete?"true":"false",cuda_fwi.segments,cuda_fwi.q_mode,
        cuda_fwi.base_objective,cuda_fwi.trial_forwards,cuda_fwi.trial_failures,
        cuda_fwi.trial_trace_d2h_calls,cuda_fwi.trial_trace_d2h_bytes,
        cuda_fwi.context_destroy_calls,
        cuda_fwi.forward.h2d_transfer_calls,cuda_fwi.forward.h2d_bytes,
        cuda_fwi.forward.segment_d2d_calls,cuda_fwi.forward.checkpoint_d2d_bytes,
        cuda_fwi.forward.d2h_transfer_calls,cuda_fwi.forward.d2h_bytes,
        cuda_fwi.sweep.observed_h2d_calls,cuda_fwi.sweep.observed_h2d_bytes,
        cuda_fwi.sweep.production_residual_h2d_calls,
        cuda_fwi.sweep.production_residual_h2d_bytes,
        cuda_fwi.physical.diagnostic_d2h_calls,
        cuda_fwi.physical.diagnostic_d2h_bytes,
        cuda_fwi.adjoint.initial_h2d_calls,cuda_fwi.adjoint.initial_zero_calls,
        cuda_fwi.sweep.reverse_timesteps,cuda_fwi.sweep.reverse_segments,
        cuda_fwi.forward.forward_synchronization_calls,
        cuda_fwi.sweep.replay_synchronization_calls,
        cuda_fwi.sweep.reverse_segment_synchronization_calls,
        cuda_fwi.sweep.per_step_residual_h2d_calls+
            cuda_fwi.native.per_step_h2d_calls+
            cuda_fwi.physical.per_step_h2d_calls,
        cuda_fwi.sweep.per_step_full_state_d2h_calls+
            cuda_fwi.native.per_step_d2h_calls+
            cuda_fwi.physical.per_step_d2h_calls,
        cuda_fwi.sweep.per_step_allocation_calls+
            cuda_fwi.native.per_step_allocation_calls+
            cuda_fwi.physical.per_step_allocation_calls,
        cuda_fwi.sweep.per_step_free_calls+
            cuda_fwi.native.per_step_free_calls+
            cuda_fwi.physical.per_step_free_calls,
        cuda_fwi.sweep.per_step_blocking_sync_calls+
            cuda_fwi.native.per_step_blocking_sync_calls+
            cuda_fwi.physical.per_step_blocking_sync_calls,
        cuda_fwi.native.allocation_calls,cuda_fwi.physical.allocation_calls,
        cuda_fwi.physical.map_kernel_launches,
        cuda_fwi.physical.map_synchronization_calls);
    for(f=0;f<CUDA_PHYSICAL_FIELDS;++f)
        fprintf(stream,
            "    \"%s\": {\"finite\": %s, \"nonzero\": %zu, "
            "\"sum\": %.17g, \"absmax\": %.17g, "
            "\"fnv1a64\": \"%016llx\", "
            "\"download_fp32_fnv1a64\": \"%016llx\", "
            "\"production_matrix_fnv1a64\": \"%016llx\"}%s\n",
            name[f],cuda_fwi.field_finite[f]?"true":"false",
            cuda_fwi.field_nonzero[f],cuda_fwi.field_sum[f],
            cuda_fwi.field_absmax[f],
            (unsigned long long)cuda_fwi.field_hash[f],
            (unsigned long long)cuda_fwi.download_fp32_hash[f],
            (unsigned long long)cuda_fwi.production_matrix_hash[f],
            f+1<CUDA_PHYSICAL_FIELDS?",":"");
    fprintf(stream,"  }\n}\n");
    fclose(stream);
}

static int diagnostic_enabled(void) {
    const char *flag=getenv("DENISE_PSV_EXACT_VISCO_GRADIENT");
    return flag&&flag[0]=='1'&&flag[1]=='\0';
}

static int write_field(const char *suffix,const double *gradient) {
    char path[STRING_SIZE+64];
    FILE *stream;
    int i,j;
    snprintf(path,sizeof(path),"%s.raw.%s",JACOBIAN,suffix);
    stream=fopen(path,"wb");
    if(!stream) return adapter_failure("cannot open %s",path);
    for(i=1;i<=NX;++i) for(j=1;j<=NY;++j) {
        float value=(float)gradient[(size_t)(j-1)*NX+(i-1)];
        if(fwrite(&value,sizeof(value),1,stream)!=1) {
            fclose(stream);
            return adapter_failure("cannot write %s",path);
        }
    }
    if(fclose(stream)!=0) return adapter_failure("cannot close %s",path);
    return 0;
}

static int test_raw_residual_enabled(void) {
    const char *flag=getenv("DENISE_CUDA_PSV_TEST_USE_RAW_MODELED_OBSERVED");
    return flag&&flag[0]=='1'&&flag[1]=='\0';
}

static int write_production_residual_diagnostic(
        const float *vx,const float *vy,size_t elements) {
    const char *path=getenv("DENISE_CUDA_PSV_PRODUCTION_RESIDUAL_FILE");
    FILE *stream;
    if(!path||!path[0]) return 0;
    stream=fopen(path,"wb");
    if(!stream) return adapter_failure("cannot open %s",path);
    if(fwrite(vx,sizeof(float),elements,stream)!=elements||
       fwrite(vy,sizeof(float),elements,stream)!=elements) {
        fclose(stream); return adapter_failure("cannot write %s",path);
    }
    if(fclose(stream)!=0) return adapter_failure("cannot close %s",path);
    return 0;
}

static int write_raw_residual_test_diagnostic(
        const struct visco_psv_exact_fwi_request *request) {
    const char *path=getenv("DENISE_CUDA_PSV_TEST_RAW_RESIDUAL_FILE");
    FILE *stream;
    int component,k,t;
    if(!path||!path[0]) return 0;
    if(!cuda_fwi.raw_modeled_test||
       cuda_fwi.raw_modeled_test_elements!=(size_t)request->ntr*request->ns||
       !request->data->sectionvxdata||
       !request->data->sectionvydata)
        return adapter_failure("raw-residual diagnostic inputs are unavailable");
    stream=fopen(path,"wb");
    if(!stream) return adapter_failure("cannot open %s",path);
    for(component=0;component<2;++component)
        for(k=1;k<=request->ntr;++k) for(t=1;t<=request->ns;++t) {
            size_t sample=(size_t)(k-1)*request->ns+(size_t)(t-1);
            float modeled=cuda_fwi.raw_modeled_test[
                (size_t)component*cuda_fwi.raw_modeled_test_elements+sample];
            float observed=component==0?request->data->sectionvxdata[k][t]:
                                        request->data->sectionvydata[k][t];
            float value=t==1?0.0f:modeled-observed;
            if(fwrite(&value,sizeof(value),1,stream)!=1) {
                fclose(stream); return adapter_failure("cannot write %s",path);
            }
        }
    if(fclose(stream)!=0) return adapter_failure("cannot close %s",path);
    return 0;
}

int denise_cuda_psv_exact_fwi_selected(void) {
    const char *selector=getenv("DENISE_PSV_BACKEND");
    return selector&&strcmp(selector,"cuda")==0;
}

const char *denise_cuda_psv_exact_fwi_last_error(void) {
    return cuda_fwi_error;
}

int denise_cuda_psv_exact_fwi_original_forward(
        const struct denise_cuda_psv_forward_config *config,
        const struct denise_cuda_psv_forward_host *host,
        float **sectionvx,float **sectionvy) {
    int requested,segment,begin,end,result=-1;
    cuda_fwi_error[0]='\0';
    if(!config||!host||!sectionvx||!sectionvy)
        return adapter_failure("original-forward adapter is incomplete");
    if(cuda_fwi.active||cuda_fwi.context)
        return adapter_failure("a CUDA exact-visco FWI context is already active");
    memset(&cuda_fwi,0,sizeof(cuda_fwi));
    cuda_fwi.config=*config;
    requested=requested_segments();
    if(requested<1)
        return adapter_failure(
            "DENISE_PSV_EXACT_SEGMENTS must be a positive integer");
    cuda_fwi.segments=requested>config->nt?config->nt:requested;
    if(denise_cuda_psv_forward_create(config,host,&cuda_fwi.context)!=0) {
        preserve_cuda_error("original forward create"); goto cleanup;
    }
    cuda_fwi.active=1;
    if(denise_cuda_psv_forward_segments_prepare(cuda_fwi.context,requested)!=0) {
        preserve_cuda_error("segment prepare"); goto cleanup;
    }
    for(segment=0;segment<cuda_fwi.segments;++segment) {
        if(denise_cuda_psv_forward_segment_bounds(
                cuda_fwi.context,segment,&begin,&end)!=0) {
            preserve_cuda_error("segment bounds"); goto cleanup;
        }
        if(denise_cuda_psv_forward_run_range(
                cuda_fwi.context,begin,end)!=0) {
            preserve_cuda_error("original forward segment"); goto cleanup;
        }
        if(segment+1<cuda_fwi.segments&&
           denise_cuda_psv_forward_segment_capture(
                cuda_fwi.context,segment)!=0) {
            preserve_cuda_error("segment checkpoint capture"); goto cleanup;
        }
    }
    if(denise_cuda_psv_forward_download_traces(
            cuda_fwi.context,sectionvx,sectionvy)!=0) {
        preserve_cuda_error("modeled trace download"); goto cleanup;
    }
    if(getenv("DENISE_CUDA_PSV_TEST_RAW_RESIDUAL_FILE")) {
        size_t elements=(size_t)config->ntr*config->nt;
        cuda_fwi.raw_modeled_test=(float *)malloc(2*elements*sizeof(float));
        if(!cuda_fwi.raw_modeled_test) {
            adapter_failure("raw modeled-trace diagnostic allocation failed");
            goto cleanup;
        }
        memcpy(cuda_fwi.raw_modeled_test,&sectionvx[1][1],
               elements*sizeof(float));
        memcpy(cuda_fwi.raw_modeled_test+elements,&sectionvy[1][1],
               elements*sizeof(float));
        cuda_fwi.raw_modeled_test_elements=elements;
    }
    if(denise_cuda_psv_forward_get_stats(
            cuda_fwi.context,&cuda_fwi.forward)!=0) {
        preserve_cuda_error("forward statistics"); goto cleanup;
    }
    return 1;
cleanup:
    cuda_fwi.active=0;
    destroy_context();
    free(cuda_fwi.raw_modeled_test);
    cuda_fwi.raw_modeled_test=NULL;
    cuda_fwi.raw_modeled_test_elements=0;
    write_report();
    return result;
}

int denise_cuda_psv_exact_fwi_finish(
        const struct visco_psv_exact_fwi_request *request) {
    struct q_tau_mapping mapping;
    struct denise_cuda_psv_physical_material_host material;
    struct denise_cuda_psv_physical_gradient_host download;
    double *storage=NULL,*field[CUDA_PHYSICAL_FIELDS];
    float *production_residual=NULL;
    size_t cells=(size_t)NX*NY,p,residual_elements;
    int f,i,j,k,t,result=-1;
    static const char *suffix[CUDA_PHYSICAL_FIELDS]={"vp","vs","rho","qp","qs"};
    cuda_fwi_error[0]='\0';
    if(!request||!request->material||!request->fwi||!request->data||
       !request->data->sectionvxdiff||!request->data->sectionvydiff||
       !cuda_fwi.active||!cuda_fwi.context)
        return adapter_failure(
            "finish has no owning CUDA context or calc_res_PSV residual");
    if(request->ntr!=cuda_fwi.config.ntr||request->ns!=cuda_fwi.config.nt)
        return adapter_failure("residual dimensions changed before CUDA reverse sweep");
    cuda_fwi.base_objective=request->base_objective;
    cuda_fwi.q_mode=Q_PARAMETERIZATION_MODE;
    if(denise_cuda_psv_adjoint_prepare_zero(cuda_fwi.context)!=0) {
        preserve_cuda_error("zero adjoint prepare"); goto cleanup;
    }
    residual_elements=(size_t)request->ntr*request->ns;
    production_residual=(float *)malloc(2*residual_elements*sizeof(float));
    if(!production_residual) {
        adapter_failure("host production-residual allocation failed"); goto cleanup;
    }
    for(k=1;k<=request->ntr;++k) for(t=1;t<=request->ns;++t) {
        size_t sample=(size_t)(k-1)*request->ns+(size_t)(t-1);
        production_residual[sample]=
            request->data->sectionvxdiff[k][request->ns-t+1];
        production_residual[residual_elements+sample]=
            request->data->sectionvydiff[k][request->ns-t+1];
    }
    if(write_production_residual_diagnostic(production_residual,
            production_residual+residual_elements,residual_elements)!=0)
        goto cleanup;
    if(write_raw_residual_test_diagnostic(request)!=0) goto cleanup;
    if(test_raw_residual_enabled()) {
        if(!request->data->sectionvxdata||!request->data->sectionvydata) {
            adapter_failure("raw-residual test hook has no observed traces");
            goto cleanup;
        }
        if(denise_cuda_psv_adjoint_sweep_prepare(cuda_fwi.context,
                &request->data->sectionvxdata[1][1],
                &request->data->sectionvydata[1][1],residual_elements)!=0) {
            preserve_cuda_error("raw modeled-observed test sweep prepare");
            goto cleanup;
        }
    } else if(denise_cuda_psv_adjoint_production_residual_prepare(
                  cuda_fwi.context,production_residual,
                  production_residual+residual_elements,residual_elements)!=0) {
        preserve_cuda_error("production residual upload/sweep prepare");
        goto cleanup;
    }
    free(production_residual); production_residual=NULL;
    if(denise_cuda_psv_native_gradient_prepare(cuda_fwi.context)!=0) {
        preserve_cuda_error("native gradient prepare"); goto cleanup;
    }
    init_q_tau_mapping(&mapping,Q_PARAMETERIZATION_MODE,L,FL,
                       Q_APPROX_FMIN,Q_APPROX_FMAX,Q_APPROX_DF);
    memset(&material,0,sizeof(material));
    material.prho=request->material->prho;
    material.ppi=request->material->ppi;
    material.pu=request->material->pu;
    material.ptaus=request->material->ptaus;
    material.ptaup=request->material->ptaup;
    material.puipjp=request->material->puipjp;
    material.ptausipjp=request->material->ptausipjp;
    material.eta=request->material->peta[1];
    material.q_parameterization_mode=Q_PARAMETERIZATION_MODE;
    material.inverse_tau_per_q=mapping.inverse_tau_per_q;
    material.inverse_tau_offset=mapping.inverse_tau_offset;
    if(denise_cuda_psv_physical_gradient_prepare(
            cuda_fwi.context,&material)!=0) {
        preserve_cuda_error("physical gradient prepare"); goto cleanup;
    }
    if(denise_cuda_psv_adjoint_reverse_sweep(cuda_fwi.context)!=0) {
        preserve_cuda_error("segmented reverse sweep"); goto cleanup;
    }
    if(denise_cuda_psv_physical_gradient_map(cuda_fwi.context)!=0) {
        preserve_cuda_error("native-to-physical map"); goto cleanup;
    }
    storage=(double *)malloc(CUDA_PHYSICAL_FIELDS*cells*sizeof(double));
    if(!storage) { adapter_failure("host physical-gradient allocation failed"); goto cleanup; }
    for(f=0;f<CUDA_PHYSICAL_FIELDS;++f) field[f]=storage+(size_t)f*cells;
    download.vp=field[0]; download.vs=field[1]; download.rho=field[2];
    download.qp=field[3]; download.qs=field[4];
    if(denise_cuda_psv_physical_gradient_download(
            cuda_fwi.context,&download,cells)!=0) {
        preserve_cuda_error("five-field physical-gradient download"); goto cleanup;
    }
    if(denise_cuda_psv_forward_get_stats(cuda_fwi.context,&cuda_fwi.forward)!=0||
       denise_cuda_psv_adjoint_get_stats(cuda_fwi.context,&cuda_fwi.adjoint)!=0||
       denise_cuda_psv_adjoint_sweep_get_stats(cuda_fwi.context,&cuda_fwi.sweep)!=0||
       denise_cuda_psv_native_gradient_get_stats(cuda_fwi.context,&cuda_fwi.native)!=0||
       denise_cuda_psv_physical_gradient_get_stats(
            cuda_fwi.context,&cuda_fwi.physical)!=0) {
        preserve_cuda_error("lifecycle statistics"); goto cleanup;
    }
    summarize_fields(field,cells);
    for(f=0;f<CUDA_PHYSICAL_FIELDS;++f)
        if(!cuda_fwi.field_finite[f]) {
            adapter_failure("physical gradient %s contains non-finite values",suffix[f]);
            goto cleanup;
        }
    cuda_fwi.active=0;
    if(destroy_context()!=0) goto cleanup;
    for(j=1;j<=NY;++j) for(i=1;i<=NX;++i) {
        p=(size_t)(j-1)*NX+(i-1);
        request->fwi->waveconv[j][i]+=(float)field[0][p];
        request->fwi->waveconv_u[j][i]+=(float)field[1][p];
        request->fwi->waveconv_rho[j][i]+=(float)field[2][p];
        request->fwi->waveconv_qp_exact[j][i]+=(float)field[3][p];
        request->fwi->waveconv_qs_exact[j][i]+=(float)field[4][p];
    }
    {
        float **production[CUDA_PHYSICAL_FIELDS]={request->fwi->waveconv,
            request->fwi->waveconv_u,request->fwi->waveconv_rho,
            request->fwi->waveconv_qp_exact,request->fwi->waveconv_qs_exact};
        for(f=0;f<CUDA_PHYSICAL_FIELDS;++f) {
            uint64_t direct=UINT64_C(1469598103934665603);
            uint64_t associated=UINT64_C(1469598103934665603);
            for(j=1;j<=NY;++j) for(i=1;i<=NX;++i) {
                float value=(float)field[f][(size_t)(j-1)*NX+(i-1)];
                direct=hash_float_value(direct,value);
                associated=hash_float_value(associated,production[f][j][i]);
            }
            cuda_fwi.download_fp32_hash[f]=direct;
            cuda_fwi.production_matrix_hash[f]=associated;
            if(direct!=associated) {
                adapter_failure(
                    "production matrix association differs from direct CUDA %s download",
                    suffix[f]);
                goto cleanup;
            }
        }
    }
    if(diagnostic_enabled())
        for(f=0;f<CUDA_PHYSICAL_FIELDS;++f)
            if(write_field(suffix[f],field[f])!=0) goto cleanup;
    cuda_fwi.complete=1;
    result=1;
cleanup:
    if(cuda_fwi.context) {
        cuda_fwi.active=0;
        destroy_context();
    }
    free(storage);
    free(production_residual);
    free(cuda_fwi.raw_modeled_test);
    cuda_fwi.raw_modeled_test=NULL;
    cuda_fwi.raw_modeled_test_elements=0;
    write_report();
    return result;
}

void denise_cuda_psv_exact_fwi_trial_complete(
        const struct denise_cuda_psv_forward_stats *stats) {
    ++cuda_fwi.trial_forwards;
    if(stats) {
        cuda_fwi.trial_trace_d2h_calls+=stats->d2h_transfer_calls;
        cuda_fwi.trial_trace_d2h_bytes+=stats->d2h_bytes;
    }
    write_report();
}

void denise_cuda_psv_exact_fwi_trial_failed(const char *message) {
    (void)message;
    ++cuda_fwi.trial_failures;
    write_report();
}
