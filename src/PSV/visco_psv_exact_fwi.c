/* Active five-parameter viscoelastic P/SV steepest-descent lifecycle. */
#include "fd.h"

extern int NX, NY, MYID_SHOT, POS[3], STEPMAX, NPROCX, NPROCY;
extern int INV_VP_ITER, INV_VS_ITER, INV_RHO_ITER, INV_QS_ITER;
extern int Q_PARAMETERIZATION_MODE;
extern float EPS_SCALE, SCALEFAC;
extern float VPLOWERLIM, VPUPPERLIM, VSLOWERLIM, VSUPPERLIM;
extern float RHOLOWERLIM, RHOUPPERLIM, QSLOWERLIM, QSUPPERLIM;
extern float *FL, Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF;
extern char INV_MODELFILE[STRING_SIZE], JACOBIAN[STRING_SIZE];
extern MPI_Comm SHOT_COMM;

enum { EXACT_VP, EXACT_VS, EXACT_RHO, EXACT_QP, EXACT_QS, EXACT_FIELDS };
static const char *field_name[EXACT_FIELDS]={"vp","vs","rho","qp","qs"};

static void copy_field(float **from,float **to){
    int i,j;
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++) to[j][i]=from[j][i];
}

static double maximum_abs(float **field){
    int i,j;
    double result=0.0;
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++)
        if(fabs((double)field[j][i])>result) result=fabs((double)field[j][i]);
    return result;
}

static void global_maxima(float ***fields,double result[EXACT_FIELDS],
                          int owner[EXACT_FIELDS]){
    struct { double value; int rank; } local,global;
    double local_max[EXACT_FIELDS];
    int k;
    for(k=0;k<EXACT_FIELDS;k++) local_max[k]=maximum_abs(fields[k]);
    MPI_Allreduce(local_max,result,EXACT_FIELDS,MPI_DOUBLE,MPI_MAX,SHOT_COMM);
    for(k=0;k<EXACT_FIELDS;k++){
        local.value=local_max[k]; local.rank=MYID_SHOT;
        MPI_Allreduce(&local,&global,1,MPI_DOUBLE_INT,MPI_MAXLOC,SHOT_COMM);
        owner[k]=global.rank;
    }
}

static void rebuild_tau(struct matPSV *material,
                        const struct q_tau_mapping *mapping){
    int i,j;
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++){
        material->ptaup[j][i]=q_to_tau(material->pqp[j][i],mapping);
        material->ptaus[j][i]=q_to_tau(material->pqs[j][i],mapping);
    }
}

static int trial_from_base(struct matPSV *material,float ***base,
                           float ***direction,double alpha){
    int i,j,k;
    double value[EXACT_FIELDS];
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++){
        for(k=0;k<EXACT_FIELDS;k++)
            value[k]=(double)base[k][j][i]-alpha*(double)direction[k][j][i];
        if(!isfinite(value[EXACT_VP]) || value[EXACT_VP]<VPLOWERLIM || value[EXACT_VP]>VPUPPERLIM ||
           !isfinite(value[EXACT_VS]) || value[EXACT_VS]<VSLOWERLIM || value[EXACT_VS]>VSUPPERLIM ||
           !isfinite(value[EXACT_RHO]) || value[EXACT_RHO]<RHOLOWERLIM || value[EXACT_RHO]>RHOUPPERLIM ||
           !isfinite(value[EXACT_QP]) || value[EXACT_QP]<=0.0 ||
           !isfinite(value[EXACT_QS]) || value[EXACT_QS]<=0.0 ||
           (QSLOWERLIM>0.0 && (value[EXACT_QP]<QSLOWERLIM || value[EXACT_QS]<QSLOWERLIM)) ||
           (QSUPPERLIM>0.0 && (value[EXACT_QP]>QSUPPERLIM || value[EXACT_QS]>QSUPPERLIM)))
            return 0;
    }
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++){
        material->ppi[j][i]=(float)(base[EXACT_VP][j][i]-alpha*direction[EXACT_VP][j][i]);
        material->pu[j][i]=(float)(base[EXACT_VS][j][i]-alpha*direction[EXACT_VS][j][i]);
        material->prho[j][i]=(float)(base[EXACT_RHO][j][i]-alpha*direction[EXACT_RHO][j][i]);
        material->pqp[j][i]=(float)(base[EXACT_QP][j][i]-alpha*direction[EXACT_QP][j][i]);
        material->pqs[j][i]=(float)(base[EXACT_QS][j][i]-alpha*direction[EXACT_QS][j][i]);
    }
    return 1;
}

static void restore_base(struct matPSV *material,float ***base,
                         const struct q_tau_mapping *mapping){
    copy_field(base[EXACT_VP],material->ppi);
    copy_field(base[EXACT_VS],material->pu);
    copy_field(base[EXACT_RHO],material->prho);
    copy_field(base[EXACT_QP],material->pqp);
    copy_field(base[EXACT_QS],material->pqs);
    rebuild_tau(material,mapping);
}

static void refresh_material(struct matPSV *material,
                             const struct q_tau_mapping *mapping){
    rebuild_tau(material,mapping);
    matcopy_PSV(material->prho,material->ppi,material->pu,
                material->ptaus,material->ptaup);
    MPI_Barrier(SHOT_COMM);
    av_mue(material->pu,material->puipjp,material->prho);
    av_rho(material->prho,material->prip,material->prjp);
    av_tau(material->ptaus,material->ptausipjp);
    prepare_update_s_visc_PSV(material->etajm,material->etaip,material->peta,
                              material->fipjp,material->pu,material->puipjp,
                              material->ppi,material->prho,material->ptaus,
                              material->ptaup,material->ptausipjp,material->f,
                              material->g,material->bip,material->bjm,
                              material->cip,material->cjm,material->dip,
                              material->d,material->e);
}

static int forced_invalid_rank(void){
    const char *value=getenv("DENISE_PSV_EXACT_TEST_INVALID_TRIAL_RANK");
    char *end;
    long rank;
    if(!value || !*value) return -1;
    rank=strtol(value,&end,10);
    if(*end!='\0' || rank<0 || rank>=NPROCX*NPROCY)
        err(" DENISE_PSV_EXACT_TEST_INVALID_TRIAL_RANK is invalid. ");
    return (int)rank;
}

static void persist_one(const char *prefix,const char *suffix,float **field){
    char filename[STRING_SIZE*2],local[STRING_SIZE*2];
    snprintf(filename,sizeof(filename),"%s.%s",prefix,suffix);
    writemod(filename,field,3);
    MPI_Barrier(SHOT_COMM);
    if(MYID_SHOT==0) mergemod(filename,3);
    MPI_Barrier(SHOT_COMM);
    snprintf(local,sizeof(local),"%s.%i.%i",filename,POS[1],POS[2]);
    remove(local);
}

static void persist_model(const char *prefix,struct matPSV *material){
    persist_one(prefix,"vp",material->ppi);
    persist_one(prefix,"vs",material->pu);
    persist_one(prefix,"rho",material->prho);
    persist_one(prefix,"qp",material->pqp);
    persist_one(prefix,"qs",material->pqs);
}

static void write_named_values(FILE *report,const char *name,
                               const double value[EXACT_FIELDS]){
    int k;
    fprintf(report,"  \"%s\": {",name);
    for(k=0;k<EXACT_FIELDS;k++)
        fprintf(report,"\"%s\": %.17g%s",field_name[k],value[k],
                k+1<EXACT_FIELDS?", ":"},\n");
}

static void write_named_ranks(FILE *report,const char *name,
                              const int value[EXACT_FIELDS]){
    int k;
    fprintf(report,"  \"%s\": {",name);
    for(k=0;k<EXACT_FIELDS;k++)
        fprintf(report,"\"%s\": %d%s",field_name[k],value[k],
                k+1<EXACT_FIELDS?", ":"},\n");
}

double visco_psv_exact_active_step(
        const struct visco_psv_exact_fwi_request *request){
    struct matPSV *material=request->material;
    struct fwiPSV *fwi=request->fwi;
    struct q_tau_mapping mapping;
    float **base[EXACT_FIELDS],**direction[EXACT_FIELDS],**gradient[EXACT_FIELDS];
    double model_max[EXACT_FIELDS],gradient_max[EXACT_FIELDS];
    double alpha,divisor,objective,accepted_objective=-1.0;
    double update_norm[EXACT_FIELDS]={0.0,0.0,0.0,0.0,0.0};
    double local_norm[EXACT_FIELDS]={0.0,0.0,0.0,0.0,0.0};
    double *trial_alpha,*trial_objective;
    double base_min,base_max,alpha_min,alpha_max,objective_min,objective_max;
    int model_owner[EXACT_FIELDS],gradient_owner[EXACT_FIELDS];
    int *trial_valid,*trial_decision,*trial_forward,*invalid_rank_count;
    int active[EXACT_FIELDS],i,j,k,trial_count=0,accepted=0,max_trials;
    int local_valid,global_valid,local_decision,decision_min,decision_max;
    int forward_count=0,forward_min,forward_max,invalid_rank;
    int finite_local,finite_min,finite_max;
    char report_path[STRING_SIZE*2],prefix[STRING_SIZE*2];
    FILE *report;

    if(!request || !visco_psv_exact_supported())
        err(" Exact visco PSV active step called outside its verified configuration. ");
    init_q_tau_mapping(&mapping,Q_PARAMETERIZATION_MODE,1,FL,
                       Q_APPROX_FMIN,Q_APPROX_FMAX,Q_APPROX_DF);
    gradient[EXACT_VP]=fwi->waveconv;
    gradient[EXACT_VS]=fwi->waveconv_u;
    gradient[EXACT_RHO]=fwi->waveconv_rho;
    gradient[EXACT_QP]=fwi->waveconv_qp_exact;
    gradient[EXACT_QS]=fwi->waveconv_qs_exact;
    active[EXACT_VP]=request->iter>=INV_VP_ITER;
    active[EXACT_VS]=request->iter>=INV_VS_ITER;
    active[EXACT_RHO]=request->iter>=INV_RHO_ITER;
    active[EXACT_QP]=active[EXACT_QS]=request->iter>=INV_QS_ITER;
    for(k=0;k<EXACT_FIELDS;k++){
        base[k]=matrix(1,NY,1,NX);
        direction[k]=matrix(1,NY,1,NX);
    }
    copy_field(material->ppi,base[EXACT_VP]);
    copy_field(material->pu,base[EXACT_VS]);
    copy_field(material->prho,base[EXACT_RHO]);
    copy_field(material->pqp,base[EXACT_QP]);
    copy_field(material->pqs,base[EXACT_QS]);

    global_maxima(base,model_max,model_owner);
    global_maxima(gradient,gradient_max,gradient_owner);
    for(k=0;k<EXACT_FIELDS;k++){
        if(active[k] && (!(gradient_max[k]>0.0) || !isfinite(gradient_max[k])))
            err(" Exact visco PSV active physical gradient is zero or non-finite. ");
        for(i=1;i<=NX;i++) for(j=1;j<=NY;j++)
            direction[k][j][i]=active[k]
                ? (float)(gradient[k][j][i]*model_max[k]/gradient_max[k]):0.0f;
    }
    MPI_Allreduce(&request->base_objective,&base_min,1,MPI_DOUBLE,MPI_MIN,SHOT_COMM);
    MPI_Allreduce(&request->base_objective,&base_max,1,MPI_DOUBLE,MPI_MAX,SHOT_COMM);
    if(base_min!=base_max) err(" Exact visco PSV base-objective consensus failed. ");

    max_trials=STEPMAX>0?STEPMAX+1:8;
    trial_alpha=calloc((size_t)max_trials,sizeof(double));
    trial_objective=calloc((size_t)max_trials,sizeof(double));
    trial_valid=calloc((size_t)max_trials,sizeof(int));
    trial_decision=calloc((size_t)max_trials,sizeof(int));
    trial_forward=calloc((size_t)max_trials,sizeof(int));
    invalid_rank_count=calloc((size_t)max_trials,sizeof(int));
    if(!trial_alpha || !trial_objective || !trial_valid || !trial_decision ||
       !trial_forward || !invalid_rank_count)
        err(" Exact visco PSV line-search allocation failed. ");
    invalid_rank=forced_invalid_rank();
    alpha=EPS_SCALE>0.0?EPS_SCALE:0.01;
    divisor=SCALEFAC>1.0?SCALEFAC:2.0;
    for(k=0;k<max_trials;k++){
        MPI_Allreduce(&alpha,&alpha_min,1,MPI_DOUBLE,MPI_MIN,SHOT_COMM);
        MPI_Allreduce(&alpha,&alpha_max,1,MPI_DOUBLE,MPI_MAX,SHOT_COMM);
        if(alpha_min!=alpha_max) err(" Exact visco PSV alpha consensus failed. ");
        trial_alpha[trial_count]=alpha;
        local_valid=trial_from_base(material,base,direction,alpha);
        if(k==0 && MYID_SHOT==invalid_rank) local_valid=0;
        MPI_Allreduce(&local_valid,&global_valid,1,MPI_INT,MPI_MIN,SHOT_COMM);
        i=local_valid?0:1;
        MPI_Allreduce(&i,&invalid_rank_count[trial_count],1,MPI_INT,MPI_SUM,SHOT_COMM);
        trial_valid[trial_count]=global_valid;
        if(!global_valid){
            local_decision=0;
            MPI_Allreduce(&local_decision,&decision_min,1,MPI_INT,MPI_MIN,SHOT_COMM);
            MPI_Allreduce(&local_decision,&decision_max,1,MPI_INT,MPI_MAX,SHOT_COMM);
            if(decision_min!=decision_max)
                err(" Exact visco PSV invalid-trial decision consensus failed. ");
            trial_objective[trial_count]=-1.0;
            trial_count++;
            alpha/=divisor;
            continue;
        }
        /* obj_psv performs matcopy_PSV, av_mue, av_rho, av_tau and
         * prepare_update_s_visc_PSV before the distributed trial forward. */
        rebuild_tau(material,&mapping);
        objective=obj_psv(request->wave,request->pml,material,fwi,request->mpi,
                          request->seis,request->data,request->acquisition,
                          request->hc,request->nsrc,request->nsrc_loc,
                          request->nsrc_glob,request->ntr,request->ntr_glob,
                          request->ns,2,request->iter,request->Ws,request->Wr,
                          request->hin,request->DTINV_help,(float)alpha,
                          request->req_send,request->req_rec);
        forward_count++;
        trial_forward[trial_count]=1;
        trial_objective[trial_count]=objective;
        finite_local=isfinite(objective)?1:0;
        MPI_Allreduce(&finite_local,&finite_min,1,MPI_INT,MPI_MIN,SHOT_COMM);
        MPI_Allreduce(&finite_local,&finite_max,1,MPI_INT,MPI_MAX,SHOT_COMM);
        if(finite_min!=finite_max)
            err(" Exact visco PSV objective finiteness consensus failed. ");
        if(finite_min){
            MPI_Allreduce(&objective,&objective_min,1,MPI_DOUBLE,MPI_MIN,SHOT_COMM);
            MPI_Allreduce(&objective,&objective_max,1,MPI_DOUBLE,MPI_MAX,SHOT_COMM);
            if(objective_min!=objective_max)
                err(" Exact visco PSV objective consensus failed. ");
        }
        local_decision=finite_min && objective<base_max;
        MPI_Allreduce(&local_decision,&decision_min,1,MPI_INT,MPI_MIN,SHOT_COMM);
        MPI_Allreduce(&local_decision,&decision_max,1,MPI_INT,MPI_MAX,SHOT_COMM);
        if(decision_min!=decision_max)
            err(" Exact visco PSV acceptance-decision consensus failed. ");
        trial_decision[trial_count]=decision_min;
        trial_count++;
        if(decision_min){
            accepted=1;
            accepted_objective=objective;
            break;
        }
        alpha/=divisor;
    }
    if(!accepted){
        restore_base(material,base,&mapping);
        err(" Exact visco PSV line search found no strictly descending valid Trial. ");
    }

    /* Commit collectively: physical Q stays authoritative; Tau, all five
     * model halos and all derived coefficients are rebuilt after acceptance. */
    refresh_material(material,&mapping);
    for(k=0;k<EXACT_FIELDS;k++) for(i=1;i<=NX;i++) for(j=1;j<=NY;j++){
        double difference=(double)base[k][j][i]-
            (k==EXACT_VP?material->ppi[j][i]:k==EXACT_VS?material->pu[j][i]:
             k==EXACT_RHO?material->prho[j][i]:k==EXACT_QP?material->pqp[j][i]:
             material->pqs[j][i]);
        local_norm[k]+=difference*difference;
    }
    MPI_Allreduce(local_norm,update_norm,EXACT_FIELDS,MPI_DOUBLE,MPI_SUM,SHOT_COMM);
    for(k=0;k<EXACT_FIELDS;k++) update_norm[k]=sqrt(update_norm[k]);
    MPI_Allreduce(&forward_count,&forward_min,1,MPI_INT,MPI_MIN,SHOT_COMM);
    MPI_Allreduce(&forward_count,&forward_max,1,MPI_INT,MPI_MAX,SHOT_COMM);
    if(forward_min!=forward_max)
        err(" Exact visco PSV trial-forward-count consensus failed. ");

    snprintf(prefix,sizeof(prefix),"%s_stage_%d_it_%d",INV_MODELFILE,
             request->stage,request->iter);
    persist_model(prefix,material);
    if(MYID_SHOT==0){
        snprintf(report_path,sizeof(report_path),"%s.active_fwi.json",JACOBIAN);
        report=fopen(report_path,"w");
        if(!report) err(" Could not write exact visco PSV lifecycle report. ");
        fprintf(report,"{\n  \"decomposition\": [%d, %d],\n",NPROCX,NPROCY);
        fprintf(report,"  \"collective_communicator\": \"SHOT_COMM\",\n");
        fprintf(report,"  \"base_objective\": %.17g,\n",base_max);
        write_named_values(report,"model_maxima",model_max);
        write_named_ranks(report,"model_max_owner_ranks",model_owner);
        write_named_values(report,"gradient_maxima",gradient_max);
        write_named_ranks(report,"gradient_max_owner_ranks",gradient_owner);
        fprintf(report,"  \"trials\": [\n");
        for(k=0;k<trial_count;k++)
            fprintf(report,"    {\"alpha\": %.17g, \"valid\": %s, \"objective\": %.17g, \"accepted\": %s, \"forward_executed\": %s, \"invalid_rank_count\": %d}%s\n",
                    trial_alpha[k],trial_valid[k]?"true":"false",trial_objective[k],
                    trial_decision[k]?"true":"false",trial_forward[k]?"true":"false",
                    invalid_rank_count[k],k+1<trial_count?",":"");
        fprintf(report,"  ],\n  \"accepted_alpha\": %.17g,\n  \"accepted_objective\": %.17g,\n",alpha,accepted_objective);
        write_named_values(report,"update_norms",update_norm);
        fprintf(report,"  \"alphas_attempted\": %d,\n",trial_count);
        fprintf(report,"  \"trial_forwards_min\": %d,\n  \"trial_forwards_max\": %d,\n",forward_min,forward_max);
        fprintf(report,"  \"decision_consensus\": true,\n");
        fprintf(report,"  \"line_search_control_identical\": true,\n");
        fprintf(report,"  \"trial_forwards_forward_only\": true,\n");
        fprintf(report,"  \"global_field_replication\": false,\n");
        fprintf(report,"  \"persisted_prefix\": \"%s\"\n}\n",prefix);
        fclose(report);
        printf("Exact visco PSV accepted alpha %.8e: J %.12e -> %.12e\n",
               alpha,base_max,accepted_objective);
    }
    free(trial_alpha); free(trial_objective); free(trial_valid);
    free(trial_decision); free(trial_forward); free(invalid_rank_count);
    for(k=0;k<EXACT_FIELDS;k++){
        free_matrix(base[k],1,NY,1,NX);
        free_matrix(direction[k],1,NY,1,NX);
    }
    return accepted_objective;
}
