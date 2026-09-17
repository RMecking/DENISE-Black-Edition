/* Active five-parameter viscoelastic P/SV steepest-descent lifecycle. */
#include "fd.h"

extern int NX, NY, MYID, POS[3], STEPMAX;
extern int INV_VP_ITER, INV_VS_ITER, INV_RHO_ITER, INV_QS_ITER;
extern int Q_PARAMETERIZATION_MODE;
extern float EPS_SCALE, SCALEFAC;
extern float VPLOWERLIM, VPUPPERLIM, VSLOWERLIM, VSUPPERLIM;
extern float RHOLOWERLIM, RHOUPPERLIM, QSLOWERLIM, QSUPPERLIM;
extern float *FL, Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF;
extern char INV_MODELFILE[STRING_SIZE], JACOBIAN[STRING_SIZE];

enum { EXACT_VP, EXACT_VS, EXACT_RHO, EXACT_QP, EXACT_QS, EXACT_FIELDS };

static void copy_field(float **from, float **to) {
    int i,j;
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++) to[j][i]=from[j][i];
}

static double maximum_abs(float **field) {
    int i,j;
    double result=0.0;
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++)
        if(fabs((double)field[j][i])>result) result=fabs((double)field[j][i]);
    return result;
}

static void rebuild_tau(struct matPSV *material,
                        const struct q_tau_mapping *mapping) {
    int i,j;
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++) {
        if(!(material->pqp[j][i]>0.0f) || !isfinite(material->pqp[j][i]) ||
           !(material->pqs[j][i]>0.0f) || !isfinite(material->pqs[j][i])) {
            fprintf(stderr,"Invalid exact visco Q at (%d,%d): Qp=%g Qs=%g\n",
                    i,j,(double)material->pqp[j][i],(double)material->pqs[j][i]);
            err(" Exact visco PSV Trial contains invalid physical Q. ");
        }
        material->ptaup[j][i]=q_to_tau(material->pqp[j][i],mapping);
        material->ptaus[j][i]=q_to_tau(material->pqs[j][i],mapping);
    }
}

static int trial_from_base(struct matPSV *material, float ***base,
                           float ***direction, double alpha,
                           const struct q_tau_mapping *mapping) {
    int i,j,k;
    double value[EXACT_FIELDS];
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++) {
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
    for(i=1;i<=NX;i++) for(j=1;j<=NY;j++) {
        material->ppi[j][i]=(float)(base[EXACT_VP][j][i]-alpha*direction[EXACT_VP][j][i]);
        material->pu[j][i]=(float)(base[EXACT_VS][j][i]-alpha*direction[EXACT_VS][j][i]);
        material->prho[j][i]=(float)(base[EXACT_RHO][j][i]-alpha*direction[EXACT_RHO][j][i]);
        material->pqp[j][i]=(float)(base[EXACT_QP][j][i]-alpha*direction[EXACT_QP][j][i]);
        material->pqs[j][i]=(float)(base[EXACT_QS][j][i]-alpha*direction[EXACT_QS][j][i]);
    }
    rebuild_tau(material,mapping);
    return 1;
}

static void restore_base(struct matPSV *material, float ***base,
                         const struct q_tau_mapping *mapping) {
    copy_field(base[EXACT_VP],material->ppi);
    copy_field(base[EXACT_VS],material->pu);
    copy_field(base[EXACT_RHO],material->prho);
    copy_field(base[EXACT_QP],material->pqp);
    copy_field(base[EXACT_QS],material->pqs);
    rebuild_tau(material,mapping);
}

static void persist_one(const char *prefix, const char *suffix, float **field) {
    char filename[STRING_SIZE*2], local[STRING_SIZE*2];
    snprintf(filename,sizeof(filename),"%s.%s",prefix,suffix);
    writemod(filename,field,3);
    MPI_Barrier(MPI_COMM_WORLD);
    if(MYID==0) mergemod(filename,3);
    MPI_Barrier(MPI_COMM_WORLD);
    snprintf(local,sizeof(local),"%s.%i.%i",filename,POS[1],POS[2]);
    remove(local);
}

static void persist_model(const char *prefix, struct matPSV *material) {
    persist_one(prefix,"vp",material->ppi);
    persist_one(prefix,"vs",material->pu);
    persist_one(prefix,"rho",material->prho);
    persist_one(prefix,"qp",material->pqp);
    persist_one(prefix,"qs",material->pqs);
}

double visco_psv_exact_active_step(
        const struct visco_psv_exact_fwi_request *request) {
    struct matPSV *material=request->material;
    struct fwiPSV *fwi=request->fwi;
    struct q_tau_mapping mapping;
    float **base[EXACT_FIELDS], **direction[EXACT_FIELDS], **gradient[EXACT_FIELDS];
    double model_max[EXACT_FIELDS], gradient_max[EXACT_FIELDS];
    double alpha, divisor, objective, accepted_objective=-1.0;
    double update_norm[EXACT_FIELDS]={0.0,0.0,0.0,0.0,0.0};
    double *trial_alpha, *trial_objective;
    int active[EXACT_FIELDS], i,j,k,trial_count=0,accepted=0,max_trials;
    char report_path[STRING_SIZE*2], prefix[STRING_SIZE*2];
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
    for(k=0;k<EXACT_FIELDS;k++) {
        base[k]=matrix(1,NY,1,NX);
        direction[k]=matrix(1,NY,1,NX);
    }
    copy_field(material->ppi,base[EXACT_VP]);
    copy_field(material->pu,base[EXACT_VS]);
    copy_field(material->prho,base[EXACT_RHO]);
    copy_field(material->pqp,base[EXACT_QP]);
    copy_field(material->pqs,base[EXACT_QS]);
    /* Preserve the production maximum-relative-change scaling independently
       for each physical parameter.  Every multiplier is strictly positive. */
    for(k=0;k<EXACT_FIELDS;k++) {
        model_max[k]=maximum_abs(base[k]);
        gradient_max[k]=maximum_abs(gradient[k]);
        if(active[k] && (!(gradient_max[k]>0.0) || !isfinite(gradient_max[k])))
            err(" Exact visco PSV active physical gradient is zero or non-finite. ");
        for(i=1;i<=NX;i++) for(j=1;j<=NY;j++)
            direction[k][j][i]=active[k]
                ? (float)(gradient[k][j][i]*model_max[k]/gradient_max[k]) : 0.0f;
    }
    max_trials=STEPMAX>0?STEPMAX+1:8;
    trial_alpha=calloc((size_t)max_trials,sizeof(double));
    trial_objective=calloc((size_t)max_trials,sizeof(double));
    if(!trial_alpha || !trial_objective) err(" Exact visco PSV line-search allocation failed. ");
    alpha=EPS_SCALE>0.0?EPS_SCALE:0.01;
    divisor=SCALEFAC>1.0?SCALEFAC:2.0;
    for(k=0;k<max_trials;k++) {
        trial_alpha[trial_count]=alpha;
        if(!trial_from_base(material,base,direction,alpha,&mapping)) {
            trial_objective[trial_count]=-1.0;
            trial_count++;
            alpha/=divisor;
            continue;
        }
        objective=obj_psv(request->wave,request->pml,material,fwi,request->mpi,
                          request->seis,request->data,request->acquisition,
                          request->hc,request->nsrc,request->nsrc_loc,
                          request->nsrc_glob,request->ntr,request->ntr_glob,
                          request->ns,2,request->iter,request->Ws,request->Wr,
                          request->hin,request->DTINV_help,(float)alpha,
                          request->req_send,request->req_rec);
        trial_objective[trial_count++]=objective;
        if(isfinite(objective) && objective<request->base_objective) {
            accepted=1;
            accepted_objective=objective;
            break;
        }
        alpha/=divisor;
    }
    if(!accepted) {
        restore_base(material,base,&mapping);
        err(" Exact visco PSV line search found no strictly descending valid Trial. ");
    }
    for(k=0;k<EXACT_FIELDS;k++) for(i=1;i<=NX;i++) for(j=1;j<=NY;j++) {
        double difference=(double)base[k][j][i]-
            (k==EXACT_VP?material->ppi[j][i]:k==EXACT_VS?material->pu[j][i]:
             k==EXACT_RHO?material->prho[j][i]:k==EXACT_QP?material->pqp[j][i]:
             material->pqs[j][i]);
        update_norm[k]+=difference*difference;
    }
    for(k=0;k<EXACT_FIELDS;k++) update_norm[k]=sqrt(update_norm[k]);
    snprintf(prefix,sizeof(prefix),"%s_stage_%d_it_%d",INV_MODELFILE,
             request->stage,request->iter);
    persist_model(prefix,material);
    if(MYID==0) {
        snprintf(report_path,sizeof(report_path),"%s.active_fwi.json",JACOBIAN);
        report=fopen(report_path,"w");
        if(!report) err(" Could not write exact visco PSV lifecycle report. ");
        fprintf(report,"{\n  \"base_objective\": %.17g,\n  \"trials\": [\n",request->base_objective);
        for(k=0;k<trial_count;k++)
            fprintf(report,"    {\"alpha\": %.17g, \"objective\": %.17g}%s\n",
                    trial_alpha[k],trial_objective[k],k+1<trial_count?",":"");
        fprintf(report,"  ],\n  \"accepted_alpha\": %.17g,\n  \"accepted_objective\": %.17g,\n",alpha,accepted_objective);
        fprintf(report,"  \"update_norms\": {\"vp\": %.17g, \"vs\": %.17g, \"rho\": %.17g, \"qp\": %.17g, \"qs\": %.17g},\n",
                update_norm[0],update_norm[1],update_norm[2],update_norm[3],update_norm[4]);
        fprintf(report,"  \"persisted_prefix\": \"%s\"\n}\n",prefix);
        fclose(report);
        printf("Exact visco PSV accepted alpha %.8e: J %.12e -> %.12e\n",
               alpha,request->base_objective,accepted_objective);
    }
    free(trial_alpha); free(trial_objective);
    for(k=0;k<EXACT_FIELDS;k++) {
        free_matrix(base[k],1,NY,1,NX);
        free_matrix(direction[k],1,NY,1,NX);
    }
    return accepted_objective;
}
