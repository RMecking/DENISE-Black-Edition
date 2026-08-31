#define main m63c_locked_c5a_harness_main
#include "m63c_full_state_step_harness.c"
#undef main
#define M63C_FULL_STATE_ADJOINT_EMBEDDED
#include "../../src/SH/visco_sh_full_state_adjoint_step.c"
#undef M63C_FULL_STATE_ADJOINT_EMBEDDED

struct dfield { double **v, **rows, *data; int nrows, ncols; };

static struct dfield dfield_new(int rows, int cols) {
    struct dfield f; int j;
    f.nrows=rows;f.ncols=cols;
    f.rows=(double **)calloc((size_t)rows,sizeof(double *));
    f.data=(double *)calloc((size_t)rows*cols,sizeof(double));
    if(!f.rows||!f.data)MPI_Abort(MPI_COMM_WORLD,141);
    f.v=f.rows-1;for(j=1;j<=rows;++j)f.v[j]=f.data+(j-1)*cols-1;
    return f;
}
static void dfield_free(struct dfield *f){free(f->data);free(f->rows);}
static double max_state_difference(const struct owned_state *a,const struct owned_state *b,int fw){
    double m=0.0;int k,n;
#define CMP_FIELD(x) n=a->x.nrows*a->x.ncols;for(k=0;k<n;++k)m=fmax(m,fabs((double)a->x.data[k]-b->x.data[k]))
    CMP_FIELD(vz);CMP_FIELD(sxz);CMP_FIELD(syz);
    n=a->r.nrows*a->r.ncols*(a->r.mechanisms+1);for(k=0;k<n;++k){m=fmax(m,fabs((double)a->r.data[k]-b->r.data[k]));m=fmax(m,fabs((double)a->q.data[k]-b->q.data[k]));}
    if(fw){CMP_FIELD(psi_sxz_x);CMP_FIELD(psi_syz_y);CMP_FIELD(psi_vzx);CMP_FIELD(psi_vzy);}
#undef CMP_FIELD
    return m;
}
static double field_checksum(const struct field *f){double s=0.0;int k,n=f->nrows*f->ncols;for(k=0;k<n;++k)s+=(k+1)*(double)f->data[k];return s;}
static double observable_checksum(const struct visco_sh_material_observable_step *step){double s=0.0;int i,j,k=0;for(j=1;j<=NY;++j)for(i=1;i<=NX;++i){++k;s+=k*((double)step->qsum[j][i]+2.0*step->strain_x[j][i]+3.0*step->strain_y[j][i]);}return s;}
static double gradient_difference(const struct visco_sh_native_material_gradient_fields *a,const struct visco_sh_native_material_gradient_fields *b){double m=0.0;int i,j;for(j=1;j<=NY;++j)for(i=1;i<=NX;++i){m=fmax(m,fabs(a->g_rhoi[j][i]-b->g_rhoi[j][i]));m=fmax(m,fabs(a->g_mu_x[j][i]-b->g_mu_x[j][i]));m=fmax(m,fabs(a->g_mu_y[j][i]-b->g_mu_y[j][i]));m=fmax(m,fabs(a->g_tau_x[j][i]-b->g_tau_x[j][i]));m=fmax(m,fabs(a->g_tau_y[j][i]-b->g_tau_y[j][i]));}return m;}
struct material_hook_snapshot {
    struct dfield sxz, syz, vz;
    double *r, *q;
    int mechanisms;
};
static struct material_hook_snapshot material_hook_snapshot_new(int mechanisms){
    struct material_hook_snapshot s;
    s.sxz=dfield_new(NY,NX);s.syz=dfield_new(NY,NX);s.vz=dfield_new(NY,NX);
    s.mechanisms=mechanisms;s.r=(double *)calloc((size_t)NY*NX*mechanisms,sizeof(double));s.q=(double *)calloc((size_t)NY*NX*mechanisms,sizeof(double));
    if(!s.r||!s.q)MPI_Abort(MPI_COMM_WORLD,150);return s;
}
static void material_hook_snapshot_free(struct material_hook_snapshot *s){dfield_free(&s->sxz);dfield_free(&s->syz);dfield_free(&s->vz);free(s->r);free(s->q);}
static size_t material_hook_index(int i,int j,int l,int mechanisms){return ((size_t)(j-1)*NX+(i-1))*mechanisms+(l-1);}
static void snapshot_stress_hook(struct material_hook_snapshot *s,const struct visco_sh_full_state *work){int i,j,l;for(j=1;j<=NY;++j)for(i=1;i<=NX;++i){s->sxz.v[j][i]=work->sxz[j][i];s->syz.v[j][i]=work->syz[j][i];for(l=1;l<=s->mechanisms;++l){size_t k=material_hook_index(i,j,l,s->mechanisms);s->r[k]=work->r[j][i][l];s->q[k]=work->q[j][i][l];}}}
static void snapshot_velocity_hook(struct material_hook_snapshot *s,const struct visco_sh_full_state *work){int i,j;for(j=1;j<=NY;++j)for(i=1;i<=NX;++i)s->vz.v[j][i]=work->vz[j][i];}
static int independent_one_shot_c7b(const struct visco_sh_full_step_config *cfg,const struct visco_sh_material_adjoint_step_context *material,const struct material_hook_snapshot *stress,const struct material_hook_snapshot *velocity,int swap_strains,struct visco_sh_native_material_gradient_fields *output){
    struct visco_sh_material_timestep_vjp_input input;struct visco_sh_material_timestep_vjp_output result;
    double *bar_r,*bar_q,*eta_x,*eta_y,*b_x,*b_y,*a_x,*a_y,*c_x,*c_y;int i,j,l,status=-1;size_t bytes=(size_t)cfg->mechanisms*sizeof(double);
    bar_r=(double *)calloc(1,bytes);bar_q=(double *)calloc(1,bytes);eta_x=(double *)calloc(1,bytes);eta_y=(double *)calloc(1,bytes);b_x=(double *)calloc(1,bytes);b_y=(double *)calloc(1,bytes);a_x=(double *)calloc(1,bytes);a_y=(double *)calloc(1,bytes);c_x=(double *)calloc(1,bytes);c_y=(double *)calloc(1,bytes);
    if(!bar_r||!bar_q||!eta_x||!eta_y||!b_x||!b_y||!a_x||!a_y||!c_x||!c_y)goto cleanup;
    for(l=0;l<cfg->mechanisms;++l){eta_x[l]=material->eta_x[l+1];eta_y[l]=material->eta_y[l+1];b_x[l]=cfg->bip[l+1];b_y[l]=cfg->bjm[l+1];a_x[l]=(double)cfg->bip[l+1]*cfg->cip[l+1];a_y[l]=(double)cfg->bjm[l+1]*cfg->cjm[l+1];}
    memset(&input,0,sizeof(input));input.mechanisms=cfg->mechanisms;input.dt=cfg->dt;input.dh=cfg->dh;input.reference_sum=material->reference_sum;input.eta_x=eta_x;input.b_x=b_x;input.eta_y=eta_y;input.b_y=b_y;input.forward_a_x=a_x;input.forward_a_y=a_y;input.forward_c_x=c_x;input.forward_c_y=c_y;input.bar_r_next=bar_r;input.bar_q_next=bar_q;
    for(j=1;j<=cfg->ny;++j)for(i=1;i<=cfg->nx;++i){input.qsum=material->observable->qsum[j][i];input.strain_x=swap_strains?material->observable->strain_y[j][i]:material->observable->strain_x[j][i];input.strain_y=swap_strains?material->observable->strain_x[j][i]:material->observable->strain_y[j][i];input.bar_v_post_velocity=velocity->vz.v[j][i];input.bar_sxz_next=stress->sxz.v[j][i];input.bar_syz_next=stress->syz.v[j][i];input.mu_x=material->mu_x[j][i];input.tau_x=material->tau_x[j][i];input.mu_y=material->mu_y[j][i];input.tau_y=material->tau_y[j][i];input.forward_f_x=cfg->fipjp[j][i];input.forward_f_y=cfg->f[j][i];for(l=0;l<cfg->mechanisms;++l){size_t k=material_hook_index(i,j,l+1,cfg->mechanisms);bar_r[l]=stress->r[k];bar_q[l]=stress->q[k];c_x[l]=-(double)cfg->bip[l+1]*cfg->dip[j][i][l+1];c_y[l]=-(double)cfg->bjm[l+1]*cfg->d[j][i][l+1];}status=visco_sh_material_timestep_vjp(&input,&result);if(status!=0)goto cleanup;output->g_rhoi[j][i]=result.g_rhoi;output->g_mu_x[j][i]=result.g_mu_x;output->g_mu_y[j][i]=result.g_mu_y;output->g_tau_x[j][i]=result.g_tau_x;output->g_tau_y[j][i]=result.g_tau_y;}
    status=0;
cleanup:free(bar_r);free(bar_q);free(eta_x);free(eta_y);free(b_x);free(b_y);free(a_x);free(a_y);free(c_x);free(c_y);return status;
}
static int explicit_hook_reference(const struct visco_sh_full_step_config *cfg,struct visco_sh_full_state *work,struct visco_sh_full_state *prev,double *bar_signal,const struct visco_sh_material_adjoint_step_context *material,struct visco_sh_native_material_gradient_fields *correct,struct visco_sh_native_material_gradient_fields *wrong_stress,struct visco_sh_native_material_gradient_fields *wrong_velocity,struct visco_sh_native_material_gradient_fields *mutant){
    int h=cfg->fdorder/2,row_min=-h,row_max=cfg->ny+h+1,col_min=1-h,col_max=cfg->nx+h,status,source;
    struct material_hook_snapshot correct_hook=material_hook_snapshot_new(cfg->mechanisms),early_stress=material_hook_snapshot_new(cfg->mechanisms),early_velocity=material_hook_snapshot_new(cfg->mechanisms);
    zero_field(prev->vz,row_min,row_max,col_min,col_max);zero_field(prev->sxz,row_min,row_max,col_min,col_max);zero_field(prev->syz,row_min,row_max,col_min,col_max);copy_memory(prev->r,work->r,row_min,row_max,col_min,col_max,cfg->mechanisms);copy_memory(prev->q,work->q,row_min,row_max,col_min,col_max,cfg->mechanisms);if(cfg->fw>0){copy_field(prev->psi_sxz_x,work->psi_sxz_x,1,cfg->ny,1,2*cfg->fw);copy_field(prev->psi_syz_y,work->psi_syz_y,1,2*cfg->fw,1,cfg->nx);copy_field(prev->psi_vzx,work->psi_vzx,1,cfg->ny,1,2*cfg->fw);copy_field(prev->psi_vzy,work->psi_vzy,1,2*cfg->fw,1,cfg->nx);}for(source=0;source<cfg->nsrc;++source)bar_signal[source]=0.0;
    status=receiver_transpose(cfg,work,row_min,row_max,col_min,col_max);if(status!=0)goto cleanup;snapshot_stress_hook(&early_stress,work);status=exchange_s_SH_adjoint(work->sxz,work->syz,cfg->nx,cfg->ny,cfg->fdorder,cfg->boundary,cfg->pos,cfg->nproc_x,cfg->nproc_y,cfg->index,cfg->comm);if(status!=MPI_SUCCESS)goto cleanup;if(cfg->free_surface&&(cfg->pos[2]==0))surface_elastic_SH_stress_adjoint(work->syz,cfg->nx,h);snapshot_stress_hook(&correct_hook,work);copy_field(prev->sxz,work->sxz,row_min,row_max,col_min,col_max);copy_field(prev->syz,work->syz,row_min,row_max,col_min,col_max);status=reverse_stress_block(cfg,work,prev);if(status!=0)goto cleanup;snapshot_velocity_hook(&early_velocity,work);if(cfg->free_surface&&(cfg->pos[2]==0))surface_elastic_SH_velocity_adjoint(work->vz,cfg->nx,h);status=exchange_v_SH_adjoint(work->vz,cfg->nx,cfg->ny,cfg->fdorder,cfg->boundary,cfg->pos,cfg->nproc_x,cfg->nproc_y,cfg->index,cfg->comm);if(status!=MPI_SUCCESS)goto cleanup;status=source_transpose(cfg,work->vz,prev->vz,row_min,row_max,col_min,col_max,bar_signal);if(status!=0)goto cleanup;snapshot_velocity_hook(&correct_hook,work);
    status=independent_one_shot_c7b(cfg,material,&correct_hook,&correct_hook,0,correct);if(status!=0)goto cleanup;status=independent_one_shot_c7b(cfg,material,&early_stress,&correct_hook,0,wrong_stress);if(status!=0)goto cleanup;status=independent_one_shot_c7b(cfg,material,&correct_hook,&early_velocity,0,wrong_velocity);if(status!=0)goto cleanup;status=independent_one_shot_c7b(cfg,material,&correct_hook,&correct_hook,1,mutant);if(status!=0)goto cleanup;status=reverse_velocity_block(cfg,work,prev);
cleanup:material_hook_snapshot_free(&correct_hook);material_hook_snapshot_free(&early_stress);material_hook_snapshot_free(&early_velocity);return status;
}

int main(int argc,char **argv){
    struct owned_state input,forward_state,dual,fixed_work,material_work,reference_work,fixed_prev,material_prev,reference_prev;
    struct field rhoi,fipjp,f,mu_x,tau_x,mu_y,tau_y,diag[6],dummy;
    struct volume dip,d,pp;
    struct dfield grhoi,gmx,gmy,gtx,gty,rrhoi,rmx,rmy,rtx,rty;
    struct dfield srhoi,smx,smy,stx,sty,vrhoi,vmx,vmy,vtx,vty;
    struct dfield mrhoi,mmx,mmy,mtx,mty;
    struct visco_sh_full_step_config cfg;
    struct visco_sh_material_observable_trajectory trajectory;
    struct visco_sh_native_material_gradient_fields gradients,reference_gradients,wrong_stress_gradients,wrong_velocity_gradients,mutant_gradients;
    struct visco_sh_material_adjoint_step_context context;
    float *hc,*bip,*bjm,*cip,*cjm,*eta_x,*eta_y,*K,*Kh,*ca,*cah,*cb,*cbh;
    float **srcpos,**signals,*src_storage,*signal_storage;
    float **bl,**br,**bt,**bb,*sl,*sr,*st,*sb;
    MPI_Request req[4];FILE *out;char path[4096];
    int rank,size,h,i,j,l,k,source_i,source_j,receiver_i,receiver_j,status,step_id;
    int rec_x[1],rec_y[1],src_x[1],src_y[1],src_type[1];
    double bar_receiver[1],fixed_signal[1],material_signal[1],reference_signal[1],omega,reference_sum=0.0;
    double local_passivity,passivity,input_before,input_after,nonzero=0.0,reference_error;
    double wrong_stress_distance,wrong_velocity_distance,mutant_distance;

    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);MYID=rank;FP=stderr;
    if(argc!=12){if(rank==0)fprintf(stderr,"usage: npx npy boundary fs fd L fw nx ny step outdir\n");MPI_Abort(MPI_COMM_WORLD,142);}
    NPROCX=atoi(argv[1]);NPROCY=atoi(argv[2]);BOUNDARY=atoi(argv[3]);FREE_SURF=atoi(argv[4]);FDORDER=atoi(argv[5]);L=atoi(argv[6]);FW=atoi(argv[7]);NX=atoi(argv[8]);NY=atoi(argv[9]);step_id=atoi(argv[10]);
    if(size!=NPROCX*NPROCY)MPI_Abort(MPI_COMM_WORLD,143);topology(rank);h=FDORDER/2;DT=0.0013f;DH=7.5f;GRAD_FORM=2;ADJ_SIGN=1;MODE=0;QUELLTYPB=1;
    input=state_new(h,FW,L);forward_state=state_new(h,FW,L);dual=state_new(h,FW,L);fixed_work=state_new(h,FW,L);material_work=state_new(h,FW,L);reference_work=state_new(h,FW,L);fixed_prev=state_new(h,FW,L);material_prev=state_new(h,FW,L);reference_prev=state_new(h,FW,L);
    state_initialize(&input,rank,0,FW,L);state_initialize(&dual,rank,1,FW,L);state_copy(&forward_state,&input,FW);state_copy(&fixed_work,&dual,FW);state_copy(&material_work,&dual,FW);state_copy(&reference_work,&dual,FW);
    rhoi=field_new(-h,NY+h+1,1-h,NX+h);fipjp=field_new(-h,NY+h+1,1-h,NX+h);f=field_new(-h,NY+h+1,1-h,NX+h);mu_x=field_new(-h,NY+h+1,1-h,NX+h);tau_x=field_new(-h,NY+h+1,1-h,NX+h);mu_y=field_new(-h,NY+h+1,1-h,NX+h);tau_y=field_new(-h,NY+h+1,1-h,NX+h);dummy=field_new(-h,NY+h+1,1-h,NX+h);
    dip=volume_new(-h,NY+h+1,1-h,NX+h,L);d=volume_new(-h,NY+h+1,1-h,NX+h,L);pp=volume_new(-h,NY+h+1,1-h,NX+h,L);
    hc=(float *)calloc((size_t)h+1,sizeof(float));bip=(float *)calloc((size_t)L+1,sizeof(float));bjm=(float *)calloc((size_t)L+1,sizeof(float));cip=(float *)calloc((size_t)L+1,sizeof(float));cjm=(float *)calloc((size_t)L+1,sizeof(float));eta_x=(float *)calloc((size_t)L+1,sizeof(float));eta_y=(float *)calloc((size_t)L+1,sizeof(float));
    {double values[6]={1.0,-0.041,0.007,-0.0014,0.00031,-0.00007};for(k=1;k<=h;++k)hc[k]=(float)values[k-1];}
    omega=2.0*3.14159265358979323846*6.0;for(l=1;l<=L;++l){double theta=1.0/(2.0*3.14159265358979323846*(4.0+5.0*l));eta_x[l]=(float)(DT/theta);eta_y[l]=eta_x[l];bip[l]=bjm[l]=(float)(1.0/(1.0+0.5*eta_x[l]));cip[l]=cjm[l]=(float)(1.0-0.5*eta_x[l]);reference_sum+=(omega*theta)*(omega*theta)/(1.0+(omega*theta)*(omega*theta));}
    for(j=-h;j<=NY+h+1;++j)for(i=1-h;i<=NX+h;++i){double rx,ry;rhoi.v[j][i]=(float)(0.00043+0.000002*rank+0.0000003*j);mu_x.v[j][i]=(float)(4.7+0.012*i+0.008*j+0.02*rank);mu_y.v[j][i]=(float)(4.2+0.009*i+0.006*j+0.015*rank);tau_x.v[j][i]=(float)(0.035+0.0003*i+0.0002*j);tau_y.v[j][i]=(float)(0.052+0.0002*i+0.00025*j);rx=mu_x.v[j][i]/(1.0+reference_sum*tau_x.v[j][i]);ry=mu_y.v[j][i]/(1.0+reference_sum*tau_y.v[j][i]);fipjp.v[j][i]=(float)(DT*rx*(1.0+L*tau_x.v[j][i]));f.v[j][i]=(float)(DT*ry*(1.0+L*tau_y.v[j][i]));for(l=1;l<=L;++l){dip.v[j][i][l]=(float)(rx*eta_x[l]*tau_x.v[j][i]);d.v[j][i][l]=(float)(ry*eta_y[l]*tau_y.v[j][i]);}}
    K=(float *)calloc((size_t)2*FW+1,sizeof(float));Kh=(float *)calloc((size_t)2*FW+1,sizeof(float));ca=(float *)calloc((size_t)2*FW+1,sizeof(float));cah=(float *)calloc((size_t)2*FW+1,sizeof(float));cb=(float *)calloc((size_t)2*FW+1,sizeof(float));cbh=(float *)calloc((size_t)2*FW+1,sizeof(float));for(k=1;k<=2*FW;++k){K[k]=(float)(1.11+0.017*k);Kh[k]=(float)(1.08+0.013*k);ca[k]=(float)(-0.031-0.001*k);cah[k]=(float)(-0.027-0.0008*k);cb[k]=(float)(0.82+0.006*k);cbh[k]=(float)(0.84+0.005*k);}
    for(k=0;k<6;++k){diag[k]=field_new(-h,NY+h+1,1-h,NX+h);for(j=diag[k].rmin;j<=diag[k].rmax;++j)for(i=diag[k].cmin;i<=diag[k].cmax;++i)diag[k].v[j][i]=7.25f+(float)k;}
    srcpos=buffer_new(8,1,&src_storage);signals=buffer_new(1,1,&signal_storage);source_i=3+rank%2;source_j=4+rank%3;receiver_i=6+rank%3;receiver_j=5+rank%2;srcpos[1][1]=(float)source_i;srcpos[2][1]=(float)source_j;srcpos[8][1]=1.0f;signals[1][1]=(float)(0.021+0.003*rank);
    bl=buffer_new(NY,2*(h+1),&sl);br=buffer_new(NY,2*(h+1),&sr);bt=buffer_new(NX,2*(h+1),&st);bb=buffer_new(NX,2*(h+1),&sb);
    memset(&trajectory,0,sizeof(trajectory));if(visco_sh_material_observable_trajectory_init(&trajectory,NX,NY,1,1,FW,FREE_SURF,BOUNDARY,NPROCX,NPROCY)!=0)MPI_Abort(MPI_COMM_WORLD,144);if(visco_sh_material_observable_begin_step(&trajectory,0)!=0)MPI_Abort(MPI_COMM_WORLD,145);
    update_v_PML_SH(1,NX,1,NY,1,forward_state.view.vz,diag[0].v,diag[1].v,diag[2].v,forward_state.view.sxz,forward_state.view.syz,dummy.v,rhoi.v,srcpos,signals,1,dummy.v,hc,0,0,0,K,ca,cb,Kh,cah,cbh,K,ca,cb,Kh,cah,cbh,forward_state.view.psi_sxz_x,forward_state.view.psi_syz_y);exchange_v_SH(forward_state.view.vz,bl,br,bt,bb,req,req);if(FREE_SURF&&POS[2]==0)surface_elastic_SH_velocity(forward_state.view.vz,NX,h);
    update_s_visc_PML_SH(1,NX,1,NY,forward_state.view.vz,diag[3].v,diag[4].v,forward_state.view.syz,forward_state.view.sxz,dummy.v,dummy.v,dummy.v,hc,0,forward_state.view.r,pp.v,forward_state.view.q,fipjp.v,f.v,dummy.v,bip,bjm,cip,cjm,d.v,pp.v,dip.v,K,ca,cb,Kh,cah,cbh,K,ca,cb,Kh,cah,cbh,forward_state.view.psi_vzx,forward_state.view.psi_vzy,NULL,0);visco_sh_material_observable_end_step();if(FREE_SURF&&POS[2]==0)surface_elastic_SH_stress(forward_state.view.syz,NX,h);exchange_s_SH(forward_state.view.sxz,forward_state.view.syz,bl,br,bt,bb,req,req);
    if(step_id){for(j=1;j<=NY;++j)for(i=1;i<=NX;++i){trajectory.steps[0].qsum[j][i]+=(float)(10.0+rank);trajectory.steps[0].strain_x[j][i]+=(float)(20.0+rank);trajectory.steps[0].strain_y[j][i]-=(float)(30.0+rank);}}
    bar_receiver[0]=-0.17+0.019*rank;rec_x[0]=receiver_i;rec_y[0]=receiver_j;src_x[0]=source_i;src_y[0]=source_j;src_type[0]=1;
    memset(&cfg,0,sizeof(cfg));cfg.nx=NX;cfg.ny=NY;cfg.fdorder=FDORDER;cfg.mechanisms=L;cfg.fw=FW;cfg.free_surface=FREE_SURF;cfg.boundary=BOUNDARY;memcpy(cfg.pos,POS,sizeof(POS));memcpy(cfg.index,INDEX,sizeof(INDEX));cfg.nproc_x=NPROCX;cfg.nproc_y=NPROCY;cfg.dt=DT;cfg.dh=DH;cfg.hc=hc;cfg.rhoi=rhoi.v;cfg.fipjp=fipjp.v;cfg.f=f.v;cfg.bip=bip;cfg.bjm=bjm;cfg.cip=cip;cfg.cjm=cjm;cfg.dip=dip.v;cfg.d=d.v;cfg.K_x=K;cfg.a_x=ca;cfg.b_x=cb;cfg.K_x_half=Kh;cfg.a_x_half=cah;cfg.b_x_half=cbh;cfg.K_y=K;cfg.a_y=ca;cfg.b_y=cb;cfg.K_y_half=Kh;cfg.a_y_half=cah;cfg.b_y_half=cbh;cfg.nrec=1;cfg.rec_x=rec_x;cfg.rec_y=rec_y;cfg.bar_receiver=bar_receiver;cfg.nsrc=1;cfg.src_x=src_x;cfg.src_y=src_y;cfg.source_type=src_type;cfg.comm=MPI_COMM_WORLD;
    grhoi=dfield_new(NY,NX);gmx=dfield_new(NY,NX);gmy=dfield_new(NY,NX);gtx=dfield_new(NY,NX);gty=dfield_new(NY,NX);rrhoi=dfield_new(NY,NX);rmx=dfield_new(NY,NX);rmy=dfield_new(NY,NX);rtx=dfield_new(NY,NX);rty=dfield_new(NY,NX);srhoi=dfield_new(NY,NX);smx=dfield_new(NY,NX);smy=dfield_new(NY,NX);stx=dfield_new(NY,NX);sty=dfield_new(NY,NX);vrhoi=dfield_new(NY,NX);vmx=dfield_new(NY,NX);vmy=dfield_new(NY,NX);vtx=dfield_new(NY,NX);vty=dfield_new(NY,NX);mrhoi=dfield_new(NY,NX);mmx=dfield_new(NY,NX);mmy=dfield_new(NY,NX);mtx=dfield_new(NY,NX);mty=dfield_new(NY,NX);for(k=0;k<NY*NX;++k){grhoi.data[k]=101.0+k;gmx.data[k]=201.0+k;gmy.data[k]=301.0+k;gtx.data[k]=401.0+k;gty.data[k]=501.0+k;}gradients.g_rhoi=grhoi.v;gradients.g_mu_x=gmx.v;gradients.g_mu_y=gmy.v;gradients.g_tau_x=gtx.v;gradients.g_tau_y=gty.v;reference_gradients.g_rhoi=rrhoi.v;reference_gradients.g_mu_x=rmx.v;reference_gradients.g_mu_y=rmy.v;reference_gradients.g_tau_x=rtx.v;reference_gradients.g_tau_y=rty.v;wrong_stress_gradients.g_rhoi=srhoi.v;wrong_stress_gradients.g_mu_x=smx.v;wrong_stress_gradients.g_mu_y=smy.v;wrong_stress_gradients.g_tau_x=stx.v;wrong_stress_gradients.g_tau_y=sty.v;wrong_velocity_gradients.g_rhoi=vrhoi.v;wrong_velocity_gradients.g_mu_x=vmx.v;wrong_velocity_gradients.g_mu_y=vmy.v;wrong_velocity_gradients.g_tau_x=vtx.v;wrong_velocity_gradients.g_tau_y=vty.v;mutant_gradients.g_rhoi=mrhoi.v;mutant_gradients.g_mu_x=mmx.v;mutant_gradients.g_mu_y=mmy.v;mutant_gradients.g_tau_x=mtx.v;mutant_gradients.g_tau_y=mty.v;memset(&context,0,sizeof(context));context.observable=&trajectory.steps[0];context.mu_x=mu_x.v;context.tau_x=tau_x.v;context.mu_y=mu_y.v;context.tau_y=tau_y.v;context.reference_sum=reference_sum;context.eta_x=eta_x;context.eta_y=eta_y;context.native_output=&gradients;
    input_before=field_checksum(&mu_x)+field_checksum(&tau_x)+field_checksum(&mu_y)+field_checksum(&tau_y)+observable_checksum(&trajectory.steps[0]);status=visco_sh_full_state_adjoint_step(&cfg,&fixed_work.view,&fixed_prev.view,fixed_signal);if(status!=0)MPI_Abort(MPI_COMM_WORLD,146);status=visco_sh_full_state_adjoint_step_material(&cfg,&material_work.view,&material_prev.view,material_signal,&context);if(status!=0)MPI_Abort(MPI_COMM_WORLD,147);status=explicit_hook_reference(&cfg,&reference_work.view,&reference_prev.view,reference_signal,&context,&reference_gradients,&wrong_stress_gradients,&wrong_velocity_gradients,&mutant_gradients);if(status!=0)MPI_Abort(MPI_COMM_WORLD,149);input_after=field_checksum(&mu_x)+field_checksum(&tau_x)+field_checksum(&mu_y)+field_checksum(&tau_y)+observable_checksum(&trajectory.steps[0]);
    local_passivity=fmax(max_state_difference(&fixed_prev,&material_prev,FW),max_state_difference(&fixed_work,&material_work,FW));local_passivity=fmax(local_passivity,fabs(fixed_signal[0]-material_signal[0]));reference_error=fmax(gradient_difference(&gradients,&reference_gradients),fmax(max_state_difference(&material_prev,&reference_prev,FW),max_state_difference(&material_work,&reference_work,FW)));reference_error=fmax(reference_error,fabs(material_signal[0]-reference_signal[0]));wrong_stress_distance=gradient_difference(&reference_gradients,&wrong_stress_gradients);wrong_velocity_distance=gradient_difference(&reference_gradients,&wrong_velocity_gradients);mutant_distance=gradient_difference(&reference_gradients,&mutant_gradients);MPI_Allreduce(&local_passivity,&passivity,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);{double tmp=reference_error;MPI_Allreduce(&tmp,&reference_error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);tmp=wrong_stress_distance;MPI_Allreduce(&tmp,&wrong_stress_distance,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);tmp=wrong_velocity_distance;MPI_Allreduce(&tmp,&wrong_velocity_distance,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);tmp=mutant_distance;MPI_Allreduce(&tmp,&mutant_distance,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);}
    for(k=0;k<NY*NX;++k)nonzero=fmax(nonzero,fabs(grhoi.data[k])+fabs(gmx.data[k])+fabs(gmy.data[k])+fabs(gtx.data[k])+fabs(gty.data[k]));snprintf(path,sizeof(path),"%s/rank_%d.bin",argv[11],rank);out=fopen(path,"wb");if(!out)MPI_Abort(MPI_COMM_WORLD,148);fwrite(grhoi.data,sizeof(double),(size_t)NY*NX,out);fwrite(gmx.data,sizeof(double),(size_t)NY*NX,out);fwrite(gmy.data,sizeof(double),(size_t)NY*NX,out);fwrite(gtx.data,sizeof(double),(size_t)NY*NX,out);fwrite(gty.data,sizeof(double),(size_t)NY*NX,out);fclose(out);if(rank==0)printf("{\"passivity_max\":%.17g,\"reference_error\":%.17g,\"wrong_stress_hook_distance\":%.17g,\"wrong_velocity_hook_distance\":%.17g,\"wiring_mutant_distance\":%.17g,\"input_change\":%.17g,\"nonzero\":%.17g}\n",passivity,reference_error,wrong_stress_distance,wrong_velocity_distance,mutant_distance,fabs(input_after-input_before),nonzero);
    dfield_free(&grhoi);dfield_free(&gmx);dfield_free(&gmy);dfield_free(&gtx);dfield_free(&gty);visco_sh_material_observable_trajectory_release(&trajectory);MPI_Finalize();return 0;
}
