#define _POSIX_C_SOURCE 200809L

#include "fd.h"
#include "denise_cuda_psv_dispatch.h"
#include "globvar.h"

#include <stdint.h>
#include <time.h>

#define DEFAULT_NX 96
#define DEFAULT_NY 80
#define DEFAULT_NT 500
#define DEFAULT_NTR 5
#define TEST_FW 8
#define MIB ((size_t)1024*(size_t)1024)

struct snapshot {
    float *field[16];
    size_t count[16];
    float *trace_vx;
    float *trace_vy;
    size_t trace_count;
};

struct metric {
    double max_abs,max_rel,sum_sq;
    size_t count,max_index;
};

struct distribution {
    double minimum,q1,median,q3,maximum,iqr;
};

static int fail(const char *message) {
    fprintf(stderr,"CUDA PSV B2A harness failure: %s\n",message);
    return 1;
}

static int compare_double(const void *left,const void *right) {
    double a=*(const double*)left,b=*(const double*)right;
    return (a>b)-(a<b);
}

static double quantile(const double *sorted,int count,double probability) {
    double position=(double)(count-1)*probability;
    int lower=(int)floor(position),upper=(int)ceil(position);
    double fraction=position-(double)lower;
    return sorted[lower]+fraction*(sorted[upper]-sorted[lower]);
}

static struct distribution summarize(const double *values,int count) {
    struct distribution result;
    double *sorted=(double*)malloc((size_t)count*sizeof(double));
    memcpy(sorted,values,(size_t)count*sizeof(double));
    qsort(sorted,(size_t)count,sizeof(double),compare_double);
    result.minimum=sorted[0]; result.maximum=sorted[count-1];
    result.q1=quantile(sorted,count,0.25); result.median=quantile(sorted,count,0.5);
    result.q3=quantile(sorted,count,0.75); result.iqr=result.q3-result.q1;
    free(sorted); return result;
}

static void print_samples(const char *name,const double *values,int count) {
    int k;
    printf("BENCHMARK_RAW metric=%s values=",name);
    for(k=0;k<count;++k) printf("%s%.6f",k?",":"",values[k]);
    printf("\n");
}

static double monotonic_ms(void) {
    struct timespec now;
    if(clock_gettime(CLOCK_MONOTONIC,&now)!=0) return -1.0;
    return (double)now.tv_sec*1000.0+(double)now.tv_nsec/1000000.0;
}

static void configure_globals(int nx,int ny,int nt) {
    NX=NXG=nx; NY=NYG=ny; NT=nt; FW=TEST_FW; FDORDER=4; L=1;
    NPROCX=NPROCY=1; NP=NPROC=NCOLORS=1; MYID=MYID_SHOT=0;
    SHOT_COMM=MPI_COMM_WORLD; DOMAIN_COMM=MPI_COMM_WORLD;
    POS[1]=POS[2]=0; INDEX[1]=INDEX[2]=INDEX[3]=INDEX[4]=0;
    BOUNDARY=FREE_SURF=MODE=SNAP=INV_STF=0;
    QUELLTYP=QUELLTYPB=QUELLART=SEISMO=NDT=1;
    GRAD_FORM=0; INVMAT1=1; EPRECOND=0; IDXI=IDYI=DTINV=1;
    NTDTINV=nt; NXNYI=nx*ny; DT=0.0005f; DH=10.0f;
    TSNAP1=TSNAP2=TSNAPINC=1.0f; DAMPING=2800.0f; FPML=20.0f;
    npower=2.0f; k_max_PML=1.0f; FP=stderr;
    snprintf(JACOBIAN,STRING_SIZE,"/tmp/m8e1b2a_harness");
}

static void initialize_material(struct matPSV *m) {
    int i,j;
    int lo=-2,hi_i=NX+3,hi_j=NY+3;
    for(j=lo;j<=hi_j;++j) for(i=lo;i<=hi_i;++i) {
        int ci=i<1?1:(i>NX?NX:i),cj=j<1?1:(j>NY?NY:j);
        float rho=1980.0f+0.55f*(float)ci+0.35f*(float)cj;
        float vs=1450.0f+0.45f*(float)ci-0.18f*(float)cj;
        float vp=2675.0f+0.25f*(float)ci+0.16f*(float)cj;
        m->prho[j][i]=rho; m->pu[j][i]=vs; m->ppi[j][i]=vp;
        m->ptaus[j][i]=0.018f+0.000008f*(float)(ci+cj);
        m->ptaup[j][i]=0.012f+0.000006f*(float)(2*ci+cj);
    }
    av_mue(m->pu,m->puipjp,m->prho);
    av_rho(m->prho,m->prip,m->prjp);
    av_tau(m->ptaus,m->ptausipjp);
    FL=vector(1,1); FL[1]=12.0f;
    m->peta[1]=2.0f*(float)PI*FL[1]*DT;
    prepare_update_s_visc_PSV(m->etajm,m->etaip,m->peta,m->fipjp,
        m->pu,m->puipjp,m->ppi,m->prho,m->ptaus,m->ptaup,
        m->ptausipjp,m->f,m->g,m->bip,m->bjm,m->cip,m->cjm,
        m->dip,m->d,m->e);
}

static void initialize_cpml(struct wavePSV_PML *pml) {
    PML_pro(pml->d_x,pml->K_x,pml->alpha_prime_x,pml->a_x,pml->b_x,
        pml->d_x_half,pml->K_x_half,pml->alpha_prime_x_half,
        pml->a_x_half,pml->b_x_half,pml->d_y,pml->K_y,
        pml->alpha_prime_y,pml->a_y,pml->b_y,pml->d_y_half,
        pml->K_y_half,pml->alpha_prime_y_half,pml->a_y_half,pml->b_y_half);
}

static void initialize_acquisition(struct acq *a,struct seisPSV *seis,
                                   int ntr) {
    int r,n;
    int source_i=NX/2,source_j=NY/2;
    double center=fmin(45.0,fmax(8.0,(double)NT/6.0));
    double width=fmax(3.0,center/3.5);
    memset(a,0,sizeof(*a)); memset(seis,0,sizeof(*seis));
    a->srcpos_loc=fmatrix(1,8,1,1);
    a->signals=matrix(1,1,1,NT);
    a->recpos_loc=imatrix(1,3,1,ntr);
    a->srcpos_loc[1][1]=(float)source_i;
    a->srcpos_loc[2][1]=(float)source_j;
    a->srcpos_loc[8][1]=1.0f;
    for(n=1;n<=NT;++n) {
        double x=((double)n-center)/width;
        a->signals[1][n]=(float)(0.014*(1.0-2.0*x*x)*exp(-x*x));
    }
    for(r=1;r<=ntr;++r) {
        int span=NX-2*FW-4;
        int offset=ntr==1?0:(r-1)*span/(ntr-1)-span/2;
        a->recpos_loc[1][r]=source_i+offset;
        a->recpos_loc[2][r]=source_j+((r%3)-1)*3;
    }
    alloc_seisPSV(ntr,NT,seis);
}

static void free_material(struct matPSV *m) {
    int lo=-2,hi_i=NX+3,hi_j=NY+3;
#define FM(x) free_matrix(m->x,lo,hi_j,lo,hi_i)
#define FT(x) free_f3tensor(m->x,lo,hi_j,lo,hi_i,1,1)
    FM(prho); FM(prip); FM(prjp); FM(ppi); FM(pu); FM(puipjp);
    FM(pqp); FM(pqs); FT(dip); FT(d); FT(e); FM(ptaus); FM(ptausipjp);
    FM(ptaup); FM(fipjp); FM(f); FM(g);
#undef FM
#undef FT
    free_vector(m->peta,1,1); free_vector(m->etaip,1,1);
    free_vector(m->etajm,1,1); free_vector(m->bip,1,1);
    free_vector(m->bjm,1,1); free_vector(m->cip,1,1); free_vector(m->cjm,1,1);
}

static void free_mpi(struct mpiPSV *m) {
    int fdo3=2*(FDORDER/2+1);
    free_matrix(m->bufferlef_to_rig,1,NY,1,fdo3);
    free_matrix(m->bufferrig_to_lef,1,NY,1,fdo3);
    free_matrix(m->buffertop_to_bot,1,NX,1,fdo3);
    free_matrix(m->bufferbot_to_top,1,NX,1,fdo3);
}

static int snapshot_alloc(struct snapshot *s,int ntr) {
    size_t full=(size_t)(NY+6)*(size_t)(NX+6);
    size_t x=(size_t)NY*(size_t)(2*FW),y=(size_t)(2*FW)*(size_t)NX;
    int k;
    memset(s,0,sizeof(*s));
    for(k=0;k<8;++k) s->count[k]=full;
    for(k=8;k<12;++k) s->count[k]=x;
    for(k=12;k<16;++k) s->count[k]=y;
    for(k=0;k<16;++k) {
        s->field[k]=(float*)malloc(s->count[k]*sizeof(float));
        if(!s->field[k]) return -1;
    }
    s->trace_count=(size_t)ntr*(size_t)NT;
    s->trace_vx=(float*)malloc(s->trace_count*sizeof(float));
    s->trace_vy=(float*)malloc(s->trace_count*sizeof(float));
    return s->trace_vx&&s->trace_vy?0:-1;
}

static void snapshot_capture(struct snapshot *s,const struct wavePSV *w,
                             const struct wavePSV_PML *p,
                             const struct seisPSV *seis) {
    float **matrix_field[5]={w->pvx,w->pvy,w->psxx,w->psyy,w->psxy};
    float ***tensor_field[3]={w->pr,w->pp,w->pq};
    float **x_field[4]={p->psi_sxx_x,p->psi_sxy_x,p->psi_vxx,p->psi_vyx};
    float **y_field[4]={p->psi_syy_y,p->psi_sxy_y,p->psi_vyy,p->psi_vxy};
    int k;
    for(k=0;k<5;++k) memcpy(s->field[k],&matrix_field[k][-2][-2],s->count[k]*sizeof(float));
    for(k=0;k<3;++k) memcpy(s->field[k+5],&tensor_field[k][-2][-2][1],s->count[k+5]*sizeof(float));
    for(k=0;k<4;++k) memcpy(s->field[k+8],&x_field[k][1][1],s->count[k+8]*sizeof(float));
    for(k=0;k<4;++k) memcpy(s->field[k+12],&y_field[k][1][1],s->count[k+12]*sizeof(float));
    memcpy(s->trace_vx,&seis->sectionvx[1][1],s->trace_count*sizeof(float));
    memcpy(s->trace_vy,&seis->sectionvy[1][1],s->trace_count*sizeof(float));
}

static void snapshot_free(struct snapshot *s) {
    int k; for(k=0;k<16;++k) free(s->field[k]);
    free(s->trace_vx); free(s->trace_vy); memset(s,0,sizeof(*s));
}

static void metric_add(struct metric *m,float a,float b,size_t index) {
    double difference=fabs((double)a-(double)b);
    double denominator=fmax(fmax(fabs((double)a),fabs((double)b)),1.0e-20);
    double relative=difference/denominator;
    if(difference>m->max_abs) { m->max_abs=difference; m->max_index=index; }
    if(relative>m->max_rel) m->max_rel=relative;
    m->sum_sq+=difference*difference; ++m->count;
}

static struct metric compare_values(const float *a,const float *b,size_t count) {
    struct metric m={0.0,0.0,0.0,0,0}; size_t k;
    for(k=0;k<count;++k) metric_add(&m,a[k],b[k],k);
    return m;
}

static double correlation(const float *a,const float *b,int count) {
    int k; double ma=0.0,mb=0.0,num=0.0,da=0.0,db=0.0;
    for(k=0;k<count;++k) { ma+=a[k]; mb+=b[k]; }
    ma/=count; mb/=count;
    for(k=0;k<count;++k) {
        double x=a[k]-ma,y=b[k]-mb; num+=x*y; da+=x*x; db+=y*y;
    }
    return num/sqrt(da*db);
}

static int report_comparison(const struct snapshot *cpu,const struct snapshot *gpu,
                             int ntr) {
    static const char *names[16]={"vx","vy","sxx","syy","sxy","r1","p1","q1",
        "psi_sxx_x","psi_sxy_x","psi_vxx","psi_vyx",
        "psi_syy_y","psi_sxy_y","psi_vyy","psi_vxy"};
    struct metric overall={0.0,0.0,0.0,0,0},tx,ty;
    double min_cx=1.0,min_cy=1.0; int r,k;
    for(k=0;k<16;++k) {
        struct metric m=compare_values(cpu->field[k],gpu->field[k],cpu->count[k]);
        double rms=sqrt(m.sum_sq/(double)m.count);
        printf("FINAL_PARITY field=%s max_abs=%.9e max_rel=%.9e rms=%.9e at=%zu\n",
               names[k],m.max_abs,m.max_rel,rms,m.max_index);
        if(m.max_abs>overall.max_abs) { overall.max_abs=m.max_abs; overall.max_index=(size_t)k; }
        if(m.max_rel>overall.max_rel) overall.max_rel=m.max_rel;
        overall.sum_sq+=m.sum_sq; overall.count+=m.count;
    }
    tx=compare_values(cpu->trace_vx,gpu->trace_vx,cpu->trace_count);
    ty=compare_values(cpu->trace_vy,gpu->trace_vy,cpu->trace_count);
    for(r=0;r<ntr;++r) {
        double cx=correlation(cpu->trace_vx+(size_t)r*NT,gpu->trace_vx+(size_t)r*NT,NT);
        double cy=correlation(cpu->trace_vy+(size_t)r*NT,gpu->trace_vy+(size_t)r*NT,NT);
        if(cx<min_cx) min_cx=cx;
        if(cy<min_cy) min_cy=cy;
    }
    printf("TRACE_PARITY component=vx max_abs=%.9e max_rel=%.9e rms=%.9e receiver=%zu timestep=%zu min_correlation=%.12f\n",
        tx.max_abs,tx.max_rel,sqrt(tx.sum_sq/(double)tx.count),tx.max_index/(size_t)NT+1,tx.max_index%(size_t)NT+1,min_cx);
    printf("TRACE_PARITY component=vy max_abs=%.9e max_rel=%.9e rms=%.9e receiver=%zu timestep=%zu min_correlation=%.12f\n",
        ty.max_abs,ty.max_rel,sqrt(ty.sum_sq/(double)ty.count),ty.max_index/(size_t)NT+1,ty.max_index%(size_t)NT+1,min_cy);
    printf("FINAL_STATE_ENVELOPE max_abs=%.9e max_rel=%.9e rms=%.9e field_index=%zu\n",
        overall.max_abs,overall.max_rel,sqrt(overall.sum_sq/(double)overall.count),overall.max_index);
    return 0;
}

static int snapshots_equal(const struct snapshot *a,const struct snapshot *b,
                           int traces) {
    int k; for(k=0;k<16;++k)
        if(memcmp(a->field[k],b->field[k],a->count[k]*sizeof(float))) return 0;
    if(traces&&(memcmp(a->trace_vx,b->trace_vx,a->trace_count*sizeof(float))||
                memcmp(a->trace_vy,b->trace_vy,a->trace_count*sizeof(float)))) return 0;
    return 1;
}

static int snapshot_field_range_equal(const struct snapshot *a,
                                      const struct snapshot *b,
                                      int first,int last) {
    int k;
    for(k=first;k<=last;++k)
        if(memcmp(a->field[k],b->field[k],a->count[k]*sizeof(float))) return 0;
    return 1;
}

static int snapshot_traces_equal(const struct snapshot *a,
                                 const struct snapshot *b) {
    return memcmp(a->trace_vx,b->trace_vx,a->trace_count*sizeof(float))==0&&
           memcmp(a->trace_vy,b->trace_vy,a->trace_count*sizeof(float))==0;
}

static void fill_mutable_sentinels(struct wavePSV *w,struct wavePSV_PML *p,
                                   struct seisPSV *seis,int ntr) {
    float **matrix_field[5]={w->pvx,w->pvy,w->psxx,w->psyy,w->psxy};
    float ***tensor_field[3]={w->pr,w->pp,w->pq};
    float **x_field[4]={p->psi_sxx_x,p->psi_sxy_x,p->psi_vxx,p->psi_vyx};
    float **y_field[4]={p->psi_syy_y,p->psi_sxy_y,p->psi_vyy,p->psi_vxy};
    size_t full=(size_t)(NY+6)*(size_t)(NX+6);
    size_t x=(size_t)NY*(size_t)(2*FW),y=(size_t)(2*FW)*(size_t)NX;
    size_t q,trace_count=(size_t)ntr*(size_t)NT;
    int k;
    for(k=0;k<5;++k) for(q=0;q<full;++q)
        (&matrix_field[k][-2][-2])[q]=100.0f+(float)(17*k)+(float)(q%251)*0.03125f;
    for(k=0;k<3;++k) for(q=0;q<full;++q)
        (&tensor_field[k][-2][-2][1])[q]=300.0f+(float)(19*k)+(float)(q%241)*0.0625f;
    for(k=0;k<4;++k) for(q=0;q<x;++q)
        (&x_field[k][1][1])[q]=500.0f+(float)(23*k)+(float)(q%233)*0.125f;
    for(k=0;k<4;++k) for(q=0;q<y;++q)
        (&y_field[k][1][1])[q]=700.0f+(float)(29*k)+(float)(q%227)*0.25f;
    for(q=0;q<trace_count;++q) {
        (&seis->sectionvx[1][1])[q]=900.0f+(float)(q%211)*0.5f;
        (&seis->sectionvy[1][1])[q]=1100.0f+(float)(q%199)*0.75f;
    }
}

static int rejected_state_unchanged(const struct snapshot *before,
                                    const struct snapshot *after,
                                    int *primary,int *gsls,int *cpml,
                                    int *receivers) {
    *primary=snapshot_field_range_equal(before,after,0,4);
    *gsls=snapshot_field_range_equal(before,after,5,7);
    *cpml=snapshot_field_range_equal(before,after,8,15);
    *receivers=snapshot_traces_equal(before,after);
    return *primary&&*gsls&&*cpml&&*receivers;
}

static int mutation_oracle(struct wavePSV *w,struct wavePSV_PML *p,
                           struct seisPSV *seis,int ntr,
                           struct snapshot *before,struct snapshot *after,
                           int no_device_only) {
    static const char *names[10]={"unknown","fdorder","L","free_surface",
        "topology","boundary","source_type","source_count","seismo","snap"};
    enum denise_psv_backend backend=DENISE_PSV_BACKEND_CPU;
    int accepted[10]={0};
    int primary=1,gsls=1,cpml=1,receivers=1,k,result,nsrc;
    fill_mutable_sentinels(w,p,seis,ntr);
    snapshot_capture(before,w,p,seis);
    if(no_device_only) {
        setenv("DENISE_PSV_BACKEND","cuda",1);
        result=denise_cuda_psv_backend_preflight(1,ntr,0,&backend);
        snapshot_capture(after,w,p,seis);
        if(result>=0||!rejected_state_unchanged(before,after,&primary,&gsls,&cpml,&receivers))
            return fail("no-device preflight mutated state or did not reject");
        printf("NO_DEVICE_FAIL_BEFORE_MUTATION failure=1 primary=%d gsls=%d cpml=%d receivers=%d\n",
            primary,gsls,cpml,receivers);
        return 0;
    }
    for(k=0;k<10;++k) {
        FDORDER=4; L=1; FREE_SURF=0; NPROCX=NPROCY=1; BOUNDARY=0;
        QUELLTYP=1; SEISMO=1; SNAP=0; nsrc=1;
        setenv("DENISE_PSV_BACKEND",k==0?"bogus":"cuda",1);
        if(k==1) FDORDER=8;
        else if(k==2) L=2;
        else if(k==3) FREE_SURF=1;
        else if(k==4) NPROCX=2;
        else if(k==5) BOUNDARY=1;
        else if(k==6) QUELLTYP=2;
        else if(k==7) nsrc=2;
        else if(k==8) SEISMO=2;
        else if(k==9) SNAP=1;
        result=denise_cuda_psv_backend_preflight(nsrc,ntr,0,&backend);
        snapshot_capture(after,w,p,seis);
        if(result>=0||!rejected_state_unchanged(before,after,&primary,&gsls,&cpml,&receivers)) {
            fprintf(stderr,"mutation oracle case failed: %s\n",names[k]);
            return 1;
        }
        accepted[k]=1;
    }
    FDORDER=4; L=1; FREE_SURF=0; NPROCX=NPROCY=1; BOUNDARY=0;
    QUELLTYP=1; SEISMO=1; SNAP=0;
    printf("FAIL_BEFORE_MUTATION unknown=%d fdorder=%d L=%d free_surface=%d topology=%d boundary=%d source_type=%d source_count=%d seismo=%d snap=%d primary=%d gsls=%d cpml=%d receivers=%d\n",
        accepted[0],accepted[1],accepted[2],accepted[3],accepted[4],accepted[5],
        accepted[6],accepted[7],accepted[8],accepted[9],primary,gsls,cpml,receivers);
    return 0;
}

static void bind_forward(struct denise_cuda_psv_forward_config *c,
                         struct denise_cuda_psv_forward_host *h,
                         struct wavePSV *w,struct wavePSV_PML *p,
                         struct matPSV *m,struct seisPSV *seis,
                         struct acq *a,float *hc,int ntr) {
    memset(c,0,sizeof(*c)); memset(h,0,sizeof(*h));
    c->core.nx=NX; c->core.ny=NY; c->core.fw=FW; c->core.fdorder=FDORDER;
    c->core.mechanisms=L; c->core.mpi_ranks_x=NPROCX; c->core.mpi_ranks_y=NPROCY;
    c->core.dt=DT; c->core.dh=DH; c->core.hc1=hc[1]; c->core.hc2=hc[2];
    c->core.bip1=m->bip[1]; c->core.bjm1=m->bjm[1];
    c->core.cip1=m->cip[1]; c->core.cjm1=m->cjm[1];
    c->core.safety_reserve_bytes=64*MIB;
    c->nt=NT; c->ntr=ntr; c->source_count=1; c->source_type=1;
    c->seismo=1; c->ndt=1;
    h->core.vx=w->pvx; h->core.vy=w->pvy; h->core.sxx=w->psxx;
    h->core.syy=w->psyy; h->core.sxy=w->psxy;
    h->core.r=w->pr; h->core.p=w->pp; h->core.q=w->pq;
    h->core.rip=m->prip; h->core.rjp=m->prjp; h->core.fipjp=m->fipjp;
    h->core.f=m->f; h->core.g=m->g; h->core.dip=m->dip; h->core.d=m->d; h->core.e=m->e;
    h->core.K_x=p->K_x; h->core.a_x=p->a_x; h->core.b_x=p->b_x;
    h->core.K_x_half=p->K_x_half; h->core.a_x_half=p->a_x_half; h->core.b_x_half=p->b_x_half;
    h->core.K_y=p->K_y; h->core.a_y=p->a_y; h->core.b_y=p->b_y;
    h->core.K_y_half=p->K_y_half; h->core.a_y_half=p->a_y_half; h->core.b_y_half=p->b_y_half;
    h->core.psi_sxx_x=p->psi_sxx_x; h->core.psi_sxy_x=p->psi_sxy_x;
    h->core.psi_syy_y=p->psi_syy_y; h->core.psi_sxy_y=p->psi_sxy_y;
    h->core.psi_vxx=p->psi_vxx; h->core.psi_vyx=p->psi_vyx;
    h->core.psi_vyy=p->psi_vyy; h->core.psi_vxy=p->psi_vxy;
    h->source_positions=a->srcpos_loc; h->source_signals=a->signals;
    h->receiver_positions=a->recpos_loc; h->sectionvx=seis->sectionvx; h->sectionvy=seis->sectionvy;
}

static int failure_lifecycle(struct wavePSV *w,struct wavePSV_PML *p,
                             struct matPSV *m,struct seisPSV *seis,
                             struct acq *a,float *hc,int ntr) {
    struct denise_cuda_psv_forward_config c,bad;
    struct denise_cuda_psv_forward_host h;
    struct denise_cuda_psv_forward_stats plan;
    struct denise_cuda_psv_forward *context=NULL;
    int saved;
    bind_forward(&c,&h,w,p,m,seis,a,hc,ntr);
    if(denise_cuda_psv_forward_required_bytes(&c,&plan)!=0) return fail("forward planner rejected valid fixture");
    bad=c; bad.core.user_cap_bytes=bad.core.safety_reserve_bytes+plan.total_mandatory_bytes-1;
    if(denise_cuda_psv_forward_create(&bad,&h,&context)==0||context) return fail("aggregate budget did not fail closed");
    saved=(int)a->srcpos_loc[1][1]; a->srcpos_loc[1][1]=0.0f;
    if(denise_cuda_psv_forward_create(&c,&h,&context)==0||context) return fail("invalid source accepted");
    a->srcpos_loc[1][1]=(float)saved;
    saved=a->recpos_loc[1][1]; a->recpos_loc[1][1]=0;
    if(denise_cuda_psv_forward_create(&c,&h,&context)==0||context) return fail("invalid receiver accepted");
    a->recpos_loc[1][1]=saved;
    bad=c; bad.core.fdorder=8;
    if(denise_cuda_psv_forward_required_bytes(&bad,&plan)==0) return fail("unsupported envelope accepted");
    if(denise_cuda_psv_forward_create(&c,&h,&context)!=0) return fail("valid retry after failures failed");
    if(denise_cuda_psv_forward_destroy(&context)!=0||context) return fail("forward destroy failed");
    printf("FAILURE_LIFECYCLE budget=1 source_bounds=1 receiver_bounds=1 unsupported=1 retry=1 destroy=1\n");
    return 0;
}

static int benchmark_solver(struct wavePSV *wave,struct wavePSV_PML *pml,
        struct matPSV *material,struct mpiPSV *mpi,struct seisPSV *seis,
        struct seisPSVfwi *seisfwi,struct fwiPSV *fwi,struct acq *acquisition,
        float *hc,int *dtinv,MPI_Request *request,int ntr,int warmups,
        int repetitions) {
    double *cpu_host=NULL,*cuda_host=NULL,*cuda_internal=NULL,*resident=NULL;
    double *internal_host_ratio=NULL,*setup=NULL,*upload=NULL;
    double *trace_download=NULL,*mutable_download=NULL;
    struct denise_cuda_psv_forward_stats stats;
    struct distribution cpu_summary,cuda_host_summary,cuda_internal_summary;
    struct distribution resident_summary,ratio_summary;
    int k,status=1,agreement_warnings=0;
#define ALLOC_SAMPLE(name) do { name=(double*)calloc((size_t)repetitions,sizeof(double)); \
    if(!(name)) goto cleanup; } while(0)
    ALLOC_SAMPLE(cpu_host); ALLOC_SAMPLE(cuda_host); ALLOC_SAMPLE(cuda_internal);
    ALLOC_SAMPLE(resident); ALLOC_SAMPLE(internal_host_ratio);
    ALLOC_SAMPLE(setup); ALLOC_SAMPLE(upload); ALLOC_SAMPLE(trace_download);
    ALLOC_SAMPLE(mutable_download);
#undef ALLOC_SAMPLE
    unsetenv("DENISE_CUDA_PROFILE");
    for(k=0;k<warmups;++k) {
        setenv("DENISE_PSV_BACKEND","cpu",1);
        psv(wave,pml,material,fwi,mpi,seis,seisfwi,acquisition,hc,
            1,1,1,NT,ntr,NULL,NULL,1,dtinv,0,request,request);
    }
    for(k=0;k<repetitions;++k) {
        double start,stop;
        setenv("DENISE_PSV_BACKEND","cpu",1); start=monotonic_ms();
        if(start<0.0) { fail("CLOCK_MONOTONIC start failed"); goto cleanup; }
        psv(wave,pml,material,fwi,mpi,seis,seisfwi,acquisition,hc,
            1,1,1,NT,ntr,NULL,NULL,1,dtinv,0,request,request);
        stop=monotonic_ms();
        if(stop<start) { fail("CLOCK_MONOTONIC CPU stop failed"); goto cleanup; }
        cpu_host[k]=stop-start;
    }
    for(k=0;k<warmups;++k) {
        setenv("DENISE_PSV_BACKEND","cuda",1);
        psv(wave,pml,material,fwi,mpi,seis,seisfwi,acquisition,hc,
            1,1,1,NT,ntr,NULL,NULL,1,dtinv,0,request,request);
    }
    for(k=0;k<repetitions;++k) {
        double start,stop;
        setenv("DENISE_PSV_BACKEND","cuda",1); start=monotonic_ms();
        if(start<0.0) { fail("CLOCK_MONOTONIC start failed"); goto cleanup; }
        psv(wave,pml,material,fwi,mpi,seis,seisfwi,acquisition,hc,
            1,1,1,NT,ntr,NULL,NULL,1,dtinv,0,request,request);
        stop=monotonic_ms();
        if(stop<start) { fail("CLOCK_MONOTONIC CUDA stop failed"); goto cleanup; }
        cuda_host[k]=stop-start;
        if(denise_cuda_psv_dispatch_last_stats(&stats)!=0) {
            fail("benchmark CUDA dispatch stats unavailable"); goto cleanup;
        }
        cuda_internal[k]=stats.total_forward_ms;
        resident[k]=stats.resident_timestep_ms;
        internal_host_ratio[k]=cuda_internal[k]/cuda_host[k];
        if(internal_host_ratio[k]<0.80||internal_host_ratio[k]>1.20) {
            ++agreement_warnings;
            fprintf(stderr,
                "CUDA timing scope warning: repetition=%d internal_host_ratio=%.6f host_ms=%.6f internal_ms=%.6f\n",
                k+1,internal_host_ratio[k],cuda_host[k],cuda_internal[k]);
        }
        setup[k]=stats.context_setup_ms; upload[k]=stats.initial_upload_ms;
        trace_download[k]=stats.trace_download_ms;
        mutable_download[k]=stats.mutable_download_ms;
    }
    cpu_summary=summarize(cpu_host,repetitions);
    cuda_host_summary=summarize(cuda_host,repetitions);
    cuda_internal_summary=summarize(cuda_internal,repetitions);
    resident_summary=summarize(resident,repetitions);
    ratio_summary=summarize(internal_host_ratio,repetitions);
    printf("BENCHMARK_METHOD warmup=%d repetitions=%d fresh_cuda_context=1 primary=median profile=0 schedule=cpu_block_then_cuda_block clock=CLOCK_MONOTONIC headline=WARM_RUNTIME_SOLVER_END_TO_END_HOST_WALL cold_start_excluded=1 cleanup_included=1\n",
           warmups,repetitions);
    printf("BENCHMARK_CASE nx=%d ny=%d nt=%d nrec=%d\n",NX,NY,NT,ntr);
    printf("BENCHMARK_VRAM state_bytes=%zu source_bytes=%zu receiver_geometry_bytes=%zu trace_bytes=%zu usable_bytes=%zu remaining_bytes=%zu managed_memory=0 paging_fallback=0\n",
           stats.b1_core_bytes,stats.source_signal_bytes+stats.source_geometry_bytes,
           stats.receiver_geometry_bytes,stats.trace_bytes,
           stats.usable_budget_bytes,stats.remaining_budget_bytes);
    print_samples("cpu_host_e2e_ms",cpu_host,repetitions);
    print_samples("cuda_host_e2e_ms",cuda_host,repetitions);
    print_samples("cuda_internal_e2e_ms",cuda_internal,repetitions);
    print_samples("cuda_resident_propagation_ms",resident,repetitions);
    print_samples("cuda_internal_host_ratio",internal_host_ratio,repetitions);
    print_samples("cuda_context_setup_ms",setup,repetitions);
    print_samples("cuda_upload_ms",upload,repetitions);
    print_samples("cuda_trace_download_ms",trace_download,repetitions);
    print_samples("cuda_mutable_download_ms",mutable_download,repetitions);
    printf("BENCHMARK_SUMMARY metric=cpu_host_e2e_ms min=%.6f q1=%.6f median=%.6f q3=%.6f max=%.6f iqr=%.6f\n",
           cpu_summary.minimum,cpu_summary.q1,cpu_summary.median,cpu_summary.q3,
           cpu_summary.maximum,cpu_summary.iqr);
    printf("BENCHMARK_SUMMARY metric=cuda_host_e2e_ms min=%.6f q1=%.6f median=%.6f q3=%.6f max=%.6f iqr=%.6f\n",
           cuda_host_summary.minimum,cuda_host_summary.q1,cuda_host_summary.median,
           cuda_host_summary.q3,cuda_host_summary.maximum,cuda_host_summary.iqr);
    printf("BENCHMARK_SUMMARY metric=cuda_internal_e2e_ms min=%.6f q1=%.6f median=%.6f q3=%.6f max=%.6f iqr=%.6f\n",
           cuda_internal_summary.minimum,cuda_internal_summary.q1,
           cuda_internal_summary.median,cuda_internal_summary.q3,
           cuda_internal_summary.maximum,cuda_internal_summary.iqr);
    printf("BENCHMARK_SUMMARY metric=cuda_resident_propagation_ms min=%.6f q1=%.6f median=%.6f q3=%.6f max=%.6f iqr=%.6f\n",
           resident_summary.minimum,resident_summary.q1,resident_summary.median,
           resident_summary.q3,resident_summary.maximum,resident_summary.iqr);
    printf("BENCHMARK_AGREEMENT metric=cuda_internal_host_ratio min=%.6f median=%.6f max=%.6f warning_threshold_low=0.80 warning_threshold_high=1.20 warnings=%d headline_valid=%d\n",
           ratio_summary.minimum,ratio_summary.median,ratio_summary.maximum,
           agreement_warnings,agreement_warnings==0);
    printf("BENCHMARK_HEADLINE valid=%d cpu_host_median_ms=%.6f cuda_host_median_ms=%.6f speedup=%.9f runtime_fraction=%.9f runtime_reduction=%.9f cuda_internal_median_ms=%.6f cuda_resident_median_ms=%.6f resident_only_ratio=%.9f cell_timesteps_per_second=%.3f\n",
           agreement_warnings==0,cpu_summary.median,cuda_host_summary.median,
           cpu_summary.median/cuda_host_summary.median,
           cuda_host_summary.median/cpu_summary.median,
           1.0-cuda_host_summary.median/cpu_summary.median,
           cuda_internal_summary.median,resident_summary.median,
           cpu_summary.median/resident_summary.median,
           (double)NX*(double)NY*(double)NT/(resident_summary.median/1000.0));
    printf("BENCHMARK_SYNC forward_syncs_per_run=%zu per_timestep=0 profile_syncs=0 profile_event_records=%zu profile_elapsed_queries=%zu\n",
           stats.forward_synchronization_calls,stats.profile_event_records,
           stats.profile_elapsed_queries);
    status=0;
cleanup:
    free(mutable_download); free(trace_download); free(upload); free(setup);
    free(internal_host_ratio); free(resident); free(cuda_internal);
    free(cuda_host); free(cpu_host); return status;
}

int main(int argc,char **argv) {
    struct wavePSV wave={0}; struct wavePSV_PML pml={0}; struct matPSV material={0};
    struct mpiPSV mpi={0}; struct seisPSV seis={0}; struct seisPSVfwi seisfwi={0};
    struct fwiPSV fwi={0}; struct acq acquisition={0};
    struct snapshot cpu_unset={0},cpu={0},gpu={0},repeat={0};
    struct denise_cuda_psv_forward_stats stats,profile_stats;
    MPI_Request request[4]={MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL,MPI_REQUEST_NULL};
    float *hc=NULL; int *dtinv=NULL; int nx=DEFAULT_NX,ny=DEFAULT_NY,nt=DEFAULT_NT,ntr=DEFAULT_NTR;
    int status=1,k; double cpu_start,cpu_ms,gpu_start,gpu_ms,profile_start,profile_ms=0.0;
    const char *probe=NULL; int run_mutation_oracle=0,run_no_device_oracle=0;
    int benchmark=0,profile_compare=0,warmups=1,repetitions=7;
    MPI_Init(&argc,&argv); MPI_Comm_rank(MPI_COMM_WORLD,&MYID);
    for(k=1;k<argc;++k) {
        if(!strcmp(argv[k],"--nx")&&k+1<argc) nx=atoi(argv[++k]);
        else if(!strcmp(argv[k],"--ny")&&k+1<argc) ny=atoi(argv[++k]);
        else if(!strcmp(argv[k],"--nt")&&k+1<argc) nt=atoi(argv[++k]);
        else if(!strcmp(argv[k],"--ntr")&&k+1<argc) ntr=atoi(argv[++k]);
        else if(!strcmp(argv[k],"--probe-backend")&&k+1<argc) probe=argv[++k];
        else if(!strcmp(argv[k],"--mutation-oracle")) run_mutation_oracle=1;
        else if(!strcmp(argv[k],"--mutation-oracle-no-device")) run_no_device_oracle=1;
        else if(!strcmp(argv[k],"--benchmark")) benchmark=1;
        else if(!strcmp(argv[k],"--profile-compare")) profile_compare=1;
        else if(!strcmp(argv[k],"--warmup")&&k+1<argc) warmups=atoi(argv[++k]);
        else if(!strcmp(argv[k],"--repetitions")&&k+1<argc) repetitions=atoi(argv[++k]);
        else { fail("unknown command-line option"); goto cleanup; }
    }
    if(nx<2*TEST_FW+5||ny<2*TEST_FW+5||nt<2||ntr<1||warmups<1||repetitions<1) {
        fail("invalid benchmark dimensions or repetition counts"); goto cleanup;
    }
    configure_globals(nx,ny,nt);
    alloc_PSV(&wave,&pml); alloc_matPSV(&material); alloc_mpiPSV(&mpi);
    initialize_cpml(&pml); initialize_material(&material);
    initialize_acquisition(&acquisition,&seis,ntr);
    hc=vector(1,2); hc[1]=9.0f/8.0f; hc[2]=-1.0f/24.0f;
    dtinv=(int*)calloc((size_t)NT+1,sizeof(int));
    if(!hc||!dtinv||(!benchmark&&(snapshot_alloc(&cpu_unset,ntr)||
       snapshot_alloc(&cpu,ntr)||snapshot_alloc(&gpu,ntr)||snapshot_alloc(&repeat,ntr)))) {
        fail("host fixture allocation failed"); goto cleanup;
    }
    if(benchmark) {
        status=benchmark_solver(&wave,&pml,&material,&mpi,&seis,&seisfwi,&fwi,
            &acquisition,hc,dtinv,request,ntr,warmups,repetitions);
        goto cleanup;
    }
    if(run_mutation_oracle||run_no_device_oracle) {
        if(mutation_oracle(&wave,&pml,&seis,ntr,&cpu,&gpu,run_no_device_oracle))
            goto cleanup;
        status=0; goto cleanup;
    }
    if(probe) {
        setenv("DENISE_PSV_BACKEND",strcmp(probe,"unsupported")==0 ? "cuda" : probe,1);
        if(!strcmp(probe,"unsupported")) FREE_SURF=1;
        psv(&wave,&pml,&material,&fwi,&mpi,&seis,&seisfwi,&acquisition,hc,
            1,1,1,NT,ntr,NULL,NULL,1,dtinv,0,request,request);
        fail("failure probe unexpectedly returned"); goto cleanup;
    }
    if(failure_lifecycle(&wave,&pml,&material,&seis,&acquisition,hc,ntr)) goto cleanup;

    unsetenv("DENISE_PSV_BACKEND");
    psv(&wave,&pml,&material,&fwi,&mpi,&seis,&seisfwi,&acquisition,hc,
        1,1,1,NT,ntr,NULL,NULL,1,dtinv,0,request,request);
    snapshot_capture(&cpu_unset,&wave,&pml,&seis);

    setenv("DENISE_PSV_BACKEND","cpu",1); cpu_start=MPI_Wtime();
    psv(&wave,&pml,&material,&fwi,&mpi,&seis,&seisfwi,&acquisition,hc,
        1,1,1,NT,ntr,NULL,NULL,1,dtinv,0,request,request);
    cpu_ms=(MPI_Wtime()-cpu_start)*1000.0; snapshot_capture(&cpu,&wave,&pml,&seis);

    if(profile_compare) unsetenv("DENISE_CUDA_PROFILE");
    setenv("DENISE_PSV_BACKEND","cuda",1); gpu_start=monotonic_ms();
    psv(&wave,&pml,&material,&fwi,&mpi,&seis,&seisfwi,&acquisition,hc,
        1,1,1,NT,ntr,NULL,NULL,1,dtinv,0,request,request);
    gpu_ms=monotonic_ms()-gpu_start; snapshot_capture(&gpu,&wave,&pml,&seis);
    if(denise_cuda_psv_dispatch_last_stats(&stats)!=0) { fail("CUDA dispatch stats unavailable"); goto cleanup; }

    if(profile_compare) setenv("DENISE_CUDA_PROFILE","1",1);
    profile_start=monotonic_ms();
    psv(&wave,&pml,&material,&fwi,&mpi,&seis,&seisfwi,&acquisition,hc,
        1,1,1,NT,ntr,NULL,NULL,1,dtinv,0,request,request);
    profile_ms=monotonic_ms()-profile_start;
    snapshot_capture(&repeat,&wave,&pml,&seis);
    if(profile_compare&&denise_cuda_psv_dispatch_last_stats(&profile_stats)!=0) {
        fail("profile CUDA dispatch stats unavailable"); goto cleanup;
    }
    report_comparison(&cpu,&gpu,ntr);
    printf("GPU_REPEAT traces=%d final_state=%d\n",
        snapshots_equal(&gpu,&repeat,1),snapshots_equal(&gpu,&repeat,0));
    printf("CPU_SELECTOR_PARITY traces=%d final_state=%d\n",
        snapshot_traces_equal(&cpu_unset,&cpu),snapshots_equal(&cpu_unset,&cpu,0));
    printf("NO_FMA_DIAGNOSTIC traces_vx=%d traces_vy=%d final_state=%d\n",
        memcmp(cpu.trace_vx,gpu.trace_vx,cpu.trace_count*sizeof(float))==0,
        memcmp(cpu.trace_vy,gpu.trace_vy,cpu.trace_count*sizeof(float))==0,
        snapshots_equal(&cpu,&gpu,0));
    printf("MEMORY_PLAN b1=%zu source_signal=%zu source_geometry=%zu receiver_geometry=%zu traces=%zu workspace=%zu total=%zu usable=%zu remaining=%zu\n",
        stats.b1_core_bytes,stats.source_signal_bytes,stats.source_geometry_bytes,
        stats.receiver_geometry_bytes,stats.trace_bytes,stats.workspace_bytes,
        stats.total_mandatory_bytes,stats.usable_budget_bytes,stats.remaining_budget_bytes);
    printf("TRANSFER_CONTRACT h2d_calls=%zu d2h_calls=%zu h2d_bytes=%zu d2h_bytes=%zu full_h2d_per_step=%zu full_d2h_per_step=%zu source_h2d_per_step=%zu receiver_d2h_per_step=%zu\n",
        stats.h2d_transfer_calls,stats.d2h_transfer_calls,stats.h2d_bytes,stats.d2h_bytes,
        stats.full_grid_h2d_per_timestep,stats.full_grid_d2h_per_timestep,
        stats.source_sample_h2d_per_timestep,stats.receiver_sample_d2h_per_timestep);
    printf("TIMING_MODE profile=%d forward_syncs=%zu per_timestep_syncs=0 resident_event_records=%zu resident_elapsed_queries=%zu profile_event_records=%zu profile_elapsed_queries=%zu\n",
        stats.profiling_enabled,stats.forward_synchronization_calls,
        stats.resident_event_records,stats.resident_elapsed_queries,
        stats.profile_event_records,stats.profile_elapsed_queries);
    if(profile_compare)
        printf("PROFILE_MODE diagnostic=1 numerical_traces_equal=%d numerical_final_state_equal=%d forward_syncs=%zu event_records=%zu elapsed_queries=%zu velocity_ms=%.6f stress_ms=%.6f source_ms=%.6f receiver_ms=%.6f resident_ms=%.6f normal_host_e2e_ms=%.6f profile_host_e2e_ms=%.6f overhead_ratio=%.6f definitive_target_evidence=0\n",
            snapshots_equal(&gpu,&repeat,1),snapshots_equal(&gpu,&repeat,0),
            profile_stats.forward_synchronization_calls,
            profile_stats.profile_event_records,profile_stats.profile_elapsed_queries,
            profile_stats.velocity_kernel_ms,profile_stats.stress_kernel_ms,
            profile_stats.source_kernel_ms,profile_stats.receiver_kernel_ms,
            profile_stats.resident_timestep_ms,gpu_ms,profile_ms,profile_ms/gpu_ms);
    printf("TIMING cpu_total_ms=%.6f cuda_total_wall_ms=%.6f cuda_invocation_ms=%.6f resident_ms=%.6f velocity_ms=%.6f stress_ms=%.6f source_ms=%.6f receiver_ms=%.6f context_setup_ms=%.6f upload_ms=%.6f trace_download_ms=%.6f mutable_download_ms=%.6f\n",
        cpu_ms,gpu_ms,stats.total_forward_ms,stats.resident_timestep_ms,
        stats.velocity_kernel_ms,stats.stress_kernel_ms,stats.source_kernel_ms,
        stats.receiver_kernel_ms,stats.context_setup_ms,stats.initial_upload_ms,
        stats.trace_download_ms,stats.mutable_download_ms);
    printf("SOLVER_FORWARD_PASS nx=%d ny=%d nt=%d ntr=%d source=explosive seismo=velocity exchange_state_effect=0\n",
        NX,NY,NT,ntr);
    status=0;
cleanup:
    snapshot_free(&repeat); snapshot_free(&gpu); snapshot_free(&cpu); snapshot_free(&cpu_unset);
    if(dtinv) free(dtinv);
    if(hc) free_vector(hc,1,2);
    if(seis.sectionvx) free_matrix(seis.sectionvx,1,ntr,1,NT);
    if(seis.sectionvy) free_matrix(seis.sectionvy,1,ntr,1,NT);
    if(acquisition.recpos_loc) free_imatrix(acquisition.recpos_loc,1,3,1,ntr);
    if(acquisition.signals) free_matrix(acquisition.signals,1,1,1,NT);
    if(acquisition.srcpos_loc) free_matrix(acquisition.srcpos_loc,1,8,1,1);
    if(material.prho) free_material(&material);
    if(FL) { free_vector(FL,1,1); FL=NULL; }
    if(mpi.bufferlef_to_rig) free_mpi(&mpi);
    if(wave.pvx) dealloc_PSV(&wave,&pml);
    MPI_Finalize(); return status;
}
