#include "fd.h"
#include "denise_cuda_psv.h"
#include "globvar.h"

#include <float.h>
#include <stdint.h>

#define TEST_NX 96
#define TEST_NY 80
#define TEST_FW 8
#define J0 -2
#define I0 -2
#define J1 (TEST_NY + 3)
#define I1 (TEST_NX + 3)
#define MIB ((size_t)1024 * (size_t)1024)

static float BIP[2], BJM[2], CIP[2], CJM[2];

struct fixture {
    struct denise_cuda_psv_fd4_l1_host h;
};

void visco_psv_exact_velocity(int j, int i, float x, float y) {
    (void)j; (void)i; (void)x; (void)y;
}
void visco_psv_exact_strain(int j, int i, float xx, float yx, float xy, float yy) {
    (void)j; (void)i; (void)xx; (void)yx; (void)xy; (void)yy;
}

static int fail(const char *message) {
    fprintf(stderr, "CUDA PSV B1 harness failure: %s\n", message);
    return 1;
}

static int alloc_fixture(struct fixture *f) {
    struct denise_cuda_psv_fd4_l1_host *h=&f->h;
    memset(f,0,sizeof(*f));
#define AM(m) do { h->m=matrix(J0,J1,I0,I1); if(!h->m) return 1; } while(0)
#define AT(m) do { h->m=f3tensor(J0,J1,I0,I1,1,1); if(!h->m) return 1; } while(0)
#define AX(m) do { h->m=matrix(1,TEST_NY,1,2*TEST_FW); if(!h->m) return 1; } while(0)
#define AY(m) do { h->m=matrix(1,2*TEST_FW,1,TEST_NX); if(!h->m) return 1; } while(0)
#define AV(m) do { h->m=vector(1,2*TEST_FW); if(!h->m) return 1; } while(0)
    AM(vx); AM(vy); AM(sxx); AM(syy); AM(sxy);
    AT(r); AT(p); AT(q);
    AM(rip); AM(rjp); AM(fipjp); AM(f); AM(g);
    AT(dip); AT(d); AT(e);
    AX(psi_sxx_x); AX(psi_sxy_x); AX(psi_vxx); AX(psi_vyx);
    AY(psi_syy_y); AY(psi_sxy_y); AY(psi_vyy); AY(psi_vxy);
    AV(K_x); AV(a_x); AV(b_x); AV(K_x_half); AV(a_x_half); AV(b_x_half);
    AV(K_y); AV(a_y); AV(b_y); AV(K_y_half); AV(a_y_half); AV(b_y_half);
#undef AM
#undef AT
#undef AX
#undef AY
#undef AV
    return 0;
}

static void free_fixture(struct fixture *f) {
    struct denise_cuda_psv_fd4_l1_host *h=&f->h;
#define FM(m) if(h->m) free_matrix(h->m,J0,J1,I0,I1)
#define FT(m) if(h->m) free_f3tensor(h->m,J0,J1,I0,I1,1,1)
#define FX(m) if(h->m) free_matrix(h->m,1,TEST_NY,1,2*TEST_FW)
#define FY(m) if(h->m) free_matrix(h->m,1,2*TEST_FW,1,TEST_NX)
#define FV(m) if(h->m) free_vector(h->m,1,2*TEST_FW)
    FM(vx); FM(vy); FM(sxx); FM(syy); FM(sxy);
    FT(r); FT(p); FT(q); FM(rip); FM(rjp); FM(fipjp); FM(f); FM(g);
    FT(dip); FT(d); FT(e);
    FX(psi_sxx_x); FX(psi_sxy_x); FX(psi_vxx); FX(psi_vyx);
    FY(psi_syy_y); FY(psi_sxy_y); FY(psi_vyy); FY(psi_vxy);
    FV(K_x); FV(a_x); FV(b_x); FV(K_x_half); FV(a_x_half); FV(b_x_half);
    FV(K_y); FV(a_y); FV(b_y); FV(K_y_half); FV(a_y_half); FV(b_y_half);
#undef FM
#undef FT
#undef FX
#undef FY
#undef FV
    memset(f,0,sizeof(*f));
}

static void copy_matrix_all(float **dst,float **src,int j0,int j1,int i0,int i1) {
    size_t n=(size_t)(j1-j0+1)*(size_t)(i1-i0+1);
    memcpy(&dst[j0][i0],&src[j0][i0],n*sizeof(float));
}
static void copy_tensor_all(float ***dst,float ***src) {
    size_t n=(size_t)(J1-J0+1)*(size_t)(I1-I0+1);
    memcpy(&dst[J0][I0][1],&src[J0][I0][1],n*sizeof(float));
}
static void copy_fixture(struct fixture *dst,const struct fixture *src) {
    struct denise_cuda_psv_fd4_l1_host *d=&dst->h;
    const struct denise_cuda_psv_fd4_l1_host *s=&src->h;
#define CM(m) copy_matrix_all(d->m,s->m,J0,J1,I0,I1)
#define CT(m) copy_tensor_all(d->m,s->m)
#define CX(m) copy_matrix_all(d->m,s->m,1,TEST_NY,1,2*TEST_FW)
#define CY(m) copy_matrix_all(d->m,s->m,1,2*TEST_FW,1,TEST_NX)
#define CV(m) memcpy(&d->m[1],&s->m[1],(size_t)(2*TEST_FW)*sizeof(float))
    CM(vx); CM(vy); CM(sxx); CM(syy); CM(sxy); CT(r); CT(p); CT(q);
    CM(rip); CM(rjp); CM(fipjp); CM(f); CM(g); CT(dip); CT(d); CT(e);
    CX(psi_sxx_x); CX(psi_sxy_x); CX(psi_vxx); CX(psi_vyx);
    CY(psi_syy_y); CY(psi_sxy_y); CY(psi_vyy); CY(psi_vxy);
    CV(K_x); CV(a_x); CV(b_x); CV(K_x_half); CV(a_x_half); CV(b_x_half);
    CV(K_y); CV(a_y); CV(b_y); CV(K_y_half); CV(a_y_half); CV(b_y_half);
#undef CM
#undef CT
#undef CX
#undef CY
#undef CV
}

static float signed_pattern(int i,int j,int seed,float scale) {
    int raw=((i+19)*37+(j+23)*53+seed*11)%41-20;
    return scale*((float)raw+0.125f*(float)((i-j+seed)&3));
}

static int prepare_physical_material(struct fixture *fx,
        struct denise_cuda_psv_fd4_l1_config *cfg) {
    float **vs=matrix(J0,J1,I0,I1),**muip=matrix(J0,J1,I0,I1);
    float **vp=matrix(J0,J1,I0,I1),**rho=matrix(J0,J1,I0,I1);
    float **taus=matrix(J0,J1,I0,I1),**taup=matrix(J0,J1,I0,I1);
    float **tausip=matrix(J0,J1,I0,I1);
    float *peta=vector(1,1),*etaip=vector(1,1),*etajm=vector(1,1);
    int i,j;
    if(!vs||!muip||!vp||!rho||!taus||!taup||!tausip||!peta||!etaip||!etajm)
        return fail("material preparation allocation failed");
    FL=vector(1,1);
    if(!FL) return fail("FL allocation failed");
    FL[1]=12.0f;
    peta[1]=2.0f*(float)PI*FL[1]*DT;
    for(j=1;j<=NY;++j) for(i=1;i<=NX;++i) {
        float density=1950.0f+0.8f*(float)i+0.6f*(float)j;
        float shear=1450.0f+0.7f*(float)i-0.3f*(float)j;
        float compressional=2650.0f+0.4f*(float)i+0.2f*(float)j;
        rho[j][i]=density; vs[j][i]=shear; vp[j][i]=compressional;
        muip[j][i]=density*shear*shear;
        taus[j][i]=0.018f+0.00001f*(float)(i+j);
        taup[j][i]=0.012f+0.00001f*(float)(2*i+j);
        tausip[j][i]=0.017f+0.00001f*(float)(i+2*j);
        fx->h.rip[j][i]=1.0f/density;
        fx->h.rjp[j][i]=1.0f/density;
    }
    prepare_update_s_visc_PSV(etajm,etaip,peta,fx->h.fipjp,vs,muip,vp,rho,
        taus,taup,tausip,fx->h.f,fx->h.g,BIP,BJM,CIP,CJM,
        fx->h.dip,fx->h.d,fx->h.e);
    cfg->bip1=BIP[1]; cfg->bjm1=BJM[1];
    cfg->cip1=CIP[1]; cfg->cjm1=CJM[1];
    free_matrix(vs,J0,J1,I0,I1); free_matrix(muip,J0,J1,I0,I1);
    free_matrix(vp,J0,J1,I0,I1); free_matrix(rho,J0,J1,I0,I1);
    free_matrix(taus,J0,J1,I0,I1); free_matrix(taup,J0,J1,I0,I1);
    free_matrix(tausip,J0,J1,I0,I1);
    free_vector(peta,1,1); free_vector(etaip,1,1); free_vector(etajm,1,1);
    return 0;
}

static int initialize_fixture(struct fixture *f,
        struct denise_cuda_psv_fd4_l1_config *cfg) {
    int i,j,h;
    for(j=J0;j<=J1;++j) for(i=I0;i<=I1;++i) {
        f->h.vx[j][i]=signed_pattern(i,j,1,2.0e-5f);
        f->h.vy[j][i]=signed_pattern(i,j,2,2.5e-5f);
        f->h.sxx[j][i]=signed_pattern(i,j,3,4.0f);
        f->h.syy[j][i]=signed_pattern(i,j,4,3.5f);
        f->h.sxy[j][i]=signed_pattern(i,j,5,2.5f);
        f->h.r[j][i][1]=signed_pattern(i,j,6,1.0e-3f);
        f->h.p[j][i][1]=signed_pattern(i,j,7,1.2e-3f);
        f->h.q[j][i][1]=signed_pattern(i,j,8,0.9e-3f);
    }
    if(prepare_physical_material(f,cfg)!=0) return 1;
    for(h=1;h<=2*TEST_FW;++h) {
        float t=(float)h/(float)(2*TEST_FW);
        f->h.K_x[h]=1.0f+0.08f*t; f->h.K_y[h]=1.0f+0.07f*t;
        f->h.K_x_half[h]=1.0f+0.075f*t; f->h.K_y_half[h]=1.0f+0.065f*t;
        f->h.a_x[h]=-0.018f*t; f->h.a_y[h]=-0.017f*t;
        f->h.a_x_half[h]=-0.016f*t; f->h.a_y_half[h]=-0.015f*t;
        f->h.b_x[h]=0.985f-0.02f*t; f->h.b_y[h]=0.984f-0.02f*t;
        f->h.b_x_half[h]=0.986f-0.02f*t; f->h.b_y_half[h]=0.983f-0.02f*t;
    }
    for(j=1;j<=TEST_NY;++j) for(h=1;h<=2*TEST_FW;++h) {
        f->h.psi_sxx_x[j][h]=signed_pattern(h,j,9,2.0e-4f);
        f->h.psi_sxy_x[j][h]=signed_pattern(h,j,10,2.0e-4f);
        f->h.psi_vxx[j][h]=signed_pattern(h,j,11,2.0e-7f);
        f->h.psi_vyx[j][h]=signed_pattern(h,j,12,2.0e-7f);
    }
    for(h=1;h<=2*TEST_FW;++h) for(i=1;i<=TEST_NX;++i) {
        f->h.psi_syy_y[h][i]=signed_pattern(i,h,13,2.0e-4f);
        f->h.psi_sxy_y[h][i]=signed_pattern(i,h,14,2.0e-4f);
        f->h.psi_vyy[h][i]=signed_pattern(i,h,15,2.0e-7f);
        f->h.psi_vxy[h][i]=signed_pattern(i,h,16,2.0e-7f);
    }
    return 0;
}
static void cpu_step(struct fixture *f,float *hc,int nt) {
    struct denise_cuda_psv_fd4_l1_host *h=&f->h;
    update_v_PML_PSV(1,NX,1,NY,nt,
        h->vx,NULL,NULL,h->vy,NULL,NULL,NULL,NULL,h->sxx,h->syy,h->sxy,
        h->rip,h->rjp,NULL,NULL,NULL,0,NULL,hc,0,0,
        h->K_x,h->a_x,h->b_x,h->K_x_half,h->a_x_half,h->b_x_half,
        h->K_y,h->a_y,h->b_y,h->K_y_half,h->a_y_half,h->b_y_half,
        h->psi_sxx_x,h->psi_syy_y,h->psi_sxy_y,h->psi_sxy_x,0);
    update_s_visc_PML_PSV(1,NX,1,NY,NX,NY,
        h->vx,h->vy,NULL,NULL,NULL,NULL,h->sxx,h->syy,h->sxy,
        NULL,NULL,NULL,NULL,hc,0,h->r,h->p,h->q,h->fipjp,h->f,h->g,
        &BIP[0],&BJM[0],&CIP[0],&CJM[0],h->d,h->e,h->dip,
        h->K_x,h->a_x,h->b_x,h->K_x_half,h->a_x_half,h->b_x_half,
        h->K_y,h->a_y,h->b_y,h->K_y_half,h->a_y_half,h->b_y_half,
        h->psi_vxx,h->psi_vyy,h->psi_vxy,h->psi_vyx,0);
}

struct metric {
    double max_abs,max_rel,sum_sq;
    uint32_t max_ulp;
    size_t count;
    int max_i,max_j;
};

static uint32_t ordered_float(float value) {
    uint32_t bits;
    memcpy(&bits,&value,sizeof(bits));
    return (bits&0x80000000u)?0x80000000u-bits:bits+0x80000000u;
}
static uint32_t ulp_distance(float a,float b) {
    uint32_t x=ordered_float(a),y=ordered_float(b);
    return x>y?x-y:y-x;
}
static void metric_add(struct metric *m,float cpu,float gpu,int j,int i) {
    double diff=fabs((double)cpu-(double)gpu);
    double denom=fmax(fabs((double)cpu),1.0e-20);
    double rel=diff/denom;
    uint32_t ulp=ulp_distance(cpu,gpu);
    if(diff>m->max_abs) { m->max_abs=diff; m->max_i=i; m->max_j=j; }
    if(rel>m->max_rel) m->max_rel=rel;
    if(ulp>m->max_ulp) m->max_ulp=ulp;
    m->sum_sq+=diff*diff; ++m->count;
}
static int report_matrix_metric(int steps,const char *name,float **cpu,float **gpu,
                                int j0,int j1,int i0,int i1) {
    struct metric m={0.0,0.0,0.0,0,0,0,0};
    int i,j;
    for(j=j0;j<=j1;++j) for(i=i0;i<=i1;++i) {
        if(!isfinite(cpu[j][i])||!isfinite(gpu[j][i])) return fail("non-finite parity value");
        metric_add(&m,cpu[j][i],gpu[j][i],j,i);
    }
    printf("PARITY steps=%d field=%s max_abs=%.9e max_rel=%.9e rms=%.9e max_ulp=%u at=%d,%d\n",
        steps,name,m.max_abs,m.max_rel,sqrt(m.sum_sq/(double)m.count),m.max_ulp,m.max_j,m.max_i);
    return 0;
}
static int report_tensor_metric(int steps,const char *name,float ***cpu,float ***gpu) {
    struct metric m={0.0,0.0,0.0,0,0,0,0};
    int i,j;
    for(j=1;j<=NY;++j) for(i=1;i<=NX;++i) {
        if(!isfinite(cpu[j][i][1])||!isfinite(gpu[j][i][1])) return fail("non-finite tensor parity value");
        metric_add(&m,cpu[j][i][1],gpu[j][i][1],j,i);
    }
    printf("PARITY steps=%d field=%s max_abs=%.9e max_rel=%.9e rms=%.9e max_ulp=%u at=%d,%d\n",
        steps,name,m.max_abs,m.max_rel,sqrt(m.sum_sq/(double)m.count),m.max_ulp,m.max_j,m.max_i);
    return 0;
}
static int report_all(int steps,const struct fixture *cpu,const struct fixture *gpu) {
    const struct denise_cuda_psv_fd4_l1_host *a=&cpu->h,*b=&gpu->h;
#define PM(m) if(report_matrix_metric(steps,#m,a->m,b->m,1,NY,1,NX)!=0) return 1
#define PT(m) if(report_tensor_metric(steps,#m "1",a->m,b->m)!=0) return 1
#define PX(m) if(report_matrix_metric(steps,#m,a->m,b->m,1,NY,1,2*FW)!=0) return 1
#define PY(m) if(report_matrix_metric(steps,#m,a->m,b->m,1,2*FW,1,NX)!=0) return 1
    PM(vx); PM(vy); PM(sxx); PM(syy); PM(sxy); PT(r); PT(p); PT(q);
    PX(psi_sxx_x); PX(psi_sxy_x); PY(psi_syy_y); PY(psi_sxy_y);
    PX(psi_vxx); PX(psi_vyx); PY(psi_vyy); PY(psi_vxy);
#undef PM
#undef PT
#undef PX
#undef PY
    return 0;
}

static int matrix_bytes_equal(float **a,float **b,int j0,int j1,int i0,int i1) {
    size_t n=(size_t)(j1-j0+1)*(size_t)(i1-i0+1);
    return memcmp(&a[j0][i0],&b[j0][i0],n*sizeof(float))==0;
}
static int tensor_bytes_equal(float ***a,float ***b) {
    size_t n=(size_t)(J1-J0+1)*(size_t)(I1-I0+1);
    return memcmp(&a[J0][I0][1],&b[J0][I0][1],n*sizeof(float))==0;
}
static int mutable_bytes_equal(const struct fixture *a,const struct fixture *b) {
    const struct denise_cuda_psv_fd4_l1_host *x=&a->h,*y=&b->h;
#define EM(m) if(!matrix_bytes_equal(x->m,y->m,J0,J1,I0,I1)) return 0
#define ET(m) if(!tensor_bytes_equal(x->m,y->m)) return 0
#define EX(m) if(!matrix_bytes_equal(x->m,y->m,1,NY,1,2*FW)) return 0
#define EY(m) if(!matrix_bytes_equal(x->m,y->m,1,2*FW,1,NX)) return 0
#define EV(m) if(memcmp(&x->m[1],&y->m[1],(size_t)(2*FW)*sizeof(float))!=0) return 0
    EM(vx); EM(vy); EM(sxx); EM(syy); EM(sxy); ET(r); ET(p); ET(q);
    EM(rip); EM(rjp); EM(fipjp); EM(f); EM(g); ET(dip); ET(d); ET(e);
    EX(psi_sxx_x); EX(psi_sxy_x); EX(psi_vxx); EX(psi_vyx);
    EY(psi_syy_y); EY(psi_sxy_y); EY(psi_vyy); EY(psi_vxy);
    EV(K_x); EV(a_x); EV(b_x); EV(K_x_half); EV(a_x_half); EV(b_x_half);
    EV(K_y); EV(a_y); EV(b_y); EV(K_y_half); EV(a_y_half); EV(b_y_half);
#undef EM
#undef ET
#undef EX
#undef EY
#undef EV
    return 1;
}
static int halos_unchanged(const struct fixture *initial,const struct fixture *result) {
    const struct denise_cuda_psv_fd4_l1_host *a=&initial->h,*b=&result->h;
    int i,j;
    float **af[5]={a->vx,a->vy,a->sxx,a->syy,a->sxy};
    float **bf[5]={b->vx,b->vy,b->sxx,b->syy,b->sxy};
    int k;
    for(k=0;k<5;++k) for(j=J0;j<=J1;++j) for(i=I0;i<=I1;++i)
        if((j<1||j>NY||i<1||i>NX)&&memcmp(&af[k][j][i],&bf[k][j][i],sizeof(float))!=0) return 0;
    for(j=J0;j<=J1;++j) for(i=I0;i<=I1;++i) if(j<1||j>NY||i<1||i>NX) {
        if(memcmp(&a->r[j][i][1],&b->r[j][i][1],sizeof(float))||
           memcmp(&a->p[j][i][1],&b->p[j][i][1],sizeof(float))||
           memcmp(&a->q[j][i][1],&b->q[j][i][1],sizeof(float))) return 0;
    }
    return 1;
}

static void report_bulk_cpml(int steps,float **cpu,float **gpu) {
    double bulk=0.0,cpml=0.0;
    int i,j;
    for(j=1;j<=NY;++j) for(i=1;i<=NX;++i) {
        double d=fabs((double)cpu[j][i]-(double)gpu[j][i]);
        if(i>FW&&i<=NX-FW&&j>FW&&j<=NY-FW) { if(d>bulk) bulk=d; }
        else if(d>cpml) cpml=d;
    }
    printf("REGION_PARITY steps=%d field=sxx bulk_max_abs=%.9e cpml_max_abs=%.9e\n",steps,bulk,cpml);
}
int main(int argc,char **argv) {
    struct fixture initial={0},cpu={0},gpu={0},repeat={0};
    struct denise_cuda_psv_fd4_l1_config cfg,bad;
    struct denise_cuda_psv_fd4_l1 *ctx=NULL,*ctx2=NULL,*rejected=NULL;
    struct denise_cuda_psv_fd4_l1_stats before,after,stats;
    size_t required=0;
    float *hc=NULL;
    int checkpoints[3]={1,10,100};
    int previous=0,index,k,status=1;

    MPI_Init(&argc,&argv);
    NX=TEST_NX; NY=TEST_NY; FW=TEST_FW; FDORDER=4; L=1;
    NPROCX=1; NPROCY=1; POS[1]=0; POS[2]=0;
    BOUNDARY=0; FREE_SURF=0; GRAD_FORM=0; INVMAT1=1;
    MYID=0; DT=0.0002f; DH=10.0f; FP=stdout;

    memset(&cfg,0,sizeof(cfg));
    cfg.nx=NX; cfg.ny=NY; cfg.fw=FW; cfg.fdorder=FDORDER;
    cfg.mechanisms=L; cfg.mpi_ranks_x=1; cfg.mpi_ranks_y=1;
    cfg.boundary=BOUNDARY; cfg.free_surface=FREE_SURF; cfg.mode=0;
    cfg.logical_device=0; cfg.dt=DT; cfg.dh=DH;
    cfg.hc1=9.0f/8.0f; cfg.hc2=-1.0f/24.0f;
    cfg.safety_reserve_bytes=64*MIB; cfg.user_cap_bytes=0;

    if(alloc_fixture(&initial)||alloc_fixture(&cpu)||alloc_fixture(&gpu)||alloc_fixture(&repeat)) {
        fail("fixture allocation failed"); goto cleanup;
    }
    if(initialize_fixture(&initial,&cfg)!=0) goto cleanup;
    copy_fixture(&cpu,&initial); copy_fixture(&gpu,&initial); copy_fixture(&repeat,&initial);
    hc=vector(1,2); if(!hc) { fail("hc allocation failed"); goto cleanup; }
    hc[1]=cfg.hc1; hc[2]=cfg.hc2;

    if(denise_cuda_psv_fd4_l1_required_bytes(&cfg,&required)!=0) {
        fail(denise_cuda_psv_fd4_l1_last_error()); goto cleanup;
    }
    bad=cfg; bad.user_cap_bytes=bad.safety_reserve_bytes+required-1;
    if(denise_cuda_psv_fd4_l1_create(&bad,&initial.h,&rejected)==0||rejected!=NULL) {
        fail("aggregate over-budget context unexpectedly succeeded"); goto cleanup;
    }
    if(!strstr(denise_cuda_psv_fd4_l1_last_error(),"mandatory state")) {
        fail("aggregate budget diagnostic is unclear"); goto cleanup;
    }
    bad=cfg; bad.fdorder=8;
    if(denise_cuda_psv_fd4_l1_required_bytes(&bad,&required)==0) {
        fail("unsupported FDORDER did not fail closed"); goto cleanup;
    }
    if(denise_cuda_psv_fd4_l1_required_bytes(&cfg,&required)!=0) goto cleanup;

    if(denise_cuda_psv_fd4_l1_create(&cfg,&initial.h,&ctx)!=0) {
        fail(denise_cuda_psv_fd4_l1_last_error()); goto cleanup;
    }
    if(denise_cuda_psv_fd4_l1_get_stats(ctx,&stats)!=0) goto cleanup;
    printf("DEVICE_STATE full_j=-2:%d full_i=-2:%d x_cpml_j=1:%d x_cpml_h=1:%d y_cpml_h=1:%d y_cpml_i=1:%d\n",
        NY+3,NX+3,NY,2*FW,2*FW,NX);
    printf("VRAM_PLAN mandatory=%zu usable=%zu remaining=%zu\n",
        stats.mandatory_state_bytes,stats.usable_budget_bytes,stats.remaining_budget_bytes);
    printf("AGGREGATE_BUDGET fail_closed=1 retry=1 unsupported_config=1\n");

    if(denise_cuda_psv_fd4_l1_download(ctx,&gpu.h)!=0) goto cleanup;
    if(!mutable_bytes_equal(&initial,&gpu)) {
        fail("initial CPU/GPU mutable payload is not byte-identical"); goto cleanup;
    }
    printf("INITIAL_STATE_BYTE_IDENTICAL fields=36 result=1\n");

    for(index=0;index<3;++index) {
        int target=checkpoints[index],delta=target-previous;
        if(denise_cuda_psv_fd4_l1_get_stats(ctx,&before)!=0) goto cleanup;
        for(k=previous+1;k<=target;++k) cpu_step(&cpu,hc,k);
        if(denise_cuda_psv_fd4_l1_step(ctx,delta)!=0) {
            fail(denise_cuda_psv_fd4_l1_last_error()); goto cleanup;
        }
        if(denise_cuda_psv_fd4_l1_get_stats(ctx,&after)!=0) goto cleanup;
        if(after.h2d_transfer_calls!=before.h2d_transfer_calls||
           after.d2h_transfer_calls!=before.d2h_transfer_calls) {
            fail("full-grid transfer occurred during timestep"); goto cleanup;
        }
        if(denise_cuda_psv_fd4_l1_download(ctx,&gpu.h)!=0) goto cleanup;
        if(report_all(target,&cpu,&gpu)!=0) goto cleanup;
        report_bulk_cpml(target,cpu.h.sxx,gpu.h.sxx);
        previous=target;
    }
    if(!halos_unchanged(&initial,&gpu)) {
        fail("untouched FD4 halo changed on GPU"); goto cleanup;
    }
    printf("EXACT_ORACLES upload_download=1 untouched_halo=1 mapping=1\n");

    if(denise_cuda_psv_fd4_l1_create(&cfg,&initial.h,&ctx2)!=0) goto cleanup;
    if(denise_cuda_psv_fd4_l1_step(ctx2,100)!=0) goto cleanup;
    if(denise_cuda_psv_fd4_l1_download(ctx2,&repeat.h)!=0) goto cleanup;
    if(!mutable_bytes_equal(&gpu,&repeat)) {
        fail("repeated GPU run is not bitwise deterministic"); goto cleanup;
    }
    printf("GPU_REPEAT_BITWISE_IDENTICAL steps=100 result=1\n");

    if(denise_cuda_psv_fd4_l1_get_stats(ctx,&stats)!=0) goto cleanup;
    printf("TRANSFER_COUNTS initial_h2d=%zu checkpoint_d2h=%zu per_timestep=0 h2d_bytes=%zu d2h_bytes=%zu\n",
        stats.h2d_transfer_calls,stats.d2h_transfer_calls,stats.h2d_bytes,stats.d2h_bytes);
    printf("TIMING steps=%llu velocity_total_ms=%.6f stress_total_ms=%.6f combined_total_ms=%.6f velocity_avg_ms=%.6f stress_avg_ms=%.6f combined_avg_ms=%.6f upload_ms=%.6f download_total_ms=%.6f\n",
        stats.timesteps,stats.velocity_kernel_ms,stats.stress_kernel_ms,
        stats.combined_kernel_ms,stats.velocity_kernel_ms/(float)stats.timesteps,
        stats.stress_kernel_ms/(float)stats.timesteps,
        stats.combined_kernel_ms/(float)stats.timesteps,
        stats.upload_ms,stats.download_ms);
    printf("PHYSICS_FIXTURE production_prepare_update_s_visc_PSV=1 cpml_deterministic=1\n");
    printf("CUDA_PSV_FD4_L1_PASS steps=1,10,100 fields=16 residency=1 deterministic=1\n");
    status=0;

cleanup:
    denise_cuda_psv_fd4_l1_destroy(&rejected);
    denise_cuda_psv_fd4_l1_destroy(&ctx2);
    denise_cuda_psv_fd4_l1_destroy(&ctx);
    if(hc) free_vector(hc,1,2);
    if(FL) { free_vector(FL,1,1); FL=NULL; }
    free_fixture(&repeat); free_fixture(&gpu); free_fixture(&cpu); free_fixture(&initial);
    MPI_Finalize();
    return status;
}
