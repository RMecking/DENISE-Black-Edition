/* MPI harness for one actual production viscoelastic SH forward step and the
 * independently callable M6.3c-5a full-state transpose integration. */

#include "fd.h"

#include <errno.h>
#include <stdint.h>

float DT, DH;
int NX, NY, POS[3], NPROCX, NPROCY, BOUNDARY, FDORDER, FREE_SURF, FW, L;
int INDEX[5], MYID, GRAD_FORM, ADJ_SIGN, QUELLTYP, QUELLTYPB, MODE;
const int TAG1 = 1, TAG2 = 2, TAG5 = 5, TAG6 = 6;
FILE *FP;

struct field {
    float **v, **rows, *data;
    int rmin, rmax, cmin, cmax, nrows, ncols;
};

struct volume {
    float ***v, ***rows, **cols, *data;
    int rmin, rmax, cmin, cmax, mechanisms, nrows, ncols;
};

struct owned_state {
    struct visco_sh_full_state view;
    struct field vz, sxz, syz;
    struct volume r, q;
    struct field psi_sxz_x, psi_syz_y, psi_vzx, psi_vzy;
};

static struct field field_new(int rmin, int rmax, int cmin, int cmax) {
    struct field f;
    int j;
    f.rmin = rmin; f.rmax = rmax; f.cmin = cmin; f.cmax = cmax;
    f.nrows = rmax-rmin+1; f.ncols = cmax-cmin+1;
    f.rows = (float **)calloc((size_t)f.nrows, sizeof(float *));
    f.data = (float *)calloc((size_t)f.nrows*f.ncols, sizeof(float));
    if (!f.rows || !f.data) MPI_Abort(MPI_COMM_WORLD, 121);
    f.v = f.rows-rmin;
    for (j=rmin; j<=rmax; ++j)
        f.v[j] = f.data+(j-rmin)*f.ncols-cmin;
    return f;
}

static struct volume volume_new(
        int rmin, int rmax, int cmin, int cmax, int mechanisms) {
    struct volume v;
    int j, i;
    v.rmin=rmin; v.rmax=rmax; v.cmin=cmin; v.cmax=cmax;
    v.mechanisms=mechanisms; v.nrows=rmax-rmin+1; v.ncols=cmax-cmin+1;
    v.rows=(float ***)calloc((size_t)v.nrows,sizeof(float **));
    v.cols=(float **)calloc((size_t)v.nrows*v.ncols,sizeof(float *));
    v.data=(float *)calloc((size_t)v.nrows*v.ncols*(mechanisms+1),sizeof(float));
    if(!v.rows||!v.cols||!v.data) MPI_Abort(MPI_COMM_WORLD,122);
    v.v=v.rows-rmin;
    for(j=rmin;j<=rmax;++j){
        v.v[j]=v.cols+(j-rmin)*v.ncols-cmin;
        for(i=cmin;i<=cmax;++i)
            v.v[j][i]=v.data+((j-rmin)*v.ncols+(i-cmin))*(mechanisms+1);
    }
    return v;
}

static void field_free(struct field *f){ free(f->data); free(f->rows); }
static void volume_free(struct volume *v){ free(v->data);free(v->cols);free(v->rows); }

static struct owned_state state_new(int h, int fw, int mechanisms) {
    struct owned_state s;
    int rmin=-h, rmax=NY+h+1, cmin=1-h, cmax=NX+h;
    s.vz=field_new(rmin,rmax,cmin,cmax);
    s.sxz=field_new(rmin,rmax,cmin,cmax);
    s.syz=field_new(rmin,rmax,cmin,cmax);
    s.r=volume_new(rmin,rmax,cmin,cmax,mechanisms);
    s.q=volume_new(rmin,rmax,cmin,cmax,mechanisms);
    if(fw>0){
        s.psi_sxz_x=field_new(1,NY,1,2*fw);
        s.psi_syz_y=field_new(1,2*fw,1,NX);
        s.psi_vzx=field_new(1,NY,1,2*fw);
        s.psi_vzy=field_new(1,2*fw,1,NX);
    }else{
        memset(&s.psi_sxz_x,0,sizeof(s.psi_sxz_x));
        memset(&s.psi_syz_y,0,sizeof(s.psi_syz_y));
        memset(&s.psi_vzx,0,sizeof(s.psi_vzx));
        memset(&s.psi_vzy,0,sizeof(s.psi_vzy));
    }
    s.view.vz=s.vz.v;s.view.sxz=s.sxz.v;s.view.syz=s.syz.v;
    s.view.r=s.r.v;s.view.q=s.q.v;
    s.view.psi_sxz_x=fw?s.psi_sxz_x.v:NULL;
    s.view.psi_syz_y=fw?s.psi_syz_y.v:NULL;
    s.view.psi_vzx=fw?s.psi_vzx.v:NULL;
    s.view.psi_vzy=fw?s.psi_vzy.v:NULL;
    return s;
}

static void state_free(struct owned_state *s,int fw){
    field_free(&s->vz);field_free(&s->sxz);field_free(&s->syz);
    volume_free(&s->r);volume_free(&s->q);
    if(fw){field_free(&s->psi_sxz_x);field_free(&s->psi_syz_y);
        field_free(&s->psi_vzx);field_free(&s->psi_vzy);}
}

static void field_copy(struct field *dst,const struct field *src){
    memcpy(dst->data,src->data,(size_t)src->nrows*src->ncols*sizeof(float));
}
static void volume_copy(struct volume *dst,const struct volume *src){
    memcpy(dst->data,src->data,(size_t)src->nrows*src->ncols*(src->mechanisms+1)*sizeof(float));
}
static void state_copy(struct owned_state *dst,const struct owned_state *src,int fw){
    field_copy(&dst->vz,&src->vz);field_copy(&dst->sxz,&src->sxz);field_copy(&dst->syz,&src->syz);
    volume_copy(&dst->r,&src->r);volume_copy(&dst->q,&src->q);
    if(fw){field_copy(&dst->psi_sxz_x,&src->psi_sxz_x);field_copy(&dst->psi_syz_y,&src->psi_syz_y);
        field_copy(&dst->psi_vzx,&src->psi_vzx);field_copy(&dst->psi_vzy,&src->psi_vzy);}
}

static float deterministic(int rank,int group,int j,int i,int dual){
    double z=0.021*(rank+1)+0.009*(group+1)+0.0013*j-0.0008*i+(dual?0.39:-0.17);
    return (float)(0.31*sin(4.3*z)+0.19*cos(2.7*z));
}
static float deterministic_aux(int rank,int group,int k,int dual){
    return (float)(0.13*sin(0.071*(k+1)+0.17*rank+0.09*group+(dual?0.23:-0.11)));
}

static void state_initialize(struct owned_state *s,int rank,int dual,int fw,int mechanisms){
    int i,j,l,k;
    for(j=s->vz.rmin;j<=s->vz.rmax;++j)for(i=s->vz.cmin;i<=s->vz.cmax;++i){
        s->view.vz[j][i]=deterministic(rank,0,j,i,dual);
        s->view.sxz[j][i]=deterministic(rank,1,j,i,dual);
        s->view.syz[j][i]=deterministic(rank,2,j,i,dual);
        for(l=1;l<=mechanisms;++l){
            s->view.r[j][i][l]=deterministic(rank,2+l,j,i,dual);
            s->view.q[j][i][l]=deterministic(rank,2+mechanisms+l,j,i,dual);
        }
    }
    if(fw){
        k=0;for(j=1;j<=NY;++j)for(i=1;i<=2*fw;++i){s->view.psi_sxz_x[j][i]=deterministic_aux(rank,20,k,dual);s->view.psi_vzx[j][i]=deterministic_aux(rank,22,k++,dual);}
        k=0;for(j=1;j<=2*fw;++j)for(i=1;i<=NX;++i){s->view.psi_syz_y[j][i]=deterministic_aux(rank,21,k,dual);s->view.psi_vzy[j][i]=deterministic_aux(rank,23,k++,dual);}
    }
}

static float **buffer_new(int rows,int cols,float **storage){
    float **base=(float **)calloc((size_t)rows,sizeof(float *));
    float *data=(float *)calloc((size_t)rows*cols,sizeof(float));int j;
    if(!base||!data)MPI_Abort(MPI_COMM_WORLD,123);
    for(j=1;j<=rows;++j)base[j-1]=data+(j-1)*cols-1;
    *storage=data;return base-1;
}
static void buffer_free(float **b,float *s){free(s);free(b+1);}

static void topology(int rank){
    int ranks=NPROCX*NPROCY;POS[1]=rank%NPROCX;POS[2]=rank/NPROCX;
    INDEX[1]=rank-1;INDEX[2]=rank+1;INDEX[3]=rank-NPROCX;INDEX[4]=rank+NPROCX;
    if(POS[1]==0)INDEX[1]+=NPROCX;if(POS[1]==NPROCX-1)INDEX[2]-=NPROCX;
    if(POS[2]==0)INDEX[3]=ranks+rank-NPROCX;if(POS[2]==NPROCY-1)INDEX[4]=rank+NPROCX-ranks;
}

static double field_dot(const struct field *a,const struct field *b){
    double sum=0.0;int k,n=a->nrows*a->ncols;for(k=0;k<n;++k)sum+=(double)a->data[k]*b->data[k];return sum;
}
static double volume_dot(const struct volume *a,const struct volume *b){
    double sum=0.0;int j,i,l;for(j=a->rmin;j<=a->rmax;++j)for(i=a->cmin;i<=a->cmax;++i)for(l=1;l<=a->mechanisms;++l)sum+=(double)a->v[j][i][l]*b->v[j][i][l];return sum;
}
static double state_dot(const struct owned_state *a,const struct owned_state *b,int fw){
    double sum=field_dot(&a->vz,&b->vz)+field_dot(&a->sxz,&b->sxz)+field_dot(&a->syz,&b->syz)+volume_dot(&a->r,&b->r)+volume_dot(&a->q,&b->q);
    if(fw)sum+=field_dot(&a->psi_sxz_x,&b->psi_sxz_x)+field_dot(&a->psi_syz_y,&b->psi_syz_y)+field_dot(&a->psi_vzx,&b->psi_vzx)+field_dot(&a->psi_vzy,&b->psi_vzy);return sum;
}

static void write_field(FILE *fp,const struct field *f){fwrite(f->data,sizeof(float),(size_t)f->nrows*f->ncols,fp);}
static void write_volume(FILE *fp,const struct volume *v){int j,i,l;for(l=1;l<=v->mechanisms;++l)for(j=v->rmin;j<=v->rmax;++j)for(i=v->cmin;i<=v->cmax;++i)fwrite(&v->v[j][i][l],sizeof(float),1,fp);}
static void write_state(FILE *fp,const struct owned_state *s,int fw){
    write_field(fp,&s->vz);write_field(fp,&s->sxz);write_field(fp,&s->syz);write_volume(fp,&s->r);write_volume(fp,&s->q);
    if(fw){write_field(fp,&s->psi_sxz_x);write_field(fp,&s->psi_syz_y);write_field(fp,&s->psi_vzx);write_field(fp,&s->psi_vzy);}
}

int main(int argc,char **argv){
    struct owned_state input,forward_state,dual,work,prev;
    struct field rhoi,fipjp,f,diag[6],dummy;
    struct volume dip,d,pp;
    struct visco_sh_full_step_config cfg;
    float *hc,*bip,*bjm,*cip,*cjm,*K,*Kh,*a,*ah,*b,*bh;
    float **srcpos,**signals,*src_storage,*signal_storage;
    float **bl,**br,**bt,**bb,*sl,*sr,*st,*sb;
    MPI_Request req[4];FILE *out;char path[4096];
    int rank,size,h,i,j,l,k,source_i,source_j,receiver_i,receiver_j,status;
    int precondition_only;
    int rec_x[1],rec_y[1],src_x[1],src_y[1],src_type[1];
    double bar_receiver[1],bar_signal[1],lhs_local,rhs_local,lhs,rhs,residual,diag_change=0.0;

    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);MYID=rank;FP=stderr;
    if((argc!=11)&&(argc!=12)){if(rank==0)fprintf(stderr,"usage: npx npy boundary fs fd L fw nx ny outdir [precondition]\n");MPI_Abort(MPI_COMM_WORLD,124);}
    precondition_only=(argc==12)&&!strcmp(argv[11],"precondition");
    NPROCX=atoi(argv[1]);NPROCY=atoi(argv[2]);BOUNDARY=atoi(argv[3]);FREE_SURF=atoi(argv[4]);FDORDER=atoi(argv[5]);L=atoi(argv[6]);FW=atoi(argv[7]);NX=atoi(argv[8]);NY=atoi(argv[9]);
    if(size!=NPROCX*NPROCY)MPI_Abort(MPI_COMM_WORLD,125);topology(rank);h=FDORDER/2;DT=0.0013f;DH=7.5f;GRAD_FORM=2;ADJ_SIGN=1;MODE=0;QUELLTYPB=1;
    input=state_new(h,FW,L);forward_state=state_new(h,FW,L);dual=state_new(h,FW,L);work=state_new(h,FW,L);prev=state_new(h,FW,L);
    state_initialize(&input,rank,0,FW,L);state_initialize(&dual,rank,1,FW,L);state_copy(&forward_state,&input,FW);state_copy(&work,&dual,FW);
    rhoi=field_new(-h,NY+h+1,1-h,NX+h);fipjp=field_new(-h,NY+h+1,1-h,NX+h);f=field_new(-h,NY+h+1,1-h,NX+h);dummy=field_new(-h,NY+h+1,1-h,NX+h);
    dip=volume_new(-h,NY+h+1,1-h,NX+h,L);d=volume_new(-h,NY+h+1,1-h,NX+h,L);pp=volume_new(-h,NY+h+1,1-h,NX+h,L);
    for(j=-h;j<=NY+h+1;++j)for(i=1-h;i<=NX+h;++i){rhoi.v[j][i]=(float)(0.00043+0.000002*rank+0.0000003*j);fipjp.v[j][i]=(float)(0.0047+0.00001*i+0.000006*j);f.v[j][i]=(float)(0.0042+0.000008*i+0.000004*j);for(l=1;l<=L;++l){dip.v[j][i][l]=(float)(0.16+0.004*(l-1)+0.0002*i);d.v[j][i][l]=(float)(0.14+0.003*(l-1)+0.0002*j);}}
    hc=(float *)calloc((size_t)h+1,sizeof(float));bip=(float *)calloc((size_t)L+1,sizeof(float));bjm=(float *)calloc((size_t)L+1,sizeof(float));cip=(float *)calloc((size_t)L+1,sizeof(float));cjm=(float *)calloc((size_t)L+1,sizeof(float));
    {double values[6]={1.0,-0.041,0.007,-0.0014,0.00031,-0.00007};for(k=1;k<=h;++k)hc[k]=(float)values[k-1];}
    for(l=1;l<=L;++l){bip[l]=(float)(0.79+0.025*(l-1));bjm[l]=(float)(0.77+0.021*(l-1));cip[l]=(float)(0.91-0.018*(l-1));cjm[l]=(float)(0.89-0.015*(l-1));}
    K=(float *)calloc((size_t)2*FW+1,sizeof(float));Kh=(float *)calloc((size_t)2*FW+1,sizeof(float));a=(float *)calloc((size_t)2*FW+1,sizeof(float));ah=(float *)calloc((size_t)2*FW+1,sizeof(float));b=(float *)calloc((size_t)2*FW+1,sizeof(float));bh=(float *)calloc((size_t)2*FW+1,sizeof(float));
    for(k=1;k<=2*FW;++k){K[k]=(float)(1.11+0.017*k);Kh[k]=(float)(1.08+0.013*k);a[k]=(float)(-0.031-0.001*k);ah[k]=(float)(-0.027-0.0008*k);b[k]=(float)(0.82+0.006*k);bh[k]=(float)(0.84+0.005*k);}
    for(k=0;k<6;++k){diag[k]=field_new(-h,NY+h+1,1-h,NX+h);for(j=diag[k].rmin;j<=diag[k].rmax;++j)for(i=diag[k].cmin;i<=diag[k].cmax;++i)diag[k].v[j][i]=7.25f+(float)k;}
    srcpos=buffer_new(8,1,&src_storage);signals=buffer_new(1,1,&signal_storage);source_i=3+rank%2;source_j=4+rank%3;receiver_i=6+rank%3;receiver_j=5+rank%2;
    srcpos[1][1]=(float)source_i;srcpos[2][1]=(float)source_j;srcpos[8][1]=1.0f;signals[1][1]=(float)(0.021+0.003*rank);
    bl=buffer_new(NY,2*(h+1),&sl);br=buffer_new(NY,2*(h+1),&sr);bt=buffer_new(NX,2*(h+1),&st);bb=buffer_new(NX,2*(h+1),&sb);
    update_v_PML_SH(1,NX,1,NY,1,forward_state.view.vz,diag[0].v,diag[1].v,diag[2].v,forward_state.view.sxz,forward_state.view.syz,dummy.v,rhoi.v,srcpos,signals,1,dummy.v,hc,0,0,0,K,a,b,Kh,ah,bh,K,a,b,Kh,ah,bh,forward_state.view.psi_sxz_x,forward_state.view.psi_syz_y);
    exchange_v_SH(forward_state.view.vz,bl,br,bt,bb,req,req);if(FREE_SURF&&POS[2]==0)surface_elastic_SH_velocity(forward_state.view.vz,NX,h);
    update_s_visc_PML_SH(1,NX,1,NY,forward_state.view.vz,diag[3].v,diag[4].v,forward_state.view.syz,forward_state.view.sxz,dummy.v,dummy.v,dummy.v,hc,0,forward_state.view.r,pp.v,forward_state.view.q,fipjp.v,f.v,dummy.v,bip,bjm,cip,cjm,d.v,pp.v,dip.v,K,a,b,Kh,ah,bh,K,a,b,Kh,ah,bh,forward_state.view.psi_vzx,forward_state.view.psi_vzy,NULL,0);
    if(FREE_SURF&&POS[2]==0)surface_elastic_SH_stress(forward_state.view.syz,NX,h);exchange_s_SH(forward_state.view.sxz,forward_state.view.syz,bl,br,bt,bb,req,req);
    bar_receiver[0]=-0.17+0.019*rank;rec_x[0]=receiver_i;rec_y[0]=receiver_j;src_x[0]=source_i;src_y[0]=source_j;src_type[0]=1;
    memset(&cfg,0,sizeof(cfg));cfg.nx=NX;cfg.ny=NY;cfg.fdorder=FDORDER;cfg.mechanisms=L;cfg.fw=FW;cfg.free_surface=FREE_SURF;cfg.boundary=BOUNDARY;memcpy(cfg.pos,POS,sizeof(POS));memcpy(cfg.index,INDEX,sizeof(INDEX));cfg.nproc_x=NPROCX;cfg.nproc_y=NPROCY;cfg.dt=DT;cfg.dh=DH;cfg.hc=hc;cfg.rhoi=rhoi.v;cfg.fipjp=fipjp.v;cfg.f=f.v;cfg.bip=bip;cfg.bjm=bjm;cfg.cip=cip;cfg.cjm=cjm;cfg.dip=dip.v;cfg.d=d.v;cfg.K_x=K;cfg.a_x=a;cfg.b_x=b;cfg.K_x_half=Kh;cfg.a_x_half=ah;cfg.b_x_half=bh;cfg.K_y=K;cfg.a_y=a;cfg.b_y=b;cfg.K_y_half=Kh;cfg.a_y_half=ah;cfg.b_y_half=bh;cfg.nrec=1;cfg.rec_x=rec_x;cfg.rec_y=rec_y;cfg.bar_receiver=bar_receiver;cfg.nsrc=1;cfg.src_x=src_x;cfg.src_y=src_y;cfg.source_type=src_type;cfg.comm=MPI_COMM_WORLD;
    status=visco_sh_full_state_adjoint_step(&cfg,&work.view,&prev.view,bar_signal);
    if(precondition_only){
        if(status!=-2){fprintf(stderr,"rank %d expected overlap status -2, got %d\n",rank,status);MPI_Abort(MPI_COMM_WORLD,126);}
        if(rank==0)printf("{\"precondition_status\":%d}\n",status);
        MPI_Finalize();return 0;
    }
    if(status!=0){fprintf(stderr,"rank %d adjoint status %d\n",rank,status);MPI_Abort(MPI_COMM_WORLD,126);}
    lhs_local=state_dot(&forward_state,&dual,FW)+(double)forward_state.view.vz[receiver_j][receiver_i]*bar_receiver[0];rhs_local=state_dot(&input,&prev,FW)+(double)signals[1][1]*bar_signal[0];MPI_Reduce(&lhs_local,&lhs,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);MPI_Reduce(&rhs_local,&rhs,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    for(k=0;k<6;++k)for(j=1;j<=NY;++j)for(i=1;i<=NX;++i)diag_change=fmax(diag_change,fabs((double)diag[k].v[j][i]-(7.25+k)));
    snprintf(path,sizeof(path),"%s/rank_%d.bin",argv[10],rank);out=fopen(path,"wb");if(!out){fprintf(stderr,"cannot write %s: %s\n",path,strerror(errno));MPI_Abort(MPI_COMM_WORLD,127);}write_state(out,&forward_state,FW);write_state(out,&prev,FW);{float values[3]={forward_state.view.vz[receiver_j][receiver_i],(float)bar_signal[0],(float)diag_change};fwrite(values,sizeof(float),3,out);}fclose(out);
    if(rank==0){residual=fabs(lhs-rhs)/fmax(fmax(fabs(lhs),fabs(rhs)),1.0e-300);printf("{\"lhs\":%.17g,\"rhs\":%.17g,\"dot_residual\":%.17g,\"diagnostic_change\":%.9g}\n",lhs,rhs,residual,diag_change);}
    buffer_free(bl,sl);buffer_free(br,sr);buffer_free(bt,st);buffer_free(bb,sb);buffer_free(srcpos,src_storage);buffer_free(signals,signal_storage);
    for(k=0;k<6;++k)field_free(&diag[k]);field_free(&rhoi);field_free(&fipjp);field_free(&f);field_free(&dummy);volume_free(&dip);volume_free(&d);volume_free(&pp);free(hc);free(bip);free(bjm);free(cip);free(cjm);free(K);free(Kh);free(a);free(ah);free(b);free(bh);state_free(&input,FW);state_free(&forward_state,FW);state_free(&dual,FW);state_free(&work,FW);state_free(&prev,FW);MPI_Finalize();return 0;
}
