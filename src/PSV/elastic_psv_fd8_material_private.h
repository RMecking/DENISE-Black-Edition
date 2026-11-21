/* M9L-3: private fixed-density material J/JT. Include after accepted M9L2B.
 * No public config/header change. Background/CPML/source are immutable.
 */
#ifndef DENISE_ELASTIC_PSV_FD8_MATERIAL_PRIVATE_H
#define DENISE_ELASTIC_PSV_FD8_MATERIAL_PRIVATE_H
struct m9l3_material {
    struct denise_elastic_psv_forward_fd8 *state;
    float *operands;           /* actual post-CPML primal strains, [NT][4][N] */
    double *corner_weights;   /* four material incidences per corner */
    size_t operand_count;
    int prepared;
};

static int m9l3_material_valid(const struct denise_elastic_psv_born *c,
                              const float *lambda,const float *mu) {
    size_t p;double stiffness=0.0,smax=0.0,upper;int j,i;
    if(!lambda||!mu||!fd8_environment())return 0;
    for(p=0;p<c->cells;p++) {
        double l=lambda[p],m=mu[p],row;
        if(!isfinite(l)||!isfinite(m)||m<=0.0||l+m<=0.0
           ||!isfinite(fd8_cast(2.0*m)))return 0;
        row=2.0*fabs(l)+2.0*m;if(row>stiffness)stiffness=row;
    }
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        float corner=fd8_corner(c,mu,j,i);
        if(!isfinite(corner)||corner<=0)return 0;
        if(2.0*corner>stiffness)stiffness=2.0*corner;
    }
    for(i=0;i<4;i++)smax+=fabs((double)fd8_coefficients[i]);
    smax*=2.0;
    /* Conservative private variable-material row-norm guard, not a changed
     * public Foundation CFL. ||sym grad||^2 <=2*smax^2, constitutive row bound
     * above, fixed rho=2. The 0.64 margin is stricter than the leapfrog bound4.
     * This first implementation does not claim an optimized CPML CFL theorem.
     */
    upper=2.0*(double)c->coefficient*c->coefficient*0.5*stiffness*smax*smax;
    return isfinite(upper)&&upper>0&&upper<=0.64;
}
static void m9l3_corner_cells(const struct denise_elastic_psv_born *c,
                             int j,int i,size_t p[4]) {
    int ip=fd8_material_index(i+1,c->nx),jp=fd8_material_index(j+1,c->ny);
    p[0]=cell(c,j,i);p[1]=cell(c,j,ip);p[2]=cell(c,jp,i);p[3]=cell(c,jp,ip);
}
static int m9l3_weights(struct m9l3_material *h) {
    struct denise_elastic_psv_born *c=h->state->core;int j,i,k;size_t p[4],at;
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        double sum=0.0,common;
        m9l3_corner_cells(c,j,i,p);at=cell(c,j,i);
        sum=fd8_dadd(fd8_ddiv(1.0,c->mu[p[0]]),fd8_ddiv(1.0,c->mu[p[1]]));
        sum=fd8_dadd(sum,fd8_ddiv(1.0,c->mu[p[2]]));
        sum=fd8_dadd(sum,fd8_ddiv(1.0,c->mu[p[3]]));
        /* Differentiate the actual reciprocal/sum/divide/cast graph under
         * identity tangents at rounding nodes. Use its stored FP64 sum, NOT
         * a rounded corner^2/4 shortcut. Repeated edge incidences stay repeated.
         */
        common=fd8_ddiv(4.0,fd8_dmul(sum,sum));
        for(k=0;k<4;k++) {
            double m=c->mu[p[k]],w=fd8_ddiv(common,fd8_dmul(m,m));
            if(!isfinite(w))return -1;
            h->corner_weights[4*at+(size_t)k]=w;
        }
    }
    return 0;
}
static int m9l3_context(struct m9l3_material *h) {
    struct denise_elastic_psv_born *c;
    if(!h||!h->state||!(c=h->state->core)||!c->fd8||c->free_surface
       ||!fd8_environment())return fd8_fail("M9L3 invalid context/environment");
    h->state->fault=0;
    if(!m9l3_material_valid(c,c->lambda,c->mu))return fd8_fail("M9L3 material/row-bound admission");
    return 0;
}
static double m9l3_derivative(const struct denise_elastic_psv_born *c,
                              const double *f,int j,int i,int kind) {
    double value=0.0;int m;
    for(m=1;m<=4;m++) {
        int fwd=kind==DX_FWD||kind==DY_FWD;
        int plus=fwd?m:m-1,minus=fwd?1-m:-m;
        int jp=j,ip=i,jm=j,im=i;double a=0,b=0,term;
        if(kind==DX_FWD||kind==DX_BACK){ip+=plus;im+=minus;}
        else {jp+=plus;jm+=minus;}
        if(jp>=0&&jp<c->ny&&ip>=0&&ip<c->nx)a=f[cell(c,jp,ip)];
        if(jm>=0&&jm<c->ny&&im>=0&&im<c->nx)b=f[cell(c,jm,im)];
        term=fd8_dmul((double)fd8_coefficients[m-1],fd8_dadd(a,-b));
        value=m==1?term:fd8_dadd(value,term);
    }
    return fd8_dmul((double)c->coefficient,value);
}
static double m9l3_pml(const struct denise_elastic_psv_born *c,int kind,
                        int j,int i,double q,double *m) {
    const struct pml_profile *p=profile_for(c,kind);int k=profile_index(kind,j,i);
    double u=fd8_dmul((double)p->b[k],*m),w=fd8_dmul((double)p->a[k],q);
    *m=fd8_dadd(u,w);return fd8_dadd(fd8_ddiv(q,(double)p->kappa[k]),*m);
}
static int m9l3_prepare(struct m9l3_material *h,float *data,size_t capacity) {
    struct denise_elastic_psv_born *c;float *q=NULL,*tmp=NULL;
    if(m9l3_context(h))return -1;
    c=h->state->core;
    if(!data||capacity<c->data_count)return fd8_fail("M9L3 background capacity");
    q=checked_calloc(h->operand_count,sizeof(float));tmp=checked_calloc(c->data_count,sizeof(float));
    if(!q||!tmp)goto bad;
    if(run_nonlinear(c,c->lambda,c->mu,tmp,q,0,0))goto bad;
    free(h->operands);h->operands=q;h->prepared=1;
    memcpy(data,tmp,c->data_count*sizeof(float));free(tmp);return 0;
bad:
    free(q);free(tmp);return fd8_fail("M9L3 background allocation/nonfinite");
}
static int m9l3_nonlinear(struct m9l3_material *h,const float *lambda,
                         const float *mu,float *data,size_t capacity) {
    struct denise_elastic_psv_born *c;float *tmp;int status;
    if(m9l3_context(h))return -1;
    c=h->state->core;
    if(!data||capacity<c->data_count||!m9l3_material_valid(c,lambda,mu))
        return fd8_fail("M9L3 invalid trial material/capacity");
    tmp=checked_calloc(c->data_count,sizeof(float));if(!tmp)return fd8_fail("M9L3 trial allocation");
    status=run_nonlinear(c,lambda,mu,tmp,NULL,0,0);
    if(!status)memcpy(data,tmp,c->data_count*sizeof(float));
    free(tmp);
    return status?fd8_fail("M9L3 trial nonfinite"):0;
}

static int m9l3_apply_j(struct m9l3_material *h,const double *dl,const double *dm,
                       double *data,size_t capacity) {
    struct denise_elastic_psv_born *c;double *state=NULL,*dc=NULL,*tmp=NULL,*v[13];
    size_t n,count,p,indices[4];int k,j,i,t,r;
    if(m9l3_context(h))return -1;
    c=h->state->core;n=c->cells;
    if(!h->prepared||!data||capacity<c->data_count||!m9l2a_finite_d(dl,n)
       ||!m9l2a_finite_d(dm,n)||checked_product(n,13,&count))return fd8_fail("M9L3 invalid J inputs/unprepared");
    state=checked_calloc(count,sizeof(double));dc=checked_calloc(n,sizeof(double));tmp=checked_calloc(c->data_count,sizeof(double));
    if(!state||!dc||!tmp)goto bad;
    for(k=0;k<13;k++)v[k]=state+(size_t)k*n;
    for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
        p=cell(c,j,i);m9l3_corner_cells(c,j,i,indices);
        for(k=0;k<4;k++)dc[p]=fd8_dadd(dc[p],fd8_dmul(h->corner_weights[4*p+(size_t)k],dm[indices[k]]));
    }
    for(t=0;t<c->nt;t++) {
        const float *bg=h->operands+(size_t)t*4*n;
        for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
            double xx,xyy,xyx,yy;p=cell(c,j,i);
            xx=m9l3_pml(c,PSXX,j,i,m9l3_derivative(c,v[SXX],j,i,DX_FWD),&v[5+PSXX][p]);
            xyy=m9l3_pml(c,PSXYY,j,i,m9l3_derivative(c,v[SXY],j,i,DY_BACK),&v[5+PSXYY][p]);
            xyx=m9l3_pml(c,PSXYX,j,i,m9l3_derivative(c,v[SXY],j,i,DX_BACK),&v[5+PSXYX][p]);
            yy=m9l3_pml(c,PSYY,j,i,m9l3_derivative(c,v[SYY],j,i,DY_FWD),&v[5+PSYY][p]);
            v[VX][p]=fd8_dadd(v[VX][p],fd8_dmul((double)c->invrho_x[p],fd8_dadd(xx,xyy)));
            v[VY][p]=fd8_dadd(v[VY][p],fd8_dmul((double)c->invrho_y[p],fd8_dadd(xyx,yy)));
        }
        for(r=0;r<c->receiver_count;r++) {
            size_t slot=((size_t)t*c->receiver_count+(size_t)r)*2;
            p=cell(c,c->receiver_j[r],c->receiver_i[r]);tmp[slot]=v[VX][p];tmp[slot+1]=v[VY][p];
        }
        for(j=0;j<c->ny;j++)for(i=0;i<c->nx;i++) {
            double q0,q1,q2,q3,lt,sx,sy,sxy,two,bdiv,bshear;
            p=cell(c,j,i);
            q0=m9l3_pml(c,PVXX,j,i,m9l3_derivative(c,v[VX],j,i,DX_BACK),&v[5+PVXX][p]);
            q1=m9l3_pml(c,PVYX,j,i,m9l3_derivative(c,v[VY],j,i,DX_FWD),&v[5+PVYX][p]);
            q2=m9l3_pml(c,PVXY,j,i,m9l3_derivative(c,v[VX],j,i,DY_FWD),&v[5+PVXY][p]);
            q3=m9l3_pml(c,PVYY,j,i,m9l3_derivative(c,v[VY],j,i,DY_BACK),&v[5+PVYY][p]);
            bdiv=(double)fd8_add(c,bg[p],bg[3*n+p]);bshear=(double)fd8_add(c,bg[n+p],bg[2*n+p]);
            two=(double)fd8_mul(c,2.0f,c->mu[p]);
            lt=fd8_dadd(fd8_dmul(dl[p],bdiv),fd8_dmul((double)c->lambda[p],fd8_dadd(q0,q3)));
            sx=fd8_dadd(lt,fd8_dadd(fd8_dmul(fd8_dmul(2.0,dm[p]),(double)bg[p]),fd8_dmul(two,q0)));
            sy=fd8_dadd(lt,fd8_dadd(fd8_dmul(fd8_dmul(2.0,dm[p]),(double)bg[3*n+p]),fd8_dmul(two,q3)));
            sxy=fd8_dadd(fd8_dmul(dc[p],bshear),fd8_dmul((double)c->mu_corner[p],fd8_dadd(q1,q2)));
            v[SXX][p]=fd8_dadd(v[SXX][p],sx);v[SYY][p]=fd8_dadd(v[SYY][p],sy);v[SXY][p]=fd8_dadd(v[SXY][p],sxy);
        }
        if(h->state->fault||!m9l2a_finite_d(state,count))goto bad;
    }
    if(!m9l2a_finite_d(tmp,c->data_count))goto bad;
    memcpy(data,tmp,c->data_count*sizeof(double));free(state);free(dc);free(tmp);return 0;
bad:
    free(state);free(dc);free(tmp);return fd8_fail("M9L3 J allocation/nonfinite");
}
static int m9l3_apply_jt(struct m9l3_material *h,const double *data,
                        double *gl,double *gm,size_t capacity) {
    struct denise_elastic_psv_born *c;double *bar=NULL,*l=NULL,*m=NULL,*src=NULL;
    size_t n,count,p,indices[4];int j,i,k,t;
    if(m9l3_context(h))return -1;
    c=h->state->core;n=c->cells;
    if(!h->prepared||!gl||!gm||capacity<n||!m9l2a_finite_d(data,c->data_count)
       ||m9l2a_overlap(gl,n*sizeof(double),gm,n*sizeof(double))||checked_product(n,13,&count))
        return fd8_fail("M9L3 invalid JT inputs/unprepared");
    bar=checked_calloc(count,sizeof(double));l=checked_calloc(n,sizeof(double));m=checked_calloc(n,sizeof(double));src=checked_calloc((size_t)c->nt,sizeof(double));
    if(!bar||!l||!m||!src)goto bad;
    for(t=c->nt-1;t>=0;t--) {
        const float *bg=h->operands+(size_t)t*4*n;
        for(j=c->ny-1;j>=0;j--)for(i=c->nx-1;i>=0;i--) {
            double bx,by,bxy,div,shear,gcorner,normal;
            p=cell(c,j,i);bx=bar[2*n+p];by=bar[3*n+p];bxy=bar[4*n+p];
            div=(double)fd8_add(c,bg[p],bg[3*n+p]);shear=(double)fd8_add(c,bg[n+p],bg[2*n+p]);
            l[p]=fd8_dadd(l[p],fd8_dmul(fd8_dadd(bx,by),div));
            normal=fd8_dmul(2.0,fd8_dadd(fd8_dmul(bx,(double)bg[p]),fd8_dmul(by,(double)bg[3*n+p])));
            m[p]=fd8_dadd(m[p],normal);gcorner=fd8_dmul(bxy,shear);
            m9l3_corner_cells(c,j,i,indices);
            for(k=3;k>=0;k--)m[indices[k]]=fd8_dadd(m[indices[k]],fd8_dmul(h->corner_weights[4*p+(size_t)k],gcorner));
        }
        /* Reuse the independently accepted complete CPML state reverse.
         * It is algebraically variable-map capable; only its original public
         * context admission was homogeneous. New private admission above owns
         * the scoped heterogeneous extension, without changing that old API.
         */
        if(m9l2b_reverse_step(c,bar,t,data,src))goto bad;
    }
    if(h->state->fault||!m9l2a_finite_d(l,n)||!m9l2a_finite_d(m,n))goto bad;
    memcpy(gl,l,n*sizeof(double));memcpy(gm,m,n*sizeof(double));
    free(bar);free(l);free(m);free(src);return 0;
bad:
    free(bar);free(l);free(m);free(src);return fd8_fail("M9L3 JT allocation/nonfinite");
}
#endif
