/* M9L-2B private 13-field state: five elastic fields then PSI_COUNT memories.
 * Included after the accepted M9L-2A translation unit. No public ABI change.
 */
#ifndef DENISE_ELASTIC_PSV_FD8_CPML_STATE_PRIVATE_H
#define DENISE_ELASTIC_PSV_FD8_CPML_STATE_PRIVATE_H
static int m9l2b_context(struct denise_elastic_psv_forward_fd8 *h) {
    struct denise_elastic_psv_born *c;int k,i;
    if(!h||!(c=h->core)||!c->fd8||c->free_surface||c->nx<9||c->ny<9
       ||!fd8_environment())return fd8_fail("M9L2B unsupported context/environment");
    h->fault=0;
    if(!fd8_admission(c,c->lambda,c->mu))return fd8_fail("M9L2B material admission");
    for(k=0;k<PROFILE_COUNT;k++)for(i=0;i<c->profile[k].length;i++) {
        const struct pml_profile *p=&c->profile[k];
        if(!isfinite(p->kappa[i])||p->kappa[i]<1||!isfinite(p->a[i])
           ||p->a[i]>0||!isfinite(p->b[i])||p->b[i]<0||p->b[i]>1)
            return fd8_fail("M9L2B invalid fixed CPML profile");
    }
    return 0;
}

/* For q=gamma*D(field), old m:
 * m'=RN32(RN32(b*m)+RN32(a*q)); z=RN32(RN32(q/kappa)+m').
 * Reverse under the fixed-coefficient arithmetic-graph convention:
 * total=bar(m')+bar(z), bar(q)=bar(z)/kappa+a*total, bar(m)=b*total.
 * Division and each multiply/add are separate binary64 rounding barriers.
 * The output memory cotangent must join bar(z) BEFORE it is overwritten.
 */
static double m9l2b_pml_reverse(const struct denise_elastic_psv_born *c,
        int kind,int j,int i,double zbar,double *mbar) {
    const struct pml_profile *p=profile_for(c,kind);int k=profile_index(kind,j,i);
    double total=fd8_dadd(*mbar,zbar);
    double qbar=fd8_dadd(fd8_ddiv(zbar,(double)p->kappa[k]),
                        fd8_dmul((double)p->a[k],total));
    *mbar=fd8_dmul((double)p->b[k],total);return qbar;
}

static int m9l2b_forward(struct denise_elastic_psv_forward_fd8 *h,
        int begin,int end,const float *initial,float *data,size_t dc,
        float *final,size_t sc) {
    struct denise_elastic_psv_born *c;float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4];
    float *trace=NULL;size_t count,bytes;int k,t,status=-1;
    if(m9l2b_context(h))return -1;
    c=h->core;
    if(checked_product(c->cells,13,&count)||checked_product(count,sizeof(float),&bytes)
       ||!data||!final||begin<0||end<begin||end>c->nt||sc<count||dc<c->data_count
       ||!m9l2a_finite_f(initial,count)
       ||m9l2a_overlap(data,c->data_count*sizeof(float),final,bytes))
        return fd8_fail("M9L2B invalid Forward arguments");
    if(allocate_forward(f,psi,q,c->cells)) {
        free_forward(f,psi,q);return fd8_fail("M9L2B allocation failure");
    }
    trace=checked_calloc(c->data_count,sizeof(*trace));if(!trace)goto done;
    for(k=0;k<5;k++)copy_floats(f[k],initial+(size_t)k*c->cells,c->cells);
    for(k=0;k<8;k++)copy_floats(psi[k],initial+(size_t)(5+k)*c->cells,c->cells);
    for(t=begin;t<end;t++) {
        forward_timestep(c,f,psi,q,c->lambda,c->mu,c->mu_corner,t,trace,NULL,0);
        if(h->fault)goto done;
    }
    for(k=0;k<5;k++)if(!m9l2a_finite_f(f[k],c->cells))goto done;
    for(k=0;k<8;k++)if(!m9l2a_finite_f(psi[k],c->cells))goto done;
    memcpy(data,trace,c->data_count*sizeof(float));
    for(k=0;k<5;k++)copy_floats(final+(size_t)k*c->cells,f[k],c->cells);
    for(k=0;k<8;k++)copy_floats(final+(size_t)(5+k)*c->cells,psi[k],c->cells);
    status=0;
done:
    free(trace);free_forward(f,psi,q);
    return status?fd8_fail("M9L2B nonfinite Forward/allocation"):0;
}

static int m9l2b_reverse_step(const struct denise_elastic_psv_born *c,
        double *bar,int t,const double *data,double *source) {
    double *v[13];int k,j,i,r;size_t p;
    for(k=0;k<13;k++)v[k]=bar+(size_t)k*c->cells;
    p=cell(c,c->source_j,c->source_i);
    source[t]=fd8_dadd(v[SYY][p],v[SXX][p]);
    for(j=c->ny-1;j>=0;--j)for(i=c->nx-1;i>=0;--i) {
        double bx,by,xy,two,shared,qxx,qyy,qxy,qyx;
        p=cell(c,j,i);bx=v[SXX][p];by=v[SYY][p];xy=v[SXY][p];
        two=(double)fd8_mul(c,2.0f,c->mu[p]);
        shared=fd8_dmul((double)c->lambda[p],fd8_dadd(by,bx));
        qxx=fd8_dadd(fd8_dmul(two,bx),shared);
        qyy=fd8_dadd(fd8_dmul(two,by),shared);
        qxy=fd8_dmul((double)c->mu_corner[p],xy);qyx=qxy;
        qyy=m9l2b_pml_reverse(c,PVYY,j,i,qyy,&v[5+PVYY][p]);
        qxy=m9l2b_pml_reverse(c,PVXY,j,i,qxy,&v[5+PVXY][p]);
        qyx=m9l2b_pml_reverse(c,PVYX,j,i,qyx,&v[5+PVYX][p]);
        qxx=m9l2b_pml_reverse(c,PVXX,j,i,qxx,&v[5+PVXX][p]);
        m9l2a_scatter(c,v[VY],j,i,DY_BACK,fd8_dmul((double)c->coefficient,qyy));
        m9l2a_scatter(c,v[VX],j,i,DY_FWD,fd8_dmul((double)c->coefficient,qxy));
        m9l2a_scatter(c,v[VY],j,i,DX_FWD,fd8_dmul((double)c->coefficient,qyx));
        m9l2a_scatter(c,v[VX],j,i,DX_BACK,fd8_dmul((double)c->coefficient,qxx));
    }
    for(r=c->receiver_count-1;r>=0;--r) {
        size_t slot=((size_t)t*c->receiver_count+(size_t)r)*2;
        p=cell(c,c->receiver_j[r],c->receiver_i[r]);
        v[VY][p]=fd8_dadd(v[VY][p],data[slot+1]);
        v[VX][p]=fd8_dadd(v[VX][p],data[slot]);
    }
    for(j=c->ny-1;j>=0;--j)for(i=c->nx-1;i>=0;--i) {
        double x,y,qyy,qxyx,qxyy,qxx;
        p=cell(c,j,i);
        y=fd8_dmul((double)c->invrho_y[p],v[VY][p]);
        x=fd8_dmul((double)c->invrho_x[p],v[VX][p]);
        qyy=m9l2b_pml_reverse(c,PSYY,j,i,y,&v[5+PSYY][p]);
        qxyx=m9l2b_pml_reverse(c,PSXYX,j,i,y,&v[5+PSXYX][p]);
        qxyy=m9l2b_pml_reverse(c,PSXYY,j,i,x,&v[5+PSXYY][p]);
        qxx=m9l2b_pml_reverse(c,PSXX,j,i,x,&v[5+PSXX][p]);
        m9l2a_scatter(c,v[SYY],j,i,DY_FWD,fd8_dmul((double)c->coefficient,qyy));
        m9l2a_scatter(c,v[SXY],j,i,DX_BACK,fd8_dmul((double)c->coefficient,qxyx));
        m9l2a_scatter(c,v[SXY],j,i,DY_BACK,fd8_dmul((double)c->coefficient,qxyy));
        m9l2a_scatter(c,v[SXX],j,i,DX_FWD,fd8_dmul((double)c->coefficient,qxx));
    }
    return *c->fd8_fault||!m9l2a_finite_d(bar,13*c->cells)||!isfinite(source[t])?-1:0;
}
static int m9l2b_transpose(struct denise_elastic_psv_forward_fd8 *h,
        int begin,int end,const double *terminal,const double *data,
        double *initial,size_t sc,double *source,size_t qc) {
    struct denise_elastic_psv_born *c;double *bar=NULL,*src=NULL;
    size_t count,bytes;int t;
    if(m9l2b_context(h))return -1;
    c=h->core;
    if(checked_product(13,c->cells,&count)||checked_product(count,sizeof(double),&bytes)
       ||!initial||!source||begin<0||end<begin||end>c->nt||sc<count||qc<(size_t)c->nt
       ||!m9l2a_finite_d(terminal,count)||!m9l2a_finite_d(data,c->data_count)
       ||m9l2a_overlap(initial,bytes,source,(size_t)c->nt*sizeof(double)))
        return fd8_fail("M9L2B invalid reverse arguments");
    bar=checked_calloc(count,sizeof(double));src=checked_calloc((size_t)c->nt,sizeof(double));
    if(!bar||!src){free(bar);free(src);return fd8_fail("M9L2B reverse allocation");}
    memcpy(bar,terminal,bytes);
    for(t=end-1;t>=begin;--t)if(m9l2b_reverse_step(c,bar,t,data,src)) {
        free(bar);free(src);return fd8_fail("M9L2B nonfinite reverse");
    }
    memcpy(initial,bar,bytes);memcpy(source,src,(size_t)c->nt*sizeof(double));
    free(bar);free(src);return 0;
}
#endif
