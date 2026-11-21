/* M9L-2A private implementation, included only after elastic_psv_born.c.
 * No public/exported declarations. State layout: [VX,VY,SXX,SYY,SXY][ny][nx].
 * Fixed stored material maps; no material VJP. Five dynamic fields, no CPML.
 * Transpose is reverse AD of the stored-coefficient arithmetic graph, with
 * rounding treated as identity in the discrete tangent convention. It is NOT
 * a derivative of the discontinuous bit-to-bit IEEE rounding function.
 */
#ifndef DENISE_ELASTIC_PSV_FD8_STATE_PRIVATE_H
#define DENISE_ELASTIC_PSV_FD8_STATE_PRIVATE_H

struct m9l2a_checkpoint {
    const struct denise_elastic_psv_forward_fd8 *owner;
    int layout, nx, ny, next_timestep;
    size_t count;
    float *state;
};

static int m9l2a_context(struct denise_elastic_psv_forward_fd8 *h) {
    struct denise_elastic_psv_born *c;
    int k,i;
    if(!h||!(c=h->core)||!c->fd8||!c->fd8_fault||c->nx<9||c->ny<9
       ||c->free_surface||c->fw||c->cpml_enabled||c->nt<1)
        return fd8_fail("M9L2A unsupported private state context");
    if(!fd8_environment())return fd8_fail("M9L2A requires RNE/gradual underflow");
    for(k=0;k<PROFILE_COUNT;k++)for(i=0;i<c->profile[k].length;i++)
        if(c->profile[k].kappa[i]!=1.0f||c->profile[k].a[i]!=0.0f||c->profile[k].b[i]!=1.0f)
            return fd8_fail("M9L2A requires identity, inactive CPML maps");
    h->fault=0;
    if(!fd8_admission(c,c->lambda,c->mu))
        return fd8_fail("M9L2A immutable material/CFL admission failed");
    return 0;
}

static int m9l2a_finite_f(const float *x,size_t n) {
    size_t p;if(!x)return 0;
    for(p=0;p<n;p++)if(!isfinite(x[p]))return 0;
    return 1;
}
static int m9l2a_finite_d(const double *x,size_t n) {
    size_t p;if(!x)return 0;
    for(p=0;p<n;p++)if(!isfinite(x[p]))return 0;
    return 1;
}
static int m9l2a_overlap(const void *a,size_t na,const void *b,size_t nb) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    if(na>UINTPTR_MAX-x||nb>UINTPTR_MAX-y)return 1;
    return x<y+nb&&y<x+na;
}

/* Signed finite incidence scatter. No wrap or material extrapolation here:
 * the transpose of constant zero FIELD extension discards exterior targets.
 * Reverse order mirrors result=sum_{m=1..4} a_m*(plus-minus).
 */
static void m9l2a_scatter(const struct denise_elastic_psv_born *c,double *out,
                          int j,int i,int kind,double value) {
    int m;
    for(m=4;m>=1;--m) {
        int fwd=kind==DX_FWD||kind==DY_FWD;
        int plus=fwd?m:m-1,minus=fwd?1-m:-m;
        int jp=j,ip=i,jm=j,im=i;
        double v=fd8_dmul((double)fd8_coefficients[m-1],value);
        if(kind==DX_BACK||kind==DX_FWD){ip+=plus;im+=minus;}
        else {jp+=plus;jm+=minus;}
        if(jm>=0&&jm<c->ny&&im>=0&&im<c->nx) {
            size_t p=cell(c,jm,im);out[p]=fd8_dadd(out[p],-v);
        }
        if(jp>=0&&jp<c->ny&&ip>=0&&ip<c->nx) {
            size_t p=cell(c,jp,ip);out[p]=fd8_dadd(out[p],v);
        }
    }
}

static int m9l2a_derivative_transpose(
        struct denise_elastic_psv_forward_fd8 *h,int kind,
        const double *input,double *output,size_t capacity) {
    struct denise_elastic_psv_born *c;double *tmp;int i,j;
    if(m9l2a_context(h))return -1;
    c=h->core;
    if(kind<DX_BACK||kind>DY_FWD||capacity<c->cells||!output
       ||!m9l2a_finite_d(input,c->cells))return fd8_fail("M9L2A derivative arguments");
    tmp=checked_calloc(c->cells,sizeof(*tmp));
    if(!tmp)return fd8_fail("M9L2A derivative allocation failed");
    for(j=c->ny-1;j>=0;--j)for(i=c->nx-1;i>=0;--i)
        m9l2a_scatter(c,tmp,j,i,kind,input[cell(c,j,i)]);
    if(!m9l2a_finite_d(tmp,c->cells)) {
        free(tmp);return fd8_fail("M9L2A derivative cotangent overflow");
    }
    memcpy(output,tmp,c->cells*sizeof(*tmp));free(tmp);return 0;
}

/* Forward state interval [begin,end). Checkpoint boundary is post source of
 * timestep begin-1, equivalently before momentum(begin). Uses the EXACT modern
 * momentum -> gather -> strains -> stress -> prepared source path and ordinary
 * zero FIELD fetch; no analytical exterior hooks are installed by this unit.
 * data/final outputs are published together only after complete success.
 */
static int m9l2a_forward(struct denise_elastic_psv_forward_fd8 *h,
        int begin,int end,const float *initial,float *data,size_t data_capacity,
        float *final,size_t state_capacity) {
    struct denise_elastic_psv_born *c;
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4],*trace=NULL;
    size_t count,bytes;int k,t,status=-1;
    if(m9l2a_context(h))return -1;
    c=h->core;
    if(checked_product(c->cells,5,&count)||checked_product(count,sizeof(float),&bytes)
       ||!initial||!data||!final||begin<0||end<begin||end>c->nt
       ||state_capacity<count||data_capacity<c->data_count
       ||!m9l2a_finite_f(initial,count)
       ||m9l2a_overlap(data,c->data_count*sizeof(float),final,bytes))
        return fd8_fail("M9L2A invalid state interval/output capacity");
    if(allocate_forward(f,psi,q,c->cells)) {
        free_forward(f,psi,q);return fd8_fail("M9L2A forward state allocation");
    }
    trace=checked_calloc(c->data_count,sizeof(*trace));
    if(!trace)goto done;
    for(k=0;k<5;k++)copy_floats(f[k],initial+(size_t)k*c->cells,c->cells);
    for(t=begin;t<end;t++) {
        forward_timestep(c,f,psi,q,c->lambda,c->mu,c->mu_corner,t,trace,NULL,0);
        if(h->fault)goto done;
    }
    /* Check final state too, including an empty interval. */
    for(k=0;k<5;k++)if(!m9l2a_finite_f(f[k],c->cells))goto done;
    memcpy(data,trace,c->data_count*sizeof(*trace));
    for(k=0;k<5;k++)copy_floats(final+(size_t)k*c->cells,f[k],c->cells);
    status=0;
done:
    free(trace);free_forward(f,psi,q);
    return status?fd8_fail("M9L2A forward allocation/nonfinite result"):0;
}

/* One whole reverse timestep, on five carried double cotangents.
 * S(out) -> source(t)^T -> stress^T -> strain^T -> gather(t)^T -> momentum^T.
 * The velocity carry and three stress carries are identities. Source sample
 * t is injected AFTER sampling t; its data-only cotangent is therefore zero
 * at the final t. Duplicate receivers accumulate, in reverse receiver order.
 */
static int m9l2a_reverse_step(const struct denise_elastic_psv_born *c,
        double *bar,int t,const double *data_bar,double *source_bar) {
    double *v[5];int k,j,i,r;size_t p;
    for(k=0;k<5;k++)v[k]=bar+(size_t)k*c->cells;
    p=cell(c,c->source_j,c->source_i);
    source_bar[t]=fd8_dadd(v[SYY][p],v[SXX][p]);
    for(j=c->ny-1;j>=0;--j)for(i=c->nx-1;i>=0;--i) {
        double bx,by,bxy,shared,two,qxx,qyy,qyx,qxy;
        p=cell(c,j,i);bx=v[SXX][p];by=v[SYY][p];bxy=v[SXY][p];
        /* two is the stored FP32 intermediate in the true stress graph. */
        two=(double)fd8_mul(c,2.0f,c->mu[p]);
        shared=fd8_dmul((double)c->lambda[p],fd8_dadd(by,bx));
        qxx=fd8_dadd(fd8_dmul(two,bx),shared);
        qyy=fd8_dadd(fd8_dmul(two,by),shared);
        qyx=fd8_dmul((double)c->mu_corner[p],bxy);qxy=qyx;
        m9l2a_scatter(c,v[VY],j,i,DY_BACK,fd8_dmul((double)c->coefficient,qyy));
        m9l2a_scatter(c,v[VX],j,i,DY_FWD,fd8_dmul((double)c->coefficient,qxy));
        m9l2a_scatter(c,v[VY],j,i,DX_FWD,fd8_dmul((double)c->coefficient,qyx));
        m9l2a_scatter(c,v[VX],j,i,DX_BACK,fd8_dmul((double)c->coefficient,qxx));
    }
    for(r=c->receiver_count-1;r>=0;--r) {
        size_t slot=((size_t)t*c->receiver_count+(size_t)r)*2;
        p=cell(c,c->receiver_j[r],c->receiver_i[r]);
        v[VY][p]=fd8_dadd(v[VY][p],data_bar[slot+1]);
        v[VX][p]=fd8_dadd(v[VX][p],data_bar[slot]);
    }
    for(j=c->ny-1;j>=0;--j)for(i=c->nx-1;i>=0;--i) {
        double x,y;
        p=cell(c,j,i);
        y=fd8_dmul((double)c->invrho_y[p],v[VY][p]);
        x=fd8_dmul((double)c->invrho_x[p],v[VX][p]);
        y=fd8_dmul((double)c->coefficient,y);
        x=fd8_dmul((double)c->coefficient,x);
        m9l2a_scatter(c,v[SYY],j,i,DY_FWD,y);
        m9l2a_scatter(c,v[SXY],j,i,DX_BACK,y);
        m9l2a_scatter(c,v[SXY],j,i,DY_BACK,x);
        m9l2a_scatter(c,v[SXX],j,i,DX_FWD,x);
    }
    return (*c->fd8_fault||!m9l2a_finite_d(bar,5*c->cells)
            ||!isfinite(source_bar[t]))?-1:0;
}

/* All times reversed; no Forward trajectory is required for a fixed-material
 * linear state map. Memory is O(5*N + NT), NOT O(NT*N); a future material VJP
 * must explicitly request/reconstruct the necessary Forward operands instead.
 */
static int m9l2a_transpose(struct denise_elastic_psv_forward_fd8 *h,
        int begin,int end,const double *final_bar,const double *data_bar,
        double *initial_bar,size_t state_capacity,double *source_bar,size_t source_capacity) {
    struct denise_elastic_psv_born *c;double *bar=NULL,*src=NULL;
    size_t n,bytes;int t;
    if(m9l2a_context(h))return -1;
    c=h->core;
    if(checked_product(5,c->cells,&n)||checked_product(n,sizeof(double),&bytes)
       ||begin<0||end<begin||end>c->nt||state_capacity<n||source_capacity<(size_t)c->nt
       ||!initial_bar||!source_bar||!m9l2a_finite_d(final_bar,n)
       ||!m9l2a_finite_d(data_bar,c->data_count)
       ||m9l2a_overlap(initial_bar,bytes,source_bar,(size_t)c->nt*sizeof(double)))
        return fd8_fail("M9L2A invalid transpose inputs/capacity");
    bar=checked_calloc(n,sizeof(*bar));src=checked_calloc((size_t)c->nt,sizeof(*src));
    if(!bar||!src){free(bar);free(src);return fd8_fail("M9L2A cotangent allocation");}
    memcpy(bar,final_bar,bytes);
    for(t=end-1;t>=begin;--t)if(m9l2a_reverse_step(c,bar,t,data_bar,src)) {
        free(src);free(bar);return fd8_fail("M9L2A nonfinite reverse graph");
    }
    memcpy(initial_bar,bar,bytes);memcpy(source_bar,src,(size_t)c->nt*sizeof(*src));
    free(src);free(bar);return 0;
}

static void m9l2a_checkpoint_destroy(struct m9l2a_checkpoint **pointer) {
    if(pointer&&*pointer){free((*pointer)->state);free(*pointer);*pointer=NULL;}
}
static int m9l2a_checkpoint_capture(struct denise_elastic_psv_forward_fd8 *h,
        int next,const float *state,struct m9l2a_checkpoint **out) {
    struct m9l2a_checkpoint *q;size_t count;
    if(!out)return fd8_fail("M9L2A checkpoint output is null");
    *out=NULL;
    if(m9l2a_context(h))return -1;
    if(checked_product(5,h->core->cells,&count)||next<0||next>h->core->nt
       ||!m9l2a_finite_f(state,count))return fd8_fail("M9L2A checkpoint state invalid");
    q=checked_calloc(1,sizeof(*q));if(!q)return fd8_fail("M9L2A checkpoint allocation");
    q->state=checked_calloc(count,sizeof(float));
    if(!q->state){free(q);return fd8_fail("M9L2A checkpoint payload allocation");}
    q->owner=h;q->layout=1;q->nx=h->core->nx;q->ny=h->core->ny;
    q->next_timestep=next;q->count=count;copy_floats(q->state,state,count);
    *out=q;return 0;
}
static int m9l2a_checkpoint_restore(struct denise_elastic_psv_forward_fd8 *h,
        const struct m9l2a_checkpoint *q,float *state,size_t capacity,int *next) {
    if(m9l2a_context(h))return -1;
    if(!q||!state||!next||q->owner!=h||q->layout!=1||q->nx!=h->core->nx||q->ny!=h->core->ny
       ||q->count!=5*h->core->cells||capacity<q->count||q->next_timestep<0
       ||q->next_timestep>h->core->nt||!m9l2a_finite_f(q->state,q->count))
        return fd8_fail("M9L2A checkpoint identity/layout mismatch");
    copy_floats(state,q->state,q->count);*next=q->next_timestep;return 0;
}
#endif
