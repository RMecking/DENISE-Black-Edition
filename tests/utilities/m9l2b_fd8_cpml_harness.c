/* Additive private build; accepted M9L-2A/public translation units unchanged. */
#include "m9l2a_fd8_state_harness.c"
#include "../../src/PSV/elastic_psv_fd8_cpml_state_private.h"

void *m9l2b_test_create(int nx,int ny,int nt,int fw,float lam,float dt,
                      const float *source,int si,int sj,int nr,const int *ri,const int *rj) {
    struct denise_elastic_psv_forward_fd8 *h;
    struct denise_elastic_psv_born *c;int k;
    if(fw<0||fw>(nx-1)/2||fw>(ny-1)/2)return NULL;
    h=m9l2a_test_create(nx,ny,nt,lam,0.75f,2.0f,dt,source,si,sj,nr,ri,rj);
    if(!h)return NULL;
    c=h->core;c->fw=fw;c->cpml_enabled=fw>0;
    for(k=0;k<PROFILE_COUNT;k++) {
        free_profile(&c->profile[k]);
        if(build_profile(&c->profile[k],k<2?nx:ny,1.0f,dt,fw,k&1,
                         2.0f,0x1p-10f,2.0f,2.0f,0.125f))goto bad;
    }
    if(m9l2b_context(h))goto bad;
    return h;
bad:
    denise_elastic_psv_forward_fd8_destroy_v1(&h);return NULL;
}
int m9l2b_test_forward(void *h,int begin,int end,const float *x,float *d,size_t dc,float *y,size_t sc) {
    return m9l2b_forward(h,begin,end,x,d,dc,y,sc);
}
int m9l2b_test_transpose(void *h,int begin,int end,const double *z,const double *d,double *x,size_t sc,double *q,size_t qc) {
    return m9l2b_transpose(h,begin,end,z,d,x,sc,q,qc);
}
int m9l2b_test_profile(void *h,int kind,float *kappa,float *a,float *b,size_t count) {
    struct denise_elastic_psv_forward_fd8 *v=h;const struct pml_profile *p;
    if(m9l2b_context(v)||kind<0||kind>=PROFILE_COUNT||!kappa||!a||!b)return -1;
    p=&v->core->profile[kind];if(count<(size_t)p->length)return -1;
    copy_floats(kappa,p->kappa,(size_t)p->length);copy_floats(a,p->a,(size_t)p->length);copy_floats(b,p->b,(size_t)p->length);return 0;
}
