/* Additive private material build; accepted public/state sources unchanged. */
#include "m9l2b_fd8_cpml_harness.c"
#include "../../src/PSV/elastic_psv_fd8_material_private.h"

void m9l3_test_destroy(void **pointer) {
    struct m9l3_material *h;
    if(!pointer||!*pointer)return;
    h=*pointer;free(h->operands);free(h->corner_weights);
    denise_elastic_psv_forward_fd8_destroy_v1(&h->state);free(h);*pointer=NULL;
}
void *m9l3_test_create(int nx,int ny,int nt,int fw,const float *lambda,
                      const float *mu,const float *source,int si,int sj,
                      int nr,const int *ri,const int *rj) {
    struct m9l3_material *h=NULL;struct denise_elastic_psv_born *c;size_t four;
    h=checked_calloc(1,sizeof(*h));if(!h)return NULL;
    h->state=m9l2b_test_create(nx,ny,nt,fw,1.5f,0.03125f,source,si,sj,nr,ri,rj);
    if(!h->state)goto bad;
    c=h->state->core;
    if(checked_product(c->cells,4,&four)||checked_product(four,(size_t)nt,&h->operand_count)
       ||!m9l3_material_valid(c,lambda,mu))goto bad;
    copy_floats(c->lambda,lambda,c->cells);copy_floats(c->mu,mu,c->cells);
    build_material_maps(c);
    h->corner_weights=checked_calloc(four,sizeof(double));
    if(!h->corner_weights||m9l3_weights(h)||m9l3_context(h))goto bad;
    return h;
bad:
    {void *tmp=h;m9l3_test_destroy(&tmp);}return NULL;
}
int m9l3_test_prepare(void *h,float *d,size_t count) {return m9l3_prepare(h,d,count);}
int m9l3_test_nonlinear(void *h,const float *l,const float *m,float *d,size_t count) {return m9l3_nonlinear(h,l,m,d,count);}
int m9l3_test_j(void *h,const double *l,const double *m,double *d,size_t count) {return m9l3_apply_j(h,l,m,d,count);}
int m9l3_test_jt(void *h,const double *d,double *l,double *m,size_t count) {return m9l3_apply_jt(h,d,l,m,count);}
int m9l3_test_operands(void *pointer,float *out,size_t count) {
    struct m9l3_material *h=pointer;
    if(m9l3_context(h)||!h->prepared||!out||count<h->operand_count)return -1;
    copy_floats(out,h->operands,h->operand_count);return 0;
}
int m9l3_test_weights(void *pointer,double *out,size_t count) {
    struct m9l3_material *h=pointer;size_t n;
    if(m9l3_context(h))return -1;
    n=4*h->state->core->cells;
    if(!out||count<n)return -1;
    memcpy(out,h->corner_weights,n*sizeof(double));return 0;
}
