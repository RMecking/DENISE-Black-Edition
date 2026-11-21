/* Private experiment construction only. Accepted operators included unchanged. */
#include "m9l3_material_harness.c"
void *m9l5p_create(int nx,int ny,int nt,int fw,float dt,const float *source,
                  int si,int sj,int nr,const int *ri,const int *rj) {
    struct m9l3_material *h=NULL;struct denise_elastic_psv_born *c;size_t four;
    h=checked_calloc(1,sizeof(*h));if(!h)return NULL;
    h->state=m9l2b_test_create(nx,ny,nt,fw,1.5f,dt,source,si,sj,nr,ri,rj);
    if(!h->state)goto bad;
    c=h->state->core;
    if(checked_product(c->cells,4,&four)
       ||checked_product(four,(size_t)nt,&h->operand_count))goto bad;
    h->corner_weights=checked_calloc(four,sizeof(double));
    if(!h->corner_weights||m9l3_weights(h)||m9l3_context(h))goto bad;
    return h;
bad:
    {void *p=h;m9l3_test_destroy(&p);}return NULL;
}
int m9l5p_initial_forward(void *pointer,const float *initial,float *data,size_t dc,
                          float *final,size_t sc) {
    struct m9l3_material *h=pointer;
    if(!h||!h->state)return -1;
    return m9l2b_forward(h->state,0,h->state->core->nt,initial,data,dc,final,sc);
}
