/* Capture final arrays from the unmodified public CPU J at their destruction.
 * Allocation interception is test-only; no duplicated tangent numerics. */
#include <stdlib.h>
#include <string.h>
static int capturing,index_capture;
static void *captured[18];
static float *final_state,*final_q,*final_corner;
static size_t capture_cells;
static void *capture_calloc(size_t n,size_t w) {
    void *p=calloc(n,w);
    if(capturing && index_capture<18)captured[index_capture++]=p;
    return p;
}
static void capture_free(void *p) {
    int k;
    if(capturing && p)for(k=0;k<18;k++)if(p==captured[k]) {
        float *out=k<13?final_state+(size_t)k*capture_cells:
                   (k<17?final_q+(size_t)(k-13)*capture_cells:final_corner);
        if(out)memcpy(out,p,capture_cells*sizeof(float));
        captured[k]=NULL;
    }
    free(p);
}
#define calloc capture_calloc
#define free capture_free
#include "m9e1_cuda_elastic_psv_forward_harness.c"
#undef calloc
#undef free
int m9e2_cpu_ghost(const struct denise_elastic_psv_born_config *cfg,int kind,
                    const float *field,const float *q,const float *bg,
                    const float *dl,const float *dm,float *out) {
    struct denise_elastic_psv_born *c=NULL;float *qs[4];int m,i;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    for(i=0;i<4;i++)qs[i]=(float *)q+(size_t)i*c->cells;
    for(m=1;m<=2;m++)for(i=0;i<c->nx;i++)
        out[(size_t)(2-m)*c->nx+i]=surface_value(c,field,-m,i,kind,qs,c->lambda,c->mu,bg,dl,dm);
    denise_elastic_psv_born_destroy(&c);return 0;
}
int m9e2_cpu_j(const struct denise_elastic_psv_born_config *cfg,
              const float *dl,const float *dm,float *data,float *state,float *q,float *corner) {
    struct denise_elastic_psv_born *c=NULL;int rc;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    if(denise_elastic_psv_born_prepare(c,NULL)){denise_elastic_psv_born_destroy(&c);return -1;}
    capture_cells=c->cells;final_state=state;final_q=q;final_corner=corner;
    memset(captured,0,sizeof(captured));index_capture=0;capturing=1;
    rc=denise_elastic_psv_born_apply_j(c,dl,dm,data);
    capturing=0;if(index_capture!=18)rc=-1;
    denise_elastic_psv_born_destroy(&c);return rc;
}
int m9e2_cpu_data(const struct denise_elastic_psv_born_config *cfg,float *data) {
    struct denise_elastic_psv_born *c=NULL;int rc;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    rc=denise_elastic_psv_born_nonlinear(c,cfg->lambda,cfg->mu,data);
    denise_elastic_psv_born_destroy(&c);return rc;
}
