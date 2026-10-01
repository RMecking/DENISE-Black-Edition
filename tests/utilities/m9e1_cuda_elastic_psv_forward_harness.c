/* Canonical CPU production view, compiled without CUDA. No new numerics. */
#include "../../src/PSV/elastic_psv_born.c"

int m9e1_cpu_view(const struct denise_elastic_psv_born_config *cfg,
                 const float *initial, const float *profiles,
                 float *data, float *state, float *trajectory) {
    struct denise_elastic_psv_born *c=NULL;
    float *f[5],*psi[8],*q[4];
    int k,j,t; size_t offset=0;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    if(allocate_forward(f,psi,q,c->cells)) {
        free_forward(f,psi,q);denise_elastic_psv_born_destroy(&c);return -1;
    }
    if(initial) {
        for(k=0;k<5;k++)memcpy(f[k],initial+(size_t)k*c->cells,c->cells*4);
        for(k=0;k<8;k++)memcpy(psi[k],initial+(size_t)(k+5)*c->cells,c->cells*4);
    }
    if(profiles)for(k=0;k<4;k++) {
        struct pml_profile *p=&c->profile[k];
        memcpy(p->kappa,profiles+offset,(size_t)p->length*4);offset+=p->length;
        memcpy(p->a,profiles+offset,(size_t)p->length*4);offset+=p->length;
        memcpy(p->b,profiles+offset,(size_t)p->length*4);offset+=p->length;
    }
    memset(data,0,c->data_count*4);
    for(t=0;t<c->nt;t++)forward_timestep(c,f,psi,q,c->lambda,c->mu,c->mu_corner,t,data,
                                        trajectory+(size_t)t*4*c->cells,0);
    for(j=0;j<5;j++)memcpy(state+(size_t)j*c->cells,f[j],c->cells*4);
    for(j=0;j<8;j++)memcpy(state+(size_t)(j+5)*c->cells,psi[j],c->cells*4);
    free_forward(f,psi,q);denise_elastic_psv_born_destroy(&c);return 0;
}

int m9e1_cpu_static(const struct denise_elastic_psv_born_config *cfg,float *maps,float *profiles) {
    struct denise_elastic_psv_born *c=NULL;int k,j,i;size_t offset=0;
    const float *m[5];size_t padded=(size_t)(cfg->nx+4)*(cfg->ny+4);
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    m[0]=c->lambda;m[1]=c->mu;m[2]=c->invrho_x;m[3]=c->invrho_y;m[4]=c->mu_corner;
    for(k=0;k<5;k++)for(j=-2;j<c->ny+2;j++)for(i=-2;i<c->nx+2;i++)
        maps[(size_t)k*padded+(size_t)(j+2)*(c->nx+4)+i+2]=m[k][cell(c,wrap(j,c->ny),wrap(i,c->nx))];
    for(k=0;k<4;k++) {
        const struct pml_profile *p=&c->profile[k];
        memcpy(profiles+offset,p->kappa,(size_t)p->length*4);offset+=p->length;
        memcpy(profiles+offset,p->a,(size_t)p->length*4);offset+=p->length;
        memcpy(profiles+offset,p->b,(size_t)p->length*4);offset+=p->length;
    }
    denise_elastic_psv_born_destroy(&c);return 0;
}
