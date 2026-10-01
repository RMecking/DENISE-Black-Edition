/* Test-only views of production maps. No alternate production ABI/state. */
#include "../../src/PSV/elastic_psv_born.c"

int m9d3_step(const struct denise_elastic_psv_born_config *cfg,float *state,
              int timestep,float *strain,float *samples,float *ghosts) {
    struct denise_elastic_psv_born *c=NULL;
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4];
    size_t n;
    int k,m,i,status=-1;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    if(allocate_forward(f,psi,q,c->cells)) {free_forward(f,psi,q);goto done;}
    for(k=0;k<FIELD_COUNT;k++)copy_floats(f[k],state+(size_t)k*c->cells,c->cells);
    for(k=0;k<PSI_COUNT;k++)copy_floats(psi[k],state+(size_t)(k+5)*c->cells,c->cells);
    forward_timestep(c,f,psi,q,c->lambda,c->mu,c->mu_corner,timestep,samples,strain,0);
    for(k=0;k<FIELD_COUNT;k++)copy_floats(state+(size_t)k*c->cells,f[k],c->cells);
    for(k=0;k<PSI_COUNT;k++)copy_floats(state+(size_t)(k+5)*c->cells,psi[k],c->cells);
    n=0;
    for(k=0;k<FIELD_COUNT;k++)if(k!=SXX)for(m=1;m<=2;m++)for(i=0;i<c->nx;i++)
        ghosts[n++]=surface_value(c,f[k],-m,i,k,q,c->lambda,c->mu,NULL,NULL,NULL);
    status=0;free_forward(f,psi,q);
done:
    denise_elastic_psv_born_destroy(&c);return status;
}

int m9d3_reverse(const struct denise_elastic_psv_born_config *cfg,double *statebar,
                 const float *strain,const float *samples,double *gl,double *gm) {
    struct denise_elastic_psv_born *c=NULL;
    double *bar[FIELD_COUNT],*psi[PSI_COUNT],*q=NULL,*gc=NULL,*xy=NULL;
    int k,status=-1;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    if(allocate_adjoint(bar,psi,&q,c->cells)) {free_adjoint(bar,psi,q);goto done;}
    gc=checked_calloc(c->cells,sizeof(double));xy=checked_calloc(2*c->cells,sizeof(double));
    if(!gc||!xy)goto cleanup;
    memset(gl,0,c->cells*sizeof(double));memset(gm,0,c->cells*sizeof(double));
    for(k=0;k<FIELD_COUNT;k++)memcpy(bar[k],statebar+(size_t)k*c->cells,c->cells*sizeof(double));
    for(k=0;k<PSI_COUNT;k++)memcpy(psi[k],statebar+(size_t)(k+5)*c->cells,c->cells*sizeof(double));
    surface_reverse_step(c,bar,psi,q,gc,xy,strain,gl,gm,samples,0);
    for(k=0;k<FIELD_COUNT;k++)memcpy(statebar+(size_t)k*c->cells,bar[k],c->cells*sizeof(double));
    for(k=0;k<PSI_COUNT;k++)memcpy(statebar+(size_t)(k+5)*c->cells,psi[k],c->cells*sizeof(double));
    status=0;
cleanup:
    free(gc);free(xy);free_adjoint(bar,psi,q);
done:
    denise_elastic_psv_born_destroy(&c);return status;
}

int m9d3_extend(const struct denise_elastic_psv_born_config *cfg,int kind,
                const float *field,const float *xx,const float *yx,float *out) {
    struct denise_elastic_psv_born *c=NULL;
    float *q[4];int j,i;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    q[0]=(float *)xx;q[1]=(float *)yx;q[2]=q[3]=NULL;
    for(j=-2;j<c->ny+2;j++)for(i=0;i<c->nx;i++)
        out[(size_t)(j+2)*c->nx+i]=(kind==SYY && j==0)?0.0f:
            surface_value(c,field,j,i,kind,q,c->lambda,c->mu,NULL,NULL,NULL);
    denise_elastic_psv_born_destroy(&c);return 0;
}

int m9d3_extend_reverse(const struct denise_elastic_psv_born_config *cfg,int kind,
                        const double *bar,const float *bg,double *physical,
                        double *xx,double *yx,double *gl,double *gm) {
    struct denise_elastic_psv_born *c=NULL;
    int j,i;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    memset(physical,0,c->cells*sizeof(double));memset(xx,0,c->cells*sizeof(double));
    memset(yx,0,c->cells*sizeof(double));memset(gl,0,c->cells*sizeof(double));
    memset(gm,0,c->cells*sizeof(double));
    for(j=-2;j<c->ny+2;j++)for(i=0;i<c->nx;i++)if(kind!=SYY||j!=0)
        surface_value_transpose(c,physical,j,i,kind,bar[(size_t)(j+2)*c->nx+i],xx,yx,bg,gl,gm);
    denise_elastic_psv_born_destroy(&c);return 0;
}

int m9d3_surface_material(const struct denise_elastic_psv_born_config *cfg,
                          const float *qxx,const double *sxxbar,
                          const double *ghostbar,double *al,double *am,
                          double *bl,double *bm) {
    struct denise_elastic_psv_born *c=NULL;
    int i,m;
    double h;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    h=(float)(1.0f/c->coefficient);
    for(i=0;i<c->nx;i++) {
        size_t p=cell(c,0,i);
        struct surface_material s=surface_material(c->lambda[p],c->mu[p]);
        double abar=0.0;
        for(m=1;m<=2;m++)abar+=(2.0*m-1.0)*h*qxx[p]*ghostbar[(size_t)(m-1)*c->nx+i];
        al[i]=s.al*abar;am[i]=s.am*abar;
        bl[i]=s.bl*qxx[p]*sxxbar[i];bm[i]=s.bm*qxx[p]*sxxbar[i];
    }
    denise_elastic_psv_born_destroy(&c);return 0;
}

/* Same production forward_timestep, with no full retained trajectory. */
int m9d3_run(const struct denise_elastic_psv_born_config *cfg,float *samples) {
    struct denise_elastic_psv_born *c=NULL;
    float *f[FIELD_COUNT],*psi[PSI_COUNT],*q[4];
    int timestep,status=-1;
    if(denise_elastic_psv_born_create(cfg,&c))return -1;
    if(allocate_forward(f,psi,q,c->cells)){free_forward(f,psi,q);goto done;}
    for(timestep=0;timestep<c->nt;timestep++)
        forward_timestep(c,f,psi,q,c->lambda,c->mu,c->mu_corner,timestep,samples,NULL,0);
    status=0;free_forward(f,psi,q);
done:
    denise_elastic_psv_born_destroy(&c);return status;
}
