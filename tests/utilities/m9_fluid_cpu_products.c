/* Test-only inspection of actual canonical CPU material/state products. */
#ifndef DENISE_FLUID_SERIAL_SOURCE
#define DENISE_FLUID_SERIAL_SOURCE "../../src/PSV/elastic_psv_born.c"
#endif
#include DENISE_FLUID_SERIAL_SOURCE
#include <fenv.h>

void fluid_clear_arithmetic(void) { feclearexcept(FE_ALL_EXCEPT); }
int fluid_singular_arithmetic(void) { return fetestexcept(FE_DIVBYZERO|FE_INVALID); }
int fluid_copy_maps(const struct denise_elastic_psv_born *c,float *out) {
    const float *map[3]={c->invrho_x,c->invrho_y,c->mu_corner};int k;
    if(!c||!out)return -1;
    for(k=0;k<3;k++)memcpy(out+(size_t)k*c->cells,map[k],4*c->cells);
    return 0;
}
/* Same production timestep called by prepare(), no alternative state graph. */
int fluid_products(struct denise_elastic_psv_born *c,float *data,float *state,
                   float *strain,double *late_ratio) {
    float *f[5],*psi[8],*q[4];int k,t;size_t p;
    double peak=0,late=0;
    if(!c||!data||!state||!strain||!late_ratio)return -1;
    if(allocate_forward(f,psi,q,c->cells)){free_forward(f,psi,q);return -1;}
    for(t=0;t<c->nt;t++) {
        double norm=0;
        forward_timestep(c,f,psi,q,c->lambda,c->mu,c->mu_corner,t,data,strain,0);
        for(p=0;p<c->cells;p++)
            norm+=(double)c->rho[p]*((double)f[0][p]*f[0][p]+(double)f[1][p]*f[1][p])
                +((double)f[2][p]*f[2][p]+(double)f[3][p]*f[3][p]+2.0*f[4][p]*f[4][p])
                /((double)c->lambda[p]+2.0*c->mu[p]);
        if(norm>peak)peak=norm;
        if(t>=3*c->nt/4&&norm>late)late=norm;
    }
    for(k=0;k<5;k++)memcpy(state+(size_t)k*c->cells,f[k],4*c->cells);
    for(k=0;k<8;k++)memcpy(state+(size_t)(k+5)*c->cells,psi[k],4*c->cells);
    *late_ratio=peak?late/peak:0;
    free_forward(f,psi,q);return 0;
}
