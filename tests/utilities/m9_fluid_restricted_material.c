/* Only exposes actual private production material products to focused tests. */
#include "m9_fluid_cpu_products.c"
int fluid2_material_products(const struct denise_elastic_psv_born *c,
                             const float *dm,const double *bar,float *dh,double *gm) {
    if(!c||!dm||!bar||!dh||!gm)return -1;
    memset(gm,0,c->cells*sizeof(double));
    harmonic_tangent(c,dm,dh);
    harmonic_transpose_add(c,bar,gm);
    return 0;
}
