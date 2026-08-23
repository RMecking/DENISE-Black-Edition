/*------------------------------------------------------------------------
 * Complete the elastic SH fields at a flat traction-free top boundary.
 * The caller owns the physical-boundary and MPI-rank gating.
 *----------------------------------------------------------------------*/

#include "fd.h"

void surface_elastic_SH_velocity(float **vz, int nx, int half_order){

    int i, k;

    for (i=1; i<=nx; i++){
        for (k=1; k<=half_order; k++){
            vz[1-k][i]=vz[k][i];
        }
    }
}

void surface_elastic_SH_stress(float **syz, int nx, int half_order){

    int i, k;

    for (i=1; i<=nx; i++){
        syz[0][i]=0.0;
        for (k=1; k<half_order; k++){
            syz[-k][i]=-syz[k][i];
        }
    }
}
