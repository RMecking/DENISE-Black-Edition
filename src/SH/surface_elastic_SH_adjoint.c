/*------------------------------------------------------------------------
 * Exact Euclidean transposes of the flat elastic SH free-surface maps.
 * The caller owns the physical-boundary and MPI-rank gating.
 *----------------------------------------------------------------------*/

#include "fd.h"

void surface_elastic_SH_velocity_adjoint(
        float **bar_vz, int nx, int half_order){

    int i, k;

    for (i=1; i<=nx; ++i){
        for (k=1; k<=half_order; ++k){
            bar_vz[k][i] += bar_vz[1-k][i];
            bar_vz[1-k][i] = 0.0f;
        }
    }
}

void surface_elastic_SH_stress_adjoint(
        float **bar_syz, int nx, int half_order){

    int i, k;

    for (i=1; i<=nx; ++i){
        bar_syz[0][i] = 0.0f;
        for (k=1; k<half_order; ++k){
            bar_syz[k][i] -= bar_syz[-k][i];
            bar_syz[-k][i] = 0.0f;
        }
    }
}
