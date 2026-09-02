#include "fd.h"

#include <stdlib.h>
#include <string.h>

int NX = 4;
int NY = 3;
int L = 1;
int FW = 0;
int FDORDER = 4;

float **matrix(int nrl, int nrh, int ncl, int nch) {
    (void)nrl; (void)nrh; (void)ncl; (void)nch;
    return (float **)calloc(1, sizeof(float *));
}

float *vector(int nl, int nh) {
    (void)nl; (void)nh;
    return (float *)calloc(1, sizeof(float));
}

float ***f3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh) {
    (void)nrl; (void)nrh; (void)ncl; (void)nch; (void)ndl; (void)ndh;
    return (float ***)calloc(1, sizeof(float **));
}

static void release_material(struct matSH *material) {
    free(material->prho);
    free(material->prhoi);
    free(material->puip);
    free(material->pujp);
    free(material->pu);
    free(material->puipjp);
    if (L > 0) {
        free(material->dip);
        free(material->d);
        free(material->e);
        free(material->pqs);
        free(material->ptaus);
        free(material->ptausipjp);
        free(material->fipjp);
        free(material->f);
        free(material->g);
        free(material->peta);
        free(material->etaip);
        free(material->etajm);
        free(material->bip);
        free(material->bjm);
        free(material->cip);
        free(material->cjm);
    }
}

int m63c8b1_owned_q_allocation_contract(int mechanisms) {
    struct matSH material;
    int result;

    memset(&material, 0, sizeof(material));
    L = mechanisms;
    alloc_matSH(&material);
    if (L > 0)
        result = material.pqs != NULL && material.ptaus != NULL &&
                 material.pqs != material.ptaus;
    else
        result = material.pqs == NULL && material.ptaus == NULL;
    release_material(&material);
    return result;
}
