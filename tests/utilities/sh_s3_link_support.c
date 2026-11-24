/* SH-S3 link support for stub harnesses that compile the exact SH
 * multi-shot wrappers (src/SH/obj_sh_visc_exact.c, grad_obj_sh_visc_exact.c)
 * without the source time function units.  The harnesses keep INV_STF at 0,
 * so the estimation is inactive and these stubs are never asked to estimate
 * (prepare_shot fails closed if it were).  The real units
 * (src/SH/visco_sh_exact_stf.c, visco_sh_exact_stf_shot.c) are covered by
 * tests/test_sh_s3_stf.py with the DENISE executable. */
#include "fd.h"

int visco_sh_exact_stf_active(void) {
    return 0;
}

double visco_sh_exact_stf_shot_data_term(void) {
    return 0.0;
}

int visco_sh_exact_stf_prepare_shot(
        const struct visco_sh_exact_multi_shot_request *request, int ishot,
        int nshots, int nsrc_local, int nrec_local, int nrec_global,
        double **data_weight, double *phi_shot) {
    (void)request; (void)ishot; (void)nshots; (void)nsrc_local;
    (void)nrec_local; (void)nrec_global; (void)data_weight; (void)phi_shot;
    return -1;
}
