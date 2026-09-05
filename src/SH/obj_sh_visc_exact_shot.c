/* Inactive exact viscoelastic SH single-shot objective evaluator. */

#include "fd.h"

static int exact_objective_shot_preflight(
        const struct visco_sh_exact_objective_shot_request *request,
        const struct visco_sh_exact_objective_shot_result *result) {
    extern int DTINV, LNORM, GRAD_FORM, N_ORDER, TIMEWIN, OFFSET_MUTE;
    extern int TRKILL, NX, NY, NT, L, INVMAT1, SEISMO;

    if ((request == NULL) || (result == NULL)) return -1;
    if ((request->wave == NULL) || (request->pml == NULL) ||
            (request->material == NULL) || (request->fwi == NULL) ||
            (request->mpi == NULL) || (request->seismogram == NULL) ||
            (request->legacy_fwi_seismogram == NULL) ||
            (request->acquisition == NULL) || (request->hc == NULL) ||
            (request->dtinv_help == NULL) || (request->ns != NT) ||
            (request->nrec_local < 0) || (request->nsrc_local < 0) ||
            (NX < 1) || (NY < 1) || (L < 1) ||
            ((INVMAT1 != 1) && (INVMAT1 != 3)) || (DTINV != 1) ||
            (LNORM != 2) || (GRAD_FORM != 2) || (N_ORDER != 0) ||
            (TIMEWIN != 0) || (OFFSET_MUTE != 0) || (TRKILL != 0) ||
            (SEISMO != 1)) return -1;
    if ((request->nrec_local > 0) &&
            ((request->observed_vz == NULL) ||
             (request->seismogram->sectionvz == NULL) ||
             (request->acquisition->recpos_loc == NULL))) return -1;
    if ((request->nsrc_local > 0) &&
            ((request->acquisition->srcpos_loc == NULL) ||
             (request->acquisition->signals == NULL))) return -1;
    return 0;
}

int visco_sh_exact_objective_shot(
        const struct visco_sh_exact_objective_shot_request *request,
        struct visco_sh_exact_objective_shot_result *result) {
    extern int NT;
    double local_objective = 0.0, global_objective = 0.0;
    int i, n;

    if (exact_objective_shot_preflight(request, result) != 0) return -1;

    /* Share the exact-gradient bridge's physical forward stepping.  Mode 2
     * samples receivers without trajectory capture or gradient storage. */
    sh_visc(request->wave, request->pml, request->material, request->fwi,
            request->mpi, request->seismogram,
            request->legacy_fwi_seismogram, request->acquisition, request->hc,
            request->ishot, request->nshots, request->nsrc_local, request->ns,
            request->nrec_local, request->source_energy,
            request->receiver_energy, request->hin, request->dtinv_help, 2,
            request->request_send, request->request_receive);

    /* Intentionally identical to the exact-gradient bridge: modeled minus
     * observed, with the same chronological sample indexing and 0.5*r^2. */
    for (n = 1; n < NT; ++n) {
        for (i = 0; i < request->nrec_local; ++i) {
            double residual =
                    request->seismogram->sectionvz[i + 1][n + 1] -
                    request->observed_vz[i + 1][n + 1];
            local_objective += 0.5 * residual * residual;
        }
    }
    if (MPI_Allreduce(&local_objective, &global_objective, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) return -2;

    result->objective = global_objective;
    return 0;
}
