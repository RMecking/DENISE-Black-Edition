/* Inactive exact viscoelastic SH multi-shot objective/raw-gradient wrapper. */

#include "fd.h"

static int exact_multi_preflight(
        const struct visco_sh_exact_multi_shot_request *request) {
    extern int DTINV, LNORM, GRAD_FORM, N_ORDER, TIMEWIN, OFFSET_MUTE;
    extern int TRKILL, NX, NY, NT, L, INVMAT1, SEISMO, EPRECOND;
    extern int SWS_TAPER_CIRCULAR_PER_SHOT, TIME_FILT, INV_STF;
    extern int QUELLTYPB, READREC;
    if ((request == NULL) || (request->wave == NULL) ||
            (request->pml == NULL) || (request->material == NULL) ||
            (request->fwi == NULL) || (request->mpi == NULL) ||
            (request->seismogram == NULL) ||
            (request->legacy_fwi_seismogram == NULL) ||
            (request->acquisition == NULL) || (request->hc == NULL) ||
            (request->dtinv_help == NULL) ||
            (request->grad_primary == NULL) ||
            (request->grad_rho == NULL) || (request->grad_q == NULL) ||
            (request->acquisition->srcpos == NULL) ||
            (request->acquisition->srcpos1 == NULL) ||
            (request->nsrc < 1) || (request->ns != NT) ||
            (request->nrec_local < 0) || (request->nrec_global < 1) ||
            (request->nrec_local > request->nrec_global) ||
            (NX < 1) || (NY < 1) || (L < 1) ||
            ((INVMAT1 != 1) && (INVMAT1 != 3)) ||
            (DTINV != 1) || (LNORM != 2) || (GRAD_FORM != 2) ||
            (N_ORDER != 0) || (TIMEWIN != 0) || (OFFSET_MUTE != 0) ||
            (TRKILL != 0) || (SEISMO != 1) || (EPRECOND != 0) ||
            (SWS_TAPER_CIRCULAR_PER_SHOT != 0) || (TIME_FILT != 0) ||
            (INV_STF != 0) || (QUELLTYPB == 0) ||
            ((READREC != 2) &&
             ((request->acquisition->recpos == NULL) ||
              (request->acquisition->recpos_loc == NULL) ||
              (request->seismogram->sectionvz == NULL) ||
              (request->legacy_fwi_seismogram->sectionread == NULL) ||
              (request->legacy_fwi_seismogram->sectionvzdata == NULL))))
        return -1;
    return 0;
}

static void exact_multi_zero_owned(float **field) {
    extern int NX, NY;
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i) field[j][i] = 0.0f;
}

static void exact_multi_add_owned(float **total, float **shot) {
    extern int NX, NY;
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i) total[j][i] += shot[j][i];
}

static void exact_multi_map_observed(
        struct seisSHfwi *seis, const struct acq *acquisition,
        int nrec_local, int ns) {
    int i, n;
    for (i = 1; i <= nrec_local; ++i)
        for (n = 1; n <= ns; ++n)
            seis->sectionvzdata[i][n] =
                seis->sectionread[acquisition->recpos_loc[3][i]][n];
}

static void exact_multi_release_dynamic_receivers(
        struct visco_sh_exact_multi_shot_request const *request,
        int nrec_local, int nrec_global) {
    struct acq *acquisition = request->acquisition;
    struct seisSH *seismogram = request->seismogram;
    struct seisSHfwi *legacy = request->legacy_fwi_seismogram;
    if (nrec_local > 0) {
        free_matrix(seismogram->sectionvz, 1, nrec_local, 1, request->ns);
        free_matrix(legacy->sectionvzdata, 1, nrec_local, 1, request->ns);
        free_matrix(legacy->sectionvzdiff, 1, nrec_local, 1, request->ns);
        free_matrix(legacy->sectionvzdiffold, 1, nrec_local, 1, request->ns);
        free_imatrix(acquisition->recpos_loc, 1, 3, 1, nrec_local);
    }
    free_matrix(legacy->sectionread, 1, nrec_global, 1, request->ns);
    free_ivector(acquisition->recswitch, 1, nrec_global);
    free_imatrix(acquisition->recpos, 1, 3, 1, nrec_global);
    seismogram->sectionvz = NULL;
    legacy->sectionread = NULL;
    legacy->sectionvzdata = NULL;
    legacy->sectionvzdiff = NULL;
    legacy->sectionvzdiffold = NULL;
    acquisition->recpos = NULL;
    acquisition->recpos_loc = NULL;
    acquisition->recswitch = NULL;
}

int visco_sh_exact_objective_gradient(
        const struct visco_sh_exact_multi_shot_request *request,
        struct visco_sh_exact_multi_shot_result *result) {
    extern int RUN_MULTIPLE_SHOTS, READREC, QUELLTYP, QUELLART;
    extern int ORDER_SPIKE, NX, NY;
    extern float FC_SPIKE_1, FC_SPIKE_2;
    extern FILE *FP;
    struct visco_sh_exact_shot_request shot_request;
    struct visco_sh_exact_shot_result shot_result;
    float **shot_primary = NULL, **shot_rho = NULL, **shot_q = NULL;
    int nshots, ishot, nsrc_local, nrec_local, nrec_global, source_columns;
    int nt, shot_status, status = -1;
    double objective = 0.0;

    if ((result == NULL) || exact_multi_preflight(request) != 0) return -1;
    nshots = RUN_MULTIPLE_SHOTS ? request->nsrc : 1;
    source_columns = RUN_MULTIPLE_SHOTS ? 1 : request->nsrc;
    shot_primary = matrix(1, NY, 1, NX);
    shot_rho = matrix(1, NY, 1, NX);
    shot_q = matrix(1, NY, 1, NX);
    exact_multi_zero_owned(request->grad_primary);
    exact_multi_zero_owned(request->grad_rho);
    exact_multi_zero_owned(request->grad_q);

    for (ishot = 1; ishot <= nshots; ++ishot) {
        nrec_local = request->nrec_local;
        nrec_global = request->nrec_global;
        if (READREC == 2) {
            request->acquisition->recpos = receiver(FP, &nrec_global, ishot);
            request->acquisition->recswitch = ivector(1, nrec_global);
            request->acquisition->recpos_loc = splitrec(
                    request->acquisition->recpos, &nrec_local, nrec_global,
                    request->acquisition->recswitch);
            alloc_seisSH(nrec_local, request->ns, request->seismogram);
            alloc_seisSHfwi(nrec_local, nrec_global, request->ns,
                            request->legacy_fwi_seismogram);
        }

        for (nt = 1; nt <= 8; ++nt)
            request->acquisition->srcpos1[nt][1] =
                request->acquisition->srcpos[nt][ishot];
        QUELLTYP = iround(request->acquisition->srcpos[8][ishot]);
        request->acquisition->srcpos_loc = splitsrc(
                RUN_MULTIPLE_SHOTS ? request->acquisition->srcpos1
                                   : request->acquisition->srcpos,
                &nsrc_local, source_columns);
        MPI_Barrier(MPI_COMM_WORLD);
        request->acquisition->signals = wavelet(
                request->acquisition->srcpos_loc, nsrc_local, ishot);
        if ((QUELLART == 6) && (nsrc_local > 0))
            apply_tdfilt(request->acquisition->signals, nsrc_local,
                         request->ns, ORDER_SPIKE, FC_SPIKE_2, FC_SPIKE_1);

        inseis(ishot, request->legacy_fwi_seismogram->sectionread,
               nrec_global, request->ns, 2, request->iter);
        exact_multi_map_observed(request->legacy_fwi_seismogram,
                                 request->acquisition, nrec_local, request->ns);
        exact_multi_zero_owned(shot_primary);
        exact_multi_zero_owned(shot_rho);
        exact_multi_zero_owned(shot_q);
        memset(&shot_request, 0, sizeof(shot_request));
        memset(&shot_result, 0, sizeof(shot_result));
        shot_request.wave = request->wave;
        shot_request.pml = request->pml;
        shot_request.material = request->material;
        shot_request.fwi = request->fwi;
        shot_request.mpi = request->mpi;
        shot_request.seismogram = request->seismogram;
        shot_request.legacy_fwi_seismogram = request->legacy_fwi_seismogram;
        shot_request.acquisition = request->acquisition;
        shot_request.hc = request->hc;
        shot_request.ishot = ishot;
        shot_request.nshots = nshots;
        shot_request.nsrc_local = nsrc_local;
        shot_request.ns = request->ns;
        shot_request.nrec_local = nrec_local;
        shot_request.hin = request->hin;
        shot_request.dtinv_help = request->dtinv_help;
        shot_request.source_energy = request->source_energy;
        shot_request.receiver_energy = request->receiver_energy;
        shot_request.request_send = request->request_send;
        shot_request.request_receive = request->request_receive;
        shot_request.observed_vz =
            request->legacy_fwi_seismogram->sectionvzdata;
        shot_request.grad_primary = shot_primary;
        shot_request.grad_rho = shot_rho;
        shot_request.grad_q = shot_q;
        shot_status = visco_sh_exact_objective_gradient_shot(
                &shot_request, &shot_result);
        if (shot_status == 0) {
            objective += shot_result.objective;
            exact_multi_add_owned(request->grad_primary, shot_primary);
            exact_multi_add_owned(request->grad_rho, shot_rho);
            exact_multi_add_owned(request->grad_q, shot_q);
        }

        if (request->acquisition->signals != NULL) {
            free_matrix(request->acquisition->signals, 1, nsrc_local,
                        1, request->ns);
            request->acquisition->signals = NULL;
        }
        if (request->acquisition->srcpos_loc != NULL) {
            free_matrix(request->acquisition->srcpos_loc, 1, 8,
                        1, nsrc_local);
            request->acquisition->srcpos_loc = NULL;
        }
        if (READREC == 2)
            exact_multi_release_dynamic_receivers(
                    request, nrec_local, nrec_global);
        if (shot_status != 0) goto cleanup;
    }
    result->objective = objective;
    result->shot_count = nshots;
    status = 0;

cleanup:
    if (shot_q != NULL) free_matrix(shot_q, 1, NY, 1, NX);
    if (shot_rho != NULL) free_matrix(shot_rho, 1, NY, 1, NX);
    if (shot_primary != NULL) free_matrix(shot_primary, 1, NY, 1, NX);
    return status;
}
