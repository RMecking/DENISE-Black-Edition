#ifndef DENISE_EXACT_SH_DATA_OPS_H
#define DENISE_EXACT_SH_DATA_OPS_H

/* SH-S2: per-shot handling of the data-space operators shared by the exact
 * SH objective and objective/gradient multi-shot wrappers and the driver
 * preflight.  Contract: docs/sh_s2_data_operators_contract.md. */

#include <math.h>
#include <stdlib.h>

/* Accepted SH-S2 controls of the current workflow stage. */
static inline int exact_sh_s2_config_valid(void) {
    extern int TIME_FILT, TIMEWIN, OFFSET_MUTE, ORDER;
    extern float FC_START, FC_END, DT, GAMMA, TWLENGTH_MINUS, TWLENGTH_PLUS;
    extern float OFFSETC;

    if ((TIME_FILT != 0) && (TIME_FILT != 1)) return 0;
    if ((TIMEWIN != 0) && (TIMEWIN != 1)) return 0;
    if ((OFFSET_MUTE < 0) || (OFFSET_MUTE > 2)) return 0;
    if (TIME_FILT == 1) {
        if ((ORDER < 1) || !isfinite(FC_END) || !isfinite(FC_START) ||
                !(FC_END > 0.0f) || !((double)FC_END < 0.5 / (double)DT))
            return 0;
        if ((FC_START > 0.0f) && !(FC_START < FC_END)) return 0;
    }
    if (TIMEWIN != 0) {
        if (!isfinite(GAMMA) || !isfinite(TWLENGTH_MINUS) ||
                !isfinite(TWLENGTH_PLUS) || !(GAMMA >= 0.0f) ||
                !(TWLENGTH_MINUS >= 0.0f) || !(TWLENGTH_PLUS >= 0.0f))
            return 0;
    }
    if ((OFFSET_MUTE != 0) && (!isfinite(OFFSETC) || !(OFFSETC > 0.0f)))
        return 0;
    return 1;
}

/* Data weights are needed for the window, the mask, and with TIME_FILT = 1
 * for the excluded last sample (Owner decision 2026-11-21). */
static inline int exact_sh_s2_weighted(void) {
    extern int TIME_FILT, TIMEWIN, OFFSET_MUTE;
    return (TIME_FILT == 1) || (TIMEWIN != 0) || (OFFSET_MUTE != 0);
}

/* TIME_FILT = 1: filter the local source signals (after the QUELLART = 6
 * spike filter) and the global observed traces of one shot with the legacy
 * apply_tdfilt (Owner decision 2026-11-21: identical to legacy and to the
 * PSV strand, including its unfiltered last sample).  Local status. */
static inline int exact_sh_s2_filter_shot(float **signals, int nsrc_local,
                                          float **observed_global,
                                          int nrec_global, int ns) {
    extern int TIME_FILT, ORDER;
    extern float FC_START, FC_END;

    if (TIME_FILT != 1) return 0;
    if (nsrc_local > 0)
        apply_tdfilt(signals, nsrc_local, ns, ORDER, FC_END, FC_START);
    if (nrec_global > 0)
        apply_tdfilt(observed_global, nrec_global, ns, ORDER, FC_END, FC_START);
    return 0;
}

/* Data weights of one shot (NULL when no data-space operator is on).  The
 * offset uses source column ishot, as legacy calc_res.  Local status. */
static inline int exact_sh_s2_shot_weights(const struct acq *acquisition,
        int ishot, int nrec_local, int nrec_global, int ns, double **weight) {
    extern int TIME_FILT, TIMEWIN, OFFSET_MUTE;
    extern float DH, DT, GAMMA, TWLENGTH_MINUS, TWLENGTH_PLUS, OFFSETC;
    extern char PICKS_FILE[STRING_SIZE];
    float *picks = NULL;
    int status = 0, kept;

    *weight = NULL;
    if (!exact_sh_s2_weighted()) return 0;
    *weight = (double *)malloc((size_t)(nrec_local > 0 ? nrec_local : 1) *
                               (size_t)ns * sizeof(double));
    if (*weight == NULL) return -2;
    if (TIMEWIN == 1) {
        picks = (float *)malloc((size_t)(nrec_global + 1) * sizeof(float));
        if (picks == NULL) return -2;
        status = visco_sh_exact_read_picks(PICKS_FILE, ishot, nrec_global,
                                           picks);
    }
    if (status == 0) {
        kept = visco_sh_exact_data_weights(*weight, nrec_local, ns,
                acquisition->recpos, acquisition->recpos_loc,
                acquisition->srcpos[1][ishot], acquisition->srcpos[2][ishot],
                DH, DT, TIMEWIN, GAMMA, TWLENGTH_MINUS, TWLENGTH_PLUS, picks,
                OFFSET_MUTE, OFFSETC, TIME_FILT == 1);
        status = (kept < 0) ? kept : 0;
    }
    free(picks);
    return status;
}

#endif
