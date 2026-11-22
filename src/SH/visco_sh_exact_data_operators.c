/* SH-S2: data-space operators of the exact viscoelastic SH objective.
 *
 *   Phi(m) = 1/2 sum_s || M_s T_s ( F_s(m; B w_s) - B d_s ) ||^2
 *
 * B (stage filter, Koehn semantics, legacy apply_tdfilt) acts once on the
 * source signal and once on the observed traces; synthetics and residual are
 * not filtered, so no transpose of B is needed.  This unit provides T (time
 * window around first-arrival picks) and M (offset mask), diagonal per trace
 * and sample; the objective uses (w r)^2 and the adjoint source w^2 r.
 * Contract: docs/sh_s2_data_operators_contract.md.
 *
 * The functions are local (no MPI) and do not call err(); callers make the
 * collective decision. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "fd.h"

/* Reads ntr_glob first-arrival picks (seconds) for one shot from
 * "<prefix><ishot>.dat", the legacy TIMEWIN = 1 file convention. */
int visco_sh_exact_read_picks(const char *prefix, int ishot, int ntr_glob,
                              float *picks) {
    char name[STRING_SIZE + 32];
    FILE *fp;
    int i, status = 0;

    if ((prefix == NULL) || (picks == NULL) || (ntr_glob < 1)) return -1;
    if (snprintf(name, sizeof(name), "%s%i.dat", prefix, ishot) >=
            (int)sizeof(name)) return -1;
    fp = fopen(name, "r");
    if (fp == NULL) return -3;
    for (i = 1; i <= ntr_glob; ++i) {
        float value;
        if ((fscanf(fp, "%f", &value) != 1) || !isfinite(value)) {
            status = -4;
            break;
        }
        picks[i] = value;
    }
    fclose(fp);
    return status;
}

/* Diagonal weights w(i, k) of T and M for the local receivers of one shot,
 * stored as weight[(i - 1) * ns + (k - 1)] for i = 1..nrec_local and sample
 * k = 1..ns at time t_k = k * dt (legacy time_window convention).
 *
 * TIMEWIN 1: pick p from picks[global receiver] (a constant window is a pick
 *   file with equal times; TIMEWIN 2 is not supported);
 *   t_lo = p - twin_minus, t_hi = p + twin_plus;
 *   w = 1 on [t_lo, t_hi], exp(-gamma (t - t_hi)^2) after,
 *   exp(-gamma (t - t_lo)^2) before.
 * OFFSET_MUTE 1 (2): w = 0 for offset >= (<=) offsetc, offset between
 *   (recpos[1]*dh, recpos[2]*dh) and (source_x, source_y) as legacy calc_res.
 * mask_last_sample (TIME_FILT = 1): w = 0 at k = ns, the sample the legacy
 *   apply_tdfilt leaves unfiltered (Owner decision 2026-11-21), so that the
 *   objective and the adjoint source exclude it.
 * Returns the number of kept traces (>= 0) or a negative error code. */
int visco_sh_exact_data_weights(double *weight, int nrec_local, int ns,
        int **recpos, int **recpos_loc, float source_x, float source_y,
        float dh, float dt, int timewin, float gamma, float twin_minus,
        float twin_plus, const float *picks, int offset_mute, float offsetc,
        int mask_last_sample) {
    int i, k, kept = 0;

    if ((nrec_local < 0) || (ns < 1) || !(dt > 0.0f) || !(dh > 0.0f) ||
            (timewin < 0) || (timewin > 1) || (offset_mute < 0) ||
            (offset_mute > 2) || (mask_last_sample < 0) ||
            (mask_last_sample > 1)) return -1;
    if ((timewin != 0) && (!(gamma >= 0.0f) || !(twin_minus >= 0.0f) ||
            !(twin_plus >= 0.0f) || !isfinite(gamma) ||
            !isfinite(twin_minus) || !isfinite(twin_plus))) return -1;
    if ((timewin == 1) && (picks == NULL)) return -1;
    if ((offset_mute != 0) && (!(offsetc > 0.0f) || !isfinite(offsetc)))
        return -1;
    if (nrec_local == 0) return 0;
    if ((weight == NULL) || (recpos == NULL) || (recpos_loc == NULL))
        return -1;

    for (i = 1; i <= nrec_local; ++i) {
        int global = recpos_loc[3][i];
        double *w = weight + (size_t)(i - 1) * (size_t)ns;
        int muted = 0;

        if (offset_mute != 0) {
            /* binary32 as legacy calc_res, so that ties at OFFSETC agree */
            float xr = (float)recpos[1][global] * dh;
            float yr = (float)recpos[2][global] * dh;
            float dx = source_x - xr, dy = source_y - yr;
            float offset = (float)sqrt((double)(dx * dx + dy * dy));
            if ((offset_mute == 1) && (offset >= offsetc)) muted = 1;
            if ((offset_mute == 2) && (offset <= offsetc)) muted = 1;
        }
        if (muted) {
            for (k = 0; k < ns; ++k) w[k] = 0.0;
            continue;
        }
        ++kept;
        if (timewin == 0) {
            for (k = 0; k < ns; ++k) w[k] = 1.0;
        } else {
            double p = (double)picks[global];
            double t_lo = p - (double)twin_minus;
            double t_hi = p + (double)twin_plus;
            for (k = 1; k <= ns; ++k) {
                double t = (double)k * (double)dt;
                double d = 0.0;
                if (t > t_hi) d = t - t_hi;
                else if (t < t_lo) d = t - t_lo;
                w[k - 1] = (d == 0.0) ? 1.0 : exp(-(double)gamma * d * d);
            }
        }
        if (mask_last_sample) w[ns - 1] = 0.0;
    }
    return kept;
}
