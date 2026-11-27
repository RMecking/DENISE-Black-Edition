/*
 * SH-MASK: update mask from file and source taper at the exact viscoelastic
 * SH optimizer boundary (task packet SH-MASKE, 2026-11-26).
 *
 * The mask selects the free parameters (cells with mask 1); cells with mask 0
 * keep their start value in every iteration and stage.  The source taper T is
 * a diagonal factor of the search-direction metric (D = w * T * P); the raw
 * gradient and the objective are not changed.  Cells with T = 0 are fixed,
 * because the metric must be positive on free cells.
 *
 * The unit does not call err() and links stand-alone with a harness that
 * provides err().  Only visco_sh_exact_mask_change_report is collective.
 */

#include "fd.h"

static int mask_rows_valid(float **field, int ny) {
    int j;
    if (field == NULL) return 0;
    for (j = 1; j <= ny; ++j)
        if (field[j] == NULL) return 0;
    return 1;
}

/* Reads a global float32 field (x outer, y inner) of exactly nxg*nyg values. */
static int read_global_field(const char *path, int nxg, int nyg, float *values) {
    FILE *fp;
    long bytes;
    size_t count = (size_t)nxg * (size_t)nyg, got;

    fp = fopen(path, "rb");
    if (fp == NULL) return -2;
    if (fseek(fp, 0L, SEEK_END) != 0) { fclose(fp); return -2; }
    bytes = ftell(fp);
    if ((bytes < 0) || ((size_t)bytes != count * sizeof(float))) {
        fclose(fp);
        return -3;
    }
    rewind(fp);
    got = fread(values, sizeof(float), count, fp);
    fclose(fp);
    return (got == count) ? 0 : -3;
}

/* Mask file of the exact SH path: <tfile>_u.bin and <tfile>_rho.bin, values
 * exactly 0 or 1, both files equal.  Cells with file value 0 are set to 0.0f
 * in mask (logical AND with the CPML mask); the mask is untouched on error.
 * Returns 0, -1 arguments, -2 a file cannot be opened, -3 wrong size,
 * -4 a value other than 0 or 1, -5 the two files differ, -6 allocation. */
int visco_sh_exact_apply_mask_file(float **mask, int nx, int ny, int pos_x,
        int pos_y, int nxg, int nyg, const char *tfile) {
    char path[STRING_SIZE2];
    float *u = NULL, *rho = NULL, v;
    size_t count, k;
    int status = 0, i, j, ig, jg;

    if ((mask == NULL) || (tfile == NULL) || (nx < 1) || (ny < 1) ||
            (nxg < nx) || (nyg < ny) || (pos_x < 0) || (pos_y < 0) ||
            ((pos_x + 1) * nx > nxg) || ((pos_y + 1) * ny > nyg) ||
            !mask_rows_valid(mask, ny)) return -1;
    count = (size_t)nxg * (size_t)nyg;
    u = (float *)malloc(count * sizeof(float));
    rho = (float *)malloc(count * sizeof(float));
    if ((u == NULL) || (rho == NULL)) { status = -6; goto done; }

    if (snprintf(path, sizeof(path), "%s_u.bin", tfile) >= (int)sizeof(path)) {
        status = -2; goto done;
    }
    status = read_global_field(path, nxg, nyg, u);
    if (status != 0) goto done;
    if (snprintf(path, sizeof(path), "%s_rho.bin", tfile) >= (int)sizeof(path)) {
        status = -2; goto done;
    }
    status = read_global_field(path, nxg, nyg, rho);
    if (status != 0) goto done;

    for (k = 0; k < count; ++k) {
        if (!((u[k] == 0.0f) || (u[k] == 1.0f)) ||
                !((rho[k] == 0.0f) || (rho[k] == 1.0f))) { status = -4; goto done; }
    }
    for (k = 0; k < count; ++k) {
        if (u[k] != rho[k]) { status = -5; goto done; }
    }

    for (j = 1; j <= ny; ++j) {
        for (i = 1; i <= nx; ++i) {
            ig = pos_x * nx + i;
            jg = pos_y * ny + j;
            v = u[(size_t)(ig - 1) * (size_t)nyg + (size_t)(jg - 1)];
            if (v == 0.0f) mask[j][i] = 0.0f;
        }
    }
done:
    free(u);
    free(rho);
    return status;
}

/* Source taper T = min over sources of T_s with
 *   T_s = 0                      |I-I_s| <= filtsize and |J-J_s| <= filtsize
 *   T_s = sin^2(pi r / (2 R))    0 < r < R,  r = dh * sqrt(dI^2 + dJ^2)
 *   T_s = 1                      r >= R
 * evaluated in double, stored as float.  Source grid indices as in the
 * forward injection (splitsrc): iround(srcpos/dh).  Cells with T = 0 are set
 * to 0.0f in mask.  Returns 0 or -1 for invalid arguments (nothing written). */
int visco_sh_exact_build_source_taper(float **taper, float **mask, int nx,
        int ny, int pos_x, int pos_y, float dh, float **srcpos, int nsrc,
        float radius, int filtsize) {
    const double pi = 3.14159265358979323846;
    int i, j, s, ig, jg, is, js, di, dj;
    double t, ts, r;

    if ((taper == NULL) || (mask == NULL) || (srcpos == NULL) ||
            (srcpos[1] == NULL) || (srcpos[2] == NULL) || (nx < 1) ||
            (ny < 1) || (pos_x < 0) || (pos_y < 0) || (nsrc < 1) ||
            !isfinite(dh) || (dh <= 0.0f) || !isfinite(radius) ||
            (radius < dh) || (filtsize < 0) ||
            !mask_rows_valid(taper, ny) || !mask_rows_valid(mask, ny)) return -1;
    for (s = 1; s <= nsrc; ++s)
        if (!isfinite(srcpos[1][s]) || !isfinite(srcpos[2][s])) return -1;

    for (j = 1; j <= ny; ++j) {
        for (i = 1; i <= nx; ++i) {
            ig = pos_x * nx + i;
            jg = pos_y * ny + j;
            t = 1.0;
            for (s = 1; s <= nsrc; ++s) {
                is = iround(srcpos[1][s] / dh);
                js = iround(srcpos[2][s] / dh);
                di = ig - is;
                dj = jg - js;
                if ((abs(di) <= filtsize) && (abs(dj) <= filtsize)) {
                    ts = 0.0;
                } else {
                    r = (double)dh * sqrt((double)di * (double)di +
                                          (double)dj * (double)dj);
                    if (r >= (double)radius) {
                        ts = 1.0;
                    } else {
                        ts = sin(pi * r / (2.0 * (double)radius));
                        ts = ts * ts;
                    }
                }
                if (ts < t) t = ts;
            }
            taper[j][i] = (float)t;
            if (taper[j][i] == 0.0f) mask[j][i] = 0.0f;
        }
    }
    return 0;
}

/* metric <- (float)((double)taper * (double)metric) on all own cells. */
int visco_sh_exact_apply_taper_metric(float **metric, float **taper, int nx,
        int ny) {
    int i, j;

    if ((nx < 1) || (ny < 1) || !mask_rows_valid(metric, ny) ||
            !mask_rows_valid(taper, ny)) return -1;
    for (j = 1; j <= ny; ++j)
        for (i = 1; i <= nx; ++i)
            metric[j][i] = (float)((double)taper[j][i] * (double)metric[j][i]);
    return 0;
}

/* Collective diagnostics of one accepted update (MPI_COMM_WORLD).
 * fixed_change_max: max |trial - base| over cells with mask != 1 (all classes);
 * taper_share: sum over cells with taper < 1 of d^2 / sum of d^2 for the
 *   primary class (d = trial - base), -1 without taper, 0 for a zero update;
 * max_change: max |d| of the primary class, at global cell (max_i, max_j),
 *   smallest x-outer linear index on ties.  Returns 0 or -1 (arguments,
 *   collective) or -2 (MPI). */
int visco_sh_exact_mask_change_report(float **base[3], float **trial[3],
        float **mask, float **taper, int nx, int ny, int pos_x, int pos_y,
        int nyg, struct visco_sh_exact_mask_report *report) {
    int i, j, c, k, local_bad = 0, any_bad = 0;
    double d, fixed = 0.0, sums[2] = {0.0, 0.0}, global_sums[2];
    struct { double value; int index; } local_max = {-1.0, 0}, global_max;

    if ((report == NULL) || (nx < 1) || (ny < 1) || (nyg < ny) ||
            !mask_rows_valid(mask, ny) ||
            ((taper != NULL) && !mask_rows_valid(taper, ny))) local_bad = 1;
    for (c = 0; (c < 3) && !local_bad; ++c)
        if ((base == NULL) || (trial == NULL) ||
                !mask_rows_valid(base[c], ny) || !mask_rows_valid(trial[c], ny))
            local_bad = 1;
    if (MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (any_bad) return -1;

    for (j = 1; j <= ny; ++j) {
        for (i = 1; i <= nx; ++i) {
            if (mask[j][i] != 1.0f) {
                for (c = 0; c < 3; ++c) {
                    d = fabs((double)trial[c][j][i] - (double)base[c][j][i]);
                    if (!(d <= fixed)) fixed = d;   /* NaN propagates */
                }
            }
            d = (double)trial[0][j][i] - (double)base[0][j][i];
            sums[0] += d * d;
            if ((taper != NULL) && (taper[j][i] < 1.0f)) sums[1] += d * d;
            k = (pos_x * nx + i - 1) * nyg + (pos_y * ny + j - 1);
            if ((fabs(d) > local_max.value) ||
                    ((fabs(d) == local_max.value) && (k < local_max.index))) {
                local_max.value = fabs(d);
                local_max.index = k;
            }
        }
    }
    if (MPI_Allreduce(&fixed, &report->fixed_change_max, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (MPI_Allreduce(sums, global_sums, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
                      MPI_COMM_WORLD) != MPI_SUCCESS) return -2;
    if (taper == NULL) report->taper_share = -1.0;
    else report->taper_share = (global_sums[0] > 0.0) ? global_sums[1] / global_sums[0] : 0.0;
    report->max_change = global_max.value;
    report->max_i = global_max.index / nyg + 1;
    report->max_j = global_max.index % nyg + 1;
    return 0;
}
