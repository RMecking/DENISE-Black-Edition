/* SH-S2 unit harness for src/SH/visco_sh_exact_data_operators.c.
 *
 *   legacy  <in> <out> <ntr> <ns> <order> <fc_high> <fc_low> <dt>
 *           (legacy apply_tdfilt, the stage filter of the exact path)
 *   weights <ns> <dt> <dh> <timewin> <gamma> <twin-> <twin+> <offset_mute>
 *           <offsetc> <src_x> <src_y> <picks_prefix|-> <nrec_global> <mask_last>
 *           stdin: nrec_local lines "global_index gx gy" (recpos grid
 *           indices); stdout: "kept <n>" then one line of ns weights per
 *           local receiver (%.17e), or "status <code>"
 *   picks   <prefix> <ishot> <ntr_glob>   stdout: "status <code>" and values
 *
 * Traces are ASCII, trace-major, one value per line (%.9e round-trips
 * binary32 exactly). */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fd.h"

float DT;

void apply_tdfilt(float **section, int ntr, int ns, int order, float fc2,
                  float fc1);

/* Minimal NR-style allocation as src/util.c (NR_END = 1), for the legacy
 * timedomain_filt and this harness. */
double *dvector(int nl, int nh) {
    double *v = (double *)calloc((size_t)(nh - nl + 2), sizeof(double));
    if (v == NULL) { fprintf(stderr, "alloc\n"); exit(2); }
    return v - nl + 1;
}
void free_dvector(double *v, int nl, int nh) { (void)nh; free(v + nl - 1); }
void err(char err_text[]) { fprintf(stderr, "err: %s\n", err_text); exit(3); }

static float **traces_alloc(int ntr, int ns) {
    float **m = (float **)malloc((size_t)(ntr + 1) * sizeof(float *));
    int i;
    for (i = 1; i <= ntr; ++i)
        m[i] = (float *)calloc((size_t)(ns + 1), sizeof(float));
    return m;
}

static int read_traces(const char *path, float **t, int ntr, int ns) {
    FILE *fp = fopen(path, "r");
    int i, j;
    if (fp == NULL) return -1;
    for (i = 1; i <= ntr; ++i)
        for (j = 1; j <= ns; ++j)
            if (fscanf(fp, "%e", &t[i][j]) != 1) { fclose(fp); return -1; }
    fclose(fp);
    return 0;
}

static int write_traces(const char *path, float **t, int ntr, int ns) {
    FILE *fp = fopen(path, "w");
    int i, j;
    if (fp == NULL) return -1;
    for (i = 1; i <= ntr; ++i)
        for (j = 1; j <= ns; ++j) fprintf(fp, "%.9e\n", t[i][j]);
    fclose(fp);
    return 0;
}

int main(int argc, char **argv) {
    if ((argc == 10) && !strcmp(argv[1], "legacy")) {
        int ntr = atoi(argv[4]), ns = atoi(argv[5]), order = atoi(argv[6]);
        float fc_high = (float)atof(argv[7]), fc_low = (float)atof(argv[8]);
        float **t;
        int status = 0;
        DT = (float)atof(argv[9]);
        t = traces_alloc(ntr, ns);
        if (read_traces(argv[2], t, ntr, ns) != 0) return 4;
        apply_tdfilt(t, ntr, ns, order, fc_high, fc_low);
        printf("status %d\n", status);
        if ((status == 0) && (write_traces(argv[3], t, ntr, ns) != 0)) return 5;
        return 0;
    }
    if ((argc == 16) && !strcmp(argv[1], "weights")) {
        int ns = atoi(argv[2]), timewin = atoi(argv[5]);
        float dt = (float)atof(argv[3]), dh = (float)atof(argv[4]);
        float gamma = (float)atof(argv[6]), twm = (float)atof(argv[7]);
        float twp = (float)atof(argv[8]), offsetc = (float)atof(argv[10]);
        int offset_mute = atoi(argv[9]), nrec_global = atoi(argv[14]);
        int mask_last = atoi(argv[15]);
        float sx = (float)atof(argv[11]), sy = (float)atof(argv[12]);
        float *picks = NULL;
        int **recpos, **recpos_loc, nloc = 0, cap = 64, g, gx, gy, i, k, kept;
        double *w;

        recpos = (int **)malloc(4 * sizeof(int *));
        recpos_loc = (int **)malloc(4 * sizeof(int *));
        for (i = 1; i <= 3; ++i) {
            recpos[i] = (int *)calloc((size_t)(nrec_global + 1), sizeof(int));
            recpos_loc[i] = (int *)calloc((size_t)(cap + 1), sizeof(int));
        }
        while (scanf("%d %d %d", &g, &gx, &gy) == 3) {
            if ((g < 1) || (g > nrec_global) || (nloc >= cap)) return 6;
            ++nloc;
            recpos[1][g] = gx;
            recpos[2][g] = gy;
            recpos_loc[1][nloc] = gx;
            recpos_loc[2][nloc] = gy;
            recpos_loc[3][nloc] = g;
        }
        if (strcmp(argv[13], "-") != 0) {
            int status;
            picks = (float *)calloc((size_t)(nrec_global + 1), sizeof(float));
            status = visco_sh_exact_read_picks(argv[13], 1, nrec_global, picks);
            if (status != 0) { printf("status %d\n", status); return 0; }
        }
        w = (double *)malloc((size_t)(nloc > 0 ? nloc : 1) * (size_t)ns *
                             sizeof(double));
        kept = visco_sh_exact_data_weights(w, nloc, ns, recpos, recpos_loc,
                sx, sy, dh, dt, timewin, gamma, twm, twp, picks, offset_mute,
                offsetc, mask_last);
        if (kept < 0) { printf("status %d\n", kept); return 0; }
        printf("kept %d\n", kept);
        for (i = 0; i < nloc; ++i) {
            for (k = 0; k < ns; ++k)
                printf("%.17e%c", w[(size_t)i * ns + k], k + 1 < ns ? ' ' : '\n');
        }
        return 0;
    }
    if ((argc == 5) && !strcmp(argv[1], "picks")) {
        int ntr = atoi(argv[4]), i, status;
        float *picks = (float *)calloc((size_t)(ntr + 1), sizeof(float));
        status = visco_sh_exact_read_picks(argv[2], atoi(argv[3]), ntr, picks);
        printf("status %d\n", status);
        if (status == 0)
            for (i = 1; i <= ntr; ++i) printf("%.9e\n", picks[i]);
        return 0;
    }
    fprintf(stderr, "usage: see header\n");
    return 1;
}
