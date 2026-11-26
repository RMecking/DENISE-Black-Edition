/*
 * SH-MASK-01 unit harness (docs/sh_mask_01_contract.md, sections 3 and 4):
 * calls the non-collective public API of src/SH/visco_sh_exact_update_mask.c
 * for one subdomain (pos_x, pos_y) of a global NXG x NYG grid.
 *
 * usage: sh_mask_01_harness <case file> <result file>
 *
 * case (native byte order):
 *   int32 op, nx, ny, pos_x, pos_y, nxg, nyg, nsrc, filtsize, fault
 *   float dh, radius
 *   char  tfile[256]                    (zero padded)
 *   float srcpos[nsrc][2]               (x, y in metres)
 *   op 1 (apply_mask_file):     float mask[ny][nx]
 *   op 2 (build_source_taper):  float mask[ny][nx]
 *   op 3 (apply_taper_metric):  float metric[ny][nx], float taper[ny][nx]
 * fault: 0 none; 1 first field argument NULL; 2 last row of the first field
 *   NULL; 3 tfile / srcpos / taper argument NULL (op 1 / 2 / 3).
 * result: int32 status, halo_ok, nfields, then nfields x float[ny][nx]
 *   op 1: mask; op 2: taper, mask; op 3: metric.
 * halo_ok is 1 when no cell outside the own nx x ny cells was written.
 */
#include "fd.h"

#define HALO_SENTINEL (-98765.25f)
#define TAPER_SENTINEL (-12345.5f)

void err(char err_text[]) {
    fprintf(stderr, "unexpected err(): %s\n", err_text);
    exit(3);
}

static float **field_alloc(int nx, int ny, float value) {
    int i, j;
    float **f = (float **)malloc((size_t)(ny + 2) * sizeof(*f));
    if (f == NULL) return NULL;
    for (j = 0; j < ny + 2; ++j) {
        f[j] = (float *)malloc((size_t)(nx + 2) * sizeof(float));
        if (f[j] == NULL) return NULL;
        for (i = 0; i < nx + 2; ++i) f[j][i] = value;
    }
    return f;
}

static int read_own(FILE *fp, float **f, int nx, int ny) {
    int i, j;
    for (j = 1; j <= ny; ++j)
        for (i = 1; i <= nx; ++i)
            if (fread(&f[j][i], sizeof(float), 1, fp) != 1) return -1;
    return 0;
}

static int halo_intact(float **f, int nx, int ny) {
    int i, j;
    for (j = 0; j < ny + 2; ++j)
        for (i = 0; i < nx + 2; ++i)
            if (((j == 0) || (j == ny + 1) || (i == 0) || (i == nx + 1)) &&
                    (f[j][i] != HALO_SENTINEL)) return 0;
    return 1;
}

static void write_own(FILE *fp, float **f, int nx, int ny) {
    int i, j;
    for (j = 1; j <= ny; ++j)
        for (i = 1; i <= nx; ++i) fwrite(&f[j][i], sizeof(float), 1, fp);
}

int main(int argc, char **argv) {
    int head[10], op, nx, ny, pos_x, pos_y, nxg, nyg, nsrc, filtsize, fault;
    int status = -99, halo = 1, nfields = 0, s;
    float par[2], dh, radius, **a = NULL, **b = NULL, **srcpos = NULL, *saved = NULL;
    char tfile[257];
    FILE *fp;

    if (argc != 3) { fprintf(stderr, "usage: see header\n"); return 2; }
    fp = fopen(argv[1], "rb");
    if (fp == NULL) return 4;
    if ((fread(head, sizeof(int), 10, fp) != 10) || (fread(par, sizeof(float), 2, fp) != 2) ||
            (fread(tfile, 1, 256, fp) != 256)) { fclose(fp); return 4; }
    tfile[256] = '\0';
    op = head[0]; nx = head[1]; ny = head[2]; pos_x = head[3]; pos_y = head[4];
    nxg = head[5]; nyg = head[6]; nsrc = head[7]; filtsize = head[8]; fault = head[9];
    dh = par[0]; radius = par[1];
    if ((nx < 1) || (ny < 1) || (nsrc < 0)) { fclose(fp); return 4; }

    srcpos = (float **)malloc(3 * sizeof(*srcpos));
    srcpos[0] = NULL;
    srcpos[1] = (float *)malloc((size_t)(nsrc + 1) * sizeof(float));
    srcpos[2] = (float *)malloc((size_t)(nsrc + 1) * sizeof(float));
    for (s = 1; s <= nsrc; ++s)
        if ((fread(&srcpos[1][s], sizeof(float), 1, fp) != 1) ||
                (fread(&srcpos[2][s], sizeof(float), 1, fp) != 1)) { fclose(fp); return 4; }

    a = field_alloc(nx, ny, HALO_SENTINEL);
    b = field_alloc(nx, ny, HALO_SENTINEL);
    if ((a == NULL) || (b == NULL) || (read_own(fp, a, nx, ny) != 0)) { fclose(fp); return 4; }
    if (op == 3) {
        if (read_own(fp, b, nx, ny) != 0) { fclose(fp); return 4; }
    } else if (op == 2) {
        int i, j;
        for (j = 1; j <= ny; ++j)
            for (i = 1; i <= nx; ++i) b[j][i] = TAPER_SENTINEL;
    }
    fclose(fp);

    saved = (fault == 2) ? a[ny] : NULL;
    if (fault == 2) a[ny] = NULL;
    if (op == 1) {
        status = visco_sh_exact_apply_mask_file(fault == 1 ? NULL : a, nx, ny, pos_x, pos_y,
                                                nxg, nyg, fault == 3 ? NULL : tfile);
        nfields = 1;
    } else if (op == 2) {
        /* the mask (a) is the field whose argument faults 1/2 corrupt */
        status = visco_sh_exact_build_source_taper(b, fault == 1 ? NULL : a, nx, ny, pos_x,
                                                   pos_y, dh, fault == 3 ? NULL : srcpos,
                                                   nsrc, radius, filtsize);
        nfields = 2;
    } else if (op == 3) {
        status = visco_sh_exact_apply_taper_metric(fault == 1 ? NULL : a, fault == 3 ? NULL : b,
                                                   nx, ny);
        nfields = 1;
    } else {
        return 5;
    }
    if (fault == 2) a[ny] = saved;
    halo = halo_intact(a, nx, ny) && halo_intact(b, nx, ny);

    fp = fopen(argv[2], "wb");
    if (fp == NULL) return 6;
    fwrite(&status, sizeof(int), 1, fp);
    fwrite(&halo, sizeof(int), 1, fp);
    fwrite(&nfields, sizeof(int), 1, fp);
    if (op == 2) write_own(fp, b, nx, ny);
    write_own(fp, a, nx, ny);
    fclose(fp);
    return 0;
}
