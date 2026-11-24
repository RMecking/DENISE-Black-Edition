/* SH-S3 unit harness: runs the source time function estimator
 * (src/SH/visco_sh_exact_stf.c) on traces from a binary input file.
 *
 *   sh_s3_stf_harness <in.bin> <out.bin>
 *
 * in.bin  (native endian): int32 ntr, nt, nw, weighted; float64 eps_rel;
 *         float32 g[ntr*nt]; float32 d[ntr*nt]; float64 w[ntr*nt] if weighted.
 * out.bin: int32 status, iterations; float64 mean_diag, eps, relres,
 *          data_term, eps_term; float64 v[nw].
 * Prints "status <s>". */
#include <stdio.h>
#include <stdlib.h>
#include "fd.h"

int main(int argc, char **argv) {
    FILE *fp;
    int head[4], status = -9, iterations = 0;
    double eps_rel, md = 0.0, eps = 0.0, relres = 0.0, dterm = 0.0, eterm = 0.0;
    float *g, *d;
    double *w = NULL, *v;
    struct visco_sh_exact_stf_problem p;
    size_t n;

    if (argc != 3) return 2;
    fp = fopen(argv[1], "rb");
    if (!fp || fread(head, sizeof(int), 4, fp) != 4 || fread(&eps_rel, sizeof(double), 1, fp) != 1)
        return 3;
    n = (size_t)head[0] * (size_t)head[1];
    g = (float *)malloc(n * sizeof(float));
    d = (float *)malloc(n * sizeof(float));
    v = (double *)calloc((size_t)(head[2] > 0 ? head[2] : 1), sizeof(double));
    if (head[3]) w = (double *)malloc(n * sizeof(double));
    if (!g || !d || !v || (head[3] && !w)) return 4;
    if (fread(g, sizeof(float), n, fp) != n || fread(d, sizeof(float), n, fp) != n ||
            (head[3] && fread(w, sizeof(double), n, fp) != n)) return 5;
    fclose(fp);
    p.ntr = head[0]; p.nt = head[1]; p.nw = head[2]; p.g = g; p.d = d; p.w = w;
    md = visco_sh_exact_stf_mean_diag(&p);
    eps = eps_rel * md;
    status = visco_sh_exact_stf_solve(&p, eps, 1.0e-11, 5000, v, &iterations, &relres);
    if (status == 0) status = visco_sh_exact_stf_terms(&p, v, eps, &dterm, &eterm);
    fp = fopen(argv[2], "wb");
    if (!fp) return 6;
    fwrite(&status, sizeof(int), 1, fp);
    fwrite(&iterations, sizeof(int), 1, fp);
    fwrite(&md, sizeof(double), 1, fp);
    fwrite(&eps, sizeof(double), 1, fp);
    fwrite(&relres, sizeof(double), 1, fp);
    fwrite(&dterm, sizeof(double), 1, fp);
    fwrite(&eterm, sizeof(double), 1, fp);
    fwrite(v, sizeof(double), (size_t)(head[2] > 0 ? head[2] : 0), fp);
    fclose(fp);
    printf("status %d iterations %d relres %e\n", status, iterations, relres);
    return 0;
}
