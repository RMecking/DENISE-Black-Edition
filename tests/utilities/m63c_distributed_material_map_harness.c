/* MPI harness for the actual distributed SH material map and its transpose. */

#include "fd.h"
#include <errno.h>

int NX, NY, MYID, INVMAT1;
int INDEX[5];
const int TAG1 = 1, TAG2 = 2, TAG5 = 5, TAG6 = 6;
FILE *FP;

struct field { float **v; float **rows; float *data; int count; };

float **matrix(int nrl, int nrh, int ncl, int nch) {
    int row, rows = nrh - nrl + 1, cols = nch - ncl + 1;
    float **base = (float **)malloc((size_t)rows * sizeof(float *));
    float *data = (float *)calloc((size_t)rows * (size_t)cols, sizeof(float));
    float **view;
    if (base == NULL || data == NULL) MPI_Abort(MPI_COMM_WORLD, 181);
    view = base - nrl;
    for (row = nrl; row <= nrh; ++row)
        view[row] = data + (row - nrl) * cols - ncl;
    return view;
}

void free_matrix(float **m, int nrl, int nrh, int ncl, int nch) {
    (void)nrh; (void)nch;
    free(m[nrl] + ncl);
    free(m + nrl);
}

void err(char text[]) { fprintf(stderr, "%s\n", text); MPI_Abort(MPI_COMM_WORLD, 182); }

static struct field field_new(void) {
    struct field f;
    int j;
    f.count = (NX + 2) * (NY + 2);
    f.rows = (float **)calloc((size_t)(NY + 2), sizeof(float *));
    f.data = (float *)calloc((size_t)f.count, sizeof(float));
    if (f.rows == NULL || f.data == NULL) MPI_Abort(MPI_COMM_WORLD, 183);
    f.v = f.rows;
    for (j = 0; j <= NY + 1; ++j) f.v[j] = f.data + j * (NX + 2);
    return f;
}

static void field_free(struct field *f) { free(f->data); free(f->rows); }
static void field_copy(struct field *f, const float *source) { memcpy(f->data, source, (size_t)f->count * sizeof(float)); }

static void topology(int rank, int npx, int npy) {
    int x = rank % npx, y = rank / npx;
    INDEX[1] = y * npx + (x + npx - 1) % npx;
    INDEX[2] = y * npx + (x + 1) % npx;
    INDEX[3] = ((y + npy - 1) % npy) * npx + x;
    INDEX[4] = ((y + 1) % npy) * npx + x;
}

static double relative(double a, double b) {
    return fabs(a - b) / fmax(fmax(fabs(a), fabs(b)), 1.0e-300);
}

static void compare(const float *actual, const float *expected, int count,
                    double *difference, double *reference) {
    int k;
    for (k = 0; k < count; ++k) {
        double d = fabs((double)actual[k] - expected[k]);
        if (d > *difference) *difference = d;
        if (fabs(expected[k]) > *reference) *reference = fabs(expected[k]);
    }
}

static void compare_channel(const float *actual, const float *expected,
                            int count, double *difference, double *scale) {
    int k;
    for (k = 0; k < count; ++k) {
        double d = fabs((double)actual[k] - expected[k]);
        double magnitude = fmax(fabs((double)actual[k]), fabs((double)expected[k]));
        if (d > *difference) *difference = d;
        if (magnitude > *scale) *scale = magnitude;
    }
}

static int read_payload(const char *directory, int rank, float *payload, size_t count) {
    char path[4096]; FILE *stream; size_t got;
    snprintf(path, sizeof(path), "%s/rank_%d.bin", directory, rank);
    stream = fopen(path, "rb");
    if (stream == NULL) { fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno)); return -1; }
    got = fread(payload, sizeof(float), count, stream); fclose(stream);
    return got == count ? 0 : -2;
}

static void raw_case(const float *payload) {
    struct field input[3], bars[3], forward[3], transpose[3];
    int field, cells = (NX + 2) * (NY + 2), status;
    double lhs_local = 0.0, rhs_local = 0.0, lhs, rhs;
    double diff_local = 0.0, ref_local = 0.0, diff, ref;
    for (field = 0; field < 3; ++field) {
        int k;
        input[field] = field_new(); bars[field] = field_new();
        forward[field] = field_new(); transpose[field] = field_new();
        field_copy(&input[field], payload + field * cells);
        field_copy(&bars[field], payload + (3 + field) * cells);
        field_copy(&forward[field], input[field].data);
        field_copy(&transpose[field], bars[field].data);
        for (k = 0; k < cells; ++k) lhs_local += (double)bars[field].data[k] * input[field].data[k];
    }
    matcopy_SH(forward[0].v, forward[1].v, forward[2].v);
    status = matcopy_SH_adjoint(transpose[0].v, transpose[1].v, transpose[2].v);
    if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, status);
    lhs_local = rhs_local = 0.0;
    for (field = 0; field < 3; ++field) {
        int k;
        for (k = 0; k < cells; ++k) {
            lhs_local += (double)forward[field].data[k] * bars[field].data[k];
            rhs_local += (double)input[field].data[k] * transpose[field].data[k];
        }
        compare(forward[field].data, payload + (6 + field) * cells, cells, &diff_local, &ref_local);
        compare(transpose[field].data, payload + (9 + field) * cells, cells, &diff_local, &ref_local);
    }
    MPI_Reduce(&lhs_local, &lhs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rhs_local, &rhs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&diff_local, &diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ref_local, &ref, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (MYID == 0) printf("{\"kind\":\"raw\",\"lhs\":%.17g,\"rhs\":%.17g,\"dot_residual\":%.17g,\"reference_error\":%.17g}\n", lhs, rhs, relative(lhs, rhs), diff / fmax(ref, 1.0e-300));
    for (field = 0; field < 3; ++field) { field_free(&input[field]); field_free(&bars[field]); field_free(&forward[field]); field_free(&transpose[field]); }
}

static void local_material_vjp(struct field *primary, struct field *rho,
        struct field *tau, const float *bars, struct field *bp,
        struct field *br, struct field *bt) {
    int i, j, o;
    struct field bm = field_new();
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i) {
        double mc, me, ms, bc = 0.0, be = 0.0, bs = 0.0;
        double bar_tau_cells[4] = {0.0, 0.0, 0.0, 0.0};
        o = (j - 1) * NX + i - 1;
        mc = INVMAT1 == 1 ? rho->v[j][i] * primary->v[j][i] * primary->v[j][i] : primary->v[j][i];
        me = INVMAT1 == 1 ? rho->v[j][i+1] * primary->v[j][i+1] * primary->v[j][i+1] : primary->v[j][i+1];
        ms = INVMAT1 == 1 ? rho->v[j+1][i] * primary->v[j+1][i] * primary->v[j+1][i] : primary->v[j+1][i];
        if (visco_sh_harmonic_pair_vjp(mc, me, bars[o], &bc, &be) != 0) MPI_Abort(MPI_COMM_WORLD, 184);
        if (visco_sh_harmonic_pair_vjp(mc, ms, bars[NX*NY + o], &bc, &bs) != 0) MPI_Abort(MPI_COMM_WORLD, 185);
        bm.v[j][i] += (float)bc; bm.v[j][i+1] += (float)be; bm.v[j+1][i] += (float)bs;
        visco_sh_av_tau_local_vjp(bars[2*NX*NY + o],
                                  bars[3*NX*NY + o], bar_tau_cells);
        bt->v[j][i] += (float)bar_tau_cells[0];
        bt->v[j][i+1] += (float)bar_tau_cells[1];
        bt->v[j+1][i] += (float)bar_tau_cells[2];
        bt->v[j+1][i+1] += (float)bar_tau_cells[3];
        br->v[j][i] += (float)visco_sh_rhoi_vjp(rho->v[j][i], bars[4*NX*NY + o]);
    }
    for (j = 0; j <= NY + 1; ++j) for (i = 0; i <= NX + 1; ++i) {
        if (INVMAT1 == 1) {
            bp->v[j][i] += 2.0f * rho->v[j][i] * primary->v[j][i] * bm.v[j][i];
            br->v[j][i] += primary->v[j][i] * primary->v[j][i] * bm.v[j][i];
        } else bp->v[j][i] += bm.v[j][i];
    }
    field_free(&bm); (void)tau;
}

static void map_case(const float *payload, const struct q_tau_mapping *mapping) {
    int cells = (NX + 2) * (NY + 2), owned = NX * NY, i, j, k, status;
    struct field p = field_new(), r = field_new(), q = field_new(), tau = field_new();
    struct field bp = field_new(), br = field_new(), bt = field_new(), bq = field_new();
    struct field uip = field_new(), ujp = field_new(), taux = field_new(), rhoi = field_new();
    const float *dp = payload + 3*cells, *dr = payload + 4*cells, *dq = payload + 5*cells;
    const float *expected_bp = payload + 6*cells, *expected_br = payload + 7*cells, *expected_bq = payload + 8*cells;
    const float *bars = payload + 9*cells, *expected_output = bars + 5*owned, *expected_jvp = expected_output + 5*owned;
    double lhs_local = 0.0, rhs_local = 0.0, lhs, rhs;
    double channel_diff_local[8] = {0.0}, channel_scale_local[8] = {0.0};
    double channel_diff[8], channel_scale[8], channel_error[8], reference_error = 0.0;
    field_copy(&p, payload); field_copy(&r, payload + cells); field_copy(&q, payload + 2*cells);
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i)
        tau.v[j][i] = q_to_tau(q.v[j][i], mapping);
    matcopy_SH(r.v, p.v, tau.v);
    av_mu_SH(p.v, uip.v, ujp.v, r.v); inv_rho_SH(r.v, rhoi.v); av_tau(tau.v, taux.v);
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i) {
        int o = (j - 1) * NX + i - 1;
        float actual[5] = {uip.v[j][i], ujp.v[j][i], taux.v[j][i], tau.v[j][i], rhoi.v[j][i]};
        int f;
        for (f = 0; f < 5; ++f) {
            compare_channel(&actual[f], expected_output + f*owned + o, 1,
                            &channel_diff_local[f], &channel_scale_local[f]);
            lhs_local += (double)expected_jvp[f*owned + o] * bars[f*owned + o];
        }
    }
    local_material_vjp(&p, &r, &tau, bars, &bp, &br, &bt);
    status = matcopy_SH_adjoint(br.v, bp.v, bt.v); if (status != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, status);
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i) {
        k = j * (NX + 2) + i;
        bq.v[j][i] = (float)(q_to_tau_derivative(q.v[j][i], mapping) * bt.v[j][i]);
        rhs_local += (double)dp[k] * bp.v[j][i] + (double)dr[k] * br.v[j][i] + (double)dq[k] * bq.v[j][i];
    }
    compare_channel(bp.data, expected_bp, cells,
                    &channel_diff_local[5], &channel_scale_local[5]);
    compare_channel(br.data, expected_br, cells,
                    &channel_diff_local[6], &channel_scale_local[6]);
    compare_channel(bq.data, expected_bq, cells,
                    &channel_diff_local[7], &channel_scale_local[7]);
    MPI_Reduce(&lhs_local, &lhs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rhs_local, &rhs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(channel_diff_local, channel_diff, 8, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(channel_scale_local, channel_scale, 8, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (MYID == 0) {
        int channel;
        for (channel = 0; channel < 8; ++channel) {
            channel_error[channel] = channel_diff[channel]
                                   / fmax(channel_scale[channel], 1.0e-300);
            if (channel_error[channel] > reference_error)
                reference_error = channel_error[channel];
        }
        printf("{\"kind\":\"map\",\"lhs\":%.17g,\"rhs\":%.17g,"
               "\"dot_residual\":%.17g,\"reference_error\":%.17g,"
               "\"reference_errors\":{\"mu_x\":%.17g,\"mu_y\":%.17g,"
               "\"tau_x\":%.17g,\"tau_y\":%.17g,\"rhoi\":%.17g,"
               "\"bar_primary\":%.17g,\"bar_rho\":%.17g,\"bar_q\":%.17g}}\n",
               lhs, rhs, relative(lhs, rhs), reference_error,
               channel_error[0], channel_error[1], channel_error[2],
               channel_error[3], channel_error[4], channel_error[5],
               channel_error[6], channel_error[7]);
    }
    field_free(&p); field_free(&r); field_free(&q); field_free(&tau); field_free(&bp); field_free(&br); field_free(&bt); field_free(&bq); field_free(&uip); field_free(&ujp); field_free(&taux); field_free(&rhoi);
}

int main(int argc, char **argv) {
    int size, npx, npy, mode, qmode, cells, owned; size_t count;
    float *payload, frequencies[4] = {0.0f, 3.0f, 7.0f, 13.0f};
    struct q_tau_mapping mapping; void *bsend_buffer; int bsend_size = 1 << 20;
    MPI_Init(&argc, &argv); MPI_Comm_rank(MPI_COMM_WORLD, &MYID); MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 9) MPI_Abort(MPI_COMM_WORLD, 186);
    npx = atoi(argv[1]); npy = atoi(argv[2]); NX = atoi(argv[3]); NY = atoi(argv[4]); mode = strcmp(argv[5], "map") == 0; INVMAT1 = atoi(argv[6]); qmode = atoi(argv[7]);
    if (size != npx*npy) MPI_Abort(MPI_COMM_WORLD, 187); topology(MYID, npx, npy);
    FP = tmpfile(); bsend_buffer = malloc((size_t)bsend_size); if (FP == NULL || bsend_buffer == NULL) MPI_Abort(MPI_COMM_WORLD, 188); MPI_Buffer_attach(bsend_buffer, bsend_size);
    cells = (NX + 2) * (NY + 2); owned = NX * NY;
    count = mode ? (size_t)(9*cells + 15*owned) : (size_t)(12*cells);
    payload = (float *)malloc(count*sizeof(float)); if (payload == NULL || read_payload(argv[8], MYID, payload, count) != 0) MPI_Abort(MPI_COMM_WORLD, 189);
    if (mode) { init_q_tau_mapping(&mapping, qmode, qmode ? 3 : 1, frequencies, 2.0f, 18.0f, 0.5f); map_case(payload, &mapping); }
    else raw_case(payload);
    free(payload); fclose(FP); MPI_Buffer_detach(&bsend_buffer, &bsend_size); free(bsend_buffer); MPI_Finalize(); return 0;
}
