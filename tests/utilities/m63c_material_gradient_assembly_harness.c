/* MPI harness for the C7c-a temporal and distributed assembly primitives. */

#include "fd.h"
#include <errno.h>

int NX, NY, MYID, INVMAT1;
int INDEX[5];
const int TAG1 = 1, TAG2 = 2, TAG5 = 5, TAG6 = 6;
FILE *FP;

struct field { float **v; float **rows; float *data; int count; };
struct dfield { double **v; double **rows; double *data; int count; };

float **matrix(int nrl, int nrh, int ncl, int nch) {
    int row, rows = nrh - nrl + 1, cols = nch - ncl + 1;
    float **base = (float **)malloc((size_t)rows * sizeof(float *));
    float *data = (float *)calloc((size_t)rows * cols, sizeof(float));
    float **view;
    if (base == NULL || data == NULL) MPI_Abort(MPI_COMM_WORLD, 701);
    view = base - nrl;
    for (row = nrl; row <= nrh; ++row)
        view[row] = data + (row - nrl) * cols - ncl;
    return view;
}

void free_matrix(float **m, int nrl, int nrh, int ncl, int nch) {
    (void)nrh; (void)nch; free(m[nrl] + ncl); free(m + nrl);
}

void err(char text[]) {
    fprintf(stderr, "%s\n", text); MPI_Abort(MPI_COMM_WORLD, 702);
}

static struct field field_new(void) {
    struct field f; int j;
    f.count = (NX + 2) * (NY + 2);
    f.rows = (float **)calloc((size_t)(NY + 2), sizeof(float *));
    f.data = (float *)calloc((size_t)f.count, sizeof(float));
    if (f.rows == NULL || f.data == NULL) MPI_Abort(MPI_COMM_WORLD, 703);
    f.v = f.rows;
    for (j = 0; j <= NY + 1; ++j) f.v[j] = f.data + j * (NX + 2);
    return f;
}

static struct dfield dfield_new(void) {
    struct dfield f; int j;
    f.count = (NX + 2) * (NY + 2);
    f.rows = (double **)calloc((size_t)(NY + 2), sizeof(double *));
    f.data = (double *)calloc((size_t)f.count, sizeof(double));
    if (f.rows == NULL || f.data == NULL) MPI_Abort(MPI_COMM_WORLD, 704);
    f.v = f.rows;
    for (j = 0; j <= NY + 1; ++j) f.v[j] = f.data + j * (NX + 2);
    return f;
}

static void field_free(struct field *f) { free(f->data); free(f->rows); }
static void dfield_free(struct dfield *f) { free(f->data); free(f->rows); }
static void field_copy(struct field *f, const float *source) {
    memcpy(f->data, source, (size_t)f->count * sizeof(float));
}

static void topology(int rank, int npx, int npy) {
    int x = rank % npx, y = rank / npx;
    INDEX[1] = y * npx + (x + npx - 1) % npx;
    INDEX[2] = y * npx + (x + 1) % npx;
    INDEX[3] = ((y + npy - 1) % npy) * npx + x;
    INDEX[4] = ((y + 1) % npy) * npx + x;
}

static int read_payload(const char *directory, int rank, float *payload,
                        size_t count) {
    char path[4096]; FILE *stream; size_t got;
    snprintf(path, sizeof(path), "%s/rank_%d.bin", directory, rank);
    stream = fopen(path, "rb");
    if (stream == NULL) {
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    got = fread(payload, sizeof(float), count, stream); fclose(stream);
    return got == count ? 0 : -2;
}

static void compare(double actual, double expected,
                    double *difference, double *scale) {
    double delta = fabs(actual - expected);
    double magnitude = fmax(fabs(actual), fabs(expected));
    if (delta > *difference) *difference = delta;
    if (magnitude > *scale) *scale = magnitude;
}

int main(int argc, char **argv) {
    int size, npx, npy, qmode, nsteps, dtinv, cells, owned;
    int i, j, n, channel, point, status;
    size_t count, offset;
    double driver_dt, temporal_diff = 0.0, temporal_scale = 0.0;
    double temporal_checksum = 0.0, global_temporal_checksum;
    double map_diff[3] = {0.0}, map_scale[3] = {0.0};
    double reduced_diff[3], reduced_scale[3], wrong_diff = 0.0, wrong_scale = 0.0;
    double global_temporal_diff, global_temporal_scale;
    float *payload, frequencies[4] = {0.0f, 3.0f, 7.0f, 13.0f};
    struct q_tau_mapping mapping;
    struct field primary, rho, q, tau, gp, gr, gq;
    struct dfield native[5];
    struct visco_sh_native_material_gradient_fields native_fields;
    struct visco_sh_material_timestep_vjp_output *series, *series_copy, *sum;
    void *bsend_buffer; int bsend_size = 1 << 20;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MYID);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 11) MPI_Abort(MPI_COMM_WORLD, 705);
    npx = atoi(argv[1]); npy = atoi(argv[2]); NX = atoi(argv[3]); NY = atoi(argv[4]);
    INVMAT1 = atoi(argv[5]); qmode = atoi(argv[6]); nsteps = atoi(argv[7]);
    driver_dt = atof(argv[8]); dtinv = atoi(argv[9]);
    if (size != npx * npy || nsteps < 1) MPI_Abort(MPI_COMM_WORLD, 706);
    topology(MYID, npx, npy);
    FP = tmpfile(); bsend_buffer = malloc((size_t)bsend_size);
    if (FP == NULL || bsend_buffer == NULL) MPI_Abort(MPI_COMM_WORLD, 707);
    MPI_Buffer_attach(bsend_buffer, bsend_size);

    cells = (NX + 2) * (NY + 2); owned = NX * NY;
    count = (size_t)(3 * cells + nsteps * 5 * owned + 5 * owned
                     + 3 * cells + cells);
    payload = (float *)malloc(count * sizeof(float));
    if (payload == NULL || read_payload(argv[10], MYID, payload, count) != 0)
        MPI_Abort(MPI_COMM_WORLD, 708);

    primary = field_new(); rho = field_new(); q = field_new(); tau = field_new();
    gp = field_new(); gr = field_new(); gq = field_new();
    for (channel = 0; channel < 5; ++channel) native[channel] = dfield_new();
    field_copy(&primary, payload); field_copy(&rho, payload + cells);
    field_copy(&q, payload + 2 * cells);
    init_q_tau_mapping(&mapping, qmode, qmode ? 3 : 1, frequencies,
                       2.0f, 18.0f, 0.5f);
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i)
        tau.v[j][i] = q_to_tau(q.v[j][i], &mapping);
    matcopy_SH(rho.v, primary.v, tau.v);

    offset = (size_t)3 * cells;
    series = (struct visco_sh_material_timestep_vjp_output *)calloc(
            (size_t)nsteps * owned, sizeof(*series));
    series_copy = (struct visco_sh_material_timestep_vjp_output *)calloc(
            (size_t)nsteps * owned, sizeof(*series_copy));
    sum = (struct visco_sh_material_timestep_vjp_output *)calloc(
            (size_t)owned, sizeof(*sum));
    if (series == NULL || series_copy == NULL || sum == NULL)
        MPI_Abort(MPI_COMM_WORLD, 709);
    for (n = 0; n < nsteps; ++n) for (channel = 0; channel < 5; ++channel)
        for (point = 0; point < owned; ++point) {
            double value = payload[offset + (size_t)(n * 5 + channel) * owned + point];
            struct visco_sh_material_timestep_vjp_output *entry =
                    &series[n * owned + point];
            if (channel == 0) entry->g_rhoi = value;
            if (channel == 1) entry->g_mu_x = value;
            if (channel == 2) entry->g_mu_y = value;
            if (channel == 3) entry->g_tau_x = value;
            if (channel == 4) entry->g_tau_y = value;
        }
    memcpy(series_copy, series, (size_t)nsteps * owned * sizeof(*series));
    status = visco_sh_temporal_native_gradient_accumulate(
            nsteps, owned, dtinv, series, sum);
    if (dtinv != 1) {
        if (MYID == 0)
            printf("{\"temporal_status\":%d,\"driver_dt\":%.17g}\n",
                   status, driver_dt);
        free(payload); free(series); free(series_copy); free(sum);
        field_free(&primary); field_free(&rho); field_free(&q); field_free(&tau);
        field_free(&gp); field_free(&gr); field_free(&gq);
        for (channel = 0; channel < 5; ++channel) dfield_free(&native[channel]);
        fclose(FP); MPI_Buffer_detach(&bsend_buffer, &bsend_size);
        free(bsend_buffer); MPI_Finalize(); return 0;
    }
    if (status != 0 || memcmp(series, series_copy,
                              (size_t)nsteps * owned * sizeof(*series)) != 0)
        MPI_Abort(MPI_COMM_WORLD, 710);
    offset += (size_t)nsteps * 5 * owned;
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i) {
        point = (j - 1) * NX + i - 1;
        native[0].v[j][i] = sum[point].g_rhoi;
        native[1].v[j][i] = sum[point].g_mu_x;
        native[2].v[j][i] = sum[point].g_mu_y;
        native[3].v[j][i] = sum[point].g_tau_x;
        native[4].v[j][i] = sum[point].g_tau_y;
        for (channel = 0; channel < 5; ++channel)
            compare(native[channel].v[j][i],
                    payload[offset + (size_t)channel * owned + point],
                    &temporal_diff, &temporal_scale);
        temporal_checksum += (point + 1) * (
                sum[point].g_rhoi + 2.0 * sum[point].g_mu_x +
                3.0 * sum[point].g_mu_y + 4.0 * sum[point].g_tau_x +
                5.0 * sum[point].g_tau_y);
    }
    offset += (size_t)5 * owned;
    native_fields.g_rhoi = native[0].v;
    native_fields.g_mu_x = native[1].v; native_fields.g_mu_y = native[2].v;
    native_fields.g_tau_x = native[3].v; native_fields.g_tau_y = native[4].v;
    status = visco_sh_distributed_material_gradient_vjp(
            INVMAT1, &mapping, primary.v, rho.v, q.v, &native_fields,
            gp.v, gr.v, gq.v);
    if (status != 0) MPI_Abort(MPI_COMM_WORLD, 711);
    for (point = 0; point < cells; ++point) {
        compare(gp.data[point], payload[offset + point], &map_diff[0], &map_scale[0]);
        compare(gr.data[point], payload[offset + cells + point], &map_diff[1], &map_scale[1]);
        compare(gq.data[point], payload[offset + 2 * cells + point], &map_diff[2], &map_scale[2]);
        compare(gq.data[point], payload[offset + 3 * cells + point], &wrong_diff, &wrong_scale);
    }
    MPI_Reduce(&temporal_diff, &global_temporal_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&temporal_scale, &global_temporal_scale, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&temporal_checksum, &global_temporal_checksum, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(map_diff, reduced_diff, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(map_scale, reduced_scale, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    {
        double global_wrong_diff, global_wrong_scale;
        MPI_Reduce(&wrong_diff, &global_wrong_diff, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&wrong_scale, &global_wrong_scale, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (MYID == 0) printf(
            "{\"temporal_error\":%.17g,\"primary_error\":%.17g,"
            "\"rho_error\":%.17g,\"q_error\":%.17g,"
            "\"wrong_q_difference\":%.17g,\"temporal_checksum\":%.17g}\n",
            global_temporal_diff / fmax(global_temporal_scale, 1.0e-300),
            reduced_diff[0] / fmax(reduced_scale[0], 1.0e-300),
            reduced_diff[1] / fmax(reduced_scale[1], 1.0e-300),
            reduced_diff[2] / fmax(reduced_scale[2], 1.0e-300),
            global_wrong_diff / fmax(global_wrong_scale, 1.0e-300),
            global_temporal_checksum);
    }

    free(payload); free(series); free(series_copy); free(sum);
    field_free(&primary); field_free(&rho); field_free(&q); field_free(&tau);
    field_free(&gp); field_free(&gr); field_free(&gq);
    for (channel = 0; channel < 5; ++channel) dfield_free(&native[channel]);
    fclose(FP); MPI_Buffer_detach(&bsend_buffer, &bsend_size); free(bsend_buffer);
    MPI_Finalize(); return 0;
}
