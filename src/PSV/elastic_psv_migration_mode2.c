#include "fd.h"
#include "denise_elastic_psv_migration.h"
#ifdef DENISE_ENABLE_CUDA_M9_MODE2
#include "denise_cuda_m9_migration.h"
#endif
#include "denise_elastic_psv_migration_mpi.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/stat.h>

/*
 * Production-file adapter for the clean M9c MODE=2 path.  This file performs
 * only strict model/acquisition/prepared-data I/O.  The numerical operation is
 * delegated to denise_elastic_psv_migrate(), which in turn owns M9b-1 contexts.
 */

struct mode2_source {
    int i;
    int j;
    int type;
    float *samples;
};

static char mode2_error[1024];

static int mode2_fail(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(mode2_error, sizeof(mode2_error), format, arguments);
    va_end(arguments);
    return -1;
}

const char *denise_elastic_psv_migration_mode2_last_error(void) {
    return mode2_error[0] ? mode2_error : "no M9c MODE=2 error";
}

static int checked_product(size_t left, size_t right, size_t *product) {
    if (left != 0u && right > SIZE_MAX / left) return -1;
    *product = left * right;
    return 0;
}

static int make_path(char *path, size_t capacity, const char *format,
                     const char *prefix, int shot) {
    int written;
    if (!prefix || !prefix[0]) return mode2_fail("M9c file prefix is empty");
    written = snprintf(path, capacity, format, prefix, shot);
    if (written < 0 || (size_t)written >= capacity)
        return mode2_fail("M9c path constructed from prefix '%s' is too long", prefix);
    return 0;
}

static int exact_file_bytes(FILE *stream, const char *path, size_t expected) {
    long actual;
    if (fseek(stream, 0L, SEEK_END) != 0 || (actual = ftell(stream)) < 0
        || fseek(stream, 0L, SEEK_SET) != 0)
        return mode2_fail("M9c cannot determine exact size of %s", path);
    if ((uintmax_t)actual != (uintmax_t)expected)
        return mode2_fail("M9c file %s has %ld bytes; expected exactly %lu",
                          path, actual, (unsigned long)expected);
    return 0;
}

static int read_float_file(const char *path, size_t count, float **values) {
    FILE *stream = NULL;
    size_t bytes;
    *values = NULL;
    if (checked_product(count, sizeof(float), &bytes) != 0)
        return mode2_fail("M9c float input size overflows for %s", path);
    stream = fopen(path, "rb");
    if (!stream) return mode2_fail("M9c cannot open %s: %s", path, strerror(errno));
    if (exact_file_bytes(stream, path, bytes) != 0) goto failure;
    *values = (float *)malloc(bytes ? bytes : 1u);
    if (!*values) {
        mode2_fail("M9c cannot allocate %lu bytes for %s", (unsigned long)bytes, path);
        goto failure;
    }
    if (count && fread(*values, sizeof(float), count, stream) != count) {
        mode2_fail("M9c short read from %s", path);
        goto failure;
    }
    if (fclose(stream) != 0) {
        stream = NULL;
        mode2_fail("M9c close failed for %s", path);
        goto failure;
    }
    return 0;
failure:
    if (stream) fclose(stream);
    free(*values);
    *values = NULL;
    return -1;
}

static int finite_float_array(const float *values, size_t count, const char *name) {
    size_t index;
    for (index = 0; index < count; ++index)
        if (!isfinite(values[index]))
            return mode2_fail("M9c %s contains NaN or Inf at flat sample %lu",
                              name, (unsigned long)index);
    return 0;
}

static int read_model_grid(const char *prefix, const char *suffix,
                           int nx, int ny, float **row_major) {
    char path[1024];
    float *file_order = NULL;
    size_t cells;
    int written;
    int i, j;
    if (checked_product((size_t)nx, (size_t)ny, &cells) != 0)
        return mode2_fail("M9c NX*NY overflows while reading model");
    written = snprintf(path, sizeof(path), "%s.%s", prefix, suffix);
    if (written < 0 || (size_t)written >= sizeof(path))
        return mode2_fail("M9c model path for %s is too long", suffix);
    if (read_float_file(path, cells, &file_order) != 0) return -1;
    if (finite_float_array(file_order, cells, path) != 0) {
        free(file_order);
        return -1;
    }
    *row_major = (float *)malloc(cells * sizeof(float));
    if (!*row_major) {
        free(file_order);
        return mode2_fail("M9c cannot allocate row-major model grid %s", path);
    }
    /* DENISE model files are x-major (i outer, j inner); M9b-1 is [y][x]. */
    for (i = 0; i < nx; ++i)
        for (j = 0; j < ny; ++j)
            (*row_major)[(size_t)j * (size_t)nx + (size_t)i] =
                file_order[(size_t)i * (size_t)ny + (size_t)j];
    free(file_order);
    return 0;
}

static int grid_index(double coordinate, double reference, float dh,
                      int extent, const char *kind, int ordinal) {
    double scaled = (coordinate + reference) / (double)dh;
    long one_based;
    if (!isfinite(scaled) || scaled < (double)LONG_MIN || scaled > (double)LONG_MAX) {
        mode2_fail("M9c %s %d coordinate is not representable", kind, ordinal);
        return -1;
    }
    /* This is the existing DENISE iround(x/DH) convention, then one conversion. */
    one_based = (long)floor(scaled + 0.5);
    if (one_based < 1 || one_based > extent) {
        mode2_fail("M9c %s %d maps outside the grid", kind, ordinal);
        return -1;
    }
    return (int)one_based - 1;
}

static int read_sources(const char *path, int expected_shots, int nx, int ny,
                        float dh, struct mode2_source **sources) {
    FILE *stream = NULL;
    char line[1024], extra;
    long declared;
    char *end;
    int shot;
    *sources = NULL;
    stream = fopen(path, "r");
    if (!stream) return mode2_fail("M9c cannot open source geometry %s: %s",
                                   path, strerror(errno));
    if (!fgets(line, sizeof(line), stream)) {
        mode2_fail("M9c source geometry %s is empty", path);
        goto failure;
    }
    errno = 0;
    declared = strtol(line, &end, 10);
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') ++end;
    if (errno || *end || declared != expected_shots || declared < 1) {
        mode2_fail("M9c source geometry declares %ld shots; expected exactly %d",
                   declared, expected_shots);
        goto failure;
    }
    *sources = (struct mode2_source *)calloc((size_t)expected_shots, sizeof(**sources));
    if (!*sources) {
        mode2_fail("M9c cannot allocate source metadata");
        goto failure;
    }
    for (shot = 0; shot < expected_shots; ++shot) {
        double x, z, y, shift, frequency, amplitude, azimuth, type_value;
        int type;
        if (!fgets(line, sizeof(line), stream)
            || sscanf(line, "%lf %lf %lf %lf %lf %lf %lf %lf %c",
                      &x, &z, &y, &shift, &frequency, &amplitude,
                      &azimuth, &type_value, &extra) != 8) {
            mode2_fail("M9c source geometry row %d must contain exactly 8 values",
                       shot + 1);
            goto failure;
        }
        (void)z;
        if (!isfinite(type_value) || type_value < (double)INT_MIN
            || type_value > (double)INT_MAX) {
            mode2_fail("M9c shot %d source type is not a finite integer", shot + 1);
            goto failure;
        }
        type = (int)type_value;
        if (type_value != (double)type || type != 1) {
            mode2_fail("M9c shot %d has unsupported source type %.17g; explosive type 1 is required",
                       shot + 1, type_value);
            goto failure;
        }
        if (shift != 0.0 || frequency != 0.0 || amplitude != 1.0 || azimuth != 0.0) {
            mode2_fail("M9c shot %d source metadata must be neutral (tshift=0, fc=0, amp=1, azimuth=0) because the source signal is prepared",
                       shot + 1);
            goto failure;
        }
        (*sources)[shot].i = grid_index(x, 0.0, dh, nx, "source x", shot + 1);
        if ((*sources)[shot].i < 0) goto failure;
        (*sources)[shot].j = grid_index(y, 0.0, dh, ny, "source y", shot + 1);
        if ((*sources)[shot].j < 0) goto failure;
        (*sources)[shot].type = type;
    }
    while (fgets(line, sizeof(line), stream)) {
        char *cursor = line;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') ++cursor;
        if (*cursor) {
            mode2_fail("M9c source geometry %s contains rows beyond declared shot count", path);
            goto failure;
        }
    }
    if (fclose(stream) != 0) {
        stream = NULL;
        mode2_fail("M9c close failed for source geometry %s", path);
        goto failure;
    }
    return 0;
failure:
    if (stream) fclose(stream);
    free(*sources);
    *sources = NULL;
    return -1;
}

static int read_receivers(const char *prefix, int nx, int ny, float dh,
                          const float *reference, int **receiver_i,
                          int **receiver_j, int *receiver_count) {
    char path[1024], line[1024], extra;
    FILE *stream = NULL;
    int capacity = 0, count = 0;
    int *is = NULL, *js = NULL;
    if (make_path(path, sizeof(path), "%s.dat", prefix, 0) != 0) return -1;
    stream = fopen(path, "r");
    if (!stream) return mode2_fail("M9c cannot open receiver geometry %s: %s",
                                   path, strerror(errno));
    while (fgets(line, sizeof(line), stream)) {
        double x, y;
        int i, j, prior;
        char *cursor = line;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') ++cursor;
        if (!*cursor) continue;
        if (sscanf(line, "%lf %lf %c", &x, &y, &extra) != 2) {
            mode2_fail("M9c receiver geometry row %d must contain exactly x and y",
                       count + 1);
            goto failure;
        }
        i = grid_index(x, reference[1], dh, nx, "receiver x", count + 1);
        if (i < 0) goto failure;
        j = grid_index(y, reference[2], dh, ny, "receiver y", count + 1);
        if (j < 0) goto failure;
        for (prior = 0; prior < count; ++prior)
            if (is[prior] == i && js[prior] == j) {
                mode2_fail("M9c receiver %d duplicates receiver %d after grid mapping",
                           count + 1, prior + 1);
                goto failure;
            }
        if (count == capacity) {
            int next = capacity ? capacity * 2 : 16;
            int *new_i = (int *)realloc(is, (size_t)next * sizeof(int));
            int *new_j;
            if (!new_i) {
                mode2_fail("M9c cannot grow receiver geometry");
                goto failure;
            }
            is = new_i;
            new_j = (int *)realloc(js, (size_t)next * sizeof(int));
            if (!new_j) {
                mode2_fail("M9c cannot grow receiver geometry");
                goto failure;
            }
            js = new_j;
            capacity = next;
        }
        is[count] = i;
        js[count] = j;
        ++count;
    }
    if (ferror(stream)) {
        mode2_fail("M9c read failed for receiver geometry %s", path);
        goto failure;
    }
    if (count < 1) {
        mode2_fail("M9c receiver geometry %s contains no receivers", path);
        goto failure;
    }
    if (fclose(stream) != 0) {
        stream = NULL;
        mode2_fail("M9c close failed for receiver geometry %s", path);
        goto failure;
    }
    *receiver_i = is;
    *receiver_j = js;
    *receiver_count = count;
    return 0;
failure:
    if (stream) fclose(stream);
    free(is); free(js);
    return -1;
}

static int validate_cfl(const float *lambda, const float *mu, const float *rho,
                        size_t cells, float dh, float dt, float *vmax_out) {
    size_t cell;
    double vmax = 0.0;
    const double fd4_bound = 29.0 / 12.0;
    for (cell = 0; cell < cells; ++cell) {
        double speed2;
        if (!(rho[cell] > 0.0f) || !(mu[cell] >= 0.0f) || !isfinite(lambda[cell])
            || (mu[cell] == 0.0f && !isfinite(rho[cell])))
            return mode2_fail("M9c lambda/mu/rho model has an invalid cell at flat index %lu",
                              (unsigned long)cell);
        speed2 = ((double)lambda[cell] + 2.0 * (double)mu[cell]) / (double)rho[cell];
        if (!(speed2 > 0.0) || !isfinite(speed2))
            return mode2_fail("M9c model has invalid compressional speed at flat index %lu",
                              (unsigned long)cell);
        if (sqrt(speed2) > vmax) vmax = sqrt(speed2);
    }
    if ((double)dt * vmax * sqrt(2.0) * fd4_bound / (double)dh >= 2.0)
        return mode2_fail("M9c FD4 CFL check failed: DT=%g, vmax=%g, DH=%g",
                          (double)dt, vmax, (double)dh);
    if (vmax > FLT_MAX) return mode2_fail("M9c model vmax overflows float");
    *vmax_out = (float)vmax;
    return 0;
}

static int write_f64_file(const char *path, const double *values, size_t count) {
    FILE *stream = fopen(path, "wb");
    if (!stream) return mode2_fail("M9c cannot create temporary output %s: %s",
                                   path, strerror(errno));
    if (count && fwrite(values, sizeof(double), count, stream) != count) {
        mode2_fail("M9c short write to temporary output %s", path);
        fclose(stream);
        return -1;
    }
    if (fclose(stream) != 0)
        return mode2_fail("M9c close failed for temporary output %s", path);
    return 0;
}

static void free_inputs(struct mode2_source *sources, int shot_count,
                        struct denise_elastic_psv_migration_shot *shots,
                        float *lambda, float *mu, float *rho,
                        int *receiver_i, int *receiver_j) {
    int shot;
    if (sources)
        for (shot = 0; shot < shot_count; ++shot) free(sources[shot].samples);
    if (shots)
        for (shot = 0; shot < shot_count; ++shot) free((void *)shots[shot].migration_data);
    free(sources); free(shots);
    free(lambda); free(mu); free(rho);
    free(receiver_i); free(receiver_j);
}

int denise_elastic_psv_migration_mode2(void) {
    extern int NX, NY, NT, NSHOTS, L, INVMAT1, FDORDER, NDT, DTINV;
    extern int FREE_SURF, BOUNDARY, NP, NPROCX, NPROCY, READMOD, READREC;
    extern int SRCREC, RUN_MULTIPLE_SHOTS, SEISMO, INV_STF, FW, MYID, QUELLART;
    extern float DH, TIME, DT, DAMPING, FPML, npower, k_max_PML, REFREC[4];
    extern char MFILE[STRING_SIZE], SOURCE_FILE[STRING_SIZE], REC_FILE[STRING_SIZE];
    extern char MIGRATION_SOURCE_PREFIX[STRING_SIZE2];
    extern char MIGRATION_DATA_PREFIX[STRING_SIZE2];
    extern char MIGRATION_IMAGE_PREFIX[STRING_SIZE2];
    extern FILE *FP;

    struct denise_elastic_psv_migration_request request;
    struct denise_elastic_psv_migration_result result;
    struct denise_elastic_psv_migration_shot *shots = NULL;
    struct mode2_source *sources = NULL;
    float *lambda = NULL, *mu = NULL, *rho = NULL;
    int *receiver_i = NULL, *receiver_j = NULL;
    int receiver_count = 0, shot;
    size_t cells, data_samples;
    float vmax = 0.0f;
    char path[1024], lambda_final[1024], mu_final[1024];
    char lambda_temp[1024], mu_temp[1024];
    int lambda_published = 0;

    mode2_error[0] = '\0';
    memset(&request, 0, sizeof(request));
    memset(&result, 0, sizeof(result));
    lambda_final[0] = mu_final[0] = lambda_temp[0] = mu_temp[0] = '\0';

#ifdef DENISE_ENABLE_CUDA_M9_MODE2
    if (NP != 1 || NPROCX != 1 || NPROCY != 1)
        return mode2_fail("CUDA-M9e-4 unsupported envelope: exactly one MPI rank and NPROCX=NPROCY=1 required");
    fprintf(stdout,"M9 MODE=2 backend: CUDA-M9e-4 (no CPU fallback)\n");
#else
    if (NP != 1 || NPROCX != 1 || NPROCY != 1) return denise_elastic_psv_migration_mode2_mpi();
    fprintf(stdout,"M9 MODE=2 backend: CPU-M9\n");
#endif
    if (MYID != 0 || NP != 1 || NPROCX != 1 || NPROCY != 1)
        return mode2_fail("M9c MODE=2 requires one MPI rank and NPROCX=NPROCY=1");
    if (make_path(lambda_final, sizeof(lambda_final), "%s.image_lambda_raw.bin",
                  MIGRATION_IMAGE_PREFIX, 0) != 0
        || make_path(mu_final, sizeof(mu_final), "%s.image_mu_raw.bin",
                     MIGRATION_IMAGE_PREFIX, 0) != 0
        || make_path(lambda_temp, sizeof(lambda_temp), "%s.image_lambda_raw.bin.tmp",
                     MIGRATION_IMAGE_PREFIX, 0) != 0
        || make_path(mu_temp, sizeof(mu_temp), "%s.image_mu_raw.bin.tmp",
                     MIGRATION_IMAGE_PREFIX, 0) != 0)
        return -1;

    /* A failed run must not leave either member of a successful-looking pair. */
    remove(lambda_temp); remove(mu_temp); remove(lambda_final); remove(mu_final);

    if (L != 0) return mode2_fail("M9c MODE=2 requires L=0 (got %d)", L);
    if (INVMAT1 != 3)
        return mode2_fail("M9c MODE=2 requires INVMAT1=3 lambda/mu/rho semantics");
    if (FDORDER != 4)
        return mode2_fail("M9c MODE=2 requires FDORDER=4 (got %d)", FDORDER);
    if (NDT != 1 || DTINV != 1)
        return mode2_fail("M9c MODE=2 requires NDT=DTINV=1");
    if (FREE_SURF != 0 && FREE_SURF != 1) return mode2_fail("M9 MODE=2 FREE_SURF must be 0 or 1");
    if (BOUNDARY != 0) return mode2_fail("M9c MODE=2 does not support BOUNDARY");
    if (INV_STF != 0)
        return mode2_fail("M9c MODE=2 requires INV_STF=0 and prepared source samples");
    if (!READMOD) return mode2_fail("M9c MODE=2 requires READMOD=1");
    if (SRCREC != 1 || RUN_MULTIPLE_SHOTS != 1)
        return mode2_fail("M9c MODE=2 requires SRCREC=1 and RUN_MULTIPLE_SHOTS=1");
    if (READREC != 1)
        return mode2_fail("M9c MODE=2 requires fixed receiver geometry READREC=1");
    if (SEISMO != 1)
        return mode2_fail("M9c MODE=2 requires SEISMO=1 direct vx/vy components");
    if (QUELLART != 3)
        return mode2_fail("M9c MODE=2 requires QUELLART=3 prepared source samples");
    if (NSHOTS < 1) return mode2_fail("M9c MODE=2 requires at least one shot");
    if (!(TIME > 0.0f) || !(DT > 0.0f))
        return mode2_fail("M9c MODE=2 requires positive TIME and DT");
    NT = iround(TIME / DT);
    if (NT < 1) return mode2_fail("M9c MODE=2 TIME/DT yields no timesteps");
    if (checked_product((size_t)NX, (size_t)NY, &cells) != 0)
        return mode2_fail("M9c MODE=2 NX*NY overflows size_t");
    if (FW < 0 || 2 * FW >= NX || 2 * FW >= NY)
        return mode2_fail("M9c MODE=2 FW is invalid for this grid");
    if (FW > 0 && (!(DAMPING > 0.0f) || !(npower > 0.0f)
                   || !(k_max_PML >= 1.0f) || !(FPML >= 0.0f)))
        return mode2_fail("M9c MODE=2 CPML parameters are invalid");
    FP = stdout;
    write_par(FP);

    if (read_model_grid(MFILE, "lam", NX, NY, &lambda) != 0
        || read_model_grid(MFILE, "mu", NX, NY, &mu) != 0
        || read_model_grid(MFILE, "rho", NX, NY, &rho) != 0)
        goto failure;
    if (validate_cfl(lambda, mu, rho, cells, DH, DT, &vmax) != 0) goto failure;
    if (read_sources(SOURCE_FILE, NSHOTS, NX, NY, DH, &sources) != 0) goto failure;
    if(FREE_SURF)for(shot=0;shot<NSHOTS;shot++)if(sources[shot].j==0) {mode2_fail("M9 free surface rejects explosive source at j=1");goto failure;}
    if (read_receivers(REC_FILE, NX, NY, DH, REFREC,
                       &receiver_i, &receiver_j, &receiver_count) != 0)
        goto failure;
    if (checked_product((size_t)NT, (size_t)receiver_count, &data_samples) != 0
        || checked_product(data_samples, 2u, &data_samples) != 0) {
        mode2_fail("M9c MODE=2 NT*NREC*2 overflows size_t");
        goto failure;
    }
    shots = (struct denise_elastic_psv_migration_shot *)calloc(
        (size_t)NSHOTS, sizeof(*shots));
    if (!shots) {
        mode2_fail("M9c cannot allocate shot descriptors");
        goto failure;
    }
    for (shot = 0; shot < NSHOTS; ++shot) {
        float *vx = NULL, *vy = NULL, *packed = NULL;
        size_t component_samples;
        if (make_path(path, sizeof(path), "%s.shot_%d.bin",
                      MIGRATION_SOURCE_PREFIX, shot + 1) != 0
            || read_float_file(path, (size_t)NT, &sources[shot].samples) != 0
            || finite_float_array(sources[shot].samples, (size_t)NT,
                                  "prepared source signal") != 0)
            goto failure;
        if (checked_product((size_t)NT, (size_t)receiver_count,
                            &component_samples) != 0) {
            mode2_fail("M9c component sample count overflows size_t");
            goto failure;
        }
        if (make_path(path, sizeof(path), "%s.vx.shot_%d.bin",
                      MIGRATION_DATA_PREFIX, shot + 1) != 0
            || read_float_file(path, component_samples, &vx) != 0)
            goto shot_failure;
        if (make_path(path, sizeof(path), "%s.vy.shot_%d.bin",
                      MIGRATION_DATA_PREFIX, shot + 1) != 0
            || read_float_file(path, component_samples, &vy) != 0)
            goto shot_failure;
        packed = (float *)malloc(data_samples * sizeof(float));
        if (!packed) {
            mode2_fail("M9c cannot allocate prepared migration data for shot %d", shot + 1);
            goto shot_failure;
        }
        if (denise_elastic_psv_migration_pack_components(
                vx, vy, NT, receiver_count, packed) != 0) {
            mode2_fail("M9c shot %d data adapter failed: %s", shot + 1,
                       denise_elastic_psv_migration_last_error());
            goto shot_failure;
        }
        free(vx); free(vy);
        shots[shot].physical_shot_index = shot + 1;
        shots[shot].source_type = sources[shot].type;
        shots[shot].source_i = sources[shot].i;
        shots[shot].source_j = sources[shot].j;
        shots[shot].source_samples = sources[shot].samples;
        shots[shot].receiver_count = receiver_count;
        shots[shot].receiver_i = receiver_i;
        shots[shot].receiver_j = receiver_j;
        shots[shot].migration_data = packed;
        continue;
shot_failure:
        free(vx); free(vy); free(packed);
        goto failure;
    }

    request.nx=NX; request.ny=NY; request.nt=NT; request.fw=FW;
    request.dh=DH; request.dt=DT;
    request.l=L; request.invmat1=INVMAT1; request.fdorder=FDORDER;
    request.ndt=NDT; request.dtinv=DTINV; request.free_surface=FREE_SURF;
    request.boundary=BOUNDARY; request.mpi_size=NP;
    request.receiver_components=2; request.inv_stf=INV_STF;
    request.lambda=lambda; request.mu=mu; request.rho=rho;
    request.cpml_enabled=(FW > 0); request.pml_reflection=0.001f;
    request.pml_power=npower; request.pml_kmax=k_max_PML;
    request.pml_fpml=FPML; request.pml_damping_speed=DAMPING;
    request.shot_count=NSHOTS; request.shots=shots;
#ifdef DENISE_ENABLE_CUDA_M9_MODE2
    if (denise_cuda_m9_migrate_request(&request, &result) != 0) {
        mode2_fail("CUDA-M9e-4 migration failed: %s",denise_cuda_m9_migration_last_error());
#else
    if (denise_elastic_psv_migrate(&request, &result) != 0) {
        mode2_fail("M9c migration failed: %s",
                   denise_elastic_psv_migration_last_error());
#endif
        goto failure;
    }
    if (write_f64_file(lambda_temp, result.image_lambda_raw, result.cell_count) != 0
        || write_f64_file(mu_temp, result.image_mu_raw, result.cell_count) != 0)
        goto failure;
    if (rename(lambda_temp, lambda_final) != 0) {
        mode2_fail("M9c cannot publish %s: %s", lambda_final, strerror(errno));
        goto failure;
    }
    lambda_published = 1;
    if (rename(mu_temp, mu_final) != 0) {
        mode2_fail("M9c cannot publish %s: %s", mu_final, strerror(errno));
        goto failure;
    }
    lambda_published = 0;
    fprintf(FP ? FP : stdout,
            "\nM9c clean elastic P/SV migration complete\n"
            "  shots: %d (strict ascending physical index)\n"
            "  model: lambda/mu/rho, row-major operator view\n"
            "  prepared data: chronological [time][receiver][vx,vy]\n"
            "  image lambda: %s; NX=%d NY=%d float64 bytes=%lu\n"
            "  image mu: %s; NX=%d NY=%d float64 bytes=%lu\n"
            "  trajectory bytes per shot: %lu\n"
            "  trajectory backend: %s\n"
            "  replay segments/checkpoints/max length: %d / %d / %d\n"
            "  checkpoint payload/total payload bytes: %lu / %lu\n"
            "  replay object/pointer/schedule bytes: %lu / %lu / %lu\n"
            "  segment operand/retained replay bytes: %lu / %lu\n"
            "  initial/replayed forward steps per shot: %lu / %lu\n"
            "  global image bytes: %lu\n"
            "  maximum shot data bytes: %lu\n"
#ifdef DENISE_ENABLE_CUDA_M9_MODE2
            "  CPML peak CPU-only diagnostic (not sampled by CUDA): %.9g\n"
#else
            "  CPML memory peak: %.9g\n"
#endif
            "  CFL vmax diagnostic: %.9g m/s\n",
            NSHOTS, lambda_final, NX, NY,
            (unsigned long)(result.cell_count * sizeof(double)),
            mu_final, NX, NY,
            (unsigned long)(result.cell_count * sizeof(double)),
            (unsigned long)result.trajectory_bytes,
            result.segment_count > 0 ? "SEGMENTED" : "FULL",
            result.segment_count, result.checkpoint_count,
            result.max_segment_length,
            (unsigned long)result.checkpoint_payload_bytes,
            (unsigned long)result.checkpoint_bytes,
            (unsigned long)result.checkpoint_metadata_bytes,
            (unsigned long)result.checkpoint_pointer_bytes,
            (unsigned long)result.segment_schedule_bytes,
            (unsigned long)result.segment_operand_bytes,
            (unsigned long)result.peak_replay_storage_bytes,
            (unsigned long)result.initial_forward_steps,
            (unsigned long)result.replayed_steps,
            (unsigned long)result.global_image_bytes,
            (unsigned long)result.maximum_shot_data_bytes,
            (double)result.cpml_memory_peak, (double)vmax);

    denise_elastic_psv_migration_result_destroy(&result);
    free_inputs(sources, NSHOTS, shots, lambda, mu, rho, receiver_i, receiver_j);
    return 0;

failure:
    denise_elastic_psv_migration_result_destroy(&result);
    remove(lambda_temp); remove(mu_temp);
    if (lambda_published) remove(lambda_final);
    remove(lambda_final); remove(mu_final);
    free_inputs(sources, NSHOTS, shots, lambda, mu, rho, receiver_i, receiver_j);
    return -1;
}

/* Root file operations are separated by collective status gates. The adapter
 * deliberately passes WORLD only after excluding independent shot groups. */
static int mode2_collective(int bad) {
    int any;
    char detail[1024];
    bad=bad!=0;
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    snprintf(detail,sizeof(detail),"%s",mode2_error[0]?mode2_error:"remote rank setup/I/O failure");
    return any?mode2_fail("M9d2 collective MODE=2 failure: %s",detail):0;
}
/* Main's legacy count_src aborts before adapter cleanup on a missing geometry
 * file. MODE=2 uses this collective count preflight instead, including 1x1.
 * Successful inputs produce the same NSHOTS and no additional diagnostics. */
int denise_elastic_psv_migration_mode2_count_sources(void) {
    extern char SOURCE_FILE[STRING_SIZE],MIGRATION_IMAGE_PREFIX[STRING_SIZE2];
    extern int NSHOTS;
    FILE *stream=NULL;
    char line[1024],*end,path[1024];
    long declared=0;
    int rank,bad=0,k;
    const char *suffix[4]={"%s.image_lambda_raw.bin","%s.image_mu_raw.bin",
                          "%s.image_lambda_raw.bin.tmp","%s.image_mu_raw.bin.tmp"};
    mode2_error[0]='\0';MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank==0){
        stream=fopen(SOURCE_FILE,"r");
        if(!stream)bad=mode2_fail("M9d2 cannot open source geometry %s: %s",SOURCE_FILE,strerror(errno));
        else {
            if(!fgets(line,sizeof(line),stream))bad=mode2_fail("M9d2 source geometry is empty");
            else {
                errno=0;declared=strtol(line,&end,10);
                while(*end==' '||*end=='\t'||*end=='\r'||*end=='\n')end++;
                if(errno||*end||declared<1||declared>INT_MAX)bad=mode2_fail("M9d2 source geometry has invalid declared shot count");
            }
            if(fclose(stream)!=0&&!bad)bad=mode2_fail("M9d2 cannot close source geometry");
        }
        if(!bad)NSHOTS=(int)declared;
        else for(k=0;k<4;k++)if(make_path(path,sizeof(path),suffix[k],MIGRATION_IMAGE_PREFIX,0)==0)remove(path);
    }
    if(mode2_collective(bad))return -1;
    MPI_Bcast(&NSHOTS,1,MPI_INT,0,MPI_COMM_WORLD);return 0;
}
static int mode2_model_scatter(float *global,float *local,int gx,int nx,int ny,int px,int size,int rank) {
    float *packed=NULL;int r,j;
    if(rank==0)packed=malloc((size_t)gx*ny*(size_t)(size/px)*sizeof(float));
    if(mode2_collective(rank==0&&!packed))return -1;
    if(rank==0)for(r=0;r<size;r++)for(j=0;j<ny;j++)memcpy(packed+(size_t)r*nx*ny+(size_t)j*nx,global+(size_t)(r/px*ny+j)*gx+r%px*nx,(size_t)nx*sizeof(float));
    MPI_Scatter(packed,nx*ny,MPI_FLOAT,local,nx*ny,MPI_FLOAT,0,MPI_COMM_WORLD);
    free(packed);return 0;
}
static int mode2_trace_scatter(float *global,float *local,int nt,int nrec,const int *ri,const int *rj,int nx,int ny,int px,int size,int rank) {
    float *buffer=NULL;int r,t,q,k,n;
    if(rank==0)buffer=malloc((size_t)nt*nrec*2*sizeof(float));
    if(mode2_collective(rank==0&&!buffer))return -1;
    for(r=0;r<size;r++){
        n=0;for(q=0;q<nrec;q++)if(ri[q]/nx+px*(rj[q]/ny)==r)n++;
        if(rank==0){
            int index=0;
            for(t=0;t<nt;t++)for(q=0;q<nrec;q++)if(ri[q]/nx+px*(rj[q]/ny)==r)for(k=0;k<2;k++)buffer[index++]=global[((size_t)t*nrec+q)*2+k];
        }
        if(r==0){if(rank==0)memcpy(local,buffer,(size_t)nt*n*2*sizeof(float));}
        else {if(rank==0)MPI_Send(buffer,nt*n*2,MPI_FLOAT,r,41,MPI_COMM_WORLD);if(rank==r)MPI_Recv(local,nt*n*2,MPI_FLOAT,0,41,MPI_COMM_WORLD,MPI_STATUS_IGNORE);}
    }
    free(buffer);return 0;
}
static int mode2_image_gather(const double *local,double *global,int gx,int nx,int ny,int px,int size,int rank) {
    double *tiles=NULL;int r,j;
    if(rank==0)tiles=malloc((size_t)nx*ny*size*sizeof(double));
    if(mode2_collective(rank==0&&!tiles))return -1;
    MPI_Gather(local,nx*ny,MPI_DOUBLE,tiles,nx*ny,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(rank==0)for(r=0;r<size;r++)for(j=0;j<ny;j++)memcpy(global+(size_t)(r/px*ny+j)*gx+r%px*nx,tiles+(size_t)r*nx*ny+(size_t)j*nx,(size_t)nx*sizeof(double));
    free(tiles);return 0;
}
int denise_elastic_psv_migration_mode2_mpi(void) {
    extern int NX,NY,NT,NSHOTS,L,INVMAT1,FDORDER,NDT,DTINV,FREE_SURF,BOUNDARY,NP,NPROCX,NPROCY,NCOLORS;
    extern int READMOD,READREC,SRCREC,RUN_MULTIPLE_SHOTS,SEISMO,INV_STF,FW,QUELLART;
    extern float DH,TIME,DT,DAMPING,FPML,npower,k_max_PML,REFREC[4];
    extern char MFILE[STRING_SIZE],SOURCE_FILE[STRING_SIZE],REC_FILE[STRING_SIZE];
    extern char MIGRATION_SOURCE_PREFIX[STRING_SIZE2],MIGRATION_DATA_PREFIX[STRING_SIZE2],MIGRATION_IMAGE_PREFIX[STRING_SIZE2];
    struct denise_elastic_psv_migration_request request;
    struct denise_elastic_psv_migration_result result;
    struct denise_elastic_psv_born_mpi_diagnostics diagnostics,*all_diagnostics=NULL;
    struct mode2_source *sources=NULL;
    struct denise_elastic_psv_migration_shot *shots=NULL;
    float *model[3]={NULL,NULL,NULL},*global_model[3]={NULL,NULL,NULL};
    double *image[2]={NULL,NULL};
    int *ri=NULL,*rj=NULL,rank,size,nrec=0,local_nrec=0,s,k,bad=0,nx=0,ny=0;
    size_t cells=0,local_cells=0;
    float vmax=0;
    char path[1024],final[2][1024],temp[2][1024];
    memset(&request,0,sizeof(request));memset(&result,0,sizeof(result));memset(final,0,sizeof(final));memset(temp,0,sizeof(temp));
    mode2_error[0]='\0';
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    /* Establish paths and remove stale pairs on root even for invalid topology. */
    if(rank==0){
        bad=make_path(final[0],sizeof(final[0]),"%s.image_lambda_raw.bin",MIGRATION_IMAGE_PREFIX,0)||make_path(final[1],sizeof(final[1]),"%s.image_mu_raw.bin",MIGRATION_IMAGE_PREFIX,0)||make_path(temp[0],sizeof(temp[0]),"%s.image_lambda_raw.bin.tmp",MIGRATION_IMAGE_PREFIX,0)||make_path(temp[1],sizeof(temp[1]),"%s.image_mu_raw.bin.tmp",MIGRATION_IMAGE_PREFIX,0);
        if(!bad)for(k=0;k<2;k++){remove(final[k]);remove(temp[k]);}
    }
    if(mode2_collective(bad))goto failure;
    bad=NCOLORS!=1||NP!=size||NPROCX<1||NPROCY<1||(long long)NPROCX*NPROCY!=size;
    bad|=NX<5||NY<5||L!=0||INVMAT1!=3||FDORDER!=4||NDT!=1||DTINV!=1||(FREE_SURF!=0&&FREE_SURF!=1)||BOUNDARY!=0||INV_STF!=0||!READMOD||READREC!=1||SRCREC!=1||RUN_MULTIPLE_SHOTS!=1||SEISMO!=1||QUELLART!=3||NSHOTS<1;
    bad|=!isfinite(TIME)||!isfinite(DT)||!isfinite(DH)||!(TIME>0)||!(DT>0)||!(DH>0)||TIME/DT>(float)(INT_MAX-1);
    if(NPROCX>0&&NPROCY>0){nx=NX/NPROCX;ny=NY/NPROCY;bad|=NX%NPROCX!=0||NY%NPROCY!=0||nx<2||ny<(FREE_SURF?4:2);}
    if(mode2_collective(bad)){mode2_fail("M9 unsupported MPI topology/configuration (NCOLORS=1, equal FD4 tiles, L=0, FREE_SURF=0/1, BOUNDARY=0 required)");goto failure;}
    NT=iround(TIME/DT);cells=(size_t)NX*NY;local_cells=(size_t)nx*ny;
    bad=NT<1||cells>INT_MAX||FW<0||2LL*FW>=NX||2LL*FW>=NY;
    if(mode2_collective(bad))goto failure;
    for(k=0;k<3;k++)model[k]=malloc(local_cells*sizeof(float));
    if(mode2_collective(!model[0]||!model[1]||!model[2]))goto failure;
    if(rank==0){
        bad=read_model_grid(MFILE,"lam",NX,NY,&global_model[0])||read_model_grid(MFILE,"mu",NX,NY,&global_model[1])||read_model_grid(MFILE,"rho",NX,NY,&global_model[2]);
        if(!bad)bad=validate_cfl(global_model[0],global_model[1],global_model[2],cells,DH,DT,&vmax);
    }
    if(mode2_collective(bad))goto failure;
    for(k=0;k<3;k++){
        if(mode2_model_scatter(global_model[k],model[k],NX,nx,ny,NPROCX,size,rank))goto failure;
        free(global_model[k]);global_model[k]=NULL;
    }
    MPI_Bcast(&vmax,1,MPI_FLOAT,0,MPI_COMM_WORLD);
    if(rank==0)bad=read_sources(SOURCE_FILE,NSHOTS,NX,NY,DH,&sources)||read_receivers(REC_FILE,NX,NY,DH,REFREC,&ri,&rj,&nrec);
    if(rank==0 && !bad && FREE_SURF)for(s=0;s<NSHOTS;s++)if(sources[s].j==0)bad=mode2_fail("M9 free surface rejects explosive source at j=1")!=0;
    if(mode2_collective(bad))goto failure;
    MPI_Bcast(&nrec,1,MPI_INT,0,MPI_COMM_WORLD);
    if(rank!=0){sources=calloc((size_t)NSHOTS,sizeof(*sources));ri=malloc((size_t)nrec*sizeof(int));rj=malloc((size_t)nrec*sizeof(int));}
    shots=calloc((size_t)NSHOTS,sizeof(*shots));
    if(mode2_collective(!sources||!ri||!rj||!shots||(uint64_t)NT*(uint64_t)nrec*2>INT_MAX))goto failure;
    MPI_Bcast(ri,nrec,MPI_INT,0,MPI_COMM_WORLD);MPI_Bcast(rj,nrec,MPI_INT,0,MPI_COMM_WORLD);
    for(k=0;k<nrec;k++)if(ri[k]/nx+NPROCX*(rj[k]/ny)==rank)local_nrec++;
    for(s=0;s<NSHOTS;s++){
        int geometry[3];float *vx=NULL,*vy=NULL,*packed=NULL,*local=NULL;
        if(rank==0){geometry[0]=sources[s].i;geometry[1]=sources[s].j;geometry[2]=sources[s].type;}
        MPI_Bcast(geometry,3,MPI_INT,0,MPI_COMM_WORLD);
        sources[s].i=geometry[0];sources[s].j=geometry[1];sources[s].type=geometry[2];
        if(rank==0)bad=make_path(path,sizeof(path),"%s.shot_%d.bin",MIGRATION_SOURCE_PREFIX,s+1)||read_float_file(path,(size_t)NT,&sources[s].samples)||finite_float_array(sources[s].samples,(size_t)NT,"prepared source");
        else sources[s].samples=malloc((size_t)NT*sizeof(float));
        if(mode2_collective(bad||!sources[s].samples))goto failure;
        MPI_Bcast(sources[s].samples,NT,MPI_FLOAT,0,MPI_COMM_WORLD);
        local=malloc((size_t)NT*(local_nrec?local_nrec:1)*2*sizeof(float));
        shots[s].migration_data=local;
        if(mode2_collective(!local))goto failure;
        if(rank==0){
            bad=make_path(path,sizeof(path),"%s.vx.shot_%d.bin",MIGRATION_DATA_PREFIX,s+1)||read_float_file(path,(size_t)NT*nrec,&vx);
            if(!bad)bad=make_path(path,sizeof(path),"%s.vy.shot_%d.bin",MIGRATION_DATA_PREFIX,s+1)||read_float_file(path,(size_t)NT*nrec,&vy);
            if(!bad){packed=malloc((size_t)NT*nrec*2*sizeof(float));bad=!packed;if(!bad)bad=denise_elastic_psv_migration_pack_components(vx,vy,NT,nrec,packed)!=0;}
        }
        if(mode2_collective(bad)){free(vx);free(vy);free(packed);goto failure;}
        bad=mode2_trace_scatter(packed,local,NT,nrec,ri,rj,nx,ny,NPROCX,size,rank);
        free(vx);free(vy);free(packed);if(bad)goto failure;
        shots[s].physical_shot_index=s+1;shots[s].source_type=1;shots[s].source_i=geometry[0];shots[s].source_j=geometry[1];shots[s].source_samples=sources[s].samples;shots[s].receiver_count=nrec;shots[s].receiver_i=ri;shots[s].receiver_j=rj;
    }
    request.nx=NX;request.ny=NY;request.nt=NT;request.fw=FW;request.dh=DH;request.dt=DT;request.l=L;request.invmat1=INVMAT1;request.fdorder=FDORDER;request.ndt=NDT;request.dtinv=DTINV;request.free_surface=FREE_SURF;request.boundary=BOUNDARY;request.mpi_size=size;request.receiver_components=2;request.inv_stf=INV_STF;
    request.lambda=model[0];request.mu=model[1];request.rho=model[2];request.cpml_enabled=FW>0;request.pml_reflection=.001f;request.pml_power=npower;request.pml_kmax=k_max_PML;request.pml_fpml=FPML;request.pml_damping_speed=DAMPING;request.shot_count=NSHOTS;request.shots=shots;
    if(denise_elastic_psv_migrate_mpi(&request,MPI_COMM_WORLD,NPROCX,NPROCY,&result,&diagnostics)){mode2_fail("M9d2 MPI migration: %s",denise_elastic_psv_born_mpi_last_error());goto failure;}
    if(rank==0){image[0]=malloc(cells*sizeof(double));image[1]=malloc(cells*sizeof(double));all_diagnostics=malloc((size_t)size*sizeof(diagnostics));}
    if(mode2_collective(rank==0&&(!image[0]||!image[1]||!all_diagnostics)))goto failure;
    if(mode2_image_gather(result.image_lambda_raw,image[0],NX,nx,ny,NPROCX,size,rank)||mode2_image_gather(result.image_mu_raw,image[1],NX,nx,ny,NPROCX,size,rank))goto failure;
    MPI_Gather(&diagnostics,(int)sizeof(diagnostics),MPI_BYTE,all_diagnostics,(int)sizeof(diagnostics),MPI_BYTE,0,MPI_COMM_WORLD);
    /* All ranks have completed before root writes/publishes the pair. */
    if(mode2_collective(0))goto failure;
    if(rank==0){
        bad=write_f64_file(temp[0],image[0],cells)||write_f64_file(temp[1],image[1],cells);
        if(!bad&&rename(temp[0],final[0]))bad=mode2_fail("M9d2 cannot publish lambda: %s",strerror(errno));
        if(!bad&&rename(temp[1],final[1]))bad=mode2_fail("M9d2 cannot publish mu: %s",strerror(errno));
    }
    if(mode2_collective(bad))goto failure;
    if(rank==0){
        printf("M9d2 distributed elastic P/SV MODE=2 complete: %dx%d; %d ascending shots; chronological [time][receiver][vx,vy]; float64 native row-major images\n",NPROCX,NPROCY,NSHOTS);
        for(k=0;k<size;k++){struct denise_elastic_psv_born_mpi_diagnostics *d=&all_diagnostics[k];printf("  rank %d FULL/replay/retained bytes: %lu / %lu / %lu; backend %s; payload/object/pointer/schedule/operand: %lu / %lu / %lu / %lu / %lu; local traces bytes %lu; forward/adjoint halo bytes/step %lu / %lu\n",k,(unsigned long)d->full_bytes,(unsigned long)d->replay_estimate,(unsigned long)d->retained_backend_bytes,d->replay.segmented?"SEGMENTED":"FULL",(unsigned long)d->replay.checkpoint_bytes,(unsigned long)d->replay.checkpoint_metadata_bytes,(unsigned long)d->replay.checkpoint_pointer_bytes,(unsigned long)d->replay.segment_schedule_bytes,(unsigned long)d->replay.segment_operand_bytes,(unsigned long)d->local_data_bytes,(unsigned long)d->forward_halo_bytes_per_step,(unsigned long)d->adjoint_halo_bytes_per_step);}
        printf("  communicator FULL min/max/sum: %llu / %llu / %llu; replay min/max/sum: %llu / %llu / %llu\n",diagnostics.full_min,diagnostics.full_max,diagnostics.full_sum,diagnostics.replay_min,diagnostics.replay_max,diagnostics.replay_sum);
    }
    bad=0;goto cleanup;
failure:
    bad=-1;if(rank==0)for(k=0;k<2;k++){if(final[k][0])remove(final[k]);if(temp[k][0])remove(temp[k]);}
cleanup:
    denise_elastic_psv_migration_result_destroy(&result);
    for(k=0;k<3;k++){free(model[k]);free(global_model[k]);}
    for(k=0;k<2;k++)free(image[k]);
    free(all_diagnostics);
    free_inputs(sources,NSHOTS,shots,NULL,NULL,NULL,ri,rj);return bad;
}
