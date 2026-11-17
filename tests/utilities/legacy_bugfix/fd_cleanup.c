#include "fd.h"
#include <sanitizer/lsan_interface.h>

enum { GRID_X = 12, GRID_Y = 10, STEPS = 10, MAX_SHOTS = 12 };
enum kind { SIGNAL, GEOMETRY };

int NT = STEPS, MYID, N_ORDER, FDORDER = 2, POS[3], IENDX = GRID_X, IENDY = GRID_Y;
int QUELLART = 2, QUELLTYP, QUELLTYPB, ORDER_SPIKE, RUN_MULTIPLE_SHOTS;
int GRAD_FORM = 2, ADJ_SIGN = 1, L, FW, FREE_SURF, BOUNDARY, NPROCX = 2, NPROCY = 1;
int NDT = 1, SEISMO = 1, INVMAT1 = 3;
float DH = 1.0f, DT = .0625f, TS = 1.0f, FC_SPIKE_1, FC_SPIKE_2;
char SIGNAL_FILE[STRING_SIZE] = "unused", MFILE[STRING_SIZE] = "fd-sh-test";
FILE *FP;

struct allocation {
    float **pointer;
    enum kind kind;
    int row_high, column_high, shot;
};
static struct allocation owned[2 * MAX_SHOTS];
static size_t registered;
static int allocs[2], frees[2], shot_number, expect_count, releasing, exports;

extern float **__real_splitsrc(float **global, int *local_count, int count);
extern float **__real_wavelet(float **local, int count, int shot);
extern void __real_free_matrix(float **m, int nrl, int nrh, int ncl, int nch);

static void need(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL shot %d: %s\n", shot_number, message);
        exit(43);
    }
}

int MPI_Barrier(MPI_Comm communicator)
{
    (void)communicator;
    return 0;
}

int MPI_Abort(MPI_Comm communicator, int error)
{
    (void)communicator;
    fprintf(stderr, "unexpected MPI_Abort %d\n", error);
    exit(44);
}

void apply_tdfilt(float **s, int ntr, int ns, int order, float fc2, float fc1)
{
    (void)s; (void)ntr; (void)ns; (void)order; (void)fc2; (void)fc1;
    need(0, "test does not select spike filtering");
}

void output_source_signal(FILE *file, float **signals, int ns, int format)
{
    need(file && signals == acq.signals && expect_count == 1 && ns == NT &&
         format == 1 && !releasing, "source export lifetime");
    need(fclose(file) == 0, "close test export stream");
    ++exports;
}

static void remember(float **ptr, enum kind kind, int row_high, int col_high)
{
    need(ptr && registered < sizeof(owned)/sizeof(owned[0]), "owned allocation");
    for (size_t k = 0; k < registered; ++k)
        need(owned[k].pointer != ptr, "one live handle per allocation");
    owned[registered++] = (struct allocation){ptr, kind, row_high, col_high, shot_number};
    ++allocs[kind];
}

float **__wrap_splitsrc(float **source, int *count, int nsrc)
{
    float **local;
    need(*count == 0 && !releasing, "source count reset before split");
    local = __real_splitsrc(source, count, nsrc);
    need(*count == expect_count, "actual splitsrc count");
    if (*count > 0)
        remember(local, GEOMETRY, 8, *count);
    else
        need(local == NULL, "empty local geometry pointer");
    return local;
}

float **__wrap_wavelet(float **local, int count, int shot)
{
    float **signals;
    need(local == acq.srcpos_loc && count == expect_count && shot == shot_number &&
         !releasing, "actual wavelet request");
    signals = __real_wavelet(local, count, shot);
    remember(signals, SIGNAL, count, NT);
    return signals;
}

void __wrap_free_matrix(float **matrix, int nrl, int nrh, int ncl, int nch)
{
    for (size_t k = 0; matrix && k < registered; ++k) {
        struct allocation *entry = &owned[k];
        if (entry->pointer != matrix)
            continue;
        need(entry->shot == shot_number && releasing, "release after model in same shot");
        need(nrl == 1 && ncl == 1 && nrh == entry->row_high &&
             nch == entry->column_high, "release with original allocation bounds");
        need(matrix == (entry->kind == SIGNAL ? acq.signals : acq.srcpos_loc),
             "release of actual active acquisition pointer");
        ++frees[entry->kind];
        entry->pointer = NULL;
        break;
    }
    __real_free_matrix(matrix, nrl, nrh, ncl, nch);
}

static void write_words(FILE *out, const float *words, size_t count)
{
    need(fwrite(words, sizeof(*words), count, out) == count, "result write");
}

static void model_shot(int local, FILE *raw)
{
    float hc[] = {0, 1};
    int rx[] = {0, 7, 9}, ry[] = {0, 5, 5};
    int *rec[] = {NULL, rx, ry};
    float **data = matrix(1, 2, 1, STEPS);
    float **fields[9];
    int has_state = 0;
    need(local == expect_count && !releasing, "model before release");
    need(acq.signals != NULL && (local == 0) == (acq.srcpos_loc == NULL),
         "actual source pointers at model entry");
    for (size_t f = 0; f < sizeof(fields)/sizeof(fields[0]); ++f)
        fields[f] = matrix(-1, GRID_Y+2, -1, GRID_X+2);
    float **vz = fields[0], **vzp1 = fields[1], **sxz = fields[2], **syz = fields[3],
          **rho = fields[4], **rhoi = fields[5], **mu = fields[6],
          **uz = fields[7], **uzx = fields[8];
    for (int y = 1; y <= GRID_Y; ++y)
        for (int x = 1; x <= GRID_X; ++x) {
            rho[y][x] = 2;
            rhoi[y][x] = .5f;
            mu[y][x] = 3;
        }
    for (int step = 1; step <= STEPS; ++step) {
        update_v_PML_SH(1, GRID_X, 1, GRID_Y, step, vz, vzp1, NULL, NULL,
            sxz, syz, rho, rhoi, acq.srcpos_loc, acq.signals, local, NULL,
            hc, 0, 0, 0,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL);
        update_s_elastic_PML_SH(1, GRID_X, 1, GRID_Y, vz, uz, uzx, syz, sxz,
            mu, mu, rho, hc, 0,
            NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            NULL, NULL, 0);
        seismo_ssg(step, 2, rec, data, data, NULL, NULL, NULL,
                   vz, vz, NULL, NULL, NULL, NULL, NULL, hc);
    }
    if (raw) {
        for (int src = 1; src <= local; ++src)
            write_words(raw, &acq.signals[src][1], STEPS);
        for (int receiver = 1; receiver <= 2; ++receiver)
            write_words(raw, &data[receiver][1], STEPS);
    }
    for (int f = 0; f < 4; ++f)
        for (int y = 1; y <= GRID_Y; ++y) {
            for (int x = 1; x <= GRID_X; ++x) {
                const float value = fields[f][y][x];
                need(isfinite(value), "finite wavefield");
                has_state |= value != 0.0f;
            }
            if (raw)
                write_words(raw, &fields[f][y][1], GRID_X);
        }
    need(has_state == (local > 0), "wavefield reflects local source presence");
    for (size_t f = 0; f < sizeof(fields)/sizeof(fields[0]); ++f)
        free_matrix(fields[f], -1, GRID_Y+2, -1, GRID_X+2);
    free_matrix(data, 1, 2, 1, STEPS);
}

static void run_shots(const int *local_counts, int shots, int nsrc, FILE *raw)
{
    // The included regions are verbatim from the pinned or corrected FD_SH.c.
    int nt, ns = NT, ishot, nsrc_loc = 0;
    int geometry_expected = 0;
    for (ishot = 1; ishot <= shots; ++ishot) {
        shot_number = ishot;
        expect_count = local_counts[ishot-1];
        releasing = 0;
        geometry_expected += expect_count > 0;
        if (!HISTORICAL)
            need(!acq.signals && !acq.srcpos_loc, "no inherited handles at shot start");
#include "prepare.inc"
        model_shot(nsrc_loc, raw);
        releasing = 1;
#include "finish.inc"
        need(nsrc_loc == 0, "count reset at shot end");
        need(allocs[SIGNAL] == ishot && allocs[GEOMETRY] == geometry_expected,
             "one signal and optional geometry allocation per shot");
        if (HISTORICAL) {
            need(frees[SIGNAL] == 0 && frees[GEOMETRY] == 0, "historical release observation");
        } else {
            need(!acq.signals && !acq.srcpos_loc, "handles cleared at shot end");
            need(frees[SIGNAL] == ishot && frees[GEOMETRY] == geometry_expected,
                 "balanced ownership at every shot boundary");
        }
    }
#include "after_loop.inc"
    need(exports == (RUN_MULTIPLE_SHOTS ? geometry_expected : 0), "source export count");
    need(frees[SIGNAL] == (HISTORICAL ? 0 : allocs[SIGNAL]) &&
         frees[GEOMETRY] == (HISTORICAL ? 0 : allocs[GEOMETRY]), "final ownership");
    printf("signals=%d/%d geometry=%d/%d shots=%d\n",
           allocs[SIGNAL], frees[SIGNAL], allocs[GEOMETRY], frees[GEOMETRY], shots);
}

static void clear_evidence_handles(int reap)
{
    for (size_t k = 0; k < registered; ++k) {
        struct allocation *entry = &owned[k];
        if (reap && entry->pointer)
            __real_free_matrix(entry->pointer, 1, entry->row_high, 1, entry->column_high);
        entry->pointer = NULL;
    }
    acq.signals = NULL;
    acq.srcpos_loc = NULL;
}

int main(int argc, char **argv)
{
    static const int single[] = {1}, simultaneous[] = {3}, empty[] = {0};
    static const int two_local_shots[] = {1, 1};
    static const int repeat[] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    const int *counts;
    int shots = 1, global = 1, leak_probe;
    FILE *output = NULL;
    FP = stderr;
    need(argc == 3, "arguments");
    leak_probe = strcmp(argv[1], "leak-probe") == 0;
    RUN_MULTIPLE_SHOTS = 1;
    if (leak_probe) {
        need(HISTORICAL, "isolated historical leak probe");
        counts = two_local_shots;
        shots = global = 2;
    } else if (strcmp(argv[1], "single") == 0) {
        counts = single;
    } else if (strcmp(argv[1], "simultaneous") == 0) {
        counts = simultaneous;
        global = 3;
        RUN_MULTIPLE_SHOTS = 0;
    } else if (strcmp(argv[1], "empty") == 0) {
        counts = empty;
    } else {
        need(strcmp(argv[1], "repeat") == 0, "test case");
        counts = repeat;
        shots = global = MAX_SHOTS;
    }
    acq.srcpos = matrix(1, 8, 1, global);
    acq.srcpos1 = fmatrix(1, 8, 1, 1);
    for (int i = 1; i <= global; ++i) {
        int here = RUN_MULTIPLE_SHOTS ? counts[i-1] : 1;
        acq.srcpos[1][i] = here ? 6 + (RUN_MULTIPLE_SHOTS ? 0 : i-1) : GRID_X + 3;
        acq.srcpos[2][i] = 5;
        acq.srcpos[5][i] = 4;
        acq.srcpos[6][i] = 1;
        acq.srcpos[8][i] = 1;
    }
    if (strcmp(argv[2], "-") != 0) {
        output = fopen(argv[2], "wb");
        need(output != NULL, "result open");
    }
    run_shots(counts, shots, global, output);
    if (output)
        need(fclose(output) == 0, "result close");
    free_matrix(acq.srcpos1, 1, 8, 1, 1);
    free_matrix(acq.srcpos, 1, 8, 1, global);
    acq.srcpos1 = acq.srcpos = NULL;
    if (leak_probe) {
        /* Release only the second shot, still reachable via acq. */
        __real_free_matrix(acq.signals, 1, 1, 1, NT);
        __real_free_matrix(acq.srcpos_loc, 1, 8, 1, 1);
    }
    clear_evidence_handles(HISTORICAL && !leak_probe);
    if (leak_probe) {
        need(__lsan_do_recoverable_leak_check() == 1, "expected historical LSan diagnosis");
        fflush(stdout);
        _Exit(42);
    }
    return 0;
}

