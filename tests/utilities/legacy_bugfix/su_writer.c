#define _GNU_SOURCE
#include "fd.h"
#include "segy.h"
#include <errno.h>
#include <float.h>
#include <stdint.h>

int MYID_SHOT, NDT;
float TIME, DT, DH = 1.0f, REFREC[4];
char DATA_DIR[STRING_SIZE], DATA_DIR_T0[STRING_SIZE], SEIS_FILE_P[STRING_SIZE];
char SEIS_FILE_VX[STRING_SIZE], SEIS_FILE_VY[STRING_SIZE];

enum producer { LOCAL, GLOBAL, VECTOR, SOURCE };
struct example {
    const char *name;
    enum producer producer;
    int ntr, available, stride, steps, legacy;
};
static const struct example examples[] = {
#include "cases.inc"
};

struct failing_stream {
    size_t capacity, accepted, expected_accepted;
    int closed;
};
static struct failing_stream fault;
static const char *expected_error;
static int expect_cookie;

static void require(int okay, const char *message)
{
    if (!okay) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(43);
    }
}

void err(char message[])
{
    require(expected_error && strcmp(expected_error, message) == 0, message);
    if (expect_cookie)
        require(fault.closed && fault.accepted == fault.expected_accepted,
                "failing stream close and byte count");
    printf("EXPECTED: %s\n", message);
    exit(42);
}

static ssize_t fault_write(void *cookie, const char *bytes, size_t count)
{
    struct failing_stream *state = cookie;
    size_t remaining = state->capacity - state->accepted;
    (void)bytes;
    if (remaining == 0) {
        errno = ENOSPC;
        return -1;
    }
    if (count > remaining) {
        errno = ENOSPC;
        count = remaining;
    }
    state->accepted += count;
    return (ssize_t)count;
}

static int fault_close(void *cookie)
{
    ((struct failing_stream *)cookie)->closed = 1;
    return 0;
}

static void invoke(enum producer producer, FILE *out, float **values, int **recpos,
                   int **recpos_loc, float **srcpos, int ntr, int ns, int sws)
{
    switch (producer) {
    case LOCAL:
        outseis(stderr, out, 7, values, recpos, recpos_loc, ntr, srcpos, 1,
                ns, 1, 1, sws);
        break;
    case GLOBAL:
        outseis_glob(stderr, out, 7, values, recpos, recpos_loc, ntr, srcpos, 1,
                     ns, 1, 1, sws);
        break;
    case VECTOR:
        outseis_vector(stderr, out, 7, values ? values[1] : NULL, recpos,
                       recpos_loc, ntr, srcpos, 1, ns, 1, 1, sws);
        break;
    case SOURCE:
        output_source_signal(out, values, ns, 1);
        break;
    }
}

static const float end_guard = -614.75f;
static float *test_row(int count, int padded)
{
    float *row = malloc((size_t)(count + 1 + padded) * sizeof(float));
    require(row != NULL, "row allocation");
    row[0] = end_guard;
    if (padded)
        row[count+1] = end_guard;
    return row;
}

static void check_guards(const float *row, int count, int padded)
{
    require(row[0] == end_guard, "leading canary");
    if (padded)
        require(row[count+1] == end_guard, "trailing canary");
}

static void run_example(const struct example *example, const char *prefix,
                        int historical, int padded)
{
    static float sx[] = {0, 800}, sy[] = {0, 800};
    float *srcpos[] = {NULL, sx, sy};
    int r, j;
    int *rx = calloc((size_t)example->ntr + 1, sizeof(int));
    int *ry = calloc((size_t)example->ntr + 1, sizeof(int));
    int *rid = calloc((size_t)example->ntr + 1, sizeof(int));
    int *local_ids = calloc((size_t)example->ntr + 1, sizeof(int));
    int *recpos[] = {NULL, rx, ry, rid};
    int *local[] = {NULL, NULL, NULL, local_ids};
    float **values = calloc((size_t)example->ntr + 1, sizeof(float *));
    float **got = calloc((size_t)example->ntr + 1, sizeof(float *));
    float *input_ref = malloc((size_t)example->available * sizeof(float));
    int count = (example->steps ? example->steps : example->available) / example->stride;
    FILE *input, *out;
    char srcfile[100], output[100];
    require(rx && ry && rid && local_ids && values && got && input_ref, "storage");
    snprintf(srcfile, sizeof(srcfile), "fixtures/%s.input", example->name);
    snprintf(DATA_DIR, sizeof(DATA_DIR), "%s-%s", prefix, example->name);
    snprintf(output, sizeof(output), "%s_y.su.shot1", DATA_DIR);
    input = fopen(srcfile, "rb");
    require(input != NULL, "open independent input");
    for (r = 1; r <= example->ntr; ++r) {
        values[r] = test_row(example->available, padded);
        require(fread(&values[r][1], sizeof(float), (size_t)example->available, input)
                == (size_t)example->available, "read independent input");
        rx[r] = 800 + 4*(r-1);
        ry[r] = 800 + 3*(r-1);
        rid[r] = 20 + r;
        local_ids[r] = example->ntr - r + 1;
    }
    require(fgetc(input) == EOF && !ferror(input), "input record end");

    NDT = strcmp(example->name, "local-two") == 0 ? 17 : example->stride;
    DT = 0.001f;
    TIME = (float)(example->steps ? example->steps : example->available) * DT;
    out = fopen(output, "wb");
    require(out != NULL, "open test output");
    invoke(example->producer, out, values, recpos, local, srcpos,
           example->ntr, example->available, example->steps != 0);

    if (!historical) {
        for (r = 1; r <= example->ntr; ++r)
            got[r] = test_row(count, padded);
        inseis(1, got, example->ntr, count, 2, 0);
    }
    rewind(input);
    for (r = 1; r <= example->ntr; ++r) {
        require(fread(input_ref, sizeof(float), (size_t)example->available, input)
                == (size_t)example->available, "reread independent input");
        check_guards(values[r], example->available, padded);
        require(memcmp(&values[r][1], input_ref, (size_t)example->available * sizeof(float))
                == 0, "writer changed source words");
        if (!historical) {
            check_guards(got[r], count, padded);
            for (j = 1; j <= count; ++j)
                require(memcmp(&got[r][j], &input_ref[j*example->stride-1], sizeof(float))
                        == 0, "roundtrip sample differs");
            free(got[r]);
        }
        free(values[r]);
    }
    require(fclose(input) == 0, "close independent input");
    free(input_ref); free(got); free(values);
    free(local_ids); free(rid); free(ry); free(rx);
    printf("PASS %s\n", example->name);
}

struct negative {
    const char *name, *diagnostic;
    enum producer producer;
    int available, stride, mode;
    float dt, time;
    int cookie;
    size_t writable;
    int buffered;
};
static const struct negative negatives[] = {
    {"zero", "Invalid SU output sample count or stride", LOCAL, 0, 1, 0, .001f, 0, 1, 0, 0},
    {"over-local", "SU output sample count outside supported range", LOCAL, 60769, 1, 0, .001f, 0, 1, 0, 0},
    {"over-global", "SU output sample count outside supported range", GLOBAL, 60769, 1, 1, .001f, 60.769f, 1, 0, 0},
    {"over-vector", "SU output sample count outside supported range", VECTOR, 60769, 1, 0, .001f, 0, 1, 0, 0},
    {"over-source", "SU output sample count outside supported range", SOURCE, 60769, 1, 0, .001f, 0, 1, 0, 0},
    {"stride-zero", "Invalid SU output sample count or stride", SOURCE, 6, 0, 0, .001f, 0, 1, 0, 0},
    {"mode", "Invalid SU output sampling mode", VECTOR, 4, 1, 2, .001f, 0, 1, 0, 0},
    {"short-input", "SU output sample count outside supported range", GLOBAL, 5, 3, 1, .001f, .007f, 1, 0, 0},
    {"time-overflow", "Invalid SU output time extent", LOCAL, 6, 1, 1, .001f, FLT_MAX, 1, 0, 0},
    {"interval-overflow", "SU output interval does not fit the header", SOURCE, 4, 1, 0, .1f, 0, 1, 0, 0},
    {"open", "Could not open SU output", LOCAL, 2, 1, 0, .001f, 0, 0, 0, 0},
    {"header", "Incomplete SU output header", LOCAL, 2, 1, 0, .001f, 0, 1, 239, 0},
    {"payload", "Incomplete SU output samples", VECTOR, 2, 1, 0, .001f, 0, 1, 247, 0},
    {"close", "Could not close SU output", SOURCE, 2, 1, 0, .001f, 0, 1, 0, 1},
};

static void run_negative(const char *name)
{
    static char buffer[4096];
    float wave[] = {0, 3, 4}, sx[] = {0, 800}, sy[] = {0, 800};
    float *values[] = {NULL, wave}, *srcpos[] = {NULL, sx, sy};
    int coord[] = {0, 800}, row_id[] = {0, 1};
    int *recpos[] = {NULL, coord, coord, row_id};
    const cookie_io_functions_t io = {.write = fault_write, .close = fault_close};
    for (size_t n = 0; n < sizeof(negatives) / sizeof(negatives[0]); ++n) {
        const struct negative *test = &negatives[n];
        FILE *stream = NULL;
        int data_needed;
        if (strcmp(name, test->name) != 0)
            continue;
        expected_error = test->diagnostic;
        expect_cookie = test->cookie;
        data_needed = test->writable || test->buffered;
        if (expect_cookie) {
            fault.capacity = fault.expected_accepted = test->writable;
            stream = fopencookie(&fault, "wb", io);
            require(stream != NULL, "fopencookie");
            require(setvbuf(stream, test->buffered ? buffer : NULL,
                            test->buffered ? _IOFBF : _IONBF,
                            test->buffered ? sizeof(buffer) : 0) == 0, "setvbuf");
        }
        TIME = test->time;
        DT = test->dt;
        NDT = test->stride;
        invoke(test->producer, stream, data_needed ? values : NULL,
               data_needed ? recpos : NULL, data_needed ? recpos : NULL,
               data_needed ? srcpos : NULL, 1, test->available, test->mode);
        require(0, "unexpected success");
    }
    require(0, "unknown negative case");
}

int main(int argc, char **argv)
{
    const unsigned short little = 1;
    int historical, padded;
    require(*(const unsigned char *)&little == 1 && sizeof(float) == 4 &&
            offsetof(segy, ns) == 114 && offsetof(segy, dt) == 116 &&
            offsetof(segy, d1) == 180 && offsetof(segy, data) == 240,
            "little-endian SU ABI");
    require(argc == 3, "arguments");
    if (strcmp(argv[1], "negative") == 0)
        run_negative(argv[2]);
    historical = strcmp(argv[1], "historical") == 0;
    padded = strcmp(argv[1], "asan") != 0;
    require(historical || !padded || strcmp(argv[1], "normal") == 0, "mode");
    for (size_t n = 0; n < sizeof(examples) / sizeof(examples[0]); ++n)
        if (!historical || examples[n].legacy)
            run_example(&examples[n], argv[2], historical, padded);
    return 0;
}

