#include "fd.h"
#include "segy.h"

int MYID_SHOT;
char DATA_DIR[STRING_SIZE] = "fixture";
char DATA_DIR_T0[STRING_SIZE], SEIS_FILE_P[STRING_SIZE];
char SEIS_FILE_VX[STRING_SIZE], SEIS_FILE_VY[STRING_SIZE];

static float *rows[3];
static const float guard = -1234567.0f;
static const float two_traces[2][4] = {
    {0.5f, -2.0f, 3.0f, -4.25f}, {-9.5f, 0.25f, -1.5f, 6.75f}
};
static int shot, ns, ntr, allocated_ns, padded;
static const char *expected_error;
static int completed_traces;

static void require(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL shot %d: %s\n", shot, message);
        exit(43);
    }
}

static float expected_sample(int r, int j)
{
    if (shot == 2)
        return 42.125f;
    if (shot == 3) {
        if (j == 1)
            return 123.5f;
        if (j == 60768)
            return -45.25f;
        return ((j - 1) % 31 - 15) * 0.125f;
    }
    return two_traces[r - 1][j - 1];
}

static void check_rows(void)
{
    int r, j;
    for (r = 1; r <= ntr; ++r) {
        require(rows[r][0] == guard, "lower destination canary");
        if (padded)
            require(rows[r][allocated_ns + 1] == guard, "upper destination canary");
        for (j = 1; j <= allocated_ns; ++j) {
            const float expected = r <= completed_traces
                ? expected_sample(r, j) : guard;
            require(memcmp(&rows[r][j], &expected, sizeof(float)) == 0,
                    "sample bits or untouched destination row");
        }
    }
}

static void free_rows(void)
{
    int r;
    for (r = 1; r <= ntr; ++r)
        free(rows[r]);
}

void err(char message[])
{
    require(expected_error && strcmp(message, expected_error) == 0, message);
    check_rows();
    free_rows();
    printf("EXPECTED SU ERROR %d\n", shot);
    exit(42);
}

int main(int argc, char **argv)
{
    const unsigned short endian = 1;
    static const struct {
        int ntr, ns, complete;
        const char *error;
    } cases[] = {
        {2, 4, 2, NULL},
        {1, 1, 1, NULL},
        {1, 60768, 1, NULL},
        {2, 4, 1, "Incomplete SU trace header"},
        {2, 4, 1, "Incomplete SU trace samples"},
        {2, 4, 1, "SU trace sample count does not match requested count"},
        {1, 0, 0, "Invalid SU sample count"},
        {1, 60769, 0, "Invalid SU sample count"},
        {1, 4, 0, " Field data file could not be opened !"}
    };
    int r, j;
    require(argc == 3, "arguments MODE SHOT");
    require(strcmp(argv[1], "canary") == 0 || strcmp(argv[1], "asan") == 0, "mode");
    padded = strcmp(argv[1], "canary") == 0;
    shot = atoi(argv[2]);
    require(shot > 0 && (size_t)shot <= sizeof(cases) / sizeof(cases[0]), "shot");
    require(*(const unsigned char *)&endian == 1 && sizeof(float) == 4 &&
            offsetof(segy, ns) == 114 && offsetof(segy, data) == 240 &&
            SU_NFLTS == 60768, "native SU layout");
    ntr = cases[shot - 1].ntr;
    ns = cases[shot - 1].ns;
    completed_traces = cases[shot - 1].complete;
    expected_error = cases[shot - 1].error;
    allocated_ns = ns >= 1 && ns <= SU_NFLTS ? ns : 4;
    for (r = 1; r <= ntr; ++r) {
        rows[r] = malloc((size_t)(allocated_ns + 1 + padded) * sizeof(float));
        require(rows[r] != NULL, "row allocation");
        for (j = 0; j <= allocated_ns + padded; ++j)
            rows[r][j] = guard;
    }
    inseis(shot, rows, ntr, ns, 2, 0);
    require(!expected_error, "expected rejection");
    check_rows();
    free_rows();
    printf("SU OK %d\n", shot);
    return 0;
}

