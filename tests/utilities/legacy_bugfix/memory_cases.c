#include "fd.h"
#include "fwi_misfit_history.h"
#include <fenv.h>
#include <limits.h>

extern int memory_sh_case(int empty, const char *output);
static int assertions;

#define CHECK(c) do { \
    ++assertions; \
    if (!(c)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); exit(43); } \
} while (0)

/* Keep util.c's real diagnostics. Replace MPI's process termination only. */
int MPI_Abort(MPI_Comm comm, int status)
{
    (void)comm;
    exit(status == 1 ? 42 : 44);
}

static void stop_case(const struct fwi_misfit_history *history, int iter,
                      int minimum, int failed, float threshold,
                      int expected_stop, float expected_change)
{
    float change = -7.0f;
    feclearexcept(FE_ALL_EXCEPT);
    CHECK(fwi_misfit_should_stop(history, iter, minimum, failed,
                                 threshold, &change) == expected_stop);
    CHECK(memcmp(&change, &expected_change, sizeof(change)) == 0);
    CHECK((fetestexcept(FE_DIVBYZERO | FE_INVALID)) == 0);
}

static void test_history(void)
{
    struct fwi_misfit_history history = {0};
    int minimum, iter;
    stop_case(&history, 3, 0, 0, 0.25f, 0, -7.0f);
    for (minimum = 0; minimum <= 2; ++minimum) {
        history = (struct fwi_misfit_history){0};
        for (iter = 1; iter <= 3; ++iter) {
            fwi_misfit_record(&history, iter, iter == 1 ? 8 : 6);
            stop_case(&history, iter, minimum, 0, 0.25f, iter == 3,
                      iter == 3 ? 0.25f : -7.0f);
            stop_case(&history, iter, minimum, 1, 0.25f, iter > minimum, -7.0f);
        }
    }
    stop_case(&history, 3, 3, 0, 0.25f, 0, -7.0f);
    stop_case(&history, 3, 0, 0, 0.125f, 0, 0.25f);

    history = (struct fwi_misfit_history){0};
    fwi_misfit_record(&history, 3, 6);
    stop_case(&history, 3, 0, 0, 0.25f, 0, -7.0f);
    fwi_misfit_record(&history, 1, -0.0);
    stop_case(&history, 3, 0, 0, 0.25f, 0, -7.0f);
    stop_case(&history, 3, 0, 1, 0.25f, 1, -7.0f);
    fwi_misfit_record(&history, 1, INFINITY);
    stop_case(&history, 3, 0, 0, 0.25f, 0, -7.0f);
    fwi_misfit_record(&history, 1, NAN);
    stop_case(&history, 3, 0, 0, 0.25f, 0, -7.0f);
    fwi_misfit_record(&history, 1, 8);
    fwi_misfit_record(&history, 3, NAN);
    stop_case(&history, 3, 0, 0, 0.25f, 0, -7.0f);
    fwi_misfit_record(&history, 3, INFINITY);
    stop_case(&history, 3, 0, 0, 0.25f, 0, -7.0f);
    fwi_misfit_record(&history, 3, 0);
    stop_case(&history, 3, 0, 0, 1.0f, 1, 1.0f);
    fwi_misfit_record(&history, 1, -8);
    fwi_misfit_record(&history, 3, -6);
    stop_case(&history, 3, 0, 0, 0.25f, 1, 0.25f);
    fwi_misfit_record(&history, 1, 3);
    fwi_misfit_record(&history, 3, 2);
    stop_case(&history, 3, 0, 0, 0x1.555556p-2f, 1, 0x1.555556p-2f);

    fwi_misfit_record(&history, 998, 16);
    fwi_misfit_record(&history, 999, 20);
    fwi_misfit_record(&history, 1000, 12);
    stop_case(&history, 1000, 0, 0, 0.25f, 1, 0.25f);
    fwi_misfit_record(&history, 1001, 15);
    stop_case(&history, 1001, 0, 0, 0.25f, 1, 0.25f);
    fwi_misfit_record(&history, INT_MAX - 2, 32);
    fwi_misfit_record(&history, INT_MAX, 24);
    stop_case(&history, INT_MAX, 0, 0, 0.25f, 1, 0.25f);
}

#define MATRIX_CASE(type, allocate, release) do { \
    type **m = allocate(5, 6, -9, -7); \
    for (int r = 5; r <= 6; ++r) \
        for (int c = -9; c <= -7; ++c) { \
            int offset = 3 * (r - 5) + c + 9; \
            CHECK(&m[r][c] == &m[5][-9] + offset); \
            CHECK(m[r][c] == 0); \
            m[r][c] = (type)(offset + 1); \
        } \
    CHECK(m[5][-9] == 1 && m[6][-7] == 6); \
    release(m, 5, 6, -9, -7); \
    m = allocate(1, 0, 1, 3); \
    CHECK(&m[1][1] != NULL); \
    release(m, 1, 0, 1, 3); \
    m = allocate(2, 3, 7, 6); \
    CHECK(&m[2][7] == &m[3][7]); \
    release(m, 2, 3, 7, 6); \
    m = allocate(-9, -10, 3, 2); \
    CHECK(&m[-9][3] != NULL); \
    release(m, -9, -10, 3, 2); \
    release(NULL, 1, 0, 1, 0); \
} while (0)

#define TENSOR3_CASE(type, allocate, release) do { \
    type ***t = allocate(-5, -4, 3, 4, -7, -5); \
    for (int r = -5; r <= -4; ++r) \
        for (int c = 3; c <= 4; ++c) \
            for (int d = -7; d <= -5; ++d) { \
                int offset = ((r + 5) * 2 + c - 3) * 3 + d + 7; \
                CHECK(&t[r][c][d] == &t[-5][3][-7] + offset); \
                CHECK(t[r][c][d] == 0); \
                t[r][c][d] = (type)(offset + 1); \
            } \
    CHECK(t[-5][3][-7] == 1 && t[-4][4][-5] == 12); \
    release(t, -5, -4, 3, 4, -7, -5); \
    t = allocate(1, 0, 1, 3, 1, 4); \
    CHECK(&t[1][1][1] != NULL); \
    release(t, 1, 0, 1, 3, 1, 4); \
    t = allocate(1, 2, 1, 0, 1, 4); \
    CHECK(&t[1][1][1] == &t[2][1][1]); \
    release(t, 1, 2, 1, 0, 1, 4); \
    t = allocate(1, 2, 1, 2, 1, 0); \
    CHECK(&t[1][1][1] == &t[2][2][1]); \
    release(t, 1, 2, 1, 2, 1, 0); \
    t = allocate(1, 0, 1, 0, 1, 0); \
    CHECK(&t[1][1][1] != NULL); \
    release(t, 1, 0, 1, 0, 1, 0); \
    release(NULL, 1, 0, 1, 0, 1, 0); \
} while (0)

static void test_tensor4(void)
{
    float ****t = f4tensor(-3, -2, 2, 3, -4, -3, 4, 6);
    for (int r = -3; r <= -2; ++r)
        for (int c = 2; c <= 3; ++c)
            for (int d = -4; d <= -3; ++d)
                for (int v = 4; v <= 6; ++v) {
                    int offset = (((r + 3) * 2 + c - 2) * 2 + d + 4) * 3 + v - 4;
                    CHECK(&t[r][c][d][v] == &t[-3][2][-4][4] + offset);
                    CHECK(t[r][c][d][v] == 0);
                    t[r][c][d][v] = (float)(offset + 1);
                }
    CHECK(t[-3][2][-4][4] == 1 && t[-2][3][-3][6] == 24);
    free_f4tensor(t, -3, -2, 2, 3, -4, -3, 4, 6);

    for (int axis = 0; axis < 5; ++axis) {
        int last[4] = {2, 2, 2, 2};
        for (int a = 0; a < 4; ++a)
            if (axis == a || axis == 4)
                last[a] = 0;
        t = f4tensor(1, last[0], 1, last[1], 1, last[2], 1, last[3]);
        CHECK(&t[1][1][1][1] != NULL);
        if (axis == 3)
            CHECK(&t[1][1][1][1] == &t[2][2][2][1]);
        free_f4tensor(t, 1, last[0], 1, last[1], 1, last[2], 1, last[3]);
    }
    free_f4tensor(NULL, 1, 0, 1, 0, 1, 0, 1, 0);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    if (strcmp(argv[1], "history") == 0) {
        test_history();
    } else if (strcmp(argv[1], "alloc") == 0) {
        MATRIX_CASE(float, matrix, free_matrix);
        MATRIX_CASE(float, fmatrix, free_matrix);
        MATRIX_CASE(int, imatrix, free_imatrix);
        MATRIX_CASE(double, dmatrix, free_dmatrix);
        MATRIX_CASE(unsigned short int, usmatrix, free_usmatrix);
        TENSOR3_CASE(float, f3tensor, free_f3tensor);
        TENSOR3_CASE(int, i3tensor, free_i3tensor);
        test_tensor4();
    } else if (strcmp(argv[1], "bad-row") == 0) {
        matrix(1, -1, 1, 3);
        return 45;
    } else if (strcmp(argv[1], "bad-column") == 0) {
        dmatrix(1, 2, 0, -2);
        return 45;
    } else if (strcmp(argv[1], "bad-depth") == 0) {
        i3tensor(1, 2, 1, 2, 4, 2);
        return 45;
    } else if (strcmp(argv[1], "bad-values") == 0) {
        f4tensor(1, 1, 1, 1, 1, 1, 2, 0);
        return 45;
    } else if (strcmp(argv[1], "bad-product") == 0) {
        f3tensor(1, INT_MAX, 1, INT_MAX, 1, INT_MAX);
        return 45;
    } else if (strcmp(argv[1], "bad-bytes") == 0) {
        dmatrix(1, INT_MAX, 1, INT_MAX);
        return 45;
    } else if (strcmp(argv[1], "empty-sh") == 0) {
        return memory_sh_case(1, NULL);
    } else if (strcmp(argv[1], "positive-sh") == 0) {
        return memory_sh_case(0, "sh-sanitized.bin");
    } else {
        return 46;
    }
    printf("PASS %s (%d assertions)\n", argv[1], assertions);
    return 0;
}

