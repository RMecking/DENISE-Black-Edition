#include "fd.h"
#include "denise_cuda_backend.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>

#define MIB ((size_t)1024 * (size_t)1024)

/* util.c reports allocation failures through this canonical rank identifier. */
int MYID = 0;

static int fail(const char *message) {
    fprintf(stderr, "CUDA foundation harness failure: %s\n", message);
    return 1;
}

static void fill_transfer_pattern(float **values,
                                  int j0, int j1, int i0, int i1) {
    static const float pattern[] = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.25f, 3.125f, -17.75f,
        2048.0f, -4096.5f
    };
    const size_t pattern_count = sizeof(pattern) / sizeof(pattern[0]);
    size_t position = 0;
    int j, i;
    for (j = j0; j <= j1; ++j)
        for (i = i0; i <= i1; ++i) {
            values[j][i] = pattern[position % pattern_count];
            ++position;
        }
}

static void fill_integer_matrix(float **values,
                                int j0, int j1, int i0, int i1,
                                int offset) {
    int j, i;
    for (j = j0; j <= j1; ++j)
        for (i = i0; i <= i1; ++i)
            values[j][i] = (float)(
                offset + 101 * (j - j0) + 3 * (i - i0));
}

static void copy_matrix_cpu(float **destination, float **source,
                            int j0, int j1, int i0, int i1) {
    int j, i;
    for (j = j0; j <= j1; ++j)
        for (i = i0; i <= i1; ++i)
            destination[j][i] = source[j][i];
}

static int payload_equal(float **left, float **right,
                         int j0, int j1, int i0, int i1) {
    const size_t rows = (size_t)(j1 - j0 + 1);
    const size_t columns = (size_t)(i1 - i0 + 1);
    return memcmp(&left[j0][i0], &right[j0][i0],
                  rows * columns * sizeof(float)) == 0;
}

static void apply_cpu_oracle(float **values,
                             int j_lo, int j_hi,
                             int i_lo, int i_hi) {
    int j, i;
    for (j = j_lo; j <= j_hi; ++j)
        for (i = i_lo; i <= i_hi; ++i)
            values[j][i] += (float)(3 * i + 5 * j + 7);
}

static int outside_rectangle_unchanged(
        float **result, float **source,
        int j0, int j1, int i0, int i1,
        int j_lo, int j_hi, int i_lo, int i_hi) {
    int j, i;
    for (j = j0; j <= j1; ++j)
        for (i = i0; i <= i1; ++i)
            if ((j < j_lo || j > j_hi || i < i_lo || i > i_hi) &&
                memcmp(&result[j][i], &source[j][i], sizeof(float)) != 0)
                return 0;
    return 1;
}

static int run_matrix_case(const char *name,
                           int j0, int j1, int i0, int i1,
                           int *roundtrip_ok,
                           int *sentinel_ok,
                           int *oracle_ok,
                           int *repeat_ok) {
    float **source = NULL;
    float **roundtrip = NULL;
    float **expected = NULL;
    float **result = NULL;
    float **repeated = NULL;
    struct denise_cuda_view2d_f32 view;
    int status = 1;
    const int j_lo = j0 + 1;
    const int j_hi = j1 - 1;
    const int i_lo = i0 + 1;
    const int i_hi = i1 - 1;

    memset(&view, 0, sizeof(view));
    if (j_hi < j_lo || i_hi < i_lo)
        return fail("test rectangle is empty");

    source = matrix(j0, j1, i0, i1);
    roundtrip = matrix(j0, j1, i0, i1);
    expected = matrix(j0, j1, i0, i1);
    result = matrix(j0, j1, i0, i1);
    repeated = matrix(j0, j1, i0, i1);

    fill_transfer_pattern(source, j0, j1, i0, i1);
    fill_integer_matrix(roundtrip, j0, j1, i0, i1, -7000);
    copy_matrix_cpu(expected, source, j0, j1, i0, i1);
    apply_cpu_oracle(expected, j_lo, j_hi, i_lo, i_hi);

    if (denise_cuda_allocate_view2d(&view, j0, j1, i0, i1) != 0)
        goto cleanup;
    if (denise_cuda_copy_matrix_to_device(&view, source) != 0)
        goto cleanup;
    if (denise_cuda_copy_matrix_to_host(roundtrip, &view) != 0)
        goto cleanup;
    if (!payload_equal(source, roundtrip, j0, j1, i0, i1))
        goto cleanup;
    *roundtrip_ok = 1;

    if (denise_cuda_apply_index_affine(
            &view, j_lo, j_hi, i_lo, i_hi) != 0)
        goto cleanup;
    if (denise_cuda_copy_matrix_to_host(result, &view) != 0)
        goto cleanup;
    if (!outside_rectangle_unchanged(
            result, source, j0, j1, i0, i1,
            j_lo, j_hi, i_lo, i_hi))
        goto cleanup;
    *sentinel_ok = 1;
    if (!payload_equal(expected, result, j0, j1, i0, i1))
        goto cleanup;
    *oracle_ok = 1;

    if (denise_cuda_copy_matrix_to_device(&view, source) != 0)
        goto cleanup;
    if (denise_cuda_apply_index_affine(
            &view, j_lo, j_hi, i_lo, i_hi) != 0)
        goto cleanup;
    if (denise_cuda_copy_matrix_to_host(repeated, &view) != 0)
        goto cleanup;
    if (!payload_equal(result, repeated, j0, j1, i0, i1))
        goto cleanup;
    *repeat_ok = 1;

    printf("CUDA_MATRIX_CASE name=%s j=%d:%d i=%d:%d pitch=%zu\n",
           name, j0, j1, i0, i1, view.pitch_elements);
    status = 0;

cleanup:
    if (view.device_base && denise_cuda_release_view2d(&view) != 0)
        status = 1;
    if (source) free_matrix(source, j0, j1, i0, i1);
    if (roundtrip) free_matrix(roundtrip, j0, j1, i0, i1);
    if (expected) free_matrix(expected, j0, j1, i0, i1);
    if (result) free_matrix(result, j0, j1, i0, i1);
    if (repeated) free_matrix(repeated, j0, j1, i0, i1);
    return status;
}

static int expect_budget_selection_failure(const char *name,
                                           size_t reserve_bytes,
                                           size_t cap_bytes) {
    struct denise_cuda_device_info info;
    if (denise_cuda_select_device(
            0, reserve_bytes, cap_bytes, &info) == 0) {
        fprintf(stderr,
                "CUDA foundation harness failure: budget case %s "
                "unexpectedly succeeded with budget=%zu\n",
                name, info.usable_budget_bytes);
        return 1;
    }
    if (!strstr(denise_cuda_last_error(), "effective available memory"))
        return fail("budget rejection did not explain effective memory/reserve");
    printf("CUDA_BUDGET_FAIL_CLOSED name=%s reserve=%zu cap=%zu\n",
           name, reserve_bytes, cap_bytes);
    return 0;
}

static int run_budget_tests(void) {
    struct denise_cuda_device_info info;
    struct denise_cuda_view2d_f32 view;
    const int over_budget_i1 = (int)(MIB / sizeof(float));

    memset(&view, 0, sizeof(view));
    if (denise_cuda_select_device(0, 1 * MIB, 2 * MIB, &info) != 0)
        return fail("valid controlled VRAM budget was rejected");
    if (info.usable_budget_bytes != 1 * MIB)
        return fail("valid controlled VRAM budget is inconsistent");
    if (denise_cuda_allocate_view2d(&view, 0, 15, 0, 15) != 0)
        return fail("valid in-budget allocation was rejected");
    if (denise_cuda_release_view2d(&view) != 0)
        return fail("valid in-budget allocation could not be released");

    if (denise_cuda_select_device(0, 1 * MIB, 2 * MIB, &info) != 0)
        return fail("controlled VRAM budget could not be restored");
    if (denise_cuda_allocate_view2d(
            &view, 0, 0, 0, over_budget_i1) == 0) {
        denise_cuda_release_view2d(&view);
        return fail("over-budget allocation unexpectedly succeeded");
    }
    if (view.device_base != NULL)
        return fail("over-budget allocation returned a device pointer");
    if (!strstr(denise_cuda_last_error(), "exceeds usable budget"))
        return fail("over-budget allocation diagnostic is unclear");

    if (expect_budget_selection_failure(
            "reserve-equals-effective", 64 * MIB, 64 * MIB) != 0)
        return 1;
    memset(&view, 0, sizeof(view));
    if (denise_cuda_allocate_view2d(&view, 0, 0, 0, 0) == 0) {
        denise_cuda_release_view2d(&view);
        return fail("allocation succeeded after zero-budget selection failed");
    }
    if (view.device_base != NULL)
        return fail("zero-budget allocation returned a device pointer");
    if (!strstr(denise_cuda_last_error(), "not initialized"))
        return fail("zero-budget allocation diagnostic is unclear");
    if (expect_budget_selection_failure(
            "reserve-greater-than-free", SIZE_MAX, 0) != 0)
        return 1;
    if (expect_budget_selection_failure(
            "cap-below-reserve", 64 * MIB, 63 * MIB) != 0)
        return 1;
    if (expect_budget_selection_failure(
            "cap-equals-reserve", 64 * MIB, 64 * MIB) != 0)
        return 1;

    printf("CUDA_BUDGET_GATES normal=1 equal=1 greater=1 "
           "cap_below=1 cap_equal=1 zero_alloc=1 "
           "over_alloc=1 in_budget=1\n");
    return 0;
}

static int parse_device(const char *text, int *device) {
    char *end = NULL;
    long value;
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno || !end || *end != '\0' ||
        value < INT_MIN || value > INT_MAX)
        return -1;
    *device = (int)value;
    return 0;
}

static int run_select_only(int logical_device) {
    struct denise_cuda_device_info info;
    if (denise_cuda_select_device(
            logical_device, 64 * MIB, 0, &info) != 0)
        return 2;
    printf("CUDA_SELECTED logical=%d name=%s cc=%d.%d\n",
           info.selected_logical_device, info.name,
           info.compute_capability_major,
           info.compute_capability_minor);
    return 0;
}

static int run_expected_no_device(void) {
    int count = -1;
    struct denise_cuda_device_info info;
    if (denise_cuda_visible_device_count(&count) != 0)
        return fail("device enumeration itself failed");
    if (count != 0)
        return fail("CUDA_VISIBLE_DEVICES did not hide every device");
    if (denise_cuda_select_device(0, 64 * MIB, 0, &info) == 0)
        return fail("device selection silently succeeded with no visible GPU");
    if (!denise_cuda_last_error()[0])
        return fail("no-device failure did not provide diagnostics");
    printf("CUDA_NO_DEVICE_FAIL_CLOSED visible=0\n");
    return 0;
}

int main(int argc, char **argv) {
    int count = 0;
    int device;
    int logical;
    int roundtrip_ok = 0;
    int sentinel_ok = 0;
    int oracle_ok = 0;
    int repeat_ok = 0;
    int halo_ok = 0;
    struct denise_cuda_device_info info;
    struct denise_cuda_device_info capped_info;

    MPI_Init(&argc, &argv);

    if (argc == 2 && strcmp(argv[1], "--expect-no-device") == 0) {
        const int status = run_expected_no_device();
        MPI_Finalize();
        return status;
    }
    if (argc == 3 && strcmp(argv[1], "--select-device") == 0) {
        if (parse_device(argv[2], &device) != 0) {
            MPI_Finalize();
            return fail("invalid --select-device argument");
        }
        logical = run_select_only(device);
        MPI_Finalize();
        return logical;
    }
    if (argc != 1) {
        MPI_Finalize();
        return fail("unexpected command line");
    }

    if (denise_cuda_visible_device_count(&count) != 0) {
        MPI_Finalize();
        return 1;
    }
    if (count == 0) {
        printf("CUDA_FOUNDATION_SKIP no_visible_device\n");
        MPI_Finalize();
        return 77;
    }

    for (logical = 0; logical < count; ++logical) {
        if (denise_cuda_select_device(logical, 64 * MIB, 0, &info) != 0) {
            MPI_Finalize();
            return 1;
        }
        printf("CUDA_DEVICE logical=%d visible=%d name=%s cc=%d.%d "
               "total=%zu free=%zu reserve=%zu cap=%zu budget=%zu\n",
               info.selected_logical_device, info.visible_device_count,
               info.name, info.compute_capability_major,
               info.compute_capability_minor, info.total_bytes,
               info.free_bytes, info.safety_reserve_bytes,
               info.user_cap_bytes, info.usable_budget_bytes);
        if (info.usable_budget_bytes !=
            (info.free_bytes > info.safety_reserve_bytes
                 ? info.free_bytes - info.safety_reserve_bytes
                 : 0)) {
            MPI_Finalize();
            return fail("VRAM budget calculation is inconsistent");
        }
    }

    if (denise_cuda_select_device(0, 64 * MIB, 0, &info) != 0) {
        MPI_Finalize();
        return 1;
    }
    if (denise_cuda_select_device(
            0, 64 * MIB, 256 * MIB, &capped_info) != 0) {
        MPI_Finalize();
        return 1;
    }
    {
        const size_t limited =
            capped_info.free_bytes < capped_info.user_cap_bytes
                ? capped_info.free_bytes
                : capped_info.user_cap_bytes;
        const size_t expected_budget =
            limited > capped_info.safety_reserve_bytes
                ? limited - capped_info.safety_reserve_bytes
                : 0;
        if (capped_info.usable_budget_bytes != expected_budget) {
            MPI_Finalize();
            return fail("capped VRAM budget calculation is inconsistent");
        }
        printf("CUDA_BUDGET_CAP cap=%zu reserve=%zu budget=%zu\n",
               capped_info.user_cap_bytes,
               capped_info.safety_reserve_bytes,
               capped_info.usable_budget_bytes);
    }

    if (run_budget_tests() != 0) {
        MPI_Finalize();
        return 1;
    }
    if (denise_cuda_select_device(0, 64 * MIB, 0, &info) != 0) {
        MPI_Finalize();
        return 1;
    }

    if (run_matrix_case("positive-origin", 3, 7, 4, 10,
                        &roundtrip_ok, &sentinel_ok,
                        &oracle_ok, &repeat_ok) != 0 ||
        run_matrix_case("one-based", 1, 5, 1, 8,
                        &roundtrip_ok, &sentinel_ok,
                        &oracle_ok, &repeat_ok) != 0 ||
        run_matrix_case("negative-halo", -2, 7, -2, 9,
                        &roundtrip_ok, &sentinel_ok,
                        &oracle_ok, &repeat_ok) != 0 ||
        run_matrix_case("rectangular", -1, 4, 2, 13,
                        &roundtrip_ok, &sentinel_ok,
                        &oracle_ok, &repeat_ok) != 0) {
        MPI_Finalize();
        return 1;
    }
    halo_ok = 1;

    printf("CUDA_TRANSFER_PATTERNS positive_integer=1 negative=1 zero=1 "
           "positive_fraction=1 negative_fraction=1\n");
    printf("CUDA_FOUNDATION_PASS cases=4 roundtrip=%d halo=%d "
           "sentinel=%d oracle=%d repeat=%d\n",
           roundtrip_ok, halo_ok, sentinel_ok, oracle_ok, repeat_ok);
    MPI_Finalize();
    return 0;
}
