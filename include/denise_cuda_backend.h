#ifndef DENISE_CUDA_BACKEND_H
#define DENISE_CUDA_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DENISE_CUDA_DEVICE_NAME_CAPACITY 256

/*
 * Flat device representation for a DENISE matrix with logical bounds.
 * device_base points at logical element [j0][i0]. CPU pointer tables and
 * Numerical-Recipes pointer shifts never cross this ABI.
 *
 * Logical element [j][i] is stored at:
 *   device_base[(j - j0) * pitch_elements + (i - i0)]
 */
struct denise_cuda_view2d_f32 {
    float *device_base;
    size_t pitch_elements;
    int i0;
    int j0;
    size_t nx_alloc;
    size_t ny_alloc;
};

/*
 * Frozen GSLS device layout. Current CPU f3tensor storage is [j][i][l],
 * with l contiguous. For M8e-1 L=1, l0=1 and nl_alloc=1, so
 * host[j][i][1] maps to:
 *   device_base[(j - j0) * pitch_elements + (i - i0)]
 *
 * Generic L uses device SoA [l][j][i]:
 *   device_base[(l - l0) * plane_stride_elements
 *             + (j - j0) * pitch_elements + (i - i0)]
 * A later generic-L adapter will pack/unpack the existing CPU layout; the
 * CPU allocation is not changed by this contract.
 */
struct denise_cuda_view3d_soa_f32 {
    float *device_base;
    size_t pitch_elements;
    size_t plane_stride_elements;
    int i0;
    int j0;
    int l0;
    size_t nx_alloc;
    size_t ny_alloc;
    size_t nl_alloc;
};

struct denise_cuda_device_info {
    int visible_device_count;
    int selected_logical_device;
    char name[DENISE_CUDA_DEVICE_NAME_CAPACITY];
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_bytes;
    size_t free_bytes;
    size_t safety_reserve_bytes;
    size_t user_cap_bytes;
    size_t usable_budget_bytes;
};

/* Device indices are CUDA logical indices after CUDA_VISIBLE_DEVICES masking.
 * A zero user_cap_bytes value means no user cap. Selection succeeds only with
 * a positive usable budget; later backend allocations are bounded by it.
 * All functions return 0 on success and -1 on a diagnosed failure. */
const char *denise_cuda_last_error(void);
int denise_cuda_visible_device_count(int *count);
int denise_cuda_select_device(int logical_device,
                              size_t safety_reserve_bytes,
                              size_t user_cap_bytes,
                              struct denise_cuda_device_info *info);

int denise_cuda_allocate_view2d(struct denise_cuda_view2d_f32 *view,
                                int j0, int j1, int i0, int i1);
int denise_cuda_release_view2d(struct denise_cuda_view2d_f32 *view);

int denise_cuda_copy_matrix_to_device(
        struct denise_cuda_view2d_f32 *view,
        float *const *host_matrix);
int denise_cuda_copy_matrix_to_host(
        float **host_matrix,
        const struct denise_cuda_view2d_f32 *view);

/*
 * Foundation-only deterministic oracle. For every requested logical cell:
 *   value += 3*i + 5*j + 7
 * It is deliberately not a scientific P/SV kernel.
 */
int denise_cuda_apply_index_affine(
        struct denise_cuda_view2d_f32 *view,
        int j_lo, int j_hi, int i_lo, int i_hi);

#ifdef __cplusplus
}
#endif

#endif
