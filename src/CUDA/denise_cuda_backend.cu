#include "denise_cuda_backend.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

thread_local char last_error_message[768] = "";
thread_local bool backend_initialized = false;
thread_local size_t backend_usable_budget_bytes = 0;

void clear_error(void) {
    last_error_message[0] = '\0';
}

int contract_failure(const char *operation, const char *file, int line,
                     const char *format, ...) {
    char detail[448];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(detail, sizeof(detail), format, arguments);
    va_end(arguments);
    std::snprintf(last_error_message, sizeof(last_error_message),
                  "CUDA backend contract failure during %s at %s:%d: %s",
                  operation, file, line, detail);
    std::fprintf(stderr, "%s\n", last_error_message);
    return -1;
}

int cuda_failure(const char *operation, cudaError_t status,
                 const char *file, int line) {
    const char *name = cudaGetErrorName(status);
    const char *description = cudaGetErrorString(status);
    std::snprintf(last_error_message, sizeof(last_error_message),
                  "CUDA backend failure during %s at %s:%d: %s (%s)",
                  operation, file, line,
                  name ? name : "unknown CUDA error",
                  description ? description : "no CUDA error description");
    std::fprintf(stderr, "%s\n", last_error_message);
    return -1;
}

#define CUDA_CALL(operation, expression)                                      \
    do {                                                                       \
        cudaError_t denise_cuda_status = (expression);                         \
        if (denise_cuda_status != cudaSuccess)                                 \
            return cuda_failure((operation), denise_cuda_status,               \
                                __FILE__, __LINE__);                            \
    } while (0)

int validate_view(const struct denise_cuda_view2d_f32 *view,
                  const char *operation) {
    if (!view)
        return contract_failure(operation, __FILE__, __LINE__,
                                "device view is null");
    if (!view->device_base)
        return contract_failure(operation, __FILE__, __LINE__,
                                "device allocation is null");
    if (!view->nx_alloc || !view->ny_alloc)
        return contract_failure(operation, __FILE__, __LINE__,
                                "device dimensions must be positive");
    if (view->pitch_elements < view->nx_alloc)
        return contract_failure(operation, __FILE__, __LINE__,
                                "pitch %zu is smaller than row width %zu",
                                view->pitch_elements, view->nx_alloc);
    const long long i_last =
        static_cast<long long>(view->i0) +
        static_cast<long long>(view->nx_alloc) - 1;
    const long long j_last =
        static_cast<long long>(view->j0) +
        static_cast<long long>(view->ny_alloc) - 1;
    if (i_last > INT_MAX || j_last > INT_MAX)
        return contract_failure(operation, __FILE__, __LINE__,
                                "logical upper bound exceeds INT_MAX");
    return 0;
}

int validate_host_matrix(float *const *host,
                         const struct denise_cuda_view2d_f32 *view,
                         const char *operation) {
    if (validate_view(view, operation) != 0) return -1;
    if (!host)
        return contract_failure(operation, __FILE__, __LINE__,
                                "host matrix is null");
    if (!host[view->j0])
        return contract_failure(operation, __FILE__, __LINE__,
                                "host matrix row %d is null", view->j0);
    float *payload = host[view->j0] + view->i0;
    for (size_t row = 0; row < view->ny_alloc; ++row) {
        const int j = view->j0 + static_cast<int>(row);
        if (!host[j])
            return contract_failure(operation, __FILE__, __LINE__,
                                    "host matrix row %d is null", j);
        if (host[j] + view->i0 != payload + row * view->nx_alloc)
            return contract_failure(
                operation, __FILE__, __LINE__,
                "host matrix row %d is not part of one contiguous payload", j);
    }
    return 0;
}

bool rectangle_inside(const struct denise_cuda_view2d_f32 *view,
                      int j_lo, int j_hi, int i_lo, int i_hi) {
    if (j_hi < j_lo || i_hi < i_lo) return false;
    const long long view_i_hi =
        static_cast<long long>(view->i0) +
        static_cast<long long>(view->nx_alloc) - 1;
    const long long view_j_hi =
        static_cast<long long>(view->j0) +
        static_cast<long long>(view->ny_alloc) - 1;
    return i_lo >= view->i0 && j_lo >= view->j0 &&
           static_cast<long long>(i_hi) <= view_i_hi &&
           static_cast<long long>(j_hi) <= view_j_hi;
}

__global__ void index_affine_kernel(struct denise_cuda_view2d_f32 view,
                                    int j_lo, int j_hi,
                                    int i_lo, int i_hi) {
    const int i = i_lo + static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    const int j = j_lo + static_cast<int>(
        blockIdx.y * blockDim.y + threadIdx.y);
    if (i > i_hi || j > j_hi) return;
    const size_t offset =
        static_cast<size_t>(j - view.j0) * view.pitch_elements +
        static_cast<size_t>(i - view.i0);
    const long long delta_integer =
        3LL * static_cast<long long>(i) +
        5LL * static_cast<long long>(j) + 7LL;
    view.device_base[offset] += static_cast<float>(delta_integer);
}

}  // namespace

extern "C" {

const char *denise_cuda_last_error(void) {
    return last_error_message;
}

int denise_cuda_visible_device_count(int *count) {
    clear_error();
    if (!count)
        return contract_failure("visible_device_count", __FILE__, __LINE__,
                                "output count is null");
    *count = 0;
    const cudaError_t status = cudaGetDeviceCount(count);
    if (status == cudaErrorNoDevice) {
        /* No visible device is a valid discovery result. Selection still
         * fails closed when CUDA execution is explicitly requested. */
        const cudaError_t cleared = cudaGetLastError();
        if (cleared != cudaSuccess && cleared != cudaErrorNoDevice)
            return cuda_failure("cudaGetLastError after no-device discovery",
                                cleared, __FILE__, __LINE__);
        *count = 0;
        return 0;
    }
    if (status != cudaSuccess)
        return cuda_failure("cudaGetDeviceCount", status,
                            __FILE__, __LINE__);
    return 0;
}

int denise_cuda_select_device(int logical_device,
                              size_t safety_reserve_bytes,
                              size_t user_cap_bytes,
                              struct denise_cuda_device_info *info) {
    clear_error();
    backend_initialized = false;
    backend_usable_budget_bytes = 0;
    if (!info)
        return contract_failure("select_device", __FILE__, __LINE__,
                                "device information output is null");

    int count = 0;
    if (denise_cuda_visible_device_count(&count) != 0) return -1;
    if (count < 1)
        return contract_failure("select_device", __FILE__, __LINE__,
                                "no CUDA device is visible");
    if (logical_device < 0 || logical_device >= count)
        return contract_failure(
            "select_device", __FILE__, __LINE__,
            "logical device %d is outside visible range [0,%d)",
            logical_device, count);

    CUDA_CALL("cudaSetDevice", cudaSetDevice(logical_device));

    cudaDeviceProp properties;
    std::memset(&properties, 0, sizeof(properties));
    CUDA_CALL("cudaGetDeviceProperties",
              cudaGetDeviceProperties(&properties, logical_device));

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CALL("cudaMemGetInfo", cudaMemGetInfo(&free_bytes, &total_bytes));

    std::memset(info, 0, sizeof(*info));
    info->visible_device_count = count;
    info->selected_logical_device = logical_device;
    std::snprintf(info->name, sizeof(info->name), "%s", properties.name);
    info->compute_capability_major = properties.major;
    info->compute_capability_minor = properties.minor;
    info->total_bytes = total_bytes;
    info->free_bytes = free_bytes;
    info->safety_reserve_bytes = safety_reserve_bytes;
    info->user_cap_bytes = user_cap_bytes;

    size_t effective_available_bytes = free_bytes;
    if (user_cap_bytes && user_cap_bytes < effective_available_bytes)
        effective_available_bytes = user_cap_bytes;
    if (effective_available_bytes <= safety_reserve_bytes)
        return contract_failure(
            "select_device VRAM budget", __FILE__, __LINE__,
            "effective available memory %zu must exceed safety reserve %zu "
            "(free=%zu, cap=%zu; cap=0 means unlimited)",
            effective_available_bytes, safety_reserve_bytes,
            free_bytes, user_cap_bytes);

    info->usable_budget_bytes =
        effective_available_bytes - safety_reserve_bytes;
    backend_usable_budget_bytes = info->usable_budget_bytes;
    backend_initialized = true;
    return 0;
}

int denise_cuda_allocate_view2d(struct denise_cuda_view2d_f32 *view,
                                int j0, int j1, int i0, int i1) {
    clear_error();
    if (!view)
        return contract_failure("allocate_view2d", __FILE__, __LINE__,
                                "device view is null");
    std::memset(view, 0, sizeof(*view));
    if (j1 < j0 || i1 < i0)
        return contract_failure("allocate_view2d", __FILE__, __LINE__,
                                "invalid logical bounds j=[%d,%d] i=[%d,%d]",
                                j0, j1, i0, i1);

    if (!backend_initialized)
        return contract_failure(
            "allocate_view2d", __FILE__, __LINE__,
            "CUDA device is not initialized with a positive usable budget");

    const unsigned long long nx =
        static_cast<unsigned long long>(
            static_cast<long long>(i1) - static_cast<long long>(i0)) + 1ULL;
    const unsigned long long ny =
        static_cast<unsigned long long>(
            static_cast<long long>(j1) - static_cast<long long>(j0)) + 1ULL;
    if (nx > static_cast<unsigned long long>(SIZE_MAX) ||
        ny > static_cast<unsigned long long>(SIZE_MAX) ||
        (nx && ny > static_cast<unsigned long long>(SIZE_MAX) / nx))
        return contract_failure("allocate_view2d", __FILE__, __LINE__,
                                "requested dimensions overflow size_t");
    const size_t elements =
        static_cast<size_t>(nx) * static_cast<size_t>(ny);
    if (elements > SIZE_MAX / sizeof(float))
        return contract_failure("allocate_view2d", __FILE__, __LINE__,
                                "requested allocation size overflows size_t");
    const size_t allocation_bytes = elements * sizeof(float);
    if (allocation_bytes > backend_usable_budget_bytes)
        return contract_failure(
            "allocate_view2d VRAM budget", __FILE__, __LINE__,
            "requested allocation %zu exceeds usable budget %zu",
            allocation_bytes, backend_usable_budget_bytes);

    float *device_base = nullptr;
    CUDA_CALL("cudaMalloc",
              cudaMalloc(reinterpret_cast<void **>(&device_base),
                         allocation_bytes));

    view->device_base = device_base;
    view->pitch_elements = static_cast<size_t>(nx);
    view->i0 = i0;
    view->j0 = j0;
    view->nx_alloc = static_cast<size_t>(nx);
    view->ny_alloc = static_cast<size_t>(ny);
    return 0;
}

int denise_cuda_release_view2d(struct denise_cuda_view2d_f32 *view) {
    clear_error();
    if (!view)
        return contract_failure("release_view2d", __FILE__, __LINE__,
                                "device view is null");
    if (view->device_base)
        CUDA_CALL("cudaFree", cudaFree(view->device_base));
    std::memset(view, 0, sizeof(*view));
    return 0;
}

int denise_cuda_copy_matrix_to_device(
        struct denise_cuda_view2d_f32 *view,
        float *const *host_matrix) {
    clear_error();
    if (validate_host_matrix(host_matrix, view,
                             "copy_matrix_to_device") != 0)
        return -1;
    const size_t row_bytes = view->nx_alloc * sizeof(float);
    CUDA_CALL("cudaMemcpy2D host-to-device",
              cudaMemcpy2D(view->device_base,
                           view->pitch_elements * sizeof(float),
                           &host_matrix[view->j0][view->i0],
                           row_bytes, row_bytes, view->ny_alloc,
                           cudaMemcpyHostToDevice));
    return 0;
}

int denise_cuda_copy_matrix_to_host(
        float **host_matrix,
        const struct denise_cuda_view2d_f32 *view) {
    clear_error();
    if (validate_host_matrix(host_matrix, view,
                             "copy_matrix_to_host") != 0)
        return -1;
    const size_t row_bytes = view->nx_alloc * sizeof(float);
    CUDA_CALL("cudaMemcpy2D device-to-host",
              cudaMemcpy2D(&host_matrix[view->j0][view->i0],
                           row_bytes, view->device_base,
                           view->pitch_elements * sizeof(float),
                           row_bytes, view->ny_alloc,
                           cudaMemcpyDeviceToHost));
    return 0;
}

int denise_cuda_apply_index_affine(
        struct denise_cuda_view2d_f32 *view,
        int j_lo, int j_hi, int i_lo, int i_hi) {
    clear_error();
    if (validate_view(view, "apply_index_affine") != 0) return -1;
    if (!rectangle_inside(view, j_lo, j_hi, i_lo, i_hi))
        return contract_failure(
            "apply_index_affine", __FILE__, __LINE__,
            "rectangle j=[%d,%d] i=[%d,%d] is outside the device view",
            j_lo, j_hi, i_lo, i_hi);

    const dim3 block(16, 16);
    const dim3 grid(
        static_cast<unsigned int>((i_hi - i_lo + 1 + block.x - 1) / block.x),
        static_cast<unsigned int>((j_hi - j_lo + 1 + block.y - 1) / block.y));
    index_affine_kernel<<<grid, block>>>(*view, j_lo, j_hi, i_lo, i_hi);
    CUDA_CALL("index_affine_kernel launch", cudaGetLastError());
    CUDA_CALL("index_affine_kernel completion", cudaDeviceSynchronize());
    return 0;
}

}  // extern "C"
