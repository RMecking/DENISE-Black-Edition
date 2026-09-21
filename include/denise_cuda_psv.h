#ifndef DENISE_CUDA_PSV_H
#define DENISE_CUDA_PSV_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * B1 intentionally excludes source injection, receiver sampling, gradient
 * auxiliaries (ux/uy/uxy), adjoint/gradient state, MPI exchange, free surface,
 * periodic boundaries, arbitrary FD order, and generic L. Production psv()
 * dispatch remains CPU-only until B2.
 */
struct denise_cuda_psv_fd4_l1;

struct denise_cuda_psv_fd4_l1_config {
    int nx;
    int ny;
    int fw;
    int fdorder;
    int mechanisms;
    int mpi_ranks_x;
    int mpi_ranks_y;
    int boundary;
    int free_surface;
    int mode;
    int logical_device;
    float dt;
    float dh;
    float hc1;
    float hc2;
    float bip1;
    float bjm1;
    float cip1;
    float cjm1;
    size_t safety_reserve_bytes;
    size_t user_cap_bytes;
};

/*
 * Host adapter for the frozen B1 envelope. Wavefield, material, and L=1
 * tensor planes use the production FD4 bounds j=-2..ny+3, i=-2..nx+3.
 * L=1 tensors remain CPU f3tensor objects and are adapted from [...][...][1].
 * x-CPML matrices use j=1..ny, h=1..2*fw; y-CPML matrices use
 * h=1..2*fw, i=1..nx. Coefficient vectors use h=1..2*fw.
 */
struct denise_cuda_psv_fd4_l1_host {
    float **vx;
    float **vy;
    float **sxx;
    float **syy;
    float **sxy;

    float ***r;
    float ***p;
    float ***q;

    float **rip;
    float **rjp;
    float **fipjp;
    float **f;
    float **g;
    float ***dip;
    float ***d;
    float ***e;

    float *K_x;
    float *a_x;
    float *b_x;
    float *K_x_half;
    float *a_x_half;
    float *b_x_half;
    float *K_y;
    float *a_y;
    float *b_y;
    float *K_y_half;
    float *a_y_half;
    float *b_y_half;

    float **psi_sxx_x;
    float **psi_sxy_x;
    float **psi_syy_y;
    float **psi_sxy_y;
    float **psi_vxx;
    float **psi_vyx;
    float **psi_vyy;
    float **psi_vxy;
};

struct denise_cuda_psv_fd4_l1_stats {
    size_t mandatory_state_bytes;
    size_t usable_budget_bytes;
    size_t remaining_budget_bytes;
    size_t h2d_transfer_calls;
    size_t d2h_transfer_calls;
    size_t h2d_bytes;
    size_t d2h_bytes;
    unsigned long long timesteps;
    float velocity_kernel_ms;
    float stress_kernel_ms;
    float combined_kernel_ms;
    float upload_ms;
    float download_ms;
};

const char *denise_cuda_psv_fd4_l1_last_error(void);

int denise_cuda_psv_fd4_l1_required_bytes(
        const struct denise_cuda_psv_fd4_l1_config *config,
        size_t *required_bytes);

int denise_cuda_psv_fd4_l1_create(
        const struct denise_cuda_psv_fd4_l1_config *config,
        const struct denise_cuda_psv_fd4_l1_host *initial,
        struct denise_cuda_psv_fd4_l1 **context);

int denise_cuda_psv_fd4_l1_step(
        struct denise_cuda_psv_fd4_l1 *context,
        int timestep_count);

int denise_cuda_psv_fd4_l1_download(
        struct denise_cuda_psv_fd4_l1 *context,
        struct denise_cuda_psv_fd4_l1_host *output);

int denise_cuda_psv_fd4_l1_get_stats(
        const struct denise_cuda_psv_fd4_l1 *context,
        struct denise_cuda_psv_fd4_l1_stats *stats);

int denise_cuda_psv_fd4_l1_destroy(
        struct denise_cuda_psv_fd4_l1 **context);

#ifdef __cplusplus
}
#endif

#endif
