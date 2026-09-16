/* Passive, time-major capture of the three forward observables required by
 * the exact viscoelastic SH material VJPs.  The active step is process-local;
 * DENISE uses one MPI process per subdomain and the forward kernels are not
 * concurrently entered within a process. */

#include "fd.h"

static struct visco_sh_material_observable_step *active_step = NULL;
static int active_nx = 0;
static int active_ny = 0;
#if defined(M63C_MATERIAL_OBSERVABLE_TEST_COUNTERS)
static size_t qsum_call_count = 0;
static size_t strain_call_count = 0;
#endif

static float **owned_matrix(int nx, int ny) {
    float **rows;
    float *values;
    int j;

    rows = (float **)calloc((size_t)ny + 1, sizeof(float *));
    values = (float *)calloc(((size_t)ny + 1) * ((size_t)nx + 1),
                             sizeof(float));
    if ((rows == NULL) || (values == NULL)) {
        free(values);
        free(rows);
        return NULL;
    }
    for (j = 0; j <= ny; ++j)
        rows[j] = values + (size_t)j * ((size_t)nx + 1);
    return rows;
}

static void free_owned_matrix(float **matrix) {
    if (matrix != NULL) {
        free(matrix[0]);
        free(matrix);
    }
}

int visco_sh_material_observable_trajectory_init(
        struct visco_sh_material_observable_trajectory *trajectory,
        int nx, int ny, int nsteps, int dtinv, int fw, int free_surface,
        int boundary, int nproc_x, int nproc_y) {
    int n;

    if ((trajectory == NULL) || (nx < 1) || (ny < 1) || (nsteps < 1) ||
            (dtinv != 1) || (fw < 0) || (nproc_x < 1) || (nproc_y < 1))
        return -1;
    if ((fw > 0) && !boundary && (nproc_x == 1) && (nx <= 2 * fw))
        return -2;
    if ((fw > 0) && !free_surface && (nproc_y == 1) && (ny <= 2 * fw))
        return -2;

    memset(trajectory, 0, sizeof(*trajectory));
    trajectory->nx = nx;
    trajectory->ny = ny;
    trajectory->nsteps = nsteps;
    trajectory->dtinv = dtinv;
    trajectory->steps = (struct visco_sh_material_observable_step *)calloc(
            (size_t)nsteps, sizeof(*trajectory->steps));
    if (trajectory->steps == NULL) return -3;

    for (n = 0; n < nsteps; ++n) {
        trajectory->steps[n].qsum = owned_matrix(nx, ny);
        trajectory->steps[n].strain_x = owned_matrix(nx, ny);
        trajectory->steps[n].strain_y = owned_matrix(nx, ny);
        if ((trajectory->steps[n].qsum == NULL) ||
                (trajectory->steps[n].strain_x == NULL) ||
                (trajectory->steps[n].strain_y == NULL)) {
            visco_sh_material_observable_trajectory_release(trajectory);
            return -3;
        }
    }
    return 0;
}

void visco_sh_material_observable_trajectory_release(
        struct visco_sh_material_observable_trajectory *trajectory) {
    int n;

    if (trajectory == NULL) return;
    if (active_step != NULL) visco_sh_material_observable_end_step();
    for (n = 0; n < trajectory->nsteps; ++n) {
        free_owned_matrix(trajectory->steps[n].qsum);
        free_owned_matrix(trajectory->steps[n].strain_x);
        free_owned_matrix(trajectory->steps[n].strain_y);
    }
    free(trajectory->steps);
    memset(trajectory, 0, sizeof(*trajectory));
}

int visco_sh_material_observable_begin_step(
        struct visco_sh_material_observable_trajectory *trajectory, int step) {
    if ((trajectory == NULL) || (trajectory->steps == NULL) ||
            (trajectory->dtinv != 1) || (step < 0) ||
            (step >= trajectory->nsteps) || (active_step != NULL))
        return -1;
    active_step = &trajectory->steps[step];
    active_nx = trajectory->nx;
    active_ny = trajectory->ny;
    return 0;
}

void visco_sh_material_observable_end_step(void) {
    active_step = NULL;
    active_nx = 0;
    active_ny = 0;
}

int visco_sh_material_observable_is_active(void) {
    return active_step != NULL;
}

void visco_sh_material_observable_capture_qsum(int j, int i, float qsum) {
#if defined(M63C_MATERIAL_OBSERVABLE_TEST_COUNTERS)
    ++qsum_call_count;
#endif
    if ((active_step != NULL) && (j >= 1) && (j <= active_ny) &&
            (i >= 1) && (i <= active_nx))
        active_step->qsum[j][i] = qsum;
}

void visco_sh_material_observable_capture_strain(
        int j, int i, float strain_x, float strain_y) {
#if defined(M63C_MATERIAL_OBSERVABLE_TEST_COUNTERS)
    ++strain_call_count;
#endif
    if ((active_step != NULL) && (j >= 1) && (j <= active_ny) &&
            (i >= 1) && (i <= active_nx)) {
        active_step->strain_x[j][i] = strain_x;
        active_step->strain_y[j][i] = strain_y;
    }
}

#if defined(M63C_MATERIAL_OBSERVABLE_TEST_COUNTERS)
void visco_sh_material_observable_test_reset_counts(void) {
    qsum_call_count = 0;
    strain_call_count = 0;
}

size_t visco_sh_material_observable_test_qsum_count(void) {
    return qsum_call_count;
}

size_t visco_sh_material_observable_test_strain_count(void) {
    return strain_call_count;
}
#endif
