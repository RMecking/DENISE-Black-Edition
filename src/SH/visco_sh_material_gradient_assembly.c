/* C7c-a temporal reduction and exact distributed material-gradient mapping. */

#include "fd.h"

struct local_field {
    float **row;
    float *data;
};

static int field_allocate(struct local_field *field, int nx, int ny) {
    int j;
    field->row = (float **)calloc((size_t)(ny + 2), sizeof(float *));
    field->data = (float *)calloc((size_t)(nx + 2) * (ny + 2), sizeof(float));
    if (field->row == NULL || field->data == NULL) {
        free(field->row);
        free(field->data);
        field->row = NULL;
        field->data = NULL;
        return -1;
    }
    for (j = 0; j <= ny + 1; ++j)
        field->row[j] = field->data + (size_t)j * (nx + 2);
    return 0;
}

static void field_release(struct local_field *field) {
    free(field->data);
    free(field->row);
}

int visco_sh_temporal_native_gradient_accumulate(
        int timesteps, int points, double dt, int dtinv,
        const struct visco_sh_material_timestep_vjp_output *series,
        struct visco_sh_material_timestep_vjp_output *accumulated) {
    double weight;
    int n, point;

    if (timesteps < 1 || points < 1 || !(dt > 0.0) || dtinv < 1 ||
            !isfinite(dt) || series == NULL || accumulated == NULL)
        return -1;
    weight = dt * (double)dtinv;
    for (point = 0; point < points; ++point)
        memset(&accumulated[point], 0, sizeof(accumulated[point]));
    for (n = 0; n < timesteps; ++n) {
        for (point = 0; point < points; ++point) {
            const struct visco_sh_material_timestep_vjp_output *step =
                    &series[n * points + point];
            struct visco_sh_material_timestep_vjp_output *sum =
                    &accumulated[point];
            sum->g_rhoi += step->g_rhoi;
            sum->g_mu_x += step->g_mu_x;
            sum->g_mu_y += step->g_mu_y;
            sum->g_tau_x += step->g_tau_x;
            sum->g_tau_y += step->g_tau_y;
        }
    }
    for (point = 0; point < points; ++point) {
        accumulated[point].g_rhoi *= weight;
        accumulated[point].g_mu_x *= weight;
        accumulated[point].g_mu_y *= weight;
        accumulated[point].g_tau_x *= weight;
        accumulated[point].g_tau_y *= weight;
    }
    return 0;
}

int visco_sh_distributed_material_gradient_vjp(
        int invmat1, const struct q_tau_mapping *mapping,
        float **primary_post, float **rho_post, float **owned_q,
        const struct visco_sh_native_material_gradient_fields *native,
        float **grad_primary, float **grad_rho, float **grad_q) {
    extern int NX, NY;
    struct local_field bar_primary = {NULL, NULL};
    struct local_field bar_rho = {NULL, NULL};
    struct local_field bar_tau = {NULL, NULL};
    struct local_field bar_mu = {NULL, NULL};
    double center, east, south, bar_center, bar_east, bar_south;
    double bar_tau_cells[4];
    int i, j, status = -1;

    if ((invmat1 != 1 && invmat1 != 3) || mapping == NULL ||
            primary_post == NULL || rho_post == NULL || owned_q == NULL ||
            native == NULL || native->g_rhoi == NULL ||
            native->g_mu_x == NULL || native->g_mu_y == NULL ||
            native->g_tau_x == NULL || native->g_tau_y == NULL ||
            grad_primary == NULL || grad_rho == NULL || grad_q == NULL ||
            NX < 1 || NY < 1)
        return -1;
    if (field_allocate(&bar_primary, NX, NY) != 0 ||
            field_allocate(&bar_rho, NX, NY) != 0 ||
            field_allocate(&bar_tau, NX, NY) != 0 ||
            field_allocate(&bar_mu, NX, NY) != 0) {
        status = MPI_ERR_NO_MEM;
        goto cleanup;
    }

    for (j = 0; j <= NY + 1; ++j) {
        for (i = 0; i <= NX + 1; ++i) {
            if (!(primary_post[j][i] > 0.0f) ||
                    !(rho_post[j][i] > 0.0f) ||
                    !isfinite(primary_post[j][i]) ||
                    !isfinite(rho_post[j][i])) {
                status = -2;
                goto cleanup;
            }
            grad_primary[j][i] = 0.0f;
            grad_rho[j][i] = 0.0f;
            grad_q[j][i] = 0.0f;
        }
    }

    for (j = 1; j <= NY; ++j) {
        for (i = 1; i <= NX; ++i) {
            center = invmat1 == 1
                   ? rho_post[j][i] * primary_post[j][i] * primary_post[j][i]
                   : primary_post[j][i];
            east = invmat1 == 1
                 ? rho_post[j][i + 1] * primary_post[j][i + 1]
                   * primary_post[j][i + 1]
                 : primary_post[j][i + 1];
            south = invmat1 == 1
                  ? rho_post[j + 1][i] * primary_post[j + 1][i]
                    * primary_post[j + 1][i]
                  : primary_post[j + 1][i];
            bar_center = bar_east = bar_south = 0.0;
            status = visco_sh_harmonic_pair_vjp(
                    center, east, native->g_mu_x[j][i],
                    &bar_center, &bar_east);
            if (status != 0) goto cleanup;
            status = visco_sh_harmonic_pair_vjp(
                    center, south, native->g_mu_y[j][i],
                    &bar_center, &bar_south);
            if (status != 0) goto cleanup;
            bar_mu.row[j][i] += (float)bar_center;
            bar_mu.row[j][i + 1] += (float)bar_east;
            bar_mu.row[j + 1][i] += (float)bar_south;

            memset(bar_tau_cells, 0, sizeof(bar_tau_cells));
            visco_sh_av_tau_local_vjp(
                    native->g_tau_x[j][i], native->g_tau_y[j][i],
                    bar_tau_cells);
            bar_tau.row[j][i] += (float)bar_tau_cells[0];
            bar_tau.row[j][i + 1] += (float)bar_tau_cells[1];
            bar_tau.row[j + 1][i] += (float)bar_tau_cells[2];
            bar_tau.row[j + 1][i + 1] += (float)bar_tau_cells[3];
            bar_rho.row[j][i] += (float)visco_sh_rhoi_vjp(
                    rho_post[j][i], native->g_rhoi[j][i]);
        }
    }

    for (j = 0; j <= NY + 1; ++j) {
        for (i = 0; i <= NX + 1; ++i) {
            if (invmat1 == 1) {
                bar_primary.row[j][i] += 2.0f * rho_post[j][i]
                        * primary_post[j][i] * bar_mu.row[j][i];
                bar_rho.row[j][i] += primary_post[j][i]
                        * primary_post[j][i] * bar_mu.row[j][i];
            } else {
                bar_primary.row[j][i] += bar_mu.row[j][i];
            }
        }
    }

    /* Reverse the actual V-then-H matcopy graph as H^T then V^T. */
    status = matcopy_SH_adjoint(
            bar_rho.row, bar_primary.row, bar_tau.row);
    if (status != MPI_SUCCESS) goto cleanup;

    /* Q -> tau occurs before matcopy in the forward graph, hence its VJP is
     * applied only now, on owned tau cotangents returned by matcopy^T. */
    for (j = 1; j <= NY; ++j) {
        for (i = 1; i <= NX; ++i) {
            grad_primary[j][i] = bar_primary.row[j][i];
            grad_rho[j][i] = bar_rho.row[j][i];
            grad_q[j][i] = (float)(q_to_tau_derivative(
                    owned_q[j][i], mapping) * bar_tau.row[j][i]);
        }
    }
    status = 0;

cleanup:
    field_release(&bar_primary);
    field_release(&bar_rho);
    field_release(&bar_tau);
    field_release(&bar_mu);
    return status;
}
