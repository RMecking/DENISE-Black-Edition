/* Inactive exact viscoelastic SH single-shot objective/raw-gradient bridge. */

#include "fd.h"

struct exact_shot_workspace {
    struct waveSH wave;
    struct waveSH_PML pml;
    struct visco_sh_full_state state;
};

static void exact_shot_map_state(struct exact_shot_workspace *workspace) {
    workspace->state.vz = workspace->wave.pvz;
    workspace->state.sxz = workspace->wave.psxz;
    workspace->state.syz = workspace->wave.psyz;
    workspace->state.r = workspace->wave.pr;
    workspace->state.q = workspace->wave.pq;
    workspace->state.psi_sxz_x = workspace->pml.psi_sxz_x;
    workspace->state.psi_syz_y = workspace->pml.psi_syz_y;
    workspace->state.psi_vzx = workspace->pml.psi_vzx;
    workspace->state.psi_vzy = workspace->pml.psi_vzy;
}

static void exact_shot_zero_state(
        struct exact_shot_workspace *workspace, int nx, int ny,
        int fdorder, int mechanisms, int fw) {
    int i, j, l, half = fdorder / 2;
    for (j = -half; j <= ny + half + 1; ++j)
        for (i = -half; i <= nx + half + 1; ++i) {
            workspace->state.vz[j][i] = 0.0f;
            workspace->state.sxz[j][i] = 0.0f;
            workspace->state.syz[j][i] = 0.0f;
            for (l = 1; l <= mechanisms; ++l) {
                workspace->state.r[j][i][l] = 0.0f;
                workspace->state.q[j][i][l] = 0.0f;
            }
        }
    for (j = 1; j <= ny; ++j)
        for (i = 1; i <= 2 * fw; ++i) {
            workspace->state.psi_sxz_x[j][i] = 0.0f;
            workspace->state.psi_vzx[j][i] = 0.0f;
        }
    for (j = 1; j <= 2 * fw; ++j)
        for (i = 1; i <= nx; ++i) {
            workspace->state.psi_syz_y[j][i] = 0.0f;
            workspace->state.psi_vzy[j][i] = 0.0f;
        }
}

static int exact_shot_allocate_workspace(
        struct exact_shot_workspace *workspace, int nx, int ny,
        int fdorder, int mechanisms, int fw) {
    memset(workspace, 0, sizeof(*workspace));
    alloc_SH(&workspace->wave, &workspace->pml);
    exact_shot_map_state(workspace);
    exact_shot_zero_state(workspace, nx, ny, fdorder, mechanisms, fw);
    return 0;
}

static void exact_shot_release_workspace(
        struct exact_shot_workspace *workspace) {
    dealloc_SH(&workspace->wave, &workspace->pml);
    memset(workspace, 0, sizeof(*workspace));
}

static double exact_shot_reference_sum(int mechanisms, const float *fl) {
    double sum = 0.0, omega;
    int l;
    omega = 2.0 * PI * fl[1];
    for (l = 1; l <= mechanisms; ++l) {
        double theta = 1.0 / (2.0 * PI * fl[l]);
        double value = omega * theta;
        sum += value * value / (1.0 + value * value);
    }
    return sum;
}

static int exact_shot_preflight(
        const struct visco_sh_exact_shot_request *request) {
    extern int DTINV, LNORM, GRAD_FORM, N_ORDER, TIMEWIN, OFFSET_MUTE;
    extern int TRKILL, NX, NY, NT, L, INVMAT1, SEISMO;
    if ((request == NULL) || (request->wave == NULL) ||
            (request->pml == NULL) || (request->material == NULL) ||
            (request->fwi == NULL) || (request->mpi == NULL) ||
            (request->seismogram == NULL) ||
            (request->legacy_fwi_seismogram == NULL) ||
            (request->acquisition == NULL) || (request->hc == NULL) ||
            (request->dtinv_help == NULL) ||
            (request->grad_primary == NULL) ||
            (request->grad_rho == NULL) || (request->grad_q == NULL) ||
            (request->ns != NT) || (request->nrec_local < 0) ||
            (request->nsrc_local < 0) || (NX < 1) || (NY < 1) ||
            (L < 1) || ((INVMAT1 != 1) && (INVMAT1 != 3)) ||
            (DTINV != 1) || (LNORM != 2) || (GRAD_FORM != 2) ||
            (N_ORDER != 0) || (TIMEWIN != 0) || (OFFSET_MUTE != 0) ||
            (TRKILL != 0) || (SEISMO != 1)) return -1;
    if ((request->nrec_local > 0) &&
            ((request->observed_vz == NULL) ||
             (request->seismogram->sectionvz == NULL) ||
             (request->acquisition->recpos_loc == NULL))) return -1;
    if ((request->nsrc_local > 0) &&
            ((request->acquisition->srcpos_loc == NULL) ||
             (request->acquisition->signals == NULL))) return -1;
    return 0;
}

int visco_sh_exact_objective_gradient_shot(
        const struct visco_sh_exact_shot_request *request,
        struct visco_sh_exact_shot_result *result) {
    extern int NX, NY, NT, FDORDER, L, FW, FREE_SURF, BOUNDARY;
    extern int POS[3], NPROCX, NPROCY, INDEX[5], INVMAT1;
    extern int Q_PARAMETERIZATION_MODE;
    extern float DT, DH, *FL, Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF;
    struct visco_sh_material_observable_trajectory trajectory;
    struct visco_sh_full_step_config config;
    struct visco_sh_reverse_time_material_context material_context;
    struct q_tau_mapping mapping;
    struct exact_shot_workspace terminal, initial, scratch;
    double *bar_receiver = NULL, *bar_signal = NULL;
    int *rec_x = NULL, *rec_y = NULL, *src_x = NULL, *src_y = NULL;
    int *source_type = NULL;
    double local_objective = 0.0, global_objective = 0.0;
    int i, n, status = -1, trajectory_ready = 0;

    memset(&trajectory, 0, sizeof(trajectory));
    memset(&terminal, 0, sizeof(terminal));
    memset(&initial, 0, sizeof(initial));
    memset(&scratch, 0, sizeof(scratch));
    if ((result == NULL) || exact_shot_preflight(request) != 0) return -1;
    if (visco_sh_material_observable_trajectory_init(
                &trajectory, NX, NY, NT, 1, FW, FREE_SURF, BOUNDARY,
                NPROCX, NPROCY) != 0) goto cleanup;
    trajectory_ready = 1;

    sh_visc_with_material_trajectory(
            request->wave, request->pml, request->material, request->fwi,
            request->mpi, request->seismogram,
            request->legacy_fwi_seismogram, request->acquisition, request->hc,
            request->ishot, request->nshots, request->nsrc_local, request->ns,
            request->nrec_local, request->source_energy,
            request->receiver_energy, request->hin, request->dtinv_help, 0,
            request->request_send, request->request_receive, &trajectory);
    if (visco_sh_material_observable_is_active()) goto cleanup;

    if (request->nrec_local > 0) {
        bar_receiver = (double *)calloc(
                (size_t)NT * request->nrec_local, sizeof(*bar_receiver));
        rec_x = (int *)malloc((size_t)request->nrec_local * sizeof(*rec_x));
        rec_y = (int *)malloc((size_t)request->nrec_local * sizeof(*rec_y));
        if ((bar_receiver == NULL) || (rec_x == NULL) || (rec_y == NULL))
            goto cleanup;
        for (i = 0; i < request->nrec_local; ++i) {
            rec_x[i] = request->acquisition->recpos_loc[1][i + 1];
            rec_y[i] = request->acquisition->recpos_loc[2][i + 1];
        }
        for (n = 1; n < NT; ++n)
            for (i = 0; i < request->nrec_local; ++i) {
                double residual =
                    request->seismogram->sectionvz[i + 1][n + 1] -
                    request->observed_vz[i + 1][n + 1];
                bar_receiver[(size_t)n * request->nrec_local + i] = residual;
                local_objective += 0.5 * residual * residual;
            }
        if (request->receiver_cotangent != NULL)
            memcpy(request->receiver_cotangent, bar_receiver,
                   (size_t)NT * request->nrec_local * sizeof(*bar_receiver));
    }
    MPI_Allreduce(&local_objective, &global_objective, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    if (request->nsrc_local > 0) {
        bar_signal = (double *)calloc(
                (size_t)NT * request->nsrc_local, sizeof(*bar_signal));
        src_x = (int *)malloc((size_t)request->nsrc_local * sizeof(*src_x));
        src_y = (int *)malloc((size_t)request->nsrc_local * sizeof(*src_y));
        source_type = (int *)malloc(
                (size_t)request->nsrc_local * sizeof(*source_type));
        if ((bar_signal == NULL) || (src_x == NULL) || (src_y == NULL) ||
                (source_type == NULL)) goto cleanup;
        for (i = 0; i < request->nsrc_local; ++i) {
            src_x[i] = iround(request->acquisition->srcpos_loc[1][i + 1]);
            src_y[i] = iround(request->acquisition->srcpos_loc[2][i + 1]);
            source_type[i] = iround(
                    request->acquisition->srcpos_loc[8][i + 1]);
        }
    }

    memset(&config, 0, sizeof(config));
    config.nx = NX; config.ny = NY; config.fdorder = FDORDER;
    config.mechanisms = L; config.fw = FW;
    config.free_surface = FREE_SURF; config.boundary = BOUNDARY;
    for (i = 0; i < 3; ++i) config.pos[i] = POS[i];
    config.nproc_x = NPROCX; config.nproc_y = NPROCY;
    for (i = 0; i < 5; ++i) config.index[i] = INDEX[i];
    config.dt = DT; config.dh = DH; config.hc = request->hc;
    config.rhoi = request->material->prhoi;
    config.fipjp = request->material->fipjp; config.f = request->material->f;
    config.bip = request->material->bip; config.bjm = request->material->bjm;
    config.cip = request->material->cip; config.cjm = request->material->cjm;
    config.dip = request->material->dip; config.d = request->material->d;
    config.K_x = request->pml->K_x; config.a_x = request->pml->a_x;
    config.b_x = request->pml->b_x;
    config.K_x_half = request->pml->K_x_half;
    config.a_x_half = request->pml->a_x_half;
    config.b_x_half = request->pml->b_x_half;
    config.K_y = request->pml->K_y; config.a_y = request->pml->a_y;
    config.b_y = request->pml->b_y;
    config.K_y_half = request->pml->K_y_half;
    config.a_y_half = request->pml->a_y_half;
    config.b_y_half = request->pml->b_y_half;
    config.nrec = request->nrec_local; config.rec_x = rec_x;
    config.rec_y = rec_y; config.nsrc = request->nsrc_local;
    config.src_x = src_x; config.src_y = src_y;
    config.source_type = source_type; config.comm = MPI_COMM_WORLD;

    init_q_tau_mapping(&mapping, Q_PARAMETERIZATION_MODE, L, FL,
                       Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF);
    memset(&material_context, 0, sizeof(material_context));
    material_context.trajectory = &trajectory;
    material_context.mu_x = request->material->puip;
    material_context.tau_x = request->material->ptausipjp;
    material_context.mu_y = request->material->pujp;
    material_context.tau_y = request->material->ptaus;
    material_context.reference_sum = exact_shot_reference_sum(L, FL);
    material_context.eta_x = request->material->etaip;
    material_context.eta_y = request->material->etajm;
    material_context.invmat1 = INVMAT1; material_context.mapping = &mapping;
    material_context.primary_post = request->material->pu;
    material_context.rho_post = request->material->prho;
    material_context.owned_q = request->material->pqs;
    material_context.grad_primary = request->grad_primary;
    material_context.grad_rho = request->grad_rho;
    material_context.grad_q = request->grad_q;

    if ((exact_shot_allocate_workspace(
                 &terminal, NX, NY, FDORDER, L, FW) != 0) ||
            (exact_shot_allocate_workspace(
                 &initial, NX, NY, FDORDER, L, FW) != 0) ||
            (exact_shot_allocate_workspace(
                 &scratch, NX, NY, FDORDER, L, FW) != 0)) goto cleanup;
    status = visco_sh_reverse_time_adjoint_material(
            &config, NT, bar_receiver, &terminal.state, &initial.state,
            &scratch.state, bar_signal, &material_context);
    if (status == 0) result->objective = global_objective;

cleanup:
    if (scratch.state.vz != NULL) exact_shot_release_workspace(&scratch);
    if (initial.state.vz != NULL) exact_shot_release_workspace(&initial);
    if (terminal.state.vz != NULL) exact_shot_release_workspace(&terminal);
    free(source_type); free(src_y); free(src_x); free(rec_y); free(rec_x);
    free(bar_signal); free(bar_receiver);
    if (trajectory_ready)
        visco_sh_material_observable_trajectory_release(&trajectory);
    return status;
}
