/* Raw, discrete viscoelastic P/SV physical gradient for the supported L=1,
 * FD4 receiver-velocity experiment, including Cartesian MPI decomposition.
 * The forward operators themselves remain in update_v_PML_PSV and
 * update_s_visc_PML_PSV; the hooks below record their post-CPML derivative
 * operands, including the velocity-update force. */
#include "fd.h"

extern int NX, NY, NXG, NYG, NT, FW, BOUNDARY, FREE_SURF, NPROCX, NPROCY, NDT;
extern int MODE, L, INVMAT1, GRAD_FORM, FDORDER, Q_PARAMETERIZATION_MODE;
extern int DTINV, LNORM, QUELLTYP, READMOD, NSRC, MYID_SHOT, POS[3], INDEX[5];
extern float DT, DH, *FL, Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF;
extern char JACOBIAN[STRING_SIZE];
extern MPI_Comm SHOT_COMM;

enum {VXX, VYX, VXY, VYY, FX, FY, NRECORD};
enum {GF, GG, GFC, GD, GE, GDC, GRX, GRY, NNATIVE};
enum {PSXX, PSXYX, PSXYY, PSYY, PVXX, PVYX, PVXY, PVYY, NPSI};

static struct {
    int active, step, pitch, area, replay_compare, replay_first;
    int full_storage, recording, record_first, record_capacity;
    int requested_segments, segment_count, max_segment_length;
    int checkpoint_count, checkpoints_captured, replayed_steps;
    int *segment_start, *segment_end;
    struct visco_psv_checkpoint **checkpoint;
    float *record_payload;
    float *record[NRECORD];
    double *send_a, *send_b, *recv_a, *recv_b;
    size_t communication_capacity;
    size_t replay_compared[NRECORD], replay_mismatches[NRECORD];
    double initial_started, initial_seconds, replay_seconds, reverse_seconds;
    double global_objective;
    int source_owners, receiver_ownership_valid;
} exact;

static size_t record_index(int t, int j, int i) {
    int local_t = exact.full_storage ? t : t - exact.record_first;
    return ((size_t)local_t * NY + (j - 1)) * NX + (i - 1);
}
static int cell(int j, int i) { return (j + 2) * exact.pitch + i + 2; }

static int enabled_flag(const char *name) {
    const char *value = getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
}

static int requested_segment_count(void) {
    const char *value = getenv("DENISE_PSV_EXACT_SEGMENTS");
    char *end = NULL;
    long parsed;
    if (!value || !value[0]) return 32;
    parsed = strtol(value, &end, 10);
    if (!end || *end || parsed < 1 || parsed > 2147483647L)
        err(" DENISE_PSV_EXACT_SEGMENTS must be a positive integer. ");
    return (int)parsed;
}

static void allocate_records(int capacity) {
    size_t field_values = (size_t)capacity * NX * NY;
    int k;
    exact.record_payload = calloc((size_t)NRECORD * field_values, sizeof(float));
    if (!exact.record_payload)
        err(" Out of memory recording exact visco PSV forward operands. ");
    for (k = 0; k < NRECORD; ++k)
        exact.record[k] = exact.record_payload + (size_t)k * field_values;
    exact.record_capacity = capacity;
}

int visco_psv_exact_supported(void) {
    int halo = FDORDER / 2 + 1;
    return MODE == 1 && L == 1 && INVMAT1 == 1 && GRAD_FORM == 2 &&
           FDORDER == 4 && !FREE_SURF && !BOUNDARY && FW > 0 &&
           NDT == 1 && DTINV == 1 && LNORM == 2 && READMOD == 1 &&
           NSRC == 1 && NX >= halo && NY >= halo && NX >= FW && NY >= FW;
}

int visco_psv_exact_enabled(void) {
    const char *flag = getenv("DENISE_PSV_EXACT_VISCO_GRADIENT");
    return visco_psv_exact_supported() ||
           (flag && flag[0] == '1' && flag[1] == '\0');
}

void visco_psv_exact_begin(void) {
    int k;
    if (!visco_psv_exact_enabled()) return;
    if (!visco_psv_exact_supported())
        err(" Exact visco PSV raw gradient supports one-source READMOD=1 L=1 FD4, INVMAT1=1, GRAD_FORM=2, NDT=DTINV=1, LNORM=2, CPML interior only, with local domains large enough for FD and CPML halos. ");
    /* Forward matrices span -2..NX+3/-2..NY+3 for FD4.  The one-rank
     * reverse stencil only touched through +2, but the distributed transpose
     * must hold the complete component-specific +3 halo. */
    exact.pitch = NX + 6;
    exact.area = (NY + 6) * exact.pitch;
    exact.full_storage = enabled_flag("DENISE_PSV_EXACT_FULL_STORAGE_REFERENCE") ||
                         enabled_flag("DENISE_PSV_CHECKPOINT_REPLAY_TEST");
    exact.requested_segments = requested_segment_count();
    exact.segment_count = exact.requested_segments > NT ? NT : exact.requested_segments;
    exact.segment_start = exact.segment_end = NULL;
    exact.checkpoint = NULL;
    exact.checkpoint_count = exact.checkpoints_captured = 0;
    exact.max_segment_length = 0;
    exact.replayed_steps = 0;
    exact.global_objective = 0.0;
    exact.source_owners = 0;
    exact.receiver_ownership_valid = 0;
    exact.send_a = exact.send_b = exact.recv_a = exact.recv_b = NULL;
    exact.communication_capacity = 0;
    exact.initial_seconds = exact.replay_seconds = exact.reverse_seconds = 0.0;
    if (exact.full_storage) {
        allocate_records(NT + 1);
        exact.record_first = 0;
        exact.recording = 1;
    } else {
        exact.segment_start = calloc((size_t)exact.segment_count, sizeof(int));
        exact.segment_end = calloc((size_t)exact.segment_count, sizeof(int));
        if (!exact.segment_start || !exact.segment_end)
            err(" Out of memory allocating exact visco PSV segment schedule. ");
        for (k = 0; k < exact.segment_count; ++k) {
            int length;
            exact.segment_start[k] = (int)(((long long)k * NT) / exact.segment_count);
            exact.segment_end[k] = (int)(((long long)(k + 1) * NT) / exact.segment_count);
            length = exact.segment_end[k] - exact.segment_start[k];
            if (length < 1) err(" Invalid exact visco PSV segment schedule. ");
            if (length > exact.max_segment_length) exact.max_segment_length = length;
        }
        exact.checkpoint_count = exact.segment_count - 1;
        if (exact.checkpoint_count) {
            exact.checkpoint = calloc((size_t)exact.checkpoint_count,
                                      sizeof(*exact.checkpoint));
            if (!exact.checkpoint)
                err(" Out of memory allocating exact visco PSV checkpoint table. ");
            for (k = 0; k < exact.checkpoint_count; ++k)
                exact.checkpoint[k] = visco_psv_checkpoint_create();
        }
        allocate_records(exact.max_segment_length);
        exact.record_first = 1;
        exact.recording = 0;
    }
    if (NPROCX * NPROCY > 1) {
        size_t maximum = (size_t)(NX > NY ? NX : NY) + 1;
        exact.communication_capacity = 5 * maximum;
        exact.send_a = calloc(exact.communication_capacity, sizeof(double));
        exact.send_b = calloc(exact.communication_capacity, sizeof(double));
        exact.recv_a = calloc(exact.communication_capacity, sizeof(double));
        exact.recv_b = calloc(exact.communication_capacity, sizeof(double));
        if (!exact.send_a || !exact.send_b || !exact.recv_a || !exact.recv_b)
            err(" Out of memory allocating distributed exact-visco adjoint exchange buffers. ");
    }
    exact.step = 0;
    exact.replay_compare = 0;
    exact.replay_first = 0;
    memset(exact.replay_compared, 0, sizeof(exact.replay_compared));
    memset(exact.replay_mismatches, 0, sizeof(exact.replay_mismatches));
    exact.active = 1;
    exact.initial_started = MPI_Wtime();
}

void visco_psv_exact_step(int t) { if (exact.active) exact.step = t; }

void visco_psv_exact_velocity(int j, int i, float force_x, float force_y) {
    size_t p;
    if (!exact.active || (!exact.recording && !exact.replay_compare)) return;
    p = record_index(exact.step, j, i);
    if (exact.replay_compare) {
        exact.replay_compared[FX]++;
        exact.replay_compared[FY]++;
        if (memcmp(&exact.record[FX][p], &force_x, sizeof(float)) != 0)
            exact.replay_mismatches[FX]++;
        if (memcmp(&exact.record[FY][p], &force_y, sizeof(float)) != 0)
            exact.replay_mismatches[FY]++;
    } else {
        exact.record[FX][p] = force_x;
        exact.record[FY][p] = force_y;
    }
}

void visco_psv_exact_strain(int j, int i, float vxx, float vyx,
                            float vxy, float vyy) {
    size_t p;
    if (!exact.active || (!exact.recording && !exact.replay_compare)) return;
    p = record_index(exact.step, j, i);
    if (exact.replay_compare) {
        float value[NRECORD] = {vxx, vyx, vxy, vyy, 0.0f, 0.0f};
        int k;
        for (k = VXX; k <= VYY; ++k) {
            exact.replay_compared[k]++;
            if (memcmp(&exact.record[k][p], &value[k], sizeof(float)) != 0)
                exact.replay_mismatches[k]++;
        }
    } else {
        exact.record[VXX][p] = vxx;
        exact.record[VYX][p] = vyx;
        exact.record[VXY][p] = vxy;
        exact.record[VYY][p] = vyy;
    }
}

void visco_psv_exact_replay_begin(int first_timestep) {
    if (!exact.active || !exact.full_storage || exact.replay_compare || first_timestep < 1 ||
        first_timestep > NT)
        err(" Invalid exact visco PSV operand-replay comparison start. ");
    exact.replay_first = first_timestep;
    memset(exact.replay_compared, 0, sizeof(exact.replay_compared));
    memset(exact.replay_mismatches, 0, sizeof(exact.replay_mismatches));
    exact.replay_compare = 1;
}

void visco_psv_exact_replay_end(size_t compared[NRECORD],
                                size_t mismatches[NRECORD]) {
    int k;
    if (!exact.active || !exact.replay_compare || exact.replay_first < 1)
        err(" Invalid exact visco PSV operand-replay comparison finish. ");
    for (k = 0; k < NRECORD; ++k) {
        compared[k] = exact.replay_compared[k];
        mismatches[k] = exact.replay_mismatches[k];
    }
    exact.replay_compare = 0;
    exact.replay_first = 0;
}

void visco_psv_exact_forward_boundary(struct wavePSV *wave,
                                      struct wavePSV_PML *pml,
                                      int timestep) {
    if (!exact.active || exact.full_storage ||
        exact.checkpoints_captured >= exact.checkpoint_count) return;
    if (timestep == exact.segment_end[exact.checkpoints_captured]) {
        visco_psv_checkpoint_capture(
            exact.checkpoint[exact.checkpoints_captured], wave, pml, timestep);
        exact.checkpoints_captured++;
    }
}

static void zero_forward_state(struct wavePSV *wave, struct wavePSV_PML *pml) {
    int nd = FDORDER / 2 + 1;
    zero_denise_visc_PSV(-nd + 1, NY + nd, -nd + 1, NX + nd,
                         wave->pvx, wave->pvy, wave->psxx, wave->psyy,
                         wave->psxy, wave->ux, wave->uy, wave->uxy,
                         wave->pvxp1, wave->pvyp1, pml->psi_sxx_x,
                         pml->psi_sxy_x, pml->psi_vxx, pml->psi_vyx,
                         pml->psi_syy_y, pml->psi_sxy_y, pml->psi_vyy,
                         pml->psi_vxy, pml->psi_vxxs,
                         wave->pr, wave->pp, wave->pq);
}

static void replay_forward_segment(
        const struct visco_psv_exact_fwi_request *request, int segment) {
    struct wavePSV *wave = request->wave;
    struct wavePSV_PML *pml = request->pml;
    struct matPSV *mat = request->material;
    struct mpiPSV *mpi = request->mpi;
    struct acq *acq = request->acquisition;
    int t, begin = exact.segment_start[segment] + 1;
    int end = exact.segment_end[segment];

    if (segment == 0) zero_forward_state(wave, pml);
    else visco_psv_checkpoint_restore(exact.checkpoint[segment - 1], wave, pml);
    memset(exact.record_payload, 0,
           (size_t)NRECORD * exact.record_capacity * NX * NY * sizeof(float));
    exact.record_first = begin;
    exact.recording = 1;
    for (t = begin; t <= end; ++t) {
        visco_psv_exact_step(t);
        update_v_PML_PSV(1, NX, 1, NY, t,
                         wave->pvx, wave->pvxp1, wave->pvxm1,
                         wave->pvy, wave->pvyp1, wave->pvym1,
                         wave->uttx, wave->utty,
                         wave->psxx, wave->psyy, wave->psxy,
                         mat->prip, mat->prjp, acq->srcpos_loc,
                         acq->signals, acq->signals, request->nsrc_loc,
                         pml->absorb_coeff, request->hc, 0, 0,
                         pml->K_x, pml->a_x, pml->b_x,
                         pml->K_x_half, pml->a_x_half, pml->b_x_half,
                         pml->K_y, pml->a_y, pml->b_y,
                         pml->K_y_half, pml->a_y_half, pml->b_y_half,
                         pml->psi_sxx_x, pml->psi_syy_y,
                         pml->psi_sxy_y, pml->psi_sxy_x, 0);
        exchange_v_PSV(wave->pvx, wave->pvy,
                       mpi->bufferlef_to_rig, mpi->bufferrig_to_lef,
                       mpi->buffertop_to_bot, mpi->bufferbot_to_top,
                       request->req_send, request->req_rec);
        update_s_visc_PML_PSV(1, NX, 1, NY, NX, NY,
                              wave->pvx, wave->pvy, wave->ux, wave->uy,
                              wave->uxy, wave->uyx,
                              wave->psxx, wave->psyy, wave->psxy,
                              mat->ppi, mat->pu, mat->puipjp, mat->prho,
                              request->hc, 0, wave->pr, wave->pp, wave->pq,
                              mat->fipjp, mat->f, mat->g,
                              mat->bip, mat->bjm, mat->cip, mat->cjm,
                              mat->d, mat->e, mat->dip,
                              pml->K_x, pml->a_x, pml->b_x,
                              pml->K_x_half, pml->a_x_half, pml->b_x_half,
                              pml->K_y, pml->a_y, pml->b_y,
                              pml->K_y_half, pml->a_y_half, pml->b_y_half,
                              pml->psi_vxx, pml->psi_vyy,
                              pml->psi_vxy, pml->psi_vyx, 0);
        if (QUELLTYP == 1)
            psource(t, wave->psxx, wave->psyy, acq->srcpos_loc,
                    acq->signals, request->nsrc_loc, 0);
        if (QUELLTYP == 5)
            msource(t, wave->psxx, wave->psyy, wave->psxy,
                    acq->srcpos_loc, acq->signals, request->nsrc_loc, 0);
        exchange_s_PSV(wave->psxx, wave->psyy, wave->psxy,
                       mpi->bufferlef_to_rig, mpi->bufferrig_to_lef,
                       mpi->buffertop_to_bot, mpi->bufferbot_to_top,
                       request->req_send, request->req_rec);
        exact.replayed_steps++;
    }
    exact.recording = 0;
}

static size_t adjoint_cell(int j, int i, int pitch) {
    return (size_t)(j + 2) * pitch + i + 2;
}

/* Transpose of the stress halo-copy operator.  Forward exchange is vertical
 * then horizontal, so this applies horizontal^T before vertical^T.  Ghost
 * cotangents are extracted and cleared; received values are added to owners. */
void exchange_s_adjoint_PSV(double *asxx, double *asyy, double *asxy,
                            int pitch) {
    int fdo = FDORDER / 2 + 1;
    int left = POS[1] > 0 ? INDEX[1] : MPI_PROC_NULL;
    int right = POS[1] < NPROCX - 1 ? INDEX[2] : MPI_PROC_NULL;
    int top = POS[2] > 0 ? INDEX[3] : MPI_PROC_NULL;
    int bottom = POS[2] < NPROCY - 1 ? INDEX[4] : MPI_PROC_NULL;
    int i, j, l, n, horizontal_count, vertical_count;
    MPI_Status status;

    if (NPROCX * NPROCY == 1) return;
    horizontal_count = NY * (2 * fdo - 3);
    vertical_count = NX * (2 * fdo - 1);
    if ((size_t)(horizontal_count > vertical_count ? horizontal_count : vertical_count) >
        exact.communication_capacity)
        err(" Exact-visco stress transpose exchange buffer is too small. ");

    n = 0;
    for (j = 1; j <= NY; ++j) {
        for (l = 1; l < fdo - 1; ++l) {
            size_t p = adjoint_cell(j, NX + l, pitch);
            exact.send_a[n++] = asxy[p]; asxy[p] = 0.0;
        }
        for (l = 1; l < fdo; ++l) {
            size_t p = adjoint_cell(j, NX + l, pitch);
            exact.send_a[n++] = asxx[p]; asxx[p] = 0.0;
        }
    }
    n = 0;
    for (j = 1; j <= NY; ++j) {
        for (l = 1; l < fdo; ++l) {
            size_t p = adjoint_cell(j, 1 - l, pitch);
            exact.send_b[n++] = asxy[p]; asxy[p] = 0.0;
        }
        for (l = 1; l < fdo - 1; ++l) {
            size_t p = adjoint_cell(j, 1 - l, pitch);
            exact.send_b[n++] = asxx[p]; asxx[p] = 0.0;
        }
    }
    memset(exact.recv_a, 0, (size_t)horizontal_count * sizeof(double));
    memset(exact.recv_b, 0, (size_t)horizontal_count * sizeof(double));
    MPI_Sendrecv(exact.send_a, horizontal_count, MPI_DOUBLE, right, 1811,
                 exact.recv_a, horizontal_count, MPI_DOUBLE, left, 1811,
                 SHOT_COMM, &status);
    MPI_Sendrecv(exact.send_b, horizontal_count, MPI_DOUBLE, left, 1812,
                 exact.recv_b, horizontal_count, MPI_DOUBLE, right, 1812,
                 SHOT_COMM, &status);
    n = 0;
    for (j = 1; j <= NY; ++j) {
        for (l = 1; l < fdo - 1; ++l)
            asxy[adjoint_cell(j, l, pitch)] += exact.recv_a[n++];
        for (l = 1; l < fdo; ++l)
            asxx[adjoint_cell(j, l, pitch)] += exact.recv_a[n++];
    }
    n = 0;
    for (j = 1; j <= NY; ++j) {
        for (l = 1; l < fdo; ++l)
            asxy[adjoint_cell(j, NX - l + 1, pitch)] += exact.recv_b[n++];
        for (l = 1; l < fdo - 1; ++l)
            asxx[adjoint_cell(j, NX - l + 1, pitch)] += exact.recv_b[n++];
    }

    n = 0;
    for (i = 1; i <= NX; ++i) {
        for (l = 1; l < fdo; ++l) {
            size_t p = adjoint_cell(NY + l, i, pitch);
            exact.send_a[n++] = asxy[p]; asxy[p] = 0.0;
        }
        for (l = 1; l <= fdo; ++l) {
            size_t p = adjoint_cell(NY + l, i, pitch);
            exact.send_a[n++] = asyy[p]; asyy[p] = 0.0;
        }
    }
    n = 0;
    for (i = 1; i <= NX; ++i) {
        for (l = 1; l <= fdo; ++l) {
            size_t p = adjoint_cell(1 - l, i, pitch);
            exact.send_b[n++] = asxy[p]; asxy[p] = 0.0;
        }
        for (l = 1; l < fdo; ++l) {
            size_t p = adjoint_cell(1 - l, i, pitch);
            exact.send_b[n++] = asyy[p]; asyy[p] = 0.0;
        }
    }
    memset(exact.recv_a, 0, (size_t)vertical_count * sizeof(double));
    memset(exact.recv_b, 0, (size_t)vertical_count * sizeof(double));
    MPI_Sendrecv(exact.send_a, vertical_count, MPI_DOUBLE, bottom, 1813,
                 exact.recv_a, vertical_count, MPI_DOUBLE, top, 1813,
                 SHOT_COMM, &status);
    MPI_Sendrecv(exact.send_b, vertical_count, MPI_DOUBLE, top, 1814,
                 exact.recv_b, vertical_count, MPI_DOUBLE, bottom, 1814,
                 SHOT_COMM, &status);
    n = 0;
    for (i = 1; i <= NX; ++i) {
        for (l = 1; l < fdo; ++l)
            asxy[adjoint_cell(l, i, pitch)] += exact.recv_a[n++];
        for (l = 1; l <= fdo; ++l)
            asyy[adjoint_cell(l, i, pitch)] += exact.recv_a[n++];
    }
    n = 0;
    for (i = 1; i <= NX; ++i) {
        for (l = 1; l <= fdo; ++l)
            asxy[adjoint_cell(NY - l + 1, i, pitch)] += exact.recv_b[n++];
        for (l = 1; l < fdo; ++l)
            asyy[adjoint_cell(NY - l + 1, i, pitch)] += exact.recv_b[n++];
    }
}

/* Transpose of the staggered velocity halo-copy operator, with the same
 * extract/clear/send/add ownership semantics as the stress transpose. */
void exchange_v_adjoint_PSV(double *avx, double *avy, int pitch) {
    int fdo = FDORDER / 2 + 1;
    int left = POS[1] > 0 ? INDEX[1] : MPI_PROC_NULL;
    int right = POS[1] < NPROCX - 1 ? INDEX[2] : MPI_PROC_NULL;
    int top = POS[2] > 0 ? INDEX[3] : MPI_PROC_NULL;
    int bottom = POS[2] < NPROCY - 1 ? INDEX[4] : MPI_PROC_NULL;
    int i, j, l, n, horizontal_count, vertical_count;
    MPI_Status status;

    if (NPROCX * NPROCY == 1) return;
    horizontal_count = NY * (2 * fdo - 3);
    vertical_count = NX * (2 * fdo - 1);
    if ((size_t)(horizontal_count > vertical_count ? horizontal_count : vertical_count) >
        exact.communication_capacity)
        err(" Exact-visco velocity transpose exchange buffer is too small. ");

    n = 0;
    for (j = 1; j <= NY; ++j) {
        for (l = 1; l < fdo; ++l) {
            size_t p = adjoint_cell(j, NX + l, pitch);
            exact.send_a[n++] = avy[p]; avy[p] = 0.0;
        }
        for (l = 1; l < fdo - 1; ++l) {
            size_t p = adjoint_cell(j, NX + l, pitch);
            exact.send_a[n++] = avx[p]; avx[p] = 0.0;
        }
    }
    n = 0;
    for (j = 1; j <= NY; ++j) {
        for (l = 1; l < fdo - 1; ++l) {
            size_t p = adjoint_cell(j, 1 - l, pitch);
            exact.send_b[n++] = avy[p]; avy[p] = 0.0;
        }
        for (l = 1; l < fdo; ++l) {
            size_t p = adjoint_cell(j, 1 - l, pitch);
            exact.send_b[n++] = avx[p]; avx[p] = 0.0;
        }
    }
    memset(exact.recv_a, 0, (size_t)horizontal_count * sizeof(double));
    memset(exact.recv_b, 0, (size_t)horizontal_count * sizeof(double));
    MPI_Sendrecv(exact.send_a, horizontal_count, MPI_DOUBLE, right, 1821,
                 exact.recv_a, horizontal_count, MPI_DOUBLE, left, 1821,
                 SHOT_COMM, &status);
    MPI_Sendrecv(exact.send_b, horizontal_count, MPI_DOUBLE, left, 1822,
                 exact.recv_b, horizontal_count, MPI_DOUBLE, right, 1822,
                 SHOT_COMM, &status);
    n = 0;
    for (j = 1; j <= NY; ++j) {
        for (l = 1; l < fdo; ++l)
            avy[adjoint_cell(j, l, pitch)] += exact.recv_a[n++];
        for (l = 1; l < fdo - 1; ++l)
            avx[adjoint_cell(j, l, pitch)] += exact.recv_a[n++];
    }
    n = 0;
    for (j = 1; j <= NY; ++j) {
        for (l = 1; l < fdo - 1; ++l)
            avy[adjoint_cell(j, NX - l + 1, pitch)] += exact.recv_b[n++];
        for (l = 1; l < fdo; ++l)
            avx[adjoint_cell(j, NX - l + 1, pitch)] += exact.recv_b[n++];
    }

    n = 0;
    for (i = 1; i <= NX; ++i) {
        for (l = 1; l < fdo; ++l) {
            size_t p = adjoint_cell(NY + l, i, pitch);
            exact.send_a[n++] = avy[p]; avy[p] = 0.0;
        }
        for (l = 1; l <= fdo; ++l) {
            size_t p = adjoint_cell(NY + l, i, pitch);
            exact.send_a[n++] = avx[p]; avx[p] = 0.0;
        }
    }
    n = 0;
    for (i = 1; i <= NX; ++i) {
        for (l = 1; l <= fdo; ++l) {
            size_t p = adjoint_cell(1 - l, i, pitch);
            exact.send_b[n++] = avy[p]; avy[p] = 0.0;
        }
        for (l = 1; l < fdo; ++l) {
            size_t p = adjoint_cell(1 - l, i, pitch);
            exact.send_b[n++] = avx[p]; avx[p] = 0.0;
        }
    }
    memset(exact.recv_a, 0, (size_t)vertical_count * sizeof(double));
    memset(exact.recv_b, 0, (size_t)vertical_count * sizeof(double));
    MPI_Sendrecv(exact.send_a, vertical_count, MPI_DOUBLE, bottom, 1823,
                 exact.recv_a, vertical_count, MPI_DOUBLE, top, 1823,
                 SHOT_COMM, &status);
    MPI_Sendrecv(exact.send_b, vertical_count, MPI_DOUBLE, top, 1824,
                 exact.recv_b, vertical_count, MPI_DOUBLE, bottom, 1824,
                 SHOT_COMM, &status);
    n = 0;
    for (i = 1; i <= NX; ++i) {
        for (l = 1; l < fdo; ++l)
            avy[adjoint_cell(l, i, pitch)] += exact.recv_a[n++];
        for (l = 1; l <= fdo; ++l)
            avx[adjoint_cell(l, i, pitch)] += exact.recv_a[n++];
    }
    n = 0;
    for (i = 1; i <= NX; ++i) {
        for (l = 1; l <= fdo; ++l)
            avy[adjoint_cell(NY - l + 1, i, pitch)] += exact.recv_b[n++];
        for (l = 1; l < fdo; ++l)
            avx[adjoint_cell(NY - l + 1, i, pitch)] += exact.recv_b[n++];
    }
}

/* Transpose of psi'=b psi+a D, D'=D/K+psi'.  psi_adj is the
 * adjoint of the NEW psi on entry and the OLD psi on return. */
static double reverse_cpml(double corrected, double *psi_adj, int p,
                           int coordinate, int extent, int axis, float *K,
                           float *a, float *b) {
    int h;
    double combined;
    int at_low_boundary = axis == 1 ? POS[1] == 0 : POS[2] == 0;
    int at_high_boundary = axis == 1 ? POS[1] == NPROCX - 1
                                     : POS[2] == NPROCY - 1;
    if (at_low_boundary && coordinate <= FW) h = coordinate;
    else if (at_high_boundary && coordinate >= extent - FW + 1)
        h = coordinate - extent + 2 * FW;
    else return corrected;
    combined = psi_adj[p] + corrected;
    psi_adj[p] = b[h] * combined;
    return corrected / K[h] + a[h] * combined;
}

static void add_backward_x(double *field, int j, int i, double value,
                           float *hc, double scale) {
    field[cell(j,i)] += scale * hc[1] * value;
    field[cell(j,i-1)] -= scale * hc[1] * value;
    field[cell(j,i+1)] += scale * hc[2] * value;
    field[cell(j,i-2)] -= scale * hc[2] * value;
}
static void add_forward_x(double *field, int j, int i, double value,
                          float *hc, double scale) {
    field[cell(j,i+1)] += scale * hc[1] * value;
    field[cell(j,i)] -= scale * hc[1] * value;
    field[cell(j,i+2)] += scale * hc[2] * value;
    field[cell(j,i-1)] -= scale * hc[2] * value;
}
static void add_backward_y(double *field, int j, int i, double value,
                           float *hc, double scale) {
    field[cell(j,i)] += scale * hc[1] * value;
    field[cell(j-1,i)] -= scale * hc[1] * value;
    field[cell(j+1,i)] += scale * hc[2] * value;
    field[cell(j-2,i)] -= scale * hc[2] * value;
}
static void add_forward_y(double *field, int j, int i, double value,
                          float *hc, double scale) {
    field[cell(j+1,i)] += scale * hc[1] * value;
    field[cell(j,i)] -= scale * hc[1] * value;
    field[cell(j+2,i)] += scale * hc[2] * value;
    field[cell(j-1,i)] -= scale * hc[2] * value;
}

/* Sum constitutive-map contributions that land on the right and bottom
 * model halos into the rank that owns the physical cell.  Horizontal first
 * routes the bottom-right corner into the neighbour's bottom halo; vertical
 * then completes that diagonal transfer. */
static void reduce_physical_gradient_halos(double *physical[5]) {
    int left = POS[1] > 0 ? INDEX[1] : MPI_PROC_NULL;
    int right = POS[1] < NPROCX - 1 ? INDEX[2] : MPI_PROC_NULL;
    int top = POS[2] > 0 ? INDEX[3] : MPI_PROC_NULL;
    int bottom = POS[2] < NPROCY - 1 ? INDEX[4] : MPI_PROC_NULL;
    int f, i, j, n, horizontal_count, vertical_count;
    MPI_Status status;

    if (NPROCX * NPROCY == 1) return;
    horizontal_count = 5 * (NY + 1);
    vertical_count = 5 * NX;
    if ((size_t)(horizontal_count > vertical_count ? horizontal_count : vertical_count) >
        exact.communication_capacity)
        err(" Exact-visco physical-gradient reduction buffer is too small. ");

    n = 0;
    for (f = 0; f < 5; ++f)
        for (j = 1; j <= NY + 1; ++j) {
            size_t p = adjoint_cell(j, NX + 1, exact.pitch);
            exact.send_a[n++] = physical[f][p];
            physical[f][p] = 0.0;
        }
    memset(exact.recv_a, 0, (size_t)horizontal_count * sizeof(double));
    MPI_Sendrecv(exact.send_a, horizontal_count, MPI_DOUBLE, right, 1831,
                 exact.recv_a, horizontal_count, MPI_DOUBLE, left, 1831,
                 SHOT_COMM, &status);
    n = 0;
    for (f = 0; f < 5; ++f)
        for (j = 1; j <= NY + 1; ++j)
            physical[f][adjoint_cell(j, 1, exact.pitch)] += exact.recv_a[n++];

    n = 0;
    for (f = 0; f < 5; ++f)
        for (i = 1; i <= NX; ++i) {
            size_t p = adjoint_cell(NY + 1, i, exact.pitch);
            exact.send_a[n++] = physical[f][p];
            physical[f][p] = 0.0;
        }
    memset(exact.recv_a, 0, (size_t)vertical_count * sizeof(double));
    MPI_Sendrecv(exact.send_a, vertical_count, MPI_DOUBLE, bottom, 1832,
                 exact.recv_a, vertical_count, MPI_DOUBLE, top, 1832,
                 SHOT_COMM, &status);
    n = 0;
    for (f = 0; f < 5; ++f)
        for (i = 1; i <= NX; ++i)
            physical[f][adjoint_cell(1, i, exact.pitch)] += exact.recv_a[n++];
}

static void write_field(const char *suffix, double *gradient) {
    char path[STRING_SIZE + 64], local_path[STRING_SIZE + 80];
    FILE *out;
    int i, j;
    float value;
    snprintf(path, sizeof(path), "%s.raw.%s", JACOBIAN, suffix);
    if (NPROCX * NPROCY == 1) {
        snprintf(local_path, sizeof(local_path), "%s", path);
    } else {
        snprintf(local_path, sizeof(local_path), "%s.%d.%d",
                 path, POS[1], POS[2]);
    }
    out = fopen(local_path, "wb");
    if (!out) err(" Could not open exact visco PSV raw-gradient output. ");
    for (i = 1; i <= NX; i++) for (j = 1; j <= NY; j++) {
        value = (float)gradient[cell(j,i)];
        if (fwrite(&value, sizeof(value), 1, out) != 1)
            err(" Could not write exact visco PSV raw gradient. ");
    }
    fclose(out);
    if (NPROCX * NPROCY > 1) {
        MPI_Barrier(SHOT_COMM);
        if (MYID_SHOT == 0) mergemod(path, 3);
        MPI_Barrier(SHOT_COMM);
        remove(local_path);
    }
}

static size_t checkpoint_formula_bytes(int nx, int ny, int fw) {
    return (8 * (size_t)(nx + 6) * (ny + 6) +
            8 * (size_t)fw * (nx + ny)) * sizeof(float);
}

static void write_segment_report(void) {
    char path[STRING_SIZE + 72];
    FILE *report;
    size_t cells = (size_t)NX * NY;
    size_t legacy_bytes = (size_t)NRECORD * (NT + 1) * cells * sizeof(float);
    size_t checkpoint_each = exact.checkpoint_count ?
        visco_psv_checkpoint_payload_bytes(exact.checkpoint[0]) : 0;
    size_t checkpoint_total = (size_t)exact.checkpoint_count * checkpoint_each;
    size_t segment_bytes = exact.full_storage ? 0 :
        (size_t)NRECORD * exact.max_segment_length * cells * sizeof(float);
    size_t full_bytes = exact.full_storage ? legacy_bytes : 0;
    size_t combined = exact.full_storage ? full_bytes : checkpoint_total + segment_bytes;
    const int large_nt = 5000, large_segments = 32, large_checkpoints = 31;
    const int large_max = (large_nt + large_segments - 1) / large_segments;
    int k;

    if (NPROCX * NPROCY == 1)
        snprintf(path, sizeof(path), "%s.segmented_gradient.json", JACOBIAN);
    else
        snprintf(path, sizeof(path), "%s.segmented_gradient.rank%d.json",
                 JACOBIAN, MYID_SHOT);
    report = fopen(path, "w");
    if (!report) err(" Could not open exact visco PSV segmented-gradient report. ");
    fprintf(report,
            "{\n"
            "  \"storage_mode\": \"%s\",\n"
            "  \"nt\": %d,\n"
            "  \"requested_segments\": %d,\n"
            "  \"segment_count\": %d,\n"
            "  \"segment_count_clamped\": %s,\n"
            "  \"max_segment_length\": %d,\n"
            "  \"checkpoint_count\": %d,\n"
            "  \"checkpoint_bytes_each\": %zu,\n"
            "  \"checkpoint_bytes_total\": %zu,\n"
            "  \"segment_buffer_bytes\": %zu,\n"
            "  \"full_storage_bytes_allocated\": %zu,\n"
            "  \"combined_working_set_bytes\": %zu,\n"
            "  \"legacy_six_field_bytes\": %zu,\n"
            "  \"reduction_factor\": %.17g,\n"
            "  \"replayed_forward_steps\": %d,\n"
            "  \"rank\": %d,\n"
            "  \"position\": [%d, %d],\n"
            "  \"decomposition\": [%d, %d],\n"
            "  \"local_grid\": [%d, %d],\n"
            "  \"global_objective\": %.17g,\n"
            "  \"source_owners\": %d,\n"
            "  \"receiver_ownership_valid\": %s,\n"
            "  \"initial_forward_seconds\": %.17g,\n"
            "  \"forward_replay_seconds\": %.17g,\n"
            "  \"reverse_seconds\": %.17g,\n"
            "  \"segment_boundaries\": [",
            exact.full_storage ? "full_storage_reference" : "segmented",
            NT, exact.requested_segments,
            exact.full_storage ? 1 : exact.segment_count,
            (!exact.full_storage && exact.requested_segments > NT) ? "true" : "false",
            exact.full_storage ? NT : exact.max_segment_length,
            exact.full_storage ? 0 : exact.checkpoint_count,
            exact.full_storage ? 0 : checkpoint_each,
            exact.full_storage ? 0 : checkpoint_total,
            segment_bytes, full_bytes, combined, legacy_bytes,
            combined ? (double)legacy_bytes / combined : 0.0,
            exact.replayed_steps, MYID_SHOT, POS[1], POS[2],
            NPROCX, NPROCY, NX, NY, exact.global_objective,
            exact.source_owners,
            exact.receiver_ownership_valid ? "true" : "false",
            exact.initial_seconds,
            exact.replay_seconds, exact.reverse_seconds);
    if (exact.full_storage) {
        fprintf(report, "[1, %d]", NT);
    } else {
        for (k = 0; k < exact.segment_count; ++k)
            fprintf(report, "%s[%d, %d]", k ? ", " : "",
                    exact.segment_start[k] + 1, exact.segment_end[k]);
    }
    fprintf(report,
            "],\n"
            "  \"representative_grids\": {\n"
            "    \"500x500x5000\": {\"combined_bytes\": %zu, \"legacy_bytes\": %zu, \"combined_gib\": %.17g, \"legacy_gib\": %.17g},\n"
            "    \"1000x1000x5000\": {\"combined_bytes\": %zu, \"legacy_bytes\": %zu, \"combined_gib\": %.17g, \"legacy_gib\": %.17g}\n"
            "  }\n"
            "}\n",
            (size_t)large_checkpoints * checkpoint_formula_bytes(500,500,FW) +
                (size_t)NRECORD * 500 * 500 * large_max * sizeof(float),
            (size_t)NRECORD * 500 * 500 * (large_nt + 1) * sizeof(float),
            ((double)large_checkpoints * checkpoint_formula_bytes(500,500,FW) +
                (double)NRECORD * 500 * 500 * large_max * sizeof(float)) /
                (1024.0 * 1024.0 * 1024.0),
            ((double)NRECORD * 500 * 500 * (large_nt + 1) * sizeof(float)) /
                (1024.0 * 1024.0 * 1024.0),
            (size_t)large_checkpoints * checkpoint_formula_bytes(1000,1000,FW) +
                (size_t)NRECORD * 1000 * 1000 * large_max * sizeof(float),
            (size_t)NRECORD * 1000 * 1000 * (large_nt + 1) * sizeof(float),
            ((double)large_checkpoints * checkpoint_formula_bytes(1000,1000,FW) +
                (double)NRECORD * 1000 * 1000 * large_max * sizeof(float)) /
                (1024.0 * 1024.0 * 1024.0),
            ((double)NRECORD * 1000 * 1000 * (large_nt + 1) * sizeof(float)) /
                (1024.0 * 1024.0 * 1024.0));
    fclose(report);

    if (NPROCX * NPROCY > 1) {
        int checkpoint_min, checkpoint_max, replay_min, replay_max;
        unsigned long long checkpoint_local_bytes =
            (unsigned long long)checkpoint_total;
        unsigned long long checkpoint_min_bytes, checkpoint_max_bytes;
        MPI_Allreduce(&exact.checkpoint_count, &checkpoint_min, 1, MPI_INT,
                      MPI_MIN, SHOT_COMM);
        MPI_Allreduce(&exact.checkpoint_count, &checkpoint_max, 1, MPI_INT,
                      MPI_MAX, SHOT_COMM);
        MPI_Allreduce(&exact.replayed_steps, &replay_min, 1, MPI_INT,
                      MPI_MIN, SHOT_COMM);
        MPI_Allreduce(&exact.replayed_steps, &replay_max, 1, MPI_INT,
                      MPI_MAX, SHOT_COMM);
        MPI_Allreduce(&checkpoint_local_bytes, &checkpoint_min_bytes, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MIN, SHOT_COMM);
        MPI_Allreduce(&checkpoint_local_bytes, &checkpoint_max_bytes, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, SHOT_COMM);
        if (MYID_SHOT == 0) {
            snprintf(path, sizeof(path), "%s.distributed_gradient.json", JACOBIAN);
            report = fopen(path, "w");
            if (!report) err(" Could not open distributed exact-visco gradient report. ");
            fprintf(report,
                    "{\n"
                    "  \"decomposition\": [%d, %d],\n"
                    "  \"global_grid\": [%d, %d],\n"
                    "  \"global_objective\": %.17g,\n"
                    "  \"source_owners\": %d,\n"
                    "  \"receiver_ownership_valid\": %s,\n"
                    "  \"checkpoint_count_min\": %d,\n"
                    "  \"checkpoint_count_max\": %d,\n"
                    "  \"checkpoint_bytes_total_min\": %llu,\n"
                    "  \"checkpoint_bytes_total_max\": %llu,\n"
                    "  \"replayed_forward_steps_min\": %d,\n"
                    "  \"replayed_forward_steps_max\": %d,\n"
                    "  \"global_trajectory_replication\": false\n"
                    "}\n",
                    NPROCX, NPROCY, NXG, NYG, exact.global_objective,
                    exact.source_owners,
                    exact.receiver_ownership_valid ? "true" : "false",
                    checkpoint_min, checkpoint_max,
                    checkpoint_min_bytes, checkpoint_max_bytes,
                    replay_min, replay_max);
            fclose(report);
        }
    }
}

void visco_psv_exact_finish(
        const struct visco_psv_exact_fwi_request *request) {
    struct wavePSV_PML *pml = request->pml;
    struct matPSV *mat = request->material;
    struct fwiPSV *fwi = request->fwi;
    struct seisPSV *seis = request->seis;
    struct seisPSVfwi *data = request->data;
    struct acq *acq = request->acquisition;
    float *hc = request->hc;
    int ntr = request->ntr;
    double *avx, *avy, *asxx, *asyy, *asxy, *ar, *ap, *aq;
    double *psi[NPSI], *native[NNATIVE], *physical[5];
    struct q_tau_mapping mapping;
    int t, i, j, k, p, cx, cy, gi, gj, segment, t_begin, t_end;
    double phase_started;
    double eta, b, c, dt2, div, shear, ax, ay, xx, yy, xy, yx;
    double tr, tp, tq, lambda_r, lambda_p, lambda_q;
    size_t r;
    if (!exact.active) return;
    if (exact.replay_compare)
        err(" Exact visco PSV operand replay was not finalized. ");
    exact.initial_seconds = MPI_Wtime() - exact.initial_started;
    if (!exact.full_storage && exact.checkpoints_captured != exact.checkpoint_count)
        err(" Exact visco PSV forward did not capture every segment checkpoint. ");
    {
        int local_sources = request->nsrc_loc;
        int *receiver_owners = calloc((size_t)request->ntr_glob, sizeof(int));
        int receiver;
        if (!receiver_owners)
            err(" Out of memory validating exact-visco receiver ownership. ");
        MPI_Allreduce(&local_sources, &exact.source_owners, 1, MPI_INT,
                      MPI_SUM, SHOT_COMM);
        MPI_Allreduce(acq->recswitch + 1, receiver_owners,
                      request->ntr_glob, MPI_INT, MPI_SUM, SHOT_COMM);
        exact.receiver_ownership_valid = 1;
        for (receiver = 0; receiver < request->ntr_glob; ++receiver)
            if (receiver_owners[receiver] != 1)
                exact.receiver_ownership_valid = 0;
        free(receiver_owners);
        if (exact.source_owners != 1)
            err(" Exact-visco distributed source ownership is not unique. ");
        if (!exact.receiver_ownership_valid)
            err(" Exact-visco distributed receiver ownership is not unique. ");
    }
    exact.global_objective = request->base_objective;
    avx=calloc(exact.area,sizeof(double)); avy=calloc(exact.area,sizeof(double));
    asxx=calloc(exact.area,sizeof(double)); asyy=calloc(exact.area,sizeof(double));
    asxy=calloc(exact.area,sizeof(double)); ar=calloc(exact.area,sizeof(double));
    ap=calloc(exact.area,sizeof(double)); aq=calloc(exact.area,sizeof(double));
    if (!avx || !avy || !asxx || !asyy || !asxy || !ar || !ap || !aq)
        err(" Out of memory for exact visco PSV adjoint. ");
    for (k=0;k<NPSI;k++) {
        psi[k]=calloc(exact.area,sizeof(double));
        if (!psi[k]) err(" Out of memory for exact visco PSV CPML adjoint. ");
    }
    for (k=0;k<NNATIVE;k++) {
        native[k]=calloc(exact.area,sizeof(double));
        if (!native[k]) err(" Out of memory for exact visco PSV material adjoint. ");
    }
    for (k=0;k<5;k++) {
        physical[k]=calloc(exact.area,sizeof(double));
        if (!physical[k]) err(" Out of memory for exact visco PSV physical gradient. ");
    }
    eta=mat->peta[1]; b=mat->bjm[1]; c=mat->cjm[1]; dt2=DT*0.5;
    for (segment = exact.full_storage ? 0 : exact.segment_count - 1;
         segment >= 0; --segment) {
        if (exact.full_storage) {
            t_begin = 1;
            t_end = NT;
        } else {
            phase_started = MPI_Wtime();
            replay_forward_segment(request, segment);
            exact.replay_seconds += MPI_Wtime() - phase_started;
            t_begin = exact.segment_start[segment] + 1;
            t_end = exact.segment_end[segment];
        }
        phase_started = MPI_Wtime();
        for (t=t_end;t>=t_begin;t--) {
        /* Receiver samples are recorded after stress update; velocity is
         * unchanged there. Production sample one is excluded from L2. */
        if (t>1) for (k=1;k<=ntr;k++) {
            i=acq->recpos_loc[1][k]; j=acq->recpos_loc[2][k]; p=cell(j,i);
            avx[p] += (double)seis->sectionvx[k][t]-data->sectionvxdata[k][t];
            avy[p] += (double)seis->sectionvy[k][t]-data->sectionvydata[k][t];
        }
        exchange_s_adjoint_PSV(asxx, asyy, asxy, exact.pitch);
        /* Transpose stress and all three relaxation-memory recurrences. */
        for (j=1;j<=NY;j++) for (i=1;i<=NX;i++) {
            p=cell(j,i); r=record_index(t,j,i);
            xx=exact.record[VXX][r]; yx=exact.record[VYX][r];
            xy=exact.record[VXY][r]; yy=exact.record[VYY][r];
            div=xx+yy; shear=xy+yx;
            lambda_r=ar[p]+dt2*asxy[p];
            lambda_p=ap[p]+dt2*asxx[p];
            lambda_q=aq[p]+dt2*asyy[p];
            native[GFC][p]+=asxy[p]*shear;
            native[GF][p]+=-2.0*(asxx[p]*yy+asyy[p]*xx);
            native[GG][p]+=(asxx[p]+asyy[p])*div;
            native[GDC][p]+=-b*lambda_r*shear;
            native[GD][p]+=2.0*b*(lambda_p*yy+lambda_q*xx);
            native[GE][p]+=-b*(lambda_p+lambda_q)*div;
            ax=asxx[p]*mat->g[j][i]+asyy[p]*(mat->g[j][i]-2.0*mat->f[j][i]);
            ay=asyy[p]*mat->g[j][i]+asxx[p]*(mat->g[j][i]-2.0*mat->f[j][i]);
            ax+=b*(-mat->e[j][i][1]*(lambda_p+lambda_q)+2.0*mat->d[j][i][1]*lambda_q);
            ay+=b*(-mat->e[j][i][1]*(lambda_p+lambda_q)+2.0*mat->d[j][i][1]*lambda_p);
            xy=yx=asxy[p]*mat->fipjp[j][i]-b*lambda_r*mat->dip[j][i][1];
            ar[p]=dt2*asxy[p]+b*c*lambda_r;
            ap[p]=dt2*asxx[p]+b*c*lambda_p;
            aq[p]=dt2*asyy[p]+b*c*lambda_q;
            xx=reverse_cpml(ax,psi[PVXX],p,i,NX,1,pml->K_x,pml->a_x,pml->b_x);
            yx=reverse_cpml(yx,psi[PVYX],p,i,NX,1,pml->K_x_half,pml->a_x_half,pml->b_x_half);
            xy=reverse_cpml(xy,psi[PVXY],p,j,NY,2,pml->K_y_half,pml->a_y_half,pml->b_y_half);
            yy=reverse_cpml(ay,psi[PVYY],p,j,NY,2,pml->K_y,pml->a_y,pml->b_y);
            add_backward_x(avx,j,i,xx,hc,1.0/DH);
            add_forward_x(avy,j,i,yx,hc,1.0/DH);
            add_forward_y(avx,j,i,xy,hc,1.0/DH);
            add_backward_y(avy,j,i,yy,hc,1.0/DH);
        }
        exchange_v_adjoint_PSV(avx, avy, exact.pitch);
        /* Transpose velocity and its four independent CPML recurrences. */
        for (j=1;j<=NY;j++) for (i=1;i<=NX;i++) {
            p=cell(j,i); r=record_index(t,j,i);
            native[GRX][p]+=avx[p]*DT*exact.record[FX][r]/DH;
            native[GRY][p]+=avy[p]*DT*exact.record[FY][r]/DH;
            ax=avx[p]*DT*mat->prip[j][i]/DH;
            ay=avy[p]*DT*mat->prjp[j][i]/DH;
            xx=reverse_cpml(ax,psi[PSXX],p,i,NX,1,pml->K_x_half,pml->a_x_half,pml->b_x_half);
            yx=reverse_cpml(ay,psi[PSXYX],p,i,NX,1,pml->K_x,pml->a_x,pml->b_x);
            xy=reverse_cpml(ax,psi[PSXYY],p,j,NY,2,pml->K_y,pml->a_y,pml->b_y);
            yy=reverse_cpml(ay,psi[PSYY],p,j,NY,2,pml->K_y_half,pml->a_y_half,pml->b_y_half);
            add_forward_x(asxx,j,i,xx,hc,1.0);
            add_backward_x(asxy,j,i,yx,hc,1.0);
            add_backward_y(asxy,j,i,xy,hc,1.0);
            add_forward_y(asyy,j,i,yy,hc,1.0);
        }
        }
        exact.reverse_seconds += MPI_Wtime() - phase_started;
        if (exact.full_storage) break;
    }
    exact.active = 0;
    init_q_tau_mapping(&mapping,Q_PARAMETERIZATION_MODE,L,FL,
                       Q_APPROX_FMIN,Q_APPROX_FMAX,Q_APPROX_DF);
    /* Local constitutive map: native f,g,d,e and corner f,dip. */
    for (j=1;j<=NY;j++) for (i=1;i<=NX;i++) {
        double rho=mat->prho[j][i], vp=mat->ppi[j][i], vs=mat->pu[j][i];
        double M=rho*vs*vs, P=rho*vp*vp, ts=mat->ptaus[j][i];
        double tp0=mat->ptaup[j][i], den_s=1.0+0.5*ts, den_p=1.0+0.5*tp0;
        double gM, gP, gts, gtp, H, T, den_c, gH, gT;
        p=cell(j,i);
        gi=POS[1]*NX+i;
        gj=POS[2]*NY+j;
        gM=native[GF][p]*DT*(1.0+ts)/den_s+
           native[GD][p]*eta*ts/den_s;
        gP=native[GG][p]*DT*(1.0+tp0)/den_p+
           native[GE][p]*eta*tp0/den_p;
        gts=native[GF][p]*DT*M*0.5/(den_s*den_s)+
            native[GD][p]*eta*M/(den_s*den_s);
        gtp=native[GG][p]*DT*P*0.5/(den_p*den_p)+
            native[GE][p]*eta*P/(den_p*den_p);
        physical[0][p]+=gP*2.0*rho*vp;
        physical[1][p]+=gM*2.0*rho*vs;
        physical[2][p]+=gP*vp*vp+gM*vs*vs;
        physical[3][p]+=gtp;
        physical[4][p]+=gts;
        H=mat->puipjp[j][i]; T=mat->ptausipjp[j][i];
        den_c=1.0+0.5*T;
        gH=native[GFC][p]*DT*(1.0+T)/den_c+
           native[GDC][p]*eta*T/den_c;
        gT=native[GFC][p]*DT*H*0.5/(den_c*den_c)+
            native[GDC][p]*eta*H/(den_c*den_c);
        for (cy=j;cy<=j+1;cy++) for (cx=i;cx<=i+1;cx++) {
            if (gj+(cy-j)>NYG || gi+(cx-i)>NXG) continue;
            {
                double local_rho=mat->prho[cy][cx], local_vs=mat->pu[cy][cx];
                double local_M=local_rho*local_vs*local_vs;
                double weight=gH*H*H/(4.0*local_M*local_M);
                int q=cell(cy,cx);
                physical[1][q]+=weight*2.0*local_rho*local_vs;
                physical[2][q]+=weight*local_vs*local_vs;
                physical[4][q]+=0.25*gT;
            }
        }
        /* R_x and R_y are reciprocal arithmetic face densities. */
        physical[2][p]+=-0.5*mat->prip[j][i]*mat->prip[j][i]*native[GRX][p];
        physical[2][p]+=-0.5*mat->prjp[j][i]*mat->prjp[j][i]*native[GRY][p];
        if (gi<NXG) physical[2][cell(j,i+1)]+=
            -0.5*mat->prip[j][i]*mat->prip[j][i]*native[GRX][p];
        if (gj<NYG) physical[2][cell(j+1,i)]+=
            -0.5*mat->prjp[j][i]*mat->prjp[j][i]*native[GRY][p];
    }
    reduce_physical_gradient_halos(physical);
    for (j=1;j<=NY;j++) for (i=1;i<=NX;i++) {
        double qp,qs;
        p=cell(j,i);
        if (mapping.mode == Q_PARAMETERIZATION_LEGACY) {
            qp=2.0/mat->ptaup[j][i]; qs=2.0/mat->ptaus[j][i];
        } else {
            qp=(1.0/mat->ptaup[j][i]-mapping.inverse_tau_offset)/mapping.inverse_tau_per_q;
            qs=(1.0/mat->ptaus[j][i]-mapping.inverse_tau_offset)/mapping.inverse_tau_per_q;
        }
        physical[3][p]*=q_to_tau_derivative((float)qp,&mapping);
        physical[4][p]*=q_to_tau_derivative((float)qs,&mapping);
    }
    for (j=1;j<=NY;j++) for (i=1;i<=NX;i++) {
        p=cell(j,i);
        fwi->waveconv[j][i]+=(float)physical[0][p];
        fwi->waveconv_u[j][i]+=(float)physical[1][p];
        fwi->waveconv_rho[j][i]+=(float)physical[2][p];
        fwi->waveconv_qp_exact[j][i]+=(float)physical[3][p];
        fwi->waveconv_qs_exact[j][i]+=(float)physical[4][p];
    }
    {
        const char *flag=getenv("DENISE_PSV_EXACT_VISCO_GRADIENT");
        if (flag && flag[0]=='1' && flag[1]=='\0') {
            write_field("vp",physical[0]); write_field("vs",physical[1]);
            write_field("rho",physical[2]); write_field("qp",physical[3]);
            write_field("qs",physical[4]);
        }
    }
    write_segment_report();
    free(exact.record_payload); exact.record_payload=NULL;
    for (k=0;k<NRECORD;k++) exact.record[k]=NULL;
    for (k=0;k<exact.checkpoint_count;k++)
        visco_psv_checkpoint_destroy(exact.checkpoint[k]);
    free(exact.checkpoint); exact.checkpoint=NULL;
    free(exact.segment_start); exact.segment_start=NULL;
    free(exact.segment_end); exact.segment_end=NULL;
    free(exact.send_a); exact.send_a=NULL;
    free(exact.send_b); exact.send_b=NULL;
    free(exact.recv_a); exact.recv_a=NULL;
    free(exact.recv_b); exact.recv_b=NULL;
    exact.communication_capacity=0;
    for (k=0;k<NPSI;k++) free(psi[k]);
    for (k=0;k<NNATIVE;k++) free(native[k]);
    for (k=0;k<5;k++) free(physical[k]);
    free(avx); free(avy); free(asxx); free(asyy); free(asxy);
    free(ar); free(ap); free(aq);
}
