/* In-memory restart state for the frozen exact viscoelastic P/SV envelope.
 *
 * A timestep-t checkpoint is captured after the complete production forward
 * timestep: velocity update/exchange, stress/GSLS/CPML update, source
 * injection, stress exchange, receiver sampling, and trajectory recording.
 * Restoring it therefore resumes by executing timestep t+1.
 */
#include "fd.h"

extern int NX, NY, NT, FW, FDORDER, L, MODE, INVMAT1, GRAD_FORM;
extern int NPROCX, NPROCY, FREE_SURF, BOUNDARY, NDT, DTINV, LNORM;
extern int READMOD, NSRC;

enum { VX, VY, SXX, SYY, SXY, R, P, Q,
       PSI_SXX_X, PSI_SXY_X, PSI_VXX, PSI_VYX,
       PSI_SYY_Y, PSI_SXY_Y, PSI_VYY, PSI_VXY, CHECKPOINT_FIELD_COUNT };

struct visco_psv_checkpoint {
    unsigned int layout_version;
    int timestep, nx, ny, fw, halo, mechanisms;
    size_t full_count, x_cpml_count, y_cpml_count, payload_count;
    float *payload;
    float *field[CHECKPOINT_FIELD_COUNT];
};

static void require_supported(void) {
    if (MODE != 1 || L != 1 || INVMAT1 != 1 || GRAD_FORM != 2 ||
        FDORDER != 4 || NPROCX != 1 || NPROCY != 1 || FREE_SURF ||
        BOUNDARY || FW <= 0 || NDT != 1 || DTINV != 1 || LNORM != 2 ||
        READMOD != 1 || NSRC != 1)
        err(" Exact visco PSV checkpointing supports MODE=1, one-rank/one-source L=1 FD4, INVMAT1=1, GRAD_FORM=2, NDT=DTINV=1, LNORM=2, READMOD=1, CPML interior only. ");
}

static void require_layout(const struct visco_psv_checkpoint *checkpoint) {
    if (!checkpoint) err(" Null exact visco PSV checkpoint. ");
    require_supported();
    if (checkpoint->layout_version != 1U || checkpoint->nx != NX ||
        checkpoint->ny != NY || checkpoint->fw != FW ||
        checkpoint->halo != 3 || checkpoint->mechanisms != L)
        err(" Incompatible exact visco PSV checkpoint layout. ");
}

static void copy_matrix_to_flat(float *target, float **source,
                                int j0, int j1, int i0, int i1) {
    int i, j;
    size_t p = 0;
    for (j = j0; j <= j1; ++j)
        for (i = i0; i <= i1; ++i) target[p++] = source[j][i];
}

static void copy_flat_to_matrix(float **target, const float *source,
                                int j0, int j1, int i0, int i1) {
    int i, j;
    size_t p = 0;
    for (j = j0; j <= j1; ++j)
        for (i = i0; i <= i1; ++i) target[j][i] = source[p++];
}

static void copy_tensor_to_flat(float *target, float ***source,
                                int j0, int j1, int i0, int i1) {
    int i, j;
    size_t p = 0;
    for (j = j0; j <= j1; ++j)
        for (i = i0; i <= i1; ++i) target[p++] = source[j][i][1];
}

static void copy_flat_to_tensor(float ***target, const float *source,
                                int j0, int j1, int i0, int i1) {
    int i, j;
    size_t p = 0;
    for (j = j0; j <= j1; ++j)
        for (i = i0; i <= i1; ++i) target[j][i][1] = source[p++];
}

struct visco_psv_checkpoint *visco_psv_checkpoint_create(void) {
    struct visco_psv_checkpoint *checkpoint;
    size_t cursor;
    int k;
    require_supported();
    checkpoint = calloc(1, sizeof(*checkpoint));
    if (!checkpoint) err(" Out of memory allocating exact visco PSV checkpoint. ");
    checkpoint->layout_version = 1U;
    checkpoint->nx = NX; checkpoint->ny = NY; checkpoint->fw = FW;
    checkpoint->halo = FDORDER / 2 + 1; checkpoint->mechanisms = L;
    checkpoint->full_count = (size_t)(NX + 6) * (NY + 6);
    checkpoint->x_cpml_count = (size_t)NY * 2 * FW;
    checkpoint->y_cpml_count = (size_t)NX * 2 * FW;
    checkpoint->payload_count = 8 * checkpoint->full_count +
        4 * checkpoint->x_cpml_count + 4 * checkpoint->y_cpml_count;
    checkpoint->payload = malloc(checkpoint->payload_count * sizeof(float));
    if (!checkpoint->payload)
        err(" Out of memory allocating exact visco PSV checkpoint payload. ");
    cursor = 0;
    for (k = VX; k <= Q; ++k) {
        checkpoint->field[k] = checkpoint->payload + cursor;
        cursor += checkpoint->full_count;
    }
    for (k = PSI_SXX_X; k <= PSI_VYX; ++k) {
        checkpoint->field[k] = checkpoint->payload + cursor;
        cursor += checkpoint->x_cpml_count;
    }
    for (k = PSI_SYY_Y; k <= PSI_VXY; ++k) {
        checkpoint->field[k] = checkpoint->payload + cursor;
        cursor += checkpoint->y_cpml_count;
    }
    if (cursor != checkpoint->payload_count)
        err(" Internal exact visco PSV checkpoint layout error. ");
    return checkpoint;
}

void visco_psv_checkpoint_capture(struct visco_psv_checkpoint *checkpoint,
                                  const struct wavePSV *wave,
                                  const struct wavePSV_PML *pml,
                                  int timestep) {
    int lo = -2, jhi = NY + 3, ihi = NX + 3;
    require_layout(checkpoint);
    if (!wave || !pml || timestep < 0 || timestep > NT)
        err(" Invalid exact visco PSV checkpoint capture. ");
    copy_matrix_to_flat(checkpoint->field[VX], wave->pvx, lo, jhi, lo, ihi);
    copy_matrix_to_flat(checkpoint->field[VY], wave->pvy, lo, jhi, lo, ihi);
    copy_matrix_to_flat(checkpoint->field[SXX], wave->psxx, lo, jhi, lo, ihi);
    copy_matrix_to_flat(checkpoint->field[SYY], wave->psyy, lo, jhi, lo, ihi);
    copy_matrix_to_flat(checkpoint->field[SXY], wave->psxy, lo, jhi, lo, ihi);
    copy_tensor_to_flat(checkpoint->field[R], wave->pr, lo, jhi, lo, ihi);
    copy_tensor_to_flat(checkpoint->field[P], wave->pp, lo, jhi, lo, ihi);
    copy_tensor_to_flat(checkpoint->field[Q], wave->pq, lo, jhi, lo, ihi);
    copy_matrix_to_flat(checkpoint->field[PSI_SXX_X], pml->psi_sxx_x, 1, NY, 1, 2*FW);
    copy_matrix_to_flat(checkpoint->field[PSI_SXY_X], pml->psi_sxy_x, 1, NY, 1, 2*FW);
    copy_matrix_to_flat(checkpoint->field[PSI_VXX], pml->psi_vxx, 1, NY, 1, 2*FW);
    copy_matrix_to_flat(checkpoint->field[PSI_VYX], pml->psi_vyx, 1, NY, 1, 2*FW);
    copy_matrix_to_flat(checkpoint->field[PSI_SYY_Y], pml->psi_syy_y, 1, 2*FW, 1, NX);
    copy_matrix_to_flat(checkpoint->field[PSI_SXY_Y], pml->psi_sxy_y, 1, 2*FW, 1, NX);
    copy_matrix_to_flat(checkpoint->field[PSI_VYY], pml->psi_vyy, 1, 2*FW, 1, NX);
    copy_matrix_to_flat(checkpoint->field[PSI_VXY], pml->psi_vxy, 1, 2*FW, 1, NX);
    checkpoint->timestep = timestep;
}

void visco_psv_checkpoint_restore(const struct visco_psv_checkpoint *checkpoint,
                                  struct wavePSV *wave,
                                  struct wavePSV_PML *pml) {
    int lo = -2, jhi = NY + 3, ihi = NX + 3;
    require_layout(checkpoint);
    if (!wave || !pml) err(" Invalid exact visco PSV checkpoint restore. ");
    copy_flat_to_matrix(wave->pvx, checkpoint->field[VX], lo, jhi, lo, ihi);
    copy_flat_to_matrix(wave->pvy, checkpoint->field[VY], lo, jhi, lo, ihi);
    copy_flat_to_matrix(wave->psxx, checkpoint->field[SXX], lo, jhi, lo, ihi);
    copy_flat_to_matrix(wave->psyy, checkpoint->field[SYY], lo, jhi, lo, ihi);
    copy_flat_to_matrix(wave->psxy, checkpoint->field[SXY], lo, jhi, lo, ihi);
    copy_flat_to_tensor(wave->pr, checkpoint->field[R], lo, jhi, lo, ihi);
    copy_flat_to_tensor(wave->pp, checkpoint->field[P], lo, jhi, lo, ihi);
    copy_flat_to_tensor(wave->pq, checkpoint->field[Q], lo, jhi, lo, ihi);
    copy_flat_to_matrix(pml->psi_sxx_x, checkpoint->field[PSI_SXX_X], 1, NY, 1, 2*FW);
    copy_flat_to_matrix(pml->psi_sxy_x, checkpoint->field[PSI_SXY_X], 1, NY, 1, 2*FW);
    copy_flat_to_matrix(pml->psi_vxx, checkpoint->field[PSI_VXX], 1, NY, 1, 2*FW);
    copy_flat_to_matrix(pml->psi_vyx, checkpoint->field[PSI_VYX], 1, NY, 1, 2*FW);
    copy_flat_to_matrix(pml->psi_syy_y, checkpoint->field[PSI_SYY_Y], 1, 2*FW, 1, NX);
    copy_flat_to_matrix(pml->psi_sxy_y, checkpoint->field[PSI_SXY_Y], 1, 2*FW, 1, NX);
    copy_flat_to_matrix(pml->psi_vyy, checkpoint->field[PSI_VYY], 1, 2*FW, 1, NX);
    copy_flat_to_matrix(pml->psi_vxy, checkpoint->field[PSI_VXY], 1, 2*FW, 1, NX);
}

void visco_psv_checkpoint_destroy(struct visco_psv_checkpoint *checkpoint) {
    if (!checkpoint) return;
    free(checkpoint->payload);
    free(checkpoint);
}

size_t visco_psv_checkpoint_payload_bytes(const struct visco_psv_checkpoint *checkpoint) {
    if (!checkpoint) return 0;
    return checkpoint->payload_count * sizeof(float);
}

int visco_psv_checkpoint_timestep(const struct visco_psv_checkpoint *checkpoint) {
    if (!checkpoint) return -1;
    return checkpoint->timestep;
}

static unsigned int compare_payloads(const struct visco_psv_checkpoint *left,
                                     const struct visco_psv_checkpoint *right) {
    unsigned int mask = 0;
    int k;
    for (k = 0; k < CHECKPOINT_FIELD_COUNT; ++k) {
        size_t count = k <= Q ? left->full_count :
                       k <= PSI_VYX ? left->x_cpml_count : left->y_cpml_count;
        if (memcmp(left->field[k], right->field[k], count*sizeof(float)) != 0)
            mask |= 1U << k;
    }
    return mask;
}

int visco_psv_checkpoint_equal(const struct visco_psv_checkpoint *left,
                               const struct visco_psv_checkpoint *right,
                               unsigned int *field_mismatch_mask) {
    unsigned int mask;
    require_layout(left); require_layout(right);
    if (left->layout_version != right->layout_version || left->nx != right->nx ||
        left->ny != right->ny || left->fw != right->fw ||
        left->payload_count != right->payload_count) {
        if (field_mismatch_mask) *field_mismatch_mask = ~0U;
        return 0;
    }
    mask = compare_payloads(left, right);
    if (left->timestep != right->timestep) mask |= 1U << 31;
    if (field_mismatch_mask) *field_mismatch_mask = mask;
    return mask == 0;
}

int visco_psv_checkpoint_matches_live(const struct visco_psv_checkpoint *checkpoint,
                                      const struct wavePSV *wave,
                                      const struct wavePSV_PML *pml,
                                      unsigned int *field_mismatch_mask) {
    struct visco_psv_checkpoint *live = visco_psv_checkpoint_create();
    unsigned int mask;
    int equal;
    visco_psv_checkpoint_capture(live, wave, pml, checkpoint->timestep);
    equal = visco_psv_checkpoint_equal(checkpoint, live, &mask);
    visco_psv_checkpoint_destroy(live);
    if (field_mismatch_mask) *field_mismatch_mask = mask;
    return equal;
}
