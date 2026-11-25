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

/* ---- K-5: segmented storage with checkpoint/replay ------------------------
 * docs/sh_k5_checkpoint_replay_contract.md.  One segmented trajectory at a
 * time per process (the exact SH gradient runs one shot at a time); the
 * state is reset completely when the trajectory is released. */
static struct {
    int enabled, verify, segments, nsteps, maxlen, loaded, replaying;
    int *first;                 /* first[k]: first step of segment k; first[segments] = nsteps */
    struct visco_sh_material_observable_step *pool;   /* records of the loaded segment */
    struct visco_sh_material_observable_step *full;   /* verify: complete history */
    struct visco_sh_material_observable_trajectory *trajectory;
    float **checkpoint;         /* checkpoint[k]: forward state before step first[k], k >= 1 */
    size_t checkpoint_floats;
    void (*save)(void *context, float *payload);
    void (*restore)(void *context, const float *payload);
    void (*run)(void *context, int first_step, int last_step);
    void *context;
    size_t replayed_steps, compared, mismatches;
    double replay_seconds;
} seg;
static int active_index = -1;
/* segment request for every following trajectory_init: 0 = full storage,
 * the state of every caller that never requests segments */
static int requested_segments = 0;
static size_t requested_checkpoint_floats = 0;

/* The functions below use only their arguments (no parameter globals), so
 * every harness that links this module links it unchanged. */

/* floats of one forward-state checkpoint of the SH forward (the copy in
 * sh_visc.c): vz, sxz, syz with halos, r and q (l mechanisms each), and the
 * four CPML memory strips of width fw */
size_t visco_sh_material_observable_replay_floats(int nx, int ny, int fdorder,
                                                  int fw, int l) {
    size_t nd = (size_t)(fdorder / 2 + 1);
    size_t field = ((size_t)ny + 2 * nd) * ((size_t)nx + 2 * nd);
    return field * (3 + 2 * (size_t)l) + 4 * (size_t)fw * ((size_t)nx + (size_t)ny);
}

/* effective number of segments for nsteps records of 3 (nx+1)(ny+1) floats:
 * requested 0 = full storage (SH_EXACT_SEGMENTS = 0, the default); > 0:
 * min(requested, nsteps); -1: automatic,
 * S = round(sqrt(nsteps * records / checkpoint floats)), clipped to 1..nsteps,
 * which minimises records of one segment plus S - 1 checkpoints. */
int visco_sh_material_observable_segments(int requested, int nx, int ny, int nsteps,
                                          size_t checkpoint_floats) {
    double records, s;
    if ((requested == 0) || (nsteps < 1)) return 0;
    if (requested > 0)
        return (requested < nsteps) ? requested : nsteps;
    records = 3.0 * ((double)nx + 1.0) * ((double)ny + 1.0);
    s = floor(sqrt((double)nsteps * records / (double)checkpoint_floats) + 0.5);
    if (s < 1.0) s = 1.0;
    if (s > (double)nsteps) s = (double)nsteps;
    return (int)s;
}

/* bytes of the exact history of one process (computed in double, no int
 * overflow): full storage, or one segment of records plus S - 1 checkpoints */
double visco_sh_material_observable_history_bytes(int nx, int ny, int nsteps,
                                                  int segments, size_t checkpoint_floats) {
    double record = 3.0 * ((double)nx + 1.0) * ((double)ny + 1.0) * sizeof(float);
    double maxlen;
    if (segments < 1) return (double)nsteps * record;
    maxlen = ceil((double)nsteps / (double)segments);
    return maxlen * record + ((double)segments - 1.0) *
           ((double)checkpoint_floats + 1.0) * sizeof(float);
}

/* segmented storage for the following trajectory_init calls: an effective
 * segment count (visco_sh_material_observable_segments) and the checkpoint
 * size of the propagator that will replay it; segments < 1 = full storage */
void visco_sh_material_observable_request_segments(int segments,
                                                   size_t checkpoint_floats) {
    requested_segments = (segments > 0) ? segments : 0;
    requested_checkpoint_floats = checkpoint_floats;
}

int visco_sh_material_observable_is_segmented(
        const struct visco_sh_material_observable_trajectory *trajectory) {
    return seg.enabled && (trajectory == seg.trajectory);
}

static int seg_of(int step) {
    int k = (int)(((long long)step * seg.segments) / seg.nsteps);
    while (seg.first[k] > step) --k;
    while (seg.first[k + 1] <= step) ++k;
    return k;
}

static void seg_map(int k) {
    static const struct visco_sh_material_observable_step none = {NULL, NULL, NULL};
    int n, m;
    if (seg.loaded >= 0)
        for (n = seg.first[seg.loaded]; n < seg.first[seg.loaded + 1]; ++n)
            seg.trajectory->steps[n] = none;
    for (n = seg.first[k], m = 0; n < seg.first[k + 1]; ++n, ++m)
        seg.trajectory->steps[n] = seg.pool[m];
    seg.loaded = k;
}

static void seg_release_steps(struct visco_sh_material_observable_step *steps,
                              int count) {
    int n;
    if (steps == NULL) return;
    for (n = 0; n < count; ++n) {
        free_owned_matrix(steps[n].qsum);
        free_owned_matrix(steps[n].strain_x);
        free_owned_matrix(steps[n].strain_y);
    }
    free(steps);
}

static struct visco_sh_material_observable_step *seg_allocate_steps(
        int count, int nx, int ny) {
    struct visco_sh_material_observable_step *steps;
    int n;
    steps = (struct visco_sh_material_observable_step *)calloc(
            (size_t)count, sizeof(*steps));
    if (steps == NULL) return NULL;
    for (n = 0; n < count; ++n) {
        steps[n].qsum = owned_matrix(nx, ny);
        steps[n].strain_x = owned_matrix(nx, ny);
        steps[n].strain_y = owned_matrix(nx, ny);
        if ((steps[n].qsum == NULL) || (steps[n].strain_x == NULL) ||
                (steps[n].strain_y == NULL)) {
            seg_release_steps(steps, count);
            return NULL;
        }
    }
    return steps;
}

static void seg_release(struct visco_sh_material_observable_trajectory *trajectory) {
    extern int MYID;
    int k;
    double field = 4.0 * ((double)trajectory->nx + 1) * ((double)trajectory->ny + 1);
    if (MYID == 0)
        printf("Exact SH replay: segments %d steps %d max_segment %d "
               "record_MB %.3f checkpoint_MB %.3f full_record_MB %.3f "
               "replayed_steps %zu replay_seconds %.3f verify %d compared %zu "
               "mismatches %zu\n",
               seg.segments, seg.nsteps, seg.maxlen,
               3.0 * seg.maxlen * field / 1048576.0,
               4.0 * (seg.segments - 1) * (double)seg.checkpoint_floats / 1048576.0,
               3.0 * seg.nsteps * field / 1048576.0,
               seg.replayed_steps, seg.replay_seconds, seg.verify,
               seg.compared, seg.mismatches);
    if (seg.checkpoint != NULL)
        for (k = 0; k < seg.segments; ++k) free(seg.checkpoint[k]);
    free(seg.checkpoint);
    seg_release_steps(seg.pool, seg.maxlen);
    seg_release_steps(seg.full, seg.verify ? seg.nsteps : 0);
    free(seg.first);
    free(trajectory->steps);
    memset(trajectory, 0, sizeof(*trajectory));
    memset(&seg, 0, sizeof(seg));
}

static int seg_init(struct visco_sh_material_observable_trajectory *trajectory,
                    int nx, int ny, int nsteps, int segments) {
    const char *verify = getenv("DENISE_SH_EXACT_SEGMENTS_VERIFY");
    int k;
    if (seg.enabled) return -1;                 /* one trajectory at a time */
    memset(&seg, 0, sizeof(seg));
    memset(trajectory, 0, sizeof(*trajectory));
    trajectory->nx = nx; trajectory->ny = ny;
    trajectory->nsteps = nsteps; trajectory->dtinv = 1;
    seg.enabled = 1; seg.trajectory = trajectory; seg.loaded = -1;
    seg.segments = segments; seg.nsteps = nsteps;
    seg.verify = (verify != NULL) && (verify[0] == '1');
    trajectory->steps = (struct visco_sh_material_observable_step *)calloc(
            (size_t)nsteps, sizeof(*trajectory->steps));
    seg.first = (int *)malloc(((size_t)segments + 1) * sizeof(int));
    seg.checkpoint = (float **)calloc((size_t)segments, sizeof(float *));
    if ((trajectory->steps == NULL) || (seg.first == NULL) ||
            (seg.checkpoint == NULL)) { seg_release(trajectory); return -3; }
    for (k = 0; k <= segments; ++k)
        seg.first[k] = (int)(((long long)k * nsteps) / segments);
    for (k = 0; k < segments; ++k)
        if (seg.first[k + 1] - seg.first[k] > seg.maxlen)
            seg.maxlen = seg.first[k + 1] - seg.first[k];
    seg.pool = seg_allocate_steps(seg.maxlen, nx, ny);
    if (seg.verify) seg.full = seg_allocate_steps(nsteps, nx, ny);
    if ((seg.pool == NULL) || (seg.verify && (seg.full == NULL))) {
        seg_release(trajectory);
        return -3;
    }
    /* all checkpoints before the forward run: a shortage fails here, at the
     * collectively checked trajectory allocation, never during the run */
    seg.checkpoint_floats = requested_checkpoint_floats;
    for (k = 1; k < segments; ++k) {
        seg.checkpoint[k] = (float *)malloc((seg.checkpoint_floats + 1) * sizeof(float));
        if (seg.checkpoint[k] == NULL) {
            seg_release(trajectory);
            return -3;
        }
    }
    return 0;
}

int visco_sh_material_observable_set_replay(
        struct visco_sh_material_observable_trajectory *trajectory,
        void (*save)(void *context, float *payload),
        void (*restore)(void *context, const float *payload),
        void (*run)(void *context, int first_step, int last_step),
        void *context, size_t checkpoint_floats) {
    if (!visco_sh_material_observable_is_segmented(trajectory)) return 0;
    if ((save == NULL) || (restore == NULL) || (run == NULL) ||
            (checkpoint_floats != seg.checkpoint_floats)) return -1;
    seg.save = save; seg.restore = restore; seg.run = run;
    seg.context = context;
    return 1;
}

/* forward run, before step `step`: keep the state a later replay starts from */
int visco_sh_material_observable_forward_boundary(
        struct visco_sh_material_observable_trajectory *trajectory, int step) {
    int k;
    if (!visco_sh_material_observable_is_segmented(trajectory) || seg.replaying)
        return 0;
    k = seg_of(step);
    if ((k < 1) || (seg.first[k] != step)) return 0;
    if ((seg.save == NULL) || (seg.checkpoint[k] == NULL)) return -1;
    seg.save(seg.context, seg.checkpoint[k]);
    return 0;
}

/* reverse run: make the records of `step` available, recomputing its segment if needed */
int visco_sh_material_observable_load_step(
        const struct visco_sh_material_observable_trajectory *trajectory,
        int step) {
    int k;
    double started;
    if (!visco_sh_material_observable_is_segmented(trajectory)) return 0;
    if ((step < 0) || (step >= seg.nsteps)) return -1;
    if (seg.trajectory->steps[step].qsum != NULL) return 0;
    if ((seg.run == NULL) || (seg.restore == NULL) || (active_step != NULL))
        return -1;
    k = seg_of(step);
    if ((k >= 1) && (seg.checkpoint[k] == NULL)) return -1;
    started = MPI_Wtime();
    seg.restore(seg.context, (k >= 1) ? seg.checkpoint[k] : NULL);
    seg.replaying = 1;
    seg.run(seg.context, seg.first[k], seg.first[k + 1] - 1);
    seg.replaying = 0;
    seg.replayed_steps += (size_t)(seg.first[k + 1] - seg.first[k]);
    seg.replay_seconds += MPI_Wtime() - started;
    return (seg.trajectory->steps[step].qsum != NULL) ? 0 : -1;
}

static void seg_end_step(void) {
    size_t bytes = ((size_t)active_ny + 1) * ((size_t)active_nx + 1) * sizeof(float);
    float **now[3], **ref[3];
    int c;
    if (!seg.enabled || !seg.verify || (active_step == NULL) || (active_index < 0))
        return;
    now[0] = active_step->qsum; now[1] = active_step->strain_x;
    now[2] = active_step->strain_y;
    ref[0] = seg.full[active_index].qsum; ref[1] = seg.full[active_index].strain_x;
    ref[2] = seg.full[active_index].strain_y;
    for (c = 0; c < 3; ++c) {
        if (!seg.replaying) {
            memcpy(ref[c][0], now[c][0], bytes);
        } else {
            ++seg.compared;
            if (memcmp(ref[c][0], now[c][0], bytes) != 0) ++seg.mismatches;
        }
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

    n = (requested_segments < nsteps) ? requested_segments : nsteps;
    if ((n >= 1) && (requested_checkpoint_floats > 0))
        return seg_init(trajectory, nx, ny, nsteps, n);

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
    if (visco_sh_material_observable_is_segmented(trajectory)) {
        seg_release(trajectory);
        return;
    }
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
    if (visco_sh_material_observable_is_segmented(trajectory) &&
            (seg_of(step) != seg.loaded)) seg_map(seg_of(step));
    active_index = step;
    active_step = &trajectory->steps[step];
    active_nx = trajectory->nx;
    active_ny = trajectory->ny;
    return 0;
}

void visco_sh_material_observable_end_step(void) {
    seg_end_step();
    active_index = -1;
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
