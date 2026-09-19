#include "fd.h"

#define LO (-4)
#define XHI (NX + 5)
#define YHI (NY + 5)
#define PMAX 6
#define RECORD_MAX 32

float DT = 0.001f, DH = 10.0f;
int MYID = 0, FDORDER = 4, INVMAT1 = 1, FW = 2;
int FREE_SURF = 0, BOUNDARY = 0, GRAD_FORM = 2;
int NPROCX = 3, NPROCY = 3, POS[3] = {0, 0, 0}, L = 1;
int NX = 18, NY = 17;
FILE *FP;

static float recorded[6][RECORD_MAX][RECORD_MAX];
static unsigned char record_hits[RECORD_MAX][RECORD_MAX];

void visco_psv_exact_strain(int j, int i, float vxx, float vyx,
                            float vxy, float vyy) {
    recorded[2][j][i] = vxx;
    recorded[3][j][i] = vyx;
    recorded[4][j][i] = vxy;
    recorded[5][j][i] = vyy;
    record_hits[j][i]++;
}

void update_s_visc_PSV_region_test_reset(void);
size_t update_s_visc_PSV_region_test_fast_cells(void);

struct state {
    float **vx, **vy, **ux, **uy, **uxy, **uyx;
    float **sxx, **syy, **sxy;
    float **pi, **u, **uipjp, **rho, **absorb;
    float **fipjp, **f, **g;
    float ***r, ***p, ***q, ***d, ***e, ***dip;
    float **psi_vxx, **psi_vyx, **psi_vyy, **psi_vxy;
};

struct coefficients {
    float *hc, *K_x, *a_x, *b_x, *K_x_half, *a_x_half, *b_x_half;
    float *K_y, *a_y, *b_y, *K_y_half, *a_y_half, *b_y_half;
    float *bip, *bjm, *cip, *cjm;
};

struct rect { int x1, x2, y1, y2; };

static float value(int id, int j, int i) {
    return 0.01f * (float)id + 0.00013f * (float)(j + 7) +
           0.000017f * (float)(i + 9);
}

static float **new_full_matrix(void) { return matrix(LO, YHI, LO, XHI); }

static struct state make_state(void) {
    struct state s;
    int i, j;
#define NEW_FULL(name) s.name = new_full_matrix()
    NEW_FULL(vx); NEW_FULL(vy); NEW_FULL(ux); NEW_FULL(uy);
    NEW_FULL(uxy); NEW_FULL(uyx); NEW_FULL(sxx); NEW_FULL(syy);
    NEW_FULL(sxy); NEW_FULL(pi); NEW_FULL(u); NEW_FULL(uipjp);
    NEW_FULL(rho); NEW_FULL(absorb); NEW_FULL(fipjp); NEW_FULL(f);
    NEW_FULL(g);
#undef NEW_FULL
    s.r = f3tensor(LO, YHI, LO, XHI, 1, 1);
    s.p = f3tensor(LO, YHI, LO, XHI, 1, 1);
    s.q = f3tensor(LO, YHI, LO, XHI, 1, 1);
    s.d = f3tensor(LO, YHI, LO, XHI, 1, 1);
    s.e = f3tensor(LO, YHI, LO, XHI, 1, 1);
    s.dip = f3tensor(LO, YHI, LO, XHI, 1, 1);
    s.psi_vxx = matrix(1, NY, 1, PMAX);
    s.psi_vyx = matrix(1, NY, 1, PMAX);
    s.psi_vyy = matrix(1, PMAX, 1, NX);
    s.psi_vxy = matrix(1, PMAX, 1, NX);
    for (j = LO; j <= YHI; ++j) {
        for (i = LO; i <= XHI; ++i) {
            s.vx[j][i] = value(1, j, i); s.vy[j][i] = value(2, j, i);
            s.ux[j][i] = value(3, j, i); s.uy[j][i] = value(4, j, i);
            s.uxy[j][i] = value(5, j, i); s.uyx[j][i] = value(6, j, i);
            s.sxx[j][i] = value(7, j, i); s.syy[j][i] = value(8, j, i);
            s.sxy[j][i] = value(9, j, i);
            s.pi[j][i] = 2.4f + value(10, j, i);
            s.u[j][i] = 1.3f + value(11, j, i);
            s.uipjp[j][i] = 0.8f + value(12, j, i);
            s.rho[j][i] = 1.7f + value(13, j, i);
            s.absorb[j][i] = 1.0f;
            s.fipjp[j][i] = 0.7f + value(14, j, i);
            s.f[j][i] = 0.6f + value(15, j, i);
            s.g[j][i] = 1.8f + value(16, j, i);
            s.r[j][i][1] = value(17, j, i);
            s.p[j][i][1] = value(18, j, i);
            s.q[j][i][1] = value(19, j, i);
            s.d[j][i][1] = 0.02f + value(20, j, i);
            s.e[j][i][1] = 0.03f + value(21, j, i);
            s.dip[j][i][1] = 0.04f + value(22, j, i);
        }
    }
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= PMAX; ++i) {
            s.psi_vxx[j][i] = value(23, j, i);
            s.psi_vyx[j][i] = value(24, j, i);
        }
    for (j = 1; j <= PMAX; ++j)
        for (i = 1; i <= NX; ++i) {
            s.psi_vyy[j][i] = value(25, j, i);
            s.psi_vxy[j][i] = value(26, j, i);
        }
    return s;
}

static struct coefficients make_coefficients(void) {
    struct coefficients c;
    int i;
    c.hc = vector(1, 4);
    c.K_x = vector(1, PMAX); c.a_x = vector(1, PMAX); c.b_x = vector(1, PMAX);
    c.K_x_half = vector(1, PMAX); c.a_x_half = vector(1, PMAX); c.b_x_half = vector(1, PMAX);
    c.K_y = vector(1, PMAX); c.a_y = vector(1, PMAX); c.b_y = vector(1, PMAX);
    c.K_y_half = vector(1, PMAX); c.a_y_half = vector(1, PMAX); c.b_y_half = vector(1, PMAX);
    c.bip = vector(1, 1); c.bjm = vector(1, 1); c.cip = vector(1, 1); c.cjm = vector(1, 1);
    c.hc[1] = 1.1962890625f; c.hc[2] = -0.0797526042f;
    c.hc[3] = 0.0095703125f; c.hc[4] = -0.0006975446f;
    for (i = 1; i <= PMAX; ++i) {
        c.K_x[i] = 1.05f + 0.01f * i; c.a_x[i] = 0.004f * i; c.b_x[i] = 0.82f;
        c.K_x_half[i] = 1.07f + 0.01f * i; c.a_x_half[i] = 0.003f * i; c.b_x_half[i] = 0.84f;
        c.K_y[i] = 1.06f + 0.01f * i; c.a_y[i] = 0.005f * i; c.b_y[i] = 0.81f;
        c.K_y_half[i] = 1.08f + 0.01f * i; c.a_y_half[i] = 0.002f * i; c.b_y_half[i] = 0.83f;
    }
    c.bip[1] = 0.91f; c.bjm[1] = 0.89f; c.cip[1] = 0.96f; c.cjm[1] = 0.94f;
    return c;
}

static void free_state(struct state *s) {
#define FREE_FULL(name) free_matrix(s->name, LO, YHI, LO, XHI)
    FREE_FULL(vx); FREE_FULL(vy); FREE_FULL(ux); FREE_FULL(uy);
    FREE_FULL(uxy); FREE_FULL(uyx); FREE_FULL(sxx); FREE_FULL(syy);
    FREE_FULL(sxy); FREE_FULL(pi); FREE_FULL(u); FREE_FULL(uipjp);
    FREE_FULL(rho); FREE_FULL(absorb); FREE_FULL(fipjp); FREE_FULL(f); FREE_FULL(g);
#undef FREE_FULL
    free_f3tensor(s->r, LO, YHI, LO, XHI, 1, 1);
    free_f3tensor(s->p, LO, YHI, LO, XHI, 1, 1);
    free_f3tensor(s->q, LO, YHI, LO, XHI, 1, 1);
    free_f3tensor(s->d, LO, YHI, LO, XHI, 1, 1);
    free_f3tensor(s->e, LO, YHI, LO, XHI, 1, 1);
    free_f3tensor(s->dip, LO, YHI, LO, XHI, 1, 1);
    free_matrix(s->psi_vxx, 1, NY, 1, PMAX); free_matrix(s->psi_vyx, 1, NY, 1, PMAX);
    free_matrix(s->psi_vyy, 1, PMAX, 1, NX); free_matrix(s->psi_vxy, 1, PMAX, 1, NX);
}

static void free_coefficients(struct coefficients *c) {
#define FREE_PML(name) free_vector(c->name, 1, PMAX)
    free_vector(c->hc, 1, 4);
    FREE_PML(K_x); FREE_PML(a_x); FREE_PML(b_x); FREE_PML(K_x_half); FREE_PML(a_x_half); FREE_PML(b_x_half);
    FREE_PML(K_y); FREE_PML(a_y); FREE_PML(b_y); FREE_PML(K_y_half); FREE_PML(a_y_half); FREE_PML(b_y_half);
#undef FREE_PML
    free_vector(c->bip, 1, 1); free_vector(c->bjm, 1, 1);
    free_vector(c->cip, 1, 1); free_vector(c->cjm, 1, 1);
}

static int same_matrix(float **a, float **b, int j0, int j1, int i0, int i1) {
    int j;
    for (j = j0; j <= j1; ++j)
        if (memcmp(&a[j][i0], &b[j][i0], (size_t)(i1 - i0 + 1) * sizeof(float))) return 0;
    return 1;
}

static int same_tensor(float ***a, float ***b) {
    int i, j;
    for (j = LO; j <= YHI; ++j)
        for (i = LO; i <= XHI; ++i)
            if (memcmp(&a[j][i][1], &b[j][i][1], sizeof(float))) return 0;
    return 1;
}

static int same_mutable(const struct state *a, const struct state *b, int viscous) {
#define CHECK_FULL(name) if (!same_matrix(a->name, b->name, LO, YHI, LO, XHI)) return 0
    CHECK_FULL(ux); CHECK_FULL(uy); CHECK_FULL(uxy); CHECK_FULL(uyx);
    CHECK_FULL(sxx); CHECK_FULL(syy); CHECK_FULL(sxy);
#undef CHECK_FULL
    if (!same_matrix(a->psi_vxx, b->psi_vxx, 1, NY, 1, PMAX) ||
        !same_matrix(a->psi_vyx, b->psi_vyx, 1, NY, 1, PMAX) ||
        !same_matrix(a->psi_vyy, b->psi_vyy, 1, PMAX, 1, NX) ||
        !same_matrix(a->psi_vxy, b->psi_vxy, 1, PMAX, 1, NX)) return 0;
    return !viscous || (same_tensor(a->r, b->r) && same_tensor(a->p, b->p) && same_tensor(a->q, b->q));
}

static void call_elastic(struct state *s, const struct coefficients *c, struct rect z) {
    if (z.x1 > z.x2 || z.y1 > z.y2) return;
    update_s_elastic_PML_PSV(z.x1, z.x2, z.y1, z.y2, NX, NY,
        s->vx, s->vy, s->ux, s->uy, s->uxy, s->uyx, s->sxx, s->syy, s->sxy,
        s->pi, s->u, s->uipjp, s->absorb, s->rho, c->hc, 0,
        c->K_x, c->a_x, c->b_x, c->K_x_half, c->a_x_half, c->b_x_half,
        c->K_y, c->a_y, c->b_y, c->K_y_half, c->a_y_half, c->b_y_half,
        s->psi_vxx, s->psi_vyy, s->psi_vxy, s->psi_vyx, 0);
}

static void call_viscous(struct state *s, const struct coefficients *c, struct rect z) {
    if (z.x1 > z.x2 || z.y1 > z.y2) return;
    update_s_visc_PML_PSV(z.x1, z.x2, z.y1, z.y2, NX, NY,
        s->vx, s->vy, s->ux, s->uy, s->uxy, s->uyx, s->sxx, s->syy, s->sxy,
        s->pi, s->u, s->uipjp, s->rho, c->hc, 0, s->r, s->p, s->q,
        s->fipjp, s->f, s->g, c->bip, c->bjm, c->cip, c->cjm,
        s->d, s->e, s->dip,
        c->K_x, c->a_x, c->b_x, c->K_x_half, c->a_x_half, c->b_x_half,
        c->K_y, c->a_y, c->b_y, c->K_y_half, c->a_y_half, c->b_y_half,
        s->psi_vxx, s->psi_vyy, s->psi_vxy, s->psi_vyx, 0);
}

static void canonical_partition(struct rect z[5]) {
    int r = FDORDER / 2;
    int left = BOUNDARY || POS[1] > 0, right = BOUNDARY || POS[1] < NPROCX - 1;
    int top = POS[2] > 0, bottom = POS[2] < NPROCY - 1;
    int ix0 = 1 + r * left, ix1 = NX - r * right;
    int iy0 = 1 + r * top, iy1 = NY - r * bottom;
    z[0] = (struct rect){ix0, ix1, iy0, iy1};
    z[1] = (struct rect){1, ix0 - 1, 1, NY};
    z[2] = (struct rect){ix1 + 1, NX, 1, NY};
    z[3] = (struct rect){ix0, ix1, 1, iy0 - 1};
    z[4] = (struct rect){ix0, ix1, iy1 + 1, NY};
}

static void physical_edge_partition(struct rect z[5]) {
    int margin = 4;
    z[0] = (struct rect){margin + 1, NX - margin, margin + 1, NY - margin};
    z[1] = (struct rect){1, margin, 1, NY};
    z[2] = (struct rect){NX - margin + 1, NX, 1, NY};
    z[3] = (struct rect){margin + 1, NX - margin, 1, margin};
    z[4] = (struct rect){margin + 1, NX - margin, NY - margin + 1, NY};
}

static void clear_records(void) {
    memset(recorded, 0, sizeof(recorded));
    memset(record_hits, 0, sizeof(record_hits));
}

static int records_once(void) {
    int i, j;
    for (j = 1; j <= NY; ++j)
        for (i = 1; i <= NX; ++i)
            if (record_hits[j][i] != 1) return 0;
    return 1;
}

static int run_case(const char *name, int viscous, int order, int fw,
                    int npx, int npy, int px, int py, int physical_partition,
                    size_t *fast_full, size_t *fast_split) {
    struct coefficients c = make_coefficients();
    struct state reference = make_state(), candidate = make_state();
    struct rect regions[5], whole = {1, NX, 1, NY};
    float saved_records[6][RECORD_MAX][RECORD_MAX];
    int k, records_ok = 1;
    FDORDER = order; FW = fw; NPROCX = npx; NPROCY = npy; POS[1] = px; POS[2] = py;
    if (physical_partition) physical_edge_partition(regions); else canonical_partition(regions);
    clear_records();
    update_s_visc_PSV_region_test_reset();
    if (viscous) call_viscous(&reference, &c, whole); else call_elastic(&reference, &c, whole);
    if (order == 4 && viscous) {
        memcpy(saved_records, recorded, sizeof(recorded));
        records_ok = records_once();
    }
    *fast_full = update_s_visc_PSV_region_test_fast_cells();
    clear_records();
    update_s_visc_PSV_region_test_reset();
    for (k = 0; k < 5; ++k)
        if (viscous) call_viscous(&candidate, &c, regions[k]); else call_elastic(&candidate, &c, regions[k]);
    *fast_split = update_s_visc_PSV_region_test_fast_cells();
    if (!same_mutable(&reference, &candidate, viscous)) {
        fprintf(stderr, "%s: mutable state differs\n", name); return 0;
    }
    if (order == 4 && viscous &&
        (!records_ok || !records_once() || memcmp(saved_records, recorded, sizeof(recorded)))) {
        fprintf(stderr, "%s: exact strain records differ\n", name); return 0;
    }
    if (order == 8 && viscous && *fast_full != *fast_split) {
        fprintf(stderr, "%s: fast coverage differs (%zu != %zu)\n", name, *fast_full, *fast_split); return 0;
    }
    free_state(&reference); free_state(&candidate); free_coefficients(&c);
    printf("PASS %s\n", name);
    return 1;
}

static int outside_region_unchanged(void) {
    struct coefficients c = make_coefficients();
    struct state before = make_state(), candidate = make_state();
    struct state empty_before = make_state(), empty_candidate = make_state();
    struct rect small = {7, 8, 8, 9}, empty = {9, 8, 5, 7};
    int i, j, ok = 1;
    FDORDER = 4; FW = 2; NPROCX = NPROCY = 3; POS[1] = POS[2] = 1;
    clear_records();
    call_viscous(&candidate, &c, small);
    for (j = LO; j <= YHI; ++j) for (i = LO; i <= XHI; ++i) {
        if (i >= small.x1 && i <= small.x2 && j >= small.y1 && j <= small.y2) continue;
#define OUTSIDE_SAME(name) if (memcmp(&before.name[j][i], &candidate.name[j][i], sizeof(float))) ok = 0
        OUTSIDE_SAME(sxx); OUTSIDE_SAME(syy); OUTSIDE_SAME(sxy);
        OUTSIDE_SAME(ux); OUTSIDE_SAME(uy); OUTSIDE_SAME(uxy); OUTSIDE_SAME(uyx);
#undef OUTSIDE_SAME
        if (memcmp(&before.r[j][i][1], &candidate.r[j][i][1], sizeof(float)) ||
            memcmp(&before.p[j][i][1], &candidate.p[j][i][1], sizeof(float)) ||
            memcmp(&before.q[j][i][1], &candidate.q[j][i][1], sizeof(float))) ok = 0;
    }
    if (record_hits[8][7] != 1 || record_hits[8][8] != 1 ||
        record_hits[9][7] != 1 || record_hits[9][8] != 1) ok = 0;
    call_viscous(&empty_candidate, &c, empty);
    if (!same_mutable(&empty_before, &empty_candidate, 1)) ok = 0;
    free_state(&before); free_state(&candidate);
    free_state(&empty_before); free_state(&empty_candidate); free_coefficients(&c);
    if (!ok) fprintf(stderr, "small/empty rectangle contract failed\n");
    else printf("PASS small_and_empty_rectangles\n");
    return ok;
}

int main(void) {
    size_t full = 0, split = 0, fd8_full = 0, fd8_split = 0;
    int ok = 1;
    FP = stdout;
    ok &= run_case("elastic_fd4_no_cpml_interior", 0, 4, 0, 3, 3, 1, 1, 0, &full, &split);
    ok &= run_case("elastic_fd4_cpml_top_left", 0, 4, 2, 3, 3, 0, 0, 0, &full, &split);
    ok &= run_case("elastic_fd4_cpml_bottom_right", 0, 4, 2, 3, 3, 2, 2, 0, &full, &split);
    ok &= run_case("elastic_fd8_cpml", 0, 8, 2, 3, 3, 0, 0, 0, &full, &split);
    ok &= run_case("visco_fd4_l1_cpml", 1, 4, 2, 3, 3, 2, 2, 0, &full, &split);
    ok &= run_case("visco_fd4_false_cpml_oracle", 1, 4, 2, 1, 1, 0, 0, 1, &full, &split);
    ok &= run_case("visco_fd8_l1_fast_path", 1, 8, 2, 1, 1, 0, 0, 1, &fd8_full, &fd8_split);
    ok &= outside_region_unchanged();
    printf("FAST_PATH full=%zu split=%zu\n", fd8_full, fd8_split);
    printf("EXACT_RECORDS fields=6 strain_hits=once\n");
    if (ok) printf("ALL_REGION_ORACLES_PASS\n");
    return ok ? 0 : 1;
}
