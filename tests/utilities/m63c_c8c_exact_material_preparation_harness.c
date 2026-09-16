/* Runtime oracle for the inactive C8c exact SH material-preparation bridge. */
#include "fd.h"
#include "globvar.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LOCAL_NX 2
#define LOCAL_NY 2
#define MECHANISMS 1
#define SENTINEL (-913.25f)

#define REQUIRE(condition, text) do { \
    if (!(condition)) { fprintf(stderr, "material preparation oracle: %s\\n", text); return 0; } \
} while (0)

#ifdef INSTRUMENT_MATCOPY
static int matcopy_calls;
void matcopy_SH(float **rho, float **u, float **taus) {
    (void)rho; (void)u; (void)taus;
    ++matcopy_calls;
}
#endif

static float **new_matrix(void) { return matrix(0, NY + 1, 0, NX + 1); }
static float ***new_tensor(void) { return f3tensor(0, NY + 1, 0, NX + 1, 1, L); }
static void drop_matrix(float **m) { if (m) free_matrix(m, 0, NY + 1, 0, NX + 1); }
static void drop_tensor(float ***m) { if (m) free_f3tensor(m, 0, NY + 1, 0, NX + 1, 1, L); }

static void fill_matrix(float **m, float value) {
    int i, j;
    for (j = 0; j <= NY + 1; ++j)
        for (i = 0; i <= NX + 1; ++i) m[j][i] = value;
}
static void copy_matrix(float **to, float **from) {
    int j;
    for (j = 0; j <= NY + 1; ++j)
        memcpy(to[j], from[j], (size_t)(NX + 2) * sizeof(float));
}
static int same_matrix(float **a, float **b) {
    int j;
    for (j = 0; j <= NY + 1; ++j)
        if (memcmp(a[j], b[j], (size_t)(NX + 2) * sizeof(float)) != 0) return 0;
    return 1;
}
static void fill_tensor(float ***m, float value) {
    int i, j, l;
    for (j = 0; j <= NY + 1; ++j)
        for (i = 0; i <= NX + 1; ++i)
            for (l = 1; l <= L; ++l) m[j][i][l] = value;
}
static void copy_tensor(float ***to, float ***from) {
    int i, j, l;
    for (j = 0; j <= NY + 1; ++j)
        for (i = 0; i <= NX + 1; ++i)
            for (l = 1; l <= L; ++l) to[j][i][l] = from[j][i][l];
}
static int same_tensor(float ***a, float ***b) {
    int i, j, l;
    for (j = 0; j <= NY + 1; ++j)
        for (i = 0; i <= NX + 1; ++i)
            for (l = 1; l <= L; ++l)
                if (memcmp(&a[j][i][l], &b[j][i][l], sizeof(float)) != 0) return 0;
    return 1;
}
static int same_vector(float *a, float *b) {
    return memcmp(&a[1], &b[1], (size_t)L * sizeof(float)) == 0;
}
static void copy_vector(float *to, float *from) {
    memcpy(&to[1], &from[1], (size_t)L * sizeof(float));
}

static int allocate_target(struct matSH *t) {
    memset(t, 0, sizeof(*t));
    t->prho = new_matrix(); t->prhoi = new_matrix(); t->puip = new_matrix();
    t->pujp = new_matrix(); t->pu = new_matrix(); t->puipjp = new_matrix();
    t->pqs = new_matrix(); t->ptaus = new_matrix(); t->ptausipjp = new_matrix();
    t->fipjp = new_matrix(); t->f = new_matrix(); t->g = new_matrix();
    t->dip = new_tensor(); t->d = new_tensor(); t->e = new_tensor();
    t->peta = vector(1, L); t->etaip = vector(1, L); t->etajm = vector(1, L);
    t->bip = vector(1, L); t->bjm = vector(1, L); t->cip = vector(1, L); t->cjm = vector(1, L);
    return t->prho && t->prhoi && t->puip && t->pujp && t->pu && t->puipjp &&
        t->pqs && t->ptaus && t->ptausipjp && t->fipjp && t->f && t->g &&
        t->dip && t->d && t->e && t->peta && t->etaip && t->etajm &&
        t->bip && t->bjm && t->cip && t->cjm;
}
static void release_target(struct matSH *t) {
    drop_matrix(t->prho); drop_matrix(t->prhoi); drop_matrix(t->puip); drop_matrix(t->pujp);
    drop_matrix(t->pu); drop_matrix(t->puipjp); drop_matrix(t->pqs); drop_matrix(t->ptaus);
    drop_matrix(t->ptausipjp); drop_matrix(t->fipjp); drop_matrix(t->f); drop_matrix(t->g);
    drop_tensor(t->dip); drop_tensor(t->d); drop_tensor(t->e);
    if (t->peta) free_vector(t->peta, 1, L); if (t->etaip) free_vector(t->etaip, 1, L);
    if (t->etajm) free_vector(t->etajm, 1, L); if (t->bip) free_vector(t->bip, 1, L);
    if (t->bjm) free_vector(t->bjm, 1, L); if (t->cip) free_vector(t->cip, 1, L);
    if (t->cjm) free_vector(t->cjm, 1, L); memset(t, 0, sizeof(*t));
}
static void fill_target(struct matSH *t, float value) {
    int l;
    fill_matrix(t->prho, value); fill_matrix(t->prhoi, value); fill_matrix(t->puip, value);
    fill_matrix(t->pujp, value); fill_matrix(t->pu, value); fill_matrix(t->puipjp, value);
    fill_matrix(t->pqs, value); fill_matrix(t->ptaus, value); fill_matrix(t->ptausipjp, value);
    fill_matrix(t->fipjp, value); fill_matrix(t->f, value); fill_matrix(t->g, value);
    fill_tensor(t->dip, value); fill_tensor(t->d, value); fill_tensor(t->e, value);
    for (l = 1; l <= L; ++l) {
        t->peta[l] = value; t->etaip[l] = value; t->etajm[l] = value;
        t->bip[l] = value; t->bjm[l] = value; t->cip[l] = value; t->cjm[l] = value;
    }
}
static void copy_target(struct matSH *to, struct matSH *from) {
    copy_matrix(to->prho, from->prho); copy_matrix(to->prhoi, from->prhoi); copy_matrix(to->puip, from->puip);
    copy_matrix(to->pujp, from->pujp); copy_matrix(to->pu, from->pu); copy_matrix(to->puipjp, from->puipjp);
    copy_matrix(to->pqs, from->pqs); copy_matrix(to->ptaus, from->ptaus); copy_matrix(to->ptausipjp, from->ptausipjp);
    copy_matrix(to->fipjp, from->fipjp); copy_matrix(to->f, from->f); copy_matrix(to->g, from->g);
    copy_tensor(to->dip, from->dip); copy_tensor(to->d, from->d); copy_tensor(to->e, from->e);
    copy_vector(to->peta, from->peta); copy_vector(to->etaip, from->etaip); copy_vector(to->etajm, from->etajm);
    copy_vector(to->bip, from->bip); copy_vector(to->bjm, from->bjm); copy_vector(to->cip, from->cip); copy_vector(to->cjm, from->cjm);
}
static int same_target(struct matSH *a, struct matSH *b) {
    return same_matrix(a->prho,b->prho) && same_matrix(a->prhoi,b->prhoi) && same_matrix(a->puip,b->puip) &&
        same_matrix(a->pujp,b->pujp) && same_matrix(a->pu,b->pu) && same_matrix(a->puipjp,b->puipjp) &&
        same_matrix(a->pqs,b->pqs) && same_matrix(a->ptaus,b->ptaus) && same_matrix(a->ptausipjp,b->ptausipjp) &&
        same_matrix(a->fipjp,b->fipjp) && same_matrix(a->f,b->f) && same_matrix(a->g,b->g) &&
        same_tensor(a->dip,b->dip) && same_tensor(a->d,b->d) && same_tensor(a->e,b->e) &&
        same_vector(a->peta,b->peta) && same_vector(a->etaip,b->etaip) && same_vector(a->etajm,b->etajm) &&
        same_vector(a->bip,b->bip) && same_vector(a->bjm,b->bjm) && same_vector(a->cip,b->cip) && same_vector(a->cjm,b->cjm);
}

static void fill_state(float **primary, float **rho, float **q, int rank, float shift) {
    int i, j;
    fill_matrix(primary, -7.0f); fill_matrix(rho, -8.0f); fill_matrix(q, -9.0f);
    for (j = 1; j <= NY; ++j) for (i = 1; i <= NX; ++i) {
        primary[j][i] = 20.0f + 10.0f * rank + 2.0f * j + i + shift;
        rho[j][i] = 2.0f + rank + 0.2f * j + 0.03f * i + shift * 0.01f;
        q[j][i] = 50.0f + 20.0f * rank + 3.0f * j + i + shift;
    }
}
static void setup_globals(int rank, int size) {
    int n;
    NX = LOCAL_NX; NY = LOCAL_NY; L = MECHANISMS; INVMAT1 = 3; DT = 0.001f;
    Q_PARAMETERIZATION_MODE = Q_PARAMETERIZATION_PHYSICAL;
    Q_APPROX_FMIN = 1.0f; Q_APPROX_FMAX = 10.0f; Q_APPROX_DF = 1.0f;
    MYID = rank; NP = NPROC = size; NPROCX = size; NPROCY = 1; FDORDER = 2; FW = 0;
    for (n = 0; n < 5; ++n) INDEX[n] = (size == 1) ? 0 : (rank + 1) % size;
    FL = vector(1, L); FL[1] = 10.0f;
    FP = fopen("/dev/null", "w");
}
static void cleanup_globals(void) {
    if (FP) fclose(FP); FP = NULL;
    if (FL) free_vector(FL, 1, L); FL = NULL;
}
static int all_true(int local) {
    int global = 0;
    return MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD) == MPI_SUCCESS && global;
}
static int finite_target(struct matSH *t) {
    int i,j,l;
    for (j=1;j<=NY;++j) for (i=1;i<=NX;++i) {
        if (!isfinite(t->pu[j][i]) || !isfinite(t->prho[j][i]) || !isfinite(t->pqs[j][i]) ||
            !isfinite(t->ptaus[j][i]) || !isfinite(t->puip[j][i]) || !isfinite(t->pujp[j][i]) ||
            !isfinite(t->prhoi[j][i]) || !isfinite(t->ptausipjp[j][i]) || !isfinite(t->fipjp[j][i]) || !isfinite(t->f[j][i])) return 0;
        for (l=1;l<=L;++l) if (!isfinite(t->dip[j][i][l]) || !isfinite(t->d[j][i][l])) return 0;
    }
    for (l=1;l<=L;++l) if (!isfinite(t->peta[l]) || !isfinite(t->etaip[l]) || !isfinite(t->etajm[l]) || !isfinite(t->bip[l]) || !isfinite(t->bjm[l]) || !isfinite(t->cip[l]) || !isfinite(t->cjm[l])) return 0;
    return 1;
}
static void request_init(struct visco_sh_exact_material_preparation_request *r, float **p, float **rho, float **q, struct matSH *t, float *peta) {
    r->primary=p; r->rho=rho; r->physical_q=q; r->target=t; r->mechanisms=L; r->dt=DT; r->frequencies_hz=FL; r->peta=peta;
}
static int expected_tau(float q) {
    struct q_tau_mapping m;
    init_q_tau_mapping(&m, Q_PARAMETERIZATION_PHYSICAL, L, FL, Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF);
    return isfinite(q_to_tau(q, &m));
}

static int run_failure(int invalid_rank, int rank) {
    struct matSH t, before;
    struct visco_sh_exact_material_preparation_request r;
    float **p = new_matrix(), **rho = new_matrix(), **q = new_matrix(), *peta = vector(1,L);
    int rc, calls = 0;
    REQUIRE(p && rho && q && peta && allocate_target(&t) && allocate_target(&before), "failure allocations");
    fill_state(p,rho,q,rank,0.0f); peta[1]=DT/(1.0f/(2.0f*PI*FL[1]));
    if (rank == invalid_rank) q[1][1] = -1.0f;
    fill_target(&t, SENTINEL); copy_target(&before,&t); request_init(&r,p,rho,q,&t,peta);
    rc=visco_sh_exact_prepare_visco_material(&r);
#ifdef INSTRUMENT_MATCOPY
    calls=matcopy_calls;
#endif
    REQUIRE(all_true(rc == -1 && same_target(&t,&before)), "global invalid configuration must be fail-closed and transactional");
    REQUIRE(MPI_Allreduce(&calls,&calls,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD)==MPI_SUCCESS && calls==0, "matcopy must not run after global validation failure");
    release_target(&t); release_target(&before); drop_matrix(p); drop_matrix(rho); drop_matrix(q); free_vector(peta,1,L);
    return 1;
}

static int run_success(int rank, int size) {
    struct matSH t, a_snapshot, trial_target;
    struct visco_sh_exact_material_preparation_request r;
    struct q_tau_mapping mapping;
    struct visco_sh_exact_trial_state_request trial_request;
    float **p=new_matrix(), **rho=new_matrix(), **q=new_matrix(), **trial_p=new_matrix(), **trial_rho=new_matrix(), **trial_q=new_matrix(), **trial_tau=new_matrix();
    float **step_p=new_matrix(), **step_rho=new_matrix(), **step_q=new_matrix(), *peta=vector(1,L), *bad_peta=vector(1,L);
    int i,j,neighbor=(size==1?0:(rank+1)%size);
    float old_tau, old_tau_average, old_f, old_d, old_puip, old_prhoi;
    REQUIRE(p&&rho&&q&&trial_p&&trial_rho&&trial_q&&trial_tau&&step_p&&step_rho&&step_q&&peta&&bad_peta&&allocate_target(&t)&&allocate_target(&a_snapshot)&&allocate_target(&trial_target),"success allocations");
    fill_state(p,rho,q,rank,0.0f); fill_matrix(step_p,0.0f); fill_matrix(step_rho,0.0f); fill_matrix(step_q,0.0f);
    peta[1]=DT/(1.0f/(2.0f*PI*FL[1])); bad_peta[1]=peta[1]*1.25f;
    fill_target(&t,SENTINEL); request_init(&r,p,rho,q,&t,peta);
    REQUIRE(all_true(visco_sh_exact_prepare_visco_material(&r)==0),"real material preparation succeeds");
    init_q_tau_mapping(&mapping,Q_PARAMETERIZATION_PHYSICAL,L,FL,Q_APPROX_FMIN,Q_APPROX_FMAX,Q_APPROX_DF);
    for(j=1;j<=NY;++j) for(i=1;i<=NX;++i) REQUIRE(t.pu[j][i]==p[j][i] && t.prho[j][i]==rho[j][i] && t.pqs[j][i]==q[j][i] && t.ptaus[j][i]==q_to_tau(q[j][i],&mapping),"authoritative Q and derived tau copied");
    REQUIRE(expected_tau(q[1][1]) && t.ptaus[1][1] != SENTINEL,"pre-call tau sentinel cannot survive");
    if(size>1) {
        float neighbor_left_q=50.0f+20.0f*neighbor+3.0f*1.0f+(float)NX;
        float neighbor_right_q=50.0f+20.0f*neighbor+3.0f*1.0f+1.0f;
        float mu_right=20.0f+10.0f*neighbor+2.0f+1.0f;
        float expected_average;
        REQUIRE(t.prho[1][0] == 2.0f+neighbor+0.2f+0.03f*(float)NX,"left rho halo comes from neighbor");
        REQUIRE(t.pqs[1][NX+1] == SENTINEL,"physical Q halo remains non-authoritative");
        REQUIRE(t.ptaus[1][0] == q_to_tau(neighbor_left_q,&mapping) && t.ptaus[1][NX+1] == q_to_tau(neighbor_right_q,&mapping),"tau halo comes from neighboring material");
        REQUIRE(t.pu[1][NX+1] == mu_right,"right shear halo comes from neighbor");
        REQUIRE(t.puip[1][NX] == (float)(2.0/((1.0/t.pu[1][NX])+(1.0/t.pu[1][NX+1]))),"av_mu consumes exchanged right halo");
        expected_average=(float)(0.25*(t.ptaus[1][NX]+t.ptaus[1][NX+1]+t.ptaus[2][NX]+t.ptaus[2][NX+1]));
        REQUIRE(t.ptausipjp[1][NX] == expected_average,"av_tau consumes exchanged tau halo");
    }
    REQUIRE(finite_target(&t),"solver-relevant real material caches are finite");
    copy_target(&a_snapshot,&t);

    /* B2 alpha=0 creates the independently allocated trial authoritative state. */
    trial_request.nx=NX; trial_request.ny=NY; trial_request.alpha=0.0f; trial_request.primary_bounds_enabled=0;
    trial_request.primary_lower=0.0f; trial_request.primary_upper=0.0f; trial_request.rho_lower=0.1f; trial_request.rho_upper=100.0f; trial_request.q_lower=1.0f; trial_request.q_upper=1000.0f;
    trial_request.q_mapping=&mapping; trial_request.base_primary=p; trial_request.base_rho=rho; trial_request.base_q=q;
    trial_request.optimizer_step_primary=step_p; trial_request.optimizer_step_rho=step_rho; trial_request.optimizer_step_q=step_q;
    trial_request.trial_primary=trial_p; trial_request.trial_rho=trial_rho; trial_request.trial_q=trial_q; trial_request.trial_tau=trial_tau;
    REQUIRE(visco_sh_exact_build_trial_parameter_state(&trial_request)==0,"real B2 zero-step trial state");
    fill_target(&trial_target,SENTINEL); request_init(&r,trial_p,trial_rho,trial_q,&trial_target,peta);
    REQUIRE(all_true(visco_sh_exact_prepare_visco_material(&r)==0) && same_target(&a_snapshot,&trial_target),"base/trial alpha-zero material state is bitwise identical");

    /* Q-only: tau and attenuation coefficients change, shear/rho-derived state does not. */
    request_init(&r,p,rho,q,&t,peta); q[1][1]+=17.0f; old_tau=a_snapshot.ptaus[1][1]; old_tau_average=a_snapshot.ptausipjp[1][1]; old_f=a_snapshot.f[1][1]; old_d=a_snapshot.d[1][1][1]; old_puip=a_snapshot.puip[1][1]; old_prhoi=a_snapshot.prhoi[1][1];
    REQUIRE(all_true(visco_sh_exact_prepare_visco_material(&r)==0),"Q perturbation preparation");
    REQUIRE(t.ptaus[1][1]!=old_tau && t.ptausipjp[1][1]!=old_tau_average && t.f[1][1]!=old_f && t.d[1][1][1]!=old_d && t.puip[1][1]==old_puip && t.prhoi[1][1]==old_prhoi,"Q perturbation updates tau caches only");

    /* Primary-only changes direct-mu staggered state while physical-Q tau is fixed. */
    q[1][1]-=17.0f; p[1][1]+=5.0f; request_init(&r,p,rho,q,&t,peta); old_puip=a_snapshot.puip[1][1]; old_tau=a_snapshot.ptaus[1][1];
    REQUIRE(all_true(visco_sh_exact_prepare_visco_material(&r)==0),"primary perturbation preparation");
    REQUIRE(t.puip[1][1]!=old_puip && t.ptaus[1][1]==old_tau,"primary perturbation updates staggered shear not tau");

    /* Rho-only updates reciprocal-density state; direct-mu and physical Q remain fixed. */
    p[1][1]-=5.0f; rho[1][1]+=1.0f; request_init(&r,p,rho,q,&t,peta); old_prhoi=a_snapshot.prhoi[1][1]; old_puip=a_snapshot.puip[1][1]; old_tau=a_snapshot.ptaus[1][1];
    REQUIRE(all_true(visco_sh_exact_prepare_visco_material(&r)==0),"rho perturbation preparation");
    REQUIRE(t.prhoi[1][1]!=old_prhoi && t.puip[1][1]==old_puip && t.ptaus[1][1]==old_tau,"rho perturbation updates reciprocal density without stale tau");

    /* Reusing an allocated target must be deterministic. */
    rho[1][1]-=1.0f; request_init(&r,p,rho,q,&t,peta); REQUIRE(all_true(visco_sh_exact_prepare_visco_material(&r)==0) && same_target(&t,&a_snapshot),"A->B->A target reuse restores exact state");
    fill_target(&t,SENTINEL); copy_target(&trial_target,&t); request_init(&r,p,rho,q,&t,bad_peta);
    REQUIRE(all_true(visco_sh_exact_prepare_visco_material(&r)==-1 && same_target(&t,&trial_target)),"incompatible relaxation configuration is transactional fail-closed");

    release_target(&t); release_target(&a_snapshot); release_target(&trial_target);
    drop_matrix(p); drop_matrix(rho); drop_matrix(q); drop_matrix(trial_p); drop_matrix(trial_rho); drop_matrix(trial_q); drop_matrix(trial_tau); drop_matrix(step_p); drop_matrix(step_rho); drop_matrix(step_q); free_vector(peta,1,L); free_vector(bad_peta,1,L);
    return 1;
}

int main(int argc, char **argv) {
    int rank, size, ok, invalid_rank = -1, buffer_size = 65536;
    void *buffer;
    if (argc != 2) return 2;
    MPI_Init(&argc,&argv); MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    setup_globals(rank,size); buffer=malloc((size_t)buffer_size); if (!buffer || MPI_Buffer_attach(buffer,buffer_size)!=MPI_SUCCESS) return 3;
    if (strcmp(argv[1],"failure-rank0")==0) invalid_rank=0;
    else if (strcmp(argv[1],"failure-rank1")==0) invalid_rank=1;
    if (invalid_rank >= 0) ok=run_failure(invalid_rank,rank); else if (strcmp(argv[1],"success")==0) ok=run_success(rank,size); else ok=0;
    ok=all_true(ok); if(rank==0) printf("{\"mode\":\"%s\",\"ok\":%s}\n",argv[1],ok?"true":"false");
    MPI_Buffer_detach(&buffer,&buffer_size); free(buffer); cleanup_globals(); MPI_Finalize(); return ok?0:1;
}
