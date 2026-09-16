/* MPI harness for repeated actual production forward timesteps and the
 * M6.3c-5b reverse-time fixed-material full-state transpose driver. */

#define main m63c_embedded_c5a_main
#include "m63c_full_state_step_harness.c"
#undef main

static void state_zero(struct owned_state *state) {
    memset(state->vz.data, 0,
           (size_t)state->vz.nrows * state->vz.ncols * sizeof(float));
    memset(state->sxz.data, 0,
           (size_t)state->sxz.nrows * state->sxz.ncols * sizeof(float));
    memset(state->syz.data, 0,
           (size_t)state->syz.nrows * state->syz.ncols * sizeof(float));
    memset(state->r.data, 0, (size_t)state->r.nrows * state->r.ncols *
           (state->r.mechanisms + 1) * sizeof(float));
    memset(state->q.data, 0, (size_t)state->q.nrows * state->q.ncols *
           (state->q.mechanisms + 1) * sizeof(float));
    if (FW > 0) {
        memset(state->psi_sxz_x.data, 0, (size_t)state->psi_sxz_x.nrows *
               state->psi_sxz_x.ncols * sizeof(float));
        memset(state->psi_syz_y.data, 0, (size_t)state->psi_syz_y.nrows *
               state->psi_syz_y.ncols * sizeof(float));
        memset(state->psi_vzx.data, 0, (size_t)state->psi_vzx.nrows *
               state->psi_vzx.ncols * sizeof(float));
        memset(state->psi_vzy.data, 0, (size_t)state->psi_vzy.nrows *
               state->psi_vzy.ncols * sizeof(float));
    }
}

static double field_difference(
        const struct field *left, const struct field *right) {
    double numerator = 0.0, denominator = 0.0;
    int k, count = left->nrows * left->ncols;
    for (k = 0; k < count; ++k) {
        double delta = (double)left->data[k] - right->data[k];
        numerator += delta * delta;
        denominator += (double)right->data[k] * right->data[k];
    }
    return sqrt(numerator / fmax(denominator, 1.0e-300));
}

static double volume_difference(
        const struct volume *left, const struct volume *right) {
    double numerator = 0.0, denominator = 0.0;
    int i, j, l;
    for (j = left->rmin; j <= left->rmax; ++j)
        for (i = left->cmin; i <= left->cmax; ++i)
            for (l = 1; l <= left->mechanisms; ++l) {
                double delta = (double)left->v[j][i][l] - right->v[j][i][l];
                numerator += delta * delta;
                denominator += (double)right->v[j][i][l] * right->v[j][i][l];
            }
    return sqrt(numerator / fmax(denominator, 1.0e-300));
}

static double state_difference(
        const struct owned_state *left, const struct owned_state *right) {
    double maximum = 0.0;
    maximum = fmax(maximum, field_difference(&left->vz, &right->vz));
    maximum = fmax(maximum, field_difference(&left->sxz, &right->sxz));
    maximum = fmax(maximum, field_difference(&left->syz, &right->syz));
    maximum = fmax(maximum, volume_difference(&left->r, &right->r));
    maximum = fmax(maximum, volume_difference(&left->q, &right->q));
    if (FW > 0) {
        maximum = fmax(maximum, field_difference(
                &left->psi_sxz_x, &right->psi_sxz_x));
        maximum = fmax(maximum, field_difference(
                &left->psi_syz_y, &right->psi_syz_y));
        maximum = fmax(maximum, field_difference(
                &left->psi_vzx, &right->psi_vzx));
        maximum = fmax(maximum, field_difference(
                &left->psi_vzy, &right->psi_vzy));
    }
    return maximum;
}

static void forward_step(
        struct owned_state *state, int nt, float **srcpos, float **signals,
        struct field diag[6], struct field *dummy, struct field *rhoi,
        struct field *fipjp, struct field *f, struct volume *dip,
        struct volume *d, struct volume *pp, float *hc, float *bip,
        float *bjm, float *cip, float *cjm, float *K, float *Kh,
        float *a, float *ah, float *b, float *bh, float **bl, float **br,
        float **bt, float **bb, MPI_Request *request) {
    update_v_PML_SH(
            1, NX, 1, NY, nt, state->view.vz, diag[0].v, diag[1].v,
            diag[2].v, state->view.sxz, state->view.syz, dummy->v, rhoi->v,
            srcpos, signals, 1, dummy->v, hc, 0, 0, 0, K, a, b, Kh, ah,
            bh, K, a, b, Kh, ah, bh, state->view.psi_sxz_x,
            state->view.psi_syz_y);
    exchange_v_SH(state->view.vz, bl, br, bt, bb, request, request);
    if (FREE_SURF && POS[2] == 0)
        surface_elastic_SH_velocity(state->view.vz, NX, FDORDER / 2);
    update_s_visc_PML_SH(
            1, NX, 1, NY, state->view.vz, diag[3].v, diag[4].v,
            state->view.syz, state->view.sxz, dummy->v, dummy->v,
            dummy->v, hc, 0, state->view.r, pp->v, state->view.q,
            fipjp->v, f->v, dummy->v, bip, bjm, cip, cjm, d->v, pp->v,
            dip->v, K, a, b, Kh, ah, bh, K, a, b, Kh, ah, bh,
            state->view.psi_vzx, state->view.psi_vzy, NULL, 0);
    if (FREE_SURF && POS[2] == 0)
        surface_elastic_SH_stress(state->view.syz, NX, FDORDER / 2);
    exchange_s_SH(state->view.sxz, state->view.syz,
                  bl, br, bt, bb, request, request);
}

int main(int argc, char **argv) {
    struct owned_state input, forward_state, dual, terminal, initial, scratch;
    struct owned_state direct_work, direct_prev, direct_scratch;
    struct field rhoi, fipjp, f, diag[6], dummy;
    struct volume dip, d, pp;
    struct visco_sh_full_step_config cfg, direct_cfg;
    float *hc, *bip, *bjm, *cip, *cjm, *K, *Kh, *a, *ah, *b, *bh;
    float **srcpos, **signals, *src_storage, *signal_storage;
    float **bl, **br, **bt, **bb, *sl, *sr, *st, *sb;
    double *bar_receiver_series, *bar_signal_series, *direct_signal;
    float *receiver_series;
    MPI_Request request[4];
    FILE *out;
    char path[4096];
    const char *mode;
    int rank, size, h, i, j, l, k, n, nsteps, impulse, status;
    int source_i, source_j, receiver_i, receiver_j, precondition_only;
    int rec_x[1], rec_y[1], src_x[1], src_y[1], src_type[1];
    double lhs_local, rhs_local, lhs, rhs, residual, structural = 0.0;
    double source_modified = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MYID = rank;
    FP = stderr;
    if (argc != 14) {
        if (rank == 0)
            fprintf(stderr, "usage: npx npy boundary fs fd L fw nx ny N mode impulse outdir\n");
        MPI_Abort(MPI_COMM_WORLD, 141);
    }
    NPROCX = atoi(argv[1]); NPROCY = atoi(argv[2]); BOUNDARY = atoi(argv[3]);
    FREE_SURF = atoi(argv[4]); FDORDER = atoi(argv[5]); L = atoi(argv[6]);
    FW = atoi(argv[7]); NX = atoi(argv[8]); NY = atoi(argv[9]);
    nsteps = atoi(argv[10]); mode = argv[11]; impulse = atoi(argv[12]);
    precondition_only = !strcmp(mode, "precondition");
    if ((size != NPROCX * NPROCY) || (nsteps < 1))
        MPI_Abort(MPI_COMM_WORLD, 142);
    topology(rank);
    h = FDORDER / 2; DT = 0.0013f; DH = 7.5f;
    GRAD_FORM = 2; ADJ_SIGN = 1; MODE = 0; QUELLTYPB = 1;

    input = state_new(h, FW, L); forward_state = state_new(h, FW, L);
    dual = state_new(h, FW, L); terminal = state_new(h, FW, L);
    initial = state_new(h, FW, L); scratch = state_new(h, FW, L);
    direct_work = state_new(h, FW, L); direct_prev = state_new(h, FW, L);
    direct_scratch = state_new(h, FW, L);
    state_initialize(&input, rank, 0, FW, L);
    state_copy(&forward_state, &input, FW);
    if (strcmp(mode, "receiver") && strcmp(mode, "impulse"))
        state_initialize(&dual, rank, 1, FW, L);
    else
        state_zero(&dual);
    state_copy(&terminal, &dual, FW);

    rhoi = field_new(-h, NY+h+1, 1-h, NX+h);
    fipjp = field_new(-h, NY+h+1, 1-h, NX+h);
    f = field_new(-h, NY+h+1, 1-h, NX+h);
    dummy = field_new(-h, NY+h+1, 1-h, NX+h);
    dip = volume_new(-h, NY+h+1, 1-h, NX+h, L);
    d = volume_new(-h, NY+h+1, 1-h, NX+h, L);
    pp = volume_new(-h, NY+h+1, 1-h, NX+h, L);
    for (j = -h; j <= NY+h+1; ++j)
        for (i = 1-h; i <= NX+h; ++i) {
            rhoi.v[j][i] = (float)(0.00043 + 0.000002*rank + 0.0000003*j);
            fipjp.v[j][i] = (float)(0.0047 + 0.00001*i + 0.000006*j);
            f.v[j][i] = (float)(0.0042 + 0.000008*i + 0.000004*j);
            for (l = 1; l <= L; ++l) {
                dip.v[j][i][l] = (float)(0.16 + 0.004*(l-1) + 0.0002*i);
                d.v[j][i][l] = (float)(0.14 + 0.003*(l-1) + 0.0002*j);
            }
        }
    hc = (float *)calloc((size_t)h+1, sizeof(float));
    bip = (float *)calloc((size_t)L+1, sizeof(float));
    bjm = (float *)calloc((size_t)L+1, sizeof(float));
    cip = (float *)calloc((size_t)L+1, sizeof(float));
    cjm = (float *)calloc((size_t)L+1, sizeof(float));
    { double values[6] = {1.0,-0.041,0.007,-0.0014,0.00031,-0.00007};
      for (k = 1; k <= h; ++k) hc[k] = (float)values[k-1]; }
    for (l = 1; l <= L; ++l) {
        bip[l] = (float)(0.79 + 0.025*(l-1));
        bjm[l] = (float)(0.77 + 0.021*(l-1));
        cip[l] = (float)(0.91 - 0.018*(l-1));
        cjm[l] = (float)(0.89 - 0.015*(l-1));
    }
    K = (float *)calloc((size_t)2*FW+1, sizeof(float));
    Kh = (float *)calloc((size_t)2*FW+1, sizeof(float));
    a = (float *)calloc((size_t)2*FW+1, sizeof(float));
    ah = (float *)calloc((size_t)2*FW+1, sizeof(float));
    b = (float *)calloc((size_t)2*FW+1, sizeof(float));
    bh = (float *)calloc((size_t)2*FW+1, sizeof(float));
    for (k = 1; k <= 2*FW; ++k) {
        K[k]=(float)(1.11+0.017*k); Kh[k]=(float)(1.08+0.013*k);
        a[k]=(float)(-0.031-0.001*k); ah[k]=(float)(-0.027-0.0008*k);
        b[k]=(float)(0.82+0.006*k); bh[k]=(float)(0.84+0.005*k);
    }
    for (k = 0; k < 6; ++k) diag[k] = field_new(-h,NY+h+1,1-h,NX+h);
    srcpos = buffer_new(8, 1, &src_storage);
    signals = buffer_new(1, nsteps, &signal_storage);
    source_i = 3 + rank%2; source_j = 4 + rank%3;
    receiver_i = 6 + rank%3; receiver_j = 5 + rank%2;
    srcpos[1][1]=(float)source_i; srcpos[2][1]=(float)source_j;
    srcpos[8][1]=1.0f;
    for (n = 0; n < nsteps; ++n)
        signals[1][n+1]=(float)(0.021+0.003*rank+0.0017*n);
    bl=buffer_new(NY,2*(h+1),&sl); br=buffer_new(NY,2*(h+1),&sr);
    bt=buffer_new(NX,2*(h+1),&st); bb=buffer_new(NX,2*(h+1),&sb);
    receiver_series = (float *)calloc((size_t)nsteps, sizeof(float));
    bar_receiver_series = (double *)calloc((size_t)nsteps, sizeof(double));
    bar_signal_series = (double *)calloc((size_t)nsteps, sizeof(double));
    direct_signal = (double *)calloc((size_t)nsteps, sizeof(double));
    for (n = 0; n < nsteps; ++n)
        bar_signal_series[n] = 73.25 + 0.375 * rank + 0.625 * n;
    for (n = 0; n < nsteps; ++n) {
        double value = -0.17 + 0.019*rank - 0.0023*n;
        if (!strcmp(mode, "terminal")) value = 0.0;
        if (!strcmp(mode, "impulse") && n != impulse) value = 0.0;
        bar_receiver_series[n] = value;
    }
    for (n = 0; n < nsteps; ++n) {
        forward_step(&forward_state, n+1, srcpos, signals, diag, &dummy,
                     &rhoi, &fipjp, &f, &dip, &d, &pp, hc, bip, bjm, cip,
                     cjm, K, Kh, a, ah, b, bh, bl, br, bt, bb, request);
        receiver_series[n] = forward_state.view.vz[receiver_j][receiver_i];
    }
    rec_x[0]=receiver_i; rec_y[0]=receiver_j;
    src_x[0]=source_i; src_y[0]=source_j; src_type[0]=1;
    memset(&cfg, 0, sizeof(cfg));
    cfg.nx=NX; cfg.ny=NY; cfg.fdorder=FDORDER; cfg.mechanisms=L;
    cfg.fw=FW; cfg.free_surface=FREE_SURF; cfg.boundary=BOUNDARY;
    memcpy(cfg.pos,POS,sizeof(POS)); memcpy(cfg.index,INDEX,sizeof(INDEX));
    cfg.nproc_x=NPROCX; cfg.nproc_y=NPROCY; cfg.dt=DT; cfg.dh=DH;
    cfg.hc=hc; cfg.rhoi=rhoi.v; cfg.fipjp=fipjp.v; cfg.f=f.v;
    cfg.bip=bip; cfg.bjm=bjm; cfg.cip=cip; cfg.cjm=cjm;
    cfg.dip=dip.v; cfg.d=d.v; cfg.K_x=K; cfg.a_x=a; cfg.b_x=b;
    cfg.K_x_half=Kh; cfg.a_x_half=ah; cfg.b_x_half=bh;
    cfg.K_y=K; cfg.a_y=a; cfg.b_y=b; cfg.K_y_half=Kh;
    cfg.a_y_half=ah; cfg.b_y_half=bh; cfg.nrec=1;
    cfg.rec_x=rec_x; cfg.rec_y=rec_y; cfg.nsrc=1; cfg.src_x=src_x;
    cfg.src_y=src_y; cfg.source_type=src_type; cfg.comm=MPI_COMM_WORLD;

    state_copy(&direct_work, &terminal, FW);
    status = visco_sh_reverse_time_adjoint(
            &cfg, nsteps, bar_receiver_series, &terminal.view,
            &initial.view, &scratch.view, bar_signal_series);
    if (precondition_only) {
        double consumed = state_difference(&terminal, &direct_work);
        for (n = 0; n < nsteps; ++n)
            source_modified = fmax(source_modified, fabs(
                    bar_signal_series[n] -
                    (73.25 + 0.375 * rank + 0.625 * n)));
        if ((status != -2) || (consumed != 0.0) ||
                (source_modified != 0.0)) MPI_Abort(MPI_COMM_WORLD, 143);
        if (rank == 0)
            printf("{\"precondition_status\":%d,\"source_modified\":%.17g,\"state_consumed\":%.17g}\n",
                   status, source_modified, consumed);
        MPI_Finalize();
        return 0;
    }
    if (status != 0) MPI_Abort(MPI_COMM_WORLD, 144);

    if (nsteps == 1) {
        direct_cfg = cfg; direct_cfg.bar_receiver = bar_receiver_series;
        status = visco_sh_full_state_adjoint_step(
                &direct_cfg, &direct_work.view, &direct_prev.view, direct_signal);
        if (status != 0) MPI_Abort(MPI_COMM_WORLD, 145);
        structural = fmax(state_difference(&initial, &direct_prev),
                          fabs(bar_signal_series[0]-direct_signal[0]) /
                          fmax(fabs(direct_signal[0]), 1.0e-300));
    } else if (nsteps == 2) {
        direct_cfg = cfg; direct_cfg.bar_receiver = bar_receiver_series + 1;
        status = visco_sh_full_state_adjoint_step(
                &direct_cfg, &direct_work.view, &direct_scratch.view,
                direct_signal + 1);
        if (status != 0) MPI_Abort(MPI_COMM_WORLD, 146);
        direct_cfg.bar_receiver = bar_receiver_series;
        status = visco_sh_full_state_adjoint_step(
                &direct_cfg, &direct_scratch.view, &direct_prev.view,
                direct_signal);
        if (status != 0) MPI_Abort(MPI_COMM_WORLD, 147);
        structural = state_difference(&initial, &direct_prev);
        for (n = 0; n < 2; ++n)
            structural = fmax(structural,
                    fabs(bar_signal_series[n]-direct_signal[n]) /
                    fmax(fabs(direct_signal[n]), 1.0e-300));
    }

    lhs_local = state_dot(&forward_state, &dual, FW);
    rhs_local = state_dot(&input, &initial, FW);
    for (n = 0; n < nsteps; ++n) {
        lhs_local += (double)receiver_series[n] * bar_receiver_series[n];
        rhs_local += (double)signals[1][n+1] * bar_signal_series[n];
    }
    MPI_Reduce(&lhs_local,&lhs,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&rhs_local,&rhs,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
    snprintf(path,sizeof(path),"%s/rank_%d.bin",argv[13],rank);
    out=fopen(path,"wb"); if(!out) MPI_Abort(MPI_COMM_WORLD,148);
    write_state(out,&forward_state,FW); write_state(out,&initial,FW);
    fwrite(receiver_series,sizeof(float),(size_t)nsteps,out);
    for(n=0;n<nsteps;++n){float value=(float)bar_signal_series[n];fwrite(&value,sizeof(float),1,out);}
    fclose(out);
    if(rank==0){
        residual=fabs(lhs-rhs)/fmax(fmax(fabs(lhs),fabs(rhs)),1.0e-300);
        printf("{\"lhs\":%.17g,\"rhs\":%.17g,\"dot_residual\":%.17g,\"structural_relative\":%.17g,\"nsteps\":%d}\n",
               lhs,rhs,residual,structural,nsteps);
    }
    MPI_Finalize();
    return 0;
}
