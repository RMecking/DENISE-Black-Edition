/* Actual production-forward harness for the M6.3c-7a observable trajectory. */

#define main m63c_full_state_step_unused_main
#include "m63c_full_state_step_harness.c"
#undef main

static int state_equal_exact(
        const struct owned_state *left, const struct owned_state *right, int fw) {
    size_t field_bytes = (size_t)left->vz.nrows * left->vz.ncols * sizeof(float);
    size_t volume_bytes = (size_t)left->r.nrows * left->r.ncols *
            (left->r.mechanisms + 1) * sizeof(float);
    if (memcmp(left->vz.data, right->vz.data, field_bytes) ||
            memcmp(left->sxz.data, right->sxz.data, field_bytes) ||
            memcmp(left->syz.data, right->syz.data, field_bytes) ||
            memcmp(left->r.data, right->r.data, volume_bytes) ||
            memcmp(left->q.data, right->q.data, volume_bytes)) return 0;
    if (fw && (memcmp(left->psi_sxz_x.data, right->psi_sxz_x.data,
                         (size_t)left->psi_sxz_x.nrows *
                                 left->psi_sxz_x.ncols * sizeof(float)) ||
            memcmp(left->psi_syz_y.data, right->psi_syz_y.data,
                         (size_t)left->psi_syz_y.nrows *
                                 left->psi_syz_y.ncols * sizeof(float)) ||
            memcmp(left->psi_vzx.data, right->psi_vzx.data,
                         (size_t)left->psi_vzx.nrows *
                                 left->psi_vzx.ncols * sizeof(float)) ||
            memcmp(left->psi_vzy.data, right->psi_vzy.data,
                         (size_t)left->psi_vzy.nrows *
                                 left->psi_vzy.ncols * sizeof(float)))) return 0;
    return 1;
}

static void run_forward_step(
        struct owned_state *state, int nt, float **srcpos, float **signals,
        struct field *diag, struct field *dummy, struct field *rhoi,
        struct field *fipjp, struct field *f, struct volume *dip,
        struct volume *d, struct volume *pp, float *hc, float *bip,
        float *bjm, float *cip, float *cjm, float *K, float *Kh,
        float *a, float *ah, float *b, float *bh, float **bl, float **br,
        float **bt, float **bb, MPI_Request *req) {
    int h = FDORDER / 2;
    update_v_PML_SH(
            1, NX, 1, NY, nt, state->view.vz, diag[0].v, diag[1].v,
            diag[2].v, state->view.sxz, state->view.syz, dummy->v, rhoi->v,
            srcpos, signals, 1, dummy->v, hc, 0, 0, 0, K, a, b, Kh, ah, bh,
            K, a, b, Kh, ah, bh, state->view.psi_sxz_x,
            state->view.psi_syz_y);
    exchange_v_SH(state->view.vz, bl, br, bt, bb, req, req);
    if (FREE_SURF && POS[2] == 0)
        surface_elastic_SH_velocity(state->view.vz, NX, h);
    update_s_visc_PML_SH(
            1, NX, 1, NY, state->view.vz, diag[3].v, diag[4].v,
            state->view.syz, state->view.sxz, dummy->v, dummy->v, dummy->v,
            hc, 0, state->view.r, pp->v, state->view.q, fipjp->v, f->v,
            dummy->v, bip, bjm, cip, cjm, d->v, pp->v, dip->v, K, a, b,
            Kh, ah, bh, K, a, b, Kh, ah, bh, state->view.psi_vzx,
            state->view.psi_vzy, NULL, 0);
    if (FREE_SURF && POS[2] == 0)
        surface_elastic_SH_stress(state->view.syz, NX, h);
    exchange_s_SH(state->view.sxz, state->view.syz, bl, br, bt, bb, req, req);
}

int main(int argc, char **argv) {
    struct owned_state initial, captured, plain;
    struct field rhoi, fipjp, f, diag[6], dummy;
    struct volume dip, d, pp;
    struct visco_sh_material_observable_trajectory trajectory;
    float *hc, *bip, *bjm, *cip, *cjm, *K, *Kh, *a, *ah, *b, *bh;
    float **srcpos, **signals, *src_storage, *signal_storage;
    float **bl, **br, **bt, **bb, *sl, *sr, *st, *sb;
    float *captured_receiver, *plain_receiver;
    MPI_Request req[4];
    FILE *out;
    char path[4096];
    int rank, size, h, i, j, l, k, n, nsteps, dtinv, status;
    int source_i, source_j, receiver_i, receiver_j, passive;
    size_t active_qsum_calls, active_strain_calls;
    size_t inactive_qsum_calls, inactive_strain_calls;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MYID = rank;
    FP = stderr;
    if (argc != 13) {
        if (rank == 0)
            fprintf(stderr, "usage: npx npy boundary fs fd L fw nx ny nsteps dtinv outdir\n");
        MPI_Abort(MPI_COMM_WORLD, 141);
    }
    NPROCX = atoi(argv[1]);
    NPROCY = atoi(argv[2]);
    BOUNDARY = atoi(argv[3]);
    FREE_SURF = atoi(argv[4]);
    FDORDER = atoi(argv[5]);
    L = atoi(argv[6]);
    FW = atoi(argv[7]);
    NX = atoi(argv[8]);
    NY = atoi(argv[9]);
    nsteps = atoi(argv[10]);
    dtinv = atoi(argv[11]);
    if (size != NPROCX * NPROCY) MPI_Abort(MPI_COMM_WORLD, 142);
    topology(rank);
    h = FDORDER / 2;
    DT = 0.0013f;
    DH = 7.5f;
    GRAD_FORM = 2;
    ADJ_SIGN = 1;
    MODE = 0;
    QUELLTYPB = 1;

    memset(&trajectory, 0, sizeof(trajectory));
    status = visco_sh_material_observable_trajectory_init(
            &trajectory, NX, NY, nsteps, dtinv, FW, FREE_SURF, BOUNDARY,
            NPROCX, NPROCY);
    if (status != 0) {
        if (rank == 0) printf("{\"precondition_status\":%d}\n", status);
        MPI_Finalize();
        return 0;
    }

    initial = state_new(h, FW, L);
    captured = state_new(h, FW, L);
    plain = state_new(h, FW, L);
    state_initialize(&initial, rank, 0, FW, L);
    state_copy(&captured, &initial, FW);
    state_copy(&plain, &initial, FW);
    rhoi = field_new(-h, NY+h+1, 1-h, NX+h);
    fipjp = field_new(-h, NY+h+1, 1-h, NX+h);
    f = field_new(-h, NY+h+1, 1-h, NX+h);
    dummy = field_new(-h, NY+h+1, 1-h, NX+h);
    dip = volume_new(-h, NY+h+1, 1-h, NX+h, L);
    d = volume_new(-h, NY+h+1, 1-h, NX+h, L);
    pp = volume_new(-h, NY+h+1, 1-h, NX+h, L);
    for (j=-h; j<=NY+h+1; ++j) for (i=1-h; i<=NX+h; ++i) {
        rhoi.v[j][i] = (float)(0.00043 + 0.000002*rank + 0.0000003*j);
        fipjp.v[j][i] = (float)(0.0047 + 0.00001*i + 0.000006*j);
        f.v[j][i] = (float)(0.0042 + 0.000008*i + 0.000004*j);
        for (l=1; l<=L; ++l) {
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
      for (k=1; k<=h; ++k) hc[k] = (float)values[k-1]; }
    for (l=1; l<=L; ++l) {
        bip[l]=(float)(0.79+0.025*(l-1)); bjm[l]=(float)(0.77+0.021*(l-1));
        cip[l]=(float)(0.91-0.018*(l-1)); cjm[l]=(float)(0.89-0.015*(l-1));
    }
    K=(float *)calloc((size_t)2*FW+1,sizeof(float));
    Kh=(float *)calloc((size_t)2*FW+1,sizeof(float));
    a=(float *)calloc((size_t)2*FW+1,sizeof(float));
    ah=(float *)calloc((size_t)2*FW+1,sizeof(float));
    b=(float *)calloc((size_t)2*FW+1,sizeof(float));
    bh=(float *)calloc((size_t)2*FW+1,sizeof(float));
    for (k=1; k<=2*FW; ++k) {
        K[k]=(float)(1.11+0.017*k); Kh[k]=(float)(1.08+0.013*k);
        a[k]=(float)(-0.031-0.001*k); ah[k]=(float)(-0.027-0.0008*k);
        b[k]=(float)(0.82+0.006*k); bh[k]=(float)(0.84+0.005*k);
    }
    for (k=0; k<6; ++k) diag[k]=field_new(-h,NY+h+1,1-h,NX+h);
    srcpos=buffer_new(8,1,&src_storage);
    signals=buffer_new(1,nsteps,&signal_storage);
    source_i=3+rank%2; source_j=4+rank%3;
    receiver_i=6+rank%3; receiver_j=5+rank%2;
    srcpos[1][1]=(float)source_i; srcpos[2][1]=(float)source_j;
    srcpos[8][1]=1.0f;
    for (n=0; n<nsteps; ++n)
        signals[1][n+1]=(float)((0.013+0.0021*rank)*(n+1)+(n%2?0.004:-0.003));
    bl=buffer_new(NY,2*(h+1),&sl); br=buffer_new(NY,2*(h+1),&sr);
    bt=buffer_new(NX,2*(h+1),&st); bb=buffer_new(NX,2*(h+1),&sb);
    captured_receiver=(float *)calloc((size_t)nsteps,sizeof(float));
    plain_receiver=(float *)calloc((size_t)nsteps,sizeof(float));

    for (n=0; n<nsteps; ++n) {
        if (visco_sh_material_observable_begin_step(&trajectory,n) != 0)
            MPI_Abort(MPI_COMM_WORLD,143);
        run_forward_step(&captured,n+1,srcpos,signals,diag,&dummy,&rhoi,&fipjp,
                &f,&dip,&d,&pp,hc,bip,bjm,cip,cjm,K,Kh,a,ah,b,bh,bl,br,bt,bb,req);
        visco_sh_material_observable_end_step();
        captured_receiver[n]=captured.view.vz[receiver_j][receiver_i];
    }
    active_qsum_calls=visco_sh_material_observable_test_qsum_count();
    active_strain_calls=visco_sh_material_observable_test_strain_count();
    visco_sh_material_observable_test_reset_counts();
    for (n=0; n<nsteps; ++n) {
        run_forward_step(&plain,n+1,srcpos,signals,diag,&dummy,&rhoi,&fipjp,&f,
                &dip,&d,&pp,hc,bip,bjm,cip,cjm,K,Kh,a,ah,b,bh,bl,br,bt,bb,req);
        plain_receiver[n]=plain.view.vz[receiver_j][receiver_i];
    }
    inactive_qsum_calls=visco_sh_material_observable_test_qsum_count();
    inactive_strain_calls=visco_sh_material_observable_test_strain_count();
    passive=state_equal_exact(&captured,&plain,FW) &&
            !memcmp(captured_receiver,plain_receiver,(size_t)nsteps*sizeof(float));

    snprintf(path,sizeof(path),"%s/rank_%d.bin",argv[12],rank);
    out=fopen(path,"wb");
    if(!out) MPI_Abort(MPI_COMM_WORLD,144);
    for(n=0;n<nsteps;++n){
        for(j=1;j<=NY;++j)fwrite(&trajectory.steps[n].qsum[j][1],sizeof(float),(size_t)NX,out);
        for(j=1;j<=NY;++j)fwrite(&trajectory.steps[n].strain_x[j][1],sizeof(float),(size_t)NX,out);
        for(j=1;j<=NY;++j)fwrite(&trajectory.steps[n].strain_y[j][1],sizeof(float),(size_t)NX,out);
    }
    fclose(out);
    if(rank==0)printf(
            "{\"passive_exact\":%s,\"nsteps\":%d,"
            "\"active_qsum_calls\":%zu,\"active_strain_calls\":%zu,"
            "\"inactive_qsum_calls\":%zu,\"inactive_strain_calls\":%zu}\n",
            passive?"true":"false",nsteps,active_qsum_calls,
            active_strain_calls,inactive_qsum_calls,inactive_strain_calls);

    free(captured_receiver);free(plain_receiver);
    buffer_free(bl,sl);buffer_free(br,sr);buffer_free(bt,st);buffer_free(bb,sb);
    buffer_free(srcpos,src_storage);buffer_free(signals,signal_storage);
    for(k=0;k<6;++k)field_free(&diag[k]);field_free(&rhoi);field_free(&fipjp);
    field_free(&f);field_free(&dummy);volume_free(&dip);volume_free(&d);volume_free(&pp);
    free(hc);free(bip);free(bjm);free(cip);free(cjm);free(K);free(Kh);free(a);free(ah);free(b);free(bh);
    state_free(&initial,FW);state_free(&captured,FW);state_free(&plain,FW);
    visco_sh_material_observable_trajectory_release(&trajectory);
    MPI_Finalize();
    return passive ? 0 : 145;
}
