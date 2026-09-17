/* Raw, discrete viscoelastic P/SV physical gradient for the one-rank L=1,
 * FD4 receiver-velocity experiment.  The forward operators themselves remain
 * in update_v_PML_PSV and update_s_visc_PML_PSV; the hooks below record their
 * post-CPML derivative operands, including the velocity-update force. */
#include "fd.h"

extern int NX, NY, NT, FW, BOUNDARY, FREE_SURF, NPROCX, NPROCY, NDT;
extern int MODE, L, INVMAT1, GRAD_FORM, FDORDER, Q_PARAMETERIZATION_MODE;
extern int DTINV, LNORM;
extern float DT, DH, *FL, Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF;
extern char JACOBIAN[STRING_SIZE];

enum {VXX, VYX, VXY, VYY, FX, FY, NRECORD};
enum {GF, GG, GFC, GD, GE, GDC, GRX, GRY, NNATIVE};
enum {PSXX, PSXYX, PSXYY, PSYY, PVXX, PVYX, PVXY, PVYY, NPSI};

static struct {
    int active, step, pitch, area;
    float *record[NRECORD];
} exact;

static size_t record_index(int t, int j, int i) {
    return ((size_t)t * NY + (j - 1)) * NX + (i - 1);
}
static int cell(int j, int i) { return (j + 2) * exact.pitch + i + 2; }

int visco_psv_exact_supported(void) {
    return MODE == 1 && L == 1 && INVMAT1 == 1 && GRAD_FORM == 2 &&
           FDORDER == 4 && NPROCX == 1 && NPROCY == 1 && !FREE_SURF &&
           !BOUNDARY && FW > 0 && NDT == 1 && DTINV == 1 && LNORM == 2;
}

int visco_psv_exact_enabled(void) {
    const char *flag = getenv("DENISE_PSV_EXACT_VISCO_GRADIENT");
    return visco_psv_exact_supported() ||
           (flag && flag[0] == '1' && flag[1] == '\0');
}

void visco_psv_exact_begin(void) {
    int k;
    if (!visco_psv_exact_enabled()) return;
    if (MODE != 1 || L != 1 || INVMAT1 != 1 || GRAD_FORM != 2 ||
        FDORDER != 4 || NPROCX != 1 || NPROCY != 1 || FREE_SURF ||
        BOUNDARY || FW <= 0 || NDT != 1 || DTINV != 1 || LNORM != 2)
        err(" Exact visco PSV raw gradient supports one-rank L=1 FD4, INVMAT1=1, GRAD_FORM=2, NDT=DTINV=1, LNORM=2, CPML interior only. ");
    exact.pitch = NX + 5;
    exact.area = (NY + 5) * exact.pitch;
    for (k = 0; k < NRECORD; k++) {
        exact.record[k] = calloc((size_t)(NT + 1) * NX * NY, sizeof(float));
        if (!exact.record[k]) err(" Out of memory recording exact visco PSV forward operands. ");
    }
    exact.step = 0;
    exact.active = 1;
}

void visco_psv_exact_step(int t) { if (exact.active) exact.step = t; }

void visco_psv_exact_velocity(int j, int i, float force_x, float force_y) {
    size_t p;
    if (!exact.active) return;
    p = record_index(exact.step, j, i);
    exact.record[FX][p] = force_x;
    exact.record[FY][p] = force_y;
}

void visco_psv_exact_strain(int j, int i, float vxx, float vyx,
                            float vxy, float vyy) {
    size_t p;
    if (!exact.active) return;
    p = record_index(exact.step, j, i);
    exact.record[VXX][p] = vxx;
    exact.record[VYX][p] = vyx;
    exact.record[VXY][p] = vxy;
    exact.record[VYY][p] = vyy;
}

/* Transpose of psi'=b psi+a D, D'=D/K+psi'.  psi_adj is the
 * adjoint of the NEW psi on entry and the OLD psi on return. */
static double reverse_cpml(double corrected, double *psi_adj, int p,
                           int coordinate, int extent, float *K, float *a,
                           float *b) {
    int h;
    double combined;
    if (coordinate <= FW) h = coordinate;
    else if (coordinate >= extent - FW + 1)
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

static void write_field(const char *suffix, double *gradient) {
    char path[STRING_SIZE + 40];
    FILE *out;
    int i, j;
    float value;
    snprintf(path, sizeof(path), "%s.raw.%s", JACOBIAN, suffix);
    out = fopen(path, "wb");
    if (!out) err(" Could not open exact visco PSV raw-gradient output. ");
    for (i = 1; i <= NX; i++) for (j = 1; j <= NY; j++) {
        value = (float)gradient[cell(j,i)];
        if (fwrite(&value, sizeof(value), 1, out) != 1)
            err(" Could not write exact visco PSV raw gradient. ");
    }
    fclose(out);
}

void visco_psv_exact_finish(struct wavePSV_PML *pml, struct matPSV *mat,
                            struct fwiPSV *fwi,
                            struct seisPSV *seis, struct seisPSVfwi *data,
                            struct acq *acq, float *hc, int ntr) {
    double *avx, *avy, *asxx, *asyy, *asxy, *ar, *ap, *aq;
    double *psi[NPSI], *native[NNATIVE], *physical[5];
    struct q_tau_mapping mapping;
    int t, i, j, k, p, cx, cy;
    double eta, b, c, dt2, div, shear, ax, ay, xx, yy, xy, yx;
    double tr, tp, tq, lambda_r, lambda_p, lambda_q;
    size_t r;
    if (!exact.active) return;
    exact.active = 0;
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
    for (t=NT;t>=1;t--) {
        /* Receiver samples are recorded after stress update; velocity is
         * unchanged there. Production sample one is excluded from L2. */
        if (t>1) for (k=1;k<=ntr;k++) {
            i=acq->recpos_loc[1][k]; j=acq->recpos_loc[2][k]; p=cell(j,i);
            avx[p] += (double)seis->sectionvx[k][t]-data->sectionvxdata[k][t];
            avy[p] += (double)seis->sectionvy[k][t]-data->sectionvydata[k][t];
        }
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
            xx=reverse_cpml(ax,psi[PVXX],p,i,NX,pml->K_x,pml->a_x,pml->b_x);
            yx=reverse_cpml(yx,psi[PVYX],p,i,NX,pml->K_x_half,pml->a_x_half,pml->b_x_half);
            xy=reverse_cpml(xy,psi[PVXY],p,j,NY,pml->K_y_half,pml->a_y_half,pml->b_y_half);
            yy=reverse_cpml(ay,psi[PVYY],p,j,NY,pml->K_y,pml->a_y,pml->b_y);
            add_backward_x(avx,j,i,xx,hc,1.0/DH);
            add_forward_x(avy,j,i,yx,hc,1.0/DH);
            add_forward_y(avx,j,i,xy,hc,1.0/DH);
            add_backward_y(avy,j,i,yy,hc,1.0/DH);
        }
        /* Transpose velocity and its four independent CPML recurrences. */
        for (j=1;j<=NY;j++) for (i=1;i<=NX;i++) {
            p=cell(j,i); r=record_index(t,j,i);
            native[GRX][p]+=avx[p]*DT*exact.record[FX][r]/DH;
            native[GRY][p]+=avy[p]*DT*exact.record[FY][r]/DH;
            ax=avx[p]*DT*mat->prip[j][i]/DH;
            ay=avy[p]*DT*mat->prjp[j][i]/DH;
            xx=reverse_cpml(ax,psi[PSXX],p,i,NX,pml->K_x_half,pml->a_x_half,pml->b_x_half);
            yx=reverse_cpml(ay,psi[PSXYX],p,i,NX,pml->K_x,pml->a_x,pml->b_x);
            xy=reverse_cpml(ax,psi[PSXYY],p,j,NY,pml->K_y,pml->a_y,pml->b_y);
            yy=reverse_cpml(ay,psi[PSYY],p,j,NY,pml->K_y_half,pml->a_y_half,pml->b_y_half);
            add_forward_x(asxx,j,i,xx,hc,1.0);
            add_backward_x(asxy,j,i,yx,hc,1.0);
            add_backward_y(asxy,j,i,xy,hc,1.0);
            add_forward_y(asyy,j,i,yy,hc,1.0);
        }
    }
    init_q_tau_mapping(&mapping,Q_PARAMETERIZATION_MODE,L,FL,
                       Q_APPROX_FMIN,Q_APPROX_FMAX,Q_APPROX_DF);
    /* Local constitutive map: native f,g,d,e and corner f,dip. */
    for (j=1;j<=NY;j++) for (i=1;i<=NX;i++) {
        double rho=mat->prho[j][i], vp=mat->ppi[j][i], vs=mat->pu[j][i];
        double M=rho*vs*vs, P=rho*vp*vp, ts=mat->ptaus[j][i];
        double tp0=mat->ptaup[j][i], den_s=1.0+0.5*ts, den_p=1.0+0.5*tp0;
        double gM, gP, gts, gtp, H, T, den_c, gH, gT;
        p=cell(j,i);
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
            if (cy>NY || cx>NX) continue;
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
        if (i<NX) physical[2][cell(j,i+1)]+=
            -0.5*mat->prip[j][i]*mat->prip[j][i]*native[GRX][p];
        if (j<NY) physical[2][cell(j+1,i)]+=
            -0.5*mat->prjp[j][i]*mat->prjp[j][i]*native[GRY][p];
    }
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
    for (k=0;k<NRECORD;k++) { free(exact.record[k]); exact.record[k]=NULL; }
    for (k=0;k<NPSI;k++) free(psi[k]);
    for (k=0;k<NNATIVE;k++) free(native[k]);
    for (k=0;k<5;k++) free(physical[k]);
    free(avx); free(avy); free(asxx); free(asyy); free(asxy);
    free(ar); free(ap); free(aq);
}
