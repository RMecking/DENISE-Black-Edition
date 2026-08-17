/*------------------------------------------------------------------------
 * Assemble the exact elastic PSV INVMAT1=1 shot gradient.
 *
 * Normal stresses are cell-centred, shear stress and mu_xy are located at
 * cell corners, and inverse densities Rx/Ry live on their native faces.
 * The native correlations remain separate until the corresponding spatial
 * material-map transposes have been applied.
 * ----------------------------------------------------------------------*/

#include "fd.h"

static double cell_mu(float rho, float vs){
    return (double)rho*(double)vs*(double)vs;
}

static double corner_mu(float **rho, float **vs, int j, int i){
    double m00=cell_mu(rho[j][i],vs[j][i]);
    double m10=cell_mu(rho[j][i+1],vs[j][i+1]);
    double m01=cell_mu(rho[j+1][i],vs[j+1][i]);
    double m11=cell_mu(rho[j+1][i+1],vs[j+1][i+1]);
    if((m00<=0.0)||(m10<=0.0)||(m01<=0.0)||(m11<=0.0)) return 0.0;
    return 4.0/(1.0/m00+1.0/m10+1.0/m01+1.0/m11);
}

static double corner_cell_weight(float **rho, float **vs,
                                 int corner_j, int corner_i,
                                 int cell_j, int cell_i){
    double mu_corner=corner_mu(rho,vs,corner_j,corner_i);
    double mu_cell=cell_mu(rho[cell_j][cell_i],vs[cell_j][cell_i]);
    if((mu_corner<=0.0)||(mu_cell<=0.0)) return 0.0;
    return mu_corner*mu_corner/(4.0*mu_cell*mu_cell);
}

static double inverse_density_x(float **rho, int j, int i){
    double denominator=(double)rho[j][i]+(double)rho[j][i+1];
    return (denominator>0.0)?2.0/denominator:0.0;
}

static double inverse_density_y(float **rho, int j, int i){
    double denominator=(double)rho[j][i]+(double)rho[j+1][i];
    return (denominator>0.0)?2.0/denominator:0.0;
}

void assemble_gradPSV_exact(struct fwiPSV *fwiPSV, struct matPSV *matPSV,
                            struct mpiPSV *mpiPSV, int iter,
                            MPI_Request *req_send, MPI_Request *req_rec){
    extern int NX,NY,NXG,NYG,POS[3],BOUNDARY,GRAD_FORM,DTINV;
    extern int INV_VP_ITER,INV_VS_ITER,INV_RHO_ITER;
    extern float DT;
    int i,j,gi,gj,cj,ci;
    double rho,vp,vs,mu,lambda,bulk;
    double csum,cdiff,cxy,g_lambda,g_mu,g_mu_corner,g_rho_mass;
    double mu_corner,rx,ry,g_rx,g_ry;

    /* The third exchange component has the sxy/corner halo layout. */
    exchange_s_PSV((*fwiPSV).waveconv_lam_exact,
                   (*fwiPSV).waveconv_mu_normal_exact,
                   (*fwiPSV).waveconv_mu_xy_exact,
                   (*mpiPSV).bufferlef_to_rig,
                   (*mpiPSV).bufferrig_to_lef,
                   (*mpiPSV).buffertop_to_bot,
                   (*mpiPSV).bufferbot_to_top,req_send,req_rec);

    /* Each face correlation is owned by the lower-index adjacent cell.  The
     * VJP needs that value in the neighbour's negative halo; the sxx/ry
     * stress exchange has precisely this ownership direction in x/y. */
    exchange_s_PSV((*fwiPSV).waveconv_rho_x_exact,
                   (*fwiPSV).waveconv_rho_y_exact,
                   (*fwiPSV).waveconv_mu_xy_exact,
                   (*mpiPSV).bufferlef_to_rig,
                   (*mpiPSV).bufferrig_to_lef,
                   (*mpiPSV).buffertop_to_bot,
                   (*mpiPSV).bufferbot_to_top,req_send,req_rec);

    for(i=1;i<=NX;i++){
        gi=POS[1]*NX+i;
        for(j=1;j<=NY;j++){
            gj=POS[2]*NY+j;
            rho=(*matPSV).prho[j][i];
            vp=(*matPSV).ppi[j][i];
            vs=(*matPSV).pu[j][i];
            mu=cell_mu((float)rho,(float)vs);
            lambda=rho*vp*vp-2.0*mu;
            bulk=lambda+mu;
            csum=(*fwiPSV).waveconv_lam_exact[j][i];
            cdiff=(*fwiPSV).waveconv_mu_normal_exact[j][i];

            g_lambda=0.0;
            g_mu=0.0;
            if((rho>0.0)&&(vp>0.0)&&(vs>0.0)&&(mu>0.0)&&(bulk>0.0)){
                /* GF1 stores stress, whereas GF2 stores stress rate.  The
                 * legacy PSV adjoint stress variable is compliance-weighted;
                 * the same -DT compliance VJP therefore closes both forms. */
                g_lambda=-(double)DT*(double)DTINV*csum/
                         (4.0*bulk*bulk);
                g_mu=-(double)DT*(double)DTINV*0.25*
                     (csum/(bulk*bulk)+cdiff/(mu*mu));

                /* H_mu^T: this cell contributes to four surrounding corners. */
                for(cj=j-1;cj<=j;cj++){
                    for(ci=i-1;ci<=i;ci++){
                        if((!BOUNDARY)&&
                           (((gi==1)&&(ci==i-1))||
                            ((gi==NXG)&&(ci==i)))) continue;
                        if(((gj==1)&&(cj==j-1))||
                           ((gj==NYG)&&(cj==j))) continue;
                        mu_corner=corner_mu((*matPSV).prho,(*matPSV).pu,cj,ci);
                        cxy=(*fwiPSV).waveconv_mu_xy_exact[cj][ci];
                        if(mu_corner>0.0){
                            g_mu_corner=-(double)DT*(double)DTINV*cxy/
                                        (mu_corner*mu_corner);
                            g_mu+=corner_cell_weight((*matPSV).prho,(*matPSV).pu,
                                                     cj,ci,j,i)*g_mu_corner;
                        }
                    }
                }
            }

            /* R_x/R_y VJPs.  Keeping g_R explicit documents the exact face
             * map even though R^2 cancels algebraically for positive density. */
            g_rho_mass=0.0;
            if((BOUNDARY)||(gi<NXG)){
                rx=inverse_density_x((*matPSV).prho,j,i);
                if(rx>0.0){
                    g_rx=(double)DT*(double)DTINV*
                         (*fwiPSV).waveconv_rho_x_exact[j][i]/(rx*rx);
                    g_rho_mass+=-0.5*rx*rx*g_rx;
                }
            }
            if((BOUNDARY)||(gi>1)){
                rx=inverse_density_x((*matPSV).prho,j,i-1);
                if(rx>0.0){
                    g_rx=(double)DT*(double)DTINV*
                         (*fwiPSV).waveconv_rho_x_exact[j][i-1]/(rx*rx);
                    g_rho_mass+=-0.5*rx*rx*g_rx;
                }
            }
            if(gj<NYG){
                ry=inverse_density_y((*matPSV).prho,j,i);
                if(ry>0.0){
                    g_ry=(double)DT*(double)DTINV*
                         (*fwiPSV).waveconv_rho_y_exact[j][i]/(ry*ry);
                    g_rho_mass+=-0.5*ry*ry*g_ry;
                }
            }
            if(gj>1){
                ry=inverse_density_y((*matPSV).prho,j-1,i);
                if(ry>0.0){
                    g_ry=(double)DT*(double)DTINV*
                         (*fwiPSV).waveconv_rho_y_exact[j-1][i]/(ry*ry);
                    g_rho_mass+=-0.5*ry*ry*g_ry;
                }
            }

            (*fwiPSV).waveconv_lam[j][i]=(float)g_lambda;
            (*fwiPSV).waveconv_mu[j][i]=(float)g_mu;
            (*fwiPSV).waveconv_rho_s[j][i]=(float)g_rho_mass;
            (*fwiPSV).waveconv_shot[j][i]=(iter<INV_VP_ITER)?0.0f:
                (float)(2.0*rho*vp*g_lambda);
            (*fwiPSV).waveconv_u_shot[j][i]=(iter<INV_VS_ITER)?0.0f:
                (float)(-4.0*rho*vs*g_lambda+2.0*rho*vs*g_mu);
            (*fwiPSV).waveconv_rho_shot[j][i]=(iter<INV_RHO_ITER)?0.0f:
                (float)((vp*vp-2.0*vs*vs)*g_lambda+vs*vs*g_mu+
                        g_rho_mass);
        }
    }
}
