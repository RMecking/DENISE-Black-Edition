/*------------------------------------------------------------------------
 * Assemble the exact elastic SH INVMAT1=1 shot gradient.
 *
 * Native X/Y material correlations live on the same staggered locations as
 * sxz/syz. They remain separate through the material-map VJP and their halos
 * are exchanged before cell-centred assembly.
 * ----------------------------------------------------------------------*/

#include "fd.h"

static double cell_mu(float rho, float vs){
    return (double)rho * (double)vs * (double)vs;
}

void assemble_gradSH_exact(struct fwiSH *fwiSH, struct matSH *matSH,
                           struct mpiPSV *mpiPSV, MPI_Request *req_send,
                           MPI_Request *req_rec){

    extern int NX, NY, NXG, NYG, POS[3], BOUNDARY;
    extern int INVMAT1, GRAD_FORM, MODE, DTINV;
    extern float DT;
    int i, j, global_i, global_j;
    double rho, vs, g_native, g_material, g_rho_material;
    double a, b, denominator;

    if((MODE!=1)||(INVMAT1!=1)) return;

    /* These arrays occupy the sxz/syz native edges, so exchange_s_SH gives
     * each PE the -x/-y sensitivities required by its boundary cells. */
    exchange_s_SH((*fwiSH).waveconv_u_x_shot,
                  (*fwiSH).waveconv_u_y_shot,
                  (*mpiPSV).bufferlef_to_rig,
                  (*mpiPSV).bufferrig_to_lef,
                  (*mpiPSV).buffertop_to_bot,
                  (*mpiPSV).bufferbot_to_top,
                  req_send,req_rec);

    for(i=1;i<=NX;i++){
        global_i=POS[1]*NX+i;
        for(j=1;j<=NY;j++){
            global_j=POS[2]*NY+j;
            rho=(*matSH).prho[j][i];
            vs=(*matSH).pu[j][i];
            g_material=0.0;

            if((rho>0.0)&&(vs>0.0)){
                if(GRAD_FORM==2){
                    /* Form 2 stores the forward strain increment including DT.
                     * Its native derivative is -DTINV*C/mu_edge. */
                    if(BOUNDARY || (global_i<NXG)){
                        a=cell_mu((*matSH).prho[j][i],(*matSH).pu[j][i]);
                        b=cell_mu((*matSH).prho[j][i+1],(*matSH).pu[j][i+1]);
                        denominator=(a+b)*(a+b);
                        if((a>0.0)&&(b>0.0)&&(denominator>0.0)){
                            g_native=-(double)DTINV*
                                (*fwiSH).waveconv_u_x_shot[j][i]*
                                (a+b)/(2.0*a*b);
                            g_material+=2.0*b*b/denominator*g_native;
                        }
                    }
                    if(BOUNDARY || (global_i>1)){
                        a=cell_mu((*matSH).prho[j][i-1],(*matSH).pu[j][i-1]);
                        b=cell_mu((*matSH).prho[j][i],(*matSH).pu[j][i]);
                        denominator=(a+b)*(a+b);
                        if((a>0.0)&&(b>0.0)&&(denominator>0.0)){
                            g_native=-(double)DTINV*
                                (*fwiSH).waveconv_u_x_shot[j][i-1]*
                                (a+b)/(2.0*a*b);
                            g_material+=2.0*a*a/denominator*g_native;
                        }
                    }
                    if(BOUNDARY || (global_j<NYG)){
                        a=cell_mu((*matSH).prho[j][i],(*matSH).pu[j][i]);
                        b=cell_mu((*matSH).prho[j+1][i],(*matSH).pu[j+1][i]);
                        denominator=(a+b)*(a+b);
                        if((a>0.0)&&(b>0.0)&&(denominator>0.0)){
                            g_native=-(double)DTINV*
                                (*fwiSH).waveconv_u_y_shot[j][i]*
                                (a+b)/(2.0*a*b);
                            g_material+=2.0*b*b/denominator*g_native;
                        }
                    }
                    if(BOUNDARY || (global_j>1)){
                        a=cell_mu((*matSH).prho[j-1][i],(*matSH).pu[j-1][i]);
                        b=cell_mu((*matSH).prho[j][i],(*matSH).pu[j][i]);
                        denominator=(a+b)*(a+b);
                        if((a>0.0)&&(b>0.0)&&(denominator>0.0)){
                            g_native=-(double)DTINV*
                                (*fwiSH).waveconv_u_y_shot[j-1][i]*
                                (a+b)/(2.0*a*b);
                            g_material+=2.0*a*a/denominator*g_native;
                        }
                    }
                    (*fwiSH).waveconv_u_shot[j][i]=(float)(
                        2.0*rho*vs*g_material);
                    g_rho_material=vs*vs*g_material;
                }else{
                    /* Form 1 maps cell compliance arithmetically to edges. */
                    if(BOUNDARY || (global_i<NXG)){
                        g_material+=0.5*(double)DT*(double)DTINV*
                            (*fwiSH).waveconv_u_x_shot[j][i];
                    }
                    if(BOUNDARY || (global_i>1)){
                        g_material+=0.5*(double)DT*(double)DTINV*
                            (*fwiSH).waveconv_u_x_shot[j][i-1];
                    }
                    if(BOUNDARY || (global_j<NYG)){
                        g_material+=0.5*(double)DT*(double)DTINV*
                            (*fwiSH).waveconv_u_y_shot[j][i];
                    }
                    if(BOUNDARY || (global_j>1)){
                        g_material+=0.5*(double)DT*(double)DTINV*
                            (*fwiSH).waveconv_u_y_shot[j-1][i];
                    }
                    (*fwiSH).waveconv_u_shot[j][i]=(float)(
                        -2.0*g_material/(rho*vs*vs*vs));
                    g_rho_material=-g_material/(rho*rho*vs*vs);
                }

                (*fwiSH).waveconv_rho_shot[j][i]=(float)(
                    -(double)DT*(double)DTINV*
                    (*fwiSH).waveconv_rho_shot[j][i]+g_rho_material);
            }else{
                (*fwiSH).waveconv_u_shot[j][i]=0.0;
                (*fwiSH).waveconv_rho_shot[j][i]=0.0;
            }
        }
    }
}
