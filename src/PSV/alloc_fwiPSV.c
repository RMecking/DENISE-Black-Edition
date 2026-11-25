/*
 * Allocate memory for FWI parameters (PSV problem) 
 *
 * Daniel Koehn
 * Kiel, 21/01/2016
 */

#include "fd.h"

void alloc_fwiPSV(struct fwiPSV *fwiPSV){

        /* global variables */
	extern int NX, NY, L, FW, FDORDER, NTDTINV, NXNYI, MODE;

	/* local variables */
	int nd;

        nd = FDORDER/2 + 1;	

	(*fwiPSV).prho_old =  matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).ppi_old  =  matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).pu_old   =  matrix(-nd+1,NY+nd,-nd+1,NX+nd);

	(*fwiPSV).Vp0  =  matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).Vs0  =  matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).Rho0  =  matrix(-nd+1,NY+nd,-nd+1,NX+nd);

	(*fwiPSV).waveconv = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_lam = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_shot = matrix(-nd+1,NY+nd,-nd+1,NX+nd);

	(*fwiPSV).gradg = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).gradp = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	  
	(*fwiPSV).gradg_rho = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).gradp_rho = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_rho = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_rho_s = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_rho_shot = matrix(-nd+1,NY+nd,-nd+1,NX+nd);

	(*fwiPSV).gradg_u = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).gradp_u = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_u = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_mu = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_u_shot = matrix(-nd+1,NY+nd,-nd+1,NX+nd);

	(*fwiPSV).waveconv_lam_exact = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_mu_normal_exact = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_mu_xy_exact = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_rho_x_exact = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_rho_y_exact = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_qp_exact = matrix(-nd+1,NY+nd,-nd+1,NX+nd);
	(*fwiPSV).waveconv_qs_exact = matrix(-nd+1,NY+nd,-nd+1,NX+nd);

	/* Legacy forward-wavefield storage (5 x NX*NY*NT floats).  The exact
	 * visco-PSV FWI (FWI_PSV with L > 0, which aborts unless the exact path
	 * is supported) never reads it: it keeps its own checkpoints.  There
	 * only one element is allocated; psv() then skips the store and refuses
	 * any adjoint that would read it.  All other paths are unchanged. */
	if (MODE == 1 && L > 0) {
		(*fwiPSV).forward_prop_x =  vector(1,1);
		(*fwiPSV).forward_prop_y =  vector(1,1);
		(*fwiPSV).forward_prop_rho_x =  vector(1,1);
		(*fwiPSV).forward_prop_rho_y =  vector(1,1);
		(*fwiPSV).forward_prop_u =  vector(1,1);
		(*fwiPSV).forward_prop_length = 1;
	} else {
		(*fwiPSV).forward_prop_x =  vector(1,NXNYI*(NTDTINV));
		(*fwiPSV).forward_prop_y =  vector(1,NXNYI*(NTDTINV));
		(*fwiPSV).forward_prop_rho_x =  vector(1,NXNYI*(NTDTINV));
		(*fwiPSV).forward_prop_rho_y =  vector(1,NXNYI*(NTDTINV));
		(*fwiPSV).forward_prop_u =  vector(1,NXNYI*(NTDTINV));
		(*fwiPSV).forward_prop_length = (size_t)NXNYI*(size_t)NTDTINV;
	}

}



