/*  --------------------------------------------------------------------------
 *   Solving the visco-elastic 2D SH-forward problem by finite-differences 
 *   for a single shot 
 *
 *   mode = 0 - forward modelling only, STF estimation or FWI gradient calculation
 *   mode = 1 - adjoint modelling
 *   mode = 2 - evaluation objective function for step length estimation  
 * 
 *   
 *   D. Koehn
 *   Kiel, 13.08.2018
 *
 *  --------------------------------------------------------------------------*/

#include "fd.h"

/* ---- K-5: checkpoint and replay of the forward state of the exact visco-SH
 * gradient (docs/sh_k5_checkpoint_replay_contract.md).  The context points to
 * the structures of the forward run in progress; one exact gradient forward
 * runs at a time per process. ---- */
static struct {
    struct waveSH *wave; struct waveSH_PML *pml; struct matSH *mat;
    struct fwiSH *fwi; struct mpiPSV *mpi; struct acq *acq;
    float *hc; int nsrc_loc; MPI_Request *req_send, *req_rec;
    struct visco_sh_material_observable_trajectory *trajectory;
} sh_replay;

/* K-5 wrappers on the grid globals, for the exact driver: checkpoint size
 * of this propagator, effective segment count of a requested
 * SH_EXACT_SEGMENTS value (record 135) for the NT-step exact history, the
 * bytes of that history per process, and the storage of every following
 * exact gradient trajectory */
size_t sh_visc_replay_floats(void) {
    extern int NX, NY, FDORDER, FW, L;
    return visco_sh_material_observable_replay_floats(NX, NY, FDORDER, FW, L);
}

int sh_visc_exact_segments(int requested) {
    extern int NX, NY, NT;
    return visco_sh_material_observable_segments(requested, NX, NY, NT,
                                                 sh_visc_replay_floats());
}

double sh_visc_exact_history_bytes(int segments) {
    extern int NX, NY, NT;
    return visco_sh_material_observable_history_bytes(NX, NY, NT, segments,
                                                      sh_visc_replay_floats());
}

void sh_visc_use_exact_segments(int segments) {
    visco_sh_material_observable_request_segments(segments, sh_visc_replay_floats());
}

/* direction 1: state -> payload; 0: payload -> state (payload NULL: zero state) */
static void sh_replay_copy(float *payload, int direction) {
    extern int NX, NY, FDORDER, FW, L;
    int nd = FDORDER / 2 + 1;
    size_t field = ((size_t)NY + 2 * nd) * ((size_t)NX + 2 * nd);
    float *p = payload;
#define SH_REPLAY_BLOCK(first, count) do { \
        size_t bytes = (size_t)(count) * sizeof(float); \
        if (direction) memcpy(p, (first), bytes); \
        else if (payload != NULL) memcpy((first), p, bytes); \
        else memset((first), 0, bytes); \
        if (payload != NULL) p += (count); } while (0)
    SH_REPLAY_BLOCK(&sh_replay.wave->pvz[-nd + 1][-nd + 1], field);
    SH_REPLAY_BLOCK(&sh_replay.wave->psxz[-nd + 1][-nd + 1], field);
    SH_REPLAY_BLOCK(&sh_replay.wave->psyz[-nd + 1][-nd + 1], field);
    if (L > 0) {
        SH_REPLAY_BLOCK(&sh_replay.wave->pr[-nd + 1][-nd + 1][1], field * L);
        SH_REPLAY_BLOCK(&sh_replay.wave->pq[-nd + 1][-nd + 1][1], field * L);
    }
    if (FW > 0) {
        SH_REPLAY_BLOCK(&sh_replay.pml->psi_sxz_x[1][1], (size_t)NY * 2 * FW);
        SH_REPLAY_BLOCK(&sh_replay.pml->psi_vzx[1][1], (size_t)NY * 2 * FW);
        SH_REPLAY_BLOCK(&sh_replay.pml->psi_syz_y[1][1], (size_t)NX * 2 * FW);
        SH_REPLAY_BLOCK(&sh_replay.pml->psi_vzy[1][1], (size_t)NX * 2 * FW);
    }
#undef SH_REPLAY_BLOCK
}

static void sh_replay_save(void *context, float *payload) {
    (void)context;
    sh_replay_copy(payload, 1);
}

static void sh_replay_restore(void *context, const float *payload) {
    (void)context;
    sh_replay_copy((float *)payload, 0);
}

/* the kernel sequence of one forward time step, as in the loop below (mode 0
 * with trajectory): no source energy, no seismograms, no snapshots, no legacy
 * history; infoout only prints timings and is 0 here */
static void sh_replay_run(void *context, int first_step, int last_step) {
    extern int NX, NY, FDORDER, FREE_SURF, POS[3];
    struct waveSH *waveSH = sh_replay.wave;
    struct waveSH_PML *waveSH_PML = sh_replay.pml;
    struct matSH *matSH = sh_replay.mat;
    struct fwiSH *fwiSH = sh_replay.fwi;
    struct mpiPSV *mpiPSV = sh_replay.mpi;
    struct acq *acq = sh_replay.acq;
    float *hc = sh_replay.hc;
    MPI_Request *req_send = sh_replay.req_send, *req_rec = sh_replay.req_rec;
    int nt, nsrc_loc = sh_replay.nsrc_loc, infoout = 0, mode = 0;
    (void)context;

    for (nt = first_step + 1; nt <= last_step + 1; nt++) {
        if (visco_sh_material_observable_begin_step(sh_replay.trajectory, nt - 1) != 0)
            err(" Cannot activate viscoelastic SH material-observable step (replay) !");

		update_v_PML_SH(1, NX, 1, NY, nt, (*waveSH).pvz, (*waveSH).pvzp1, (*waveSH).pvzm1, (*waveSH).uttz, (*waveSH).psxz, (*waveSH).psyz, (*matSH).prho, (*matSH).prhoi, (*acq).srcpos_loc, (*acq).signals, nsrc_loc,
			       (*waveSH_PML).absorb_coeff,hc, infoout, 0, 0, (*waveSH_PML).K_x, (*waveSH_PML).a_x, (*waveSH_PML).b_x, (*waveSH_PML).K_x_half, (*waveSH_PML).a_x_half, (*waveSH_PML).b_x_half, (*waveSH_PML).K_y,
			       (*waveSH_PML).a_y, (*waveSH_PML).b_y, (*waveSH_PML).K_y_half, (*waveSH_PML).a_y_half, (*waveSH_PML).b_y_half, (*waveSH_PML).psi_sxz_x, (*waveSH_PML).psi_syz_y);

		exchange_v_SH((*waveSH).pvz, (*mpiPSV).bufferlef_to_rig, (*mpiPSV).bufferrig_to_lef, (*mpiPSV).buffertop_to_bot, (*mpiPSV).bufferbot_to_top, req_send, req_rec);

		if(FREE_SURF && (POS[2]==0)){
			surface_elastic_SH_velocity((*waveSH).pvz, NX, FDORDER/2);
		}

		update_s_visc_PML_SH(1, NX, 1, NY, (*waveSH).pvz, (*waveSH).uz, (*waveSH).uzx, (*waveSH).psyz, (*waveSH).psxz, (*matSH).pujp, (*matSH).puip, (*matSH).prho, hc, infoout,
				    (*waveSH).pr, (*waveSH).pp, (*waveSH).pq, (*matSH).fipjp, (*matSH).f, (*matSH).g, (*matSH).bip, (*matSH).bjm, (*matSH).cip, (*matSH).cjm, (*matSH).d,
			            (*matSH).e, (*matSH).dip, (*waveSH_PML).K_x, (*waveSH_PML).a_x, (*waveSH_PML).b_x, (*waveSH_PML).K_x_half, (*waveSH_PML).a_x_half, (*waveSH_PML).b_x_half,
        			    (*waveSH_PML).K_y, (*waveSH_PML).a_y, (*waveSH_PML).b_y, (*waveSH_PML).K_y_half, (*waveSH_PML).a_y_half, (*waveSH_PML).b_y_half,
        			    (*waveSH_PML).psi_vzx, (*waveSH_PML).psi_vzy, fwiSH, mode);

	   if(FREE_SURF && (POS[2]==0)){
		surface_elastic_SH_stress((*waveSH).psyz, NX, FDORDER/2);
	   }

	   exchange_s_SH((*waveSH).psxz,(*waveSH).psyz, (*mpiPSV).bufferlef_to_rig, (*mpiPSV).bufferrig_to_lef,
	      (*mpiPSV).buffertop_to_bot, (*mpiPSV).bufferbot_to_top, req_send, req_rec);

        visco_sh_material_observable_end_step();
    }
}

void sh_visc_with_material_trajectory(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV,
         struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc, 
         int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request * req_send, MPI_Request * req_rec,
         struct visco_sh_material_observable_trajectory *trajectory){

        /* global variables */
	extern float DT, DH, TSNAP1, TSNAP2, TSNAPINC;
	extern int MYID, FDORDER, FW, L, GRAD_FORM, FC_SPIKE_1, FC_SPIKE_2, ORDER_SPIKE;
        extern int NX, NY, FREE_SURF, BOUNDARY, MODE, QUELLTYP, QUELLTYPB, QUELLART, FDORDER;
	extern int NPROCX, NPROCY, POS[3], NDT, SEISMO, IDXI, IDYI, GRAD_FORM, DTINV;
        extern int SNAP, INVMAT1, INV_STF, EPRECOND, NTDTINV, NXNYI, NT, ADJ_SIGN;
	extern FILE *FP;

        /* local variables */
	int i,j,l,nt,lsamp,lsnap,nsnap, nd, hin1, imat, imat1, imat2, infoout;
        float tmp, tmp1, muss, lamss, P3, P5;
        float hess_rho, hess_mu, hess_ts;
	float SUMr, SUMq;
	int sh_segmented=0;   /* K-5: segmented exact history */

        nd = FDORDER/2 + 1;

	/* C8b2-a exposes the already locked C7a capture hooks only for the
	 * physical forward trajectory on which their time indexing was verified. */
	if(trajectory!=NULL){
	   if((mode!=0)||(DTINV!=1)||(trajectory->dtinv!=1)||
	      (trajectory->nx!=NX)||(trajectory->ny!=NY)||
	      (trajectory->nsteps!=NT)||(trajectory->steps==NULL)){
	      err(" Invalid viscoelastic SH material-observable trajectory !");
	   }
	   if(visco_sh_material_observable_is_active()){
	      err(" Nested viscoelastic SH material-observable trajectory !");
	   }
	}

	/*MPI_Barrier(MPI_COMM_WORLD);*/
        
        if(MYID==0){

		if((INV_STF==0)&&(mode==0)){

		   fprintf(FP,"\n==================================================================================\n");
		   fprintf(FP,"\n *****  Starting simulation (forward model) for shot %d of %d  ********** \n",ishot,nshots);
		   fprintf(FP,"\n==================================================================================\n\n");

		}


		if((INV_STF==1)&&(mode==0)){

		   fprintf(FP,"\n==================================================================================\n");
		   fprintf(FP,"\n *****  Starting simulation (STF) for shot %d of %d  ********** \n",ishot,nshots);
		   fprintf(FP,"\n==================================================================================\n\n");

		}

		if(mode==1){

		   fprintf(FP,"\n==================================================================================\n");
		   fprintf(FP,"\n *****  Starting simulation (adjoint wavefield)  ********** \n");
		   fprintf(FP,"\n==================================================================================\n\n");

		}

        }	                                                        	
		    
	/* initialize SH wavefields with zero */
	zero_denise_visc_SH(-nd+1,NY+nd,-nd+1,NX+nd, (*waveSH).pvz, (*waveSH).psxz, (*waveSH).psyz, (*waveSH).pvzm1, (*waveSH).pvzp1, 
		            (*waveSH_PML).psi_sxz_x, (*waveSH_PML).psi_syz_y, (*waveSH_PML).psi_vzx,  (*waveSH_PML).psi_vzy, 
                            (*waveSH).pr, (*waveSH).pp, (*waveSH).pq, (*fwiSH).Rxz, (*fwiSH).Ryz);
	                                                        	     
	if(trajectory!=NULL){
	   sh_replay.wave=waveSH; sh_replay.pml=waveSH_PML; sh_replay.mat=matSH;
	   sh_replay.fwi=fwiSH; sh_replay.mpi=mpiPSV; sh_replay.acq=acq; sh_replay.hc=hc;
	   sh_replay.nsrc_loc=nsrc_loc; sh_replay.req_send=req_send; sh_replay.req_rec=req_rec;
	   sh_replay.trajectory=trajectory;
	   sh_segmented=visco_sh_material_observable_set_replay(trajectory,
	         sh_replay_save,sh_replay_restore,sh_replay_run,&sh_replay,
	         sh_visc_replay_floats());
	   if(sh_segmented<0) err(" Cannot set up the viscoelastic SH replay !");
	}

	/*----------------------  loop over timesteps (forward model) ------------------*/

	lsnap=iround(TSNAP1/DT);  
	lsamp=NDT;
	nsnap=0;

        if(mode==0){
	   hin=1;
	   hin1=1;
	   imat=1;
	   imat1=1;
	   imat2=1;
	   ADJ_SIGN=1;
        }

	if(mode==2){
	   ADJ_SIGN=1;
        }

        if(mode==1){
	   hin=1;
	   hin1=1;
	   ADJ_SIGN=-1;
        }

	for (nt=1;nt<=NT;nt++){     
		        
		/* Check if simulation is still stable */
		if (isnan((*waveSH).pvz[NY/2][NX/2])) {
		   fprintf(FP,"\n Time step: %d; pvy: %f \n",nt,(*waveSH).pvz[NY/2][NX/2]);
		   err(" Simulation is unstable !");}

	   infoout = !(nt%10000);

	   if(sh_segmented && (visco_sh_material_observable_forward_boundary(trajectory,nt-1)!=0)){
	      err(" Cannot store viscoelastic SH replay checkpoint !");
	   }

	   if (MYID==0){
	      if (infoout)  fprintf(FP,"\n Computing timestep %d of %d \n",nt,NT);
	      /*time3=MPI_Wtime();*/
	   }

	   if(trajectory!=NULL){
	      if(visco_sh_material_observable_begin_step(trajectory,nt-1)!=0){
	         err(" Cannot activate viscoelastic SH material-observable step !");
	      }
	   }

	      /* particle velocity update */
              if(mode==0 || mode==2){
		update_v_PML_SH(1, NX, 1, NY, nt, (*waveSH).pvz, (*waveSH).pvzp1, (*waveSH).pvzm1, (*waveSH).uttz, (*waveSH).psxz, (*waveSH).psyz, (*matSH).prho, (*matSH).prhoi, (*acq).srcpos_loc, (*acq).signals, nsrc_loc, 
			       (*waveSH_PML).absorb_coeff,hc, infoout, 0, 0, (*waveSH_PML).K_x, (*waveSH_PML).a_x, (*waveSH_PML).b_x, (*waveSH_PML).K_x_half, (*waveSH_PML).a_x_half, (*waveSH_PML).b_x_half, (*waveSH_PML).K_y,
			       (*waveSH_PML).a_y, (*waveSH_PML).b_y, (*waveSH_PML).K_y_half, (*waveSH_PML).a_y_half, (*waveSH_PML).b_y_half, (*waveSH_PML).psi_sxz_x, (*waveSH_PML).psi_syz_y);
              }

              if(mode==1){
		update_v_PML_SH(1, NX, 1, NY, nt, (*waveSH).pvz, (*waveSH).pvzp1, (*waveSH).pvzm1, (*waveSH).uttz, (*waveSH).psxz, (*waveSH).psyz, (*matSH).prho, (*matSH).prhoi, (*acq).srcpos_loc_back, (*seisSHfwi).sectionvzdiff, ntr, 
			       (*waveSH_PML).absorb_coeff,hc, infoout, 1, 0, (*waveSH_PML).K_x, (*waveSH_PML).a_x, (*waveSH_PML).b_x, (*waveSH_PML).K_x_half, (*waveSH_PML).a_x_half, (*waveSH_PML).b_x_half, (*waveSH_PML).K_y,
			       (*waveSH_PML).a_y, (*waveSH_PML).b_y, (*waveSH_PML).K_y_half, (*waveSH_PML).a_y_half, (*waveSH_PML).b_y_half, (*waveSH_PML).psi_sxz_x, (*waveSH_PML).psi_syz_y);
              }
		                 

		                                           
		/* exchange of particle velocities between PEs */
		exchange_v_SH((*waveSH).pvz, (*mpiPSV).bufferlef_to_rig, (*mpiPSV).bufferrig_to_lef, (*mpiPSV).buffertop_to_bot, (*mpiPSV).bufferbot_to_top, req_send, req_rec);
		                                                                                                                                            	
		if(FREE_SURF && (POS[2]==0)){
			surface_elastic_SH_velocity((*waveSH).pvz, NX, FDORDER/2);
		}

                /* stress update */
		update_s_visc_PML_SH(1, NX, 1, NY, (*waveSH).pvz, (*waveSH).uz, (*waveSH).uzx, (*waveSH).psyz, (*waveSH).psxz, (*matSH).pujp, (*matSH).puip, (*matSH).prho, hc, infoout,
				    (*waveSH).pr, (*waveSH).pp, (*waveSH).pq, (*matSH).fipjp, (*matSH).f, (*matSH).g, (*matSH).bip, (*matSH).bjm, (*matSH).cip, (*matSH).cjm, (*matSH).d, 
			            (*matSH).e, (*matSH).dip, (*waveSH_PML).K_x, (*waveSH_PML).a_x, (*waveSH_PML).b_x, (*waveSH_PML).K_x_half, (*waveSH_PML).a_x_half, (*waveSH_PML).b_x_half,
        			    (*waveSH_PML).K_y, (*waveSH_PML).a_y, (*waveSH_PML).b_y, (*waveSH_PML).K_y_half, (*waveSH_PML).a_y_half, (*waveSH_PML).b_y_half,
        			    (*waveSH_PML).psi_vzx, (*waveSH_PML).psi_vzy, fwiSH, mode);

	   if(FREE_SURF && (POS[2]==0)){
		surface_elastic_SH_stress((*waveSH).psyz, NX, FDORDER/2);
	   }


	   /*if ((FREE_SURF) && (POS[2]==0)){
	   	if (L)   
			surface_visc_PML_PSV(1, (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy, (*wavePSV).pp, (*wavePSV).pq, (*matPSV).ppi, 
					    (*matPSV).pu, (*matPSV).prho, (*matPSV).ptaup, (*matPSV).ptaus, (*matPSV).etajm, (*matPSV).peta, hc, (*wavePSV_PML).K_x, (*wavePSV_PML).a_x, 
					    (*wavePSV_PML).b_x, (*wavePSV_PML).psi_vxxs);
		else  
	   		surface_elastic_PML_PSV(1, (*wavePSV).pvx, (*wavePSV).pvy, (*wavePSV).psxx, (*wavePSV).psyy, (*wavePSV).psxy, (*matPSV).ppi, (*matPSV).pu, (*matPSV).prho, hc, 
					       (*wavePSV_PML).K_x, (*wavePSV_PML).a_x, (*wavePSV_PML).b_x, (*wavePSV_PML).psi_vxxs);
	   }*/



	   /* stress exchange between PEs */
	   exchange_s_SH((*waveSH).psxz,(*waveSH).psyz, (*mpiPSV).bufferlef_to_rig, (*mpiPSV).bufferrig_to_lef, 
	      (*mpiPSV).buffertop_to_bot, (*mpiPSV).bufferbot_to_top, req_send, req_rec);

	   if(trajectory!=NULL){
	      visco_sh_material_observable_end_step();
	   }

	   /* SH-PRECOND-01: forward energy of the exact gradient run only. */
	   if((trajectory!=NULL)&&(Ws!=NULL)){
	      visco_sh_exact_energy_step(Ws,(*waveSH).pvz,NX,NY);
	   }


	   /* store amplitudes at receivers in section-arrays */
	   if (SEISMO && (mode==0 || mode==2)){
	       seismo_ssg(nt, ntr, (*acq).recpos_loc, (*seisSH).sectionvz, (*seisSH).sectionvz, (*seisSH).sectionvz, (*seisSH).sectionvz, (*seisSH).sectionvz, 
			  (*waveSH).pvz, (*waveSH).pvz, (*waveSH).pvz, (*waveSH).pvz, (*matSH).pu, (*matSH).pu, (*matSH).prho, hc);
			/*lsamp+=NDT;*/
	   }

	   /* WRITE SNAPSHOTS TO DISK */
	   if ((SNAP) && (nt==lsnap) && (nt<=iround(TSNAP2/DT))){

	      snap(FP,nt,++nsnap,(*waveSH).pvz,(*waveSH).pvz,(*waveSH).pvz,(*waveSH).pvz,(*matSH).pu,(*matSH).pu,hc);

	      lsnap=lsnap+iround(TSNAPINC/DT);
	   }
	      

	if((nt==hin1)&&(mode==0)&&(MODE==1||MODE==2)){

	  /* store forward wavefields for time-domain inversion and RTM */
          /* ---------------------------------------------------------- */	    
	    
		if(trajectory==NULL)   /* K-5: the exact path keeps no legacy history */
		for (i=1;i<=NX;i=i+IDXI){ 
		    for (j=1;j<=NY;j=j+IDYI){

			/* store forward wavefield for density gradient */
		        (*fwiSH).forward_prop_rho_z[imat2] = (*waveSH).pvzp1[j][i];

			/* store forward wavefield for mu gradient */
	 	        (*fwiSH).forward_prop_sxz[imat2] = (*waveSH).uzx[j][i];
	 	        (*fwiSH).forward_prop_syz[imat2] = (*waveSH).uz[j][i];
			 
			imat2++;			 			 
		    
		    }
		}
	   
	    if((EPRECOND==1)||(EPRECOND==3)){
	      eprecond_SH(Ws,(*waveSH).pvz);
	    }
	 
	    hin++;
	    hin1=hin1+DTINV;

            DTINV_help[nt]=1;
		                                                     
	}

	/* save backpropagated wavefields for time-domain inversion and partially assemble gradients */ 
        /* ----------------------------------------------------------------------------------------- */
	if((mode==1)&&(DTINV_help[NT-nt+1]==1)){
	    
		imat=((NXNYI*(NTDTINV)) - hin*NXNYI)+1;
	                
		    for (i=1;i<=NX;i=i+IDXI){   
			for (j=1;j<=NY;j=j+IDYI){ 
		                 
				 /* Density gradient */                  
			   	(*fwiSH).waveconv_rho_shot[j][i] += (*waveSH).pvzp1[j][i] * (*fwiSH).forward_prop_rho_z[imat];
			
				/* Vs-gradient (symmetrized stress-velocity formulation) */
		                if(GRAD_FORM==2){			

		           	    if(INVMAT1==1){
		               	        muss = (*matSH).prho[j][i] * (*matSH).pu[j][i] * (*matSH).pu[j][i];
	                   	    }
	           
		           	    if(INVMAT1==3){
		               	        muss = (*matSH).pu[j][i];
				    }  

				    if(muss>0.0){			
				         (*fwiSH).waveconv_u_shot[j][i] += ((*fwiSH).forward_prop_syz[imat] * (*waveSH).psyz[j][i]) + ((*fwiSH).forward_prop_sxz[imat] * (*waveSH).psxz[j][i]);
				    } 		                  
		                }			
						                                                                                                     
			imat++;
			}
		    }  
		  
		  if(EPRECOND==1){
		     eprecond_SH(Wr,(*waveSH).pvz);
		  }
		                                                                                                                       
	    hin++;
	    }

	   }/*--------------------  End  of loop over timesteps ----------*/		

}

void sh_visc(struct waveSH *waveSH, struct waveSH_PML *waveSH_PML, struct matSH *matSH, struct fwiSH *fwiSH, struct mpiPSV *mpiPSV,
         struct seisSH *seisSH, struct seisSHfwi *seisSHfwi, struct acq *acq, float *hc, int ishot, int nshots, int nsrc_loc,
         int ns, int ntr, float **Ws, float **Wr, int hin, int *DTINV_help, int mode, MPI_Request * req_send, MPI_Request * req_rec){

	sh_visc_with_material_trajectory(waveSH,waveSH_PML,matSH,fwiSH,mpiPSV,
	         seisSH,seisSHfwi,acq,hc,ishot,nshots,nsrc_loc,ns,ntr,Ws,Wr,hin,
	         DTINV_help,mode,req_send,req_rec,NULL);
}
