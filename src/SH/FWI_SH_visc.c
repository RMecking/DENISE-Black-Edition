/*
 * Full Waveform Inversion (2D visco-elastic SH problem)  
 *
 * Daniel Koehn
 * Kiel, 12/12/2017
 */

#include "fd.h"

static void exact_sh_require_collective_success(int local_status,
        char *message){
    int local_failure = (local_status != 0);
    int any_failure = 0;

    if(MPI_Allreduce(&local_failure,&any_failure,1,MPI_INT,MPI_MAX,
                     MPI_COMM_WORLD) != MPI_SUCCESS){
        err("Exact SH FWI status reconciliation failed.");
    }
    if(any_failure){
        err(message);
    }
}

static void exact_sh_free_material(struct matSH *material, int nd,
        int nx, int ny, int mechanisms){
    free_matrix(material->prho,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->prhoi,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->puip,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->pujp,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->pu,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->puipjp,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->pqs,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->ptaus,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->ptausipjp,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->fipjp,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->f,-nd+1,ny+nd,-nd+1,nx+nd);
    free_matrix(material->g,-nd+1,ny+nd,-nd+1,nx+nd);
    free_vector(material->peta,1,mechanisms);
    free_vector(material->etaip,1,mechanisms);
    free_vector(material->etajm,1,mechanisms);
    free_vector(material->bip,1,mechanisms);
    free_vector(material->bjm,1,mechanisms);
    free_vector(material->cip,1,mechanisms);
    free_vector(material->cjm,1,mechanisms);
    free_f3tensor(material->dip,-nd+1,ny+nd,-nd+1,nx+nd,1,mechanisms);
    free_f3tensor(material->d,-nd+1,ny+nd,-nd+1,nx+nd,1,mechanisms);
    free_f3tensor(material->e,-nd+1,ny+nd,-nd+1,nx+nd,1,mechanisms);
}

void FWI_SH_visc(){

/* global variables */
/* ---------------- */

/* forward modelling */
extern int MYID, FDORDER, NX, NY, NT, L, READMOD, TIME_FILT, READREC;
extern int LOG, SEISMO, FW, NXG, NYG, IENDX, IENDY, NTDTINV, IDXI, IDYI, NXNYI, DTINV;
extern float FC, FC_START, TIME, DT;
extern char LOG_FILE[STRING_SIZE];
extern FILE *FP;

/* gravity modelling/inversion */
extern int GRAVITY, NZGRAV, NGRAVB, GRAV_TYPE;
extern char GRAV_DATA_OUT[STRING_SIZE], GRAV_DATA_IN[STRING_SIZE], GRAV_STAT_POS[STRING_SIZE];
extern float LAM_GRAV, LAM_GRAV_GRAD;

/* full waveform inversion */
extern int GRAD_METHOD, ITERMAX, IDX, IDY, INVMAT1, EPRECOND, LNORM;
extern int GRAD_FORM, QUELLTYPB, MIN_ITER, INV_MOD_OUT, ROWI;
extern float FC_END, PRO, C_vs, C_rho, C_vs_min, C_rho_min, C_taus_min;
extern float EPS_SCALE, SCALEFAC, VSUPPERLIM, VSLOWERLIM;
extern float RHOUPPERLIM, RHOLOWERLIM, QSUPPERLIM, QSLOWERLIM, *FL;
extern float Q_APPROX_FMIN, Q_APPROX_FMAX, Q_APPROX_DF;
extern int STEPMAX, Q_PARAMETERIZATION_MODE;
extern char MISFIT_LOG_FILE[STRING_SIZE];
extern char *FILEINP1;

/* local variables */
int ns, nseismograms=0, nd, fdo3, j, i, iter, iter_true;
int buffsize, ntr=0, ntr_loc=0, ntr_glob=0, nsrc=0, ishot=1;

float eps_scale, opteps_vp, opteps_vs, opteps_rho, Vs_max, rho_max, taus_max, Vs_sum, rho_sum, taus_sum;
float Vs_min, rho_min, taus_min, Vs_avg, rho_avg;
char *buff_addr, ext[10];

double time1, time8, time_av_v_update=0.0, time_av_s_update=0.0, time_av_v_exchange=0.0;
double time_av_s_exchange=0.0, time_av_timestep=0.0;
	
double L2sum, *L2t;
	
float * epst1, *hc=NULL;
int * DTINV_help;

MPI_Request *req_send, *req_rec;

/* Variables for exact physical-Q steepest descent. */
int step3=0, exact_status;
int exact_i, exact_j, exact_l;
float **exact_base_primary, **exact_base_rho, **exact_base_q;
float **exact_grad_primary, **exact_grad_rho, **exact_grad_q;
float **exact_step_primary, **exact_step_rho, **exact_step_q;
float **exact_trial_primary, **exact_trial_rho, **exact_trial_q;
float **exact_trial_tau, *exact_peta;
struct q_tau_mapping exact_q_mapping;
struct visco_sh_exact_material_preparation_request exact_material_request;
struct visco_sh_exact_multi_shot_request exact_objective_request;
struct visco_sh_exact_multi_shot_result exact_objective_result;
struct visco_sh_exact_optimizer_boundary exact_optimizer_boundary;
struct visco_sh_exact_trial_state_request exact_trial_state;
struct visco_sh_exact_trial_objective_request exact_trial_objective;
struct visco_sh_exact_line_search_request exact_line_search;
struct visco_sh_exact_line_search_result exact_line_search_result;
struct matSH exact_trial_material;

/* parameters for FWI-workflow */
int stagemax=0, nstage;

/*vector for abort criterion*/
double * L2_hist=NULL;

/* help variable for MIN_ITER */
int min_iter_help=0;

/* parameters for gravity inversion */
float * gz_mod, * gz_res;
float ** gravpos=NULL, ** rho_grav=NULL, ** rho_grav_ext=NULL;
float ** grad_grav=NULL;
int ngrav=0, nxgrav, nygrav;
float L2_grav, FWImax_all, GRAVmax_all;

/* parameters for random number generation */
int ra, ra1;

FILE *FPL2, *FP_stage, *LAMBDA;

if (MYID == 0){
   time1=MPI_Wtime(); 
   clock();
}

/* open log-file (each PE is using different file) */
/*	fp=stdout; */
sprintf(ext,".%i",MYID);  
strcat(LOG_FILE,ext);

if ((MYID==0) && (LOG==1)) FP=stdout;
else FP=fopen(LOG_FILE,"w");
fprintf(FP," This is the log-file generated by PE %d \n\n",MYID);

/* ----------------------- */
/* define FD grid geometry */
/* ----------------------- */

/* domain decomposition */
initproc();

/* Exact C8c activation currently supports steepest descent only.  Abort
   before loading or changing any model state for unsupported optimizers. */
if(GRAD_METHOD!=0){
    err("Exact viscoelastic SH FWI currently supports GRAD_METHOD == 0 only.");
}
if(L<1){
    err("Exact viscoelastic SH FWI requires at least one relaxation mechanism.");
}
if(Q_PARAMETERIZATION_MODE!=Q_PARAMETERIZATION_PHYSICAL){
    err("Exact viscoelastic SH FWI requires physical-Q parameterization.");
}
if((INVMAT1!=1)&&(INVMAT1!=3)){
    err("Exact viscoelastic SH FWI supports INVMAT1 == 1 or INVMAT1 == 3 only.");
}
if(!READMOD){
    err("Exact viscoelastic SH FWI requires a loaded physical-Q model.");
}
if(GRAVITY!=0){
    err("Exact viscoelastic SH FWI does not yet support coupled gravity inversion.");
}

NT=iround(TIME/DT); /* number of timesteps */

/* output of parameters to log-file or stdout */
if (MYID==0) write_par(FP);

/* NXG, NYG denote size of the entire (global) grid */
NXG=NX;
NYG=NY;

/* In the following, NX and NY denote size of the local grid ! */
NX = IENDX;
NY = IENDY;

NTDTINV=ceil((float)NT/(float)DTINV);		/* round towards next higher integer value */

/* save every IDXI and IDYI spatial point during the forward modelling */
IDXI=1;
IDYI=1;

NXNYI=(NX/IDXI)*(NY/IDYI);

/* use only every DTINV time sample for the inversion */
DTINV_help=ivector(1,NT);

/* read parameters from workflow-file (stdin) */
FP_stage=fopen(FILEINP1,"r");
if(FP_stage==NULL) {
	if (MYID == 0){
		printf("\n==================================================================\n");
		printf(" Cannot open Denise workflow input file %s \n",FILEINP1);
		printf("\n==================================================================\n\n");
		err(" --- ");
	}
}

/* estimate number of lines in FWI-workflow */
i=0;
stagemax=0;
while ((i=fgetc(FP_stage)) != EOF)
if (i=='\n') ++stagemax;
rewind(FP_stage);
stagemax--;
fclose(FP_stage);

/* define data structures for PSV problem */
struct waveSH;
struct waveSH_PML;
struct matSH;
struct fwiSH;
struct mpiPSV;
struct seisSH;
struct seisSHfwi;
struct acq;

nd = FDORDER/2 + 1;
fdo3 = 2*nd;
buffsize=2.0*2.0*fdo3*(NX +NY)*sizeof(MPI_FLOAT);

/* allocate buffer for buffering messages */
buff_addr=malloc(buffsize);
if (!buff_addr) err("allocation failure for buffer for MPI_Bsend !");
MPI_Buffer_attach(buff_addr,buffsize);

/* allocation for request and status arrays */
req_send=(MPI_Request *)malloc(REQUEST_COUNT*sizeof(MPI_Request));
req_rec=(MPI_Request *)malloc(REQUEST_COUNT*sizeof(MPI_Request));

/* --------- add different modules here ------------------------ */
ns=NT;	/* in a FWI one has to keep all samples of the forward modeled data
	at the receiver positions to calculate the adjoint sources and to do 
	the backpropagation; look at function saveseis_glob.c to see that every
	NDT sample for the forward modeled wavefield is written to su files*/

if (SEISMO && (READREC!=2)){

   acq.recpos=receiver(FP, &ntr, ishot);
   acq.recswitch = ivector(1,ntr);
   acq.recpos_loc = splitrec(acq.recpos,&ntr_loc, ntr, acq.recswitch);
   ntr_glob=ntr;
   ntr=ntr_loc;
   
}

if(READREC!=2){

   /* Memory for seismic data */
   alloc_seisSH(ntr,ns,&seisSH);

   /* Memory for FWI seismic data */ 
   alloc_seisSHfwi(ntr,ntr_glob,ns,&seisSHfwi);
   
   /* Memory for full data seismograms */
   alloc_seisSHfull(&seisSH,ntr_glob);

}

/* memory allocation for abort criterion*/
L2_hist = dvector(1,1000);

/* estimate memory requirement of the variables in megabytes*/
	
switch (SEISMO){
case 1 : /* particle velocities only */
	nseismograms=1;	
	break;	
}		

/* calculate memory requirements for PSV forward problem */
mem_fwiPSV(nseismograms,ntr,ns,fdo3,nd,buffsize,ntr_glob);

/* Define gradient formulation */
/* GRAD_FORM = 2 - stress-velocity gradients for symmetrized impedance matrix */
GRAD_FORM = 2;

if(GRAVITY==1 || GRAVITY==2){
  
  if(GRAV_TYPE == 1){
  sprintf(GRAV_DATA_OUT, "./gravity/grav_mod.dat"); /* output file of gravity data */
  sprintf(GRAV_DATA_IN, "./gravity/grav_field.dat");  /* input file of gravity data */
  }
  if(GRAV_TYPE == 2){
  sprintf(GRAV_DATA_OUT, "./gravity/grav_grad_mod.dat"); /* output file of gravity gradient data */
  sprintf(GRAV_DATA_IN, "./gravity/grav_grad_field.dat");  /* input file of gravity gradientdata */
  }
  sprintf(GRAV_STAT_POS, "./gravity/grav_stat.dat"); /* file with station positions for gravity modelling */

  /* size of the extended gravity model */
  nxgrav = NXG + 2*NGRAVB;
  nygrav = NYG + NGRAVB;

}

/* allocate memory for SH forward problem */
alloc_SH(&waveSH,&waveSH_PML);

/* calculate damping coefficients for CPMLs (SH problem)*/
if(FW>0){PML_pro_SH(waveSH_PML.d_x, waveSH_PML.K_x, waveSH_PML.alpha_prime_x, waveSH_PML.a_x, waveSH_PML.b_x, waveSH_PML.d_x_half, waveSH_PML.K_x_half, waveSH_PML.alpha_prime_x_half, waveSH_PML.a_x_half, 
                 waveSH_PML.b_x_half, waveSH_PML.d_y, waveSH_PML.K_y, waveSH_PML.alpha_prime_y, waveSH_PML.a_y, waveSH_PML.b_y, waveSH_PML.d_y_half, waveSH_PML.K_y_half, waveSH_PML.alpha_prime_y_half, 
                 waveSH_PML.a_y_half, waveSH_PML.b_y_half);
}

/* allocate memory for SH material parameters */
alloc_matSH(&matSH);
alloc_matSH(&exact_trial_material);

/* allocate memory for SH FWI parameters */
alloc_fwiSH(&fwiSH);

/* allocate memory for PSV MPI variables */
alloc_mpiPSV(&mpiPSV);

exact_base_primary = matrix(1,NY,1,NX);
exact_base_rho = matrix(1,NY,1,NX);
exact_base_q = matrix(1,NY,1,NX);
exact_grad_primary = matrix(1,NY,1,NX);
exact_grad_rho = matrix(1,NY,1,NX);
exact_grad_q = matrix(1,NY,1,NX);
exact_step_primary = matrix(1,NY,1,NX);
exact_step_rho = matrix(1,NY,1,NX);
exact_step_q = matrix(1,NY,1,NX);
exact_trial_primary = matrix(1,NY,1,NX);
exact_trial_rho = matrix(1,NY,1,NX);
exact_trial_q = matrix(1,NY,1,NX);
exact_trial_tau = matrix(1,NY,1,NX);
exact_peta = vector(1,L);

/* memory for source position definition */
acq.srcpos1=fmatrix(1,8,1,1);

/* memory of L2 norm */
L2t = dvector(1,4);
epst1 = vector(1,3);
	
fprintf(FP," ... memory allocation for PE %d was successfull.\n\n", MYID);

/* Holberg coefficients for FD operators*/
hc = holbergcoeff();

MPI_Barrier(MPI_COMM_WORLD);

/* Reading source positions from SOURCE_FILE */ 	
acq.srcpos=sources(&nsrc);


/* create model grids */
if (READMOD) readmod_visc_SH(matSH.prho,matSH.pu,matSH.pqs,matSH.ptaus,matSH.peta);
/*else model(matPSV.prho,matPSV.ppi,matPSV.pu,matPSV.ptaus,matPSV.ptaup,matPSV.peta);*/

/* Establish the disjoint authoritative Base state.  Solver Tau and all
   material caches are derived from these primary/rho/physical-Q fields. */
for(exact_j=1;exact_j<=NY;exact_j++){
    for(exact_i=1;exact_i<=NX;exact_i++){
        exact_base_primary[exact_j][exact_i]=matSH.pu[exact_j][exact_i];
        exact_base_rho[exact_j][exact_i]=matSH.prho[exact_j][exact_i];
        exact_base_q[exact_j][exact_i]=matSH.pqs[exact_j][exact_i];
    }
}
for(exact_l=1;exact_l<=L;exact_l++){
    exact_peta[exact_l]=matSH.peta[exact_l];
}
init_q_tau_mapping(&exact_q_mapping,Q_PARAMETERIZATION_PHYSICAL,L,FL,
                   Q_APPROX_FMIN,Q_APPROX_FMAX,Q_APPROX_DF);

exact_material_request.primary=exact_base_primary;
exact_material_request.rho=exact_base_rho;
exact_material_request.physical_q=exact_base_q;
exact_material_request.target=&matSH;
exact_material_request.mechanisms=L;
exact_material_request.dt=DT;
exact_material_request.frequencies_hz=FL;
exact_material_request.peta=exact_peta;
exact_status=visco_sh_exact_prepare_visco_material(&exact_material_request);
exact_sh_require_collective_success(exact_status,
        "Exact SH Base material initialization failed.");


/* check if the FD run will be stable and free of numerical dispersion */
checkfd_visc_SH(FP,matSH.prho,matSH.pu,matSH.ptaus,matSH.peta,hc);


if(GRAVITY==1 || GRAVITY==2){
 
  /* read station positions */
  MPI_Barrier(MPI_COMM_WORLD);
  gravpos=read_grav_pos(&ngrav);

  /* define model and residual data vector for gz (z-component of the gravity field) */
  gz_mod = vector(1,ngrav);
  gz_res = vector(1,ngrav);

  /* only forward modelling of gravity data */
  if(GRAVITY==1){

    /* global density model */
    rho_grav =  matrix(1,NYG,1,NXG);
    rho_grav_ext =  matrix(1,nygrav,1,nxgrav);

    read_density_glob(rho_grav,1);
    extend_mod(rho_grav,rho_grav_ext,nxgrav,nygrav);
    grav_mod(rho_grav_ext,ngrav,gravpos,gz_mod,nxgrav,nygrav,NZGRAV);

    free_matrix(rho_grav,1,NYG,1,NXG);
    free_matrix(rho_grav_ext,1,nygrav,1,nxgrav);

  }

  if(GRAVITY==2){
    grad_grav =  matrix(1,NY,1,NX);
  }

} 
      
iter_true=1;

/* Begin of FWI-workflow */
for(nstage=1;nstage<=stagemax;nstage++){

/* read workflow input file *.inp */
FP_stage=fopen(FILEINP1,"r");
read_par_inv(FP_stage,nstage,stagemax);
/*fclose(FP_stage);*/

FC=FC_END;

iter=1;
/* --------------------------------------
 * Begin of Full Waveform iteration loop
 * -------------------------------------- */
while(iter<=ITERMAX){

      /* Apply random objective waveform inversion */
      if(ROWI){
      
         /* fetch random number on MYID==0 */
         if(MYID==0){
      
            /* initialize random number generator */
      	    srand((unsigned)time(NULL));
      
      	    /* generate random number between 1 and 100 */
            ra = rand();
      	    ra1 = (ra % 100) + 1;
            
         }
      
         /* broadcast random number over all MPI processes */	       
         MPI_Barrier(MPI_COMM_WORLD);
         MPI_Bcast(&ra1,1,MPI_INT,0,MPI_COMM_WORLD);
      
         if(ra1<=50){LNORM=8;}
	 if(ra1>50){LNORM=2;}
      
         if(MYID==0){
            printf("ra1 = %d \t LNORM = %d \n", ra1, LNORM);      
         }
      
      }
        
      MPI_Barrier(MPI_COMM_WORLD);

if (MYID==0)
   {
   fprintf(FP,"\n\n\n ------------------------------------------------------------------\n");
   fprintf(FP,"\n\n\n                   TDFWI ITERATION %d \t of %d \n",iter,ITERMAX);
   fprintf(FP,"\n\n\n ------------------------------------------------------------------\n");
   }

/* Rebuild solver-ready Base material from authoritative physical Q. */
exact_status=visco_sh_exact_prepare_visco_material(&exact_material_request);
exact_sh_require_collective_success(exact_status,
        "Exact SH Base material preparation failed.");

if(iter_true==1){

    for (i=1;i<=NX;i=i+IDX){ 
	for (j=1;j<=NY;j=j+IDY){
	
	if(INVMAT1==1){
	
	  fwiSH.Vs0[j][i] = matSH.pu[j][i];
	  fwiSH.Rho0[j][i] = matSH.prho[j][i];
	  fwiSH.Taus0[j][i] = matSH.ptaus[j][i];

        }
	  
                 
		 
	if(INVMAT1==2){
        
	  fwiSH.Vs0[j][i] = sqrt(matSH.pu[j][i]*matSH.prho[j][i]);
	  fwiSH.Rho0[j][i] = matSH.prho[j][i];
	  fwiSH.Taus0[j][i] = matSH.ptaus[j][i];
	
	}
	 
	if(INVMAT1==3){
        
	  fwiSH.Vs0[j][i] = matSH.pu[j][i];
	  fwiSH.Rho0[j][i] = matSH.prho[j][i];
	  fwiSH.Taus0[j][i] = matSH.ptaus[j][i];
	
	}  
	
    }
    }

/* ---------------------------------------------- */
/* calculate minimum and maximum model parameters */
/* ---------------------------------------------- */

	Vs_max = 0.0;
	rho_max = 0.0;
	taus_max = 0.0;
	
	Vs_min = 1e10;
	rho_min = 1e10;
	taus_min = 1e10; 
	
	Vs_avg = 0.0;
	rho_avg = 0.0; 
	 
        for (i=1;i<=NX;i=i+IDX){
           for (j=1;j<=NY;j=j+IDY){
	  
		 /* calculate maximum Vs */
		 if(matSH.pu[j][i] > Vs_max){
		     Vs_max = matSH.pu[j][i];
		 }
		 
		 /* calculate minimum Vs */
		 if(matSH.pu[j][i] < Vs_min){
		     Vs_min = matSH.pu[j][i];
		 }
		 
		 /* calculate average vs value */
		 Vs_avg += matSH.pu[j][i];		 
		 
		 /* calculate maximum rho */
		 if(matSH.prho[j][i] > rho_max){
		     rho_max = matSH.prho[j][i];
		 }

		 /* calculate minimum rho */
		 if(matSH.prho[j][i] < rho_min){
		     rho_min = matSH.prho[j][i];
		 }
		 
		 /* calculate average rho value */
		 rho_avg += matSH.prho[j][i];

		 /* calculate maximum taus */
		 if(matSH.ptaus[j][i] > taus_max){
		     taus_max = matSH.ptaus[j][i];
		 }

		 /* calculate minimum taus */
		 if(matSH.ptaus[j][i] < taus_min){
		     taus_min = matSH.ptaus[j][i];
		 }

	
           }
        }

	/* calculate minimum Vs, rho and taus of all CPUs*/
	
	Vs_sum = 0.0;
        MPI_Allreduce(&Vs_min,&Vs_sum,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
        C_vs_min=Vs_sum;
	
	rho_sum = 0.0;
        MPI_Allreduce(&rho_min,&rho_sum,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
        C_rho_min=rho_sum;

	taus_sum = 0.0;
        MPI_Allreduce(&taus_min,&taus_sum,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
        C_taus_min=taus_sum;
		
	/*if(MYID==0){
           printf("Vs_min = %e \t rho_min = %e \t taus_min = %e \n ",C_vs_min, C_rho_min, C_taus_min);	
	}*/
		
        /* calculate maximum Vs, rho and taus of all CPUs*/
	
	Vs_sum = 0.0;
        MPI_Allreduce(&Vs_max,&Vs_sum,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
        Vs_max=Vs_sum;
	
	rho_sum = 0.0;
        MPI_Allreduce(&rho_max,&rho_sum,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
        rho_max=rho_sum;

	taus_sum = 0.0;
        MPI_Allreduce(&taus_max,&taus_sum,1,MPI_FLOAT,MPI_MAX,MPI_COMM_WORLD);
        taus_max=taus_sum;
	
	/* calculate average Vs, rho and taus of all CPUs*/
	
	Vs_sum = 0.0;
        MPI_Allreduce(&Vs_avg,&Vs_sum,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
        Vs_avg=Vs_sum / (NXG*NYG);
	
	rho_sum = 0.0;
        MPI_Allreduce(&rho_avg,&rho_sum,1,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
        rho_avg=rho_sum / (NXG*NYG);

		
	/*if(MYID==0){
           printf("Vs_max = %e \t rho_max = %e \t taus_max = %e \n ",Vs_max, rho_max, taus_max);	
	}*/
	
	if(MYID==0){
           printf("Vs_avg = %e \t rho_max = %e \n ",Vs_avg, rho_avg);	
	}
	
	/* scaling factor for gradients normalized relative to mininum and maximum values*/
	/*C_vs = Vs_max - C_vs_min;
	C_rho = rho_max - C_rho_min;
	C_taus = taus_max - C_taus_min;*/
	
	/* scaling factor for gradients normalized relative to maximum values*/
	/*C_vs = Vs_max;
	C_rho = rho_max;
	C_taus = taus_max;*/
	
	/* scaling factor for gradients normalized relative to average values*/
	C_vs = Vs_avg;
	C_rho = rho_avg;
	
	C_vs_min = 0.0;
	C_rho_min = 0.0;
	C_taus_min = 0.0;

	/* scaling factor for gradients not normalized */
	/*C_vs = 1.0;
	C_rho = 1.0;
	C_taus = 1.0;
	
	C_vs_min = 0.0;
	C_rho_min = 0.0;
	C_taus_min = 0.0;*/
			       	
}

/* Open Log File for L2 norm */
if(MYID==0){
  if(iter_true==1){
    FPL2=fopen(MISFIT_LOG_FILE,"w");
  }

  if(iter_true>1){
    FPL2=fopen(MISFIT_LOG_FILE,"a");
  }
}

/* ---------------------------------------------------------------------------------------------------- */
/* --------- Calculate gradient and objective function using the adjoint state method ----------------- */
/* ---------------------------------------------------------------------------------------------------- */

memset(&exact_objective_request,0,sizeof(exact_objective_request));
exact_objective_request.wave=&waveSH;
exact_objective_request.pml=&waveSH_PML;
exact_objective_request.material=&matSH;
exact_objective_request.fwi=&fwiSH;
exact_objective_request.mpi=&mpiPSV;
exact_objective_request.seismogram=&seisSH;
exact_objective_request.legacy_fwi_seismogram=&seisSHfwi;
exact_objective_request.acquisition=&acq;
exact_objective_request.hc=hc;
exact_objective_request.iter=iter;
exact_objective_request.nsrc=nsrc;
exact_objective_request.ns=ns;
exact_objective_request.nrec_local=ntr;
exact_objective_request.nrec_global=ntr_glob;
exact_objective_request.hin=1;
exact_objective_request.dtinv_help=DTINV_help;
exact_objective_request.source_energy=NULL;
exact_objective_request.receiver_energy=NULL;
exact_objective_request.request_send=req_send;
exact_objective_request.request_receive=req_rec;
exact_objective_request.grad_primary=exact_grad_primary;
exact_objective_request.grad_rho=exact_grad_rho;
exact_objective_request.grad_q=exact_grad_q;

exact_status=visco_sh_exact_objective_gradient(
        &exact_objective_request,&exact_objective_result);
exact_sh_require_collective_success(exact_status,
        "Exact SH objective-gradient evaluation failed.");
L2sum=exact_objective_result.objective;
seisSHfwi.L2=L2sum;

L2t[1]=L2sum;
L2t[4]=L2sum;

memset(&exact_optimizer_boundary,0,sizeof(exact_optimizer_boundary));
exact_optimizer_boundary.nx=NX;
exact_optimizer_boundary.ny=NY;
exact_optimizer_boundary.grad_raw_primary=exact_grad_primary;
exact_optimizer_boundary.grad_raw_rho=exact_grad_rho;
exact_optimizer_boundary.grad_raw_q=exact_grad_q;
exact_optimizer_boundary.optimizer_step_primary=exact_step_primary;
exact_optimizer_boundary.optimizer_step_rho=exact_step_rho;
exact_optimizer_boundary.optimizer_step_q=exact_step_q;
exact_status=visco_sh_exact_build_steepest_subtractive_step(
        &exact_optimizer_boundary);
exact_sh_require_collective_success(exact_status,
        "Exact SH optimizer-boundary construction failed.");

opteps_vs=0.0;
opteps_rho=0.0;

/* ============================================================================================================================*/
/* =============================================== test loop L2 ===============================================================*/
/* ============================================================================================================================*/

/* set min_iter_help to initial global value of MIN_ITER */
if(iter==1){min_iter_help=MIN_ITER;}

memset(&exact_trial_state,0,sizeof(exact_trial_state));
exact_trial_state.nx=NX;
exact_trial_state.ny=NY;
exact_trial_state.primary_bounds_enabled=(INVMAT1==1);
exact_trial_state.primary_lower=VSLOWERLIM;
exact_trial_state.primary_upper=VSUPPERLIM;
exact_trial_state.rho_lower=RHOLOWERLIM;
exact_trial_state.rho_upper=RHOUPPERLIM;
exact_trial_state.q_lower=QSLOWERLIM;
exact_trial_state.q_upper=QSUPPERLIM;
exact_trial_state.q_mapping=&exact_q_mapping;
exact_trial_state.base_primary=exact_base_primary;
exact_trial_state.base_rho=exact_base_rho;
exact_trial_state.base_q=exact_base_q;
exact_trial_state.optimizer_step_primary=exact_step_primary;
exact_trial_state.optimizer_step_rho=exact_step_rho;
exact_trial_state.optimizer_step_q=exact_step_q;
exact_trial_state.trial_primary=exact_trial_primary;
exact_trial_state.trial_rho=exact_trial_rho;
exact_trial_state.trial_q=exact_trial_q;
exact_trial_state.trial_tau=exact_trial_tau;

memset(&exact_trial_objective,0,sizeof(exact_trial_objective));
exact_trial_objective.trial_state=exact_trial_state;
exact_trial_objective.trial_material=&exact_trial_material;
exact_trial_objective.mechanisms=L;
exact_trial_objective.dt=DT;
exact_trial_objective.frequencies_hz=FL;
exact_trial_objective.peta=exact_peta;
exact_trial_objective.objective=exact_objective_request;
exact_trial_objective.objective.material=&exact_trial_material;
exact_trial_objective.objective.grad_primary=NULL;
exact_trial_objective.objective.grad_rho=NULL;
exact_trial_objective.objective.grad_q=NULL;

memset(&exact_line_search,0,sizeof(exact_line_search));
exact_line_search.trial_objective=exact_trial_objective;
exact_line_search.base_objective=L2sum;
exact_line_search.initial_alpha=EPS_SCALE;
exact_line_search.scale_factor=SCALEFAC;
exact_line_search.max_retries=STEPMAX;
exact_status=step_length_est_sh_visc_exact(
        &exact_line_search,&exact_line_search_result);
exact_sh_require_collective_success(exact_status,
        "Exact physical-Q SH line search failed.");

/* Rebuild the selected accepted state through B2, then reconcile all ranks
   before committing any authoritative Base cell. */
exact_trial_state.alpha=exact_line_search_result.selected_alpha;
exact_status=visco_sh_exact_build_trial_parameter_state(&exact_trial_state);
exact_sh_require_collective_success(exact_status,
        "Exact SH accepted Trial construction failed.");

for(exact_j=1;exact_j<=NY;exact_j++){
    for(exact_i=1;exact_i<=NX;exact_i++){
        exact_base_primary[exact_j][exact_i]=exact_trial_primary[exact_j][exact_i];
        exact_base_rho[exact_j][exact_i]=exact_trial_rho[exact_j][exact_i];
        exact_base_q[exact_j][exact_i]=exact_trial_q[exact_j][exact_i];
    }
}

/* Regenerate accepted Tau, halos, and material caches solely from Base Q. */
exact_status=visco_sh_exact_prepare_visco_material(&exact_material_request);
exact_sh_require_collective_success(exact_status,
        "Exact SH accepted Base material rebuild failed.");

eps_scale=exact_line_search_result.selected_alpha;
opteps_vp=eps_scale;
step3=0;
epst1[1]=0.0f;
epst1[2]=eps_scale;
epst1[3]=eps_scale;
L2t[2]=exact_line_search_result.selected_objective;
L2t[3]=exact_line_search_result.selected_objective;

/* write log-parameter files */
if(MYID==0){
printf("MYID = %d \t opteps_vp = %e \t opteps_vs = %e \t opteps_rho = %e \n",MYID,opteps_vp,opteps_vs,opteps_rho);
printf("MYID = %d \t L2t[1] = %e \t L2t[2] = %e \t L2t[3] = %e \t L2t[4] = %e \n",MYID,L2t[1],L2t[2],L2t[3],L2t[4]);
printf("MYID = %d \t epst1[1] = %e \t epst1[2] = %e \t epst1[3] = %e \n",MYID,epst1[1],epst1[2],epst1[3]);

/*output of log file for combined inversion*/
if(iter_true==1 && GRAVITY){
    LAMBDA = fopen("gravity/lambda.dat","w");
}
if(iter_true>1 && GRAVITY){
    LAMBDA = fopen("gravity/lambda.dat","a");
}

if(GRAVITY){
    fprintf(LAMBDA,"%d \t %d \t %e \t %e \t %e \t %e \t %e \t %e \t %e \n",nstage,iter,LAM_GRAV,L2sum,L2_grav,L2t[4],LAM_GRAV_GRAD,FWImax_all,GRAVmax_all);
    fclose(LAMBDA);
}

}

if(MYID==0){
if (TIME_FILT==0){
	fprintf(FPL2,"%e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %d \n",opteps_vp,epst1[1],epst1[2],epst1[3],L2t[1],L2t[2],L2t[3],L2t[4],nstage);}
else{
	fprintf(FPL2,"%e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \t %f \t %f \t %d \n",opteps_vp,epst1[1],epst1[2],epst1[3],L2t[1],L2t[2],L2t[3],L2t[4],FC_START,FC,nstage);}}


/* saving history of final L2*/
L2_hist[iter]=L2t[4];

if(MYID==0){	
/*	fprintf(FPL2,"=============================================================\n");
	fprintf(FPL2,"=============================================================\n");
	fprintf(FPL2,"STATISTICS FOR ITERATION STEP %d \n",iter);
	fprintf(FPL2,"=============================================================\n");
	fprintf(FPL2,"=============================================================\n");*/
/*	fprintf(FPL2,"Low-pass filter at %e Hz\n",freq);
	fprintf(FPL2,"----------------------------------------------\n");
*/	/*fprintf(FPL2,"L2 at iteration step n = %e \n",L2);*/
/*        fprintf(FPL2,"%e \t %e \t %e \t %e \t %e \t %e \t %e \t %e \n",EPSILON,EPSILON_u,EPSILON_rho,L2t[4],betaVp,betaVs,betarho,sqrt(C_vp));*/

	/*fprintf(FPL2,"----------------------------------------------\n");*/
/*	fprintf(FPL2,"=============================================================\n");
	fprintf(FPL2,"=============================================================\n\n\n");*/
}

if(MYID==0){
  fclose(FPL2);
}

if (iter>min_iter_help){

float diff=0.0, pro=PRO;

/* calculating differnce of the actual L2 and before two iterations, dividing with L2_hist[iter-2] provide changing in procent*/
diff=fabs((L2_hist[iter-2]-L2_hist[iter])/L2_hist[iter-2]);
	
	if((diff<=pro)||(step3==1)){
        
        	/* output of the model at the end of given FWI stage */
		if(INV_MOD_OUT==0){
        	    model_freq_out_SH_visc(matSH.prho,matSH.pu,matSH.ptaus,nstage,FC);
		}

		min_iter_help=0;
		min_iter_help=iter+MIN_ITER;
		iter=0;

        	if(MYID==0){
			if(step3==1){
			        printf("\n Steplength estimation failed step3=%d \n Changing to next FWI stage \n",step3);
			}
			else{
  				printf("\n Reached the abort criterion of pro=%e and diff=%e \n Changing to next FWI stage \n",pro,diff);
			}
	
		}
		break;
	}
}

/* output of the model after each FWI iteration */
if(INV_MOD_OUT==1){
    model_it_out_SH_visc(matSH.prho,matSH.pu,matSH.ptaus,nstage,iter,FC);
}

iter++;
iter_true++;

/* ====================================== */
} /* end of fullwaveform iteration loop*/
/* ====================================== */

} /* End of FWI-workflow loop */

/* deallocate memory for SH forward problem */
dealloc_SH(&waveSH,&waveSH_PML);

/* deallocation of memory */
free_matrix(exact_base_primary,1,NY,1,NX);
free_matrix(exact_base_rho,1,NY,1,NX);
free_matrix(exact_base_q,1,NY,1,NX);
free_matrix(exact_grad_primary,1,NY,1,NX);
free_matrix(exact_grad_rho,1,NY,1,NX);
free_matrix(exact_grad_q,1,NY,1,NX);
free_matrix(exact_step_primary,1,NY,1,NX);
free_matrix(exact_step_rho,1,NY,1,NX);
free_matrix(exact_step_q,1,NY,1,NX);
free_matrix(exact_trial_primary,1,NY,1,NX);
free_matrix(exact_trial_rho,1,NY,1,NX);
free_matrix(exact_trial_q,1,NY,1,NX);
free_matrix(exact_trial_tau,1,NY,1,NX);
free_vector(exact_peta,1,L);
exact_sh_free_material(&exact_trial_material,nd,NX,NY,L);

free_matrix(fwiSH.Vs0,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.Rho0,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.Taus0,-nd+1,NY+nd,-nd+1,NX+nd);

free_matrix(matSH.prho,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.prho_old,-nd+1,NY+nd,-nd+1,NX+nd);

free_matrix(matSH.pu,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.pu_old,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(matSH.puipjp,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(matSH.puip,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(matSH.pujp,-nd+1,NY+nd,-nd+1,NX+nd);

free_matrix(fwiSH.ptaus_old,-nd+1,NY+nd,-nd+1,NX+nd);

free_matrix(mpiPSV.bufferlef_to_rig,1,NY,1,fdo3);
free_matrix(mpiPSV.bufferrig_to_lef,1,NY,1,fdo3);
free_matrix(mpiPSV.buffertop_to_bot,1,NX,1,fdo3);
free_matrix(mpiPSV.bufferbot_to_top,1,NX,1,fdo3);

free_vector(hc,0,6);

free_matrix(fwiSH.gradg_rho,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.gradp_rho,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_rho,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_rho_s,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_rho_shot,-nd+1,NY+nd,-nd+1,NX+nd);

free_matrix(fwiSH.gradg_u,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.gradp_u,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_u,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_mu,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_u_shot,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_u_x_shot,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_u_y_shot,-nd+1,NY+nd,-nd+1,NX+nd);

free_matrix(fwiSH.gradg_ts,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.gradp_ts,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_ts,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_ts_s,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(fwiSH.waveconv_ts_shot,-nd+1,NY+nd,-nd+1,NX+nd);

free_vector(fwiSH.forward_prop_sxz,1,NY*NX*NT);
free_vector(fwiSH.forward_prop_syz,1,NY*NX*NT);
free_vector(fwiSH.forward_prop_rho_z,1,NY*NX*NT);

free_matrix(fwiSH.forward_prop_rxz,1,NY*NX*NT,1,L);
free_matrix(fwiSH.forward_prop_ryz,1,NY*NX*NT,1,L);

if(EPRECOND==4){
   free_matrix(fwiSH.hess_mu2,-nd+1,NY+nd,-nd+1,NX+nd);
   free_matrix(fwiSH.hess_rho2,-nd+1,NY+nd,-nd+1,NX+nd);
   free_matrix(fwiSH.hess_ts2,-nd+1,NY+nd,-nd+1,NX+nd);
   free_matrix(fwiSH.hess_vs2,-nd+1,NY+nd,-nd+1,NX+nd);
   free_matrix(fwiSH.hess_rho2p,-nd+1,NY+nd,-nd+1,NX+nd);
   
   free_matrix(fwiSH.hess_muts,-nd+1,NY+nd,-nd+1,NX+nd);
   free_matrix(fwiSH.hess_murho,-nd+1,NY+nd,-nd+1,NX+nd);
   free_matrix(fwiSH.hess_tsrho,-nd+1,NY+nd,-nd+1,NX+nd);   
}

 /* free memory for global source positions */
 free_matrix(acq.srcpos,1,8,1,nsrc);

 /* free memory for source position definition */
 free_matrix(acq.srcpos1,1,8,1,1);
 
 /* free memory for abort criterion */
 free_dvector(L2_hist,1,1000);
 		
 free_dvector(L2t,1,4);
 free_vector(epst1,1,3);

 if(READREC!=2){

    if (SEISMO) free_imatrix(acq.recpos,1,3,1,ntr_glob);

    if ((ntr>0) && (SEISMO)){

            free_imatrix(acq.recpos_loc,1,3,1,ntr);
            acq.recpos_loc = NULL;
 
            switch (SEISMO){
            case 1 : /* particle velocities only */
                    free_matrix(seisSH.sectionvz,1,ntr,1,ns);
                    seisSH.sectionvz=NULL;
                    break;

             }

    }

    free_matrix(seisSHfwi.sectionread,1,ntr_glob,1,ns);
    free_ivector(acq.recswitch,1,ntr);
    
    if(QUELLTYPB){
       free_matrix(seisSHfwi.sectionvzdata,1,ntr,1,ns);
       free_matrix(seisSHfwi.sectionvzdiff,1,ntr,1,ns);
       free_matrix(seisSHfwi.sectionvzdiffold,1,ntr,1,ns);
    }

    if(SEISMO){
        free_matrix(seisSH.fulldata_vz,1,ntr_glob,1,NT); 
    }
 
 }

 free_ivector(DTINV_help,1,NT);
 
 /* free memory for viscoelastic modeling variables */ 
free_matrix(matSH.pqs,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(matSH.ptaus,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(matSH.ptausipjp,-nd+1,NY+nd,-nd+1,NX+nd);
free_vector(matSH.peta,1,L);
free_vector(matSH.etaip,1,L);
free_vector(matSH.etajm,1,L);
free_vector(matSH.bip,1,L);
free_vector(matSH.bjm,1,L);
free_vector(matSH.cip,1,L);
free_vector(matSH.cjm,1,L);
free_matrix(matSH.f,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(matSH.fipjp,-nd+1,NY+nd,-nd+1,NX+nd);
free_matrix(matSH.g,-nd+1,NY+nd,-nd+1,NX+nd);
free_f3tensor(matSH.dip,-nd+1,NY+nd,-nd+1,NX+nd,1,L);
free_f3tensor(matSH.d,-nd+1,NY+nd,-nd+1,NX+nd,1,L);
free_f3tensor(matSH.e,-nd+1,NY+nd,-nd+1,NX+nd,1,L);


if(GRAVITY){

  free_matrix(gravpos,1,2,1,ngrav);
  free_vector(gz_mod,1,ngrav);
  free_vector(gz_res,1,ngrav);

  if(GRAVITY==2){
    free_matrix(grad_grav,1,NY,1,NX);
  }

}
 
/* de-allocate buffer for messages */
MPI_Buffer_detach(buff_addr,&buffsize);

MPI_Barrier(MPI_COMM_WORLD);

if (MYID==0){
	fprintf(FP,"\n **Info from main (written by PE %d): \n",MYID);
	fprintf(FP," CPU time of program per PE: %li seconds.\n",clock()/CLOCKS_PER_SEC);
	time8=MPI_Wtime();
	fprintf(FP," Total real time of program: %4.2f seconds.\n",time8-time1);
	time_av_v_update=time_av_v_update/(double)NT;
	time_av_s_update=time_av_s_update/(double)NT;
	time_av_v_exchange=time_av_v_exchange/(double)NT;
	time_av_s_exchange=time_av_s_exchange/(double)NT;
	time_av_timestep=time_av_timestep/(double)NT;
	/* fprintf(FP," Average times for \n");
	fprintf(FP," velocity update:  \t %5.3f seconds  \n",time_av_v_update);
	fprintf(FP," stress update:  \t %5.3f seconds  \n",time_av_s_update);
	fprintf(FP," velocity exchange:  \t %5.3f seconds  \n",time_av_v_exchange);
	fprintf(FP," stress exchange:  \t %5.3f seconds  \n",time_av_s_exchange);
	fprintf(FP," timestep:  \t %5.3f seconds  \n",time_av_timestep);*/		
}

fclose(FP);


}



