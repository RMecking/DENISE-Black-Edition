/*------------------------------------------------------------------------
 *   output of material parameters after each stage
 *   
 *   Daniel Koehn
 *   last update 22.12.2017
 *
 *  ---------------------------------------------------------------------*/

#include "fd.h"
void model_freq_out_SH_visc(float **rho, float **primary, float **physical_q, int nstage, float freq){

	/* extern variables */
	extern int POS[3], MYID, INVMAT1;
	extern char INV_MODELFILE[STRING_SIZE];

	/* local variables */
	char modfile[STRING_SIZE2], model_prefix[STRING_SIZE2];
	const char *primary_suffix;

	if (INVMAT1 == 1) primary_suffix = "vs";
	else if (INVMAT1 == 3) primary_suffix = "mu";
	else err("model_freq_out_SH_visc: INVMAT1 must be 1 (Vs) or 3 (mu)");

	sprintf(model_prefix,"%s_stage_%d",INV_MODELFILE,nstage);
	sprintf(modfile,"%s.%s",model_prefix,primary_suffix);
	writemod(modfile,primary,3);
	MPI_Barrier(MPI_COMM_WORLD);
	if (MYID==0) mergemod(modfile,3);
	MPI_Barrier(MPI_COMM_WORLD);
	sprintf(modfile,"%s.%s.%i.%i",model_prefix,primary_suffix,POS[1],POS[2]);
	remove(modfile);

	sprintf(modfile,"%s.rho",model_prefix);
	writemod(modfile,rho,3);
	MPI_Barrier(MPI_COMM_WORLD);
	if (MYID==0) mergemod(modfile,3);
	MPI_Barrier(MPI_COMM_WORLD);
	sprintf(modfile,"%s.rho.%i.%i",model_prefix,POS[1],POS[2]);
	remove(modfile);

	sprintf(modfile,"%s.qs",model_prefix);
	writemod(modfile,physical_q,3);
	MPI_Barrier(MPI_COMM_WORLD);
	if (MYID==0) mergemod(modfile,3);
	MPI_Barrier(MPI_COMM_WORLD);
	sprintf(modfile,"%s.qs.%i.%i",model_prefix,POS[1],POS[2]);
	remove(modfile);
}

