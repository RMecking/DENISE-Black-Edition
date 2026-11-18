/*------------------------------------------------------------------------
 *   output source signal e.g. for cross-correlation of comparison with analytical solutions                                  
 *   last update 08/06/05, T. Bohlen
 *  ----------------------------------------------------------------------*/
#include "fd.h"
#include "segy.h"
#include "su_write.h"

void  output_source_signal(FILE *fp, float **signals, int ns, int seis_form){

	/* declaration of extern variables */
	extern float DT;
	extern int NDT;
	
	/* declaration of local variables */
	int i,j, ntr=1;
	segy tr;
	struct su_sampling sampling;

	switch(seis_form){
	case 1 :
		sampling = su_output_sampling(fp, ns, NDT, DT, 0, 0.0f);
		memset(&tr, 0, SU_WRITE_HEADER_BYTES);
		tr.tracl=1;
		tr.trid=1;
		tr.ns=(unsigned short)sampling.count;
		tr.dt=sampling.interval_us;
		tr.d1=(float)NDT*DT;
		su_write_trace(fp, &tr, signals[1], sampling);
		break;


	case 2 :
		for(i=1;i<=ntr;i++){         /*ASCII ONE COLUMN*/
			for(j=1;j<=ns;j+=NDT) fprintf(fp,"%e\n", signals[i][j]);
		}
		break;

	case 3 :                             /*BINARY */

		for(i=1;i<=ntr;i++)
			for(j=1;j<=ns;j+=NDT){
				fwrite(&signals[i][j],sizeof(float),1,fp); }
		break;

	default :
		fprintf(stdout," Message from output_source_signal: Don't know data format for seismograms !\n");
		fprintf(stdout," No output written. ");
	}

	if (seis_form == 1)
		su_close_output(fp);
	else
		fclose(fp);
}
