/*  Update_s_visc_PML_PSV
 *
 *  updating stress components at gridpoints [nx1...nx2][ny1...ny2]
 *  by a staggered grid finite difference scheme of arbitrary (FDORDER) order accuracy in space
 *  and second order accuracy in time for the visco-elastic PSV problem
 *   
 *  Daniel Koehn
 *  Kiel, 24.07.2016
 *  ----------------------------------------------------------------------*/

#include "fd.h"

void visco_psv_exact_strain(int j, int i, float vxx, float vyx,
                            float vxy, float vyy);

#ifdef DENISE_REGION_TEST_HOOKS
static size_t region_test_fd8_l1_fast_cells;

void update_s_visc_PSV_region_test_reset(void) {
	region_test_fd8_l1_fast_cells=0;
}

size_t update_s_visc_PSV_region_test_fast_cells(void) {
	return region_test_fd8_l1_fast_cells;
}
#endif

void update_s_visc_PML_PSV(int nx1, int nx2, int ny1, int ny2,
	int domain_nx, int domain_ny,
	float **  vx, float **   vy, float **  ux, float **   uy, float **  uxy, float **   uyx, float **   sxx, float **   syy,
	float **   sxy, float ** pi, float ** u, float ** uipjp, float **rho, float *hc, int infoout,
	float ***r, float ***p, float ***q, float **fipjp, float **f, float **g, float *bip, float *bjm, float *cip, float *cjm, float ***d, float ***e, float ***dip, 
      float * K_x, float * a_x, float * b_x, float * K_x_half, float * a_x_half, float * b_x_half,
      float * K_y, float * a_y, float * b_y, float * K_y_half, float * a_y_half, float * b_y_half,
      float ** psi_vxx, float ** psi_vyy, float ** psi_vxy, float ** psi_vyx, int mode){

	int i,j, m, fdoh, h, h1, l;
	int bulk_ix1, bulk_ix2, bulk_iy1, bulk_iy2, bulk_fd8_l1;
	float  vxx, vyy, vxy, vyx;
	float  dhi, dthalbe;	
	extern float DT, DH;
	extern int MYID, FDORDER, FW, L, GRAD_FORM;
        extern int FREE_SURF, BOUNDARY;
	extern int NPROCX, NPROCY, POS[3];
	extern FILE *FP;
	double time1, time2;
	
	float sumr=0.0, sump=0.0, sumq=0.0;
	
	

	/*dhi = DT/DH;*/
	dhi=1.0/DH;
	fdoh = FDORDER/2;
	dthalbe = DT/2.0;

	
	if (infoout && (MYID==0)){
		time1=MPI_Wtime();
		fprintf(FP,"\n **Message from update_s (printed by PE %d):\n",MYID);
		fprintf(FP," Updating stress components ...");
	}

	/* FD8/L=1 is the dominant production case.  Traverse its regular interior
	 * through contiguous row aliases, then let the existing case-8 loop handle
	 * only the boundary strips and corners.  The generic L path remains unchanged. */
	bulk_ix1=nx1;
	bulk_ix2=nx2;
	bulk_iy1=ny1;
	bulk_iy2=ny2;
	if ((!BOUNDARY) && (POS[1]==0) && (bulk_ix1<=FW)) bulk_ix1=FW+1;
	if ((!BOUNDARY) && (POS[1]==NPROCX-1) && (bulk_ix2>domain_nx-FW))
		bulk_ix2=domain_nx-FW;
	if ((POS[2]==0) && (bulk_iy1<=FW)) bulk_iy1=FW+1;
	if ((POS[2]==NPROCY-1) && (bulk_iy2>domain_ny-FW))
		bulk_iy2=domain_ny-FW;
	bulk_fd8_l1=(FDORDER==8) && (L==1) &&
	             (bulk_ix1<=bulk_ix2) && (bulk_iy1<=bulk_iy2);
	if (bulk_fd8_l1) {
#ifdef DENISE_REGION_TEST_HOOKS
		region_test_fd8_l1_fast_cells+=(size_t)(bulk_ix2-bulk_ix1+1)*
		                                  (size_t)(bulk_iy2-bulk_iy1+1);
#endif
		const float hc1=hc[1], hc2=hc[2], hc3=hc[3], hc4=hc[4];
		const float bip1=bip[1], bjm1=bjm[1], cip1=cip[1], cjm1=cjm[1];
		for (j=bulk_iy1;j<=bulk_iy2;j++) {
			float *restrict vxjm3=vx[j-3], *restrict vxjm2=vx[j-2];
			float *restrict vxjm1=vx[j-1], *restrict vxj=vx[j];
			float *restrict vxjp1=vx[j+1], *restrict vxjp2=vx[j+2];
			float *restrict vxjp3=vx[j+3], *restrict vxjp4=vx[j+4];
			float *restrict vyjm4=vy[j-4], *restrict vyjm3=vy[j-3];
			float *restrict vyjm2=vy[j-2], *restrict vyjm1=vy[j-1];
			float *restrict vyj=vy[j], *restrict vyjp1=vy[j+1];
			float *restrict vyjp2=vy[j+2], *restrict vyjp3=vy[j+3];
			float *restrict sxxj=sxx[j], *restrict syyj=syy[j], *restrict sxyj=sxy[j];
			float *restrict fj=f[j], *restrict gj=g[j], *restrict fipjpj=fipjp[j];
			float *restrict uxj=ux[j], *restrict uyj=uy[j], *restrict uxyj=uxy[j];
			float *restrict r1=&r[j][bulk_ix1][1]-bulk_ix1;
			float *restrict p1=&p[j][bulk_ix1][1]-bulk_ix1;
			float *restrict q1=&q[j][bulk_ix1][1]-bulk_ix1;
			float *restrict dip1=&dip[j][bulk_ix1][1]-bulk_ix1;
			float *restrict d1=&d[j][bulk_ix1][1]-bulk_ix1;
			float *restrict e1=&e[j][bulk_ix1][1]-bulk_ix1;
#pragma GCC ivdep
			for (i=bulk_ix1;i<=bulk_ix2;i++) {
				float sr=0.0, sp=0.0, sq=0.0;
				float lvxx, lvyx, lvxy, lvyy;
				lvxx=(hc1*(vxj[i]-vxj[i-1])+hc2*(vxj[i+1]-vxj[i-2])+
				     hc3*(vxj[i+2]-vxj[i-3])+hc4*(vxj[i+3]-vxj[i-4]))*dhi;
				lvyx=(hc1*(vyj[i+1]-vyj[i])+hc2*(vyj[i+2]-vyj[i-1])+
				     hc3*(vyj[i+3]-vyj[i-2])+hc4*(vyj[i+4]-vyj[i-3]))*dhi;
				lvxy=(hc1*(vxjp1[i]-vxj[i])+hc2*(vxjp2[i]-vxjm1[i])+
				     hc3*(vxjp3[i]-vxjm2[i])+hc4*(vxjp4[i]-vxjm3[i]))*dhi;
				lvyy=(hc1*(vyj[i]-vyjm1[i])+hc2*(vyjp1[i]-vyjm2[i])+
				     hc3*(vyjp2[i]-vyjm3[i])+hc4*(vyjp3[i]-vyjm4[i]))*dhi;
				sr+=r1[i]; sp+=p1[i]; sq+=q1[i];
				sxyj[i]+=(fipjpj[i]*(lvxy+lvyx))+(dthalbe*sr);
				sxxj[i]+=(gj[i]*(lvxx+lvyy))-(2.0*fj[i]*lvyy)+(dthalbe*sp);
				syyj[i]+=(gj[i]*(lvxx+lvyy))-(2.0*fj[i]*lvxx)+(dthalbe*sq);
				sr=sp=sq=0.0;
				r1[i]=bip1*(r1[i]*cip1-(dip1[i]*(lvxy+lvyx)));
				p1[i]=bjm1*(p1[i]*cjm1-(e1[i]*(lvxx+lvyy))+(2.0*d1[i]*lvyy));
				q1[i]=bjm1*(q1[i]*cjm1-(e1[i]*(lvxx+lvyy))+(2.0*d1[i]*lvxx));
				sr+=r1[i]; sp+=p1[i]; sq+=q1[i];
				sxyj[i]+=(dthalbe*sr);
				sxxj[i]+=(dthalbe*sp);
				syyj[i]+=(dthalbe*sq);
			}
			if((mode==0)&&(GRAD_FORM==2))
				for (i=bulk_ix1;i<=bulk_ix2;i++) {
					uxj[i]=p1[i]; uyj[i]=q1[i]; uxyj[i]=r1[i];
				}
		}
	}
	


	switch (FDORDER){

	case 2:
		for (j=ny1;j<=ny2;j++){
			for (i=nx1;i<=nx2;i++){
			vxx = (  hc[1]*(vx[j][i]  -vx[j][i-1]))*dhi;
			
			vyx = (  hc[1]*(vy[j][i+1]-vy[j][i]))*dhi;

                        vxy = (  hc[1]*(vx[j+1][i]-vx[j][i]))*dhi;

                        vyy = (  hc[1]*(vy[j][i]  -vy[j-1][i]))*dhi; 

        /* left boundary */                                         
        if((!BOUNDARY) && (POS[1]==0) && (i<=FW)){
                        
                        psi_vxx[j][i] = b_x[i] * psi_vxx[j][i] + a_x[i] * vxx;
                        vxx = vxx / K_x[i] + psi_vxx[j][i];

                        psi_vyx[j][i] = b_x_half[i] * psi_vyx[j][i] + a_x_half[i] * vyx;
                        vyx = vyx / K_x_half[i] + psi_vyx[j][i];                 
         }

        /* right boundary */                                         
        if((!BOUNDARY) && (POS[1]==NPROCX-1) && (i>=domain_nx-FW+1)){
		
                        h1 = (i-domain_nx+2*FW);
                        h = i;
                        
                        psi_vxx[j][h1] = b_x[h1] * psi_vxx[j][h1] + a_x[h1] * vxx;
                        vxx = vxx / K_x[h1] + psi_vxx[j][h1]; 

                        /*psi_vyx[j][h] = b_x_half[h] * psi_vyx[j][h] + a_x_half[h] * vyx;
                        vyx = vyx / K_x_half[h] + psi_vyx[j][h];*/
                        
                        psi_vyx[j][h1] = b_x_half[h1] * psi_vyx[j][h1] + a_x_half[h1] * vyx;
			vyx = vyx / K_x_half[h1] + psi_vyx[j][h1];
                                           
         }

	  /* top boundary */                                         
        if((POS[2]==0) && (!(FREE_SURF)) && (j<=FW)){
                                                
                        psi_vyy[j][i] = b_y[j] * psi_vyy[j][i] + a_y[j] * vyy;                                            
                        psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                     
                        vyy = vyy / K_y[j] + psi_vyy[j][i];
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];

        }
	
	  /* bottom boundary */                                         
        if((POS[2]==NPROCY-1) && (j>=domain_ny-FW+1)){

                        h1 = (j-domain_ny+2*FW);
                        h = j;
                                                
                        psi_vyy[h1][i] = b_y[h1] * psi_vyy[h1][i] + a_y[h1] * vyy;                                            
                        vyy = vyy / K_y[h1] + psi_vyy[h1][i];
                        
                        /*psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];*/
                        
                        psi_vxy[h1][i] = b_y_half[h1] * psi_vxy[h1][i] + a_y_half[h1] * vxy;
			vxy = vxy / K_y_half[h1] + psi_vxy[h1][i];
        
        }
	
	/* computing sums of the old memory variables */
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				sumr+=r[j][i][l];
				sump+=p[j][i][l];
				sumq+=q[j][i][l];
			}
			
			
                        /* updating components of the stress tensor, partially */
			sxy[j][i] += (fipjp[j][i]*(vxy+vyx))+(dthalbe*sumr);
			sxx[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vyy)+(dthalbe*sump);
			syy[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vxx)+(dthalbe*sumq);
				
			
			/* now updating the memory-variables and sum them up*/
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				r[j][i][l] = bip[l]*(r[j][i][l]*cip[l]-(dip[j][i][l]*(vxy+vyx)));
				p[j][i][l] = bjm[l]*(p[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vyy));
				q[j][i][l] = bjm[l]*(q[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vxx));
				sumr += r[j][i][l];
				sump += p[j][i][l];
				sumq += q[j][i][l];
			}
			
			
			/* and now the components of the stress tensor are
			   completely updated */
			sxy[j][i]+=(dthalbe*sumr);
			sxx[j][i]+=(dthalbe*sump);
			syy[j][i]+=(dthalbe*sumq);

			/* save forward wavefield for gradient calculation */
			if((mode==0)&&(GRAD_FORM==2)){
			  ux[j][i] = sump;
		          uy[j][i] = sumq;
			  uxy[j][i] = sumr;
			}			

		}
		}
		break;

	case 4:
		for (j=ny1;j<=ny2;j++){
			for (i=nx1;i<=nx2;i++){
			vxx = (  hc[1]*(vx[j][i]  -vx[j][i-1])
				       + hc[2]*(vx[j][i+1]-vx[j][i-2]))*dhi;
			
			vyx = (  hc[1]*(vy[j][i+1]-vy[j][i])
				       + hc[2]*(vy[j][i+2]-vy[j][i-1]))*dhi;

                        vxy = (  hc[1]*(vx[j+1][i]-vx[j][i])
				       + hc[2]*(vx[j+2][i]-vx[j-1][i]))*dhi;

                        vyy = (  hc[1]*(vy[j][i]  -vy[j-1][i])
				       + hc[2]*(vy[j+1][i]-vy[j-2][i]))*dhi; 

        /* left boundary */                                         
        if((!BOUNDARY) && (POS[1]==0) && (i<=FW)){
                        
                        psi_vxx[j][i] = b_x[i] * psi_vxx[j][i] + a_x[i] * vxx;
                        vxx = vxx / K_x[i] + psi_vxx[j][i];

                        psi_vyx[j][i] = b_x_half[i] * psi_vyx[j][i] + a_x_half[i] * vyx;
                        vyx = vyx / K_x_half[i] + psi_vyx[j][i];                 
         }

        /* right boundary */                                         
        if((!BOUNDARY) && (POS[1]==NPROCX-1) && (i>=domain_nx-FW+1)){
		
                        h1 = (i-domain_nx+2*FW);
                        h = i;
                        
                        psi_vxx[j][h1] = b_x[h1] * psi_vxx[j][h1] + a_x[h1] * vxx;
                        vxx = vxx / K_x[h1] + psi_vxx[j][h1]; 

                        /*psi_vyx[j][h] = b_x_half[h] * psi_vyx[j][h] + a_x_half[h] * vyx;
                        vyx = vyx / K_x_half[h] + psi_vyx[j][h];*/
                        
                        psi_vyx[j][h1] = b_x_half[h1] * psi_vyx[j][h1] + a_x_half[h1] * vyx;
			vyx = vyx / K_x_half[h1] + psi_vyx[j][h1];
                                           
         }

	  /* top boundary */                                         
        if((POS[2]==0) && (!(FREE_SURF)) && (j<=FW)){
                                                
                        psi_vyy[j][i] = b_y[j] * psi_vyy[j][i] + a_y[j] * vyy;                                            
                        psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                     
                        vyy = vyy / K_y[j] + psi_vyy[j][i];
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];

        }
	
	  /* bottom boundary */                                         
        if((POS[2]==NPROCY-1) && (j>=domain_ny-FW+1)){

                        h1 = (j-domain_ny+2*FW);
                        h = j;
                                                
                        psi_vyy[h1][i] = b_y[h1] * psi_vyy[h1][i] + a_y[h1] * vyy;                                            
                        vyy = vyy / K_y[h1] + psi_vyy[h1][i];
                        
                        /*psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];*/
                        
                        psi_vxy[h1][i] = b_y_half[h1] * psi_vxy[h1][i] + a_y_half[h1] * vxy;
			vxy = vxy / K_y_half[h1] + psi_vxy[h1][i];
        
        }

	if(mode==0) visco_psv_exact_strain(j,i,vxx,vyx,vxy,vyy);

	/* computing sums of the old memory variables */
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				sumr+=r[j][i][l];
				sump+=p[j][i][l];
				sumq+=q[j][i][l];
			}


                        /* updating components of the stress tensor, partially */
			sxy[j][i] += (fipjp[j][i]*(vxy+vyx))+(dthalbe*sumr);
			sxx[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vyy)+(dthalbe*sump);
			syy[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vxx)+(dthalbe*sumq);
			
			
			/* now updating the memory-variables and sum them up*/
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				r[j][i][l] = bip[l]*(r[j][i][l]*cip[l]-(dip[j][i][l]*(vxy+vyx)));
				p[j][i][l] = bjm[l]*(p[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0f*d[j][i][l]*vyy));
				q[j][i][l] = bjm[l]*(q[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0f*d[j][i][l]*vxx));
				sumr += r[j][i][l];
				sump += p[j][i][l];
				sumq += q[j][i][l];
			}

			/* and now the components of the stress tensor are
			   completely updated */
			sxy[j][i]+=(dthalbe*sumr);
			sxx[j][i]+=(dthalbe*sump);
			syy[j][i]+=(dthalbe*sumq);

			/* save forward wavefield for gradient calculation */
			if((mode==0)&&(GRAD_FORM==2)){
			  ux[j][i] = sump;
		          uy[j][i] = sumq;
			  uxy[j][i] = sumr;
			}			
			
			}
		}
		break;

	case 6:
		for (j=ny1;j<=ny2;j++){
			for (i=nx1;i<=nx2;i++){
			vxx = (  hc[1]*(vx[j][i]  -vx[j][i-1])
				       + hc[2]*(vx[j][i+1]-vx[j][i-2])
				       + hc[3]*(vx[j][i+2]-vx[j][i-3]))*dhi;
			
			vyx = (  hc[1]*(vy[j][i+1]-vy[j][i])
				       + hc[2]*(vy[j][i+2]-vy[j][i-1])
				       + hc[3]*(vy[j][i+3]-vy[j][i-2]))*dhi;

                        vxy = (  hc[1]*(vx[j+1][i]-vx[j][i])
				       + hc[2]*(vx[j+2][i]-vx[j-1][i])
				       + hc[3]*(vx[j+3][i]-vx[j-2][i]))*dhi;

                        vyy = (  hc[1]*(vy[j][i]  -vy[j-1][i])
				       + hc[2]*(vy[j+1][i]-vy[j-2][i])
				       + hc[3]*(vy[j+2][i]-vy[j-3][i]))*dhi; 

        /* left boundary */                                         
        if((!BOUNDARY) && (POS[1]==0) && (i<=FW)){
                        
                        psi_vxx[j][i] = b_x[i] * psi_vxx[j][i] + a_x[i] * vxx;
                        vxx = vxx / K_x[i] + psi_vxx[j][i];

                        psi_vyx[j][i] = b_x_half[i] * psi_vyx[j][i] + a_x_half[i] * vyx;
                        vyx = vyx / K_x_half[i] + psi_vyx[j][i];                 
         }

        /* right boundary */                                         
        if((!BOUNDARY) && (POS[1]==NPROCX-1) && (i>=domain_nx-FW+1)){
		
                        h1 = (i-domain_nx+2*FW);
                        h = i;
                        
                        psi_vxx[j][h1] = b_x[h1] * psi_vxx[j][h1] + a_x[h1] * vxx;
                        vxx = vxx / K_x[h1] + psi_vxx[j][h1]; 

                        /*psi_vyx[j][h] = b_x_half[h] * psi_vyx[j][h] + a_x_half[h] * vyx;
                        vyx = vyx / K_x_half[h] + psi_vyx[j][h];*/
                        
                        psi_vyx[j][h1] = b_x_half[h1] * psi_vyx[j][h1] + a_x_half[h1] * vyx;
			vyx = vyx / K_x_half[h1] + psi_vyx[j][h1];
                                           
         }

	  /* top boundary */                                         
        if((POS[2]==0) && (!(FREE_SURF)) && (j<=FW)){
                                                
                        psi_vyy[j][i] = b_y[j] * psi_vyy[j][i] + a_y[j] * vyy;                                            
                        psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                     
                        vyy = vyy / K_y[j] + psi_vyy[j][i];
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];

        }
	
	  /* bottom boundary */                                         
        if((POS[2]==NPROCY-1) && (j>=domain_ny-FW+1)){

                        h1 = (j-domain_ny+2*FW);
                        h = j;
                                                
                        psi_vyy[h1][i] = b_y[h1] * psi_vyy[h1][i] + a_y[h1] * vyy;                                            
                        vyy = vyy / K_y[h1] + psi_vyy[h1][i];
                        
                        /*psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];*/
                        
                        psi_vxy[h1][i] = b_y_half[h1] * psi_vxy[h1][i] + a_y_half[h1] * vxy;
			vxy = vxy / K_y_half[h1] + psi_vxy[h1][i];
        
        }

	/* computing sums of the old memory variables */
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				sumr+=r[j][i][l];
				sump+=p[j][i][l];
				sumq+=q[j][i][l];
			}


                        /* updating components of the stress tensor, partially */
			sxy[j][i] += (fipjp[j][i]*(vxy+vyx))+(dthalbe*sumr);
			sxx[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vyy)+(dthalbe*sump);
			syy[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vxx)+(dthalbe*sumq);
			
			
			/* now updating the memory-variables and sum them up*/
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				r[j][i][l] = bip[l]*(r[j][i][l]*cip[l]-(dip[j][i][l]*(vxy+vyx)));
				p[j][i][l] = bjm[l]*(p[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vyy));
				q[j][i][l] = bjm[l]*(q[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vxx));
				sumr += r[j][i][l];
				sump += p[j][i][l];
				sumq += q[j][i][l];
			}

			/* and now the components of the stress tensor are
			   completely updated */
			sxy[j][i]+=(dthalbe*sumr);
			sxx[j][i]+=(dthalbe*sump);
			syy[j][i]+=(dthalbe*sumq);

			/* save forward wavefield for gradient calculation */
			if((mode==0)&&(GRAD_FORM==2)){
			  ux[j][i] = sump;
		          uy[j][i] = sumq;
			  uxy[j][i] = sumr;
			}			

			}
		}
		break;

	case 8:

    for (j=ny1;j<=ny2;j++){
	for (i=nx1;i<=nx2;i++){
			if (bulk_fd8_l1 && i>=bulk_ix1 && i<=bulk_ix2 &&
			    j>=bulk_iy1 && j<=bulk_iy2) continue;

			vxx = (  hc[1]*(vx[j][i]  -vx[j][i-1])
				       + hc[2]*(vx[j][i+1]-vx[j][i-2])
				       + hc[3]*(vx[j][i+2]-vx[j][i-3])
				       + hc[4]*(vx[j][i+3]-vx[j][i-4]))*dhi;
			
			vyx = (  hc[1]*(vy[j][i+1]-vy[j][i])
				       + hc[2]*(vy[j][i+2]-vy[j][i-1])
				       + hc[3]*(vy[j][i+3]-vy[j][i-2])
				       + hc[4]*(vy[j][i+4]-vy[j][i-3]))*dhi;

                        vxy = (  hc[1]*(vx[j+1][i]-vx[j][i])
				       + hc[2]*(vx[j+2][i]-vx[j-1][i])
				       + hc[3]*(vx[j+3][i]-vx[j-2][i])
				       + hc[4]*(vx[j+4][i]-vx[j-3][i]))*dhi;

                        vyy = (  hc[1]*(vy[j][i]  -vy[j-1][i])
				       + hc[2]*(vy[j+1][i]-vy[j-2][i])
				       + hc[3]*(vy[j+2][i]-vy[j-3][i])
				       + hc[4]*(vy[j+3][i]-vy[j-4][i]))*dhi; 

        /* left boundary */                                         
        if((!BOUNDARY) && (POS[1]==0) && (i<=FW)){
                        
                        psi_vxx[j][i] = b_x[i] * psi_vxx[j][i] + a_x[i] * vxx;
                        vxx = vxx / K_x[i] + psi_vxx[j][i];

                        psi_vyx[j][i] = b_x_half[i] * psi_vyx[j][i] + a_x_half[i] * vyx;
                        vyx = vyx / K_x_half[i] + psi_vyx[j][i];                 
         }

        /* right boundary */                                         
        if((!BOUNDARY) && (POS[1]==NPROCX-1) && (i>=domain_nx-FW+1)){
		
                        h1 = (i-domain_nx+2*FW);
                        h = i;
                        
                        psi_vxx[j][h1] = b_x[h1] * psi_vxx[j][h1] + a_x[h1] * vxx;
                        vxx = vxx / K_x[h1] + psi_vxx[j][h1]; 

                        /*psi_vyx[j][h] = b_x_half[h] * psi_vyx[j][h] + a_x_half[h] * vyx;
                        vyx = vyx / K_x_half[h] + psi_vyx[j][h];*/
                        
                        psi_vyx[j][h1] = b_x_half[h1] * psi_vyx[j][h1] + a_x_half[h1] * vyx;
			vyx = vyx / K_x_half[h1] + psi_vyx[j][h1];
                                           
         }

	  /* top boundary */                                         
        if((POS[2]==0) && (!(FREE_SURF)) && (j<=FW)){
                                                
                        psi_vyy[j][i] = b_y[j] * psi_vyy[j][i] + a_y[j] * vyy;                                            
                        psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                     
                        vyy = vyy / K_y[j] + psi_vyy[j][i];
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];

        }
	
	  /* bottom boundary */                                         
        if((POS[2]==NPROCY-1) && (j>=domain_ny-FW+1)){

                        h1 = (j-domain_ny+2*FW);
                        h = j;
                                                
                        psi_vyy[h1][i] = b_y[h1] * psi_vyy[h1][i] + a_y[h1] * vyy;                                            
                        vyy = vyy / K_y[h1] + psi_vyy[h1][i];
                        
                        /*psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];*/
                        
                        psi_vxy[h1][i] = b_y_half[h1] * psi_vxy[h1][i] + a_y_half[h1] * vxy;
			vxy = vxy / K_y_half[h1] + psi_vxy[h1][i];
        
        }

	/* computing sums of the old memory variables */
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				sumr+=r[j][i][l];
				sump+=p[j][i][l];
				sumq+=q[j][i][l];
			}


                        /* updating components of the stress tensor, partially */
			sxy[j][i] += (fipjp[j][i]*(vxy+vyx))+(dthalbe*sumr);
			sxx[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vyy)+(dthalbe*sump);
			syy[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vxx)+(dthalbe*sumq);
			
			
			/* now updating the memory-variables and sum them up*/
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				r[j][i][l] = bip[l]*(r[j][i][l]*cip[l]-(dip[j][i][l]*(vxy+vyx)));
				p[j][i][l] = bjm[l]*(p[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vyy));
				q[j][i][l] = bjm[l]*(q[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vxx));
				sumr += r[j][i][l];
				sump += p[j][i][l];
				sumq += q[j][i][l];
			}

			/* and now the components of the stress tensor are
			   completely updated */
			sxy[j][i]+=(dthalbe*sumr);
			sxx[j][i]+=(dthalbe*sump);
			syy[j][i]+=(dthalbe*sumq);
			
			/* save forward wavefield for gradient calculation */
			if((mode==0)&&(GRAD_FORM==2)){
			  ux[j][i] = sump;
		          uy[j][i] = sumq;
			  uxy[j][i] = sumr;
			}			

   }}
		break;

	case 10:
		for (j=ny1;j<=ny2;j++){
			for (i=nx1;i<=nx2;i++){
			
			vxx = (  hc[1]*(vx[j][i]  -vx[j][i-1])
				       + hc[2]*(vx[j][i+1]-vx[j][i-2])
				       + hc[3]*(vx[j][i+2]-vx[j][i-3])
				       + hc[4]*(vx[j][i+3]-vx[j][i-4])
				       + hc[5]*(vx[j][i+4]-vx[j][i-5]))*dhi;
			
			vyx = (  hc[1]*(vy[j][i+1]-vy[j][i])
				       + hc[2]*(vy[j][i+2]-vy[j][i-1])
				       + hc[3]*(vy[j][i+3]-vy[j][i-2])
				       + hc[4]*(vy[j][i+4]-vy[j][i-3])
				       + hc[5]*(vy[j][i+5]-vy[j][i-4]))*dhi;

                        vxy = (  hc[1]*(vx[j+1][i]-vx[j][i])
				       + hc[2]*(vx[j+2][i]-vx[j-1][i])
				       + hc[3]*(vx[j+3][i]-vx[j-2][i])
				       + hc[4]*(vx[j+4][i]-vx[j-3][i])
				       + hc[5]*(vx[j+5][i]-vx[j-4][i]))*dhi;

                        vyy = (  hc[1]*(vy[j][i]  -vy[j-1][i])
				       + hc[2]*(vy[j+1][i]-vy[j-2][i])
				       + hc[3]*(vy[j+2][i]-vy[j-3][i])
				       + hc[4]*(vy[j+3][i]-vy[j-4][i])
				       + hc[5]*(vy[j+4][i]-vy[j-5][i]))*dhi; 

        /* left boundary */                                         
        if((!BOUNDARY) && (POS[1]==0) && (i<=FW)){
                        
                        psi_vxx[j][i] = b_x[i] * psi_vxx[j][i] + a_x[i] * vxx;
                        vxx = vxx / K_x[i] + psi_vxx[j][i];

                        psi_vyx[j][i] = b_x_half[i] * psi_vyx[j][i] + a_x_half[i] * vyx;
                        vyx = vyx / K_x_half[i] + psi_vyx[j][i];                 
         }

        /* right boundary */                                         
        if((!BOUNDARY) && (POS[1]==NPROCX-1) && (i>=domain_nx-FW+1)){
		
                        h1 = (i-domain_nx+2*FW);
                        h = i;
                        
                        psi_vxx[j][h1] = b_x[h1] * psi_vxx[j][h1] + a_x[h1] * vxx;
                        vxx = vxx / K_x[h1] + psi_vxx[j][h1]; 

                        /*psi_vyx[j][h] = b_x_half[h] * psi_vyx[j][h] + a_x_half[h] * vyx;
                        vyx = vyx / K_x_half[h] + psi_vyx[j][h];*/
                        
                        psi_vyx[j][h1] = b_x_half[h1] * psi_vyx[j][h1] + a_x_half[h1] * vyx;
			vyx = vyx / K_x_half[h1] + psi_vyx[j][h1];
                                           
         }

	  /* top boundary */                                         
        if((POS[2]==0) && (!(FREE_SURF)) && (j<=FW)){
                                                
                        psi_vyy[j][i] = b_y[j] * psi_vyy[j][i] + a_y[j] * vyy;                                            
                        psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                     
                        vyy = vyy / K_y[j] + psi_vyy[j][i];
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];

        }
	
	  /* bottom boundary */                                         
        if((POS[2]==NPROCY-1) && (j>=domain_ny-FW+1)){

                        h1 = (j-domain_ny+2*FW);
                        h = j;
                                                
                        psi_vyy[h1][i] = b_y[h1] * psi_vyy[h1][i] + a_y[h1] * vyy;                                            
                        vyy = vyy / K_y[h1] + psi_vyy[h1][i];
                        
                        /*psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];*/
                        
                        psi_vxy[h1][i] = b_y_half[h1] * psi_vxy[h1][i] + a_y_half[h1] * vxy;
			vxy = vxy / K_y_half[h1] + psi_vxy[h1][i];
        
        }

	/* computing sums of the old memory variables */
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				sumr+=r[j][i][l];
				sump+=p[j][i][l];
				sumq+=q[j][i][l];
			}


                        /* updating components of the stress tensor, partially */
			sxy[j][i] += (fipjp[j][i]*(vxy+vyx))+(dthalbe*sumr);
			sxx[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vyy)+(dthalbe*sump);
			syy[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vxx)+(dthalbe*sumq);
			
			
			/* now updating the memory-variables and sum them up*/
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				r[j][i][l] = bip[l]*(r[j][i][l]*cip[l]-(dip[j][i][l]*(vxy+vyx)));
				p[j][i][l] = bjm[l]*(p[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vyy));
				q[j][i][l] = bjm[l]*(q[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vxx));
				sumr += r[j][i][l];
				sump += p[j][i][l];
				sumq += q[j][i][l];
			}

			/* and now the components of the stress tensor are
			   completely updated */
			sxy[j][i]+=(dthalbe*sumr);
			sxx[j][i]+=(dthalbe*sump);
			syy[j][i]+=(dthalbe*sumq);
			
			/* save forward wavefield for gradient calculation */
			if((mode==0)&&(GRAD_FORM==2)){
			  ux[j][i] = sump;
		          uy[j][i] = sumq;
			  uxy[j][i] = sumr;
			}			
			

			}
		}
		break;
		
	case 12:
		for (j=ny1;j<=ny2;j++){
			for (i=nx1;i<=nx2;i++){
			
			vxx = (  hc[1]*(vx[j][i]  -vx[j][i-1])
				       + hc[2]*(vx[j][i+1]-vx[j][i-2])
				       + hc[3]*(vx[j][i+2]-vx[j][i-3])
				       + hc[4]*(vx[j][i+3]-vx[j][i-4])
				       + hc[5]*(vx[j][i+4]-vx[j][i-5])
				       + hc[6]*(vx[j][i+5]-vx[j][i-6]))*dhi;
			
			vyx = (  hc[1]*(vy[j][i+1]-vy[j][i])
				       + hc[2]*(vy[j][i+2]-vy[j][i-1])
				       + hc[3]*(vy[j][i+3]-vy[j][i-2])
				       + hc[4]*(vy[j][i+4]-vy[j][i-3])
				       + hc[5]*(vy[j][i+5]-vy[j][i-4])
				       + hc[6]*(vy[j][i+6]-vy[j][i-5]))*dhi;

                        vxy = (  hc[1]*(vx[j+1][i]-vx[j][i])
				       + hc[2]*(vx[j+2][i]-vx[j-1][i])
				       + hc[3]*(vx[j+3][i]-vx[j-2][i])
				       + hc[4]*(vx[j+4][i]-vx[j-3][i])
				       + hc[5]*(vx[j+5][i]-vx[j-4][i])
				       + hc[6]*(vx[j+6][i]-vx[j-5][i]))*dhi;

                        vyy = (  hc[1]*(vy[j][i]  -vy[j-1][i])
				       + hc[2]*(vy[j+1][i]-vy[j-2][i])
				       + hc[3]*(vy[j+2][i]-vy[j-3][i])
				       + hc[4]*(vy[j+3][i]-vy[j-4][i])
				       + hc[5]*(vy[j+4][i]-vy[j-5][i])
				       + hc[6]*(vy[j+5][i]-vy[j-6][i]))*dhi; 

        /* left boundary */                                         
        if((!BOUNDARY) && (POS[1]==0) && (i<=FW)){
                        
                        psi_vxx[j][i] = b_x[i] * psi_vxx[j][i] + a_x[i] * vxx;
                        vxx = vxx / K_x[i] + psi_vxx[j][i];

                        psi_vyx[j][i] = b_x_half[i] * psi_vyx[j][i] + a_x_half[i] * vyx;
                        vyx = vyx / K_x_half[i] + psi_vyx[j][i];                 
         }

        /* right boundary */                                         
        if((!BOUNDARY) && (POS[1]==NPROCX-1) && (i>=domain_nx-FW+1)){
		
                        h1 = (i-domain_nx+2*FW);
                        h = i;
                        
                        psi_vxx[j][h1] = b_x[h1] * psi_vxx[j][h1] + a_x[h1] * vxx;
                        vxx = vxx / K_x[h1] + psi_vxx[j][h1]; 

                        /*psi_vyx[j][h] = b_x_half[h] * psi_vyx[j][h] + a_x_half[h] * vyx;
                        vyx = vyx / K_x_half[h] + psi_vyx[j][h];*/
                        
                        psi_vyx[j][h1] = b_x_half[h1] * psi_vyx[j][h1] + a_x_half[h1] * vyx;
			vyx = vyx / K_x_half[h1] + psi_vyx[j][h1];
                                           
         }

	  /* top boundary */                                         
        if((POS[2]==0) && (!(FREE_SURF)) && (j<=FW)){
                                                
                        psi_vyy[j][i] = b_y[j] * psi_vyy[j][i] + a_y[j] * vyy;                                            
                        psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                     
                        vyy = vyy / K_y[j] + psi_vyy[j][i];
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];

        }
	
	  /* bottom boundary */                                         
        if((POS[2]==NPROCY-1) && (j>=domain_ny-FW+1)){

                        h1 = (j-domain_ny+2*FW);
                        h = j;
                                                
                        psi_vyy[h1][i] = b_y[h1] * psi_vyy[h1][i] + a_y[h1] * vyy;                                            
                        vyy = vyy / K_y[h1] + psi_vyy[h1][i];
                        
                        /*psi_vxy[j][i] = b_y_half[j] * psi_vxy[j][i] + a_y_half[j] * vxy;
                        vxy = vxy / K_y_half[j] + psi_vxy[j][i];*/
                        
                        psi_vxy[h1][i] = b_y_half[h1] * psi_vxy[h1][i] + a_y_half[h1] * vxy;
			vxy = vxy / K_y_half[h1] + psi_vxy[h1][i];
        
        }

	                /* computing sums of the old memory variables */
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				sumr+=r[j][i][l];
				sump+=p[j][i][l];
				sumq+=q[j][i][l];
			}


                        /* updating components of the stress tensor, partially */
			sxy[j][i] += (fipjp[j][i]*(vxy+vyx))+(dthalbe*sumr);
			sxx[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vyy)+(dthalbe*sump);
			syy[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vxx)+(dthalbe*sumq);
			
			
			/* now updating the memory-variables and sum them up*/
			sumr=sump=sumq=0.0;
			for (l=1;l<=L;l++){
				r[j][i][l] = bip[l]*(r[j][i][l]*cip[l]-(dip[j][i][l]*(vxy+vyx)));
				p[j][i][l] = bjm[l]*(p[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vyy));
				q[j][i][l] = bjm[l]*(q[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vxx));
				sumr += r[j][i][l];
				sump += p[j][i][l];
				sumq += q[j][i][l];
			}

			/* and now the components of the stress tensor are
			   completely updated */
			sxy[j][i]+=(dthalbe*sumr);
			sxx[j][i]+=(dthalbe*sump);
			syy[j][i]+=(dthalbe*sumq);
			
			/* save forward wavefield for gradient calculation */
			if((mode==0)&&(GRAD_FORM==2)){
			  ux[j][i] = sump;
		          uy[j][i] = sumq;
			  uxy[j][i] = sumr;
			}			

		
			}
		}
		break;

		
	default:
		for (j=ny1;j<=ny2;j++){
			for (i=nx1;i<=nx2;i++){
				vxx = 0.0;
				vyy = 0.0;
				vyx = 0.0;
				vxy = 0.0;
				for (m=1; m<=fdoh; m++) {
					vxx += hc[m]*(vx[j][i+m-1] -vx[j][i-m]  );
					vyy += hc[m]*(vy[j+m-1][i] -vy[j-m][i]  );
					vyx += hc[m]*(vy[j][i+m]   -vy[j][i-m+1]);
					vxy += hc[m]*(vx[j+m][i]   -vx[j-m+1][i]);
				}
				vxx *= dhi;
				vyy *= dhi;
				vyx *= dhi;
				vxy *= dhi;

				sumr=sump=sumq=0.0;
				for (l=1;l<=L;l++){
					sumr+=r[j][i][l];
					sump+=p[j][i][l];
					sumq+=q[j][i][l];
				}

				sxy[j][i] += (fipjp[j][i]*(vxy+vyx))+(dthalbe*sumr);
				sxx[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vyy)+(dthalbe*sump);
				syy[j][i] += (g[j][i]*(vxx+vyy))-(2.0*f[j][i]*vxx)+(dthalbe*sumq);

				sumr=sump=sumq=0.0;
				for (l=1;l<=L;l++){
					r[j][i][l] = bip[l]*(r[j][i][l]*cip[l]-(dip[j][i][l]*(vxy+vyx)));
					p[j][i][l] = bjm[l]*(p[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vyy));
					q[j][i][l] = bjm[l]*(q[j][i][l]*cjm[l]-(e[j][i][l]*(vxx+vyy))+(2.0*d[j][i][l]*vxx));
					sumr += r[j][i][l];
					sump += p[j][i][l];
					sumq += q[j][i][l];
				}

				sxy[j][i]+=(dthalbe*sumr);
				sxx[j][i]+=(dthalbe*sump);
				syy[j][i]+=(dthalbe*sumq);
			}
		}
		break;
		
	} /* end of switch(FDORDER) */


	if (infoout && (MYID==0)){
		time2=MPI_Wtime();
		fprintf(FP," finished (real time: %4.2f s).\n",time2-time1);
	}
}
