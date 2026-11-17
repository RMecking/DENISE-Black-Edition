#define NR_END 1
#define FREE_ARG char *

#include "fd.h"
#include <stdint.h>


void err(char err_text[]){
        extern int MYID;

        fprintf(stdout,"Message from PE %d\n",MYID);
        fprintf(stdout,"R U N - T I M E  E R R O R: \n");
        fprintf(stdout,"%s\n",err_text);
        fprintf(stdout,"...now exiting to system.\n");

        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
}

void warning(char warn_text[]){
	/* standard warnings handler */
	fprintf(stdout,"W A R N I N G   M E S S A G E: \n");
	fprintf(stdout,"%s\n",warn_text);
}


double maximum(float **a, int nx, int ny){
	/* find absolute maximum of array a[1...nx][1...ny] */
	double maxi=0.0;
	int i, j;


	for (j=1;j<=ny;j++)
		for (i=1;i<=nx;i++)
			if (fabs((double) a[i][j])>maxi) maxi=fabs((double)a[i][j]);
	return maxi;
}

float minimum_m(float **mat, int nx, int ny)
{
        float minm;
        int i,j;

        minm = mat[1][1];
        for (i=1;i<=nx;i++)
                for (j=1;j<=ny;j++)
                {
                        if (i*j==1) continue;
                        if (mat[j][i] < minm)
                        {
                                minm = mat[j][i];
                        }
                }
        return minm;
}

float maximum_m(float **mat, int nx, int ny)
{
        float maxm;
        int i,j;

        maxm = mat[1][1];
        for (i=1;i<=nx;i++)
                for (j=1;j<=ny;j++)
                {
                        if (i*j==1) continue;
                        if (mat[j][i] > maxm)
                        {
                                maxm = mat[j][i];
                        }
                }
        return maxm;
}




float *vector(int nl, int nh){
	/* allocate a float vector with subscript range v[nl..nh] and initializing
		   this vector, eg. vector[nl..nh]=0.0 */
	float *v;
	int i;

	v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
	if (!v) err("allocation failure in function vector()");
	for (i=0;i<(nh-nl+1+NR_END);i++) v[i]=0.0;
	return v-nl+NR_END;
}


int *ivector(int nl, int nh){
	/* allocate an int vector with subscript range v[nl..nh] and initializing
		   this vector, eg. ivector[nl..nh]=0 */
	int *v;
	int i;

	v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
	if (!v) err("allocation failure in function ivector()");
	for (i=0;i<(nh-nl+1+NR_END);i++) v[i]=0;
	return v-nl+NR_END;
}

unsigned short int *usvector(int nl, int nh){
	/* allocate an short int vector with subscript range v[nl..nh] and initializing
		   this vector, eg. ivector[nl..nh]=0 */
	unsigned short int *v;
	int i;

	v=(unsigned short int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(unsigned short int)));
	if (!v) err("allocation failure in function usvector()");
	for (i=0;i<(nh-nl+1+NR_END);i++) v[i]=0;
	return v-nl+NR_END;
}


unsigned char *cvector(int nl, int nh){
	/* allocate an unsigned char vector with subscript range v[nl..nh] */
	unsigned char *v;

	v=(unsigned char *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(unsigned char)));
	if (!v) err("allocation failure in function cvector()");
	return v-nl+NR_END;
}


unsigned long *lvector(int nl, int nh){
	/* allocate an unsigned long vector with subscript range v[nl..nh] and
		  initializing this vector, eg. vector[nl..nh]=0.0 */
	unsigned long *v;
	int i;

	v=(unsigned long *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(unsigned long)));
	if (!v) err("allocation failure in function lvector()");
	for (i=0;i<(nh-nl+1+NR_END);i++) v[i]=0;
	return v-nl+NR_END;
}

double *dvector(int nl, int nh){
	/* allocate a double vector with subscript range v[nl..nh] and initializing
		   this vector, eg. vector[nl..nh]=0.0 */
	double *v;
	int i;

	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) err("allocation failure in function dvector()");
	for (i=0;i<(nh-nl+1+NR_END);i++) v[i]=0.0;
	return v-nl+NR_END;
}

/* NR matrices and tensors are flat within each level. Empty dimensions
   have a first-element anchor for pairing with free and count-zero calls.
   They have no logical samples. Bounds with a gap larger than one are errors. */

static size_t nr_extent(int low, int high)
{
    int64_t count = (int64_t)high - (int64_t)low + 1;
    if (count < 0 || (uint64_t)count > SIZE_MAX)
        err("invalid Numerical Recipes array bounds");
    return (size_t)count;
}

static size_t nr_product(size_t left, size_t right)
{
    if (right != 0 && left > SIZE_MAX / right)
        err("Numerical Recipes array extent overflow");
    return left * right;
}

static size_t nr_first(int low)
{
    return low > NR_END ? (size_t)low : NR_END;
}

/* The index-zero view and its first logical element must both be inside
   their allocation, even for a short range far from zero. */
static size_t nr_zero(int low)
{
    if (low < 0)
        return nr_first(low) + (size_t)(-(int64_t)low);
    return nr_first(low) - (size_t)low;
}

static size_t nr_bytes(size_t count, int low, size_t item_size)
{
    size_t padding = low < 0 ? nr_zero(low) : nr_first(low);
    size_t limit = (size_t)PTRDIFF_MAX / item_size;
    if (count == 0)
        count = 1;
    if (padding > limit || count > limit - padding)
        err("Numerical Recipes array storage overflow");
    return (padding + count) * item_size;
}

static void *nr_allocate(size_t bytes)
{
    void *storage = malloc(bytes);
    if (!storage)
        err("allocation failure in Numerical Recipes array");
    return storage;
}


float **matrix(int nrl, int nrh, int ncl, int nch)
{
    size_t rows = nr_extent(nrl, nrh);
    size_t cols = nr_extent(ncl, nch);
    size_t total = nr_product(rows, cols);
    size_t row_bytes = nr_bytes(rows, nrl, sizeof(float *));
    size_t data_bytes = nr_bytes(total, ncl, sizeof(float));
    float **row = nr_allocate(row_bytes);
    float *data = nr_allocate(data_bytes);
    size_t row_first = nr_first(nrl);
    size_t data_first = nr_first(ncl);
    size_t row_zero = nr_zero(nrl);
    size_t data_zero = nr_zero(ncl);
    size_t i;

    row[row_first] = data + data_zero;
    for (i = 1; i < rows; ++i)
        row[row_first + i] = data + data_zero + i * cols;
    data[data_first] = 0;
    for (i = 1; i < total; ++i)
        data[data_first + i] = 0;
    return row + row_zero;
}

float **fmatrix(int nrl, int nrh, int ncl, int nch)
{
    return matrix(nrl, nrh, ncl, nch);
}

double **dmatrix(int nrl, int nrh, int ncl, int nch)
{
    size_t rows = nr_extent(nrl, nrh);
    size_t cols = nr_extent(ncl, nch);
    size_t total = nr_product(rows, cols);
    size_t row_bytes = nr_bytes(rows, nrl, sizeof(double *));
    size_t data_bytes = nr_bytes(total, ncl, sizeof(double));
    double **row = nr_allocate(row_bytes);
    double *data = nr_allocate(data_bytes);
    size_t row_first = nr_first(nrl);
    size_t data_first = nr_first(ncl);
    size_t row_zero = nr_zero(nrl);
    size_t data_zero = nr_zero(ncl);
    size_t i;

    row[row_first] = data + data_zero;
    for (i = 1; i < rows; ++i)
        row[row_first + i] = data + data_zero + i * cols;
    data[data_first] = 0;
    for (i = 1; i < total; ++i)
        data[data_first + i] = 0;
    return row + row_zero;
}

int **imatrix(int nrl, int nrh, int ncl, int nch)
{
    size_t rows = nr_extent(nrl, nrh);
    size_t cols = nr_extent(ncl, nch);
    size_t total = nr_product(rows, cols);
    size_t row_bytes = nr_bytes(rows, nrl, sizeof(int *));
    size_t data_bytes = nr_bytes(total, ncl, sizeof(int));
    int **row = nr_allocate(row_bytes);
    int *data = nr_allocate(data_bytes);
    size_t row_first = nr_first(nrl);
    size_t data_first = nr_first(ncl);
    size_t row_zero = nr_zero(nrl);
    size_t data_zero = nr_zero(ncl);
    size_t i;

    row[row_first] = data + data_zero;
    for (i = 1; i < rows; ++i)
        row[row_first + i] = data + data_zero + i * cols;
    data[data_first] = 0;
    for (i = 1; i < total; ++i)
        data[data_first + i] = 0;
    return row + row_zero;
}

unsigned short int **usmatrix(int nrl, int nrh, int ncl, int nch)
{
    size_t rows = nr_extent(nrl, nrh);
    size_t cols = nr_extent(ncl, nch);
    size_t total = nr_product(rows, cols);
    size_t row_bytes = nr_bytes(rows, nrl, sizeof(unsigned short int *));
    size_t data_bytes = nr_bytes(total, ncl, sizeof(unsigned short int));
    unsigned short int **row = nr_allocate(row_bytes);
    unsigned short int *data = nr_allocate(data_bytes);
    size_t row_first = nr_first(nrl);
    size_t data_first = nr_first(ncl);
    size_t row_zero = nr_zero(nrl);
    size_t data_zero = nr_zero(ncl);
    size_t i;

    row[row_first] = data + data_zero;
    for (i = 1; i < rows; ++i)
        row[row_first + i] = data + data_zero + i * cols;
    data[data_first] = 0;
    for (i = 1; i < total; ++i)
        data[data_first + i] = 0;
    return row + row_zero;
}

float ***f3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh)
{
    size_t rows = nr_extent(nrl, nrh);
    size_t cols = nr_extent(ncl, nch);
    size_t depth = nr_extent(ndl, ndh);
    size_t cells = nr_product(rows, cols);
    size_t total = nr_product(cells, depth);
    size_t row_bytes = nr_bytes(rows, nrl, sizeof(float **));
    size_t column_bytes = nr_bytes(cells, ncl, sizeof(float *));
    size_t data_bytes = nr_bytes(total, ndl, sizeof(float));
    float ***row = nr_allocate(row_bytes);
    float **column = nr_allocate(column_bytes);
    float *data = nr_allocate(data_bytes);
    size_t row_first = nr_first(nrl);
    size_t column_first = nr_first(ncl);
    size_t data_first = nr_first(ndl);
    size_t row_zero = nr_zero(nrl);
    size_t column_zero = nr_zero(ncl);
    size_t data_zero = nr_zero(ndl);
    size_t i;

    row[row_first] = column + column_zero;
    for (i = 1; i < rows; ++i)
        row[row_first + i] = column + column_zero + i * cols;
    column[column_first] = data + data_zero;
    for (i = 1; i < cells; ++i)
        column[column_first + i] = data + data_zero + i * depth;
    data[data_first] = 0;
    for (i = 1; i < total; ++i)
        data[data_first + i] = 0;
    return row + row_zero;
}

int ***i3tensor(int nrl, int nrh, int ncl, int nch, int ndl, int ndh)
{
    size_t rows = nr_extent(nrl, nrh);
    size_t cols = nr_extent(ncl, nch);
    size_t depth = nr_extent(ndl, ndh);
    size_t cells = nr_product(rows, cols);
    size_t total = nr_product(cells, depth);
    size_t row_bytes = nr_bytes(rows, nrl, sizeof(int **));
    size_t column_bytes = nr_bytes(cells, ncl, sizeof(int *));
    size_t data_bytes = nr_bytes(total, ndl, sizeof(int));
    int ***row = nr_allocate(row_bytes);
    int **column = nr_allocate(column_bytes);
    int *data = nr_allocate(data_bytes);
    size_t row_first = nr_first(nrl);
    size_t column_first = nr_first(ncl);
    size_t data_first = nr_first(ndl);
    size_t row_zero = nr_zero(nrl);
    size_t column_zero = nr_zero(ncl);
    size_t data_zero = nr_zero(ndl);
    size_t i;

    row[row_first] = column + column_zero;
    for (i = 1; i < rows; ++i)
        row[row_first + i] = column + column_zero + i * cols;
    column[column_first] = data + data_zero;
    for (i = 1; i < cells; ++i)
        column[column_first + i] = data + data_zero + i * depth;
    data[data_first] = 0;
    for (i = 1; i < total; ++i)
        data[data_first + i] = 0;
    return row + row_zero;
}

float ****f4tensor(int nrl, int nrh, int ncl, int nch,
                  int ndl, int ndh, int nvl, int nvh)
{
    size_t rows = nr_extent(nrl, nrh);
    size_t cols = nr_extent(ncl, nch);
    size_t depth = nr_extent(ndl, ndh);
    size_t values = nr_extent(nvl, nvh);
    size_t cells = nr_product(rows, cols);
    size_t layers = nr_product(cells, depth);
    size_t total = nr_product(layers, values);
    size_t row_bytes = nr_bytes(rows, nrl, sizeof(float ***));
    size_t column_bytes = nr_bytes(cells, ncl, sizeof(float **));
    size_t layer_bytes = nr_bytes(layers, ndl, sizeof(float *));
    size_t data_bytes = nr_bytes(total, nvl, sizeof(float));
    float ****row = nr_allocate(row_bytes);
    float ***column = nr_allocate(column_bytes);
    float **layer = nr_allocate(layer_bytes);
    float *data = nr_allocate(data_bytes);
    size_t row_first = nr_first(nrl);
    size_t column_first = nr_first(ncl);
    size_t layer_first = nr_first(ndl);
    size_t data_first = nr_first(nvl);
    size_t row_zero = nr_zero(nrl);
    size_t column_zero = nr_zero(ncl);
    size_t layer_zero = nr_zero(ndl);
    size_t data_zero = nr_zero(nvl);
    size_t i;

    row[row_first] = column + column_zero;
    for (i = 1; i < rows; ++i)
        row[row_first + i] = column + column_zero + i * cols;
    column[column_first] = layer + layer_zero;
    for (i = 1; i < cells; ++i)
        column[column_first + i] = layer + layer_zero + i * depth;
    layer[layer_first] = data + data_zero;
    for (i = 1; i < layers; ++i)
        layer[layer_first + i] = data + data_zero + i * values;
    data[data_first] = 0;
    for (i = 1; i < total; ++i)
        data[data_first + i] = 0;
    return row + row_zero;
}

void free_vector(float *v, int nl, int nh){
	/* free a float vector allocated with vector() */
	free((FREE_ARG) (v+nl-NR_END));
}

void free_ivector(int *v, int nl, int nh){
	/* free a int vector allocated with vector() */
	free((FREE_ARG) (v+nl-NR_END));
}

void free_cvector(char *v, int nl, int nh){
	/* free a char vector allocated with vector() */
	free((FREE_ARG) (v+nl-NR_END));
}

void free_dvector(double *v, int nl, int nh){
        /* free a char vector allocated with vector() */
	free((FREE_ARG) (v+nl-NR_END));
        }
                
void free_matrix(float **view, int nrl, int nrh, int ncl, int nch)
{
    (void)nrh; (void)nch;
    if (!view)
        return;
    free(view[nrl] + ncl - nr_first(ncl));
    free(view + nrl - nr_first(nrl));
}

void free_dmatrix(double **view, int nrl, int nrh, int ncl, int nch)
{
    (void)nrh; (void)nch;
    if (!view)
        return;
    free(view[nrl] + ncl - nr_first(ncl));
    free(view + nrl - nr_first(nrl));
}

void free_imatrix(int **view, int nrl, int nrh, int ncl, int nch)
{
    (void)nrh; (void)nch;
    if (!view)
        return;
    free(view[nrl] + ncl - nr_first(ncl));
    free(view + nrl - nr_first(nrl));
}

void free_usmatrix(unsigned short int **view, int nrl, int nrh, int ncl, int nch)
{
    (void)nrh; (void)nch;
    if (!view)
        return;
    free(view[nrl] + ncl - nr_first(ncl));
    free(view + nrl - nr_first(nrl));
}

void free_f3tensor(float ***view, int nrl, int nrh, int ncl, int nch, int ndl, int ndh)
{
    (void)nrh; (void)nch; (void)ndh;
    if (!view)
        return;
    free(view[nrl][ncl] + ndl - nr_first(ndl));
    free(view[nrl] + ncl - nr_first(ncl));
    free(view + nrl - nr_first(nrl));
}

void free_i3tensor(int ***view, int nrl, int nrh, int ncl, int nch, int ndl, int ndh)
{
    (void)nrh; (void)nch; (void)ndh;
    if (!view)
        return;
    free(view[nrl][ncl] + ndl - nr_first(ndl));
    free(view[nrl] + ncl - nr_first(ncl));
    free(view + nrl - nr_first(nrl));
}

void free_f4tensor(float ****view, int nrl, int nrh, int ncl, int nch,
                   int ndl, int ndh, int nvl, int nvh)
{
    (void)nrh; (void)nch; (void)ndh; (void)nvh;
    if (!view)
        return;
    free(view[nrl][ncl][ndl] + nvl - nr_first(nvl));
    free(view[nrl][ncl] + ndl - nr_first(ndl));
    free(view[nrl] + ncl - nr_first(ncl));
    free(view + nrl - nr_first(nrl));
}

void zero(float *A, int u_max){
	int u=0;
	while(++u<=u_max) *(A++)=0.0;
}

void normalize_data(float **data, int ntr, int ns){

	float *max_min_trace=NULL;
	float max, min, tmp;
	int i,j;


	max_min_trace = vector(1,ntr); 
	for(i=1;i<=ntr;i++){

      	    max=0.0;
      	    min=0.0;

     	    for(j=2;j<=ns;j++){

      	       /* Estimate max and min */
               if(data[i][j]>max){max = data[i][j];}
               if(data[i][j]<min){min = data[i][j];}

	       if (max>(fabs(min))){tmp=max;}
	       else{tmp=fabs(min);}

            }
 
	    max_min_trace[i]=tmp;	

	    /* set maximum_values=0.0 to 1.0 */
	    if(max_min_trace[i]==0.0){max_min_trace[i]=1.0;}
   

	    for(j=2;j<=ns;j++){
		data[i][j] = data[i][j]/max_min_trace[i];
	    }
	
	}

        free_vector(max_min_trace,1,ntr);
}

void normalize_data_rel(float **data, float **data_obs, int ntr, int ns){

	float *max_min_trace=NULL, *max_min_obs_trace=NULL;
	float max, min, max_obs, min_obs, tmp;
	int i,j;

	max_min_trace = vector(1,ntr); 
	max_min_obs_trace = vector(1,ntr); 

	for(i=1;i<=ntr;i++){

      	    max=0.0;
      	    min=0.0;

      	    max_obs=0.0;
      	    min_obs=0.0;
        
     	    for(j=2;j<=ns;j++){

      	       /* Estimate max and min of model and field data */
               if(data[i][j]>max){max = data[i][j];}
               if(data[i][j]<min){min = data[i][j];}

               if(data_obs[i][j]>max_obs){max_obs = data_obs[i][j];}
               if(data_obs[i][j]<min_obs){min_obs = data_obs[i][j];}

            }

	    if (max>(fabs(min))){tmp=max;}
	    else{tmp=fabs(min);}

            max_min_trace[i]=tmp;

	    if (max_obs>(fabs(min_obs))){tmp=max_obs;}
	    else{tmp=fabs(min_obs);}

	    max_min_obs_trace[i]=tmp;	

	    /* set maximum_values=0.0 to 1.0 */
	    if(max_min_trace[i]==0.0){max_min_trace[i]=1.0;}
            if(max_min_obs_trace[i]==0.0){max_min_obs_trace[i]=1.0;}

	    for(j=2;j<=ns;j++){
		data_obs[i][j] = data_obs[i][j]*(max_min_trace[i]/max_min_obs_trace[i]);
	    }
	
	}

        free_vector(max_min_trace,1,ntr);
        free_vector(max_min_obs_trace,1,ntr);

}

void normalize_data_rms(float **data, float **data_obs, int ntr, int ns){

	float rms, rms_obs;
	int i,j;
 
        rms=0.0;
      	rms_obs=0.0;

	for(i=1;i<=ntr;i++){        
     	    for(j=2;j<=ns;j++){

      	       /* Estimate rms-values of model and field data */
               rms += fabs(data[i][j]);
               rms_obs += fabs(data_obs[i][j]);
               
            }

	}

        rms=rms/(ntr*(ns-1));
        rms_obs=rms_obs/(ntr*(ns-1));

        /* set rms-values=0.0 to 1.0 */
	if(rms==0.0){rms=1.0;}
        if(rms_obs==0.0){rms_obs=1.0;}

	for(i=1;i<=ntr;i++){        

	    for(j=2;j<=ns;j++){
		data_obs[i][j] = data_obs[i][j]*(rms/rms_obs);
	    }

	}

}

void normalize_STF(float *data, int ns){

	float max;
	int j;

        max=fabs(data[1]);

     	for(j=2;j<=ns;j++){

      	   /* Estimate maximum */
           if(fabs(data[j])>max){max = fabs(data[j]);}
           

	}
   
	for(j=1;j<=ns;j++){
	   data[j] = data[j]/max;
	}
	
}

void tripd(float *d, float *e, float *b, int n)
/*****************************************************************************
Given an n-by-n symmetric, tridiagonal, positive definite matrix A and
 n-vector b, following algorithm overwrites b with the solution to Ax = b

*****************************************************************************
Input:
d[]     the diagonal of A
e[]     the superdiagonal of A
b[]     the rhs of Ax=b

Output:
b[]     b[] is overwritten with the solution to Ax=b

*****************************************************************************
Notes:

Given an n-by-n symmetric, tridiagonal, positive definite matrix A and
 n-vector b, following algorithm overwrites b with the solution to Ax = b

*****************************************************************************
Author: Zhenyue Liu, Colorado School of Mines, 1993.
*****************************************************************************/
{
        int k;
        float temp;
       
        /* decomposition */
        for(k=1; k<n; ++k){
           temp = e[k-1];
           e[k-1] = temp/d[k-1];
           d[k] -= temp*e[k-1];
        }

        /* substitution */
        for(k=1; k<n; ++k)  b[k] -= e[k-1]*b[k-1];
       
        b[n-1] /=d[n-1];
        for(k=n-1; k>0; --k)  b[k-1] = b[k-1]/d[k-1] - e[k-1]*b[k];
       
}

