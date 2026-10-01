/* Test-only unit-basis exposure of the actual staged MPI/surface copy graph. */
#include "../../src/PSV/elastic_psv_born_mpi.c"

static void io(FILE *f,void *p,size_t n,size_t w,int writing) {
    if(!f || (writing?fwrite(p,w,n,f):fread(p,w,n,f))!=n)
        MPI_Abort(MPI_COMM_WORLD,5);
}
int main(int argc,char **argv) {
    struct denise_elastic_psv_born_mpi c;
    int rank,size,kind,i,m,k,n,px,py;
    float *all=NULL,*x;
    double *bars=NULL,*y,*result=NULL;
    FILE *input=NULL,*output=NULL;
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc!=6)MPI_Abort(MPI_COMM_WORLD,2);
    px=atoi(argv[3]);py=atoi(argv[4]);kind=atoi(argv[5]);
    if(size!=px*py || (kind!=SYY && kind!=SXY))MPI_Abort(MPI_COMM_WORLD,3);
    memset(&c,0,sizeof(c));c.comm=MPI_COMM_WORLD;c.nx=8/px;c.ny=8/py;
    c.oy=(rank/px)*c.ny;c.ox=(rank%px)*c.nx;c.free_surface=1;
    c.cells=(size_t)(c.nx+4)*(c.ny+4);c.owned_cells=(size_t)c.nx*c.ny;
    c.left=rank/px*px+(rank%px+px-1)%px;c.right=rank/px*px+(rank%px+1)%px;
    c.top=((rank/px+py-1)%py)*px+rank%px;c.bottom=((rank/px+1)%py)*px+rank%px;
    c.halo_capacity=2u*(size_t)(c.nx+4)>2u*(size_t)c.ny?2u*(size_t)(c.nx+4):2u*(size_t)c.ny;
    c.send_buffer=calloc(c.halo_capacity,sizeof(double));c.receive_buffer=calloc(c.halo_capacity,sizeof(double));
    x=calloc(c.cells,sizeof(float));y=calloc(c.cells,sizeof(double));
    if(!c.send_buffer||!c.receive_buffer||!x||!y)MPI_Abort(MPI_COMM_WORLD,4);
    if(rank==0){
        input=fopen(argv[1],"rb");output=fopen(argv[2],"wb");io(input,&n,1,sizeof(n),0);
        all=calloc(c.cells*(size_t)size,sizeof(float));bars=calloc(c.cells*(size_t)size,sizeof(double));
        result=calloc(c.cells*(size_t)size,sizeof(double));
        if(!all||!bars||!result)MPI_Abort(MPI_COMM_WORLD,4);
    }
    MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
    for(k=0;k<n;k++) {
        if(rank==0){io(input,all,c.cells*(size_t)size,sizeof(float),0);io(input,bars,c.cells*(size_t)size,sizeof(double),0);}
        MPI_Scatter(all,(int)c.cells,MPI_FLOAT,x,(int)c.cells,MPI_FLOAT,0,MPI_COMM_WORLD);
        MPI_Scatter(bars,(int)c.cells,MPI_DOUBLE,y,(int)c.cells,MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(kind==SYY)surface_project(&c,x);
        halo_float(&c,x,0);
        if(c.oy==0) {
            for(i=-2;i<c.nx+2;i++) {
                if(kind==SYY)x[cell(&c,0,i)]=0.0f;
                for(m=1;m<=2;m++)x[cell(&c,-m,i)]=surface_value(&c,x,-m,i,kind,NULL,NULL,NULL,NULL,NULL,NULL);
            }
        }
        MPI_Gather(x,(int)c.cells,MPI_FLOAT,all,(int)c.cells,MPI_FLOAT,0,MPI_COMM_WORLD);
        if(rank==0)io(output,all,c.cells*(size_t)size,sizeof(float),1);
        if(c.oy==0) {
            for(i=-2;i<c.nx+2;i++) {
                if(kind==SYY)y[cell(&c,0,i)]=0.0;
                for(m=1;m<=2;m++) {
                    size_t p=cell(&c,-m,i);double v=y[p];y[p]=0.0;
                    surface_value_transpose(&c,y,-m,i,kind,v,NULL,NULL,NULL,NULL,NULL);
                }
            }
        }
        halo_double(&c,y,1);
        if(kind==SYY && c.oy==0)for(i=0;i<c.nx;i++)y[cell(&c,0,i)]=0.0;
        MPI_Gather(y,(int)c.cells,MPI_DOUBLE,result,(int)c.cells,MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(rank==0)io(output,result,c.cells*(size_t)size,sizeof(double),1);
    }
    if(rank==0){fclose(input);fclose(output);free(all);free(bars);free(result);}
    free(x);free(y);free(c.send_buffer);free(c.receive_buffer);
    MPI_Finalize();return 0;
}
