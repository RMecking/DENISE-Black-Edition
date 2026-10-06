/* Focused adapter: executes both the actual candidate and authentic BASE. */
#include "fd.h"
#include <stdarg.h>
#include <stdint.h>
#include <inttypes.h>

int NX, NY, MYID, INVMAT1, FW=0;
float DH=0.05f, DT=0.001f, TS=8.0f;
void checkfd_ssg_elastic_BASE(FILE *,float **,float **,float **,float *);
struct quantities { float cmin,cmax,dh,dt,wavelength; unsigned seen; int grid_warnings; };
static struct quantities measurements[2];
static int active;
static uint32_t bits(float x) { uint32_t b;memcpy(&b,&x,sizeof(b));return b; }

/* Observe promoted numerical arguments before decimal formatting. This changes
   neither the production formulas nor their arithmetic/printing order. */
int __wrap_fprintf(FILE *stream,const char *format,...) {
    va_list args,probe;
    struct quantities *q=&measurements[active];
    int result;
    va_start(args,format);va_copy(probe,args);
    if(strstr(format,"Vp_max= %e")) {
        q->cmax=(float)va_arg(probe,double);q->cmin=(float)va_arg(probe,double);q->seen|=3;
    } else if(strstr(format,"Maximum phase velocity (CFL)=")) {
        q->cmax=(float)va_arg(probe,double);q->seen|=1;
    } else if(strstr(format,"Minimum positive propagating phase velocity (dispersion)=")) {
        q->cmin=(float)va_arg(probe,double);q->seen|=2;
    } else if(strstr(format,"recommended value for DH")) {
        q->dh=(float)va_arg(probe,double);q->seen|=4;
    } else if(strstr(format,"stability limit for timestep DT")) {
        q->dt=(float)va_arg(probe,double);q->seen|=8;
    } else if(!strcmp(format," be %e meter.\n")) {
        q->wavelength=(float)va_arg(probe,double);q->seen|=16;
    }
    va_end(probe);result=vfprintf(stream,format,args);va_end(args);return result;
}
void warning(char *message) {
    if(strstr(message,"Grid dispersion"))++measurements[active].grid_warnings;
}
void err(char *message) {
    fprintf(stderr,"%s\n",message);MPI_Abort(MPI_COMM_WORLD,20);exit(20);
}
static void emit(const char *label,struct quantities *q) {
    printf("\"%s\":{\"seen\":%u,\"cmin_bits\":%" PRIu32 ",\"cmax_bits\":%" PRIu32
           ",\"dh_bits\":%" PRIu32 ",\"dt_bits\":%" PRIu32 ",\"wavelength_bits\":%" PRIu32
           ",\"grid_warnings\":%d}",label,q->seen,bits(q->cmin),bits(q->cmax),
           bits(q->dh),bits(q->dt),bits(q->wavelength),q->grid_warnings);
}
int main(int argc,char **argv) {
    int px,py,size,x,y,a,i,j;
    FILE *input,*log;
    char path[1024];
    float global[3][8][8],storage[3][10][10],*rows[3][10];
    float hc[7]={6.0f,9.0f/8.0f,-1.0f/24.0f,0,0,0,0};
    uint32_t before[3][8][8],after[3][8][8];
    MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&MYID);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc!=7)MPI_Abort(MPI_COMM_WORLD,21);
    INVMAT1=atoi(argv[1]);px=atoi(argv[2]);py=atoi(argv[3]);
    if(size!=px*py || px<1 || py<1 || 8%px || 8%py)MPI_Abort(MPI_COMM_WORLD,22);
    NX=8/px;NY=8/py;x=MYID%px;y=MYID/px;
    if(atoi(argv[6])==1) { DH=ldexpf(1.0f,-144);DT=ldexpf(1.0f,-149); }
    if(atoi(argv[6])==2) DT=ldexpf(1.0f,-50);
    input=fopen(argv[4],"rb");
    if(!input || fread(global,sizeof(float),192,input)!=192)MPI_Abort(MPI_COMM_WORLD,23);
    if(fclose(input))MPI_Abort(MPI_COMM_WORLD,24);
    for(a=0;a<3;++a)for(j=0;j<10;++j)rows[a][j]=storage[a][j];
    for(a=0;a<3;++a)for(i=0;i<NX;++i)for(j=0;j<NY;++j)
        storage[a][j+1][i+1]=global[a][x*NX+i][y*NY+j];
    memcpy(before,global,sizeof(before));
    for(active=0;active<2;++active) {
        snprintf(path,sizeof(path),"%s/%s-%d.txt",argv[5],active ? "candidate" : "base",MYID);
        log=fopen(path,"wb");if(!log)MPI_Abort(MPI_COMM_WORLD,25);
        if(active)checkfd_ssg_elastic(log,rows[2],rows[0],rows[1],hc);
        else checkfd_ssg_elastic_BASE(log,rows[2],rows[0],rows[1],hc);
        if(fclose(log))MPI_Abort(MPI_COMM_WORLD,26);
    }
    for(a=0;a<3;++a)for(i=0;i<NX;++i)for(j=0;j<NY;++j)
        global[a][x*NX+i][y*NY+j]=storage[a][j+1][i+1];
    memcpy(after,global,sizeof(after));
    i=memcmp(before,after,sizeof(before))==0;
    MPI_Allreduce(MPI_IN_PLACE,&i,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    if(!MYID) {
        printf("{\"ranks\":%d,\"material_unchanged\":%d,",size,i);
        emit("base",&measurements[0]);printf(",");emit("candidate",&measurements[1]);
        printf("}\n");
    }
    MPI_Finalize();return 0;
}
