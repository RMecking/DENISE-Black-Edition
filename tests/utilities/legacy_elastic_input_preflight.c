/* Links the actual reader; no material or wavefield calculation is substituted. */
#include "fd.h"
#include <stdint.h>
int NX, NY, NXG=8, NYG=6, POS[3], MYID, INVMAT1, WRITEMOD=1;
char MFILE[STRING_SIZE];
FILE *FP;
static float data[3][8][10], *rows[3][8];
static int opened, writes, merges, fail_alloc_rank=-1;
void *__real_malloc(size_t);
FILE *__real_fopen(const char *,const char *);
int __real_fclose(FILE *);
void *__wrap_malloc(size_t n) {
    return MYID==fail_alloc_rank ? NULL : __real_malloc(n);
}
FILE *__wrap_fopen(const char *name,const char *mode) {
    FILE *f=__real_fopen(name,mode); if(f)++opened; return f;
}
int __wrap_fclose(FILE *f) { --opened; return __real_fclose(f); }
static int unchanged(void) {
    int a,j,i;
    for(a=0;a<3;++a)for(j=0;j<8;++j)for(i=0;i<10;++i)
        if(data[a][j][i]!=-777.0f)return 0;
    return 1;
}
void err(char *message) {
    int okay=unchanged() && !writes && !merges && !opened, all;
    unsigned long hash=5381, min, max;
    const unsigned char *p=(const unsigned char *)message;
    while(*p)hash=hash*33+*p++;
    MPI_Allreduce(&okay,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(&hash,&min,1,MPI_UNSIGNED_LONG,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(&hash,&max,1,MPI_UNSIGNED_LONG,MPI_MAX,MPI_COMM_WORLD);
    fprintf(stderr,"rank=%d %s\n",MYID,message);
    printf("{\"rank\":%d,\"rejected\":true,\"unchanged_closed\":%d,\"same_error\":%d}\n",
           MYID,all,min==max);fflush(stdout);
    fclose(FP);
    MPI_Finalize();
    exit(23);
}
void writemod(char name[STRING_SIZE],float **values,int format) {
    const char *suffix[] = {".denise.pi",".denise.mu",".denise.rho"};
    float **expected[] = {rows[0],rows[1],rows[2]};
    char path[STRING_SIZE2];
    snprintf(path,sizeof(path),"%s%s",MFILE,suffix[writes%3]);
    if(strcmp(name,path) || values!=expected[writes%3] || format!=3)
        MPI_Abort(MPI_COMM_WORLD,24);
    ++writes;
}
void mergemod(char name[STRING_SIZE],int format) {
    (void)name;if(MYID || format!=3)MPI_Abort(MPI_COMM_WORLD,25);++merges;
}
int main(int argc,char **argv) {
    int size,px,py,a,j,i,halo=1;
    char output[1024];
    FILE *f;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&MYID);MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(argc!=9)MPI_Abort(MPI_COMM_WORLD,26);
    INVMAT1=atoi(argv[1]);px=atoi(argv[2]);py=atoi(argv[3]);
    if(size!=px*py || NXG%px || NYG%py)MPI_Abort(MPI_COMM_WORLD,27);
    NX=NXG/px;NY=NYG/py;POS[1]=MYID%px;POS[2]=MYID/px;
    if(strlen(argv[4])>=sizeof(MFILE))MPI_Abort(MPI_COMM_WORLD,28);
    strcpy(MFILE,argv[4]);
    if(MYID==atoi(argv[6])) {
        if(strlen(argv[7])>=sizeof(MFILE))MPI_Abort(MPI_COMM_WORLD,29);
        strcpy(MFILE,argv[7]);
    }
    fail_alloc_rank=atoi(argv[8]);
    FP=tmpfile();if(!FP)MPI_Abort(MPI_COMM_WORLD,30);
    for(a=0;a<3;++a)for(j=0;j<8;++j) {
        rows[a][j]=data[a][j];for(i=0;i<10;++i)data[a][j][i]=-777.0f;
    }
    readmod_elastic_PSV(rows[2],rows[0],rows[1]);
    for(a=0;a<3;++a)for(j=0;j<8;++j)for(i=0;i<10;++i)
        if((j<1 || j>NY || i<1 || i>NX) && data[a][j][i]!=-777.0f)halo=0;
    snprintf(output,sizeof(output),"%s/%d.bin",argv[5],MYID);
    f=__real_fopen(output,"wb");if(!f)MPI_Abort(MPI_COMM_WORLD,31);
    for(a=0;a<3;++a)for(i=1;i<=NX;++i)for(j=1;j<=NY;++j)
        if(fwrite(&data[a][j][i],sizeof(float),1,f)!=1)MPI_Abort(MPI_COMM_WORLD,32);
    if(__real_fclose(f))MPI_Abort(MPI_COMM_WORLD,33);
    printf("{\"rank\":%d,\"rejected\":false,\"halo\":%d,\"closed\":%d,\"writes\":%d,\"merges\":%d}\n",
           MYID,halo,opened==0,writes,merges);fflush(stdout);
    fclose(FP);MPI_Finalize();return 0;
}
