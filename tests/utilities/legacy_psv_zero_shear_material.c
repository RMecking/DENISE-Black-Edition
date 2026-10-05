/* Focused helper/material test adapter; links the actual production routines. */
#include "fd.h"
#include <fenv.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int NX=1, NY=1, INVMAT1=3, MYID=0, INDEX[5];
const int TAG1=11, TAG2=12, TAG5=15, TAG6=16;
FILE *FP;
MPI_Comm SHOT_COMM;
void av_mue_base(float **u, float **corner, float **rho);

static uint32_t bits(float x) {
    uint32_t b;
    memcpy(&b,&x,sizeof(b));
    return b;
}
static float converted(float rho,float u) {
    return INVMAT1==1 ? rho*u*u : u;
}
static void local_case(const char *kind,int index) {
    float **u=matrix(0,2,0,2), **rho=matrix(0,2,0,2);
    float **corner=matrix(0,2,0,2), **base=matrix(0,2,0,2);
    float values[4]={0.5f,2.0f,3.0f,7.0f};
    float densities[4]={1.0f,2.0f,1.5f,3.0f};
    uint32_t mu_bits[4];
    int flags,base_flags,k;
    if(!strcmp(kind,"occupancy"))
        for(k=0;k<4;k++)if(!(index&(1<<k)))values[k]=0.0f;
    if(!strcmp(kind,"solid") && index>0) {
        const int exponents[3][4]={{-20,13,-7,8},{-120,-118,-119,-117},{-140,-142,-141,-139}};
        for(k=0;k<4;k++) {
            int e=exponents[index-1][k];
            values[k]=ldexpf(1.0f,INVMAT1==1 ? e/2 : e);
        }
    }
    if(!strcmp(kind,"underflow"))values[0]=ldexpf(1.0f,-100);
    if(!strcmp(kind,"signed_zero"))values[0]=-0.0f;
    for(k=0;k<4;k++) {
        int j=1+k/2,i=1+k%2;
        u[j][i]=values[k];rho[j][i]=densities[k];
        mu_bits[k]=bits(converted(densities[k],values[k]));
    }
    feclearexcept(FE_ALL_EXCEPT);
    av_mue(u,corner,rho);
    flags=fetestexcept(FE_DIVBYZERO|FE_INVALID);
    feclearexcept(FE_ALL_EXCEPT);
    av_mue_base(u,base,rho);
    base_flags=fetestexcept(FE_DIVBYZERO|FE_INVALID);
    printf("{\"mode\":%d,\"bits\":%" PRIu32 ",\"base_bits\":%" PRIu32
           ",\"divide\":%d,\"invalid\":%d,\"base_divide\":%d,\"base_invalid\":%d"
           ",\"mu_bits\":[%" PRIu32 ",%" PRIu32 ",%" PRIu32 ",%" PRIu32 "]}\n",
           INVMAT1,bits(corner[1][1]),bits(base[1][1]),
           !!(flags&FE_DIVBYZERO),!!(flags&FE_INVALID),
           !!(base_flags&FE_DIVBYZERO),!!(base_flags&FE_INVALID),
           mu_bits[0],mu_bits[1],mu_bits[2],mu_bits[3]);
    free_matrix(u,0,2,0,2);free_matrix(rho,0,2,0,2);
    free_matrix(corner,0,2,0,2);free_matrix(base,0,2,0,2);
}
static int fluid(int x,int y,const char *pattern) {
    if(!strcmp(pattern,"horizontal"))return y<4;
    if(!strcmp(pattern,"vertical"))return x<4;
    return y<4+(x>=4);
}
static float density(int x,int y) { return 1.0f+0.125f*((x+2*y)%5); }
static float primary(int x,int y,const char *pattern) {
    return fluid(x,y,pattern) ? 0.0f : 1.0f+0.125f*x+0.0625f*y;
}
static float pi_value(int x,int y) { return 4.0f+0.25f*x+0.125f*y; }
static void halo_case(int px,int py,const char *pattern,const char *path) {
    int size,x,y,i,j,k,bad=0,any,flags,division,invalid;
    float **rho,**pi,**u,**corner,*packed,*all=NULL;
    void *bsend;
    int capacity,detached;
    MPI_Comm_rank(MPI_COMM_WORLD,&MYID);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if(size!=px*py || px<1 || py<1 || px>2 || py>2)MPI_Abort(MPI_COMM_WORLD,2);
    SHOT_COMM=MPI_COMM_WORLD;
    NX=8/px;NY=8/py;x=MYID%px;y=MYID/px;
    INDEX[1]=y*px+(x+px-1)%px;INDEX[2]=y*px+(x+1)%px;
    INDEX[3]=((y+py-1)%py)*px+x;INDEX[4]=((y+1)%py)*px+x;
    FP=tmpfile();if(!FP)MPI_Abort(MPI_COMM_WORLD,3);
    rho=matrix(0,NY+1,0,NX+1);pi=matrix(0,NY+1,0,NX+1);
    u=matrix(0,NY+1,0,NX+1);corner=matrix(0,NY+1,0,NX+1);
    for(j=0;j<=NY+1;j++)for(i=0;i<=NX+1;i++)rho[j][i]=pi[j][i]=u[j][i]=-777.0f;
    for(j=1;j<=NY;j++)for(i=1;i<=NX;i++) {
        int gx=x*NX+i-1,gy=y*NY+j-1;
        rho[j][i]=density(gx,gy);pi[j][i]=pi_value(gx,gy);u[j][i]=primary(gx,gy,pattern);
    }
    capacity=4*((8+2)*3*(int)sizeof(float)+MPI_BSEND_OVERHEAD);
    bsend=malloc((size_t)capacity);if(!bsend)MPI_Abort(MPI_COMM_WORLD,4);
    MPI_Buffer_attach(bsend,capacity);
    matcopy_elastic_PSV(rho,pi,u);
    for(j=0;j<=NY+1;j++)for(i=0;i<=NX+1;i++) {
        int gx=(x*NX+i-1+8)%8,gy=(y*NY+j-1+8)%8;
        if(bits(rho[j][i])!=bits(density(gx,gy))
            || bits(pi[j][i])!=bits(pi_value(gx,gy))
            || bits(u[j][i])!=bits(primary(gx,gy,pattern)))bad=1;
    }
    feclearexcept(FE_ALL_EXCEPT);av_mue(u,corner,rho);
    flags=fetestexcept(FE_DIVBYZERO|FE_INVALID);
    division=!!(flags&FE_DIVBYZERO);invalid=!!(flags&FE_INVALID);
    packed=malloc((size_t)NX*NY*sizeof(float));
    if(!packed)MPI_Abort(MPI_COMM_WORLD,5);
    for(j=1;j<=NY;j++)for(i=1;i<=NX;i++)packed[(j-1)*NX+i-1]=corner[j][i];
    if(!MYID) {all=malloc(64*sizeof(float));if(!all)MPI_Abort(MPI_COMM_WORLD,6);}
    MPI_Gather(packed,NX*NY,MPI_FLOAT,all,NX*NY,MPI_FLOAT,0,MPI_COMM_WORLD);
    MPI_Allreduce(&bad,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&division,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&invalid,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(!MYID) {
        float global[64];
        FILE *out;
        for(k=0;k<size;k++)for(j=0;j<NY;j++)for(i=0;i<NX;i++)
            global[(k/px*NY+j)*8+k%px*NX+i]=all[k*NX*NY+j*NX+i];
        out=fopen(path,"wb");
        if(!out || fwrite(global,sizeof(float),64,out)!=64)MPI_Abort(MPI_COMM_WORLD,7);
        if(fclose(out))MPI_Abort(MPI_COMM_WORLD,8);
        printf("{\"halo_mismatch\":%d,\"divide\":%d,\"invalid\":%d,\"ranks\":%d}\n",
               any,division,invalid,size);
    }
    free(all);free(packed);
    free_matrix(rho,0,NY+1,0,NX+1);free_matrix(pi,0,NY+1,0,NX+1);
    free_matrix(u,0,NY+1,0,NX+1);free_matrix(corner,0,NY+1,0,NX+1);
    MPI_Buffer_detach(&bsend,&detached);free(bsend);fclose(FP);
}
int main(int argc,char **argv) {
    if(argc==5 && !strcmp(argv[1],"local")) {
        INVMAT1=atoi(argv[2]);local_case(argv[3],atoi(argv[4]));return 0;
    }
    if(argc==7 && !strcmp(argv[1],"halo")) {
        MPI_Init(&argc,&argv);
        INVMAT1=atoi(argv[4]);halo_case(atoi(argv[2]),atoi(argv[3]),argv[5],argv[6]);
        MPI_Finalize();return 0;
    }
    return 2;
}
