/* External B3-S evaluator: link the unchanged production apply_tdfilt,
 * timedomain_filt and libcseife. No filter implementation is replaced. */
#include "fd.h"
float DT;
double *dvector(int nl,int nh) {
    double *v; if(nl!=1 || nh<1) exit(91);
    v=calloc((size_t)nh+1,sizeof(double)); if(!v) exit(92); return v;
}
void free_dvector(double *v,int nl,int nh) { (void)nl;(void)nh;free(v); }
void err(char text[]) { fprintf(stderr,"%s\n",text);exit(93); }
int main(int argc,char **argv) {
 int ns,ntr,order,i; float lo,hi; float **a; FILE *in,*out;
 if(argc!=10) return 90;
 ns=atoi(argv[1]);ntr=atoi(argv[2]);DT=strtof(argv[3],0);
 order=atoi(argv[4]);lo=strtof(argv[5],0);hi=strtof(argv[6],0);
 if(ns<2 || ntr<1 || !DT || !order) return 89;
 a=calloc(ntr+1,sizeof(float*));in=fopen(argv[7],"rb");out=fopen(argv[8],"wb");
 if(!a || !in || !out || strcmp(argv[9],"exact-production-filter")) return 88;
 for(i=1;i<=ntr;i++){a[i]=calloc(ns+1,sizeof(float));if(!a[i] || fread(a[i]+1,sizeof(float),ns,in)!=(size_t)ns)return 87;}
 if(fgetc(in)!=EOF)return 86;
 apply_tdfilt(a,ntr,ns,order,lo,hi);
 for(i=1;i<=ntr;i++)if(fwrite(a[i]+1,sizeof(float),ns,out)!=(size_t)ns)return 85;
 fclose(in);return fclose(out)!=0;
}
