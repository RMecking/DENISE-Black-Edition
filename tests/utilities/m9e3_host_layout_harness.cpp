/* ASan/UBSan: actual new checked layout + transactional image scratch owner.
 * Does not initialize CUDA or repeat the old canonical bridge sanitizer sweep. */
#define DENISE_M9_HOST_SANITIZER_ONLY
#include "../../src/CUDA/m9_elastic_forward.cu"
int main() {
    struct denise_cuda_m9_adjoint_diagnostics d={};
    size_t total=0;
    for(size_t base=0;base<16;base++) {
        if(!adjoint_layout(81,25,48,base,d,total))return 1;
        if(total!=base+(8-base%8)%8+40*81+112*25+48)return 2;
        if((base+d.alignment_bytes)%8)return 3;
    }
    size_t before=total;struct denise_cuda_m9_adjoint_diagnostics keep=d;
    const size_t overflow[][4]={{SIZE_MAX,1,8,0},{81,SIZE_MAX,8,0},
        {81,25,SIZE_MAX,0},{81,25,48,SIZE_MAX-4}};
    for(const auto &x:overflow) {
        if(adjoint_layout(x[0],x[1],x[2],x[3],d,total))return 4;
        if(memcmp(&d,&keep,sizeof(d)) || total!=before)return 5;
    }
    for(size_t at=0;at<=2;at++) {
        denise_cuda_m9_fault(at);
        void *metadata=m9_host_calloc(1,sizeof(d));
        double *images=(double *)m9_host_calloc(50,sizeof(double));
        if(images)for(int k=0;k<50;k++)images[k]=.125*k;
        m9_host_free(images);m9_host_free(metadata);
        if(host_owned || device_owned || events_owned)return 6;
        if(at && !strstr(denise_cuda_m9_last_error(),"injected"))return 7;
    }
    printf("M9E3_HOST_LAYOUT_PASS alignments=16 overflow_cases=4 allocation_faults=2 ownership=0,0,0\n");
    return 0;
}
