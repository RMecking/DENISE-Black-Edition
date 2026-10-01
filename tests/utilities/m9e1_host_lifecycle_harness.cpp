/* ASan without WSL driver initialization: actual production allocator + bridge. */
#define DENISE_M9_HOST_SANITIZER_ONLY
#include "../../src/CUDA/m9_elastic_forward.cu"
int main() {
    float l[63],m[63],rho[63],source[3]={1.25f,-3.5f,.875f};
    int ri[3]={6,2,6},rj[3]={3,5,3};
    denise_elastic_psv_born_config q={};
    for(int k=0;k<63;k++){l[k]=5e9f+k*1e6f;m[k]=3e9f+k*3e5f;rho[k]=2000+k;}
    q.nx=9;q.ny=7;q.nt=3;q.fw=2;q.dh=10;q.dt=.0005f;q.invmat1=3;q.fdorder=4;
    q.ndt=q.dtinv=q.mpi_size=1;q.receiver_components=2;q.lambda=l;q.mu=m;q.rho=rho;
    q.source_i=4;q.source_j=3;q.source_samples=source;q.receiver_count=3;q.receiver_i=ri;q.receiver_j=rj;
    q.cpml_enabled=1;q.pml_reflection=.001f;q.pml_power=2;q.pml_kmax=1.3f;q.pml_fpml=15;q.pml_damping_speed=2500;
    denise_cuda_m9_fault(0);m9_host *h=NULL;
    if(m9_host_create(&q,&h))return 1;
    float *maps=(float *)m9_host_calloc(5*13*11,sizeof(float));
    float *profiles=(float *)m9_host_calloc(6*(9+7),sizeof(float));
    m9_host_maps(h,maps);m9_host_profiles(h,profiles);
    for(int k=0;k<5*13*11;k++)if(!std::isfinite(maps[k]))return 2;
    m9_host_free(maps);m9_host_free(profiles);m9_host_destroy(&h);
    size_t allocation_count=calls;
    for(size_t at=1;at<=allocation_count;at++) {
        denise_cuda_m9_fault(at);
        if(m9_host_create(&q,&h)==0) {
            maps=(float *)m9_host_calloc(5*13*11,sizeof(float));
            profiles=(float *)m9_host_calloc(6*(9+7),sizeof(float));
            m9_host_free(maps);m9_host_free(profiles);
        }
        m9_host_destroy(&h);if(host_owned)return 3;
        if(!strstr(denise_cuda_m9_last_error(),"injected"))return 4;
    }
    printf("M9E1_HOST_ASAN_PASS allocation_sites=%zu host_bytes=%zu\n",allocation_count,host_owned);
    return 0;
}
