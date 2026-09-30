#ifndef DENISE_ELASTIC_PSV_BORN_MPI_H
#define DENISE_ELASTIC_PSV_BORN_MPI_H
#include "denise_elastic_psv_born.h"
#include <mpi.h>

/* Collective APIs: all communicator ranks call in the same order. Global
 * config geometry is identical on ranks; model inputs are compact OWNED tiles.
 * Local traces retain ascending global receiver ordinals; no global data is
 * retained by the numerical context. Returned images/strains are owned tiles.
 * The communicator is duplicated and released by collective destroy(). */
struct denise_elastic_psv_born_mpi;
struct denise_elastic_psv_born_mpi_config {
    struct denise_elastic_psv_born_config global;
    MPI_Comm communicator;
    int nprocx,nprocy;
};
struct denise_elastic_psv_born_mpi_diagnostics {
    struct denise_elastic_psv_born_storage_diagnostics replay;
    int rank,size,local_nx,local_ny,offset_x,offset_y,local_receivers,source_owned;
    size_t full_bytes,replay_estimate,retained_backend_bytes;
    size_t model_bytes,halo_buffer_bytes,local_data_bytes;
    size_t forward_halo_bytes_per_step,adjoint_halo_bytes_per_step,material_bytes;
    unsigned long long full_min,full_max,full_sum,replay_min,replay_max,replay_sum;
    size_t forward_stress_exchanges,forward_velocity_exchanges;
    size_t adjoint_stress_exchanges,adjoint_velocity_exchanges,material_transposes;
    size_t forward_halo_bytes,adjoint_halo_bytes;
};
const char *denise_elastic_psv_born_mpi_last_error(void);
int denise_elastic_psv_born_mpi_create(const struct denise_elastic_psv_born_mpi_config *,struct denise_elastic_psv_born_mpi **);
void denise_elastic_psv_born_mpi_destroy(struct denise_elastic_psv_born_mpi **);
int denise_elastic_psv_born_mpi_prepare(struct denise_elastic_psv_born_mpi *,float *);
int denise_elastic_psv_born_mpi_apply_j(struct denise_elastic_psv_born_mpi *,const float *,const float *,float *);
int denise_elastic_psv_born_mpi_apply_jt(struct denise_elastic_psv_born_mpi *,const float *,double *,double *);
int denise_elastic_psv_born_mpi_nonlinear(struct denise_elastic_psv_born_mpi *,const float *,const float *,float *);
int denise_elastic_psv_born_mpi_copy_strain(const struct denise_elastic_psv_born_mpi *,int,float *);
int denise_elastic_psv_born_mpi_checkpoint_roundtrip(struct denise_elastic_psv_born_mpi *,int);
int denise_elastic_psv_born_mpi_set_replay_segments(struct denise_elastic_psv_born_mpi *,int);
int denise_elastic_psv_born_mpi_estimate_replay_storage(const struct denise_elastic_psv_born_mpi *,int,size_t *);
int denise_elastic_psv_born_mpi_select_backend(struct denise_elastic_psv_born_mpi *,int);
int denise_elastic_psv_born_mpi_diagnostics(struct denise_elastic_psv_born_mpi *,struct denise_elastic_psv_born_mpi_diagnostics *);
int denise_elastic_psv_born_mpi_local_receivers(const struct denise_elastic_psv_born_mpi *);
int denise_elastic_psv_born_mpi_receiver_ordinal(const struct denise_elastic_psv_born_mpi *,int);
/* Root-only global pointers. These assemble/copy unique ownership, never sum. */
int denise_elastic_psv_born_mpi_gather_data(struct denise_elastic_psv_born_mpi *,const float *,float *);
int denise_elastic_psv_born_mpi_scatter_data(struct denise_elastic_psv_born_mpi *,const float *,float *);
int denise_elastic_psv_born_mpi_gather_image(struct denise_elastic_psv_born_mpi *,const double *,double *);
/* Explicit material/profile inspection used by interface/wrap oracle gates. */
int denise_elastic_psv_born_mpi_copy_maps(const struct denise_elastic_psv_born_mpi *,float *,float *,float *);
int denise_elastic_psv_born_mpi_copy_profile(const struct denise_elastic_psv_born_mpi *,int,float *,float *,float *);
int denise_elastic_psv_born_mpi_get_segment_bounds(const struct denise_elastic_psv_born_mpi *,int,int *,int *);
int denise_elastic_psv_born_mpi_storage_diagnostics(const struct denise_elastic_psv_born_mpi *,struct denise_elastic_psv_born_storage_diagnostics *);
int denise_elastic_psv_born_mpi_is_prepared(const struct denise_elastic_psv_born_mpi *);
float denise_elastic_psv_born_mpi_cpml_memory_peak(const struct denise_elastic_psv_born_mpi *);
#endif
