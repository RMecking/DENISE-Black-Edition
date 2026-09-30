#ifndef DENISE_ELASTIC_PSV_MIGRATION_MPI_H
#define DENISE_ELASTIC_PSV_MIGRATION_MPI_H
#include "denise_elastic_psv_born_mpi.h"
#include "denise_elastic_psv_migration.h"
/* Request models and shot data are OWNED/local; geometry is global. Result
 * images remain owned tiles. All ranks visit each physical shot in order. */
int denise_elastic_psv_migrate_mpi(
    const struct denise_elastic_psv_migration_request *,MPI_Comm,int,int,
    struct denise_elastic_psv_migration_result *,
    struct denise_elastic_psv_born_mpi_diagnostics *);
int denise_elastic_psv_migration_mode2_mpi(void);
int denise_elastic_psv_migration_mode2_count_sources(void);
#endif
