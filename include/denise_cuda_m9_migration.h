#ifndef DENISE_CUDA_M9_MIGRATION_H
#define DENISE_CUDA_M9_MIGRATION_H
#include "denise_elastic_psv_migration.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Same canonical request/result layout and ordinary malloc/free output ownership. */
int denise_cuda_m9_migrate_request(const struct denise_elastic_psv_migration_request *,struct denise_elastic_psv_migration_result *);
const char *denise_cuda_m9_migration_last_error(void);
void denise_cuda_m9_migration_result_destroy(struct denise_elastic_psv_migration_result *);
#ifdef __cplusplus
}
#endif
#endif
