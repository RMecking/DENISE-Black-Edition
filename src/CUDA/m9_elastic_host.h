#ifndef M9_ELASTIC_HOST_H
#define M9_ELASTIC_HOST_H
#include "denise_elastic_psv_born.h"
#include "denise_elastic_psv_migration.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Private canonical host preparation, CPU numerical source remains unchanged. */
struct m9_host;
void *m9_host_calloc(size_t,size_t);
void m9_host_free(void *);
int m9_host_gate(const char *);
void m9_host_begin(void);
void *m9_host_publish(size_t,size_t);
int m9_host_create(const struct denise_elastic_psv_born_config *,struct m9_host **);
void m9_host_destroy(struct m9_host **);
const char *m9_host_error(void);
const struct denise_elastic_psv_born_config *m9_host_config(const struct m9_host *);
int m9_host_has_fluid(const struct m9_host *);
void m9_host_maps(const struct m9_host *,float *five_padded);
void m9_host_profiles(const struct m9_host *,float *packed);
size_t m9_host_metadata_bytes(void);
void m9_host_active_counts(const struct m9_host *,size_t active[8]);
int m9_host_validate_migration(const struct denise_elastic_psv_migration_request *,size_t *,size_t *,size_t *,size_t *);
const char *m9_host_migration_error(void);
#ifdef __cplusplus
}
#endif
#endif
