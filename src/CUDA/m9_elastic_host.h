#ifndef M9_ELASTIC_HOST_H
#define M9_ELASTIC_HOST_H
#include "denise_elastic_psv_born.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Private canonical host preparation, CPU numerical source remains unchanged. */
struct m9_host;
void *m9_host_calloc(size_t,size_t);
void m9_host_free(void *);
int m9_host_create(const struct denise_elastic_psv_born_config *,struct m9_host **);
void m9_host_destroy(struct m9_host **);
const char *m9_host_error(void);
const struct denise_elastic_psv_born_config *m9_host_config(const struct m9_host *);
void m9_host_maps(const struct m9_host *,float *five_padded);
void m9_host_profiles(const struct m9_host *,float *packed);
size_t m9_host_metadata_bytes(void);
#ifdef __cplusplus
}
#endif
#endif
