/* Isolated host preparation bridge. No CPU/M8e numerical source changes. */
#include <stdlib.h>
#define denise_elastic_psv_born_apply_j m9_private_denise_elastic_psv_born_apply_j
#define denise_elastic_psv_born_apply_jt m9_private_denise_elastic_psv_born_apply_jt
#define denise_elastic_psv_born_checkpoint_roundtrip m9_private_denise_elastic_psv_born_checkpoint_roundtrip
#define denise_elastic_psv_born_copy_strain m9_private_denise_elastic_psv_born_copy_strain
#define denise_elastic_psv_born_cpml_memory_peak m9_private_denise_elastic_psv_born_cpml_memory_peak
#define denise_elastic_psv_born_create m9_private_denise_elastic_psv_born_create
#define denise_elastic_psv_born_destroy m9_private_denise_elastic_psv_born_destroy
#define denise_elastic_psv_born_estimate_replay_storage m9_private_denise_elastic_psv_born_estimate_replay_storage
#define denise_elastic_psv_born_get_segment_bounds m9_private_denise_elastic_psv_born_get_segment_bounds
#define denise_elastic_psv_born_is_prepared m9_private_denise_elastic_psv_born_is_prepared
#define denise_elastic_psv_born_last_error m9_private_denise_elastic_psv_born_last_error
#define denise_elastic_psv_born_nonlinear m9_private_denise_elastic_psv_born_nonlinear
#define denise_elastic_psv_born_prepare m9_private_denise_elastic_psv_born_prepare
#define denise_elastic_psv_born_set_replay_segments m9_private_denise_elastic_psv_born_set_replay_segments
#define denise_elastic_psv_born_storage_diagnostics m9_private_denise_elastic_psv_born_storage_diagnostics

#include "m9_elastic_host.h"
#define calloc m9_host_calloc
#define free m9_host_free
#include "../PSV/elastic_psv_born.c"
#undef calloc
#undef free
struct m9_host { struct denise_elastic_psv_born *cpu; struct denise_elastic_psv_born_config cfg; };
int m9_host_create(const struct denise_elastic_psv_born_config *cfg,struct m9_host **out){
 struct m9_host *h;
 if(!out)return -1;
 *out=NULL;h=m9_host_calloc(1,sizeof(*h));if(!h)return -1;
 if(m9_private_denise_elastic_psv_born_create(cfg,&h->cpu)){m9_host_free(h);return -1;}
 h->cfg=*cfg;h->cfg.lambda=h->cpu->lambda;h->cfg.mu=h->cpu->mu;h->cfg.rho=h->cpu->rho;
 h->cfg.source_samples=h->cpu->source_samples;h->cfg.receiver_i=h->cpu->receiver_i;h->cfg.receiver_j=h->cpu->receiver_j;
 *out=h;return 0;
}
void m9_host_destroy(struct m9_host **out){
 if(out && *out){m9_private_denise_elastic_psv_born_destroy(&(*out)->cpu);m9_host_free(*out);*out=NULL;}
}
const char *m9_host_error(void){return m9_private_denise_elastic_psv_born_last_error();}
const struct denise_elastic_psv_born_config *m9_host_config(const struct m9_host *h){return &h->cfg;}
void m9_host_maps(const struct m9_host *h,float *out){
 const struct denise_elastic_psv_born *c=h->cpu;
 const float *map[5]={c->lambda,c->mu,c->invrho_x,c->invrho_y,c->mu_corner};
 size_t padded=(size_t)(c->nx+4)*(c->ny+4);int k,j,i;
 for(k=0;k<5;k++)for(j=-2;j<c->ny+2;j++)for(i=-2;i<c->nx+2;i++)
  out[(size_t)k*padded+(size_t)(j+2)*(c->nx+4)+i+2]=map[k][cell(c,wrap(j,c->ny),wrap(i,c->nx))];
}
void m9_host_profiles(const struct m9_host *h,float *out){
 int k;size_t n=0;
 for(k=0;k<4;k++){const struct pml_profile *p=&h->cpu->profile[k];
  memcpy(out+n,p->kappa,(size_t)p->length*sizeof(float));n+=p->length;
  memcpy(out+n,p->a,(size_t)p->length*sizeof(float));n+=p->length;
  memcpy(out+n,p->b,(size_t)p->length*sizeof(float));n+=p->length;}
}

size_t m9_host_metadata_bytes(void){return sizeof(struct m9_host)+sizeof(struct denise_elastic_psv_born);}
