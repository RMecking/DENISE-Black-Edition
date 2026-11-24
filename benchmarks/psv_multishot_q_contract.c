/* Report the production physical-Q mapping and equivalent legacy input. */
#include "fd.h"
void err(char text[]) { fprintf(stderr,"%s\n",text); exit(93); }
int main(void) {
 struct q_tau_mapping m;float fl[2]={0,3.0f};float tau,legacy,legacy_tau;
 init_q_tau_mapping(&m,1,1,fl,1.0f,10.0f,.25f);
 tau=q_to_tau(80.0f,&m);legacy=(float)(2.0/(double)tau);
 legacy_tau=(float)(2.0/(double)legacy);
 if(memcmp(&tau,&legacy_tau,sizeof(float))) return 1;
 printf("{\"inverse_tau_per_q\":%.17g,\"inverse_tau_offset\":%.17g,\"tau_f32\":%.17g,\"legacy_q_f32\":%.17g,\"legacy_tau_f32\":%.17g}\n",m.inverse_tau_per_q,m.inverse_tau_offset,(double)tau,(double)legacy,(double)legacy_tau);
 return 0;
}
