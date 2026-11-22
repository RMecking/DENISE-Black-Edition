/* SH-S2 link support for stub harnesses that compile the exact SH
 * multi-shot wrappers.  Tentative definitions (-fcommon) of the SH-S2
 * workflow globals read by src/SH/exact_sh_data_ops.h; they merge with any
 * definition a harness already provides.  The harnesses keep TIME_FILT,
 * TIMEWIN and OFFSET_MUTE at 0, so these values are only read by the
 * configuration check. */
#include "fd.h"

float FC_START, FC_END, DT, GAMMA, TWLENGTH_MINUS, TWLENGTH_PLUS, OFFSETC, DH;
int ORDER;
char PICKS_FILE[STRING_SIZE];
