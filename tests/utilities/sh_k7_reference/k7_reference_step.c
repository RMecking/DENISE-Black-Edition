/* K-7 test oracle: the exact SH reverse step of the base b24161a1, frozen
 * verbatim in visco_sh_full_state_adjoint_step_base.c (git blob a13d88a0),
 * compiled as a separate translation unit under k7ref_* names.  The harness
 * compares the production step against it bitwise. */

#include "fd.h"

#define visco_sh_full_state_adjoint_step k7ref_full_state_adjoint_step
#define visco_sh_full_state_adjoint_step_material k7ref_full_state_adjoint_step_material
#define M63C_FULL_STATE_ADJOINT_EMBEDDED
#include "visco_sh_full_state_adjoint_step_base.c"
