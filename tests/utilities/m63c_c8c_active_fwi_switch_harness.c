/* B5B harness anchor: compiled production FWI_SH_visc.o is required by the
 * Python oracle's build stage.  Runtime full-driver fixtures are deliberately
 * kept separate from B1--B5A numerical fixtures to avoid reimplementing the
 * driver sequence in test code. */
#include "fd.h"

int main(void) {
    /* This anchor makes the production entry point a required link symbol. */
    void (*entry)(void) = FWI_SH_visc;
    return entry == 0;
}
