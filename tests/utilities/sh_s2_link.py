"""SH-S2: extra translation units for stub harnesses that compile the exact
SH multi-shot wrappers (src/SH/obj_sh_visc_exact.c, grad_obj_sh_visc_exact.c),
which now call the time window and offset mask of
src/SH/visco_sh_exact_data_operators.c.  The stage filter is the legacy
apply_tdfilt, which these harnesses already provide."""

S2_LINK_SOURCES = (
    "src/SH/visco_sh_exact_data_operators.c",
    "tests/utilities/sh_s2_link_support.c",
)
