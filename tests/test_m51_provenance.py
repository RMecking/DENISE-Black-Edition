from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


EXPECTED_PRODUCTION_FILES = {
    "include/fd.h",
    "src/Makefile",
    "src/SH/FWI_SH.c",
    "src/SH/FWI_SH_visc.c",
    "src/SH/alloc_fwiSH.c",
    "src/SH/assemble_gradSH_exact.c",
    "src/SH/ass_gradSH.c",
    "src/SH/debug/sh.c",
    "src/SH/debug/update_v_PML_SH.c",
    "src/SH/grad_obj_sh.c",
    "src/SH/grad_obj_sh_visc.c",
    "src/SH/sh.c",
    "src/SH/sh_visc.c",
    "src/SH/update_v_PML_SH.c",
}


def test_m51_production_patch_provenance(repository_root: Path):
    patch_path = repository_root / "tests" / "m5.1_sh_gradient_production_repair.patch"
    validation_path = (
        repository_root / "tests" / "m5.1_sh_gradient_production_validation.json"
    )
    patch = patch_path.read_bytes()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    assert hashlib.sha256(patch).hexdigest() == validation["production_patch_sha256"]
    changed_files = set(
        re.findall(rb"^diff --git a/(.+?) b/.+$", patch, flags=re.MULTILINE)
    )
    changed_files = {path.decode("utf-8") for path in changed_files}
    assert changed_files == EXPECTED_PRODUCTION_FILES
    assert changed_files == set(validation["changed_production_files"])
    assert "src/Makefile" in changed_files
    assert "src/SH/assemble_gradSH_exact.c" in changed_files

    # src/SH/debug contains historical/manual copies and is not compiled by the
    # normal DENISE target.
    makefile = (repository_root / "src" / "Makefile").read_text(encoding="utf-8")
    assert "SH/debug/" not in makefile
