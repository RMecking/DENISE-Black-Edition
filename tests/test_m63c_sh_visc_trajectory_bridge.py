"""M6.3c-8b2-a production ``sh_visc`` trajectory-bridge contracts."""

from __future__ import annotations

import re
from pathlib import Path


def _source(repository_root: Path, path: str) -> str:
    return (repository_root / path).read_text(encoding="utf-8")


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"\bvoid\s+{name}\s*\(", source)
    assert match, f"missing function {name}"
    start = source.index("{", match.end())
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1 : index]
    raise AssertionError(f"unterminated function {name}")


def test_existing_api_is_a_null_trajectory_wrapper(repository_root: Path):
    header = _source(repository_root, "include/fd.h")
    source = _source(repository_root, "src/SH/sh_visc.c")
    wrapper = _compact(_function_body(source, "sh_visc"))

    assert re.search(r"\bvoid\s+sh_visc\s*\(", header)
    assert re.search(r"\bvoid\s+sh_visc_with_material_trajectory\s*\(", header)
    assert wrapper.count("sh_visc_with_material_trajectory(") == 1
    assert wrapper.endswith("NULL);")


def test_new_entry_owns_the_only_forward_loop(repository_root: Path):
    source = _source(repository_root, "src/SH/sh_visc.c")
    implementation = _compact(
        _function_body(source, "sh_visc_with_material_trajectory")
    )
    wrapper = _compact(_function_body(source, "sh_visc"))

    assert implementation.count("for(nt=1;nt<=NT;nt++)") == 1
    assert "for(nt=1;nt<=NT;nt++)" not in wrapper


def test_trajectory_preflight_is_explicit_and_fail_fast(repository_root: Path):
    implementation = _compact(
        _function_body(
            _source(repository_root, "src/SH/sh_visc.c"),
            "sh_visc_with_material_trajectory",
        )
    )
    preflight = implementation.split("zero_denise_visc_SH", 1)[0]

    for condition in (
        "mode!=0",
        "DTINV!=1",
        "trajectory->dtinv!=1",
        "trajectory->nx!=NX",
        "trajectory->ny!=NY",
        "trajectory->nsteps!=NT",
        "trajectory->steps==NULL",
    ):
        assert condition in preflight
    assert "visco_sh_material_observable_is_active()" in preflight
    assert preflight.count("err(") == 2


def test_capture_lifecycle_brackets_real_kernel_updates(repository_root: Path):
    implementation = _compact(
        _function_body(
            _source(repository_root, "src/SH/sh_visc.c"),
            "sh_visc_with_material_trajectory",
        )
    )
    begin = implementation.index(
        "visco_sh_material_observable_begin_step(trajectory,nt-1)"
    )
    velocity = implementation.index("update_v_PML_SH(", begin)
    stress = implementation.index("update_s_visc_PML_SH(", velocity)
    end = implementation.index("visco_sh_material_observable_end_step()", stress)

    assert begin < velocity < stress < end
    assert implementation.count("visco_sh_material_observable_begin_step(") == 1
    assert implementation.count("visco_sh_material_observable_end_step()") == 1


def test_capture_kernels_and_active_fwi_path_remain_unchanged(repository_root: Path):
    velocity = _source(repository_root, "src/SH/update_v_PML_SH.c")
    stress = _source(repository_root, "src/SH/update_s_visc_PML_SH.c")
    driver = _compact(_source(repository_root, "src/SH/FWI_SH_visc.c"))

    assert velocity.count("visco_sh_material_observable_is_active()") == 1
    assert stress.count("visco_sh_material_observable_is_active()") == 1
    assert "L2sum=grad_obj_sh(" in driver
    assert "grad_obj_sh_visc_exact(" not in driver
    assert "visco_sh_reverse_time_adjoint_material(" not in driver
