"""M6.3c-8b1 owned physical-Q storage and lifecycle contracts."""

from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.utilities.m63c_material_map_reference import (
    QMapping,
    physical_mapping,
    q_to_tau,
)


EXPECTED_REAL_CALLERS = {
    "src/SH/FD_SH.c",
    "src/SH/FD_grad_SH.c",
    "src/SH/FWI_SH_visc.c",
}


@pytest.fixture(scope="module")
def allocation_contract(tmp_path_factory, repository_root: Path):
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the C8b1 allocation contract"
    library = tmp_path_factory.mktemp("m63c8b1") / "libm63c8b1.so"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fPIC",
        "-shared",
        "-fcommon",
        "-I",
        str(repository_root / "include"),
        str(repository_root / "tests/utilities/m63c_owned_q_allocation_harness.c"),
        str(repository_root / "src/SH/alloc_matSH.c"),
        "-o",
        str(library),
    ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    api = ctypes.CDLL(str(library))
    api.m63c8b1_owned_q_allocation_contract.argtypes = [ctypes.c_int]
    api.m63c8b1_owned_q_allocation_contract.restype = ctypes.c_int
    return api.m63c8b1_owned_q_allocation_contract


def _source(repository_root: Path, path: str) -> str:
    return (repository_root / path).read_text(encoding="utf-8")


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_owned_q_and_tau_are_distinct_l_gated_allocations(allocation_contract):
    assert allocation_contract(1) == 1
    assert allocation_contract(3) == 1
    assert allocation_contract(0) == 1


def test_reader_preserves_q_and_derives_tau_authoritatively(repository_root: Path):
    reader = _compact(_source(repository_root, "src/SH/readmod_visc_SH.c"))
    assert "fread(&q_value,sizeof(float),1,fp_qs);" in reader
    assert "qs[jj][ii]=q_value;" in reader
    assert "taus[jj][ii]=q_to_tau(qs[jj][ii],&q_mapping);" in reader
    assert "tau_to_q" not in reader
    assert "if(INVMAT1==1)" in reader and "if(INVMAT1==3)" in reader

    legacy_mapping = QMapping(0)
    physical_q_mapping = physical_mapping((3.0, 7.0, 13.0), 2.0, 18.0, 0.5)
    for q_value in (24.0, 41.5, 97.0):
        legacy = q_to_tau(q_value, legacy_mapping)
        physical = q_to_tau(q_value, physical_q_mapping)
        assert legacy > 0.0 and physical > 0.0


def test_all_real_reader_callers_forward_owned_q(repository_root: Path):
    callers = set()
    for path in (repository_root / "src/SH").glob("*.c"):
        text = _source(repository_root, str(path.relative_to(repository_root)))
        if path.name != "readmod_visc_SH.c" and re.search(r"\breadmod_visc_SH\s*\(", text):
            callers.add(path.relative_to(repository_root).as_posix())
            compact = _compact(text)
            assert "readmod_visc_SH(matSH.prho,matSH.pu,matSH.pqs,matSH.ptaus,matSH.peta)" in compact
    assert callers == EXPECTED_REAL_CALLERS


def test_all_visco_material_lifecycles_free_owned_q(repository_root: Path):
    for path in EXPECTED_REAL_CALLERS:
        compact = _compact(_source(repository_root, path))
        assert compact.count("alloc_matSH(&matSH)") == 1
        assert compact.count("free_matrix(matSH.pqs,-nd+1,NY+nd,-nd+1,NX+nd)") == 1


def test_c8b1_does_not_switch_active_gradient_or_touch_c8a(repository_root: Path):
    driver = _compact(_source(repository_root, "src/SH/FWI_SH_visc.c"))
    assert "L2sum=grad_obj_sh(" in driver
    assert "grad_obj_sh_visc_exact(" not in driver
    assert "visco_sh_reverse_time_adjoint_material(" not in driver

    inventory = _source(repository_root, "tests/m6.3c_c8_active_path_inventory.json")
    assert '"locked_input_sha": "47caffc441c5f3862682f3e36bbb45e11997e151"' in inventory
    assert '"base_gradient_entry": {' in inventory
    assert '"status": "RED"' in inventory


def test_c8b2_trajectory_hook_analysis_is_source_supported(repository_root: Path):
    propagator = _source(repository_root, "src/SH/sh_visc.c")
    velocity = _source(repository_root, "src/SH/update_v_PML_SH.c")
    stress = _source(repository_root, "src/SH/update_s_visc_PML_SH.c")
    assert "sh_visc_with_material_trajectory" in propagator
    assert "visco_sh_material_observable_begin_step" in propagator
    assert "visco_sh_material_observable_end_step" in propagator
    assert "visco_sh_material_observable_is_active" in velocity
    assert "visco_sh_material_observable_is_active" in stress
