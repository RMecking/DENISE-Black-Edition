"""Freeze the M6.3c-8a active viscoelastic SH FWI integration audit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "tests" / "m6.3c_c8_active_path_inventory.json"
AUDIT_PATH = ROOT / "docs" / "m6.3c_c8_active_path_audit.md"
LOCKED_INPUT_SHA = "47caffc441c5f3862682f3e36bbb45e11997e151"


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, capture_output=True, text=True
    )


def _locked_source(path: str) -> str:
    return _git("show", f"{LOCKED_INPUT_SHA}:{path}").stdout


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def _body(text: str, function_name: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    match = re.search(rf"\b{re.escape(function_name)}\s*\([^;]*?\)\s*\{{", text, re.DOTALL)
    assert match, f"function {function_name} not found"
    start = text.find("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise AssertionError(f"unterminated function {function_name}")


def _inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def test_inventory_is_complete_and_sha_locked() -> None:
    data = _inventory()
    assert data["locked_input_sha"] == LOCKED_INPUT_SHA
    assert _git("merge-base", "--is-ancestor", LOCKED_INPUT_SHA, "HEAD", check=False).returncode == 0

    statuses = Counter(item["status"] for item in data["inventory"].values())
    assert dict(statuses) == data["summary"] == {"PASS": 3, "RED": 14, "UNRESOLVED": 1}
    assert len(data["inventory"]) == 18
    assert set(data["c8b_minimal_scope"]) == {
        "include/fd.h",
        "src/Makefile",
        "src/SH/alloc_matSH.c",
        "src/SH/readmod_visc_SH.c",
        "src/SH/FWI_SH_visc.c",
        "src/SH/grad_obj_sh_visc_exact.c",
    }

    for path, expected in data["source_sha256"].items():
        actual = hashlib.sha256(_locked_source(path).encode()).hexdigest()
        assert actual == expected, path


def test_active_base_and_trial_use_different_forward_physics() -> None:
    physics = _compact(_body(_locked_source("src/SH/physics_SH.c"), "physics_SH"))
    driver = _compact(_body(_locked_source("src/SH/FWI_SH_visc.c"), "FWI_SH_visc"))
    base = _compact(_body(_locked_source("src/SH/grad_obj_sh.c"), "grad_obj_sh"))
    trial = _compact(_body(_locked_source("src/SH/obj_sh.c"), "obj_sh"))
    line_search = _compact(_body(_locked_source("src/SH/step_length_est_sh.c"), "step_length_est_sh"))

    assert "if(MODE==1)" in physics and "if(L){FWI_SH_visc();}" in physics
    assert "grad_obj_sh(" in driver
    assert "grad_obj_sh_visc(" not in driver
    assert "visco_sh_reverse_time_adjoint_material(" not in driver
    assert "sh(" in base and ",0," in base and ",1," in base
    assert "sh_visc(" not in base
    assert "if(L){sh_visc(" in trial and ",2," in trial
    assert "matcopy_elastic_SH(" in trial and "matcopy_SH(" not in trial
    assert "calc_mat_change_test_SH_visc(" in line_search and "obj_sh(" in line_search


def test_owned_q_gradient_and_trial_update_are_disconnected() -> None:
    fd_header = _locked_source("include/fd.h")
    mat_struct = _compact(fd_header)
    reader = _compact(_body(_locked_source("src/SH/readmod_visc_SH.c"), "readmod_visc_SH"))
    update = _compact(_body(_locked_source("src/SH/calc_mat_change_test_SH_visc.c"), "calc_mat_change_test_SH_visc"))
    store_pcg = _compact(_body(_locked_source("src/SH/store_PCG_SH_visc.c"), "store_PCG_SH_visc"))
    extract_pcg = _compact(_body(_locked_source("src/SH/extract_PCG_SH_visc.c"), "extract_PCG_SH_visc"))

    assert "q_to_tau(" in reader and "taus[jj][ii]=q_to_tau(" in reader
    assert "ptaus" in mat_struct and "pqs" not in mat_struct
    assert "tsnp1[" not in update and "waveconv_ts[" not in update
    assert "waveconv_ts[" not in store_pcg and "waveconv_ts[" not in extract_pcg
    assert "unp1[j][i]=u[j][i]-EPSILON_u*waveconv_u[j][i]" in update
    assert "rhonp1[j][i]=rho[j][i]-EPSILON_rho*waveconv_rho[j][i]" in update


def test_legacy_visco_gradient_is_not_the_locked_c7_chain() -> None:
    legacy = _compact(_body(_locked_source("src/SH/grad_obj_sh_visc.c"), "grad_obj_sh_visc"))
    elastic = _compact(_body(_locked_source("src/SH/assemble_gradSH_exact.c"), "assemble_gradSH_exact"))

    assert "sh_visc(" in legacy and ",0," in legacy and ",1," in legacy
    assert "visco_sh_reverse_time_adjoint_material(" not in legacy
    assert "waveconv_ts[j][i]+=" not in legacy
    assert "INVMAT1!=1" in elastic and "return;" in elastic
    assert "waveconv_ts" not in elastic and "ptaus" not in elastic


def test_audit_document_freezes_scope_without_claiming_integration() -> None:
    text = AUDIT_PATH.read_text(encoding="utf-8")
    for phrase in (
        LOCKED_INPUT_SHA,
        "3 `PASS`, 14 `RED`, and 1 `UNRESOLVED`",
        "src/SH/grad_obj_sh_visc_exact.c",
        "later C8c scope",
        "No active FWI switch is made by this audit.",
    ):
        assert phrase in text
