"""Direct deterministic C8c physical-Q trial-state boundary oracle."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest


def _read(repository_root: Path, relative_path: str) -> str:
    return (repository_root / relative_path).read_text(encoding="utf-8")


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_trial_state_api_static_legacy_exclusion_and_inactive_driver(
    repository_root: Path,
) -> None:
    header = _read(repository_root, "include/fd.h")
    makefile = _read(repository_root, "src/Makefile")
    source = _compact(_read(repository_root, "src/SH/visco_sh_exact_trial_state.c"))
    active_driver = _compact(_read(repository_root, "src/SH/FWI_SH_visc.c"))
    q_mapping = _compact(_read(repository_root, "src/q_parameterization.c"))

    assert "struct visco_sh_exact_trial_state_request" in header
    assert "visco_sh_exact_build_trial_parameter_state(" in header
    assert "visco_sh_exact_trial_state.c" in makefile
    assert "q_to_tau(" in source
    for forbidden in (
        "waveconv_ts",
        "gradg_ts",
        "gradp_ts",
        "ptaus_old",
        "calc_mat_change_test_SH_visc",
        "ass_gradSH_visc",
        "descent(",
        "PCG(",
        "LBFGS(",
        "matcopy_SH(",
        "av_tau(",
        "prepare_update_s_visc_SH(",
    ):
        assert forbidden not in source
    assert "visco_sh_exact_build_trial_parameter_state(" not in active_driver

    # q_to_tau either returns a finite positive tau or calls err before it can
    # return an invalid value. Consequently the helper's -13 path cannot be
    # exercised as a returning public-API failure without replacing production
    # q_to_tau, which this oracle deliberately never does.
    assert "if(!(inverse_tau>0.0)||!isfinite(inverse_tau))err(" in q_mapping
    assert "return(float)(1.0/inverse_tau)" in q_mapping


@pytest.fixture(scope="module")
def trial_state_harness(
    tmp_path_factory: pytest.TempPathFactory, repository_root: Path
) -> Path:
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the C8c trial-state harness"
    executable = tmp_path_factory.mktemp("m63c_c8c_trial_state") / "trial_state"
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-fcommon",
        "-I",
        str(repository_root / "include"),
        str(repository_root / "tests/utilities/m63c_c8c_trial_state_harness.c"),
        str(repository_root / "src/SH/visco_sh_exact_trial_state.c"),
        str(repository_root / "src/q_parameterization.c"),
        "-o",
        str(executable),
        "-lm",
    ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    return executable


def test_real_trial_state_helper_contract(trial_state_harness: Path) -> None:
    result = subprocess.run(
        [str(trial_state_harness)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    assert records == [
        {
            "normal_subtractive": True,
            "zero_step": True,
            "physical_q": True,
            "reject_to_base": True,
            "no_clipping": True,
            "input_immutable": True,
            "halos_untouched": True,
            "transactional_failures": True,
            "legacy_mode_fail_closed": True,
            "tau_return_failure_unreachable": True,
        }
    ]
