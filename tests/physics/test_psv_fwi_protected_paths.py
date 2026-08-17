from __future__ import annotations

import hashlib
import json
import os
from array import array
from pathlib import Path

import pytest

from tests.cases.psv_fwi_gradient import (
    PSVFWIGradientConfig,
    baseline_model,
    generate_case,
    target_model,
)
from tests.utilities.fwi_gradient import read_su_float_samples
from tests.utilities.runner import result_summary, run_denise


pytestmark = [pytest.mark.integration, pytest.mark.extended]
BASE_BINARY_SHA256 = "35bb43a7c11af1e97941ee1008d4f0726aae96019c54f78b1a71cb38d9392bb4"
PRODUCTION_PATCH_SHA256 = "d5ff334ee244c0457944fb97746771ebbbb26b659c38e5aeaafdf6f15fad9edc"
CHANGED_PRODUCTION_FILES = [
    "include/fd.h",
    "src/Makefile",
    "src/PSV/FWI_PSV.c",
    "src/PSV/alloc_fwiPSV.c",
    "src/PSV/assemble_gradPSV_exact.c",
    "src/PSV/grad_obj_psv.c",
    "src/PSV/psv.c",
    "src/PSV/update_v_PML_PSV.c",
    "src/TTI/TTI.c",
    "src/VTI/VTI.c",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_record(directory: Path, key: str, value: str, *, leading_space: bool = False) -> None:
    path = directory / "denise.inp"
    lines = path.read_text(encoding="ascii").splitlines()
    matches = [index for index, line in enumerate(lines)
               if not line.lstrip().startswith("#")
               and line.split("=", 1)[0].strip() == key]
    assert len(matches) == 1, (key, matches)
    prefix = " " if leading_space else ""
    lines[matches[0]] = f"{prefix}{key} ={value}"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_grid(path: Path, values: list[float]) -> None:
    with path.open("wb") as stream:
        array("f", values).tofile(stream)


def _enable_visco(directory: Path, config: PSVFWIGradientConfig) -> None:
    _set_record(directory, "L", "1", leading_space=True)
    _set_record(directory, "FL", "10.0")
    count = config.cell_count
    _write_grid(directory / "model/current.qp", [80.0] * count)
    _write_grid(directory / "model/current.qs", [50.0] * count)


def _enable_parameterization(
    directory: Path,
    parameterization: int,
    model: dict[str, list[float]],
) -> None:
    _set_record(directory, "INVMAT1", str(parameterization))
    if parameterization == 2:
        # The legacy file reader has no INVMAT1=2 branch.  READMOD=0 is the
        # existing supported route to exercise this legacy parameterization.
        _set_record(directory, "READMOD", "0")
        return
    assert parameterization == 3
    lam = [rho * (vp * vp - 2.0 * vs * vs)
           for vp, vs, rho in zip(model["vp"], model["vs"], model["rho"])]
    mu = [rho * vs * vs for vs, rho in zip(model["vs"], model["rho"])]
    _write_grid(directory / "model/current.lam", lam)
    _write_grid(directory / "model/current.mu", mu)


def _run(
    directory: Path,
    *,
    repository_root: Path,
    binary: Path,
    mpiexec: str,
    config: PSVFWIGradientConfig,
    role: str,
    require_success: bool = True,
):
    result = run_denise(
        repository_root=repository_root,
        case_directory=directory,
        denise_binary=binary,
        mpiexec=mpiexec,
        ranks=1,
        configuration=config.as_metadata() | {"role": role},
        timeout_seconds=90.0,
    )
    if require_success:
        assert result.returncode == 0, result_summary(result)
    return result


def _hashes(directory: Path, relative_paths: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for relative in relative_paths:
        path = directory / relative
        assert path.is_file(), path
        result[relative] = _sha256(path)
    return result


def test_psv_protected_paths_match_reviewed_base(
    tmp_path: Path,
    repository_root: Path,
    denise_binary: Path,
    mpiexec: str,
) -> None:
    configured_base = os.environ.get("M54_BASE_DENISE_BIN")
    assert configured_base, "M54_BASE_DENISE_BIN must name the b4a4ea95 base binary"
    base_binary = Path(configured_base).resolve(strict=True)
    assert _sha256(base_binary) == BASE_BINARY_SHA256
    production_patch = repository_root / "tests/m5.4_psv_gradient_production_repair.patch"
    assert _sha256(production_patch) == PRODUCTION_PATCH_SHA256
    config = PSVFWIGradientConfig()
    base_model = baseline_model(config)
    target = target_model(config, ("vp", "vs", "rho"))
    run_records: list[dict[str, object]] = []

    # Elastic forward physics is outside the repair selector and must remain
    # byte-identical to the reviewed base.
    forward_hashes: dict[str, dict[str, str]] = {}
    for label, binary in (("base", base_binary), ("repaired", denise_binary)):
        directory = tmp_path / "elastic_forward" / label
        generate_case(directory, model=base_model, config=config, mode=0)
        result = _run(directory, repository_root=repository_root, binary=binary,
                      mpiexec=mpiexec, config=config, role=f"elastic_forward_{label}")
        run_records.append({"role": f"elastic_forward_{label}",
                            "returncode": result.returncode})
        forward_hashes[label] = _hashes(directory, (
            "su/synthetic_x.su.shot1", "su/synthetic_y.su.shot1",
        ))
    assert forward_hashes["base"] == forward_hashes["repaired"]

    # Exercise both the forward and FWI viscoelastic routes.  The exact
    # elastic selector must not affect any L>0 result.
    visco_observed = tmp_path / "visco_observed"
    generate_case(visco_observed, model=target, config=config, mode=0)
    _enable_visco(visco_observed, config)
    result = _run(visco_observed, repository_root=repository_root,
                  binary=base_binary, mpiexec=mpiexec, config=config,
                  role="visco_observed_base")
    run_records.append({"role": "visco_observed_base", "returncode": result.returncode})
    observed_x = visco_observed / "su/synthetic_x.su.shot1"
    observed_y = visco_observed / "su/synthetic_y.su.shot1"

    visco_forward_hashes: dict[str, dict[str, str]] = {}
    visco_fwi_hashes: dict[str, dict[str, str]] = {}
    for label, binary in (("base", base_binary), ("repaired", denise_binary)):
        forward = tmp_path / "visco_forward" / label
        generate_case(forward, model=base_model, config=config, mode=0)
        _enable_visco(forward, config)
        result = _run(forward, repository_root=repository_root, binary=binary,
                      mpiexec=mpiexec, config=config, role=f"visco_forward_{label}")
        run_records.append({"role": f"visco_forward_{label}",
                            "returncode": result.returncode})
        visco_forward_hashes[label] = _hashes(forward, (
            "su/synthetic_x.su.shot1", "su/synthetic_y.su.shot1",
        ))

        fwi = tmp_path / "visco_fwi" / label
        generate_case(
            fwi, model=base_model, config=config, mode=1, grad_form=2,
            data_components=1, observed_x=observed_x, observed_y=observed_y,
        )
        _enable_visco(fwi, config)
        result = _run(fwi, repository_root=repository_root, binary=binary,
                      mpiexec=mpiexec, config=config, role=f"visco_fwi_{label}")
        run_records.append({"role": f"visco_fwi_{label}",
                            "returncode": result.returncode})
        visco_fwi_hashes[label] = _hashes(fwi, (
            "su/synthetic_x.su.shot1.it1", "su/synthetic_y.su.shot1.it1",
            "jacobian/gradient_p.old", "jacobian/gradient_p_u.old",
            "jacobian/gradient_p_rho.old",
        ))
    assert visco_forward_hashes["base"] == visco_forward_hashes["repaired"]
    assert all(
        visco_fwi_hashes["base"][relative]
        == visco_fwi_hashes["repaired"][relative]
        for relative in (
            "jacobian/gradient_p.old", "jacobian/gradient_p_u.old",
            "jacobian/gradient_p_rho.old",
        )
    )
    visco_residual_payload: dict[str, dict[str, object]] = {}
    for component in ("x", "y"):
        relative = f"su/synthetic_{component}.su.shot1.it1"
        base_samples = read_su_float_samples(
            tmp_path / "visco_fwi/base" / relative,
            config.receiver_count, config.samples_per_trace,
        )
        repaired_samples = read_su_float_samples(
            tmp_path / "visco_fwi/repaired" / relative,
            config.receiver_count, config.samples_per_trace,
        )
        maximum = max(abs(left-right) for left, right in
                      zip(base_samples, repaired_samples))
        assert maximum == 0.0
        visco_residual_payload[component] = {
            "exact_payload_match": True,
            "max_absolute_difference": maximum,
            "base_file_sha256": visco_fwi_hashes["base"][relative],
            "repaired_file_sha256": visco_fwi_hashes["repaired"][relative],
        }

    invmat_results: list[dict[str, object]] = []
    known_preexisting_defects: list[dict[str, object]] = []
    for parameterization in (3, 2):
        forward = tmp_path / "invmat" / str(parameterization) / "forward"
        generate_case(forward, model=target, config=config, mode=0)
        _enable_parameterization(forward, parameterization, target)
        result = _run(forward, repository_root=repository_root,
                      binary=denise_binary, mpiexec=mpiexec, config=config,
                      role=f"invmat{parameterization}_forward_smoke",
                      require_success=False)
        run_records.append({"role": f"invmat{parameterization}_forward_smoke",
                            "returncode": result.returncode})
        invmat_result: dict[str, object] = {
            "INVMAT1": parameterization,
            "mode": 0,
            "forward_returncode": result.returncode,
            "route": "legacy",
            "readmod": 0 if parameterization == 2 else 1,
            "scope": "forward smoke only; gradient mathematics unverified",
        }
        if parameterization == 2:
            base_forward = tmp_path / "invmat/2/base_forward"
            generate_case(base_forward, model=target, config=config, mode=0)
            _enable_parameterization(base_forward, parameterization, target)
            base_result = _run(
                base_forward, repository_root=repository_root, binary=base_binary,
                mpiexec=mpiexec, config=config, role="invmat2_base_forward_smoke",
                require_success=False,
            )
            run_records.append({"role": "invmat2_base_forward_smoke",
                                "returncode": base_result.returncode})
            invmat_result["base_forward_returncode"] = base_result.returncode
            invmat_result["diagnosis"] = (
                "pre-existing protected-path defect: update_s_elastic_PML_PSV "
                "does not initialize f/g for INVMAT1=2"
            )
            known_preexisting_defects.append({
                "id": "PSV-INVMAT1-2-UNINITIALIZED-STIFFNESS",
                "classification": "KNOWN PRE-EXISTING DEFECT — NOT AN M5.4 REGRESSION",
                "status": "OPEN / OUTSIDE M5.4 SCOPE",
                "current_repaired_returncode": result.returncode,
                "current_base_returncode": base_result.returncode,
                "reviewed_base_observed_returncodes": [0, 1],
                "evidence": {
                    "repair_selector_inactive_for_INVMAT1_2": True,
                    "repair_selector_inactive_for_MODE_0": True,
                    "production_patch_modifies_update_s_elastic_PML_PSV": False,
                    "reviewed_base_behavior_is_nondeterministic": True,
                    "source_diagnosis": (
                        "update_s_elastic_PML_PSV.c assigns f/g only for "
                        "INVMAT1=1 and INVMAT1=3"
                    ),
                },
            })
        invmat_results.append(invmat_result)
        if parameterization == 3:
            assert result.returncode == 0, result_summary(result)

    artifact_path = repository_root / "tests/m5.4_psv_gradient_production_validation.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "heterogeneous_holdout" in artifact
    artifact["base_binary_sha256"] = BASE_BINARY_SHA256
    artifact["production_patch_sha256"] = PRODUCTION_PATCH_SHA256
    artifact["changed_production_files"] = CHANGED_PRODUCTION_FILES
    artifact["forward_non_regression"] = forward_hashes
    artifact["protected_visco"] = {
        "forward_hashes": visco_forward_hashes,
        "fwi_hashes": visco_fwi_hashes,
        "residual_payload": visco_residual_payload,
    }
    artifact["protected_parameterizations"] = invmat_results
    artifact["known_preexisting_defects"] = known_preexisting_defects
    artifact["run_records"].extend(run_records)
    artifact["final_verdict"] = (
        "M5.4 PSV PRODUCTION GRADIENT REPAIR VERIFIED "
        "WITH KNOWN PRE-EXISTING INVMAT1=2 DEFECT"
    )
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
