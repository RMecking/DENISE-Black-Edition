from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from tests.cases.sh_fwi_gradient import (
    SHFWIGradientConfig,
    _records,
    _write_common,
    _write_float_grid,
    _write_parameters,
)


def _validated(values: Sequence[float], config: SHFWIGradientConfig, name: str) -> list[float]:
    result = list(values)
    if len(result) != config.cell_count:
        raise ValueError(f"{name} size differs from model grid")
    if any(value <= 0.0 for value in result):
        raise ValueError(f"{name} must remain positive")
    return result


def _write_workflow(directory: Path, *, active_vs: bool, active_rho: bool) -> None:
    if not (active_vs or active_rho):
        raise ValueError("At least one Taylor parameter must be active")
    header = (
        "PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ "
        "INV_VP_ITER INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP "
        "WD_DAMP1 EPRECOND LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE "
        "OFFSET_MUTE OFFSETC SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER\n"
    )
    inv_vs_iter = 1 if active_vs else 99
    inv_rho_iter = 1 if active_rho else 99
    values = (
        f"0.01 0 0.0 0.0 4 0 0.0 0.0 0.0 99 {inv_vs_iter} "
        f"{inv_rho_iter} 99 0 0.0 0.0 0 2 0 0 0.0 0.0 0 0 0.0 1.0 "
        "1.0 0 0.0 0\n"
    )
    (directory / "workflow.inp").write_text(header + values, encoding="ascii")


def _write_model(
    directory: Path,
    *,
    config: SHFWIGradientConfig,
    vs_background: Sequence[float],
    rho_background: Sequence[float],
    delta_vs: Sequence[float],
    delta_rho: Sequence[float],
    epsilon: float,
) -> tuple[list[float], list[float]]:
    vs0 = _validated(vs_background, config, "Vs background")
    rho0 = _validated(rho_background, config, "density background")
    dvs = list(delta_vs)
    drho = list(delta_rho)
    if len(dvs) != config.cell_count or len(drho) != config.cell_count:
        raise ValueError("Taylor direction size differs from model grid")
    vs = _validated(
        [value + epsilon * perturbation for value, perturbation in zip(vs0, dvs)],
        config,
        "perturbed Vs",
    )
    rho = _validated(
        [value + epsilon * perturbation for value, perturbation in zip(rho0, drho)],
        config,
        "perturbed density",
    )
    _write_float_grid(directory / "model" / "current.vs", vs)
    _write_float_grid(directory / "model" / "current.rho", rho)
    return vs, rho


def generate_taylor_observed_case(
    directory: Path,
    *,
    config: SHFWIGradientConfig,
    target_vs: Sequence[float],
    target_rho: Sequence[float],
) -> None:
    _write_common(directory, config)
    _write_float_grid(
        directory / "model" / "current.vs",
        _validated(target_vs, config, "target Vs"),
    )
    _write_float_grid(
        directory / "model" / "current.rho",
        _validated(target_rho, config, "target density"),
    )
    records = _records(config, mode=0, grad_form=1)
    records[65] = "INVMAT1 =1"
    _write_parameters(directory, records)
    _write_workflow(directory, active_vs=True, active_rho=True)
    (directory / "case.json").write_text(
        json.dumps(config.as_metadata() | {"role": "m5.2_observed_target"}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def generate_taylor_fwi_case(
    directory: Path,
    *,
    config: SHFWIGradientConfig,
    observed_su: Path,
    vs_background: Sequence[float],
    rho_background: Sequence[float],
    delta_vs: Sequence[float],
    delta_rho: Sequence[float],
    epsilon: float,
    grad_form: int,
    active_vs: bool,
    active_rho: bool,
) -> None:
    _write_common(directory, config)
    _write_model(
        directory,
        config=config,
        vs_background=vs_background,
        rho_background=rho_background,
        delta_vs=delta_vs,
        delta_rho=delta_rho,
        epsilon=epsilon,
    )
    (directory / "observed_y.su.shot1").write_bytes(observed_su.read_bytes())
    records = _records(config, mode=1, grad_form=grad_form)
    records[65] = "INVMAT1 =1"
    _write_parameters(directory, records)
    _write_workflow(directory, active_vs=active_vs, active_rho=active_rho)
    (directory / "case.json").write_text(
        json.dumps(
            config.as_metadata()
            | {
                "role": "m5.2_taylor",
                "epsilon": epsilon,
                "grad_form": grad_form,
                "active_vs": active_vs,
                "active_rho": active_rho,
                "model_parameterization": "Vs,rho physical INVMAT1=1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
