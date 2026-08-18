from __future__ import annotations

import hashlib
import math
import struct
from array import array
from dataclasses import dataclass
from typing import Mapping, Sequence

from tests.cases.psv_fwi_gradient import PSVFWIGradientConfig
from tests.utilities.fwi_gradient import gaussian_direction


PARAMETERS = ("vp", "vs", "rho")


@dataclass(frozen=True)
class PSVTaylorCase:
    name: str
    config: PSVFWIGradientConfig
    background: Mapping[str, tuple[float, ...]]
    target: Mapping[str, tuple[float, ...]]
    direction: Mapping[str, tuple[float, ...]]
    delta_model: Mapping[str, tuple[float, ...]]
    active: tuple[str, ...]
    holdout: bool = False


def _float32(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(array("f", values))


def _zero(config: PSVFWIGradientConfig) -> tuple[float, ...]:
    return (0.0,) * config.cell_count


def _gaussian(
    config: PSVFWIGradientConfig, *, x_m: float, y_m: float, sigma_m: float
) -> tuple[float, ...]:
    return tuple(
        gaussian_direction(
            nx=config.nx,
            ny=config.ny,
            dh_m=config.dh_m,
            center_x_m=x_m,
            center_y_m=y_m,
            sigma_m=sigma_m,
        )
    )


def _normalized(values: Sequence[float]) -> tuple[float, ...]:
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        raise ValueError("smooth pattern is degenerate")
    return tuple(value / scale for value in values)


def _scaled(
    background: Sequence[float], pattern: Sequence[float], fraction: float
) -> tuple[float, ...]:
    result = _float32(
        value * (1.0 + fraction * weight)
        for value, weight in zip(background, pattern)
    )
    if any(value <= 0.0 for value in result):
        raise ValueError("target model must remain positive")
    return result


def _case(
    *,
    name: str,
    config: PSVFWIGradientConfig,
    background: Mapping[str, tuple[float, ...]],
    target_patterns: Mapping[str, tuple[float, ...]],
    directions: Mapping[str, tuple[float, ...]],
    active: tuple[str, ...],
    target_fraction: float = 0.02,
    holdout: bool = False,
) -> PSVTaylorCase:
    zero = _zero(config)
    target = {
        component: _scaled(
            background[component],
            target_patterns.get(component, zero),
            target_fraction,
        )
        for component in PARAMETERS
    }
    direction = {
        component: tuple(directions.get(component, zero))
        for component in PARAMETERS
    }
    delta = {
        component: tuple(
            value * weight
            for value, weight in zip(background[component], direction[component])
        )
        for component in PARAMETERS
    }
    for component in PARAMETERS:
        if len(background[component]) != config.cell_count:
            raise ValueError(f"{component} background size differs from grid")
        if component in active and max(abs(value) for value in direction[component]) == 0.0:
            raise ValueError(f"active {component} direction is degenerate")
        if component not in active and any(direction[component]):
            raise ValueError(f"inactive {component} direction is nonzero")
    return PSVTaylorCase(
        name=name,
        config=config,
        background=background,
        target=target,
        direction=direction,
        delta_model=delta,
        active=active,
        holdout=holdout,
    )


def _homogeneous_cases(config: PSVFWIGradientConfig) -> tuple[PSVTaylorCase, ...]:
    background = {
        "vp": _float32([config.vp_m_s] * config.cell_count),
        "vs": _float32([config.vs_m_s] * config.cell_count),
        "rho": _float32([config.density_kg_m3] * config.cell_count),
    }
    vp_target = _gaussian(config, x_m=455.0, y_m=355.0, sigma_m=68.0)
    vs_target = _gaussian(config, x_m=485.0, y_m=385.0, sigma_m=72.0)
    rho_target = _gaussian(config, x_m=515.0, y_m=410.0, sigma_m=76.0)
    vp_direction = _gaussian(config, x_m=505.0, y_m=385.0, sigma_m=82.0)
    vs_direction = _gaussian(config, x_m=535.0, y_m=415.0, sigma_m=78.0)
    rho_direction = _gaussian(config, x_m=475.0, y_m=440.0, sigma_m=74.0)

    return (
        _case(
            name="homogeneous_vp_only",
            config=config,
            background=background,
            target_patterns={"vp": vp_target},
            directions={"vp": vp_direction},
            active=("vp",),
        ),
        _case(
            name="homogeneous_vs_only",
            config=config,
            background=background,
            target_patterns={"vs": vs_target},
            directions={"vs": vs_direction},
            active=("vs",),
        ),
        _case(
            name="homogeneous_rho_only",
            config=config,
            background=background,
            target_patterns={"rho": rho_target},
            directions={"rho": rho_direction},
            active=("rho",),
        ),
        _case(
            name="homogeneous_joint",
            config=config,
            background=background,
            target_patterns={
                "vp": _gaussian(config, x_m=430.0, y_m=345.0, sigma_m=70.0),
                "vs": _gaussian(config, x_m=505.0, y_m=405.0, sigma_m=66.0),
                "rho": _gaussian(config, x_m=560.0, y_m=365.0, sigma_m=75.0),
            },
            directions={
                "vp": _gaussian(config, x_m=480.0, y_m=375.0, sigma_m=84.0),
                "vs": _gaussian(config, x_m=545.0, y_m=435.0, sigma_m=73.0),
                "rho": _gaussian(config, x_m=510.0, y_m=455.0, sigma_m=79.0),
            },
            active=PARAMETERS,
        ),
    )


def _heterogeneous_holdout(config: PSVFWIGradientConfig) -> PSVTaylorCase:
    fields = {name: [] for name in PARAMETERS}
    target_patterns = {name: [] for name in PARAMETERS}
    directions = {name: [] for name in PARAMETERS}
    for ix in range(1, config.nx + 1):
        x = (ix - 0.5) / config.nx
        for iy in range(1, config.ny + 1):
            y = (iy - 0.5) / config.ny
            fields["vp"].append(
                config.vp_m_s
                * (1.0 + 0.045 * math.sin(3.0 * math.pi * x + 0.2)
                   * math.cos(2.0 * math.pi * y - 0.1))
            )
            fields["vs"].append(
                config.vs_m_s
                * (1.0 + 0.052 * math.cos(2.0 * math.pi * x - 0.35)
                   * math.sin(3.0 * math.pi * y + 0.15))
            )
            fields["rho"].append(
                config.density_kg_m3
                * (1.0 + 0.038 * math.sin(2.0 * math.pi * (x + 0.7 * y))
                   + 0.012 * math.cos(4.0 * math.pi * x - math.pi * y))
            )

            target_patterns["vp"].append(
                math.exp(-((x - 0.43) ** 2 + (y - 0.58) ** 2) / 0.022)
                * math.cos(2.0 * math.pi * x + math.pi * y)
            )
            target_patterns["vs"].append(
                math.exp(-((x - 0.62) ** 2 + (y - 0.41) ** 2) / 0.026)
                * math.sin(3.0 * math.pi * x - 2.0 * math.pi * y)
            )
            target_patterns["rho"].append(
                math.exp(-((x - 0.54) ** 2 + (y - 0.52) ** 2) / 0.031)
                * math.cos(4.0 * math.pi * x + 2.0 * math.pi * y)
            )

            taper = math.sin(math.pi * x) ** 2 * math.sin(math.pi * y) ** 2
            directions["vp"].append(
                taper * math.exp(-((x - 0.60) ** 2 + (y - 0.46) ** 2) / 0.055)
                * math.cos(3.0 * math.pi * x - math.pi * y + 0.1)
            )
            directions["vs"].append(
                taper * math.exp(-((x - 0.47) ** 2 + (y - 0.61) ** 2) / 0.048)
                * math.sin(2.0 * math.pi * x + 3.0 * math.pi * y - 0.3)
            )
            directions["rho"].append(
                taper * math.exp(-((x - 0.57) ** 2 + (y - 0.38) ** 2) / 0.052)
                * math.cos(4.0 * math.pi * x + 2.0 * math.pi * y + 0.25)
            )

    background = {name: _float32(values) for name, values in fields.items()}
    normalized_targets = {
        name: _normalized(values) for name, values in target_patterns.items()
    }
    normalized_directions = {
        name: _normalized(values) for name, values in directions.items()
    }
    return _case(
        name="heterogeneous_joint_holdout",
        config=config,
        background=background,
        target_patterns=normalized_targets,
        directions=normalized_directions,
        active=PARAMETERS,
        target_fraction=0.018,
        holdout=True,
    )


def taylor_cases() -> tuple[PSVTaylorCase, ...]:
    config = PSVFWIGradientConfig()
    return (*_homogeneous_cases(config), _heterogeneous_holdout(config))


def model_at_epsilon(case: PSVTaylorCase, epsilon: float) -> dict[str, list[float]]:
    model = {
        component: list(
            _float32(
                value * (1.0 + epsilon * weight)
                for value, weight in zip(
                    case.background[component], case.direction[component]
                )
            )
        )
        for component in PARAMETERS
    }
    if any(value <= 0.0 for values in model.values() for value in values):
        raise ValueError("Taylor perturbation produced a non-positive model")
    return model


def sequence_sha256(values: Sequence[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def case_hashes(case: PSVTaylorCase) -> dict[str, dict[str, str]]:
    return {
        group: {
            component: sequence_sha256(getattr(case, group)[component])
            for component in PARAMETERS
        }
        for group in ("background", "target", "direction", "delta_model")
    }


def gradient_contributions(
    gradients: Mapping[str, Sequence[float]],
    delta_model: Mapping[str, Sequence[float]],
) -> dict[str, float]:
    contributions = {
        component: math.fsum(
            gradient * delta
            for gradient, delta in zip(gradients[component], delta_model[component])
        )
        for component in PARAMETERS
    }
    contributions["total"] = math.fsum(contributions.values())
    return contributions
