from __future__ import annotations

import hashlib
import json
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, generate_case as generate_elastic_psv
from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case as generate_elastic_sh


@dataclass(frozen=True)
class ViscoelasticSHConfig(HomogeneousSHConfig):
    qs: float = 50.0
    relaxation_frequencies_hz: tuple[float, ...] = (10.0,)
    q_parameterization_mode: int = 0
    q_approx_fmin_hz: float = 0.0
    q_approx_fmax_hz: float = 0.0
    q_approx_df_hz: float = 0.0


@dataclass(frozen=True)
class ViscoelasticPSVConfig(HomogeneousPSVConfig):
    qp: float = 50.0
    qs: float = 50.0
    relaxation_frequencies_hz: tuple[float, ...] = (10.0,)
    q_parameterization_mode: int = 0
    q_approx_fmin_hz: float = 0.0
    q_approx_fmax_hz: float = 0.0
    q_approx_df_hz: float = 0.0


def _write_constant_float_model(path: Path, value: float, count: int) -> None:
    values = array("f", [value]) * count
    with path.open("wb") as stream:
        values.tofile(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enable_viscoelasticity(directory: Path, config) -> None:
    frequencies_hz = config.relaxation_frequencies_hz
    if not frequencies_hz or any(value <= 0.0 for value in frequencies_hz):
        raise ValueError("At least one positive relaxation frequency is required")
    path = directory / "denise.inp"
    content = path.read_text(encoding="ascii")
    # read_par.c consumes the first character of every non-comment record before
    # calling fscanf. A leading space is therefore required for the one-letter L
    # key; without it the parser silently leaves the global default L=0 in place.
    content = content.replace("\nL =0\n", f"\n L ={len(frequencies_hz)}\n", 1)
    frequency_record = " ".join(str(value) for value in frequencies_hz)
    content = content.replace("FL =10.0", f"FL ={frequency_record}", 1)
    content = content.replace("homogeneous elastic", "homogeneous viscoelastic", 1)
    if config.q_parameterization_mode not in (0, 1):
        raise ValueError("Q parameterization mode must be 0 (legacy) or 1 (physical-Q)")
    if config.q_parameterization_mode == 1:
        if not (
            config.q_approx_fmin_hz > 0.0
            and config.q_approx_fmax_hz >= config.q_approx_fmin_hz
            and config.q_approx_df_hz > 0.0
        ):
            raise ValueError("Physical-Q mode requires a valid explicit approximation band")
        content += (
            "# optional positional parameter 116\n"
            "Q_PARAMETERIZATION_MODE =1\n"
            "# optional positional parameter 117\n"
            f"Q_APPROX_FMIN ={config.q_approx_fmin_hz}\n"
            "# optional positional parameter 118\n"
            f"Q_APPROX_FMAX ={config.q_approx_fmax_hz}\n"
            "# optional positional parameter 119\n"
            f"Q_APPROX_DF ={config.q_approx_df_hz}\n"
        )
    path.write_text(content, encoding="ascii")


def _write_metadata(directory: Path, config, nprocx: int, nprocy: int, model_names: tuple[str, ...]) -> None:
    model = directory / "model" / "homogeneous"
    metadata = asdict(config) | {
        "nprocx": nprocx,
        "nprocy": nprocy,
        "model_sha256": {
            name: _sha256(model.with_suffix(f".{name}")) for name in model_names
        },
    }
    (directory / "case.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate_viscoelastic_sh_case(
    directory: Path,
    *,
    config: ViscoelasticSHConfig,
    nprocx: int = 1,
    nprocy: int = 1,
) -> ViscoelasticSHConfig:
    if config.qs <= 0.0:
        raise ValueError("Qs must be positive")
    generate_elastic_sh(directory, config=config, nprocx=nprocx, nprocy=nprocy)
    model = directory / "model" / "homogeneous"
    _write_constant_float_model(model.with_suffix(".qs"), config.qs, config.nx * config.ny)
    _enable_viscoelasticity(directory, config)
    _write_metadata(directory, config, nprocx, nprocy, ("vs", "rho", "qs"))
    return config


def generate_viscoelastic_psv_case(
    directory: Path,
    *,
    config: ViscoelasticPSVConfig,
    nprocx: int = 1,
    nprocy: int = 1,
) -> ViscoelasticPSVConfig:
    if config.qp <= 0.0 or config.qs <= 0.0:
        raise ValueError("Qp and Qs must be positive")
    generate_elastic_psv(directory, config=config, nprocx=nprocx, nprocy=nprocy)
    model = directory / "model" / "homogeneous"
    cell_count = config.nx * config.ny
    _write_constant_float_model(model.with_suffix(".qp"), config.qp, cell_count)
    _write_constant_float_model(model.with_suffix(".qs"), config.qs, cell_count)
    _enable_viscoelasticity(directory, config)
    _write_metadata(directory, config, nprocx, nprocy, ("vp", "vs", "rho", "qp", "qs"))
    return config
