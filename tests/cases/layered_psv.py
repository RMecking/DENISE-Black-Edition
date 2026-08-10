from __future__ import annotations

import json
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

from tests.cases.homogeneous_psv import HomogeneousPSVConfig, _parameter_lines


@dataclass(frozen=True)
class LayeredPSVConfig:
    nx: int = 240
    ny: int = 240
    dh_m: float = 10.0
    time_s: float = 0.85
    dt_s: float = 0.0004
    vp1_m_s: float = 3000.0
    vs1_m_s: float = 1800.0
    rho1_kg_m3: float = 2000.0
    vp2_m_s: float = 3600.0
    vs2_m_s: float = 2100.0
    rho2_kg_m3: float = 2300.0
    interface_upper_row: int = 120
    source_x_m: float = 1200.0
    source_y_m: float = 500.0
    source_frequency_hz: float = 10.0
    source_type: int = 1
    source_azimuth_deg: float = 0.0
    receivers_m: tuple[tuple[float, float], ...] = ((1200.0, 700.0),)
    fd_order: int = 8
    max_relative_error: int = 1
    absorbing_width_gridpoints: int = 15
    pml_frequency_hz: float = 10.0

    @property
    def interface_y_m(self) -> float:
        """Nominal continuum boundary between the upper and lower row centres.

        For DH=10 m and interface_upper_row=120, material row centres are at
        1195 m and 1205 m, so their boundary remains exactly 1200 m. Staggered
        field locations and material averaging do not redefine this property.
        """
        return self.interface_upper_row * self.dh_m

    @property
    def samples_per_trace(self) -> int:
        return round(self.time_s / self.dt_s)

    @property
    def receiver_count(self) -> int:
        return len(self.receivers_m)

    def run_config(self) -> HomogeneousPSVConfig:
        return HomogeneousPSVConfig(
            nx=self.nx, ny=self.ny, dh_m=self.dh_m, time_s=self.time_s, dt_s=self.dt_s,
            vp_m_s=max(self.vp1_m_s, self.vp2_m_s),
            vs_m_s=min(self.vs1_m_s, self.vs2_m_s),
            density_kg_m3=self.rho1_kg_m3,
            source_x_m=self.source_x_m, source_y_m=self.source_y_m,
            source_frequency_hz=self.source_frequency_hz, source_type=self.source_type,
            source_azimuth_deg=self.source_azimuth_deg, receivers_m=self.receivers_m,
            fd_order=self.fd_order, max_relative_error=self.max_relative_error,
            absorbing_width_gridpoints=self.absorbing_width_gridpoints,
            damping_velocity_m_s=max(self.vp1_m_s, self.vp2_m_s),
            pml_frequency_hz=self.pml_frequency_hz,
        )

    def as_metadata(self) -> dict[str, object]:
        data = asdict(self)
        data["receivers_m"] = [list(receiver) for receiver in self.receivers_m]
        data["interface_y_m"] = self.interface_y_m
        data["upper_rows_1_based"] = [1, self.interface_upper_row]
        data["lower_rows_1_based"] = [self.interface_upper_row + 1, self.ny]
        return data


def _write_layered(path: Path, upper: float, lower: float, config: LayeredPSVConfig) -> None:
    column = array("f", [upper]) * config.interface_upper_row
    column.extend(array("f", [lower]) * (config.ny - config.interface_upper_row))
    with path.open("wb") as stream:
        for _ in range(config.nx):
            column.tofile(stream)


def generate_case(
    directory: Path,
    *,
    config: LayeredPSVConfig | None = None,
    nprocx: int = 1,
    nprocy: int = 1,
) -> LayeredPSVConfig:
    config = config or LayeredPSVConfig()
    if config.nx % nprocx or config.ny % nprocy:
        raise ValueError("Grid dimensions must be divisible by the MPI decomposition")
    if not 1 <= config.interface_upper_row < config.ny:
        raise ValueError("Interface must lie between two model rows")
    for name in (
        "model", "su", "log", "snap", "wavelet", "jacobian", "taper",
        "picked_times", "trace_kill", "gravity",
    ):
        (directory / name).mkdir(parents=True, exist_ok=True)
    model = directory / "model" / "layered"
    _write_layered(model.with_suffix(".vp"), config.vp1_m_s, config.vp2_m_s, config)
    _write_layered(model.with_suffix(".vs"), config.vs1_m_s, config.vs2_m_s, config)
    _write_layered(model.with_suffix(".rho"), config.rho1_kg_m3, config.rho2_kg_m3, config)

    source = (
        f"{config.source_x_m} 0.0 {config.source_y_m} 0.0 "
        f"{config.source_frequency_hz} 1.0 {config.source_azimuth_deg} {config.source_type}\n"
    )
    (directory / "source.dat").write_text("1\n" + source, encoding="ascii")
    (directory / "receiver.dat").write_text(
        "".join(f"{x} {y}\n" for x, y in config.receivers_m), encoding="ascii"
    )
    run_config = config.run_config()
    records = _parameter_lines(run_config, nprocx, nprocy)
    records[22] = "MFILE =model/layered"
    parameters = "# Generated two-layer elastic P/SV verification case\n" + "".join(
        f"# positional parameter {index:03d}\n{line}\n"
        for index, line in enumerate(records, start=1)
    )
    (directory / "denise.inp").write_text(parameters, encoding="ascii")
    (directory / "workflow.inp").write_text(
        "# Unused in MODE=0; argv[2] remains mandatory.\n", encoding="ascii"
    )
    metadata = config.as_metadata() | {"nprocx": nprocx, "nprocy": nprocy}
    (directory / "case.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config
