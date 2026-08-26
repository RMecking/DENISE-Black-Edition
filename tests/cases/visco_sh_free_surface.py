from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case as generate_elastic
from tests.cases.homogeneous_viscoelastic import (
    ViscoelasticSHConfig,
    generate_viscoelastic_sh_case,
)
from tests.utilities.qstd_reference import target_q_to_tau
from tests.utilities.sh_free_surface import denise_grid_index, native_vz_position
from tests.utilities.visco_sh_free_surface import translated_image_geometry


GeometryName = Literal["normal", "oblique"]
Role = Literal[
    "candidate",
    "absorbing",
    "reference_combined",
    "reference_real",
    "reference_image",
]

PHYSICAL_L4_FREQUENCIES_HZ = (2.7105, 12.2792, 68.1930, 265.2297)


@dataclass(frozen=True)
class ViscoSHSurfaceScenario:
    name: GeometryName
    candidate: ViscoelasticSHConfig
    reference_plane_y_m: float = 1200.0
    reference_ny: int = 360
    reflection_window_half_width_s: float = 0.075

    @property
    def candidate_source_input(self) -> tuple[float, float]:
        return self.candidate.source_x_m, self.candidate.source_y_m

    @property
    def candidate_receiver_input(self) -> tuple[float, float]:
        return self.candidate.receiver_x_m[0], self.candidate.receiver_y_m

    def native_geometry(self):
        return translated_image_geometry(
            candidate_source=native_vz_position(
                self.candidate_source_input, self.candidate.dh_m
            ),
            candidate_receiver=native_vz_position(
                self.candidate_receiver_input, self.candidate.dh_m
            ),
            reference_plane_y=self.reference_plane_y_m,
            dh=self.candidate.dh_m,
        )

    def _input_from_native(self, point: tuple[float, float]) -> tuple[float, float]:
        half = 0.5 * self.candidate.dh_m
        return point[0] + half, point[1] + half

    def sources_for(self, role: Role) -> tuple[tuple[float, float], ...]:
        if role in ("candidate", "absorbing"):
            return (self.candidate_source_input,)
        geometry = self.native_geometry()
        real = self._input_from_native(geometry.reference_real_source)
        image = self._input_from_native(geometry.reference_image_source)
        if role == "reference_real":
            return (real,)
        if role == "reference_image":
            return (image,)
        return real, image

    def receiver_for(self, role: Role) -> tuple[float, float]:
        if role in ("candidate", "absorbing"):
            return self.candidate_receiver_input
        return self._input_from_native(self.native_geometry().reference_receiver)

    def config_for(self, role: Role) -> ViscoelasticSHConfig:
        receiver = self.receiver_for(role)
        source = self.sources_for(role)[0]
        return replace(
            self.candidate,
            ny=self.candidate.ny if role in ("candidate", "absorbing") else self.reference_ny,
            source_x_m=source[0],
            source_y_m=source[1],
            receiver_x_m=(receiver[0],),
            receiver_y_m=receiver[1],
        )

    def metadata(self, role: Role) -> dict[str, object]:
        config = self.config_for(role)
        geometry = self.native_geometry()
        sources = self.sources_for(role)
        receiver = self.receiver_for(role)
        source_native = [native_vz_position(source, config.dh_m) for source in sources]
        receiver_native = native_vz_position(receiver, config.dh_m)
        source_peak_s = 1.5 / config.source_frequency_hz
        direct_peak_s = source_peak_s + geometry.candidate_direct_distance / config.vs_m_s
        image_peak_s = source_peak_s + geometry.candidate_image_distance / config.vs_m_s
        comparison_stop_s = image_peak_s + self.reflection_window_half_width_s
        domain_x = config.nx * config.dh_m
        domain_y = config.ny * config.dh_m

        external_paths = []
        if role.startswith("reference"):
            for sx, sy in source_native:
                rx, ry = receiver_native
                external_paths.extend(
                    (
                        math.hypot(sx + rx, sy - ry),
                        math.hypot(2.0 * domain_x - sx - rx, sy - ry),
                        math.hypot(sx - rx, sy + ry),
                        math.hypot(sx - rx, 2.0 * domain_y - sy - ry),
                    )
                )
        earliest_external_s = (
            source_peak_s + min(external_paths) / config.vs_m_s
            if external_paths
            else None
        )
        tau_s = target_q_to_tau(
            target_q=config.qs,
            relaxation_frequencies_hz=config.relaxation_frequencies_hz,
            fmin_hz=config.q_approx_fmin_hz,
            fmax_hz=config.q_approx_fmax_hz,
            df_hz=config.q_approx_df_hz,
        )
        return {
            "name": self.name,
            "role": role,
            "free_surface": role == "candidate",
            "nominal_denise_input_coordinates_m": {
                "sources": [list(value) for value in sources],
                "receiver": list(receiver),
            },
            "rounded_grid_indices": {
                "sources": [
                    [denise_grid_index(value, config.dh_m) for value in source]
                    for source in sources
                ],
                "receiver": [denise_grid_index(value, config.dh_m) for value in receiver],
            },
            "native_vz_coordinates_m": {
                "sources": [list(value) for value in source_native],
                "receiver": list(receiver_native),
            },
            "reference_syz_plane_y_m": self.reference_plane_y_m,
            "candidate_direct_distance_m": geometry.candidate_direct_distance,
            "candidate_image_distance_m": geometry.candidate_image_distance,
            "reference_direct_distance_m": geometry.reference_direct_distance,
            "reference_image_distance_m": geometry.reference_image_distance,
            "source_peak_s": source_peak_s,
            "expected_direct_window_s": [
                direct_peak_s - self.reflection_window_half_width_s,
                direct_peak_s + self.reflection_window_half_width_s,
            ],
            "expected_reflected_image_window_s": [
                image_peak_s - self.reflection_window_half_width_s,
                image_peak_s + self.reflection_window_half_width_s,
            ],
            "comparison_stop_s": comparison_stop_s,
            "earliest_analytic_external_return_s": earliest_external_s,
            "external_return_outside_comparison": (
                earliest_external_s is None or earliest_external_s > comparison_stop_s
            ),
            "rheology": {
                "q_parameterization_mode": config.q_parameterization_mode,
                "qs": config.qs,
                "l": len(config.relaxation_frequencies_hz),
                "fl_hz": list(config.relaxation_frequencies_hz),
                "approximation_band_hz": [
                    config.q_approx_fmin_hz,
                    config.q_approx_fmax_hz,
                ],
                "approximation_df_hz": config.q_approx_df_hz,
                "fitted_tau_s": tau_s,
            },
            "numerics": asdict(config),
        }


def _base_config(
    *,
    geometry: GeometryName,
    fd_order: int,
    qs: float,
    frequencies_hz: tuple[float, ...],
) -> ViscoelasticSHConfig:
    if geometry == "normal":
        source_x, receiver_x, receiver_y = 1200.0, 1200.0, 1000.0
    else:
        source_x, receiver_x, receiver_y = 900.0, 1500.0, 900.0
    return ViscoelasticSHConfig(
        nx=240,
        ny=240,
        time_s=1.30,
        dt_s=0.0005,
        source_x_m=source_x,
        source_y_m=700.0,
        receiver_x_m=(receiver_x,),
        receiver_y_m=receiver_y,
        source_frequency_hz=8.0,
        fd_order=fd_order,
        qs=qs,
        relaxation_frequencies_hz=frequencies_hz,
        q_parameterization_mode=1,
        q_approx_fmin_hz=5.0,
        q_approx_fmax_hz=120.0,
        q_approx_df_hz=5.0,
    )


def runtime_scenario(
    *,
    geometry: GeometryName = "normal",
    fd_order: int = 12,
    qs: float = 50.0,
    frequencies_hz: tuple[float, ...] = (10.0,),
    reference_plane_y_m: float = 1200.0,
) -> ViscoSHSurfaceScenario:
    return ViscoSHSurfaceScenario(
        name=geometry,
        candidate=_base_config(
            geometry=geometry,
            fd_order=fd_order,
            qs=qs,
            frequencies_hz=frequencies_hz,
        ),
        reference_plane_y_m=reference_plane_y_m,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_case(
    directory: Path,
    *,
    scenario: ViscoSHSurfaceScenario,
    role: Role,
    nprocx: int = 1,
    nprocy: int = 1,
    elastic: bool = False,
) -> HomogeneousSHConfig:
    config = scenario.config_for(role)
    if elastic:
        elastic_config = HomogeneousSHConfig(
            **{
                field: getattr(config, field)
                for field in HomogeneousSHConfig.__dataclass_fields__
            }
        )
        generate_elastic(directory, config=elastic_config, nprocx=nprocx, nprocy=nprocy)
        generated = elastic_config
    else:
        generate_viscoelastic_sh_case(
            directory, config=config, nprocx=nprocx, nprocy=nprocy
        )
        generated = config

    sources = scenario.sources_for(role)
    source_lines = "".join(
        f"{x} 0.0 {y} 0.0 {generated.source_frequency_hz} 1.0 0.0 1\n"
        for x, y in sources
    )
    (directory / "source.dat").write_text(
        f"{len(sources)}\n" + source_lines, encoding="ascii"
    )
    parameter_path = directory / "denise.inp"
    parameters = parameter_path.read_text(encoding="ascii")
    if parameters.count("RUN_MULTIPLE_SHOTS =1") != 1:
        raise AssertionError("Expected exactly one RUN_MULTIPLE_SHOTS record")
    if parameters.count("FREE_SURF =0") != 1:
        raise AssertionError("Expected exactly one FREE_SURF record")
    parameters = parameters.replace(
        "RUN_MULTIPLE_SHOTS =1", "RUN_MULTIPLE_SHOTS =0"
    ).replace("FREE_SURF =0", f"FREE_SURF ={int(role == 'candidate')}")
    parameter_path.write_text(parameters, encoding="ascii")

    metadata = scenario.metadata(role) | {
        "elastic": elastic,
        "nprocx": nprocx,
        "nprocy": nprocy,
        "model_sha256": {
            path.suffix[1:]: _sha256(path)
            for path in sorted((directory / "model").glob("homogeneous.*"))
        },
    }
    (directory / "case.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return generated
