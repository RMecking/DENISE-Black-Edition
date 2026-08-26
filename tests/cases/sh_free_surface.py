from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

from tests.cases.homogeneous_sh import HomogeneousSHConfig, generate_case as generate_homogeneous
from tests.utilities.sh_free_surface import (
    arrival_tolerance,
    denise_grid_index,
    image_path_distance,
    image_source,
    native_vz_position,
    numerical_dispersion,
    ricker_f95,
    surface_candidate_times,
)


GeometryName = Literal["normal", "oblique"]
Role = Literal["free_surface", "absorbing", "calibration"]


@dataclass(frozen=True)
class SHFreeSurfaceScenario:
    name: GeometryName
    free_surface: HomogeneousSHConfig
    absorbing: HomogeneousSHConfig
    calibration: HomogeneousSHConfig
    surface_y_m: float = 0.0
    expected_vz_reflection_coefficient: float = 1.0
    reflection_window_half_width_s: float = 0.075

    def config_for(self, role: Role) -> HomogeneousSHConfig:
        return getattr(self, role)

    def metadata(self) -> dict[str, object]:
        config = self.free_surface
        source_input = (config.source_x_m, config.source_y_m)
        receiver_input = (config.receiver_x_m[0], config.receiver_y_m)
        source_index = tuple(denise_grid_index(value, config.dh_m) for value in source_input)
        receiver_index = tuple(denise_grid_index(value, config.dh_m) for value in receiver_input)
        source_native = native_vz_position(source_input, config.dh_m)
        receiver_native = native_vz_position(receiver_input, config.dh_m)
        image = image_source(source_native, self.surface_y_m)
        reflection_distance = image_path_distance(
            source_native, receiver_native, surface_y_m=self.surface_y_m
        )

        calibration_source_input = (
            self.calibration.source_x_m, self.calibration.source_y_m
        )
        calibration_receiver_input = (
            self.calibration.receiver_x_m[0], self.calibration.receiver_y_m
        )
        calibration_source_native = native_vz_position(
            calibration_source_input, config.dh_m
        )
        calibration_receiver_native = native_vz_position(
            calibration_receiver_input, config.dh_m
        )
        calibration_distance = math.dist(
            calibration_source_native, calibration_receiver_native
        )
        angle = math.atan2(
            receiver_native[1] - image[1], receiver_native[0] - image[0]
        )
        calibration_angle = math.atan2(
            calibration_receiver_native[1] - calibration_source_native[1],
            calibration_receiver_native[0] - calibration_source_native[0],
        )
        f95 = ricker_f95(config.source_frequency_hz)
        reflection_dispersion = numerical_dispersion(
            distance_m=reflection_distance,
            angle_rad=angle,
            frequency_hz=f95,
            vs_m_s=config.vs_m_s,
            dt_s=config.dt_s,
            dh_m=config.dh_m,
            fd_order=config.fd_order,
            max_relative_error=config.max_relative_error,
        )
        calibration_dispersion = numerical_dispersion(
            distance_m=calibration_distance,
            angle_rad=calibration_angle,
            frequency_hz=f95,
            vs_m_s=config.vs_m_s,
            dt_s=config.dt_s,
            dh_m=config.dh_m,
            fd_order=config.fd_order,
            max_relative_error=config.max_relative_error,
        )
        differential_dispersion = (
            reflection_dispersion.delay_error_s - calibration_dispersion.delay_error_s
        )
        tolerance = arrival_tolerance(
            dt_s=config.dt_s,
            reference_distance_m=reflection_distance,
            calibration_distance_m=calibration_distance,
            vs_m_s=config.vs_m_s,
            differential_dispersion_s=differential_dispersion,
        )
        candidate_times = surface_candidate_times(
            source_native,
            receiver_native,
            dh_m=config.dh_m,
            vs_m_s=config.vs_m_s,
        )
        source_peak_s = 1.5 / config.source_frequency_hz
        direct_distance = math.dist(source_native, receiver_native)

        return {
            "name": self.name,
            "surface_y_m": self.surface_y_m,
            "nominal_input_coordinates_m": {
                "source": list(source_input),
                "receiver": list(receiver_input),
                "calibration_source": list(calibration_source_input),
                "calibration_receiver": list(calibration_receiver_input),
            },
            "rounded_grid_indices": {
                "source": list(source_index),
                "receiver": list(receiver_index),
            },
            "native_vz_coordinates_m": {
                "source": list(source_native),
                "receiver": list(receiver_native),
                "calibration_source": list(calibration_source_native),
                "calibration_receiver": list(calibration_receiver_native),
            },
            "image_source_m": list(image),
            "image_distance_m": reflection_distance,
            "calibration_distance_m": calibration_distance,
            "direct_distance_m": direct_distance,
            "ray_angle_rad": angle,
            "calibration_ray_angle_rad": calibration_angle,
            "expected_vz_reflection_coefficient": self.expected_vz_reflection_coefficient,
            "source_peak_s": source_peak_s,
            "expected_direct_peak_s": source_peak_s + direct_distance / config.vs_m_s,
            "expected_reflection_peak_s": source_peak_s + candidate_times.y0_s,
            "expected_calibration_peak_s": source_peak_s + calibration_distance / config.vs_m_s,
            "surface_candidate_propagation_times_s": asdict(candidate_times),
            "timing_tolerance_s": tolerance,
            "surface_location_half_minimum_separation_s": (
                candidate_times.half_minimum_separation_s
            ),
            "surface_location_resolved": (
                tolerance < candidate_times.half_minimum_separation_s
            ),
            "source_spectrum": {
                "source_frequency_hz": config.source_frequency_hz,
                "f95_hz": f95,
                "definition": "95% energy of analytic DENISE QUELLART=1 Ricker spectrum",
            },
            "dispersion": {
                "symbol": "2*sum(c_m*sin((m-0.5)*xi))",
                "reflection": asdict(reflection_dispersion),
                "calibration": asdict(calibration_dispersion),
                "differential_delay_error_s": differential_dispersion,
            },
            "numerics": {
                "dt_s": config.dt_s,
                "dh_m": config.dh_m,
                "vs_m_s": config.vs_m_s,
                "density_kg_m3": config.density_kg_m3,
                "fd_order": config.fd_order,
                "max_relative_error": config.max_relative_error,
            },
            "acceptance": {
                "signed_amplitude_error_max": 0.05,
                "normalized_correlation_min": 0.99,
                "absorbing_l2_ratio_max": 0.10,
                "mpi_relative_l2_max": 1.0e-6,
                "mpi_correlation_min": 0.999999,
            },
        }


def normal_scenario(fd_order: int = 4) -> SHFreeSurfaceScenario:
    free = HomogeneousSHConfig(
        nx=240,
        ny=240,
        time_s=1.30,
        dt_s=0.0005,
        source_x_m=1200.0,
        source_y_m=700.0,
        receiver_x_m=(1200.0,),
        receiver_y_m=1000.0,
        source_frequency_hz=8.0,
        fd_order=fd_order,
    )
    calibration = replace(
        free,
        ny=360,
        source_y_m=500.0,
        receiver_y_m=2190.0,
    )
    return SHFreeSurfaceScenario(
        name="normal",
        free_surface=free,
        absorbing=free,
        calibration=calibration,
    )


def oblique_scenario(fd_order: int = 4) -> SHFreeSurfaceScenario:
    free = HomogeneousSHConfig(
        nx=240,
        ny=240,
        time_s=1.30,
        dt_s=0.0005,
        source_x_m=900.0,
        source_y_m=700.0,
        receiver_x_m=(1500.0,),
        receiver_y_m=900.0,
        source_frequency_hz=8.0,
        fd_order=fd_order,
    )
    calibration = replace(
        free,
        ny=360,
        source_y_m=500.0,
        receiver_y_m=2090.0,
    )
    return SHFreeSurfaceScenario(
        name="oblique",
        free_surface=free,
        absorbing=free,
        calibration=calibration,
    )


def generate_case(
    directory: Path,
    *,
    scenario: SHFreeSurfaceScenario,
    role: Role,
    nprocx: int = 1,
    nprocy: int = 1,
) -> HomogeneousSHConfig:
    config = scenario.config_for(role)
    generate_homogeneous(directory, config=config, nprocx=nprocx, nprocy=nprocy)
    free_surface = role == "free_surface"
    parameter_path = directory / "denise.inp"
    parameters = parameter_path.read_text(encoding="ascii")
    if parameters.count("FREE_SURF =0") != 1:
        raise AssertionError("Expected exactly one homogeneous FREE_SURF record")
    parameter_path.write_text(
        parameters.replace("FREE_SURF =0", f"FREE_SURF ={int(free_surface)}"),
        encoding="ascii",
    )
    metadata = scenario.metadata() | {
        "role": role,
        "free_surface": free_surface,
        "nprocx": nprocx,
        "nprocy": nprocy,
    }
    (directory / "case.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return config
