from __future__ import annotations

import json
import math

import pytest

from tests.utilities.attenuation import peak_absolute, root_mean_square, spectral_band_rms
from tests.cases.homogeneous_viscoelastic import (
    ViscoelasticPSVConfig,
    ViscoelasticSHConfig,
    generate_viscoelastic_psv_case,
    generate_viscoelastic_sh_case,
)


def test_basic_amplitude_metrics():
    trace = [0.0, -3.0, 4.0, 0.0]
    assert peak_absolute(trace) == 4.0
    assert root_mean_square(trace) == 2.5


def test_spectral_band_rms_identifies_tone():
    dt = 0.001
    trace = [math.sin(2.0 * math.pi * 10.0 * index * dt) for index in range(1000)]
    at_tone = spectral_band_rms(trace, dt_s=dt, lower_hz=9.0, upper_hz=11.0)
    away = spectral_band_rms(trace, dt_s=dt, lower_hz=19.0, upper_hz=21.0)
    assert at_tone > 100.0 * away


@pytest.mark.parametrize("trace", ([],))
def test_amplitude_metrics_reject_empty_trace(trace):
    with pytest.raises(ValueError):
        peak_absolute(trace)
    with pytest.raises(ValueError):
        root_mean_square(trace)


def test_viscoelastic_generators_write_q_models_and_relaxation_parameters(tmp_path):
    sh_directory = tmp_path / "sh"
    sh = generate_viscoelastic_sh_case(
        sh_directory,
        config=ViscoelasticSHConfig(qs=20.0, relaxation_frequencies_hz=(5.0, 20.0)),
    )
    assert (sh_directory / "model" / "homogeneous.qs").stat().st_size == sh.nx * sh.ny * 4
    parameters = (sh_directory / "denise.inp").read_text(encoding="ascii")
    assert "L =2" in parameters
    assert "FL =5.0 20.0" in parameters
    sh_metadata = json.loads((sh_directory / "case.json").read_text())
    assert sh_metadata["qs"] == 20.0
    assert len(sh_metadata["model_sha256"]["qs"]) == 64

    psv_directory = tmp_path / "psv"
    psv = generate_viscoelastic_psv_case(
        psv_directory, config=ViscoelasticPSVConfig(qp=20.0, qs=200.0)
    )
    assert (psv_directory / "model" / "homogeneous.qp").stat().st_size == psv.nx * psv.ny * 4
    assert (psv_directory / "model" / "homogeneous.qs").stat().st_size == psv.nx * psv.ny * 4


def test_viscoelastic_generators_reject_non_positive_q(tmp_path):
    with pytest.raises(ValueError, match="Qs"):
        generate_viscoelastic_sh_case(tmp_path / "sh", config=ViscoelasticSHConfig(qs=0.0))
    with pytest.raises(ValueError, match="Qp and Qs"):
        generate_viscoelastic_psv_case(
            tmp_path / "psv", config=ViscoelasticPSVConfig(qp=-1.0)
        )
