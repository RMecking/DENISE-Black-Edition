from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tools.denise_case.config import ConfigError, load_case
from tools.denise_case.generate import materialize_run
from tools.denise_case.geometry import GeometryError, geometry_issues, read_receivers, read_sources
from tools.denise_case.model_io import ModelIOError, read_grid, write_grid
from tools.denise_case.qc import compare_gathers, data_qc, geometry_qc, model_qc
from tools.denise_case.report import build_report
from tools.denise_case.seismic import gather, read_su
from tools.denise_case.validation import validate_case


def _manifest(case_root: Path, *, nx: int = 4, ny: int = 3) -> Path:
    data = {
        "case": {"id": "fixture"},
        "provenance": {
            "denise_core_sha": "deadbeef", "benchmark_sha": "benchmark",
            "benchmark_checkout": "benchmark",
        },
        "backend": {
            "executable": "bin/denise", "launcher": "mpiexec", "environment": "test",
            "ranks": 1, "nprocx": 1, "nprocy": 1,
        },
        "grid": {"nx": nx, "ny": ny, "dh_m": 10.0},
        "time": {"dt_s": 0.001, "time_s": 0.01, "sample_stride": 1},
        "physics": {"fd_order": 8, "max_relative_error": 1},
        "boundary": {
            "free_surface": False, "absorbing_width_gridpoints": 0,
            "damping_velocity_m_s": 1500.0, "pml_frequency_hz": 10.0,
        },
        "models": {
            "true": {"prefix": "models/true"}, "start_1d": {"prefix": "models/start"},
            "smooth2": {"prefix": "models/smooth"},
        },
        "acquisition": {
            "sources": "source.dat", "receivers": "receiver.dat", "require_grid_alignment": True,
        },
        "wavelet": {"type": 6, "highpass_hz": -5.0, "lowpass_hz": 15.0, "filter_order": 5},
        "outputs": {"forward_data": "generated/true_forward/data"},
        "qc": {"selected_shots": [7], "selected_receivers": [1, 2]},
    }
    path = case_root / "case.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _inputs(case_root: Path, nx: int = 4, ny: int = 3) -> None:
    base = np.arange(nx * ny, dtype=np.float32).reshape(ny, nx)
    for name, offset in (("true", 1500.0), ("start_1d", 1400.0), ("smooth2", 1450.0)):
        for component, extra in (("vp", 0.0), ("vs", -1000.0), ("rho", 500.0)):
            values = base + offset + extra
            write_grid(case_root / "input" / "models" / f"{name}.{component}", values)
    geometry = case_root / "input" / "geometry"
    geometry.mkdir(parents=True)
    (geometry / "sources.dat").write_text("1\n10 0 10 0 10 1 0 1\n", encoding="ascii")
    (geometry / "receivers.dat").write_text("10 20\n20 20\n", encoding="ascii")


def _su_trace(*, shot: int, receiver: int, samples: list[float], dt_us: int = 1000) -> bytes:
    header = bytearray(240)
    struct.pack_into("<i", header, 0, receiver)
    struct.pack_into("<i", header, 8, shot)
    struct.pack_into("<i", header, 12, receiver)
    struct.pack_into("<h", header, 70, -1)
    struct.pack_into("<i", header, 72, 10)
    struct.pack_into("<i", header, 76, 20)
    struct.pack_into("<i", header, 80, receiver * 10)
    struct.pack_into("<i", header, 84, 20)
    struct.pack_into("<H", header, 114, len(samples))
    struct.pack_into("<H", header, 116, dt_us)
    return bytes(header) + np.asarray(samples, dtype="<f4").tobytes()


def test_model_io_roundtrip_is_byte_exact_and_orientation_explicit(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = tmp_path / "model.vp"
    write_grid(path, values)
    assert path.read_bytes() == np.asarray(values.T, dtype="<f4", order="C").tobytes()
    np.testing.assert_array_equal(read_grid(path, 4, 3), values)
    assert read_grid(path, 4, 3)[0, 0] == 0.0
    assert read_grid(path, 4, 3)[-1, -1] == 11.0


def test_model_io_rejects_invalid_size(tmp_path):
    path = tmp_path / "bad.vp"
    path.write_bytes(b"\0" * 12)
    with pytest.raises(ModelIOError, match="expected 48"):
        read_grid(path, 4, 3)


def test_model_writer_rejects_nonfinite_and_overwrite(tmp_path):
    path = tmp_path / "model.vp"
    with pytest.raises(ModelIOError, match="NaN"):
        write_grid(path, np.array([[np.nan]], dtype=np.float32))
    write_grid(path, np.ones((2, 2)))
    with pytest.raises(FileExistsError):
        write_grid(path, np.ones((2, 2)))


def test_geometry_parsing_and_declared_count(tmp_path):
    source = tmp_path / "source.dat"
    receiver = tmp_path / "receiver.dat"
    source.write_text("2\n10 0 20 0 8 1 0 1\n30 0 20 0 8 1 0 1\n", encoding="ascii")
    receiver.write_text("10 20\n30 20\n", encoding="ascii")
    assert [item.x_m for item in read_sources(source)] == [10.0, 30.0]
    assert [item.ordinal for item in read_receivers(receiver)] == [1, 2]
    source.write_text("3\n10 0 20 0 8 1 0 1\n", encoding="ascii")
    with pytest.raises(GeometryError, match="declares 3"):
        read_sources(source)


def test_geometry_bounds_alignment_and_duplicates(tmp_path):
    source = tmp_path / "source.dat"
    receiver = tmp_path / "receiver.dat"
    source.write_text("1\n10 0 20 0 8 1 0 1\n", encoding="ascii")
    receiver.write_text("10 20\n10 20\n31 20\n", encoding="ascii")
    errors, _ = geometry_issues(
        read_sources(source), read_receivers(receiver), nx=4, ny=3, dh_m=10,
        absorbing_width=0, free_surface=False,
    )
    assert any("duplicate receiver" in item for item in errors)
    assert any("not aligned" in item for item in errors)


def test_manifest_parsing_and_required_fields(tmp_path):
    path = _manifest(tmp_path)
    case = load_case(path)
    assert case.require("grid.nx") == 4
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="case.id"):
        load_case(path)


def test_generated_run_is_deterministic(tmp_path):
    case_root = tmp_path / "applications" / "fixture"
    config = load_case(_manifest(case_root))
    _inputs(case_root)
    first = materialize_run(config)
    before = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    second = materialize_run(config)
    after = {path.relative_to(second): path.read_bytes() for path in second.rglob("*") if path.is_file()}
    assert before == after
    records = [line for line in (first / "denise.inp").read_text().splitlines() if line and not line.startswith("#")]
    assert len(records) == 115


def test_generated_run_refuses_nonidentical_existing_file(tmp_path):
    case_root = tmp_path / "applications" / "fixture"
    config = load_case(_manifest(case_root))
    _inputs(case_root)
    run = materialize_run(config)
    (run / "workflow.inp").write_text("changed\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="non-identical"):
        materialize_run(config)


def test_validation_failure_for_bad_model_and_geometry(tmp_path):
    case_root = tmp_path / "applications" / "fixture"
    config = load_case(_manifest(case_root))
    _inputs(case_root)
    (case_root / "input" / "models" / "true.vp").write_bytes(b"bad")
    (case_root / "input" / "geometry" / "receivers.dat").write_text("999 20\n", encoding="ascii")
    report = validate_case(config, require_executable=False)
    assert not report.ok
    assert any("invalid model size" in item for item in report.errors)
    assert any("outside" in item for item in report.errors)


def test_seismic_reader_preserves_header_identity_order_and_metadata(tmp_path):
    path = tmp_path / "observed_vx.su.shot7"
    path.write_bytes(
        _su_trace(shot=7, receiver=2, samples=[2.0, 3.0])
        + _su_trace(shot=7, receiver=1, samples=[0.0, 1.0])
    )
    traces = read_su(path, component="vx")
    assert [trace.shot_id for trace in traces] == [7, 7]
    assert traces[0].component == "vx"
    assert traces[0].dt_s == pytest.approx(0.001)
    assert traces[0].receiver_xy_m == (20.0, 20.0)
    matrix, ordinals, dt = gather(traces, 7)
    assert ordinals == [1, 2]
    np.testing.assert_array_equal(matrix, [[0.0, 1.0], [2.0, 3.0]])
    assert dt == pytest.approx(0.001)


def test_comparison_api_uses_synthetic_minus_observed():
    observed = np.array([[1.0, 2.0]])
    synthetic = np.array([[3.0, 1.0]])
    result = compare_gathers(observed, synthetic)
    np.testing.assert_array_equal(result["residual"], [[2.0, -1.0]])


def test_model_geometry_and_report_qc_smoke(tmp_path):
    case_root = tmp_path / "applications" / "fixture"
    config = load_case(_manifest(case_root))
    _inputs(case_root)
    materialize_run(config)
    models = model_qc(config)
    geometry = geometry_qc(config)
    assert all((case_root / details["path"]).read_bytes().startswith(b"\x89PNG") for details in models["figures"].values())
    assert (case_root / geometry["figure"]).read_bytes().startswith(b"\x89PNG")
    report = build_report(config)
    assert "Marmousi-II A0/A1" in report.read_text(encoding="utf-8")


def test_data_qc_smoke_on_small_su_fixture(tmp_path):
    case_root = tmp_path / "applications" / "fixture"
    config = load_case(_manifest(case_root))
    _inputs(case_root)
    run = materialize_run(config)
    blob = _su_trace(shot=7, receiver=1, samples=[0.0, 1.0, 0.0]) + _su_trace(shot=7, receiver=2, samples=[1.0, 0.0, -1.0])
    (run / "data" / "observed_vx.su.shot7").write_bytes(blob)
    (run / "data" / "observed_vy.su.shot7").write_bytes(blob)
    result = data_qc(config)
    assert result["components"]["vx"]["shots"]["7"]["receivers"] == 2
    figure = case_root / result["components"]["vy"]["shots"]["7"]["gather_figure"]
    assert figure.read_bytes().startswith(b"\x89PNG")
