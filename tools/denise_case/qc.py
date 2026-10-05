from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import CaseConfig
from .coordinates import coordinate_contract, pixel_coordinate
from .geometry import read_receivers, read_sources
from .model_io import read_grid
from .png import draw_points, heatmap, hstack, line_plot, write_png
from .seismic import SUTrace, gather, read_su, residual


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def model_qc(config: CaseConfig) -> dict[str, Any]:
    nx, ny = int(config.require("grid.nx")), int(config.require("grid.ny"))
    output = config.root / "qc" / "models"
    result: dict[str, Any] = {"orientation": "row 0 is shallow; depth increases downward; no stored-data flip", "figures": {}}
    for component in ("vp", "vs", "rho"):
        true = read_grid(config.root / "input" / "models" / f"true.{component}", nx, ny)
        start = read_grid(config.root / "input" / "models" / f"start_1d.{component}", nx, ny)
        smooth = read_grid(config.root / "input" / "models" / f"smooth2.{component}", nx, ny)
        difference = true - start
        lo = float(min(true.min(), start.min(), smooth.min()))
        hi = float(max(true.max(), start.max(), smooth.max()))
        delta = float(np.max(np.abs(difference))) or 1.0
        panels = [
            heatmap(true, vmin=lo, vmax=hi), heatmap(start, vmin=lo, vmax=hi),
            heatmap(smooth, vmin=lo, vmax=hi), heatmap(difference, vmin=-delta, vmax=delta, diverging=True),
        ]
        path = output / f"{component}_true_start_smooth_difference.png"
        write_png(path, hstack(panels))
        result["figures"][component] = {
            "path": str(path.relative_to(config.root)),
            "panel_order": ["true", "provided start_1d", "provided smooth2", "true - start_1d"],
            "shared_model_scale": [lo, hi], "difference_scale": [-delta, delta],
            "difference_stats": {"min": float(difference.min()), "max": float(difference.max()), "rms": float(np.sqrt(np.mean(difference ** 2)))},
        }
    _write_json(output / "model_qc.json", result)
    return result


def geometry_qc(config: CaseConfig) -> dict[str, Any]:
    nx, ny, dh = int(config.require("grid.nx")), int(config.require("grid.ny")), float(config.require("grid.dh_m"))
    vp = read_grid(config.root / "input" / "models" / "true.vp", nx, ny)
    image = heatmap(vp, vmin=float(vp.min()), vmax=float(vp.max()))
    sources = read_sources(config.root / "input" / "geometry" / "sources.dat")
    receivers = read_receivers(config.root / "input" / "geometry" / "receivers.dat")
    draw_points(image, [pixel_coordinate(s.x_m, s.y_m, dh) for s in sources], (255, 0, 0), 2)
    draw_points(image, [pixel_coordinate(r.x_m, r.y_m, dh) for r in receivers], (255, 255, 255), 1)
    output = config.root / "qc" / "geometry"
    path = output / "acquisition_over_true_vp.png"
    write_png(path, image)
    result = {
        "figure": str(path.relative_to(config.root)), "background_scale_m_s": [float(vp.min()), float(vp.max())],
        "source_color": "red", "receiver_color": "white", "source_count": len(sources), "receiver_count": len(receivers),
        "source_spacing_m": sorted(set(np.diff([s.x_m for s in sources]).tolist())),
        "receiver_spacing_m": sorted(set(np.diff([r.x_m for r in receivers]).tolist())),
        "coordinates": coordinate_contract(nx, ny, dh),
    }
    _write_json(output / "geometry_qc.json", result)
    return result


def amplitude_statistics(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    return {
        "min": float(data.min()), "max": float(data.max()), "mean": float(data.mean()),
        "rms": float(np.sqrt(np.mean(data ** 2))), "abs_p99": float(np.percentile(np.abs(data), 99.0)),
    }


def compare_gathers(observed: np.ndarray, synthetic: np.ndarray | None = None) -> dict[str, np.ndarray]:
    result = {"observed": np.asarray(observed)}
    if synthetic is not None:
        result["synthetic"] = np.asarray(synthetic)
        result["residual"] = residual(result["synthetic"], result["observed"])
    return result


def _load_component(
    run: Path, stem: str, component: str, dt_s: float,
    shot_lookup: dict[tuple[float, float], int],
    receiver_lookup: dict[tuple[float, float], int],
) -> list[SUTrace]:
    files = sorted(run.glob(f"data/{stem}_{component}.su.shot*"))
    if not files and (run / "data" / f"{stem}_{component}.su").exists():
        files = [run / "data" / f"{stem}_{component}.su"]
    traces: list[SUTrace] = []
    for path in files:
        traces.extend(
            read_su(
                path, component=component, fallback_dt_s=dt_s,
                shot_lookup=shot_lookup, receiver_lookup=receiver_lookup,
            )
        )
    return traces


def data_qc(config: CaseConfig, run_name: str = "true_forward") -> dict[str, Any]:
    run = config.root / "generated" / run_name
    output = config.root / "qc" / "data"
    dt_s = float(config.require("time.dt_s")) * int(config.get("time.sample_stride", 1))
    selected_shots = [int(value) for value in config.get("qc.selected_shots", [1])]
    selected_receivers = [int(value) for value in config.get("qc.selected_receivers", [1, 200, 400])]
    sources = read_sources(config.root / "input" / "geometry" / "sources.dat")
    receivers = read_receivers(config.root / "input" / "geometry" / "receivers.dat")
    shot_lookup = {(source.x_m, source.y_m): source.ordinal for source in sources}
    receiver_lookup = {(receiver.x_m, receiver.y_m): receiver.ordinal for receiver in receivers}
    expected_samples = int(round(float(config.require("time.time_s")) / dt_s))
    result: dict[str, Any] = {"run": str(run), "components": {}, "display_policy": "linear amplitude; symmetric 99th-percentile clipping for gather images only; underlying samples unchanged"}
    for component in ("vx", "vy"):
        traces = _load_component(run, "observed", component, dt_s, shot_lookup, receiver_lookup)
        if not traces:
            result["components"][component] = {"status": "missing", "expected": f"data/observed_{component}.su.shot*"}
            continue
        shot_ids = sorted(set(t.shot_id for t in traces))
        per_shot = {shot: sum(trace.shot_id == shot for trace in traces) for shot in shot_ids}
        finite = all(np.isfinite(trace.samples).all() for trace in traces)
        complete = (
            shot_ids == list(range(1, len(sources) + 1))
            and set(per_shot.values()) == {len(receivers)}
            and {trace.samples.size for trace in traces} == {expected_samples}
            and {trace.dt_s for trace in traces} == {dt_s}
            and finite
        )
        component_result: dict[str, Any] = {
            "trace_count": len(traces), "shot_ids": shot_ids, "shots": {},
            "identity_sources": {
                "shot": sorted(set(trace.shot_identity_source for trace in traces)),
                "receiver": sorted(set(trace.receiver_identity_source for trace in traces)),
            },
            "completeness": {
                "complete": complete, "expected_shots": len(sources), "actual_shots": len(shot_ids),
                "expected_receivers_per_shot": len(receivers), "receiver_counts": sorted(set(per_shot.values())),
                "expected_samples_per_trace": expected_samples,
                "sample_counts": sorted(set(trace.samples.size for trace in traces)),
                "sample_intervals_s": sorted(set(trace.dt_s for trace in traces)), "all_finite": finite,
            },
        }
        for shot_id in selected_shots:
            matrix, ordinals, actual_dt = gather(traces, shot_id)
            clip = float(np.percentile(np.abs(matrix), 99.0)) or 1.0
            gather_path = output / f"observed_{component}_shot{shot_id}_gather.png"
            write_png(gather_path, heatmap(matrix.T, vmin=-clip, vmax=clip, diverging=True))
            overlays = [matrix[ordinals.index(ordinal)] for ordinal in selected_receivers if ordinal in ordinals]
            overlay_path = output / f"observed_{component}_shot{shot_id}_trace_overlay.png"
            write_png(overlay_path, line_plot(overlays))
            component_result["shots"][str(shot_id)] = {
                "receivers": len(ordinals), "samples": int(matrix.shape[1]), "dt_s": actual_dt,
                "amplitude": amplitude_statistics(matrix), "display_clip_abs_p99": clip,
                "gather_figure": str(gather_path.relative_to(config.root)),
                "trace_overlay_figure": str(overlay_path.relative_to(config.root)),
                "overlay_receiver_ordinals": [ordinal for ordinal in selected_receivers if ordinal in ordinals],
            }
        result["components"][component] = component_result
    wavelet_files = sorted(run.glob("wavelet/*")) + sorted(run.glob("model/*source_signal*.su.shot*"))
    result["wavelet"] = {"status": "not available"}
    for path in wavelet_files:
        if not path.is_file() or path.stat().st_size < 244:
            continue
        try:
            traces = read_su(path, component="wavelet", fallback_dt_s=dt_s)
        except ValueError:
            continue
        if not traces:
            continue
        values = traces[0].samples
        wavelet_path = output / "source_wavelet.png"
        spectrum_path = output / "source_wavelet_spectrum.png"
        write_png(wavelet_path, line_plot([values]))
        spectrum = np.abs(np.fft.rfft(values))
        write_png(spectrum_path, line_plot([spectrum]))
        result["wavelet"] = {
            "status": "read from DENISE output", "source": str(path.relative_to(run)),
            "figure": str(wavelet_path.relative_to(config.root)), "spectrum_figure": str(spectrum_path.relative_to(config.root)),
            "amplitude": amplitude_statistics(values),
        }
        break
    _write_json(output / "data_qc.json", result)
    return result
