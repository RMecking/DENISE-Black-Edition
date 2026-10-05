from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: str | Path, image: np.ndarray) -> None:
    """Write a deterministic RGB PNG using only NumPy and the standard library."""

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"PNG image must have shape (height,width,3), got {array.shape}")
    height, width, _ = array.shape
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _palette(values: np.ndarray, diverging: bool) -> np.ndarray:
    stops = (
        np.array([[49, 54, 149], [69, 117, 180], [255, 255, 191], [244, 109, 67], [165, 0, 38]], dtype=float)
        if diverging
        else np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], dtype=float)
    )
    scaled = np.clip(values, 0.0, 1.0) * (len(stops) - 1)
    lower = np.floor(scaled).astype(int)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (scaled - lower)[..., None]
    return np.asarray(stops[lower] * (1.0 - fraction) + stops[upper] * fraction, dtype=np.uint8)


def heatmap(values: np.ndarray, *, vmin: float, vmax: float, diverging: bool = False) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if vmax <= vmin:
        normalized = np.zeros_like(data)
    else:
        normalized = (data - vmin) / (vmax - vmin)
    return _palette(normalized, diverging)


def hstack(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    height = max(image.shape[0] for image in images)
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    x = 0
    for image in images:
        canvas[: image.shape[0], x : x + image.shape[1]] = image
        x += image.shape[1] + gap
    return canvas


def draw_points(image: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int], radius: int = 2) -> None:
    height, width, _ = image.shape
    for x, y in points:
        cx, cy = int(round(x)), int(round(y))
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        image[y0:y1, x0:x1] = color


def line_plot(series: list[np.ndarray], *, width: int = 1000, height: int = 420) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    colors = ((31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14))
    finite = np.concatenate([np.asarray(values, dtype=float)[np.isfinite(values)] for values in series])
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    scale = scale or 1.0
    canvas[height // 2, :] = (210, 210, 210)
    for number, values in enumerate(series):
        data = np.asarray(values, dtype=float)
        if data.size < 2:
            continue
        xs = np.linspace(0, width - 1, data.size).astype(int)
        ys = np.clip(np.rint((height - 1) * (0.5 - 0.45 * data / scale)), 0, height - 1).astype(int)
        color = colors[number % len(colors)]
        for i in range(1, data.size):
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            steps = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
            xx = np.linspace(x0, x1, steps).astype(int)
            yy = np.linspace(y0, y1, steps).astype(int)
            canvas[yy, xx] = color
    return canvas
