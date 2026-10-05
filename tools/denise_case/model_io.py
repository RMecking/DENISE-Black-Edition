from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class ModelIOError(ValueError):
    """Raised for an invalid DENISE float-grid contract."""


@dataclass(frozen=True)
class GridIdentity:
    path: str
    sha256: str
    bytes: int
    nx: int
    ny: int
    dtype: str
    disk_order: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_grid(path: str | Path, nx: int, ny: int, endian: str = "little") -> np.ndarray:
    """Read a DENISE grid as ``[depth_y, horizontal_x]`` without display flips.

    DENISE stores FP32 values x-major: all y samples for x=1, then all y
    samples for x=2.  The returned scientific array has depth increasing with
    row index and is therefore displayed with ``origin='upper'``.
    """

    source = Path(path)
    expected = nx * ny * 4
    actual = source.stat().st_size
    if actual != expected:
        raise ModelIOError(
            f"invalid model size for {source}: {actual} bytes, expected {expected} ({nx}x{ny} FP32)"
        )
    code = "<f4" if endian == "little" else ">f4"
    raw = np.fromfile(source, dtype=code)
    return raw.reshape(nx, ny).T.copy()


def write_grid(
    path: str | Path, values: np.ndarray, *, endian: str = "little", overwrite: bool = False
) -> GridIdentity:
    target = Path(path)
    array = np.asarray(values)
    if array.ndim != 2:
        raise ModelIOError(f"DENISE model must be 2-D, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ModelIOError("DENISE model contains NaN or infinite values")
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite model: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    code = "<f4" if endian == "little" else ">f4"
    disk = np.asarray(array.T, dtype=code, order="C")
    disk.tofile(target)
    ny, nx = array.shape
    return grid_identity(target, nx, ny, endian)


def grid_identity(path: str | Path, nx: int, ny: int, endian: str = "little") -> GridIdentity:
    source = Path(path)
    return GridIdentity(
        path=str(source),
        sha256=sha256_file(source),
        bytes=source.stat().st_size,
        nx=nx,
        ny=ny,
        dtype=f"{endian}-endian IEEE-754 FP32",
        disk_order="x-major, y-fastest",
    )
