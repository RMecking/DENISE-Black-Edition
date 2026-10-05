from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class SeismicIOError(ValueError):
    """Raised when an SU file is incomplete or inconsistent."""


@dataclass(frozen=True)
class SUTrace:
    shot_id: int
    receiver_ordinal: int
    component: str
    dt_s: float
    samples: np.ndarray
    source_xy_m: tuple[float, float]
    receiver_xy_m: tuple[float, float]
    shot_identity_source: str
    receiver_identity_source: str
    header: dict[str, int]


def _i16(header: bytes, offset: int, endian: str) -> int:
    return struct.unpack_from(("<" if endian == "little" else ">") + "h", header, offset)[0]


def _u16(header: bytes, offset: int, endian: str) -> int:
    return struct.unpack_from(("<" if endian == "little" else ">") + "H", header, offset)[0]


def _i32(header: bytes, offset: int, endian: str) -> int:
    return struct.unpack_from(("<" if endian == "little" else ">") + "i", header, offset)[0]


def _scaled(value: int, scalar: int) -> float:
    if scalar > 0:
        return value * scalar
    if scalar < 0:
        return value / abs(scalar)
    return float(value)


def read_su(
    path: str | Path, *, component: str, endian: str = "little", fallback_dt_s: float | None = None,
    shot_lookup: dict[tuple[float, float], int] | None = None,
    receiver_lookup: dict[tuple[float, float], int] | None = None,
) -> list[SUTrace]:
    """Read header-preserving SU traces in chronological sample order."""

    source = Path(path)
    blob = source.read_bytes()
    offset = 0
    result: list[SUTrace] = []
    filename_shot = re.search(r"\.shot(\d+)(?:\.|$)", source.name)
    while offset < len(blob):
        if len(blob) - offset < 240:
            raise SeismicIOError(f"truncated SU header at byte {offset} in {source}")
        header = blob[offset : offset + 240]
        ns = _u16(header, 114, endian)
        dt_us = _u16(header, 116, endian)
        if ns <= 0:
            raise SeismicIOError(f"invalid ns={ns} at trace {len(result)+1} in {source}")
        payload_start = offset + 240
        payload_end = payload_start + ns * 4
        if payload_end > len(blob):
            raise SeismicIOError(f"truncated SU payload at trace {len(result)+1} in {source}")
        samples = np.frombuffer(blob[payload_start:payload_end], dtype="<f4" if endian == "little" else ">f4").copy()
        scalco = _i16(header, 70, endian)
        source_xy = (_scaled(_i32(header, 72, endian), scalco), _scaled(_i32(header, 76, endian), scalco))
        receiver_xy = (_scaled(_i32(header, 80, endian), scalco), _scaled(_i32(header, 84, endian), scalco))
        fldr = _i32(header, 8, endian)
        if fldr:
            shot_id, shot_identity_source = fldr, "SU.fldr"
        elif shot_lookup is not None and source_xy in shot_lookup:
            shot_id, shot_identity_source = shot_lookup[source_xy], "SU source coordinates + source geometry"
        elif filename_shot:
            shot_id, shot_identity_source = int(filename_shot.group(1)), "filename fallback"
        else:
            shot_id, shot_identity_source = 0, "unresolved"
        tracf, tracl = _i32(header, 12, endian), _i32(header, 0, endian)
        if tracf:
            receiver_ordinal, receiver_identity_source = tracf, "SU.tracf"
        elif tracl:
            receiver_ordinal, receiver_identity_source = tracl, "SU.tracl"
        elif receiver_lookup is not None and receiver_xy in receiver_lookup:
            receiver_ordinal, receiver_identity_source = receiver_lookup[receiver_xy], "SU receiver coordinates + receiver geometry"
        else:
            receiver_ordinal, receiver_identity_source = len(result) + 1, "file order fallback"
        dt_s = dt_us * 1.0e-6 if dt_us else fallback_dt_s
        if dt_s is None:
            raise SeismicIOError(f"trace {len(result)+1} has no sample interval in header")
        result.append(
            SUTrace(
                shot_id=shot_id,
                receiver_ordinal=receiver_ordinal,
                component=component,
                dt_s=float(dt_s),
                samples=samples,
                source_xy_m=source_xy,
                receiver_xy_m=receiver_xy,
                shot_identity_source=shot_identity_source,
                receiver_identity_source=receiver_identity_source,
                header={"fldr": fldr, "tracf": _i32(header, 12, endian), "ns": ns, "dt_us": dt_us},
            )
        )
        offset = payload_end
    return result


def gather(traces: list[SUTrace], shot_id: int) -> tuple[np.ndarray, list[int], float]:
    selected = sorted((trace for trace in traces if trace.shot_id == shot_id), key=lambda trace: trace.receiver_ordinal)
    if not selected:
        raise SeismicIOError(f"shot {shot_id} is absent")
    ns = {trace.samples.size for trace in selected}
    dts = {trace.dt_s for trace in selected}
    if len(ns) != 1 or len(dts) != 1:
        raise SeismicIOError(f"shot {shot_id} has inconsistent sample counts or intervals")
    return np.vstack([trace.samples for trace in selected]), [trace.receiver_ordinal for trace in selected], selected[0].dt_s


def residual(synthetic: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if synthetic.shape != observed.shape:
        raise SeismicIOError(f"cannot subtract gather shapes {synthetic.shape} and {observed.shape}")
    return np.asarray(synthetic) - np.asarray(observed)
