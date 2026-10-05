from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an application manifest is malformed."""


REQUIRED_PATHS = (
    "case.id",
    "provenance.denise_core_sha",
    "provenance.benchmark_sha",
    "backend.executable",
    "grid.nx",
    "grid.ny",
    "grid.dh_m",
    "time.dt_s",
    "time.time_s",
    "models.true.prefix",
    "models.start_1d.prefix",
    "models.smooth2.prefix",
    "acquisition.sources",
    "acquisition.receivers",
)


def _lookup(data: dict[str, Any], dotted: str) -> Any:
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"manifest is missing required field: {dotted}")
        value = value[part]
    return value


@dataclass(frozen=True)
class CaseConfig:
    """Loaded high-level case manifest with path-resolution helpers."""

    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent

    def get(self, dotted: str, default: Any = None) -> Any:
        try:
            return _lookup(self.data, dotted)
        except ConfigError:
            return default

    def require(self, dotted: str) -> Any:
        return _lookup(self.data, dotted)

    def resolve(self, dotted: str) -> Path:
        value = Path(str(self.require(dotted)))
        return value if value.is_absolute() else (self.root / value).resolve()

    def canonical_bytes(self) -> bytes:
        return (json.dumps(self.data, sort_keys=True, separators=(",", ":")) + "\n").encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def load_case(path: str | Path) -> CaseConfig:
    """Load a JSON-form YAML 1.2 manifest without an optional YAML dependency.

    JSON is a strict subset of YAML 1.2.  Keeping the authoritative example in
    this subset makes the CLI dependency-light and deterministic.
    """

    manifest = Path(path).resolve()
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"manifest not found: {manifest}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"{manifest} must use the dependency-free JSON subset of YAML 1.2: {error}"
        ) from error
    if not isinstance(data, dict):
        raise ConfigError("manifest root must be a mapping")
    for dotted in REQUIRED_PATHS:
        _lookup(data, dotted)
    for dotted in ("grid.nx", "grid.ny"):
        if not isinstance(_lookup(data, dotted), int) or _lookup(data, dotted) <= 0:
            raise ConfigError(f"{dotted} must be a positive integer")
    for dotted in ("grid.dh_m", "time.dt_s", "time.time_s"):
        if not isinstance(_lookup(data, dotted), (int, float)) or _lookup(data, dotted) <= 0:
            raise ConfigError(f"{dotted} must be positive")
    return CaseConfig(manifest, data)


def dump_json_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
