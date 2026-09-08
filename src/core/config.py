from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the project YAML configuration from one central location."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return data


def with_mm_per_pixel(config: dict[str, Any], value: float) -> dict[str, Any]:
    """Return a copy with the legacy-MAT scale fallback updated."""
    if value <= 0:
        raise ValueError("mm_per_pixel must be > 0")
    updated = deepcopy(config)
    updated.setdefault("experiment", {})["mm_per_pixel"] = float(value)
    return updated
