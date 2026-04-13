"""JSON serialization for identifyFeatures PipelineConfig.

Used by both the GUI (QSettings persistence + Import/Export buttons) and
the CLI (--config flag) to round-trip the full PipelineConfig dataclass.

Unknown keys in input JSON are silently ignored so old saved configs keep
working when PipelineConfig gains new fields.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from identify_features.config import PipelineConfig
from identify_features.models.datatypes import PruneMethod, SkeletonMethod

# Dataclass field name -> enum class for enum-list fields.
_ENUM_FIELDS: dict[str, type] = {
    "skeleton_methods": SkeletonMethod,
    "prune_methods": PruneMethod,
}


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """Convert a PipelineConfig to a JSON-serializable dict."""
    out: dict[str, Any] = {}
    for f in fields(config):
        val = getattr(config, f.name)
        if f.name in _ENUM_FIELDS:
            out[f.name] = [e.value for e in val]
        else:
            out[f.name] = val
    return out


def config_from_dict(data: dict[str, Any]) -> PipelineConfig:
    """Build a PipelineConfig from a dict. Unknown keys are ignored."""
    known = {f.name for f in fields(PipelineConfig)}
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        if key not in known:
            continue
        if key in _ENUM_FIELDS:
            enum_cls = _ENUM_FIELDS[key]
            kwargs[key] = [enum_cls(v) for v in val]
        else:
            kwargs[key] = val
    return PipelineConfig(**kwargs)


def load_config(path: Path) -> PipelineConfig:
    """Read a PipelineConfig from a JSON file."""
    with open(path) as fh:
        data = json.load(fh)
    return config_from_dict(data)


def save_config(config: PipelineConfig, path: Path) -> None:
    """Write a PipelineConfig to a JSON file (pretty-printed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(config_to_dict(config), fh, indent=2)


def config_to_json(config: PipelineConfig) -> str:
    """Serialize a PipelineConfig to a compact JSON string (for QSettings)."""
    return json.dumps(config_to_dict(config))


def config_from_json(text: str) -> PipelineConfig:
    """Parse a PipelineConfig from a JSON string."""
    return config_from_dict(json.loads(text))
