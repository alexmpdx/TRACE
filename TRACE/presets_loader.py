"""Load pipeline-config presets from TRACE/presets/*.json.

Each JSON file in `TRACE/presets/` defines one preset; the file's stem becomes
the preset name (e.g. `length-based.json` → "length-based"). The JSON contents
are a partial PipelineConfig override dict — only fields you care about — and
unknown fields are passed through to `dataclasses.replace` as-is.

Enum-list fields (`prune_methods`, `skeleton_methods`) are stored as their string
values in JSON and converted back to enum instances here.

Adding a new preset is just dropping a new `<name>.json` into the folder; no
code edits needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from TRACE.config_io import _ENUM_FIELDS, _migrate_boundary_smooth

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _convert_enum_lists(data: dict[str, Any]) -> dict[str, Any]:
    data = _migrate_boundary_smooth(data)
    out: dict[str, Any] = {}
    for key, val in data.items():
        if key in _ENUM_FIELDS and val is not None:
            enum_cls = _ENUM_FIELDS[key]
            out[key] = [enum_cls(item) for item in val]
        else:
            out[key] = val
    return out


def load_presets() -> dict[str, dict[str, Any]]:
    """Read every *.json in PRESETS_DIR. Returns {preset_name: override_dict}."""
    presets: dict[str, dict[str, Any]] = {}
    if not PRESETS_DIR.is_dir():
        return presets
    for f in sorted(PRESETS_DIR.glob("*.json")):
        if f.name.startswith("."):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        presets[f.stem] = _convert_enum_lists(data)
    return presets
