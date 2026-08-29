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
import logging
from pathlib import Path
from typing import Any

from TRACE.config_io import _DROP, _ENUM_FIELDS, _migrate_boundary_smooth, coerce_enum_list

logger = logging.getLogger(__name__)

PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _convert_enum_lists(data: dict[str, Any], source: str = "preset") -> dict[str, Any]:
    """Convert this preset's enum-list fields to enum members.

    Unrecognized values are dropped with a warning rather than raising: a
    preset is user data that outlives the build that wrote it, so a retired or
    misspelled enum value must not make it unloadable. A field left with
    nothing usable is omitted entirely so PipelineConfig's default applies.
    """
    data = _migrate_boundary_smooth(data)
    out: dict[str, Any] = {}
    for key, val in data.items():
        if key in _ENUM_FIELDS and val is not None:
            coerced = coerce_enum_list(_ENUM_FIELDS[key], val, key, source=source)
            if coerced is not _DROP:
                out[key] = coerced
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
            logger.warning("Skipping unreadable preset %s", f.name)
            continue
        if not isinstance(data, dict):
            logger.warning("Skipping preset %s: expected a JSON object, got %s", f.name, type(data).__name__)
            continue
        # Belt and braces: coerce_enum_list already downgrades bad enum values
        # to warnings, but conversion must not be able to take out the whole
        # preset listing over one malformed file — that would leave the GUI
        # with no presets at all rather than one missing entry.
        try:
            presets[f.stem] = _convert_enum_lists(data, source=f"preset {f.name}")
        except Exception:
            logger.exception("Skipping preset %s: could not be converted", f.name)
    return presets
