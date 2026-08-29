"""JSON serialization for identifyFeatures PipelineConfig.

Used by both the GUI (QSettings persistence + Import/Export buttons) and
the CLI (--config flag) to round-trip the full PipelineConfig dataclass.

Unknown keys in input JSON are silently ignored so old saved configs keep
working when PipelineConfig gains new fields. Unknown *values* in enum-list
fields are dropped with a warning for the same reason: a stored config must
never be unloadable just because an enum member was renamed or retired.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path
from typing import Any

from identify_features.config import PipelineConfig
from identify_features.models.datatypes import PruneMethod, SkeletonMethod

logger = logging.getLogger(__name__)

# Dataclass field name -> enum class for enum-list fields.
_ENUM_FIELDS: dict[str, type] = {
    "skeleton_methods": SkeletonMethod,
    "prune_methods": PruneMethod,
}

# Sentinel: this field could not be coerced at all, so the caller should omit
# it and let PipelineConfig's own default stand.
_DROP = object()


def coerce_enum_list(enum_cls: type, value: Any, field_name: str, source: str = "config") -> Any:
    """Convert a list of enum *values* to enum members, dropping bad entries.

    ``enum_cls(item)`` raises ValueError on an unrecognized string, which used
    to make a single stale or typo'd entry blow up the whole load — and in
    ``load_presets`` that took down every preset, not just the bad one. Saved
    settings and presets are user data that outlives any given build, so an
    unknown value is treated as skew to be logged, not an error.

    Returns ``_DROP`` when nothing usable survives from a non-empty input, so
    the caller omits the key and PipelineConfig's default applies rather than
    an empty list — ``skeleton_methods: []`` silently means plain Zhang-Suen
    thinning, which is not what a user with one bad entry meant. An input that
    was *already* empty is preserved, since ``prune_methods: []`` is a
    legitimate, meaningful setting (it is what the length-based preset ships).
    """
    if value is None:
        return None
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        logger.warning(
            "%s: %r should be a list of %s values, got %r — ignoring", source, field_name, enum_cls.__name__, value
        )
        return _DROP

    out, bad = [], []
    for item in value:
        try:
            out.append(enum_cls(item))
        except (ValueError, KeyError):
            bad.append(item)

    if bad:
        valid = ", ".join(m.value for m in enum_cls)
        logger.warning(
            "%s: ignoring unrecognized %s value(s) %s — valid values are: %s",
            source,
            field_name,
            ", ".join(repr(b) for b in bad),
            valid,
        )

    if not out and value:
        logger.warning("%s: no usable %s values left; falling back to the built-in default", source, field_name)
        return _DROP
    return out


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
    data = _migrate_boundary_smooth(data)
    kwargs: dict[str, Any] = {}
    for key, val in data.items():
        if key not in known:
            continue
        if key in _ENUM_FIELDS:
            coerced = coerce_enum_list(_ENUM_FIELDS[key], val, key)
            if coerced is not _DROP:
                kwargs[key] = coerced
        else:
            kwargs[key] = val
    return PipelineConfig(**kwargs)


def _migrate_boundary_smooth(data: dict[str, Any]) -> dict[str, Any]:
    """Translate the retired ``"boundary-smooth"`` skeleton method to its flag.

    Boundary smoothing used to be a ``SkeletonMethod`` enum member even though
    it is mask preprocessing that composes with a skeletonizer rather than an
    alternative to one. It now lives on PipelineConfig as
    ``enable_boundary_smooth``. Configs saved before that change still carry the
    string in ``skeleton_methods``, where it would raise ValueError on the enum
    lookup — so strip it here and set the flag instead.

    An explicit ``enable_boundary_smooth`` already in the dict wins, so a config
    written by a current build round-trips untouched.
    """
    methods = data.get("skeleton_methods")
    if not isinstance(methods, list) or "boundary-smooth" not in methods:
        return data

    data = dict(data)
    data["skeleton_methods"] = [m for m in methods if m != "boundary-smooth"]
    data.setdefault("enable_boundary_smooth", True)
    return data


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


def save_settings(
    config: PipelineConfig,
    gate_override: dict | None,
    path: Path,
    gui_state: dict[str, Any] | None = None,
) -> None:
    """Write a PipelineConfig, landmark gate override, and full GUI state to JSON.

    The file is the PipelineConfig dict with extra top-level keys:
      - ``gate_override`` — the landmark gate-config override (omitted when None/empty)
      - ``gui_state`` — every GUI-only flag the main window exposes
        (Settings-tab toggles, model paths, custom distance pairs, etc.)
        so a saved preset round-trips the user's full configuration, not
        just the PipelineConfig portion.

    Old loaders that only know about PipelineConfig fields silently ignore
    the extra keys (see ``config_from_dict``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config_to_dict(config)
    if gate_override:
        data["gate_override"] = gate_override
    if gui_state:
        data["gui_state"] = gui_state
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def load_settings(path: Path) -> tuple[PipelineConfig, dict | None, dict | None]:
    """Read a PipelineConfig + gate override + GUI state from a JSON file.

    Returns ``(config, gate_override, gui_state)``. ``gate_override`` and
    ``gui_state`` are ``None`` when the file predates those fields or the
    keys are empty.
    """
    with open(path) as fh:
        data = json.load(fh)
    gate_override = data.get("gate_override") or None
    gui_state = data.get("gui_state") or None
    return config_from_dict(data), gate_override, gui_state


def config_to_json(config: PipelineConfig) -> str:
    """Serialize a PipelineConfig to a compact JSON string (for QSettings)."""
    return json.dumps(config_to_dict(config))


def config_from_json(text: str) -> PipelineConfig:
    """Parse a PipelineConfig from a JSON string."""
    return config_from_dict(json.loads(text))
