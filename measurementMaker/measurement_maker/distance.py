"""Read landmark GeoJSONs and compute pair distances."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_landmarks_from_geojson(path: Path) -> dict[str, tuple[float, float]]:
    """Read a *_landmarks.geojson and return {raw_name: (x, y)}.

    Uses the raw `properties.classification.name` as the key so callers can
    match names exactly as identifyFeatures' csv_export does (e.g. "DTip",
    "L1-Rs", "ACV.p"). Falls back to `properties.name` then `properties.class`
    when classification.name is absent.

    Returns an empty dict if the file is unreadable or contains no Point
    features.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("load_landmarks: cannot read %s: %s", path, exc)
        return {}

    out: dict[str, tuple[float, float]] = {}
    for feat in data.get("features", []) if isinstance(data, dict) else []:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        props = feat.get("properties", {}) or {}
        classification = props.get("classification")
        name: Optional[str] = None
        if isinstance(classification, dict):
            name = classification.get("name")
        if not name:
            name = props.get("name") or props.get("class")
        if not name:
            continue
        out[str(name)] = (float(coords[0]), float(coords[1]))
    return out


def load_landmark_geojson_fc_props(path: Path) -> dict:
    """Return the top-level FeatureCollection ``properties`` dict, or {} when absent.

    v0.2.27+ stamps ``effective_um_per_px`` here (the per-image µm/px the
    pipeline actually used, after auto-detect / resolutionAdjust — which
    can differ from any single UI-set value). Older geojsons return {}
    and callers should fall back to a passed-in default scale.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("load_landmark_geojson_fc_props: cannot read %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    props = data.get("properties")
    return props if isinstance(props, dict) else {}


def compute_pair_distance_px(
    landmarks: dict[str, tuple[float, float]],
    name_a: str,
    name_b: str,
) -> Optional[float]:
    """Straight-line distance in pixels between landmarks `name_a` and `name_b`.

    Returns None if either landmark is missing.
    """
    a = landmarks.get(name_a)
    b = landmarks.get(name_b)
    if a is None or b is None:
        return None
    return math.hypot(b[0] - a[0], b[1] - a[1])
