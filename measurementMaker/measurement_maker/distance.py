"""Read landmark GeoJSONs and compute pair distances."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _normalize_landmark_key(name: str) -> str:
    """Snake-case a landmark key so short and display forms compare equal.

    Mirrors LandmarkLocator's own ``_normalize_name`` so we don't depend on
    it directly (avoids pulling landmark_locator + torch into every caller
    of the CSV writer). Whitespace, dots, and hyphens all collapse to ``_``
    and the result is lower-cased. Examples:

        "L1-Rs"                        → "l1_rs"
        "L1-Rs junction"               → "l1_rs_junction"
        "L1-Rs junction junction"      → "l1_rs_junction_junction"
        "DTip"                         → "dtip"
        "L3 distal end"                → "l3_distal_end"
    """
    return re.sub(r"[\s.\-]+", "_", name.strip()).lower()


def resolve_landmark_key(landmarks: dict, name: str) -> Optional[str]:
    """Find the actual key in ``landmarks`` that corresponds to ``name``.

    Handles the gap between the short GeoJSON key form the user configures
    (``DTip``, ``L1-Rs``, ``ACV.p``, ``L2.d``) and whatever the training-
    set-derived model actually wrote into the landmark geojson — which in
    practice is often the display-name form (``L3 distal end``,
    ``L1-Rs junction junction``, ``ACV-L4 junction``, ``L2 distal end``).
    Match order (fall-through on miss):

      1. Exact key match.
      2. LANDMARK_DISPLAY_NAMES[name] — canonical display name for a known
         short key (e.g. ``DTip`` → ``L3 distal end``). This handles the
         canonical case in the mapping table.
      3. Normalized-substring match — normalize the target(s) to
         snake_case (``dtip``, ``l3_distal_end``) and find any landmark
         key whose normalized form contains one as a prefix or the entire
         key contains the target. Handles training-set variants like
         ``L1-Rs junction junction`` where the display name matches as a
         prefix.

    Returns the resolved key (usable directly against ``landmarks``) or
    ``None`` if no match is found.
    """
    if not landmarks:
        return None
    # 1. Exact match.
    if name in landmarks:
        return name

    # Lazy import to avoid a circular measurement_maker.distance ↔
    # measurement_maker.__init__ ↔ measurement_maker.landmark_names loop
    # at module import time.
    from measurement_maker.landmark_names import LANDMARK_DISPLAY_NAMES

    # 2. Canonical display name.
    display = LANDMARK_DISPLAY_NAMES.get(name)
    if display and display in landmarks:
        return display

    # 3. Normalized-substring / prefix match. Build candidates from both
    # the short form and the canonical display form so training variants
    # that ADD text to the display name (e.g. "L1-Rs junction" →
    # "L1-Rs junction junction") still resolve. Use the LONGEST candidate
    # first so "l3_distal_end" beats a spurious "l3_" match against
    # "l3_something_else".
    candidates_norm: list[str] = []
    if display:
        candidates_norm.append(_normalize_landmark_key(display))
    candidates_norm.append(_normalize_landmark_key(name))
    # Longer first — more specific candidates win.
    candidates_norm.sort(key=len, reverse=True)

    normalized_keys = {_normalize_landmark_key(k): k for k in landmarks}
    for cand in candidates_norm:
        # Exact normalized match — the strongest signal.
        if cand in normalized_keys:
            return normalized_keys[cand]
    for cand in candidates_norm:
        # Prefix match — handles training variants that append text
        # ("l1_rs_junction" prefix-matches "l1_rs_junction_junction").
        for nk, orig in normalized_keys.items():
            if nk.startswith(cand + "_") or nk == cand:
                return orig
    for cand in candidates_norm:
        # Substring match as a last resort. Weakest signal — keep it
        # last so it doesn't shadow a cleaner prefix or exact match.
        for nk, orig in normalized_keys.items():
            if cand in nk:
                return orig
    return None


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

    Names are resolved through :func:`resolve_landmark_key` so short
    (``DTip``, ``L1-Rs``) and display (``L3 distal end``,
    ``L1-Rs junction junction``) forms both work — the user's saved pairs
    always use the short form, but the geojson can carry either
    depending on the trained model's naming convention.

    Returns None if either landmark can't be resolved.
    """
    key_a = resolve_landmark_key(landmarks, name_a)
    key_b = resolve_landmark_key(landmarks, name_b)
    if key_a is None or key_b is None:
        return None
    a = landmarks[key_a]
    b = landmarks[key_b]
    return math.hypot(b[0] - a[0], b[1] - a[1])
