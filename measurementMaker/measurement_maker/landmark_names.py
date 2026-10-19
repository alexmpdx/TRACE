"""Canonical raw-key → friendly-name mappings for LandmarkLocator landmarks.

Two key formats coexist in the stack:

  - GeoJSON-style: ``L3.d``, ``L1-Rs``, ``ACV.a``, ``alula notch`` — what
    the locator writes into ``_landmarks.geojson`` and what the picker /
    Custom Measurements UI pass around.
  - Gate-config-style: ``l3_d``, ``l1_rs``, ``acv_a``, ``alula_notch`` —
    snake_case keys used in ``gate_config.yaml`` and in
    ``LowConfidenceLandmarkError``'s failure dicts (so they surface
    verbatim in TRACE's run log when a gate trips).

Both formats map to the same friendly anatomical names. Callers that only
deal with one format should import that format's dict directly; callers
that translate arbitrary log text (e.g. TRACE's log handler) should use
the combined ``ALL_LANDMARK_KEY_DISPLAY_NAMES``.

This module is the single source of truth — TRACE/settings_dialog.py
keeps its own copy of the snake_case dict for backwards-compat, but new
additions should land here first.
"""

# Legacy landmark names, renamed on read (landmark GeoJSONs, saved
# measurement pairs) so existing files keep working. Local copy of
# LandmarkLocator's landmark_locator/names.py (measurementMaker stays
# independent of LandmarkLocator); update them together.
LEGACY_LANDMARK_NAMES: dict[str, str] = {"DTip": "L3.d", "dtip": "l3_d"}


def canonical_landmark_name(name: str) -> str:
    """Map a legacy landmark name (GeoJSON or internal form) to its current name."""
    return LEGACY_LANDMARK_NAMES.get(name.strip(), name)


# GeoJSON-style keys (what the locator emits, what the picker uses).
LANDMARK_DISPLAY_NAMES: dict[str, str] = {
    "ACV.a": "ACV-L3 junction",
    "ACV.p": "ACV-L4 junction",
    "alula notch": "alula notch",
    "L3.d": "L3 distal end",
    "L1-Rs": "L1-Rs junction",
    "L2.d": "L2 distal end",
    "L2-L3": "L2-L3-Rs junction",
    "L4.d": "L4 distal end",
    "L4-L5": "L4-L5 junction",
    "L5.d": "L5 distal end",
    "PCV.a": "PCV-L4 junction",
    "PCV.p": "PCV-L5 junction",
    "subcostal break": "subcostal break",
}

# Gate-config / LowConfidenceLandmarkError keys (snake_case).
# Same display strings as LANDMARK_DISPLAY_NAMES; the keys are what
# gate_config.yaml uses for per_landmark thresholds.
LANDMARK_GATE_KEY_DISPLAY_NAMES: dict[str, str] = {
    "acv_a": "ACV-L3 junction",
    "acv_p": "ACV-L4 junction",
    "alula_notch": "alula notch",
    "l3_d": "L3 distal end",
    "l1_rs": "L1-Rs junction",
    "l2_d": "L2 distal end",
    "l2_l3": "L2-L3-Rs junction",
    "l4_d": "L4 distal end",
    "l4_l5": "L4-L5 junction",
    "l5_d": "L5 distal end",
    "pcv_a": "PCV-L4 junction",
    "pcv_p": "PCV-L5 junction",
    "subcostal_break": "subcostal break",
}

# Combined view for translators that need to catch either format. No key
# appears in both dicts (case + punctuation make them disjoint), so the
# union is well-defined.
ALL_LANDMARK_KEY_DISPLAY_NAMES: dict[str, str] = {
    **LANDMARK_DISPLAY_NAMES,
    **LANDMARK_GATE_KEY_DISPLAY_NAMES,
}


def landmark_display_name(name: str) -> str:
    """Friendly anatomical name for any raw landmark key, falling back to the raw form.

    Accepts either the GeoJSON-style key (``L3.d``) or the gate-config
    snake_case key (``l3_d``) — both map to the same friendly string.
    """
    return ALL_LANDMARK_KEY_DISPLAY_NAMES.get(name, name)
