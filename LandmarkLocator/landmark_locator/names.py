"""Landmark naming shared across the pipeline. No heavy imports, so any module can use it."""

# Legacy landmark names that have since been renamed. Keys are the old names
# in both their GeoJSON form and their snake_case internal form; values are
# the names used everywhere now. Applied on every read (annotations,
# checkpoints, gate sidecars, override files) so existing data keeps working
# without being rewritten. The old distal-end-of-L3 name now follows the
# other distal-end landmarks L2.d / L4.d / L5.d.
#
# HingeChopper, wingRotator, identifyFeatures and measurementMaker keep their
# own copy of this table so they stay independent of LandmarkLocator; update
# them together.
LEGACY_LANDMARK_NAMES: dict[str, str] = {
    "DTip": "L3.d",
    "dtip": "l3_d",
}


def canonical_landmark_name(name: str) -> str:
    """Map a legacy landmark name (GeoJSON or internal form) to its current name."""
    name = name.strip()
    return LEGACY_LANDMARK_NAMES.get(name, name)
