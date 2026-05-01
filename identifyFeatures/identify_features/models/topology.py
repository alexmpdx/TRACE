"""Static Drosophila wing vein topology — single source of truth.

All vein ordering, junction topology, and region boundary definitions
live here. No other module should hardcode these relationships.
"""

# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------

# Reliability is no longer a hard-coded set; it now comes from the per-landmark
# `reliable` flag in each landmark geojson (LandmarkLocator's confidence-gate
# verdict). The previous RELIABLE_LANDMARKS / UNRELIABLE_LANDMARKS / SOFT_LANDMARKS
# sets have been retired — see geojson_io.load_landmarks_geojson.

# ---------------------------------------------------------------------------
# Vein definitions
# ---------------------------------------------------------------------------

# Canonical longitudinal veins in anterior-to-posterior order
LONGITUDINAL_VEINS: list[str] = ["costa", "L1", "L2", "L3", "L4", "L5", "L6"]

CROSSVEINS: list[str] = ["ACV", "PCV"]

# Reliable landmark endpoint pair for each longitudinal that can be traced
# by shortest-path between its anchor landmarks. costa is detected
# separately (margin band); L6 has no clean landmark endpoint pair.
LONGITUDINAL_ENDPOINTS: dict[str, tuple[str, str]] = {
    "L1": ("subcostal break", "L1-Rs"),
    "Rs": ("L1-Rs", "L2-L3"),
    "L2": ("L2-L3", "L2.d"),
    "L3": ("L2-L3", "DTip"),
    "L4": ("L4-L5", "L4.d"),
    "L5": ("L4-L5", "L5.d"),
}

# Radial sector is the short proximal stem connecting L1-Rs to L2-L3
OTHER_VEINS: list[str] = ["Rs"]

ALL_CANONICAL_VEINS: list[str] = LONGITUDINAL_VEINS + CROSSVEINS + OTHER_VEINS

# Full anterior-to-posterior ordering (for display / sorting)
VEIN_AP_ORDER: list[str] = [
    "costa",
    "L1",
    "Rs",
    "L2",
    "L3",
    "ACV",
    "L4",
    "PCV",
    "L5",
    "L6",
]

# ---------------------------------------------------------------------------
# Junction topology — which veins meet at each landmark junction
# ---------------------------------------------------------------------------

# At each junction landmark, "incoming" veins arrive from the proximal side
# and "outgoing" veins depart toward the distal side. This defines the
# tracing directions from each landmark.
JUNCTION_TOPOLOGY: dict[str, dict[str, list[str]]] = {
    "L1-Rs": {
        "incoming": ["L1"],  # L1 arrives from subcostal break
        "outgoing": ["Rs"],  # Rs departs toward L2-L3
    },
    "L2-L3": {
        "incoming": ["Rs"],  # Rs arrives from L1-Rs
        "outgoing": ["L2", "L3"],  # L2 departs anterior, L3 toward DTip
    },
    "L4-L5": {
        "incoming": [],  # Proximal convergence (from wing base)
        "outgoing": ["L4", "L5"],  # L4 anterior, L5 posterior
    },
}

# ---------------------------------------------------------------------------
# Crossvein connectivity
# ---------------------------------------------------------------------------

CROSSVEIN_CONNECTIONS: dict[str, tuple[str, str]] = {
    "ACV": ("L3", "L4"),
    "PCV": ("L4", "L5"),
}

# ---------------------------------------------------------------------------
# Intervein regions
# ---------------------------------------------------------------------------

# 7 canonical intervein regions in anterior-to-posterior order.
# The costal cell is removed in preprocessing and never appears in the
# intervein polygon list, so it is intentionally omitted.
REGION_AP_ORDER: list[str] = [
    "marginal",
    "submarginal",
    "1st basal",
    "1st posterior",
    "discal",
    "2nd posterior",
    "3rd posterior",
]

# Which veins bound each region (used for naming by adjacency).
# No entry for the costal cell — it is removed in preprocessing.
REGION_EXPECTED_VEINS: dict[str, set[str]] = {
    "marginal": {"L1", "L2", "costa", "Rs"},
    "submarginal": {"L2", "L3", "costa"},
    "1st basal": {"L3", "L4", "ACV", "Rs"},
    "1st posterior": {"L3", "L4", "ACV", "costa"},
    "discal": {"L4", "L5", "PCV"},
    "2nd posterior": {"L4", "L5", "PCV"},
    "3rd posterior": {"L5"},
}


def build_region_forbidden_veins(
    expected: dict[str, set[str]] | None = None,
) -> dict[str, set[str]]:
    """Return forbidden = ALL_CANONICAL_VEINS - expected[R] for each region.

    Parameterized on ``expected`` so callers can pass the runtime-augmented
    effective_expected (e.g. 3rd posterior with L6 added).
    """
    src = expected if expected is not None else REGION_EXPECTED_VEINS
    canonical = set(ALL_CANONICAL_VEINS)
    return {region: canonical - veins for region, veins in src.items()}


# Tied-region pairs ordered proximal → distal. Used by intervein_namer to
# break ties when two candidate regions share an identical expected vein set.
# The first element is proximal, the second is distal.
REGION_PD_PAIRS: list[tuple[str, str]] = [
    ("discal", "2nd posterior"),
]

# Vein boundary pairs — each vein separates these region pairs
VEIN_BOUNDARIES: dict[str, list[tuple[str, str]]] = {
    "L2": [("marginal", "submarginal")],
    "Rs": [("marginal", "1st basal")],
    "L3": [("submarginal", "1st basal"), ("submarginal", "1st posterior")],
    "ACV": [("1st basal", "1st posterior")],
    "L4": [("1st basal", "discal"), ("discal", "1st posterior"), ("1st posterior", "2nd posterior")],
    "PCV": [("discal", "2nd posterior")],
    "L5": [("discal", "3rd posterior"), ("2nd posterior", "3rd posterior")],
}

# ---------------------------------------------------------------------------
# Display colors (RGB) matching GT_naming format
# ---------------------------------------------------------------------------

REGION_COLORS: dict[str, list[int]] = {
    "costal": [255, 255, 255],
    "marginal": [255, 0, 0],
    "submarginal": [255, 94, 0],
    "1st basal": [255, 201, 0],
    "1st posterior": [0, 255, 0],
    "discal": [0, 187, 255],
    "2nd posterior": [0, 0, 255],
    "3rd posterior": [128, 0, 255],
}

VEIN_COLORS: dict[str, list[int]] = {
    "costa": [200, 200, 200],
    "L1": [255, 100, 100],
    "Rs": [255, 150, 50],
    "L2": [255, 200, 0],
    "L3": [100, 255, 100],
    "ACV": [0, 200, 200],
    "L4": [0, 128, 0],
    "PCV": [150, 50, 255],
    "L5": [200, 0, 200],
    "L6": [128, 128, 128],
}
