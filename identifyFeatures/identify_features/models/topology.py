"""Static Drosophila wing vein topology — single source of truth.

All vein ordering, junction topology, and region boundary definitions
live here. No other module should hardcode these relationships.
"""

# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------

RELIABLE_LANDMARKS: set[str] = {
    "subcostal break",
    "alula notch",
    "L1-Rs",
    "L2-L3",
    "L4-L5",
}

# Soft landmarks: helpful when they agree with the skeleton, but never
# required. Use as hints, not hard constraints. If the skeleton doesn't
# reach the landmark, the vein is partial — don't force extension.
# These may be unreliable in mutants with premature vein termination.
SOFT_LANDMARKS: set[str] = {
    "DTip",
    "L2.d",
    "L4.d",
    "L5.d",
}

UNRELIABLE_LANDMARKS: set[str] = {
    "ACV.a",
    "ACV.p",
    "PCV.a",
    "PCV.p",
}

# ---------------------------------------------------------------------------
# Vein definitions
# ---------------------------------------------------------------------------

# Canonical longitudinal veins in anterior-to-posterior order
LONGITUDINAL_VEINS: list[str] = ["costa", "L1", "L2", "L3", "L4", "L5", "L6"]

CROSSVEINS: list[str] = ["ACV", "PCV"]

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
# Trace rules — how to trace each vein from its landmark anchor
# ---------------------------------------------------------------------------

TRACE_RULES: dict[str, dict] = {
    "L1": {
        "start_landmark": "L1-Rs",
        "trace_toward": "subcostal break",
        "direction": "proximal",  # L1 goes proximal from junction
    },
    "Rs": {
        "start_landmark": "L1-Rs",
        "trace_toward": "L2-L3",
        "direction": "distal",
    },
    "L2": {
        "start_landmark": "L2-L3",
        "trace_toward": "anterior_margin",
        "direction": "distal-anterior",
    },
    "L3": {
        "start_landmark": "L2-L3",
        "trace_toward": "DTip",
        "direction": "distal",
    },
    "L4": {
        "start_landmark": "L4-L5",
        "trace_toward": "distal-anterior",
        "direction": "distal-anterior",
    },
    "L5": {
        "start_landmark": "L4-L5",
        "trace_toward": "posterior_margin",
        "direction": "distal-posterior",
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

# 8 canonical intervein regions in anterior-to-posterior order
REGION_AP_ORDER: list[str] = [
    "costal",
    "marginal",
    "submarginal",
    "1st basal",
    "1st posterior",
    "discal",
    "2nd posterior",
    "3rd posterior",
]

# Which veins bound each region (used for naming by adjacency)
REGION_EXPECTED_VEINS: dict[str, set[str]] = {
    "costal": {"costa", "L1"},
    "marginal": {"L1", "L2"},
    "submarginal": {"L2", "L3"},
    "1st basal": {"L3", "L4", "ACV"},
    "1st posterior": {"L3", "L4", "ACV"},
    "discal": {"L4", "L5", "PCV"},
    "2nd posterior": {"L4", "L5", "PCV"},
    "3rd posterior": {"L5"},
}

# Vein boundary pairs — each vein separates these region pairs
VEIN_BOUNDARIES: dict[str, list[tuple[str, str]]] = {
    "L1": [("costal", "marginal")],
    "L2": [("marginal", "submarginal")],
    "L3": [("submarginal", "1st basal"), ("submarginal", "1st posterior")],
    "ACV": [("1st basal", "1st posterior")],
    "L4": [("1st basal", "discal"), ("discal", "1st posterior"), ("1st posterior", "2nd posterior")],
    "PCV": [("discal", "2nd posterior")],
    "L5": [("discal", "3rd posterior"), ("2nd posterior", "3rd posterior")],
}

# Disambiguation rules for regions with identical bounding vein sets.
# The key is the frozenset of bounding veins; value maps position to name.
REGION_DISAMBIGUATION: dict[frozenset[str], dict[str, str]] = {
    frozenset({"L3", "L4", "ACV"}): {
        "proximal": "1st basal",
        "distal": "1st posterior",
    },
    frozenset({"L4", "L5", "PCV"}): {
        "proximal": "discal",
        "distal": "2nd posterior",
    },
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
