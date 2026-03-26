"""Static Drosophila vein topology rules and identity cues."""

from __future__ import annotations

LONGITUDINAL_VEINS = ["L1", "L2", "L3", "L4", "L5"]
CROSSVEINS = ["ACV", "PCV"]
ALL_VEINS = ["costa"] + LONGITUDINAL_VEINS + CROSSVEINS

# Priority-ordered cue types for vein identification
CUE_TYPES = [
    "spatial_anterior_posterior",
    "margin_connectivity",
    "crossvein_anchor",
    "relative_length_rank",
    "entry_angle",
]

# Approximate normalized X positions (anterior=0, posterior=1) for each longitudinal vein
SPATIAL_PRIORS: dict[str, tuple[float, float]] = {
    "costa": (0.0, 0.05),
    "L1": (0.05, 0.15),
    "L2": (0.15, 0.30),
    "L3": (0.30, 0.50),
    "L4": (0.50, 0.70),
    "L5": (0.70, 0.90),
}

# Normalized Y median ranges (anterior=0.0, posterior=1.0)
# Tuned against line-extent medians across test wings 1–3
SPATIAL_PRIORS_Y: dict[str, tuple[float, float]] = {
    "costa": (0.00, 0.15),
    "L1": (0.02, 0.20),
    "L2": (0.05, 0.32),
    "L3": (0.10, 0.42),
    "L4": (0.25, 0.62),
    "L5": (0.38, 0.78),
}

# Crossvein connection topology
CROSSVEIN_CONNECTIONS: dict[str, tuple[str, str]] = {
    "ACV": ("L3", "L4"),
    "PCV": ("L4", "L5"),
}

# All intervein space names ordered anterior to posterior (used internally)
_ALL_INTERVEIN_SPACE_NAMES = [
    "marginal_cell",  # between L1 and L2
    "submarginal_cell",  # between L2 and L3
    "1st_basal_cell",  # between L3 and L4, proximal to ACV
    "1st_posterior_cell",  # between L3 and L4, distal to ACV
    "discal_cell",  # between L4 and L5, proximal to PCV
    "2nd_posterior_cell",  # between L4 and L5, distal to PCV
    "3rd_posterior_cell",  # posterior to L5
]

# Intervein spaces included in measurements and overlays
INTERVEIN_SPACE_NAMES = [
    "marginal_cell",
    "submarginal_cell",
    "1st_basal_cell",
    "1st_posterior_cell",
    "discal_cell",
    "2nd_posterior_cell",
    "3rd_posterior_cell",
]

# Colors for intervein spaces (BGR for OpenCV, matching ground-truth overlay)
INTERVEIN_COLORS: dict[str, tuple[int, int, int]] = {
    "marginal_cell": (0, 0, 255),  # red
    "submarginal_cell": (0, 94, 255),  # orange
    "1st_basal_cell": (0, 191, 219),  # gold
    "1st_posterior_cell": (0, 128, 0),  # green
    "discal_cell": (255, 0, 0),  # blue
    "2nd_posterior_cell": (255, 187, 0),  # cyan
    "3rd_posterior_cell": (255, 0, 147),  # purple
}

# Colors for individual veins on skeleton overlay (BGR for OpenCV, matching ground-truth)
VEIN_COLORS: dict[str, tuple[int, int, int]] = {
    "costa": (255, 255, 255),  # white
    "L1": (0, 0, 255),  # red
    "Rs": (128, 0, 255),  # magenta (fused L2+L3 proximal stem)
    "L2": (0, 94, 255),  # orange
    "L3": (0, 191, 219),  # gold
    "L4": (255, 0, 0),  # blue
    "L5": (255, 0, 147),  # purple
    "ACV": (0, 128, 0),  # green
    "PCV": (255, 187, 0),  # cyan
}

# Vein boundary definitions: which intervein pair each vein separates
# Each tuple is (anterior_region, posterior_region)
VEIN_BOUNDARIES: dict[str, list[tuple[str, str]]] = {
    "L2": [("marginal_cell", "submarginal_cell")],
    "L3": [
        ("submarginal_cell", "1st_basal_cell"),
        ("submarginal_cell", "1st_posterior_cell"),
    ],
    "ACV": [("1st_basal_cell", "1st_posterior_cell")],
    "L4": [
        ("1st_basal_cell", "discal_cell"),
        ("discal_cell", "1st_posterior_cell"),
        ("1st_posterior_cell", "2nd_posterior_cell"),
    ],
    "PCV": [("discal_cell", "2nd_posterior_cell")],
    "L5": [
        ("discal_cell", "3rd_posterior_cell"),
        ("2nd_posterior_cell", "3rd_posterior_cell"),
    ],
}

# --- Geometric priors for independent vein/region identification ---

# Orientation: (min_deg, max_deg) from horizontal
VEIN_ORIENTATION_PRIORS: dict[str, tuple[float, float]] = {
    "L1": (0, 25),
    "L2": (0, 20),
    "L3": (0, 25),
    "L4": (0, 30),
    "L5": (0, 30),
    "ACV": (50, 90),
    "PCV": (40, 90),
}

# Length as fraction of wing span (min, max) — wide enough for natural variation
VEIN_LENGTH_PRIORS: dict[str, tuple[float, float]] = {
    "L1": (0.05, 0.25),
    "L2": (0.40, 0.90),
    "L3": (0.50, 1.00),
    "L4": (0.50, 1.00),
    "L5": (0.25, 0.90),
    "ACV": (0.01, 0.12),
    "PCV": (0.03, 0.18),
}

# Region area as fraction of total intervein area (min, max)
REGION_AREA_PRIORS: dict[str, tuple[float, float]] = {
    "marginal_cell": (0.05, 0.18),
    "submarginal_cell": (0.08, 0.25),
    "1st_basal_cell": (0.01, 0.06),
    "1st_posterior_cell": (0.10, 0.25),
    "discal_cell": (0.03, 0.12),
    "2nd_posterior_cell": (0.10, 0.25),
    "3rd_posterior_cell": (0.12, 0.35),
}

# Reverse lookup: which veins bound each region
REGION_EXPECTED_VEINS: dict[str, set[str]] = {
    "marginal_cell": {"L1", "L2"},
    "submarginal_cell": {"L2", "L3"},
    "1st_basal_cell": {"L3", "L4", "ACV"},
    "1st_posterior_cell": {"L3", "L4", "ACV"},
    "discal_cell": {"L4", "L5", "PCV"},
    "2nd_posterior_cell": {"L4", "L5", "PCV"},
    "3rd_posterior_cell": {"L5"},
}

# Anterior-to-posterior ordering of all vein types (used for topology validation)
VEIN_Y_ORDER = ["L1", "L2", "L3", "ACV", "L4", "PCV", "L5"]

# Expected region centroid Y ordering (anterior=0.0, posterior=1.0)
REGION_Y_ORDER = [
    "marginal_cell",
    "submarginal_cell",
    "1st_basal_cell",
    "1st_posterior_cell",
    "discal_cell",
    "2nd_posterior_cell",
    "3rd_posterior_cell",
]

# Vein shape thresholds
STRAIGHTNESS_THRESHOLD = 0.65
MAX_ANGLE_CHANGE_DEG = 60.0

# ---------------------------------------------------------------------------
# Scale infrastructure — all distance/area constants in micrometers
# ---------------------------------------------------------------------------

DEFAULT_UM_PER_PX: float = 0.483
_um_per_px: float = DEFAULT_UM_PER_PX


def set_scale(um_per_px: float | None = None) -> None:
    """Set the global µm-per-pixel scale factor."""
    global _um_per_px
    _um_per_px = um_per_px if um_per_px is not None else DEFAULT_UM_PER_PX


def get_scale() -> float:
    """Return the current µm-per-pixel scale factor."""
    return _um_per_px


def um_to_px(um: float) -> float:
    """Convert a distance in micrometers to pixels."""
    return um / _um_per_px


def um2_to_px2(um2: float) -> float:
    """Convert an area in µm² to pixels²."""
    return um2 / (_um_per_px**2)


# ---------------------------------------------------------------------------
# Distance constants in micrometers (calibrated at 0.483 µm/px)
# ---------------------------------------------------------------------------

# -- Junction / node snapping --
SNAP_RADIUS_UM = 14.49  # 30 px — junction clustering / node snapping
SNAP_RADIUS_LARGE_UM = 19.32  # 40 px — junction endpoint proximity check

# -- Tangent / angle sampling --
TANGENT_DIST_UM = 38.64  # 80 px — tangent vector distance from junction
STEP_DIST_UM = 24.15  # 50 px — walk step for angle-change sampling

# -- Path length thresholds --
MIN_SEGMENT_LENGTH_UM = 4.83  # 10 px — minimum merged/centerline segment
MIN_SPLIT_LENGTH_UM = 96.6  # 200 px — minimum piece after sharp-turn split
MIN_PATH_LENGTH_UM = 241.5  # 500 px — minimum path length to attempt split

# -- Smoothing parameters --
SMOOTH_SIGMA_SPLIT_UM = 24.15  # 50 px — sigma for path-splitting smoothing
SMOOTH_SPACING_UM = 2.415  # 5 px — general smooth sample spacing
SMOOTH_SPACING_FINE_UM = 1.449  # 3 px — fine smooth sample spacing (crossveins)
SIMPLIFY_UM = 1.449  # 3 px — general simplification tolerance
SIMPLIFY_DARKBAND_UM = 2.415  # 5 px — dark band line simplification

# -- Crossvein thresholds --
MAX_CROSSVEIN_FLOOR_UM = 193.2  # 400 px — floor for max crossvein length
SHORT_CROSSVEIN_UM = 144.9  # 300 px — short crossvein threshold
MAX_CROSSVEIN_DEFAULT_UM = 289.8  # 600 px — default max crossvein (no bbox)
CV_PROXIMITY_UM = 48.3  # 100 px — crossvein validation proximity
CV_NORM_DIST_UM = 96.6  # 200 px — crossvein scoring normalization distance
CV_CONNECTIVITY_UM = 24.15  # 50 px — crossvein connectivity / L4-L5 swap

# -- Buffers --
BUFFER_OUTLINE_UM = 9.66  # 20 px — wing outline polygon buffer
BUFFER_VEIN_UM = 2.415  # 5 px — vein polygon buffer in outline
BUFFER_SMOOTH_UM = 2.415  # 5 px — outline smooth buffer
BUFFER_SPATIAL_UM = 12.075  # 25 px — spatial proximity for poly-vein mapping
MIN_SPATIAL_LENGTH_UM = 14.49  # 30 px — min intersection length for poly-vein

# -- Skeleton / centerline extraction --
BRIDGE_THRESHOLD_UM = 14.49  # 30 px — bridge dangling endpoints
MIN_POLY_AREA_UM2 = 23.33  # 100 px² — min polygon area (erosion)

# -- Wing geometry --
HINGE_EXTENSION_UM = 48.3  # 100 px — hinge line extension
COMPARTMENT_SIMPLIFY_UM = 4.83  # 10 px — L4 simplification for compartments
COMPARTMENT_EXTENSION_UM = 241.5  # 500 px — L4 extension beyond wing boundary

# -- Graph --
MAX_GAP_UM = 38.64  # 80 px — max polygon gap for graph building
GRAPH_SNAP_VEINS_UM = 24.15  # 50 px — snap tolerance for vein LineStrings

# -- GeoJSON parser --
MIN_POLY_WIDTH_UM = 48.3  # 100 px — min polygon width for constriction
MIN_CROSS_WIDTH_UM = 9.66  # 20 px — min cross-section width
CUT_EXTENSION_UM = 4.83  # 10 px — cut line extension beyond polygon


# -- Erosion amount lists --
BOTTLENECK_EROSION_UM = [4.83, 7.245, 9.66, 14.49]  # [10,15,20,30] px
SPLIT_EROSION_UM = [4.83, 7.245, 9.66, 14.49, 24.15]  # [10,15,20,30,50] px

# -- Misc --
GT_TOLERANCE_UM = 12.075  # 25 px — ground truth validation tolerance
