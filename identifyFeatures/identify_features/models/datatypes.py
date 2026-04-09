"""Data structures for the identification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VeinStatus(Enum):
    """Status of a vein identification."""

    IDENTIFIED = "identified"  # Traced from landmark, high confidence
    INFERRED = "inferred"  # Identified by position/topology, not from landmark
    PARTIAL = "partial"  # Traced but doesn't reach expected endpoint
    ABSENT = "absent"  # Expected but not found in skeleton
    ECTOPIC = "ectopic"  # Present but unexpected (extra vein)


class VeinType(Enum):
    """Anatomical category of a vein."""

    LONGITUDINAL = "longitudinal"
    CROSSVEIN = "crossvein"
    COSTA = "costa"
    RADIAL_SECTOR = "radial_sector"


class SkeletonMethod(Enum):
    """Available skeletonization methods."""

    MEDIAL_AXIS = "medial-axis"
    VORONOI = "voronoi"
    BOUNDARY_SMOOTH = "boundary-smooth"
    RIDGE = "ridge"  # Hessian-based distance-map ridge extraction
    PATH_TRACE = "path-trace"  # Weighted path tracing on medial axis


class PruneMethod(Enum):
    """Available skeleton pruning methods."""

    DISTANCE_MAP = "distance-map"  # r_endpoint/r_junction ratio + ribbon area
    FULL_BOUNDARY = "full-boundary"  # Boundary reconstruction significance
    MULTI_SCALE = "multi-scale"  # Multi-scale persistence
    SINGLE_SCALE_COMPARE = "single-scale-compare"  # Original vs smoothed skeleton overlap
    SINGLE_SCALE = "single-scale"  # Skeletonize smoothed mask only


# ---------------------------------------------------------------------------
# Landmarks
# ---------------------------------------------------------------------------


@dataclass
class Landmark:
    """A detected landmark point with reliability metadata."""

    name: str
    point: Point
    reliable: bool
    soft: bool = False  # Helpful hint, but don't force — may be wrong in mutants
    snapped_node: Optional[int] = None
    snap_distance: float = 0.0

    @property
    def x(self) -> float:
        return self.point.x

    @property
    def y(self) -> float:
        return self.point.y


# ---------------------------------------------------------------------------
# Wing axis
# ---------------------------------------------------------------------------


@dataclass
class WingAxis:
    """Wing-level proximal/distal reference axis.

    Defined by two landmarks (conventionally alula notch → DTip). Provides
    a signed scalar PD coordinate for any point in the wing frame: 0 at the
    proximal anchor, 1 at the distal anchor, linear in between, extrapolating
    beyond when needed.
    """

    proximal_point: Point
    distal_point: Point
    unit_vector: tuple[float, float]  # pointing proximal → distal
    length: float  # pixel distance between the two anchors

    def project(self, point: Point) -> float:
        """Return the normalized PD coordinate of a point (0=proximal, 1=distal)."""
        dx = point.x - self.proximal_point.x
        dy = point.y - self.proximal_point.y
        scalar = dx * self.unit_vector[0] + dy * self.unit_vector[1]
        return scalar / self.length if self.length > 0 else 0.0

    @property
    def ap_vector(self) -> tuple[float, float]:
        """Unit vector perpendicular to the PD axis, pointing posterior.

        Derived by rotating ``unit_vector`` 90° via (dx, dy) → (-dy, dx).
        Sign convention: when the wing is un-rotated in the image frame
        (alula left, DTip right → PD ≈ (+1, 0)), this yields (0, +1),
        which is positive-Y in image coordinates — i.e. "down" in the
        image, which is posterior for a standard upright wing.
        """
        dx, dy = self.unit_vector
        return (-dy, dx)


# ---------------------------------------------------------------------------
# Skeleton graph
# ---------------------------------------------------------------------------


@dataclass
class SkeletonGraph:
    """NetworkX graph built from skeletonized vein mask.

    Nodes have attributes: {x: float, y: float, degree: int}
    Edges have attributes: {edge_id: int, line: LineString, length_px: float}
    """

    graph: object  # nx.Graph — typed loosely to avoid import at module level
    vein_mask: np.ndarray
    skeleton: np.ndarray
    image_shape: tuple[int, int]
    distance_map: Optional[np.ndarray] = None  # from medial_axis
    voronoi_labels: Optional[np.ndarray] = None  # pixel → intervein polygon index
    median_vein_width_px: float = 0.0  # median full vein width in pixels


# ---------------------------------------------------------------------------
# Vein identification
# ---------------------------------------------------------------------------


@dataclass
class VeinIdentification:
    """A single identified vein with full provenance."""

    vein_id: str  # "L1", "L2", "ACV", "EV1", etc.
    vein_type: VeinType
    status: VeinStatus
    centerline: Optional[LineString] = None
    tissue_polygon: Optional[Polygon | MultiPolygon] = None
    edge_ids: list[int] = field(default_factory=list)
    length_px: float = 0.0
    length_um: Optional[float] = None
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    landmark_anchors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Intervein regions
# ---------------------------------------------------------------------------


@dataclass
class InterveinRegion:
    """A named intervein region."""

    name: str  # e.g. "marginal", "discal"
    polygon: Optional[Polygon | MultiPolygon] = None
    bounding_veins: set[str] = field(default_factory=set)
    area_px2: float = 0.0
    area_um2: Optional[float] = None
    confidence: float = 0.0
    status: str = "identified"  # "identified", "inferred", "merged"


# ---------------------------------------------------------------------------
# Parsed inputs
# ---------------------------------------------------------------------------


@dataclass
class ParsedInput:
    """Parsed contents of the input GeoJSON files."""

    vein_polygons: list[Polygon | MultiPolygon]
    intervein_polygons: list[Polygon | MultiPolygon]
    landmarks: dict[str, Landmark]
    wing_outline: Optional[Polygon] = None
    image_shape: Optional[tuple[int, int]] = None  # (height, width)


# ---------------------------------------------------------------------------
# Full result
# ---------------------------------------------------------------------------


@dataclass
class WingResult:
    """Complete output for a single wing specimen."""

    specimen_id: str
    veins: list[VeinIdentification] = field(default_factory=list)
    intervein_regions: list[InterveinRegion] = field(default_factory=list)
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    wing_outline: Optional[Polygon] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def vein_map(self) -> dict[str, VeinIdentification]:
        """Lookup veins by ID."""
        return {v.vein_id: v for v in self.veins}

    @property
    def region_map(self) -> dict[str, InterveinRegion]:
        """Lookup regions by name."""
        return {r.name: r for r in self.intervein_regions}
