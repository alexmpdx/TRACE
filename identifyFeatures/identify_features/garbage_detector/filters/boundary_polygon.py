"""Boundary-polygon filter — abort when the wing-margin landmarks form a self-intersecting ring.

Joining the eight landmarks that lie on the wing margin in anatomical order
(subcostal break → L2.d → L3.d → L4.d → L5.d → alula notch → L4-L5 → L1-Rs → back) gives a
simple, roughly convex teardrop on any correctly landmarked wing. If two edges of that ring
cross, at least one landmark is grossly misplaced (on the hinge, on a neighbouring wing, or
stacked on another landmark) — a failure the per-landmark confidence gate cannot see because
it judges each point alone. The test is invariant to handedness, rotation and scale, so it
has no threshold.

Runs at the earliest hook (``FilterStage.WING_OUTLINE``): it needs only the raw landmark
coordinates, so it fires before the image read and before the ``skip_vein_tracing`` early
return, protecting the landmark-only outputs too.

Validated 2026-10-18 on the landmark GeoJSON sets on disk: 0/117 good wings rejected,
4/47 ``identifyFeatures/problemWings/`` rejected — all four visually confirmed landmark
failures. See ``documentation/spec_boundary_polygon_filter.md``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Mapping, Optional

from identify_features.garbage_detector.base import FilterContext, FilterStage, GarbageVerdict
from identify_features.garbage_detector.registry import register_filter
from shapely.geometry import LineString

if TYPE_CHECKING:
    from identify_features.config import PipelineConfig
    from identify_features.models.datatypes import Landmark

logger = logging.getLogger(__name__)

_FILTER_NAME = "boundary_polygon"

# Canonical GeoJSON landmark keys, in margin order. Legacy ``DTip`` GeoJSONs are renamed
# to ``L3.d`` by ``load_landmarks_geojson``, so the filter only ever sees canonical keys.
BOUNDARY_LANDMARK_ORDER: tuple[str, ...] = (
    "subcostal break",
    "L2.d",
    "L3.d",
    "L4.d",
    "L5.d",
    "alula notch",
    "L4-L5",
    "L1-Rs",
)
# Fewer vertices than this and a "polygon" can't meaningfully self-intersect; pass.
MIN_BOUNDARY_VERTICES = 4


def boundary_polygon_vertices(landmarks: Mapping[str, "Landmark"]) -> list[tuple[str, tuple[float, float]]]:
    """``(name, (x, y))`` for each boundary landmark present, in margin order."""
    return [(n, (landmarks[n].x, landmarks[n].y)) for n in BOUNDARY_LANDMARK_ORDER if n in landmarks]


def boundary_polygon_crossings(landmarks: Mapping[str, "Landmark"]) -> Optional[list[tuple[str, str]]]:
    """Pairs of ring edges that intersect, as ``("A→B", "C→D")`` strings; ``[]`` if simple.

    Returns ``None`` when fewer than ``MIN_BOUNDARY_VERTICES`` boundary landmarks are present
    (undefined — caller should pass). Adjacent edges share a vertex by construction and are
    skipped, including the wrap-around pair. ``intersects`` (not ``crosses``) is used so a
    non-adjacent edge that merely touches another, or two non-adjacent vertices that
    coincide, also count.
    """
    verts = boundary_polygon_vertices(landmarks)
    n = len(verts)
    if n < MIN_BOUNDARY_VERTICES:
        return None
    edges = [(verts[i][0], verts[(i + 1) % n][0], LineString([verts[i][1], verts[(i + 1) % n][1]])) for i in range(n)]
    crossings: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # wrap-around neighbours share the first vertex
            if edges[i][2].intersects(edges[j][2]):
                crossings.append((f"{edges[i][0]}→{edges[i][1]}", f"{edges[j][0]}→{edges[j][1]}"))
    return crossings


@register_filter
class BoundaryPolygonFilter:
    """Reject wings whose margin-landmark polygon self-intersects."""

    name = _FILTER_NAME
    stage = FilterStage.WING_OUTLINE

    def enabled(self, config: "PipelineConfig") -> bool:
        return bool(getattr(config, "boundary_polygon_filter_enabled", True))

    def check(self, ctx: FilterContext) -> GarbageVerdict:
        crossings = boundary_polygon_crossings(ctx.landmarks or {})
        if crossings is None:
            logger.info(
                "%s: fewer than %d boundary landmarks present — boundary polygon check skipped",
                ctx.specimen_id,
                MIN_BOUNDARY_VERTICES,
            )
            return GarbageVerdict.ok(self.name, self.stage, metric_name="boundary_polygon_crossings")
        ctx.scratch["boundary_polygon_crossings"] = len(crossings)
        if crossings:
            a, b = crossings[0]
            more = f" (+{len(crossings) - 1} more)" if len(crossings) > 1 else ""
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                f"boundary landmark polygon self-intersects: {a} crosses {b}{more}",
                metric_name="boundary_polygon_crossings",
                metric_value=float(len(crossings)),
            )
        return GarbageVerdict.ok(self.name, self.stage, metric_name="boundary_polygon_crossings", metric_value=0.0)
