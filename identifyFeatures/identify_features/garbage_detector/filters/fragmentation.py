"""Fragmentation filter — catches disconnected secondary regions (e.g. a partial second wing).

The wing outline is built by ``_compute_wing_outline`` as ``unary_union(...).buffer(+20)
.buffer(-20)`` followed by **keeping only the largest connected piece**. That largest-piece
drop is exactly what hides a second object in the frame: a partial neighbouring wing or a
debris blob sitting more than ~20px from the main wing survives as its own component, gets
discarded, and never perturbs the solidity measurement — so a clearly-bad image sails through.

This filter looks at the components *before* that drop and aborts when a secondary
disconnected component is too large relative to the main wing. On real data the separation is
wide: good wings carry at most ~0.6%-of-wing specks, whereas a partial second wing is several
percent up to >10%.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from identify_features.garbage_detector.base import FilterContext, FilterStage, GarbageVerdict
from identify_features.garbage_detector.registry import register_filter

if TYPE_CHECKING:
    from shapely.geometry import MultiPolygon, Polygon

    from identify_features.config import PipelineConfig

_FILTER_NAME = "fragmentation"
# Morphological-close radius — matches _compute_wing_outline's buffer so within-wing gaps
# stay merged while a genuinely separate object remains its own component.
_CLOSE_BUFFER_PX = 20.0


def _component_areas(polygons: "Sequence[Polygon | MultiPolygon]", buffer_px: float = _CLOSE_BUFFER_PX) -> list[float]:
    """Areas of the closed-union's connected components, largest first."""
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union

    if not polygons:
        return []
    union = unary_union(list(polygons)).buffer(buffer_px).buffer(-buffer_px)
    geoms = list(union.geoms) if isinstance(union, MultiPolygon) else [union]
    areas = [g.area for g in geoms if not g.is_empty and g.area > 0]
    return sorted(areas, reverse=True)


def compute_secondary_fraction(polygons: "Sequence[Polygon | MultiPolygon]") -> float:
    """Largest secondary component area ÷ main component area (0.0 if a single component)."""
    areas = _component_areas(polygons)
    if len(areas) < 2 or areas[0] <= 0:
        return 0.0
    return areas[1] / areas[0]


@register_filter
class FragmentationFilter:
    """Reject wings whose mask has a large disconnected secondary region."""

    name = _FILTER_NAME
    stage = FilterStage.WING_OUTLINE

    def enabled(self, config: "PipelineConfig") -> bool:
        return bool(getattr(config, "fragmentation_filter_enabled", True))

    def check(self, ctx: FilterContext) -> GarbageVerdict:
        polys: Optional[list] = ctx.all_polys
        if not polys:
            # Nothing to inspect (empty input is the solidity filter's concern).
            return GarbageVerdict.ok(self.name, self.stage, metric_name="secondary_component_frac")

        frac = compute_secondary_fraction(polys)
        ctx.scratch["secondary_component_frac"] = frac
        max_frac = ctx.config.fragmentation_max_secondary_frac

        if frac > max_frac:
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                f"disconnected secondary region is {frac:.1%} of the wing "
                f"(> {max_frac:.1%}) — likely a second wing/debris in frame",
                metric_name="secondary_component_frac",
                metric_value=frac,
                threshold=(0.0, max_frac),
            )

        return GarbageVerdict.ok(
            self.name,
            self.stage,
            metric_name="secondary_component_frac",
            metric_value=frac,
            threshold=(0.0, max_frac),
        )
