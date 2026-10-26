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

Threshold modes (``fragmentation_mode``, machinery in ``garbage_detector/batch.py``): ``fixed``
uses ``fragmentation_max_secondary_frac``; ``batch_mad`` fences at ``median + k·1.4826·MAD``
over the batch — inline when a pre-pass injected ``fragmentation_batch_range``, otherwise
deferred to the end of the batch. Because most good wings score exactly 0.0 the batch MAD is
often zero, in which case the fixed fraction is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from identify_features.garbage_detector.base import FilterContext, FilterStage, GarbageVerdict
from identify_features.garbage_detector.batch import DEFERRED_METHOD, FRAGMENTATION_SPEC, inline_threshold, outside
from identify_features.garbage_detector.registry import register_filter

if TYPE_CHECKING:
    from identify_features.config import PipelineConfig
    from shapely.geometry import MultiPolygon, Polygon

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

        thr = inline_threshold(FRAGMENTATION_SPEC, ctx.config)
        if thr is None:  # batch_mad without a pre-pass: judged at the end of the batch
            return GarbageVerdict.ok(
                self.name, self.stage, metric_name="secondary_component_frac", metric_value=frac, method=DEFERRED_METHOD
            )
        lo, hi, method = thr

        if outside(frac, lo, hi):
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                FRAGMENTATION_SPEC.reason(frac, lo, hi, method),
                metric_name="secondary_component_frac",
                metric_value=frac,
                threshold=(lo, hi),
                method=method,
            )

        return GarbageVerdict.ok(
            self.name,
            self.stage,
            metric_name="secondary_component_frac",
            metric_value=frac,
            threshold=(lo, hi),
            method=method,
        )
