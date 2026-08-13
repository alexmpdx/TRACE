"""Intervein-association filter — catches segmented intervein tissue no region explains.

The mirror of ``vein_association``, one class down. The segmentation model paints
"intervein" pixels; the pipeline then splits and names those into anatomical regions
(``InterveinRegion``). When a large share of the segmented intervein area is **not**
covered by any named region, something is wrong — the model painted intervein where
there is none, or region naming failed on it. Either way the wing is bad data.

Metric (at the REGIONS hook, after region naming):

    unassigned_fraction = area(intervein_mask \\ ⋃ region_polygons) / area(intervein_mask)

i.e. the fraction of segmented intervein area not associated with any named region.
Aborts when it exceeds ``max_unassigned_intervein_frac``. Skipped when intervein regions
aren't computed at all (``skip_intervein_regions``), since there'd be nothing to compare.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from identify_features.garbage_detector.base import FilterContext, FilterStage, GarbageVerdict
from identify_features.garbage_detector.registry import register_filter

if TYPE_CHECKING:
    from identify_features.config import PipelineConfig

_FILTER_NAME = "intervein_association"


def compute_unassigned_intervein_fraction(intervein_polys, regions) -> float | None:
    """Fraction of segmented intervein area not covered by any named region polygon.

    Returns ``None`` when there is no segmented intervein area to measure.
    """
    from shapely.ops import unary_union

    if not intervein_polys:
        return None
    intervein_mask = unary_union(list(intervein_polys))
    total = intervein_mask.area
    if total <= 0:
        return None

    region_polys = [r.polygon for r in (regions or []) if getattr(r, "polygon", None) is not None]
    if not region_polys:
        return 1.0  # intervein tissue exists but nothing was named

    associated = intervein_mask.intersection(unary_union(region_polys)).area
    return max(0.0, 1.0 - associated / total)


@register_filter
class InterveinAssociationFilter:
    """Reject wings where too much segmented intervein tissue isn't associated with a region."""

    name = _FILTER_NAME
    stage = FilterStage.REGIONS

    def enabled(self, config: "PipelineConfig") -> bool:
        # No regions are computed when intervein labeling is skipped — nothing to compare.
        if getattr(config, "skip_intervein_regions", False):
            return False
        return bool(getattr(config, "intervein_association_filter_enabled", True))

    def check(self, ctx: FilterContext) -> GarbageVerdict:
        frac = compute_unassigned_intervein_fraction(ctx.intervein_polys, ctx.regions)
        if frac is None:
            # No segmented intervein tissue at all — not this filter's concern.
            return GarbageVerdict.ok(self.name, self.stage, metric_name="unassigned_intervein_frac")

        ctx.scratch["unassigned_intervein_frac"] = frac
        max_frac = ctx.config.max_unassigned_intervein_frac

        if frac > max_frac:
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                f"{frac:.1%} of segmented intervein tissue is not associated with any named region "
                f"(> {max_frac:.1%}) — bad segmentation or failed region naming",
                metric_name="unassigned_intervein_frac",
                metric_value=frac,
                threshold=(0.0, max_frac),
            )

        return GarbageVerdict.ok(
            self.name,
            self.stage,
            metric_name="unassigned_intervein_frac",
            metric_value=frac,
            threshold=(0.0, max_frac),
        )
