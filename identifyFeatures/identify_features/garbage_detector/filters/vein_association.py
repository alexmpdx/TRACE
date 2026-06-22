"""Vein-association filter — catches segmented vein tissue the tracer couldn't explain.

The segmentation model paints "vein" pixels; the vein tracer then traces named veins
through them and ``assign_vein_tissue_polygons`` buffers each traced centerline into a
tissue polygon. When a large share of the segmented vein area is **not** covered by any
traced vein's tissue (including ectopic veins), something is wrong — either the model
hallucinated vein where there is none, or the tracing failed badly. Either way the wing
is bad data.

Metric (at the TISSUE hook, after tissue polygons are assigned):

    unassigned_fraction = area(vein_mask \\ ⋃ tissue_polygons) / area(vein_mask)

i.e. the fraction of segmented vein area not associated with any identified vein. Aborts
when it exceeds ``max_unassigned_vein_frac``. The tissue buffer (~1 vein width) over-covers
a genuinely-traced vein, so well-traced wings score near zero and only un-traced vein blobs
contribute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from identify_features.garbage_detector.base import FilterContext, FilterStage, GarbageVerdict
from identify_features.garbage_detector.registry import register_filter

if TYPE_CHECKING:
    from identify_features.config import PipelineConfig

_FILTER_NAME = "vein_association"


def compute_unassigned_vein_fraction(vein_polys, veins) -> float | None:
    """Fraction of segmented vein area not covered by any traced vein's tissue polygon.

    Returns ``None`` when there is no segmented vein area to measure.
    """
    from shapely.ops import unary_union

    if not vein_polys:
        return None
    vein_mask = unary_union(list(vein_polys))
    total = vein_mask.area
    if total <= 0:
        return None

    tissues = [v.tissue_polygon for v in (veins or []) if getattr(v, "tissue_polygon", None) is not None]
    if not tissues:
        return 1.0  # vein tissue exists but nothing was traced through it

    associated = vein_mask.intersection(unary_union(tissues)).area
    return max(0.0, 1.0 - associated / total)


@register_filter
class VeinAssociationFilter:
    """Reject wings where too much segmented vein tissue isn't associated with a traced vein."""

    name = _FILTER_NAME
    stage = FilterStage.TISSUE

    def enabled(self, config: "PipelineConfig") -> bool:
        return bool(getattr(config, "vein_association_filter_enabled", True))

    def check(self, ctx: FilterContext) -> GarbageVerdict:
        frac = compute_unassigned_vein_fraction(ctx.vein_polys, ctx.veins)
        if frac is None:
            # No segmented vein tissue at all — not this filter's concern.
            return GarbageVerdict.ok(self.name, self.stage, metric_name="unassigned_vein_frac")

        ctx.scratch["unassigned_vein_frac"] = frac
        max_frac = ctx.config.max_unassigned_vein_frac

        if frac > max_frac:
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                f"{frac:.1%} of segmented vein tissue is not associated with any traced vein "
                f"(> {max_frac:.1%}) — bad segmentation or failed tracing",
                metric_name="unassigned_vein_frac",
                metric_value=frac,
                threshold=(0.0, max_frac),
            )

        return GarbageVerdict.ok(
            self.name,
            self.stage,
            metric_name="unassigned_vein_frac",
            metric_value=frac,
            threshold=(0.0, max_frac),
        )
