"""Wing-solidity filter — first garbage detector, catches gross segmentation-shape failures.

Solidity = ``wing_outline.area / wing_outline.convex_hull.area`` — how completely the wing
fills its convex hull. A clean *Drosophila* wing is a smooth, near-convex teardrop, so its
solidity sits very high and very tightly (~0.983–0.990 across real data). A badly segmented
wing — a half-missing blob, a big concave bite — drops well below that; a featureless convex
blob with none of a real wing's slight alula/hinge concavity climbs toward 1.0. Both are
caught by a two-sided range.

Runs at the earliest hook (``FilterStage.WING_OUTLINE``), right after the outline is built.

Threshold modes (shared machinery in ``garbage_detector/batch.py``):
- ``fixed`` (default, primary): a transparent ``[solidity_min, solidity_max]`` range. Setting
  those two config fields *is* the user-defined-range path.
- ``batch_mad`` (opt-in): robust ``median ± k·1.4826·MAD`` over the batch. Robust statistics
  are used because the garbage we're detecting would otherwise corrupt a mean/stdev fence;
  ``k`` is user-modifiable (``solidity_batch_k``). Falls back to the fixed range below
  ``solidity_min_batch_size`` wings. Applied inline when a pre-pass injected
  ``solidity_batch_range``, otherwise deferred to the end of the batch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

from identify_features.garbage_detector.base import FilterContext, FilterStage, GarbageVerdict
from identify_features.garbage_detector.batch import (
    DEFERRED_METHOD,
    SOLIDITY_SPEC,
    inline_threshold,
    outside,
    precompute_outline_metrics,
    resolve_batch_threshold,
)
from identify_features.garbage_detector.registry import register_filter

if TYPE_CHECKING:
    from identify_features.config import PipelineConfig
    from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

_FILTER_NAME = "solidity"


def compute_solidity(outline: "Optional[Polygon]") -> Optional[float]:
    """Solidity of a wing outline, or ``None`` when it can't be computed.

    Single source of truth for the metric — also used by ``views/csv_export.py``.
    """
    if outline is None or outline.is_empty:
        return None
    hull_area = outline.convex_hull.area
    if hull_area <= 0:
        return None
    return outline.area / hull_area


def resolve_solidity_range(
    config: "PipelineConfig",
    batch_solidities: Optional[Sequence[float]] = None,
) -> tuple[float, float, str]:
    """Resolve the ``(lo, hi, method)`` acceptance range for solidity.

    Thin wrapper over ``batch.resolve_batch_threshold`` kept for callers that predate the
    shared batch machinery. ``batch_mad`` mode requires at least ``solidity_min_batch_size``
    finite solidities and a non-zero MAD; otherwise (and in ``fixed`` mode) the configured
    fixed range is returned.
    """
    return resolve_batch_threshold(SOLIDITY_SPEC, config, batch_solidities)


def precompute_solidities(detection_geojsons: Iterable[Path]) -> list[float]:
    """Cheaply compute solidity for each detection GeoJSON (for ``batch_mad`` resolution).

    Wrapper over ``batch.precompute_outline_metrics`` (which also yields the fragmentation
    metric from the same parse). Files that fail to parse or yield no usable outline are
    skipped.
    """
    return precompute_outline_metrics(detection_geojsons)[SOLIDITY_SPEC.metric_name]


@register_filter
class SolidityFilter:
    """Reject wings whose outline is missing/degenerate or whose solidity is out of range."""

    name = _FILTER_NAME
    stage = FilterStage.WING_OUTLINE

    def enabled(self, config: "PipelineConfig") -> bool:
        return bool(getattr(config, "solidity_filter_enabled", True))

    def check(self, ctx: FilterContext) -> GarbageVerdict:
        outline = ctx.wing_outline

        # Earliest, strongest catch: no usable wing at all.
        if outline is None or outline.is_empty:
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                "no wing outline (empty/degenerate segmentation)",
                metric_name="wing_solidity",
            )

        solidity = compute_solidity(outline)
        # Hand the value back to the pipeline so it lands on WingResult without recompute.
        ctx.scratch["wing_solidity"] = solidity

        if solidity is None:
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                "wing solidity undefined (degenerate convex hull)",
                metric_name="wing_solidity",
            )

        # A pre-resolved batch range (injected by a pre-pass) or the fixed range applies
        # inline; batch_mad without a pre-pass defers to the end of the batch.
        thr = inline_threshold(SOLIDITY_SPEC, ctx.config)
        if thr is None:
            return GarbageVerdict.ok(
                self.name, self.stage, metric_name="wing_solidity", metric_value=solidity, method=DEFERRED_METHOD
            )
        lo, hi, method = thr

        if outside(solidity, lo, hi):
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                SOLIDITY_SPEC.reason(solidity, lo, hi, method),
                metric_name="wing_solidity",
                metric_value=solidity,
                threshold=(lo, hi),
                method=method,
            )

        return GarbageVerdict.ok(
            self.name,
            self.stage,
            metric_name="wing_solidity",
            metric_value=solidity,
            threshold=(lo, hi),
            method=method,
        )
