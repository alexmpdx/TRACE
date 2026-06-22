"""Wing-solidity filter — first garbage detector, catches gross segmentation-shape failures.

Solidity = ``wing_outline.area / wing_outline.convex_hull.area`` — how completely the wing
fills its convex hull. A clean *Drosophila* wing is a smooth, near-convex teardrop, so its
solidity sits very high and very tightly (~0.983–0.990 across real data). A badly segmented
wing — a half-missing blob, a big concave bite — drops well below that; a featureless convex
blob with none of a real wing's slight alula/hinge concavity climbs toward 1.0. Both are
caught by a two-sided range.

Runs at the earliest hook (``FilterStage.WING_OUTLINE``), right after the outline is built.

Threshold modes (see ``resolve_solidity_range``):
- ``fixed`` (default, primary): a transparent ``[solidity_min, solidity_max]`` range. Setting
  those two config fields *is* the user-defined-range path.
- ``batch_mad`` (opt-in): robust ``median ± k·1.4826·MAD`` over the batch. Robust statistics
  are used because the garbage we're detecting would otherwise corrupt a mean/stdev fence;
  ``k`` is user-modifiable (``solidity_batch_k``). Falls back to the fixed range below
  ``solidity_min_batch_size`` wings.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

from identify_features.garbage_detector.base import FilterContext, FilterStage, GarbageVerdict
from identify_features.garbage_detector.registry import register_filter

if TYPE_CHECKING:
    from shapely.geometry import Polygon

    from identify_features.config import PipelineConfig

logger = logging.getLogger(__name__)

_FILTER_NAME = "solidity"
# MAD -> standard-deviation scale factor for a normal distribution (1 / 0.6745).
_MAD_TO_SIGMA = 1.4826


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

    ``batch_mad`` mode requires at least ``solidity_min_batch_size`` finite solidities;
    otherwise (and in ``fixed`` mode) the configured fixed range is returned.
    """
    if getattr(config, "solidity_mode", "fixed") == "batch_mad" and batch_solidities is not None:
        vals = [s for s in batch_solidities if s is not None and s == s]  # drop None/NaN
        if len(vals) >= config.solidity_min_batch_size:
            median = statistics.median(vals)
            mad = statistics.median([abs(v - median) for v in vals])
            robust_sigma = _MAD_TO_SIGMA * mad
            k = config.solidity_batch_k
            lo = median - k * robust_sigma
            hi = median + k * robust_sigma
            return lo, hi, f"batch MAD k={k:g} (n={len(vals)})"
    return config.solidity_min, config.solidity_max, "fixed range"


def precompute_solidities(detection_geojsons: Iterable[Path]) -> list[float]:
    """Cheaply compute solidity for each detection GeoJSON (for ``batch_mad`` resolution).

    Parses each file, builds the wing outline, and computes solidity. Files that fail to
    parse or yield no usable outline are skipped. Used by the batch orchestrator to resolve
    one global solidity range before the (expensive) full per-wing pass.
    """
    from identify_features.models.geojson_io import _compute_wing_outline, load_detection_geojson

    solidities: list[float] = []
    for det in detection_geojsons:
        try:
            vein_polys, intervein_polys = load_detection_geojson(Path(det))
            outline = _compute_wing_outline(vein_polys + intervein_polys)
            sol = compute_solidity(outline)
            if sol is not None:
                solidities.append(sol)
        except Exception:
            logger.warning("precompute_solidities: failed to read %s", det, exc_info=True)
    return solidities


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

        # Prefer a pre-resolved batch range (injected into config by the batch
        # orchestrator); otherwise resolve from config alone (fixed / user range).
        batch_range = getattr(ctx.config, "solidity_batch_range", None)
        if batch_range is not None:
            lo, hi = batch_range
            method = f"batch MAD k={ctx.config.solidity_batch_k:g}"
        else:
            lo, hi, method = resolve_solidity_range(ctx.config)

        if solidity < lo or solidity > hi:
            return GarbageVerdict.reject(
                self.name,
                self.stage,
                f"wing solidity {solidity:.3f} outside [{lo:.3f}, {hi:.3f}] ({method})",
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
