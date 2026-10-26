"""Batch-specific thresholds for the metric-based garbage filters.

Every garbage filter that compares a per-wing *metric* against a numeric threshold
(solidity, fragmentation, vein association, intervein association) supports two
threshold modes, selected per filter through ``<filter>_mode`` on ``PipelineConfig``:

- ``fixed`` (default): the configured fixed threshold / range.
- ``batch_mad`` (opt-in): a robust fence derived from the current batch —
  ``median ± k·1.4826·MAD`` over every wing's metric. Robust statistics are used because
  the garbage being detected would otherwise corrupt a mean/stdev fence. ``k`` is
  user-modifiable (``<filter>_batch_k``). Below ``<filter>_min_batch_size`` wings the batch
  is too small to estimate a distribution and the fixed threshold is used instead; the
  same fallback applies when the batch MAD is zero (e.g. most wings score exactly 0.0).

Two-sided metrics (solidity) get both fences; one-sided "fraction of unexplained tissue"
metrics get only the upper fence, clamped to ``[0, 1]``.

How a batch fence reaches a wing
--------------------------------
A batch fence needs *every* wing's metric before any wing can be judged, so there are two
delivery paths:

1. **Pre-pass** (cheap outline metrics only — solidity and fragmentation). Both come
   straight from the detection GeoJSON, so a batch orchestrator can compute them for the
   whole batch up front (``precompute_outline_metrics``), resolve the fence, and inject it
   into the transient ``<filter>_batch_range`` config field. The filter then aborts a wing
   inline, as early as the fixed mode would.

2. **Deferred** (any metric). When a filter is in ``batch_mad`` mode and no pre-resolved
   range was injected, it records the metric and *passes* the wing. After the whole batch
   has run, the orchestrator calls ``apply_deferred_batch_fences`` on the collected
   ``WingResult``s; wings outside the fence are rejected post hoc with the same reason
   text an inline rejection would carry. This is the only possible path for the
   association metrics (they exist only after vein tracing / region naming), and the path
   TRACE uses for every metric (its chunked Stage 1 → Stage 2 loop never holds the whole
   batch's GeoJSONs at once).

``BATCH_METRICS`` is the single table that describes each metric; filters, the CLI and
TRACE all read it so the four filters never drift apart.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence

from identify_features.garbage_detector.base import FilterStage, GarbageVerdict

if TYPE_CHECKING:
    from identify_features.config import PipelineConfig
    from identify_features.models.datatypes import WingResult

logger = logging.getLogger(__name__)

# MAD -> standard-deviation scale factor for a normal distribution (1 / 0.6745).
MAD_TO_SIGMA = 1.4826

FIXED_METHOD = "fixed range"
DEFERRED_METHOD = "batch_mad: resolved at end of batch"


@dataclass(frozen=True)
class BatchMetricSpec:
    """Everything the batch machinery needs to know about one metric filter."""

    filter_name: str  # GarbageFilter.name
    stage: FilterStage
    metric_name: str  # key in FilterContext.scratch / WingResult.qc_metrics
    mode_field: str  # PipelineConfig field: "fixed" | "batch_mad"
    k_field: str  # PipelineConfig field: robust-sigma multiplier
    min_size_field: str  # PipelineConfig field: minimum batch size for batch_mad
    range_field: str  # transient PipelineConfig field: pre-resolved (lo, hi) or None
    fixed_range: Callable[["PipelineConfig"], tuple[float, float]]
    two_sided: bool
    # Fence is clamped to this domain (fractions live in [0, 1]); None = unclamped.
    domain: Optional[tuple[float, float]]
    # Rejection reason text: (value, lo, hi, method) -> one line for the user.
    reason: Callable[[float, float, float, str], str]
    # Computed in the cheap detection-GeoJSON pre-pass (``precompute_outline_metrics``).
    outline_metric: bool = False

    def mode(self, config: "PipelineConfig") -> str:
        return str(getattr(config, self.mode_field, "fixed"))

    def batch_mode(self, config: "PipelineConfig") -> bool:
        return self.mode(config) == "batch_mad"

    def k(self, config: "PipelineConfig") -> float:
        return float(getattr(config, self.k_field))

    def min_size(self, config: "PipelineConfig") -> int:
        return int(getattr(config, self.min_size_field))

    def preresolved_range(self, config: "PipelineConfig") -> Optional[tuple[float, float]]:
        rng = getattr(config, self.range_field, None)
        if rng is None:
            return None
        return float(rng[0]), float(rng[1])

    def is_deferred(self, config: "PipelineConfig") -> bool:
        """True when the fence can only be applied after the whole batch has run."""
        return self.batch_mode(config) and self.preresolved_range(config) is None


def _solidity_reason(value: float, lo: float, hi: float, method: str) -> str:
    return f"wing solidity {value:.3f} outside [{lo:.3f}, {hi:.3f}] ({method})"


def _fragmentation_reason(value: float, lo: float, hi: float, method: str) -> str:
    return (
        f"disconnected secondary region is {value:.1%} of the wing (> {hi:.1%}, {method}) "
        f"— likely a second wing/debris in frame"
    )


def _vein_association_reason(value: float, lo: float, hi: float, method: str) -> str:
    return (
        f"{value:.1%} of segmented vein tissue is not associated with any traced vein "
        f"(> {hi:.1%}, {method}) — bad segmentation or failed tracing"
    )


def _intervein_association_reason(value: float, lo: float, hi: float, method: str) -> str:
    return (
        f"{value:.1%} of segmented intervein tissue is not associated with any named region "
        f"(> {hi:.1%}, {method}) — bad segmentation or failed region naming"
    )


SOLIDITY_SPEC = BatchMetricSpec(
    filter_name="solidity",
    stage=FilterStage.WING_OUTLINE,
    metric_name="wing_solidity",
    mode_field="solidity_mode",
    k_field="solidity_batch_k",
    min_size_field="solidity_min_batch_size",
    range_field="solidity_batch_range",
    fixed_range=lambda c: (float(c.solidity_min), float(c.solidity_max)),
    two_sided=True,
    domain=None,
    reason=_solidity_reason,
    outline_metric=True,
)

FRAGMENTATION_SPEC = BatchMetricSpec(
    filter_name="fragmentation",
    stage=FilterStage.WING_OUTLINE,
    metric_name="secondary_component_frac",
    mode_field="fragmentation_mode",
    k_field="fragmentation_batch_k",
    min_size_field="fragmentation_min_batch_size",
    range_field="fragmentation_batch_range",
    fixed_range=lambda c: (0.0, float(c.fragmentation_max_secondary_frac)),
    two_sided=False,
    domain=(0.0, 1.0),
    reason=_fragmentation_reason,
    outline_metric=True,
)

VEIN_ASSOCIATION_SPEC = BatchMetricSpec(
    filter_name="vein_association",
    stage=FilterStage.TISSUE,
    metric_name="unassigned_vein_frac",
    mode_field="vein_association_mode",
    k_field="vein_association_batch_k",
    min_size_field="vein_association_min_batch_size",
    range_field="vein_association_batch_range",
    fixed_range=lambda c: (0.0, float(c.max_unassigned_vein_frac)),
    two_sided=False,
    domain=(0.0, 1.0),
    reason=_vein_association_reason,
)

INTERVEIN_ASSOCIATION_SPEC = BatchMetricSpec(
    filter_name="intervein_association",
    stage=FilterStage.REGIONS,
    metric_name="unassigned_intervein_frac",
    mode_field="intervein_association_mode",
    k_field="intervein_association_batch_k",
    min_size_field="intervein_association_min_batch_size",
    range_field="intervein_association_batch_range",
    fixed_range=lambda c: (0.0, float(c.max_unassigned_intervein_frac)),
    two_sided=False,
    domain=(0.0, 1.0),
    reason=_intervein_association_reason,
)

# Pipeline order — also the precedence when a wing trips several deferred fences.
BATCH_METRICS: dict[str, BatchMetricSpec] = {
    s.filter_name: s for s in (SOLIDITY_SPEC, FRAGMENTATION_SPEC, VEIN_ASSOCIATION_SPEC, INTERVEIN_ASSOCIATION_SPEC)
}


# ---------------------------------------------------------------------------
# Fence maths
# ---------------------------------------------------------------------------


def _finite(values: Iterable[Optional[float]]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(v)]


def robust_fence(values: Sequence[float], k: float, two_sided: bool = True) -> Optional[tuple[float, float]]:
    """``median ± k·1.4826·MAD`` over ``values``; ``None`` when the MAD is zero/undefined.

    One-sided fences return ``(-inf, median + k·σ)``. A zero MAD (at least half the batch
    sits on one value) would collapse the fence onto the median and reject anything that
    deviates at all, so it is reported as degenerate and the caller falls back.
    """
    vals = _finite(values)
    if not vals:
        return None
    median = statistics.median(vals)
    mad = statistics.median([abs(v - median) for v in vals])
    robust_sigma = MAD_TO_SIGMA * mad
    if not robust_sigma > 0:
        return None
    hi = median + k * robust_sigma
    lo = median - k * robust_sigma if two_sided else -math.inf
    return lo, hi


def resolve_batch_threshold(
    spec: BatchMetricSpec,
    config: "PipelineConfig",
    batch_values: Optional[Sequence[Optional[float]]] = None,
) -> tuple[float, float, str]:
    """Resolve ``(lo, hi, method)`` for ``spec`` given the batch's metric values.

    Returns the fixed range in ``fixed`` mode, when ``batch_values`` is missing, when
    fewer than ``<filter>_min_batch_size`` finite values exist, or when the batch MAD is
    zero. ``method`` says which of those happened so the rejection text is honest.
    """
    lo, hi = spec.fixed_range(config)
    if not spec.batch_mode(config) or batch_values is None:
        return lo, hi, FIXED_METHOD
    vals = _finite(batch_values)
    n = len(vals)
    min_size = spec.min_size(config)
    if n < min_size:
        return lo, hi, f"{FIXED_METHOD}: batch of {n} < {min_size}"
    k = spec.k(config)
    fence = robust_fence(vals, k, two_sided=spec.two_sided)
    if fence is None:
        return lo, hi, f"{FIXED_METHOD}: batch MAD is zero (n={n})"
    blo, bhi = fence
    if spec.domain is not None:
        blo = max(blo, spec.domain[0])
        bhi = min(bhi, spec.domain[1])
    return blo, bhi, f"batch MAD k={k:g} (n={n})"


def inline_threshold(spec: BatchMetricSpec, config: "PipelineConfig") -> Optional[tuple[float, float, str]]:
    """Threshold a filter applies *inline* during ``check``; ``None`` means "defer".

    A pre-resolved batch range (injected by a pre-pass) wins; otherwise ``fixed`` mode
    returns the fixed range and ``batch_mad`` mode defers to ``apply_deferred_batch_fences``.
    """
    pre = spec.preresolved_range(config)
    if pre is not None:
        return pre[0], pre[1], f"batch MAD k={spec.k(config):g}"
    if spec.batch_mode(config):
        return None
    lo, hi = spec.fixed_range(config)
    return lo, hi, FIXED_METHOD


def outside(value: float, lo: float, hi: float) -> bool:
    return value < lo or value > hi


# ---------------------------------------------------------------------------
# Pre-pass (cheap outline metrics straight from the detection GeoJSON)
# ---------------------------------------------------------------------------


def precompute_outline_metrics(detection_geojsons: Iterable[Path]) -> dict[str, list[float]]:
    """Per-metric value lists for every outline metric, from the detection GeoJSONs alone.

    Keys are ``spec.metric_name`` for each ``outline_metric`` spec. Files that fail to
    parse are skipped (logged). Used by a batch orchestrator that can see the whole batch
    up front to resolve fences *before* the expensive per-wing pass.
    """
    from identify_features.garbage_detector.filters.fragmentation import compute_secondary_fraction
    from identify_features.garbage_detector.filters.solidity import compute_solidity
    from identify_features.models.geojson_io import _compute_wing_outline, load_detection_geojson

    out: dict[str, list[float]] = {s.metric_name: [] for s in BATCH_METRICS.values() if s.outline_metric}
    for det in detection_geojsons:
        try:
            vein_polys, intervein_polys = load_detection_geojson(Path(det))
            all_polys = vein_polys + intervein_polys
            sol = compute_solidity(_compute_wing_outline(all_polys))
            if sol is not None:
                out[SOLIDITY_SPEC.metric_name].append(sol)
            out[FRAGMENTATION_SPEC.metric_name].append(compute_secondary_fraction(all_polys))
        except Exception:
            logger.warning("precompute_outline_metrics: failed to read %s", det, exc_info=True)
    return out


def resolve_outline_batch_ranges(
    config: "PipelineConfig",
    metrics: dict[str, Sequence[float]],
) -> dict[str, tuple[float, float, str]]:
    """``{range_field: (lo, hi, method)}`` for each enabled outline metric in batch mode."""
    resolved: dict[str, tuple[float, float, str]] = {}
    for spec in BATCH_METRICS.values():
        if not spec.outline_metric or not spec.batch_mode(config) or not _filter_enabled(spec, config):
            continue
        resolved[spec.range_field] = resolve_batch_threshold(spec, config, metrics.get(spec.metric_name, []))
    return resolved


# ---------------------------------------------------------------------------
# Deferred (post-hoc) application
# ---------------------------------------------------------------------------


@dataclass
class ResolvedFence:
    spec: BatchMetricSpec
    lo: float
    hi: float
    method: str
    n: int

    def describe(self) -> str:
        label = "accept" if self.spec.two_sided else "accept ≤"
        rng = f"[{self.lo:.4f}, {self.hi:.4f}]" if self.spec.two_sided else f"{self.hi:.4f}"
        return f"{self.spec.filter_name}: {label} {rng} ({self.method}, {self.n} wing(s) measured)"


@dataclass
class DeferredOutcome:
    """What ``apply_deferred_batch_fences`` decided."""

    fences: list[ResolvedFence]
    # stem -> verdict for every wing rejected post hoc (first tripped fence, pipeline order).
    rejections: dict[str, GarbageVerdict]


def _filter_enabled(spec: BatchMetricSpec, config: "PipelineConfig") -> bool:
    """Ask the registered filter itself (it knows e.g. about ``skip_intervein_regions``)."""
    from identify_features.garbage_detector.registry import REGISTRY

    for flt in REGISTRY.get(spec.stage, []):
        if flt.name == spec.filter_name:
            try:
                return bool(flt.enabled(config))
            except Exception:
                logger.exception("garbage filter %s.enabled() raised; treating as disabled", flt.name)
                return False
    return False


def deferred_specs(config: "PipelineConfig") -> list[BatchMetricSpec]:
    """Specs whose fence must be applied after the batch (enabled, batch mode, no pre-pass)."""
    return [s for s in BATCH_METRICS.values() if s.is_deferred(config) and _filter_enabled(s, config)]


def apply_deferred_batch_fences(
    config: "PipelineConfig",
    wings: Sequence[tuple[str, "WingResult"]],
) -> DeferredOutcome:
    """Resolve every deferred fence over ``wings`` and return the post-hoc rejections.

    ``wings`` are ``(stem, WingResult)`` pairs for the wings that completed the pipeline.
    Each ``WingResult.qc_metrics`` carries the metric its filter recorded while deferring.
    Nothing is mutated; the caller removes rejected wings from its outputs.
    """
    fences: list[ResolvedFence] = []
    rejections: dict[str, GarbageVerdict] = {}
    for spec in deferred_specs(config):
        values = [(stem, _metric(wr, spec)) for stem, wr in wings]
        measured = [v for _s, v in values if v is not None]
        lo, hi, method = resolve_batch_threshold(spec, config, measured)
        fences.append(ResolvedFence(spec=spec, lo=lo, hi=hi, method=method, n=len(measured)))
        for stem, v in values:
            if v is None or stem in rejections or not outside(v, lo, hi):
                continue
            rejections[stem] = GarbageVerdict.reject(
                spec.filter_name,
                spec.stage,
                spec.reason(v, lo, hi, method),
                metric_name=spec.metric_name,
                metric_value=v,
                threshold=(lo, hi),
                method=method,
            )
    return DeferredOutcome(fences=fences, rejections=rejections)


def _metric(wr: "WingResult", spec: BatchMetricSpec) -> Optional[float]:
    v = (getattr(wr, "qc_metrics", None) or {}).get(spec.metric_name)
    if v is None and spec is SOLIDITY_SPEC:
        v = getattr(wr, "wing_solidity", None)
    if v is None or not math.isfinite(v):
        return None
    return float(v)
