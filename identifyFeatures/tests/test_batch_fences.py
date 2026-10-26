"""Tests for the shared batch-threshold machinery (``garbage_detector/batch.py``).

Covers the robust fence maths, the per-filter spec table, inline-vs-deferred resolution
inside each metric filter, and the post-hoc application that the CLI and TRACE both call
after a batch has run.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

import pytest
from identify_features.config import PipelineConfig
from identify_features.garbage_detector import (
    BATCH_METRICS,
    DEFERRED_METHOD,
    FilterContext,
    FilterStage,
    GarbageRejection,
    apply_deferred_batch_fences,
    deferred_specs,
    resolve_batch_threshold,
    resolve_solidity_range,
    robust_fence,
    run_stage_filters,
)
from identify_features.garbage_detector.batch import (
    FIXED_METHOD,
    FRAGMENTATION_SPEC,
    INTERVEIN_ASSOCIATION_SPEC,
    MAD_TO_SIGMA,
    SOLIDITY_SPEC,
    VEIN_ASSOCIATION_SPEC,
    inline_threshold,
)
from identify_features.garbage_detector.filters.fragmentation import FragmentationFilter
from identify_features.garbage_detector.filters.intervein_association import InterveinAssociationFilter
from identify_features.garbage_detector.filters.solidity import SolidityFilter
from identify_features.garbage_detector.filters.vein_association import VeinAssociationFilter
from identify_features.models.datatypes import WingResult
from shapely.geometry import Polygon, box

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _wing(stem: str, **metrics: float) -> tuple[str, WingResult]:
    wr = WingResult(specimen_id=stem)
    wr.qc_metrics.update(metrics)
    if "wing_solidity" in metrics:
        wr.wing_solidity = metrics["wing_solidity"]
    return stem, wr


def _batch(metric: str, values: list[float], prefix: str = "w") -> list[tuple[str, WingResult]]:
    return [_wing(f"{prefix}{i}", **{metric: v}) for i, v in enumerate(values)]


def _all_batch_mode(config: PipelineConfig, k: float = 5.0, min_size: int = 3) -> PipelineConfig:
    for spec in BATCH_METRICS.values():
        setattr(config, spec.mode_field, "batch_mad")
        setattr(config, spec.k_field, k)
        setattr(config, spec.min_size_field, min_size)
    return config


def _notched_wing() -> Polygon:
    """A square with a bite out of it — solidity well below 1 but a single component."""
    return box(0, 0, 100, 100).difference(box(40, 60, 60, 100))


# ---------------------------------------------------------------------------
# fence maths
# ---------------------------------------------------------------------------


class TestRobustFence:
    def test_two_sided_matches_hand_computation(self):
        vals = [0.980, 0.982, 0.984, 0.985, 0.986, 0.987, 0.990]
        med = statistics.median(vals)
        mad = statistics.median([abs(v - med) for v in vals])
        lo, hi = robust_fence(vals, k=3.0, two_sided=True)
        assert lo == pytest.approx(med - 3.0 * MAD_TO_SIGMA * mad)
        assert hi == pytest.approx(med + 3.0 * MAD_TO_SIGMA * mad)

    def test_one_sided_has_no_lower_fence(self):
        lo, hi = robust_fence([0.01, 0.02, 0.03, 0.04, 0.05], k=2.0, two_sided=False)
        assert lo == -math.inf
        assert hi == pytest.approx(0.03 + 2.0 * MAD_TO_SIGMA * 0.01)

    def test_zero_mad_is_degenerate(self):
        # More than half the batch sits on exactly 0.0 → MAD is 0 → no usable fence.
        assert robust_fence([0.0, 0.0, 0.0, 0.0, 0.02, 0.05], k=5.0, two_sided=False) is None

    def test_empty_and_non_finite_values_are_dropped(self):
        assert robust_fence([], k=5.0) is None
        assert robust_fence([math.nan, None, math.inf], k=5.0) is None  # type: ignore[list-item]
        fence = robust_fence([1.0, 2.0, 3.0, math.nan, None], k=1.0)  # type: ignore[list-item]
        assert fence is not None

    def test_outliers_do_not_drag_the_fence(self):
        # A classic mean/stdev fence would be dominated by the 0.5 garbage wing; MAD isn't.
        good = [0.984, 0.985, 0.986, 0.985, 0.984, 0.987, 0.986, 0.985]
        lo, hi = robust_fence(good + [0.5], k=5.0)
        assert lo > 0.95
        assert hi < 1.01


class TestResolveBatchThreshold:
    def test_fixed_mode_returns_fixed_range(self):
        cfg = PipelineConfig()
        lo, hi, method = resolve_batch_threshold(SOLIDITY_SPEC, cfg, [0.98] * 50)
        assert (lo, hi) == (cfg.solidity_min, cfg.solidity_max)
        assert method == FIXED_METHOD

    def test_small_batch_falls_back_to_fixed(self):
        cfg = _all_batch_mode(PipelineConfig(), min_size=25)
        lo, hi, method = resolve_batch_threshold(VEIN_ASSOCIATION_SPEC, cfg, [0.01, 0.02, 0.03])
        assert (lo, hi) == (0.0, cfg.max_unassigned_vein_frac)
        assert "batch of 3 < 25" in method

    def test_zero_mad_falls_back_to_fixed(self):
        cfg = _all_batch_mode(PipelineConfig(), min_size=3)
        lo, hi, method = resolve_batch_threshold(FRAGMENTATION_SPEC, cfg, [0.0] * 10 + [0.03])
        assert (lo, hi) == (0.0, cfg.fragmentation_max_secondary_frac)
        assert "MAD is zero" in method

    def test_one_sided_fence_is_clamped_to_unit_interval(self):
        cfg = _all_batch_mode(PipelineConfig(), k=100.0, min_size=3)
        # median 0.07, MAD 0.01 → raw upper fence 0.07 + 100·1.4826·0.01 ≈ 1.55, clamped to 1.0
        lo, hi, method = resolve_batch_threshold(INTERVEIN_ASSOCIATION_SPEC, cfg, [0.05, 0.06, 0.07, 0.08, 0.09])
        assert lo == 0.0
        assert hi == 1.0
        assert method.startswith("batch MAD k=100")

    def test_two_sided_fence_reports_n(self):
        cfg = _all_batch_mode(PipelineConfig(), k=5.0, min_size=3)
        vals = [0.980, 0.982, 0.984, 0.985, 0.986, 0.987, 0.990]
        lo, hi, method = resolve_batch_threshold(SOLIDITY_SPEC, cfg, vals)
        assert lo < min(vals) and hi > max(vals)
        assert method == "batch MAD k=5 (n=7)"

    def test_legacy_solidity_wrapper_delegates(self):
        cfg = _all_batch_mode(PipelineConfig(), min_size=3)
        vals = [0.980, 0.982, 0.984, 0.985, 0.986, 0.987, 0.990]
        assert resolve_solidity_range(cfg, vals) == resolve_batch_threshold(SOLIDITY_SPEC, cfg, vals)


class TestInlineThreshold:
    def test_fixed_mode_is_inline(self):
        cfg = PipelineConfig()
        assert inline_threshold(FRAGMENTATION_SPEC, cfg) == (0.0, cfg.fragmentation_max_secondary_frac, FIXED_METHOD)

    def test_batch_mode_without_prepass_defers(self):
        cfg = _all_batch_mode(PipelineConfig())
        for spec in BATCH_METRICS.values():
            assert inline_threshold(spec, cfg) is None
            assert spec.is_deferred(cfg)

    def test_preresolved_range_wins(self):
        cfg = _all_batch_mode(PipelineConfig())
        cfg.vein_association_batch_range = (0.0, 0.042)
        assert inline_threshold(VEIN_ASSOCIATION_SPEC, cfg) == (0.0, 0.042, "batch MAD k=5")
        assert not VEIN_ASSOCIATION_SPEC.is_deferred(cfg)


# ---------------------------------------------------------------------------
# filters: inline vs deferred behaviour
# ---------------------------------------------------------------------------


class TestFiltersDefer:
    def test_solidity_fixed_mode_rejects_inline(self):
        cfg = PipelineConfig()
        ctx = FilterContext(config=cfg, specimen_id="s", wing_outline=_notched_wing())
        verdict = SolidityFilter().check(ctx)
        assert not verdict.passed
        assert "fixed range" in verdict.reason
        assert ctx.scratch["wing_solidity"] == pytest.approx(0.92)

    def test_solidity_batch_mode_records_and_passes(self):
        cfg = _all_batch_mode(PipelineConfig())
        ctx = FilterContext(config=cfg, specimen_id="s", wing_outline=_notched_wing())
        verdict = SolidityFilter().check(ctx)
        assert verdict.passed
        assert verdict.method == DEFERRED_METHOD
        assert ctx.metrics()["wing_solidity"] == pytest.approx(0.92)

    def test_solidity_hard_failures_still_abort_in_batch_mode(self):
        cfg = _all_batch_mode(PipelineConfig())
        ctx = FilterContext(config=cfg, specimen_id="s", wing_outline=None)
        assert not SolidityFilter().check(ctx).passed

    def test_fragmentation_batch_mode_defers(self):
        cfg = _all_batch_mode(PipelineConfig())
        polys = [box(0, 0, 100, 100), box(300, 300, 350, 350)]  # 25% secondary blob
        ctx = FilterContext(config=cfg, specimen_id="s", all_polys=polys)
        verdict = FragmentationFilter().check(ctx)
        assert verdict.passed and verdict.method == DEFERRED_METHOD
        assert ctx.scratch["secondary_component_frac"] == pytest.approx(0.25, abs=1e-3)  # buffer rounding
        # The same wing is rejected inline in fixed mode.
        assert not FragmentationFilter().check(replace(ctx, config=PipelineConfig())).passed

    def test_fragmentation_preresolved_range_applies_inline(self):
        cfg = _all_batch_mode(PipelineConfig())
        cfg.fragmentation_batch_range = (0.0, 0.10)
        polys = [box(0, 0, 100, 100), box(300, 300, 350, 350)]
        verdict = FragmentationFilter().check(FilterContext(config=cfg, specimen_id="s", all_polys=polys))
        assert not verdict.passed
        assert "batch MAD k=5" in verdict.reason

    def test_vein_association_batch_mode_defers(self):
        cfg = _all_batch_mode(PipelineConfig())
        ctx = FilterContext(config=cfg, specimen_id="s", vein_polys=[box(0, 0, 10, 10)], veins=[])
        verdict = VeinAssociationFilter().check(ctx)  # 100% unassigned
        assert verdict.passed and verdict.method == DEFERRED_METHOD
        assert ctx.scratch["unassigned_vein_frac"] == 1.0
        assert not VeinAssociationFilter().check(replace(ctx, config=PipelineConfig())).passed

    def test_intervein_association_batch_mode_defers(self):
        cfg = _all_batch_mode(PipelineConfig())
        ctx = FilterContext(config=cfg, specimen_id="s", intervein_polys=[box(0, 0, 10, 10)], regions=[])
        verdict = InterveinAssociationFilter().check(ctx)
        assert verdict.passed and verdict.method == DEFERRED_METHOD
        assert ctx.scratch["unassigned_intervein_frac"] == 1.0

    def test_run_stage_filters_does_not_raise_when_deferred(self):
        cfg = _all_batch_mode(PipelineConfig())
        ctx = FilterContext(config=cfg, specimen_id="s", wing_outline=_notched_wing(), all_polys=[_notched_wing()])
        run_stage_filters(FilterStage.WING_OUTLINE, ctx)  # no GarbageRejection
        with pytest.raises(GarbageRejection):
            run_stage_filters(FilterStage.WING_OUTLINE, replace(ctx, config=PipelineConfig()))


# ---------------------------------------------------------------------------
# post-hoc application
# ---------------------------------------------------------------------------


class TestApplyDeferredBatchFences:
    def test_nothing_deferred_in_fixed_mode(self):
        outcome = apply_deferred_batch_fences(PipelineConfig(), _batch("wing_solidity", [0.5, 0.98, 0.99]))
        assert outcome.fences == [] and outcome.rejections == {}

    def test_outlier_rejected_with_batch_reason(self):
        cfg = _all_batch_mode(PipelineConfig(), k=5.0, min_size=5)
        vals = [0.984, 0.985, 0.986, 0.985, 0.984, 0.987, 0.986, 0.985, 0.60]
        outcome = apply_deferred_batch_fences(cfg, _batch("wing_solidity", vals))
        assert list(outcome.rejections) == ["w8"]
        verdict = outcome.rejections["w8"]
        assert verdict.filter_name == "solidity"
        assert verdict.metric_value == pytest.approx(0.60)
        assert "batch MAD k=5 (n=9)" in verdict.reason
        [fence] = [f for f in outcome.fences if f.spec is SOLIDITY_SPEC]
        assert fence.n == 9
        assert "solidity: accept [" in fence.describe()

    def test_small_batch_applies_fixed_range_post_hoc(self):
        cfg = _all_batch_mode(PipelineConfig(), min_size=25)
        outcome = apply_deferred_batch_fences(cfg, _batch("unassigned_vein_frac", [0.01, 0.02, 0.50]))
        assert list(outcome.rejections) == ["w2"]
        assert "batch of 3 < 25" in outcome.rejections["w2"].reason
        assert outcome.rejections["w2"].threshold == (0.0, cfg.max_unassigned_vein_frac)

    def test_one_sided_metric_never_rejects_low_values(self):
        cfg = _all_batch_mode(PipelineConfig(), k=5.0, min_size=3)
        # median 0.055, MAD 0.005 → upper fence ≈ 0.092; 0.0 sits far below the median but
        # a one-sided metric has no lower fence, so nothing is rejected.
        vals = [0.05, 0.05, 0.06, 0.06, 0.07, 0.0]
        outcome = apply_deferred_batch_fences(cfg, _batch("unassigned_intervein_frac", vals))
        assert outcome.rejections == {}

    def test_first_tripped_fence_wins_in_pipeline_order(self):
        cfg = _all_batch_mode(PipelineConfig(), k=2.0, min_size=3)
        wings = _batch("wing_solidity", [0.984, 0.985, 0.986, 0.985, 0.984, 0.987])
        for _stem, wr in wings:
            wr.qc_metrics["unassigned_vein_frac"] = 0.01
        bad = _wing("bad", wing_solidity=0.5, unassigned_vein_frac=0.9)
        outcome = apply_deferred_batch_fences(cfg, wings + [bad])
        assert outcome.rejections["bad"].filter_name == "solidity"

    def test_wings_without_the_metric_are_ignored(self):
        cfg = _all_batch_mode(PipelineConfig(), min_size=3)
        wings = _batch("unassigned_vein_frac", [0.01, 0.02, 0.01, 0.02, 0.9])
        wings.append(_wing("no_metric"))  # e.g. a skip_vein_tracing fast-path wing
        outcome = apply_deferred_batch_fences(cfg, wings)
        assert "no_metric" not in outcome.rejections
        assert "w4" in outcome.rejections

    def test_disabled_filter_is_not_applied(self):
        cfg = _all_batch_mode(PipelineConfig(), min_size=3)
        cfg.vein_association_filter_enabled = False
        outcome = apply_deferred_batch_fences(cfg, _batch("unassigned_vein_frac", [0.01, 0.02, 0.01, 0.9]))
        assert outcome.rejections == {}
        assert VEIN_ASSOCIATION_SPEC not in [f.spec for f in outcome.fences]

    def test_intervein_filter_respects_skip_intervein_regions(self):
        cfg = _all_batch_mode(PipelineConfig(), min_size=3)
        cfg.skip_intervein_regions = True
        assert INTERVEIN_ASSOCIATION_SPEC not in deferred_specs(cfg)

    def test_preresolved_range_is_not_reapplied(self):
        cfg = _all_batch_mode(PipelineConfig(), min_size=3)
        cfg.solidity_batch_range = (0.9, 1.0)  # a pre-pass already applied this inline
        assert SOLIDITY_SPEC not in deferred_specs(cfg)

    def test_solidity_falls_back_to_wing_solidity_attribute(self):
        cfg = _all_batch_mode(PipelineConfig(), min_size=3)
        wings = []
        for i, v in enumerate([0.984, 0.985, 0.986, 0.985, 0.984, 0.5]):
            wr = WingResult(specimen_id=f"w{i}", wing_solidity=v)  # qc_metrics left empty
            wings.append((f"w{i}", wr))
        outcome = apply_deferred_batch_fences(cfg, wings)
        assert list(outcome.rejections) == ["w5"]
