"""Headless tiered-cache orchestrator for the live vein-tuning preview.

`LiveTuneSession` re-runs only the identifyFeatures stages that a changed
`PipelineConfig` field actually invalidates. The stage sequence of
``identify_wing`` maps onto four cache tiers (see ../IMPLEMENTATION_SPEC.md §2):

    A  build_skeleton_graph                                  (Wing Graph params)
    B  anchor_landmarks -> compute_wing_axis ->
       trace_veins_from_landmarks -> assign_vein_tissue_polygons   (Tracing params)
    C  split_merged_intervein_polygons -> name_intervein_regions   (Intervein params)
    D  render_overlay                                        (Appearance params)

A changed field invalidates its tier *and everything downstream*. The live
update path covers A/B/D; tier C (intervein) is slow (~6x tier B) and is run
only on demand via ``compute_intervein``.

This module has no Qt dependency and is unit-testable in isolation.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import Optional

# Bound on each config->result LRU (entries hold preview-resolution arrays).
_LRU_CAP = 16

import numpy as np
from identify_features.config import PipelineConfig
from identify_features.models.datatypes import (
    InterveinRegion,
    Landmark,
    VeinIdentification,
)
from identify_features.models.intervein_namer import name_intervein_regions
from identify_features.models.intervein_splitter import (
    assign_vein_tissue_polygons,
    split_merged_intervein_polygons,
)
from identify_features.models.landmark_anchor import anchor_landmarks
from identify_features.models.skeleton import (
    _build_skeleton_core,
    _finish_skeleton_graph,
    build_skeleton_graph,
)
from identify_features.models.vein_tracer import trace_veins_from_landmarks
from identify_features.models.wing_axis import compute_wing_axis
from identify_features.views.overlay import render_overlay
from shapely.geometry import MultiPolygon, Polygon

from .preview_render import render_skeleton, render_traced

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier labels. Ordering for the live cascade (A < B < D) is in _LIVE_ORDER;
# tier C is handled out-of-band (compute_intervein) and is NOT in the cascade.
# ---------------------------------------------------------------------------
TIER_A = "A"  # skeleton  (expensive ~1s)
TIER_B = "B"  # trace     (~300ms, the live target)
TIER_C = "C"  # intervein (~1.8s, manual refresh only)
TIER_D = "D"  # render    (~25ms, instant)

_LIVE_ORDER = {TIER_A: 0, TIER_B: 1, TIER_D: 2}

# ---------------------------------------------------------------------------
# View modes. Each names an intermediate (or final) pipeline product to show,
# and — crucially — the deepest tier its render needs:
#   SKELETON  needs only Tier A (no tracing) → Wing Graph tuning skips Tier B.
#   TRACED    needs Tier B (veins + snapped landmarks), no intervein.
#   FINAL     needs Tier B (+ on-demand Tier C intervein), the original overlay.
# ---------------------------------------------------------------------------
VIEW_SKELETON = "skeleton"
VIEW_TRACED = "traced"
VIEW_FINAL = "final"

# Deepest live tier each view needs to recompute through.
_VIEW_MAX_TIER = {
    VIEW_SKELETON: _LIVE_ORDER[TIER_A],
    VIEW_TRACED: _LIVE_ORDER[TIER_B],
    VIEW_FINAL: _LIVE_ORDER[TIER_B],
}

# ---------------------------------------------------------------------------
# FIELD_TIER: every PipelineConfig field -> the lowest tier it invalidates.
# Built from prefix/membership rules, then validated at import time against the
# live dataclass so a newly-added config field can never silently fall through.
# ---------------------------------------------------------------------------

# Tier B: landmark anchoring + vein tracing + costa/crossvein/ectopic detection.
_TIER_B_FIELDS = {
    "snap_radius_um",
    "snap_radius_vw",
    "departure_sample_um",
    "departure_sample_vw",
    "tangent_continuity_max_angle",
    "merge_max_gap_um",
    "distal_landmark_search_vw",
    "soft_landmark_reach_metric",
    "costa_min_in_band_fraction",
    "costa_propagation_max_distance_vw",
    "crossvein_min_angle",
    "crossvein_max_length_frac",
    "crossvein_min_length_vw",
    "crossvein_max_length_vw",
    "synthesize_missing_crossveins",
    "ectopic_min_length_um",
    "ectopic_min_length_vw",
    # Used only by vein_tracer._assign_absent_and_partial (a trace phase), so
    # they belong to Tier B, not the skeleton. (Previously defaulted to Tier A
    # because they weren't listed here — corrected while wiring the core/finish
    # split, which is what surfaced it.)
    "assign_absent_partial_status",
    "partial_endpoint_search_vw",
}

# Tier C: intervein splitting + naming (and the master skip toggle).
_TIER_C_FIELDS = {
    "skip_intervein_regions",
    "vein_buffer_vw",
    "adjacency_min_length_vw",
    "max_merge_size",
    "intervein_split_h_vw",
    "intervein_split_reseed_min_area_um2",
    "intervein_split_vein_barrier_vw",
    "intervein_split_wing_buffer_vw",
}

# Tier D: render-time appearance only (passed straight to render_overlay).
APPEARANCE_FIELDS = {
    "vein_opacity",
    "intervein_opacity",
    "vein_colors",
    "region_colors",
}

# Fields that don't affect ANY stage the live preview renders. Routed to
# Tier D so the assertion at import time (which requires FIELD_TIER to
# cover every PipelineConfig field) passes without lying about which
# fields feed the skeleton — CORE_FIELDS + FINISH_FIELDS below partitions
# only the real skeleton-relevant Tier-A set.
#
# Membership rationale:
#   - auto_detect_um_per_px    — controls how *the pipeline* reads scale
#                                from TIFF metadata at the top of Stage 1;
#                                the preview receives whatever um_per_px
#                                is already on the config, no re-read.
#   - solidity_*, fragmentation_*, vein_association_*,
#     intervein_association_*, required_veins,
#     max_unassigned_vein_frac, max_unassigned_intervein_frac —
#                                garbage-detector filters; they abort a
#                                run at pipeline hooks but never change
#                                any pixel or geometry the preview draws.
_INERT_FIELDS = {
    "auto_detect_um_per_px",
    # Tier-2 fast path (added in v0.2.22 as PipelineConfig.skip_vein_tracing).
    # Gates whether identify_wing runs Steps 2-6 at all — but the live
    # preview always renders the full pipeline output, so from the
    # preview's perspective this flag is inert. Missing entry here caused
    # the v0.2.22–v0.2.24 startup AssertionError
    # ("CORE/FINISH must partition Tier A: missing={'skip_vein_tracing'}").
    "skip_vein_tracing",
    # Solidity filter
    "solidity_filter_enabled",
    "solidity_min",
    "solidity_max",
    "solidity_mode",
    "solidity_batch_k",
    "solidity_min_batch_size",
    "solidity_batch_range",
    # Fragmentation filter
    "fragmentation_filter_enabled",
    "fragmentation_max_secondary_frac",
    # Vein-association filter + required-veins list
    "vein_association_filter_enabled",
    "max_unassigned_vein_frac",
    "required_veins",
    # Intervein-association filter (added 2026-07-22 in
    # identify_features/garbage_detector/filters/intervein_association.py;
    # commit 38843e8f). Aborts wings with excess unassigned intervein
    # tissue — same shape as the vein-association filter.
    "intervein_association_filter_enabled",
    "max_unassigned_intervein_frac",
}


def _build_field_tier() -> dict[str, str]:
    """Map each PipelineConfig field to its tier; everything else is tier A.

    Anything not explicitly B/C/D (or _INERT_FIELDS, routed to D) feeds
    skeleton construction (or scale, which feeds every um->px conversion
    upstream), so the safe default is tier A.
    """
    tier: dict[str, str] = {}
    for f in fields(PipelineConfig):
        name = f.name
        if name in _TIER_B_FIELDS:
            tier[name] = TIER_B
        elif name in _TIER_C_FIELDS:
            tier[name] = TIER_C
        elif name in APPEARANCE_FIELDS or name in _INERT_FIELDS:
            tier[name] = TIER_D
        else:
            tier[name] = TIER_A
    return tier


FIELD_TIER: dict[str, str] = _build_field_tier()

# Fail loudly at import if the dataclass and our partition ever disagree.
_ALL_FIELDS = {f.name for f in fields(PipelineConfig)}
assert set(FIELD_TIER) == _ALL_FIELDS, (
    "FIELD_TIER out of sync with PipelineConfig: "
    f"missing={_ALL_FIELDS - set(FIELD_TIER)} extra={set(FIELD_TIER) - _ALL_FIELDS}"
)

# ---------------------------------------------------------------------------
# Tier-A sub-partition: CORE vs FINISH (matches identifyFeatures' skeleton split).
# build_skeleton_graph was split into _build_skeleton_core (expensive: rasterize
# + skeletonize + prune, ~60-75% of the cost) and _finish_skeleton_graph (cheap:
# graph cleanup / bridging / merge / cull). A change to a FINISH-only field can
# reuse the cached core and re-run only the cheap finish.
#   CORE   = fields feeding the expensive half (skeletonization + scale + prune).
#   FINISH = fields feeding only the cheap graph-cleanup half.
# Every Tier-A field is exactly one of these (asserted below).
# ---------------------------------------------------------------------------
CORE_FIELDS = {
    "um_per_px",  # feeds every um->px conversion, including prune thresholds (core)
    "skeleton_methods",
    "smooth_sigma",
    "enable_basic_prune",
    "prune_methods",
    "prune_min_length_um",
    "prune_min_length_vein_widths",
    "prune_radius_ratio_threshold",
    "prune_scale_sigmas",
    "prune_single_scale_sigma",
}

FINISH_FIELDS = {
    "collinear_min_angle",
    "junction_merge_vein_widths",
    "final_stub_vein_widths",
    "enable_small_fragment_removal",
    "min_component_edge_fraction",
    # Bridging pass 1
    "bridge_max_gap_um",
    "bridge_gap_fraction",
    "bridge_direction_window_um",
    "bridge_min_combined_length_um",
    "bridge_on_axis_max_angle",
    "bridge_on_axis_relaxed_cap",
    "bridge_min_facing_angle",
    "bridge_direction_max_edge_fraction",  # Tier-A but unused by builder; finish is harmless
    # Bridging pass 2
    "bridge2_max_gap_um",
    "bridge2_gap_fraction",
    "bridge2_min_gap_vw",
    "bridge2_direction_window_um",
    "bridge2_min_combined_length_um",
    "bridge2_min_combined_length_vw",
    "bridge2_on_axis_max_angle",
    "bridge2_on_axis_relaxed_cap",
    "bridge2_min_facing_angle",
    # Bridging pass 3
    "bridge3_max_gap_vw",
    "bridge3_short_edge_vw",
    "bridge3_relaxed_facing_angle",
    "bridge3_direction_window_um",
    "bridge3_on_axis_max_angle",
    "bridge3_on_axis_relaxed_cap",
}

# CORE_FIELDS and FINISH_FIELDS must exactly partition the Tier-A fields, so a
# newly added Tier-A config field forces an explicit bucket choice (fail-loud).
_TIER_A_FIELDS = {n for n, t in FIELD_TIER.items() if t == TIER_A}
assert CORE_FIELDS | FINISH_FIELDS == _TIER_A_FIELDS, (
    "CORE/FINISH must partition Tier A: "
    f"missing={_TIER_A_FIELDS - (CORE_FIELDS | FINISH_FIELDS)} "
    f"extra={(CORE_FIELDS | FINISH_FIELDS) - _TIER_A_FIELDS}"
)
assert not (CORE_FIELDS & FINISH_FIELDS), f"CORE/FINISH overlap: {CORE_FIELDS & FINISH_FIELDS}"


@dataclass
class Appearance:
    """UI-only render flags (not part of PipelineConfig)."""

    show_veins: bool = True
    show_regions: bool = True
    show_vein_tissue: bool = False


@dataclass
class RenderResult:
    """Outcome of a ``LiveTuneSession.update`` call."""

    overlay_bgr: Optional[np.ndarray]
    tier_ran: str  # "A" | "B" | "D" | "none"
    timings_ms: dict = field(default_factory=dict)
    n_veins: int = 0
    regions_stale: bool = False
    error: Optional[str] = None


def _changed_fields(old: Optional[PipelineConfig], new: PipelineConfig) -> set[str]:
    if old is None:
        return set(FIELD_TIER)
    return {name for name in FIELD_TIER if getattr(old, name) != getattr(new, name)}


class LiveTuneSession:
    """Tiered-cache orchestrator for one input wing.

    Call :meth:`set_input` once per sample, then :meth:`update` on each
    parameter change. :meth:`compute_intervein` runs the slow tier C on demand.
    """

    def __init__(self) -> None:
        # Pristine S1 inputs (never mutated after set_input).
        self._base_image: Optional[np.ndarray] = None
        self._vein_polys: list = []
        self._intervein_polys: list = []
        self._landmarks_raw: dict[str, Landmark] = {}
        self._wing_outline: Optional[Polygon] = None
        self._image_shape: Optional[tuple[int, int]] = None

        # Tier caches.
        self._skel_core = None  # Tier A-core: expensive half (rasterize/skeletonize/prune)
        self._pristine_skel = None  # Tier A-finish output; NEVER anchored in place
        self._anchored_skel = None  # Tier B working copy (landmark nodes inserted)
        self._veins: list[VeinIdentification] = []
        self._anchored_landmarks: dict[str, Landmark] = {}
        self._wing_axis = None
        self._regions: list[InterveinRegion] = []
        self._regions_stale: bool = True

        # Config -> result LRUs so revisiting a previously-seen config returns
        # without recompute. Keyed on the EFFECTIVE config's field signatures,
        # prefixed by an input epoch so a prior wing's caches never collide.
        # _core_lru: core-field signature -> _SkeletonCore.
        # _skel_lru: core+finish signature -> finished SkeletonGraph.
        self._core_lru: "OrderedDict[tuple, object]" = OrderedDict()
        self._skel_lru: "OrderedDict[tuple, object]" = OrderedDict()
        self._input_epoch: int = 0
        # Veins are stale relative to the current config — set when a Tier-A/B
        # param changed but the active view didn't need tracing (skeleton view),
        # so the work was deferred. Switching to a tracing view recomputes them.
        self._veins_dirty: bool = True

        self._last_config: Optional[PipelineConfig] = None
        self._last_appearance: Optional[Appearance] = None
        self._last_overlay: Optional[np.ndarray] = None

        # Reduced-resolution preview factor. The image + polygons fed to
        # set_input are already downscaled by this factor (1.0 = full res);
        # we only need it here to rescale um_per_px so micron-based thresholds
        # stay consistent — see _effective().
        self._preview_scale: float = 1.0

    # -- input ------------------------------------------------------------
    def set_input(
        self,
        base_image_bgr: np.ndarray,
        vein_polys: list,
        intervein_polys: list,
        landmarks_raw: dict[str, Landmark],
        wing_outline: Optional[Polygon],
        image_shape: tuple[int, int],
        preview_scale: float = 1.0,
    ) -> None:
        """Store parsed S1 results (pristine) and clear all tier caches.

        ``base_image_bgr``, ``vein_polys``, ``intervein_polys``, ``landmarks_raw``
        and ``wing_outline`` are expected to already be downscaled by
        ``preview_scale`` (1.0 = full resolution). ``preview_scale`` is retained
        only so :meth:`_effective` can divide ``um_per_px`` to match.
        """
        self._base_image = base_image_bgr
        self._vein_polys = vein_polys
        self._intervein_polys = intervein_polys
        self._landmarks_raw = landmarks_raw
        self._wing_outline = wing_outline
        self._image_shape = image_shape
        self._preview_scale = preview_scale if preview_scale and preview_scale > 0 else 1.0
        self._invalidate_all()

    def _invalidate_all(self) -> None:
        self._skel_core = None
        self._pristine_skel = None
        self._anchored_skel = None
        self._veins = []
        self._anchored_landmarks = {}
        self._wing_axis = None
        self._regions = []
        self._regions_stale = True
        self._veins_dirty = True
        self._last_config = None
        self._last_appearance = None
        self._last_overlay = None
        # A new input wing invalidates every cached skeleton; free the memory and
        # bump the epoch so any stragglers can never key-collide with the new wing.
        self._core_lru.clear()
        self._skel_lru.clear()
        self._input_epoch += 1

    @property
    def has_input(self) -> bool:
        return self._base_image is not None

    @property
    def median_vein_width_px(self) -> Optional[float]:
        return None if self._pristine_skel is None else self._pristine_skel.median_vein_width_px

    def _effective(self, config: PipelineConfig) -> PipelineConfig:
        """Config adjusted for the active preview scale.

        When the preview runs at reduced resolution, the geometry handed to the
        stages is already downscaled by ``preview_scale``. Micron-based
        thresholds convert to pixels through ``um_per_px`` (px = um / um_per_px),
        so dividing ``um_per_px`` by ``preview_scale`` shrinks every pixel
        threshold by the same factor as the image — keeping behaviour matched to
        full resolution. Vein-width-relative thresholds need no adjustment: they
        scale automatically with the (smaller) median vein width measured from
        the downscaled skeleton.
        """
        if self._preview_scale == 1.0 or config.um_per_px is None:
            return config
        return replace(config, um_per_px=config.um_per_px / self._preview_scale)

    # -- LRU keys (built from the EFFECTIVE config; epoch-prefixed) -------
    @staticmethod
    def _sig(cfg: PipelineConfig, names) -> tuple:
        """Hashable signature of the given config fields (lists -> tuples)."""
        out = []
        for name in sorted(names):
            v = getattr(cfg, name)
            out.append(tuple(v) if isinstance(v, list) else v)
        return tuple(out)

    def _core_key(self, cfg: PipelineConfig) -> tuple:
        return (self._input_epoch, "core", self._sig(cfg, CORE_FIELDS))

    def _skel_key(self, cfg: PipelineConfig) -> tuple:
        return (self._input_epoch, "skel", self._sig(cfg, CORE_FIELDS | FINISH_FIELDS))

    @staticmethod
    def _lru_put(lru: "OrderedDict", key, value) -> None:
        lru[key] = value
        lru.move_to_end(key)
        while len(lru) > _LRU_CAP:
            lru.popitem(last=False)

    @staticmethod
    def _lru_get(lru: "OrderedDict", key):
        if key in lru:
            lru.move_to_end(key)
            return lru[key]
        return None

    # -- live update (tiers A/B/D) ---------------------------------------
    def update(
        self,
        config: PipelineConfig,
        appearance: Optional[Appearance] = None,
        view: str = VIEW_FINAL,
    ) -> RenderResult:
        """Recompute the minimal tiers the ``view`` needs and render it.

        ``view`` caps how deep recomputation goes: the skeleton view stops at
        Tier A (no tracing), so Wing-Graph tuning never pays the Tier-B cost.
        A Tier-B parameter change while a skeleton view is active is *deferred*
        (veins marked dirty) and applied lazily when a tracing view is shown.

        Tier C (intervein) is never run here. On any Tier A/B recompute the
        cached intervein regions become stale; the result flags that so the UI
        can prompt for a manual refresh. On a stage exception the previous
        caches and overlay are preserved and the error is returned.
        """
        if not self.has_input:
            return RenderResult(overlay_bgr=None, tier_ran="none", error="No input loaded")
        if appearance is None:
            appearance = self._last_appearance or Appearance()
        view_cap = _VIEW_MAX_TIER.get(view, _LIVE_ORDER[TIER_B])

        changed = _changed_fields(self._last_config, config)
        live_changed = [c for c in changed if FIELD_TIER[c] in _LIVE_ORDER]
        c_changed = [c for c in changed if FIELD_TIER[c] == TIER_C]
        appearance_changed = appearance != self._last_appearance

        # Decide the lowest live tier the changes invalidate.
        need: Optional[int] = None
        if self._pristine_skel is None:
            need = _LIVE_ORDER[TIER_A]  # first run
        if live_changed:
            lo = min(_LIVE_ORDER[FIELD_TIER[c]] for c in live_changed)
            need = lo if need is None else min(need, lo)
        if appearance_changed and need is None:
            need = _LIVE_ORDER[TIER_D]

        timings: dict = {}
        try:
            cfg = self._effective(config)
            # Skeleton (Tier A) recompute — needed by every view when the graph
            # is invalid. A new skeleton also invalidates the veins built on it.
            # The skeleton is split into a CACHEABLE expensive core (steps 1-7)
            # and a CHEAP finish (steps 8-17): a finish-only param change reuses
            # the cached core and re-runs only the finish.
            if need is not None and need <= _LIVE_ORDER[TIER_A]:
                self._recompute_skeleton_tier(cfg, changed, timings)
                self._veins_dirty = True
                self._regions_stale = True
            elif need is not None and need <= _LIVE_ORDER[TIER_B]:
                # A Tier-B param changed: veins are stale regardless of view.
                self._veins_dirty = True
                self._regions_stale = True
            elif c_changed:
                self._regions_stale = True

            # Run the deferred Tier-B trace only if this view needs it AND the
            # veins are dirty. The skeleton view skips this entirely.
            if view_cap >= _LIVE_ORDER[TIER_B] and self._veins_dirty and self._pristine_skel is not None:
                self._recompute_veins(cfg, timings)
                self._veins_dirty = False
                self._regions_stale = True

            tier_ran = self._tier_label(need, c_changed)
            overlay = self._render(cfg, appearance, view, timings)
            self._last_config = deepcopy(config)
            self._last_appearance = deepcopy(appearance)
            self._last_overlay = overlay
            return RenderResult(
                overlay_bgr=overlay,
                tier_ran=tier_ran,
                timings_ms=timings,
                n_veins=sum(1 for v in self._veins if v.centerline is not None),
                regions_stale=self._regions_stale,
            )
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            logger.exception("Live update failed")
            return RenderResult(
                overlay_bgr=self._last_overlay,
                tier_ran="error",
                timings_ms=timings,
                n_veins=sum(1 for v in self._veins if v.centerline is not None),
                regions_stale=self._regions_stale,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _tier_label(self, need: Optional[int], c_changed: list) -> str:
        if need is None:
            return TIER_C if c_changed else "none"
        for label, order in _LIVE_ORDER.items():
            if order == need:
                return label
        return "none"

    # -- tier implementations --------------------------------------------
    def _recompute_skeleton_tier(
        self, cfg: PipelineConfig, changed: set, timings: dict
    ) -> None:
        """Recompute the skeleton with the minimal work the change requires.

        Decision order:
          1. Full-skeleton LRU hit (this exact core+finish config seen before)
             → restore the finished skeleton instantly, skip all compute.
          2. First run or a CORE-field changed → (maybe LRU-hit the) core, then finish.
          3. Only FINISH-field(s) changed → reuse the cached core, re-run finish.
        ``changed`` is the set of changed field names from this update().
        """
        first_run = self._skel_core is None

        # 1. Instant revisit: the whole finished skeleton was cached for this config.
        if not first_run:
            skel_hit = self._lru_get(self._skel_lru, self._skel_key(cfg))
            if skel_hit is not None:
                self._pristine_skel = skel_hit
                # Restore the matching core so a later finish-only edit can resume.
                core_hit = self._lru_get(self._core_lru, self._core_key(cfg))
                if core_hit is not None:
                    self._skel_core = core_hit
                timings["A_cached"] = 0.0
                return

        core_changed = any(c in CORE_FIELDS for c in changed)
        finish_changed = any(c in FINISH_FIELDS for c in changed)

        if first_run or core_changed:
            self._recompute_skeleton_core(cfg, timings)
            self._recompute_skeleton_finish(cfg, timings)
        elif finish_changed:
            # The cheap win: reuse the cached expensive core, re-run only finish.
            self._recompute_skeleton_finish(cfg, timings)
        else:
            # No core/finish field changed but Tier A was requested anyway
            # (e.g. a stale pristine skeleton). Rebuild defensively from the core.
            self._recompute_skeleton_finish(cfg, timings)

    def _recompute_skeleton_core(self, cfg: PipelineConfig, timings: dict) -> None:
        """Tier A-core: the expensive half (rasterize + skeletonize + prune).

        Probes the core LRU first; on a miss, builds and caches the core.
        """
        key = self._core_key(cfg)
        hit = self._lru_get(self._core_lru, key)
        if hit is not None:
            self._skel_core = hit
            timings["A_core_cached"] = 0.0
            return
        t0 = time.perf_counter()
        self._skel_core = _build_skeleton_core(self._vein_polys, self._image_shape, cfg)
        timings["A_core"] = (time.perf_counter() - t0) * 1000
        self._lru_put(self._core_lru, key, self._skel_core)

    def _recompute_skeleton_finish(self, cfg: PipelineConfig, timings: dict) -> None:
        """Tier A-finish: the cheap half (graph cleanup / bridging / merge / cull).

        Reuses the cached core. ``_finish_skeleton_graph`` deep-copies the core's
        raw graph internally, so the same core can drive many finish re-runs.
        """
        if self._skel_core is None:
            raise RuntimeError("finish requested before core built")
        t0 = time.perf_counter()
        self._pristine_skel = _finish_skeleton_graph(self._skel_core, cfg)
        timings["A_finish"] = (time.perf_counter() - t0) * 1000
        self._lru_put(self._skel_lru, self._skel_key(cfg), self._pristine_skel)

    def _recompute_veins(self, cfg: PipelineConfig, timings: dict) -> None:
        if self._pristine_skel is None:
            raise RuntimeError("Tier B requested before Tier A skeleton built")
        t0 = time.perf_counter()
        # CRITICAL: anchor_landmarks mutates skel + landmarks in place; always
        # work from pristine deepcopies so repeated runs are idempotent.
        skel = deepcopy(self._pristine_skel)
        lms = deepcopy(self._landmarks_raw)
        anchor_landmarks(skel, lms, cfg)
        axis = compute_wing_axis(lms)
        veins = trace_veins_from_landmarks(skel, lms, self._wing_outline, cfg, wing_axis=axis)
        assign_vein_tissue_polygons(veins, skel.median_vein_width_px, cfg, self._wing_outline)
        self._veins = veins
        self._anchored_landmarks = lms
        self._wing_axis = axis
        # Keep the anchored graph: landmark snapped_node indexes into THIS graph
        # (not the pristine one), so the traced view needs it to mark landmarks.
        self._anchored_skel = skel
        timings["B_trace"] = (time.perf_counter() - t0) * 1000

    def _render(
        self, cfg: PipelineConfig, appearance: Appearance, view: str, timings: dict
    ) -> np.ndarray:
        t0 = time.perf_counter()
        if view == VIEW_SKELETON:
            overlay = render_skeleton(self._base_image, self._pristine_skel)
        elif view == VIEW_TRACED:
            overlay = render_traced(
                self._base_image,
                self._veins,
                self._anchored_landmarks,
                self._anchored_skel,
                vein_color_overrides=cfg.vein_colors,
                vein_opacity=cfg.vein_opacity,
            )
        else:  # VIEW_FINAL
            regions = self._regions if (appearance.show_regions and not self._regions_stale) else []
            overlay = render_overlay(
                self._base_image,
                self._veins,
                regions,
                show_vein_tissue=appearance.show_vein_tissue,
                show_veins=appearance.show_veins,
                show_regions=appearance.show_regions,
                vein_color_overrides=cfg.vein_colors,
                region_color_overrides=cfg.region_colors,
                vein_opacity=cfg.vein_opacity,
                intervein_opacity=cfg.intervein_opacity,
                # The live preview shows a static UI-side legend beside the image,
                # so suppress the in-image key (it would occlude the wing and waste
                # the limited preview canvas). Batch output keeps its baked-in key.
                show_color_key=False,
            )
        timings["D_render"] = (time.perf_counter() - t0) * 1000
        return overlay

    # -- on-demand tier C -------------------------------------------------
    def compute_intervein(self, config: PipelineConfig) -> list[InterveinRegion]:
        """Run the slow intervein tier from the cached veins.

        Requires a prior :meth:`update` to have produced veins. Result is cached
        and marked fresh; call :meth:`update` afterward (or use the returned
        regions) to render them.
        """
        if self._pristine_skel is None or not self._veins:
            raise RuntimeError("compute_intervein requires veins from a prior update()")
        if config.skip_intervein_regions:
            self._regions = []
            self._regions_stale = False
            return []
        mvw = self._pristine_skel.median_vein_width_px
        cfg = self._effective(config)
        intervein_polys: list[Polygon | MultiPolygon] = split_merged_intervein_polygons(
            self._intervein_polys,
            self._veins,
            self._wing_outline,
            self._image_shape,
            mvw,
            cfg,
        )
        regions = name_intervein_regions(
            intervein_polys,
            self._veins,
            self._anchored_landmarks,
            cfg,
            mvw,
            self._wing_outline,
            self._wing_axis,
        )
        self._regions = regions
        self._regions_stale = False
        return regions

    def render_current(
        self, config: PipelineConfig, appearance: Appearance, view: str = VIEW_FINAL
    ) -> np.ndarray:
        """Re-render with current caches (used after compute_intervein)."""
        return self._render(self._effective(config), appearance, view, {})
