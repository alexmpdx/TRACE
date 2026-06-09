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
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from typing import Optional

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
from identify_features.models.skeleton import build_skeleton_graph
from identify_features.models.vein_tracer import trace_veins_from_landmarks
from identify_features.models.wing_axis import compute_wing_axis
from identify_features.views.overlay import render_overlay
from shapely.geometry import MultiPolygon, Polygon

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


def _build_field_tier() -> dict[str, str]:
    """Map each PipelineConfig field to its tier; everything else is tier A.

    Anything not explicitly B/C/D feeds skeleton construction (or scale, which
    feeds every um->px conversion upstream), so the safe default is tier A.
    """
    tier: dict[str, str] = {}
    for f in fields(PipelineConfig):
        name = f.name
        if name in _TIER_B_FIELDS:
            tier[name] = TIER_B
        elif name in _TIER_C_FIELDS:
            tier[name] = TIER_C
        elif name in APPEARANCE_FIELDS:
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
        self._pristine_skel = None  # Tier A output; NEVER anchored in place
        self._veins: list[VeinIdentification] = []
        self._anchored_landmarks: dict[str, Landmark] = {}
        self._wing_axis = None
        self._regions: list[InterveinRegion] = []
        self._regions_stale: bool = True

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
        self._pristine_skel = None
        self._veins = []
        self._anchored_landmarks = {}
        self._wing_axis = None
        self._regions = []
        self._regions_stale = True
        self._last_config = None
        self._last_appearance = None
        self._last_overlay = None

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

    # -- live update (tiers A/B/D) ---------------------------------------
    def update(self, config: PipelineConfig, appearance: Optional[Appearance] = None) -> RenderResult:
        """Recompute from the lowest invalidated tier (A/B/D) and re-render.

        Tier C (intervein) is never run here. On any tier A/B recompute the
        cached intervein regions become stale; the result flags that so the UI
        can prompt for a manual refresh. On a stage exception the previous
        caches and overlay are preserved and the error is returned.
        """
        if not self.has_input:
            return RenderResult(overlay_bgr=None, tier_ran="none", error="No input loaded")
        if appearance is None:
            appearance = self._last_appearance or Appearance()

        changed = _changed_fields(self._last_config, config)
        live_changed = [c for c in changed if FIELD_TIER[c] in _LIVE_ORDER]
        c_changed = [c for c in changed if FIELD_TIER[c] == TIER_C]
        appearance_changed = appearance != self._last_appearance

        # Decide the lowest live tier to recompute from.
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
            if need is not None and need <= _LIVE_ORDER[TIER_A]:
                self._recompute_skeleton(cfg, timings)
                self._recompute_veins(cfg, timings)
                self._regions_stale = True
            elif need is not None and need <= _LIVE_ORDER[TIER_B]:
                self._recompute_veins(cfg, timings)
                self._regions_stale = True
            elif c_changed:
                # Only intervein params moved: veins unchanged, just mark stale.
                self._regions_stale = True

            tier_ran = self._tier_label(need, c_changed)
            overlay = self._render(cfg, appearance, timings)
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
    def _recompute_skeleton(self, cfg: PipelineConfig, timings: dict) -> None:
        t0 = time.perf_counter()
        # Fresh polygons each build; build_skeleton_graph rasterizes from them.
        self._pristine_skel = build_skeleton_graph(self._vein_polys, self._image_shape, cfg)
        timings["A_skeleton"] = (time.perf_counter() - t0) * 1000

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
        timings["B_trace"] = (time.perf_counter() - t0) * 1000

    def _render(self, cfg: PipelineConfig, appearance: Appearance, timings: dict) -> np.ndarray:
        t0 = time.perf_counter()
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

    def render_current(self, config: PipelineConfig, appearance: Appearance) -> np.ndarray:
        """Re-render with current caches (used after compute_intervein)."""
        return self._render(config, appearance, {})
