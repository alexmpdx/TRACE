"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from identify_features.models.datatypes import PruneMethod, SkeletonMethod


@dataclass
class PipelineConfig:
    """All configurable parameters for the identification pipeline.

    Distance thresholds are specified in µm and converted to pixels
    using um_per_px. If um_per_px is None, pixel values are used directly.
    """

    # -- Scale --
    um_per_px: float | None = 0.483  # Microns per pixel (0.483 for standard test data)

    # -- Skeletonization --
    skeleton_methods: list[SkeletonMethod] = field(default_factory=lambda: [SkeletonMethod.RIDGE])
    smooth_sigma: float = 2.0  # Gaussian sigma for boundary smoothing

    # -- Pruning --
    enable_basic_prune: bool = True  # Step 4 length-based branch prune; disable to keep short branches
    prune_methods: list[PruneMethod] = field(default_factory=list)  # Empty = length-based pruning only
    prune_min_length_um: float | None = None  # Minimum branch length to keep (µm; None = use median vein width)
    prune_min_length_vein_widths: float = 2.0  # Multiplier of median vein width for auto prune threshold
    final_stub_vein_widths: float = 3.0  # Final stub removal: multiplier of median vein width
    junction_merge_vein_widths: float = (
        2.0  # Tight junction merge: merge deg-2/3 nodes within this × vein width (0 = disabled)
    )
    enable_small_fragment_removal: bool = True  # Steps 11/14 small-fragment prune; disable to keep isolated components
    # Final-pass orphan-component cull: at the end of skeleton building (step 17),
    # discard any connected component whose total edge length is below this
    # fraction of the graph's combined edge length. Set to 0 to keep every
    # component; default 0.05 (= 5%) is permissive enough that disconnected
    # but legitimate vein chains (e.g. an L4/L5 island when the bridge passes
    # leave a gap) are preserved while pure noise fragments are still removed.
    min_component_edge_fraction: float = 0.05
    prune_radius_ratio_threshold: float = 0.3  # For distance-map: r_endpoint/r_junction below this = noise
    prune_scale_sigmas: list[float] = field(
        default_factory=lambda: [2.0, 4.0, 8.0, 16.0]  # For multi-scale persistence
    )
    prune_single_scale_sigma: float = 4.0  # For single-scale methods

    # -- Collinear merging --
    collinear_min_angle: float = 150.0  # Min angle (degrees) for collinear edge pairs

    # -- Gap bridging (first pass) --
    bridge_max_gap_um: float = 200.0  # Absolute max gap distance (µm)
    bridge_gap_fraction: float = 0.15  # Gap allowance as fraction of max(edge lengths)
    bridge_direction_window_um: float = 100.0  # Edge window for direction computation (µm)
    bridge_min_combined_length_um: float = 100.0  # Min total length of both edges (µm)
    bridge_on_axis_max_angle: float = 45.0  # Strict on-axis angle for longer edge (degrees)
    bridge_on_axis_relaxed_cap: float = 45.0  # Cap for relaxed angle on shorter edge (degrees)
    bridge_min_facing_angle: float = 150.0  # Min angle between opposing directions (degrees)
    bridge_direction_max_edge_fraction: float = 0.25  # Max fraction of edge length for direction window (long edges)

    # -- Gap bridging (second pass, after cleanup) --
    bridge2_max_gap_um: float = 200.0
    bridge2_gap_fraction: float = 0.5
    bridge2_min_gap_vw: float = 2.0  # Floor on adaptive gap as × median vein width
    bridge2_direction_window_um: float = 100.0
    bridge2_min_combined_length_um: float = 100.0  # Used if bridge2_min_combined_length_vw is None
    bridge2_min_combined_length_vw: float | None = 3.5  # Min combined as × median vein width (overrides _um)
    bridge2_on_axis_max_angle: float = 45.0
    bridge2_on_axis_relaxed_cap: float = 45.0
    bridge2_min_facing_angle: float = 150.0

    # -- Gap bridging (third pass, relaxed facing for short stubs) --
    bridge3_max_gap_vw: float = 4.0  # Max gap as × median vein width
    bridge3_short_edge_vw: float = 3.0  # "Short edge" threshold (× median vein width)
    bridge3_relaxed_facing_angle: float = 120.0  # Relaxed facing angle for qualifying pairs
    bridge3_direction_window_um: float = 100.0
    bridge3_on_axis_max_angle: float = 45.0
    bridge3_on_axis_relaxed_cap: float = 45.0

    # -- Landmark anchoring --
    # Snap radius is primarily expressed in µm (absolute anatomical scale).
    # The vein-width multiplier is the fallback used only when um_per_px is
    # unavailable (no scale calibration).
    snap_radius_um: float = 100.0  # Snap radius in µm (primary)
    snap_radius_vw: float = 4.0  # Fallback as × median vein width (when um_per_px is None)

    # -- Vein tracing --
    # Departure sample is primarily expressed in µm. The vein-width multiplier
    # is the fallback used only when um_per_px is unavailable.
    departure_sample_um: float = 100.0  # µm along edge to compute departure direction
    departure_sample_vw: float = 4.0  # Fallback as × median vein width (when um_per_px is None)
    tangent_continuity_max_angle: float = 90.0  # Max deflection (degrees) at junctions
    merge_max_gap_um: float = 50.0  # Max gap between line segments when merging (µm)
    distal_landmark_search_vw: float = 2.0  # Search radius for distal landmark extension (× vein width)
    # Metric used by Tier 2c (graph-reach fallback) when deciding which L4-L5
    # junction neighbor reaches a reliably-snapped soft landmark (L4.d / L5.d).
    # "path_length" (default) uses Dijkstra over edge length_px — robust to
    # chains fragmented by crossvein intersections and to short-hop detours
    # through ectopic crossveins. "hops" uses BFS for cheaper but fragile
    # topology-only counting.
    soft_landmark_reach_metric: str = "path_length"  # "path_length" | "hops"

    # -- Costa detection --
    costa_min_in_band_fraction: float = 0.5  # Edge must have ≥50% in margin band to be costa
    costa_propagation_max_distance_vw: float = 4.0  # Max distance from band for costa propagation (× vein width)

    # -- Crossvein detection --
    crossvein_min_angle: float = 40.0
    crossvein_max_length_frac: float = 0.15
    crossvein_min_length_vw: float = 4.0  # Min crossvein length as × median vein width
    crossvein_max_length_vw: float = 25.0  # Max crossvein length as × median vein width
    synthesize_missing_crossveins: bool = (
        True  # Phase 5b: draw ACV/PCV centerlines from landmarks when graph detection fails; disable to preserve fused-region output
    )

    # -- Intervein naming --
    vein_buffer_vw: float = 1.1  # Buffer radius around vein centerlines (× median vein width)
    adjacency_min_length_vw: float = 1.3  # Min shared boundary length (× median vein width)
    max_merge_size: int | None = None  # Max regions in an N-way merge; None = no cap

    # -- Intervein splitter (morphological open-under-constraint) --
    intervein_split_h_vw: float = 2.0  # h-maxima depth threshold as × median vein width
    intervein_split_reseed_min_area_um2: float = 10_000.0  # Reseed threshold for large absorbed regions
    intervein_split_vein_barrier_vw: float = 1.0  # Vein centerline buffer radius (× median vein width)
    intervein_split_wing_buffer_vw: float = 1.0  # Wing outline inset (× median vein width)

    # -- Ectopic detection --
    # Min length is primarily expressed in µm. The vein-width multiplier
    # is the fallback used only when um_per_px is unavailable.
    ectopic_min_length_um: float = 25.0  # Primary: absolute anatomical scale
    ectopic_min_length_vw: float = 1.0  # Fallback: × median vein width (when um_per_px is None)

    def to_px(self, um: float) -> float:
        """Convert µm to pixels using um_per_px."""
        if self.um_per_px is not None and self.um_per_px > 0:
            return um / self.um_per_px
        return um  # no conversion available

    def snap_radius_px(self, median_vein_width_px: float = 0.0) -> float:
        """Snap radius in pixels.

        Primary: snap_radius_um converted via um_per_px (absolute anatomical
        scale). Fallback: snap_radius_vw × median_vein_width_px, used only
        when um_per_px is unavailable (no scale calibration).
        """
        if self.um_per_px is not None and self.um_per_px > 0:
            return self.to_px(self.snap_radius_um)
        if median_vein_width_px > 0:
            return median_vein_width_px * self.snap_radius_vw
        return self.snap_radius_um  # degenerate: no scale, no vein width

    def departure_sample_px(self, median_vein_width_px: float = 0.0) -> float:
        """Departure sample distance in pixels.

        Primary: departure_sample_um converted via um_per_px. Fallback:
        departure_sample_vw × median_vein_width_px, used only when
        um_per_px is unavailable (no scale calibration).
        """
        if self.um_per_px is not None and self.um_per_px > 0:
            return self.to_px(self.departure_sample_um)
        if median_vein_width_px > 0:
            return median_vein_width_px * self.departure_sample_vw
        return self.departure_sample_um  # degenerate: no scale, no vein width

    def ectopic_min_length_px(self, median_vein_width_px: float = 0.0) -> float:
        """Ectopic-vein noise floor in pixels.

        Primary: ectopic_min_length_um converted via um_per_px. Fallback:
        ectopic_min_length_vw × median_vein_width_px, used only when
        um_per_px is unavailable (no scale calibration).
        """
        if self.um_per_px is not None and self.um_per_px > 0:
            return self.to_px(self.ectopic_min_length_um)
        if median_vein_width_px > 0:
            return median_vein_width_px * self.ectopic_min_length_vw
        return self.ectopic_min_length_um  # degenerate: no scale, no vein width


# Named pipeline presets. Each entry is a dict of PipelineConfig field overrides.
# Switching a preset replaces every field listed here; fields not listed are left
# untouched. Any PipelineConfig field is fair game — pruning, bridging, tracing,
# crossvein detection, intervein, etc. — so these dicts can grow as we capture
# more settings into a known-good combination. Each preset is a pinned snapshot,
# not a live copy of PipelineConfig defaults.
PIPELINE_PRESETS: dict[str, dict[str, Any]] = {
    "length-based": {
        # Pruning
        "enable_basic_prune": True,
        "enable_small_fragment_removal": True,
        "min_component_edge_fraction": 0.05,
        "synthesize_missing_crossveins": True,
        "prune_methods": [],
        "prune_min_length_um": None,
        "prune_min_length_vein_widths": 2.0,
        "final_stub_vein_widths": 3.0,
        "junction_merge_vein_widths": 2.0,
        "prune_radius_ratio_threshold": 0.3,
        "prune_scale_sigmas": [2.0, 4.0, 8.0, 16.0],
        "prune_single_scale_sigma": 4.0,
        "collinear_min_angle": 150.0,
        # Bridging — pass 1
        "bridge_max_gap_um": 200.0,
        "bridge_gap_fraction": 0.15,
        "bridge_direction_window_um": 100.0,
        "bridge_min_combined_length_um": 100.0,
        "bridge_on_axis_max_angle": 45.0,
        "bridge_on_axis_relaxed_cap": 45.0,
        "bridge_min_facing_angle": 150.0,
        "bridge_direction_max_edge_fraction": 0.25,
        # Bridging — pass 2
        "bridge2_max_gap_um": 200.0,
        "bridge2_gap_fraction": 0.5,
        "bridge2_min_gap_vw": 2.0,
        "bridge2_direction_window_um": 100.0,
        "bridge2_min_combined_length_um": 100.0,
        "bridge2_min_combined_length_vw": 3.5,
        "bridge2_on_axis_max_angle": 45.0,
        "bridge2_on_axis_relaxed_cap": 45.0,
        "bridge2_min_facing_angle": 150.0,
        # Bridging — pass 3
        "bridge3_max_gap_vw": 4.0,
        "bridge3_short_edge_vw": 3.0,
        "bridge3_relaxed_facing_angle": 120.0,
        "bridge3_direction_window_um": 100.0,
        "bridge3_on_axis_max_angle": 45.0,
        "bridge3_on_axis_relaxed_cap": 45.0,
    },
    # Sandbox: length-based defaults with DISTANCE_MAP layered on top.
    # Length-based pruning always runs first (step 4 in skeleton.py); the
    # methods list adds further passes. Iterate on prune_radius_ratio_threshold
    # here without disturbing the length-based preset.
    "distance-map": {
        # Pruning
        "enable_basic_prune": False,
        "enable_small_fragment_removal": False,
        "min_component_edge_fraction": 0.05,
        "synthesize_missing_crossveins": True,
        "prune_methods": [PruneMethod.DISTANCE_MAP],
        "prune_min_length_um": None,
        "prune_min_length_vein_widths": 2.0,
        "final_stub_vein_widths": 3.0,
        "junction_merge_vein_widths": 0.0,
        "prune_radius_ratio_threshold": 0.3,
        "prune_scale_sigmas": [2.0, 4.0, 8.0, 16.0],
        "prune_single_scale_sigma": 4.0,
        "collinear_min_angle": 150.0,
        # Bridging — pass 1
        "bridge_max_gap_um": 200.0,
        "bridge_gap_fraction": 0.15,
        "bridge_direction_window_um": 100.0,
        "bridge_min_combined_length_um": 100.0,
        "bridge_on_axis_max_angle": 45.0,
        "bridge_on_axis_relaxed_cap": 45.0,
        "bridge_min_facing_angle": 150.0,
        "bridge_direction_max_edge_fraction": 0.25,
        # Bridging — pass 2
        "bridge2_max_gap_um": 200.0,
        "bridge2_gap_fraction": 0.5,
        "bridge2_min_gap_vw": 2.0,
        "bridge2_direction_window_um": 100.0,
        "bridge2_min_combined_length_um": 100.0,
        "bridge2_min_combined_length_vw": 3.5,
        "bridge2_on_axis_max_angle": 45.0,
        "bridge2_on_axis_relaxed_cap": 45.0,
        "bridge2_min_facing_angle": 150.0,
        # Bridging — pass 3
        "bridge3_max_gap_vw": 4.0,
        "bridge3_short_edge_vw": 3.0,
        "bridge3_relaxed_facing_angle": 120.0,
        "bridge3_direction_window_um": 100.0,
        "bridge3_on_axis_max_angle": 45.0,
        "bridge3_on_axis_relaxed_cap": 45.0,
    },
}


def apply_preset(config: PipelineConfig, name: str) -> PipelineConfig:
    """Return a copy of `config` with preset fields overridden."""
    if name not in PIPELINE_PRESETS:
        raise KeyError(f"Unknown preset: {name!r} (available: {list(PIPELINE_PRESETS)})")
    # Copy lists so callers can't mutate the preset in place.
    updates = {k: (list(v) if isinstance(v, list) else v) for k, v in PIPELINE_PRESETS[name].items()}
    return replace(config, **updates)
