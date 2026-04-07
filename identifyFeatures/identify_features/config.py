"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    prune_methods: list[PruneMethod] = field(default_factory=list)  # Empty = length-based pruning only
    prune_min_length_px: int | None = None  # Minimum branch length to keep (None = use median vein width)
    prune_min_length_vein_widths: float = 2.0  # Multiplier of median vein width for auto prune threshold
    final_stub_vein_widths: float = 3.0  # Final stub removal: multiplier of median vein width
    junction_merge_vein_widths: float = 2.0  # Tight junction merge: merge deg-2/3 nodes within this × vein width
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

    # -- Gap bridging (second pass, after cleanup) --
    bridge2_max_gap_um: float = 200.0
    bridge2_gap_fraction: float = 0.5
    bridge2_direction_window_um: float = 100.0
    bridge2_min_combined_length_um: float = 100.0
    bridge2_on_axis_max_angle: float = 45.0
    bridge2_on_axis_relaxed_cap: float = 45.0
    bridge2_min_facing_angle: float = 150.0

    # -- Landmark anchoring --
    snap_radius_um: float = 100.0  # Max distance to snap landmark to graph node (µm)
    snap_radius_px: float = 207.0  # Fallback if um_per_px not set

    # -- Vein tracing --
    departure_sample_um: float = 100.0  # µm along edge to compute departure direction
    departure_sample_px: float = 80.0  # Fallback
    tangent_continuity_max_angle: float = 90.0  # Max deflection (degrees) at junctions
    merge_max_gap_um: float = 50.0  # Max gap between line segments when merging (µm)

    # -- Costa detection --
    costa_min_in_band_fraction: float = 0.5  # Edge must have ≥50% in margin band to be costa

    # -- Crossvein detection --
    crossvein_proximity_px: float = 100.0
    crossvein_min_angle: float = 40.0
    crossvein_max_length_frac: float = 0.15
    crossvein_min_length_vw: float = 4.0  # Min crossvein length as × median vein width
    crossvein_max_length_vw: float = 25.0  # Max crossvein length as × median vein width

    # -- Intervein naming --
    vein_buffer_px: float = 25.0
    adjacency_min_length_px: float = 30.0

    # -- Ectopic detection --
    ectopic_min_length_px: float = 50.0

    def to_px(self, um: float) -> float:
        """Convert µm to pixels using um_per_px."""
        if self.um_per_px is not None and self.um_per_px > 0:
            return um / self.um_per_px
        return um  # no conversion available

    @property
    def snap_radius(self) -> float:
        """Snap radius in pixels."""
        if self.um_per_px is not None:
            return self.to_px(self.snap_radius_um)
        return self.snap_radius_px

    @property
    def departure_sample(self) -> float:
        """Departure sample distance in pixels."""
        if self.um_per_px is not None:
            return self.to_px(self.departure_sample_um)
        return self.departure_sample_px
