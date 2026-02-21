"""Aggregate vein measurements and apply scale calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from shapely.geometry import LineString, Polygon

from WingVeinAnalyzer.models.vein_labeler import VeinAssignment


@dataclass
class WingMeasurements:
    """All measurements for a single wing."""

    # Per-vein lengths (pixels)
    vein_lengths_px: dict[str, float] = field(default_factory=dict)
    # Per-vein lengths (microns, if scale provided)
    vein_lengths_um: dict[str, Optional[float]] = field(default_factory=dict)

    # Crossvein distance (ACV to PCV along L4)
    crossvein_distance_px: Optional[float] = None
    crossvein_distance_um: Optional[float] = None

    # Wing dimensions
    wing_length_px: Optional[float] = None
    wing_length_um: Optional[float] = None
    wing_width_px: Optional[float] = None
    wing_width_um: Optional[float] = None

    # Areas
    total_wing_area_px2: Optional[float] = None
    total_wing_area_um2: Optional[float] = None
    intervein_areas_px2: dict[str, float] = field(default_factory=dict)
    intervein_areas_um2: dict[str, Optional[float]] = field(default_factory=dict)
    anterior_compartment_area_px2: Optional[float] = None
    anterior_compartment_area_um2: Optional[float] = None
    posterior_compartment_area_px2: Optional[float] = None
    posterior_compartment_area_um2: Optional[float] = None


def compile_results(
    assignments: list[VeinAssignment],
    microns_per_pixel: float | None = None,
) -> list[VeinAssignment]:
    """Apply scale calibration to assignments and compute derived metrics."""
    if microns_per_pixel is not None:
        for a in assignments:
            a.length_um = a.length_px * microns_per_pixel
    return assignments


def compute_measurements(
    assignments: list[VeinAssignment],
    wing_polygon: Optional[Polygon] = None,
    intervein_regions: Optional[dict[str, Polygon]] = None,
    anterior_compartment: Optional[Polygon] = None,
    posterior_compartment: Optional[Polygon] = None,
    microns_per_pixel: Optional[float] = None,
) -> WingMeasurements:
    """Compute all wing measurements from vein assignments and geometry."""
    m = WingMeasurements()
    scale = microns_per_pixel
    scale2 = microns_per_pixel**2 if microns_per_pixel else None

    # Per-vein lengths
    assignment_map: dict[str, VeinAssignment] = {}
    for a in assignments:
        m.vein_lengths_px[a.vein_id] = a.length_px
        m.vein_lengths_um[a.vein_id] = a.length_px * scale if scale else None
        assignment_map[a.vein_id] = a

    # Crossvein distance (ACV to PCV along L4)
    m.crossvein_distance_px = _compute_crossvein_distance(assignment_map)
    if m.crossvein_distance_px is not None and scale:
        m.crossvein_distance_um = m.crossvein_distance_px * scale

    # Wing dimensions from bounding box of outline
    if wing_polygon and not wing_polygon.is_empty:
        m.total_wing_area_px2 = wing_polygon.area
        if scale2:
            m.total_wing_area_um2 = wing_polygon.area * scale2

        bounds = wing_polygon.bounds
        m.wing_length_px = bounds[2] - bounds[0]  # max_x - min_x
        m.wing_width_px = bounds[3] - bounds[1]   # max_y - min_y
        if scale:
            m.wing_length_um = m.wing_length_px * scale
            m.wing_width_um = m.wing_width_px * scale

    # Intervein areas
    if intervein_regions:
        for name, poly in intervein_regions.items():
            m.intervein_areas_px2[name] = poly.area
            m.intervein_areas_um2[name] = poly.area * scale2 if scale2 else None

    # Compartment areas
    if anterior_compartment and not anterior_compartment.is_empty:
        m.anterior_compartment_area_px2 = anterior_compartment.area
        if scale2:
            m.anterior_compartment_area_um2 = anterior_compartment.area * scale2
    if posterior_compartment and not posterior_compartment.is_empty:
        m.posterior_compartment_area_px2 = posterior_compartment.area
        if scale2:
            m.posterior_compartment_area_um2 = posterior_compartment.area * scale2

    return m


def _compute_crossvein_distance(
    assignment_map: dict[str, VeinAssignment],
) -> Optional[float]:
    """Compute distance between ACV and PCV along L4."""
    acv = assignment_map.get("ACV")
    pcv = assignment_map.get("PCV")
    l4 = assignment_map.get("L4")

    if not all([acv, pcv, l4]) or not all([acv.line, pcv.line, l4.line]):
        return None

    # Find intersection points of ACV and PCV with L4
    acv_on_l4 = l4.line.interpolate(l4.line.project(acv.line.centroid))
    pcv_on_l4 = l4.line.interpolate(l4.line.project(pcv.line.centroid))

    return acv_on_l4.distance(pcv_on_l4)
