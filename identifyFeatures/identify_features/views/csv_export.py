"""Export vein and region measurements as CSV.

Two formats:
- Long format (single mode): one row per feature, all columns
- Wide format (batch mode): one row per specimen, measurements as columns
"""

from __future__ import annotations

import csv
import logging
import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from identify_features.models.datatypes import InterveinRegion, VeinIdentification
from identify_features.models.topology import REGION_AP_ORDER, VEIN_AP_ORDER
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import split

if TYPE_CHECKING:
    from identify_features.models.datatypes import Landmark, WingResult

logger = logging.getLogger(__name__)

# Display name mapping (internal id → CSV column name) for wide format
_VEIN_DISPLAY = {v: ("costal vein" if v == "costa" else v) for v in VEIN_AP_ORDER}

NOT_IDENTIFIED = "not identified"


def _compute_ap_areas(
    wing_result: Optional[WingResult],
) -> tuple[Optional[float], Optional[float]]:
    """Split wing into anterior/posterior compartments along L4 axis.

    Algorithm:
    1. Get L4 centerline and its minimum rotated bounding box.
    2. Take the anterior long edge (closer to L3) of the bounding box.
    3. Extend that edge to bisect the entire wing outline.
    4. Split the wing; label halves by proximity to L3 (anterior) vs L5 (posterior).

    Returns (anterior_area_px, posterior_area_px) or (None, None).
    """
    if wing_result is None or wing_result.wing_outline is None:
        return None, None

    # Find L4 and L3 centerlines
    veins_by_id = {v.vein_id: v for v in wing_result.veins}
    l4 = veins_by_id.get("L4")
    if l4 is None or l4.centerline is None:
        return None, None

    l4_line = l4.centerline
    outline = wing_result.wing_outline

    # Step 1: minimum rotated bounding box around L4 centerline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bbox = l4_line.minimum_rotated_rectangle
    coords = list(bbox.exterior.coords)[:4]  # 4 corners

    # Step 2: extract 4 edges and find the 2 longest
    edges = []
    for i in range(4):
        p1, p2 = np.array(coords[i]), np.array(coords[(i + 1) % 4])
        length = np.linalg.norm(p2 - p1)
        midpoint = (p1 + p2) / 2
        edges.append((p1, p2, length, midpoint))

    edges.sort(key=lambda e: e[2], reverse=True)
    long_edges = edges[:2]  # the two longest edges

    # Step 3: pick the anterior edge (closer to L3 centroid)
    # Fall back to L2 or L1 if L3 is missing
    anterior_ref = None
    for vid in ("L3", "L2", "L1"):
        v = veins_by_id.get(vid)
        if v is not None and v.centerline is not None:
            anterior_ref = np.array(v.centerline.centroid.coords[0])
            break

    if anterior_ref is None:
        return None, None

    # Pick the long edge whose midpoint is closer to the anterior reference
    d0 = np.linalg.norm(long_edges[0][3] - anterior_ref)
    d1 = np.linalg.norm(long_edges[1][3] - anterior_ref)
    ant_edge = long_edges[0] if d0 < d1 else long_edges[1]

    # Step 4: extend the anterior edge to fully bisect the wing
    p1, p2 = ant_edge[0], ant_edge[1]
    direction = p2 - p1
    direction = direction / np.linalg.norm(direction)

    # Shift split line by one median L4 vein width toward anterior
    median_width = 0.0
    if l4.tissue_polygon is not None and l4_line.length > 0:
        median_width = l4.tissue_polygon.area / l4_line.length
    if median_width > 0:
        perp_a = np.array([-direction[1], direction[0]])
        perp_b = np.array([direction[1], -direction[0]])
        midpt = (p1 + p2) / 2
        if np.linalg.norm(midpt + perp_a * 10 - anterior_ref) < np.linalg.norm(midpt + perp_b * 10 - anterior_ref):
            perp = perp_a
        else:
            perp = perp_b
        # Only shift the DTip (distal) end
        landmarks = wing_result.landmarks if wing_result else {}
        dtip = landmarks.get("DTip")
        dtip_pt = np.array([dtip.x, dtip.y]) if dtip else np.array(l4_line.coords[-1])
        if np.linalg.norm(p1 - dtip_pt) < np.linalg.norm(p2 - dtip_pt):
            p1 = p1 + perp * median_width * 0.5
        else:
            p2 = p2 + perp * median_width * 0.5

    # Extend by 2x the wing bounding box diagonal in each direction
    minx, miny, maxx, maxy = outline.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    extend = diag * 2

    midpt = (p1 + p2) / 2
    ext_p1 = midpt - direction * extend
    ext_p2 = midpt + direction * extend
    split_line = LineString([ext_p1.tolist(), ext_p2.tolist()])

    # Step 5: split the wing outline
    try:
        parts = split(outline.buffer(0), split_line)
        polygons = [g for g in parts.geoms if isinstance(g, (Polygon, MultiPolygon)) and g.area > 0]
    except Exception:
        logger.debug("AP split failed via shapely.ops.split, trying difference approach")
        polygons = []

    # Fallback: thin-rectangle difference
    if len(polygons) < 2:
        thin_rect = split_line.buffer(0.5)
        remainder = outline.buffer(0).difference(thin_rect)
        if isinstance(remainder, MultiPolygon):
            polygons = [g for g in remainder.geoms if g.area > 0]
        elif isinstance(remainder, Polygon) and remainder.area > 0:
            polygons = [remainder]
        else:
            return None, None

    if len(polygons) < 2:
        return None, None

    # Take the two largest pieces
    polygons.sort(key=lambda g: g.area, reverse=True)
    half_a, half_b = polygons[0], polygons[1]

    # Step 6: label anterior (contains L3 centroid) vs posterior
    ref_point = Point(anterior_ref.tolist())
    if half_a.distance(ref_point) < half_b.distance(ref_point):
        return half_a.area, half_b.area
    else:
        return half_b.area, half_a.area


def _wing_measurements(
    wing_result: Optional[WingResult],
    scale: Optional[float],
) -> dict[str, str]:
    """Compute wing-level measurements from a WingResult.

    Returns dict with keys: wing_area_px, wing_area_um2, wing_length_px,
    wing_length_um, crossvein_distance_px, crossvein_distance_um.
    """
    vals: dict[str, str] = {}

    # Wing area
    outline = wing_result.wing_outline if wing_result else None
    if outline is not None:
        vals["wing_area_px"] = f"{outline.area:.1f}"
        vals["wing_area_um2"] = f"{outline.area * scale**2:.1f}" if scale else ""
    else:
        vals["wing_area_px"] = ""
        vals["wing_area_um2"] = ""

    landmarks = wing_result.landmarks if wing_result else {}

    # Wing length: L1-Rs to DTip
    l1rs = landmarks.get("L1-Rs")
    dtip = landmarks.get("DTip")
    if l1rs and dtip:
        dist = math.hypot(dtip.x - l1rs.x, dtip.y - l1rs.y)
        vals["wing_length_px"] = f"{dist:.1f}"
        vals["wing_length_um"] = f"{dist * scale:.1f}" if scale else ""
    else:
        vals["wing_length_px"] = ""
        vals["wing_length_um"] = ""

    # Crossvein distance: ACV.p to PCV.a
    acvp = landmarks.get("ACV.p")
    pcva = landmarks.get("PCV.a")
    if acvp and pcva:
        dist = math.hypot(pcva.x - acvp.x, pcva.y - acvp.y)
        vals["crossvein_distance_px"] = f"{dist:.1f}"
        vals["crossvein_distance_um"] = f"{dist * scale:.1f}" if scale else ""
    else:
        vals["crossvein_distance_px"] = ""
        vals["crossvein_distance_um"] = ""

    # CV ratio: crossvein distance / wing length (dimensionless)
    if vals["crossvein_distance_px"] and vals["wing_length_px"]:
        cv_ratio = float(vals["crossvein_distance_px"]) / float(vals["wing_length_px"])
        vals["cv_ratio"] = f"{cv_ratio:.4f}"
    else:
        vals["cv_ratio"] = ""

    # Anterior/posterior compartment areas
    ant_area, post_area = _compute_ap_areas(wing_result)
    if ant_area is not None:
        vals["anterior_area_px"] = f"{ant_area:.1f}"
        vals["anterior_area_um2"] = f"{ant_area * scale**2:.1f}" if scale else ""
    else:
        vals["anterior_area_px"] = ""
        vals["anterior_area_um2"] = ""
    if post_area is not None:
        vals["posterior_area_px"] = f"{post_area:.1f}"
        vals["posterior_area_um2"] = f"{post_area * scale**2:.1f}" if scale else ""
    else:
        vals["posterior_area_px"] = ""
        vals["posterior_area_um2"] = ""

    return vals


# ---------------------------------------------------------------------------
# Long format (single-specimen detailed CSV)
# ---------------------------------------------------------------------------

_LONG_FIELDS = [
    "specimen",
    "feature",
    "category",
    "type",
    "status",
    "area_px",
    "area_um2",
    "length_px",
    "length_um",
    "ratio",
]


def export_csv(
    veins: list[VeinIdentification],
    regions: list[InterveinRegion],
    out_path: Path,
    um_per_px: Optional[float] = None,
    specimen_id: Optional[str] = None,
    wing_result: Optional[WingResult] = None,
) -> None:
    """Write long-format measurements CSV for a single specimen.

    One row per feature with columns: specimen, feature, category, type,
    status, area_px, area_um2, length_px, length_um.
    Wing-level measurements (wing area, wing length, crossvein distance)
    appear as rows with category "wing".
    """
    scale = um_per_px if um_per_px is not None and um_per_px > 0 else None
    rows: list[dict] = []
    sid = specimen_id or ""

    # Wing-level measurements
    wm = _wing_measurements(wing_result, scale)
    rows.append(
        {
            "specimen": sid,
            "feature": "wing",
            "category": "wing",
            "type": "",
            "status": "",
            "area_px": wm["wing_area_px"],
            "area_um2": wm["wing_area_um2"],
            "length_px": wm["wing_length_px"],
            "length_um": wm["wing_length_um"],
        }
    )
    rows.append(
        {
            "specimen": sid,
            "feature": "crossvein distance",
            "category": "wing",
            "type": "",
            "status": "",
            "area_px": "",
            "area_um2": "",
            "length_px": wm["crossvein_distance_px"],
            "length_um": wm["crossvein_distance_um"],
            "ratio": "",
        }
    )
    rows.append(
        {
            "specimen": sid,
            "feature": "CV ratio",
            "category": "wing",
            "type": "",
            "status": "",
            "area_px": "",
            "area_um2": "",
            "length_px": "",
            "length_um": "",
            "ratio": wm["cv_ratio"],
        }
    )
    rows.append(
        {
            "specimen": sid,
            "feature": "anterior compartment",
            "category": "wing",
            "type": "",
            "status": "",
            "area_px": wm["anterior_area_px"],
            "area_um2": wm["anterior_area_um2"],
            "length_px": "",
            "length_um": "",
        }
    )
    rows.append(
        {
            "specimen": sid,
            "feature": "posterior compartment",
            "category": "wing",
            "type": "",
            "status": "",
            "area_px": wm["posterior_area_px"],
            "area_um2": wm["posterior_area_um2"],
            "length_px": "",
            "length_um": "",
        }
    )

    for v in veins:
        area_px = f"{v.tissue_polygon.area:.1f}" if v.tissue_polygon is not None else ""
        area_um2 = f"{v.tissue_polygon.area * scale**2:.1f}" if v.tissue_polygon is not None and scale else ""
        length_px = f"{v.centerline.length:.1f}" if v.centerline is not None else ""
        length_um = f"{v.centerline.length * scale:.1f}" if v.centerline is not None and scale else ""
        rows.append(
            {
                "specimen": sid,
                "feature": v.vein_id,
                "category": "vein",
                "type": v.vein_type.value,
                "status": v.status.value,
                "area_px": area_px,
                "area_um2": area_um2,
                "length_px": length_px,
                "length_um": length_um,
            }
        )

    for r in regions:
        area_px = f"{r.area_px2:.1f}" if r.polygon is not None else ""
        area_um2 = f"{r.area_px2 * scale**2:.1f}" if r.polygon is not None and scale else ""
        rows.append(
            {
                "specimen": sid,
                "feature": r.name,
                "category": "region",
                "type": "",
                "status": r.status if r.status else "identified",
                "area_px": area_px,
                "area_um2": area_um2,
                "length_px": "",
                "length_um": "",
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LONG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Wide format (batch combined CSV)
# ---------------------------------------------------------------------------


def _build_fieldnames(include_um: bool) -> list[str]:
    """Build the canonical wide-format column list."""
    fields = ["specimen"]
    # Wing-level measurements first
    fields.append("wing area_px")
    if include_um:
        fields.append("wing area_um2")
    fields.append("wing length_px")
    if include_um:
        fields.append("wing length_um")
    fields.append("crossvein distance_px")
    if include_um:
        fields.append("crossvein distance_um")
    fields.append("CV ratio")
    fields.append("anterior area_px")
    if include_um:
        fields.append("anterior area_um2")
    fields.append("posterior area_px")
    if include_um:
        fields.append("posterior area_um2")
    # Per-vein
    for vein_id in VEIN_AP_ORDER:
        name = _VEIN_DISPLAY[vein_id]
        fields.append(f"{name} length_px")
        if include_um:
            fields.append(f"{name} length_um")
    # Per-region
    for region in REGION_AP_ORDER:
        fields.append(f"{region} area_px")
        if include_um:
            fields.append(f"{region} area_um2")
    return fields


def _build_row(
    veins: list[VeinIdentification],
    regions: list[InterveinRegion],
    um_per_px: Optional[float] = None,
    specimen_id: Optional[str] = None,
    wing_result: Optional[WingResult] = None,
) -> dict[str, str]:
    """Build a single wide-format row dict for one specimen."""
    scale = um_per_px if um_per_px is not None and um_per_px > 0 else None
    include_um = scale is not None

    row: dict[str, str] = {"specimen": specimen_id or ""}

    # Wing-level measurements
    wm = _wing_measurements(wing_result, scale)
    row["wing area_px"] = wm["wing_area_px"]
    if include_um:
        row["wing area_um2"] = wm["wing_area_um2"]
    row["wing length_px"] = wm["wing_length_px"]
    if include_um:
        row["wing length_um"] = wm["wing_length_um"]
    row["crossvein distance_px"] = wm["crossvein_distance_px"]
    if include_um:
        row["crossvein distance_um"] = wm["crossvein_distance_um"]
    row["CV ratio"] = wm["cv_ratio"]
    row["anterior area_px"] = wm["anterior_area_px"]
    if include_um:
        row["anterior area_um2"] = wm["anterior_area_um2"]
    row["posterior area_px"] = wm["posterior_area_px"]
    if include_um:
        row["posterior area_um2"] = wm["posterior_area_um2"]

    # Index veins by id (skip ectopic)
    vein_map: dict[str, VeinIdentification] = {}
    for v in veins:
        if not v.vein_id.startswith("EV"):
            vein_map[v.vein_id] = v

    # Vein columns
    for vein_id in VEIN_AP_ORDER:
        name = _VEIN_DISPLAY[vein_id]
        v = vein_map.get(vein_id)
        if v is None or v.centerline is None:
            row[f"{name} length_px"] = NOT_IDENTIFIED
            if include_um:
                row[f"{name} length_um"] = NOT_IDENTIFIED
        else:
            row[f"{name} length_px"] = f"{v.centerline.length:.1f}"
            if include_um:
                row[f"{name} length_um"] = f"{v.centerline.length * scale:.1f}"

    # Index regions by name
    region_map: dict[str, InterveinRegion] = {}
    for r in regions:
        region_map[r.name] = r

    # Region columns
    for region_name in REGION_AP_ORDER:
        r = region_map.get(region_name)
        if r is None or r.polygon is None:
            row[f"{region_name} area_px"] = NOT_IDENTIFIED
            if include_um:
                row[f"{region_name} area_um2"] = NOT_IDENTIFIED
        else:
            row[f"{region_name} area_px"] = f"{r.area_px2:.1f}"
            if include_um:
                row[f"{region_name} area_um2"] = f"{r.area_px2 * scale**2:.1f}"

    return row


def export_csv_batch(
    all_results: list[tuple[str, WingResult]],
    out_path: Path,
    um_per_px: Optional[float] = None,
) -> None:
    """Write wide-format measurements CSV for multiple specimens (one row each)."""
    scale = um_per_px if um_per_px is not None and um_per_px > 0 else None
    include_um = scale is not None
    fieldnames = _build_fieldnames(include_um)

    rows = []
    for specimen_id, wing_result in sorted(all_results, key=lambda x: x[0]):
        rows.append(
            _build_row(
                wing_result.veins,
                wing_result.intervein_regions,
                um_per_px,
                specimen_id,
                wing_result=wing_result,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
