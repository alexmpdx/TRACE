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
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from identify_features.garbage_detector import compute_solidity
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

# Measurement groups exposed to the CSV-output UI. Each group toggles a coherent
# block of columns AND the compute work behind it (e.g. dropping "ap_areas"
# skips compute_ap_split). The "specimen" column is always present.
#
# Wing length lives in `cv_ratio` rather than `wing_area` because the CV ratio is
# (crossvein distance / wing length) and the two values are usually reported
# together. Selecting `cv_ratio` writes wing-length, crossvein-distance, and
# CV-ratio columns; selecting `wing_area` only writes wing-area columns.
MEASUREMENT_GROUPS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("wing_area", "Wing area"),
        ("wing_shape", "Wing shape (aspect ratio, solidity)"),
        ("vein_lengths", "Vein lengths"),
        ("intervein_areas", "Intervein region areas"),
        ("cv_ratio", "CV ratio (CV distance/wing length)"),
        ("ap_areas", "A/P compartment areas"),
    ]
)

# Per-group tooltips for the GUI checkboxes. Keys are a subset of
# MEASUREMENT_GROUPS — groups without an entry render as a bare label.
MEASUREMENT_GROUP_TOOLTIPS: dict[str, str] = {
    "wing_shape": (
        "Aspect ratio: sqrt(λ₁/λ₂) of the wing outline's PCA eigenvalues — the "
        "ratio of the wing's long axis to its short axis. Wildtype ~2.3.\n\n"
        "Solidity: wing area / convex-hull area. Wildtype ~0.97–0.99; drops on "
        "notched mutants (e.g. Notch, Serrate)."
    ),
    "cv_ratio": (
        "Crossvein distance / wing length.\n\n"
        "CV distance measurement: ACV-L4 junction to PCV-L4 junction.\n"
        "Wing length measurement: L1-Rs junction to L3 distal end."
    ),
    "ap_areas": "Area of the anterior and posterior compartments.",
}

ALL_MEASUREMENT_GROUPS: frozenset[str] = frozenset(MEASUREMENT_GROUPS.keys())


def compute_ap_split(
    wing_result: Optional[WingResult],
) -> tuple[Optional[Polygon], Optional[Polygon]]:
    """Split wing into anterior/posterior compartment polygons along L4 axis.

    Returns (anterior_polygon, posterior_polygon) or (None, None).
    """
    if wing_result is None or wing_result.wing_outline is None:
        return None, None

    veins_by_id = {v.vein_id: v for v in wing_result.veins}
    l4 = veins_by_id.get("L4")
    if l4 is None or l4.centerline is None:
        return None, None

    l4_line = l4.centerline
    outline = wing_result.wing_outline

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bbox = l4_line.minimum_rotated_rectangle
    coords = list(bbox.exterior.coords)[:4]

    edges = []
    for i in range(4):
        p1, p2 = np.array(coords[i]), np.array(coords[(i + 1) % 4])
        length = np.linalg.norm(p2 - p1)
        midpoint = (p1 + p2) / 2
        edges.append((p1, p2, length, midpoint))

    edges.sort(key=lambda e: e[2], reverse=True)
    long_edges = edges[:2]

    anterior_ref = None
    for vid in ("L3", "L2", "L1"):
        v = veins_by_id.get(vid)
        if v is not None and v.centerline is not None:
            anterior_ref = np.array(v.centerline.centroid.coords[0])
            break

    if anterior_ref is None:
        return None, None

    d0 = np.linalg.norm(long_edges[0][3] - anterior_ref)
    d1 = np.linalg.norm(long_edges[1][3] - anterior_ref)
    ant_edge = long_edges[0] if d0 < d1 else long_edges[1]

    p1, p2 = ant_edge[0], ant_edge[1]
    direction = p2 - p1
    direction = direction / np.linalg.norm(direction)

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
        landmarks = wing_result.landmarks if wing_result else {}
        dtip = landmarks.get("DTip")
        dtip_pt = np.array([dtip.x, dtip.y]) if dtip else np.array(l4_line.coords[-1])
        if np.linalg.norm(p1 - dtip_pt) < np.linalg.norm(p2 - dtip_pt):
            p1 = p1 + perp * median_width * 0.5
        else:
            p2 = p2 + perp * median_width * 0.5

    minx, miny, maxx, maxy = outline.bounds
    diag = math.hypot(maxx - minx, maxy - miny)
    extend = diag * 2

    midpt = (p1 + p2) / 2
    ext_p1 = midpt - direction * extend
    ext_p2 = midpt + direction * extend
    split_line = LineString([ext_p1.tolist(), ext_p2.tolist()])

    try:
        parts = split(outline.buffer(0), split_line)
        polygons = [g for g in parts.geoms if isinstance(g, (Polygon, MultiPolygon)) and g.area > 0]
    except Exception:
        logger.debug("AP split failed via shapely.ops.split, trying difference approach")
        polygons = []

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

    polygons.sort(key=lambda g: g.area, reverse=True)
    half_a, half_b = polygons[0], polygons[1]

    ref_point = Point(anterior_ref.tolist())
    if half_a.distance(ref_point) < half_b.distance(ref_point):
        return half_a, half_b
    else:
        return half_b, half_a


def _compute_ap_areas(
    wing_result: Optional[WingResult],
) -> tuple[Optional[float], Optional[float]]:
    """Return (anterior_area_px, posterior_area_px) or (None, None)."""
    anterior, posterior = compute_ap_split(wing_result)
    if anterior is None:
        return None, None
    return anterior.area, posterior.area


def _wing_measurements(
    wing_result: Optional[WingResult],
    scale: Optional[float],
    groups: Optional[set[str]] = None,
) -> dict[str, str]:
    """Compute wing-level measurements from a WingResult.

    Returns dict with keys: wing_area_px, wing_area_um2, wing_length_px,
    wing_length_um, crossvein_distance_px, crossvein_distance_um, cv_ratio,
    anterior_area_px, anterior_area_um2, posterior_area_px, posterior_area_um2.

    When ``groups`` is provided, computations and corresponding keys for groups
    not in the set are skipped (the keys are still present in the dict but as
    empty strings, so callers can do unconditional dict lookups).
    """
    g = set(groups) if groups is not None else set(ALL_MEASUREMENT_GROUPS)
    vals: dict[str, str] = {
        "wing_area_px": "",
        "wing_area_um2": "",
        "wing_aspect_ratio": "",
        "wing_solidity": "",
        "wing_length_px": "",
        "wing_length_um": "",
        "crossvein_distance_px": "",
        "crossvein_distance_um": "",
        "cv_ratio": "",
        "anterior_area_px": "",
        "anterior_area_um2": "",
        "posterior_area_px": "",
        "posterior_area_um2": "",
    }

    landmarks = wing_result.landmarks if wing_result else {}

    # Wing area
    if "wing_area" in g:
        outline = wing_result.wing_outline if wing_result else None
        if outline is not None:
            vals["wing_area_px"] = f"{outline.area:.1f}"
            vals["wing_area_um2"] = f"{outline.area * scale**2:.1f}" if scale else ""

    # Wing shape: aspect ratio (elongation) + solidity (notch / missing-area).
    if "wing_shape" in g:
        outline = wing_result.wing_outline if wing_result else None
        if outline is not None and not outline.is_empty:
            coords = np.asarray(outline.exterior.coords)[:-1]
            if len(coords) >= 3:
                centered = coords - coords.mean(axis=0)
                eigvals = np.linalg.eigvalsh(np.cov(centered, rowvar=False))
                if eigvals[0] > 0:
                    vals["wing_aspect_ratio"] = f"{float(np.sqrt(eigvals[1] / eigvals[0])):.4f}"
            # Reuse the value the garbage detector already computed when present;
            # otherwise compute it (shared single source of truth).
            solidity = wing_result.wing_solidity if wing_result else None
            if solidity is None:
                solidity = compute_solidity(outline)
            if solidity is not None:
                vals["wing_solidity"] = f"{solidity:.4f}"

    # CV ratio block: wing length + crossvein distance + CV ratio
    if "cv_ratio" in g:
        # Wing length: L1-Rs to DTip
        l1rs = landmarks.get("L1-Rs")
        dtip = landmarks.get("DTip")
        if l1rs and dtip:
            dist = math.hypot(dtip.x - l1rs.x, dtip.y - l1rs.y)
            vals["wing_length_px"] = f"{dist:.1f}"
            vals["wing_length_um"] = f"{dist * scale:.1f}" if scale else ""

        # Crossvein distance: ACV.p to PCV.a
        acvp = landmarks.get("ACV.p")
        pcva = landmarks.get("PCV.a")
        if acvp and pcva:
            dist = math.hypot(pcva.x - acvp.x, pcva.y - acvp.y)
            vals["crossvein_distance_px"] = f"{dist:.1f}"
            vals["crossvein_distance_um"] = f"{dist * scale:.1f}" if scale else ""

        # CV ratio = crossvein distance / wing length (dimensionless)
        if vals["crossvein_distance_px"] and vals["wing_length_px"]:
            cv_ratio = float(vals["crossvein_distance_px"]) / float(vals["wing_length_px"])
            vals["cv_ratio"] = f"{cv_ratio:.4f}"

    # Anterior/posterior compartment areas (skip compute_ap_split when not needed)
    if "ap_areas" in g:
        ant_area, post_area = _compute_ap_areas(wing_result)
        if ant_area is not None:
            vals["anterior_area_px"] = f"{ant_area:.1f}"
            vals["anterior_area_um2"] = f"{ant_area * scale**2:.1f}" if scale else ""
        if post_area is not None:
            vals["posterior_area_px"] = f"{post_area:.1f}"
            vals["posterior_area_um2"] = f"{post_area * scale**2:.1f}" if scale else ""

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
    groups: Optional[set[str]] = None,
) -> None:
    """Write long-format measurements CSV for a single specimen.

    One row per feature with columns: specimen, feature, category, type,
    status, area_px, area_um2, length_px, length_um.

    ``groups`` filters which measurement groups produce rows (see
    MEASUREMENT_GROUPS). None = all groups (back-compat default).
    """
    scale = um_per_px if um_per_px is not None and um_per_px > 0 else None
    g = set(groups) if groups is not None else set(ALL_MEASUREMENT_GROUPS)
    rows: list[dict] = []
    sid = specimen_id or ""

    # Wing-level measurements (computed only for groups in scope)
    wm = _wing_measurements(wing_result, scale, groups=g)
    if "wing_area" in g:
        rows.append(
            {
                "specimen": sid,
                "feature": "wing",
                "category": "wing",
                "type": "",
                "status": "",
                "area_px": wm["wing_area_px"],
                "area_um2": wm["wing_area_um2"],
                "length_px": "",
                "length_um": "",
            }
        )
    if "wing_shape" in g:
        rows.append(
            {
                "specimen": sid,
                "feature": "wing aspect ratio",
                "category": "wing",
                "type": "",
                "status": "",
                "area_px": "",
                "area_um2": "",
                "length_px": "",
                "length_um": "",
                "ratio": wm["wing_aspect_ratio"],
            }
        )
        rows.append(
            {
                "specimen": sid,
                "feature": "wing solidity",
                "category": "wing",
                "type": "",
                "status": "",
                "area_px": "",
                "area_um2": "",
                "length_px": "",
                "length_um": "",
                "ratio": wm["wing_solidity"],
            }
        )
    if "cv_ratio" in g:
        rows.append(
            {
                "specimen": sid,
                "feature": "wing length",
                "category": "wing",
                "type": "",
                "status": "",
                "area_px": "",
                "area_um2": "",
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
    if "ap_areas" in g:
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

    if "vein_lengths" in g:
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

    if "intervein_areas" in g:
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
    # utf-8-sig so Excel on Windows auto-detects UTF-8 on double-click
    # (specimen names like "29ºC" would otherwise mangle). TRACE's
    # merge_resume_csv reads via utf-8-sig fallback ladder so this is
    # also merge-safe. See v0.2.25 CSV-encoding fix.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_LONG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Wide format (batch combined CSV)
# ---------------------------------------------------------------------------


def _build_fieldnames(include_um: bool, groups: Optional[set[str]] = None) -> list[str]:
    """Build the canonical wide-format column list, filtered by ``groups``.

    When ``groups`` is None, all measurement groups are included.
    """
    g = set(groups) if groups is not None else set(ALL_MEASUREMENT_GROUPS)
    fields = ["specimen"]
    if "wing_area" in g:
        fields.append("wing area_px")
        if include_um:
            fields.append("wing area_um2")
    if "wing_shape" in g:
        fields.append("wing aspect ratio")
        fields.append("wing solidity")
    if "cv_ratio" in g:
        fields.append("wing length_px")
        if include_um:
            fields.append("wing length_um")
        fields.append("crossvein distance_px")
        if include_um:
            fields.append("crossvein distance_um")
        fields.append("CV ratio")
    if "ap_areas" in g:
        fields.append("anterior area_px")
        if include_um:
            fields.append("anterior area_um2")
        fields.append("posterior area_px")
        if include_um:
            fields.append("posterior area_um2")
    if "vein_lengths" in g:
        for vein_id in VEIN_AP_ORDER:
            name = _VEIN_DISPLAY[vein_id]
            fields.append(f"{name} length_px")
            if include_um:
                fields.append(f"{name} length_um")
    if "intervein_areas" in g:
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
    groups: Optional[set[str]] = None,
) -> dict[str, str]:
    """Build a single wide-format row dict for one specimen, filtered by ``groups``."""
    scale = um_per_px if um_per_px is not None and um_per_px > 0 else None
    include_um = scale is not None
    g = set(groups) if groups is not None else set(ALL_MEASUREMENT_GROUPS)

    row: dict[str, str] = {"specimen": specimen_id or ""}

    # Wing-level measurements (computed only for groups in scope)
    wm = _wing_measurements(wing_result, scale, groups=g)
    if "wing_area" in g:
        row["wing area_px"] = wm["wing_area_px"]
        if include_um:
            row["wing area_um2"] = wm["wing_area_um2"]
    if "wing_shape" in g:
        row["wing aspect ratio"] = wm["wing_aspect_ratio"]
        row["wing solidity"] = wm["wing_solidity"]
    if "cv_ratio" in g:
        row["wing length_px"] = wm["wing_length_px"]
        if include_um:
            row["wing length_um"] = wm["wing_length_um"]
        row["crossvein distance_px"] = wm["crossvein_distance_px"]
        if include_um:
            row["crossvein distance_um"] = wm["crossvein_distance_um"]
        row["CV ratio"] = wm["cv_ratio"]
    if "ap_areas" in g:
        row["anterior area_px"] = wm["anterior_area_px"]
        if include_um:
            row["anterior area_um2"] = wm["anterior_area_um2"]
        row["posterior area_px"] = wm["posterior_area_px"]
        if include_um:
            row["posterior area_um2"] = wm["posterior_area_um2"]

    # Per-vein lengths
    if "vein_lengths" in g:
        vein_map: dict[str, VeinIdentification] = {}
        for v in veins:
            if not v.vein_id.startswith("EV"):
                vein_map[v.vein_id] = v
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

    # Per-region areas
    if "intervein_areas" in g:
        region_map: dict[str, InterveinRegion] = {}
        for r in regions:
            region_map[r.name] = r
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
    groups: Optional[set[str]] = None,
) -> None:
    """Write wide-format measurements CSV for multiple specimens (one row each).

    ``groups`` controls which measurement groups appear as columns (see
    MEASUREMENT_GROUPS). None = all groups (back-compat default).

    Each row's µm conversion uses ``wing_result.um_per_px`` when set (TRACE's
    auto-detect-from-metadata mode stamps the per-image scale there), falling
    back to the caller-supplied ``um_per_px`` for specimens without a
    per-image scale.
    """
    # µm columns are included whenever ANY row has a resolvable scale — either
    # the batch-wide arg OR at least one specimen carrying its own um_per_px.
    def _resolve_scale(wing_result: WingResult) -> Optional[float]:
        per_specimen = getattr(wing_result, "um_per_px", None)
        if per_specimen is not None and per_specimen > 0:
            return float(per_specimen)
        if um_per_px is not None and um_per_px > 0:
            return float(um_per_px)
        return None

    include_um = any(_resolve_scale(r) is not None for _, r in all_results)
    fieldnames = _build_fieldnames(include_um, groups=groups)

    rows = []
    for specimen_id, wing_result in sorted(all_results, key=lambda x: x[0]):
        rows.append(
            _build_row(
                wing_result.veins,
                wing_result.intervein_regions,
                _resolve_scale(wing_result),
                specimen_id,
                wing_result=wing_result,
                groups=groups,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel on Windows auto-detects UTF-8 on double-click
    # (specimen names like "29ºC" would otherwise mangle). TRACE's
    # merge_resume_csv reads via utf-8-sig fallback ladder so this is
    # also merge-safe. See v0.2.25 CSV-encoding fix.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
