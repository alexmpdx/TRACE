"""Export vein and region measurements as CSV.

Two formats:
- Long format (single mode): one row per feature, all columns
- Wide format (batch mode): one row per specimen, measurements as columns
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from identify_features.models.datatypes import InterveinRegion, VeinIdentification
from identify_features.models.topology import REGION_AP_ORDER, VEIN_AP_ORDER

if TYPE_CHECKING:
    from identify_features.models.datatypes import Landmark, WingResult

# Display name mapping (internal id → CSV column name) for wide format
_VEIN_DISPLAY = {v: ("costal vein" if v == "costa" else v) for v in VEIN_AP_ORDER}

NOT_IDENTIFIED = "not identified"


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
