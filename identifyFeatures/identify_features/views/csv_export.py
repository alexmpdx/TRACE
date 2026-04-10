"""Export vein and region measurements as CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from identify_features.models.datatypes import InterveinRegion, VeinIdentification
from shapely.geometry import MultiPolygon


def export_csv(
    veins: list[VeinIdentification],
    regions: list[InterveinRegion],
    out_path: Path,
    um_per_px: Optional[float] = None,
    specimen_id: Optional[str] = None,
) -> None:
    """Write vein and region measurements to a CSV file.

    Columns:
        specimen, feature, category, type, status,
        area_px, area_um2, length_px, length_um, bounding_veins
    """
    scale = um_per_px if um_per_px is not None and um_per_px > 0 else None

    rows: list[dict] = []

    for v in veins:
        tissue_area = 0.0
        if v.tissue_polygon is not None:
            poly = v.tissue_polygon
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)
            tissue_area = poly.area

        length_px = v.centerline.length if v.centerline is not None else 0.0

        rows.append(
            {
                "specimen": specimen_id or "",
                "feature": v.vein_id,
                "category": "vein",
                "type": v.vein_type.value,
                "status": v.status.value,
                "area_px": f"{tissue_area:.1f}",
                "area_um2": f"{tissue_area * scale**2:.1f}" if scale else "",
                "length_px": f"{length_px:.1f}",
                "length_um": f"{length_px * scale:.1f}" if scale else "",
                "bounding_veins": "",
            }
        )

    for r in regions:
        area = r.area_px2
        rows.append(
            {
                "specimen": specimen_id or "",
                "feature": r.name,
                "category": "region",
                "type": "",
                "status": r.status,
                "area_px": f"{area:.1f}",
                "area_um2": f"{area * scale**2:.1f}" if scale else "",
                "length_px": "",
                "length_um": "",
                "bounding_veins": ";".join(sorted(r.bounding_veins)),
            }
        )

    fieldnames = [
        "specimen",
        "feature",
        "category",
        "type",
        "status",
        "area_px",
        "area_um2",
        "length_px",
        "length_um",
        "bounding_veins",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
