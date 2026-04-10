"""Export identified veins and intervein regions as GeoJSON.

Produces a FeatureCollection matching the GT_naming format used by
QuPath/ground-truth annotations: each feature has ``objectType: "annotation"``
and a ``classification`` block with ``name`` and ``color`` (RGB).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from identify_features.models.datatypes import InterveinRegion, VeinIdentification
from identify_features.models.topology import REGION_COLORS, VEIN_COLORS
from shapely.geometry import MultiPolygon, Polygon, mapping


def export_geojson(
    veins: list[VeinIdentification],
    regions: list[InterveinRegion],
    out_path: Path,
    um_per_px: Optional[float] = None,
) -> None:
    """Write veins and regions to a GeoJSON file in GT_naming format.

    Args:
        veins: Identified veins with tissue_polygon populated.
        regions: Named intervein regions with polygon populated.
        out_path: Output file path.
        um_per_px: If provided, include area measurements in um^2.
    """
    features: list[dict] = []

    # Vein features (tissue polygons)
    for v in veins:
        if v.tissue_polygon is None:
            continue
        poly = v.tissue_polygon
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        color = VEIN_COLORS.get(v.vein_id, [128, 128, 128])
        props: dict = {
            "objectType": "annotation",
            "classification": {
                "name": v.vein_id,
                "color": color,
            },
        }
        measurements: dict = {"Area (pixels)": poly.area}
        if um_per_px is not None and um_per_px > 0:
            measurements["Area (um^2)"] = poly.area * um_per_px**2
        if v.centerline is not None:
            measurements["Length (pixels)"] = v.centerline.length
            if um_per_px is not None and um_per_px > 0:
                measurements["Length (um)"] = v.centerline.length * um_per_px
        props["measurements"] = measurements
        features.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": mapping(poly),
                "properties": props,
            }
        )

    # Region features (intervein polygons)
    for r in regions:
        if r.polygon is None:
            continue
        poly = r.polygon
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        color_key = r.name.split(" + ")[0]
        color = REGION_COLORS.get(color_key, [128, 128, 128])
        props = {
            "objectType": "annotation",
            "classification": {
                "name": r.name,
                "color": color,
            },
        }
        measurements = {"Area (pixels)": r.area_px2}
        if um_per_px is not None and um_per_px > 0:
            measurements["Area (um^2)"] = r.area_px2 * um_per_px**2
        props["measurements"] = measurements
        features.append(
            {
                "type": "Feature",
                "id": str(uuid.uuid4()),
                "geometry": mapping(poly),
                "properties": props,
            }
        )

    collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(collection, f)
