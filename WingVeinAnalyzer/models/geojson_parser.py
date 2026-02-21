"""Parse GeoJSON annotations into typed dataclasses for wing vein analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, shape


@dataclass
class ParsedVein:
    """A single vein LineString extracted from annotations."""

    feature_id: str
    line: LineString
    coords: np.ndarray
    length_px: float
    centroid_x: float
    centroid_y: float


@dataclass
class ParsedOutline:
    """A posterior/wing outline segment."""

    feature_id: str
    line: LineString
    coords: np.ndarray


@dataclass
class ParsedAnnotations:
    """All parsed features from a GeoJSON annotation file."""

    veins: list[ParsedVein] = field(default_factory=list)
    posterior_segments: list[ParsedOutline] = field(default_factory=list)
    wing_outline_segments: list[ParsedOutline] = field(default_factory=list)
    intervein_polygons: list[Polygon] = field(default_factory=list)
    vein_polygons: list[Polygon] = field(default_factory=list)


def parse_geojson(path: Path) -> ParsedAnnotations:
    """Load a GeoJSON file and classify features by annotation type.

    Handles two formats:
    - Features with LineString geometries classified as "vein", "posterior outline", etc.
    - Features with (Multi)Polygon geometries classified as "intervein" space regions.
    """
    with open(path) as f:
        data = json.load(f)

    annotations = ParsedAnnotations()

    for feat in data.get("features", []):
        geom = shape(feat["geometry"])
        props = feat.get("properties", {})
        feat_id = feat.get("id", props.get("objectType", "unknown"))
        cls = props.get("classification", {})
        cls_name = cls.get("name", "").lower().strip()

        # Skip ignored features
        if cls_name.startswith("ignore") or cls_name == "hair":
            continue

        # Handle LineString features (vein traces, outline segments)
        if isinstance(geom, LineString):
            coords = np.array(geom.coords)
            if cls_name == "vein":
                annotations.veins.append(
                    ParsedVein(
                        feature_id=str(feat_id),
                        line=geom,
                        coords=coords,
                        length_px=geom.length,
                        centroid_x=geom.centroid.x,
                        centroid_y=geom.centroid.y,
                    )
                )
            elif cls_name in ("posterior outline", "wing outline"):
                segment = ParsedOutline(
                    feature_id=str(feat_id), line=geom, coords=coords
                )
                if cls_name == "posterior outline":
                    annotations.posterior_segments.append(segment)
                else:
                    annotations.wing_outline_segments.append(segment)

        # Handle Polygon / MultiPolygon features (intervein spaces or vein mask)
        elif isinstance(geom, (Polygon, MultiPolygon)):
            if cls_name in ("intervein", "intervein space"):
                if isinstance(geom, MultiPolygon):
                    annotations.intervein_polygons.extend(list(geom.geoms))
                else:
                    annotations.intervein_polygons.append(geom)
            elif cls_name == "vein":
                if isinstance(geom, MultiPolygon):
                    annotations.vein_polygons.extend(list(geom.geoms))
                else:
                    annotations.vein_polygons.append(geom)

    # Split polygons with narrow constrictions into separate regions
    annotations.intervein_polygons = _split_and_clean_regions(
        annotations.intervein_polygons
    )

    return annotations


def _split_and_clean_regions(
    polygons: list[Polygon],
    constriction_ratio: float = 0.15,
) -> list[Polygon]:
    """Split polygons connected through narrow constrictions.

    If any vertical cross-section of a polygon is less than constriction_ratio
    of that polygon's maximum cross-section width, split at that point.
    This handles annotation errors where two regions are barely connected.
    """
    result: list[Polygon] = []

    for poly in polygons:
        splits = _try_split_at_constriction(poly, constriction_ratio)
        result.extend(splits)

    return result


def _try_split_at_constriction(
    poly: Polygon,
    constriction_ratio: float,
) -> list[Polygon]:
    """Split a single polygon at its narrowest constriction, if one exists.

    Only splits if the narrowest cross-section is < constriction_ratio of the
    max cross-section, AND both resulting pieces are substantial (>10% of
    the original area).
    """
    orig_area = poly.area
    bounds = poly.bounds  # (minx, miny, maxx, maxy)
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x

    if width < 100:
        return [poly]

    # Sweep vertical cross-sections
    n_samples = max(50, int(width / 10))
    xs = np.linspace(min_x + 1, max_x - 1, n_samples)

    cross_widths: list[tuple[float, float]] = []  # (x, width_at_x)
    for x in xs:
        line = LineString([(x, min_y - 1), (x, max_y + 1)])
        intersection = poly.intersection(line)
        if intersection.is_empty:
            cross_widths.append((x, 0.0))
        else:
            cross_widths.append((x, intersection.length))

    if not cross_widths:
        return [poly]

    max_cross_width = max(w for _, w in cross_widths)
    if max_cross_width < 20:
        return [poly]

    # Find narrowest point (skip outer 20% on each side)
    margin = max(5, len(cross_widths) // 5)
    interior = cross_widths[margin:-margin]
    if not interior:
        return [poly]

    narrowest_x, narrowest_w = min(interior, key=lambda t: t[1])

    if narrowest_w > constriction_ratio * max_cross_width:
        return [poly]

    # Split at narrowest X using a thin vertical cut
    cut_line = LineString([(narrowest_x, min_y - 10), (narrowest_x, max_y + 10)])
    try:
        from shapely.ops import split
        parts = split(poly, cut_line)
        # Only accept splits where both pieces are substantial (>10% of original)
        min_area = orig_area * 0.10
        pieces = [g for g in parts.geoms if g.area > min_area]
        if len(pieces) >= 2:
            return pieces
    except Exception:
        pass

    return [poly]
