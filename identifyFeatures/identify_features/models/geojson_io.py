"""Parse detection and landmark GeoJSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from identify_features.models.datatypes import Landmark, ParsedInput
from identify_features.models.topology import RELIABLE_LANDMARKS, SOFT_LANDMARKS
from shapely.geometry import MultiPolygon, Point, Polygon, shape

logger = logging.getLogger(__name__)


def load_detection_geojson(path: Path) -> tuple[list[Polygon | MultiPolygon], list[Polygon | MultiPolygon]]:
    """Parse a detection GeoJSON into vein and intervein polygon lists.

    Returns:
        (vein_polygons, intervein_polygons)
    """
    data = _read_geojson(path)
    vein_polys: list[Polygon | MultiPolygon] = []
    intervein_polys: list[Polygon | MultiPolygon] = []

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        cls = props.get("class", "")
        geom = _safe_shape(feat)
        if geom is None or geom.is_empty:
            continue

        if cls == "vein":
            vein_polys.append(geom)
        elif cls == "intervein":
            intervein_polys.append(geom)
        elif cls in ("hinge junk", "wing"):
            logger.debug("Skipping %s feature", cls)
        else:
            logger.warning("Unknown class %r in %s — skipping", cls, path.name)

    logger.info(
        "Loaded %s: %d vein polygons, %d intervein polygons",
        path.name,
        len(vein_polys),
        len(intervein_polys),
    )
    return vein_polys, intervein_polys


def load_landmarks_geojson(path: Path) -> dict[str, Landmark]:
    """Parse a landmarks GeoJSON into a dict of Landmark objects.

    Only reliable landmarks are marked as such; all are returned.
    """
    data = _read_geojson(path)
    landmarks: dict[str, Landmark] = {}

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        name = props.get("classification", {}).get("name")
        if name is None:
            logger.warning("Landmark feature missing classification.name — skipping")
            continue

        coords = feat.get("geometry", {}).get("coordinates")
        if coords is None or len(coords) < 2:
            logger.warning("Landmark %r has invalid coordinates — skipping", name)
            continue

        is_soft = name in SOFT_LANDMARKS
        landmarks[name] = Landmark(
            name=name,
            point=Point(coords[0], coords[1]),
            reliable=name in RELIABLE_LANDMARKS or is_soft,
            soft=is_soft,
        )

    reliable_found = [n for n, lm in landmarks.items() if lm.reliable]
    logger.info(
        "Loaded %s: %d landmarks (%d reliable: %s)",
        path.name,
        len(landmarks),
        len(reliable_found),
        ", ".join(reliable_found),
    )
    return landmarks


def load_inputs(
    detection_path: Path,
    landmarks_path: Path,
    image_shape: tuple[int, int] | None = None,
) -> ParsedInput:
    """Load and combine all inputs into a ParsedInput."""
    vein_polys, intervein_polys = load_detection_geojson(detection_path)
    landmarks = load_landmarks_geojson(landmarks_path)

    # Compute wing outline as union of all polygons
    wing_outline = _compute_wing_outline(vein_polys + intervein_polys)

    return ParsedInput(
        vein_polygons=vein_polys,
        intervein_polygons=intervein_polys,
        landmarks=landmarks,
        wing_outline=wing_outline,
        image_shape=image_shape,
    )


def _compute_wing_outline(
    polygons: list[Polygon | MultiPolygon],
    buffer_px: float = 20.0,
) -> Polygon | None:
    """Compute wing outline as the union of all polygons, buffered."""
    if not polygons:
        return None
    from shapely.ops import unary_union

    union = unary_union(polygons).buffer(buffer_px).buffer(-buffer_px / 2)
    if isinstance(union, MultiPolygon):
        # Take the largest polygon
        union = max(union.geoms, key=lambda g: g.area)
    if isinstance(union, Polygon) and not union.is_empty:
        return union
    return None


def _read_geojson(path: Path) -> dict[str, Any]:
    """Read a GeoJSON file, handling encoding issues."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_shape(feature: dict) -> Polygon | MultiPolygon | None:
    """Convert a GeoJSON feature geometry to a Shapely shape, or None."""
    try:
        geom = shape(feature["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        return geom
    except Exception as e:
        logger.warning("Failed to parse geometry: %s", e)
        return None
