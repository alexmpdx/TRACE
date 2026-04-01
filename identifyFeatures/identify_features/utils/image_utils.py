"""Image loading and mask rasterization utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

logger = logging.getLogger(__name__)


def load_image(path: Path) -> np.ndarray:
    """Load an image file (TIFF, BMP, PNG, JPG) as BGR numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def get_image_shape(path: Path) -> tuple[int, int]:
    """Get (height, width) of an image without fully loading it."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img.shape[:2]


def rasterize_polygons(
    polygons: list[Polygon | MultiPolygon],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a list of Shapely polygons to a binary mask.

    Args:
        polygons: List of Shapely Polygon or MultiPolygon objects.
        shape: (height, width) of the output mask.

    Returns:
        Binary uint8 mask (255 = inside polygon, 0 = outside).
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in polygons:
        _fill_polygon(mask, poly)
    return mask


def _fill_polygon(mask: np.ndarray, poly: Polygon | MultiPolygon) -> None:
    """Fill a single polygon (or multipolygon) onto a mask."""
    if isinstance(poly, MultiPolygon):
        for p in poly.geoms:
            _fill_polygon(mask, p)
        return

    if poly.is_empty:
        return

    # Exterior ring
    coords = np.array(poly.exterior.coords, dtype=np.int32)
    cv2.fillPoly(mask, [coords], 255)

    # Cut out interior rings (holes)
    for interior in poly.interiors:
        hole_coords = np.array(interior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [hole_coords], 0)
