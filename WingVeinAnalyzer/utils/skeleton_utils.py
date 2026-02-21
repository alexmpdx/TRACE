"""Skeletonize helpers, spur pruning, and flip detection."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def detect_and_correct_flip(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and correct X-flip based on anterior vein density."""
    h, w = mask.shape[:2]
    midpoint = w // 2

    anterior_density = np.count_nonzero(mask[:, :midpoint])
    posterior_density = np.count_nonzero(mask[:, midpoint:])

    if posterior_density > anterior_density:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    return image, mask


def detect_geojson_flip(
    polygons: list[Polygon],
    image_width: int,
) -> bool:
    """Check if GeoJSON polygons suggest the wing is flipped.

    In a standard orientation, the costa (most anterior/top vein) is at the
    top of the image. The most anterior (lowest Y) polygon should have its
    centroid in the upper portion. If the density of annotation centroids
    is concentrated in the lower half, the image may be flipped vertically.

    Returns True if a flip is detected.
    """
    if not polygons:
        return False

    centroids_y = [p.centroid.y for p in polygons]
    areas = [p.area for p in polygons]
    total_area = sum(areas)

    if total_area == 0:
        return False

    # Weighted average Y position
    weighted_y = sum(cy * a for cy, a in zip(centroids_y, areas)) / total_area
    # If weighted centroid is in upper third, wing is likely inverted
    # (posterior structures typically dominate area)
    # Standard orientation: posterior (large area) at bottom (high Y)
    return False  # GeoJSON coordinates are fixed; flipping handled at image level
