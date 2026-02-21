"""Voronoi-based vein centerline extraction from vein mask + intervein polygons."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Polygon

logger = logging.getLogger(__name__)


def extract_veins_from_mask(
    vein_polygons: list[Polygon],
    intervein_polygons: list[Polygon],
    image_shape: tuple[int, int],
) -> dict[tuple[int, int], LineString]:
    """Extract vein centerlines using Voronoi partition of vein mask.

    Uses distance_transform_edt to assign every pixel to its nearest
    intervein polygon.  Within the vein mask, boundaries between adjacent
    Voronoi regions are the equidistant centerlines.

    Parameters
    ----------
    vein_polygons : list[Polygon]
        Polygons defining where vein tissue exists.
    intervein_polygons : list[Polygon]
        Intervein region polygons (seeds for Voronoi partition).
    image_shape : (height, width)
        Image dimensions for rasterization.

    Returns
    -------
    dict[(poly_idx_a, poly_idx_b), LineString]
        One centerline per adjacent polygon pair found in the vein mask.
        Keys are sorted (a < b) polygon index pairs.
    """
    h, w = image_shape

    # 1. Rasterize intervein polygons to a label map
    label_map = np.zeros((h, w), dtype=np.int32)
    for i, poly in enumerate(intervein_polygons):
        label = i + 1  # labels 1..N, 0 = background
        _fill_polygon(label_map, poly, label)

    # 2. Rasterize vein mask to binary
    vein_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in vein_polygons:
        _fill_polygon(vein_mask, poly, 1)

    # 3. Voronoi partition via distance transform
    # For every pixel not inside an intervein polygon, find the nearest polygon
    background = (label_map == 0)
    _, nearest_indices = ndimage.distance_transform_edt(
        background, return_distances=True, return_indices=True
    )
    # nearest_indices has shape (2, h, w) — [0] = row indices, [1] = col indices
    # Map each background pixel to the label of its nearest polygon pixel
    nearest_labels = label_map[nearest_indices[0], nearest_indices[1]]
    # Keep original labels where polygons exist
    nearest_labels[~background] = label_map[~background]

    logger.info(
        "Voronoi partition: %d unique labels, vein mask covers %d pixels",
        len(np.unique(nearest_labels)), int(vein_mask.sum()),
    )

    # 4. Extract label boundaries within the vein mask
    boundary_pixels = _extract_boundary_pixels(nearest_labels, vein_mask)

    # 5. Filter and trace into LineStrings
    centerlines: dict[tuple[int, int], LineString] = {}
    for (label_a, label_b), pixels in boundary_pixels.items():
        # Convert labels back to polygon indices (label = idx + 1)
        idx_a = label_a - 1
        idx_b = label_b - 1
        if idx_a < 0 or idx_b < 0:
            continue

        line = _trace_pixels_to_line(pixels)
        if line is not None and line.length > 10:
            pair = (min(idx_a, idx_b), max(idx_a, idx_b))
            centerlines[pair] = line

    logger.info("Extracted %d centerline segments from vein mask", len(centerlines))
    return centerlines


def extract_anterior_boundary(
    vein_polygons: list[Polygon],
    intervein_polygons: list[Polygon],
    image_shape: tuple[int, int],
) -> Optional[tuple[int, LineString]]:
    """Extract the boundary between the most-anterior polygon and background.

    When no costal cell polygon exists, L1 sits at the boundary between the
    most-anterior intervein polygon (marginal cell) and the background (label=0)
    within the vein mask.  This function specifically extracts that boundary.

    Returns (polygon_index, LineString) or None if too short / not found.
    """
    if not vein_polygons or not intervein_polygons:
        return None

    h, w = image_shape

    # 1. Build label map and vein mask (same as extract_veins_from_mask)
    label_map = np.zeros((h, w), dtype=np.int32)
    for i, poly in enumerate(intervein_polygons):
        _fill_polygon(label_map, poly, i + 1)

    vein_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in vein_polygons:
        _fill_polygon(vein_mask, poly, 1)

    # 2. Voronoi partition
    background = (label_map == 0)
    _, nearest_indices = ndimage.distance_transform_edt(
        background, return_distances=True, return_indices=True
    )
    nearest_labels = label_map[nearest_indices[0], nearest_indices[1]]
    nearest_labels[~background] = label_map[~background]

    # 3. Find the most-anterior polygon (lowest Y centroid)
    ant_idx = min(
        range(len(intervein_polygons)),
        key=lambda i: intervein_polygons[i].centroid.y,
    )
    ant_label = ant_idx + 1

    # 4. Extract boundary pixels: ant_label pixels at the edge of the vein mask
    #    (nearest_labels has no 0s after distance transform, so we detect the
    #    boundary by checking which ant-polygon pixels neighbor outside the vein mask)
    is_ant_in_vein = (nearest_labels == ant_label) & (vein_mask > 0)

    padded_vm = np.pad(vein_mask, 1, mode='constant', constant_values=0)
    shifts_vm = [
        padded_vm[0:-2, 1:-1],  # up
        padded_vm[2:,   1:-1],  # down
        padded_vm[1:-1, 0:-2],  # left
        padded_vm[1:-1, 2:],    # right
    ]

    at_vein_edge = np.zeros((h, w), dtype=bool)
    for sv in shifts_vm:
        at_vein_edge |= (sv == 0)

    # All boundary pixels of the anterior polygon at the vein mask edge
    has_bg_neighbor = is_ant_in_vein & at_vein_edge

    # Filter to anterior-facing boundary only (Y < polygon centroid)
    ant_centroid_y = intervein_polygons[ant_idx].centroid.y
    ys_all, xs_all = np.where(has_bg_neighbor)
    anterior_mask = ys_all < ant_centroid_y
    has_bg_neighbor = np.zeros((h, w), dtype=bool)
    has_bg_neighbor[ys_all[anterior_mask], xs_all[anterior_mask]] = True

    ys, xs = np.where(has_bg_neighbor)
    if len(ys) < 5:
        return None

    pixels = list(zip(ys.tolist(), xs.tolist()))
    line = _trace_pixels_to_line(pixels)

    if line is None or line.length < 50:
        return None

    logger.info(
        "Anterior boundary: polygon %d, %d pixels, length=%.0f",
        ant_idx, len(pixels), line.length,
    )
    return (ant_idx, line)


def _fill_polygon(
    raster: np.ndarray, poly: Polygon, value: int
) -> None:
    """Fill a shapely Polygon (with holes) into a numpy raster."""
    # Exterior ring
    exterior_pts = np.array(poly.exterior.coords, dtype=np.int32)
    cv2.fillPoly(raster, [exterior_pts], value)
    # Interior rings (holes) — fill with 0
    for interior in poly.interiors:
        hole_pts = np.array(interior.coords, dtype=np.int32)
        cv2.fillPoly(raster, [hole_pts], 0)


def _extract_boundary_pixels(
    nearest_labels: np.ndarray,
    vein_mask: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Find pixels within the vein mask at Voronoi region boundaries.

    Uses vectorized numpy operations for speed.  Returns a dict mapping
    sorted (label_a, label_b) pairs to lists of (row, col) pixel coordinates.
    """
    h, w = nearest_labels.shape

    # Pad labels to handle boundary checking without conditionals
    padded = np.pad(nearest_labels, 1, mode='constant', constant_values=0)

    # Check all 4 cardinal shifts (sufficient for boundary detection;
    # 8-connectivity just adds diagonal redundancy)
    center = padded[1:-1, 1:-1]
    shifts = [
        padded[0:-2, 1:-1],  # up
        padded[2:,   1:-1],  # down
        padded[1:-1, 0:-2],  # left
        padded[1:-1, 2:],    # right
    ]

    # A pixel is a boundary pixel if any neighbor has a different label
    is_boundary = np.zeros((h, w), dtype=bool)
    # For each boundary pixel, record ONE differing neighbor label
    diff_label = np.zeros((h, w), dtype=np.int32)
    for shifted in shifts:
        different = (shifted != center) & (shifted > 0) & (center > 0)
        # Only record the first difference found (don't overwrite)
        new_boundary = different & ~is_boundary
        diff_label[new_boundary] = shifted[new_boundary]
        is_boundary |= different

    # Restrict to vein mask
    is_boundary &= (vein_mask > 0)

    # Extract boundary pixel coordinates and their label pairs
    ys, xs = np.where(is_boundary)
    center_labels = nearest_labels[ys, xs]
    neighbor_labels = diff_label[ys, xs]

    # Build pair dict
    label_a = np.minimum(center_labels, neighbor_labels)
    label_b = np.maximum(center_labels, neighbor_labels)

    boundary_dict: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(len(ys)):
        pair = (int(label_a[i]), int(label_b[i]))
        if pair not in boundary_dict:
            boundary_dict[pair] = []
        boundary_dict[pair].append((int(ys[i]), int(xs[i])))

    return boundary_dict


def _trace_pixels_to_line(
    pixels: list[tuple[int, int]],
    simplify_tolerance: float = 3.0,
) -> Optional[LineString]:
    """Convert boundary pixels into an ordered LineString using scan-median.

    The boundary band is typically 2+ pixels wide.  Scan-median collapses
    it to a 1-pixel-wide centerline by taking the median position along the
    perpendicular axis at each scan position.
    """
    if len(pixels) < 2:
        return None

    # Build binary image for connected components
    pixels_arr = np.array(pixels)
    min_y, min_x = pixels_arr.min(axis=0)
    max_y, max_x = pixels_arr.max(axis=0)

    pad = 2
    local_h = max_y - min_y + 1 + 2 * pad
    local_w = max_x - min_x + 1 + 2 * pad
    local_img = np.zeros((local_h, local_w), dtype=np.uint8)
    for y, x in pixels:
        local_img[y - min_y + pad, x - min_x + pad] = 1

    # Find connected components (8-connectivity)
    structure = ndimage.generate_binary_structure(2, 2)
    labeled, n_components = ndimage.label(local_img, structure=structure)

    # Process each component with scan-median, keep longest
    best_line: Optional[LineString] = None
    best_length = 0.0

    for comp_id in range(1, n_components + 1):
        comp_ys, comp_xs = np.where(labeled == comp_id)
        if len(comp_ys) < 2:
            continue

        global_xs = (comp_xs + min_x - pad).astype(np.float64)
        global_ys = (comp_ys + min_y - pad).astype(np.float64)

        line = _scan_median_to_line(global_xs, global_ys, simplify_tolerance)
        if line is not None and line.length > best_length:
            best_length = line.length
            best_line = line

    return best_line


def _scan_median_to_line(
    xs: np.ndarray,
    ys: np.ndarray,
    simplify_tolerance: float,
) -> Optional[LineString]:
    """Order pixels by scanning along the dominant axis, taking median position."""
    x_range = float(xs.max() - xs.min())
    y_range = float(ys.max() - ys.min())

    if x_range >= y_range:
        # Roughly horizontal: scan by X, compute median Y
        groups: dict[int, list[float]] = {}
        for x, y in zip(xs, ys):
            key = int(x)
            if key not in groups:
                groups[key] = []
            groups[key].append(y)
        sorted_keys = sorted(groups.keys())
        points = [(float(k), float(np.median(groups[k]))) for k in sorted_keys]
    else:
        # Roughly vertical: scan by Y, compute median X
        groups = {}
        for x, y in zip(xs, ys):
            key = int(y)
            if key not in groups:
                groups[key] = []
            groups[key].append(x)
        sorted_keys = sorted(groups.keys())
        points = [(float(np.median(groups[k])), float(k)) for k in sorted_keys]

    if len(points) < 2:
        return None

    line = LineString(points).simplify(simplify_tolerance)
    return line if line.length > 0 else None
