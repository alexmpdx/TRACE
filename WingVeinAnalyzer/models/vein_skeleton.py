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
    closing_kernel_size: int = 11,
) -> tuple[dict[tuple[int, int], LineString], np.ndarray, np.ndarray]:
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
    (centerlines, nearest_labels, vein_mask)
        centerlines: dict[(poly_idx_a, poly_idx_b), LineString]
        nearest_labels: Voronoi label array (for reuse by edge boundary extraction)
        vein_mask: binary vein mask array
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

    # 2b. Morphological closing to bridge small gaps in vein mask
    if closing_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
        )
        vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, kernel)

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

    # 6. Bridge dangling endpoints (connect nearby segment tips)
    centerlines = bridge_dangling_endpoints(centerlines, bridge_threshold=30.0)

    return centerlines, nearest_labels, vein_mask


def bridge_dangling_endpoints(
    centerlines: dict[tuple[int, int], LineString],
    bridge_threshold: float = 30.0,
) -> dict[tuple[int, int], LineString]:
    """Bridge dangling endpoints by extending the shorter segment.

    Collects all segment endpoints, identifies junctions (3+ endpoints from
    different segments within snap radius), and for each remaining dangling
    endpoint finds the nearest dangling endpoint from a different segment
    within bridge_threshold.  Extends the shorter segment to close the gap.
    """
    if not centerlines:
        return centerlines

    # Collect all endpoints: (x, y, key, endpoint_index)
    endpoints: list[tuple[float, float, tuple[int, int], int]] = []
    for key, line in centerlines.items():
        coords = list(line.coords)
        endpoints.append((coords[0][0], coords[0][1], key, 0))
        endpoints.append((coords[-1][0], coords[-1][1], key, -1))

    n = len(endpoints)
    snap_radius = bridge_threshold

    # Identify junction endpoints (near 3+ endpoints from different segments)
    junction_set: set[int] = set()
    for i in range(n):
        nearby_keys: set[tuple[int, int]] = {endpoints[i][2]}
        for j in range(n):
            if i == j:
                continue
            dx = endpoints[i][0] - endpoints[j][0]
            dy = endpoints[i][1] - endpoints[j][1]
            if dx * dx + dy * dy < snap_radius * snap_radius:
                nearby_keys.add(endpoints[j][2])
        if len(nearby_keys) >= 3:
            junction_set.add(i)

    # Build list of dangling endpoints (not at junctions)
    dangling: list[int] = [i for i in range(n) if i not in junction_set]

    # For each dangling endpoint, find nearest dangling from a different segment
    bridged: set[int] = set()
    result = dict(centerlines)
    bridges_made = 0

    for i in dangling:
        if i in bridged:
            continue
        xi, yi, key_i, ep_i = endpoints[i]
        best_j = -1
        best_dist2 = bridge_threshold * bridge_threshold
        for j in dangling:
            if j in bridged or j == i:
                continue
            xj, yj, key_j, ep_j = endpoints[j]
            if key_j == key_i:
                continue
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            if d2 < best_dist2:
                best_dist2 = d2
                best_j = j

        if best_j < 0:
            continue

        xj, yj, key_j, ep_j = endpoints[best_j]

        # Extend the shorter segment toward the bridge target
        line_i = result[key_i]
        line_j = result[key_j]
        if line_i.length <= line_j.length:
            # Extend line_i toward (xj, yj)
            result[key_i] = _extend_line(line_i, ep_i, xj, yj)
        else:
            # Extend line_j toward (xi, yi)
            result[key_j] = _extend_line(line_j, ep_j, xi, yi)

        bridged.add(i)
        bridged.add(best_j)
        bridges_made += 1

    if bridges_made:
        logger.info("Bridged %d dangling endpoint pair(s)", bridges_made)

    return result


def _extend_line(
    line: LineString, endpoint_idx: int, target_x: float, target_y: float,
) -> LineString:
    """Extend a LineString by appending/prepending a target coordinate."""
    coords = list(line.coords)
    target = (target_x, target_y)
    if endpoint_idx == 0:
        coords.insert(0, target)
    else:
        coords.append(target)
    return LineString(coords)


def extract_centerline_between_polygons(
    poly_a: Polygon,
    poly_b: Polygon,
    vein_polygons: list[Polygon],
    image_shape: tuple[int, int],
    min_length: float = 50.0,
) -> Optional[LineString]:
    """Extract a centerline between two adjacent polygons via local Voronoi partition.

    Used after split_merged_polygons() to recover the vein centerline that was
    missing because the two regions were a single polygon during the initial
    Voronoi extraction.

    Works in a local bounding box (poly_a ∪ poly_b + padding) for efficiency.
    Returns a LineString in global image coordinates, or None if too short.
    """
    from shapely.ops import unary_union

    h, w = image_shape
    pad = 20

    # 1. Bounding box of poly_a ∪ poly_b + padding, clipped to image bounds
    combined = unary_union([poly_a, poly_b])
    bx0, by0, bx1, by1 = combined.bounds
    x0 = max(0, int(bx0) - pad)
    y0 = max(0, int(by0) - pad)
    x1 = min(w, int(bx1) + pad + 1)
    y1 = min(h, int(by1) + pad + 1)
    local_h = y1 - y0
    local_w = x1 - x0
    if local_h < 5 or local_w < 5:
        return None

    # 2. Rasterize poly_a (label 1) and poly_b (label 2) into local seed map
    seed_map = np.zeros((local_h, local_w), dtype=np.int32)
    # Translate polygons to local coordinates
    from shapely.affinity import translate
    local_a = translate(poly_a, xoff=-x0, yoff=-y0)
    local_b = translate(poly_b, xoff=-x0, yoff=-y0)
    _fill_polygon(seed_map, local_a, 1)
    _fill_polygon(seed_map, local_b, 2)

    # 3. Rasterize vein mask (cropped to local bbox)
    local_vein_mask = np.zeros((local_h, local_w), dtype=np.uint8)
    for vpoly in vein_polygons:
        # Quick bounds check to skip distant polygons
        vbx0, vby0, vbx1, vby1 = vpoly.bounds
        if vbx1 < x0 or vbx0 > x1 or vby1 < y0 or vby0 > y1:
            continue
        local_vpoly = translate(vpoly, xoff=-x0, yoff=-y0)
        _fill_polygon(local_vein_mask, local_vpoly, 1)

    # 4. Voronoi partition via distance transform
    background = (seed_map == 0)
    if background.sum() == 0:
        return None
    _, nearest_indices = ndimage.distance_transform_edt(
        background, return_distances=True, return_indices=True
    )
    nearest_labels = seed_map[nearest_indices[0], nearest_indices[1]]
    nearest_labels[~background] = seed_map[~background]

    # 5. Extract boundary pixels where label transitions 1↔2, restricted to vein mask
    padded = np.pad(nearest_labels, 1, mode='constant', constant_values=0)
    center = padded[1:-1, 1:-1]
    shifts = [
        padded[0:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, 0:-2],
        padded[1:-1, 2:],
    ]

    is_boundary = np.zeros((local_h, local_w), dtype=bool)
    for shifted in shifts:
        # Boundary between labels 1 and 2 specifically
        is_12 = ((center == 1) & (shifted == 2)) | ((center == 2) & (shifted == 1))
        is_boundary |= is_12

    # Restrict to vein mask
    is_boundary &= (local_vein_mask > 0)

    ys, xs = np.where(is_boundary)
    if len(ys) < 5:
        return None

    # 6. Trace to LineString (in local coordinates)
    pixels = list(zip(ys.tolist(), xs.tolist()))
    line = _trace_pixels_to_line(pixels)
    if line is None or line.length < min_length:
        return None

    # 7. Translate back to global coordinates
    global_coords = [(px + x0, py + y0) for px, py in line.coords]
    global_line = LineString(global_coords)

    logger.info(
        "Post-split centerline: %d boundary pixels, length=%.0fpx",
        len(ys), global_line.length,
    )
    return global_line


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


def extract_edge_boundary_veins(
    vein_polygons: list[Polygon],
    intervein_polygons: list[Polygon],
    image_shape: tuple[int, int],
    nearest_labels: np.ndarray,
    vein_mask: np.ndarray,
    existing_centerlines: dict[tuple[int, int], LineString],
    min_length: float = 100.0,
    max_length: float = 4000.0,
) -> dict[tuple[int, int], LineString]:
    """Extract vein segments at the edge of the vein mask.

    Specifically targets vein-mask-edge regions that are NOT already
    captured by inter-polygon boundaries: the Voronoi centerline between
    a polygon's region and the background within the vein mask.

    Only keeps segments that are plausibly veins (not the entire wing
    perimeter): length between min_length and max_length, reasonably
    narrow band width.

    Parameters
    ----------
    nearest_labels, vein_mask : pre-computed from extract_veins_from_mask
    existing_centerlines : already extracted inter-polygon boundaries

    Returns dict keyed as (poly_idx, -1) where -1 = background.
    """
    if not vein_polygons or not intervein_polygons:
        return {}

    h, w = image_shape

    # Find vein-mask edge pixels: vein_mask pixels whose neighbor is outside vein_mask
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
    at_vein_edge &= (vein_mask > 0)

    # Exclude pixels that are at a boundary between two labeled regions
    # (those are already captured by _extract_boundary_pixels)
    padded_nl = np.pad(nearest_labels, 1, mode='constant', constant_values=0)
    center_nl = padded_nl[1:-1, 1:-1]
    has_diff_neighbor = np.zeros((h, w), dtype=bool)
    for shift in [
        padded_nl[0:-2, 1:-1], padded_nl[2:, 1:-1],
        padded_nl[1:-1, 0:-2], padded_nl[1:-1, 2:],
    ]:
        has_diff_neighbor |= ((shift != center_nl) & (shift > 0) & (center_nl > 0))

    # Edge-only boundary pixels: at vein mask edge AND NOT between two polygons
    edge_boundary = at_vein_edge & ~has_diff_neighbor

    # Use connected components to separate distinct segments (8-connectivity)
    structure = ndimage.generate_binary_structure(2, 2)
    labeled_cc, n_cc = ndimage.label(edge_boundary, structure=structure)

    result: dict[tuple[int, int], LineString] = {}

    for cc_id in range(1, n_cc + 1):
        cc_ys, cc_xs = np.where(labeled_cc == cc_id)
        if len(cc_ys) < 10:
            continue

        # Determine which polygon label dominates this component
        labels_here = nearest_labels[cc_ys, cc_xs]
        # Use the most common label
        unique_labels, counts = np.unique(labels_here, return_counts=True)
        dominant_label = int(unique_labels[counts.argmax()])
        if dominant_label == 0:
            continue

        # Compute band width: if pixels span a wide area, it's the wing perimeter
        y_span = float(cc_ys.max() - cc_ys.min())
        x_span = float(cc_xs.max() - cc_xs.min())
        n_pixels = len(cc_ys)

        # Estimate band width as area / max_span
        max_span = max(x_span, y_span, 1.0)
        band_width = n_pixels / max_span

        # Skip wide bands (these are wing perimeter, not veins)
        # Vein edges are typically 2-10 pixels wide
        if band_width > 30:
            continue

        pixels = list(zip(cc_ys.tolist(), cc_xs.tolist()))
        line = _trace_pixels_to_line(pixels)
        if line is None:
            continue

        if line.length < min_length or line.length > max_length:
            continue

        poly_idx = dominant_label - 1
        key = (poly_idx, -1)

        # Check if this segment is near an existing centerline endpoint
        # (should be a continuation of an existing vein)
        near_existing = False
        for cl in existing_centerlines.values():
            if line.distance(cl) < 50:
                near_existing = True
                break

        if not near_existing:
            continue

        if key not in result or line.length > result[key].length:
            result[key] = line
            logger.info(
                "Edge boundary vein: polygon %d, %d pixels, length=%.0f, "
                "band_width=%.1f",
                poly_idx, n_pixels, line.length, band_width,
            )

    logger.info("Extracted %d edge boundary vein segments", len(result))
    return result


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
