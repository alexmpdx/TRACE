"""Voronoi-based vein centerline extraction from vein mask."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


@dataclass
class VoronoiResult:
    """Result of Voronoi-based vein extraction from vein mask."""
    centerlines: dict[tuple[int, int], LineString]
    voronoi_polygons: list[Polygon]
    nearest_labels: np.ndarray
    vein_mask: np.ndarray
    hull_mask: np.ndarray
    seed_labels: np.ndarray


def extract_veins_from_mask(
    vein_polygons: list[Polygon],
    image_shape: tuple[int, int],
    closing_kernel_size: int = 11,
    min_seed_area: int = 10000,
    intervein_polygons: list[Polygon] | None = None,
) -> VoronoiResult:
    """Extract vein centerlines using Voronoi partition of vein mask.

    Hull-minus-vein connected components become Voronoi seeds with
    sequential labels 1..M. No input polygon overlap mapping is needed;
    each large connected component defines its own region.  Voronoi-derived
    polygons replace the input intervein polygons downstream.

    Parameters
    ----------
    vein_polygons : list[Polygon]
        Polygons defining where vein tissue exists.
    image_shape : (height, width)
        Image dimensions for rasterization.
    closing_kernel_size : int
        Morphological closing kernel size for vein mask.
    min_seed_area : int
        Minimum component area (in pixels) to use as a Voronoi seed.

    Returns
    -------
    VoronoiResult
        centerlines, voronoi_polygons, nearest_labels, vein_mask,
        hull_mask, seed_labels.
    """
    h, w = image_shape

    # 1. Rasterize vein mask to binary
    vein_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in vein_polygons:
        _fill_polygon(vein_mask, poly, 1)

    # 2. Morphological closing to bridge small gaps in vein mask
    if closing_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
        )
        vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Vein-mask hull seeding
    vm_ys, vm_xs = np.where(vein_mask > 0)
    step = max(1, len(vm_ys) // 10000)  # subsample for speed
    hull = MultiPoint(
        list(zip(vm_xs[::step].tolist(), vm_ys[::step].tolist()))
    ).convex_hull
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    _fill_polygon(hull_mask, hull, 1)

    seed_mask = (hull_mask > 0) & (vein_mask == 0)

    structure = ndimage.generate_binary_structure(2, 2)
    component_labels, n_components = ndimage.label(seed_mask, structure=structure)

    # 4. Assign sequential labels 1..M to large components
    seed_labels = np.zeros((h, w), dtype=np.int32)
    comp_sizes = ndimage.sum(
        np.ones_like(component_labels), component_labels,
        range(1, n_components + 1),
    )
    next_label = 1
    for comp_id in range(1, n_components + 1):
        if comp_sizes[comp_id - 1] < min_seed_area:
            continue
        comp_mask = (component_labels == comp_id)
        seed_labels[comp_mask] = next_label
        next_label += 1

    # Remove phantom seeds outside the annotation footprint
    if intervein_polygons:
        ivn_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in intervein_polygons:
            _fill_polygon(ivn_mask, poly, 1)
        phantom_ids = []
        for label_id in range(1, next_label):
            comp_pixels = seed_labels == label_id
            overlap = (comp_pixels & (ivn_mask > 0)).sum()
            if overlap / comp_pixels.sum() < 0.1:
                phantom_ids.append(label_id)
                seed_labels[comp_pixels] = 0
        if phantom_ids:
            # Renumber remaining labels sequentially
            old_to_new = {}
            new_id = 1
            for old_id in range(1, next_label):
                if old_id not in phantom_ids:
                    old_to_new[old_id] = new_id
                    new_id += 1
            new_seed_labels = np.zeros_like(seed_labels)
            for old_id, new_id_val in old_to_new.items():
                new_seed_labels[seed_labels == old_id] = new_id_val
            seed_labels = new_seed_labels
            next_label = new_id
            logger.info(
                "Removed %d phantom seed(s) outside annotation boundary",
                len(phantom_ids),
            )

    n_seeds = next_label - 1
    logger.info(
        "Hull seeding: %d large components from %d total (min_area=%d)",
        n_seeds, n_components, min_seed_area,
    )

    # 5. Voronoi partition via EDT
    background = (seed_labels == 0)
    _, nearest_indices = ndimage.distance_transform_edt(
        background, return_distances=True, return_indices=True
    )
    nearest_labels = seed_labels[nearest_indices[0], nearest_indices[1]]
    nearest_labels[~background] = seed_labels[~background]

    logger.info(
        "Voronoi partition: %d unique labels, vein mask covers %d pixels",
        len(np.unique(nearest_labels)), int(vein_mask.sum()),
    )

    # 6. Extract label boundaries within the vein mask
    boundary_pixels = _extract_boundary_pixels(nearest_labels, vein_mask)

    # 7. Filter and trace into LineStrings
    centerlines: dict[tuple[int, int], LineString] = {}
    for (label_a, label_b), pixels in boundary_pixels.items():
        idx_a = label_a - 1
        idx_b = label_b - 1
        if idx_a < 0 or idx_b < 0:
            continue

        line = _trace_pixels_to_line(pixels)
        if line is not None and line.length > 10:
            pair = (min(idx_a, idx_b), max(idx_a, idx_b))
            centerlines[pair] = line

    logger.info("Extracted %d centerline segments from vein mask", len(centerlines))

    # 8. Bridge dangling endpoints
    centerlines = bridge_dangling_endpoints(centerlines, bridge_threshold=30.0)

    # 9. Vectorize Voronoi regions into polygons
    voronoi_polygons = _vectorize_voronoi_regions(
        nearest_labels, hull_mask, n_seeds,
    )

    return VoronoiResult(
        centerlines=centerlines,
        voronoi_polygons=voronoi_polygons,
        nearest_labels=nearest_labels,
        vein_mask=vein_mask,
        hull_mask=hull_mask,
        seed_labels=seed_labels,
    )


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


def _vectorize_voronoi_regions(
    nearest_labels: np.ndarray,
    hull_mask: np.ndarray,
    n_labels: int,
) -> list[Polygon]:
    """Convert Voronoi label regions into Shapely Polygons.

    For each label 1..n_labels, creates a binary mask restricted to the
    hull, finds the largest contour, and converts to a Shapely Polygon.
    Returns a list where index i corresponds to label i+1.
    """
    polygons: list[Polygon] = []
    for label in range(1, n_labels + 1):
        region_mask = ((nearest_labels == label) & (hull_mask > 0)).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            coords = [(float(pt[0][0]), float(pt[0][1])) for pt in largest]
            if len(coords) >= 4:
                p = Polygon(coords)
                if p.is_valid and p.area > 100:
                    polygons.append(p)
                    continue
        # Placeholder for invalid/empty regions
        polygons.append(Polygon())

    logger.info("Vectorized %d Voronoi regions into polygons", len(polygons))
    return polygons


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

    # 4b. Extract the vein mask anterior to the most-anterior polygon and
    # skeletonize it to get L1's centerline (not just the edge).
    # L1 = vein tissue between the wing edge and the marginal cell.
    ant_bounds = intervein_polygons[ant_idx].bounds  # (xmin, ymin, xmax, ymax)
    ant_centroid_y = intervein_polygons[ant_idx].centroid.y

    # Region of interest: vein mask pixels in the anterior polygon's Voronoi
    # region, anterior to the polygon's 75th percentile Y.  This captures
    # the wing tip where L1 continues past the centroid.
    y_cutoff = ant_centroid_y + (ant_bounds[3] - ant_centroid_y) * 0.5
    roi = is_ant_in_vein.copy()
    roi[int(y_cutoff):, :] = False

    # Skeletonize this region to get the centerline
    from skimage.morphology import skeletonize as sk_skeletonize
    skeleton = sk_skeletonize(roi)

    skel_ys, skel_xs = np.where(skeleton)
    if len(skel_ys) < 5:
        return None

    pixels = list(zip(skel_ys.tolist(), skel_xs.tolist()))
    line = _trace_pixels_to_line(pixels)

    if line is None or line.length < 50:
        return None

    logger.info(
        "Anterior boundary: polygon %d, %d skeleton pixels, length=%.0f",
        ant_idx, len(skel_ys), line.length,
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


# ---------------------------------------------------------------------------
# Pre-Voronoi polygon splitting
# ---------------------------------------------------------------------------


def split_oversized_polygons(
    intervein_polygons: list[Polygon],
    image: np.ndarray,
    max_single_frac: float = 0.28,
) -> tuple[list[Polygon], list[LineString]]:
    """Detect and split oversized intervein polygons before Voronoi.

    When the pixel classifier merges two adjacent regions into one polygon
    (e.g. missing L5 merges discal + 3rd_posterior), the Voronoi gets too
    few seeds and produces wrong centerlines.  This function detects such
    oversized polygons by area fraction alone (no naming needed), splits
    them using erosion or image-guided ridge detection, and returns the
    updated polygon list plus synthetic centerlines along split boundaries.

    Returns (updated_polygons, synthetic_centerlines).
    """
    total_area = sum(p.area for p in intervein_polygons)
    if total_area == 0:
        return list(intervein_polygons), []

    polygons = list(intervein_polygons)
    synthetic_centerlines: list[LineString] = []

    for i in range(len(intervein_polygons)):
        poly = polygons[i]
        frac = poly.area / total_area
        if frac <= max_single_frac:
            continue

        logger.info(
            "Pre-Voronoi split: P%d area=%.1f%% exceeds threshold %.0f%%",
            i, frac * 100, max_single_frac * 100,
        )

        # Attempt 1: erosion-based splitting (larger erosion amounts)
        split_line, pieces = _pre_split_by_erosion(poly)

        # Attempt 2: image-guided ridge detection
        if split_line is None:
            split_line, pieces = _pre_split_by_ridge(
                poly, image, image.shape[:2],
            )

        if split_line is None or pieces is None:
            logger.warning(
                "  Could not split P%d — continuing with original polygon", i,
            )
            continue

        piece_a, piece_b = pieces
        logger.info(
            "  Split P%d into two pieces: %.0f px² + %.0f px², "
            "split line length=%.0f px",
            i, piece_a.area, piece_b.area, split_line.length,
        )

        # Replace original polygon with piece_a, append piece_b
        polygons[i] = piece_a
        polygons.append(piece_b)
        synthetic_centerlines.append(split_line)

    if len(polygons) != len(intervein_polygons):
        logger.info(
            "Pre-Voronoi split: %d → %d polygons, %d synthetic centerlines",
            len(intervein_polygons), len(polygons), len(synthetic_centerlines),
        )

    return polygons, synthetic_centerlines


def _pre_split_by_erosion(
    poly: Polygon,
    min_part_frac: float = 0.15,
) -> tuple[Optional[LineString], Optional[tuple[Polygon, Polygon]]]:
    """Split a polygon at its narrowest neck using progressive erosion.

    Uses larger erosion amounts than the post-identification split because
    pre-Voronoi merged polygons are typically very large.
    """
    erode_amount = None
    parts: list[Polygon] = []

    for amount in [20, 40, 60, 80, 100, 150]:
        eroded = poly.buffer(-amount)
        if eroded.is_empty:
            continue
        if eroded.geom_type == 'MultiPolygon':
            parts = [g for g in eroded.geoms if g.area > poly.area * min_part_frac]
            if len(parts) >= 2:
                erode_amount = amount
                break

    if erode_amount is None:
        return None, None

    # Sort by area, keep two largest
    parts.sort(key=lambda g: g.area, reverse=True)
    seed_a = parts[0]
    seed_b = parts[1]

    # Watershed fill: rasterize polygon and seeds, distance-transform fill
    bounds = poly.bounds  # (minx, miny, maxx, maxy)
    pad = 10
    x0 = int(bounds[0]) - pad
    y0 = int(bounds[1]) - pad
    x1 = int(bounds[2]) + pad + 1
    y1 = int(bounds[3]) + pad + 1
    local_h = y1 - y0
    local_w = x1 - x0

    from shapely.affinity import translate

    local_poly = translate(poly, xoff=-x0, yoff=-y0)
    local_a = translate(seed_a, xoff=-x0, yoff=-y0)
    local_b = translate(seed_b, xoff=-x0, yoff=-y0)

    poly_mask = np.zeros((local_h, local_w), dtype=np.uint8)
    _fill_polygon(poly_mask, local_poly, 1)

    seed_map = np.zeros((local_h, local_w), dtype=np.int32)
    _fill_polygon(seed_map, local_a, 1)
    _fill_polygon(seed_map, local_b, 2)
    seed_map[poly_mask == 0] = 0

    background = (seed_map == 0) & (poly_mask > 0)
    if background.sum() == 0:
        return None, None

    _, nearest_idx = ndimage.distance_transform_edt(background, return_indices=True)
    filled = seed_map[nearest_idx[0], nearest_idx[1]]
    filled[poly_mask == 0] = 0
    filled[seed_map > 0] = seed_map[seed_map > 0]

    # Extract boundary between label 1 and 2 as the split line
    split_line = _extract_split_boundary(filled, x0, y0)

    # Convert filled regions back to polygons
    piece_polys = _filled_to_polygons(filled, x0, y0)
    if piece_polys is None:
        return None, None

    return split_line, piece_polys


def _pre_split_by_ridge(
    poly: Polygon,
    image: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[Optional[LineString], Optional[tuple[Polygon, Polygon]]]:
    """Split a polygon using intensity-profile-based dark-band detection.

    In brightfield wing images, veins appear as dark bands. This function
    projects mean intensity along the polygon's minor axis, finds the
    darkest band (valley in the profile), and uses it to split the polygon.
    """
    from shapely.affinity import translate

    h, w = image_shape
    bounds = poly.bounds  # (minx, miny, maxx, maxy)
    pad = 5
    x0 = max(0, int(bounds[0]) - pad)
    y0 = max(0, int(bounds[1]) - pad)
    x1 = min(w, int(bounds[2]) + pad + 1)
    y1 = min(h, int(bounds[3]) + pad + 1)

    # Extract grayscale ROI
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    else:
        gray = image[y0:y1, x0:x1].copy()

    local_h, local_w = gray.shape

    # Create polygon mask in local coordinates
    local_poly = translate(poly, xoff=-x0, yoff=-y0)
    poly_mask = np.zeros((local_h, local_w), dtype=np.uint8)
    _fill_polygon(poly_mask, local_poly, 1)

    # Determine scan axis: scan along the minor axis of the polygon
    x_span = bounds[2] - bounds[0]
    y_span = bounds[3] - bounds[1]

    # Try both axes and pick the one with a clearer valley
    best_result = None
    best_valley_depth = 0.0

    for scan_axis in ("y", "x"):
        result = _find_dark_band(
            gray, poly_mask, local_h, local_w, scan_axis,
        )
        if result is not None:
            valley_pos, valley_depth, band_width = result
            if valley_depth > best_valley_depth:
                best_valley_depth = valley_depth
                best_result = (scan_axis, valley_pos, band_width)

    if best_result is None:
        return None, None

    scan_axis, valley_pos, band_width = best_result
    logger.info(
        "  Dark band found: axis=%s, pos=%d, depth=%.1f, width=%d",
        scan_axis, valley_pos, best_valley_depth, band_width,
    )

    # Build a split line through the valley position, following the
    # polygon boundary to get the right extent
    split_line = _trace_dark_band_line(
        gray, poly_mask, scan_axis, valley_pos, band_width,
        x0, y0,
    )
    if split_line is None or split_line.length < 50:
        return None, None

    # Split the polygon using the dark band line
    split_zone = split_line.buffer(max(band_width / 2, 5.0))
    remainder = poly.difference(split_zone)

    if remainder.is_empty:
        return None, None

    if remainder.geom_type == 'MultiPolygon':
        pieces = sorted(remainder.geoms, key=lambda g: g.area, reverse=True)
        big_pieces = [p for p in pieces if p.area > poly.area * 0.10]
        if len(big_pieces) >= 2:
            return split_line, (big_pieces[0], big_pieces[1])

    return None, None


def _find_dark_band(
    gray: np.ndarray,
    poly_mask: np.ndarray,
    local_h: int,
    local_w: int,
    scan_axis: str,
    band_size: int = 10,
) -> Optional[tuple[int, float, int]]:
    """Find the darkest band along a given axis within the polygon.

    Computes mean intensity in bands along scan_axis, smooths the profile,
    and finds the deepest valley. Returns (valley_pos, valley_depth, band_width)
    or None if no clear valley.
    """
    if scan_axis == "y":
        n_positions = local_h
    else:
        n_positions = local_w

    if n_positions < band_size * 5:
        return None

    # Compute mean intensity for each position along the scan axis
    profile = np.full(n_positions, np.nan)
    for pos in range(n_positions):
        if scan_axis == "y":
            row_mask = poly_mask[pos, :]
            row_vals = gray[pos, :]
        else:
            row_mask = poly_mask[:, pos]
            row_vals = gray[:, pos]

        valid = row_mask > 0
        if valid.sum() > 5:
            profile[pos] = float(np.mean(row_vals[valid]))

    # Interpolate NaN gaps
    valid_idx = np.where(~np.isnan(profile))[0]
    if len(valid_idx) < 20:
        return None

    profile_clean = np.interp(
        np.arange(n_positions), valid_idx, profile[valid_idx],
    )

    # Smooth with a wide kernel to find the broad valley
    kernel_size = max(band_size * 3, 31)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = np.convolve(
        profile_clean, np.ones(kernel_size) / kernel_size, mode='same',
    )

    # Find the deepest valley: exclude edges (first/last 15% of range)
    margin = int(n_positions * 0.15)
    interior = smoothed[margin:n_positions - margin]
    if len(interior) < 20:
        return None

    valley_idx = int(np.argmin(interior)) + margin
    valley_val = smoothed[valley_idx]

    # Compute depth relative to surrounding peaks
    left_peak = np.max(smoothed[margin:valley_idx]) if valley_idx > margin else smoothed[margin]
    right_peak = np.max(smoothed[valley_idx:n_positions - margin]) if valley_idx < n_positions - margin else smoothed[n_positions - margin - 1]
    surrounding = min(left_peak, right_peak)
    depth = surrounding - valley_val

    # The valley must be meaningful (at least 3 intensity units deep)
    if depth < 3.0:
        return None

    # Estimate band width: how wide is the valley (below surrounding - depth/2)?
    threshold = surrounding - depth / 2
    below = smoothed < threshold
    # Find the contiguous region around valley_idx that's below threshold
    left = valley_idx
    while left > 0 and below[left - 1]:
        left -= 1
    right = valley_idx
    while right < n_positions - 1 and below[right + 1]:
        right += 1
    band_width = right - left + 1

    return (valley_idx, depth, band_width)


def _trace_dark_band_line(
    gray: np.ndarray,
    poly_mask: np.ndarray,
    scan_axis: str,
    valley_pos: int,
    band_width: int,
    x_offset: int,
    y_offset: int,
) -> Optional[LineString]:
    """Trace a line through the darkest pixels at the valley position.

    For each position along the primary axis, finds the darkest pixel
    within the band width centered on valley_pos along the scan axis.
    """
    local_h, local_w = gray.shape
    half_band = max(band_width, 20)

    points: list[tuple[float, float]] = []

    if scan_axis == "y":
        # Valley is a horizontal band at row=valley_pos
        # For each column, find the darkest row near valley_pos
        y_lo = max(0, valley_pos - half_band)
        y_hi = min(local_h, valley_pos + half_band + 1)

        step = max(1, (local_w) // 200)  # Sample ~200 points
        for col in range(0, local_w, step):
            band_mask = poly_mask[y_lo:y_hi, col]
            if band_mask.sum() == 0:
                continue
            band_vals = gray[y_lo:y_hi, col].astype(np.float64)
            band_vals[band_mask == 0] = 999  # ignore non-polygon pixels
            best_row = int(np.argmin(band_vals)) + y_lo
            points.append((float(col + x_offset), float(best_row + y_offset)))
    else:
        # Valley is a vertical band at col=valley_pos
        x_lo = max(0, valley_pos - half_band)
        x_hi = min(local_w, valley_pos + half_band + 1)

        step = max(1, (local_h) // 200)
        for row in range(0, local_h, step):
            band_mask = poly_mask[row, x_lo:x_hi]
            if band_mask.sum() == 0:
                continue
            band_vals = gray[row, x_lo:x_hi].astype(np.float64)
            band_vals[band_mask == 0] = 999
            best_col = int(np.argmin(band_vals)) + x_lo
            points.append((float(best_col + x_offset), float(row + y_offset)))

    if len(points) < 5:
        return None

    line = LineString(points).simplify(5.0)
    return line if line.length > 0 else None


def _extract_split_boundary(
    filled: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> Optional[LineString]:
    """Extract the boundary line between label 1 and 2 in a filled array."""
    local_h, local_w = filled.shape

    padded = np.pad(filled, 1, mode='constant', constant_values=0)
    center = padded[1:-1, 1:-1]
    shifts = [
        padded[0:-2, 1:-1],
        padded[2:, 1:-1],
        padded[1:-1, 0:-2],
        padded[1:-1, 2:],
    ]

    is_boundary = np.zeros((local_h, local_w), dtype=bool)
    for shifted in shifts:
        is_12 = ((center == 1) & (shifted == 2)) | ((center == 2) & (shifted == 1))
        is_boundary |= is_12

    ys, xs = np.where(is_boundary)
    if len(ys) < 5:
        return None

    pixels = list(zip(ys.tolist(), xs.tolist()))
    line = _trace_pixels_to_line(pixels)
    if line is None:
        return None

    # Convert to global coordinates
    global_coords = [(px + x_offset, py + y_offset) for px, py in line.coords]
    return LineString(global_coords)


def _filled_to_polygons(
    filled: np.ndarray,
    x_offset: int,
    y_offset: int,
) -> Optional[tuple[Polygon, Polygon]]:
    """Convert a 2-label filled array back to two Polygons in global coords."""
    piece_polys: list[Polygon] = []
    for lbl in [1, 2]:
        piece_mask = ((filled == lbl) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if contours:
            largest = max(contours, key=cv2.contourArea)
            coords = [
                (float(pt[0][0]) + x_offset, float(pt[0][1]) + y_offset)
                for pt in largest
            ]
            if len(coords) >= 4:
                p = Polygon(coords)
                if p.is_valid and p.area > 100:
                    piece_polys.append(p)

    if len(piece_polys) < 2:
        return None

    piece_polys.sort(key=lambda p: p.area, reverse=True)
    return piece_polys[0], piece_polys[1]


def find_poly_pair_for_line(
    syn_line: LineString,
    polygons: list[Polygon],
    buffer_dist: float = 15.0,
) -> Optional[tuple[int, int]]:
    """Find which two polygons a synthetic centerline separates."""
    buffered = syn_line.buffer(buffer_dist)
    touching: list[int] = []

    for idx, poly in enumerate(polygons):
        if buffered.intersects(poly):
            touching.append(idx)

    if len(touching) >= 2:
        # Return the pair with largest overlap
        touching.sort(key=lambda idx: buffered.intersection(polygons[idx]).area, reverse=True)
        a, b = touching[0], touching[1]
        return (min(a, b), max(a, b))

    return None
