"""Vein centerline extraction from vein mask via morphological skeletonization."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, Polygon
from skimage.morphology import skeletonize

from WingVeinAnalyzer.models.vein_map import (
    BRIDGE_THRESHOLD_UM,
    MIN_SEGMENT_LENGTH_UM,
    um_to_px,
)

logger = logging.getLogger(__name__)


@dataclass
class CenterlineResult:
    """Result of centerline extraction from vein mask."""

    centerlines: dict[tuple[int, int], LineString]
    vein_mask: np.ndarray
    nearest_labels: np.ndarray
    bridge_segments: dict[tuple[int, int], LineString] = dataclasses.field(default_factory=dict)


# Backward-compatible alias
VoronoiResult = CenterlineResult


def extract_veins_from_mask(
    vein_polygons: list[Polygon],
    image_shape: tuple[int, int],
    closing_kernel_size: int = 11,
    intervein_polygons: list[Polygon] | None = None,
    prune_threshold: int = 200,
    **kwargs,
) -> CenterlineResult:
    """Extract vein centerlines via morphological skeletonization of vein mask.

    Parameters
    ----------
    vein_polygons : list[Polygon]
        Polygons defining where vein tissue exists.
    image_shape : (height, width)
        Image dimensions for rasterization.
    closing_kernel_size : int
        Morphological closing kernel size for vein mask.
    intervein_polygons : list[Polygon] | None
        Intervein space polygons for polygon-pair assignment.
    prune_threshold : int
        Remove skeleton branches shorter than this (pixels).

    Returns
    -------
    CenterlineResult
        centerlines, vein_mask, nearest_labels, bridge_segments.
    """
    h, w = image_shape

    # 1. Rasterize vein mask to binary
    vein_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in vein_polygons:
        _fill_polygon(vein_mask, poly, 1)

    # 2. Morphological closing to bridge small gaps in vein mask
    if closing_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
        vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Skeletonize the vein mask
    skeleton = skeletonize(vein_mask > 0).astype(np.uint8)
    raw_pixels = int(skeleton.sum())
    logger.info("Raw skeleton: %d pixels from %d vein mask pixels", raw_pixels, int(vein_mask.sum()))

    # 4. Prune short terminal branches
    skeleton = _prune_skeleton(skeleton, min_branch_length=prune_threshold)
    pruned_pixels = int(skeleton.sum())
    logger.info("Pruned skeleton: %d pixels (%d removed)", pruned_pixels, raw_pixels - pruned_pixels)

    # 5. Trace skeleton into LineString segments at junction points
    centerlines = _trace_skeleton_segments(skeleton, min_length=um_to_px(MIN_SEGMENT_LENGTH_UM))
    logger.info("Traced %d centerline segments from skeleton", len(centerlines))

    # 6. Bridge dangling endpoints
    centerlines, bridge_segments = bridge_dangling_endpoints(centerlines)

    # 7. Build nearest-label map from intervein polygons via EDT (for downstream use)
    label_seeds = np.zeros((h, w), dtype=np.int32)
    if intervein_polygons:
        for i, poly in enumerate(intervein_polygons):
            _fill_polygon(label_seeds, poly, i + 1)

    background = label_seeds == 0
    _, nearest_indices = ndimage.distance_transform_edt(
        background,
        return_distances=True,
        return_indices=True,
    )
    nearest_labels = label_seeds[nearest_indices[0], nearest_indices[1]]
    nearest_labels[~background] = label_seeds[~background]

    return CenterlineResult(
        centerlines=centerlines,
        vein_mask=vein_mask,
        nearest_labels=nearest_labels,
        bridge_segments=bridge_segments,
    )


def bridge_dangling_endpoints(
    centerlines: dict[tuple[int, int], LineString],
    bridge_threshold: float | None = None,
) -> tuple[dict[tuple[int, int], LineString], dict[tuple[int, int], LineString]]:
    """Bridge dangling endpoints by extending the shorter segment.

    Collects all segment endpoints, identifies junctions (3+ endpoints from
    different segments within snap radius), and for each remaining dangling
    endpoint finds the nearest dangling endpoint from a different segment
    within bridge_threshold.  Extends the shorter segment to close the gap.
    """
    if bridge_threshold is None:
        bridge_threshold = um_to_px(BRIDGE_THRESHOLD_UM)
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
    bridge_segments: dict[tuple[int, int], LineString] = {}
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

        # Record the bridge segment (straight line between endpoints)
        bridge_line = LineString([(xi, yi), (xj, yj)])

        # Extend the shorter segment toward the bridge target
        line_i = result[key_i]
        line_j = result[key_j]
        if line_i.length <= line_j.length:
            bridge_segments[key_i] = bridge_line
            result[key_i] = _extend_line(line_i, ep_i, xj, yj)
        else:
            bridge_segments[key_j] = bridge_line
            result[key_j] = _extend_line(line_j, ep_j, xi, yi)

        bridged.add(i)
        bridged.add(best_j)
        bridges_made += 1

    if bridges_made:
        logger.info("Bridged %d dangling endpoint pair(s)", bridges_made)

    return result, bridge_segments


def _extend_line(
    line: LineString,
    endpoint_idx: int,
    target_x: float,
    target_y: float,
) -> LineString:
    """Extend a LineString by appending/prepending a target coordinate."""
    coords = list(line.coords)
    target = (target_x, target_y)
    if endpoint_idx == 0:
        coords.insert(0, target)
    else:
        coords.append(target)
    return LineString(coords)


def _prune_skeleton(skeleton: np.ndarray, min_branch_length: int = 200) -> np.ndarray:
    """Remove terminal branches shorter than min_branch_length (single-pass)."""
    pruned = skeleton.copy()
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(pruned, -1, kernel)

    endpoints = (pruned > 0) & (neighbor_count == 1)
    ep_ys, ep_xs = np.where(endpoints)

    to_remove: list[tuple[int, int]] = []
    for ey, ex in zip(ep_ys, ep_xs):
        branch_pixels: list[tuple[int, int]] = []
        cy, cx = int(ey), int(ex)
        visited: set[tuple[int, int]] = set()

        while True:
            branch_pixels.append((cy, cx))
            visited.add((cy, cx))

            neighbors = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < pruned.shape[0]
                        and 0 <= nx < pruned.shape[1]
                        and pruned[ny, nx] > 0
                        and (ny, nx) not in visited
                    ):
                        neighbors.append((ny, nx))

            if len(neighbors) == 0:
                break
            elif len(neighbors) == 1:
                cy, cx = neighbors[0]
            else:
                break

        if len(branch_pixels) < min_branch_length:
            to_remove.extend(branch_pixels)

    for py, px in to_remove:
        pruned[py, px] = 0

    return pruned


def _trace_skeleton_segments(
    skeleton: np.ndarray,
    min_length: float = 10.0,
) -> dict[tuple[int, int], LineString]:
    """Trace a skeleton into LineString segments between junction points.

    Finds junction pixels (3+ skeleton neighbors) and endpoints (1 neighbor),
    then traces paths between them.  Each segment gets a unique sequential key.
    Segments shorter than min_length are discarded.
    """
    h, w = skeleton.shape

    # Compute neighbor count for each skeleton pixel
    kern = np.ones((3, 3), dtype=np.uint8)
    kern[1, 1] = 0
    neighbor_count = cv2.filter2D(skeleton, -1, kern)

    # Classify skeleton pixels
    skel_mask = skeleton > 0
    junctions = skel_mask & (neighbor_count >= 3)
    endpoints = skel_mask & (neighbor_count == 1)

    # Start points for tracing: all endpoints + all junction pixels
    start_ys, start_xs = np.where(endpoints | junctions)
    junction_set = set(zip(*np.where(junctions))) if junctions.any() else set()

    # Track which pixels have been assigned to a segment
    visited_edges: set[tuple[int, int]] = set()
    centerlines: dict[tuple[int, int], LineString] = {}
    seg_id = 0

    for sy, sx in zip(start_ys, start_xs):
        # From each start pixel, try tracing along each unvisited neighbor
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = sy + dy, sx + dx
                if not (0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0):
                    continue
                if (ny, nx) in visited_edges and (sy, sx) not in junction_set:
                    continue

                # Trace from (sy, sx) through (ny, nx) until hitting
                # another junction, endpoint, or dead end
                path = [(int(sx), int(sy))]  # x, y order for LineString
                cy, cx = ny, nx
                prev_y, prev_x = sy, sx

                while True:
                    if (cy, cx) in visited_edges and (cy, cx) not in junction_set:
                        break

                    path.append((int(cx), int(cy)))

                    # Stop at junctions (but include the junction pixel)
                    if (cy, cx) in junction_set and len(path) > 1:
                        break

                    # Find next unvisited neighbor (excluding where we came from)
                    next_pixel = None
                    for ddy in (-1, 0, 1):
                        for ddx in (-1, 0, 1):
                            if ddy == 0 and ddx == 0:
                                continue
                            nny, nnx = cy + ddy, cx + ddx
                            if (nny, nnx) == (prev_y, prev_x):
                                continue
                            if 0 <= nny < h and 0 <= nnx < w and skeleton[nny, nnx] > 0:
                                next_pixel = (nny, nnx)
                                break
                        if next_pixel is not None:
                            break

                    if next_pixel is None:
                        break  # endpoint or dead end

                    prev_y, prev_x = cy, cx
                    cy, cx = next_pixel

                # Mark interior pixels (not junctions) as visited
                for px, py in path[1:-1]:
                    visited_edges.add((py, px))

                if len(path) >= 2:
                    line = LineString(path)
                    if line.length >= min_length:
                        centerlines[(seg_id, seg_id + 1)] = line
                        seg_id += 2

    logger.info(
        "Skeleton tracing: %d junctions, %d endpoints, %d segments",
        len(junction_set),
        int(endpoints.sum()),
        len(centerlines),
    )
    return centerlines


def _fill_polygon(raster: np.ndarray, poly: Polygon, value: int) -> None:
    """Fill a shapely Polygon (with holes) into a numpy raster."""
    # Exterior ring
    exterior_pts = np.array(poly.exterior.coords, dtype=np.int32)
    cv2.fillPoly(raster, [exterior_pts], value)
    # Interior rings (holes) — fill with 0
    for interior in poly.interiors:
        hole_pts = np.array(interior.coords, dtype=np.int32)
        cv2.fillPoly(raster, [hole_pts], 0)
