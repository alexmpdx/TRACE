"""Vein mask → skeleton → NetworkX graph.

Skeletonizes the vein mask directly using one of 3 user-selectable methods,
then cleans up the skeleton with configurable pruning and collinear merging.

Skeletonization methods:
1. Boundary smoothing (preprocessing on the binary mask)
2. Distance-transform medial axis
3. Voronoi-based medial axis with angle filtering

Pruning methods (applied sequentially):
1. Distance-map approximation (r_endpoint/r_junction ratio)
2. Full boundary reconstruction significance
3. Multi-scale persistence
4. Single-scale + comparison
5. Single-scale filtering
"""

from __future__ import annotations

import logging
import math

import cv2
import networkx as nx
import numpy as np
from identify_features.models.datatypes import (
    PruneMethod,
    SkeletonGraph,
    SkeletonMethod,
)
from identify_features.utils.image_utils import rasterize_polygons
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Polygon
from skimage.morphology import medial_axis, skeletonize

logger = logging.getLogger(__name__)

# 8-connected neighbor offsets
_NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_skeleton_graph(
    vein_polygons: list[Polygon | MultiPolygon],
    image_shape: tuple[int, int],
    config: "PipelineConfig | None" = None,
) -> SkeletonGraph:
    """Full pipeline: vein polygons → binary mask → skeleton → graph.

    All parameters are read from *config* (a PipelineConfig instance).
    If *config* is None a default PipelineConfig is created.

    Returns:
        SkeletonGraph with the NetworkX graph and supporting arrays.
    """
    from identify_features.config import PipelineConfig

    if config is None:
        config = PipelineConfig()

    methods = config.skeleton_methods
    smooth_sigma = config.smooth_sigma
    prune_methods = config.prune_methods
    prune_min_length_px = config.prune_min_length_px  # None = auto from median vein width
    prune_radius_ratio = config.prune_radius_ratio_threshold
    prune_scale_sigmas = config.prune_scale_sigmas
    prune_single_scale_sigma = config.prune_single_scale_sigma
    collinear_min_angle = config.collinear_min_angle
    max_gap_px = config.to_px(config.bridge_max_gap_um)
    bridge_gap_fraction = config.bridge_gap_fraction
    direction_window_px = config.to_px(config.bridge_direction_window_um)
    min_combined_length_px = config.to_px(config.bridge_min_combined_length_um)
    bridge_min_facing_angle = config.bridge_min_facing_angle
    bridge_on_axis_max_angle = config.bridge_on_axis_max_angle
    bridge_on_axis_relaxed_cap = config.bridge_on_axis_relaxed_cap

    # Step 1: Rasterize vein polygons to binary mask
    vein_mask = rasterize_polygons(vein_polygons, image_shape)
    logger.info("Vein mask: %d non-zero pixels", np.count_nonzero(vein_mask))

    # Step 2: Optional boundary smoothing (preprocessing)
    if SkeletonMethod.BOUNDARY_SMOOTH in methods:
        vein_mask = _boundary_smooth(vein_mask, sigma=smooth_sigma)
        logger.info("After boundary smoothing (sigma=%.1f): %d pixels", smooth_sigma, np.count_nonzero(vein_mask))

    # Step 3: Skeletonize
    distance_map = None
    if SkeletonMethod.RIDGE in methods:
        skel, distance_map = _skeletonize_ridge(vein_mask, sigma=smooth_sigma)
    elif SkeletonMethod.VORONOI in methods:
        skel = _skeletonize_voronoi(vein_mask)
    elif SkeletonMethod.MEDIAL_AXIS in methods:
        skel, distance_map = _skeletonize_medial_axis(vein_mask)
    else:
        skel = _skeletonize_standard(vein_mask)

    logger.info("Skeleton: %d pixels", np.count_nonzero(skel))

    # Compute median vein width from the raw skeleton (before pruning)
    median_vein_width = _compute_median_vein_width(skel, distance_map, vein_mask)
    logger.info("Median vein width: %.1fpx", median_vein_width)

    # Step 4: Basic length-based pruning
    # Use median vein width if no explicit pixel threshold was set
    if prune_min_length_px is not None:
        prune_threshold = prune_min_length_px
    else:
        prune_threshold = max(10, int(median_vein_width * config.prune_min_length_vein_widths))
    skel = _prune_branches(skel, min_length=prune_threshold)
    logger.info("After basic pruning (min=%dpx): %d pixels", prune_threshold, np.count_nonzero(skel))

    # Step 5: Advanced pruning methods (applied sequentially)
    for method in prune_methods:
        if method == PruneMethod.DISTANCE_MAP:
            skel = _prune_distance_map(skel, vein_mask, distance_map, ratio_threshold=prune_radius_ratio)
        elif method == PruneMethod.FULL_BOUNDARY:
            skel = _prune_full_boundary(skel, vein_mask, distance_map)
        elif method == PruneMethod.MULTI_SCALE:
            skel = _prune_multi_scale(skel, vein_mask, sigmas=prune_scale_sigmas)
        elif method == PruneMethod.SINGLE_SCALE_COMPARE:
            skel = _prune_single_scale_compare(skel, vein_mask, sigma=prune_single_scale_sigma)
        elif method == PruneMethod.SINGLE_SCALE:
            skel = _prune_single_scale(vein_mask, sigma=prune_single_scale_sigma)

        logger.info("After %s pruning: %d pixels", method.value, np.count_nonzero(skel))

    # Step 6: Build graph from skeleton
    graph = _skeleton_to_graph(skel)
    logger.info("Raw graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    # Step 7: Contract degree-2 nodes
    graph = _simplify_graph(graph)
    logger.info("Simplified graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    # Step 7b: Merge nearby degree-2/3 junction nodes (tight radius)
    # Nearly overlapping nodes at junctions cause bridging and labeling issues.
    _merge_junction_nodes(graph, min_dist=median_vein_width * config.junction_merge_vein_widths)
    graph = _simplify_graph(graph)

    # Step 8: Gap bridging + re-simplify (iterative, hierarchical)
    # No collinear merge — preserves all degree-3 junctions.
    graph = _bridge_and_simplify(
        graph,
        max_gap_px=max_gap_px,
        gap_fraction=bridge_gap_fraction,
        direction_window_px=direction_window_px,
        min_combined_length_px=min_combined_length_px,
        min_facing_angle=bridge_min_facing_angle,
        max_on_axis_angle=bridge_on_axis_max_angle,
        on_axis_relaxed_cap=bridge_on_axis_relaxed_cap,
        collinear_min_angle=collinear_min_angle,
        collinear_min_edge_length=median_vein_width * 2,
        prune_min_length=prune_threshold,
        do_collinear_merge=False,
    )

    # Step 9: Remove overlapping/redundant edges
    _remove_redundant_edges(graph)
    graph = _simplify_graph(graph)

    # Step 10: Absorb or remove tiny segments (1x median vein width)
    # Must be below the pruning threshold to avoid cascading collapse
    _absorb_tiny_segments(graph, min_length=median_vein_width)
    graph = _simplify_graph(graph)

    # Step 11: Merge nodes closer than median vein width
    _merge_close_nodes(graph, min_dist=median_vein_width)
    graph = _simplify_graph(graph)

    # No final collinear merge or stub removal here — merge_through_junctions
    # in the vein tracer handles these with landmark-aware guards
    # (protected nodes, label protection, perpendicularity check).

    # Step 12: Remove isolated fragments shorter than 4x median vein width
    _remove_small_fragments(graph, min_length=median_vein_width * 4)

    # Remove isolated nodes (degree 0)
    isolated = [n for n in graph.nodes() if graph.degree(n) == 0]
    graph.remove_nodes_from(isolated)

    # Step 13: Second bridging pass — cleanup may have exposed new bridgeable endpoints.
    # No collinear merge here — the vein tracer handles that with landmark guards.
    graph = _bridge_and_simplify(
        graph,
        max_gap_px=config.to_px(config.bridge2_max_gap_um),
        gap_fraction=config.bridge2_gap_fraction,
        direction_window_px=config.to_px(config.bridge2_direction_window_um),
        min_combined_length_px=config.to_px(config.bridge2_min_combined_length_um),
        min_facing_angle=config.bridge2_min_facing_angle,
        max_on_axis_angle=config.bridge2_on_axis_max_angle,
        on_axis_relaxed_cap=config.bridge2_on_axis_relaxed_cap,
        collinear_min_angle=collinear_min_angle,
        collinear_min_edge_length=median_vein_width * 2,
        prune_min_length=prune_threshold,
        do_collinear_merge=False,
    )

    # Step 14: Final single-pass stub removal
    # Last step before snapping — removes tiny dead-end stubs that survived
    # earlier cleanup. Single sweep, no simplify after, no cascade.
    _remove_stubs_single_pass(graph, max_length=median_vein_width * config.final_stub_vein_widths)

    # Step 15: Snap edge LineString endpoints to node positions
    _snap_edge_endpoints(graph)

    logger.info("Final graph: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())

    return SkeletonGraph(
        graph=graph,
        vein_mask=vein_mask,
        skeleton=skel,
        image_shape=image_shape,
        distance_map=distance_map,
        median_vein_width_px=median_vein_width,
    )


# ---------------------------------------------------------------------------
# Skeletonization methods
# ---------------------------------------------------------------------------


def _boundary_smooth(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-blur the binary mask and re-threshold."""
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return (blurred > 127).astype(np.uint8) * 255


def _skeletonize_medial_axis(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-transform medial axis, re-thinned for clean 1px skeleton."""
    binary = mask > 0
    skel_raw, distance = medial_axis(binary, return_distance=True)
    skel = skeletonize(skel_raw)
    dist_map = np.zeros_like(distance)
    dist_map[skel] = distance[skel]
    return skel.astype(np.uint8) * 255, dist_map


def _skeletonize_voronoi(mask: np.ndarray) -> np.ndarray:
    """Voronoi-based medial axis with angle filtering."""
    from scipy.spatial import Voronoi

    binary = mask > 0
    eroded = ndimage.binary_erosion(binary)
    boundary = binary & ~eroded
    boundary_pts = np.argwhere(boundary)

    if len(boundary_pts) < 4:
        logger.warning("Too few boundary pixels for Voronoi — falling back to medial axis")
        skel, _ = _skeletonize_medial_axis(mask)
        return skel

    max_pts = 20000
    if len(boundary_pts) > max_pts:
        indices = np.random.default_rng(42).choice(len(boundary_pts), max_pts, replace=False)
        boundary_pts = boundary_pts[indices]

    vor_pts = boundary_pts[:, ::-1].astype(np.float64)
    vor = Voronoi(vor_pts)

    skel = np.zeros(mask.shape, dtype=np.uint8)
    min_angle = np.radians(60)

    for ridge_idx, (p1_idx, p2_idx) in enumerate(vor.ridge_points):
        v_indices = vor.ridge_vertices[ridge_idx]
        if -1 in v_indices:
            continue

        v1 = vor.vertices[v_indices[0]]
        v2 = vor.vertices[v_indices[1]]

        r1, c1 = int(round(v1[1])), int(round(v1[0]))
        r2, c2 = int(round(v2[1])), int(round(v2[0]))
        if not _in_bounds(r1, c1, mask.shape) or not _in_bounds(r2, c2, mask.shape):
            continue
        if not binary[r1, c1] or not binary[r2, c2]:
            continue

        bp1 = vor_pts[p1_idx]
        bp2 = vor_pts[p2_idx]
        mid = (np.array(v1) + np.array(v2)) / 2
        vec1 = bp1 - mid
        vec2 = bp2 - mid
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        if angle >= min_angle:
            cv2.line(skel, (c1, r1), (c2, r2), 255, 1)

    skel = skeletonize(skel > 0).astype(np.uint8) * 255
    return skel


def _skeletonize_standard(mask: np.ndarray) -> np.ndarray:
    """Standard Zhang-Suen morphological thinning."""
    return skeletonize(mask > 0).astype(np.uint8) * 255


def _skeletonize_ridge(
    mask: np.ndarray,
    sigma: float = 2.0,
    ridge_threshold: float = -0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-map ridge extraction via Hessian non-maximum suppression.

    Finds vein centerlines as ridges of the distance transform — pixels
    where the distance value is a local maximum perpendicular to the
    ridge direction. Produces inherently clean centerlines with few
    spurious branches.

    Args:
        mask: Binary vein mask (uint8, 255=vein).
        sigma: Gaussian smoothing sigma for Hessian computation.
        ridge_threshold: Minimum lambda2 eigenvalue (must be negative).
            More negative = only strong ridges. Default -0.05 captures
            even thick veins (~30px) with shallow ridges.

    Returns:
        (skeleton, distance_map) — same format as _skeletonize_medial_axis.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    from skimage.feature import peak_local_max

    binary = mask > 0
    dist = ndimage.distance_transform_edt(binary).astype(np.float64)

    # Step 1: Compute Hessian of the smoothed distance field
    Dxx = gaussian_filter(dist, sigma=sigma, order=[0, 2])
    Dyy = gaussian_filter(dist, sigma=sigma, order=[2, 0])
    Dxy = gaussian_filter(dist, sigma=sigma, order=[1, 1])

    # Step 2: Eigenvalues of the 2x2 Hessian at each pixel
    trace = Dxx + Dyy
    disc = np.sqrt(np.maximum((Dxx - Dyy) ** 2 + 4 * Dxy**2, 0))
    lambda2 = (trace - disc) / 2  # smaller (more negative) eigenvalue

    # Cross-ridge direction (eigenvector of lambda2)
    cross_x = Dxy.copy()
    cross_y = (lambda2 - Dxx).copy()
    cross_mag = np.hypot(cross_x, cross_y)
    cross_mag[cross_mag < 1e-10] = 1.0
    cross_x /= cross_mag
    cross_y /= cross_mag

    # Step 3: Non-maximum suppression along cross-ridge direction
    rows, cols = np.where(binary)
    if len(rows) == 0:
        return np.zeros(mask.shape, dtype=np.uint8), dist

    cx = cross_x[rows, cols]
    cy = cross_y[rows, cols]

    # Sample distance at +1 and -1 along cross-ridge direction
    coords_plus = np.array([rows + cy, cols + cx], dtype=np.float64)
    coords_minus = np.array([rows - cy, cols - cx], dtype=np.float64)

    vals = dist[rows, cols]
    vals_plus = map_coordinates(dist, coords_plus, order=1, mode="constant", cval=0)
    vals_minus = map_coordinates(dist, coords_minus, order=1, mode="constant", cval=0)

    # Ridge condition: local max AND strong negative curvature
    is_ridge = (vals >= vals_plus) & (vals >= vals_minus) & (lambda2[rows, cols] < ridge_threshold)

    ridge_mask = np.zeros(mask.shape, dtype=bool)
    ridge_mask[rows, cols] = is_ridge

    # Step 4: Fill junction gaps
    # At junctions, the distance field has saddle points where NMS fails.
    # Find distance-field local maxima (junction centers) and connect
    # nearby ridge fragments through them.
    ridge_mask = _fill_ridge_junction_gaps(ridge_mask, dist, binary)

    # Step 5: Thin to 1px and produce output
    skel = skeletonize(ridge_mask).astype(np.uint8) * 255

    # Distance map on the skeleton
    dist_map = np.zeros_like(dist)
    dist_map[skel > 0] = dist[skel > 0]

    logger.info(
        "Ridge extraction: %d ridge pixels (sigma=%.1f, threshold=%.2f)", np.count_nonzero(skel), sigma, ridge_threshold
    )
    return skel, dist_map


def _fill_ridge_junction_gaps(
    ridge_mask: np.ndarray,
    dist: np.ndarray,
    vein_mask: np.ndarray,
) -> np.ndarray:
    """Fill gaps in the ridge mask at junction zones.

    At vein junctions, the distance-field NMS produces gaps because
    the Hessian has saddle points. This fills them by:
    1. Finding distance-field local maxima (junction centers)
    2. Dilating the ridge mask into junction zones to reconnect fragments
    """
    from skimage.feature import peak_local_max

    result = ridge_mask.copy()

    # Find local maxima of the distance field inside the vein mask
    # These are junction centers and wide vein centers
    peaks = peak_local_max(
        dist,
        min_distance=10,
        threshold_abs=3.0,
        labels=vein_mask.astype(np.int32),
    )

    if len(peaks) == 0:
        return result

    # Create a junction zone: area around each peak that's inside the vein mask
    # and has high distance values
    junction_zone = np.zeros_like(ridge_mask)
    for r, c in peaks:
        radius = max(5, int(dist[r, c] * 1.5))
        r_min = max(0, r - radius)
        r_max = min(ridge_mask.shape[0], r + radius + 1)
        c_min = max(0, c - radius)
        c_max = min(ridge_mask.shape[1], c + radius + 1)
        local = junction_zone[r_min:r_max, c_min:c_max]
        local_mask = vein_mask[r_min:r_max, c_min:c_max]
        local_dist = dist[r_min:r_max, c_min:c_max]
        # Fill where: inside vein mask AND distance > 50% of peak distance
        threshold = dist[r, c] * 0.5
        local[(local_mask > 0) & (local_dist >= threshold)] = True

    # Dilate ridge into junction zones to connect fragments
    dilated = cv2.dilate(result.astype(np.uint8), np.ones((3, 3), np.uint8))
    result = result | (dilated.astype(bool) & junction_zone)

    # Also add the junction zone center pixels directly to ensure connectivity
    for r, c in peaks:
        if vein_mask[r, c]:
            result[r, c] = True

    return result


def _in_bounds(r: int, c: int, shape: tuple[int, ...]) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


# ---------------------------------------------------------------------------
# Basic pruning
# ---------------------------------------------------------------------------


def _prune_branches(
    skeleton: np.ndarray,
    min_length: int = 30,
) -> np.ndarray:
    """Remove short terminal branches iteratively."""
    skel = skeleton.copy()
    pruned = True
    while pruned:
        pruned = False
        endpoints = _find_endpoints(skel)
        for r, c in endpoints:
            branch = _trace_branch(skel, r, c)
            if len(branch) < min_length:
                for br, bc in branch:
                    skel[br, bc] = 0
                pruned = True
    return skel


def _find_endpoints(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Find skeleton pixels with exactly 1 neighbor."""
    endpoints = []
    rows, cols = np.where(skeleton > 0)
    for r, c in zip(rows, cols):
        n = sum(
            1 for dr, dc in _NEIGHBORS_8 if _in_bounds(r + dr, c + dc, skeleton.shape) and skeleton[r + dr, c + dc] > 0
        )
        if n == 1:
            endpoints.append((r, c))
    return endpoints


def _trace_branch(
    skeleton: np.ndarray,
    start_r: int,
    start_c: int,
) -> list[tuple[int, int]]:
    """Trace from endpoint to nearest junction. Returns branch pixels."""
    pixels = [(start_r, start_c)]
    visited = {(start_r, start_c)}
    r, c = start_r, start_c

    while True:
        neighbors = [
            (r + dr, c + dc)
            for dr, dc in _NEIGHBORS_8
            if _in_bounds(r + dr, c + dc, skeleton.shape)
            and skeleton[r + dr, c + dc] > 0
            and (r + dr, c + dc) not in visited
        ]
        if len(neighbors) == 0:
            break
        elif len(neighbors) == 1:
            r, c = neighbors[0]
            visited.add((r, c))
            total = sum(
                1
                for dr, dc in _NEIGHBORS_8
                if _in_bounds(r + dr, c + dc, skeleton.shape) and skeleton[r + dr, c + dc] > 0
            )
            if total > 2:
                break  # reached junction
            pixels.append((r, c))
        else:
            break  # multiple neighbors = junction
    return pixels


# ---------------------------------------------------------------------------
# Advanced pruning methods
# ---------------------------------------------------------------------------


def _prune_distance_map(
    skeleton: np.ndarray,
    vein_mask: np.ndarray,
    distance_map: np.ndarray | None,
    ratio_threshold: float = 0.3,
) -> np.ndarray:
    """Prune branches where endpoint radius / junction radius < threshold.

    Noise spurs taper from wide (at junction) to narrow (at tip).
    Real vein branches maintain consistent width.
    """
    if distance_map is None:
        # Compute distance map if not available
        distance_map = ndimage.distance_transform_edt(vein_mask > 0)

    skel = skeleton.copy()
    pruned = True
    while pruned:
        pruned = False
        endpoints = _find_endpoints(skel)
        for r, c in endpoints:
            branch = _trace_branch(skel, r, c)
            if len(branch) < 3:
                continue

            r_endpoint = distance_map[branch[0][0], branch[0][1]]
            r_junction = distance_map[branch[-1][0], branch[-1][1]]

            if r_junction > 0 and r_endpoint / r_junction < ratio_threshold:
                for br, bc in branch:
                    skel[br, bc] = 0
                pruned = True

    return skel


def _prune_full_boundary(
    skeleton: np.ndarray,
    vein_mask: np.ndarray,
    distance_map: np.ndarray | None,
) -> np.ndarray:
    """Prune by boundary reconstruction significance.

    For each terminal branch, compute the boundary arc length it generates
    by following the distance-transform gradient to the boundary from each
    skeleton pixel. Branches with small arc length are noise.
    """
    if distance_map is None:
        distance_map = ndimage.distance_transform_edt(vein_mask > 0)

    # Compute distance-transform gradient (points toward nearest boundary)
    grad_y, grad_x = np.gradient(distance_map)

    skel = skeleton.copy()
    total_boundary = _boundary_length(vein_mask)

    pruned = True
    while pruned:
        pruned = False
        endpoints = _find_endpoints(skel)
        for r, c in endpoints:
            branch = _trace_branch(skel, r, c)
            if len(branch) < 3:
                continue

            # Estimate boundary arc length for this branch
            boundary_pts = set()
            for br, bc in branch:
                radius = distance_map[br, bc]
                if radius < 1:
                    continue
                # Follow gradient to boundary (both sides)
                gx, gy = grad_x[br, bc], grad_y[br, bc]
                mag = math.hypot(gx, gy)
                if mag < 1e-6:
                    continue
                gx, gy = gx / mag, gy / mag
                for sign in (-1, 1):
                    bx = int(round(bc + sign * gx * radius))
                    by = int(round(br + sign * gy * radius))
                    if _in_bounds(by, bx, vein_mask.shape):
                        boundary_pts.add((by, bx))

            arc_length = len(boundary_pts)
            significance = arc_length / max(1, total_boundary)

            if significance < 0.005:  # < 0.5% of total boundary
                for br, bc in branch:
                    skel[br, bc] = 0
                pruned = True

    return skel


def _boundary_length(mask: np.ndarray) -> int:
    """Count boundary pixels of a binary mask."""
    binary = mask > 0
    eroded = ndimage.binary_erosion(binary)
    return int(np.count_nonzero(binary & ~eroded))


def _prune_multi_scale(
    skeleton: np.ndarray,
    vein_mask: np.ndarray,
    sigmas: list[float] | None = None,
) -> np.ndarray:
    """Keep only branches that persist at all smoothing scales."""
    if sigmas is None:
        sigmas = [2.0, 4.0, 8.0, 16.0]

    # Skeletonize at each scale
    scale_skeletons = []
    for sigma in sigmas:
        smoothed = _boundary_smooth(vein_mask, sigma=sigma)
        skel_s = skeletonize(smoothed > 0).astype(np.uint8) * 255
        # Dilate slightly for overlap tolerance
        skel_s = cv2.dilate(skel_s, np.ones((5, 5), np.uint8))
        scale_skeletons.append(skel_s)

    # Keep original skeleton pixels that overlap with ALL coarser skeletons
    skel = skeleton.copy()
    for scale_skel in scale_skeletons:
        skel[scale_skel == 0] = 0

    # Re-thin (dilation in scale skeletons can leave thick patches)
    skel = skeletonize(skel > 0).astype(np.uint8) * 255
    return skel


def _prune_single_scale_compare(
    skeleton: np.ndarray,
    vein_mask: np.ndarray,
    sigma: float = 4.0,
) -> np.ndarray:
    """Keep original branches that overlap with a smoothed skeleton."""
    smoothed = _boundary_smooth(vein_mask, sigma=sigma)
    smooth_skel = skeletonize(smoothed > 0).astype(np.uint8) * 255
    # Dilate smoothed skeleton for overlap tolerance
    smooth_skel = cv2.dilate(smooth_skel, np.ones((7, 7), np.uint8))

    skel = skeleton.copy()
    skel[smooth_skel == 0] = 0
    skel = skeletonize(skel > 0).astype(np.uint8) * 255
    return skel


def _prune_single_scale(
    vein_mask: np.ndarray,
    sigma: float = 4.0,
) -> np.ndarray:
    """Simply skeletonize the smoothed mask."""
    smoothed = _boundary_smooth(vein_mask, sigma=sigma)
    return skeletonize(smoothed > 0).astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# Skeleton → NetworkX graph
# ---------------------------------------------------------------------------


def _skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """Convert pixel skeleton to NetworkX graph with junction clustering."""
    skel_coords = np.argwhere(skeleton > 0)
    if len(skel_coords) == 0:
        return nx.Graph()

    neighbor_count = _compute_neighbor_counts(skeleton, skel_coords)

    node_pixels = set()
    for r, c in skel_coords:
        if neighbor_count[r, c] != 2:
            node_pixels.add((r, c))

    pixel_to_node, node_positions = _cluster_junctions(node_pixels)

    G = nx.Graph()
    edge_id = 0

    for node_id, (nr, nc) in node_positions.items():
        G.add_node(node_id, x=float(nc), y=float(nr))

    traced_segments: set[tuple[int, int]] = set()

    for px in node_pixels:
        px_node_id = pixel_to_node[px]
        r, c = px

        for dr, dc in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if (
                not _in_bounds(nr, nc, skeleton.shape)
                or skeleton[nr, nc] == 0
                or (nr, nc) in node_pixels
                or (nr, nc) in traced_segments
            ):
                continue

            segment_pixels = []
            local_visited: set[tuple[int, int]] = set()
            cr, cc = nr, nc
            reached_node = False

            while True:
                if (cr, cc) in node_pixels:
                    reached_node = True
                    break
                segment_pixels.append((cr, cc))
                local_visited.add((cr, cc))
                traced_segments.add((cr, cc))

                next_px = None
                for dr2, dc2 in _NEIGHBORS_8:
                    r2, c2 = cr + dr2, cc + dc2
                    if (
                        not _in_bounds(r2, c2, skeleton.shape)
                        or skeleton[r2, c2] == 0
                        or (r2, c2) in local_visited
                        or (r2, c2) == px
                    ):
                        continue
                    next_px = (r2, c2)
                    break
                if next_px is None:
                    break
                cr, cc = next_px

            if not reached_node:
                continue

            end_node_id = pixel_to_node[(cr, cc)]
            start_node_id = px_node_id
            if start_node_id == end_node_id:
                continue

            start_pos = node_positions[start_node_id]
            end_pos = node_positions[end_node_id]
            all_pts = [start_pos] + segment_pixels + [end_pos]
            line_coords = [(c, r) for r, c in all_pts]
            line = LineString(line_coords)

            if not G.has_edge(start_node_id, end_node_id):
                G.add_edge(
                    start_node_id,
                    end_node_id,
                    edge_id=edge_id,
                    line=line,
                    length_px=line.length,
                    pixel_count=len(segment_pixels) + 2,
                )
                edge_id += 1

    # Handle direct connections between adjacent junction clusters
    for px in node_pixels:
        r, c = px
        for dr, dc in _NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if (nr, nc) in node_pixels and pixel_to_node[px] != pixel_to_node[(nr, nc)]:
                n1 = pixel_to_node[px]
                n2 = pixel_to_node[(nr, nc)]
                if not G.has_edge(n1, n2):
                    p1 = node_positions[n1]
                    p2 = node_positions[n2]
                    line = LineString([(p1[1], p1[0]), (p2[1], p2[0])])
                    G.add_edge(
                        n1,
                        n2,
                        edge_id=edge_id,
                        line=line,
                        length_px=line.length,
                        pixel_count=2,
                    )
                    edge_id += 1

    return G


def _compute_neighbor_counts(
    skeleton: np.ndarray,
    skel_coords: np.ndarray,
) -> np.ndarray:
    neighbor_count = np.zeros(skeleton.shape, dtype=np.int32)
    for r, c in skel_coords:
        n = sum(
            1 for dr, dc in _NEIGHBORS_8 if _in_bounds(r + dr, c + dc, skeleton.shape) and skeleton[r + dr, c + dc] > 0
        )
        neighbor_count[r, c] = n
    return neighbor_count


def _cluster_junctions(
    node_pixels: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], int], dict[int, tuple[int, int]]]:
    """Cluster adjacent junction/endpoint pixels into single nodes."""
    pixel_to_node: dict[tuple[int, int], int] = {}
    node_positions: dict[int, tuple[int, int]] = {}
    visited: set[tuple[int, int]] = set()
    cluster_id = 0

    for px in node_pixels:
        if px in visited:
            continue
        component = []
        queue = [px]
        visited.add(px)
        while queue:
            cur = queue.pop(0)
            component.append(cur)
            r, c = cur
            for dr, dc in _NEIGHBORS_8:
                neighbor = (r + dr, c + dc)
                if neighbor in node_pixels and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        rows = [p[0] for p in component]
        cols = [p[1] for p in component]
        centroid = (int(round(sum(rows) / len(rows))), int(round(sum(cols) / len(cols))))

        for p in component:
            pixel_to_node[p] = cluster_id
        node_positions[cluster_id] = centroid
        cluster_id += 1

    return pixel_to_node, node_positions


# ---------------------------------------------------------------------------
# Collinear merging
# ---------------------------------------------------------------------------


def _collinear_merge(
    G: nx.Graph,
    min_angle: float = 150.0,
    min_edge_length: float = 0.0,
) -> nx.Graph:
    """At each degree-3+ junction, merge the most collinear edge pair.

    Two edges at a junction whose tangent vectors make an angle > min_angle
    (close to 180° = straight through) are merged into a single edge.
    The remaining edge(s) become spurs from a point on the merged edge.

    Edges shorter than min_edge_length are excluded from being merged.
    This prevents tiny stubs from being incorrectly fused with real veins.

    Iterates until no more merges are possible.
    """
    from identify_features.utils.geometry_utils import angle_between_vectors
    from identify_features.utils.graph_utils import edge_departure_direction

    result = G.copy()
    next_edge_id = (
        max(
            (d.get("edge_id", 0) for _, _, d in result.edges(data=True)),
            default=-1,
        )
        + 1
    )

    changed = True
    while changed:
        changed = False
        junctions = [n for n in result.nodes() if result.degree(n) >= 3]

        for node in junctions:
            if node not in result or result.degree(node) < 3:
                continue

            neighbors = list(result.neighbors(node))
            if len(neighbors) < 2:
                continue

            # Find the most collinear pair
            best_pair = None
            best_angle = 0.0

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    n1, n2 = neighbors[i], neighbors[j]
                    # Skip if either edge is too short
                    if min_edge_length > 0:
                        l1 = result[node][n1].get("length_px", 0)
                        l2 = result[node][n2].get("length_px", 0)
                        if l1 < min_edge_length or l2 < min_edge_length:
                            continue
                    d1 = edge_departure_direction(result, node, n1, 80.0)
                    d2 = edge_departure_direction(result, node, n2, 80.0)
                    # Angle between opposite directions (should be close to 180°)
                    angle = angle_between_vectors(d1, d2)
                    if angle > best_angle:
                        best_angle = angle
                        best_pair = (n1, n2)

            if best_pair is None or best_angle < min_angle:
                continue

            # Divergence check: if 2+ pairs exceed the collinear threshold,
            # this is a divergence junction (one stem, two branches).
            # Don't merge — it would destroy the divergence point.
            collinear_count = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    di = edge_departure_direction(result, node, neighbors[i], 80.0)
                    dj = edge_departure_direction(result, node, neighbors[j], 80.0)
                    if di is not None and dj is not None:
                        if angle_between_vectors(di, dj) >= min_angle:
                            collinear_count += 1
            if collinear_count >= 2:
                continue

            n1, n2 = best_pair

            # Perpendicularity guard: the unmerged edge(s) must branch off
            # at a steep angle from both merged edges.  If any unmerged edge
            # is roughly collinear with either merged edge (angle > 150°),
            # this is a divergence junction — don't merge.
            d1 = edge_departure_direction(result, node, n1, 80.0)
            d2 = edge_departure_direction(result, node, n2, 80.0)
            skip = False
            for nb in neighbors:
                if nb == n1 or nb == n2:
                    continue
                d_other = edge_departure_direction(result, node, nb, 80.0)
                if d_other is not None:
                    if d1 is not None and angle_between_vectors(d_other, d1) > 150:
                        skip = True
                        break
                    if d2 is not None and angle_between_vectors(d_other, d2) > 150:
                        skip = True
                        break
            if skip:
                continue

            e1_data = result[node][n1]
            e2_data = result[node][n2]

            # Merge the two edges into one bypassing the junction node
            merged_line = _merge_lines(e1_data["line"], e2_data["line"], node, result)

            # Remove the two edges, add the merged one
            result.remove_edge(node, n1)
            result.remove_edge(node, n2)

            if not result.has_edge(n1, n2):
                result.add_edge(
                    n1,
                    n2,
                    edge_id=next_edge_id,
                    line=merged_line,
                    length_px=merged_line.length,
                    pixel_count=e1_data.get("pixel_count", 0) + e2_data.get("pixel_count", 0),
                )
                next_edge_id += 1

            # If the junction node is now degree 0 or 1, clean up
            if result.degree(node) == 0:
                result.remove_node(node)
            # degree 1 or 2 will be handled by _simplify_graph later

            changed = True

    return result


# ---------------------------------------------------------------------------
# Graph simplification
# ---------------------------------------------------------------------------


def _bridge_and_simplify(
    G: nx.Graph,
    max_gap_px: float = 100.0,
    gap_fraction: float = 0.1,
    direction_window_px: float = 207.0,
    min_combined_length_px: float = 207.0,
    min_facing_angle: float = 150.0,
    max_on_axis_angle: float = 20.0,
    on_axis_relaxed_cap: float = 45.0,
    collinear_min_angle: float = 150.0,
    collinear_min_edge_length: float = 0.0,
    prune_min_length: int = 30,
    max_iterations: int = 10,
    do_collinear_merge: bool = True,
) -> nx.Graph:
    """Bridge nearby endpoint gaps and re-simplify, hierarchically.

    Bridges longest-edge pairs first (most reliable directions), then
    re-simplifies and repeats with progressively shorter edges.

    Gap distance is adaptive: min(max_gap_px, gap_fraction * max(edge_lengths)).
    On-axis angle is asymmetric: strict for the longer edge, relaxed for shorter.
    """
    result = G.copy()

    for iteration in range(max_iterations):
        bridges_added = _bridge_pass(
            result,
            max_gap_px=max_gap_px,
            gap_fraction=gap_fraction,
            direction_window_px=direction_window_px,
            min_combined_length_px=min_combined_length_px,
            min_facing_angle=min_facing_angle,
            max_on_axis_angle=max_on_axis_angle,
            on_axis_relaxed_cap=on_axis_relaxed_cap,
        )

        if bridges_added == 0:
            break

        logger.debug("Iteration %d: bridged %d gaps", iteration + 1, bridges_added)

        # Re-simplify + optional collinear merge
        result = _simplify_graph(result)
        if do_collinear_merge:
            result = _collinear_merge(result, min_angle=collinear_min_angle, min_edge_length=collinear_min_edge_length)
            result = _simplify_graph(result)

        # No stub pruning inside bridge loop — removing stubs at degree-3
        # junctions demotes them to degree-2, and subsequent simplify
        # iterations cascade through the demoted nodes, collapsing the graph.
        # Stubs are handled by the final single-pass removal at the end.

    return result


def _bridge_pass(
    G: nx.Graph,
    max_gap_px: float,
    gap_fraction: float,
    direction_window_px: float,
    min_combined_length_px: float,
    min_facing_angle: float,
    max_on_axis_angle: float,
    on_axis_relaxed_cap: float,
) -> int:
    """Single pass: find and add valid bridge edges. Returns count added.

    Gap distance is adaptive: min(max_gap_px, gap_fraction * max(edge_lengths)).
    On-axis angle is asymmetric: strict for longer edge, relaxed for shorter.
    Bridges longest-edge pairs first (sorted by combined length descending).
    """
    from identify_features.utils.geometry_utils import angle_between_vectors

    endpoints = [n for n in G.nodes() if G.degree(n) == 1]

    # Collect endpoint data with full-edge direction
    ep_info = []
    for n in endpoints:
        neighbor = list(G.neighbors(n))[0]
        edge_data = G[n][neighbor]
        edge_len = edge_data.get("length_px", 0)
        nd = G.nodes[n]
        direction = _full_edge_direction(G, n, direction_window_px)
        ep_info.append(
            {
                "node": n,
                "x": nd["x"],
                "y": nd["y"],
                "edge_len": edge_len,
                "direction": direction,
            }
        )

    # Sort by edge length descending — bridge longest pairs first
    ep_info.sort(key=lambda e: -e["edge_len"])

    bridged_nodes: set[int] = set()
    next_edge_id = (
        max(
            (d.get("edge_id", 0) for _, _, d in G.edges(data=True)),
            default=-1,
        )
        + 1
    )
    bridges_added = 0

    for i, ep1 in enumerate(ep_info):
        n1 = ep1["node"]
        if n1 in bridged_nodes:
            continue

        best = None
        best_score = float("inf")

        for j, ep2 in enumerate(ep_info):
            if j <= i:
                continue
            n2 = ep2["node"]
            if n2 in bridged_nodes:
                continue

            # Min combined edge length
            combined = ep1["edge_len"] + ep2["edge_len"]
            if combined < min_combined_length_px:
                continue

            # Adaptive gap distance: fraction of the longer edge, capped
            longer_len = max(ep1["edge_len"], ep2["edge_len"])
            adaptive_gap = min(max_gap_px, gap_fraction * longer_len)

            dist = math.hypot(ep2["x"] - ep1["x"], ep2["y"] - ep1["y"])
            if dist > adaptive_gap or dist < 0.5:
                continue

            d1 = ep1["direction"]
            d2 = ep2["direction"]
            if d1 is None or d2 is None:
                continue

            # Facing check
            facing_angle = angle_between_vectors(d1, d2)
            if facing_angle < min_facing_angle:
                continue

            # Asymmetric on-axis check
            ab = (ep2["x"] - ep1["x"], ep2["y"] - ep1["y"])
            ab_mag = math.hypot(*ab)
            if ab_mag < 1e-6:
                continue
            ab_unit = (ab[0] / ab_mag, ab[1] / ab_mag)
            ba_unit = (-ab_unit[0], -ab_unit[1])

            axis_angle_1 = angle_between_vectors(d1, ab_unit)
            axis_angle_2 = angle_between_vectors(d2, ba_unit)

            # Determine which edge is longer → gets strict angle
            shorter_len = min(ep1["edge_len"], ep2["edge_len"])
            if ep1["edge_len"] >= ep2["edge_len"]:
                strict_angle = axis_angle_1
                relaxed_angle = axis_angle_2
            else:
                strict_angle = axis_angle_2
                relaxed_angle = axis_angle_1

            # Strict check for longer edge
            if strict_angle > max_on_axis_angle:
                continue

            # Relaxed check for shorter edge (scales with length ratio)
            ratio = longer_len / max(shorter_len, 1.0)
            relaxed_threshold = min(
                on_axis_relaxed_cap,
                max_on_axis_angle * (1 + ratio * 0.1),
            )
            if relaxed_angle > relaxed_threshold:
                continue

            score = dist + strict_angle + relaxed_angle * 0.5
            if score < best_score:
                best_score = score
                best = ep2

        if best is not None:
            n2 = best["node"]
            bridge_line = _build_bridge_line(
                G,
                n1,
                n2,
                (ep1["x"], ep1["y"]),
                (best["x"], best["y"]),
                direction_window_px,
            )
            G.add_edge(
                n1,
                n2,
                edge_id=next_edge_id,
                line=bridge_line,
                length_px=bridge_line.length,
                pixel_count=max(2, int(bridge_line.length)),
            )
            next_edge_id += 1
            bridges_added += 1
            bridged_nodes.add(n1)
            bridged_nodes.add(n2)

    return bridges_added


def _full_edge_direction(
    G: nx.Graph,
    endpoint: int,
    window_px: float,
) -> tuple[float, float] | None:
    """Compute the outward direction at an endpoint using a window of the edge.

    Samples the edge from (endpoint - window_px) to endpoint, giving
    the direction the vein was heading when it terminated.
    """
    from identify_features.utils.graph_utils import edge_line_from_node

    neighbor = list(G.neighbors(endpoint))[0]
    line = edge_line_from_node(G, neighbor, endpoint)  # oriented toward endpoint

    if line.length < 2:
        return None

    # Use up to window_px of the edge's tail
    sample_len = min(window_px, line.length * 0.8)
    pt_a = line.interpolate(line.length - sample_len)
    pt_b = line.interpolate(line.length)  # = endpoint position

    dx = pt_b.x - pt_a.x
    dy = pt_b.y - pt_a.y
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return None
    return (dx / mag, dy / mag)


def _build_bridge_line(
    G: nx.Graph,
    n1: int,
    n2: int,
    p1: tuple[float, float],
    p2: tuple[float, float],
    direction_window_px: float,
) -> LineString:
    """Build a straight bridge LineString between two endpoints."""
    return LineString([p1, p2])


def _compute_median_vein_width(
    skeleton: np.ndarray,
    distance_map: np.ndarray | None,
    vein_mask: np.ndarray,
) -> float:
    """Compute the median full vein width from the distance map at skeleton pixels.

    distance_transform_edt(vein_mask) gives the half-width (distance to
    nearest non-vein pixel) at each vein pixel. Sampling at skeleton pixels
    gives the half-width at the vein center. 2 * median = median full width.
    """
    if distance_map is None:
        dist = ndimage.distance_transform_edt(vein_mask > 0)
    else:
        dist = distance_map

    skel_pixels = skeleton > 0
    half_widths = dist[skel_pixels]
    half_widths = half_widths[half_widths > 0]

    if len(half_widths) == 0:
        return 0.0

    return 2.0 * float(np.median(half_widths))


def _snap_edge_endpoints(G: nx.Graph) -> None:
    """Snap each edge LineString's start/end to its node positions (in-place).

    After graph simplification, edge LineStrings may start/end at
    original skeleton pixel positions rather than at the node centroid.
    This creates visible stubs. Fix by replacing the first/last
    coordinate of each LineString with the node's (x, y).
    """
    import math

    for u, v, data in G.edges(data=True):
        line = data.get("line")
        if line is None or line.is_empty:
            continue

        coords = list(line.coords)
        if len(coords) < 2:
            continue

        nd_u = G.nodes[u]
        nd_v = G.nodes[v]
        u_pos = (nd_u["x"], nd_u["y"])
        v_pos = (nd_v["x"], nd_v["y"])

        # Determine which end of the LineString is closer to u vs v
        d_start_u = math.hypot(coords[0][0] - u_pos[0], coords[0][1] - u_pos[1])
        d_start_v = math.hypot(coords[0][0] - v_pos[0], coords[0][1] - v_pos[1])

        if d_start_u <= d_start_v:
            # coords[0] is near u, coords[-1] is near v
            coords[0] = u_pos
            coords[-1] = v_pos
        else:
            # coords[0] is near v, coords[-1] is near u
            coords[0] = v_pos
            coords[-1] = u_pos

        data["line"] = LineString(coords)


def _merge_junction_nodes(G: nx.Graph, min_dist: float) -> None:
    """Merge degree-2 and degree-3 nodes that are within min_dist of each other.

    Only merges pairs where both nodes are degree 2 or 3. Keeps the
    higher-degree node; if equal, takes the median position.
    Single pass — collects all pairs first, then merges.
    """
    import math

    # Collect eligible pairs (both deg 2 or 3, within min_dist)
    pairs = []
    nodes = [n for n in G.nodes() if 2 <= G.degree(n) <= 3]
    for i, n1 in enumerate(nodes):
        nd1 = G.nodes[n1]
        for n2 in nodes[i + 1 :]:
            nd2 = G.nodes[n2]
            dist = math.hypot(nd1["x"] - nd2["x"], nd1["y"] - nd2["y"])
            if dist < min_dist:
                pairs.append((n1, n2, dist))

    # Sort by distance (merge closest first)
    pairs.sort(key=lambda p: p[2])

    merged = set()
    for n1, n2, dist in pairs:
        if n1 in merged or n2 in merged:
            continue
        if n1 not in G or n2 not in G:
            continue

        deg1 = G.degree(n1)
        deg2 = G.degree(n2)

        if deg1 > deg2:
            keep, drop = n1, n2
        elif deg2 > deg1:
            keep, drop = n2, n1
        else:
            # Same degree — keep one, set to median position
            keep, drop = n1, n2
            nd1, nd2 = G.nodes[n1], G.nodes[n2]
            G.nodes[keep]["x"] = (nd1["x"] + nd2["x"]) / 2
            G.nodes[keep]["y"] = (nd1["y"] + nd2["y"]) / 2

        # Remove direct edge if exists
        if G.has_edge(keep, drop):
            G.remove_edge(keep, drop)

        # Transfer drop's edges to keep
        for neighbor in list(G.neighbors(drop)):
            if neighbor == keep:
                continue
            edge_data = G[drop][neighbor].copy()
            G.remove_edge(drop, neighbor)
            if not G.has_edge(keep, neighbor):
                G.add_edge(keep, neighbor, **edge_data)

        G.remove_node(drop)
        merged.add(drop)
        logger.debug(
            "Merged junction node %d into %d (dist=%.0fpx, deg %d+%d)",
            drop,
            keep,
            dist,
            deg1,
            deg2,
        )

    if merged:
        logger.info("Junction merge: merged %d node pairs (radius=%.0fpx)", len(merged), min_dist)


def _merge_close_nodes(G: nx.Graph, min_dist: float) -> None:
    """Merge nodes closer than min_dist into a single node (in-place).

    When two nodes are spatially close (< min_dist), they likely
    represent the same junction point. Merge them: keep one node,
    transfer the other's edges to it, remove the other.

    If there's an edge between the two close nodes, it's removed
    (it's a tiny connecting segment). Other edges from the removed
    node are reconnected to the kept node.
    """
    import math

    changed = True
    while changed:
        changed = False
        nodes = list(G.nodes())

        for i, n1 in enumerate(nodes):
            if n1 not in G:
                continue
            nd1 = G.nodes[n1]

            for n2 in nodes[i + 1 :]:
                if n2 not in G:
                    continue
                nd2 = G.nodes[n2]

                dist = math.hypot(nd1["x"] - nd2["x"], nd1["y"] - nd2["y"])
                if dist >= min_dist:
                    continue

                # Keep the higher-degree node (more likely at a real junction)
                if G.degree(n2) > G.degree(n1):
                    keep, drop = n2, n1
                else:
                    keep, drop = n1, n2

                # Remove any direct edge between the pair
                if G.has_edge(keep, drop):
                    G.remove_edge(keep, drop)

                # Reconnect drop's remaining neighbors to keep
                for neighbor in list(G.neighbors(drop)):
                    if neighbor == keep:
                        continue
                    edge_data = G[drop][neighbor].copy()
                    G.remove_edge(drop, neighbor)
                    if not G.has_edge(keep, neighbor):
                        G.add_edge(keep, neighbor, **edge_data)

                G.remove_node(drop)
                changed = True
                logger.debug(
                    "Merged node %d into %d (dist=%.0fpx, kept deg=%d)",
                    drop,
                    keep,
                    dist,
                    G.degree(keep),
                )
                break  # restart after modification

            if changed:
                break


def _remove_small_fragments(G: nx.Graph, min_length: float) -> None:
    """Remove isolated graph fragments shorter than min_length (in-place).

    An isolated fragment is a connected component where all nodes are
    degree-1 (no connection to the rest of the graph). If the total
    edge length of the fragment is below min_length, remove it.
    """
    import networkx as nx

    for component in list(nx.connected_components(G)):
        # Check if this component is isolated (no degree-3+ junctions)
        has_junction = any(G.degree(n) >= 3 for n in component)
        if has_junction:
            continue

        # Total edge length in this component
        total_length = sum(G[u][v].get("length_px", 0) for u, v in G.edges() if u in component and v in component)

        if total_length < min_length:
            G.remove_nodes_from(component)
            logger.debug(
                "Removed isolated fragment: %d nodes, %.0fpx total",
                len(component),
                total_length,
            )


def _remove_stubs_single_pass(G: nx.Graph, max_length: float) -> None:
    """Remove degree-1 stubs at junctions in a single pass (no cascade).

    Collects all eligible stubs first, then removes them all at once.
    This prevents the cascade where removing one stub demotes a junction
    to degree-2, which then gets contracted, exposing more stubs.
    """
    to_remove = []
    for u, v, data in G.edges(data=True):
        length = data.get("length_px", 0)
        if length >= max_length:
            continue
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        if (deg_u == 1 and deg_v >= 3) or (deg_v == 1 and deg_u >= 3):
            free_node = u if deg_u == 1 else v
            to_remove.append((free_node, u, v, length))

    for free_node, u, v, length in to_remove:
        if free_node in G:
            G.remove_node(free_node)
            logger.debug("Removed stub (single pass): %d↔%d (%.0fpx)", u, v, length)

    if to_remove:
        logger.info("Single-pass stub removal: removed %d stubs (max %.0fpx)", len(to_remove), max_length)


def _remove_dead_end_stubs(G: nx.Graph, max_length: float) -> None:
    """Remove degree-1 stubs at junctions that are shorter than max_length.

    Only removes edges where one end is degree-1 and the other is degree-3+.
    Does NOT contract degree-2 nodes or cascade — just clips the stubs.
    Safe to run as a final cleanup after collinear merge.
    """
    changed = True
    while changed:
        changed = False
        for u, v, data in list(G.edges(data=True)):
            if not G.has_edge(u, v):
                continue
            length = data.get("length_px", 0)
            if length >= max_length:
                continue
            deg_u = G.degree(u)
            deg_v = G.degree(v)
            if (deg_u == 1 and deg_v >= 3) or (deg_v == 1 and deg_u >= 3):
                free_node = u if deg_u == 1 else v
                G.remove_node(free_node)
                changed = True
                logger.debug("Removed dead-end stub: %d↔%d (%.0fpx)", u, v, length)
                break  # restart scan


def _absorb_tiny_segments(G: nx.Graph, min_length: float = 30.0) -> None:
    """Absorb or remove tiny segments (in-place).

    Case 1: Tiny edge connects a single edge (A) to a junction (B, C).
            Merge the tiny edge into A (extend A to the junction).
    Case 2: Tiny edge is a dead-end stub at a junction (degree-1 on one
            end, junction on the other). Remove it.
    """
    changed = True
    while changed:
        changed = False
        for u, v, data in list(G.edges(data=True)):
            if not G.has_edge(u, v):
                continue
            if data.get("length_px", 0) >= min_length:
                continue

            deg_u = G.degree(u)
            deg_v = G.degree(v)

            # Case 2: dead-end stub at a junction — remove it
            # One end is degree-1 (free), other is degree-3+ (junction)
            if (deg_u == 1 and deg_v >= 3) or (deg_v == 1 and deg_u >= 3):
                free_node = u if deg_u == 1 else v
                G.remove_node(free_node)
                changed = True
                logger.debug(
                    "Removed tiny dead-end stub: %d↔%d (%.0fpx)",
                    u,
                    v,
                    data.get("length_px", 0),
                )
                continue

            if data.get("length_px", 0) >= min_length:
                continue

            # Case 1: bridges a single edge to a junction — merge into the single edge
            # One end is degree-2 (pass-through), other is degree-3+ (junction)
            if (deg_u == 2 and deg_v >= 3) or (deg_v == 2 and deg_u >= 3):
                pass_node = u if deg_u == 2 else v
                # pass_node connects to the tiny edge AND one other edge
                # Merge tiny edge into the other edge by contracting pass_node
                # (this is what _simplify_graph does for degree-2 nodes,
                # but we trigger it explicitly here for tiny edges)
                # Just remove the tiny edge — _simplify_graph will contract
                # the now-degree-1 pass_node on the next pass
                # Actually, easier: just let _simplify_graph handle it
                # after we remove the tiny edge
                pass  # _simplify_graph after this function handles it

            # Case 1 variant: both ends are degree-2 (tiny edge between two edges)
            # Just let _simplify_graph contract through it
            # Nothing to do here

    # Also remove remaining degree-1 stubs shorter than min_length
    # (catches stubs at degree-2 nodes that weren't at junctions)
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes()):
            if node not in G or G.degree(node) != 1:
                continue
            neighbor = list(G.neighbors(node))[0]
            edge_len = G[node][neighbor].get("length_px", 0)
            if edge_len < min_length:
                G.remove_node(node)
                changed = True


def _remove_redundant_edges(G: nx.Graph, buffer_px: float = 15.0) -> None:
    """Remove shorter edges that overlap spatially with a longer edge.

    An edge is redundant if most of its length lies within `buffer_px`
    of another longer edge. The shorter edge is removed in-place.
    """
    edges_by_length = sorted(
        [(u, v, d) for u, v, d in G.edges(data=True)],
        key=lambda e: -e[2].get("length_px", 0),
    )

    to_remove = []
    kept_lines = []  # (u, v, buffered_line)

    for u, v, data in edges_by_length:
        line = data.get("line")
        if line is None or line.is_empty:
            continue

        # Check if this edge is redundant (mostly inside a longer edge's buffer)
        is_redundant = False
        for ku, kv, kept_buf in kept_lines:
            if (ku, kv) == (u, v) or (ku, kv) == (v, u):
                continue
            try:
                overlap = line.intersection(kept_buf)
                if overlap.length >= line.length * 0.7:
                    is_redundant = True
                    break
            except Exception:
                continue

        if is_redundant:
            to_remove.append((u, v))
        else:
            kept_lines.append((u, v, line.buffer(buffer_px)))

    for u, v in to_remove:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            logger.debug("Removed redundant edge %d↔%d", u, v)

    if to_remove:
        logger.info("Removed %d redundant overlapping edges", len(to_remove))


def _simplify_graph(G: nx.Graph) -> nx.Graph:
    """Contract degree-2 nodes, merging their two edges into one."""
    simplified = G.copy()
    next_edge_id = (
        max(
            (d.get("edge_id", 0) for _, _, d in simplified.edges(data=True)),
            default=-1,
        )
        + 1
    )

    changed = True
    while changed:
        changed = False
        deg2_nodes = [n for n in simplified.nodes() if simplified.degree(n) == 2]

        for node in deg2_nodes:
            if node not in simplified:
                continue
            neighbors = list(simplified.neighbors(node))
            if len(neighbors) != 2:
                continue

            n1, n2 = neighbors
            if n1 == n2:
                simplified.remove_node(node)
                changed = True
                continue

            e1_data = simplified[node][n1]
            e2_data = simplified[node][n2]
            merged_line = _merge_lines(e1_data["line"], e2_data["line"], node, simplified)

            simplified.remove_node(node)
            if not simplified.has_edge(n1, n2):
                simplified.add_edge(
                    n1,
                    n2,
                    edge_id=next_edge_id,
                    line=merged_line,
                    length_px=merged_line.length,
                    pixel_count=e1_data.get("pixel_count", 0) + e2_data.get("pixel_count", 0),
                )
                next_edge_id += 1
            changed = True

    return simplified


def _merge_lines(
    line1: LineString,
    line2: LineString,
    via_node: int,
    G: nx.Graph,
) -> LineString:
    """Merge two LineStrings that meet at a shared node."""
    node_data = G.nodes[via_node]
    node_x, node_y = node_data["x"], node_data["y"]

    coords1 = list(line1.coords)
    coords2 = list(line2.coords)

    d_start1 = (coords1[0][0] - node_x) ** 2 + (coords1[0][1] - node_y) ** 2
    d_end1 = (coords1[-1][0] - node_x) ** 2 + (coords1[-1][1] - node_y) ** 2
    if d_start1 < d_end1:
        coords1 = coords1[::-1]

    d_start2 = (coords2[0][0] - node_x) ** 2 + (coords2[0][1] - node_y) ** 2
    d_end2 = (coords2[-1][0] - node_x) ** 2 + (coords2[-1][1] - node_y) ** 2
    if d_start2 > d_end2:
        coords2 = coords2[::-1]

    return LineString(coords1 + coords2[1:])
