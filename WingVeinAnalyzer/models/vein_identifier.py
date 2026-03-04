"""Independent geometry-based vein and region identification with cross-validation."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from WingVeinAnalyzer.models.vein_labeler import (
    VeinAssignment,
    VeinStatus,
    _extract_costa,
    _merge_vein_lines,
)
from WingVeinAnalyzer.models.vein_skeleton import extract_centerline_between_polygons
from WingVeinAnalyzer.utils.skeleton_utils import smooth_line
from WingVeinAnalyzer.models.vein_map import (
    CROSSVEIN_CONNECTIONS,
    MAX_ANGLE_CHANGE_DEG,
    MIDLINE_PRIORS,
    REGION_AREA_PRIORS,
    REGION_EXPECTED_VEINS,
    REGION_Y_ORDER,
    STRAIGHTNESS_THRESHOLD,
    VEIN_BOUNDARIES,
    VEIN_LENGTH_PRIORS,
    VEIN_ORIENTATION_PRIORS,
    VEIN_Y_ORDER,
    SPATIAL_PRIORS_Y,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class JunctionPoint:
    """A triple (or higher) junction where vein segments converge."""
    x: float
    y: float
    segment_keys: list[tuple[int, int]]  # which centerline segments arrive here
    endpoint_indices: list[int]  # 0=start, -1=end for each segment


@dataclass
class MergedPath:
    """A merged vein path composed of one or more chained centerline segments."""
    segment_keys: list[tuple[int, int]]
    line: LineString
    orientation_deg: float = 0.0
    y_centroid_norm: float = 0.0
    x_centroid_norm: float = 0.0
    y_median_norm: float = 0.0
    y_min_norm: float = 0.0
    y_max_norm: float = 0.0
    y_iqr_lo: float = 0.0
    y_iqr_hi: float = 0.0
    x_min_norm: float = 0.0
    x_max_norm: float = 0.0
    straightness: float = 0.0
    length_px: float = 0.0


@dataclass
class VeinMetrics:
    """Per-vein accuracy metrics against ground truth."""
    vein_name: str
    hausdorff_px: float
    mean_deviation_px: float
    p95_deviation_px: float
    coverage_ratio: float  # fraction of GT within tolerance of predicted
    gt_length_px: float
    pred_length_px: float


@dataclass
class VeinValidationReport:
    """Results of vein centerline validation against ground truth."""
    per_vein: list[VeinMetrics] = field(default_factory=list)
    matched_count: int = 0
    total_gt_veins: int = 0
    total_pred_veins: int = 0
    mean_hausdorff: float = 0.0
    mean_coverage: float = 0.0


@dataclass
class SplitInfo:
    """Metadata about a polygon split performed by split_merged_polygons()."""
    orig_idx: int       # index of the original (oversized) polygon
    new_idx: int        # index of the newly appended polygon
    orig_name: str      # region name kept by the original polygon
    new_name: str       # region name assigned to the new polygon
    separating_vein: str  # vein that separates these two regions


@dataclass
class ValidationReport:
    """Results of cross-validation checks."""
    warnings: list[str] = field(default_factory=list)
    boundary_mismatches: list[str] = field(default_factory=list)
    area_flags: list[str] = field(default_factory=list)
    coverage_fraction: float = 0.0


@dataclass
class IdentificationResult:
    """Complete output from identify_veins_and_regions()."""
    assignments: list[VeinAssignment] = field(default_factory=list)
    poly_names: dict[int, str] = field(default_factory=dict)
    polygons: list[Polygon] = field(default_factory=list)  # possibly updated by splitting
    validation_report: ValidationReport = field(default_factory=ValidationReport)


# ---------------------------------------------------------------------------
# 3a. Junction Detection
# ---------------------------------------------------------------------------

def find_triple_junctions(
    centerlines: dict[tuple[int, int], LineString],
    snap_radius: float = 30.0,
) -> list[JunctionPoint]:
    """Find triple junctions where 3+ vein segments converge."""
    # Collect all segment endpoints
    endpoints: list[tuple[float, float, tuple[int, int], int]] = []
    for key, line in centerlines.items():
        coords = list(line.coords)
        endpoints.append((coords[0][0], coords[0][1], key, 0))
        endpoints.append((coords[-1][0], coords[-1][1], key, -1))

    n = len(endpoints)
    used = [False] * n
    junctions: list[JunctionPoint] = []

    for i in range(n):
        if used[i]:
            continue
        cluster_indices = [i]
        used[i] = True
        xi, yi = endpoints[i][0], endpoints[i][1]

        # Greedy: merge all unmatched neighbors within snap_radius
        for j in range(i + 1, n):
            if used[j]:
                continue
            xj, yj = endpoints[j][0], endpoints[j][1]
            if (xi - xj) ** 2 + (yi - yj) ** 2 < snap_radius ** 2:
                cluster_indices.append(j)
                used[j] = True

        if len(cluster_indices) < 3:
            continue

        # Compute cluster centroid
        cx = np.mean([endpoints[k][0] for k in cluster_indices])
        cy = np.mean([endpoints[k][1] for k in cluster_indices])

        seg_keys = [endpoints[k][2] for k in cluster_indices]
        ep_indices = [endpoints[k][3] for k in cluster_indices]

        junctions.append(JunctionPoint(
            x=float(cx), y=float(cy),
            segment_keys=seg_keys,
            endpoint_indices=ep_indices,
        ))

    logger.info("Found %d triple junctions", len(junctions))
    return junctions


# ---------------------------------------------------------------------------
# 3b. Segment Merging at Junctions
# ---------------------------------------------------------------------------

def _get_tangent_away_from_junction(
    line: LineString, endpoint_idx: int, tangent_dist: float = 80.0,
) -> np.ndarray:
    """Compute tangent vector pointing AWAY from a junction endpoint."""
    # Smooth the line to reduce pixel-level noise near junctions
    smoothed = smooth_line(line, sigma=5.0, sample_spacing=5.0)
    coords = list(smoothed.coords)
    if endpoint_idx == 0:
        # Junction is at start → tangent points from start toward interior
        jx, jy = coords[0]
        dist = min(tangent_dist, 0.2 * smoothed.length)
        pt = smoothed.interpolate(dist)
        return np.array([pt.x - jx, pt.y - jy])
    else:
        # Junction is at end → tangent points from end toward interior
        jx, jy = coords[-1]
        dist = max(0, smoothed.length - min(tangent_dist, 0.2 * smoothed.length))
        pt = smoothed.interpolate(dist)
        return np.array([pt.x - jx, pt.y - jy])


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in degrees [0, 180]."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _line_orientation(line: LineString) -> float:
    """Compute overall orientation of a LineString in degrees from horizontal."""
    coords = list(line.coords)
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    return math.degrees(math.atan2(abs(dy), abs(dx)))


def merge_segments_at_junctions(
    centerlines: dict[tuple[int, int], LineString],
    junctions: list[JunctionPoint],
    collinearity_threshold_deg: float = 45.0,
    min_gap_deg: float = 15.0,
) -> tuple[list[MergedPath], dict]:
    """Merge segments at triple junctions by tangent continuity.

    When the best collinear pair is clearly better than the second-best
    (gap > min_gap_deg), merges on angle alone.  When the gap is small,
    uses fragment length as a tiebreaker: the shortest fragment is likely
    the crossvein, so the two longer fragments should merge.

    Returns (paths, merge_decisions) where merge_decisions maps
    junction (x, y) → {"chosen": (key_a, key_b), "alternative": (key_c, key_d),
    "arrivals": [...]} for post-classification validation.
    """
    # Union-find for segment merging
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(k: tuple[int, int]) -> tuple[int, int]:
        while parent.get(k, k) != k:
            parent[k] = parent.get(parent[k], parent[k])
            k = parent[k]
        return k

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Initialize all segments as their own root
    for key in centerlines:
        parent[key] = key

    merge_decisions: dict[tuple[float, float], dict] = {}

    # At each junction, find the most collinear pair and merge them
    for junc in junctions:
        # Deduplicate: a segment might appear twice at a junction if both
        # endpoints are near the junction (very short segment)
        seen_keys: set[tuple[int, int]] = set()
        arrivals: list[tuple[tuple[int, int], int, np.ndarray]] = []
        for seg_key, ep_idx in zip(junc.segment_keys, junc.endpoint_indices):
            if seg_key in seen_keys:
                continue
            if seg_key not in centerlines:
                continue
            seen_keys.add(seg_key)
            tangent = _get_tangent_away_from_junction(
                centerlines[seg_key], ep_idx
            )
            arrivals.append((seg_key, ep_idx, tangent))

        if len(arrivals) < 3:
            continue

        # Score all pairs: deviation from 180° (lower = more collinear)
        pair_scores: list[tuple[float, int, int]] = []
        for i in range(len(arrivals)):
            for j in range(i + 1, len(arrivals)):
                angle = _angle_between_vectors(arrivals[i][2], arrivals[j][2])
                score = abs(angle - 180.0)
                pair_scores.append((score, i, j))

        pair_scores.sort()
        best_score, bi, bj = pair_scores[0]
        second_score = pair_scores[1][0] if len(pair_scores) > 1 else 999.0
        gap = second_score - best_score

        # Orientation guard: prevent merging a longitudinal (<25°) with
        # a crossvein (>60°) regardless of collinearity score
        key_a = arrivals[bi][0]
        key_b = arrivals[bj][0]
        ori_a = _line_orientation(centerlines[key_a])
        ori_b = _line_orientation(centerlines[key_b])
        orientation_mismatch = (
            (ori_a < 25 and ori_b > 60) or (ori_b < 25 and ori_a > 60)
        )

        # Determine the second-best pair for storing as alternative
        si, sj = pair_scores[1][1], pair_scores[1][2] if len(pair_scores) > 1 else (0, 0)
        alt_key_a = arrivals[si][0] if len(pair_scores) > 1 else None
        alt_key_b = arrivals[sj][0] if len(pair_scores) > 1 else None

        junc_coord = (junc.x, junc.y)

        if best_score >= collinearity_threshold_deg or orientation_mismatch:
            reason = "orientation mismatch" if orientation_mismatch else "above threshold"
            logger.debug(
                "Skipped merge at junction (%.0f, %.0f): "
                "best=%.1f°, gap=%.1f°, reason=%s",
                junc.x, junc.y, best_score, gap, reason,
            )
            continue

        if gap > min_gap_deg:
            # Clear winner on angle alone
            union(key_a, key_b)
            merge_decisions[junc_coord] = {
                "chosen": (key_a, key_b),
                "alternative": (alt_key_a, alt_key_b),
                "arrivals": [(a[0], a[1]) for a in arrivals],
            }
            logger.debug(
                "Merged %s + %s at junction (%.0f, %.0f), "
                "collinearity=%.1f°, gap=%.1f°",
                key_a, key_b, junc.x, junc.y,
                best_score, gap,
            )
        else:
            # Close angles — use fragment length as tiebreaker
            # The shortest fragment is likely the crossvein; the two
            # longer fragments should be merged
            lengths = [
                (k, centerlines[arrivals[k][0]].length)
                for k in range(len(arrivals))
            ]
            shortest_idx = min(range(len(arrivals)), key=lambda k: lengths[k][1])
            non_shortest = [k for k in range(len(arrivals)) if k != shortest_idx]
            length_pair = tuple(sorted(non_shortest))
            angle_pair = tuple(sorted([bi, bj]))

            if length_pair == angle_pair:
                # Length and angle agree — merge confidently
                union(key_a, key_b)
                merge_decisions[junc_coord] = {
                    "chosen": (key_a, key_b),
                    "alternative": (alt_key_a, alt_key_b),
                    "arrivals": [(a[0], a[1]) for a in arrivals],
                }
                logger.info(
                    "Merged %s + %s at junction (%.0f, %.0f), "
                    "collinearity=%.1f°, gap=%.1f° (length agrees)",
                    key_a, key_b, junc.x, junc.y,
                    best_score, gap,
                )
            else:
                # Disagreement — prefer length-based pair (shortest is
                # the crossvein, merge the other two)
                len_key_a = arrivals[non_shortest[0]][0]
                len_key_b = arrivals[non_shortest[1]][0]
                # Check orientation guard for the length-based pair too
                len_ori_a = _line_orientation(centerlines[len_key_a])
                len_ori_b = _line_orientation(centerlines[len_key_b])
                len_mismatch = (
                    (len_ori_a < 25 and len_ori_b > 55)
                    or (len_ori_b < 25 and len_ori_a > 55)
                )
                if not len_mismatch:
                    union(len_key_a, len_key_b)
                    merge_decisions[junc_coord] = {
                        "chosen": (len_key_a, len_key_b),
                        "alternative": (key_a, key_b),
                        "arrivals": [(a[0], a[1]) for a in arrivals],
                    }
                    logger.info(
                        "Merged %s + %s at junction (%.0f, %.0f), "
                        "collinearity=%.1f°, gap=%.1f° "
                        "(length tiebreak: shortest=%s)",
                        len_key_a, len_key_b, junc.x, junc.y,
                        best_score, gap, arrivals[shortest_idx][0],
                    )
                else:
                    # Length pair has orientation mismatch — fall back to angle
                    union(key_a, key_b)
                    merge_decisions[junc_coord] = {
                        "chosen": (key_a, key_b),
                        "alternative": (alt_key_a, alt_key_b),
                        "arrivals": [(a[0], a[1]) for a in arrivals],
                    }
                    logger.info(
                        "Merged %s + %s at junction (%.0f, %.0f), "
                        "collinearity=%.1f°, gap=%.1f° "
                        "(length pair had orientation mismatch)",
                        key_a, key_b, junc.x, junc.y,
                        best_score, gap,
                    )

    # Collect connected components
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for key in centerlines:
        root = find(key)
        if root not in groups:
            groups[root] = []
        groups[root].append(key)

    # Build MergedPath for each group
    paths: list[MergedPath] = []
    for root, seg_keys in groups.items():
        lines = [centerlines[k] for k in seg_keys]
        merged = _merge_vein_lines(lines)
        if merged is None or merged.length < 10:
            continue

        paths.append(MergedPath(
            segment_keys=seg_keys,
            line=merged,
            length_px=merged.length,
        ))

    logger.info(
        "Merged %d segments into %d paths", len(centerlines), len(paths),
    )
    return paths, merge_decisions


# ---------------------------------------------------------------------------
# 3b'. Post-Merge Sharp Turn Splitting
# ---------------------------------------------------------------------------

def _split_on_sharp_turns(
    paths: list[MergedPath],
    centerlines: dict[tuple[int, int], LineString],
    angle_threshold_deg: float = 70.0,
    step_dist: float = 50.0,
    min_path_length: float = 500.0,
    min_split_length: float = 200.0,
) -> list[MergedPath]:
    """Split merged paths at points where direction changes sharply.

    Walks each MergedPath in step_dist increments, computing the angle change
    at each step.  If a turn exceeds angle_threshold_deg, the path is split.
    Multi-segment paths split at the nearest segment boundary; single-segment
    paths split the LineString directly.  Paths shorter than min_path_length
    are never split, and both halves must be >= min_split_length.
    """
    result: list[MergedPath] = []

    for path in paths:
        if path.length_px < min_path_length:
            result.append(path)
            continue
        split_paths = _try_split_path(
            path, centerlines, angle_threshold_deg, step_dist,
            min_path_length, min_split_length,
        )
        result.extend(split_paths)

    if len(result) != len(paths):
        logger.info(
            "Sharp-turn splitting: %d paths → %d paths",
            len(paths), len(result),
        )
    return result


def _try_split_path(
    path: MergedPath,
    centerlines: dict[tuple[int, int], LineString],
    angle_threshold_deg: float,
    step_dist: float,
    min_path_length: float = 500.0,
    min_split_length: float = 200.0,
) -> list[MergedPath]:
    """Attempt to split a single MergedPath at its sharpest turn."""
    from shapely.ops import substring

    # Smooth the line before measuring angle changes to prevent
    # noisy pixel jitter from creating false sharp turns
    line = smooth_line(path.line, sigma=50.0, sample_spacing=5.0)
    n_steps = max(2, int(line.length / step_dist))

    # Sample points along the merged line
    sample_pts = [
        line.interpolate(i / n_steps, normalized=True)
        for i in range(n_steps + 1)
    ]

    # Find the sharpest turn at a position that can produce a valid split
    # (both resulting pieces must be >= min_split_length)
    max_angle = 0.0
    max_angle_dist = 0.0
    for i in range(1, len(sample_pts) - 1):
        dist = (i / n_steps) * line.length
        if dist < min_split_length or (line.length - dist) < min_split_length:
            continue  # skip turns too close to ends

        dx1 = sample_pts[i].x - sample_pts[i - 1].x
        dy1 = sample_pts[i].y - sample_pts[i - 1].y
        dx2 = sample_pts[i + 1].x - sample_pts[i].x
        dy2 = sample_pts[i + 1].y - sample_pts[i].y

        v1 = np.array([dx1, dy1])
        v2 = np.array([dx2, dy2])
        angle_change = _angle_between_vectors(v1, v2)

        if angle_change > max_angle:
            max_angle = angle_change
            max_angle_dist = dist

    # If narrow window didn't find a valid split, try wider tangent window
    # to catch gradual transitions (e.g., crossvein→longitudinal junctions)
    if max_angle <= angle_threshold_deg:
        wide_window = 3  # 3× step_dist ≈ 150px
        for i in range(wide_window, len(sample_pts) - wide_window):
            dist = (i / n_steps) * line.length
            if dist < min_split_length or (line.length - dist) < min_split_length:
                continue

            dx1 = sample_pts[i].x - sample_pts[i - wide_window].x
            dy1 = sample_pts[i].y - sample_pts[i - wide_window].y
            dx2 = sample_pts[i + wide_window].x - sample_pts[i].x
            dy2 = sample_pts[i + wide_window].y - sample_pts[i].y
            v1 = np.array([dx1, dy1])
            v2 = np.array([dx2, dy2])
            angle_change = _angle_between_vectors(v1, v2)
            if angle_change > max_angle:
                max_angle = angle_change
                max_angle_dist = dist

    if max_angle <= angle_threshold_deg:
        return [path]

    seg_keys = path.segment_keys

    if len(seg_keys) == 1:
        # Single segment: split the LineString at the sharp turn point
        if max_angle_dist < min_split_length or (line.length - max_angle_dist) < min_split_length:
            return [path]  # resulting halves too short

        line_a = substring(line, 0, max_angle_dist)
        line_b = substring(line, max_angle_dist, line.length)

        if line_a.length < min_split_length or line_b.length < min_split_length:
            return [path]

        logger.info(
            "Split single segment %s at %.0f° turn (dist=%.0f): "
            "%.0fpx + %.0fpx",
            seg_keys[0], max_angle, max_angle_dist,
            line_a.length, line_b.length,
        )

        result_paths = [
            MergedPath(
                segment_keys=list(seg_keys),
                line=line_a,
                length_px=line_a.length,
            ),
            MergedPath(
                segment_keys=list(seg_keys),
                line=line_b,
                length_px=line_b.length,
            ),
        ]

        # Recursively check sub-paths
        final: list[MergedPath] = []
        for p in result_paths:
            if p.length_px >= min_path_length:
                final.extend(_try_split_path(
                    p, centerlines, angle_threshold_deg, step_dist,
                    min_path_length, min_split_length,
                ))
            else:
                final.append(p)
        return final

    # Multi-segment: split at the nearest segment boundary
    cumulative_lengths = []
    running = 0.0
    for key in seg_keys:
        seg_line = centerlines.get(key)
        if seg_line is not None:
            running += seg_line.length
        cumulative_lengths.append(running)

    # Find split point: the segment boundary closest to max_angle_dist
    best_split_idx = 0
    best_diff = float("inf")
    for i in range(len(cumulative_lengths) - 1):
        diff = abs(cumulative_lengths[i] - max_angle_dist)
        if diff < best_diff:
            best_diff = diff
            best_split_idx = i + 1  # split AFTER this segment

    if best_split_idx == 0 or best_split_idx >= len(seg_keys):
        return [path]

    # Split into two groups
    group_a = seg_keys[:best_split_idx]
    group_b = seg_keys[best_split_idx:]

    logger.info(
        "Split path at %.0f° turn (dist=%.0f): %s → %s + %s",
        max_angle, max_angle_dist, seg_keys, group_a, group_b,
    )

    # Build new MergedPaths from each group
    result_paths = []
    for group in [group_a, group_b]:
        lines = [centerlines[k] for k in group if k in centerlines]
        merged = _merge_vein_lines(lines)
        if merged is None or merged.length < 10:
            continue
        result_paths.append(MergedPath(
            segment_keys=list(group),
            line=merged,
            length_px=merged.length,
        ))

    if not result_paths:
        return [path]

    # Recursively check sub-paths for additional sharp turns
    final = []
    for p in result_paths:
        if p.length_px >= min_path_length:
            final.extend(_try_split_path(
                p, centerlines, angle_threshold_deg, step_dist,
                min_path_length, min_split_length,
            ))
        else:
            final.append(p)

    return final


# ---------------------------------------------------------------------------
# 3c. Geometry-Based Vein Classification
# ---------------------------------------------------------------------------

def _compute_path_features(
    path: MergedPath,
    wing_bbox: tuple[float, float, float, float],
) -> None:
    """Compute geometric features for a merged path in-place."""
    min_x, min_y, max_x, max_y = wing_bbox
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y

    coords = np.array(path.line.coords)
    centroid = coords.mean(axis=0)
    path.x_centroid_norm = (centroid[0] - min_x) / bbox_w if bbox_w > 0 else 0.5
    path.y_centroid_norm = (centroid[1] - min_y) / bbox_h if bbox_h > 0 else 0.5

    # Line-based Y features (more robust than single centroid for curved veins)
    ys_norm = (coords[:, 1] - min_y) / bbox_h if bbox_h > 0 else np.full(len(coords), 0.5)
    xs_norm = (coords[:, 0] - min_x) / bbox_w if bbox_w > 0 else np.full(len(coords), 0.5)
    path.y_median_norm = float(np.median(ys_norm))
    path.y_min_norm = float(ys_norm.min())
    path.y_max_norm = float(ys_norm.max())
    path.y_iqr_lo = float(np.percentile(ys_norm, 25))
    path.y_iqr_hi = float(np.percentile(ys_norm, 75))
    path.x_min_norm = float(xs_norm.min())
    path.x_max_norm = float(xs_norm.max())

    # Orientation: angle from horizontal of the line connecting endpoints
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    path.orientation_deg = angle

    # Straightness: chord / arc
    chord = math.hypot(dx, dy)
    path.straightness = chord / path.length_px if path.length_px > 0 else 0.0
    path.length_px = path.line.length


def classify_merged_paths(
    paths: list[MergedPath],
    wing_bbox: tuple[float, float, float, float],
    midline=None,
    junctions: list[JunctionPoint] | None = None,
) -> dict[str, MergedPath]:
    """Classify merged paths into named veins by geometry.

    Identifies crossveins FIRST (they're reliably identified by steep
    orientation), then uses their positions to inform longitudinal scoring.
    """
    min_x, min_y, max_x, max_y = wing_bbox
    bbox_w = max_x - min_x

    # Compute features for all paths
    for p in paths:
        _compute_path_features(p, wing_bbox)

    # Split into longitudinal vs crossvein candidates
    longitudinals: list[MergedPath] = []
    crossveins: list[MergedPath] = []

    # Max plausible crossvein length: ~15% of wing span
    max_crossvein_len = bbox_w * 0.15 if bbox_w > 0 else 600.0
    max_crossvein_len = max(max_crossvein_len, 400.0)  # floor of 400px

    for p in paths:
        jn = (_count_junction_endpoints(p, junctions) if junctions else 0)
        if p.orientation_deg > 60 and p.length_px < max_crossvein_len:
            crossveins.append(p)
        elif p.orientation_deg < 30:
            longitudinals.append(p)
        elif p.orientation_deg >= 60 and p.length_px >= max_crossvein_len:
            # Steep but too long for crossvein — likely an artifact, skip
            logger.info(
                "Skipping steep long path (%.0f°, %.0fpx) — too long for crossvein",
                p.orientation_deg, p.length_px,
            )
        elif jn == 2 and p.orientation_deg > 40 and p.length_px < max_crossvein_len:
            # Both endpoints at triple junctions — strong crossvein signal
            logger.info(
                "Junction-promoted crossvein (%.0f°, %.0fpx, 2 junction endpoints)",
                p.orientation_deg, p.length_px,
            )
            crossveins.append(p)
        elif p.orientation_deg >= 50 and p.length_px <= 300:
            # Ambiguous 50-60° but short — crossvein candidate
            crossveins.append(p)
        else:
            # 30-50° or >300px — longitudinal (all real crossveins are >60°)
            longitudinals.append(p)

    vein_map: dict[str, MergedPath] = {}

    # Anchor L3/L4 from midline FIRST (most reliable — midline center is stable)
    if midline is not None and len(longitudinals) >= 2:
        l3, l4 = _anchor_l3_l4_from_midline(longitudinals, midline)
        if l3:
            vein_map["L3"] = l3
        if l4:
            vein_map["L4"] = l4

    # Assign crossveins (ACV/PCV by orientation + position)
    _assign_crossveins(crossveins, vein_map)

    # Validate crossveins: demote any that aren't near >=2 longitudinal candidates
    # L3/L4 are now in the vein_map, so they count as known longitudinals for
    # proximity checks
    known_longitudinals = longitudinals + [
        vein_map[k] for k in ("L3", "L4") if k in vein_map
    ]
    demoted = _validate_crossveins(vein_map, known_longitudinals, crossveins,
                                    junctions=junctions)
    longitudinals.extend(demoted)

    # Assign remaining longitudinals (L1, L2, L5)
    if midline is not None and ("L3" in vein_map or "L4" in vein_map):
        _assign_remaining_longitudinals(longitudinals, midline, vein_map)
    else:
        # No midline or no anchors — fall back to scored assignment
        acv_path = vein_map.get("ACV")
        pcv_path = vein_map.get("PCV")
        long_map = _assign_longitudinals_scored(
            longitudinals, bbox_w, acv_path, pcv_path, midline=midline,
        )
        vein_map.update(long_map)

    logger.info(
        "Classified veins: %s",
        {k: f"{v.length_px:.0f}px" for k, v in vein_map.items()},
    )
    return vein_map


def _count_junction_endpoints(
    path: MergedPath,
    junctions: list[JunctionPoint],
    snap_radius: float = 40.0,
) -> int:
    """Count how many of a path's endpoints are near a triple junction (0, 1, or 2)."""
    coords = list(path.line.coords)
    start = coords[0]
    end = coords[-1]
    count = 0
    for pt in (start, end):
        for j in junctions:
            if (pt[0] - j.x) ** 2 + (pt[1] - j.y) ** 2 <= snap_radius ** 2:
                count += 1
                break
    return count


def _assign_crossveins(
    crossveins: list[MergedPath],
    vein_map: dict[str, MergedPath],
) -> None:
    """Assign crossvein identities (ACV/PCV) by position."""
    if len(crossveins) >= 2:
        # ACV is more anterior (lower median Y), PCV is more posterior
        crossveins.sort(key=lambda p: p.y_median_norm)
        vein_map["ACV"] = crossveins[0]
        vein_map["PCV"] = crossveins[1]
    elif len(crossveins) == 1:
        cv = crossveins[0]
        # With only one crossvein, determine identity by median Y-position
        # ACV is more anterior (lower Y), PCV is more posterior (higher Y)
        acv_mid = (SPATIAL_PRIORS_Y["L3"][1] + SPATIAL_PRIORS_Y["L4"][0]) / 2
        pcv_mid = (SPATIAL_PRIORS_Y["L4"][1] + SPATIAL_PRIORS_Y["L5"][0]) / 2
        if abs(cv.y_median_norm - acv_mid) <= abs(cv.y_median_norm - pcv_mid):
            vein_map["ACV"] = cv
        else:
            vein_map["PCV"] = cv


def _validate_crossveins(
    vein_map: dict[str, MergedPath],
    longitudinals: list[MergedPath],
    crossveins: list[MergedPath],
    proximity_threshold: float = 100.0,
    min_nearby_longitudinals: int = 2,
    junctions: list[JunctionPoint] | None = None,
) -> list[MergedPath]:
    """Validate crossvein assignments by checking proximity to longitudinals.

    A real crossvein connects two longitudinal veins, so it should be within
    proximity_threshold of at least min_nearby_longitudinals longitudinal
    candidates.  If both endpoints are at triple junctions, relax to 1.
    Demotes failures back to the longitudinal pool.
    """
    demoted: list[MergedPath] = []
    cv_names_to_check = [n for n in ("ACV", "PCV") if n in vein_map]

    for cv_name in cv_names_to_check:
        cv_path = vein_map[cv_name]
        nearby = sum(
            1 for lp in longitudinals
            if cv_path.line.distance(lp.line) < proximity_threshold
        )
        # Relax threshold if both endpoints sit at triple junctions
        jn = (_count_junction_endpoints(cv_path, junctions) if junctions else 0)
        required = 1 if jn == 2 else min_nearby_longitudinals
        if nearby < required:
            logger.info(
                "Crossvein %s only near %d longitudinal(s) (need %d, "
                "threshold=%.0fpx, jn_endpoints=%d) — demoting to longitudinal pool",
                cv_name, nearby, required, proximity_threshold, jn,
            )
            demoted.append(cv_path)
            del vein_map[cv_name]

    # If any were demoted, reassign only the lost slot(s) from remaining candidates
    if demoted:
        remaining_cv = [
            p for p in crossveins
            if p not in demoted and not any(
                p is vein_map.get(n) for n in ("ACV", "PCV")
            )
        ]
        if remaining_cv:
            # Only reassign the slot(s) that were demoted — keep valid ones
            lost_slots = [n for n in ("ACV", "PCV") if n not in vein_map]
            if lost_slots and len(remaining_cv) >= 1:
                # Sort remaining by Y to assign: lower Y → ACV, higher Y → PCV
                remaining_cv.sort(key=lambda p: p.y_median_norm)
                for slot in lost_slots:
                    if not remaining_cv:
                        break
                    if slot == "ACV":
                        vein_map["ACV"] = remaining_cv.pop(0)
                    elif slot == "PCV":
                        vein_map["PCV"] = remaining_cv.pop(-1)

    return demoted


def _validate_junction_orientations(
    vein_map: dict[str, MergedPath],
    junctions: list[JunctionPoint],
    centerlines: dict[tuple[int, int], LineString],
    merge_decisions: dict[tuple[float, float], dict],
    wing_bbox: tuple[float, float, float, float],
    midline=None,
    snap_radius: float = 40.0,
) -> dict[str, MergedPath]:
    """Validate that triple junctions have correct vein topology after classification.

    At crossvein junctions (ACV/PCV), we expect exactly 2 longitudinals + 1 crossvein.
    If the topology is wrong and an alternative merge was recorded, retry with that
    alternative and re-classify.
    """
    if not merge_decisions or not junctions:
        return vein_map

    # Build reverse map: segment_key → vein name
    def _build_seg_to_vein(vm: dict[str, MergedPath]) -> dict[tuple[int, int], str]:
        result: dict[tuple[int, int], str] = {}
        for name, mp in vm.items():
            for sk in mp.segment_keys:
                result[sk] = name
        return result

    seg_to_vein = _build_seg_to_vein(vein_map)

    # Check each junction that had a merge decision
    for junc_coord, decision in merge_decisions.items():
        arrivals = decision.get("arrivals", [])
        if len(arrivals) < 3:
            continue

        # Find which veins meet at this junction
        vein_names_at_junction: set[str] = set()
        for seg_key, _ep_idx in arrivals:
            name = seg_to_vein.get(seg_key)
            if name:
                vein_names_at_junction.add(name)

        if len(vein_names_at_junction) < 2:
            continue

        # Classify arrivals as longitudinal or crossvein
        longitudinals_here = {n for n in vein_names_at_junction if n.startswith("L")}
        crossveins_here = {n for n in vein_names_at_junction if n in ("ACV", "PCV")}

        # Expected: 2 longitudinals + 1 crossvein at CV junctions
        # Check ACV junction: should have L3 + L4 + ACV
        # Check PCV junction: should have L4 + L5 + PCV
        is_cv_junction = len(crossveins_here) > 0

        if not is_cv_junction:
            continue  # non-crossvein junctions don't need validation

        # Topology is correct if we have exactly 1 crossvein + 2 longitudinals
        if len(crossveins_here) == 1 and len(longitudinals_here) >= 2:
            continue  # correct topology

        # Wrong topology — try the alternative merge
        alternative = decision.get("alternative")
        chosen = decision.get("chosen")
        if alternative is None or chosen is None:
            continue
        if alternative[0] is None or alternative[1] is None:
            continue

        logger.info(
            "Junction (%.0f, %.0f): wrong topology (veins: %s) — "
            "trying alternative merge %s → %s",
            junc_coord[0], junc_coord[1],
            vein_names_at_junction, chosen, alternative,
        )

        # Re-merge: undo the chosen merge and apply the alternative
        # Build new groups from scratch using all merge decisions except
        # this junction's, then apply the alternative for this junction
        # This is expensive but junctions are few (typically 2-4)
        parent: dict[tuple[int, int], tuple[int, int]] = {
            k: k for k in centerlines
        }

        def find(k: tuple[int, int]) -> tuple[int, int]:
            while parent.get(k, k) != k:
                parent[k] = parent.get(parent[k], parent[k])
                k = parent[k]
            return k

        def union(a: tuple[int, int], b: tuple[int, int]) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Replay all merge decisions, substituting this junction's alternative
        for jc, dec in merge_decisions.items():
            ch = dec.get("chosen")
            if ch is None or ch[0] is None or ch[1] is None:
                continue
            if jc == junc_coord:
                # Apply alternative instead
                union(alternative[0], alternative[1])
            else:
                union(ch[0], ch[1])

        # Rebuild paths from the new groups
        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for key in centerlines:
            root = find(key)
            if root not in groups:
                groups[root] = []
            groups[root].append(key)

        new_paths: list[MergedPath] = []
        for root, seg_keys in groups.items():
            lines = [centerlines[k] for k in seg_keys]
            merged = _merge_vein_lines(lines)
            if merged is None or merged.length < 10:
                continue
            new_paths.append(MergedPath(
                segment_keys=seg_keys,
                line=merged,
                length_px=merged.length,
            ))

        # Re-split and re-classify
        new_paths = _split_on_sharp_turns(new_paths, centerlines, angle_threshold_deg=70.0)
        new_vein_map = classify_merged_paths(
            new_paths, wing_bbox, midline=midline, junctions=junctions,
        )

        # Check if the new topology is better at this junction
        new_seg_to_vein = _build_seg_to_vein(new_vein_map)
        new_veins: set[str] = set()
        for seg_key, _ep_idx in arrivals:
            name = new_seg_to_vein.get(seg_key)
            if name:
                new_veins.add(name)

        new_longs = {n for n in new_veins if n.startswith("L")}
        new_cvs = {n for n in new_veins if n in ("ACV", "PCV")}

        if len(new_cvs) == 1 and len(new_longs) >= 2:
            logger.info(
                "  Alternative merge fixes topology: %s → accepting",
                new_veins,
            )
            vein_map = new_vein_map
            seg_to_vein = new_seg_to_vein
        else:
            logger.info(
                "  Alternative merge didn't improve topology: %s → keeping original",
                new_veins,
            )

    return vein_map


def _path_y_at_x(path: MergedPath, target_x: float) -> float:
    """Return the path's Y value at *target_x* (or nearest point)."""
    coords = list(path.line.coords)
    best_y = coords[0][1]
    best_dx = abs(coords[0][0] - target_x)
    for x, y in coords:
        dx = abs(x - target_x)
        if dx < best_dx:
            best_dx = dx
            best_y = y
    return best_y


def _mean_signed_distance_from_midline(
    path: MergedPath,
    midline,
) -> float:
    """Compute the mean signed distance of a path from the midline.

    Samples the path at multiple X positions and compares each Y to the
    midline Y at that X.  Returns mean signed distance: negative = anterior
    (above midline), positive = posterior (below midline).
    """
    mid_coords = list(midline.line.coords)
    mid_xs = np.array([c[0] for c in mid_coords])
    mid_ys = np.array([c[1] for c in mid_coords])

    path_coords = list(path.line.coords)
    if len(path_coords) < 2:
        center_x = (mid_xs[0] + mid_xs[-1]) / 2.0
        return _path_y_at_x(path, center_x) - float(np.interp(center_x, mid_xs, mid_ys))

    dists: list[float] = []
    for px, py in path_coords:
        if mid_xs[0] <= px <= mid_xs[-1]:
            midline_y = float(np.interp(px, mid_xs, mid_ys))
            dists.append(py - midline_y)

    if not dists:
        center_x = (mid_xs[0] + mid_xs[-1]) / 2.0
        return _path_y_at_x(path, center_x) - float(np.interp(center_x, mid_xs, mid_ys))

    return float(np.mean(dists))


def _anchor_l3_l4_from_midline(
    longitudinals: list[MergedPath],
    midline,
) -> tuple[Optional[MergedPath], Optional[MergedPath]]:
    """Anchor L3 and L4 from the midline before crossvein assignment.

    L3 is the nearest vein anterior to (above) the midline.
    L4 is the nearest vein posterior to (below) the midline.
    Uses mean signed distance along the full path extent (not a single
    point) for robust anterior/posterior classification.
    Only considers paths >=30% of wing span.
    Returns (l3_path, l4_path); either may be None if no candidates exist.
    """
    if len(longitudinals) < 2:
        return None, None

    mid_coords = list(midline.line.coords)
    mid_xs = [c[0] for c in mid_coords]
    wing_span = mid_xs[-1] - mid_xs[0]
    center_x = (mid_xs[0] + mid_xs[-1]) / 2.0

    # L3/L4 are the longest veins in the wing — require >=30% of wing span
    min_length = wing_span * 0.30

    # Measure each path's mean signed distance from the midline
    anterior: list[tuple[float, MergedPath]] = []
    posterior: list[tuple[float, MergedPath]] = []
    for p in longitudinals:
        if p.length_px < min_length:
            continue
        mean_dist = _mean_signed_distance_from_midline(p, midline)
        if mean_dist < 0:
            anterior.append((mean_dist, p))
        else:
            posterior.append((mean_dist, p))

    l3 = None
    l4 = None

    if anterior:
        # L3 = nearest anterior (least negative mean distance)
        anterior.sort(key=lambda t: t[0])
        l3 = anterior[-1][1]

    if posterior:
        # L4 = nearest posterior (least positive mean distance)
        posterior.sort(key=lambda t: t[0])
        l4 = posterior[0][1]

    if l3:
        d = _mean_signed_distance_from_midline(l3, midline)
        logger.info(
            "Midline anchor: L3 → mean_dist=%.0fpx above midline, len=%.0fpx",
            -d, l3.length_px,
        )
    if l4:
        d = _mean_signed_distance_from_midline(l4, midline)
        logger.info(
            "Midline anchor: L4 → mean_dist=%.0fpx below midline, len=%.0fpx",
            d, l4.length_px,
        )

    return l3, l4


def _assign_remaining_longitudinals(
    longitudinals: list[MergedPath],
    midline,
    vein_map: dict[str, MergedPath],
) -> None:
    """Assign L1, L2, L5 from remaining longitudinals after L3/L4 are anchored.

    L2 = longest anterior vein remaining above L3.
    L1 = most anterior short vein above L2.
    L5 = longest posterior vein remaining below L4.
    """
    from WingVeinAnalyzer.models.vein_map import VEIN_LENGTH_PRIORS

    # Skip paths already assigned (L3, L4)
    assigned = set(id(v) for v in vein_map.values())
    remaining = [p for p in longitudinals if id(p) not in assigned]
    if not remaining:
        return

    mid_coords = list(midline.line.coords)
    mid_xs = [c[0] for c in mid_coords]
    mid_ys = [c[1] for c in mid_coords]
    center_x = (mid_xs[0] + mid_xs[-1]) / 2.0
    center_y = float(np.interp(center_x, mid_xs, mid_ys))
    wing_span = mid_xs[-1] - mid_xs[0]

    # Get L3/L4 mean signed distances as reference
    l3 = vein_map.get("L3")
    l4 = vein_map.get("L4")
    l3_dist = _mean_signed_distance_from_midline(l3, midline) if l3 else 0.0
    l4_dist = _mean_signed_distance_from_midline(l4, midline) if l4 else 0.0

    # Split remaining into anterior (above L3) and posterior (below L4)
    anterior: list[MergedPath] = []
    posterior: list[MergedPath] = []
    for p in remaining:
        d = _mean_signed_distance_from_midline(p, midline)
        if d < l3_dist:
            anterior.append(p)
        elif d > l4_dist:
            posterior.append(p)
        # Paths between L3 and L4 are discarded (likely ACV/PCV artifacts)

    # Anterior: assign L2 (longest) then L1 (most anterior short vein)
    if anterior:
        anterior.sort(key=lambda p: p.length_px, reverse=True)
        l2_min = VEIN_LENGTH_PRIORS["L2"][0] * wing_span
        l1_max = VEIN_LENGTH_PRIORS["L1"][1] * wing_span
        best = anterior[0]
        if best.length_px >= l2_min:
            vein_map["L2"] = best
            rest = anterior[1:]
            if rest:
                candidate = min(rest, key=lambda p: _path_y_at_x(p, center_x))
                if candidate.length_px <= l1_max:
                    vein_map["L1"] = candidate
        elif best.length_px <= l1_max:
            vein_map["L1"] = min(anterior, key=lambda p: _path_y_at_x(p, center_x))
        else:
            anterior.sort(key=lambda p: _path_y_at_x(p, center_x))
            for i, p in enumerate(anterior[:2]):
                vein_map[["L1", "L2"][i]] = p

    # Posterior: longest → L5
    if posterior:
        posterior.sort(key=lambda p: p.length_px, reverse=True)
        vein_map["L5"] = posterior[0]

    for name in ["L1", "L2", "L5"]:
        if name in vein_map and id(vein_map[name]) not in assigned:
            p = vein_map[name]
            logger.debug(
                "Remaining longitudinal: %s → Y_at_center=%.0f, len=%.0fpx",
                name, _path_y_at_x(p, center_x), p.length_px,
            )


def _assign_longitudinals_from_midline(
    longitudinals: list[MergedPath],
    midline,
) -> dict[str, MergedPath]:
    """Assign longitudinal veins anchored by the wing midline.

    L3 is the nearest vein anterior to (above) the midline center.
    L4 is the nearest vein posterior to (below) the midline center.
    Remaining anterior veins are assigned using length priors: L2 is
    always the longest vein above L3, L1 is the most anterior short vein.
    Remaining posterior veins → L5.
    """
    from WingVeinAnalyzer.models.vein_map import VEIN_LENGTH_PRIORS

    result: dict[str, MergedPath] = {}

    # Anchor L3/L4 first
    l3, l4 = _anchor_l3_l4_from_midline(longitudinals, midline)
    if l3:
        result["L3"] = l3
    if l4:
        result["L4"] = l4

    # Assign remaining
    _assign_remaining_longitudinals(longitudinals, midline, result)

    return result


def _assign_longitudinals_scored(
    longitudinals: list[MergedPath],
    wing_span_px: float,
    acv_path: Optional[MergedPath] = None,
    pcv_path: Optional[MergedPath] = None,
    midline=None,
) -> dict[str, MergedPath]:
    """Assign longitudinal veins using midline anchoring or combinatorial fallback.

    When a midline is available, L3/L4 are anchored directly from the
    midline center and the rest follow by anterior-posterior order.
    Falls back to exhaustive combinatorial scoring when no midline.
    """
    from itertools import combinations, permutations

    long_names = ["L1", "L2", "L3", "L4", "L5"]
    result: dict[str, MergedPath] = {}

    if not longitudinals:
        return result

    # Midline-anchored assignment (primary path)
    if midline is not None and len(longitudinals) >= 2:
        return _assign_longitudinals_from_midline(longitudinals, midline)

    # Sort by Y centroid for ordering constraint
    sorted_paths = sorted(longitudinals, key=lambda p: p.y_centroid_norm)
    n = len(sorted_paths)

    # Precompute scores for all (path, name) pairs
    score_matrix: dict[tuple[int, str], float] = {}
    for pi, path in enumerate(sorted_paths):
        for name in long_names:
            score_matrix[(pi, name)] = _longitudinal_match_score(
                path, name, wing_span_px, acv_path, pcv_path,
                midline=midline,
            )

    # Try all k from min(n,5) down to max(1, min(n,5)-2)
    # Use normalized score: total / k, with a small bonus per extra vein
    best_norm_score = -1.0
    best_assignment: list[tuple[int, str]] = []

    max_k = min(n, 5)
    min_k = max(1, max_k - 2)
    min_per_vein = 0.25  # reject assignments with any vein below this

    for k in range(max_k, min_k - 1, -1):
        for path_indices in combinations(range(n), k):
            for name_subset in combinations(range(5), k):
                # Try all permutations of name assignment (not just Y-ordered)
                # Adjacent veins (e.g. L1/L2) can have overlapping Y centroids
                for name_perm in permutations(name_subset):
                    scores = [
                        score_matrix[(path_indices[i], long_names[name_perm[i]])]
                        for i in range(k)
                    ]
                    # Skip if any single assignment is terrible
                    if min(scores) < min_per_vein:
                        continue
                    total = sum(scores)
                    # Normalize: average score + bonus for more veins
                    norm = total / k + 0.08 * k
                    if norm > best_norm_score:
                        best_norm_score = norm
                        best_assignment = [
                            (path_indices[i], long_names[name_perm[i]])
                            for i in range(k)
                        ]

    used_paths: set[int] = set()
    for pi, name in best_assignment:
        result[name] = sorted_paths[pi]
        used_paths.add(pi)
        logger.debug(
            "Longitudinal: %s → Y=%.3f, len=%.0fpx, score=%.2f",
            name, sorted_paths[pi].y_centroid_norm,
            sorted_paths[pi].length_px,
            score_matrix[(pi, name)],
        )

    # Log unassigned paths
    for i in range(n):
        if i not in used_paths:
            logger.info(
                "Unassigned longitudinal: len=%.0fpx, Y=%.3f (boundary artifact)",
                sorted_paths[i].length_px, sorted_paths[i].y_centroid_norm,
            )

    # Post-processing: swap L4/L5 if crossvein proximity strongly favors it
    # (Y centroids can overlap when paths have different extents)
    _swap_l4_l5_if_needed(result, acv_path, pcv_path)

    return result


def _swap_l4_l5_if_needed(
    result: dict[str, MergedPath],
    acv_path: Optional[MergedPath],
    pcv_path: Optional[MergedPath],
) -> None:
    """Swap L4/L5 assignment if ACV proximity strongly favors it.

    ACV connects L3–L4, so L4 should be close to ACV and L5 far from it.
    PCV connects L4–L5, so both are always near PCV — not useful for swapping.
    Only swaps when the distance difference exceeds 50px to avoid false
    positives from tiny jitter at shared junction points.
    """
    if "L4" not in result or "L5" not in result:
        return
    if acv_path is None:
        return

    l4 = result["L4"]
    l5 = result["L5"]

    acv_to_l4 = l4.line.distance(acv_path.line)
    acv_to_l5 = l5.line.distance(acv_path.line)

    # L4 should be near ACV; L5 should be far from ACV.
    # Swap only when: (1) L5 is notably closer, (2) the separation is meaningful,
    # and (3) the closer vein is actually adjacent to ACV (< 50px).
    # Without check (3), we'd swap when both veins are far from ACV (meaningless).
    if (acv_to_l5 < acv_to_l4
            and (acv_to_l4 - acv_to_l5) > 50
            and min(acv_to_l5, acv_to_l4) < 50):
        logger.info(
            "Swapping L4/L5: ACV closer to L5 (%.0fpx) than L4 (%.0fpx)",
            acv_to_l5, acv_to_l4,
        )
        result["L4"], result["L5"] = result["L5"], result["L4"]


def _longitudinal_match_score(
    path: MergedPath,
    vein_name: str,
    wing_span_px: float,
    acv_path: Optional[MergedPath] = None,
    pcv_path: Optional[MergedPath] = None,
    midline=None,
) -> float:
    """Score how well a path matches a specific longitudinal vein identity."""
    # Y-position score: blend of median-Y closeness (60%) and Y-range overlap (40%)
    y_lo, y_hi = SPATIAL_PRIORS_Y[vein_name]
    y_mid = (y_lo + y_hi) / 2
    y_range = y_hi - y_lo

    # Median Y score (more robust than centroid for curved veins)
    y_dist = abs(path.y_median_norm - y_mid)
    if y_lo <= path.y_median_norm <= y_hi:
        median_y_score = 1.0 - (y_dist / y_range)
    else:
        median_y_score = max(0.0, 0.5 - y_dist)

    # Y-range overlap: Jaccard of path's [y_min, y_max] vs prior's [y_lo, y_hi]
    overlap_lo = max(path.y_min_norm, y_lo)
    overlap_hi = min(path.y_max_norm, y_hi)
    overlap = max(0.0, overlap_hi - overlap_lo)
    union_lo = min(path.y_min_norm, y_lo)
    union_hi = max(path.y_max_norm, y_hi)
    union_span = union_hi - union_lo
    y_overlap_score = overlap / union_span if union_span > 0 else 0.0

    y_score = 0.6 * median_y_score + 0.4 * y_overlap_score

    # Length score: closeness to expected length range
    length_frac = path.length_px / wing_span_px if wing_span_px > 0 else 0.0
    len_lo, len_hi = VEIN_LENGTH_PRIORS[vein_name]
    len_mid = (len_lo + len_hi) / 2
    len_range = len_hi - len_lo
    if len_lo <= length_frac <= len_hi:
        len_score = 1.0 - abs(length_frac - len_mid) / len_range
    else:
        # Out of range — strong penalty
        overshoot = max(len_lo - length_frac, length_frac - len_hi, 0)
        len_score = max(0.0, 0.3 - overshoot * 2)

    # Crossvein proximity score
    cv_score = _compute_crossvein_score(path, vein_name, acv_path, pcv_path)

    # Midline distance score
    mid_score = _compute_midline_score(path, vein_name, midline) if midline is not None else None

    # Combined score — weights depend on available information
    has_cv = acv_path is not None or pcv_path is not None
    has_mid = mid_score is not None

    if has_mid and has_cv:
        return 0.30 * y_score + 0.25 * len_score + 0.20 * cv_score + 0.25 * mid_score
    elif has_mid:
        return 0.30 * y_score + 0.25 * len_score + 0.45 * mid_score
    elif has_cv:
        return 0.40 * y_score + 0.30 * len_score + 0.30 * cv_score
    else:
        return 0.6 * y_score + 0.4 * len_score


def _compute_midline_score(
    path: MergedPath,
    vein_name: str,
    midline,
) -> float:
    """Score how well a path's signed distance from the midline matches priors."""
    if vein_name not in MIDLINE_PRIORS:
        return 0.5  # neutral for unknown veins

    mid_lo, mid_hi = MIDLINE_PRIORS[vein_name]
    mid_center = (mid_lo + mid_hi) / 2
    mid_range = mid_hi - mid_lo

    # Sample points along the path
    line = path.line
    n_samples = max(5, min(20, int(line.length / 50)))
    midline_xs = np.array([c[0] for c in midline.line.coords])
    midline_ys = np.array([c[1] for c in midline.line.coords])
    half_heights = midline.half_heights

    signed_dists: list[float] = []
    for i in range(n_samples):
        frac = i / max(n_samples - 1, 1)
        pt = line.interpolate(frac, normalized=True)
        x, y = pt.x, pt.y

        # Interpolate midline Y and half-height at this X
        if x < midline_xs[0] or x > midline_xs[-1]:
            continue
        mid_y = float(np.interp(x, midline_xs, midline_ys))
        hh = float(np.interp(x, midline_xs, half_heights))
        if hh < 5.0:
            continue

        signed_dist = (y - mid_y) / hh
        signed_dists.append(signed_dist)

    if not signed_dists:
        return 0.5  # neutral

    median_dist = float(np.median(signed_dists))

    # Score using same pattern as Y-position scoring
    dist_from_center = abs(median_dist - mid_center)
    if mid_lo <= median_dist <= mid_hi:
        score = 1.0 - (dist_from_center / mid_range) if mid_range > 0 else 1.0
    else:
        # Outside range — decaying penalty
        overshoot = max(mid_lo - median_dist, median_dist - mid_hi, 0)
        score = max(0.0, 0.5 - overshoot)

    return score


def _compute_crossvein_score(
    path: MergedPath,
    vein_name: str,
    acv_path: Optional[MergedPath],
    pcv_path: Optional[MergedPath],
) -> float:
    """Score how well a longitudinal path relates to known crossvein positions.

    L3: should be near ACV (anterior side)
    L4: should be near both ACV and PCV (between them)
    L5: should be posterior to PCV
    L1, L2: neutral (far from crossveins, no strong signal)
    """
    if vein_name in ("L1", "L2"):
        return 0.5  # neutral — crossveins don't help distinguish these

    # Compute distances from the path to crossveins
    acv_dist = path.line.distance(acv_path.line) if acv_path else None
    pcv_dist = path.line.distance(pcv_path.line) if pcv_path else None

    # Normalize distances by a reference scale (~200px = typical crossvein gap)
    ref_dist = 200.0

    if vein_name == "L3":
        # L3 should be near ACV (it connects to ACV)
        if acv_dist is not None:
            proximity = max(0.0, 1.0 - acv_dist / ref_dist)
            return proximity
        return 0.5

    elif vein_name == "L4":
        # L4 should be near PCV (it connects to PCV) and near ACV
        score = 0.5
        if pcv_dist is not None:
            score = max(0.0, 1.0 - pcv_dist / ref_dist) * 0.6
        if acv_dist is not None:
            score += max(0.0, 1.0 - acv_dist / ref_dist) * 0.4
        return score

    elif vein_name == "L5":
        # L5 should be near PCV (it connects to PCV)
        # Note: L5 centroid can be anterior to PCV because L5 is a long
        # horizontal vein — use proximity only, not centroid comparison
        if pcv_dist is not None:
            proximity = max(0.0, 1.0 - pcv_dist / ref_dist)
            return proximity
        return 0.5

    return 0.5


# ---------------------------------------------------------------------------
# 3d. Vein Shape Validation
# ---------------------------------------------------------------------------

def validate_vein_shapes(
    vein_map: dict[str, MergedPath],
) -> dict[str, list[str]]:
    """Validate vein shapes and return per-vein warnings."""
    warnings: dict[str, list[str]] = {}

    for vein_id, path in vein_map.items():
        vein_warnings: list[str] = []
        is_crossvein = vein_id in ("ACV", "PCV")

        # Straightness check
        threshold = 0.5 if is_crossvein else STRAIGHTNESS_THRESHOLD
        if path.straightness < threshold:
            vein_warnings.append(
                f"Low straightness: {path.straightness:.2f} (threshold {threshold:.2f})"
            )

        # Angular continuity: walk in ~50px steps
        coords = list(path.line.coords)
        step_dist = 50.0
        n_steps = max(2, int(path.length_px / step_dist))
        sample_pts = [
            path.line.interpolate(i / n_steps, normalized=True)
            for i in range(n_steps + 1)
        ]

        for i in range(1, len(sample_pts) - 1):
            dx1 = sample_pts[i].x - sample_pts[i - 1].x
            dy1 = sample_pts[i].y - sample_pts[i - 1].y
            dx2 = sample_pts[i + 1].x - sample_pts[i].x
            dy2 = sample_pts[i + 1].y - sample_pts[i].y

            v1 = np.array([dx1, dy1])
            v2 = np.array([dx2, dy2])
            angle_change = _angle_between_vectors(v1, v2)

            if angle_change > MAX_ANGLE_CHANGE_DEG:
                vein_warnings.append(
                    f"Abrupt direction change: {angle_change:.0f}° at step {i}"
                )

        # Monotonicity check for longitudinals (direction-agnostic)
        if not is_crossvein and len(sample_pts) > 2:
            increasing = 0
            decreasing = 0
            for i in range(1, len(sample_pts)):
                if sample_pts[i].x > sample_pts[i - 1].x:
                    increasing += 1
                elif sample_pts[i].x < sample_pts[i - 1].x:
                    decreasing += 1
            # The dominant direction determines "forward"; minority is "backwards"
            total_moves = increasing + decreasing
            if total_moves > 0:
                minority = min(increasing, decreasing)
                frac_backwards = minority / total_moves
                if frac_backwards > 0.20:
                    vein_warnings.append(
                        f"Non-monotonic X: {frac_backwards:.0%} of steps go backwards"
                    )

        if vein_warnings:
            warnings[vein_id] = vein_warnings

    return warnings


# ---------------------------------------------------------------------------
# 3e. Region Identification from Veins
# ---------------------------------------------------------------------------

def name_regions_from_veins(
    polygons: list[Polygon],
    vein_map: dict[str, MergedPath],
    wing_bbox: tuple[float, float, float, float],
) -> dict[int, str]:
    """Name intervein polygons based on which veins bound them.

    Uses the segment_keys from each MergedPath to directly determine
    which polygon indices each named vein borders — no proximity sampling.
    """
    min_x, min_y, max_x, max_y = wing_bbox
    bbox_h = max_y - min_y

    # Build polygon → set of bounding veins from segment keys
    poly_veins: dict[int, set[str]] = {i: set() for i in range(len(polygons))}
    for vein_id, mp in vein_map.items():
        for seg_key in mp.segment_keys:
            idx_a, idx_b = seg_key
            if 0 <= idx_a < len(polygons):
                poly_veins[idx_a].add(vein_id)
            if 0 <= idx_b < len(polygons):
                poly_veins[idx_b].add(vein_id)

    poly_names: dict[int, str] = {}
    total_area = sum(p.area for p in polygons)

    for idx, poly in enumerate(polygons):
        bounding = poly_veins[idx]
        cx = poly.centroid.x
        cy = poly.centroid.y
        y_norm = (cy - min_y) / bbox_h if bbox_h > 0 else 0.5

        name = _region_from_bounding_veins(
            bounding, poly, vein_map, wing_bbox, bbox_h, total_area,
        )
        if name:
            poly_names[idx] = name
            logger.debug("P%d → %s (veins: %s)", idx, name, bounding)
        else:
            logger.warning(
                "P%d: could not identify region (veins: %s)", idx, bounding,
            )

    # Resolve conflicts: if two polygons got the same name, use area/position
    _resolve_name_conflicts(poly_names, polygons, vein_map, wing_bbox)

    # Validate and correct region positions
    _validate_and_correct_region_positions(poly_names, polygons, wing_bbox)

    return poly_names


def _region_from_bounding_veins(
    bounding_veins: set[str],
    poly: Polygon,
    vein_map: dict[str, MergedPath],
    wing_bbox: tuple[float, float, float, float],
    bbox_h: float,
    total_area: float = 0.0,
) -> Optional[str]:
    """Determine region name from its bounding veins, position, and area."""
    cx = poly.centroid.x
    cy = poly.centroid.y
    min_x, min_y, max_x, max_y = wing_bbox
    y_norm = (cy - min_y) / bbox_h if bbox_h > 0 else 0.5

    # Area fraction for area-prior scoring
    area_frac = poly.area / total_area if total_area > 0 else 0.0

    # Special case: costal_cell is anterior to L1, bounded only by L1
    if bounding_veins == {"L1"}:
        l1_y = vein_map["L1"].y_centroid_norm if "L1" in vein_map else 0.1
        if y_norm < l1_y:
            return "costal_cell"

    # Check each region's expected veins — best match wins
    # Score = Jaccard overlap × area-prior multiplier
    best_name = None
    best_score = -1.0

    for region_name, expected_veins in REGION_EXPECTED_VEINS.items():
        if region_name == "costal_cell":
            continue  # handled above
        matched = bounding_veins & expected_veins
        if not matched:
            continue
        # Score: Jaccard-like — reward overlap, penalize extra/missing
        jaccard = len(matched) / len(expected_veins | bounding_veins)
        # Bonus only for exact match (bounding veins == expected veins)
        if bounding_veins == expected_veins:
            jaccard += 1.0

        # Area-prior multiplier: penalize extreme area mismatches
        # Within range → 1.0; outside → decays toward 0.1
        area_mult = 1.0
        if total_area > 0 and region_name in REGION_AREA_PRIORS:
            lo, hi = REGION_AREA_PRIORS[region_name]
            if lo <= area_frac <= hi:
                area_mult = 1.0
            else:
                distance = max(lo - area_frac, area_frac - hi, 0)
                area_mult = max(0.1, 1.0 - distance * 4.0)

        score = jaccard * area_mult

        if score > best_score:
            best_score = score
            best_name = region_name

    if best_name is None:
        return None

    # Disambiguate regions that share the same expected veins using area
    # Area-based disambiguation is orientation-independent, unlike X-position
    if best_name in ("1st_basal_cell", "1st_posterior_cell"):
        # 1st_basal_cell is always much smaller than 1st_posterior_cell
        if total_area > 0:
            lo_basal, hi_basal = REGION_AREA_PRIORS["1st_basal_cell"]
            lo_post, hi_post = REGION_AREA_PRIORS["1st_posterior_cell"]
            mid_basal = (lo_basal + hi_basal) / 2
            mid_post = (lo_post + hi_post) / 2
            dist_to_basal = abs(area_frac - mid_basal)
            dist_to_post = abs(area_frac - mid_post)
            best_name = "1st_basal_cell" if dist_to_basal < dist_to_post else "1st_posterior_cell"
        else:
            # Fallback: smaller polygon is basal
            best_name = "1st_basal_cell"

    elif best_name in ("discal_cell", "2nd_posterior_cell"):
        # discal_cell is always smaller than 2nd_posterior_cell
        if total_area > 0:
            lo_disc, hi_disc = REGION_AREA_PRIORS["discal_cell"]
            lo_post, hi_post = REGION_AREA_PRIORS["2nd_posterior_cell"]
            mid_disc = (lo_disc + hi_disc) / 2
            mid_post = (lo_post + hi_post) / 2
            dist_to_disc = abs(area_frac - mid_disc)
            dist_to_post = abs(area_frac - mid_post)
            best_name = "discal_cell" if dist_to_disc < dist_to_post else "2nd_posterior_cell"
        else:
            # Fallback: smaller polygon is discal
            best_name = "discal_cell"

    return best_name


def _resolve_name_conflicts(
    poly_names: dict[int, str],
    polygons: list[Polygon],
    vein_map: dict[str, MergedPath],
    wing_bbox: tuple[float, float, float, float],
) -> None:
    """If two polygons got the same region name, resolve by area match and reassign."""
    min_x, min_y, max_x, max_y = wing_bbox
    bbox_h = max_y - min_y

    # Build polygon → bounding veins for reassignment attempts
    poly_veins: dict[int, set[str]] = {i: set() for i in range(len(polygons))}
    for vein_id, mp in vein_map.items():
        for seg_key in mp.segment_keys:
            idx_a, idx_b = seg_key
            if 0 <= idx_a < len(polygons):
                poly_veins[idx_a].add(vein_id)
            if 0 <= idx_b < len(polygons):
                poly_veins[idx_b].add(vein_id)

    # Iterate until no conflicts remain (max 3 rounds)
    for _ in range(3):
        name_to_indices: dict[str, list[int]] = {}
        for idx, name in poly_names.items():
            if name not in name_to_indices:
                name_to_indices[name] = []
            name_to_indices[name].append(idx)

        had_conflict = False
        for name, indices in name_to_indices.items():
            if len(indices) <= 1:
                continue

            had_conflict = True
            logger.warning("Name conflict: %s assigned to polygons %s", name, indices)

            # Keep the polygon with better area match
            total_area = sum(p.area for p in polygons)
            if name in REGION_AREA_PRIORS and total_area > 0:
                lo, hi = REGION_AREA_PRIORS[name]
                expected_mid = (lo + hi) / 2 * total_area
                indices.sort(key=lambda i: abs(polygons[i].area - expected_mid))

            # First index keeps the name; try to reassign others
            for idx in indices[1:]:
                del poly_names[idx]
                # Try second-best region name
                alt_name = _region_from_bounding_veins(
                    poly_veins[idx], polygons[idx], vein_map, wing_bbox, bbox_h,
                    total_area,
                )
                # Check the alt_name isn't already taken by exactly one polygon
                already_used = alt_name in poly_names.values() if alt_name else True
                if alt_name and alt_name != name and not already_used:
                    poly_names[idx] = alt_name
                    logger.info("Reassigned P%d from %s to %s", idx, name, alt_name)
                else:
                    # Try all regions in order of score
                    reassigned = False
                    for region_name in REGION_EXPECTED_VEINS:
                        if region_name == name:
                            continue
                        if region_name in poly_names.values():
                            continue
                        matched = poly_veins[idx] & REGION_EXPECTED_VEINS[region_name]
                        if matched:
                            poly_names[idx] = region_name
                            logger.info("Reassigned P%d to %s (fallback)", idx, region_name)
                            reassigned = True
                            break
                    if not reassigned:
                        logger.warning("Could not reassign P%d (dropped from %s)", idx, name)

        if not had_conflict:
            break


def _validate_and_correct_region_positions(
    poly_names: dict[int, str],
    polygons: list[Polygon],
    wing_bbox: tuple[float, float, float, float],
) -> None:
    """Validate and correct region positions based on canonical Y/X ordering.

    Checks that region centroids follow expected anterior-to-posterior ordering
    and that proximal/distal pairs have correct X relationships.  Swaps names
    when violations are found.
    """
    min_x, min_y, max_x, max_y = wing_bbox
    bbox_h = max_y - min_y
    bbox_w = max_x - min_x

    if bbox_h == 0 or bbox_w == 0:
        return

    # Build reverse lookup: name → polygon index
    name_to_idx: dict[str, int] = {}
    for idx, name in poly_names.items():
        name_to_idx[name] = idx

    def _get_centroid(name: str) -> Optional[tuple[float, float]]:
        idx = name_to_idx.get(name)
        if idx is None or idx >= len(polygons):
            return None
        c = polygons[idx].centroid
        y_norm = (c.y - min_y) / bbox_h
        x_norm = (c.x - min_x) / bbox_w
        return (y_norm, x_norm)

    # Y-ordering checks: each group should have lower Y than the next
    y_order_groups = [
        ("marginal_cell", "submarginal_cell"),
        ("submarginal_cell", "1st_basal_cell"),
        ("submarginal_cell", "1st_posterior_cell"),
        ("1st_posterior_cell", "2nd_posterior_cell"),
        ("2nd_posterior_cell", "3rd_posterior_cell"),
    ]

    # Only swap when the violation is significant (>0.10 normalized units)
    # to avoid overriding correct bounding-vein assignments due to centroid noise
    y_threshold = 0.10

    for name_a, name_b in y_order_groups:
        ca = _get_centroid(name_a)
        cb = _get_centroid(name_b)
        if ca is None or cb is None:
            continue
        violation = ca[0] - cb[0]
        if violation > y_threshold:  # name_a should be more anterior (lower Y)
            idx_a = name_to_idx[name_a]
            idx_b = name_to_idx[name_b]
            logger.warning(
                "Region Y-order violation: %s (Y=%.2f) > %s (Y=%.2f), swapping",
                name_a, ca[0], name_b, cb[0],
            )
            poly_names[idx_a] = name_b
            poly_names[idx_b] = name_a
            name_to_idx[name_a] = idx_b
            name_to_idx[name_b] = idx_a

    # X-ordering checks removed: proximal/distal disambiguation is now
    # handled by area-based logic in _region_from_bounding_veins(), which
    # is orientation-independent. X-position checks were breaking for wings
    # with hinge on the right (high X) side.


# ---------------------------------------------------------------------------
# 3e''. Merged Polygon Detection and Splitting
# ---------------------------------------------------------------------------

# Which vein separates each pair of adjacent regions
_REGION_SPLIT_VEINS: dict[tuple[str, str], str] = {
    ("2nd_posterior_cell", "3rd_posterior_cell"): "L5",
    ("3rd_posterior_cell", "2nd_posterior_cell"): "L5",
    ("discal_cell", "2nd_posterior_cell"): "PCV",
    ("2nd_posterior_cell", "discal_cell"): "PCV",
    ("1st_basal_cell", "1st_posterior_cell"): "ACV",
    ("1st_posterior_cell", "1st_basal_cell"): "ACV",
    ("discal_cell", "3rd_posterior_cell"): "L5",
    ("3rd_posterior_cell", "discal_cell"): "L5",
    ("1st_posterior_cell", "2nd_posterior_cell"): "L4",
    ("2nd_posterior_cell", "1st_posterior_cell"): "L4",
}


def split_merged_polygons(
    poly_names: dict[int, str],
    polygons: list[Polygon],
    vein_map: dict[str, MergedPath],
    wing_bbox: tuple[float, float, float, float],
    image_shape: tuple[int, int] = (0, 0),
) -> tuple[dict[int, str], list[Polygon], list[SplitInfo]]:
    """Detect and split input polygons that appear to cover two merged regions.

    The pixel classifier may merge adjacent regions into a single polygon
    when a vein is absent or poorly detected.  This function detects
    oversized polygons and splits them at their narrowest neck using
    erosion + distance-transform watershed.

    Returns updated (poly_names, polygons, split_infos).
    """
    total_area = sum(p.area for p in polygons)
    if total_area == 0:
        return poly_names, polygons, []

    # Find which expected regions have no assigned polygon
    assigned_regions = set(poly_names.values())
    all_regions = set(REGION_EXPECTED_VEINS.keys())
    missing_regions = all_regions - assigned_regions - {"costal_cell"}

    if not missing_regions:
        return poly_names, polygons, []

    # Check each named polygon for area overshoot
    polygons = list(polygons)  # make mutable copy
    splits_made = 0
    split_infos: list[SplitInfo] = []

    for idx in list(poly_names.keys()):
        name = poly_names[idx]
        if idx >= len(polygons):
            continue
        poly = polygons[idx]
        area_frac = poly.area / total_area

        if name not in REGION_AREA_PRIORS:
            continue
        lo, hi = REGION_AREA_PRIORS[name]

        # Multi-signal merge detection
        area_over_max = area_frac > hi
        if not area_over_max:
            continue

        # Strong area signal alone is sufficient
        area_strongly_over = area_frac > hi * 1.3

        # Moderate area + shape-based confirmation
        solidity = poly.area / poly.convex_hull.area
        has_poor_shape = solidity < 0.85
        has_bottleneck = _detect_bottleneck(poly)

        if not (area_strongly_over or has_poor_shape or has_bottleneck):
            continue

        logger.info(
            "P%d (%s): area=%.1f%% exceeds expected max %.1f%% — "
            "possible merged region",
            idx, name, area_frac * 100, hi * 100,
        )

        # Find the best missing region to split off
        best_missing = None
        for missing in missing_regions:
            split_key = (name, missing)
            if split_key in _REGION_SPLIT_VEINS:
                best_missing = missing
                break

        if best_missing is None:
            logger.info(
                "  No adjacent missing region found for %s → %s",
                name, missing_regions,
            )
            continue

        # Use erosion-watershed to split at the polygon's natural neck
        pieces = _split_polygon_by_erosion(poly, image_shape)
        if pieces is None:
            logger.info(
                "  Could not split P%d by erosion",
                idx,
            )
            continue

        piece_a, piece_b = pieces

        # Assign names: use area priors to decide which piece is which
        lo_orig, hi_orig = REGION_AREA_PRIORS[name]
        lo_miss, hi_miss = REGION_AREA_PRIORS[best_missing]
        mid_orig = (lo_orig + hi_orig) / 2 * total_area
        mid_miss = (lo_miss + hi_miss) / 2 * total_area

        # Figure out which piece better fits which name
        dist_a_orig = abs(piece_a.area - mid_orig)
        dist_a_miss = abs(piece_a.area - mid_miss)

        if dist_a_orig <= dist_a_miss:
            orig_piece, new_piece = piece_a, piece_b
        else:
            orig_piece, new_piece = piece_b, piece_a

        # Replace the original polygon with the correctly-sized piece
        polygons[idx] = orig_piece
        new_idx = len(polygons)
        polygons.append(new_piece)
        poly_names[new_idx] = best_missing
        missing_regions.discard(best_missing)
        splits_made += 1

        # Record split metadata for post-split centerline extraction
        sep_vein = _REGION_SPLIT_VEINS.get((name, best_missing), "")
        split_infos.append(SplitInfo(
            orig_idx=idx,
            new_idx=new_idx,
            orig_name=name,
            new_name=best_missing,
            separating_vein=sep_vein,
        ))

        logger.info(
            "  Split P%d by erosion: %s (%.0f px²) + P%d:%s (%.0f px²), "
            "separating vein: %s",
            idx, name, orig_piece.area,
            new_idx, best_missing, new_piece.area, sep_vein,
        )

    if splits_made:
        logger.info("Split %d merged polygon(s)", splits_made)

    return poly_names, polygons, split_infos


def _detect_bottleneck(poly: Polygon, min_part_frac: float = 0.15) -> bool:
    """Check if polygon has a thin neck that splits on erosion."""
    for amount in [10, 15, 20, 30]:
        eroded = poly.buffer(-amount)
        if eroded.geom_type == 'MultiPolygon':
            parts = [g for g in eroded.geoms if g.area > poly.area * min_part_frac]
            if len(parts) >= 2:
                return True
    return False


def _fill_polygon_np(
    mask: np.ndarray,
    poly: Polygon,
    value: int,
) -> None:
    """Fill a shapely Polygon into a numpy array using cv2.fillPoly."""
    coords = np.array(poly.exterior.coords, dtype=np.int32)
    cv2.fillPoly(mask, [coords], int(value))


def _split_polygon_by_erosion(
    poly: Polygon,
    image_shape: tuple[int, int],
    min_part_frac: float = 0.15,
) -> Optional[tuple[Polygon, Polygon]]:
    """Split a polygon at its narrowest neck using erosion + distance transform."""
    # 1. Find erosion amount that separates into 2+ large parts
    erode_amount = None
    parts: list[Polygon] = []
    for amount in [10, 15, 20, 30, 50]:
        eroded = poly.buffer(-amount)
        if eroded.geom_type == 'MultiPolygon':
            parts = [g for g in eroded.geoms if g.area > poly.area * min_part_frac]
            if len(parts) >= 2:
                erode_amount = amount
                break
    if erode_amount is None:
        return None

    # 2. Rasterize polygon and eroded seeds
    h, w = image_shape
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    _fill_polygon_np(poly_mask, poly, 1)

    seed_map = np.zeros((h, w), dtype=np.int32)
    parts_sorted = sorted(parts, key=lambda g: g.area, reverse=True)[:2]
    for i, part in enumerate(parts_sorted):
        _fill_polygon_np(seed_map, part, i + 1)
    seed_map[poly_mask == 0] = 0

    # 3. Distance transform to fill unseeded pixels
    background = (seed_map == 0) & (poly_mask > 0)
    if background.sum() == 0:
        return None
    _, nearest_idx = ndimage.distance_transform_edt(background, return_indices=True)
    filled = seed_map[nearest_idx[0], nearest_idx[1]]
    filled[poly_mask == 0] = 0
    filled[seed_map > 0] = seed_map[seed_map > 0]

    # 4. Convert back to Polygons via cv2.findContours
    piece_polys: list[Polygon] = []
    for lbl in [1, 2]:
        piece_mask = ((filled == lbl) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            coords = [(float(pt[0][0]), float(pt[0][1])) for pt in largest]
            if len(coords) >= 4:
                p = Polygon(coords)
                if p.is_valid and p.area > 100:
                    piece_polys.append(p)
    if len(piece_polys) < 2:
        return None
    piece_polys.sort(key=lambda p: p.area, reverse=True)
    return piece_polys[0], piece_polys[1]


# ---------------------------------------------------------------------------
# 3f. Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate(
    vein_map: dict[str, MergedPath],
    poly_names: dict[int, str],
    centerlines: dict[tuple[int, int], LineString],
    polygons: list[Polygon],
    wing_bbox: tuple[float, float, float, float],
) -> ValidationReport:
    """Cross-validate vein and region assignments for consistency."""
    report = ValidationReport()
    min_x, min_y, max_x, max_y = wing_bbox
    bbox_h = max_y - min_y

    # 1. Boundary consistency: for each centerline segment, check if the vein
    # it belongs to matches VEIN_BOUNDARIES for the named regions
    # Build reverse lookup: which vein does each segment belong to?
    seg_to_vein: dict[tuple[int, int], str] = {}
    for vein_id, mp in vein_map.items():
        for seg_key in mp.segment_keys:
            seg_to_vein[seg_key] = vein_id

    for (idx_a, idx_b), line in centerlines.items():
        name_a = poly_names.get(idx_a)
        name_b = poly_names.get(idx_b)
        if name_a is None or name_b is None:
            continue

        assigned_vein = seg_to_vein.get((idx_a, idx_b))
        if assigned_vein is None:
            continue

        # Check if this pair matches VEIN_BOUNDARIES for the assigned vein
        expected_pairs = VEIN_BOUNDARIES.get(assigned_vein, [])
        pair = (name_a, name_b)
        pair_rev = (name_b, name_a)
        if not any(pair == ep or pair_rev == ep for ep in expected_pairs):
            msg = (
                f"Boundary mismatch: segment ({idx_a},{idx_b}) = "
                f"{name_a}↔{name_b} assigned to {assigned_vein}, "
                f"but expected {expected_pairs}"
            )
            report.boundary_mismatches.append(msg)
            report.warnings.append(msg)

    # 2. Full VEIN_Y_ORDER check using line-based median Y and Y-extents
    prev_median_y = -float("inf")
    prev_name = None
    prev_extent = (0.0, 0.0)
    y_tolerance = 0.05

    for name in VEIN_Y_ORDER:
        if name not in vein_map:
            continue
        mp = vein_map[name]
        coords = np.array(mp.line.coords)
        ys_norm = (coords[:, 1] - min_y) / bbox_h if bbox_h > 0 else np.full(len(coords), 0.5)
        median_y = float(np.median(ys_norm))
        y_min_norm = float(ys_norm.min())
        y_max_norm = float(ys_norm.max())

        if prev_name is not None and (prev_median_y - median_y) > y_tolerance:
            msg = (
                f"Vein ordering violation: {name} "
                f"(median_Y={median_y:.3f}, extent={y_min_norm:.2f}-{y_max_norm:.2f}) "
                f"is anterior to {prev_name} "
                f"(median_Y={prev_median_y:.3f}, extent={prev_extent[0]:.2f}-{prev_extent[1]:.2f})"
            )
            report.warnings.append(msg)
        elif prev_name is not None:
            # Log Y-extent overlap as debug info
            overlap_lo = max(prev_extent[0], y_min_norm)
            overlap_hi = min(prev_extent[1], y_max_norm)
            if overlap_lo < overlap_hi:
                logger.debug(
                    "Y-extent overlap: %s ↔ %s: %.2f-%.2f",
                    prev_name, name, overlap_lo, overlap_hi,
                )

        prev_median_y = median_y
        prev_name = name
        prev_extent = (y_min_norm, y_max_norm)

    # 2b. Crossvein connectivity: verify crossveins are near their expected longitudinals
    cv_proximity_threshold = 50.0  # pixels
    for cv_name, (long_a, long_b) in CROSSVEIN_CONNECTIONS.items():
        if cv_name not in vein_map:
            continue
        cv_line = vein_map[cv_name].line
        for long_name in (long_a, long_b):
            if long_name not in vein_map:
                continue
            dist = cv_line.distance(vein_map[long_name].line)
            if dist > cv_proximity_threshold:
                msg = (
                    f"Crossvein connectivity: {cv_name} is {dist:.0f}px from "
                    f"{long_name} (expected <{cv_proximity_threshold:.0f}px)"
                )
                report.warnings.append(msg)

    # 3. Region area check
    total_area = sum(polygons[i].area for i in poly_names)
    if total_area > 0:
        for idx, name in poly_names.items():
            if name in REGION_AREA_PRIORS:
                frac = polygons[idx].area / total_area
                lo, hi = REGION_AREA_PRIORS[name]
                if frac < lo * 0.5 or frac > hi * 2.0:
                    msg = f"Area outlier: {name} = {frac:.1%} (expected {lo:.1%}-{hi:.1%})"
                    report.area_flags.append(msg)
                    report.warnings.append(msg)

    # 4. Coverage: fraction of total centerline length assigned to named veins
    total_centerline_len = sum(l.length for l in centerlines.values())
    assigned_len = sum(mp.length_px for mp in vein_map.values())
    if total_centerline_len > 0:
        report.coverage_fraction = assigned_len / total_centerline_len
        if report.coverage_fraction < 0.5:
            msg = f"Low vein coverage: {report.coverage_fraction:.0%} of centerline length assigned"
            report.warnings.append(msg)

    logger.info(
        "Cross-validation: %d warnings, coverage=%.0f%%",
        len(report.warnings), report.coverage_fraction * 100,
    )
    return report


# ---------------------------------------------------------------------------
# 3g. Top-Level Orchestrator
# ---------------------------------------------------------------------------

def identify_veins_and_regions(
    centerlines: dict[tuple[int, int], LineString],
    intervein_polygons: list[Polygon],
    vein_polygons: list[Polygon],
    image_shape: tuple[int, int],
    wing_bbox: tuple[float, float, float, float],
    midline=None,
) -> IdentificationResult:
    """Identify veins and regions independently using geometry, then cross-validate."""
    result = IdentificationResult()

    if not centerlines:
        logger.warning("No centerlines to identify")
        return result

    # 3a. Find triple junctions
    junctions = find_triple_junctions(centerlines, snap_radius=30.0)

    # 3b. Merge segments at junctions
    paths, merge_decisions = merge_segments_at_junctions(centerlines, junctions)

    # 3b'. Split merged paths at sharp turns
    paths = _split_on_sharp_turns(paths, centerlines, angle_threshold_deg=70.0)

    # 3c. Classify merged paths → named veins
    vein_map = classify_merged_paths(paths, wing_bbox, midline=midline,
                                     junctions=junctions)

    # 3c'. Validate junction topology — retry merges if crossvein/longitudinal
    # orientation is wrong at any triple junction
    vein_map = _validate_junction_orientations(
        vein_map, junctions, centerlines, merge_decisions,
        wing_bbox, midline=midline,
    )

    # 3d. Validate vein shapes
    shape_warnings = validate_vein_shapes(vein_map)
    for vein_id, warns in shape_warnings.items():
        for w in warns:
            logger.warning("Shape: %s — %s", vein_id, w)

    # 3e. Name regions from bounding veins
    poly_names = name_regions_from_veins(
        intervein_polygons, vein_map, wing_bbox,
    )

    # 3e''. Split merged polygons where pixel classifier merged regions
    poly_names, intervein_polygons, split_infos = split_merged_polygons(
        poly_names, intervein_polygons, vein_map, wing_bbox,
        image_shape=image_shape,
    )

    # 3e'''. Post-split centerline extraction
    # For each split, extract the missing centerline between the two new pieces
    # and merge it into the corresponding vein
    for si in split_infos:
        if not si.separating_vein or si.separating_vein not in vein_map:
            continue
        if not vein_polygons:
            continue

        new_line = extract_centerline_between_polygons(
            intervein_polygons[si.orig_idx],
            intervein_polygons[si.new_idx],
            vein_polygons,
            image_shape,
        )
        if new_line is None:
            logger.info(
                "  No post-split centerline extracted for %s (P%d↔P%d)",
                si.separating_vein, si.orig_idx, si.new_idx,
            )
            continue

        # Merge into the existing vein
        mp = vein_map[si.separating_vein]
        merged = _merge_vein_lines([mp.line, new_line])
        if merged is not None and merged.length > mp.length_px:
            old_len = mp.length_px
            mp.line = merged
            mp.length_px = merged.length
            # Add a synthetic segment key for the new piece
            new_key = (si.orig_idx, si.new_idx)
            if new_key not in mp.segment_keys:
                mp.segment_keys.append(new_key)
            logger.info(
                "  Merged post-split centerline into %s: %.0fpx → %.0fpx (+%.0fpx)",
                si.separating_vein, old_len, mp.length_px, mp.length_px - old_len,
            )

    # 3f. Cross-validate
    validation = cross_validate(
        vein_map, poly_names, centerlines, intervein_polygons, wing_bbox,
    )
    # Add shape warnings to validation report
    for vein_id, warns in shape_warnings.items():
        for w in warns:
            validation.warnings.append(f"Shape/{vein_id}: {w}")

    result.poly_names = poly_names
    result.polygons = intervein_polygons
    result.validation_report = validation

    # Build VeinAssignment objects from vein_map
    assignments: list[VeinAssignment] = []
    for vein_id in ["L1", "L2", "L3", "L4", "L5", "ACV", "PCV"]:
        if vein_id in vein_map:
            mp = vein_map[vein_id]
            coords = list(mp.line.coords)
            status = (
                VeinStatus.COMPLETE if len(mp.segment_keys) == 1
                else VeinStatus.FRAGMENTED
            )
            assignments.append(VeinAssignment(
                vein_id=vein_id,
                status=status,
                edge_ids=[],
                confidence=0.85,
                evidence=["geometry_classification"],
                length_px=mp.length_px,
                line=mp.line,
                endpoints=[coords[0], coords[-1]],
            ))
        else:
            assignments.append(VeinAssignment(
                vein_id=vein_id,
                status=VeinStatus.ABSENT,
                edge_ids=[],
                confidence=0.0,
                evidence=["not_found_in_geometry"],
            ))

    # Costa is added by the caller (extracted separately)
    result.assignments = assignments

    return result


# ---------------------------------------------------------------------------
# 3h. Ground-Truth Diagnostic Validation
# ---------------------------------------------------------------------------

def _normalize_region_name(name: str) -> str:
    """Normalize a ground-truth region name to internal format."""
    # Lowercase, replace spaces with underscores, ensure _cell suffix
    normalized = name.strip().lower().replace(" ", "_")
    if not normalized.endswith("_cell"):
        normalized += "_cell"
    return normalized


def validate_regions_against_ground_truth(
    poly_names: dict[int, str],
    polygons: list[Polygon],
    expected_geojson_path: Path,
) -> dict:
    """Compare named regions against ground-truth expected overlay.

    Returns a dict with per-polygon results and overall accuracy.
    """
    # Load expected overlay GeoJSON
    with open(expected_geojson_path) as f:
        data = json.load(f)

    # Parse expected regions: list of (name, Polygon)
    expected_regions: list[tuple[str, Polygon]] = []
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        classification = props.get("classification", {})
        raw_name = classification.get("name", "")

        if not raw_name:
            continue

        geom_type = geom.get("type")
        norm_name = _normalize_region_name(raw_name)

        if geom_type == "Polygon":
            coords = geom.get("coordinates", [])
            if not coords:
                continue
            try:
                poly = Polygon(coords[0])
                if poly.is_valid and not poly.is_empty:
                    expected_regions.append((norm_name, poly))
            except Exception:
                continue
        elif geom_type == "MultiPolygon":
            # Use the largest polygon from the MultiPolygon
            best_poly = None
            best_area = 0.0
            for poly_coords in geom.get("coordinates", []):
                if not poly_coords:
                    continue
                try:
                    poly = Polygon(poly_coords[0])
                    if poly.is_valid and not poly.is_empty and poly.area > best_area:
                        best_area = poly.area
                        best_poly = poly
                except Exception:
                    continue
            if best_poly is not None:
                expected_regions.append((norm_name, best_poly))

    if not expected_regions:
        logger.warning("No valid expected regions found in %s", expected_geojson_path)
        return {"accuracy": 0.0, "per_polygon": {}, "n_expected": 0}

    # For each input polygon, find the best IoU match among expected regions
    results: dict[int, dict] = {}
    correct = 0
    validated = 0  # polygons that have a GT match (IoU > threshold)
    iou_threshold = 0.05  # minimum IoU to consider a GT match exists

    for idx, our_name in poly_names.items():
        if idx >= len(polygons):
            continue
        poly = polygons[idx]

        best_iou = 0.0
        best_expected_name = ""
        for exp_name, exp_poly in expected_regions:
            try:
                intersection = poly.intersection(exp_poly).area
                union = poly.union(exp_poly).area
                iou = intersection / union if union > 0 else 0.0
            except Exception:
                iou = 0.0

            if iou > best_iou:
                best_iou = iou
                best_expected_name = exp_name

        if best_iou < iou_threshold:
            # No GT region overlaps this polygon — can't validate it
            results[idx] = {
                "our_name": our_name,
                "expected_name": "",
                "iou": best_iou,
                "match": None,  # not validatable
                "area": poly.area,
            }
        else:
            match = (our_name == best_expected_name)
            validated += 1
            if match:
                correct += 1
            results[idx] = {
                "our_name": our_name,
                "expected_name": best_expected_name,
                "iou": best_iou,
                "match": match,
                "area": poly.area,
            }

    accuracy = correct / validated if validated > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "validated": validated,
        "total": len(poly_names),
        "per_polygon": results,
        "n_expected": len(expected_regions),
    }


def validate_veins_against_ground_truth(
    assignments: list[VeinAssignment],
    expected_geojson_path: Path,
    tolerance_px: float = 25.0,
    n_samples: int = 200,
) -> VeinValidationReport:
    """Compare predicted vein centerlines against ground-truth skeleton lines.

    Loads GT skeleton from a GeoJSON FeatureCollection of LineStrings with
    properties.classification.name = "L1"/"L2"/etc.  Skips "wing outline".

    Per-vein metrics:
    - Hausdorff distance (px)
    - Mean lateral deviation: sample n_samples points along predicted, measure to GT
    - P95 lateral deviation: 95th percentile of those distances
    - Coverage ratio: fraction of n_samples GT points within tolerance_px of predicted
    """
    report = VeinValidationReport()

    # Load GT skeleton
    with open(expected_geojson_path) as f:
        data = json.load(f)

    gt_veins: dict[str, LineString] = {}
    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        classification = props.get("classification", {})
        name = classification.get("name", "")

        if geom.get("type") != "LineString":
            continue
        if not name or name.lower() == "wing outline":
            continue

        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        try:
            line = LineString(coords)
            if line.is_valid and line.length > 0:
                gt_veins[name] = line
        except Exception:
            continue

    report.total_gt_veins = len(gt_veins)
    report.total_pred_veins = sum(
        1 for a in assignments if a.line is not None and a.status != VeinStatus.ABSENT
    )

    if not gt_veins:
        logger.warning("No valid GT vein lines found in %s", expected_geojson_path)
        return report

    # Match predicted veins to GT by name
    for assignment in assignments:
        if assignment.line is None or assignment.status == VeinStatus.ABSENT:
            continue

        vein_name = assignment.vein_id
        gt_line = gt_veins.get(vein_name)
        if gt_line is None:
            continue

        pred_line = assignment.line

        # Hausdorff distance (Shapely built-in)
        hausdorff = pred_line.hausdorff_distance(gt_line)

        # Sample points along predicted line, measure distance to GT
        pred_dists = []
        for i in range(n_samples):
            pt = pred_line.interpolate(i / max(n_samples - 1, 1), normalized=True)
            pred_dists.append(pt.distance(gt_line))
        pred_dists_arr = np.array(pred_dists)
        mean_dev = float(pred_dists_arr.mean())
        p95_dev = float(np.percentile(pred_dists_arr, 95))

        # Coverage: fraction of GT samples within tolerance of predicted
        gt_covered = 0
        for i in range(n_samples):
            pt = gt_line.interpolate(i / max(n_samples - 1, 1), normalized=True)
            if pt.distance(pred_line) <= tolerance_px:
                gt_covered += 1
        coverage = gt_covered / n_samples

        metrics = VeinMetrics(
            vein_name=vein_name,
            hausdorff_px=hausdorff,
            mean_deviation_px=mean_dev,
            p95_deviation_px=p95_dev,
            coverage_ratio=coverage,
            gt_length_px=gt_line.length,
            pred_length_px=pred_line.length,
        )
        report.per_vein.append(metrics)
        report.matched_count += 1

    if report.per_vein:
        report.mean_hausdorff = float(np.mean([m.hausdorff_px for m in report.per_vein]))
        report.mean_coverage = float(np.mean([m.coverage_ratio for m in report.per_vein]))

    logger.info(
        "Vein GT validation: %d/%d matched, mean Hausdorff=%.1fpx, mean coverage=%.1f%%",
        report.matched_count, report.total_gt_veins,
        report.mean_hausdorff, report.mean_coverage * 100,
    )

    return report
