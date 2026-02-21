"""Independent geometry-based vein and region identification with cross-validation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from WingVeinAnalyzer.models.vein_labeler import (
    VeinAssignment,
    VeinStatus,
    _extract_costa,
    _merge_vein_lines,
)
from WingVeinAnalyzer.models.vein_map import (
    MAX_ANGLE_CHANGE_DEG,
    REGION_AREA_PRIORS,
    REGION_EXPECTED_VEINS,
    STRAIGHTNESS_THRESHOLD,
    VEIN_BOUNDARIES,
    VEIN_LENGTH_PRIORS,
    VEIN_ORIENTATION_PRIORS,
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
    straightness: float = 0.0
    length_px: float = 0.0


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
    line: LineString, endpoint_idx: int, tangent_dist: float = 50.0,
) -> np.ndarray:
    """Compute tangent vector pointing AWAY from a junction endpoint."""
    coords = list(line.coords)
    if endpoint_idx == 0:
        # Junction is at start → tangent points from start toward interior
        jx, jy = coords[0]
        dist = min(tangent_dist, 0.2 * line.length)
        pt = line.interpolate(dist)
        return np.array([pt.x - jx, pt.y - jy])
    else:
        # Junction is at end → tangent points from end toward interior
        jx, jy = coords[-1]
        dist = max(0, line.length - min(tangent_dist, 0.2 * line.length))
        pt = line.interpolate(dist)
        return np.array([pt.x - jx, pt.y - jy])


def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in degrees [0, 180]."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def merge_segments_at_junctions(
    centerlines: dict[tuple[int, int], LineString],
    junctions: list[JunctionPoint],
    collinearity_threshold_deg: float = 45.0,
    min_gap_deg: float = 15.0,
) -> list[MergedPath]:
    """Merge segments at triple junctions by tangent continuity.

    Only merges when the best collinear pair is clearly better than
    the second-best pair (gap > min_gap_deg), preventing ambiguous merges
    at Y-junctions where multiple pairs look equally collinear.
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

        # Only merge if: (1) best is below threshold, (2) clear gap over 2nd
        if best_score < collinearity_threshold_deg and (second_score - best_score) > min_gap_deg:
            key_a = arrivals[bi][0]
            key_b = arrivals[bj][0]
            union(key_a, key_b)
            logger.debug(
                "Merged %s + %s at junction (%.0f, %.0f), "
                "collinearity=%.1f°, gap=%.1f°",
                key_a, key_b, junc.x, junc.y,
                best_score, second_score - best_score,
            )
        else:
            logger.debug(
                "Skipped merge at junction (%.0f, %.0f): "
                "best=%.1f°, gap=%.1f° (need >%.1f°)",
                junc.x, junc.y, best_score,
                second_score - best_score, min_gap_deg,
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
    return paths


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

    line = path.line
    n_steps = max(2, int(line.length / step_dist))

    # Sample points along the merged line
    sample_pts = [
        line.interpolate(i / n_steps, normalized=True)
        for i in range(n_steps + 1)
    ]

    # Find the sharpest turn
    max_angle = 0.0
    max_angle_dist = 0.0
    for i in range(1, len(sample_pts) - 1):
        dx1 = sample_pts[i].x - sample_pts[i - 1].x
        dy1 = sample_pts[i].y - sample_pts[i - 1].y
        dx2 = sample_pts[i + 1].x - sample_pts[i].x
        dy2 = sample_pts[i + 1].y - sample_pts[i].y

        v1 = np.array([dx1, dy1])
        v2 = np.array([dx2, dy2])
        angle_change = _angle_between_vectors(v1, v2)

        if angle_change > max_angle:
            max_angle = angle_change
            # Distance along the line where the sharp turn occurs
            max_angle_dist = (i / n_steps) * line.length

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
) -> dict[str, MergedPath]:
    """Classify merged paths into named veins by geometry."""
    min_x, min_y, max_x, max_y = wing_bbox
    bbox_w = max_x - min_x

    # Compute features for all paths
    for p in paths:
        _compute_path_features(p, wing_bbox)

    # Split into longitudinal vs crossvein candidates
    longitudinals: list[MergedPath] = []
    crossveins: list[MergedPath] = []

    for p in paths:
        if p.orientation_deg > 60:
            crossveins.append(p)
        elif p.orientation_deg < 30:
            longitudinals.append(p)
        else:
            # Ambiguous orientation — classify by length
            if p.length_px > 300:
                longitudinals.append(p)
            else:
                crossveins.append(p)

    # Assign longitudinals using combined Y-position + length scoring
    vein_map = _assign_longitudinals_scored(longitudinals, bbox_w)

    # Crossvein assignment
    if len(crossveins) >= 2:
        # ACV is more proximal (lower X centroid), PCV is more distal
        crossveins.sort(key=lambda p: p.x_centroid_norm)
        vein_map["ACV"] = crossveins[0]
        vein_map["PCV"] = crossveins[1]
    elif len(crossveins) == 1:
        cv = crossveins[0]
        # Determine identity by which longitudinal pair it lies between
        l3_y = vein_map["L3"].y_centroid_norm if "L3" in vein_map else 0.35
        l4_y = vein_map["L4"].y_centroid_norm if "L4" in vein_map else 0.5
        l5_y = vein_map["L5"].y_centroid_norm if "L5" in vein_map else 0.65
        mid_acv = (l3_y + l4_y) / 2
        mid_pcv = (l4_y + l5_y) / 2
        if abs(cv.y_centroid_norm - mid_acv) < abs(cv.y_centroid_norm - mid_pcv):
            vein_map["ACV"] = cv
        else:
            vein_map["PCV"] = cv

    logger.info(
        "Classified veins: %s",
        {k: f"{v.length_px:.0f}px" for k, v in vein_map.items()},
    )
    return vein_map


def _assign_longitudinals_scored(
    longitudinals: list[MergedPath],
    wing_span_px: float,
) -> dict[str, MergedPath]:
    """Assign longitudinal veins using optimal combinatorial scoring.

    Tries all valid subsets of paths (k=1..min(n,5)) and vein names,
    respecting Y ordering, to find the globally optimal assignment.
    Allows fewer than 5 veins when some are absent.
    """
    from itertools import combinations

    long_names = ["L1", "L2", "L3", "L4", "L5"]
    result: dict[str, MergedPath] = {}

    if not longitudinals:
        return result

    # Sort by Y centroid for ordering constraint
    sorted_paths = sorted(longitudinals, key=lambda p: p.y_centroid_norm)
    n = len(sorted_paths)

    # Precompute scores for all (path, name) pairs
    score_matrix: dict[tuple[int, str], float] = {}
    for pi, path in enumerate(sorted_paths):
        for name in long_names:
            score_matrix[(pi, name)] = _longitudinal_match_score(
                path, name, wing_span_px,
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
            for name_indices in combinations(range(5), k):
                scores = [
                    score_matrix[(path_indices[i], long_names[name_indices[i]])]
                    for i in range(k)
                ]
                # Skip if any single assignment is terrible
                if min(scores) < min_per_vein:
                    continue
                total = sum(scores)
                # Normalize: average score + small bonus for more veins
                norm = total / k + 0.05 * k
                if norm > best_norm_score:
                    best_norm_score = norm
                    best_assignment = [
                        (path_indices[i], long_names[name_indices[i]])
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

    return result


def _longitudinal_match_score(
    path: MergedPath, vein_name: str, wing_span_px: float,
) -> float:
    """Score how well a path matches a specific longitudinal vein identity."""
    # Y-position score: closeness to expected Y centroid range
    y_lo, y_hi = SPATIAL_PRIORS_Y[vein_name]
    y_mid = (y_lo + y_hi) / 2
    y_range = y_hi - y_lo
    y_dist = abs(path.y_centroid_norm - y_mid)
    if y_lo <= path.y_centroid_norm <= y_hi:
        y_score = 1.0 - (y_dist / y_range)
    else:
        y_score = max(0.0, 0.5 - y_dist)

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

    # Combined score (weighted: Y position is primary, length is secondary)
    return 0.6 * y_score + 0.4 * len_score


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

    for idx, poly in enumerate(polygons):
        bounding = poly_veins[idx]
        cx = poly.centroid.x
        cy = poly.centroid.y
        y_norm = (cy - min_y) / bbox_h if bbox_h > 0 else 0.5

        name = _region_from_bounding_veins(
            bounding, poly, vein_map, wing_bbox, bbox_h,
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

    return poly_names


def _region_from_bounding_veins(
    bounding_veins: set[str],
    poly: Polygon,
    vein_map: dict[str, MergedPath],
    wing_bbox: tuple[float, float, float, float],
    bbox_h: float,
) -> Optional[str]:
    """Determine region name from its bounding veins and position."""
    cx = poly.centroid.x
    cy = poly.centroid.y
    min_x, min_y, max_x, max_y = wing_bbox
    y_norm = (cy - min_y) / bbox_h if bbox_h > 0 else 0.5

    # Special case: costal_cell is anterior to L1, bounded only by L1
    if bounding_veins == {"L1"}:
        l1_y = vein_map["L1"].y_centroid_norm if "L1" in vein_map else 0.1
        if y_norm < l1_y:
            return "costal_cell"

    # Check each region's expected veins — best match wins
    best_name = None
    best_score = -1.0

    for region_name, expected_veins in REGION_EXPECTED_VEINS.items():
        if region_name == "costal_cell":
            continue  # handled above
        matched = bounding_veins & expected_veins
        if not matched:
            continue
        # Score: Jaccard-like — reward overlap, penalize extra/missing
        score = len(matched) / len(expected_veins | bounding_veins)
        # Bonus only for exact match (bounding veins == expected veins)
        if bounding_veins == expected_veins:
            score += 1.0

        if score > best_score:
            best_score = score
            best_name = region_name

    if best_name is None:
        return None

    # Disambiguate regions that share the same expected veins using position
    if best_name in ("1st_basal_cell", "1st_posterior_cell"):
        acv_x = None
        if "ACV" in vein_map:
            acv_coords = np.array(vein_map["ACV"].line.coords)
            acv_x = acv_coords[:, 0].mean()
        if acv_x is not None:
            best_name = "1st_basal_cell" if cx < acv_x else "1st_posterior_cell"
        else:
            bbox_mid_x = (min_x + max_x) / 2
            best_name = "1st_basal_cell" if cx < bbox_mid_x else "1st_posterior_cell"

    elif best_name in ("discal_cell", "2nd_posterior_cell"):
        pcv_x = None
        if "PCV" in vein_map:
            pcv_coords = np.array(vein_map["PCV"].line.coords)
            pcv_x = pcv_coords[:, 0].mean()
        if pcv_x is not None:
            best_name = "discal_cell" if cx < pcv_x else "2nd_posterior_cell"
        else:
            bbox_mid_x = (min_x + max_x) / 2
            best_name = "discal_cell" if cx < bbox_mid_x else "2nd_posterior_cell"

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

    # 2. Vein Y ordering: L1.y < L2.y < L3.y < L4.y < L5.y
    long_names = ["L1", "L2", "L3", "L4", "L5"]
    prev_y = -float("inf")
    for name in long_names:
        if name in vein_map:
            y = vein_map[name].y_centroid_norm
            if y < prev_y:
                msg = f"Vein ordering violation: {name} (Y={y:.2f}) is anterior to previous (Y={prev_y:.2f})"
                report.warnings.append(msg)
            prev_y = y

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
) -> IdentificationResult:
    """Identify veins and regions independently using geometry, then cross-validate."""
    result = IdentificationResult()

    if not centerlines:
        logger.warning("No centerlines to identify")
        return result

    # 3a. Find triple junctions
    junctions = find_triple_junctions(centerlines, snap_radius=30.0)

    # 3b. Merge segments at junctions
    paths = merge_segments_at_junctions(centerlines, junctions)

    # 3b'. Split merged paths at sharp turns
    paths = _split_on_sharp_turns(paths, centerlines, angle_threshold_deg=70.0)

    # 3c. Classify merged paths → named veins
    vein_map = classify_merged_paths(paths, wing_bbox)

    # 3d. Validate vein shapes
    shape_warnings = validate_vein_shapes(vein_map)
    for vein_id, warns in shape_warnings.items():
        for w in warns:
            logger.warning("Shape: %s — %s", vein_id, w)

    # 3e. Name regions from bounding veins
    poly_names = name_regions_from_veins(
        intervein_polygons, vein_map, wing_bbox,
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
