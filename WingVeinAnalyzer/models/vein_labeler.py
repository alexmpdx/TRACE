"""Topology-based vein identity assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Polygon

from WingVeinAnalyzer.models.vein_graph import VeinEdge
from WingVeinAnalyzer.models.vein_map import (
    _ALL_INTERVEIN_SPACE_NAMES,
    VEIN_BOUNDARIES,
)


class VeinStatus(Enum):
    COMPLETE = "complete"
    FRAGMENTED = "fragmented"
    TRUNCATED = "truncated"
    ABSENT = "absent"


@dataclass
class VeinAssignment:
    vein_id: str
    status: VeinStatus
    edge_ids: list[int]
    confidence: float
    evidence: list[str] = field(default_factory=list)
    length_px: float = 0.0
    gap_px: float | None = None
    length_um: float | None = None
    line: Optional[LineString] = None
    endpoints: Optional[list[tuple[float, float]]] = None


def assign_veins_from_polygons(
    polygons: list[Polygon],
    edges: list[VeinEdge],
    graph: nx.Graph,
    wing_bbox: Optional[tuple[float, float, float, float]] = None,
) -> tuple[list[VeinAssignment], dict[int, str]]:
    """Assign vein identities based on polygon adjacency topology.

    Returns a list of VeinAssignments and a dict mapping polygon index to
    intervein region name.
    """
    if not polygons or not edges:
        return [], {}

    # Sort polygons by Y centroid (anterior=low Y to posterior=high Y)
    centroids = [(p.centroid.x, p.centroid.y) for p in polygons]
    y_sorted_indices = sorted(range(len(polygons)), key=lambda i: centroids[i][1])

    # Compute wing bounding box for normalization
    if wing_bbox is None:
        all_bounds = [p.bounds for p in polygons]
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        wing_bbox = (min_x, min_y, max_x, max_y)

    bbox_w = wing_bbox[2] - wing_bbox[0]
    bbox_h = wing_bbox[3] - wing_bbox[1]

    # --- Step 1: Identify intervein regions by spatial position ---
    poly_names = _assign_intervein_names(polygons, centroids, y_sorted_indices, bbox_h, wing_bbox)

    # --- Step 2: Identify veins from polygon pair adjacency ---
    assignments: list[VeinAssignment] = []
    used_edge_ids: set[int] = set()

    for vein_id, boundary_pairs in VEIN_BOUNDARIES.items():
        vein_edges = []
        vein_lines = []
        for edge in edges:
            if edge.edge_id in used_edge_ids:
                continue
            if edge.poly_pair is None:
                continue
            pi, pj = edge.poly_pair
            name_i = poly_names.get(pi, "")
            name_j = poly_names.get(pj, "")
            pair = (name_i, name_j)
            pair_rev = (name_j, name_i)
            for expected_pair in boundary_pairs:
                if pair == expected_pair or pair_rev == expected_pair:
                    vein_edges.append(edge)
                    vein_lines.append(edge.line)
                    used_edge_ids.add(edge.edge_id)
                    break

        if not vein_edges:
            assignments.append(
                VeinAssignment(
                    vein_id=vein_id,
                    status=VeinStatus.ABSENT,
                    edge_ids=[],
                    confidence=0.0,
                    evidence=["no_matching_polygon_boundary"],
                )
            )
            continue

        total_length = sum(e.length_px for e in vein_edges)
        combined_line = _merge_vein_lines(vein_lines)
        endpoints = None
        if combined_line:
            coords = list(combined_line.coords)
            endpoints = [coords[0], coords[-1]]

        status = VeinStatus.COMPLETE if len(vein_edges) == 1 else VeinStatus.FRAGMENTED
        confidence = 0.85 if status == VeinStatus.COMPLETE else 0.7

        assignments.append(
            VeinAssignment(
                vein_id=vein_id,
                status=status,
                edge_ids=[e.edge_id for e in vein_edges],
                confidence=confidence,
                evidence=["polygon_boundary_match"],
                length_px=total_length,
                line=combined_line,
                endpoints=endpoints,
            )
        )

    # --- Step 3: Add costa as anterior wing margin ---
    costa_line = _extract_costa(polygons, poly_names, wing_bbox)
    if costa_line:
        assignments.append(
            VeinAssignment(
                vein_id="costa",
                status=VeinStatus.COMPLETE,
                edge_ids=[],
                confidence=0.9,
                evidence=["anterior_margin"],
                length_px=costa_line.length,
                line=costa_line,
                endpoints=[
                    list(costa_line.coords)[0],
                    list(costa_line.coords)[-1],
                ],
            )
        )

    return assignments, poly_names


def assign_veins(
    veins: list,
    graph: nx.Graph,
    nodes: dict,
    wing_bbox: tuple,
) -> list[VeinAssignment]:
    """Assign vein identities using spatial priors and topology (LineString mode)."""
    assignments: list[VeinAssignment] = []
    bbox_min_y, bbox_max_y = wing_bbox[1], wing_bbox[3]
    bbox_h = bbox_max_y - bbox_min_y
    bbox_min_x, bbox_max_x = wing_bbox[0], wing_bbox[2]
    bbox_w = bbox_max_x - bbox_min_x

    if not veins:
        return assignments

    # Classify orientation
    longitudinals = []
    crossvein_candidates = []
    for v in veins:
        coords = np.array(v.line.coords)
        dx = abs(coords[-1][0] - coords[0][0])
        dy = abs(coords[-1][1] - coords[0][1])
        ratio = dx / (dx + dy + 1e-6)
        if ratio > 0.5 and v.length_px > 200:
            longitudinals.append(v)
        elif v.length_px < 400:
            crossvein_candidates.append(v)
        else:
            longitudinals.append(v)

    # Sort longitudinals by Y centroid
    longitudinals.sort(key=lambda v: v.centroid_y)

    # Assign longitudinal veins by Y position
    long_names = ["costa", "L1", "L2", "L3", "L4", "L5"]
    for i, v in enumerate(longitudinals):
        if i >= len(long_names):
            break
        name = long_names[i]
        assignments.append(
            VeinAssignment(
                vein_id=name,
                status=VeinStatus.COMPLETE,
                edge_ids=[],
                confidence=0.7,
                evidence=["y_position_ordering"],
                length_px=v.length_px,
                line=v.line,
                endpoints=[
                    list(v.line.coords)[0],
                    list(v.line.coords)[-1],
                ],
            )
        )

    # Sort crossvein candidates by X midpoint (proximal first)
    crossvein_candidates.sort(key=lambda v: v.centroid_x)
    cv_names = ["ACV", "PCV"]
    for i, v in enumerate(crossvein_candidates[:2]):
        assignments.append(
            VeinAssignment(
                vein_id=cv_names[i],
                status=VeinStatus.COMPLETE,
                edge_ids=[],
                confidence=0.6,
                evidence=["crossvein_position"],
                length_px=v.length_px,
                line=v.line,
                endpoints=[
                    list(v.line.coords)[0],
                    list(v.line.coords)[-1],
                ],
            )
        )

    return assignments


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _assign_intervein_names(
    polygons: list[Polygon],
    centroids: list[tuple[float, float]],
    y_sorted_indices: list[int],
    bbox_h: float,
    wing_bbox: tuple[float, float, float, float],
) -> dict[int, str]:
    """Assign intervein region names to polygons based on spatial position.

    Handles 8 polygons (the expected case) with specific heuristics, and
    falls back to adjacency-aware assignment for other counts.
    """
    n = len(polygons)
    areas = [p.area for p in polygons]
    names: dict[int, str] = {}

    if n == 8:
        return _assign_8_polygons(polygons, centroids, areas)

    # Fallback for unexpected polygon counts
    return _assign_fallback(polygons, centroids, y_sorted_indices, areas)


def _assign_8_polygons(
    polygons: list[Polygon],
    centroids: list[tuple[float, float]],
    areas: list[float],
) -> dict[int, str]:
    """Assign names to exactly 8 intervein polygons using X+Y position and area."""
    n = len(polygons)
    indices = list(range(n))
    names: dict[int, str] = {}

    # Compute normalized centroids for comparison
    all_cx = [centroids[i][0] for i in indices]
    all_cy = [centroids[i][1] for i in indices]
    min_cx, max_cx = min(all_cx), max(all_cx)
    min_cy, max_cy = min(all_cy), max(all_cy)
    range_x = max_cx - min_cx if max_cx > min_cx else 1.0
    range_y = max_cy - min_cy if max_cy > min_cy else 1.0

    def norm_x(i: int) -> float:
        return (centroids[i][0] - min_cx) / range_x

    def norm_y(i: int) -> float:
        return (centroids[i][1] - min_cy) / range_y

    remaining = set(indices)

    # 1. costal_cell: smallest area AND most anterior (min Y centroid)
    # Among the 3 most anterior polygons, pick the smallest
    by_y = sorted(remaining, key=lambda i: centroids[i][1])
    anterior_3 = by_y[:3]
    costal = min(anterior_3, key=lambda i: areas[i])
    names[costal] = "costal_cell"
    remaining.discard(costal)

    # 2. marginal_cell: next most anterior by Y, significantly larger than costal
    by_y = sorted(remaining, key=lambda i: centroids[i][1])
    # Among the 2 most anterior remaining, pick the more anterior one
    marginal = by_y[0]
    names[marginal] = "marginal_cell"
    remaining.discard(marginal)

    # 3. 3rd_posterior_cell: most posterior (highest Y centroid)
    by_y = sorted(remaining, key=lambda i: centroids[i][1], reverse=True)
    # Among the 2 most posterior, pick the more proximal one (lower X)
    posterior_candidates = by_y[:2]
    third_post = min(posterior_candidates, key=lambda i: centroids[i][0])
    names[third_post] = "3rd_posterior_cell"
    remaining.discard(third_post)

    # 4. 2nd_posterior_cell: most posterior remaining, distal (high X, high Y)
    by_y = sorted(remaining, key=lambda i: centroids[i][1], reverse=True)
    second_post = by_y[0]
    names[second_post] = "2nd_posterior_cell"
    remaining.discard(second_post)

    # Remaining 4: submarginal, 1st_basal, 1st_posterior, discal
    rem_list = sorted(remaining, key=lambda i: centroids[i][0])

    # 5. 1st_basal_cell: small, proximal (low X centroid)
    proximal = sorted(remaining, key=lambda i: centroids[i][0])
    basal = proximal[0]
    names[basal] = "1st_basal_cell"
    remaining.discard(basal)

    # 6. submarginal_cell: anterior + distal (low Y, high X) among remaining 3
    rem_sorted_y = sorted(remaining, key=lambda i: centroids[i][1])
    submarginal = rem_sorted_y[0]
    names[submarginal] = "submarginal_cell"
    remaining.discard(submarginal)

    # 7+8. 1st_posterior_cell vs discal_cell: both mid-range
    # 1st_posterior has higher X centroid (distal); discal has lower X (proximal)
    rem_list = sorted(remaining, key=lambda i: centroids[i][0])
    names[rem_list[0]] = "discal_cell"
    names[rem_list[1]] = "1st_posterior_cell"

    return names


def _assign_fallback(
    polygons: list[Polygon],
    centroids: list[tuple[float, float]],
    y_sorted_indices: list[int],
    areas: list[float],
) -> dict[int, str]:
    """Fallback assignment for non-8 polygon counts using Y-band clustering."""
    names: dict[int, str] = {}
    for rank, idx in enumerate(y_sorted_indices):
        if rank < len(_ALL_INTERVEIN_SPACE_NAMES):
            names[idx] = _ALL_INTERVEIN_SPACE_NAMES[rank]
    return names


def _merge_vein_lines(lines: list[LineString]) -> Optional[LineString]:
    """Merge multiple vein line segments into a single continuous LineString."""
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]

    # Try shapely linemerge first
    from shapely.ops import linemerge

    merged = linemerge(lines)
    if isinstance(merged, LineString):
        return merged

    # Manual merge: chain segments by spatial proximity with correct orientation
    segments = list(lines)

    # Find the two outermost endpoints across all segments (farthest pair).
    # These define the natural start and end of the vein.
    all_endpoints: list[tuple[int, bool]] = []  # (seg_idx, is_end)
    for i, seg in enumerate(segments):
        all_endpoints.append((i, False))  # start of segment
        all_endpoints.append((i, True))  # end of segment

    def _ep_coord(seg_idx: int, is_end: bool) -> np.ndarray:
        c = list(segments[seg_idx].coords)
        return np.array(c[-1] if is_end else c[0])

    best_pair_dist = -1.0
    start_seg_idx = 0
    start_is_end = False
    for a in range(len(all_endpoints)):
        for b in range(a + 1, len(all_endpoints)):
            ai, ae = all_endpoints[a]
            bi, be = all_endpoints[b]
            d = float(np.linalg.norm(_ep_coord(ai, ae) - _ep_coord(bi, be)))
            if d > best_pair_dist:
                best_pair_dist = d
                start_seg_idx = ai
                start_is_end = ae

    # Orient the starting segment so the outermost endpoint comes first
    first_coords = list(segments[start_seg_idx].coords)
    if start_is_end:
        first_coords = first_coords[::-1]
    chain = first_coords
    used = {start_seg_idx}

    # Chain remaining segments by nearest endpoint to chain end
    for _ in range(len(segments) - 1):
        chain_end = np.array(chain[-1])
        best_dist = float("inf")
        best_idx = -1
        reverse = False

        for i, seg in enumerate(segments):
            if i in used:
                continue
            seg_coords = list(seg.coords)
            d_start = float(np.linalg.norm(chain_end - np.array(seg_coords[0])))
            d_end = float(np.linalg.norm(chain_end - np.array(seg_coords[-1])))
            if d_start < best_dist:
                best_dist = d_start
                best_idx = i
                reverse = False
            if d_end < best_dist:
                best_dist = d_end
                best_idx = i
                reverse = True

        if best_idx < 0:
            break

        used.add(best_idx)
        seg_coords = list(segments[best_idx].coords)
        if reverse:
            seg_coords = seg_coords[::-1]

        # Append without duplicating the junction point
        chain.extend(seg_coords[1:])

    if len(chain) < 2:
        return None

    result = LineString(chain)

    # Post-merge validation: warn about large jumps
    coords = list(result.coords)
    import logging

    logger = logging.getLogger(__name__)
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i - 1][0]
        dy = coords[i][1] - coords[i - 1][1]
        jump = (dx * dx + dy * dy) ** 0.5
        if jump > 500:
            logger.warning(
                "Large jump %.0fpx between points %d and %d in merged vein",
                jump, i - 1, i,
            )

    return result


def _extract_costa(
    polygons: list[Polygon],
    poly_names: dict[int, str],
    wing_bbox: tuple[float, float, float, float],
) -> Optional[LineString]:
    """Extract the costa as the anterior margin of the most anterior polygon."""
    # Find the marginal cell (or costal cell) - the most anterior large polygon
    marginal_idx = None
    costal_idx = None
    for idx, name in poly_names.items():
        if name == "marginal_cell":
            marginal_idx = idx
        elif name == "costal_cell":
            costal_idx = idx

    target_idx = marginal_idx if marginal_idx is not None else costal_idx
    if target_idx is None:
        return None

    poly = polygons[target_idx]
    ring = poly.exterior
    coords = np.array(ring.coords)

    # The costa is the top (most anterior = lowest Y) boundary of the wing
    # Extract points that are within the top 20% of this polygon's Y range
    min_y = coords[:, 1].min()
    max_y = coords[:, 1].max()
    y_range = max_y - min_y
    top_threshold = min_y + y_range * 0.25

    top_coords = coords[coords[:, 1] < top_threshold]
    if len(top_coords) < 2:
        return None

    # Sort by X to get a proper line from proximal to distal
    top_coords = top_coords[top_coords[:, 0].argsort()]
    return LineString(top_coords).simplify(5.0)
