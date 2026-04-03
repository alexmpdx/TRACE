"""Identify veins by assigning labels to graph edges.

The skeleton graph may be fragmented (multiple disconnected components).
Labels are assigned in three phases:

1. Landmark-based: edges containing landmark nodes get labeled by
   departure direction and landmark identity.
2. Spatial assignment: unlabeled edges are matched to the nearest
   compatible labeled vein by proximity and direction.
3. Junction resolution: at degree-3+ nodes, tangent continuity
   determines which vein continues through.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Optional

import networkx as nx
from identify_features.config import PipelineConfig
from identify_features.models.datatypes import (
    Landmark,
    SkeletonGraph,
    VeinIdentification,
    VeinStatus,
    VeinType,
)
from identify_features.utils.geometry_utils import (
    angle_between_vectors,
    direction_toward,
)
from identify_features.utils.graph_utils import (
    edge_departure_direction,
    edge_line_from_node,
)
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)


def trace_veins_from_landmarks(
    skel_graph: SkeletonGraph,
    landmarks: dict[str, Landmark],
    wing_outline: "Polygon | None" = None,
    config: PipelineConfig | None = None,
) -> list[VeinIdentification]:
    """Identify veins in the skeleton graph using landmarks.

    Args:
        skel_graph: Skeleton graph (after landmark anchoring).
        landmarks: Anchored landmarks dict.
        wing_outline: Wing outline polygon for costa detection.
        config: Pipeline configuration.
    """
    if config is None:
        config = PipelineConfig()

    G = skel_graph.graph
    edge_labels: dict[tuple, str] = {}

    # Phase 0: Merge longitudinals through crossvein junctions (graph-level)
    # Done BEFORE labeling so merged edges get a single label.
    # Landmark nodes are protected from being contracted.
    from identify_features.models.junction_resolver import merge_through_junctions

    protected = {lm.snapped_node for lm in landmarks.values() if lm.snapped_node is not None}
    skel_graph.graph = merge_through_junctions(G, edge_labels, config, protected_nodes=protected)
    G = skel_graph.graph

    # Phase 1: Detect costa edges using margin band (on the merged graph)
    if wing_outline is not None:
        from identify_features.models.costa_detector import detect_costa_edges

        costa_keys, _ = detect_costa_edges(skel_graph, landmarks, wing_outline, config)
        for key in costa_keys:
            edge_labels[key] = "costa"

    # Phase 2: Label edges at landmark positions (on the merged graph)
    _label_landmark_edges(G, landmarks, edge_labels, config)

    # Phase 3: Detect L6 (short posterior branch off L5 near L4-L5)
    _detect_l6(G, edge_labels, landmarks)

    # Phase 4: Detect crossveins (ACV between L3↔L4, PCV between L4↔L5)
    _detect_crossveins(G, edge_labels)

    # Phase 5: Build VeinIdentification objects
    merge_gap = config.to_px(config.merge_max_gap_um) if config.um_per_px else 100.0
    veins = _build_vein_identifications(G, edge_labels, max_merge_gap_px=merge_gap)

    return veins


def _label_landmark_edges(
    G: nx.Graph,
    landmarks: dict[str, Landmark],
    edge_labels: dict[tuple, str],
    config: PipelineConfig,
) -> None:
    """Label edges connected to landmark nodes."""

    # Helper: label the edge at a degree-1 landmark node
    def _label_endpoint_edge(landmark_name: str, vein_id: str):
        lm = landmarks.get(landmark_name)
        if lm is None or lm.snapped_node is None:
            return
        node = lm.snapped_node
        if node not in G:
            return
        for neighbor in G.neighbors(node):
            key = _edge_key(node, neighbor)
            if key not in edge_labels:
                edge_labels[key] = vein_id
                logger.info("Labeled edge %s as %s (from %s landmark)", key, vein_id, landmark_name)
                break

    # DTip → the edge there is L3's distal end
    _label_endpoint_edge("DTip", "L3")

    # Helper: get non-costa neighbors at a junction node
    def _unlabeled_neighbors(node):
        """Return neighbors whose edges aren't already labeled (e.g. costa)."""
        result = []
        for n in G.neighbors(node):
            key = _edge_key(node, n)
            if key not in edge_labels:
                result.append(n)
        return result

    # Helper: find the edge whose LineString passes closest to a landmark point
    def _nearest_edge_to_landmark(node, neighbors, landmark):
        """Among edges from node to neighbors, find which passes closest to landmark."""
        best_n = None
        best_dist = float("inf")
        for n in neighbors:
            line = G[node][n].get("line")
            if line is None:
                continue
            dist = line.distance(landmark.point)
            if dist < best_dist:
                best_dist = dist
                best_n = n
        return best_n, best_dist

    # L2-L3 junction
    lm_l2l3 = landmarks.get("L2-L3")
    lm_dtip = landmarks.get("DTip")
    lm_l1rs = landmarks.get("L1-Rs")
    lm_l2d = landmarks.get("L2.d")

    if lm_l2l3 and lm_l2l3.snapped_node is not None:
        node = lm_l2l3.snapped_node
        if node in G:
            neighbors = _unlabeled_neighbors(node)
            sample_px = config.departure_sample

            if len(neighbors) >= 1:
                # 1) Edge nearest to L2.d → L2 (identify first to prevent DTip stealing it)
                if lm_l2d and lm_l2d.snapped_node is not None:
                    best_l2, _ = _nearest_edge_to_landmark(node, neighbors, lm_l2d)
                    if best_l2 is not None:
                        key_l2 = _edge_key(node, best_l2)
                        if key_l2 not in edge_labels:
                            edge_labels[key_l2] = "L2"
                            logger.info("Labeled edge %s as L2 (from L2-L3, nearest to L2.d)", key_l2)

                remaining = [n for n in neighbors if _edge_key(node, n) not in edge_labels]

                # 2) L3: check if DTip's edge already connects to this node
                # (already labeled by _label_endpoint_edge). If not, use
                # direction toward DTip among remaining edges.
                l3_already_at_junction = any(edge_labels.get(_edge_key(node, n)) == "L3" for n in G.neighbors(node))
                if not l3_already_at_junction and remaining and lm_dtip:
                    toward_dtip = direction_toward(
                        (G.nodes[node]["x"], G.nodes[node]["y"]),
                        (lm_dtip.x, lm_dtip.y),
                    )
                    scored = []
                    for n in remaining:
                        dep = edge_departure_direction(G, node, n, sample_px)
                        angle = angle_between_vectors(dep, toward_dtip)
                        scored.append((n, angle))
                    scored.sort(key=lambda s: s[1])
                    best_l3 = scored[0][0]
                    key = _edge_key(node, best_l3)
                    if key not in edge_labels:
                        edge_labels[key] = "L3"
                        logger.info("Labeled edge %s as L3 (from L2-L3, toward DTip)", key)

                remaining = [n for n in neighbors if _edge_key(node, n) not in edge_labels]

                # 3) Best toward L1-Rs → Rs. Any still remaining → Rs.
                if remaining and lm_l1rs:
                    toward_l1rs = direction_toward(
                        (G.nodes[node]["x"], G.nodes[node]["y"]),
                        (lm_l1rs.x, lm_l1rs.y),
                    )
                    scored_rs = []
                    for n in remaining:
                        dep = edge_departure_direction(G, node, n, sample_px)
                        angle = angle_between_vectors(dep, toward_l1rs)
                        scored_rs.append((n, angle))
                    scored_rs.sort(key=lambda s: s[1])
                    best_rs = scored_rs[0][0]
                    key_rs = _edge_key(node, best_rs)
                    if key_rs not in edge_labels:
                        edge_labels[key_rs] = "Rs"
                        logger.info("Labeled edge %s as Rs (from L2-L3, toward L1-Rs)", key_rs)

                remaining = [n for n in neighbors if _edge_key(node, n) not in edge_labels]

                # 4) Any still remaining → Rs (after L2 and L3, everything else at L2-L3 is Rs)
                for n in remaining:
                    key_rs = _edge_key(node, n)
                    if key_rs not in edge_labels:
                        edge_labels[key_rs] = "Rs"
                        logger.info("Labeled edge %s as Rs (from L2-L3, remaining)", key_rs)

    # L1-Rs junction
    lm_l1rs = landmarks.get("L1-Rs")
    lm_sc = landmarks.get("subcostal break")

    if lm_l1rs and lm_l1rs.snapped_node is not None:
        node = lm_l1rs.snapped_node
        if node in G:
            neighbors = _unlabeled_neighbors(node)
            sample_px = config.departure_sample

            for n in neighbors:
                key = _edge_key(node, n)
                if key in edge_labels:
                    continue
                if lm_sc:
                    dep = edge_departure_direction(G, node, n, sample_px)
                    toward_sc = direction_toward(
                        (G.nodes[node]["x"], G.nodes[node]["y"]),
                        (lm_sc.x, lm_sc.y),
                    )
                    angle = angle_between_vectors(dep, toward_sc)
                    if angle < 60:
                        edge_labels[key] = "L1"
                        logger.info("Labeled edge %s as L1 (from L1-Rs, toward SC)", key)
                    else:
                        edge_labels[key] = "Rs"
                        logger.info("Labeled edge %s as Rs (from L1-Rs, away from SC)", key)
                else:
                    edge_labels[key] = "Rs"

    # Subcostal break → L1
    _label_endpoint_edge("subcostal break", "L1")

    # L4-L5 junction: use L4.d and L5.d landmarks for identification
    lm_l4l5 = landmarks.get("L4-L5")
    lm_l4d = landmarks.get("L4.d")
    lm_l5d = landmarks.get("L5.d")

    if lm_l4l5 and lm_l4l5.snapped_node is not None:
        node = lm_l4l5.snapped_node
        if node in G:
            neighbors = _unlabeled_neighbors(node)

            if len(neighbors) >= 2 and lm_l4d and lm_l4d.snapped_node is not None:
                # Edge nearest to L4.d → L4
                best_l4, _ = _nearest_edge_to_landmark(node, neighbors, lm_l4d)
                if best_l4 is not None:
                    key = _edge_key(node, best_l4)
                    if key not in edge_labels:
                        edge_labels[key] = "L4"
                        logger.info("Labeled edge %s as L4 (from L4-L5, nearest to L4.d)", key)

                remaining = [n for n in neighbors if _edge_key(node, n) not in edge_labels]

                # Edge nearest to L5.d → L5. Fallback: remaining.
                if remaining:
                    if lm_l5d and lm_l5d.snapped_node is not None:
                        best_l5, _ = _nearest_edge_to_landmark(node, remaining, lm_l5d)
                        if best_l5 is not None:
                            key = _edge_key(node, best_l5)
                            if key not in edge_labels:
                                edge_labels[key] = "L5"
                                logger.info("Labeled edge %s as L5 (from L4-L5, nearest to L5.d)", key)
                                remaining.remove(best_l5)
                    # Any still remaining → L5
                    for n in remaining:
                        key = _edge_key(node, n)
                        if key not in edge_labels:
                            edge_labels[key] = "L5"
                            logger.info("Labeled edge %s as L5 (from L4-L5, remaining)", key)

            elif len(neighbors) >= 1 and lm_dtip:
                # Fallback if no L4.d landmark: use DTip direction
                toward_dtip = direction_toward(
                    (G.nodes[node]["x"], G.nodes[node]["y"]),
                    (lm_dtip.x, lm_dtip.y),
                )
                sample_px = config.departure_sample
                scored = []
                for n in neighbors:
                    dep = edge_departure_direction(G, node, n, sample_px)
                    angle = angle_between_vectors(dep, toward_dtip)
                    scored.append((n, angle))
                scored.sort(key=lambda s: s[1])
                key = _edge_key(node, scored[0][0])
                if key not in edge_labels:
                    edge_labels[key] = "L4"
                    logger.info("Labeled edge %s as L4 (from L4-L5, toward DTip fallback)", key)
                if len(scored) >= 2:
                    key = _edge_key(node, scored[-1][0])
                    if key not in edge_labels:
                        edge_labels[key] = "L5"
                        logger.info("Labeled edge %s as L5 (from L4-L5, remaining fallback)", key)


def _assign_by_proximity(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    config: PipelineConfig,
) -> None:
    """Assign unlabeled edges to the nearest compatible labeled vein.

    For each unlabeled edge, find the labeled edge whose LineString is
    closest and roughly parallel. Assign the same label.
    """
    # Build labeled line index
    labeled_lines: dict[str, list[LineString]] = defaultdict(list)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            labeled_lines[label].append(G[u][v]["line"])

    if not labeled_lines:
        return

    # Assign unlabeled edges
    changed = True
    max_rounds = 5

    for _ in range(max_rounds):
        if not changed:
            break
        changed = False

        for u, v, data in list(G.edges(data=True)):
            key = _edge_key(u, v)
            if key in edge_labels:
                continue

            line = data.get("line")
            if line is None:
                continue

            # Find nearest labeled vein
            best_label = None
            best_dist = float("inf")

            midpoint = line.interpolate(0.5, normalized=True)

            for label, lines in labeled_lines.items():
                for lline in lines:
                    dist = lline.distance(midpoint)
                    if dist < best_dist:
                        best_dist = dist
                        best_label = label

            # Only assign if reasonably close (within ~500px)
            if best_label is not None and best_dist < 500:
                edge_labels[key] = best_label
                labeled_lines[best_label].append(line)
                changed = True
                logger.debug("Proximity-assigned edge %s as %s (dist=%.0fpx)", key, best_label, best_dist)


def _build_vein_identifications(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    max_merge_gap_px: float = float("inf"),
) -> list[VeinIdentification]:
    """Build VeinIdentification objects from labeled edges."""
    vein_edges: dict[str, list[tuple]] = defaultdict(list)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            vein_edges[label].append((u, v))

    veins = []
    for vein_id, edges in sorted(vein_edges.items()):
        # Collect all LineStrings for this vein
        lines = []
        for u, v in edges:
            line = G[u][v].get("line")
            if line is not None:
                lines.append(line)

        # Merge into single LineString if connected, or MultiLineString
        if len(lines) == 1:
            merged = lines[0]
        elif len(lines) > 1:
            merged = _merge_nearby_lines(lines, max_gap_px=max_merge_gap_px)
        else:
            merged = None

        vein = VeinIdentification(
            vein_id=vein_id,
            vein_type=_vein_type(vein_id),
            status=VeinStatus.IDENTIFIED,
            centerline=merged,
            edge_ids=[G[u][v].get("edge_id", -1) for u, v in edges],
            length_px=sum(l.length for l in lines),
            evidence=[f"{len(edges)} edges"],
        )
        veins.append(vein)
        logger.info("Identified %s: %.0fpx (%d edges)", vein_id, vein.length_px, len(edges))

    # Report unlabeled
    all_keys = {_edge_key(u, v) for u, v in G.edges()}
    unlabeled = all_keys - set(edge_labels.keys())
    if unlabeled:
        logger.info("%d edges remain unlabeled", len(unlabeled))

    return veins


def _merge_nearby_lines(
    lines: list[LineString],
    max_gap_px: float = float("inf"),
) -> LineString:
    """Merge multiple LineStrings into one, ordering by spatial proximity.

    Lines that are farther apart than max_gap_px are NOT connected —
    only lines within the gap threshold are chained together. Distant
    lines are skipped to avoid drawing long straight connectors.
    """
    if len(lines) <= 1:
        return lines[0] if lines else LineString()

    # Greedy nearest-neighbor chain
    remaining = list(lines)
    result_coords = list(remaining.pop(0).coords)

    while remaining:
        end = Point(result_coords[-1])
        start = Point(result_coords[0])

        best_idx = None
        best_dist = float("inf")
        best_reverse = False
        best_prepend = False

        for i, line in enumerate(remaining):
            coords = list(line.coords)
            for prepend in (False, True):
                ref = start if prepend else end
                for reverse in (False, True):
                    candidate_end = Point(coords[-1] if reverse else coords[0])
                    dist = ref.distance(candidate_end)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                        best_reverse = reverse
                        best_prepend = prepend

        # Stop if the nearest remaining line is too far away
        if best_idx is None or best_dist > max_gap_px:
            break

        next_line = remaining.pop(best_idx)
        next_coords = list(next_line.coords)

        if best_prepend:
            # Shared point must be at next_coords[-1] — invert reverse logic
            if not best_reverse:
                next_coords = next_coords[::-1]
            result_coords = next_coords[:-1] + result_coords
        else:
            # Shared point must be at next_coords[0]
            if best_reverse:
                next_coords = next_coords[::-1]
            result_coords = result_coords + next_coords[1:]

    return LineString(result_coords)


def _detect_l6(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    landmarks: dict[str, Landmark],
) -> None:
    """Detect L6: a short posterior branch off L5 near L4-L5.

    L6 branches from L5 near the proximal end (within 0.5-1.5× Rs length
    from L4-L5) and heads posteriorly. It's similar in length to Rs/L1
    and may be absent.
    """
    # Need Rs length as reference
    rs_length = None
    for key, label in edge_labels.items():
        if label == "Rs":
            u, v = key
            if G.has_edge(u, v):
                rs_length = G[u][v].get("length_px", 0)
                break

    if rs_length is None or rs_length < 10:
        return

    # Find L4-L5 landmark position
    l4l5 = landmarks.get("L4-L5")
    if l4l5 is None:
        return

    l4l5_x, l4l5_y = l4l5.x, l4l5.y

    # Look for unlabeled edges that:
    # 1. Have at least one endpoint near the L4-L5 area (within 1.5× Rs)
    # 2. Are short (0.5-1.5× Rs length)
    # 3. Head posteriorly (positive Y direction = toward bottom of wing)
    min_length = rs_length * 0.5
    max_length = rs_length * 1.5
    max_dist_from_l4l5 = rs_length * 1.5

    best_candidate = None
    best_score = float("inf")

    for u, v, data in G.edges(data=True):
        key = _edge_key(u, v)
        if key in edge_labels:
            continue

        length = data.get("length_px", 0)
        if length < min_length or length > max_length:
            continue

        # Check if either endpoint is near L4-L5
        nd_u = G.nodes[u]
        nd_v = G.nodes[v]
        dist_u = math.hypot(nd_u["x"] - l4l5_x, nd_u["y"] - l4l5_y)
        dist_v = math.hypot(nd_v["x"] - l4l5_x, nd_v["y"] - l4l5_y)
        min_dist = min(dist_u, dist_v)

        if min_dist > max_dist_from_l4l5:
            continue

        # Check direction: must head posteriorly (positive Y)
        line = data.get("line")
        if line is None:
            continue
        start = line.coords[0]
        end = line.coords[-1]
        dy = end[1] - start[1]
        # Positive dy = heading posteriorly (down in image)
        if abs(dy) < length * 0.3:
            continue  # not heading substantially in Y direction
        if dy < 0:
            # Check reversed
            dy = -dy
            if dy < length * 0.3:
                continue

        # Score: prefer edges closer to L4-L5 and more posterior
        score = min_dist
        if score < best_score:
            best_score = score
            best_candidate = key

    if best_candidate is not None:
        edge_labels[best_candidate] = "L6"
        u, v = best_candidate
        length = G[u][v].get("length_px", 0)
        logger.info("Detected L6: edge %d↔%d, %.0fpx", u, v, length)


def _detect_crossveins(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
) -> None:
    """Detect ACV and PCV crossveins.

    After junction merging (Phase 1), crossvein edges are unlabeled
    branches whose endpoints sit on or near two different longitudinal
    veins.  ACV connects L3↔L4, PCV connects L4↔L5.

    Detection: for each crossvein, find unlabeled edges where one
    endpoint is near vein_a and the other near vein_b.  "Near" means
    the node is an endpoint of a labeled edge (shared graph node) or
    its coordinates lie on/close to a labeled vein LineString (typical
    after junction merging contracts the junction node into the
    longitudinal's line).
    """
    from identify_features.models.topology import CROSSVEIN_CONNECTIONS

    # Build lookup: labeled vein LineStrings and endpoint node sets
    vein_lines: dict[str, list[LineString]] = defaultdict(list)
    vein_nodes: dict[str, set[int]] = defaultdict(set)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            line = G[u][v].get("line")
            if line:
                vein_lines[label].append(line)
            vein_nodes[label].add(u)
            vein_nodes[label].add(v)

    for cv_name, (vein_a, vein_b) in CROSSVEIN_CONNECTIONS.items():
        if vein_a not in vein_lines or vein_b not in vein_lines:
            logger.info("Cannot detect %s: %s or %s not labeled", cv_name, vein_a, vein_b)
            continue

        best_key = None
        best_score = float("inf")

        for u, v, data in G.edges(data=True):
            key = _edge_key(u, v)
            if key in edge_labels:
                continue

            if data.get("line") is None:
                continue

            # Try both orientations: u→vein_a / v→vein_b  and  u→vein_b / v→vein_a
            for end_a, end_b in [(u, v), (v, u)]:
                dist_a = _node_vein_distance(G, end_a, vein_a, vein_lines, vein_nodes)
                dist_b = _node_vein_distance(G, end_b, vein_b, vein_lines, vein_nodes)

                if dist_a is not None and dist_b is not None:
                    score = dist_a + dist_b
                    if score < best_score:
                        best_score = score
                        best_key = key
                    break  # valid orientation found

        if best_key is not None:
            edge_labels[best_key] = cv_name
            u, v = best_key
            length = G[u][v].get("length_px", 0)
            logger.info("Detected %s: edge %d↔%d, %.0fpx (score=%.1f)", cv_name, u, v, length, best_score)
        else:
            logger.info("No candidate found for %s", cv_name)


def _node_vein_distance(
    G: nx.Graph,
    node: int,
    vein_label: str,
    vein_lines: dict[str, list[LineString]],
    vein_nodes: dict[str, set[int]],
    max_dist: float = 50.0,
) -> Optional[float]:
    """Distance from a graph node to a labeled vein.

    Returns 0 if the node shares a graph edge endpoint with the vein.
    Returns geometric distance to the nearest vein LineString if within
    *max_dist* (covers the post-merge case where the junction node's
    coordinates are embedded in the merged longitudinal line).
    Returns None if too far — the node is not connected to this vein.
    """
    # Direct graph connectivity: node is an endpoint of a vein edge
    if node in vein_nodes.get(vein_label, set()):
        return 0.0

    # Geometric proximity (post-merge: node coords on the LineString)
    pt = Point(G.nodes[node]["x"], G.nodes[node]["y"])
    min_dist = float("inf")
    for line in vein_lines.get(vein_label, []):
        d = line.distance(pt)
        if d < min_dist:
            min_dist = d

    if min_dist <= max_dist:
        return min_dist

    return None


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (min(u, v), max(u, v))


def _vein_type(vein_id: str) -> VeinType:
    if vein_id == "Rs":
        return VeinType.RADIAL_SECTOR
    elif vein_id in ("ACV", "PCV"):
        return VeinType.CROSSVEIN
    elif vein_id == "costa":
        return VeinType.COSTA
    else:
        return VeinType.LONGITUDINAL
