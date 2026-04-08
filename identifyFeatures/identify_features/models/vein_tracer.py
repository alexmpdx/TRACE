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

    # Phase 2b: Propagate labels through degree-2 pass-through nodes
    _propagate_through_degree2(G, edge_labels)

    # Phase 3: Detect L6 (short posterior branch off L5 near L4-L5)
    _detect_l6(G, edge_labels, landmarks)

    # Phase 4: Detect crossveins (ACV between L3↔L4, PCV between L4↔L5)
    _detect_crossveins(G, edge_labels)

    # Phase 4b: Fallback crossvein detection using crossvein landmarks
    _detect_crossveins_fallback(G, edge_labels, landmarks, config, skel_graph.median_vein_width_px)

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

    # L2-L3 junction: simultaneous matching with L2.d and DTip
    lm_l2l3 = landmarks.get("L2-L3")
    lm_dtip = landmarks.get("DTip")
    lm_l1rs = landmarks.get("L1-Rs")
    lm_l2d = landmarks.get("L2.d")
    max_lm_dist = config.snap_radius

    if lm_l2l3 and lm_l2l3.snapped_node is not None:
        node = lm_l2l3.snapped_node
        if node in G:
            neighbors = _unlabeled_neighbors(node)
            sample_px = config.departure_sample

            if len(neighbors) >= 1:
                # Check if L3 already labeled at this junction (from DTip endpoint)
                l3_already = any(edge_labels.get(_edge_key(node, n)) == "L3" for n in G.neighbors(node))

                # Simultaneous: for each edge, compute distance to L2.d and DTip
                # Assign L2 to edge nearest L2.d, L3 to edge nearest DTip
                l2_assigned = False
                l3_assigned = l3_already

                if not l3_already and lm_l2d and lm_l2d.snapped_node is not None and lm_dtip:
                    # Score all edges against both landmarks
                    scores = []
                    for n in neighbors:
                        _, dist_l2d = _nearest_edge_to_landmark(node, [n], lm_l2d)
                        _, dist_dtip = _nearest_edge_to_landmark(node, [n], lm_dtip)
                        scores.append((n, dist_l2d, dist_dtip))

                    # L2: edge with smallest dist to L2.d (if within snap radius)
                    scores_l2 = sorted(scores, key=lambda s: s[1])
                    if scores_l2[0][1] <= max_lm_dist:
                        best_l2 = scores_l2[0][0]
                        key = _edge_key(node, best_l2)
                        if key not in edge_labels:
                            edge_labels[key] = "L2"
                            logger.info(
                                "Labeled edge %s as L2 (from L2-L3, nearest to L2.d, dist=%.0f)", key, scores_l2[0][1]
                            )
                            l2_assigned = True

                    # L3: edge with smallest dist to DTip (excluding L2 edge)
                    remaining_scores = [s for s in scores if _edge_key(node, s[0]) not in edge_labels]
                    if remaining_scores:
                        scores_l3 = sorted(remaining_scores, key=lambda s: s[2])
                        best_l3 = scores_l3[0][0]
                        key = _edge_key(node, best_l3)
                        if key not in edge_labels:
                            edge_labels[key] = "L3"
                            logger.info(
                                "Labeled edge %s as L3 (from L2-L3, nearest to DTip, dist=%.0f)", key, scores_l3[0][2]
                            )
                            l3_assigned = True

                elif lm_l2d and lm_l2d.snapped_node is not None:
                    # Only L2.d available (L3 already labeled or no DTip)
                    best_l2, dist = _nearest_edge_to_landmark(node, neighbors, lm_l2d)
                    if best_l2 is not None and dist <= max_lm_dist:
                        key = _edge_key(node, best_l2)
                        if key not in edge_labels:
                            edge_labels[key] = "L2"
                            logger.info("Labeled edge %s as L2 (from L2-L3, nearest to L2.d, dist=%.0f)", key, dist)
                            l2_assigned = True

                elif not l3_already and lm_dtip:
                    # Only DTip available (no L2.d)
                    toward_dtip = direction_toward(
                        (G.nodes[node]["x"], G.nodes[node]["y"]),
                        (lm_dtip.x, lm_dtip.y),
                    )
                    scored = []
                    for n in neighbors:
                        dep = edge_departure_direction(G, node, n, sample_px)
                        angle = angle_between_vectors(dep, toward_dtip)
                        scored.append((n, angle))
                    scored.sort(key=lambda s: s[1])
                    key = _edge_key(node, scored[0][0])
                    if key not in edge_labels:
                        edge_labels[key] = "L3"
                        logger.info("Labeled edge %s as L3 (from L2-L3, toward DTip fallback)", key)

                # Remaining → Rs
                remaining = [n for n in neighbors if _edge_key(node, n) not in edge_labels]
                for n in remaining:
                    key = _edge_key(node, n)
                    if key not in edge_labels:
                        edge_labels[key] = "Rs"
                        logger.info("Labeled edge %s as Rs (from L2-L3, remaining)", key)

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

    # L4-L5 junction: simultaneous matching with L4.d and L5.d
    lm_l4l5 = landmarks.get("L4-L5")
    lm_l4d = landmarks.get("L4.d")
    lm_l5d = landmarks.get("L5.d")

    if lm_l4l5 and lm_l4l5.snapped_node is not None:
        node = lm_l4l5.snapped_node
        if node in G:
            neighbors = _unlabeled_neighbors(node)
            used_soft = False

            if (
                len(neighbors) >= 2
                and lm_l4d
                and lm_l5d
                and lm_l4d.snapped_node is not None
                and lm_l5d.snapped_node is not None
            ):
                # Simultaneous: for each edge, compute distance to both L4.d and L5.d
                # Assign each edge to whichever landmark it's closer to
                scores = []
                for n in neighbors:
                    _, dist_l4d = _nearest_edge_to_landmark(node, [n], lm_l4d)
                    _, dist_l5d = _nearest_edge_to_landmark(node, [n], lm_l5d)
                    scores.append((n, dist_l4d, dist_l5d))

                # Only use soft landmarks if at least one edge is within snap radius
                any_close = any(min(s[1], s[2]) <= max_lm_dist for s in scores)
                if any_close:
                    # Find the single best edge for each landmark
                    best_for_l4 = min(scores, key=lambda s: s[1])
                    best_for_l5 = min(scores, key=lambda s: s[2])

                    if best_for_l4[0] != best_for_l5[0]:
                        # Different edges — assign directly
                        for n, dist_l4, dist_l5 in scores:
                            key = _edge_key(node, n)
                            if key in edge_labels:
                                continue
                            if n == best_for_l4[0]:
                                edge_labels[key] = "L4"
                                logger.info(
                                    "Labeled edge %s as L4 (from L4-L5, nearest to L4.d, dist=%.0f)", key, dist_l4
                                )
                            else:
                                edge_labels[key] = "L5"
                                logger.info(
                                    "Labeled edge %s as L5 (from L4-L5, nearest to L5.d, dist=%.0f)", key, dist_l5
                                )
                    else:
                        # Both landmarks point to same edge — give it to whichever
                        # landmark is closer, assign the other edge the other vein
                        winner = best_for_l4[0]
                        if best_for_l4[1] <= best_for_l5[2]:
                            # L4.d is closer to this edge
                            for n, dist_l4, dist_l5 in scores:
                                key = _edge_key(node, n)
                                if key in edge_labels:
                                    continue
                                if n == winner:
                                    edge_labels[key] = "L4"
                                    logger.info(
                                        "Labeled edge %s as L4 (from L4-L5, contested, L4.d closer: %.0f)", key, dist_l4
                                    )
                                else:
                                    edge_labels[key] = "L5"
                                    logger.info(
                                        "Labeled edge %s as L5 (from L4-L5, contested, assigned remaining)", key
                                    )
                        else:
                            # L5.d is closer
                            for n, dist_l4, dist_l5 in scores:
                                key = _edge_key(node, n)
                                if key in edge_labels:
                                    continue
                                if n == winner:
                                    edge_labels[key] = "L5"
                                    logger.info(
                                        "Labeled edge %s as L5 (from L4-L5, contested, L5.d closer: %.0f)", key, dist_l5
                                    )
                                else:
                                    edge_labels[key] = "L4"
                                    logger.info(
                                        "Labeled edge %s as L4 (from L4-L5, contested, assigned remaining)", key
                                    )
                    used_soft = True

            if not used_soft and len(neighbors) >= 1 and lm_dtip:
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


def _propagate_through_degree2(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
) -> None:
    """Propagate vein labels through degree-2 pass-through nodes.

    At any degree-2 node where one edge is labeled and the other is not,
    the unlabeled edge gets the same label. Repeats until stable.
    """
    changed = True
    while changed:
        changed = False
        for node in G.nodes():
            if G.degree(node) != 2:
                continue
            neighbors = list(G.neighbors(node))
            key0 = _edge_key(node, neighbors[0])
            key1 = _edge_key(node, neighbors[1])
            lbl0 = edge_labels.get(key0)
            lbl1 = edge_labels.get(key1)

            if lbl0 is not None and lbl1 is None:
                edge_labels[key1] = lbl0
                changed = True
                logger.debug("Propagated %s through deg-2 node %d", lbl0, node)
            elif lbl1 is not None and lbl0 is None:
                edge_labels[key0] = lbl1
                changed = True
                logger.debug("Propagated %s through deg-2 node %d", lbl1, node)


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


def _detect_crossveins_fallback(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    landmarks: dict[str, Landmark],
    config: "PipelineConfig",
    median_vein_width: float,
) -> None:
    """Fallback crossvein detection using crossvein landmark points.

    Runs after primary detection. For each crossvein not yet found:
    - Tier 2: Use the more reliable landmark (ACV.p for ACV, PCV.a for PCV)
    - Tier 3: Use the less reliable landmark (ACV.a for ACV, PCV.p for PCV)

    Candidates are scored by length (prefer crossvein-sized) and
    perpendicularity to nearby labeled longitudinals.
    """
    from identify_features.utils.geometry_utils import (
        angle_between_vectors,
        line_direction,
    )

    min_len = median_vein_width * config.crossvein_min_length_vw
    max_len = median_vein_width * config.crossvein_max_length_vw
    search_radius = config.snap_radius
    sample_px = config.departure_sample

    # Build labeled vein lines for perpendicularity checks
    vein_lines: dict[str, list[LineString]] = defaultdict(list)
    for (u, v), label in edge_labels.items():
        if G.has_edge(u, v):
            line = G[u][v].get("line")
            if line:
                vein_lines[label].append(line)

    # Crossvein landmark tiers: (cv_name, [(landmark_name, adjacent_longitudinals), ...])
    cv_tiers = {
        "ACV": [("ACV.p", ["L3", "L4"]), ("ACV.a", ["L3", "L4"])],
        "PCV": [("PCV.a", ["L4", "L5"]), ("PCV.p", ["L4", "L5"])],
    }

    for cv_name, tiers in cv_tiers.items():
        # Skip if already found by primary detection
        if any(label == cv_name for label in edge_labels.values()):
            continue

        for lm_name, adj_veins in tiers:
            lm = landmarks.get(lm_name)
            if lm is None:
                continue

            # Find unlabeled edges near this landmark
            candidates = []
            for u, v, data in G.edges(data=True):
                key = _edge_key(u, v)
                if key in edge_labels:
                    continue
                line = data.get("line")
                if line is None:
                    continue
                length = data.get("length_px", 0)

                # Length filter
                if length < min_len or length > max_len:
                    continue

                # Distance to landmark
                dist = line.distance(lm.point)
                if dist > search_radius:
                    continue

                # Perpendicularity score against adjacent longitudinals
                perp_score = 0.0
                perp_count = 0
                edge_dir = line_direction(line, sample_px=line.length)

                for adj_vein in adj_veins:
                    for adj_line in vein_lines.get(adj_vein, []):
                        # Find direction of longitudinal at nearest point to candidate
                        mid = line.interpolate(0.5, normalized=True)
                        proj_dist = adj_line.project(mid)
                        if proj_dist <= 0 or proj_dist >= adj_line.length:
                            continue
                        # Sample longitudinal direction at the projected point
                        half_win = min(sample_px / 2, proj_dist, adj_line.length - proj_dist)
                        if half_win < 1:
                            continue
                        pt_a = adj_line.interpolate(proj_dist - half_win)
                        pt_b = adj_line.interpolate(proj_dist + half_win)
                        long_dir = (pt_b.x - pt_a.x, pt_b.y - pt_a.y)
                        mag = (long_dir[0] ** 2 + long_dir[1] ** 2) ** 0.5
                        if mag < 1e-6:
                            continue
                        long_dir = (long_dir[0] / mag, long_dir[1] / mag)

                        angle = angle_between_vectors(edge_dir, long_dir)
                        # Normalize to 0-90 (direction doesn't matter)
                        if angle > 90:
                            angle = 180 - angle
                        # Score: 1.0 at 90° (perfect perp), 0.0 at 0° (parallel)
                        perp_score += angle / 90.0
                        perp_count += 1

                if perp_count > 0:
                    perp_score /= perp_count
                else:
                    perp_score = 0.5  # no longitudinal to check — neutral

                # Length score: 1.0 at ideal length, lower at extremes
                ideal_len = (min_len + max_len) / 2
                length_score = 1.0 - abs(length - ideal_len) / ideal_len
                length_score = max(0.0, length_score)

                # Combined: proximity + perpendicularity + length
                score = dist + (1 - perp_score) * 200 + (1 - length_score) * 100
                candidates.append((key, score, length, dist, perp_score))

            if candidates:
                candidates.sort(key=lambda c: c[1])
                best_key, best_score, best_len, best_dist, best_perp = candidates[0]
                edge_labels[best_key] = cv_name
                u, v = best_key
                logger.info(
                    "Detected %s (fallback via %s): edge %d↔%d, %.0fpx, dist=%.0f, perp=%.2f",
                    cv_name,
                    lm_name,
                    u,
                    v,
                    best_len,
                    best_dist,
                    best_perp,
                )
                break  # Found it, don't try next tier


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
