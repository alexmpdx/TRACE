"""Propagate vein labels through junctions.

At degree-3 junctions where one labeled edge arrives, determines which
unlabeled edge continues the same vein (gets the same label) and which
is a branching vein (stays unlabeled for later identification).

Criteria (applied in order):
1. Slope similarity: the pair of edges with the most similar departure
   directions (most collinear, angle closest to 180°) is the continuation.
2. Length asymmetry: if the shortest edge is < threshold fraction of the
   longest, it's likely the branching vein (crossvein), confirming the
   merge of the two longer edges.
3. Perpendicularity: the branching edge must depart at a steep angle
   from both continuation edges (not collinear with either).

All criteria use a configurable direction window for computing slopes.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import networkx as nx
from identify_features.config import PipelineConfig
from identify_features.utils.geometry_utils import angle_between_vectors
from identify_features.utils.graph_utils import edge_departure_direction

logger = logging.getLogger(__name__)


def resolve_junctions(
    G: nx.Graph,
    edge_labels: dict[tuple, str],
    config: PipelineConfig | None = None,
    median_vein_width_px: float = 0.0,
    max_iterations: int = 20,
) -> None:
    """Propagate vein labels through degree-3+ junctions (in-place).

    At each junction with at least 1 labeled edge and at least 1 unlabeled
    edge, determines the best continuation and assigns the same label.

    Iterates pass-by-pass (snapshot-based) to prevent cascading within
    a single pass. Skips junctions where the vein already passes through
    (2+ edges with the same label = enters and exits).

    Args:
        G: Skeleton graph.
        edge_labels: Edge label dict, modified in-place. Keys are
            (min_node, max_node) tuples.
        config: Pipeline configuration for direction window etc.
        max_iterations: Max propagation passes.
    """
    if config is None:
        config = PipelineConfig()

    direction_window = config.departure_sample_px(median_vein_width_px)

    for iteration in range(max_iterations):
        # Snapshot: only resolve from labels that existed before this pass
        existing_labels = dict(edge_labels)
        new_labels = 0

        for node in list(G.nodes()):
            if G.degree(node) < 3:
                continue

            neighbors = list(G.neighbors(node))

            # Classify edges at this junction as labeled or unlabeled
            labeled = []
            unlabeled = []
            for n in neighbors:
                key = _edge_key(node, n)
                if key in existing_labels:
                    labeled.append((n, existing_labels[key]))
                elif key not in edge_labels:
                    unlabeled.append(n)

            if not labeled or not unlabeled:
                continue

            # Count labels: skip if a vein already passes through (2 edges same label)
            label_counts = Counter(label for _, label in labeled)
            for label, count in label_counts.items():
                if count >= 2:
                    # This vein already enters and exits — don't propagate more
                    labeled = [(n, l) for n, l in labeled if l != label]

            if not labeled:
                continue

            # For each labeled edge, find the best unlabeled continuation
            for labeled_n, label in labeled:
                best = _find_best_continuation(
                    G,
                    node,
                    labeled_n,
                    unlabeled,
                    direction_window,
                    edge_labels,
                )
                if best is not None:
                    key = _edge_key(node, best)
                    edge_labels[key] = label
                    unlabeled.remove(best)
                    new_labels += 1
                    logger.debug(
                        "Junction resolution: %s propagated through node %d",
                        label,
                        node,
                    )

        if new_labels == 0:
            break

        logger.info(
            "Junction resolution pass %d: propagated %d labels",
            iteration + 1,
            new_labels,
        )


def _find_best_continuation(
    G: nx.Graph,
    node: int,
    incoming_neighbor: int,
    candidates: list[int],
    direction_window: float,
    edge_labels: dict[tuple, str],
) -> Optional[int]:
    """Find the best unlabeled continuation of a vein through a junction.

    Uses tangent continuity: the incoming edge's arrival direction is
    compared to each candidate's departure direction. The most collinear
    candidate (angle closest to 180°) is the continuation.

    Returns the best candidate node, or None if no suitable continuation.
    """
    if not candidates:
        return None

    # Compute incoming arrival direction (reversed departure)
    incoming_dep = edge_departure_direction(
        G,
        node,
        incoming_neighbor,
        min(direction_window, G[node][incoming_neighbor].get("length_px", 100) * 0.8),
    )
    if incoming_dep is None:
        return None

    best = None
    best_angle = 0.0

    for candidate in candidates:
        key = _edge_key(node, candidate)
        if key in edge_labels:
            continue  # already labeled in this pass

        dep = edge_departure_direction(
            G,
            node,
            candidate,
            min(direction_window, G[node][candidate].get("length_px", 100) * 0.8),
        )
        if dep is None:
            continue

        # Angle between incoming and candidate departure directions.
        # If they're opposite (180°), the candidate continues straight.
        angle = angle_between_vectors(incoming_dep, dep)
        if angle > best_angle:
            best_angle = angle
            best = candidate

    # Only accept if reasonably straight (> 120° = < 60° deflection)
    if best is not None and best_angle >= 120.0:
        return best

    return None


def merge_through_junctions(
    G: nx.Graph,
    edge_labels: dict[tuple, str] | None = None,
    config: PipelineConfig | None = None,
    protected_nodes: set[int] | None = None,
    median_vein_width_px: float = 0.0,
) -> nx.Graph:
    """Merge longitudinal vein segments through crossvein junctions (graph-level).

    At each degree-3 junction, finds the most collinear edge pair and
    merges them into a single edge, leaving the third edge (crossvein)
    as a branch. The junction node is kept if the crossvein is still
    attached; removed if it becomes isolated.

    Protected edges (those already in edge_labels, e.g. costa) are NOT
    merged with unprotected edges. Protected nodes (e.g. landmark positions)
    are never merged through — they stay as graph nodes.

    Args:
        G: Skeleton graph (modified in-place via copy).
        edge_labels: Already-labeled edges to protect from cross-type merging.
        config: Pipeline configuration.
        protected_nodes: Node IDs that must not be contracted (landmark nodes).

    Returns:
        Modified graph with longitudinals merged through crossvein junctions.
    """
    from shapely.geometry import LineString

    if config is None:
        config = PipelineConfig()
    if edge_labels is None:
        edge_labels = {}
    if protected_nodes is None:
        protected_nodes = set()

    direction_window = config.departure_sample_px(median_vein_width_px)
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
        junctions = [n for n in result.nodes() if result.degree(n) == 3]

        for node in junctions:
            if node not in result or result.degree(node) != 3:
                continue

            # Don't merge through landmark/protected nodes
            if node in protected_nodes:
                continue

            neighbors = list(result.neighbors(node))
            if len(neighbors) != 3:
                continue

            # Compute departure directions and lengths
            edges_info = []
            for n in neighbors:
                data = result[node][n]
                length = data.get("length_px", 0)
                dep = edge_departure_direction(
                    result,
                    node,
                    n,
                    min(direction_window, length * 0.8),
                )
                label = edge_labels.get(_edge_key(node, n))
                # A non-landmark degree-1 neighbor is a dead-end stub (likely
                # ectopic). Such edges should never be treated as the
                # through-vein continuation: a true vein crossing a junction
                # connects two non-stub paths, and the third edge is the
                # crossvein/branch. Without this guard, a short stub whose
                # initial direction happens to be collinear with one of the
                # real vein edges can outscore the correct pairing — which is
                # exactly how L3 on 0004/0010 loses its proximal edge to an
                # ectopic branch.
                is_stub = result.degree(n) == 1 and n not in protected_nodes
                edges_info.append((n, length, dep, label, is_stub))

            # Find the most collinear pair (angle closest to 180°)
            best_pair = None
            best_angle = 0.0

            for i in range(3):
                for j in range(i + 1, 3):
                    ni, li, di, lbl_i, stub_i = edges_info[i]
                    nj, lj, dj, lbl_j, stub_j = edges_info[j]

                    # Don't merge edges with different labels
                    if lbl_i is not None and lbl_j is not None and lbl_i != lbl_j:
                        continue

                    # Don't pair a dead-end stub as the through-vein continuation
                    if stub_i or stub_j:
                        continue

                    if di is not None and dj is not None:
                        angle = angle_between_vectors(di, dj)
                        if angle > best_angle:
                            best_angle = angle
                            best_pair = (i, j)

            if best_pair is None or best_angle < 120.0:
                continue

            i, j = best_pair
            n1, l1, d1, lbl1, _ = edges_info[i]
            n2, l2, d2, lbl2, _ = edges_info[j]

            # The third edge (not in the pair) is the crossvein/branch
            third_idx = 3 - i - j
            n3, l3, d3, lbl3, _ = edges_info[third_idx]

            # Additional check: the third edge should branch off at a
            # steep angle from the merged pair (not collinear with either)
            if d3 is not None:
                angle_to_1 = angle_between_vectors(d3, d1) if d1 else 90
                angle_to_2 = angle_between_vectors(d3, d2) if d2 else 90
                # If the third edge is collinear with either (angle > 150°),
                # this might be a divergence point, not a crossvein junction
                if angle_to_1 > 150 or angle_to_2 > 150:
                    continue

            # Merge the pair
            e1_data = result[node][n1]
            e2_data = result[node][n2]

            merged_line = _merge_edge_lines(e1_data["line"], e2_data["line"], node, result)

            # Determine label for merged edge
            merged_label = lbl1 or lbl2  # keep whichever label exists

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
                # Update edge_labels for the new merged edge
                new_key = _edge_key(n1, n2)
                old_key1 = _edge_key(node, n1)
                old_key2 = _edge_key(node, n2)
                if old_key1 in edge_labels:
                    del edge_labels[old_key1]
                if old_key2 in edge_labels:
                    del edge_labels[old_key2]
                if merged_label:
                    edge_labels[new_key] = merged_label

                next_edge_id += 1

            if result.degree(node) == 0:
                result.remove_node(node)

            changed = True
            logger.debug(
                "Merged through junction node %d: " "pair angle=%.0f°, branch angle=%.0f°/%.0f°",
                node,
                best_angle,
                angle_to_1,
                angle_to_2,
            )

    return result


def _merge_edge_lines(
    line1: "LineString",
    line2: "LineString",
    via_node: int,
    G: nx.Graph,
) -> "LineString":
    """Merge two LineStrings that meet at a shared node."""
    from shapely.geometry import LineString

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


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (min(u, v), max(u, v))
