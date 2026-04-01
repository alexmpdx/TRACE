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
    config: PipelineConfig | None = None,
) -> list[VeinIdentification]:
    """Identify veins in the skeleton graph using landmarks."""
    if config is None:
        config = PipelineConfig()

    G = skel_graph.graph
    edge_labels: dict[tuple, str] = {}

    # Phase 1: Label edges at landmark positions
    _label_landmark_edges(G, landmarks, edge_labels, config)

    # Phase 2: Reserved for future spatial/crossvein assignment
    # (proximity-only assignment is too greedy — direction + topology needed)

    # Phase 3: Build VeinIdentification objects
    veins = _build_vein_identifications(G, edge_labels)

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

    # L2-L3 junction: need to distinguish L2 vs L3 direction
    lm_l2l3 = landmarks.get("L2-L3")
    lm_dtip = landmarks.get("DTip")
    lm_l1rs = landmarks.get("L1-Rs")

    if lm_l2l3 and lm_l2l3.snapped_node is not None:
        node = lm_l2l3.snapped_node
        if node in G:
            neighbors = list(G.neighbors(node))
            sample_px = config.departure_sample

            if len(neighbors) >= 1:
                if lm_dtip:
                    # The edge heading toward DTip is L3 (or L2)
                    toward_dtip = direction_toward(
                        (G.nodes[node]["x"], G.nodes[node]["y"]),
                        (lm_dtip.x, lm_dtip.y),
                    )
                    # Also check toward L1-Rs for Rs
                    toward_l1rs = None
                    if lm_l1rs:
                        toward_l1rs = direction_toward(
                            (G.nodes[node]["x"], G.nodes[node]["y"]),
                            (lm_l1rs.x, lm_l1rs.y),
                        )

                    scored = []
                    for n in neighbors:
                        dep = edge_departure_direction(G, node, n, sample_px)
                        angle_dtip = angle_between_vectors(dep, toward_dtip)
                        angle_l1rs = angle_between_vectors(dep, toward_l1rs) if toward_l1rs else 180
                        scored.append((n, angle_dtip, angle_l1rs))

                    # Best toward DTip → L3-side (could be L2 or L3)
                    scored.sort(key=lambda s: s[1])
                    best_dtip = scored[0]
                    key = _edge_key(node, best_dtip[0])
                    # If it's not already labeled as L3, label as L2
                    # (the DTip edge itself should already be L3 if DTip is in same component)
                    if key not in edge_labels:
                        edge_labels[key] = "L2"
                        logger.info("Labeled edge %s as L2 (from L2-L3, toward DTip)", key)

                    # If there's a second edge heading toward L1-Rs → Rs
                    if len(scored) >= 2 and toward_l1rs:
                        scored_rs = sorted(scored, key=lambda s: s[2])
                        best_rs = scored_rs[0]
                        key_rs = _edge_key(node, best_rs[0])
                        if key_rs not in edge_labels:
                            edge_labels[key_rs] = "Rs"
                            logger.info("Labeled edge %s as Rs (from L2-L3, toward L1-Rs)", key_rs)

    # L1-Rs junction
    lm_l1rs = landmarks.get("L1-Rs")
    lm_sc = landmarks.get("subcostal break")

    if lm_l1rs and lm_l1rs.snapped_node is not None:
        node = lm_l1rs.snapped_node
        if node in G:
            neighbors = list(G.neighbors(node))
            sample_px = config.departure_sample

            for n in neighbors:
                key = _edge_key(node, n)
                if key in edge_labels:
                    continue
                # Check if this edge heads toward subcostal break → L1
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

    # L4-L5 junction
    lm_l4l5 = landmarks.get("L4-L5")
    if lm_l4l5 and lm_l4l5.snapped_node is not None:
        node = lm_l4l5.snapped_node
        if node in G:
            neighbors = list(G.neighbors(node))
            if lm_dtip and len(neighbors) >= 1:
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
                # Most toward DTip (anterior-distal) → L4
                key = _edge_key(node, scored[0][0])
                if key not in edge_labels:
                    edge_labels[key] = "L4"
                    logger.info("Labeled edge %s as L4 (from L4-L5)", key)

                # Remaining → L5
                if len(scored) >= 2:
                    key = _edge_key(node, scored[-1][0])
                    if key not in edge_labels:
                        edge_labels[key] = "L5"
                        logger.info("Labeled edge %s as L5 (from L4-L5)", key)


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
            merged = _merge_nearby_lines(lines)
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


def _merge_nearby_lines(lines: list[LineString]) -> LineString:
    """Merge multiple LineStrings into one, ordering by spatial proximity."""
    if len(lines) <= 1:
        return lines[0] if lines else LineString()

    # Greedy nearest-neighbor chain
    remaining = list(lines)
    result_coords = list(remaining.pop(0).coords)

    while remaining:
        end = Point(result_coords[-1])
        start = Point(result_coords[0])

        best_idx = 0
        best_dist = float("inf")
        best_reverse = False
        best_prepend = False

        for i, line in enumerate(remaining):
            coords = list(line.coords)
            # Check both ends of result against both ends of candidate
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

        next_line = remaining.pop(best_idx)
        next_coords = list(next_line.coords)
        if best_reverse:
            next_coords = next_coords[::-1]

        if best_prepend:
            result_coords = next_coords + result_coords[1:]
        else:
            result_coords = result_coords + next_coords[1:]

    return LineString(result_coords)


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
