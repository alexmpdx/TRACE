"""Snap landmarks to skeleton graph nodes."""

from __future__ import annotations

import logging
import math
from typing import Optional

import networkx as nx
from identify_features.models.datatypes import Landmark, SkeletonGraph
from identify_features.models.topology import RELIABLE_LANDMARKS
from identify_features.utils.graph_utils import nearest_node

logger = logging.getLogger(__name__)

# Landmarks that represent junctions (should snap to high-degree nodes)
_JUNCTION_LANDMARKS = {"L1-Rs", "L2-L3", "L4-L5"}

# Landmarks that represent endpoints (prefer degree-1 nodes)
_ENDPOINT_LANDMARKS = {"subcostal break", "DTip"}


def anchor_landmarks(
    skel_graph: SkeletonGraph,
    landmarks: dict[str, Landmark],
    snap_radius: float = 80.0,
    prefer_junction_radius: float = 160.0,
    snap_radius_vw: float = 4.0,
) -> dict[str, Landmark]:
    """Snap each reliable landmark to the nearest appropriate graph node.

    Junction landmarks (L1-Rs, L2-L3, L4-L5) prefer high-degree nodes (≥3).
    Endpoint landmarks (subcostal break, DTip) prefer degree-1 nodes.

    Args:
        skel_graph: The skeleton graph.
        landmarks: Dict of landmarks by name.
        snap_radius: Maximum distance to snap a landmark to a node.
        prefer_junction_radius: Search radius for preferring junction nodes.
        snap_radius_vw: Snap radius as × median vein width (overrides snap_radius).

    Returns:
        Updated landmarks dict with snapped_node and snap_distance set.
    """
    G = skel_graph.graph
    result = dict(landmarks)  # shallow copy

    # Use vein-width-based snap radius when median vein width is available
    if skel_graph.median_vein_width_px > 0:
        snap_radius = skel_graph.median_vein_width_px * snap_radius_vw
        prefer_junction_radius = snap_radius * 2
        logger.info("Snap radius: %.0fpx (%.1f × vein width)", snap_radius, snap_radius_vw)

    for name, lm in result.items():
        if not lm.reliable:
            continue

        if name in _JUNCTION_LANDMARKS:
            node = nearest_node(
                G,
                lm.x,
                lm.y,
                max_dist=snap_radius,
                prefer_degree=3,
                prefer_degree_radius=snap_radius,
            )
            # Junction landmarks must snap to degree-3+ nodes.
            # If only a degree-1 node was found, reject it and fall
            # through to edge insertion (which creates a split point).
            if node is not None and G.degree(node) < 2:
                logger.debug(
                    "Landmark %r: rejecting degree-%d node %s, will insert on edge",
                    name,
                    G.degree(node),
                    node,
                )
                node = None
        elif name in _ENDPOINT_LANDMARKS:
            node = nearest_node(
                G,
                lm.x,
                lm.y,
                max_dist=snap_radius,
                prefer_degree=1,
                prefer_degree_radius=snap_radius,
            )
        elif name == "alula notch":
            # Alula notch is a wing margin reference point — don't modify the graph
            logger.debug("Landmark %r: margin reference, skipping graph modification", name)
            continue
        else:
            # Other landmarks — snap to nearest node, no degree preference
            node = nearest_node(G, lm.x, lm.y, max_dist=snap_radius)

        if node is not None:
            node_data = G.nodes[node]
            dist = math.hypot(node_data["x"] - lm.x, node_data["y"] - lm.y)
            lm.snapped_node = node
            lm.snap_distance = dist
            logger.info(
                "Landmark %r snapped to node %s (degree %d) at distance %.1fpx",
                name,
                node,
                G.degree(node),
                dist,
            )
        else:
            # No node close enough — try to split the nearest edge
            inserted = _insert_node_on_nearest_edge(
                G,
                lm.x,
                lm.y,
                max_dist=prefer_junction_radius * 2,
            )
            if inserted is not None:
                lm.snapped_node = inserted
                node_data = G.nodes[inserted]
                lm.snap_distance = math.hypot(node_data["x"] - lm.x, node_data["y"] - lm.y)
                logger.info(
                    "Landmark %r inserted on edge at distance %.1fpx (degree %d)",
                    name,
                    lm.snap_distance,
                    G.degree(inserted),
                )
            else:
                logger.warning(
                    "Landmark %r could not be snapped or inserted",
                    name,
                )

    return result


def _insert_node_on_nearest_edge(
    G: nx.Graph,
    x: float,
    y: float,
    max_dist: float,
) -> int | None:
    """Insert a new node on the nearest edge to (x, y).

    Splits the edge at the nearest point to the landmark, creating
    a new node and two new edges.

    Returns the new node ID, or None if no edge is within max_dist.
    """
    from shapely.geometry import Point

    target = Point(x, y)
    best_edge = None
    best_dist = max_dist
    best_point = None

    for u, v, data in G.edges(data=True):
        line = data["line"]
        dist = line.distance(target)
        if dist < best_dist:
            best_dist = dist
            best_edge = (u, v)
            best_point = line.interpolate(line.project(target))

    if best_edge is None:
        return None

    u, v = best_edge
    edge_data = G[u][v].copy()
    line = edge_data["line"]

    # Split the line at the projection point
    split_dist = line.project(best_point)
    min_split_dist = 20.0  # Don't create segments shorter than this
    if split_dist < min_split_dist or split_dist > line.length - min_split_dist:
        # Too close to an existing endpoint — just snap to that node
        if split_dist < 1.0:
            return (
                u
                if math.hypot(G.nodes[u]["x"] - x, G.nodes[u]["y"] - y)
                < math.hypot(G.nodes[v]["x"] - x, G.nodes[v]["y"] - y)
                else v
            )
        else:
            return (
                v
                if math.hypot(G.nodes[v]["x"] - x, G.nodes[v]["y"] - y)
                < math.hypot(G.nodes[u]["x"] - x, G.nodes[u]["y"] - y)
                else u
            )

    # Create new node
    new_node_id = max(G.nodes()) + 1
    G.add_node(new_node_id, x=best_point.x, y=best_point.y)

    # Split the line into two segments
    coords = list(line.coords)
    split_idx = _find_split_index(coords, best_point.x, best_point.y)

    from shapely.geometry import LineString

    coords1 = coords[: split_idx + 1] + [(best_point.x, best_point.y)]
    coords2 = [(best_point.x, best_point.y)] + coords[split_idx + 1 :]

    line1 = LineString(coords1)
    line2 = LineString(coords2)

    # Determine which end is u and which is v
    u_data = G.nodes[u]
    dist_u_to_start = math.hypot(u_data["x"] - coords[0][0], u_data["y"] - coords[0][1])
    dist_u_to_end = math.hypot(u_data["x"] - coords[-1][0], u_data["y"] - coords[-1][1])

    if dist_u_to_start < dist_u_to_end:
        # u is at the start of the line
        seg_u = line1
        seg_v = line2
    else:
        seg_u = line2
        seg_v = line1

    # Remove old edge and add two new ones
    next_edge_id = (
        max(
            (d.get("edge_id", 0) for _, _, d in G.edges(data=True)),
            default=-1,
        )
        + 1
    )

    # Preserve edge attributes (especially region_pair)
    preserved_attrs = {}
    for attr in ("region_pair",):
        if attr in edge_data:
            preserved_attrs[attr] = edge_data[attr]

    G.remove_edge(u, v)
    G.add_edge(
        u,
        new_node_id,
        edge_id=next_edge_id,
        line=seg_u,
        length_px=seg_u.length,
        pixel_count=len(seg_u.coords),
        **preserved_attrs,
    )
    G.add_edge(
        new_node_id,
        v,
        edge_id=next_edge_id + 1,
        line=seg_v,
        length_px=seg_v.length,
        pixel_count=len(seg_v.coords),
        **preserved_attrs,
    )

    return new_node_id


def _find_split_index(
    coords: list[tuple[float, float]],
    x: float,
    y: float,
) -> int:
    """Find the index in coords closest to (x, y)."""
    best_idx = 0
    best_dist = float("inf")
    for i, (cx, cy) in enumerate(coords):
        d = math.hypot(cx - x, cy - y)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx
