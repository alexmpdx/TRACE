"""NetworkX graph helper functions for skeleton graph operations."""

from __future__ import annotations

import math
from typing import Optional

import networkx as nx
from identify_features.utils.geometry_utils import (
    angle_between_vectors,
    direction_toward,
    line_direction,
    line_end_direction,
)
from shapely.geometry import LineString, Point


def nearest_node(
    G: nx.Graph,
    x: float,
    y: float,
    max_dist: float = float("inf"),
    prefer_degree: Optional[int] = None,
    prefer_degree_radius: float = 0.0,
) -> Optional[int]:
    """Find the graph node nearest to (x, y).

    Args:
        G: Skeleton graph with node attributes x, y.
        x, y: Target position in pixel coordinates.
        max_dist: Maximum distance to consider.
        prefer_degree: If set, prefer nodes with this degree or higher.
        prefer_degree_radius: Search radius within which to prefer
            high-degree nodes over the absolute nearest.

    Returns:
        Node ID, or None if no node within max_dist.
    """
    best_node = None
    best_dist = max_dist
    best_pref_node = None
    best_pref_dist = float("inf")

    for node, data in G.nodes(data=True):
        nx_, ny = data["x"], data["y"]
        dist = math.hypot(nx_ - x, ny - y)

        if dist < best_dist:
            best_dist = dist
            best_node = node

        if (
            prefer_degree is not None
            and dist < prefer_degree_radius
            and G.degree(node) >= prefer_degree
            and dist < best_pref_dist
        ):
            best_pref_dist = dist
            best_pref_node = node

    # If we found a preferred-degree node within the preference radius,
    # use it instead of the absolute nearest
    if best_pref_node is not None:
        return best_pref_node
    return best_node


def edge_departure_direction(
    G: nx.Graph,
    node: int,
    neighbor: int,
    sample_px: float = 80.0,
) -> tuple[float, float]:
    """Get the direction vector of an edge as it departs from `node`.

    Returns a unit vector (dx, dy) pointing away from `node` along the edge.
    """
    data = G[node][neighbor]
    line = data["line"]

    # Check which end of the line is closer to node
    node_data = G.nodes[node]
    node_x, node_y = node_data["x"], node_data["y"]

    start = line.coords[0]
    end = line.coords[-1]
    d_start = math.hypot(start[0] - node_x, start[1] - node_y)
    d_end = math.hypot(end[0] - node_x, end[1] - node_y)

    if d_start <= d_end:
        # Line starts at this node — use forward direction
        return line_direction(line, sample_px)
    else:
        # Line ends at this node — use reverse direction
        reversed_line = LineString(list(line.coords)[::-1])
        return line_direction(reversed_line, sample_px)


def best_continuation_edge(
    G: nx.Graph,
    node: int,
    incoming_direction: tuple[float, float],
    exclude_neighbors: set | None = None,
    max_angle: float = 90.0,
) -> Optional[int]:
    """Find the edge at `node` that best continues the incoming direction.

    Uses tangent continuity: looks for the edge whose departure direction
    is most collinear with the incoming direction (angle closest to 180°).

    Args:
        G: Skeleton graph.
        node: Current node.
        incoming_direction: Unit vector of the incoming direction.
        exclude_neighbors: Neighbors to skip (e.g., the one we came from).
        max_angle: Maximum deflection from straight-ahead (degrees).
            An edge must have a continuation angle of (180 ± max_angle).

    Returns:
        Neighbor node ID, or None if no suitable continuation.
    """
    if exclude_neighbors is None:
        exclude_neighbors = set()

    # Reverse incoming to get the "straight ahead" direction
    straight_ahead = (-incoming_direction[0], -incoming_direction[1])

    best = None
    best_angle = float("inf")

    for neighbor in G.neighbors(node):
        if neighbor in exclude_neighbors:
            continue

        departure = edge_departure_direction(G, node, neighbor)
        angle = angle_between_vectors(departure, straight_ahead)

        if angle < best_angle and angle <= max_angle:
            best_angle = angle
            best = neighbor

    return best


def edge_line_from_node(
    G: nx.Graph,
    node: int,
    neighbor: int,
) -> LineString:
    """Get the edge LineString oriented to start from `node`."""
    data = G[node][neighbor]
    line = data["line"]

    node_data = G.nodes[node]
    node_x, node_y = node_data["x"], node_data["y"]

    start = line.coords[0]
    d_start = math.hypot(start[0] - node_x, start[1] - node_y)
    end = line.coords[-1]
    d_end = math.hypot(end[0] - node_x, end[1] - node_y)

    if d_start <= d_end:
        return line
    else:
        return LineString(list(line.coords)[::-1])
