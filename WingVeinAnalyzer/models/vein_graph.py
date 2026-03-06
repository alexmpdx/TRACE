"""Graph construction from intervein polygon boundaries or vein LineStrings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from WingVeinAnalyzer.models.vein_map import (
    um_to_px,
    MAX_GAP_UM,
    SNAP_RADIUS_UM,
    MIN_SEGMENT_LENGTH_UM,
    GRAPH_SNAP_VEINS_UM,
    SIMPLIFY_UM,
)


@dataclass
class VeinNode:
    """A junction or endpoint in the vein graph."""

    node_id: int
    x: float
    y: float
    degree: int = 0


@dataclass
class VeinEdge:
    """A vein segment between two nodes."""

    edge_id: int
    src: int
    dst: int
    line: LineString
    length_px: float
    poly_pair: Optional[tuple[int, int]] = None


def build_graph_from_polygons(
    polygons: list[Polygon],
    max_gap: float | None = None,
    num_samples: int = 800,
) -> tuple[nx.Graph, list[VeinEdge]]:
    """Build a vein graph by extracting midlines between adjacent polygon pairs."""
    if max_gap is None:
        max_gap = um_to_px(MAX_GAP_UM)
    n = len(polygons)
    edges: list[VeinEdge] = []
    all_midlines: list[tuple[int, int, LineString]] = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = polygons[i].distance(polygons[j])
            if dist > max_gap:
                continue
            midline = _extract_midline(
                polygons[i], polygons[j], max_gap=max_gap, num_samples=num_samples
            )
            if midline is not None and midline.length > um_to_px(MIN_SEGMENT_LENGTH_UM):
                all_midlines.append((i, j, midline))

    graph = nx.Graph()
    node_coords: list[tuple[float, float]] = []
    node_map: dict[tuple[float, float], int] = {}
    snap_tol = um_to_px(SNAP_RADIUS_UM)

    def _get_or_create_node(x: float, y: float) -> int:
        for (nx_, ny_), nid in node_map.items():
            if (nx_ - x) ** 2 + (ny_ - y) ** 2 < snap_tol**2:
                return nid
        nid = len(node_coords)
        node_coords.append((x, y))
        node_map[(x, y)] = nid
        graph.add_node(nid, x=x, y=y, degree=0)
        return nid

    for idx, (pi, pj, midline) in enumerate(all_midlines):
        coords = list(midline.coords)
        if len(coords) < 2:
            continue
        src_xy = coords[0]
        dst_xy = coords[-1]
        src = _get_or_create_node(src_xy[0], src_xy[1])
        dst = _get_or_create_node(dst_xy[0], dst_xy[1])

        edge = VeinEdge(
            edge_id=idx,
            src=src,
            dst=dst,
            line=midline,
            length_px=midline.length,
            poly_pair=(pi, pj),
        )
        edges.append(edge)
        graph.add_edge(
            src,
            dst,
            edge_id=idx,
            length_px=midline.length,
            line=midline,
            poly_pair=(pi, pj),
        )

    for nid in graph.nodes:
        graph.nodes[nid]["degree"] = graph.degree(nid)

    return graph, edges


def build_graph_from_veins(
    veins: list,
    snap_tolerance: float | None = None,
) -> tuple[nx.Graph, dict[int, VeinNode]]:
    """Build a vein graph from pre-traced vein LineStrings."""
    if snap_tolerance is None:
        snap_tolerance = um_to_px(GRAPH_SNAP_VEINS_UM)
    graph = nx.Graph()
    nodes: dict[int, VeinNode] = {}
    junction_points: list[tuple[float, float]] = []
    node_counter = 0

    def _find_or_create_node(x: float, y: float) -> int:
        nonlocal node_counter
        for nid, node in nodes.items():
            if (node.x - x) ** 2 + (node.y - y) ** 2 < snap_tolerance**2:
                return nid
        nid = node_counter
        node_counter += 1
        nodes[nid] = VeinNode(node_id=nid, x=x, y=y)
        graph.add_node(nid, x=x, y=y, degree=0)
        return nid

    # Find all intersection points between vein pairs
    for i, vi in enumerate(veins):
        for j, vj in enumerate(veins):
            if j <= i:
                continue
            intersection = vi.line.intersection(vj.line)
            if intersection.is_empty:
                for ep in [Point(vi.line.coords[0]), Point(vi.line.coords[-1])]:
                    nearest = vj.line.interpolate(vj.line.project(ep))
                    if ep.distance(nearest) < snap_tolerance:
                        mid = ((ep.x + nearest.x) / 2, (ep.y + nearest.y) / 2)
                        junction_points.append(mid)
                for ep in [Point(vj.line.coords[0]), Point(vj.line.coords[-1])]:
                    nearest = vi.line.interpolate(vi.line.project(ep))
                    if ep.distance(nearest) < snap_tolerance:
                        mid = ((ep.x + nearest.x) / 2, (ep.y + nearest.y) / 2)
                        junction_points.append(mid)
            elif intersection.geom_type == "Point":
                junction_points.append((intersection.x, intersection.y))
            elif intersection.geom_type == "MultiPoint":
                for pt in intersection.geoms:
                    junction_points.append((pt.x, pt.y))

    for x, y in junction_points:
        _find_or_create_node(x, y)

    for v in veins:
        coords = list(v.line.coords)
        _find_or_create_node(coords[0][0], coords[0][1])
        _find_or_create_node(coords[-1][0], coords[-1][1])

    edge_counter = 0
    for v in veins:
        coords = list(v.line.coords)
        start_node = _find_or_create_node(coords[0][0], coords[0][1])
        end_node = _find_or_create_node(coords[-1][0], coords[-1][1])
        graph.add_edge(
            start_node,
            end_node,
            edge_id=edge_counter,
            length_px=v.length_px,
            line=v.line,
            vein_feature_id=v.feature_id,
        )
        edge_counter += 1

    for nid in graph.nodes:
        graph.nodes[nid]["degree"] = graph.degree(nid)
        if nid in nodes:
            nodes[nid].degree = graph.degree(nid)

    return graph, nodes


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_midline(
    poly_a: Polygon,
    poly_b: Polygon,
    max_gap: float | None = None,
    num_samples: int = 800,
) -> Optional[LineString]:
    """Extract the midline between facing boundary segments of two polygons."""
    if max_gap is None:
        max_gap = um_to_px(MAX_GAP_UM)
    ring_a = poly_a.exterior
    ring_b = poly_b.exterior
    min_dist = poly_a.distance(poly_b)
    threshold = min(max_gap, min_dist * 3 + 10)

    midpoints: list[tuple[float, float]] = []
    for t in np.linspace(0, 1, num_samples, endpoint=False):
        pa = ring_a.interpolate(t, normalized=True)
        proj = ring_b.project(pa)
        pb = ring_b.interpolate(proj)
        d = pa.distance(pb)
        if d < threshold:
            midpoints.append(((pa.x + pb.x) / 2, (pa.y + pb.y) / 2))

    if len(midpoints) < 2:
        return None

    filtered = [midpoints[0]]
    for mp in midpoints[1:]:
        dx = mp[0] - filtered[-1][0]
        dy = mp[1] - filtered[-1][1]
        if dx * dx + dy * dy > 1.0:
            filtered.append(mp)

    if len(filtered) < 2:
        return None

    line = LineString(filtered)
    return line.simplify(um_to_px(SIMPLIFY_UM))
