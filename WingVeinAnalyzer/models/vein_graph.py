"""Skeletonization and graph construction from binary wing masks."""

from __future__ import annotations

from typing import Optional

import cv2
import networkx as nx
import numpy as np
from skan import Skeleton, summarize
from skimage.morphology import skeletonize


def skeletonize_mask(mask: np.ndarray, spur_threshold: int = 10) -> np.ndarray:
    """Skeletonize a binary mask and prune short spurs."""
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    if spur_threshold > 0 and skeleton.any():
        skeleton = _prune_spurs(skeleton, spur_threshold)
    return skeleton


def build_graph(
    skeleton: np.ndarray,
    mask: Optional[np.ndarray] = None,
    margin_tolerance_px: float = 5.0,
) -> tuple[Optional[Skeleton], nx.Graph]:
    """Convert a skeleton image to a skan Skeleton and networkx graph."""
    if not skeleton.any():
        return None, nx.Graph()

    skan_skeleton = Skeleton(skeleton)
    summary = summarize(separator="-", skel=skan_skeleton)
    graph = nx.Graph()

    margin_contour = _find_wing_margin(mask) if mask is not None else None

    # Collect unique node IDs
    node_ids = set(summary["node-id-src"]).union(summary["node-id-dst"])

    for node_id in node_ids:
        row, col = skan_skeleton.coordinates[node_id]
        on_margin = _is_on_margin(row, col, margin_contour, margin_tolerance_px)
        graph.add_node(
            node_id,
            x=float(col),
            y=float(row),
            degree=int(skan_skeleton.degrees[node_id]),
            on_margin=on_margin,
        )

    for idx, row in summary.iterrows():
        src = int(row["node-id-src"])
        dst = int(row["node-id-dst"])
        graph.add_edge(
            src,
            dst,
            length_px=float(row["branch-distance"]),
            branch_id=int(idx),
        )

    return skan_skeleton, graph


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _prune_spurs(skeleton: np.ndarray, spur_threshold: int) -> np.ndarray:
    """Iteratively remove skeleton branches shorter than *spur_threshold* px."""
    skel = skeleton.copy()
    for _ in range(10):
        if not skel.any():
            break
        tmp_skel = Skeleton(skel)
        summary = summarize(separator="-", skel=tmp_skel)

        # branch-type 0/1 = endpoint-to-endpoint / endpoint-to-junction
        is_spur = summary["branch-type"].isin([0, 1])
        is_short = summary["branch-distance"] < spur_threshold
        to_remove = summary[is_spur & is_short]

        if to_remove.empty:
            break

        for idx in to_remove.index:
            coords = tmp_skel.path_coordinates(idx).astype(int)
            skel[coords[:, 0], coords[:, 1]] = 0

    return skel


def _find_wing_margin(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return the largest external contour of *mask*, or None."""
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _is_on_margin(
    row: float, col: float, contour: Optional[np.ndarray], tolerance: float
) -> bool:
    """Return True if the point is within *tolerance* px of *contour*."""
    if contour is None:
        return False
    dist = cv2.pointPolygonTest(contour, (float(col), float(row)), measureDist=True)
    return abs(dist) <= tolerance
