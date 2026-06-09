"""Renderers for the live preview's intermediate view modes.

Two products of the pipeline that the final-output overlay doesn't show:

* :func:`render_skeleton` — the wing graph at the end of skeletonization
  (Tier A). Edges as polylines, nodes colored by degree. Needs no tracing, so
  a skeleton-only view lets Wing Graph tuning skip the expensive Tier B.
* :func:`render_traced` — the labeled veins + snapped landmarks at the end of
  vein tracing (Tier B). Reuses the tested vein-centerline drawing from
  ``render_overlay`` (veins only, no key/regions) and adds landmark markers.

Drawing follows the cv2 patterns proven in skeleton.py's ``_DebugDumper`` and
vein_tracer.py's ``_TracerDumper`` debug dumpers.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from identify_features.models.datatypes import Landmark
from identify_features.views.overlay import render_overlay


def _stroke_scale(img: np.ndarray) -> float:
    """Match render_overlay's size-proportional stroke scaling."""
    h, w = img.shape[:2]
    return max(1.0, min(h, w) / 1800.0)


def render_skeleton(base_image: np.ndarray, skel) -> np.ndarray:
    """Draw the skeleton graph (edges + degree-colored nodes) over the image.

    ``skel`` is a SkeletonGraph; nodes carry x/y attrs (or are (x, y) tuples)
    and edges carry a shapely ``line``. Returns a fresh BGR image.
    """
    img = base_image.copy()
    if skel is None or getattr(skel, "graph", None) is None:
        return img
    graph = skel.graph
    scale = _stroke_scale(img)
    edge_thick = max(1, int(round(2 * scale)))
    node_r = max(2, int(round(4 * scale)))

    # Edges first, so nodes sit on top.
    for _u, _v, data in graph.edges(data=True):
        line = data.get("line")
        if line is None:
            continue
        pts = np.asarray(line.coords, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (0, 255, 255), edge_thick, cv2.LINE_AA)

    # Nodes: cyan for degree<=2 (path), magenta for junctions (degree>=3).
    for node, nd in graph.nodes(data=True):
        if isinstance(node, tuple) and len(node) >= 2:
            x, y = int(node[0]), int(node[1])
        elif "x" in nd and "y" in nd:
            x, y = int(nd["x"]), int(nd["y"])
        else:
            continue
        color = (0, 128, 255) if graph.degree(node) <= 2 else (255, 80, 255)
        cv2.circle(img, (x, y), node_r, color, -1)

    return img


def _draw_landmarks(
    img: np.ndarray,
    landmarks: dict[str, Landmark],
    skel,
    scale: float,
) -> None:
    """Draw snapped landmark markers + labels in place.

    A landmark's snapped position is the graph node it anchored to
    (``G.nodes[lm.snapped_node]``); ``lm.point`` is the pre-snap prediction.
    Draw the snapped node when available (that's what tracing actually used),
    falling back to the raw point, and tie the two with a thin line when they
    differ so the snap is visible.
    """
    graph = getattr(skel, "graph", None)
    ring_r = max(6, int(round(12 * scale)))
    font_scale = 0.6 * scale
    thick = max(1, int(round(scale)))
    for name, lm in landmarks.items():
        raw = None if lm.point is None else (int(lm.point.x), int(lm.point.y))
        snapped = None
        if graph is not None and lm.snapped_node is not None and lm.snapped_node in graph.nodes:
            nd = graph.nodes[lm.snapped_node]
            if "x" in nd and "y" in nd:
                snapped = (int(nd["x"]), int(nd["y"]))
        anchor = snapped or raw
        if anchor is None:
            continue
        if snapped is not None and raw is not None and snapped != raw:
            cv2.line(img, raw, snapped, (180, 180, 180), thick, cv2.LINE_AA)
        cv2.circle(img, anchor, ring_r, (255, 255, 255), thick + 1, cv2.LINE_AA)
        cv2.putText(
            img, name, (anchor[0] + ring_r + 2, anchor[1] + 6),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick, cv2.LINE_AA,
        )


def render_traced(
    base_image: np.ndarray,
    veins: list,
    landmarks: dict[str, Landmark],
    skel,
    vein_color_overrides: Optional[dict] = None,
    vein_opacity: float = 1.0,
) -> np.ndarray:
    """Draw labeled vein centerlines + snapped landmarks (no regions, no key).

    Vein strokes reuse ``render_overlay`` (veins only) so the colors/labels
    match the final output exactly; the snapped landmarks are layered on top.
    """
    img = render_overlay(
        base_image,
        veins,
        [],
        show_vein_tissue=False,
        show_veins=True,
        show_regions=False,
        vein_color_overrides=vein_color_overrides,
        vein_opacity=vein_opacity,
        show_color_key=False,
    )
    _draw_landmarks(img, landmarks or {}, skel, _stroke_scale(img))
    return img
