"""Detect the costal vein using a margin band along the wing outline.

Pipeline:
1. Compute margin band: wing-outline pixels within 1 median vein width of
   the wing edge, trimmed at the subcostal-break → alula-notch line to
   exclude the hinge region.
2. For each graph edge, compute the fraction of its length that falls
   within the margin band.
3. Edges with ≥ costa_min_in_band_fraction in the band are costa candidates.
4. Edges where more than 50% extends beyond the band are rejected (interior
   veins that happen to touch the margin).
"""

from __future__ import annotations

import logging

import numpy as np
from identify_features.config import PipelineConfig
from identify_features.models.datatypes import Landmark, SkeletonGraph
from identify_features.utils.image_utils import rasterize_polygons
from scipy import ndimage
from shapely.geometry import LineString, Polygon
from shapely.ops import split

logger = logging.getLogger(__name__)


def detect_costa_edges(
    skel_graph: SkeletonGraph,
    landmarks: dict[str, Landmark],
    wing_outline: Polygon | None,
    config: PipelineConfig | None = None,
) -> tuple[set[tuple[int, int]], np.ndarray]:
    """Identify graph edges that are part of the costal vein.

    Args:
        skel_graph: Skeleton graph (after landmark anchoring).
        landmarks: Anchored landmarks dict.
        wing_outline: Wing outline polygon (union of vein + intervein).
        config: Pipeline configuration.

    Returns:
        (costa_edge_keys, margin_band) where costa_edge_keys is a set of
        (min_node, max_node) edge keys, and margin_band is the boolean
        mask used for detection.
    """
    if config is None:
        config = PipelineConfig()

    G = skel_graph.graph
    image_shape = skel_graph.image_shape
    median_w = skel_graph.median_vein_width_px

    if wing_outline is None or median_w <= 0:
        logger.warning("Cannot detect costa: missing wing outline or vein width")
        return set(), np.zeros(image_shape, dtype=bool)

    # Step 1: Build margin band
    margin_band = _build_margin_band(wing_outline, image_shape, median_w, landmarks)
    logger.info(
        "Margin band: %d pixels (median vein width=%.1fpx)",
        np.count_nonzero(margin_band),
        median_w,
    )

    # Step 2: Score each edge by fraction in the band
    min_fraction = config.costa_min_in_band_fraction
    costa_keys: set[tuple[int, int]] = set()

    # SC node and proximal direction for rejecting L1 edges
    sc = landmarks.get("subcostal break")
    an = landmarks.get("alula notch")
    sc_node = sc.snapped_node if sc is not None else None
    # Proximal direction = SC → AN
    prox_dx = prox_dy = 0.0
    if sc is not None and an is not None:
        prox_dx = an.x - sc.x
        prox_dy = an.y - sc.y
        pmag = max((prox_dx**2 + prox_dy**2) ** 0.5, 1e-6)
        prox_dx /= pmag
        prox_dy /= pmag

    for u, v, data in G.edges(data=True):
        line = data.get("line")
        if line is None or line.is_empty:
            continue

        fraction = _edge_in_band_fraction(line, margin_band)

        if fraction >= min_fraction:
            # Reject edges departing SC in the proximal direction (toward AN).
            # Costa runs distal from SC; anything proximal is L1.
            if sc_node is not None and (prox_dx != 0 or prox_dy != 0):
                if u == sc_node or v == sc_node:
                    # Get the other endpoint
                    other = v if u == sc_node else u
                    other_data = G.nodes[other]
                    sc_data = G.nodes[sc_node]
                    edge_dx = other_data["x"] - sc_data["x"]
                    edge_dy = other_data["y"] - sc_data["y"]
                    emag = max((edge_dx**2 + edge_dy**2) ** 0.5, 1e-6)
                    edge_dx /= emag
                    edge_dy /= emag
                    # Dot product with proximal direction: >0 means proximal
                    dot = edge_dx * prox_dx + edge_dy * prox_dy
                    if dot > 0:
                        logger.info(
                            "Costa reject %d↔%d: departs SC proximally (dot=%.2f)",
                            u,
                            v,
                            dot,
                        )
                        continue

            key = (min(u, v), max(u, v))
            costa_keys.add(key)
            logger.info(
                "Costa edge %d↔%d: %.0fpx, %.1f%% in band",
                u,
                v,
                data.get("length_px", 0),
                fraction * 100,
            )

    logger.info("Detected %d costa edges", len(costa_keys))
    return costa_keys, margin_band


def _build_margin_band(
    wing_outline: Polygon,
    image_shape: tuple[int, int],
    median_vein_width: float,
    landmarks: dict[str, Landmark],
) -> np.ndarray:
    """Build the margin band: pixels within 2x median vein width of the wing edge.

    Trimmed at the subcostal-break → alula-notch line to exclude the hinge.
    """
    # Rasterize wing outline → distance from wing edge inward
    wing_mask = rasterize_polygons([wing_outline], image_shape)
    wing_edge_dist = ndimage.distance_transform_edt(wing_mask > 0)

    # Band = inside wing AND within 2x median vein width of edge
    margin_band = (wing_mask > 0) & (wing_edge_dist <= median_vein_width * 2)

    # Trim hinge: keep only band in the largest piece of the SC-AN split
    sc = landmarks.get("subcostal break")
    an = landmarks.get("alula notch")

    if sc is not None and an is not None:
        _trim_hinge(margin_band, wing_outline, sc, an, image_shape)

    # Cut band at subcostal break — costa ends at SC, everything
    # proximal on the anterior margin is L1
    if sc is not None:
        _cut_at_subcostal_break(margin_band, wing_outline, sc, image_shape, landmarks)

    return margin_band


def _trim_hinge(
    margin_band: np.ndarray,
    wing_outline: Polygon,
    sc: Landmark,
    an: Landmark,
    image_shape: tuple[int, int],
) -> None:
    """Remove the hinge portion of the margin band (in-place).

    Draws a line from subcostal break to alula notch, splits the wing
    outline, and keeps band pixels only in the largest piece (distal wing).
    """
    dx = an.x - sc.x
    dy = an.y - sc.y
    mag = max((dx**2 + dy**2) ** 0.5, 1e-6)

    # Extend line well past the wing to ensure clean split
    ext = 500
    cut_line = LineString(
        [
            (sc.x - ext * dx / mag, sc.y - ext * dy / mag),
            (an.x + ext * dx / mag, an.y + ext * dy / mag),
        ]
    )

    try:
        pieces = split(wing_outline, cut_line)
        if len(pieces.geoms) >= 2:
            largest = max(pieces.geoms, key=lambda g: g.area)
            distal_mask = rasterize_polygons([largest], image_shape)
            # Keep band only inside the distal (largest) piece
            margin_band[distal_mask == 0] = False
            trimmed = np.count_nonzero(distal_mask == 0)
            logger.debug("Trimmed hinge: removed %d band pixels", trimmed)
    except Exception as e:
        logger.warning("Failed to trim hinge at SC-AN line: %s", e)


def _cut_at_subcostal_break(
    margin_band: np.ndarray,
    wing_outline: "Polygon",
    sc: "Landmark",
    image_shape: tuple[int, int],
    landmarks: dict | None = None,
) -> None:
    """Cut the costa band at the subcostal break (in-place).

    Removes band pixels that are:
    1. Proximal to SC (along the wing margin tangent)
    2. On the anterior side of the SC→AN line
    This targets only the L1-territory band on the anterior margin
    without touching the posterior margin or the distal costa.
    """
    from shapely.geometry import Point

    an = landmarks.get("alula notch") if landmarks else None
    if an is None:
        return

    # Project SC onto wing outline to find local margin direction
    sc_point = Point(sc.x, sc.y)
    proj_dist = wing_outline.exterior.project(sc_point)
    outline_length = wing_outline.exterior.length

    # Sample two points near SC on the outline to get tangent direction
    sample = 50.0
    d1 = max(0, proj_dist - sample)
    d2 = min(outline_length, proj_dist + sample)
    pt1 = wing_outline.exterior.interpolate(d1)
    pt2 = wing_outline.exterior.interpolate(d2)

    # Tangent direction along the margin
    tx = pt2.x - pt1.x
    ty = pt2.y - pt1.y
    tmag = max((tx**2 + ty**2) ** 0.5, 1e-6)
    tx /= tmag
    ty /= tmag

    # SC→AN direction (for anterior/posterior check)
    ax = an.x - sc.x
    ay = an.y - sc.y

    rows, cols = np.where(margin_band)
    if len(rows) == 0:
        return

    dx = cols - sc.x
    dy = rows - sc.y

    # Condition 1: proximal to SC (dot with tangent > 0 means
    # same direction as pt1→pt2 which goes proximal on this specimen;
    # use < 0 for the opposite. Need to determine which sign is proximal.
    # Proximal = toward the hinge = toward AN from SC.
    # dot(pixel-SC, tangent) tells us position along the tangent.
    # We want pixels on the side AWAY from DTip (distal).
    # Simpler: proximal = dot(pixel-SC, SC→AN direction) > 0
    dot_proximal = dx * ax + dy * ay
    is_proximal = dot_proximal > 0

    # Condition 2: anterior side of SC→AN line
    # Cross product SC→AN × SC→pixel: positive = one side, negative = other
    # The anterior side has smaller Y values (top of image)
    cross = ax * dy - ay * dx
    # Determine which sign is anterior by checking the wing centroid
    centroid = wing_outline.centroid
    cx_cross = ax * (centroid.y - sc.y) - ay * (centroid.x - sc.x)
    # The centroid is in the interior (posterior-ish). Anterior is the opposite sign.
    is_anterior = cross * cx_cross < 0  # opposite side from centroid

    to_remove = is_proximal & is_anterior
    before = np.count_nonzero(margin_band)
    margin_band[rows[to_remove], cols[to_remove]] = False
    after = np.count_nonzero(margin_band)
    logger.debug(
        "Cut band at SC: removed %d pixels (%.0f%% of band)",
        before - after,
        100 * (before - after) / max(before, 1),
    )


def _edge_in_band_fraction(
    line: LineString,
    margin_band: np.ndarray,
) -> float:
    """Compute the fraction of an edge's length that falls within the margin band."""
    length = line.length
    if length < 1:
        return 0.0

    n_samples = max(10, int(length / 5))
    in_band = 0
    h, w = margin_band.shape

    for i in range(n_samples):
        pt = line.interpolate((i + 0.5) / n_samples, normalized=True)
        r = min(int(round(pt.y)), h - 1)
        c = min(int(round(pt.x)), w - 1)
        if r >= 0 and c >= 0 and margin_band[r, c]:
            in_band += 1

    return in_band / n_samples
