"""Split merged intervein polygons via h-maxima seed detection + watershed.

The pixel classifier occasionally fuses adjacent intervein regions where a
crossvein is short, interrupted, or missed. This module re-splits such
polygons by (1) detecting significant peaks in each polygon's distance
transform via h-maxima filtering — peaks that rise at least ``h`` pixels
above the nearest saddle point become seeds, (2) competitively dilating
every seed outward until it meets a barrier (canonical vein centerlines
excluding L6/EVs, or the wing outline), and (3) reseeding any large
originally-intervein area that ended up unclaimed so small-but-real
regions like the costal cell aren't lost.

The h-maxima approach is adaptive: a uniform thin strip has no deep
saddle, so it produces one seed regardless of width. A fused polygon
with a fat body and a thin bridge has a deep saddle between two peaks,
producing two seeds — exactly where the split should occur.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from identify_features.config import PipelineConfig
from identify_features.models.datatypes import VeinIdentification
from identify_features.utils.image_utils import rasterize_polygons
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import MultiPolygon, Polygon
from skimage.morphology import h_maxima
from skimage.segmentation import find_boundaries, watershed

logger = logging.getLogger(__name__)


def split_merged_intervein_polygons(
    intervein_polys: list[Polygon | MultiPolygon],
    veins: list[VeinIdentification],
    wing_outline: Polygon,
    image_shape: tuple[int, int],
    median_vein_width_px: float,
    config: PipelineConfig,
    debug_out: Optional[Path] = None,
    debug_base_image: Optional[np.ndarray] = None,
) -> list[Polygon]:
    """Split classifier-merged intervein polygons via erode-then-dilate.

    Args:
        intervein_polys: Raw intervein polygons from the detection GeoJSON.
        veins: Identified veins from the tracer. EV* and L6 are excluded
            from the barrier mask per the current spec.
        wing_outline: Wing outline polygon. Acts as the outer boundary —
            dilation cannot leak outside the wing.
        image_shape: (height, width) of the working raster.
        median_vein_width_px: Used to buffer barrier centerlines to actual
            tissue width.
        config: Pipeline configuration; reads ``intervein_split_h_vw``
            and ``intervein_split_reseed_min_area_um2``.
        debug_out: If set, write a diagnostic PNG showing the barrier mask
            and watershed label boundaries. Used to spot barrier leaks.
        debug_base_image: BGR base image to overlay the diagnostic on. If
            None, uses a gray background.

    Returns:
        List of bare ``Polygon`` objects (no MultiPolygons). May contain
        more or fewer entries than the input depending on how many split
        or vanished.
    """
    if wing_outline is None:
        raise ValueError("split_merged_intervein_polygons requires a wing_outline")

    H, W = image_shape
    h_maxima_h = max(1, round(median_vein_width_px * config.intervein_split_h_vw))
    if config.um_per_px is not None and config.um_per_px > 0:
        reseed_min_area_px = max(1, round(config.intervein_split_reseed_min_area_um2 / (config.um_per_px**2)))
    else:
        reseed_min_area_px = max(1, round(config.intervein_split_reseed_min_area_um2))
    vein_barrier_px = max(1, round(median_vein_width_px * config.intervein_split_vein_barrier_vw))
    wing_buffer_px = max(0, round(median_vein_width_px * config.intervein_split_wing_buffer_vw))

    # --- Step 1: build the barrier mask -------------------------------------
    wing_mask_raw = rasterize_polygons([wing_outline], (H, W)) > 0

    # Inset the wing mask by wing_buffer_px so the wing edge itself is a
    # finite-width barrier, not a zero-thickness cutoff. Prevents label
    # bleed into the narrow strip between a near-margin vein (e.g. costa)
    # and the wing outline.
    if wing_buffer_px > 0:
        wing_mask = distance_transform_edt(wing_mask_raw) > wing_buffer_px
    else:
        wing_mask = wing_mask_raw

    barrier_centerlines = np.zeros((H, W), dtype=np.uint8)
    for v in veins:
        if v.centerline is None:
            continue
        if v.vein_id.startswith("EV") or v.vein_id == "L6":
            continue
        pts = np.array(v.centerline.coords, dtype=np.int32)
        cv2.polylines(barrier_centerlines, [pts], False, 1, 1)

    # Buffer centerlines to tissue width via inverse EDT.
    vein_barrier = distance_transform_edt(barrier_centerlines == 0) <= vein_barrier_px
    interior_mask = wing_mask & ~vein_barrier

    # --- Step 2: build seeds via h-maxima peak detection ---------------------
    # For each polygon, compute the EDT (inscribed-radius landscape) and find
    # peaks that rise at least h pixels above the nearest saddle.  Each such
    # peak becomes one seed.  Uniform thin strips have no deep saddle → 1 seed.
    # Fused polygons with a fat body and a thin bridge have a deep saddle → 2+
    # seeds at exactly the right locations.
    seeds = np.zeros((H, W), dtype=np.int32)
    next_label = 1
    lost_poly_masks: list[np.ndarray] = []  # masks of polys with no peaks

    for poly in intervein_polys:
        poly_mask = rasterize_polygons([poly], (H, W)) > 0
        if not poly_mask.any():
            continue
        edt = distance_transform_edt(poly_mask)
        # Smooth the EDT to eliminate plateau ripples that cause spurious
        # 1-pixel peaks on flat ridges.  Sigma = median vein width keeps
        # real bottlenecks intact while merging near-identical peak pixels.
        edt_smooth = gaussian_filter(edt, sigma=median_vein_width_px)
        peaks = h_maxima(edt_smooth, h_maxima_h)
        if not peaks.any():
            lost_poly_masks.append(poly_mask)
            continue
        num, comp_labels = cv2.connectedComponents((peaks > 0).astype(np.uint8))
        for comp in range(1, num):
            seeds[comp_labels == comp] = next_label
            next_label += 1

    # --- Step 3: competitive dilation via watershed -------------------------
    # Surface = -EDT(interior) so labels flood from markers outward and meet
    # along midlines of the legal territory.
    surface = -distance_transform_edt(interior_mask)
    labels_out = watershed(surface, markers=seeds, mask=interior_mask)

    # --- Step 4: reseed large "lost" polygons and re-run --------------------
    # A polygon is "lost" if h-maxima found no significant peaks — its EDT
    # landscape is flat (max inscribed radius < h). Its territory is now
    # claimed by neighboring labels. If the original footprint was big
    # enough to be a real region (>= reseed threshold), drop a single seed
    # in its interior so it can reclaim that territory.
    reseed_count = 0
    for poly_mask in lost_poly_masks:
        area = int(poly_mask.sum())
        if area < reseed_min_area_px:
            continue
        # Seed from an inner point of the original polygon, constrained to
        # the legal interior (must not land on a barrier or outside the wing).
        seed_candidate = poly_mask & interior_mask
        if not seed_candidate.any():
            continue
        ys, xs = np.where(seed_candidate)
        cy, cx = int(np.median(ys)), int(np.median(xs))
        if not seed_candidate[cy, cx]:
            cy, cx = int(ys[0]), int(xs[0])
        seeds[cy, cx] = next_label
        next_label += 1
        reseed_count += 1
        logger.info(
            "Intervein splitter: reseeded lost polygon (%d px²) at (%d, %d)",
            area,
            cx,
            cy,
        )
    if reseed_count:
        labels_out = watershed(surface, markers=seeds, mask=interior_mask)

    # --- Step 5: raster → polygons ------------------------------------------
    out: list[Polygon] = []
    for label in range(1, next_label):
        mask = (labels_out == label).astype(np.uint8)
        if not mask.any():
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue
        poly = Polygon(contour.reshape(-1, 2))
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            # buffer(0) can hand back a MultiPolygon; take the largest piece.
            poly = max(poly.geoms, key=lambda g: g.area)
        out.append(poly)

    logger.info(
        "Intervein splitter: %d input polys → %d output polys "
        "(h=%dpx [%.1f×vw], vein_buffer=%dpx, wing_inset=%dpx, reseed_min=%dpx², reseeded=%d)",
        len(intervein_polys),
        len(out),
        h_maxima_h,
        config.intervein_split_h_vw,
        vein_barrier_px,
        wing_buffer_px,
        reseed_min_area_px,
        reseed_count,
    )

    if debug_out is not None:
        _write_debug_overlay(
            debug_out,
            debug_base_image,
            image_shape=(H, W),
            wing_mask=wing_mask,
            vein_barrier=vein_barrier,
            seeds=seeds,
            labels_out=labels_out,
            veins=veins,
        )

    return out


def _write_debug_overlay(
    out_path: Path,
    base_image: Optional[np.ndarray],
    image_shape: tuple[int, int],
    wing_mask: np.ndarray,
    vein_barrier: np.ndarray,
    seeds: np.ndarray,
    labels_out: np.ndarray,
    veins: list[VeinIdentification],
) -> None:
    """Render a diagnostic PNG showing barrier mask, seeds, and watershed boundaries."""
    H, W = image_shape
    if base_image is not None and base_image.shape[:2] == (H, W):
        canvas = base_image.copy()
    else:
        canvas = np.full((H, W, 3), 180, dtype=np.uint8)

    # Layer 1: red tint wherever watershed cannot go (barrier or outside wing)
    blocked = vein_barrier | ~wing_mask
    red = np.zeros_like(canvas)
    red[..., 2] = 255
    alpha = 0.35
    canvas = np.where(
        blocked[..., None],
        (canvas.astype(np.float32) * (1 - alpha) + red.astype(np.float32) * alpha).astype(np.uint8),
        canvas,
    )

    # Layer 2: green tint for erosion seed pixels (surviving eroded regions)
    seed_mask = seeds > 0
    green = np.zeros_like(canvas)
    green[..., 1] = 255
    canvas = np.where(
        seed_mask[..., None],
        (canvas.astype(np.float32) * 0.5 + green.astype(np.float32) * 0.5).astype(np.uint8),
        canvas,
    )

    # Layer 3: cyan outlines of watershed labels (boundary between any two labels)
    boundaries = find_boundaries(labels_out, mode="thick")
    canvas[boundaries] = (255, 255, 0)  # BGR cyan

    # Layer 4: thin yellow unbuffered canonical vein centerlines so we can see
    # the tracer's actual geometry vs. the buffered barrier footprint
    for v in veins:
        if v.centerline is None:
            continue
        if v.vein_id.startswith("EV") or v.vein_id == "L6":
            continue
        pts = np.array(v.centerline.coords, dtype=np.int32)
        cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)  # BGR yellow
        if len(pts) > 0:
            # Endpoint markers so we can see where a short vein actually stops
            cv2.circle(canvas, tuple(pts[0]), 18, (0, 255, 255), 2)
            cv2.circle(canvas, tuple(pts[-1]), 18, (0, 255, 255), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    logger.info("Intervein splitter debug overlay → %s", out_path)


def assign_vein_tissue_polygons(
    veins: list[VeinIdentification],
    median_vein_width_px: float,
    config: PipelineConfig,
    wing_outline: Optional[Polygon] = None,
) -> None:
    """Populate tissue_polygon on each vein by buffering its centerline.

    Uses the same buffer radius as the intervein splitter barrier mask
    (``median_vein_width_px * config.intervein_split_vein_barrier_vw``),
    clipped to the wing outline.
    """
    buffer_px = max(1, round(median_vein_width_px * config.intervein_split_vein_barrier_vw))
    for v in veins:
        if v.centerline is None:
            continue
        tissue = v.centerline.buffer(buffer_px)
        if wing_outline is not None:
            tissue = tissue.intersection(wing_outline)
        if tissue.is_empty:
            continue
        if isinstance(tissue, MultiPolygon):
            tissue = max(tissue.geoms, key=lambda g: g.area)
        v.tissue_polygon = tissue
    logger.info(
        "Assigned tissue polygons to %d/%d veins (buffer=%dpx)",
        sum(1 for v in veins if v.tissue_polygon is not None),
        len(veins),
        buffer_px,
    )
