"""Main pipeline orchestrator: identify_wing().

Runs all pipeline steps in sequence and returns a WingResult.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
from identify_features.config import PipelineConfig
from identify_features.models.datatypes import VeinIdentification, WingResult
from identify_features.models.geojson_io import (
    _compute_wing_outline,
    load_detection_geojson,
    load_landmarks_geojson,
)
from identify_features.models.intervein_namer import name_intervein_regions
from identify_features.models.intervein_splitter import (
    assign_vein_tissue_polygons,
    split_merged_intervein_polygons,
)
from identify_features.models.landmark_anchor import anchor_landmarks
from identify_features.models.skeleton import build_skeleton_graph
from identify_features.models.vein_tracer import trace_veins_from_landmarks
from identify_features.models.wing_axis import compute_wing_axis
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def _fill_interior_ectopic_veins(
    intervein_polys: list[Polygon | MultiPolygon],
    veins: list[VeinIdentification],
    wing_outline: Optional[Polygon],
    median_vein_width_px: float,
) -> list[Polygon | MultiPolygon]:
    """Union purely-interior EV* centerlines back into the intervein mask.

    An ectopic sprout that lives entirely inside a single intervein region
    can split that region into two polygons in the segmentation output (the
    pixel classifier draws a vein-pixel band where the sprout is). The
    intervein splitter then names the two halves as different regions
    (e.g. half of marginal becomes submarginal on 0009).

    Filter out EVs that look like fragments of a real-but-undetected vein:
    if an EV endpoint is close to a labeled longitudinal/crossvein endpoint
    OR close to the wing margin, the EV probably anchors a real boundary
    and should not be filled in. Only EVs with both endpoints in the
    interior get unioned into adjacent intervein polygons.
    """
    if not veins or not intervein_polys:
        return intervein_polys

    buffer_px = max(median_vein_width_px * 1.5, 5.0)
    margin_buffer = max(median_vein_width_px * 2.0, 5.0)

    real_endpoints: list[Point] = []
    for v in veins:
        if v.centerline is None or v.vein_id.startswith("EV"):
            continue
        coords = list(v.centerline.coords)
        real_endpoints.append(Point(*coords[0]))
        real_endpoints.append(Point(*coords[-1]))

    wing_boundary = wing_outline.boundary if wing_outline is not None else None

    fills: list[Polygon] = []
    for v in veins:
        if v.centerline is None or not v.vein_id.startswith("EV"):
            continue
        coords = list(v.centerline.coords)
        if len(coords) < 2:
            continue
        ev_p1 = Point(*coords[0])
        ev_p2 = Point(*coords[-1])

        anchored_to_real_vein = any(ep.distance(rp) < buffer_px for ep in (ev_p1, ev_p2) for rp in real_endpoints)
        if anchored_to_real_vein:
            logger.info("EV fill: skipping %s (endpoint near real-vein anchor)", v.vein_id)
            continue
        if wing_boundary is not None:
            min_margin = min(ev_p1.distance(wing_boundary), ev_p2.distance(wing_boundary))
            if min_margin < margin_buffer:
                logger.info(
                    "EV fill: skipping %s (endpoint within %.0fpx of wing margin)",
                    v.vein_id,
                    margin_buffer,
                )
                continue
        fills.append(v.centerline.buffer(buffer_px))
        logger.info("EV fill: %s flagged for intervein mask union", v.vein_id)

    if not fills:
        return intervein_polys

    remaining = list(intervein_polys)
    for fill in fills:
        touched: list[Polygon | MultiPolygon] = []
        others: list[Polygon | MultiPolygon] = []
        for p in remaining:
            if p.intersects(fill) or p.distance(fill) < 1.0:
                touched.append(p)
            else:
                others.append(p)
        if not touched:
            remaining = others
            continue
        merged = unary_union(touched + [fill])
        if isinstance(merged, MultiPolygon):
            others.extend(list(merged.geoms))
        else:
            others.append(merged)
        remaining = others
    return remaining


def identify_wing(
    detection_geojson: Path,
    landmarks_geojson: Path,
    image_path: Optional[Path] = None,
    config: Optional[PipelineConfig] = None,
    specimen_id: Optional[str] = None,
) -> WingResult:
    """Run the full identification pipeline on a single specimen.

    Args:
        detection_geojson: Path to detection GeoJSON (vein/intervein polygons).
        landmarks_geojson: Path to landmarks GeoJSON.
        image_path: Path to original image (TIFF/BMP). Required for
            image dimensions; if None, dimensions are estimated from
            polygon bounding boxes.
        config: Pipeline configuration. Uses defaults if None.
        specimen_id: Optional label for this specimen in output.

    Returns:
        WingResult with identified veins, regions, landmarks, and wing outline.
    """
    if config is None:
        config = PipelineConfig()
    if specimen_id is None:
        specimen_id = detection_geojson.stem.replace("_detections", "")

    result = WingResult(specimen_id=specimen_id)
    warnings: list[str] = []

    # Step 1: Parse inputs
    logger.info("Step 1: Parsing inputs for %s", specimen_id)
    vein_polys, intervein_polys = load_detection_geojson(detection_geojson)
    landmarks = load_landmarks_geojson(landmarks_geojson)
    all_polys = vein_polys + intervein_polys
    wing_outline = _compute_wing_outline(all_polys)
    result.wing_outline = wing_outline
    result.landmarks = landmarks

    if image_path is not None:
        from identify_features.utils.psd_loader import imread_any

        img = imread_any(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_shape = (img.shape[0], img.shape[1])
    else:
        # Estimate from polygon bounds
        from shapely.ops import unary_union

        bounds = unary_union(all_polys).bounds  # (minx, miny, maxx, maxy)
        image_shape = (int(bounds[3]) + 100, int(bounds[2]) + 100)
        warnings.append("Image not provided; dimensions estimated from polygons")

    # Step 2: Build skeleton graph
    logger.info("Step 2: Building skeleton graph")
    skel = build_skeleton_graph(vein_polys, image_shape, config)

    # Step 3: Anchor landmarks
    logger.info("Step 3: Anchoring landmarks")
    anchor_landmarks(skel, landmarks, config)

    # Step 4: Compute wing axis
    wing_axis = compute_wing_axis(landmarks)

    # Step 5: Trace veins
    logger.info("Step 5: Tracing veins")
    veins = trace_veins_from_landmarks(skel, landmarks, wing_outline, config, wing_axis=wing_axis)

    # Step 6: Intervein Labeling — three passes (see docs/pipeline_reference.md §6)
    # Execution order is tissue first so the splitter's barrier mask can reuse vein tissue geometry.

    # Step 6.3: Vein Tissue Polygon Assignment
    # Always runs (cheap, and downstream rendering / GeoJSON export need tissue polygons).
    assign_vein_tissue_polygons(veins, skel.median_vein_width_px, config, wing_outline)

    if config.skip_intervein_regions:
        logger.info("Step 6.1/6.2 skipped (skip_intervein_regions=True)")
        regions: list = []
    else:
        # Step 6.3 follow-up: Re-merge regions split by purely-interior ectopic veins
        intervein_polys = _fill_interior_ectopic_veins(intervein_polys, veins, wing_outline, skel.median_vein_width_px)

        # Step 6.1: Intervein Polygon Splitting
        logger.info("Step 6.1: Splitting merged intervein polygons")
        intervein_polys = split_merged_intervein_polygons(
            intervein_polys,
            veins,
            wing_outline,
            image_shape,
            skel.median_vein_width_px,
            config,
        )

        # Step 6.2: Intervein Region Naming
        logger.info("Step 6.2: Naming intervein regions")
        regions = name_intervein_regions(
            intervein_polys,
            veins,
            landmarks,
            config,
            skel.median_vein_width_px,
            wing_outline,
            wing_axis,
        )

    result.veins = veins
    result.intervein_regions = regions
    result.warnings = warnings

    logger.info(
        "Done: %s — %d veins, %d regions",
        specimen_id,
        sum(1 for v in veins if v.centerline is not None),
        len(regions),
    )
    return result
