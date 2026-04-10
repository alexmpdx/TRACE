"""Main pipeline orchestrator: identify_wing().

Runs all pipeline steps in sequence and returns a WingResult.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
from identify_features.config import PipelineConfig
from identify_features.models.datatypes import WingResult
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

logger = logging.getLogger(__name__)


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
        img = cv2.imread(str(image_path))
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
    logger.info("Step 4-5: Tracing veins")
    veins = trace_veins_from_landmarks(skel, landmarks, wing_outline, config, wing_axis=wing_axis)

    # Step 5.5a: Assign vein tissue polygons
    assign_vein_tissue_polygons(veins, skel.median_vein_width_px, config, wing_outline)

    # Step 5.5b: Split merged intervein polygons
    logger.info("Step 5.5: Splitting merged intervein polygons")
    intervein_polys = split_merged_intervein_polygons(
        intervein_polys,
        veins,
        wing_outline,
        image_shape,
        skel.median_vein_width_px,
        config,
    )

    # Step 6: Name intervein regions
    logger.info("Step 6: Naming intervein regions")
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
