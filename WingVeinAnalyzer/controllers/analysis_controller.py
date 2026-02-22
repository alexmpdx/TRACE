"""Orchestrates the full GeoJSON-based analysis pipeline per image."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon

from WingVeinAnalyzer.controllers.measurement_controller import (
    WingMeasurements,
    compile_results,
    compute_measurements,
)
from WingVeinAnalyzer.models.geojson_parser import ParsedAnnotations, parse_geojson
from WingVeinAnalyzer.models.vein_graph import (
    VeinEdge,
    build_graph_from_polygons,
    build_graph_from_veins,
)
from WingVeinAnalyzer.models.vein_labeler import (
    VeinAssignment,
    VeinStatus,
    assign_veins,
    _extract_costa,
)
from WingVeinAnalyzer.models.vein_identifier import (
    identify_veins_and_regions,
    validate_regions_against_ground_truth,
)
from WingVeinAnalyzer.models.vein_skeleton import (
    extract_anterior_boundary,
    extract_edge_boundary_veins,
    extract_veins_from_mask,
)
from WingVeinAnalyzer.models.wing_geometry import (
    WingOutline,
    build_wing_outline,
    compute_compartments,
    detect_hinge_landmarks,
    partition_intervein_spaces,
    remove_hinge,
)
from WingVeinAnalyzer.views.overlay_view import render_rainbow_overlay, render_skeleton_overlay
from WingVeinAnalyzer.views.results_view import export_csv


@dataclass
class PipelineResult:
    """Complete results from a single pipeline run."""

    assignments: list[VeinAssignment] = field(default_factory=list)
    poly_names: dict[int, str] = field(default_factory=dict)
    wing_outline: Optional[WingOutline] = None
    wing_blade: Optional[Polygon] = None
    intervein_regions: dict[str, Polygon] = field(default_factory=dict)
    anterior_compartment: Optional[Polygon] = None
    posterior_compartment: Optional[Polygon] = None
    measurements: Optional[WingMeasurements] = None
    skeleton_overlay_path: Optional[Path] = None
    rainbow_overlay_path: Optional[Path] = None
    csv_path: Optional[Path] = None


def run_pipeline(
    image_path: Path,
    geojson_path: Path,
    output_dir: Optional[Path] = None,
    microns_per_pixel: Optional[float] = None,
    snap_tolerance: float = 50.0,
    max_gap: float = 80.0,
) -> PipelineResult:
    """Run the full vein analysis pipeline on a TIFF image + GeoJSON annotations."""
    result = PipelineResult()

    # Setup output directory
    if output_dir is None:
        output_dir = image_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Parse GeoJSON annotations
    annotations = parse_geojson(geojson_path)

    # Route based on annotation type
    if annotations.intervein_polygons:
        result = _run_polygon_pipeline(
            image, annotations, output_dir, stem, microns_per_pixel, max_gap,
            geojson_path,
        )
    elif annotations.veins:
        result = _run_vein_pipeline(
            image, annotations, output_dir, stem, microns_per_pixel, snap_tolerance
        )
    else:
        raise ValueError("GeoJSON contains no vein or intervein annotations")

    return result


def _run_polygon_pipeline(
    image: np.ndarray,
    annotations: ParsedAnnotations,
    output_dir: Path,
    stem: str,
    microns_per_pixel: Optional[float],
    max_gap: float,
    geojson_path: Optional[Path] = None,
) -> PipelineResult:
    """Pipeline for intervein polygon annotations."""
    result = PipelineResult()
    polygons = annotations.intervein_polygons

    import logging
    logger = logging.getLogger(__name__)

    # Compute wing bounding box
    all_bounds = [p.bounds for p in polygons]
    wing_bbox = (
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    )

    if annotations.vein_polygons:
        # --- Vein-mask-primary path ---
        logger.info("Using vein-mask-primary pipeline (%d vein polygons)", len(annotations.vein_polygons))

        # Extract centerlines from vein mask (no poly_names needed)
        centerlines, nearest_labels, vein_mask_arr = extract_veins_from_mask(
            annotations.vein_polygons, polygons, image.shape[:2]
        )

        # Identify veins and regions independently via geometry
        id_result = identify_veins_and_regions(
            centerlines, polygons, annotations.vein_polygons,
            image.shape[:2], wing_bbox,
        )
        assignments = id_result.assignments
        poly_names = id_result.poly_names
        if id_result.polygons:
            polygons = id_result.polygons  # may have been updated by splitting
        for w in id_result.validation_report.warnings:
            logger.warning("Validation: %s", w)

        # Ground-truth validation (if expected overlay file exists)
        _run_ground_truth_validation(
            geojson_path, poly_names, polygons, logger,
            image_shape=image.shape, output_dir=output_dir,
        )

        # Add costa only if costal region exists
        has_costal = "costal_cell" in poly_names.values()
        if has_costal:
            costa_line = _extract_costa(polygons, poly_names, wing_bbox)
            if costa_line:
                assignments.append(
                    VeinAssignment(
                        vein_id="costa",
                        status=VeinStatus.COMPLETE,
                        edge_ids=[],
                        confidence=0.9,
                        evidence=["anterior_margin"],
                        length_px=costa_line.length,
                        line=costa_line,
                        endpoints=[
                            list(costa_line.coords)[0],
                            list(costa_line.coords)[-1],
                        ],
                    )
                )
        else:
            assignments.append(
                VeinAssignment(
                    vein_id="costa",
                    status=VeinStatus.ABSENT,
                    edge_ids=[],
                    confidence=0.0,
                    evidence=["no_costal_region"],
                )
            )

        result.assignments = assignments
        result.poly_names = poly_names

        # Apply scale calibration
        compile_results(assignments, microns_per_pixel)

        # Save diagnostics (vein-mask path)
        _save_diagnostics_voronoi(
            image, polygons, annotations.vein_polygons,
            poly_names, assignments, output_dir,
        )
    else:
        # --- Fallback: midline-only path (also uses new identifier) ---
        logger.info("Using midline-only fallback pipeline")

        graph, edges = build_graph_from_polygons(polygons, max_gap=max_gap)
        centerlines = {
            e.poly_pair: e.line for e in edges if e.poly_pair
        }

        id_result = identify_veins_and_regions(
            centerlines, polygons, [],
            image.shape[:2], wing_bbox,
        )
        assignments = id_result.assignments
        poly_names = id_result.poly_names
        if id_result.polygons:
            polygons = id_result.polygons  # may have been updated by splitting
        for w in id_result.validation_report.warnings:
            logger.warning("Validation: %s", w)

        # Ground-truth validation (if expected overlay file exists)
        _run_ground_truth_validation(
            geojson_path, poly_names, polygons, logger,
            image_shape=image.shape, output_dir=output_dir,
        )

        # Add costa only if costal region exists
        has_costal = "costal_cell" in poly_names.values()
        if has_costal:
            costa_line = _extract_costa(polygons, poly_names, wing_bbox)
            if costa_line:
                assignments.append(
                    VeinAssignment(
                        vein_id="costa",
                        status=VeinStatus.COMPLETE,
                        edge_ids=[],
                        confidence=0.9,
                        evidence=["anterior_margin"],
                        length_px=costa_line.length,
                        line=costa_line,
                        endpoints=[
                            list(costa_line.coords)[0],
                            list(costa_line.coords)[-1],
                        ],
                    )
                )
        else:
            assignments.append(
                VeinAssignment(
                    vein_id="costa",
                    status=VeinStatus.ABSENT,
                    edge_ids=[],
                    confidence=0.0,
                    evidence=["no_costal_region"],
                )
            )

        result.assignments = assignments
        result.poly_names = poly_names

        # Apply scale calibration
        compile_results(assignments, microns_per_pixel)

        # Save diagnostics (midline path)
        _save_diagnostics(image, polygons, edges, poly_names, assignments, output_dir)

    # Build wing outline (include vein polygons for full wing tip coverage)
    outline = build_wing_outline(
        polygons, vein_polygons=annotations.vein_polygons or None,
    )
    result.wing_outline = outline

    # Detect hinge and remove it
    landmarks = detect_hinge_landmarks(outline, polygons, poly_names)
    if landmarks:
        wing_blade = remove_hinge(outline, landmarks, polygons, poly_names)
    else:
        wing_blade = outline.polygon
    result.wing_blade = wing_blade

    # Partition intervein spaces
    all_regions = partition_intervein_spaces(wing_blade, polygons, poly_names)
    # Exclude costal cell from output regions
    regions = {k: v for k, v in all_regions.items() if k != "costal_cell"}
    result.intervein_regions = regions

    # Compute compartments
    l4_assignment = next((a for a in assignments if a.vein_id == "L4"), None)
    l4_line = l4_assignment.line if l4_assignment else None
    anterior, posterior = compute_compartments(wing_blade, l4_line)
    result.anterior_compartment = anterior
    result.posterior_compartment = posterior

    # Compute measurements (using filtered regions without costal cell)
    measurements = compute_measurements(
        assignments,
        wing_polygon=wing_blade,
        intervein_regions=regions,
        anterior_compartment=anterior,
        posterior_compartment=posterior,
        microns_per_pixel=microns_per_pixel,
    )
    result.measurements = measurements

    # Render overlays
    skel_path = output_dir / f"{stem}_skeleton_overlay.jpg"
    render_skeleton_overlay(
        image, assignments, outline_polygon=outline.polygon, output_path=skel_path,
    )
    result.skeleton_overlay_path = skel_path

    rainbow_path = output_dir / f"{stem}_rainbow_overlay.jpg"
    render_rainbow_overlay(image, regions, output_path=rainbow_path)
    result.rainbow_overlay_path = rainbow_path

    # Export CSV
    csv_path = output_dir / f"{stem}_measurements.csv"
    export_csv(
        assignments, csv_path, image_name=stem, measurements=measurements
    )
    result.csv_path = csv_path

    return result


# ---------------------------------------------------------------------------
# Ground-truth validation helper
# ---------------------------------------------------------------------------

def _run_ground_truth_validation(
    geojson_path: Optional[Path],
    poly_names: dict[int, str],
    polygons: list[Polygon],
    logger,
    image_shape: tuple[int, ...] = (),
    output_dir: Optional[Path] = None,
) -> None:
    """Check for expected overlay file and run ground-truth validation."""
    if geojson_path is None:
        return

    geojson_dir = geojson_path.parent
    stem = geojson_path.stem
    expected_path = geojson_dir / f"{stem}_expected_intervein_overlay.geojson"

    if not expected_path.exists():
        return

    logger.info("Found ground-truth overlay: %s", expected_path.name)
    report = validate_regions_against_ground_truth(
        poly_names, polygons, expected_path,
    )

    accuracy = report.get("accuracy", 0.0)
    correct = report.get("correct", 0)
    validated = report.get("validated", 0)
    total = report.get("total", 0)
    logger.info(
        "Ground-truth accuracy: %d/%d validated (%.0f%%), %d total polygons",
        correct, validated, accuracy * 100, total,
    )

    for idx, info in sorted(report.get("per_polygon", {}).items()):
        if info["match"] is None:
            logger.info(
                "  P%d: %s — no GT region found (not validated)",
                idx, info["our_name"],
            )
        elif info["match"]:
            logger.info(
                "  P%d: %s ✓ (IoU=%.3f)",
                idx, info["our_name"], info["iou"],
            )
        else:
            logger.warning(
                "  P%d: %s ✗ expected %s (IoU=%.3f)",
                idx, info["our_name"], info["expected_name"], info["iou"],
            )

    # Save mask PNGs if we have image dimensions
    if len(image_shape) >= 2 and output_dir is not None:
        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)
        h, w = image_shape[:2]
        _save_region_mask(
            poly_names, polygons, (h, w), diag_dir / "region_mask.png",
        )
        _save_ground_truth_mask(
            expected_path, (h, w), diag_dir / "ground_truth_mask.png",
        )


def _save_region_mask(
    poly_names: dict[int, str],
    polygons: list[Polygon],
    image_hw: tuple[int, int],
    output_path: Path,
) -> None:
    """Render a solid-color mask PNG from pipeline polygon naming (no overlay)."""
    from WingVeinAnalyzer.models.vein_map import INTERVEIN_COLORS
    from WingVeinAnalyzer.models.vein_skeleton import _fill_polygon

    h, w = image_hw
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    for idx, name in poly_names.items():
        if idx >= len(polygons):
            continue
        color = INTERVEIN_COLORS.get(name, (180, 180, 180))
        # Create a single-channel mask for this polygon, then apply color
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        _fill_polygon(poly_mask, polygons[idx], 255)
        mask[poly_mask > 0] = color

    # Add region name labels
    for idx, name in poly_names.items():
        if idx >= len(polygons):
            continue
        cx = int(polygons[idx].centroid.x)
        cy = int(polygons[idx].centroid.y)
        label = f"P{idx}: {name}"
        cv2.putText(
            mask, label, (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), mask)


def _save_ground_truth_mask(
    expected_geojson_path: Path,
    image_hw: tuple[int, int],
    output_path: Path,
) -> None:
    """Render a solid-color mask PNG from ground-truth expected overlay GeoJSON."""
    import json
    from WingVeinAnalyzer.models.vein_identifier import _normalize_region_name
    from WingVeinAnalyzer.models.vein_map import INTERVEIN_COLORS
    from WingVeinAnalyzer.models.vein_skeleton import _fill_polygon

    h, w = image_hw
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    with open(expected_geojson_path) as f:
        data = json.load(f)

    for feature in data.get("features", []):
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})
        classification = props.get("classification", {})
        raw_name = classification.get("name", "")

        if geom.get("type") != "Polygon" or not raw_name:
            continue

        coords = geom.get("coordinates", [])
        if not coords:
            continue

        try:
            poly = Polygon(coords[0])
            if not poly.is_valid or poly.is_empty:
                continue
        except Exception:
            continue

        norm_name = _normalize_region_name(raw_name)
        color = INTERVEIN_COLORS.get(norm_name, (180, 180, 180))

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        _fill_polygon(poly_mask, poly, 255)
        mask[poly_mask > 0] = color

        # Label
        cx = int(poly.centroid.x)
        cy = int(poly.centroid.y)
        cv2.putText(
            mask, norm_name, (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), mask)


# ---------------------------------------------------------------------------
# Diagnostic output
# ---------------------------------------------------------------------------

# Distinct colors for segment visualization (BGR)
_DIAG_COLORS = [
    (0, 0, 255),    # red
    (0, 200, 0),    # green
    (255, 0, 0),    # blue
    (0, 200, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 128, 255),  # orange
    (128, 0, 255),  # pink
    (128, 255, 0),  # spring
    (255, 128, 0),  # sky
]


def _save_diagnostics(
    image: np.ndarray,
    polygons: list[Polygon],
    edges: list[VeinEdge],
    poly_names: dict[int, str],
    assignments: list[VeinAssignment],
    output_dir: Path,
) -> None:
    """Save diagnostic images and logs to output_dir/diagnostics/."""
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    from WingVeinAnalyzer.models.vein_map import VEIN_COLORS

    # 1. Polygon map — all polygons outlined with index + region name
    poly_img = image.copy()
    for i, poly in enumerate(polygons):
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        color = _DIAG_COLORS[i % len(_DIAG_COLORS)]
        cv2.polylines(poly_img, [pts], isClosed=True, color=color, thickness=3)
        cx, cy = int(poly.centroid.x), int(poly.centroid.y)
        name = poly_names.get(i, f"P{i}")
        label = f"P{i}: {name}"
        cv2.putText(
            poly_img, label, (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
        )
    cv2.imwrite(str(diag_dir / "polygon_map.jpg"), poly_img)

    # 2. All midlines — every extracted edge, color-coded by polygon pair
    mid_img = image.copy()
    for idx, edge in enumerate(edges):
        pts = np.array(edge.line.coords, dtype=np.int32)
        color = _DIAG_COLORS[idx % len(_DIAG_COLORS)]
        cv2.polylines(mid_img, [pts], isClosed=False, color=color, thickness=3)
        mid_pt = pts[len(pts) // 2]
        pi, pj = edge.poly_pair if edge.poly_pair else (-1, -1)
        ni = poly_names.get(pi, f"P{pi}")
        nj = poly_names.get(pj, f"P{pj}")
        label = f"E{edge.edge_id}: {ni}<->{nj}"
        cv2.putText(
            mid_img, label, (int(mid_pt[0]) - 60, int(mid_pt[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
        )
    cv2.imwrite(str(diag_dir / "all_midlines.jpg"), mid_img)

    # Build edge lookup by edge_id for per-vein diagnostics
    edge_by_id = {e.edge_id: e for e in edges}

    # 3 & 4. Per-vein segment and merged images
    for assignment in assignments:
        vein_id = assignment.vein_id
        vein_color = VEIN_COLORS.get(vein_id, (40, 40, 40))

        # Per-vein segments (before merge)
        if assignment.edge_ids:
            seg_img = image.copy()
            for seg_idx, eid in enumerate(assignment.edge_ids):
                edge = edge_by_id.get(eid)
                if edge is None:
                    continue
                pts = np.array(edge.line.coords, dtype=np.int32)
                color = _DIAG_COLORS[seg_idx % len(_DIAG_COLORS)]
                cv2.polylines(seg_img, [pts], isClosed=False, color=color, thickness=4)
                # Start marker (circle)
                cv2.circle(seg_img, (int(pts[0][0]), int(pts[0][1])), 12, color, -1)
                # End marker (square)
                ex, ey = int(pts[-1][0]), int(pts[-1][1])
                cv2.rectangle(seg_img, (ex - 10, ey - 10), (ex + 10, ey + 10), color, -1)
                # Label
                cv2.putText(
                    seg_img, f"seg{seg_idx} (E{eid})",
                    (int(pts[0][0]) + 15, int(pts[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
                )
            cv2.imwrite(str(diag_dir / f"vein_{vein_id}_segments.jpg"), seg_img)

        # Per-vein merged line
        if assignment.line is not None:
            merge_img = image.copy()
            pts = np.array(assignment.line.coords, dtype=np.int32)
            cv2.polylines(merge_img, [pts], isClosed=False, color=vein_color, thickness=5)
            # Start/end markers
            cv2.circle(merge_img, (int(pts[0][0]), int(pts[0][1])), 14, (0, 255, 0), -1)
            cv2.circle(merge_img, (int(pts[-1][0]), int(pts[-1][1])), 14, (0, 0, 255), -1)
            cv2.putText(
                merge_img, f"{vein_id} merged ({len(pts)} pts, {assignment.length_px:.0f}px)",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, vein_color, 2, cv2.LINE_AA,
            )
            cv2.imwrite(str(diag_dir / f"vein_{vein_id}_merged.jpg"), merge_img)

    # 5. Edge assignments text log
    log_lines = ["edge_id\tpoly_i\tpoly_j\tname_i\tname_j\tmatched_vein\tlength_px\n"]
    # Build reverse lookup: edge_id -> vein_id
    edge_to_vein: dict[int, str] = {}
    for a in assignments:
        for eid in a.edge_ids:
            edge_to_vein[eid] = a.vein_id

    for edge in edges:
        pi, pj = edge.poly_pair if edge.poly_pair else (-1, -1)
        ni = poly_names.get(pi, f"P{pi}")
        nj = poly_names.get(pj, f"P{pj}")
        vein = edge_to_vein.get(edge.edge_id, "unassigned")
        log_lines.append(
            f"{edge.edge_id}\t{pi}\t{pj}\t{ni}\t{nj}\t{vein}\t{edge.length_px:.1f}\n"
        )
    (diag_dir / "edge_assignments.txt").write_text("".join(log_lines))


def _save_diagnostics_voronoi(
    image: np.ndarray,
    polygons: list[Polygon],
    vein_polygons: list[Polygon],
    poly_names: dict[int, str],
    assignments: list[VeinAssignment],
    output_dir: Path,
) -> None:
    """Save diagnostic images for the vein-mask Voronoi pipeline."""
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    from WingVeinAnalyzer.models.vein_map import VEIN_COLORS
    from WingVeinAnalyzer.models.vein_skeleton import _fill_polygon
    from scipy import ndimage

    h, w = image.shape[:2]

    # 1. Polygon map (same as midline path)
    poly_img = image.copy()
    for i, poly in enumerate(polygons):
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        color = _DIAG_COLORS[i % len(_DIAG_COLORS)]
        cv2.polylines(poly_img, [pts], isClosed=True, color=color, thickness=3)
        cx, cy = int(poly.centroid.x), int(poly.centroid.y)
        name = poly_names.get(i, f"P{i}")
        label = f"P{i}: {name}"
        cv2.putText(
            poly_img, label, (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
        )
    cv2.imwrite(str(diag_dir / "polygon_map.jpg"), poly_img)

    # 2. Vein mask overlay
    vein_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in vein_polygons:
        _fill_polygon(vein_mask, poly, 255)
    mask_img = image.copy()
    mask_overlay = np.zeros_like(image)
    mask_overlay[:, :, 2] = vein_mask  # red channel
    mask_img = cv2.addWeighted(mask_img, 0.7, mask_overlay, 0.3, 0)
    cv2.imwrite(str(diag_dir / "vein_mask.jpg"), mask_img)

    # 3. Voronoi partition visualization
    label_map = np.zeros((h, w), dtype=np.int32)
    for i, poly in enumerate(polygons):
        _fill_polygon(label_map, poly, i + 1)
    background = (label_map == 0)
    _, nearest_indices = ndimage.distance_transform_edt(
        background, return_distances=True, return_indices=True
    )
    nearest_labels = label_map[nearest_indices[0], nearest_indices[1]]
    nearest_labels[~background] = label_map[~background]

    # Color-code Voronoi regions within vein mask
    voronoi_img = image.copy()
    vein_pixels = (vein_mask > 0)
    for i in range(len(polygons)):
        region_mask = (nearest_labels == i + 1) & vein_pixels
        color = _DIAG_COLORS[i % len(_DIAG_COLORS)]
        voronoi_img[region_mask] = (
            np.array(voronoi_img[region_mask], dtype=np.float32) * 0.4
            + np.array(color, dtype=np.float32) * 0.6
        ).astype(np.uint8)
    cv2.imwrite(str(diag_dir / "voronoi_partition.jpg"), voronoi_img)

    # 4. Per-vein merged line diagnostics
    for assignment in assignments:
        vein_id = assignment.vein_id
        vein_color = VEIN_COLORS.get(vein_id, (40, 40, 40))

        if assignment.line is not None:
            merge_img = image.copy()
            pts = np.array(assignment.line.coords, dtype=np.int32)
            cv2.polylines(merge_img, [pts], isClosed=False, color=vein_color, thickness=5)
            cv2.circle(merge_img, (int(pts[0][0]), int(pts[0][1])), 14, (0, 255, 0), -1)
            cv2.circle(merge_img, (int(pts[-1][0]), int(pts[-1][1])), 14, (0, 0, 255), -1)
            cv2.putText(
                merge_img, f"{vein_id} ({len(pts)} pts, {assignment.length_px:.0f}px)",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, vein_color, 2, cv2.LINE_AA,
            )
            cv2.imwrite(str(diag_dir / f"vein_{vein_id}_merged.jpg"), merge_img)

    # 5. All veins overview — all assignments on one image
    all_veins_img = image.copy()
    for assignment in assignments:
        vein_id = assignment.vein_id
        vein_color = VEIN_COLORS.get(vein_id, (40, 40, 40))
        if assignment.line is not None:
            pts = np.array(assignment.line.coords, dtype=np.int32)
            cv2.polylines(all_veins_img, [pts], isClosed=False, color=vein_color, thickness=4)
            mid_pt = pts[len(pts) // 2]
            cv2.putText(
                all_veins_img, vein_id,
                (int(mid_pt[0]) - 20, int(mid_pt[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, vein_color, 2, cv2.LINE_AA,
            )
    cv2.imwrite(str(diag_dir / "all_veins_classified.jpg"), all_veins_img)

    # 6. Vein assignments text log (with validation warnings)
    log_lines = ["vein_id\tstatus\tlength_px\tconfidence\tevidence\n"]
    for a in assignments:
        log_lines.append(
            f"{a.vein_id}\t{a.status.value}\t{a.length_px:.1f}\t{a.confidence:.2f}\t{','.join(a.evidence)}\n"
        )
    log_lines.append("\n--- Region Assignments ---\n")
    for idx in sorted(poly_names.keys()):
        log_lines.append(f"P{idx}\t{poly_names[idx]}\n")
    (diag_dir / "vein_assignments.txt").write_text("".join(log_lines))


def _run_vein_pipeline(
    image: np.ndarray,
    annotations: ParsedAnnotations,
    output_dir: Path,
    stem: str,
    microns_per_pixel: Optional[float],
    snap_tolerance: float,
) -> PipelineResult:
    """Pipeline for vein LineString annotations."""
    result = PipelineResult()

    # Build graph from vein intersections
    graph, nodes = build_graph_from_veins(
        annotations.veins, snap_tolerance=snap_tolerance
    )

    # Compute wing bounding box
    all_coords = []
    for v in annotations.veins:
        all_coords.extend(list(v.line.coords))
    if all_coords:
        xs = [c[0] for c in all_coords]
        ys = [c[1] for c in all_coords]
        wing_bbox = (min(xs), min(ys), max(xs), max(ys))
    else:
        wing_bbox = (0, 0, image.shape[1], image.shape[0])

    # Assign vein identities
    assignments = assign_veins(annotations.veins, graph, nodes, wing_bbox)
    result.assignments = assignments

    # Apply scale calibration
    compile_results(assignments, microns_per_pixel)

    # Render skeleton overlay
    skel_path = output_dir / f"{stem}_skeleton_overlay.jpg"
    render_skeleton_overlay(image, assignments, output_path=skel_path)
    result.skeleton_overlay_path = skel_path

    # Export CSV
    csv_path = output_dir / f"{stem}_measurements.csv"
    export_csv(assignments, csv_path, image_name=stem)
    result.csv_path = csv_path

    return result
