"""Orchestrates the full GeoJSON-based analysis pipeline per image."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon

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
from WingVeinAnalyzer.models.vein_identifier import (
    VeinValidationReport,
    extract_l1_from_mask,
    identify_veins_and_regions,
    name_regions_from_veins,
    validate_regions_against_ground_truth,
    validate_veins_against_ground_truth,
)
from WingVeinAnalyzer.models.vein_labeler import (
    VeinAssignment,
    VeinStatus,
    _extract_costa,
    assign_veins,
)
from WingVeinAnalyzer.models.vein_map import (
    GRAPH_SNAP_VEINS_UM,
    MAX_GAP_UM,
    SMOOTH_SPACING_FINE_UM,
    SMOOTH_SPACING_UM,
    set_scale,
    um_to_px,
)
from WingVeinAnalyzer.models.vein_skeleton import extract_veins_from_mask
from WingVeinAnalyzer.models.wing_geometry import (
    HingeLandmarks,
    WingOutline,
    build_wing_outline,
    compute_compartments,
    detect_hinge_landmarks,
    partition_by_vein_extension,
    partition_intervein_spaces,
    remove_hinge,
)
from WingVeinAnalyzer.utils.skeleton_utils import smooth_line, smooth_polygon
from WingVeinAnalyzer.views.overlay_view import render_rainbow_overlay, render_skeleton_overlay
from WingVeinAnalyzer.views.results_view import export_csv


def _load_landmark_points(image_path: Path) -> Optional[dict[str, tuple[float, float]]]:
    """Load landmark points from a *_landmarks.geojson file next to the image."""
    import json

    landmarks_path = image_path.parent / f"{image_path.stem}_landmarks.geojson"
    if not landmarks_path.exists():
        return None
    try:
        with open(landmarks_path) as f:
            data = json.load(f)
        points: dict[str, tuple[float, float]] = {}
        for feat in data.get("features", []):
            name = feat.get("properties", {}).get("classification", {}).get("name")
            coords = feat.get("geometry", {}).get("coordinates")
            if name and coords and len(coords) >= 2:
                points[name] = (float(coords[0]), float(coords[1]))
        return points if points else None
    except Exception:
        logger.warning("Failed to parse landmarks file: %s", landmarks_path)
        return None


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
    snap_tolerance: float | None = None,
    max_gap: float | None = None,
    smooth_sigma: float = 3.0,
) -> PipelineResult:
    """Run the full vein analysis pipeline on a TIFF image + GeoJSON annotations."""
    set_scale(microns_per_pixel)
    if snap_tolerance is None:
        snap_tolerance = um_to_px(GRAPH_SNAP_VEINS_UM)
    if max_gap is None:
        max_gap = um_to_px(MAX_GAP_UM)
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
            image,
            annotations,
            output_dir,
            stem,
            microns_per_pixel,
            max_gap,
            geojson_path,
            smooth_sigma=smooth_sigma,
            image_path=image_path,
        )
    elif annotations.veins:
        result = _run_vein_pipeline(image, annotations, output_dir, stem, microns_per_pixel, snap_tolerance)
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
    smooth_sigma: float = 3.0,
    image_path: Optional[Path] = None,
) -> PipelineResult:
    """Pipeline for intervein polygon annotations."""
    result = PipelineResult()
    polygons = annotations.intervein_polygons

    import logging

    logger = logging.getLogger(__name__)

    # Set up file handler to capture pipeline log
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    log_path = diag_dir / "pipeline.log"
    file_handler = logging.FileHandler(str(log_path), mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    logger.info("Pipeline log: %s", log_path)

    # Compute wing bounding box
    all_bounds = [p.bounds for p in polygons]
    wing_bbox = (
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    )

    landmark_points = _load_landmark_points(image_path) if image_path else None

    # Initialize variables shared across both pipeline paths
    vein_mask_arr = None
    centerlines = {}

    if annotations.vein_polygons:
        # --- Vein-mask-primary path ---
        logger.info("Using vein-mask-primary pipeline (%d vein polygons)", len(annotations.vein_polygons))

        # Extract centerlines from vein mask via skeletonization
        centerline_result = extract_veins_from_mask(
            annotations.vein_polygons,
            image.shape[:2],
            intervein_polygons=annotations.intervein_polygons,
        )
        centerlines = centerline_result.centerlines
        vein_mask_arr = centerline_result.vein_mask

        # Identify veins and regions independently via geometry
        id_result = identify_veins_and_regions(
            centerlines,
            polygons,
            annotations.vein_polygons,
            image.shape[:2],
            wing_bbox,
            original_polygons=list(annotations.intervein_polygons),
            dtip=landmark_points.get("DTip") if landmark_points else None,
            landmark_points=landmark_points,
            wing_polygon=annotations.wing_polygon,
        )
        assignments = id_result.assignments
        poly_names = id_result.poly_names
        if id_result.polygons:
            polygons = id_result.polygons  # may have been updated by splitting
        for w in id_result.validation_report.warnings:
            logger.warning("Validation: %s", w)

        # Extract L1 from vein mask using landmarks
        if vein_mask_arr is not None and landmark_points:
            sc = landmark_points.get("subcostal break")
            l1rs = landmark_points.get("L1-Rs")
            if sc and l1rs:
                l1_line = extract_l1_from_mask(vein_mask_arr, sc, l1rs)
                if l1_line is not None:
                    assignments = [a for a in assignments if a.vein_id != "L1"]
                    coords = list(l1_line.coords)
                    assignments.append(
                        VeinAssignment(
                            vein_id="L1",
                            status=VeinStatus.COMPLETE,
                            edge_ids=[],
                            confidence=0.9,
                            evidence=["landmark_mask_extraction"],
                            length_px=l1_line.length,
                            line=l1_line,
                            endpoints=[coords[0], coords[-1]],
                        )
                    )
                    logger.info("L1 extracted from vein mask: %.0f px", l1_line.length)

        # Vein-extension clipping: intersect each named polygon with its
        # matching vein-extension region to trim oversized areas
        vein_lines_for_ext = {
            a.vein_id: a.line
            for a in assignments
            if a.line is not None
            and a.vein_id != "costa"
            and not a.vein_id.startswith("EV")
            and a.status != VeinStatus.ABSENT
        }
        if vein_lines_for_ext:
            outline_temp = build_wing_outline(
                polygons,
                vein_polygons=annotations.vein_polygons or None,
            )
            # Don't extend L1's distal end (nearest to DTip)
            skip_eps: dict[str, list[int]] = {}
            if "L1" in vein_lines_for_ext and landmark_points and "DTip" in landmark_points:
                dtip = landmark_points["DTip"]
                l1c = list(vein_lines_for_ext["L1"].coords)
                d0 = (l1c[0][0] - dtip[0]) ** 2 + (l1c[0][1] - dtip[1]) ** 2
                d1 = (l1c[-1][0] - dtip[0]) ** 2 + (l1c[-1][1] - dtip[1]) ** 2
                skip_eps["L1"] = [0 if d0 < d1 else -1]
            ext_polys, ext_poly_veins, _ext_lines = partition_by_vein_extension(
                outline_temp.polygon,
                vein_lines_for_ext,
                image.shape[:2],
                skip_endpoints=skip_eps,
                landmark_points=landmark_points,
            )
            ext_poly_names = name_regions_from_veins(
                ext_polys,
                id_result.vein_map,
                wing_bbox,
                poly_veins=ext_poly_veins,
            )
            if ext_poly_names:
                polygons, poly_names = _clip_regions_by_extension(
                    polygons,
                    poly_names,
                    ext_polys,
                    ext_poly_names,
                )

        # Ground-truth validation (if expected overlay file exists)
        _run_ground_truth_validation(
            geojson_path,
            poly_names,
            polygons,
            logger,
            image_shape=image.shape,
            output_dir=output_dir,
            assignments=assignments,
        )

        result.assignments = assignments
        result.poly_names = poly_names

        # Smooth vein geometries
        _smooth_vein_assignments(assignments, sigma=smooth_sigma)

        # Apply scale calibration
        compile_results(assignments, microns_per_pixel)

        # Save diagnostics (vein-mask path)
        try:
            _save_diagnostics(
                image,
                polygons,
                annotations.vein_polygons,
                poly_names,
                assignments,
                output_dir,
            )
        except Exception as e:
            logger.warning("Diagnostics failed: %s", e)
    else:
        # --- Fallback: polygon boundary path (no vein mask) ---
        logger.info("Using polygon boundary fallback pipeline")

        graph, edges = build_graph_from_polygons(polygons, max_gap=max_gap)
        centerlines = {e.poly_pair: e.line for e in edges if e.poly_pair}

        id_result = identify_veins_and_regions(
            centerlines,
            polygons,
            [],
            image.shape[:2],
            wing_bbox,
            dtip=landmark_points.get("DTip") if landmark_points else None,
            landmark_points=landmark_points,
            wing_polygon=annotations.wing_polygon,
        )
        assignments = id_result.assignments
        poly_names = id_result.poly_names
        if id_result.polygons:
            polygons = id_result.polygons  # may have been updated by splitting
        for w in id_result.validation_report.warnings:
            logger.warning("Validation: %s", w)

        # Ground-truth validation (if expected overlay file exists)
        _run_ground_truth_validation(
            geojson_path,
            poly_names,
            polygons,
            logger,
            image_shape=image.shape,
            output_dir=output_dir,
            assignments=assignments,
        )

        result.assignments = assignments
        result.poly_names = poly_names

        # Smooth vein geometries
        _smooth_vein_assignments(assignments, sigma=smooth_sigma)

        # Apply scale calibration
        compile_results(assignments, microns_per_pixel)

        # Save diagnostics (midline path)
        _save_diagnostics_fallback(image, polygons, edges, poly_names, assignments, output_dir)

    # Build wing outline (include vein polygons for full wing tip coverage)
    outline = build_wing_outline(
        polygons,
        vein_polygons=annotations.vein_polygons or None,
    )
    result.wing_outline = outline

    # Detect hinge and remove it — use deep-learning landmarks if available
    landmark_points = _load_landmark_points(image_path) if image_path else None
    if landmark_points and "subcostal break" in landmark_points and "alula notch" in landmark_points:
        sc = landmark_points["subcostal break"]
        al = landmark_points["alula notch"]
        hinge_pts = [sc]
        if "L1-Rs" in landmark_points:
            hinge_pts.append(landmark_points["L1-Rs"])
        if "L4-L5" in landmark_points:
            hinge_pts.append(landmark_points["L4-L5"])
        hinge_pts.append(al)
        landmarks = HingeLandmarks(
            subcostal_break=sc,
            alula_notch=al,
            hinge_line=LineString(hinge_pts),
        )
    else:
        landmarks = detect_hinge_landmarks(outline, polygons, poly_names)
    if landmarks:
        wing_blade = remove_hinge(outline, landmarks, polygons, poly_names)
    else:
        wing_blade = outline.polygon
    result.wing_blade = wing_blade

    # Partition intervein spaces (polygons already updated by vein-extension above)
    regions = partition_intervein_spaces(wing_blade, polygons, poly_names)
    # Smooth region boundaries
    if smooth_sigma > 0:
        regions = {k: smooth_polygon(v, sigma=smooth_sigma) for k, v in regions.items()}
    result.intervein_regions = regions

    # Compute compartments
    l4_assignment = next((a for a in assignments if a.vein_id == "L4"), None)
    l4_line = l4_assignment.line if l4_assignment else None
    anterior, posterior = compute_compartments(wing_blade, l4_line)
    result.anterior_compartment = anterior
    result.posterior_compartment = posterior

    measurements = compute_measurements(
        assignments,
        wing_polygon=wing_blade,
        intervein_regions=regions,
        anterior_compartment=anterior,
        posterior_compartment=posterior,
        microns_per_pixel=microns_per_pixel,
    )
    result.measurements = measurements

    # Render overlays (smooth the wing outline for display)
    outline_smooth = smooth_polygon(outline.polygon, sigma=smooth_sigma * 1.67) if smooth_sigma > 0 else outline.polygon
    skel_path = output_dir / f"{stem}_skeleton_overlay.jpg"
    render_skeleton_overlay(
        image,
        assignments,
        outline_polygon=outline_smooth,
        output_path=skel_path,
    )
    result.skeleton_overlay_path = skel_path

    rainbow_path = output_dir / f"{stem}_rainbow_overlay.jpg"
    render_rainbow_overlay(image, regions, output_path=rainbow_path)
    result.rainbow_overlay_path = rainbow_path

    # Export CSV
    csv_path = output_dir / f"{stem}_measurements.csv"
    export_csv(assignments, csv_path, image_name=stem, measurements=measurements)
    result.csv_path = csv_path

    # Write step-by-step logic summary
    _write_step_summary(
        output_dir,
        image,
        annotations,
        polygons,
        poly_names,
        assignments,
        measurements,
        outline,
        wing_blade,
        regions,
        anterior,
        posterior,
        vein_mask_arr=vein_mask_arr if annotations.vein_polygons else None,
        centerlines=centerlines,
    )

    # Remove file handler to avoid duplicate logging in subsequent runs
    root_logger.removeHandler(file_handler)
    file_handler.close()
    logger.info("Pipeline log saved to %s", log_path)

    return result


# ---------------------------------------------------------------------------
# Vein smoothing
# ---------------------------------------------------------------------------

_CROSSVEIN_IDS = {"ACV", "PCV"}


def _smooth_vein_assignments(
    assignments: list[VeinAssignment],
    sigma: float = 3.0,
) -> None:
    """Apply Gaussian smoothing to all vein LineStrings in-place."""
    if sigma <= 0:
        return
    for a in assignments:
        if a.line is None:
            continue
        if a.vein_id in _CROSSVEIN_IDS:
            a.line = smooth_line(a.line, sigma=max(sigma * 0.67, 0.5), sample_spacing=um_to_px(SMOOTH_SPACING_FINE_UM))
        else:
            a.line = smooth_line(a.line, sigma=sigma, sample_spacing=um_to_px(SMOOTH_SPACING_UM))
        a.length_px = a.line.length


# ---------------------------------------------------------------------------
# Vein-extension clipping
# ---------------------------------------------------------------------------


def _clip_regions_by_extension(
    polygons: list[Polygon],
    poly_names: dict[int, str],
    ext_polys: list[Polygon],
    ext_names: dict[int, str],
    min_area: float = 1000,
) -> tuple[list[Polygon], dict[int, str]]:
    """Split each polygon along vein-extension boundaries.

    For each original polygon, intersect with every vein-extension region.
    Significant overlaps become separate polygons with the extension region's
    name.  This divides oversized polygons into proper compartments while
    only making individual pieces smaller.
    """
    from shapely.geometry import MultiPolygon

    # Build name → union of vein-extension polygons
    ext_by_name: dict[str, Polygon] = {}
    for i, name in ext_names.items():
        if name in ext_by_name:
            ext_by_name[name] = ext_by_name[name].union(ext_polys[i])
        else:
            ext_by_name[name] = ext_polys[i]

    new_polygons: list[Polygon] = []
    new_names: dict[int, str] = {}

    for i, orig_poly in enumerate(polygons):
        if i not in poly_names:
            new_polygons.append(orig_poly)
            continue

        # Intersect with every extension region
        pieces: list[tuple[str, Polygon]] = []
        for ext_name, ext_poly in ext_by_name.items():
            inter = orig_poly.intersection(ext_poly)
            if inter.is_empty or inter.area < min_area:
                continue
            if isinstance(inter, MultiPolygon):
                inter = max(inter.geoms, key=lambda g: g.area)
            if isinstance(inter, Polygon) and inter.area >= min_area:
                pieces.append((ext_name, inter))

        if pieces:
            if len(pieces) == 1:
                # Not split — keep original name
                idx = len(new_polygons)
                new_polygons.append(pieces[0][1])
                new_names[idx] = poly_names[i]
            else:
                orig_name = poly_names[i]
                pieces.sort(key=lambda p: p[1].area, reverse=True)
                main_area = pieces[0][1].area
                # Filter out tiny edge slivers
                kept = [p for p in pieces if p[1].area >= main_area * 0.15]
                if not kept:
                    kept = [pieces[0]]
                if len(kept) == 1:
                    # Only one substantial piece — keep original name
                    idx = len(new_polygons)
                    new_polygons.append(kept[0][1])
                    new_names[idx] = orig_name
                else:
                    # Multiple substantial pieces — genuine split of a
                    # merged region; trust extension names for all pieces
                    for ext_name, piece in kept:
                        idx = len(new_polygons)
                        new_polygons.append(piece)
                        new_names[idx] = ext_name
        else:
            idx = len(new_polygons)
            new_polygons.append(orig_poly)
            new_names[idx] = poly_names[i]

    return new_polygons, new_names


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
    assignments: Optional[list[VeinAssignment]] = None,
) -> None:
    """Check for expected overlay files and run ground-truth validation."""
    if geojson_path is None:
        return

    geojson_dir = geojson_path.parent
    stem = geojson_path.stem

    # --- Region validation ---
    expected_path = geojson_dir / f"{stem}_expected_intervein_overlay.geojson"
    region_report: Optional[dict] = None

    if expected_path.exists():
        logger.info("Found ground-truth overlay: %s", expected_path.name)
        region_report = validate_regions_against_ground_truth(
            poly_names,
            polygons,
            expected_path,
        )

        accuracy = region_report.get("accuracy", 0.0)
        correct = region_report.get("correct", 0)
        validated = region_report.get("validated", 0)
        total = region_report.get("total", 0)
        logger.info(
            "Ground-truth accuracy: %d/%d validated (%.0f%%), %d total polygons",
            correct,
            validated,
            accuracy * 100,
            total,
        )

        for idx, info in sorted(region_report.get("per_polygon", {}).items()):
            if info["match"] is None:
                logger.info(
                    "  P%d: %s — no GT region found (not validated)",
                    idx,
                    info["our_name"],
                )
            elif info["match"]:
                logger.info(
                    "  P%d: %s ✓ (IoU=%.3f, area=%.0fpx²)",
                    idx,
                    info["our_name"],
                    info["iou"],
                    info["area"],
                )
            else:
                logger.warning(
                    "  P%d: %s ✗ expected %s (IoU=%.3f, area=%.0fpx²)",
                    idx,
                    info["our_name"],
                    info["expected_name"],
                    info["iou"],
                    info["area"],
                )

        # Save mask PNGs if we have image dimensions
        if len(image_shape) >= 2 and output_dir is not None:
            diag_dir = output_dir / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            h, w = image_shape[:2]
            _save_region_mask(
                poly_names,
                polygons,
                (h, w),
                diag_dir / "region_mask.png",
            )
            _save_ground_truth_mask(
                expected_path,
                (h, w),
                diag_dir / "ground_truth_mask.png",
            )

    # --- Vein skeleton validation ---
    skeleton_path = geojson_dir / f"{stem}_expected_skeleton_overlay.geojson"
    vein_report: Optional[VeinValidationReport] = None

    if skeleton_path.exists() and assignments is not None:
        logger.info("Found ground-truth skeleton: %s", skeleton_path.name)
        vein_report = validate_veins_against_ground_truth(
            assignments,
            skeleton_path,
        )

        for m in vein_report.per_vein:
            logger.info(
                "  %s: Hausdorff=%.1fpx, mean_dev=%.1fpx, P95=%.1fpx, " "coverage=%.1f%%, GT=%.0fpx, pred=%.0fpx",
                m.vein_name,
                m.hausdorff_px,
                m.mean_deviation_px,
                m.p95_deviation_px,
                m.coverage_ratio * 100,
                m.gt_length_px,
                m.pred_length_px,
            )

    # --- Write combined validation report ---
    if output_dir is not None and (region_report is not None or vein_report is not None):
        _write_validation_report(
            output_dir,
            region_report,
            vein_report,
            poly_names,
            polygons,
        )


def _write_validation_report(
    output_dir: Path,
    region_report: Optional[dict],
    vein_report: Optional[VeinValidationReport],
    poly_names: dict[int, str],
    polygons: list[Polygon],
) -> None:
    """Write a combined validation report with region + vein sections."""
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    report_path = diag_dir / "validation_report.txt"

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("VALIDATION REPORT")
    lines.append("=" * 60)

    # --- Region section ---
    lines.append("")
    lines.append("--- Region Identification ---")
    if region_report is not None:
        accuracy = region_report.get("accuracy", 0.0)
        correct = region_report.get("correct", 0)
        validated = region_report.get("validated", 0)
        lines.append(f"Accuracy: {correct}/{validated} ({accuracy:.0%})")
        lines.append("")
        lines.append(f"{'Poly':<6} {'Name':<25} {'Expected':<25} {'IoU':<8} {'Area (px²)':<12} {'Match'}")
        lines.append("-" * 90)
        for idx, info in sorted(region_report.get("per_polygon", {}).items()):
            match_str = "—" if info["match"] is None else ("✓" if info["match"] else "✗")
            lines.append(
                f"P{idx:<5} {info['our_name']:<25} {info.get('expected_name', ''):<25} "
                f"{info['iou']:<8.3f} {info['area']:<12.0f} {match_str}"
            )
    else:
        lines.append("No region ground truth available.")

    # --- Vein section ---
    lines.append("")
    lines.append("--- Vein Skeleton ---")
    if vein_report is not None and vein_report.per_vein:
        lines.append(
            f"Matched: {vein_report.matched_count}/{vein_report.total_gt_veins} GT veins, "
            f"mean Hausdorff={vein_report.mean_hausdorff:.1f}px, "
            f"mean coverage={vein_report.mean_coverage:.1%}"
        )
        lines.append("")
        lines.append(
            f"{'Vein':<8} {'Hausdorff':<12} {'Mean Dev':<12} {'P95 Dev':<12} "
            f"{'Coverage':<10} {'GT len':<10} {'Pred len':<10}"
        )
        lines.append("-" * 80)
        for m in vein_report.per_vein:
            lines.append(
                f"{m.vein_name:<8} {m.hausdorff_px:<12.1f} {m.mean_deviation_px:<12.1f} "
                f"{m.p95_deviation_px:<12.1f} {m.coverage_ratio:<10.1%} "
                f"{m.gt_length_px:<10.0f} {m.pred_length_px:<10.0f}"
            )
    else:
        lines.append("No vein skeleton ground truth available.")

    lines.append("")
    report_path.write_text("\n".join(lines))


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
            mask,
            label,
            (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
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

        if not raw_name:
            continue

        geom_type = geom.get("type")

        # Collect polygons from Polygon or MultiPolygon geometry
        polys_to_draw: list[Polygon] = []
        if geom_type == "Polygon":
            coords = geom.get("coordinates", [])
            if coords:
                try:
                    poly = Polygon(coords[0])
                    if poly.is_valid and not poly.is_empty:
                        polys_to_draw.append(poly)
                except Exception:
                    pass
        elif geom_type == "MultiPolygon":
            for poly_coords in geom.get("coordinates", []):
                if not poly_coords:
                    continue
                try:
                    poly = Polygon(poly_coords[0])
                    if poly.is_valid and not poly.is_empty:
                        polys_to_draw.append(poly)
                except Exception:
                    continue

        if not polys_to_draw:
            continue

        norm_name = _normalize_region_name(raw_name)
        color = INTERVEIN_COLORS.get(norm_name, (180, 180, 180))

        for poly in polys_to_draw:
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            _fill_polygon(poly_mask, poly, 255)
            mask[poly_mask > 0] = color

        # Label at centroid of largest polygon
        largest = max(polys_to_draw, key=lambda p: p.area)
        cx = int(largest.centroid.x)
        cy = int(largest.centroid.y)
        cv2.putText(
            mask,
            norm_name,
            (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), mask)


# ---------------------------------------------------------------------------
# Diagnostic output
# ---------------------------------------------------------------------------

# Distinct colors for segment visualization (BGR)
_DIAG_COLORS = [
    (0, 0, 255),  # red
    (0, 200, 0),  # green
    (255, 0, 0),  # blue
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
            poly_img,
            label,
            (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
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
            mid_img,
            label,
            (int(mid_pt[0]) - 60, int(mid_pt[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
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
                    seg_img,
                    f"seg{seg_idx} (E{eid})",
                    (int(pts[0][0]) + 15, int(pts[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv2.LINE_AA,
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
                merge_img,
                f"{vein_id} merged ({len(pts)} pts, {assignment.length_px:.0f}px)",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                vein_color,
                2,
                cv2.LINE_AA,
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
        log_lines.append(f"{edge.edge_id}\t{pi}\t{pj}\t{ni}\t{nj}\t{vein}\t{edge.length_px:.1f}\n")
    (diag_dir / "edge_assignments.txt").write_text("".join(log_lines))


def _save_diagnostics_fallback(
    image: np.ndarray,
    polygons: list[Polygon],
    vein_polygons: list[Polygon],
    poly_names: dict[int, str],
    assignments: list[VeinAssignment],
    output_dir: Path,
) -> None:
    """Save diagnostic images for the vein-mask pipeline."""
    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    from WingVeinAnalyzer.models.vein_map import VEIN_COLORS
    from WingVeinAnalyzer.models.vein_skeleton import _fill_polygon

    h, w = image.shape[:2]

    # 1. Polygon map
    poly_img = image.copy()
    for i, poly in enumerate(polygons):
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        color = _DIAG_COLORS[i % len(_DIAG_COLORS)]
        cv2.polylines(poly_img, [pts], isClosed=True, color=color, thickness=3)
        cx, cy = int(poly.centroid.x), int(poly.centroid.y)
        name = poly_names.get(i, f"P{i}")
        label = f"P{i}: {name}"
        cv2.putText(
            poly_img,
            label,
            (cx - 80, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
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

    # 3. Per-vein merged line diagnostics
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
                merge_img,
                f"{vein_id} ({len(pts)} pts, {assignment.length_px:.0f}px)",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                vein_color,
                2,
                cv2.LINE_AA,
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
                all_veins_img,
                vein_id,
                (int(mid_pt[0]) - 20, int(mid_pt[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                vein_color,
                2,
                cv2.LINE_AA,
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


def _write_step_summary(
    output_dir: Path,
    image: np.ndarray,
    annotations: ParsedAnnotations,
    polygons: list[Polygon],
    poly_names: dict[int, str],
    assignments: list[VeinAssignment],
    measurements: Optional["WingMeasurements"],
    outline: Optional["WingOutline"],
    wing_blade: Optional[Polygon],
    regions: dict[str, Polygon],
    anterior: Optional[Polygon],
    posterior: Optional[Polygon],
    vein_mask_arr: Optional[np.ndarray] = None,
    centerlines: Optional[dict] = None,
) -> None:
    """Write a step-by-step logic summary to diagnostics/pipeline_steps.txt."""
    import logging

    logger = logging.getLogger(__name__)

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    h, w = image.shape[:2]
    n_input_polys = len(annotations.intervein_polygons)
    n_vein_polys = len(annotations.vein_polygons)
    has_vein_mask = n_vein_polys > 0
    n_final_polys = len(polygons)

    lines: list[str] = []

    def section(title: str) -> None:
        lines.append(f"\n{'=' * 60}")
        lines.append(title)
        lines.append("=" * 60)

    def step(num: int, name: str) -> None:
        lines.append(f"\n--- Step {num}: {name} ---")

    # Header
    lines.append("PIPELINE STEP-BY-STEP SUMMARY")
    lines.append(f"Image: {w}x{h} px")
    lines.append(f"Pipeline: {'vein-mask-primary' if has_vein_mask else 'polygon-boundary-fallback'}")

    # Step 0: Load Inputs
    step(0, "Load Inputs")
    lines.append(f"  Intervein polygons: {n_input_polys}")
    lines.append(f"  Vein mask polygons: {n_vein_polys}")
    for i, p in enumerate(annotations.intervein_polygons):
        lines.append(f"    P{i}: area={p.area:.0f} px², centroid=({p.centroid.x:.0f}, {p.centroid.y:.0f})")

    # Step 1: Skeletonize & Extract Centerlines
    if has_vein_mask:
        step(1, "Skeletonize & Extract Centerlines")

        vein_px = int(vein_mask_arr.sum()) if vein_mask_arr is not None else 0
        lines.append(f"  Vein mask: {vein_px} pixels ({100.0 * vein_px / (h * w):.1f}% of image)")
        lines.append(f"  Method: morphological skeletonization + pruning (200px threshold)")
        lines.append(f"    → skeleton pixels assigned to polygon pairs via perpendicular EDT sampling")
        lines.append(f"    → traced into LineString centerlines")
    else:
        step(2, "Midline Boundary Extraction (fallback)")
        lines.append(f"  No vein mask — using boundary sampling between adjacent polygons")

    # Steps 3-4: visualization of step 2 data
    lines.append(f"\n  [Steps 3-4: Hull Seed Visualization + Centerline Extraction — visualization of step 2 data]")
    n_cl = len(centerlines) if centerlines else 0
    lines.append(f"  Centerlines extracted: {n_cl} segments")
    if centerlines:
        for key, line in sorted(centerlines.items()):
            lines.append(f"    ({key[0]},{key[1]}): {line.length:.0f} px")

    # Step 5: Identify Veins & Regions
    step(5, "Identify Veins & Regions")
    lines.append(f"  Logic:")
    lines.append(f"    1. Find triple junctions (3+ endpoints within 30px)")
    lines.append(f"    2. Merge collinear segments at junctions (tangent continuity)")
    lines.append(f"    3. Split merged paths at sharp turns (>70° direction change)")
    lines.append(
        f"    4. Classify longitudinals: L3 from DTip landmark, L4 next posterior, then L1/L2/L5 by position+length"
    )
    lines.append(f"    5. Classify crossveins: ACV near L3+L4, PCV near L4+L5")
    lines.append(f"    6. Name regions: boundary vein set → Jaccard match → area priors")
    lines.append(f"    7. Transfer names to original annotation polygons (greedy bipartite overlap)")
    lines.append(f"    8. Cross-validate: Y-ordering, boundary consistency, CV connectivity")
    lines.append(f"  [Steps 6-12: visualization sub-steps of step 5]")

    # Vein results
    lines.append(f"\n  Classified veins:")
    for a in assignments:
        if a.line is not None:
            lines.append(
                f"    {a.vein_id}: {a.length_px:.0f} px, status={a.status.value}, confidence={a.confidence:.2f}"
            )
        else:
            lines.append(f"    {a.vein_id}: ABSENT")

    # Region results
    lines.append(f"\n  Named regions ({len(poly_names)}/{n_final_polys} polygons):")
    for idx in sorted(poly_names.keys()):
        area = polygons[idx].area if idx < len(polygons) else 0
        lines.append(f"    P{idx} → {poly_names[idx]} ({area:.0f} px²)")

    # Step 13: L1 Recovery
    step(13, "L1 Recovery from Anterior Edge")
    l1 = next((a for a in assignments if a.vein_id == "L1"), None)
    if l1 and l1.line:
        lines.append(f"  L1: {l1.length_px:.0f} px")
    else:
        lines.append(f"  L1: ABSENT")

    # Step 14: Costa Extraction
    step(14, "Costa Extraction")
    costa = next((a for a in assignments if a.vein_id == "costa"), None)
    if costa and costa.line:
        lines.append(f"  Costa: {costa.length_px:.0f} px (anterior margin of marginal cell)")
    elif costa:
        lines.append(f"  Costa: ABSENT")

    # Step 15: Build Wing Outline
    step(15, "Build Wing Outline")
    if outline and outline.polygon:
        lines.append(f"  Outline area: {outline.polygon.area:.0f} px²")
        lines.append(f"  Logic: union(polygons.buffer(20px) + vein_polys.buffer(5px)).exterior")

    # Step 16: Hinge Detection & Removal
    step(16, "Hinge Detection & Removal")
    if wing_blade:
        lines.append(f"  Wing blade area: {wing_blade.area:.0f} px²")
        if outline and outline.polygon:
            pct = wing_blade.area / outline.polygon.area * 100
            lines.append(f"  Blade/outline ratio: {pct:.1f}%")

    # Step 17: Compute Compartments
    step(17, "Compute Compartments")
    if anterior and not anterior.is_empty:
        lines.append(f"  Anterior: {anterior.area:.0f} px²")
    if posterior and not posterior.is_empty:
        lines.append(f"  Posterior: {posterior.area:.0f} px²")
    if anterior and posterior and not anterior.is_empty and not posterior.is_empty:
        total = anterior.area + posterior.area
        lines.append(f"  Ratio: {anterior.area / total * 100:.1f}% / {posterior.area / total * 100:.1f}%")
    lines.append(f"  Logic: split wing blade along L4 vein")

    # Step 18: Compute Measurements
    step(18, "Compute Measurements")
    if measurements:
        lines.append(
            f"  Wing length: {measurements.wing_length_px:.0f} px"
            if measurements.wing_length_px
            else "  Wing length: N/A"
        )
        lines.append(
            f"  Wing width: {measurements.wing_width_px:.0f} px" if measurements.wing_width_px else "  Wing width: N/A"
        )
        lines.append(
            f"  Wing area: {measurements.total_wing_area_px2:.0f} px²"
            if measurements.total_wing_area_px2
            else "  Wing area: N/A"
        )
        if measurements.crossvein_distance_px:
            lines.append(f"  ACV-PCV distance: {measurements.crossvein_distance_px:.0f} px")
        lines.append(f"  Region areas:")
        for name, area in measurements.intervein_areas_px2.items():
            lines.append(f"    {name}: {area:.0f} px²")

    # Step 19: Final Overlays
    step(19, "Final Overlays")
    lines.append(f"  Skeleton overlay: veins in assigned colors on original image")
    lines.append(f"  Rainbow overlay: intervein regions filled with semi-transparent colors")

    summary_path = diag_dir / "pipeline_steps.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    logger.info("Step summary saved to %s", summary_path)


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
    graph, nodes = build_graph_from_veins(annotations.veins, snap_tolerance=snap_tolerance)

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
