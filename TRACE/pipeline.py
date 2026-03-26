"""TRACE pipeline — chains preprocessing and WingVeinAnalyzer into a single workflow.

Processes a folder of wing images through:
  1. Preprocessing (landmark detection, hinge removal, segmentation)
  2. Wing vein analysis (identification, measurement, overlay generation)

Outputs a consolidated CSV and per-wing overlay images.
"""

import logging
import shutil
import tempfile
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("TRACE")

# Measurement column groups — keys are internal IDs, values are GUI labels.
# Order matches the checkbox layout in the GUI.
MEASUREMENT_GROUPS = OrderedDict(
    [
        ("vein_lengths", "Vein lengths, status & gaps"),
        ("wing_dimensions", "Wing dimensions (length, width)"),
        ("wing_area", "Total wing area"),
        ("compartment_areas", "Compartment areas (anterior/posterior)"),
        ("intervein_areas", "Intervein region areas"),
        ("landmark_measurements", "Landmark measurements\n(wing length, CV distance, ratio)"),
    ]
)

_VEINS = ["costa", "L1", "L2", "L3", "L4", "L5", "ACV", "PCV"]


def _column_in_group(col: str, group: str) -> bool:
    """Test whether a CSV column belongs to a measurement group."""
    if group == "vein_lengths":
        return any(col.startswith(f"{v}_") for v in _VEINS)
    if group == "wing_dimensions":
        return col.startswith("wing_length") or col.startswith("wing_width")
    if group == "wing_area":
        return col.startswith("total_wing_area")
    if group == "compartment_areas":
        return "compartment_area" in col
    if group == "crossvein_distance":
        return col.startswith("crossvein_distance")
    if group == "intervein_areas":
        return col.endswith(("_area_px2", "_area_um2")) and "compartment" not in col and "wing" not in col
    if group == "landmark_measurements":
        return col.startswith("landmark_")
    return False


def filter_csv_columns(csv_path: Path, selected_groups: set[str]) -> None:
    """Rewrite a CSV file keeping only columns belonging to selected groups."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    keep = ["image"]
    for col in df.columns:
        if col == "image":
            continue
        if any(_column_in_group(col, g) for g in selected_groups):
            keep.append(col)
    df[keep].to_csv(csv_path, index=False)


@dataclass
class TraceResult:
    """Result of the combined pipeline for a single image."""

    image_path: Path
    skeleton_overlay_path: Optional[Path] = None
    rainbow_overlay_path: Optional[Path] = None
    landmark_overlay_path: Optional[Path] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None  # "preprocessing" or "analysis"


def trace_folder(
    input_dir: Path,
    output_dir: Path,
    landmark_checkpoint: Path,
    segmentation_model_dir: Path,
    scale: Optional[float] = None,
    smooth_sigma: float = 3.0,
    device=None,
    keep_intermediates: bool = False,
    measurement_groups: Optional[set[str]] = None,
    progress_callback=None,
) -> tuple[list[TraceResult], Optional[Path]]:
    """Run the full TRACE pipeline on a folder of wing images.

    Args:
        input_dir: Folder containing wing images (.tif, .bmp, .png, .jpg).
        output_dir: Where to write overlays and consolidated CSV.
        landmark_checkpoint: Path to landmark model .pt file.
        segmentation_model_dir: Path to segmentation model directory.
        scale: Microns per pixel (None for pixel-only measurements).
        smooth_sigma: Smoothing sigma for centerline extraction.
        device: Torch device (None for auto-detect).
        keep_intermediates: If True, keep preprocessing files in output/intermediates/.
        measurement_groups: Which column groups to include in the CSV.
            None or empty means include all. Keys from MEASUREMENT_GROUPS.
        progress_callback: callable(image_index, total, image_name, stage, detail).

    Returns:
        Tuple of (results, csv_path) where results is a list of TraceResult
        per image and csv_path is the consolidated CSV path (or None).
    """
    from preprocessing.pipeline import process_folder

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine preprocessing output location
    if keep_intermediates:
        preproc_dir = output_dir / "intermediates"
        preproc_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_obj = None
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="trace_preproc_")
        preproc_dir = Path(temp_dir_obj.name)

    try:
        return _run(
            input_dir=input_dir,
            output_dir=output_dir,
            preproc_dir=preproc_dir,
            landmark_checkpoint=landmark_checkpoint.resolve(),
            segmentation_model_dir=segmentation_model_dir.resolve(),
            scale=scale,
            smooth_sigma=smooth_sigma,
            device=device,
            measurement_groups=measurement_groups,
            progress_callback=progress_callback,
        )
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()


def _run(
    input_dir: Path,
    output_dir: Path,
    preproc_dir: Path,
    landmark_checkpoint: Path,
    segmentation_model_dir: Path,
    scale: Optional[float],
    smooth_sigma: float,
    device,
    measurement_groups: Optional[set[str]],
    progress_callback,
) -> tuple[list[TraceResult], Optional[Path]]:
    """Internal implementation — separated so temp dir cleanup is in the caller."""
    from preprocessing.pipeline import process_folder
    from TRACE.landmark_measures import compute_landmark_measurements, draw_landmark_overlay, load_landmarks
    from WingVeinAnalyzer.controllers.analysis_controller import run_pipeline
    from WingVeinAnalyzer.views.results_view import consolidate_csv

    # --- Stage 1: Preprocessing ---
    logger.info("=== Stage 1: Preprocessing ===")

    def _preproc_progress(idx, total, name, status):
        if progress_callback:
            progress_callback(idx, total, name, "preprocessing", status)

    preproc_results = process_folder(
        input_dir=input_dir,
        output_dir=preproc_dir,
        landmark_checkpoint=landmark_checkpoint,
        segmentation_model_dir=segmentation_model_dir,
        stages=(True, True, True, True),
        device=device,
        keep_chopped=False,
        progress_callback=_preproc_progress,
    )

    # Separate successes and failures
    successful_preproc = [r for r in preproc_results if r.error is None and r.segmentation_geojson_path]
    total_images = len(preproc_results)

    # Build initial results list (preprocessing failures recorded immediately)
    results: list[TraceResult] = []
    for r in preproc_results:
        if r.error is not None or not r.segmentation_geojson_path:
            results.append(
                TraceResult(
                    image_path=r.image_path,
                    error=r.error or "No segmentation output produced",
                    error_stage="preprocessing",
                )
            )

    if not successful_preproc:
        logger.error("No images were successfully preprocessed.")
        return results, None

    logger.info("Preprocessed %d/%d images successfully", len(successful_preproc), total_images)

    # --- Stage 2: Wing Vein Analysis ---
    logger.info("=== Stage 2: Wing Vein Analysis ===")

    analysis_tuples: list[tuple[str, list, object]] = []  # (stem, assignments, measurements)
    landmark_rows: dict[str, dict] = {}  # stem -> {wing_length_px, cv_distance_px, cv_wl_ratio}

    for i, preproc_result in enumerate(successful_preproc):
        stem = preproc_result.image_path.stem
        image_in_preproc = preproc_dir / preproc_result.image_path.name
        geojson_path = preproc_result.segmentation_geojson_path

        # WVA writes to a temp subdir, then we move overlays out
        wva_output = preproc_dir / "_wva" / stem

        if progress_callback:
            progress_callback(i, len(successful_preproc), stem, "analysis", "starting")

        t0 = time.time()
        try:
            wva_result = run_pipeline(
                image_path=image_in_preproc,
                geojson_path=geojson_path,
                output_dir=wva_output,
                microns_per_pixel=scale,
                smooth_sigma=smooth_sigma,
            )
            elapsed = time.time() - t0

            # Move overlays to final output dir
            trace_result = TraceResult(image_path=preproc_result.image_path)

            if wva_result.skeleton_overlay_path and wva_result.skeleton_overlay_path.exists():
                dest = output_dir / wva_result.skeleton_overlay_path.name
                shutil.move(str(wva_result.skeleton_overlay_path), str(dest))
                trace_result.skeleton_overlay_path = dest

            if wva_result.rainbow_overlay_path and wva_result.rainbow_overlay_path.exists():
                dest = output_dir / wva_result.rainbow_overlay_path.name
                shutil.move(str(wva_result.rainbow_overlay_path), str(dest))
                trace_result.rainbow_overlay_path = dest

            # Landmark measurements + overlay
            landmarks_geojson = preproc_dir / f"{stem}_landmarks.geojson"
            if landmarks_geojson.exists():
                landmarks = load_landmarks(landmarks_geojson)
                lm = compute_landmark_measurements(landmarks)
                if lm is not None:
                    landmark_rows[stem] = {
                        "landmark_wing_length_px": lm.wing_length_px,
                        "landmark_cv_distance_px": lm.cv_distance_px,
                        "landmark_cv_wl_ratio": lm.cv_wl_ratio,
                    }
                    overlay_dest = output_dir / f"{stem}_landmark_overlay.jpg"
                    if draw_landmark_overlay(image_in_preproc, overlay_dest, landmarks):
                        trace_result.landmark_overlay_path = overlay_dest
                else:
                    logger.warning("%s: missing required landmarks, skipping landmark measurements", stem)
            else:
                logger.warning("%s: no landmarks geojson found, skipping landmark measurements", stem)

            results.append(trace_result)
            analysis_tuples.append((stem, wva_result.assignments, wva_result.measurements))

            if progress_callback:
                progress_callback(i, len(successful_preproc), stem, "analysis", f"done ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - t0
            logger.exception("Analysis failed for %s (%.1fs)", stem, elapsed)
            results.append(
                TraceResult(
                    image_path=preproc_result.image_path,
                    error=f"{e}\n{traceback.format_exc()}",
                    error_stage="analysis",
                )
            )
        finally:
            # Clean up WVA temp output
            if wva_output.exists():
                shutil.rmtree(wva_output, ignore_errors=True)

    # --- Stage 3: Consolidate CSV ---
    csv_path = None
    if analysis_tuples:
        import pandas as pd

        csv_path = output_dir / "consolidated_measurements.csv"
        consolidate_csv(analysis_tuples, csv_path)

        # Merge landmark measurements into the consolidated CSV
        if landmark_rows:
            df = pd.read_csv(csv_path)
            lm_df = pd.DataFrame.from_dict(landmark_rows, orient="index")
            lm_df.index.name = "image"
            lm_df = lm_df.reset_index()
            df = df.merge(lm_df, on="image", how="left")
            df.to_csv(csv_path, index=False)

        # Filter columns to selected measurement groups
        if measurement_groups:
            filter_csv_columns(csv_path, measurement_groups)

        logger.info("Consolidated CSV: %s (%d wings)", csv_path, len(analysis_tuples))

    # Clean up WVA parent temp dir
    wva_parent = preproc_dir / "_wva"
    if wva_parent.exists():
        shutil.rmtree(wva_parent, ignore_errors=True)

    return results, csv_path
