"""TRACE pipeline — preprocessing + identifyFeatures vein analysis.

Stage 1: preprocessing (landmarks, hinge chop, segmentation).
Stage 2: identifyFeatures (landmark-anchored vein ID, measurements, overlays).
"""

from __future__ import annotations

import logging
import tempfile
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from identify_features.config import PipelineConfig

logger = logging.getLogger("TRACE")

# User-selectable Stage 2 outputs. Keys are internal IDs; values are GUI labels.
OUTPUT_TYPES = OrderedDict(
    [
        ("geojson", "Per-wing GeoJSON (named veins & regions)"),
        ("overlay", "Per-wing overlay PNG"),
        ("csv", "Batch measurements CSV"),
    ]
)


@dataclass
class TraceResult:
    """Result of the TRACE pipeline for a single image."""

    image_path: Path
    output_geojson_path: Optional[Path] = None
    overlay_path: Optional[Path] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None  # "preprocessing" or "analysis"


def trace_folder(
    input_dir: Path,
    output_dir: Path,
    landmark_checkpoint: Path,
    segmentation_model_dir: Path,
    config: Optional["PipelineConfig"] = None,
    device=None,
    keep_intermediates: bool = False,
    outputs: Optional[set[str]] = None,
    progress_callback=None,
) -> list[TraceResult]:
    """Run the TRACE pipeline on a folder of wing images.

    Args:
        input_dir: Folder containing wing images (.tif, .bmp, .png, .jpg).
        output_dir: Where to write results.
        landmark_checkpoint: Path to landmark model .pt file.
        segmentation_model_dir: Path to segmentation model directory.
        config: identifyFeatures PipelineConfig. None means use defaults.
            Scale (µm/px) is read from config.um_per_px.
        device: Torch device (None for auto-detect).
        keep_intermediates: If True, keep preprocessing files in output/intermediates/.
        outputs: Which Stage 2 outputs to produce. Keys from OUTPUT_TYPES.
            None means all outputs. Empty set skips Stage 2 entirely.
        progress_callback: callable(image_index, total, image_name, stage, detail).

    Returns:
        List of TraceResult, one per input image.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if outputs is None:
        outputs = set(OUTPUT_TYPES.keys())

    if config is None:
        from identify_features.config import PipelineConfig

        config = PipelineConfig()

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
            config=config,
            device=device,
            outputs=outputs,
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
    config: "PipelineConfig",
    device,
    outputs: set[str],
    progress_callback,
) -> list[TraceResult]:
    """Internal implementation — separated so temp dir cleanup is in the caller."""
    from preprocessing.pipeline import process_folder

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

    results: list[TraceResult] = []
    successful_preproc = []
    for r in preproc_results:
        if r.error is not None or not r.segmentation_geojson_path:
            results.append(
                TraceResult(
                    image_path=r.image_path,
                    error=r.error or "No segmentation output produced",
                    error_stage="preprocessing",
                )
            )
        else:
            successful_preproc.append(r)

    logger.info("Preprocessed %d/%d images successfully", len(successful_preproc), len(preproc_results))

    if not successful_preproc:
        return results

    if not outputs:
        logger.info("=== Stage 2: skipped (no outputs selected) ===")
        for preproc_result in successful_preproc:
            results.append(TraceResult(image_path=preproc_result.image_path))
        return results

    # --- Stage 2: identifyFeatures ---
    logger.info("=== Stage 2: identifyFeatures (outputs=%s) ===", sorted(outputs))

    import cv2
    from identify_features.controllers.pipeline import identify_wing
    from identify_features.views.csv_export import export_csv_batch
    from identify_features.views.geojson_export import export_geojson
    from identify_features.views.overlay import render_overlay_to_file

    scale = config.um_per_px
    batch_results: list[tuple[str, object]] = []  # (stem, WingResult)

    for i, preproc_result in enumerate(successful_preproc):
        stem = preproc_result.image_path.stem
        image_in_preproc = preproc_dir / preproc_result.image_path.name

        if progress_callback:
            progress_callback(i, len(successful_preproc), stem, "analysis", "starting")

        t0 = time.time()
        try:
            wing_result = identify_wing(
                detection_geojson=preproc_result.segmentation_geojson_path,
                landmarks_geojson=preproc_result.landmarks_geojson_path,
                image_path=image_in_preproc if image_in_preproc.exists() else preproc_result.image_path,
                config=config,
                specimen_id=stem,
            )

            trace_result = TraceResult(image_path=preproc_result.image_path)

            if "geojson" in outputs:
                gj_path = output_dir / f"{stem}_output.geojson"
                export_geojson(wing_result.veins, wing_result.intervein_regions, gj_path, um_per_px=scale)
                trace_result.output_geojson_path = gj_path

            if "overlay" in outputs:
                img_path = image_in_preproc if image_in_preproc.exists() else preproc_result.image_path
                base = cv2.imread(str(img_path))
                if base is None:
                    logger.warning("%s: could not read image for overlay, skipping", stem)
                else:
                    ov_path = output_dir / f"{stem}_overlay.png"
                    render_overlay_to_file(base, wing_result.veins, wing_result.intervein_regions, ov_path)
                    trace_result.overlay_path = ov_path

            if "csv" in outputs:
                batch_results.append((stem, wing_result))

            results.append(trace_result)
            elapsed = time.time() - t0
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

    # --- Batch CSV ---
    if "csv" in outputs and batch_results:
        csv_path = output_dir / "measurements.csv"
        try:
            export_csv_batch(batch_results, csv_path, um_per_px=scale)
            logger.info("Batch CSV: %s (%d wings)", csv_path, len(batch_results))
        except Exception:
            logger.exception("Failed to write batch CSV")

    return results
