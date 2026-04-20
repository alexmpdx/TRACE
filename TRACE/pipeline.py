"""TRACE pipeline — preprocessing + identifyFeatures vein analysis.

Stage 1: preprocessing (landmarks, hinge chop, segmentation).
Stage 2: identifyFeatures (landmark-anchored vein ID, measurements, overlays).
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from identify_features.config import PipelineConfig

logger = logging.getLogger("TRACE")

# User-selectable Stage 2 outputs. Keys are internal IDs; values are GUI labels.
OUTPUT_TYPES = OrderedDict(
    [
        ("chopped_image", "Per-wing chopped (hinge-removed) image"),
        ("landmarks_overlay", "Per-wing landmark points overlay PNG"),
        ("segmentation_overlay", "Per-wing vein/intervein inference overlay PNG"),
        ("geojson", "Per-wing GeoJSON (named veins & regions)"),
        ("overlay", "Per-wing overlay PNG"),
        ("ap_overlay", "Per-wing AP compartment overlay PNG"),
        ("cv_ratio_overlay", "Per-wing CV ratio overlay PNG"),
        ("csv", "Batch measurements CSV"),
    ]
)


@dataclass
class TraceResult:
    """Result of the TRACE pipeline for a single image."""

    image_path: Path
    output_geojson_path: Optional[Path] = None
    overlay_path: Optional[Path] = None
    ap_overlay_path: Optional[Path] = None
    cv_ratio_overlay_path: Optional[Path] = None
    chopped_image_path: Optional[Path] = None
    landmarks_overlay_path: Optional[Path] = None
    segmentation_overlay_path: Optional[Path] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None  # "preprocessing" or "analysis"


DEFAULT_MAX_WORKERS = 1

# BGR fallback colors for segmentation classes if the GeoJSON feature has no color.
_SEG_FALLBACK_COLORS = {
    "vein": (0, 0, 200),
    "intervein": (200, 120, 0),
}


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to an OpenCV BGR tuple. Returns (128,128,128) on bad input."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)
    except (ValueError, IndexError, TypeError):
        return (128, 128, 128)


def _iter_polygon_rings(geometry: dict):
    """Yield numpy int32 arrays for each polygon ring in a GeoJSON geometry."""
    import numpy as np

    if geometry is None:
        return
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if gtype == "Polygon":
        polys = [coords]
    elif gtype == "MultiPolygon":
        polys = coords
    else:
        return
    for poly in polys:
        for ring in poly:
            pts = np.array(ring, dtype=np.float32)
            if pts.shape[0] < 3:
                continue
            yield pts.astype(np.int32).reshape(-1, 1, 2)


def _render_segmentation_overlay(base_bgr, seg_geojson_path: Path, out_path: Path, alpha: float = 0.45) -> bool:
    """Render vein/intervein polygons from a segmentation GeoJSON over the base image.

    Returns True on success, False if the GeoJSON is missing or has no drawable
    features.
    """
    import cv2

    try:
        with open(seg_geojson_path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("segmentation_overlay: cannot read %s: %s", seg_geojson_path, exc)
        return False

    features = data.get("features", []) if isinstance(data, dict) else []
    if not features:
        return False

    fill = base_bgr.copy()
    drew_any = False
    for feat in features:
        props = feat.get("properties", {}) or {}
        cls = props.get("class") or props.get("classification", {}).get("name")
        if cls not in ("vein", "intervein"):
            continue
        color = (
            _hex_to_bgr(props.get("color", ""))
            if props.get("color")
            else _SEG_FALLBACK_COLORS.get(cls, (180, 180, 180))
        )
        rings = list(_iter_polygon_rings(feat.get("geometry")))
        if not rings:
            continue
        cv2.fillPoly(fill, rings, color)
        drew_any = True

    if not drew_any:
        return False

    blended = cv2.addWeighted(fill, alpha, base_bgr, 1.0 - alpha, 0)
    # Re-draw crisp outlines on the blended image.
    for feat in features:
        props = feat.get("properties", {}) or {}
        cls = props.get("class") or props.get("classification", {}).get("name")
        if cls not in ("vein", "intervein"):
            continue
        color = (
            _hex_to_bgr(props.get("color", ""))
            if props.get("color")
            else _SEG_FALLBACK_COLORS.get(cls, (180, 180, 180))
        )
        rings = list(_iter_polygon_rings(feat.get("geometry")))
        if rings:
            cv2.polylines(blended, rings, True, color, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), blended))


def _render_landmarks_overlay(base_bgr, landmarks_geojson_path: Path, out_path: Path) -> bool:
    """Render landmark points from a landmarks GeoJSON over the base image."""
    import cv2

    try:
        with open(landmarks_geojson_path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("landmarks_overlay: cannot read %s: %s", landmarks_geojson_path, exc)
        return False

    predictions: dict[str, tuple[float, float]] = {}
    for feat in data.get("features", []) if isinstance(data, dict) else []:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        props = feat.get("properties", {}) or {}
        name = props.get("classification", {}).get("name") if isinstance(props.get("classification"), dict) else None
        name = name or props.get("name") or props.get("class") or f"lm_{len(predictions)}"
        predictions[name] = (float(coords[0]), float(coords[1]))

    if not predictions:
        return False

    try:
        from landmark_locator.scripts.visualize import draw_landmarks_on_image

        rendered = draw_landmarks_on_image(base_bgr, predictions)
    except Exception:
        logger.exception("landmarks_overlay: draw_landmarks_on_image failed")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), rendered))


def trace_folder(
    input_dir: Path,
    output_dir: Path,
    landmark_checkpoint: Path,
    segmentation_model_dir: Path,
    config: Optional["PipelineConfig"] = None,
    device=None,
    keep_intermediates: bool = False,
    outputs: Optional[set[str]] = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    show_vein_tissue: bool = False,
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
        max_workers: Number of Stage 2 wings to analyze in parallel. Stage 1
            (GPU preprocessing) always runs sequentially regardless of this
            setting. Values <=1 run Stage 2 sequentially.
        show_vein_tissue: If True, the per-wing overlay PNG fills buffered vein
            tissue polygons. Default False draws skeleton centerlines only.
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
            max_workers=max(1, int(max_workers)),
            show_vein_tissue=show_vein_tissue,
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
    max_workers: int,
    show_vein_tissue: bool,
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
        stages=(True, True, True),
        device=device,
        keep_chopped=("chopped_image" in outputs),
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
    logger.info(
        "=== Stage 2: identifyFeatures (outputs=%s, workers=%d) ===",
        sorted(outputs),
        max_workers,
    )

    import cv2
    from identify_features.controllers.pipeline import identify_wing
    from identify_features.views.csv_export import export_csv_batch
    from identify_features.views.geojson_export import export_geojson
    from identify_features.views.overlay import (
        render_ap_overlay_to_file,
        render_cv_ratio_overlay_to_file,
        render_overlay_to_file,
    )

    _STAGE2_ANALYSIS_OUTPUTS = {"geojson", "overlay", "ap_overlay", "cv_ratio_overlay", "csv"}
    needs_analysis = bool(_STAGE2_ANALYSIS_OUTPUTS & outputs)

    scale = config.um_per_px
    total = len(successful_preproc)
    stage2_slots: list[Optional[TraceResult]] = [None] * total
    csv_slots: list[Optional[tuple[str, object]]] = [None] * total

    progress_lock = threading.Lock()
    cancel_event = threading.Event()

    def _emit_progress(idx: int, stem: str, detail: str):
        if progress_callback is None:
            return
        with progress_lock:
            progress_callback(idx, total, stem, "analysis", detail)

    def _analyze_one(i: int, preproc_result) -> None:
        if cancel_event.is_set():
            return
        stem = preproc_result.image_path.stem
        image_in_preproc = preproc_dir / preproc_result.image_path.name

        _emit_progress(i, stem, "starting")
        t0 = time.time()
        try:
            wing_result = None
            if needs_analysis:
                wing_result = identify_wing(
                    detection_geojson=preproc_result.segmentation_geojson_path,
                    landmarks_geojson=preproc_result.landmarks_geojson_path,
                    image_path=image_in_preproc if image_in_preproc.exists() else preproc_result.image_path,
                    config=config,
                    specimen_id=stem,
                )

            trace_result = TraceResult(image_path=preproc_result.image_path)

            if "geojson" in outputs and wing_result is not None:
                gj_path = output_dir / f"{stem}_output.geojson"
                export_geojson(wing_result.veins, wing_result.intervein_regions, gj_path, um_per_px=scale)
                trace_result.output_geojson_path = gj_path

            needs_base = bool(
                {
                    "overlay",
                    "ap_overlay",
                    "cv_ratio_overlay",
                    "landmarks_overlay",
                    "segmentation_overlay",
                }
                & outputs
            )
            base = None
            if needs_base:
                img_path = image_in_preproc if image_in_preproc.exists() else preproc_result.image_path
                base = cv2.imread(str(img_path))
                if base is None:
                    logger.warning("%s: could not read image for overlays, skipping PNG outputs", stem)

            if "overlay" in outputs and base is not None and wing_result is not None:
                ov_path = output_dir / f"{stem}_overlay.png"
                render_overlay_to_file(
                    base,
                    wing_result.veins,
                    wing_result.intervein_regions,
                    ov_path,
                    show_vein_tissue=show_vein_tissue,
                )
                trace_result.overlay_path = ov_path

            if "ap_overlay" in outputs and base is not None and wing_result is not None:
                ap_path = output_dir / f"{stem}_ap_overlay.png"
                if render_ap_overlay_to_file(base, wing_result, ap_path):
                    trace_result.ap_overlay_path = ap_path

            if "cv_ratio_overlay" in outputs and base is not None and wing_result is not None:
                cv_path = output_dir / f"{stem}_cv_ratio_overlay.png"
                if render_cv_ratio_overlay_to_file(base, wing_result, cv_path, um_per_px=scale):
                    trace_result.cv_ratio_overlay_path = cv_path

            if "landmarks_overlay" in outputs and base is not None:
                lm_gj = preproc_result.landmarks_geojson_path
                if lm_gj and Path(lm_gj).exists():
                    lm_ov_path = output_dir / f"{stem}_landmarks_overlay.png"
                    if _render_landmarks_overlay(base, Path(lm_gj), lm_ov_path):
                        trace_result.landmarks_overlay_path = lm_ov_path

            if "segmentation_overlay" in outputs and base is not None:
                seg_gj = preproc_result.segmentation_geojson_path
                if seg_gj and Path(seg_gj).exists():
                    seg_ov_path = output_dir / f"{stem}_segmentation_overlay.png"
                    if _render_segmentation_overlay(base, Path(seg_gj), seg_ov_path):
                        trace_result.segmentation_overlay_path = seg_ov_path

            if "chopped_image" in outputs:
                chopped_src = getattr(preproc_result, "chopped_image_path", None)
                if chopped_src and Path(chopped_src).exists():
                    chopped_dst = output_dir / Path(chopped_src).name
                    try:
                        shutil.copy2(chopped_src, chopped_dst)
                        trace_result.chopped_image_path = chopped_dst
                    except OSError as exc:
                        logger.warning("%s: failed to copy chopped image: %s", stem, exc)
                else:
                    logger.warning("%s: chopped_image requested but no chopped file found", stem)

            stage2_slots[i] = trace_result
            if "csv" in outputs and wing_result is not None:
                csv_slots[i] = (stem, wing_result)

            elapsed = time.time() - t0
            _emit_progress(i, stem, f"done ({elapsed:.1f}s)")

        except InterruptedError:
            cancel_event.set()
            raise
        except Exception as e:
            elapsed = time.time() - t0
            logger.exception("Analysis failed for %s (%.1fs)", stem, elapsed)
            stage2_slots[i] = TraceResult(
                image_path=preproc_result.image_path,
                error=f"{e}\n{traceback.format_exc()}",
                error_stage="analysis",
            )

    interrupted = False
    if max_workers <= 1:
        for i, preproc_result in enumerate(successful_preproc):
            try:
                _analyze_one(i, preproc_result)
            except InterruptedError:
                interrupted = True
                break
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="trace-stage2") as executor:
            futures = {executor.submit(_analyze_one, i, pr): i for i, pr in enumerate(successful_preproc)}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except InterruptedError:
                    interrupted = True
                    cancel_event.set()
                except Exception:
                    logger.exception("Unexpected error in Stage 2 worker")

    for tr in stage2_slots:
        if tr is not None:
            results.append(tr)

    batch_results: list[tuple[str, object]] = [entry for entry in csv_slots if entry is not None]

    if interrupted:
        raise InterruptedError("Cancelled by user")

    # --- Batch CSV ---
    if "csv" in outputs and batch_results:
        csv_path = output_dir / "measurements.csv"
        try:
            export_csv_batch(batch_results, csv_path, um_per_px=scale)
            logger.info("Batch CSV: %s (%d wings)", csv_path, len(batch_results))
        except Exception:
            logger.exception("Failed to write batch CSV")

    return results
