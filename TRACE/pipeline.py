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
# vein_overlay and intervein_overlay together produce a single overlay PNG; the
# checkbox combination controls what gets drawn:
#   both selected  → combined vein + intervein overlay
#   vein only      → vein-only overlay (intervein pipeline skipped for speed)
#   intervein only → intervein-only overlay (no vein layers)
#   neither        → no overlay PNG written
OUTPUT_TYPES = OrderedDict(
    [
        ("wing_isolated_image", "Isolated wing image"),
        ("chopped_image", "Wing after hinge removal"),
        ("landmarks_overlay", "Landmark points overlay PNG"),
        ("landmarks_geojson", "Landmark predictions GeoJSON"),
        ("segmentation_overlay", "Vein/intervein inference overlay PNG"),
        ("segmentation_geojson", "Vein/intervein inference GeoJSON"),
        ("geojson", "Named vein and/or intervein GeoJSON"),
        ("vein_overlay", "Vein overlay PNG"),
        ("intervein_overlay", "Intervein region overlay PNG"),
        ("ap_overlay", "AP compartment overlay PNG"),
        ("cv_ratio_overlay", "CV ratio overlay PNG"),
        ("csv", "Measurements CSV"),
    ]
)

# Per-output tooltips for the GUI checkboxes. Keys match OUTPUT_TYPES.
OUTPUT_TOOLTIPS = {
    "wing_isolated_image": (
        "Save the masked single-wing image produced by wingIsolator. "
        "Requires the wing isolation step to be enabled and a wing-isolation "
        "model to be configured in the Models tab."
    ),
    "chopped_image": ("Save the hinge-removed image written by HingeChopper before segmentation."),
    "landmarks_overlay": ("Render predicted landmark points on a PNG copy of the input image."),
    "landmarks_geojson": (
        "Save the raw per-image landmark predictions as GeoJSON points "
        "(post-rotation, so coordinates align with the final overlays)."
    ),
    "segmentation_overlay": ("Render the raw vein/intervein semantic-segmentation classes on top of the image."),
    "segmentation_geojson": (
        "Save the raw vein/intervein semantic-segmentation polygons as GeoJSON, "
        "before identifyFeatures names veins and regions."
    ),
    "geojson": (
        "Per-wing GeoJSON file with named vein centerlines and intervein region polygons "
        "(consumable by QuPath, napari, etc.)."
    ),
    "vein_overlay": ("Render labeled vein centerlines (L1–L5, ACV, PCV, costa, …) on a PNG copy of the image."),
    "intervein_overlay": (
        "Render named intervein regions (marginal, submarginal, discal, …) on a PNG copy of the image."
    ),
    "ap_overlay": ("Render the anterior/posterior compartments split by the L3-L4 axis."),
    "cv_ratio_overlay": ("Render the cv-ratio (anterior crossvein vs. posterior crossvein position) visualization."),
    "csv": (
        "Single batch-level CSV with one row per image: vein lengths, region areas, "
        "wing dimensions, and any configured custom landmark distances."
    ),
}

# Back-compat: legacy "overlay" key is rewritten to the pair below at the
# entry to trace_folder() / _run(). Anything reading OUTPUT_TYPES sees the
# new keys only.
_LEGACY_OVERLAY_ALIASES = {"overlay": ("vein_overlay", "intervein_overlay")}

# Re-export so GUI/CLI can import groups + labels from a single module.
from identify_features.views.csv_export import (  # noqa: E402
    ALL_MEASUREMENT_GROUPS,
    MEASUREMENT_GROUPS,
)

# Outputs that require Step 6.1/6.2 (intervein polygon splitting + region naming).
# Used to decide whether to set PipelineConfig.skip_intervein_regions = True
# when nothing requested actually needs the intervein output.
# Note: csv and geojson are content-gated (csv via csv_measurement_groups,
# geojson via _geojson_content_wanted()) so they only contribute to the
# intervein requirement when their respective content flags are on.
_INTERVEIN_DEPENDENT_OUTPUTS = frozenset({"intervein_overlay", "geojson", "csv"})


def _geojson_content_wanted(
    outputs: set[str],
    csv_measurement_groups: set[str],
) -> tuple[bool, bool]:
    """Decide which GeoJSON content to write to mirror the user's other choices.

    The GeoJSON intermediate output should reflect whatever the user actually
    asked for in the main Outputs panel:
      - vein_overlay or csv-with-vein_lengths   → write veins
      - intervein_overlay or csv-with-intervein_areas → write regions
      - both flavors → write both
      - neither (user picked nothing or only outputs that don't speak to
        vein vs. intervein) → write both (the safe full-content default)

    Returns ``(write_veins, write_regions)``.
    """
    wants_veins = "vein_overlay" in outputs or ("csv" in outputs and "vein_lengths" in csv_measurement_groups)
    wants_intervein = "intervein_overlay" in outputs or (
        "csv" in outputs and "intervein_areas" in csv_measurement_groups
    )
    if not (wants_veins or wants_intervein):
        # Nothing explicit — default geojson to full content (and let the
        # caller decide whether to run the full pipeline).
        return True, True
    return wants_veins, wants_intervein


# Outputs that are upstream/intermediate artifacts (preprocessing or raw analysis files).
# In the GUI these live in the Settings dialog → General tab. The remaining keys are
# "final" outputs (overlays + CSV) shown in the main-window Outputs group.
INTERMEDIATE_OUTPUTS = frozenset(
    {
        "wing_isolated_image",
        "chopped_image",
        "landmarks_overlay",
        "landmarks_geojson",
        "segmentation_overlay",
        "segmentation_geojson",
        "geojson",
    }
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
    wing_isolated_image_path: Optional[Path] = None
    landmarks_overlay_path: Optional[Path] = None
    landmarks_geojson_path: Optional[Path] = None
    segmentation_overlay_path: Optional[Path] = None
    segmentation_geojson_path: Optional[Path] = None
    error: Optional[str] = None
    # "preprocessing", "analysis", "wing_isolation", or — for a garbage-filter abort — the
    # specific filter label (e.g. "solidity", "fragmentation", "uncalled vein tissue",
    # "missing veins"; see garbage_detector.FILTER_LABELS).
    error_stage: Optional[str] = None


DEFAULT_MAX_WORKERS = 1

# Per-output preprocessing requirements: (needs_landmarks, needs_hinge, needs_segmentation).
# Used by _required_stages() to skip upstream work the requested outputs don't depend on.
# Note: Stage 2 (wing isolation) is gated by the presence of a wing-isolation model dir,
# not by this table. The "wing_isolated_image" output is produced as a side effect of
# Stage 2 having run; selecting it without a wing-isolation model just yields nothing.
_OUTPUT_STAGE_REQUIREMENTS = {
    "wing_isolated_image": (False, False, False),
    "chopped_image": (True, True, False),
    "landmarks_overlay": (True, False, False),
    "landmarks_geojson": (True, False, False),
    "segmentation_overlay": (True, True, True),
    "segmentation_geojson": (True, True, True),
    "geojson": (True, True, True),
    "vein_overlay": (True, True, True),
    "intervein_overlay": (True, True, True),
    "ap_overlay": (True, True, True),
    "cv_ratio_overlay": (True, True, True),
    "csv": (True, True, True),
}

# Outputs that require running Stage 2 (identifyFeatures analysis), as opposed to
# just copying / rendering preprocessing artifacts.
_STAGE2_ANALYSIS_OUTPUTS = {
    "geojson",
    "vein_overlay",
    "intervein_overlay",
    "ap_overlay",
    "cv_ratio_overlay",
    "csv",
}


def _required_stages(outputs: set[str]) -> tuple[bool, bool, bool]:
    """Compute the minimal set of preprocessing stages needed to produce `outputs`.

    Returns (needs_landmarks, needs_hinge, needs_segmentation).
    """
    needs_lm = needs_hinge = needs_seg = False
    for key in outputs:
        lm, hinge, seg = _OUTPUT_STAGE_REQUIREMENTS.get(key, (True, True, True))
        needs_lm = needs_lm or lm
        needs_hinge = needs_hinge or hinge
        needs_seg = needs_seg or seg
    return (needs_lm, needs_hinge, needs_seg)


# ---------------------------------------------------------------------------
# Progress-bar weight model
# ---------------------------------------------------------------------------
# Baseline wall-time shares for a "default" all-outputs run, measured on a
# 133-image / 47-min reference (Workers=5). The progress bar uses these to map
# Stage 1 and Stage 2 image-count events onto a single 0–100 scale, so the bar
# fills monotonically across the whole pipeline instead of restarting at 0%
# when Stage 2 begins.
#
# Within-stage component shares (also from the reference run):
#   Stage 1 (≈17% of total):
#     wing_isolation ≈ 7.5% of total
#     segmentation   ≈ 6.7% of total
#     everything else ≈ 2.8% of total
#   Stage 2 (≈83% of total):
#     intervein splitter ≈ 70% of total
#     everything else    ≈ 13% of total
#
# When the user turns off wing isolation or skips the intervein steps, we
# subtract that component's slice and renormalize. The shares are approximate
# (per-image timing varies wildly with image content), but they're far more
# accurate than treating Stage 1 and Stage 2 as equal-weight bars.
_PROGRESS_STAGE1_TOTAL_SHARE = 17.0
_PROGRESS_STAGE2_TOTAL_SHARE = 83.0
_PROGRESS_S1_WING_ISO_SHARE = 7.5
_PROGRESS_S2_INTERVEIN_SHARE = 70.0


def compute_progress_weights(
    outputs: set[str],
    *,
    wing_isolation_enabled: bool,
    skip_intervein_regions: bool,
) -> tuple[float, float]:
    """Return (stage1_share, stage2_share) summing to 1.0.

    Used by the GUI progress bar to place Stage 1 and Stage 2 events on a
    unified 0–100 scale that reflects their relative wall-time cost. Falls
    back to (1.0, 0.0) when no Stage 2 outputs are selected.
    """
    has_stage2 = bool(_STAGE2_ANALYSIS_OUTPUTS & outputs)
    if not has_stage2:
        return (1.0, 0.0)

    s1 = _PROGRESS_STAGE1_TOTAL_SHARE
    s2 = _PROGRESS_STAGE2_TOTAL_SHARE
    if not wing_isolation_enabled:
        s1 -= _PROGRESS_S1_WING_ISO_SHARE
    # The intervein splitter is gated by `skip_intervein_regions` (set
    # automatically when no intervein-dependent outputs are requested).
    needs_intervein = bool(_INTERVEIN_DEPENDENT_OUTPUTS & outputs)
    if skip_intervein_regions or not needs_intervein:
        s2 -= _PROGRESS_S2_INTERVEIN_SHARE

    total = s1 + s2
    if total <= 0:
        # Defensive — shouldn't happen but avoid div-by-zero.
        return (1.0, 0.0)
    return (s1 / total, s2 / total)


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


def _render_segmentation_overlay(
    base_bgr,
    seg_geojson_path: Path,
    out_path: Path,
    alpha: float = 0.45,
    inverse_scale: float = 1.0,
) -> bool:
    """Render vein/intervein polygons from a segmentation GeoJSON over the base image.

    Returns True on success, False if the GeoJSON is missing or has no drawable
    features.

    `inverse_scale` is applied to every coordinate before drawing — used when
    the GeoJSON is in rescaled-pixel space (Stage 1 resolutionAdjust active)
    but the base image has already been resized back to original resolution.
    """
    import cv2

    try:
        with open(seg_geojson_path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("segmentation_overlay: cannot read %s: %s", seg_geojson_path, exc)
        return False

    if inverse_scale != 1.0:
        from resolutionAdjust import inverse_transform_geojson

        # `inverse_transform_geojson` takes scale_factor; pass 1/inverse_scale so
        # coords get multiplied by inverse_scale.
        data = inverse_transform_geojson(data, 1.0 / inverse_scale)

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


def _render_landmarks_overlay(
    base_bgr,
    landmarks_geojson_path: Path,
    out_path: Path,
    inverse_scale: float = 1.0,
) -> bool:
    """Render landmark points from a landmarks GeoJSON over the base image.

    `inverse_scale` multiplies every point coordinate before drawing — used when
    the GeoJSON is in rescaled-pixel space and the base image has already been
    resized back to original resolution.
    """
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
        predictions[name] = (float(coords[0]) * inverse_scale, float(coords[1]) * inverse_scale)

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
    include_unreliable_landmarks: bool = False,
    landmark_batch_size: Optional[int] = None,
    gate_override: Optional[dict] = None,
    wing_isolation_model_dir: Optional[Path] = None,
    wing_expand_fraction: float = 0.05,
    recursive: bool = False,
    do_rotation: bool = True,
    rotation_mirror_correct: bool = False,
    user_landmark_distances: Optional[list[dict]] = None,
    csv_measurement_groups: Optional[set[str]] = None,
    target_um_per_px: Optional[float] = None,
    rescale_tolerance_low: float = 0.85,
    rescale_tolerance_high: float = 1.15,
    skip_image_basenames: Optional[set[str]] = None,
    csv_filename_override: Optional[str] = None,
    pause_event: Optional["threading.Event"] = None,
    on_image_complete=None,
    on_image_failed_preproc=None,
    on_image_failed_analysis=None,
    show_color_key: bool = True,
    show_ectopic_labels: bool = True,
    show_region_labels: bool = True,
    vein_simplify_tolerance_px: float = 0.0,
    ectopic_label_font_scale: float = 1.0,
    show_compartment_labels: bool = True,
) -> list[TraceResult]:
    """Run the TRACE pipeline on a folder of wing images.

    Args:
        input_dir: Folder containing wing images (.tif, .bmp, .png, .jpg, .psd).
        output_dir: Where to write results.
        landmark_checkpoint: Path to landmark model .pt file.
        segmentation_model_dir: Path to segmentation model directory.
        config: identifyFeatures PipelineConfig. None means use defaults.
            Scale (µm/px) is read from config.um_per_px.
        device: Torch device (None for auto-detect).
        keep_intermediates: If True, keep preprocessing files in output/intermediates/.
        outputs: Which Stage 2 outputs to produce. Keys from OUTPUT_TYPES.
            None means all outputs. Empty set skips Stage 2 entirely.
        max_workers: Number of wings to process in parallel. Applies to BOTH
            Stage 1 (preprocessing — hinge chop and segmentation per image)
            and Stage 2 (identifyFeatures analysis). The landmark forward pass
            is still GPU-batched in one call upfront via landmark_batch_size.
            Values <=1 run everything sequentially.
        show_vein_tissue: If True, the per-wing overlay PNG fills buffered vein
            tissue polygons. Default False draws skeleton centerlines only.
        progress_callback: callable(image_index, total, image_name, stage, detail).
        include_unreliable_landmarks: If True, landmarks that fail the LandmarkLocator
            confidence gate are still written to downstream stages (marked reliable=false).
            Core-landmark failures always abort the image regardless of this flag.
        landmark_batch_size: Batch size for the landmark forward pass. None defaults to
            `max_workers` so the GUI's Workers spinbox controls both Stage 2 parallelism
            and Stage 1 batching. 1 disables batching, larger values trade memory for
            throughput.
        gate_override: Optional confidence-gate override applied at predictor construction
            time. Same shape as the `confidence:` block in `configs/default.yaml` —
            populated by the GUI's Landmarks tab or by `--gate-override-yaml` on the CLI.
        wing_isolation_model_dir: Optional Stage 2 — when set, a modelTOjson wing-id
            model produces a wing/background segmentation, wingIsolator masks all but
            the main wing, and the masked image becomes the input to LandmarkLocator
            and downstream stages. None disables Stage 2 entirely.
        wing_expand_fraction: Stage 2 buffer (fraction of sqrt(wing area)). Default 0.05.

    Returns:
        List of TraceResult, one per input image.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if outputs is None:
        outputs = set(OUTPUT_TYPES.keys())
    else:
        # Migrate legacy "overlay" → both new keys so old configs/CLI invocations
        # keep producing the combined overlay.
        outputs = set(outputs)
        for legacy_key, replacements in _LEGACY_OVERLAY_ALIASES.items():
            if legacy_key in outputs:
                outputs.discard(legacy_key)
                outputs.update(replacements)

    if config is None:
        from identify_features.config import PipelineConfig

        config = PipelineConfig()

    # Default measurement groups: all of them.
    if csv_measurement_groups is None:
        csv_measurement_groups = set(ALL_MEASUREMENT_GROUPS)
    else:
        csv_measurement_groups = set(csv_measurement_groups)

    # Skip §6.1/§6.2 (the resource-heavy intervein passes) when no requested
    # output depends on intervein region data. §6.3 vein tissue assignment
    # still runs because vein_overlay / overlay rendering needs tissue polygons.
    # CSV is intervein-dependent only when its "intervein_areas" group is on.
    # GeoJSON is intervein-dependent only when its content (mirrored from the
    # user's other choices) actually wants regions.
    #
    # The output-driven decision is authoritative: we always write True or False
    # here (rather than only writing True when nothing needs intervein) so that
    # a config passed in with skip_intervein_regions=True from a preset or
    # saved-settings JSON can't silently kill intervein_overlay output.
    csv_needs_intervein = "csv" in outputs and "intervein_areas" in csv_measurement_groups
    _, gj_writes_regions = _geojson_content_wanted(outputs, csv_measurement_groups)
    geojson_needs_intervein = "geojson" in outputs and gj_writes_regions
    always_intervein = outputs & (_INTERVEIN_DEPENDENT_OUTPUTS - {"csv", "geojson"})
    config.skip_intervein_regions = not (always_intervein or csv_needs_intervein or geojson_needs_intervein)

    if keep_intermediates:
        preproc_dir = output_dir / "intermediates"
        preproc_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_obj = None
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="trace_preproc_")
        preproc_dir = Path(temp_dir_obj.name)

    # Default landmark batch size to max_workers if the caller didn't specify.
    effective_workers = max(1, int(max_workers))
    effective_batch = landmark_batch_size if (landmark_batch_size and landmark_batch_size > 0) else effective_workers

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
            max_workers=effective_workers,
            show_vein_tissue=show_vein_tissue,
            progress_callback=progress_callback,
            include_unreliable_landmarks=include_unreliable_landmarks,
            landmark_batch_size=effective_batch,
            gate_override=gate_override,
            wing_isolation_model_dir=wing_isolation_model_dir.resolve() if wing_isolation_model_dir else None,
            wing_expand_fraction=wing_expand_fraction,
            keep_intermediates=keep_intermediates,
            recursive=recursive,
            do_rotation=do_rotation,
            rotation_mirror_correct=rotation_mirror_correct,
            user_landmark_distances=user_landmark_distances,
            csv_measurement_groups=csv_measurement_groups,
            target_um_per_px=target_um_per_px,
            rescale_tolerance_low=rescale_tolerance_low,
            rescale_tolerance_high=rescale_tolerance_high,
            skip_image_basenames=skip_image_basenames,
            csv_filename_override=csv_filename_override,
            pause_event=pause_event,
            on_image_complete=on_image_complete,
            on_image_failed_preproc=on_image_failed_preproc,
            on_image_failed_analysis=on_image_failed_analysis,
            show_color_key=show_color_key,
            show_ectopic_labels=show_ectopic_labels,
            show_region_labels=show_region_labels,
            vein_simplify_tolerance_px=vein_simplify_tolerance_px,
            ectopic_label_font_scale=ectopic_label_font_scale,
            show_compartment_labels=show_compartment_labels,
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
    include_unreliable_landmarks: bool = False,
    landmark_batch_size: Optional[int] = None,
    gate_override: Optional[dict] = None,
    wing_isolation_model_dir: Optional[Path] = None,
    wing_expand_fraction: float = 0.05,
    keep_intermediates: bool = False,
    recursive: bool = False,
    do_rotation: bool = True,
    rotation_mirror_correct: bool = False,
    user_landmark_distances: Optional[list[dict]] = None,
    csv_measurement_groups: Optional[set[str]] = None,
    target_um_per_px: Optional[float] = None,
    rescale_tolerance_low: float = 0.85,
    rescale_tolerance_high: float = 1.15,
    skip_image_basenames: Optional[set[str]] = None,
    csv_filename_override: Optional[str] = None,
    pause_event: Optional["threading.Event"] = None,
    on_image_complete=None,
    on_image_failed_preproc=None,
    on_image_failed_analysis=None,
    show_color_key: bool = True,
    show_ectopic_labels: bool = True,
    show_region_labels: bool = True,
    vein_simplify_tolerance_px: float = 0.0,
    ectopic_label_font_scale: float = 1.0,
    show_compartment_labels: bool = True,
) -> list[TraceResult]:
    """Internal implementation — separated so temp dir cleanup is in the caller."""
    from preprocessing.pipeline import PipelineResult as _PreprocResult
    from preprocessing.pipeline import discover_images, process_folder

    # Fast path: only the batch CSV is requested and the user has configured
    # custom landmark-distance pairs. We can skip identifyFeatures entirely
    # AND skip hinge chopping + segmentation, since measurementMaker only
    # needs the landmark GeoJSONs to compute distances.
    requested_analysis_outputs = _STAGE2_ANALYSIS_OUTPUTS & outputs
    fast_csv_path = bool(user_landmark_distances) and requested_analysis_outputs == {"csv"}

    if fast_csv_path:
        # Landmarks only — measurementMaker doesn't need vein tissue or hinge masks.
        # If the user also requested intermediate artifacts that need other
        # stages (chopped_image, segmentation_overlay), promote those stages.
        needs_lm = True
        needs_hinge = "chopped_image" in outputs
        needs_seg = "segmentation_overlay" in outputs
        stages = (needs_lm, needs_hinge, needs_seg)
    else:
        stages = _required_stages(outputs)
        needs_lm, needs_hinge, needs_seg = stages

    # --- Stage 1: Preprocessing ---
    logger.info("=== Stage 1: Preprocessing (stages=%s) ===", stages)

    def _preproc_progress(idx, total, name, status):
        if progress_callback:
            progress_callback(idx, total, name, "preprocessing", status)

    if not any(stages):
        # Nothing to preprocess and nothing to analyze — emit empty TraceResults per image.
        return [TraceResult(image_path=img) for img in discover_images(input_dir, recursive=recursive)]

    preproc_results = process_folder(
        input_dir=input_dir,
        output_dir=preproc_dir,
        landmark_checkpoint=landmark_checkpoint,
        segmentation_model_dir=segmentation_model_dir,
        stages=stages,
        device=device,
        keep_chopped=("chopped_image" in outputs),
        progress_callback=_preproc_progress,
        include_unreliable_landmarks=include_unreliable_landmarks,
        landmark_batch_size=landmark_batch_size,
        gate_override=gate_override,
        max_workers=max_workers,
        wing_model_dir=wing_isolation_model_dir,
        wing_expand_fraction=wing_expand_fraction,
        keep_intermediates=keep_intermediates,
        recursive=recursive,
        do_rotation=do_rotation,
        rotation_mirror_correct=rotation_mirror_correct,
        input_um_per_px=config.um_per_px,
        target_um_per_px=target_um_per_px,
        rescale_tolerance_low=rescale_tolerance_low,
        rescale_tolerance_high=rescale_tolerance_high,
        auto_detect_um_per_px=getattr(config, "auto_detect_um_per_px", False),
        # Filter at Stage 1 itself rather than post-hoc, so the
        # already-processed (or previously-failed-on-same-settings)
        # images don't even go through landmark / hinge / segmentation.
        skip_image_basenames=skip_image_basenames,
        # Stage 1 now polls pause_event between images (and between
        # landmark mini-batches), so a Pause click takes effect within
        # seconds even on a 100+ image folder. The Stage 1→2 boundary
        # check further down skips Stage 2 entirely when pause caught us
        # here, so the user doesn't have to wait through analysis too.
        pause_event=pause_event,
    )

    results: list[TraceResult] = []
    successful_preproc: list[_PreprocResult] = []
    failed_preproc_lock = threading.Lock()

    def _signal_failed_preproc(image_basename: str, error_text: str) -> None:
        """Thread-safe wrapper around the host's on_image_failed_preproc
        callback. Called for each image whose Stage 1 errored out so the
        host (GUI) can mark it in the manifest. On resume with unchanged
        settings, the host adds these to skip_image_basenames so Stage 1
        doesn't re-attempt them — see run_state.RunManifest.

        ``error_text`` carries the human-readable error so the GUI can
        surface it next to the failed image in the list. May be the
        empty string when no upstream error message was provided.
        """
        if on_image_failed_preproc is None:
            return
        with failed_preproc_lock:
            try:
                on_image_failed_preproc(image_basename, error_text)
            except Exception:
                logger.exception("on_image_failed_preproc callback raised")

    for r in preproc_results:
        missing_seg = needs_seg and not r.segmentation_geojson_path
        if r.error is not None or missing_seg:
            err_stage = r.error_stage or "preprocessing"
            results.append(
                TraceResult(
                    image_path=r.image_path,
                    error=r.error or "No segmentation output produced",
                    error_stage=err_stage,
                )
            )
            # Record the Stage-1 failure so the host's manifest grows a
            # failed_preproc_images entry. On a same-settings resume the
            # host adds this to the next Stage 1's skip set — gate
            # failures aren't going to change without a settings change.
            error_text = r.error or "No segmentation output produced"
            _signal_failed_preproc(r.image_path.name, error_text)
        else:
            successful_preproc.append(r)

    logger.info("Preprocessed %d/%d images successfully", len(successful_preproc), len(preproc_results))

    # Stage 1 → Stage 2 boundary: if pause caught us inside or right after
    # Stage 1, do NOT start identifyFeatures. The successful_preproc list
    # may already be short (Stage 1 paused partway), but even the images
    # that did finish Stage 1 here will be re-discovered on resume — they
    # aren't in the manifest's completed_images yet, so the next run will
    # re-preprocess them and analyze them in one pass. Honor the pause
    # straight away so the user sees the button flip to "Resume" within
    # seconds rather than after a Stage 2 sweep over whatever happened to
    # finish Stage 1.
    if pause_event is not None and pause_event.is_set():
        logger.info(
            "Pipeline paused after Stage 1: %d image(s) finished preprocessing this slice "
            "but Stage 2 was skipped; they will be re-processed on resume",
            len(successful_preproc),
        )
        return results

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
    from identify_features.garbage_detector import GarbageRejection
    from identify_features.views.csv_export import export_csv_batch
    from identify_features.views.geojson_export import export_geojson
    from identify_features.views.overlay import (
        render_ap_overlay_to_file,
        render_cv_ratio_overlay_to_file,
        render_overlay_to_file,
    )

    # `fast_csv_path` was computed at the top of _run and already trimmed the
    # preprocessing stages. Here it just disables Stage 2 analysis.
    needs_analysis = bool(requested_analysis_outputs) and not fast_csv_path
    if fast_csv_path:
        logger.info(
            "=== Stage 2 fast path: writing user-distance CSV without identifyFeatures (%d pair(s)) ===",
            len(user_landmark_distances),
        )

    scale = config.um_per_px
    total = len(successful_preproc)
    stage2_slots: list[Optional[TraceResult]] = [None] * total
    csv_slots: list[Optional[tuple[str, object]]] = [None] * total
    # specimen stem → path to its landmarks geojson, for post-CSV user-distance augmentation.
    user_dist_landmark_paths: dict[str, Path] = {}

    progress_lock = threading.Lock()
    cancel_event = threading.Event()
    completion_lock = threading.Lock()

    def _emit_progress(idx: int, stem: str, detail: str):
        if progress_callback is None:
            return
        with progress_lock:
            progress_callback(idx, total, stem, "analysis", detail)

    def _signal_complete(image_basename: str, success: bool, error_text: str = "") -> None:
        """Thread-safe wrapper around the per-image-completion callbacks.

        Called once per image after Stage 2 attempt. Dispatches based on
        outcome:
          - success=True  → on_image_complete (resume bookkeeping + GUI
            "Succeeded" status).
          - success=False → on_image_failed_analysis (GUI "Failed"
            status, with error_text) AND on_image_complete (the manifest
            still records failed Stage 2 images as "done" so they're
            not retried on resume without an explicit settings change).

        ``error_text`` carries the human-readable error for the GUI to
        surface next to the failed row. Ignored on success.
        """
        with completion_lock:
            try:
                if not success and on_image_failed_analysis is not None:
                    on_image_failed_analysis(image_basename, error_text)
                if on_image_complete is not None:
                    on_image_complete(image_basename)
            except Exception:
                logger.exception("image-completion callback raised")

    def _analyze_one(i: int, preproc_result) -> None:
        if cancel_event.is_set():
            return
        # Pause: between images only — running images finish cleanly so
        # the per-image artifacts aren't half-written. The Stage 2 caller
        # below also checks the event before submitting more work, so
        # ThreadPool workers stop being scheduled after the pause click.
        if pause_event is not None and pause_event.is_set():
            return
        # When recursive discovery flattened the input path into a unique basename,
        # `processed_image_path` points at the renamed copy in preproc_dir; otherwise
        # we fall back to the original input basename for both stem and lookup.
        if preproc_result.processed_image_path is not None:
            stem = preproc_result.processed_image_path.stem
            image_in_preproc = preproc_result.processed_image_path
        else:
            stem = preproc_result.image_path.stem
            image_in_preproc = preproc_dir / preproc_result.image_path.name

        # Per-image log context — handlers prefix records with [<image>].
        # Keep the user's original basename so the log identifies the source file.
        from preprocessing.pipeline import current_image as _current_image

        _current_image.set(preproc_result.image_path.name)

        _emit_progress(i, stem, "starting")
        t0 = time.time()
        # Per-image µm/px. Preprocessing stamps ``effective_um_per_px`` on
        # the result: metadata-derived value when auto-detect was on and the
        # image carried readable metadata, otherwise the caller's fallback
        # (config.um_per_px). Every downstream converter uses this per-image
        # value instead of the shared batch ``scale`` so measurements are
        # accurate even when images came from different microscope sessions.
        per_image_scale = getattr(preproc_result, "effective_um_per_px", None)
        if per_image_scale is None or per_image_scale <= 0:
            per_image_scale = scale
        try:
            wing_result = None
            if needs_analysis:
                # Build a config carrying this image's scale so identify_wing's
                # µm-based thresholds (snap radius, sample distances, ectopic
                # length cutoff, measurements) all convert through the right
                # µm/px. Cheap dataclass replace; no downstream cache churn.
                per_image_config = config
                if per_image_scale is not None and per_image_scale != config.um_per_px:
                    from dataclasses import replace as _replace

                    per_image_config = _replace(config, um_per_px=per_image_scale)
                wing_result = identify_wing(
                    detection_geojson=preproc_result.segmentation_geojson_path,
                    landmarks_geojson=preproc_result.landmarks_geojson_path,
                    image_path=image_in_preproc if image_in_preproc.exists() else preproc_result.image_path,
                    config=per_image_config,
                    specimen_id=stem,
                )
                if wing_result is not None:
                    # Stamp the effective µm/px so batch CSV export uses THIS
                    # specimen's scale on its row instead of a shared batch value.
                    wing_result.um_per_px = per_image_scale

            # Stage 1 inverse: when preprocessing rescaled this image, identify_wing's
            # outputs are in rescaled-pixel space. Map every geometry back to original
            # pixels and recompute cached length/area so CSV + GeoJSON exporters can use
            # this image's µm/px directly.
            rescale_factor = getattr(preproc_result, "rescale_factor", 1.0)
            if wing_result is not None and rescale_factor and rescale_factor != 1.0:
                from resolutionAdjust import inverse_rescale_wing_result

                inverse_rescale_wing_result(wing_result, rescale_factor, um_per_px=per_image_scale)

            trace_result = TraceResult(image_path=preproc_result.image_path)

            if "geojson" in outputs and wing_result is not None:
                gj_path = output_dir / f"{stem}_output.geojson"
                gj_write_veins, gj_write_regions = _geojson_content_wanted(outputs, csv_measurement_groups)
                export_geojson(
                    wing_result.veins if gj_write_veins else [],
                    wing_result.intervein_regions if gj_write_regions else [],
                    gj_path,
                    um_per_px=per_image_scale,
                    show_vein_tissue=show_vein_tissue,
                )
                trace_result.output_geojson_path = gj_path

            needs_base = bool(
                {
                    "vein_overlay",
                    "intervein_overlay",
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
                from TRACE.psd_loader import imread_any

                base = imread_any(img_path)
                if base is None:
                    logger.warning("%s: could not read image for overlays, skipping PNG outputs", stem)
                elif rescale_factor and rescale_factor != 1.0:
                    # Geometries above were mapped back to original-pixel space; the
                    # overlay base needs to follow so coordinates land on the right
                    # pixels. Rotation (Stage 6) stays baked in — only resolution
                    # is undone, not orientation.
                    from resolutionAdjust import inverse_resize_image

                    base = inverse_resize_image(base, rescale_factor)

            want_vein = "vein_overlay" in outputs
            want_intervein = "intervein_overlay" in outputs
            if (want_vein or want_intervein) and base is not None and wing_result is not None:
                ov_path = output_dir / f"{stem}_overlay.png"
                render_overlay_to_file(
                    base,
                    wing_result.veins,
                    wing_result.intervein_regions,
                    ov_path,
                    show_vein_tissue=show_vein_tissue,
                    show_veins=want_vein,
                    show_regions=want_intervein,
                    vein_color_overrides=config.vein_colors,
                    region_color_overrides=config.region_colors,
                    vein_opacity=config.vein_opacity,
                    intervein_opacity=config.intervein_opacity,
                    show_color_key=show_color_key,
                    show_ectopic_labels=show_ectopic_labels,
                    show_region_labels=show_region_labels,
                    vein_simplify_tolerance_px=vein_simplify_tolerance_px,
                    ectopic_label_font_scale=ectopic_label_font_scale,
                )
                trace_result.overlay_path = ov_path

            if "ap_overlay" in outputs and base is not None and wing_result is not None:
                ap_path = output_dir / f"{stem}_ap_overlay.png"
                if render_ap_overlay_to_file(
                    base, wing_result, ap_path,
                    show_compartment_labels=show_compartment_labels,
                ):
                    trace_result.ap_overlay_path = ap_path

            if "cv_ratio_overlay" in outputs and base is not None and wing_result is not None:
                cv_path = output_dir / f"{stem}_cv_ratio_overlay.png"
                if render_cv_ratio_overlay_to_file(base, wing_result, cv_path, um_per_px=per_image_scale):
                    trace_result.cv_ratio_overlay_path = cv_path

            # When Stage 1 rescaled, the saved GeoJSONs are in rescaled-pixel
            # space; pass `inverse_scale = 1/sf` so coords match the resized
            # original-resolution base.
            overlay_inverse_scale = (1.0 / rescale_factor) if rescale_factor and rescale_factor != 1.0 else 1.0

            if "landmarks_overlay" in outputs and base is not None:
                lm_gj = preproc_result.landmarks_geojson_path
                if lm_gj and Path(lm_gj).exists():
                    lm_ov_path = output_dir / f"{stem}_landmarks_overlay.png"
                    if _render_landmarks_overlay(base, Path(lm_gj), lm_ov_path, inverse_scale=overlay_inverse_scale):
                        trace_result.landmarks_overlay_path = lm_ov_path

            if "segmentation_overlay" in outputs and base is not None:
                seg_gj = preproc_result.segmentation_geojson_path
                if seg_gj and Path(seg_gj).exists():
                    seg_ov_path = output_dir / f"{stem}_segmentation_overlay.png"
                    if _render_segmentation_overlay(
                        base, Path(seg_gj), seg_ov_path, inverse_scale=overlay_inverse_scale
                    ):
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

            if "landmarks_geojson" in outputs:
                # Stage 1 overwrites this file in-place after rotation, so the
                # copy here picks up the post-rotation coordinates that align
                # with the landmarks overlay PNG.
                lm_src = getattr(preproc_result, "landmarks_geojson_path", None)
                if lm_src and Path(lm_src).exists():
                    lm_dst = output_dir / Path(lm_src).name
                    try:
                        shutil.copy2(lm_src, lm_dst)
                        trace_result.landmarks_geojson_path = lm_dst
                    except OSError as exc:
                        logger.warning("%s: failed to copy landmarks GeoJSON: %s", stem, exc)
                else:
                    logger.warning("%s: landmarks_geojson requested but no source file found", stem)

            if "segmentation_geojson" in outputs:
                # The temp file is bare "<stem>.geojson"; rename on copy so the
                # user's folder doesn't end up with an ambiguous name next to
                # the analyzed "<stem>_output.geojson".
                seg_src = getattr(preproc_result, "segmentation_geojson_path", None)
                if seg_src and Path(seg_src).exists():
                    seg_dst = output_dir / f"{stem}_segmentation.geojson"
                    try:
                        shutil.copy2(seg_src, seg_dst)
                        trace_result.segmentation_geojson_path = seg_dst
                    except OSError as exc:
                        logger.warning("%s: failed to copy segmentation GeoJSON: %s", stem, exc)
                else:
                    logger.warning("%s: segmentation_geojson requested but no source file found", stem)

            if "wing_isolated_image" in outputs:
                wi_src = getattr(preproc_result, "wing_isolated_image_path", None)
                if wi_src and Path(wi_src).exists():
                    wi_dst = output_dir / Path(wi_src).name
                    try:
                        shutil.copy2(wi_src, wi_dst)
                        trace_result.wing_isolated_image_path = wi_dst
                    except OSError as exc:
                        logger.warning("%s: failed to copy wing-isolated image: %s", stem, exc)
                else:
                    # Quietly skip — Stage 2 simply wasn't enabled for this run.
                    pass

            stage2_slots[i] = trace_result
            if "csv" in outputs:
                if wing_result is not None:
                    csv_slots[i] = (stem, wing_result)
                # Always record the landmark path when CSV is requested — both
                # the augmenter (post-export_csv_batch) and the fast-path
                # writer rely on it. wing_result may be None on the fast path.
                lm_gj_path = preproc_result.landmarks_geojson_path
                if lm_gj_path is not None:
                    user_dist_landmark_paths[stem] = Path(lm_gj_path)

            elapsed = time.time() - t0
            _emit_progress(i, stem, f"done ({elapsed:.1f}s)")

        except InterruptedError:
            cancel_event.set()
            raise
        except GarbageRejection as e:
            # Quality filter aborted this wing — a clean, expected rejection, not a crash.
            # Record just the one-line reason (no traceback) and tag the failure with the
            # specific filter (solidity / fragmentation / uncalled vein tissue / missing
            # veins) so the GUI/log names what failed rather than a generic "quality".
            # The "Aborted by quality gate" prefix is what the GUI's
            # _classify_analysis_failure pattern-matches on to tag the failure as
            # category="gate" — same bucket as landmark confidence-gate aborts, so the
            # "Rerun failed (no quality gates)" button picks them up too.
            from identify_features.garbage_detector import filter_label

            elapsed = time.time() - t0
            stage = filter_label(e.verdict.filter_name)
            logger.info("Analysis aborted for %s (%.1fs) [%s]: %s", stem, elapsed, stage, e)
            stage2_slots[i] = TraceResult(
                image_path=preproc_result.image_path,
                error=f"Aborted by quality gate ({stage}): {e}",
                error_stage=stage,
            )
        except Exception as e:
            elapsed = time.time() - t0
            logger.exception("Analysis failed for %s (%.1fs)", stem, elapsed)
            stage2_slots[i] = TraceResult(
                image_path=preproc_result.image_path,
                error=f"{e}\n{traceback.format_exc()}",
                error_stage="analysis",
            )
        finally:
            # Stage 2 attempt finished (success or per-image error) — tell
            # the host this image is "done" for resume bookkeeping.
            #
            # Gate: signal only when stage2_slots[i] was populated. A None
            # slot means either (a) the pause-event short-circuit at the
            # top of _analyze_one fired before any work started, or (b) an
            # InterruptedError unwound through the cancel path (in which
            # case the manifest gets discarded anyway). Either way there's
            # nothing to record. Notably this is NOT gated on pause_event
            # itself — if the user clicked Pause mid-image, the image
            # still finishes cleanly and its artifacts are on disk, so
            # the manifest needs the completion entry to avoid
            # re-processing on resume.
            slot = stage2_slots[i]
            if slot is not None:
                success = slot.error is None
                error_text = "" if success else (slot.error or "Stage 2 failed")
                _signal_complete(preproc_result.image_path.name, success, error_text)

    interrupted = False
    paused = False
    if max_workers <= 1:
        for i, preproc_result in enumerate(successful_preproc):
            if pause_event is not None and pause_event.is_set():
                paused = True
                break
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
                if pause_event is not None and pause_event.is_set():
                    paused = True
                    # Don't cancel — let the in-flight workers finish their
                    # current images cleanly. _analyze_one's own pause check
                    # stops any not-yet-started worker from doing real work.

    for tr in stage2_slots:
        if tr is not None:
            results.append(tr)

    batch_results: list[tuple[str, object]] = [entry for entry in csv_slots if entry is not None]

    if interrupted:
        raise InterruptedError("Cancelled by user")

    # Pause is a clean stop — return whatever images finished without
    # raising. The caller (GUI worker) sees a normal return and knows to
    # write/update the manifest with status=paused. CSV is still written
    # below for the images that did complete in this slice; resume merges
    # via Phase 3's append-only logic (until then a resume run replaces
    # rather than merges, so the consolidated CSV reflects only the latest
    # slice — per-image GeoJSONs / overlays are preserved either way).
    if paused:
        logger.info("Pipeline paused: %d image(s) completed in this slice", len(batch_results))

    # --- Batch CSV ---
    if "csv" in outputs:
        # csv_filename_override is the rerun-failed "Write to new CSV"
        # branch. When set, write to that filename and skip the
        # resume-merge entirely — the user explicitly asked for a fresh,
        # standalone CSV. When unset, default name + normal merge path.
        csv_path = output_dir / (csv_filename_override or "measurements.csv")
        # Resume support: if we're skipping previously-completed images,
        # park the prior measurements.csv aside so its rows can be folded
        # back in after the new slice writes. The merge happens at the
        # end of this block (covering both the fast path and the normal
        # export_csv_batch path).
        # The override-branch deliberately skips this — it's writing a
        # new file by name, so there's nothing to fold in.
        csv_append_source: Optional[Path] = None
        if not csv_filename_override and skip_image_basenames and csv_path.is_file():
            csv_append_source = csv_path.with_suffix(".csv.append_source")
            try:
                csv_path.replace(csv_append_source)
            except OSError as exc:
                logger.warning("CSV resume: cannot move %s aside: %s", csv_path, exc)
                csv_append_source = None
        if fast_csv_path:
            # Fast path: identifyFeatures did not run, so there is no
            # measurements.csv to augment. Write one from scratch using only
            # the landmark coordinates + configured pairs.
            try:
                from measurement_maker import pairs_from_dicts, write_distances_csv

                pairs = pairs_from_dicts(user_landmark_distances)
                if pairs:
                    write_distances_csv(
                        csv_path,
                        user_dist_landmark_paths,
                        pairs,
                        um_per_px=scale,
                    )
            except Exception:
                logger.exception("Fast-path: failed to write user-distance CSV")
        elif batch_results:
            try:
                export_csv_batch(batch_results, csv_path, um_per_px=scale, groups=csv_measurement_groups)
                logger.info("Batch CSV: %s (%d wings)", csv_path, len(batch_results))
            except Exception:
                logger.exception("Failed to write batch CSV")
            else:
                if user_landmark_distances:
                    try:
                        from measurement_maker import augment_csv_with_user_distances, pairs_from_dicts

                        pairs = pairs_from_dicts(user_landmark_distances)
                        if pairs:
                            augment_csv_with_user_distances(
                                csv_path,
                                user_dist_landmark_paths,
                                pairs,
                                um_per_px=scale,
                            )
                    except Exception:
                        logger.exception("Failed to add user-defined distance columns to CSV")

        # Fold prior-slice rows back in for resume cases. Done after any
        # post-processing (user-distance augmentation) so the appended rows
        # are matched against the final column set written by this slice.
        if csv_append_source is not None and csv_append_source.is_file():
            try:
                from TRACE.run_state import merge_resume_csv

                appended = merge_resume_csv(csv_path, csv_append_source)
                if appended:
                    logger.info("CSV resume: folded %d row(s) from prior slice into %s", appended, csv_path)
                csv_append_source.unlink(missing_ok=True)
            except Exception:
                logger.exception("CSV resume: merge failed; leaving %s in place", csv_append_source)

    return results
