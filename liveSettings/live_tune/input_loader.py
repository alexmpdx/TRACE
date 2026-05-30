"""Resolve a live-tuning input from either existing GeoJSONs or a raw image.

Two entry points produce the same :class:`InputBundle`:

* :func:`load_from_geojsons` — fast path, no DL models. Parses a detection
  GeoJSON + landmarks GeoJSON (+ optional image) straight into S1 results.
* :func:`load_from_raw_image` — runs ``preprocessing.process_single_image``
  ONCE (the slow DL step) into a temp/output dir, then parses its outputs.
  The caller owns ``predictor_cache`` / ``model_cache`` dicts so the landmark
  and segmentation models load only once across repeated sample loads.

``setup_sibling_paths`` mirrors TRACE/run_gui.py:_setup_paths so the heavy
cross-module imports resolve when this tool runs standalone.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SIBLINGS = (
    "",
    "HingeChopper",
    "modelTOjson",
    "identifyFeatures",
    "preprocessing",
    "measurementMaker",
    "wingIsolator",
    "resolutionAdjust",
    "scaleEstimator",
    "wingRotator",
    "LandmarkLocator",
    "TRACE",
)


def setup_sibling_paths(repo_root: Optional[Path] = None) -> Path:
    """Add sibling module dirs to sys.path (idempotent). Returns the repo root."""
    if repo_root is None:
        # liveSettings/live_tune/input_loader.py -> repo root is parents[2]
        repo_root = Path(__file__).resolve().parents[2]
    for sub in _SIBLINGS:
        p = str(repo_root / sub) if sub else str(repo_root)
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo_root


@dataclass
class InputBundle:
    """Everything :meth:`LiveTuneSession.set_input` needs, plus metadata."""

    base_image: np.ndarray
    vein_polys: list
    intervein_polys: list
    landmarks_raw: dict
    wing_outline: object  # shapely Polygon | None
    image_shape: tuple[int, int]
    specimen_id: str
    um_per_px: Optional[float] = None
    detection_geojson: Optional[Path] = None
    landmarks_geojson: Optional[Path] = None
    image_path: Optional[Path] = None
    source: str = ""  # human-readable provenance for the UI
    # Resolution factor already applied to base_image/polys/landmarks
    # (1.0 = full res). scale_bundle() sets this; the session reads it back to
    # keep micron thresholds consistent.
    preview_scale: float = 1.0


def _parse(detection_geojson: Path, landmarks_geojson: Path,
           image_path: Optional[Path]) -> tuple:
    """Run S1 (parse + outline + image read). Imports lazily."""
    from identify_features.models.geojson_io import (
        _compute_wing_outline,
        load_detection_geojson,
        load_landmarks_geojson,
    )
    from identify_features.utils.psd_loader import imread_any

    vein_polys, intervein_polys = load_detection_geojson(detection_geojson)
    landmarks = load_landmarks_geojson(landmarks_geojson)
    wing_outline = _compute_wing_outline(vein_polys + intervein_polys)

    if image_path is not None and Path(image_path).exists():
        img = imread_any(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_shape = (img.shape[0], img.shape[1])
    else:
        from shapely.ops import unary_union

        bounds = unary_union(vein_polys + intervein_polys).bounds
        image_shape = (int(bounds[3]) + 100, int(bounds[2]) + 100)
        # Neutral grey canvas so the overlay is still legible without a photo.
        img = np.full((image_shape[0], image_shape[1], 3), 40, dtype=np.uint8)
        logger.info("No image supplied; using a grey canvas of %s", image_shape)
    return img, vein_polys, intervein_polys, landmarks, wing_outline, image_shape


def load_from_geojsons(
    detection_geojson: Path,
    landmarks_geojson: Path,
    image_path: Optional[Path] = None,
    um_per_px: Optional[float] = None,
) -> InputBundle:
    """Build an InputBundle from already-produced GeoJSONs (no DL models)."""
    detection_geojson = Path(detection_geojson)
    landmarks_geojson = Path(landmarks_geojson)
    img, vp, ip, lms, outline, shape = _parse(detection_geojson, landmarks_geojson, image_path)
    specimen_id = detection_geojson.stem.replace("_detections", "").replace("_segmentation", "")
    return InputBundle(
        base_image=img,
        vein_polys=vp,
        intervein_polys=ip,
        landmarks_raw=lms,
        wing_outline=outline,
        image_shape=shape,
        specimen_id=specimen_id,
        um_per_px=um_per_px,
        detection_geojson=detection_geojson,
        landmarks_geojson=landmarks_geojson,
        image_path=Path(image_path) if image_path else None,
        source=f"GeoJSONs: {detection_geojson.name}",
    )


def load_from_raw_image(
    image_path: Path,
    output_dir: Path,
    landmark_checkpoint: Path,
    segmentation_model_dir: Path,
    predictor_cache: Optional[dict] = None,
    model_cache: Optional[dict] = None,
    um_per_px: Optional[float] = None,
    device: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    wing_model_dir: Optional[Path] = None,
    wing_expand_fraction: float = 0.05,
    do_rotation: bool = False,
    rotation_mirror_correct: bool = False,
    target_um_per_px: Optional[float] = None,
    include_unreliable_landmarks: bool = True,
) -> InputBundle:
    """Run preprocessing ONCE on a raw image, then build an InputBundle.

    ``predictor_cache`` / ``model_cache`` should be long-lived dicts owned by
    the caller so the DL models load only once across multiple sample loads.

    The preprocessing-stage knobs mirror TRACE's run options so the preview's
    image reflects them: ``wing_model_dir`` (None disables wing isolation),
    ``wing_expand_fraction``, ``do_rotation`` / ``rotation_mirror_correct``
    (wingRotator), and ``target_um_per_px`` (Stage-1 rescale target).

    ``include_unreliable_landmarks`` defaults to True for the live preview
    so the tuning loop doesn't dead-end on borderline-confidence landmarks
    (issue #17). The user is in Advanced Settings adjusting parameters —
    raising LowConfidenceLandmarkError makes the preview unusable on the
    very images most worth tuning for. The production pipeline keeps the
    strict default.

    Raises RuntimeError if preprocessing reports an error.
    """
    from preprocessing.pipeline import process_single_image

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = process_single_image(
        image_path=image_path,
        output_dir=output_dir,
        landmark_checkpoint=Path(landmark_checkpoint) if landmark_checkpoint else None,
        segmentation_model_dir=Path(segmentation_model_dir) if segmentation_model_dir else None,
        stages=(True, True, True),
        predictor_cache=predictor_cache if predictor_cache is not None else {},
        model_cache=model_cache if model_cache is not None else {},
        input_um_per_px=um_per_px,
        device=device,
        progress_callback=progress_cb,
        wing_model_dir=Path(wing_model_dir) if wing_model_dir else None,
        wing_expand_fraction=wing_expand_fraction,
        do_rotation=do_rotation,
        rotation_mirror_correct=rotation_mirror_correct,
        target_um_per_px=target_um_per_px,
        include_unreliable_landmarks=include_unreliable_landmarks,
    )

    if result.error:
        raise RuntimeError(f"Preprocessing failed at {result.error_stage}: {result.error}")
    if not result.segmentation_geojson_path or not result.landmarks_geojson_path:
        raise RuntimeError("Preprocessing produced no detection/landmarks GeoJSON")

    # The GeoJSONs are in the OUTPUT image's pixel grid. If preprocessing
    # rescaled (rescale_factor != 1.0), output pixels = input pixels *
    # rescale_factor, so the effective µm/px of the output is um_per_px /
    # rescale_factor. The rotated/segmented image written to output_dir is the
    # one those GeoJSONs match — load that, not the raw input.
    eff_um_per_px = um_per_px
    rf = getattr(result, "rescale_factor", 1.0) or 1.0
    if um_per_px is not None and rf != 1.0:
        eff_um_per_px = um_per_px / rf
    out_image = result.rotated_image_path or result.processed_image_path or image_path

    bundle = load_from_geojsons(
        detection_geojson=result.segmentation_geojson_path,
        landmarks_geojson=result.landmarks_geojson_path,
        image_path=out_image,
        um_per_px=eff_um_per_px,
    )
    bundle.source = f"Image: {image_path.name} (preprocessed)"
    return bundle


def scale_bundle(bundle: InputBundle, scale: float) -> InputBundle:
    """Return a copy of ``bundle`` downscaled by ``scale`` (1.0 = full res).

    Downscaling the image and all geometry by the same factor makes the
    expensive raster stages (skeleton build, vein trace) cost ~``scale**2`` as
    much, which is the lever that makes the preview feel live on large wings.
    The session divides ``um_per_px`` by ``scale`` so micron thresholds stay
    matched; here we only transform pixel-space geometry + the image.

    ``scale`` >= ~1.0 short-circuits and returns the bundle unchanged.
    """
    if scale is None or scale >= 0.999:
        return replace(bundle, preview_scale=1.0)

    import cv2
    from shapely.affinity import scale as _affine_scale

    def _g(geom):
        if geom is None:
            return None
        return _affine_scale(geom, xfact=scale, yfact=scale, origin=(0, 0))

    h, w = bundle.image_shape
    nw = max(1, round(w * scale))
    nh = max(1, round(h * scale))
    img = cv2.resize(bundle.base_image, (nw, nh), interpolation=cv2.INTER_AREA)

    vein_polys = [_g(p) for p in bundle.vein_polys]
    intervein_polys = [_g(p) for p in bundle.intervein_polys]
    wing_outline = _g(bundle.wing_outline)
    landmarks_raw = {k: replace(lm, point=_g(lm.point)) for k, lm in bundle.landmarks_raw.items()}

    return replace(
        bundle,
        base_image=img,
        vein_polys=vein_polys,
        intervein_polys=intervein_polys,
        landmarks_raw=landmarks_raw,
        wing_outline=wing_outline,
        image_shape=(nh, nw),
        preview_scale=scale,
    )


def apply_to_session(bundle: InputBundle, session) -> None:
    """Feed an InputBundle into a LiveTuneSession."""
    session.set_input(
        bundle.base_image,
        bundle.vein_polys,
        bundle.intervein_polys,
        bundle.landmarks_raw,
        bundle.wing_outline,
        bundle.image_shape,
        preview_scale=getattr(bundle, "preview_scale", 1.0),
    )
