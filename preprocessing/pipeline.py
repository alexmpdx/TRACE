"""
Preprocessing pipeline — orchestrates resolutionAdjust, wingIsolator,
LandmarkLocator, HingeChopper, modelTOjson, and wingRotator.

Processes a folder of wing images through six stages (Stages 2 and 6 optional):
  1. Resolution adjust (resolutionAdjust)
  2. Wing isolation (wingIsolator, optional)
  3. Landmark detection (LandmarkLocator)
  4. Hinge removal (HingeChopper)
  5. Segmentation to GeoJSON (modelTOjson)
  6. Wing rotation (wingRotator, optional)

Each stage can be run independently or as part of the full pipeline.
"""

import contextvars
import json
import os
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Per-image log context. Workers set this to the image filename so log handlers
# can prepend "[<name>]" to every record. Lives here because preprocessing is
# the deepest layer shared by TRACE and other downstream callers.
current_image: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("current_image_name", default=None)


def _clean_stem(image_path: Path) -> str:
    """Strip an .ome.tif / .ome.tiff compound suffix to get a clean stem.

    Path.stem returns "<name>.ome" for "<name>.ome.tif"; this collapses both the
    ``.ome`` and the trailing extension so intermediate filenames stay readable
    (e.g. ``<name>_chopped.tif`` instead of ``<name>.ome_chopped.tif``).
    """
    name = image_path.name
    low = name.lower()
    if low.endswith(".ome.tif"):
        return name[: -len(".ome.tif")]
    if low.endswith(".ome.tiff"):
        return name[: -len(".ome.tiff")]
    return image_path.stem


IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".psd",
    ".psb",
    ".heic",
    ".heif",
    ".svg",
    ".raw",
    ".dng",
    ".nef",
    ".cr2",
    ".cr3",
    ".arw",
    ".raf",
    ".orf",
    ".pef",
    ".rw2",
    ".srw",
    # Microscopy formats — converted to OME-TIFF in process_single_image.
    ".czi",
    ".nd2",
    ".lif",
    ".lsm",
}


def _subpath_target_name(path: Path, root: Path) -> str:
    """Flatten a (possibly nested) image path into a single filename.

    Joins all components of ``path.relative_to(root)`` with ``_``, so e.g.
    ``root=dir1, path=dir1/folder1/sub/img.tif`` returns ``folder1_sub_img.tif``.
    Top-level images (relative path has only the basename) round-trip unchanged.
    Used by recursive discovery to prevent basename collisions when images live
    in subdirectories of the user's input folder.
    """
    return "_".join(path.relative_to(root).parts)


def discover_images(folder: Path, recursive: bool = False) -> list[Path]:
    """Find supported image files in a folder, skipping hidden/resource-fork files.

    When recursive=True, walk all subdirectories; otherwise only the top level.
    Hidden files/dirs and macOS resource-fork files (._*) are always skipped.
    Subdirectories that can't be read (e.g. macOS TCC-protected paths like
    ``~/Desktop`` without Files-and-Folders permission) are silently skipped
    rather than aborting the entire scan.
    """

    def _hidden(name: str) -> bool:
        return name.startswith(".") or name.startswith("._")

    images: list[Path] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(folder, onerror=lambda _e: None):
            # Prune hidden subdirs in-place so os.walk doesn't descend into them.
            dirnames[:] = [d for d in dirnames if not _hidden(d)]
            for name in filenames:
                if _hidden(name):
                    continue
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    images.append(Path(dirpath) / name)
    else:
        try:
            entries = list(folder.iterdir())
        except (PermissionError, OSError):
            return []
        for f in entries:
            if _hidden(f.name):
                continue
            try:
                if not f.is_file():
                    continue
            except (PermissionError, OSError):
                continue
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(f)
    return sorted(images)


# ---------------------------------------------------------------------------
# Stage 3: Landmark detection
# ---------------------------------------------------------------------------
def _predictor_from_cache(
    checkpoint_path: Path,
    predictor_cache: dict,
    confidence_override: Optional[dict] = None,
):
    """Lazily build/cache a predictor for the given checkpoint or fold folder.

    Cache key is `(checkpoint_path, override_signature)` so swapping the override
    rebuilds the predictor — necessary because confidence_override is applied at
    construction time and stored on the predictor's gate_config.
    """
    from landmark_locator import make_predictor

    cp_str = str(checkpoint_path)
    override_sig = json.dumps(confidence_override or {}, sort_keys=True)
    cache_key = (cp_str, override_sig)
    if predictor_cache.get("key") != cache_key:
        lock = predictor_cache.setdefault("_lock", threading.Lock())
        with lock:
            if predictor_cache.get("key") != cache_key:
                predictor_cache["predictor"] = make_predictor(checkpoint_path, confidence_override=confidence_override)
                predictor_cache["key"] = cache_key
                predictor_cache["checkpoint"] = cp_str  # back-compat for callers that read this
    return predictor_cache["predictor"]


def _shape_predict_result(predictor, result: dict) -> tuple[dict, dict]:
    """Convert a raw predictor result dict into (landmarks, metadata) keyed by GeoJSON names."""
    landmark_to_geojson = {v: k for k, v in predictor.geojson_to_landmark.items()}
    landmarks: dict = {}
    metadata: dict = {}
    for internal_name in predictor.landmark_order:
        geojson_name = landmark_to_geojson.get(internal_name, internal_name)
        metadata[geojson_name] = {
            "reliable": bool(result["reliable"].get(internal_name, True)),
            "gate_reason": result["gate_reason"].get(internal_name, ""),
            "confidence": float(result["confidences"].get(internal_name, 0.0)),
            "sharpness": float(result["sharpness"].get(internal_name, 0.0)),
            "second_peak_ratio": float(result["second_peak_ratio"].get(internal_name, 0.0)),
        }
        if internal_name in result["landmarks"]:
            landmarks[geojson_name] = result["landmarks"][internal_name]
    return landmarks, metadata


def predict_landmarks_for_paths(
    paths: list[Path],
    checkpoint_path: Path,
    predictor_cache: Optional[dict] = None,
    *,
    include_unreliable_landmarks: bool = False,
    batch_size: Optional[int] = None,
    confidence_override: Optional[dict] = None,
    pause_event: Optional[threading.Event] = None,
) -> dict:
    """Predict landmarks for many images in batches.

    Returns {path: {"landmarks", "metadata", "error"}} where `error` is None on success
    or a `LowConfidenceLandmarkError` instance when a core landmark failed the gate.
    Caller decides whether to abort the image based on `error`.

    `batch_size`:
      - None → auto-pick via `landmark_locator.auto_batch_size`.
      - 1    → process one image at a time (matches single-fold predict() semantics).
      - >1   → batch model forward passes.

    ``pause_event``: when set, the loop stops between mini-batches and returns
    the partial result dict. Paths past the pause point simply have no entry
    (the caller's per-image fallback handles them — but if the caller is also
    pause-aware it'll skip those images entirely instead).
    """
    from landmark_locator import LowConfidenceLandmarkError, auto_batch_size
    from landmark_locator.data.psd_loader import imread_any

    out: dict = {}
    # Honor pause before the (potentially seconds-long) model load on the
    # first call — otherwise a user who clicks Pause immediately on launch
    # still waits through the checkpoint deserialization + warm-up.
    if pause_event is not None and pause_event.is_set():
        return out

    if predictor_cache is None:
        predictor_cache = {}
    predictor = _predictor_from_cache(checkpoint_path, predictor_cache, confidence_override=confidence_override)

    if not paths:
        return out

    bs = batch_size if batch_size and batch_size > 0 else auto_batch_size(len(paths))
    for chunk_start in range(0, len(paths), bs):
        # Pause check is between mini-batches — a forward pass on the GPU
        # can't be cleanly interrupted mid-flight, but each mini-batch is
        # sized to ``max_workers`` (typically 1-8 images) so worst-case
        # latency is a sub-second batch's worth of work.
        if pause_event is not None and pause_event.is_set():
            return out
        chunk_paths = paths[chunk_start : chunk_start + bs]
        chunk_images = []
        valid_paths = []
        for p in chunk_paths:
            img = imread_any(p)
            if img is None:
                out[p] = {"landmarks": {}, "metadata": {}, "error": IOError(f"Failed to load image: {p}")}
                continue
            chunk_images.append(img)
            valid_paths.append(p)
        if not chunk_images:
            continue
        results = predictor.predict_batch(
            chunk_images,
            include_unreliable=include_unreliable_landmarks,
            raise_on_core_fail=False,
        )
        for p, r in zip(valid_paths, results):
            if r.get("error") is not None:
                out[p] = {"landmarks": {}, "metadata": {}, "error": r["error"]}
                continue
            try:
                landmarks, metadata = _shape_predict_result(predictor, r)
                out[p] = {"landmarks": landmarks, "metadata": metadata, "error": None}
            except LowConfidenceLandmarkError as exc:
                out[p] = {"landmarks": {}, "metadata": {}, "error": exc}
    return out


def run_landmarks(
    image_path: Path,
    checkpoint_path: Path,
    predictor_cache: dict,
    *,
    include_unreliable_landmarks: bool = False,
    confidence_override: Optional[dict] = None,
) -> tuple[dict, dict]:
    """Predict landmarks. Returns (landmarks, metadata) keyed by GeoJSON names.

    checkpoint_path may be a single `.pt` file (single-fold prediction) or a directory
    containing `best_fold*.pt` (5-fold ensemble — averaged heatmaps, more robust).

    metadata[name] -> {reliable, gate_reason, confidence, sharpness, second_peak_ratio}.

    `confidence_override`: optional gate-config override (same shape as the
    `confidence:` block in `configs/default.yaml`). Must be threaded in by the
    per-image fallback so it stays consistent with the batch-prefetch path —
    otherwise the cache rebuilds a predictor with the model's bundled core
    landmarks and aborts images the user explicitly cleared from the gate.

    Raises landmark_locator.LowConfidenceLandmarkError if a core landmark fails the gate.
    """
    predictor = _predictor_from_cache(checkpoint_path, predictor_cache, confidence_override=confidence_override)
    result = predictor.predict_from_path(image_path, include_unreliable=include_unreliable_landmarks)
    return _shape_predict_result(predictor, result)


def landmarks_to_geojson(
    landmarks: dict,
    metadata: Optional[dict] = None,
    fc_props: Optional[dict] = None,
) -> dict:
    """Convert landmarks dict to GeoJSON FeatureCollection.

    metadata (if provided) maps name -> {reliable, gate_reason, confidence, sharpness,
    second_peak_ratio} and is embedded into each feature's properties.

    fc_props (if provided) is merged into the top-level FeatureCollection
    under a ``properties`` key — used to persist per-image pipeline state
    that isn't attached to any single landmark (e.g. the effective µm/px
    the pipeline used for this image after resolutionAdjust or metadata
    auto-detect, needed to convert landmark distances to µm downstream).
    """
    features = []
    metadata = metadata or {}
    for name, (x, y) in landmarks.items():
        props: dict = {"classification": {"name": name}}
        meta = metadata.get(name)
        if meta is not None:
            props.update(
                {
                    "reliable": meta.get("reliable", True),
                    "gate_reason": meta.get("gate_reason", ""),
                    "confidence": meta.get("confidence", 0.0),
                    "sharpness": meta.get("sharpness", 0.0),
                    "second_peak_ratio": meta.get("second_peak_ratio", 0.0),
                }
            )
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [x, y]},
                "properties": props,
            }
        )
    fc: dict = {"type": "FeatureCollection", "features": features}
    if fc_props:
        fc["properties"] = dict(fc_props)
    return fc


def save_landmarks_geojson(
    landmarks: dict,
    output_path: Path,
    metadata: Optional[dict] = None,
    fc_props: Optional[dict] = None,
) -> None:
    """Save landmarks dict as GeoJSON file."""
    fc = landmarks_to_geojson(landmarks, metadata, fc_props=fc_props)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)


# Manual override sidecars (written by TRACE's landmark inspector) live in a
# dedicated subfolder next to the source image, rather than loose beside it, so
# they stay organized and aren't accidentally deleted when tidying the image
# folder. Keyed on the image's OWN parent (not the run's output folder) so an
# override is found regardless of which output folder a later run uses. The
# ``find_*`` helpers fall back to the pre-0.2.x loose location so overrides saved
# by older builds keep working.
OVERRIDE_SUBDIR = "manual_overrides"


def overrides_dir(image_path: Path) -> Path:
    """Directory holding an image's manual override sidecars."""
    return Path(image_path).parent / OVERRIDE_SUBDIR


def landmarks_override_path(image_path: Path) -> Path:
    """Canonical write location for an image's landmark override sidecar."""
    return overrides_dir(image_path) / f"{Path(image_path).stem}_landmarks_override.geojson"


def segmentation_override_path(image_path: Path) -> Path:
    """Canonical write location for an image's segmentation override sidecar."""
    return overrides_dir(image_path) / f"{Path(image_path).stem}_segmentation_override.geojson"


def _relocate_legacy_override(legacy: Path, new: Path) -> Path:
    """Best-effort one-time move of an older build's loose override into the
    ``manual_overrides/`` subfolder, so existing overrides migrate automatically
    as they're first accessed (by a run or the inspector) after the upgrade.

    Idempotent and safe: returns the new path once the file lives there, or the
    legacy path if the move can't happen (permissions, cross-device, a race) — a
    failed migration must never break override loading.
    """
    try:
        if new.exists():
            return new  # already migrated by a concurrent access; don't clobber
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(new))
        return new
    except Exception:
        return legacy if legacy.is_file() else (new if new.is_file() else legacy)


def find_landmarks_override(image_path: Path) -> Optional[Path]:
    """Return the landmark override to use, or ``None`` when none exists.

    Prefers the ``manual_overrides/`` subfolder; if only the legacy loose file
    next to the image exists, it is migrated there (see ``_relocate_legacy_override``)
    and the new path returned.
    """
    p = landmarks_override_path(image_path)
    if p.is_file():
        return p
    legacy = Path(image_path).parent / f"{Path(image_path).stem}_landmarks_override.geojson"
    return _relocate_legacy_override(legacy, p) if legacy.is_file() else None


def find_segmentation_override(image_path: Path) -> Optional[Path]:
    """Return the segmentation override to use, or ``None`` when none exists.

    Prefers the ``manual_overrides/`` subfolder; if only the legacy loose file
    next to the image exists, it is migrated there and the new path returned.
    """
    p = segmentation_override_path(image_path)
    if p.is_file():
        return p
    legacy = Path(image_path).parent / f"{Path(image_path).stem}_segmentation_override.geojson"
    return _relocate_legacy_override(legacy, p) if legacy.is_file() else None


def load_landmarks_override(path: Path) -> tuple[dict, dict]:
    """Load a manual landmark override GeoJSON into (landmarks, metadata).

    Written by TRACE's landmark inspector dialog into the image's
    ``manual_overrides/`` subfolder as
    ``<stem>_landmarks_override.geojson``. Tolerant of either the inspector's
    minimal schema (classification.name + coordinates) or the full Stage-3
    schema (confidence/sharpness/etc.).

    Returns:
        landmarks: ``{name: (x, y)}`` in pixel coordinates.
        metadata:  ``{name: {reliable, gate_reason, confidence, sharpness,
                   second_peak_ratio}}`` — defaults force the override through
                   any downstream confidence gate (reliable=True, confidence=1.0).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    landmarks: dict = {}
    metadata: dict = {}
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = feat.get("properties") or {}
        cls = props.get("classification") or {}
        name = cls.get("name") or props.get("name")
        if not name:
            continue
        name = str(name)
        landmarks[name] = (float(coords[0]), float(coords[1]))
        metadata[name] = {
            "reliable": props.get("reliable", True),
            "gate_reason": props.get("gate_reason", "manual override"),
            "confidence": props.get("confidence", 1.0),
            "sharpness": props.get("sharpness", 0.0),
            "second_peak_ratio": props.get("second_peak_ratio", 0.0),
        }
    return landmarks, metadata


# ---------------------------------------------------------------------------
# Stage 6: Wing rotation (runs after segmentation as the last preprocessing step)
# ---------------------------------------------------------------------------
def run_rotation(
    image_path: Path,
    landmarks_geojson_path: Path,
    output_dir: Path,
    landmarks: dict,
    extra_geojsons: Optional[list[Path]] = None,
    soft_reliability: bool = False,
    mirror_correct: bool = False,
):
    """Rotate image + landmarks geojson (+ optional extras) to canonical orientation.

    Returns (rotated_image_path, rotated_landmarks_geojson_path, rotated_landmarks_dict,
    rotation_result) on success, or None when there aren't enough reliable landmarks
    and the caller should pass the inputs through unchanged.
    """
    from wing_rotator import rotate_from_landmarks

    result = rotate_from_landmarks(
        image_path=image_path,
        landmarks_geojson_path=landmarks_geojson_path,
        output_dir=output_dir,
        extra_geojsons=extra_geojsons,
        soft_reliability=soft_reliability,
        mirror_correct=mirror_correct,
    )
    if result is None:
        return None

    M = result.affine
    rotated_landmarks: dict = {}
    for name, (x, y) in landmarks.items():
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        rotated_landmarks[name] = (float(nx), float(ny))
    return result.rotated_image_path, result.rotated_landmarks_path, rotated_landmarks, result


# ---------------------------------------------------------------------------
# Stage 4: Hinge chopping
# ---------------------------------------------------------------------------
def run_hinge_chop(image_path: Path, landmarks: dict, output_path: Path) -> Path:
    """Black out proximal hinge region. Returns path to chopped image."""
    from hinge_chopper import chop_hinge_from_landmarks

    chop_hinge_from_landmarks(str(image_path), landmarks, str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Stage 2: Wing isolation (optional)
# ---------------------------------------------------------------------------
def run_wing_isolation(
    image_path: Path,
    model_dir: Path,
    output_dir: Path,
    device,
    model_cache: dict,
    *,
    expand_fraction: float = 0.05,
    keep_intermediate_geojson: bool = False,
) -> tuple[Path, Path]:
    """Mask out non-main-wing pixels in `image_path` using a modelTOjson wing-id model.

    1. Run inference with the wing-identification model to get a vein/wing/background
       segmentation; convert to GeoJSON; filter for `properties.class == "wing"`.
    2. Hand the resulting shapely polygons to `wingIsolator.isolate_in_memory`,
       which picks the image-centered (or largest) wing, optionally splits merged
       wings via watershed, and dilates by `expand_fraction * sqrt(area)`.
    3. Apply the binary mask to the source image and write `<stem>_isolated<ext>`
       to `output_dir` (PSD inputs fall back to PNG via wingIsolator's writer).

    Raises:
        RuntimeError: when the wing model finds no "wing" features, or when
            wingIsolator cannot produce a valid polygon. Surfaced as
            `error_stage="wing_isolation"` by `process_folder`'s error handler.

    Returns:
        (isolated_image_path, wing_geojson_path_or_None) — GeoJSON path is
        returned only when `keep_intermediate_geojson` is True; otherwise None.
    """
    import cv2
    from modeltojson import mask_to_geojson, read_image, run_inference, save_geojson
    from shapely.geometry import MultiPolygon as _MultiPolygon
    from shapely.geometry import Polygon as _Polygon
    from shapely.geometry import mapping as _shapely_mapping
    from shapely.geometry import shape as _shape

    from preprocessing.psd_loader import imwrite_ome_tiff
    from wingIsolator import isolate_in_memory
    from wingIsolator.pipeline import apply_mask_to_image, write_masked_image

    model, metadata = _load_or_cache_modeltojson(model_dir, device, model_cache)

    image = read_image(str(image_path))
    seg_mask = run_inference(model, image, metadata, device)
    fc = mask_to_geojson(seg_mask, metadata["classes"], str(image_path))

    # Collect wing polygons. Filter by class name (exact "wing" or substring match).
    polygons: list = []
    for feat in fc.get("features", []):
        props = feat.get("properties") or {}
        cls = props.get("class") or (props.get("classification") or {}).get("name")
        if cls is None or "wing" not in str(cls).lower():
            continue
        try:
            shp = _shape(feat["geometry"])
        except Exception:
            continue
        if isinstance(shp, _Polygon) and not shp.is_empty:
            polygons.append(shp)
        elif isinstance(shp, _MultiPolygon):
            polygons.extend(p for p in shp.geoms if not p.is_empty)

    if not polygons:
        raise RuntimeError("wing isolation: no 'wing' features detected in image")

    out = isolate_in_memory(image, polygons, expand_fraction=expand_fraction)
    if out["status"] != "ok":
        raise RuntimeError(f"wing isolation: {out['status']}")

    masked = apply_mask_to_image(image, out["mask"], bg_value=0)
    stem = _clean_stem(image_path)
    ext = image_path.suffix
    # cv2/Pillow can't write PSD; coerce TIF outputs to OME-TIFF so we keep
    # multi-channel/metadata semantics; everything else falls back to write_masked_image.
    is_tiff = ext.lower() in (".tif", ".tiff", ".psd", ".psb") or ext.lower().endswith(".ome.tif")
    if is_tiff:
        # imwrite_ome_tiff expects cv2 BGR convention; modeltojson.read_image returns RGB.
        bgr = cv2.cvtColor(masked, cv2.COLOR_RGB2BGR) if masked.ndim == 3 else masked
        out_image_path = imwrite_ome_tiff(output_dir / f"{stem}_isolated", bgr)
    else:
        out_image_path = Path(write_masked_image(masked, output_dir / f"{stem}_isolated{ext}"))

    # Persist the isolated-wing polygon as a single-feature GeoJSON. Used downstream
    # by Stage 5 to drive modelTOjson's roi_mask (skips background tiles).
    geojson_out_path = output_dir / f"{stem}_wing.geojson"
    wing_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": _shapely_mapping(out["polygon"]),
                "properties": {"class": "wing"},
            }
        ],
    }
    save_geojson(wing_fc, str(geojson_out_path))

    return out_image_path, geojson_out_path


# ---------------------------------------------------------------------------
# Shared modelTOjson cache helper (multi-slot, dir-keyed)
# ---------------------------------------------------------------------------
def _load_or_cache_modeltojson(model_dir: Path, device, model_cache: dict) -> tuple:
    """Lazily load a modelTOjson model into a multi-slot, dir-keyed cache.

    model_cache layout::

        {"_lock": Lock(), "<model_dir_str>": {"model": ..., "metadata": ...}, ...}

    Multiple model directories can coexist without thrashing — used so the
    vein/intervein segmentation model and the wing-identification model
    share one cache without invalidating each other.
    """
    from modeltojson import load_model

    md_str = str(model_dir)
    entry = model_cache.get(md_str)
    if entry is None:
        lock = model_cache.setdefault("_lock", threading.Lock())
        with lock:
            entry = model_cache.get(md_str)
            if entry is None:
                model, metadata = load_model(md_str, device)
                entry = {"model": model, "metadata": metadata}
                model_cache[md_str] = entry
    return entry["model"], entry["metadata"]


# ---------------------------------------------------------------------------
# Stage 5: Segmentation
# ---------------------------------------------------------------------------
def run_segmentation(
    image_path: Path,
    model_dir: Path,
    device,
    model_cache: dict,
    *,
    roi_geojson_path: Optional[Path] = None,
) -> dict:
    """Run segmentation inference. Returns GeoJSON FeatureCollection dict.

    model_cache is a mutable dict used to cache the loaded model across calls.

    When `roi_geojson_path` is provided, modelTOjson skips tiles outside the ROI;
    out-of-ROI pixels are dropped from the returned features. Path must be in the
    same coordinate space as `image_path` (caller's responsibility).
    """
    from modeltojson import mask_to_geojson, read_image, roi_mask_from_geojson, run_inference

    model, metadata = _load_or_cache_modeltojson(model_dir, device, model_cache)

    image = read_image(str(image_path))
    roi_mask = None
    if roi_geojson_path is not None:
        roi_mask = roi_mask_from_geojson(str(roi_geojson_path), image.shape)
    mask = run_inference(model, image, metadata, device, roi_mask=roi_mask)
    fc = mask_to_geojson(mask, metadata["classes"], str(image_path))

    # Remove "hinge junk" features — residual hinge tissue not useful downstream
    fc["features"] = [f for f in fc["features"] if f["properties"].get("class") != "hinge junk"]

    return fc


def save_segmentation_geojson(geojson_fc: dict, output_path: Path) -> None:
    """Save segmentation GeoJSON to file."""
    from modeltojson import save_geojson

    save_geojson(geojson_fc, str(output_path))


def load_segmentation_override(path: Path) -> dict:
    """Load a manual vein/intervein override GeoJSON as a FeatureCollection dict.

    Written by TRACE's inspector dialog next to the source image as
    ``<stem>_segmentation_override.geojson``. The file is already a valid
    segmentation FeatureCollection (Polygon features with ``properties.class``),
    so it is returned verbatim for Stage 5 to write to the canonical location.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection")
    return data


def _scale_geojson_coords(fc: dict, factor: float) -> dict:
    """Scale every Polygon/MultiPolygon coordinate in a FeatureCollection.

    The inspector writes the segmentation override in original-input pixel
    space; when Stage 1 rescaled the image, the rest of the pipeline works in
    rescaled space, so the override is scaled by the same factor here before use.
    """

    def _ring(ring):
        return [[c[0] * factor, c[1] * factor] for c in ring]

    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        t = geom.get("type")
        coords = geom.get("coordinates") or []
        if t == "Polygon":
            geom["coordinates"] = [_ring(r) for r in coords]
        elif t == "MultiPolygon":
            geom["coordinates"] = [[_ring(r) for r in poly] for poly in coords]
    return fc


def _scale_landmarks(landmarks: dict, factor: float) -> dict:
    """Scale every ``{name: (x, y)}`` landmark coordinate by ``factor``.

    The landmark analogue of :func:`_scale_geojson_coords`. The inspector's
    Landmarks tab edits over the ORIGINAL image, so a manual landmark override
    is written in original-input pixel space; when Stage 1 rescaled the image
    the rest of the pipeline (Stages 4-6 and the Stage-2 analysis) works in
    rescaled space, so the override is scaled by the same factor before use.
    A ``factor`` of 1.0 returns the coordinates unchanged.
    """
    if factor == 1.0:
        return landmarks
    return {name: (x * factor, y * factor) for name, (x, y) in landmarks.items()}


# ---------------------------------------------------------------------------
# Pipeline result and orchestration
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    image_path: Path  # original input path; user-facing
    processed_image_path: Optional[Path] = None  # set when the input was copied under a flattened name (recursive runs)
    landmarks: Optional[dict] = None
    landmark_metadata: Optional[dict] = None
    landmarks_geojson_path: Optional[Path] = None
    wing_isolated_image_path: Optional[Path] = None
    wing_geojson_path: Optional[Path] = None
    chopped_image_path: Optional[Path] = None
    segmentation_geojson_path: Optional[Path] = None
    rotated_image_path: Optional[Path] = None
    rotated_landmarks_geojson_path: Optional[Path] = None
    rotation_angle_deg: Optional[float] = None
    rotation_rms_residual: Optional[float] = None
    rotation_n_landmarks: Optional[int] = None
    rotation_mirror_detected: Optional[bool] = None
    # Stage 1 (resolutionAdjust). When the input was rescaled, scale_factor is
    # `rescaled_pixels / original_pixels` and original_shape is the pre-rescale
    # (h, w). Downstream consumers multiply geometry coords by 1/scale_factor to
    # bring them back to the original pixel grid. Both stay at their defaults
    # (1.0, None) when no rescale happened.
    rescale_factor: float = 1.0
    original_shape: Optional[tuple[int, int]] = None
    # µm/px actually used for this image. When the caller passes
    # ``auto_detect_um_per_px=True`` and the image carried readable metadata
    # (TIFF resolution tags / OME-XML PhysicalSizeX), this is the metadata-
    # derived value; otherwise it falls back to whatever ``input_um_per_px``
    # was passed in. TRACE reads it downstream to build a per-image identify
    # config so each wing's measurements convert through its OWN scale.
    effective_um_per_px: Optional[float] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None
    stages_completed: list[str] = field(default_factory=list)
    # Set True when ``process_folder`` skipped this image because the caller's
    # pause_event fired before its turn. Distinct from ``error`` so the
    # orchestrator (TRACE/pipeline.py) can drop these from both the success
    # AND failure lists — the image was never attempted, not failed.
    paused: bool = False


def _auto_device():
    """Auto-detect best torch device: MPS > CUDA > CPU."""
    import torch

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def process_single_image(
    image_path: Path,
    output_dir: Path,
    landmark_checkpoint: Optional[Path] = None,
    segmentation_model_dir: Optional[Path] = None,
    stages: tuple[bool, bool, bool] = (True, True, True),
    predictor_cache: Optional[dict] = None,
    model_cache: Optional[dict] = None,
    device=None,
    keep_chopped: bool = False,
    progress_callback=None,
    include_unreliable_landmarks: bool = False,
    prefetched_landmarks: Optional[dict] = None,
    wing_model_dir: Optional[Path] = None,
    wing_expand_fraction: float = 0.05,
    keep_intermediates: bool = False,
    target_name: Optional[str] = None,
    do_rotation: bool = True,
    rotation_mirror_correct: bool = False,
    gate_override: Optional[dict] = None,
    input_um_per_px: Optional[float] = None,
    target_um_per_px: Optional[float] = None,
    rescale_tolerance_low: float = 0.85,
    rescale_tolerance_high: float = 1.15,
    auto_detect_um_per_px: bool = False,
) -> PipelineResult:
    """Run selected pipeline stages on a single image.

    Args:
        stages: (landmarks, hinge_chop, segmentation) booleans.
        progress_callback: callable(stage_name: str, detail: str)
        include_unreliable_landmarks: when True, landmarks that fail the LandmarkLocator
            confidence gate are still written to the output GeoJSON (marked reliable=false).
        prefetched_landmarks: optional dict of `{path: {"landmarks", "metadata", "error"}}`
            populated upstream by `predict_landmarks_for_paths`. When supplied and the
            current image_path has an entry, the landmark forward pass is skipped and the
            cached result is used directly. A non-None `error` re-raises so the caller
            handles it the same way it would handle a synchronous failure.
        wing_model_dir: optional Stage 2. When provided, run modelTOjson with this wing
            -identification model, feed the result into wingIsolator, and rebind
            image_path to the masked image so all downstream stages see a single wing.
            None disables the stage entirely.
        wing_expand_fraction: Stage 2 buffer (fraction of sqrt(polygon area)).
        keep_intermediates: when True, the wing GeoJSON intermediate is also written
            alongside the masked image as <stem>_wing.geojson.
        do_rotation: when True (default), run wingRotator as the last preprocessing step (after segmentation) to
            align the image to a canonical right-side-up, distal-right orientation.
            Skipped silently when there aren't enough reliable landmarks.
        rotation_mirror_correct: when True AND a wing is detected as opposite chirality,
            apply a vertical reflection on top of the rotation so the wing ends up
            distal-right AND anterior-up (at the cost of flipping biological chirality).
            Default False keeps chirality and lets such wings end up distal-left.
    """
    do_landmarks, do_hinge, do_segment = stages
    result = PipelineResult(image_path=image_path)
    # Captured before any stage rebinds image_path (resolution rescale, wing
    # isolation). The manual landmark override sidecar lives next to the
    # original input the user picked, keyed on its untouched stem.
    original_input_path = image_path

    if predictor_cache is None:
        predictor_cache = {}
    if model_cache is None:
        model_cache = {}
    if device is None:
        device = _auto_device()

    # Set per-image log context so handlers can prefix records with [<image>].
    # No reset: every worker calls set() at the top, so the next iteration on the
    # same thread overrides the value before any log line can use it.
    current_image.set(image_path.name)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original image to output. When `target_name` is supplied (recursive runs
    # flatten subpaths into a single basename to avoid collisions), use it as the
    # destination filename and rebind image_path so every downstream stem derives
    # from the disambiguated name.
    dest_basename = target_name if target_name else image_path.name
    dest_image = output_dir / dest_basename
    if not dest_image.exists() or dest_image != image_path:
        shutil.copy2(image_path, dest_image)
    if target_name:
        result.processed_image_path = dest_image
        image_path = dest_image

    # Coerce non-cv2-native and lossy formats to a lossless TIF up front so every
    # downstream stage (LandmarkLocator, hinge_chopper, modelTOjson) sees a path
    # cv2.imread can handle. HEIC/HEIF, RAW, and SVG are treated the same way as
    # JPEG: decoded once via psd_loader.imread_any, re-saved as TIF. Microscopy
    # formats (CZI/ND2/LIF/LSM) get a richer round-trip via OME-TIFF.
    _COERCE_TO_TIF_EXTS = {
        ".jpg",
        ".jpeg",
        ".heic",
        ".heif",
        ".svg",
        ".raw",
        ".dng",
        ".nef",
        ".cr2",
        ".cr3",
        ".arw",
        ".raf",
        ".orf",
        ".pef",
        ".rw2",
        ".srw",
    }
    _MICROSCOPY_EXTS = {".czi", ".nd2", ".lif", ".lsm"}
    if image_path.suffix.lower() in _MICROSCOPY_EXTS:
        from preprocessing.psd_loader import convert_microscopy_to_ome_tiff

        image_path = convert_microscopy_to_ome_tiff(dest_image, output_dir)
    elif image_path.suffix.lower() in _COERCE_TO_TIF_EXTS:
        import cv2

        from preprocessing.psd_loader import imread_any, imwrite_ome_tiff

        img = imread_any(str(dest_image), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {dest_image}")
        image_path = imwrite_ome_tiff(output_dir / _clean_stem(image_path), img)

    # Per-image µm/px detection (opt-in). Read this image's own µm/px from its
    # metadata (TIFF XResolution + ResolutionUnit / OME-XML PhysicalSizeX) and
    # use that as this image's ``input_um_per_px`` — every downstream stage
    # (resolutionAdjust rescale + identifyFeatures measurements) then converts
    # through the image's OWN scale rather than a shared value. Silently falls
    # back to the caller-supplied ``input_um_per_px`` when the image has no
    # parseable metadata (which the GUI has already guaranteed is non-None via
    # its Run-time pre-flight check when this flag is on).
    if auto_detect_um_per_px:
        import logging as _logging

        _log = _logging.getLogger(__name__)
        try:
            from resolutionAdjust.auto_detect import _read_um_per_px_from_tiff

            detected = _read_um_per_px_from_tiff(image_path)
        except Exception:
            detected = None
        if detected is not None and detected > 0:
            input_um_per_px = float(detected)
            _log.info(
                "%s: detected µm/px = %.4f from image metadata",
                image_path.name,
                detected,
            )
        elif input_um_per_px is not None and input_um_per_px > 0:
            _log.info(
                "%s: no readable µm/px metadata, falling back to manual scale %.4f",
                image_path.name,
                input_um_per_px,
            )
        else:
            _log.info(
                "%s: no readable µm/px metadata and no fallback scale — "
                "measurements will be in pixels",
                image_path.name,
            )
    result.effective_um_per_px = input_um_per_px

    # Stage 1: resolutionAdjust. When the user has entered a Scale (input µm/px)
    # AND the selected model has a target µm/px set, rescale the image toward the
    # target if the ratio falls outside the tolerance band. The rescaled image
    # becomes the input to every downstream stage (Stage 2 wing isolation, Stage
    # 3 landmarks, Stage 4 hinge, Stage 5 segmentation, Stage 6 rotation).
    # Skipped silently when either value is missing or the ratio is in-band.
    if input_um_per_px is not None and input_um_per_px > 0 and target_um_per_px is not None and target_um_per_px > 0:
        from resolutionAdjust import adjust_resolution as _adjust_resolution

        if progress_callback:
            progress_callback("resolution_adjust", f"Checking resolution for {image_path.name}")
        try:
            ra_result = _adjust_resolution(
                image_path=image_path,
                input_um_per_px=input_um_per_px,
                target_um_per_px=target_um_per_px,
                output_dir=output_dir,
                tolerance_low=rescale_tolerance_low,
                tolerance_high=rescale_tolerance_high,
            )
        except Exception as exc:  # noqa: BLE001
            # Soft-fail: log and proceed with the original image so a bad
            # rescale config never aborts a pipeline run.
            import logging as _logging

            _logging.getLogger(__name__).warning(
                "resolutionAdjust failed for %s: %s — using original image",
                image_path.name,
                exc,
            )
            ra_result = None

        if ra_result is not None:
            result.rescale_factor = ra_result.scale_factor
            result.original_shape = ra_result.original_shape
            if ra_result.rescaled:
                image_path = ra_result.image_path
                result.processed_image_path = image_path
                result.stages_completed.append("resolution_adjust")

    # Stage 2: Wing isolation (optional). When wing_model_dir is set, mask out
    # non-main-wing pixels and rebind image_path so all downstream stages see
    # the single-wing image. result.image_path remains the original input.
    # Stash the un-masked image path so the final rotation step can produce a
    # rotated full-background canvas for Stage 2 overlays. Masking only zeros
    # pixels (no crop/translate), so the same affine that rotates the masked
    # image + GeoJSONs also rotates the un-masked image consistently.
    unmasked_image_path = image_path
    if wing_model_dir is not None:
        if progress_callback:
            progress_callback("wing_isolation", f"Isolating main wing for {image_path.name}")
        isolated_path, wing_geojson_path = run_wing_isolation(
            image_path,
            wing_model_dir,
            output_dir,
            device,
            model_cache,
            expand_fraction=wing_expand_fraction,
            keep_intermediate_geojson=keep_intermediates,
        )
        result.wing_isolated_image_path = isolated_path
        if wing_geojson_path is not None:
            result.wing_geojson_path = wing_geojson_path
        result.stages_completed.append("wing_isolation")
        image_path = isolated_path

    stem = _clean_stem(image_path)
    ext = image_path.suffix
    # cv2.imwrite cannot write .psd; coerce intermediate outputs to OME-TIFF when
    # input is PSD so we still keep multi-channel/metadata semantics on the way out.
    raster_ext = ".ome.tif" if ext.lower() in (".psd", ".psb") else ext

    # Stage 3: Landmarks
    landmarks = None
    landmark_metadata: Optional[dict] = None
    if do_landmarks:
        if progress_callback:
            progress_callback("landmarks", f"Predicting landmarks for {image_path.name}")
        # Manual override: if the user inspected/corrected this image's landmarks
        # via TRACE's landmark inspector dialog, a sidecar lives in the image's
        # manual_overrides/ subfolder (legacy: loose next to the input). Trust it
        # and skip the predictor (and its cache) entirely — the corrected
        # positions are written to the canonical landmarks_geojson below so
        # Stages 4-6 see the same data.
        override_path = find_landmarks_override(original_input_path)
        if override_path is not None:
            import logging as _logging

            _logging.getLogger(__name__).info("%s: using manual landmark override from %s", stem, override_path)
            landmarks, landmark_metadata = load_landmarks_override(override_path)
            # The override is in original-input pixel space (the inspector's
            # Landmarks tab edits over the ORIGINAL image), whereas Stages 4-6
            # and the Stage-2 analysis all operate in the rescaled space Stage 1
            # produced. If Stage 1 rescaled, bring the override into that space —
            # mirroring the segmentation-override short-circuit below (Stage 5).
            # Without this, a hand-corrected landmark is misplaced by
            # rescale_factor on every rescaling run (silently corrupting exactly
            # the images the user took the trouble to correct).
            landmarks = _scale_landmarks(landmarks, result.rescale_factor or 1.0)
        else:
            cached = None
            if prefetched_landmarks is not None:
                # Try the original input path first, then the (post-JPEG-conversion) path.
                cached = prefetched_landmarks.get(image_path)
                if cached is None:
                    # Caller may have keyed by the original input before JPEG/PSD conversion.
                    cached = prefetched_landmarks.get(Path(str(image_path).replace(raster_ext, ext)))
            if cached is not None:
                if cached.get("error") is not None:
                    raise cached["error"]
                landmarks = cached["landmarks"]
                landmark_metadata = cached["metadata"]
            else:
                landmarks, landmark_metadata = run_landmarks(
                    image_path,
                    landmark_checkpoint,
                    predictor_cache,
                    include_unreliable_landmarks=include_unreliable_landmarks,
                    confidence_override=gate_override,
                )
        result.landmarks = landmarks
        result.landmark_metadata = landmark_metadata
        lm_path = output_dir / f"{stem}_landmarks.geojson"
        # Persist the per-image effective µm/px so downstream tools that
        # only get the landmarks geojson (e.g. tools/recover_landmark_csv.py,
        # or a future write_landmark_csv_batch call) can convert px→µm
        # accurately even when auto-detect gives every image a different
        # scale. Pre-v0.2.27 landmark geojsons lack this and fall back to
        # the pipeline's global um_per_px, which produces wrong µm values
        # on auto-detect runs where per-image metadata varies.
        _fc_props: dict = {}
        if result.effective_um_per_px is not None and result.effective_um_per_px > 0:
            _fc_props["effective_um_per_px"] = float(result.effective_um_per_px)
        # Persist the rescale factor so the landmark inspector — which displays
        # these coordinates over the ORIGINAL image — can map them back from the
        # rescaled space they are stored in (Stage 1 rescales before landmark
        # detection) to original-input space for display. The pipeline itself
        # keeps working in rescaled space; only consumers that draw on the
        # original-resolution image inverse-scale by this factor.
        _fc_props["rescale_factor"] = float(result.rescale_factor or 1.0)
        save_landmarks_geojson(landmarks, lm_path, landmark_metadata, fc_props=_fc_props)
        result.landmarks_geojson_path = lm_path
        result.stages_completed.append("landmarks")

    # Stage 4: Hinge chop
    chopped_path = None
    if do_hinge:
        if landmarks is None:
            # Try to load from existing geojson
            lm_path = output_dir / f"{stem}_landmarks.geojson"
            if not lm_path.exists():
                lm_path = image_path.parent / f"{stem}_landmarks.geojson"
            if lm_path.exists():
                from hinge_chopper import load_landmarks

                landmarks = load_landmarks(str(lm_path))
            else:
                raise FileNotFoundError(
                    f"No landmarks available for hinge chopping {image_path.name}. "
                    f"Run landmarks stage first or provide a *_landmarks.geojson file."
                )

        if progress_callback:
            progress_callback("hinge", f"Chopping hinge for {image_path.name}")
        chopped_path = output_dir / f"{stem}_chopped{raster_ext}"
        run_hinge_chop(image_path, landmarks, chopped_path)
        result.chopped_image_path = chopped_path
        result.stages_completed.append("hinge")

    # Stage 5: Segmentation
    if do_segment:
        # Manual override: if the user corrected this image's vein/intervein
        # polygons via TRACE's inspector dialog, a sidecar lives in the image's
        # manual_overrides/ subfolder (legacy: loose next to the input), in
        # original-image pixel space (which is what seg_input is too — Stage 2/4
        # only mask pixels, never crop/translate). Trust it and skip the model.
        seg_override_path = find_segmentation_override(original_input_path)
        if seg_override_path is not None:
            import logging as _logging

            _logging.getLogger(__name__).info("%s: using manual segmentation override from %s", stem, seg_override_path)
            fc = load_segmentation_override(seg_override_path)
            # The override is in original-input pixel space; if Stage 1 rescaled
            # the image, bring it into the rescaled space the rest of the run uses.
            rf = result.rescale_factor or 1.0
            if rf != 1.0:
                fc = _scale_geojson_coords(fc, rf)
        else:
            # Use chopped image if available, otherwise original
            seg_input = chopped_path if chopped_path and chopped_path.exists() else image_path
            # Wing isolation (Stage 2) and hinge chop both modify pixel values in place —
            # neither crops nor translates — so the wing polygon coords are valid against
            # any combination of (isolated, chopped, neither). Always pass the ROI when
            # wing isolation produced a polygon so modelTOjson can skip background tiles.
            roi_geojson_path = result.wing_geojson_path
            if progress_callback:
                progress_callback("segmentation", f"Segmenting {image_path.name}")
            fc = run_segmentation(
                seg_input,
                segmentation_model_dir,
                device,
                model_cache,
                roi_geojson_path=roi_geojson_path,
            )
        seg_path = output_dir / f"{stem}.geojson"
        save_segmentation_geojson(fc, seg_path)
        result.segmentation_geojson_path = seg_path
        result.stages_completed.append("segmentation")

    # Stage 6: Rotation. Runs as the LAST preprocessing step so every model
    # inference (wing isolation, landmark detection, segmentation) sees the
    # original image. The image and every produced GeoJSON (landmarks, wing,
    # segmentation) get rotated to a canonical orientation in lockstep, so
    # downstream identifyFeatures consumes a self-consistent rotated set.
    # Skipped silently when there aren't enough reliable landmarks to fit;
    # everything stays in original orientation. include_unreliable_landmarks
    # doubles as soft_reliability: when True, gate-failed landmarks contribute
    # at reduced weight.
    if do_rotation and do_landmarks and landmarks:
        if progress_callback:
            progress_callback("rotation", f"Rotating {unmasked_image_path.name} to canonical orientation")
        extras: list[Path] = []
        if result.wing_geojson_path is not None:
            extras.append(result.wing_geojson_path)
        if result.segmentation_geojson_path is not None:
            extras.append(result.segmentation_geojson_path)
        # Rotate the un-masked image so Stage 2 overlays render on the full
        # background, not the wing-isolation mask. DL stages already ran on
        # the masked image; their GeoJSON outputs are in pre-rotation pixel
        # coords that match the un-masked image 1:1 (masking only zeros
        # pixels), so the same affine produces a consistent rotated set.
        rotated = run_rotation(
            image_path=unmasked_image_path,
            landmarks_geojson_path=lm_path,
            output_dir=output_dir,
            landmarks=landmarks,
            extra_geojsons=extras,
            soft_reliability=include_unreliable_landmarks,
            mirror_correct=rotation_mirror_correct,
        )
        if rotated is not None:
            rotated_image, rotated_lm, landmarks, rot_result = rotated
            result.landmarks = landmarks
            result.landmarks_geojson_path = rotated_lm
            result.rotated_image_path = rotated_image
            result.rotated_landmarks_geojson_path = rotated_lm
            result.rotation_angle_deg = rot_result.angle_deg
            result.rotation_rms_residual = rot_result.rms_residual
            result.rotation_n_landmarks = rot_result.n_landmarks_used
            result.rotation_mirror_detected = rot_result.mirrored_detected
            # Repoint extras to their rotated counterparts so downstream
            # identifyFeatures reads from a consistent rotated set.
            if result.wing_geojson_path is not None:
                rotated_wing = rot_result.extra_outputs.get(result.wing_geojson_path.name)
                if rotated_wing is not None:
                    result.wing_geojson_path = rotated_wing
            if result.segmentation_geojson_path is not None:
                rotated_seg = rot_result.extra_outputs.get(result.segmentation_geojson_path.name)
                if rotated_seg is not None:
                    result.segmentation_geojson_path = rotated_seg
            # Make the rotated image the canonical one for downstream consumers
            # (TRACE Stage 2 looks up `processed_image_path` in preproc_dir).
            result.processed_image_path = rotated_image
            result.stages_completed.append("rotation")

    # Clean up chopped temp file
    if chopped_path and chopped_path.exists() and not keep_chopped:
        chopped_path.unlink()
        result.chopped_image_path = None

    return result


def process_folder(
    input_dir: Path,
    output_dir: Path,
    landmark_checkpoint: Optional[Path] = None,
    segmentation_model_dir: Optional[Path] = None,
    stages: tuple[bool, bool, bool] = (True, True, True),
    device=None,
    keep_chopped: bool = False,
    progress_callback=None,
    include_unreliable_landmarks: bool = False,
    landmark_batch_size: Optional[int] = None,
    gate_override: Optional[dict] = None,
    max_workers: int = 1,
    wing_model_dir: Optional[Path] = None,
    wing_expand_fraction: float = 0.05,
    keep_intermediates: bool = False,
    recursive: bool = False,
    do_rotation: bool = True,
    rotation_mirror_correct: bool = False,
    input_um_per_px: Optional[float] = None,
    target_um_per_px: Optional[float] = None,
    rescale_tolerance_low: float = 0.85,
    rescale_tolerance_high: float = 1.15,
    auto_detect_um_per_px: bool = False,
    skip_image_basenames: Optional[set[str]] = None,
    pause_event: Optional[threading.Event] = None,
    predictor_cache: Optional[dict] = None,
    model_cache: Optional[dict] = None,
) -> list[PipelineResult]:
    """Process all images in a folder. Continues on per-image errors.

    Args:
        progress_callback: callable(image_index: int, total: int, image_name: str, status: str)
        landmark_batch_size:
          - None  → auto-pick via landmark_locator.auto_batch_size (single-image fallback).
          - 1     → run landmarks per-image (original behavior).
          - >1    → batch the landmark forward pass across this many images.
        max_workers: per-image parallelism for hinge / segmentation. The landmark
            stage stays batched (one forward pass) regardless. Each worker holds
            an image and a partial GeoJSON in memory; pick a value compatible
            with system RAM. Values <= 1 keep the original sequential loop.
        wing_model_dir: optional Stage 2 — when set, every image is masked through
            wingIsolator before landmarks/hinge/segmentation. Disables landmark
            batching (the prefetch is keyed by original paths and would be wrong
            for the masked images).
        wing_expand_fraction: Stage 2 buffer (fraction of sqrt(polygon area)).
        keep_intermediates: when True, the wing GeoJSON intermediate is also
            written alongside the masked image as <stem>_wing.geojson.
    """
    if device is None:
        device = _auto_device()

    images = discover_images(input_dir, recursive=recursive)
    if skip_image_basenames:
        # Resume support: drop images the host has already processed (or
        # which previously failed under identical settings). The basename
        # match is used because the manifest stores names without folder
        # parts. See TRACE/run_state.py for the rationale.
        images = [im for im in images if im.name not in skip_image_basenames]
    if not images:
        # All input images were filtered out — likely a resume where the
        # previous slice already finished everything. Return an empty
        # result list rather than raising, so callers can complete the
        # run cleanly (CSV merge, manifest finalize, etc.).
        if skip_image_basenames:
            return []
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    # When recursing, flatten each image's path-relative-to-input into a single
    # filename so that two images from different subfolders never overwrite each
    # other in output_dir. Top-level images keep their basename.
    target_names: dict[Path, Optional[str]] = {}
    if recursive:
        for img in images:
            target_names[img] = _subpath_target_name(img, input_dir)
    else:
        for img in images:
            target_names[img] = None

    output_dir.mkdir(parents=True, exist_ok=True)

    # Caller-supplied caches survive across process_folder calls; when
    # unset (CLI / standalone caller), fall back to function-local
    # dicts that die at return. The chunked run loop in
    # TRACE.pipeline._run passes cross-chunk caches so a 49-chunk
    # batch loads each model once instead of 49 times.
    if predictor_cache is None:
        predictor_cache = {}
    if model_cache is None:
        model_cache = {}
    results: list[PipelineResult] = []

    # If the landmark stage is enabled and we have a checkpoint, run landmarks in
    # batches up front so the model amortizes its forward-pass overhead across many
    # images. The per-image loop below picks results up via prefetched_landmarks.
    # Skipped when wing isolation is on — the prefetch is keyed by original paths
    # and would not match the rebound (masked) image_path inside process_single_image.
    # Also skipped when resolutionAdjust may rescale — for the same reason, the
    # rebound (rescaled) image_path won't match the prefetch key.
    _resolution_adjust_active = (
        input_um_per_px is not None and input_um_per_px > 0 and target_um_per_px is not None and target_um_per_px > 0
    )
    prefetched: Optional[dict] = None
    if stages[0] and landmark_checkpoint is not None and wing_model_dir is None and not _resolution_adjust_active:
        if progress_callback:
            progress_callback(0, len(images), "(batch)", "landmarks: predicting in batches")
        try:
            prefetched = predict_landmarks_for_paths(
                images,
                landmark_checkpoint,
                predictor_cache,
                include_unreliable_landmarks=include_unreliable_landmarks,
                batch_size=landmark_batch_size,
                confidence_override=gate_override,
                pause_event=pause_event,
            )
        except Exception as e:
            # Fail soft — fall back to per-image predict in process_single_image.
            if progress_callback:
                progress_callback(0, len(images), "(batch)", f"batch landmarks failed, falling back per-image: {e}")
            prefetched = None

    # Pre-warm the modelTOjson model cache so parallel workers don't race on first init
    # (the lock inside _load_or_cache_modeltojson handles correctness; pre-warming
    # avoids one worker doing the slow load while the others stall waiting on the lock).
    if stages[2] and segmentation_model_dir is not None:
        try:
            _load_or_cache_modeltojson(segmentation_model_dir, device, model_cache)
        except Exception:
            # Non-fatal — workers will lazy-load and just take longer on the first call.
            pass
    if wing_model_dir is not None:
        try:
            _load_or_cache_modeltojson(wing_model_dir, device, model_cache)
        except Exception:
            pass

    progress_lock = threading.Lock()

    def _emit(idx: int, name: str, msg: str):
        if not progress_callback:
            return
        with progress_lock:
            progress_callback(idx, len(images), name, msg)

    def _process_one(i: int, img_path: Path) -> PipelineResult:
        # Pause: bail out at image-boundaries so the user's click takes
        # effect within seconds. In-flight workers finish the image they
        # started — that's what keeps per-image artifacts consistent on
        # disk — but no new image work begins. The orchestrator
        # (TRACE/pipeline.py) filters paused entries out of the result
        # list so they aren't mistaken for failures.
        if pause_event is not None and pause_event.is_set():
            return PipelineResult(image_path=img_path, paused=True)
        _emit(i, img_path.name, "starting")
        try:
            result = process_single_image(
                image_path=img_path,
                output_dir=output_dir,
                landmark_checkpoint=landmark_checkpoint,
                segmentation_model_dir=segmentation_model_dir,
                stages=stages,
                predictor_cache=predictor_cache,
                model_cache=model_cache,
                device=device,
                keep_chopped=keep_chopped,
                progress_callback=lambda stage, detail: _emit(i, img_path.name, f"{stage}: {detail}"),
                include_unreliable_landmarks=include_unreliable_landmarks,
                prefetched_landmarks=prefetched,
                wing_model_dir=wing_model_dir,
                wing_expand_fraction=wing_expand_fraction,
                keep_intermediates=keep_intermediates,
                target_name=target_names.get(img_path),
                do_rotation=do_rotation,
                rotation_mirror_correct=rotation_mirror_correct,
                gate_override=gate_override,
                input_um_per_px=input_um_per_px,
                target_um_per_px=target_um_per_px,
                rescale_tolerance_low=rescale_tolerance_low,
                rescale_tolerance_high=rescale_tolerance_high,
                auto_detect_um_per_px=auto_detect_um_per_px,
            )
            _emit(i, img_path.name, "done")
            return result
        except Exception as e:
            from landmark_locator import LowConfidenceLandmarkError

            if isinstance(e, LowConfidenceLandmarkError):
                stage = "landmarks"
                err_msg = str(e)
            elif isinstance(e, RuntimeError) and str(e).startswith("wing isolation:"):
                stage = "wing_isolation"
                err_msg = str(e)
            else:
                stage = None
                err_msg = f"{e}\n{traceback.format_exc()}"
            _emit(i, img_path.name, f"error: {e}")
            return PipelineResult(image_path=img_path, error=err_msg, error_stage=stage)

    workers = max(1, int(max_workers))
    if workers <= 1:
        for i, img_path in enumerate(images):
            # Stop submitting new image work once the user has paused —
            # matches the ThreadPoolExecutor path where the per-task pause
            # check in _process_one short-circuits not-yet-started workers.
            if pause_event is not None and pause_event.is_set():
                break
            results.append(_process_one(i, img_path))
    else:
        # Pre-allocate so we can fill by original index → preserves input order.
        results = [None] * len(images)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="preproc") as executor:
            futures = {executor.submit(_process_one, i, p): i for i, p in enumerate(images)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()

    # Drop paused entries — images whose turn never came because the user
    # clicked Pause. Keeping them in the list would force the orchestrator
    # to special-case them everywhere; filtering once here means every
    # downstream consumer sees only "really attempted" results. The
    # sequential loop's early-break already omits them; this normalizes
    # the parallel branch (which fills paused slots from _process_one).
    results = [r for r in results if r is not None and not r.paused]

    return results
