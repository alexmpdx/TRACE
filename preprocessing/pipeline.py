"""
Preprocessing pipeline — orchestrates LandmarkLocator, HingeChopper, and modelTOjson.

Processes a folder of wing images through three stages:
  1. Landmark detection (LandmarkLocator)
  2. Hinge removal (HingeChopper)
  3. Segmentation to GeoJSON (modelTOjson)

Each stage can be run independently or as part of the full pipeline.
"""

import json
import shutil
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
}


def discover_images(folder: Path) -> list[Path]:
    """Find supported image files in a folder, skipping hidden/resource-fork files."""
    images = []
    for f in sorted(folder.iterdir()):
        if f.name.startswith(".") or f.name.startswith("._"):
            continue
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return images


# ---------------------------------------------------------------------------
# Stage 1: Landmark detection
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
) -> dict:
    """Predict landmarks for many images in batches.

    Returns {path: {"landmarks", "metadata", "error"}} where `error` is None on success
    or a `LowConfidenceLandmarkError` instance when a core landmark failed the gate.
    Caller decides whether to abort the image based on `error`.

    `batch_size`:
      - None → auto-pick via `landmark_locator.auto_batch_size`.
      - 1    → process one image at a time (matches single-fold predict() semantics).
      - >1   → batch model forward passes.
    """
    from landmark_locator import LowConfidenceLandmarkError, auto_batch_size
    from landmark_locator.data.psd_loader import imread_any

    if predictor_cache is None:
        predictor_cache = {}
    predictor = _predictor_from_cache(checkpoint_path, predictor_cache, confidence_override=confidence_override)

    out: dict = {}
    if not paths:
        return out

    bs = batch_size if batch_size and batch_size > 0 else auto_batch_size(len(paths))
    for chunk_start in range(0, len(paths), bs):
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
) -> tuple[dict, dict]:
    """Predict landmarks. Returns (landmarks, metadata) keyed by GeoJSON names.

    checkpoint_path may be a single `.pt` file (single-fold prediction) or a directory
    containing `best_fold*.pt` (5-fold ensemble — averaged heatmaps, more robust).

    metadata[name] -> {reliable, gate_reason, confidence, sharpness, second_peak_ratio}.

    Raises landmark_locator.LowConfidenceLandmarkError if a core landmark fails the gate.
    """
    predictor = _predictor_from_cache(checkpoint_path, predictor_cache)
    result = predictor.predict_from_path(image_path, include_unreliable=include_unreliable_landmarks)
    return _shape_predict_result(predictor, result)


def landmarks_to_geojson(landmarks: dict, metadata: Optional[dict] = None) -> dict:
    """Convert landmarks dict to GeoJSON FeatureCollection.

    metadata (if provided) maps name -> {reliable, gate_reason, confidence, sharpness,
    second_peak_ratio} and is embedded into each feature's properties.
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
    return {"type": "FeatureCollection", "features": features}


def save_landmarks_geojson(landmarks: dict, output_path: Path, metadata: Optional[dict] = None) -> None:
    """Save landmarks dict as GeoJSON file."""
    fc = landmarks_to_geojson(landmarks, metadata)
    with open(output_path, "w") as f:
        json.dump(fc, f, indent=2)


# ---------------------------------------------------------------------------
# Stage 2: Hinge chopping
# ---------------------------------------------------------------------------
def run_hinge_chop(image_path: Path, landmarks: dict, output_path: Path) -> Path:
    """Black out proximal hinge region. Returns path to chopped image."""
    from hinge_chopper import chop_hinge_from_landmarks

    chop_hinge_from_landmarks(str(image_path), landmarks, str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Stage 0: Wing isolation (optional)
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
    from modeltojson import mask_to_geojson, read_image, run_inference, save_geojson
    from shapely.geometry import MultiPolygon as _MultiPolygon
    from shapely.geometry import Polygon as _Polygon
    from shapely.geometry import shape as _shape
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
    stem = image_path.stem
    ext = image_path.suffix
    # cv2/Pillow can't write PSD; let write_masked_image fall back to PNG.
    raster_ext = ".tif" if ext.lower() in (".psd", ".psb") else ext
    out_image_path = Path(write_masked_image(masked, output_dir / f"{stem}_isolated{raster_ext}"))

    geojson_out_path = None
    if keep_intermediate_geojson:
        geojson_out_path = output_dir / f"{stem}_wing.geojson"
        save_geojson(fc, str(geojson_out_path))

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
# Stage 3: Segmentation
# ---------------------------------------------------------------------------
def run_segmentation(image_path: Path, model_dir: Path, device, model_cache: dict) -> dict:
    """Run segmentation inference. Returns GeoJSON FeatureCollection dict.

    model_cache is a mutable dict used to cache the loaded model across calls.
    """
    from modeltojson import mask_to_geojson, read_image, run_inference

    model, metadata = _load_or_cache_modeltojson(model_dir, device, model_cache)

    image = read_image(str(image_path))
    mask = run_inference(model, image, metadata, device)
    fc = mask_to_geojson(mask, metadata["classes"], str(image_path))

    # Remove "hinge junk" features — residual hinge tissue not useful downstream
    fc["features"] = [f for f in fc["features"] if f["properties"].get("class") != "hinge junk"]

    return fc


def save_segmentation_geojson(geojson_fc: dict, output_path: Path) -> None:
    """Save segmentation GeoJSON to file."""
    from modeltojson import save_geojson

    save_geojson(geojson_fc, str(output_path))


# ---------------------------------------------------------------------------
# Pipeline result and orchestration
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    image_path: Path
    landmarks: Optional[dict] = None
    landmark_metadata: Optional[dict] = None
    landmarks_geojson_path: Optional[Path] = None
    wing_isolated_image_path: Optional[Path] = None
    wing_geojson_path: Optional[Path] = None
    chopped_image_path: Optional[Path] = None
    segmentation_geojson_path: Optional[Path] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None
    stages_completed: list[str] = field(default_factory=list)


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
        wing_model_dir: optional Stage 0. When provided, run modelTOjson with this wing
            -identification model, feed the result into wingIsolator, and rebind
            image_path to the masked image so all downstream stages see a single wing.
            None disables the stage entirely.
        wing_expand_fraction: Stage 0 buffer (fraction of sqrt(polygon area)).
        keep_intermediates: when True, the wing GeoJSON intermediate is also written
            alongside the masked image as <stem>_wing.geojson.
    """
    do_landmarks, do_hinge, do_segment = stages
    result = PipelineResult(image_path=image_path)

    if predictor_cache is None:
        predictor_cache = {}
    if model_cache is None:
        model_cache = {}
    if device is None:
        device = _auto_device()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy original image to output
    dest_image = output_dir / image_path.name
    if not dest_image.exists() or dest_image != image_path:
        shutil.copy2(image_path, dest_image)

    # Coerce non-cv2-native and lossy formats to a lossless TIF up front so every
    # downstream stage (LandmarkLocator, hinge_chopper, modelTOjson) sees a path
    # cv2.imread can handle. HEIC/HEIF, RAW, and SVG are treated the same way as
    # JPEG: decoded once via psd_loader.imread_any, re-saved as TIF.
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
    if image_path.suffix.lower() in _COERCE_TO_TIF_EXTS:
        import cv2
        from preprocessing.psd_loader import imread_any

        img = imread_any(str(dest_image), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {dest_image}")
        tif_path = output_dir / f"{image_path.stem}.tif"
        if not cv2.imwrite(str(tif_path), img):
            raise IOError(f"Failed to write converted TIFF: {tif_path}")
        image_path = tif_path

    # Stage 0: Wing isolation (optional). When wing_model_dir is set, mask out
    # non-main-wing pixels and rebind image_path so all downstream stages see
    # the single-wing image. result.image_path remains the original input.
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

    stem = image_path.stem
    ext = image_path.suffix
    # cv2.imwrite cannot write .psd; coerce intermediate outputs to .tif when input is PSD.
    raster_ext = ".tif" if ext.lower() in (".psd", ".psb") else ext

    # Stage 1: Landmarks
    landmarks = None
    landmark_metadata: Optional[dict] = None
    if do_landmarks:
        if progress_callback:
            progress_callback("landmarks", f"Predicting landmarks for {image_path.name}")
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
            )
        result.landmarks = landmarks
        result.landmark_metadata = landmark_metadata
        lm_path = output_dir / f"{stem}_landmarks.geojson"
        save_landmarks_geojson(landmarks, lm_path, landmark_metadata)
        result.landmarks_geojson_path = lm_path
        result.stages_completed.append("landmarks")

    # Stage 2: Hinge chop
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

    # Stage 3: Segmentation
    if do_segment:
        # Use chopped image if available, otherwise original
        seg_input = chopped_path if chopped_path and chopped_path.exists() else image_path
        if progress_callback:
            progress_callback("segmentation", f"Segmenting {image_path.name}")
        fc = run_segmentation(seg_input, segmentation_model_dir, device, model_cache)
        seg_path = output_dir / f"{stem}.geojson"
        save_segmentation_geojson(fc, seg_path)
        result.segmentation_geojson_path = seg_path
        result.stages_completed.append("segmentation")

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
        wing_model_dir: optional Stage 0 — when set, every image is masked through
            wingIsolator before landmarks/hinge/segmentation. Disables landmark
            batching (the prefetch is keyed by original paths and would be wrong
            for the masked images).
        wing_expand_fraction: Stage 0 buffer (fraction of sqrt(polygon area)).
        keep_intermediates: when True, the wing GeoJSON intermediate is also
            written alongside the masked image as <stem>_wing.geojson.
    """
    if device is None:
        device = _auto_device()

    images = discover_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    predictor_cache: dict = {}
    model_cache: dict = {}
    results: list[PipelineResult] = []

    # If the landmark stage is enabled and we have a checkpoint, run landmarks in
    # batches up front so the model amortizes its forward-pass overhead across many
    # images. The per-image loop below picks results up via prefetched_landmarks.
    # Skipped when wing isolation is on — the prefetch is keyed by original paths
    # and would not match the rebound (masked) image_path inside process_single_image.
    prefetched: Optional[dict] = None
    if stages[0] and landmark_checkpoint is not None and wing_model_dir is None:
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
            results.append(_process_one(i, img_path))
    else:
        # Pre-allocate so we can fill by original index → preserves input order.
        results = [None] * len(images)  # type: ignore[list-item]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="preproc") as executor:
            futures = {executor.submit(_process_one, i, p): i for i, p in enumerate(images)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()

    return results
