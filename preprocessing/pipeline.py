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
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

IMAGE_EXTENSIONS = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg", ".psd", ".psb"}


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
    from landmark_locator import make_predictor

    cp_str = str(checkpoint_path)
    if "predictor" not in predictor_cache or predictor_cache.get("checkpoint") != cp_str:
        predictor_cache["predictor"] = make_predictor(checkpoint_path)
        predictor_cache["checkpoint"] = cp_str

    predictor = predictor_cache["predictor"]
    result = predictor.predict_from_path(image_path, include_unreliable=include_unreliable_landmarks)

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
# Stage 3: Segmentation
# ---------------------------------------------------------------------------
def run_segmentation(image_path: Path, model_dir: Path, device, model_cache: dict) -> dict:
    """Run segmentation inference. Returns GeoJSON FeatureCollection dict.

    model_cache is a mutable dict used to cache the loaded model across calls.
    """
    from modeltojson import load_model, mask_to_geojson, read_image, run_inference

    md_str = str(model_dir)
    if "model" not in model_cache or model_cache.get("model_dir") != md_str:
        model, metadata = load_model(md_str, device)
        model_cache["model"] = model
        model_cache["metadata"] = metadata
        model_cache["model_dir"] = md_str

    model = model_cache["model"]
    metadata = model_cache["metadata"]

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
) -> PipelineResult:
    """Run selected pipeline stages on a single image.

    Args:
        stages: (landmarks, hinge_chop, segmentation) booleans.
        progress_callback: callable(stage_name: str, detail: str)
        include_unreliable_landmarks: when True, landmarks that fail the LandmarkLocator
            confidence gate are still written to the output GeoJSON (marked reliable=false).
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

    # Convert JPEG inputs to lossless TIF and use that for the rest of the pipeline.
    if image_path.suffix.lower() in (".jpg", ".jpeg"):
        import cv2

        img = cv2.imread(str(dest_image), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read JPEG image: {dest_image}")
        tif_path = output_dir / f"{image_path.stem}.tif"
        if not cv2.imwrite(str(tif_path), img):
            raise IOError(f"Failed to write converted TIFF: {tif_path}")
        image_path = tif_path

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
) -> list[PipelineResult]:
    """Process all images in a folder. Continues on per-image errors.

    Args:
        progress_callback: callable(image_index: int, total: int, image_name: str, status: str)
    """
    if device is None:
        device = _auto_device()

    images = discover_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    predictor_cache = {}
    model_cache = {}
    results = []

    for i, img_path in enumerate(images):
        if progress_callback:
            progress_callback(i, len(images), img_path.name, "starting")
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
                progress_callback=lambda stage, detail: (
                    progress_callback(i, len(images), img_path.name, f"{stage}: {detail}")
                    if progress_callback
                    else None
                ),
                include_unreliable_landmarks=include_unreliable_landmarks,
            )
            results.append(result)
            if progress_callback:
                progress_callback(i, len(images), img_path.name, "done")
        except Exception as e:
            from landmark_locator import LowConfidenceLandmarkError

            stage = "landmarks" if isinstance(e, LowConfidenceLandmarkError) else None
            err_msg = str(e) if isinstance(e, LowConfidenceLandmarkError) else f"{e}\n{traceback.format_exc()}"
            result = PipelineResult(image_path=img_path, error=err_msg, error_stage=stage)
            results.append(result)
            if progress_callback:
                progress_callback(i, len(images), img_path.name, f"error: {e}")

    return results
