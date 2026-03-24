"""
Preprocessing pipeline — orchestrates LandmarkLocator, HingeChopper, modelTOjson, and add_wing.

Processes a folder of wing images through four stages:
  1. Landmark detection (LandmarkLocator)
  2. Hinge removal (HingeChopper)
  3. Segmentation to GeoJSON (modelTOjson)
  4. Wing annotation (add_wing — union of all polygons)

Each stage can be run independently or as part of the full pipeline.
"""

import json
import shutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

IMAGE_EXTENSIONS = {".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg"}


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
def run_landmarks(image_path: Path, checkpoint_path: Path, predictor_cache: dict) -> dict:
    """Predict landmarks on a single image. Returns dict of name -> (x, y) with GeoJSON names.

    predictor_cache is a mutable dict used to cache the LandmarkPredictor across calls.
    """
    from landmark_locator import LandmarkPredictor

    cp_str = str(checkpoint_path)
    if "predictor" not in predictor_cache or predictor_cache.get("checkpoint") != cp_str:
        predictor_cache["predictor"] = LandmarkPredictor(checkpoint_path)
        predictor_cache["checkpoint"] = cp_str

    predictor = predictor_cache["predictor"]
    result = predictor.predict_from_path(image_path)

    # Convert internal names to GeoJSON names
    landmark_to_geojson = {v: k for k, v in predictor.geojson_to_landmark.items()}
    landmarks = {}
    for internal_name, coords in result["landmarks"].items():
        geojson_name = landmark_to_geojson.get(internal_name, internal_name)
        landmarks[geojson_name] = coords
    return landmarks


def landmarks_to_geojson(landmarks: dict) -> dict:
    """Convert landmarks dict to GeoJSON FeatureCollection."""
    features = []
    for name, (x, y) in landmarks.items():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [x, y]},
                "properties": {"classification": {"name": name}},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def save_landmarks_geojson(landmarks: dict, output_path: Path) -> None:
    """Save landmarks dict as GeoJSON file."""
    fc = landmarks_to_geojson(landmarks)
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
# Stage 4: Wing annotation
# ---------------------------------------------------------------------------
def run_add_wing(geojson_path: Path) -> None:
    """Add a wing feature (union of all polygons) to an existing GeoJSON file."""
    from preprocessing.add_wing import add_wing

    add_wing(str(geojson_path))


# ---------------------------------------------------------------------------
# Pipeline result and orchestration
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    image_path: Path
    landmarks: Optional[dict] = None
    landmarks_geojson_path: Optional[Path] = None
    chopped_image_path: Optional[Path] = None
    segmentation_geojson_path: Optional[Path] = None
    error: Optional[str] = None
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
    stages: tuple[bool, bool, bool, bool] = (True, True, True, True),
    predictor_cache: Optional[dict] = None,
    model_cache: Optional[dict] = None,
    device=None,
    keep_chopped: bool = False,
    progress_callback=None,
) -> PipelineResult:
    """Run selected pipeline stages on a single image.

    Args:
        stages: (landmarks, hinge_chop, segmentation, add_wing) booleans.
        progress_callback: callable(stage_name: str, detail: str)
    """
    do_landmarks, do_hinge, do_segment, do_wing = stages
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

    stem = image_path.stem
    ext = image_path.suffix

    # Stage 1: Landmarks
    landmarks = None
    if do_landmarks:
        if progress_callback:
            progress_callback("landmarks", f"Predicting landmarks for {image_path.name}")
        landmarks = run_landmarks(image_path, landmark_checkpoint, predictor_cache)
        result.landmarks = landmarks
        lm_path = output_dir / f"{stem}_landmarks.geojson"
        save_landmarks_geojson(landmarks, lm_path)
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
        chopped_path = output_dir / f"{stem}_chopped{ext}"
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

    # Stage 4: Wing annotation
    if do_wing:
        seg_path = result.segmentation_geojson_path or output_dir / f"{stem}.geojson"
        if seg_path.exists():
            if progress_callback:
                progress_callback("add_wing", f"Adding wing annotation to {seg_path.name}")
            run_add_wing(seg_path)
            result.stages_completed.append("add_wing")
        else:
            raise FileNotFoundError(
                f"No segmentation GeoJSON found for wing annotation: {seg_path}. " f"Run segmentation stage first."
            )

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
    stages: tuple[bool, bool, bool, bool] = (True, True, True, True),
    device=None,
    keep_chopped: bool = False,
    progress_callback=None,
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
            )
            results.append(result)
            if progress_callback:
                progress_callback(i, len(images), img_path.name, "done")
        except Exception as e:
            result = PipelineResult(image_path=img_path, error=f"{e}\n{traceback.format_exc()}")
            results.append(result)
            if progress_callback:
                progress_callback(i, len(images), img_path.name, f"error: {e}")

    return results
