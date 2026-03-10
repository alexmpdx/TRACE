"""
End-to-end pipeline: detect landmarks with LandmarkLocator, then measure distances.

Usage:
    python run_pipeline.py <image_dir> <checkpoint_path>

Outputs to <image_dir>/../output/:
    overlays/          — measurement overlay JPGs (lines between landmarks)
    landmark_measurements.csv
"""

import csv
import sys
from pathlib import Path

import cv2
from landmark_locator import LandmarkPredictor
from measure_landmarks import draw_overlay, euclidean

# Internal model names → GeoJSON display names (matching EZcheezeMeasure format)
LANDMARK_TO_GEOJSON = {
    "acv_p": "ACV.p",
    "alula_notch": "alula notch",
    "dtip": "DTip",
    "l1_rs": "L1-Rs",
    "l4_l5": "L4-L5",
    "pcv_a": "PCV.a",
    "subcostal_break": "subcostal break",
}

IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <image_dir> <checkpoint_path>")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    checkpoint_path = Path(sys.argv[2])

    if not image_dir.is_dir():
        print(f"Error: {image_dir} is not a directory")
        sys.exit(1)
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found")
        sys.exit(1)

    # Collect input images
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    # Set up output directories
    output_dir = image_dir.parent / "output"
    output_dir.mkdir(exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    # Load model
    print(f"Loading model from {checkpoint_path.name}...")
    predictor = LandmarkPredictor(checkpoint_path)
    print(f"  Landmarks: {predictor.landmark_order}")
    print(f"  Device: {predictor.device}\n")

    print("=== Step 1: Landmark detection & measurement ===")
    rows = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  WARNING: could not read {img_path}, skipping")
            continue

        result = predictor.predict(image)
        stem = img_path.stem

        # Measure distances using GeoJSON names
        geojson_landmarks = {LANDMARK_TO_GEOJSON.get(k, k): v for k, v in result["landmarks"].items()}

        missing = {"L1-Rs", "DTip", "ACV.p", "PCV.a"} - geojson_landmarks.keys()
        if missing:
            print(f"  WARNING: {stem} missing {missing}, skipping measurement")
            continue

        wing_length = euclidean(geojson_landmarks["L1-Rs"], geojson_landmarks["DTip"])
        cv_distance = euclidean(geojson_landmarks["ACV.p"], geojson_landmarks["PCV.a"])
        cv_wl_ratio = cv_distance / wing_length if wing_length > 0 else float("nan")

        rows.append(
            {
                "image_id": stem,
                "wing_length_px": round(wing_length, 2),
                "cv_distance_px": round(cv_distance, 2),
                "cv_wl_ratio": round(cv_wl_ratio, 4),
            }
        )

        # Save measurement overlay JPG (lines on original image)
        measure_path = overlay_dir / f"{stem}_measured.jpg"
        draw_overlay(img_path, measure_path, geojson_landmarks)

        confs = result["confidences"]
        min_conf = min(confs.values())
        print(
            f"  {stem}: WL={wing_length:.1f}  CV={cv_distance:.1f}  "
            f"ratio={cv_wl_ratio:.4f}  min_conf={min_conf:.3f}"
        )

    # Write CSV
    print("\n=== Results ===")
    csv_path = output_dir / "landmark_measurements.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "wing_length_px", "cv_distance_px", "cv_wl_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Processed {len(rows)} wings")
    print(f"CSV:      {csv_path}")
    print(f"Overlays: {overlay_dir}/")


if __name__ == "__main__":
    main()
