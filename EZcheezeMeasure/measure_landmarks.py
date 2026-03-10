"""
Measure distances between landmark points on Drosophila wings.

Reads GeoJSON landmark files, computes:
  - Wing length (L1-Rs to DTip)
  - CV distance (ACV.p to PCV.a)
  - CV/wing length ratio

Outputs a single CSV and annotated JPG overlays.
"""

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def load_landmarks(geojson_path: Path) -> dict[str, tuple[float, float]]:
    """Load landmark points from a GeoJSON file. Returns {name: (x, y)}."""
    with open(geojson_path) as f:
        data = json.load(f)
    landmarks = {}
    for feat in data["features"]:
        name = feat["properties"]["classification"]["name"]
        x, y = feat["geometry"]["coordinates"]
        landmarks[name] = (x, y)
    return landmarks


def euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def draw_overlay(jpg_path: Path, output_path: Path, landmarks: dict) -> None:
    """Draw measurement lines on the wing image and save."""
    img = cv2.imread(str(jpg_path))
    if img is None:
        print(f"  WARNING: could not read {jpg_path}, skipping overlay")
        return

    l1_rs = landmarks["L1-Rs"]
    dtip = landmarks["DTip"]
    acv_p = landmarks["ACV.p"]
    pcv_a = landmarks["PCV.a"]

    # Line thickness scales with image size
    thickness = max(2, img.shape[1] // 800)
    radius = thickness * 3
    font_scale = thickness * 0.5
    font_thickness = max(1, thickness)

    # Wing length line (cyan)
    cv2.line(
        img,
        (int(l1_rs[0]), int(l1_rs[1])),
        (int(dtip[0]), int(dtip[1])),
        color=(255, 255, 0),  # cyan in BGR
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # CV distance line (magenta)
    cv2.line(
        img,
        (int(acv_p[0]), int(acv_p[1])),
        (int(pcv_a[0]), int(pcv_a[1])),
        color=(255, 0, 255),  # magenta in BGR
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # Draw landmark dots and labels
    for name, pt in [
        ("L1-Rs", l1_rs),
        ("DTip", dtip),
        ("ACV.p", acv_p),
        ("PCV.a", pcv_a),
    ]:
        center = (int(pt[0]), int(pt[1]))
        cv2.circle(img, center, radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            img,
            name,
            (center[0] + radius + 2, center[1] - radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    # Legend in top-left
    legend_y = 40
    cv2.putText(
        img,
        "Wing length (L1-Rs to DTip)",
        (10, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 0),
        font_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "CV distance (ACV.p to PCV.a)",
        (10, legend_y + int(30 * font_scale * 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 0, 255),
        font_thickness,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output_path), img)


def main():
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "landmarkPoints"
    output_dir = input_dir.parent / "output"
    output_dir.mkdir(exist_ok=True)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    geojson_files = sorted(input_dir.glob("*_landmarks.geojson"))
    if not geojson_files:
        print(f"No *_landmarks.geojson files found in {input_dir}")
        sys.exit(1)

    rows = []
    for gj_path in geojson_files:
        image_id = gj_path.stem.replace("_landmarks", "")
        landmarks = load_landmarks(gj_path)

        missing = {"L1-Rs", "DTip", "ACV.p", "PCV.a"} - landmarks.keys()
        if missing:
            print(f"  WARNING: {image_id} missing landmarks {missing}, skipping")
            continue

        wing_length = euclidean(landmarks["L1-Rs"], landmarks["DTip"])
        cv_distance = euclidean(landmarks["ACV.p"], landmarks["PCV.a"])
        cv_wl_ratio = cv_distance / wing_length if wing_length > 0 else float("nan")

        rows.append(
            {
                "image_id": image_id,
                "wing_length_px": round(wing_length, 2),
                "cv_distance_px": round(cv_distance, 2),
                "cv_wl_ratio": round(cv_wl_ratio, 4),
            }
        )

        # Overlay
        jpg_path = gj_path.with_name(gj_path.name.replace(".geojson", ".jpg"))
        if jpg_path.exists():
            overlay_path = overlay_dir / f"{image_id}_measured.jpg"
            draw_overlay(jpg_path, overlay_path, landmarks)

        print(f"  {image_id}: WL={wing_length:.1f}  CV={cv_distance:.1f}  ratio={cv_wl_ratio:.4f}")

    # Write CSV
    csv_path = output_dir / "landmark_measurements.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "wing_length_px", "cv_distance_px", "cv_wl_ratio"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nProcessed {len(rows)} wings")
    print(f"CSV:      {csv_path}")
    print(f"Overlays: {overlay_dir}/")


if __name__ == "__main__":
    main()
