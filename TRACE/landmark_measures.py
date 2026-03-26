"""Landmark-based measurements for Drosophila wings.

Computes:
  - Wing length (L1-Rs to DTip)
  - CV distance (ACV.p to PCV.a)
  - CV/wing length ratio

Draws an annotated overlay showing landmark points and measurement lines.

Adapted from EZcheezeMeasure/measure_landmarks.py.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

REQUIRED_LANDMARKS = {"L1-Rs", "DTip", "ACV.p", "PCV.a"}


@dataclass
class LandmarkMeasurements:
    wing_length_px: float
    cv_distance_px: float
    cv_wl_ratio: float


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


def compute_landmark_measurements(landmarks: dict[str, tuple[float, float]]) -> Optional[LandmarkMeasurements]:
    """Compute wing length, CV distance, and ratio from landmark points.

    Returns None if required landmarks are missing.
    """
    missing = REQUIRED_LANDMARKS - landmarks.keys()
    if missing:
        return None

    wing_length = euclidean(landmarks["L1-Rs"], landmarks["DTip"])
    cv_distance = euclidean(landmarks["ACV.p"], landmarks["PCV.a"])
    cv_wl_ratio = cv_distance / wing_length if wing_length > 0 else float("nan")

    return LandmarkMeasurements(
        wing_length_px=round(wing_length, 2),
        cv_distance_px=round(cv_distance, 2),
        cv_wl_ratio=round(cv_wl_ratio, 4),
    )


def draw_landmark_overlay(image_path: Path, output_path: Path, landmarks: dict[str, tuple[float, float]]) -> bool:
    """Draw measurement lines and landmark points on the wing image.

    Returns True on success, False if image could not be read or landmarks missing.
    """
    missing = REQUIRED_LANDMARKS - landmarks.keys()
    if missing:
        return False

    img = cv2.imread(str(image_path))
    if img is None:
        return False

    l1_rs = landmarks["L1-Rs"]
    dtip = landmarks["DTip"]
    acv_p = landmarks["ACV.p"]
    pcv_a = landmarks["PCV.a"]

    # Scale rendering to image size
    thickness = max(2, img.shape[1] // 800)
    radius = thickness * 3
    font_scale = thickness * 0.5
    font_thickness = max(1, thickness)

    # Wing length line (cyan)
    cv2.line(
        img,
        (int(l1_rs[0]), int(l1_rs[1])),
        (int(dtip[0]), int(dtip[1])),
        color=(255, 255, 0),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # CV distance line (magenta)
    cv2.line(
        img,
        (int(acv_p[0]), int(acv_p[1])),
        (int(pcv_a[0]), int(pcv_a[1])),
        color=(255, 0, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )

    # Landmark dots and labels
    for name, pt in [("L1-Rs", l1_rs), ("DTip", dtip), ("ACV.p", acv_p), ("PCV.a", pcv_a)]:
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

    # Legend
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
    return True
