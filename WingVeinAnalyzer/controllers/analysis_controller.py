"""Orchestrates the full analysis pipeline per image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from WingVeinAnalyzer.models.vein_graph import build_graph, skeletonize_mask
from WingVeinAnalyzer.models.vein_labeler import VeinAssignment, assign_veins
from WingVeinAnalyzer.utils.skeleton_utils import detect_and_correct_flip


def run_pipeline(
    image_path: Path,
    mask_path: Path,
    microns_per_pixel: float | None = None,
    spur_threshold: int = 10,
) -> list[VeinAssignment]:
    """Run the full vein analysis pipeline on a single image."""
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    image, mask = detect_and_correct_flip(image, mask)
    skeleton = skeletonize_mask(mask, spur_threshold=spur_threshold)
    skan_skeleton, graph = build_graph(skeleton, mask=mask)
    assignments = assign_veins(skan_skeleton, graph)

    return assignments
