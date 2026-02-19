"""Colored skeleton overlay for visual review."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from WingVeinAnalyzer.models.vein_labeler import VeinAssignment, VeinStatus

STATUS_COLORS: dict[VeinStatus, tuple[int, int, int]] = {
    VeinStatus.COMPLETE: (0, 255, 0),       # green
    VeinStatus.TRUNCATED: (0, 255, 255),     # yellow
    VeinStatus.FRAGMENTED: (0, 0, 255),      # red
    VeinStatus.ABSENT: (128, 128, 128),      # grey
}


def render_overlay(
    image: np.ndarray,
    skeleton: np.ndarray,
    assignments: list[VeinAssignment],
    output_path: Path,
) -> None:
    """Render a colored skeleton overlay on the original image and save it."""
    overlay = image.copy()
    cv2.imwrite(str(output_path), overlay)
