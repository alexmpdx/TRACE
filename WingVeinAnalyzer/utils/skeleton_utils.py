"""Skeletonize helpers, spur pruning, and flip detection."""

from __future__ import annotations

import numpy as np


def detect_and_correct_flip(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect and correct X-flip based on anterior vein density."""
    h, w = mask.shape[:2]
    midpoint = w // 2

    anterior_density = np.count_nonzero(mask[:, :midpoint])
    posterior_density = np.count_nonzero(mask[:, midpoint:])

    if posterior_density > anterior_density:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    return image, mask
